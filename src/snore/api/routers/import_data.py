from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid

from collections.abc import AsyncGenerator, Awaitable, Callable, MutableMapping
from datetime import date
from pathlib import Path
from typing import IO, Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import UploadFile as StarletteUploadFile

from snore.api.deps import ActorDep, get_db
from snore.api.guards import RequireAuth, RequireWritable
from snore.api.import_jobs import (
    TERMINAL_STATES,
    ForcedRescanActiveError,
    ImportJob,
    JobState,
    JobType,
    cancel_job,
    create_rescan_job,
    enqueue_for_execution,
    get_job,
    list_jobs,
    remove_job,
    reserve_slot,
)
from snore.api.jobs import (
    cancel_or_409,
    merge_job_lists,
    owned_or_404,
    terminal_records_query,
)
from snore.api.schemas import (
    HealthImportResultSummary,
    ImportResultSummary,
    ImportSourceResultSummary,
    LinkedAnalysisSummary,
    PipelineJobsListResponse,
    PipelineJobStatus,
)
from snore.auth.actor import ActorContext
from snore.database import models
from snore.services.import_service import normalize_datalog_suffix, safe_relative_path

logger = logging.getLogger(__name__)


class _ByteCeilingReceive:
    """ASGI receive wrapper that raises HTTPException(413) on byte-ceiling breach.

    Used for the upload ingress ceiling.  The auth-body ceiling uses a
    pre-read buffer in AuthPathMiddleware instead, which avoids the Starlette
    body-parser translating the 413 to 400.
    """

    def __init__(
        self,
        inner: Callable[[], Awaitable[MutableMapping[str, Any]]],
        max_bytes: int,
        detail: str | None = None,
    ) -> None:
        self._inner = inner
        self._max = max_bytes
        self._detail = (
            detail or f"Request body exceeds the {max_bytes // 1024} KiB limit"
        )
        self._seen = 0

    async def __call__(self) -> MutableMapping[str, Any]:
        from fastapi import HTTPException  # noqa: PLC0415

        msg = await self._inner()
        if msg.get("type") == "http.request":
            chunk: bytes = msg.get("body", b"")
            self._seen += len(chunk)
            if self._seen > self._max:
                raise HTTPException(status_code=413, detail=self._detail)
        return msg


router = APIRouter()

# Kept as an empty router for app.py backward compatibility (conditionally included in non-multiuser mode).
local_only_router = APIRouter()

# Chunk size for the off-event-loop file copy.
_COPY_CHUNK = 65536  # 64 KiB


class _FileSizeExceeded(Exception):
    """Raised inside the thread worker when per-file byte limit is exceeded."""


def _copy_chunked(src_file: IO[bytes], dest: Path, max_bytes: int) -> None:
    """Synchronous chunk copy; raises ``_FileSizeExceeded`` on over-limit.

    Runs inside ``asyncio.to_thread`` so it never blocks the event loop.
    ``src_file`` must expose a ``read(n)`` method (SpooledTemporaryFile).
    """
    total = 0
    try:
        with dest.open("wb") as dst:
            while True:
                chunk = src_file.read(_COPY_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise _FileSizeExceeded()
                dst.write(chunk)
    except _FileSizeExceeded:
        dest.unlink(missing_ok=True)
        raise


class JobResponse(BaseModel):
    job_id: str


class PrecheckFileEntry(BaseModel):
    path: str = Field(max_length=1024)
    size: int = Field(ge=0)


class PrecheckRequest(BaseModel):
    files: list[PrecheckFileEntry] = Field(max_length=50_000)
    profile_id: int | None = None


class PrecheckResponse(BaseModel):
    skippable: list[str]


class RescanRequest(BaseModel):
    profile_id: int | None = None
    force: bool = Field(
        default=False,
        description=(
            "Delete and recreate every matched session that already exists, "
            "including its analysis and breath rows. Disabled sessions stay disabled."
        ),
    )
    from_date: date | None = Field(
        default=None,
        description=(
            "Only re-import sessions that start on or after this date. Dates match "
            "each session's start date, not its night: a night's sessions that "
            "start after midnight fall on the next day."
        ),
    )
    to_date: date | None = Field(
        default=None,
        description=(
            "Only re-import sessions that start on or before this date. To cover a "
            "night, include the following day: sessions that start after midnight "
            "are dated the next day."
        ),
    )
    analyze: bool = Field(
        default=True, description="Queue analysis for the imported sessions"
    )

    @model_validator(mode="after")
    def _check_date_order(self) -> RescanRequest:
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date must be on or before to_date")
        return self


# Anchor files must never be reported as skippable. STR.edf changes daily and
# drives parser detection and CPAP settings history; the server check is
# authoritative — the client also omits anchors from precheck requests and never
# drops them from uploads, but must not be relied on.
_ANCHOR_NAMES = frozenset({"str.edf", "identification.json", "identification.tgt"})

# Bound concurrent backup filesystem walks to prevent thread-pool exhaustion.
# Each precheck spawns a to_thread walk; N concurrent prechecks could starve
# other to_thread users (e.g. active upload copies) on the default executor.
_PRECHECK_WALK_SEM = asyncio.Semaphore(4)


def _build_backup_index(
    profile_raw_root: Path,
) -> dict[str, set[tuple[str, int]]]:
    """Return per-serial (datalog_suffix, size) pairs for every file under the profile backup.

    Keyed by serial dir name. An SD card belongs to one device, so per-serial
    indexing prevents cross-device collisions: two same-model devices can produce
    identically named, identically sized files, and merging them into one set
    would silently mark device B's unuploaded file as already present via device
    A's backup — silent data loss.

    Walks each <serial>/DATALOG/ subtree synchronously (intended for
    asyncio.to_thread). STR_Backup/ is outside DATALOG/ and is naturally
    excluded. Any OSError in an individual serial branch is caught, logged, and
    skipped so a partially-readable backup never prevents a precheck response
    (fail-open).
    """
    if not profile_raw_root.is_dir():
        return {}
    try:
        serial_dirs = list(profile_raw_root.iterdir())
    except OSError:
        logger.warning("Cannot list backup root %s", profile_raw_root)
        return {}
    indexes: dict[str, set[tuple[str, int]]] = {}
    for serial_dir in serial_dirs:
        try:
            if not serial_dir.is_dir():
                continue
            datalog_dir = serial_dir / "DATALOG"
            if not datalog_dir.is_dir():
                continue
            serial_index: set[tuple[str, int]] = set()
            for dirpath, _, filenames in os.walk(datalog_dir):
                for fname in filenames:
                    fpath = Path(dirpath) / fname
                    try:
                        rel = str(fpath.relative_to(serial_dir))
                        suffix = normalize_datalog_suffix(rel)
                        if suffix is not None:
                            serial_index.add((suffix, fpath.stat().st_size))
                    except Exception:  # noqa: BLE001
                        logger.warning("Skipping backup file %s", fpath)
            if serial_index:
                indexes[serial_dir.name] = serial_index
        except OSError:
            logger.warning("Error reading serial dir %s", serial_dir)
    return indexes


def _classify_files(
    indexes: dict[str, set[tuple[str, int]]],
    files: list[PrecheckFileEntry],
) -> list[str]:
    """Return original path strings for files safe to skip.

    Classifies files per serial backup and returns results for the dominant
    serial — the one with the most matches. An SD card belongs to one device,
    so its files should overwhelmingly match one serial dir; cross-serial
    matches are treated as coincidence and never justify skipping.

    A file is skippable iff its DATALOG-relative suffix (lowercased) and exact
    size appear in that serial's backup index. Anchor files and non-DATALOG
    paths are excluded unconditionally. Ties between serials with equal match
    counts are broken alphabetically by serial name for determinism.
    """
    if not indexes:
        return []
    best: list[str] = []
    for serial in sorted(indexes):
        index = indexes[serial]
        skippable: list[str] = []
        for f in files:
            normalized_name = f.path.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if normalized_name in _ANCHOR_NAMES:
                continue
            suffix = normalize_datalog_suffix(f.path)
            if suffix is None:
                continue
            if (suffix, f.size) in index:
                skippable.append(f.path)
        if len(skippable) > len(best):
            best = skippable
    return best


def _get_upload_limits() -> tuple[int, int, int]:
    """Return (max_upload_bytes, max_upload_files, max_file_bytes) from config.

    Falls back to the config.py env-var defaults (SNORE_MAX_UPLOAD_BYTES=2 GiB,
    SNORE_MAX_UPLOAD_FILES=10 000, SNORE_MAX_FILE_BYTES=256 MiB) when config is
    not yet initialised — e.g. unit tests that exercise routes without a lifespan.
    """
    try:
        from snore.api.config import get_config  # noqa: PLC0415

        cfg = get_config()
        return cfg.max_upload_bytes, cfg.max_upload_files, cfg.max_file_bytes
    except Exception:
        return 2 * 1024 * 1024 * 1024, 10_000, 256 * 1024 * 1024


async def _resolve_profile_id(
    db: AsyncSession,
    actor: ActorContext,
    requested_id: int | None,
) -> int:
    """Return the resolved profile ID, validating ownership when *requested_id* is given.

    Raises HTTP 403 when the requested profile does not belong to the actor or
    is being deleted.  Falls back to the actor's active profile when *requested_id*
    is None.
    """
    if requested_id is not None:
        owned = (
            (
                await db.execute(
                    select(models.Profile).where(
                        models.Profile.id == requested_id,
                        models.Profile.user_id == actor.user_id,
                        models.Profile.deleting_at.is_(None),
                    )
                )
            )
            .scalars()
            .first()
        )
        if owned is None:
            raise HTTPException(status_code=403, detail="Profile not owned by user")
        return requested_id
    return actor.profile_id


@router.post(
    "/",
    response_model=JobResponse,
    status_code=202,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["files"],
                        "properties": {
                            "files": {
                                "type": "array",
                                "items": {"type": "string", "format": "binary"},
                            },
                            "profile_id": {
                                "type": "integer",
                                "description": "Target profile ID (defaults to actor's active profile)",
                            },
                            "import_type": {
                                "type": "string",
                                "enum": ["cpap", "health"],
                                "description": (
                                    "Type of import: 'cpap' for CPAP device uploads (default), "
                                    "'health' for Apple Health export.zip"
                                ),
                            },
                        },
                    }
                }
            },
        }
    },
)
async def import_files(
    request: Request,
    actor: RequireWritable,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobResponse:
    max_upload_bytes, max_upload_files, max_file_bytes = _get_upload_limits()

    # Wrap the ASGI receive callable with the ingress byte ceiling.
    # This raises 413 as soon as the cumulative chunk stream exceeds the
    # limit — before Starlette's multipart parser spools to temp files.
    ceiling_receive = _ByteCeilingReceive(
        request.receive,
        max_upload_bytes,
        detail=f"Upload exceeds the {max_upload_bytes // (1024**2)} MiB per-upload limit",
    )
    request._receive = ceiling_receive  # noqa: SLF001

    # _job_cleanup: True until the job is handed to the worker.  The finally
    # block uses this flag to decide whether to clean up the job.
    _job_cleanup = True
    # _is_continuation: True when appending to an existing batch job.
    # Cleanup must NOT release capacity or remove the job on error — the
    # original reservation still owns that.
    _is_continuation = False
    tmp: str | None = None
    _requested_profile_id: int | None = None
    job: ImportJob | None = None

    try:
        async with request.form(max_files=max_upload_files) as form:
            # ── batch fields ──────────────────────────────────────────
            _batch_id_raw = form.get("batch_id")
            batch_id: str | None = (
                str(_batch_id_raw) if isinstance(_batch_id_raw, str) else None
            )
            _batch_final_raw = form.get("batch_final")
            batch_final: bool = (
                str(_batch_final_raw).lower() != "false"
                if isinstance(_batch_final_raw, str)
                else True
            )

            _profile_id_raw = form.get("profile_id")
            if isinstance(_profile_id_raw, str):
                try:
                    _requested_profile_id = int(_profile_id_raw)
                except ValueError:
                    raise HTTPException(
                        status_code=422, detail="profile_id must be an integer"
                    ) from None

            # ── import type ───────────────────────────────────────────
            _import_type_raw = form.get("import_type")
            import_type = (
                str(_import_type_raw) if isinstance(_import_type_raw, str) else "cpap"
            )
            if import_type not in ("cpap", "health"):
                raise HTTPException(
                    status_code=422,
                    detail="import_type must be 'cpap' or 'health'",
                )

            # Health imports are single-request only — batch continuation is not supported.
            if import_type == "health":
                if batch_id is not None:
                    raise HTTPException(
                        status_code=422,
                        detail="Batch uploads are not supported for health imports",
                    )
                if (
                    isinstance(_batch_final_raw, str)
                    and str(_batch_final_raw).lower() == "false"
                ):
                    raise HTTPException(
                        status_code=422,
                        detail="Batch uploads are not supported for health imports",
                    )

            # ── job acquisition ───────────────────────────────────────
            if import_type == "health":
                # Health uploads reserve a dedicated slot with the HEALTH_UPLOAD type.
                job = reserve_slot(actor.user_id, job_type=JobType.HEALTH_UPLOAD)
                if job is None:
                    raise HTTPException(
                        status_code=429,
                        detail="Too many active imports. Please wait for existing imports to complete.",
                    )
            elif batch_id is not None:
                # Continuation of an existing batch upload.
                job = get_job(batch_id)
                if job is None:
                    raise HTTPException(status_code=404, detail="Batch not found")
                if job.owner_user_id != actor.user_id:
                    raise HTTPException(status_code=404, detail="Batch not found")
                if job.state != JobState.PENDING_UPLOAD:
                    raise HTTPException(
                        status_code=409,
                        detail="Batch already committed",
                    )
                _is_continuation = True
                # Update activity timestamp so the stale-upload reaper doesn't
                # reclaim this job while a slow continuation chunk is in flight.
                job.touch()
            else:
                # New CPAP upload — reserve an admission slot.
                job = reserve_slot(actor.user_id)
                if job is None:
                    raise HTTPException(
                        status_code=429,
                        detail="Too many active imports. Please wait for existing imports to complete.",
                    )

            uploads = [
                f for f in form.getlist("files") if isinstance(f, StarletteUploadFile)
            ]
            if not uploads:
                raise HTTPException(status_code=422, detail="No files provided")

            if import_type == "health":
                # Health imports require exactly one file; the per-upload ceiling
                # applies (real exports exceed the per-file CPAP cap of 256 MiB).
                if len(uploads) != 1:
                    raise HTTPException(
                        status_code=422,
                        detail="Health imports require exactly one file",
                    )
                if (uploads[0].size or 0) > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the "
                            f"{max_upload_bytes // (1024**2)} MiB per-upload limit"
                        ),
                    )
            else:
                # Per-file limit: reject any individual file exceeding the cap.
                oversized = [f for f in uploads if (f.size or 0) > max_file_bytes]
                if oversized:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"One or more files exceed the "
                            f"{max_file_bytes // (1024**2)} MiB per-file limit"
                        ),
                    )
                # Defense-in-depth: also check post-spool total size (catches absent/lying CL).
                total_size = sum(f.size or 0 for f in uploads)
                if total_size > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Total upload size exceeds {max_upload_bytes // (1024**2)} MiB limit",
                    )

            # ── staging directory ─────────────────────────────────────
            if _is_continuation:
                assert job.temp_dir is not None
                tmp_path = job.temp_dir
            else:
                from snore.api.config import get_config  # noqa: PLC0415

                spool_base = get_config().upload_spool_dir
                spool_base.mkdir(parents=True, exist_ok=True)
                tmp_path = spool_base / uuid.uuid4().hex
                tmp_path.mkdir()
                job.temp_dir = tmp_path

            tmp_root = tmp_path.resolve()
            # Health imports use the full per-upload ceiling; CPAP uses the per-file cap.
            per_file_limit = (
                max_upload_bytes if import_type == "health" else max_file_bytes
            )
            per_file_limit_mb = per_file_limit // (1024**2)
            per_file_limit_label = (
                "per-upload" if import_type == "health" else "per-file"
            )
            for upload in uploads:
                filename = upload.filename or "unknown"
                rel = safe_relative_path(filename) or "unknown"
                dest = tmp_path / rel
                if not dest.resolve().is_relative_to(tmp_root):
                    logger.warning("Skipping file with unsafe path: %r", filename)
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                # Run copy in a thread so the event loop stays responsive.
                # asyncio.create_task wraps the coroutine in a task that can be
                # individually awaited; asyncio.shield lets us wait for it even
                # under task cancellation so the copy thread runs to completion
                # before we clean up.
                copy_task = asyncio.create_task(
                    asyncio.to_thread(_copy_chunked, upload.file, dest, per_file_limit)
                )
                try:
                    await asyncio.shield(copy_task)
                except asyncio.CancelledError:
                    # Request cancelled.  Loop the shielded wait until the copy
                    # task is truly done so cleanup never races a running write.
                    # The loop is robust to repeated cancellations: each inner
                    # CancelledError is absorbed and the loop re-tests .done().
                    while not copy_task.done():
                        try:
                            await asyncio.shield(copy_task)
                        except (asyncio.CancelledError, Exception):
                            pass
                    raise  # propagates to finally → cleanup
                except _FileSizeExceeded:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File '{filename}' exceeds the "
                            f"{per_file_limit_mb} MiB {per_file_limit_label} limit"
                        ),
                    ) from None
                await upload.close()
                # Refresh activity so the stale-upload reaper sees a live upload.
                job.touch()

        if not batch_final:
            # More chunks coming — keep the job in PENDING_UPLOAD.
            _job_cleanup = False
            return JobResponse(job_id=job.job_id)

        # ── final chunk: resolve profile, enqueue ─────────────────
        resolved_profile_id = await _resolve_profile_id(
            db, actor, _requested_profile_id
        )

        job.set_file_count(sum(1 for _ in tmp_path.rglob("*") if _.is_file()))
        job.convert_to_pending()
        job.target_profile_id = resolved_profile_id
        _job_cleanup = False  # Worker owns the job from here.

    finally:
        # Runs on every exit: normal (no-op), HTTPException, Exception,
        # CancelledError.  _job_cleanup is False only when the worker started
        # or when a non-final batch chunk succeeded.
        if _job_cleanup:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
            if not _is_continuation and job is not None:
                job.try_cancel()
                remove_job(job.job_id)
                job.cleanup_files()
                job.release_capacity()

    # Derive profile-scoped backup root from the resolved target profile.
    from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

    profile_raw_root = DEFAULT_RAW_BACKUP_DIR / str(job.target_profile_id)

    # Persist PENDING state BEFORE enqueuing so startup recovery always sees the
    # row, even if the server crashes between the write and the worker picking it
    # up.  Writing first also prevents the worker's RUNNING upsert from racing
    # ahead of this PENDING write and leaving the DB in a stale state.
    try:
        from snore.api.import_worker import _upsert_job_record  # noqa: PLC0415

        await _upsert_job_record(job)
    except Exception:
        logger.exception("Failed to persist PENDING state for job %s", job.job_id)
    # Enqueue for serial execution — /progress is observer-only.
    enqueue_for_execution(job, profile_raw_root)
    return JobResponse(job_id=job.job_id)


@router.post("/precheck", response_model=PrecheckResponse)
async def precheck_upload(
    body: PrecheckRequest,
    actor: RequireAuth,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PrecheckResponse:
    """Return which client files are already present in the server backup.

    Fail-open: a missing or empty backup returns nothing skippable so the client
    uploads everything. Anchor files (STR.edf, Identification.json,
    Identification.tgt) are never skippable — STR.edf changes daily and drives
    parser detection and CPAP settings history; the server check is authoritative.
    The filesystem walk runs off the event loop via asyncio.to_thread; concurrent
    walks are bounded by _PRECHECK_WALK_SEM (fail-open on timeout).
    """
    profile_id = await _resolve_profile_id(db, actor, body.profile_id)

    from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

    profile_raw_root = DEFAULT_RAW_BACKUP_DIR / str(profile_id)

    try:
        async with asyncio.timeout(2):
            await _PRECHECK_WALK_SEM.acquire()
    except TimeoutError:
        return PrecheckResponse(skippable=[])

    try:
        indexes = await asyncio.to_thread(_build_backup_index, profile_raw_root)
    finally:
        _PRECHECK_WALK_SEM.release()

    if not indexes:
        return PrecheckResponse(skippable=[])

    return PrecheckResponse(skippable=_classify_files(indexes, body.files))


@router.post("/rescan", response_model=JobResponse, status_code=202)
async def rescan_archive(
    body: RescanRequest,
    actor: RequireWritable,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JobResponse:
    """Re-import CPAP sessions from the server-side raw archive.

    Creates an import job that reads directly from the profile's backup archive
    (~/.snore/raw/<profile_id>/<serial>/DATALOG/) without requiring a new file
    upload.  Useful when DB sessions were deleted but the archive is intact —
    the UNIQUE(device_id, device_session_id) upsert-skip makes this idempotent.

    With ``force``, sessions that already exist are deleted and recreated
    instead of skipped; the date range limits which sessions are parsed and
    ``analyze=false`` skips the follow-up analysis job.

    Returns 422 when no archive exists or the archive contains no device data.
    Returns 409 when ``force`` is set and a forced rescan of the same profile is
    already queued or running.
    Returns 429 when admission caps are exceeded.
    """
    from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

    resolved_profile_id = await _resolve_profile_id(db, actor, body.profile_id)
    profile_raw_root = DEFAULT_RAW_BACKUP_DIR / str(resolved_profile_id)

    if not profile_raw_root.is_dir():
        raise HTTPException(
            status_code=422,
            detail="No archive found for this profile. Upload data first.",
        )

    # Require at least one serial/DATALOG subtree — bare profile dirs with no
    # device data cannot produce importable sessions.
    has_device_data = any(
        (d / "DATALOG").is_dir() for d in profile_raw_root.iterdir() if d.is_dir()
    )
    if not has_device_data:
        raise HTTPException(
            status_code=422,
            detail="Archive exists but contains no device data.",
        )

    try:
        job = create_rescan_job(
            owner_user_id=actor.user_id,
            target_profile_id=resolved_profile_id,
            force=body.force,
            date_from=body.from_date,
            date_to=body.to_date,
            analyze=body.analyze,
        )
    except ForcedRescanActiveError:
        raise HTTPException(
            status_code=409,
            detail=(
                "A forced re-import of this profile is already queued or running. "
                "Wait for it to finish and try again."
            ),
        ) from None
    except RuntimeError:
        raise HTTPException(
            status_code=429,
            detail="Too many active imports. Please wait for existing imports to complete.",
        ) from None

    try:
        from snore.api.import_worker import _upsert_job_record  # noqa: PLC0415

        await _upsert_job_record(job)
    except Exception:
        logger.exception(
            "Failed to persist PENDING state for rescan job %s", job.job_id
        )

    enqueue_for_execution(job, profile_raw_root)
    return JobResponse(job_id=job.job_id)


def _derive_stage(
    import_state: JobState,
    analysis_job_id: str | None,
    analysis_queued: bool | None,
    linked: LinkedAnalysisSummary | None,
) -> str:
    """Map import + analysis state to a human-readable pipeline stage string.

    ``linked`` is a LinkedAnalysisSummary instance or None (reaped / not yet set).
    """
    if import_state == JobState.PENDING_UPLOAD:
        return "uploading"
    if import_state == JobState.PENDING:
        return "queued"
    if import_state == JobState.RUNNING:
        return "importing"
    if import_state == JobState.FAILED:
        return "failed"
    if import_state == JobState.CANCELLED:
        return "cancelled"
    # SUCCEEDED — determine analysis stage.
    if analysis_job_id is None and analysis_queued is False:
        return "analysis_skipped"
    if analysis_job_id is None:
        # health_upload jobs never enqueue analysis; they reach here with
        # analysis_queued=None (not False) and return "done" directly.
        return "done"
    if linked is None:
        # Analysis job was reaped after the import job stored the link.
        return "done"
    _state_map = {
        "queued": "analysis_queued",
        "running": "analyzing",
        "succeeded": "done",
        "failed": "analysis_failed",
        "cancelled": "analysis_cancelled",
    }
    return _state_map.get(linked.state, "unknown")


def _to_import_result_summary(result_dict: dict[str, Any]) -> ImportResultSummary:
    """Build an ImportResultSummary from a raw import-result dict, stripping session IDs."""
    sources = [
        ImportSourceResultSummary(
            source=s.get("source", {}),
            imported=s.get("imported", 0),
            skipped=s.get("skipped", 0),
            failed=s.get("failed", 0),
            warnings=s.get("warnings", []),
        )
        for s in result_dict.get("sources", [])
    ]
    return ImportResultSummary(
        total_imported=result_dict.get("total_imported", 0),
        total_skipped=result_dict.get("total_skipped", 0),
        total_failed=result_dict.get("total_failed", 0),
        warnings=result_dict.get("warnings", []),
        sources=sources,
    )


@router.get("/jobs", response_model=PipelineJobsListResponse)
async def list_pipeline_jobs(
    actor: RequireAuth,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PipelineJobsListResponse:
    """List all pipeline (import) jobs visible to the authenticated actor.

    Merges in-memory active/recent jobs with persisted historical records from
    the database.  In-memory jobs take precedence when both exist for the same
    job_id (deduplication).

    Ownership: jobs with owner_user_id=None are visible to any authenticated
    user (local-mode parity); jobs with a set owner are visible only to that owner.
    """
    from snore.api import analysis_jobs as aj_module  # noqa: PLC0415

    result: list[PipelineJobStatus] = []
    in_memory_ids: set[str] = set()

    for job in list_jobs(owner_user_id=actor.user_id):
        in_memory_ids.add(job.job_id)
        analysis_id = job.analysis_job_id
        linked: LinkedAnalysisSummary | None = None
        if analysis_id is not None:
            aj = aj_module.get_job(analysis_id)
            if aj is not None:
                # Defensive: analysis jobs always inherit import job owner.
                if aj.owner_user_id is not None and aj.owner_user_id != actor.user_id:
                    aj = None
            if aj is not None:
                linked = LinkedAnalysisSummary(
                    job_id=aj.job_id,
                    state=aj.state.value,
                    progress_completed=aj.progress_completed,
                    progress_total=aj.progress_total,
                    error_message=aj.error_message,
                )

        import_state = job.state
        stage = _derive_stage(import_state, analysis_id, job.analysis_queued, linked)

        import_result_summary: ImportResultSummary | None = None
        health_import_result_summary: HealthImportResultSummary | None = None
        snapshot = job.import_result_snapshot
        if snapshot is not None:
            if job.job_type == JobType.HEALTH_UPLOAD:
                health_import_result_summary = HealthImportResultSummary(
                    inserted=snapshot.get("inserted", 0),
                    skipped=snapshot.get("skipped", 0),
                    nights_recomputed=snapshot.get("nights_recomputed", 0),
                )
            else:
                import_result_summary = _to_import_result_summary(snapshot)

        result.append(
            PipelineJobStatus(
                job_id=job.job_id,
                job_type=job.job_type.value,
                state=import_state.value,
                stage=stage,
                file_count=job.file_count,
                created_at=job.created_at_wall,
                started_at=job.started_at_wall,
                finished_at=job.finished_at_wall,
                progress_message=job.latest_progress_message,
                sessions_imported=job.sessions_imported,
                import_result=import_result_summary,
                health_import_result=health_import_result_summary,
                error_message=job.error_message,
                analysis_job_id=analysis_id,
                analysis_queued=job.analysis_queued,
                linked_analysis=linked,
            )
        )

    # Historical records from the database — terminal states only (see
    # terminal_records_query: non-terminal persisted rows are orphans whose live
    # state lives in memory and must not surface as phantom "running" entries).
    def _db_row_to_status(rec: models.ImportJobRecord) -> PipelineJobStatus:
        rec_state = JobState(rec.state)
        stage = _derive_stage(rec_state, None, rec.analysis_queued, None)

        rec_import_result_summary: ImportResultSummary | None = None
        rec_health_import_result_summary: HealthImportResultSummary | None = None
        if rec.import_result_json:
            if rec.job_type == "health_upload":
                rec_health_import_result_summary = HealthImportResultSummary(
                    inserted=rec.import_result_json.get("inserted", 0),
                    skipped=rec.import_result_json.get("skipped", 0),
                    nights_recomputed=rec.import_result_json.get(
                        "nights_recomputed", 0
                    ),
                )
            else:
                rec_import_result_summary = _to_import_result_summary(
                    rec.import_result_json
                )

        return PipelineJobStatus(
            job_id=rec.job_id,
            job_type=rec.job_type,
            state=rec.state,
            stage=stage,
            file_count=rec.file_count,
            created_at=rec.created_at,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            progress_message=None,
            sessions_imported=rec.sessions_imported,
            import_result=rec_import_result_summary,
            health_import_result=rec_health_import_result_summary,
            error_message=rec.error_message,
            analysis_job_id=None,
            analysis_queued=rec.analysis_queued,
            linked_analysis=None,
        )

    stmt = terminal_records_query(
        models.ImportJobRecord,
        actor.user_id,
        [s.value for s in TERMINAL_STATES],
    )
    db_records = (await db.execute(stmt)).scalars().all()

    merged = merge_job_lists(
        result,
        in_memory_ids,
        db_records,
        to_status=_db_row_to_status,
        sort_key=lambda j: j.created_at,
    )
    return PipelineJobsListResponse(jobs=merged)


@router.delete("/{job_id}", status_code=204)
def cancel_import(job_id: str, actor: RequireWritable) -> None:
    """Cancel an import job.

    Requires write access and ownership of the job. Returns 404 for foreign jobs
    (no information leak about other users' job IDs).

    Jobs without an owner (owner_user_id=None) are accessible in local mode.
    """
    owned_or_404(
        get_job(job_id), actor.user_id, not_found_detail="Import job not found"
    )
    cancel_or_409(
        cancel_job,
        job_id,
        already_detail="Import job is already finished and cannot be cancelled",
    )


_SSE_TIMEOUT = object()


async def _sse_generator(job: ImportJob) -> AsyncGenerator[str]:
    loop = asyncio.get_running_loop()
    ch = job.attach_observer()

    def _get_or_sentinel() -> object:
        msg = ch.get(timeout=1.0)
        return _SSE_TIMEOUT if msg is None and not ch._closed else msg

    try:
        while True:
            try:
                msg = await asyncio.wait_for(
                    loop.run_in_executor(None, _get_or_sentinel),
                    timeout=2.0,
                )
            except TimeoutError:
                yield ": keepalive\n\n"
                continue

            if msg is _SSE_TIMEOUT:
                yield ": keepalive\n\n"
                continue

            if msg is None:
                break

            assert isinstance(msg, dict)
            event_type = msg.get("event", "progress")
            data = json.dumps(msg.get("data", {}))
            yield f"event: {event_type}\ndata: {data}\n\n"

            if event_type in ("complete", "error"):
                break
    finally:
        job.detach_observer(ch)


@router.get("/{job_id}/progress")
async def import_progress(job_id: str, actor: ActorDep) -> StreamingResponse:
    """Attach an SSE observer to an existing job. Never starts/restarts the worker.

    Returns 404 for foreign jobs (no information leak about other users' job IDs).
    Jobs without an owner (owner_user_id=None) are accessible in local mode.
    """
    job = get_job(job_id)
    if job is None or (
        job.owner_user_id is not None and job.owner_user_id != actor.user_id
    ):
        raise HTTPException(status_code=404, detail="Import job not found")

    return StreamingResponse(
        _sse_generator(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
