"""Import worker: the background thread function that executes import jobs.

Kept in its own module to avoid a circular dependency — the router that accepts
uploads (routers/import_data.py) needs to enqueue jobs without importing the
worker body, and the worker body needs the ImportService which is unrelated to
HTTP routing.

The lifespan (app.py) passes _run_import to start_import_worker() so the
persistent queue thread holds no direct reference to the router module.

Import, health-import, and rescan jobs all follow one 10-step ordering contract
(persist RUNNING → validate → acquire+import → phase_complete → post-import
cancel → enqueue analysis → terminal → persist → cleanup → release capacity).
That contract is single-sourced in ``_run_job``; the only per-job-type
divergence — validation, source acquisition, and whether analysis is enqueued —
lives in a ``_WorkerSpec`` (see issue #272).
"""

from __future__ import annotations

import asyncio
import logging
import re

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from snore.api.import_jobs import ImportJob, JobPhase, JobType
from snore.services.health_import_service import HealthImportService
from snore.services.import_service import ImportService

logger = logging.getLogger(__name__)

# Matches unix absolute paths; used by _client_safe_error to strip spool paths
# that may appear in ValueError/FileNotFoundError messages from parsers or services.
_ABS_PATH_RE = re.compile(r"/\S+")


def _client_safe_error(exc: BaseException) -> str:
    """Return a client-safe error message with filesystem paths redacted.

    Full exception details are always emitted via logger.exception before this
    is called, so no diagnostic information is lost server-side.
    """
    if isinstance(exc, FileNotFoundError):
        return "Export file not found or is missing expected content."
    return _ABS_PATH_RE.sub("[path redacted]", str(exc))


def _make_terminal(
    job: ImportJob,
    event: str,
    *,
    message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a terminal SSE payload, injecting import_committed + import_result
    when the import phase already committed."""
    data: dict[str, Any] = {}
    if message is not None:
        data["message"] = message
    if extra:
        data.update(extra)
    snapshot = job.import_result_snapshot
    if snapshot is not None:
        data["import_committed"] = True
        data["import_result"] = snapshot
    return {"event": event, "data": data}


async def _upsert_job_record(job: ImportJob) -> None:
    """Upsert the current job state to the database for crash-recovery durability.

    Called at each state transition (PENDING → RUNNING → terminal) so a server
    restart can detect orphaned in-progress rows and mark them failed.  Uses
    SQLite's ``ON CONFLICT DO UPDATE`` (via ``jobs.durability``) so the same
    job_id is never double-inserted regardless of how many times this is called.
    """
    from snore.api.jobs.durability import upsert_job_record  # noqa: PLC0415
    from snore.database.models import ImportJobRecord  # noqa: PLC0415

    now = datetime.now(UTC)
    finished = job.finished_at_wall if job.is_terminal else None
    spool_dir_path = str(job.temp_dir) if job.temp_dir is not None else None

    values = {
        "job_id": job.job_id,
        "job_type": job.job_type.value,
        "owner_user_id": job.owner_user_id,
        "target_profile_id": job.target_profile_id,
        "state": job.state.value,
        "file_count": job.file_count,
        "sessions_imported": job.sessions_imported,
        "import_result_json": job.import_result_snapshot,
        "error_message": job.error_message,
        "analysis_queued": job.analysis_queued,
        "spool_dir_path": spool_dir_path,
        "force": job.force,
        "created_at": job.created_at_wall,
        "started_at": job.started_at_wall,
        "finished_at": finished,
        "updated_at": now,
    }
    await upsert_job_record(
        ImportJobRecord,
        values=values,
        update_fields=[
            "state",
            "file_count",
            "sessions_imported",
            "import_result_json",
            "error_message",
            "analysis_queued",
            "spool_dir_path",
            "started_at",
            "finished_at",
            "updated_at",
        ],
    )


# ---------------------------------------------------------------------------
# Shared runner: one 10-step ordering contract, parametrised per job type.
# ---------------------------------------------------------------------------


class _CancelledEarly(Exception):
    """Raised from a spec's ``acquire_and_import`` when cancellation is detected
    before the import commits.

    Caught by ``_run_job`` ahead of the generic ``except Exception`` so the
    terminal transition goes through ``_finish(succeeded=False)`` with no
    ``terminal_msg`` — the cancel flag already produces the cancel payload — and
    the message is never routed through ``_client_safe_error`` into an "error"
    event.
    """


@dataclass(frozen=True)
class _WorkerSpec:
    """The per-job-type divergence around the shared 10-step runner.

    ``validate`` returns the snapshotted ``target_profile_id`` (raising
    ``ValueError`` on bad config). ``acquire_and_import`` reports progress,
    performs early-cancel checks (raising ``_CancelledEarly``), locates sources,
    and runs the heavy import, returning a ``model_dump``-able service result.
    ``enqueue_analysis`` is True for import/rescan (CPAP sessions feed analysis)
    and False for health imports (which write HealthSample rows, not sessions).
    """

    label: str
    validate: Callable[[ImportJob, Path | None], int]
    acquire_and_import: Callable[[ImportJob, int, Path | None], Any]
    enqueue_analysis: bool


def _persist_running(job: ImportJob) -> None:
    """Persist RUNNING state; the worker loop already called try_start()."""
    try:
        asyncio.run(_upsert_job_record(job))
    except Exception:
        logger.exception("Failed to upsert RUNNING state for job %s", job.job_id)


def _maybe_enqueue_analysis(
    job: ImportJob,
    spec: _WorkerSpec,
    target_profile_id: int,
    result: Any,
    import_result_dict: dict[str, Any],
) -> dict[str, Any]:
    """Enqueue background analysis for imported sessions (import/rescan only) and
    build the terminal "complete" extra payload.

    Health imports (``enqueue_analysis`` False) and rescans with ``analyze``
    off skip analysis entirely and never touch ``set_analysis_link``, so the
    /import/jobs list endpoint reports them as done. Otherwise
    ``set_analysis_link`` is always called — even when nothing was imported —
    because that endpoint reads those fields.
    """
    if not (spec.enqueue_analysis and job.analyze):
        return {"result": import_result_dict}

    analysis_job_id = None
    imported_ids = result.imported_session_ids
    if imported_ids:
        from snore.api import analysis_jobs  # noqa: PLC0415

        aj = analysis_jobs.enqueue(
            profile_id=target_profile_id,
            session_ids=imported_ids,
            source=analysis_jobs.AnalysisJobSource.IMPORT,
            owner_user_id=job.owner_user_id,
        )
        if aj is not None:
            analysis_job_id = aj.job_id
        else:
            logger.warning(
                "Analysis queue full; skipping auto-analysis for %s job %s",
                spec.label.lower(),
                job.job_id,
            )

    job.set_analysis_link(
        analysis_job_id=analysis_job_id,
        queue_full=bool(imported_ids) and analysis_job_id is None,
    )

    terminal_extra: dict[str, Any] = {"result": import_result_dict}
    if analysis_job_id is not None:
        terminal_extra["analysis_job_id"] = analysis_job_id
    elif imported_ids:
        # Queue was full — tell the client so it can distinguish from
        # "nothing was imported" (where analysis_queued is absent).
        terminal_extra["analysis_queued"] = False
    return terminal_extra


def _finalize(job: ImportJob) -> None:
    """Terminal cleanup, in strict order: persist the terminal record, clean the
    spool (unless shutting down), then release the capacity slot LAST — the slot
    owns the disk it admitted, so releasing before cleanup would let a new
    request spool over the still-present temp tree.
    """
    if job.is_terminal:
        try:
            asyncio.run(_upsert_job_record(job))
        except Exception:
            logger.exception("Failed to persist job record for %s", job.job_id)
    from snore.api.import_jobs import is_shutdown_in_progress  # noqa: PLC0415

    if not is_shutdown_in_progress():
        job.cleanup_files()
    job.release_capacity()


def _run_job(job: ImportJob, spec: _WorkerSpec, profile_raw_root: Path | None) -> None:
    """Execute one import job under the shared 10-step ordering contract.

    Ordering contract:
        1. Persist RUNNING state.
        2. Validate target profile (+ archive root for rescan).
        3. Acquire sources + run the import (heavy I/O; early-cancel aware).
        4. phase_complete(IMPORT) — non-terminal milestone for observers.
        5. Post-import cancel check → terminal "Cancelled".
        6. Enqueue background analysis (import/rescan only).
        7. set_analysis_link + terminal "complete".
        8. Persist terminal record.
        9. Clean parser spool + job temp (unless shutting down).
        10. Release capacity (slot owns the disk it admitted).

    Every terminal payload carries import_committed + import_result once the
    import phase committed, even on analysis failure or cancellation.
    """
    try:
        _persist_running(job)
        target_profile_id = spec.validate(job, profile_raw_root)
        result = spec.acquire_and_import(job, target_profile_id, profile_raw_root)

        # --- Phase 1 complete: import committed ---
        import_result_dict = result.model_dump()
        job.phase_complete(JobPhase.IMPORT, import_result_dict)

        if job.cancel_requested:
            job._finish(
                succeeded=False,
                terminal_msg=_make_terminal(job, "error", message="Cancelled"),
            )
            return

        terminal_extra = _maybe_enqueue_analysis(
            job, spec, target_profile_id, result, import_result_dict
        )
        job._finish(
            succeeded=True,
            terminal_msg=_make_terminal(job, "complete", extra=terminal_extra),
        )
    except _CancelledEarly:
        # Cancel flag is already set; _finish yields the cancel payload.
        job._finish(succeeded=False)
    except Exception as e:
        logger.exception("%s job %s failed", spec.label, job.job_id)
        job._finish(
            succeeded=False,
            terminal_msg=_make_terminal(job, "error", message=_client_safe_error(e)),
        )
    finally:
        _finalize(job)


# ---------------------------------------------------------------------------
# Per-job-type validation and acquisition — the only divergent code.
# ---------------------------------------------------------------------------


def _validate_target_profile(job: ImportJob, profile_raw_root: Path | None) -> int:
    """Import/health: require a snapshotted target profile."""
    target_profile_id = job.target_profile_id
    if target_profile_id is None:
        raise ValueError("Import job has no target profile — cannot proceed")
    return target_profile_id


def _validate_rescan(job: ImportJob, profile_raw_root: Path | None) -> int:
    """Rescan: require both a target profile and an archive root."""
    target_profile_id = job.target_profile_id
    if target_profile_id is None:
        raise ValueError("Rescan job has no target profile — cannot proceed")
    if profile_raw_root is None:
        raise ValueError("Rescan job has no archive root — cannot proceed")
    return target_profile_id


def _acquire_import(
    job: ImportJob, target_profile_id: int, profile_raw_root: Path | None
) -> Any:
    """UPLOAD: detect sources from the spool and import with backup=True."""
    service = ImportService()
    if not (job.job_type == JobType.UPLOAD and job.temp_dir is not None):
        raise ValueError("Invalid job configuration")

    job.report_progress("Detecting data sources...")
    if job.cancel_requested:
        raise _CancelledEarly
    sources = service.detect_sources(job.temp_dir)
    job.report_progress(f"Detected {len(sources)} source(s)")
    if job.cancel_requested:
        raise _CancelledEarly

    return asyncio.run(
        service.import_sources(
            sources,
            force=job.force,
            backup=True,
            backup_root=profile_raw_root,
            profile_id=target_profile_id,
            progress_callback=lambda msg: job.report_progress(msg),
            cancel_predicate=lambda: job.cancel_requested,
        )
    )


def _acquire_health(
    job: ImportJob, target_profile_id: int, profile_raw_root: Path | None
) -> Any:
    """HEALTH_UPLOAD: locate the upload file/dir and run HealthImportService.

    Cancellation is checked at the start of each batch inside the service;
    already-committed batches are kept and their nightly summaries recomputed,
    so partial counts flow through to the terminal payload.
    """
    job.report_progress("Importing Apple Health data...")
    if job.cancel_requested:
        raise _CancelledEarly

    # Locate the upload: single file → pass it directly (works for a zip);
    # multiple files or just the dir → pass the temp dir itself.
    if job.temp_dir is None:
        raise ValueError("Import job has no temp directory — cannot proceed")
    files = [p for p in job.temp_dir.rglob("*") if p.is_file()]
    path = files[0] if len(files) == 1 else job.temp_dir

    return asyncio.run(
        HealthImportService().import_file(
            path,
            target_profile_id,
            progress_callback=lambda n: job.report_progress(f"Processed {n:,} records"),
            cancel_predicate=lambda: job.cancel_requested,
        )
    )


def _acquire_rescan(
    job: ImportJob, target_profile_id: int, profile_raw_root: Path | None
) -> Any:
    """RESCAN: detect sources from the archive root and import with backup=False
    — the archive IS the source. ``force`` and the date range come from the job."""
    service = ImportService()
    job.report_progress("Scanning archive for device data...")
    if job.cancel_requested:
        raise _CancelledEarly

    # Guaranteed non-None by _validate_rescan; assert narrows for the type checker.
    assert profile_raw_root is not None
    sources = service.detect_sources(profile_raw_root)
    if not sources:
        raise ValueError("No device data found in archive")

    job.report_progress(f"Detected {len(sources)} source(s) — importing from archive")
    if job.cancel_requested:
        raise _CancelledEarly

    return asyncio.run(
        service.import_sources(
            sources,
            force=job.force,
            backup=False,
            date_from=job.date_from.isoformat() if job.date_from else None,
            date_to=job.date_to.isoformat() if job.date_to else None,
            profile_id=target_profile_id,
            progress_callback=lambda msg: job.report_progress(msg),
            cancel_predicate=lambda: job.cancel_requested,
        )
    )


_IMPORT_SPEC = _WorkerSpec(
    label="Import",
    validate=_validate_target_profile,
    acquire_and_import=_acquire_import,
    enqueue_analysis=True,
)

_HEALTH_SPEC = _WorkerSpec(
    label="Health import",
    validate=_validate_target_profile,
    acquire_and_import=_acquire_health,
    enqueue_analysis=False,
)

_RESCAN_SPEC = _WorkerSpec(
    label="Rescan",
    validate=_validate_rescan,
    acquire_and_import=_acquire_rescan,
    enqueue_analysis=True,
)


def _run_import(job: ImportJob, profile_raw_root: Path | None = None) -> None:
    """Worker function for UPLOAD (CPAP) import jobs. Runs in a background thread."""
    _run_job(job, _IMPORT_SPEC, profile_raw_root)


def _run_health_import(job: ImportJob, profile_raw_root: Path | None = None) -> None:
    """Worker function for Apple Health import jobs. Runs in a background thread."""
    _run_job(job, _HEALTH_SPEC, profile_raw_root)


def _run_rescan(job: ImportJob, profile_raw_root: Path | None = None) -> None:
    """Worker function for archive rescan jobs. Runs in a background thread."""
    _run_job(job, _RESCAN_SPEC, profile_raw_root)


def _run_dispatch(job: ImportJob, profile_raw_root: Path | None = None) -> None:
    """Route import jobs to the appropriate worker function based on job type.

    JobType.HEALTH_UPLOAD → _run_health_import; JobType.RESCAN → _run_rescan;
    everything else → _run_import (handles UPLOAD/CPAP jobs).
    """
    if job.job_type == JobType.HEALTH_UPLOAD:
        _run_health_import(job, profile_raw_root)
    elif job.job_type == JobType.RESCAN:
        _run_rescan(job, profile_raw_root)
    else:
        _run_import(job, profile_raw_root)
