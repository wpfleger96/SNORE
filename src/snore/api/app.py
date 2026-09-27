from __future__ import annotations

import importlib.resources
import logging
import os
import resource
import uuid

from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from snore.api.analysis_jobs import shutdown as _shutdown_analysis_jobs
from snore.api.analysis_jobs import start_worker as _start_analysis_worker
from snore.api.errors import (
    NotFoundError,
    auth_validation_error_handler,
    not_found_handler,
    server_error_handler,
)
from snore.api.import_jobs import get_live_spool_dirs
from snore.api.import_jobs import shutdown as _shutdown_import_jobs
from snore.api.import_jobs import start_import_worker as _start_import_worker
from snore.api.import_jobs import start_reaper as _start_import_reaper
from snore.api.middleware import (
    AuthMiddleware,
    AuthPathMiddleware,
    RateLimitMiddleware,
    TotpEnforcementMiddleware,
)
from snore.api.routers import (
    about,
    admin,
    analysis,
    days,
    db,
    devices,
    equipment,
    events,
    export,
    import_data,
    profiles,
    reports,
    rx,
    sessions,
    stats,
    validation,
    waveforms,
)
from snore.api.routers import auth as auth_router
from snore.api.routers import health as health_routes
from snore.api.routers import me as me_router
from snore.api.validation_jobs import shutdown as _shutdown_validation_jobs
from snore.api.validation_jobs import start_worker as _start_validation_worker
from snore.database.session import init_database, init_database_from_url

API_V1_PREFIX = "/api/v1"

logger = logging.getLogger(__name__)

try:
    __version__ = get_version("snore")
except PackageNotFoundError:
    __version__ = "dev"

import time

_STARTUP_TIME: float = time.monotonic()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Raise the FD soft limit so Starlette's multipart parser can open a
    # SpooledTemporaryFile per uploaded file without hitting EMFILE.
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _target = min(_hard, 65_536) if _hard != resource.RLIM_INFINITY else 65_536
    if _soft < _target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_target, _hard))
        logger.info("Raised file-descriptor limit: %d → %d", _soft, _target)

    # Load and validate config first — fail fast on misconfiguration.
    from snore.api.config import load_config, set_config  # noqa: PLC0415

    bind_host = os.environ.get("SNORE_BIND_HOST", "127.0.0.1")
    cfg = load_config(bind_host_override=bind_host)
    set_config(cfg)

    if cfg.is_multiuser and not cfg.is_google_configured:
        logger.warning(
            "Google OAuth not configured (GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET absent)"
            " — /auth/google endpoints will return 503"
        )

    # Honour the canonical URL exported by `snore serve` first; fall back to
    # SNORE_DB_PATH for direct uvicorn invocations and the e2e test harness.
    database_url = os.environ.get("SNORE_DATABASE_URL")
    if database_url:
        await init_database_from_url(database_url)
    else:
        db_path = os.environ.get("SNORE_DB_PATH")
        await init_database(db_path)

    # Startup recovery + acquire lifetime shared writer lease.
    import asyncio  # noqa: PLC0415

    from snore.services.profile_service import DeletionSaga  # noqa: PLC0415
    from snore.services.writer_lease import get_writer_lease  # noqa: PLC0415

    lease = get_writer_lease()

    exclusive_held = False
    try:
        lease.acquire_exclusive()
        exclusive_held = True
    except Exception as exc:
        logger.warning(
            "Startup recovery skipped (exclusive lease unavailable): %s", exc
        )

    if exclusive_held:
        saga = DeletionSaga()
        try:
            await asyncio.to_thread(saga.recover)
        except Exception as exc:
            logger.warning("Startup recovery failed: %s", exc)
        finally:
            lease.release_exclusive()

    lease.acquire_shared()

    # Schedule pending vacuum if a prior restart interrupted it before VACUUM ran.
    from snore.constants import DEFAULT_VACUUM_PENDING_MARKER  # noqa: PLC0415
    from snore.services.database_service import _vacuum_background  # noqa: PLC0415

    if DEFAULT_VACUUM_PENDING_MARKER.exists():
        try:
            pending_db_path = DEFAULT_VACUUM_PENDING_MARKER.read_text().strip()
            if pending_db_path:
                logger.info(
                    "Startup: vacuum pending marker found — scheduling background VACUUM"
                )
                loop = asyncio.get_running_loop()
                loop.run_in_executor(None, _vacuum_background, pending_db_path)
        except Exception as exc:
            logger.warning("Startup: failed to schedule pending VACUUM: %s", exc)

    # Any container start means the deferred deploy either happened or is stale.
    from snore.constants import DEFAULT_DEPLOY_DEFERRED_MARKER  # noqa: PLC0415

    try:
        if DEFAULT_DEPLOY_DEFERRED_MARKER.exists():
            logger.info("Startup: clearing deploy-deferred marker")
        DEFAULT_DEPLOY_DEFERRED_MARKER.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Startup: failed to clear deploy-deferred marker: %s", exc)

    # Purge expired/consumed oauth_attempts at startup.
    await _startup_purge_expired_oauth_attempts()

    # Auto-create demo user and import bundled fixture data if not present.
    # Bootstrap a first admin invite when no active admin exists yet.
    if cfg.is_multiuser:
        await _startup_ensure_demo_data(app)
        await _startup_ensure_bootstrap_admin()

    # Mark orphaned import/analysis jobs as failed; collect resume candidates.
    import_resume_candidates = await _recover_orphaned_import_jobs()
    analysis_affected_profiles = await _recover_orphaned_analysis_jobs()
    # Validation runs are idempotent — mark orphans failed (no auto-resume) and
    # prune to the retention cap in the same startup sweep.
    await _recover_orphaned_validation_jobs()

    # Clean stale spool directories, skipping any that will be resumed.
    skip_paths = {c[0] for c in import_resume_candidates}
    _cleanup_stale_upload_spool_dirs(skip_paths=skip_paths)

    # Start a single lifespan-owned TTL reaper, the analysis job worker, and the
    # import job worker (serialises execution to avoid SQLite write-lock contention).
    # The spool sweep runs each reaper tick so directories orphaned by a crash or
    # restart mid-upload are removed periodically, not only at startup.  The lambda
    # calls get_live_spool_dirs() at each tick — not at capture time — so active
    # uploads are always excluded from the sweep.
    reaper_thread, reaper_stop = _start_import_reaper(
        interval=60.0,
        spool_sweep_fn=lambda: _cleanup_stale_upload_spool_dirs(
            skip_paths=get_live_spool_dirs()
        ),
    )
    _start_analysis_worker()
    _start_validation_worker()
    from snore.api.import_worker import _run_dispatch  # noqa: PLC0415

    _start_import_worker(_run_dispatch)

    # Resume interrupted jobs after workers are ready to process them.
    if import_resume_candidates:
        _startup_resume_imports(import_resume_candidates)
    if analysis_affected_profiles:
        await _startup_resume_analysis(analysis_affected_profiles)

    mcp_app = getattr(app.state, "mcp_app", None)
    try:
        if mcp_app is not None:
            # Chain the FastMCP sub-app lifespan inside the existing try/finally
            # so that reaper/worker shutdown and lease release ordering is preserved.
            # StarletteWithLifespan.lifespan is a property returning the context factory.
            async with mcp_app.lifespan(mcp_app):
                yield
        else:
            yield
    finally:
        reaper_stop.set()
        reaper_thread.join(timeout=5.0)
        _still_alive_error: RuntimeError | None = None
        try:
            still_alive = _shutdown_import_jobs()
            if still_alive:
                _still_alive_error = RuntimeError(
                    f"Shutdown incomplete: {len(still_alive)} import worker(s) still alive "
                    f"after timeout: {still_alive}. Active import writes may be interrupted."
                )
        finally:
            # Stop analysis jobs first — they may have futures in the shared process
            # pool, so the pool must still be alive while they drain.
            try:
                _shutdown_analysis_jobs()
            except Exception:
                logger.warning("Error shutting down analysis jobs", exc_info=True)
            try:
                _shutdown_validation_jobs()
            except Exception:
                logger.warning("Error shutting down validation jobs", exc_info=True)
            try:
                from snore.utils.process_pool import shutdown_pool  # noqa: PLC0415

                shutdown_pool(wait=False)
            except Exception:
                logger.warning(
                    "Error shutting down compute process pool", exc_info=True
                )
            try:
                from snore.utils.parse_pool import (  # noqa: PLC0415
                    shutdown_pool as shutdown_parse_pool,
                )

                shutdown_parse_pool(wait=False)
            except Exception:
                logger.warning("Error shutting down parse process pool", exc_info=True)
            # Release the shared writer lease unconditionally — even if import-job
            # shutdown raises the still-alive RuntimeError above.
            lease.release()
        if _still_alive_error is not None:
            raise _still_alive_error


async def _startup_purge_expired_oauth_attempts() -> None:
    """Delete expired and consumed oauth_attempts rows at startup."""
    try:
        from datetime import UTC, datetime  # noqa: PLC0415

        from snore.api.routers.auth._common import (  # noqa: PLC0415
            purge_expired_oauth_attempts,
        )
        from snore.database.session import session_scope  # noqa: PLC0415

        now = datetime.now(UTC)
        async with session_scope() as db:
            purged = await purge_expired_oauth_attempts(db, now)
        if purged:
            logger.info("Purged %d expired/consumed oauth_attempts rows", purged)
    except Exception as exc:
        logger.warning("oauth_attempts purge failed: %s", exc)


async def _startup_ensure_demo_data(app: FastAPI) -> None:
    """Auto-create demo user and import bundled fixture data if not present.

    Idempotent: exits immediately when demo data already exists.  Failures are
    logged as warnings and never prevent startup.

    The two separate session_scope() windows are deliberate: no write lock is held
    during EDF parsing (which can take tens of seconds).  Concurrent multi-worker
    first boot can double-import benignly — ensure_user_and_profile's cascade-delete
    makes re-entry idempotent.
    """
    import time as _time  # noqa: PLC0415

    from snore.database.session import session_scope  # noqa: PLC0415
    from snore.services.demo_service import DemoService  # noqa: PLC0415

    try:
        async with session_scope() as db:
            if await DemoService(db).demo_data_exists():
                logger.debug("Demo data already exists — skipping fixture import")
                app.state.demo_available = True
                return

        fixtures_dir = Path(str(importlib.resources.files("snore.demo"))) / "fixtures"
        if not fixtures_dir.is_dir() or not any(
            d for d in fixtures_dir.iterdir() if d.is_dir()
        ):
            logger.info(
                "No demo fixtures found at %s — demo login will be unavailable",
                fixtures_dir,
            )
            return

        logger.info("Initializing demo account from bundled fixtures…")
        t0 = _time.monotonic()
        async with session_scope() as db:
            counts = await DemoService(db).import_from_fixtures(fixtures_dir)
        elapsed = _time.monotonic() - t0
        logger.info(
            "Demo data ready in %.1fs — %d sessions imported, %d skipped, %d failed",
            elapsed,
            counts.get("sessions", 0),
            counts.get("skipped", 0),
            counts.get("failed", 0),
        )
        if counts.get("sessions", 0) > 0:
            app.state.demo_available = True
    except Exception:
        logger.warning(
            "Demo startup initialization failed — demo login will be unavailable",
            exc_info=True,
        )


async def _create_bootstrap_admin_invite(email: str) -> str | None:
    """Create a bootstrap admin invite if one is needed.

    Returns the raw (unhashed) token when a new invite was inserted, or None
    when skipped (active admin exists, user row already exists for the address,
    or a valid pending admin invite is already present).

    The redemption URL (containing the one-time token) is deliberately written
    to the server log as the delivery channel; the invite is single-use with a
    7-day expiry, so logs should be treated as sensitive until it is redeemed.
    """
    import secrets  # noqa: PLC0415

    from datetime import UTC, datetime, timedelta  # noqa: PLC0415

    from sqlalchemy import insert, literal, null, select  # noqa: PLC0415

    from snore.auth.invite import invite_valid_clauses  # noqa: PLC0415
    from snore.auth.invite_tokens import hash_invite_token  # noqa: PLC0415
    from snore.database import models  # noqa: PLC0415
    from snore.database.session import session_scope  # noqa: PLC0415
    from snore.database.types import UTCDateTime  # noqa: PLC0415

    async with session_scope() as db:
        now = datetime.now(UTC)

        active_admin = (
            (
                await db.execute(
                    select(models.User).where(
                        models.User.role == "admin",
                        models.User.disabled_at.is_(None),
                    )
                )
            )
            .scalars()
            .first()
        )
        if active_admin is not None:
            logger.debug("Bootstrap admin: active admin user already exists — skipping")
            return None

        # Guard: if ANY user row exists for this email (regardless of role or
        # disabled state), an invite would be unredeemable — the redemption
        # route rejects addresses that already have an account.
        existing_user = (
            (
                await db.execute(
                    select(models.User).where(
                        models.User.canonical_email == email,
                    )
                )
            )
            .scalars()
            .first()
        )
        if existing_user is not None:
            logger.warning(
                "Bootstrap admin: a user with email %s already exists but is not an"
                " active admin — recover with the snore user CLI"
                " (e.g. 'snore user enable %s')",
                email,
                email,
            )
            return None

        raw = secrets.token_urlsafe(32)
        token_hash = hash_invite_token(raw)
        expires_at = now + timedelta(days=7)

        # Atomic INSERT … SELECT … WHERE NOT EXISTS: the pending-admin check and
        # the insert are one statement so concurrent startups cannot both insert.
        # A pending MEMBER invite for the same address does not block — only a
        # pending ADMIN invite prevents creation.
        pending = (
            select(models.Invite.id)
            .where(
                models.Invite.email == email,
                models.Invite.role == "admin",
                *invite_valid_clauses(now),
            )
            .exists()
        )
        stmt = insert(models.Invite).from_select(
            ["email", "token_hash", "role", "created_by", "expires_at", "created_at"],
            select(
                literal(email),
                literal(token_hash),
                literal("admin"),
                null(),
                literal(expires_at, UTCDateTime()),
                literal(now, UTCDateTime()),
            ).where(~pending),
        )
        result = await db.execute(stmt)
        if int(result.rowcount) == 0:  # type: ignore[attr-defined]
            logger.info(
                "Bootstrap admin invite for %s is already pending"
                " — if its URL is lost, revoke it ('snore user invite-revoke %s')"
                " and restart to mint a new one",
                email,
                email,
            )
            return None

    return raw


async def _startup_ensure_bootstrap_admin() -> None:
    """Auto-create a bootstrap admin invite when no active admin user exists.

    No-op when ``SNORE_BOOTSTRAP_ADMIN_EMAIL`` is unset.  Idempotent: returns
    early if an active admin user exists, a user row already exists for the
    bootstrap address, or a valid pending admin invite for that address is
    already present.  Failures are logged as warnings and never prevent startup.

    The redemption URL (containing the one-time token) is deliberately written
    to the server log as the delivery channel; the invite is single-use with a
    7-day expiry, so logs should be treated as sensitive until it is redeemed.
    """
    from snore.api.config import get_config  # noqa: PLC0415

    cfg = get_config()
    if cfg.bootstrap_admin_email is None:
        return

    email = cfg.bootstrap_admin_email
    raw: str | None = None
    try:
        raw = await _create_bootstrap_admin_invite(email)
    except Exception:
        logger.warning(
            "Bootstrap admin invite creation failed"
            " — create one manually with: snore user invite %s --role admin",
            email,
            exc_info=True,
        )

    if raw is not None:
        base = cfg.public_base_url.rstrip("/") if cfg.public_base_url else ""
        invite_url = f"{base}/invite#{raw}" if base else f"/invite#{raw}"
        # WARNING, not INFO: the URL carries a live one-time admin credential
        # and needs operator action — pipelines that restrict WARN+ streams
        # should treat it as sensitive until redeemed.
        logger.warning(
            "Bootstrap admin invite created for %s — redeem at: %s",
            email,
            invite_url,
        )


# Stale-temp retention: any snore-upload-* dir older than this is orphaned.
_STALE_UPLOAD_TMPDIR_AGE_SECONDS: float = 2 * 3600  # 2 hours


def _cleanup_stale_upload_spool_dirs(
    skip_paths: set[Path] | frozenset[Path] = frozenset(),
) -> None:
    """Remove orphaned upload spool directories.

    Called at startup (once, to clear dirs from a previous crashed process) and
    then periodically by the reaper thread.  The periodic call passes
    ``get_live_spool_dirs()`` as *skip_paths* so active uploads are never
    removed — chunk writes land in subdirectories and do not bump the top-level
    spool dir mtime, making mtime alone an unreliable liveness signal.

    Scans both the legacy ``snore-upload-*`` prefix in the system temp dir
    (backward compat) and the durable spool directory on the persistent volume.
    Directories in *skip_paths* are preserved unconditionally.
    """
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415
    import time  # noqa: PLC0415

    from snore.api.config import get_config  # noqa: PLC0415

    now = time.time()
    cleaned = 0

    scan_dirs: list[tuple[Path, bool]] = [
        (Path(tempfile.gettempdir()), True),  # (dir, require_prefix)
    ]
    try:
        spool_dir = get_config().upload_spool_dir
        if spool_dir.exists():
            scan_dirs.append((spool_dir, False))
    except Exception:
        pass

    for parent, require_prefix in scan_dirs:
        try:
            entries = list(parent.iterdir())
        except OSError:
            continue
        for entry in entries:
            if require_prefix and not entry.name.startswith("snore-upload-"):
                continue
            if not entry.is_dir():
                continue
            if entry in skip_paths:
                continue
            try:
                age = now - entry.stat().st_mtime
                if age > _STALE_UPLOAD_TMPDIR_AGE_SECONDS:
                    shutil.rmtree(entry, ignore_errors=True)
                    logger.info(
                        "Cleaned stale upload spool dir: %s (age=%.0fs)", entry, age
                    )
                    cleaned += 1
            except OSError as exc:
                logger.warning(
                    "Could not check/remove stale spool dir %s: %s", entry, exc
                )
    if cleaned:
        logger.info("Removed %d stale upload spool dir(s)", cleaned)


async def _recover_orphaned_import_jobs() -> list[tuple[Path, int, int | None, str]]:
    """Mark orphaned import jobs as failed; return resume candidates.

    Jobs in PENDING_UPLOAD, PENDING, or RUNNING state at startup are orphans
    from a previous server run.  All are marked failed.  Those whose spool
    directory still exists on disk are returned as resume candidates — the
    caller can re-enqueue them as new import jobs.

    Returns 4-tuples of (spool_path, profile_id, owner_user_id, job_type_str).
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from sqlalchemy import select, update  # noqa: PLC0415

    from snore.api.import_jobs import ACTIVE_STATES, JobState  # noqa: PLC0415
    from snore.database import models  # noqa: PLC0415
    from snore.database.session import session_scope  # noqa: PLC0415

    non_terminal = [s.value for s in ACTIVE_STATES]
    resume_candidates: list[tuple[Path, int, int | None, str]] = []
    now = datetime.now(UTC)
    try:
        async with session_scope(immediate=True) as db:
            # Collect resume candidates before marking everything failed.
            rows = (
                await db.execute(
                    select(
                        models.ImportJobRecord.spool_dir_path,
                        models.ImportJobRecord.target_profile_id,
                        models.ImportJobRecord.owner_user_id,
                        models.ImportJobRecord.job_type,
                    ).where(
                        models.ImportJobRecord.state.in_(non_terminal),
                        models.ImportJobRecord.spool_dir_path.is_not(None),
                    )
                )
            ).all()
            for spool_path_str, profile_id, owner_user_id, job_type_str in rows:
                if spool_path_str is None:  # excluded by the is_not(None) filter
                    continue
                spool_path = Path(spool_path_str)
                if profile_id is not None and spool_path.exists():
                    resume_candidates.append(
                        (spool_path, profile_id, owner_user_id, job_type_str)
                    )

            result = await db.execute(
                update(models.ImportJobRecord)
                .where(models.ImportJobRecord.state.in_(non_terminal))
                .values(
                    state=JobState.FAILED.value,
                    error_message="Server restarted while job was in progress",
                    finished_at=now,
                    updated_at=now,
                )
            )
            count = result.rowcount or 0  # type: ignore[attr-defined]
        if count > 0:
            logger.info(
                "Startup recovery: marked %d orphaned import job(s) as failed"
                " (%d resumable)",
                count,
                len(resume_candidates),
            )
    except Exception as exc:
        logger.warning("Orphaned import job recovery failed: %s", exc)
    return resume_candidates


async def _recover_orphaned_analysis_jobs() -> set[int]:
    """Mark orphaned analysis job records as failed; return affected profile IDs.

    Non-terminal ``analysis_job_records`` rows indicate jobs that were
    interrupted by a crash or restart.  They are marked failed, and the
    distinct profile IDs are returned so the caller can gap-fill analysis.
    """
    from datetime import UTC, datetime  # noqa: PLC0415

    from sqlalchemy import select, update  # noqa: PLC0415

    from snore.api.analysis_jobs import AnalysisJobState  # noqa: PLC0415
    from snore.database import models  # noqa: PLC0415
    from snore.database.session import session_scope  # noqa: PLC0415

    affected_profiles: set[int] = set()
    now = datetime.now(UTC)
    try:
        async with session_scope(immediate=True) as db:
            # Collect affected profile IDs before marking failed.
            rows = (
                (
                    await db.execute(
                        select(models.AnalysisJobRecord.profile_id)
                        .where(
                            models.AnalysisJobRecord.state
                            == AnalysisJobState.RUNNING.value
                        )
                        .distinct()
                    )
                )
                .scalars()
                .all()
            )
            affected_profiles = set(rows)

            result = await db.execute(
                update(models.AnalysisJobRecord)
                .where(models.AnalysisJobRecord.state == AnalysisJobState.RUNNING.value)
                .values(
                    state=AnalysisJobState.FAILED.value,
                    error_message="Server restarted while job was in progress",
                    finished_at=now,
                    updated_at=now,
                )
            )
            count = result.rowcount or 0  # type: ignore[attr-defined]
        if count > 0:
            logger.info(
                "Startup recovery: marked %d orphaned analysis job(s) as failed"
                " (profiles: %s)",
                count,
                affected_profiles,
            )
    except Exception as exc:
        logger.warning("Orphaned analysis job recovery failed: %s", exc)
    return affected_profiles


async def _recover_orphaned_validation_jobs() -> None:
    """Fail orphaned validation runs and prune to the retention cap at startup.

    Non-terminal ``validation_runs`` rows are runs interrupted by a crash or
    restart; validation is idempotent so they are marked failed and NOT
    auto-resumed.  The retention prune keeps only the newest rows per
    ``(profile_id, validator_type)`` so the comparison history stays bounded.
    """
    from snore.api import validation_jobs  # noqa: PLC0415

    await validation_jobs.recover_orphaned_runs()
    await validation_jobs.prune_retention()


def _startup_resume_imports(
    candidates: list[tuple[Path, int, int | None, str]],
) -> None:
    """Re-enqueue import jobs for which spool files survived the restart."""
    from snore.api.import_jobs import (  # noqa: PLC0415
        ImportJob,
        JobType,
        enqueue_for_execution,
    )
    from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

    for spool_path, profile_id, owner_user_id, job_type_str in candidates:
        try:
            try:
                job_type = JobType(job_type_str)
            except ValueError:
                logger.warning(
                    "Startup: unknown job_type %r for spool %s — resuming as UPLOAD",
                    job_type_str,
                    spool_path,
                )
                job_type = JobType.UPLOAD
            job = ImportJob(
                job_id=uuid.uuid4().hex,
                job_type=job_type,
                owner_user_id=owner_user_id,
                target_profile_id=profile_id,
                temp_dir=spool_path,
            )
            job._state = job._state.__class__("pending")
            job._file_count = sum(1 for f in spool_path.iterdir() if f.is_file())
            # Register in the in-memory store without checking admission caps —
            # startup resume should not be refused by caps.
            from snore.api.import_jobs import _store  # noqa: PLC0415

            _store.add(job)

            profile_raw_root = DEFAULT_RAW_BACKUP_DIR / str(profile_id)
            enqueue_for_execution(job, profile_raw_root)
            logger.info(
                "Startup: re-enqueued import job %s (spool=%s, files=%d)",
                job.job_id,
                spool_path,
                job._file_count,
            )
        except Exception:
            logger.exception(
                "Startup: failed to re-enqueue import for spool %s", spool_path
            )


async def _startup_resume_analysis(affected_profile_ids: set[int]) -> None:
    """Enqueue gap-fill analysis for profiles that had interrupted analysis."""
    from snore.api import analysis_jobs  # noqa: PLC0415
    from snore.database.session import session_scope  # noqa: PLC0415
    from snore.services.analysis_facade import AnalysisFacade  # noqa: PLC0415

    for profile_id in affected_profile_ids:
        try:
            async with session_scope() as db:
                facade = AnalysisFacade(db, profile_id=profile_id)
                session_ids = await facade.list_session_ids(missing_only=True)
            if not session_ids:
                logger.debug("Startup: no gap-fill needed for profile %d", profile_id)
                continue
            aj = analysis_jobs.enqueue(
                profile_id=profile_id,
                session_ids=session_ids,
                source=analysis_jobs.AnalysisJobSource.BATCH,
            )
            if aj:
                logger.info(
                    "Startup: enqueued gap-fill analysis for profile %d (%d sessions)",
                    profile_id,
                    len(session_ids),
                )
            else:
                logger.warning(
                    "Startup: analysis queue full; gap-fill for profile %d deferred",
                    profile_id,
                )
        except Exception:
            logger.exception(
                "Startup: failed to enqueue gap-fill analysis for profile %d",
                profile_id,
            )


def create_app() -> FastAPI:
    from snore.api.config import get_config  # noqa: PLC0415

    # Ensure config is loaded.  In tests, callers set it via set_config() before
    # calling create_app(); in production, lifespan sets it.  get_config() lazily
    # loads from env when the global config is not yet set.
    cfg = get_config()

    is_multiuser = cfg.is_multiuser

    app = FastAPI(
        title="SNORE API",
        version=__version__,
        lifespan=lifespan,
    )

    # Middleware add order is innermost-first; requests are processed outermost-first.
    # AuthPathMiddleware must be innermost of the auth pair so RateLimitMiddleware
    # fires before the body pre-read, rejecting rate-limited requests cheaply.
    #
    # Request-processing order (outermost → innermost):
    #   AuthMiddleware          — resolves request.state.actor from session cookie
    #   TotpEnforcementMiddleware — gates API access for users pending enrollment;
    #                              must run after AuthMiddleware (needs actor) and
    #                              before RateLimitMiddleware (not an auth endpoint)
    #   RateLimitMiddleware     — per-IP sliding-window limiter on /api/v1/auth/
    #   AuthPathMiddleware      — CSRF origin check + body ceiling + no-store header
    app.add_middleware(AuthPathMiddleware)  # innermost of auth pair
    app.add_middleware(RateLimitMiddleware)  # outermost of auth pair
    app.add_middleware(
        TotpEnforcementMiddleware
    )  # between AuthMiddleware and RateLimitMiddleware
    app.add_middleware(AuthMiddleware)  # resolves request.state.actor
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_exception_handler(NotFoundError, not_found_handler)
    app.add_exception_handler(Exception, server_error_handler)
    # Strip credential inputs from 422 validation errors on auth routes.
    from fastapi.exceptions import RequestValidationError  # noqa: PLC0415

    app.add_exception_handler(RequestValidationError, auth_validation_error_handler)

    # Auth router — always registered (public endpoints + mode-aware).
    app.include_router(
        auth_router.router, prefix=f"{API_V1_PREFIX}/auth", tags=["auth"]
    )
    # Self-service account management; mounted under /auth/ to inherit rate
    # limiting and body-cap middleware applied to that path prefix.
    app.include_router(
        me_router.router, prefix=f"{API_V1_PREFIX}/auth/me", tags=["auth"]
    )
    # Admin management endpoints (users, invites).
    app.include_router(admin.router, prefix=f"{API_V1_PREFIX}/admin", tags=["admin"])

    app.include_router(
        devices.router, prefix=f"{API_V1_PREFIX}/devices", tags=["devices"]
    )
    app.include_router(
        sessions.router, prefix=f"{API_V1_PREFIX}/sessions", tags=["sessions"]
    )
    app.include_router(stats.router, prefix=f"{API_V1_PREFIX}/stats", tags=["stats"])

    app.include_router(
        waveforms.router, prefix=f"{API_V1_PREFIX}/sessions", tags=["waveforms"]
    )
    app.include_router(
        events.router, prefix=f"{API_V1_PREFIX}/sessions", tags=["events"]
    )

    app.include_router(analysis.router, prefix=API_V1_PREFIX, tags=["analysis"])
    app.include_router(days.router, prefix=f"{API_V1_PREFIX}/days", tags=["days"])
    app.include_router(
        health_routes.router, prefix=f"{API_V1_PREFIX}/health", tags=["health"]
    )
    app.include_router(rx.router, prefix=f"{API_V1_PREFIX}/rx", tags=["rx"])
    app.include_router(
        equipment.router, prefix=f"{API_V1_PREFIX}/equipment", tags=["equipment"]
    )

    # /import/detect and /import/path are local-mode-only (server-path import).
    # In multiuser mode these routes are NOT registered — the loopback-peer
    # check is worthless behind Cloudflare; uploads-only is the contract.
    app.include_router(
        import_data.router,
        prefix=f"{API_V1_PREFIX}/import",
        tags=["import"],
    )
    if not is_multiuser:
        app.include_router(
            import_data.local_only_router,
            prefix=f"{API_V1_PREFIX}/import",
            tags=["import"],
        )

    app.include_router(
        reports.router, prefix=f"{API_V1_PREFIX}/reports", tags=["reports"]
    )
    app.include_router(export.router, prefix=f"{API_V1_PREFIX}/export", tags=["export"])

    # /db router: reset is now available in both local and multiuser mode.
    # Multiuser reset accepts include_accounts=false (data-only) or =true (factory reset).
    app.include_router(db.router, prefix=f"{API_V1_PREFIX}/db", tags=["database"])

    app.include_router(
        validation.router, prefix=f"{API_V1_PREFIX}/validate", tags=["validation"]
    )
    app.include_router(
        profiles.router, prefix=f"{API_V1_PREFIX}/profiles", tags=["profiles"]
    )

    app.include_router(about.router, prefix=API_V1_PREFIX, tags=["about"])

    # Excluded from the OpenAPI schema deliberately — keeps the health probe
    # out of generated API clients and avoids ui/src/types/generated.ts churn.
    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # Watchtower pre-update gate — excluded from schema for the same reason.
    # Unauthenticated: the endpoint exposes no user data, only a boolean signal
    # read by the lifecycle hook running inside the same container.  Reasons are
    # computed internally and logged at debug level for operator diagnostics but
    # not included in the response — the hook only reads `busy`, and the reasons
    # categories would leak activity timing on a tunnel-exposed endpoint.
    @app.get("/health/busy", include_in_schema=False)
    async def health_busy() -> dict[str, bool]:
        from snore.api import analysis_jobs as _analysis_jobs  # noqa: PLC0415
        from snore.api import validation_jobs as _validation_jobs  # noqa: PLC0415
        from snore.api.deps import is_reset_locked  # noqa: PLC0415
        from snore.api.import_jobs import has_active_jobs  # noqa: PLC0415

        reasons: list[str] = []
        # Import jobs gate on PENDING_UPLOAD/PENDING/RUNNING — all three states
        # represent in-flight work.  Analysis and validation gate on RUNNING
        # only: QUEUED jobs have no in-progress data writes and are safe to
        # interrupt.  A multi-minute validation run must not be cut off by a
        # watchtower container replacement.
        if has_active_jobs():
            reasons.append("imports")
        if _analysis_jobs.has_running_jobs():
            reasons.append("analysis")
        if _validation_jobs.has_running_jobs():
            reasons.append("validation")
        if is_reset_locked():
            reasons.append("reset")
        logger.debug("health/busy: %s", reasons)
        return {"busy": bool(reasons)}

    _mount_spa(app)

    # Registered after _mount_spa so GZip is the outermost middleware — the SPA
    # fallback registers its own http middleware, and responses it short-circuits
    # (index.html) would bypass compression if GZip sat inside it.
    # (add_middleware is innermost-first, so later calls = outer wrapping.)
    # minimum_size=1000 avoids overhead on tiny payloads.
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Mount the embedded MCP sub-app last — after all API and SPA routes so
    # that FastMCP's catch-all OAuth paths do not shadow /api/* or /assets/*.
    from snore.api.mcp_embed import build_mcp_app  # noqa: PLC0415

    mcp_app = build_mcp_app(cfg)
    if mcp_app is not None:
        app.state.mcp_app = mcp_app
        app.mount("/", mcp_app)

    return app


class _ImmutableStaticFiles(StaticFiles):
    """StaticFiles subclass that stamps Cache-Control: public, max-age=31536000, immutable.

    Safe only for content-hashed assets (e.g. Vite output under /assets) where
    every file URL changes when content changes, making indefinite caching correct.
    """

    async def __call__(
        self,
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        async def send_with_cache_header(message: MutableMapping[str, Any]) -> None:
            if (
                message["type"] == "http.response.start"
                and message.get("status", 0) < 400
            ):
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"cache-control"
                ]
                headers.append(
                    (b"cache-control", b"public, max-age=31536000, immutable")
                )
                message = {**message, "headers": headers}
            await send(message)

        await super().__call__(scope, receive, send_with_cache_header)


def _resolve_spa_dist() -> Path | None:
    dist = Path(str(importlib.resources.files("snore"))) / "ui" / "dist"
    if dist.is_dir():
        return dist
    project_dist = Path(__file__).resolve().parents[3] / "ui" / "dist"
    if project_dist.is_dir():
        return project_dist
    return None


def _mount_spa(app: FastAPI) -> None:
    dist = _resolve_spa_dist()
    if dist is None:
        logger.info("No ui/dist found — SPA not mounted")
        return

    import datetime as _dt

    mtime = (dist / "index.html").stat().st_mtime
    built = _dt.datetime.fromtimestamp(mtime, tz=_dt.UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    logger.info("Serving SPA from %s (built %s)", dist, built)

    app.mount(
        "/assets",
        _ImmutableStaticFiles(directory=dist / "assets"),
        name="spa-assets",
    )

    from snore.api.mcp_embed import is_mcp_path as _is_mcp_path  # noqa: PLC0415

    @app.middleware("http")
    async def spa_fallback(request: Request, call_next: object) -> Response:
        response: Response = await call_next(request)  # type: ignore[operator]
        if response.status_code == 404 and not (
            request.url.path.startswith("/api/") or _is_mcp_path(request.url.path)
        ):
            stat_result = os.stat(dist / "index.html")
            fallback = FileResponse(dist / "index.html", stat_result=stat_result)
            fallback.headers["cache-control"] = "no-cache, must-revalidate"
            if "etag" in fallback.headers:
                fallback.headers["etag"] = "W/" + fallback.headers["etag"]
            return fallback
        return response
