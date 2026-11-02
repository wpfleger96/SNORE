"""In-memory background job for the database reset and per-user data deletion.

Both operations delete a whole dataset in one SQLite transaction and, if that
removed anything, VACUUM.  On a production-sized database that takes minutes
(``secure_delete`` zeroes every freed page through the WAL, deliberately), which
outlasts Cloudflare's 120-second request limit, so the routes enqueue a job and
clients poll ``GET /api/v1/db/reset-jobs/status``.

Structurally a twin of :mod:`snore.api.validation_jobs` — same
:class:`~snore.api.jobs.core.JobRecordBase` core, same
:class:`~snore.api.jobs.core.JobStore`, same
:class:`~snore.api.jobs.pool.WorkerPool`, same TTL reaping.  Deliberate
divergences:

- There is no job-record table.  State is memory-only: a restart loses the job,
  and the startup vacuum marker and quarantine sweep finish its cleanup.
- At most one job exists at a time.  :func:`enqueue` refuses while any job is
  non-terminal, which is what serializes these two destructive operations.
- QUEUED counts as active for ``/health/busy``: the user has been told their
  data is being deleted, so a container swap must not silently drop the job.
- A pending cancel does not overwrite the outcome.  Once the delete commits the
  data really is gone, so the job reports SUCCEEDED (and delivers a factory
  reset's invite URL) even if shutdown cancelled it in the meantime.
- ``job_id`` is a bearer capability: it reads the one-time admin invite URL from
  the unauthenticated status endpoint.  Never log it, at any level.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import secrets
import threading
import time

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from snore.api.constants import NO_STORE
from snore.api.errors import db_busy_maps_to_409
from snore.api.jobs.core import JOB_TTL_SECONDS, JobRecordBase, JobStore
from snore.api.jobs.pool import WorkerPool
from snore.api.schemas import ResetJobEnqueued
from snore.constants import DEFAULT_VACUUM_PENDING_MARKER
from snore.database.session import session_scope
from snore.database.target import DatabaseTarget
from snore.services.database_service import (
    DatabaseService,
    _vacuum_background,
    file_size_mb,
)
from snore.services.schemas import DeleteDataResult, ResetResult

logger = logging.getLogger(__name__)

_FAILURE_MESSAGE = "The deletion failed. An admin can find details in the server logs."


class ResetJobState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResetJobKind(StrEnum):
    RESET = "reset"
    DELETE_DATA = "delete_data"


TERMINAL_STATES = frozenset(
    {ResetJobState.SUCCEEDED, ResetJobState.FAILED, ResetJobState.CANCELLED}
)


@dataclass(kw_only=True)
class ResetJob(JobRecordBase[ResetJobState]):
    _TERMINAL_STATES = TERMINAL_STATES

    # Re-declared from the base so repr(job) cannot put the bearer token in a log.
    job_id: str = field(repr=False)
    kind: ResetJobKind
    user_id: int
    include_accounts: bool = False
    db_path: str = ""

    _state: ResetJobState = field(default=ResetJobState.QUEUED, init=False, repr=False)
    _error_message: str | None = field(default=None, init=False, repr=False)
    _finished_at: float | None = field(default=None, init=False, repr=False)
    _result: ResetResult | DeleteDataResult | None = field(
        default=None, init=False, repr=False
    )

    @property
    def error_message(self) -> str | None:
        with self._lock:
            return self._error_message

    @property
    def finished_at(self) -> float | None:
        with self._lock:
            return self._finished_at

    @property
    def result(self) -> ResetResult | DeleteDataResult | None:
        """The stored result; not None means the delete committed."""
        with self._lock:
            return self._result

    def set_result(self, result: ResetResult | DeleteDataResult) -> None:
        with self._lock:
            self._result = result

    def _stamp_finished(self) -> None:
        """Record the terminal timestamps. Caller holds ``_lock``."""
        self._finished_at = time.monotonic()
        self._finished_at_wall = datetime.now(UTC)

    def try_start(self) -> bool:
        """Atomically transition QUEUED → RUNNING, or → CANCELLED if cancel was set."""
        with self._lock:
            if self._cancel_flag:
                self._state = ResetJobState.CANCELLED
                self._stamp_finished()
                return False
            self._start_running(ResetJobState.RUNNING)
            return True

    def finish(self, succeeded: bool, error_message: str | None = None) -> None:
        """Transition to a terminal state; a pending cancel never changes the outcome."""
        with self._lock:
            if self._state in TERMINAL_STATES:
                return
            if succeeded:
                self._state = ResetJobState.SUCCEEDED
            else:
                self._state = ResetJobState.FAILED
                self._error_message = error_message
            self._stamp_finished()

    def _on_cancel_locked(self) -> None:
        """Eager-terminalize a QUEUED job. Caller holds ``_lock`` (see base)."""
        if self._state == ResetJobState.QUEUED:
            self._state = ResetJobState.CANCELLED
            self._stamp_finished()

    def to_dict(self) -> dict[str, object]:
        """A ``ResetJobStatus.model_validate`` snapshot.

        The result stays hidden until the job is terminal: it is stored before the
        VACUUM, and ``size_after_mb`` is a placeholder until the job re-measures.
        """
        with self._lock:
            result = self._result if self._state in TERMINAL_STATES else None
            return {
                "kind": self.kind.value,
                "state": self._state.value,
                "error_message": self._error_message,
                "reset_result": result if isinstance(result, ResetResult) else None,
                "delete_data_result": (
                    result if isinstance(result, DeleteDataResult) else None
                ),
            }


# ---------------------------------------------------------------------------
# Module-level store (aliases bound to the live JobStore objects — never rebind)
# ---------------------------------------------------------------------------

_queue: collections.deque[ResetJob] = collections.deque()
_store: JobStore[ResetJob] = JobStore()
_all_jobs = _store.jobs
_lock = _store.lock
_condition = threading.Condition(_lock)


def sqlite_file_path(target: DatabaseTarget) -> str:
    """Return the file VACUUM should run on, or "" for any other kind of target."""
    is_sqlite_file = target.dialect == "sqlite" and target.location not in (
        "",
        ":memory:",
    )
    return target.sqlite_path if is_sqlite_file else ""


def enqueue(
    *,
    kind: ResetJobKind,
    user_id: int,
    db_path: str,
    include_accounts: bool = False,
) -> ResetJob | None:
    """Queue a job, or return None if any job is still queued or running."""
    with _condition:
        if any(not job.is_terminal for job in _all_jobs.values()):
            return None
        job = ResetJob(
            job_id=secrets.token_urlsafe(32),
            kind=kind,
            user_id=user_id,
            include_accounts=include_accounts,
            db_path=db_path,
        )
        _queue.append(job)
        _all_jobs[job.job_id] = job
        _condition.notify_all()
    logger.info("Enqueued %s job for user id=%d", kind.value, user_id)
    return job


def accepted_or_409(job: ResetJob | None) -> JSONResponse:
    """Build the POST response for :func:`enqueue`'s result.

    ``no-store`` because the body carries the job's bearer token.
    """
    if job is None:
        raise HTTPException(
            status_code=409,
            detail="A database reset or data deletion is already in progress",
        )
    enqueued = ResetJobEnqueued.model_validate(
        {"job_id": job.job_id, "state": job.state.value}
    )
    return JSONResponse(
        status_code=202, content=enqueued.model_dump(), headers=NO_STORE
    )


def get_job(job_id: str) -> ResetJob | None:
    return _store.get(job_id)


def has_active_job() -> bool:
    """True while a job is QUEUED or RUNNING; read by ``/health/busy``.

    Stays true through the post-commit VACUUM, which runs inside the job.
    """
    with _lock:
        return any(not job.is_terminal for job in _all_jobs.values())


def _reap_terminal() -> None:
    _store.reap(JOB_TTL_SECONDS, terminal_at=lambda job: job.finished_at)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _raw_root() -> Path:
    from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

    return DEFAULT_RAW_BACKUP_DIR


def _write_vacuum_marker(db_path: str) -> None:
    """Record a pending VACUUM so a restart after the commit reschedules it.

    Best-effort: a marker-write failure is logged but never aborts the job.
    """
    if not db_path:
        return
    try:
        DEFAULT_VACUUM_PENDING_MARKER.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_VACUUM_PENDING_MARKER.write_text(db_path)
    except Exception:
        logger.warning(
            "Could not write vacuum pending marker; VACUUM may not resume after a crash"
        )


def _clear_vacuum_marker() -> None:
    """Remove the marker when no VACUUM is owed.

    Best-effort: a marker left behind only costs a redundant VACUUM at startup.
    """
    try:
        DEFAULT_VACUUM_PENDING_MARKER.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not clear vacuum pending marker: %s", exc)


def _deleted_nothing(result: ResetResult | DeleteDataResult) -> bool:
    if isinstance(result, ResetResult):
        return result.total_rows_deleted == 0
    # Device cascades cover the rest of a user's rows.
    return result.devices_deleted == 0 and result.import_jobs_deleted == 0


async def _delete(job: ResetJob, size_before: float) -> None:
    """Run the delete in one write transaction, which the service commits.

    The result is stored as soon as the service returns, inside the session
    scope, because leaving the scope can still fail after the commit.  A stored
    result therefore means the delete committed.  It reports ``size_before`` as
    its ``size_after_mb`` until :func:`_run` measures the file again.
    """
    from snore.api.config import get_config  # noqa: PLC0415

    public_base_url = get_config().public_base_url or ""
    raw_root = _raw_root()

    async with db_busy_maps_to_409(), session_scope(immediate=True) as db:
        if job.kind is ResetJobKind.RESET:
            tables_cleared, invite_url = await DatabaseService.reset_all_data(
                db,
                include_accounts=job.include_accounts,
                actor_user_id=job.user_id,
                raw_root=raw_root,
                public_base_url=public_base_url,
            )
            job.set_result(
                ResetResult(
                    status="success",
                    tables_cleared=tables_cleared,
                    total_rows_deleted=sum(tables_cleared.values()),
                    size_before_mb=size_before,
                    size_after_mb=size_before,
                    bootstrap_invite_url=invite_url,
                )
            )
            return

        (
            devices_deleted,
            import_jobs_deleted,
            profiles_processed,
        ) = await DatabaseService.delete_user_data(db, job.user_id, raw_root)
        job.set_result(
            DeleteDataResult(
                status="success",
                devices_deleted=devices_deleted,
                import_jobs_deleted=import_jobs_deleted,
                profiles_processed=profiles_processed,
                size_before_mb=size_before,
                size_after_mb=size_before,
            )
        )


def _run(job: ResetJob) -> None:
    size_before = file_size_mb(job.db_path)
    # A marker already here means an earlier VACUUM is still owed, so this job
    # only ever removes a marker it wrote itself.
    owns_marker = not DEFAULT_VACUUM_PENDING_MARKER.exists()
    # Written before the delete because a crash right after the commit needs it.
    _write_vacuum_marker(job.db_path)

    try:
        asyncio.run(_delete(job, size_before))
    except BaseException:
        if job.result is None:
            # Nothing committed, so this job owes no VACUUM.
            if owns_marker:
                _clear_vacuum_marker()
            raise
        logger.exception("Cleanup failed after a committed %s job", job.kind.value)

    result = job.result
    assert result is not None  # Every path that reaches here stored one.

    # The delete is committed, so nothing below may fail the job: a factory
    # reset's invite URL exists nowhere else.  The vacuum marker survives a
    # failed VACUUM, so the next startup retries it.
    try:
        if _deleted_nothing(result):
            if owns_marker:
                _clear_vacuum_marker()
        elif job.db_path:
            _vacuum_background(job.db_path)
        result.size_after_mb = file_size_mb(job.db_path)
    except Exception:
        logger.exception("VACUUM failed after a committed %s job", job.kind.value)


def _execute_job(job: ResetJob) -> None:
    if not job.try_start():
        return
    try:
        _run(job)
    except HTTPException as exc:
        job.finish(succeeded=False, error_message=str(exc.detail))
    except BaseException:
        # BaseException, not Exception: the worker pool's fallback would copy
        # str(exc) into error_message, which reaches an unauthenticated endpoint.
        logger.exception("%s job failed", job.kind.value)
        job.finish(succeeded=False, error_message=_FAILURE_MESSAGE)
    else:
        job.finish(succeeded=True)


_pool: WorkerPool[ResetJob] = WorkerPool(
    queue=_queue,
    condition=_condition,
    store=_store,
    execute=lambda job: _execute_job(job),
    concurrency=lambda: 1,
    thread_name_prefix="reset-job-worker",
    reap=_reap_terminal,
)


def start_worker() -> tuple[list[threading.Thread], threading.Event]:
    """Start the worker thread; return (threads, stop_event)."""
    return _pool.start()


def shutdown(timeout: float = 10.0) -> list[threading.Thread]:
    """Stop the worker and cancel any queued job; a running job finishes or dies.

    A job still running after ``timeout`` dies with the process: its transaction
    rolls back, or if it already committed, startup redoes the VACUUM.  Returns
    any worker threads still alive.
    """
    return _pool.shutdown(timeout)
