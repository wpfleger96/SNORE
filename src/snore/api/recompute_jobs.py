"""In-memory background job for the admin day recompute.

``POST /api/v1/db/recompute-days`` re-derives every Day row across all users,
profiles and devices (:func:`~snore.services.day_recompute_service.recompute_all_days`,
the same pass as ``snore db recompute-days``).  On a production-sized database
that outlasts Cloudflare's 120-second request limit, so the route enqueues a job
and admins poll ``GET /api/v1/db/recompute-days``.

Built on the same :class:`~snore.api.jobs.core.JobRecordBase`,
:class:`~snore.api.jobs.core.JobStore` and :class:`~snore.api.jobs.pool.WorkerPool`
as :mod:`snore.api.reset_jobs`, with these differences:

- Admission: at most one recompute at a time, and never alongside a database
  reset or data deletion, in either direction
  (:data:`~snore.api.jobs.core.MAINTENANCE_ADMISSION_LOCK`).
- The job id is not a secret: status is admin-only and holds only counts.
- Shutdown cancels a queued job but does not stop a running one between chunks.
  Each chunk commits on its own and recomputation is idempotent, so a job that
  dies with the process loses at most its open chunk and can simply be re-run.
- There is no job-record table.  A restart loses the job.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
import uuid

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from snore.api import reset_jobs
from snore.api.jobs.core import (
    JOB_TTL_SECONDS,
    MAINTENANCE_ADMISSION_LOCK,
    JobRecordBase,
    JobStore,
)
from snore.api.jobs.pool import WorkerPool
from snore.api.schemas import RecomputeDaysJobStatus
from snore.services.day_recompute_service import recompute_all_days
from snore.services.schemas import RecomputeDaysResult

logger = logging.getLogger(__name__)

_FAILURE_MESSAGE = (
    "The day recompute failed. An admin can find details in the server logs."
)


class RecomputeJobState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset(
    {RecomputeJobState.SUCCEEDED, RecomputeJobState.FAILED, RecomputeJobState.CANCELLED}
)


@dataclass(kw_only=True)
class RecomputeDaysJob(JobRecordBase[RecomputeJobState]):
    _TERMINAL_STATES = TERMINAL_STATES

    _state: RecomputeJobState = field(
        default=RecomputeJobState.QUEUED, init=False, repr=False
    )
    _days_processed: int = field(default=0, init=False, repr=False)
    _days_total: int | None = field(default=None, init=False, repr=False)
    _result: RecomputeDaysResult | None = field(default=None, init=False, repr=False)
    _error_message: str | None = field(default=None, init=False, repr=False)
    _finished_at: float | None = field(default=None, init=False, repr=False)

    @property
    def finished_at(self) -> float | None:
        with self._lock:
            return self._finished_at

    def report_progress(self, processed: int, total: int) -> None:
        with self._lock:
            self._days_processed = processed
            self._days_total = total

    def _stamp_finished(self) -> None:
        """Record the terminal timestamps. Caller holds ``_lock``."""
        self._finished_at = time.monotonic()
        self._finished_at_wall = datetime.now(UTC)

    def try_start(self) -> bool:
        """Atomically transition QUEUED → RUNNING, or → CANCELLED if cancel was set."""
        with self._lock:
            if self._cancel_flag:
                self._state = RecomputeJobState.CANCELLED
                self._stamp_finished()
                return False
            self._start_running(RecomputeJobState.RUNNING)
            return True

    def finish(
        self,
        succeeded: bool,
        error_message: str | None = None,
        *,
        result: RecomputeDaysResult | None = None,
    ) -> None:
        """Transition to a terminal state; a pending cancel never changes the outcome."""
        with self._lock:
            if self._state in TERMINAL_STATES:
                return
            if succeeded:
                self._state = RecomputeJobState.SUCCEEDED
                self._result = result
            else:
                self._state = RecomputeJobState.FAILED
                self._error_message = error_message
            self._stamp_finished()

    def _on_cancel_locked(self) -> None:
        """Eager-terminalize a QUEUED job. Caller holds ``_lock`` (see base)."""
        if self._state == RecomputeJobState.QUEUED:
            self._state = RecomputeJobState.CANCELLED
            self._stamp_finished()

    def to_status(self) -> RecomputeDaysJobStatus:
        with self._lock:
            return RecomputeDaysJobStatus(
                job_id=self.job_id,
                state=self._state.value,
                days_processed=self._days_processed,
                days_total=self._days_total,
                error_message=self._error_message,
                result=self._result,
            )


# ---------------------------------------------------------------------------
# Module-level store (aliases bound to the live JobStore objects — never rebind)
# ---------------------------------------------------------------------------

_queue: collections.deque[RecomputeDaysJob] = collections.deque()
_store: JobStore[RecomputeDaysJob] = JobStore()
_all_jobs = _store.jobs
_lock = _store.lock
_condition = threading.Condition(_lock)


def enqueue(*, owner_user_id: int | None) -> RecomputeDaysJob | None:
    """Queue a job, or return None while a recompute, reset or data deletion is active."""
    with MAINTENANCE_ADMISSION_LOCK:
        if reset_jobs.has_active_job():
            return None
        with _condition:
            if any(not job.is_terminal for job in _all_jobs.values()):
                return None
            job = RecomputeDaysJob(job_id=uuid.uuid4().hex, owner_user_id=owner_user_id)
            _queue.append(job)
            _all_jobs[job.job_id] = job
            _condition.notify_all()
    logger.info("Enqueued day recompute job %s", job.job_id)
    return job


def latest_job() -> RecomputeDaysJob | None:
    """The most recently created job still in memory, or None."""
    _reap_terminal()
    return max(_store.snapshot(), key=lambda job: job.created_at, default=None)


def has_active_job() -> bool:
    """True while a job is QUEUED or RUNNING; read by ``/health/busy``."""
    with _lock:
        return any(not job.is_terminal for job in _all_jobs.values())


def _reap_terminal() -> None:
    _store.reap(JOB_TTL_SECONDS, terminal_at=lambda job: job.finished_at)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _execute_job(job: RecomputeDaysJob) -> None:
    if not job.try_start():
        return
    try:
        result = asyncio.run(recompute_all_days(job.report_progress))
    except BaseException:
        # BaseException, not Exception: the worker pool's fallback would copy
        # str(exc) into error_message, which the status endpoint returns.
        logger.exception("Day recompute job %s failed", job.job_id)
        job.finish(succeeded=False, error_message=_FAILURE_MESSAGE)
    else:
        logger.info(
            "Day recompute job %s finished: %d recomputed, %d pruned",
            job.job_id,
            result.days_recomputed,
            result.days_pruned,
        )
        job.finish(succeeded=True, result=result)


_pool: WorkerPool[RecomputeDaysJob] = WorkerPool(
    queue=_queue,
    condition=_condition,
    store=_store,
    execute=lambda job: _execute_job(job),
    concurrency=lambda: 1,
    thread_name_prefix="recompute-job-worker",
    reap=_reap_terminal,
)


def start_worker() -> tuple[list[threading.Thread], threading.Event]:
    """Start the worker thread; return (threads, stop_event)."""
    return _pool.start()


def shutdown(timeout: float = 10.0) -> list[threading.Thread]:
    """Stop the worker and cancel any queued job; a running job finishes or dies.

    Returns any worker threads still alive after ``timeout``.
    """
    return _pool.shutdown(timeout)
