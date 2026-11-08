"""In-memory background job for the admin day recompute.

``POST /api/v1/db/recompute-days`` re-derives every Day row across all users,
profiles and devices (:func:`~snore.services.day_recompute.recompute_all_days`,
the same pass as ``snore db recompute-days``).  On a production-sized database
that outlasts Cloudflare's 120-second request limit, so the route enqueues a job
and admins poll ``GET /api/v1/db/recompute-days``.

Built on the same :class:`~snore.api.jobs.core.SingleShotJobRecord`,
:class:`~snore.api.jobs.core.JobStore` and :class:`~snore.api.jobs.pool.WorkerPool`
as :mod:`snore.api.reset_jobs`, with these differences:

- Admission: at most one recompute at a time, and never alongside another
  maintenance activity (reset, data deletion or VACUUM), in either direction
  (:func:`~snore.api.jobs.core.admit_maintenance`).
- The job id is not a secret: status is admin-only and holds only counts.
- Cancelling a running job (shutdown does) stops it before its next chunk, and
  it ends CANCELLED.  Each chunk commits on its own and recomputation is
  idempotent, so a stopped job keeps the chunks it committed and can simply be
  re-run.
- There is no job-record table.  A restart loses the job.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import uuid

from dataclasses import dataclass, field

from snore.api.jobs.core import (
    JOB_TTL_SECONDS,
    MAINTENANCE_TERMINAL_STATES,
    JobStore,
    MaintenanceJobState,
    SingleShotJobRecord,
    admit_maintenance,
    register_maintenance_job,
)
from snore.api.jobs.pool import WorkerPool
from snore.api.schemas import RecomputeDaysJobStatus
from snore.services.day_recompute import RecomputeStopped, recompute_all_days
from snore.services.schemas import RecomputeDaysResult

logger = logging.getLogger(__name__)

_FAILURE_MESSAGE = (
    "The day recompute failed. An admin can find details in the server logs."
)


@dataclass(kw_only=True)
class RecomputeDaysJob(SingleShotJobRecord):
    _days_processed: int = field(default=0, init=False, repr=False)
    _days_total: int | None = field(default=None, init=False, repr=False)
    _result: RecomputeDaysResult | None = field(default=None, init=False, repr=False)
    _error_message: str | None = field(default=None, init=False, repr=False)

    def report_progress(self, processed: int, total: int) -> None:
        with self._lock:
            self._days_processed = processed
            self._days_total = total

    def finish(
        self,
        succeeded: bool,
        error_message: str | None = None,
        *,
        result: RecomputeDaysResult | None = None,
    ) -> None:
        """Transition to SUCCEEDED or FAILED; a no-op once terminal."""
        with self._lock:
            if self._state in MAINTENANCE_TERMINAL_STATES:
                return
            if succeeded:
                self._state = MaintenanceJobState.SUCCEEDED
                self._result = result
            else:
                self._state = MaintenanceJobState.FAILED
                self._error_message = error_message
            self._stamp_finished()

    def finish_cancelled(self) -> None:
        """End a running job that stopped early on its cancel flag."""
        with self._lock:
            if self._state in MAINTENANCE_TERMINAL_STATES:
                return
            self._state = MaintenanceJobState.CANCELLED
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

_MAINTENANCE_NAME = "recompute"


def enqueue(*, owner_user_id: int | None) -> RecomputeDaysJob | None:
    """Queue a job, or return None while any maintenance activity is underway."""
    with admit_maintenance(_MAINTENANCE_NAME) as admitted:
        if not admitted:
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


register_maintenance_job(_MAINTENANCE_NAME, has_active_job)


def _reap_terminal() -> None:
    _store.reap(JOB_TTL_SECONDS, terminal_at=lambda job: job.finished_at)


# ---------------------------------------------------------------------------
# Job execution
# ---------------------------------------------------------------------------


def _execute_job(job: RecomputeDaysJob) -> None:
    if not job.try_start():
        return
    try:
        result = asyncio.run(
            recompute_all_days(
                job.report_progress, should_stop=lambda: job.cancel_requested
            )
        )
    except RecomputeStopped:
        logger.info("Day recompute job %s stopped on cancel", job.job_id)
        job.finish_cancelled()
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
    """Stop the worker and cancel every job; a running one stops before its next chunk.

    Returns any worker threads still alive after ``timeout``.
    """
    return _pool.shutdown(timeout)
