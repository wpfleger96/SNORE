"""In-memory FIFO analysis job queue.

By default, 4 admitted analysis jobs run concurrently — sized for a typical
4-5 worker compute pool so CPU is fully utilized without piling unstarted jobs
onto the process pool.  Set ``SNORE_ANALYSIS_JOB_CONCURRENCY`` to override.
Within each job, sessions run with configurable per-job concurrency
(``SNORE_ANALYSIS_MAX_WORKERS``, default 4): the write gate serializes SQLite
writes while the read/compute phases overlap safely.  CPU across all concurrent
jobs is capped globally by the shared ProcessPoolExecutor
(``SNORE_COMPUTE_MAX_WORKERS``).  Jobs are retained for JOB_TTL_SECONDS after
completion for status queries, then reaped.
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
from enum import Enum, StrEnum

from snore.api.jobs.core import (
    JOB_TTL_SECONDS,
    JobRecordBase,
    JobStore,
)
from snore.api.jobs.pool import WorkerPool

logger = logging.getLogger(__name__)

MAX_QUEUED: int = 10

# Per-analysis-job session concurrency.  Mirrors the config default so tests
# that skip full lifespan setup get a sensible fallback without loading config.
_DEFAULT_ANALYSIS_MAX_WORKERS: int = 4

# Default number of concurrent analysis job workers: sized for a ~4-5 worker
# compute pool so the process pool stays saturated without unstarted jobs
# piling up.  Explicit SNORE_ANALYSIS_JOB_CONCURRENCY overrides this value.
_DEFAULT_ANALYSIS_JOB_CONCURRENCY: int = 4


def _get_analysis_workers() -> int:
    """Return analysis_max_workers from config, falling back to the module default."""
    try:
        from snore.api.config import get_config  # noqa: PLC0415

        return get_config().analysis_max_workers
    except Exception:
        return _DEFAULT_ANALYSIS_MAX_WORKERS


def _get_job_concurrency() -> int:
    """Return the number of concurrent analysis job workers.

    Reads ``analysis_job_concurrency`` from config.  ``None`` (the default when
    ``SNORE_ANALYSIS_JOB_CONCURRENCY`` is unset) resolves to the module default
    (4), sized for a typical 4-5 worker compute pool.  A positive integer is
    clamped to ``MAX_QUEUED`` to prevent spawning more workers than the queue
    can ever fill.  Falls back to the module default when config is unavailable.
    """
    try:
        from snore.api.config import get_config  # noqa: PLC0415

        cfg_value = get_config().analysis_job_concurrency
        if cfg_value is None:
            return _DEFAULT_ANALYSIS_JOB_CONCURRENCY
        return min(cfg_value, MAX_QUEUED)
    except Exception:
        return _DEFAULT_ANALYSIS_JOB_CONCURRENCY


class AnalysisJobState(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisJobSource(StrEnum):
    IMPORT = "import"
    BATCH = "batch"


TERMINAL_STATES = frozenset(
    {
        AnalysisJobState.SUCCEEDED,
        AnalysisJobState.FAILED,
        AnalysisJobState.CANCELLED,
    }
)


@dataclass(kw_only=True)
class AnalysisJob(JobRecordBase[AnalysisJobState]):
    _TERMINAL_STATES = TERMINAL_STATES

    profile_id: int
    session_ids: list[int]
    source: AnalysisJobSource
    modes: list[str] | None = None
    primary_mode: str | None = None
    store_results: bool = True

    _state: AnalysisJobState = field(
        default=AnalysisJobState.QUEUED, init=False, repr=False
    )
    _progress_completed: int = field(default=0, init=False, repr=False)
    _progress_total: int = field(default=0, init=False, repr=False)
    _error_message: str | None = field(default=None, init=False, repr=False)
    # Monotonic terminal timestamp (analysis-specific field name; import uses
    # _terminal_at).  Tests assign this directly, so it stays a plain field.
    _finished_at: float | None = field(default=None, init=False, repr=False)

    @property
    def progress_completed(self) -> int:
        with self._lock:
            return self._progress_completed

    @property
    def progress_total(self) -> int:
        with self._lock:
            return self._progress_total

    @property
    def error_message(self) -> str | None:
        with self._lock:
            return self._error_message

    @property
    def finished_at(self) -> float | None:
        with self._lock:
            return self._finished_at

    def try_start(self) -> bool:
        """Atomically transition QUEUED → RUNNING, or QUEUED → CANCELLED if cancel was set.

        Closes the TOCTOU window between dequeue and RUNNING transition.
        Returns True if the job is now RUNNING; False if it was cancelled instead.
        """
        with self._lock:
            if self._cancel_flag:
                self._state = AnalysisJobState.CANCELLED
                self._finished_at = time.monotonic()
                self._finished_at_wall = datetime.now(UTC)
                return False
            self._start_running(AnalysisJobState.RUNNING)
            return True

    def finish(self, succeeded: bool, error_message: str | None = None) -> None:
        """Transition to a terminal state. Respects cancel flag.

        If the cancel flag is set, the job always lands in CANCELLED regardless of
        the ``succeeded`` argument.  Idempotent once already terminal.
        """
        with self._lock:
            if self._state in TERMINAL_STATES:
                return
            if self._cancel_flag:
                self._state = AnalysisJobState.CANCELLED
            elif succeeded:
                self._state = AnalysisJobState.SUCCEEDED
            else:
                self._state = AnalysisJobState.FAILED
                self._error_message = error_message
            self._finished_at = time.monotonic()
            self._finished_at_wall = datetime.now(UTC)

    def _on_cancel_locked(self) -> None:
        """Eager-terminalize a QUEUED job. Caller holds ``_lock`` (see base)."""
        if self._state == AnalysisJobState.QUEUED:
            self._state = AnalysisJobState.CANCELLED
            self._finished_at = time.monotonic()
            self._finished_at_wall = datetime.now(UTC)

    def update_progress(self, done: int, total: int | None) -> None:
        """Update progress counters; safe to call from any thread."""
        with self._lock:
            self._progress_completed = done
            self._progress_total = total or 0

    def to_dict(self) -> dict[str, object]:
        """Return a snapshot of this job's state for ``model_validate``.

        Timestamps are the wall-clock ``datetime`` values (never the internal
        ``time.monotonic()`` values used for ETA and TTL reaping). Returning raw
        datetimes rather than epoch floats removes the monotonic-→-1970 hazard by
        construction: a float cannot inhabit the schema's ``datetime`` field. The
        dict is consumed by ``AnalysisJobStatus.model_validate``, not JSON-dumped.

        eta_seconds is a linear extrapolation: (elapsed / done) * remaining.
        Early estimates can be imprecise; the value is clamped to zero so a
        race where progress_completed momentarily exceeds progress_total never
        produces a negative ETA.
        """
        with self._lock:
            eta: int | None = None
            if (
                self._state == AnalysisJobState.RUNNING
                and self._started_at is not None
                and self._progress_completed > 0
                and self._progress_total > 0
            ):
                elapsed = time.monotonic() - self._started_at
                remaining = max(0, self._progress_total - self._progress_completed)
                eta = round((elapsed / self._progress_completed) * remaining)
            return {
                "job_id": self.job_id,
                "session_count": len(self.session_ids),
                "source": self.source,
                "owner_user_id": self.owner_user_id,
                "state": self._state.value,
                "progress_completed": self._progress_completed,
                "progress_total": self._progress_total,
                "eta_seconds": eta,
                "error_message": self._error_message,
                "created_at": self.created_at_wall,
                "started_at": self._started_at_wall,
                "finished_at": self._finished_at_wall,
            }


# ---------------------------------------------------------------------------
# Module-level store
# ---------------------------------------------------------------------------

_queue: collections.deque[AnalysisJob] = collections.deque()
# _all_jobs and _lock are the store's live objects: tests clear _all_jobs
# directly and the module fuses its condition onto the store lock.
# Alias in place only — never rebind store.jobs/store.lock (see JobStore docstring).
_store: JobStore[AnalysisJob] = JobStore()
_all_jobs = _store.jobs
_lock = _store.lock
_condition = threading.Condition(_lock)


def enqueue(
    profile_id: int,
    session_ids: list[int],
    source: AnalysisJobSource,
    owner_user_id: int | None = None,
    modes: list[str] | None = None,
    primary_mode: str | None = None,
    store_results: bool = True,
) -> AnalysisJob | None:
    """Create a QUEUED job and append it to the queue.

    Returns None if the queue is already at MAX_QUEUED.
    """
    with _condition:
        if len(_queue) >= MAX_QUEUED:
            return None
        job = AnalysisJob(
            job_id=uuid.uuid4().hex,
            profile_id=profile_id,
            session_ids=list(session_ids),
            source=source,
            owner_user_id=owner_user_id,
            modes=modes,
            primary_mode=primary_mode,
            store_results=store_results,
        )
        _queue.append(job)
        _all_jobs[job.job_id] = job
        _condition.notify_all()
    logger.debug(
        "Enqueued analysis job %s (source=%s, sessions=%d)",
        job.job_id,
        source,
        len(session_ids),
    )
    return job


def get_job(job_id: str) -> AnalysisJob | None:
    return _store.get(job_id)


def list_jobs(owner_user_id: int | None = None) -> list[AnalysisJob]:
    """Return jobs visible to *owner_user_id*.

    A job with owner_user_id=None is visible to any caller (local-mode parity).
    A job with a set owner is visible only to that owner.
    When *owner_user_id* is None the caller receives all jobs.
    """
    return _store.list_visible_to(owner_user_id)


def cancel_job(job_id: str) -> bool:
    """Set cancel flag. If QUEUED, immediately transitions to CANCELLED.

    Returns True if the job was found and not already terminal.
    """
    with _condition:
        job = _all_jobs.get(job_id)
        if job is None:
            return False
        result = job.try_cancel()
        if result:
            try:
                _queue.remove(job)
            except ValueError:
                pass
    return result


def in_flight_session_ids(profile_id: int) -> set[int]:
    """Session ids that a QUEUED or RUNNING job for *profile_id* will analyze.

    Any owner's job counts; a job with cancellation requested does not.
    """
    with _lock:
        return {
            sid
            for job in _all_jobs.values()
            if job.profile_id == profile_id
            and not job.is_terminal
            and not job.cancel_requested
            for sid in job.session_ids
        }


def has_running_jobs() -> bool:
    """Return True if any analysis job is currently in the RUNNING state.

    Used by the /health/busy endpoint to gate watchtower container replacement.
    QUEUED jobs are excluded: they have no in-progress data writes yet, so
    interrupting them is safe.  The sessions remain committed in the database
    and analysis re-triggers on the next import or manual batch run.
    """
    with _lock:
        return any(j.state == AnalysisJobState.RUNNING for j in _all_jobs.values())


# Reaper pattern: reaped inline inside the worker loop — after each job completes
# and on idle timeouts.  The analysis store is accessed only by the worker thread
# and a low-frequency list endpoint, so the eager-reap-on-read approach used by
# import_jobs (which also runs a dedicated background reaper thread) is unnecessary
# here.  import_jobs needs eager reap because HTTP handlers poll it on every
# GET /import/{id}/progress and GET /import/jobs request.


def _reap_terminal() -> None:
    removed = _store.reap(JOB_TTL_SECONDS, terminal_at=lambda job: job.finished_at)
    for jid in removed:
        logger.debug("Reaped terminal analysis job %s", jid)


async def _upsert_analysis_record(job: AnalysisJob) -> None:
    """Upsert the current job state to the database for crash-recovery durability."""
    from snore.api.jobs.durability import upsert_job_record  # noqa: PLC0415
    from snore.database.models import AnalysisJobRecord  # noqa: PLC0415

    now = datetime.now(UTC)
    finished = job.finished_at_wall if job.is_terminal else None

    values = {
        "job_id": job.job_id,
        "source": job.source.value,
        "profile_id": job.profile_id,
        "owner_user_id": job.owner_user_id,
        "session_ids_json": job.session_ids,
        "modes": job.modes,
        "primary_mode": job.primary_mode,
        "store_results": job.store_results,
        "state": job.state.value,
        "progress_completed": job.progress_completed,
        "progress_total": job.progress_total,
        "error_message": job.error_message,
        "created_at": job.created_at_wall,
        "started_at": job.started_at_wall,
        "finished_at": finished,
        "updated_at": now,
    }
    await upsert_job_record(
        AnalysisJobRecord,
        values=values,
        update_fields=[
            "state",
            "progress_completed",
            "progress_total",
            "error_message",
            "started_at",
            "finished_at",
            "updated_at",
        ],
    )


async def _run_analysis(job: AnalysisJob) -> None:
    from snore.services.analysis_facade import AnalysisFacade  # noqa: PLC0415

    # No DB session needed here — run_batch_analysis opens its own short-lived
    # scopes for the id-list query and for each per-session read/write phase.
    # No DB transaction may remain open across the batch loop: a batch-long read
    # snapshot pins the WAL and starves checkpoints.
    facade = AnalysisFacade(None, profile_id=job.profile_id)
    await facade.run_batch_analysis(
        session_ids=job.session_ids,
        modes=job.modes,
        primary_mode=job.primary_mode,
        store_results=job.store_results,
        progress_callback=job.update_progress,
        cancel_predicate=lambda: job.cancel_requested,
        max_workers=_get_analysis_workers(),
    )


def _execute_job(job: AnalysisJob) -> None:
    if not job.try_start():
        return
    try:
        asyncio.run(_upsert_analysis_record(job))
    except Exception:
        logger.exception(
            "Failed to persist RUNNING state for analysis job %s", job.job_id
        )
    try:
        asyncio.run(_run_analysis(job))
        job.finish(succeeded=True)
    except Exception as exc:
        logger.exception("Analysis job %s failed", job.job_id)
        job.finish(succeeded=False, error_message=str(exc))
    finally:
        try:
            asyncio.run(_upsert_analysis_record(job))
        except Exception:
            logger.exception(
                "Failed to persist terminal state for analysis job %s", job.job_id
            )


# The worker-pool layer (throttled reaping, per-worker loop wiring, restartable
# start/shutdown thread bookkeeping) is shared with validation_jobs.  ``execute``
# and ``concurrency`` are wrapped so a test patching the module-level
# ``_execute_job`` / ``_get_job_concurrency`` is honoured at call time.
_pool: WorkerPool[AnalysisJob] = WorkerPool(
    queue=_queue,
    condition=_condition,
    store=_store,
    execute=lambda job: _execute_job(job),
    concurrency=lambda: _get_job_concurrency(),
    thread_name_prefix="analysis-job-worker",
    reap=_reap_terminal,
)


def start_worker() -> tuple[list[threading.Thread], threading.Event]:
    """Start N persistent worker threads (N from config, default 4).

    Returns (threads, stop_event) so the caller can shut them down.
    """
    return _pool.start()


def shutdown(timeout: float = 10.0) -> None:
    """Signal all workers to stop; cancel all queued and running jobs.

    Stops every worker generation ever started in this process, then drops
    references to exited threads. Threads that outlive ``timeout`` stay
    registered so a later call can retry.
    """
    _pool.shutdown(timeout)
