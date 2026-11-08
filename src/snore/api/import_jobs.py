"""In-memory job store for streaming import progress via SSE.

Uploads stage concurrently; execution is serialised through a single persistent
worker thread (single SQLite writer — mirrors the analysis queue).

State machine
-------------
::

    reserve()             → PENDING_UPLOAD (admission slot taken *before* body is read)
    convert_to_job()      → PENDING       (reservation becomes a real job after parsing)
    enqueue_for_execution → PENDING job appended to FIFO; worker picks it up next
    GET /{id}/progress    → attaches an SSE observer; never starts or restarts the worker
    DELETE /{id}          → cancelled (idempotent after any terminal state)
    worker finishes       → succeeded | failed
    reaper                → removes terminal jobs after TTL

Terminal states: succeeded, failed, cancelled.
Active states:   pending_upload, pending, running.

Admission
---------
Per-user and global caps include PENDING_UPLOAD reservations + PENDING + RUNNING jobs
(one counter, one state machine).  An over-limit request gets 429 *before* any body
bytes are consumed.  The reservation converts atomically to the job after parsing.
At no instant does the pair double-count (both reservation + job) or drop the slot.

Resource ownership
------------------
The slot owns the disk it admitted.  Capacity is released **only after** temp/spool
cleanup completes on every terminal and error path.  The terminal job record is
retained for SSE observation independently of capacity.

Guarantees
----------
- Serial execution: jobs run one at a time through a single persistent worker thread,
  preventing SQLite write-lock contention between concurrent imports.
- Start-once: the worker transitions a job PENDING → RUNNING exactly once.
- Fan-out: each observer has its own capacity-one/coalescing channel backed by the
  latest-progress snapshot.  Terminal delivery is never dropped.
- Late observers: connecting after terminal state immediately receive the terminal event.
- Reaper: removes terminal jobs only; active jobs are never reaped regardless of age.
- Shutdown: ``shutdown()`` cancels all non-terminal jobs, drains the FIFO, and awaits
  the worker thread.
"""

from __future__ import annotations

import collections
import logging
import shutil
import threading
import time
import uuid

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from snore.api.jobs.core import (
    JOB_TTL_SECONDS,
    JobRecordBase,
    JobStore,
    run_worker_loop,
)

logger = logging.getLogger(__name__)

# JOB_TTL_SECONDS is defined in jobs.core and re-exported here (imported above)
# so that existing ``import_jobs.JOB_TTL_SECONDS`` references keep resolving.

# How long a PENDING_UPLOAD job may be idle (no chunk received) before the
# reaper treats it as abandoned and releases its admission slot.
PENDING_UPLOAD_TIMEOUT_SECONDS: float = 1800.0

# Per-user and global admission caps.
# These are read from the app config at call time so tests can override via env.
# The module-level defaults mirror config.py's env-var defaults (SNORE_MAX_JOBS_PER_USER=3,
# SNORE_MAX_JOBS_GLOBAL=10) and are used as fallbacks when config is not yet loaded
# (e.g. unit tests that exercise the job store without a full lifespan setup).
_DEFAULT_MAX_ACTIVE_PER_USER: int = 3
_DEFAULT_MAX_ACTIVE_GLOBAL: int = 10


def _get_caps() -> tuple[int, int]:
    """Return (max_per_user, max_global) from config, falling back to defaults."""
    try:
        from snore.api.config import get_config  # noqa: PLC0415

        cfg = get_config()
        return cfg.max_jobs_per_user, cfg.max_jobs_global
    except Exception:
        return _DEFAULT_MAX_ACTIVE_PER_USER, _DEFAULT_MAX_ACTIVE_GLOBAL


class JobState(Enum):
    """All states a job can occupy."""

    PENDING_UPLOAD = "pending_upload"  # Admission slot reserved; body not yet parsed.
    PENDING = "pending"  # Files received; worker not yet started.
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})
ACTIVE_STATES = frozenset({JobState.PENDING_UPLOAD, JobState.PENDING, JobState.RUNNING})


class JobType(Enum):
    UPLOAD = "upload"
    # PATH is historical-only: the server-path import feature was removed.
    # This member is retained so existing DB rows with job_type='path' can be
    # surfaced by in-memory jobs without a ValueError on enum lookup.
    PATH = "path"
    HEALTH_UPLOAD = "health_upload"
    RESCAN = "rescan"


class JobPhase(StrEnum):
    """Non-terminal phases within a RUNNING job.

    Delivered via phase_complete() as a live milestone to attached observers.
    NOT a terminal state — the job stays RUNNING through the import phase.
    """

    IMPORT = "import"


# ---------------------------------------------------------------------------
# Per-observer coalescing channel
# ---------------------------------------------------------------------------


class ObserverChannel:
    """Capacity-one coalescing notification channel for a single SSE observer.

    Each observer holds exactly one pending message slot.  A new message
    overwrites an un-consumed one (coalescing), *except* when the pending
    message is already a terminal event — terminal events are never dropped.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._slot: dict[str, Any] | None = None
        self._closed = False

    def put(self, msg: dict[str, Any]) -> None:
        """Deliver msg to this observer, coalescing non-terminal messages."""
        with self._cond:
            if self._slot is not None and self._slot.get("event") in (
                "complete",
                "error",
            ):
                return
            self._slot = msg
            self._cond.notify_all()

    def get(self, timeout: float = 1.0) -> dict[str, Any] | None:
        """Block up to *timeout* seconds and return the next message, or None."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._slot is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            if self._closed and self._slot is None:
                return None
            msg = self._slot
            self._slot = None
            return msg

    def close(self) -> None:
        """Unblock any waiting get() call with a None return."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# ImportJob
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class ImportJob(JobRecordBase[JobState]):
    """Represents a single import operation with full lifecycle management."""

    _TERMINAL_STATES = TERMINAL_STATES

    job_type: JobType
    target_profile_id: int | None = None  # The profile data lands in.

    # UPLOAD jobs: temp dir with written files.
    temp_dir: Path | None = None

    # RESCAN options, fixed before the job is enqueued and never persisted:
    # rescans are not resumed after a restart.  ``force`` deletes and recreates
    # sessions that already exist.
    force: bool = False
    date_from: date | None = None
    date_to: date | None = None
    analyze: bool = True

    # File count — protected by _lock (written by the request handler before the worker starts).
    _file_count: int = field(default=0, init=False, repr=False)

    # State machine fields — protected by _lock.
    _state: JobState = field(default=JobState.PENDING_UPLOAD, init=False, repr=False)
    _terminal_msg: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _latest_progress: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _worker_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _observers: list[ObserverChannel] = field(
        default_factory=list, init=False, repr=False
    )
    # Monotonic terminal timestamp (import-specific field name; analysis uses
    # _finished_at).  Tests assign this directly, so it stays a plain field.
    _terminal_at: float | None = field(default=None, init=False, repr=False)
    # True while the slot still counts against admission caps.
    _capacity_held: bool = field(default=True, init=False, repr=False)
    # Wall-clock timestamp of last upload activity (initialised to created_at_wall).
    # Updated by touch() on each continuation chunk to prevent the stale-upload
    # reaper from reclaiming a slow-but-active upload.
    _last_activity_wall: datetime = field(
        default_factory=lambda: datetime.now(UTC), init=False, repr=False
    )
    # Import-phase durability fields — set after import commits, before analysis.
    # Carried in EVERY terminal payload produced after import_committed=True
    # (success, analysis failure, and cancellation) so that late observers and
    # stalled observers always learn that data landed — a committed import
    # must never be reported as lost.
    _import_committed: bool = field(default=False, init=False, repr=False)
    _import_result: dict[str, Any] | None = field(default=None, init=False, repr=False)
    # Analysis linkage — set by the worker thread via set_analysis_link() after
    # the enqueue attempt and BEFORE _finish().  The list endpoint stitches from
    # these fields; the SSE terminal payload carries the same info independently.
    _analysis_job_id: str | None = field(default=None, init=False, repr=False)
    _analysis_queued: bool | None = field(default=None, init=False, repr=False)

    @property
    def import_committed(self) -> bool:
        """True after the import phase has committed to the database."""
        with self._lock:
            return self._import_committed

    @property
    def finished_at(self) -> float | None:
        """Monotonic timestamp when the job reached a terminal state, or None."""
        with self._lock:
            return self._terminal_at

    @property
    def analysis_job_id(self) -> str | None:
        """ID of the downstream analysis job, or None if not yet set."""
        with self._lock:
            return self._analysis_job_id

    @property
    def analysis_queued(self) -> bool | None:
        """True when analysis was enqueued; False when the queue was full; None when unknown."""
        with self._lock:
            return self._analysis_queued

    @property
    def file_count(self) -> int:
        """Number of files submitted with this job."""
        with self._lock:
            return self._file_count

    def set_file_count(self, n: int) -> None:
        """Record the number of files submitted with this job."""
        with self._lock:
            self._file_count = n

    def touch(self) -> None:
        """Reset last-activity wall-clock timestamp to now.

        Called on each continuation-chunk receipt to prevent the stale-upload
        reaper from reclaiming a slow-but-active multi-chunk upload.
        """
        with self._lock:
            self._last_activity_wall = datetime.now(UTC)

    @property
    def latest_progress_message(self) -> str | None:
        """The most recent progress message text, or None."""
        with self._lock:
            if self._latest_progress is None:
                return None
            data = self._latest_progress.get("data")
            if not isinstance(data, dict):
                return None
            return data.get("message")

    @property
    def error_message(self) -> str | None:
        """The error message from the terminal payload, only for FAILED or CANCELLED jobs."""
        with self._lock:
            if self._state not in (JobState.FAILED, JobState.CANCELLED):
                return None
            if self._terminal_msg is None:
                return None
            data = self._terminal_msg.get("data")
            if not isinstance(data, dict):
                return None
            return data.get("message")

    @property
    def sessions_imported(self) -> int | None:
        """Total sessions committed to the DB, or None if import has not committed."""
        with self._lock:
            if not self._import_committed or self._import_result is None:
                return None
            return self._import_result.get("total_imported")

    @property
    def import_result_snapshot(self) -> dict[str, Any] | None:
        """Shallow copy of the import result dict, or None if not yet committed.

        Returns a SHALLOW copy — top-level dict copied, nested lists/dicts shared;
        callers must treat it as read-only (deliberate perf choice).
        """
        with self._lock:
            if not self._import_committed or self._import_result is None:
                return None
            return dict(self._import_result)

    def set_analysis_link(
        self,
        *,
        analysis_job_id: str | None,
        queue_full: bool,
    ) -> None:
        """Record the outcome of the analysis enqueue attempt.

        ``queue_full`` is True when the enqueue attempt was rejected because the
        analysis queue was full (not when the queued job later failed).

        Called by the worker thread after the enqueue attempt and BEFORE _finish() —
        the list endpoint stitches from these fields; the SSE terminal payload carries
        the same info independently.
        """
        with self._lock:
            self._analysis_job_id = analysis_job_id
            if analysis_job_id is not None:
                self._analysis_queued = True
            elif queue_full:
                self._analysis_queued = False
            # else leave _analysis_queued as None (no sessions imported, no attempt)

    def convert_to_pending(self) -> bool:
        """Transition PENDING_UPLOAD → PENDING (files received, body parsed).

        This is the atomic reservation→job conversion: at no instant does the
        counter both double-count (reservation + job) or drop the slot.
        """
        with self._lock:
            if self._state != JobState.PENDING_UPLOAD:
                return False
            self._state = JobState.PENDING
            return True

    def attach_observer(self) -> ObserverChannel:
        """Register a new SSE observer.

        If the job is already terminal, the channel is pre-loaded with the
        terminal event so the observer receives it immediately.
        """
        ch = ObserverChannel()
        with self._lock:
            if self._state in TERMINAL_STATES:
                if self._terminal_msg is not None:
                    ch.put(self._terminal_msg)
            else:
                if self._latest_progress is not None:
                    ch.put(self._latest_progress)
                self._observers.append(ch)
        return ch

    def detach_observer(self, ch: ObserverChannel) -> None:
        """Remove an observer (SSE connection closed)."""
        with self._lock:
            try:
                self._observers.remove(ch)
            except ValueError:
                pass
        ch.close()

    def _broadcast(self, msg: dict[str, Any]) -> None:
        """Deliver msg to all attached observers. Must be called without _lock."""
        with self._lock:
            observers = list(self._observers)
        for ch in observers:
            ch.put(msg)

    def report_progress(self, message: str) -> None:
        """Record a progress snapshot and broadcast to all observers."""
        msg = {"event": "progress", "data": {"message": message}}
        with self._lock:
            self._latest_progress = msg
        self._broadcast(msg)

    def phase_complete(self, phase: JobPhase, import_result: dict[str, Any]) -> None:
        """Record import-phase completion and broadcast a non-terminal milestone.

        This is NOT a terminal state — the job stays RUNNING.  The event is
        coalesced like any progress message: late/stalled observers may miss it.
        The import outcome is persisted on the job itself so that every subsequent
        terminal payload (success, analysis failure, or cancellation) carries it —
        this is the durability guarantee: a committed import must never be
        reported as lost, no matter which terminal state the job reaches.

        Must be called by the worker thread only, after the import transaction
        commits and BEFORE the analysis phase begins.
        """
        with self._lock:
            self._import_committed = True
            self._import_result = import_result
        milestone = {
            "event": "phase_complete",
            "data": {
                "phase": phase.value,
                "import_result": import_result,
            },
        }
        self._broadcast(milestone)

    def try_start(self) -> bool:
        """Transition pending → running."""
        with self._lock:
            if self._state != JobState.PENDING:
                return False
            self._start_running(JobState.RUNNING)
            return True

    def _make_cancel_payload(self) -> dict[str, Any]:
        """Build the cancelled terminal SSE payload. Caller must hold self._lock."""
        data: dict[str, Any] = {"message": "Cancelled"}
        if self._import_committed and self._import_result is not None:
            data["import_committed"] = True
            data["import_result"] = self._import_result
        return {"event": "error", "data": data}

    def _on_cancel_locked(self) -> tuple[list[ObserverChannel], dict[str, Any]] | None:
        """Eager-terminalize from PENDING_UPLOAD/PENDING. Caller holds ``_lock``.

        Returns ``(observers, terminal_msg)`` to deliver outside the lock, or
        None when the job was RUNNING (cancellation is cooperative there).
        """
        if self._state in (JobState.PENDING_UPLOAD, JobState.PENDING):
            self._state = JobState.CANCELLED
            terminal_msg = self._make_cancel_payload()
            self._terminal_msg = terminal_msg
            self._terminal_at = time.monotonic()
            self._finished_at_wall = datetime.now(UTC)
            return list(self._observers), terminal_msg
        return None

    def _notify_after_cancel(
        self, payload: tuple[list[ObserverChannel], dict[str, Any]]
    ) -> None:
        observers, terminal_msg = payload
        for ch in observers:
            ch.put(terminal_msg)

    def try_cancel_if_stale_upload(self, timeout_seconds: float) -> float | None:
        """Atomically cancel only if this job is still PENDING_UPLOAD and idle.

        Both the state check and the idle-duration check are performed under a
        single lock acquisition so a concurrent ``convert_to_pending()`` (final
        chunk success, client already received HTTP 200) can never be reaped:
        once the state transitions to PENDING the lock-held state check fails
        and this method returns None without touching the job.

        Returns the idle duration in seconds when the job was in PENDING_UPLOAD,
        was idle beyond *timeout_seconds*, and was atomically transitioned to
        CANCELLED.  Returns None otherwise — including when the job was already
        CANCELLED or had converted to PENDING before this call ran.

        Returning the idle seconds (computed under the lock) lets the caller log
        an accurate duration without re-reading ``_last_activity_wall`` outside
        the lock.
        """
        observers: list[ObserverChannel] = []
        terminal_msg: dict[str, Any]
        with self._lock:
            if self._state != JobState.PENDING_UPLOAD:
                return None
            idle = (datetime.now(UTC) - self._last_activity_wall).total_seconds()
            if idle <= timeout_seconds:
                return None
            # Both conditions satisfied — perform cancel transition atomically.
            self._cancel_flag = True
            self._state = JobState.CANCELLED
            terminal_msg = self._make_cancel_payload()
            self._terminal_msg = terminal_msg
            self._terminal_at = time.monotonic()
            self._finished_at_wall = datetime.now(UTC)
            observers = list(self._observers)
        for ch in observers:
            ch.put(terminal_msg)
        return idle

    def _finish(
        self, *, succeeded: bool, terminal_msg: dict[str, Any] | None = None
    ) -> bool:
        """Transition running → succeeded/failed/cancelled. Called by the worker.

        When ``_cancel_flag`` is set the passed ``terminal_msg`` is ignored:
        ``_make_cancel_payload()`` always produces the cancel terminal event.
        ``terminal_msg`` is required (non-None) on all non-cancel paths.

        NOTE: The caller must call release_capacity() AFTER cleanup_files()
        completes to ensure the slot owns the disk it admitted.
        """
        with self._lock:
            if self._state in TERMINAL_STATES:
                return False
            if self._cancel_flag:
                self._state = JobState.CANCELLED
                terminal_msg = self._make_cancel_payload()
            else:
                if terminal_msg is None:
                    raise RuntimeError(
                        "_finish: terminal_msg required on non-cancel paths"
                    )
                self._state = JobState.SUCCEEDED if succeeded else JobState.FAILED
                # Embed import outcome in every terminal payload after import committed.
                if self._import_committed and self._import_result is not None:
                    data = terminal_msg.get("data", {})
                    if isinstance(data, dict):
                        data["import_committed"] = True
                        data["import_result"] = self._import_result
                        terminal_msg = dict(terminal_msg)
                        terminal_msg["data"] = data
            self._terminal_msg = terminal_msg
            self._terminal_at = time.monotonic()
            self._finished_at_wall = datetime.now(UTC)
            observers = list(self._observers)
            self._observers.clear()
        for ch in observers:
            ch.put(terminal_msg)
        return True

    def release_capacity(self) -> None:
        """Release the admission slot AFTER cleanup completes.

        Must be called AFTER cleanup_files() on every terminal/error path.
        The slot owns the disk it admitted; releasing early would allow a new
        request to reserve and spool while the prior job's temp tree still exists.
        """
        with self._lock:
            if not self._capacity_held:
                return
            self._capacity_held = False
        # Decrement the store counter outside the job lock to avoid lock ordering issues.
        _decrement_capacity(self.owner_user_id)

    def wait_for_worker(self, timeout: float = 5.0) -> None:
        """Block until the worker thread exits (used during shutdown)."""
        with self._lock:
            t = self._worker_thread
        if t is not None:
            t.join(timeout=timeout)

    def cleanup_files(self) -> None:
        """Remove the temp directory if this is an UPLOAD job."""
        if self.temp_dir is not None and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Job store and admission counters
# ---------------------------------------------------------------------------

# _jobs and _lock are the store's live objects: app.py startup-resume and tests
# mutate _jobs directly, so both names must alias the store-visible dict/lock.
# Alias in place only — never rebind store.jobs/store.lock (see JobStore docstring).
_store: JobStore[ImportJob] = JobStore()
_jobs = _store.jobs
_lock = _store.lock

# Admission counters: per-user active count and global active count.
# "Active" = PENDING_UPLOAD + PENDING + RUNNING (capacity_held=True).
_per_user_count: dict[int | None, int] = {}
_global_count: int = 0
_counts_lock = threading.Lock()


def _decrement_capacity(owner_user_id: int | None) -> None:
    """Decrement both per-user and global counts for one released slot."""
    global _global_count
    with _counts_lock:
        current = _per_user_count.get(owner_user_id, 0)
        _per_user_count[owner_user_id] = max(0, current - 1)
        _global_count = max(0, _global_count - 1)


def _check_and_reserve(owner_user_id: int | None) -> bool:
    """Atomically check caps and increment counters.

    Returns True if the reservation was taken (within caps), False if over-limit.
    """
    global _global_count
    max_per_user, max_global = _get_caps()
    with _counts_lock:
        user_count = _per_user_count.get(owner_user_id, 0)
        if user_count >= max_per_user:
            return False
        if _global_count >= max_global:
            return False
        _per_user_count[owner_user_id] = user_count + 1
        _global_count += 1
        return True


def has_active_jobs() -> bool:
    """Return True if any import job is in an active (non-terminal) state.

    Active states are PENDING_UPLOAD, PENDING, and RUNNING.  Used by the
    /health/busy endpoint to gate watchtower container replacement.
    """
    with _lock:
        return any(j.state in ACTIVE_STATES for j in _jobs.values())


def reserve_slot(
    owner_user_id: int | None,
    *,
    job_type: JobType = JobType.UPLOAD,
) -> ImportJob | None:
    """Atomically check caps and create a PENDING_UPLOAD reservation.

    Must be called BEFORE reading any body bytes.

    Args:
        owner_user_id: The user who will own this job.
        job_type: The type of import job (default: UPLOAD for CPAP uploads).

    Returns:
        A new ImportJob in PENDING_UPLOAD state, or None if over-limit (429).
    """
    if not _check_and_reserve(owner_user_id):
        return None
    _reap_terminal()
    job = ImportJob(
        job_id=uuid.uuid4().hex,
        job_type=job_type,
        owner_user_id=owner_user_id,
    )
    with _lock:
        _jobs[job.job_id] = job
    logger.debug(
        "admission: reserved slot for user %s (job %s)", owner_user_id, job.job_id
    )
    return job


def create_job(
    job_type: JobType, owner_user_id: int | None = None, **kwargs: Any
) -> ImportJob:
    """Create a job directly in PENDING state, bypassing the PENDING_UPLOAD phase.

    Also increments caps.  If over-limit, raises RuntimeError.
    """
    if not _check_and_reserve(owner_user_id):
        raise RuntimeError("Admission caps exceeded")
    _reap_terminal()
    job = ImportJob(
        job_id=uuid.uuid4().hex,
        job_type=job_type,
        owner_user_id=owner_user_id,
        **kwargs,
    )
    # PATH jobs start directly in PENDING (no upload phase).
    job._state = JobState.PENDING
    with _lock:
        _jobs[job.job_id] = job
    return job


class ForcedRescanActiveError(Exception):
    """A forced rescan of the target profile is already queued or running."""


# Serializes forced-rescan admission so two requests cannot both pass the check.
_rescan_admission_lock = threading.Lock()


def create_rescan_job(
    *,
    owner_user_id: int | None,
    target_profile_id: int,
    force: bool,
    date_from: date | None,
    date_to: date | None,
    analyze: bool,
) -> ImportJob:
    """Create a PENDING rescan job for *target_profile_id*.

    At most one forced rescan per profile is queued or running at a time: a
    second would delete and recreate the same sessions the first is replacing.
    Raises :class:`ForcedRescanActiveError` for that, and RuntimeError (as
    :func:`create_job`) when admission caps are exceeded.
    """
    with _rescan_admission_lock:
        if force and any(
            job.job_type is JobType.RESCAN
            and job.force
            and job.target_profile_id == target_profile_id
            and not job.is_terminal
            for job in _store.snapshot()
        ):
            raise ForcedRescanActiveError
        return create_job(
            JobType.RESCAN,
            owner_user_id=owner_user_id,
            target_profile_id=target_profile_id,
            force=force,
            date_from=date_from,
            date_to=date_to,
            analyze=analyze,
        )


def get_job(job_id: str) -> ImportJob | None:
    _reap_terminal()
    return _store.get(job_id)


def list_jobs(owner_user_id: int | None = None) -> list[ImportJob]:
    """Return jobs visible to *owner_user_id*, reaping expired terminal jobs first.

    A job with owner_user_id=None is visible to any caller (local-mode parity).
    A job with a set owner is visible only to that owner.
    When *owner_user_id* is None the caller receives all jobs.
    """
    _reap_terminal()
    return _store.list_visible_to(owner_user_id)


def remove_job(job_id: str) -> None:
    _store.remove(job_id)


def get_live_spool_dirs() -> frozenset[Path]:
    """Return the spool directories of all in-flight upload jobs.

    Used by the periodic spool sweep to skip directories that belong to
    active uploads.  Chunk writes land in subdirectories and do not bump
    the top-level spool dir mtime, so mtime alone is not a safe liveness
    signal — callers must skip any path returned here.
    """
    with _lock:
        return frozenset(
            job.temp_dir for job in _jobs.values() if job.temp_dir is not None
        )


def cancel_job(job_id: str) -> bool:
    """Cancel a job. Returns True if the job existed and was not already terminal.

    Flipping job state via the store is sufficient — no queue eviction needed.
    ImportJob.try_cancel eagerly terminalizes a still-queued job, and the worker
    loop's try_start gate skips (and cleans up) any terminal entry when it is
    later dequeued, so leaving the queued reference in the FIFO is harmless.
    """
    return _store.cancel(job_id)


def shutdown(timeout: float = 10.0) -> list[str]:
    """Cancel all non-terminal jobs and wait for the worker thread to exit.

    Queued PENDING jobs are drained from the FIFO, cancelled, and have their
    staged files cleaned and capacity released here (run_callback never ran
    for them).  The currently RUNNING job (if any) is cancelled cooperatively
    and its cleanup is handled by the run_callback's finally block.
    """
    global _import_worker_thread, _import_stop_event, _shutdown_in_progress
    _shutdown_in_progress = True

    # Signal the worker to stop after finishing the current job (if any).
    if _import_stop_event is not None:
        _import_stop_event.set()

    # Drain the execution queue atomically so the worker cannot pick up more jobs.
    drained: list[tuple[ImportJob, Path | None]] = []
    with _import_condition:
        drained = list(_import_queue)
        _import_queue.clear()
        _import_condition.notify_all()

    # Cancel drained PENDING jobs.  Spool files are preserved during shutdown
    # so a subsequent server start can resume them.
    for job, _ in drained:
        job.try_cancel()
        job.release_capacity()

    # Cancel any job that is still non-terminal (e.g. currently RUNNING or a
    # PENDING_UPLOAD job whose upload handler is still writing).  try_cancel()
    # is idempotent so calling it again on an already-cancelled drained job is safe.
    with _lock:
        remaining = [j for j in _jobs.values() if not j.is_terminal]
    for job in remaining:
        job.try_cancel()

    # Wait for the single worker thread to exit.
    still_alive: list[str] = []
    if _import_worker_thread is not None:
        _import_worker_thread.join(timeout=timeout)
        if _import_worker_thread.is_alive():
            with _lock:
                for job in _jobs.values():
                    if not job.is_terminal:
                        logger.warning(
                            "Worker thread for job %s still alive after %.1fs shutdown timeout",
                            job.job_id,
                            timeout,
                        )
                        still_alive.append(job.job_id)
    return still_alive


# ---------------------------------------------------------------------------
# Import execution queue (single persistent worker, FIFO serial execution)
# ---------------------------------------------------------------------------

_import_queue: collections.deque[tuple[ImportJob, Path | None]] = collections.deque()
_import_condition: threading.Condition = threading.Condition()

_import_worker_thread: threading.Thread | None = None
_import_stop_event: threading.Event | None = None
_shutdown_in_progress: bool = False


def is_shutdown_in_progress() -> bool:
    return _shutdown_in_progress


def enqueue_for_execution(job: ImportJob, profile_raw_root: Path | None) -> None:
    """Append *job* to the execution FIFO.

    Must be called after the job has reached PENDING state (after
    ``convert_to_pending()`` for UPLOAD jobs, or after ``create_job()`` for
    PATH jobs).  The persistent worker thread picks jobs up in FIFO order.
    """
    with _import_condition:
        _import_queue.append((job, profile_raw_root))
        _import_condition.notify_all()
    logger.debug("Queued import job %s for execution", job.job_id)


def _import_worker_loop(
    stop_event: threading.Event,
    run_callback: Callable[[ImportJob, Path | None], None],
) -> None:
    def _is_deposed() -> bool:
        # Exit if this thread has been replaced by a newer worker, guaranteeing a
        # deposed worker terminates ≤1 wait cycle after replacement rather than
        # running indefinitely on the shared queue.  Reads the live module global
        # on each call so a replacement started mid-loop is observed.
        return _import_worker_thread is not threading.current_thread()

    def _execute(item: tuple[ImportJob, Path | None]) -> None:
        job, profile_raw_root = item
        if not job.try_start():
            # Job was cancelled while it was queued (state is now CANCELLED).
            # run_callback never ran, so we are responsible for cleanup.
            job.cleanup_files()
            job.release_capacity()
            return
        run_callback(job, profile_raw_root)

    def _on_error(item: tuple[ImportJob, Path | None], exc: BaseException) -> None:
        # run_callback's finally block (cleanup_files + release_capacity) has
        # already run.  Ensure a terminal state is recorded so observers don't
        # hang on a never-closing SSE stream.
        job, _ = item
        if not job.is_terminal:
            job._finish(
                succeeded=False,
                terminal_msg={"event": "error", "data": {"message": "Worker error"}},
            )

    run_worker_loop(
        stop_event,
        _import_condition,
        _import_queue,
        _execute,
        is_deposed=_is_deposed,
        on_execute_error=_on_error,
    )


def stop_import_worker(timeout: float = 5.0) -> None:
    """Stop the active import worker thread and wait for it to exit.

    Safe to call when no worker is running.  Used by ``shutdown()`` and test
    fixtures that need to tear down a running worker cleanly.  Notifies
    ``_import_condition`` so the worker wakes immediately rather than waiting
    out its 1.0 s poll interval.
    """
    global _import_worker_thread, _import_stop_event
    event = _import_stop_event
    thread = _import_worker_thread
    if event is not None:
        event.set()
    with _import_condition:
        _import_condition.notify_all()
    if thread is not None:
        thread.join(timeout=timeout)
    _import_worker_thread = None
    _import_stop_event = None


def start_import_worker(
    run_callback: Callable[[ImportJob, Path | None], None],
) -> tuple[threading.Thread, threading.Event]:
    """Start the single persistent import-job worker thread.

    Returns ``(thread, stop_event)`` so the caller can shut it down.
    Jobs enqueued before the worker starts are not lost — they sit in the
    deque and are processed as soon as the worker begins its loop.

    If a previous worker thread is still alive it is stopped gracefully
    (best-effort, 2 s join) before the new one is started.  The deposed-worker
    check inside ``_import_worker_loop`` is the hard guarantee — a stale thread
    that outlives the join exits as soon as it next wakes.
    """
    global _import_worker_thread, _import_stop_event
    # Gracefully retire any existing worker before starting a replacement.
    if _import_worker_thread is not None and _import_worker_thread.is_alive():
        if _import_stop_event is not None:
            _import_stop_event.set()
        with _import_condition:
            _import_condition.notify_all()
        _import_worker_thread.join(timeout=2.0)
    stop_event = threading.Event()
    _import_stop_event = stop_event
    t = threading.Thread(
        target=_import_worker_loop,
        args=(stop_event, run_callback),
        daemon=True,
        name="import-job-worker",
    )
    _import_worker_thread = t
    t.start()
    return t, stop_event


def _reap_stale_pending_uploads() -> None:
    """Cancel and remove PENDING_UPLOAD jobs that have exceeded the idle timeout.

    A PENDING_UPLOAD job is stale when no chunk has arrived for
    ``PENDING_UPLOAD_TIMEOUT_SECONDS``.  Each candidate is evaluated with
    ``try_cancel_if_stale_upload``, which atomically checks *both* state and
    idle duration under the job lock — this closes the race where a job's
    final chunk arrives and ``convert_to_pending()`` succeeds between the
    reaper's snapshot and its cancel call.  Only jobs for which the method
    returns a float (idle seconds, computed under the lock) are removed and
    cleaned up; None means "not stale" or "wrong state".

    Ordering mirrors ``shutdown()``: cancel → remove → cleanup_files →
    release_capacity (capacity is released last so the slot owns the disk it
    admitted throughout cleanup).
    """
    with _lock:
        candidates = list(_jobs.values())
    for job in candidates:
        idle = job.try_cancel_if_stale_upload(PENDING_UPLOAD_TIMEOUT_SECONDS)
        if idle is None:
            continue
        logger.info("Reaped stale PENDING_UPLOAD job %s (idle %.0fs)", job.job_id, idle)
        with _lock:
            _jobs.pop(job.job_id, None)
        job.cleanup_files()
        job.release_capacity()


def start_reaper(
    interval: float = 60.0,
    spool_sweep_fn: Callable[[], None] | None = None,
) -> tuple[threading.Thread, threading.Event]:
    """Start a background daemon thread that reaps jobs every *interval* seconds.

    Each iteration runs both the terminal-job TTL reaper and the stale
    PENDING_UPLOAD reaper so abandoned multi-chunk uploads release their
    admission slots promptly.  If *spool_sweep_fn* is provided it is called
    each iteration after the two reapers; a sweep failure never kills the thread.

    In production, *spool_sweep_fn* is wired to the stale spool-dir cleanup so
    that directories orphaned by a crash or restart are removed periodically
    rather than only at the next startup.
    """
    stop_event = threading.Event()

    def _reap_loop() -> None:
        while not stop_event.wait(timeout=interval):
            try:
                _reap_terminal()
                _reap_stale_pending_uploads()
            except Exception:
                logger.exception("Reaper iteration failed")
            if spool_sweep_fn is not None:
                try:
                    spool_sweep_fn()
                except Exception:
                    logger.exception("Spool sweep failed")

    t = threading.Thread(target=_reap_loop, daemon=True, name="import-job-reaper")
    t.start()
    return t, stop_event


# Reaper pattern: dedicated background thread (start_reaper) PLUS eager reap-on-read
# in get_job / list_jobs / reserve_slot / create_job.  The eager-reap-on-read path
# keeps the job list tidy for HTTP handlers that poll the store on every
# GET /import/{id}/progress and GET /import/jobs request.
# analysis_jobs uses a simpler per-iteration reap inside the worker loop — sufficient
# there because the analysis store is accessed only by the worker thread and a
# low-frequency list endpoint, not by every HTTP poll.


def _reap_terminal() -> None:
    """Remove terminal jobs older than JOB_TTL_SECONDS. Active jobs are NEVER reaped.

    Capacity-held jobs are skipped: the slot owns the disk it admitted, so the
    job record is retained until cleanup releases capacity.
    """
    removed = _store.reap(
        JOB_TTL_SECONDS,
        terminal_at=lambda job: job._terminal_at,
        reapable=lambda job: not job._capacity_held,
    )
    for jid in removed:
        logger.debug("Reaped terminal job %s", jid)
