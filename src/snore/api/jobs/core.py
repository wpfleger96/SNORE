"""Shared in-memory job machinery for the import and analysis pipelines.

``import_jobs`` and ``analysis_jobs`` are structural twins: both keep an
in-memory store of jobs guarded by a store lock, model each job as a
lock-guarded state machine with a cancel flag, and drain a FIFO queue from one
or more persistent worker threads.  This module single-sources the parts that
are genuinely identical between them:

- :class:`JobRecordBase` — the lock/state/cancel/timestamp core of a job.
- :class:`JobStore` — the ``{job_id: job}`` dict + lock, owner-visibility
  filter, and terminal-TTL reap.
- :func:`run_worker_loop` — the ``wait → dequeue → execute`` worker skeleton
  with its BaseException guard (log, force terminal, never re-raise).

Deliberately NOT unified here (they diverge in ways that matter):

- ``_finish`` / ``finish`` terminal transitions — import embeds the durable
  import result and broadcasts SSE observers; analysis records an error
  message.  Each subclass keeps its own.
- Observer / SSE machinery — import only.
- Two-phase admission, capacity accounting, queue eviction — policy that stays
  local to each module.
- Worker-thread topology (single deposable import worker vs N analysis
  workers) and stop-event bookkeeping — a caller concern.
"""

from __future__ import annotations

import logging
import threading
import time

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

logger = logging.getLogger(__name__)

# How long to retain terminal jobs before the reaper removes them.  Shared by
# both pipelines; single-sourced here to kill the cross-module import that used
# to couple analysis_jobs to import_jobs.
JOB_TTL_SECONDS: float = 600.0

# Held across admission of the database-wide maintenance jobs (reset/delete-data
# in ``reset_jobs`` and the day recompute in ``recompute_jobs``), which must never
# run together.  Each enqueue checks the other module's active jobs and admits its
# own under this lock, so two concurrent requests cannot both pass the check.
# Always acquire it before a module's own store lock, never while holding one.
MAINTENANCE_ADMISSION_LOCK = threading.Lock()


@dataclass(kw_only=True)
class JobRecordBase[StateT: Enum]:
    """Lock/state/cancel/timestamp core shared by every in-memory job.

    Subclasses declare their own ``_state`` field (with the concrete enum type
    and initial value) plus the ``_TERMINAL_STATES`` class attribute the state
    properties read.  ``_state`` stays a plain, settable dataclass field — tests
    assign it directly and startup-resume rewrites it.

    Every accessor snapshots under ``_lock``.  Subclasses that notify observers
    MUST snapshot the observer list under the lock and deliver outside it; no
    method here calls out while holding the lock.
    """

    # Overridden by each subclass with its concrete terminal-state frozenset.
    _TERMINAL_STATES: ClassVar[frozenset[Any]] = frozenset()

    if TYPE_CHECKING:
        # Declared as a real dataclass field by each subclass (differing enum
        # type + initial value); annotated here so the shared properties type
        # against it without the base creating a field of its own.
        _state: StateT

        # Every concrete job implements this identical terminal-transition
        # signature; declared here (TYPE_CHECKING only, so it never shadows the
        # real methods) so shared machinery — e.g. the WorkerPool error handler —
        # can force a failed job terminal without a per-subclass callback.
        def finish(self, succeeded: bool, error_message: str | None = None) -> None: ...

    job_id: str
    owner_user_id: int | None = None
    created_at: float = field(default_factory=time.monotonic)
    created_at_wall: datetime = field(default_factory=lambda: datetime.now(UTC))

    _cancel_flag: bool = field(default=False, init=False, repr=False)
    _started_at: float | None = field(default=None, init=False, repr=False)
    _started_at_wall: datetime | None = field(default=None, init=False, repr=False)
    _finished_at_wall: datetime | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    @property
    def state(self) -> StateT:
        with self._lock:
            return self._state

    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._state in self._TERMINAL_STATES

    @property
    def cancel_requested(self) -> bool:
        """True if cancellation has been requested (checked at batch boundaries)."""
        with self._lock:
            return self._cancel_flag

    @property
    def started_at(self) -> float | None:
        """Monotonic timestamp when the job entered RUNNING, or None."""
        with self._lock:
            return self._started_at

    @property
    def started_at_wall(self) -> datetime | None:
        """Wall-clock UTC timestamp when the job entered RUNNING, or None."""
        with self._lock:
            return self._started_at_wall

    @property
    def finished_at_wall(self) -> datetime | None:
        """Wall-clock UTC timestamp when the job reached a terminal state, or None."""
        with self._lock:
            return self._finished_at_wall

    def _start_running(self, running_state: StateT) -> None:
        """Transition to *running_state* and stamp start timestamps.

        Caller MUST hold ``self._lock``.  Single-sources the RUNNING-entry
        bookkeeping so both pipelines record ``started_at`` identically.
        """
        self._state = running_state
        self._started_at = time.monotonic()
        self._started_at_wall = datetime.now(UTC)

    def try_cancel(self) -> bool:
        """Request cancellation. Idempotent after terminal.

        Sets the cancel flag on any non-terminal job and returns True.  The
        eager-terminalize behaviour (which active states may cancel straight to
        a terminal state, and any notifications that entails) is supplied by the
        ``_on_cancel_locked`` / ``_notify_after_cancel`` subclass hooks so this
        skeleton never calls out while holding the lock.
        """
        payload: Any = None
        with self._lock:
            if self._state in self._TERMINAL_STATES:
                return False
            self._cancel_flag = True
            payload = self._on_cancel_locked()
        if payload is not None:
            self._notify_after_cancel(payload)
        return True

    def _on_cancel_locked(self) -> Any:
        """Hook: eager-terminalize while holding ``_lock``.

        Return a payload for :meth:`_notify_after_cancel` to deliver outside the
        lock, or None when there is nothing to notify.  Default: no-op.
        """
        return None

    def _notify_after_cancel(self, payload: Any) -> None:
        """Hook: deliver the ``_on_cancel_locked`` payload outside ``_lock``."""


class JobStore[J: JobRecordBase[Any]]:
    """The ``{job_id: job}`` dict + guarding lock shared by both pipelines.

    ``jobs`` and ``lock`` are public on purpose: each module binds its historical
    module-level ``_jobs``/``_all_jobs`` and ``_lock`` names to these SAME live
    objects, and callers that mutate them directly (startup resume, test reset
    fixtures) depend on that identity.  Those callers may only mutate the dict
    IN PLACE (``[...]=``, ``.pop()``, ``.clear()``) — never rebind
    ``store.jobs`` or ``store.lock``, or the module aliases would silently point
    at a dead object.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, J] = {}
        self.lock = threading.Lock()

    def get(self, job_id: str) -> J | None:
        with self.lock:
            return self.jobs.get(job_id)

    def add(self, job: J) -> None:
        with self.lock:
            self.jobs[job.job_id] = job

    def remove(self, job_id: str) -> J | None:
        with self.lock:
            return self.jobs.pop(job_id, None)

    def snapshot(self) -> list[J]:
        with self.lock:
            return list(self.jobs.values())

    def list_visible_to(self, owner_user_id: int | None) -> list[J]:
        """Return jobs visible to *owner_user_id*.

        A job owned by None is visible to any caller (local-mode parity); a job
        with a set owner is visible only to that owner.  When *owner_user_id* is
        None the caller receives every job.
        """
        jobs = self.snapshot()
        if owner_user_id is None:
            return jobs
        return [
            j
            for j in jobs
            if j.owner_user_id is None or j.owner_user_id == owner_user_id
        ]

    def cancel(self, job_id: str) -> bool:
        """Cancel a job. Returns True if it existed and was not already terminal.

        This only flips the job state via ``try_cancel``.  Pipelines that hold a
        separate reference to the job in a work queue (analysis) MUST NOT use
        this — they have to evict the queued reference under their own condition
        first, so those modules keep their own ``cancel_job`` instead.
        """
        job = self.get(job_id)
        if job is None:
            return False
        return job.try_cancel()

    def reap(
        self,
        ttl: float,
        *,
        terminal_at: Callable[[J], float | None],
        reapable: Callable[[J], bool] | None = None,
    ) -> list[str]:
        """Remove terminal jobs older than *ttl* seconds; never touch active jobs.

        ``terminal_at`` extracts a job's monotonic terminal timestamp (the field
        name differs between pipelines).  ``reapable`` is an extra guard applied
        after the terminal check — import uses it to keep capacity-held jobs.
        Returns the ids removed so callers can log them.
        """
        now = time.monotonic()
        with self.lock:
            to_remove = [
                jid
                for jid, job in self.jobs.items()
                if job.is_terminal
                and (reapable is None or reapable(job))
                and (ta := terminal_at(job)) is not None
                and now - ta > ttl
            ]
        removed: list[str] = []
        for jid in to_remove:
            with self.lock:
                if self.jobs.pop(jid, None) is not None:
                    removed.append(jid)
        return removed


def run_worker_loop[T](
    stop_event: threading.Event,
    condition: threading.Condition,
    queue: deque[T],
    execute: Callable[[T], None],
    *,
    is_deposed: Callable[[], bool] | None = None,
    on_idle: Callable[[], None] | None = None,
    after_execute: Callable[[], None] | None = None,
    on_execute_error: Callable[[T, BaseException], None] | None = None,
) -> None:
    """Drive a persistent worker thread over *queue* until *stop_event* is set.

    The skeleton both pipelines share::

        wait(1.0) on the condition → popleft → execute(item)

    with a BaseException guard around ``execute`` so a job that blows up never
    kills the thread.  Hooks capture the parts that differ:

    - ``is_deposed`` — import's single-worker replacement check.  Called at the
      top of every cycle and again after the wait; when it returns True the
      thread exits within one cycle.  Must read the live worker-thread global on
      each call.
    - ``on_idle`` — run when a cycle dequeued nothing (analysis's throttled reap).
    - ``after_execute`` — run after each executed item (analysis's per-job reap).
    - ``on_execute_error`` — force the item terminal after a BaseException so
      observers/pollers never hang; its own failures are swallowed.
    """
    while not stop_event.is_set():
        if is_deposed is not None and is_deposed():
            return
        item: T | None = None
        with condition:
            if not queue and not stop_event.is_set():
                condition.wait(timeout=1.0)
            # Re-check after the wait: we may have been replaced while sleeping.
            if is_deposed is not None and is_deposed():
                return
            if not stop_event.is_set() and queue:
                item = queue.popleft()

        if item is None:
            if on_idle is not None:
                on_idle()
            continue

        try:
            execute(item)
        except BaseException as exc:
            logger.exception("Unexpected exception in worker loop")
            if on_execute_error is not None:
                try:
                    on_execute_error(item, exc)
                except Exception:
                    pass
            # Do NOT re-raise — the worker thread must stay alive.

        if after_execute is not None:
            after_execute()
