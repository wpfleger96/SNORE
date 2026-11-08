"""Single-session analysis runner: load → prepare → compute → store.

``run_one`` is the shared pipeline behind ``AnalysisFacade.run_analysis`` and
``BatchAnalysisCoordinator``: it owns a short read scope for the I/O phase
(closed before compute), dispatches the CPU-bound compute phase per the
``Dispatch`` policy, and persists the result via ``store_with_retry``.
"""

from __future__ import annotations

import asyncio
import logging

from enum import StrEnum
from typing import Any

from sqlalchemy import select

from snore.analysis.types import AnalysisResult
from snore.database import models

__all__ = ["Dispatch", "run_one", "store_with_retry"]

logger = logging.getLogger(__name__)


class Dispatch(StrEnum):
    """Where the CPU-bound compute phase runs."""

    INLINE = "inline"
    THREAD = "thread"
    PROCESS = "process"


async def run_one(
    session_id: int,
    *,
    profile_id: int,
    modes: list[str] | None = None,
    primary_mode: str | None = None,
    store: bool = True,
    dispatch: Dispatch = Dispatch.THREAD,
) -> AnalysisResult:
    """Run analysis on one session.  Returns AnalysisResult (Pydantic model).

    Validates session ownership before running.  Raises NotFoundError for
    foreign or missing session IDs; ValueError for a session without flow
    waveform data or an invalid ``primary_mode``.

    Owns a short read scope for the I/O phase and closes it before compute,
    so no session is ever held across NumPy/scipy work.  The compute phase
    runs per ``dispatch``: inline on the event loop, in a thread via
    ``asyncio.to_thread()``, or in the shared process pool — only detached
    ``RawSessionBlobs`` cross the pickle boundary for PROCESS dispatch.

    Args:
        session_id: Database session ID.
        profile_id: Profile that owns the session — all I/O is scoped to it.
        modes: Detection modes to run (None = default mode).
        primary_mode: Mode whose recovery markers are persisted.  Must be a
            member of ``modes`` when supplied; defaults to DEFAULT_MODE when
            DEFAULT_MODE is in modes, else to the sole mode when ``modes`` has
            one entry; required otherwise (ValueError).
        store: Whether to persist results to DB.
        dispatch: Compute-phase dispatch policy.
    """
    import time  # noqa: PLC0415

    from snore.analysis.service import (  # noqa: PLC0415
        AnalysisService,
        _compute_session_in_process,
    )
    from snore.database.session import session_scope  # noqa: PLC0415
    from snore.exceptions import NotFoundError  # noqa: PLC0415

    t_start = time.monotonic()

    # I/O phase: open a dedicated short async scope — close it before compute.
    async with session_scope() as read_db:
        owned = (
            await read_db.execute(
                select(models.Session.id)
                .join(models.Device, models.Session.device_id == models.Device.id)
                .where(
                    models.Session.id == session_id,
                    models.Device.profile_id == profile_id,
                )
            )
        ).scalar_one_or_none()
        if owned is None:
            raise NotFoundError(f"Session {session_id} not found")

        read_svc = AnalysisService(read_db, profile_id=profile_id)
        raw = await read_svc.load_session_inputs_raw(
            session_id, modes=modes, primary_mode=primary_mode
        )

    # Compute phase: session closed; CPU-bound scipy/NumPy per dispatch policy.
    if dispatch is Dispatch.PROCESS:
        from snore.utils.process_pool import get_pool  # noqa: PLC0415

        # Only the detached RawSessionBlobs DTO crosses the pickle boundary.
        computation = await asyncio.get_running_loop().run_in_executor(
            get_pool(), _compute_session_in_process, raw
        )
    else:
        # Prepare DTO (NumPy deserialization) — still sync/fast.
        inputs = AnalysisService.prepare_inputs(raw)
        compute_svc = AnalysisService()  # no db_session — compute only
        if dispatch is Dispatch.THREAD:
            computation = await asyncio.to_thread(compute_svc.compute_analysis, inputs)
        else:
            computation = compute_svc.compute_analysis(inputs)

    processing_time_ms = int((time.monotonic() - t_start) * 1000)

    if store:
        # Write phase: gated INSERT-only scope, retried on SQLite lock contention.
        await store_with_retry(profile_id, computation, processing_time_ms)

    return computation.summary


async def store_with_retry(
    profile_id: int,
    computation: Any,
    processing_time_ms: int,
) -> None:
    """Write one analysis result to the database, retrying on SQLite contention.

    Re-acquires ``write_gate``, opens a fresh ``session_scope(immediate=True)``,
    and calls ``store_result`` on each attempt.  Safe to retry because a failed
    commit rolls back — nothing was persisted.  Non-contention exceptions
    propagate on first occurrence.

    Args:
        profile_id: Profile that owns the analysis.
        computation: Completed computation result to store.
        processing_time_ms: Processing time in milliseconds.
    """
    from snore.analysis.service import AnalysisService  # noqa: PLC0415
    from snore.database.session import session_scope  # noqa: PLC0415
    from snore.database.txn import (  # noqa: PLC0415
        MAX_ATTEMPTS,
        backoff_delay,
        is_sqlite_contention,
    )
    from snore.database.write_gate import write_gate  # noqa: PLC0415

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with write_gate():
                async with session_scope(immediate=True) as write_session:
                    write_svc = AnalysisService(write_session, profile_id=profile_id)
                    await write_svc.store_result(computation, processing_time_ms)
            return
        except Exception as exc:
            if not is_sqlite_contention(exc) or attempt >= MAX_ATTEMPTS:
                raise
            delay = backoff_delay(attempt)
            logger.warning(
                "_store_with_retry: attempt %d/%d hit contention (%s); retrying in %.3fs",
                attempt,
                MAX_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
