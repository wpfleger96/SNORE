"""Re-derive every Day row from stored session statistics and waveforms.

Shared by ``snore db recompute-days`` and the admin recompute job
(``snore.api.recompute_jobs``).  No reparse: each Day is re-aggregated by
``DayManager.recalculate_day`` from what is already stored, across every user,
profile and device.  Day rows that no session references are deleted.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy import select

from snore.database.day_manager import DayManager
from snore.database.models import Day
from snore.database.session import session_scope
from snore.database.write_gate import write_gate
from snore.services.schemas import RecomputeDaysResult
from snore.utils.db_chunk import iter_id_chunks

# Days recomputed per write transaction.  Each day decodes its sessions'
# waveform blobs while holding the SQLite write lock, so chunks stay small.
RECOMPUTE_DAYS_CHUNK_SIZE = 100

ProgressCallback = Callable[[int, int], None]
"""Called with ``(days_processed, days_total)`` before the first chunk and after each."""


async def recompute_all_days(
    on_progress: ProgressCallback | None = None,
) -> RecomputeDaysResult:
    """Recompute every Day in chunks, each committed in its own gated write.

    The SQLite write lock is released between chunks, so concurrent import and
    analysis workers are not starved.  Recomputation is idempotent, so a run
    interrupted after some chunks committed can simply be re-run.
    """
    async with session_scope() as session:
        day_ids = (
            (await session.execute(select(Day.id).order_by(Day.id))).scalars().all()
        )

    total = len(day_ids)
    processed = 0
    pruned = 0
    if on_progress is not None:
        on_progress(processed, total)
    for chunk in iter_id_chunks(day_ids, RECOMPUTE_DAYS_CHUNK_SIZE):
        async with write_gate(), session_scope(immediate=True) as session:
            days = (
                (await session.execute(select(Day).where(Day.id.in_(chunk))))
                .scalars()
                .all()
            )
            for day in days:
                if not await DayManager.recalculate_day(day, session):
                    pruned += 1
        processed += len(chunk)
        if on_progress is not None:
            on_progress(processed, total)

    return RecomputeDaysResult(days_recomputed=total - pruned, days_pruned=pruned)
