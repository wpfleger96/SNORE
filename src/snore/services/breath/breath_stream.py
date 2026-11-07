"""Streamed per-run reductions of breath rows.

A night or an epoch spans hundreds of thousands of breaths.  Buffered, their
SQLAlchemy ``Row`` objects survive across awaits, so young-generation
collections promote them and set off full collections that stall the shared
event loop; reducing them in Python on the loop stalls it as well.  Here rows
leave the cursor a fetch at a time as float blocks, and each run's reduction
runs on a worker thread.
"""

from __future__ import annotations

import asyncio

from collections.abc import Callable, Sequence

import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncResult, AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from snore.database import models
from snore.utils.db_chunk import iter_id_chunks

# Breath rows fetched from the cursor at a time.  Each fetch becomes a float
# block at once, so its Row objects die young instead of piling up per run.
# A fetch allocates a driver tuple and a Row per row; kept well under gc's
# young-generation threshold (2000 allocations), few of a fetch's rows live
# through a young collection, so few reach the old generation.  Larger fetches
# set off full collections of ~100 ms each on the event loop (measured over a
# 366-night epoch on Python 3.14: 768 rows, 7 of them; 512 rows, none).
_BREATH_FETCH_ROWS = 512


async def _fetch_breath_block(
    result: AsyncResult[*tuple[object, ...]],
) -> np.ndarray | None:
    """The next fetched rows as a ``(columns, n)`` float block.

    None once the result is exhausted.  A null becomes NaN (SQLite stores NaN
    as NULL, so NaN never stands for a stored value), True 1.0 and False 0.0.
    The rows are freed on return; ``partitions()`` would instead keep each
    partition alive through the next fetch.
    """
    rows = await result.fetchmany(_BREATH_FETCH_ROWS)
    return np.array(list(zip(*rows, strict=True)), dtype=np.float64) if rows else None


async def stream_run_reductions[T](
    db: AsyncSession,
    ar_ids: Sequence[int],
    columns: Sequence[InstrumentedAttribute[object]],
    reduce: Callable[[np.ndarray], T],
) -> dict[int, T]:
    """``reduce`` of each run's breaths by analysis run id; runs without breaths are absent.

    ``reduce`` runs on a worker thread and gets one run's breaths in breath
    order as a ``(len(columns), n)`` float64 array whose row i is
    ``columns[i]``, nulls as NaN.  Every column must be a small int, boolean or
    float, so float64 holds it exactly.  Only one run's blocks are held at a time.
    """
    reduced: dict[int, T] = {}

    async def reduce_run(blocks: list[np.ndarray]) -> None:
        reduced[int(blocks[0][0, 0])] = await asyncio.to_thread(
            lambda: reduce(np.concatenate(blocks, axis=1)[1:])
        )

    breath = models.Breath
    for chunk in iter_id_chunks(ar_ids):
        # uq_breath_run_number serves this order without a sort.
        result = await db.stream(
            select(breath.analysis_result_id, *columns)
            .where(breath.analysis_result_id.in_(chunk))
            .order_by(breath.analysis_result_id, breath.breath_number)
        )
        # Close on every exit: an unfinalized statement left on the pooled
        # connection pins a stale read snapshot (blocking WAL checkpoints and
        # later writes).  AsyncResult has no ``async with``.
        try:
            run_blocks: list[np.ndarray] = []
            while (block := await _fetch_breath_block(result)) is not None:
                # A run's rows are contiguous, so split where the id changes.
                cuts = np.flatnonzero(np.diff(block[0])) + 1
                for piece in np.split(block, cuts, axis=1):
                    if run_blocks and run_blocks[0][0, 0] != piece[0, 0]:
                        await reduce_run(run_blocks)
                        run_blocks = []
                    run_blocks.append(piece)
            if run_blocks:
                await reduce_run(run_blocks)
        finally:
            await result.close()
    return reduced
