"""Shared query helpers for AnalysisResult recency lookups.

"Latest analysis for a session" is defined everywhere as: highest
``created_at``, ties broken by highest ``id``.  These helpers are the single
source of that ordering — single-row callers use ``latest_analysis_stmt`` /
``latest_analysis_row``; batch callers use the window-function helpers.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Subquery

from snore.database import models
from snore.utils.db_chunk import iter_id_chunks

__all__ = [
    "latest_analysis_ids",
    "latest_analysis_ranked_subquery",
    "latest_analysis_row",
    "latest_analysis_stmt",
]


def latest_analysis_stmt(session_id: int) -> Select[models.AnalysisResult]:
    """SELECT for one session's AnalysisResult rows, most recent first.

    Ordered by ``created_at DESC, id DESC`` — callers take the first row as
    the latest run.  No profile scoping: callers validate ownership.
    """
    return (
        select(models.AnalysisResult)
        .where(models.AnalysisResult.session_id == session_id)
        .order_by(
            models.AnalysisResult.created_at.desc(),
            models.AnalysisResult.id.desc(),
        )
    )


async def latest_analysis_row(
    db: AsyncSession, session_id: int
) -> models.AnalysisResult | None:
    """Return the latest AnalysisResult ORM row for a session, or None."""
    return (
        (await db.execute(latest_analysis_stmt(session_id).limit(1))).scalars().first()
    )


def latest_analysis_ranked_subquery(
    session_ids: Iterable[int] | Select[int],
) -> Subquery:
    """Window-function subquery ranking AnalysisResult rows per session.

    Columns: ``session_id``, ``id``, ``recency_rank`` — rank 1 is the latest
    row (``created_at DESC, id DESC``) within each session partition.

    Args:
        session_ids: Session IDs to scope the ranking to — either a
            materialized iterable of ints or a single-column SELECT.
    """
    return (
        select(
            models.AnalysisResult.session_id,
            models.AnalysisResult.id,
            func.row_number()
            .over(
                partition_by=models.AnalysisResult.session_id,
                order_by=[
                    models.AnalysisResult.created_at.desc(),
                    models.AnalysisResult.id.desc(),
                ],
            )
            .label("recency_rank"),
        )
        .where(models.AnalysisResult.session_id.in_(session_ids))
        .subquery()
    )


async def latest_analysis_ids(
    db: AsyncSession, session_ids: Iterable[int]
) -> dict[int, int]:
    """Map each session ID to its latest AnalysisResult ID (by created_at).

    The id list is chunked to stay under SQLite's bound-parameter cap.  The
    window ranking is session-partitioned, so per-chunk results have disjoint
    session keys and merge cleanly across chunks.
    """
    ids = list(session_ids)
    if not ids:
        return {}

    result: dict[int, int] = {}
    for chunk in iter_id_chunks(ids):
        ranked = latest_analysis_ranked_subquery(chunk)
        rows = (
            await db.execute(
                select(ranked.c.session_id, ranked.c.id).where(
                    ranked.c.recency_rank == 1
                )
            )
        ).all()
        result.update({session_id: analysis_id for session_id, analysis_id in rows})
    return result
