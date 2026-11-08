"""Shared query helpers for AnalysisResult recency lookups.

"Latest analysis for a session" is defined everywhere as: highest
``created_at``, ties broken by highest ``id``.  These helpers are the single
source of that ordering — single-row callers use ``latest_analysis_stmt`` /
``latest_analysis_row``; batch callers use the window-function helpers.

The ranking reads only ``ix_analysis_results_session_latest`` (a covering index
over ``session_id, created_at, id, engine_versions_json``), so it never walks a
row's ``programmatic_result_json`` overflow chain to reach the columns stored
after it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any

from sqlalchemy import Row, Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import FromClause, Subquery

from snore.database import models
from snore.utils.db_chunk import iter_id_chunks

__all__ = [
    "latest_analysis_ids",
    "latest_analysis_ranked_subquery",
    "latest_analysis_row",
    "latest_analysis_rows",
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

    Columns: ``session_id``, ``id``, ``engine_versions_json``, ``recency_rank``
    — rank 1 is the latest row (``created_at DESC, id DESC``) within each
    session partition.  Every column is in the covering index, so the ranking
    never reads the table row.

    Args:
        session_ids: Session IDs to scope the ranking to — either a
            materialized iterable of ints or a single-column SELECT.
    """
    return (
        select(
            models.AnalysisResult.session_id,
            models.AnalysisResult.id,
            models.AnalysisResult.engine_versions_json,
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


async def latest_analysis_rows(
    db: AsyncSession, session_ids: Iterable[int] | Select[int], *columns: Any
) -> AsyncIterator[Row[Any]]:
    """Yield ``(id, session_id, engine_versions_json, *columns)`` for each
    session's latest AnalysisResult; sessions without a run yield nothing.

    The first three come from the covering index.  Only ``columns`` (expressions
    over ``models.AnalysisResult``) join back to the table row.  A SELECT of
    session ids ranks in one statement; a materialized id list is chunked under
    SQLite's bound-parameter cap.  No profile scoping: callers validate
    ownership.
    """
    scopes: Iterable[Iterable[int] | Select[int]] = (
        [session_ids]
        if isinstance(session_ids, Select)
        else iter_id_chunks(list(session_ids))
    )
    for scope in scopes:
        ranked = latest_analysis_ranked_subquery(scope)
        source: FromClause = ranked
        if columns:
            ar = models.AnalysisResult
            source = ranked.join(ar, ar.id == ranked.c.id)
        stmt = (
            select(ranked.c.id, ranked.c.session_id, ranked.c.engine_versions_json)
            .add_columns(*columns)
            .select_from(source)
            .where(ranked.c.recency_rank == 1)
        )
        for row in await db.execute(stmt):
            yield row
