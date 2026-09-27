"""Analysis facade for listing and managing analysis results."""

import asyncio
import logging

from collections.abc import Callable, Sequence
from datetime import date, datetime
from typing import Any

from sqlalchemy import ColumnElement, Select, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.queries import (
    latest_analysis_ids,
    latest_analysis_ranked_subquery,
)
from snore.analysis.runner import Dispatch, run_one, store_with_retry
from snore.analysis.shared.versioning import AlgorithmIdentity, AlgoVersions
from snore.analysis.types import AnalysisResult
from snore.database import models
from snore.services import _base
from snore.services.schemas import (
    AnalysisDeletePreview,
    AnalysisListItem,
    AnalysisSessionDetail,
    BatchAnalysisResult,
    BatchSessionResult,
)
from snore.utils.db_chunk import iter_id_chunks

__all__ = ["AnalysisFacade", "BatchAnalysisCoordinator"]

logger = logging.getLogger(__name__)


def _is_stale(engine_versions_json: dict[str, Any]) -> bool:
    """Return True when an AnalysisResult row's engine_versions_json is stale.

    Stale means: legacy flat row (no 'identity' key), unparseable, or the
    stored AlgorithmIdentity does not match the current identity exactly.
    """
    parsed = AlgoVersions.from_stored(engine_versions_json)
    if parsed is None:
        return True
    return parsed.identity.model_dump() != AlgorithmIdentity.current().model_dump()


class AnalysisFacade:
    """Facade for analysis listing and deletion operations."""

    def __init__(self, db_session: AsyncSession | None, profile_id: int) -> None:
        """
        Initialize analysis facade.

        Args:
            db_session: SQLAlchemy database session.  May be ``None`` **only**
                when the facade is used exclusively for batch analysis via
                ``run_batch_analysis``, which opens its own short-lived scopes
                for all I/O.  All other methods (list, delete, get, etc.) require
                a live session and will fail at runtime if ``None`` is passed.
            profile_id: Active profile — all queries are scoped to this profile.
        """
        self.db_session = db_session
        self.profile_id = profile_id
        self._batch_coordinator: BatchAnalysisCoordinator | None = None

    @property
    def _db(self) -> AsyncSession:
        """Return the live session.

        Raises ``RuntimeError`` when the facade was constructed with
        ``db_session=None`` (batch-only mode) and a method that requires a
        live session is called.
        """
        if self.db_session is None:
            raise RuntimeError(
                "AnalysisFacade requires a live db_session for this method. "
                "Pass a session to __init__ or use run_batch_analysis instead."
            )
        return self.db_session

    def _profile_filter(self) -> ColumnElement[bool]:
        """WHERE predicate: limit sessions to this profile via device ownership."""
        return models.Device.profile_id == self.profile_id

    @property
    def batch_coordinator(self) -> "BatchAnalysisCoordinator | None":
        """The active ``BatchAnalysisCoordinator``, or ``None`` when no batch is running.

        Set immediately before ``submit()`` is called; remains set after completion so
        callers can inspect ``progress`` after the batch finishes.  PR-2 replaces the
        sync coordinator with an async one; callers use the same ``cancel()`` handle.
        """
        return self._batch_coordinator

    def _status_select(
        self,
        start: datetime | None,
        end: datetime | None,
        analyzed_only: bool,
    ) -> Select[models.Session]:
        """Build the shared 2.0-style select for list/count of analysis status."""
        stmt = (
            select(models.Session)
            .join(models.Device, models.Session.device_id == models.Device.id)
            .join(models.Day, models.Session.day_id == models.Day.id)
            .where(self._profile_filter())
        )

        if start:
            stmt = stmt.where(models.Day.date >= start.date())
        if end:
            stmt = stmt.where(models.Day.date <= end.date())

        if analyzed_only:
            stmt = stmt.where(
                select(models.AnalysisResult.id)
                .where(models.AnalysisResult.session_id == models.Session.id)
                .exists()
            )

        return stmt

    async def _latest_analysis_ids(self, session_ids: list[int]) -> dict[int, int]:
        """Map each session ID to its latest AnalysisResult ID (by created_at)."""
        return await latest_analysis_ids(self._db, session_ids)

    async def list_sessions_with_status(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 20,
        offset: int = 0,
        analyzed_only: bool = False,
        sort_by: str = "date-desc",
    ) -> list[AnalysisListItem]:
        """List sessions with their analysis status.

        Args:
            start: Filter sessions from this datetime (inclusive)
            end: Filter sessions to this datetime (inclusive)
            limit: Maximum sessions to return (0 for unlimited)
            analyzed_only: Only return sessions that have analysis results
            sort_by: Sort order (date-asc, date-desc, session-id)

        Returns:
            List of AnalysisListItem with has_analysis and analysis_id fields
        """
        stmt = self._status_select(start, end, analyzed_only)

        sort_clauses: dict[str, Any] = {
            "date-asc": models.Day.date.asc(),
            "date-desc": models.Day.date.desc(),
            "session-id": models.Session.id.asc(),
        }

        sort_clause = sort_clauses.get(sort_by, models.Day.date.desc())
        stmt = stmt.order_by(sort_clause)

        if offset > 0:
            stmt = stmt.offset(offset)

        if limit > 0:
            stmt = stmt.limit(limit)
        else:
            stmt = stmt.limit(10000)

        # Load Day inline to avoid lazy-load access below.
        from sqlalchemy.orm import joinedload as _joinedload

        stmt = stmt.options(_joinedload(models.Session.day))
        sessions = (await self._db.execute(stmt)).unique().scalars().all()

        latest_analysis = await self._latest_analysis_ids([s.id for s in sessions])

        results = []
        for session in sessions:
            analysis_id = latest_analysis.get(session.id)

            # day is loaded via joinedload; _status_select inner-joins Day so
            # session.day is always present here.
            session_date = session.day.date

            results.append(
                AnalysisListItem(
                    session_id=session.id,
                    session_date=session_date,
                    duration_hours=(
                        session.duration_seconds / 3600
                        if session.duration_seconds
                        else None
                    ),
                    has_analysis=analysis_id is not None,
                    analysis_id=analysis_id,
                )
            )

        return results

    async def count_sessions_with_status(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
        analyzed_only: bool = False,
    ) -> int:
        """Return total count of sessions matching the same filters as list_sessions_with_status.

        Used by the API to populate the `total` field in paginated responses.
        """
        count_stmt = select(func.count()).select_from(
            self._status_select(start, end, analyzed_only).subquery()
        )
        return (await self._db.execute(count_stmt)).scalar() or 0

    async def get_delete_preview(
        self,
        session_ids: list[int] | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        delete_all: bool = False,
        all_versions: bool = False,
        stale_versions: bool = False,
    ) -> AnalysisDeletePreview:
        """Preview analysis data that would be deleted.

        Args:
            session_ids: Specific session IDs to filter
            from_date: Filter sessions from this datetime
            to_date: Filter sessions to this datetime
            delete_all: Consider all sessions
            all_versions: Count all analysis versions (affects records_to_delete).
                Ignored when stale_versions=True.
            stale_versions: When True, only count rows whose stored
                AlgorithmIdentity differs from the current identity (stale rows).

        Returns:
            AnalysisDeletePreview with counts and session details

        Raises:
            ValueError: If no filters specified
        """
        if not any([session_ids, from_date, to_date, delete_all]):
            raise ValueError(
                "At least one filter must be specified: "
                "session_ids, from_date, to_date, or delete_all"
            )

        if session_ids:
            # Dedupe: chunked IN-binds don't implicitly de-duplicate like a single IN.
            session_ids = list(dict.fromkeys(session_ids))

        base_query = (
            select(
                models.Session.id,
                models.Session.device_session_id,
                models.Session.start_time,
                models.Device.manufacturer,
                models.Device.model,
            )
            .join(models.Device, models.Session.device_id == models.Device.id)
            .join(
                models.AnalysisResult,
                models.Session.id == models.AnalysisResult.session_id,
            )
            .where(self._profile_filter())
            .distinct()
        )

        if from_date:
            base_query = base_query.where(models.Session.start_time >= from_date)

        if to_date:
            base_query = base_query.where(models.Session.start_time <= to_date)

        # session_ids is an unbounded caller list — chunk it under the bind cap.
        # Chunks are disjoint so the per-chunk (already distinct) rows concat
        # cleanly; re-sort start_time DESC to restore the single-query ordering.
        sessions_with_analysis: list[Any]
        if session_ids:
            sessions_with_analysis = []
            for chunk in iter_id_chunks(session_ids):
                sessions_with_analysis.extend(
                    (
                        await self._db.execute(
                            base_query.where(models.Session.id.in_(chunk))
                        )
                    ).fetchall()
                )
            sessions_with_analysis.sort(key=lambda s: s.start_time, reverse=True)
        else:
            sessions_with_analysis = list(
                (
                    await self._db.execute(
                        base_query.order_by(models.Session.start_time.desc())
                    )
                ).fetchall()
            )

        if not sessions_with_analysis:
            return AnalysisDeletePreview(
                sessions_with_analysis=0,
                total_analysis_records=0,
                records_to_delete=0,
                patterns_count=0,
                session_details=[],
            )

        session_ids_list = [s.id for s in sessions_with_analysis]

        if stale_versions:
            return await self._get_stale_versions_preview(
                sessions_with_analysis, session_ids_list
            )

        # Chunk the id list; the grouped key IS the chunked column so per-chunk
        # dicts have disjoint keys and merge without overwrite collisions.
        analysis_count_dict: dict[int, int] = {}
        for chunk in iter_id_chunks(session_ids_list):
            rows = (
                await self._db.execute(
                    select(models.AnalysisResult.session_id, func.count())
                    .where(models.AnalysisResult.session_id.in_(chunk))
                    .group_by(models.AnalysisResult.session_id)
                )
            ).fetchall()
            analysis_count_dict.update({row[0]: int(row[1]) for row in rows})

        total_analysis_records = sum(analysis_count_dict.values())
        records_to_delete = (
            total_analysis_records if all_versions else len(sessions_with_analysis)
        )

        # The outer .in_(select(...)) binds no params, but the inner
        # session_id.in_(...) does — chunk it and sum the disjoint counts.
        patterns_count = 0
        for chunk in iter_id_chunks(session_ids_list):
            patterns_count += (
                await self._db.execute(
                    select(func.count())
                    .select_from(models.DetectedPattern)
                    .where(
                        models.DetectedPattern.analysis_result_id.in_(
                            select(models.AnalysisResult.id).where(
                                models.AnalysisResult.session_id.in_(chunk)
                            )
                        )
                    )
                )
            ).scalar() or 0

        session_details = [
            AnalysisSessionDetail(
                id=s.id,
                start_time=s.start_time,
                manufacturer=s.manufacturer,
                model=s.model,
                version_count=analysis_count_dict.get(s.id, 0),
            )
            for s in sessions_with_analysis
        ]

        return AnalysisDeletePreview(
            sessions_with_analysis=len(sessions_with_analysis),
            total_analysis_records=total_analysis_records,
            records_to_delete=records_to_delete,
            patterns_count=patterns_count or 0,
            session_details=session_details,
        )

    async def _get_stale_versions_preview(
        self,
        sessions_with_analysis: Sequence[Any],
        session_ids_list: list[int],
    ) -> AnalysisDeletePreview:
        """Build a delete preview for the --stale-versions mode.

        Fetches all AnalysisResult rows for the scoped sessions, classifies each
        as stale or current in Python (reusing AlgoVersions.from_stored), and
        returns counts scoped to stale rows only.
        """
        # Chunk the id list; rows are classified in Python and the session_details
        # ordering is restored below, so per-chunk concat order is irrelevant.
        all_rows: list[Any] = []
        for chunk in iter_id_chunks(session_ids_list):
            all_rows.extend(
                (
                    await self._db.execute(
                        select(
                            models.AnalysisResult.id,
                            models.AnalysisResult.session_id,
                            models.AnalysisResult.engine_versions_json,
                        ).where(models.AnalysisResult.session_id.in_(chunk))
                    )
                ).fetchall()
            )

        total_analysis_records = len(all_rows)

        stale_ids: list[int] = []
        stale_count_by_session: dict[int, int] = {}
        for row in all_rows:
            if _is_stale(row.engine_versions_json):
                stale_ids.append(row.id)
                stale_count_by_session[row.session_id] = (
                    stale_count_by_session.get(row.session_id, 0) + 1
                )

        sessions_row_map = {s.id: s for s in sessions_with_analysis}
        session_details = [
            AnalysisSessionDetail(
                id=sid,
                start_time=sessions_row_map[sid].start_time,
                manufacturer=sessions_row_map[sid].manufacturer,
                model=sessions_row_map[sid].model,
                version_count=count,
            )
            for sid, count in stale_count_by_session.items()
            if sid in sessions_row_map
        ]
        # Preserve stable ordering (newest first matches the outer query).
        session_order = {s.id: i for i, s in enumerate(sessions_with_analysis)}
        session_details.sort(key=lambda d: session_order.get(d.id, 0))

        # Chunk stale_ids (can exceed the bind cap) and sum the disjoint counts.
        # Empty stale_ids yields no chunks, leaving patterns_count at 0.
        patterns_count = 0
        for chunk in iter_id_chunks(stale_ids):
            patterns_count += (
                await self._db.execute(
                    select(func.count())
                    .select_from(models.DetectedPattern)
                    .where(models.DetectedPattern.analysis_result_id.in_(chunk))
                )
            ).scalar() or 0

        return AnalysisDeletePreview(
            sessions_with_analysis=len(stale_count_by_session),
            total_analysis_records=total_analysis_records,
            records_to_delete=len(stale_ids),
            patterns_count=patterns_count,
            session_details=session_details,
        )

    async def delete_analysis(
        self,
        session_ids: list[int],
        all_versions: bool = False,
        stale_versions: bool = False,
    ) -> int:
        """Delete analysis results for given sessions.

        All requested session IDs must belong to this profile — callers should
        validate via ``get_owned_session_ids`` and return 404 before calling
        this method.  The scoped subquery is retained as defence-in-depth.

        Args:
            session_ids: Session IDs to delete analysis for
            all_versions: If True, delete all versions. If False, only latest.
                Ignored when stale_versions=True.
            stale_versions: If True, delete only rows whose stored
                AlgorithmIdentity differs from the current identity. Child rows
                (Breath, DetectedPattern) are removed via DB-level CASCADE.

        Returns:
            Number of analysis records deleted
        """
        # Dedupe: chunked IN-binds don't implicitly de-duplicate like a single IN.
        session_ids = list(dict.fromkeys(session_ids))
        if not session_ids:
            return 0

        # Chunk the whole operation by session_ids: session chunks are disjoint,
        # so their AnalysisResult sets are disjoint and rowcounts sum cleanly.
        # Stale classification and latest-per-session ranking are per-session
        # independent, so chunking preserves each branch's result.
        total = 0
        for chunk in iter_id_chunks(session_ids):
            # Scope this chunk to this profile to prevent cross-profile deletion.
            owned_sessions_subq = (
                select(models.Session.id)
                .join(models.Device, models.Session.device_id == models.Device.id)
                .where(
                    models.Session.id.in_(chunk),
                    self._profile_filter(),
                )
                .subquery()
            )

            if stale_versions:
                # Classify rows in Python to reuse the AlgoVersions parser.
                rows = (
                    await self._db.execute(
                        select(
                            models.AnalysisResult.id,
                            models.AnalysisResult.engine_versions_json,
                        ).where(
                            models.AnalysisResult.session_id.in_(
                                select(owned_sessions_subq.c.id)
                            )
                        )
                    )
                ).fetchall()
                stale_ids = [r.id for r in rows if _is_stale(r.engine_versions_json)]
                # stale_ids can exceed the bind cap within a single session chunk
                # (one session may carry many analysis versions) — chunk again.
                for id_chunk in iter_id_chunks(stale_ids):
                    result = await self._db.execute(
                        delete(models.AnalysisResult).where(
                            models.AnalysisResult.id.in_(id_chunk)
                        )
                    )
                    total += result.rowcount or 0  # type: ignore[attr-defined]

            elif all_versions:
                # Delete all analysis results for owned sessions.
                result = await self._db.execute(
                    delete(models.AnalysisResult).where(
                        models.AnalysisResult.session_id.in_(
                            select(owned_sessions_subq.c.id)
                        )
                    )
                )
                total += result.rowcount or 0  # type: ignore[attr-defined]

            else:
                # Delete only the latest (highest created_at, then id) result per
                # owned session — at most one id per session, so <= chunk size.
                ranked = latest_analysis_ranked_subquery(
                    select(owned_sessions_subq.c.id)
                )
                latest_ids: Sequence[int] = (
                    (
                        await self._db.execute(
                            select(ranked.c.id).where(ranked.c.recency_rank == 1)
                        )
                    )
                    .scalars()
                    .all()
                )
                if latest_ids:
                    result = await self._db.execute(
                        delete(models.AnalysisResult).where(
                            models.AnalysisResult.id.in_(latest_ids)
                        )
                    )
                    total += result.rowcount or 0  # type: ignore[attr-defined]

        return total

    async def get_owned_session_ids(self, session_ids: list[int]) -> set[int]:
        """Return the subset of session_ids that belong to this profile.

        Used by routes to validate ownership before mutation: any ID absent from
        the returned set is either missing or owned by a different profile.
        """
        return await _base.get_owned_session_ids(self._db, self.profile_id, session_ids)

    async def run_analysis(
        self,
        session_id: int,
        modes: list[str] | None = None,
        primary_mode: str | None = None,
        store_results: bool = True,
    ) -> AnalysisResult:
        """Run analysis on a session.  Returns AnalysisResult (Pydantic model).

        Validates session ownership before running.  Raises NotFoundError for
        foreign or missing session IDs.

        Owns a short read scope for the I/O phase and closes it before compute,
        so the injected request session is never held across NumPy/scipy work.
        CPU-bound compute runs in a thread via asyncio.to_thread().

        Args:
            session_id: Database session ID.
            modes: Detection modes to run (None = default mode).
            primary_mode: Mode whose recovery markers are persisted.  Must be a
                member of ``modes`` when supplied; defaults to DEFAULT_MODE when
                DEFAULT_MODE is in modes; required otherwise (ValueError).
            store_results: Whether to persist results to DB.
        """
        return await run_one(
            session_id,
            profile_id=self.profile_id,
            modes=modes,
            primary_mode=primary_mode,
            store=store_results,
            dispatch=Dispatch.THREAD,
        )

    async def get_analysis_result(self, session_id: int) -> AnalysisResult | None:
        """Get stored analysis result for a session, or None if not found.

        Validates session ownership — returns None for foreign IDs (treating
        "not yet analyzed" and "not found" the same to avoid oracle attacks).
        Delegates to ``AnalysisService.get_analysis_result``, the single
        implementation of this lookup.
        """
        from snore.analysis.service import AnalysisService  # noqa: PLC0415

        svc = AnalysisService(self._db, profile_id=self.profile_id)
        return await svc.get_analysis_result(session_id)

    async def list_session_ids(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        missing_only: bool = False,
    ) -> list[int]:
        """Return session IDs for this profile, optionally filtered by date range.

        Cheaper than ``list_sessions_with_status``: selects only the ID column,
        no ORM object hydration, no artificial row cap.

        When ``missing_only`` is true, restrict to sessions that have a flow
        waveform (sessions without flow data cannot be analyzed) but have no
        existing analysis result.  Composable with date filters.
        """
        stmt = (
            select(models.Session.id)
            .join(models.Device, models.Session.device_id == models.Device.id)
            .join(models.Day, models.Session.day_id == models.Day.id)
            .where(self._profile_filter())
        )
        if from_date:
            stmt = stmt.where(models.Day.date >= from_date.date())
        if to_date:
            stmt = stmt.where(models.Day.date <= to_date.date())
        if missing_only:
            stmt = stmt.where(
                select(models.Waveform.id)
                .where(models.Waveform.session_id == models.Session.id)
                .where(models.Waveform.waveform_type == "flow")
                .exists()
            )
            stmt = stmt.where(
                ~select(models.AnalysisResult.id)
                .where(models.AnalysisResult.session_id == models.Session.id)
                .exists()
            )
        stmt = stmt.order_by(models.Day.date)
        result = await self._db.execute(stmt)
        return list(result.scalars())

    async def run_batch_analysis(
        self,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        session_ids: list[int] | None = None,
        modes: Sequence[str] | None = None,
        primary_mode: str | None = None,
        store_results: bool = True,
        max_workers: int = 4,
        progress_callback: Callable[[int, int | None], None] | None = None,
        cancel_predicate: Callable[[], bool] | None = None,
    ) -> BatchAnalysisResult:
        """Run analysis on multiple sessions in parallel.

        Delegates to ``BatchAnalysisCoordinator`` so the executor
        internals can change without touching callers.

        Args:
            from_date: Filter sessions from this datetime (inclusive)
            to_date: Filter sessions to this datetime (inclusive)
            session_ids: Explicit session IDs to analyze (overrides date filters when set)
            modes: Detection modes to run (None = default)
            primary_mode: Mode whose recovery markers are persisted; defaults to
                DEFAULT_MODE when included in ``modes``, required otherwise.
            store_results: Whether to store results in the database
            max_workers: Max parallel threads (capped to session count)
            progress_callback: Called with (completed, total) after each session
            cancel_predicate: Optional callable; returns True when cancellation is requested

        Returns:
            BatchAnalysisResult with per-session outcomes and aggregate counts
        """
        from snore.database.session import session_scope  # noqa: PLC0415

        base_stmt = (
            select(
                models.Session.id.label("session_id"),
                models.Day.date.label("day_date"),
            )
            .join(models.Device, models.Session.device_id == models.Device.id)
            .join(models.Day, models.Session.day_id == models.Day.id)
            .where(self._profile_filter())
        )

        # Materialize the full list in a short-lived scope that closes before any
        # workers run.  No DB transaction may remain open across the batch loop —
        # a batch-long read snapshot pins the WAL and starves checkpoints.
        # Two scalars per row (session_id, day_date) stay tiny even at 10k sessions.
        rows: list[Any]
        async with session_scope() as _db:
            if session_ids is not None:
                # Unbounded caller list — chunk under the bind cap, then re-sort
                # by day_date to restore the single-query ORDER BY.
                rows = []
                for chunk in iter_id_chunks(session_ids):
                    rows.extend(
                        (
                            await _db.execute(
                                base_stmt.where(models.Session.id.in_(chunk))
                            )
                        ).all()
                    )
                rows.sort(key=lambda r: r.day_date)
            else:
                stmt = base_stmt
                if from_date:
                    stmt = stmt.where(models.Day.date >= from_date.date())
                if to_date:
                    stmt = stmt.where(models.Day.date <= to_date.date())
                rows = list((await _db.execute(stmt.order_by(models.Day.date))).all())

        matched_total = len(rows)  # COUNT from len(), no separate query needed
        if matched_total == 0:
            return BatchAnalysisResult(
                total=0, successful=0, failed=0, cancelled=0, results=[]
            )

        session_pairs: list[tuple[int, date | None]] = [
            (row.session_id, row.day_date) for row in rows
        ]

        coordinator = BatchAnalysisCoordinator()
        self._batch_coordinator = coordinator
        return await coordinator.submit(
            session_pairs=session_pairs,
            profile_id=self.profile_id,
            modes=modes,
            primary_mode=primary_mode,
            store_results=store_results,
            max_workers=max_workers,
            progress_callback=progress_callback,
            cancel_predicate=cancel_predicate,
        )


# ---------------------------------------------------------------------------
# Analysis store helper
# ---------------------------------------------------------------------------

# Backwards-compatible alias — the implementation lives in snore.analysis.runner.
_store_with_retry = store_with_retry


# ---------------------------------------------------------------------------
# BatchAnalysisCoordinator (§7)
# ---------------------------------------------------------------------------


class BatchAnalysisCoordinator:
    """Thin coordinator that backs batch analysis scheduling.

    Uses ``asyncio.to_thread()`` for per-session CPU-bound work (NumPy/scipy).
    Database reads and writes happen on the event loop through short-lived
    ``AsyncSession`` scopes; only detached ``RawSessionBlobs`` DTOs cross the
    thread boundary.  The ``submit`` interface is the stable boundary — callers
    are unchanged.
    """

    def __init__(self) -> None:
        self._cancel_requested = False
        self._cancel_predicate: Callable[[], bool] | None = None
        self._completed = 0
        self._total = 0
        self.session_dates: dict[int, Any] = {}  # sid → day_date; window-bounded

    def cancel(self) -> None:
        """Request cooperative cancellation.  Checked between sessions."""
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        """True if cancellation was requested via cancel() or the predicate fires."""
        return self._cancel_requested or (
            self._cancel_predicate is not None and self._cancel_predicate()
        )

    @property
    def progress(self) -> tuple[int, int]:
        """Return (completed, total) session counts."""
        return self._completed, self._total

    async def submit(
        self,
        *,
        session_pairs: Sequence[tuple[int, date | None]],
        profile_id: int,
        modes: Sequence[str] | None = None,
        primary_mode: str | None = None,
        store_results: bool = True,
        max_workers: int = 4,
        progress_callback: Callable[[int, int | None], None] | None = None,
        cancel_predicate: Callable[[], bool] | None = None,
    ) -> BatchAnalysisResult:
        """Execute batch analysis and return aggregated results.

        Each session is processed in three async phases: (1) read — fetch raw
        blobs on the event loop via short-lived ``AsyncSession`` scopes, (2)
        compute — run CPU-bound NumPy/scipy work in ``asyncio.to_thread()`` with
        only a detached ``RawSessionBlobs`` DTO crossing the thread boundary
        (zero DB access in thread), (3) write — persist the result on the event
        loop via ``AsyncSession``.

        Args:
            session_pairs: Materialized list of ``(session_id, day_date)`` tuples
                to process.  The list is built by the caller in a short-lived
                scope that closes before this method runs, so no DB transaction
                is held for the duration of the batch.
            profile_id: Profile that owns the sessions — threaded into each
                ``AnalysisService`` read/write scope so all I/O is scoped.
            modes: Detection modes to run (``None`` = default).
            primary_mode: Mode whose recovery markers are persisted; defaults to
                DEFAULT_MODE when included in ``modes``, required otherwise.
            store_results: If True, write each result to the DB.
            max_workers: Concurrency cap (number of simultaneous coroutines).
            progress_callback: Called with (completed, total) after each session.
            cancel_predicate: Optional callable queried before each session is
                dispatched; when it returns ``True`` no new sessions are started
                and all remaining pairs are drained as cancelled.  A
                ``cancel()`` call made before ``submit()`` is also honoured —
                the pre-set flag is never cleared on entry.

        Returns:
            Aggregated ``BatchAnalysisResult``.
        """
        matched_total = len(session_pairs)
        self._total = matched_total
        self._completed = 0
        self._cancel_predicate = cancel_predicate
        # Do NOT reset _cancel_requested if cancel() was called before submit().

        modes_list: list[str] | None = list(modes) if modes is not None else None

        # Plain iterator over the materialized list.  The sliding window enqueues
        # pairs up to max_workers; session_dates is pruned as tasks complete so
        # retained metadata stays window-bounded.
        pair_iter = iter(session_pairs)
        pairs_exhausted = False

        batch_results: list[BatchSessionResult] = []
        pending: dict[asyncio.Task[str], int] = {}  # task → sid
        session_dates = self.session_dates  # instance attribute for observability
        session_dates.clear()

        async def _run_one(sid: int) -> str:
            """Read → process-compute → write, all async."""
            if self._is_cancelled():
                return "cancelled"
            try:
                await run_one(
                    sid,
                    profile_id=profile_id,
                    modes=modes_list,
                    primary_mode=primary_mode,
                    store=store_results,
                    dispatch=Dispatch.PROCESS,
                )
                return "success"
            except Exception as e:
                logger.warning(
                    "Failed to analyze session %d: %s", sid, e, exc_info=True
                )
                return "error"

        def _fill_window() -> None:
            """Enqueue pairs until max_workers tasks are in flight or list exhausted."""
            nonlocal pairs_exhausted
            while len(pending) < max_workers and not pairs_exhausted:
                if self._is_cancelled():
                    break
                try:
                    sid, day_date = next(pair_iter)
                except StopIteration:
                    pairs_exhausted = True
                    break
                session_dates[sid] = day_date
                task: asyncio.Task[str] = asyncio.create_task(_run_one(sid))
                pending[task] = sid

        _fill_window()
        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                sid = pending.pop(task)
                outcome = task.result()
                cancelled = outcome == "cancelled"
                success = outcome == "success"
                error = (
                    None if outcome != "error" else f"Analysis failed for session {sid}"
                )
                batch_results.append(
                    BatchSessionResult(
                        session_id=sid,
                        session_date=session_dates.pop(sid, None),
                        success=success,
                        cancelled=cancelled,
                        error=error,
                    )
                )
                self._completed += 1
                if progress_callback:
                    progress_callback(self._completed, matched_total)
            # Refill the window after each completion.
            _fill_window()

        # Drain any pairs remaining in the list after cancellation.
        # Each unconsumed pair is counted as cancelled so `total` stays honest.
        if self._is_cancelled():
            for sid, day_date in pair_iter:
                batch_results.append(
                    BatchSessionResult(
                        session_id=sid,
                        session_date=day_date,
                        success=False,
                        cancelled=True,
                        error=None,
                    )
                )

        successful = sum(1 for r in batch_results if r.success)
        cancelled_count = sum(1 for r in batch_results if r.cancelled)
        return BatchAnalysisResult(
            total=matched_total,
            successful=successful,
            failed=matched_total - successful - cancelled_count,
            cancelled=cancelled_count,
            results=batch_results,
        )
