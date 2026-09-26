"""
Day aggregation and management logic (OSCAR-compatible).

Handles day splitting logic and aggregation of session statistics into daily records.
"""

from datetime import date, datetime, time, timedelta

from sqlalchemy import exists, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from snore.database.models import Day, Statistics
from snore.database.models import Session as SessionModel
from snore.metrics import DAY_METRIC_STAT_COLUMNS, DayAgg
from snore.therapy_hours import TherapyHoursBasis, therapy_hours
from snore.utils.stats import weighted_mean


class DayManager:
    """Manages day splitting and aggregation logic (OSCAR-compatible)."""

    DEFAULT_SPLIT_TIME = time(12, 0)

    @classmethod
    def get_day_for_session(cls, session_start: datetime) -> date:
        """
        Determine which calendar day a session belongs to based on split time.

        OSCAR logic: Sessions before the split time belong to the previous day.
        Split time is hardcoded to 12:00 noon.

        Args:
            session_start: Session start datetime

        Returns:
            Date the session belongs to
        """
        if session_start.time() < cls.DEFAULT_SPLIT_TIME:
            return session_start.date() - timedelta(days=1)
        return session_start.date()

    @classmethod
    async def get_or_create_day(
        cls,
        device_id: int,
        day_date: date,
        db_session: AsyncSession,
    ) -> Day:
        """
        Get or create day WITHOUT triggering aggregation.

        Use this when you want to defer aggregation (e.g., batch imports)
        or when aggregation will be handled separately.

        Args:
            device_id: Device ID
            day_date: Date for the day record
            db_session: SQLAlchemy async database session

        Returns:
            Day object (without aggregated statistics)
        """
        day = (
            (
                await db_session.execute(
                    select(Day).filter_by(device_id=device_id, date=day_date)
                )
            )
            .scalars()
            .first()
        )

        if not day:
            day = Day(device_id=device_id, date=day_date)
            db_session.add(day)
            await db_session.flush()

        return day

    @classmethod
    async def create_or_update_day(
        cls,
        device_id: int,
        day_date: date,
        db_session: AsyncSession,
    ) -> Day:
        """
        Create or update a day record with aggregated statistics from all sessions.

        Args:
            device_id: Device ID
            day_date: Date for the day record
            db_session: SQLAlchemy async database session

        Returns:
            Updated Day object
        """
        day = await cls.get_or_create_day(device_id, day_date, db_session)
        await cls.aggregate_day_statistics(day, db_session)
        return day

    @staticmethod
    def _effective_session_hours(
        stats: Statistics | None, session: SessionModel
    ) -> float:
        """Effective therapy hours for a session.

        Prefers statistics.usage_hours (actual mask-on time) over session span.
        usage_hours == 0.0 is treated as known-zero; no span fallback applies.
        """
        if stats is not None and stats.usage_hours is not None:
            return stats.usage_hours
        return (
            therapy_hours(
                TherapyHoursBasis.SESSION_SPAN, span_seconds=session.duration_seconds
            )
            or 0.0
        )

    @classmethod
    def _weighted_average(
        cls,
        stat_pairs: list[tuple[Statistics, SessionModel]],
        attr: str,
    ) -> float | None:
        """Calculate usage-weighted average for a statistic across sessions.

        stat_pairs must be pre-aligned: each tuple is the Statistics row and its
        owning SessionModel. Weights prefer usage_hours over session span; entries
        with zero effective hours contribute nothing (weighted_mean drops zero
        weight), so they cannot distort the average.
        """
        return weighted_mean(
            (getattr(s, attr), cls._effective_session_hours(s, sess))
            for s, sess in stat_pairs
            if getattr(s, attr) is not None
        )

    @classmethod
    async def aggregate_day_statistics(cls, day: Day, db_session: AsyncSession) -> None:
        """
        Aggregate statistics from all sessions belonging to a day.

        Args:
            day: Day object to update
            db_session: SQLAlchemy async database session
        """
        sessions = (
            (
                await db_session.execute(
                    select(SessionModel)
                    .filter_by(day_id=day.id, enabled=True)
                    .options(joinedload(SessionModel.statistics))
                )
            )
            .scalars()
            .all()
        )

        if not sessions:
            day.session_count = 0
            day.total_therapy_hours = 0.0
            day.obstructive_apneas = 0
            day.central_apneas = 0
            day.hypopneas = 0
            day.reras = 0
            day.ahi = None
            day.oai = None
            day.cai = None
            day.hi = None
            for spec in DAY_METRIC_STAT_COLUMNS:
                setattr(day, spec.name, None)
            return

        day.session_count = len(sessions)

        day.total_therapy_hours = sum(
            cls._effective_session_hours(s.statistics, s) for s in sessions
        )

        # Pre-aligned pairs: each Statistics row with its owning session.
        # Building this once avoids the zip-misalignment bug that occurs when
        # not every session has a Statistics row.
        stat_pairs: list[tuple[Statistics, SessionModel]] = [
            (s.statistics, s) for s in sessions if s.statistics
        ]
        stats_records = [s for s, _ in stat_pairs]

        if stat_pairs:
            day.obstructive_apneas = sum(
                s.obstructive_apneas for s in stats_records if s.obstructive_apneas
            )
            day.central_apneas = sum(
                s.central_apneas for s in stats_records if s.central_apneas
            )
            day.hypopneas = sum(s.hypopneas for s in stats_records if s.hypopneas)
            day.reras = sum(s.reras for s in stats_records if s.reras)

            total_hours = day.total_therapy_hours
            if total_hours > 0:
                day.ahi = cls._weighted_average(stat_pairs, "ahi")
                day.oai = cls._weighted_average(stat_pairs, "oai")
                day.cai = cls._weighted_average(stat_pairs, "cai")
                day.hi = cls._weighted_average(stat_pairs, "hi")

            for spec in DAY_METRIC_STAT_COLUMNS:
                if spec.day_agg in (DayAgg.MIN, DayAgg.MAX):
                    # Truthiness filter is intentional-legacy: 0.0 values are
                    # excluded from min/max, matching historical behavior.
                    values = [
                        getattr(s, spec.name)
                        for s in stats_records
                        if getattr(s, spec.name)
                    ]
                    reduce = min if spec.day_agg is DayAgg.MIN else max
                    setattr(day, spec.name, reduce(values) if values else None)
                elif spec.day_agg is DayAgg.USAGE_WEIGHTED_MEAN and total_hours > 0:
                    setattr(
                        day, spec.name, cls._weighted_average(stat_pairs, spec.name)
                    )

    @classmethod
    async def link_session_to_day(
        cls,
        session: SessionModel,
        device_id: int,
        db_session: AsyncSession,
    ) -> Day:
        """
        Link a session to its appropriate day record based on day-splitting logic.

        Creates or updates the day record with aggregated statistics.

        Args:
            session: Session to link
            device_id: Device ID the session belongs to
            db_session: SQLAlchemy async database session

        Returns:
            Day object the session was linked to
        """
        day_date = cls.get_day_for_session(session.start_time)

        day = await cls.get_or_create_day(device_id, day_date, db_session)

        session.day_id = day.id

        await cls.aggregate_day_statistics(day, db_session)

        return day

    @classmethod
    async def recalculate_day(cls, day: Day, db_session: AsyncSession) -> bool:
        """
        Recalculate a day after its session membership changed.

        Lifecycle rule: after any membership change a Day row survives only if
        at least one Session row, enabled or disabled, still references it.
        Deleting the last such session orphans the day and the row is pruned
        here (re-import recreates it via ``link_session_to_day``).  Disabling
        the last enabled session does not orphan the day: the disabled Session
        still points at it through the composite ``(day_id, device_id)`` FK, so
        the row stays with ``session_count == 0`` and reset aggregates.

        Args:
            day: Day object to recalculate
            db_session: SQLAlchemy async database session

        Returns:
            True if the day still exists, False if it was pruned.
        """
        # The existence probe is a Core statement and does not trigger
        # autoflush, so flush first: a pending Session that references this
        # day would otherwise be invisible and the day deleted from under it.
        await db_session.flush()
        referenced = await db_session.scalar(
            select(exists().where(SessionModel.day_id == day.id))
        )
        if not referenced:
            await db_session.delete(day)
            await db_session.flush()
            return False
        await cls.aggregate_day_statistics(day, db_session)
        return True
