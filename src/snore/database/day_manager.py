"""
Day aggregation and management logic (OSCAR-compatible).

Handles day splitting logic and aggregation of session statistics into daily records.
"""

from datetime import date, datetime, time, timedelta

from sqlalchemy.orm import Session

from snore.database.models import Day, Profile
from snore.database.models import Session as SessionModel


class DayManager:
    """Manages day splitting and aggregation logic (OSCAR-compatible)."""

    # Default day split time (noon, like OSCAR)
    DEFAULT_SPLIT_TIME = time(12, 0)

    @classmethod
    def get_day_for_session(cls, session_start: datetime, profile: Profile) -> date:
        """
        Determine which calendar day a session belongs to based on split time.

        OSCAR logic: Sessions before the split time belong to the previous day.
        Default split time is 12:00 noon.

        Args:
            session_start: Session start datetime
            profile: Profile with settings containing optional day_split_time

        Returns:
            Date the session belongs to
        """
        split_time = cls.DEFAULT_SPLIT_TIME
        if profile.settings and isinstance(profile.settings, dict):
            settings_split = profile.settings.get("day_split_time")
            if settings_split:
                if isinstance(settings_split, str):
                    hour, minute, *_ = map(int, settings_split.split(":") + [0])
                    split_time = time(hour, minute)
                elif isinstance(settings_split, time):
                    split_time = settings_split

        if session_start.time() < split_time:
            return session_start.date() - timedelta(days=1)
        return session_start.date()

    @classmethod
    def create_or_update_day(
        cls,
        profile_id: int,
        day_date: date,
        db_session: Session,
    ) -> Day:
        """
        Create or update a day record with aggregated statistics from all sessions.

        Args:
            profile_id: Profile ID
            day_date: Date for the day record
            db_session: SQLAlchemy database session

        Returns:
            Updated Day object
        """
        day = (
            db_session.query(Day)
            .filter_by(profile_id=profile_id, date=day_date)
            .first()
        )

        if not day:
            day = Day(profile_id=profile_id, date=day_date)
            db_session.add(day)
            db_session.flush()  # Get day.id for session linking

        cls._aggregate_day_statistics(day, db_session)

        return day

    @classmethod
    def _aggregate_day_statistics(cls, day: Day, db_session: Session) -> None:
        """
        Aggregate statistics from all sessions belonging to a day.

        Args:
            day: Day object to update
            db_session: SQLAlchemy database session
        """
        sessions = db_session.query(SessionModel).filter_by(day_id=day.id).all()

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
            day.pressure_min = None
            day.pressure_max = None
            day.pressure_median = None
            day.pressure_mean = None
            day.pressure_95th = None
            day.leak_min = None
            day.leak_max = None
            day.leak_median = None
            day.leak_mean = None
            day.leak_95th = None
            day.spo2_min = None
            day.spo2_max = None
            day.spo2_mean = None
            day.spo2_avg = None
            return

        day.session_count = len(sessions)

        total_seconds = sum(s.duration_seconds for s in sessions if s.duration_seconds)
        day.total_therapy_hours = total_seconds / 3600.0 if total_seconds else 0.0

        stats_records = [s.statistics for s in sessions if s.statistics]

        if stats_records:
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
                ahi_values = [
                    (s.ahi, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.ahi is not None and sess.duration_seconds
                ]
                if ahi_values:
                    day.ahi = sum(ahi * weight for ahi, weight in ahi_values) / sum(
                        weight for _, weight in ahi_values
                    )

                oai_values = [
                    (s.oai, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.oai is not None and sess.duration_seconds
                ]
                if oai_values:
                    day.oai = sum(oai * weight for oai, weight in oai_values) / sum(
                        weight for _, weight in oai_values
                    )

                cai_values = [
                    (s.cai, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.cai is not None and sess.duration_seconds
                ]
                if cai_values:
                    day.cai = sum(cai * weight for cai, weight in cai_values) / sum(
                        weight for _, weight in cai_values
                    )

                hi_values = [
                    (s.hi, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.hi is not None and sess.duration_seconds
                ]
                if hi_values:
                    day.hi = sum(hi * weight for hi, weight in hi_values) / sum(
                        weight for _, weight in hi_values
                    )

            pressure_mins = [s.pressure_min for s in stats_records if s.pressure_min]
            pressure_maxs = [s.pressure_max for s in stats_records if s.pressure_max]
            day.pressure_min = min(pressure_mins) if pressure_mins else None
            day.pressure_max = max(pressure_maxs) if pressure_maxs else None

            if total_hours > 0:
                median_values = [
                    (s.pressure_median, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.pressure_median is not None and sess.duration_seconds
                ]
                if median_values:
                    day.pressure_median = sum(
                        med * weight for med, weight in median_values
                    ) / sum(weight for _, weight in median_values)

                mean_values = [
                    (s.pressure_mean, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.pressure_mean is not None and sess.duration_seconds
                ]
                if mean_values:
                    day.pressure_mean = sum(
                        mean * weight for mean, weight in mean_values
                    ) / sum(weight for _, weight in mean_values)

                p95_values = [
                    (s.pressure_95th, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.pressure_95th is not None and sess.duration_seconds
                ]
                if p95_values:
                    day.pressure_95th = sum(
                        p95 * weight for p95, weight in p95_values
                    ) / sum(weight for _, weight in p95_values)

            leak_mins = [s.leak_min for s in stats_records if s.leak_min]
            leak_maxs = [s.leak_max for s in stats_records if s.leak_max]
            day.leak_min = min(leak_mins) if leak_mins else None
            day.leak_max = max(leak_maxs) if leak_maxs else None

            if total_hours > 0:
                leak_median_values = [
                    (s.leak_median, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.leak_median is not None and sess.duration_seconds
                ]
                if leak_median_values:
                    day.leak_median = sum(
                        leak * weight for leak, weight in leak_median_values
                    ) / sum(weight for _, weight in leak_median_values)

                leak_mean_values = [
                    (s.leak_mean, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.leak_mean is not None and sess.duration_seconds
                ]
                if leak_mean_values:
                    day.leak_mean = sum(
                        leak * weight for leak, weight in leak_mean_values
                    ) / sum(weight for _, weight in leak_mean_values)

                leak_95_values = [
                    (s.leak_95th, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.leak_95th is not None and sess.duration_seconds
                ]
                if leak_95_values:
                    day.leak_95th = sum(
                        leak * weight for leak, weight in leak_95_values
                    ) / sum(weight for _, weight in leak_95_values)

            spo2_mins = [s.spo2_min for s in stats_records if s.spo2_min]
            spo2_maxs = [s.spo2_max for s in stats_records if s.spo2_max]
            day.spo2_min = min(spo2_mins) if spo2_mins else None
            day.spo2_max = max(spo2_maxs) if spo2_maxs else None

            if total_hours > 0:
                spo2_mean_values = [
                    (s.spo2_mean, sess.duration_seconds / 3600)
                    for s, sess in zip(stats_records, sessions, strict=False)
                    if s.spo2_mean is not None and sess.duration_seconds
                ]
                if spo2_mean_values:
                    day.spo2_mean = sum(
                        spo2 * weight for spo2, weight in spo2_mean_values
                    ) / sum(weight for _, weight in spo2_mean_values)
                    day.spo2_avg = day.spo2_mean  # Alias for compatibility

    @classmethod
    def link_session_to_day(
        cls,
        session: SessionModel,
        profile: Profile,
        db_session: Session,
    ) -> Day:
        """
        Link a session to its appropriate day record based on day-splitting logic.

        Creates or updates the day record with aggregated statistics.

        Args:
            session: Session to link
            profile: Profile the session belongs to
            db_session: SQLAlchemy database session

        Returns:
            Day object the session was linked to
        """
        day_date = cls.get_day_for_session(session.start_time, profile)

        day = cls.create_or_update_day(profile.id, day_date, db_session)

        session.day_id = day.id

        cls._aggregate_day_statistics(day, db_session)

        return day

    @classmethod
    def recalculate_all_days_for_profile(
        cls, profile_id: int, db_session: Session
    ) -> int:
        """
        Recalculate all day records for a profile.

        Useful after bulk session imports or data corrections.

        Args:
            profile_id: Profile ID to recalculate
            db_session: SQLAlchemy database session

        Returns:
            Number of day records updated
        """
        days = db_session.query(Day).filter_by(profile_id=profile_id).all()

        for day in days:
            cls._aggregate_day_statistics(day, db_session)

        return len(days)
