"""
Batch validation logic for running validation across multiple sessions.
"""

import logging

from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.modes.config import AASM_CONFIG
from snore.analysis.modes.detector import EventDetector
from snore.analysis.utils import convert_machine_events
from snore.database import models
from snore.validation.report import (
    AggregateMetrics,
    CrossParserSameDay,
    IntegrityReport,
    OverlappingSessionPair,
    SessionValidation,
    ValidationReport,
)
from snore.validation.session_scoping import work_session

logger = logging.getLogger(__name__)


class BatchValidator:
    """Runs validation across multiple sessions."""

    def __init__(self, db_session: AsyncSession | None, profile_id: int):
        """
        Initialize batch validator.

        Args:
            db_session: Async database session, or None to run in JOB mode where
                work_session opens a fresh short scope per session so the WAL
                read snapshot is released between sessions.  A real session runs
                every unit of work on it — one transaction — as before.
            profile_id: Profile ID to scope all queries — required, never global.
        """
        self._injected = db_session
        self.db_session: AsyncSession = db_session  # type: ignore[assignment]
        self.profile_id = profile_id

    async def validate_date_range(
        self,
        date_from: str,
        date_to: str,
        mode: str = "aasm",
    ) -> ValidationReport:
        """
        Run validation across a date range.

        Args:
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            mode: Detection mode to validate (default: aasm)

        Returns:
            ValidationReport with aggregate and per-session metrics
        """
        stmt = (
            select(models.Session)
            .join(models.Device, models.Session.device_id == models.Device.id)
            .where(
                models.Device.profile_id == self.profile_id,
                models.Session.start_time >= datetime.fromisoformat(date_from),
                models.Session.start_time
                <= datetime.fromisoformat(f"{date_to} 23:59:59"),
            )
        )

        # Bulk-fetch the session set and its Statistics rows in one short scope.
        async with work_session(self._injected) as db:
            self.db_session = db
            sessions = (
                (await db.execute(stmt.order_by(models.Session.start_time)))
                .scalars()
                .all()
            )
            logger.info(
                f"Found {len(sessions)} sessions between {date_from} and {date_to}"
            )
            session_ids = [s.id for s in sessions]
            stats_by_session_id: dict[int, models.Statistics] = {}
            if session_ids:
                stat_rows = (
                    (
                        await db.execute(
                            select(models.Statistics).where(
                                models.Statistics.session_id.in_(session_ids)
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                stats_by_session_id = {int(r.session_id): r for r in stat_rows}

        session_validations = []

        for session_id in session_ids:
            try:
                async with work_session(self._injected) as db:
                    self.db_session = db
                    validation = await self._validate_session(
                        session_id, mode, stats_by_session_id
                    )
                if validation:
                    session_validations.append(validation)
            except Exception as e:
                logger.warning(f"Failed to validate session {session_id}: {e}")
                continue

        aggregate = self._calculate_aggregate(session_validations)

        return ValidationReport(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_range_start=date_from,
            date_range_end=date_to,
            aggregate=aggregate,
            sessions=session_validations,
        )

    async def _validate_session(
        self,
        session_id: int,
        mode: str,
        stats_by_session_id: dict[int, models.Statistics],
    ) -> SessionValidation | None:
        """
        Validate a single session.

        Args:
            session_id: Session ID to validate
            mode: Detection mode
            stats_by_session_id: Pre-fetched Statistics rows keyed by session_id;
                avoids a per-session query inside the validation loop.

        Returns:
            SessionValidation or None if validation fails
        """
        # Use an already-scoped facade so ownership is enforced on every path.
        from snore.services.analysis_facade import AnalysisFacade  # noqa: PLC0415

        facade = AnalysisFacade(self.db_session, self.profile_id)

        session = await self.db_session.get(models.Session, session_id)
        if not session:
            return None

        analysis_result = await facade.get_analysis_result(session_id)
        if not analysis_result:
            logger.info(f"Running analysis for session {session_id}...")
            analysis_result = await facade.run_analysis(session_id, modes=[mode])

        if mode not in analysis_result.mode_results:
            logger.warning(
                f"Mode {mode} not found in analysis results for {session_id}"
            )
            return None

        mode_result = analysis_result.mode_results[mode]
        machine_events = analysis_result.machine_events

        machine_apneas, machine_hypopneas = convert_machine_events(machine_events)

        # RERA validation is intentionally omitted here: this report's per-session
        # and aggregate surfaces (and the low-sensitivity gate) are apnea/hypopnea
        # only, and RERA's "no_machine_re_events" status has no home in that model.
        # Per-session RERA validation is surfaced in the single-session analysis
        # display instead. Passing RERAs would compute matches that go unread.
        detector = EventDetector(AASM_CONFIG)
        validation = detector.validate_against_machine_events(
            mode_result.apneas,
            mode_result.hypopneas,
            machine_apneas,
            machine_hypopneas,
        )

        apnea_val = validation["apnea_validation"]
        hypopnea_val = validation["hypopnea_validation"]

        notes = None
        if apnea_val.sensitivity < 0.6 or hypopnea_val.sensitivity < 0.6:
            notes = "Low sensitivity - investigate this session"

        # Device-reported nightly indices from pre-fetched Statistics map.
        # Null-safe: APAP and vAuto record different index subsets.
        stats_row = stats_by_session_id.get(session_id)

        return SessionValidation(
            session_id=session_id,
            date=session.start_time.strftime("%Y-%m-%d"),
            duration_hours=analysis_result.session_duration_hours,
            machine_event_count=len(machine_events),
            programmatic_event_count=len(mode_result.apneas)
            + len(mode_result.hypopneas),
            apnea_sensitivity=apnea_val.sensitivity,
            apnea_precision=apnea_val.precision,
            apnea_f1=apnea_val.f1_score,
            hypopnea_sensitivity=hypopnea_val.sensitivity,
            hypopnea_precision=hypopnea_val.precision,
            hypopnea_f1=hypopnea_val.f1_score,
            notes=notes,
            device_ahi=stats_row.ahi_device if stats_row is not None else None,
            device_oai=stats_row.oai_device if stats_row is not None else None,
            device_cai=stats_row.cai_device if stats_row is not None else None,
            device_hi=stats_row.hi_device if stats_row is not None else None,
            device_uai=stats_row.uai if stats_row is not None else None,
        )

    async def check_data_integrity(
        self, device_id: int | None = None
    ) -> IntegrityReport:
        """Check structural data integrity of the session database.

        Three checks:
        1. Sessions where day_id IS NULL (never linked to a Day row).
        2. Same-device session pairs whose time ranges strictly overlap.
        3. Device+day combinations with sessions from more than one import source.

        Args:
            device_id: Restrict checks to a specific device, or None for all devices
                belonging to this profile.

        Returns:
            IntegrityReport with all detected issues and a total_issues count.
        """
        # Base device filter: all devices owned by this profile.
        device_scope = select(models.Device.id).where(
            models.Device.profile_id == self.profile_id
        )
        if device_id is not None:
            device_scope = device_scope.where(models.Device.id == device_id)

        # --- Check 1: sessions with day_id IS NULL ---
        null_day_stmt = select(models.Session.id).where(
            models.Session.device_id.in_(device_scope),
            models.Session.day_id.is_(None),
        )
        null_day_ids = list(
            (await self.db_session.execute(null_day_stmt)).scalars().all()
        )

        # --- Check 2: overlapping session pairs (self-join) ---
        a = models.Session.__table__.alias("a")
        b = models.Session.__table__.alias("b")
        overlap_stmt = (
            select(
                a.c.id.label("a_id"),
                b.c.id.label("b_id"),
                a.c.device_id.label("device_id"),
                a.c.device_session_id.label("a_dsid"),
                b.c.device_session_id.label("b_dsid"),
                a.c.start_time.label("a_start"),
                a.c.end_time.label("a_end"),
                b.c.start_time.label("b_start"),
                b.c.end_time.label("b_end"),
            )
            .where(
                a.c.id < b.c.id,
                a.c.device_id == b.c.device_id,
                a.c.device_id.in_(device_scope),
                # Strict overlap: neither adjacent segments (a.end == b.start) nor
                # non-overlapping order qualify.
                a.c.start_time < b.c.end_time,
                a.c.end_time > b.c.start_time,
            )
            .order_by(a.c.device_id, a.c.start_time)
        )
        overlap_rows = (await self.db_session.execute(overlap_stmt)).all()
        overlapping_pairs = [
            OverlappingSessionPair(
                session_a_id=row.a_id,
                session_b_id=row.b_id,
                device_id=row.device_id,
                session_a_device_session_id=row.a_dsid,
                session_b_device_session_id=row.b_dsid,
                session_a_start=row.a_start,
                session_a_end=row.a_end,
                session_b_start=row.b_start,
                session_b_end=row.b_end,
            )
            for row in overlap_rows
        ]

        # --- Check 3: same device + same Day with >1 distinct import_source ---
        cross_stmt = (
            select(
                models.Session.device_id,
                models.Day.date,
                # SQLite group_concat(DISTINCT col, sep) is unsupported; use the
                # single-arg form and deduplicate the comma-separated result in
                # Python.  import_source values never contain commas.
                func.group_concat(models.Session.import_source).label("sources"),
            )
            .join(models.Day, models.Session.day_id == models.Day.id)
            .where(
                models.Session.device_id.in_(device_scope),
                models.Session.import_source.isnot(None),
            )
            .group_by(models.Session.device_id, models.Day.date)
            .having(func.count(models.Session.import_source.distinct()) > 1)
            .order_by(models.Session.device_id, models.Day.date)
        )
        cross_rows = (await self.db_session.execute(cross_stmt)).all()
        cross_parser = [
            CrossParserSameDay(
                device_id=row.device_id,
                day_date=row.date,
                import_sources=sorted({s for s in row.sources.split(",") if s}),
            )
            for row in cross_rows
        ]

        total_issues = len(null_day_ids) + len(overlapping_pairs) + len(cross_parser)

        return IntegrityReport(
            checked_at=datetime.now(UTC),
            device_id_filter=device_id,
            null_day_id_sessions=null_day_ids,
            overlapping_session_pairs=overlapping_pairs,
            cross_parser_same_day=cross_parser,
            total_issues=total_issues,
        )

    def _calculate_aggregate(
        self, sessions: list[SessionValidation]
    ) -> AggregateMetrics:
        """
        Calculate aggregate metrics across sessions.

        Args:
            sessions: List of session validations

        Returns:
            AggregateMetrics
        """
        if not sessions:
            return AggregateMetrics(
                total_sessions=0,
                total_machine_events=0,
                total_programmatic_events=0,
                avg_apnea_sensitivity=0.0,
                avg_apnea_precision=0.0,
                avg_apnea_f1=0.0,
                avg_hypopnea_sensitivity=0.0,
                avg_hypopnea_precision=0.0,
                avg_hypopnea_f1=0.0,
                low_sensitivity_sessions=[],
            )

        total_machine = sum(s.machine_event_count for s in sessions)
        total_prog = sum(s.programmatic_event_count for s in sessions)

        avg_apnea_sens = sum(s.apnea_sensitivity for s in sessions) / len(sessions)
        avg_apnea_prec = sum(s.apnea_precision for s in sessions) / len(sessions)
        avg_apnea_f1 = sum(s.apnea_f1 for s in sessions) / len(sessions)

        avg_hypopnea_sens = sum(s.hypopnea_sensitivity for s in sessions) / len(
            sessions
        )
        avg_hypopnea_prec = sum(s.hypopnea_precision for s in sessions) / len(sessions)
        avg_hypopnea_f1 = sum(s.hypopnea_f1 for s in sessions) / len(sessions)

        low_sens_sessions = [
            s.session_id
            for s in sessions
            if s.apnea_sensitivity < 0.6 or s.hypopnea_sensitivity < 0.6
        ]

        return AggregateMetrics(
            total_sessions=len(sessions),
            total_machine_events=total_machine,
            total_programmatic_events=total_prog,
            avg_apnea_sensitivity=avg_apnea_sens,
            avg_apnea_precision=avg_apnea_prec,
            avg_apnea_f1=avg_apnea_f1,
            avg_hypopnea_sensitivity=avg_hypopnea_sens,
            avg_hypopnea_precision=avg_hypopnea_prec,
            avg_hypopnea_f1=avg_hypopnea_f1,
            low_sensitivity_sessions=low_sens_sessions,
        )
