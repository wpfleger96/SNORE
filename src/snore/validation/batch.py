"""
Batch validation logic for running validation across multiple sessions.
"""

import logging

from datetime import datetime

from sqlalchemy.orm import Session

from snore.analysis.modes.config import AASM_CONFIG
from snore.analysis.modes.detector import EventDetector
from snore.analysis.service import AnalysisService
from snore.analysis.utils import convert_machine_events
from snore.database import models
from snore.validation.report import (
    AggregateMetrics,
    SessionValidation,
    ValidationReport,
)

logger = logging.getLogger(__name__)


class BatchValidator:
    """Runs validation across multiple sessions."""

    def __init__(self, db_session: Session, profile: str | None = None):
        """
        Initialize batch validator.

        Args:
            db_session: Database session
            profile: Profile name to filter sessions (optional)
        """
        self.db_session = db_session
        self.profile = profile
        self.analysis_service = AnalysisService(db_session)

    def validate_date_range(
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
        query = self.db_session.query(models.Session).filter(
            models.Session.start_time >= datetime.fromisoformat(date_from),
            models.Session.start_time <= datetime.fromisoformat(f"{date_to} 23:59:59"),
        )

        if self.profile:
            day_ids_query = (
                self.db_session.query(models.Day.id)
                .join(models.Profile)
                .filter(models.Profile.username == self.profile)
            )
            day_ids = [day_id for (day_id,) in day_ids_query.all()]
            query = query.filter(models.Session.day_id.in_(day_ids))

        sessions = query.order_by(models.Session.start_time).all()

        logger.info(f"Found {len(sessions)} sessions between {date_from} and {date_to}")

        session_validations = []

        for session in sessions:
            try:
                validation = self._validate_session(session.id, mode)
                if validation:
                    session_validations.append(validation)
            except Exception as e:
                logger.warning(f"Failed to validate session {session.id}: {e}")
                continue

        aggregate = self._calculate_aggregate(session_validations)

        return ValidationReport(
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            date_range_start=date_from,
            date_range_end=date_to,
            aggregate=aggregate,
            sessions=session_validations,
        )

    def _validate_session(self, session_id: int, mode: str) -> SessionValidation | None:
        """
        Validate a single session.

        Args:
            session_id: Session ID to validate
            mode: Detection mode

        Returns:
            SessionValidation or None if validation fails
        """
        session = self.db_session.get(models.Session, session_id)
        if not session:
            return None

        analysis_result = self.analysis_service.get_analysis_result(session_id)
        if not analysis_result:
            logger.info(f"Running analysis for session {session_id}...")
            analysis_result = self.analysis_service.analyze_session(
                session_id, modes=[mode]
            )

        if mode not in analysis_result.mode_results:
            logger.warning(
                f"Mode {mode} not found in analysis results for {session_id}"
            )
            return None

        mode_result = analysis_result.mode_results[mode]
        machine_events = analysis_result.machine_events

        machine_apneas, machine_hypopneas, _ = convert_machine_events(machine_events)

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
