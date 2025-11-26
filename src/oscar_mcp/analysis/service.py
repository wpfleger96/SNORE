"""
Analysis service for orchestrating programmatic analysis.

This module provides the main interface for running comprehensive CPAP session
analysis, loading data from the database, and storing results.
"""

import logging
import time
from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import Session

from oscar_mcp.analysis.data.waveform_loader import WaveformLoader
from oscar_mcp.analysis.engines.programmatic_engine import (
    ProgrammaticAnalysisEngine,
    ProgrammaticAnalysisResult,
)
from oscar_mcp.analysis.events import AnalysisEvent
from oscar_mcp.database import models

logger = logging.getLogger(__name__)


class AnalysisService:
    """
    Service for running programmatic analysis on CPAP sessions.

    This service handles:
    - Loading waveform data from database
    - Running the programmatic analysis engine
    - Storing results in the database
    - Providing structured results for consumption

    Example:
        >>> service = AnalysisService(db_session)
        >>> result = service.analyze_session(session_id=123)
        >>> print(f"AHI: {result['event_timeline']['ahi']}")
    """

    def __init__(self, db_session: Session):
        """
        Initialize analysis service.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db_session = db_session
        self.waveform_loader = WaveformLoader(db_session)
        self.engine = ProgrammaticAnalysisEngine()

    def _load_machine_events(self, session_id: int) -> List[AnalysisEvent]:
        """
        Load machine-flagged events from database.

        Args:
            session_id: Database session ID

        Returns:
            List of respiratory events flagged by the CPAP device
        """
        events = (
            self.db_session.query(models.Event)
            .filter_by(session_id=session_id)
            .order_by(models.Event.start_time)
            .all()
        )

        respiratory_events = []
        for event in events:
            start_timestamp = event.start_time.timestamp()

            respiratory_events.append(
                AnalysisEvent(
                    event_type=event.event_type,
                    start_time=start_timestamp,
                    duration=event.duration_seconds or 10.0,
                    source="machine",
                    confidence=1.0,
                )
            )

        return respiratory_events

    def analyze_session(
        self, session_id: int, store_results: bool = True
    ) -> ProgrammaticAnalysisResult:
        """
        Run comprehensive analysis on a single session.

        Args:
            session_id: Database session ID
            store_results: Whether to store results in database

        Returns:
            ProgrammaticAnalysisResult with complete analysis

        Raises:
            ValueError: If session not found or has no waveform data
        """
        logger.info(f"Starting analysis for session {session_id}")
        start_time = time.time()

        session = self.db_session.query(models.Session).filter_by(id=session_id).first()
        if not session:
            raise ValueError(f"Session {session_id} not found")

        try:
            timestamps, flow_values, metadata = self.waveform_loader.load_waveform(
                session_id=session_id, waveform_type="flow", apply_filter=False
            )
        except Exception as e:
            logger.error(f"Failed to load flow waveform for session {session_id}: {e}")
            raise ValueError(f"No flow waveform data available for session {session_id}")

        if len(timestamps) == 0:
            raise ValueError(f"Empty flow waveform data for session {session_id}")

        sample_rate = metadata.get("sample_rate", 25.0)
        logger.info(
            f"Loaded {len(timestamps)} flow samples at {sample_rate}Hz "
            f"({len(timestamps) / sample_rate / 3600:.1f} hours)"
        )

        machine_events = self._load_machine_events(session_id)
        logger.info(f"Loaded {len(machine_events)} machine-flagged events")

        spo2_values = None
        try:
            _, spo2_values, _ = self.waveform_loader.load_waveform(
                session_id=session_id, waveform_type="spo2", apply_filter=False
            )
            if len(spo2_values) > 0 and len(spo2_values) != len(timestamps):
                logger.warning(
                    f"SpO2 length mismatch ({len(spo2_values)} vs {len(timestamps)}), "
                    "will resample or skip"
                )
                spo2_values = None
            if spo2_values is not None:
                logger.info(f"Loaded SpO2 data: {len(spo2_values)} samples")
        except Exception as e:
            logger.info(f"No SpO2 data available: {e}")

        result = self.engine.analyze_session(
            session_id=session_id,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=sample_rate,
            spo2_values=spo2_values,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Analysis complete for session {session_id} in {processing_time_ms}ms. "
            f"AHI={result.event_timeline['ahi']:.1f}, "
            f"FLI={result.flow_analysis['fl_index']:.2f}"
        )

        if store_results:
            self._store_analysis_result(session_id, result, processing_time_ms, machine_events)

        result.machine_events = machine_events

        return result

    def analyze_sessions(
        self, session_ids: List[int], store_results: bool = True
    ) -> List[ProgrammaticAnalysisResult]:
        """
        Run analysis on multiple sessions.

        Args:
            session_ids: List of session IDs to analyze
            store_results: Whether to store results in database

        Returns:
            List of analysis results (may have fewer entries if some sessions fail)
        """
        results = []
        for session_id in session_ids:
            try:
                result = self.analyze_session(session_id, store_results=store_results)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze session {session_id}: {e}")
                continue

        return results

    def get_analysis_result(self, session_id: int) -> Optional[dict]:
        """
        Retrieve stored analysis result for a session.

        Args:
            session_id: Database session ID

        Returns:
            Analysis result dictionary or None if not found
        """
        analysis = (
            self.db_session.query(models.AnalysisResult)
            .filter_by(session_id=session_id)
            .order_by(models.AnalysisResult.created_at.desc())
            .first()
        )

        if not analysis:
            return None

        return {
            "analysis_id": analysis.id,
            "session_id": analysis.session_id,
            "timestamp_start": analysis.timestamp_start.isoformat(),
            "timestamp_end": analysis.timestamp_end.isoformat(),
            "programmatic_result": analysis.programmatic_result_json,
            "processing_time_ms": analysis.processing_time_ms,
            "created_at": analysis.created_at.isoformat(),
        }

    def _store_analysis_result(
        self,
        session_id: int,
        result: ProgrammaticAnalysisResult,
        processing_time_ms: int,
        machine_events: List[AnalysisEvent],
    ) -> None:
        """
        Store analysis result in database.

        Args:
            session_id: Database session ID
            result: Analysis result from engine
            processing_time_ms: Processing time in milliseconds
            machine_events: Machine-flagged events from device
        """
        timestamp_start = datetime.fromtimestamp(result.timestamp_start)
        timestamp_end = datetime.fromtimestamp(result.timestamp_end)

        machine_events_json = [
            {
                "event_type": event.event_type,
                "start_time": event.start_time,
                "duration": event.duration,
                "source": event.source,
            }
            for event in machine_events
        ]

        programmatic_json = {
            "flow_analysis": result.flow_analysis,
            "event_timeline": result.event_timeline,
            "csr_detection": result.csr_detection,
            "periodic_breathing": result.periodic_breathing,
            "positional_analysis": result.positional_analysis,
            "total_breaths": result.total_breaths,
            "confidence_summary": result.confidence_summary,
            "clinical_summary": result.clinical_summary,
            "machine_events": machine_events_json,
        }

        analysis = models.AnalysisResult(
            session_id=session_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            programmatic_result_json=programmatic_json,
            processing_time_ms=processing_time_ms,
            engine_versions_json={
                "programmatic_engine": "1.0.0",
                "phase": "4",
            },
        )

        self.db_session.add(analysis)
        self.db_session.commit()

        self._store_detected_patterns(analysis.id, result)

        logger.info(f"Stored analysis result with ID {analysis.id}")

    def _store_detected_patterns(
        self, analysis_id: int, result: ProgrammaticAnalysisResult
    ) -> None:
        """
        Store individual detected patterns/events.

        Args:
            analysis_id: Database analysis result ID
            result: Analysis result containing patterns
        """
        patterns = []

        if result.event_timeline:
            timeline = result.event_timeline

            for apnea in timeline.get("apneas", []):
                patterns.append(
                    models.DetectedPattern(
                        analysis_result_id=analysis_id,
                        pattern_id=f"APNEA_{apnea['event_type']}",
                        start_time=datetime.fromtimestamp(apnea["start_time"]),
                        duration=apnea["duration"],
                        confidence=apnea["confidence"],
                        detected_by="programmatic",
                        metrics_json={
                            "event_type": apnea["event_type"],
                            "flow_reduction": apnea["flow_reduction"],
                            "baseline_flow": apnea["baseline_flow"],
                        },
                    )
                )

            for hypopnea in timeline.get("hypopneas", []):
                patterns.append(
                    models.DetectedPattern(
                        analysis_result_id=analysis_id,
                        pattern_id="HYPOPNEA",
                        start_time=datetime.fromtimestamp(hypopnea["start_time"]),
                        duration=hypopnea["duration"],
                        confidence=hypopnea["confidence"],
                        detected_by="programmatic",
                        metrics_json={
                            "flow_reduction": hypopnea["flow_reduction"],
                            "has_arousal": hypopnea["has_arousal"],
                            "has_desaturation": hypopnea["has_desaturation"],
                        },
                    )
                )

            for rera in timeline.get("reras", []):
                patterns.append(
                    models.DetectedPattern(
                        analysis_result_id=analysis_id,
                        pattern_id="RERA",
                        start_time=datetime.fromtimestamp(rera["start_time"]),
                        duration=rera["duration"],
                        confidence=rera["confidence"],
                        detected_by="programmatic",
                        metrics_json={
                            "flatness_index": rera["flatness_index"],
                            "terminated_by_arousal": rera["terminated_by_arousal"],
                        },
                    )
                )

        if result.csr_detection:
            csr = result.csr_detection
            patterns.append(
                models.DetectedPattern(
                    analysis_result_id=analysis_id,
                    pattern_id="CSR",
                    start_time=datetime.fromtimestamp(csr["start_time"]),
                    duration=csr["end_time"] - csr["start_time"],
                    confidence=csr["confidence"],
                    detected_by="programmatic",
                    metrics_json={
                        "cycle_length": csr["cycle_length"],
                        "amplitude_variation": csr["amplitude_variation"],
                        "csr_index": csr["csr_index"],
                        "cycle_count": csr["cycle_count"],
                    },
                )
            )

        if result.periodic_breathing:
            periodic = result.periodic_breathing
            patterns.append(
                models.DetectedPattern(
                    analysis_result_id=analysis_id,
                    pattern_id="PERIODIC_BREATHING",
                    start_time=datetime.fromtimestamp(periodic["start_time"]),
                    duration=periodic["end_time"] - periodic["start_time"],
                    confidence=periodic["confidence"],
                    detected_by="programmatic",
                    metrics_json={
                        "cycle_length": periodic["cycle_length"],
                        "regularity_score": periodic["regularity_score"],
                        "has_apneas": periodic["has_apneas"],
                    },
                )
            )

        if patterns:
            self.db_session.bulk_save_objects(patterns)
            self.db_session.commit()
            logger.info(f"Stored {len(patterns)} detected patterns")
