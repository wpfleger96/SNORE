"""
Analysis service for orchestrating programmatic analysis.

This module provides the main interface for running comprehensive CPAP session
analysis, loading data from the database, and storing results.
"""

import logging
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from sqlalchemy.orm import Session

from snore.analysis.data.waveform_loader import WaveformLoader
from snore.analysis.engines.programmatic_engine import (
    ProgrammaticAnalysisEngine,
    ProgrammaticAnalysisResult,
)
from snore.analysis.events import AnalysisEvent
from snore.analysis.modes import DEFAULT_MODE, get_mode
from snore.analysis.modes.base import ModeResult
from snore.database import models

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Results from session analysis."""

    session_id: int
    session_duration_hours: float
    total_breaths: int
    machine_events: list[AnalysisEvent]
    mode_results: dict[str, ModeResult]
    flow_analysis: dict[str, Any] | None = None
    positional_analysis: dict[str, Any] | None = None
    timestamp_start: float = 0.0  # For storage compatibility
    timestamp_end: float = 0.0


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

    def _load_machine_events(self, session_id: int) -> list[AnalysisEvent]:
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

    def _debug_compare_events(
        self,
        machine_events: list[AnalysisEvent],
        breaths: list[Any],
        reductions: np.ndarray,
        baselines: np.ndarray,
        session_start_timestamp: float,
    ) -> None:
        """
        Compare machine events against our breath analysis for debugging.

        Args:
            machine_events: Machine-detected events
            breaths: List of BreathMetrics objects
            reductions: Flow reduction array (per breath)
            baselines: Baseline array (per breath)
            session_start_timestamp: Unix timestamp of session start (for converting machine event times)
        """
        print(f"\n{'=' * 70}")
        print(
            f"DEBUG: Comparing {len(machine_events)} machine events against breath analysis"
        )
        print(f"{'=' * 70}")

        if breaths:
            print(
                f"Breath timestamp range: {breaths[0].start_time:.1f}s - {breaths[-1].end_time:.1f}s"
            )
            print(f"Session start timestamp: {session_start_timestamp:.1f}")

        for event in machine_events:
            event_start_relative = event.start_time - session_start_timestamp
            event_end_relative = event_start_relative + event.duration

            overlapping = [
                (i, b)
                for i, b in enumerate(breaths)
                if b.end_time >= event_start_relative
                and b.start_time <= event_end_relative
            ]

            print(f"\n{'-' * 70}")
            print(f"Machine Event: {event.event_type}")
            print(
                f"  Relative time: {event_start_relative:.1f}s | Duration: {event.duration:.1f}s"
            )
            print(f"  Overlapping breaths: {len(overlapping)}")

            if len(overlapping) == 0:
                print("  WARNING: No breaths found during this event!")
                continue

            for idx, breath in overlapping:
                if idx < len(reductions) and idx < len(baselines):
                    print(
                        f"    Breath {idx:4d}: "
                        f"TV={breath.tidal_volume:6.1f}mL | "
                        f"Amp={breath.amplitude:6.1f} | "
                        f"Baseline={baselines[idx]:6.1f}mL | "
                        f"Reduction={reductions[idx] * 100:5.1f}%"
                    )

            event_reductions = [
                reductions[idx] for idx, _ in overlapping if idx < len(reductions)
            ]
            if event_reductions:
                avg_reduction = np.mean(event_reductions) * 100
                print(f"  Average reduction: {avg_reduction:.1f}%")
                if event.event_type in [
                    "Central Apnea",
                    "Obstructive Apnea",
                    "Clear Airway",
                ]:
                    if avg_reduction >= 90:
                        print("  ✓ Meets apnea threshold (≥90%)")
                    else:
                        print(f"  ✗ Below apnea threshold ({avg_reduction:.1f}% < 90%)")
                elif event.event_type == "Hypopnea":
                    if 30 <= avg_reduction < 90:
                        print("  ✓ In hypopnea range (30-89%)")
                    else:
                        print(f"  ✗ Outside hypopnea range ({avg_reduction:.1f}%)")

    def analyze_session(
        self,
        session_id: int,
        modes: list[str] | None = None,
        store_results: bool = True,
        debug: bool = False,
    ) -> AnalysisResult:
        """
        Analyze session with specified detection mode(s).

        Args:
            session_id: Database session ID
            modes: Detection modes to run (None = default mode)
            store_results: Whether to persist results
            debug: Enable debug output

        Returns:
            AnalysisResult with results from all modes

        Raises:
            ValueError: If session not found or has no waveform data
        """
        if modes is None:
            modes = [DEFAULT_MODE]

        logger.info(f"Starting analysis for session {session_id} with modes: {modes}")
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
            raise ValueError(
                f"No flow waveform data available for session {session_id}"
            ) from e

        if len(timestamps) == 0:
            raise ValueError(f"Empty flow waveform data for session {session_id}")

        sample_rate = metadata.get("sample_rate", 25.0)
        session_duration_hours = len(timestamps) / sample_rate / 3600
        logger.info(
            f"Loaded {len(timestamps)} flow samples at {sample_rate}Hz ({session_duration_hours:.1f} hours)"
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

        engine_result = self.engine.analyze_session(
            session_id=session_id,
            timestamps=timestamps,
            flow_values=flow_values,
            sample_rate=sample_rate,
            spo2_values=spo2_values,
        )
        breaths = engine_result.breaths
        if breaths is None:
            raise ValueError(f"No breaths segmented for session {session_id}")

        mode_results = {}
        for mode_name in modes:
            try:
                mode = get_mode(mode_name)
                result = mode.detect_events(
                    breaths=breaths,
                    flow_data=(timestamps, flow_values),
                    session_duration_hours=session_duration_hours,
                )
                mode_results[mode_name] = result
                logger.info(
                    f"Mode '{mode_name}': Detected {len(result.apneas)} apneas, "
                    f"{len(result.hypopneas)} hypopneas, AHI={result.ahi:.1f}"
                )
            except Exception as e:
                logger.error(f"Failed to run mode '{mode_name}': {e}")
                continue

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Analysis complete in {processing_time_ms}ms")

        # TODO: Implement debug comparison for mode-based events
        # TODO: Implement storage for mode-based results

        return AnalysisResult(
            session_id=session_id,
            session_duration_hours=session_duration_hours,
            total_breaths=len(breaths),
            machine_events=machine_events,
            mode_results=mode_results,
            flow_analysis=engine_result.flow_analysis,
            positional_analysis=engine_result.positional_analysis,
            timestamp_start=timestamps[0] if len(timestamps) > 0 else 0.0,
            timestamp_end=timestamps[-1] if len(timestamps) > 0 else 0.0,
        )

    def analyze_sessions(
        self,
        session_ids: list[int],
        modes: list[str] | None = None,
        store_results: bool = True,
    ) -> list[AnalysisResult]:
        """
        Run analysis on multiple sessions.

        Args:
            session_ids: List of session IDs to analyze
            modes: Detection modes to run (None = default mode)
            store_results: Whether to store results in database

        Returns:
            List of analysis results (may have fewer entries if some sessions fail)
        """
        results = []
        for session_id in session_ids:
            try:
                result = self.analyze_session(
                    session_id,
                    modes=modes,
                    store_results=store_results,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze session {session_id}: {e}")
                continue

        return results

    def get_analysis_result(self, session_id: int) -> dict[str, Any] | None:
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
        machine_events: list[AnalysisEvent],
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
