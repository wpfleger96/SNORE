"""
Analysis service for orchestrating programmatic analysis.

This module provides the main interface for running comprehensive CPAP session
analysis, loading data from the database, and storing results.
"""

import logging
import time

from datetime import datetime

import numpy as np

from sqlalchemy.orm import Session

from snore.analysis.data.waveform_loader import WaveformLoader
from snore.analysis.events import AnalysisEvent
from snore.analysis.modes import DEFAULT_MODE, get_mode
from snore.analysis.shared.breath_segmenter import BreathSegmenter
from snore.analysis.shared.feature_extractors import WaveformFeatureExtractor
from snore.analysis.shared.flow_limitation import FlowLimitationClassifier
from snore.analysis.shared.pattern_detector import ComplexPatternDetector
from snore.analysis.types import AnalysisResult
from snore.constants import BreathSegmentationConstants as BSC
from snore.constants import FlowLimitationConstants as FLC
from snore.database import models

logger = logging.getLogger(__name__)

__all__ = ["AnalysisService", "AnalysisResult"]


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

    def __init__(
        self,
        db_session: Session,
        min_breath_duration: float = BSC.MIN_BREATH_DURATION,
        confidence_threshold: float = FLC.CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize analysis service.

        Args:
            db_session: SQLAlchemy database session
            min_breath_duration: Minimum breath duration for segmentation (seconds)
            confidence_threshold: Minimum confidence for reliable findings
        """
        self.db_session = db_session
        self.waveform_loader = WaveformLoader(db_session)
        self.breath_segmenter = BreathSegmenter(min_breath_duration=min_breath_duration)
        self.feature_extractor = WaveformFeatureExtractor()
        self.flow_classifier = FlowLimitationClassifier(
            confidence_threshold=confidence_threshold
        )
        self.pattern_detector = ComplexPatternDetector()

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

        # Breath segmentation
        breaths = self.breath_segmenter.segment_breaths(
            timestamps, flow_values, sample_rate
        )
        logger.info(f"Segmented {len(breaths)} breaths")

        if not breaths:
            raise ValueError(f"No breaths segmented for session {session_id}")

        # Feature extraction for flow limitation analysis
        breath_features = []
        for breath in breaths:
            breath_start_idx = np.searchsorted(timestamps, breath.start_time)
            breath_end_idx = np.searchsorted(timestamps, breath.end_time)

            breath_flow = flow_values[breath_start_idx:breath_end_idx]
            insp_flow = breath_flow[breath_flow > 0]

            if len(insp_flow) > 10:
                shape = self.feature_extractor.extract_shape_features(
                    insp_flow, sample_rate
                )
                peaks = self.feature_extractor.extract_peak_features(
                    insp_flow, sample_rate
                )
                breath_features.append((breath.breath_number, shape, peaks))

        # Flow limitation analysis
        flow_analysis = self.flow_classifier.analyze_session(breath_features)
        logger.info(f"Flow limitation index: {flow_analysis.flow_limitation_index:.3f}")

        # Pattern detection
        tidal_volumes = np.array([b.tidal_volume for b in breaths])
        breath_timestamps = np.array([b.start_time for b in breaths])
        respiratory_rates = np.array([b.respiratory_rate_rolling for b in breaths])

        csr_detection = self.pattern_detector.detect_csr(
            breath_timestamps, tidal_volumes, window_minutes=10.0
        )

        periodic_breathing = self.pattern_detector.detect_periodic_breathing(
            breath_timestamps, tidal_volumes, respiratory_rates
        )

        # Event detection via modes
        mode_results = {}
        for mode_name in modes:
            try:
                mode = get_mode(mode_name)
                mode_result = mode.detect_events(
                    breaths=breaths,
                    flow_data=(timestamps, flow_values),
                    session_duration_hours=session_duration_hours,
                )
                mode_results[mode_name] = mode_result
                logger.info(
                    f"Mode '{mode_name}': Detected {len(mode_result.apneas)} apneas, "
                    f"{len(mode_result.hypopneas)} hypopneas, AHI={mode_result.ahi:.1f}"
                )
            except Exception as e:
                logger.error(f"Failed to run mode '{mode_name}': {e}")
                continue

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Analysis complete in {processing_time_ms}ms")

        result = AnalysisResult(
            session_id=session_id,
            session_duration_hours=session_duration_hours,
            total_breaths=len(breaths),
            machine_events=machine_events,
            mode_results=mode_results,
            flow_analysis=flow_analysis.model_dump(),
            csr_detection=csr_detection.model_dump() if csr_detection else None,
            periodic_breathing=periodic_breathing.model_dump()
            if periodic_breathing
            else None,
            timestamp_start=float(timestamps[0]) if len(timestamps) > 0 else 0.0,
            timestamp_end=float(timestamps[-1]) if len(timestamps) > 0 else 0.0,
        )

        if store_results:
            self._store_result(result, processing_time_ms)

        return result

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

    def get_analysis_result(self, session_id: int) -> AnalysisResult | None:
        """
        Retrieve stored analysis result for a session.

        Args:
            session_id: Database session ID

        Returns:
            AnalysisResult dataclass or None if not found
        """
        analysis = (
            self.db_session.query(models.AnalysisResult)
            .filter_by(session_id=session_id)
            .order_by(models.AnalysisResult.created_at.desc())
            .first()
        )

        if not analysis:
            return None

        return AnalysisResult.model_validate(analysis.programmatic_result_json)

    def _store_result(self, result: AnalysisResult, processing_time_ms: int) -> int:
        """
        Store analysis result to database.

        Args:
            result: Analysis result to store
            processing_time_ms: Processing time in milliseconds

        Returns:
            Database analysis result ID
        """
        analysis = models.AnalysisResult(
            session_id=result.session_id,
            timestamp_start=datetime.fromtimestamp(result.timestamp_start),
            timestamp_end=datetime.fromtimestamp(result.timestamp_end),
            programmatic_result_json=result.model_dump(),
            processing_time_ms=processing_time_ms,
            engine_versions_json={
                "format_version": 2,
                "modes": list(result.mode_results.keys()),
            },
        )

        self.db_session.add(analysis)
        self.db_session.commit()

        logger.info(f"Stored analysis result with ID {analysis.id}")
        return analysis.id
