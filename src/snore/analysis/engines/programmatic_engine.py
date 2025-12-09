"""
Programmatic analysis engine.

This module orchestrates all programmatic analysis components (flow limitation
classification, respiratory event detection, complex pattern detection) to
analyze complete CPAP sessions.
"""

import logging

from dataclasses import dataclass
from typing import Any

import numpy as np

from snore.analysis.algorithms.breath_segmenter import BreathSegmenter
from snore.analysis.algorithms.event_detector import (
    EventTimeline,
    RespiratoryEventDetector,
)
from snore.analysis.algorithms.feature_extractors import WaveformFeatureExtractor
from snore.analysis.algorithms.flow_limitation import (
    FlowLimitationClassifier,
    SessionFlowAnalysis,
)
from snore.analysis.algorithms.pattern_detector import (
    ComplexPatternDetector,
    CSRDetection,
    PeriodicBreathingDetection,
    PositionalAnalysis,
)
from snore.constants import (
    AnalysisEngineConstants as AEC,
)
from snore.constants import (
    BreathSegmentationConstants as BSC,
)
from snore.constants import (
    EventDetectionConstants as EDC,
)
from snore.constants import (
    FlowLimitationConstants as FLC,
)
from snore.utils.formatting import get_ahi_severity

logger = logging.getLogger(__name__)


@dataclass
class ProgrammaticAnalysisResult:
    """
    Complete programmatic analysis result for a session.

    Attributes:
        session_id: Database session ID
        timestamp_start: Analysis start time
        timestamp_end: Analysis end time
        duration_hours: Session duration in hours

        flow_analysis: Flow limitation classification results
        event_timeline: Respiratory event detection results
        csr_detection: Cheyne-Stokes Respiration detection (if found)
        periodic_breathing: Periodic breathing detection (if found)
        positional_analysis: Positional event clustering (if found)

        total_breaths: Total number of breaths analyzed
        processing_time_ms: Analysis processing time in milliseconds
        confidence_summary: Average confidence scores by analysis type
        clinical_summary: Human-readable summary of findings
    """

    session_id: int
    timestamp_start: float
    timestamp_end: float
    duration_hours: float

    flow_analysis: dict[str, Any]
    event_timeline: dict[str, Any]
    csr_detection: dict[str, Any] | None
    periodic_breathing: dict[str, Any] | None
    positional_analysis: dict[str, Any] | None

    total_breaths: int
    processing_time_ms: float
    confidence_summary: dict[str, float]
    clinical_summary: str
    machine_events: list[Any] | None = None


class ProgrammaticAnalysisEngine:
    """
    Orchestrates all programmatic analysis components.

    This engine coordinates breath segmentation, feature extraction,
    flow limitation classification, event detection, and pattern analysis
    to produce a comprehensive CPAP session analysis.

    Example:
        >>> engine = ProgrammaticAnalysisEngine()
        >>> result = engine.analyze_session(
        ...     session_id=123,
        ...     timestamps=timestamps,
        ...     flow_values=flow_values
        ... )
        >>> print(f"AHI: {result.event_timeline['ahi']:.1f}")
        >>> print(f"Flow Limitation Index: {result.flow_analysis['fl_index']:.2f}")
    """

    def __init__(
        self,
        min_breath_duration: float = BSC.MIN_BREATH_DURATION,
        min_event_duration: float = EDC.MIN_EVENT_DURATION,
        confidence_threshold: float = FLC.CONFIDENCE_THRESHOLD,
    ):
        """
        Initialize the programmatic analysis engine.

        Args:
            min_breath_duration: Minimum breath duration for segmentation (seconds)
            min_event_duration: Minimum event duration for detection (seconds)
            confidence_threshold: Minimum confidence for reliable findings
        """
        self.breath_segmenter = BreathSegmenter(min_breath_duration=min_breath_duration)
        self.feature_extractor = WaveformFeatureExtractor()
        self.flow_classifier = FlowLimitationClassifier(
            confidence_threshold=confidence_threshold
        )
        self.event_detector = RespiratoryEventDetector(
            min_event_duration=min_event_duration
        )
        self.pattern_detector = ComplexPatternDetector()

        logger.info("ProgrammaticAnalysisEngine initialized")

    def analyze_session(
        self,
        session_id: int,
        timestamps: np.ndarray,
        flow_values: np.ndarray,
        sample_rate: float = AEC.DEFAULT_SAMPLE_RATE,
        spo2_values: np.ndarray | None = None,
    ) -> ProgrammaticAnalysisResult:
        """
        Perform complete programmatic analysis of a CPAP session.

        Args:
            session_id: Database session ID
            timestamps: Time values (seconds)
            flow_values: Flow data (L/min)
            sample_rate: Sample rate in Hz
            spo2_values: Optional SpO2 data for desaturation analysis

        Returns:
            ProgrammaticAnalysisResult with all findings
        """
        import time

        start_time = time.time()

        logger.info(f"Starting programmatic analysis for session {session_id}")

        timestamp_start = float(timestamps[0])
        timestamp_end = float(timestamps[-1])
        duration_hours = (timestamp_end - timestamp_start) / 3600.0

        breaths = self.breath_segmenter.segment_breaths(
            timestamps, flow_values, sample_rate
        )

        logger.info(f"Segmented {len(breaths)} breaths")

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

        flow_analysis = self.flow_classifier.analyze_session(breath_features)

        logger.info(f"Flow limitation index: {flow_analysis.flow_limitation_index:.3f}")

        tidal_volumes = np.array([b.tidal_volume for b in breaths])

        apneas = self.event_detector.detect_apneas(
            breaths, flow_data=(timestamps, flow_values)
        )

        hypopneas = self.event_detector.detect_hypopneas(
            breaths, flow_data=(timestamps, flow_values), spo2_signal=spo2_values
        )

        event_timeline = self.event_detector.create_event_timeline(
            apneas, hypopneas, duration_hours
        )

        logger.info(f"Detected {len(apneas)} apneas, {len(hypopneas)} hypopneas")
        logger.info(f"AHI: {event_timeline.ahi:.1f}, RDI: {event_timeline.rdi:.1f}")

        breath_timestamps = np.array([b.start_time for b in breaths])
        respiratory_rates = np.array([b.respiratory_rate_rolling for b in breaths])

        csr_detection = self.pattern_detector.detect_csr(
            breath_timestamps, tidal_volumes, window_minutes=10.0
        )

        periodic_breathing = self.pattern_detector.detect_periodic_breathing(
            breath_timestamps, tidal_volumes, respiratory_rates
        )

        all_event_times = [a.start_time for a in apneas] + [
            h.start_time for h in hypopneas
        ]

        positional_analysis = None
        if len(all_event_times) >= 5:
            positional_analysis = self.pattern_detector.detect_positional_events(
                all_event_times, timestamp_end - timestamp_start
            )

        confidence_summary = self._calculate_confidence_summary(
            flow_analysis, event_timeline, csr_detection, periodic_breathing
        )

        clinical_summary = self._generate_clinical_summary(
            flow_analysis,
            event_timeline,
            csr_detection,
            periodic_breathing,
            positional_analysis,
        )

        processing_time = (time.time() - start_time) * 1000

        logger.info(f"Analysis complete in {processing_time:.0f}ms")

        return ProgrammaticAnalysisResult(
            session_id=session_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            duration_hours=duration_hours,
            flow_analysis=self._serialize_flow_analysis(flow_analysis),
            event_timeline=self._serialize_event_timeline(event_timeline),
            csr_detection=self._serialize_csr_detection(csr_detection),
            periodic_breathing=self._serialize_periodic_breathing(periodic_breathing),
            positional_analysis=self._serialize_positional_analysis(
                positional_analysis
            ),
            total_breaths=len(breaths),
            processing_time_ms=processing_time,
            confidence_summary=confidence_summary,
            clinical_summary=clinical_summary,
        )

    def _calculate_confidence_summary(
        self,
        flow_analysis: SessionFlowAnalysis,
        event_timeline: EventTimeline,
        csr: CSRDetection | None,
        periodic: PeriodicBreathingDetection | None,
    ) -> dict[str, float]:
        """Calculate average confidence scores by analysis type."""
        summary = {}

        if flow_analysis:
            summary["flow_classification"] = flow_analysis.average_confidence

        if event_timeline and len(event_timeline.apneas) > 0:
            summary["apnea_detection"] = float(
                np.mean([a.confidence for a in event_timeline.apneas])
            )

        if event_timeline and len(event_timeline.hypopneas) > 0:
            summary["hypopnea_detection"] = float(
                np.mean([h.confidence for h in event_timeline.hypopneas])
            )

        if csr:
            summary["csr_detection"] = csr.confidence

        if periodic:
            summary["periodic_breathing"] = periodic.confidence

        return summary

    def _generate_clinical_summary(
        self,
        flow_analysis: SessionFlowAnalysis,
        event_timeline: EventTimeline,
        csr: CSRDetection | None,
        periodic: PeriodicBreathingDetection | None,
        positional: PositionalAnalysis | None,
    ) -> str:
        """Generate human-readable clinical summary."""
        lines = []

        lines.append("SESSION ANALYSIS SUMMARY")
        lines.append("=" * 50)

        if flow_analysis:
            fli = flow_analysis.flow_limitation_index
            if fli < AEC.FLI_SEVERITY_MINIMAL:
                severity = "minimal"
            elif fli < AEC.FLI_SEVERITY_MILD:
                severity = "mild"
            elif fli < AEC.FLI_SEVERITY_MODERATE:
                severity = "moderate"
            else:
                severity = "severe"

            lines.append(f"Flow Limitation: {severity.upper()} (index: {fli:.2f})")

            class_dist = flow_analysis.class_distribution
            major_classes = [c for c in range(1, 8) if class_dist[c] > 0]
            if major_classes:
                lines.append(
                    f"  Classes detected: {', '.join(map(str, major_classes))}"
                )

        if event_timeline:
            ahi = event_timeline.ahi
            severity = get_ahi_severity(ahi).upper()
            if severity != "NORMAL":
                severity = f"{severity} OSA"

            lines.append(f"\nRespiratory Events: {severity}")
            lines.append(f"  AHI: {ahi:.1f} events/hour")
            lines.append(f"  RDI: {event_timeline.rdi:.1f} events/hour")
            lines.append(f"  Apneas: {len(event_timeline.apneas)}")
            lines.append(f"  Hypopneas: {len(event_timeline.hypopneas)}")

        if csr and csr.confidence > AEC.CSR_MIN_CONFIDENCE:
            lines.append("\nCheyne-Stokes Respiration DETECTED")
            lines.append(f"  Cycle length: {csr.cycle_length:.1f}s")
            lines.append(f"  CSR index: {csr.csr_index * 100:.1f}%")
            lines.append(f"  Confidence: {csr.confidence:.2f}")

        if periodic and periodic.confidence > AEC.PERIODIC_MIN_CONFIDENCE:
            lines.append("\nPeriodic Breathing DETECTED")
            lines.append(f"  Cycle length: {periodic.cycle_length:.1f}s")
            lines.append(f"  Regularity: {periodic.regularity_score:.2f}")

        if positional and positional.confidence > AEC.POSITIONAL_MIN_CONFIDENCE:
            lines.append("\nPositional Events DETECTED")
            lines.append("  Event clustering suggests position-dependent apnea")
            lines.append(f"  Likelihood: {positional.positional_likelihood:.2f}")

        return "\n".join(lines)

    def _serialize_flow_analysis(self, analysis: SessionFlowAnalysis) -> dict[str, Any]:
        """Convert flow analysis to dictionary."""
        return {
            "total_breaths": analysis.total_breaths,
            "class_distribution": analysis.class_distribution,
            "fl_index": analysis.flow_limitation_index,
            "average_confidence": analysis.average_confidence,
            "patterns": [
                {
                    "breath_number": p.breath_number,
                    "flow_class": p.flow_class,
                    "class_name": p.class_name,
                    "confidence": p.confidence,
                    "severity": p.severity,
                }
                for p in analysis.patterns
            ],
        }

    def _serialize_event_timeline(self, timeline: EventTimeline) -> dict[str, Any]:
        """Convert event timeline to dictionary."""
        return {
            "ahi": timeline.ahi,
            "rdi": timeline.rdi,
            "total_events": timeline.total_events,
            "apneas": [
                {
                    "start_time": a.start_time,
                    "duration": a.duration,
                    "event_type": a.event_type,
                    "flow_reduction": a.flow_reduction,
                    "confidence": a.confidence,
                    "baseline_flow": a.baseline_flow,
                }
                for a in timeline.apneas
            ],
            "hypopneas": [
                {
                    "start_time": h.start_time,
                    "duration": h.duration,
                    "flow_reduction": h.flow_reduction,
                    "confidence": h.confidence,
                    "baseline_flow": h.baseline_flow,
                    "has_arousal": h.has_arousal,
                    "has_desaturation": h.has_desaturation,
                }
                for h in timeline.hypopneas
            ],
        }

    def _serialize_csr_detection(
        self, csr: CSRDetection | None
    ) -> dict[str, Any] | None:
        """Convert CSR detection to dictionary."""
        if csr is None:
            return None

        return {
            "start_time": csr.start_time,
            "end_time": csr.end_time,
            "cycle_length": csr.cycle_length,
            "amplitude_variation": csr.amplitude_variation,
            "csr_index": csr.csr_index,
            "confidence": csr.confidence,
            "cycle_count": csr.cycle_count,
        }

    def _serialize_periodic_breathing(
        self, periodic: PeriodicBreathingDetection | None
    ) -> dict[str, Any] | None:
        """Convert periodic breathing detection to dictionary."""
        if periodic is None:
            return None

        return {
            "start_time": periodic.start_time,
            "end_time": periodic.end_time,
            "cycle_length": periodic.cycle_length,
            "regularity_score": periodic.regularity_score,
            "confidence": periodic.confidence,
            "has_apneas": periodic.has_apneas,
        }

    def _serialize_positional_analysis(
        self, positional: PositionalAnalysis | None
    ) -> dict[str, Any] | None:
        """Convert positional analysis to dictionary."""
        if positional is None:
            return None

        return {
            "cluster_times": positional.cluster_times,
            "cluster_event_counts": positional.cluster_event_counts,
            "positional_likelihood": positional.positional_likelihood,
            "confidence": positional.confidence,
            "total_clusters": positional.total_clusters,
        }
