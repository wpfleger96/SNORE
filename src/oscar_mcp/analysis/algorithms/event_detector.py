"""
Respiratory event detection algorithm.

This module implements detection of respiratory events including apneas
(obstructive, central, mixed), hypopneas, and RERAs based on flow
reduction patterns and duration criteria.
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np

from oscar_mcp.constants import EventDetectionConstants as EDC
from oscar_mcp.knowledge.patterns import RESPIRATORY_EVENTS

logger = logging.getLogger(__name__)


@dataclass
class ApneaEvent:
    """
    Detected apnea event.

    Attributes:
        start_time: Event start timestamp (seconds)
        end_time: Event end timestamp (seconds)
        duration: Event duration (seconds)
        event_type: OA (obstructive), CA (central), UA (unclassified), or MA (mixed)
        flow_reduction: Percentage flow reduction (0-1)
        confidence: Detection confidence (0-1)
        baseline_flow: Baseline flow before event (L/min)
    """

    start_time: float
    end_time: float
    duration: float
    event_type: str
    flow_reduction: float
    confidence: float
    baseline_flow: float


@dataclass
class HypopneaEvent:
    """
    Detected hypopnea event.

    Attributes:
        start_time: Event start timestamp (seconds)
        end_time: Event end timestamp (seconds)
        duration: Event duration (seconds)
        flow_reduction: Percentage flow reduction (0-1)
        confidence: Detection confidence (0-1)
        baseline_flow: Baseline flow before event (L/min)
        has_arousal: Whether arousal was detected (if available)
        has_desaturation: Whether SpO2 desaturation occurred (if available)
    """

    start_time: float
    end_time: float
    duration: float
    flow_reduction: float
    confidence: float
    baseline_flow: float
    has_arousal: Optional[bool] = None
    has_desaturation: Optional[bool] = None


@dataclass
class RERAEvent:
    """
    Detected RERA (Respiratory Effort Related Arousal) event.

    Attributes:
        start_time: Event start timestamp (seconds)
        end_time: Event end timestamp (seconds)
        duration: Event duration (seconds)
        flatness_index: Flatness of flow waveform during event (0-1)
        confidence: Detection confidence (0-1)
        terminated_by_arousal: Whether event ended with arousal
    """

    start_time: float
    end_time: float
    duration: float
    flatness_index: float
    confidence: float
    terminated_by_arousal: bool = False


@dataclass
class EventTimeline:
    """
    Complete timeline of detected respiratory events.

    Attributes:
        apneas: List of detected apnea events
        hypopneas: List of detected hypopnea events
        reras: List of detected RERA events
        total_events: Total count of all events
        ahi: Apnea-Hypopnea Index (events per hour)
        rdi: Respiratory Disturbance Index (includes RERAs)
    """

    apneas: List[ApneaEvent]
    hypopneas: List[HypopneaEvent]
    reras: List[RERAEvent]
    total_events: int
    ahi: float
    rdi: float


class RespiratoryEventDetector:
    """
    Detects respiratory events from flow waveform data.

    Uses flow reduction criteria and duration thresholds to identify
    apneas, hypopneas, and RERAs according to clinical definitions.

    Example:
        >>> detector = RespiratoryEventDetector()
        >>> apneas = detector.detect_apneas(
        ...     timestamps, flow_values, baseline_flow=30.0
        ... )
        >>> print(f"Found {len(apneas)} apneas")
    """

    def __init__(
        self,
        min_event_duration: float = EDC.MIN_EVENT_DURATION,
        baseline_window: float = EDC.BASELINE_WINDOW_SECONDS,
        merge_gap: float = EDC.MERGE_GAP_SECONDS,
    ):
        """
        Initialize the event detector.

        Args:
            min_event_duration: Minimum event duration in seconds (default 10s per clinical criteria)
            baseline_window: Window for calculating baseline flow (seconds)
            merge_gap: Gap size for merging adjacent events (seconds)
        """
        self.min_event_duration = min_event_duration
        self.baseline_window = baseline_window
        self.merge_gap = merge_gap
        self.event_criteria = RESPIRATORY_EVENTS
        logger.info(f"RespiratoryEventDetector initialized (min_duration={min_event_duration}s)")

    def detect_apneas(
        self,
        timestamps: np.ndarray,
        flow_values: np.ndarray,
        effort_signal: Optional[np.ndarray] = None,
    ) -> List[ApneaEvent]:
        """
        Detect apnea events (≥90% flow reduction for ≥10 seconds).

        Uses AASM-compliant rolling baseline calculation with 2-minute windows.

        Args:
            timestamps: Time values (seconds)
            flow_values: Flow data (L/min)
            effort_signal: Optional respiratory effort signal for OA/CA classification

        Returns:
            List of detected apnea events
        """
        logger.info(f"Detecting apneas with AASM-compliant rolling baseline (threshold={EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100}%)")

        baseline_array = self._calculate_rolling_baseline(timestamps, flow_values)
        logger.debug(f"Baseline range: {np.min(baseline_array):.1f} - {np.max(baseline_array):.1f} L/min, mean: {np.mean(baseline_array):.1f} L/min")

        apnea_threshold = EDC.APNEA_FLOW_REDUCTION_THRESHOLD

        flow_reduction = self._calculate_flow_reduction_array(flow_values, baseline_array)
        logger.debug(f"Flow reduction range: {np.min(flow_reduction)*100:.1f}% - {np.max(flow_reduction)*100:.1f}%, mean: {np.mean(flow_reduction)*100:.1f}%")

        is_reduced = flow_reduction >= apnea_threshold
        logger.debug(f"Samples meeting apnea threshold: {np.sum(is_reduced)} / {len(is_reduced)} ({np.sum(is_reduced)/len(is_reduced)*100:.2f}%)")

        events = self._find_continuous_regions(timestamps, is_reduced, self.min_event_duration)
        logger.debug(f"Found {len(events)} potential apnea events (before merging)")

        apneas = []
        for start_time, end_time, duration, indices in events:
            event_flow = flow_values[indices]
            event_baseline = baseline_array[indices]
            avg_baseline = float(np.mean(event_baseline))
            avg_reduction = np.mean(flow_reduction[indices])

            event_type = self._classify_apnea_type(
                effort_signal=effort_signal[indices] if effort_signal is not None else None,
                flow_signal=event_flow,
            )

            confidence = self._calculate_apnea_confidence(avg_reduction, duration, avg_baseline)

            logger.debug(f"  Apnea at {start_time:.1f}s: type={event_type}, duration={duration:.1f}s, reduction={avg_reduction*100:.1f}%, baseline={avg_baseline:.1f} L/min, confidence={confidence:.2f}")

            apneas.append(
                ApneaEvent(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    event_type=event_type,
                    flow_reduction=avg_reduction,
                    confidence=confidence,
                    baseline_flow=avg_baseline,
                )
            )

        apneas = self._merge_adjacent_events(apneas)

        logger.info(f"Detected {len(apneas)} apneas (after merging): {sum(1 for a in apneas if a.event_type == 'OA')} OA, {sum(1 for a in apneas if a.event_type == 'CA')} CA, {sum(1 for a in apneas if a.event_type == 'MA')} MA, {sum(1 for a in apneas if a.event_type == 'UA')} UA")

        return apneas

    def detect_hypopneas(
        self,
        timestamps: np.ndarray,
        flow_values: np.ndarray,
        spo2_signal: Optional[np.ndarray] = None,
    ) -> List[HypopneaEvent]:
        """
        Detect hypopnea events (30-90% flow reduction for ≥10 seconds).

        Uses AASM-compliant rolling baseline calculation with 2-minute windows.

        Args:
            timestamps: Time values (seconds)
            flow_values: Flow data (L/min)
            spo2_signal: Optional SpO2 data for desaturation detection

        Returns:
            List of detected hypopnea events
        """
        baseline_array = self._calculate_rolling_baseline(timestamps, flow_values)

        hypopnea_min = EDC.HYPOPNEA_MIN_REDUCTION
        hypopnea_max = EDC.HYPOPNEA_MAX_REDUCTION

        flow_reduction = self._calculate_flow_reduction_array(flow_values, baseline_array)

        is_hypopnea = (flow_reduction >= hypopnea_min) & (flow_reduction < hypopnea_max)

        events = self._find_continuous_regions(timestamps, is_hypopnea, self.min_event_duration)

        hypopneas = []
        for start_time, end_time, duration, indices in events:
            event_baseline = baseline_array[indices]
            avg_baseline = float(np.mean(event_baseline))
            avg_reduction = np.mean(flow_reduction[indices])

            has_desaturation = None
            if spo2_signal is not None:
                has_desaturation = self._check_desaturation(spo2_signal[indices])

            confidence = self._calculate_hypopnea_confidence(
                avg_reduction, duration, has_desaturation
            )

            hypopneas.append(
                HypopneaEvent(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    flow_reduction=avg_reduction,
                    confidence=confidence,
                    baseline_flow=avg_baseline,
                    has_desaturation=has_desaturation,
                )
            )

        hypopneas = self._merge_adjacent_events(hypopneas)

        return hypopneas

    def detect_reras(
        self,
        timestamps: np.ndarray,
        flow_values: np.ndarray,
        flatness_indices: np.ndarray,
    ) -> List[RERAEvent]:
        """
        Detect RERA events (flow limitation without apnea/hypopnea criteria).

        Uses AASM-compliant rolling baseline calculation with 2-minute windows.

        Args:
            timestamps: Time values (seconds)
            flow_values: Flow data (L/min)
            flatness_indices: Flatness index per breath or time window

        Returns:
            List of detected RERA events
        """
        baseline_array = self._calculate_rolling_baseline(timestamps, flow_values)

        is_flow_limited = flatness_indices > EDC.RERA_FLATNESS_THRESHOLD

        flow_reduction = self._calculate_flow_reduction_array(flow_values, baseline_array)
        is_not_apnea_hypopnea = flow_reduction < EDC.RERA_MAX_FLOW_REDUCTION

        is_rera = is_flow_limited & is_not_apnea_hypopnea

        events = self._find_continuous_regions(timestamps, is_rera, self.min_event_duration)

        reras = []
        for start_time, end_time, duration, indices in events:
            avg_flatness = np.mean(flatness_indices[indices])

            confidence = self._calculate_rera_confidence(avg_flatness, duration)

            reras.append(
                RERAEvent(
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    flatness_index=avg_flatness,
                    confidence=confidence,
                    terminated_by_arousal=False,
                )
            )

        reras = self._merge_adjacent_events(reras)

        return reras

    def create_event_timeline(
        self,
        apneas: List[ApneaEvent],
        hypopneas: List[HypopneaEvent],
        reras: List[RERAEvent],
        session_duration_hours: float,
    ) -> EventTimeline:
        """
        Create a complete event timeline with AHI and RDI calculations.

        Args:
            apneas: Detected apnea events
            hypopneas: Detected hypopnea events
            reras: Detected RERA events
            session_duration_hours: Total session duration in hours

        Returns:
            EventTimeline with all events and calculated indices
        """
        total_ah_events = len(apneas) + len(hypopneas)
        total_events = total_ah_events + len(reras)

        ahi = total_ah_events / session_duration_hours if session_duration_hours > 0 else 0.0
        rdi = total_events / session_duration_hours if session_duration_hours > 0 else 0.0

        return EventTimeline(
            apneas=apneas,
            hypopneas=hypopneas,
            reras=reras,
            total_events=total_events,
            ahi=ahi,
            rdi=rdi,
        )

    def _find_inspiratory_peaks(self, flow_values: np.ndarray) -> np.ndarray:
        """
        Find peak inspiratory flows (local maxima in positive flow).

        Returns array of peak flow values representing maximum inspiratory
        flow during each breath cycle.
        """
        from scipy import signal

        positive_flow = np.where(flow_values > 0, flow_values, 0)
        if len(positive_flow) < 10:
            return positive_flow[positive_flow > 0]

        peaks, properties = signal.find_peaks(
            positive_flow, distance=10, prominence=2.0, height=5.0
        )

        if len(peaks) == 0:
            return positive_flow[positive_flow > 5.0]

        return flow_values[peaks]

    def _calculate_rolling_baseline(
        self, timestamps: np.ndarray, flow_values: np.ndarray
    ) -> np.ndarray:
        """
        Calculate rolling baseline flow using AASM-compliant method.

        Uses 2-minute rolling window with 90th percentile of peak inspiratory
        flows. This matches commercial CPAP device algorithms (ResMed AirSense).

        Args:
            timestamps: Time values in seconds
            flow_values: Flow data in L/min

        Returns:
            Array of baseline values (one per sample) calculated from rolling window
        """
        window_seconds = EDC.BASELINE_WINDOW_SECONDS
        baseline = np.zeros_like(flow_values)

        peaks = self._find_inspiratory_peaks(flow_values)
        if len(peaks) < 10:
            fallback = float(np.percentile(flow_values[flow_values > 0], 90)) if np.any(flow_values > 0) else 30.0
            return np.full_like(flow_values, max(fallback, 10.0))

        for i in range(len(timestamps)):
            window_start = timestamps[i] - window_seconds
            window_mask = (timestamps >= window_start) & (timestamps < timestamps[i])

            window_flow = flow_values[window_mask]
            if len(window_flow) < 50:
                if i > 0:
                    baseline[i] = baseline[i - 1]
                else:
                    baseline[i] = 30.0
                continue

            window_peaks = self._find_inspiratory_peaks(window_flow)
            if len(window_peaks) >= 10:
                baseline[i] = float(np.percentile(window_peaks, 90))
            elif i > 0:
                baseline[i] = baseline[i - 1]
            else:
                baseline[i] = 30.0

            if baseline[i] < 10.0:
                baseline[i] = 10.0

        return baseline

    def _calculate_baseline_flow(self, flow_values: np.ndarray) -> float:
        """
        Calculate single baseline flow value (legacy method for backward compatibility).

        For new code, use _calculate_rolling_baseline() instead for AASM compliance.
        This method uses 90th percentile of peak inspiratory flows.
        """
        peaks = self._find_inspiratory_peaks(flow_values)
        if len(peaks) == 0:
            return 30.0

        baseline = float(np.percentile(peaks, 90))

        if baseline < 10.0:
            baseline = 10.0

        return baseline

    def _calculate_flow_reduction(self, flow_values: np.ndarray, baseline: float) -> np.ndarray:
        """Calculate flow reduction percentage relative to single baseline value."""
        if baseline <= 0:
            return np.zeros_like(flow_values)

        absolute_flow = np.abs(flow_values)
        reduction = 1.0 - (absolute_flow / baseline)
        reduction_clipped: np.ndarray = np.clip(reduction, 0.0, 1.0)

        return reduction_clipped

    def _calculate_flow_reduction_array(
        self, flow_values: np.ndarray, baseline_array: np.ndarray
    ) -> np.ndarray:
        """
        Calculate flow reduction percentage relative to array of baseline values.

        This method supports rolling baseline where each sample has its own
        baseline value calculated from the preceding 2-minute window.

        Args:
            flow_values: Flow data in L/min
            baseline_array: Array of baseline values (one per sample)

        Returns:
            Array of flow reduction percentages (0.0 to 1.0)
        """
        absolute_flow = np.abs(flow_values)

        baseline_safe = np.where(baseline_array > 0, baseline_array, 1.0)

        reduction = 1.0 - (absolute_flow / baseline_safe)
        reduction_clipped: np.ndarray = np.clip(reduction, 0.0, 1.0)

        return reduction_clipped

    def _find_continuous_regions(
        self,
        timestamps: np.ndarray,
        condition: np.ndarray,
        min_duration: float,
    ) -> List[Tuple[float, float, float, np.ndarray]]:
        """Find continuous regions where condition is True and duration >= min_duration."""
        regions = []

        starts = np.where(np.diff(np.concatenate([[False], condition, [False]]).astype(int)) == 1)[
            0
        ]
        ends = np.where(np.diff(np.concatenate([[False], condition, [False]]).astype(int)) == -1)[0]

        for start_idx, end_idx in zip(starts, ends):
            indices = np.arange(start_idx, end_idx)
            if len(indices) == 0:
                continue

            start_time = float(timestamps[start_idx])
            end_time = float(timestamps[end_idx - 1])
            duration = end_time - start_time

            if duration >= min_duration:
                regions.append((start_time, end_time, duration, indices))

        return regions

    def _classify_apnea_type(
        self, effort_signal: Optional[np.ndarray], flow_signal: Optional[np.ndarray] = None
    ) -> str:
        """
        Classify apnea as obstructive, central, or unclassified.

        If no effort signal is available, estimates effort from flow characteristics:
        - Obstructive Apnea (OA): Respiratory effort present but no airflow
          Flow shows more variability from attempted breathing
        - Central Apnea (CA): No respiratory effort, no airflow
          Flow is very flat and stable near zero

        Args:
            effort_signal: Direct effort measurement (from belts/sensors) if available
            flow_signal: Flow values during the apnea event

        Returns:
            Event type code: "OA", "CA", "MA", or "UA"
        """
        if effort_signal is not None:
            effort_magnitude = np.std(effort_signal)

            if effort_magnitude > EDC.APNEA_EFFORT_HIGH_THRESHOLD:
                return "OA"
            elif effort_magnitude < EDC.APNEA_EFFORT_LOW_THRESHOLD:
                return "CA"
            else:
                return "MA"

        if flow_signal is not None and len(flow_signal) > 5:
            effort_from_flow = self._estimate_effort_from_flow(flow_signal)

            if effort_from_flow > 0.15:
                return "OA"
            elif effort_from_flow < 0.05:
                return "CA"
            else:
                return "MA"

        return "UA"

    def _estimate_effort_from_flow(self, flow_signal: np.ndarray) -> float:
        """
        Estimate respiratory effort from flow signal characteristics.

        During obstructive apnea, continued respiratory effort causes flow oscillations
        even though net flow is near zero. Central apnea shows very flat flow.

        Args:
            flow_signal: Flow values during the event

        Returns:
            Estimated effort magnitude (0.0 = no effort, higher = more effort)
        """
        if len(flow_signal) < 5:
            return 0.0

        flow_std = np.std(flow_signal)
        flow_range = np.ptp(flow_signal)

        detrended = flow_signal - np.mean(flow_signal)
        variations = np.abs(np.diff(detrended))
        avg_variation = np.mean(variations) if len(variations) > 0 else 0.0

        effort_score = (flow_std * 0.4) + (flow_range * 0.3) + (avg_variation * 0.3)

        return float(effort_score)

    def _check_desaturation(self, spo2_values: np.ndarray) -> bool:
        """Check if SpO2 desaturation occurred (≥3% drop)."""
        if len(spo2_values) < 2:
            return False

        max_spo2 = np.max(spo2_values)
        min_spo2 = np.min(spo2_values)
        drop = max_spo2 - min_spo2

        return bool(drop >= EDC.SPO2_DESATURATION_DROP)

    def _calculate_apnea_confidence(
        self, reduction: float, duration: float, baseline: float
    ) -> float:
        """Calculate confidence score for apnea detection."""
        confidence = EDC.APNEA_BASE_CONFIDENCE

        if reduction > EDC.APNEA_HIGH_REDUCTION_THRESHOLD:
            confidence += EDC.APNEA_HIGH_REDUCTION_BONUS
        if duration > EDC.APNEA_LONG_DURATION_THRESHOLD:
            confidence += EDC.APNEA_LONG_DURATION_BONUS
        if baseline > EDC.APNEA_HIGH_BASELINE_THRESHOLD:
            confidence += EDC.APNEA_BASELINE_FLOW_BONUS

        return min(1.0, confidence)

    def _calculate_hypopnea_confidence(
        self, reduction: float, duration: float, has_desaturation: Optional[bool]
    ) -> float:
        """Calculate confidence score for hypopnea detection."""
        confidence = EDC.HYPOPNEA_BASE_CONFIDENCE

        if EDC.HYPOPNEA_IDEAL_MIN_REDUCTION <= reduction <= EDC.HYPOPNEA_IDEAL_MAX_REDUCTION:
            confidence += 0.1
        if duration > EDC.HYPOPNEA_LONG_DURATION_THRESHOLD:
            confidence += 0.1
        if has_desaturation:
            confidence += EDC.HYPOPNEA_DESATURATION_BONUS

        return min(1.0, confidence)

    def _calculate_rera_confidence(self, flatness: float, duration: float) -> float:
        """Calculate confidence score for RERA detection."""
        confidence = EDC.RERA_BASE_CONFIDENCE

        if flatness > EDC.RERA_HIGH_FLATNESS_THRESHOLD:
            confidence += EDC.RERA_HIGH_FLATNESS_BONUS
        if duration > EDC.APNEA_LONG_DURATION_THRESHOLD:
            confidence += 0.1

        return min(1.0, confidence)

    def _merge_adjacent_events(self, events: List) -> List:
        """Merge events that are close together in time."""
        if len(events) <= 1:
            return events

        merged = []
        current = events[0]

        for next_event in events[1:]:
            gap = next_event.start_time - current.end_time

            if gap <= self.merge_gap:
                current = self._merge_two_events(current, next_event)
            else:
                merged.append(current)
                current = next_event

        merged.append(current)

        return merged

    def _merge_two_events(
        self,
        event1: Any,
        event2: Any,
    ) -> Union[ApneaEvent, HypopneaEvent, RERAEvent]:
        """Merge two adjacent events of the same type."""
        merged_duration = event2.end_time - event1.start_time

        if isinstance(event1, ApneaEvent) and isinstance(event2, ApneaEvent):
            return ApneaEvent(
                start_time=event1.start_time,
                end_time=event2.end_time,
                duration=merged_duration,
                event_type=event1.event_type,
                flow_reduction=(event1.flow_reduction + event2.flow_reduction) / 2,
                confidence=min(event1.confidence, event2.confidence),
                baseline_flow=event1.baseline_flow,
            )
        elif isinstance(event1, HypopneaEvent) and isinstance(event2, HypopneaEvent):
            return HypopneaEvent(
                start_time=event1.start_time,
                end_time=event2.end_time,
                duration=merged_duration,
                flow_reduction=(event1.flow_reduction + event2.flow_reduction) / 2,
                confidence=min(event1.confidence, event2.confidence),
                baseline_flow=event1.baseline_flow,
                has_arousal=event1.has_arousal or event2.has_arousal,
                has_desaturation=event1.has_desaturation or event2.has_desaturation,
            )
        elif isinstance(event1, RERAEvent) and isinstance(event2, RERAEvent):
            return RERAEvent(
                start_time=event1.start_time,
                end_time=event2.end_time,
                duration=merged_duration,
                flatness_index=(event1.flatness_index + event2.flatness_index) / 2,
                confidence=min(event1.confidence, event2.confidence),
                terminated_by_arousal=event1.terminated_by_arousal or event2.terminated_by_arousal,
            )

        event_typed: Union[ApneaEvent, HypopneaEvent, RERAEvent] = event1
        return event_typed
