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

    def detect_hypopneas(
        self,
        breaths: List,
        flow_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        spo2_signal: Optional[np.ndarray] = None,
    ) -> List[HypopneaEvent]:
        """
        Detect hypopnea events using breath-by-breath analysis.

        Hypopnea: 30-89% flow reduction for ≥10 seconds.
        Optionally requires SpO2 desaturation (≥3%) or arousal.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values) for detailed analysis
            spo2_signal: Optional SpO2 data for desaturation detection

        Returns:
            List of detected hypopnea events
        """
        if not breaths:
            return []

        logger.info("Detecting hypopneas using breath-by-breath analysis")

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_breath_baseline(breaths, i)
            baselines[i] = baseline

            if breath.duration > 8.0:
                reductions[i] = 1.0
            elif baseline > 0:
                tv_per_second = breath.tidal_volume / breath.duration if breath.duration > 0 else 0
                baseline_per_second = baseline / 4.0
                reduction = 1.0 - (tv_per_second / baseline_per_second)
                reductions[i] = max(0.0, min(1.0, reduction))

        logger.debug(f"Baseline range: {np.min(baselines):.1f} - {np.max(baselines):.1f} mL")
        logger.debug(
            f"Reduction range: {np.min(reductions) * 100:.1f}% - {np.max(reductions) * 100:.1f}%"
        )

        hypopnea_min = EDC.HYPOPNEA_MIN_REDUCTION
        breaths_in_range = np.sum(
            (reductions >= hypopnea_min) & (reductions < EDC.APNEA_FLOW_REDUCTION_THRESHOLD)
        )
        logger.debug(f"Breaths in hypopnea range (30-89%): {breaths_in_range}")

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            hypopnea_min,
            self.min_event_duration,
        )

        hypopneas = []
        for start_idx, end_idx, duration in regions:
            event_reductions = reductions[start_idx:end_idx]
            avg_reduction = float(np.mean(event_reductions))

            if avg_reduction >= EDC.APNEA_FLOW_REDUCTION_THRESHOLD:
                logger.debug(
                    f"  Skipping region {start_idx}-{end_idx}: avg reduction {avg_reduction * 100:.1f}% >= 90% (apnea, not hypopnea)"
                )
                continue

            if not self._validate_90_percent_rule(reductions, start_idx, end_idx, hypopnea_min):
                logger.debug(f"  Rejecting region {start_idx}-{end_idx}: fails 90% duration rule")
                continue

            event_baselines = baselines[start_idx:end_idx]
            avg_baseline = float(np.mean(event_baselines))

            start_time = breaths[start_idx].start_time
            end_time = breaths[end_idx - 1].end_time

            has_desaturation = None
            if spo2_signal is not None and flow_data is not None:
                timestamps, _ = flow_data
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                if np.any(mask):
                    has_desaturation = self._check_desaturation(spo2_signal[mask])

            confidence = self._calculate_hypopnea_confidence(
                avg_reduction, duration, has_desaturation
            )

            logger.debug(
                f"  Hypopnea at {start_time:.1f}s: duration={duration:.1f}s, reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f} mL, confidence={confidence:.2f}"
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

        logger.info(f"Detected {len(hypopneas)} hypopneas (after merging)")

        return hypopneas

    def detect_reras(
        self,
        breaths: List,
        flow_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> List[RERAEvent]:
        """
        Detect RERA events using breath-by-breath analysis.

        RERA: Flow limitation (flatness >0.7) with <30% amplitude reduction.
        Represents increased respiratory effort that doesn't meet apnea/hypopnea criteria.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values) for flatness calculation

        Returns:
            List of detected RERA events
        """
        if not breaths or flow_data is None:
            logger.warning("RERA detection skipped - requires breaths and flow_data")
            return []

        logger.info("Detecting RERAs using breath-by-breath analysis")
        timestamps, flow_values = flow_data

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))
        flatness_indices = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_breath_baseline(breaths, i)
            baselines[i] = baseline

            if breath.duration > 8.0:
                reductions[i] = 1.0
            elif baseline > 0:
                tv_per_second = breath.tidal_volume / breath.duration if breath.duration > 0 else 0
                baseline_per_second = baseline / 4.0
                reduction = 1.0 - (tv_per_second / baseline_per_second)
                reductions[i] = max(0.0, min(1.0, reduction))

            mask = (timestamps >= breath.start_time) & (timestamps <= breath.middle_time)
            if np.any(mask):
                insp_flow = flow_values[mask]
                flatness_indices[i] = self._calculate_flatness_index(insp_flow)

        logger.debug(
            f"Flatness range: {np.min(flatness_indices):.2f} - {np.max(flatness_indices):.2f}"
        )
        breaths_flow_limited = np.sum(
            (flatness_indices > EDC.RERA_FLATNESS_THRESHOLD)
            & (reductions < EDC.RERA_MAX_FLOW_REDUCTION)
        )
        logger.debug(
            f"Breaths with RERA criteria (flatness >0.7, reduction <30%): {breaths_flow_limited}"
        )

        rera_min_flatness = EDC.RERA_FLATNESS_THRESHOLD
        regions = self._find_consecutive_reduced_breaths(
            breaths,
            flatness_indices,
            rera_min_flatness,
            self.min_event_duration,
        )

        reras = []
        for start_idx, end_idx, duration in regions:
            event_reductions = reductions[start_idx:end_idx]
            avg_reduction = float(np.mean(event_reductions))

            if avg_reduction >= EDC.RERA_MAX_FLOW_REDUCTION:
                logger.debug(
                    f"  Skipping region {start_idx}-{end_idx}: avg reduction {avg_reduction * 100:.1f}% >= 30% (hypopnea/apnea, not RERA)"
                )
                continue

            event_flatness = flatness_indices[start_idx:end_idx]
            avg_flatness = float(np.mean(event_flatness))

            start_time = breaths[start_idx].start_time
            end_time = breaths[end_idx - 1].end_time

            confidence = self._calculate_rera_confidence(avg_flatness, duration)

            logger.debug(
                f"  RERA at {start_time:.1f}s: duration={duration:.1f}s, flatness={avg_flatness:.2f}, reduction={avg_reduction * 100:.1f}%, confidence={confidence:.2f}"
            )

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

        logger.info(f"Detected {len(reras)} RERAs (after merging)")

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

    def detect_apneas(
        self,
        breaths: List,
        flow_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> List[ApneaEvent]:
        """
        Detect apnea events using AASM-compliant breath-by-breath amplitude analysis.

        Instead of comparing instantaneous flow to baseline, this method compares each
        breath's peak inspiratory flow to a rolling baseline of recent breath amplitudes.
        This eliminates false positives from expiratory flow and dramatically improves
        performance (O(n) over breaths instead of O(n²) over samples).

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values) for detailed classification

        Returns:
            List of detected apnea events
        """
        if not breaths:
            logger.warning("No breaths provided for apnea detection")
            return []

        logger.info(
            f"Detecting apneas using breath-by-breath analysis (threshold={EDC.APNEA_FLOW_REDUCTION_THRESHOLD * 100}%)"
        )

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_breath_baseline(breaths, i)
            baselines[i] = baseline

            if breath.duration > 8.0:
                reductions[i] = 1.0
                logger.debug(
                    f"  Breath {i} marked as apnea (long duration: {breath.duration:.1f}s)"
                )
            elif baseline > 0:
                tv_per_second = breath.tidal_volume / breath.duration if breath.duration > 0 else 0
                baseline_per_second = baseline / 4.0
                reduction = 1.0 - (tv_per_second / baseline_per_second)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        logger.debug(
            f"Baseline range: {np.min(baselines):.1f} - {np.max(baselines):.1f} mL, mean: {np.mean(baselines):.1f} mL"
        )
        logger.debug(
            f"Reduction range: {np.min(reductions) * 100:.1f}% - {np.max(reductions) * 100:.1f}%, mean: {np.mean(reductions) * 100:.1f}%"
        )
        logger.debug(f"Breaths marked as apnea (>8s duration): {np.sum(reductions >= 1.0)}")

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            EDC.APNEA_FLOW_REDUCTION_THRESHOLD,
            self.min_event_duration,
        )

        logger.debug(f"Found {len(regions)} potential apnea events (before merging)")

        apneas = []
        for start_idx, end_idx, duration in regions:
            event_breaths = breaths[start_idx:end_idx]
            event_reductions = reductions[start_idx:end_idx]
            event_baselines = baselines[start_idx:end_idx]

            start_time = event_breaths[0].start_time
            end_time = event_breaths[-1].end_time
            avg_reduction = float(np.mean(event_reductions))
            avg_baseline = float(np.mean(event_baselines))

            flow_signal = None
            if flow_data is not None:
                timestamps, flow_values = flow_data
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                flow_signal = flow_values[mask]

            event_type = self._classify_apnea_type(
                effort_signal=None,
                flow_signal=flow_signal,
            )

            confidence = self._calculate_apnea_confidence(avg_reduction, duration, avg_baseline)

            logger.debug(
                f"  Apnea at {start_time:.1f}s: type={event_type}, duration={duration:.1f}s, reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f} L/min, confidence={confidence:.2f}"
            )

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

        logger.info(
            f"Detected {len(apneas)} apneas (after merging): {sum(1 for a in apneas if a.event_type == 'OA')} OA, {sum(1 for a in apneas if a.event_type == 'CA')} CA, {sum(1 for a in apneas if a.event_type == 'MA')} MA, {sum(1 for a in apneas if a.event_type == 'UA')} UA"
        )

        return apneas

    def _calculate_breath_baseline(
        self,
        breaths: List,
        current_idx: int,
        window_breaths: int = 30,
    ) -> float:
        """
        Calculate baseline from preceding breaths using 90th percentile.

        Uses a 2-minute window of breaths (~30 breaths at 15 breaths/min) to
        calculate the 90th percentile of tidal volumes. Tidal volume is a better
        indicator of actual ventilation than peak inspiratory flow, especially
        for detecting apneas where effort may be present but airflow is minimal.

        Args:
            breaths: List of BreathMetrics objects
            current_idx: Index of current breath
            window_breaths: Number of preceding breaths to include in window

        Returns:
            Baseline tidal volume (mL), minimum 100 mL
        """
        if current_idx == 0:
            return 300.0

        start_idx = max(0, current_idx - window_breaths)
        window = breaths[start_idx:current_idx]

        if len(window) < 5:
            return 300.0

        tidal_volumes = [
            b.tidal_volume for b in window if hasattr(b, "tidal_volume") and b.tidal_volume > 0
        ]
        if not tidal_volumes:
            return 300.0

        baseline = float(np.percentile(tidal_volumes, 90))
        return max(baseline, 100.0)

    def _validate_90_percent_rule(
        self,
        reductions: np.ndarray,
        start_idx: int,
        end_idx: int,
        threshold: float,
    ) -> bool:
        """
        Validate that 90% of breaths in event meet the reduction threshold.

        Per AASM standards, at least 90% of the event duration must meet
        the amplitude reduction criteria for the event to qualify.

        Args:
            reductions: Array of reduction values per breath (0.0-1.0)
            start_idx: Event start index in breaths array
            end_idx: Event end index in breaths array
            threshold: Minimum reduction threshold (e.g., 0.9 for apnea)

        Returns:
            True if >=90% of breaths meet the threshold
        """
        event_reductions = reductions[start_idx:end_idx]
        breaths_meeting = np.sum(event_reductions >= threshold)
        total_breaths = end_idx - start_idx
        return (
            (breaths_meeting / total_breaths) >= EDC.EVENT_DURATION_RULE_THRESHOLD
            if total_breaths > 0
            else False
        )

    def _calculate_flatness_index(self, inspiratory_flow: np.ndarray) -> float:
        """
        Calculate flatness index from inspiratory flow waveform.

        Flatness indicates flow limitation - the proportion of inspiration
        where flow remains at or near peak (≥80% of peak).

        Args:
            inspiratory_flow: Flow values during inspiratory phase

        Returns:
            Flatness index (0.0-1.0): proportion of samples above 80% of peak
        """
        if len(inspiratory_flow) < 5:
            return 0.0

        max_flow = np.max(inspiratory_flow)
        if max_flow <= 0:
            return 0.0

        threshold = 0.8 * max_flow
        samples_above = np.sum(inspiratory_flow >= threshold)
        return samples_above / len(inspiratory_flow)

    def _find_consecutive_reduced_breaths(
        self,
        breaths: List,
        reductions: np.ndarray,
        threshold: float,
        min_duration: float,
    ) -> List[Tuple[int, int, float]]:
        """
        Find runs of consecutive breaths meeting reduction threshold with AASM-compliant termination.

        Per AASM/ResMed standards, events terminate when 2+ consecutive breaths
        fall below the recovery threshold (50% of baseline = 50% reduction).

        Args:
            breaths: List of BreathMetrics objects
            reductions: Array of reduction values (0.0-1.0) per breath
            threshold: Minimum reduction to qualify (e.g., 0.9 for 90%)
            min_duration: Minimum total duration in seconds

        Returns:
            List of (start_idx, end_idx, total_duration) tuples
        """
        regions = []
        in_region = False
        region_start = 0
        recovery_count = 0
        recovery_threshold = EDC.EVENT_TERMINATION_RECOVERY
        min_recovery_breaths = EDC.EVENT_TERMINATION_MIN_BREATHS

        breaths_meeting_threshold = np.sum(reductions >= threshold)
        logger.debug(
            f"Finding consecutive reduced breaths: {breaths_meeting_threshold} breaths meet threshold >= {threshold * 100:.1f}%"
        )

        for i, reduction in enumerate(reductions):
            if reduction >= threshold and not in_region:
                in_region = True
                region_start = i
                recovery_count = 0
                logger.debug(f"  Starting region at breath {i} (reduction={reduction * 100:.1f}%)")
            elif in_region:
                if reduction < recovery_threshold:
                    recovery_count += 1
                    logger.debug(
                        f"  Recovery breath {recovery_count}/{min_recovery_breaths} at {i} (reduction={reduction * 100:.1f}%)"
                    )
                    if recovery_count >= min_recovery_breaths:
                        end_idx = i - min_recovery_breaths + 1
                        start_time = breaths[region_start].start_time
                        end_time = breaths[end_idx - 1].end_time
                        duration = end_time - start_time
                        logger.debug(
                            f"  Ending region at breath {end_idx - 1}: duration={duration:.1f}s (min={min_duration}s)"
                        )
                        if duration >= min_duration:
                            regions.append((region_start, end_idx, duration))
                            logger.debug(f"    ✓ Region accepted (duration >= {min_duration}s)")
                        else:
                            logger.debug(f"    ✗ Region rejected (duration < {min_duration}s)")
                        in_region = False
                        recovery_count = 0
                else:
                    recovery_count = 0

        if in_region:
            start_time = breaths[region_start].start_time
            end_time = breaths[-1].end_time
            duration = end_time - start_time
            logger.debug(f"  Final region at end: duration={duration:.1f}s")
            if duration >= min_duration:
                regions.append((region_start, len(breaths), duration))
                logger.debug("    ✓ Final region accepted")
            else:
                logger.debug(f"    ✗ Final region rejected (duration < {min_duration}s)")

        return regions

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

    def _calculate_spectral_effort(
        self,
        flow_signal: np.ndarray,
        sample_rate: float = 25.0,
    ) -> float:
        """
        Calculate spectral power in breathing frequency range (0.1-0.5 Hz).

        OA: Continued effort creates rhythmic oscillations visible in spectrum
        CA: No effort results in flat spectrum with no breathing frequency peaks

        Args:
            flow_signal: Flow values during the event
            sample_rate: Sampling rate in Hz (default 25 Hz for CPAP devices)

        Returns:
            Normalized spectral power in breathing frequency range (0.0-1.0)
        """
        from scipy import signal

        if len(flow_signal) < EDC.SPECTRAL_MIN_SAMPLES:
            return 0.0

        detrended = flow_signal - np.mean(flow_signal)
        freqs, power = signal.periodogram(detrended, fs=sample_rate)

        breathing_mask = (freqs >= EDC.BREATHING_FREQ_MIN) & (freqs <= EDC.BREATHING_FREQ_MAX)
        breathing_power = np.sum(power[breathing_mask])

        total_power = np.sum(power)
        if total_power > 0:
            return float(breathing_power / total_power)
        return 0.0

    def _estimate_effort_from_flow(self, flow_signal: np.ndarray) -> float:
        """
        Estimate respiratory effort from flow signal characteristics.

        During obstructive apnea, continued respiratory effort causes flow oscillations
        even though net flow is near zero. Central apnea shows very flat flow.

        Uses AASM-recommended weighting:
        - Standard deviation (30%): Overall variability
        - Peak-to-peak range (30%): Amplitude of oscillations
        - Roughness/variation (20%): High-frequency changes
        - Spectral power (20%): Rhythmic breathing frequency content

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

        spectral_power = self._calculate_spectral_effort(flow_signal)

        effort_score = (
            flow_std * 0.3 + flow_range * 0.3 + avg_variation * 0.2 + spectral_power * 0.2
        )

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
        """
        Merge events that are close together in time AND of the same type.

        Per AASM standards, only events of the same type should be merged.
        Different event types (e.g., OA vs CA) should remain separate even
        if temporally adjacent.
        """
        if len(events) <= 1:
            return events

        merged = []
        current = events[0]

        for next_event in events[1:]:
            gap = next_event.start_time - current.end_time
            same_type = getattr(next_event, "event_type", None) == getattr(
                current, "event_type", None
            )

            if gap <= self.merge_gap and same_type:
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
