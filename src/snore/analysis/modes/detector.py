"""Unified respiratory event detector using configuration."""

import logging

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np

from scipy import signal

from snore.analysis.modes.config import DetectionModeConfig
from snore.analysis.modes.types import BaselineMethod, ModeResult
from snore.analysis.shared.types import ApneaEvent, BreathMetrics, HypopneaEvent
from snore.constants import EventDetectionConstants as EDC

logger = logging.getLogger(__name__)


class EventDetector:
    """
    Configurable respiratory event detector.

    Uses DetectionModeConfig to parameterize detection behavior.
    Supports both AASM-compliant and relaxed detection modes.
    """

    def __init__(self, config: DetectionModeConfig):
        """
        Initialize detector with configuration.

        Args:
            config: Detection mode configuration
        """
        self.config = config

    @property
    def name(self) -> str:
        """Get mode name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get mode description."""
        return self.config.description

    def detect_events(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray] | None,
        session_duration_hours: float,
    ) -> ModeResult:
        """
        Run detection algorithm and return results.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)
            session_duration_hours: Total session duration in hours

        Returns:
            ModeResult with detected events and metrics
        """
        # Detect apneas
        apneas = self._detect_apneas(breaths, flow_data)

        # Detect hypopneas (requires SpO2 data per AASM)
        hypopneas = self._detect_hypopneas(breaths, flow_data, spo2_signal=None)

        # Calculate AHI and RDI
        total_events = len(apneas) + len(hypopneas)
        ahi = (
            total_events / session_duration_hours if session_duration_hours > 0 else 0.0
        )
        rdi = ahi  # Without RERA detection, RDI equals AHI

        return ModeResult(
            mode_name=self.config.name,
            apneas=apneas,
            hypopneas=hypopneas,
            ahi=ahi,
            rdi=rdi,
            metadata={
                "config": self.config.name,
                "baseline_method": self.config.baseline_method.value,
                "apnea_threshold": self.config.apnea_threshold,
                "validation_threshold": self.config.apnea_validation_threshold,
            },
        )

    # ========================================================================
    # Event Detection
    # ========================================================================

    def _detect_apneas(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[ApneaEvent]:
        """
        Detect apnea events using configured thresholds.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)

        Returns:
            List of detected apnea events
        """
        if not breaths:
            logger.warning("No breaths provided for apnea detection")
            return []

        logger.info(
            f"{self.config.name}: Detecting apneas (threshold={self.config.apnea_threshold * 100}%, "
            f"validation={self.config.apnea_validation_threshold * 100}%)"
        )

        # Calculate baselines and reductions
        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_baseline(breaths, i)
            baselines[i] = baseline

            if baseline > 0:
                if self.config.metric == "amplitude":
                    value = breath.amplitude
                else:
                    value = breath.tidal_volume
                reduction = 1.0 - (value / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        logger.debug(
            f"Baseline range: {np.min(baselines):.1f} - {np.max(baselines):.1f}, mean: {np.mean(baselines):.1f}"
        )
        logger.debug(
            f"Reduction range: {np.min(reductions) * 100:.1f}% - {np.max(reductions) * 100:.1f}%, mean: {np.mean(reductions) * 100:.1f}%"
        )

        # Find consecutive reduced breaths
        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            self.config.apnea_threshold,
            self.config.min_event_duration,
        )

        logger.debug(f"Found {len(regions)} potential apnea events (before merging)")

        # Create apnea events
        apneas = []
        for start_idx, end_idx, duration in regions:
            event_breaths = breaths[start_idx:end_idx]
            event_reductions = reductions[start_idx:end_idx]
            event_baselines = baselines[start_idx:end_idx]

            if (
                len(event_breaths) == 0
                or len(event_reductions) == 0
                or len(event_baselines) == 0
            ):
                continue

            # Validate event using configured threshold
            if not self._validate_event(reductions, start_idx, end_idx):
                logger.debug(
                    f"  Rejecting apnea {start_idx}-{end_idx}: fails validation"
                )
                continue

            start_time = event_breaths[0].start_time
            end_time = event_breaths[-1].end_time
            avg_reduction = float(np.mean(event_reductions))
            avg_baseline = float(np.mean(event_baselines))

            # Extract flow signal for classification
            flow_signal = None
            if flow_data is not None:
                timestamps, flow_values = flow_data
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                flow_signal = flow_values[mask]

            event_type = self._classify_apnea_type(flow_signal=flow_signal)
            confidence = self._calculate_apnea_confidence(
                avg_reduction, duration, avg_baseline
            )

            logger.debug(
                f"  Apnea at {start_time:.1f}s: type={event_type}, duration={duration:.1f}s, "
                f"reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f}, confidence={confidence:.2f}"
            )

            apneas.append(
                ApneaEvent(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    duration=float(duration),
                    event_type=event_type,
                    flow_reduction=float(avg_reduction),
                    confidence=float(confidence),
                    baseline_flow=float(avg_baseline),
                )
            )

        # Merge adjacent apneas
        apneas = cast(
            list[ApneaEvent], self._merge_adjacent_events(apneas, self.config.merge_gap)
        )

        # Count by type
        oa = sum(1 for a in apneas if a.event_type == "OA")
        ca = sum(1 for a in apneas if a.event_type == "CA")
        ma = sum(1 for a in apneas if a.event_type == "MA")
        ua = sum(1 for a in apneas if a.event_type == "UA")

        logger.info(
            f"{self.config.name}: Detected {len(apneas)} apneas: {oa} OA, {ca} CA, {ma} MA, {ua} UA"
        )

        # Mark breaths as part of events
        for apnea in apneas:
            for breath in breaths:
                if (
                    breath.start_time >= apnea.start_time
                    and breath.end_time <= apnea.end_time
                ):
                    breath.in_event = True

        return apneas

    def _detect_hypopneas(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray] | None = None,
        spo2_signal: np.ndarray | None = None,
    ) -> list[HypopneaEvent]:
        """
        Detect hypopnea events.

        Per AASM standards, requires SpO2 desaturation (≥3%) or arousal.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)
            spo2_signal: SpO2 data for desaturation detection (required per AASM)

        Returns:
            List of detected hypopnea events
        """
        if not breaths:
            return []

        if spo2_signal is None:
            logger.info(
                f"{self.config.name}: Skipping hypopnea detection - no SpO2 data (AASM requirement)"
            )
            return []

        logger.info(f"{self.config.name}: Detecting hypopneas")

        # Calculate baselines and reductions
        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_baseline(breaths, i)
            baselines[i] = baseline

            if baseline > 0:
                if self.config.metric == "amplitude":
                    value = breath.amplitude
                else:
                    value = breath.tidal_volume
                reduction = 1.0 - (value / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        # Find breaths in hypopnea range (30-89%)
        breaths_in_range = np.sum(
            (reductions >= self.config.hypopnea_min_threshold)
            & (reductions < EDC.APNEA_FLOW_REDUCTION_THRESHOLD)
        )
        logger.debug(f"Breaths in hypopnea range (30-89%): {breaths_in_range}")

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            self.config.hypopnea_min_threshold,
            self.config.min_event_duration,
        )

        hypopneas = []
        for start_idx, end_idx, duration in regions:
            event_reductions = reductions[start_idx:end_idx]
            if len(event_reductions) == 0:
                continue
            avg_reduction = float(np.mean(event_reductions))

            # Skip if reduction is apnea-level
            if avg_reduction >= EDC.APNEA_FLOW_REDUCTION_THRESHOLD:
                logger.debug(
                    f"  Skipping region {start_idx}-{end_idx}: avg reduction {avg_reduction * 100:.1f}% >= 90% (apnea)"
                )
                continue

            # Validate event
            if not self._validate_event(
                reductions,
                start_idx,
                end_idx,
                threshold=self.config.hypopnea_min_threshold,
            ):
                logger.debug(
                    f"  Rejecting hypopnea {start_idx}-{end_idx}: fails validation"
                )
                continue

            event_baselines = baselines[start_idx:end_idx]
            if len(event_baselines) == 0:
                continue
            avg_baseline = float(np.mean(event_baselines))

            start_time = breaths[start_idx].start_time
            end_time = breaths[end_idx - 1].end_time

            # Check for desaturation
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
                f"  Hypopnea at {start_time:.1f}s: duration={duration:.1f}s, "
                f"reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f}, confidence={confidence:.2f}"
            )

            hypopneas.append(
                HypopneaEvent(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    duration=float(duration),
                    flow_reduction=float(avg_reduction),
                    confidence=float(confidence),
                    baseline_flow=float(avg_baseline),
                    has_desaturation=has_desaturation,
                )
            )

        hypopneas = cast(
            list[HypopneaEvent],
            self._merge_adjacent_events(hypopneas, self.config.merge_gap),
        )

        logger.info(f"{self.config.name}: Detected {len(hypopneas)} hypopneas")

        return hypopneas

    def _validate_event(
        self,
        reductions: np.ndarray,
        start_idx: int,
        end_idx: int,
        threshold: float | None = None,
    ) -> bool:
        """
        Validate that event contains at least one breath meeting the threshold.

        Uses configured validation threshold.

        Args:
            reductions: Array of reduction values per breath (0.0-1.0)
            start_idx: Event start index in breaths array
            end_idx: Event end index in breaths array
            threshold: Override threshold (if None, uses config.apnea_validation_threshold)

        Returns:
            True if at least one breath meets the threshold
        """
        event_reductions = reductions[start_idx:end_idx]
        if len(event_reductions) == 0:
            return False

        max_reduction = float(np.max(event_reductions))

        # Use provided threshold or fall back to config
        validation_threshold = (
            threshold
            if threshold is not None
            else self.config.apnea_validation_threshold
        )
        return max_reduction >= validation_threshold

    # ========================================================================
    # Baseline Calculation (branches on config)
    # ========================================================================

    def _calculate_baseline(self, breaths: list[Any], current_idx: int) -> float:
        """Calculate baseline using configured method."""
        if self.config.baseline_method == BaselineMethod.TIME:
            return self._calculate_time_based_baseline(breaths, current_idx)
        else:
            return self._calculate_breath_based_baseline(breaths, current_idx)

    def _calculate_time_based_baseline(
        self, breaths: list[Any], current_idx: int
    ) -> float:
        """
        Calculate baseline from breaths within time window (AASM-compliant).

        Uses a time-based window (default 2 minutes per AASM) of preceding breaths
        to calculate baseline, excluding breaths that are part of detected events.

        Args:
            breaths: List of BreathMetrics objects
            current_idx: Index of current breath

        Returns:
            Baseline value, minimum 10.0 for amplitude or 100.0 for tidal_volume
        """
        if current_idx == 0:
            return 30.0 if self.config.metric == "amplitude" else 300.0

        current_breath = breaths[current_idx]
        current_time = current_breath.start_time
        window_start = current_time - self.config.baseline_window

        # Collect breaths within time window
        values = []
        for i in range(current_idx - 1, -1, -1):
            breath = breaths[i]
            if breath.start_time < window_start:
                break

            # Extract metric value, excluding event breaths
            if not breath.in_event:
                if self.config.metric == "amplitude":
                    if breath.amplitude > 0:
                        values.append(breath.amplitude)
                elif self.config.metric == "tidal_volume":
                    if breath.tidal_volume > 0:
                        values.append(breath.tidal_volume)

        if len(values) < 5:
            return 30.0 if self.config.metric == "amplitude" else 300.0

        baseline = float(np.percentile(values, self.config.baseline_percentile))
        min_baseline = 10.0 if self.config.metric == "amplitude" else 100.0
        return max(baseline, min_baseline)

    def _calculate_breath_based_baseline(
        self, breaths: list[Any], current_idx: int
    ) -> float:
        """
        Calculate baseline from preceding breath count.

        Uses a rolling window of recent breaths to calculate baseline,
        excluding breaths that are part of detected events.

        Args:
            breaths: List of BreathMetrics objects
            current_idx: Index of current breath

        Returns:
            Baseline value, minimum 10.0 for amplitude or 100.0 for tidal_volume
        """
        if current_idx == 0:
            return 30.0 if self.config.metric == "amplitude" else 300.0

        window_breaths = int(
            self.config.baseline_window
        )  # baseline_window is breath count
        start_idx = max(0, current_idx - window_breaths)
        window = breaths[start_idx:current_idx]

        if len(window) < 5:
            return 30.0 if self.config.metric == "amplitude" else 300.0

        # Extract metric values, excluding event breaths
        values = []
        for b in window:
            if self.config.metric == "amplitude":
                if b.amplitude > 0 and not b.in_event:
                    values.append(b.amplitude)
            elif self.config.metric == "tidal_volume":
                if b.tidal_volume > 0 and not b.in_event:
                    values.append(b.tidal_volume)

        if not values:
            return 30.0 if self.config.metric == "amplitude" else 300.0

        baseline = float(np.percentile(values, self.config.baseline_percentile))
        min_baseline = 10.0 if self.config.metric == "amplitude" else 100.0
        return max(baseline, min_baseline)

    # ========================================================================
    # Shared Utilities (no duplication - single implementation)
    # ========================================================================

    def _find_consecutive_reduced_breaths(
        self,
        breaths: list[BreathMetrics],
        reductions: np.ndarray,
        threshold: float,
        min_duration: float,
    ) -> list[tuple[int, int, float]]:
        """
        Find runs of consecutive breaths meeting reduction threshold.

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
            elif in_region:
                if reduction < recovery_threshold:
                    recovery_count += 1
                    if recovery_count >= min_recovery_breaths:
                        end_idx = i - min_recovery_breaths + 1
                        start_time = breaths[region_start].start_time
                        end_time = breaths[end_idx - 1].end_time
                        duration = end_time - start_time
                        if duration >= min_duration:
                            regions.append((region_start, end_idx, duration))
                        in_region = False
                        recovery_count = 0
                else:
                    recovery_count = 0

        if in_region:
            start_time = breaths[region_start].start_time
            end_time = breaths[-1].end_time
            duration = end_time - start_time
            if duration >= min_duration:
                regions.append((region_start, len(breaths), duration))

        return regions

    def _merge_adjacent_events(
        self,
        events: Sequence[ApneaEvent | HypopneaEvent],
        max_gap: float,
    ) -> list[ApneaEvent | HypopneaEvent]:
        """
        Merge events that are close together in time AND of the same type.

        Per AASM standards, only events of the same type should be merged.

        Args:
            events: List of ApneaEvent or HypopneaEvent objects
            max_gap: Maximum gap in seconds to merge

        Returns:
            List of merged events
        """
        if len(events) <= 1:
            return list(events)

        merged = []
        current = events[0]

        for next_event in events[1:]:
            gap = next_event.start_time - current.end_time
            same_type = type(next_event) == type(current)

            if gap <= max_gap and same_type:
                current = self._merge_two_events(current, next_event)
            else:
                merged.append(current)
                current = next_event

        merged.append(current)
        return merged

    def _merge_two_events(
        self,
        event1: ApneaEvent | HypopneaEvent,
        event2: ApneaEvent | HypopneaEvent,
    ) -> ApneaEvent | HypopneaEvent:
        """
        Merge two adjacent events of the same type.

        Args:
            event1: First event
            event2: Second event

        Returns:
            Merged event
        """
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

        # Fallback
        event_typed: ApneaEvent | HypopneaEvent = event1
        return event_typed

    def _classify_apnea_type(
        self,
        flow_signal: np.ndarray | None = None,
    ) -> Literal["OA", "CA", "MA", "UA"]:
        """
        Classify apnea as obstructive, central, or unclassified.

        Without effort sensors, estimates effort from flow characteristics.

        Args:
            flow_signal: Flow values during the apnea event

        Returns:
            Event type code: "OA", "CA", "MA", or "UA"
        """
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
            flow_std * 0.3
            + flow_range * 0.3
            + avg_variation * 0.2
            + spectral_power * 0.2
        )

        return float(effort_score)

    def _calculate_spectral_effort(
        self,
        flow_signal: np.ndarray,
        sample_rate: float = 25.0,
    ) -> float:
        """
        Calculate spectral power in breathing frequency range (0.1-0.5 Hz).

        Args:
            flow_signal: Flow values during the event
            sample_rate: Sampling rate in Hz (default 25 Hz for CPAP devices)

        Returns:
            Normalized spectral power in breathing frequency range (0.0-1.0)
        """
        if len(flow_signal) < EDC.SPECTRAL_MIN_SAMPLES:
            return 0.0

        detrended = flow_signal - np.mean(flow_signal)
        freqs, power = signal.periodogram(detrended, fs=sample_rate)

        breathing_mask = (freqs >= EDC.BREATHING_FREQ_MIN) & (
            freqs <= EDC.BREATHING_FREQ_MAX
        )
        breathing_power = np.sum(power[breathing_mask])

        total_power = np.sum(power)
        if total_power > 0:
            return float(breathing_power / total_power)
        return 0.0

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
        self, reduction: float, duration: float, has_desaturation: bool | None
    ) -> float:
        """Calculate confidence score for hypopnea detection."""
        confidence = EDC.HYPOPNEA_BASE_CONFIDENCE

        if (
            EDC.HYPOPNEA_IDEAL_MIN_REDUCTION
            <= reduction
            <= EDC.HYPOPNEA_IDEAL_MAX_REDUCTION
        ):
            confidence += 0.1
        if duration > EDC.HYPOPNEA_LONG_DURATION_THRESHOLD:
            confidence += 0.1
        if has_desaturation:
            confidence += EDC.HYPOPNEA_DESATURATION_BONUS

        return min(1.0, confidence)

    def _check_desaturation(self, spo2_values: np.ndarray) -> bool:
        """Check if SpO2 desaturation occurred (≥3% drop)."""
        if len(spo2_values) < 2:
            return False

        max_spo2 = np.max(spo2_values)
        min_spo2 = np.min(spo2_values)
        drop = max_spo2 - min_spo2

        return bool(drop >= EDC.SPO2_DESATURATION_DROP)
