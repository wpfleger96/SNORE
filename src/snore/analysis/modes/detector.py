"""Unified respiratory event detector using configuration."""

import logging

from collections.abc import Sequence
from typing import Any, Literal, cast

import numpy as np

from scipy import signal

from snore.analysis.modes.config import DetectionModeConfig
from snore.analysis.modes.types import BaselineMethod, HypopneaMode, ModeResult
from snore.analysis.shared.types import (
    ApneaEvent,
    BreathMetrics,
    HypopneaEvent,
    RERAEvent,
)
from snore.constants import EventDetectionConstants as EDC

logger = logging.getLogger(__name__)


def _calculate_event_overlap(event1: ApneaEvent, event2: ApneaEvent) -> float:
    """
    Calculate overlap ratio between two events.

    Args:
        event1: First apnea event
        event2: Second apnea event

    Returns:
        Overlap ratio (0.0-1.0) relative to shorter event duration
    """
    overlap_start = max(event1.start_time, event2.start_time)
    overlap_end = min(event1.end_time, event2.end_time)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_duration = overlap_end - overlap_start
    shorter_duration = min(event1.duration, event2.duration)

    return overlap_duration / shorter_duration


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

    def _detect_events_resmed(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray] | None,
    ) -> list[ApneaEvent]:
        """
        ResMed-style detection combining multiple strategies.

        1. Gap detection - finds periods with no breaths
        2. Near-zero flow - finds sustained low flow in raw signal
        3. Amplitude reduction - lower threshold (50% vs 90%)

        Deduplicates overlapping detections from different methods.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)

        Returns:
            List of deduplicated and merged apnea events
        """
        logger.info(f"{self.config.name}: Running multi-strategy detection")

        all_events = []

        gap_events = self._detect_breath_gaps(breaths, min_gap_seconds=10.0)
        all_events.extend(gap_events)

        if flow_data is not None:
            timestamps, flow_values = flow_data
            if len(timestamps) > 1:
                zero_events = self._detect_near_zero_flow(
                    flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
                )
                all_events.extend(zero_events)

        amplitude_events = self._detect_apneas(breaths, flow_data)
        all_events.extend(amplitude_events)

        logger.info(
            f"{self.config.name}: Combined {len(all_events)} events from all strategies"
        )

        deduplicated = self._deduplicate_events(all_events, overlap_threshold=0.5)

        merged = cast(
            list[ApneaEvent],
            self._merge_adjacent_events(deduplicated, self.config.merge_gap),
        )

        return merged

    def detect_events(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray] | None,
        session_duration_hours: float,
    ) -> ModeResult:
        """
        Run detection algorithm and return results.

        Branches to mode-specific detection:
        - ResMed: Multi-strategy (gap + near-zero + amplitude)
        - AASM/aasm_relaxed: Amplitude-based only

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)
            session_duration_hours: Total session duration in hours

        Returns:
            ModeResult with detected events and metrics
        """
        if self.config.name == "resmed":
            apneas = self._detect_events_resmed(breaths, flow_data)
        else:
            apneas = self._detect_apneas(breaths, flow_data)

        hypopneas = self._detect_hypopneas(breaths, flow_data, spo2_signal=None)

        reras: list[RERAEvent] = []
        if self.config.rera_detection_enabled:
            reras = self._detect_reras(breaths, apneas, hypopneas)

        total_events = len(apneas) + len(hypopneas)
        ahi = (
            total_events / session_duration_hours if session_duration_hours > 0 else 0.0
        )

        rdi = (
            (total_events + len(reras)) / session_duration_hours
            if session_duration_hours > 0
            else 0.0
        )

        return ModeResult(
            mode_name=self.config.name,
            apneas=apneas,
            hypopneas=hypopneas,
            reras=reras,
            ahi=ahi,
            rdi=rdi,
            metadata={
                "config": self.config.name,
                "baseline_method": self.config.baseline_method.value,
                "apnea_threshold": self.config.apnea_threshold,
                "validation_threshold": self.config.apnea_validation_threshold,
                "rera_count": len(reras),
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

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            self.config.apnea_threshold,
            self.config.min_event_duration,
        )

        logger.debug(f"Found {len(regions)} potential apnea events (before merging)")

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

            if not self._validate_event(reductions, start_idx, end_idx):
                logger.debug(
                    f"  Rejecting apnea {start_idx}-{end_idx}: fails validation"
                )
                continue

            start_time = event_breaths[0].start_time
            end_time = event_breaths[-1].end_time
            avg_reduction = float(np.mean(event_reductions))
            avg_baseline = float(np.mean(event_baselines))

            flow_signal = None
            if flow_data is not None:
                timestamps, flow_values = flow_data
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                flow_signal = flow_values[mask]

            event_type, classification_confidence = self._classify_apnea_type(
                flow_signal=flow_signal
            )
            confidence = self._calculate_apnea_confidence(
                avg_reduction, duration, avg_baseline
            )

            logger.debug(
                f"  Apnea at {start_time:.1f}s: type={event_type}, duration={duration:.1f}s, "
                f"reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f}, "
                f"confidence={confidence:.2f}, classification_confidence={classification_confidence:.2f}"
            )

            apneas.append(
                ApneaEvent(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    duration=float(duration),
                    event_type=event_type,
                    flow_reduction=float(avg_reduction),
                    confidence=float(confidence),
                    classification_confidence=float(classification_confidence),
                    baseline_flow=float(avg_baseline),
                )
            )

        apneas = cast(
            list[ApneaEvent], self._merge_adjacent_events(apneas, self.config.merge_gap)
        )

        oa = sum(1 for a in apneas if a.event_type == "OA")
        ca = sum(1 for a in apneas if a.event_type == "CA")
        ma = sum(1 for a in apneas if a.event_type == "MA")
        ua = sum(1 for a in apneas if a.event_type == "UA")

        logger.info(
            f"{self.config.name}: Detected {len(apneas)} apneas: {oa} OA, {ca} CA, {ma} MA, {ua} UA"
        )

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
        Detect hypopnea events using configured mode.

        Supports multiple detection modes:
        - AASM_3PCT/4PCT: Requires SpO2 desaturation (3% or 4%)
        - FLOW_ONLY: 40% flow reduction without SpO2
        - DISABLED: Skip detection

        Falls back to FLOW_ONLY if SpO2 unavailable and fallback enabled.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)
            spo2_signal: SpO2 data for desaturation detection

        Returns:
            List of detected hypopnea events
        """
        if not breaths:
            return []

        if self.config.hypopnea_mode == HypopneaMode.DISABLED:
            logger.info(f"{self.config.name}: Hypopnea detection disabled")
            return []

        has_spo2 = spo2_signal is not None
        actual_mode = self.config.hypopnea_mode

        if not has_spo2:
            if self.config.hypopnea_mode in (
                HypopneaMode.AASM_3PCT,
                HypopneaMode.AASM_4PCT,
            ):
                if self.config.hypopnea_flow_only_fallback:
                    logger.info(
                        f"{self.config.name}: No SpO2 data, falling back to flow-only hypopnea detection"
                    )
                    actual_mode = HypopneaMode.FLOW_ONLY
                else:
                    logger.info(
                        f"{self.config.name}: Skipping hypopnea detection - no SpO2 data and fallback disabled"
                    )
                    return []

        logger.info(
            f"{self.config.name}: Detecting hypopneas (mode: {actual_mode.value})"
        )

        if actual_mode == HypopneaMode.FLOW_ONLY:
            min_threshold = 0.40  # 40% reduction for flow-only
        else:
            min_threshold = self.config.hypopnea_min_threshold  # 30%

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

        breaths_in_range = np.sum(
            (reductions >= min_threshold)
            & (reductions < EDC.APNEA_FLOW_REDUCTION_THRESHOLD)
        )
        logger.debug(
            f"Breaths in hypopnea range ({min_threshold * 100:.0f}-89%): {breaths_in_range}"
        )

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            min_threshold,
            self.config.min_event_duration,
        )

        hypopneas = []
        for start_idx, end_idx, duration in regions:
            event_reductions = reductions[start_idx:end_idx]
            if len(event_reductions) == 0:
                continue
            avg_reduction = float(np.mean(event_reductions))

            if avg_reduction >= EDC.APNEA_FLOW_REDUCTION_THRESHOLD:
                logger.debug(
                    f"  Skipping region {start_idx}-{end_idx}: avg reduction {avg_reduction * 100:.1f}% >= 90% (apnea)"
                )
                continue

            if not self._validate_event(
                reductions,
                start_idx,
                end_idx,
                threshold=min_threshold,
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

            has_desaturation = None
            if spo2_signal is not None and flow_data is not None:
                timestamps, _ = flow_data
                if len(spo2_signal) != len(timestamps):
                    logger.warning(
                        f"SpO2/flow timestamp mismatch: {len(spo2_signal)} vs {len(timestamps)} - "
                        "skipping desaturation check"
                    )
                    has_desaturation = None
                else:
                    mask = (timestamps >= start_time) & (timestamps <= end_time)
                    if np.any(mask):
                        if actual_mode == HypopneaMode.AASM_4PCT:
                            has_desaturation = self._check_desaturation(
                                spo2_signal[mask], threshold=4.0
                            )
                        else:
                            has_desaturation = self._check_desaturation(
                                spo2_signal[mask]
                            )

            confidence = self._calculate_hypopnea_confidence(
                avg_reduction, duration, has_desaturation, detection_mode=actual_mode
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

    def _detect_reras(
        self,
        breaths: list[BreathMetrics],
        apneas: list[ApneaEvent],
        hypopneas: list[HypopneaEvent],
    ) -> list[RERAEvent]:
        """
        Detect RERA-like events using FLOW event algorithm.

        Detects sequences of flow-limited breaths ending with recovery breath,
        without EEG arousal detection. Uses amplitude reduction as proxy for
        flow limitation.

        Algorithm:
        1. Find sequences of ≥2 breaths with moderate flow reduction (20-30%)
        2. Look for recovery breath with ≥50% amplitude increase
        3. Ensure ≥2-breath separation from apneas/hypopneas

        Args:
            breaths: List of BreathMetrics objects
            apneas: Detected apnea events (to avoid overlap)
            hypopneas: Detected hypopnea events (to avoid overlap)

        Returns:
            List of detected RERA events
        """
        if not breaths or len(breaths) < 3:
            return []

        logger.info(f"{self.config.name}: Detecting RERA events")

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_baseline(breaths, i)
            baselines[i] = baseline

            if baseline > 0:
                reduction = 1.0 - (breath.amplitude / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        excluded = np.zeros(len(breaths), dtype=bool)
        for event in list(apneas) + list(hypopneas):
            for i, breath in enumerate(breaths):
                if (
                    breath.start_time >= event.start_time
                    and breath.end_time <= event.end_time
                ):
                    excluded[i] = True

        reras = []
        i = 0
        while i < len(breaths) - 2:
            if excluded[i]:
                i += 1
                continue

            if 0.20 <= reductions[i] < 0.30:
                seq_start = i
                seq_count = 0
                while (
                    i < len(breaths)
                    and not excluded[i]
                    and 0.20 <= reductions[i] < 0.30
                ):
                    seq_count += 1
                    i += 1

                if seq_count >= 2 and i < len(breaths):
                    recovery_found = False
                    recovery_idx = -1

                    for j in range(i, min(i + 2, len(breaths))):
                        if excluded[j]:
                            continue

                        if reductions[j] < 0.10:
                            seq_avg_amplitude = np.mean(
                                [breaths[k].amplitude for k in range(seq_start, i)]
                            )
                            recovery_amplitude = breaths[j].amplitude

                            if seq_avg_amplitude > 0:
                                increase_pct = (
                                    recovery_amplitude - seq_avg_amplitude
                                ) / seq_avg_amplitude

                                if increase_pct >= 0.50:
                                    recovery_found = True
                                    recovery_idx = j
                                    break

                    if recovery_found and recovery_idx >= 0:
                        start_time = breaths[seq_start].start_time
                        end_time = breaths[recovery_idx].end_time
                        duration = end_time - start_time

                        if duration >= self.config.min_event_duration:
                            seq_baseline = np.mean(baselines[seq_start:i])
                            recovery_amplitude = breaths[recovery_idx].amplitude
                            seq_avg_amplitude = np.mean(
                                [breaths[k].amplitude for k in range(seq_start, i)]
                            )

                            amplitude_increase = (
                                (recovery_amplitude - seq_avg_amplitude)
                                / seq_avg_amplitude
                                if seq_avg_amplitude > 0
                                else 0.0
                            )

                            confidence = self._calculate_rera_confidence(
                                seq_count, float(amplitude_increase), duration
                            )

                            logger.debug(
                                f"  RERA at {start_time:.1f}s: duration={duration:.1f}s, "
                                f"obstructed_breaths={seq_count}, recovery_increase={amplitude_increase * 100:.1f}%, "
                                f"confidence={confidence:.2f}"
                            )

                            reras.append(
                                RERAEvent(
                                    start_time=float(start_time),
                                    end_time=float(end_time),
                                    duration=float(duration),
                                    obstructed_breath_count=seq_count,
                                    recovery_breath_amplitude=float(
                                        amplitude_increase * 100
                                    ),
                                    confidence=float(confidence),
                                    baseline_flow=float(seq_baseline),
                                )
                            )

                            i = recovery_idx + 1
                            continue

            i += 1

        logger.info(f"{self.config.name}: Detected {len(reras)} RERA events")

        return reras

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

        values = []
        for i in range(current_idx - 1, -1, -1):
            breath = breaths[i]
            if breath.start_time < window_start:
                break

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

    def _detect_breath_gaps(
        self,
        breaths: list[BreathMetrics],
        min_gap_seconds: float = 10.0,
    ) -> list[ApneaEvent]:
        """
        Detect apneas based on absence of breaths (gap detection).

        Finds periods ≥min_gap_seconds between consecutive breaths.
        Gaps indicate breathing cessation - classified as Central Apnea
        since no respiratory effort is detectable.

        This complements amplitude-based detection for events where
        breath segmentation fails entirely (no breaths to measure).

        Args:
            breaths: List of BreathMetrics objects
            min_gap_seconds: Minimum gap duration to qualify as apnea (default 10.0s)

        Returns:
            List of detected gap-based apnea events
        """
        events = []
        for i in range(1, len(breaths)):
            prev_end = breaths[i - 1].end_time
            curr_start = breaths[i].start_time
            gap = curr_start - prev_end

            if gap >= min_gap_seconds:
                events.append(
                    ApneaEvent(
                        start_time=float(prev_end),
                        end_time=float(curr_start),
                        duration=float(gap),
                        event_type="CA",  # Gap = no effort = central
                        flow_reduction=1.0,  # 100% - no breathing at all
                        confidence=0.85,
                        classification_confidence=0.9,  # High confidence: gap = no effort = CA
                        baseline_flow=0.0,  # No flow during gap
                        detection_method="gap",
                    )
                )

        if events:
            logger.info(f"{self.config.name}: Detected {len(events)} gap-based apneas")

        return events

    def _detect_near_zero_flow(
        self,
        flow_signal: np.ndarray,
        timestamps: np.ndarray,
        zero_threshold: float = 2.0,
        min_duration: float = 10.0,
    ) -> list[ApneaEvent]:
        """
        Detect apneas based on sustained near-zero flow in raw signal.

        Complements breath-based detection for cases where:
        - Breath segmentation misses the event entirely
        - Flow is too low to segment into distinct breaths

        Uses contiguous region detection to find periods where |flow| < threshold.

        Note: This is DIFFERENT from flow limitation flatness, which
        measures time at PEAK flow. This measures time at ZERO flow.

        Args:
            flow_signal: Raw flow signal values (L/min)
            timestamps: Timestamp array corresponding to flow samples
            zero_threshold: Flow threshold for "near-zero" (default 2.0 L/min)
            min_duration: Minimum event duration in seconds (default 10.0s)

        Returns:
            List of detected near-zero flow apnea events
        """
        total_duration = timestamps[-1] - timestamps[0]
        sample_rate = len(timestamps) / total_duration
        min_samples = int(min_duration * sample_rate)

        near_zero_mask = np.abs(flow_signal) < zero_threshold

        events = []
        in_event = False
        event_start = 0

        for i, is_near_zero in enumerate(near_zero_mask):
            if is_near_zero and not in_event:
                in_event = True
                event_start = i
            elif not is_near_zero and in_event:
                in_event = False
                event_length = i - event_start
                if event_length >= min_samples:
                    start_time = float(timestamps[event_start])
                    end_time = float(timestamps[i - 1])
                    events.append(
                        ApneaEvent(
                            start_time=start_time,
                            end_time=end_time,
                            duration=end_time - start_time,
                            event_type="CA",
                            flow_reduction=1.0,
                            confidence=0.80,
                            classification_confidence=0.85,  # High confidence: near-zero = CA
                            baseline_flow=0.0,
                            detection_method="near_zero_flow",
                        )
                    )

        if in_event:
            event_length = len(near_zero_mask) - event_start
            if event_length >= min_samples:
                start_time = float(timestamps[event_start])
                end_time = float(timestamps[-1])
                events.append(
                    ApneaEvent(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        event_type="CA",
                        flow_reduction=1.0,
                        confidence=0.80,
                        classification_confidence=0.85,  # High confidence: near-zero = CA
                        baseline_flow=0.0,
                        detection_method="near_zero_flow",
                    )
                )

        if events:
            logger.info(
                f"{self.config.name}: Detected {len(events)} near-zero flow apneas"
            )

        return events

    def _deduplicate_events(
        self,
        events: list[ApneaEvent],
        overlap_threshold: float = 0.5,
    ) -> list[ApneaEvent]:
        """
        Remove duplicate/overlapping events, keeping highest confidence.

        When multiple detection methods find the same event, keep the
        detection with highest confidence. Merge events that overlap
        by more than overlap_threshold (50% default).

        Args:
            events: List of apnea events (potentially overlapping)
            overlap_threshold: Minimum overlap ratio to consider duplicates (0.0-1.0)

        Returns:
            List of deduplicated events
        """
        if not events:
            return []

        sorted_events = sorted(events, key=lambda e: e.start_time)

        deduplicated = []
        current = sorted_events[0]

        for next_event in sorted_events[1:]:
            overlap = _calculate_event_overlap(current, next_event)

            if overlap > overlap_threshold:
                if next_event.confidence > current.confidence:
                    logger.debug(
                        f"  Replacing {current.detection_method} event at {current.start_time:.1f}s "
                        f"(conf={current.confidence:.2f}) with {next_event.detection_method} "
                        f"(conf={next_event.confidence:.2f})"
                    )
                    current = next_event
                else:
                    logger.debug(
                        f"  Keeping {current.detection_method} event at {current.start_time:.1f}s "
                        f"(conf={current.confidence:.2f}), dropping {next_event.detection_method} "
                        f"(conf={next_event.confidence:.2f})"
                    )
            else:
                deduplicated.append(current)
                current = next_event

        deduplicated.append(current)

        if len(events) > len(deduplicated):
            logger.info(
                f"{self.config.name}: Deduplicated {len(events)} events to {len(deduplicated)}"
            )

        return deduplicated

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
                classification_confidence=min(
                    event1.classification_confidence, event2.classification_confidence
                ),
                baseline_flow=event1.baseline_flow,
                detection_method=event1.detection_method,
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

        event_typed: ApneaEvent | HypopneaEvent = event1
        return event_typed

    def _classify_apnea_type(
        self,
        flow_signal: np.ndarray | None = None,
    ) -> tuple[Literal["OA", "CA", "MA", "UA"], float]:
        """
        Classify apnea as obstructive, central, or unclassified.

        Without effort sensors, estimates effort from flow characteristics.

        Args:
            flow_signal: Flow values during the apnea event

        Returns:
            Tuple of (event_type, classification_confidence)
            - event_type: "OA", "CA", "MA", or "UA"
            - classification_confidence: 0-1 score based on effort score distinctiveness
        """
        if flow_signal is not None and len(flow_signal) > 5:
            effort_from_flow = self._estimate_effort_from_flow(flow_signal)

            if effort_from_flow > 0.15:
                distance_from_boundary = min(effort_from_flow - 0.15, 0.35)
                classification_confidence = 0.5 + (distance_from_boundary / 0.35) * 0.5
                return "OA", float(classification_confidence)

            elif effort_from_flow < 0.05:
                distance_from_boundary = min(0.05 - effort_from_flow, 0.05)
                classification_confidence = 0.5 + (distance_from_boundary / 0.05) * 0.5
                return "CA", float(classification_confidence)

            else:
                distance_from_midpoint = abs(effort_from_flow - 0.10)
                classification_confidence = 0.3 + (distance_from_midpoint / 0.05) * 0.2
                return "MA", float(classification_confidence)

        return "UA", 0.2

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
        self,
        reduction: float,
        duration: float,
        has_desaturation: bool | None,
        detection_mode: HypopneaMode,
    ) -> float:
        """
        Calculate confidence score for hypopnea detection.

        Confidence levels by detection method:
        - HIGH: SpO2-validated (≥50% reduction OR 30-50% with desaturation)
        - MEDIUM: Flow-only ≥50% reduction
        - LOW: Flow-only 30-50% reduction

        Args:
            reduction: Flow reduction percentage (0-1)
            duration: Event duration in seconds
            has_desaturation: Whether SpO2 desaturation occurred (if available)
            detection_mode: Detection mode used

        Returns:
            Confidence score (0-1)
        """
        if detection_mode == HypopneaMode.FLOW_ONLY:
            if reduction >= 0.50:
                confidence = 0.6  # MEDIUM
            else:
                confidence = 0.4  # LOW
        else:
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

    def _check_desaturation(
        self, spo2_values: np.ndarray, threshold: float = 3.0
    ) -> bool:
        """
        Check if SpO2 desaturation occurred.

        Args:
            spo2_values: SpO2 signal values
            threshold: Desaturation threshold (default 3% for AASM, 4% for CMS)

        Returns:
            True if desaturation >= threshold occurred
        """
        if len(spo2_values) < 2:
            return False

        max_spo2 = np.max(spo2_values)
        min_spo2 = np.min(spo2_values)
        drop = max_spo2 - min_spo2

        return bool(drop >= threshold)

    def _calculate_rera_confidence(
        self, breath_count: int, amplitude_increase: float, duration: float
    ) -> float:
        """
        Calculate confidence score for RERA detection.

        RERAs detected from flow patterns (without EEG) have inherently
        lower confidence than EEG-confirmed events.

        Confidence factors:
        - More obstructed breaths = higher confidence
        - Larger recovery amplitude = higher confidence
        - Longer duration = higher confidence

        Args:
            breath_count: Number of flow-limited breaths in sequence
            amplitude_increase: Recovery breath amplitude increase (0-1 = 0-100%)
            duration: Event duration in seconds

        Returns:
            Confidence score (0-1), typically 0.4-0.7 for flow-only detection
        """
        confidence = 0.4

        if breath_count >= 3:
            confidence += 0.1
        if breath_count >= 5:
            confidence += 0.1

        if amplitude_increase >= 1.0:  # 100% increase (doubling)
            confidence += 0.2
        elif amplitude_increase >= 0.75:  # 75% increase
            confidence += 0.1

        if duration >= 15.0:
            confidence += 0.1

        return min(0.7, confidence)  # Cap at 0.7 without EEG

    def validate_against_machine_events(
        self,
        programmatic_apneas: list[ApneaEvent],
        programmatic_hypopneas: list[HypopneaEvent],
        machine_apneas: list[ApneaEvent],
        machine_hypopneas: list[HypopneaEvent],
        tolerance_seconds: float = 5.0,
    ) -> dict[str, Any]:
        """
        Validate programmatic event detection against machine-detected events.

        Compares timing of detected events with machine events and calculates
        agreement statistics (sensitivity, precision, F1 score).

        Args:
            programmatic_apneas: Apneas detected by our algorithm
            programmatic_hypopneas: Hypopneas detected by our algorithm
            machine_apneas: Apneas reported by the CPAP machine
            machine_hypopneas: Hypopneas reported by the CPAP machine
            tolerance_seconds: Max time difference for event matching (default 5s)

        Returns:
            Dictionary with validation metrics for apneas and hypopneas
        """
        from snore.models.analysis import EventValidationResult

        def validate_event_type(
            programmatic: Sequence[ApneaEvent | HypopneaEvent],
            machine: Sequence[ApneaEvent | HypopneaEvent],
        ) -> EventValidationResult:
            """Validate a single event type."""
            matched = 0
            matched_machine_indices = set()

            for prog_event in programmatic:
                for m_idx, mach_event in enumerate(machine):
                    if m_idx in matched_machine_indices:
                        continue

                    time_diff = abs(prog_event.start_time - mach_event.start_time)
                    if time_diff <= tolerance_seconds:
                        matched += 1
                        matched_machine_indices.add(m_idx)
                        break

            machine_count = len(machine)
            programmatic_count = len(programmatic)
            false_positives = programmatic_count - matched
            false_negatives = machine_count - matched

            if machine_count == 0:
                sensitivity = 1.0 if programmatic_count == 0 else 0.0
            elif matched + false_negatives > 0:
                sensitivity = matched / (matched + false_negatives)
            else:
                sensitivity = 0.0

            if matched + false_positives > 0:
                precision = matched / (matched + false_positives)
            else:
                precision = 0.0 if machine_count == 0 else 1.0

            if precision + sensitivity > 0:
                f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
            else:
                f1_score = 0.0

            total_unique = machine_count + programmatic_count - matched
            agreement_percentage = (
                (matched / total_unique * 100) if total_unique > 0 else 100.0
            )

            return EventValidationResult(
                machine_event_count=machine_count,
                programmatic_event_count=programmatic_count,
                matched_events=matched,
                false_positives=false_positives,
                false_negatives=false_negatives,
                sensitivity=sensitivity,
                precision=precision,
                f1_score=f1_score,
                agreement_percentage=agreement_percentage,
            )

        apnea_validation = validate_event_type(programmatic_apneas, machine_apneas)
        hypopnea_validation = validate_event_type(
            programmatic_hypopneas, machine_hypopneas
        )

        return {
            "apnea_validation": apnea_validation,
            "hypopnea_validation": hypopnea_validation,
            "overall_agreement": {
                "total_machine_events": len(machine_apneas) + len(machine_hypopneas),
                "total_programmatic_events": len(programmatic_apneas)
                + len(programmatic_hypopneas),
                "average_sensitivity": (
                    apnea_validation.sensitivity + hypopnea_validation.sensitivity
                )
                / 2,
                "average_precision": (
                    apnea_validation.precision + hypopnea_validation.precision
                )
                / 2,
                "average_f1": (apnea_validation.f1_score + hypopnea_validation.f1_score)
                / 2,
            },
        }
