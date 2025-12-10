"""
AASM-compliant respiratory event detection mode.

Implements detection according to AASM Scoring Manual v2.6 (2023) standards:
- Apnea: ≥90% flow reduction for ≥10 seconds
- Hypopnea: 30-89% reduction + SpO2 desaturation or arousal
"""

import logging

from typing import Any

import numpy as np

from snore.analysis.algorithms.event_detector import ApneaEvent, HypopneaEvent
from snore.analysis.modes.base import DetectionMode, ModeResult
from snore.constants import EventDetectionConstants as EDC

logger = logging.getLogger(__name__)


class AASMDetectionMode(DetectionMode):
    """
    AASM-compliant respiratory event detection.

    Thresholds per AASM Scoring Manual v2.6 (2023):
    - Apnea: ≥90% flow reduction for ≥10 seconds
    - Hypopnea: 30-89% reduction + SpO2 desaturation or arousal
    """

    name = "aasm"
    description = "AASM-compliant detection (90% apnea, 30% hypopnea)"

    def __init__(
        self,
        apnea_threshold: float = 0.90,
        hypopnea_min: float = 0.30,
        min_duration: float = 10.0,
        baseline_window_seconds: float = 120.0,
        metric: str = "amplitude",
        merge_gap: float = 3.0,
    ):
        """
        Initialize AASM detection mode.

        Args:
            apnea_threshold: Minimum flow reduction for apnea (default 0.90 per AASM)
            hypopnea_min: Minimum flow reduction for hypopnea (default 0.30 per AASM)
            min_duration: Minimum event duration in seconds (default 10.0 per AASM)
            baseline_window_seconds: Time window for baseline in seconds (default 120.0 = 2 minutes per AASM)
            metric: Metric to use for reduction ("amplitude" per AASM)
            merge_gap: Maximum gap to merge adjacent events (default 3.0s)
        """
        self.apnea_threshold = apnea_threshold
        self.hypopnea_min = hypopnea_min
        self.min_duration = min_duration
        self.baseline_window_seconds = baseline_window_seconds
        self.metric = metric
        self.merge_gap = merge_gap

    def detect_events(
        self,
        breaths: list[Any],
        flow_data: tuple[np.ndarray, np.ndarray] | None,
        session_duration_hours: float,
    ) -> ModeResult:
        """
        Run AASM-compliant detection algorithm.

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
            mode_name=self.name,
            apneas=apneas,
            hypopneas=hypopneas,
            ahi=ahi,
            rdi=rdi,
            metadata={
                "apnea_threshold": self.apnea_threshold,
                "hypopnea_min": self.hypopnea_min,
                "metric": self.metric,
            },
        )

    def _detect_apneas(
        self,
        breaths: list[Any],
        flow_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[ApneaEvent]:
        """
        Detect apnea events using AASM-compliant breath-by-breath analysis.

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
            f"AASM mode: Detecting apneas (threshold={self.apnea_threshold * 100}%)"
        )

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_time_based_baseline(
                breaths, i, self.baseline_window_seconds, 90, self.metric
            )
            baselines[i] = baseline

            if baseline > 0:
                if self.metric == "amplitude":
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
            self.apnea_threshold,
            self.min_duration,
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

            if not self._validate_event(
                reductions, start_idx, end_idx, self.apnea_threshold
            ):
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

            event_type = self._classify_apnea_type(flow_signal=flow_signal)

            confidence = self._calculate_apnea_confidence(
                avg_reduction, duration, avg_baseline
            )

            logger.debug(
                f"  Apnea at {start_time:.1f}s: type={event_type}, duration={duration:.1f}s, reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f}, confidence={confidence:.2f}"
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

        apneas = self._merge_adjacent_events(apneas, self.merge_gap)

        logger.info(
            f"AASM mode: Detected {len(apneas)} apneas: {sum(1 for a in apneas if a.event_type == 'OA')} OA, {sum(1 for a in apneas if a.event_type == 'CA')} CA, {sum(1 for a in apneas if a.event_type == 'MA')} MA, {sum(1 for a in apneas if a.event_type == 'UA')} UA"
        )

        # Mark breaths as part of events for baseline exclusion
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
        breaths: list[Any],
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
                "AASM mode: Skipping hypopnea detection - no SpO2 data (AASM requirement)"
            )
            return []

        logger.info("AASM mode: Detecting hypopneas")

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = self._calculate_time_based_baseline(
                breaths, i, self.baseline_window_seconds, 90, self.metric
            )
            baselines[i] = baseline

            if baseline > 0:
                if self.metric == "amplitude":
                    value = breath.amplitude
                else:
                    value = breath.tidal_volume
                reduction = 1.0 - (value / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        # Find breaths in hypopnea range (30-89%)
        breaths_in_range = np.sum(
            (reductions >= self.hypopnea_min)
            & (reductions < EDC.APNEA_FLOW_REDUCTION_THRESHOLD)
        )
        logger.debug(f"Breaths in hypopnea range (30-89%): {breaths_in_range}")

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            self.hypopnea_min,
            self.min_duration,
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

            if not self._validate_event(
                reductions, start_idx, end_idx, self.hypopnea_min
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
                mask = (timestamps >= start_time) & (timestamps <= end_time)
                if np.any(mask):
                    has_desaturation = self._check_desaturation(spo2_signal[mask])

            confidence = self._calculate_hypopnea_confidence(
                avg_reduction, duration, has_desaturation
            )

            logger.debug(
                f"  Hypopnea at {start_time:.1f}s: duration={duration:.1f}s, reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f}, confidence={confidence:.2f}"
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

        hypopneas = self._merge_adjacent_events(hypopneas, self.merge_gap)

        logger.info(f"AASM mode: Detected {len(hypopneas)} hypopneas")

        return hypopneas

    def _validate_event(
        self,
        reductions: np.ndarray,
        start_idx: int,
        end_idx: int,
        threshold: float,
    ) -> bool:
        """
        Validate that event contains at least one breath meeting the threshold (AASM-compliant).

        Per AASM Scoring Manual v2.6, apnea requires ≥90% flow reduction.
        Check if at least one breath meets the specified threshold.

        Args:
            reductions: Array of reduction values per breath (0.0-1.0)
            start_idx: Event start index in breaths array
            end_idx: Event end index in breaths array
            threshold: Minimum reduction threshold (e.g., 0.9 for apnea per AASM)

        Returns:
            True if at least one breath meets the threshold
        """
        event_reductions = reductions[start_idx:end_idx]
        if len(event_reductions) == 0:
            return False

        max_reduction = float(np.max(event_reductions))
        return (
            max_reduction >= threshold
        )  # Use specified threshold (90% for AASM compliance)

    def _find_consecutive_reduced_breaths(
        self,
        breaths: list[Any],
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
