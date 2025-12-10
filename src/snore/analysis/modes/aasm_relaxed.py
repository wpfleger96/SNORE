"""
AASM-Relaxed respiratory event detection mode.

Relaxed version of AASM detection for better machine matching:
- Uses 30-breath baseline window instead of 2-minute time-based (AASM standard)
- Uses 85% validation threshold instead of 90% (AASM standard)
"""

import logging

from typing import Any

import numpy as np

from snore.analysis.algorithms.event_detector import ApneaEvent, HypopneaEvent
from snore.analysis.modes.aasm import AASMDetectionMode

logger = logging.getLogger(__name__)


class AASMRelaxedDetectionMode(AASMDetectionMode):
    """
    Relaxed AASM-based detection for better machine matching.

    Differences from strict AASM:
    - Uses 30-breath baseline window instead of 2-minute time-based (AASM standard)
    - Uses 85% validation threshold instead of 90% (AASM standard)

    These relaxed thresholds better match machine detection behavior while
    maintaining AASM-based detection logic.
    """

    name = "aasm_relaxed"
    description = "AASM-based with relaxed thresholds for machine matching"

    def __init__(
        self,
        apnea_threshold: float = 0.90,
        hypopnea_min: float = 0.30,
        min_duration: float = 10.0,
        baseline_window_breaths: int = 30,
        metric: str = "amplitude",
        merge_gap: float = 3.0,
    ):
        """
        Initialize relaxed AASM detection mode.

        Args:
            apnea_threshold: Minimum flow reduction for apnea (default 0.90)
            hypopnea_min: Minimum flow reduction for hypopnea (default 0.30)
            min_duration: Minimum event duration in seconds (default 10.0)
            baseline_window_breaths: Number of breaths for baseline (default 30, not time-based)
            metric: Metric to use for reduction ("amplitude" or "tidal_volume")
            merge_gap: Maximum gap to merge adjacent events (default 3.0s)
        """
        # Call parent init with time-based parameter (will be overridden)
        super().__init__(
            apnea_threshold=apnea_threshold,
            hypopnea_min=hypopnea_min,
            min_duration=min_duration,
            baseline_window_seconds=120.0,  # Ignored, we use breath-based
            metric=metric,
            merge_gap=merge_gap,
        )
        # Override with breath-based window
        self.baseline_window_breaths = baseline_window_breaths

    def _detect_apneas(
        self, breaths: list[Any], flow_data: tuple[np.ndarray, np.ndarray] | None = None
    ) -> list[ApneaEvent]:
        """Detect apneas using breath-based baseline (not time-based)."""
        logger.info(
            "AASM Relaxed mode: Detecting apneas (threshold=90.0%, 30-breath baseline, 85% validation)"
        )

        apneas = []
        reductions = np.zeros(len(breaths))
        baselines = np.zeros(len(breaths))

        # Calculate baselines using breath-based window (not time-based)
        for i, breath in enumerate(breaths):
            baseline = self._calculate_rolling_baseline(
                breaths, i, self.baseline_window_breaths, 90, self.metric
            )
            baselines[i] = baseline

            if baseline > 0:
                if self.metric == "amplitude":
                    reduction = 1.0 - (breath.amplitude / baseline)
                else:
                    reduction = 1.0 - (breath.tidal_volume / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))

        # Find consecutive reduced breaths
        regions = self._find_consecutive_reduced_breaths(
            breaths, reductions, self.apnea_threshold, self.min_duration
        )

        # Validate and create apnea events
        for start_idx, end_idx, duration in regions:
            # Use relaxed 85% validation threshold
            if self._validate_event_relaxed(reductions, start_idx, end_idx):
                start_time = breaths[start_idx].start_time
                end_time = breaths[end_idx - 1].end_time

                if duration >= self.min_duration:
                    # Extract flow signal from tuple if available
                    flow_signal = flow_data[1] if flow_data is not None else None
                    event_type = self._classify_apnea_type(flow_signal)

                    # Calculate average reduction for confidence
                    avg_reduction = float(np.mean(reductions[start_idx:end_idx]))
                    avg_baseline = float(np.mean(baselines[start_idx:end_idx]))
                    confidence = self._calculate_apnea_confidence(
                        avg_reduction, duration, avg_baseline
                    )

                    apneas.append(
                        ApneaEvent(
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            event_type=event_type,
                            flow_reduction=float(
                                np.mean(reductions[start_idx:end_idx])
                            ),
                            confidence=confidence,
                            baseline_flow=float(np.mean(baselines[start_idx:end_idx])),
                        )
                    )

        # Merge adjacent apneas
        apneas = self._merge_adjacent_events(apneas, self.merge_gap)

        # Count by type
        oa_count = sum(1 for a in apneas if a.event_type == "OA")
        ca_count = sum(1 for a in apneas if a.event_type == "CA")
        ma_count = sum(1 for a in apneas if a.event_type == "MA")
        ua_count = sum(1 for a in apneas if a.event_type == "UA")

        logger.info(
            f"AASM Relaxed mode: Detected {len(apneas)} apneas: "
            f"{oa_count} OA, {ca_count} CA, {ma_count} MA, {ua_count} UA"
        )

        return apneas

    def _detect_hypopneas(
        self,
        breaths: list[Any],
        flow_data: tuple[np.ndarray, np.ndarray] | None = None,
        spo2_signal: np.ndarray | None = None,
    ) -> list[HypopneaEvent]:
        """Detect hypopneas using breath-based baseline (not time-based)."""
        if spo2_signal is None:
            logger.info(
                "AASM Relaxed mode: Skipping hypopnea detection - no SpO2 data (AASM requirement)"
            )
            return []

        logger.info(
            "AASM Relaxed mode: Detecting hypopneas (30-89% reduction, 30-breath baseline)"
        )

        hypopneas = []
        reductions = np.zeros(len(breaths))
        baselines = np.zeros(len(breaths))

        # Calculate baselines using breath-based window (not time-based)
        for i, breath in enumerate(breaths):
            baseline = self._calculate_rolling_baseline(
                breaths, i, self.baseline_window_breaths, 90, self.metric
            )
            baselines[i] = baseline

            if baseline > 0:
                if self.metric == "amplitude":
                    reduction = 1.0 - (breath.amplitude / baseline)
                else:
                    reduction = 1.0 - (breath.tidal_volume / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))

        # Find consecutive reduced breaths in hypopnea range
        hypopnea_threshold = self.hypopnea_min
        regions = self._find_consecutive_reduced_breaths(
            breaths, reductions, hypopnea_threshold, self.min_duration
        )

        # Filter to hypopnea range (30-89%) and check for desaturation
        for start_idx, end_idx, duration in regions:
            avg_reduction = float(np.mean(reductions[start_idx:end_idx]))

            if self.hypopnea_min <= avg_reduction < self.apnea_threshold:
                start_time = breaths[start_idx].start_time
                end_time = breaths[end_idx - 1].end_time

                if duration >= self.min_duration:
                    # For now, assume desaturation check requires proper SPO2 slicing
                    # This is a simplified implementation
                    has_desaturation = (
                        self._check_desaturation(spo2_signal)
                        if spo2_signal is not None
                        else False
                    )

                    if has_desaturation:
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
                                baseline_flow=float(
                                    np.mean(baselines[start_idx:end_idx])
                                ),
                                has_desaturation=True,
                            )
                        )

        hypopneas = self._merge_adjacent_events(hypopneas, self.merge_gap)

        logger.info(f"AASM Relaxed mode: Detected {len(hypopneas)} hypopneas")

        return hypopneas

    def _validate_event_relaxed(
        self,
        reductions: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> bool:
        """
        Validate event using relaxed 85% threshold (not 90% AASM standard).

        This relaxed validation catches borderline events that machines detect
        but strict AASM 90% validation would miss.

        Args:
            reductions: Array of reduction values per breath (0.0-1.0)
            start_idx: Event start index in breaths array
            end_idx: Event end index in breaths array

        Returns:
            True if at least one breath meets 85% reduction
        """
        event_reductions = reductions[start_idx:end_idx]
        if len(event_reductions) == 0:
            return False

        max_reduction = float(np.max(event_reductions))
        return max_reduction >= 0.85  # Relaxed 85% threshold (not 90% AASM standard)
