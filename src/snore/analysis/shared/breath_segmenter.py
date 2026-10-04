"""
Breath segmentation algorithm for flow waveform analysis.

This module provides algorithms for segmenting continuous flow waveform data
into individual breaths, identifying inspiration/expiration phases, and
calculating breath-level metrics.
"""

import logging

import numpy as np

from snore.analysis.shared.types import BreathMetrics
from snore.constants import BreathSegmentationConstants

logger = logging.getLogger(__name__)

__all__ = ["BreathSegmenter", "BreathMetrics"]


class BreathSegmenter:
    """
    Segments flow waveform data into individual breaths.

    Uses zero-crossing detection to identify breath boundaries and calculates
    comprehensive metrics for each breath.

    Example:
        >>> segmenter = BreathSegmenter(min_breath_duration=1.0)
        >>> breaths = segmenter.segment_breaths(
        ...     timestamps, flow_values, sample_rate=25.0
        ... )
        >>> print(f"Found {len(breaths)} breaths")
    """

    def __init__(
        self,
        min_breath_duration: float = 1.0,
        max_breath_duration: float = 20.0,
        hysteresis: float = 2.0,
    ):
        """
        Initialize breath segmenter with configuration.

        Args:
            min_breath_duration: Minimum valid breath duration in seconds
            max_breath_duration: Maximum valid breath duration in seconds
            hysteresis: Zero-crossing hysteresis threshold in L/min
                (prevents false triggers from noise near zero)
        """
        self.min_breath_duration = min_breath_duration
        self.max_breath_duration = max_breath_duration
        self.hysteresis = hysteresis

    def segment_breaths(
        self,
        timestamps: np.ndarray,
        flow_data: np.ndarray,
        sample_rate: float,
    ) -> list[BreathMetrics]:
        """
        Segment flow waveform into individual breaths.

        Args:
            timestamps: 1D array of time offsets in seconds
            flow_data: 1D array of flow values in L/min
            sample_rate: Sample rate in Hz

        Returns:
            List of BreathMetrics objects, one per detected breath

        Example:
            >>> breaths = segmenter.segment_breaths(t, flow, 25.0)
        """
        if len(flow_data) == 0 or len(timestamps) == 0:
            logger.debug("Empty input arrays, returning no breaths")
            return []

        logger.debug(
            f"Segmenting {len(flow_data)} samples at {sample_rate} Hz "
            f"(duration: {timestamps[-1] - timestamps[0]:.1f}s)"
        )

        crossings = self.detect_zero_crossings(flow_data)

        boundaries = self.identify_breath_boundaries(
            crossings, timestamps, sample_rate, flow_data
        )

        logger.debug(f"Identified {len(boundaries)} potential breaths")

        breaths: list[BreathMetrics] = []
        tv_history: list[float] = []

        for idx, (start_idx, end_idx) in enumerate(boundaries):
            breath_segment = flow_data[start_idx:end_idx]
            breath_timestamps = timestamps[start_idx:end_idx]

            metrics = self.calculate_breath_metrics(
                breath_number=idx + 1,
                timestamps=breath_timestamps,
                flow_values=breath_segment,
                sample_rate=sample_rate,
                tv_history=tv_history,
                all_breaths=breaths,
            )

            if metrics.is_complete:
                breaths.append(metrics)
                tv_history.append(metrics.tidal_volume)
                if len(tv_history) > 3:
                    tv_history.pop(0)

        if len(breaths) > 0:
            logger.info(
                f"Segmented {len(breaths)} valid breaths "
                f"(avg duration: {np.mean([b.duration for b in breaths]):.2f}s)"
            )
        else:
            logger.info("Segmented 0 valid breaths")

        return breaths

    def detect_zero_crossings(self, flow_data: np.ndarray) -> list[tuple[int, str]]:
        """
        Detect zero crossings in flow data with hysteresis.

        A zero crossing occurs when flow transitions from positive to negative
        (end of inspiration) or negative to positive (end of expiration).
        Hysteresis prevents false triggers from noise near zero.

        Args:
            flow_data: 1D array of flow values in L/min

        Returns:
            List of (index, direction) tuples where:
                - index: Sample index of zero crossing
                - direction: "positive" (exp→insp) or "negative" (insp→exp)

        Example:
            >>> crossings = segmenter.detect_zero_crossings(flow)
            >>> print(f"Found {len(crossings)} zero crossings")
        """
        # Classify each sample: +1 (above hysteresis), -1 (below -hysteresis), 0 (dead band).
        classified = np.zeros(len(flow_data), dtype=np.int8)
        classified[flow_data > self.hysteresis] = 1
        classified[flow_data < -self.hysteresis] = -1

        nonzero_indices = np.flatnonzero(classified)
        if len(nonzero_indices) == 0:
            logger.debug("Detected 0 zero crossings")
            return []

        nonzero_values = classified[nonzero_indices]

        # Candidates are the first classified sample plus each sign-change point.
        # These are exactly the state-transition events the scalar loop encounters,
        # reducing ~720K samples to ~1-2K candidates for the accept/reject pass.
        transition_mask = np.concatenate([[True], np.diff(nonzero_values) != 0])
        candidate_indices = nonzero_indices[transition_mask].tolist()
        candidate_values = nonzero_values[transition_mask].tolist()

        crossings: list[tuple[int, str]] = []
        last_crossing_idx = -1

        for idx, val in zip(candidate_indices, candidate_values, strict=True):
            direction = "positive" if val > 0 else "negative"
            if not crossings:
                crossings.append((idx, direction))
                last_crossing_idx = idx
            elif idx - last_crossing_idx > 5:
                crossings.append((idx, direction))
                last_crossing_idx = idx

        logger.debug(f"Detected {len(crossings)} zero crossings")
        return crossings

    def identify_breath_boundaries(
        self,
        crossings: list[tuple[int, str]],
        timestamps: np.ndarray,
        sample_rate: float,
        flow_data: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Group zero crossings into complete breath boundaries.

        A complete breath cycle is: positive → negative → positive
        (expiration → inspiration → expiration)

        Applies amplitude filter: (max - min) > 2 L/min

        Args:
            crossings: List of (index, direction) from detect_zero_crossings
            timestamps: Timestamp array for duration validation
            sample_rate: Sample rate in Hz
            flow_data: Flow values for amplitude validation

        Returns:
            List of (start_idx, end_idx) tuples defining breath boundaries

        Example:
            >>> boundaries = segmenter.identify_breath_boundaries(
            ...     crossings, timestamps, 25.0, flow
            ... )
        """
        boundaries = []

        i = 0
        while i < len(crossings) - 1:
            crossing_idx, direction = crossings[i]

            if direction == "positive":
                found_complete = False
                for j in range(i + 1, len(crossings)):
                    next_idx, next_dir = crossings[j]
                    if next_dir == "positive":
                        start_idx = crossing_idx
                        end_idx = next_idx

                        duration = timestamps[end_idx] - timestamps[start_idx]
                        if not (
                            self.min_breath_duration
                            <= duration
                            <= self.max_breath_duration
                        ):
                            i = j - 1
                            break

                        # Validate amplitude - lowered from 8.0 to 2.0 to detect breaths during low-flow periods
                        breath_segment = flow_data[start_idx:end_idx]
                        amplitude = np.max(breath_segment) - np.min(breath_segment)
                        if (
                            amplitude
                            <= BreathSegmentationConstants.MIN_BREATH_AMPLITUDE
                        ):
                            i = j - 1
                            break

                        boundaries.append((start_idx, end_idx))
                        found_complete = True

                        i = j - 1
                        break

                if not found_complete and i == len(crossings) - 2:
                    if i + 1 < len(crossings) and crossings[i + 1][1] == "negative":
                        start_idx = crossing_idx
                        end_idx = len(flow_data) - 1

                        duration = timestamps[end_idx] - timestamps[start_idx]
                        if (
                            self.min_breath_duration
                            <= duration
                            <= self.max_breath_duration
                        ):
                            breath_segment = flow_data[start_idx : end_idx + 1]
                            amplitude = np.max(breath_segment) - np.min(breath_segment)
                            if (
                                amplitude
                                > BreathSegmentationConstants.MIN_BREATH_AMPLITUDE
                            ):
                                boundaries.append((start_idx, end_idx))

            i += 1

        return boundaries

    def calculate_breath_metrics(
        self,
        breath_number: int,
        timestamps: np.ndarray,
        flow_values: np.ndarray,
        sample_rate: float,
        tv_history: list[float],
        all_breaths: list[BreathMetrics],
    ) -> BreathMetrics:
        """
        Calculate comprehensive metrics for a single breath.

        Args:
            breath_number: Sequential breath number
            timestamps: Timestamps for this breath segment
            flow_values: Flow values for this breath segment (L/min)
            sample_rate: Sample rate in Hz
            tv_history: List of previous TV values for smoothing
            all_breaths: List of all breaths processed so far

        Returns:
            BreathMetrics object with all calculated metrics

        Example:
            >>> metrics = segmenter.calculate_breath_metrics(
            ...     1, timestamps, flow, 25.0, tv_history, all_breaths
            ... )
        """
        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = end_time - start_time

        inspiration_mask = flow_values > 0
        inspiration_indices = np.where(inspiration_mask)[0]
        inspiration_values = flow_values[inspiration_mask]

        expiration_mask = flow_values < 0
        expiration_indices = np.where(expiration_mask)[0]
        expiration_values = flow_values[expiration_mask]

        has_inspiration = len(inspiration_indices) > 0
        has_expiration = len(expiration_indices) > 0
        is_complete = has_inspiration and has_expiration

        if has_inspiration and has_expiration:
            insp_time = len(inspiration_indices) / sample_rate
            exp_time = len(expiration_indices) / sample_rate
            peak_insp_flow = np.max(inspiration_values)
            peak_exp_flow = np.abs(np.min(expiration_values))

            # Middle time: transition from inspiration to expiration
            # This is where the last inspiration sample ends
            last_insp_idx = inspiration_indices[-1]
            middle_time = timestamps[last_insp_idx]
        elif has_inspiration:
            insp_time = len(inspiration_indices) / sample_rate
            exp_time = 0.0
            peak_insp_flow = np.max(inspiration_values)
            peak_exp_flow = 0.0
            middle_time = timestamps[-1]  # No expiration, so middle is at end
        elif has_expiration:
            insp_time = 0.0
            exp_time = len(expiration_indices) / sample_rate
            peak_insp_flow = 0.0
            peak_exp_flow = np.abs(np.min(expiration_values))
            middle_time = timestamps[0]  # No inspiration, so middle is at start
        else:
            insp_time = 0.0
            exp_time = 0.0
            peak_insp_flow = 0.0
            peak_exp_flow = 0.0
            middle_time = (start_time + end_time) / 2

        if exp_time > 0:
            i_e_ratio = insp_time / exp_time
        else:
            i_e_ratio = 0.0

        amplitude = peak_insp_flow + peak_exp_flow

        # Calculate tidal volume (integrate flow over inspiration)
        # Flow is in L/min, need to convert to L/s then integrate
        if has_inspiration:
            flow_L_per_s = inspiration_values / 60.0
            time_steps = 1.0 / sample_rate
            tidal_volume_L = np.trapezoid(flow_L_per_s, dx=time_steps)
            tidal_volume = float(tidal_volume_L * 1000.0)
        else:
            tidal_volume = 0.0

        # Calculate smoothed tidal volume (OSCAR's 5-point weighted average)
        tidal_volume_smoothed = self.calculate_smoothed_tidal_volume(
            tv_history, tidal_volume
        )

        if duration > 0:
            respiratory_rate = 60.0 / duration
        else:
            respiratory_rate = 0.0

        # Rolling window rate (OSCAR's method)
        respiratory_rate_rolling = self.calculate_rolling_respiratory_rate(
            all_breaths,
            len(all_breaths),
        )

        # Calculate minute ventilation using rolling RR (more stable)
        # MV = tidal_volume_smoothed (mL) × respiratory_rate_rolling (breaths/min)
        if respiratory_rate_rolling > 0:
            minute_ventilation = (
                tidal_volume_smoothed / 1000.0
            ) * respiratory_rate_rolling
        else:
            minute_ventilation = (tidal_volume_smoothed / 1000.0) * respiratory_rate

        return BreathMetrics(
            breath_number=breath_number,
            start_time=float(start_time),
            middle_time=float(middle_time),
            end_time=float(end_time),
            duration=float(duration),
            tidal_volume=float(tidal_volume),
            tidal_volume_smoothed=float(tidal_volume_smoothed),
            peak_inspiratory_flow=float(peak_insp_flow),
            peak_expiratory_flow=float(peak_exp_flow),
            inspiration_time=float(insp_time),
            expiration_time=float(exp_time),
            i_e_ratio=float(i_e_ratio),
            respiratory_rate=float(respiratory_rate),
            respiratory_rate_rolling=float(respiratory_rate_rolling),
            minute_ventilation=float(minute_ventilation),
            amplitude=float(amplitude),
            is_complete=is_complete,
        )

    def calculate_rolling_respiratory_rate(
        self,
        breaths: list[BreathMetrics],
        current_breath_idx: int,
        window_seconds: float = 60.0,
    ) -> float:
        """
        Calculate respiratory rate using rolling 60-second window.

        This matches OSCAR's algorithm (calcs.cpp lines 642-701) which counts
        breaths in the last minute with proportional weighting for partial breaths.

        Args:
            breaths: List of all breaths so far
            current_breath_idx: Index of current breath
            window_seconds: Rolling window size in seconds (default 60)

        Returns:
            Respiratory rate in breaths/min

        Example:
            >>> rr = segmenter.calculate_rolling_respiratory_rate(breaths, 10)
        """
        if current_breath_idx < 0 or current_breath_idx >= len(breaths):
            return 0.0

        current_breath = breaths[current_breath_idx]
        window_start = current_breath.end_time - window_seconds

        breath_count = 0.0

        for i in range(current_breath_idx, -1, -1):
            breath = breaths[i]

            if breath.end_time < window_start:
                break

            if breath.start_time < window_start:
                overlap = breath.end_time - window_start
                weight = overlap / breath.duration if breath.duration > 0 else 0
                breath_count += weight
            else:
                breath_count += 1.0

        if current_breath_idx == 0:
            actual_window = current_breath.end_time - current_breath.start_time
        else:
            first_breath = breaths[0]
            actual_window = current_breath.end_time - max(
                first_breath.start_time, window_start
            )

        if actual_window < window_seconds and actual_window > 0:
            breath_count *= window_seconds / actual_window

        return breath_count

    def calculate_smoothed_tidal_volume(
        self, tv_history: list[float], current_tv: float
    ) -> float:
        """
        Calculate smoothed tidal volume using 5-point weighted average.

        This matches OSCAR's algorithm (calcs.cpp lines 620-639):
        tv_smoothed = (tv[-3] + tv[-2] + tv[-1] + tv[current]*2) / 5

        Args:
            tv_history: List of previous TV values (last 3)
            current_tv: Current tidal volume

        Returns:
            Smoothed tidal volume in mL

        Example:
            >>> tv_smooth = segmenter.calculate_smoothed_tidal_volume(
            ...     [450, 460, 455], 465
            ... )
        """
        if len(tv_history) == 0:
            return current_tv
        elif len(tv_history) == 1:
            return (tv_history[0] + current_tv * 2) / 3
        elif len(tv_history) == 2:
            return (tv_history[0] + tv_history[1] + current_tv * 2) / 4
        else:
            return (
                tv_history[-3] + tv_history[-2] + tv_history[-1] + current_tv * 2
            ) / 5
