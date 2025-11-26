"""
Breath segmentation algorithm for flow waveform analysis.

This module provides algorithms for segmenting continuous flow waveform data
into individual breaths, identifying inspiration/expiration phases, and
calculating breath-level metrics.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from oscar_mcp.constants import BreathSegmentationConstants

logger = logging.getLogger(__name__)


@dataclass
class BreathPhases:
    """
    Inspiration and expiration phases of a single breath.

    Attributes:
        inspiration_indices: Indices where flow > 0 (breathing in)
        expiration_indices: Indices where flow < 0 (breathing out)
        inspiration_values: Flow values during inspiration
        expiration_values: Flow values during expiration
    """

    inspiration_indices: np.ndarray
    expiration_indices: np.ndarray
    inspiration_values: np.ndarray
    expiration_values: np.ndarray


@dataclass
class BreathMetrics:
    """
    Comprehensive metrics for a single breath.

    Attributes:
        breath_number: Sequential breath number in session
        start_time: Timestamp of breath start (seconds)
        middle_time: Timestamp of inspiration→expiration transition (seconds)
        end_time: Timestamp of breath end (seconds)
        duration: Total breath time (seconds)
        tidal_volume: Volume of air breathed in (mL)
        tidal_volume_smoothed: Smoothed TV using 5-point weighted average (mL)
        peak_inspiratory_flow: Maximum flow during inspiration (L/min)
        peak_expiratory_flow: Maximum absolute flow during expiration (L/min)
        inspiration_time: Duration of inspiration phase (seconds)
        expiration_time: Duration of expiration phase (seconds)
        i_e_ratio: Inspiration to expiration time ratio
        respiratory_rate: Instantaneous rate (60/duration) in breaths/min
        respiratory_rate_rolling: Rolling 60s window rate (breaths/min)
        minute_ventilation: Estimated ventilation using rolling RR (L/min)
        amplitude: Peak-to-peak amplitude (peak_insp - |peak_exp|) in L/min
        is_complete: Whether breath has both inspiration and expiration
    """

    breath_number: int
    start_time: float
    middle_time: float
    end_time: float
    duration: float
    tidal_volume: float
    tidal_volume_smoothed: float
    peak_inspiratory_flow: float
    peak_expiratory_flow: float
    inspiration_time: float
    expiration_time: float
    i_e_ratio: float
    respiratory_rate: float
    respiratory_rate_rolling: float
    minute_ventilation: float
    amplitude: float
    is_complete: bool


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
    ) -> List[BreathMetrics]:
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
        # Handle empty input
        if len(flow_data) == 0 or len(timestamps) == 0:
            logger.debug("Empty input arrays, returning no breaths")
            return []

        logger.debug(
            f"Segmenting {len(flow_data)} samples at {sample_rate} Hz "
            f"(duration: {timestamps[-1] - timestamps[0]:.1f}s)"
        )

        # Detect zero crossings
        crossings = self.detect_zero_crossings(flow_data)

        # Identify breath boundaries
        boundaries = self.identify_breath_boundaries(crossings, timestamps, sample_rate, flow_data)

        logger.debug(f"Identified {len(boundaries)} potential breaths")

        # Process each breath with rolling calculations
        breaths: List[BreathMetrics] = []
        tv_history: List[float] = []  # For 5-point TV smoothing

        for idx, (start_idx, end_idx) in enumerate(boundaries):
            breath_segment = flow_data[start_idx:end_idx]
            breath_timestamps = timestamps[start_idx:end_idx]

            # Calculate initial metrics (without rolling RR/smoothed TV)
            metrics = self.calculate_breath_metrics(
                breath_number=idx + 1,
                timestamps=breath_timestamps,
                flow_values=breath_segment,
                sample_rate=sample_rate,
                tv_history=tv_history,
                all_breaths=breaths,  # For rolling RR calculation
            )

            # Only include complete, valid breaths
            if metrics.is_complete:
                breaths.append(metrics)
                # Update TV history for smoothing (keep last 3)
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

    def detect_zero_crossings(self, flow_data: np.ndarray) -> List[Tuple[int, str]]:
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
        crossings = []

        # State machine for hysteresis
        # States: "positive" (flow > hysteresis), "negative" (flow < -hysteresis)
        current_state = None
        last_crossing_idx = -1

        for i, value in enumerate(flow_data):
            # Determine current state based on hysteresis
            if value > self.hysteresis:
                new_state = "positive"
            elif value < -self.hysteresis:
                new_state = "negative"
            else:
                # In hysteresis band - maintain current state
                continue

            # Check for state transition (zero crossing)
            if current_state is None:
                # First state establishment - record as initial crossing
                crossings.append((i, new_state))
                last_crossing_idx = i
            elif new_state != current_state:
                # State transition - record if enough time has passed since last crossing
                if i - last_crossing_idx > 5:  # At least 5 samples apart
                    crossings.append((i, new_state))
                    last_crossing_idx = i

            current_state = new_state

        logger.debug(f"Detected {len(crossings)} zero crossings")
        return crossings

    def identify_breath_boundaries(
        self,
        crossings: List[Tuple[int, str]],
        timestamps: np.ndarray,
        sample_rate: float,
        flow_data: np.ndarray,
    ) -> List[Tuple[int, int]]:
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

        # Look for positive → negative → positive sequences
        i = 0
        while i < len(crossings) - 1:
            crossing_idx, direction = crossings[i]

            # Start breath at positive crossing (start of inspiration)
            if direction == "positive":
                # Find next positive crossing (end of next expiration)
                found_complete = False
                for j in range(i + 1, len(crossings)):
                    next_idx, next_dir = crossings[j]
                    if next_dir == "positive":
                        # Found complete breath boundary
                        start_idx = crossing_idx
                        end_idx = next_idx

                        # Validate duration
                        duration = timestamps[end_idx] - timestamps[start_idx]
                        if not (self.min_breath_duration <= duration <= self.max_breath_duration):
                            # Move to this crossing for next iteration
                            i = j - 1
                            break

                        # Validate amplitude - lowered from 8.0 to 2.0 to detect breaths during low-flow periods
                        breath_segment = flow_data[start_idx:end_idx]
                        amplitude = np.max(breath_segment) - np.min(breath_segment)
                        if amplitude <= BreathSegmentationConstants.MIN_BREATH_AMPLITUDE:
                            # Insufficient amplitude - skip this breath
                            i = j - 1
                            break

                        # Passed all validations
                        boundaries.append((start_idx, end_idx))
                        found_complete = True

                        # Move to this crossing for next iteration
                        i = j - 1
                        break

                # If no complete boundary found, check for incomplete breath at end
                if not found_complete and i == len(crossings) - 2:
                    # This might be a partial breath at the end
                    # Check if there's a negative crossing after this positive
                    if i + 1 < len(crossings) and crossings[i + 1][1] == "negative":
                        start_idx = crossing_idx
                        end_idx = len(flow_data) - 1  # Extend to end of data

                        # Validate duration and amplitude (lowered from 8.0 to 2.0)
                        duration = timestamps[end_idx] - timestamps[start_idx]
                        if self.min_breath_duration <= duration <= self.max_breath_duration:
                            breath_segment = flow_data[start_idx : end_idx + 1]
                            amplitude = np.max(breath_segment) - np.min(breath_segment)
                            if amplitude > BreathSegmentationConstants.MIN_BREATH_AMPLITUDE:
                                boundaries.append((start_idx, end_idx))

            i += 1

        return boundaries

    def classify_breath_phase(
        self, timestamps: np.ndarray, flow_values: np.ndarray
    ) -> BreathPhases:
        """
        Separate breath into inspiration and expiration phases.

        Args:
            timestamps: 1D array of timestamps for this breath
            flow_values: 1D array of flow values for this breath

        Returns:
            BreathPhases object with separated inspiration/expiration data

        Example:
            >>> phases = segmenter.classify_breath_phase(t, flow)
            >>> print(f"Inspiration: {len(phases.inspiration_indices)} samples")
        """
        # Inspiration: flow > 0
        inspiration_mask = flow_values > 0
        inspiration_indices = np.where(inspiration_mask)[0]
        inspiration_values = flow_values[inspiration_mask]

        # Expiration: flow < 0
        expiration_mask = flow_values < 0
        expiration_indices = np.where(expiration_mask)[0]
        expiration_values = flow_values[expiration_mask]

        return BreathPhases(
            inspiration_indices=inspiration_indices,
            expiration_indices=expiration_indices,
            inspiration_values=inspiration_values,
            expiration_values=expiration_values,
        )

    def calculate_breath_metrics(
        self,
        breath_number: int,
        timestamps: np.ndarray,
        flow_values: np.ndarray,
        sample_rate: float,
        tv_history: List[float],
        all_breaths: List[BreathMetrics],
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
        # Basic timing
        start_time = timestamps[0]
        end_time = timestamps[-1]
        duration = end_time - start_time

        # Separate phases
        phases = self.classify_breath_phase(timestamps, flow_values)

        # Check if breath is complete (has both inspiration and expiration)
        has_inspiration = len(phases.inspiration_indices) > 0
        has_expiration = len(phases.expiration_indices) > 0
        is_complete = has_inspiration and has_expiration

        # Calculate phase durations and find middle (transition point)
        if has_inspiration and has_expiration:
            insp_time = len(phases.inspiration_indices) / sample_rate
            exp_time = len(phases.expiration_indices) / sample_rate
            peak_insp_flow = np.max(phases.inspiration_values)
            peak_exp_flow = np.abs(np.min(phases.expiration_values))

            # Middle time: transition from inspiration to expiration
            # This is where the last inspiration sample ends
            last_insp_idx = phases.inspiration_indices[-1]
            middle_time = timestamps[last_insp_idx]
        elif has_inspiration:
            insp_time = len(phases.inspiration_indices) / sample_rate
            exp_time = 0.0
            peak_insp_flow = np.max(phases.inspiration_values)
            peak_exp_flow = 0.0
            middle_time = timestamps[-1]  # No expiration, so middle is at end
        elif has_expiration:
            insp_time = 0.0
            exp_time = len(phases.expiration_indices) / sample_rate
            peak_insp_flow = 0.0
            peak_exp_flow = np.abs(np.min(phases.expiration_values))
            middle_time = timestamps[0]  # No inspiration, so middle is at start
        else:
            insp_time = 0.0
            exp_time = 0.0
            peak_insp_flow = 0.0
            peak_exp_flow = 0.0
            middle_time = (start_time + end_time) / 2

        # Calculate I:E ratio
        if exp_time > 0:
            i_e_ratio = insp_time / exp_time
        else:
            i_e_ratio = 0.0

        # Calculate amplitude (peak-to-peak)
        # peak_exp_flow is already positive (abs of min), so add them
        amplitude = peak_insp_flow + peak_exp_flow

        # Calculate tidal volume (integrate flow over inspiration)
        # Flow is in L/min, need to convert to L/s then integrate
        if has_inspiration:
            # Numerical integration using trapezoidal rule
            # Convert L/min to L/s by dividing by 60
            flow_L_per_s = phases.inspiration_values / 60.0
            time_steps = 1.0 / sample_rate  # Time between samples
            tidal_volume_L = np.trapezoid(flow_L_per_s, dx=time_steps)  # Liters
            tidal_volume = float(tidal_volume_L * 1000.0)  # Convert to mL
        else:
            tidal_volume = 0.0

        # Calculate smoothed tidal volume (OSCAR's 5-point weighted average)
        tidal_volume_smoothed = self.calculate_smoothed_tidal_volume(tv_history, tidal_volume)

        # Calculate respiratory rates
        # Instantaneous rate (simple)
        if duration > 0:
            respiratory_rate = 60.0 / duration  # breaths/min
        else:
            respiratory_rate = 0.0

        # Rolling window rate (OSCAR's method)
        respiratory_rate_rolling = self.calculate_rolling_respiratory_rate(
            all_breaths,
            len(all_breaths),  # Current breath will be added next
        )

        # Calculate minute ventilation using rolling RR (more stable)
        # MV = tidal_volume_smoothed (mL) × respiratory_rate_rolling (breaths/min)
        if respiratory_rate_rolling > 0:
            minute_ventilation = (tidal_volume_smoothed / 1000.0) * respiratory_rate_rolling
        else:
            # Fallback to instantaneous if rolling not available
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
        breaths: List[BreathMetrics],
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

        # Step backward through breaths counting those in window
        for i in range(current_breath_idx, -1, -1):
            breath = breaths[i]

            # Check if breath ends before window
            if breath.end_time < window_start:
                break

            # Check if breath starts before window (partial breath)
            if breath.start_time < window_start:
                # Weight proportionally
                overlap = breath.end_time - window_start
                weight = overlap / breath.duration if breath.duration > 0 else 0
                breath_count += weight
            else:
                # Fully in window
                breath_count += 1.0

        # Calculate actual window duration (may be less than 60s at start)
        if current_breath_idx == 0:
            actual_window = current_breath.end_time - current_breath.start_time
        else:
            first_breath = breaths[0]
            actual_window = current_breath.end_time - max(first_breath.start_time, window_start)

        # Normalize to full minute if window is shorter
        if actual_window < window_seconds and actual_window > 0:
            breath_count *= window_seconds / actual_window

        return breath_count

    def calculate_smoothed_tidal_volume(self, tv_history: List[float], current_tv: float) -> float:
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
            # Full 5-point average (last 3 + current*2)
            return (tv_history[-3] + tv_history[-2] + tv_history[-1] + current_tv * 2) / 5

    def detect_flow_restriction(
        self,
        breaths: List[BreathMetrics],
        restriction_percent: float = 50.0,
        duration_threshold: float = 10.0,
    ) -> List[Tuple[int, int]]:
        """
        Detect flow restriction events using OSCAR's percentile-based algorithm.

        This implements OSCAR's user event flagging algorithm (calcs.cpp lines 766-879)
        which detects sustained periods where breath amplitude falls below a
        threshold calculated from the 60th percentile of all breaths.

        Args:
            breaths: List of BreathMetrics objects
            restriction_percent: Percentage threshold (e.g., 50 = flag when <50% of peak)
            duration_threshold: Minimum duration in seconds to flag

        Returns:
            List of (start_breath_idx, end_breath_idx) tuples for restriction events

        Example:
            >>> restrictions = segmenter.detect_flow_restriction(
            ...     breaths, restriction_percent=50.0, duration_threshold=10.0
            ... )
        """
        if len(breaths) == 0:
            return []

        # Calculate 60th percentile of breath amplitudes (OSCAR's approach)
        amplitudes = np.array([b.amplitude for b in breaths])
        percentile_60 = np.percentile(amplitudes, 60)

        # Calculate cutoff value
        cutoff_amplitude = percentile_60 * (restriction_percent / 100.0)

        logger.debug(
            f"Flow restriction detection: p60={percentile_60:.2f}, "
            f"cutoff={cutoff_amplitude:.2f} ({restriction_percent}%)"
        )

        # Scan for restriction events
        restriction_events = []
        current_event_start = None
        current_event_duration = 0.0

        for i, breath in enumerate(breaths):
            if breath.amplitude < cutoff_amplitude:
                # Breath is restricted
                if current_event_start is None:
                    # Start new event
                    current_event_start = i
                    current_event_duration = breath.duration
                else:
                    # Continue current event
                    current_event_duration += breath.duration
            else:
                # Breath is not restricted
                if current_event_start is not None:
                    # End current event if it meets duration threshold
                    if current_event_duration >= duration_threshold:
                        restriction_events.append((current_event_start, i - 1))
                        logger.debug(
                            f"Flow restriction event: breaths {current_event_start}-{i - 1} "
                            f"({current_event_duration:.1f}s)"
                        )
                    current_event_start = None
                    current_event_duration = 0.0

        # Check for event at end of session
        if current_event_start is not None:
            if current_event_duration >= duration_threshold:
                restriction_events.append((current_event_start, len(breaths) - 1))
                logger.debug(
                    f"Flow restriction event: breaths {current_event_start}-{len(breaths) - 1} "
                    f"({current_event_duration:.1f}s)"
                )

        logger.info(
            f"Detected {len(restriction_events)} flow restriction events "
            f"(cutoff: {restriction_percent}%, duration: ≥{duration_threshold}s)"
        )

        return restriction_events

    def handle_incomplete_breaths(self, breaths: List[BreathMetrics]) -> List[BreathMetrics]:
        """
        Filter or merge incomplete breaths.

        Incomplete breaths occur at segment boundaries or during mask-off
        periods. This method filters them out for now. Future enhancement
        could attempt to merge with adjacent segments.

        Args:
            breaths: List of BreathMetrics including incomplete breaths

        Returns:
            List of complete BreathMetrics only

        Example:
            >>> complete_breaths = segmenter.handle_incomplete_breaths(breaths)
        """
        complete_breaths = [b for b in breaths if b.is_complete]

        removed_count = len(breaths) - len(complete_breaths)
        if removed_count > 0:
            logger.debug(f"Filtered out {removed_count} incomplete breaths")

        return complete_breaths
