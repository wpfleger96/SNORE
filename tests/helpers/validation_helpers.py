"""
Validation helper functions for testing breath and feature data.

Provides assertion helpers and validation utilities for test assertions.
"""

import numpy as np
from typing import Optional, Dict, Any

from oscar_mcp.analysis.algorithms.breath_segmenter import BreathMetrics
from oscar_mcp.analysis.algorithms.feature_extractors import (
    ShapeFeatures,
    PeakFeatures,
    StatisticalFeatures,
)


def assert_breath_valid(
    breath: BreathMetrics,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    min_amplitude: float = 2.0,
) -> None:
    """
    Assert that a breath has valid physiological characteristics.

    Args:
        breath: Breath to validate
        min_duration: Minimum valid duration in seconds
        max_duration: Maximum valid duration in seconds
        min_amplitude: Minimum valid amplitude in L/min

    Raises:
        AssertionError: If breath is invalid
    """
    # Duration checks
    assert breath.duration > 0, "Breath duration must be positive"
    assert min_duration <= breath.duration <= max_duration, (
        f"Breath duration {breath.duration}s outside valid range [{min_duration}, {max_duration}]"
    )

    # Timing checks
    assert breath.start_time >= 0, "Start time must be non-negative"
    assert breath.end_time > breath.start_time, "End time must be after start time"
    assert breath.start_time <= breath.middle_time <= breath.end_time, (
        "Middle time must be between start and end"
    )

    # Amplitude checks
    assert breath.amplitude >= min_amplitude, (
        f"Breath amplitude {breath.amplitude} below minimum {min_amplitude}"
    )

    # Tidal volume checks
    assert breath.tidal_volume >= 0, "Tidal volume must be non-negative"
    assert breath.tidal_volume_smoothed >= 0, "Smoothed tidal volume must be non-negative"

    # Flow checks
    assert breath.peak_inspiratory_flow > 0, "Peak inspiratory flow must be positive"
    assert breath.peak_expiratory_flow >= 0, "Peak expiratory flow must be non-negative"

    # Phase duration checks
    assert breath.inspiration_time >= 0, "Inspiration time must be non-negative"
    assert breath.expiration_time >= 0, "Expiration time must be non-negative"

    # I:E ratio check
    assert breath.i_e_ratio >= 0, "I:E ratio must be non-negative"

    # Respiratory rate checks (physiologically reasonable: 5-60 breaths/min)
    assert 5 <= breath.respiratory_rate <= 60, (
        f"Respiratory rate {breath.respiratory_rate} outside physiological range [5, 60]"
    )
    assert breath.respiratory_rate_rolling >= 0, "Rolling RR must be non-negative"

    # Minute ventilation check
    assert breath.minute_ventilation >= 0, "Minute ventilation must be non-negative"

    # Completeness check
    if breath.is_complete:
        assert breath.inspiration_time > 0, "Complete breath must have inspiration"
        assert breath.expiration_time > 0, "Complete breath must have expiration"


def assert_features_in_range(
    shape: Optional[ShapeFeatures] = None,
    peak: Optional[PeakFeatures] = None,
    statistical: Optional[StatisticalFeatures] = None,
) -> None:
    """
    Assert that extracted features are within valid ranges.

    Args:
        shape: Shape features to validate
        peak: Peak features to validate
        statistical: Statistical features to validate

    Raises:
        AssertionError: If features are out of valid ranges
    """
    if shape is not None:
        # Flatness index: 0-1
        assert 0 <= shape.flatness_index <= 1, (
            f"Flatness index {shape.flatness_index} outside [0, 1]"
        )

        # Plateau duration: non-negative, reasonable
        assert shape.plateau_duration >= 0, "Plateau duration must be non-negative"
        assert shape.plateau_duration < 10, f"Plateau duration {shape.plateau_duration}s too long"

        # Symmetry score: -1 to 1
        assert -1 <= shape.symmetry_score <= 1, (
            f"Symmetry score {shape.symmetry_score} outside [-1, 1]"
        )

        # Kurtosis: reasonable range
        assert -10 <= shape.kurtosis <= 10, f"Kurtosis {shape.kurtosis} outside reasonable range"

        # Rise/fall times: non-negative, reasonable
        assert shape.rise_time >= 0, "Rise time must be non-negative"
        assert shape.fall_time >= 0, "Fall time must be non-negative"
        assert shape.rise_time < 5, f"Rise time {shape.rise_time}s too long"
        assert shape.fall_time < 5, f"Fall time {shape.fall_time}s too long"

    if peak is not None:
        # Peak count: non-negative, reasonable
        assert peak.peak_count >= 0, "Peak count must be non-negative"
        assert peak.peak_count <= 10, f"Peak count {peak.peak_count} too high"

        # Peak positions: 0-1 range
        for i, pos in enumerate(peak.peak_positions):
            assert 0 <= pos <= 1, f"Peak position {i} = {pos} outside [0, 1]"

        # Peak prominences: non-negative
        for i, prom in enumerate(peak.peak_prominences):
            assert prom >= 0, f"Peak prominence {i} = {prom} must be non-negative"

        # Inter-peak intervals: positive
        for i, interval in enumerate(peak.inter_peak_intervals):
            assert interval > 0, f"Inter-peak interval {i} = {interval} must be positive"

    if statistical is not None:
        # Mean and median: should be reasonable for flow
        assert -100 <= statistical.mean <= 100, (
            f"Mean {statistical.mean} outside reasonable flow range"
        )
        assert -100 <= statistical.median <= 100, (
            f"Median {statistical.median} outside reasonable flow range"
        )

        # Std dev: non-negative
        assert statistical.std_dev >= 0, "Standard deviation must be non-negative"

        # Percentiles: should be ordered
        assert statistical.percentile_25 <= statistical.percentile_50, (
            "25th percentile should be <= median"
        )
        assert statistical.percentile_50 <= statistical.percentile_75, (
            "Median should be <= 75th percentile"
        )
        assert statistical.percentile_75 <= statistical.percentile_95, (
            "75th percentile should be <= 95th percentile"
        )

        # Coefficient of variation: non-negative
        assert statistical.coefficient_of_variation >= 0, (
            "Coefficient of variation must be non-negative"
        )

        # Zero crossing rate: 0-1
        assert 0 <= statistical.zero_crossing_rate <= 1, (
            f"Zero crossing rate {statistical.zero_crossing_rate} outside [0, 1]"
        )


def assert_no_data_corruption(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> None:
    """
    Assert that waveform data is not corrupted.

    Args:
        timestamps: Timestamp array
        values: Value array

    Raises:
        AssertionError: If data appears corrupted
    """
    # Same length
    assert len(timestamps) == len(values), "Timestamps and values must have same length"

    # No NaN or Inf
    assert np.all(np.isfinite(timestamps)), "Timestamps contain NaN or Inf"
    assert np.all(np.isfinite(values)), "Values contain NaN or Inf"

    # Timestamps strictly increasing
    assert np.all(np.diff(timestamps) > 0), "Timestamps must be strictly increasing"

    # Non-empty
    assert len(timestamps) > 0, "Data arrays must not be empty"


def compare_breaths(
    breath1: BreathMetrics,
    breath2: BreathMetrics,
    tolerance: float = 0.1,
) -> dict:
    """
    Compare two breaths and return similarity metrics.

    Args:
        breath1: First breath
        breath2: Second breath
        tolerance: Relative tolerance for comparison (0.1 = 10%)

    Returns:
        Dict with comparison results and similarity scores
    """
    results: Dict[str, Any] = {
        "duration_match": abs(breath1.duration - breath2.duration) / breath1.duration < tolerance,
        "amplitude_match": abs(breath1.amplitude - breath2.amplitude) / breath1.amplitude
        < tolerance,
        "tv_match": abs(breath1.tidal_volume - breath2.tidal_volume) / breath1.tidal_volume
        < tolerance,
        "rr_match": abs(breath1.respiratory_rate - breath2.respiratory_rate)
        / breath1.respiratory_rate
        < tolerance,
        "ie_match": abs(breath1.i_e_ratio - breath2.i_e_ratio) / max(breath1.i_e_ratio, 0.1)
        < tolerance,
    }

    results["overall_match"] = all(results.values())
    results["match_score"] = sum(results.values()) / len(results)

    return results


def assert_smoothed_close_to_raw(
    raw_values: np.ndarray,
    smoothed_values: np.ndarray,
    max_deviation_percent: float = 20.0,
) -> None:
    """
    Assert that smoothed values are close to raw values.

    Args:
        raw_values: Raw unsmoothed values
        smoothed_values: Smoothed values
        max_deviation_percent: Maximum allowed deviation percentage

    Raises:
        AssertionError: If smoothing deviates too much
    """
    assert len(raw_values) == len(smoothed_values), "Raw and smoothed arrays must have same length"

    deviations = np.abs(raw_values - smoothed_values) / np.abs(raw_values + 1e-10)
    max_deviation = np.max(deviations) * 100

    assert max_deviation <= max_deviation_percent, (
        f"Maximum smoothing deviation {max_deviation:.1f}% exceeds "
        f"threshold {max_deviation_percent}%"
    )


def assert_rolling_window_accurate(
    instantaneous_values: np.ndarray,
    rolling_values: np.ndarray,
    tolerance_percent: float = 15.0,
) -> None:
    """
    Assert that rolling window values are reasonable vs instantaneous.

    Args:
        instantaneous_values: Instantaneous measurements
        rolling_values: Rolling window averages
        tolerance_percent: Allowed deviation percentage

    Raises:
        AssertionError: If rolling values deviate too much
    """
    assert len(instantaneous_values) == len(rolling_values), "Arrays must have same length"

    # Rolling should smooth out extremes
    inst_std = np.std(instantaneous_values)
    roll_std = np.std(rolling_values)

    assert roll_std <= inst_std, "Rolling window should smooth (reduce std dev)"

    # Means should be similar
    inst_mean = np.mean(instantaneous_values)
    roll_mean = np.mean(rolling_values)
    deviation_pct = abs(inst_mean - roll_mean) / inst_mean * 100

    assert deviation_pct <= tolerance_percent, (
        f"Rolling mean deviates {deviation_pct:.1f}% from instantaneous "
        f"(threshold: {tolerance_percent}%)"
    )
