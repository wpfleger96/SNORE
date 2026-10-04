"""
Validation helper functions for testing breath and feature data.

Provides assertion helpers and validation utilities for test assertions.
"""

from snore.analysis.shared.breath_segmenter import BreathMetrics
from snore.analysis.shared.feature_extractors import PeakFeatures, ShapeFeatures


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
    assert breath.duration > 0, "Breath duration must be positive"
    assert min_duration <= breath.duration <= max_duration, (
        f"Breath duration {breath.duration}s outside valid range [{min_duration}, {max_duration}]"
    )

    assert breath.start_time >= 0, "Start time must be non-negative"
    assert breath.end_time > breath.start_time, "End time must be after start time"
    assert breath.start_time <= breath.middle_time <= breath.end_time, (
        "Middle time must be between start and end"
    )

    assert breath.amplitude >= min_amplitude, (
        f"Breath amplitude {breath.amplitude} below minimum {min_amplitude}"
    )

    assert breath.tidal_volume >= 0, "Tidal volume must be non-negative"
    assert breath.tidal_volume_smoothed >= 0, (
        "Smoothed tidal volume must be non-negative"
    )

    assert breath.peak_inspiratory_flow > 0, "Peak inspiratory flow must be positive"
    assert breath.peak_expiratory_flow >= 0, "Peak expiratory flow must be non-negative"

    assert breath.inspiration_time >= 0, "Inspiration time must be non-negative"
    assert breath.expiration_time >= 0, "Expiration time must be non-negative"

    assert breath.i_e_ratio >= 0, "I:E ratio must be non-negative"

    assert 5 <= breath.respiratory_rate <= 60, (
        f"Respiratory rate {breath.respiratory_rate} outside physiological range [5, 60]"
    )
    assert breath.respiratory_rate_rolling >= 0, "Rolling RR must be non-negative"

    assert breath.minute_ventilation >= 0, "Minute ventilation must be non-negative"

    if breath.is_complete:
        assert breath.inspiration_time > 0, "Complete breath must have inspiration"
        assert breath.expiration_time > 0, "Complete breath must have expiration"


def assert_features_in_range(
    shape: ShapeFeatures | None = None,
    peak: PeakFeatures | None = None,
) -> None:
    """
    Assert that extracted features are within valid ranges.

    Args:
        shape: Shape features to validate
        peak: Peak features to validate

    Raises:
        AssertionError: If features are out of valid ranges
    """
    if shape is not None:
        assert 0 <= shape.flatness_index <= 1, (
            f"Flatness index {shape.flatness_index} outside [0, 1]"
        )

        assert shape.plateau_duration >= 0, "Plateau duration must be non-negative"
        assert shape.plateau_duration < 10, (
            f"Plateau duration {shape.plateau_duration}s too long"
        )

        assert -1 <= shape.symmetry_score <= 1, (
            f"Symmetry score {shape.symmetry_score} outside [-1, 1]"
        )

        assert -10 <= shape.kurtosis <= 10, (
            f"Kurtosis {shape.kurtosis} outside reasonable range"
        )

        assert shape.rise_time >= 0, "Rise time must be non-negative"
        assert shape.fall_time >= 0, "Fall time must be non-negative"
        assert shape.rise_time < 5, f"Rise time {shape.rise_time}s too long"
        assert shape.fall_time < 5, f"Fall time {shape.fall_time}s too long"

    if peak is not None:
        assert peak.peak_count >= 0, "Peak count must be non-negative"
        assert peak.peak_count <= 10, f"Peak count {peak.peak_count} too high"

        for i, pos in enumerate(peak.peak_positions):
            assert 0 <= pos <= 1, f"Peak position {i} = {pos} outside [0, 1]"

        for i, prom in enumerate(peak.peak_prominences):
            assert prom >= 0, f"Peak prominence {i} = {prom} must be non-negative"

        for i, interval in enumerate(peak.inter_peak_intervals):
            assert interval > 0, (
                f"Inter-peak interval {i} = {interval} must be positive"
            )
