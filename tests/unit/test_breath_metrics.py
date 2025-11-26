"""
Unit tests for breath metrics calculation.

Tests tidal volume, respiratory rate, I:E ratio, and other breath-level metrics.
"""

import numpy as np
import pytest

from oscar_mcp.analysis.algorithms.breath_segmenter import BreathSegmenter
from tests.helpers.synthetic_data import create_session, generate_sinusoidal_breath
from tests.helpers.validation_helpers import assert_breath_valid


class TestTidalVolumeCalculation:
    """Test tidal volume numerical integration."""

    def test_tidal_volume_positive(self):
        """Tidal volume should always be non-negative."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0, amplitude=30.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            assert breath.tidal_volume >= 0

    def test_tidal_volume_reasonable_range(self):
        """Tidal volume should be in physiological range (200-800 mL typical)."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0, amplitude=30.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            # Reasonable range for adults
            assert 100 <= breath.tidal_volume <= 1500

    def test_tidal_volume_scales_with_amplitude(self):
        """Higher amplitude should give higher tidal volume."""
        segmenter = BreathSegmenter()

        t1, flow1 = generate_sinusoidal_breath(duration=4.0, amplitude=20.0)
        t2, flow2 = generate_sinusoidal_breath(duration=4.0, amplitude=40.0)

        breaths1 = segmenter.segment_breaths(t1, flow1, sample_rate=25.0)
        breaths2 = segmenter.segment_breaths(t2, flow2, sample_rate=25.0)

        if breaths1 and breaths2:
            assert breaths2[0].tidal_volume > breaths1[0].tidal_volume


class TestRespiratoryRateCalculation:
    """Test respiratory rate calculations."""

    def test_respiratory_rate_boundary_conditions(self):
        """RR should be in physiological range (5-60 breaths/min)."""
        segmenter = BreathSegmenter()
        t, flow = create_session(num_breaths=20, avg_duration=4.0, sample_rate=25.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            assert 5 <= breath.respiratory_rate <= 60

    def test_respiratory_rate_fast_breathing(self):
        """Fast breathing (short duration) should give high RR."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(
            duration=2.0, amplitude=30.0
        )  # 30 breaths/min

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        if breaths:
            # 2 second breath = 30 breaths/min
            assert breaths[0].respiratory_rate >= 25

    def test_respiratory_rate_slow_breathing(self):
        """Slow breathing (long duration) should give low RR."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(
            duration=6.0, amplitude=30.0
        )  # 10 breaths/min

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        if breaths:
            # 6 second breath = 10 breaths/min
            assert breaths[0].respiratory_rate <= 12

    def test_rolling_rr_vs_instantaneous(self):
        """Rolling RR should be more stable than instantaneous."""
        segmenter = BreathSegmenter()
        t, flow = create_session(
            num_breaths=30, duration_variability=1.0, sample_rate=25.0
        )

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        if len(breaths) >= 10:
            inst_rr = [b.respiratory_rate for b in breaths[5:]]
            roll_rr = [b.respiratory_rate_rolling for b in breaths[5:]]

            # Rolling should have lower std dev (more stable)
            assert np.std(roll_rr) <= np.std(inst_rr)


class TestIERatioCalculation:
    """Test inspiration:expiration ratio calculation."""

    def test_ie_ratio_normal_range(self):
        """I:E ratio should be in normal range (0.5-2.0 typical)."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            if breath.is_complete:
                # Physiological range
                assert 0.3 <= breath.i_e_ratio <= 3.0

    def test_ie_ratio_symmetric_breath(self):
        """Symmetric breath should have I:E ratio near 1.0."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0)  # Symmetric

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        if breaths and breaths[0].is_complete:
            # Should be close to 1:1
            assert 0.8 <= breaths[0].i_e_ratio <= 1.2


class TestAmplitudeCalculation:
    """Test peak-to-peak amplitude calculation."""

    def test_amplitude_accuracy(self):
        """Amplitude should match peak_insp + peak_exp (peak-to-peak).

        Note: peak_expiratory_flow is stored as absolute value, so we add them.
        """
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0, amplitude=30.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            # peak_exp_flow is already absolute value (see breath_segmenter.py:390)
            calculated_amp = breath.peak_inspiratory_flow + breath.peak_expiratory_flow
            assert abs(breath.amplitude - calculated_amp) < 0.01

    def test_amplitude_minimum_threshold(self):
        """All returned breaths should have amplitude > 8 L/min."""
        segmenter = BreathSegmenter()
        t, flow = create_session(num_breaths=20, avg_amplitude=30.0, sample_rate=25.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            assert breath.amplitude > 8.0


class TestBreathCompleteness:
    """Test breath completeness checking."""

    def test_complete_breath_has_both_phases(self):
        """Complete breath must have both inspiration and expiration."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            if breath.is_complete:
                assert breath.inspiration_time > 0
                assert breath.expiration_time > 0

    def test_incomplete_breaths_filtered(self):
        """Incomplete breaths should not be returned."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        # All returned breaths should be complete
        for breath in breaths:
            assert breath.is_complete


class TestBreathDurationValidation:
    """Test breath duration validation."""

    def test_duration_matches_timestamps(self):
        """Duration should equal end_time - start_time."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            expected_duration = breath.end_time - breath.start_time
            assert abs(breath.duration - expected_duration) < 0.01

    def test_duration_in_valid_range(self):
        """All breaths should have duration in valid range."""
        segmenter = BreathSegmenter(min_breath_duration=1.0, max_breath_duration=20.0)
        t, flow = create_session(num_breaths=20, sample_rate=25.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        for breath in breaths:
            assert 1.0 <= breath.duration <= 20.0


class TestBreathValidationHelpers:
    """Test breath validation helper functions."""

    def test_assert_breath_valid_passes_good_breath(self):
        """Valid breath should pass validation."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0, amplitude=30.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        if breaths:
            # Should not raise
            assert_breath_valid(breaths[0])

    def test_assert_breath_valid_catches_invalid(self):
        """Invalid breath should fail validation."""
        from oscar_mcp.analysis.algorithms.breath_segmenter import BreathMetrics

        # Create invalid breath (negative duration)
        bad_breath = BreathMetrics(
            breath_number=1,
            start_time=10.0,
            middle_time=5.0,  # Before start!
            end_time=0.0,  # Before start!
            duration=-5.0,  # Negative!
            tidal_volume=500.0,
            tidal_volume_smoothed=500.0,
            peak_inspiratory_flow=30.0,
            peak_expiratory_flow=20.0,
            inspiration_time=2.0,
            expiration_time=2.0,
            i_e_ratio=1.0,
            respiratory_rate=15.0,
            respiratory_rate_rolling=15.0,
            minute_ventilation=7.5,
            amplitude=50.0,
            is_complete=True,
        )

        with pytest.raises(AssertionError):
            assert_breath_valid(bad_breath)
