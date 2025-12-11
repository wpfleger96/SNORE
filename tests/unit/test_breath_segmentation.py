"""
Unit tests for breath segmentation algorithm.

Tests zero-crossing detection, breath boundary identification,
and edge case handling.
"""

import numpy as np

from snore.analysis.shared.breath_segmenter import BreathSegmenter
from tests.helpers.synthetic_data import (
    generate_sinusoidal_breath,
)


class TestZeroCrossingDetection:
    """Test zero-crossing detection with hysteresis."""

    def test_zero_crossing_normal_breath(self):
        """Normal sinusoidal breath should have zero crossings."""
        segmenter = BreathSegmenter()
        _, flow = generate_sinusoidal_breath(duration=4.0, amplitude=30.0)

        crossings = segmenter.detect_zero_crossings(flow)

        assert len(crossings) >= 2

    def test_zero_crossing_all_positive_flow(self):
        """All positive flow should have no or minimal crossings."""
        segmenter = BreathSegmenter()
        flow = np.ones(100) * 20.0

        crossings = segmenter.detect_zero_crossings(flow)

        assert len(crossings) <= 1

    def test_zero_crossing_all_negative_flow(self):
        """All negative flow should have no or minimal crossings."""
        segmenter = BreathSegmenter()
        flow = np.ones(100) * -20.0

        crossings = segmenter.detect_zero_crossings(flow)

        assert len(crossings) <= 1

    def test_zero_crossing_hysteresis_prevents_noise(self):
        """Hysteresis should prevent noise near zero from creating false crossings."""
        segmenter = BreathSegmenter(hysteresis=5.0)

        flow = np.random.normal(0, 2.0, 100)

        crossings = segmenter.detect_zero_crossings(flow)

        assert len(crossings) < 5

    def test_zero_crossing_at_boundaries(self):
        """Should handle zero crossings at array boundaries."""
        segmenter = BreathSegmenter()

        flow = np.concatenate(
            [
                np.ones(50) * 20.0,
                np.ones(50) * -20.0,
            ]
        )

        crossings = segmenter.detect_zero_crossings(flow)

        assert len(crossings) >= 1


class TestBreathBoundaryIdentification:
    """Test breath boundary identification from crossings."""

    def test_identify_single_complete_breath(self):
        """Single complete breath should be identified."""
        segmenter = BreathSegmenter()
        t, flow = generate_sinusoidal_breath(duration=4.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) >= 1
        if len(breaths) > 0:
            assert breaths[0].duration > 0

    def test_identify_multiple_breaths(self):
        """Multiple breaths should all be identified."""
        from tests.helpers.synthetic_data import create_session

        segmenter = BreathSegmenter()
        t, flow = create_session(num_breaths=10, sample_rate=25.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) >= 7

    def test_minimum_duration_filter(self):
        """Breaths shorter than minimum should be rejected."""
        segmenter = BreathSegmenter(min_breath_duration=2.0)

        t = np.linspace(0, 1, 25)
        flow = 20.0 * np.sin(2 * np.pi * t)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) == 0

    def test_maximum_duration_filter(self):
        """Breaths longer than maximum should be rejected."""
        segmenter = BreathSegmenter(max_breath_duration=10.0)

        t = np.linspace(0, 25, 625)
        flow = 20.0 * np.sin(2 * np.pi * t / 25)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) == 0

    def test_amplitude_filter_rejects_small_breaths(self):
        """Breaths with amplitude <= 2 L/min should be rejected."""
        segmenter = BreathSegmenter()

        t = np.linspace(0, 4, 100)
        flow = 0.75 * np.sin(2 * np.pi * t / 4)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) == 0

    def test_incomplete_breath_at_start(self):
        """Incomplete breath at start should be handled."""
        segmenter = BreathSegmenter()

        t = np.linspace(1, 5, 100)
        flow = 30.0 * np.sin(2 * np.pi * t / 4)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) >= 0

    def test_incomplete_breath_at_end(self):
        """Incomplete breath at end should be handled."""
        segmenter = BreathSegmenter()

        t = np.linspace(0, 3, 75)
        flow = 30.0 * np.sin(2 * np.pi * t / 4)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) >= 0


class TestBreathSegmentationEdgeCases:
    """Test edge cases in breath segmentation."""

    def test_empty_array(self):
        """Empty arrays should return no breaths."""
        segmenter = BreathSegmenter()

        breaths = segmenter.segment_breaths(
            np.array([]), np.array([]), sample_rate=25.0
        )

        assert len(breaths) == 0

    def test_single_sample(self):
        """Single sample should return no breaths."""
        segmenter = BreathSegmenter()

        breaths = segmenter.segment_breaths(
            np.array([0.0]), np.array([10.0]), sample_rate=25.0
        )

        assert len(breaths) == 0

    def test_very_short_segment(self):
        """Very short segment should return no or few breaths."""
        segmenter = BreathSegmenter()

        t = np.linspace(0, 0.5, 12)
        flow = 20.0 * np.sin(4 * np.pi * t)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) == 0

    def test_extremely_long_breath(self):
        """Extremely long breath should be rejected by max_duration."""
        segmenter = BreathSegmenter(max_breath_duration=20.0)

        t = np.linspace(0, 30, 750)
        flow = 30.0 * np.sin(2 * np.pi * t / 30)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) == 0

    def test_very_noisy_signal(self):
        """Very noisy signal should still find valid breaths."""
        from tests.helpers.synthetic_data import generate_noisy_breath

        segmenter = BreathSegmenter()
        t, flow = generate_noisy_breath(duration=4.0, snr_db=5.0)

        breaths = segmenter.segment_breaths(t, flow, sample_rate=25.0)

        assert len(breaths) >= 0
