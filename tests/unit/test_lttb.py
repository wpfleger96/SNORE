"""Unit tests for LTTB downsampling algorithm."""

import numpy as np
import pytest

from snore.services.lttb import lttb_downsample


class TestLTTBDownsampling:
    """Tests for Largest Triangle Three Buckets algorithm."""

    def test_passthrough_when_below_target(self):
        """When data has fewer points than target, return unchanged."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([10.0, 20.0, 15.0, 25.0])

        t_down, v_down = lttb_downsample(timestamps, values, target_points=10)

        np.testing.assert_array_equal(t_down, timestamps)
        np.testing.assert_array_equal(v_down, values)

    def test_passthrough_when_equal_target(self):
        """When data has exactly target points, return unchanged."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([10.0, 20.0, 15.0, 25.0, 18.0])

        t_down, v_down = lttb_downsample(timestamps, values, target_points=5)

        np.testing.assert_array_equal(t_down, timestamps)
        np.testing.assert_array_equal(v_down, values)

    def test_preserves_endpoints(self):
        """First and last points are always preserved."""
        timestamps = np.arange(0, 100, dtype=float)
        values = np.sin(timestamps / 10)

        t_down, v_down = lttb_downsample(timestamps, values, target_points=20)

        assert t_down[0] == timestamps[0]
        assert t_down[-1] == timestamps[-1]
        assert v_down[0] == values[0]
        assert v_down[-1] == values[-1]

    def test_reduces_to_target_size(self):
        """Output length matches target_points exactly."""
        timestamps = np.arange(0, 1000, dtype=float)
        values = np.random.randn(1000)

        target = 50
        t_down, v_down = lttb_downsample(timestamps, values, target_points=target)

        assert len(t_down) == target
        assert len(v_down) == target

    def test_preserves_peaks_and_valleys(self):
        """Triangle wave peaks/valleys are preserved."""
        timestamps = np.arange(0, 100, dtype=float)
        values = np.abs(timestamps % 20 - 10)

        t_down, v_down = lttb_downsample(timestamps, values, target_points=15)

        max_original = values.max()
        max_downsampled = v_down.max()
        min_original = values.min()
        min_downsampled = v_down.min()

        assert max_downsampled == max_original
        assert min_downsampled == min_original

    def test_empty_input_returns_empty(self):
        """Empty arrays return empty arrays."""
        timestamps = np.array([])
        values = np.array([])

        t_down, v_down = lttb_downsample(timestamps, values, target_points=10)

        assert len(t_down) == 0
        assert len(v_down) == 0

    def test_single_point_returns_single_point(self):
        """Single data point returns as-is."""
        timestamps = np.array([5.0])
        values = np.array([42.0])

        t_down, v_down = lttb_downsample(timestamps, values, target_points=10)

        np.testing.assert_array_equal(t_down, timestamps)
        np.testing.assert_array_equal(v_down, values)

    def test_two_points_with_target_two(self):
        """Two points with target=2 returns both points."""
        timestamps = np.array([0.0, 1.0])
        values = np.array([10.0, 20.0])

        t_down, v_down = lttb_downsample(timestamps, values, target_points=2)

        np.testing.assert_array_equal(t_down, timestamps)
        np.testing.assert_array_equal(v_down, values)

    def test_target_two_returns_endpoints(self):
        """Target=2 on a longer series returns the first and last points."""
        timestamps = np.arange(0, 100, dtype=float)
        values = np.sin(timestamps)

        t_down, v_down = lttb_downsample(timestamps, values, target_points=2)

        np.testing.assert_array_equal(t_down, [0.0, 99.0])
        np.testing.assert_array_equal(v_down, [values[0], values[-1]])

    def test_mismatched_lengths_raises_error(self):
        """Timestamps and values with different lengths raise ValueError."""
        timestamps = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0])

        with pytest.raises(ValueError, match="must have same length"):
            lttb_downsample(timestamps, values, target_points=2)

    def test_target_less_than_two_raises_error(self):
        """target_points < 2 raises ValueError."""
        timestamps = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 15.0])

        with pytest.raises(ValueError, match="at least 2"):
            lttb_downsample(timestamps, values, target_points=1)

    def test_maintains_temporal_order(self):
        """Downsampled timestamps are non-decreasing (monotonic)."""
        timestamps = np.arange(0, 1000, dtype=float)
        values = np.random.randn(1000)

        t_down, _ = lttb_downsample(timestamps, values, target_points=50)

        assert np.all(np.diff(t_down) >= 0)

    def test_realistic_waveform_scenario(self):
        """8-hour flow waveform (720k points) downsampled to 2000 points."""
        sample_rate = 25.0
        duration_hours = 8.0
        duration_seconds = duration_hours * 3600
        num_samples = int(duration_seconds * sample_rate)

        timestamps = np.linspace(0, duration_seconds, num_samples)
        values = np.sin(timestamps / 10) + 0.1 * np.random.randn(num_samples)

        t_down, v_down = lttb_downsample(timestamps, values, target_points=2000)

        assert len(t_down) == 2000
        assert len(v_down) == 2000
        assert t_down[0] == timestamps[0]
        assert t_down[-1] == timestamps[-1]
