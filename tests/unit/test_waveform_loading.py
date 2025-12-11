"""
Unit tests for waveform data loading and deserialization.

Tests the waveform_loader module's ability to:
- Deserialize binary waveform data
- Load waveforms from database
- Handle errors and edge cases
"""

from unittest.mock import Mock

import numpy as np
import pytest

from snore.analysis.data.waveform_loader import (
    apply_noise_filter,
    deserialize_waveform_blob,
    detect_and_mark_artifacts,
    handle_discontinuities,
    handle_sample_rate_conversion,
    load_waveform_from_db,
)


class TestWaveformDeserialization:
    """Test waveform blob deserialization."""

    def test_deserialize_valid_blob(self):
        """Valid blob should deserialize correctly."""
        timestamps = np.array([0.0, 0.04, 0.08, 0.12], dtype=np.float32)
        values = np.array([10.0, 20.0, 15.0, 5.0], dtype=np.float32)
        data = np.column_stack([timestamps, values])
        blob = data.tobytes()

        result_t, result_v = deserialize_waveform_blob(blob, sample_count=4)

        np.testing.assert_array_almost_equal(result_t, timestamps)
        np.testing.assert_array_almost_equal(result_v, values)

    def test_deserialize_empty_blob(self):
        """Empty blob should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid waveform blob"):
            deserialize_waveform_blob(b"", sample_count=0)

    def test_deserialize_size_mismatch(self):
        """Blob with wrong size should raise ValueError."""
        data = np.zeros((4, 2), dtype=np.float32)
        blob = data.tobytes()

        with pytest.raises(ValueError, match="Blob size mismatch"):
            deserialize_waveform_blob(blob, sample_count=5)

    def test_deserialize_corrupted_blob(self):
        """Corrupted blob should raise ValueError."""
        corrupted_blob = b"invalid_data_here"

        with pytest.raises(ValueError, match="Invalid waveform blob"):
            deserialize_waveform_blob(corrupted_blob, sample_count=2)

    def test_deserialize_preserves_precision(self):
        """Deserialization should preserve float32 precision."""
        timestamps = np.array([0.0, 1.234567, 2.345678], dtype=np.float32)
        values = np.array([12.34, -56.78, 90.12], dtype=np.float32)
        data = np.column_stack([timestamps, values])
        blob = data.tobytes()

        result_t, result_v = deserialize_waveform_blob(blob, sample_count=3)

        np.testing.assert_array_equal(result_t, timestamps)
        np.testing.assert_array_equal(result_v, values)


class TestWaveformLoading:
    """Test loading waveforms from database."""

    def test_load_missing_waveform(self):
        """Loading non-existent waveform should raise ValueError."""
        mock_session = Mock()
        mock_session.query().filter_by().first.return_value = None

        with pytest.raises(ValueError, match="Waveform not found"):
            load_waveform_from_db(mock_session, session_id=999, waveform_type="flow")

    def test_load_waveform_returns_metadata(self):
        """Should return timestamps, values, and complete metadata."""
        mock_waveform = Mock()
        timestamps = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        values = np.array([10.0, 20.0, 15.0], dtype=np.float32)
        data = np.column_stack([timestamps, values])

        mock_waveform.data_blob = data.tobytes()
        mock_waveform.sample_count = 3
        mock_waveform.id = 1
        mock_waveform.session_id = 123
        mock_waveform.waveform_type = "flow"
        mock_waveform.sample_rate = 25.0
        mock_waveform.unit = "L/min"
        mock_waveform.min_value = 10.0
        mock_waveform.max_value = 20.0
        mock_waveform.mean_value = 15.0

        mock_session = Mock()
        mock_session.query().filter_by().first.return_value = mock_waveform

        t, v, metadata = load_waveform_from_db(mock_session, 123, "flow")

        assert len(t) == 3
        assert len(v) == 3

        assert metadata["session_id"] == 123
        assert metadata["waveform_type"] == "flow"
        assert metadata["sample_rate"] == 25.0
        assert metadata["unit"] == "L/min"
        assert metadata["sample_count"] == 3


class TestNoiseFiltering:
    """Test Butterworth noise filtering."""

    def test_filter_removes_high_frequency(self):
        """Butterworth filter should attenuate high frequencies."""
        sample_rate = 100.0
        t = np.linspace(0, 1, 100)

        signal = np.sin(2 * np.pi * 1 * t)
        noise = 0.5 * np.sin(2 * np.pi * 20 * t)
        noisy = signal + noise

        filtered = apply_noise_filter(noisy, sample_rate, cutoff_hz=5.0)

        noise_power_before = np.mean((noisy - signal) ** 2)
        noise_power_after = np.mean((filtered - signal) ** 2)

        assert noise_power_after < noise_power_before

    def test_filter_invalid_cutoff_raises_error(self):
        """Cutoff above Nyquist frequency should raise ValueError."""
        data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        sample_rate = 10.0

        with pytest.raises(ValueError, match="Nyquist"):
            apply_noise_filter(data, sample_rate, cutoff_hz=6.0)


class TestSampleRateConversion:
    """Test sample rate conversion (resampling)."""

    def test_resample_upsampling(self):
        """Upsampling should increase sample count."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0])
        values = np.array([1.0, 2.0, 3.0, 2.0])

        new_t, new_v = handle_sample_rate_conversion(
            timestamps, values, from_rate=1.0, to_rate=2.0
        )

        assert len(new_v) > len(values)
        assert len(new_t) == len(new_v)

    def test_resample_downsampling(self):
        """Downsampling should decrease sample count."""
        timestamps = np.linspace(0, 1, 100)
        values = np.sin(2 * np.pi * timestamps)

        new_t, new_v = handle_sample_rate_conversion(
            timestamps, values, from_rate=100.0, to_rate=10.0
        )

        assert len(new_v) < len(values)
        assert len(new_v) >= 10

    def test_resample_no_change(self):
        """Same sample rate should return original arrays."""
        timestamps = np.array([0.0, 1.0, 2.0])
        values = np.array([1.0, 2.0, 1.0])

        new_t, new_v = handle_sample_rate_conversion(
            timestamps, values, from_rate=1.0, to_rate=1.0
        )

        np.testing.assert_array_equal(new_t, timestamps)
        np.testing.assert_array_equal(new_v, values)


class TestArtifactDetection:
    """Test artifact detection in waveforms."""

    def test_detect_out_of_range_values(self):
        """Should detect values outside physiological range."""
        flow_data = np.array([10.0, 20.0, 150.0, 15.0, -200.0])

        artifacts = detect_and_mark_artifacts(flow_data, "flow")

        assert artifacts[2]  # 150 L/min too high
        assert artifacts[4]  # -200 L/min too low
        assert not artifacts[0]  # Normal values
        assert not artifacts[1]

    def test_detect_nan_values(self):
        """Should detect NaN values."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        artifacts = detect_and_mark_artifacts(data, "flow")

        assert artifacts[2]
        assert not artifacts[0]

    def test_detect_inf_values(self):
        """Should detect Inf values."""
        data = np.array([1.0, np.inf, 3.0, -np.inf, 5.0])

        artifacts = detect_and_mark_artifacts(data, "flow")

        assert artifacts[1]
        assert artifacts[3]

    def test_detect_sudden_jumps(self):
        """Should detect sudden large jumps (sensor disconnection)."""
        data = np.array(
            [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
                15.0,
                14.0,
                13.0,
                12.0,
                11.0,
                10.0,
                100.0,
                11.0,
            ]
        )

        artifacts = detect_and_mark_artifacts(data, "flow")

        assert artifacts[11] or artifacts[10]

    def test_no_artifacts_in_clean_data(self):
        """Clean data should have no artifacts."""
        data = np.array([10.0, 15.0, 20.0, 15.0, 10.0])

        artifacts = detect_and_mark_artifacts(data, "flow")

        assert not np.any(artifacts)


class TestDiscontinuityHandling:
    """Test handling of discontinuities (mask-off events)."""

    def test_no_discontinuities(self):
        """Continuous data should return single segment."""
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        values = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        segments = handle_discontinuities(timestamps, values, gap_threshold=10.0)

        assert len(segments) == 1
        np.testing.assert_array_equal(segments[0][0], timestamps)
        np.testing.assert_array_equal(segments[0][1], values)

    def test_single_discontinuity(self):
        """Single large gap should create two segments."""
        timestamps = np.array([0.0, 1.0, 2.0, 102.0, 103.0])
        values = np.array([1.0, 2.0, 3.0, 2.0, 1.0])

        segments = handle_discontinuities(timestamps, values, gap_threshold=50.0)

        assert len(segments) == 2
        assert len(segments[0][0]) == 3
        assert len(segments[1][0]) == 2

    def test_multiple_discontinuities(self):
        """Multiple gaps should create multiple segments."""
        timestamps = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 200.0, 201.0])
        values = np.ones(8)

        segments = handle_discontinuities(timestamps, values, gap_threshold=50.0)

        assert len(segments) == 3
        assert len(segments[0][0]) == 3
        assert len(segments[1][0]) == 3
        assert len(segments[2][0]) == 2

    def test_empty_array(self):
        """Empty arrays should return empty list."""
        segments = handle_discontinuities(np.array([]), np.array([]))

        assert len(segments) == 0

    def test_single_sample(self):
        """Single sample should return one segment."""
        segments = handle_discontinuities(np.array([0.0]), np.array([1.0]))

        assert len(segments) == 1
        assert len(segments[0][0]) == 1
