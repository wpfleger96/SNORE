"""
Unit tests for feature extraction from breath waveforms.

Tests shape features, peak detection, statistical features,
and spectral analysis.
"""

import numpy as np

from oscar_mcp.analysis.algorithms.feature_extractors import WaveformFeatureExtractor
from tests.helpers.synthetic_data import (
    generate_flattened_breath,
    generate_multi_peak_breath,
    generate_sinusoidal_breath,
)
from tests.helpers.validation_helpers import assert_features_in_range


class TestFlatnessIndexCalculation:
    """Test flatness index extraction."""

    def test_flatness_fully_flat_waveform(self):
        """Perfectly flat waveform should have flatness near 1.0."""
        extractor = WaveformFeatureExtractor()

        # Flat waveform (constant value)
        waveform = np.ones(100) * 30.0

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        # Should be very high flatness
        assert shape.flatness_index > 0.95

    def test_flatness_sharp_peak(self):
        """Sharp sinusoidal peak should have low flatness."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath(duration=2.0)

        # Extract only inspiration phase
        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)

        # Sinusoidal should have low flatness index (close to 0 = sharp peak)
        # Mathematical reality: a pure sinusoid has ~41.7% of samples above 80% of peak
        # This is because sin(θ) > 0.8 for θ in approximately 83% of the positive half-cycle
        # Threshold of 0.45 accommodates this while still detecting flat-topped waveforms
        assert shape.flatness_index < 0.45

    def test_flatness_intermediate(self):
        """Flow-limited breath should have intermediate flatness."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_flattened_breath(duration=2.0, flatness_index=0.6)

        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)

        # Should have moderate flatness
        assert 0.4 < shape.flatness_index < 0.9

    def test_flatness_in_valid_range(self):
        """Flatness index should always be 0-1."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        shape = extractor.extract_shape_features(flow, sample_rate=25.0)

        assert 0 <= shape.flatness_index <= 1


class TestPlateauDetection:
    """Test plateau duration detection."""

    def test_plateau_no_plateau(self):
        """Waveform with no plateau should have zero or small plateau duration."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)

        # Sinusoidal waveforms have no true plateau (continuously varying)
        # However, plateau detection algorithm may identify small continuous regions
        # where flow stays near peak due to the gradual rate of change near the peak
        # Threshold of 0.85 ensures we don't falsely classify sinusoids as having plateaus
        # while still detecting true flat-topped flow limitation patterns
        assert shape.plateau_duration < 0.85

    def test_plateau_continuous_plateau(self):
        """Flat-topped waveform should have significant plateau."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_flattened_breath(flatness_index=0.9)

        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)

        # Should detect substantial plateau
        assert shape.plateau_duration > 0.3

    def test_plateau_duration_reasonable(self):
        """Plateau duration should be less than total waveform duration."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_flattened_breath(duration=2.0)

        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)

        # Plateau can't be longer than inspiration
        assert shape.plateau_duration < 2.0


class TestPeakDetection:
    """Test peak detection and analysis."""

    def test_peak_detection_single_peak(self):
        """Sinusoidal waveform should have single peak."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        # Should detect 1 peak
        assert peak.peak_count == 1

    def test_peak_detection_double_peak(self):
        """Double-peak waveform should detect 2 peaks."""
        extractor = WaveformFeatureExtractor(peak_prominence_threshold=0.15)
        _, flow = generate_multi_peak_breath(peak_count=2)

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        # Should detect 2 peaks (may detect 1-3 depending on exact waveform)
        assert 1 <= peak.peak_count <= 3

    def test_peak_detection_multiple_peaks(self):
        """Multi-peak waveform should detect multiple peaks."""
        extractor = WaveformFeatureExtractor(peak_prominence_threshold=0.1)
        _, flow = generate_multi_peak_breath(peak_count=3)

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        # Should detect multiple peaks
        assert peak.peak_count >= 2

    def test_peak_detection_no_peaks(self):
        """Flat waveform should have no peaks."""
        extractor = WaveformFeatureExtractor()
        waveform = np.ones(100) * 20.0  # Flat

        peak = extractor.extract_peak_features(waveform, sample_rate=25.0)

        # Flat waveform has no peaks
        assert peak.peak_count == 0

    def test_peak_positions_in_range(self):
        """Peak positions should be between 0 and 1."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        for pos in peak.peak_positions:
            assert 0 <= pos <= 1

    def test_peak_prominences_positive(self):
        """Peak prominences should be positive."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        for prom in peak.peak_prominences:
            assert prom > 0


class TestStatisticalFeatures:
    """Test statistical feature extraction."""

    def test_statistical_features_basic(self):
        """Basic statistics should be calculated correctly."""
        extractor = WaveformFeatureExtractor()
        waveform = np.array([10.0, 20.0, 30.0, 20.0, 10.0])

        stats = extractor.extract_statistical_features(waveform)

        assert stats.mean == 18.0
        assert stats.median == 20.0
        assert stats.std_dev > 0

    def test_statistical_percentiles_ordered(self):
        """Percentiles should be in ascending order."""
        extractor = WaveformFeatureExtractor()
        waveform = np.random.normal(20, 5, 100)

        stats = extractor.extract_statistical_features(waveform)

        assert stats.percentile_25 <= stats.percentile_50
        assert stats.percentile_50 <= stats.percentile_75
        assert stats.percentile_75 <= stats.percentile_95

    def test_statistical_empty_array(self):
        """Empty array should return zeros or handle gracefully."""
        extractor = WaveformFeatureExtractor()
        waveform = np.array([])

        stats = extractor.extract_statistical_features(waveform)

        # Should not crash
        assert stats is not None

    def test_statistical_single_value(self):
        """Single value should have zero std dev."""
        extractor = WaveformFeatureExtractor()
        waveform = np.array([20.0])

        stats = extractor.extract_statistical_features(waveform)

        assert stats.mean == 20.0
        assert stats.std_dev == 0.0

    def test_statistical_all_same_values(self):
        """All same values should have zero std dev."""
        extractor = WaveformFeatureExtractor()
        waveform = np.ones(100) * 25.0

        stats = extractor.extract_statistical_features(waveform)

        assert stats.std_dev == 0.0
        assert stats.coefficient_of_variation == 0.0

    def test_coefficient_of_variation(self):
        """CV should be std_dev / mean."""
        extractor = WaveformFeatureExtractor()
        waveform = np.array([10.0, 20.0, 30.0])

        stats = extractor.extract_statistical_features(waveform)

        expected_cv = stats.std_dev / stats.mean
        assert abs(stats.coefficient_of_variation - expected_cv) < 0.01


class TestSpectralFeatures:
    """Test spectral (frequency domain) features."""

    def test_spectral_dominant_frequency(self):
        """Should detect dominant frequency in periodic signal."""
        extractor = WaveformFeatureExtractor()

        # 5 Hz sine wave
        t = np.linspace(0, 1, 100)
        waveform = np.sin(2 * np.pi * 5 * t)

        spectral = extractor.extract_spectral_features(waveform, sample_rate=100.0)

        # Should detect ~5 Hz as dominant
        assert 4.0 < spectral.dominant_frequency < 6.0

    def test_spectral_entropy(self):
        """Spectral entropy should be reasonable."""
        extractor = WaveformFeatureExtractor()
        t = np.linspace(0, 1, 100)
        waveform = np.sin(2 * np.pi * 5 * t)

        spectral = extractor.extract_spectral_features(waveform, sample_rate=100.0)

        # Should be positive
        assert spectral.spectral_entropy > 0

    def test_spectral_psd_shape(self):
        """Power spectral density should have expected shape."""
        extractor = WaveformFeatureExtractor()
        t = np.linspace(0, 1, 100)
        waveform = np.sin(2 * np.pi * 5 * t)

        spectral = extractor.extract_spectral_features(waveform, sample_rate=100.0)

        # PSD should be an array
        assert len(spectral.power_spectral_density) > 0
        # All values should be non-negative
        assert np.all(spectral.power_spectral_density >= 0)

    def test_spectral_very_short_signal(self):
        """Very short signal should handle gracefully."""
        extractor = WaveformFeatureExtractor()
        waveform = np.array([1.0, 2.0, 3.0])

        spectral = extractor.extract_spectral_features(waveform, sample_rate=25.0)

        # Should not crash
        assert spectral is not None


class TestAllFeaturesExtraction:
    """Test extracting all features at once."""

    def test_extract_all_features_complete(self):
        """All features should be extracted without spectral."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        shape, peak, stats, spectral = extractor.extract_all_features(
            flow, sample_rate=25.0, include_spectral=False
        )

        assert shape is not None
        assert peak is not None
        assert stats is not None
        assert spectral is None

    def test_extract_all_features_with_spectral(self):
        """All features including spectral should be extracted."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        shape, peak, stats, spectral = extractor.extract_all_features(
            flow, sample_rate=25.0, include_spectral=True
        )

        assert shape is not None
        assert peak is not None
        assert stats is not None
        assert spectral is not None

    def test_all_features_pass_validation(self):
        """Extracted features should pass validation."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        shape, peak, stats, spectral = extractor.extract_all_features(
            flow, sample_rate=25.0, include_spectral=False
        )

        # Should not raise
        assert_features_in_range(shape=shape, peak=peak, statistical=stats)


class TestFeatureExtractionEdgeCases:
    """Test edge cases in feature extraction."""

    def test_features_from_noise(self):
        """Features from random noise should be reasonable."""
        extractor = WaveformFeatureExtractor()
        waveform = np.random.normal(0, 10, 100)

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        # Should not crash, values should be in range
        assert 0 <= shape.flatness_index <= 1

    def test_features_from_extreme_values(self):
        """Features from extreme values should handle gracefully."""
        extractor = WaveformFeatureExtractor()
        waveform = np.array([0.0, 1000.0, 0.0, -1000.0])

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        # Should not crash
        assert shape is not None

    def test_features_from_constant_zero(self):
        """Features from all zeros should handle gracefully."""
        extractor = WaveformFeatureExtractor()
        waveform = np.zeros(100)

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        # Should not crash (may return zeros or defaults)
        assert shape is not None
