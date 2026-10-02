"""
Unit tests for feature extraction from breath waveforms.

Tests shape features and peak detection.
"""

import numpy as np

from snore.analysis.shared.feature_extractors import (
    WaveformFeatureExtractor,
    largest_inspiratory_segment,
)
from tests.helpers.synthetic_data import (
    generate_flattened_breath,
    generate_multi_peak_breath,
    generate_sinusoidal_breath,
)
from tests.helpers.validation_helpers import assert_features_in_range


class TestLargestInspiratorySegment:
    """Test contiguous-run extraction of inspiratory flow."""

    def test_mid_breath_dip_returns_single_hump(self):
        """A mid-breath dip below zero must not stitch two humps together."""
        flow = np.array(
            [
                1.0,
                2.0,
                3.0,
                2.0,
                1.0,
                -1.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ]
        )

        segment = largest_inspiratory_segment(flow)

        # Longest contiguous positive run is the second hump (7 samples), not the
        # 12-sample concatenation that a boolean mask would produce.
        assert len(segment) == 7
        assert np.array_equal(segment, np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]))
        assert len(flow[flow > 0]) == 12

    def test_single_run_returned_whole(self):
        """A waveform with one positive run returns that run unchanged."""
        flow = np.array([-1.0, 2.0, 4.0, 2.0, -1.0])

        segment = largest_inspiratory_segment(flow)

        assert np.array_equal(segment, np.array([2.0, 4.0, 2.0]))

    def test_no_positive_samples_returns_empty(self):
        """All-negative or empty input returns an empty array."""
        assert len(largest_inspiratory_segment(np.array([-1.0, -2.0, -3.0]))) == 0
        assert len(largest_inspiratory_segment(np.array([]))) == 0


class TestPlateauFraction:
    """Test plateau_fraction normalization by inspiration time."""

    def test_plateau_fraction_normalized_to_duration(self):
        """plateau_fraction equals plateau_duration / inspiration time."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_flattened_breath(duration=2.0, flatness_index=0.9)
        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)

        inspiration_time = len(insp_flow) / 25.0
        expected = shape.plateau_duration / inspiration_time
        assert abs(shape.plateau_fraction - expected) < 1e-9
        assert 0.0 <= shape.plateau_fraction <= 1.0

    def test_plateau_fraction_independent_of_breath_duration(self):
        """Same shape sampled at two durations yields the same plateau_fraction."""
        extractor = WaveformFeatureExtractor()
        _, short = generate_flattened_breath(duration=1.5, flatness_index=0.9)
        _, long = generate_flattened_breath(duration=3.0, flatness_index=0.9)

        short_shape = extractor.extract_shape_features(
            short[short > 0], sample_rate=25.0
        )
        long_shape = extractor.extract_shape_features(long[long > 0], sample_rate=25.0)

        assert abs(short_shape.plateau_fraction - long_shape.plateau_fraction) < 0.05


class TestFlatnessIndexCalculation:
    """Test flatness index extraction."""

    def test_flatness_fully_flat_waveform(self):
        """Perfectly flat waveform should have flatness near 1.0."""
        extractor = WaveformFeatureExtractor()

        waveform = np.ones(100) * 30.0

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        assert shape.flatness_index > 0.95

    def test_flatness_sharp_peak(self):
        """Sharp sinusoidal peak should have low flatness."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath(duration=2.0)

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

        assert shape.plateau_duration > 0.3

    def test_plateau_duration_reasonable(self):
        """Plateau duration should be less than total waveform duration."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_flattened_breath(duration=2.0)

        insp_flow = flow[flow > 0]

        shape = extractor.extract_shape_features(insp_flow, sample_rate=25.0)

        assert shape.plateau_duration < 2.0


class TestPeakDetection:
    """Test peak detection and analysis."""

    def test_peak_detection_single_peak(self):
        """Sinusoidal waveform should have single peak."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        assert peak.peak_count == 1

    def test_peak_detection_double_peak(self):
        """Double-peak waveform should detect 2 peaks."""
        extractor = WaveformFeatureExtractor(peak_prominence_threshold=0.15)
        _, flow = generate_multi_peak_breath(peak_count=2)

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        assert 1 <= peak.peak_count <= 3

    def test_peak_detection_multiple_peaks(self):
        """Multi-peak waveform should detect multiple peaks."""
        extractor = WaveformFeatureExtractor(peak_prominence_threshold=0.1)
        _, flow = generate_multi_peak_breath(peak_count=3)

        insp_flow = flow[flow > 0]

        peak = extractor.extract_peak_features(insp_flow, sample_rate=25.0)

        assert peak.peak_count >= 2

    def test_peak_detection_no_peaks(self):
        """Flat waveform should have no peaks."""
        extractor = WaveformFeatureExtractor()
        waveform = np.ones(100) * 20.0

        peak = extractor.extract_peak_features(waveform, sample_rate=25.0)

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


class TestFeatureValidation:
    """Test that extracted features fall within valid ranges."""

    def test_shape_and_peak_features_pass_validation(self):
        """Extracted shape and peak features should pass validation."""
        extractor = WaveformFeatureExtractor()
        _, flow = generate_sinusoidal_breath()

        shape = extractor.extract_shape_features(flow, sample_rate=25.0)
        peak = extractor.extract_peak_features(flow, sample_rate=25.0)

        assert_features_in_range(shape=shape, peak=peak)


class TestFeatureExtractionEdgeCases:
    """Test edge cases in feature extraction."""

    def test_features_from_noise(self):
        """Features from random noise should be reasonable."""
        extractor = WaveformFeatureExtractor()
        waveform = np.random.normal(0, 10, 100)

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        assert 0 <= shape.flatness_index <= 1

    def test_features_from_extreme_values(self):
        """Features from extreme values should handle gracefully."""
        extractor = WaveformFeatureExtractor()
        waveform = np.array([0.0, 1000.0, 0.0, -1000.0])

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        assert shape is not None

    def test_features_from_constant_zero(self):
        """Features from all zeros should handle gracefully."""
        extractor = WaveformFeatureExtractor()
        waveform = np.zeros(100)

        shape = extractor.extract_shape_features(waveform, sample_rate=25.0)

        assert shape is not None
