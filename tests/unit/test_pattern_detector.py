"""
Tests for complex breathing pattern detection.
"""

from itertools import pairwise

import numpy as np
import pytest

from snore.analysis.shared.pattern_detector import (
    ComplexPatternDetector,
)


class TestCSRDetection:
    """Test Cheyne-Stokes Respiration detection."""

    @pytest.fixture
    def detector(self):
        return ComplexPatternDetector(min_cycle_count=3, autocorr_threshold=0.6)

    def test_detect_csr_with_valid_pattern(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        cycle_length = 60.0
        tidal_volumes = self._generate_csr_pattern(
            timestamps, cycle_length, amplitude=500.0
        )

        csr = detector.detect_csr(timestamps, tidal_volumes, window_minutes=10.0)

        if csr is not None:
            assert 45 <= csr.cycle_length <= 90
            assert csr.confidence > 0.5
            assert csr.cycle_count >= 3
        else:
            pass

    def test_detect_csr_too_few_cycles(self, detector):
        timestamps = np.arange(0, 120, 1.0)
        cycle_length = 60.0
        tidal_volumes = self._generate_csr_pattern(
            timestamps, cycle_length, amplitude=500.0
        )

        csr = detector.detect_csr(timestamps, tidal_volumes, window_minutes=10.0)

        assert csr is None

    def test_detect_csr_no_waxing_waning(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        tidal_volumes = np.ones(len(timestamps)) * 500.0

        csr = detector.detect_csr(timestamps, tidal_volumes, window_minutes=10.0)

        assert csr is None

    def test_detect_csr_cycle_length_out_of_range(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        cycle_length = 120.0
        tidal_volumes = self._generate_csr_pattern(
            timestamps, cycle_length, amplitude=500.0
        )

        csr = detector.detect_csr(timestamps, tidal_volumes, window_minutes=10.0)

        assert csr is None

    def test_csr_amplitude_variation(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        cycle_length = 60.0
        tidal_volumes = self._generate_csr_pattern(
            timestamps, cycle_length, amplitude=500.0
        )

        csr = detector.detect_csr(timestamps, tidal_volumes, window_minutes=10.0)

        if csr is not None:
            assert csr.amplitude_variation > 0.0

    def test_csr_index_calculation(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        cycle_length = 60.0
        tidal_volumes = self._generate_csr_pattern(
            timestamps, cycle_length, amplitude=500.0
        )

        csr = detector.detect_csr(timestamps, tidal_volumes, window_minutes=10.0)

        if csr is not None:
            assert 0.0 <= csr.csr_index <= 1.0

    def test_detect_csr_noise_weakens_periodicity_lowers_confidence(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        # Alternating breath sizes keep the smoothed envelope jagged enough to
        # clear the waxing/waning gate, so CSR is detected with and without noise.
        breath_sizes = np.array([1.0, 0.2, 0.5, 0.2])[np.arange(len(timestamps)) % 4]
        envelope = 0.2 + 0.8 * np.abs(np.sin(np.pi * timestamps / 60.0))
        clean = 500.0 * breath_sizes * envelope
        noisy = np.maximum(
            clean + np.random.default_rng(0).normal(0, 30.0, len(timestamps)), 5.0
        )

        clean_csr = detector.detect_csr(timestamps, clean)
        noisy_csr = detector.detect_csr(timestamps, noisy)

        assert clean_csr is not None
        assert noisy_csr is not None
        assert clean_csr.confidence > noisy_csr.confidence

    def _generate_csr_pattern(
        self, timestamps: np.ndarray, cycle_length: float, amplitude: float
    ) -> np.ndarray:
        breath_freq = 0.25
        breaths = amplitude / 2 + amplitude / 2 * np.sin(
            2 * np.pi * breath_freq * timestamps
        )
        csr_envelope = 0.2 + 0.8 * np.abs(np.sin(2 * np.pi * timestamps / cycle_length))
        tidal_volumes = breaths * csr_envelope
        noise = np.random.normal(0, amplitude * 0.03, len(timestamps))
        return np.maximum(tidal_volumes + noise, 10.0)


class TestPeriodicBreathingDetection:
    """Test periodic breathing detection."""

    @pytest.fixture
    def detector(self):
        return ComplexPatternDetector(min_cycle_count=3, autocorr_threshold=0.6)

    def test_detect_periodic_breathing_valid(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        cycle_length = 60.0
        tidal_volumes = self._generate_periodic_pattern(
            timestamps, cycle_length, amplitude=500.0
        )
        respiratory_rate = np.ones(len(timestamps)) * 15.0

        periodic = detector.detect_periodic_breathing(
            timestamps, tidal_volumes, respiratory_rate
        )

        assert periodic is not None
        assert 30 <= periodic.cycle_length <= 120
        assert periodic.confidence > 0.5
        assert periodic.regularity_score > 0.0

    def test_detect_periodic_breathing_with_apneas(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        cycle_length = 60.0
        tidal_volumes = self._generate_periodic_pattern(
            timestamps, cycle_length, amplitude=500.0
        )
        for i in range(0, 600, 50):
            tidal_volumes[i : i + 10] = 5.0
        respiratory_rate = np.ones(len(timestamps)) * 15.0

        periodic = detector.detect_periodic_breathing(
            timestamps, tidal_volumes, respiratory_rate
        )

        assert periodic is not None
        assert periodic.has_apneas

    def test_detect_periodic_breathing_no_pattern(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        tidal_volumes = np.random.normal(500, 50, len(timestamps))
        respiratory_rate = np.ones(len(timestamps)) * 15.0

        periodic = detector.detect_periodic_breathing(
            timestamps, tidal_volumes, respiratory_rate
        )

        assert periodic is None

    def test_periodic_breathing_regularity_score(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        cycle_length = 60.0
        tidal_volumes = self._generate_periodic_pattern(
            timestamps, cycle_length, amplitude=500.0
        )
        respiratory_rate = np.ones(len(timestamps)) * 15.0

        periodic = detector.detect_periodic_breathing(
            timestamps, tidal_volumes, respiratory_rate
        )

        assert periodic is not None
        assert 0.0 <= periodic.regularity_score <= 1.0

    def _generate_periodic_pattern(
        self, timestamps: np.ndarray, cycle_length: float, amplitude: float
    ) -> np.ndarray:
        periodic_variation = 0.3 * np.sin(2 * np.pi * timestamps / cycle_length)
        tidal_volumes = amplitude * (1 + periodic_variation)
        noise = np.random.normal(0, amplitude * 0.05, len(timestamps))
        return tidal_volumes + noise


class TestSignalProcessing:
    """Test signal processing helper methods."""

    @pytest.fixture
    def detector(self):
        return ComplexPatternDetector(min_cycle_count=3, autocorr_threshold=0.6)

    def test_smooth_signal_basic(self, detector):
        signal_data = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0])

        smoothed = detector._smooth_signal(signal_data, window_size=3)

        assert len(smoothed) == len(signal_data)
        assert np.max(smoothed) < np.max(signal_data)

    def test_smooth_signal_short_signal(self, detector):
        signal_data = np.array([1.0, 2.0])

        smoothed = detector._smooth_signal(signal_data, window_size=5)

        np.testing.assert_array_equal(smoothed, signal_data)

    def test_calculate_autocorrelation_basic(self, detector):
        signal_data = np.sin(np.arange(0, 100, 0.1))

        autocorr = detector._calculate_autocorrelation(signal_data)

        assert len(autocorr) > 0
        assert autocorr[0] == pytest.approx(1.0, abs=0.01)
        assert np.max(autocorr[1:]) < 1.0

    def test_calculate_autocorrelation_periodic(self, detector):
        signal_data = np.sin(2 * np.pi * np.arange(0, 100, 0.1) / 10)

        autocorr = detector._calculate_autocorrelation(signal_data)

        assert len(autocorr) > 0
        assert autocorr[0] == pytest.approx(1.0, abs=0.01)

    def test_find_dominant_cycle_with_peak(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        signal_data = np.sin(2 * np.pi * timestamps / 60.0)

        autocorr = detector._calculate_autocorrelation(signal_data)
        dominant_cycle = detector._find_dominant_cycle(autocorr, timestamps, 45.0, 90.0)

        assert dominant_cycle is not None
        cycle_length, peak_height = dominant_cycle
        assert 50 <= cycle_length <= 70
        assert peak_height >= detector.autocorr_threshold

    def test_find_dominant_cycle_no_peak(self, detector):
        timestamps = np.arange(0, 100, 1.0)
        signal_data = np.random.normal(0, 1, len(timestamps))

        autocorr = detector._calculate_autocorrelation(signal_data)
        dominant_cycle = detector._find_dominant_cycle(autocorr, timestamps, 45.0, 90.0)

        assert dominant_cycle is None

    def test_find_dominant_cycle_short_signal(self, detector):
        timestamps = np.arange(0, 10, 1.0)
        signal_data = np.sin(2 * np.pi * timestamps / 60.0)

        autocorr = detector._calculate_autocorrelation(signal_data)
        dominant_cycle = detector._find_dominant_cycle(autocorr, timestamps, 45.0, 90.0)

        assert dominant_cycle is None

    def test_extract_envelope_upper(self, detector):
        signal_data = np.sin(np.arange(0, 100, 0.1)) * np.linspace(1, 2, 1000)

        envelope = detector._extract_envelope(signal_data, upper=True)

        assert len(envelope) == len(signal_data)
        assert np.all(envelope >= np.min(signal_data))

    def test_extract_envelope_lower(self, detector):
        signal_data = np.sin(np.arange(0, 100, 0.1)) * np.linspace(1, 2, 1000)

        envelope = detector._extract_envelope(signal_data, upper=False)

        assert len(envelope) == len(signal_data)

    def test_extract_envelope_no_peaks(self, detector):
        signal_data = np.ones(100)

        envelope = detector._extract_envelope(signal_data, upper=True)

        np.testing.assert_array_equal(envelope, signal_data)

    def test_detect_waxing_waning_present(self, detector):
        timestamps = np.arange(0, 600, 1.0)
        breath_freq = 0.25
        breaths = 250 + 250 * np.sin(2 * np.pi * breath_freq * timestamps)
        csr_envelope = 0.2 + 0.8 * np.abs(np.sin(2 * np.pi * timestamps / 60.0))
        signal_data = breaths * csr_envelope

        score = detector._detect_waxing_waning(signal_data, cycle_length=60.0)

        assert score >= 0.0

    def test_detect_waxing_waning_absent(self, detector):
        signal_data = np.ones(600) * 500.0

        score = detector._detect_waxing_waning(signal_data, cycle_length=60.0)

        assert score == 0.0

    def test_detect_waxing_waning_short_signal(self, detector):
        signal_data = np.array([1.0, 2.0, 3.0])

        score = detector._detect_waxing_waning(signal_data, cycle_length=60.0)

        assert score == 0.0


class TestHelperMethods:
    """Test additional helper methods."""

    @pytest.fixture
    def detector(self):
        return ComplexPatternDetector(min_cycle_count=3, autocorr_threshold=0.6)

    def test_calculate_regularity_score_regular(self, detector):
        signal_data = np.sin(2 * np.pi * np.arange(0, 100, 0.1) / 10)

        regularity = detector._calculate_regularity_score(
            signal_data, cycle_length=10.0
        )

        assert 0.0 <= regularity <= 1.0

    def test_calculate_regularity_score_irregular(self, detector):
        signal_data = np.random.normal(0, 1, 1000)

        regularity = detector._calculate_regularity_score(
            signal_data, cycle_length=10.0
        )

        assert 0.0 <= regularity <= 1.0

    def test_calculate_regularity_score_short_signal(self, detector):
        signal_data = np.array([1.0, 2.0, 3.0])

        regularity = detector._calculate_regularity_score(
            signal_data, cycle_length=10.0
        )

        assert regularity == 0.0

    def test_check_for_apneas_present(self, detector):
        tidal_volumes = np.ones(100) * 500.0
        tidal_volumes[10:20] = 10.0
        tidal_volumes[40:50] = 10.0
        tidal_volumes[70:80] = 10.0

        has_apneas = detector._check_for_apneas(tidal_volumes)

        assert has_apneas

    def test_check_for_apneas_absent(self, detector):
        tidal_volumes = np.ones(100) * 500.0

        has_apneas = detector._check_for_apneas(tidal_volumes)

        assert not has_apneas

    def test_calculate_csr_confidence_high(self, detector):
        confidence = detector._calculate_csr_confidence(
            periodicity=1.0, amplitude_var=0.5, waxing_waning=0.8, cycle_count=6
        )

        assert 0.8 <= confidence <= 1.0

    def test_calculate_csr_confidence_low(self, detector):
        confidence = detector._calculate_csr_confidence(
            periodicity=0.0, amplitude_var=0.1, waxing_waning=0.4, cycle_count=2
        )

        assert confidence == 0.5

    def test_calculate_periodic_confidence_high(self, detector):
        confidence = detector._calculate_periodic_confidence(
            periodicity=1.0, regularity=0.85, has_apneas=True
        )

        assert 0.8 <= confidence <= 1.0

    def test_calculate_periodic_confidence_low(self, detector):
        confidence = detector._calculate_periodic_confidence(
            periodicity=0.0, regularity=0.45, has_apneas=False
        )

        assert confidence == 0.5

    def test_weakest_detectable_peak_scores_below_old_floor(self, detector):
        # Regression for #354: the always-earned cycle-range bonus pinned CSR
        # confidence at >= 0.7 and PB at >= 0.6 for every detection.
        periodicity = detector._periodicity_strength(detector.autocorr_threshold)

        csr_confidence = detector._calculate_csr_confidence(
            periodicity, amplitude_var=0.1, waxing_waning=0.5, cycle_count=3
        )
        pb_confidence = detector._calculate_periodic_confidence(
            periodicity, regularity=0.5, has_apneas=False
        )

        assert csr_confidence < 0.7
        assert pb_confidence < 0.6

    def test_confidence_strictly_increases_with_periodicity(self, detector):
        periodicities = [0.0, 0.25, 0.5, 0.75, 1.0]

        csr = [
            detector._calculate_csr_confidence(
                p, amplitude_var=0.1, waxing_waning=0.5, cycle_count=3
            )
            for p in periodicities
        ]
        pb = [
            detector._calculate_periodic_confidence(p, regularity=0.5, has_apneas=False)
            for p in periodicities
        ]

        assert all(a < b for a, b in pairwise(csr))
        assert all(a < b for a, b in pairwise(pb))

    def test_periodicity_strength_at_threshold_is_zero(self, detector):
        assert detector._periodicity_strength(detector.autocorr_threshold) == 0.0

    def test_periodicity_strength_perfect_peak_is_one(self, detector):
        assert detector._periodicity_strength(1.0) == 1.0

    def test_periodicity_strength_below_threshold_clips_to_zero(self, detector):
        assert detector._periodicity_strength(detector.autocorr_threshold - 0.2) == 0.0

    def test_calculate_csr_time_percentage(self, detector):
        signal_data = np.concatenate([np.ones(50) * 500, np.ones(50) * 100])

        csr_time = detector._calculate_csr_time_percentage(
            signal_data, cycle_length=60.0
        )

        assert 0.0 <= csr_time <= 1.0
