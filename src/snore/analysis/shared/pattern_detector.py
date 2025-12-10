"""
Complex breathing pattern detection algorithm.

This module implements detection of complex breathing patterns including
Cheyne-Stokes Respiration (CSR) and periodic breathing using time-series
analysis and clustering techniques.
"""

import logging

import numpy as np

from scipy import signal, stats

from snore.analysis.shared.types import (
    CSRDetection,
    PeriodicBreathingDetection,
)
from snore.constants import PatternDetectionConstants as PDC

logger = logging.getLogger(__name__)

__all__ = [
    "ComplexPatternDetector",
    "CSRDetection",
    "PeriodicBreathingDetection",
]


class ComplexPatternDetector:
    """
    Detects complex breathing patterns from time-series data.

    Uses autocorrelation, spectral analysis, and clustering to identify
    Cheyne-Stokes Respiration, periodic breathing, and positional events.

    Example:
        >>> detector = ComplexPatternDetector()
        >>> csr = detector.detect_csr(
        ...     timestamps, tidal_volumes, window_minutes=10
        ... )
        >>> if csr.confidence > 0.7:
        ...     print(f"CSR detected: {csr.cycle_length:.1f}s cycles")
    """

    def __init__(
        self,
        min_cycle_count: int = PDC.MIN_CYCLE_COUNT,
        autocorr_threshold: float = PDC.AUTOCORR_THRESHOLD,
    ):
        """
        Initialize the pattern detector.

        Args:
            min_cycle_count: Minimum number of cycles to confirm pattern
            autocorr_threshold: Minimum autocorrelation for periodic detection
        """
        self.min_cycle_count = min_cycle_count
        self.autocorr_threshold = autocorr_threshold
        logger.info("ComplexPatternDetector initialized")

    def detect_csr(
        self,
        timestamps: np.ndarray,
        tidal_volumes: np.ndarray,
        window_minutes: float = 10.0,
    ) -> CSRDetection | None:
        """
        Detect Cheyne-Stokes Respiration pattern.

        CSR is characterized by cyclical waxing and waning of tidal volume
        with central apneas at the nadir, typically 45-90 second cycles.

        Args:
            timestamps: Time values (seconds)
            tidal_volumes: Tidal volume measurements (mL)
            window_minutes: Analysis window size (minutes)

        Returns:
            CSRDetection if pattern found, None otherwise
        """
        min_cycle = PDC.CSR_MIN_CYCLE_LENGTH
        max_cycle = PDC.CSR_MAX_CYCLE_LENGTH

        smoothed_tv = self._smooth_signal(
            tidal_volumes, window_size=PDC.SIGNAL_SMOOTHING_WINDOW
        )

        autocorr = self._calculate_autocorrelation(smoothed_tv)

        cycle_length = self._find_dominant_cycle(
            autocorr, timestamps, min_cycle, max_cycle
        )

        if cycle_length is None:
            return None

        amplitude_var = np.std(smoothed_tv) / np.mean(smoothed_tv)

        waxing_waning_score = self._detect_waxing_waning(smoothed_tv, cycle_length)

        if waxing_waning_score < PDC.WAXING_WANING_MIN_SCORE:
            return None

        cycle_count = int((timestamps[-1] - timestamps[0]) / cycle_length)

        if cycle_count < self.min_cycle_count:
            return None

        csr_time = self._calculate_csr_time_percentage(smoothed_tv, cycle_length)

        confidence = self._calculate_csr_confidence(
            cycle_length, amplitude_var, waxing_waning_score, cycle_count
        )

        return CSRDetection(
            start_time=float(timestamps[0]),
            end_time=float(timestamps[-1]),
            cycle_length=cycle_length,
            amplitude_variation=float(amplitude_var),
            csr_index=float(csr_time),
            confidence=confidence,
            cycle_count=cycle_count,
        )

    def detect_periodic_breathing(
        self,
        timestamps: np.ndarray,
        tidal_volumes: np.ndarray,
        respiratory_rate: np.ndarray,
    ) -> PeriodicBreathingDetection | None:
        """
        Detect periodic breathing pattern.

        Similar to CSR but with broader criteria - any regular waxing/waning
        pattern with 30-120 second cycles.

        Args:
            timestamps: Time values (seconds)
            tidal_volumes: Tidal volume measurements (mL)
            respiratory_rate: Respiratory rate measurements (breaths/min)

        Returns:
            PeriodicBreathingDetection if pattern found, None otherwise
        """
        min_cycle = PDC.PERIODIC_MIN_CYCLE
        max_cycle = PDC.PERIODIC_MAX_CYCLE

        smoothed_tv = self._smooth_signal(
            tidal_volumes, window_size=PDC.SIGNAL_SMOOTHING_WINDOW
        )

        autocorr = self._calculate_autocorrelation(smoothed_tv)

        cycle_length = self._find_dominant_cycle(
            autocorr, timestamps, min_cycle, max_cycle
        )

        if cycle_length is None:
            return None

        regularity = self._calculate_regularity_score(smoothed_tv, cycle_length)

        if regularity < PDC.REGULARITY_MIN_SCORE:
            return None

        has_apneas = self._check_for_apneas(tidal_volumes)

        confidence = self._calculate_periodic_confidence(
            cycle_length, regularity, has_apneas
        )

        return PeriodicBreathingDetection(
            start_time=float(timestamps[0]),
            end_time=float(timestamps[-1]),
            cycle_length=cycle_length,
            regularity_score=float(regularity),
            confidence=confidence,
            has_apneas=has_apneas,
        )

    def _smooth_signal(
        self, signal_data: np.ndarray, window_size: int = 5
    ) -> np.ndarray:
        """Apply moving average smoothing to signal."""
        if len(signal_data) < window_size:
            return signal_data

        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(signal_data, kernel, mode="same")

        return smoothed

    def _calculate_autocorrelation(self, signal_data: np.ndarray) -> np.ndarray:
        """Calculate normalized autocorrelation of signal."""
        signal_normalized = signal_data - np.mean(signal_data)
        autocorr = np.correlate(signal_normalized, signal_normalized, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]

        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        return autocorr

    def _find_dominant_cycle(
        self,
        autocorr: np.ndarray,
        timestamps: np.ndarray,
        min_period: float,
        max_period: float,
    ) -> float | None:
        """Find dominant cycle length from autocorrelation."""
        if len(autocorr) < 10 or len(timestamps) < 2:
            return None

        sample_interval = (timestamps[-1] - timestamps[0]) / len(timestamps)

        min_lag = int(min_period / sample_interval)
        max_lag = min(int(max_period / sample_interval), len(autocorr) - 1)

        if min_lag >= max_lag or max_lag <= 0:
            return None

        search_region = autocorr[min_lag:max_lag]

        peaks, properties = signal.find_peaks(
            search_region, height=self.autocorr_threshold
        )

        if len(peaks) == 0:
            return None

        highest_peak_idx = peaks[np.argmax(properties["peak_heights"])]

        cycle_lag = min_lag + highest_peak_idx
        cycle_length = cycle_lag * sample_interval

        return float(cycle_length)

    def _detect_waxing_waning(
        self, signal_data: np.ndarray, cycle_length: float
    ) -> float:
        """Detect crescendo-decrescendo (waxing-waning) pattern."""
        if len(signal_data) < 10:
            return 0.0

        envelope_upper = self._extract_envelope(signal_data, upper=True)
        self._extract_envelope(signal_data, upper=False)

        envelope_variation = np.std(envelope_upper) / np.mean(envelope_upper)

        if envelope_variation < PDC.ENVELOPE_VARIATION_MIN:
            return 0.0

        gradients = np.gradient(envelope_upper)

        waxing_regions = gradients > 0

        alternation_score = np.mean(waxing_regions[:-1] != waxing_regions[1:])

        return float(min(1.0, alternation_score * 2))

    def _extract_envelope(
        self, signal_data: np.ndarray, upper: bool = True
    ) -> np.ndarray:
        """Extract upper or lower envelope of signal."""
        if upper:
            peaks, _ = signal.find_peaks(signal_data)
        else:
            peaks, _ = signal.find_peaks(-signal_data)

        if len(peaks) < 2:
            return signal_data

        envelope: np.ndarray = np.interp(
            np.arange(len(signal_data)), peaks, signal_data[peaks]
        )

        return envelope

    def _calculate_csr_time_percentage(
        self, signal_data: np.ndarray, cycle_length: float
    ) -> float:
        """Calculate percentage of time spent in CSR pattern."""
        envelope = self._extract_envelope(signal_data, upper=True)
        threshold = np.median(envelope) * PDC.CSR_THRESHOLD_FACTOR

        in_csr = envelope < threshold

        return float(np.mean(in_csr))

    def _calculate_regularity_score(
        self, signal_data: np.ndarray, cycle_length: float
    ) -> float:
        """Calculate regularity score using spectral concentration."""
        if len(signal_data) < 10:
            return 0.0

        freqs, psd = signal.periodogram(signal_data)

        if len(psd) == 0 or np.sum(psd) == 0:
            return 0.0

        spectral_entropy = stats.entropy(psd / np.sum(psd))

        max_entropy = np.log(len(psd))
        regularity = 1.0 - (spectral_entropy / max_entropy)

        return float(regularity)

    def _check_for_apneas(self, tidal_volumes: np.ndarray) -> bool:
        """Check if pattern includes near-zero tidal volumes (apneas)."""
        median_tv = np.median(tidal_volumes)
        low_tv_threshold = median_tv * PDC.LOW_TV_THRESHOLD_FACTOR

        low_tv_breaths = tidal_volumes < low_tv_threshold
        has_apneas = np.mean(low_tv_breaths) > PDC.APNEA_PRESENCE_THRESHOLD

        return bool(has_apneas)

    def _calculate_csr_confidence(
        self,
        cycle_length: float,
        amplitude_var: float,
        waxing_waning: float,
        cycle_count: int,
    ) -> float:
        """Calculate confidence score for CSR detection."""
        confidence = 0.5

        if PDC.CSR_MIN_CYCLE_LENGTH <= cycle_length <= PDC.CSR_MAX_CYCLE_LENGTH:
            confidence += 0.2

        if amplitude_var > PDC.CSR_MIN_AMPLITUDE_VAR:
            confidence += 0.1

        if waxing_waning > PDC.CSR_MIN_WAXING_WANING:
            confidence += 0.1

        if cycle_count >= PDC.CSR_MIN_CYCLES_HIGH_CONF:
            confidence += 0.1

        return min(1.0, confidence)

    def _calculate_periodic_confidence(
        self,
        cycle_length: float,
        regularity: float,
        has_apneas: bool,
    ) -> float:
        """Calculate confidence score for periodic breathing detection."""
        confidence = 0.5

        if PDC.PERIODIC_MIN_CYCLE <= cycle_length <= PDC.PERIODIC_MAX_CYCLE:
            confidence += 0.1

        if regularity > PDC.PERIODIC_HIGH_REGULARITY:
            confidence += 0.2

        if has_apneas:
            confidence += 0.2

        return min(1.0, confidence)
