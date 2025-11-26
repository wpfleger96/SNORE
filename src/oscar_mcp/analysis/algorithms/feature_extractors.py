"""
Feature extraction algorithms for waveform analysis.

This module provides comprehensive feature extraction from breath waveforms,
including shape characteristics, peak analysis, statistical features, and
optional spectral features. These features are used for flow limitation
classification and respiratory pattern analysis.
"""

import logging

from dataclasses import dataclass

import numpy as np

from scipy import signal, stats

logger = logging.getLogger(__name__)


@dataclass
class ShapeFeatures:
    """
    Shape characteristics of a breath waveform.

    These features describe the overall shape of the inspiratory flow curve
    and are critical for flow limitation classification.

    Attributes:
        flatness_index: Ratio of time spent >80% of peak (0-1)
            High values indicate plateau/flattened waveforms
        plateau_duration: Duration of plateau phase in seconds
        symmetry_score: Statistical skewness (-1 to 1)
            0 = symmetric, + = right-skewed, - = left-skewed
        kurtosis: Measure of peakedness vs flatness
            High = sharp peak, Low = flat plateau
        rise_time: Time from 10% to 90% of peak flow (seconds)
        fall_time: Time from 90% to 10% of peak flow (seconds)
    """

    flatness_index: float
    plateau_duration: float
    symmetry_score: float
    kurtosis: float
    rise_time: float
    fall_time: float


@dataclass
class PeakFeatures:
    """
    Peak analysis features for breath waveform.

    Multiple peaks in the inspiratory flow curve indicate specific flow
    limitation patterns (e.g., Class 2 double peak, Class 3 multiple peaks).

    Attributes:
        peak_count: Number of significant peaks detected
        peak_positions: Relative positions of peaks (0-1 scale)
            0 = start of inspiration, 1 = end
        peak_prominences: Height of each peak above surroundings
        inter_peak_intervals: Time spacing between consecutive peaks (seconds)
    """

    peak_count: int
    peak_positions: list[float]
    peak_prominences: list[float]
    inter_peak_intervals: list[float]


@dataclass
class StatisticalFeatures:
    """
    Statistical features of breath waveform.

    Basic statistical measures that help characterize the distribution
    and variability of flow values.

    Attributes:
        mean: Mean flow value
        median: Median flow value
        std_dev: Standard deviation
        percentile_25: 25th percentile
        percentile_50: 50th percentile (median)
        percentile_75: 75th percentile
        percentile_95: 95th percentile
        coefficient_of_variation: std_dev / mean
        zero_crossing_rate: Frequency of sign changes
    """

    mean: float
    median: float
    std_dev: float
    percentile_25: float
    percentile_50: float
    percentile_75: float
    percentile_95: float
    coefficient_of_variation: float
    zero_crossing_rate: float


@dataclass
class SpectralFeatures:
    """
    Spectral (frequency domain) features of breath waveform.

    Optional features that analyze frequency content using FFT.
    Useful for detecting periodic patterns.

    Attributes:
        dominant_frequency: Primary frequency component (Hz)
        spectral_entropy: Measure of spectral regularity
        power_spectral_density: Power distribution across frequencies
    """

    dominant_frequency: float
    spectral_entropy: float
    power_spectral_density: np.ndarray


class WaveformFeatureExtractor:
    """
    Extracts comprehensive features from breath waveforms.

    Provides methods to extract shape, peak, statistical, and spectral
    features from individual breath segments. These features are used
    for flow limitation classification and pattern detection.

    Example:
        >>> extractor = WaveformFeatureExtractor()
        >>> shape = extractor.extract_shape_features(
        ...     inspiration_flow, sample_rate=25.0
        ... )
        >>> print(f"Flatness index: {shape.flatness_index:.3f}")
    """

    def __init__(
        self,
        flatness_threshold: float = 0.8,
        peak_prominence_threshold: float = 0.2,
    ):
        """
        Initialize feature extractor with configuration.

        Args:
            flatness_threshold: Percentage of peak flow for flatness calculation
            peak_prominence_threshold: Minimum peak prominence (fraction of max)
        """
        self.flatness_threshold = flatness_threshold
        self.peak_prominence_threshold = peak_prominence_threshold

    def extract_all_features(
        self,
        waveform: np.ndarray,
        sample_rate: float,
        include_spectral: bool = False,
    ) -> tuple[
        ShapeFeatures, PeakFeatures, StatisticalFeatures, SpectralFeatures | None
    ]:
        """
        Extract all features from a waveform.

        Convenience method that extracts all feature types in one call.

        Args:
            waveform: 1D array of flow values (typically inspiration only)
            sample_rate: Sample rate in Hz
            include_spectral: Whether to compute spectral features

        Returns:
            Tuple of (shape, peak, statistical, spectral) features
            spectral will be None if include_spectral=False

        Example:
            >>> shape, peak, stats, spectral = extractor.extract_all_features(
            ...     flow, 25.0, include_spectral=True
            ... )
        """
        shape = self.extract_shape_features(waveform, sample_rate)
        peak = self.extract_peak_features(waveform, sample_rate)
        statistical = self.extract_statistical_features(waveform)
        spectral = None
        if include_spectral:
            spectral = self.extract_spectral_features(waveform, sample_rate)

        return shape, peak, statistical, spectral

    def extract_shape_features(
        self, waveform: np.ndarray, sample_rate: float
    ) -> ShapeFeatures:
        """
        Extract shape characteristics from waveform.

        Args:
            waveform: 1D array of flow values
            sample_rate: Sample rate in Hz

        Returns:
            ShapeFeatures object with all shape metrics

        Example:
            >>> shape = extractor.extract_shape_features(flow, 25.0)
            >>> if shape.flatness_index > 0.7:
            ...     print("Flattened waveform detected")
        """
        if len(waveform) == 0:
            return ShapeFeatures(0, 0, 0, 0, 0, 0)

        peak_flow = np.max(waveform)
        if peak_flow <= 0:
            return ShapeFeatures(0, 0, 0, 0, 0, 0)

        # Flatness index: ratio of time spent >80% of peak
        flatness_threshold_value = self.flatness_threshold * peak_flow
        above_threshold = waveform > flatness_threshold_value
        flatness_index = np.sum(above_threshold) / len(waveform)

        # Plateau duration: continuous time at high flow
        plateau_duration = self._calculate_plateau_duration(
            waveform, flatness_threshold_value, sample_rate
        )

        # Symmetry score: statistical skewness
        # Normalize to [-1, 1] range for easier interpretation
        # Check for constant or near-constant data to avoid scipy warnings
        if np.std(waveform) < 1e-10:
            symmetry_score = 0.0
            kurtosis_value = 0.0
        else:
            raw_skewness = stats.skew(waveform)
            symmetry_score = float(np.clip(raw_skewness / 3.0, -1.0, 1.0))
            kurtosis_value = float(stats.kurtosis(waveform))

        # Rise and fall times
        rise_time = self._calculate_rise_time(waveform, peak_flow, sample_rate)
        fall_time = self._calculate_fall_time(waveform, peak_flow, sample_rate)

        return ShapeFeatures(
            flatness_index=flatness_index,
            plateau_duration=plateau_duration,
            symmetry_score=symmetry_score,
            kurtosis=kurtosis_value,
            rise_time=rise_time,
            fall_time=fall_time,
        )

    def _calculate_plateau_duration(
        self, waveform: np.ndarray, threshold: float, sample_rate: float
    ) -> float:
        """
        Calculate longest continuous plateau duration.

        Args:
            waveform: Flow values
            threshold: Minimum value to consider plateau
            sample_rate: Sample rate in Hz

        Returns:
            Plateau duration in seconds
        """
        above_threshold = waveform > threshold

        # Find continuous runs above threshold
        changes = np.diff(np.concatenate([[0], above_threshold, [0]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        if len(starts) == 0:
            return 0.0

        # Find longest run
        run_lengths = ends - starts
        max_length = np.max(run_lengths)

        return float(max_length / sample_rate)

    def _calculate_rise_time(
        self, waveform: np.ndarray, peak_flow: float, sample_rate: float
    ) -> float:
        """
        Calculate rise time (10% to 90% of peak).

        Args:
            waveform: Flow values
            peak_flow: Peak flow value
            sample_rate: Sample rate in Hz

        Returns:
            Rise time in seconds
        """
        threshold_10 = 0.1 * peak_flow
        threshold_90 = 0.9 * peak_flow

        # Find first crossing of 10% threshold
        above_10 = np.where(waveform >= threshold_10)[0]
        if len(above_10) == 0:
            return 0.0
        idx_10 = above_10[0]

        # Find first crossing of 90% threshold after 10%
        above_90 = np.where(waveform[idx_10:] >= threshold_90)[0]
        if len(above_90) == 0:
            return 0.0
        idx_90 = idx_10 + above_90[0]

        return float((idx_90 - idx_10) / sample_rate)

    def _calculate_fall_time(
        self, waveform: np.ndarray, peak_flow: float, sample_rate: float
    ) -> float:
        """
        Calculate fall time (90% to 10% of peak).

        Args:
            waveform: Flow values
            peak_flow: Peak flow value
            sample_rate: Sample rate in Hz

        Returns:
            Fall time in seconds
        """
        threshold_90 = 0.9 * peak_flow
        threshold_10 = 0.1 * peak_flow

        # Find peak index
        peak_idx = np.argmax(waveform)

        # Find last crossing of 90% threshold after peak
        after_peak = waveform[peak_idx:]
        above_90 = np.where(after_peak >= threshold_90)[0]
        if len(above_90) == 0:
            return 0.0
        idx_90 = peak_idx + above_90[-1]

        # Find first crossing below 10% threshold after 90%
        after_90 = waveform[idx_90:]
        below_10 = np.where(after_90 < threshold_10)[0]
        if len(below_10) == 0:
            return 0.0
        idx_10 = idx_90 + below_10[0]

        return float((idx_10 - idx_90) / sample_rate)

    def extract_peak_features(
        self, waveform: np.ndarray, sample_rate: float
    ) -> PeakFeatures:
        """
        Extract peak analysis features from waveform.

        Uses scipy.signal.find_peaks to detect significant peaks and
        analyze their characteristics.

        Args:
            waveform: 1D array of flow values
            sample_rate: Sample rate in Hz

        Returns:
            PeakFeatures object with peak analysis

        Example:
            >>> peak = extractor.extract_peak_features(flow, 25.0)
            >>> if peak.peak_count == 2:
            ...     print("Double peak pattern detected (Class 2)")
        """
        if len(waveform) == 0:
            return PeakFeatures(0, [], [], [])

        peak_flow = np.max(waveform)
        if peak_flow <= 0:
            return PeakFeatures(0, [], [], [])

        # Find peaks with prominence threshold
        min_prominence = self.peak_prominence_threshold * peak_flow
        peaks, properties = signal.find_peaks(
            waveform, prominence=min_prominence, distance=5
        )

        peak_count = len(peaks)

        # Calculate relative positions (0-1 scale)
        if peak_count > 0:
            peak_positions = (peaks / len(waveform)).tolist()
            peak_prominences = properties["prominences"].tolist()

            # Calculate inter-peak intervals
            if peak_count > 1:
                inter_peak_intervals = (np.diff(peaks) / sample_rate).tolist()
            else:
                inter_peak_intervals = []
        else:
            peak_positions = []
            peak_prominences = []
            inter_peak_intervals = []

        return PeakFeatures(
            peak_count=peak_count,
            peak_positions=peak_positions,
            peak_prominences=peak_prominences,
            inter_peak_intervals=inter_peak_intervals,
        )

    def extract_statistical_features(self, waveform: np.ndarray) -> StatisticalFeatures:
        """
        Extract statistical features from waveform.

        Args:
            waveform: 1D array of flow values

        Returns:
            StatisticalFeatures object with statistical metrics

        Example:
            >>> stats = extractor.extract_statistical_features(flow)
            >>> print(f"Mean: {stats.mean:.2f}, StdDev: {stats.std_dev:.2f}")
        """
        if len(waveform) == 0:
            return StatisticalFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0)

        mean_val = np.mean(waveform)
        median_val = np.median(waveform)
        std_val = np.std(waveform)

        # Percentiles
        p25, p50, p75, p95 = np.percentile(waveform, [25, 50, 75, 95])

        # Coefficient of variation
        if mean_val != 0:
            cv = std_val / abs(mean_val)
        else:
            cv = 0.0

        # Zero crossing rate (for sign changes)
        if len(waveform) > 1:
            zero_crossings = np.sum(np.diff(np.sign(waveform)) != 0)
            zcr = zero_crossings / (len(waveform) - 1)
        else:
            zcr = 0.0

        return StatisticalFeatures(
            mean=mean_val,
            median=median_val,
            std_dev=std_val,
            percentile_25=p25,
            percentile_50=p50,
            percentile_75=p75,
            percentile_95=p95,
            coefficient_of_variation=cv,
            zero_crossing_rate=zcr,
        )

    def extract_spectral_features(
        self, waveform: np.ndarray, sample_rate: float
    ) -> SpectralFeatures:
        """
        Extract spectral (frequency domain) features from waveform.

        Uses FFT to analyze frequency content. Useful for detecting
        periodic patterns like CSR or periodic breathing.

        Args:
            waveform: 1D array of flow values
            sample_rate: Sample rate in Hz

        Returns:
            SpectralFeatures object with frequency analysis

        Example:
            >>> spectral = extractor.extract_spectral_features(flow, 25.0)
            >>> print(f"Dominant frequency: {spectral.dominant_frequency:.3f} Hz")
        """
        if len(waveform) < 4:
            return SpectralFeatures(0.0, 0.0, np.array([]))

        # Compute power spectral density using Welch's method
        frequencies, psd = signal.welch(
            waveform, fs=sample_rate, nperseg=min(len(waveform), 256)
        )

        # Find dominant frequency (excluding DC component)
        if len(frequencies) > 1:
            dominant_idx = np.argmax(psd[1:]) + 1
            dominant_frequency = frequencies[dominant_idx]
        else:
            dominant_frequency = 0.0

        # Calculate spectral entropy
        # Normalize PSD to probability distribution
        psd_norm = psd / np.sum(psd)
        # Remove zeros to avoid log(0)
        psd_nonzero = psd_norm[psd_norm > 0]
        spectral_entropy = -np.sum(psd_nonzero * np.log2(psd_nonzero))

        return SpectralFeatures(
            dominant_frequency=dominant_frequency,
            spectral_entropy=spectral_entropy,
            power_spectral_density=psd,
        )
