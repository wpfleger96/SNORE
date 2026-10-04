"""
Feature extraction algorithms for waveform analysis.

This module provides feature extraction from breath waveforms, including
shape characteristics and peak analysis. These features are used for flow
limitation classification and respiratory pattern analysis.
"""

import logging

import numpy as np

from scipy import signal, stats

from snore.analysis.shared.types import PeakFeatures, ShapeFeatures

logger = logging.getLogger(__name__)

__all__ = [
    "WaveformFeatureExtractor",
    "ShapeFeatures",
    "PeakFeatures",
    "compute_mid_insp_flattening",
    "largest_inspiratory_segment",
]


def largest_inspiratory_segment(flow: np.ndarray) -> np.ndarray:
    """Return the longest contiguous run of positive samples in ``flow``.

    Masking positive samples (``flow[flow > 0]``) concatenates non-adjacent
    inspiratory runs: a mid-breath dip below zero stitches two humps into one,
    corrupting peak counts and mid-inspiratory flattening.  This returns only
    the single longest positive run, preserving contiguity.

    Args:
        flow: 1-D flow array (L/min), time-ordered.

    Returns:
        The longest contiguous positive-valued slice, or an empty array if no
        sample is positive.  This is a view into ``flow``; do not mutate it.
    """
    if len(flow) == 0:
        return flow
    positive = (flow > 0).astype(int)
    changes = np.diff(np.concatenate([[0], positive, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    if len(starts) == 0:
        return flow[:0]
    idx = int(np.argmax(ends - starts))
    return flow[starts[idx] : ends[idx]]


def compute_mid_insp_flattening(insp_flow: np.ndarray) -> float | None:
    """Compute mid-inspiratory flattening index (new; distinct from flatness_index).

    Definition: mean flow in the middle third of inspiration ÷ peak
    inspiratory flow.  A value near 1.0 indicates a full, unimpeded
    mid-inspiration; a value below ~0.7 typically indicates flow limitation.

    Versioned as ``FLATTENING_ALGO_VERSION`` in ``versioning.py``.

    Args:
        insp_flow: 1-D NumPy array of **inspiratory** (positive) flow values
                   (L/min), time-ordered.  Should contain ≥6 samples for a
                   meaningful result.

    Returns:
        Float in [0, 1], or ``None`` if the array is too short or peak is
        non-positive.

    Example:
        >>> mid = compute_mid_insp_flattening(np.array([10, 30, 28, 26, 20, 8], dtype=float))
        >>> 0.0 <= mid <= 1.0
        True
    """
    if len(insp_flow) < 6:
        return None
    peak = float(np.max(insp_flow))
    if peak <= 0:
        return None
    n = len(insp_flow)
    mid_start = n // 3
    mid_end = (2 * n) // 3
    mid_mean = float(np.mean(insp_flow[mid_start:mid_end]))
    return float(np.clip(mid_mean / peak, 0.0, 1.0))


class WaveformFeatureExtractor:
    """
    Extracts comprehensive features from breath waveforms.

    Provides methods to extract shape and peak features from individual
    breath segments. These features are used
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
            return ShapeFeatures(
                flatness_index=0,
                plateau_duration=0,
                plateau_fraction=0,
                symmetry_score=0,
                kurtosis=0,
                rise_time=0,
                fall_time=0,
            )

        peak_flow = np.max(waveform)
        if peak_flow <= 0:
            return ShapeFeatures(
                flatness_index=0,
                plateau_duration=0,
                plateau_fraction=0,
                symmetry_score=0,
                kurtosis=0,
                rise_time=0,
                fall_time=0,
            )

        # Flatness index: ratio of time spent >80% of peak
        flatness_threshold_value = self.flatness_threshold * peak_flow
        above_threshold = waveform > flatness_threshold_value
        flatness_index = np.sum(above_threshold) / len(waveform)

        # Plateau duration: continuous time at high flow
        plateau_duration = self._calculate_plateau_duration(
            waveform, flatness_threshold_value, sample_rate
        )

        # Plateau fraction: plateau duration normalized by inspiration time, so
        # slow deep breaths don't clear absolute-second thresholds trivially.
        inspiration_time = len(waveform) / sample_rate
        plateau_fraction = (
            float(np.clip(plateau_duration / inspiration_time, 0.0, 1.0))
            if inspiration_time > 0
            else 0.0
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

        rise_time = self._calculate_rise_time(waveform, peak_flow, sample_rate)
        fall_time = self._calculate_fall_time(waveform, peak_flow, sample_rate)

        return ShapeFeatures(
            flatness_index=flatness_index,
            plateau_duration=plateau_duration,
            plateau_fraction=plateau_fraction,
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

        changes = np.diff(np.concatenate([[0], above_threshold, [0]]).astype(int))
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]

        if len(starts) == 0:
            return 0.0

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

        above_10 = np.where(waveform >= threshold_10)[0]
        if len(above_10) == 0:
            return 0.0
        idx_10 = above_10[0]

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

        peak_idx = np.argmax(waveform)

        after_peak = waveform[peak_idx:]
        above_90 = np.where(after_peak >= threshold_90)[0]
        if len(above_90) == 0:
            return 0.0
        idx_90 = peak_idx + above_90[-1]

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
            return PeakFeatures(
                peak_count=0,
                peak_positions=[],
                peak_prominences=[],
                inter_peak_intervals=[],
            )

        peak_flow = np.max(waveform)
        if peak_flow <= 0:
            return PeakFeatures(
                peak_count=0,
                peak_positions=[],
                peak_prominences=[],
                inter_peak_intervals=[],
            )

        min_prominence = self.peak_prominence_threshold * peak_flow
        peaks, properties = signal.find_peaks(
            waveform, prominence=min_prominence, distance=5
        )

        peak_count = len(peaks)

        if peak_count > 0:
            peak_positions = (peaks / len(waveform)).tolist()
            peak_prominences = properties["prominences"].tolist()

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
