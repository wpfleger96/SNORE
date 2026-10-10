"""
Complex breathing pattern detection algorithm.

This module detects Cheyne-Stokes Respiration (CSR) and periodic breathing
from tidal-volume time series using autocorrelation, envelope analysis, and
spectral analysis.
"""

import logging

from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from scipy import signal, stats

from snore.analysis.shared.types import (
    CSRDetection,
    PeriodicBreathingDetection,
)
from snore.constants import PatternDetectionConstants as PDC

T = TypeVar("T", CSRDetection, PeriodicBreathingDetection)

logger = logging.getLogger(__name__)

__all__ = [
    "ComplexPatternDetector",
    "CSRDetection",
    "PeriodicBreathingDetection",
]


class ComplexPatternDetector:
    """
    Detects complex breathing patterns from time-series data.

    Finds the dominant cycle by autocorrelation, then confirms Cheyne-Stokes
    Respiration with envelope (waxing/waning) analysis and periodic breathing
    with spectral regularity.

    Example:
        >>> detector = ComplexPatternDetector()
        >>> csr = detector.detect_csr(
        ...     timestamps, tidal_volumes, window_minutes=10
        ... )
        >>> if csr is not None and csr.confidence > 0.7:
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

        cycle_and_peak = self._find_dominant_cycle(
            autocorr, timestamps, min_cycle, max_cycle
        )

        if cycle_and_peak is None:
            return None

        cycle_length, peak_height = cycle_and_peak

        amplitude_var = np.std(smoothed_tv) / np.mean(smoothed_tv)

        waxing_waning_score = self._detect_waxing_waning(smoothed_tv, cycle_length)

        if waxing_waning_score < PDC.WAXING_WANING_MIN_SCORE:
            return None

        cycle_count = int((timestamps[-1] - timestamps[0]) / cycle_length)

        if cycle_count < self.min_cycle_count:
            return None

        csr_time = self._calculate_csr_time_percentage(smoothed_tv, cycle_length)

        confidence = self._calculate_csr_confidence(
            self._periodicity_strength(peak_height),
            amplitude_var,
            waxing_waning_score,
            cycle_count,
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

        cycle_and_peak = self._find_dominant_cycle(
            autocorr, timestamps, min_cycle, max_cycle
        )

        if cycle_and_peak is None:
            return None

        cycle_length, peak_height = cycle_and_peak

        regularity = self._calculate_regularity_score(smoothed_tv, cycle_length)

        if regularity < PDC.REGULARITY_MIN_SCORE:
            return None

        has_apneas = self._check_for_apneas(tidal_volumes)

        confidence = self._calculate_periodic_confidence(
            self._periodicity_strength(peak_height), regularity, has_apneas
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
    ) -> tuple[float, float] | None:
        """Find the dominant cycle from autocorrelation.

        Returns (cycle_length, peak_height), where peak_height is bias-corrected
        for grading; detection gates on the raw autocorrelation.
        """
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

        heights = properties["peak_heights"]
        best = int(np.argmax(heights))
        cycle_lag = min_lag + peaks[best]
        cycle_length = cycle_lag * sample_interval

        # The biased autocorrelation (normalized only by lag 0) sums N - lag
        # products, so even a perfect cycle peaks near (N - lag) / N and long
        # cycles would grade weaker than short ones. Rescale to undo that.
        n = len(autocorr)
        unbiased_peak_height = heights[best] * n / (n - cycle_lag)

        return float(cycle_length), float(unbiased_peak_height)

    def _periodicity_strength(self, peak_height: float) -> float:
        """Grade how far the autocorrelation peak clears the detection gate.

        0 at ``autocorr_threshold`` (barely periodic), 1 at a perfect peak.
        """
        if self.autocorr_threshold >= 1:
            return 1.0

        strength = (peak_height - self.autocorr_threshold) / (
            1 - self.autocorr_threshold
        )

        if not np.isfinite(strength):
            return 0.0

        return float(np.clip(strength, 0.0, 1.0))

    def _detect_waxing_waning(
        self, signal_data: np.ndarray, cycle_length: float
    ) -> float:
        """Detect crescendo-decrescendo (waxing-waning) pattern."""
        if len(signal_data) < 10:
            return 0.0

        envelope_upper = self._extract_envelope(signal_data, upper=True)

        envelope_variation = np.std(envelope_upper) / np.mean(envelope_upper)

        if envelope_variation < PDC.ENVELOPE_VARIATION_MIN:
            return 0.0

        envelope_max = np.max(envelope_upper)
        envelope_min = np.min(envelope_upper)
        amplitude_ratio = envelope_max / envelope_min if envelope_min > 0 else 0.0

        if amplitude_ratio < PDC.CSR_MIN_ENVELOPE_RATIO:
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
        periodicity_strength: float,
        amplitude_var: float,
        waxing_waning: float,
        cycle_count: int,
    ) -> float:
        """Score a CSR detection in [0.5, 1.0].

        0.5 base for passing every detection gate, up to +0.2 graded from
        ``periodicity_strength``, and +0.1 each for strong amplitude variation,
        strong waxing/waning, and enough cycles.
        """
        earned = sum(
            (
                amplitude_var > PDC.CSR_MIN_AMPLITUDE_VAR,
                waxing_waning > PDC.CSR_MIN_WAXING_WANING,
                cycle_count >= PDC.CSR_MIN_CYCLES_HIGH_CONF,
            )
        )

        return min(1.0, 0.5 + 0.2 * periodicity_strength + 0.1 * earned)

    def _calculate_periodic_confidence(
        self,
        periodicity_strength: float,
        regularity: float,
        has_apneas: bool,
    ) -> float:
        """Score a periodic breathing detection in [0.5, 1.0].

        0.5 base for passing every detection gate, up to +0.1 graded from
        ``periodicity_strength``, and +0.2 each for high spectral regularity
        and the presence of apneas.
        """
        earned = sum((regularity > PDC.PERIODIC_HIGH_REGULARITY, has_apneas))

        return min(1.0, 0.5 + 0.1 * periodicity_strength + 0.2 * earned)

    def detect_csr_episodes(
        self,
        timestamps: np.ndarray,
        tidal_volumes: np.ndarray,
        window_minutes: float = 10.0,
        step_minutes: float = 2.0,
    ) -> list[CSRDetection]:
        """
        Detect CSR episodes using windowed analysis.

        Uses a sliding window approach to identify time-localized CSR patterns
        rather than analyzing the entire session as a single window.

        Args:
            timestamps: Time values (seconds from session start)
            tidal_volumes: Tidal volume measurements (mL)
            window_minutes: Analysis window size (minutes)
            step_minutes: Step size for sliding window (minutes)

        Returns:
            List of CSRDetection objects with accurate per-episode timing
        """
        if len(timestamps) == 0 or len(tidal_volumes) == 0:
            return []

        window_seconds = window_minutes * 60.0
        step_seconds = step_minutes * 60.0

        session_duration = timestamps[-1] - timestamps[0]
        if session_duration < window_seconds:
            detection = self.detect_csr(timestamps, tidal_volumes, window_minutes)
            return [detection] if detection else []

        episodes: list[CSRDetection] = []
        current_time = timestamps[0]

        while current_time + window_seconds <= timestamps[-1]:
            window_end = current_time + window_seconds

            mask = (timestamps >= current_time) & (timestamps < window_end)
            window_timestamps = timestamps[mask]
            window_tv = tidal_volumes[mask]

            if len(window_timestamps) > 10:
                detection = self.detect_csr(
                    window_timestamps, window_tv, window_minutes
                )
                if detection:
                    episodes.append(detection)

            current_time += step_seconds

        merged_episodes = self._merge_overlapping_episodes(
            episodes, gap_threshold=PDC.CSR_EPISODE_MERGE_GAP_SECONDS
        )

        logger.debug(
            f"Detected {len(episodes)} CSR windows, merged to {len(merged_episodes)} episodes"
        )

        return merged_episodes

    def detect_periodic_breathing_episodes(
        self,
        timestamps: np.ndarray,
        tidal_volumes: np.ndarray,
        respiratory_rate: np.ndarray,
        window_minutes: float = 10.0,
        step_minutes: float = 2.0,
    ) -> list[PeriodicBreathingDetection]:
        """
        Detect periodic breathing episodes using windowed analysis.

        Args:
            timestamps: Time values (seconds from session start)
            tidal_volumes: Tidal volume measurements (mL)
            respiratory_rate: Respiratory rate measurements (breaths/min)
            window_minutes: Analysis window size (minutes)
            step_minutes: Step size for sliding window (minutes)

        Returns:
            List of PeriodicBreathingDetection objects
        """
        if len(timestamps) == 0 or len(tidal_volumes) == 0:
            return []

        window_seconds = window_minutes * 60.0
        step_seconds = step_minutes * 60.0

        session_duration = timestamps[-1] - timestamps[0]
        if session_duration < window_seconds:
            detection = self.detect_periodic_breathing(
                timestamps, tidal_volumes, respiratory_rate
            )
            return [detection] if detection else []

        episodes: list[PeriodicBreathingDetection] = []
        current_time = timestamps[0]

        while current_time + window_seconds <= timestamps[-1]:
            window_end = current_time + window_seconds

            mask = (timestamps >= current_time) & (timestamps < window_end)
            window_timestamps = timestamps[mask]
            window_tv = tidal_volumes[mask]
            window_rr = respiratory_rate[mask]

            if len(window_timestamps) > 10:
                detection = self.detect_periodic_breathing(
                    window_timestamps, window_tv, window_rr
                )
                if detection:
                    episodes.append(detection)

            current_time += step_seconds

        merged_episodes = self._merge_overlapping_pb_episodes(
            episodes, gap_threshold=PDC.CSR_EPISODE_MERGE_GAP_SECONDS
        )

        logger.debug(
            f"Detected {len(episodes)} periodic breathing windows, "
            f"merged to {len(merged_episodes)} episodes"
        )

        return merged_episodes

    def _merge_episodes_generic(
        self,
        episodes: Sequence[T],
        gap_threshold: float,
        merge_fn: Callable[[T, T], T],
    ) -> list[T]:
        """
        Generic merge for overlapping or nearby episodes.

        Args:
            episodes: List of episode objects (CSRDetection or PeriodicBreathingDetection)
            gap_threshold: Maximum gap in seconds to merge episodes
            merge_fn: Callback function to merge two episodes

        Returns:
            List of merged episode objects
        """
        if not episodes:
            return []

        sorted_episodes = sorted(episodes, key=lambda e: e.start_time)
        merged = []

        current = sorted_episodes[0]

        for next_episode in sorted_episodes[1:]:
            gap = next_episode.start_time - current.end_time

            if gap <= gap_threshold:
                current = merge_fn(current, next_episode)
            else:
                merged.append(current)
                current = next_episode

        merged.append(current)
        return merged

    def _merge_overlapping_episodes(
        self, episodes: list[CSRDetection], gap_threshold: float
    ) -> list[CSRDetection]:
        """
        Merge overlapping or nearby CSR episodes.

        Args:
            episodes: List of CSRDetection objects
            gap_threshold: Maximum gap in seconds to merge episodes

        Returns:
            List of merged CSRDetection objects
        """

        def merge_csr(current: CSRDetection, next_ep: CSRDetection) -> CSRDetection:
            return CSRDetection(
                start_time=current.start_time,
                end_time=next_ep.end_time,
                cycle_length=(current.cycle_length + next_ep.cycle_length) / 2,
                amplitude_variation=(
                    current.amplitude_variation + next_ep.amplitude_variation
                )
                / 2,
                csr_index=(current.csr_index + next_ep.csr_index) / 2,
                confidence=max(current.confidence, next_ep.confidence),
                cycle_count=current.cycle_count + next_ep.cycle_count,
            )

        return self._merge_episodes_generic(episodes, gap_threshold, merge_csr)

    def _merge_overlapping_pb_episodes(
        self, episodes: list[PeriodicBreathingDetection], gap_threshold: float
    ) -> list[PeriodicBreathingDetection]:
        """
        Merge overlapping or nearby periodic breathing episodes.

        Args:
            episodes: List of PeriodicBreathingDetection objects
            gap_threshold: Maximum gap in seconds to merge episodes

        Returns:
            List of merged PeriodicBreathingDetection objects
        """

        def merge_pb(
            current: PeriodicBreathingDetection, next_ep: PeriodicBreathingDetection
        ) -> PeriodicBreathingDetection:
            return PeriodicBreathingDetection(
                start_time=current.start_time,
                end_time=next_ep.end_time,
                cycle_length=(current.cycle_length + next_ep.cycle_length) / 2,
                regularity_score=(current.regularity_score + next_ep.regularity_score)
                / 2,
                confidence=max(current.confidence, next_ep.confidence),
                has_apneas=current.has_apneas or next_ep.has_apneas,
            )

        return self._merge_episodes_generic(episodes, gap_threshold, merge_pb)
