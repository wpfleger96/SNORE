"""
Base detection mode for respiratory event detection.

Provides abstract interface and shared utilities for different detection algorithms.
"""

import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from scipy import signal

from snore.analysis.algorithms.event_detector import ApneaEvent, HypopneaEvent
from snore.constants import EventDetectionConstants as EDC

logger = logging.getLogger(__name__)


@dataclass
class ModeResult:
    """Result from a single detection mode."""

    mode_name: str
    apneas: list[ApneaEvent]
    hypopneas: list[HypopneaEvent]
    ahi: float
    rdi: float
    metadata: dict[str, Any]  # Mode-specific debug info


class DetectionMode(ABC):
    """
    Abstract base class for respiratory event detection algorithms.

    Each mode implements a different detection strategy (AASM-compliant,
    machine approximation, etc.) but shares common utilities for baseline
    calculation, event merging, and apnea classification.
    """

    name: str  # e.g., "aasm", "resmed"
    description: str  # Human-readable description

    @abstractmethod
    def detect_events(
        self,
        breaths: list[Any],
        flow_data: tuple[np.ndarray, np.ndarray] | None,
        session_duration_hours: float,
    ) -> ModeResult:
        """
        Run detection algorithm and return results.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)
            session_duration_hours: Total session duration in hours

        Returns:
            ModeResult with detected events and metrics
        """
        pass

    def _calculate_rolling_baseline(
        self,
        breaths: list[Any],
        current_idx: int,
        window_breaths: int = 30,
        percentile: int = 90,
        metric: str = "amplitude",
    ) -> float:
        """
        Calculate baseline from preceding breaths using percentile.

        Uses a rolling window of recent breaths to calculate baseline,
        excluding breaths that are part of detected events to avoid
        contaminating the baseline.

        Args:
            breaths: List of BreathMetrics objects
            current_idx: Index of current breath
            window_breaths: Number of preceding breaths to include
            percentile: Percentile to use (default 90th)
            metric: Which metric to use ("amplitude" or "tidal_volume")

        Returns:
            Baseline value, minimum 10.0 for amplitude or 100.0 for tidal_volume
        """
        if current_idx == 0:
            return 30.0 if metric == "amplitude" else 300.0

        start_idx = max(0, current_idx - window_breaths)
        window = breaths[start_idx:current_idx]

        if len(window) < 5:
            return 30.0 if metric == "amplitude" else 300.0

        # Extract metric values, excluding event breaths
        values = []
        for b in window:
            if metric == "amplitude":
                if hasattr(b, "amplitude") and b.amplitude > 0:
                    if not getattr(b, "in_event", False):
                        values.append(b.amplitude)
            elif metric == "tidal_volume":
                if hasattr(b, "tidal_volume") and b.tidal_volume > 0:
                    if not getattr(b, "in_event", False):
                        values.append(b.tidal_volume)

        if not values:
            return 30.0 if metric == "amplitude" else 300.0

        baseline = float(np.percentile(values, percentile))
        min_baseline = 10.0 if metric == "amplitude" else 100.0
        return max(baseline, min_baseline)

    def _calculate_time_based_baseline(
        self,
        breaths: list[Any],
        current_idx: int,
        window_seconds: float = 120.0,
        percentile: int = 90,
        metric: str = "amplitude",
    ) -> float:
        """
        Calculate baseline from breaths within time window (AASM-compliant).

        Uses a time-based window (default 2 minutes per AASM) of preceding breaths
        to calculate baseline, excluding breaths that are part of detected events.

        Args:
            breaths: List of BreathMetrics objects
            current_idx: Index of current breath
            window_seconds: Time window in seconds (default 120.0 = 2 minutes per AASM)
            percentile: Percentile to use (default 90th per AASM)
            metric: Which metric to use ("amplitude" or "tidal_volume")

        Returns:
            Baseline value, minimum 10.0 for amplitude or 100.0 for tidal_volume
        """
        if current_idx == 0:
            return 30.0 if metric == "amplitude" else 300.0

        current_breath = breaths[current_idx]
        if not hasattr(current_breath, "start_time"):
            # Fallback to breath-based if no timestamps
            return self._calculate_rolling_baseline(
                breaths,
                current_idx,
                window_breaths=30,
                percentile=percentile,
                metric=metric,
            )

        current_time = current_breath.start_time
        window_start = current_time - window_seconds

        # Collect breaths within time window
        values = []
        for i in range(current_idx - 1, -1, -1):
            breath = breaths[i]
            if not hasattr(breath, "start_time"):
                break
            if breath.start_time < window_start:
                break

            # Extract metric value, excluding event breaths
            if not getattr(breath, "in_event", False):
                if metric == "amplitude":
                    if hasattr(breath, "amplitude") and breath.amplitude > 0:
                        values.append(breath.amplitude)
                elif metric == "tidal_volume":
                    if hasattr(breath, "tidal_volume") and breath.tidal_volume > 0:
                        values.append(breath.tidal_volume)

        if len(values) < 5:
            return 30.0 if metric == "amplitude" else 300.0

        baseline = float(np.percentile(values, percentile))
        min_baseline = 10.0 if metric == "amplitude" else 100.0
        return max(baseline, min_baseline)

    def _merge_adjacent_events(
        self,
        events: list[Any],
        max_gap: float = 3.0,
    ) -> list[Any]:
        """
        Merge events that are close together in time AND of the same type.

        Per AASM standards, only events of the same type should be merged.
        Different event types (e.g., OA vs CA) should remain separate even
        if temporally adjacent.

        Args:
            events: List of ApneaEvent or HypopneaEvent objects
            max_gap: Maximum gap in seconds to merge (default 3.0)

        Returns:
            List of merged events
        """
        if len(events) <= 1:
            return events

        merged = []
        current = events[0]

        for next_event in events[1:]:
            gap = next_event.start_time - current.end_time
            same_type = getattr(next_event, "event_type", None) == getattr(
                current, "event_type", None
            )

            if gap <= max_gap and same_type:
                current = self._merge_two_events(current, next_event)
            else:
                merged.append(current)
                current = next_event

        merged.append(current)

        return merged

    def _merge_two_events(
        self,
        event1: Any,
        event2: Any,
    ) -> ApneaEvent | HypopneaEvent:
        """
        Merge two adjacent events of the same type.

        Args:
            event1: First event
            event2: Second event

        Returns:
            Merged event
        """
        merged_duration = event2.end_time - event1.start_time

        if isinstance(event1, ApneaEvent) and isinstance(event2, ApneaEvent):
            return ApneaEvent(
                start_time=event1.start_time,
                end_time=event2.end_time,
                duration=merged_duration,
                event_type=event1.event_type,
                flow_reduction=(event1.flow_reduction + event2.flow_reduction) / 2,
                confidence=min(event1.confidence, event2.confidence),
                baseline_flow=event1.baseline_flow,
            )
        elif isinstance(event1, HypopneaEvent) and isinstance(event2, HypopneaEvent):
            return HypopneaEvent(
                start_time=event1.start_time,
                end_time=event2.end_time,
                duration=merged_duration,
                flow_reduction=(event1.flow_reduction + event2.flow_reduction) / 2,
                confidence=min(event1.confidence, event2.confidence),
                baseline_flow=event1.baseline_flow,
                has_arousal=event1.has_arousal or event2.has_arousal,
                has_desaturation=event1.has_desaturation or event2.has_desaturation,
            )

        event_typed: ApneaEvent | HypopneaEvent = event1
        return event_typed

    def _classify_apnea_type(
        self,
        flow_signal: np.ndarray | None = None,
    ) -> str:
        """
        Classify apnea as obstructive, central, or unclassified.

        Without effort sensors, estimates effort from flow characteristics:
        - Obstructive Apnea (OA): Respiratory effort present but no airflow
          Flow shows more variability from attempted breathing
        - Central Apnea (CA): No respiratory effort, no airflow
          Flow is very flat and stable near zero

        Args:
            flow_signal: Flow values during the apnea event

        Returns:
            Event type code: "OA", "CA", "MA", or "UA"
        """
        if flow_signal is not None and len(flow_signal) > 5:
            effort_from_flow = self._estimate_effort_from_flow(flow_signal)

            if effort_from_flow > 0.15:
                return "OA"
            elif effort_from_flow < 0.05:
                return "CA"
            else:
                return "MA"

        return "UA"

    def _estimate_effort_from_flow(self, flow_signal: np.ndarray) -> float:
        """
        Estimate respiratory effort from flow signal characteristics.

        During obstructive apnea, continued respiratory effort causes flow
        oscillations even though net flow is near zero. Central apnea shows
        very flat flow.

        Uses AASM-recommended weighting:
        - Standard deviation (30%): Overall variability
        - Peak-to-peak range (30%): Amplitude of oscillations
        - Roughness/variation (20%): High-frequency changes
        - Spectral power (20%): Rhythmic breathing frequency content

        Args:
            flow_signal: Flow values during the event

        Returns:
            Estimated effort magnitude (0.0 = no effort, higher = more effort)
        """
        if len(flow_signal) < 5:
            return 0.0

        flow_std = np.std(flow_signal)
        flow_range = np.ptp(flow_signal)

        detrended = flow_signal - np.mean(flow_signal)
        variations = np.abs(np.diff(detrended))
        avg_variation = np.mean(variations) if len(variations) > 0 else 0.0

        spectral_power = self._calculate_spectral_effort(flow_signal)

        effort_score = (
            flow_std * 0.3
            + flow_range * 0.3
            + avg_variation * 0.2
            + spectral_power * 0.2
        )

        return float(effort_score)

    def _calculate_spectral_effort(
        self,
        flow_signal: np.ndarray,
        sample_rate: float = 25.0,
    ) -> float:
        """
        Calculate spectral power in breathing frequency range (0.1-0.5 Hz).

        OA: Continued effort creates rhythmic oscillations visible in spectrum
        CA: No effort results in flat spectrum with no breathing frequency peaks

        Args:
            flow_signal: Flow values during the event
            sample_rate: Sampling rate in Hz (default 25 Hz for CPAP devices)

        Returns:
            Normalized spectral power in breathing frequency range (0.0-1.0)
        """
        if len(flow_signal) < EDC.SPECTRAL_MIN_SAMPLES:
            return 0.0

        detrended = flow_signal - np.mean(flow_signal)
        freqs, power = signal.periodogram(detrended, fs=sample_rate)

        breathing_mask = (freqs >= EDC.BREATHING_FREQ_MIN) & (
            freqs <= EDC.BREATHING_FREQ_MAX
        )
        breathing_power = np.sum(power[breathing_mask])

        total_power = np.sum(power)
        if total_power > 0:
            return float(breathing_power / total_power)
        return 0.0
