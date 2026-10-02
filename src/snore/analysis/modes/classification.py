"""Apnea type classification, effort estimation and confidence scoring."""

from typing import Literal

import numpy as np

from scipy import signal

from snore.analysis.modes.types import HypopneaMode
from snore.constants import EventDetectionConstants as EDC


def _classify_apnea_type(
    flow_signal: np.ndarray | None,
    sample_rate: float,
) -> tuple[Literal["OA", "CA", "MA", "UA"], float]:
    """
    Classify apnea as obstructive, central, or unclassified.

    Without effort sensors, estimates effort from flow characteristics.

    Args:
        flow_signal: Flow values during the apnea event
        sample_rate: Sampling rate in Hz

    Returns:
        Tuple of (event_type, classification_confidence)
        - event_type: "OA", "CA", "MA", or "UA"
        - classification_confidence: 0-1 score based on effort score distinctiveness
    """
    if flow_signal is not None and len(flow_signal) > 5:
        effort_from_flow = _estimate_effort_from_flow(flow_signal, sample_rate)

        if effort_from_flow > EDC.APNEA_EFFORT_OBSTRUCTIVE_MIN:
            distance_from_boundary = min(
                effort_from_flow - EDC.APNEA_EFFORT_OBSTRUCTIVE_MIN, 0.35
            )
            classification_confidence = 0.5 + (distance_from_boundary / 0.35) * 0.5
            return "OA", float(classification_confidence)

        elif effort_from_flow < EDC.APNEA_EFFORT_CENTRAL_MAX:
            distance_from_boundary = min(
                EDC.APNEA_EFFORT_CENTRAL_MAX - effort_from_flow, 0.05
            )
            classification_confidence = 0.5 + (distance_from_boundary / 0.05) * 0.5
            return "CA", float(classification_confidence)

        else:
            distance_from_midpoint = abs(
                effort_from_flow - EDC.APNEA_EFFORT_MIXED_MIDPOINT
            )
            classification_confidence = 0.3 + (distance_from_midpoint / 0.05) * 0.2
            return "MA", float(classification_confidence)

    return "UA", 0.2


def _estimate_effort_from_flow(flow_signal: np.ndarray, sample_rate: float) -> float:
    """
    Estimate respiratory effort from flow signal characteristics.

    Args:
        flow_signal: Flow values during the event
        sample_rate: Sampling rate in Hz

    Returns:
        Normalized effort magnitude (0.0-1.0, where 0.0 = no effort, 1.0 = maximum effort)
    """
    if len(flow_signal) < 5:
        return 0.0

    flow_std = np.std(flow_signal)
    flow_range = np.ptp(flow_signal)

    detrended = flow_signal - np.mean(flow_signal)
    variations = np.abs(np.diff(detrended))
    avg_variation = np.mean(variations) if len(variations) > 0 else 0.0

    spectral_power = _calculate_spectral_effort(flow_signal, sample_rate)

    normalized_std = min(flow_std / 30.0, 1.0)
    normalized_range = min(flow_range / 100.0, 1.0)
    normalized_variation = min(avg_variation / 20.0, 1.0)

    effort_score = (
        normalized_std * 0.3
        + normalized_range * 0.3
        + normalized_variation * 0.2
        + spectral_power * 0.2
    )

    return float(effort_score)


def _calculate_spectral_effort(
    flow_signal: np.ndarray,
    sample_rate: float,
) -> float:
    """
    Calculate spectral power in breathing frequency range (0.1-0.5 Hz).

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


def _calculate_apnea_confidence(
    reduction: float, duration: float, baseline: float
) -> float:
    """Calculate confidence score for apnea detection."""
    confidence = EDC.APNEA_BASE_CONFIDENCE

    if reduction > EDC.APNEA_HIGH_REDUCTION_THRESHOLD:
        confidence += EDC.APNEA_HIGH_REDUCTION_BONUS
    if duration > EDC.APNEA_LONG_DURATION_THRESHOLD:
        confidence += EDC.APNEA_LONG_DURATION_BONUS
    if baseline > EDC.APNEA_HIGH_BASELINE_THRESHOLD:
        confidence += EDC.APNEA_BASELINE_FLOW_BONUS

    return min(1.0, confidence)


def _calculate_hypopnea_confidence(
    reduction: float,
    duration: float,
    has_desaturation: bool | None,
    detection_mode: HypopneaMode,
) -> float:
    """
    Calculate confidence score for hypopnea detection.

    Confidence levels by detection method:
    - HIGH: SpO2-validated (≥50% reduction OR 30-50% with desaturation)
    - MEDIUM: Flow-only ≥50% reduction
    - LOW: Flow-only 30-50% reduction

    Args:
        reduction: Flow reduction percentage (0-1)
        duration: Event duration in seconds
        has_desaturation: Whether SpO2 desaturation occurred (if available)
        detection_mode: Detection mode used

    Returns:
        Confidence score (0-1)
    """
    if detection_mode == HypopneaMode.FLOW_ONLY:
        if reduction >= 0.50:
            confidence = 0.6  # MEDIUM
        else:
            confidence = 0.4  # LOW
    else:
        confidence = EDC.HYPOPNEA_BASE_CONFIDENCE

    if (
        EDC.HYPOPNEA_IDEAL_MIN_REDUCTION
        <= reduction
        <= EDC.HYPOPNEA_IDEAL_MAX_REDUCTION
    ):
        confidence += 0.1

    if duration > EDC.HYPOPNEA_LONG_DURATION_THRESHOLD:
        confidence += 0.1

    if has_desaturation:
        confidence += EDC.HYPOPNEA_DESATURATION_BONUS

    return min(1.0, confidence)


def _check_desaturation(spo2_values: np.ndarray, threshold: float = 3.0) -> bool:
    """
    Check if SpO2 desaturation occurred.

    Args:
        spo2_values: SpO2 signal values
        threshold: Desaturation threshold (default 3% for AASM, 4% for CMS)

    Returns:
        True if desaturation >= threshold occurred
    """
    if len(spo2_values) < 2:
        return False

    max_spo2 = np.max(spo2_values)
    min_spo2 = np.min(spo2_values)
    drop = max_spo2 - min_spo2

    return bool(drop >= threshold)


def _calculate_rera_confidence(
    breath_count: int, amplitude_increase: float, duration: float
) -> float:
    """
    Calculate confidence score for RERA detection.

    RERAs detected from flow patterns (without EEG) have inherently
    lower confidence than EEG-confirmed events.

    Confidence factors:
    - More obstructed breaths = higher confidence
    - Larger recovery amplitude = higher confidence
    - Longer duration = higher confidence

    Args:
        breath_count: Number of flow-limited breaths in sequence
        amplitude_increase: Recovery breath amplitude increase (0-1 = 0-100%)
        duration: Event duration in seconds

    Returns:
        Confidence score (0-1), typically 0.4-0.7 for flow-only detection
    """
    confidence = 0.4

    if breath_count >= 3:
        confidence += 0.1
    if breath_count >= 5:
        confidence += 0.1

    if amplitude_increase >= 1.0:  # 100% increase (doubling)
        confidence += 0.2
    elif amplitude_increase >= 0.75:  # 75% increase
        confidence += 0.1

    if duration >= 15.0:
        confidence += 0.1

    return min(0.7, confidence)  # Cap at 0.7 without EEG
