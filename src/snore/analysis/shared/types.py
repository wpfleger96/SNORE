"""Shared analysis algorithm type definitions."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ============================================================================
# Breath Segmentation Types
# ============================================================================


@dataclass
class BreathPhases:
    """
    Inspiration and expiration phases of a single breath.

    Attributes:
        inspiration_indices: Indices where flow > 0 (breathing in)
        expiration_indices: Indices where flow < 0 (breathing out)
        inspiration_values: Flow values during inspiration
        expiration_values: Flow values during expiration
    """

    inspiration_indices: np.ndarray
    expiration_indices: np.ndarray
    inspiration_values: np.ndarray
    expiration_values: np.ndarray


@dataclass
class BreathMetrics:
    """
    Comprehensive metrics for a single breath.

    Attributes:
        breath_number: Sequential breath number in session
        start_time: Timestamp of breath start (seconds)
        middle_time: Timestamp of inspiration→expiration transition (seconds)
        end_time: Timestamp of breath end (seconds)
        duration: Total breath time (seconds)
        tidal_volume: Volume of air breathed in (mL)
        tidal_volume_smoothed: Smoothed TV using 5-point weighted average (mL)
        peak_inspiratory_flow: Maximum flow during inspiration (L/min)
        peak_expiratory_flow: Maximum absolute flow during expiration (L/min)
        inspiration_time: Duration of inspiration phase (seconds)
        expiration_time: Duration of expiration phase (seconds)
        i_e_ratio: Inspiration to expiration time ratio
        respiratory_rate: Instantaneous rate (60/duration) in breaths/min
        respiratory_rate_rolling: Rolling 60s window rate (breaths/min)
        minute_ventilation: Estimated ventilation using rolling RR (L/min)
        amplitude: Peak-to-peak amplitude (peak_insp - |peak_exp|) in L/min
        is_complete: Whether breath has both inspiration and expiration
    """

    breath_number: int
    start_time: float
    middle_time: float
    end_time: float
    duration: float
    tidal_volume: float
    tidal_volume_smoothed: float
    peak_inspiratory_flow: float
    peak_expiratory_flow: float
    inspiration_time: float
    expiration_time: float
    i_e_ratio: float
    respiratory_rate: float
    respiratory_rate_rolling: float
    minute_ventilation: float
    amplitude: float
    is_complete: bool


# ============================================================================
# Event Detection Types
# ============================================================================


@dataclass
class ApneaEvent:
    """
    Detected apnea event.

    Attributes:
        start_time: Event start timestamp (seconds)
        end_time: Event end timestamp (seconds)
        duration: Event duration (seconds)
        event_type: OA (obstructive), CA (central), UA (unclassified), or MA (mixed)
        flow_reduction: Percentage flow reduction (0-1)
        confidence: Detection confidence (0-1)
        baseline_flow: Baseline flow before event (L/min)
    """

    start_time: float
    end_time: float
    duration: float
    event_type: str
    flow_reduction: float
    confidence: float
    baseline_flow: float


@dataclass
class HypopneaEvent:
    """
    Detected hypopnea event.

    Attributes:
        start_time: Event start timestamp (seconds)
        end_time: Event end timestamp (seconds)
        duration: Event duration (seconds)
        flow_reduction: Percentage flow reduction (0-1)
        confidence: Detection confidence (0-1)
        baseline_flow: Baseline flow before event (L/min)
        has_arousal: Whether arousal was detected (if available)
        has_desaturation: Whether SpO2 desaturation occurred (if available)
    """

    start_time: float
    end_time: float
    duration: float
    flow_reduction: float
    confidence: float
    baseline_flow: float
    has_arousal: bool | None = None
    has_desaturation: bool | None = None


@dataclass
class EventTimeline:
    """
    Complete timeline of detected respiratory events.

    Attributes:
        apneas: List of detected apnea events
        hypopneas: List of detected hypopnea events
        total_events: Total count of all events
        ahi: Apnea-Hypopnea Index (events per hour)
        rdi: Respiratory Disturbance Index
    """

    apneas: list[ApneaEvent]
    hypopneas: list[HypopneaEvent]
    total_events: int
    ahi: float
    rdi: float


# ============================================================================
# Feature Extraction Types
# ============================================================================


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


# ============================================================================
# Flow Limitation Types
# ============================================================================


@dataclass
class FlowPattern:
    """
    Classification result for a single breath.

    Attributes:
        breath_number: Sequential breath number
        flow_class: Class number (1-7)
        class_name: Human-readable class name
        confidence: Confidence score (0-1)
        matched_features: Features that supported this classification
        severity: Clinical severity level
    """

    breath_number: int
    flow_class: int
    class_name: str
    confidence: float
    matched_features: dict[str, Any]
    severity: str


@dataclass
class SessionFlowAnalysis:
    """
    Flow limitation analysis for an entire session.

    Attributes:
        total_breaths: Total number of breaths analyzed
        class_distribution: Count of breaths in each class
        flow_limitation_index: Overall FL index (0-1)
        average_confidence: Mean confidence across all classifications
        patterns: Individual breath classifications
    """

    total_breaths: int
    class_distribution: dict[int, int]
    flow_limitation_index: float
    average_confidence: float
    patterns: list[FlowPattern]


# ============================================================================
# Pattern Detection Types
# ============================================================================


@dataclass
class CSRDetection:
    """
    Detected Cheyne-Stokes Respiration pattern.

    Attributes:
        start_time: Pattern start timestamp (seconds)
        end_time: Pattern end timestamp (seconds)
        cycle_length: Average cycle length (seconds)
        amplitude_variation: Coefficient of variation in tidal volume
        csr_index: Percentage of time in CSR pattern (0-1)
        confidence: Detection confidence (0-1)
        cycle_count: Number of complete cycles detected
    """

    start_time: float
    end_time: float
    cycle_length: float
    amplitude_variation: float
    csr_index: float
    confidence: float
    cycle_count: int


@dataclass
class PeriodicBreathingDetection:
    """
    Detected periodic breathing pattern.

    Attributes:
        start_time: Pattern start timestamp (seconds)
        end_time: Pattern end timestamp (seconds)
        cycle_length: Average cycle length (seconds)
        regularity_score: Measure of pattern regularity (0-1)
        confidence: Detection confidence (0-1)
        has_apneas: Whether pattern includes apneas
    """

    start_time: float
    end_time: float
    cycle_length: float
    regularity_score: float
    confidence: float
    has_apneas: bool


@dataclass
class PositionalAnalysis:
    """
    Positional event clustering analysis.

    Attributes:
        cluster_times: List of (start, end) tuples for event clusters
        cluster_event_counts: Number of events in each cluster
        positional_likelihood: Likelihood that clustering is position-related (0-1)
        confidence: Detection confidence (0-1)
        total_clusters: Number of clusters identified
    """

    cluster_times: list[tuple[float, float]]
    cluster_event_counts: list[int]
    positional_likelihood: float
    confidence: float
    total_clusters: int
