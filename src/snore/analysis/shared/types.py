"""Shared analysis algorithm type definitions."""

from typing import Any, Literal

import numpy as np

from pydantic import BaseModel, ConfigDict, Field

# ============================================================================
# Breath Segmentation Types
# ============================================================================


class BreathPhases(BaseModel):
    """
    Inspiration and expiration phases of a single breath.

    Attributes:
        inspiration_indices: Indices where flow > 0 (breathing in)
        expiration_indices: Indices where flow < 0 (breathing out)
        inspiration_values: Flow values during inspiration
        expiration_values: Flow values during expiration
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inspiration_indices: np.ndarray
    expiration_indices: np.ndarray
    inspiration_values: np.ndarray
    expiration_values: np.ndarray


class BreathMetrics(BaseModel):
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

    breath_number: int = Field(description="Sequential breath number in session")
    start_time: float = Field(description="Breath start timestamp (seconds)")
    middle_time: float = Field(
        description="Inspiration→expiration transition (seconds)"
    )
    end_time: float = Field(description="Breath end timestamp (seconds)")
    duration: float = Field(ge=0, description="Total breath duration (seconds)")
    tidal_volume: float = Field(description="Volume of air breathed in (mL)")
    tidal_volume_smoothed: float = Field(description="Smoothed tidal volume (mL)")
    peak_inspiratory_flow: float = Field(description="Max inspiratory flow (L/min)")
    peak_expiratory_flow: float = Field(description="Max expiratory flow (L/min)")
    inspiration_time: float = Field(ge=0, description="Inspiration duration (seconds)")
    expiration_time: float = Field(ge=0, description="Expiration duration (seconds)")
    i_e_ratio: float = Field(ge=0, description="Inspiration/expiration time ratio")
    respiratory_rate: float = Field(ge=0, description="Instantaneous RR (breaths/min)")
    respiratory_rate_rolling: float = Field(
        ge=0, description="Rolling 60s RR (breaths/min)"
    )
    minute_ventilation: float = Field(ge=0, description="Estimated ventilation (L/min)")
    amplitude: float = Field(description="Peak-to-peak amplitude (L/min)")
    is_complete: bool = Field(description="Has both inspiration and expiration")
    in_event: bool = Field(
        default=False, description="Whether breath is part of a detected event"
    )


# ============================================================================
# Event Detection Types
# ============================================================================


class ApneaEvent(BaseModel):
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

    start_time: float = Field(description="Event start timestamp (seconds)")
    end_time: float = Field(description="Event end timestamp (seconds)")
    duration: float = Field(ge=0, description="Event duration (seconds)")
    event_type: Literal["OA", "CA", "MA", "UA"] = Field(description="Apnea type")
    flow_reduction: float = Field(ge=0, le=1, description="Flow reduction (0-1)")
    confidence: float = Field(ge=0, le=1, description="Detection confidence (0-1)")
    baseline_flow: float = Field(description="Baseline flow before event (L/min)")


class HypopneaEvent(BaseModel):
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

    start_time: float = Field(description="Event start timestamp (seconds)")
    end_time: float = Field(description="Event end timestamp (seconds)")
    duration: float = Field(ge=0, description="Event duration (seconds)")
    flow_reduction: float = Field(ge=0, le=1, description="Flow reduction (0-1)")
    confidence: float = Field(ge=0, le=1, description="Detection confidence (0-1)")
    baseline_flow: float = Field(description="Baseline flow before event (L/min)")
    has_arousal: bool | None = Field(default=None, description="Arousal detected")
    has_desaturation: bool | None = Field(
        default=None, description="SpO2 desaturation ≥3%"
    )


class EventTimeline(BaseModel):
    """
    Complete timeline of detected respiratory events.

    Attributes:
        apneas: List of detected apnea events
        hypopneas: List of detected hypopnea events
        total_events: Total count of all events
        ahi: Apnea-Hypopnea Index (events per hour)
        rdi: Respiratory Disturbance Index
    """

    apneas: list[ApneaEvent] = Field(description="Detected apnea events")
    hypopneas: list[HypopneaEvent] = Field(description="Detected hypopnea events")
    total_events: int = Field(ge=0, description="Total event count")
    ahi: float = Field(ge=0, description="Apnea-Hypopnea Index (events/hour)")
    rdi: float = Field(ge=0, description="Respiratory Disturbance Index")


# ============================================================================
# Feature Extraction Types
# ============================================================================


class ShapeFeatures(BaseModel):
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

    flatness_index: float = Field(ge=0, le=1, description="Plateau time ratio")
    plateau_duration: float = Field(ge=0, description="Plateau duration (seconds)")
    symmetry_score: float = Field(description="Statistical skewness")
    kurtosis: float = Field(description="Peakedness measure")
    rise_time: float = Field(ge=0, description="10-90% rise time (seconds)")
    fall_time: float = Field(ge=0, description="90-10% fall time (seconds)")


class PeakFeatures(BaseModel):
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

    peak_count: int = Field(ge=0, description="Number of peaks detected")
    peak_positions: list[float] = Field(description="Peak positions (0-1 scale)")
    peak_prominences: list[float] = Field(description="Peak heights")
    inter_peak_intervals: list[float] = Field(description="Peak spacing (seconds)")


class StatisticalFeatures(BaseModel):
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

    mean: float = Field(description="Mean flow value")
    median: float = Field(description="Median flow value")
    std_dev: float = Field(ge=0, description="Standard deviation")
    percentile_25: float = Field(description="25th percentile")
    percentile_50: float = Field(description="50th percentile (median)")
    percentile_75: float = Field(description="75th percentile")
    percentile_95: float = Field(description="95th percentile")
    coefficient_of_variation: float = Field(ge=0, description="CV (std/mean)")
    zero_crossing_rate: float = Field(ge=0, description="Sign change frequency")


class SpectralFeatures(BaseModel):
    """
    Spectral (frequency domain) features of breath waveform.

    Optional features that analyze frequency content using FFT.
    Useful for detecting periodic patterns.

    Attributes:
        dominant_frequency: Primary frequency component (Hz)
        spectral_entropy: Measure of spectral regularity
        power_spectral_density: Power distribution across frequencies
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dominant_frequency: float = Field(ge=0, description="Dominant frequency (Hz)")
    spectral_entropy: float = Field(ge=0, description="Spectral regularity measure")
    power_spectral_density: np.ndarray = Field(description="Power distribution")


# ============================================================================
# Flow Limitation Types
# ============================================================================


class FlowPattern(BaseModel):
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

    breath_number: int = Field(description="Sequential breath number")
    flow_class: int = Field(ge=1, le=7, description="Flow limitation class (1-7)")
    class_name: str = Field(description="Human-readable class name")
    confidence: float = Field(ge=0, le=1, description="Classification confidence")
    matched_features: dict[str, Any] = Field(description="Supporting features")
    severity: str = Field(description="Clinical severity level")


class SessionFlowAnalysis(BaseModel):
    """
    Flow limitation analysis for an entire session.

    Attributes:
        total_breaths: Total number of breaths analyzed
        class_distribution: Count of breaths in each class
        flow_limitation_index: Overall FL index (0-1)
        average_confidence: Mean confidence across all classifications
        patterns: Individual breath classifications
    """

    total_breaths: int = Field(ge=0, description="Total breaths analyzed")
    class_distribution: dict[int, int] = Field(description="Breaths per class")
    flow_limitation_index: float = Field(ge=0, le=1, description="Overall FL index")
    average_confidence: float = Field(ge=0, le=1, description="Mean confidence")
    patterns: list[FlowPattern] = Field(description="Individual breath classifications")


# ============================================================================
# Pattern Detection Types
# ============================================================================


class CSRDetection(BaseModel):
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

    start_time: float = Field(description="Pattern start timestamp (seconds)")
    end_time: float = Field(description="Pattern end timestamp (seconds)")
    cycle_length: float = Field(ge=0, description="Average cycle length (seconds)")
    amplitude_variation: float = Field(ge=0, description="Tidal volume variation")
    csr_index: float = Field(ge=0, le=1, description="% time in CSR (0-1)")
    confidence: float = Field(ge=0, le=1, description="Detection confidence")
    cycle_count: int = Field(ge=0, description="Complete cycles detected")


class PeriodicBreathingDetection(BaseModel):
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

    start_time: float = Field(description="Pattern start timestamp (seconds)")
    end_time: float = Field(description="Pattern end timestamp (seconds)")
    cycle_length: float = Field(ge=0, description="Average cycle length (seconds)")
    regularity_score: float = Field(ge=0, le=1, description="Pattern regularity")
    confidence: float = Field(ge=0, le=1, description="Detection confidence")
    has_apneas: bool = Field(description="Pattern includes apneas")
