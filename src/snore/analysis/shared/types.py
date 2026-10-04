"""Shared analysis algorithm type definitions."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from snore.constants import ApneaEventType
from snore.provenance import Provenance, provenance_field


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
        amplitude: Peak-to-peak amplitude (peak_insp + |peak_exp|) in L/min
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
        classification_confidence: Confidence in OA vs CA vs MA classification (0-1)
        baseline_flow: Baseline flow before event (L/min)
        detection_method: Method used to detect event (amplitude, gap, near_zero_flow)
    """

    start_time: float = Field(description="Event start timestamp (seconds)")
    end_time: float = Field(description="Event end timestamp (seconds)")
    duration: float = provenance_field(
        Provenance.EXPERIMENTAL, "Event duration (seconds)", ge=0
    )
    event_type: ApneaEventType = Field(description="Apnea type")
    flow_reduction: float = provenance_field(
        Provenance.EXPERIMENTAL, "Flow reduction (0-1)", ge=0, le=1
    )
    confidence: float = provenance_field(
        Provenance.EXPERIMENTAL, "Detection confidence (0-1)", ge=0, le=1
    )
    classification_confidence: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "Confidence in OA/CA/MA classification (0-1)",
        default=0.5,
        ge=0,
        le=1,
    )
    baseline_flow: float = provenance_field(
        Provenance.EXPERIMENTAL, "Baseline flow before event (L/min)"
    )
    detection_method: str = Field(
        default="amplitude",
        description="Detection method (amplitude, gap, near_zero_flow)",
    )


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
    duration: float = provenance_field(
        Provenance.EXPERIMENTAL, "Event duration (seconds)", ge=0
    )
    flow_reduction: float = provenance_field(
        Provenance.EXPERIMENTAL, "Flow reduction (0-1)", ge=0, le=1
    )
    confidence: float = provenance_field(
        Provenance.EXPERIMENTAL, "Detection confidence (0-1)", ge=0, le=1
    )
    baseline_flow: float = provenance_field(
        Provenance.EXPERIMENTAL, "Baseline flow before event (L/min)"
    )
    has_arousal: bool | None = provenance_field(
        Provenance.EXPERIMENTAL, "Arousal detected", default=None
    )
    has_desaturation: bool | None = provenance_field(
        Provenance.EXPERIMENTAL, "SpO2 desaturation ≥3%", default=None
    )


class RERAEvent(BaseModel):
    """
    Detected RERA (Respiratory Effort-Related Arousal) event.

    Detected from flow patterns (FLOW event algorithm) without EEG.
    Represents sequences of flow-limited breaths ending with a recovery breath.

    Attributes:
        start_time: Event start timestamp (seconds)
        end_time: Event end timestamp (seconds)
        duration: Event duration (seconds)
        obstructed_breath_count: Number of breaths showing flow limitation
        recovery_amplitude_increase_pct: Recovery breath amplitude increase (% vs baseline)
        confidence: Detection confidence (0-1, lower without EEG)
        baseline_flow: Baseline flow before event (L/min)
    """

    start_time: float = Field(description="Event start timestamp (seconds)")
    end_time: float = Field(description="Event end timestamp (seconds)")
    duration: float = provenance_field(
        Provenance.EXPERIMENTAL, "Event duration (seconds)", ge=0
    )
    obstructed_breath_count: int = provenance_field(
        Provenance.EXPERIMENTAL, "Breaths showing flow limitation", ge=2
    )
    recovery_amplitude_increase_pct: float = provenance_field(
        Provenance.EXPERIMENTAL, "Recovery breath amplitude increase (%)", ge=0
    )
    confidence: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "Detection confidence (0-1, lower without EEG)",
        ge=0,
        le=1,
    )
    baseline_flow: float = provenance_field(
        Provenance.EXPERIMENTAL, "Baseline flow before event (L/min)"
    )


class EventTimeline(BaseModel):
    """
    Complete timeline of detected respiratory events.

    Attributes:
        apneas: List of detected apnea events
        hypopneas: List of detected hypopnea events
        reras: List of detected RERA events
        total_events: Total count of all events
        ahi: Apnea-Hypopnea Index (events per hour)
        rdi: Respiratory Disturbance Index (AHI + RERAs per hour)
    """

    apneas: list[ApneaEvent] = Field(description="Detected apnea events")
    hypopneas: list[HypopneaEvent] = Field(description="Detected hypopnea events")
    reras: list[RERAEvent] = Field(
        default_factory=list, description="Detected RERA events"
    )
    total_events: int = Field(ge=0, description="Total event count")
    ahi: float = Field(ge=0, description="Apnea-Hypopnea Index (events/hour)")
    rdi: float = Field(
        ge=0, description="Respiratory Disturbance Index (AHI + RERAs/hour)"
    )


class ShapeFeatures(BaseModel):
    """
    Shape characteristics of a breath waveform.

    These features describe the overall shape of the inspiratory flow curve
    and are critical for flow limitation classification.

    Attributes:
        flatness_index: Ratio of time spent >80% of peak (0-1)
            High values indicate plateau/flattened waveforms
        plateau_duration: Duration of plateau phase in seconds
            Retained as a standalone feature and as the intermediate from which
            plateau_fraction is derived.
        plateau_fraction: Plateau duration as a fraction of inspiration time (0-1)
            Duration-normalized form used by the classifier.
        symmetry_score: Statistical skewness (-1 to 1)
            0 = symmetric, + = right-skewed, - = left-skewed
        kurtosis: Measure of peakedness vs flatness
            High = sharp peak, Low = flat plateau
        rise_time: Time from 10% to 90% of peak flow (seconds)
        fall_time: Time from 90% to 10% of peak flow (seconds)
    """

    flatness_index: float = Field(ge=0, le=1, description="Plateau time ratio")
    plateau_duration: float = Field(ge=0, description="Plateau duration (seconds)")
    plateau_fraction: float = Field(
        ge=0, le=1, description="Plateau duration / inspiration time (0-1)"
    )
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

    @model_validator(mode="after")
    def _positions_align_with_prominences(self) -> "PeakFeatures":
        """Dominant-peak selection indexes ``peak_positions`` by the argmax of
        ``peak_prominences``; the two lists must stay parallel."""
        if len(self.peak_positions) != len(self.peak_prominences):
            raise ValueError(
                "peak_positions and peak_prominences must have equal length "
                f"(got {len(self.peak_positions)} and {len(self.peak_prominences)})"
            )
        return self


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
