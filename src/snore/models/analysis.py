"""
Pydantic models for analysis results.

These models provide structured representations of analysis results for
API responses, CLI output, and report generation.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class EventSummary(BaseModel):
    """Summary of a single respiratory event."""

    event_type: str = Field(description="Event type (apnea, hypopnea)")
    start_time: float = Field(description="Event start timestamp (seconds)")
    duration: float = Field(description="Event duration (seconds)")
    confidence: float = Field(description="Detection confidence (0-1)")


class ApneaSummary(EventSummary):
    """Detailed apnea event information."""

    apnea_type: str = Field(
        description="OA (obstructive), CA (central), MA (mixed), UA (unclassified)"
    )
    flow_reduction: float = Field(description="Flow reduction percentage (0-1)")
    baseline_flow: float = Field(description="Baseline flow rate (L/min)")


class HypopneaSummary(EventSummary):
    """Detailed hypopnea event information."""

    flow_reduction: float = Field(description="Flow reduction percentage (0-1)")
    has_arousal: bool = Field(description="Whether event terminated with arousal")
    has_desaturation: bool = Field(description="Whether event had ≥3% SpO2 drop")


class EventTimeline(BaseModel):
    """Complete respiratory event timeline for a session."""

    ahi: float = Field(description="Apnea-Hypopnea Index (events per hour)")
    rdi: float = Field(description="Respiratory Disturbance Index (events per hour)")
    total_events: int = Field(description="Total number of respiratory events")
    apneas: list[ApneaSummary] = Field(default_factory=list)
    hypopneas: list[HypopneaSummary] = Field(default_factory=list)


class FlowLimitationSummary(BaseModel):
    """Flow limitation classification summary."""

    flow_limitation_index: float = Field(description="Flow Limitation Index (0-1)")
    total_breaths: int = Field(description="Total breaths analyzed")
    class_distribution: dict[int, int] = Field(
        description="Breath count by flow class (1-7)"
    )
    average_confidence: float = Field(description="Average classification confidence")
    severity: str = Field(description="Overall severity (minimal/mild/moderate/severe)")


class CSRDetection(BaseModel):
    """Cheyne-Stokes Respiration pattern detection."""

    detected: bool = Field(description="Whether CSR was detected")
    cycle_length: float | None = Field(None, description="CSR cycle length (seconds)")
    amplitude_variation: float | None = Field(
        None, description="Amplitude variation (0-1)"
    )
    csr_index: float | None = Field(None, description="Percentage of time in CSR (0-1)")
    confidence: float | None = Field(None, description="Detection confidence (0-1)")
    cycle_count: int | None = Field(None, description="Number of CSR cycles detected")


class PeriodicBreathingDetection(BaseModel):
    """Periodic breathing pattern detection."""

    detected: bool = Field(description="Whether periodic breathing was detected")
    cycle_length: float | None = Field(None, description="Cycle length (seconds)")
    regularity_score: float | None = Field(None, description="Regularity score (0-1)")
    confidence: float | None = Field(None, description="Detection confidence (0-1)")
    has_apneas: bool | None = Field(None, description="Whether pattern includes apneas")


class PositionalAnalysis(BaseModel):
    """Positional apnea clustering analysis."""

    detected: bool = Field(description="Whether positional clustering was detected")
    cluster_count: int | None = Field(None, description="Number of event clusters")
    positional_likelihood: float | None = Field(
        None, description="Likelihood of positional OSA (0-1)"
    )
    confidence: float | None = Field(None, description="Detection confidence (0-1)")


class AnalysisSummary(BaseModel):
    """
    High-level analysis summary for a CPAP session.

    This model provides the key metrics and findings from analysis.
    """

    session_id: int = Field(description="Database session ID")
    analysis_id: int | None = Field(None, description="Database analysis result ID")
    timestamp_start: datetime = Field(description="Analysis start time")
    timestamp_end: datetime = Field(description="Analysis end time")
    duration_hours: float = Field(description="Session duration (hours)")

    ahi: float = Field(description="Apnea-Hypopnea Index")
    rdi: float = Field(description="Respiratory Disturbance Index")
    flow_limitation_index: float = Field(description="Flow Limitation Index")

    total_breaths: int = Field(description="Total breaths analyzed")
    total_events: int = Field(description="Total respiratory events")
    apnea_count: int = Field(description="Number of apneas")
    hypopnea_count: int = Field(description="Number of hypopneas")

    csr_detected: bool = Field(description="Cheyne-Stokes Respiration detected")
    periodic_breathing_detected: bool = Field(description="Periodic breathing detected")
    positional_events_detected: bool = Field(
        description="Positional event clustering detected"
    )

    severity_assessment: str = Field(
        description="Overall severity (normal/mild/moderate/severe)"
    )
    processing_time_ms: int = Field(
        description="Analysis processing time (milliseconds)"
    )


class DetailedAnalysisResult(BaseModel):
    """
    Complete detailed analysis result.

    This model contains all analysis data including individual events,
    patterns, and breath-by-breath classifications.
    """

    summary: AnalysisSummary = Field(description="High-level summary")
    event_timeline: EventTimeline = Field(description="All respiratory events")
    flow_limitation: FlowLimitationSummary = Field(
        description="Flow limitation analysis"
    )
    csr_detection: CSRDetection | None = Field(
        None, description="CSR pattern detection"
    )
    periodic_breathing: PeriodicBreathingDetection | None = Field(
        None, description="Periodic breathing detection"
    )
    positional_analysis: PositionalAnalysis | None = Field(
        None, description="Positional event analysis"
    )
    confidence_scores: dict[str, float] = Field(
        description="Confidence scores by analysis type"
    )
    clinical_summary: str = Field(description="Human-readable clinical summary")


class SessionAnalysisStatus(BaseModel):
    """Status of analysis for a session."""

    session_id: int = Field(description="Database session ID")
    session_date: datetime = Field(description="Session date")
    duration_hours: float = Field(description="Session duration (hours)")
    has_analysis: bool = Field(description="Whether analysis has been run")
    analysis_id: int | None = Field(None, description="Analysis result ID if available")
    analyzed_at: datetime | None = Field(
        None, description="When analysis was performed"
    )


class BatchAnalysisResult(BaseModel):
    """Results from batch analysis of multiple sessions."""

    total_sessions: int = Field(description="Total sessions processed")
    successful: int = Field(description="Successfully analyzed sessions")
    failed: int = Field(description="Failed analysis attempts")
    session_results: list[AnalysisSummary] = Field(
        description="Individual session results"
    )
    total_processing_time_ms: int = Field(description="Total processing time")
