"""
Pydantic models for analysis results.

These models may include aggregated/transformed data beyond what's in
the core analysis types.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from snore.analysis.shared.types import EventTimeline


class EventValidationResult(BaseModel):
    """
    Validation results comparing programmatic vs machine-detected events.

    Useful for tuning detection thresholds and assessing algorithm accuracy.
    """

    machine_event_count: int = Field(description="Events detected by CPAP machine")
    programmatic_event_count: int = Field(
        description="Events detected programmatically"
    )
    matched_events: int = Field(
        description="Events matched between machine and programmatic (within 5s)"
    )
    false_positives: int = Field(
        description="Programmatic events not matched to machine events"
    )
    false_negatives: int = Field(
        description="Machine events not matched to programmatic events"
    )
    sensitivity: float = Field(
        ge=0, le=1, description="Recall: matched / (matched + false_negatives)"
    )
    precision: float = Field(
        ge=0, le=1, description="Precision: matched / (matched + false_positives)"
    )
    f1_score: float = Field(
        ge=0,
        le=1,
        description="F1 score: 2 * (precision * sensitivity) / (precision + sensitivity)",
    )
    agreement_percentage: float = Field(
        ge=0,
        le=100,
        description="Overall agreement: matched / max(machine, programmatic) * 100",
    )


class AnalysisSummary(BaseModel):
    """
    High-level analysis summary for a CPAP session.

    This model provides the key metrics and findings from analysis.
    """

    session_id: int = Field(description="Database session ID")
    analysis_id: int | None = Field(
        default=None, description="Database analysis result ID"
    )
    timestamp_start: datetime = Field(description="Analysis start time")
    timestamp_end: datetime = Field(description="Analysis end time")
    duration_hours: float = Field(description="Session duration (hours)")

    mode_name: str = Field(description="Detection mode used (e.g., 'aasm')")
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


class FlowLimitationSummary(BaseModel):
    """Flow limitation classification summary."""

    flow_limitation_index: float = Field(description="Flow Limitation Index (0-1)")
    total_breaths: int = Field(description="Total breaths analyzed")
    class_distribution: dict[int, int] = Field(
        description="Breath count by flow class (1-7)"
    )
    average_confidence: float = Field(description="Average classification confidence")
    severity: str = Field(description="Overall severity (minimal/mild/moderate/severe)")


class DetailedAnalysisResult(BaseModel):
    """
    Complete detailed analysis result.

    This model contains all analysis data including individual events,
    patterns, and breath-by-breath classifications.
    """

    summary: AnalysisSummary = Field(description="High-level summary")
    mode_results: dict[str, EventTimeline] = Field(
        description="Event timelines by mode (e.g., {'aasm': ...})"
    )
    flow_limitation: FlowLimitationSummary = Field(
        description="Flow limitation analysis"
    )
    csr_detection: Any | None = Field(None, description="CSR pattern detection")
    periodic_breathing: Any | None = Field(
        None, description="Periodic breathing detection"
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
    analysis_id: int | None = Field(
        default=None, description="Analysis result ID if available"
    )
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
