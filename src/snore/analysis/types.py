"""Analysis pipeline type definitions."""

from typing import Any

from pydantic import BaseModel, Field

from snore.analysis.modes.types import ModeResult


class AnalysisEvent(BaseModel):
    """
    Respiratory event structure for analysis processing.

    Note: This is distinct from models.unified.RespiratoryEvent which is the
    canonical storage format. AnalysisEvent uses float timestamps for performance
    and includes analysis-specific metadata (source, confidence).
    """

    event_type: str = Field(description="Event type")
    start_time: float = Field(description="Unix timestamp")
    duration: float = Field(ge=0, description="Event duration (seconds)")
    source: str = Field(description="Event source (machine/programmatic)")
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Detection confidence"
    )
    flow_reduction: float | None = Field(
        default=None, ge=0, le=1, description="Flow reduction (0-1)"
    )
    has_desaturation: bool | None = Field(
        default=None, description="Has SpO2 desaturation"
    )
    baseline_flow: float | None = Field(
        default=None, description="Baseline flow (L/min)"
    )


class AnalysisResult(BaseModel):
    """Results from session analysis."""

    session_id: int = Field(description="Database session ID")
    session_duration_hours: float = Field(ge=0, description="Session duration (hours)")
    total_breaths: int = Field(ge=0, description="Total breaths segmented")
    machine_events: list[AnalysisEvent] = Field(description="Machine-flagged events")
    mode_results: dict[str, ModeResult] = Field(description="Results by detection mode")
    flow_analysis: dict[str, Any] | None = Field(
        default=None, description="Flow limitation analysis"
    )
    csr_detection: dict[str, Any] | None = Field(
        default=None, description="Cheyne-Stokes Respiration detection"
    )
    periodic_breathing: dict[str, Any] | None = Field(
        default=None, description="Periodic breathing detection"
    )
    timestamp_start: float = Field(default=0.0, description="Session start timestamp")
    timestamp_end: float = Field(default=0.0, description="Session end timestamp")
