"""Analysis engine type definitions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ProgrammaticAnalysisResult:
    """
    Complete programmatic analysis result for a session.

    Attributes:
        session_id: Database session ID
        timestamp_start: Analysis start time
        timestamp_end: Analysis end time
        duration_hours: Session duration in hours

        flow_analysis: Flow limitation classification results
        event_timeline: Respiratory event detection results
        csr_detection: Cheyne-Stokes Respiration detection (if found)
        periodic_breathing: Periodic breathing detection (if found)
        positional_analysis: Positional event clustering (if found)

        total_breaths: Total number of breaths analyzed
        processing_time_ms: Analysis processing time in milliseconds
        confidence_summary: Average confidence scores by analysis type
        clinical_summary: Human-readable summary of findings
    """

    session_id: int
    timestamp_start: float
    timestamp_end: float
    duration_hours: float

    flow_analysis: dict[str, Any]
    event_timeline: dict[str, Any]
    csr_detection: dict[str, Any] | None
    periodic_breathing: dict[str, Any] | None
    positional_analysis: dict[str, Any] | None

    total_breaths: int
    processing_time_ms: float
    confidence_summary: dict[str, float]
    clinical_summary: str
    machine_events: list[Any] | None = None
    breaths: list[Any] | None = None
