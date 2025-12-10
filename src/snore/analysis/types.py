"""Analysis pipeline type definitions."""

from dataclasses import dataclass
from typing import Any

from snore.analysis.modes.types import ModeResult


@dataclass
class AnalysisEvent:
    """
    Respiratory event structure for analysis processing.

    Note: This is distinct from models.unified.RespiratoryEvent which is the
    canonical storage format. AnalysisEvent uses float timestamps for performance
    and includes analysis-specific metadata (source, confidence).
    """

    event_type: str
    start_time: float  # Unix timestamp for fast numeric operations
    duration: float
    source: str  # 'machine' or 'programmatic'
    confidence: float | None = None
    flow_reduction: float | None = None
    has_desaturation: bool | None = None
    baseline_flow: float | None = None


@dataclass
class AnalysisResult:
    """Results from session analysis."""

    session_id: int
    session_duration_hours: float
    total_breaths: int
    machine_events: list[AnalysisEvent]
    mode_results: dict[str, ModeResult]
    flow_analysis: dict[str, Any] | None = None
    positional_analysis: dict[str, Any] | None = None
    timestamp_start: float = 0.0  # For storage compatibility
    timestamp_end: float = 0.0
