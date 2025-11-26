"""Event data structures for analysis pipeline."""

from dataclasses import dataclass


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
