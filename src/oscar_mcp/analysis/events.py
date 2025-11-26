"""Event data structures for analysis pipeline."""

from dataclasses import dataclass
from typing import Optional


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
    confidence: Optional[float] = None
    flow_reduction: Optional[float] = None
    has_desaturation: Optional[bool] = None
    baseline_flow: Optional[float] = None
