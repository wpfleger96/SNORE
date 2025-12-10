"""Detection mode type definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from snore.constants import EventDetectionConstants as EDC


class BaselineMethod(str, Enum):
    """Method for calculating breath baseline."""

    TIME = "time"  # Time-based window (AASM-compliant)
    BREATH = "breath"  # Breath-count window


@dataclass(frozen=True)
class DetectionModeConfig:
    """
    Configuration for a detection mode.

    All threshold values use decimal format (0.9 = 90%, 0.3 = 30%).
    Uses constants from EventDetectionConstants as defaults.
    """

    # Identity
    name: str
    description: str

    # Baseline calculation
    baseline_method: BaselineMethod
    baseline_window: float  # seconds for TIME, count for BREATH
    baseline_percentile: int = 90

    # Apnea detection
    apnea_threshold: float = EDC.APNEA_FLOW_REDUCTION_THRESHOLD  # 0.90
    apnea_validation_threshold: float = EDC.APNEA_FLOW_REDUCTION_THRESHOLD  # 0.90

    # Hypopnea detection
    hypopnea_min_threshold: float = EDC.HYPOPNEA_MIN_REDUCTION  # 0.30
    hypopnea_max_threshold: float = EDC.HYPOPNEA_MAX_REDUCTION  # 0.89

    # Shared parameters
    min_event_duration: float = EDC.MIN_EVENT_DURATION  # 10.0
    merge_gap: float = EDC.MERGE_GAP_SECONDS  # 3.0
    metric: Literal["amplitude", "tidal_volume"] = "amplitude"


@dataclass
class ModeResult:
    """Result from a single detection mode."""

    mode_name: str
    apneas: list[Any]  # list[ApneaEvent] - from shared/types
    hypopneas: list[Any]  # list[HypopneaEvent] - from shared/types
    ahi: float
    rdi: float
    metadata: dict[str, Any]  # Mode-specific debug info
