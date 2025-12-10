"""Detection mode type definitions."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from snore.analysis.shared.types import ApneaEvent, HypopneaEvent
from snore.constants import EventDetectionConstants as EDC


class BaselineMethod(str, Enum):
    """Method for calculating breath baseline."""

    TIME = "time"  # Time-based window (AASM-compliant)
    BREATH = "breath"  # Breath-count window


class DetectionModeConfig(BaseModel):
    """
    Configuration for a detection mode.

    All threshold values use decimal format (0.9 = 90%, 0.3 = 30%).
    Uses constants from EventDetectionConstants as defaults.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str = Field(description="Mode name (e.g., 'aasm')")
    description: str = Field(description="Mode description")

    # Baseline calculation
    baseline_method: BaselineMethod = Field(description="Baseline calculation method")
    baseline_window: float = Field(
        description="Window size (seconds for TIME, count for BREATH)"
    )
    baseline_percentile: int = Field(
        default=90, ge=0, le=100, description="Baseline percentile"
    )

    # Apnea detection
    apnea_threshold: float = Field(
        default=EDC.APNEA_FLOW_REDUCTION_THRESHOLD,
        ge=0,
        le=1,
        description="Apnea flow reduction threshold",
    )
    apnea_validation_threshold: float = Field(
        default=EDC.APNEA_FLOW_REDUCTION_THRESHOLD,
        ge=0,
        le=1,
        description="Apnea validation threshold",
    )

    # Hypopnea detection
    hypopnea_min_threshold: float = Field(
        default=EDC.HYPOPNEA_MIN_REDUCTION,
        ge=0,
        le=1,
        description="Hypopnea minimum threshold",
    )
    hypopnea_max_threshold: float = Field(
        default=EDC.HYPOPNEA_MAX_REDUCTION,
        ge=0,
        le=1,
        description="Hypopnea maximum threshold",
    )

    # Shared parameters
    min_event_duration: float = Field(
        default=EDC.MIN_EVENT_DURATION,
        ge=0,
        description="Minimum event duration (seconds)",
    )
    merge_gap: float = Field(
        default=EDC.MERGE_GAP_SECONDS, ge=0, description="Event merge gap (seconds)"
    )
    metric: Literal["amplitude", "tidal_volume"] = Field(
        default="amplitude", description="Metric for baseline calculation"
    )


class ModeResult(BaseModel):
    """Result from a single detection mode."""

    mode_name: str = Field(description="Mode name")
    apneas: list[ApneaEvent] = Field(description="Detected apnea events")
    hypopneas: list[HypopneaEvent] = Field(description="Detected hypopnea events")
    ahi: float = Field(ge=0, description="Apnea-Hypopnea Index")
    rdi: float = Field(ge=0, description="Respiratory Disturbance Index")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Mode-specific debug info"
    )
