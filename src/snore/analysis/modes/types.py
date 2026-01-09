"""Detection mode type definitions."""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from snore.analysis.shared.types import ApneaEvent, HypopneaEvent, RERAEvent
from snore.constants import EventDetectionConstants as EDC


class BaselineMethod(str, Enum):
    """Method for calculating breath baseline."""

    TIME = "time"  # Time-based window (AASM-compliant)
    BREATH = "breath"  # Breath-count window


class HypopneaMode(str, Enum):
    """Method for detecting hypopnea events."""

    AASM_3PCT = "aasm_3pct"  # 30% flow + 3% SpO2 drop (AASM recommended)
    AASM_4PCT = "aasm_4pct"  # 30% flow + 4% SpO2 drop (CMS/Medicare)
    FLOW_ONLY = "flow_only"  # 40% flow reduction (ResMed-style, no SpO2)
    DISABLED = "disabled"  # Skip hypopnea detection


class DetectionModeConfig(BaseModel):
    """
    Configuration for a detection mode.

    All threshold values use decimal format (0.9 = 90%, 0.3 = 30%).
    Uses constants from EventDetectionConstants as defaults.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Mode name (e.g., 'aasm')")
    description: str = Field(description="Mode description")

    baseline_method: BaselineMethod = Field(description="Baseline calculation method")
    baseline_window: float = Field(
        description="Window size (seconds for TIME, count for BREATH)"
    )
    baseline_percentile: int = Field(
        default=90, ge=0, le=100, description="Baseline percentile"
    )

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

    hypopnea_mode: HypopneaMode = Field(
        default=HypopneaMode.AASM_3PCT, description="Hypopnea detection mode"
    )
    hypopnea_flow_only_fallback: bool = Field(
        default=True,
        description="Fall back to flow-only detection if no SpO2 data available",
    )
    rera_detection_enabled: bool = Field(
        default=True, description="Enable RERA-like event detection"
    )


class ModeResult(BaseModel):
    """Result from a single detection mode."""

    mode_name: str = Field(description="Mode name")
    apneas: list[ApneaEvent] = Field(description="Detected apnea events")
    hypopneas: list[HypopneaEvent] = Field(description="Detected hypopnea events")
    reras: list[RERAEvent] = Field(
        default_factory=list, description="Detected RERA events"
    )
    ahi: float = Field(ge=0, description="Apnea-Hypopnea Index")
    rdi: float = Field(
        ge=0, description="Respiratory Disturbance Index (AHI + RERAs/hour)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Mode-specific debug info"
    )
