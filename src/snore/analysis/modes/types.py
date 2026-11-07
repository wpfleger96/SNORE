"""Detection mode type definitions."""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from snore.analysis.shared.types import ApneaEvent, HypopneaEvent, RERAEvent
from snore.constants import EventDetectionConstants as EDC
from snore.provenance import (
    Provenance,
    provenance_computed_field,
    provenance_field,
)


class BaselineMethod(StrEnum):
    """Method for calculating breath baseline."""

    TIME = "time"  # Time-based window (AASM-compliant)
    BREATH = "breath"  # Breath-count window


class HypopneaMode(StrEnum):
    """Method for detecting hypopnea events."""

    # Every mode scores hypopneas from the configured flow-reduction band; a
    # desaturation only adds confidence.
    AASM_3PCT = "aasm_3pct"  # checks a 3% SpO2 drop (AASM recommended)
    AASM_4PCT = "aasm_4pct"  # checks a 4% SpO2 drop (CMS/Medicare)
    FLOW_ONLY = "flow_only"  # flow band only, SpO2 ignored (ResMed-style)
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
        description="Hypopnea minimum flow reduction (inclusive)",
    )
    hypopnea_max_threshold: float = Field(
        default=EDC.HYPOPNEA_MAX_REDUCTION,
        ge=0,
        le=1,
        description="Hypopnea maximum average flow reduction (exclusive)",
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
    rera_reduction_min: float = Field(
        default=EDC.RERA_REDUCTION_MIN,
        ge=0,
        le=1,
        description="RERA sequence: minimum amplitude reduction (inclusive)",
    )
    rera_reduction_max: float = Field(
        default=EDC.RERA_REDUCTION_MAX,
        ge=0,
        le=1,
        description="RERA sequence: maximum amplitude reduction (exclusive)",
    )
    rera_recovery_reduction_max: float = Field(
        default=EDC.RERA_RECOVERY_REDUCTION_MAX,
        ge=0,
        le=1,
        description="RERA recovery breath: maximum amplitude reduction (exclusive)",
    )
    rera_recovery_increase_min: float = Field(
        default=EDC.RERA_RECOVERY_INCREASE_MIN,
        ge=0,
        description="RERA recovery breath: minimum amplitude increase over sequence mean",
    )

    @model_validator(mode="after")
    def _validate_threshold_bands(self) -> "DetectionModeConfig":
        """Reject empty half-open [min, max) reduction bands (min >= max)."""
        if self.rera_reduction_min >= self.rera_reduction_max:
            raise ValueError(
                "rera_reduction_min must be < rera_reduction_max "
                f"(got {self.rera_reduction_min} >= {self.rera_reduction_max})"
            )
        if self.hypopnea_min_threshold >= self.hypopnea_max_threshold:
            raise ValueError(
                "hypopnea_min_threshold must be < hypopnea_max_threshold "
                f"(got {self.hypopnea_min_threshold} >= {self.hypopnea_max_threshold})"
            )
        return self


class ModeResult(BaseModel):
    """Result from a single detection mode."""

    mode_name: str = Field(description="Mode name")
    apneas: list[ApneaEvent] = Field(description="Detected apnea events")
    hypopneas: list[HypopneaEvent] = Field(description="Detected hypopnea events")
    reras: list[RERAEvent] = Field(
        default_factory=list, description="Detected RERA events"
    )
    ahi: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "Apnea-Hypopnea Index from programmatic detection",
        ge=0,
    )
    rdi: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "Respiratory Disturbance Index (AHI + RERAs/hour). RERAs here come "
        "from the analysis-time amplitude-crescendo detector "
        "(detector.py::_detect_reras). This is a DIFFERENT RERA definition "
        "from the nightly rdi in NightlyAnalysisSummary, which uses the "
        "query-time FL-run proxy over stored breath rows; the two indices "
        "disagree by construction.",
        ge=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Mode-specific debug info"
    )

    @provenance_computed_field(Provenance.EXPERIMENTAL, "Detected apnea count")  # type: ignore[prop-decorator]
    @property
    def apnea_count(self) -> int:
        return len(self.apneas)

    @provenance_computed_field(Provenance.EXPERIMENTAL, "Detected hypopnea count")  # type: ignore[prop-decorator]
    @property
    def hypopnea_count(self) -> int:
        return len(self.hypopneas)

    @provenance_computed_field(Provenance.EXPERIMENTAL, "Detected RERA count")  # type: ignore[prop-decorator]
    @property
    def rera_count(self) -> int:
        return len(self.reras)
