"""Analysis pipeline type definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from snore.analysis.modes.types import ModeResult
from snore.constants import (
    EVENT_TYPE_CENTRAL_APNEA,
    EVENT_TYPE_CLEAR_AIRWAY,
    EVENT_TYPE_HYPOPNEA,
    EVENT_TYPE_MIXED_APNEA,
    EVENT_TYPE_OBSTRUCTIVE_APNEA,
)
from snore.provenance import Provenance, provenance_field

# Machine event types counted toward the machine-event AHI recount (apneas +
# hypopneas).  Device-flagged RERAs are excluded.
_MACHINE_AHI_EVENT_TYPES: frozenset[str] = frozenset(
    {
        EVENT_TYPE_OBSTRUCTIVE_APNEA,
        EVENT_TYPE_CENTRAL_APNEA,
        EVENT_TYPE_CLEAR_AIRWAY,
        EVENT_TYPE_MIXED_APNEA,
        EVENT_TYPE_HYPOPNEA,
    }
)


class AnalysisEvent(BaseModel):
    """
    Respiratory event structure for analysis processing.

    Note: This is distinct from models.unified.RespiratoryEvent which is the
    canonical storage format. AnalysisEvent uses float timestamps for performance
    and includes analysis-specific metadata (source, confidence).
    """

    event_type: str = Field(description="Event type")
    start_time: float = Field(description="Session offset (seconds from session start)")
    duration: float = provenance_field(
        Provenance.DEVICE,
        "Event duration (seconds); experimental for programmatic events",
        source_field="source",
        ge=0,
    )
    source: str = Field(description="Event source (machine/programmatic)")
    confidence: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Detection confidence", default=None, ge=0, le=1
    )
    flow_reduction: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Flow reduction (0-1)", default=None, ge=0, le=1
    )
    has_desaturation: bool | None = provenance_field(
        Provenance.EXPERIMENTAL, "Has SpO2 desaturation", default=None
    )
    baseline_flow: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Baseline flow (L/min)", default=None
    )


def _machine_ahi_rdi(
    machine_events: list[AnalysisEvent], session_duration_hours: float
) -> tuple[float | None, float | None]:
    """Recount machine events into AHI/RDI over waveform-coverage hours.

    Returns ``(None, None)`` when there are no machine events.  The RDI slot
    mirrors the AHI (device-flagged RERAs are not added), so it is not a true
    RDI; it is kept only for the persisted ``machine_rdi`` field.
    """
    if not machine_events:
        return None, None
    count = sum(1 for e in machine_events if e.event_type in _MACHINE_AHI_EVENT_TYPES)
    ahi = count / session_duration_hours if session_duration_hours > 0 else 0.0
    return ahi, ahi


class AnalysisResult(BaseModel):
    """Results from session analysis."""

    session_id: int = Field(description="Database session ID")
    session_duration_hours: float = provenance_field(
        Provenance.DERIVED, "Flow-waveform coverage (hours)", ge=0
    )
    total_breaths: int = provenance_field(
        Provenance.EXPERIMENTAL, "Total breaths segmented", ge=0
    )
    machine_events: list[AnalysisEvent] = Field(description="Machine-flagged events")
    machine_ahi: float | None = provenance_field(
        Provenance.DERIVED,
        "Machine-scored apneas + hypopneas per waveform-coverage hour; a recount, "
        "not the device-reported AHI (None if no events)",
        default=None,
        ge=0,
    )
    machine_rdi: float | None = provenance_field(
        Provenance.DERIVED,
        "Mirrors machine_ahi (machine-scored apneas + hypopneas per hour); "
        "device-flagged RERAs are not added, so this is not a true RDI "
        "(None if no events)",
        default=None,
        ge=0,
    )
    mode_results: dict[str, ModeResult] = Field(description="Results by detection mode")
    flow_analysis: dict[str, Any] | None = Field(
        default=None, description="Flow limitation analysis"
    )
    csr_detection: dict[str, Any] | None = Field(
        default=None, description="Cheyne-Stokes Respiration detection (summary)"
    )
    periodic_breathing: dict[str, Any] | None = Field(
        default=None, description="Periodic breathing detection (summary)"
    )
    csr_episodes: list[dict[str, Any]] | None = Field(
        default=None, description="Time-localized CSR episodes"
    )
    periodic_breathing_episodes: list[dict[str, Any]] | None = Field(
        default=None, description="Time-localized periodic breathing episodes"
    )
    pulse_change_count: int | None = provenance_field(
        Provenance.EXPERIMENTAL, "Total pulse change events detected", default=None
    )
    pulse_change_index: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Pulse changes per hour", default=None
    )
    timestamp_start: float = Field(default=0.0, description="Session start timestamp")
    timestamp_end: float = Field(default=0.0, description="Session end timestamp")

    @classmethod
    def from_stored_json(cls, data: dict[str, Any]) -> AnalysisResult:
        """Deserialize a stored ``programmatic_result_json`` payload.

        Analyses persisted before ``machine_ahi``/``machine_rdi`` were added to
        this DTO have no such keys, so ``model_validate`` defaults them to
        ``None`` even when the run had machine events.  Backfill them here from
        the still-present machine events and session duration so stored analyses
        render the same index a fresh run would.  Freshly computed payloads
        already carry the fields and are left untouched.
        """
        result = cls.model_validate(data)
        if result.machine_ahi is None and result.machine_events:
            result.machine_ahi, result.machine_rdi = _machine_ahi_rdi(
                result.machine_events, result.session_duration_hours
            )
        return result


# ---------------------------------------------------------------------------
# Per-breath compute envelope
# ---------------------------------------------------------------------------


@dataclass
class ComputedBreath:
    """Per-breath derived fields produced during compute_analysis.

    This is the private compute-layer representation — it NEVER enters the
    public AnalysisResult DTO or programmatic_result_json.  It is persisted
    as a models.Breath row via AnalysisComputation.
    """

    breath_number: int
    start_offset_s: float
    end_offset_s: float

    # Timing
    inspiration_time_s: float | None
    expiration_time_s: float | None
    total_time_s: float | None
    i_e_ratio: float | None
    duty_cycle: float | None

    # Amplitude
    peak_flow_lpm: float | None  # peak inspiratory flow L/min
    peak_exp_flow_lpm: float | None  # peak expiratory flow L/min
    tidal_volume_ml: float | None
    respiratory_rate_rolling: float | None

    # Flow shape
    flatness_index: float | None  # time-above-80%-peak
    mid_insp_flattening: float | None  # mid-insp flow ÷ peak (new)

    # Flow classification
    flow_class: int | None
    flow_confidence: float | None

    # Recovery flag (from primary mode's RERA detector)
    is_recovery_breath: bool | None

    # Trigger/cycle (experimental)
    inferred_trigger_type: str | None
    trigger_confidence: float | None
    inferred_cycle_type: str | None
    cycle_confidence: float | None
    trigger_cycle_applicable: bool | None
    trigger_cycle_reason: str | None

    # Quality flags
    leak_valid: bool | None
    leak_valid_reason: str | None
    ramp_active: bool | None
    ramp_active_reason: str | None
    mask_off: bool | None
    mask_off_reason: str | None


@dataclass
class AnalysisComputation:
    """Private compute envelope returned by compute_analysis().

    Contains both the public AnalysisResult summary (stored in
    programmatic_result_json) and the per-breath list (stored as models.Breath
    children of the AnalysisResult row).  Breaths NEVER enter the public DTO.
    """

    summary: AnalysisResult
    breaths: list[ComputedBreath] = field(default_factory=list)
    primary_mode: str = "aasm"
