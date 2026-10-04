"""Pydantic response schemas for SNORE MCP tools.

Timestamp contract (three tiers, A6):
  Tier 1 — absolute audit instants (e.g. ``AnalysisResult.created_at``):
    UTC ISO 8601 with ``Z`` suffix.
  Tier 2 — device/session wall-clock times (e.g. ``Event.start_time``,
    ``Session.start_time``): ISO 8601 string, conditionally offset-qualified.
    When ``timezone_status`` is ``"user_declared"`` (the profile declares an
    IANA timezone in ``timezone_name``, e.g. "America/New_York"), wall-clock
    strings carry a UTC offset (e.g. "2026-08-08T22:31:00-04:00") via
    ``localize_wall_clock()``.  When ``timezone_status`` is ``"unknown"``
    (no TZ declared), strings stay offset-free (naive ISO 8601), preserving
    the original DB representation.  The DB stores these as naive datetimes;
    no UTC offset is ever fabricated via ``.timestamp()`` / ``astimezone()``.
  Tier 3 — in-session positions: numeric ``offset_seconds`` from
    ``Session.start_time``.

Absent data is ``null`` with a companion ``*_reason`` field
(e.g. ``rera_index: null, rera_index_reason: "analysis_not_run"``).
All measurement fields carry their unit in the field name or tool docstring.

Metric fields declare a provenance tier via ``provenance_field``: the
description starts with ``[DEVICE]`` / ``[DERIVED]`` / ``[EXPERIMENTAL]`` and
the JSON schema carries ``x-provenance``.  Fields nested in a tagged parent
(e.g. ``EpochDistribution`` under ``EpochStats.mid_insp_flattening``) take the
less certain of the two tiers (``snore.provenance.response_provenance``).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

from snore.provenance import IndexSource, Provenance, provenance_field
from snore.services.schemas import (
    HEADLINE_INDEX_DESCRIPTION,
    INDEX_SOURCE_DESCRIPTION,
    MergedSettingsChange,
)


def tz_fields(source: Any) -> dict[str, Any]:
    """Schema kwargs for the tier-2 timezone companion fields.

    Accepts any service DTO with ``timezone_status`` / ``timezone_name``
    attributes, or the ``(TimezoneStatus, str | None)`` tuple returned by
    ``BreathService.resolve_timezone``.  Splat into a response constructor:
    ``**tz_fields(dto)``.
    """
    if isinstance(source, tuple):
        status, name = source
    else:
        status, name = source.timezone_status, source.timezone_name
    return {"timezone_status": str(status), "timezone_name": name}


def localize_wall_clock(dt: datetime, tz_status: str, tz_name: str | None) -> str:
    """Naive device wall-clock -> ISO 8601. Offset-qualified when the profile timezone is known."""
    if tz_status == "user_declared" and tz_name:
        try:
            return dt.replace(tzinfo=ZoneInfo(tz_name)).isoformat()
        except (ZoneInfoNotFoundError, ValueError):
            pass  # corrupted profile timezone: degrade to naive rather than break every tool
    return dt.isoformat()


class DeviceCapabilities(BaseModel):
    """Capabilities declared by the device/dataset for a queried range (G2)."""

    model_config = ConfigDict(populate_by_name=True)

    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
    has_flow_waveform: bool
    has_pressure_waveform: bool
    has_leak_waveform: bool
    has_spo2: bool
    has_events: bool
    has_analysis: bool
    notes: list[str] = []


class DeviceInfo(BaseModel):
    """Summary of a single device."""

    model_config = ConfigDict(populate_by_name=True)

    id: int
    manufacturer: str
    model: str
    serial_number: str
    first_session_date: date | None = None
    last_session_date: date | None = None
    session_count: int = 0
    therapy_modes: list[str] = []
    device_capabilities: DeviceCapabilities | None = None


class DataOverviewResponse(BaseModel):
    """Response from get_data_overview."""

    model_config = ConfigDict(populate_by_name=True)

    devices: list[DeviceInfo]
    date_range_start: date | None = None
    date_range_end: date | None = None
    total_sessions: int = 0
    available_waveform_channels: list[str] = []
    available_event_types: list[str] = []
    analysis_run: bool = False
    analysis_session_count: int = 0
    timezone_status: str = "unknown"
    timezone_name: str | None = None


class SettingsEpoch(BaseModel):
    """A contiguous period of stable therapy settings."""

    model_config = ConfigDict(populate_by_name=True)

    start_date: date
    end_date: date
    nights: int = provenance_field(
        Provenance.DERIVED, "Nights between start_date and end_date."
    )
    settings: dict[str, str | None] = provenance_field(
        Provenance.DEVICE, "Therapy settings recorded by the device."
    )
    changed_keys: list[str] = []
    device_id: int | None = None


class SettingsTimelineResponse(BaseModel):
    """Response from get_settings_timeline."""

    model_config = ConfigDict(populate_by_name=True)

    epochs: list[SettingsEpoch]
    total_epochs: int
    device_capabilities_by_device: dict[str, DeviceCapabilities] = {}


class SettingsChangeEntry(MergedSettingsChange):
    """One settings change from either the device settings log or the mask log.

    Field-identical subclass of services.schemas.MergedSettingsChange — kept as
    a distinct MCP-layer name for the docs://schemas registry.
    """

    model_config = ConfigDict(populate_by_name=True)


class SettingsChangesResponse(BaseModel):
    """Response from get_settings_changes."""

    model_config = ConfigDict(populate_by_name=True)

    changes: list[SettingsChangeEntry]
    total_changes: int


class NightlyRow(BaseModel):
    """Per-night summary row returned by get_nightly_summary."""

    model_config = ConfigDict(populate_by_name=True)

    date: date
    usage_hours: float | None = provenance_field(
        Provenance.DERIVED, "Mask-on therapy hours for the night.", default=None
    )
    session_count: int = 0

    # Headline indices: the device-reported daily value when trusted, otherwise
    # SNORE's recount (index_source says which); the recount, when computable,
    # is in *_computed.
    ahi: float | None = provenance_field(
        Provenance.DEVICE,
        f"Headline AHI (events/hr): {HEADLINE_INDEX_DESCRIPTION}.",
        source_field="index_source",
        default=None,
    )
    oai: float | None = provenance_field(
        Provenance.DEVICE,
        "Obstructive apnea index (events/hr); same source rule as ahi.",
        source_field="index_source",
        default=None,
    )
    cai: float | None = provenance_field(
        Provenance.DEVICE,
        "Central apnea index (events/hr); same source rule as ahi.",
        source_field="index_source",
        default=None,
    )
    hi: float | None = provenance_field(
        Provenance.DEVICE,
        "Hypopnea index (events/hr); same source rule as ahi.",
        source_field="index_source",
        default=None,
    )
    index_source: IndexSource | None = Field(
        default=None, description=f"{INDEX_SOURCE_DESCRIPTION}."
    )
    ahi_computed: float | None = provenance_field(
        Provenance.DERIVED,
        "SNORE recount AHI (events/hr): device-scored events over SNORE mask-on "
        "hours, usage-weighted across sessions; null when there are no mask-on "
        "hours or no session statistics.",
        default=None,
    )
    oai_computed: float | None = provenance_field(
        Provenance.DERIVED, "SNORE recount OAI (see ahi_computed).", default=None
    )
    cai_computed: float | None = provenance_field(
        Provenance.DERIVED, "SNORE recount CAI (see ahi_computed).", default=None
    )
    hi_computed: float | None = provenance_field(
        Provenance.DERIVED, "SNORE recount HI (see ahi_computed).", default=None
    )

    # Null when analysis has not been run.  rera_index/rdi use the query-time
    # FL-run proxy over stored breath rows — a DIFFERENT RERA definition from
    # the per-session analysis-time amplitude-crescendo detector
    # (ModeResult.rdi); the two disagree by construction.
    rera_index: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "RERA-proxy events per mask-on hour of analyzed sessions, from "
        "flow-limitation runs (FL-run proxy, not device-reported); "
        "rera_index_reason='duration_zero' when those sessions have zero "
        "mask-on hours.",
        default=None,
    )
    rera_index_reason: str | None = None
    rdi: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "SNORE-recount AHI (ahi_computed) plus the query-time experimental "
        "RERA-proxy index.",
        default=None,
    )
    rdi_reason: str | None = None

    pressure_median_cmh2o: float | None = provenance_field(
        Provenance.DERIVED, "Median pressure (cmH₂O).", default=None
    )
    pressure_95th_cmh2o: float | None = provenance_field(
        Provenance.DERIVED, "95th-percentile pressure (cmH₂O).", default=None
    )
    epap_median_cmh2o: float | None = provenance_field(
        Provenance.DERIVED, "Median EPAP (cmH₂O).", default=None
    )

    leak_median_lpm: float | None = provenance_field(
        Provenance.DERIVED, "Median leak (L/min).", default=None
    )
    leak_95th_lpm: float | None = provenance_field(
        Provenance.DERIVED, "95th-percentile leak (L/min).", default=None
    )
    leak_above_24_pct: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Percent of segmented breaths flagged as leak above 24 L/min.",
        default=None,
    )
    leak_above_24_pct_reason: str | None = None

    rr_mean_bpm: float | None = provenance_field(
        Provenance.DEVICE,
        "Respiratory rate (breaths/min), device session summary of the night's "
        "first session.",
        default=None,
    )
    tv_mean_ml: float | None = provenance_field(
        Provenance.DEVICE,
        "Tidal volume (mL), device session summary of the night's first session.",
        default=None,
    )
    mv_mean_lpm: float | None = provenance_field(
        Provenance.DEVICE,
        "Minute ventilation (L/min), device session summary of the night's "
        "first session.",
        default=None,
    )

    spo2_mean_pct: float | None = provenance_field(
        Provenance.DERIVED, "Mean SpO₂ (%).", default=None
    )

    # Breath-level FL/RERA fields (from BreathService.get_nightly_summary)
    fl_median: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Median per-breath flow-limitation score.",
        default=None,
    )
    fl_median_reason: str | None = None
    fl_p95: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "95th-percentile per-breath flow-limitation score.",
        default=None,
    )
    fl_p95_reason: str | None = None
    fl_max: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Maximum per-breath flow-limitation score.",
        default=None,
    )
    fl_max_reason: str | None = None
    fl_class_ge4_pct: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Percent of leak-valid, rule-matched classified breaths with "
        "flow_class >= 4; the confidence gate excludes fallback guesses "
        "(flow-limitation proxy).",
        default=None,
    )
    fl_class_ge4_pct_reason: str | None = None
    rera_proxy_count: int | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Count from the query-time FL-run proxy: runs of flow_class >= 4 "
        "ending in a recovery breath, over stored breath rows.",
        default=None,
    )
    rera_proxy_reason: str | None = None
    # Version of the query-time RERA-proxy criterion (independent of the
    # persisted AlgorithmIdentity); stamped only when the RERA scan ran
    # (rera_proxy_count non-null), null otherwise.
    rera_proxy_version: str | None = None

    # Null + reason when breath-level analysis hasn't run
    ti_median_s: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Median inspiratory time (s) over SNORE-segmented breaths.",
        default=None,
    )
    ti_median_reason: str | None = None
    ie_ratio: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Median I:E ratio over SNORE-segmented breaths.",
        default=None,
    )
    ie_ratio_reason: str | None = None

    # Device flow-limitation ("fl") and snore channels.  Negative sentinel
    # values filtered before aggregation; zeros retained.  Reason tokens:
    # "channel_absent" (session(s) had no such channel), "no_sessions" (the
    # day had no enabled sessions at all).
    device_flg_median: float | None = provenance_field(
        Provenance.DERIVED,
        "Median of the device flow-limitation channel (0–1 unitless).",
        default=None,
    )
    device_flg_95th: float | None = provenance_field(
        Provenance.DERIVED,
        "95th percentile of the device flow-limitation channel (0–1 unitless).",
        default=None,
    )
    device_flg_max: float | None = provenance_field(
        Provenance.DERIVED,
        "Maximum of the device flow-limitation channel (0–1 unitless).",
        default=None,
    )
    device_flg_reason: str | None = None
    snore_median: float | None = provenance_field(
        Provenance.DERIVED,
        "Median of the device snore channel (0–5 unitless).",
        default=None,
    )
    snore_95th: float | None = provenance_field(
        Provenance.DERIVED,
        "95th percentile of the device snore channel (0–5 unitless).",
        default=None,
    )
    snore_pct_time: float | None = provenance_field(
        Provenance.DERIVED,
        "Fraction of snore-channel samples (0–1) above 0.5.",
        default=None,
    )
    snore_reason: str | None = None

    periodic_breathing_pct: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Percent of analyzed (OK-session) time in periodic-breathing episodes "
        "found by analysis; 0.0 when detection ran and found none.",
        default=None,
    )
    pb_reason: str | None = None

    device_id: int | None = None


class ComplianceFields(BaseModel):
    """Compliance summary appended to range-mode nightly summary."""

    model_config = ConfigDict(populate_by_name=True)

    threshold_hours: float
    days_compliant: int = provenance_field(
        Provenance.DERIVED, "Calendar nights with usage at or above threshold_hours."
    )
    days_total: int = provenance_field(
        Provenance.DERIVED, "Calendar nights in the requested range."
    )
    compliance_pct: float = provenance_field(
        Provenance.DERIVED, "days_compliant as a percent of days_total."
    )


class NightlySummaryResponse(BaseModel):
    """Response from get_nightly_summary."""

    model_config = ConfigDict(populate_by_name=True)

    nights: list[NightlyRow]
    total_nights: int
    page: int
    page_size: int
    # Compliance block only present in range mode
    compliance: ComplianceFields | None = None
    device_capabilities: DeviceCapabilities | None = None


class EventContext(BaseModel):
    """Per-event contextual snapshot."""

    model_config = ConfigDict(populate_by_name=True)

    # Null companions' reasons live on EventRow.  The MV-based fields are
    # derived from the device MV channel, or experimental when mv_source is
    # "flow_derived".
    pressure_at_event_cmh2o: float | None = provenance_field(
        Provenance.DERIVED, "Mean pressure at the event (cmH₂O).", default=None
    )
    leak_at_event_lpm: float | None = provenance_field(
        Provenance.DERIVED, "Mean leak at the event (L/min).", default=None
    )
    mv_prior_120s_lpm: float | None = provenance_field(
        Provenance.DERIVED,
        "Mean minute ventilation over the 120 s before the event (L/min); "
        "experimental when mv_source is flow_derived.",
        source_field="mv_source",
        default=None,
    )
    minutes_since_session_start: float | None = provenance_field(
        Provenance.DEVICE, "Event start relative to session start (min).", default=None
    )
    preceding_mv_slope_lpm_per_min: float | None = provenance_field(
        Provenance.DERIVED,
        "Least-squares MV slope over the 60 s before the event (L/min per min); "
        "experimental when mv_source is flow_derived.",
        source_field="mv_source",
        default=None,
    )
    stability_index: float | None = provenance_field(
        Provenance.DERIVED,
        "MV stdev / mean (CV) over the 60 s before the event; experimental "
        "when mv_source is flow_derived.",
        source_field="mv_source",
        default=None,
    )
    ps_delivered_cmh2o: float | None = provenance_field(
        Provenance.DERIVED,
        "Delivered pressure support: mean(therapy pressure − EPAP) over ±5 s (cmH₂O).",
        default=None,
    )
    # MV provenance: "device" | "flow_derived" | null (no MV or flow channel)
    mv_source: str | None = None


class EventRow(BaseModel):
    """A single respiratory event with inline context.

    Timestamp contract (A6):
    - ``start_time_wall_clock``: device wall-clock; offset-qualified when
      ``timezone_status == "user_declared"``, else naive (tier 2).
    - ``session_start_wall_clock``: per-event session anchor; offset-qualified when
      ``timezone_status == "user_declared"``, else naive (tier 2).
    - ``timezone_status``: ``"unknown"``, or ``"user_declared"`` when the profile
      declares an IANA timezone (carried in ``timezone_name``).
    - ``offset_seconds``: position from this event's session start (tier 3).
    """

    model_config = ConfigDict(populate_by_name=True)

    session_id: int  # session that produced this event (per-event anchor)
    session_start_wall_clock: str  # ISO 8601 wall-clock; offset-qualified when timezone_status == "user_declared", else naive (tier 2)
    event_type: str
    start_time_wall_clock: str  # ISO 8601 wall-clock; offset-qualified when timezone_status == "user_declared", else naive (tier 2)
    timezone_status: str = "unknown"  # "unknown" | "user_declared"
    timezone_name: str | None = None  # IANA name when user_declared
    offset_seconds: float  # seconds from this event's Session.start_time (tier 3)
    duration_seconds: float | None = provenance_field(
        Provenance.DEVICE, "Device-scored event duration (s).", default=None
    )
    spo2_drop_pct: float | None = provenance_field(
        Provenance.DEVICE,
        "SpO₂ drop during the event (%), from the events table; null when the "
        "source did not record it (no parser fills it today).",
        default=None,
    )
    peak_flow_limitation: float | None = provenance_field(
        Provenance.DEVICE,
        "Peak device flow-limitation value during the event, from the events "
        "table; null when the source did not record it (no parser fills it today).",
        default=None,
    )
    pressure_reason: str | None = None
    leak_reason: str | None = None
    mv_reason: str | None = None
    preceding_mv_slope_reason: str | None = None
    stability_reason: str | None = None
    ps_reason: str | None = None
    context: EventContext | None = None


class EventsResponse(BaseModel):
    """Response from get_events.

    ``session_id`` and ``session_start_wall_clock`` are the response-level anchors.
    They are null when no events were returned or when events span multiple sessions.
    Per-event anchors (``EventRow.session_id`` and ``EventRow.session_start_wall_clock``)
    are always populated on individual events.
    """

    model_config = ConfigDict(populate_by_name=True)

    date: str
    session_id: int | None = None  # null when empty or multi-session
    session_start_wall_clock: str | None = None  # null when empty or multi-session
    timezone_status: str = "unknown"
    timezone_name: str | None = None  # IANA name when user_declared
    events: list[EventRow]
    total_events: int
    truncated: bool = False
    # Version of the query-time flow-derived MV fallback (MV_FALLBACK_ALGO_VERSION);
    # non-null only when at least one returned event has mv_source "flow_derived".
    mv_fallback_version: str | None = None
    device_capabilities: DeviceCapabilities | None = None


class CapabilityEntry(BaseModel):
    """One entry in the capabilities resource."""

    model_config = ConfigDict(populate_by_name=True)

    channel: str
    description: str
    unit: str | None = None
    present_in_dataset: bool
    sample_rate_hz: float | None = provenance_field(
        Provenance.DEVICE, "Channel sample rate (Hz).", default=None
    )


# ---------------------------------------------------------------------------
# Stage-2 schemas: get_breath_table, find_windows, compare_epochs
# ---------------------------------------------------------------------------


class BreathTableQuery(BaseModel):
    """Echo of the breath-table query as resolved by the service."""

    model_config = ConfigDict(populate_by_name=True)

    therapy_date: str
    device_id: int | None = None
    session_id: int | None = None
    offset_start: float
    offset_end: float
    page: int
    page_size: int
    bin_minutes: float | None = None


class BreathTableRow(BaseModel):
    """One analyzed breath (tier-2 wall-clock anchor + tier-3 offsets).

    Nullable measurement fields carry companion ``*_reason`` fields where the
    service provides them; absence of a value is never coerced to zero.
    """

    model_config = ConfigDict(populate_by_name=True)

    analysis_result_id: int
    session_id: int
    breath_number: int
    session_start_wall_clock: str
    timezone_status: str = "unknown"
    timezone_name: str | None = None  # IANA name when user_declared
    start_offset_seconds: float
    end_offset_seconds: float
    # Every measurement below comes from SNORE breath segmentation and
    # classification of the flow waveform.
    ti_s: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Inspiratory time (s).", default=None
    )
    te_s: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Expiratory time (s).", default=None
    )
    ttot_s: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Total breath time (s).", default=None
    )
    ie_ratio: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Inspiratory:expiratory time ratio.", default=None
    )
    duty_cycle: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Ti / Ttot.", default=None
    )
    peak_insp_flow_lpm: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Peak inspiratory flow (L/min).", default=None
    )
    peak_exp_flow_lpm: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Peak expiratory flow (L/min).", default=None
    )
    tidal_volume_ml: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Tidal volume integrated from the flow waveform (mL).",
        default=None,
    )
    flatness_index: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Inspiratory flatness index.", default=None
    )
    mid_insp_flattening: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Mid-inspiratory flattening score.", default=None
    )
    flow_class: int | None = provenance_field(
        Provenance.EXPERIMENTAL, "Flow-limitation class (1–7).", default=None
    )
    flow_class_confidence: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Classifier confidence for flow_class.", default=None
    )
    is_recovery_breath: bool | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Analysis-time recovery-breath flag (ends a RERA candidate).",
        default=None,
    )
    trigger_type: str | None = provenance_field(
        Provenance.EXPERIMENTAL, "Inferred trigger type (heuristic).", default=None
    )
    cycle_type: str | None = provenance_field(
        Provenance.EXPERIMENTAL, "Inferred cycle type (heuristic).", default=None
    )
    trigger_cycle_confidence: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Confidence of the trigger/cycle inference.",
        default=None,
    )
    trigger_cycle_applicability: str | None = None
    trigger_cycle_reason: str | None = None
    leak_valid: bool | None = provenance_field(
        Provenance.EXPERIMENTAL, "Breath is free of excess leak.", default=None
    )
    leak_valid_reason: str | None = None
    ramp_active: bool | None = provenance_field(
        Provenance.EXPERIMENTAL, "Breath falls in the pressure ramp.", default=None
    )
    ramp_active_reason: str | None = None
    mask_off: bool | None = provenance_field(
        Provenance.EXPERIMENTAL, "Breath falls in a mask-off period.", default=None
    )
    mask_off_reason: str | None = None


class BreathTableBin(BaseModel):
    """Aggregated breath metrics for one time bin."""

    model_config = ConfigDict(populate_by_name=True)

    session_start_wall_clock: str
    timezone_status: str = "unknown"
    timezone_name: str | None = None  # IANA name when user_declared
    bin_start_offset: float
    bin_end_offset: float
    breath_count: int = provenance_field(
        Provenance.EXPERIMENTAL, "SNORE-segmented breaths in the bin."
    )
    flatness_index_median: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Median flatness index in the bin.", default=None
    )
    mid_insp_flattening_median: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Median mid-inspiratory flattening in the bin.",
        default=None,
    )
    flow_class_mode: int | None = provenance_field(
        Provenance.EXPERIMENTAL, "Most common flow class in the bin.", default=None
    )
    tidal_volume_median_ml: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Median tidal volume in the bin (mL).", default=None
    )
    ie_ratio_median: float | None = provenance_field(
        Provenance.EXPERIMENTAL, "Median I:E ratio in the bin.", default=None
    )
    leak_valid_fraction: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Fraction of breaths in the bin flagged leak-valid.",
        default=None,
    )
    analysis_status: str


class BreathTableResponse(BaseModel):
    """Response from get_breath_table. Exactly one of rows/bins is populated."""

    model_config = ConfigDict(populate_by_name=True)

    query: BreathTableQuery
    session_id: int | None = None
    session_start_wall_clock: str | None = None
    timezone_status: str = "unknown"
    timezone_name: str | None = None  # IANA name when user_declared
    analysis_status: str
    algo_versions: dict[str, Any] | None = None
    null_reason: str | None = None
    is_binned: bool
    total_breaths: int = provenance_field(
        Provenance.EXPERIMENTAL, "SNORE-segmented breaths matching the query."
    )
    page: int
    page_size: int
    rows: list[BreathTableRow] = []
    bins: list[BreathTableBin] = []
    device_capabilities: DeviceCapabilities | None = None


class SessionCoverageEntry(BaseModel):
    """Per-session analysis coverage for a queried day."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: int
    analysis_status: str
    algo_versions: dict[str, Any] | None = None


class WindowRow(BaseModel):
    """One found window (ordering per criterion; see find_windows)."""

    model_config = ConfigDict(populate_by_name=True)

    criterion: str
    session_id: int
    session_start_wall_clock: str
    timezone_status: str = "unknown"
    timezone_name: str | None = None  # IANA name when user_declared
    window_start_offset: float
    window_end_offset: float
    reason_summary: str
    worst_mid_insp_flattening: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Lowest (most flow-limited) mid-inspiratory flattening in the window's "
        "anchor breath or FL run.",
        default=None,
    )
    fl_run_length: int | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Length (breaths) of the flow-limitation run.",
        default=None,
    )
    anchor_event_offset: float | None = None
    analysis_result_id: int | None = None
    analysis_status: str
    analysis_reason: str | None = None


class FindWindowsResponse(BaseModel):
    """Response from find_windows."""

    model_config = ConfigDict(populate_by_name=True)

    query_date: str
    device_id: int | None = None
    criterion: str
    day_status: str
    session_coverage: list[SessionCoverageEntry] = []
    algorithm_identity: dict[str, Any] | None = None
    null_reason: str | None = None
    primary_mode: str | None = None
    windows: list[WindowRow] = []
    device_capabilities: DeviceCapabilities | None = None


class EpochSpec(BaseModel):
    """Input epoch for compare_epochs (dates are YYYY-MM-DD strings)."""

    model_config = ConfigDict(populate_by_name=True)

    label: str
    date_start: str
    date_end: str
    device_id: int | None = None


class EpochDistribution(BaseModel):
    """Descriptive stats for one metric over one epoch (leak-valid breaths only).

    The stats are derived; the containing ``EpochStats`` field's tier applies
    when it is less certain (experimental for breath features).
    """

    model_config = ConfigDict(populate_by_name=True)

    median: float | None = provenance_field(Provenance.DERIVED, "Median.", default=None)
    iqr: float | None = provenance_field(
        Provenance.DERIVED, "Interquartile range.", default=None
    )
    p95: float | None = provenance_field(
        Provenance.DERIVED, "95th percentile.", default=None
    )
    n_breaths: int = provenance_field(
        Provenance.DERIVED,
        "Count of values in the distribution (breaths, or device-channel "
        "samples for device_flg / snore_dist).",
    )
    n_nights: int


class EpochRxViolationRow(BaseModel):
    """A therapy-settings change detected inside one epoch's date range."""

    model_config = ConfigDict(populate_by_name=True)

    epoch_label: str
    changed_keys: list[str] = []
    change_dates: list[str] = []


class EpochStats(BaseModel):
    """Breath-feature distributions for one epoch."""

    model_config = ConfigDict(populate_by_name=True)

    label: str
    date_start: str
    date_end: str
    nights_with_data: int
    nights_missing_analysis: int
    algorithm_identity: dict[str, Any] | None = None
    null_reason: str | None = None
    primary_mode: str | None = None
    mid_insp_flattening: EpochDistribution = provenance_field(
        Provenance.EXPERIMENTAL, "Per-breath mid-inspiratory flattening."
    )
    flatness_index: EpochDistribution = provenance_field(
        Provenance.EXPERIMENTAL, "Per-breath flatness index."
    )
    # Keys are strings because JSON object keys are always strings.
    flow_class_distribution: dict[str, int] = provenance_field(
        Provenance.EXPERIMENTAL,
        "Rule-matched FL classifications only; the class>=4 fraction "
        "reconciles with nightly fl_class_ge4_pct.",
        default_factory=dict,
    )
    flow_class_distribution_fallback: dict[str, int] = provenance_field(
        Provenance.EXPERIMENTAL,
        "Low-confidence fallback flatness-triage guesses (confidence at the "
        "default), reported separately from flow_class_distribution so they "
        "don't inflate FL rates. Missing or below-default confidence values "
        "are excluded from both distributions.",
        default_factory=dict,
    )
    tidal_volume_ml: EpochDistribution = provenance_field(
        Provenance.EXPERIMENTAL, "Per-breath tidal volume integrated from flow (mL)."
    )
    ie_ratio: EpochDistribution = provenance_field(
        Provenance.EXPERIMENTAL, "Per-breath I:E ratio."
    )
    rera_proxy_count: int | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Count from the query-time FL-run proxy: runs of flow_class >= 4 "
        "ending in a recovery breath.",
        default=None,
    )
    rera_reason: str | None = None
    # Version of the query-time RERA-proxy criterion (independent of the
    # persisted AlgorithmIdentity); stamped only when the RERA scan ran
    # (rera_proxy_count non-null), null otherwise.
    rera_proxy_version: str | None = None
    rx_settings: dict[str, str] = {}
    # n_breaths carries the sample count; null stats when the channel was not
    # recorded in the epoch's sessions.
    device_flg: EpochDistribution = provenance_field(
        Provenance.DERIVED,
        "Device flow-limitation channel samples (0–1 unitless).",
        default_factory=lambda: EpochDistribution(n_breaths=0, n_nights=0),
    )
    snore_dist: EpochDistribution = provenance_field(
        Provenance.DERIVED,
        "Device snore channel samples (0–5 unitless).",
        default_factory=lambda: EpochDistribution(n_breaths=0, n_nights=0),
    )


class CompareEpochsResponse(BaseModel):
    """Response from compare_epochs."""

    model_config = ConfigDict(populate_by_name=True)

    epochs: list[EpochStats] = []
    null_reason: str | None = None
    rx_violations: list[EpochRxViolationRow] = []
    # Per-field warnings when algorithm identity fields differ across epochs; also
    # includes a warning when rera_proxy_version differs across epochs.
    # Non-empty means distributions were computed across sessions with different
    # algorithm versions — callers should review before drawing conclusions.
    version_warnings: list[str] = []


# ---------------------------------------------------------------------------
# Stage-3 schemas: get_waveform
# ---------------------------------------------------------------------------


class WaveformChannelSchema(BaseModel):
    """One deserialized, windowed waveform channel returned by get_waveform."""

    model_config = ConfigDict(populate_by_name=True)

    channel_type: str
    unit: str | None = None
    sample_rate_hz: float = provenance_field(
        Provenance.DEVICE, "Channel sample rate (Hz)."
    )
    offset_seconds: list[float]  # tier-3 positions from session start
    values: list[float] = provenance_field(
        Provenance.DEVICE,
        "Device samples; LTTB keeps a subset of them when is_downsampled.",
    )
    original_sample_count: int  # pre-LTTB count within the window
    is_downsampled: bool


class WaveformWindowResponse(BaseModel):
    """Response from get_waveform."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: int | None = None  # null when no session on the date
    session_start_wall_clock: str | None = (
        None  # ISO 8601 wall-clock; offset-qualified when timezone_status == "user_declared", else naive (tier 2); null when session_id null
    )
    timezone_status: str = "unknown"
    timezone_name: str | None = None  # IANA name when user_declared
    window_start_offset_s: float
    window_end_offset_s: float
    channels: list[WaveformChannelSchema]
    missing_channels: list[str]
    missing_channel_reason: str | None = None


# Mapping used for docs://schemas/{type} — maps schema name to model class
SCHEMA_MODEL_MAP: dict[str, type[BaseModel]] = {
    "device_capabilities": DeviceCapabilities,
    "device_info": DeviceInfo,
    "data_overview": DataOverviewResponse,
    "settings_epoch": SettingsEpoch,
    "settings_timeline": SettingsTimelineResponse,
    "settings_change_entry": SettingsChangeEntry,
    "settings_changes": SettingsChangesResponse,
    "nightly_row": NightlyRow,
    "compliance_fields": ComplianceFields,
    "nightly_summary": NightlySummaryResponse,
    "event_context": EventContext,
    "event_row": EventRow,
    "events_response": EventsResponse,
    "capability_entry": CapabilityEntry,
    # Stage 2
    "breath_table_query": BreathTableQuery,
    "breath_table_row": BreathTableRow,
    "breath_table_bin": BreathTableBin,
    "breath_table_response": BreathTableResponse,
    "window_row": WindowRow,
    "session_coverage_entry": SessionCoverageEntry,
    "find_windows_response": FindWindowsResponse,
    "epoch_spec": EpochSpec,
    "epoch_distribution": EpochDistribution,
    "epoch_stats": EpochStats,
    "epoch_rx_violation": EpochRxViolationRow,
    "compare_epochs_response": CompareEpochsResponse,
    # Stage 3
    "waveform_channel": WaveformChannelSchema,
    "waveform_window": WaveformWindowResponse,
}


def model_to_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Return the JSON schema for a Pydantic model."""
    return model.model_json_schema()
