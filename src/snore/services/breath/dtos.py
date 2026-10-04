"""DTOs, enums, and exception classes for the breath service package."""

from __future__ import annotations

import math

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from snore.analysis.shared.versioning import (
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisStatus,
    DayAnalysisStatus,
    NullReason,
    TimezoneStatus,
)
from snore.constants import RERAProxyConstants

# ---------------------------------------------------------------------------
# Analysis-status/versioning types re-exported from versioning (single source of truth)
# ---------------------------------------------------------------------------


class SessionCoverage(BaseModel):
    """Per-session analysis coverage entry."""

    session_id: int
    analysis_status: AnalysisStatus
    algo_versions: AlgoVersions | None


# ---------------------------------------------------------------------------
# §4 — TriggerType / CycleType / TriggerCycleApplicability
# ---------------------------------------------------------------------------


class TriggerType(StrEnum):
    NORMAL = "normal"
    PREMATURE = "premature"
    DELAYED = "delayed"


class CycleType(StrEnum):
    NORMAL = "normal"
    PREMATURE = "premature"


class TriggerCycleApplicability(StrEnum):
    VALIDATED = "validated"
    UNVALIDATED_DEVICE = "unvalidated_device"


# ---------------------------------------------------------------------------
# §2 — BreathQueryRange + SessionSummary + MultiSessionAmbiguityError
# ---------------------------------------------------------------------------


class SessionSummary(BaseModel):
    session_id: int
    start_wall_clock: datetime  # naive — tier-2 device wall-clock
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN
    timezone_name: str | None = None  # IANA name when USER_DECLARED
    duration_seconds: float


class MultiSessionAmbiguityError(Exception):
    """Raised when session_id is required but not supplied for a multi-session day."""

    def __init__(
        self,
        therapy_date: date,
        device_id: int,
        sessions: list[SessionSummary],
    ) -> None:
        self.therapy_date = therapy_date
        self.device_id = device_id
        self.sessions = sessions
        super().__init__(
            f"Multiple sessions on {therapy_date}: pass session_id to disambiguate"
        )


class DeviceNotOwnedError(Exception):
    """Raised when an explicit device_id is not owned by the requesting profile."""

    def __init__(self, device_id: int, profile_id: int) -> None:
        self.device_id = device_id
        self.profile_id = profile_id
        super().__init__(
            f"device_id={device_id} not found or not owned by profile {profile_id}"
        )


class DeviceAmbiguityError(Exception):
    """Raised when device_id is required but not supplied for a multi-device day.

    Contains only owned device IDs so that cross-profile leakage is impossible.
    """

    def __init__(
        self,
        therapy_date: date,
        profile_id: int,
        owned_device_ids: list[int],
        device_serials: dict[int, str],
    ) -> None:
        self.therapy_date = therapy_date
        self.profile_id = profile_id
        self.owned_device_ids = owned_device_ids
        self.device_serials = device_serials
        super().__init__(
            f"Multiple devices have sessions on {therapy_date}: "
            "pass device_id to disambiguate"
        )


class NoSessionsInRangeError(ValueError):
    """Raised by _resolve_range when no owned sessions exist in the queried date range."""

    def __init__(self, date_start: date, date_end: date) -> None:
        self.date_start = date_start
        self.date_end = date_end
        super().__init__(f"No sessions found in range {date_start} to {date_end}")


class BreathQueryRange(BaseModel):
    """Identifies a contiguous waveform window within a therapy day."""

    therapy_date: date
    device_id: int | None = None
    session_id: int | None = None
    offset_start: float = Field(ge=0.0)
    offset_end: float = Field(gt=0.0)

    # Pagination / binning
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=500, ge=1, le=2000)
    bin_minutes: float | None = Field(default=None, ge=1.0)

    @model_validator(mode="after")
    def validate_window(self) -> BreathQueryRange:
        if not (math.isfinite(self.offset_start) and math.isfinite(self.offset_end)):
            raise ValueError("offset_start and offset_end must be finite")
        if self.offset_end <= self.offset_start:
            raise ValueError("offset_end must be > offset_start")
        window_minutes = (self.offset_end - self.offset_start) / 60
        if self.bin_minutes is None and window_minutes > 15:
            raise ValueError(
                f"Raw window {window_minutes:.1f} min exceeds 15-min cap; "
                "set bin_minutes to aggregate"
            )
        return self


# ---------------------------------------------------------------------------
# §4 — BreathRow
# ---------------------------------------------------------------------------


class BreathRow(BaseModel):
    """One row from the breaths table.

    Nullable fields reflect absent measurements — never coerced to zero or
    a default class.  Consumers must check for ``None`` before using
    timing/amplitude/shape values.
    """

    analysis_result_id: int
    session_id: int
    breath_number: int

    session_start_wall_clock: datetime  # naive — tier-2
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN
    timezone_name: str | None = None  # IANA name when USER_DECLARED
    start_offset_seconds: float
    end_offset_seconds: float

    # Timing (None when the segmenter could not resolve them)
    ti: float | None  # inspiration time s
    te: float | None  # expiration time s
    ttot: float | None  # total time s
    ie_ratio: float | None
    duty_cycle: float | None

    # Amplitude (None when channel absent or computation failed)
    peak_insp_flow: float | None  # L/min
    peak_exp_flow: float | None  # L/min
    tidal_volume: float | None  # mL

    # Flow limitation features (None when channel absent)
    flatness_index: float | None
    mid_insp_flattening: float | None

    # Classification (None when not computed)
    flow_class: int | None
    flow_class_confidence: float | None
    is_recovery_breath: bool | None

    # Trigger/cycle heuristic (experimental)
    trigger_type: TriggerType | None
    cycle_type: CycleType | None
    trigger_cycle_confidence: float | None
    trigger_cycle_applicability: TriggerCycleApplicability | None
    trigger_cycle_reason: NullReason | None

    # Quality flags
    leak_valid: bool | None
    leak_valid_reason: NullReason | None
    ramp_active: bool | None
    ramp_active_reason: NullReason | None
    mask_off: bool | None
    mask_off_reason: NullReason | None


# ---------------------------------------------------------------------------
# §5 — BreathPage
# ---------------------------------------------------------------------------


class BreathBin(BaseModel):
    """Aggregated metrics for one time bin."""

    session_start_wall_clock: datetime  # naive — tier-2 anchor
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN
    timezone_name: str | None = None  # IANA name when USER_DECLARED
    bin_start_offset: float
    bin_end_offset: float
    breath_count: int
    flatness_index_median: float | None
    mid_insp_flattening_median: float | None
    flow_class_mode: int | None
    tidal_volume_median: float | None
    ie_ratio_median: float | None
    leak_valid_fraction: float | None
    analysis_status: AnalysisStatus


class BreathPage(BaseModel):
    """Result of get_breath_table()."""

    query: BreathQueryRange
    analysis_status: AnalysisStatus
    algo_versions: AlgoVersions | None
    null_reason: NullReason | None
    is_binned: bool
    total_breaths: int
    page: int
    page_size: int
    rows: list[BreathRow] = Field(default_factory=list)
    bins: list[BreathBin] = Field(default_factory=list)
    session_id: int | None = None
    """Resolved session for this page; None only on legacy constructions."""


# ---------------------------------------------------------------------------
# §6 — WindowCriterion / WindowCriterionOptions / FindWindowsResult
# ---------------------------------------------------------------------------


class WindowCriterion(StrEnum):
    WORST_FLATTENING_LEAK_VALID = "worst_flattening_leak_valid"
    CA_CENTERED = "ca_centered"
    FL_RUN_ENDING_IN_RECOVERY = "fl_run_ending_in_recovery"
    RERA_PROXY_CENTERED = "rera_proxy_centered"


class WindowCriterionOptions(BaseModel):
    """Criterion-specific options."""

    include_unknown_leak: bool = False
    # Maximum anchor mid-insp flattening (lower = more flow-limited); None = no cutoff.
    flattening_threshold: float | None = None
    min_window_breaths: int = 3
    context_breaths_before: int = Field(default=3, ge=0)
    context_breaths_after: int = Field(default=3, ge=0)
    context_seconds: float = 120.0
    min_fl_run_length: int = RERAProxyConstants.MIN_FL_RUN_LENGTH
    fl_class_threshold: int = RERAProxyConstants.FL_CLASS_THRESHOLD
    recovery_amplitude_margin: float = Field(
        default=RERAProxyConstants.RECOVERY_AMPLITUDE_MARGIN, ge=0.0
    )


class WindowResult(BaseModel):
    """One found window."""

    criterion: WindowCriterion
    session_id: int
    session_start_wall_clock: datetime
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN
    timezone_name: str | None = None  # IANA name when USER_DECLARED
    window_start_offset: float
    window_end_offset: float
    reason_summary: str
    worst_mid_insp_flattening: float | None
    fl_run_length: int | None
    anchor_event_offset: float | None
    analysis_result_id: int | None
    analysis_status: AnalysisStatus
    analysis_reason: NullReason | None


class FindWindowsResult(BaseModel):
    """Result of find_windows()."""

    query_date: date
    device_id: int
    criterion: WindowCriterion
    day_status: DayAnalysisStatus
    session_coverage: list[SessionCoverage] = Field(default_factory=list)
    algorithm_identity: AlgorithmIdentity | None
    null_reason: NullReason | None
    primary_mode: str | None
    windows: list[WindowResult]


# ---------------------------------------------------------------------------
# §7 — Epoch DTOs
# ---------------------------------------------------------------------------


class EpochRequest(BaseModel):
    """One settings epoch for comparison."""

    label: str
    date_start: date
    date_end: date
    device_id: int | None = None


class DistributionMetric(StrEnum):
    MID_INSP_FLATTENING = "mid_insp_flattening"
    FLATNESS_INDEX = "flatness_index"
    TIDAL_VOLUME_ML = "tidal_volume_ml"
    IE_RATIO = "ie_ratio"
    DEVICE_FLG = "device_flg"
    SNORE = "snore"


class DistributionStats(BaseModel):
    """Descriptive stats for one metric over one epoch."""

    median: float | None
    iqr: float | None
    p95: float | None
    n_breaths: int
    n_nights: int


class EpochRxViolation(BaseModel):
    epoch_label: str
    changed_keys: list[str]
    change_dates: list[date]


class EpochBreathStats(BaseModel):
    """Breath-feature distributions for one epoch."""

    label: str
    date_start: date
    date_end: date
    nights_with_data: int
    nights_missing_analysis: int
    algorithm_identity: AlgorithmIdentity | None
    null_reason: NullReason | None
    primary_mode: str | None
    mid_insp_flattening: DistributionStats
    flatness_index: DistributionStats
    # Rule-matched FL classifications only (flow_confidence > FL_DEFAULT_CONFIDENCE),
    # so the class>=4 fraction reconciles with nightly fl_class_ge4_pct.
    flow_class_distribution: dict[int, int]
    # Low-confidence fallback flatness-triage guesses (flow_confidence exactly at
    # FL_DEFAULT_CONFIDENCE); reported separately so they don't inflate FL rates.
    # Missing or below-default confidence values are excluded from both buckets.
    flow_class_distribution_fallback: dict[int, int] = Field(default_factory=dict)
    tidal_volume_ml: DistributionStats
    ie_ratio: DistributionStats
    rera_proxy_count: int | None
    rera_reason: NullReason | None
    # Version of the query-time RERA-proxy criterion (not part of
    # AlgorithmIdentity — see RERA_PROXY_ALGO_VERSION).  Stamped only when the
    # RERA scan actually ran (rera_proxy_count is non-null); None otherwise.
    rera_proxy_version: str | None = None
    rx_settings: dict[str, str]
    # Device waveform channel distributions (samples, not breaths; n_breaths
    # field carries sample count).  Null when the channel was not recorded.
    device_flg: DistributionStats = Field(
        default_factory=lambda: DistributionStats(
            median=None, iqr=None, p95=None, n_breaths=0, n_nights=0
        )
    )
    snore_dist: DistributionStats = Field(
        default_factory=lambda: DistributionStats(
            median=None, iqr=None, p95=None, n_breaths=0, n_nights=0
        )
    )


class CompareEpochsResult(BaseModel):
    """Result of compare_epochs()."""

    epochs: list[EpochBreathStats]
    null_reason: NullReason | None
    rx_violations: list[EpochRxViolation] = Field(default_factory=list)
    # Per-field warnings when algorithm identity fields in CROSS_VERSION_REFUSAL_KEYS
    # differ across contributing sessions (previously a hard refusal, now a warning).
    # Also includes a warning when rera_proxy_version differs across epochs.
    version_warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# §8 — ContextualEvent
# ---------------------------------------------------------------------------


class MvSource(StrEnum):
    """Provenance of the MV series behind per-event ventilatory metrics."""

    DEVICE = "device"
    FLOW_DERIVED = "flow_derived"


class VentilatoryContext(BaseModel):
    """Ventilatory-control metrics around one event (see
    ``compute_ventilatory_context``, which pairs every null value with
    ``NOT_AVAILABLE``).  The all-null default (reasons included) is the
    "context not requested" value."""

    # Linear-regression MV slope over the 60 s preceding the event.
    preceding_mv_slope_lpm_per_min: float | None = None
    preceding_mv_slope_reason: NullReason | None = None
    # Coefficient of variation (stdev / mean) of MV over the same 60 s window.
    stability_index: float | None = None
    stability_reason: NullReason | None = None
    # mean(THERAPY_PRESSURE) − mean(EPAP) over ±5 s around the event start.
    ps_delivered_cmh2o: float | None = None
    ps_reason: NullReason | None = None


class ContextualEvent(VentilatoryContext):
    """One machine-flagged event with surrounding context."""

    session_id: int
    session_start_wall_clock: datetime  # naive — tier-2
    event_type: str
    event_start_wall_clock: datetime  # naive — tier-2
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN
    timezone_name: str | None = None  # IANA name when USER_DECLARED
    offset_seconds: float
    duration_seconds: float | None

    pressure_at_event_cmh2o: float | None
    pressure_reason: NullReason | None
    leak_at_event_lpm: float | None
    leak_reason: NullReason | None
    mv_prior_120s_lpm: float | None
    mv_reason: NullReason | None
    minutes_since_session_start: float
    # Provenance of the MV series behind mv_prior_120s_lpm and the MV-derived
    # ventilatory metrics; None when neither device MV nor flow was usable.
    mv_source: MvSource | None = None


# ---------------------------------------------------------------------------
# §9 — Multi-channel waveform-window DTOs + split seam functions
# ---------------------------------------------------------------------------


class WaveformChannelName(StrEnum):
    """Typed channel names (persisted Waveform.waveform_type values)."""

    FLOW = "flow"
    PRESSURE = "pressure"
    THERAPY_PRESSURE = "therapy_pressure"
    EPAP = "epap"
    LEAK = "leak"
    MV = "mv"
    RR = "rr"
    TV = "tv"
    SPO2 = "spo2"
    PULSE = "pulse"
    FL = "fl"
    SNORE = "snore"


class RawWaveformChannel(BaseModel):
    waveform_type: WaveformChannelName
    unit: str | None
    sample_rate: float
    sample_count: int
    raw_bytes: bytes


class RawWaveformWindow(BaseModel):
    request: WaveformWindowRequest
    session_id: int
    session_start_wall_clock: datetime  # naive — tier-2 anchor
    # Set at fetch time; compute_waveform_window copies these into the output.
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN
    timezone_name: str | None = None  # IANA name when USER_DECLARED
    channels: list[RawWaveformChannel]
    missing_channels: list[WaveformChannelName]


class WaveformChannel(BaseModel):
    """One deserialized, windowed waveform channel."""

    channel_type: WaveformChannelName
    unit: str | None
    sample_rate: float
    offset_seconds: list[float]
    values: list[float]
    original_sample_count: int
    is_downsampled: bool


class WaveformWindow(BaseModel):
    """Multi-channel waveform data for a time window."""

    session_id: int
    session_start_wall_clock: datetime  # naive — tier-2 anchor
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN
    timezone_name: str | None = None  # IANA name when USER_DECLARED
    window_start_offset: float
    window_end_offset: float
    channels: list[WaveformChannel]
    missing_channels: list[WaveformChannelName]
    missing_channel_reason: NullReason | None


# LTTB target-point bounds for waveform windows; shared with the MCP tool schemas.
MAX_POINTS_MIN = 2
MAX_POINTS_MAX = 1000


class WaveformWindowRequest(BaseModel):
    """Request for a multi-channel waveform window."""

    therapy_date: date
    device_id: int | None = None
    session_id: int | None = None
    offset_start: float = Field(ge=0.0)
    offset_end: float = Field(gt=0.0)
    channels: list[WaveformChannelName] = Field(
        default_factory=list,
        max_length=len(WaveformChannelName),
    )
    max_points: int | None = Field(default=None, ge=MAX_POINTS_MIN, le=MAX_POINTS_MAX)
    window_cap_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def validate_request(self) -> WaveformWindowRequest:
        if self.offset_end <= self.offset_start:
            raise ValueError("offset_end must be > offset_start")
        window_seconds = self.offset_end - self.offset_start
        if window_seconds > self.window_cap_seconds:
            raise ValueError(
                f"Window {window_seconds:.0f} s exceeds the {self.window_cap_seconds:.0f} s "
                "cap for this tool; narrow the window"
            )
        if not self.channels:
            self.channels = [
                WaveformChannelName.FLOW,
                WaveformChannelName.PRESSURE,
                WaveformChannelName.LEAK,
            ]
        # Dedup, order-preserving
        self.channels = list(dict.fromkeys(self.channels))
        return self


# ---------------------------------------------------------------------------
# §10 — Nightly-aggregation DTOs
# ---------------------------------------------------------------------------


class NightlyAnalysisSummary(BaseModel):
    """Latest-analysis-run fields for one therapy night."""

    therapy_date: date
    device_id: int

    day_status: DayAnalysisStatus
    session_coverage: list[SessionCoverage]
    eligible_session_count: int
    analyzed_session_count: int
    missing_or_stale_session_ids: list[int] = Field(default_factory=list)
    algorithm_identity: AlgorithmIdentity | None

    # RERA count from the query-time FL-run proxy v2 over stored breath rows:
    # runs of flow_class >= 4 ending in a recovery breath (see
    # _count_fl_run_reras). This is a DIFFERENT RERA definition from the
    # analysis-time amplitude-crescendo detector behind ModeResult.rdi; the two
    # indices disagree by construction.
    rera_count: int | None
    rera_reason: NullReason | None
    # Version of the query-time RERA-proxy criterion (not part of
    # AlgorithmIdentity — see RERA_PROXY_ALGO_VERSION).  Stamped only when the
    # RERA scan actually ran (rera_count is non-null); None otherwise.
    rera_proxy_version: str | None = None
    primary_mode: str | None
    # Over raw mid_insp_flattening (inverse severity): lower = more flow-limited,
    # so the reported tail is the low one (5th percentile and minimum).
    fl_median: float | None
    fl_5th: float | None
    fl_min: float | None
    fl_reason: NullReason | None
    # Percent of leak-valid, rule-matched classified breaths with flow_class >= 4.
    # Both numerator and denominator count only breaths whose flow_confidence
    # exceeds 0.5 (rule-matched, not the 0.5 fallback guess), so fallback class-4
    # guesses do not inflate the FL rate.  Denominator follows fl_median's
    # leak-valid convention.
    fl_class_ge4_pct: float | None
    fl_class_ge4_pct_reason: NullReason | None

    ti_median_s: float | None
    ti_median_reason: NullReason | None
    ie_ratio_median: float | None
    ie_ratio_reason: NullReason | None

    total_therapy_hours: float
    compliance_threshold_hours: float
    is_compliant: bool

    # rera_index = rera_count (FL-run proxy v2) per mask-on hour of analyzed
    # sessions (DURATION_ZERO when those hours are 0); rdi = day AHI + rera_index. RERAs come from the query-time FL-run proxy, NOT the
    # analysis-time amplitude-crescendo detector behind ModeResult.rdi, so this
    # nightly rdi and the per-session ModeResult.rdi disagree by construction.
    rera_index: float | None = None
    rera_index_reason: NullReason | None = None
    rdi: float | None = None
    rdi_reason: NullReason | None = None
    # Percent of breaths where leak_valid is False (i.e. leak > 24 L/min).
    # Denominator: breaths where leak_valid is not None (excludes indeterminate).
    # None when no breath has a determinate leak_valid value.
    leak_above_24_pct: float | None = None
    leak_above_24_pct_reason: NullReason | None = None

    # Device-recorded flow-limitation channel (raw FL waveform "fl", 0–1 unitless).
    # Negative sentinel values (−0.01 from digital −1 at mask-off) are filtered before
    # aggregation; zeros are legitimate data.
    # None + reason when the channel was not recorded for this night.
    device_flg_median: float | None = None
    device_flg_95th: float | None = None
    device_flg_max: float | None = None
    device_flg_reason: NullReason | None = None

    # Device-recorded snore channel (raw "snore", 0–5 unitless).
    # snore_pct_time: fraction of samples (0–1) where snore > 0.5.
    # None + reason when the channel was not recorded.
    snore_median: float | None = None
    snore_95th: float | None = None
    snore_pct_time: float | None = None
    snore_reason: NullReason | None = None

    # Percent of OK-session time in persisted periodic-breathing episodes.
    # 0.0 when PB detection ran and found none; null + reason otherwise
    # (ALGO_VERSION_MISMATCH on mixed-version nights).
    periodic_breathing_pct: float | None = None
    pb_reason: NullReason | None = None


class NightlyRangeSummary(BaseModel):
    """Compliance + analysis summary over a date range."""

    date_start: date
    date_end: date
    device_id: int
    compliance_threshold_hours: float
    n_calendar_nights: int
    n_nights: int
    days_compliant: int
    compliance_pct: float
    nights: list[NightlyAnalysisSummary]


# ---------------------------------------------------------------------------
# §11 — DeviceCapabilities
# ---------------------------------------------------------------------------


class DeviceCapabilities(BaseModel):
    """What data is actually present for a device over a (requested) date range."""

    device_id: int
    requested_date_start: date | None
    requested_date_end: date | None
    actual_date_start: date | None
    actual_date_end: date | None
    null_reason: NullReason | None
    channels_present: list[str]
    all_setting_keys_present: list[str]
    rx_keys_present: list[str]
    event_types_present: list[str]
    session_count: int
    nights_with_data: int
    supported_vendor_models: list[str]

    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None
