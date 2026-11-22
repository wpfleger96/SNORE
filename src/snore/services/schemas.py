"""Pydantic response schemas for service layer.

These models define the contract between services and consumers (CLI/API).
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from snore.constants import (
    COMPLIANCE_THRESHOLD_HOURS,
    AhiTrendDirection,
    TherapyEffectiveness,
)
from snore.metrics import (
    COVERAGE_TOLERANCE_FRACTION,
    COVERAGE_TOLERANCE_HOURS,
    pooled_percentile_description,
)
from snore.provenance import IndexSource, experimental_field

__all__ = [
    "HEADLINE_INDEX_DESCRIPTION",
    "INDEX_SOURCE_DESCRIPTION",
    "TIME_IN_APNEA_DESCRIPTION",
    "PeriodStatistics",
    "EventValidationResult",
    "DatabaseStats",
    "SessionListItem",
    "SessionListResult",
    "SessionDetail",
    "SessionStatistics",
    "SessionSetting",
    "DeviceChannelStats",
    "MaskOnSpan",
    "LargeLeakSpan",
    "LeakRedlineStats",
    "DeletePreview",
    "TherapySummary",
    "EventTypeCount",
    "WaveformInfo",
    "AnalysisListItem",
    "AnalysisSessionDetail",
    "AnalysisDeletePreview",
    "AnalysisSelection",
    "EMPTY_SELECTION_MESSAGES",
    "DeviceInfo",
    "SettingChangeEntry",
    "SettingsChange",
    "DeviceUsageSummary",
    "DeviceDetail",
    # Mask equipment log schemas (consumed by MaskLogService and API routers)
    "MaskLogEntryResponse",
    "MaskEpochResponse",
    # RX / Day / Event schemas (consumed by RxService, DayService, and API routers)
    "DayListItem",
    "DayDetail",
    "RxPeriodResponse",
    "RxComparisonResponse",
    "RxSettingChange",
    "RxChangesResponse",
    "RxAllResponse",
    "MergedSettingsChange",
    # Import schemas
    "ImportSource",
    "ImportSourceResult",
    "ImportResult",
    # Batch analysis schemas
    "BatchSessionResult",
    "BatchAnalysisResult",
    # Event comparison schemas
    "EventComparisonDetail",
    "EventComparisonResult",
    # Vacuum schema
    "VacuumResult",
    # Reset schema
    "ResetResult",
    # Per-user data deletion schema
    "DeleteDataResult",
    # Stats range schema
    "DataRange",
    # Stats trends/records schemas
    "TrendsResponse",
    "RecordExtremes",
    "RecordsResponse",
    # Apple Health import schema
    "HealthImportResult",
    # Apple Health read schemas
    "HealthNightCoverage",
    "HealthNightSummaryRead",
    "HealthNightDetailRead",
    "HealthSampleRead",
]

# Shared descriptions of the Day headline indices (REST day schemas and the MCP
# nightly summary), built from the trust-rule constants so they cannot drift.
HEADLINE_INDEX_DESCRIPTION = (
    "device-reported daily value when trusted, otherwise SNORE's recount; "
    "index_source says which. Trusted = no session of the day is disabled, "
    "every session reports the same device AHI/OAI/CAI/HI and daily mask-on "
    "hours, both mask-on times are nonzero, and SNORE's imported time is within "
    f"the larger of {COVERAGE_TOLERANCE_HOURS * 60:g} min or "
    f"{COVERAGE_TOLERANCE_FRACTION:.0%} of the device's"
)
# Large Leak spans listed at most per response (about 53 KB of JSON, well under
# the MCP response size limit); a night over it lists its longest.
MAX_LARGE_LEAK_SPANS = 500

COMPLIANT_DAYS_DESCRIPTION = (
    f"Nights with at least {COMPLIANCE_THRESHOLD_HOURS:g} mask-on hours from enabled "
    "sessions, summed across devices"
)
INDEX_SOURCE_DESCRIPTION = (
    "Source of the headline ahi/oai/cai/hi: 'device' (device-reported daily "
    "value) or 'derived' (SNORE's recount); null when the day has no index"
)

# The query-time RERA proxy scans every flow_class >= 4 breath, with no
# confidence gate (REST experimental day metrics and the MCP experimental tools).
RERA_PROXY_FALLBACK_NOTE = (
    "Its FL runs include fallback flatness-triage guesses (classifier "
    "confidence == FL_DEFAULT_CONFIDENCE), which the fl_class_ge4_pct and "
    "flow_class_distribution confidence gate excludes, so the proxy inherits "
    "the fallback share's sensitivity to waveform shape (e.g. across therapy "
    "modes or pressure-support levels)."
)

# What each stored waveform channel measures, for every statistic computed over
# it.  Each statistic pools samples, never averages per session.
_FL_CHANNEL = (
    "device flow-limitation channel (0-1 unitless), excluding mask-off sentinels"
)
_SNORE_CHANNEL = "device snore channel (0-5 unitless)"
_RR_CHANNEL = (
    "device respiratory rate channel (breaths/min), excluding out-of-range samples"
)
_TV_CHANNEL = "device tidal volume channel (mL), excluding out-of-range samples"
_MV_CHANNEL = "device minute ventilation channel (L/min)"
_TI_CHANNEL = (
    "device inspiratory time channel (seconds), excluding out-of-range samples"
)
_IE_CHANNEL = "device I:E ratio channel (%), excluding out-of-range samples"
_PULSE_CHANNEL = "oximeter pulse channel (bpm), excluding invalid readings"


class PeriodStatistics(BaseModel):
    """Statistics for a time period (week, month, year)."""

    period_type: str = Field(description="Type: daily, weekly, monthly, yearly")
    period_start: date
    period_end: date

    days_used: int = Field(default=0, description="Number of days with therapy")
    days_in_period: int = Field(default=0, description="Total days in period")
    days_compliant: int = Field(default=0, description=COMPLIANT_DAYS_DESCRIPTION)
    compliance_pct: float | None = Field(
        default=None,
        description=(
            "days_compliant as a percent of the period's calendar days (to 0.1), "
            "leaving out days before the range's first or after its last day with "
            "therapy hours from enabled sessions, as OSCAR's Statistics page does; "
            "null when the period has no calendar day in that span"
        ),
    )
    avg_hours_per_day: float | None = Field(
        default=None, description="Average hours per day used"
    )

    avg_ahi: float | None = Field(
        default=None,
        description=(
            "Usage-weighted average of daily headline AHI"
            " (can mix device-reported and recounted days)"
        ),
    )
    median_ahi: float | None = Field(
        default=None,
        description="Median of daily headline AHI (can mix device-reported and recounted days)",
    )
    avg_pressure: float | None = Field(
        default=None,
        description="Usage-weighted mean of the daily median pressure (cmH₂O)",
    )
    avg_leak: float | None = Field(
        default=None, description="Usage-weighted mean of the daily median leak (L/min)"
    )

    avg_spo2: float | None = Field(default=None, description="Average SpO₂ (%)")
    min_spo2: float | None = Field(default=None, description="Minimum SpO₂ (%)")

    avg_total_sleep_hours: float | None = Field(
        default=None, description="Average total sleep hours per night (Apple Health)"
    )
    avg_sleep_efficiency_pct: float | None = Field(
        default=None, description="Average sleep efficiency % per night (Apple Health)"
    )

    avg_oai: float | None = Field(
        default=None,
        description=(
            "Usage-weighted average of daily headline OAI (events/hour)"
            " (can mix device-reported and recounted days)"
        ),
    )
    avg_cai: float | None = Field(
        default=None,
        description=(
            "Usage-weighted average of daily headline CAI (events/hour)"
            " (can mix device-reported and recounted days)"
        ),
    )
    avg_hi: float | None = Field(
        default=None,
        description=(
            "Usage-weighted average of daily headline HI (events/hour)"
            " (can mix device-reported and recounted days)"
        ),
    )
    avg_rera: float | None = Field(
        default=None,
        description=(
            "Usage-weighted device-scored RERA index (events/hour) over days from "
            "devices that have scored a RERA; null when none in the period has"
        ),
    )
    device_flg_95th: float | None = Field(
        default=None,
        description=(
            f"95th percentile of the {_FL_CHANNEL}, over the pooled samples of every "
            "enabled session in the period; null when none has the channel"
        ),
    )
    snore_95th: float | None = Field(
        default=None,
        description=(
            f"95th percentile of the {_SNORE_CHANNEL}, over the pooled samples of "
            "every enabled session in the period; null when none has the channel"
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "period_type": "monthly",
                "period_start": "2024-01-01",
                "period_end": "2024-01-31",
                "days_used": 29,
                "days_in_period": 31,
                "avg_hours_per_day": 7.2,
                "avg_ahi": 2.8,
                "median_ahi": 2.3,
                "avg_pressure": 10.5,
                "avg_leak": 9.2,
                "avg_spo2": 96.2,
                "min_spo2": 89,
            }
        }
    )


class EventValidationResult(BaseModel):
    """
    Validation results comparing programmatic vs machine-detected events.

    Useful for tuning detection thresholds and assessing algorithm accuracy.
    """

    machine_event_count: int = Field(description="Events detected by CPAP machine")
    programmatic_event_count: int = experimental_field(
        "Events detected programmatically"
    )
    matched_events: int = experimental_field(
        "Events matched between machine and programmatic (within 5s)"
    )
    false_positives: int = experimental_field(
        "Programmatic events not matched to machine events"
    )
    false_negatives: int = experimental_field(
        "Machine events not matched to programmatic events"
    )
    sensitivity: float = experimental_field(
        "Recall: matched / (matched + false_negatives)", ge=0, le=1
    )
    precision: float = experimental_field(
        "Precision: matched / (matched + false_positives)", ge=0, le=1
    )
    f1_score: float = experimental_field(
        "F1 score: 2 * (precision * sensitivity) / (precision + sensitivity)",
        ge=0,
        le=1,
    )
    agreement_percentage: float = experimental_field(
        "Overall agreement: matched / max(machine, programmatic) * 100", ge=0, le=100
    )


class DatabaseStats(BaseModel):
    """Database statistics including table row counts and coverage metrics."""

    db_path: str = Field(description="Path to the database file")
    size_mb: float = Field(description="Database file size in megabytes")
    profile_count: int = Field(description="Number of profiles")
    device_count: int = Field(description="Number of devices")
    session_count: int = Field(description="Number of sessions")
    day_count: int = Field(description="Number of days")
    event_count: int = Field(description="Number of events")
    waveform_count: int = Field(description="Number of waveform records")
    analysis_count: int = Field(description="Number of analysis results")
    pattern_count: int = Field(description="Number of detected patterns")
    sessions_with_waveforms: int = Field(description="Sessions that have waveform data")
    sessions_with_events: int = Field(description="Sessions that have event data")
    sessions_with_analysis: int = Field(
        description="Distinct sessions having at least one analysis result"
    )
    analyzable_session_count: int = Field(
        description="Sessions having a flow waveform (prerequisite for analysis)"
    )
    waveform_coverage_pct: float = Field(
        description="Percentage of sessions with waveforms"
    )
    event_coverage_pct: float = Field(description="Percentage of sessions with events")
    analysis_coverage_pct: float = Field(
        description=(
            "Percentage of analyzable sessions (those with a flow waveform) that have "
            "been analyzed: sessions_with_analysis / analyzable_session_count * 100"
        )
    )
    first_session: datetime | None = Field(
        default=None, description="Earliest session date"
    )
    last_session: datetime | None = Field(
        default=None, description="Latest session date"
    )


class SessionListItem(BaseModel):
    """Single session item in a list view."""

    id: int = Field(description="Session database ID")
    therapy_day: date = Field(
        description="Therapy day (noon-cutoff date): sessions before 12:00 belong to the previous calendar day"
    )
    start_time: datetime = Field(description="Session start timestamp")
    duration_hours: float = Field(description="Session duration in hours")
    enabled: bool = Field(description="Whether session is enabled for stats")
    manufacturer: str = Field(description="Device manufacturer")
    model: str = Field(description="Device model")
    serial_number: str = Field(description="Device serial number")
    ahi: float | None = Field(
        default=None, description="Apnea-Hypopnea Index (device events / mask-on hours)"
    )


class SessionListResult(BaseModel):
    """Result of a session list query with pagination info."""

    sessions: list[SessionListItem] = Field(description="List of session items")
    total_count: int = Field(description="Total sessions matching filters")
    limit: int = Field(description="Result limit applied")


class SessionStatistics(BaseModel):
    """Statistics for a single session (from Statistics table)."""

    model_config = ConfigDict(from_attributes=True)

    usage_hours: float | None = Field(default=None, description="Mask-on therapy hours")
    ahi: float | None = Field(
        default=None,
        description="AHI: device-scored apneas of every type and hypopneas per mask-on hour",
    )
    rei: float | None = Field(default=None, description="Respiratory Event Index")
    ahi_device: float | None = Field(
        default=None, description="AHI as reported by the device (STR)"
    )
    oai_device: float | None = Field(
        default=None,
        description="Obstructive apnea index as reported by the device (STR)",
    )
    cai_device: float | None = Field(
        default=None, description="Central apnea index as reported by the device (STR)"
    )
    hi_device: float | None = Field(
        default=None, description="Hypopnea index as reported by the device (STR)"
    )
    usage_hours_device: float | None = Field(
        default=None,
        description=(
            "Device-reported (STR) mask-on hours for the whole day, not this "
            "session: the same daily value is repeated on every session of the day"
        ),
    )
    oai: float | None = Field(
        default=None, description="Device-scored obstructive apneas per mask-on hour"
    )
    cai: float | None = Field(
        default=None, description="Device-scored central apneas per mask-on hour"
    )
    hi: float | None = Field(
        default=None, description="Device-scored hypopneas per mask-on hour"
    )
    obstructive_apneas: int | None = Field(
        default=None, description="Device-scored obstructive apnea count"
    )
    central_apneas: int | None = Field(
        default=None,
        description="Device-scored central apnea count, including clear airway events",
    )
    mixed_apneas: int | None = Field(
        default=None,
        description="Device-scored mixed apnea count; null when the device has never scored one",
    )
    hypopneas: int | None = Field(
        default=None, description="Device-scored hypopnea count"
    )
    reras: int | None = Field(
        default=None,
        description=(
            "Device-scored RERA count; null in API responses when the device has never "
            "scored one, while exports keep the raw 0"
        ),
    )
    unclassified_apneas: int | None = Field(
        default=None,
        description=(
            "Device-scored unclassified apnea count; null in API responses when the "
            "device has never scored one, while exports keep the raw 0"
        ),
    )
    flow_limitations: int | None = Field(
        default=None,
        description=(
            "Device-flagged flow limitation count; null when the device has never "
            "flagged one"
        ),
    )
    pressure_mean: float | None = Field(
        default=None, description="Mean pressure (cmH2O)"
    )
    pressure_min: float | None = Field(default=None, description="Min pressure (cmH2O)")
    pressure_max: float | None = Field(
        default=None,
        description="Max pressure (cmH2O); recomputed from the waveform when available, otherwise the device's value",
    )
    pressure_median: float | None = Field(
        default=None,
        description="Median pressure (cmH2O); recomputed from the waveform when available, otherwise the device's value",
    )
    pressure_95th: float | None = Field(
        default=None,
        description="95th percentile pressure (cmH2O); recomputed from the waveform when available, otherwise the device's value",
    )
    epap_mean: float | None = Field(default=None, description="Mean EPAP (cmH2O)")
    epap_min: float | None = Field(default=None, description="Min EPAP (cmH2O)")
    epap_max: float | None = Field(
        default=None,
        description="Max EPAP (cmH2O); recomputed from the waveform when available, otherwise the device's value",
    )
    epap_median: float | None = Field(
        default=None,
        description="Median EPAP (cmH2O); recomputed from the waveform when available, otherwise the device's value",
    )
    epap_95th: float | None = Field(
        default=None,
        description="95th percentile EPAP (cmH2O); recomputed from the waveform when available, otherwise the device's value",
    )
    ipap_median: float | None = Field(default=None, description="Median IPAP (cmH2O)")
    ipap_95th: float | None = Field(
        default=None, description="95th percentile IPAP (cmH2O)"
    )
    ipap_max: float | None = Field(default=None, description="Max IPAP (cmH2O)")
    leak_mean: float | None = Field(default=None, description="Mean leak (L/min)")
    leak_min: float | None = Field(default=None, description="Min leak (L/min)")
    leak_max: float | None = Field(
        default=None,
        description="Max leak (L/min); recomputed from the waveform when available, otherwise the device's value",
    )
    leak_median: float | None = Field(
        default=None,
        description="Median leak (L/min); recomputed from the waveform when available, otherwise the device's value",
    )
    leak_percentile_70: float | None = Field(
        default=None, description="70th percentile leak (L/min)"
    )
    leak_95th: float | None = Field(
        default=None,
        description="95th percentile leak (L/min); recomputed from the waveform when available, otherwise the device's value",
    )
    spo2_mean: float | None = Field(default=None, description="Mean SpO2 (%)")
    spo2_min: float | None = Field(default=None, description="Min SpO2 (%)")
    spo2_max: float | None = Field(
        default=None,
        description="Max SpO2 (%); recomputed from the waveform when available, otherwise the device's value",
    )
    spo2_median: float | None = Field(default=None, description="Median SpO2 (%)")
    spo2_95th: float | None = Field(
        default=None, description="95th percentile SpO2 (%)"
    )
    spo2_time_below_90: int | None = Field(
        default=None, description="Seconds with SpO2 below 90%"
    )
    pulse_mean: float | None = Field(default=None, description="Mean pulse (BPM)")
    pulse_min: float | None = Field(default=None, description="Min pulse (BPM)")
    pulse_max: float | None = Field(default=None, description="Max pulse (BPM)")
    respiratory_rate_mean: float | None = Field(
        default=None,
        description="Respiratory rate (breaths/min); device STR median on ResMed, OSCAR session average on OSCAR imports",
    )
    respiratory_rate_min: float | None = Field(
        default=None,
        description="Min respiratory rate (breaths/min); OSCAR session summary (OSCAR imports only)",
    )
    respiratory_rate_max: float | None = Field(
        default=None,
        description="Max respiratory rate (breaths/min); device STR value on ResMed, OSCAR session summary on OSCAR imports",
    )
    respiratory_rate_95th: float | None = Field(
        default=None, description="95th percentile respiratory rate (breaths/min)"
    )
    tidal_volume_mean: float | None = Field(
        default=None,
        description="Mean tidal volume (mL); device STR median on ResMed, OSCAR session average on OSCAR imports",
    )
    tidal_volume_min: float | None = Field(
        default=None,
        description="Min tidal volume (mL); OSCAR session summary (OSCAR imports only)",
    )
    tidal_volume_max: float | None = Field(
        default=None,
        description="Max tidal volume (mL); device STR value on ResMed, OSCAR session summary on OSCAR imports",
    )
    tidal_volume_95th: float | None = Field(
        default=None, description="95th percentile tidal volume (mL)"
    )
    minute_ventilation_mean: float | None = Field(
        default=None,
        description="Minute ventilation (L/min); device STR median on ResMed, OSCAR session average on OSCAR imports",
    )
    minute_ventilation_min: float | None = Field(
        default=None,
        description="Min minute ventilation (L/min); OSCAR session summary (OSCAR imports only)",
    )
    minute_ventilation_max: float | None = Field(
        default=None,
        description="Max minute ventilation (L/min); device STR value on ResMed, OSCAR session summary on OSCAR imports",
    )
    minute_ventilation_95th: float | None = Field(
        default=None, description="95th percentile minute ventilation (L/min)"
    )
    uai: float | None = Field(default=None, description="Unknown apnea index")
    ai: float | None = Field(default=None, description="Apnea index")
    rin: float | None = Field(default=None, description="RERA index")
    csr_pct: float | None = Field(
        default=None,
        description=(
            "Device STR CSR value. The unit is unverified: OSCAR stores this signal as a "
            "per-hour rate, not a percent of session time"
        ),
    )
    spont_cyc_pct: float | None = Field(
        default=None, description="Percent of breaths spontaneously cycled"
    )
    ie_ratio_median: float | None = Field(default=None, description="Median I:E ratio")
    ie_ratio_95th: float | None = Field(
        default=None, description="95th percentile I:E ratio"
    )
    ie_ratio_max: float | None = Field(default=None, description="Max I:E ratio")
    ti_median: float | None = Field(
        default=None, description="Median inspiratory time (s)"
    )
    ti_95th: float | None = Field(
        default=None, description="95th percentile inspiratory time (s)"
    )
    ti_max: float | None = Field(default=None, description="Max inspiratory time (s)")
    flow_5th: float | None = Field(
        default=None, description="5th percentile flow (L/min)"
    )
    flow_95th: float | None = Field(
        default=None, description="95th percentile flow (L/min)"
    )
    blow_press_5th: float | None = Field(
        default=None, description="5th percentile blower pressure (cmH2O)"
    )
    blow_press_95th: float | None = Field(
        default=None, description="95th percentile blower pressure (cmH2O)"
    )
    blow_flow_median: float | None = Field(
        default=None, description="Median blower flow (L/min)"
    )
    amb_humidity_median: float | None = Field(
        default=None, description="Median ambient humidity (%)"
    )
    hum_temp_median: float | None = Field(
        default=None, description="Median humidifier temperature (C)"
    )
    htube_temp_median: float | None = Field(
        default=None, description="Median heated-tube temperature (C)"
    )
    htube_pow_median: float | None = Field(
        default=None, description="Median heated-tube power (%)"
    )
    hum_pow_median: float | None = Field(
        default=None, description="Median humidifier power (%)"
    )
    mask_events: float | None = Field(default=None, description="Mask-on events")


class MaskLogEntryResponse(BaseModel):
    """A single user-entered mask equipment log entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    brand: str | None = None
    model: str | None = None
    size: str | None = None
    style: str | None = None
    start_date: date | None = None
    notes: str | None = None


class MaskEpochResponse(BaseModel):
    """A contiguous run of nights sharing one device-reported mask type.

    style is the normalized mask_log-style value (None when the device value is
    unrecognized).  device_id identifies the reporting device for multi-device
    installs.
    """

    mask_type: str
    style: str | None
    start_date: date
    end_date: date
    days_count: int
    device_id: int | None
    device_name: str | None


class SessionSetting(BaseModel):
    """Single setting key-value pair for a session."""

    key: str = Field(description="Setting key")
    value: str | None = Field(default=None, description="Setting value")


def _pooled_field(statistic: str, channel: str, note: str = "") -> Any:
    """A null-by-default statistic of ``channel``'s pooled samples."""
    return Field(
        default=None, description=f"{statistic} of the {channel}, pooled{note}"
    )


def _waveform_field(statistic: str, channel: str) -> Any:
    """``_pooled_field`` of a channel that ``SessionStatistics`` also summarizes."""
    return _pooled_field(
        statistic,
        channel,
        "; computed from the recorded waveform samples, unlike the same-named "
        "stored summary in session statistics (on ResMed, one STR value for the "
        "whole therapy day), so the two can differ in value and unit",
    )


class DeviceChannelStats(BaseModel):
    """Distribution of the stored FL, snore, respiratory and pulse waveforms.

    Each statistic is taken over the samples pooled across the sessions of the
    parent response; a channel none of them has reads null.
    """

    model_config = ConfigDict(extra="forbid")

    device_flg_median: float | None = _pooled_field("Median", _FL_CHANNEL)
    device_flg_95th: float | None = _pooled_field("95th percentile", _FL_CHANNEL)
    device_flg_max: float | None = _pooled_field("Maximum", _FL_CHANNEL)
    snore_median: float | None = _pooled_field("Median", _SNORE_CHANNEL)
    snore_95th: float | None = _pooled_field("95th percentile", _SNORE_CHANNEL)
    snore_max: float | None = _pooled_field("Maximum", _SNORE_CHANNEL)
    respiratory_rate_median: float | None = _waveform_field("Median", _RR_CHANNEL)
    respiratory_rate_95th: float | None = _waveform_field(
        "95th percentile", _RR_CHANNEL
    )
    respiratory_rate_max: float | None = _waveform_field("Maximum", _RR_CHANNEL)
    tidal_volume_median: float | None = _waveform_field("Median", _TV_CHANNEL)
    tidal_volume_95th: float | None = _waveform_field("95th percentile", _TV_CHANNEL)
    tidal_volume_max: float | None = _waveform_field("Maximum", _TV_CHANNEL)
    minute_ventilation_median: float | None = _waveform_field("Median", _MV_CHANNEL)
    minute_ventilation_95th: float | None = _waveform_field(
        "95th percentile", _MV_CHANNEL
    )
    minute_ventilation_max: float | None = _waveform_field("Maximum", _MV_CHANNEL)
    ti_median: float | None = _waveform_field("Median", _TI_CHANNEL)
    ti_95th: float | None = _waveform_field("95th percentile", _TI_CHANNEL)
    ti_max: float | None = _waveform_field("Maximum", _TI_CHANNEL)
    ie_ratio_median: float | None = _waveform_field("Median", _IE_CHANNEL)
    ie_ratio_95th: float | None = _waveform_field("95th percentile", _IE_CHANNEL)
    ie_ratio_max: float | None = _waveform_field("Maximum", _IE_CHANNEL)
    pulse_median: float | None = _waveform_field("Median", _PULSE_CHANNEL)
    pulse_95th: float | None = _waveform_field("95th percentile", _PULSE_CHANNEL)
    pulse_max: float | None = _waveform_field("Maximum", _PULSE_CHANNEL)


class MaskOnSpan(BaseModel):
    """One continuous stretch of mask-on time, on the device's wall clock."""

    session_id: int
    start: datetime = Field(description="Device wall-clock start of the span")
    end: datetime = Field(description="Device wall-clock end of the span")
    duration_seconds: float = Field(description="Span length in seconds")


class LargeLeakSpan(BaseModel):
    """One stretch of leak at or above the 24 L/min redline (OSCAR's Large Leak)."""

    session_id: int
    offset_seconds: float = Field(
        description="Seconds from the session's start to the span's start"
    )
    start: datetime = Field(
        description="Device wall-clock time of the first sample at or above the redline"
    )
    end: datetime = Field(
        description="Device wall-clock time of the first sample below the redline (a "
        "sentinel or missing sample counts as below), or one sample period after "
        "the last sample when the recording ends or pauses first"
    )
    duration_seconds: float = Field(
        description="Span length in seconds (end minus start)"
    )


class LeakRedlineStats(BaseModel):
    """OSCAR's leak redline views of the stored leak channel (24 L/min).

    Every field covers the same sessions: those of the parent response with at
    least one valid leak sample.
    """

    model_config = ConfigDict(extra="forbid")

    leak_above_24_pct: float = Field(
        description=(
            "Percent of the pooled valid leak samples at or above 24 L/min (to 0.1). "
            "OSCAR's time over the leak redline is time-weighted over the day's hours, "
            "so the two agree only approximately"
        )
    )
    large_leak_pct: float | None = Field(
        default=None,
        description=(
            "OSCAR's Large Leak index: the summed durations of every Large Leak span "
            "(not only the listed ones) as a percent of the covered sessions' mask-on "
            "hours (to 0.01); null when those hours are zero"
        ),
    )
    large_leak_span_count: int = Field(
        description="Number of Large Leak spans, listed or not"
    )
    large_leak_spans: list[LargeLeakSpan] = Field(
        description="Stretches of leak at or above 24 L/min, ascending by start; a "
        f"span never crosses a recording gap. At most {MAX_LARGE_LEAK_SPANS} are "
        "listed: when large_leak_span_count is higher, these are the longest",
    )


TIME_IN_APNEA_DESCRIPTION = (
    "OSCAR's total time in apnea: the summed durations of {scope} device-scored "
    "AHI events, in seconds. Events with no recorded duration add nothing, "
    "which on ResMed is every hypopnea"
)


class SessionDetail(BaseModel):
    """Detailed view of a single session with all metadata."""

    id: int
    device_session_id: str
    device_manufacturer: str | None
    device_model: str | None
    device_serial: str | None
    therapy_day: date = Field(
        description="Therapy day (noon-cutoff date): sessions before 12:00 belong to the previous calendar day"
    )
    start_time: datetime
    end_time: datetime
    duration_hours: float = Field(description="Session duration in hours")
    duration_seconds: float = Field(description="Session duration in seconds")
    therapy_mode: str | None
    enabled: bool
    event_count: int
    waveform_count: int
    waveform_types: list[str]
    has_statistics: bool
    has_event_data: bool
    import_source: str | None = None
    parser_version: str | None = None
    data_quality_notes: list[str] = Field(default_factory=list)
    statistics: SessionStatistics | None = None
    settings: list[SessionSetting] | None = None
    active_mask: MaskLogEntryResponse | None = None
    channel_stats: DeviceChannelStats | None = Field(
        default=None,
        description="This session's device channel statistics, whether or not it "
        "is enabled; null when it has none of the channels, and always null on "
        "PATCH responses, which omit it",
    )
    time_in_apnea_seconds: float = Field(
        description=TIME_IN_APNEA_DESCRIPTION.format(scope="this session's")
    )
    leak_redline: LeakRedlineStats | None = Field(
        default=None,
        description="This session's time over the leak redline and Large Leak "
        "spans; null when it has no valid leak sample or its leak data is "
        "unusable, and always null on PATCH responses",
    )
    mask_on_spans: list[MaskOnSpan] | None = Field(
        default=None,
        description="This session's mask-on spans; null when its mask-on segments "
        "are unknown (sessions imported before segments were stored, or a "
        "malformed stored value), empty when it is known to have none",
    )


class DeletePreview(BaseModel):
    """Preview of sessions and related data to be deleted."""

    sessions: list[SessionListItem] = Field(description="Sessions to be deleted")
    event_count: int = Field(description="Total events to be deleted")
    waveform_count: int = Field(description="Total waveform records to be deleted")
    stats_count: int = Field(description="Total statistics records to be deleted")


class EventTypeCount(BaseModel):
    """Event type with count and percentage."""

    event_type: str
    count: int = Field(description="Device-scored events of this type")
    percentage: float = Field(description="Share of all device-scored events (%)")


class TherapySummary(BaseModel):
    """Aggregated therapy statistics summary."""

    first_date: date = Field(
        description="First day with therapy hours (first day in range if none)"
    )
    last_date: date = Field(
        description="Last day with therapy hours (last day in range if none)"
    )
    days_since_last: int = Field(description="Days since the last therapy day")
    total_hours: float = Field(description="Total therapy hours")
    avg_hours: float = Field(description="Average therapy hours per day with usage")
    days_with_data: int = Field(
        description="Days with therapy hours from enabled sessions"
    )
    days_compliant: int = Field(default=0, description=COMPLIANT_DAYS_DESCRIPTION)
    compliance_pct: float = Field(
        default=0.0,
        description=(
            "days_compliant as a percent of the calendar days from first_date to "
            "last_date (to 0.1)"
        ),
    )
    avg_ahi: float | None = Field(
        default=None,
        description=(
            "Usage-weighted average of daily headline AHI"
            " (can mix device-reported and recounted days)"
        ),
    )
    avg_rei: float | None = Field(default=None, description="Average REI")
    avg_pressure: float | None = Field(
        default=None,
        description="Usage-weighted mean of the daily median pressure (cmH2O)",
    )
    min_pressure: float | None = Field(
        default=None, description="Lowest daily median pressure (cmH2O)"
    )
    max_pressure: float | None = Field(
        default=None, description="Highest daily median pressure (cmH2O)"
    )
    avg_epap: float | None = Field(default=None, description="Average EPAP (cmH2O)")
    avg_leak: float | None = Field(
        default=None, description="Usage-weighted mean of the daily median leak (L/min)"
    )
    avg_spo2: float | None = Field(default=None, description="Average SpO2 (%)")
    min_spo2: float | None = Field(default=None, description="Minimum SpO2 (%)")
    total_spo2_time_below_90: int = Field(
        default=0, description="Total seconds with SpO2 below 90%"
    )
    avg_pulse: float | None = Field(default=None, description="Average pulse (BPM)")
    avg_respiratory_rate: float | None = Field(
        default=None, description="Average respiratory rate (breaths/min)"
    )
    avg_tidal_volume: float | None = Field(
        default=None, description="Average tidal volume (mL)"
    )
    avg_minute_ventilation: float | None = Field(
        default=None, description="Average minute ventilation (L/min)"
    )
    event_counts: list[EventTypeCount] = Field(default_factory=list)


class ExperimentalStats(BaseModel):
    """SNORE's own judgements about a therapy range, kept out of ``TherapySummary``."""

    effectiveness: TherapyEffectiveness = experimental_field(
        "SNORE's therapy effectiveness band from average AHI "
        "(excellent, good, fair, poor; unknown without an average AHI)."
    )
    ahi_trend_direction: AhiTrendDirection | None = experimental_field(
        "SNORE's AHI trend call: the latest week's AHI against the earlier weeks "
        "in the range (improving, worsening, stable; null with too little data or "
        "when the earlier weeks average 0).",
        default=None,
    )


class WaveformInfo(BaseModel):
    """Waveform metadata for listing."""

    waveform_type: str
    sample_rate: float = Field(description="Sample rate (Hz)")
    sample_count: int = Field(description="Number of samples")
    unit: str | None = None
    duration_hours: float = Field(description="Recorded duration in hours")


AnalysisSelection = Literal["missing", "stale"]
"""Batch session selector: ``missing`` = never analyzed; ``stale`` = latest
analysis came from an older algorithm version (shown as "outdated" in the UI)."""

EMPTY_SELECTION_MESSAGES: dict[AnalysisSelection, str] = {
    "missing": "No unanalyzed sessions to backfill",
    "stale": "No sessions with outdated analysis",
}
"""What the API and CLI report when an ``AnalysisSelection`` matches nothing."""


class AnalysisListItem(BaseModel):
    """Session with analysis status for listing."""

    session_id: int
    session_date: date
    duration_hours: float | None = Field(
        default=None, description="Session duration in hours"
    )
    has_analysis: bool
    analysis_id: int | None = None


class AnalysisSessionDetail(BaseModel):
    """Session detail for analysis deletion preview."""

    id: int
    start_time: datetime
    manufacturer: str | None = None
    model: str | None = None
    version_count: int


class AnalysisDeletePreview(BaseModel):
    """Preview of analysis data to be deleted."""

    sessions_with_analysis: int
    total_analysis_records: int
    records_to_delete: int
    patterns_count: int
    session_details: list[AnalysisSessionDetail] = Field(default_factory=list)


class DeviceInfo(BaseModel):
    """Device information for listing."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    manufacturer: str
    model: str
    serial_number: str
    firmware_version: str | None = None
    hardware_version: str | None = None
    product_code: str | None = None
    first_seen: datetime
    last_import: datetime | None = None


class SettingChangeEntry(BaseModel):
    """A single setting that changed between two consecutive sessions."""

    key: str
    old_value: str | None
    new_value: str | None


class SettingsChange(BaseModel):
    """Settings that changed for a particular session relative to the prior one."""

    session_id: int
    date: date
    changes: list[SettingChangeEntry]


class DeviceUsageSummary(BaseModel):
    """Aggregated usage statistics for a device."""

    session_count: int
    first_session_date: date | None
    last_session_date: date | None
    total_therapy_hours: float = Field(
        description=(
            "Total therapy hours from enabled sessions, summed over the device's days "
            "(mask-on time, falling back to session span)"
        )
    )
    therapy_modes: list[str]


class DeviceDetail(DeviceInfo):
    """Full device detail including usage summary, current settings, and settings history."""

    usage: DeviceUsageSummary
    current_settings: dict[str, str] | None
    settings_history: list[SettingsChange]


class HealthNightCoverage(BaseModel):
    """The span and count of a profile's Apple Health nights (nightly sleep summaries)."""

    first_night_date: date
    last_night_date: date
    night_count: int


class HealthNightSummaryRead(BaseModel):
    """Derived nightly sleep summary from Apple Health data."""

    model_config = ConfigDict(from_attributes=True)

    night_date: date
    preferred_source: str | None = None
    time_in_bed_seconds: float | None = Field(
        default=None, description="Time in bed (s)"
    )
    total_sleep_seconds: float | None = Field(
        default=None, description="Total sleep (s)"
    )
    core_seconds: float | None = Field(default=None, description="Core sleep (s)")
    deep_seconds: float | None = Field(default=None, description="Deep sleep (s)")
    rem_seconds: float | None = Field(default=None, description="REM sleep (s)")
    awake_seconds: float | None = Field(default=None, description="Awake time (s)")
    unspecified_seconds: float | None = Field(
        default=None, description="Asleep, unspecified stage (s)"
    )
    sleep_efficiency_pct: float | None = Field(
        default=None, description="Total sleep / time in bed (%)"
    )
    stage_coverage_pct: float | None = Field(
        default=None, description="Share of sleep with a known stage (%)"
    )
    computed_at: datetime


class HealthNightDetailRead(HealthNightSummaryRead):
    """Nightly sleep summary with oximetry, respiratory rate and breathing disturbances."""

    avg_spo2_pct: float | None = Field(default=None, description="Average SpO2 (%)")
    min_spo2_pct: float | None = Field(default=None, description="Minimum SpO2 (%)")
    avg_rr: float | None = Field(
        default=None, description="Average respiratory rate (breaths/min)"
    )
    breathing_disturbances: float | None = Field(
        default=None,
        description=(
            "Apple's nightly breathing-disturbances value as recorded (HealthKit "
            "unit 'count', a non-integer nightly value; Apple writes one per night); "
            "when several sources wrote one, the mean of their values; null without "
            "a sample"
        ),
    )


class HealthSampleRead(BaseModel):
    """Single Apple Health sample (sleep stage or quantity record)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    record_type: str
    source_name: str
    start_time: datetime
    end_time: datetime
    value_text: str | None = Field(
        default=None,
        description=(
            "Sleep stage for sleep-analysis samples (InBed, Awake, AsleepCore, "
            "AsleepDeep, AsleepREM or AsleepUnspecified); null for quantity samples"
        ),
    )
    value_num: float | None = Field(
        default=None,
        description=(
            "Numeric sample value as recorded; SpO2 recorded as a fraction is "
            "stored on the percent scale"
        ),
    )
    unit: str | None = None
    night_date: date


class DayListItem(BaseModel):
    """Summary of a single therapy day."""

    model_config = ConfigDict(from_attributes=True)

    date: date
    device_id: int
    session_count: int
    total_therapy_hours: float | None = Field(
        default=None, description="Total mask-on hours across the day's sessions"
    )
    ahi: float | None = Field(
        default=None,
        description=f"Headline AHI (events/hr): {HEADLINE_INDEX_DESCRIPTION}",
    )
    index_source: IndexSource | None = Field(
        default=None, description=INDEX_SOURCE_DESCRIPTION
    )
    ahi_computed: float | None = Field(
        default=None,
        description=(
            "SNORE recount AHI: device-scored events over SNORE mask-on hours, "
            "usage-weighted across sessions; null when there are no mask-on hours "
            "or no session statistics"
        ),
    )


class DayDetail(DayListItem):
    """Full detail of a therapy day including per-metric stats."""

    oai: float | None = Field(
        default=None, description="Headline OAI: same source rule as ahi"
    )
    cai: float | None = Field(
        default=None, description="Headline CAI: same source rule as ahi"
    )
    hi: float | None = Field(
        default=None, description="Headline HI: same source rule as ahi"
    )
    oai_computed: float | None = Field(
        default=None, description="SNORE recount OAI (see ahi_computed)"
    )
    cai_computed: float | None = Field(
        default=None, description="SNORE recount CAI (see ahi_computed)"
    )
    hi_computed: float | None = Field(
        default=None, description="SNORE recount HI (see ahi_computed)"
    )
    avg_pressure: float | None = Field(
        default=None,
        description=pooled_percentile_description("Median", "pressure", "cmH2O"),
    )
    avg_leak: float | None = Field(
        default=None,
        description=pooled_percentile_description("Median", "leak", "L/min"),
    )
    avg_spo2: float | None = Field(
        default=None, description="Usage-weighted mean SpO2 (%)"
    )
    # Pressure detail
    pressure_min: float | None = Field(default=None, description="Min pressure (cmH2O)")
    pressure_max: float | None = Field(default=None, description="Max pressure (cmH2O)")
    pressure_median: float | None = Field(
        default=None,
        description=pooled_percentile_description("Median", "pressure", "cmH2O"),
    )
    pressure_95th: float | None = Field(
        default=None,
        description=pooled_percentile_description(
            "95th percentile", "pressure", "cmH2O"
        ),
    )
    # EPAP detail
    epap_min: float | None = Field(default=None, description="Min EPAP (cmH2O)")
    epap_max: float | None = Field(default=None, description="Max EPAP (cmH2O)")
    epap_median: float | None = Field(
        default=None,
        description=pooled_percentile_description("Median", "EPAP", "cmH2O"),
    )
    epap_mean: float | None = Field(
        default=None, description="Usage-weighted mean EPAP (cmH2O)"
    )
    epap_95th: float | None = Field(
        default=None,
        description=pooled_percentile_description("95th percentile", "EPAP", "cmH2O"),
    )
    # Leak detail
    leak_min: float | None = Field(default=None, description="Min leak (L/min)")
    leak_max: float | None = Field(default=None, description="Max leak (L/min)")
    leak_mean: float | None = Field(
        default=None, description="Usage-weighted mean leak (L/min)"
    )
    leak_95th: float | None = Field(
        default=None,
        description=pooled_percentile_description("95th percentile", "leak", "L/min"),
    )
    # SpO2 detail
    spo2_min: float | None = Field(default=None, description="Min SpO2 (%)")
    spo2_max: float | None = Field(default=None, description="Max SpO2 (%)")
    # Raw event counts
    obstructive_apneas: int = Field(
        default=0, description="Device-scored obstructive apnea count for the night"
    )
    central_apneas: int = Field(
        default=0,
        description=(
            "Device-scored central apnea count for the night, including clear airway "
            "events"
        ),
    )
    hypopneas: int = Field(
        default=0, description="Device-scored hypopnea count for the night"
    )
    reras: int | None = Field(
        default=None,
        description=(
            "Device-scored RERA count for the night; null when the device has never "
            "scored one"
        ),
    )
    unclassified_apneas: int | None = Field(
        default=None,
        description=(
            "Device-scored unclassified apnea count for the night; null when the "
            "device has never scored one"
        ),
    )
    session_ids: list[int] = Field(default_factory=list)
    health_sleep: HealthNightSummaryRead | None = None
    channel_stats: DeviceChannelStats | None = Field(
        default=None,
        description="Device channel statistics over the pooled samples of the "
        "day's enabled sessions; null when none has any of the channels",
    )
    time_in_apnea_seconds: float | None = Field(
        description=TIME_IN_APNEA_DESCRIPTION.format(
            scope="the day's enabled sessions'"
        )
        + "; null when the day has no enabled session",
    )
    leak_redline: LeakRedlineStats | None = Field(
        default=None,
        description="Time over the leak redline and Large Leak spans of the day's "
        "enabled sessions with leak data; null when none has a valid leak sample "
        "or any one's leak data is unusable",
    )
    mask_on_spans: list[MaskOnSpan] | None = Field(
        default=None,
        description="Mask-on spans of the day's enabled sessions, ascending by "
        "start; null when any of them has unknown segments, empty when the day "
        "has no enabled session",
    )


class DayExperimentalMetrics(BaseModel):
    """Nightly FL / RERA-proxy metrics, sourced read-time from BreathService.

    Each metric carries a ``*_reason`` companion because null here is
    ambiguous ("analysis not run" versus a genuine zero), mirroring the MCP
    null-with-reason convention.  An ordinary un-analyzed night (sessions
    present, none with an OK analysis) is null with "not_available"; a lookup
    failure (no sessions for the device, breath-table DB error, device
    resolution declining) or a night with every session disabled is null with
    "analysis_not_run".
    """

    date: date
    fl_class_ge4_pct: float | None = experimental_field(
        "Percent of rule-classified breaths flagged flow-class >= 4 "
        "(flow-limitation proxy).",
        default=None,
    )
    fl_class_ge4_pct_reason: str | None = None
    rera_index: float | None = experimental_field(
        "RERA-proxy events per mask-on hour of analyzed sessions (FL-run "
        f"proxy, not device-scored). {RERA_PROXY_FALLBACK_NOTE}",
        default=None,
    )
    rera_index_reason: str | None = None
    rera_count: int | None = experimental_field(
        "RERA-proxy count from flow-limitation runs ending in a recovery breath; "
        f"distinct from device-scored `reras`. {RERA_PROXY_FALLBACK_NOTE}",
        default=None,
    )
    rera_count_reason: str | None = None


class RxPeriodResponse(BaseModel):
    """Single therapy prescription period with aggregated stats."""

    settings: dict[str, str]
    start_date: date
    end_date: date
    days_count: int
    avg_ahi: float | None = Field(
        default=None,
        description=(
            "Usage-hours-weighted average of daily headline AHI over the period"
            " (can mix device-reported and recounted days)"
        ),
    )
    median_ahi: float | None = Field(
        default=None,
        description=(
            "Median of daily headline AHI over the period"
            " (can mix device-reported and recounted days)"
        ),
    )
    avg_hours: float | None = Field(
        default=None, description="Average therapy hours per day with usage"
    )
    total_hours: float = Field(default=0.0, description="Total therapy hours")
    avg_leak: float | None = Field(
        default=None,
        description="Usage-weighted mean of the daily median leak over the period (L/min)",
    )
    device_id: int | None = None
    device_name: str | None = None


class RxComparisonResponse(BaseModel):
    """RX period comparison result with best/worst indices."""

    periods: list[RxPeriodResponse]
    best_index: int | None = None
    worst_index: int | None = None


class RxSettingChange(BaseModel):
    """A single per-key settings change on a given day for a device."""

    date: date
    device_id: int
    device_name: str
    key: str
    old_value: str | None = None
    new_value: str | None = None


class RxChangesResponse(BaseModel):
    """All settings changes across all devices, sorted by (date, device_id, key)."""

    changes: list[RxSettingChange]


class MergedSettingsChange(BaseModel):
    """One settings change from either the device settings log or the mask log."""

    date: date
    source: str  # "device_settings" | "mask_log"
    device_id: int | None = None  # null for mask_log entries
    device_name: str | None = None
    key: str  # settings key, or "mask_equipment"
    old_value: str | None = None
    new_value: str | None = None
    mask_brand: str | None = None  # mask_log-only detail, null for device_settings
    mask_model: str | None = None
    mask_size: str | None = None
    mask_style: str | None = None
    notes: str | None = None


class RxAllResponse(BaseModel):
    """Combined RX data derived from a single database query."""

    history: list[RxPeriodResponse]
    current: RxPeriodResponse | None = None
    best_index: int | None = None
    worst_index: int | None = None
    changes: RxChangesResponse


class ImportSource(BaseModel):
    """Detected data source for import."""

    parser_name: str = Field(description="Parser identifier (e.g., 'resmed')")
    device_serial: str | None = Field(default=None, description="Device serial number")
    profile_name: str | None = Field(default=None, description="Data profile name")
    structure_type: str | None = Field(
        default=None, description="Directory structure type"
    )
    root_path: str = Field(description="Root path of data source")
    data_root: str | None = Field(default=None, description="Data root within source")


class ImportSourceResult(BaseModel):
    """Result of importing a single data source."""

    source: ImportSource = Field(description="The source that was imported")
    imported: int = Field(default=0, description="Sessions successfully imported")
    skipped: int = Field(default=0, description="Sessions skipped (already exist)")
    failed: int = Field(default=0, description="Sessions that failed to import")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal warnings")
    imported_session_ids: list[int] = Field(
        default_factory=list,
        description="DB Session.id values for sessions that were successfully imported",
    )


class ImportResult(BaseModel):
    """Aggregate result of an import operation across all sources."""

    total_imported: int = Field(default=0, description="Total sessions imported")
    total_skipped: int = Field(default=0, description="Total sessions skipped")
    total_failed: int = Field(default=0, description="Total sessions that failed")
    sources: list[ImportSourceResult] = Field(
        default_factory=list, description="Per-source results"
    )
    warnings: list[str] = Field(default_factory=list, description="Global warnings")
    imported_session_ids: list[int] = Field(
        default_factory=list,
        description="All successfully imported Session.id values across all sources",
    )


class BatchSessionResult(BaseModel):
    """Result of analyzing a single session in a batch."""

    session_id: int = Field(description="Session database ID")
    session_date: date | None = Field(default=None, description="Session date")
    success: bool = Field(description="Whether analysis succeeded")
    cancelled: bool = Field(
        default=False,
        description="Whether this session was skipped due to cancellation",
    )
    error: str | None = Field(default=None, description="Error message if failed")


class BatchAnalysisResult(BaseModel):
    """Aggregate result of batch analysis across multiple sessions."""

    total: int = Field(description="Total sessions processed")
    successful: int = Field(default=0, description="Sessions analyzed successfully")
    failed: int = Field(default=0, description="Sessions that failed analysis")
    cancelled: int = Field(
        default=0, description="Sessions skipped due to cancellation"
    )
    results: list[BatchSessionResult] = Field(
        default_factory=list, description="Per-session results"
    )


class EventComparisonDetail(BaseModel):
    """Detail of a single unmatched event in a comparison."""

    event_type: str = Field(description="Event type (OA, CA, MA, H, etc.)")
    start_time: float = Field(
        description="Event start time in seconds from session start"
    )
    duration: float = Field(
        description=(
            "Event duration (seconds), 0 when the device recorded none; experimental "
            "for programmatic events"
        )
    )
    source: Literal["machine", "programmatic"] = Field(
        description="Event source (machine/programmatic)"
    )
    confidence: float | None = experimental_field(
        "Detection confidence (programmatic events only)", default=None
    )
    flow_reduction: float | None = experimental_field(
        "Flow reduction fraction (programmatic events only)", default=None
    )


class EventComparisonResult(BaseModel):
    """Result of comparing machine vs programmatic events for a session."""

    session_id: int = Field(description="Session database ID")
    mode: str = Field(description="Detection mode used (e.g., 'aasm')")
    machine_event_count: int = Field(
        description="Machine-scored apneas and hypopneas, plus RERAs when the device scored any"
    )
    programmatic_event_count: int = experimental_field(
        "Programmatically-detected apneas and hypopneas, plus RERAs when the device scored any"
    )
    matched_events: int = experimental_field(
        "Programmatic/machine event pairs matched one-to-one within tolerance "
        "(each event in at most one pair)"
    )
    false_negatives: list[EventComparisonDetail] = experimental_field(
        "Machine events missed by programmatic detection", default_factory=list
    )
    false_positives_apnea: list[EventComparisonDetail] = experimental_field(
        "Programmatic apneas not in machine events", default_factory=list
    )
    false_positives_hypopnea: list[EventComparisonDetail] = experimental_field(
        "Programmatic hypopneas not in machine events", default_factory=list
    )
    false_positives_rera: list[EventComparisonDetail] = experimental_field(
        "Programmatic RERAs not in machine events; always empty when the device "
        "scored no RERAs",
        default_factory=list,
    )


class VacuumResult(BaseModel):
    """Result of a database vacuum operation."""

    status: str = Field(description="Operation status ('success')")
    size_before_mb: float = Field(description="Database size before vacuum in MB")
    size_after_mb: float = Field(description="Database size after vacuum in MB")


class RecomputeDaysResult(BaseModel):
    """Result of re-deriving every Day row from stored session data."""

    days_recomputed: int = Field(description="Day rows re-aggregated and kept")
    days_pruned: int = Field(
        description="Day rows deleted because no session references them"
    )


class ResetResult(BaseModel):
    """Result of a database reset (delete all data, preserve schema) operation."""

    status: str = Field(description="Operation status ('success')")
    tables_cleared: dict[str, int] = Field(
        description="Rows deleted per table (table_name -> count)"
    )
    total_rows_deleted: int = Field(description="Total rows deleted across all tables")
    size_before_mb: float = Field(description="Database size before reset in MB")
    size_after_mb: float = Field(
        description=(
            "Database size in MB once the job's VACUUM finished. Equals"
            " size_before_mb when the VACUUM could not run."
        ),
    )
    bootstrap_invite_url: str | None = Field(
        default=None,
        description=(
            "Admin invite redemption URL (only present after include_accounts=true reset)."
            " The caller's account was deleted; redeem this URL to create a new admin account."
        ),
    )


class DataRange(BaseModel):
    """Profile data availability — always all-time, unaffected by days_limit."""

    earliest_date: date | None = None
    latest_date: date | None = None


# ``[period_start, value]`` pairs, one per period (value null when absent).
type TrendSeries = list[tuple[date, float | None]]


class TrendsResponse(BaseModel):
    """Per-period trend series for ``GET /stats/trends``."""

    ahi: TrendSeries = Field(
        description=(
            "Average daily headline AHI per period"
            " (can mix device-reported and recounted days)"
        )
    )
    usage: TrendSeries = Field(description="Average therapy hours per day, per period")
    spo2: TrendSeries = Field(description="Average SpO2 (%) per period")
    leak: TrendSeries = Field(description="Average leak (L/min) per period")
    pressure: TrendSeries = Field(description="Average pressure (cmH2O) per period")
    oai: TrendSeries = Field(
        description=(
            "Average daily headline OAI per period"
            " (can mix device-reported and recounted days)"
        )
    )
    cai: TrendSeries = Field(
        description=(
            "Average daily headline CAI per period"
            " (can mix device-reported and recounted days)"
        )
    )
    hi: TrendSeries = Field(
        description=(
            "Average daily headline HI per period"
            " (can mix device-reported and recounted days)"
        )
    )
    rera: TrendSeries = Field(description="Average device-scored RERA index per period")
    epap: TrendSeries = Field(description="Average EPAP (cmH2O) per period")
    rr: TrendSeries = Field(
        description="Average respiratory rate (breaths/min) per period"
    )
    pulse: TrendSeries = Field(description="Average pulse (BPM) per period")
    mv: TrendSeries = Field(description="Average minute ventilation (L/min) per period")
    # Apple Health series: keys are omitted (not null) when no night has data.
    total_sleep_hours: TrendSeries | None = Field(
        default=None,
        description="Average total sleep hours per night, per period (Apple Health)",
    )
    sleep_efficiency: TrendSeries | None = Field(
        default=None,
        description="Average sleep efficiency % per night, per period (Apple Health)",
    )


class RecordExtremes(BaseModel):
    """Top-N best and worst ``[date, value]`` days for one metric."""

    best: list[tuple[date, float]] = Field(description="Best days, best first")
    worst: list[tuple[date, float]] = Field(description="Worst days, worst first")


class RecordsResponse(BaseModel):
    """Best/worst days for ``GET /stats/records``.

    Keys are omitted (not null) when no qualifying day (>= 1 h therapy) has the
    metric.
    """

    ahi: RecordExtremes | None = Field(
        default=None,
        description=(
            "Daily headline AHI (device-reported when trusted, else SNORE's"
            " recount; records can mix both)"
        ),
    )
    leak: RecordExtremes | None = Field(
        default=None, description="Nightly median leak (L/min)"
    )
    therapy_hours: RecordExtremes | None = Field(
        default=None, description="Nightly therapy hours"
    )
    spo2_min: RecordExtremes | None = Field(
        default=None, description="Nightly minimum SpO2 (%)"
    )
    total_sleep_hours: RecordExtremes | None = Field(
        default=None, description="Nightly total sleep hours (Apple Health)"
    )


class HealthImportResult(BaseModel):
    """Result of an Apple Health export.xml import operation."""

    inserted: int = Field(default=0, description="Health samples successfully inserted")
    skipped: int = Field(default=0, description="Samples skipped as duplicates")
    unknown_metrics: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Unhandled HealthKit record types from the XML export with their record counts."
        ),
    )
    nights_recomputed: int = Field(
        default=0,
        description="Nightly sleep summaries recomputed (0 on dry_run)",
    )
    dry_run: bool = Field(
        default=False,
        description="True when no writes were performed",
    )


class DeleteDataResult(BaseModel):
    """Result of a per-user delete-all-data operation."""

    status: str = Field(description="Operation status ('success')")
    devices_deleted: int = Field(
        description="Device rows deleted (cascades removed all sleep data)"
    )
    import_jobs_deleted: int = Field(
        description="Import job records deleted for this user"
    )
    profiles_processed: int = Field(
        description="Profiles whose raw backup dirs were purged"
    )
    size_before_mb: float = Field(description="Database size before deletion in MB")
    size_after_mb: float = Field(
        description=(
            "Database size in MB once the job's VACUUM finished. Equals"
            " size_before_mb when the VACUUM could not run."
        ),
    )
