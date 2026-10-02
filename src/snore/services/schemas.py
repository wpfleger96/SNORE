"""Pydantic response schemas for service layer.

These models define the contract between services and consumers (CLI/API).
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from snore.metrics import COVERAGE_TOLERANCE_FRACTION, COVERAGE_TOLERANCE_HOURS
from snore.provenance import IndexSource, Provenance, provenance_field

__all__ = [
    "HEADLINE_INDEX_DESCRIPTION",
    "INDEX_SOURCE_DESCRIPTION",
    "PeriodStatistics",
    "EventValidationResult",
    "DatabaseStats",
    "SessionListItem",
    "SessionListResult",
    "SessionDetail",
    "SessionStatistics",
    "SessionSetting",
    "DeletePreview",
    "TherapySummary",
    "EventTypeCount",
    "WaveformInfo",
    "AnalysisListItem",
    "AnalysisSessionDetail",
    "AnalysisDeletePreview",
    "EventMatchResult",
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
INDEX_SOURCE_DESCRIPTION = (
    "Source of the headline ahi/oai/cai/hi: 'device' (device-reported daily "
    "value) or 'derived' (SNORE's recount); null when the day has no index"
)


class PeriodStatistics(BaseModel):
    """Statistics for a time period (week, month, year)."""

    period_type: str = Field(description="Type: daily, weekly, monthly, yearly")
    period_start: date
    period_end: date

    days_used: int = provenance_field(
        Provenance.DERIVED, "Number of days with therapy", default=0
    )
    days_in_period: int = Field(default=0, description="Total days in period")
    avg_hours_per_day: float | None = provenance_field(
        Provenance.DERIVED, "Average hours per day used", default=None
    )

    avg_ahi: float | None = provenance_field(
        Provenance.DERIVED,
        "Usage-weighted average of daily headline AHI"
        " (can mix device-reported and recounted days)",
        default=None,
    )
    median_ahi: float | None = provenance_field(
        Provenance.DERIVED,
        "Median of daily headline AHI (can mix device-reported and recounted days)",
        default=None,
    )
    avg_pressure: float | None = provenance_field(
        Provenance.DERIVED, "Average pressure (cmH₂O)", default=None
    )
    avg_leak: float | None = provenance_field(
        Provenance.DERIVED, "Average leak rate (L/min)", default=None
    )

    avg_spo2: float | None = provenance_field(
        Provenance.DERIVED, "Average SpO₂ (%)", default=None
    )
    min_spo2: float | None = provenance_field(
        Provenance.DERIVED, "Minimum SpO₂ (%)", default=None
    )

    avg_total_sleep_hours: float | None = provenance_field(
        Provenance.DERIVED,
        "Average total sleep hours per night (Apple Health)",
        default=None,
    )
    avg_sleep_efficiency_pct: float | None = provenance_field(
        Provenance.DERIVED,
        "Average sleep efficiency % per night (Apple Health)",
        default=None,
    )

    avg_oai: float | None = provenance_field(
        Provenance.DERIVED,
        "Usage-weighted average of daily headline OAI (events/hour)"
        " (can mix device-reported and recounted days)",
        default=None,
    )
    avg_cai: float | None = provenance_field(
        Provenance.DERIVED,
        "Usage-weighted average of daily headline CAI (events/hour)"
        " (can mix device-reported and recounted days)",
        default=None,
    )
    avg_hi: float | None = provenance_field(
        Provenance.DERIVED,
        "Usage-weighted average of daily headline HI (events/hour)"
        " (can mix device-reported and recounted days)",
        default=None,
    )
    avg_rera: float | None = provenance_field(
        Provenance.DERIVED,
        "Average device-scored RERA index (events/hour)",
        default=None,
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

    machine_event_count: int = provenance_field(
        Provenance.DEVICE, "Events detected by CPAP machine"
    )
    programmatic_event_count: int = provenance_field(
        Provenance.EXPERIMENTAL, "Events detected programmatically"
    )
    matched_events: int = provenance_field(
        Provenance.EXPERIMENTAL,
        "Events matched between machine and programmatic (within 5s)",
    )
    false_positives: int = provenance_field(
        Provenance.EXPERIMENTAL, "Programmatic events not matched to machine events"
    )
    false_negatives: int = provenance_field(
        Provenance.EXPERIMENTAL, "Machine events not matched to programmatic events"
    )
    sensitivity: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "Recall: matched / (matched + false_negatives)",
        ge=0,
        le=1,
    )
    precision: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "Precision: matched / (matched + false_positives)",
        ge=0,
        le=1,
    )
    f1_score: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "F1 score: 2 * (precision * sensitivity) / (precision + sensitivity)",
        ge=0,
        le=1,
    )
    agreement_percentage: float = provenance_field(
        Provenance.EXPERIMENTAL,
        "Overall agreement: matched / max(machine, programmatic) * 100",
        ge=0,
        le=100,
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
    duration_hours: float = provenance_field(
        Provenance.DEVICE, "Session duration in hours"
    )
    enabled: bool = Field(description="Whether session is enabled for stats")
    manufacturer: str = Field(description="Device manufacturer")
    model: str = Field(description="Device model")
    serial_number: str = Field(description="Device serial number")
    ahi: float | None = provenance_field(
        Provenance.DERIVED,
        "Apnea-Hypopnea Index (device events / mask-on hours)",
        default=None,
    )


class SessionListResult(BaseModel):
    """Result of a session list query with pagination info."""

    sessions: list[SessionListItem] = Field(description="List of session items")
    total_count: int = Field(description="Total sessions matching filters")
    limit: int = Field(description="Result limit applied")


class SessionStatistics(BaseModel):
    """Statistics for a single session (from Statistics table)."""

    model_config = ConfigDict(from_attributes=True)

    usage_hours: float | None = provenance_field(
        Provenance.DERIVED, "Mask-on therapy hours", default=None
    )
    ahi: float | None = provenance_field(
        Provenance.DERIVED, "AHI: device-scored OA+CA+H per mask-on hour", default=None
    )
    rei: float | None = provenance_field(
        Provenance.DERIVED, "Respiratory Event Index", default=None
    )
    ahi_device: float | None = provenance_field(
        Provenance.DEVICE, "AHI as reported by the device (STR)", default=None
    )
    oai_device: float | None = provenance_field(
        Provenance.DEVICE,
        "Obstructive apnea index as reported by the device (STR)",
        default=None,
    )
    cai_device: float | None = provenance_field(
        Provenance.DEVICE,
        "Central apnea index as reported by the device (STR)",
        default=None,
    )
    hi_device: float | None = provenance_field(
        Provenance.DEVICE,
        "Hypopnea index as reported by the device (STR)",
        default=None,
    )
    usage_hours_device: float | None = provenance_field(
        Provenance.DEVICE,
        "Device-reported (STR) mask-on hours for the whole day, not this "
        "session: the same daily value is repeated on every session of the day",
        default=None,
    )
    oai: float | None = provenance_field(
        Provenance.DERIVED,
        "Device-scored obstructive apneas per mask-on hour",
        default=None,
    )
    cai: float | None = provenance_field(
        Provenance.DERIVED,
        "Device-scored central apneas per mask-on hour",
        default=None,
    )
    hi: float | None = provenance_field(
        Provenance.DERIVED, "Device-scored hypopneas per mask-on hour", default=None
    )
    obstructive_apneas: int | None = provenance_field(
        Provenance.DEVICE, "Device-scored obstructive apnea count", default=None
    )
    central_apneas: int | None = provenance_field(
        Provenance.DEVICE, "Device-scored central apnea count", default=None
    )
    mixed_apneas: int | None = provenance_field(
        Provenance.DEVICE, "Device-scored mixed apnea count", default=None
    )
    hypopneas: int | None = provenance_field(
        Provenance.DEVICE, "Device-scored hypopnea count", default=None
    )
    reras: int | None = provenance_field(
        Provenance.DEVICE, "Device-scored RERA count", default=None
    )
    flow_limitations: int | None = provenance_field(
        Provenance.DEVICE, "Device-flagged flow limitation count", default=None
    )
    pressure_mean: float | None = provenance_field(
        Provenance.DERIVED, "Mean pressure (cmH2O)", default=None
    )
    pressure_min: float | None = provenance_field(
        Provenance.DERIVED, "Min pressure (cmH2O)", default=None
    )
    pressure_max: float | None = provenance_field(
        Provenance.DERIVED,
        "Max pressure (cmH2O); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    pressure_median: float | None = provenance_field(
        Provenance.DERIVED,
        "Median pressure (cmH2O); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    pressure_95th: float | None = provenance_field(
        Provenance.DERIVED,
        "95th percentile pressure (cmH2O); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    epap_mean: float | None = provenance_field(
        Provenance.DERIVED, "Mean EPAP (cmH2O)", default=None
    )
    epap_min: float | None = provenance_field(
        Provenance.DERIVED, "Min EPAP (cmH2O)", default=None
    )
    epap_max: float | None = provenance_field(
        Provenance.DERIVED,
        "Max EPAP (cmH2O); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    epap_median: float | None = provenance_field(
        Provenance.DERIVED,
        "Median EPAP (cmH2O); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    epap_95th: float | None = provenance_field(
        Provenance.DERIVED,
        "95th percentile EPAP (cmH2O); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    ipap_median: float | None = provenance_field(
        Provenance.DEVICE, "Median IPAP (cmH2O)", default=None
    )
    ipap_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile IPAP (cmH2O)", default=None
    )
    ipap_max: float | None = provenance_field(
        Provenance.DEVICE, "Max IPAP (cmH2O)", default=None
    )
    leak_mean: float | None = provenance_field(
        Provenance.DERIVED, "Mean leak (L/min)", default=None
    )
    leak_min: float | None = provenance_field(
        Provenance.DERIVED, "Min leak (L/min)", default=None
    )
    leak_max: float | None = provenance_field(
        Provenance.DERIVED,
        "Max leak (L/min); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    leak_median: float | None = provenance_field(
        Provenance.DERIVED,
        "Median leak (L/min); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    leak_percentile_70: float | None = provenance_field(
        Provenance.DEVICE, "70th percentile leak (L/min)", default=None
    )
    leak_95th: float | None = provenance_field(
        Provenance.DERIVED,
        "95th percentile leak (L/min); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    spo2_mean: float | None = provenance_field(
        Provenance.DERIVED, "Mean SpO2 (%)", default=None
    )
    spo2_min: float | None = provenance_field(
        Provenance.DERIVED, "Min SpO2 (%)", default=None
    )
    spo2_max: float | None = provenance_field(
        Provenance.DERIVED,
        "Max SpO2 (%); recomputed from the waveform when available, otherwise the device's value",
        default=None,
    )
    spo2_median: float | None = provenance_field(
        Provenance.DEVICE, "Median SpO2 (%)", default=None
    )
    spo2_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile SpO2 (%)", default=None
    )
    spo2_time_below_90: int | None = provenance_field(
        Provenance.DERIVED, "Seconds with SpO2 below 90%", default=None
    )
    pulse_mean: float | None = provenance_field(
        Provenance.DERIVED, "Mean pulse (BPM)", default=None
    )
    pulse_min: float | None = provenance_field(
        Provenance.DERIVED, "Min pulse (BPM)", default=None
    )
    pulse_max: float | None = provenance_field(
        Provenance.DERIVED, "Max pulse (BPM)", default=None
    )
    respiratory_rate_mean: float | None = provenance_field(
        Provenance.DEVICE,
        "Respiratory rate (breaths/min); device STR median on ResMed, OSCAR session average on OSCAR imports",
        default=None,
    )
    respiratory_rate_min: float | None = provenance_field(
        Provenance.DERIVED,
        "Min respiratory rate (breaths/min); OSCAR session summary (OSCAR imports only)",
        default=None,
    )
    respiratory_rate_max: float | None = provenance_field(
        Provenance.DEVICE,
        "Max respiratory rate (breaths/min); device STR value on ResMed, OSCAR session summary on OSCAR imports",
        default=None,
    )
    respiratory_rate_95th: float | None = provenance_field(
        Provenance.DEVICE,
        "95th percentile respiratory rate (breaths/min)",
        default=None,
    )
    tidal_volume_mean: float | None = provenance_field(
        Provenance.DEVICE,
        "Tidal volume (mL); device STR median on ResMed, OSCAR session average on OSCAR imports",
        default=None,
    )
    tidal_volume_min: float | None = provenance_field(
        Provenance.DERIVED,
        "Min tidal volume (mL); OSCAR session summary (OSCAR imports only)",
        default=None,
    )
    tidal_volume_max: float | None = provenance_field(
        Provenance.DEVICE,
        "Max tidal volume (mL); device STR value on ResMed, OSCAR session summary on OSCAR imports",
        default=None,
    )
    tidal_volume_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile tidal volume (mL)", default=None
    )
    minute_ventilation_mean: float | None = provenance_field(
        Provenance.DEVICE,
        "Minute ventilation (L/min); device STR median on ResMed, OSCAR session average on OSCAR imports",
        default=None,
    )
    minute_ventilation_min: float | None = provenance_field(
        Provenance.DERIVED,
        "Min minute ventilation (L/min); OSCAR session summary (OSCAR imports only)",
        default=None,
    )
    minute_ventilation_max: float | None = provenance_field(
        Provenance.DEVICE,
        "Max minute ventilation (L/min); device STR value on ResMed, OSCAR session summary on OSCAR imports",
        default=None,
    )
    minute_ventilation_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile minute ventilation (L/min)", default=None
    )
    uai: float | None = provenance_field(
        Provenance.DEVICE, "Unknown apnea index", default=None
    )
    ai: float | None = provenance_field(Provenance.DEVICE, "Apnea index", default=None)
    rin: float | None = provenance_field(Provenance.DEVICE, "RERA index", default=None)
    csr_pct: float | None = provenance_field(
        Provenance.DEVICE,
        "Percent of session in Cheyne-Stokes respiration",
        default=None,
    )
    spont_cyc_pct: float | None = provenance_field(
        Provenance.DEVICE, "Percent of breaths spontaneously cycled", default=None
    )
    ie_ratio_median: float | None = provenance_field(
        Provenance.DEVICE, "Median I:E ratio", default=None
    )
    ie_ratio_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile I:E ratio", default=None
    )
    ie_ratio_max: float | None = provenance_field(
        Provenance.DEVICE, "Max I:E ratio", default=None
    )
    ti_median: float | None = provenance_field(
        Provenance.DEVICE, "Median inspiratory time (s)", default=None
    )
    ti_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile inspiratory time (s)", default=None
    )
    ti_max: float | None = provenance_field(
        Provenance.DEVICE, "Max inspiratory time (s)", default=None
    )
    flow_5th: float | None = provenance_field(
        Provenance.DEVICE, "5th percentile flow (L/min)", default=None
    )
    flow_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile flow (L/min)", default=None
    )
    blow_press_5th: float | None = provenance_field(
        Provenance.DEVICE, "5th percentile blower pressure (cmH2O)", default=None
    )
    blow_press_95th: float | None = provenance_field(
        Provenance.DEVICE, "95th percentile blower pressure (cmH2O)", default=None
    )
    blow_flow_median: float | None = provenance_field(
        Provenance.DEVICE, "Median blower flow (L/min)", default=None
    )
    amb_humidity_median: float | None = provenance_field(
        Provenance.DEVICE, "Median ambient humidity (%)", default=None
    )
    hum_temp_median: float | None = provenance_field(
        Provenance.DEVICE, "Median humidifier temperature (C)", default=None
    )
    htube_temp_median: float | None = provenance_field(
        Provenance.DEVICE, "Median heated-tube temperature (C)", default=None
    )
    htube_pow_median: float | None = provenance_field(
        Provenance.DEVICE, "Median heated-tube power (%)", default=None
    )
    hum_pow_median: float | None = provenance_field(
        Provenance.DEVICE, "Median humidifier power (%)", default=None
    )
    mask_events: float | None = provenance_field(
        Provenance.DEVICE, "Mask-on events", default=None
    )


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
    duration_hours: float = provenance_field(
        Provenance.DEVICE, "Session duration in hours"
    )
    duration_seconds: float = provenance_field(
        Provenance.DEVICE, "Session duration in seconds"
    )
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


class DeletePreview(BaseModel):
    """Preview of sessions and related data to be deleted."""

    sessions: list[SessionListItem] = Field(description="Sessions to be deleted")
    event_count: int = Field(description="Total events to be deleted")
    waveform_count: int = Field(description="Total waveform records to be deleted")
    stats_count: int = Field(description="Total statistics records to be deleted")


class EventTypeCount(BaseModel):
    """Event type with count and percentage."""

    event_type: str
    count: int = provenance_field(
        Provenance.DEVICE, "Device-scored events of this type"
    )
    percentage: float = provenance_field(
        Provenance.DERIVED, "Share of all device-scored events (%)"
    )


class TherapySummary(BaseModel):
    """Aggregated therapy statistics summary."""

    first_date: date = Field(
        description="First day with therapy hours (first day in range if none)"
    )
    last_date: date = Field(
        description="Last day with therapy hours (last day in range if none)"
    )
    days_since_last: int = provenance_field(
        Provenance.DERIVED, "Days since the last therapy day"
    )
    total_hours: float = provenance_field(Provenance.DERIVED, "Total therapy hours")
    avg_hours: float = provenance_field(
        Provenance.DERIVED, "Average therapy hours per day with usage"
    )
    days_with_data: int = provenance_field(
        Provenance.DERIVED, "Days with therapy hours from enabled sessions"
    )
    avg_ahi: float | None = provenance_field(
        Provenance.DERIVED,
        "Usage-weighted average of daily headline AHI"
        " (can mix device-reported and recounted days)",
        default=None,
    )
    effectiveness: str = provenance_field(
        Provenance.DERIVED,
        "Therapy effectiveness band from average AHI",
        default="unknown",
    )
    avg_rei: float | None = provenance_field(
        Provenance.DERIVED, "Average REI", default=None
    )
    avg_pressure: float | None = provenance_field(
        Provenance.DERIVED, "Average pressure (cmH2O)", default=None
    )
    min_pressure: float | None = provenance_field(
        Provenance.DERIVED, "Minimum pressure (cmH2O)", default=None
    )
    max_pressure: float | None = provenance_field(
        Provenance.DERIVED, "Maximum pressure (cmH2O)", default=None
    )
    avg_epap: float | None = provenance_field(
        Provenance.DERIVED, "Average EPAP (cmH2O)", default=None
    )
    avg_leak: float | None = provenance_field(
        Provenance.DERIVED, "Average leak (L/min)", default=None
    )
    avg_spo2: float | None = provenance_field(
        Provenance.DERIVED, "Average SpO2 (%)", default=None
    )
    min_spo2: float | None = provenance_field(
        Provenance.DERIVED, "Minimum SpO2 (%)", default=None
    )
    total_spo2_time_below_90: int = provenance_field(
        Provenance.DERIVED, "Total seconds with SpO2 below 90%", default=0
    )
    avg_pulse: float | None = provenance_field(
        Provenance.DERIVED, "Average pulse (BPM)", default=None
    )
    avg_respiratory_rate: float | None = provenance_field(
        Provenance.DERIVED, "Average respiratory rate (breaths/min)", default=None
    )
    avg_tidal_volume: float | None = provenance_field(
        Provenance.DERIVED, "Average tidal volume (mL)", default=None
    )
    avg_minute_ventilation: float | None = provenance_field(
        Provenance.DERIVED, "Average minute ventilation (L/min)", default=None
    )
    ahi_trend_direction: str | None = provenance_field(
        Provenance.DERIVED, "AHI trend direction over the range", default=None
    )
    event_counts: list[EventTypeCount] = Field(default_factory=list)


class WaveformInfo(BaseModel):
    """Waveform metadata for listing."""

    waveform_type: str
    sample_rate: float = provenance_field(Provenance.DEVICE, "Sample rate (Hz)")
    sample_count: int = provenance_field(Provenance.DEVICE, "Number of samples")
    unit: str | None = None
    duration_hours: float = provenance_field(
        Provenance.DEVICE, "Recorded duration in hours"
    )


class EventMatchResult(BaseModel):
    """Result of matching machine vs programmatic events."""

    machine_count: int = provenance_field(
        Provenance.DEVICE, "Machine-scored apneas and hypopneas"
    )
    programmatic_count: int = provenance_field(
        Provenance.EXPERIMENTAL, "Programmatically detected apneas and hypopneas"
    )
    matched: int = provenance_field(
        Provenance.EXPERIMENTAL,
        "Programmatic/machine event pairs matched one-to-one within tolerance "
        "(each event in at most one pair)",
    )
    false_positives: int = provenance_field(
        Provenance.EXPERIMENTAL, "Programmatic events unmatched"
    )
    false_negatives: int = provenance_field(
        Provenance.EXPERIMENTAL, "Machine events unmatched"
    )


class AnalysisListItem(BaseModel):
    """Session with analysis status for listing."""

    session_id: int
    session_date: date
    duration_hours: float | None = provenance_field(
        Provenance.DEVICE, "Session duration in hours", default=None
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
    total_therapy_hours: float = provenance_field(
        Provenance.DERIVED,
        "Total therapy hours from enabled sessions, summed over the device's days "
        "(mask-on time, falling back to session span)",
    )
    therapy_modes: list[str]


class DeviceDetail(DeviceInfo):
    """Full device detail including usage summary, current settings, and settings history."""

    usage: DeviceUsageSummary
    current_settings: dict[str, str] | None
    settings_history: list[SettingsChange]


class HealthNightSummaryRead(BaseModel):
    """Derived nightly sleep summary from Apple Health data."""

    model_config = ConfigDict(from_attributes=True)

    night_date: date
    preferred_source: str | None = None
    time_in_bed_seconds: float | None = provenance_field(
        Provenance.DERIVED, "Time in bed (s)", default=None
    )
    total_sleep_seconds: float | None = provenance_field(
        Provenance.DERIVED, "Total sleep (s)", default=None
    )
    core_seconds: float | None = provenance_field(
        Provenance.DERIVED, "Core sleep (s)", default=None
    )
    deep_seconds: float | None = provenance_field(
        Provenance.DERIVED, "Deep sleep (s)", default=None
    )
    rem_seconds: float | None = provenance_field(
        Provenance.DERIVED, "REM sleep (s)", default=None
    )
    awake_seconds: float | None = provenance_field(
        Provenance.DERIVED, "Awake time (s)", default=None
    )
    unspecified_seconds: float | None = provenance_field(
        Provenance.DERIVED, "Asleep, unspecified stage (s)", default=None
    )
    sleep_efficiency_pct: float | None = provenance_field(
        Provenance.DERIVED, "Total sleep / time in bed (%)", default=None
    )
    stage_coverage_pct: float | None = provenance_field(
        Provenance.DERIVED, "Share of sleep with a known stage (%)", default=None
    )
    computed_at: datetime


class HealthNightDetailRead(HealthNightSummaryRead):
    """Nightly sleep summary with aggregated oximetry and respiratory rate metrics."""

    avg_spo2_pct: float | None = provenance_field(
        Provenance.DERIVED, "Average SpO2 (%)", default=None
    )
    min_spo2_pct: float | None = provenance_field(
        Provenance.DERIVED, "Minimum SpO2 (%)", default=None
    )
    avg_rr: float | None = provenance_field(
        Provenance.DERIVED, "Average respiratory rate (breaths/min)", default=None
    )


class HealthSampleRead(BaseModel):
    """Single Apple Health sample (sleep stage or quantity record)."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    record_type: str
    source_name: str
    start_time: datetime
    end_time: datetime
    value_text: str | None = None
    value_num: float | None = provenance_field(
        Provenance.DEVICE, "Numeric sample value as recorded", default=None
    )
    unit: str | None = None
    night_date: date


class DayListItem(BaseModel):
    """Summary of a single therapy day."""

    model_config = ConfigDict(from_attributes=True)

    date: date
    device_id: int
    session_count: int
    total_therapy_hours: float | None = provenance_field(
        Provenance.DERIVED,
        "Total mask-on hours across the day's sessions",
        default=None,
    )
    ahi: float | None = provenance_field(
        Provenance.DEVICE,
        f"Headline AHI (events/hr): {HEADLINE_INDEX_DESCRIPTION}",
        source_field="index_source",
        default=None,
    )
    index_source: IndexSource | None = Field(
        default=None, description=INDEX_SOURCE_DESCRIPTION
    )
    ahi_computed: float | None = provenance_field(
        Provenance.DERIVED,
        "SNORE recount AHI: device-scored events over SNORE mask-on hours, "
        "usage-weighted across sessions; null when there are no mask-on hours "
        "or no session statistics",
        default=None,
    )


class DayDetail(DayListItem):
    """Full detail of a therapy day including per-metric stats.

    The FL/RERA proxy fields below carry ``*_reason`` companions (unlike the
    older nullable stats) because null here is ambiguous — "analysis not run"
    versus a genuine zero — so a companion code disambiguates, mirroring the
    MCP nightly-summary null-with-reason convention.
    """

    oai: float | None = provenance_field(
        Provenance.DEVICE,
        "Headline OAI: same source rule as ahi",
        source_field="index_source",
        default=None,
    )
    cai: float | None = provenance_field(
        Provenance.DEVICE,
        "Headline CAI: same source rule as ahi",
        source_field="index_source",
        default=None,
    )
    hi: float | None = provenance_field(
        Provenance.DEVICE,
        "Headline HI: same source rule as ahi",
        source_field="index_source",
        default=None,
    )
    oai_computed: float | None = provenance_field(
        Provenance.DERIVED, "SNORE recount OAI (see ahi_computed)", default=None
    )
    cai_computed: float | None = provenance_field(
        Provenance.DERIVED, "SNORE recount CAI (see ahi_computed)", default=None
    )
    hi_computed: float | None = provenance_field(
        Provenance.DERIVED, "SNORE recount HI (see ahi_computed)", default=None
    )
    avg_pressure: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted mean pressure (cmH2O)", default=None
    )
    avg_leak: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted mean leak (L/min)", default=None
    )
    avg_spo2: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted mean SpO2 (%)", default=None
    )
    # Pressure detail
    pressure_min: float | None = provenance_field(
        Provenance.DERIVED, "Min pressure (cmH2O)", default=None
    )
    pressure_max: float | None = provenance_field(
        Provenance.DERIVED, "Max pressure (cmH2O)", default=None
    )
    pressure_median: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted median pressure (cmH2O)", default=None
    )
    pressure_95th: float | None = provenance_field(
        Provenance.DERIVED,
        "Usage-weighted 95th percentile pressure (cmH2O)",
        default=None,
    )
    # EPAP detail
    epap_min: float | None = provenance_field(
        Provenance.DERIVED, "Min EPAP (cmH2O)", default=None
    )
    epap_max: float | None = provenance_field(
        Provenance.DERIVED, "Max EPAP (cmH2O)", default=None
    )
    epap_median: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted median EPAP (cmH2O)", default=None
    )
    epap_mean: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted mean EPAP (cmH2O)", default=None
    )
    epap_95th: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted 95th percentile EPAP (cmH2O)", default=None
    )
    # Leak detail
    leak_min: float | None = provenance_field(
        Provenance.DERIVED, "Min leak (L/min)", default=None
    )
    leak_max: float | None = provenance_field(
        Provenance.DERIVED, "Max leak (L/min)", default=None
    )
    leak_mean: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted mean leak (L/min)", default=None
    )
    leak_95th: float | None = provenance_field(
        Provenance.DERIVED, "Usage-weighted 95th percentile leak (L/min)", default=None
    )
    # SpO2 detail
    spo2_min: float | None = provenance_field(
        Provenance.DERIVED, "Min SpO2 (%)", default=None
    )
    spo2_max: float | None = provenance_field(
        Provenance.DERIVED, "Max SpO2 (%)", default=None
    )
    # Raw event counts
    obstructive_apneas: int = provenance_field(
        Provenance.DEVICE,
        "Device-scored obstructive apnea count for the night",
        default=0,
    )
    central_apneas: int = provenance_field(
        Provenance.DEVICE, "Device-scored central apnea count for the night", default=0
    )
    hypopneas: int = provenance_field(
        Provenance.DEVICE, "Device-scored hypopnea count for the night", default=0
    )
    reras: int = provenance_field(
        Provenance.DEVICE, "Device-scored RERA count for the night", default=0
    )
    # Nightly breath-analysis proxy metrics, sourced read-time from
    # BreathService (same path as the MCP nightly summary).  Reason semantics:
    # an ordinary un-analyzed night (sessions present, none with an OK analysis)
    # is null with "not_available"; a lookup failure (no sessions for the
    # device, breath-table DB error, device resolution declining) is null with
    # "analysis_not_run".  Day detail never fails on missing breath analysis.
    fl_class_ge4_pct: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Percent of rule-classified breaths flagged flow-class >= 4 "
        "(flow-limitation proxy).",
        default=None,
    )
    fl_class_ge4_pct_reason: str | None = None
    rera_index: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "RERA-proxy events per hour of analyzed sessions (FL-run proxy, "
        "not device-scored).",
        default=None,
    )
    rera_index_reason: str | None = None
    rera_count: int | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "RERA-proxy count from flow-limitation runs ending in a recovery breath; "
        "distinct from device-scored `reras`.",
        default=None,
    )
    rera_count_reason: str | None = None
    session_ids: list[int] = Field(default_factory=list)
    health_sleep: HealthNightSummaryRead | None = None


class RxPeriodResponse(BaseModel):
    """Single therapy prescription period with aggregated stats."""

    settings: dict[str, str]
    start_date: date
    end_date: date
    days_count: int
    avg_ahi: float | None = provenance_field(
        Provenance.DERIVED,
        "Usage-hours-weighted average of daily headline AHI over the period"
        " (can mix device-reported and recounted days)",
        default=None,
    )
    median_ahi: float | None = provenance_field(
        Provenance.DERIVED,
        "Median of daily headline AHI over the period"
        " (can mix device-reported and recounted days)",
        default=None,
    )
    avg_hours: float | None = provenance_field(
        Provenance.DERIVED, "Average therapy hours per day with usage", default=None
    )
    total_hours: float = provenance_field(
        Provenance.DERIVED, "Total therapy hours", default=0.0
    )
    avg_leak: float | None = provenance_field(
        Provenance.DERIVED, "Usage-hours-weighted average leak (L/min)", default=None
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
    duration: float = provenance_field(
        Provenance.DEVICE,
        "Event duration (seconds); experimental for programmatic events",
        source_field="source",
    )
    source: Literal["machine", "programmatic"] = Field(
        description="Event source (machine/programmatic)"
    )
    confidence: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Detection confidence (programmatic events only)",
        default=None,
    )
    flow_reduction: float | None = provenance_field(
        Provenance.EXPERIMENTAL,
        "Flow reduction fraction (programmatic events only)",
        default=None,
    )


class EventComparisonResult(BaseModel):
    """Result of comparing machine vs programmatic events for a session."""

    session_id: int = Field(description="Session database ID")
    mode: str = Field(description="Detection mode used (e.g., 'aasm')")
    machine_event_count: int = provenance_field(
        Provenance.DEVICE, "Total machine-detected events"
    )
    programmatic_event_count: int = provenance_field(
        Provenance.EXPERIMENTAL, "Total programmatically-detected events"
    )
    false_negatives: list[EventComparisonDetail] = provenance_field(
        Provenance.EXPERIMENTAL,
        "Machine events missed by programmatic detection",
        default_factory=list,
    )
    false_positives_apnea: list[EventComparisonDetail] = provenance_field(
        Provenance.EXPERIMENTAL,
        "Programmatic apneas not in machine events",
        default_factory=list,
    )
    false_positives_hypopnea: list[EventComparisonDetail] = provenance_field(
        Provenance.EXPERIMENTAL,
        "Programmatic hypopneas not in machine events",
        default_factory=list,
    )


class VacuumResult(BaseModel):
    """Result of a database vacuum operation."""

    status: str = Field(description="Operation status ('success')")
    size_before_mb: float = Field(description="Database size before vacuum in MB")
    size_after_mb: float = Field(description="Database size after vacuum in MB")


class ResetResult(BaseModel):
    """Result of a database reset (delete all data, preserve schema) operation."""

    status: str = Field(description="Operation status ('success')")
    tables_cleared: dict[str, int] = Field(
        description="Rows deleted per table (table_name -> count)"
    )
    total_rows_deleted: int = Field(description="Total rows deleted across all tables")
    size_before_mb: float = Field(description="Database size before reset in MB")
    size_after_mb: float | None = Field(
        default=None,
        description=(
            "Database size after vacuum in MB. Null when vacuum_scheduled=true — the"
            " vacuum is still running as a post-response background task."
        ),
    )
    vacuum_scheduled: bool = Field(
        default=False,
        description=(
            "True when VACUUM has been queued as a post-response background task."
            " size_after_mb will be null in this case."
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

    ahi: TrendSeries = provenance_field(
        Provenance.DERIVED,
        "Average daily headline AHI per period"
        " (can mix device-reported and recounted days)",
    )
    usage: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average therapy hours per day, per period"
    )
    spo2: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average SpO2 (%) per period"
    )
    leak: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average leak (L/min) per period"
    )
    pressure: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average pressure (cmH2O) per period"
    )
    oai: TrendSeries = provenance_field(
        Provenance.DERIVED,
        "Average daily headline OAI per period"
        " (can mix device-reported and recounted days)",
    )
    cai: TrendSeries = provenance_field(
        Provenance.DERIVED,
        "Average daily headline CAI per period"
        " (can mix device-reported and recounted days)",
    )
    hi: TrendSeries = provenance_field(
        Provenance.DERIVED,
        "Average daily headline HI per period"
        " (can mix device-reported and recounted days)",
    )
    rera: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average device-scored RERA index per period"
    )
    epap: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average EPAP (cmH2O) per period"
    )
    rr: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average respiratory rate (breaths/min) per period"
    )
    pulse: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average pulse (BPM) per period"
    )
    mv: TrendSeries = provenance_field(
        Provenance.DERIVED, "Average minute ventilation (L/min) per period"
    )
    # Apple Health series: keys are omitted (not null) when no night has data.
    total_sleep_hours: TrendSeries | None = provenance_field(
        Provenance.DERIVED,
        "Average total sleep hours per night, per period (Apple Health)",
        default=None,
    )
    sleep_efficiency: TrendSeries | None = provenance_field(
        Provenance.DERIVED,
        "Average sleep efficiency % per night, per period (Apple Health)",
        default=None,
    )


class RecordExtremes(BaseModel):
    """Top-N best and worst ``[date, value]`` days for one metric."""

    best: list[tuple[date, float]] = provenance_field(
        Provenance.DERIVED, "Best days, best first"
    )
    worst: list[tuple[date, float]] = provenance_field(
        Provenance.DERIVED, "Worst days, worst first"
    )


class RecordsResponse(BaseModel):
    """Best/worst days for ``GET /stats/records``.

    Keys are omitted (not null) when no qualifying day (>= 1 h therapy) has the
    metric.
    """

    ahi: RecordExtremes | None = provenance_field(
        Provenance.DERIVED,
        "Daily headline AHI (device-reported when trusted, else SNORE's"
        " recount; records can mix both)",
        default=None,
    )
    leak: RecordExtremes | None = provenance_field(
        Provenance.DERIVED, "Nightly median leak (L/min)", default=None
    )
    therapy_hours: RecordExtremes | None = provenance_field(
        Provenance.DERIVED, "Nightly therapy hours", default=None
    )
    spo2_min: RecordExtremes | None = provenance_field(
        Provenance.DERIVED, "Nightly minimum SpO2 (%)", default=None
    )
    total_sleep_hours: RecordExtremes | None = provenance_field(
        Provenance.DERIVED, "Nightly total sleep hours (Apple Health)", default=None
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
    size_after_mb: float | None = Field(
        default=None,
        description=(
            "Database size after vacuum in MB. Null when vacuum_scheduled=true — the"
            " vacuum is still running as a post-response background task."
        ),
    )
    vacuum_scheduled: bool = Field(
        default=False,
        description=(
            "True when VACUUM has been queued as a post-response background task."
            " size_after_mb will be null in this case."
        ),
    )
