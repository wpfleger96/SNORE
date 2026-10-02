"""
Unified Data Model for CPAP Data Platform

This module defines the universal data structures that ALL parsers must convert their
data into, regardless of manufacturer or file format. This enables complete separation
between the parser layer and the rest of the system.

Key Principle: The database layer and analysis tools only work with these
unified structures - they never see parser-specific formats.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, model_validator

from snore.therapy_hours import TherapyHoursBasis, therapy_hours


def extract_basic_stats(values: np.ndarray) -> tuple[float, float, float]:
    """Compute (min, max, mean) summary statistics for a waveform array."""
    return float(np.min(values)), float(np.max(values)), float(np.mean(values))


class RespiratoryEventType(Enum):
    """Universal respiratory event types across all devices."""

    OBSTRUCTIVE_APNEA = "OA"
    CENTRAL_APNEA = "CA"
    MIXED_APNEA = "MA"
    HYPOPNEA = "H"
    RERA = "RE"  # Respiratory Effort Related Arousal
    FLOW_LIMITATION = "FL"
    PERIODIC_BREATHING = "PB"
    LARGE_LEAK = "LL"
    CLEAR_AIRWAY = "CAA"  # Changed from "CA" to avoid collision with CENTRAL_APNEA
    VIBRATORY_SNORE = "VS"
    UNCLASSIFIED = "UC"
    UNCLASSIFIED_APNEA = "UA"  # Apnea without OA/CA classification


class WaveformType(Enum):
    """Universal waveform/signal types."""

    FLOW_RATE = "flow"  # L/min
    MASK_PRESSURE = "pressure"  # cmH2O
    THERAPY_PRESSURE = "therapy_pressure"  # cmH2O
    EPAP = "epap"  # cmH2O
    LEAK_RATE = "leak"  # L/min
    MINUTE_VENTILATION = "mv"  # L/min
    RESPIRATORY_RATE = "rr"  # breaths/min
    TIDAL_VOLUME = "tv"  # mL
    SPO2 = "spo2"  # %
    PULSE = "pulse"  # BPM
    FLOW_LIMITATION = "fl"  # arbitrary units
    SNORE = "snore"  # arbitrary units
    IE_RATIO = "ie_ratio"  # % (inspiratory-to-expiratory ratio, VAuto)
    TI = "ti"  # seconds (inspiratory time, VAuto)
    PRESSURE_HR = "pressure_hr"  # cmH2O (high-rate mask pressure from BRP, 25 Hz)
    TRIGGER_CYCLE = (
        "trigger_cycle"  # raw codes 0–16 (VAuto trigger/cycle events, verbatim)
    )


class TherapyMode(Enum):
    """Universal therapy mode types."""

    CPAP = "CPAP"  # Fixed pressure
    APAP = "APAP"  # Auto-adjusting pressure
    BIPAP = "BiPAP"  # Bi-level
    BIPAP_ST = "BiPAP S/T"  # Spontaneous/Timed
    BIPAP_AUTO = "BiPAP Auto"  # Auto bi-level
    ASV = "ASV"  # Adaptive servo-ventilation
    ASV_AUTO = "ASV Auto"  # ASV with variable EPAP (AirCurve 10/11 ASV-A)
    IVAPS = "iVAPS"  # Intelligent Volume-Assured Pressure Support


class DeviceInfo(BaseModel):
    """Universal device information - same structure for all manufacturers."""

    manufacturer: str = Field(description="Manufacturer (e.g., 'ResMed', 'Philips')")
    model: str = Field(description="Device model")
    serial_number: str = Field(description="Device serial number")

    firmware_version: str | None = Field(default=None, description="Firmware version")
    hardware_version: str | None = Field(default=None, description="Hardware version")
    product_code: str | None = Field(default=None, description="Product code")
    manufacturing_date: datetime | None = Field(
        default=None, description="Manufacturing date"
    )


class TherapySettings(BaseModel):
    """Universal therapy settings across all devices."""

    mode: TherapyMode = Field(description="Therapy mode")

    pressure_min: float | None = Field(default=None, description="Minimum pressure")
    pressure_max: float | None = Field(default=None, description="Maximum pressure")
    pressure_fixed: float | None = Field(
        default=None, description="Fixed pressure (CPAP mode)"
    )

    ipap: float | None = Field(default=None, description="Inspiratory pressure")
    epap: float | None = Field(default=None, description="Expiratory pressure")
    ps: float | None = Field(default=None, description="Pressure support (IPAP - EPAP)")

    epr_level: int | None = Field(
        default=None, ge=0, le=3, description="EPR level (0-3)"
    )
    ramp_time: int | None = Field(default=None, ge=0, description="Ramp time (minutes)")
    ramp_start_pressure: float | None = Field(
        default=None, description="Ramp start pressure"
    )

    humidity_level: int | None = Field(default=None, description="Humidity level")
    tube_temp: float | None = Field(
        default=None, description="Tube temperature (stored in °C)"
    )

    mask_type: str | None = Field(default=None, description="Mask type")

    epr_mode: str | None = Field(
        default=None, description="EPR mode: Off, Ramp Only, or Full Time"
    )
    ramp_enabled: bool | None = Field(default=None, description="Ramp enabled")
    humidity_enabled: bool | None = Field(default=None, description="Humidity enabled")
    tube_temp_enabled: bool | None = Field(
        default=None, description="Heated tube enabled"
    )
    climate_control: str | None = Field(
        default=None, description="Climate control mode: Manual or Auto"
    )
    smart_start: bool | None = Field(default=None, description="Smart start enabled")
    ab_filter: str | None = Field(
        default=None, description="Filter type: Standard or Antibacterial"
    )

    other_settings: dict[str, str] = Field(
        default_factory=dict, description="Other settings"
    )


class SessionStatistics(BaseModel):
    """Universal session statistics."""

    obstructive_apneas: int = Field(default=0, ge=0, description="OA count")
    central_apneas: int = Field(default=0, ge=0, description="CA count")
    mixed_apneas: int = Field(default=0, ge=0, description="MA count")
    hypopneas: int = Field(default=0, ge=0, description="Hypopnea count")
    reras: int = Field(default=0, ge=0, description="RERA count")
    flow_limitations: int = Field(default=0, ge=0, description="FL count")

    ahi: float | None = Field(default=None, ge=0, description="Apnea-Hypopnea Index")
    oai: float | None = Field(default=None, ge=0, description="Obstructive Apnea Index")
    cai: float | None = Field(default=None, ge=0, description="Central Apnea Index")
    hi: float | None = Field(default=None, ge=0, description="Hypopnea Index")
    rei: float | None = Field(default=None, ge=0, description="Respiratory Event Index")

    # Device-reported (ResMed STR) indices, preserved alongside the computed
    # ahi/oai/cai/hi above (which finalize_statistics recomputes from events).
    ahi_device: float | None = Field(
        default=None, ge=0, description="Device-reported AHI (from STR)"
    )
    oai_device: float | None = Field(
        default=None, ge=0, description="Device-reported OAI (from STR)"
    )
    cai_device: float | None = Field(
        default=None, ge=0, description="Device-reported CAI (from STR)"
    )
    hi_device: float | None = Field(
        default=None, ge=0, description="Device-reported HI (from STR)"
    )
    usage_hours_device: float | None = Field(
        default=None,
        ge=0,
        description="Device-reported mask-on hours for the whole day (from STR), "
        "repeated on every session of the day",
    )

    pressure_min: float | None = Field(default=None, description="Minimum pressure")
    pressure_max: float | None = Field(default=None, description="Maximum pressure")
    pressure_median: float | None = Field(default=None, description="Median pressure")
    pressure_mean: float | None = Field(default=None, description="Mean pressure")
    pressure_95th: float | None = Field(
        default=None, description="95th percentile pressure"
    )

    epap_min: float | None = Field(default=None, description="Minimum EPAP")
    epap_max: float | None = Field(default=None, description="Maximum EPAP")
    epap_median: float | None = Field(default=None, description="Median EPAP")
    epap_mean: float | None = Field(default=None, description="Mean EPAP")
    epap_95th: float | None = Field(default=None, description="95th percentile EPAP")

    ipap_median: float | None = Field(default=None, description="Median IPAP")
    ipap_95th: float | None = Field(default=None, description="95th percentile IPAP")
    ipap_max: float | None = Field(default=None, description="Maximum IPAP")

    leak_min: float | None = Field(default=None, description="Minimum leak")
    leak_max: float | None = Field(default=None, description="Maximum leak")
    leak_median: float | None = Field(default=None, description="Median leak")
    leak_mean: float | None = Field(default=None, description="Mean leak")
    leak_95th: float | None = Field(default=None, description="95th percentile leak")
    leak_percentile_70: float | None = Field(
        default=None, description="70th percentile leak"
    )

    respiratory_rate_min: float | None = Field(default=None, description="Min RR")
    respiratory_rate_max: float | None = Field(default=None, description="Max RR")
    respiratory_rate_mean: float | None = Field(default=None, description="Mean RR")

    tidal_volume_min: float | None = Field(
        default=None, description="Min tidal volume (mL)"
    )
    tidal_volume_max: float | None = Field(default=None, description="Max tidal volume")
    tidal_volume_mean: float | None = Field(
        default=None, description="Mean tidal volume"
    )

    minute_ventilation_min: float | None = Field(
        default=None, description="Min MV (L/min)"
    )
    minute_ventilation_max: float | None = Field(default=None, description="Max MV")
    minute_ventilation_mean: float | None = Field(default=None, description="Mean MV")

    spo2_min: float | None = Field(default=None, description="Min SpO2 (%)")
    spo2_max: float | None = Field(default=None, description="Max SpO2")
    spo2_mean: float | None = Field(default=None, description="Mean SpO2")
    spo2_median: float | None = Field(
        default=None, description="Median SpO2 (from STR)"
    )
    spo2_95th: float | None = Field(
        default=None, description="95th percentile SpO2 (from STR)"
    )
    spo2_time_below_90: int | None = Field(
        default=None, description="Time below 90% (seconds)"
    )

    pulse_min: float | None = Field(default=None, description="Min pulse (BPM)")
    pulse_max: float | None = Field(default=None, description="Max pulse")
    pulse_mean: float | None = Field(default=None, description="Mean pulse")

    usage_hours: float | None = Field(
        default=None, ge=0, description="Usage time (hours)"
    )

    # --- STR daily summary extras ---
    uai: float | None = Field(
        default=None, ge=0, description="Unclassified Apnea Index (from STR)"
    )
    ai: float | None = Field(
        default=None, ge=0, description="Apnea Index — all apneas (from STR)"
    )
    rin: float | None = Field(
        default=None, ge=0, description="RIN (APAP-only, from STR)"
    )
    csr_pct: float | None = Field(
        default=None, ge=0, description="Cheyne-Stokes % time (APAP-only, from STR)"
    )
    spont_cyc_pct: float | None = Field(
        default=None, ge=0, description="Spontaneous cycle % (VAuto-only, from STR)"
    )
    respiratory_rate_95th: float | None = Field(
        default=None, description="95th percentile respiratory rate (from STR)"
    )
    tidal_volume_95th: float | None = Field(
        default=None, description="95th percentile tidal volume (from STR)"
    )
    minute_ventilation_95th: float | None = Field(
        default=None, description="95th percentile minute ventilation (from STR)"
    )
    ie_ratio_median: float | None = Field(
        default=None, description="Median I:E ratio (VAuto-only, from STR)"
    )
    ie_ratio_95th: float | None = Field(
        default=None, description="95th percentile I:E ratio (VAuto-only, from STR)"
    )
    ie_ratio_max: float | None = Field(
        default=None, description="Max I:E ratio (VAuto-only, from STR)"
    )
    ti_median: float | None = Field(
        default=None, description="Median inspiratory time in s (VAuto-only, from STR)"
    )
    ti_95th: float | None = Field(
        default=None,
        description="95th percentile inspiratory time in s (VAuto-only, from STR)",
    )
    ti_max: float | None = Field(
        default=None,
        description="Max inspiratory time in s (VAuto-only, from STR)",
    )
    flow_5th: float | None = Field(
        default=None, description="5th percentile flow (from STR)"
    )
    flow_95th: float | None = Field(
        default=None, description="95th percentile flow (from STR)"
    )
    blow_press_5th: float | None = Field(
        default=None, description="5th percentile blow pressure (from STR)"
    )
    blow_press_95th: float | None = Field(
        default=None, description="95th percentile blow pressure (from STR)"
    )
    blow_flow_median: float | None = Field(
        default=None, description="Median blow flow (from STR)"
    )
    amb_humidity_median: float | None = Field(
        default=None, description="Median ambient humidity (from STR)"
    )
    hum_temp_median: float | None = Field(
        default=None, description="Median humidifier temperature (from STR)"
    )
    htube_temp_median: float | None = Field(
        default=None, description="Median heated tube temperature (from STR)"
    )
    htube_pow_median: float | None = Field(
        default=None, description="Median heated tube power (from STR)"
    )
    hum_pow_median: float | None = Field(
        default=None, description="Median humidifier power (from STR)"
    )
    mask_events: float | None = Field(
        default=None, ge=0, description="Mask-on event count (from STR)"
    )


class RespiratoryEvent(BaseModel):
    """A single respiratory event (apnea, hypopnea, etc.).

    ``start_time`` is the true start of the event. Parsers that read device
    formats which flag events at their end (ResMed EVE EDF annotations, OSCAR
    binary event lists) normalize to the true start by subtracting the
    duration at parse time.
    """

    event_type: RespiratoryEventType = Field(description="Event type")
    start_time: datetime = Field(description="True start time of the event")
    duration_seconds: float = Field(ge=0, description="Event duration (seconds)")

    peak_flow_limitation: float | None = Field(
        default=None, description="Peak FL value"
    )
    spo2_drop: float | None = Field(default=None, description="SpO2 drop (%)")
    end_time: datetime | None = Field(default=None, description="Event end time")


class WaveformData(BaseModel):
    """
    Time-series waveform data for a single channel.

    Timestamps are stored as numpy arrays of seconds offset from session start (float32).
    Values are stored as float32 numpy arrays for memory efficiency.

    A typical 8-hour session at 25Hz has 720,000 samples per channel:
    - Numpy float32 arrays: 2.7 MB per waveform
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    waveform_type: WaveformType = Field(description="Waveform type")
    sample_rate: float = Field(ge=0, description="Sample rate (Hz)")
    unit: str = Field(description="Units (e.g., 'L/min', 'cmH2O')")

    timestamps: list[float] | np.ndarray = Field(
        description="Seconds from session start"
    )
    values: list[float] | np.ndarray = Field(description="Waveform values")

    min_value: float | None = Field(default=None, description="Minimum value")
    max_value: float | None = Field(default=None, description="Maximum value")
    mean_value: float | None = Field(default=None, description="Mean value")

    @model_validator(mode="after")
    def convert_to_numpy(self) -> WaveformData:
        """Convert lists to numpy arrays for efficiency."""
        if isinstance(self.timestamps, list):
            self.timestamps = np.array(self.timestamps, dtype=np.float32)
        elif (
            isinstance(self.timestamps, np.ndarray)
            and self.timestamps.dtype != np.float32
        ):
            self.timestamps = self.timestamps.astype(np.float32)

        if isinstance(self.values, list):
            self.values = np.array(self.values, dtype=np.float32)
        elif isinstance(self.values, np.ndarray) and self.values.dtype != np.float32:
            self.values = self.values.astype(np.float32)

        return self

    @property
    def duration_seconds(self) -> float:
        """Calculate duration from timestamps."""
        if len(self.timestamps) < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def sample_count(self) -> int:
        """Number of samples in this waveform."""
        return len(self.values)


class UnifiedSession(BaseModel):
    """
    Universal session format that ALL parsers must produce.

    This is the lingua franca of the CPAP data platform - every parser
    converts its native format into this structure, and all downstream
    components (database, analysis) work exclusively with this.
    """

    session_id: UUID = Field(default_factory=uuid4, description="Internal session ID")
    device_session_id: str = Field(default="", description="Device session ID")

    device_info: DeviceInfo = Field(description="Device information")

    start_time: datetime = Field(description="Session start time")
    end_time: datetime = Field(description="Session end time")

    settings: TherapySettings | None = Field(
        default=None, description="Therapy settings"
    )
    statistics: SessionStatistics = Field(
        default_factory=SessionStatistics, description="Session statistics"
    )

    waveforms: dict[WaveformType, WaveformData] = Field(
        default_factory=dict, description="Waveform data by type"
    )
    events: list[RespiratoryEvent] = Field(
        default_factory=list, description="Respiratory events"
    )

    import_source: str = Field(default="", description="Parser ID")
    import_date: datetime = Field(
        default_factory=datetime.now, description="Import timestamp"
    )
    raw_data_path: str | None = Field(
        default=None, description="Path to original files"
    )
    parser_version: str = Field(default="", description="Parser version")

    has_waveform_data: bool = Field(default=False, description="Has waveform data")
    has_event_data: bool = Field(default=False, description="Has event data")
    has_statistics: bool = Field(default=False, description="Has statistics")
    mask_on_segments: list[tuple[float, float]] | None = Field(
        default=None,
        description=(
            "Ascending, disjoint [start_offset_s, end_offset_s] mask-on intervals in "
            "merged-session offset seconds. A single-segment session stores "
            "[(0.0, duration)] so 'known, no gaps' is distinguishable from "
            "None = unknown (e.g. OSCAR imports)."
        ),
    )
    data_quality_notes: list[str] = Field(
        default_factory=list, description="Data quality warnings"
    )

    @model_validator(mode="after")
    def validate_session(self) -> UnifiedSession:
        """Validate session data after initialization."""
        errors = []

        if self.end_time <= self.start_time:
            errors.append(
                f"end_time ({self.end_time}) must be after start_time ({self.start_time})"
            )

        duration_hours = self.duration_hours
        if duration_hours > 24:
            self.data_quality_notes.append(
                f"Warning: Unusually long session duration: {duration_hours:.1f} hours"
            )
        if duration_hours < 0:
            errors.append(f"Negative session duration: {duration_hours:.1f} hours")

        for waveform_type, waveform in self.waveforms.items():
            if waveform.timestamps is not None and len(waveform.timestamps) > 0:
                if isinstance(waveform.timestamps, np.ndarray):
                    duration = self.duration_seconds
                    first_offset = float(waveform.timestamps[0])
                    last_offset = float(waveform.timestamps[-1])

                    tolerance = 1.0  # seconds
                    if first_offset < -tolerance:
                        errors.append(
                            f"{waveform_type.value}: first timestamp offset {first_offset:.2f}s is negative"
                        )
                    if last_offset > duration + tolerance:
                        errors.append(
                            f"{waveform_type.value}: last timestamp offset {last_offset:.2f}s exceeds session duration {duration:.2f}s"
                        )
                elif (
                    isinstance(waveform.timestamps, list)
                    and len(waveform.timestamps) > 0
                ):
                    first_ts = waveform.timestamps[0]
                    last_ts = waveform.timestamps[-1]

                    if isinstance(first_ts, datetime):
                        tolerance = timedelta(seconds=1)
                        if first_ts < self.start_time - tolerance:
                            errors.append(
                                f"{waveform_type.value}: first timestamp {first_ts} before session start {self.start_time}"
                            )
                        if last_ts > self.end_time + tolerance:
                            errors.append(
                                f"{waveform_type.value}: last timestamp {last_ts} after session end {self.end_time}"
                            )

        if errors:
            raise ValueError(
                "Session validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        return self

    @property
    def duration_hours(self) -> float:
        """Calculate session duration in hours."""
        delta = self.end_time - self.start_time
        return delta.total_seconds() / 3600.0

    @property
    def duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        delta = self.end_time - self.start_time
        return delta.total_seconds()

    def add_waveform(self, waveform: WaveformData) -> None:
        """Add a waveform to this session."""
        self.waveforms[waveform.waveform_type] = waveform
        self.has_waveform_data = True

    def add_event(self, event: RespiratoryEvent) -> None:
        """Add a respiratory event to this session."""
        self.events.append(event)
        self.has_event_data = True

    def finalize_statistics(self) -> None:
        """Calculate all statistics from parsed events and waveforms."""
        event_counts = {"OA": 0, "CA": 0, "H": 0, "RE": 0, "UA": 0}
        for event in self.events:
            event_type_str = (
                event.event_type.value
                if isinstance(event.event_type, RespiratoryEventType)
                else event.event_type
            )
            if event_type_str in event_counts:
                event_counts[event_type_str] += 1

        self.statistics.obstructive_apneas = event_counts["OA"]
        self.statistics.central_apneas = event_counts["CA"]
        self.statistics.hypopneas = event_counts["H"]
        self.statistics.reras = event_counts["RE"]

        hours = therapy_hours(
            TherapyHoursBasis.MASK_ON, mask_on_segments=self.mask_on_segments
        )
        if hours is None:
            # Unknown mask-on time: documented span fallback.
            hours = therapy_hours(
                TherapyHoursBasis.SESSION_SPAN, span_seconds=self.duration_seconds
            )

        if hours and hours > 0:
            total_events = event_counts["OA"] + event_counts["CA"] + event_counts["H"]
            self.statistics.ahi = total_events / hours
            self.statistics.oai = event_counts["OA"] / hours
            self.statistics.cai = event_counts["CA"] / hours
            self.statistics.hi = event_counts["H"] / hours
            self.statistics.usage_hours = hours
        elif self.mask_on_segments is not None:
            # Explicitly empty segments: known-zero mask-on time; rates undefined.
            self.statistics.usage_hours = 0.0

        therapy_wf = self.waveforms.get(WaveformType.THERAPY_PRESSURE)
        if therapy_wf is None:
            therapy_wf = self.waveforms.get(WaveformType.MASK_PRESSURE)
        if therapy_wf and therapy_wf.values is not None and len(therapy_wf.values) > 0:
            data = therapy_wf.values
            self.statistics.pressure_min = therapy_wf.min_value
            self.statistics.pressure_max = therapy_wf.max_value
            self.statistics.pressure_mean = therapy_wf.mean_value
            self.statistics.pressure_median = float(np.median(data))
            self.statistics.pressure_95th = float(np.percentile(data, 95))

        epap_wf = self.waveforms.get(WaveformType.EPAP)
        if epap_wf and epap_wf.values is not None and len(epap_wf.values) > 0:
            data = epap_wf.values
            self.statistics.epap_min = epap_wf.min_value
            self.statistics.epap_max = epap_wf.max_value
            self.statistics.epap_mean = epap_wf.mean_value
            self.statistics.epap_median = float(np.median(data))
            self.statistics.epap_95th = float(np.percentile(data, 95))

        leak_wf = self.waveforms.get(WaveformType.LEAK_RATE)
        if leak_wf and leak_wf.values is not None and len(leak_wf.values) > 0:
            data = leak_wf.values
            self.statistics.leak_min = leak_wf.min_value
            self.statistics.leak_max = leak_wf.max_value
            self.statistics.leak_mean = leak_wf.mean_value
            self.statistics.leak_median = float(np.median(data))
            self.statistics.leak_95th = float(np.percentile(data, 95))

        self.has_statistics = True
