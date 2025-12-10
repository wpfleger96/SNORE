"""
Unified Data Model for CPAP Data Platform

This module defines the universal data structures that ALL parsers must convert their
data into, regardless of manufacturer or file format. This enables complete separation
between the parser layer and the rest of the system.

Key Principle: The MCP server, database layer, and analysis tools only work with
these unified structures - they never see parser-specific formats.
"""

from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

import numpy as np

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    LEAK_RATE = "leak"  # L/min
    MINUTE_VENTILATION = "mv"  # L/min
    RESPIRATORY_RATE = "rr"  # breaths/min
    TIDAL_VOLUME = "tv"  # mL
    SPO2 = "spo2"  # %
    PULSE = "pulse"  # BPM
    FLOW_LIMITATION = "fl"  # arbitrary units
    SNORE = "snore"  # arbitrary units


class TherapyMode(Enum):
    """Universal therapy mode types."""

    CPAP = "CPAP"  # Fixed pressure
    APAP = "APAP"  # Auto-adjusting pressure
    BIPAP = "BiPAP"  # Bi-level
    BIPAP_ST = "BiPAP S/T"  # Spontaneous/Timed
    BIPAP_AUTO = "BiPAP Auto"  # Auto bi-level
    ASV = "ASV"  # Adaptive servo-ventilation


class DeviceInfo(BaseModel):
    """Universal device information - same structure for all manufacturers."""

    manufacturer: str = Field(description="Manufacturer (e.g., 'ResMed', 'Philips')")
    model: str = Field(description="Device model")
    serial_number: str = Field(description="Device serial number")

    # Optional fields
    firmware_version: str | None = Field(default=None, description="Firmware version")
    hardware_version: str | None = Field(default=None, description="Hardware version")
    product_code: str | None = Field(default=None, description="Product code")
    manufacturing_date: datetime | None = Field(
        default=None, description="Manufacturing date"
    )


class TherapySettings(BaseModel):
    """Universal therapy settings across all devices."""

    mode: TherapyMode = Field(description="Therapy mode")

    # Pressure settings (in cmH2O)
    pressure_min: float | None = Field(default=None, description="Minimum pressure")
    pressure_max: float | None = Field(default=None, description="Maximum pressure")
    pressure_fixed: float | None = Field(
        default=None, description="Fixed pressure (CPAP mode)"
    )

    # BiPAP settings
    ipap: float | None = Field(default=None, description="Inspiratory pressure")
    epap: float | None = Field(default=None, description="Expiratory pressure")
    ps: float | None = Field(default=None, description="Pressure support (IPAP - EPAP)")

    # Comfort settings
    epr_level: int | None = Field(
        default=None, ge=0, le=3, description="EPR level (0-3)"
    )
    ramp_time: int | None = Field(default=None, ge=0, description="Ramp time (minutes)")
    ramp_start_pressure: float | None = Field(
        default=None, description="Ramp start pressure"
    )

    # Humidifier
    humidity_level: int | None = Field(default=None, description="Humidity level")
    tube_temp: float | None = Field(default=None, description="Tube temperature (°C)")

    # Mask settings
    mask_type: str | None = Field(default=None, description="Mask type")

    # Other settings stored as key-value pairs
    other_settings: dict[str, str] = Field(
        default_factory=dict, description="Other settings"
    )


class SessionStatistics(BaseModel):
    """Universal session statistics."""

    # Event counts
    obstructive_apneas: int = Field(default=0, ge=0, description="OA count")
    central_apneas: int = Field(default=0, ge=0, description="CA count")
    mixed_apneas: int = Field(default=0, ge=0, description="MA count")
    hypopneas: int = Field(default=0, ge=0, description="Hypopnea count")
    reras: int = Field(default=0, ge=0, description="RERA count")
    flow_limitations: int = Field(default=0, ge=0, description="FL count")

    # Indices (events per hour)
    ahi: float | None = Field(default=None, ge=0, description="Apnea-Hypopnea Index")
    oai: float | None = Field(default=None, ge=0, description="Obstructive Apnea Index")
    cai: float | None = Field(default=None, ge=0, description="Central Apnea Index")
    hi: float | None = Field(default=None, ge=0, description="Hypopnea Index")
    rei: float | None = Field(default=None, ge=0, description="Respiratory Event Index")

    # Pressure statistics (cmH2O)
    pressure_min: float | None = Field(default=None, description="Minimum pressure")
    pressure_max: float | None = Field(default=None, description="Maximum pressure")
    pressure_median: float | None = Field(default=None, description="Median pressure")
    pressure_mean: float | None = Field(default=None, description="Mean pressure")
    pressure_95th: float | None = Field(
        default=None, description="95th percentile pressure"
    )

    # Leak statistics (L/min)
    leak_min: float | None = Field(default=None, description="Minimum leak")
    leak_max: float | None = Field(default=None, description="Maximum leak")
    leak_median: float | None = Field(default=None, description="Median leak")
    leak_mean: float | None = Field(default=None, description="Mean leak")
    leak_95th: float | None = Field(default=None, description="95th percentile leak")
    leak_percentile_70: float | None = Field(
        default=None, description="70th percentile leak"
    )

    # Respiratory statistics
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

    # SpO2 statistics (if oximetry available)
    spo2_min: float | None = Field(default=None, description="Min SpO2 (%)")
    spo2_max: float | None = Field(default=None, description="Max SpO2")
    spo2_mean: float | None = Field(default=None, description="Mean SpO2")
    spo2_time_below_90: int | None = Field(
        default=None, description="Time below 90% (seconds)"
    )

    pulse_min: float | None = Field(default=None, description="Min pulse (BPM)")
    pulse_max: float | None = Field(default=None, description="Max pulse")
    pulse_mean: float | None = Field(default=None, description="Mean pulse")

    # Usage
    usage_hours: float | None = Field(
        default=None, ge=0, description="Usage time (hours)"
    )


class RespiratoryEvent(BaseModel):
    """A single respiratory event (apnea, hypopnea, etc.)."""

    event_type: RespiratoryEventType = Field(description="Event type")
    start_time: datetime = Field(description="Event start time")
    duration_seconds: float = Field(ge=0, description="Event duration (seconds)")

    # Optional additional data
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

    # Time-series data - MUST be seconds offset from session start (not datetime!)
    timestamps: list[float] | np.ndarray = Field(
        description="Seconds from session start"
    )
    values: list[float] | np.ndarray = Field(description="Waveform values")

    # Statistics for this waveform
    min_value: float | None = Field(default=None, description="Minimum value")
    max_value: float | None = Field(default=None, description="Maximum value")
    mean_value: float | None = Field(default=None, description="Mean value")

    @model_validator(mode="after")
    def convert_to_numpy(self) -> "WaveformData":
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
    components (database, MCP server, analysis) work exclusively with this.
    """

    # Unique identifiers
    session_id: UUID = Field(default_factory=uuid4, description="Internal session ID")
    device_session_id: str = Field(default="", description="Device session ID")

    # Device information
    device_info: DeviceInfo = Field(description="Device information")

    # Timing
    start_time: datetime = Field(description="Session start time")
    end_time: datetime = Field(description="Session end time")

    # Therapy data
    settings: TherapySettings | None = Field(
        default=None, description="Therapy settings"
    )
    statistics: SessionStatistics = Field(
        default_factory=SessionStatistics, description="Session statistics"
    )

    # Time-series data
    waveforms: dict[WaveformType, WaveformData] = Field(
        default_factory=dict, description="Waveform data by type"
    )
    events: list[RespiratoryEvent] = Field(
        default_factory=list, description="Respiratory events"
    )

    # Import metadata
    import_source: str = Field(default="", description="Parser ID")
    import_date: datetime = Field(
        default_factory=datetime.now, description="Import timestamp"
    )
    raw_data_path: str | None = Field(
        default=None, description="Path to original files"
    )
    parser_version: str = Field(default="", description="Parser version")

    # Data quality flags
    has_waveform_data: bool = Field(default=False, description="Has waveform data")
    has_event_data: bool = Field(default=False, description="Has event data")
    has_statistics: bool = Field(default=False, description="Has statistics")
    data_quality_notes: list[str] = Field(
        default_factory=list, description="Data quality warnings"
    )

    @model_validator(mode="after")
    def validate_session(self) -> "UnifiedSession":
        """Validate session data after initialization."""
        errors = []

        # Check time ordering
        if self.end_time <= self.start_time:
            errors.append(
                f"end_time ({self.end_time}) must be after start_time ({self.start_time})"
            )

        # Check duration is reasonable
        duration_hours = self.duration_hours
        if duration_hours > 24:
            self.data_quality_notes.append(
                f"Warning: Unusually long session duration: {duration_hours:.1f} hours"
            )
        if duration_hours < 0:
            errors.append(f"Negative session duration: {duration_hours:.1f} hours")

        # Validate waveform timestamps are within session bounds (allow 1 second tolerance)
        for waveform_type, waveform in self.waveforms.items():
            if waveform.timestamps is not None and len(waveform.timestamps) > 0:
                # Check if timestamps are datetime objects or numeric offsets
                if isinstance(waveform.timestamps, np.ndarray):
                    # Numeric offsets - validate against session duration
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
                    # Datetime objects - validate against session start/end
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

    def get_waveform(self, waveform_type: WaveformType) -> WaveformData | None:
        """Get a specific waveform by type."""
        return self.waveforms.get(waveform_type)

    def has_waveform(self, waveform_type: WaveformType) -> bool:
        """Check if session has data for a specific waveform type."""
        return waveform_type in self.waveforms
