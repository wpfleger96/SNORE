"""
Unified Data Model for CPAP Data Platform

This module defines the universal data structures that ALL parsers must convert their
data into, regardless of manufacturer or file format. This enables complete separation
between the parser layer and the rest of the system.

Key Principle: The MCP server, database layer, and analysis tools only work with
these unified structures - they never see parser-specific formats.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from uuid import UUID, uuid4

import numpy as np


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


@dataclass
class DeviceInfo:
    """Universal device information - same structure for all manufacturers."""

    manufacturer: str  # "ResMed", "Philips", etc.
    model: str  # "AirSense 11 AutoSet"
    serial_number: str  # Device serial number

    # Optional fields
    firmware_version: str | None = None
    hardware_version: str | None = None
    product_code: str | None = None
    manufacturing_date: datetime | None = None


@dataclass
class TherapySettings:
    """Universal therapy settings across all devices."""

    mode: TherapyMode

    # Pressure settings (in cmH2O)
    pressure_min: float | None = None
    pressure_max: float | None = None
    pressure_fixed: float | None = None  # For CPAP mode

    # BiPAP settings
    ipap: float | None = None  # Inspiratory pressure
    epap: float | None = None  # Expiratory pressure
    ps: float | None = None  # Pressure support (IPAP - EPAP)

    # Comfort settings
    epr_level: int | None = None  # Expiratory pressure relief (0-3)
    ramp_time: int | None = None  # Minutes
    ramp_start_pressure: float | None = None

    # Humidifier
    humidity_level: int | None = None  # 0-8 or device-specific scale
    tube_temp: float | None = None  # Celsius

    # Mask settings
    mask_type: str | None = None  # "Full Face", "Nasal", etc.

    # Other settings stored as key-value pairs
    other_settings: dict[str, str] = field(default_factory=dict)


@dataclass
class SessionStatistics:
    """Universal session statistics."""

    # Event counts
    obstructive_apneas: int = 0
    central_apneas: int = 0
    mixed_apneas: int = 0
    hypopneas: int = 0
    reras: int = 0
    flow_limitations: int = 0

    # Indices (events per hour)
    ahi: float | None = None  # Apnea-Hypopnea Index
    oai: float | None = None  # Obstructive Apnea Index
    cai: float | None = None  # Central Apnea Index
    hi: float | None = None  # Hypopnea Index
    rei: float | None = None  # Respiratory Event Index

    # Pressure statistics (cmH2O)
    pressure_min: float | None = None
    pressure_max: float | None = None
    pressure_median: float | None = None
    pressure_mean: float | None = None
    pressure_95th: float | None = None

    # Leak statistics (L/min)
    leak_min: float | None = None
    leak_max: float | None = None
    leak_median: float | None = None
    leak_mean: float | None = None
    leak_95th: float | None = None
    leak_percentile_70: float | None = None  # Some devices use 70th percentile

    # Respiratory statistics
    respiratory_rate_min: float | None = None
    respiratory_rate_max: float | None = None
    respiratory_rate_mean: float | None = None

    tidal_volume_min: float | None = None  # mL
    tidal_volume_max: float | None = None
    tidal_volume_mean: float | None = None

    minute_ventilation_min: float | None = None  # L/min
    minute_ventilation_max: float | None = None
    minute_ventilation_mean: float | None = None

    # SpO2 statistics (if oximetry available)
    spo2_min: float | None = None
    spo2_max: float | None = None
    spo2_mean: float | None = None
    spo2_time_below_90: int | None = None  # Seconds

    pulse_min: float | None = None  # BPM
    pulse_max: float | None = None
    pulse_mean: float | None = None

    # Usage
    usage_hours: float | None = None


@dataclass
class RespiratoryEvent:
    """A single respiratory event (apnea, hypopnea, etc.)."""

    event_type: RespiratoryEventType
    start_time: datetime
    duration_seconds: float

    # Optional additional data
    peak_flow_limitation: float | None = None
    spo2_drop: float | None = None
    end_time: datetime | None = None


@dataclass
class WaveformData:
    """
    Time-series waveform data for a single channel.

    Timestamps are stored as numpy arrays of seconds offset from session start (float32).
    Values are stored as float32 numpy arrays for memory efficiency.

    A typical 8-hour session at 25Hz has 720,000 samples per channel:
    - Numpy float32 arrays: 2.7 MB per waveform
    """

    waveform_type: WaveformType
    sample_rate: float  # Samples per second (Hz)
    unit: str  # "L/min", "cmH2O", "%", etc.

    # Time-series data - MUST be seconds offset from session start (not datetime!)
    timestamps: list[float] | np.ndarray  # Seconds from session start
    values: list[float] | np.ndarray

    # Statistics for this waveform
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None

    def __post_init__(self) -> None:
        """Convert lists to numpy arrays for efficiency."""
        # Convert to numpy arrays (skip if already correct dtype)
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


@dataclass
class UnifiedSession:
    """
    Universal session format that ALL parsers must produce.

    This is the lingua franca of the CPAP data platform - every parser
    converts its native format into this structure, and all downstream
    components (database, MCP server, analysis) work exclusively with this.
    """

    # Unique identifiers
    session_id: UUID = field(default_factory=uuid4)  # Our internal ID
    device_session_id: str = ""  # Device's own session ID

    # Device information
    device_info: DeviceInfo = field(default_factory=lambda: DeviceInfo("", "", ""))

    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)

    # Therapy data
    settings: TherapySettings = field(
        default_factory=lambda: TherapySettings(TherapyMode.CPAP)
    )
    statistics: SessionStatistics = field(default_factory=SessionStatistics)

    # Time-series data
    waveforms: dict[WaveformType, WaveformData] = field(default_factory=dict)
    events: list[RespiratoryEvent] = field(default_factory=list)

    # Import metadata
    import_source: str = ""  # "resmed_edf", "oscar_binary", "philips_binary"
    import_date: datetime = field(default_factory=datetime.now)
    raw_data_path: str | None = None  # Path to original files
    parser_version: str = ""  # Version of parser that created this

    # Data quality flags
    has_waveform_data: bool = False
    has_event_data: bool = False
    has_statistics: bool = False
    data_quality_notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate session data after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate session data consistency."""
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
        # Note: After WaveformData.__post_init__, timestamps are numpy arrays of seconds offset
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


# Type aliases for convenience
SessionList = list[UnifiedSession]
WaveformDict = dict[WaveformType, WaveformData]
EventList = list[RespiratoryEvent]
