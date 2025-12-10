"""Parser type definitions."""

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal


# ============================================================================
# Parser Metadata Types
# ============================================================================


@dataclass
class ParserMetadata:
    """Metadata about a parser implementation."""

    parser_id: str  # Unique identifier (e.g., "resmed_edf")
    parser_version: str  # Version of this parser implementation
    manufacturer: str  # Device manufacturer name
    supported_formats: list[str]  # File formats this parser handles
    supported_models: list[str]  # Device models supported
    description: str  # Human-readable description
    requires_libraries: list[str] | None = None  # External dependencies


# ============================================================================
# Discovery Types
# ============================================================================


@dataclass
class DataRoot:
    """Information about a discovered CPAP data root."""

    path: Path
    structure_type: Literal["raw_sd", "oscar_profile"]
    profile_name: str | None
    device_serial: str | None
    confidence: float


# ============================================================================
# OSCAR Event Types
# ============================================================================


class EventListType(IntEnum):
    """Type of EventList data."""

    WAVEFORM = 0  # Continuous sampled data
    EVENT = 1  # Discrete events


@dataclass
class EventList:
    """
    A single EventList containing waveform or event data.

    EventLists store either continuous waveform data (pressure, flow, etc.)
    or discrete event data (apneas, etc.) for a single channel.
    """

    channel_id: int
    first_timestamp: int  # milliseconds since epoch
    last_timestamp: int
    count: int
    event_type: EventListType
    sample_rate: float  # Events per second for waveforms
    gain: float  # Multiplier to convert stored -> actual values
    offset: float
    min_value: float
    max_value: float
    dimension: str  # Units (e.g., "cmH2O", "L/min")
    has_second_field: bool = False
    min_value2: float = 0.0
    max_value2: float = 0.0

    # Data arrays
    data: list[int] = field(
        default_factory=list
    )  # Primary data (EventStoreType = int16)
    data2: list[int] = field(
        default_factory=list
    )  # Secondary data (if has_second_field)
    time_deltas: list[int] = field(
        default_factory=list
    )  # Time offsets in ms (for events)

    def get_actual_values(self) -> list[float]:
        """
        Convert stored int16 values to actual float values using gain/offset.

        Returns:
            List of actual data values
        """
        return [v * self.gain + self.offset for v in self.data]

    def get_actual_values2(self) -> list[float]:
        """
        Convert stored int16 values to actual float values for secondary field.

        Returns:
            List of actual data values for second field
        """
        if not self.has_second_field:
            return []
        return [v * self.gain + self.offset for v in self.data2]

    def get_timestamps(self) -> list[int]:
        """
        Get actual timestamps for each data point.

        For waveforms: Calculate from sample rate
        For events: Use delta time array

        Returns:
            List of timestamps in milliseconds since epoch
        """
        if self.event_type == EventListType.WAVEFORM:
            # Calculate timestamps from sample rate
            if self.sample_rate <= 0:
                return []

            interval_ms = 1000.0 / self.sample_rate
            return [
                int(self.first_timestamp + i * interval_ms) for i in range(self.count)
            ]
        else:
            # Use delta times
            return [self.first_timestamp + delta for delta in self.time_deltas]

    @property
    def duration_seconds(self) -> float:
        """Get duration of this EventList in seconds."""
        return (self.last_timestamp - self.first_timestamp) / 1000.0


@dataclass
class SessionEvents:
    """
    Complete event/waveform data for a session from .001 file.

    Contains all EventLists for all channels in the session.
    """

    # Header fields
    magic: int
    version: int
    file_type: int
    machine_id: int
    session_id: int
    first_timestamp: int
    last_timestamp: int

    # EventLists keyed by channel ID
    event_lists: dict[int, EventList] = field(default_factory=dict)


# ============================================================================
# OSCAR Summary Types
# ============================================================================


@dataclass
class SessionSummary:
    """
    Session summary data from .000 file.

    Contains all statistics and metadata for a single therapy session.
    """

    # Header fields
    magic: int
    version: int
    file_type: int
    machine_id: int
    session_id: int
    first_timestamp: int  # milliseconds since epoch
    last_timestamp: int  # milliseconds since epoch

    # Session data
    settings: dict[int, Any] = field(default_factory=dict)
    counts: dict[int, float] = field(default_factory=dict)
    sums: dict[int, float] = field(default_factory=dict)
    averages: dict[int, float] = field(default_factory=dict)
    weighted_averages: dict[int, float] = field(default_factory=dict)
    minimums: dict[int, float] = field(default_factory=dict)
    maximums: dict[int, float] = field(default_factory=dict)
    physical_minimums: dict[int, float] = field(default_factory=dict)
    physical_maximums: dict[int, float] = field(default_factory=dict)
    counts_per_hour: dict[int, float] = field(default_factory=dict)
    sums_per_hour: dict[int, float] = field(default_factory=dict)
    first_channel_time: dict[int, int] = field(default_factory=dict)
    last_channel_time: dict[int, int] = field(default_factory=dict)
    value_summaries: dict[int, dict[int, int]] = field(default_factory=dict)
    time_summaries: dict[int, dict[int, int]] = field(default_factory=dict)
    gains: dict[int, float] = field(default_factory=dict)
    available_channels: list[int] = field(default_factory=list)

    # Additional fields
    time_above_threshold: dict[int, int] = field(default_factory=dict)
    upper_threshold: dict[int, float] = field(default_factory=dict)
    time_below_threshold: dict[int, int] = field(default_factory=dict)
    lower_threshold: dict[int, float] = field(default_factory=dict)

    summary_only: bool = False
    no_settings: bool = False
