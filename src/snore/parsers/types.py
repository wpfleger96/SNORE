"""Parser type definitions."""

from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ParserMetadata(BaseModel):
    """Metadata about a parser implementation."""

    parser_id: str = Field(description="Unique parser identifier")
    parser_version: str = Field(description="Parser version")
    manufacturer: str = Field(description="Device manufacturer")
    supported_formats: list[str] = Field(description="Supported file formats")
    supported_models: list[str] = Field(description="Supported device models")
    description: str = Field(description="Parser description")
    requires_libraries: list[str] | None = Field(
        None, description="External library dependencies"
    )


class DataRoot(BaseModel):
    """Information about a discovered CPAP data root."""

    path: Path = Field(description="Root directory path")
    structure_type: Literal["raw_sd", "oscar_profile"] = Field(
        description="Data structure type"
    )
    profile_name: str | None = Field(
        default=None, description="Profile name if applicable"
    )
    device_serial: str | None = Field(default=None, description="Device serial number")
    confidence: float = Field(ge=0, le=1, description="Discovery confidence")


class EventListType(IntEnum):
    """Type of EventList data."""

    WAVEFORM = 0  # Continuous sampled data
    EVENT = 1  # Discrete events


class EventList(BaseModel):
    """
    A single EventList containing waveform or event data.

    EventLists store either continuous waveform data (pressure, flow, etc.)
    or discrete event data (apneas, etc.) for a single channel.
    """

    channel_id: int = Field(description="Channel ID")
    first_timestamp: int = Field(description="First timestamp (ms since epoch)")
    last_timestamp: int = Field(description="Last timestamp (ms since epoch)")
    count: int = Field(ge=0, description="Data point count")
    event_type: EventListType = Field(description="Waveform or event data")
    sample_rate: float = Field(ge=0, description="Sample rate (Hz)")
    gain: float = Field(description="Value conversion multiplier")
    offset: float = Field(description="Value conversion offset")
    min_value: float = Field(description="Minimum value")
    max_value: float = Field(description="Maximum value")
    dimension: str = Field(description="Units (e.g., 'cmH2O', 'L/min')")
    has_second_field: bool = Field(
        default=False, description="Has secondary data field"
    )
    min_value2: float = Field(default=0.0, description="Min value for field 2")
    max_value2: float = Field(default=0.0, description="Max value for field 2")

    data: list[int] = Field(default_factory=list, description="Primary data (int16)")
    data2: list[int] = Field(default_factory=list, description="Secondary data")
    time_deltas: list[int] = Field(
        default_factory=list, description="Time offsets (ms)"
    )

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
            if self.sample_rate <= 0:
                return []

            interval_ms = 1000.0 / self.sample_rate
            return [
                int(self.first_timestamp + i * interval_ms) for i in range(self.count)
            ]
        else:
            return [self.first_timestamp + delta for delta in self.time_deltas]

    @property
    def duration_seconds(self) -> float:
        """Get duration of this EventList in seconds."""
        return (self.last_timestamp - self.first_timestamp) / 1000.0


class SessionEvents(BaseModel):
    """
    Complete event/waveform data for a session from .001 file.

    Contains all EventLists for all channels in the session.
    """

    magic: int = Field(description="File magic number")
    version: int = Field(description="File format version")
    file_type: int = Field(description="File type identifier")
    machine_id: int = Field(description="Machine ID")
    session_id: int = Field(description="Session ID")
    first_timestamp: int = Field(description="First timestamp (ms)")
    last_timestamp: int = Field(description="Last timestamp (ms)")

    event_lists: dict[int, list[EventList]] = Field(
        default_factory=dict, description="EventLists by channel ID"
    )


class SessionSummary(BaseModel):
    """
    Session summary data from .000 file.

    Contains all statistics and metadata for a single therapy session.
    """

    magic: int = Field(description="File magic number")
    version: int = Field(description="File format version")
    file_type: int = Field(description="File type identifier")
    machine_id: int = Field(description="Machine ID")
    session_id: int = Field(description="Session ID")
    first_timestamp: int = Field(description="First timestamp (ms since epoch)")
    last_timestamp: int = Field(description="Last timestamp (ms since epoch)")

    settings: dict[int, Any] = Field(
        default_factory=dict, description="Device settings"
    )
    counts: dict[int, float] = Field(default_factory=dict, description="Event counts")
    sums: dict[int, float] = Field(default_factory=dict, description="Summed values")
    averages: dict[int, float] = Field(
        default_factory=dict, description="Average values"
    )
    weighted_averages: dict[int, float] = Field(
        default_factory=dict, description="Weighted averages"
    )
    minimums: dict[int, float] = Field(
        default_factory=dict, description="Minimum values"
    )
    maximums: dict[int, float] = Field(
        default_factory=dict, description="Maximum values"
    )
    physical_minimums: dict[int, float] = Field(
        default_factory=dict, description="Physical minimums"
    )
    physical_maximums: dict[int, float] = Field(
        default_factory=dict, description="Physical maximums"
    )
    counts_per_hour: dict[int, float] = Field(
        default_factory=dict, description="Event counts per hour"
    )
    sums_per_hour: dict[int, float] = Field(
        default_factory=dict, description="Summed values per hour"
    )
    first_channel_time: dict[int, int] = Field(
        default_factory=dict, description="First timestamp by channel"
    )
    last_channel_time: dict[int, int] = Field(
        default_factory=dict, description="Last timestamp by channel"
    )
    value_summaries: dict[int, dict[int, int]] = Field(
        default_factory=dict, description="Value distribution summaries"
    )
    time_summaries: dict[int, dict[int, int]] = Field(
        default_factory=dict, description="Time distribution summaries"
    )
    gains: dict[int, float] = Field(default_factory=dict, description="Channel gains")
    available_channels: list[int] = Field(
        default_factory=list, description="Available channel IDs"
    )

    time_above_threshold: dict[int, int] = Field(
        default_factory=dict, description="Time above threshold"
    )
    upper_threshold: dict[int, float] = Field(
        default_factory=dict, description="Upper threshold values"
    )
    time_below_threshold: dict[int, int] = Field(
        default_factory=dict, description="Time below threshold"
    )
    lower_threshold: dict[int, float] = Field(
        default_factory=dict, description="Lower threshold values"
    )

    summary_only: bool = Field(default=False, description="Summary-only flag")
    no_settings: bool = Field(default=False, description="No settings flag")
