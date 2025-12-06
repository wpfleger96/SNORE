"""
OSCAR Events File Parser

Parses .001 files containing waveform and event data.
Format version 10 (current OSCAR version).
"""

import io
import struct

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO

from snore.constants import OSCAR_MAGIC_NUMBER
from snore.parsers.compression import QtCompressionError, qUncompress
from snore.parsers.qdatastream import QDataStreamReader


class OscarEventsParseError(Exception):
    """Exception raised when parsing OSCAR events file fails."""

    pass


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
    compression: int  # 0=none, 1=qCompress
    machine_type: int
    data_size: int
    crc16: int

    # Channel data
    event_lists: dict[int, list[EventList]] = field(default_factory=dict)

    @property
    def start_time(self) -> datetime:
        """Get session start time as datetime."""
        return datetime.fromtimestamp(self.first_timestamp / 1000.0)

    @property
    def end_time(self) -> datetime:
        """Get session end time as datetime."""
        return datetime.fromtimestamp(self.last_timestamp / 1000.0)

    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return (self.last_timestamp - self.first_timestamp) / 1000.0

    @property
    def duration_hours(self) -> float:
        """Get session duration in hours."""
        return self.duration_seconds / 3600.0

    def get_channel_event_lists(self, channel_id: int) -> list[EventList]:
        """
        Get all EventLists for a specific channel.

        Args:
            channel_id: Channel identifier

        Returns:
            List of EventLists for the channel
        """
        return self.event_lists.get(channel_id, [])

    @property
    def available_channels(self) -> list[int]:
        """Get list of channel IDs that have data."""
        return list(self.event_lists.keys())


class OscarEventsParser:
    """
    Parser for OSCAR .001 event/waveform files.

    Reads detailed session data including time-series waveforms and events.
    """

    def __init__(self, file_path: Path):
        """
        Initialize parser.

        Args:
            file_path: Path to .001 file to parse
        """
        self.file_path = Path(file_path)

    def parse(self) -> SessionEvents:
        """
        Parse the events file.

        Returns:
            SessionEvents object with all waveform/event data

        Raises:
            OscarEventsParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Events file not found: {self.file_path}")

        try:
            with open(self.file_path, "rb") as f:
                return self._parse_stream(f)
        except Exception as e:
            raise OscarEventsParseError(f"Failed to parse {self.file_path}: {e}") from e

    def _parse_stream(self, stream: BinaryIO) -> SessionEvents:
        """Parse events file from binary stream."""
        # Read and validate header
        header = self._parse_header(stream)

        # Create session events object
        events = SessionEvents(
            magic=header["magic"],
            version=header["version"],
            file_type=header["file_type"],
            machine_id=header["machine_id"],
            session_id=header["session_id"],
            first_timestamp=header["first_timestamp"],
            last_timestamp=header["last_timestamp"],
            compression=header["compression"],
            machine_type=header["machine_type"],
            data_size=header["data_size"],
            crc16=header["crc16"],
        )

        # Handle compression if present
        if events.compression == 1:
            # Read compressed data
            compressed_data = stream.read()

            try:
                # Decompress using Qt's qUncompress
                decompressed_data = qUncompress(compressed_data)

                # Create new stream from decompressed data
                stream = io.BytesIO(decompressed_data)
            except QtCompressionError as e:
                raise OscarEventsParseError(f"Failed to decompress data: {e}") from e

        # Parse event data using QDataStream
        reader = QDataStreamReader(stream)

        try:
            # Parse channel metadata first
            channel_metadata = self._parse_channel_metadata(reader, events.version)

            # Parse channel data
            events.event_lists = self._parse_channel_data(reader, channel_metadata)

        except EOFError as e:
            raise OscarEventsParseError(f"Unexpected end of file: {e}") from e
        except Exception as e:
            raise OscarEventsParseError(f"Error parsing events data: {e}") from e

        return events

    def _parse_header(self, stream: BinaryIO) -> dict[str, Any]:
        """
        Parse 42-byte header from events file.

        Header format:
        - 4 bytes: magic number (0xC73216AB)
        - 2 bytes: version (10)
        - 2 bytes: file type (1 for events)
        - 4 bytes: machine ID
        - 4 bytes: session ID
        - 8 bytes: first timestamp (ms since epoch)
        - 8 bytes: last timestamp (ms since epoch)
        - 2 bytes: compression method (0=none, 1=qCompress)
        - 2 bytes: machine type
        - 4 bytes: uncompressed data size
        - 2 bytes: CRC16 checksum

        Returns:
            Dictionary with header fields

        Raises:
            OscarEventsParseError: If header is invalid
        """
        header_data = stream.read(42)
        if len(header_data) != 42:
            raise OscarEventsParseError("File too short to contain header")

        # Unpack header (little-endian)
        (
            magic,
            version,
            file_type,
            machine_id,
            session_id,
            first_timestamp,
            last_timestamp,
            compression,
            machine_type,
            data_size,
            crc16,
        ) = struct.unpack("<IHH II qq HH iH", header_data)

        # Validate magic number
        if magic != OSCAR_MAGIC_NUMBER:
            raise OscarEventsParseError(
                f"Invalid magic number: 0x{magic:08x} (expected 0x{OSCAR_MAGIC_NUMBER:08x})"
            )

        # Validate file type
        if file_type != 1:
            raise OscarEventsParseError(
                f"Invalid file type: {file_type} (expected 1 for events)"
            )

        return {
            "magic": magic,
            "version": version,
            "file_type": file_type,
            "machine_id": machine_id,
            "session_id": session_id,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "compression": compression,
            "machine_type": machine_type,
            "data_size": data_size,
            "crc16": crc16,
        }

    def _parse_channel_metadata(
        self, reader: QDataStreamReader, version: int
    ) -> list[dict[str, Any]]:
        """
        Parse channel metadata section.

        This section contains information about all EventLists in the file.

        Returns:
            List of metadata dictionaries for each EventList
        """
        metadata_list = []

        # Read number of channels
        num_channels = reader.read_int16()

        for _ in range(num_channels):
            channel_id = reader.read_uint32()
            num_eventlists = reader.read_int16()

            for _ in range(num_eventlists):
                first_ts = reader.read_int64()
                last_ts = reader.read_int64()
                count = reader.read_int32()
                event_type = reader.read_int8()
                sample_rate = reader.read_float()
                gain = reader.read_float()
                offset = reader.read_float()
                min_value = reader.read_float()
                max_value = reader.read_float()
                dimension = reader.read_qstring()

                # Handle event type - mask off high bits if present
                # OSCAR may use additional flags in the type field
                event_type_value = event_type & 0x0F  # Get low 4 bits
                try:
                    parsed_event_type = EventListType(event_type_value)
                except ValueError:
                    # Unknown event type, default to WAVEFORM
                    import warnings

                    warnings.warn(
                        f"Unknown event type {event_type} (0x{event_type:02x}), treating as WAVEFORM",
                        stacklevel=2,
                    )
                    parsed_event_type = EventListType.WAVEFORM

                metadata = {
                    "channel_id": channel_id,
                    "first_timestamp": first_ts,
                    "last_timestamp": last_ts,
                    "count": count,
                    "event_type": parsed_event_type,
                    "sample_rate": sample_rate,
                    "gain": gain,
                    "offset": offset,
                    "min_value": min_value,
                    "max_value": max_value,
                    "dimension": dimension if dimension else "",
                    "has_second_field": False,
                    "min_value2": 0.0,
                    "max_value2": 0.0,
                }

                # Version 7+ supports second field
                if version >= 7:
                    has_second = reader.read_bool()
                    metadata["has_second_field"] = has_second
                    if has_second:
                        metadata["min_value2"] = reader.read_float()
                        metadata["max_value2"] = reader.read_float()

                metadata_list.append(metadata)

        return metadata_list

    def _parse_channel_data(
        self, reader: QDataStreamReader, metadata_list: list[dict[str, Any]]
    ) -> dict[int, list[EventList]]:
        """
        Parse actual channel data arrays.

        Reads the raw data for all EventLists described in metadata.

        Returns:
            Dictionary mapping channel IDs to lists of EventLists
        """
        result: dict[int, list[EventList]] = {}

        for metadata in metadata_list:
            # Read primary data array
            data = reader.read_qvector_int16()

            # Read secondary data array if present
            data2 = []
            if metadata["has_second_field"]:
                data2 = reader.read_qvector_int16()

            # Read time delta array for events (not waveforms)
            time_deltas = []
            if metadata["event_type"] != EventListType.WAVEFORM:
                time_deltas = reader.read_qvector_uint32()

            # Create EventList object
            event_list = EventList(
                channel_id=metadata["channel_id"],
                first_timestamp=metadata["first_timestamp"],
                last_timestamp=metadata["last_timestamp"],
                count=metadata["count"],
                event_type=metadata["event_type"],
                sample_rate=metadata["sample_rate"],
                gain=metadata["gain"],
                offset=metadata["offset"],
                min_value=metadata["min_value"],
                max_value=metadata["max_value"],
                dimension=metadata["dimension"],
                has_second_field=metadata["has_second_field"],
                min_value2=metadata["min_value2"],
                max_value2=metadata["max_value2"],
                data=data,
                data2=data2,
                time_deltas=time_deltas,
            )

            # Add to result dictionary
            channel_id = metadata["channel_id"]
            if channel_id not in result:
                result[channel_id] = []
            result[channel_id].append(event_list)

        return result


def parse_events_file(file_path: Path) -> SessionEvents:
    """
    Convenience function to parse an events file.

    Args:
        file_path: Path to .001 file

    Returns:
        SessionEvents object

    Raises:
        OscarEventsParseError: If parsing fails
    """
    parser = OscarEventsParser(file_path)
    return parser.parse()
