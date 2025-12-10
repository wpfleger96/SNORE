"""
OSCAR Summary File Parser

Parses .000 files containing session summary data and statistics.
Format version 18 (current OSCAR version).
"""

import struct

from pathlib import Path
from typing import Any

from snore.constants import OSCAR_MAGIC_NUMBER
from snore.parsers.qdatastream import QDataStreamReader
from snore.parsers.types import SessionSummary


class OscarSummaryParseError(Exception):
    """Exception raised when parsing OSCAR summary file fails."""

    pass


class OscarSummaryParser:
    """
    Parser for OSCAR .000 summary files.

    Reads session summary data including statistics for all channels.
    """

    def __init__(self, file_path: Path):
        """
        Initialize parser.

        Args:
            file_path: Path to .000 file to parse
        """
        self.file_path = Path(file_path)

    def parse(self) -> SessionSummary:
        """
        Parse the summary file.

        Returns:
            SessionSummary object with all session data

        Raises:
            OscarSummaryParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Summary file not found: {self.file_path}")

        try:
            with open(self.file_path, "rb") as f:
                return self._parse_stream(f)
        except Exception as e:
            raise OscarSummaryParseError(
                f"Failed to parse {self.file_path}: {e}"
            ) from e

    def _parse_stream(self, stream: Any) -> SessionSummary:
        """Parse summary file from binary stream."""
        # Read and validate header
        header = self._parse_header(stream)

        # Create session summary object
        summary = SessionSummary(
            magic=header["magic"],
            version=header["version"],
            file_type=header["file_type"],
            machine_id=header["machine_id"],
            session_id=header["session_id"],
            first_timestamp=header["first_timestamp"],
            last_timestamp=header["last_timestamp"],
        )

        # Skip settings by finding where statistics start
        # Settings contain custom Qt types that we can't reliably parse
        # Statistics start with m_cnt (QHash<uint32, float> of event counts)
        self._skip_to_statistics(stream)

        # Parse session data using QDataStream
        reader = QDataStreamReader(stream)

        try:
            # Version 18 format
            if summary.version >= 18:
                # Now we're at m_cnt (counts per channel)
                summary.counts = reader.read_qhash_uint32_float()
                summary.sums = reader.read_qhash_uint32_double()
                summary.averages = reader.read_qhash_uint32_float()
                summary.weighted_averages = reader.read_qhash_uint32_float()
                summary.minimums = reader.read_qhash_uint32_float()
                summary.maximums = reader.read_qhash_uint32_float()
                summary.physical_minimums = reader.read_qhash_uint32_float()
                summary.physical_maximums = reader.read_qhash_uint32_float()
                summary.counts_per_hour = reader.read_qhash_uint32_float()
                summary.sums_per_hour = reader.read_qhash_uint32_float()
                summary.first_channel_time = reader.read_qhash_uint32_uint64()
                summary.last_channel_time = reader.read_qhash_uint32_uint64()
                summary.value_summaries = reader.read_qhash_nested()
                summary.time_summaries = reader.read_qhash_nested_time()
                summary.gains = reader.read_qhash_uint32_float()
                summary.available_channels = reader.read_qlist_uint32()

                # Additional fields (version 18+)
                summary.time_above_threshold = reader.read_qhash_uint32_uint64()
                summary.upper_threshold = reader.read_qhash_uint32_float()
                summary.time_below_threshold = reader.read_qhash_uint32_uint64()
                summary.lower_threshold = reader.read_qhash_uint32_float()

                summary.summary_only = reader.read_bool()
                summary.no_settings = reader.read_bool()

                # Skip session slices for now (not critical for basic stats)
                # TODO: Parse session slices if needed

            else:
                raise OscarSummaryParseError(
                    f"Unsupported summary version: {summary.version}"
                )

        except EOFError as e:
            raise OscarSummaryParseError(f"Unexpected end of file: {e}") from e
        except Exception as e:
            raise OscarSummaryParseError(f"Error parsing session data: {e}") from e

        return summary

    def _skip_to_statistics(self, stream: Any) -> None:
        """
        Skip past settings section to find where statistics begin.

        Settings contain custom Qt types that we can't reliably parse.
        Statistics start with m_cnt which is QHash<uint32, float>.

        We search for a pattern:
        - A small count (5-50 channels is reasonable)
        - Followed by channel IDs in range 0x1000-0x3000
        - Followed by reasonable float values
        """
        # Save current position
        start_pos = stream.tell()

        # Read rest of file for searching
        data = stream.read()

        # Search for potential start of m_cnt
        for offset in range(0, len(data) - 100, 4):
            # Read count
            count_bytes = data[offset : offset + 4]
            if len(count_bytes) < 4:
                continue

            count = struct.unpack("<I", count_bytes)[0]

            # Check if count is reasonable
            if not (5 <= count <= 50):
                continue

            # Check if next values look like channel ID + float pairs
            valid = True
            pos = offset + 4
            for _ in range(min(3, count)):
                if pos + 8 > len(data):
                    valid = False
                    break

                channel_id = struct.unpack("<I", data[pos : pos + 4])[0]
                value_bytes = data[pos + 4 : pos + 8]
                try:
                    value = struct.unpack("<f", value_bytes)[0]
                except struct.error:
                    valid = False
                    break

                # Validate channel ID range and value reasonableness
                # Channel IDs should be in typical OSCAR ranges
                if not (
                    (0x0001 <= channel_id <= 0x0100) or (0x1000 <= channel_id <= 0x3000)
                ):
                    valid = False
                    break
                # Values should be non-negative and not NaN/Inf
                import math

                if value < 0 or not math.isfinite(value) or value > 1000000:
                    valid = False
                    break

                pos += 8

            if valid:
                # Found the statistics section!
                stream.seek(start_pos + offset)
                return

        # If we couldn't find it, raise an error
        raise OscarSummaryParseError("Could not locate statistics section in file")

    def _parse_header(self, stream: Any) -> dict[str, Any]:
        """
        Parse 32-byte header from summary file.

        Header format:
        - 4 bytes: magic number (0xC73216AB)
        - 2 bytes: version (18)
        - 2 bytes: file type (0 for summary)
        - 4 bytes: machine ID
        - 4 bytes: session ID
        - 8 bytes: first timestamp (ms since epoch)
        - 8 bytes: last timestamp (ms since epoch)

        Returns:
            Dictionary with header fields

        Raises:
            OscarSummaryParseError: If header is invalid
        """
        header_data = stream.read(32)
        if len(header_data) != 32:
            raise OscarSummaryParseError("File too short to contain header")

        # Unpack header (little-endian)
        (
            magic,
            version,
            file_type,
            machine_id,
            session_id,
            first_timestamp,
            last_timestamp,
        ) = struct.unpack("<IHH II qq", header_data)

        # Validate magic number
        if magic != OSCAR_MAGIC_NUMBER:
            raise OscarSummaryParseError(
                f"Invalid magic number: 0x{magic:08x} (expected 0x{OSCAR_MAGIC_NUMBER:08x})"
            )

        # Validate file type
        if file_type != 0:
            raise OscarSummaryParseError(
                f"Invalid file type: {file_type} (expected 0 for summary)"
            )

        return {
            "magic": magic,
            "version": version,
            "file_type": file_type,
            "machine_id": machine_id,
            "session_id": session_id,
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
        }


def parse_summary_file(file_path: Path) -> SessionSummary:
    """
    Convenience function to parse a summary file.

    Args:
        file_path: Path to .000 file

    Returns:
        SessionSummary object

    Raises:
        OscarSummaryParseError: If parsing fails
    """
    parser = OscarSummaryParser(file_path)
    return parser.parse()
