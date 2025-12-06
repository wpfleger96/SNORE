"""
Parser for OSCAR binary file format (.000 and .001 files).

OSCAR stores session data in two files:
- .000: Summary data (metadata, settings, cached statistics)
- .001: Event/waveform data (compressed time-series data)

File Format Structure:
1. Magic number: 0xC73216AB (4 bytes, little-endian)
2. Version number: (4 bytes, little-endian)
3. Gzip-compressed QDataStream content

NOTE: This is a simplified parser for the initial skeleton.
Full implementation will require:
- Complete QDataStream format parsing (Qt's serialization format)
- All channel type handling
- EDF+ format parsing for ResMed devices
- PRS1 binary format for Philips devices
- Other device-specific parsers
"""

import logging
import os
import struct

from typing import Any

from snore.constants import EVENT_FILE_EXT, OSCAR_MAGIC_NUMBER, SUMMARY_FILE_EXT
from snore.parsers.compression import (
    decompress_gzip,
)

logger = logging.getLogger(__name__)


class OscarParseError(Exception):
    """Error parsing OSCAR file format."""

    pass


class SessionFileParser:
    """Parser for OSCAR session files (.000 and .001)."""

    def __init__(self, base_path: str, session_id: str):
        """
        Initialize parser for a session's files.

        Args:
            base_path: Directory containing session files
            session_id: Session identifier (filename without extension)
        """
        self.base_path = base_path
        self.session_id = session_id
        self.summary_path = os.path.join(base_path, f"{session_id}{SUMMARY_FILE_EXT}")
        self.event_path = os.path.join(base_path, f"{session_id}{EVENT_FILE_EXT}")

    def parse_summary(self) -> dict[str, Any]:
        """
        Parse session summary file (.000).

        Returns:
            Dictionary containing session metadata and statistics

        Raises:
            OscarParseError: If file cannot be parsed
        """
        if not os.path.exists(self.summary_path):
            raise OscarParseError(f"Summary file not found: {self.summary_path}")

        try:
            with open(self.summary_path, "rb") as f:
                data = f.read()

            # Parse header
            magic, version = self._parse_header(data)

            # Decompress content
            compressed_data = data[8:]  # Skip magic and version
            try:
                decompressed = decompress_gzip(compressed_data)
            except Exception as e:
                raise OscarParseError(f"Failed to decompress summary data: {e}") from e

            # Parse QDataStream content
            # TODO: Implement full QDataStream parser
            # For now, extract basic information
            summary = {
                "session_id": self.session_id,
                "magic_number": hex(magic),
                "version": version,
                "file_size": len(data),
                "decompressed_size": len(decompressed),
            }

            # TODO: Parse the decompressed QDataStream data to extract:
            # - start_time, end_time
            # - session settings (pressure, mode, etc.)
            # - cached statistics (AHI, pressure stats, leak stats)
            # - available channels

            logger.info(f"Parsed summary for session {self.session_id}")
            return summary

        except Exception as e:
            logger.error(f"Error parsing summary file {self.summary_path}: {e}")
            raise OscarParseError(f"Failed to parse summary: {e}") from e

    def parse_events(self) -> dict[int, dict[str, Any]]:
        """
        Parse session event/waveform file (.001).

        Returns:
            Dictionary mapping channel IDs to their event data

        Raises:
            OscarParseError: If file cannot be parsed
        """
        if not os.path.exists(self.event_path):
            logger.warning(f"Event file not found: {self.event_path}")
            return {}

        try:
            with open(self.event_path, "rb") as f:
                data = f.read()

            # Parse header
            magic, version = self._parse_header(data)

            # Decompress content
            compressed_data = data[8:]
            try:
                _decompressed = decompress_gzip(
                    compressed_data
                )  # Will be used in TODO below
            except Exception as e:
                raise OscarParseError(f"Failed to decompress event data: {e}") from e

            # TODO: Parse QDataStream content to extract:
            # - Channel ID
            # - Channel type (WAVEFORM, FLAG, etc.)
            # - Data array (int16 values)
            # - Time array (delta-encoded timestamps for events)
            # - Gain, offset, min, max
            # - Sample rate (for waveforms)

            events: dict[int, dict[str, Any]] = {}
            logger.info(f"Parsed events for session {self.session_id}")
            return events

        except Exception as e:
            logger.error(f"Error parsing event file {self.event_path}: {e}")
            raise OscarParseError(f"Failed to parse events: {e}") from e

    def _parse_header(self, data: bytes) -> tuple[int, int]:
        """
        Parse OSCAR file header (magic number and version).

        Args:
            data: Raw file data

        Returns:
            Tuple of (magic_number, version)

        Raises:
            OscarParseError: If header is invalid
        """
        if len(data) < 8:
            raise OscarParseError("File too small to contain header")

        # Read magic number and version (little-endian)
        magic, version = struct.unpack("<II", data[:8])

        if magic != OSCAR_MAGIC_NUMBER:
            raise OscarParseError(
                f"Invalid magic number: expected {hex(OSCAR_MAGIC_NUMBER)}, got {hex(magic)}"
            )

        return magic, version


class MachineDirectoryScanner:
    """Scanner for OSCAR machine directories."""

    def __init__(self, machine_path: str):
        """
        Initialize scanner for a machine directory.

        Args:
            machine_path: Path to machine directory (e.g., ProfilePath/MachineID/)
        """
        self.machine_path = machine_path
        self.summaries_path = os.path.join(machine_path, "Summaries")
        self.events_path = os.path.join(machine_path, "Events")

    def scan_sessions(self) -> list[str]:
        """
        Scan for available session files.

        Returns:
            List of session IDs (filenames without extensions)
        """
        if not os.path.exists(self.summaries_path):
            logger.warning(f"Summaries directory not found: {self.summaries_path}")
            return []

        session_ids = []
        try:
            for filename in os.listdir(self.summaries_path):
                if filename.endswith(SUMMARY_FILE_EXT):
                    session_id = filename[: -len(SUMMARY_FILE_EXT)]
                    session_ids.append(session_id)
        except Exception as e:
            logger.error(f"Error scanning sessions in {self.summaries_path}: {e}")

        logger.info(f"Found {len(session_ids)} sessions in {self.machine_path}")
        return sorted(session_ids)

    def get_session_parser(self, session_id: str) -> SessionFileParser:
        """
        Get a parser for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            SessionFileParser instance
        """
        # Session files can be in either Summaries/ or Events/ directories
        # Typically both .000 and .001 are in their respective directories
        return SessionFileParser(self.summaries_path, session_id)


class ProfileScanner:
    """Scanner for OSCAR profile directories."""

    def __init__(self, profile_path: str):
        """
        Initialize scanner for a profile directory.

        Args:
            profile_path: Path to OSCAR profile directory
        """
        self.profile_path = profile_path

    def scan_machines(self) -> list[str]:
        """
        Scan for machine directories in the profile.

        Returns:
            List of machine IDs (directory names)
        """
        if not os.path.exists(self.profile_path):
            logger.error(f"Profile path does not exist: {self.profile_path}")
            return []

        machine_ids = []
        try:
            for item in os.listdir(self.profile_path):
                item_path = os.path.join(self.profile_path, item)
                if os.path.isdir(item_path):
                    # Check if it has Summaries subdirectory (indicates machine dir)
                    summaries_path = os.path.join(item_path, "Summaries")
                    if os.path.exists(summaries_path):
                        machine_ids.append(item)
        except Exception as e:
            logger.error(f"Error scanning machines in {self.profile_path}: {e}")

        logger.info(f"Found {len(machine_ids)} machines in profile")
        return sorted(machine_ids)

    def get_machine_scanner(self, machine_id: str) -> MachineDirectoryScanner:
        """
        Get a scanner for a specific machine.

        Args:
            machine_id: Machine identifier (directory name)

        Returns:
            MachineDirectoryScanner instance
        """
        machine_path = os.path.join(self.profile_path, machine_id)
        return MachineDirectoryScanner(machine_path)


# Helper functions for common operations


def scan_oscar_profile(profile_path: str) -> dict[str, list[str]]:
    """
    Scan an OSCAR profile and list all machines and sessions.

    Args:
        profile_path: Path to OSCAR profile directory

    Returns:
        Dictionary mapping machine IDs to lists of session IDs
    """
    scanner = ProfileScanner(profile_path)
    machines = scanner.scan_machines()

    result = {}
    for machine_id in machines:
        machine_scanner = scanner.get_machine_scanner(machine_id)
        sessions = machine_scanner.scan_sessions()
        result[machine_id] = sessions

    return result


def parse_session_files(
    machine_path: str, session_id: str
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """
    Parse both summary and event files for a session.

    Args:
        machine_path: Path to machine directory
        session_id: Session identifier

    Returns:
        Tuple of (summary_data, event_data)
    """
    summaries_path = os.path.join(machine_path, "Summaries")
    parser = SessionFileParser(summaries_path, session_id)

    summary = parser.parse_summary()
    events = parser.parse_events()

    return summary, events
