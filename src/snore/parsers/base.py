"""
Abstract Parser Interface

This module defines the base class that ALL device parsers must implement.
This ensures a consistent interface across all manufacturers and file formats.

Key Principle: Any new device parser just needs to inherit from DeviceParser
and implement these methods - the rest of the system automatically works.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from snore.models.unified import DeviceInfo, UnifiedSession
from snore.parsers.types import ParserMetadata


class ParserDetectionResult:
    """Result of parser detection scan."""

    def __init__(
        self,
        detected: bool,
        confidence: float = 1.0,
        device_info: DeviceInfo | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        self.detected = detected
        self.confidence = confidence  # 0.0 to 1.0
        self.device_info = device_info
        self.message = message
        self.metadata = metadata or {}


class DeviceParser(ABC):
    """
    Abstract base class for all CPAP device parsers.

    Every parser (ResMed, Philips, OSCAR binary, etc.) must inherit from
    this class and implement all abstract methods. This ensures consistency
    and allows the system to work with any parser without modification.

    Usage Example:
        class ResmedEDFParser(DeviceParser):
            def detect(self, path):
                return (path / "STR.edf").exists()

            def parse_sessions(self, path):
                # Parse ResMed EDF+ files
                yield unified_session

    The rest of the system doesn't need to know anything about ResMed!
    """

    def __init__(self) -> None:
        """Initialize the parser."""
        self._metadata = self.get_metadata()

    @abstractmethod
    def get_metadata(self) -> ParserMetadata:
        """
        Return metadata about this parser.

        Returns:
            ParserMetadata with parser information

        Example:
            return ParserMetadata(
                parser_id="resmed_edf",
                parser_version="1.0.0",
                manufacturer="ResMed",
                supported_formats=["EDF+", "EDF"],
                supported_models=["AirSense 10", "AirSense 11"],
                description="Parser for ResMed EDF+ files"
            )
        """
        pass

    @abstractmethod
    def detect(self, path: Path) -> ParserDetectionResult:
        """
        Detect if this parser can handle the data at the given path.

        This method should quickly check if the directory/file structure
        matches what this parser expects. It should NOT do full parsing.

        Args:
            path: Path to directory or file to check

        Returns:
            ParserDetectionResult indicating if this parser can handle the data

        Example:
            # ResMed parser checks for STR.edf and DATALOG folder
            str_file = path / "STR.edf"
            datalog = path / "DATALOG"
            if str_file.exists() and datalog.is_dir():
                return ParserDetectionResult(
                    detected=True,
                    confidence=1.0,
                    message="Found ResMed EDF+ structure"
                )
            return ParserDetectionResult(detected=False)
        """
        pass

    @abstractmethod
    def get_device_info(self, path: Path) -> DeviceInfo:
        """
        Extract device information from the data files.

        This is called after detection succeeds to get basic device info
        before doing full parsing. Should be fast.

        Args:
            path: Path to data directory/file

        Returns:
            DeviceInfo with manufacturer, model, serial number, etc.

        Raises:
            ParserError: If device info cannot be extracted
        """
        pass

    @abstractmethod
    def parse_sessions(
        self,
        path: Path,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = None,
        sort_by: str | None = None,
    ) -> Iterator[UnifiedSession]:
        """
        Parse all sessions from the given path and yield unified sessions.

        This is the core parsing method. It should:
        1. Locate all session data files
        2. Parse each session's native format
        3. Convert to UnifiedSession format
        4. Yield one UnifiedSession at a time (memory efficient)

        Args:
            path: Path to data directory/file
            date_from: Optional start date filter (ISO format: YYYY-MM-DD)
            date_to: Optional end date filter (ISO format: YYYY-MM-DD)
            limit: Optional maximum number of sessions to yield
            sort_by: Optional sort order - "date-asc", "date-desc", or None for filesystem order

        Yields:
            UnifiedSession objects

        Raises:
            ParserError: If parsing fails

        Example:
            for session_file in self._find_session_files(path):
                # Parse native format
                native_data = self._parse_native_format(session_file)

                # Convert to unified format
                unified = self._to_unified_session(native_data)

                yield unified
        """
        pass

    def parse_single_session(
        self, path: Path, session_id: str
    ) -> UnifiedSession | None:
        """
        Parse a single specific session by ID.

        Default implementation iterates through all sessions. Subclasses
        can override for more efficient direct lookup.

        Args:
            path: Path to data directory
            session_id: Device-specific session identifier

        Returns:
            UnifiedSession or None if not found
        """
        for session in self.parse_sessions(path):
            if session.device_session_id == session_id:
                return session
        return None

    def validate_data(self, path: Path) -> dict[str, Any]:
        """
        Validate the data at the given path.

        Default implementation just checks if detection succeeds.
        Subclasses can override for deeper validation.

        Args:
            path: Path to validate

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'file_count': int,
                'date_range': tuple,
                etc.
            }
        """
        result = self.detect(path)
        return {
            "valid": result.detected,
            "errors": [] if result.detected else [result.message],
            "warnings": [],
            "confidence": result.confidence,
        }

    @property
    def metadata(self) -> ParserMetadata:
        """Get parser metadata."""
        return self._metadata

    @property
    def parser_id(self) -> str:
        """Get unique parser identifier."""
        return self._metadata.parser_id

    @property
    def manufacturer(self) -> str:
        """Get manufacturer name this parser handles."""
        return self._metadata.manufacturer

    @property
    def supported_formats(self) -> list[str]:
        """Get list of file formats this parser supports."""
        return self._metadata.supported_formats

    def __str__(self) -> str:
        """String representation of parser."""
        return f"{self.parser_id} (v{self._metadata.parser_version}): {self._metadata.description}"

    def __repr__(self) -> str:
        """Developer representation of parser."""
        return f"<{self.__class__.__name__} id={self.parser_id} manufacturer={self.manufacturer}>"


class ParserError(Exception):
    """Base exception for parser errors."""

    def __init__(self, message: str, parser: DeviceParser | None = None):
        super().__init__(message)
        self.parser = parser
