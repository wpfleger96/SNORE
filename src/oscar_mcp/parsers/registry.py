"""
Parser Registry System

Central registry for all device parsers. Provides auto-detection,
parser lookup, and unified access to all parser implementations.

Key Features:
- Auto-detect device type from data files
- Register parsers automatically on import
- Query available parsers
- Confidence-based selection when multiple parsers match
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging

from oscar_mcp.parsers.base import DeviceParser, ParserDetectionResult


logger = logging.getLogger(__name__)


class ParserRegistry:
    """
    Global registry for all device parsers.

    Parsers self-register by calling register() when their module is imported.
    This allows the system to automatically discover and use all available parsers.

    Usage:
        # In a parser module:
        parser_registry.register(ResmedEDFParser())

        # In application code:
        parser = parser_registry.detect_parser(data_path)
        for session in parser.parse_sessions(data_path):
            process(session)
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        self._parsers: List[DeviceParser] = []
        self._parsers_by_id: Dict[str, DeviceParser] = {}
        self._parsers_by_manufacturer: Dict[str, List[DeviceParser]] = {}
        logger.info("Parser registry initialized")

    def register(self, parser: DeviceParser) -> None:
        """
        Register a new parser.

        Args:
            parser: DeviceParser instance to register

        Raises:
            ValueError: If parser ID is already registered

        Example:
            registry.register(ResmedEDFParser())
        """
        parser_id = parser.parser_id

        # Check for duplicate IDs
        if parser_id in self._parsers_by_id:
            existing = self._parsers_by_id[parser_id]
            raise ValueError(
                f"Parser ID '{parser_id}' already registered by {existing.__class__.__name__}"
            )

        # Add to main list
        self._parsers.append(parser)

        # Index by ID
        self._parsers_by_id[parser_id] = parser

        # Index by manufacturer
        manufacturer = parser.manufacturer.lower()
        if manufacturer not in self._parsers_by_manufacturer:
            self._parsers_by_manufacturer[manufacturer] = []
        self._parsers_by_manufacturer[manufacturer].append(parser)

        logger.info(f"Registered parser: {parser}")

    def unregister(self, parser_id: str) -> bool:
        """
        Unregister a parser by ID.

        Args:
            parser_id: ID of parser to remove

        Returns:
            True if parser was removed, False if not found
        """
        if parser_id not in self._parsers_by_id:
            return False

        parser = self._parsers_by_id[parser_id]

        # Remove from main list
        self._parsers.remove(parser)

        # Remove from ID index
        del self._parsers_by_id[parser_id]

        # Remove from manufacturer index
        manufacturer = parser.manufacturer.lower()
        if manufacturer in self._parsers_by_manufacturer:
            self._parsers_by_manufacturer[manufacturer].remove(parser)
            if not self._parsers_by_manufacturer[manufacturer]:
                del self._parsers_by_manufacturer[manufacturer]

        logger.info(f"Unregistered parser: {parser_id}")
        return True

    def detect_parser(
        self, path: Path, manufacturer_hint: Optional[str] = None
    ) -> Optional[DeviceParser]:
        """
        Auto-detect which parser can handle the data at the given path.

        Tries all registered parsers and returns the one with highest confidence.
        If multiple parsers match with equal confidence, the first one wins.

        Args:
            path: Path to data directory/file
            manufacturer_hint: Optional manufacturer name to try first

        Returns:
            DeviceParser that can handle the data, or None if no match

        Example:
            parser = registry.detect_parser(Path("~/CPAP_Data"))
            if parser:
                print(f"Detected: {parser.manufacturer}")
            else:
                print("No compatible parser found")
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return None

        # Build list of parsers to try
        parsers_to_try = []

        # If manufacturer hint provided, try those first
        if manufacturer_hint:
            hint_lower = manufacturer_hint.lower()
            if hint_lower in self._parsers_by_manufacturer:
                parsers_to_try.extend(self._parsers_by_manufacturer[hint_lower])

        # Add all other parsers
        for parser in self._parsers:
            if parser not in parsers_to_try:
                parsers_to_try.append(parser)

        # Try each parser and track results
        best_match: Optional[Tuple[DeviceParser, ParserDetectionResult]] = None

        for parser in parsers_to_try:
            try:
                result = parser.detect(path)

                if result.detected:
                    logger.info(
                        f"Parser {parser.parser_id} detected data with confidence {result.confidence}"
                    )

                    # Track best match
                    if best_match is None or result.confidence > best_match[1].confidence:
                        best_match = (parser, result)

                    # If perfect confidence, stop searching
                    if result.confidence >= 1.0:
                        break

            except Exception as e:
                logger.warning(f"Parser {parser.parser_id} detection failed: {e}")
                continue

        if best_match:
            parser, result = best_match
            logger.info(f"Selected parser: {parser.parser_id} (confidence: {result.confidence})")
            return parser

        logger.warning(f"No parser detected for path: {path}")
        return None

    def detect_all_parsers(
        self, path: Path, manufacturer_hint: Optional[str] = None
    ) -> List[Tuple[DeviceParser, ParserDetectionResult]]:
        """
        Detect all parsers that can handle the data at the given path.

        Unlike detect_parser which returns only the best match, this method
        returns all parsers that successfully detect the data, sorted by
        confidence (highest first).

        Useful for discovering multiple profiles or data sources.

        Args:
            path: Path to data directory/file
            manufacturer_hint: Optional manufacturer name to try first

        Returns:
            List of (DeviceParser, ParserDetectionResult) tuples, sorted by confidence

        Example:
            results = registry.detect_all_parsers(Path("~/OSCAR/Profiles"))
            for parser, detection in results:
                print(f"{parser.manufacturer}: {detection.message}")
                print(f"  Data root: {detection.metadata.get('data_root')}")
        """
        path = Path(path)

        if not path.exists():
            logger.warning(f"Path does not exist: {path}")
            return []

        parsers_to_try = []

        if manufacturer_hint:
            hint_lower = manufacturer_hint.lower()
            if hint_lower in self._parsers_by_manufacturer:
                parsers_to_try.extend(self._parsers_by_manufacturer[hint_lower])

        for parser in self._parsers:
            if parser not in parsers_to_try:
                parsers_to_try.append(parser)

        matches = []

        for parser in parsers_to_try:
            try:
                result = parser.detect(path)

                if result.detected:
                    logger.info(
                        f"Parser {parser.parser_id} detected data with confidence {result.confidence}"
                    )
                    matches.append((parser, result))

            except Exception as e:
                logger.warning(f"Parser {parser.parser_id} detection failed: {e}")
                continue

        matches.sort(key=lambda x: x[1].confidence, reverse=True)

        if matches:
            logger.info(f"Found {len(matches)} parser(s) for path: {path}")
        else:
            logger.warning(f"No parsers detected for path: {path}")

        return matches

    def get_parser(self, parser_id: str) -> Optional[DeviceParser]:
        """
        Get a specific parser by ID.

        Args:
            parser_id: Parser identifier (e.g., "resmed_edf")

        Returns:
            DeviceParser or None if not found

        Example:
            parser = registry.get_parser("resmed_edf")
        """
        return self._parsers_by_id.get(parser_id)

    def get_parsers_by_manufacturer(self, manufacturer: str) -> List[DeviceParser]:
        """
        Get all parsers for a specific manufacturer.

        Args:
            manufacturer: Manufacturer name (case-insensitive)

        Returns:
            List of DeviceParser instances for that manufacturer

        Example:
            parsers = registry.get_parsers_by_manufacturer("ResMed")
            # Returns: [ResmedEDFParser, ResmedS9Parser, ...]
        """
        manufacturer_lower = manufacturer.lower()
        return self._parsers_by_manufacturer.get(manufacturer_lower, [])

    def list_parsers(self) -> List[DeviceParser]:
        """
        Get list of all registered parsers.

        Returns:
            List of all DeviceParser instances

        Example:
            for parser in registry.list_parsers():
                print(f"{parser.manufacturer}: {parser.parser_id}")
        """
        return self._parsers.copy()

    def list_manufacturers(self) -> List[str]:
        """
        Get list of all supported manufacturers.

        Returns:
            List of manufacturer names

        Example:
            manufacturers = registry.list_manufacturers()
            # Returns: ["ResMed", "Philips", ...]
        """
        return list(set(p.manufacturer for p in self._parsers))

    def get_parser_info(self) -> Dict[str, Dict]:
        """
        Get detailed information about all registered parsers.

        Returns:
            Dictionary mapping parser IDs to their metadata

        Example:
            info = registry.get_parser_info()
            # Returns:
            # {
            #     "resmed_edf": {
            #         "manufacturer": "ResMed",
            #         "version": "1.0.0",
            #         "formats": ["EDF+", "EDF"],
            #         "models": ["AirSense 10", "AirSense 11"],
            #         ...
            #     },
            #     ...
            # }
        """
        info = {}
        for parser in self._parsers:
            metadata = parser.metadata
            info[parser.parser_id] = {
                "manufacturer": metadata.manufacturer,
                "version": metadata.parser_version,
                "formats": metadata.supported_formats,
                "models": metadata.supported_models,
                "description": metadata.description,
            }
        return info

    @property
    def parser_count(self) -> int:
        """Get number of registered parsers."""
        return len(self._parsers)

    def __len__(self) -> int:
        """Return number of registered parsers."""
        return len(self._parsers)

    def __repr__(self) -> str:
        """Developer representation."""
        return f"<ParserRegistry parsers={self.parser_count} manufacturers={len(self.list_manufacturers())}>"

    def __str__(self) -> str:
        """Human-readable representation."""
        manufacturers = self.list_manufacturers()
        return f"ParserRegistry with {self.parser_count} parsers for {len(manufacturers)} manufacturers"


# Global singleton registry instance
# Import this in other modules to register/use parsers
parser_registry = ParserRegistry()


def get_registry() -> ParserRegistry:
    """
    Get the global parser registry instance.

    Returns:
        Global ParserRegistry singleton

    Example:
        from oscar_mcp.parsers.registry import get_registry

        registry = get_registry()
        parser = registry.detect_parser(path)
    """
    return parser_registry
