"""
Abstract Parser Interface

This module defines the base class that ALL device parsers must implement.
This ensures a consistent interface across all manufacturers and file formats.

Key Principle: Any new device parser just needs to inherit from DeviceParser
and implement these methods - the rest of the system automatically works.
"""

from __future__ import annotations

import logging
import os

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from concurrent.futures import as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from snore.parsers.types import DataRoot, ParserMetadata
from snore.parsers.unified import DeviceInfo, UnifiedSession
from snore.utils.parse_pool import cancel_pending, get_pool

logger = logging.getLogger(__name__)

__all__ = [
    "DeviceParser",
    "ParserDetectionResult",
    "ParserMetadata",
    "RawFileManifest",
    "build_root_metadata",
    "date_in_range",
    "filter_units_by_date",
    "night_date",
]

# Sessions starting before this wall-clock time belong to the previous night.
_NIGHT_SPLIT_TIME = time(12, 0)


def night_date(start: datetime) -> date:
    """Return the night a session starting at ``start`` belongs to.

    Sessions starting before noon belong to the previous day's night; noon
    itself starts a new night (OSCAR's day-split rule).  This is the date a
    session's ``Day`` is filed under and the date import ranges select by.
    """
    if start.time() < _NIGHT_SPLIT_TIME:
        return start.date() - timedelta(days=1)
    return start.date()


def date_in_range(value: date, date_from: str | None, date_to: str | None) -> bool:
    """Whether ``value`` lies within the inclusive ISO-date range.

    A None bound is open, so with both None every date is in range.
    """
    on_or_after = not date_from or value >= datetime.fromisoformat(date_from).date()
    on_or_before = not date_to or value <= datetime.fromisoformat(date_to).date()
    return on_or_after and on_or_before


def build_root_metadata(roots: list[DataRoot]) -> dict[str, Any]:
    """Build the shared ``detect`` metadata dict from discovered data roots.

    The first root supplies the primary fields; ``all_roots`` and
    ``root_metadata`` describe every discovered root.  Both the ResMed and
    OSCAR ``detect`` methods return this identical structure.
    """
    first = roots[0]
    return {
        "data_root": str(first.path),
        "structure_type": first.structure_type,
        "profile_name": first.profile_name,
        "device_serial": first.device_serial,
        "all_roots": [str(r.path) for r in roots],
        "root_metadata": {
            str(r.path): {
                "profile_name": r.profile_name,
                "structure_type": r.structure_type,
                "device_serial": r.device_serial,
            }
            for r in roots
        },
    }


def filter_units_by_date[Unit](
    units: list[Unit],
    key_date_fn: Callable[[Unit], date | None],
    date_from: str | None,
    date_to: str | None,
) -> list[Unit]:
    """Filter parse units to those whose night date is within the range.

    ``key_date_fn`` maps a unit to its night date (see ``night_date``), or None
    when the date is unknown/unparseable — such units are always kept (the
    caller logs why).  ``date_from``/``date_to`` are inclusive ISO dates; when
    both are None the input list is returned unchanged.
    """
    if not (date_from or date_to):
        return units

    kept: list[Unit] = []
    for unit in units:
        unit_date = key_date_fn(unit)
        if unit_date is None or date_in_range(unit_date, date_from, date_to):
            kept.append(unit)
    return kept


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


@dataclass
class RawFileManifest:
    """Generic manifest of raw files grouped by night date.

    Used by BackupService and ExportService to copy files without
    knowing manufacturer-specific file layouts.
    """

    device_files: list[Path] = field(default_factory=list)
    """Device-level files (e.g., settings summary, identification). Copied once."""

    nights: dict[date, list[Path]] = field(default_factory=dict)
    """Per-night session files keyed by night date."""

    source_root: Path = field(default_factory=lambda: Path("."))
    """Root path that all file paths are relative children of.
    Export uses file.relative_to(source_root) for destination path construction."""

    files_copied: int = 0
    """Number of files that were newly copied (not skipped)."""

    files_skipped: int = 0
    """Number of files that were skipped (already existed with matching size/mtime)."""

    files_failed: int = 0
    """Number of files where the copy failed (I/O or metadata error)."""

    @property
    def total_files(self) -> int:
        """Total number of files across device files and all nights."""
        return len(self.device_files) + sum(
            len(files) for files in self.nights.values()
        )

    @property
    def total_bytes(self) -> int:
        """Total size of all files in bytes."""
        total = sum(f.stat().st_size for f in self.device_files if f.exists())
        for files in self.nights.values():
            total += sum(f.stat().st_size for f in files if f.exists())
        return total


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

    #: Whether a single parse unit may yield no session (dropped by a
    #: post-parse guard or a parse failure).  True selects the submit-all +
    #: stop-after-``limit``-yields driver; False lets ``limit`` pre-slice the
    #: sorted units up front for a bounded, deterministic first-N set.
    _unit_may_yield_nothing: bool = True

    def parse_sessions(
        self,
        path: Path,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = None,
        sort_by: str | None = None,
        parallel: bool = True,
        progress_callback: Callable[[str], None] | None = None,
        timezone_name: str | None = None,
    ) -> Iterator[UnifiedSession]:
        """
        Parse all sessions from the given path and yield unified sessions.

        Template method shared by every parser: resolve the parse units, filter
        them by date, then run each either across the process pool (parallel and
        more than one unit) or sequentially, mapping a crashed worker pool to a
        RuntimeError.  Subclasses supply the manufacturer-specific behavior via
        the hooks below.

        ``limit`` is honored one of two ways, selected by
        ``_unit_may_yield_nothing``: when a unit always yields exactly one
        session (flag False) the sorted, filtered units are pre-sliced to the
        first ``limit`` — bounded work, deterministic set; when a unit may yield
        nothing (flag True) all units are submitted and the driver stops after
        ``limit`` yields, cancelling the rest, since pre-slicing could
        under-deliver.

        Args:
            path: Path to data directory/file
            date_from: Optional start date filter (ISO format: YYYY-MM-DD)
            date_to: Optional end date filter (ISO format: YYYY-MM-DD)
            limit: Optional maximum number of sessions to yield
            sort_by: Optional sort order - "date-asc", "date-desc", or None for filesystem order
            progress_callback: Optional callback invoked with a progress message string as each session is parsed
            timezone_name: Optional user-declared IANA timezone for the owning
                profile.  Parsers whose source data encodes absolute instants
                (e.g. OSCAR epoch-ms) use it to produce device-local wall-clock
                datetimes; parsers whose source data is already device-local
                wall-clock (e.g. ResMed EDF) ignore it.

        Yields:
            UnifiedSession objects

        Raises:
            ParserError: If parsing fails
        """
        path = Path(path)

        root, units = self._resolve_units(path, sort_by, timezone_name)
        units = filter_units_by_date(
            units, lambda u: self._unit_date(u, timezone_name), date_from, date_to
        )
        if limit is not None and not self._unit_may_yield_nothing:
            units = units[:limit]
        total = len(units)
        ctx = self._build_context(root, timezone_name)

        sessions_yielded = 0
        completed = 0

        def emit_progress() -> None:
            nonlocal completed
            completed += 1
            if progress_callback:
                progress_callback(f"Parsing session {completed}/{total}...")

        if parallel and len(units) > 1:
            logger.info(
                f"Parsing {total} sessions in parallel with {os.cpu_count()} workers"
            )
            futures: dict[Any, Any] = {}
            try:
                pool = get_pool()
                for unit in units:
                    fn, args = self._worker_dispatch(
                        unit, ctx, root, date_from, date_to
                    )
                    futures[pool.submit(fn, *args)] = unit

                for future in as_completed(futures):
                    if limit is not None and sessions_yielded >= limit:
                        cancel_pending(futures)
                        break

                    try:
                        session = future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as e:
                        logger.error(
                            f"Failed to parse {self._unit_label(futures[future])}: {e}"
                        )
                        emit_progress()
                        continue

                    emit_progress()
                    if session is None:
                        continue

                    yield session
                    sessions_yielded += 1
            except BrokenProcessPool as exc:
                raise RuntimeError(
                    f"Parser worker process crashed: {exc}. "
                    "Reduce SNORE_PARSE_MAX_WORKERS if memory is constrained."
                ) from exc
            finally:
                cancel_pending(futures)
        else:
            for unit in units:
                if limit is not None and sessions_yielded >= limit:
                    break

                try:
                    session = self._parse_unit_sequential(
                        unit, ctx, root, date_from, date_to
                    )
                except Exception as e:
                    logger.error(f"Failed to parse {self._unit_label(unit)}: {e}")
                    emit_progress()
                    continue

                emit_progress()
                if session is None:
                    continue

                yield session
                sessions_yielded += 1

    # ------------------------------------------------------------------
    # parse_sessions hooks — implemented by each parser
    # ------------------------------------------------------------------

    @abstractmethod
    def _resolve_units(
        self, path: Path, sort_by: str | None, timezone_name: str | None
    ) -> tuple[Any, list[Any]]:
        """Resolve the data root and its ordered list of parse units.

        Returns ``(root, units)``.  ``root`` is passed unchanged to the other
        hooks; each element of ``units`` is one session (or session chain) to
        parse.  ``sort_by`` orders the units; ``timezone_name`` lets parsers
        that need it validate the zone before parsing.
        """

    @abstractmethod
    def _unit_date(self, unit: Any, timezone_name: str | None) -> date | None:
        """Return a unit's night date for range filtering, or None.

        The night date is ``night_date`` of the unit's session start, the same
        date its ``Day`` is filed under, not the start's calendar date.
        """

    @abstractmethod
    def _build_context(self, root: Any, timezone_name: str | None) -> Any:
        """Build the shared per-parse context passed to the per-unit hooks.

        Runs once after unit resolution and filtering; use it for work shared
        across all units (device info, caches, the declared timezone).
        """

    @abstractmethod
    def _worker_dispatch(
        self,
        unit: Any,
        ctx: Any,
        root: Any,
        date_from: str | None,
        date_to: str | None,
    ) -> tuple[Callable[..., UnifiedSession | None], tuple[Any, ...]]:
        """Return ``(worker_fn, args)`` for parsing ``unit`` in a subprocess.

        ``worker_fn`` MUST be a module-level function (picklable for the process
        pool), never a bound method.
        """

    @abstractmethod
    def _parse_unit_sequential(
        self,
        unit: Any,
        ctx: Any,
        root: Any,
        date_from: str | None,
        date_to: str | None,
    ) -> UnifiedSession | None:
        """Parse ``unit`` in-process (sequential path); None drops the unit."""

    def _unit_label(self, unit: Any) -> str:
        """Short identifier for ``unit`` used in failure logs.

        Default is the raw unit; parsers override to return a bounded, readable
        id rather than dumping a large unit structure.
        """
        return str(unit)

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

    # ------------------------------------------------------------------
    # Raw file backup / export (optional — override in subclasses)
    # ------------------------------------------------------------------

    @property
    def supports_raw_backup(self) -> bool:
        """Whether this parser supports raw file backup and export.

        Parsers that read from already-backed-up data (e.g., OSCAR binary
        cache) should leave this as False.
        """
        return False

    def backup_raw_data(
        self,
        source_root: Path,
        dest_root: Path,
        progress_callback: Callable[[str], None] | None = None,
    ) -> RawFileManifest:
        """Copy raw data files from source to dest, preserving layout.

        Skips files that already exist at dest with matching size and mtime.
        Returns a RawFileManifest describing what was backed up.

        The parser knows its own file structure and versioning rules (e.g.,
        STR.edf historical snapshots for ResMed). The service layer calls
        this method generically without knowing file layout details.

        MUST be stateless — use only the passed path arguments, never
        instance state from detect() (which may be stale or from a
        different path).

        Args:
            source_root: Path to raw data (e.g., SD card root).
            dest_root: Backup destination (e.g., ~/.snore/raw/<serial>/).
            progress_callback: Optional callback for progress messages.

        Raises:
            NotImplementedError: If this parser does not support raw backup.
        """
        raise NotImplementedError(
            f"Parser '{self.parser_id}' does not support raw file backup"
        )

    def get_raw_file_manifest(
        self,
        root: Path,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> RawFileManifest:
        """Return manifest of raw files at root, optionally filtered by date.

        Used by ExportService to enumerate files for export without knowing
        the manufacturer-specific file layout.

        MUST be stateless — same requirements as backup_raw_data().

        Raises:
            NotImplementedError: If this parser does not support raw file manifest.
        """
        raise NotImplementedError(
            f"Parser '{self.parser_id}' does not support raw file manifest"
        )

    def trim_device_summary(
        self,
        output_root: Path,
        date_from: date,
        date_to: date,
    ) -> None:
        """Trim device-level summary files to the given date range.

        Called after files are copied to the export output directory.
        Parsers that have a trimmable summary file (e.g., ResMed's STR.edf)
        override this to rewrite the file in-place.

        Default: no-op.
        """
        return  # noqa: B027

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
