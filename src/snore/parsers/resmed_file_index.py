"""ResMed file indexing utilities.

Stable public functions for scanning and grouping ResMed EDF files.
Used by ResmedEDFParser for parsing and by backup/export methods
for file discovery. These are extracted from ResmedEDFParser to
provide a clean API boundary — services never call parser internals.
"""

from __future__ import annotations

import logging
import os
import re

from datetime import datetime, timedelta
from pathlib import Path

from snore.parsers.base import session_night

logger = logging.getLogger(__name__)

# Matches ResMed DATALOG EDF filenames: YYYYMMDD_HHMMSS_TYPE.edf
_EDF_FILENAME_PATTERN = re.compile(r"\d{8}_\d{6}_[A-Z0-9]+\.edf")
_EDF_SESSION_PATTERN = re.compile(r"(\d{8}_\d{6})_([A-Z0-9]+)\.edf")


def is_resmed_root(path: Path) -> bool:
    """Check if path contains ResMed data signature (STR.edf + DATALOG)."""
    if not path.is_dir():
        return False
    return (path / "STR.edf").exists() and (path / "DATALOG").is_dir()


def get_night_date(timestamp: datetime) -> str:
    """Get the night date for a session using OSCAR's noon cutoff rule.

    Sessions starting before noon belong to the previous day's night.
    This matches ResMed's commercial software and OSCAR's behavior.

    Args:
        timestamp: Session start time.

    Returns:
        Night date as YYYYMMDD string.
    """
    return session_night(timestamp).strftime("%Y%m%d")


def scan_edf_files(datalog_dir: Path) -> list[Path]:
    """Scan for ResMed EDF files using os.scandir (faster than rglob).

    Recursively scans the DATALOG directory and pre-filters by the
    ResMed filename pattern (YYYYMMDD_HHMMSS_TYPE.edf).

    Args:
        datalog_dir: Directory to scan (typically source_root / "DATALOG").

    Returns:
        List of EDF file paths found.
    """
    edf_files: list[Path] = []

    def _scan_dir(path: Path) -> None:
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith(".edf"):
                        if _EDF_FILENAME_PATTERN.match(entry.name):
                            edf_files.append(Path(entry.path))
                    elif entry.is_dir():
                        _scan_dir(Path(entry.path))
        except PermissionError:
            logger.warning(f"Permission denied accessing directory: {path}")

    _scan_dir(datalog_dir)
    return edf_files


def index_segment_files(datalog_dir: Path) -> dict[str, dict[str, Path]]:
    """Scan DATALOG and return a flat session_id -> {file_type: Path} mapping.

    This is the single scan point used by both ``group_session_files`` and
    ``chain_session_segments``.

    Args:
        datalog_dir: Directory to scan (typically source_root / "DATALOG").

    Returns:
        Dict mapping session_id (YYYYMMDD_HHMMSS) to its file-type map.
    """
    segments: dict[str, dict[str, Path]] = {}
    for edf_file in scan_edf_files(datalog_dir):
        match = _EDF_SESSION_PATTERN.match(edf_file.name)
        if not match:
            continue
        session_id = match.group(1)
        file_type = match.group(2)
        if session_id not in segments:
            segments[session_id] = {}
        segments[session_id][file_type] = edf_file
    return segments


def group_session_files(
    datalog_dir: Path,
) -> dict[str, dict[str, dict[str, Path]]]:
    """Group EDF files by night date (noon-to-noon periods).

    Multiple sessions within the same night (mask removals/bathroom breaks)
    are grouped together to match OSCAR's behavior.

    Args:
        datalog_dir: Directory containing DATALOG files.

    Returns:
        Dict mapping night_date -> session_id -> file_type -> Path.
        Example::

            {
                "20240621": {
                    "20240621_013454": {
                        "BRP": Path("...BRP.edf"),
                        "PLD": Path("...PLD.edf"),
                    },
                    "20240621_053022": {
                        "BRP": Path("...BRP.edf"),
                    },
                },
            }
    """
    groups: dict[str, dict[str, dict[str, Path]]] = {}

    for session_id, file_map in index_segment_files(datalog_dir).items():
        timestamp = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        night = get_night_date(timestamp)
        if night not in groups:
            groups[night] = {}
        groups[night][session_id] = file_map

    return groups


def flatten_night_files(
    grouped: dict[str, dict[str, dict[str, Path]]],
) -> dict[str, list[Path]]:
    """Flatten grouped session files to night_date -> flat list of paths.

    Useful for backup/export where we don't need the session_id/file_type
    structure, just all files for a given night.
    """
    result: dict[str, list[Path]] = {}
    for night_date, sessions in grouped.items():
        files: list[Path] = []
        for file_map in sessions.values():
            files.extend(file_map.values())
        result[night_date] = files
    return result


# OSCAR's "Combine Close Sessions" default is 240 minutes (14400 seconds).
# This threshold correctly:
# - Chains noon-rollover pairs (observed gap: 36-47 s) into one session.
# - Splits isolated diagnostic blips 5.76 h before sleep into separate sessions.
OSCAR_COMBINE_CLOSE_SECONDS: int = 4 * 60 * 60


def get_segment_duration_seconds(
    session_id: str, files: dict[str, Path]
) -> float | None:
    """Return the total recording duration of a segment in seconds.

    Reads the EDF header (first 256 bytes) for the first available file type
    among BRP, PLD, SA2 — without opening the file via pyedflib. This reuses
    the same byte-offset layout as ``formats.edf.get_edf_record_count`` but
    also extracts record_duration (bytes 244:252) to compute total length.

    Args:
        session_id: Segment identifier (used only for contextual logging).
        files: Mapping of file type to path for a single EDF segment.

    Returns:
        Duration in seconds (num_records × record_duration) when a readable
        file with positive values is found; None otherwise.
        ``num_records == -1`` (EDF+C unknown) is treated as unusable.
        Values exceeding 172800 s (48 h) are treated as corrupt header data.
    """
    for file_type in ("BRP", "PLD", "SA2"):
        path = files.get(file_type)
        if path is None or not path.exists():
            continue
        try:
            with open(path, "rb") as f:
                header = f.read(256)
            if len(header) < 256:
                continue
            num_records_str = header[236:244].decode("ascii", errors="ignore").strip()
            record_duration_str = (
                header[244:252].decode("ascii", errors="ignore").strip()
            )
            num_records = int(num_records_str)
            record_duration = float(record_duration_str)
            # -1 means "number of records unknown" in EDF+C — treat as unusable.
            if num_records == -1:
                continue
            if num_records > 0 and record_duration > 0:
                duration = num_records * record_duration
                # No physical segment can exceed ~24 h (noon rollover); cap at
                # 48 h to reject corrupt headers that would poison chain end times.
                if duration > 172800:
                    continue
                return duration
        except (ValueError, OSError):
            logger.debug(
                f"Could not read EDF header for {file_type} in segment {session_id}"
            )
            continue
    return None


def chain_session_segments(
    datalog_dir: Path,
    max_gap_seconds: int = OSCAR_COMBINE_CLOSE_SECONDS,
) -> list[tuple[str, str, dict[str, dict[str, Path]]]]:
    """Group EDF segments into chains based on chronological proximity.

    Replaces noon-bucket merging with proximity-based chaining: segments are
    linked into a chain while the gap between the previous chain's end and the
    next segment's start is within ``max_gap_seconds``. Each chain becomes one
    session; its night date is derived from the chain's first segment using the
    same noon-cutoff rule as ``get_night_date``.

    This correctly handles:
    - Noon rollover: therapy running at 12:00 produces two EDF segment groups
      ~36-47 s apart that chain into one session.
    - Diagnostic blips: a brief self-test segment 5+ h before real sleep does
      NOT chain with the sleep session.
    - CSL/EVE annotation stubs: real ResMed devices write a CSL+EVE-only group
      ~9-14 s before each BRP/PLD/SA2 waveform group. These stubs have no
      duration-bearing file types, so their duration is unknown. The walker
      uses the segment's own start time as a lower-bound end (equivalent to
      zero duration), so `last_end` is always a real datetime and gaps are
      measured from the latest known activity in the chain.

    Args:
        datalog_dir: DATALOG directory to scan.
        max_gap_seconds: Maximum gap in seconds to chain segments together.
            Defaults to OSCAR_COMBINE_CLOSE_SECONDS (4 h).

    Returns:
        Chronologically ordered list of ``(night_date, chain_id, segments)``
        tuples where:
        - ``night_date`` is YYYYMMDD (noon-cutoff applied to chain start).
        - ``chain_id`` is the first segment's session_id.
        - ``segments`` maps ``session_id -> {file_type: Path}``.
    """
    all_sessions = index_segment_files(datalog_dir)

    if not all_sessions:
        return []

    # Lexicographic sort == chronological for YYYYMMDD_HHMMSS format.
    sorted_ids = sorted(all_sessions)

    chains: list[tuple[str, str, dict[str, dict[str, Path]]]] = []

    first_id = sorted_ids[0]
    first_files = all_sessions[first_id]
    chain_first_id = first_id
    chain_segments: dict[str, dict[str, Path]] = {first_id: first_files}
    first_start = datetime.strptime(first_id, "%Y%m%d_%H%M%S")
    first_duration = get_segment_duration_seconds(first_id, first_files)
    # Use seg_start as a lower-bound end when duration is unknown. This keeps
    # last_end a real datetime so gaps are always measurable, and naturally
    # chains CSL/EVE-only stub groups that precede BRP/PLD/SA2 groups by ~9-14 s.
    last_end: datetime = first_start + timedelta(seconds=first_duration or 0)

    for session_id in sorted_ids[1:]:
        files = all_sessions[session_id]
        seg_start = datetime.strptime(session_id, "%Y%m%d_%H%M%S")

        gap_seconds = (seg_start - last_end).total_seconds()

        if gap_seconds <= max_gap_seconds:
            chain_segments[session_id] = files
            duration = get_segment_duration_seconds(session_id, files)
            seg_end_lower = seg_start + timedelta(seconds=duration or 0)
            last_end = max(last_end, seg_end_lower)
        else:
            night_date = get_night_date(
                datetime.strptime(chain_first_id, "%Y%m%d_%H%M%S")
            )
            chains.append((night_date, chain_first_id, chain_segments))
            chain_first_id = session_id
            chain_segments = {session_id: files}
            duration = get_segment_duration_seconds(session_id, files)
            last_end = seg_start + timedelta(seconds=duration or 0)

    night_date = get_night_date(datetime.strptime(chain_first_id, "%Y%m%d_%H%M%S"))
    chains.append((night_date, chain_first_id, chain_segments))

    return chains
