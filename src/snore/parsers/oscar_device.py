"""
OSCAR Device Parser

Parser for OSCAR binary cache files (.000 summary, .001 events).
Supports importing data from OSCAR Profiles directory structure.
"""

import logging
import math
import xml.etree.ElementTree as ET

from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import numpy as np

from snore.constants import (
    CPAP_CLEAR_AIRWAY,
    CPAP_EPAP,
    CPAP_EPAP_HI,
    CPAP_EPAP_LO,
    CPAP_FLOW_LIMIT,
    CPAP_HYPOPNEA,
    CPAP_IPAP,
    CPAP_IPAP_HI,
    CPAP_LEAK,
    CPAP_MASK_PRESSURE,
    CPAP_MINUTE_VENT,
    CPAP_MODE,
    CPAP_OBSTRUCTIVE,
    CPAP_PERIODIC_BREATHING,
    CPAP_PRESSURE,
    CPAP_PRESSURE_MAX,
    CPAP_PRESSURE_MIN,
    CPAP_PS,
    CPAP_PS_MAX,
    CPAP_PS_MIN,
    CPAP_RAMP_PRESSURE,
    CPAP_RAMP_TIME,
    CPAP_RERA,
    CPAP_RESPRATE,
    CPAP_TIDAL_VOLUME,
    OSCAR_BINARY_PARSER_ID,
    OXI_PULSE,
    OXI_SPO2,
    PARSER_MAX_SEARCH_DEPTH,
    RMAS1X_CYCLE,
    RMAS1X_RISE_ENABLE,
    RMAS1X_RISE_TIME,
    RMAS1X_TI_MAX,
    RMAS1X_TI_MIN,
    RMAS1X_TRIGGER,
    RMS9_EPR,
    RMS9_EPR_LEVEL,
    RMS9_RAMP_ENABLE,
    RMVENT_AUTO_EPAP,
    RMVENT_START_EPAP,
)
from snore.parsers.base import (
    DeviceParser,
    ParserDetectionResult,
    ParserError,
    ParserMetadata,
    build_root_metadata,
    night_date,
)
from snore.parsers.discovery import DataRoot, DataRootFinder
from snore.parsers.oscar_events import parse_events_file
from snore.parsers.oscar_mappings import (
    OSCAR_EVENT_CHANNEL_IDS,
    OSCAR_EVENT_TYPE_MAP,
    OSCAR_MODE_MAP,
    OSCAR_WAVEFORM_CHANNEL_IDS,
    OSCAR_WAVEFORM_TYPE_MAP,
    OSCAR_WAVEFORM_UNITS,
)
from snore.parsers.oscar_summary import parse_summary_file
from snore.parsers.settings_labels import CYCLE_MAP, EPR_TYPE_MAP, TRIGGER_MAP
from snore.parsers.types import EventListType, SessionEvents, SessionSummary
from snore.parsers.unified import (
    DeviceInfo,
    RespiratoryEvent,
    TherapyMode,
    TherapySettings,
    UnifiedSession,
    WaveformData,
    extract_basic_stats,
)
from snore.therapy_hours import TherapyHoursBasis, therapy_hours

logger = logging.getLogger(__name__)


def _epoch_to_wall_clock(epoch_seconds: float, timezone_name: str | None) -> datetime:
    """Convert epoch seconds to a naive wall-clock datetime.

    With a declared profile timezone, the wall clock is that zone's local time —
    matching ResMed's device-local semantics.  Without one, legacy behavior is
    kept: naive UTC wall-clock (see the KNOWN A6 INCONSISTENCY note in
    ``_parse_single_session``).
    """
    tz = ZoneInfo(timezone_name) if timezone_name else UTC
    return datetime.fromtimestamp(epoch_seconds, tz=tz).replace(tzinfo=None)


def _finite(value: Any) -> float | None:
    """Return ``value`` as a float when it is a finite real number, else None.

    OSCAR session settings are arbitrary QVariants: a value can be a string,
    a bool, a list or a non-finite float.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _non_negative(settings: dict[int, Any], channel: int) -> float | None:
    """Return a session setting when it is a finite number >= 0, else None."""
    value = _finite(settings.get(channel))
    return value if value is not None and value >= 0 else None


def _code(settings: dict[int, Any], channel: int) -> int | None:
    """Return an integer setting (enum code, minutes, milliseconds), or None.

    OSCAR stores settings as float32 ``gain * raw + offset``, so the value is
    rounded rather than truncated.
    """
    value = _non_negative(settings, channel)
    return round(value) if value is not None else None


def _pressure(settings: dict[int, Any], channel: int) -> float | None:
    """Return a pressure setting when it is a finite number > 0, else None.

    Only positive pressures are real: OSCAR's Yuwell loader stores -1 as a
    "fill in later" marker.
    """
    value = _finite(settings.get(channel))
    return value if value is not None and value > 0 else None


def _first(*values: float | None) -> float | None:
    """Return the first value that is not None."""
    return next((value for value in values if value is not None), None)


def _added(base: float | None, support: float | None) -> float | None:
    """Return ``base + support`` rounded to 0.01; None if unknown or overflowing."""
    if base is None or support is None:
        return None
    return _finite(round(base + support, 2))


def _pressure_ranges(
    min_epap: float | None,
    max_epap: float | None,
    min_ps: float | None,
    max_ps: float | None,
) -> dict[str, str]:
    """Format the known EPAP and PS bounds as ``other_settings`` entries."""
    bounds = {
        "min_epap": min_epap,
        "max_epap": max_epap,
        "min_ps": min_ps,
        "max_ps": max_ps,
    }
    return {key: f"{value:.1f}" for key, value in bounds.items() if value is not None}


class _Pressures(NamedTuple):
    """Resolved bilevel pressures plus their ``other_settings`` entries."""

    ipap: float | None
    epap: float | None
    ps: float | None
    other_settings: dict[str, str]


def _pressure_settings(settings: dict[int, Any], mode: TherapyMode) -> _Pressures:
    """Resolve IPAP, EPAP and PS the way the ResMed SD-card parser does.

    OSCAR loaders store different channel combinations per mode and manufacturer,
    so each value prefers its exact channel and falls back to what is stored.
    The headline EPAP is the minimum EPAP in variable-EPAP modes.
    """
    if mode == TherapyMode.IVAPS:
        return _ivaps_pressures(settings)

    fixed_ipap = _pressure(settings, CPAP_IPAP)
    fixed_epap = _pressure(settings, CPAP_EPAP)
    epap = _first(_pressure(settings, CPAP_EPAP_LO), fixed_epap)
    # A pressure-support range can start at 0 (ResMed ASV).
    min_ps = _non_negative(settings, CPAP_PS_MIN)
    max_ps = _non_negative(settings, CPAP_PS_MAX)
    ipap = _first(fixed_ipap, _pressure(settings, CPAP_IPAP_HI), _added(epap, max_ps))

    if min_ps is None and max_ps is None:
        ps = _pressure(settings, CPAP_PS)
        # Only a fixed IPAP/EPAP pair defines PS; IPAPHi - EPAPLo spans a range.
        if ps is None and fixed_ipap is not None and fixed_epap is not None:
            ps = round(fixed_ipap - fixed_epap, 2) if fixed_ipap > fixed_epap else None
        return _Pressures(ipap, epap, ps, {})

    # Pressure support varies within the range, so a stored single CPAP_PS
    # (Prisma writes PSMin there) is not reported, as on the SD-card path.
    max_epap = _first(_pressure(settings, CPAP_EPAP_HI), fixed_epap)
    ranges = _pressure_ranges(epap, max_epap, min_ps, max_ps)
    return _Pressures(ipap, epap, None, ranges)


def _ivaps_pressures(settings: dict[int, Any]) -> _Pressures:
    """Resolve iVAPS pressures from either OSCAR storage layout.

    Since OSCAR commit e26b095a (2026-06-28) the ResMed loader stores EPAP
    (auto EPAP off) or EPAPLo/EPAPHi (on), plus RMVENT_AutoEPAP and no IPAP.
    Older OSCAR used its generic branch: EPAP and EPAPLo/EPAPHi together (the
    latter read from the inactive S.i.MinEPAP/MaxEPAP), IPAPHi = active EPAP +
    max PS, and no auto EPAP flag. Both fill CPAP_PS from the VAuto signal bank,
    which does not apply to iVAPS, so PS is left unset.
    """
    fixed_epap = _pressure(settings, CPAP_EPAP)
    min_epap = _pressure(settings, CPAP_EPAP_LO)
    max_epap = _pressure(settings, CPAP_EPAP_HI)
    min_ps = _non_negative(settings, CPAP_PS_MIN)
    max_ps = _non_negative(settings, CPAP_PS_MAX)
    ipap_hi = _pressure(settings, CPAP_IPAP_HI)

    auto_code = _code(settings, RMVENT_AUTO_EPAP)
    auto: bool | None
    if auto_code is not None:
        auto = auto_code >= 1
    elif fixed_epap is not None and ipap_hi is not None and max_ps is not None:
        # Older OSCAR: IPAPHi was built from the active EPAP.
        auto = not math.isclose(ipap_hi - max_ps, fixed_epap, abs_tol=0.05)
    elif fixed_epap is not None or min_epap is not None:
        auto = fixed_epap is None
    else:
        auto = None

    if auto:
        epap = min_epap
    else:
        epap = min_epap = max_epap = fixed_epap

    ranges = _pressure_ranges(min_epap, max_epap, min_ps, max_ps)
    if auto is not None:
        ranges["epap_auto"] = str(auto)
    # IPAP is EPAP + max PS, as the SD-card parser derives it.
    return _Pressures(_added(epap, max_ps), epap, None, ranges)


def _epr_settings(settings: dict[int, Any]) -> tuple[int | None, str | None]:
    """Return ``(epr_level, epr_mode)`` as the SD-card parser's EPR decode does."""
    code = _code(settings, RMS9_EPR)
    level = _code(settings, RMS9_EPR_LEVEL)
    if level is None and code == 0:
        # OSCAR stores the EPR level only while EPR is on.
        level = 0
    mode = EPR_TYPE_MAP.get(code, "Unknown") if code is not None else None
    return (level if level is not None and level <= 3 else None), mode


def _resmed_other_settings(settings: dict[int, Any]) -> dict[str, str]:
    """Read ResMed bilevel timing and SmartRamp into ``other_settings`` form.

    These are ResMed-loader channels (0xE2xx) that other manufacturers'
    loaders never store.
    """
    other: dict[str, str] = {}
    for key, channel in (("ti_max", RMAS1X_TI_MAX), ("ti_min", RMAS1X_TI_MIN)):
        seconds = _non_negative(settings, channel)
        if seconds is not None:
            other[key] = f"{seconds:.1f}"

    for key, channel, labels in (
        ("trigger", RMAS1X_TRIGGER, TRIGGER_MAP),
        ("cycle", RMAS1X_CYCLE, CYCLE_MAP),
    ):
        code = _code(settings, channel)
        if code is not None:
            other[key] = labels.get(code, str(code))

    rise_enable = _code(settings, RMAS1X_RISE_ENABLE)
    rise_time = _code(settings, RMAS1X_RISE_TIME)
    if rise_enable is not None and rise_enable >= 1 and rise_time is not None:
        other["rise_time"] = str(rise_time)

    if _code(settings, RMS9_RAMP_ENABLE) == 2:
        other["smart_ramp"] = "True"
    return other


def _oscar_parse_session_worker(
    session_id: int,
    summary_path: "Path | None",
    events_path: "Path | None",
    device_info: DeviceInfo,
    base_path: "Path",
    timezone_name: "str | None" = None,
) -> UnifiedSession:
    """Parse one OSCAR session in a subprocess.

    Instantiates a fresh ``OscarDeviceParser`` — no meaningful instance state
    is transferred across the process boundary.
    """
    return OscarDeviceParser()._parse_single_session(
        session_id, summary_path, events_path, device_info, base_path, timezone_name
    )


class _OscarParseContext(NamedTuple):
    """Shared per-parse state passed to the OSCAR parse_sessions hooks."""

    device_info: DeviceInfo
    timezone_name: str | None


class OscarDeviceParser(DeviceParser):
    """
    Parser for OSCAR binary cache files.

    Supports the standard OSCAR profile structure:
    - Profiles/{profile_name}/
      - {manufacturer}_{serial}/
        - machine.xml
        - Summaries/{session_id}.000
        - Events/{session_id}.001
    """

    def __init__(self) -> None:
        """Initialize OSCAR parser."""
        super().__init__()
        self._data_roots: list[DataRoot] = []
        self._finder = DataRootFinder()

    def get_metadata(self) -> ParserMetadata:
        """Return OSCAR parser metadata."""
        return ParserMetadata(
            parser_id=OSCAR_BINARY_PARSER_ID,
            parser_version="1.2.0",
            manufacturer="OSCAR",
            supported_formats=["OSCAR Binary Cache"],
            supported_models=[
                "ResMed AirSense 10/11",
                "ResMed AirCurve 10/11",
                "Philips DreamStation",
                "F&P SleepStyle",
                "Any OSCAR-supported device",
            ],
            description="Parser for OSCAR binary cache files (.000/.001)",
            requires_libraries=["numpy"],
        )

    def detect(self, path: Path) -> ParserDetectionResult:
        """
        Detect OSCAR binary cache data structure.

        Searches for Profiles/{profile}/{device}/ structure with:
        - machine.xml (optional but preferred)
        - Summaries/*.000 files
        - Events/*.001 files
        """
        path = Path(path)

        if not path.exists():
            return ParserDetectionResult(
                detected=False, message=f"Path does not exist: {path}"
            )

        roots = self._finder.find_data_roots(
            path,
            validator_func=self._is_oscar_device_root,
            metadata_extractor_func=self._create_data_root,
            max_levels_down=PARSER_MAX_SEARCH_DEPTH,
        )

        if not roots:
            return ParserDetectionResult(
                detected=False,
                message="No OSCAR profile data found",
            )

        self._data_roots = roots

        profile_count = len(set(r.profile_name for r in roots if r.profile_name))
        device_count = len(roots)

        first_root = roots[0]
        return ParserDetectionResult(
            detected=True,
            confidence=first_root.confidence,
            message=f"Found {device_count} OSCAR device(s) across {profile_count} profile(s)",
            metadata=build_root_metadata(roots),
        )

    def _is_oscar_device_root(self, path: Path) -> bool:
        """Check if path is an OSCAR device data root."""
        summaries = path / "Summaries"
        events = path / "Events"

        if not (summaries.is_dir() and events.is_dir()):
            return False

        has_summary = any(summaries.glob("*.000"))
        has_events = any(events.glob("*.001"))

        return has_summary or has_events

    def _create_data_root(self, path: Path) -> DataRoot:
        """Create DataRoot with metadata from OSCAR directory structure."""
        parts = path.parts

        profile_name = None
        device_serial = None

        if "Profiles" in parts:
            profiles_idx = parts.index("Profiles")
            if profiles_idx + 1 < len(parts):
                profile_name = parts[profiles_idx + 1]
            if profiles_idx + 2 < len(parts):
                device_str = parts[profiles_idx + 2]
                if "_" in device_str:
                    device_serial = device_str.split("_", 1)[1]

        return DataRoot(
            path=path,
            structure_type="oscar_profile",
            profile_name=profile_name,
            device_serial=device_serial,
            confidence=0.95,
        )

    def get_device_info(self, path: Path) -> DeviceInfo:
        """
        Extract device information from machine.xml or directory name.

        machine.xml contains:
        <machine id="..." type="..." class="...">
            <brand>ResMed</brand>
            <model>AirSense 10 AutoSet</model>
            <serial>12345678</serial>
            ...
        </machine>
        """
        machine_xml = path / "machine.xml"

        if machine_xml.exists():
            try:
                return self._parse_machine_xml(machine_xml)
            except Exception as e:
                logger.warning(f"Failed to parse machine.xml: {e}")

        device_dir_name = path.name
        if "_" in device_dir_name:
            manufacturer, serial = device_dir_name.split("_", 1)
            return DeviceInfo(
                manufacturer=manufacturer,
                model="Unknown",
                serial_number=serial,
            )

        return DeviceInfo(
            manufacturer="Unknown",
            model="Unknown",
            serial_number="Unknown",
        )

    def _parse_machine_xml(self, xml_path: Path) -> DeviceInfo:
        """Parse machine.xml for device information."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        return DeviceInfo(
            manufacturer=root.findtext("brand", "Unknown"),
            model=root.findtext("model", "Unknown"),
            serial_number=root.findtext("serial", "Unknown"),
            firmware_version=root.findtext("version", None),
        )

    # ------------------------------------------------------------------
    # parse_sessions hooks (driver lives in DeviceParser.parse_sessions).
    # Each .000/.001 pair is one session; OSCAR already groups sessions.
    # ------------------------------------------------------------------

    # Every OSCAR session yields exactly one UnifiedSession (no post-parse
    # drop), so `limit` pre-slices the sorted units for a deterministic set.
    _unit_may_yield_nothing = False

    def _resolve_units(
        self, path: Path, sort_by: str | None, timezone_name: str | None
    ) -> tuple[Path, list[tuple[int, Path | None, Path | None]]]:
        """Resolve one OSCAR data root and its ordered session files.

        Single-root contract: one ``parse_sessions`` call parses one data root
        (callers expand a multi-root path into one source per root).  If a path
        resolves to several roots only the first is parsed and the rest are
        logged; an empty resolution yields no units.
        """
        path = Path(path)

        # Defense in depth: the name was validated at `snore profile
        # set-timezone` time, but an invalid stored name must fail loudly here
        # rather than silently mislabel every imported timestamp.
        if timezone_name is not None:
            ZoneInfo(timezone_name)

        if self._data_roots:
            matching_roots = [r for r in self._data_roots if r.path == path]
            if matching_roots:
                roots_to_parse = matching_roots
            else:
                detection = self.detect(path)
                if not detection.detected:
                    raise ParserError("No OSCAR data found at path", self)
                roots_to_parse = self._data_roots
        else:
            detection = self.detect(path)
            if not detection.detected:
                raise ParserError("No OSCAR data found at path", self)
            roots_to_parse = self._data_roots

        if not roots_to_parse:
            return path, []

        if len(roots_to_parse) > 1:
            skipped = ", ".join(str(r.path) for r in roots_to_parse[1:])
            logger.warning(
                "OSCAR path resolved to %d data roots; parsing only %s and "
                "skipping %s. Import one root at a time for full coverage.",
                len(roots_to_parse),
                roots_to_parse[0].path,
                skipped,
            )

        root = roots_to_parse[0].path
        session_files = self._find_session_files(root)

        if sort_by == "date-asc":
            session_files = sorted(session_files, key=lambda x: x[0])
        elif sort_by == "date-desc":
            session_files = sorted(session_files, key=lambda x: x[0], reverse=True)

        return root, session_files

    def _unit_date(
        self, unit: tuple[int, Path | None, Path | None], timezone_name: str | None
    ) -> date | None:
        """Night of the session-start epoch, on the same clock as parsing."""
        return night_date(_epoch_to_wall_clock(unit[0], timezone_name))

    def _unit_label(self, unit: tuple[int, Path | None, Path | None]) -> str:
        """Failure-log id: the session's epoch id."""
        return f"session {unit[0]}"

    def _build_context(
        self, root: Path, timezone_name: str | None
    ) -> _OscarParseContext:
        """Device info plus the declared timezone, shared across all sessions."""
        return _OscarParseContext(
            device_info=self.get_device_info(root),
            timezone_name=timezone_name,
        )

    def _worker_dispatch(
        self,
        unit: tuple[int, Path | None, Path | None],
        ctx: _OscarParseContext,
        root: Path,
        date_from: str | None,
        date_to: str | None,
    ) -> tuple[Callable[..., UnifiedSession | None], tuple[Any, ...]]:
        """Submit one .000/.001 pair to the picklable module-level worker."""
        session_id, summary_path, events_path = unit
        args = (
            session_id,
            summary_path,
            events_path,
            ctx.device_info,
            root,
            ctx.timezone_name,
        )
        return _oscar_parse_session_worker, args

    def _parse_unit_sequential(
        self,
        unit: tuple[int, Path | None, Path | None],
        ctx: _OscarParseContext,
        root: Path,
        date_from: str | None,
        date_to: str | None,
    ) -> UnifiedSession | None:
        """Parse one session in-process (it finalizes statistics itself)."""
        session_id, summary_path, events_path = unit
        return self._parse_single_session(
            session_id,
            summary_path,
            events_path,
            ctx.device_info,
            root,
            ctx.timezone_name,
        )

    def _find_session_files(
        self, device_path: Path
    ) -> list[tuple[int, Path | None, Path | None]]:
        """
        Find all session files in Summaries/ and Events/ directories.

        Returns list of (session_id, summary_path, events_path) tuples.
        """
        summaries_dir = device_path / "Summaries"
        events_dir = device_path / "Events"

        session_map: dict[int, dict[str, Path]] = {}

        if summaries_dir.exists():
            for summary_file in summaries_dir.glob("*.000"):
                try:
                    session_id = int(summary_file.stem, 16)
                    if session_id not in session_map:
                        session_map[session_id] = {}
                    session_map[session_id]["summary"] = summary_file
                except ValueError:
                    logger.warning(f"Invalid summary filename: {summary_file.name}")

        if events_dir.exists():
            for events_file in events_dir.glob("*.001"):
                try:
                    session_id = int(events_file.stem, 16)
                    if session_id not in session_map:
                        session_map[session_id] = {}
                    session_map[session_id]["events"] = events_file
                except ValueError:
                    logger.warning(f"Invalid events filename: {events_file.name}")

        result = []
        for session_id in sorted(session_map.keys()):
            files = session_map[session_id]
            result.append(
                (
                    session_id,
                    files.get("summary"),
                    files.get("events"),
                )
            )

        return result

    def _parse_single_session(
        self,
        session_id: int,
        summary_path: Path | None,
        events_path: Path | None,
        device_info: DeviceInfo,
        base_path: Path,
        timezone_name: str | None = None,
    ) -> UnifiedSession:
        """Parse a single session from .000 and .001 files."""
        summary: SessionSummary | None = None
        events: SessionEvents | None = None

        if summary_path and summary_path.exists():
            try:
                summary = parse_summary_file(summary_path)
            except Exception as e:
                logger.warning(
                    f"Summary parsing failed for {summary_path.name}: {e}. "
                    "Continuing with events file only."
                )

        if events_path and events_path.exists():
            events = parse_events_file(events_path)

        if summary:
            first_ts = summary.first_timestamp
            last_ts = summary.last_timestamp
        elif events:
            first_ts = events.first_timestamp
            last_ts = events.last_timestamp
        else:
            raise ValueError(f"No data for session {session_id}")

        # OSCAR stores epoch-ms instants.  With a declared profile timezone
        # they are converted to that zone's wall clock, matching ResMed's
        # device-local semantics in the naive tier-2 columns.
        #
        # KNOWN A6 INCONSISTENCY (only when timezone_name is None): without a
        # declared timezone these datetimes fall back to UTC wall-clock, while
        # ResMed session/event times are device-local wall-clock — so
        # OSCAR-imported rows carry UTC-derived values under the "unknown"
        # timezone label.  Declare a profile timezone
        # (`snore profile set-timezone`) to resolve it.
        start_time = _epoch_to_wall_clock(first_ts / 1000, timezone_name)
        end_time = _epoch_to_wall_clock(last_ts / 1000, timezone_name)

        # One OSCAR .000 session file is a single wear period, so the session
        # span is the mask-on time by construction.
        duration_seconds = (end_time - start_time).total_seconds()

        session = UnifiedSession(
            device_session_id=str(session_id),
            device_info=device_info,
            start_time=start_time,
            end_time=end_time,
            import_source=OSCAR_BINARY_PARSER_ID,
            parser_version=self.metadata.parser_version,
            raw_data_path=str(base_path),
            mask_on_segments=[(0.0, duration_seconds)],
        )

        if summary:
            self._populate_statistics_from_summary(summary, session)
            settings = self._convert_summary_to_therapy_settings(summary)
            if settings:
                session.settings = settings

        if events:
            self._populate_waveforms_from_events(events, session)
            self._populate_events_from_events(events, session, timezone_name)

        session.finalize_statistics()
        return session

    def _populate_statistics_from_summary(
        self, summary: SessionSummary, session: UnifiedSession
    ) -> None:
        """Populate session statistics from OSCAR summary data."""
        stats = session.statistics

        stats.obstructive_apneas = int(summary.counts.get(CPAP_OBSTRUCTIVE, 0))
        stats.hypopneas = int(summary.counts.get(CPAP_HYPOPNEA, 0))
        stats.central_apneas = int(summary.counts.get(CPAP_CLEAR_AIRWAY, 0))
        stats.reras = int(summary.counts.get(CPAP_RERA, 0))
        stats.flow_limitations = int(summary.counts.get(CPAP_FLOW_LIMIT, 0))

        if CPAP_PERIODIC_BREATHING in summary.counts:
            session.data_quality_notes.append(
                f"Periodic breathing events: {int(summary.counts[CPAP_PERIODIC_BREATHING])}"
            )

        stats.pressure_min = summary.minimums.get(
            CPAP_PRESSURE
        ) or summary.minimums.get(CPAP_MASK_PRESSURE)
        stats.pressure_max = summary.maximums.get(
            CPAP_PRESSURE
        ) or summary.maximums.get(CPAP_MASK_PRESSURE)
        stats.pressure_mean = summary.averages.get(
            CPAP_PRESSURE
        ) or summary.averages.get(CPAP_MASK_PRESSURE)

        stats.epap_min = summary.minimums.get(CPAP_EPAP)
        stats.epap_max = summary.maximums.get(CPAP_EPAP)
        stats.epap_mean = summary.averages.get(CPAP_EPAP)

        stats.leak_min = summary.minimums.get(CPAP_LEAK)
        stats.leak_max = summary.maximums.get(CPAP_LEAK)
        stats.leak_mean = summary.averages.get(CPAP_LEAK)

        stats.respiratory_rate_min = summary.minimums.get(CPAP_RESPRATE)
        stats.respiratory_rate_max = summary.maximums.get(CPAP_RESPRATE)
        stats.respiratory_rate_mean = summary.averages.get(CPAP_RESPRATE)

        stats.tidal_volume_min = summary.minimums.get(CPAP_TIDAL_VOLUME)
        stats.tidal_volume_max = summary.maximums.get(CPAP_TIDAL_VOLUME)
        stats.tidal_volume_mean = summary.averages.get(CPAP_TIDAL_VOLUME)

        stats.minute_ventilation_min = summary.minimums.get(CPAP_MINUTE_VENT)
        stats.minute_ventilation_max = summary.maximums.get(CPAP_MINUTE_VENT)
        stats.minute_ventilation_mean = summary.averages.get(CPAP_MINUTE_VENT)

        stats.spo2_min = summary.minimums.get(OXI_SPO2)
        stats.spo2_max = summary.maximums.get(OXI_SPO2)
        stats.spo2_mean = summary.averages.get(OXI_SPO2)
        if OXI_SPO2 in summary.time_below_threshold:
            stats.spo2_time_below_90 = int(
                summary.time_below_threshold[OXI_SPO2] / 1000
            )

        stats.pulse_min = summary.minimums.get(OXI_PULSE)
        stats.pulse_max = summary.maximums.get(OXI_PULSE)
        stats.pulse_mean = summary.averages.get(OXI_PULSE)

        duration_ms = summary.last_timestamp - summary.first_timestamp
        stats.usage_hours = therapy_hours(
            TherapyHoursBasis.SESSION_SPAN, span_seconds=duration_ms / 1000
        )

        session.has_statistics = True

    def _convert_summary_to_therapy_settings(
        self, summary: SessionSummary
    ) -> TherapySettings | None:
        """
        Convert OSCAR summary settings dict to TherapySettings model.

        Args:
            summary: Parsed OSCAR summary data

        Returns:
            TherapySettings instance or None if no settings available
        """
        settings = summary.settings
        if not settings:
            return None

        raw_mode = settings.get(CPAP_MODE)
        mode_number = _finite(raw_mode)
        mode = OSCAR_MODE_MAP.get(int(mode_number)) if mode_number is not None else None
        if mode is None:
            if raw_mode is not None:
                logger.warning(
                    "Unknown OSCAR therapy mode value %r, defaulting to CPAP", raw_mode
                )
            mode = TherapyMode.CPAP

        epr_level, epr_mode = _epr_settings(settings)
        ramp_code = _code(settings, RMS9_RAMP_ENABLE)
        pressures = _pressure_settings(settings, mode)
        return TherapySettings(
            mode=mode,
            pressure_min=_pressure(settings, CPAP_PRESSURE_MIN),
            pressure_max=_pressure(settings, CPAP_PRESSURE_MAX),
            pressure_fixed=_pressure(settings, CPAP_PRESSURE),
            ipap=pressures.ipap,
            epap=pressures.epap,
            ps=pressures.ps,
            epr_level=epr_level,
            epr_mode=epr_mode,
            ramp_enabled=ramp_code == 1 if ramp_code is not None else None,
            ramp_time=_code(settings, CPAP_RAMP_TIME),
            # Bilevel, ASV and iVAPS sessions store the ramp start as a start EPAP.
            ramp_start_pressure=_first(
                _pressure(settings, CPAP_RAMP_PRESSURE),
                _pressure(settings, RMVENT_START_EPAP),
            ),
            other_settings={
                **_resmed_other_settings(settings),
                **pressures.other_settings,
            },
        )

    def _populate_waveforms_from_events(
        self, events: SessionEvents, session: UnifiedSession
    ) -> None:
        """Convert OSCAR EventLists to WaveformData."""
        session_start_ms = events.first_timestamp

        for channel_id, event_lists in events.event_lists.items():
            if channel_id not in OSCAR_WAVEFORM_CHANNEL_IDS:
                continue

            waveform_type = OSCAR_WAVEFORM_TYPE_MAP[channel_id]
            unit = OSCAR_WAVEFORM_UNITS.get(channel_id, "")

            for event_list in event_lists:
                if event_list.event_type != EventListType.WAVEFORM:
                    continue

                values = np.array(event_list.get_actual_values(), dtype=np.float32)

                if len(values) == 0:
                    continue

                timestamps_ms = event_list.get_timestamps()
                timestamps_seconds = np.array(
                    [(ts - session_start_ms) / 1000.0 for ts in timestamps_ms],
                    dtype=np.float32,
                )

                min_value, max_value, mean_value = extract_basic_stats(values)
                waveform = WaveformData(
                    waveform_type=waveform_type,
                    sample_rate=event_list.sample_rate,
                    unit=event_list.dimension or unit,
                    timestamps=timestamps_seconds,
                    values=values,
                    min_value=min_value,
                    max_value=max_value,
                    mean_value=mean_value,
                )

                session.add_waveform(waveform)
                logger.debug(
                    f"Added {waveform_type.value} waveform: {len(values)} samples "
                    f"at {event_list.sample_rate} Hz"
                )

    def _populate_events_from_events(
        self,
        events: SessionEvents,
        session: UnifiedSession,
        timezone_name: str | None = None,
    ) -> None:
        """Convert OSCAR EventLists to RespiratoryEvent objects."""
        for channel_id, event_lists in events.event_lists.items():
            if channel_id not in OSCAR_EVENT_CHANNEL_IDS:
                continue

            event_type = OSCAR_EVENT_TYPE_MAP[channel_id]

            for event_list in event_lists:
                if event_list.event_type != EventListType.EVENT:
                    continue

                timestamps = event_list.get_timestamps()
                durations = event_list.get_actual_values()

                for i, timestamp_ms in enumerate(timestamps):
                    event_time = _epoch_to_wall_clock(
                        timestamp_ms / 1000, timezone_name
                    )

                    duration = durations[i] if i < len(durations) else None
                    # OSCAR stores event timestamps at the event's end for all
                    # event channels (its UI draws spans backward by duration).
                    session.add_event(
                        RespiratoryEvent.from_end_flag(event_type, event_time, duration)
                    )
