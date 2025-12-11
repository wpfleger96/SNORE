"""
ResMed EDF+ Parser

Parser for ResMed CPAP devices that output EDF+ format files.
Supports AirSense 10/11, AirCurve 10/11, and S9 series.

File Types:
- STR.edf: Device settings and configuration
- BRP.edf: Breathing waveforms (Flow Rate)
- PLD.edf: Pressure & Leak Data
- EVE.edf: Events (Apneas, Hypopneas, etc.)
- SA2.edf: Statistics and summary data
- CSL.edf: Compliance/Summary Log
"""

import json
import logging
import re

from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from snore.constants import PARSER_MAX_SEARCH_DEPTH
from snore.models.unified import (
    DeviceInfo,
    RespiratoryEvent,
    RespiratoryEventType,
    UnifiedSession,
    WaveformData,
    WaveformType,
)
from snore.parsers.base import (
    DeviceParser,
    ParserDetectionResult,
    ParserError,
    ParserMetadata,
)
from snore.parsers.discovery import DataRoot, DataRootFinder
from snore.parsers.formats.edf import EDFReader

logger = logging.getLogger(__name__)


class ResmedEDFParser(DeviceParser):
    """
    Parser for ResMed EDF+ data format.

    This parser handles the standard ResMed SD card structure:
    - Backup/
      - STR.edf (settings/configuration)
      - Identification.json (device info)
      - DATALOG/YYYY/
        - YYYYMMDD_HHMMSS_BRP.edf (breathing waveforms)
        - YYYYMMDD_HHMMSS_PLD.edf (pressure/leak)
        - YYYYMMDD_HHMMSS_SA2.edf (statistics)
        - YYYYMMDD_HHMMSS_EVE.edf (events)
        - YYYYMMDD_HHMMSS_CSL.edf (compliance)
    """

    FILE_TYPE_BRP = "_BRP.edf"  # Breathing waveforms
    FILE_TYPE_PLD = "_PLD.edf"  # Pressure/Leak data
    FILE_TYPE_SA2 = "_SA2.edf"  # Statistics
    FILE_TYPE_EVE = "_EVE.edf"  # Events
    FILE_TYPE_CSL = "_CSL.edf"  # Compliance

    # Event type mapping from EDF annotations to unified types
    # Based on OSCAR's ResMed annotation mappings
    EVENT_TYPE_MAP = {
        "Obstructive Apnea": RespiratoryEventType.OBSTRUCTIVE_APNEA,
        "ObstructiveApnea": RespiratoryEventType.OBSTRUCTIVE_APNEA,
        "Obstructive apnea": RespiratoryEventType.OBSTRUCTIVE_APNEA,
        "OA": RespiratoryEventType.OBSTRUCTIVE_APNEA,
        "Central Apnea": RespiratoryEventType.CENTRAL_APNEA,
        "CentralApnea": RespiratoryEventType.CENTRAL_APNEA,
        "Central apnea": RespiratoryEventType.CENTRAL_APNEA,
        "CA": RespiratoryEventType.CENTRAL_APNEA,
        "Clear Airway": RespiratoryEventType.CLEAR_AIRWAY,  # (same as Central Apnea in some ResMed devices)
        "ClearAirway": RespiratoryEventType.CLEAR_AIRWAY,
        "Apnea": RespiratoryEventType.UNCLASSIFIED_APNEA,
        "UA": RespiratoryEventType.UNCLASSIFIED_APNEA,
        "Hypopnea": RespiratoryEventType.HYPOPNEA,
        "H": RespiratoryEventType.HYPOPNEA,
        "RERA": RespiratoryEventType.RERA,  # (Respiratory Effort Related Arousal)
        "RE": RespiratoryEventType.RERA,
        "Arousal": RespiratoryEventType.RERA,  # OSCAR uses "Arousal" for RERA
        "Flow Limitation": RespiratoryEventType.FLOW_LIMITATION,
        "FlowLimitation": RespiratoryEventType.FLOW_LIMITATION,
        "FL": RespiratoryEventType.FLOW_LIMITATION,
        "Periodic Breathing": RespiratoryEventType.PERIODIC_BREATHING,
        "PeriodicBreathing": RespiratoryEventType.PERIODIC_BREATHING,
        "PB": RespiratoryEventType.PERIODIC_BREATHING,
        "Large Leak": RespiratoryEventType.LARGE_LEAK,
        "LargeLeak": RespiratoryEventType.LARGE_LEAK,
        "LL": RespiratoryEventType.LARGE_LEAK,
        "Vibratory Snore": RespiratoryEventType.VIBRATORY_SNORE,
        "VibratorySnore": RespiratoryEventType.VIBRATORY_SNORE,
        "VS": RespiratoryEventType.VIBRATORY_SNORE,
    }

    # Special annotations that should be filtered out (not actual events)
    FILTERED_ANNOTATIONS = {
        "Recording starts",
        "SpO2 Desaturation",  # handled separately if needed
    }

    def __init__(self) -> None:
        """Initialize ResMed parser."""
        super().__init__()
        self._data_root: Path | None = None
        self._root_metadata: DataRoot | None = None
        self._finder = DataRootFinder()

    def get_metadata(self) -> ParserMetadata:
        """Return ResMed parser metadata."""
        return ParserMetadata(
            parser_id="resmed_edf",
            parser_version="1.0.0",
            manufacturer="ResMed",
            supported_formats=["EDF+", "EDF"],
            supported_models=[
                "AirSense 10 AutoSet",
                "AirSense 10 Elite",
                "AirSense 10 CPAP",
                "AirSense 11 AutoSet",
                "AirCurve 10 S",
                "AirCurve 10 VAuto",
                "AirCurve 10 ASV",
                "AirCurve 11 VAuto",
                "S9 AutoSet",
                "S9 Elite",
                "S9 VPAP Auto",
            ],
            description="Parser for ResMed CPAP devices using EDF+ format",
            requires_libraries=["pyedflib", "numpy"],
        )

    def detect(self, path: Path) -> ParserDetectionResult:
        """
        Detect ResMed EDF+ data structure with smart path discovery.

        Searches for STR.edf + DATALOG signature in:
        - Current path
        - Parent directories (up to 5 levels)
        - Child directories (up to 3 levels)

        Supports both raw SD card format and OSCAR profile format.
        """
        path = Path(path)

        if not path.exists():
            return ParserDetectionResult(
                detected=False, message=f"Path does not exist: {path}"
            )

        roots = self._finder.find_data_roots(
            path,
            validator_func=self._is_resmed_root,
            metadata_extractor_func=self._create_data_root,
            max_levels_down=PARSER_MAX_SEARCH_DEPTH,
        )

        if not roots:
            return ParserDetectionResult(
                detected=False,
                message=f"No ResMed data found. Searched {path} and parent/child directories for STR.edf + DATALOG structure.",
            )

        self._data_root = roots[0].path
        self._root_metadata = roots[0]

        metadata_dict = {
            "data_root": str(self._data_root),
            "structure_type": self._root_metadata.structure_type,
            "profile_name": self._root_metadata.profile_name,
            "device_serial": self._root_metadata.device_serial,
        }

        if self._data_root != path:
            location_desc = f"in {'parent' if self._data_root in path.parents else 'child'} directory"
        else:
            location_desc = "at provided path"

        structure_name = self._root_metadata.structure_type.replace("_", " ")

        if len(roots) > 1:
            message = f"Found {len(roots)} ResMed data locations ({location_desc})"
        else:
            message = f"ResMed {structure_name} data detected ({location_desc})"

        return ParserDetectionResult(
            detected=True,
            confidence=self._root_metadata.confidence,
            message=message,
            metadata=metadata_dict,
        )

    def _is_resmed_root(self, path: Path) -> bool:
        """Check if path contains ResMed data signature (STR.edf + DATALOG)."""
        if not path.is_dir():
            return False
        return (path / "STR.edf").exists() and (path / "DATALOG").is_dir()

    def _create_data_root(self, path: Path) -> DataRoot:
        """Create DataRoot with metadata extracted from path structure."""
        parts = path.parts

        if "Profiles" in parts and "Backup" in parts:
            try:
                profiles_idx = parts.index("Profiles")
                profile_name = (
                    parts[profiles_idx + 1] if profiles_idx + 1 < len(parts) else None
                )
                device_str = (
                    parts[profiles_idx + 2] if profiles_idx + 2 < len(parts) else None
                )

                serial = None
                if device_str and "_" in device_str:
                    serial = device_str.split("_", 1)[1]

                return DataRoot(
                    path=path,
                    structure_type="oscar_profile",
                    profile_name=profile_name,
                    device_serial=serial,
                    confidence=0.95,
                )
            except (IndexError, ValueError):
                pass

        serial = self._extract_serial_from_identification(path)
        return DataRoot(
            path=path,
            structure_type="raw_sd",
            profile_name=None,
            device_serial=serial,
            confidence=0.90,
        )

    def _extract_serial_from_identification(self, path: Path) -> str | None:
        """Extract device serial number from Identification.json."""
        id_file = path / "Identification.json"
        if not id_file.exists():
            return None

        try:
            with open(id_file) as f:
                data = json.load(f)

            fg = data.get("FlowGenerator", {})
            profiles = fg.get("IdentificationProfiles", {})
            product = profiles.get("Product", {})
            serial = product.get("SerialNumber")
            return serial if isinstance(serial, str) else None
        except Exception:
            return None

    def get_device_info(self, path: Path) -> DeviceInfo:
        """
        Extract ResMed device information.

        Tries Identification.json first, falls back to STR.edf.
        """
        path = Path(self._data_root if self._data_root else path)

        id_file = path / "Identification.json"
        if id_file.exists():
            try:
                with open(id_file) as f:
                    data = json.load(f)

                fg = data.get("FlowGenerator", {})
                profiles = fg.get("IdentificationProfiles", {})
                product = profiles.get("Product", {})
                software = profiles.get("Software", {})

                return DeviceInfo(
                    manufacturer="ResMed",
                    model=product.get("ProductName", "Unknown"),
                    serial_number=product.get("SerialNumber", "Unknown"),
                    firmware_version=software.get("ApplicationIdentifier", None),
                    product_code=product.get("ProductCode", None),
                )

            except Exception as e:
                logger.warning(f"Failed to parse Identification.json: {e}")

        str_file = path / "STR.edf"
        if str_file.exists():
            try:
                with EDFReader(str_file) as edf:
                    header = edf.get_header()

                    recording_info = header.recording_info

                    model = "Unknown"
                    serial = "Unknown"

                    if "AirSense" in recording_info:
                        match = re.search(r"(AirSense \d+ [A-Za-z]+)", recording_info)
                        if match:
                            model = match.group(1)
                    elif "AirCurve" in recording_info:
                        match = re.search(r"(AirCurve \d+ [A-Za-z]+)", recording_info)
                        if match:
                            model = match.group(1)

                    serial_match = re.search(r"SN[:\s]+(\d+)", recording_info)
                    if serial_match:
                        serial = serial_match.group(1)

                    return DeviceInfo(
                        manufacturer="ResMed", model=model, serial_number=serial
                    )

            except Exception as e:
                logger.warning(f"Failed to parse STR.edf: {e}")

        return DeviceInfo(
            manufacturer="ResMed", model="Unknown", serial_number="Unknown"
        )

    def parse_sessions(
        self,
        path: Path,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int | None = None,
        sort_by: str | None = None,
    ) -> Iterator[UnifiedSession]:
        """
        Parse all ResMed sessions from the given path.

        Yields one UnifiedSession per therapy session.
        """
        path = Path(self._data_root if self._data_root else path)
        datalog_dir = path / "DATALOG"

        if not datalog_dir.exists():
            raise ParserError("DATALOG directory not found", self)

        device_info = self.get_device_info(path)

        night_groups = self._group_session_files(datalog_dir)

        total_segments = sum(len(segments) for segments in night_groups.values())
        logger.info(
            f"Found {len(night_groups)} nights with {total_segments} total segments "
            f"(avg {total_segments / len(night_groups):.1f} segments per night)"
        )

        if sort_by == "date-asc":
            night_items = sorted(night_groups.items(), key=lambda x: x[0])
        elif sort_by == "date-desc":
            night_items = sorted(night_groups.items(), key=lambda x: x[0], reverse=True)
        else:
            night_items = list(night_groups.items())

        sessions_yielded = 0

        for night_date, segments in night_items:
            if date_from or date_to:
                try:
                    night_date_obj = datetime.strptime(night_date, "%Y%m%d").date()

                    if date_from:
                        filter_date_from = datetime.fromisoformat(date_from).date()
                        if night_date_obj < filter_date_from:
                            logger.debug(
                                f"Skipping night {night_date}: before {filter_date_from}"
                            )
                            continue

                    if date_to:
                        filter_date_to = datetime.fromisoformat(date_to).date()
                        if night_date_obj > filter_date_to:
                            logger.debug(
                                f"Skipping night {night_date}: after {filter_date_to}"
                            )
                            continue
                except (ValueError, IndexError) as e:
                    logger.warning(f"Could not parse night date {night_date}: {e}")

            if limit is not None and sessions_yielded >= limit:
                logger.info(f"Reached session limit of {limit}, stopping")
                break

            try:
                session = self._parse_night_session(
                    night_date, segments, device_info, path
                )

                if session is None:
                    continue

                if date_from:
                    if (
                        session.start_time.date()
                        < datetime.fromisoformat(date_from).date()
                    ):
                        logger.warning(
                            f"Night {night_date} has mismatched date in ID vs file contents"
                        )
                        continue
                if date_to:
                    if (
                        session.start_time.date()
                        > datetime.fromisoformat(date_to).date()
                    ):
                        logger.warning(
                            f"Night {night_date} has mismatched date in ID vs file contents"
                        )
                        continue

                yield session
                sessions_yielded += 1

            except Exception as e:
                logger.error(f"Failed to parse night {night_date}: {e}")
                continue

    def _get_night_date(self, timestamp: datetime) -> str:
        """
        Get the "night date" for a session using OSCAR's noon cutoff rule.

        Sessions starting before noon belong to the previous day's night.
        This matches ResMed's commercial software and OSCAR's behavior.

        Args:
            timestamp: Session start time

        Returns:
            Night date as YYYYMMDD string
        """
        # ResMed splits days at noon - sessions before noon go to previous day
        # Reference: OSCAR resmed_loader.cpp lines 1112-1116
        if timestamp.hour < 12:
            night_date = (timestamp - timedelta(days=1)).date()
        else:
            night_date = timestamp.date()

        return night_date.strftime("%Y%m%d")

    def _group_session_files(
        self, datalog_dir: Path
    ) -> dict[str, dict[str, dict[str, Path]]]:
        """
        Group EDF files by night date (noon-to-noon periods).

        Multiple sessions within the same night (mask removals/bathroom breaks)
        are grouped together to match OSCAR's behavior.

        Returns:
            Dict mapping night_date to dict of session_ids to file types
            Example: {
                "20240621": {  # Night of June 21
                    "20240621_013454": {
                        "BRP": Path("20240621_013454_BRP.edf"),
                        "PLD": Path("20240621_013454_PLD.edf"),
                        ...
                    },
                    "20240621_053022": {  # Another segment same night
                        "BRP": Path("20240621_053022_BRP.edf"),
                        ...
                    }
                }
            }
        """
        groups: dict[str, dict[str, dict[str, Path]]] = {}

        for edf_file in datalog_dir.rglob("*.edf"):
            filename = edf_file.name

            match = re.match(r"(\d{8}_\d{6})_([A-Z0-9]+)\.edf", filename)
            if not match:
                continue

            session_id = match.group(1)
            file_type = match.group(2)

            timestamp = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
            night_date = self._get_night_date(timestamp)

            if night_date not in groups:
                groups[night_date] = {}
            if session_id not in groups[night_date]:
                groups[night_date][session_id] = {}

            groups[night_date][session_id][file_type] = edf_file

        return groups

    def _parse_night_session(
        self,
        night_date: str,
        segments: dict[str, dict[str, Path]],
        device_info: DeviceInfo,
        base_path: Path,
    ) -> UnifiedSession | None:
        """
        Parse all segments for a single night into one unified session.

        Multiple segments occur when the mask is removed/replaced during the night
        (bathroom breaks, water, etc.). This matches OSCAR's behavior of combining
        segments from the same night (noon-to-noon period).

        Args:
            night_date: Night date (YYYYMMDD format)
            segments: Dict mapping session_id to file dict
            device_info: Device information
            base_path: Base data path

        Returns:
            Single UnifiedSession representing the entire night
        """
        sorted_segments = sorted(segments.items(), key=lambda x: x[0])

        logger.info(
            f"Parsing night {night_date} with {len(sorted_segments)} segment(s): "
            f"{[seg_id for seg_id, _ in sorted_segments]}"
        )

        # Collect ALL EVE files from all segments (including zero-record ones)
        # Per OSCAR's behavior: EVE files store data for whole day and should be applied to all sessions
        eve_files = []
        for segment_id, files in sorted_segments:
            if "EVE" in files:
                eve_files.append(files["EVE"])
                logger.debug(
                    f"Found EVE file for segment {segment_id}: {files['EVE'].name}"
                )

        segment_sessions = []
        for segment_id, files in sorted_segments:
            try:
                segment_session = self._parse_session_group(
                    segment_id, files, device_info, base_path
                )
                segment_sessions.append(segment_session)
            except ValueError as e:
                if "No valid data records" in str(e):
                    logger.info(f"Skipping zero-record segment {segment_id}")
                    continue
                raise
            except Exception as e:
                logger.warning(f"Failed to parse segment {segment_id}: {e}")
                continue

        if not segment_sessions:
            # OSCAR allows "summary only" nights with CSL/EVE but no waveform data
            # These occur from device self-tests, brief power-ons, or compliance checks
            logger.warning(
                f"Night {night_date} has no valid therapy segments (only CSL/EVE stub files). "
                f"This is likely a device self-test or brief power-on event. Skipping night."
            )
            return None

        if len(segment_sessions) == 1:
            session = segment_sessions[0]
            if eve_files:
                logger.info(
                    f"Parsing {len(eve_files)} EVE file(s) for night {night_date}"
                )
                self._parse_eve_files_for_night(eve_files, session)
            return session

        logger.info(f"Merging {len(segment_sessions)} segments for night {night_date}")

        merged_session = segment_sessions[0]

        merged_session.device_session_id = f"{night_date}_merged"

        merged_session.end_time = segment_sessions[-1].end_time

        cumulative_time_offset = 0.0
        for i, segment in enumerate(segment_sessions):
            if i == 0:
                cumulative_time_offset = (
                    segment.end_time - segment.start_time
                ).total_seconds()
            else:
                segment_start_offset = (
                    segment.start_time - merged_session.start_time
                ).total_seconds()

                gap_duration = segment_start_offset - cumulative_time_offset
                if gap_duration > 0:
                    merged_session.data_quality_notes.append(
                        f"Gap {i}: {gap_duration / 60:.1f} minutes "
                        f"({segment_sessions[i - 1].end_time.strftime('%H:%M:%S')} - "
                        f"{segment.start_time.strftime('%H:%M:%S')})"
                    )

                for waveform_type, segment_waveform in segment.waveforms.items():
                    if waveform_type in merged_session.waveforms:
                        merged_waveform = merged_session.waveforms[waveform_type]

                        adjusted_timestamps = (
                            np.asarray(segment_waveform.timestamps)
                            + segment_start_offset
                        )

                        merged_waveform.timestamps = np.concatenate(
                            [merged_waveform.timestamps, adjusted_timestamps]
                        )
                        merged_waveform.values = np.concatenate(
                            [merged_waveform.values, segment_waveform.values]
                        )

                        merged_waveform.min_value = float(
                            np.min(merged_waveform.values)
                        )
                        merged_waveform.max_value = float(
                            np.max(merged_waveform.values)
                        )
                        merged_waveform.mean_value = float(
                            np.mean(merged_waveform.values)
                        )
                    else:
                        merged_session.add_waveform(segment_waveform)

                for event in segment.events:
                    event.start_time = event.start_time + timedelta(
                        seconds=(
                            segment_start_offset
                            - (event.start_time - segment.start_time).total_seconds()
                        )
                    )
                    merged_session.add_event(event)

                cumulative_time_offset = (
                    segment.end_time - merged_session.start_time
                ).total_seconds()

        merged_session.data_quality_notes.insert(
            0,
            f"Night composed of {len(segment_sessions)} segment(s) - mask removed during sleep",
        )

        logger.info(
            f"Merged night {night_date}: {len(segment_sessions)} segments, "
            f"total duration {(merged_session.end_time - merged_session.start_time).total_seconds() / 3600:.2f}h"
        )

        if eve_files:
            logger.info(f"Parsing {len(eve_files)} EVE file(s) for night {night_date}")
            self._parse_eve_files_for_night(eve_files, merged_session)

        return merged_session

    def _parse_session_group(
        self,
        session_id: str,
        files: dict[str, Path],
        device_info: DeviceInfo,
        base_path: Path,
    ) -> UnifiedSession:
        """Parse a single session from its file group."""
        from .formats.edf import get_edf_record_count

        start_time = datetime.strptime(session_id, "%Y%m%d_%H%M%S")

        session_duration_seconds = None
        for file_type in ["BRP", "PLD", "SA2"]:
            if file_type in files:
                try:
                    record_count = get_edf_record_count(files[file_type])
                    if record_count > 0:
                        with EDFReader(files[file_type]) as edf:
                            header = edf.get_header()
                            session_duration_seconds = (
                                record_count * header.record_duration
                            )
                            logger.debug(
                                f"Calculated session duration from {file_type}: "
                                f"{record_count} records × {header.record_duration}s = {session_duration_seconds}s "
                                f"({session_duration_seconds / 3600:.2f} hours)"
                            )
                            break
                except Exception as e:
                    logger.warning(f"Could not read duration from {file_type}: {e}")
                    continue

        if session_duration_seconds is None or session_duration_seconds == 0:
            logger.info(
                f"Session {session_id}: All files have 0 data records "
                f"(device turned on briefly but not used) - skipping"
            )
            raise ValueError("No valid data records in any files for this session")

        session = UnifiedSession(
            device_session_id=session_id,
            device_info=device_info,
            start_time=start_time,
            end_time=start_time + timedelta(seconds=session_duration_seconds),
            import_source="resmed_edf",
            parser_version=self.metadata.parser_version,
            raw_data_path=str(base_path),
        )

        if "SA2" in files:
            self._parse_statistics(files["SA2"], session)

        if "BRP" in files:
            self._parse_breathing_waveforms(files["BRP"], session)

        if "PLD" in files:
            self._parse_pressure_leak(files["PLD"], session)

        return session

    def _parse_statistics(self, file_path: Path, session: UnifiedSession) -> None:
        """
        Parse SA2 oximetry data file.

        SA2 files contain:
        - Pulse (heart rate in bpm) at 1Hz
        - SpO2 (oxygen saturation %) at 1Hz

        Note: Values of -1 indicate no oximeter was connected.
        """
        from .formats.edf import get_edf_record_count

        try:
            record_count = get_edf_record_count(file_path)
            file_size = file_path.stat().st_size

            if record_count == 0:
                logger.info(
                    f"SA2 file {file_path.name} has 0 data records (device on but not used, size={file_size} bytes)"
                )
                session.data_quality_notes.append(
                    "SA2: No data (device turned on briefly, not used)"
                )
                session.has_statistics = True
                return

            logger.debug(
                f"SA2 file {file_path.name} has {record_count} records (size={file_size} bytes)"
            )

            with EDFReader(file_path) as edf:
                has_valid_data = False

                spo2_signal = self._find_signal(edf, ["SpO2"])
                if spo2_signal:
                    data, info = edf.read_signal(spo2_signal)

                    valid_mask = (data >= 70) & (data <= 100)
                    valid_data = data[valid_mask]

                    if len(valid_data) > 0:
                        timestamps_seconds = edf.get_timestamps(spo2_signal, data)

                        spo2_min = float(np.min(valid_data))
                        spo2_max = float(np.max(valid_data))
                        spo2_mean = float(np.mean(valid_data))

                        below_90 = np.sum(valid_data < 90)
                        time_below_90_seconds = int(below_90)

                        waveform = WaveformData(
                            waveform_type=WaveformType.SPO2,
                            sample_rate=edf.get_sample_rate(spo2_signal),
                            unit=info.physical_dimension or "%",
                            timestamps=timestamps_seconds,
                            values=data,
                            min_value=spo2_min,
                            max_value=spo2_max,
                            mean_value=spo2_mean,
                        )

                        session.add_waveform(waveform)

                        session.statistics.spo2_min = spo2_min
                        session.statistics.spo2_max = spo2_max
                        session.statistics.spo2_mean = spo2_mean
                        session.statistics.spo2_time_below_90 = time_below_90_seconds

                        has_valid_data = True
                        logger.debug(
                            f"Parsed SpO2 data: {len(valid_data)} valid samples, {time_below_90_seconds}s below 90%"
                        )
                    else:
                        logger.debug("SpO2 signal present but no valid data (all -1)")

                pulse_signal = self._find_signal(edf, ["Pulse"])
                if pulse_signal:
                    data, info = edf.read_signal(pulse_signal)

                    valid_mask = (data >= 40) & (data <= 200)
                    valid_data = data[valid_mask]

                    if len(valid_data) > 0:
                        timestamps_seconds = edf.get_timestamps(pulse_signal, data)

                        pulse_min = float(np.min(valid_data))
                        pulse_max = float(np.max(valid_data))
                        pulse_mean = float(np.mean(valid_data))

                        waveform = WaveformData(
                            waveform_type=WaveformType.PULSE,
                            sample_rate=edf.get_sample_rate(pulse_signal),
                            unit=info.physical_dimension or "bpm",
                            timestamps=timestamps_seconds,
                            values=data,
                            min_value=pulse_min,
                            max_value=pulse_max,
                            mean_value=pulse_mean,
                        )

                        session.add_waveform(waveform)

                        session.statistics.pulse_min = pulse_min
                        session.statistics.pulse_max = pulse_max
                        session.statistics.pulse_mean = pulse_mean

                        has_valid_data = True
                        logger.debug(
                            f"Parsed Pulse data: {len(valid_data)} valid samples"
                        )
                    else:
                        logger.debug("Pulse signal present but no valid data (all -1)")

                session.has_statistics = True

                if not has_valid_data:
                    logger.info("No oximeter connected - SA2 file has no valid data")

        except Exception as e:
            logger.warning(f"Failed to parse SA2 statistics: {e}")
            session.data_quality_notes.append(f"SA2 parsing failed: {e}")

    def _parse_breathing_waveforms(
        self, file_path: Path, session: UnifiedSession
    ) -> None:
        """Parse BRP breathing waveform file."""
        from .formats.edf import get_edf_record_count

        try:
            record_count = get_edf_record_count(file_path)
            file_size = file_path.stat().st_size

            if record_count == 0:
                logger.info(
                    f"BRP file {file_path.name} has 0 data records (device on but not used, size={file_size} bytes)"
                )
                session.data_quality_notes.append(
                    "BRP: No data (device turned on briefly, not used)"
                )
                return

            logger.debug(
                f"BRP file {file_path.name} has {record_count} records (size={file_size} bytes)"
            )

            with EDFReader(file_path) as edf:
                # BRP typically contains Flow Rate signal
                # ResMed uses names like "Flow", "Flow.40ms", "FlowRate"
                flow_signal = self._find_signal(edf, ["Flow"])

                if flow_signal:
                    data, info = edf.read_signal(flow_signal)
                    timestamps_seconds = edf.get_timestamps(flow_signal, data)

                    if len(data) == 0:
                        logger.warning(f"No data in flow signal {flow_signal}")
                        return

                    unit = info.physical_dimension or "L/min"

                    if unit == "L/s":
                        data = data * 60.0
                        unit = "L/min"

                    waveform = WaveformData(
                        waveform_type=WaveformType.FLOW_RATE,
                        sample_rate=edf.get_sample_rate(flow_signal),
                        unit=unit,
                        timestamps=timestamps_seconds,
                        values=data,
                        min_value=float(np.min(data)),
                        max_value=float(np.max(data)),
                        mean_value=float(np.mean(data)),
                    )

                    session.add_waveform(waveform)
                    logger.debug(
                        f"Parsed {len(data)} flow samples from {file_path.name}"
                    )

        except Exception as e:
            logger.warning(f"Failed to parse breathing waveforms: {e}")
            session.data_quality_notes.append(f"BRP parsing failed: {e}")

    def _find_signal(self, edf: Any, patterns: list[str]) -> str | None:
        """Find signal name matching any of the patterns."""
        signals = edf.list_signal_labels()
        for pattern in patterns:
            signal: str
            for signal in signals:
                if pattern.lower() in signal.lower():
                    return signal
        return None

    def _parse_pressure_leak(self, file_path: Path, session: UnifiedSession) -> None:
        """Parse PLD pressure/leak file."""
        from .formats.edf import get_edf_record_count

        try:
            record_count = get_edf_record_count(file_path)
            file_size = file_path.stat().st_size

            if record_count == 0:
                logger.info(
                    f"PLD file {file_path.name} has 0 data records (device on but not used, size={file_size} bytes)"
                )
                session.data_quality_notes.append(
                    "PLD: No data (device turned on briefly, not used)"
                )
                return

            logger.debug(
                f"PLD file {file_path.name} has {record_count} records (size={file_size} bytes)"
            )

            with EDFReader(file_path) as edf:
                # ResMed uses names like "Press.2s", "MaskPress.2s", "Pressure", "MaskPressure"
                pressure_signal = self._find_signal(edf, ["Press", "MaskPress"])

                if pressure_signal:
                    data, info = edf.read_signal(pressure_signal)
                    timestamps_seconds = edf.get_timestamps(pressure_signal, data)

                    if len(data) == 0:
                        logger.warning(f"No data in pressure signal {pressure_signal}")
                    else:
                        waveform = WaveformData(
                            waveform_type=WaveformType.MASK_PRESSURE,
                            sample_rate=edf.get_sample_rate(pressure_signal),
                            unit=info.physical_dimension or "cmH2O",
                            timestamps=timestamps_seconds,
                            values=data,
                            min_value=float(np.min(data)),
                            max_value=float(np.max(data)),
                            mean_value=float(np.mean(data)),
                        )

                        session.add_waveform(waveform)

                # ResMed uses names like "Leak.2s", "LeakRate"
                leak_signal = self._find_signal(edf, ["Leak"])

                if leak_signal:
                    data, info = edf.read_signal(leak_signal)
                    timestamps_seconds = edf.get_timestamps(leak_signal, data)

                    if len(data) == 0:
                        logger.warning(f"No data in leak signal {leak_signal}")
                    else:
                        unit = info.physical_dimension or "L/min"

                        if unit == "L/s":
                            data = data * 60.0
                            unit = "L/min"

                        waveform = WaveformData(
                            waveform_type=WaveformType.LEAK_RATE,
                            sample_rate=edf.get_sample_rate(leak_signal),
                            unit=unit,
                            timestamps=timestamps_seconds,
                            values=data,
                            min_value=float(np.min(data)),
                            max_value=float(np.max(data)),
                            mean_value=float(np.mean(data)),
                        )

                        session.add_waveform(waveform)

                logger.debug(f"Parsed pressure/leak from {file_path.name}")

        except Exception as e:
            logger.warning(f"Failed to parse pressure/leak: {e}")
            session.data_quality_notes.append(f"PLD parsing failed: {e}")

    def _parse_events(self, file_path: Path, session: UnifiedSession) -> None:
        """Parse EVE events file."""
        from .formats.edf import EDFDiscontinuousReader, is_discontinuous_edf

        is_discontinuous = is_discontinuous_edf(file_path)

        if is_discontinuous:
            logger.info(
                f"EVE file {file_path.name} is discontinuous (EDF+D format) - "
                f"using MNE library to read annotations"
            )
            session.data_quality_notes.append(
                "EVE file is discontinuous (mask removal detected during session)"
            )

        try:
            if is_discontinuous:
                with EDFDiscontinuousReader(file_path) as edf:
                    annotations = edf.read_annotations()
            else:
                with EDFReader(file_path) as edf:
                    annotations = edf.read_annotations()

            event_count = 0
            filtered_count = 0
            unknown_count = 0
            unknown_annotations = set()

            for annotation in annotations:
                event_type = None
                annotation_text = None

                for text in annotation.annotations:
                    if text in self.FILTERED_ANNOTATIONS:
                        filtered_count += 1
                        break

                    if text in self.EVENT_TYPE_MAP:
                        event_type = self.EVENT_TYPE_MAP[text]
                        annotation_text = text
                        break

                if annotation_text is None and event_type is None:
                    for text in annotation.annotations:
                        if text not in self.FILTERED_ANNOTATIONS:
                            unknown_annotations.add(text)
                            unknown_count += 1
                    continue

                if event_type is None:
                    continue

                duration = annotation.duration if annotation.duration else 10.0

                event = RespiratoryEvent(
                    event_type=event_type,
                    start_time=annotation.to_datetime(session.start_time),
                    duration_seconds=duration,
                )

                session.add_event(event)
                event_count += 1

            if is_discontinuous and event_count > 0:
                logger.info(
                    f"Successfully parsed {event_count} events from discontinuous EVE file "
                    f"(mask removal periods detected)"
                )
            else:
                logger.info(f"Parsed {event_count} events from {file_path.name}")

            if filtered_count > 0:
                logger.debug(f"Filtered out {filtered_count} non-event annotations")
            if unknown_count > 0:
                logger.warning(
                    f"Encountered {unknown_count} unknown annotations: {unknown_annotations}"
                )
                session.data_quality_notes.append(
                    f"Unknown event annotations: {', '.join(sorted(unknown_annotations))}"
                )

        except Exception as e:
            if "discontinuous" in str(e).lower():
                logger.warning(
                    "EVE file is discontinuous (mask removal during session) - events not imported"
                )
                session.data_quality_notes.append(
                    "EVE file is discontinuous (mask removal detected) - events cannot be imported"
                )
            else:
                logger.warning(f"Failed to parse events: {e}")
                session.data_quality_notes.append(f"EVE parsing failed: {e}")

    def _parse_eve_files_for_night(
        self, eve_files: list[Path], session: UnifiedSession
    ) -> None:
        """
        Parse all EVE files for a night and apply events to session based on timestamp filtering.

        Following OSCAR's behavior: EVE files store data for the whole day, so we read all EVE files
        and filter events to only include those within this session's time range.

        Args:
            eve_files: List of paths to EVE files for this night
            session: The session to add events to
        """
        from .formats.edf import (
            EDFDiscontinuousReader,
            get_edf_record_count,
            is_discontinuous_edf,
        )

        total_events_found = 0
        total_events_added = 0
        total_events_filtered = 0

        for eve_file in eve_files:
            try:
                record_count = get_edf_record_count(eve_file)
                if record_count == 0:
                    logger.debug(f"Skipping zero-record EVE file: {eve_file.name}")
                    continue

                is_discontinuous = is_discontinuous_edf(eve_file)

                if is_discontinuous:
                    with EDFDiscontinuousReader(eve_file) as edf:
                        annotations = edf.read_annotations()
                        eve_start_time = edf.get_header().start_datetime
                else:
                    with EDFReader(eve_file) as edf:
                        annotations = edf.read_annotations()
                        eve_start_time = edf.get_header().start_datetime

                logger.debug(
                    f"Processing EVE file {eve_file.name} with {len(annotations)} annotation(s)"
                )

                for annotation in annotations:
                    event_timestamp = annotation.to_datetime(eve_start_time)

                    # Check if event falls within session time range (OSCAR's checkInside logic)
                    if not (session.start_time <= event_timestamp <= session.end_time):
                        total_events_filtered += 1
                        continue

                    event_type = None

                    for text in annotation.annotations:
                        if text in self.FILTERED_ANNOTATIONS:
                            break

                        if text in self.EVENT_TYPE_MAP:
                            event_type = self.EVENT_TYPE_MAP[text]
                            break

                    if event_type is None:
                        continue

                    duration = annotation.duration if annotation.duration else 10.0

                    event = RespiratoryEvent(
                        event_type=event_type,
                        start_time=event_timestamp,
                        duration_seconds=duration,
                    )

                    session.add_event(event)
                    total_events_added += 1
                    total_events_found += 1

            except Exception as e:
                logger.warning(f"Failed to parse EVE file {eve_file.name}: {e}")
                continue

        if total_events_added > 0:
            logger.info(
                f"Added {total_events_added} events to session from {len(eve_files)} EVE file(s) "
                f"({total_events_filtered} events filtered out by timestamp)"
            )
        elif total_events_found == 0:
            logger.debug(f"No events found in {len(eve_files)} EVE file(s)")
        else:
            logger.info(
                f"No events within session time range (found {total_events_found} total events, "
                f"all filtered out)"
            )
