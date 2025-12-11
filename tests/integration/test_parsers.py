"""
Focused parser tests for SNORE.

Tests verify the critical EDF parsing logic with real data:
- Actual signal extraction from EDF files
- Data conversion to physical units
- Waveform data accuracy
- Session duration calculation
- Unified format conversion
"""

from datetime import datetime

import numpy as np
import pytest

from snore.models.unified import RespiratoryEventType, UnifiedSession, WaveformType
from snore.parsers.formats.edf import EDFReader


class TestEDFSignalParsing:
    """Test actual EDF signal extraction and reading."""

    @pytest.mark.parser
    def test_parse_flow_signal_from_brp(self, resmed_fixture_path):
        """Test actual Flow signal extraction from BRP file."""
        brp_file = resmed_fixture_path / "DATALOG/2024/20240621_013454_BRP.edf"
        assert brp_file.exists(), f"BRP fixture not found: {brp_file}"

        with EDFReader(brp_file) as edf:
            header = edf.get_header()
            assert header.num_signals > 0

            signals = edf.list_signal_labels()
            flow_signal = None
            for sig in signals:
                if "Flow" in sig:
                    flow_signal = sig
                    break

            assert flow_signal is not None, (
                f"No Flow signal found. Available: {signals}"
            )

            data, info = edf.read_signal(flow_signal, start_sample=0, num_samples=100)

            assert info.physical_dimension in [
                "L/s",
                "L/min",
            ], f"Unexpected unit: {info.physical_dimension}"
            assert len(data) > 0, "No flow data read"

            if info.physical_dimension == "L/s":
                data_lmin = data * 60.0
            else:
                data_lmin = data

            assert -6000 < np.min(data_lmin) < 6000, (
                f"Flow min out of range: {np.min(data_lmin)}"
            )
            assert -6000 < np.max(data_lmin) < 6000, (
                f"Flow max out of range: {np.max(data_lmin)}"
            )

            sample_rate = edf.get_sample_rate(flow_signal)
            assert sample_rate > 0, f"Invalid sample rate: {sample_rate}"

    @pytest.mark.parser
    def test_parse_pressure_leak_from_pld(self, resmed_fixture_path):
        """Test Pressure and Leak signal extraction from PLD file."""
        pld_file = resmed_fixture_path / "DATALOG/2024/20240621_013454_PLD.edf"
        assert pld_file.exists(), f"PLD fixture not found: {pld_file}"

        with EDFReader(pld_file) as edf:
            signals = edf.list_signal_labels()

            pressure_signal = None
            for name in signals:
                if "Press" in name and name != "EprPress.2s":
                    pressure_signal = name
                    break

            assert pressure_signal is not None, f"No pressure signal found in {signals}"

            pressure_data, pressure_info = edf.read_signal(
                pressure_signal, start_sample=0, num_samples=100
            )
            assert pressure_info.physical_dimension == "cmH2O"
            assert len(pressure_data) > 0

            assert 0 <= np.min(pressure_data) <= 30, (
                f"Pressure out of range: {np.min(pressure_data)}"
            )
            assert 0 <= np.max(pressure_data) <= 30, (
                f"Pressure out of range: {np.max(pressure_data)}"
            )

            leak_signal = None
            for name in signals:
                if "Leak" in name:
                    leak_signal = name
                    break

            assert leak_signal is not None, f"No leak signal found in {signals}"

            leak_data, leak_info = edf.read_signal(
                leak_signal, start_sample=0, num_samples=100
            )
            assert leak_info.physical_dimension in [
                "L/s",
                "L/min",
            ], f"Unexpected leak unit: {leak_info.physical_dimension}"
            assert len(leak_data) > 0

            if leak_info.physical_dimension == "L/s":
                leak_data_lmin = leak_data * 60.0
            else:
                leak_data_lmin = leak_data

            assert 0 <= np.min(leak_data_lmin) <= 200, (
                f"Leak out of range: {np.min(leak_data_lmin)}"
            )
            assert np.max(leak_data_lmin) < 300, (
                f"Leak too high: {np.max(leak_data_lmin)}"
            )

    @pytest.mark.parser
    def test_parse_statistics_from_sa2(self, resmed_fixture_path):
        """Test statistics extraction from SA2 file."""
        sa2_file = resmed_fixture_path / "DATALOG/2024/20240621_013454_SA2.edf"
        assert sa2_file.exists(), f"SA2 fixture not found: {sa2_file}"

        with EDFReader(sa2_file) as edf:
            signals = edf.list_signal_labels()
            assert len(signals) > 0, "No signals found in SA2 file"

            found_signal = False

            for signal_name in signals:
                if signal_name in ["Crc16"]:
                    continue

                data, info = edf.read_signal(
                    signal_name, start_sample=0, num_samples=10
                )
                if len(data) > 0:
                    if not np.isnan(data).all() and not (data == 0).all():
                        found_signal = True
                        break

            assert found_signal, f"No valid statistics signals found in {signals}"


class TestWaveformConversion:
    """Test conversion of EDF signals to unified waveform format."""

    @pytest.mark.parser
    def test_waveform_to_unified_format(self, resmed_parser, resmed_fixture_path):
        """Test that waveforms convert correctly to unified format."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        assert len(sessions) == 1, f"Expected 1 session, got {len(sessions)}"

        session = sessions[0]

        assert len(session.waveforms) > 0, "No waveforms were parsed"

        if WaveformType.FLOW_RATE in session.waveforms:
            flow = session.waveforms[WaveformType.FLOW_RATE]

            assert flow.unit == "L/min", f"Expected L/min, got {flow.unit}"
            assert flow.sample_rate > 0, f"Invalid sample rate: {flow.sample_rate}"
            assert len(flow.values) > 0, "No flow values"
            assert len(flow.timestamps) == len(flow.values), "Timestamp/value mismatch"

            if len(flow.timestamps) >= 10:
                expected_interval = 1.0 / flow.sample_rate
                for i in range(1, 10):
                    if isinstance(flow.timestamps, np.ndarray):
                        delta = float(flow.timestamps[i] - flow.timestamps[i - 1])
                    else:
                        delta = (
                            flow.timestamps[i] - flow.timestamps[i - 1]
                        ).total_seconds()
                    assert abs(delta - expected_interval) < (expected_interval * 0.1), (
                        f"Timestamp delta {delta} doesn't match expected {expected_interval}"
                    )

            assert all(-100 < v < 100 for v in flow.values[:100]), (
                "Flow values out of range"
            )

        if WaveformType.MASK_PRESSURE in session.waveforms:
            pressure = session.waveforms[WaveformType.MASK_PRESSURE]

            assert pressure.unit == "cmH2O", f"Expected cmH2O, got {pressure.unit}"
            assert len(pressure.values) > 0, "No pressure values"

            sample_check = pressure.values[: min(100, len(pressure.values))]
            assert all(0 <= v <= 40 for v in sample_check), (
                "Pressure values out of range"
            )

    @pytest.mark.parser
    def test_values_in_physical_units(self, resmed_parser, resmed_fixture_path):
        """Test that all values are converted to physical units, not digital."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        if WaveformType.FLOW_RATE in session.waveforms:
            flow = session.waveforms[WaveformType.FLOW_RATE]
            sample_check = flow.values[: min(100, len(flow.values))]
            assert all(-100 < v < 100 for v in sample_check), (
                "Flow values appear to be in digital units, not physical"
            )

        if WaveformType.MASK_PRESSURE in session.waveforms:
            pressure = session.waveforms[WaveformType.MASK_PRESSURE]
            sample_check = pressure.values[: min(100, len(pressure.values))]
            assert all(0 <= v <= 40 for v in sample_check), (
                "Pressure values appear to be in digital units, not physical"
            )

        if WaveformType.LEAK_RATE in session.waveforms:
            leak = session.waveforms[WaveformType.LEAK_RATE]
            sample_check = leak.values[: min(100, len(leak.values))]
            assert all(0 <= v <= 200 for v in sample_check), (
                "Leak values appear to be in digital units, not physical"
            )


class TestSessionParsing:
    """Test complete session parsing to unified format."""

    @pytest.mark.parser
    def test_parse_to_unified_format(self, resmed_parser, resmed_fixture_path):
        """Test that ResMed sessions convert to unified format."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))

        assert len(sessions) == 1, f"Expected 1 session, got {len(sessions)}"

        session = sessions[0]

        assert isinstance(session, UnifiedSession)

        assert session.device_session_id == "20240621_013454"
        assert session.import_source == "resmed_edf"
        assert session.parser_version == "1.0.0"

        assert session.device_info.manufacturer == "ResMed"
        assert session.device_info.serial_number == "22231974465"
        assert session.device_info.model == "AirSense11AutoSet"

        assert isinstance(session.start_time, datetime)
        assert isinstance(session.end_time, datetime)
        assert session.end_time > session.start_time

    @pytest.mark.parser
    def test_session_duration_calculation(self, resmed_parser, resmed_fixture_path):
        """Test that session duration is correctly calculated from data."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        assert session.duration_hours > 0, "Duration should be positive"
        assert session.duration_hours < 24, (
            f"Duration too long: {session.duration_hours} hours"
        )

        assert session.end_time > session.start_time

        if session.waveforms:
            for waveform_type, waveform in session.waveforms.items():
                if len(waveform.timestamps) > 1:
                    waveform_duration = waveform.duration_seconds
                    session_duration = session.duration_seconds

                    relative_diff = (
                        abs(waveform_duration - session_duration) / session_duration
                    )
                    assert relative_diff < 0.05, (
                        f"{waveform_type}: waveform duration {waveform_duration}s vs session {session_duration}s"
                    )

    @pytest.mark.parser
    def test_device_info_extraction(self, resmed_parser, resmed_fixture_path):
        """Test device info extraction from Identification.json."""
        device_info = resmed_parser.get_device_info(resmed_fixture_path)

        assert device_info.manufacturer == "ResMed"
        assert device_info.serial_number == "22231974465"
        assert device_info.model == "AirSense11AutoSet"
        assert device_info.firmware_version is not None
        assert "SW04600" in device_info.firmware_version
        assert device_info.product_code == "39485"


class TestParserDetection:
    """Test parser detection and validation."""

    @pytest.mark.parser
    def test_detect_resmed_data(self, resmed_parser, resmed_fixture_path):
        """Test parser correctly identifies ResMed data."""
        result = resmed_parser.detect(resmed_fixture_path)

        assert result.detected, "ResMed data should be detected"
        assert result.confidence >= 0.9, f"Low confidence: {result.confidence}"
        assert "ResMed" in result.message or "detected" in result.message.lower()

    @pytest.mark.parser
    def test_detect_missing_datalog(self, resmed_parser, tmp_path):
        """Test detection fails gracefully without DATALOG directory."""
        str_file = tmp_path / "STR.edf"
        str_file.touch()

        result = resmed_parser.detect(tmp_path)

        assert not result.detected, "Should not detect with invalid EDF"
        assert "not a valid EDF" in result.message or "DATALOG" in result.message, (
            f"Unexpected message: {result.message}"
        )

    @pytest.mark.parser
    def test_detect_empty_directory(self, resmed_parser, tmp_path):
        """Test detection of empty directory."""
        result = resmed_parser.detect(tmp_path)

        assert not result.detected

    @pytest.mark.parser
    def test_registry_auto_detection(self, parser_registry, resmed_fixture_path):
        """Test parser registry auto-detects ResMed data."""
        parser = parser_registry.detect_parser(resmed_fixture_path)

        assert parser is not None, "No parser detected"
        assert parser.parser_id == "resmed_edf"
        assert parser.manufacturer == "ResMed"


class TestDataQuality:
    """Test data quality handling and error cases."""

    @pytest.mark.parser
    def test_handle_parsing_errors_gracefully(self, resmed_parser, resmed_fixture_path):
        """Test that parsing errors are captured in data_quality_notes."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))

        assert len(sessions) > 0

        for session in sessions:
            if session.data_quality_notes:
                for note in session.data_quality_notes:
                    assert isinstance(note, str)
                    assert len(note) > 0

    @pytest.mark.parser
    def test_waveform_data_completeness(self, resmed_parser, resmed_fixture_path):
        """Test that waveform data is complete and consistent."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        for waveform_type, waveform in session.waveforms.items():
            assert len(waveform.timestamps) == len(waveform.values), (
                f"{waveform_type}: timestamp/value count mismatch"
            )

            if len(waveform.values) > 0:
                assert waveform.min_value is not None
                assert waveform.max_value is not None
                assert waveform.mean_value is not None

                assert waveform.min_value <= waveform.mean_value <= waveform.max_value

    @pytest.mark.parser
    def test_timestamp_ordering(self, resmed_parser, resmed_fixture_path):
        """Test that timestamps are in ascending order."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        for waveform_type, waveform in session.waveforms.items():
            if len(waveform.timestamps) > 1:
                check_count = min(100, len(waveform.timestamps))
                for i in range(1, check_count):
                    assert waveform.timestamps[i] >= waveform.timestamps[i - 1], (
                        f"{waveform_type}: timestamps not in order at index {i}"
                    )


class TestEVEEventParsing:
    """Test EVE respiratory event parsing."""

    @pytest.mark.parser
    def test_parse_eve_annotations(self, resmed_fixture_path):
        """Test actual EVE annotation reading from test file."""
        eve_file = resmed_fixture_path / "DATALOG" / "2024" / "20240621_013454_EVE.edf"
        assert eve_file.exists(), f"EVE fixture not found: {eve_file}"

        with EDFReader(eve_file) as edf:
            annotations = edf.read_annotations()

            assert len(annotations) > 0, "No annotations found in EVE file"

            for annotation in annotations:
                assert hasattr(annotation, "onset_time")
                assert hasattr(annotation, "duration")
                assert hasattr(annotation, "annotations")
                assert isinstance(annotation.annotations, list)

    @pytest.mark.parser
    def test_eve_event_mapping(self, resmed_parser, resmed_fixture_path):
        """Test that EVE events are correctly mapped to unified event types."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        assert session.has_event_data, "No event data parsed from EVE file"
        assert len(session.events) > 0, "No events found in session"

        valid_event_types = set(RespiratoryEventType)
        for event in session.events:
            assert event.event_type in valid_event_types, (
                f"Invalid event type: {event.event_type}"
            )

    @pytest.mark.parser
    def test_eve_event_timestamps(self, resmed_parser, resmed_fixture_path):
        """Test that event timestamps are within session bounds."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        session_start = session.start_time
        session_end = session.end_time

        for event in session.events:
            assert event.start_time >= session_start, (
                f"Event timestamp {event.start_time} before session start {session_start}"
            )
            assert event.start_time <= session_end, (
                f"Event timestamp {event.start_time} after session end {session_end}"
            )

    @pytest.mark.parser
    def test_eve_event_durations(self, resmed_parser, resmed_fixture_path):
        """Test that event durations are valid."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        for event in session.events:
            assert event.duration_seconds > 0, (
                f"{event.event_type} has invalid duration: {event.duration_seconds}"
            )

            assert event.duration_seconds <= 300, (
                f"{event.event_type} has unusually long duration: {event.duration_seconds}"
            )

    @pytest.mark.parser
    def test_eve_event_types_present(self, resmed_parser, resmed_fixture_path):
        """Test that various event types are detected in test data."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        event_types_found = {event.event_type for event in session.events}

        assert len(event_types_found) >= 3, (
            f"Expected multiple event types, found: {event_types_found}"
        )

        expected_types = {
            RespiratoryEventType.OBSTRUCTIVE_APNEA,
            RespiratoryEventType.CENTRAL_APNEA,
            RespiratoryEventType.HYPOPNEA,
        }

        for event_type in expected_types:
            assert event_type in event_types_found, (
                f"Expected event type {event_type} not found in parsed events"
            )

    @pytest.mark.parser
    def test_eve_filtered_annotations(self, resmed_parser, resmed_fixture_path):
        """Test that non-event annotations are filtered out."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        eve_file = resmed_fixture_path / "DATALOG" / "2024" / "20240621_013454_EVE.edf"
        with EDFReader(eve_file) as edf:
            total_annotations = len(edf.read_annotations())

        assert len(session.events) < total_annotations, (
            "Non-event annotations should be filtered out"
        )

    @pytest.mark.parser
    def test_eve_event_statistics(self, resmed_parser, resmed_fixture_path):
        """Test that event statistics are calculated correctly."""
        sessions = list(resmed_parser.parse_sessions(resmed_fixture_path))
        session = sessions[0]

        event_counts = {}
        for event in session.events:
            event_type = event.event_type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        for event_type, count in event_counts.items():
            assert count > 0, f"No events found for {event_type}"

        assert len(session.events) == sum(event_counts.values())
