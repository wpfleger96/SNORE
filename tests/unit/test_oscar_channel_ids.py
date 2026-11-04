"""OSCAR-binary import against OSCAR's real on-disk channel IDs (issue #432).

Channel IDs are written here as literal hex, copied from OSCAR's
``oscar/SleepLib/schema.cpp`` (and ``resmed_loader.cpp`` for 0xE202), so these
tests fail if SNORE's constants drift from what OSCAR stores in .000/.001 files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from snore.parsers.oscar_device import OscarDeviceParser
from snore.parsers.types import EventList, EventListType, SessionEvents, SessionSummary
from snore.parsers.unified import (
    DeviceInfo,
    RespiratoryEventType,
    TherapyMode,
    UnifiedSession,
    WaveformType,
)

pytestmark = pytest.mark.unit

START_MS = 1_705_278_600_000  # 2024-01-15 00:30 UTC
END_MS = START_MS + 6 * 3600 * 1000


def _summary(**fields: Any) -> SessionSummary:
    return SessionSummary(
        magic=0xC73216AB,
        version=10,
        file_type=0,
        machine_id=1,
        session_id=START_MS // 1000,
        first_timestamp=START_MS,
        last_timestamp=END_MS,
        **fields,
    )


def _waveform(channel_id: int) -> EventList:
    return EventList(
        channel_id=channel_id,
        first_timestamp=START_MS,
        last_timestamp=START_MS + 1000,
        count=4,
        event_type=EventListType.WAVEFORM,
        sample_rate=4.0,
        gain=1.0,
        offset=0.0,
        min_value=0.0,
        max_value=30.0,
        dimension="",
        data=[10, 11, 12, 13],
    )


def _event(channel_id: int) -> EventList:
    return EventList(
        channel_id=channel_id,
        first_timestamp=START_MS + 60_000,
        last_timestamp=START_MS + 60_000,
        count=1,
        event_type=EventListType.EVENT,
        sample_rate=0.0,
        gain=1.0,
        offset=0.0,
        min_value=0.0,
        max_value=30.0,
        dimension="s",
        data=[12],
        time_deltas=[0],
    )


def _events(*event_lists: EventList) -> SessionEvents:
    return SessionEvents(
        magic=0xC73216AB,
        version=10,
        file_type=1,
        machine_id=1,
        session_id=START_MS // 1000,
        first_timestamp=START_MS,
        last_timestamp=END_MS,
        event_lists={el.channel_id: [el] for el in event_lists},
    )


def _parse(
    tmp_path: Path, summary: SessionSummary, events: SessionEvents | None = None
) -> UnifiedSession:
    summary_path = tmp_path / "1.000"
    summary_path.touch()
    events_path = tmp_path / "1.001"
    events_path.touch()

    with (
        patch("snore.parsers.oscar_device.parse_summary_file", return_value=summary),
        patch("snore.parsers.oscar_device.parse_events_file", return_value=events),
    ):
        return OscarDeviceParser()._parse_single_session(
            session_id=START_MS // 1000,
            summary_path=summary_path,
            events_path=events_path,
            device_info=DeviceInfo(
                manufacturer="OSCAR", model="Test", serial_number="TEST123"
            ),
            base_path=tmp_path,
        )


def test_flow_and_leak_channels_become_waveforms_csr_does_not(tmp_path):
    # schema.cpp: CPAP_FlowRate=0x1100, CPAP_Leak=0x1108, CPAP_CSR=0x1000 (span).
    events = _events(_waveform(0x1100), _waveform(0x1108), _waveform(0x1000))

    session = _parse(tmp_path, _summary(), events)

    assert set(session.waveforms) == {WaveformType.FLOW_RATE, WaveformType.LEAK_RATE}


@pytest.mark.parametrize(
    ("channel_id", "expected"),
    [
        # schema.cpp event flags.
        (0x1002, RespiratoryEventType.OBSTRUCTIVE_APNEA),  # CPAP_Obstructive
        (0x1003, RespiratoryEventType.HYPOPNEA),  # CPAP_Hypopnea
        (0x1001, RespiratoryEventType.CENTRAL_APNEA),  # CPAP_ClearAirway
        (0x1004, RespiratoryEventType.UNCLASSIFIED_APNEA),  # CPAP_Apnea
        (0x1010, RespiratoryEventType.UNCLASSIFIED_APNEA),  # CPAP_AllApnea
        (0x1006, RespiratoryEventType.RERA),  # CPAP_RERA
    ],
)
def test_event_channel_maps_to_event_type(tmp_path, channel_id, expected):
    session = _parse(tmp_path, _summary(), _events(_event(channel_id)))

    assert [event.event_type for event in session.events] == [expected]


def test_apap_settings_read_mode_pressure_range_and_epr(tmp_path):
    # schema.cpp: CPAP_Mode=0x1200 (2 = MODE_APAP), CPAP_PressureMin=0x1020,
    # CPAP_PressureMax=0x1021; resmed_loader.cpp: RMS9_EPRLevel=0xE202.
    summary = _summary(settings={0x1200: 2, 0x1020: 6.0, 0x1021: 12.0, 0xE202: 3})

    settings = _parse(tmp_path, summary).settings

    assert settings is not None
    assert settings.mode == TherapyMode.APAP
    assert settings.pressure_min == 6.0
    assert settings.pressure_max == 12.0
    assert settings.epr_level == 3


def test_fixed_cpap_settings_read_set_pressure(tmp_path):
    # schema.cpp: CPAP_Mode=0x1200 (1 = MODE_CPAP), CPAP_Pressure=0x110C.
    summary = _summary(settings={0x1200: 1, 0x110C: 9.0})

    settings = _parse(tmp_path, summary).settings

    assert settings is not None
    assert settings.mode == TherapyMode.CPAP
    assert settings.pressure_fixed == 9.0


def test_summary_averages_populate_pressure_leak_and_spo2_means(tmp_path):
    # schema.cpp: CPAP_Pressure=0x110C, CPAP_Leak=0x1108, OXI_SPO2=0x1801.
    summary = _summary(averages={0x110C: 8.5, 0x1108: 4.2, 0x1801: 95.0})

    stats = _parse(tmp_path, summary).statistics

    assert stats.pressure_mean == 8.5
    assert stats.leak_mean == 4.2
    assert stats.spo2_mean == 95.0
