"""OSCAR-binary import against OSCAR's real on-disk channel IDs (issue #432).

Channel IDs are written here as literal hex, copied from OSCAR's
``oscar/SleepLib/schema.cpp`` (and ``resmed_loader.cpp`` for 0xE202), so these
tests fail if SNORE's constants drift from what OSCAR stores in .000/.001 files.
"""

from __future__ import annotations

import logging

from typing import Any

import pytest

from snore.parsers.types import SessionEvents, SessionSummary
from snore.parsers.unified import RespiratoryEventType, TherapyMode, WaveformType
from tests.helpers.oscar_builders import (
    make_event_list,
    make_events,
    make_summary,
    make_waveform_list,
    parse_oscar_session,
)

pytestmark = pytest.mark.unit

START_MS = 1_705_278_600_000  # 2024-01-15 00:30 UTC
END_MS = START_MS + 6 * 3600 * 1000


def _hex_or_name(value: Any) -> str:
    return hex(value) if isinstance(value, int) else value.name


def _summary(**fields: Any) -> SessionSummary:
    return make_summary(START_MS, END_MS, **fields)


def _waveform_events(channel_id: int) -> SessionEvents:
    return make_events(
        START_MS, END_MS, make_waveform_list(channel_id, START_MS, [10, 11, 12, 13])
    )


@pytest.mark.parametrize(
    ("channel_id", "expected"),
    [
        (0x1100, WaveformType.FLOW_RATE),  # CPAP_FlowRate
        (0x1101, WaveformType.MASK_PRESSURE),  # CPAP_MaskPressure
        (0x1103, WaveformType.TIDAL_VOLUME),  # CPAP_TidalVolume
        (0x1105, WaveformType.MINUTE_VENTILATION),  # CPAP_MinuteVent
        (0x1106, WaveformType.RESPIRATORY_RATE),  # CPAP_RespRate
        (0x1108, WaveformType.LEAK_RATE),  # CPAP_Leak
        (0x110C, WaveformType.THERAPY_PRESSURE),  # CPAP_Pressure
        (0x110E, WaveformType.EPAP),  # CPAP_EPAP
        (0x1800, WaveformType.PULSE),  # OXI_Pulse
        (0x1801, WaveformType.SPO2),  # OXI_SPO2
    ],
    ids=_hex_or_name,
)
def test_waveform_channel_maps_to_waveform_type(tmp_path, channel_id, expected):
    session = parse_oscar_session(tmp_path, _summary(), _waveform_events(channel_id))

    assert set(session.waveforms) == {expected}


def test_csr_channel_is_not_a_waveform(tmp_path):
    # schema.cpp: CPAP_CSR=0x1000 is a span flag, not a sampled channel.
    session = parse_oscar_session(tmp_path, _summary(), _waveform_events(0x1000))

    assert not session.waveforms


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
        (0x1005, RespiratoryEventType.FLOW_LIMITATION),  # CPAP_FlowLimit
        (0x1028, RespiratoryEventType.PERIODIC_BREATHING),  # CPAP_PB
        (0x1000, RespiratoryEventType.PERIODIC_BREATHING),  # CPAP_CSR
    ],
    ids=_hex_or_name,
)
def test_event_channel_maps_to_event_type(tmp_path, channel_id, expected):
    events = make_events(
        START_MS, END_MS, make_event_list(channel_id, START_MS + 60_000, [12])
    )

    session = parse_oscar_session(tmp_path, _summary(), events)

    assert [event.event_type for event in session.events] == [expected]


@pytest.mark.parametrize(
    ("mode_value", "expected"),
    [
        # machine_common.h CPAPMode values.
        (1, TherapyMode.CPAP),  # MODE_CPAP
        (2, TherapyMode.APAP),  # MODE_APAP
        (3, TherapyMode.BIPAP),  # MODE_BILEVEL_FIXED
        (4, TherapyMode.BIPAP_AUTO),  # MODE_BILEVEL_AUTO_FIXED_PS
        (5, TherapyMode.BIPAP_AUTO),  # MODE_BILEVEL_AUTO_VARIABLE_PS
        (6, TherapyMode.ASV),  # MODE_ASV
        (7, TherapyMode.ASV_AUTO),  # MODE_ASV_VARIABLE_EPAP
        (8, TherapyMode.IVAPS),  # MODE_AVAPS (ResMed iVAPS)
    ],
    ids=str,
)
def test_mode_setting_maps_to_therapy_mode(tmp_path, mode_value, expected):
    settings = parse_oscar_session(
        tmp_path, _summary(settings={0x1200: mode_value})
    ).settings

    assert settings is not None
    assert settings.mode == expected


def test_unknown_mode_defaults_to_cpap_with_warning(tmp_path, caplog):
    # machine_common.h: MODE_UNKNOWN=0 has no TherapyMode.
    with caplog.at_level(logging.WARNING, logger="snore.parsers.oscar_device"):
        settings = parse_oscar_session(
            tmp_path, _summary(settings={0x1200: 0})
        ).settings

    assert settings is not None
    assert settings.mode == TherapyMode.CPAP
    assert "Unknown OSCAR therapy mode value 0" in caplog.text


def test_apap_settings_read_mode_pressure_range_and_epr(tmp_path):
    # schema.cpp: CPAP_Mode=0x1200 (2 = MODE_APAP), CPAP_PressureMin=0x1020,
    # CPAP_PressureMax=0x1021; resmed_loader.cpp: RMS9_EPRLevel=0xE202.
    summary = _summary(settings={0x1200: 2, 0x1020: 6.0, 0x1021: 12.0, 0xE202: 3})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.mode == TherapyMode.APAP
    assert settings.pressure_min == 6.0
    assert settings.pressure_max == 12.0
    assert settings.epr_level == 3


def test_fixed_cpap_settings_read_set_pressure(tmp_path):
    # schema.cpp: CPAP_Mode=0x1200 (1 = MODE_CPAP), CPAP_Pressure=0x110C.
    summary = _summary(settings={0x1200: 1, 0x110C: 9.0})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.mode == TherapyMode.CPAP
    assert settings.pressure_fixed == 9.0


def test_placeholder_pressures_are_not_stored(tmp_path):
    # yuwell_loader.cpp writes -1 to the pressure settings as a "fill in later" marker.
    summary = _summary(settings={0x1200: 2, 0x1020: -1, 0x1021: -1, 0x110C: -1})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.mode == TherapyMode.APAP
    assert settings.pressure_min is None
    assert settings.pressure_max is None
    assert settings.pressure_fixed is None


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), "9", True, [1]],
    ids=["nan", "inf", "string", "bool", "list"],
)
def test_non_numeric_settings_are_not_stored(tmp_path, value):
    channels = (0x1200, 0x1020, 0x1021, 0x110C, 0xE202)
    summary = _summary(settings=dict.fromkeys(channels, value))

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.mode == TherapyMode.CPAP
    assert settings.pressure_min is None
    assert settings.pressure_max is None
    assert settings.pressure_fixed is None
    assert settings.epr_level is None


def test_summary_averages_populate_pressure_leak_and_spo2_means(tmp_path):
    # schema.cpp: CPAP_Pressure=0x110C, CPAP_Leak=0x1108, OXI_SPO2=0x1801.
    summary = _summary(averages={0x110C: 8.5, 0x1108: 4.2, 0x1801: 95.0})

    stats = parse_oscar_session(tmp_path, summary).statistics

    assert stats.pressure_mean == 8.5
    assert stats.leak_mean == 4.2
    assert stats.spo2_mean == 95.0
