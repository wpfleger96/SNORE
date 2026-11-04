"""OSCAR-binary import against OSCAR's real on-disk channel IDs (issue #432).

Channel IDs are written here as literal hex, copied from OSCAR's
``oscar/SleepLib/schema.cpp`` (and ``resmed_loader.cpp`` for 0xE2xx), so these
tests fail if SNORE's constants drift from what OSCAR stores in .000/.001 files.
"""

from __future__ import annotations

import logging

from typing import Any

import pytest

from snore.parsers.resmed_edf import ResmedEDFParser
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

# schema.cpp: CPAP_Mode, CPAP_PressureMin/Max, CPAP_Pressure, CPAP_RampTime/Pressure.
CORE_SETTINGS = (0x1200, 0x1020, 0x1021, 0x110C, 0x1022, 0x1023)
# schema.cpp: CPAP_IPAP, EPAP, PS, IPAPHi, PSMin, PSMax, EPAPLo, EPAPHi.
BILEVEL_PRESSURES = (0x110D, 0x110E, 0x110F, 0x1111, 0x111A, 0x111B, 0x111C, 0x111D)
# resmed_loader.cpp: RMS9_EPR/EPRLevel/RampEnable, RMAS1x_RiseEnable/RiseTime/
# Cycle/Trigger/TiMax/TiMin, RMVENT_AutoEPAP/StartEPAP.
RESMED_SETTINGS = (0xE201, 0xE202, 0xE20D, *range(0xE212, 0xE218), 0xE22F, 0xE231)


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
    channels = (*CORE_SETTINGS, *BILEVEL_PRESSURES, *RESMED_SETTINGS)
    summary = _summary(settings=dict.fromkeys(channels, value))

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.mode == TherapyMode.CPAP
    assert settings.pressure_min is None
    assert settings.pressure_max is None
    assert settings.pressure_fixed is None
    assert (settings.ipap, settings.epap, settings.ps) == (None, None, None)
    assert settings.ramp_enabled is None
    assert settings.ramp_time is None
    assert settings.ramp_start_pressure is None
    assert settings.epr_mode is None
    assert settings.epr_level is None
    assert settings.other_settings == {}


# Each case is one ResMed AirSense/AirCurve 10 night as the SD-card parser sees
# its STR.edf record, and as OSCAR's ResMed loader (resmed_loader.cpp
# StoreSettings) stores it in the session's settings.
# schema.cpp: CPAP_Mode=0x1200, CPAP_Pressure=0x110C, CPAP_PressureMin/Max=
# 0x1020/0x1021, CPAP_RampTime=0x1022, CPAP_RampPressure=0x1023, CPAP_IPAP=
# 0x110D, CPAP_EPAP=0x110E, CPAP_PS=0x110F, CPAP_IPAPLo/Hi=0x1110/0x1111,
# CPAP_PSMin/Max=0x111A/0x111B, CPAP_EPAPLo/Hi=0x111C/0x111D.
# resmed_loader.cpp: RMS9_EPR=0xE201, RMS9_EPRLevel=0xE202, RMS9_RampEnable=
# 0xE20D, RMAS1x_RiseEnable/RiseTime/Cycle/Trigger/TiMax/TiMin=0xE212..0xE217,
# RMVENT_AutoEPAP=0xE22F, RMVENT_StartEPAP=0xE231.
_IVAPS_FIXED_EPAP_STR = {
    "mode": 9,
    "iv_epap": 6.0,
    "iv_epap_auto": 0,
    "iv_min_epap": 4.0,
    "iv_max_epap": 10.0,
    "iv_min_ps": 4.0,
    "iv_max_ps": 12.0,
    "iv_start_press": 5.0,
    "ramp_enabled": 1,
    "ramp_time": 10,
}
_IVAPS_AUTO_EPAP_STR = {**_IVAPS_FIXED_EPAP_STR, "iv_epap_auto": 1}

PARITY_CASES = [
    pytest.param(
        {
            "mode": 0,
            "pressure_fixed": 9.0,
            "epr_type_raw": 1,  # S10 raw + 1 = Full Time
            "epr_level": 2,
            "ramp_enabled": 1,
            "ramp_time": 20,
            "ramp_start_pressure": 4.0,
        },
        {
            0x1200: 1,
            0x110C: 9.0,
            0xE201: 2,
            0xE202: 2,
            0xE20D: 1,
            0x1022: 20,
            0x1023: 4.0,
        },
        id="cpap_epr_full_time_ramp",
    ),
    pytest.param(
        {
            "mode": 1,
            "pressure_min": 6.0,
            "pressure_max": 12.0,
            "epr_enable_raw": 0,
            "ramp_enabled": 2,  # SmartRamp
        },
        {0x1200: 2, 0x1020: 6.0, 0x1021: 12.0, 0xE201: 0, 0xE20D: 2},
        id="apap_epr_off_smart_ramp",
    ),
    pytest.param(
        {
            "mode": 3,
            "bl_ipap": 12.0,
            "bl_epap": 6.0,
            "bl_start_press": 4.0,
            "s10_ti_max": 1.5,
            "s10_ti_min": 0.3,
            "s10_trigger": 1,
            "s10_cycle": 2,
            "s10_rise_enable": 1,
            "s10_rise_time": 200,
            "ramp_enabled": 1,
            "ramp_time": 15,
        },
        {
            0x1200: 3,
            0x110E: 6.0,
            0x110D: 12.0,
            0x110F: 6.0,
            0xE216: 1.5,
            0xE217: 0.3,
            0xE215: 1,
            0xE214: 2,
            0xE212: 1,
            0xE213: 200,
            0xE20D: 1,
            0x1022: 15,
            0xE231: 4.0,
        },
        id="bipap",
    ),
    pytest.param(
        {
            "mode": 6,
            "va_min_epap": 5.0,
            "va_max_ipap": 15.0,
            "va_ps": 4.0,
            "va_start_press": 4.5,
            "s10_ti_max": 2.0,
            "s10_ti_min": 0.5,
            "s10_trigger": 4,
            "s10_cycle": 0,
            "ramp_enabled": 1,
            "ramp_time": 5,
        },
        {
            0x1200: 4,
            0x111C: 5.0,
            0x1111: 15.0,
            0x110F: 4.0,
            0xE216: 2.0,
            0xE217: 0.5,
            0xE215: 4,
            0xE214: 0,
            0xE20D: 1,
            0x1022: 5,
            0xE231: 4.5,
        },
        id="bipap_auto",
    ),
    pytest.param(
        {
            "mode": 7,
            "av_epap": 7.0,
            "av_min_ps": 0.0,
            "av_max_ps": 10.0,
            "av_start_press": 5.0,
            "ramp_enabled": 1,
            "ramp_time": 10,
        },
        {
            0x1200: 6,
            0x110E: 7.0,
            0x111A: 0.0,
            0x111B: 10.0,
            0x1111: 17.0,
            0xE20D: 1,
            0x1022: 10,
            0xE231: 5.0,
        },
        id="asv_min_ps_zero",
    ),
    pytest.param(
        {
            "mode": 8,
            "aa_min_epap": 5.0,
            "aa_max_epap": 9.0,
            "aa_min_ps": 3.0,
            "aa_max_ps": 10.0,
            "aa_start_press": 4.0,
            "ramp_enabled": 1,
            "ramp_time": 10,
        },
        {
            0x1200: 7,
            0x111C: 5.0,
            0x111D: 9.0,
            0x1110: 8.0,
            0x1111: 19.0,
            0x111A: 3.0,
            0x111B: 10.0,
            0xE20D: 1,
            0x1022: 10,
            0xE231: 4.0,
        },
        id="asv_auto",
    ),
    pytest.param(
        {
            **_IVAPS_FIXED_EPAP_STR,
            "s10_ti_max": 1.2,
            "s10_trigger": 2,
            "s10_cycle": 3,
            "s10_rise_enable": 1,
            "s10_rise_time": 300,
        },
        {
            # 0x110F is OSCAR's VAuto-bank PS, which iVAPS does not use.
            0x1200: 8,
            0x110E: 6.0,
            0x111A: 4.0,
            0x111B: 12.0,
            0x110F: 5.0,
            0xE22F: 0,
            0xE231: 5.0,
            0xE20D: 1,
            0x1022: 10,
            0xE216: 1.2,
            0xE215: 2,
            0xE214: 3,
            0xE212: 1,
            0xE213: 300,
        },
        id="ivaps_fixed_epap",
    ),
    pytest.param(
        _IVAPS_AUTO_EPAP_STR,
        {
            0x1200: 8,
            0x111C: 4.0,
            0x111D: 10.0,
            0x111A: 4.0,
            0x111B: 12.0,
            0xE22F: 1,
            0xE231: 5.0,
            0xE20D: 1,
            0x1022: 10,
        },
        id="ivaps_auto_epap",
    ),
    # OSCAR before commit e26b095a stored iVAPS through its generic branch: EPAP
    # and EPAPLo/EPAPHi together, IPAPHi = active EPAP + max PS, no 0xE22F, and
    # the ramp start as CPAP_RampPressure.
    pytest.param(
        _IVAPS_FIXED_EPAP_STR,
        {
            0x1200: 8,
            0x110E: 6.0,
            0x111C: 4.0,
            0x111D: 10.0,
            0x1110: 10.0,
            0x1111: 18.0,
            0x111A: 4.0,
            0x111B: 12.0,
            0x110F: 5.0,
            0xE20D: 1,
            0x1022: 10,
            0x1023: 5.0,
        },
        id="ivaps_fixed_epap_older_oscar",
    ),
    pytest.param(
        _IVAPS_AUTO_EPAP_STR,
        {
            0x1200: 8,
            0x110E: 6.0,
            0x111C: 4.0,
            0x111D: 10.0,
            0x1110: 8.0,
            0x1111: 22.0,
            0x111A: 4.0,
            0x111B: 12.0,
            0xE20D: 1,
            0x1022: 10,
            0x1023: 5.0,
        },
        id="ivaps_auto_epap_older_oscar",
    ),
]

# ramp_start_pressure is comparable only with ramp on: OSCAR stores it only
# when RampEnable >= 1, while the SD-card parser reports it regardless.
PARITY_FIELDS = {
    "mode",
    "pressure_fixed",
    "pressure_min",
    "pressure_max",
    "ipap",
    "epap",
    "ps",
    "epr_level",
    "epr_mode",
    "ramp_enabled",
    "ramp_time",
    "ramp_start_pressure",
    "other_settings",
}


@pytest.mark.parametrize(("str_record", "oscar_settings"), PARITY_CASES)
def test_oscar_settings_match_sd_card_import_of_same_night(
    tmp_path, str_record, oscar_settings
):
    sd_card_parser = ResmedEDFParser()
    sd_card_parser._str_series11 = False
    sd_card = sd_card_parser._convert_str_to_therapy_settings(
        {key: float(value) for key, value in str_record.items()}
    )

    oscar = parse_oscar_session(tmp_path, _summary(settings=oscar_settings)).settings

    assert sd_card is not None
    assert oscar is not None
    assert oscar.model_dump(include=PARITY_FIELDS) == sd_card.model_dump(
        include=PARITY_FIELDS
    )


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        # CPAP_EPAPLo wins over CPAP_EPAP for the headline EPAP.
        ({0x1200: 4, 0x111C: 5.0, 0x110E: 7.0}, {"epap": 5.0}),
        # A fixed CPAP_IPAP wins over CPAP_IPAPHi.
        ({0x1200: 3, 0x110D: 12.0, 0x1111: 20.0}, {"ipap": 12.0}),
        # CPAP_RampPressure wins over RMVENT_StartEPAP.
        ({0x1200: 1, 0x1023: 4.0, 0xE231: 5.0}, {"ramp_start_pressure": 4.0}),
        # bmc_loader.cpp stores CPAP_PS as 0 next to IPAP and EPAP.
        ({0x1200: 3, 0x110D: 14.0, 0x110E: 8.5, 0x110F: 0}, {"ps": 5.5}),
        # PS is not derived from a pair that does not rise.
        ({0x1200: 3, 0x110D: 8.0, 0x110E: 8.0}, {"ps": None}),
        # Derived IPAP is rounded: 7.4 + 3.2 is 10.600000000000001 in floats.
        ({0x1200: 8, 0x110E: 7.4, 0x111B: 3.2, 0xE22F: 0}, {"ipap": 10.6}),
        # Float32 integer settings round to the nearest integer.
        (
            {0x1200: 2, 0xE201: 2, 0xE202: 1.9999999, 0x1022: 19.999998},
            {"epr_level": 2, "ramp_time": 20},
        ),
    ],
    ids=[
        "epap_lo_over_epap",
        "ipap_over_ipap_hi",
        "ramp_pressure_over_start_epap",
        "ps_derived_from_ipap_and_epap",
        "ps_not_derived_without_rise",
        "derived_ipap_rounded",
        "integer_settings_rounded",
    ],
)
def test_setting_resolution_reads_expected_value(tmp_path, stored, expected):
    settings = parse_oscar_session(tmp_path, _summary(settings=stored)).settings

    assert settings is not None
    assert settings.model_dump(include=set(expected)) == expected


def test_variable_ps_with_only_min_ps_stores_range_without_ipap(tmp_path):
    summary = _summary(settings={0x1200: 6, 0x110E: 7.0, 0x111A: 3.0})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert (settings.ipap, settings.epap, settings.ps) == (None, 7.0, None)
    assert settings.other_settings == {
        "min_epap": "7.0",
        "max_epap": "7.0",
        "min_ps": "3.0",
    }


def test_bilevel_rise_time_without_rise_enable_is_not_stored(tmp_path):
    # RMAS1x_RiseEnable=0xE212 off; RMAS1x_RiseTime=0xE213 is then not a setting.
    summary = _summary(settings={0x1200: 3, 0xE212: 0, 0xE213: 200})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert "rise_time" not in settings.other_settings


@pytest.mark.parametrize(
    ("value", "expected_other"),
    [
        (0, {"min_ps": "0.0", "max_ps": "0.0"}),  # A PS range can start at 0
        (-1, {}),
    ],
    ids=["zero", "negative"],
)
def test_non_positive_bilevel_pressures_are_not_stored(tmp_path, value, expected_other):
    ramp_starts = (0x1023, 0xE231)  # CPAP_RampPressure, RMVENT_StartEPAP
    summary = _summary(
        settings={0x1200: 6, **dict.fromkeys((*BILEVEL_PRESSURES, *ramp_starts), value)}
    )

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert (settings.ipap, settings.epap, settings.ps) == (None, None, None)
    assert settings.ramp_start_pressure is None
    assert settings.other_settings == expected_other


@pytest.mark.parametrize(
    ("ramp_code", "expected_enabled", "expected_other"),
    [(0, False, {}), (1, True, {}), (2, False, {"smart_ramp": "True"})],
    ids=["off", "on", "smart_ramp"],
)
def test_ramp_enable_reads_enabled_and_smart_ramp(
    tmp_path, ramp_code, expected_enabled, expected_other
):
    # resmed_loader.cpp: RMS9_RampEnable=0xE20D, 0 Off, 1 On, 2 Auto (SmartRamp).
    summary = _summary(settings={0x1200: 1, 0xE20D: ramp_code})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.ramp_enabled is expected_enabled
    assert settings.other_settings == expected_other


@pytest.mark.parametrize(
    ("epr_code", "expected_mode"),
    [(1, "Ramp Only"), (2, "Full Time"), (3, "Unknown")],
    ids=str,
)
def test_epr_on_reads_mode_and_level(tmp_path, epr_code, expected_mode):
    summary = _summary(settings={0x1200: 2, 0xE201: epr_code, 0xE202: 2})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.epr_mode == expected_mode
    assert settings.epr_level == 2


@pytest.mark.parametrize("level", [4, -1], ids=str)
def test_out_of_range_epr_level_is_not_stored(tmp_path, level):
    summary = _summary(settings={0x1200: 2, 0xE201: 2, 0xE202: level})

    settings = parse_oscar_session(tmp_path, summary).settings

    assert settings is not None
    assert settings.epr_mode == "Full Time"
    assert settings.epr_level is None


def test_summary_averages_populate_pressure_leak_and_spo2_means(tmp_path):
    # schema.cpp: CPAP_Pressure=0x110C, CPAP_Leak=0x1108, OXI_SPO2=0x1801.
    summary = _summary(averages={0x110C: 8.5, 0x1108: 4.2, 0x1801: 95.0})

    stats = parse_oscar_session(tmp_path, summary).statistics

    assert stats.pressure_mean == 8.5
    assert stats.leak_mean == 4.2
    assert stats.spo2_mean == 95.0
