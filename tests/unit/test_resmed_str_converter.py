"""Unit tests for ResmedEDFParser STR.edf conversion helpers.

Covers the OSCAR-parity findings F1–F20 plus the existing noon-anchor and
mode/mask decode contracts.
"""

import json

from datetime import date, datetime
from pathlib import Path

import pytest

from snore.parsers.resmed_edf import ResmedEDFParser
from snore.parsers.unified import TherapyMode


class TestTherapyDateHelper:
    """Tests for _therapy_date — ResMed noon-to-noon day convention."""

    def test_therapy_date_after_midnight_maps_to_previous_day(self):
        """Session starting at 01:11 belongs to the therapy night of the day before."""
        assert ResmedEDFParser._therapy_date(datetime(2026, 5, 28, 1, 11)) == date(
            2026, 5, 27
        )

    def test_therapy_date_before_midnight_maps_to_same_day(self):
        """Session starting at 23:50 belongs to the therapy night that started that day."""
        assert ResmedEDFParser._therapy_date(datetime(2026, 5, 27, 23, 50)) == date(
            2026, 5, 27
        )

    def test_therapy_date_at_noon_maps_to_same_day(self):
        """Noon exactly is the start of the new therapy day — maps to that day."""
        assert ResmedEDFParser._therapy_date(datetime(2026, 5, 27, 12, 0)) == date(
            2026, 5, 27
        )

    def test_therapy_date_one_minute_before_noon_maps_to_previous_day(self):
        """11:59 is still within the previous day's therapy window."""
        assert ResmedEDFParser._therapy_date(datetime(2026, 5, 27, 11, 59)) == date(
            2026, 5, 26
        )


class TestStrNoonAnchor:
    """F13: regression lock on the STR.edf noon-to-noon date indexing contract.

    pyedflib returns STR.edf header start times as naive local datetimes.
    ResMed STR.edf files always start at noon local time (OSCAR :1595,
    comment :1293 "each STR.edf record starts at 12 noon").  Calling .date()
    on the returned datetime must therefore yield the correct first-record
    calendar date without any timezone conversion.
    """

    def test_noon_datetime_date_extraction(self):
        """Noon datetime .date() gives the same calendar date — no offset needed."""
        noon = datetime(2025, 2, 21, 12, 0, 0)
        assert noon.date() == date(2025, 2, 21)

    def test_afternoon_datetime_still_same_date(self):
        """Any time between noon and midnight maps to the same calendar day."""
        afternoon = datetime(2025, 2, 21, 18, 30, 0)
        assert afternoon.date() == date(2025, 2, 21)

    def test_therapy_date_noon_start_maps_correctly(self):
        """A session recorded at noon on day D belongs to therapy day D."""
        noon_session = datetime(2025, 6, 15, 12, 0, 0)
        assert ResmedEDFParser._therapy_date(noon_session) == date(2025, 6, 15)


# ---------------------------------------------------------------------------
# Mode matrix tests — F1, F7, F15, F16
# ---------------------------------------------------------------------------


class TestModeMatrixSeries11:
    """S11 raw mode → expected TherapyMode (or None for warn+skip)."""

    @pytest.fixture
    def parser(self):
        p = ResmedEDFParser()
        p._str_series11 = True
        return p

    @pytest.mark.parametrize(
        "raw_mode, expected",
        [
            (1, TherapyMode.APAP),  # AutoSet
            (2, TherapyMode.APAP),  # AutoSet for Her
            (3, TherapyMode.CPAP),
            (4, TherapyMode.BIPAP),  # F1: was missing
            (6, TherapyMode.ASV),  # F1: was missing
            (7, TherapyMode.ASV_AUTO),  # F15: ASV variable-EPAP
            (8, TherapyMode.BIPAP_AUTO),  # VAuto
        ],
    )
    def test_known_modes(self, parser, raw_mode, expected):
        record = {"mode": float(raw_mode)}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None, f"S11 mode {raw_mode} should map to {expected}"
        assert settings.mode == expected

    @pytest.mark.parametrize("raw_mode", [0, 5])
    def test_warn_skip_modes(self, parser, raw_mode):
        """S11 modes 0 and 5 are unimplemented — must return None (warn+skip)."""
        record = {"mode": float(raw_mode)}
        assert parser._convert_str_to_therapy_settings(record) is None


class TestModeMatrixSeries10:
    """S9/S10 raw mode → expected TherapyMode (or None for warn+skip)."""

    @pytest.fixture
    def parser(self):
        p = ResmedEDFParser()
        p._str_series11 = False
        return p

    @pytest.mark.parametrize(
        "raw_mode, expected",
        [
            (0, TherapyMode.CPAP),
            (1, TherapyMode.APAP),
            (2, TherapyMode.BIPAP),
            (3, TherapyMode.BIPAP),
            (4, TherapyMode.BIPAP_ST),
            (5, TherapyMode.BIPAP_ST),
            (6, TherapyMode.BIPAP_AUTO),  # VAuto
            (7, TherapyMode.ASV),
            (8, TherapyMode.ASV_AUTO),  # F15: was ASV
            (9, TherapyMode.IVAPS),  # F7: new
            (11, TherapyMode.APAP),  # A4Her
        ],
    )
    def test_known_modes(self, parser, raw_mode, expected):
        record = {"mode": float(raw_mode)}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None, f"S10 mode {raw_mode} should map to {expected}"
        assert settings.mode == expected

    def test_mode10_pac_warn_skip(self, parser):
        """S10 mode 10 (PAC) is unimplemented — must return None."""
        record = {"mode": 10.0}
        assert parser._convert_str_to_therapy_settings(record) is None


# ---------------------------------------------------------------------------
# Sentinel and basic field tests
# ---------------------------------------------------------------------------


class TestConvertStrToTherapySettings:
    """Tests for _convert_str_to_therapy_settings sentinel detection and field mapping."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    def test_all_sentinel_values_returns_none(self, parser):
        """All-negative dict (no-usage day) must return None, not a degenerate TherapySettings."""
        parser._str_series11 = False
        sentinel_record = {
            "mode": -1.0,
            "pressure_fixed": -0.02,
            "pressure_min": -0.02,
            "pressure_max": -0.02,
            "epr_level": -0.02,
            "epr_type_raw": -1.0,
            "climate_control": -1.0,
            "humidity_enabled": -1.0,
            "humidity_level": -1.0,
            "smart_start": -1.0,
            "ab_filter": -1.0,
            "mask_type": -1.0,
            "tube_temp": -0.1,
        }
        assert parser._convert_str_to_therapy_settings(sentinel_record) is None

    def test_valid_s10_cpap_record(self, parser):
        """Valid S10 CPAP record: mode 0, S10-basis enum values throughout."""
        parser._str_series11 = False
        record = {
            "mode": 0.0,
            "pressure_fixed": 10.0,
            "epr_level": 2.0,
            "epr_type_raw": 1.0,  # S10 raw+1 → code 2 → "Full Time"
            "climate_control": 0.0,  # S10: 0=Auto (OSCAR :233-234)
            "humidity_enabled": 1.0,  # S10: 1=on
            "humidity_level": 4.0,
            "smart_start": 1.0,  # S10: 1=on
            "ab_filter": 1.0,  # 0=Standard, 1=Antibacterial (same basis)
            "mask_type": 2.0,  # S10: 2=Nasal
            "tube_temp": 27.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.CPAP
        assert settings.pressure_fixed == 10.0
        assert settings.epr_level == 2
        assert settings.epr_mode == "Full Time"
        assert settings.climate_control == "Auto"
        assert settings.humidity_enabled is True
        assert settings.humidity_level == 4
        assert settings.smart_start is True
        assert settings.ab_filter == "Antibacterial"
        assert settings.mask_type == "Nasal"
        assert settings.tube_temp == 27.0

    def test_valid_s11_cpap_record(self, parser):
        """Valid S11 CPAP record: mode 3 (→S10 0), S11-basis enum values (shifted +1)."""
        parser._str_series11 = True
        record = {
            "mode": 3.0,  # S11 raw 3 → S10 0 = CPAP
            "pressure_fixed": 9.0,
            "epr_level": 1.0,
            "epr_type_raw": 2.0,  # S11 raw, unchanged (+1−1=0) → code 2 → "Full Time"
            "climate_control": 1.0,  # S11 raw 1 → norm 0 = Auto (OSCAR :233-234)
            "humidity_enabled": 2.0,  # S11 raw 2 → norm 1 = on
            "humidity_level": 3.0,
            "smart_start": 2.0,  # S11 raw 2 → norm 1 = on
            "ab_filter": 2.0,  # S11 raw 2 → norm 1 = Antibacterial
            "mask_type": 4.0,  # S11 raw 4 → −2 = 2 = Nasal
            "tube_temp": 25.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.CPAP
        assert settings.pressure_fixed == 9.0
        assert settings.epr_level == 1
        assert settings.epr_mode == "Full Time"
        assert settings.climate_control == "Auto"
        assert settings.humidity_enabled is True
        assert settings.humidity_level == 3
        assert settings.smart_start is True
        assert settings.ab_filter == "Antibacterial"
        assert settings.mask_type == "Nasal"
        assert settings.tube_temp == 25.0

    def test_mixed_record_returns_settings_with_sentinel_fields_as_none(self, parser):
        """Dict with some sentinels and some valid values must not be discarded."""
        parser._str_series11 = False
        mixed_record = {
            "mode": 1.0,
            "pressure_min": -0.02,  # sentinel → None
            "pressure_max": 20.0,
            "epr_level": 2.0,
        }
        settings = parser._convert_str_to_therapy_settings(mixed_record)
        assert settings is not None
        assert settings.mode == TherapyMode.APAP
        assert settings.pressure_min is None
        assert settings.pressure_max == 20.0
        assert settings.epr_level == 2

    def test_series11_vauto_mode8_maps_bipap_auto(self, parser):
        """S11 mode 8 with VA signals maps to BIPAP_AUTO; stale presets don't leak."""
        parser._str_series11 = True
        record = {
            "mode": 8.0,
            "pressure_fixed": 10.0,  # stale — must not appear
            "pressure_min": 4.0,  # stale
            "pressure_max": 20.0,  # stale
            "epr_level": 1.0,  # stale
            "epr_type_raw": 2.0,  # stale
            "va_min_epap": 5.0,
            "va_max_ipap": 14.0,
            "va_ps": 3.0,
            "va_start_press": 5.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.BIPAP_AUTO
        assert settings.epap == 5.0
        assert settings.ipap == 14.0
        assert settings.ps == 3.0
        assert settings.pressure_fixed is None
        assert settings.pressure_min is None
        assert settings.pressure_max is None
        assert settings.epr_level is None
        assert settings.epr_mode is None

    def test_vauto_ti_trigger_cycle_land_in_other_settings(self, parser):
        """VAuto timing parameters are stored as strings in other_settings."""
        parser._str_series11 = True
        record = {
            "mode": 8.0,
            "va_min_epap": 5.0,
            "va_max_ipap": 14.0,
            "va_ps": 3.0,
            "va_ti_max": 2.5,
            "va_ti_min": 0.3,
            # S11 raw 4: _norm applies −1 → 3 (OSCAR :2400-2410 does --R.s_Trigger for S11).
            "va_trigger": 4.0,
            "va_cycle": 2.0,  # S11 raw 2 → norm 1
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings["ti_max"] == "2.5"
        assert settings.other_settings["ti_min"] == "0.3"
        assert settings.other_settings["trigger"] == "High"
        assert settings.other_settings["cycle"] == "Low"

    def test_series11_mode3_cpap_has_pressure_fixed_and_epr(self, parser):
        """S11 mode 3 → CPAP with pressure_fixed and EPR; no bilevel fields."""
        parser._str_series11 = True
        record = {
            "mode": 3.0,
            "pressure_fixed": 9.0,
            "epr_level": 1.0,
            "epr_type_raw": 1.0,  # S11 raw → unchanged → code 1 → "Ramp Only"
            "va_min_epap": 4.0,  # stale VAuto keys must not leak
            "va_max_ipap": 12.0,
            "va_ps": 2.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.CPAP
        assert settings.pressure_fixed == 9.0
        assert settings.epr_level == 1
        assert settings.epr_mode == "Ramp Only"
        assert settings.ipap is None
        assert settings.epap is None

    def test_series11_mode1_apap_pressure_min_max(self, parser):
        """S11 mode 1 → APAP with pressure_min/pressure_max."""
        parser._str_series11 = True
        record = {
            "mode": 1.0,
            "pressure_min": 4.0,
            "pressure_max": 20.0,
            "ramp_start_pressure": 4.0,
            "epr_level": 2.0,
            "epr_type_raw": 2.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.APAP
        assert settings.pressure_min == 4.0
        assert settings.pressure_max == 20.0
        assert settings.pressure_fixed is None

    def test_series11_mode2_apap_via_afh_signals(self, parser):
        """S11 mode 2 (A4Her) falls back to AFH signals for pressure range."""
        parser._str_series11 = True
        record = {
            "mode": 2.0,
            "afh_min_press": 5.0,
            "afh_max_press": 18.0,
            "afh_start_press": 5.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.APAP
        assert settings.pressure_min == 5.0
        assert settings.pressure_max == 18.0
        assert settings.ramp_start_pressure == 5.0

    def test_unknown_mode_series11_returns_none(self, parser):
        """Unmapped mode value on Series 11 returns None (warn+skip)."""
        parser._str_series11 = True
        record = {"mode": 5.0, "pressure_fixed": 10.0}
        assert parser._convert_str_to_therapy_settings(record) is None

    def test_unknown_mode_series10_returns_none(self, parser):
        """Unmapped mode value on Series 10 returns None."""
        parser._str_series11 = False
        record = {"mode": 99.0, "pressure_fixed": 10.0}
        assert parser._convert_str_to_therapy_settings(record) is None

    def test_missing_mode_key_returns_none(self, parser):
        """Missing mode key returns None; a settings record without a mode is unusable."""
        record = {"pressure_fixed": 10.0, "humidity_level": 4.0}
        assert parser._convert_str_to_therapy_settings(record) is None

    def test_series10_mode6_maps_bipap_auto(self, parser):
        """S10 mode 6 (VAuto) maps to BIPAP_AUTO with VA signals."""
        parser._str_series11 = False
        record = {
            "mode": 6.0,
            "va_min_epap": 6.0,
            "va_max_ipap": 15.0,
            "va_ps": 4.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.BIPAP_AUTO
        assert settings.epap == 6.0
        assert settings.ipap == 15.0

    def test_vauto_ps_sentinel_becomes_none(self, parser):
        """VAuto ps sentinel value (-0.02) results in ps=None."""
        parser._str_series11 = True
        record = {
            "mode": 8.0,
            "va_min_epap": 5.0,
            "va_max_ipap": 14.0,
            "va_ps": -0.02,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.ps is None


# ---------------------------------------------------------------------------
# F2 — S10 bilevel S.BL.* signals
# ---------------------------------------------------------------------------


class TestS10BilevelSignals:
    """F2: S10 BIPAP/BIPAP_ST reads S.BL.* instead of S.S.*."""

    @pytest.fixture
    def parser(self):
        p = ResmedEDFParser()
        p._str_series11 = False
        return p

    def test_s10_mode2_bipap_uses_bl_signals(self, parser):
        """S10 mode 2 (BIPAP) reads ipap/epap from bl_ipap/bl_epap."""
        record = {
            "mode": 2.0,
            "bl_ipap": 16.0,
            "bl_epap": 8.0,
            "bl_start_press": 8.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.BIPAP
        assert settings.ipap == 16.0
        assert settings.epap == 8.0
        assert settings.ps == pytest.approx(8.0)
        assert settings.ramp_start_pressure == 8.0

    def test_s10_mode3_bipap_computes_ps(self, parser):
        """S10 mode 3 (VPAP S) maps to BIPAP with computed ps = ipap - epap."""
        record = {
            "mode": 3.0,
            "bl_ipap": 14.0,
            "bl_epap": 8.0,
            "bl_start_press": 8.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.BIPAP
        assert settings.ipap == 14.0
        assert settings.epap == 8.0
        assert settings.ps == 6.0

    def test_s10_bilevel_timing_in_other_settings(self, parser):
        """S10 BIPAP timing from bare S.* signals lands in other_settings."""
        record = {
            "mode": 3.0,
            "bl_ipap": 14.0,
            "bl_epap": 8.0,
            # Bare s10_* keys mirror OSCAR :2347-2392 sigprefix "S." for S10.
            "s10_ti_max": 2.0,
            "s10_ti_min": 0.3,
            "s10_trigger": 2.0,
            "s10_cycle": 1.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("ti_max") == "2.0"
        assert settings.other_settings.get("ti_min") == "0.3"
        assert settings.other_settings.get("trigger") == "Med"
        assert settings.other_settings.get("cycle") == "Low"

    def test_s10_bilevel_bl_fallback_when_bare_absent(self, parser):
        """S10 BIPAP falls back to S.BL.* timing keys when bare S.* keys are absent."""
        record = {
            "mode": 3.0,
            "bl_ipap": 14.0,
            "bl_epap": 8.0,
            # No s10_* keys present — converter must fall back to bl_* fallback.
            "bl_ti_max": 2.0,
            "bl_ti_min": 0.3,
            "bl_trigger": 2.0,
            "bl_cycle": 1.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("ti_max") == "2.0"
        assert settings.other_settings.get("ti_min") == "0.3"
        assert settings.other_settings.get("trigger") == "Med"
        assert settings.other_settings.get("cycle") == "Low"

    def test_s11_bipap_still_uses_smode_signals(self, parser):
        """S11 BIPAP (raw mode 4) still uses S.S.* (STR_SMODE_SIGNALS)."""
        parser._str_series11 = True
        record = {
            "mode": 4.0,  # S11 raw 4 → S10 mode 3 = BIPAP
            "s_ipap": 18.0,
            "s_epap": 10.0,
            "s_start_press": 10.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.BIPAP
        assert settings.ipap == 18.0
        assert settings.epap == 10.0


# ---------------------------------------------------------------------------
# F3 — S10 vAuto bare timing signals
# ---------------------------------------------------------------------------


class TestS10VAutoTimingSignals:
    """F3: S10 vAuto Cycle/Trigger/TiMax/TiMin use bare S.* signals."""

    @pytest.fixture
    def parser(self):
        p = ResmedEDFParser()
        p._str_series11 = False
        return p

    def test_s10_vauto_timing_uses_bare_signals(self, parser):
        """S10 VAuto timing comes from s10_cycle/s10_trigger/s10_ti_max/s10_ti_min."""
        record = {
            "mode": 6.0,
            "va_min_epap": 5.0,
            "va_max_ipap": 14.0,
            "va_ps": 3.0,
            "s10_ti_max": 1.8,
            "s10_ti_min": 0.3,
            "s10_trigger": 2.0,
            "s10_cycle": 1.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("ti_max") == "1.8"
        assert settings.other_settings.get("ti_min") == "0.3"
        assert settings.other_settings.get("trigger") == "Med"
        assert settings.other_settings.get("cycle") == "Low"

    def test_s11_vauto_timing_uses_va_signals(self, parser):
        """S11 VAuto timing comes from va_cycle/va_trigger/va_ti_max/va_ti_min."""
        parser._str_series11 = True
        record = {
            "mode": 8.0,  # S11 raw 8 = VAuto
            "va_min_epap": 5.0,
            "va_max_ipap": 14.0,
            "va_ps": 3.0,
            "va_ti_max": 2.0,
            "va_ti_min": 0.5,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("ti_max") == "2.0"
        assert settings.other_settings.get("ti_min") == "0.5"


# ---------------------------------------------------------------------------
# F4, F5, F14 — enum normalization for both families
# ---------------------------------------------------------------------------


class TestEnumNormalization:
    """F4 (ClimateControl), F5 (ABFilter), F14 (Hum/Temp enable) normalization."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    @pytest.mark.parametrize(
        "series11, raw, expected",
        [
            (False, 0.0, "Auto"),  # S10 raw 0 = Auto
            (False, 1.0, "Manual"),  # S10 raw 1 = Manual
            (True, 1.0, "Auto"),  # S11 raw 1 → norm 0 = Auto
            (True, 2.0, "Manual"),  # S11 raw 2 → norm 1 = Manual
        ],
    )
    def test_climate_control(self, parser, series11, raw, expected):
        """F4: climate control normalized to S10 basis before map lookup."""
        parser._str_series11 = series11
        record = {"mode": 0.0 if not series11 else 3.0, "climate_control": raw}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.climate_control == expected

    @pytest.mark.parametrize(
        "series11, raw, expected",
        [
            (False, 0.0, "Standard"),  # S10 raw 0 = Standard
            (False, 1.0, "Antibacterial"),  # S10 raw 1 = Antibacterial
            (True, 1.0, "Standard"),  # S11 raw 1 → norm 0 = Standard
            (True, 2.0, "Antibacterial"),  # S11 raw 2 → norm 1 = Antibacterial
        ],
    )
    def test_ab_filter(self, parser, series11, raw, expected):
        """F5: AB filter normalized to S10 basis before map lookup."""
        parser._str_series11 = series11
        record = {"mode": 0.0 if not series11 else 3.0, "ab_filter": raw}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.ab_filter == expected

    @pytest.mark.parametrize(
        "series11, raw_hum, expected_bool",
        [
            (False, 0.0, False),  # S10 raw 0 = off
            (False, 1.0, True),  # S10 raw 1 = on
            (True, 1.0, False),  # S11 raw 1 → norm 0 = off
            (True, 2.0, True),  # S11 raw 2 → norm 1 = on
        ],
    )
    def test_humidity_enable(self, parser, series11, raw_hum, expected_bool):
        """F14: HumEnable normalized to S10 basis (0=off, 1=on)."""
        parser._str_series11 = series11
        record = {"mode": 0.0 if not series11 else 3.0, "humidity_enabled": raw_hum}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.humidity_enabled is expected_bool


# ---------------------------------------------------------------------------
# F6 — ASV and ASVAuto pressures
# ---------------------------------------------------------------------------


class TestASVPressures:
    """F6: ASV and ASVAuto read S.AV.* and S.AA.* signals."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    def test_s10_asv_mode7_reads_av_signals(self, parser):
        """S10 mode 7 (ASV) reads S.AV.EPAP/MinPS/MaxPS."""
        parser._str_series11 = False
        record = {
            "mode": 7.0,
            "av_epap": 8.0,
            "av_min_ps": 2.0,
            "av_max_ps": 10.0,
            "av_start_press": 8.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.ASV
        assert settings.epap == 8.0
        assert settings.other_settings.get("min_epap") == "8.0"
        assert settings.other_settings.get("max_epap") == "8.0"
        assert settings.other_settings.get("min_ps") == "2.0"
        assert settings.other_settings.get("max_ps") == "10.0"
        assert settings.ramp_start_pressure == 8.0

    def test_s10_asv_auto_mode8_reads_aa_signals(self, parser):
        """S10 mode 8 (ASVAuto) reads S.AA.MinEPAP/MaxEPAP/MinPS/MaxPS. (F15)"""
        parser._str_series11 = False
        record = {
            "mode": 8.0,
            "aa_min_epap": 5.0,
            "aa_max_epap": 10.0,
            "aa_min_ps": 2.0,
            "aa_max_ps": 12.0,
            "aa_start_press": 5.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.ASV_AUTO
        assert settings.epap == 5.0
        assert settings.other_settings.get("min_epap") == "5.0"
        assert settings.other_settings.get("max_epap") == "10.0"
        assert settings.other_settings.get("min_ps") == "2.0"
        assert settings.other_settings.get("max_ps") == "12.0"

    def test_s11_asv_auto_mode7_reads_aa_signals(self, parser):
        """S11 raw mode 7 → S10 mode 8 = ASV_AUTO; reads AA signals. (F15)"""
        parser._str_series11 = True
        record = {
            "mode": 7.0,  # S11 raw 7 = ASVAuto
            "aa_min_epap": 4.0,
            "aa_max_epap": 9.0,
            "aa_min_ps": 1.0,
            "aa_max_ps": 8.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.ASV_AUTO
        assert settings.other_settings.get("min_epap") == "4.0"
        assert settings.other_settings.get("max_epap") == "9.0"


# ---------------------------------------------------------------------------
# F7 — iVAPS
# ---------------------------------------------------------------------------


class TestIVAPS:
    """F7: S10 mode 9 = iVAPS with S.i.* signals."""

    @pytest.fixture
    def parser(self):
        p = ResmedEDFParser()
        p._str_series11 = False
        return p

    def test_s10_mode9_ivaps_fixed_epap(self, parser):
        """S10 mode 9 with fixed EPAP (EPAPAuto=0): EPAP stored as min=max."""
        record = {
            "mode": 9.0,
            "iv_epap": 8.0,
            "iv_epap_auto": 0.0,
            "iv_min_ps": 2.0,
            "iv_max_ps": 10.0,
            "iv_start_press": 8.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.IVAPS
        assert settings.epap == 8.0
        assert settings.other_settings.get("min_epap") == "8.0"
        assert settings.other_settings.get("max_epap") == "8.0"
        assert settings.other_settings.get("min_ps") == "2.0"
        assert settings.other_settings.get("max_ps") == "10.0"
        assert settings.other_settings.get("epap_auto") == "False"

    def test_s10_mode9_ivaps_auto_epap(self, parser):
        """S10 mode 9 with auto EPAP: min/max EPAP stored."""
        record = {
            "mode": 9.0,
            "iv_epap_auto": 1.0,  # auto on
            "iv_min_epap": 4.0,
            "iv_max_epap": 12.0,
            "iv_min_ps": 2.0,
            "iv_max_ps": 8.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.IVAPS
        assert settings.other_settings.get("min_epap") == "4.0"
        assert settings.other_settings.get("max_epap") == "12.0"
        assert settings.other_settings.get("epap_auto") == "True"

    def test_s10_mode9_ivaps_reads_shared_bilevel_timing(self, parser):
        """iVAPS reads rise/cycle/trigger/Ti from the bare S.* signals (OSCAR :2413-2441)."""
        record = {
            "mode": 9.0,
            "iv_epap": 6.0,
            "iv_epap_auto": 0.0,
            "s10_ti_max": 1.2,
            "s10_ti_min": 0.4,
            "s10_trigger": 2.0,
            "s10_cycle": 3.0,
            "s10_rise_enable": 1.0,
            "s10_rise_time": 300.0,
            "s10_easy_breathe": 1.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings == {
            "min_epap": "6.0",
            "max_epap": "6.0",
            "epap_auto": "False",
            "ti_max": "1.2",
            "ti_min": "0.4",
            "trigger": "Med",
            "cycle": "High",
            "rise_time": "300",
        }

    def test_s10_mode9_ivaps_rise_time_requires_rise_enable(self, parser):
        record = {"mode": 9.0, "s10_rise_enable": 0.0, "s10_rise_time": 300.0}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert "rise_time" not in settings.other_settings


# ---------------------------------------------------------------------------
# F8 — EPR type off-by-one for S10
# ---------------------------------------------------------------------------


class TestEPRTypeDecode:
    """F8: S10 EPRType raw+1; S11 raw unchanged (net 0 shift)."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    @pytest.mark.parametrize(
        "series11, raw, expected_mode",
        [
            # S10: raw+1 before map {0:Off, 1:Ramp Only, 2:Full Time}
            (False, 0.0, "Ramp Only"),  # 0+1=1
            (False, 1.0, "Full Time"),  # 1+1=2
            # S11: raw unchanged (net +1−1=0)
            (True, 0.0, "Off"),
            (True, 1.0, "Ramp Only"),
            (True, 2.0, "Full Time"),
        ],
    )
    def test_epr_type_decode(self, parser, series11, raw, expected_mode):
        """EPRType raw decodes correctly for both families."""
        parser._str_series11 = series11
        record = {
            "mode": 0.0 if not series11 else 3.0,
            "epr_type_raw": raw,
            "epr_level": 1.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.epr_mode == expected_mode


# ---------------------------------------------------------------------------
# F9 — SmartRamp / SmartStart family decode
# ---------------------------------------------------------------------------


class TestSmartRampSmartStart:
    """F9: SmartRamp is S10 raw=2 for both S.SmartStart and S.RampEnable. S11 normalizes -1."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    @pytest.mark.parametrize(
        "series11, smart_start_raw, expected_smart_start, expect_smart_ramp",
        [
            # S10: S.SmartStart 0=off, 1=on, 2=SmartRamp (also a form of enabled start)
            (False, 0.0, False, False),
            (False, 1.0, True, False),
            (
                False,
                2.0,
                True,
                True,
            ),  # SmartRamp: smart_start=True + other_settings["smart_ramp"]
            # S11: raw-1 → S10 basis
            (True, 1.0, False, False),  # S11 raw 1→norm 0=off
            (True, 2.0, True, False),  # S11 raw 2→norm 1=on
            (True, 3.0, True, True),  # S11 raw 3→norm 2=SmartRamp
        ],
    )
    def test_smart_start_encoding(
        self, parser, series11, smart_start_raw, expected_smart_start, expect_smart_ramp
    ):
        parser._str_series11 = series11
        record = {
            "mode": 0.0 if not series11 else 3.0,
            "smart_start": smart_start_raw,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.smart_start is expected_smart_start
        if expect_smart_ramp:
            assert settings.other_settings.get("smart_ramp") == "True"
        else:
            assert "smart_ramp" not in settings.other_settings

    @pytest.mark.parametrize(
        "series11, ramp_raw, expected_ramp_enabled, expect_smart_ramp",
        [
            # S10 S.RampEnable: 0=off, 1=on, 2=SmartRamp
            (False, 0.0, False, False),
            (False, 1.0, True, False),
            (
                False,
                2.0,
                False,
                True,
            ),  # SmartRamp: ramp_enabled=False, other_settings["smart_ramp"]
            # S11 S.RampEnable: raw-1
            (True, 1.0, False, False),  # S11 1→0=off
            (True, 2.0, True, False),  # S11 2→1=on
            (True, 3.0, False, True),  # S11 3→2=SmartRamp
        ],
    )
    def test_ramp_enable_encoding(
        self, parser, series11, ramp_raw, expected_ramp_enabled, expect_smart_ramp
    ):
        parser._str_series11 = series11
        record = {
            "mode": 0.0 if not series11 else 3.0,
            "ramp_enabled": ramp_raw,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.ramp_enabled is expected_ramp_enabled
        if expect_smart_ramp:
            assert settings.other_settings.get("smart_ramp") == "True"


# ---------------------------------------------------------------------------
# F10 — SmartStop, Tube, PtAccess
# ---------------------------------------------------------------------------


class TestSmartStopTubePtAccess:
    """F10: SmartStop/Tube/PtAccess stored in other_settings."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    def test_smart_stop_stored(self, parser):
        """S11 SmartStop raw=2→norm=1=True stored in other_settings."""
        parser._str_series11 = True
        record = {"mode": 3.0, "smart_stop_raw": 2.0}  # S11 raw 2→norm 1 = on
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("smart_stop") == "True"

    def test_smart_stop_off_s10(self, parser):
        """S10 SmartStop raw=0 → off."""
        parser._str_series11 = False
        record = {"mode": 0.0, "smart_stop_raw": 0.0}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("smart_stop") == "False"

    def test_tube_stored(self, parser):
        """Tube type raw value stored via TUBE_TYPE_MAP; unmapped codes pass through as int strings."""
        parser._str_series11 = False
        record = {"mode": 0.0, "tube_raw": 3.0}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("tube") == "3"

    def test_pt_access_stored(self, parser):
        """S11 PtAccess emits pt_view (normalized -1); pt_access key must be absent."""
        parser._str_series11 = True
        record = {"mode": 3.0, "pt_access_raw": 2.0}  # S11 raw 2→norm 1 = "Simple"
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("pt_view") == "Simple"
        assert "pt_access" not in settings.other_settings

    @pytest.mark.parametrize(
        "raw, expected_label",
        [
            (0.0, "Plus"),  # S10 raw 0 = Plus
            (1.0, "On"),  # S10 raw 1 = On
        ],
    )
    def test_s10_pt_access_stores_plus_on_labels(self, parser, raw, expected_label):
        """S10 PtAccess emits pt_access with PT_ACCESS_MAP labels; pt_view must be absent."""
        parser._str_series11 = False
        record = {"mode": 0.0, "pt_access_raw": raw}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("pt_access") == expected_label
        assert "pt_view" not in settings.other_settings

    @pytest.mark.parametrize(
        "raw, expected_label",
        [
            (15.0, "SlimLine"),
            (19.0, "Standard"),
            (3.0, "3"),  # unmapped raw passes through as str(int)
        ],
    )
    def test_tube_type_mapped(self, parser, raw, expected_label):
        """Tube type uses TUBE_TYPE_MAP; unmapped values pass through as strings."""
        parser._str_series11 = False
        record = {"mode": 0.0, "tube_raw": raw}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("tube") == expected_label


# ---------------------------------------------------------------------------
# F11 — S.AS.StartPress as APAP ramp start primary (OSCAR :1938)
# ---------------------------------------------------------------------------


class TestASStartPress:
    """F11: S.AS.StartPress is first-choice ramp start for APAP on S10."""

    @pytest.fixture
    def parser(self):
        p = ResmedEDFParser()
        p._str_series11 = False
        return p

    def test_as_start_press_used_when_present(self, parser):
        """as_start_press (S.AS.StartPress) is preferred over ramp_start_pressure."""
        record = {
            "mode": 1.0,
            "pressure_min": 4.0,
            "pressure_max": 20.0,
            "as_start_press": 5.0,
            "ramp_start_pressure": 4.0,  # secondary
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.ramp_start_pressure == 5.0

    def test_falls_back_to_ramp_start_pressure(self, parser):
        """Falls back to S.A.StartPress when S.AS.StartPress absent."""
        record = {
            "mode": 1.0,
            "pressure_min": 4.0,
            "pressure_max": 20.0,
            "ramp_start_pressure": 4.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.ramp_start_pressure == 4.0


# ---------------------------------------------------------------------------
# F15 — S10 mode 8 → ASV_AUTO (was ASV)
# ---------------------------------------------------------------------------


class TestASVAuto:
    """F15: S10 mode 8 maps to ASV_AUTO, not ASV."""

    def test_s10_mode8_is_asv_auto(self):
        parser = ResmedEDFParser()
        parser._str_series11 = False
        record = {"mode": 8.0, "aa_min_epap": 5.0, "aa_max_epap": 9.0}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.ASV_AUTO

    def test_s11_mode7_is_asv_auto(self):
        parser = ResmedEDFParser()
        parser._str_series11 = True
        record = {"mode": 7.0, "aa_min_epap": 5.0, "aa_max_epap": 9.0}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.ASV_AUTO


# ---------------------------------------------------------------------------
# F17 — Response (S.AS.Comfort)
# ---------------------------------------------------------------------------


class TestResponseField:
    """F17: S.AS.Comfort stored as 'response' in other_settings with S11 norm."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    def test_s10_response_stored(self, parser):
        """S10 S.AS.Comfort raw value stored directly as response."""
        parser._str_series11 = False
        record = {"mode": 1.0, "comfort_raw": 2.0}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("response") == "2"

    def test_s11_response_normalized(self, parser):
        """S11 S.AS.Comfort raw normalized -1 (OSCAR :2180-2183)."""
        parser._str_series11 = True
        record = {"mode": 1.0, "comfort_raw": 3.0}  # S11 raw 3→norm 2 → "2" (unmapped)
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("response") == "2"

    @pytest.mark.parametrize(
        "raw, expected_label",
        [
            (0.0, "Standard"),  # RESPONSE_MAP[0]
            (1.0, "Soft"),  # RESPONSE_MAP[1]
            (2.0, "2"),  # unmapped → passthrough
        ],
    )
    def test_response_known_codes_mapped(self, parser, raw, expected_label):
        """RESPONSE_MAP labels codes 0 and 1; unmapped codes pass through as strings."""
        parser._str_series11 = False
        record = {"mode": 1.0, "comfort_raw": raw}
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("response") == expected_label


# ---------------------------------------------------------------------------
# F20 — EPR enable gating
# ---------------------------------------------------------------------------


class TestEPREnableGating:
    """F20: EPREnable/ClinEnable gate zeroes epr_mode/epr_level when disabled."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    def test_both_enabled_epr_passes_through(self, parser):
        """EPREnable=on and ClinEnable=on: EPR values pass through."""
        parser._str_series11 = False
        record = {
            "mode": 0.0,
            "epr_level": 2.0,
            "epr_type_raw": 1.0,  # S10 raw+1 → "Full Time"
            "epr_enable_raw": 1.0,  # S10: 1=on
            "epr_clin_enable_raw": 1.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.epr_level == 2
        assert settings.epr_mode == "Full Time"

    def test_epr_disabled_zeroes_epr(self, parser):
        """EPREnable=off: epr_mode=Off and epr_level=0 regardless of raw values."""
        parser._str_series11 = False
        record = {
            "mode": 0.0,
            "epr_level": 2.0,
            "epr_type_raw": 1.0,
            "epr_enable_raw": 0.0,  # disabled
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.epr_level == 0
        assert settings.epr_mode == "Off"

    def test_clin_disable_zeroes_epr(self, parser):
        """ClinEnable=off with EPREnable=on: zeroed (OSCAR :2207-2214)."""
        parser._str_series11 = False
        record = {
            "mode": 0.0,
            "epr_level": 3.0,
            "epr_type_raw": 1.0,
            "epr_enable_raw": 1.0,
            "epr_clin_enable_raw": 0.0,  # clinician disabled
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.epr_level == 0
        assert settings.epr_mode == "Off"

    def test_s11_epr_gating_with_norm(self, parser):
        """S11 EPREnable raw=2→norm=1=on; ClinEnable raw=2→norm=1=on: passes."""
        parser._str_series11 = True
        record = {
            "mode": 3.0,  # CPAP
            "epr_level": 2.0,
            "epr_type_raw": 2.0,  # S11 raw → "Full Time"
            "epr_enable_raw": 2.0,  # S11 raw 2→norm 1=on
            "epr_clin_enable_raw": 2.0,  # same
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.epr_level == 2
        assert settings.epr_mode == "Full Time"


# ---------------------------------------------------------------------------
# Mask type mapping (regression contract — unchanged from original)
# ---------------------------------------------------------------------------


class TestMaskTypeMapping:
    """Tests for generation-aware mask code decoding."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (0, "Pillows"),
            (1, "Full Face"),
            (2, "Nasal"),
            (3, "Unknown"),
            (4, "Unknown"),
            (5, "Unknown"),
        ],
    )
    def test_ten_series_mask_codes(self, parser, raw, expected):
        """S9/10-series codes 0–2 map to mask names; anything else → Unknown."""
        parser._str_series11 = False
        values = {"mode": 1.0, "mask_type": float(raw)}
        settings = parser._convert_str_to_therapy_settings(values)
        assert settings is not None
        assert settings.mask_type == expected

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (2, "Pillows"),
            (3, "Full Face"),
            (4, "Nasal"),
            (0, "Unknown"),
            (1, "Unknown"),
            (5, "Unknown"),
        ],
    )
    def test_eleven_series_mask_codes(self, parser, raw, expected):
        """11-series raw codes 2–4 are shifted down by 2 before lookup; others → Unknown."""
        parser._str_series11 = True
        values = {"mode": 1.0, "mask_type": float(raw)}
        settings = parser._convert_str_to_therapy_settings(values)
        assert settings is not None
        assert settings.mask_type == expected


# ---------------------------------------------------------------------------
# F18 — S9 Identification.tgt parsing
# ---------------------------------------------------------------------------


class TestIdentificationTgt:
    """F18: _detect_series11 and serial extraction fall back to .tgt for S9 devices."""

    def test_detect_series11_returns_false_for_s9_tgt(self, tmp_path):
        """S9 device with .tgt has ProductCode < 39000 → series11=False."""
        tgt = tmp_path / "Identification.tgt"
        tgt.write_text(
            "SerialNumber=12345678\nProductCode=30001\nProductName=S9 AutoSet\n"
        )
        parser = ResmedEDFParser()
        assert parser._detect_series11(tmp_path) is False

    def test_detect_series11_returns_false_when_only_tgt_present(self, tmp_path):
        """No .json present: falls back to .tgt; S9 ProductCode → False."""
        tgt = tmp_path / "Identification.tgt"
        tgt.write_text("ProductCode=36003\n")
        parser = ResmedEDFParser()
        assert parser._detect_series11(tmp_path) is False

    def test_detect_series11_returns_true_for_high_product_code_tgt(self, tmp_path):
        """A .tgt file with ProductCode >= 39000 → series11=True (future-proofing)."""
        tgt = tmp_path / "Identification.tgt"
        tgt.write_text("ProductCode=39485\n")
        parser = ResmedEDFParser()
        assert parser._detect_series11(tmp_path) is True

    def test_serial_extracted_from_tgt(self, tmp_path):
        """Serial number parsed from Identification.tgt when .json absent."""
        tgt = tmp_path / "Identification.tgt"
        tgt.write_text(
            "ProductName=S9 AutoSet\nSerialNumber=98765432\nProductCode=30001\n"
        )
        parser = ResmedEDFParser()
        serial = parser._extract_serial_from_identification(tmp_path)
        assert serial == "98765432"

    def test_json_takes_precedence_over_tgt(self, tmp_path):
        """When both .json and .tgt exist, .json wins (OSCAR :2470-2472)."""
        import json

        (tmp_path / "Identification.json").write_text(
            json.dumps(
                {
                    "FlowGenerator": {
                        "IdentificationProfiles": {
                            "Product": {
                                "SerialNumber": "jsonserial",
                                "ProductCode": 39485,
                            }
                        }
                    }
                }
            )
        )
        (tmp_path / "Identification.tgt").write_text("SerialNumber=tgtserial\n")
        parser = ResmedEDFParser()
        serial = parser._extract_serial_from_identification(tmp_path)
        assert serial == "jsonserial"

    def test_tgt_missing_returns_false(self, tmp_path):
        """No .json and no .tgt → _detect_series11 returns False."""
        parser = ResmedEDFParser()
        assert parser._detect_series11(tmp_path) is False


# ---------------------------------------------------------------------------
# _is_eleven_series static (unchanged)
# ---------------------------------------------------------------------------


class TestIsElevenSeries:
    """Tests for _is_eleven_series model-string detection."""

    @pytest.mark.parametrize(
        "model, expected",
        [
            ("AirSense11AutoSet", True),
            ("AirCurve 11 VAuto", True),
            ("AirSense 10 AutoSet", False),
            ("AirCurve 10 VAuto", False),
            ("S9 AutoSet", False),
            ("Unknown", False),
            ("", False),
        ],
    )
    def test_model_detection(self, model, expected):
        assert ResmedEDFParser._is_eleven_series(model) is expected


class TestDetectSeries11:
    """Tests for _detect_series11 ProductCode-based family detection."""

    def _make_id_json(self, tmp_path: Path, product_code: int | float | str) -> Path:
        """Write a minimal Identification.json and return the directory path."""
        data = {
            "FlowGenerator": {
                "IdentificationProfiles": {"Product": {"ProductCode": product_code}}
            }
        }
        (tmp_path / "Identification.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        return tmp_path

    def test_product_code_39000_is_series11(self, tmp_path):
        root = self._make_id_json(tmp_path, 39000)
        assert ResmedEDFParser()._detect_series11(root) is True

    def test_product_code_38999_is_not_series11(self, tmp_path):
        root = self._make_id_json(tmp_path, 38999)
        assert ResmedEDFParser()._detect_series11(root) is False

    def test_product_code_float_string_39000_0_is_series11(self, tmp_path):
        """ProductCode stored as the string "39000.0" must not raise ValueError."""
        root = self._make_id_json(tmp_path, "39000.0")
        assert ResmedEDFParser()._detect_series11(root) is True

    def test_product_code_float_39000_0_is_series11(self, tmp_path):
        """ProductCode stored as a JSON float (39000.0) must be handled correctly."""
        root = self._make_id_json(tmp_path, 39000.0)
        assert ResmedEDFParser()._detect_series11(root) is True

    def test_missing_product_code_key_returns_false(self, tmp_path):
        data = {"FlowGenerator": {"IdentificationProfiles": {"Product": {}}}}
        (tmp_path / "Identification.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        assert ResmedEDFParser()._detect_series11(tmp_path) is False

    def test_missing_identification_json_returns_false(self, tmp_path):
        assert ResmedEDFParser()._detect_series11(tmp_path) is False

    def test_malformed_json_returns_false(self, tmp_path):
        (tmp_path / "Identification.json").write_text(
            "not valid json {{{", encoding="utf-8"
        )
        assert ResmedEDFParser()._detect_series11(tmp_path) is False


class TestSeries10ModesBipapST:
    """Tests for Series 10 modes 4 and 5 — BIPAP_ST deliberate deviation from OSCAR."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    @pytest.mark.parametrize("mode_int", [4, 5])
    def test_series10_modes_4_5_map_to_bipap_st(self, parser, mode_int):
        """S10 modes 4 and 5 are S/T variants; mapped to BIPAP_ST (OSCAR maps them to BILEVEL_FIXED)."""
        parser._str_series11 = False
        record = {
            "mode": float(mode_int),
            "s_ipap": 14.0,
            "s_epap": 8.0,
            "s_start_press": 8.0,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.mode == TherapyMode.BIPAP_ST


class TestNanModeValue:
    """Test that a NaN mode value is discarded gracefully."""

    @pytest.fixture
    def parser(self):
        return ResmedEDFParser()

    def test_nan_mode_returns_none_without_raising(self, parser):
        """NaN mode value is treated like a missing mode — returns None, does not raise."""
        record = {"mode": float("nan"), "pressure_fixed": 10.0}
        result = parser._convert_str_to_therapy_settings(record)
        assert result is None


# ---------------------------------------------------------------------------
# Trigger/Cycle label coverage
# ---------------------------------------------------------------------------


class TestTriggerCycleLabels:
    """All five TRIGGER_MAP / CYCLE_MAP labels are produced correctly."""

    @pytest.fixture
    def parser(self):
        p = ResmedEDFParser()
        p._str_series11 = True
        return p

    @pytest.mark.parametrize(
        "s10_code, expected_label",
        [
            (0, "Very Low"),
            (1, "Low"),
            (2, "Med"),
            (3, "High"),
            (4, "Very High"),
        ],
    )
    def test_trigger_cycle_all_five_levels(self, parser, s10_code, expected_label):
        """S11 VAuto: raw = s10_code + 1 (S11 offset); _norm subtracts 1 back.

        Both trigger and cycle use the same five-level map, so one parametrized
        test exercises both paths simultaneously.
        """
        s11_raw = float(s10_code + 1)
        record = {
            "mode": 8.0,  # S11 raw 8 = VAuto
            "va_min_epap": 5.0,
            "va_max_ipap": 14.0,
            "va_ps": 3.0,
            "va_trigger": s11_raw,
            "va_cycle": s11_raw,
        }
        settings = parser._convert_str_to_therapy_settings(record)
        assert settings is not None
        assert settings.other_settings.get("trigger") == expected_label
        assert settings.other_settings.get("cycle") == expected_label
