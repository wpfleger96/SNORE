"""Unit tests for ResMed EDF waveform imports (parser v1.1.0).

Covers the signals added in the waveform-import PR:
- PLD: FlowLim, Snore, RespRate, TidVol (L→mL), IERatio, Ti (VAuto-only)
- BRP: high-rate mask pressure (Press.40ms → PRESSURE_HR)
- TCV: TrigCycEvt verbatim codes (VAuto-only)
- Multi-segment merge: stats combined from segment stats, not raw blob

Tests against the real APAP fixture where the signals exist and use
synthetic pyedflib EDFs for sentinel-exclusion and TCV coverage.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from snore.parsers.resmed_edf import ResmedEDFParser
from snore.parsers.unified import DeviceInfo, UnifiedSession, WaveformData, WaveformType

FIXTURE_ROOT = Path(__file__).parent.parent / "fixtures" / "device_data" / "resmed"
PLD_FIXTURE = FIXTURE_ROOT / "DATALOG" / "2024" / "20240621_013454_PLD.edf"
BRP_FIXTURE = FIXTURE_ROOT / "DATALOG" / "2024" / "20240621_013454_BRP.edf"

_DEVICE = DeviceInfo(
    manufacturer="ResMed", model="AirSense 10 AutoSet", serial_number="FIXTURE"
)


def _make_edf(
    tmp_path: Path,
    filename: str,
    signals: list[dict],
    n_records: int = 1,
    record_duration: int = 2,
) -> Path:
    """Create a minimal EDF file via pyedflib.

    Each entry in ``signals`` must have keys: label, dimension, data (np.ndarray),
    physical_min, physical_max, digital_min, digital_max.
    """
    import pyedflib

    edf_path = tmp_path / filename
    n_channels = len(signals)
    with pyedflib.EdfWriter(
        str(edf_path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS
    ) as f:
        f.setStartdatetime(datetime(2024, 6, 21, 1, 34, 54))
        headers = []
        for sig in signals:
            n_samples = len(sig["data"])
            headers.append(
                {
                    "label": sig["label"],
                    "dimension": sig["dimension"],
                    "sample_frequency": n_samples // (n_records * record_duration),
                    "physical_max": sig["physical_max"],
                    "physical_min": sig["physical_min"],
                    "digital_max": sig.get("digital_max", 32767),
                    "digital_min": sig.get("digital_min", -32768),
                }
            )
        f.setSignalHeaders(headers)
        f.setDatarecordDuration(record_duration)
        for sig in signals:
            f.writePhysicalSamples(sig["data"].astype(np.float64))
    return edf_path


# ─────────────────────────────────────────────────────────────────────────────
# PLD new signals — tested against real fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fixture_session():
    """Parse the real APAP fixture once per test module."""
    parser = ResmedEDFParser()
    sessions = list(parser.parse_sessions(FIXTURE_ROOT))
    assert len(sessions) == 1
    return sessions[0]


class TestPLDNewSignals:
    """New PLD waveforms imported from the real APAP fixture."""

    def test_flow_limitation_present(self, fixture_session):
        session = fixture_session
        assert WaveformType.FLOW_LIMITATION in session.waveforms, (
            "No fl waveform parsed from PLD"
        )
        wf = session.waveforms[WaveformType.FLOW_LIMITATION]
        assert len(wf.values) > 0
        # EDF physical dimension for this signal is empty; keep it as-is.
        assert wf.unit == ""
        # Fixture has FlowLim values of 0.0–0.08, well within valid_range (0,1).
        assert 0.0 <= wf.min_value <= wf.max_value <= 1.0

    def test_snore_present(self, fixture_session):
        session = fixture_session
        assert WaveformType.SNORE in session.waveforms, (
            "No snore waveform parsed from PLD"
        )
        wf = session.waveforms[WaveformType.SNORE]
        assert len(wf.values) > 0
        assert wf.unit == ""
        assert 0.0 <= wf.min_value <= wf.max_value <= 5.0

    def test_respiratory_rate_present(self, fixture_session):
        session = fixture_session
        assert WaveformType.RESPIRATORY_RATE in session.waveforms, (
            "No rr waveform parsed from PLD"
        )
        wf = session.waveforms[WaveformType.RESPIRATORY_RATE]
        assert wf.unit == "bpm"
        assert len(wf.values) > 0
        assert 0.0 < wf.min_value <= wf.max_value <= 90.0

    def test_tidal_volume_present_and_in_ml(self, fixture_session):
        session = fixture_session
        assert WaveformType.TIDAL_VOLUME in session.waveforms, (
            "No tv waveform parsed from PLD"
        )
        wf = session.waveforms[WaveformType.TIDAL_VOLUME]
        assert wf.unit == "mL", f"Expected mL, got {wf.unit!r}"
        assert len(wf.values) > 0
        # Fixture raw: 0.22–1.52 L → 220–1520 mL after ×1000 conversion.
        assert wf.min_value >= 220.0, f"TV min {wf.min_value} too low"
        assert wf.max_value <= 1520.0, f"TV max {wf.max_value} too high"

    def test_vauto_only_signals_absent_on_apap_fixture(self, fixture_session):
        """IERatio and Ti are absent from the APAP fixture — must be skipped silently."""
        # The session must have parsed successfully (no exception) and simply
        # not contain the VAuto-only waveform types.
        assert WaveformType.IE_RATIO not in fixture_session.waveforms
        assert WaveformType.TI not in fixture_session.waveforms


# ─────────────────────────────────────────────────────────────────────────────
# TidVol L→mL conversion detail
# ─────────────────────────────────────────────────────────────────────────────


class TestTidVolConversion:
    """TidVol values in the unified model are always in mL, not L."""

    def test_tidal_volume_values_are_1000x_raw(self):
        from snore.parsers.formats.edf import EDFReader

        parser = ResmedEDFParser()

        # Read raw L values directly from the fixture.
        with EDFReader(PLD_FIXTURE) as edf:
            raw_l_values = None
            for name in edf.list_signal_labels():
                if "TidVol" in name:
                    data, _ = edf.read_signal(name)
                    raw_l_values = data
                    break

        assert raw_l_values is not None, "No TidVol signal in PLD fixture"

        sessions = list(parser.parse_sessions(FIXTURE_ROOT))
        session = sessions[0]
        assert WaveformType.TIDAL_VOLUME in session.waveforms

        wf = session.waveforms[WaveformType.TIDAL_VOLUME]
        np.testing.assert_allclose(
            np.asarray(wf.values, dtype=np.float64),
            raw_l_values.astype(np.float64) * 1000.0,
            rtol=1e-5,
            err_msg="TidVol values should be raw_L × 1000",
        )


# ─────────────────────────────────────────────────────────────────────────────
# valid_range sentinel exclusion
# ─────────────────────────────────────────────────────────────────────────────


class TestValidRangeSentinelExclusion:
    """Out-of-range samples are stored in the blob but excluded from stats.

    We test _read_waveform directly by mocking EDFReader so there is no
    EDF file I/O — and therefore no pyedflib padding artefacts.
    """

    def _make_mock_edf(
        self, data: np.ndarray, unit: str, sample_rate: float
    ) -> MagicMock:
        """Return a minimal EDFReader mock for a single signal."""
        info = MagicMock()
        info.physical_dimension = unit

        mock_edf = MagicMock()
        mock_edf.read_signal.return_value = (data, info)
        mock_edf.get_sample_rate.return_value = sample_rate
        mock_edf.get_timestamps.return_value = (
            np.arange(len(data), dtype=np.float64) / sample_rate
        )
        return mock_edf

    def test_out_of_range_excluded_from_stats_but_stored_in_blob(self):
        """Sentinel (-1.0) must appear in wf.values but be excluded from stats.

        valid_range=(0.0, 1.0): samples outside that range remain in the
        stored blob (wf.values) but do NOT influence min/max/mean.
        """
        valid = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        sentinel = np.array([-1.0])
        data = np.concatenate([valid, sentinel])

        parser = ResmedEDFParser()
        mock_edf = self._make_mock_edf(data, unit="", sample_rate=0.5)

        result = parser._read_waveform(
            mock_edf,
            "FlowLim.2s",
            WaveformType.FLOW_LIMITATION,
            "",
            valid_range=(0.0, 1.0),
        )

        assert result is not None, "_read_waveform returned None; no valid samples?"
        wf, valid_data = result

        # Full blob retains the sentinel.
        assert -1.0 in wf.values, "Sentinel must be retained in the stored blob"
        assert len(wf.values) == len(data)

        # Stats are over valid samples only.
        assert wf.min_value >= 0.0, f"min {wf.min_value} must exclude sentinel"
        assert wf.max_value <= 1.0, f"max {wf.max_value} must be within valid_range"
        np.testing.assert_allclose(wf.mean_value, float(np.mean(valid)), rtol=1e-5)

    def test_all_out_of_range_returns_none(self):
        """If every sample is outside valid_range, _read_waveform must return None."""
        data = np.array([-1.0, -2.0, 99.0])
        parser = ResmedEDFParser()
        mock_edf = self._make_mock_edf(data, unit="", sample_rate=0.5)

        result = parser._read_waveform(
            mock_edf,
            "FlowLim.2s",
            WaveformType.FLOW_LIMITATION,
            "",
            valid_range=(0.0, 1.0),
        )
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# BRP hi-res pressure
# ─────────────────────────────────────────────────────────────────────────────


class TestBRPPressureHR:
    """High-rate mask pressure is extracted from the BRP file."""

    def test_pressure_hr_present(self, fixture_session):
        assert WaveformType.PRESSURE_HR in fixture_session.waveforms, (
            "No pressure_hr waveform parsed from BRP"
        )
        wf = fixture_session.waveforms[WaveformType.PRESSURE_HR]
        assert wf.unit == "cmH2O", f"Expected cmH2O, got {wf.unit!r}"
        assert len(wf.values) > 0

    def test_pressure_hr_sample_rate_is_25hz(self, fixture_session):
        wf = fixture_session.waveforms[WaveformType.PRESSURE_HR]
        # BRP is 25 Hz; sample_rate should be ~25.
        assert 20.0 <= wf.sample_rate <= 30.0, f"Expected ~25 Hz, got {wf.sample_rate}"

    def test_pressure_hr_values_in_plausible_range(self, fixture_session):
        wf = fixture_session.waveforms[WaveformType.PRESSURE_HR]
        sample = np.asarray(wf.values[:100])
        assert np.all(sample >= 0.0), "Pressure values must be non-negative"
        assert np.all(sample <= 40.0), "Pressure values must be ≤ 40 cmH2O"

    def test_pressure_hr_matches_raw_fixture(self, fixture_session):
        from snore.parsers.formats.edf import EDFReader

        with EDFReader(BRP_FIXTURE) as edf:
            raw_data = None
            for name in edf.list_signal_labels():
                if "Press" in name:
                    raw_data, _ = edf.read_signal(name)
                    break

        assert raw_data is not None
        wf = fixture_session.waveforms[WaveformType.PRESSURE_HR]
        np.testing.assert_allclose(
            np.asarray(wf.values[:100], dtype=np.float64),
            raw_data[:100].astype(np.float64),
            rtol=1e-5,
            err_msg="pressure_hr values should match raw EDF data",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TCV parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestTCVParsing:
    """TCV trigger/cycle event waveform (VAuto-only)."""

    def test_tcv_absent_yields_no_trigger_cycle_waveform(self, fixture_session):
        """Real APAP fixture has no TCV file — trigger_cycle must be absent."""
        assert WaveformType.TRIGGER_CYCLE not in fixture_session.waveforms

    def test_tcv_present_yields_trigger_cycle_verbatim(self, tmp_path):
        """Synthetic TCV with known codes → trigger_cycle waveform with raw values."""

        # Integer codes 0–16 repeated; some at the boundary.
        codes = np.array([0.0, 1.0, 5.0, 10.0, 16.0, 0.0, 3.0, 8.0], dtype=np.float64)
        signals = [
            {
                "label": "TrigCycEvt.40ms",
                "dimension": "",
                "data": codes,
                "physical_min": 0.0,
                "physical_max": 16.0,
                "digital_min": 0,
                "digital_max": 16,
            }
        ]
        tcv_path = _make_edf(
            tmp_path, "20240621_013454_TCV.edf", signals, n_records=1, record_duration=8
        )

        parser = ResmedEDFParser()
        session = UnifiedSession(
            device_session_id="20240621_013454",
            device_info=_DEVICE,
            start_time=datetime(2024, 6, 21, 1, 34, 54),
            end_time=datetime(2024, 6, 21, 5, 30, 0),
            import_source="resmed_edf",
            parser_version="1.1.0",
        )
        parser._parse_tcv(tcv_path, session)

        assert WaveformType.TRIGGER_CYCLE in session.waveforms, (
            "No trigger_cycle waveform after parsing synthetic TCV"
        )
        wf = session.waveforms[WaveformType.TRIGGER_CYCLE]
        assert len(wf.values) > 0
        assert wf.unit == ""
        assert 0.0 <= wf.min_value <= wf.max_value <= 16.0

    def test_tcv_out_of_range_sentinel_excluded_from_stats(self, tmp_path):
        """A code outside 0–16 must be stored but excluded from stats."""
        # 5 valid codes + one sentinel (99.0 — out of range).
        data = np.array([1.0, 5.0, 10.0, 3.0, 8.0, 99.0], dtype=np.float64)
        signals = [
            {
                "label": "TrigCycEvt.40ms",
                "dimension": "",
                "data": data,
                "physical_min": 0.0,
                "physical_max": 100.0,
                "digital_min": 0,
                "digital_max": 100,
            }
        ]
        tcv_path = _make_edf(
            tmp_path,
            "20240621_013454_TCV.edf",
            signals,
            n_records=1,
            record_duration=12,
        )

        parser = ResmedEDFParser()
        session = UnifiedSession(
            device_session_id="20240621_013454",
            device_info=_DEVICE,
            start_time=datetime(2024, 6, 21, 1, 34, 54),
            end_time=datetime(2024, 6, 21, 5, 30, 0),
            import_source="resmed_edf",
            parser_version="1.1.0",
        )
        parser._parse_tcv(tcv_path, session)

        wf = session.waveforms[WaveformType.TRIGGER_CYCLE]
        assert 99.0 in wf.values, "Sentinel must be in the blob"
        assert wf.max_value <= 16.0, "max_value must exclude the sentinel"

    def test_tcv_zero_record_file_is_noop(self, tmp_path):
        """A zero-record TCV file (device powered but unused) → no waveform added.

        get_edf_record_count is imported inside _parse_tcv via a local
        'from .formats.edf import …', so we patch the function at its
        definition site in the formats.edf module.
        """
        from unittest.mock import patch

        import pyedflib

        # Any valid EDF file; content doesn't matter since record_count is mocked.
        tcv_path = tmp_path / "20240621_013454_TCV.edf"
        with pyedflib.EdfWriter(
            str(tcv_path), 1, file_type=pyedflib.FILETYPE_EDFPLUS
        ) as f:
            f.setStartdatetime(datetime(2024, 6, 21, 1, 34, 54))
            f.setSignalHeaders(
                [
                    {
                        "label": "TrigCycEvt.40ms",
                        "dimension": "",
                        "sample_frequency": 25,
                        "physical_max": 16.0,
                        "physical_min": 0.0,
                        "digital_max": 16,
                        "digital_min": 0,
                    }
                ]
            )

        parser = ResmedEDFParser()
        session = UnifiedSession(
            device_session_id="20240621_013454",
            device_info=_DEVICE,
            start_time=datetime(2024, 6, 21, 1, 34, 54),
            end_time=datetime(2024, 6, 21, 5, 30, 0),
            import_source="resmed_edf",
            parser_version="1.1.0",
        )
        # Patch at the definition site so the local import inside _parse_tcv
        # sees the mock.
        with patch("snore.parsers.formats.edf.get_edf_record_count", return_value=0):
            parser._parse_tcv(tcv_path, session)

        assert WaveformType.TRIGGER_CYCLE not in session.waveforms


# ─────────────────────────────────────────────────────────────────────────────
# Parser version
# ─────────────────────────────────────────────────────────────────────────────


class TestParserVersion:
    def test_parser_version_is_1_2_1(self):
        parser = ResmedEDFParser()
        assert parser.get_metadata().parser_version == "1.2.1"

    def test_session_carries_version_1_2_1(self):
        parser = ResmedEDFParser()
        session = list(parser.parse_sessions(FIXTURE_ROOT))[0]
        assert session.parser_version == "1.2.1"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-segment merge: stats must not be recomputed from raw blob
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiSegmentMergeStats:
    """Waveform stats in merged sessions must come from per-segment stats.

    _read_waveform intentionally retains out-of-range sentinels in wf.values
    (the blob) while computing min/max/mean only over in-range samples. The
    merge path must honour this contract — combining pre-computed stats rather
    than recomputing from the raw blob — otherwise a single sentinel in any
    second segment corrupts the merged stat for the whole night.
    """

    def _make_segment(
        self,
        session_id: str,
        start: datetime,
        end: datetime,
        fl_values: np.ndarray,
        fl_min: float,
        fl_max: float,
        fl_mean: float,
    ) -> UnifiedSession:
        """Return a UnifiedSession with one FlowLimitation waveform."""
        session = UnifiedSession(
            device_session_id=session_id,
            device_info=_DEVICE,
            start_time=start,
            end_time=end,
            import_source="resmed_edf",
            parser_version="1.1.0",
        )
        n = len(fl_values)
        session.add_waveform(
            WaveformData(
                waveform_type=WaveformType.FLOW_LIMITATION,
                sample_rate=0.5,
                unit="",
                values=fl_values.astype(np.float32),
                timestamps=np.arange(n, dtype=np.float32) * 2.0,
                min_value=fl_min,
                max_value=fl_max,
                mean_value=fl_mean,
            )
        )
        return session

    def test_sentinel_in_second_segment_does_not_corrupt_merged_min(self):
        """One -0.01 sentinel in segment 2 must not pull merged min_value below 0.

        Validates the fix in the multi-segment merge path: stats are combined
        from segment-level stats (which correctly exclude sentinels), not
        recomputed from the raw concatenated blob.
        """
        t0 = datetime(2026, 8, 9, 22, 0, 0)
        t1 = t0 + timedelta(hours=4)
        t2 = t1 + timedelta(minutes=5)  # mask-off gap
        t3 = t2 + timedelta(hours=2)

        # Segment 1: three valid FlowLim samples in [0, 1].
        seg1_values = np.array([0.0, 0.3, 0.5])
        seg1 = self._make_segment(
            "20260809_220000",
            t0,
            t1,
            fl_values=seg1_values,
            fl_min=0.0,
            fl_max=0.5,
            fl_mean=float(np.mean(seg1_values)),
        )

        # Segment 2: one digital -1 sentinel (-0.01) plus valid samples.
        # _read_waveform stores the sentinel in the blob but excludes it
        # from stats, so min=0.1, max=0.2, mean=0.15.
        seg2_values = np.array([-0.01, 0.1, 0.2])
        seg2 = self._make_segment(
            "20260810_020500",
            t2,
            t3,
            fl_values=seg2_values,
            fl_min=0.1,
            fl_max=0.2,
            fl_mean=0.15,
        )

        dummy_segments = {
            "20260809_220000": {"BRP": Path("/fake/seg1_BRP.edf")},
            "20260810_020500": {"BRP": Path("/fake/seg2_BRP.edf")},
        }

        parser = ResmedEDFParser()
        with patch.object(parser, "_parse_session_group", side_effect=[seg1, seg2]):
            merged = parser._parse_night_session(
                "20260809", min(dummy_segments), dummy_segments, _DEVICE, Path("/fake")
            )

        assert merged is not None
        assert WaveformType.FLOW_LIMITATION in merged.waveforms

        fl = merged.waveforms[WaveformType.FLOW_LIMITATION]

        # Sentinel is retained in the concatenated blob.
        assert np.any(np.asarray(fl.values) < 0.0), (
            "Sentinel (-0.01) should be retained in the merged values blob"
        )

        # Stats reflect only valid samples from each segment's pre-computed stats.
        assert fl.min_value >= 0.0, (
            f"min_value {fl.min_value} must not include the -0.01 sentinel"
        )
        assert fl.max_value <= 1.0
        assert fl.max_value == pytest.approx(0.5, rel=1e-3)

        # Sample-count-weighted mean: (mean1 * n1 + mean2 * n2) / (n1 + n2)
        # = (0.2667 * 3 + 0.15 * 3) / 6 ≈ 0.2083
        n1, n2 = len(seg1_values), len(seg2_values)
        expected_mean = (float(np.mean(seg1_values)) * n1 + 0.15 * n2) / (n1 + n2)
        assert fl.mean_value == pytest.approx(expected_mean, rel=1e-3)

    def test_waveform_first_seen_in_later_segment_is_offset_to_that_segment(self):
        """A channel absent from segment 1 keeps segment 2's place on the timeline.

        Typical case: the oximeter is not reading during segment 1, so only
        segment 2 has an SpO2 waveform; its samples must start at segment 2's
        offset, not at the night's start.
        """
        t0 = datetime(2026, 8, 9, 22, 0, 0)
        t1 = t0 + timedelta(hours=4)
        t2 = t1 + timedelta(minutes=5)
        t3 = t2 + timedelta(hours=2)
        seg1 = self._make_segment(
            "20260809_220000", t0, t1, np.array([0.1, 0.2]), 0.1, 0.2, 0.15
        )
        seg2 = self._make_segment(
            "20260810_020500", t2, t3, np.array([0.1, 0.2]), 0.1, 0.2, 0.15
        )
        seg2.add_waveform(
            WaveformData(
                waveform_type=WaveformType.SPO2,
                sample_rate=1.0,
                unit="%",
                values=np.array([96.0, 95.0, 94.0], dtype=np.float32),
                timestamps=np.array([0.0, 1.0, 2.0], dtype=np.float32),
            )
        )
        segments = {
            "20260809_220000": {"BRP": Path("/fake/seg1_BRP.edf")},
            "20260810_020500": {"BRP": Path("/fake/seg2_BRP.edf")},
        }

        parser = ResmedEDFParser()
        with patch.object(parser, "_parse_session_group", side_effect=[seg1, seg2]):
            merged = parser._parse_night_session(
                "20260809", min(segments), segments, _DEVICE, Path("/fake")
            )

        assert merged is not None
        offset = (t2 - t0).total_seconds()
        np.testing.assert_allclose(
            merged.waveforms[WaveformType.SPO2].timestamps,
            [offset, offset + 1.0, offset + 2.0],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unit normalization for Ti ("s" → "seconds")
# ─────────────────────────────────────────────────────────────────────────────


class TestTiUnitNormalization:
    """EDF physical_dimension "s" must be normalized to "seconds"."""

    def _make_mock_edf(
        self, data: np.ndarray, unit: str, sample_rate: float
    ) -> MagicMock:
        info = MagicMock()
        info.physical_dimension = unit
        mock_edf = MagicMock()
        mock_edf.read_signal.return_value = (data, info)
        mock_edf.get_sample_rate.return_value = sample_rate
        mock_edf.get_timestamps.return_value = (
            np.arange(len(data), dtype=np.float64) / sample_rate
        )
        return mock_edf

    def test_si_abbreviation_s_normalized_to_seconds(self):
        """physical_dimension "s" is normalized to "seconds" before storing."""
        data = np.array([1.2, 1.4, 1.3, 1.5], dtype=np.float64)
        parser = ResmedEDFParser()
        mock_edf = self._make_mock_edf(data, unit="s", sample_rate=0.5)

        result = parser._read_waveform(
            mock_edf, "Ti.2s", WaveformType.TI, "seconds", valid_range=(0.0, 10.0)
        )
        assert result is not None
        wf, _ = result
        assert wf.unit == "seconds", f"Expected 'seconds', got {wf.unit!r}"

    def test_seconds_unchanged(self):
        """physical_dimension already "seconds" passes through unchanged."""
        data = np.array([1.0, 1.1], dtype=np.float64)
        parser = ResmedEDFParser()
        mock_edf = self._make_mock_edf(data, unit="seconds", sample_rate=0.5)

        result = parser._read_waveform(
            mock_edf, "Ti.2s", WaveformType.TI, "seconds", valid_range=(0.0, 10.0)
        )
        assert result is not None
        wf, _ = result
        assert wf.unit == "seconds"


# ─────────────────────────────────────────────────────────────────────────────
# scale_factor / convert_lps_to_lpm mutual exclusion
# ─────────────────────────────────────────────────────────────────────────────


class TestReadWaveformMutualExclusion:
    """Passing both scale_factor and convert_lps_to_lpm raises ValueError."""

    def test_both_flags_raises_value_error(self):
        info = MagicMock()
        info.physical_dimension = "L/s"
        mock_edf = MagicMock()
        data = np.array([0.1, 0.2], dtype=np.float64)
        mock_edf.read_signal.return_value = (data, info)

        parser = ResmedEDFParser()
        with pytest.raises(ValueError, match="mutually exclusive"):
            parser._read_waveform(
                mock_edf,
                "Flow",
                WaveformType.FLOW_RATE,
                "L/min",
                convert_lps_to_lpm=True,
                scale_factor=60.0,
            )
