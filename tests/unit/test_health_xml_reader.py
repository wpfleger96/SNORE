"""Unit tests for the Apple Health export.xml reader.

All tests are synchronous — no database access required.
"""

from __future__ import annotations

import zipfile

from datetime import date, datetime
from pathlib import Path
from xml.etree import ElementTree

import pytest

from snore.parsers.apple_health.models import RawHealthRecord
from snore.parsers.apple_health.type_handlers import parse_xml_record
from snore.parsers.apple_health.xml_reader import iter_records

FIXTURE_XML = Path(__file__).parent.parent / "fixtures" / "health_data" / "export.xml"

SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"


def _all_records(source: Path, **kwargs: object) -> list[RawHealthRecord]:
    return list(iter_records(source, **kwargs))


class TestDTDTolerance:
    """The iOS-16 malformed DOCTYPE must not cause parse errors."""

    def test_dtd_in_fixture_does_not_raise(self) -> None:
        """Parsing the fixture (which has a <!DOCTYPE...> block) must succeed."""
        records = _all_records(FIXTURE_XML)
        assert len(records) > 0


class TestSleepStageParsing:
    """Sleep stage normalisation from XML value strings."""

    def _sleep_records(self, source: Path) -> list[RawHealthRecord]:
        return [r for r in _all_records(source) if r.record_type == SLEEP_TYPE]

    def test_all_handled_sleep_stages_present(self) -> None:
        """Fixture covers every canonical sleep stage."""
        stages = {r.value_text for r in self._sleep_records(FIXTURE_XML)}
        expected = {
            "InBed",
            "AsleepUnspecified",
            "Awake",
            "AsleepCore",
            "AsleepDeep",
            "AsleepREM",
        }
        assert expected.issubset(stages)

    def test_legacy_asleep_normalized_to_unspecified(self) -> None:
        """Legacy HKCategoryValueSleepAnalysisAsleep maps to AsleepUnspecified."""
        records = [
            r
            for r in _all_records(FIXTURE_XML)
            if r.record_type == SLEEP_TYPE
            and r.source_name == "Will's iPhone"
            and r.start_time == datetime(2024, 1, 16, 1, 0, 0)
        ]
        assert len(records) == 1
        assert records[0].value_text == "AsleepUnspecified"
        assert records[0].ingest_channel == "export_xml"

    def test_sleep_record_has_no_value_num(self) -> None:
        """Sleep (category) records must have value_num == None."""
        for r in self._sleep_records(FIXTURE_XML):
            assert r.value_num is None

    def test_sleep_record_utc_offset_parsed(self) -> None:
        """UTC offset for -0500 is -18000 seconds."""
        records = self._sleep_records(FIXTURE_XML)
        for r in records:
            assert r.utc_offset_seconds == -18000

    def test_iphone_inbed_midnight_crossing(self) -> None:
        """Midnight-crossing InBed record from iPhone is parsed correctly."""
        records = [
            r
            for r in self._sleep_records(FIXTURE_XML)
            if r.source_name == "Will's iPhone" and r.value_text == "InBed"
        ]
        assert len(records) == 1
        rec = records[0]
        assert rec.start_time == datetime(2024, 1, 15, 23, 0, 0)
        assert rec.end_time == datetime(2024, 1, 16, 7, 0, 0)
        assert rec.night_date == date(2024, 1, 15)

    def test_asleeprem_11_30_previous_night_date(self) -> None:
        """11:30 start maps night_date to the previous calendar date."""
        records = [
            r for r in self._sleep_records(FIXTURE_XML) if r.value_text == "AsleepREM"
        ]
        assert len(records) == 1
        assert records[0].night_date == date(2024, 1, 14)

    def test_asleepcore_23_30_same_night_date(self) -> None:
        """23:30 start maps night_date to the same calendar date."""
        records = [
            r for r in self._sleep_records(FIXTURE_XML) if r.value_text == "AsleepCore"
        ]
        assert len(records) == 1
        assert records[0].night_date == date(2024, 1, 15)


class TestQuantityRecords:
    """Quantity record (SpO2, respiratory rate) parsing."""

    def test_spo2_value_and_unit(self) -> None:
        """SpO2 recorded as a fraction (0.962) is stored as percent; unit as-arrived."""
        spo2 = [
            r
            for r in _all_records(FIXTURE_XML)
            if r.record_type == "HKQuantityTypeIdentifierOxygenSaturation"
        ]
        assert len(spo2) == 1
        rec = spo2[0]
        assert rec.value_num == pytest.approx(96.2, abs=1e-6)
        assert rec.unit == "%"
        assert rec.value_text is None

    @pytest.mark.parametrize(
        ("recorded", "stored"),
        [
            ("0.95", 95.0),  # fraction → percent
            ("0.93", 93.0),  # no float residue (0.93 * 100 != 93.0)
            ("0.9512", 95.12),
            ("0.5", 50.0),  # fraction range boundaries
            ("1", 100.0),
            ("95", 95.0),  # already percent
            ("100", 100.0),
            ("0.2", 0.2),  # implausible on either scale: kept as recorded
            ("1.5", 1.5),
            ("20", 20.0),
        ],
    )
    def test_spo2_stored_on_percent_scale(self, recorded: str, stored: float) -> None:
        """SpO2 fractions in [0.5, 1] are scaled to percent; other values are kept."""
        elem = ElementTree.fromstring(
            '<Record type="HKQuantityTypeIdentifierOxygenSaturation" '
            'sourceName="Watch" startDate="2024-01-16 02:00:00 -0500" '
            f'endDate="2024-01-16 02:00:00 -0500" value="{recorded}" unit="%"/>'
        )

        rec = parse_xml_record(elem)

        assert rec is not None
        assert rec.value_num == stored
        assert rec.unit == "%"

    def test_non_spo2_fraction_not_scaled(self) -> None:
        """Only SpO2 is rescaled: a fractional respiratory rate is kept as recorded."""
        elem = ElementTree.fromstring(
            '<Record type="HKQuantityTypeIdentifierRespiratoryRate" '
            'sourceName="Watch" startDate="2024-01-16 02:00:00 -0500" '
            'endDate="2024-01-16 02:00:00 -0500" value="0.95" unit="count/min"/>'
        )

        rec = parse_xml_record(elem)

        assert rec is not None
        assert rec.value_num == 0.95

    def test_respiratory_rate_value_and_unit(self) -> None:
        """Respiratory rate record parses numeric value and unit correctly."""
        rr = [
            r
            for r in _all_records(FIXTURE_XML)
            if r.record_type == "HKQuantityTypeIdentifierRespiratoryRate"
        ]
        assert len(rr) == 1
        assert rr[0].value_num == pytest.approx(14.5, abs=1e-6)
        assert rr[0].unit == "count/min"

    def test_quantity_point_sample_start_equals_end(self) -> None:
        """Point-sample quantity records have start_time == end_time."""
        spo2 = [
            r
            for r in _all_records(FIXTURE_XML)
            if r.record_type == "HKQuantityTypeIdentifierOxygenSaturation"
        ]
        assert spo2[0].start_time == spo2[0].end_time

    def test_value_num_rounded_to_4dp(self) -> None:
        """value_num is rounded to 4 decimal places at parse time."""
        recs = [r for r in _all_records(FIXTURE_XML) if r.value_num is not None]
        assert len(recs) > 0
        for r in recs:
            assert r.value_num == round(r.value_num, 4)


class TestZipAndDirectoryInput:
    """iter_records works from zip files and directory layouts."""

    def _build_zip(self, tmp_path: Path, member_name: str) -> Path:
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(FIXTURE_XML, arcname=member_name)
        return zip_path

    def test_zip_with_prefixed_member(self, tmp_path: Path) -> None:
        """Zip with apple_health_export/export.xml member is handled."""
        zip_path = self._build_zip(tmp_path, "apple_health_export/export.xml")
        records = _all_records(zip_path)
        assert len(records) > 0

    def test_zip_with_bare_member(self, tmp_path: Path) -> None:
        """Zip with export.xml at root is handled."""
        zip_path = self._build_zip(tmp_path, "export.xml")
        records = _all_records(zip_path)
        assert len(records) > 0

    def test_zip_yields_same_records_as_file(self, tmp_path: Path) -> None:
        """Zip and direct file inputs produce the same records."""
        zip_path = self._build_zip(tmp_path, "apple_health_export/export.xml")
        from_zip = _all_records(zip_path)
        from_file = _all_records(FIXTURE_XML)
        assert len(from_zip) == len(from_file)

    def test_directory_with_direct_export_xml(self, tmp_path: Path) -> None:
        """Directory containing export.xml directly is handled."""
        import shutil  # noqa: PLC0415

        shutil.copy(FIXTURE_XML, tmp_path / "export.xml")
        records = _all_records(tmp_path)
        assert len(records) > 0

    def test_directory_with_nested_apple_health_export(self, tmp_path: Path) -> None:
        """Directory with apple_health_export/export.xml subdirectory is handled."""
        import shutil  # noqa: PLC0415

        nested = tmp_path / "apple_health_export"
        nested.mkdir()
        shutil.copy(FIXTURE_XML, nested / "export.xml")
        records = _all_records(tmp_path)
        assert len(records) > 0


class TestDateFiltering:
    """Date range filtering on night_date."""

    def test_date_from_excludes_earlier_records(self) -> None:
        """Records with night_date before date_from are excluded."""
        all_recs = _all_records(FIXTURE_XML)
        filtered = list(iter_records(FIXTURE_XML, date_from=date(2024, 1, 15)))
        excluded = [r for r in all_recs if r.night_date < date(2024, 1, 15)]
        assert len(excluded) > 0, "Fixture should have records before 2024-01-15"
        assert all(r.night_date >= date(2024, 1, 15) for r in filtered)

    def test_date_to_excludes_later_records(self) -> None:
        """Records with night_date after date_to are excluded."""
        filtered = list(iter_records(FIXTURE_XML, date_to=date(2024, 1, 14)))
        assert all(r.night_date <= date(2024, 1, 14) for r in filtered)

    def test_date_range_both_bounds(self) -> None:
        """Both bounds applied together."""
        filtered = list(
            iter_records(
                FIXTURE_XML,
                date_from=date(2024, 1, 15),
                date_to=date(2024, 1, 15),
            )
        )
        assert all(r.night_date == date(2024, 1, 15) for r in filtered)
        assert len(filtered) > 0


class TestLimit:
    """limit parameter caps the number of yielded records."""

    def test_limit_caps_output(self) -> None:
        """Requesting limit=2 yields at most 2 records."""
        records = list(iter_records(FIXTURE_XML, limit=2))
        assert len(records) == 2

    def test_limit_none_yields_all(self) -> None:
        """limit=None (default) yields all records."""
        without_limit = _all_records(FIXTURE_XML)
        with_none = list(iter_records(FIXTURE_XML, limit=None))
        assert len(with_none) == len(without_limit)


class TestDecompressedSizeCap:
    """iter_records raises ValueError when the decompressed XML exceeds MAX_DECOMPRESSED_BYTES."""

    def test_oversized_stream_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A plain XML file larger than the monkeypatched cap raises ValueError."""
        import snore.parsers.apple_health.xml_reader as xml_reader_mod

        monkeypatch.setattr(xml_reader_mod, "MAX_DECOMPRESSED_BYTES", 50)

        # 200 bytes of valid XML — well over the tiny cap.
        large_xml = b"<HealthData>" + b" " * 188 + b"</HealthData>"
        xml_path = tmp_path / "export.xml"
        xml_path.write_bytes(large_xml)

        with pytest.raises(ValueError, match="zip-bomb"):
            list(iter_records(xml_path))

    def test_oversized_zip_member_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A zip member larger than the monkeypatched cap raises ValueError."""
        import snore.parsers.apple_health.xml_reader as xml_reader_mod

        monkeypatch.setattr(xml_reader_mod, "MAX_DECOMPRESSED_BYTES", 50)

        large_xml = b"<HealthData>" + b" " * 188 + b"</HealthData>"
        zip_path = tmp_path / "export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("apple_health_export/export.xml", large_xml)

        with pytest.raises(ValueError, match="zip-bomb"):
            list(iter_records(zip_path))

    def test_normal_fixture_within_default_cap(self) -> None:
        """The standard test fixture is well under the production 10 GiB cap."""
        records = list(iter_records(FIXTURE_XML))
        assert len(records) > 0


class TestSkipCounter:
    """skip_counter tracks unhandled or unparseable record types."""

    def test_skip_counter_counts_step_count(self) -> None:
        """StepCount is unhandled in v1 and must appear in skip_counter."""
        counter: dict[str, int] = {}
        _all_records(FIXTURE_XML, skip_counter=counter)
        assert "HKQuantityTypeIdentifierStepCount" in counter
        assert counter["HKQuantityTypeIdentifierStepCount"] >= 1

    def test_skip_counter_none_does_not_raise(self) -> None:
        """Passing skip_counter=None (default) works without errors."""
        records = list(iter_records(FIXTURE_XML))
        assert len(records) > 0
