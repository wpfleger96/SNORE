"""Unit tests for ExportService (raw, CSV, JSON exports)."""

from __future__ import annotations

import csv
import json
import zipfile

from collections.abc import AsyncGenerator
from datetime import date, datetime
from pathlib import Path

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.metrics import EXPORT_STAT_KEYS
from snore.services.export_service import ExportService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backup_root(tmp_path: Path) -> Path:
    return tmp_path / "raw"


@pytest.fixture()
def export_service(backup_root: Path) -> ExportService:
    return ExportService(1, backup_root=backup_root)


@pytest.fixture()
def fake_backup(backup_root: Path) -> Path:
    """Create a fake ResMed backup directory that looks like an SD card root."""
    device = backup_root / "SN12345"
    device.mkdir(parents=True)

    # STR.edf — minimal valid EDF header (256 bytes)
    header = bytearray(256)
    # version (bytes 0-7)
    header[0:8] = b"0       "
    # start date (bytes 168-175): 01.08.25
    header[168:176] = b"01.08.25"
    # start time (bytes 176-183): 00.00.00
    header[176:184] = b"00.00.00"
    # num records (bytes 236-243)
    header[236:244] = b"30      "
    # record duration (bytes 244-251)
    header[244:252] = b"86400   "
    # num signals (bytes 252-255)
    header[252:256] = b"1   "
    (device / "STR.edf").write_bytes(bytes(header))
    (device / "Identification.json").write_text('{"serial": "SN12345"}')

    datalog = device / "DATALOG"
    # Night 1: 20250806 (session at 01:34 AM → night of 20250806)
    d1 = datalog / "20250807"
    d1.mkdir(parents=True)
    (d1 / "20250807_013454_BRP.edf").write_bytes(b"brp1")
    (d1 / "20250807_013454_PLD.edf").write_bytes(b"pld1")

    # Night 2: 20250807 (session at 22:00 → night of 20250807)
    d2 = datalog / "20250807"  # same folder, different session time
    (d2 / "20250807_220000_BRP.edf").write_bytes(b"brp2")
    (d2 / "20250807_220000_PLD.edf").write_bytes(b"pld2")

    # Night 3: 20250809 (session at 23:00 → night of 20250809)
    d3 = datalog / "20250809"
    d3.mkdir(parents=True)
    (d3 / "20250809_230000_BRP.edf").write_bytes(b"brp3")

    return device


# ---------------------------------------------------------------------------
# export_raw
# ---------------------------------------------------------------------------


class TestExportRaw:
    def test_export_to_directory(
        self, export_service: ExportService, fake_backup: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "export"
        result = export_service.export_raw(out, device_serial="SN12345")

        assert result.format == "raw"
        assert result.nights_exported > 0
        assert result.files_written > 0
        assert (out / "STR.edf").exists()
        assert (out / "Identification.json").exists()

    def test_export_to_zip(
        self, export_service: ExportService, fake_backup: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "export.zip"
        result = export_service.export_raw(out, device_serial="SN12345", as_zip=True)

        assert result.format == "raw"
        assert out.exists()
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "STR.edf" in names
            assert "Identification.json" in names

    def test_export_zip_by_extension(
        self, export_service: ExportService, fake_backup: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "export.zip"
        export_service.export_raw(out, device_serial="SN12345")
        assert out.exists()
        assert zipfile.is_zipfile(out)

    def test_dry_run(
        self, export_service: ExportService, fake_backup: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "export"
        result = export_service.export_raw(out, device_serial="SN12345", dry_run=True)

        assert result.nights_exported > 0
        assert result.files_written > 0
        assert not out.exists()

    def test_date_filter(
        self, export_service: ExportService, fake_backup: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "export"
        result = export_service.export_raw(
            out,
            device_serial="SN12345",
            date_from=date(2025, 8, 9),
            date_to=date(2025, 8, 9),
        )

        # Only night 20250809 should be included
        assert result.nights_exported == 1
        # Device files always included
        assert (out / "STR.edf").exists()

    def test_date_filter_warns(
        self, export_service: ExportService, fake_backup: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "export"
        result = export_service.export_raw(
            out,
            device_serial="SN12345",
            date_from=date(2025, 8, 9),
            date_to=date(2025, 8, 9),
        )
        assert any("outside" in w.lower() for w in result.warnings)

    def test_auto_resolves_single_device(
        self, export_service: ExportService, fake_backup: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "export"
        # Don't specify device — should auto-detect the single one
        result = export_service.export_raw(out)
        assert result.nights_exported > 0

    def test_multiple_devices_no_serial_raises(
        self, export_service: ExportService, backup_root: Path, fake_backup: Path
    ) -> None:
        (backup_root / "SN99999").mkdir()
        with pytest.raises(ValueError, match="Multiple devices"):
            export_service.export_raw(Path("/tmp/out"))

    def test_no_backup_raises(self, tmp_path: Path) -> None:
        svc = ExportService(1, backup_root=tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            svc.export_raw(Path("/tmp/out"), device_serial="NOPE")


# ---------------------------------------------------------------------------
# export_csv and export_json (with in-memory SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def db_session(tmp_path: Path) -> AsyncGenerator[AsyncSession]:
    """Create a fresh async SQLite DB with test data per test."""
    import snore.database.session as db_mod

    from snore.database import models

    db_path = str(tmp_path / "test.db")

    # Full cleanup of the singleton — resets engine, factory, db_path,
    # _init_future, and _init_lock so concurrent xdist workers can't see
    # a stale future from another test's init cycle.
    await db_mod.cleanup_database()
    await db_mod.init_database(db_path)

    async with db_mod.session_scope() as session:
        user = models.User(canonical_email="export@example.com", role="admin")
        session.add(user)
        await session.flush()
        profile = models.Profile(user_id=user.id, name="Test Profile")
        session.add(profile)
        await session.flush()

        device = models.Device(
            profile_id=profile.id,
            manufacturer="ResMed",
            model="AirSense 11",
            serial_number="SN12345",
        )
        session.add(device)
        await session.flush()

        s = models.Session(
            device_id=device.id,
            device_session_id="20250807_220000",
            start_time=datetime(2025, 8, 7, 22, 0, 0),
            end_time=datetime(2025, 8, 8, 6, 0, 0),
            duration_seconds=28800.0,
            therapy_mode="APAP",
            import_source="resmed_edf",
            has_waveform_data=False,
            has_event_data=True,
            has_statistics=True,
            enabled=True,
        )
        session.add(s)
        await session.flush()

        session.add(
            models.Statistics(
                session_id=s.id,
                ahi=3.5,
                oai=1.0,
                cai=0.5,
                hi=2.0,
                ahi_device=3.2,
                obstructive_apneas=8,
                central_apneas=4,
                hypopneas=16,
                pressure_mean=10.5,
                pressure_95th=12.0,
                leak_mean=5.0,
                leak_95th=15.0,
                usage_hours=8.0,
            )
        )

        session.add(
            models.Event(
                session_id=s.id,
                event_type="OA",
                start_time=datetime(2025, 8, 7, 23, 15, 0),
                duration_seconds=12.5,
            )
        )

        session.add(models.Setting(session_id=s.id, key="mode", value="APAP"))
        session.add(models.Setting(session_id=s.id, key="pressure_min", value="6.0"))

        yield session

    # Full cleanup — disposes engine and resets all singleton state including
    # _init_future and _init_lock so the next test starts clean.
    await db_mod.cleanup_database()


class TestExportCsv:
    async def test_creates_csv_files(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "csv_export"
        result = await svc.export_csv(db_session, out)

        assert result.format == "csv"
        assert (out / "sessions.csv").exists()
        assert (out / "events.csv").exists()
        assert (out / "settings.csv").exists()
        assert (out / "columns.csv").exists()
        assert result.files_written == 4

    async def test_sessions_csv_content(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "csv_export"
        await svc.export_csv(db_session, out)

        with open(out / "sessions.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["device_session_id"] == "20250807_220000"
        assert row["device_serial"] == "SN12345"
        assert row["ahi"] == "3.5"
        assert row["ahi_device"] == "3.2"
        assert row["oai_device"] == ""
        assert row["timezone"] == "local"

    async def test_columns_csv_covers_session_metric_columns(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "csv_export"
        await svc.export_csv(db_session, out)

        with open(out / "columns.csv") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["name", "provenance", "description"]
            rows = {r["name"]: r for r in reader}
        with open(out / "sessions.csv") as f:
            session_columns = next(csv.reader(f))

        metric_columns = ["duration_hours", *EXPORT_STAT_KEYS]
        assert set(metric_columns) <= set(session_columns)
        assert set(metric_columns) <= rows.keys()
        assert all(r["description"] for r in rows.values())
        assert not any(r["description"].startswith("[") for r in rows.values())
        assert rows["ahi"]["provenance"] == "derived"
        assert rows["ahi_device"]["provenance"] == "device"
        assert rows["obstructive_apneas"]["provenance"] == "device"
        assert rows["events.csv"]["provenance"] == "device"

    async def test_events_csv_content(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "csv_export"
        await svc.export_csv(db_session, out)

        with open(out / "events.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["event_type"] == "OA"
        assert rows[0]["timezone"] == "local"

    async def test_settings_csv_content(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "csv_export"
        await svc.export_csv(db_session, out)

        with open(out / "settings.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        keys = {r["key"] for r in rows}
        assert "mode" in keys
        assert "pressure_min" in keys

    async def test_date_filter_excludes(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "csv_export"
        result = await svc.export_csv(db_session, out, date_from=date(2026, 1, 1))
        assert result.nights_exported == 0

    async def test_no_sessions_warns(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "csv_export"
        result = await svc.export_csv(db_session, out, date_from=date(2099, 1, 1))
        assert any("No sessions" in w for w in result.warnings)


class TestExportJson:
    async def test_creates_json_file(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "export.json"
        result = await svc.export_json(db_session, out)

        assert result.format == "json"
        assert out.exists()
        assert result.files_written == 1

    async def test_json_structure(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "export.json"
        await svc.export_json(db_session, out)

        with open(out) as f:
            doc = json.load(f)

        assert "exported_at" in doc
        assert "sessions" in doc
        assert doc["session_count"] == 1

        s = doc["sessions"][0]
        assert s["device_session_id"] == "20250807_220000"
        assert s["timezone"] == "local"
        assert s["device"]["serial"] == "SN12345"
        assert s["statistics"]["ahi"] == 3.5
        assert len(s["events"]) == 1
        assert s["events"][0]["event_type"] == "OA"
        assert s["settings"]["mode"] == "APAP"

    async def test_json_header_declares_provenance(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "export.json"
        await svc.export_json(db_session, out)

        with open(out) as f:
            doc = json.load(f)

        assert doc["snore_export_format"] == "1.1"
        columns = doc["provenance"]["columns"]
        assert columns.keys() == {"duration_hours", *EXPORT_STAT_KEYS}
        assert columns["ahi"] == "derived"
        assert columns["ahi_device"] == "device"
        assert columns["obstructive_apneas"] == "device"
        assert columns["usage_hours"] == "derived"
        assert doc["provenance"]["sections"] == {
            "events": "device",
            "settings": "device",
        }
        stats = doc["sessions"][0]["statistics"]
        assert stats["ahi_device"] == 3.2
        assert "oai_device" not in stats

    async def test_date_filter(self, db_session: AsyncSession, tmp_path: Path) -> None:
        svc = ExportService(1)
        out = tmp_path / "export.json"
        await svc.export_json(db_session, out, date_from=date(2099, 1, 1))

        with open(out) as f:
            doc = json.load(f)
        assert doc["session_count"] == 0

    async def test_device_filter(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        svc = ExportService(1)
        out = tmp_path / "export.json"
        await svc.export_json(db_session, out, device_serial="NONEXISTENT")

        with open(out) as f:
            doc = json.load(f)
        assert doc["session_count"] == 0


# ---------------------------------------------------------------------------
# STR.edf slicing
# ---------------------------------------------------------------------------


def _make_synthetic_str_edf(
    path: Path,
    start_date: str = "01.01.25",
    num_records: int = 30,
    num_signals: int = 2,
) -> None:
    """Create a minimal valid STR.edf for slicing tests.

    Each signal has 1 sample per record (int16), so record_size = num_signals * 2.
    Total file = 256 (header) + num_signals*256 (signal headers) + num_records * record_size.
    """
    header = bytearray(256)
    header[0:8] = b"0       "
    header[168:176] = f"{start_date:<8}".encode("ascii")
    header[176:184] = b"12.00.00"
    header_size = 256 + num_signals * 256
    header[184:192] = f"{header_size:<8}".encode("ascii")
    header[236:244] = f"{num_records:<8}".encode("ascii")
    header[244:252] = b"86400   "
    header[252:256] = f"{num_signals:<4}".encode("ascii")

    sig_headers = bytearray(num_signals * 256)
    # samples_per_record at offset num_signals * 216 within signal headers
    spr_offset = num_signals * 216
    for i in range(num_signals):
        sig_headers[spr_offset + i * 8 : spr_offset + (i + 1) * 8] = b"1       "
    # signal labels (16 bytes each, at the start)
    for i in range(num_signals):
        label = f"Signal{i:<10}".encode("ascii")[:16]
        sig_headers[i * 16 : (i + 1) * 16] = label

    record_size = num_signals * 2
    data = bytearray(num_records * record_size)
    for rec in range(num_records):
        for sig in range(num_signals):
            offset = rec * record_size + sig * 2
            data[offset : offset + 2] = (rec + sig).to_bytes(2, "little", signed=True)

    with open(path, "wb") as f:
        f.write(bytes(header))
        f.write(bytes(sig_headers))
        f.write(bytes(data))


class TestSliceStrEdf:
    def test_full_range_preserves_file(self, tmp_path: Path) -> None:
        from snore.parsers.resmed_edf import ResmedEDFParser

        src = tmp_path / "STR.edf"
        _make_synthetic_str_edf(src, "01.01.25", num_records=30, num_signals=2)
        original_size = src.stat().st_size

        dest = tmp_path / "out.edf"
        ResmedEDFParser._slice_str_edf(src, dest, date(2025, 1, 1), date(2025, 1, 30))

        assert dest.stat().st_size == original_size

    def test_subset_reduces_size(self, tmp_path: Path) -> None:
        from snore.parsers.resmed_edf import ResmedEDFParser

        src = tmp_path / "STR.edf"
        _make_synthetic_str_edf(src, "01.01.25", num_records=30, num_signals=2)

        dest = tmp_path / "out.edf"
        ResmedEDFParser._slice_str_edf(src, dest, date(2025, 1, 10), date(2025, 1, 19))

        # 10 records, record_size=4, header=256+512=768
        expected = 768 + 10 * 4
        assert dest.stat().st_size == expected

        header = dest.read_bytes()[:256]
        num_records = int(header[236:244].decode("ascii").strip())
        assert num_records == 10
        date_str = header[168:176].decode("ascii").strip()
        assert date_str == "10.01.25"

    def test_clamp_before_start(self, tmp_path: Path) -> None:
        from snore.parsers.resmed_edf import ResmedEDFParser

        src = tmp_path / "STR.edf"
        _make_synthetic_str_edf(src, "01.01.25", num_records=30, num_signals=2)

        dest = tmp_path / "out.edf"
        ResmedEDFParser._slice_str_edf(src, dest, date(2024, 12, 1), date(2025, 1, 10))

        header = dest.read_bytes()[:256]
        num_records = int(header[236:244].decode("ascii").strip())
        assert num_records == 10
        assert header[168:176].decode("ascii").strip() == "01.01.25"

    def test_clamp_after_end(self, tmp_path: Path) -> None:
        from snore.parsers.resmed_edf import ResmedEDFParser

        src = tmp_path / "STR.edf"
        _make_synthetic_str_edf(src, "01.01.25", num_records=30, num_signals=2)

        dest = tmp_path / "out.edf"
        ResmedEDFParser._slice_str_edf(src, dest, date(2025, 1, 20), date(2025, 12, 31))

        header = dest.read_bytes()[:256]
        num_records = int(header[236:244].decode("ascii").strip())
        assert num_records == 11  # records 19..29 inclusive

    def test_no_overlap_writes_empty(self, tmp_path: Path) -> None:
        from snore.parsers.resmed_edf import ResmedEDFParser

        src = tmp_path / "STR.edf"
        _make_synthetic_str_edf(src, "01.01.25", num_records=30, num_signals=2)

        dest = tmp_path / "out.edf"
        ResmedEDFParser._slice_str_edf(src, dest, date(2026, 1, 1), date(2026, 12, 31))

        header = dest.read_bytes()[:256]
        num_records = int(header[236:244].decode("ascii").strip())
        assert num_records == 0
        # File should be header + signal headers only, no data
        assert dest.stat().st_size == 768

    def test_in_place_rewrite(self, tmp_path: Path) -> None:
        from snore.parsers.resmed_edf import ResmedEDFParser

        f = tmp_path / "STR.edf"
        _make_synthetic_str_edf(f, "01.01.25", num_records=30, num_signals=2)
        original_size = f.stat().st_size

        ResmedEDFParser._slice_str_edf(f, f, date(2025, 1, 5), date(2025, 1, 14))

        assert f.stat().st_size < original_size
        header = f.read_bytes()[:256]
        assert int(header[236:244].decode("ascii").strip()) == 10
