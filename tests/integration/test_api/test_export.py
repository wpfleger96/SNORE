import csv
import io
import json
import shutil
import tempfile
import zipfile

from datetime import date
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from snore.api.routers import export as export_router
from snore.auth.actor import ActorContext, AuthMode, Role
from tests.helpers.api_client import make_test_client
from tests.helpers.str_edf import read_str_header, write_synthetic_str_edf
from tests.integration.conftest import _make_day_session, _make_device, _make_profile


async def _make_fake_csv_export(
    self: object, db: object, output: Path, **kwargs: object
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "sessions.csv").write_text("date,ahi\n")


async def _failing_csv_export(
    self: object, db: object, output: Path, **kwargs: object
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "sessions.csv").write_text("partial\n")
    raise RuntimeError("export failed")


def _fail_make_archive(*args: object, **kwargs: object) -> str:
    raise RuntimeError("zip failed")


@pytest.fixture
def export_tmpdirs(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every temp dir the export routes create."""
    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def _recording_mkdtemp() -> str:
        path = real_mkdtemp()
        created.append(Path(path))
        return path

    monkeypatch.setattr(export_router.tempfile, "mkdtemp", _recording_mkdtemp)
    return created


class TestExportCsv:
    def test_csv_returns_200(self, api_client):
        with patch(
            "snore.api.routers.export.ExportService.export_csv",
            _make_fake_csv_export,
        ):
            response = api_client.get("/api/v1/export/csv")
        assert response.status_code == 200

    def test_csv_content_type(self, api_client):
        with patch(
            "snore.api.routers.export.ExportService.export_csv",
            _make_fake_csv_export,
        ):
            response = api_client.get("/api/v1/export/csv")
        assert "application/zip" in response.headers["content-type"]

    def test_csv_content_disposition(self, api_client):
        with patch(
            "snore.api.routers.export.ExportService.export_csv",
            _make_fake_csv_export,
        ):
            response = api_client.get("/api/v1/export/csv")
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="snore_export_csv.zip"'
        )

    async def test_csv_streams_zip_of_real_export_scoped_to_profile(
        self, async_db_session
    ):
        profile = await _make_profile(async_db_session)
        device = await _make_device(async_db_session, profile.id)
        _, sess = await _make_day_session(async_db_session, device, date(2026, 1, 15))
        other_profile = await _make_profile(async_db_session)
        other_device = await _make_device(async_db_session, other_profile.id)
        await _make_day_session(async_db_session, other_device, date(2026, 1, 15))
        await async_db_session.commit()
        actor = ActorContext(
            user_id=profile.user_id,
            profile_id=profile.id,
            role=Role.MEMBER,
            mode=AuthMode.LOCAL,
        )
        app = make_test_client(async_db_session, actor=actor).app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/export/csv")

        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            assert {"sessions.csv", "events.csv", "settings.csv"} <= set(zf.namelist())
            rows = list(csv.DictReader(io.StringIO(zf.read("sessions.csv").decode())))
        assert [row["device_session_id"] for row in rows] == [sess.device_session_id]

    def test_csv_temp_dir_removed_after_stream(self, api_client, export_tmpdirs):
        with patch(
            "snore.api.routers.export.ExportService.export_csv",
            _make_fake_csv_export,
        ):
            response = api_client.get("/api/v1/export/csv")
        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            assert zf.read("sessions.csv") == b"date,ahi\n"
        assert len(export_tmpdirs) == 1
        assert not export_tmpdirs[0].exists()

    def test_csv_temp_dir_removed_when_export_raises(self, api_client, export_tmpdirs):
        with (
            patch(
                "snore.api.routers.export.ExportService.export_csv",
                _failing_csv_export,
            ),
            pytest.raises(RuntimeError, match="export failed"),
        ):
            api_client.get("/api/v1/export/csv")
        assert len(export_tmpdirs) == 1
        assert not export_tmpdirs[0].exists()

    def test_csv_temp_dir_removed_when_zip_raises(self, api_client, export_tmpdirs):
        with (
            patch(
                "snore.api.routers.export.ExportService.export_csv",
                _make_fake_csv_export,
            ),
            patch.object(export_router.shutil, "make_archive", _fail_make_archive),
            pytest.raises(RuntimeError, match="zip failed"),
        ):
            api_client.get("/api/v1/export/csv")
        assert len(export_tmpdirs) == 1
        assert not export_tmpdirs[0].exists()


class TestExportDeviceFilter:
    """The UI sends a device's serial number as ``device``; only its sessions export."""

    @staticmethod
    def _session_ids(fmt: str, body: bytes) -> list[str]:
        if fmt == "json":
            return [s["device_session_id"] for s in json.loads(body)["sessions"]]
        with zipfile.ZipFile(io.BytesIO(body)) as zf:
            rows = csv.DictReader(io.StringIO(zf.read("sessions.csv").decode()))
            return [row["device_session_id"] for row in rows]

    @pytest.mark.parametrize("fmt", ["csv", "json"])
    async def test_device_serial_filters_to_that_device(self, async_db_session, fmt):
        profile = await _make_profile(async_db_session)
        # Same model on both devices: only the serial tells them apart.
        device = await _make_device(async_db_session, profile.id, serial_number="SN_A")
        other = await _make_device(async_db_session, profile.id, serial_number="SN_B")
        _, sess = await _make_day_session(async_db_session, device, date(2026, 1, 15))
        await _make_day_session(async_db_session, other, date(2026, 1, 16))
        await async_db_session.commit()
        actor = ActorContext(
            user_id=profile.user_id,
            profile_id=profile.id,
            role=Role.MEMBER,
            mode=AuthMode.LOCAL,
        )
        app = make_test_client(async_db_session, actor=actor).app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/v1/export/{fmt}?device=SN_A")

        assert response.status_code == 200
        assert self._session_ids(fmt, response.content) == [sess.device_session_id]


class TestTempExportResponse:
    async def test_temp_dir_removed_when_client_disconnects(self, tmp_path):
        tmpdir = tmp_path / "export"
        tmpdir.mkdir()
        payload = tmpdir / "snore_export_csv.zip"
        payload.write_bytes(b"x" * 10)
        response = export_router._TempExportResponse(
            tmpdir, payload, "application/zip", "snore_export_csv.zip"
        )

        async def receive() -> dict[str, str]:
            return {"type": "http.disconnect"}

        async def send(message: dict[str, object]) -> None:
            if message["type"] == "http.response.body":
                raise OSError("client went away")

        scope = {
            "type": "http",
            "method": "GET",
            "headers": [],
            "asgi": {"spec_version": "2.4"},
        }
        with pytest.raises(OSError, match="client went away"):
            await response(scope, receive, send)
        assert not tmpdir.exists()


class TestExportJson:
    def test_json_returns_200(self, api_client):
        response = api_client.get("/api/v1/export/json")
        assert response.status_code == 200

    def test_json_content_type(self, api_client):
        response = api_client.get("/api/v1/export/json")
        assert "application/json" in response.headers["content-type"]

    def test_json_content_disposition(self, api_client):
        response = api_client.get("/api/v1/export/json")
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="snore_export.json"'
        )


class TestExportRaw:
    def test_raw_returns_200(self, api_client, tmp_path):
        """Mock ExportService.export_raw to write a dummy zip file."""
        dummy_zip = tmp_path / "dummy.zip"
        with zipfile.ZipFile(dummy_zip, "w") as zf:
            zf.writestr("dummy.txt", "test content")

        def fake_export_raw(self, output, **kwargs):
            shutil.copy(dummy_zip, output)
            return type("R", (), {"output_path": output})()

        with patch(
            "snore.api.routers.export.ExportService.export_raw",
            fake_export_raw,
        ):
            response = api_client.get("/api/v1/export/raw")
        assert response.status_code == 200

    def test_raw_content_type(self, api_client, tmp_path):
        dummy_zip = tmp_path / "dummy.zip"
        with zipfile.ZipFile(dummy_zip, "w") as zf:
            zf.writestr("dummy.txt", "test")

        def fake_export_raw(self, output, **kwargs):
            shutil.copy(dummy_zip, output)
            return type("R", (), {"output_path": output})()

        with patch(
            "snore.api.routers.export.ExportService.export_raw",
            fake_export_raw,
        ):
            response = api_client.get("/api/v1/export/raw")
        assert "application/zip" in response.headers["content-type"]


class TestExportRawTrimStr:
    @staticmethod
    async def _client_with_backup(
        async_db_session: AsyncSession,
        raw_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> TestClient:
        profile = await _make_profile(async_db_session)
        await async_db_session.commit()
        monkeypatch.setattr(export_router, "DEFAULT_RAW_BACKUP_DIR", raw_root)
        device = raw_root / str(profile.id) / "SN12345"
        datalog = device / "DATALOG"
        for night in ("20250805", "20250809", "20250812"):
            folder = datalog / night
            folder.mkdir(parents=True)
            (folder / f"{night}_230000_BRP.edf").write_bytes(b"brp")
        write_synthetic_str_edf(device / "STR.edf", "01.08.25", num_records=30)
        (device / "Identification.json").write_text('{"serial": "SN12345"}')
        actor = ActorContext(
            user_id=profile.user_id,
            profile_id=profile.id,
            role=Role.MEMBER,
            mode=AuthMode.LOCAL,
        )
        return make_test_client(async_db_session, actor=actor)

    async def test_zip_contains_str_trimmed_to_requested_days(
        self, async_db_session, tmp_path, monkeypatch
    ):
        client = await self._client_with_backup(
            async_db_session, tmp_path / "raw", monkeypatch
        )

        response = client.get(
            "/api/v1/export/raw",
            params={
                "trim_str": "true",
                "from_date": "2025-08-09",
                "to_date": "2025-08-10",
            },
        )

        assert response.status_code == 200
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            assert read_str_header(zf.read("STR.edf")) == ("09.08.25", 2)
            nights = {n for n in zf.namelist() if n.startswith("DATALOG/")}
        assert nights == {"DATALOG/20250809/20250809_230000_BRP.edf"}

    @pytest.mark.parametrize(
        "dates",
        [{}, {"from_date": "2025-08-09"}, {"to_date": "2025-08-10"}],
    )
    async def test_trim_without_both_dates_is_rejected(
        self, async_db_session, tmp_path, monkeypatch, dates
    ):
        client = await self._client_with_backup(
            async_db_session, tmp_path / "raw", monkeypatch
        )

        response = client.get(
            "/api/v1/export/raw", params={"trim_str": "true", **dates}
        )

        assert response.status_code == 400
        assert response.json() == {
            "detail": "trim_str requires both from_date and to_date"
        }


class TestExportRawDeviceSerial:
    async def test_serial_reaching_another_profile_is_rejected(
        self, async_db_session, tmp_path, monkeypatch
    ):
        profile = await _make_profile(async_db_session)
        other = await _make_profile(async_db_session)
        await async_db_session.commit()
        raw_root = tmp_path / "raw"
        monkeypatch.setattr(export_router, "DEFAULT_RAW_BACKUP_DIR", raw_root)
        (raw_root / str(profile.id)).mkdir(parents=True)
        victim = raw_root / str(other.id) / "SN_OTHER"
        (victim / "DATALOG" / "20250807").mkdir(parents=True)
        (victim / "STR.edf").write_bytes(bytes(256))
        (victim / "DATALOG" / "20250807" / "20250807_013454_BRP.edf").write_bytes(
            b"secret"
        )
        actor = ActorContext(
            user_id=profile.user_id,
            profile_id=profile.id,
            role=Role.MEMBER,
            mode=AuthMode.LOCAL,
        )
        client = make_test_client(async_db_session, actor=actor)

        for device in (f"../{other.id}/SN_OTHER", str(victim)):
            response = client.get("/api/v1/export/raw", params={"device": device})

            assert response.status_code == 400
            assert response.json() == {"detail": "Invalid device serial"}
