import csv
import io
import shutil
import zipfile

from datetime import date
from pathlib import Path
from unittest.mock import patch

import httpx

from snore.auth.actor import ActorContext, AuthMode, Role
from tests.helpers.api_client import make_test_client
from tests.integration.conftest import _make_day_session, _make_device, _make_profile


async def _make_fake_csv_export(
    self: object, db: object, output: Path, **kwargs: object
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "sessions.csv").write_text("date,ahi\n")


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
        assert "snore_export.zip" in response.headers.get("content-disposition", "")

    async def test_csv_streams_zip_of_real_export(self, async_db_session):
        profile = await _make_profile(async_db_session)
        device = await _make_device(async_db_session, profile.id)
        _, sess = await _make_day_session(async_db_session, device, date(2026, 1, 15))
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


class TestExportJson:
    def test_json_returns_200(self, api_client):
        response = api_client.get("/api/v1/export/json")
        assert response.status_code == 200

    def test_json_content_type(self, api_client):
        response = api_client.get("/api/v1/export/json")
        assert "application/json" in response.headers["content-type"]

    def test_json_content_disposition(self, api_client):
        response = api_client.get("/api/v1/export/json")
        assert "snore_export.json" in response.headers.get("content-disposition", "")


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
