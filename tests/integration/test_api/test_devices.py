"""Integration tests for the /devices API router."""

from datetime import date, datetime

from sqlalchemy.orm import Session as OrmSession

from snore.database.models import Day, Setting


def _add_settings(
    db_session: OrmSession, session_id: int, settings: dict[str, str]
) -> None:
    for key, value in settings.items():
        db_session.add(Setting(session_id=session_id, key=key, value=value))
    db_session.flush()


class TestListDevices:
    def test_list_devices_empty(self, api_client):
        response = api_client.get("/api/v1/devices/")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_devices_with_data(self, api_client, test_device):
        response = api_client.get("/api/v1/devices/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        d = data[0]
        assert d["id"] == test_device.id
        assert d["serial_number"] == test_device.serial_number
        assert "firmware_version" in d
        assert "hardware_version" in d
        assert "product_code" in d
        assert "first_seen" in d
        assert "last_import" in d


class TestGetDeviceDetail:
    def test_detail_404_on_unknown_id(self, api_client):
        response = api_client.get("/api/v1/devices/99999")
        assert response.status_code == 404

    def test_detail_shape_no_sessions(self, api_client, test_device):
        response = api_client.get(f"/api/v1/devices/{test_device.id}")
        assert response.status_code == 200
        data = response.json()
        # Identity fields
        assert data["id"] == test_device.id
        assert data["manufacturer"] == test_device.manufacturer
        assert data["model"] == test_device.model
        assert data["serial_number"] == test_device.serial_number
        # Usage with no sessions
        assert data["usage"]["session_count"] == 0
        assert data["usage"]["first_session_date"] is None
        assert data["usage"]["last_session_date"] is None
        assert data["usage"]["total_therapy_hours"] == 0.0
        assert data["usage"]["therapy_modes"] == []
        # Settings
        assert data["current_settings"] is None
        assert data["settings_history"] == []

    def test_detail_usage_shape(
        self, api_client, db_session, test_device, test_session_factory
    ):
        s = test_session_factory(
            test_device.id,
            start_time=datetime(2024, 1, 1, 22, 0),
            duration_hours=7.5,
        )
        # Hours set by hand: shape test only; derivation is in test_device_service.py.
        day = Day(
            device_id=test_device.id,
            date=date(2024, 1, 1),
            session_count=1,
            total_therapy_hours=7.5,
        )
        db_session.add(day)
        db_session.flush()
        s.day_id = day.id
        db_session.flush()
        response = api_client.get(f"/api/v1/devices/{test_device.id}")
        assert response.status_code == 200
        usage = response.json()["usage"]
        assert usage["session_count"] == 1
        assert usage["first_session_date"] == "2024-01-01"
        assert usage["last_session_date"] == "2024-01-01"
        assert abs(usage["total_therapy_hours"] - 7.5) < 0.01

    def test_detail_current_settings(
        self, api_client, db_session, test_device, test_session_factory
    ):
        s = test_session_factory(test_device.id, start_time=datetime(2024, 1, 1, 22, 0))
        _add_settings(db_session, s.id, {"mode": "AutoSet", "pressure_min": "4.0"})
        response = api_client.get(f"/api/v1/devices/{test_device.id}")
        assert response.status_code == 200
        settings = response.json()["current_settings"]
        assert settings == {"mode": "AutoSet", "pressure_min": "4.0"}

    def test_detail_settings_history(
        self, api_client, db_session, test_device, test_session_factory
    ):
        s1 = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        s2 = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        _add_settings(db_session, s1.id, {"pressure_max": "12.0"})
        _add_settings(db_session, s2.id, {"pressure_max": "10.0"})
        response = api_client.get(f"/api/v1/devices/{test_device.id}")
        assert response.status_code == 200
        history = response.json()["settings_history"]
        assert len(history) == 1
        assert history[0]["session_id"] == s2.id
        assert history[0]["date"] == "2024-01-02"
        assert len(history[0]["changes"]) == 1
        change = history[0]["changes"][0]
        assert change["key"] == "pressure_max"
        assert change["old_value"] == "12.0"
        assert change["new_value"] == "10.0"

    def test_detail_no_history_when_settings_unchanged(
        self, api_client, db_session, test_device, test_session_factory
    ):
        s1 = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        s2 = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 2, 22, 0)
        )
        _add_settings(db_session, s1.id, {"mode": "AutoSet"})
        _add_settings(db_session, s2.id, {"mode": "AutoSet"})
        response = api_client.get(f"/api/v1/devices/{test_device.id}")
        assert response.status_code == 200
        assert response.json()["settings_history"] == []
