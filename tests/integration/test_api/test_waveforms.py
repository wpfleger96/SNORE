from datetime import datetime

import numpy as np
import pytest

from sqlalchemy.orm import Session

from snore.database.models import Waveform
from snore.services import waveform_service as waveform_service_module


def _seed_waveform(
    db_session: Session, session_id: int, waveform_type: str = "flow", n: int = 2500
) -> Waveform:
    """Insert a Waveform row with a valid float32 (n, 2) [timestamp, value] blob."""
    timestamps = (np.arange(n, dtype=np.float32) * 0.04).astype(np.float32)  # 25 Hz
    values = np.sin(timestamps).astype(np.float32)
    blob = np.column_stack([timestamps, values]).astype(np.float32).tobytes()
    waveform = Waveform(
        session_id=session_id,
        waveform_type=waveform_type,
        sample_rate=25.0,
        unit="L/min",
        min_value=float(values.min()),
        max_value=float(values.max()),
        mean_value=float(values.mean()),
        data_blob=blob,
        sample_count=n,
    )
    db_session.add(waveform)
    db_session.flush()
    return waveform


class TestListWaveforms:
    def test_list_waveforms_empty(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        response = api_client.get(f"/api/v1/sessions/{session.id}/waveforms")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_waveforms_not_found_session(self, api_client):
        response = api_client.get("/api/v1/sessions/99999/waveforms")
        assert response.status_code == 404

    def test_list_waveforms_reports_sample_count_and_duration(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)

        response = api_client.get(f"/api/v1/sessions/{session.id}/waveforms")

        assert response.status_code == 200
        [info] = response.json()
        assert info["waveform_type"] == "flow"
        assert info["sample_count"] == 2500
        assert info["duration_hours"] == pytest.approx(2500 / 25.0 / 3600)


class TestGetWaveform:
    def test_get_waveform_not_found(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        response = api_client.get(f"/api/v1/sessions/{session.id}/waveforms/flow")
        assert response.status_code == 404

    def test_get_waveform_max_points_below_minimum_rejected(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"max_points": 99},
        )
        assert response.status_code == 422

    def test_get_waveform_max_points_above_maximum_rejected(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"max_points": 10001},
        )
        assert response.status_code == 422

    def test_get_waveform_max_points_at_default_accepted(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"max_points": 2000},
        )
        assert response.status_code == 200
        assert response.json()["returned_samples"] == 2000

    def test_get_waveform_max_points_at_maximum_accepted(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"max_points": 10000},
        )
        assert response.status_code == 200

    def test_get_waveform_negative_start_rejected(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"start_seconds": -1.0},
        )
        assert response.status_code == 422

    def test_get_waveform_negative_end_rejected(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"end_seconds": -0.5},
        )
        assert response.status_code == 422

    def test_get_waveform_nan_start_rejected(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"start_seconds": "nan"},
        )
        assert response.status_code == 422

    def test_get_waveform_inf_end_rejected(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"end_seconds": "inf"},
        )
        assert response.status_code == 422

    def test_get_waveform_valid_window_accepted(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        response = api_client.get(
            f"/api/v1/sessions/{session.id}/waveforms/flow",
            params={"start_seconds": 5.0, "end_seconds": 20.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["returned_samples"] > 0

    def test_get_waveform_windowed_miss_then_hit_reports_full_total(
        self, api_client, db_session, test_device, test_session_factory, monkeypatch
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        _seed_waveform(db_session, session.id)
        url = f"/api/v1/sessions/{session.id}/waveforms/flow"
        params = {"start_seconds": 5, "end_seconds": 20}
        calls = 0
        real_deser = waveform_service_module.deserialize_waveform_blob

        def counting_deser(*args, **kwargs):
            nonlocal calls
            calls += 1
            return real_deser(*args, **kwargs)

        monkeypatch.setattr(
            waveform_service_module, "deserialize_waveform_blob", counting_deser
        )

        miss = api_client.get(url, params=params)
        hit = api_client.get(url, params=params)

        assert miss.status_code == 200
        data = miss.json()
        assert data["total_samples"] == 2500
        assert data["returned_samples"] < data["total_samples"]
        assert hit.json() == data
        assert calls == 1  # only the miss deserialized; the second was a cache hit


class TestWaveformCompare:
    def test_compare_no_analysis_returns_404(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        response = api_client.get(f"/api/v1/sessions/{session.id}/waveforms/compare")
        assert response.status_code == 404
