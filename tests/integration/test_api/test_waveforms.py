from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from sqlalchemy.orm import Session

from snore.analysis.shared.types import ApneaEvent
from snore.analysis.types import AnalysisEvent
from snore.database.models import Waveform
from snore.services.waveform_service import WaveformService


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


class TestWaveformCompare:
    def test_compare_no_analysis_returns_404(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        response = api_client.get(f"/api/v1/sessions/{session.id}/waveforms/compare")
        assert response.status_code == 404

    def test_compare_returns_matched_count(
        self, api_client, db_session, test_device, test_session_factory, monkeypatch
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2024, 1, 1, 22, 0)
        )
        apnea = ApneaEvent(
            start_time=100.0,
            end_time=110.0,
            duration=10.0,
            event_type="OA",
            flow_reduction=0.9,
            confidence=0.8,
            baseline_flow=30.0,
        )
        analysis = SimpleNamespace(
            machine_events=[
                AnalysisEvent(
                    event_type=event_type,
                    start_time=100.0,
                    duration=10.0,
                    source="machine",
                )
                for event_type in ("OA", "FL")
            ],
            mode_results={"aasm": SimpleNamespace(apneas=[apnea], hypopneas=[])},
        )
        monkeypatch.setattr(
            WaveformService,
            "_load_analysis_result",
            AsyncMock(return_value=analysis),
        )

        response = api_client.get(f"/api/v1/sessions/{session.id}/waveforms/compare")

        assert response.status_code == 200
        data = response.json()
        assert data["machine_event_count"] == 1
        assert data["matched_events"] == 1


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/sessions/{id}/waveforms",
        "/api/v1/sessions/{id}/waveforms/compare",
        "/api/v1/sessions/{id}/waveforms/flow",
    ],
)
def test_oversized_session_id_returns_422(api_client, path):
    response = api_client.get(path.format(id=2**63))

    assert response.status_code == 422
