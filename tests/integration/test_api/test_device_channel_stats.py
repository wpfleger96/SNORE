"""REST shape of the FL and snore statistics and mask-on spans."""

from datetime import date, datetime

import numpy as np
import pytest

from sqlalchemy.orm import Session as DbSession

from snore.database.models import Day, Session, Waveform

_NIGHT = date(2025, 3, 1)


def _add_channel(
    db_session: DbSession, session: Session, channel: str, values: list[float]
) -> None:
    pairs = [(i * 2.0, value) for i, value in enumerate(values)]
    db_session.add(
        Waveform(
            session_id=session.id,
            waveform_type=channel,
            sample_rate=0.5,
            data_blob=np.array(pairs, dtype=np.float32).tobytes(),
            sample_count=len(values),
        )
    )
    db_session.flush()


@pytest.fixture
def night(db_session, test_device, test_session_factory):
    """One session on a Day row, with both channels and two mask-on segments."""
    day = Day(
        device_id=test_device.id,
        date=_NIGHT,
        session_count=1,
        total_therapy_hours=8.0,
    )
    db_session.add(day)
    db_session.flush()
    session = test_session_factory(
        test_device.id, start_time=datetime(2025, 3, 1, 22, 0), ahi_device=3.1
    )
    session.day_id = day.id
    session.mask_on_segments = [[0.0, 3600.0], [5400.0, 9000.0]]
    db_session.flush()
    _add_channel(db_session, session, "fl", [-0.01, 0.0, 0.5])
    _add_channel(db_session, session, "snore", [0.0, 2.0])
    return session


_EXPECTED_STATS = {
    "device_flg_median": 0.25,
    "device_flg_95th": 0.5,
    "device_flg_max": 0.5,
    "snore_median": 1.0,
    "snore_95th": 2.0,
    "snore_max": 2.0,
}


def _expected_spans(session_id: int) -> list[dict[str, object]]:
    return [
        {
            "session_id": session_id,
            "start": "2025-03-01T22:00:00",
            "end": "2025-03-01T23:00:00",
            "duration_seconds": 3600.0,
        },
        {
            "session_id": session_id,
            "start": "2025-03-01T23:30:00",
            "end": "2025-03-02T00:30:00",
            "duration_seconds": 3600.0,
        },
    ]


def test_session_detail_serializes_channels_and_spans(api_client, night):
    response = api_client.get(f"/api/v1/sessions/{night.id}")

    assert response.status_code == 200
    data = response.json()
    assert data["channel_stats"] == _EXPECTED_STATS
    assert data["mask_on_spans"] == _expected_spans(night.id)
    assert data["statistics"]["ahi_device"] == pytest.approx(3.1)


def test_session_without_channels_or_segments_serializes_nulls(
    api_client, test_device, test_session_factory
):
    session = test_session_factory(
        test_device.id, start_time=datetime(2025, 3, 1, 22, 0)
    )

    data = api_client.get(f"/api/v1/sessions/{session.id}").json()

    assert data["channel_stats"] is None
    assert data["mask_on_spans"] is None


def test_day_detail_serializes_channels_and_spans(api_client, night):
    response = api_client.get(f"/api/v1/days/{_NIGHT}")

    assert response.status_code == 200
    data = response.json()
    assert data["channel_stats"] == _EXPECTED_STATS
    assert data["mask_on_spans"] == _expected_spans(night.id)


def test_list_items_carry_no_channel_detail(api_client, night):
    days = api_client.get("/api/v1/days/").json()
    sessions = api_client.get("/api/v1/sessions/").json()

    assert not {"channel_stats", "mask_on_spans"} & set(days["items"][0])
    assert not {"channel_stats", "mask_on_spans"} & set(sessions["items"][0])


def test_periods_serialize_channel_percentiles(api_client, night):
    response = api_client.get("/api/v1/stats/periods", params={"period_type": "month"})

    assert response.status_code == 200
    [period] = response.json()
    assert period["device_flg_95th"] == 0.5
    assert period["snore_95th"] == 2.0


def test_trends_serialize_channel_series(api_client, night):
    response = api_client.get("/api/v1/stats/trends", params={"period_type": "month"})

    assert response.status_code == 200
    data = response.json()
    assert data["device_flg_95th"] == [["2025-03-01", 0.5]]
    assert data["snore_95th"] == [["2025-03-01", 2.0]]
