from datetime import datetime


class TestEventsRouter:
    def test_list_events_session_not_found(self, api_client):
        response = api_client.get("/api/v1/sessions/99999/events")
        assert response.status_code == 404

    def test_list_events_empty(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        response = api_client.get(f"/api/v1/sessions/{session.id}/events")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_events_with_data(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import Event

        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        event = Event(
            session_id=session.id,
            event_type="OA",
            start_time=datetime(2025, 1, 10, 23, 0),
            duration_seconds=15.0,
        )
        db_session.add(event)
        db_session.flush()

        response = api_client.get(f"/api/v1/sessions/{session.id}/events")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["event_type"] == "OA"
        assert data[0]["duration_seconds"] == 15.0
        assert data[0]["offset_seconds"] == 3600.0
        assert "spo2_drop" in data[0]
        assert "peak_flow_limitation" in data[0]

    def test_list_events_returns_null_for_missing_duration(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import Event

        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        db_session.add(
            Event(
                session_id=session.id,
                event_type="H",
                start_time=datetime(2025, 1, 10, 23, 0),
            )
        )
        db_session.flush()

        response = api_client.get(f"/api/v1/sessions/{session.id}/events")

        assert response.status_code == 200
        assert response.json()[0]["duration_seconds"] is None

    def test_list_events_filter_by_type(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import Event

        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        db_session.add(
            Event(
                session_id=session.id,
                event_type="OA",
                start_time=datetime(2025, 1, 10, 23, 0),
                duration_seconds=10.0,
            )
        )
        db_session.add(
            Event(
                session_id=session.id,
                event_type="H",
                start_time=datetime(2025, 1, 10, 23, 30),
                duration_seconds=12.0,
            )
        )
        db_session.flush()

        response = api_client.get(f"/api/v1/sessions/{session.id}/events?event_type=OA")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["event_type"] == "OA"

    def test_match_route_is_gone(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        response = api_client.get(f"/api/v1/sessions/{session.id}/events/match")
        assert response.status_code == 404
        assert response.json()["detail"] == "Not Found"
