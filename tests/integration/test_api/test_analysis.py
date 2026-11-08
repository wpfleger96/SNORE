from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.orm import Session

import snore.api.analysis_jobs as aj_store

from snore.database import models


def _assert_utc_timestamp(value: str) -> None:
    assert datetime.fromisoformat(value).utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Fixture: clean analysis job state between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_analysis_jobs():
    """Reset global analysis job state before and after each test."""
    aj_store._all_jobs.clear()
    aj_store._queue.clear()
    yield
    aj_store._all_jobs.clear()
    aj_store._queue.clear()


class TestAnalysisSessionsRouter:
    def test_list_analysis_sessions_empty(self, api_client):
        response = api_client.get("/api/v1/analysis/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_list_analysis_sessions_with_data(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import Day

        day = Day(
            device_id=test_device.id, date=datetime(2025, 1, 10).date(), session_count=1
        )
        db_session.add(day)
        db_session.flush()

        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        session.day_id = day.id
        db_session.flush()

        response = api_client.get("/api/v1/analysis/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["items"][0]["session_id"] == session.id
        assert data["items"][0]["has_analysis"] is False

    def test_get_analysis_not_found(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        response = api_client.get(f"/api/v1/sessions/{session.id}/analysis")
        assert response.status_code == 404

    def test_get_analysis_pre_count_payload_returns_derived_counts(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import AnalysisResult

        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )

        def apnea(start: float) -> dict[str, object]:
            return {
                "start_time": start,
                "end_time": start + 12.0,
                "duration": 12.0,
                "event_type": "OA",
                "flow_reduction": 0.95,
                "confidence": 0.9,
                "baseline_flow": 30.0,
            }

        # Stored before the count fields existed: no count keys, no ``reras``,
        # a stale ``apnea_count``, and JSON's string keys in class_distribution.
        db_session.add(
            AnalysisResult(
                session_id=session.id,
                timestamp_start=datetime(2025, 1, 10, 22, 0),
                timestamp_end=datetime(2025, 1, 11, 6, 0),
                programmatic_result_json={
                    "session_id": session.id,
                    "session_duration_hours": 8.0,
                    "total_breaths": 10,
                    "machine_events": [],
                    "mode_results": {
                        "aasm": {
                            "mode_name": "aasm",
                            "apneas": [apnea(100.0), apnea(400.0)],
                            "hypopneas": [],
                            "ahi": 0.25,
                            "rdi": 0.25,
                            "apnea_count": 99,
                        }
                    },
                    "flow_analysis": {
                        "total_breaths": 10,
                        "class_distribution": {"1": 7, "4": 3},
                        "flow_limitation_index": 0.3,
                        "average_confidence": 0.8,
                        "patterns": [],
                    },
                },
            )
        )
        db_session.flush()

        response = api_client.get(f"/api/v1/sessions/{session.id}/analysis")

        assert response.status_code == 200
        data = response.json()
        aasm = data["mode_results"]["aasm"]
        assert (aasm["apnea_count"], aasm["hypopnea_count"], aasm["rera_count"]) == (
            2,
            0,
            0,
        )
        assert data["flow_analysis"] == {
            "total_breaths": 10,
            "class_distribution": {"1": 7, "4": 3},
            "flow_limitation_index": 0.3,
            "average_confidence": 0.8,
            "patterns": [],
        }

    def test_delete_analysis_empty(self, api_client):
        # Use request() since TestClient.delete() doesn't support body
        response = api_client.request(
            "DELETE",
            "/api/v1/analysis",
            json={"session_ids": []},
        )
        assert response.status_code == 200
        assert response.json()["deleted_count"] == 0


class TestAnalysisDeletePreview:
    def test_delete_preview_no_filters(self, api_client):
        response = api_client.get("/api/v1/analysis/delete-preview")
        assert response.status_code == 200
        data = response.json()
        assert data["sessions_with_analysis"] == 0
        assert data["records_to_delete"] == 0

    def test_delete_preview_with_ids(
        self, api_client, db_session, test_device, test_session_factory
    ):
        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        response = api_client.get(
            f"/api/v1/analysis/delete-preview?session_ids={session.id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sessions_with_analysis"] == 0
        assert data["records_to_delete"] == 0


class TestBatchAnalysis:
    def test_both_dates_none_returns_400(self, api_client):
        response = api_client.post("/api/v1/analysis/batch", json={})
        assert response.status_code == 400
        assert (
            "from_date" in response.json()["detail"]
            or "to_date" in response.json()["detail"]
        )

    def test_from_date_only_not_400(self, api_client):
        # from_date alone must not cause a 400; 202 (queued) or 422 (no sessions
        # or invalid mode) are both valid depending on database state.
        response = api_client.post(
            "/api/v1/analysis/batch", json={"from_date": "2025-01-01"}
        )
        assert response.status_code != 400

    def test_from_date_with_sessions_returns_202(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import Day

        day = Day(
            device_id=test_device.id, date=datetime(2025, 1, 10).date(), session_count=1
        )
        db_session.add(day)
        db_session.flush()

        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 10, 22, 0)
        )
        session.day_id = day.id
        db_session.flush()

        response = api_client.post(
            "/api/v1/analysis/batch", json={"from_date": "2025-01-01"}
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["session_count"] >= 1

    def test_empty_range_returns_422(self, api_client):
        # A date range with no sessions returns 422.
        response = api_client.post(
            "/api/v1/analysis/batch",
            json={"from_date": "2099-01-01", "to_date": "2099-01-31"},
        )
        assert response.status_code == 422
        assert "No sessions found" in response.json()["detail"]

    def test_invalid_primary_mode_returns_422(self, api_client):
        # primary_mode not in modes → 422 at the endpoint before any DB call.
        response = api_client.post(
            "/api/v1/analysis/batch",
            json={
                "from_date": "2025-01-01",
                "modes": ["aasm"],
                "primary_mode": "resmed",
            },
        )
        assert response.status_code == 422

    def test_only_missing_no_dates_returns_202_for_unanalyzed_flow_sessions(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import AnalysisResult, Day, Waveform

        # session_a: has flow waveform, no analysis → should be included
        day_a = Day(
            device_id=test_device.id,
            date=datetime(2025, 3, 1).date(),
            session_count=1,
        )
        db_session.add(day_a)
        db_session.flush()
        session_a = test_session_factory(
            test_device.id, start_time=datetime(2025, 3, 1, 22, 0)
        )
        session_a.day_id = day_a.id
        db_session.flush()
        db_session.add(
            Waveform(
                session_id=session_a.id,
                waveform_type="flow",
                sample_rate=1.0,
                data_blob=b"",
            )
        )

        # session_b: no flow waveform, no analysis → should be excluded
        day_b = Day(
            device_id=test_device.id,
            date=datetime(2025, 3, 2).date(),
            session_count=1,
        )
        db_session.add(day_b)
        db_session.flush()
        session_b = test_session_factory(
            test_device.id, start_time=datetime(2025, 3, 2, 22, 0)
        )
        session_b.day_id = day_b.id
        db_session.flush()

        # session_c: has flow waveform and an analysis result → should be excluded
        day_c = Day(
            device_id=test_device.id,
            date=datetime(2025, 3, 3).date(),
            session_count=1,
        )
        db_session.add(day_c)
        db_session.flush()
        session_c = test_session_factory(
            test_device.id, start_time=datetime(2025, 3, 3, 22, 0)
        )
        session_c.day_id = day_c.id
        db_session.flush()
        db_session.add(
            Waveform(
                session_id=session_c.id,
                waveform_type="flow",
                sample_rate=1.0,
                data_blob=b"",
            )
        )
        db_session.add(
            AnalysisResult(
                session_id=session_c.id,
                timestamp_start=datetime(2025, 3, 3, 22, 0),
                timestamp_end=datetime(2025, 3, 4, 6, 0),
                programmatic_result_json={},
            )
        )
        db_session.flush()

        response = api_client.post("/api/v1/analysis/batch", json={"only": "missing"})
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["session_count"] == 1

    def test_no_dates_and_no_only_returns_400(self, api_client):
        response = api_client.post("/api/v1/analysis/batch", json={})
        assert response.status_code == 400
        assert (
            "from_date" in response.json()["detail"]
            or "to_date" in response.json()["detail"]
        )

    def test_only_missing_with_date_range_filters_compose(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.database.models import Day, Waveform

        # session_in_range: flow waveform, no analysis, within date range → included
        day_in = Day(
            device_id=test_device.id,
            date=datetime(2025, 4, 10).date(),
            session_count=1,
        )
        db_session.add(day_in)
        db_session.flush()
        session_in = test_session_factory(
            test_device.id, start_time=datetime(2025, 4, 10, 22, 0)
        )
        session_in.day_id = day_in.id
        db_session.flush()
        db_session.add(
            Waveform(
                session_id=session_in.id,
                waveform_type="flow",
                sample_rate=1.0,
                data_blob=b"",
            )
        )

        # session_out_of_range: flow waveform, no analysis, outside date range → excluded
        day_out = Day(
            device_id=test_device.id,
            date=datetime(2025, 5, 1).date(),
            session_count=1,
        )
        db_session.add(day_out)
        db_session.flush()
        session_out = test_session_factory(
            test_device.id, start_time=datetime(2025, 5, 1, 22, 0)
        )
        session_out.day_id = day_out.id
        db_session.flush()
        db_session.add(
            Waveform(
                session_id=session_out.id,
                waveform_type="flow",
                sample_rate=1.0,
                data_blob=b"",
            )
        )
        db_session.flush()

        response = api_client.post(
            "/api/v1/analysis/batch",
            json={
                "only": "missing",
                "from_date": "2025-04-01",
                "to_date": "2025-04-30",
            },
        )
        assert response.status_code == 202
        data = response.json()
        assert data["session_count"] == 1

    @staticmethod
    def _add_flow_session_with_analysis(
        db_session: Session,
        device_id: int,
        session_factory: Callable[..., models.Session],
        day_date: date,
        engine_versions_json: dict[str, Any],
    ) -> models.Session:
        from snore.database.models import AnalysisResult, Day, Waveform

        day = Day(device_id=device_id, date=day_date, session_count=1)
        db_session.add(day)
        db_session.flush()
        start = datetime.combine(day_date, datetime.min.time()).replace(hour=22)
        session = session_factory(device_id, start_time=start)
        session.day_id = day.id
        db_session.flush()
        db_session.add(
            Waveform(
                session_id=session.id,
                waveform_type="flow",
                sample_rate=1.0,
                data_blob=b"",
            )
        )
        db_session.add(
            AnalysisResult(
                session_id=session.id,
                timestamp_start=start,
                timestamp_end=start + timedelta(hours=8),
                programmatic_result_json={},
                engine_versions_json=engine_versions_json,
            )
        )
        db_session.flush()
        return session

    def test_only_stale_no_dates_returns_202_for_outdated_sessions(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.analysis.shared.versioning import (
            AlgorithmIdentity,
            AlgoVersions,
            AnalysisRunMetadata,
        )

        current = AlgoVersions(
            identity=AlgorithmIdentity.current(),
            run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
        ).model_dump()
        self._add_flow_session_with_analysis(
            db_session, test_device.id, test_session_factory, date(2025, 6, 1), current
        )
        self._add_flow_session_with_analysis(
            db_session, test_device.id, test_session_factory, date(2025, 6, 2), {}
        )

        response = api_client.post("/api/v1/analysis/batch", json={"only": "stale"})

        assert response.status_code == 202
        assert response.json()["session_count"] == 1

    def test_only_stale_with_nothing_outdated_returns_422(
        self, api_client, db_session, test_device, test_session_factory
    ):
        from snore.analysis.shared.versioning import (
            AlgorithmIdentity,
            AlgoVersions,
            AnalysisRunMetadata,
        )

        current = AlgoVersions(
            identity=AlgorithmIdentity.current(),
            run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
        ).model_dump()
        self._add_flow_session_with_analysis(
            db_session, test_device.id, test_session_factory, date(2025, 6, 1), current
        )

        response = api_client.post("/api/v1/analysis/batch", json={"only": "stale"})

        assert response.status_code == 422
        assert response.json()["detail"] == "No sessions with outdated analysis"

    def test_only_unknown_value_returns_422(self, api_client):
        response = api_client.post("/api/v1/analysis/batch", json={"only": "all"})

        assert response.status_code == 422


class TestAnalysisJobsAPI:
    def test_list_jobs_returns_empty_initially(self, api_client):
        response = api_client.get("/api/v1/analysis/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
        assert data["jobs"] == []

    def test_delete_unknown_job_returns_404(self, api_client):
        response = api_client.delete("/api/v1/analysis/jobs/does-not-exist")
        assert response.status_code == 404

    def test_delete_queued_job_returns_204_and_cancels(self, api_client):
        # Enqueue a job directly via the module so we control its state.
        job = aj_store.enqueue(
            profile_id=1,
            session_ids=[1, 2],
            source=aj_store.AnalysisJobSource.BATCH,
            owner_user_id=None,  # Matches local-mode actor (user_id may be None)
        )
        assert job is not None

        response = api_client.delete(f"/api/v1/analysis/jobs/{job.job_id}")
        assert response.status_code == 204
        assert job.state == aj_store.AnalysisJobState.CANCELLED

    def test_delete_terminal_job_returns_409(self, api_client):
        """DELETE /analysis/jobs/{job_id} for a job already in CANCELLED state returns 409."""
        job = aj_store.enqueue(
            profile_id=1,
            session_ids=[1, 2],
            source=aj_store.AnalysisJobSource.BATCH,
            owner_user_id=None,
        )
        assert job is not None
        # Drive the job to a terminal state by cancelling it directly.
        aj_store.cancel_job(job.job_id)
        assert job.state == aj_store.AnalysisJobState.CANCELLED

        response = api_client.delete(f"/api/v1/analysis/jobs/{job.job_id}")
        assert response.status_code == 409
        assert "cannot be cancelled" in response.json()["detail"]

    def test_list_jobs_includes_enqueued_job(self, api_client):
        job = aj_store.enqueue(
            profile_id=1,
            session_ids=[10],
            source=aj_store.AnalysisJobSource.BATCH,
            owner_user_id=None,
        )
        assert job is not None

        response = api_client.get("/api/v1/analysis/jobs")
        assert response.status_code == 200
        data = response.json()
        ids = [j["job_id"] for j in data["jobs"]]
        assert job.job_id in ids

        j = next(j for j in data["jobs"] if j["job_id"] == job.job_id)
        _assert_utc_timestamp(j["created_at"])
        assert j["started_at"] is None
        assert j["finished_at"] is None

    def test_list_jobs_includes_persisted_terminal_job(self, api_client, db_session):
        from snore.database.models import AnalysisJobRecord

        now = datetime.now(UTC)
        db_session.add(
            AnalysisJobRecord(
                job_id="persisted-analysis-job",
                source="batch",
                profile_id=1,
                owner_user_id=None,
                session_ids_json=[10],
                modes=["aasm"],
                primary_mode="aasm",
                store_results=True,
                state="succeeded",
                progress_completed=1,
                progress_total=1,
                error_message=None,
                created_at=now,
                started_at=now,
                finished_at=now,
                updated_at=now,
            )
        )
        db_session.commit()

        response = api_client.get("/api/v1/analysis/jobs")
        assert response.status_code == 200
        job = next(
            job
            for job in response.json()["jobs"]
            if job["job_id"] == "persisted-analysis-job"
        )
        _assert_utc_timestamp(job["created_at"])
        _assert_utc_timestamp(job["started_at"])
        _assert_utc_timestamp(job["finished_at"])


# ---------------------------------------------------------------------------
# Route-level two-profile isolation: DELETE /analysis must 404 on foreign IDs
# ---------------------------------------------------------------------------


class TestDeleteAnalysisCrossProfileIsolation:
    """Route-level proof that DELETE /api/v1/analysis returns 404 when any
    requested session ID belongs to a different profile -- foreign analysis rows
    must survive.
    """

    def _make_client_as_profile(
        self, async_db_session: object, db_session: object, profile_id: int
    ) -> object:
        """Return a TestClient whose actor is locked to *profile_id*."""
        from collections.abc import AsyncGenerator
        from typing import Annotated

        from fastapi import Depends
        from fastapi.testclient import TestClient
        from sqlalchemy.ext.asyncio import AsyncSession

        from snore.api.app import create_app
        from snore.api.deps import get_actor, get_db, get_db_immediate
        from snore.auth.actor import ActorContext, AuthMode, Role

        actor = ActorContext(
            user_id=1,
            profile_id=profile_id,
            role=Role.ADMIN,
            mode=AuthMode.LOCAL,
        )

        app = create_app()

        async def override_get_db():
            async with async_db_session.begin():
                yield async_db_session

        async def override_get_db_immediate(
            db: Annotated[AsyncSession, Depends(get_db)],
        ) -> AsyncGenerator[AsyncSession]:
            yield db

        async def override_get_actor():
            return actor

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_db_immediate] = override_get_db_immediate
        app.dependency_overrides[get_actor] = override_get_actor
        client = TestClient(app, raise_server_exceptions=True)
        return client

    def test_delete_analysis_foreign_session_ids_returns_404(
        self, async_db_session, db_session
    ):
        """Profile A requesting deletion of profile B's analysis -> 404.

        The foreign AnalysisResult row must survive -- no rows are deleted.
        """
        from datetime import UTC, datetime

        from snore.database.models import AnalysisResult, Device, Profile, Session, User

        # --- Seed two profiles ---
        user_a = User(canonical_email="an_iso_a@test", role="admin")
        user_b = User(canonical_email="an_iso_b@test", role="admin")
        db_session.add(user_a)
        db_session.add(user_b)
        db_session.flush()

        profile_a = Profile(user_id=user_a.id, name="A")
        profile_b = Profile(user_id=user_b.id, name="B")
        db_session.add(profile_a)
        db_session.add(profile_b)
        db_session.flush()

        dev_b = Device(
            profile_id=profile_b.id,
            manufacturer="Mfr",
            model="Model",
            serial_number="AN_ISO_DEV_B",
        )
        db_session.add(dev_b)
        db_session.flush()

        now = datetime.now(UTC)
        foreign_session = Session(
            device_id=dev_b.id,
            device_session_id="an_iso_foreign_session",
            start_time=datetime(2025, 3, 1, 22, 0),
            end_time=datetime(2025, 3, 2, 6, 0),
            duration_seconds=28800.0,
        )
        db_session.add(foreign_session)
        db_session.flush()

        foreign_ar = AnalysisResult(
            session_id=foreign_session.id,
            timestamp_start=foreign_session.start_time,
            timestamp_end=foreign_session.end_time,
            programmatic_result_json={},
            processing_time_ms=0,
            created_at=now,
        )
        db_session.add(foreign_ar)
        db_session.flush()
        foreign_session_id = foreign_session.id
        foreign_ar_id = foreign_ar.id

        # --- Make request as profile A with profile B's session ID ---
        client = self._make_client_as_profile(
            async_db_session, db_session, profile_a.id
        )
        response = client.request(
            "DELETE",
            "/api/v1/analysis",
            json={"session_ids": [foreign_session_id]},
        )

        assert response.status_code == 404, (
            f"Expected 404 for foreign session ID; got {response.status_code}: "
            f"{response.text}"
        )

        # Foreign AnalysisResult must still exist.
        surviving = db_session.get(AnalysisResult, foreign_ar_id)
        assert surviving is not None, (
            "Foreign AnalysisResult must survive a cross-profile DELETE attempt"
        )

    def test_delete_analysis_mixed_own_and_foreign_returns_404_nothing_deleted(
        self, async_db_session, db_session
    ):
        """Mixed list of own + foreign session IDs -> 404; neither row is deleted."""
        from datetime import UTC, datetime

        from snore.database.models import AnalysisResult, Device, Profile, Session, User

        user_a = User(canonical_email="an_mix_a@test", role="admin")
        user_b = User(canonical_email="an_mix_b@test", role="admin")
        db_session.add(user_a)
        db_session.add(user_b)
        db_session.flush()

        profile_a = Profile(user_id=user_a.id, name="A")
        profile_b = Profile(user_id=user_b.id, name="B")
        db_session.add(profile_a)
        db_session.add(profile_b)
        db_session.flush()

        dev_a = Device(
            profile_id=profile_a.id,
            manufacturer="Mfr",
            model="Model",
            serial_number="AN_MIX_DEV_A",
        )
        dev_b = Device(
            profile_id=profile_b.id,
            manufacturer="Mfr",
            model="Model",
            serial_number="AN_MIX_DEV_B",
        )
        db_session.add(dev_a)
        db_session.add(dev_b)
        db_session.flush()

        now = datetime.now(UTC)
        own_session = Session(
            device_id=dev_a.id,
            device_session_id="an_mix_own_session",
            start_time=datetime(2025, 4, 1, 22, 0),
            end_time=datetime(2025, 4, 2, 6, 0),
            duration_seconds=28800.0,
        )
        foreign_session = Session(
            device_id=dev_b.id,
            device_session_id="an_mix_foreign_session",
            start_time=datetime(2025, 4, 1, 22, 0),
            end_time=datetime(2025, 4, 2, 6, 0),
            duration_seconds=28800.0,
        )
        db_session.add(own_session)
        db_session.add(foreign_session)
        db_session.flush()

        own_ar = AnalysisResult(
            session_id=own_session.id,
            timestamp_start=own_session.start_time,
            timestamp_end=own_session.end_time,
            programmatic_result_json={},
            processing_time_ms=0,
            created_at=now,
        )
        foreign_ar = AnalysisResult(
            session_id=foreign_session.id,
            timestamp_start=foreign_session.start_time,
            timestamp_end=foreign_session.end_time,
            programmatic_result_json={},
            processing_time_ms=0,
            created_at=now,
        )
        db_session.add(own_ar)
        db_session.add(foreign_ar)
        db_session.flush()
        own_session_id = own_session.id
        foreign_session_id = foreign_session.id
        own_ar_id = own_ar.id
        foreign_ar_id = foreign_ar.id

        client = self._make_client_as_profile(
            async_db_session, db_session, profile_a.id
        )
        response = client.request(
            "DELETE",
            "/api/v1/analysis",
            json={"session_ids": [own_session_id, foreign_session_id]},
        )

        assert response.status_code == 404, (
            f"Expected 404 for mixed list; got {response.status_code}: {response.text}"
        )

        # Both analysis rows must survive -- no partial delete on 404.
        assert db_session.get(AnalysisResult, own_ar_id) is not None, (
            "Own AnalysisResult must survive"
        )
        assert db_session.get(AnalysisResult, foreign_ar_id) is not None, (
            "Foreign AnalysisResult must survive"
        )
