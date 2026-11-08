"""``POST /db/recompute-days`` and ``GET /db/recompute-days``: the admin day recompute job."""

from __future__ import annotations

import uuid

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from snore.api import recompute_jobs, reset_jobs
from snore.api.jobs.core import MAINTENANCE_CONFLICT_DETAIL
from snore.auth.actor import ActorContext, AuthMode, Role
from snore.database import models
from snore.services import day_recompute
from tests.helpers.api_client import make_test_client
from tests.helpers.recompute_jobs import RECOMPUTE_PATH, run_queued_recompute_job
from tests.helpers.reset_jobs import RESET_PATH


@pytest.fixture
def recompute_job_session(async_db_session, monkeypatch):
    """Point the recompute job at this test's database.

    The job opens ``session_scope`` itself rather than taking a request session,
    so ``make_test_client``'s ``get_db`` override never reaches it.
    """

    @asynccontextmanager
    async def _scope(*, immediate=False):
        async with async_db_session.begin():
            yield async_db_session

    monkeypatch.setattr(day_recompute, "session_scope", _scope)


@pytest.fixture
def member_client(async_db_session, db_session):
    member = models.User(
        canonical_email=f"member_{uuid.uuid4().hex[:8]}@test.local", role="member"
    )
    db_session.add(member)
    db_session.flush()
    actor = ActorContext(
        user_id=member.id, profile_id=999, role=Role.MEMBER, mode=AuthMode.LOCAL
    )
    return make_test_client(async_db_session, actor=actor)


def _seed_stale_day(db_session: OrmSession, device: models.Device, night: date) -> int:
    """Seed a Day whose aggregates disagree with its one 2-hour session; return its id."""
    stale = models.Day(
        device_id=device.id, date=night, session_count=5, total_therapy_hours=99.0
    )
    db_session.add(stale)
    db_session.flush()
    start = datetime.combine(night, datetime.min.time()) + timedelta(hours=22)
    db_session.add(
        models.Session(
            device_id=device.id,
            day_id=stale.id,
            device_session_id=f"sess_{uuid.uuid4().hex[:8]}",
            start_time=start,
            end_time=start + timedelta(hours=2),
            duration_seconds=7200,
        )
    )
    db_session.flush()
    return stale.id


def _seed_stale_day_and_orphan_day(
    db_session: OrmSession, device: models.Device
) -> int:
    """Seed a stale Day plus a Day no session references.  Returns the stale Day's id."""
    stale_id = _seed_stale_day(db_session, device, date(2025, 1, 10))
    db_session.add(
        models.Day(device_id=device.id, date=date(2025, 1, 11), session_count=1)
    )
    db_session.flush()
    return stale_id


class TestRecomputeDaysAccess:
    def test_member_cannot_start_a_recompute(self, member_client):
        response = member_client.post(RECOMPUTE_PATH)

        assert response.status_code == 403
        assert recompute_jobs.latest_job() is None

    def test_member_cannot_read_recompute_status(self, member_client):
        assert member_client.get(RECOMPUTE_PATH).status_code == 403

    def test_status_is_404_before_any_job(self, api_client):
        assert api_client.get(RECOMPUTE_PATH).status_code == 404


@pytest.mark.usefixtures("recompute_job_session")
class TestRecomputeDaysJob:
    def test_post_queues_job_without_touching_days(
        self, api_client, db_session, test_device
    ):
        stale_id = _seed_stale_day_and_orphan_day(db_session, test_device)

        response = api_client.post(RECOMPUTE_PATH)

        assert response.status_code == 202
        body = response.json()
        assert body["state"] == "queued"
        assert body["result"] is None
        assert db_session.get(models.Day, stale_id).session_count == 5

    def test_job_recomputes_days_and_reports_counts(
        self, api_client, db_session, test_device
    ):
        stale_id = _seed_stale_day_and_orphan_day(db_session, test_device)
        api_client.post(RECOMPUTE_PATH)

        run_queued_recompute_job()

        status = api_client.get(RECOMPUTE_PATH).json()
        assert status["state"] == "succeeded"
        assert status["result"] == {"days_recomputed": 1, "days_pruned": 1}
        assert (status["days_processed"], status["days_total"]) == (2, 2)
        db_session.expire_all()
        days = db_session.scalars(select(models.Day)).all()
        assert [(d.id, d.session_count) for d in days] == [(stale_id, 1)]
        assert days[0].total_therapy_hours == pytest.approx(2.0)

    def test_job_over_no_days_succeeds_with_zero_counts(self, api_client):
        api_client.post(RECOMPUTE_PATH)

        run_queued_recompute_job()

        status = api_client.get(RECOMPUTE_PATH).json()
        assert status["state"] == "succeeded"
        assert (status["days_processed"], status["days_total"]) == (0, 0)
        assert status["result"] == {"days_recomputed": 0, "days_pruned": 0}

    def test_cancel_while_running_stops_before_next_chunk(
        self, api_client, db_session, test_device, monkeypatch
    ):
        first_id = _seed_stale_day(db_session, test_device, date(2025, 1, 10))
        second_id = _seed_stale_day(db_session, test_device, date(2025, 1, 11))
        monkeypatch.setattr(day_recompute, "RECOMPUTE_DAYS_CHUNK_SIZE", 1)
        report_progress = recompute_jobs.RecomputeDaysJob.report_progress

        def _cancel_after_first_chunk(job, processed, total):
            # Stands in for shutdown cancelling the job while it runs.
            report_progress(job, processed, total)
            if processed == 1:
                job.try_cancel()

        monkeypatch.setattr(
            recompute_jobs.RecomputeDaysJob,
            "report_progress",
            _cancel_after_first_chunk,
        )
        api_client.post(RECOMPUTE_PATH)

        run_queued_recompute_job()

        status = api_client.get(RECOMPUTE_PATH).json()
        assert status["state"] == "cancelled"
        assert (status["days_processed"], status["days_total"]) == (1, 2)
        assert status["result"] is None
        db_session.expire_all()
        assert db_session.get(models.Day, first_id).session_count == 1
        assert db_session.get(models.Day, second_id).session_count == 5

    def test_status_is_not_cached(self, api_client):
        api_client.post(RECOMPUTE_PATH)

        response = api_client.get(RECOMPUTE_PATH)

        assert response.status_code == 200
        assert "no-store" in response.headers["cache-control"]

    def test_failure_reports_fixed_message_only(self, api_client):
        api_client.post(RECOMPUTE_PATH)

        with patch.object(
            recompute_jobs,
            "recompute_all_days",
            side_effect=RuntimeError("disk I/O error at /data/.snore/snore.db"),
        ):
            run_queued_recompute_job()

        status = api_client.get(RECOMPUTE_PATH).json()
        assert status["state"] == "failed"
        assert status["error_message"] == recompute_jobs._FAILURE_MESSAGE
        assert "/data" not in status["error_message"]

    def test_new_job_accepted_after_previous_one_finished(self, api_client):
        first = api_client.post(RECOMPUTE_PATH).json()["job_id"]
        run_queued_recompute_job()

        second = api_client.post(RECOMPUTE_PATH)

        assert second.status_code == 202
        assert second.json()["job_id"] != first
        assert (
            api_client.get(RECOMPUTE_PATH).json()["job_id"] == second.json()["job_id"]
        )


class TestRecomputeDaysConcurrency:
    def test_409_while_recompute_queued(self, api_client):
        api_client.post(RECOMPUTE_PATH)

        second = api_client.post(RECOMPUTE_PATH)

        assert second.status_code == 409
        assert second.json()["detail"] == MAINTENANCE_CONFLICT_DETAIL
        assert len(recompute_jobs._queue) == 1

    def test_409_while_recompute_running(self, api_client):
        first = api_client.post(RECOMPUTE_PATH).json()["job_id"]
        recompute_jobs._store.get(first).try_start()

        assert api_client.post(RECOMPUTE_PATH).status_code == 409

    def test_409_while_reset_active(self, api_client):
        assert api_client.post(RESET_PATH).status_code == 202

        response = api_client.post(RECOMPUTE_PATH)

        assert response.status_code == 409
        assert recompute_jobs.latest_job() is None

    def test_reset_409_while_recompute_active(self, api_client):
        assert api_client.post(RECOMPUTE_PATH).status_code == 202

        response = api_client.post(RESET_PATH)

        assert response.status_code == 409
        assert response.json()["detail"] == MAINTENANCE_CONFLICT_DETAIL
        assert not reset_jobs.has_active_job()
