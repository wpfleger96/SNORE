"""Unit tests for the database reset and data deletion job.

Grouped as:

- In-memory state machine: single-job admission, cancel semantics, TTL reaping.
- Execution against a real migrated SQLite DB through the production
  ``session_scope()``, whose foreign keys and transaction recipe the API tests'
  shared session does not have.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from collections.abc import Awaitable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from sqlalchemy import func, select

from snore.api import reset_jobs
from snore.api.jobs.core import JOB_TTL_SECONDS
from snore.api.reset_jobs import ResetJob, ResetJobKind, ResetJobState
from snore.auth.invite_tokens import hash_invite_token
from snore.database import models
from snore.database.session import cleanup_database, init_database, session_scope
from snore.services.database_service import DatabaseService
from snore.services.schemas import DeleteDataResult, ResetResult
from tests.helpers.reset_jobs import clear_reset_jobs


@pytest.fixture(autouse=True)
def clean_reset_jobs() -> Iterator[None]:
    clear_reset_jobs()
    yield
    clear_reset_jobs()


def _enqueue(
    kind: ResetJobKind = ResetJobKind.RESET, *, include_accounts: bool = False
) -> ResetJob:
    job = reset_jobs.enqueue(
        kind=kind, user_id=1, db_path="", include_accounts=include_accounts
    )
    assert job is not None
    return job


# ---------------------------------------------------------------------------
# 1. In-memory state machine
# ---------------------------------------------------------------------------


def test_enqueue_returns_queued_job_with_unguessable_token():
    job = _enqueue()

    assert job.state == ResetJobState.QUEUED
    assert reset_jobs.get_job(job.job_id) is job
    assert len(job.job_id) >= 43  # 32 random bytes, base64url


def test_enqueue_refused_while_job_queued():
    _enqueue()

    assert (
        reset_jobs.enqueue(kind=ResetJobKind.DELETE_DATA, user_id=2, db_path="") is None
    )


def test_enqueue_refused_while_job_running():
    _enqueue().try_start()

    assert reset_jobs.enqueue(kind=ResetJobKind.RESET, user_id=2, db_path="") is None


@pytest.mark.parametrize("succeeded", [True, False])
def test_enqueue_accepted_after_job_finishes(succeeded: bool) -> None:
    first = _enqueue()
    first.try_start()
    first.finish(succeeded=succeeded, error_message="boom")

    assert reset_jobs.enqueue(kind=ResetJobKind.RESET, user_id=2, db_path="")


def test_has_active_job_true_until_job_finishes():
    assert reset_jobs.has_active_job() is False

    job = _enqueue()
    assert reset_jobs.has_active_job() is True

    job.try_start()
    assert reset_jobs.has_active_job() is True

    job.finish(succeeded=True)
    assert reset_jobs.has_active_job() is False


def test_finish_keeps_success_despite_pending_cancel():
    """Committed data is really gone, so a late cancel must not report otherwise."""
    job = _enqueue()
    job.try_start()
    job.try_cancel()

    job.finish(succeeded=True)

    assert job.state == ResetJobState.SUCCEEDED


def test_finish_keeps_failure_message_despite_pending_cancel():
    job = _enqueue()
    job.try_start()
    job.try_cancel()

    job.finish(succeeded=False, error_message="boom")

    assert job.state == ResetJobState.FAILED
    assert job.error_message == "boom"


def test_cancel_before_start_prevents_any_work():
    job = _enqueue()
    job.try_cancel()

    with patch.object(reset_jobs, "_run") as run:
        reset_jobs._execute_job(job)

    run.assert_not_called()
    assert job.state == ResetJobState.CANCELLED


def test_shutdown_cancels_queued_job():
    job = _enqueue()

    reset_jobs.shutdown(timeout=1.0)

    assert job.state == ResetJobState.CANCELLED
    assert reset_jobs.has_active_job() is False


def test_terminal_job_reaped_after_ttl():
    job = _enqueue()
    job.try_start()
    job.finish(succeeded=True)
    job._finished_at = time.monotonic() - JOB_TTL_SECONDS - 1

    reset_jobs._reap_terminal()

    assert reset_jobs.get_job(job.job_id) is None


def test_recent_terminal_job_survives_reap():
    job = _enqueue()
    job.try_start()
    job.finish(succeeded=True)

    reset_jobs._reap_terminal()

    assert reset_jobs.get_job(job.job_id) is job


def test_to_dict_routes_result_to_the_field_for_its_kind():
    job = _enqueue(ResetJobKind.DELETE_DATA)
    result = DeleteDataResult(
        status="success",
        devices_deleted=1,
        import_jobs_deleted=0,
        profiles_processed=1,
        size_before_mb=2.0,
        size_after_mb=1.0,
    )
    job.set_result(result)

    snapshot = job.to_dict()

    assert snapshot["kind"] == "delete_data"
    assert snapshot["delete_data_result"] is result
    assert snapshot["reset_result"] is None


# ---------------------------------------------------------------------------
# 2. Execution against a real migrated DB
# ---------------------------------------------------------------------------


def _run(coro: Awaitable[Any]) -> Any:
    return asyncio.run(coro)


@pytest.fixture
def init_db(temp_db: Path) -> Iterator[None]:
    """Initialise the global database at a temp path; migrate + tear down."""
    _run(cleanup_database())
    _run(init_database(str(temp_db)))
    yield
    _run(cleanup_database())


async def _seed_user_with_session() -> tuple[int, str]:
    """Insert user → profile → device → session; return (user id, email)."""
    email = f"admin_{uuid.uuid4().hex[:8]}@test.local"
    async with session_scope(immediate=True) as db:
        user = models.User(canonical_email=email, role="admin")
        db.add(user)
        await db.flush()
        profile = models.Profile(user_id=user.id, name="Default")
        db.add(profile)
        await db.flush()
        device = models.Device(
            profile_id=profile.id,
            manufacturer="ResMed",
            model="AirSense 10",
            serial_number=uuid.uuid4().hex[:8],
        )
        db.add(device)
        await db.flush()
        db.add(
            models.Session(
                device_id=device.id,
                device_session_id="s1",
                start_time=models.utc_now(),
                end_time=models.utc_now(),
                duration_seconds=0,
            )
        )
        return user.id, email


async def _count(model: type) -> int:
    async with session_scope() as db:
        return (await db.execute(select(func.count()).select_from(model))).scalar_one()


async def _invite_token_hashes() -> list[str]:
    async with session_scope() as db:
        return list((await db.execute(select(models.Invite.token_hash))).scalars())


def _queued_job(
    kind: ResetJobKind, user_id: int, *, include_accounts: bool = False
) -> ResetJob:
    job = reset_jobs.enqueue(
        kind=kind, user_id=user_id, db_path="", include_accounts=include_accounts
    )
    assert job is not None
    reset_jobs._queue.clear()
    return job


def test_factory_reset_wipes_accounts_and_mints_redeemable_invite(init_db):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id, include_accounts=True)

    reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    result = job.to_dict()["reset_result"]
    assert isinstance(result, ResetResult)
    assert _run(_count(models.User)) == 0
    assert _run(_count(models.Session)) == 0
    assert result.bootstrap_invite_url is not None
    token = result.bootstrap_invite_url.split("#", 1)[1]
    assert _run(_invite_token_hashes()) == [hash_invite_token(token)]


def test_data_only_reset_keeps_accounts_and_deletes_sleep_data(init_db):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id)

    reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    assert _run(_count(models.User)) == 1
    assert _run(_count(models.Profile)) == 1
    assert _run(_count(models.Session)) == 0


def test_delete_data_cascades_through_real_foreign_keys(init_db):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.DELETE_DATA, user_id)

    reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    result = job.to_dict()["delete_data_result"]
    assert isinstance(result, DeleteDataResult)
    assert result.devices_deleted == 1
    assert _run(_count(models.User)) == 1
    assert _run(_count(models.Session)) == 0


def test_factory_reset_without_caller_email_deletes_nothing(init_db):
    _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id=999_999, include_accounts=True)

    reset_jobs._execute_job(job)

    assert job.state == ResetJobState.FAILED
    assert job.error_message == reset_jobs._FAILURE_MESSAGE
    assert _run(_count(models.User)) == 1
    assert _run(_count(models.Session)) == 1


def test_worker_thread_runs_queued_job_to_completion(init_db):
    user_id, _ = _run(_seed_user_with_session())
    reset_jobs.start_worker()

    job = reset_jobs.enqueue(kind=ResetJobKind.RESET, user_id=user_id, db_path="")
    assert job is not None

    deadline = time.monotonic() + 10.0
    while not job.is_terminal and time.monotonic() < deadline:
        time.sleep(0.01)

    assert job.state == ResetJobState.SUCCEEDED


@pytest.mark.parametrize(
    "fail_delete",
    [False, True],
    ids=["success", "failure"],
)
def test_job_id_never_logged(init_db, caplog, fail_delete):
    """The job_id reads the invite URL, so no log line may ever contain it."""
    user_id, _ = _run(_seed_user_with_session())

    with caplog.at_level(logging.DEBUG):
        job = reset_jobs.enqueue(kind=ResetJobKind.RESET, user_id=user_id, db_path="")
        assert job is not None
        reset_jobs._queue.clear()
        if fail_delete:
            with patch.object(
                DatabaseService, "reset_all_data", side_effect=RuntimeError("boom")
            ):
                reset_jobs._execute_job(job)
        else:
            reset_jobs._execute_job(job)
        job._finished_at = time.monotonic() - JOB_TTL_SECONDS - 1
        reset_jobs._reap_terminal()

    assert reset_jobs.get_job(job.job_id) is None
    assert job.job_id not in caplog.text
