"""Unit tests for the database reset and data deletion job.

Grouped as:

- In-memory state machine: single-job admission, cancel semantics, TTL reaping.
- Execution against a real migrated SQLite DB through the production
  ``session_scope()``, whose foreign keys and transaction recipe the API tests'
  shared session does not have.
- Result delivery and the pending-vacuum marker around the commit.
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
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from snore.api import reset_jobs
from snore.api.jobs.core import JOB_TTL_SECONDS
from snore.api.reset_jobs import ResetJob, ResetJobKind, ResetJobState
from snore.auth.invite_tokens import hash_invite_token
from snore.database import models
from snore.database.session import cleanup_database, init_database, session_scope
from snore.services import database_service
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


def _delete_data_result() -> DeleteDataResult:
    return DeleteDataResult(
        status="success",
        devices_deleted=1,
        import_jobs_deleted=0,
        profiles_processed=1,
        size_before_mb=2.0,
        size_after_mb=1.0,
    )


def test_to_dict_routes_result_to_the_field_for_its_kind():
    job = _enqueue(ResetJobKind.DELETE_DATA)
    job.try_start()
    result = _delete_data_result()
    job.set_result(result)
    job.finish(succeeded=True)

    snapshot = job.to_dict()

    assert snapshot["kind"] == "delete_data"
    assert snapshot["delete_data_result"] is result
    assert snapshot["reset_result"] is None


def test_to_dict_hides_result_until_job_is_terminal():
    """It is stored before the VACUUM, so size_after_mb is still a placeholder."""
    job = _enqueue(ResetJobKind.DELETE_DATA)
    job.try_start()
    job.set_result(_delete_data_result())

    assert job.to_dict()["delete_data_result"] is None


def test_repr_omits_job_id():
    """The job_id reads the invite URL, and a repr can reach a log line."""
    job = _enqueue()

    assert job.job_id not in repr(job)


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
    kind: ResetJobKind,
    user_id: int,
    *,
    include_accounts: bool = False,
    db_path: str = "",
) -> ResetJob:
    job = reset_jobs.enqueue(
        kind=kind, user_id=user_id, db_path=db_path, include_accounts=include_accounts
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


# ---------------------------------------------------------------------------
# 3. Result delivery and the pending-vacuum marker around the commit
# ---------------------------------------------------------------------------

# Only ever written into the marker: VACUUM is a no-op or a spy in these tests.
_JOB_DB_PATH = "job-target.db"


@pytest.fixture
def marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "vacuum.pending"
    monkeypatch.setattr(reset_jobs, "DEFAULT_VACUUM_PENDING_MARKER", path)
    return path


@pytest.fixture
def vacuum_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(reset_jobs, "_vacuum_background", calls.append)
    return calls


async def _seed_user_without_data() -> tuple[int, int]:
    """Insert a user and an empty profile; return (user id, profile id)."""
    async with session_scope(immediate=True) as db:
        user = models.User(
            canonical_email=f"admin_{uuid.uuid4().hex[:8]}@test.local", role="admin"
        )
        db.add(user)
        await db.flush()
        profile = models.Profile(user_id=user.id, name="Default")
        db.add(profile)
        await db.flush()
        return user.id, profile.id


async def _add_second_profile(user_id: int) -> tuple[int, int]:
    """Return (the user's existing profile id, the new profile's id)."""
    async with session_scope(immediate=True) as db:
        first_id = (
            await db.execute(
                select(models.Profile.id).where(models.Profile.user_id == user_id)
            )
        ).scalar_one()
        profile = models.Profile(user_id=user_id, name="Second")
        db.add(profile)
        await db.flush()
        return first_id, profile.id


async def _seed_import_job(user_id: int, profile_id: int) -> None:
    now = models.utc_now()
    async with session_scope(immediate=True) as db:
        db.add(
            models.ImportJobRecord(
                job_id=uuid.uuid4().hex,
                job_type="upload",
                owner_user_id=user_id,
                target_profile_id=profile_id,
                state="succeeded",
                file_count=1,
                created_at=now,
                finished_at=now,
                updated_at=now,
            )
        )


def test_job_stays_active_while_its_vacuum_runs(init_db, marker, monkeypatch):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id, db_path=_JOB_DB_PATH)
    seen: dict[str, object] = {}

    def _spy(db_path: str) -> None:
        seen["active"] = reset_jobs.has_active_job()
        seen["state"] = job.state
        seen["reset_result"] = job.to_dict()["reset_result"]

    monkeypatch.setattr(reset_jobs, "_vacuum_background", _spy)

    reset_jobs._execute_job(job)

    assert seen == {
        "active": True,
        "state": ResetJobState.RUNNING,
        "reset_result": None,
    }
    assert job.state == ResetJobState.SUCCEEDED


def test_delete_runs_in_an_immediate_transaction(init_db):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id)
    calls: list[bool] = []
    real_scope = session_scope

    def _spy(*, immediate: bool = False) -> Any:
        calls.append(immediate)
        return real_scope(immediate=immediate)

    with patch.object(reset_jobs, "session_scope", _spy):
        reset_jobs._execute_job(job)

    assert calls == [True]


def test_marker_exists_before_the_delete_commits(init_db, marker):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id, db_path=_JOB_DB_PATH)
    real_commit = AsyncSession.commit
    seen: list[bool] = []

    async def _spy(self: AsyncSession) -> None:
        seen.append(marker.exists())
        await real_commit(self)

    with patch.object(AsyncSession, "commit", _spy):
        reset_jobs._execute_job(job)

    assert seen == [True]


def test_cache_clear_failure_after_commit_still_delivers_invite(init_db):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id, include_accounts=True)

    with patch.object(
        database_service,
        "clear_waveform_array_cache",
        side_effect=RuntimeError("cache"),
    ):
        reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    result = job.to_dict()["reset_result"]
    assert isinstance(result, ResetResult)
    assert result.bootstrap_invite_url


@pytest.mark.parametrize(
    "error",
    [OSError("connection reset"), asyncio.CancelledError()],
    ids=["exception", "base_exception"],
)
def test_session_close_failure_after_commit_still_delivers_invite(
    init_db, marker, vacuum_calls, error
):
    """Leaving the session scope can fail after the commit, and the job carries on."""
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(
        ResetJobKind.RESET, user_id, include_accounts=True, db_path=_JOB_DB_PATH
    )
    real_commit = AsyncSession.commit
    real_close = AsyncSession.close
    committed: list[bool] = []

    async def _commit(self: AsyncSession) -> None:
        await real_commit(self)
        committed.append(True)

    async def _close(self: AsyncSession) -> None:
        await real_close(self)
        if committed:
            committed.clear()
            raise error

    with (
        patch.object(AsyncSession, "commit", _commit),
        patch.object(AsyncSession, "close", _close),
    ):
        reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    result = job.to_dict()["reset_result"]
    assert isinstance(result, ResetResult)
    assert result.bootstrap_invite_url
    assert vacuum_calls == [_JOB_DB_PATH]


def test_quarantine_failure_partway_restores_dirs_and_keeps_rows(init_db, tmp_path):
    """A rename failing on a later profile must not strand the earlier dirs."""
    from snore.services.profile_service import (  # noqa: PLC0415
        quarantine_profile_raw_dir,
    )

    user_id, _ = _run(_seed_user_with_session())
    first, second = _run(_add_second_profile(user_id))
    raw_root = tmp_path / "raw"
    for profile_id in (first, second):
        (raw_root / str(profile_id)).mkdir(parents=True)
        (raw_root / str(profile_id) / "data.edf").write_text("backup")

    def _quarantine(profile_id: int, root: Path) -> Path | None:
        if profile_id == second:
            raise OSError("rename failed")
        return quarantine_profile_raw_dir(profile_id, root)

    job = _queued_job(ResetJobKind.RESET, user_id)
    with (
        patch.object(reset_jobs, "_raw_root", return_value=raw_root),
        patch("snore.services.profile_service.quarantine_profile_raw_dir", _quarantine),
    ):
        reset_jobs._execute_job(job)

    assert job.state == ResetJobState.FAILED
    assert _run(_count(models.Profile)) == 2
    assert _run(_count(models.Session)) == 1
    assert (raw_root / str(first) / "data.edf").read_text() == "backup"
    assert not (raw_root / ".quarantine" / str(first)).exists()


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("boom"),
        OperationalError("BEGIN", {}, Exception("database is locked")),
    ],
    ids=["unexpected", "busy"],
)
def test_failure_before_commit_removes_the_marker_it_wrote(init_db, marker, error):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id, db_path=_JOB_DB_PATH)

    with patch.object(DatabaseService, "reset_all_data", side_effect=error):
        reset_jobs._execute_job(job)

    assert job.state == ResetJobState.FAILED
    assert not marker.exists()


def test_failure_before_commit_leaves_an_earlier_marker(init_db, marker):
    """A marker that predates the job means an earlier VACUUM is still owed."""
    marker.write_text("owed")
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.RESET, user_id, db_path=_JOB_DB_PATH)

    with patch.object(
        DatabaseService, "reset_all_data", side_effect=RuntimeError("boom")
    ):
        reset_jobs._execute_job(job)

    assert job.state == ResetJobState.FAILED
    assert marker.exists()


@pytest.mark.parametrize("kind", list(ResetJobKind))
def test_job_deleting_nothing_skips_vacuum_and_removes_marker(
    init_db, marker, vacuum_calls, kind
):
    user_id, _ = _run(_seed_user_without_data())
    job = _queued_job(kind, user_id, db_path=_JOB_DB_PATH)

    with patch.object(reset_jobs, "file_size_mb", side_effect=[3.0, 1.0]):
        reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    assert vacuum_calls == []
    assert not marker.exists()
    assert job.result is not None
    assert job.result.size_after_mb == 1.0


@pytest.mark.parametrize("kind", list(ResetJobKind))
def test_job_deleting_nothing_leaves_an_earlier_marker(
    init_db, marker, vacuum_calls, kind
):
    marker.write_text("owed")
    user_id, _ = _run(_seed_user_without_data())
    job = _queued_job(kind, user_id, db_path=_JOB_DB_PATH)

    reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    assert vacuum_calls == []
    assert marker.exists()


def test_delete_data_deleting_a_device_vacuums(init_db, marker, vacuum_calls):
    user_id, _ = _run(_seed_user_with_session())
    job = _queued_job(ResetJobKind.DELETE_DATA, user_id, db_path=_JOB_DB_PATH)

    reset_jobs._execute_job(job)

    assert job.state == ResetJobState.SUCCEEDED
    assert vacuum_calls == [_JOB_DB_PATH]
    assert marker.exists(), "_vacuum_background clears the marker, not the job"


def test_delete_data_deleting_only_import_jobs_vacuums(init_db, marker, vacuum_calls):
    user_id, profile_id = _run(_seed_user_without_data())
    _run(_seed_import_job(user_id, profile_id))
    job = _queued_job(ResetJobKind.DELETE_DATA, user_id, db_path=_JOB_DB_PATH)

    reset_jobs._execute_job(job)

    assert vacuum_calls == [_JOB_DB_PATH]
