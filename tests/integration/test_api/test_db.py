import asyncio
import sqlite3
import time
import typing
import uuid

from collections.abc import AsyncIterator, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session as SyncSession
from sqlalchemy.pool import NullPool

from snore.api import reset_jobs
from snore.api.app import create_app
from snore.api.config import load_config, set_config
from snore.auth.actor import ActorContext, AuthMode, Role
from snore.auth.session_cookie import encode_session
from snore.database import models
from snore.database.models import Base
from snore.database.session import cleanup_database, init_database, session_scope
from snore.services.database_service import DatabaseService
from snore.services.database_service import _vacuum_background as _REAL_VACUUM_BG
from snore.services.waveform_service import _waveform_array_cache
from tests.helpers.api_client import make_test_client
from tests.helpers.reset_jobs import (
    DELETE_DATA_PATH,
    RESET_PATH,
    STATUS_PATH,
    post_and_run_job,
    read_status,
    run_queued_job,
    run_reset,
    token_header,
)
from tests.integration.test_api.conftest import _multiuser_env


@contextmanager
def _patch_target(temp_db: object, client: TestClient) -> Generator[None]:
    """Context manager that installs a dependency override on *client*'s app.

    Uses app.dependency_overrides so FastAPI's Depends(_get_target) captures
    the replacement at request time, not at route-definition time.
    """
    from snore.api.routers.db import _get_target
    from snore.database.target import DatabaseTarget

    target = DatabaseTarget.from_url(str(temp_db))
    app = client.app
    app.dependency_overrides[_get_target] = lambda: target
    try:
        yield
    finally:
        del app.dependency_overrides[_get_target]


def _member_actor(user_id: int) -> ActorContext:
    return ActorContext(
        user_id=user_id,
        profile_id=999,
        role=Role.MEMBER,
        mode=AuthMode.LOCAL,
    )


def _admin_actor(user_id: int, profile_id: int = 1) -> ActorContext:
    return ActorContext(
        user_id=user_id,
        profile_id=profile_id,
        role=Role.ADMIN,
        mode=AuthMode.MULTIUSER,
    )


def _demo_actor(user_id: int) -> ActorContext:
    return ActorContext(
        user_id=user_id,
        profile_id=999,
        role=Role.DEMO,
        mode=AuthMode.MULTIUSER,
    )


def _session_count(db_session: SyncSession) -> int:
    return db_session.execute(
        select(func.count()).select_from(models.Session)
    ).scalar_one()


class TestDbStats:
    def test_stats_excludes_db_path(self, api_client, temp_db):
        with _patch_target(temp_db, api_client):
            response = api_client.get("/api/v1/db/stats")
        data = response.json()
        assert "db_path" not in data

    def test_stats_empty_db_counts_zero(self, api_client, temp_db):
        with _patch_target(temp_db, api_client):
            response = api_client.get("/api/v1/db/stats")
        data = response.json()
        assert data["session_count"] == 0
        assert data["device_count"] == 0

    def test_stats_counts_after_insert(
        self, api_client, db_session, test_device, test_session_factory, temp_db
    ):
        test_session_factory(test_device.id, start_time=datetime(2025, 1, 1, 22, 0))
        db_session.commit()
        with _patch_target(temp_db, api_client):
            response = api_client.get("/api/v1/db/stats")
        data = response.json()
        assert data["device_count"] >= 1
        assert data["session_count"] >= 1

    def test_admin_gets_200(self, api_client, temp_db):
        with _patch_target(temp_db, api_client):
            response = api_client.get("/api/v1/db/stats")
        assert response.status_code == 200

    def test_member_gets_403(self, async_db_session, db_session, temp_db):
        member = models.User(
            canonical_email=f"member_{uuid.uuid4().hex[:8]}@test.local",
            role="member",
        )
        db_session.add(member)
        db_session.flush()
        client = make_test_client(async_db_session, actor=_member_actor(member.id))
        with _patch_target(temp_db, client):
            response = client.get("/api/v1/db/stats")
        assert response.status_code == 403


class TestDbVacuum:
    def test_vacuum_status_is_success(self, api_client, tmp_path):
        import os
        import sqlite3

        # VACUUM requires exclusive file access.  The test client's async_db_session
        # holds an open connection to temp_db, so we target a separate fresh SQLite
        # file that has no other connections.  This also proves the dependency
        # override is in effect: the real database is ~4 GB, our file is tiny.
        vacuum_db = tmp_path / "vacuum_only.db"
        sqlite3.connect(str(vacuum_db)).close()

        from snore.api.routers.db import _get_target
        from snore.database.target import DatabaseTarget

        vt = DatabaseTarget.from_url(str(vacuum_db))
        api_client.app.dependency_overrides[_get_target] = lambda: vt
        try:
            response = api_client.post("/api/v1/db/vacuum")
        finally:
            del api_client.app.dependency_overrides[_get_target]

        data = response.json()
        assert data["status"] == "success"
        # vacuum_sqlite() measures the file size after VACUUM and returns it as
        # size_after_mb.  Compare against the actual on-disk size to verify the
        # override sent VACUUM to vacuum_db, not the real ~/.snore/snore.db.
        expected_mb = os.path.getsize(str(vacuum_db)) / (1024 * 1024)
        assert data["size_after_mb"] == pytest.approx(expected_mb, abs=0.01), (
            f"vacuum targeted the wrong file: expected vacuum_db ({expected_mb:.6f} MB) "
            f"but got size_after_mb={data['size_after_mb']:.6f} MB — "
            "dependency override may have silently stopped working"
        )


# ---------------------------------------------------------------------------
# Reset — local mode (backward compat) and multiuser mode
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_job_session")
class TestDbReset:
    def test_reset_status_is_success(self, api_client, temp_db):
        with _patch_target(temp_db, api_client):
            result = run_reset(api_client)
        assert result["status"] == "success"

    def test_post_returns_202_and_deletes_only_in_the_job(
        self, api_client, db_session, test_device, test_session_factory, temp_db
    ):
        """The request returns a token at once; the rows go when the job runs."""
        test_session_factory(test_device.id, start_time=datetime(2025, 1, 1, 22, 0))
        db_session.commit()

        with _patch_target(temp_db, api_client):
            response = api_client.post(RESET_PATH)

        assert response.status_code == 202
        assert response.json()["state"] == "queued"
        assert response.headers["cache-control"] == "no-store"
        assert _session_count(db_session) == 1

        run_queued_job()

        assert _session_count(db_session) == 0

    def test_reset_clears_data(
        self, api_client, db_session, test_device, test_session_factory, temp_db
    ):
        test_session_factory(test_device.id, start_time=datetime(2025, 1, 1, 22, 0))
        db_session.commit()
        with _patch_target(temp_db, api_client):
            result = run_reset(api_client)
        assert result["total_rows_deleted"] >= 1

    def test_reset_reports_measured_size_after_vacuum(self, api_client, temp_db):
        """size_after_mb is always a number now that VACUUM runs inside the job."""
        with _patch_target(temp_db, api_client):
            result = run_reset(api_client)
        assert isinstance(result["size_after_mb"], float)
        assert "vacuum_scheduled" not in result

    def test_reset_no_body_uses_data_only_default(self, api_client, temp_db):
        """Empty POST (no body) defaults to data-only reset — no bootstrap_invite_url."""
        with _patch_target(temp_db, api_client):
            data = run_reset(api_client)
        assert data["status"] == "success"
        assert data.get("bootstrap_invite_url") is None

    def test_reset_explicit_data_only(self, api_client, temp_db):
        """include_accounts=false explicitly — same result as default."""
        with _patch_target(temp_db, api_client):
            data = run_reset(api_client, json={"include_accounts": False})
        assert data["status"] == "success"
        assert data.get("bootstrap_invite_url") is None

    def test_reset_empties_waveform_array_cache(
        self, api_client, db_session, test_device, test_session_factory, temp_db
    ):
        """Arrays cached by a waveform read are gone after the reset."""
        session = test_session_factory(
            test_device.id, start_time=datetime(2025, 1, 1, 22, 0)
        )
        db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="flow",
                sample_rate=25.0,
                unit="L/min",
                sample_count=10,
                data_blob=bytes(10 * 8),  # 10 (timestamp, value) float32 pairs
            )
        )
        db_session.commit()
        read = api_client.get(f"/api/v1/sessions/{session.id}/waveforms/flow")
        assert read.status_code == 200
        assert len(_waveform_array_cache._entries) == 1

        with _patch_target(temp_db, api_client):
            result = run_reset(api_client)

        assert result["status"] == "success"
        assert not _waveform_array_cache._entries


@pytest.mark.usefixtures("reset_job_session")
class TestDbResetMultiuser:
    """Reset tests with an explicit admin actor in multiuser mode."""

    def _setup_multiuser(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _multiuser_env(monkeypatch)
        from snore.api.config import load_config, set_config  # noqa: PLC0415

        cfg = load_config(
            auth_mode_override="multiuser", bind_host_override="127.0.0.1"
        )
        set_config(cfg)

    def _make_admin_client(
        self,
        async_db_session: AsyncSession,
        db_session: SyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[object, models.User, models.Profile]:
        self._setup_multiuser(monkeypatch)

        user = models.User(
            canonical_email=f"admin_{uuid.uuid4().hex[:8]}@test.local",
            role="admin",
        )
        db_session.add(user)
        db_session.flush()
        profile = models.Profile(user_id=user.id, name="Default")
        db_session.add(profile)
        db_session.flush()
        user.default_profile_id = profile.id
        db_session.flush()  # persist default_profile_id before any resets can delete users

        actor = _admin_actor(user.id, profile.id)
        client = make_test_client(async_db_session, actor=actor)
        # Include origin so multiuser CSRF middleware allows mutating requests.
        client.headers.update({"origin": "http://127.0.0.1:8000"})
        return client, user, profile

    def test_member_gets_403(self, async_db_session, db_session, monkeypatch, temp_db):
        """Non-admin users cannot trigger a reset."""
        self._setup_multiuser(monkeypatch)

        member = models.User(
            canonical_email=f"member_{uuid.uuid4().hex[:8]}@test.local",
            role="member",
        )
        db_session.add(member)
        db_session.flush()
        client = make_test_client(async_db_session, actor=_member_actor(member.id))

        with _patch_target(temp_db, client):
            response = client.post("/api/v1/db/reset")
        assert response.status_code == 403
        assert not reset_jobs.has_active_job()

    def test_demo_gets_403(self, async_db_session, db_session, monkeypatch, temp_db):
        """Demo role cannot trigger a reset."""
        self._setup_multiuser(monkeypatch)

        demo_user = models.User(
            canonical_email=f"demo_{uuid.uuid4().hex[:8]}@test.local",
            role="demo",
        )
        db_session.add(demo_user)
        db_session.flush()
        client = make_test_client(async_db_session, actor=_demo_actor(demo_user.id))

        with _patch_target(temp_db, client):
            response = client.post("/api/v1/db/reset")
        assert response.status_code == 403
        assert not reset_jobs.has_active_job()

    def test_data_only_reset_zeroes_sleep_data_preserves_accounts(
        self,
        async_db_session,
        db_session,
        test_device,
        test_session_factory,
        monkeypatch,
        temp_db,
    ):
        """Data-only reset (default) clears sleep data but preserves users and profiles."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        test_session_factory(test_device.id, start_time=datetime(2025, 1, 1, 22, 0))
        db_session.commit()

        client, user, profile = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )

        with _patch_target(temp_db, client):
            data = run_reset(client, json={"include_accounts": False})

        assert data["status"] == "success"
        assert data["total_rows_deleted"] >= 1
        assert data.get("bootstrap_invite_url") is None

        # Verify: user and profile rows survive the data-only reset.
        db_session.expire_all()
        surviving_user = db_session.get(models.User, user.id)
        surviving_profile = db_session.get(models.Profile, profile.id)
        assert surviving_user is not None, "User row must survive data-only reset"
        assert surviving_profile is not None, "Profile row must survive data-only reset"

        # Verify: sleep data is gone.
        session_count = db_session.execute(
            sa_select(func.count()).select_from(models.Session)
        ).scalar()
        assert session_count == 0, "All sessions must be deleted by data-only reset"

    def test_data_only_reset_preserves_totp_recovery_codes(
        self, async_db_session, db_session, monkeypatch, temp_db
    ):
        """Data-only reset keeps 2FA recovery codes; totp_secret lives on users and survives."""
        client, user, _ = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )
        recovery_code = models.TotpRecoveryCode(user_id=user.id, code_hash="hash")
        db_session.add(recovery_code)
        db_session.flush()
        recovery_code_id = recovery_code.id

        with _patch_target(temp_db, client):
            data = run_reset(client, json={"include_accounts": False})

        db_session.expire_all()
        assert db_session.get(models.TotpRecoveryCode, recovery_code_id) is not None, (
            "Recovery codes must survive data-only reset"
        )
        assert "totp_recovery_codes" not in data["tables_cleared"]

    def test_include_accounts_reset_empties_users_returns_invite_url(
        self,
        async_db_session,
        db_session,
        monkeypatch,
        temp_db,
    ):
        """include_accounts=true wipes everything and returns a usable bootstrap_invite_url."""
        from sqlalchemy import func  # noqa: PLC0415
        from sqlalchemy import select as sa_select

        client, user, profile = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )
        db_session.add(models.TotpRecoveryCode(user_id=user.id, code_hash="hash"))
        db_session.flush()

        with _patch_target(temp_db, client):
            data = run_reset(client, json={"include_accounts": True})

        assert data["status"] == "success"
        assert data.get("bootstrap_invite_url") is not None
        invite_url: str = data["bootstrap_invite_url"]
        # The URL must contain a token after the # fragment separator.
        assert "#" in invite_url, (
            f"bootstrap_invite_url must contain a '#' fragment: {invite_url!r}"
        )
        token = invite_url.split("#", 1)[1]
        assert len(token) >= 20, f"Token looks too short: {token!r}"

        # Expunge stale objects (full reset deleted the seeded user/profile rows)
        # so that the next db_session queries don't trigger a spurious autoflush.
        db_session.expunge_all()

        # One new invite row should exist (for the admin re-registration).
        invite_count = db_session.execute(
            sa_select(func.count()).select_from(models.Invite)
        ).scalar()
        assert invite_count == 1, (
            "Exactly one bootstrap admin invite should exist after full reset"
        )

        recovery_code_count = db_session.execute(
            sa_select(func.count()).select_from(models.TotpRecoveryCode)
        ).scalar()
        assert recovery_code_count == 0, "Factory reset must delete recovery codes"
        assert data["tables_cleared"]["totp_recovery_codes"] == 1

        # Every table in the schema must appear in tables_cleared.
        assert set(data["tables_cleared"].keys()) == set(Base.metadata.tables.keys()), (
            "tables_cleared must include every table for a full factory reset"
        )

    def test_data_only_reset_clears_exactly_the_data_tables(
        self, async_db_session, db_session, monkeypatch, temp_db
    ):
        """Data-only reset clears only the listed data tables, even with no sleep data."""
        client, _, _ = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )

        with _patch_target(temp_db, client):
            data = run_reset(client, json={"include_accounts": False})

        assert data["status"] == "success"

        # Literal on purpose: a new table in the schema fails this test until
        # someone decides whether a data-only reset should delete it.
        expected_keys = {
            "analysis_job_records",
            "analysis_results",
            "breaths",
            "days",
            "detected_patterns",
            "devices",
            "events",
            "health_nightly_summaries",
            "health_samples",
            "import_job_records",
            "mask_log",
            "sessions",
            "settings",
            "statistics",
            "validation_runs",
            "waveforms",
        }
        assert set(data["tables_cleared"].keys()) == expected_keys, (
            "List each new table here if a data-only reset should delete it, "
            "or in _DATA_RESET_SKIP if it must survive"
        )

    def test_data_only_reset_purges_raw_dirs(
        self,
        async_db_session,
        db_session,
        test_device,
        monkeypatch,
        temp_db,
        tmp_path,
    ):
        """Data-only reset quarantine-renames then purges each profile's raw/ dir."""
        client, _, profile = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )

        # Create a fake raw/<profile_id>/ directory.
        raw_root = tmp_path / "raw"
        profile_dir = raw_root / str(profile.id)
        profile_dir.mkdir(parents=True)
        (profile_dir / "data.edf").write_text("fake")

        with (
            _patch_target(temp_db, client),
            patch("snore.api.reset_jobs._raw_root", return_value=raw_root),
        ):
            run_reset(client, json={"include_accounts": False})

        # The raw dir should no longer exist (purged via quarantine).
        assert not profile_dir.exists(), "Profile raw dir must be purged after reset"

    def test_factory_reset_invite_reaches_client_with_no_session(
        self, async_db_session, db_session, monkeypatch, temp_db
    ):
        """The invite survives the session it replaced: only the token is needed."""
        client, _, _ = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )
        with _patch_target(temp_db, client):
            response = client.post(RESET_PATH, json={"include_accounts": True})
        run_queued_job()

        anonymous = make_test_client(async_db_session, unauthenticated=True)
        status = read_status(anonymous, response.json()["job_id"]).json()

        assert status["state"] == "succeeded"
        assert "#" in status["reset_result"]["bootstrap_invite_url"]

    def test_vacuum_failure_after_commit_still_delivers_invite(
        self, async_db_session, db_session, monkeypatch, temp_db
    ):
        """A factory reset whose VACUUM blows up is committed, so it succeeded."""
        client, _, _ = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )

        def _vacuum_raises(db_path: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(reset_jobs, "_vacuum_background", _vacuum_raises)

        with _patch_target(temp_db, client):
            data = run_reset(client, json={"include_accounts": True})

        assert data["bootstrap_invite_url"] is not None
        assert data["size_after_mb"] == data["size_before_mb"]

    def test_rmtree_failure_after_commit_still_delivers_invite(
        self, async_db_session, db_session, monkeypatch, temp_db, tmp_path
    ):
        """A factory reset whose raw-dir purge blows up still returns its invite."""
        client, _, profile = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )
        raw_root = tmp_path / "raw"
        profile_dir = raw_root / str(profile.id)
        profile_dir.mkdir(parents=True)

        with (
            _patch_target(temp_db, client),
            patch("snore.api.reset_jobs._raw_root", return_value=raw_root),
            patch(
                "snore.services.database_service.shutil.rmtree",
                side_effect=OSError("disk error"),
            ),
        ):
            data = run_reset(client, json={"include_accounts": True})

        assert data["bootstrap_invite_url"] is not None


# ---------------------------------------------------------------------------
# Concurrency guards
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_job_session")
class TestDbResetConcurrency:
    def test_409_while_reset_job_active(self, api_client, temp_db):
        """A second reset returns 409 without queueing another job."""
        with _patch_target(temp_db, api_client):
            first = api_client.post(RESET_PATH)
            second = api_client.post(RESET_PATH)

        assert first.status_code == 202
        assert second.status_code == 409
        assert "already in progress" in second.json()["detail"]
        assert len(reset_jobs._queue) == 1

    def test_409_while_delete_data_job_active(self, api_client, temp_db):
        """A queued delete-data also blocks a reset: they share one job slot."""
        with _patch_target(temp_db, api_client):
            first = api_client.post(DELETE_DATA_PATH)
            second = api_client.post(RESET_PATH)

        assert first.status_code == 202
        assert second.status_code == 409

    def test_409_while_job_running(self, api_client, temp_db):
        """The slot stays taken once the worker has picked the job up."""
        with _patch_target(temp_db, api_client):
            first = api_client.post(RESET_PATH)
            job = reset_jobs.get_job(first.json()["job_id"])
            job.try_start()
            second = api_client.post(RESET_PATH)

        assert second.status_code == 409

    def test_no_database_service_in_reset_db_signature(self):
        """reset_db must not take a DatabaseService parameter.

        The job owns the only write transaction.  A DatabaseService dependency
        constructs its own get_db session, which a request has no use for and
        which could block on that transaction's write lock.

        Resolves annotations via typing.get_type_hints so deferred string
        annotations (PEP 563 / from __future__ import annotations) are
        evaluated correctly before the check.
        """
        from snore.api.routers.db import reset_db  # noqa: PLC0415
        from snore.services.database_service import DatabaseService  # noqa: PLC0415

        hints = typing.get_type_hints(reset_db, include_extras=True)

        def _references_db_service(annotation: object) -> bool:
            if annotation is DatabaseService:
                return True
            return any(
                _references_db_service(arg) for arg in typing.get_args(annotation)
            )

        for name, hint in hints.items():
            assert not _references_db_service(hint), (
                f"reset_db parameter '{name}' annotation resolves to DatabaseService — "
                "the request must not open a session alongside the job's"
            )

    def test_sequential_resets_both_succeed(self, api_client, temp_db):
        """Two consecutive data-only resets both succeed: a finished job frees the slot."""
        with _patch_target(temp_db, api_client):
            first = run_reset(api_client)
            second = run_reset(api_client)

        assert first["status"] == "success"
        assert second["status"] == "success"


# ---------------------------------------------------------------------------
# Job status endpoint
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_job_session")
class TestResetJobStatus:
    def test_unauthenticated_client_reads_status(
        self, api_client, async_db_session, temp_db
    ):
        """No session is needed: a factory reset deletes the caller's mid-job."""
        with _patch_target(temp_db, api_client):
            job_id = api_client.post(RESET_PATH).json()["job_id"]

        anonymous = make_test_client(async_db_session, unauthenticated=True)
        assert anonymous.get("/api/v1/db/stats").status_code == 401

        response = read_status(anonymous, job_id)

        assert response.status_code == 200
        assert response.json() == {
            "kind": "reset",
            "state": "queued",
            "error_message": None,
            "reset_result": None,
            "delete_data_result": None,
        }

    def test_status_is_not_cacheable(self, api_client, temp_db):
        with _patch_target(temp_db, api_client):
            job_id = api_client.post(RESET_PATH).json()["job_id"]

        assert read_status(api_client, job_id).headers["cache-control"] == "no-store"

    def test_unknown_token_returns_404(self, api_client):
        assert read_status(api_client, "not-a-real-token").status_code == 404

    def test_missing_token_header_returns_422(self, api_client):
        assert api_client.get(STATUS_PATH).status_code == 422

    def test_token_in_query_string_is_not_accepted(self, api_client, temp_db):
        """URLs land in access logs, so the token only ever travels in the header."""
        with _patch_target(temp_db, api_client):
            job_id = api_client.post(RESET_PATH).json()["job_id"]

        response = api_client.get(STATUS_PATH, params={"token": job_id})

        assert response.status_code == 422

    def test_expired_job_returns_404(self, api_client, temp_db):
        with _patch_target(temp_db, api_client):
            job_id = api_client.post(RESET_PATH).json()["job_id"]
        run_queued_job()
        reset_jobs.get_job(job_id)._finished_at = 0.0

        reset_jobs._reap_terminal()

        assert read_status(api_client, job_id).status_code == 404


# ---------------------------------------------------------------------------
# Job failures
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_job_session")
class TestResetJobFailures:
    def test_locked_database_fails_job_with_busy_message(
        self,
        api_client,
        db_session,
        test_device,
        test_session_factory,
        temp_db,
        monkeypatch,
    ):
        """A real writer holding the lock fails the job; the data survives."""
        test_session_factory(test_device.id, start_time=datetime(2025, 1, 1, 22, 0))
        db_session.commit()
        impatient_engine = create_async_engine(
            f"sqlite+aiosqlite:///{temp_db}",
            connect_args={"timeout": 0.05},
            poolclass=NullPool,
        )
        factory = async_sessionmaker(bind=impatient_engine, class_=AsyncSession)

        @asynccontextmanager
        async def _impatient_scope(*, immediate=False):
            async with factory() as session, session.begin():
                yield session

        monkeypatch.setattr(reset_jobs, "session_scope", _impatient_scope)

        with _patch_target(temp_db, api_client):
            job_id = api_client.post(RESET_PATH).json()["job_id"]

        writer = sqlite3.connect(temp_db, isolation_level=None)
        try:
            writer.execute("BEGIN IMMEDIATE")
            run_queued_job()
            writer.execute("ROLLBACK")
        finally:
            writer.close()

        status = read_status(api_client, job_id).json()
        assert status["state"] == "failed"
        assert "busy" in status["error_message"].lower()
        assert status["reset_result"] is None
        assert _session_count(db_session) == 1

    def test_lock_timeout_opening_the_transaction_gives_busy_message(
        self, api_client, temp_db, monkeypatch
    ):
        """BEGIN IMMEDIATE itself is where production contention surfaces."""

        @asynccontextmanager
        async def _locked_scope(*, immediate=False):
            raise OperationalError(
                "BEGIN IMMEDIATE", {}, Exception("database is locked")
            )
            yield

        monkeypatch.setattr(reset_jobs, "session_scope", _locked_scope)

        with _patch_target(temp_db, api_client):
            status = post_and_run_job(api_client, RESET_PATH)

        assert status["state"] == "failed"
        assert "busy" in status["error_message"].lower()

    def test_unexpected_exception_reports_generic_message(self, api_client, temp_db):
        """Exception text reaches an unauthenticated endpoint, so it never gets in."""
        with (
            _patch_target(temp_db, api_client),
            patch.object(
                DatabaseService,
                "reset_all_data",
                side_effect=RuntimeError("/data/.snore/snore.db: disk I/O error"),
            ),
        ):
            status = post_and_run_job(api_client, RESET_PATH)

        assert status["state"] == "failed"
        assert status["error_message"] == reset_jobs._FAILURE_MESSAGE
        assert "disk I/O" not in status["error_message"]

    def test_new_job_accepted_after_failure(self, api_client, temp_db):
        with (
            _patch_target(temp_db, api_client),
            patch.object(
                DatabaseService, "reset_all_data", side_effect=RuntimeError("boom")
            ),
        ):
            assert post_and_run_job(api_client, RESET_PATH)["state"] == "failed"

        with _patch_target(temp_db, api_client):
            assert run_reset(api_client)["status"] == "success"


# ---------------------------------------------------------------------------
# End to end: real session cookies, the lifespan worker, and a real VACUUM
# ---------------------------------------------------------------------------


async def _wait_for_terminal_status(
    client: httpx.AsyncClient, job_id: str, *, cookie: str = ""
) -> dict[str, typing.Any]:
    headers = token_header(job_id) | ({"cookie": cookie} if cookie else {})
    deadline = time.monotonic() + 15.0
    while True:
        response = await client.get(STATUS_PATH, headers=headers)
        assert response.status_code == 200, response.text
        status = response.json()
        if status["state"] not in ("queued", "running"):
            return status
        assert time.monotonic() < deadline, f"job still {status['state']}"
        await asyncio.sleep(0.02)


class TestResetJobEndToEnd:
    @pytest.fixture
    async def lifespan_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> AsyncIterator[tuple[httpx.AsyncClient, Path]]:
        """The app with its real lifespan, worker thread, and VACUUM."""
        db_path = tmp_path / "reset_lifecycle.db"
        monkeypatch.setenv("SNORE_DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
        monkeypatch.setattr(reset_jobs, "_vacuum_background", _REAL_VACUUM_BG)
        await cleanup_database()

        app = create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            async with app.router.lifespan_context(app):
                yield client, db_path

        await cleanup_database()

    async def test_lifespan_worker_deletes_vacuums_and_frees_the_slot(
        self, lifespan_client
    ):
        """POST → poll → success, with the file really shrunk on disk.

        The server's pooled connections keep the WAL alive throughout, which is
        what left production's file at 11 GB until the VACUUM began checkpointing.
        """
        client, db_path = lifespan_client
        async with session_scope(immediate=True) as db:
            user = models.User(canonical_email="owner@test.local", role="admin")
            db.add(user)
            await db.flush()
            profile = models.Profile(user_id=user.id, name="Default")
            db.add(profile)
            await db.flush()
            device = models.Device(
                profile_id=profile.id,
                manufacturer="ResMed",
                model="AirSense 10",
                serial_number="E2E",
            )
            db.add(device)
            await db.flush()
            session = models.Session(
                device_id=device.id,
                device_session_id="s1",
                start_time=datetime(2025, 1, 1, 22, 0),
                end_time=datetime(2025, 1, 2, 6, 0),
                duration_seconds=28800,
            )
            db.add(session)
            await db.flush()
            db.add(
                models.Waveform(
                    session_id=session.id,
                    waveform_type="flow",
                    sample_rate=25.0,
                    unit="L/min",
                    sample_count=1024,
                    data_blob=bytes(8 * 1024 * 1024),
                )
            )

        response = await client.post(RESET_PATH)
        assert response.status_code == 202, response.text

        status = await _wait_for_terminal_status(client, response.json()["job_id"])

        assert status["state"] == "succeeded", status
        result = status["reset_result"]
        assert result["tables_cleared"]["waveforms"] == 1
        assert result["size_after_mb"] < result["size_before_mb"]
        assert result["size_after_mb"] < 1.0
        assert db_path.stat().st_size < 1024 * 1024
        assert Path(f"{db_path}-wal").stat().st_size == 0
        assert (await client.get("/health/busy")).json() == {"busy": False}
        assert (await client.post(RESET_PATH)).status_code == 202

    async def test_factory_reset_kills_session_but_token_still_reads_invite(
        self, temp_db, monkeypatch
    ):
        """The caller's cookie dies mid-job; polling with it keeps working."""
        _multiuser_env(monkeypatch)
        cfg = load_config(
            auth_mode_override="multiuser", bind_host_override="127.0.0.1"
        )
        set_config(cfg)
        await cleanup_database()
        await init_database(str(temp_db))
        async with session_scope() as db:
            user = models.User(
                canonical_email="admin@test.local", role="admin", session_version=0
            )
            db.add(user)
            await db.flush()
            profile = models.Profile(user_id=user.id, name="Default")
            db.add(profile)
            await db.flush()
            user.default_profile_id = profile.id
            user_id, profile_id = user.id, profile.id

        cookie = "snore_session=" + encode_session(
            cfg.session_secret,
            user_id=user_id,
            active_profile_id=profile_id,
            session_version=0,
        )
        reset_jobs.start_worker()

        app = create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
        ) as client:
            response = await client.post(
                RESET_PATH,
                json={"include_accounts": True},
                headers={"origin": "http://127.0.0.1:8000", "cookie": cookie},
            )
            assert response.status_code == 202, response.text

            status = await _wait_for_terminal_status(
                client, response.json()["job_id"], cookie=cookie
            )
            me = await client.get("/api/v1/auth/me", headers={"cookie": cookie})

        assert status["state"] == "succeeded", status
        assert status["reset_result"]["bootstrap_invite_url"].startswith(
            "http://127.0.0.1:8000/invite#"
        )
        assert me.status_code == 401
        await cleanup_database()


# ---------------------------------------------------------------------------
# db_busy_maps_to_409 unit tests
# ---------------------------------------------------------------------------


class TestDbBusyMapsTo409:
    """Unit tests for the db_busy_maps_to_409 context manager."""

    async def test_locked_database_raises_409(self):
        """OperationalError with 'database is locked' maps to HTTPException 409."""
        from snore.api.errors import db_busy_maps_to_409  # noqa: PLC0415

        locked_exc = OperationalError("SELECT 1", {}, Exception("database is locked"))

        with pytest.raises(HTTPException) as exc_info:
            async with db_busy_maps_to_409():
                raise locked_exc

        assert exc_info.value.status_code == 409
        assert "busy" in exc_info.value.detail.lower()

    async def test_other_operational_error_propagates_unchanged(self):
        """OperationalError with a different message propagates without conversion."""
        from snore.api.errors import db_busy_maps_to_409  # noqa: PLC0415

        other_exc = OperationalError("SELECT 1", {}, Exception("disk I/O error"))

        with pytest.raises(OperationalError) as exc_info:
            async with db_busy_maps_to_409():
                raise other_exc

        assert exc_info.value is other_exc

    async def test_no_exception_passes_through(self):
        """Body that raises no exception completes normally."""
        from snore.api.errors import db_busy_maps_to_409  # noqa: PLC0415

        result: list[int] = []
        async with db_busy_maps_to_409():
            result.append(42)

        assert result == [42]


# ---------------------------------------------------------------------------
# Crash-safe cleanup: vacuum marker and quarantine-before-commit
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("reset_job_session")
class TestDbResetVacuumMarker:
    """The reset job writes the vacuum pending marker before the commit."""

    def _make_admin_client(
        self,
        async_db_session: AsyncSession,
        db_session: SyncSession,
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[object, models.User, models.Profile]:
        """Reuse the multiuser setup pattern for admin client."""
        _multiuser_env(monkeypatch)
        from snore.api.config import load_config, set_config  # noqa: PLC0415

        cfg = load_config(
            auth_mode_override="multiuser", bind_host_override="127.0.0.1"
        )
        set_config(cfg)

        user = models.User(
            canonical_email=f"admin_{uuid.uuid4().hex[:8]}@test.local",
            role="admin",
        )
        db_session.add(user)
        db_session.flush()
        profile = models.Profile(user_id=user.id, name="Default")
        db_session.add(profile)
        db_session.flush()
        user.default_profile_id = profile.id
        db_session.flush()

        actor = _admin_actor(user.id, profile.id)
        client = make_test_client(async_db_session, actor=actor)
        client.headers.update({"origin": "http://127.0.0.1:8000"})
        return client, user, profile

    def test_reset_writes_vacuum_marker(
        self, async_db_session, db_session, monkeypatch, temp_db, tmp_path
    ):
        """POST /db/reset writes a vacuum pending marker file before committing.

        The marker content is whatever sqlite_path is resolved in the test
        environment (may not equal temp_db since target resolution uses env
        vars); we only assert it is non-empty and the marker file is created.
        """
        from unittest.mock import patch  # noqa: PLC0415

        client, _, _ = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )
        marker = tmp_path / "vacuum.pending"

        with (
            _patch_target(temp_db, client),
            patch("snore.api.reset_jobs.DEFAULT_VACUUM_PENDING_MARKER", marker),
        ):
            run_reset(client, json={"include_accounts": False})

        # _vacuum_background is patched to a no-op in tests so the marker persists.
        assert marker.exists(), (
            "Vacuum pending marker must be written by reset endpoint"
        )
        assert marker.read_text().strip(), "Marker must contain a non-empty DB path"

    def test_reset_quarantine_survives_rmtree_failure_and_is_swept(
        self,
        async_db_session,
        db_session,
        monkeypatch,
        temp_db,
        tmp_path,
    ):
        """When post-commit rmtree fails, dir stays in .quarantine/ for startup recovery."""
        import asyncio  # noqa: PLC0415

        from contextlib import asynccontextmanager  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from snore.services.profile_service import DeletionSaga  # noqa: PLC0415

        client, _, profile = self._make_admin_client(
            async_db_session, db_session, monkeypatch
        )

        raw_root = tmp_path / "raw"
        profile_dir = raw_root / str(profile.id)
        profile_dir.mkdir(parents=True)
        (profile_dir / "data.edf").write_text("backup")

        # Patch shutil.rmtree in the service to simulate a crash right after commit.
        def _rmtree_raise(path, **kwargs):
            raise OSError("disk error")

        with (
            _patch_target(temp_db, client),
            patch("snore.api.reset_jobs._raw_root", return_value=raw_root),
            patch(
                "snore.services.database_service.shutil.rmtree",
                side_effect=_rmtree_raise,
            ),
        ):
            # The reset is committed; the job must still succeed.
            result = run_reset(client, json={"include_accounts": False})

        assert result["status"] == "success"

        # Quarantine dir must exist (rmtree was skipped/failed).
        quarantine_dst = raw_root / ".quarantine" / str(profile.id)
        assert quarantine_dst.exists(), "Quarantine dir must remain when rmtree fails"

        # Simulate startup recovery sweeping the orphaned quarantine dir.
        @asynccontextmanager
        async def _mock_session_scope():
            yield async_db_session

        with patch("snore.services.profile_service.session_scope", _mock_session_scope):
            saga = DeletionSaga(raw_root=raw_root)
            asyncio.run(saga._recover_async())

        assert not quarantine_dst.exists(), (
            "DeletionSaga.recover() case 2 must sweep quarantine dir after crash"
        )
