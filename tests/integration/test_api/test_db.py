import typing
import uuid

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session as SyncSession

from snore.auth.actor import ActorContext, AuthMode, Role
from snore.database import models
from snore.database.models import Base
from snore.services.waveform_service import _waveform_array_cache
from tests.helpers.api_client import make_test_client
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


class TestDbReset:
    def test_reset_status_is_success(self, api_client, temp_db):
        with _patch_target(temp_db, api_client):
            response = api_client.post("/api/v1/db/reset")
        assert response.json()["status"] == "success"

    def test_reset_clears_data(
        self, api_client, db_session, test_device, test_session_factory, temp_db
    ):
        test_session_factory(test_device.id, start_time=datetime(2025, 1, 1, 22, 0))
        db_session.commit()
        with _patch_target(temp_db, api_client):
            response = api_client.post("/api/v1/db/reset")
        assert response.json()["total_rows_deleted"] >= 1

    def test_reset_no_body_uses_data_only_default(self, api_client, temp_db):
        """Empty POST (no body) defaults to data-only reset — no bootstrap_invite_url."""
        with _patch_target(temp_db, api_client):
            response = api_client.post("/api/v1/db/reset")
        data = response.json()
        assert data["status"] == "success"
        assert data.get("bootstrap_invite_url") is None

    def test_reset_explicit_data_only(self, api_client, temp_db):
        """include_accounts=false explicitly — same result as default."""
        with _patch_target(temp_db, api_client):
            response = api_client.post(
                "/api/v1/db/reset", json={"include_accounts": False}
            )
        data = response.json()
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
            response = api_client.post("/api/v1/db/reset")

        assert response.json()["status"] == "success"
        assert not _waveform_array_cache._entries


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
            response = client.post("/api/v1/db/reset", json={"include_accounts": False})

        assert response.status_code == 200
        data = response.json()
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
            response = client.post("/api/v1/db/reset", json={"include_accounts": False})

        assert response.status_code == 200
        db_session.expire_all()
        assert db_session.get(models.TotpRecoveryCode, recovery_code_id) is not None, (
            "Recovery codes must survive data-only reset"
        )
        assert "totp_recovery_codes" not in response.json()["tables_cleared"]

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
            response = client.post("/api/v1/db/reset", json={"include_accounts": True})

        assert response.status_code == 200
        data = response.json()
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
            response = client.post("/api/v1/db/reset", json={"include_accounts": False})

        assert response.status_code == 200
        data = response.json()
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
            patch("snore.api.routers.db._raw_root", return_value=raw_root),
        ):
            response = client.post("/api/v1/db/reset", json={"include_accounts": False})

        assert response.status_code == 200
        # The raw dir should no longer exist (purged via quarantine).
        assert not profile_dir.exists(), "Profile raw dir must be purged after reset"


# ---------------------------------------------------------------------------
# Concurrency guards
# ---------------------------------------------------------------------------


class TestDbResetConcurrency:
    def test_409_when_lock_held(self, api_client, monkeypatch, temp_db):
        """Reset returns 409 immediately when the reset lock is already held.

        Monkeypatching the module-level lock is safe here because
        require_reset_lock does a LOAD_GLOBAL lookup at call time, so it
        sees the replacement value on the next request.
        """
        import snore.api.deps as deps  # noqa: PLC0415

        mock_lock = MagicMock()
        mock_lock.locked = lambda: True
        monkeypatch.setattr(deps, "_reset_lock", mock_lock)

        with _patch_target(temp_db, api_client):
            response = api_client.post("/api/v1/db/reset")

        assert response.status_code == 409
        assert "already in progress" in response.json()["detail"]

    def test_no_database_service_in_reset_db_signature(self):
        """reset_db must not take a DatabaseService parameter.

        A DatabaseService dependency constructs its own get_db session.
        Having that alongside ImmediateDbDep's BEGIN IMMEDIATE session means
        two concurrent sessions: the plain deferred session's first write
        blocks on the write lock held by the immediate session, producing
        OperationalError("database is locked") after busy_timeout=5000ms.
        This structural regression test guards against reintroducing that bug.

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
                "this would reintroduce the dual-session deadlock"
            )

    def test_sequential_resets_both_succeed(self, api_client, temp_db):
        """Two consecutive data-only resets both return 200, proving the lock is released."""
        with _patch_target(temp_db, api_client):
            first = api_client.post("/api/v1/db/reset")
            second = api_client.post("/api/v1/db/reset")

        assert first.status_code == 200, first.json()
        assert second.status_code == 200, second.json()


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


class TestDbResetVacuumMarker:
    """POST /db/reset writes the vacuum pending marker before the commit."""

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
            patch("snore.api.routers.db.DEFAULT_VACUUM_PENDING_MARKER", marker),
        ):
            response = client.post("/api/v1/db/reset", json={"include_accounts": False})

        assert response.status_code == 200
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

        # Patch shutil.rmtree in the db router to simulate a crash right after commit.
        def _rmtree_raise(path, **kwargs):
            raise OSError("disk error")

        with (
            _patch_target(temp_db, client),
            patch("snore.api.routers.db._raw_root", return_value=raw_root),
            patch("snore.api.routers.db.shutil.rmtree", side_effect=_rmtree_raise),
        ):
            response = client.post("/api/v1/db/reset", json={"include_accounts": False})

        # The reset is committed; response must still be 200.
        assert response.status_code == 200

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
