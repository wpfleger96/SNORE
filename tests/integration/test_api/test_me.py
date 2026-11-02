"""Integration tests for the self-service account management router.

Endpoints under test:
    GET   /api/v1/auth/me
    PATCH /api/v1/auth/me/display-name
    POST  /api/v1/auth/me/password
    GET   /api/v1/auth/me/preferences
    PATCH /api/v1/auth/me/preferences

Most tests use a sync TestClient with the actor hard-wired via a dependency
override in local mode (no CSRF, no rate limiting, no cookie handling needed).
Tests that require the full multiuser middleware stack (unauthenticated 401,
session-version invalidation) run as async tests via httpx.AsyncClient.
"""

from __future__ import annotations

import http.cookies
import uuid

from datetime import datetime

import httpx
import pytest

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from snore.api import reset_jobs
from snore.api.app import create_app
from snore.api.config import load_config, set_config
from snore.auth.actor import ActorContext, AuthMode, Role
from snore.auth.lockout import get_lockout_store
from snore.auth.passwords import hash_password, verify_password
from snore.auth.session_cookie import encode_session
from snore.database import models
from tests.helpers.api_client import make_test_client
from tests.helpers.reset_jobs import (
    DELETE_DATA_PATH,
    RESET_PATH,
    post_and_run_job,
    read_status,
    run_delete_data,
    run_queued_job,
)
from tests.integration.test_api.conftest import _multiuser_env  # noqa: PLC0415

# ---------------------------------------------------------------------------
# Seeding / client helpers
# ---------------------------------------------------------------------------


def _seed_user(
    db_session: Session,
    *,
    role: str = "member",
    password: str | None = None,
    display_name: str | None = None,
    preferences: dict | None = None,
) -> tuple[models.User, models.Profile]:
    """Insert a User + Profile via the sync AUTOCOMMIT session; return both."""
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    user = models.User(
        canonical_email=email,
        role=role,
        password_hash=hash_password(password) if password else None,
        display_name=display_name,
        preferences=preferences if preferences is not None else {},
        session_version=0,
    )
    db_session.add(user)
    db_session.flush()

    profile = models.Profile(user_id=user.id, name="Default")
    db_session.add(profile)
    db_session.flush()

    user.default_profile_id = profile.id
    db_session.flush()

    return user, profile


def _make_client(
    async_db_session: AsyncSession, user_id: int, profile_id: int, role: str
) -> TestClient:
    """Build a TestClient wired to a fixed ActorContext for a specific user."""
    actor = ActorContext(
        user_id=user_id,
        profile_id=profile_id,
        role=Role(role),
        mode=AuthMode.LOCAL,
    )
    return make_test_client(async_db_session, actor=actor)


# ---------------------------------------------------------------------------
# TestGetMe
# ---------------------------------------------------------------------------


class TestGetMe:
    def test_authenticated_member_gets_me(self, async_db_session, db_session):
        """Authenticated member receives all expected fields; has_password reflects stored hash."""
        user, profile = _seed_user(db_session, role="member", password="secret123")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.get("/api/v1/auth/me")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == user.id
        assert data["email"] == user.canonical_email
        assert data["display_name"] is None
        assert data["role"] == "member"
        assert data["has_password"] is True

    def test_google_only_user_has_password_false(self, async_db_session, db_session):
        """Google-only account (password_hash=None) reports has_password: false."""
        user, profile = _seed_user(db_session, role="member", password=None)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.get("/api/v1/auth/me")

        assert resp.status_code == 200
        assert resp.json()["has_password"] is False

    def test_google_linked_false_when_no_identity(self, async_db_session, db_session):
        """User with no auth_identities row reports google_linked: false."""
        user, profile = _seed_user(db_session, role="member", password="pw")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.get("/api/v1/auth/me")

        assert resp.status_code == 200
        assert resp.json()["google_linked"] is False

    def test_google_linked_true_when_identity_exists(
        self, async_db_session, db_session
    ):
        """User with a google auth_identities row reports google_linked: true."""
        user, profile = _seed_user(db_session, role="member", password="pw")
        db_session.add(
            models.AuthIdentity(
                user_id=user.id,
                provider="google",
                subject="google-sub-test-123",
                email=user.canonical_email,
            )
        )
        db_session.flush()
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.get("/api/v1/auth/me")

        assert resp.status_code == 200
        assert resp.json()["google_linked"] is True

    def test_unauthenticated_gets_401(self, async_db_session, monkeypatch):
        """Request with no session cookie returns 401 in multiuser mode."""
        _multiuser_env(monkeypatch)
        cfg = load_config(
            auth_mode_override="multiuser", bind_host_override="127.0.0.1"
        )
        set_config(cfg)

        # No actor override: AuthMiddleware runs and finds no session cookie → 401.
        client = make_test_client(async_db_session, no_actor_override=True)

        resp = client.get("/api/v1/auth/me")

        assert resp.status_code == 401

    def test_demo_actor_can_read(self, async_db_session, db_session):
        """Demo role is allowed on GET /me (require_auth, not require_writable)."""
        user, profile = _seed_user(db_session, role="demo")
        client = _make_client(async_db_session, user.id, profile.id, "demo")

        resp = client.get("/api/v1/auth/me")

        assert resp.status_code == 200
        assert resp.json()["role"] == "demo"


# ---------------------------------------------------------------------------
# TestDisplayName
# ---------------------------------------------------------------------------


class TestDisplayName:
    def test_update_display_name_succeeds_and_persists(
        self, async_db_session, db_session
    ):
        """PATCH /display-name updates the field; a subsequent GET /me reflects it."""
        user, profile = _seed_user(db_session, role="member", password="pw")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        patch_resp = client.patch(
            "/api/v1/auth/me/display-name",
            json={"display_name": "My Cool Name"},
        )
        assert patch_resp.status_code == 200

        get_resp = client.get("/api/v1/auth/me")
        assert get_resp.status_code == 200
        assert get_resp.json()["display_name"] == "My Cool Name"

    def test_empty_display_name_clears_to_none(self, async_db_session, db_session):
        """Whitespace-only display_name strips to None; explicit null also clears to None."""
        user, profile = _seed_user(db_session, role="member", display_name="Old Name")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        # Whitespace-only: "   ".strip() == "" → falsy → None.
        resp_ws = client.patch(
            "/api/v1/auth/me/display-name",
            json={"display_name": "   "},
        )
        assert resp_ws.status_code == 200
        assert client.get("/api/v1/auth/me").json()["display_name"] is None

        # Explicit null → None.
        resp_null = client.patch(
            "/api/v1/auth/me/display-name",
            json={"display_name": None},
        )
        assert resp_null.status_code == 200
        assert client.get("/api/v1/auth/me").json()["display_name"] is None

    def test_display_name_too_long_422(self, async_db_session, db_session):
        """display_name exceeding 150 characters is rejected with 422."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.patch(
            "/api/v1/auth/me/display-name",
            json={"display_name": "x" * 151},
        )
        assert resp.status_code == 422

    def test_demo_cannot_update_display_name(self, async_db_session, db_session):
        """Demo role hits 403 on PATCH /display-name (require_writable guard)."""
        user, profile = _seed_user(db_session, role="demo")
        client = _make_client(async_db_session, user.id, profile.id, "demo")

        resp = client.patch(
            "/api/v1/auth/me/display-name",
            json={"display_name": "Sneaky"},
        )
        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# TestChangePassword
# ---------------------------------------------------------------------------


class TestChangePassword:
    def test_change_password_happy_path(self, async_db_session, db_session):
        """Correct current_password → 200; new password validates against stored hash."""
        old_pw = "old-correct-password"
        new_pw = "new-password-456"
        user, profile = _seed_user(db_session, role="member", password=old_pw)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.post(
            "/api/v1/auth/me/password",
            json={"current_password": old_pw, "new_password": new_pw},
        )
        assert resp.status_code == 200

        # Verify new hash is stored and old password is invalidated.
        db_session.refresh(user)
        ok_new, _ = verify_password(user.password_hash, new_pw)
        assert ok_new, "New password must validate against stored hash"
        ok_old, _ = verify_password(user.password_hash, old_pw)
        assert not ok_old, "Old password must be rejected after change"

    @pytest.mark.asyncio
    async def test_session_version_semantics(self, temp_db, monkeypatch):
        """After password change the old cookie is rejected (401); the re-issued cookie works (200)."""
        _multiuser_env(monkeypatch)
        cfg = load_config(
            auth_mode_override="multiuser", bind_host_override="127.0.0.1"
        )
        set_config(cfg)

        from snore.database.session import (
            cleanup_database,
            init_database,
            session_scope,
        )

        await init_database(str(temp_db))

        pw = "test-password-123"
        async with session_scope() as db:
            user = models.User(
                canonical_email=f"sv_{uuid.uuid4().hex[:6]}@test",
                role="member",
                session_version=0,
                password_hash=hash_password(pw),
            )
            db.add(user)
            await db.flush()
            profile = models.Profile(user_id=user.id, name="Default")
            db.add(profile)
            await db.flush()
            user.default_profile_id = profile.id
            user_id = user.id
            profile_id = profile.id

        old_cookie = encode_session(
            cfg.session_secret,
            user_id=user_id,
            active_profile_id=profile_id,
            session_version=0,
        )

        app = create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
        ) as client:
            # Change password with the old cookie (session_version=0).
            change_resp = await client.post(
                "/api/v1/auth/me/password",
                json={"current_password": pw, "new_password": "new-pw-789"},
                headers={
                    "origin": "http://127.0.0.1:8000",
                    "cookie": f"snore_session={old_cookie}",
                },
            )
            assert change_resp.status_code == 200, change_resp.text

            # Extract the re-issued cookie (version 1) from Set-Cookie.
            set_cookie_header = change_resp.headers.get("set-cookie", "")
            c = http.cookies.SimpleCookie()
            c.load(set_cookie_header)
            assert "snore_session" in c, (
                f"Password change must re-issue the session cookie; "
                f"Set-Cookie: {set_cookie_header!r}"
            )
            new_cookie = c["snore_session"].value

            # Old cookie (version 0) must be rejected after the version bump.
            old_resp = await client.get(
                "/api/v1/auth/me",
                headers={"cookie": f"snore_session={old_cookie}"},
            )
            assert old_resp.status_code == 401, (
                "Stale session cookie must be rejected after session_version increment"
            )

            # New cookie (version 1) must be accepted.
            new_resp = await client.get(
                "/api/v1/auth/me",
                headers={"cookie": f"snore_session={new_cookie}"},
            )
            assert new_resp.status_code == 200, (
                "Re-issued session cookie must grant access"
            )

        await cleanup_database()

    def test_local_mode_unlink_no_set_cookie(self, async_db_session, db_session):
        """In LOCAL mode, DELETE /identities/google must carry no Set-Cookie header.

        clear_session_cookie is gated on cfg.is_multiuser, so a local-mode
        unlink response should have no Set-Cookie header at all.
        """
        user, profile = _seed_user(db_session, role="member", password="pw")
        _seed_google_identity(db_session, user.id)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.delete("/api/v1/auth/me/identities/google")

        assert resp.status_code == 200
        assert "set-cookie" not in {k.lower() for k in resp.headers}

    @pytest.mark.asyncio
    async def test_google_link_disabled_blocks_auto_relink(
        self, async_db_session, db_session
    ):
        """After unlink sets google_link_disabled, resolve_login's email auto-link
        path raises TxFailure so a subsequent 'Sign in with Google' cannot silently
        re-establish the severed identity."""
        from snore.api.config import get_config
        from snore.api.routers.auth._google_resolution import TxFailure, resolve_login

        user, profile = _seed_user(db_session, role="member", password="pw")
        # Simulate what unlink does: set the flag directly.
        user.google_link_disabled = True
        db_session.flush()

        cfg = get_config()

        async with async_db_session.begin():
            with pytest.raises(TxFailure):
                await resolve_login(
                    async_db_session,
                    cfg,
                    {
                        "sub": "brand-new-google-sub-xyz",
                        "email": user.canonical_email,
                        "email_verified": True,
                    },
                )

    @pytest.mark.asyncio
    async def test_invite_relink_clears_google_link_disabled(
        self, async_db_session, db_session
    ):
        """link_identity_ticket (used by invite-signup path b) resets
        google_link_disabled to False so future auto-links are restored."""
        from snore.api.config import get_config
        from snore.api.routers.auth._google_resolution import link_identity_ticket

        user, profile = _seed_user(db_session, role="member", password="pw")
        user.google_link_disabled = True
        db_session.flush()

        cfg = get_config()

        async with async_db_session.begin():
            await link_identity_ticket(
                async_db_session,
                cfg,
                # fetch the live user object from the async session
                await async_db_session.get(models.User, user.id),
                "re-link-sub-abc",
                user.canonical_email,
            )

        # Reload via db_session to confirm the flag was cleared in the DB.
        db_session.expire_all()
        db_session.refresh(user)
        assert user.google_link_disabled is False, (
            "link_identity_ticket must reset google_link_disabled"
        )

    def test_wrong_current_password_401(self, async_db_session, db_session):
        """Wrong current_password → 401 Authentication failed."""
        user, profile = _seed_user(db_session, role="member", password="correct-pw")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.post(
            "/api/v1/auth/me/password",
            json={"current_password": "wrong-pw", "new_password": "new-pw-123"},
        )
        assert resp.status_code == 401

    def test_missing_current_password_422(self, async_db_session, db_session):
        """Omitting current_password when the user has one set returns 422."""
        user, profile = _seed_user(db_session, role="member", password="exists")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.post(
            "/api/v1/auth/me/password",
            json={"new_password": "new-pw-456"},
        )
        assert resp.status_code == 422

    def test_google_only_sets_password(self, async_db_session, db_session):
        """Google-only account (no password) can set a password without current_password;
        has_password flips to true on the next GET /me."""
        user, profile = _seed_user(db_session, role="member", password=None)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.post(
            "/api/v1/auth/me/password",
            json={"new_password": "brand-new-password"},
        )
        assert resp.status_code == 200

        me_resp = client.get("/api/v1/auth/me")
        assert me_resp.status_code == 200
        assert me_resp.json()["has_password"] is True

    def test_google_only_with_current_rejects_422(self, async_db_session, db_session):
        """Providing current_password on a Google-only account (no password set) → 422."""
        user, profile = _seed_user(db_session, role="member", password=None)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.post(
            "/api/v1/auth/me/password",
            json={"current_password": "oops", "new_password": "new-pw"},
        )
        assert resp.status_code == 422

    def test_empty_new_password_422(self, async_db_session, db_session):
        """Empty new_password is rejected with 422 (validate_password_bytes: 0 bytes)."""
        user, profile = _seed_user(db_session, role="member", password=None)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.post(
            "/api/v1/auth/me/password",
            json={"new_password": ""},
        )
        assert resp.status_code == 422

    def test_oversized_new_password_422(self, async_db_session, db_session):
        """new_password exceeding 1024 UTF-8 bytes is rejected with 422."""
        user, profile = _seed_user(db_session, role="member", password=None)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        # 1025 ASCII bytes exceeds the 1024-byte KDF ceiling.
        resp = client.post(
            "/api/v1/auth/me/password",
            json={"new_password": "a" * 1025},
        )
        assert resp.status_code == 422

    def test_demo_cannot_change_password(self, async_db_session, db_session):
        """Demo role hits 403 on POST /password (require_writable guard)."""
        user, profile = _seed_user(db_session, role="demo")
        client = _make_client(async_db_session, user.id, profile.id, "demo")

        resp = client.post(
            "/api/v1/auth/me/password",
            json={"new_password": "whatever"},
        )
        assert resp.status_code == 403

    def test_repeated_wrong_password_triggers_lockout(
        self, async_db_session, db_session
    ):
        """Repeated wrong current_password attempts are recorded in the lockout store;
        once locked, the endpoint returns 401 before checking the password."""
        user, profile = _seed_user(
            db_session, role="member", password="correct-password"
        )
        store = get_lockout_store()
        canonical = user.canonical_email
        # TestClient uses "testclient" as the client hostname; get_client_ip returns it.
        tc_ip = "testclient"

        try:
            # Pre-seed the lockout store to simulate repeated failures.
            for _ in range(15):
                store.record_failure(canonical, tc_ip)

            client = _make_client(async_db_session, user.id, profile.id, "member")
            resp = client.post(
                "/api/v1/auth/me/password",
                json={"current_password": "wrong-pw", "new_password": "new-pw-123"},
            )
            assert resp.status_code == 401
            assert "Authentication failed" in resp.json()["detail"]
        finally:
            # Always clear so this test does not pollute the shared store.
            store.record_success(canonical, tc_ip)

    def test_wrong_password_records_failure_in_lockout_store(
        self, async_db_session, db_session
    ):
        """A single wrong current_password call records a failure — the (email, ip)
        pair becomes locked immediately (BASE_LOCKOUT_SECONDS applies on first failure)."""
        user, profile = _seed_user(db_session, role="member", password="correct-pw")
        store = get_lockout_store()
        canonical = user.canonical_email
        tc_ip = "testclient"

        try:
            client = _make_client(async_db_session, user.id, profile.id, "member")
            resp = client.post(
                "/api/v1/auth/me/password",
                json={"current_password": "wrong-pw", "new_password": "new-pw-123"},
            )
            assert resp.status_code == 401
            # The handler must have called record_failure; one failure locks immediately.
            assert store.is_locked(canonical, tc_ip)
        finally:
            store.record_success(canonical, tc_ip)

    def test_successful_password_change_clears_lockout_state(
        self, async_db_session, db_session
    ):
        """A successful password change calls record_success, leaving no active lockout entry."""
        old_pw = "old-password-abc"
        user, profile = _seed_user(db_session, role="member", password=old_pw)
        store = get_lockout_store()
        canonical = user.canonical_email
        tc_ip = "testclient"

        try:
            client = _make_client(async_db_session, user.id, profile.id, "member")
            resp = client.post(
                "/api/v1/auth/me/password",
                json={"current_password": old_pw, "new_password": "new-pw-xyz"},
            )
            assert resp.status_code == 200

            # record_success must have been called — no active lockout entry.
            assert not store.is_locked(canonical, tc_ip)
        finally:
            store.record_success(canonical, tc_ip)


# ---------------------------------------------------------------------------
# TestPreferences
# ---------------------------------------------------------------------------


class TestPreferences:
    def test_get_preferences_returns_defaults(self, async_db_session, db_session):
        """Fresh user with no stored preferences receives default values for all fields."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.get("/api/v1/auth/me/preferences")

        assert resp.status_code == 200
        assert resp.json() == {"landing_page": "dashboard", "date_format": "iso"}

    def test_patch_preferences_merges_fields(self, async_db_session, db_session):
        """Patching one field leaves the other at its default; re-GET confirms the merge."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        patch_resp = client.patch(
            "/api/v1/auth/me/preferences",
            json={"landing_page": "sessions"},
        )
        assert patch_resp.status_code == 200

        get_resp = client.get("/api/v1/auth/me/preferences")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["landing_page"] == "sessions"
        assert data["date_format"] == "iso"  # default unchanged

    def test_patch_unknown_key_422(self, async_db_session, db_session):
        """Unknown field in PATCH body returns 422 (UserPreferencesUpdate extra='forbid')."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.patch(
            "/api/v1/auth/me/preferences",
            json={"unknown_key": "value"},
        )
        assert resp.status_code == 422

    def test_patch_invalid_enum_value_422(self, async_db_session, db_session):
        """Invalid enum value for a preferences field returns 422."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.patch(
            "/api/v1/auth/me/preferences",
            json={"landing_page": "not_a_valid_page"},
        )
        assert resp.status_code == 422

    def test_demo_get_preferences_200(self, async_db_session, db_session):
        """Demo role can GET preferences (require_auth only, not require_writable)."""
        user, profile = _seed_user(db_session, role="demo")
        client = _make_client(async_db_session, user.id, profile.id, "demo")

        resp = client.get("/api/v1/auth/me/preferences")
        assert resp.status_code == 200

    def test_demo_patch_preferences_403(self, async_db_session, db_session):
        """Demo role hits 403 on PATCH /preferences (require_writable guard)."""
        user, profile = _seed_user(db_session, role="demo")
        client = _make_client(async_db_session, user.id, profile.id, "demo")

        resp = client.patch(
            "/api/v1/auth/me/preferences",
            json={"landing_page": "sessions"},
        )
        assert resp.status_code == 403

    def test_patch_all_null_body_is_noop_returns_current(
        self, async_db_session, db_session
    ):
        """Empty PATCH body returns 200 with the current (default) preferences unchanged."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.patch("/api/v1/auth/me/preferences", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data == {"landing_page": "dashboard", "date_format": "iso"}


# ---------------------------------------------------------------------------
# TestUnlinkGoogle
# ---------------------------------------------------------------------------


def _seed_google_identity(
    db_session: Session,
    user_id: int,
    *,
    subject: str = "google-sub-unlink-test",
) -> models.AuthIdentity:
    """Add a google auth_identity row for the given user."""
    identity = models.AuthIdentity(
        user_id=user_id,
        provider="google",
        subject=subject,
        email=None,
    )
    db_session.add(identity)
    db_session.flush()
    return identity


class TestUnlinkGoogle:
    def test_unlink_google_happy_path(self, async_db_session, db_session):
        """User with password + Google identity: DELETE succeeds, identity deleted,
        session_version incremented."""
        user, profile = _seed_user(
            db_session, role="member", password="pw-before-unlink"
        )
        _seed_google_identity(db_session, user.id)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.delete("/api/v1/auth/me/identities/google")

        assert resp.status_code == 200
        assert resp.json()["message"] == "Google account unlinked"

        # Identity rows must be gone.
        db_session.expire_all()
        from sqlalchemy import func
        from sqlalchemy import select as sel

        identity_count = db_session.execute(
            sel(func.count())
            .select_from(models.AuthIdentity)
            .where(
                models.AuthIdentity.user_id == user.id,
                models.AuthIdentity.provider == "google",
            )
        ).scalar_one()
        assert identity_count == 0, "Google identity must be deleted"

        # session_version must have been bumped.
        db_session.refresh(user)
        assert user.session_version == 1, "session_version must be incremented"

    def test_unlink_google_409_no_password(self, async_db_session, db_session):
        """User without a password gets 409 (lockout prevention)."""
        user, profile = _seed_user(db_session, role="member", password=None)
        _seed_google_identity(db_session, user.id)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.delete("/api/v1/auth/me/identities/google")

        assert resp.status_code == 409
        assert "password" in resp.json()["detail"].lower()

    def test_unlink_google_404_no_identity(self, async_db_session, db_session):
        """User with password but no Google identity gets 404."""
        user, profile = _seed_user(db_session, role="member", password="some-pw")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        resp = client.delete("/api/v1/auth/me/identities/google")

        assert resp.status_code == 404

    def test_demo_cannot_unlink(self, async_db_session, db_session):
        """Demo role hits 403 on DELETE /identities/google (require_writable guard)."""
        user, profile = _seed_user(db_session, role="demo", password="pw")
        _seed_google_identity(db_session, user.id)
        client = _make_client(async_db_session, user.id, profile.id, "demo")

        resp = client.delete("/api/v1/auth/me/identities/google")

        assert resp.status_code == 403

    def test_get_me_shows_not_linked_after_unlink(self, async_db_session, db_session):
        """GET /me after unlink returns google_linked: false."""
        user, profile = _seed_user(db_session, role="member", password="pw2")
        _seed_google_identity(db_session, user.id)
        client = _make_client(async_db_session, user.id, profile.id, "member")

        unlink_resp = client.delete("/api/v1/auth/me/identities/google")
        assert unlink_resp.status_code == 200

        me_resp = client.get("/api/v1/auth/me")
        assert me_resp.status_code == 200
        assert me_resp.json()["google_linked"] is False

    @pytest.mark.asyncio
    async def test_unlink_clears_cookie_and_old_cookie_is_rejected(
        self, temp_db, monkeypatch
    ):
        """Full multiuser stack: after unlink the response clears the cookie;
        the old session cookie is subsequently rejected with 401."""
        _multiuser_env(monkeypatch)
        cfg = load_config(
            auth_mode_override="multiuser", bind_host_override="127.0.0.1"
        )
        set_config(cfg)

        from snore.database.session import (
            cleanup_database,
            init_database,
            session_scope,
        )

        await init_database(str(temp_db))

        pw = "multiuser-unlink-test-pw"
        async with session_scope() as db:
            user = models.User(
                canonical_email=f"unlink_{uuid.uuid4().hex[:6]}@test",
                role="member",
                session_version=0,
                password_hash=hash_password(pw),
            )
            db.add(user)
            await db.flush()
            profile = models.Profile(user_id=user.id, name="Default")
            db.add(profile)
            await db.flush()
            user.default_profile_id = profile.id
            db.add(
                models.AuthIdentity(
                    user_id=user.id,
                    provider="google",
                    subject=f"gsub_{uuid.uuid4().hex[:8]}",
                    email=None,
                )
            )
            user_id = user.id
            profile_id = profile.id

        old_cookie = encode_session(
            cfg.session_secret,
            user_id=user_id,
            active_profile_id=profile_id,
            session_version=0,
        )

        app = create_app()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8000",
        ) as client:
            # Unlink Google with the old cookie.
            unlink_resp = await client.delete(
                "/api/v1/auth/me/identities/google",
                headers={
                    "origin": "http://127.0.0.1:8000",
                    "cookie": f"snore_session={old_cookie}",
                },
            )
            assert unlink_resp.status_code == 200, unlink_resp.text

            # Set-Cookie header must clear the session cookie (max-age=0).
            set_cookie_header = unlink_resp.headers.get("set-cookie", "")
            c = http.cookies.SimpleCookie()
            c.load(set_cookie_header)
            assert "snore_session" in c, (
                f"Unlink must issue a clear-cookie Set-Cookie; got: {set_cookie_header!r}"
            )
            assert c["snore_session"]["max-age"] == "0", (
                "Clear-cookie must set Max-Age=0"
            )

            # Old cookie (session_version=0) must now be rejected.
            old_resp = await client.get(
                "/api/v1/auth/me",
                headers={"cookie": f"snore_session={old_cookie}"},
            )
            assert old_resp.status_code == 401, (
                "Stale session cookie must be rejected after session_version increment"
            )

        await cleanup_database()


# ---------------------------------------------------------------------------
# TestDeleteData
# ---------------------------------------------------------------------------


def _count(db_session: Session, model: type, **filters: object) -> int:
    """Count rows matching the given filter kwargs."""
    from sqlalchemy import func  # noqa: PLC0415
    from sqlalchemy import select as sa_select

    stmt = sa_select(func.count()).select_from(model)
    for col_name, val in filters.items():
        stmt = stmt.where(getattr(model, col_name) == val)
    return db_session.execute(stmt).scalar() or 0


def _seed_device_with_session(db_session: Session, profile_id: int) -> models.Device:
    """Create a Device + one Session under the given profile; return the device."""
    device = models.Device(
        profile_id=profile_id,
        manufacturer="TestMfr",
        model="TestMdl",
        serial_number=f"SN_{uuid.uuid4().hex[:6]}",
    )
    db_session.add(device)
    db_session.flush()

    cpap_session = models.Session(
        device_id=device.id,
        device_session_id=f"sess_{uuid.uuid4().hex[:8]}",
        start_time=datetime(2025, 6, 1, 22, 0),
        end_time=datetime(2025, 6, 2, 6, 0),
        duration_seconds=8 * 3600,
    )
    db_session.add(cpap_session)
    db_session.flush()
    return device


@pytest.mark.usefixtures("reset_job_session")
class TestDeleteData:
    def test_demo_gets_403(self, async_db_session, db_session):
        """Demo role is explicitly rejected by the endpoint (403)."""
        user, profile = _seed_user(db_session, role="demo")
        client = _make_client(async_db_session, user.id, profile.id, "demo")

        resp = client.post("/api/v1/auth/me/delete-data")

        assert resp.status_code == 403
        assert not reset_jobs.has_active_job()

    def test_post_returns_202_and_deletes_only_in_the_job(
        self, async_db_session, db_session
    ):
        """The request returns a token at once; the data goes when the job runs."""
        user, profile = _seed_user(db_session, role="member")
        _seed_device_with_session(db_session, profile.id)
        db_session.commit()

        client = _make_client(async_db_session, user.id, profile.id, "member")
        resp = client.post(DELETE_DATA_PATH)

        assert resp.status_code == 202
        assert resp.json()["state"] == "queued"
        assert resp.headers["cache-control"] == "no-store"
        assert _count(db_session, models.Device, profile_id=profile.id) == 1

        run_queued_job()

        assert _count(db_session, models.Device, profile_id=profile.id) == 0

    def test_status_reports_delete_data_result(self, async_db_session, db_session):
        """Polling delete-data's token yields its result under the matching key."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        status = post_and_run_job(client, DELETE_DATA_PATH)

        assert status["kind"] == "delete_data"
        assert status["state"] == "succeeded"
        assert status["error_message"] is None
        assert status["reset_result"] is None
        assert isinstance(status["delete_data_result"]["size_after_mb"], float)
        assert "vacuum_scheduled" not in status["delete_data_result"]

    def test_member_can_delete_their_own_data(self, async_db_session, db_session):
        """Member with a device and session sees them deleted; profile and account survive."""
        user, profile = _seed_user(db_session, role="member")
        _seed_device_with_session(db_session, profile.id)
        db_session.commit()

        client = _make_client(async_db_session, user.id, profile.id, "member")
        data = run_delete_data(client)

        assert data["status"] == "success"
        assert data["devices_deleted"] >= 1
        assert data["profiles_processed"] == 1

        # Account and profile survive.
        db_session.expire_all()
        assert db_session.get(models.User, user.id) is not None
        assert db_session.get(models.Profile, profile.id) is not None

        # Sleep data is gone.
        assert _count(db_session, models.Device, profile_id=profile.id) == 0

    def test_delete_data_does_not_affect_other_users(
        self, async_db_session, db_session
    ):
        """Deleting one user's data leaves a second user's devices untouched."""
        user_a, profile_a = _seed_user(db_session, role="member")
        user_b, profile_b = _seed_user(db_session, role="member")
        _seed_device_with_session(db_session, profile_a.id)
        _seed_device_with_session(db_session, profile_b.id)
        db_session.commit()

        client_a = _make_client(async_db_session, user_a.id, profile_a.id, "member")
        run_delete_data(client_a)

        # user_b's devices must still exist.
        db_session.expire_all()
        assert _count(db_session, models.Device, profile_id=profile_b.id) >= 1, (
            "Other user's devices must not be deleted"
        )

    def test_delete_data_removes_import_job_records_for_caller(
        self, async_db_session, db_session
    ):
        """Import job records owned by the caller are deleted; others survive."""
        from datetime import UTC  # noqa: PLC0415
        from datetime import datetime as dt  # noqa: PLC0415

        user_a, profile_a = _seed_user(db_session, role="member")
        user_b, profile_b = _seed_user(db_session, role="member")

        now = dt.now(UTC)

        def _job(owner_id: int, profile_id: int) -> models.ImportJobRecord:
            return models.ImportJobRecord(
                job_id=uuid.uuid4().hex,
                job_type="upload",
                owner_user_id=owner_id,
                target_profile_id=profile_id,
                state="succeeded",
                file_count=1,
                created_at=now,
                finished_at=now,
                updated_at=now,
            )

        db_session.add(_job(user_a.id, profile_a.id))
        db_session.add(_job(user_b.id, profile_b.id))
        db_session.commit()

        client_a = _make_client(async_db_session, user_a.id, profile_a.id, "member")
        data = run_delete_data(client_a)

        assert data["import_jobs_deleted"] == 1

        db_session.expire_all()
        surviving = _count(db_session, models.ImportJobRecord)
        assert surviving == 1, "User B's import record must survive"

    def test_delete_data_with_multiple_profiles_clears_all(
        self, async_db_session, db_session
    ):
        """A user with two profiles has data deleted from both."""
        user, profile_a = _seed_user(db_session, role="member")
        # Second profile for the same user.
        profile_b = models.Profile(user_id=user.id, name="Profile B")
        db_session.add(profile_b)
        db_session.flush()
        _seed_device_with_session(db_session, profile_a.id)
        _seed_device_with_session(db_session, profile_b.id)
        db_session.commit()

        client = _make_client(async_db_session, user.id, profile_a.id, "member")
        data = run_delete_data(client)

        assert data["devices_deleted"] == 2
        assert data["profiles_processed"] == 2

        db_session.expire_all()
        assert _count(db_session, models.Device) == 0

    def test_delete_data_purges_raw_dirs(self, async_db_session, db_session, tmp_path):
        """Raw backup directories for the caller's profiles are purged."""
        from unittest.mock import patch  # noqa: PLC0415

        user, profile = _seed_user(db_session, role="member")

        raw_root = tmp_path / "raw"
        profile_dir = raw_root / str(profile.id)
        profile_dir.mkdir(parents=True)
        (profile_dir / "data.edf").write_text("fake")

        client = _make_client(async_db_session, user.id, profile.id, "member")
        with patch("snore.api.reset_jobs._raw_root", return_value=raw_root):
            run_delete_data(client)

        assert not profile_dir.exists(), "Profile raw dir must be purged"

    def test_delete_data_empty_account_succeeds(self, async_db_session, db_session):
        """User with no devices or sessions gets a success with zero counts."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        data = run_delete_data(client)

        assert data["status"] == "success"
        assert data["devices_deleted"] == 0
        assert data["import_jobs_deleted"] == 0

    def test_409_while_delete_data_job_active(self, async_db_session, db_session):
        """A second delete-data returns 409 without queueing another job."""
        user, profile = _seed_user(db_session, role="member")
        client = _make_client(async_db_session, user.id, profile.id, "member")

        assert client.post(DELETE_DATA_PATH).status_code == 202
        resp = client.post(DELETE_DATA_PATH)

        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"]
        assert len(reset_jobs._queue) == 1

    def test_409_while_reset_job_active(self, async_db_session, db_session):
        """A queued admin reset also blocks a member's delete-data."""
        admin, admin_profile = _seed_user(db_session, role="admin")
        member, member_profile = _seed_user(db_session, role="member")
        admin_client = _make_client(
            async_db_session, admin.id, admin_profile.id, "admin"
        )
        member_client = _make_client(
            async_db_session, member.id, member_profile.id, "member"
        )

        assert admin_client.post(RESET_PATH).status_code == 202
        resp = member_client.post(DELETE_DATA_PATH)

        assert resp.status_code == 409
        assert "already in progress" in resp.json()["detail"]

    def test_failed_delete_reports_generic_error_and_keeps_data(
        self, async_db_session, db_session
    ):
        """An unexpected exception fails the job without exposing its text."""
        from unittest.mock import patch  # noqa: PLC0415

        from snore.services.database_service import (  # noqa: PLC0415
            DatabaseService,
        )

        user, profile = _seed_user(db_session, role="member")
        _seed_device_with_session(db_session, profile.id)
        db_session.commit()
        client = _make_client(async_db_session, user.id, profile.id, "member")

        with patch.object(
            DatabaseService,
            "delete_user_data",
            side_effect=RuntimeError("/data/.snore/raw/1: disk full"),
        ):
            job_id = client.post(DELETE_DATA_PATH).json()["job_id"]
            run_queued_job()

        status = read_status(client, job_id).json()
        assert status["state"] == "failed"
        assert status["error_message"] == reset_jobs._FAILURE_MESSAGE
        assert "disk full" not in status["error_message"]
        assert _count(db_session, models.Device, profile_id=profile.id) == 1

    def test_delete_data_writes_vacuum_marker(
        self, async_db_session, db_session, tmp_path
    ):
        """POST /auth/me/delete-data writes a vacuum pending marker before committing.

        Patches DatabaseTarget.from_env_and_flags to return a known SQLite file
        target so the job carries a db_path, ensuring the marker write branch is
        exercised.
        """
        from unittest.mock import patch  # noqa: PLC0415

        from snore.database.target import DatabaseTarget  # noqa: PLC0415

        user, profile = _seed_user(db_session, role="member")
        _seed_device_with_session(db_session, profile.id)
        db_session.commit()
        client = _make_client(async_db_session, user.id, profile.id, "member")

        # Use a real file target so the job has a db_path.
        db_file = tmp_path / "test.db"
        db_file.touch()
        marker = tmp_path / "vacuum.pending"

        fake_target = DatabaseTarget.from_url(str(db_file))

        with (
            patch("snore.api.reset_jobs.DEFAULT_VACUUM_PENDING_MARKER", marker),
            patch(
                "snore.database.target.DatabaseTarget.from_env_and_flags",
                return_value=fake_target,
            ),
        ):
            run_delete_data(client)

        # _vacuum_background is patched to no-op in tests, so marker persists.
        assert marker.exists(), (
            "Vacuum pending marker must be written by delete-data endpoint"
        )
        assert marker.read_text().strip(), "Marker must contain a non-empty DB path"
