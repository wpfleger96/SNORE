"""Unit tests for snore.mcp.auth — token → ActorContext resolution."""

from __future__ import annotations

import uuid

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from snore.auth.actor import ActorContext, AuthMode, Role
from snore.database import models

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def _db(tmp_path: Path) -> AsyncGenerator[Any]:
    """Isolated SQLite DB wired into snore.database.session globals.

    Calls init_database / cleanup_database so that session_scope() inside
    actor_scope() lands in the same file as the test data.
    """
    import snore.database.session as db_mod  # noqa: PLC0415

    db_path = str(tmp_path / f"auth_test_{uuid.uuid4().hex[:8]}.db")
    await db_mod.cleanup_database()
    await db_mod.init_database(db_path)
    yield
    await db_mod.cleanup_database()


@pytest.fixture()
async def seeded_db(_db: Any) -> dict[str, Any]:
    """Seed a user + two profiles + auth_identity and return their IDs.

    Layout:
      - user: admin, not disabled, default_profile_id = profile2.id
      - profile1: first live profile (lower id)
      - profile2: second live profile (higher id), set as default
      - auth_identity: provider='google', subject='google-sub-001'
    """
    from snore.database.session import session_scope  # noqa: PLC0415

    async with session_scope() as db:
        user = models.User(
            canonical_email=f"auth_{uuid.uuid4().hex[:8]}@example.com",
            role="admin",
        )
        db.add(user)
        await db.flush()

        profile1 = models.Profile(user_id=user.id, name="Profile One")
        db.add(profile1)
        await db.flush()

        profile2 = models.Profile(user_id=user.id, name="Profile Two")
        db.add(profile2)
        await db.flush()

        user.default_profile_id = profile2.id
        await db.flush()

        identity = models.AuthIdentity(
            user_id=user.id,
            provider="google",
            subject="google-sub-001",
            email=user.canonical_email,
        )
        db.add(identity)
        await db.flush()

        return {
            "user_id": user.id,
            "profile1_id": profile1.id,
            "profile2_id": profile2.id,
            "sub": "google-sub-001",
        }


@pytest.fixture()
async def seeded_demo_user(_db: Any) -> dict[str, Any]:
    """Seed a demo-role user + profile + auth_identity."""
    from snore.database.session import session_scope  # noqa: PLC0415

    async with session_scope() as db:
        user = models.User(
            canonical_email=f"demo_{uuid.uuid4().hex[:8]}@example.com",
            role="demo",
        )
        db.add(user)
        await db.flush()

        profile = models.Profile(user_id=user.id, name="Demo Profile")
        db.add(profile)
        await db.flush()

        user.default_profile_id = profile.id
        await db.flush()

        sub = f"google-demo-{uuid.uuid4().hex[:8]}"
        identity = models.AuthIdentity(
            user_id=user.id,
            provider="google",
            subject=sub,
            email=user.canonical_email,
        )
        db.add(identity)
        await db.flush()

        return {"user_id": user.id, "profile_id": profile.id, "sub": sub}


@pytest.fixture()
async def seeded_disabled_user(_db: Any) -> dict[str, Any]:
    """Seed a disabled user + profile + auth_identity."""
    from datetime import UTC, datetime

    from snore.database.session import session_scope  # noqa: PLC0415

    async with session_scope() as db:
        user = models.User(
            canonical_email=f"disabled_{uuid.uuid4().hex[:8]}@example.com",
            role="member",
            disabled_at=datetime.now(UTC),
        )
        db.add(user)
        await db.flush()

        profile = models.Profile(user_id=user.id, name="Disabled Profile")
        db.add(profile)
        await db.flush()

        sub = f"google-disabled-{uuid.uuid4().hex[:8]}"
        identity = models.AuthIdentity(
            user_id=user.id,
            provider="google",
            subject=sub,
            email=user.canonical_email,
        )
        db.add(identity)
        await db.flush()

        return {"user_id": user.id, "sub": sub}


def _fake_token(claims: dict[str, Any]) -> Any:
    """Build a minimal AccessToken-like object with the given claims dict."""
    token = MagicMock()
    token.claims = claims
    return token


# ---------------------------------------------------------------------------
# resolve_actor tests
# ---------------------------------------------------------------------------


class TestResolveActorKnownIdentity:
    async def test_resolve_actor_known_identity_returns_actor(
        self, seeded_db: dict[str, Any]
    ) -> None:
        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        token = _fake_token({"sub": seeded_db["sub"]})

        async with session_scope() as db:
            actor = await resolve_actor(db, token)

        assert isinstance(actor, ActorContext)
        assert actor.user_id == seeded_db["user_id"]
        assert actor.mode is AuthMode.MULTIUSER
        assert actor.role is Role.ADMIN

    async def test_resolve_actor_profile_fallback_uses_default_profile(
        self, seeded_db: dict[str, Any]
    ) -> None:
        """ActorContextFactory falls back to user.default_profile_id (profile2)."""
        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        token = _fake_token({"sub": seeded_db["sub"]})

        async with session_scope() as db:
            actor = await resolve_actor(db, token)

        # default_profile_id is profile2; the factory should resolve to it.
        assert actor.profile_id == seeded_db["profile2_id"]

    async def test_resolve_actor_demo_role_maps(
        self, seeded_demo_user: dict[str, Any]
    ) -> None:
        """Demo-role actor has can_write == False."""
        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        token = _fake_token({"sub": seeded_demo_user["sub"]})

        async with session_scope() as db:
            actor = await resolve_actor(db, token)

        assert actor.role is Role.DEMO
        assert actor.can_write is False


class TestResolveActorErrors:
    async def test_resolve_actor_missing_sub_claim_raises_tool_error(
        self, _db: Any
    ) -> None:
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        token = _fake_token({})  # no 'sub' key

        async with session_scope() as db:
            with pytest.raises(ToolError, match="sub"):
                await resolve_actor(db, token)

    async def test_resolve_actor_unknown_subject_raises_tool_error(
        self, _db: Any
    ) -> None:
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        unknown_sub = "google-nonexistent-sub-xyz"
        token = _fake_token({"sub": unknown_sub})

        async with session_scope() as db:
            with pytest.raises(ToolError) as exc_info:
                await resolve_actor(db, token)

        # Message must not contain the subject value.
        assert unknown_sub not in str(exc_info.value)
        assert "SNORE account" in str(exc_info.value) or "Sign in" in str(
            exc_info.value
        )

    async def test_resolve_actor_disabled_user_raises_tool_error(
        self, seeded_disabled_user: dict[str, Any]
    ) -> None:
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        token = _fake_token({"sub": seeded_disabled_user["sub"]})

        async with session_scope() as db:
            with pytest.raises(ToolError) as exc_info:
                await resolve_actor(db, token)

        msg = str(exc_info.value)
        # Must not leak the user id.
        assert str(seeded_disabled_user["user_id"]) not in msg
        # Must be actionable.
        assert "disabled" in msg.lower()

    async def test_resolve_actor_tombstoned_profile_raises_generic_tool_error(
        self, _db: Any
    ) -> None:
        """Active user whose only profile is tombstoned → generic ToolError, no user id."""
        from datetime import UTC, datetime

        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        async with session_scope() as db:
            user = models.User(
                canonical_email=f"tombstone_{uuid.uuid4().hex[:8]}@example.com",
                role="member",
            )
            db.add(user)
            await db.flush()

            profile = models.Profile(
                user_id=user.id,
                name="Tombstoned Profile",
                deleting_at=datetime.now(UTC),
            )
            db.add(profile)
            await db.flush()

            sub = f"google-tombstone-{uuid.uuid4().hex[:8]}"
            identity = models.AuthIdentity(
                user_id=user.id,
                provider="google",
                subject=sub,
                email=user.canonical_email,
            )
            db.add(identity)
            await db.flush()
            user_id = user.id

        token = _fake_token({"sub": sub})

        async with session_scope() as db:
            with pytest.raises(ToolError) as exc_info:
                await resolve_actor(db, token)

        msg = str(exc_info.value)
        # Must not leak user id.
        assert str(user_id) not in msg
        # Must be the generic message, not the disabled message.
        assert "disabled" not in msg.lower()
        assert "Unable to resolve" in msg or "administrator" in msg

    async def test_resolve_actor_cross_provider_identity_not_matched(
        self, _db: Any
    ) -> None:
        """AuthIdentity with provider='github' and same sub must not resolve via Google lookup."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        sub = f"cross-provider-{uuid.uuid4().hex[:8]}"

        async with session_scope() as db:
            user = models.User(
                canonical_email=f"github_{uuid.uuid4().hex[:8]}@example.com",
                role="member",
            )
            db.add(user)
            await db.flush()

            profile = models.Profile(user_id=user.id, name="GitHub Profile")
            db.add(profile)
            await db.flush()

            # Seed a GitHub identity with the same subject value.
            identity = models.AuthIdentity(
                user_id=user.id,
                provider="github",
                subject=sub,
                email=user.canonical_email,
            )
            db.add(identity)
            await db.flush()

        # Token claims sub matching the GitHub identity's subject.
        token = _fake_token({"sub": sub})

        async with session_scope() as db:
            with pytest.raises(ToolError) as exc_info:
                await resolve_actor(db, token)

        msg = str(exc_info.value)
        # Must not match the GitHub identity — provider filter is Google only.
        assert sub not in msg
        assert "No SNORE account" in msg or "Sign in" in msg

    async def test_resolve_actor_empty_string_sub_raises_tool_error(
        self, _db: Any
    ) -> None:
        """Token with sub='' is treated the same as a missing sub claim."""
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        from snore.database.session import session_scope  # noqa: PLC0415
        from snore.mcp.auth import resolve_actor  # noqa: PLC0415

        token = _fake_token({"sub": ""})  # empty string — falsy, same guard as missing

        async with session_scope() as db:
            with pytest.raises(ToolError, match="sub"):
                await resolve_actor(db, token)


# ---------------------------------------------------------------------------
# actor_scope tests
# ---------------------------------------------------------------------------


class TestActorScope:
    async def test_actor_scope_no_token_raises_tool_error(
        self, _db: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastmcp.exceptions import ToolError  # noqa: PLC0415

        import snore.mcp.auth as auth_mod  # noqa: PLC0415

        monkeypatch.setattr(auth_mod, "get_access_token", lambda: None)

        with pytest.raises(ToolError, match="Authentication required"):
            async with auth_mod.actor_scope():
                pass  # pragma: no cover

    async def test_actor_scope_binds_and_resets_actor(
        self, seeded_db: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside the scope current_actor() returns the actor; after exit it raises."""
        import snore.mcp.auth as auth_mod  # noqa: PLC0415

        fake_token = _fake_token({"sub": seeded_db["sub"]})
        monkeypatch.setattr(auth_mod, "get_access_token", lambda: fake_token)

        actor_inside: ActorContext | None = None

        async with auth_mod.actor_scope():
            actor_inside = auth_mod.current_actor()
            assert isinstance(actor_inside, ActorContext)
            assert actor_inside.user_id == seeded_db["user_id"]

        # After the scope exits, _current_actor is reset → current_actor() raises.
        with pytest.raises(RuntimeError, match="actor_scope"):
            auth_mod.current_actor()

    async def test_actor_scope_resets_actor_when_body_raises(
        self, seeded_db: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ContextVar is reset even when the body of actor_scope() raises an exception."""
        import snore.mcp.auth as auth_mod  # noqa: PLC0415

        fake_token = _fake_token({"sub": seeded_db["sub"]})
        monkeypatch.setattr(auth_mod, "get_access_token", lambda: fake_token)

        class _Sentinel(Exception):
            pass

        with pytest.raises(_Sentinel):
            async with auth_mod.actor_scope():
                raise _Sentinel("body raised")

        # The finally block in actor_scope() must have reset the ContextVar.
        with pytest.raises(RuntimeError, match="actor_scope"):
            auth_mod.current_actor()


# ---------------------------------------------------------------------------
# current_actor tests
# ---------------------------------------------------------------------------


class TestCurrentActor:
    def test_current_actor_outside_scope_raises_runtime_error(self) -> None:
        from snore.mcp.auth import current_actor  # noqa: PLC0415

        with pytest.raises(RuntimeError, match="actor_scope"):
            current_actor()


# ---------------------------------------------------------------------------
# make_auth_provider tests
# ---------------------------------------------------------------------------


class TestMakeAuthProvider:
    def test_make_auth_provider_blank_base_url_raises_value_error(self) -> None:
        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        with pytest.raises(ValueError, match="SNORE_PUBLIC_BASE_URL"):
            make_auth_provider(
                base_url="   ",
                google_client_id="client-id",
                google_client_secret="secret",
            )

    def test_make_auth_provider_blank_client_id_raises_value_error(self) -> None:
        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        with pytest.raises(ValueError, match="GOOGLE_CLIENT_ID"):
            make_auth_provider(
                base_url="https://example.com",
                google_client_id="",
                google_client_secret="secret",
            )

    def test_make_auth_provider_blank_client_secret_raises_value_error(self) -> None:
        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        with pytest.raises(ValueError, match="GOOGLE_CLIENT_SECRET"):
            make_auth_provider(
                base_url="https://example.com",
                google_client_id="client-id",
                google_client_secret="",
            )

    def test_make_auth_provider_http_non_loopback_rejected(self) -> None:
        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        with pytest.raises(ValueError, match="SNORE_PUBLIC_BASE_URL"):
            make_auth_provider(
                base_url="http://example.com",
                google_client_id="client-id",
                google_client_secret="secret",
            )

    def test_make_auth_provider_http_loopback_accepted(self) -> None:
        from fastmcp.server.auth.auth import AuthProvider  # noqa: PLC0415

        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        provider = make_auth_provider(
            base_url="http://127.0.0.1:8321",
            google_client_id="valid-client-id",
            google_client_secret="valid-client-secret",
        )
        assert isinstance(provider, AuthProvider)

    def test_make_auth_provider_https_non_loopback_accepted(self) -> None:
        from fastmcp.server.auth.auth import AuthProvider  # noqa: PLC0415

        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        provider = make_auth_provider(
            base_url="https://mcp.example.com",
            google_client_id="valid-client-id",
            google_client_secret="valid-client-secret",
        )
        assert isinstance(provider, AuthProvider)

    def test_make_auth_provider_returns_provider(self) -> None:
        from fastmcp.server.auth.auth import AuthProvider  # noqa: PLC0415

        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        provider = make_auth_provider(
            base_url="https://example.com",
            google_client_id="valid-client-id",
            google_client_secret="valid-client-secret",
        )

        assert isinstance(provider, AuthProvider)
        # Pin that the configured base URL was wired into the provider.
        assert str(provider.base_url).rstrip("/") == "https://example.com"

    # Token verification goes through the provider's private ``_token_validator``
    # (fastmcp's GoogleTokenVerifier): the public ``verify_token`` only accepts
    # FastMCP-issued JWTs backed by a stored upstream token from a full OAuth flow.

    async def test_injected_client_transport_error_rejects_token(self) -> None:
        import httpx2  # noqa: PLC0415

        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        requested: list[str] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            requested.append(str(request.url))
            raise httpx2.ConnectError("unreachable", request=request)

        provider = make_auth_provider(
            base_url="https://example.com",
            google_client_id="valid-client-id",
            google_client_secret="valid-client-secret",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )

        assert await provider._token_validator.verify_token("google-token") is None
        assert requested, "the injected client was not used"

    async def test_injected_client_valid_tokeninfo_returns_access_token(
        self,
    ) -> None:
        import httpx2  # noqa: PLC0415

        from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.url.path == "/tokeninfo":
                return httpx2.Response(
                    200,
                    json={
                        "aud": "valid-client-id",
                        "sub": "google-sub-123",
                        "scope": "openid https://www.googleapis.com/auth/userinfo.email",
                        "expires_in": "3599",
                        "email": "user@example.com",
                    },
                )
            if request.url.path == "/oauth2/v2/userinfo":
                return httpx2.Response(200, json={"name": "Test User"})
            return httpx2.Response(404)

        provider = make_auth_provider(
            base_url="https://example.com",
            google_client_id="valid-client-id",
            google_client_secret="valid-client-secret",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        )

        token = await provider._token_validator.verify_token("google-token")

        assert token is not None
        assert token.subject == "google-sub-123"
        assert token.claims["sub"] == "google-sub-123"
        assert token.claims["name"] == "Test User"
