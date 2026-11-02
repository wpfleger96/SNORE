"""Route inventory test: verify demo actors receive 403 on all mutating routes.

Enumerates app.routes at runtime via the OpenAPI schema and asserts that a
demo-role actor gets 403 on every POST/PUT/PATCH/DELETE route except an
explicit allowlist.  Any NEW mutating route added later fails this test until
it is classified in the allowlist.

Guard ordering note: FastAPI dependencies run before body validation, so
require_writable raises 403 before the request body is validated (422).
Tests can send empty bodies safely.
"""

from __future__ import annotations

from typing import Annotated

import pytest

from fastapi import Depends
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from snore.api import reset_jobs
from snore.api.app import create_app
from snore.api.deps import get_actor, get_db, get_db_immediate
from snore.auth.actor import ActorContext, AuthMode

# ---------------------------------------------------------------------------
# Routes that demo actors are explicitly permitted to reach.
# POST/PUT/PATCH/DELETE routes NOT in this set must return 403 for demo.
# ---------------------------------------------------------------------------
DEMO_ALLOWED_MUTATING = {
    # Public auth endpoints — no actor required at all.
    "POST /api/v1/auth/login",
    "POST /api/v1/auth/login/totp",
    "POST /api/v1/auth/logout",
    "POST /api/v1/auth/demo-login",
    "POST /api/v1/auth/invites/lookup",
    "POST /api/v1/auth/invites/redeem",
    "POST /api/v1/auth/invites/google",
    # Demo may switch profiles (RequireAuth only, not RequireWritable).
    "POST /api/v1/auth/active-profile",
    # Read-only previews / queries — no data is mutated.
    "POST /api/v1/sessions/delete-preview",
    "POST /api/v1/import/precheck",
    # Signal validation endpoints are read-only (no writes performed).
    "POST /api/v1/validate/fl",
    "POST /api/v1/validate/breaths",
    "POST /api/v1/validate/apple",
    "POST /api/v1/validate/rera",
}

# HTTP methods that are considered mutating.
_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


@pytest.fixture
def demo_client(temp_db, async_db_session, db_session):
    """TestClient with get_db and get_actor overridden to a demo actor.

    The demo actor has role=DEMO in MULTIUSER mode, so require_writable raises
    403 on any mutating route that enforces it.

    The fixture uses local mode (test default from conftest.py) because
    can_write derives from the actor's role, not from the auth mode.  A demo
    actor in local mode is structurally identical to one in multiuser mode for
    the purpose of this guard test.
    """
    app = create_app()

    async def override_get_db():
        async with async_db_session.begin():
            yield async_db_session

    async def fail_if_immediate_db_resolved() -> None:
        raise AssertionError("demo request acquired an immediate transaction")

    async def override_get_actor(
        db: Annotated[AsyncSession, Depends(get_db)],
    ) -> ActorContext:
        # Provision a minimal user + profile so the context factory can build
        # a real ActorContext — some routes look up the actor's data via the DB.
        # Seed a demo user + profile if not already present.
        from sqlalchemy import select  # noqa: PLC0415

        from snore.auth.factory import ActorContextFactory  # noqa: PLC0415
        from snore.database.models import Profile, User  # noqa: PLC0415

        stmt = select(User).where(User.canonical_email == "demo-test@example.com")
        user = (await db.execute(stmt)).scalars().first()
        if user is None:
            user = User(
                canonical_email="demo-test@example.com",
                role="demo",
                session_version=0,
            )
            db.add(user)
            await db.flush()
            profile = Profile(user_id=user.id, name="Demo")
            db.add(profile)
            await db.flush()
            user.default_profile_id = profile.id
            await db.flush()

        factory = ActorContextFactory(db)
        return await factory.make(
            user_id=user.id,
            active_profile_id=user.default_profile_id,
            mode=AuthMode.MULTIUSER,
        )

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_db_immediate] = fail_if_immediate_db_resolved
    app.dependency_overrides[get_actor] = override_get_actor

    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


def _mutating_routes() -> list[tuple[str, str]]:
    """Return (method, path) for all mutating routes in the live app."""
    from snore.api.app import create_app  # noqa: PLC0415

    app = create_app()
    schema = app.openapi()
    results = []
    for path, methods in schema["paths"].items():
        for method in methods:
            if method.upper() in _MUTATING:
                results.append((method.upper(), path))
    return results


class TestDemoEnforcement:
    """Verify that demo actors cannot reach write endpoints."""

    def test_all_mutating_routes_classified(self, demo_client):
        """Every mutating route is either blocked (403) or in the allowlist.

        This test fails when a new mutating route is added without being
        classified — forcing the developer to make an explicit decision.
        """
        mutating = _mutating_routes()
        assert mutating, "No mutating routes found — route enumeration broken"

        live_route_keys = {f"{method} {path}" for method, path in mutating}

        # Guard against a stale allowlist: every entry must match a live route.
        stale_allowlist = DEMO_ALLOWED_MUTATING - live_route_keys
        assert not stale_allowlist, (
            "DEMO_ALLOWED_MUTATING contains entries that no longer match any "
            "registered route (remove stale entries):\n"
            + "\n".join(f"  {e}" for e in sorted(stale_allowlist))
        )

        unclassified_failures: list[str] = []

        for method, path in mutating:
            route_key = f"{method} {path}"
            if route_key in DEMO_ALLOWED_MUTATING:
                # Allowlisted — skip (we do not assert 2xx here; 404 etc. are fine)
                continue

            # Replace path parameters with placeholder values so the request
            # reaches the route handler (guards run before body/param validation).
            concrete_path = path
            for segment in path.split("/"):
                if segment.startswith("{") and segment.endswith("}"):
                    concrete_path = concrete_path.replace(segment, "1", 1)

            resp = demo_client.request(method, concrete_path, json={})

            if resp.status_code != 403:
                unclassified_failures.append(
                    f"{route_key} → {resp.status_code} (expected 403)"
                )

        assert not unclassified_failures, (
            "Demo actor did not receive 403 on these mutating routes:\n"
            + "\n".join(f"  {f}" for f in unclassified_failures)
            + "\nAdd each route to DEMO_ALLOWED_MUTATING or ensure require_writable"
            " is applied."
        )

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("PATCH", "/api/v1/sessions/1"),
            ("DELETE", "/api/v1/sessions/"),
            ("DELETE", "/api/v1/analysis"),
            ("PATCH", "/api/v1/equipment/masks/1"),
            ("DELETE", "/api/v1/equipment/masks/1"),
        ],
    )
    def test_immediate_routes_reject_demo_before_writer_lock(
        self, demo_client: TestClient, method: str, path: str
    ) -> None:
        """The writable guard must reject demo requests before BEGIN IMMEDIATE."""
        response = demo_client.request(method, path, json={})
        assert response.status_code == 403

    @pytest.mark.parametrize(
        "path", ["/api/v1/db/reset", "/api/v1/auth/me/delete-data"]
    )
    def test_destructive_routes_reject_demo_before_queueing_a_job(
        self, demo_client: TestClient, path: str
    ) -> None:
        assert demo_client.post(path, json={}).status_code == 403
        assert not reset_jobs.has_active_job()
