"""MCP OAuth authentication: token → ActorContext resolution.

This module is the single point where a validated FastMCP AccessToken is
mapped to a SNORE ActorContext.  It sits between fastmcp's auth middleware
(which validates bearer tokens and populates the request scope) and tool
code (which calls current_actor() to retrieve the resolved context).

Public API consumed by server.py:
- make_auth_provider  — construct the GoogleProvider for HTTP transport
- actor_scope         — _ScopeProvider seam: resolves actor per request
- current_actor       — retrieve actor bound by actor_scope()
- resolve_actor       — async token→ActorContext mapping (also tested directly)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import httpx2

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.auth.actor import ActorContext, AuthMode
from snore.auth.factory import ActorContextFactory
from snore.database import models
from snore.database.session import session_scope

if TYPE_CHECKING:
    from fastmcp.server.auth.auth import AccessToken, AuthProvider

GOOGLE_PROVIDER = "google"  # auth_identities.provider column value for Google OAuth

# Per-request ContextVar holding the resolved actor; None when no request is active.
_current_actor: ContextVar[ActorContext | None] = ContextVar(
    "_current_actor", default=None
)


def make_auth_provider(
    *,
    base_url: str,
    google_client_id: str,
    google_client_secret: str,
    http_client: httpx2.AsyncClient | None = None,
) -> AuthProvider:
    """Construct a GoogleProvider for HTTP transport.

    Args:
        base_url:            Public server URL (SNORE_PUBLIC_BASE_URL).  Must be
                             HTTPS for non-loopback hosts; plain HTTP is allowed
                             for loopback addresses (127.x.x.x, ::1, localhost).
        google_client_id:    OAuth client ID (GOOGLE_CLIENT_ID).
        google_client_secret: OAuth client secret (GOOGLE_CLIENT_SECRET).
        http_client:         httpx2.AsyncClient for connection pooling to Google
                             endpoints.  When None (default), a long-lived client
                             is created inside this function and passed to
                             GoogleProvider.  The client is intentionally
                             process-lifetime: the MCP server process owns it and
                             connections close with the process.  Pass an explicit
                             client in tests to control the transport.

    Returns:
        Configured GoogleProvider instance.

    Raises:
        ValueError: If any argument is empty, blank, or if base_url violates
                    the loopback-HTTP / HTTPS-required policy.

    Performance note:
        fastmcp performs both a tokeninfo and a userinfo call to Google on every
        authenticated request (``OAuthProxy.load_access_token`` delegates to
        ``GoogleTokenVerifier.verify_token``; the userinfo result is not used by
        SNORE).  There is no token_verifier construction seam in fastmcp to skip
        the userinfo call without patching internals.  Google tokeninfo outages
        or rate-limits will therefore affect all authenticated requests.
        Connection pooling via the shared ``http_client`` reuses TLS connections
        to Google's endpoints across requests.

    Deployment note:
        Dynamic client registration accepts any client redirect URI by design
        (MCP spec pattern, fastmcp ``allowed_client_redirect_uris=None`` default);
        operators serving untrusted multi-tenant contexts should restrict it.
    """
    from fastmcp.server.auth.providers.google import GoogleProvider  # noqa: PLC0415

    from snore.api.config import (  # noqa: PLC0415
        ConfigError,
        validate_origin_url,
    )

    _require_nonempty(base_url, "base_url", "SNORE_PUBLIC_BASE_URL")
    _require_nonempty(google_client_id, "google_client_id", "GOOGLE_CLIENT_ID")
    _require_nonempty(
        google_client_secret, "google_client_secret", "GOOGLE_CLIENT_SECRET"
    )

    try:
        validate_origin_url(base_url, require_http_loopback=True)
    except ConfigError as exc:
        raise ValueError(
            f"SNORE_PUBLIC_BASE_URL must be a valid HTTPS (or loopback HTTP) URL: {exc}"
        ) from exc

    _client = http_client if http_client is not None else httpx2.AsyncClient()
    return GoogleProvider(
        client_id=google_client_id,
        client_secret=google_client_secret,
        base_url=base_url,
        http_client=_client,
    )


def _require_nonempty(value: str, arg_name: str, env_var: str) -> None:
    if not value or not value.strip():
        raise ValueError(
            f"{arg_name!r} must not be empty or blank "
            f"(set the {env_var} environment variable)"
        )


async def resolve_actor(db: AsyncSession, token: AccessToken) -> ActorContext:
    """Map a validated FastMCP AccessToken to an ActorContext.

    Steps:
    1. Extract ``sub`` from token claims (missing/empty → ToolError).
    2. Look up ``auth_identities`` WHERE provider='google' AND subject=sub
       (miss → ToolError with a user-actionable message).
    3. Build ActorContext via ActorContextFactory (ValueError → ToolError).

    ToolError messages are client-visible.  They are sanitised: they never
    expose user IDs, email addresses, Google subject values, or any internal
    DB state.

    Args:
        db:    Open AsyncSession (within an active transaction).
        token: Validated FastMCP AccessToken from the bearer middleware.

    Returns:
        Resolved ActorContext for the authenticated user.

    Raises:
        fastmcp.exceptions.ToolError: On any identity-resolution failure.
    """
    sub = token.claims.get("sub")
    if not sub or not isinstance(sub, str):
        raise ToolError(
            "Authentication token is missing the required 'sub' claim. "
            "Re-authenticate and try again."
        )

    stmt = select(models.AuthIdentity).where(
        models.AuthIdentity.provider == GOOGLE_PROVIDER,
        models.AuthIdentity.subject == sub,
    )
    identity = (await db.execute(stmt)).scalars().first()
    if identity is None:
        raise ToolError(
            "No SNORE account is linked to this Google identity. "
            "Sign in to the SNORE web app first."
        )

    user = await db.get(models.User, identity.user_id)
    if user is not None and user.disabled_at is not None:
        raise ToolError("This account is disabled. Contact your administrator.")

    try:
        actor = await ActorContextFactory(db).make(
            identity.user_id, None, AuthMode.MULTIUSER
        )
    except ValueError as exc:
        raise ToolError(
            "Unable to resolve your SNORE account. Contact your administrator."
        ) from exc

    return actor


def current_actor() -> ActorContext:
    """Return the ActorContext bound by the active actor_scope().

    Raises:
        RuntimeError: If called outside an active actor_scope() — programming error.
    """
    actor = _current_actor.get()
    if actor is None:
        raise RuntimeError(
            "current_actor() called outside an active actor_scope(). "
            "Ensure actor_scope() wraps the tool call."
        )
    return actor


@asynccontextmanager
async def actor_scope() -> AsyncGenerator[AsyncSession]:
    """Scope provider implementing the _ScopeProvider seam.

    1. Retrieve the FastMCP AccessToken from the current request context.
       Raises ToolError("Authentication required…") when unauthenticated.
    2. Open a database transaction via session_scope().
    3. Resolve the token to an ActorContext via resolve_actor().
    4. Bind the actor to _current_actor for the duration of the scope,
       then reset it in the finally block.

    Yields:
        The open AsyncSession (same as session_scope()).

    Raises:
        fastmcp.exceptions.ToolError: If unauthenticated or identity cannot
            be resolved.
    """
    token = get_access_token()
    if token is None:
        raise ToolError(
            "Authentication required. Include a valid bearer token in the request."
        )

    async with session_scope() as db:
        actor = await resolve_actor(db, token)
        token_var = _current_actor.set(actor)
        try:
            yield db
        finally:
            _current_actor.reset(token_var)
