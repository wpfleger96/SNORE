"""Embed the FastMCP streamable-HTTP server into the FastAPI app.

``build_mcp_app`` returns the FastMCP Starlette sub-application when the
config enables MCP (multiuser + Google OAuth), or None when any prerequisite
is absent.  When None is returned, the caller skips mounting entirely.

The MCP OAuth issuer is ``SNORE_PUBLIC_BASE_URL`` — no separate MCP base URL
is needed.  The endpoint is served at ``{public_base_url}/mcp``.

``is_mcp_path`` identifies URL paths owned by the embedded FastMCP OAuth app
so that CSRF middleware and the SPA fallback leave them alone.

Import cycle safety: this module imports from snore.mcp (server, auth) and
snore.api.config, but never from snore.api.middleware — middleware must import
is_mcp_path from here, not the other way around.

Session idle timeout: a stateful MCP session with no request in flight for
``MCP_SESSION_IDLE_TIMEOUT_SECONDS`` is terminated and reaped; the client then
gets 404 for that session ID and re-initializes.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastmcp.server.auth import AuthProvider
    from fastmcp.server.http import StarletteWithLifespan

    from snore.api.config import AppConfig

logger = logging.getLogger(__name__)

MCP_PATH = "/mcp"
# mcp SDK's DEFAULT_SESSION_IDLE_TIMEOUT (30 min).  fastmcp passes None to the SDK
# unless told otherwise, which disables reaping, so set it explicitly.
MCP_SESSION_IDLE_TIMEOUT_SECONDS = 30 * 60
_MCP_PATH_PREFIX = MCP_PATH + "/"

# Root-level OAuth flow paths registered by fastmcp 4.0.11 GoogleProvider /
# OAuthProxy (/revoke is not registered but kept defensively).  Audit this set
# on fastmcp upgrades.
_MCP_ROOT_PATHS = frozenset(
    {"/authorize", "/token", "/register", "/revoke", "/consent", "/auth/callback"}
)


def is_mcp_path(path: str) -> bool:
    """True for any path owned by the embedded FastMCP OAuth app.

    Covers:
    - /mcp and all sub-paths (/mcp/*)
    - /.well-known/* (intentionally broader than the two registered routes —
      covers any future discovery documents fastmcp may add)
    - OAuth flow root paths (/authorize, /token, etc.)
    """
    return (
        path == MCP_PATH
        or path.startswith(_MCP_PATH_PREFIX)
        or path.startswith("/.well-known/")
        or path in _MCP_ROOT_PATHS
    )


def build_mcp_app(cfg: AppConfig) -> StarletteWithLifespan | None:
    """Build the FastMCP Starlette sub-app for embedding, or None when disabled.

    When None is returned, the caller must not mount anything — /mcp will 404
    like any other unmounted path.

    Logging policy:
    - multiuser but Google creds absent → WARNING (mirrors app.py:84-88)
    - local mode → no log (silent no-op)
    - MCP enabled → INFO with endpoint URL
    """
    if not cfg.is_multiuser:
        return None

    if not cfg.is_google_configured:
        logger.warning(
            "MCP disabled: Google OAuth not configured "
            "(GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET absent)"
        )
        return None

    from snore.mcp.server import make_server  # noqa: PLC0415

    auth = _make_mcp_auth_provider(cfg)
    server = make_server(profile_name="neutral", auth=auth, manage_database=False)
    app = server.http_app(
        path=MCP_PATH,
        host_origin_protection=False,
        session_idle_timeout=MCP_SESSION_IDLE_TIMEOUT_SECONDS,
    )

    logger.info(
        "Embedded MCP server enabled — endpoint: %s%s",
        cfg.public_base_url.rstrip("/"),
        MCP_PATH,
    )
    return app


def _make_mcp_auth_provider(cfg: AppConfig) -> AuthProvider:
    """Construct the GoogleProvider for the embedded MCP server.

    Uses public_base_url as the OAuth issuer — the MCP endpoint is same-origin
    with the main app.  Extracted as a module-level function so tests can
    monkeypatch it and substitute a JWTVerifier without touching Google APIs.
    """
    from snore.mcp.auth import make_auth_provider  # noqa: PLC0415

    return make_auth_provider(
        base_url=cfg.public_base_url,
        google_client_id=cfg.google_client_id,
        google_client_secret=cfg.google_client_secret,
    )
