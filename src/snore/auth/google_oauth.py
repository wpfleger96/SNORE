"""Google OIDC token exchange and validation — single mockable seam.

All real HTTP calls to Google are isolated in ``fetch_google_id_token_claims``.
Tests mock this one function; the route logic itself is never touched.

Usage in tests::

    monkeypatch.setattr(
        "snore.api.routers.auth.routes_google.fetch_google_id_token_claims",
        mock_claims_fn,
    )
"""

from __future__ import annotations

import logging
import time as _time

from typing import Any

logger = logging.getLogger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_CERTS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUER = "https://accounts.google.com"

# In-memory JWKS cache: url → (jwks_object, expiry_monotonic_timestamp)
_jwks_cache: dict[str, tuple[object, float]] = {}
_JWKS_TTL = 3600.0  # 1 hour


class OAuthError(Exception):
    """Raised when Google OAuth exchange or token validation fails.

    Never expose the message to the user — always return a generic error
    response.  The message is for internal logging only.
    """


async def fetch_google_id_token_claims(
    *,
    code: str,
    code_verifier: str,
    redirect_uri: str,
    client_id: str,
    client_secret: str,
    expected_nonce: str,
) -> dict[str, Any]:
    """Exchange a Google authorization code and return validated ID token claims.

    Steps performed:
    1. Fetch Google's JWKS (public key material for JWT verification).
    2. Exchange the authorization code for tokens (access_token + id_token).
    3. Decode and validate the JWT: signature, iss, aud, nonce, exp,
       email_verified=True.

    Returns a dict containing at minimum: ``sub``, ``email``,
    ``email_verified``, ``nonce``.

    Raises:
        OAuthError: On any network failure, exchange error, or OIDC
                    validation failure.  Callers must treat every
                    ``OAuthError`` as a generic authentication failure
                    and must not surface the message to the browser.
    """
    try:
        import httpx2

        from authlib.integrations.httpx_client import (  # type: ignore[import-untyped]
            AsyncOAuth2Client,
        )
        from authlib.jose import JsonWebKey  # type: ignore[import-untyped]
        from authlib.jose import jwt as authlib_jwt

        async def _fetch_jwks(*, force_refresh: bool = False) -> object:
            now_ts = _time.monotonic()
            cached = _jwks_cache.get(GOOGLE_CERTS_URL)
            if force_refresh or cached is None or now_ts >= cached[1]:
                async with httpx2.AsyncClient(timeout=10.0) as http_client:
                    certs_resp = await http_client.get(GOOGLE_CERTS_URL)
                    certs_resp.raise_for_status()
                    new_jwks = JsonWebKey.import_key_set(certs_resp.json())
                _jwks_cache[GOOGLE_CERTS_URL] = (new_jwks, now_ts + _JWKS_TTL)
            return _jwks_cache[GOOGLE_CERTS_URL][0]

        _claims_options = {
            "exp": {"essential": True},
            "sub": {"essential": True},
            "iat": {"essential": True},
        }

        # Step 1: Fetch Google's current JWKS (with 1-hour in-memory cache).
        was_cached = (
            GOOGLE_CERTS_URL in _jwks_cache
            and _time.monotonic() < _jwks_cache[GOOGLE_CERTS_URL][1]
        )
        jwks = await _fetch_jwks()

        # Step 2: Exchange the authorization code for tokens.
        async with AsyncOAuth2Client(
            client_id=client_id,
            client_secret=client_secret,
        ) as oauth_client:
            token = await oauth_client.fetch_token(
                GOOGLE_TOKEN_URL,
                code=code,
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
            )

        id_token_str = token.get("id_token")
        if not id_token_str:
            raise OAuthError("Token response missing id_token")

        # Step 3: Decode and validate the JWT — RS256 only.
        # One bounded retry on unknown-kid to handle key-set rotation.
        try:
            claims = authlib_jwt.decode(
                id_token_str, jwks, claims_options=_claims_options
            )
        except (
            ValueError
        ):  # Key-miss or other decode ValueError — bounded to one retry.
            if was_cached:
                jwks = await _fetch_jwks(force_refresh=True)
                claims = authlib_jwt.decode(
                    id_token_str, jwks, claims_options=_claims_options
                )
            else:
                raise
        # Reject tokens signed with anything other than RS256 before validating.
        if claims.header.get("alg") != "RS256":
            raise OAuthError(f"Unexpected JWT algorithm: {claims.header.get('alg')!r}")
        claims.validate(leeway=10)  # tolerate 10-second clock skew

    except OAuthError:
        raise
    except Exception as exc:
        raise OAuthError(f"Google OIDC exchange failed: {exc}") from exc

    # Validate mandatory OIDC claims.
    iss = claims.get("iss", "")
    if iss not in (GOOGLE_ISSUER, "accounts.google.com"):
        raise OAuthError(f"Unexpected iss: {iss!r}")
    if claims.get("aud") != client_id:
        raise OAuthError("aud does not match client_id")
    if claims.get("nonce") != expected_nonce:
        raise OAuthError("nonce mismatch")

    # Strict sub/email/email_verified validation.
    sub_val = claims.get("sub")
    if not isinstance(sub_val, str) or not sub_val or len(sub_val) > 255:
        raise OAuthError("sub claim invalid or missing")
    email_val = claims.get("email")
    if not isinstance(email_val, str) or not email_val:
        raise OAuthError("email claim invalid or missing")
    if claims.get("email_verified") is not True:
        raise OAuthError("email_verified is not True")

    return dict(claims)
