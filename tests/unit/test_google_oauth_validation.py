"""Unit tests for google_oauth.fetch_google_id_token_claims OIDC validation.

Uses a locally-generated RSA-2048 key pair — no internet access needed.
Both the JWKS fetch (httpx2) and the OAuth token exchange (AsyncOAuth2Client)
are patched; only the JWT decode/validate logic runs against real authlib code.
"""

from __future__ import annotations

import time

# Pre-import so the class definition runs before any httpx2.AsyncClient patch.
# Without this, patching httpx2.AsyncClient first causes a metaclass conflict
# when authlib tries to define AsyncOAuth2Client(…, httpx2.AsyncClient) during
# the lazy import triggered by mock.patch.
import authlib.integrations.httpx_client  # type: ignore[import-untyped]  # noqa: F401
import pytest

from authlib.jose import RSAKey  # type: ignore[import-untyped]
from authlib.jose import jwt as authlib_jwt  # noqa: F401

from snore.auth import google_oauth
from snore.auth.google_oauth import (
    GOOGLE_ISSUER,
    OAuthError,
    fetch_google_id_token_claims,
)

_CLIENT_ID = "test-client-id.apps.googleusercontent.com"
_NONCE = "testnonce123"

# ---------------------------------------------------------------------------
# Module-scoped RSA key pair — generated once, shared across all tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rsa_private_key():
    """Generate a fresh RSA-2048 private key for the test module."""
    return RSAKey.generate_key(2048, is_private=True)


@pytest.fixture(scope="module")
def jwks_dict(rsa_private_key):
    """Build the JWKS dict (public-key-only) that the mock JWKS endpoint returns."""
    pub_dict = rsa_private_key.as_dict()
    # Keep only public components (remove 'd', 'p', 'q', 'dp', 'dq', 'qi').
    public_only = {
        k: v for k, v in pub_dict.items() if k not in ("d", "p", "q", "dp", "dq", "qi")
    }
    public_only.setdefault("use", "sig")
    public_only.setdefault("alg", "RS256")
    return {"keys": [public_only]}


# ---------------------------------------------------------------------------
# Token factory
# ---------------------------------------------------------------------------


def _make_token(
    rsa_private_key: RSAKey,
    *,
    claims_override: dict | None = None,
    header_override: dict | None = None,
) -> str:
    """Build a signed RS256 JWT with the given claims."""
    now = int(time.time())
    base_claims: dict = {
        "iss": GOOGLE_ISSUER,
        "aud": _CLIENT_ID,
        "sub": "test-sub-123",
        "email": "test@example.com",
        "email_verified": True,
        "nonce": _NONCE,
        "iat": now,
        "exp": now + 3600,
    }
    if claims_override:
        base_claims.update(claims_override)
    # Remove keys explicitly set to None (simulate missing claims).
    claims = {k: v for k, v in base_claims.items() if v is not None}

    header: dict = {"alg": "RS256"}
    if header_override:
        header.update(header_override)

    token_bytes = authlib_jwt.encode(header, claims, rsa_private_key)
    return token_bytes.decode() if isinstance(token_bytes, bytes) else token_bytes


# ---------------------------------------------------------------------------
# Mock context manager helpers
# ---------------------------------------------------------------------------


def _make_http_mock(jwks_dict: dict) -> object:
    """Return an async context manager mock for httpx2.AsyncClient that serves jwks_dict."""
    from unittest.mock import AsyncMock, MagicMock

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value=jwks_dict)

    http_instance = AsyncMock()
    http_instance.get = AsyncMock(return_value=fake_resp)
    http_instance.__aenter__ = AsyncMock(return_value=http_instance)
    http_instance.__aexit__ = AsyncMock(return_value=False)

    mock_client_cls = MagicMock(return_value=http_instance)
    return mock_client_cls


def _make_oauth_mock(id_token: str) -> object:
    """Return an async context manager mock for AsyncOAuth2Client that returns id_token."""
    from unittest.mock import AsyncMock, MagicMock

    oauth_instance = AsyncMock()
    oauth_instance.fetch_token = AsyncMock(return_value={"id_token": id_token})
    oauth_instance.__aenter__ = AsyncMock(return_value=oauth_instance)
    oauth_instance.__aexit__ = AsyncMock(return_value=False)

    mock_oauth_cls = MagicMock(return_value=oauth_instance)
    return mock_oauth_cls


async def _call(
    rsa_private_key: RSAKey,
    jwks_dict: dict,
    *,
    claims_override: dict | None = None,
    header_override: dict | None = None,
    nonce: str = _NONCE,
    client_id: str = _CLIENT_ID,
) -> dict:
    """Build a signed token and call fetch_google_id_token_claims with mocked HTTP."""
    import unittest.mock as mock

    # Clear the JWKS cache so each test fetches fresh (avoids cache pollution).
    google_oauth._jwks_cache.clear()

    id_token = _make_token(
        rsa_private_key,
        claims_override=claims_override,
        header_override=header_override,
    )

    http_mock = _make_http_mock(jwks_dict)
    oauth_mock = _make_oauth_mock(id_token)

    with (
        mock.patch("httpx2.AsyncClient", http_mock),
        mock.patch(
            "authlib.integrations.httpx_client.AsyncOAuth2Client",
            oauth_mock,
        ),
    ):
        return await fetch_google_id_token_claims(
            code="test-code",
            code_verifier="test-verifier",
            redirect_uri="http://localhost/cb",
            client_id=client_id,
            client_secret="test-secret",
            expected_nonce=nonce,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidToken:
    @pytest.mark.asyncio
    async def test_valid_token_returns_claims(self, rsa_private_key, jwks_dict):
        """A well-formed RS256 token with all required claims succeeds."""
        claims = await _call(rsa_private_key, jwks_dict)
        assert claims["sub"] == "test-sub-123"
        assert claims["email"] == "test@example.com"
        assert claims["email_verified"] is True


class TestMissingClaims:
    @pytest.mark.asyncio
    async def test_missing_exp_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token without exp claim raises OAuthError (essential claim)."""
        with pytest.raises(OAuthError):
            await _call(rsa_private_key, jwks_dict, claims_override={"exp": None})

    @pytest.mark.asyncio
    async def test_missing_sub_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token without sub claim raises OAuthError (essential claim)."""
        with pytest.raises(OAuthError):
            await _call(rsa_private_key, jwks_dict, claims_override={"sub": None})

    @pytest.mark.asyncio
    async def test_missing_iat_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token without iat claim raises OAuthError (essential claim)."""
        with pytest.raises(OAuthError):
            await _call(rsa_private_key, jwks_dict, claims_override={"iat": None})


class TestEmailVerified:
    @pytest.mark.asyncio
    async def test_email_verified_false_raises_oauth_error(
        self, rsa_private_key, jwks_dict
    ):
        """email_verified=False raises OAuthError."""
        with pytest.raises(OAuthError, match="email_verified"):
            await _call(
                rsa_private_key, jwks_dict, claims_override={"email_verified": False}
            )

    @pytest.mark.asyncio
    async def test_email_verified_string_raises_oauth_error(
        self, rsa_private_key, jwks_dict
    ):
        """email_verified as a truthy string is not True — must raise OAuthError."""
        with pytest.raises(OAuthError, match="email_verified"):
            await _call(
                rsa_private_key, jwks_dict, claims_override={"email_verified": "true"}
            )


class TestExpiredToken:
    @pytest.mark.asyncio
    async def test_expired_token_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token with exp in the past raises OAuthError."""
        past = int(time.time()) - 7200
        with pytest.raises(OAuthError):
            await _call(
                rsa_private_key,
                jwks_dict,
                claims_override={"exp": past, "iat": past - 10},
            )


class TestIssuerAudience:
    @pytest.mark.asyncio
    async def test_wrong_issuer_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token with unexpected iss raises OAuthError."""
        with pytest.raises(OAuthError, match="iss"):
            await _call(
                rsa_private_key,
                jwks_dict,
                claims_override={"iss": "https://evil.example.com"},
            )

    @pytest.mark.asyncio
    async def test_wrong_audience_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token with mismatched aud raises OAuthError."""
        with pytest.raises(OAuthError, match="aud"):
            await _call(
                rsa_private_key,
                jwks_dict,
                client_id="real-client-id.apps.googleusercontent.com",
                # Token will have _CLIENT_ID as aud but call uses a different client_id.
            )

    @pytest.mark.asyncio
    async def test_wrong_nonce_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token nonce mismatch raises OAuthError."""
        with pytest.raises(OAuthError, match="nonce"):
            await _call(
                rsa_private_key,
                jwks_dict,
                nonce="different-nonce",  # expected nonce differs from token nonce
            )


class TestAlgorithmRestriction:
    @pytest.mark.asyncio
    async def test_hs256_algorithm_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """A token claiming alg=HS256 in its header must be rejected.

        The JWKS only contains RSA keys, so authlib itself will fail to verify
        an HMAC-signed token. Even if authlib somehow decoded it, our explicit
        alg check would catch it.
        """
        # We can't actually sign with HS256 using our RSA key via authlib, so
        # we test that a manually crafted token with alg=HS256 in the header
        # raises OAuthError (either via authlib signature verification failure
        # or our explicit algorithm check).
        import base64
        import json
        import unittest.mock as mock

        google_oauth._jwks_cache.clear()

        # Build a fake HS256-style JWT (signature won't verify, but that's
        # fine — we just want to test the algorithm rejection path).
        header = (
            base64.urlsafe_b64encode(
                json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {"sub": "x", "iss": GOOGLE_ISSUER, "aud": _CLIENT_ID}
                ).encode()
            )
            .rstrip(b"=")
            .decode()
        )
        fake_token = f"{header}.{payload}.fakesignature"

        http_mock = _make_http_mock(jwks_dict)
        oauth_mock = _make_oauth_mock(fake_token)

        with (
            mock.patch("httpx2.AsyncClient", http_mock),
            mock.patch(
                "authlib.integrations.httpx_client.AsyncOAuth2Client", oauth_mock
            ),
            pytest.raises(OAuthError),
        ):
            await fetch_google_id_token_claims(
                code="test-code",
                code_verifier="test-verifier",
                redirect_uri="http://localhost/cb",
                client_id=_CLIENT_ID,
                client_secret="test-secret",
                expected_nonce=_NONCE,
            )


class TestPS256AlgorithmRejection:
    @pytest.mark.asyncio
    async def test_ps256_algorithm_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """A PS256-signed token using the same RSA key must be rejected by policy.

        PS256 is cryptographically valid with an RSA key, so this test
        exercises the explicit ``alg == RS256`` check — not just signature
        verification.  If the alg check is removed, a PS256 token signed
        with the correct key would otherwise pass.
        """
        import unittest.mock as mock

        google_oauth._jwks_cache.clear()

        # Build a JWKS that doesn't restrict alg so Authlib will attempt PS256.
        pub_dict = rsa_private_key.as_dict()
        public_only = {
            k: v
            for k, v in pub_dict.items()
            if k not in ("d", "p", "q", "dp", "dq", "qi")
        }
        public_only.pop("alg", None)  # remove RS256 restriction so PS256 is tried
        public_only.setdefault("use", "sig")
        unrestricted_jwks = {"keys": [public_only]}

        # Sign a valid token with PS256 using the same RSA private key.
        ps256_token = _make_token(rsa_private_key, header_override={"alg": "PS256"})

        http_mock = _make_http_mock(unrestricted_jwks)
        oauth_mock = _make_oauth_mock(ps256_token)

        with (
            mock.patch("httpx2.AsyncClient", http_mock),
            mock.patch(
                "authlib.integrations.httpx_client.AsyncOAuth2Client", oauth_mock
            ),
            pytest.raises(OAuthError),
        ):
            await fetch_google_id_token_claims(
                code="test-code",
                code_verifier="test-verifier",
                redirect_uri="http://localhost/cb",
                client_id=_CLIENT_ID,
                client_secret="test-secret",
                expected_nonce=_NONCE,
            )


class TestSubAndEmailValidation:
    @pytest.mark.asyncio
    async def test_empty_sub_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Empty string sub raises OAuthError."""
        with pytest.raises(OAuthError, match="sub"):
            await _call(rsa_private_key, jwks_dict, claims_override={"sub": ""})

    @pytest.mark.asyncio
    async def test_empty_email_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Empty string email raises OAuthError."""
        with pytest.raises(OAuthError, match="email"):
            await _call(rsa_private_key, jwks_dict, claims_override={"email": ""})

    @pytest.mark.asyncio
    async def test_missing_email_raises_oauth_error(self, rsa_private_key, jwks_dict):
        """Token without email claim raises OAuthError."""
        with pytest.raises(OAuthError, match="email"):
            await _call(rsa_private_key, jwks_dict, claims_override={"email": None})


class TestJwksRetry:
    """JWKS retry behaviour: retry only on unknown-kid (ValueError), not on bad signatures."""

    @pytest.mark.asyncio
    async def test_cache_hit_skips_refetch_on_valid_token(
        self, rsa_private_key, jwks_dict
    ):
        """A cached JWKS that already contains the right key is never re-fetched."""
        import time as _time
        import unittest.mock as mock

        from authlib.jose import JsonWebKey

        # Pre-populate cache with the correct JWKS so was_cached=True and no fetch needed.
        jwks_obj = JsonWebKey.import_key_set(jwks_dict)
        google_oauth._jwks_cache[google_oauth.GOOGLE_CERTS_URL] = (
            jwks_obj,
            _time.monotonic() + 3600,
        )

        id_token = _make_token(rsa_private_key)
        http_mock = _make_http_mock(jwks_dict)
        oauth_mock = _make_oauth_mock(id_token)

        with (
            mock.patch("httpx2.AsyncClient", http_mock),
            mock.patch(
                "authlib.integrations.httpx_client.AsyncOAuth2Client", oauth_mock
            ),
        ):
            claims = await fetch_google_id_token_claims(
                code="test-code",
                code_verifier="test-verifier",
                redirect_uri="http://localhost/cb",
                client_id=_CLIENT_ID,
                client_secret="test-secret",
                expected_nonce=_NONCE,
            )

        assert claims["sub"] == "test-sub-123"
        # httpx2.AsyncClient must not have been called (JWKS came from cache).
        http_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_kid_triggers_one_refetch_then_succeeds(
        self, rsa_private_key, jwks_dict
    ):
        """Unknown kid → fetch once → success; httpx2 called exactly once."""
        import time as _time
        import unittest.mock as mock

        from authlib.jose import JsonWebKey

        # Build the JWKS the token will use (same rsa_private_key but with explicit kid).
        pub_dict = rsa_private_key.as_dict()
        public_only = {
            k: v
            for k, v in pub_dict.items()
            if k not in ("d", "p", "q", "dp", "dq", "qi")
        }
        public_only["kid"] = "current-kid"
        public_only.setdefault("use", "sig")
        public_only.setdefault("alg", "RS256")
        current_jwks_dict = {"keys": [public_only]}

        # Build a stale JWKS with a DIFFERENT kid so the cached decode raises ValueError.
        stale_key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        stale_pub = stale_key.as_dict()
        stale_pub = {
            k: v
            for k, v in stale_pub.items()
            if k not in ("d", "p", "q", "dp", "dq", "qi")
        }
        stale_pub["kid"] = "stale-kid"
        stale_pub.setdefault("use", "sig")
        stale_pub.setdefault("alg", "RS256")
        stale_jwks_obj = JsonWebKey.import_key_set({"keys": [stale_pub]})

        google_oauth._jwks_cache[google_oauth.GOOGLE_CERTS_URL] = (
            stale_jwks_obj,
            _time.monotonic() + 3600,
        )

        # Token signed with rsa_private_key and kid="current-kid" (not in stale JWKS).
        id_token = _make_token(rsa_private_key, header_override={"kid": "current-kid"})
        # httpx2 mock returns the CORRECT JWKS on the forced refresh.
        http_mock = _make_http_mock(current_jwks_dict)
        oauth_mock = _make_oauth_mock(id_token)

        with (
            mock.patch("httpx2.AsyncClient", http_mock),
            mock.patch(
                "authlib.integrations.httpx_client.AsyncOAuth2Client", oauth_mock
            ),
        ):
            claims = await fetch_google_id_token_claims(
                code="test-code",
                code_verifier="test-verifier",
                redirect_uri="http://localhost/cb",
                client_id=_CLIENT_ID,
                client_secret="test-secret",
                expected_nonce=_NONCE,
            )

        assert claims["sub"] == "test-sub-123"
        # httpx2.AsyncClient must have been called exactly once (for the forced refresh).
        assert http_mock.call_count == 1

    @pytest.mark.asyncio
    async def test_bad_signature_does_not_trigger_refetch(
        self, rsa_private_key, jwks_dict
    ):
        """Bad signature (BadSignatureError, not ValueError) must not retry."""
        import time as _time
        import unittest.mock as mock

        from authlib.jose import JsonWebKey, RSAKey

        # Generate a different signing key — same kid, so the key IS found in the
        # JWKS, but signature verification fails (BadSignatureError, not ValueError).
        wrong_signing_key = RSAKey.generate_key(2048, is_private=True)
        # Token signed with wrong_signing_key.
        id_token = _make_token(wrong_signing_key)

        # Cache the correct JWKS (right kid, right public key).
        jwks_obj = JsonWebKey.import_key_set(jwks_dict)
        google_oauth._jwks_cache[google_oauth.GOOGLE_CERTS_URL] = (
            jwks_obj,
            _time.monotonic() + 3600,
        )

        http_mock = _make_http_mock(jwks_dict)
        oauth_mock = _make_oauth_mock(id_token)

        with (
            mock.patch("httpx2.AsyncClient", http_mock),
            mock.patch(
                "authlib.integrations.httpx_client.AsyncOAuth2Client", oauth_mock
            ),
            pytest.raises(OAuthError),
        ):
            await fetch_google_id_token_claims(
                code="test-code",
                code_verifier="test-verifier",
                redirect_uri="http://localhost/cb",
                client_id=_CLIENT_ID,
                client_secret="test-secret",
                expected_nonce=_NONCE,
            )

        # httpx2.AsyncClient must NOT have been called (bad signature ≠ unknown kid).
        http_mock.assert_not_called()
