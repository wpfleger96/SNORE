"""Self-service account management for the authenticated user.

Routes
------
GET    /api/v1/auth/me
PATCH  /api/v1/auth/me/display-name
POST   /api/v1/auth/me/password
GET    /api/v1/auth/me/preferences
PATCH  /api/v1/auth/me/preferences
DELETE /api/v1/auth/me/identities/google
POST   /api/v1/auth/me/delete-data

All responses carry ``Cache-Control: no-store`` to prevent credential
caching by proxies or browsers.

Security controls
-----------------
- Password change requires the existing password when one is set; rejects
  ``current_password`` when no password is set (Google-only accounts).
- Password change applies the same lockout tracking as the login endpoint:
  repeated wrong ``current_password`` attempts are throttled per
  (canonical_email, client_ip).
- Password change bumps ``session_version`` and re-issues the caller's
  session cookie so the change takes effect immediately without logout.
- Google unlink deletes auth_identities rows, bumps ``session_version``,
  and clears the session cookie so the user must re-authenticate with their
  password.  Blocked with 409 when no password is set (lockout prevention).
- Preference updates reject unknown fields (extra="forbid") to prevent
  silent key accumulation in the stored JSON blob.
- All mutating endpoints require ``RequireWritable``, which blocks demo
  role actors at the guard layer.  delete-data is covered by
  ``RequireWritable`` — demo actors are rejected at the guard layer (403)
  without needing a separate in-handler check.
"""

from __future__ import annotations

import logging

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StringConstraints
from sqlalchemy import delete, exists, func, select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession

from snore.api import reset_jobs
from snore.api.client_ip import get_client_ip
from snore.api.constants import NO_STORE
from snore.api.deps import get_db
from snore.api.guards import RequireAuth, RequireWritable
from snore.api.schemas import (
    DISPLAY_NAME_MAX_LEN,
    MessageResponse,
    ResetJobEnqueued,
)
from snore.auth.lockout import get_lockout_store
from snore.auth.passwords import (
    hash_password_async,
    validate_password_bytes,
    verify_password_async,
)
from snore.auth.session_cookie import clear_session_cookie, set_session_cookie
from snore.database import models

logger = logging.getLogger(__name__)

router = APIRouter()

_PASSWORD_MAX_CHARS = (
    4096  # conservative char cap; byte validator refines to 1024 bytes
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class MeResponse(BaseModel):
    id: int
    email: str
    display_name: str | None
    role: str
    has_password: bool
    google_linked: bool
    totp_enabled: bool
    totp_enrollment_required: bool
    recovery_codes_remaining: int | None


class DisplayNameRequest(BaseModel):
    display_name: (
        Annotated[str, StringConstraints(max_length=DISPLAY_NAME_MAX_LEN)] | None
    )


class PasswordChangeRequest(BaseModel):
    current_password: (
        Annotated[str, StringConstraints(max_length=_PASSWORD_MAX_CHARS)] | None
    ) = None
    new_password: Annotated[str, StringConstraints(max_length=_PASSWORD_MAX_CHARS)]


class UserPreferences(BaseModel):
    landing_page: Literal["dashboard", "sessions", "stats"] = "dashboard"
    date_format: Literal["iso", "locale", "short"] = "iso"


class UserPreferencesUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    landing_page: Literal["dashboard", "sessions", "stats"] | None = None
    date_format: Literal["iso", "locale", "short"] | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _get_user_or_401(db: AsyncSession, user_id: int) -> models.User:
    """Fetch the user row by PK; raise 401 if not found."""
    user = await db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("", response_model=MeResponse)
async def get_me(
    actor: RequireAuth,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JSONResponse:
    """Return the authenticated user's account information."""
    user = await _get_user_or_401(db, actor.user_id)

    google_linked = bool(
        (
            await db.execute(
                select(
                    exists().where(
                        models.AuthIdentity.user_id == actor.user_id,
                        models.AuthIdentity.provider == "google",
                    )
                )
            )
        ).scalar()
    )

    totp_enabled = user.totp_enabled_at is not None
    recovery_codes_remaining: int | None = None
    if totp_enabled:
        recovery_codes_remaining = (
            await db.scalar(
                select(func.count()).where(
                    models.TotpRecoveryCode.user_id == actor.user_id,
                    models.TotpRecoveryCode.used_at.is_(None),
                )
            )
        ) or 0

    return JSONResponse(
        content=MeResponse(
            id=user.id,
            email=user.canonical_email,
            display_name=user.display_name,
            role=user.role,
            has_password=user.password_hash is not None,
            google_linked=google_linked,
            totp_enabled=totp_enabled,
            totp_enrollment_required=actor.enrollment_required,
            recovery_codes_remaining=recovery_codes_remaining,
        ).model_dump(),
        headers=NO_STORE,
    )


@router.patch("/display-name", response_model=MessageResponse)
async def update_display_name(
    actor: RequireWritable,
    body: DisplayNameRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JSONResponse:
    """Update the authenticated user's display name."""
    user = await _get_user_or_401(db, actor.user_id)

    # Strip; empty-after-strip → None.
    name = body.display_name
    if name is not None:
        name = name.strip() or None
    user.display_name = name

    return JSONResponse(
        content={"message": "Display name updated"},
        headers=NO_STORE,
    )


@router.post("/password", response_model=MessageResponse)
async def change_password(
    request: Request,
    actor: RequireWritable,
    body: PasswordChangeRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JSONResponse:
    """Change the authenticated user's password; bumps session_version and re-issues cookie."""
    from snore.api.config import get_config  # noqa: PLC0415

    cfg = get_config()

    # Validate new password byte length before any DB work or KDF.
    try:
        validate_password_bytes(body.new_password)
    except ValueError:
        raise HTTPException(
            status_code=422, detail="Password must be 1–1024 bytes encoded"
        ) from None

    user = await _get_user_or_401(db, actor.user_id)

    canonical = user.canonical_email
    ip = get_client_ip(request)
    lockout = get_lockout_store()

    # Check lockout before any password verification.
    if lockout.is_locked(canonical, ip):
        raise HTTPException(status_code=401, detail="Authentication failed")

    if user.password_hash is not None:
        # Existing password — current_password required.
        if body.current_password is None:
            raise HTTPException(status_code=422, detail="Current password required")
        ok, _ = await verify_password_async(user.password_hash, body.current_password)
        if not ok:
            lockout.record_failure(canonical, ip)
            raise HTTPException(status_code=401, detail="Authentication failed")
    else:
        # Google-only account — current_password must be absent/None.
        if body.current_password is not None:
            raise HTTPException(
                status_code=422,
                detail="No current password is set on this account",
            )

    lockout.record_success(canonical, ip)

    user.password_hash = await hash_password_async(body.new_password)
    user.session_version += 1

    response = JSONResponse(
        content={"message": "Password updated"},
        headers=NO_STORE,
    )
    if cfg.is_multiuser:
        set_session_cookie(
            response,
            secret=cfg.session_secret,
            user_id=actor.user_id,
            active_profile_id=actor.profile_id,
            session_version=user.session_version,
            secure=cfg.secure_cookie,
        )
    return response


@router.delete("/identities/google", response_model=MessageResponse)
async def unlink_google(
    actor: RequireWritable,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JSONResponse:
    """Delete the user's Google identity and invalidate all sessions.

    Blocked with 409 when the account has no password — removing Google would
    eliminate the user's only sign-in method.  On success, bumps
    ``session_version`` (invalidates all cookies) and clears the caller's
    session cookie so they must re-authenticate with their password.
    """
    from snore.api.config import get_config  # noqa: PLC0415

    cfg = get_config()

    user = await _get_user_or_401(db, actor.user_id)

    if user.password_hash is None:
        raise HTTPException(
            status_code=409,
            detail="Set a password before unlinking Google",
        )

    # Single DELETE — rowcount check replaces the prior SELECT-EXISTS + DELETE
    # pair, removing the TOCTOU window and one extra round-trip.
    result = await db.execute(
        delete(models.AuthIdentity).where(
            models.AuthIdentity.user_id == actor.user_id,
            models.AuthIdentity.provider == "google",
        )
    )
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise HTTPException(status_code=404, detail="No Google identity linked")

    # Server-side increment avoids a client-side read-modify-write (lost-update
    # risk under concurrent writes on non-SQLite backends).  Set
    # google_link_disabled so the email auto-link path in resolve_login cannot
    # silently re-establish this identity on the next "Sign in with Google".
    await db.execute(
        sa_update(models.User)
        .where(models.User.id == actor.user_id)
        .values(
            session_version=models.User.session_version + 1,
            google_link_disabled=True,
        )
    )

    logger.info("Unlinked Google identity for user id=%s", actor.user_id)

    # Accepted edge: the response is built before the session commits; a commit
    # failure would leave the cookie cleared but state unchanged — same race
    # window as change_password.
    response = JSONResponse(
        content={"message": "Google account unlinked"},
        headers=NO_STORE,
    )
    if cfg.is_multiuser:
        clear_session_cookie(response, secure=cfg.secure_cookie)
    return response


@router.get("/preferences", response_model=UserPreferences)
async def get_preferences(
    actor: RequireAuth,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JSONResponse:
    """Return the authenticated user's preferences, filling gaps with defaults."""
    user = await _get_user_or_401(db, actor.user_id)

    prefs = UserPreferences(**(user.preferences or {}))
    return JSONResponse(
        content=prefs.model_dump(),
        headers=NO_STORE,
    )


@router.patch("/preferences", response_model=UserPreferences)
async def update_preferences(
    actor: RequireWritable,
    body: UserPreferencesUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> JSONResponse:
    """Merge supplied preferences into stored preferences; return the merged result."""
    user = await _get_user_or_401(db, actor.user_id)

    # Load current state (defaults fill gaps, unknown stored keys stripped).
    current = UserPreferences(**(user.preferences or {}))

    # Apply only explicitly supplied (non-None) fields.
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    merged = current.model_copy(update=updates)
    dumped = merged.model_dump()
    user.preferences = dumped

    return JSONResponse(
        content=dumped,
        headers=NO_STORE,
    )


@router.post("/delete-data", status_code=202, response_model=ResetJobEnqueued)
async def delete_my_data(actor: RequireWritable) -> JSONResponse:
    """Queue a background job that deletes all sleep data owned by the caller.

    The job removes all Device rows for every live profile owned by the caller.
    The DB-level cascade (Device → Session/Day/Waveform/Event/Statistics/Setting/
    AnalysisResult/Breath/DetectedPattern) wipes all dependent sleep data.
    Import job records for this user are also deleted.  Raw backup directories
    under ``raw/<profile_id>/`` are purged via the quarantine-rename pattern.

    Profile rows, the user account, preferences, auth identities, and invites
    are NOT affected.  The caller can re-import data after this operation.

    The work outlasts the proxy's request timeout on a large database, so this
    returns 202 at once with a ``job_id`` to poll with
    ``GET /db/reset-jobs/status``.  Returns 409 while another reset or data
    deletion is queued or running.

    NOTE: Concurrent in-flight imports for this user's devices will fail if
    their device rows are deleted mid-import; this is accepted behavior.

    Demo accounts are blocked by ``RequireWritable`` (403) before reaching
    this handler — fixture data is never at risk.
    """
    from snore.database.target import DatabaseTarget  # noqa: PLC0415

    target = DatabaseTarget.from_env_and_flags(db_flag=None, warn_ignored=False)
    return reset_jobs.accepted_or_409(
        reset_jobs.enqueue(
            kind=reset_jobs.ResetJobKind.DELETE_DATA,
            user_id=actor.user_id,
            db_path=reset_jobs.sqlite_file_path(target),
        )
    )
