import asyncio
import logging
import secrets
import shutil

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.api.constants import NO_STORE
from snore.api.deps import ImmediateDbDep, ResetLockDep, service_dep
from snore.api.errors import db_busy_maps_to_409
from snore.api.guards import RequireAdmin
from snore.auth.invite_tokens import hash_invite_token
from snore.constants import DEFAULT_VACUUM_PENDING_MARKER
from snore.database import models
from snore.database.models import Base
from snore.database.target import DatabaseTarget
from snore.services.database_service import (
    DatabaseService,
    _vacuum_background,
    file_size_mb,
)
from snore.services.profile_service import quarantine_profile_raw_dir
from snore.services.schemas import DatabaseStats, ResetResult, VacuumResult
from snore.services.waveform_service import clear_waveform_array_cache

logger = logging.getLogger(__name__)

router = APIRouter()

# Kept as an empty placeholder — reset has moved to the main router.
local_only_router = APIRouter()

DatabaseServiceDep = Annotated[DatabaseService, Depends(service_dep(DatabaseService))]

# Tables preserved by the data-only reset (include_accounts=False).
# Sleep data tables are deleted; account/auth/profile containers survive.
_DATA_RESET_SKIP = frozenset(
    {
        "users",
        "auth_identities",
        "invites",
        "profiles",
        "oauth_attempts",
        "totp_recovery_codes",
    }
)


class DatabaseStatsPublic(DatabaseStats):
    # Exclude server filesystem path from API responses.
    db_path: str = Field(default="", exclude=True)


class ResetRequest(BaseModel):
    include_accounts: bool = False


def _get_target() -> DatabaseTarget:
    """Resolve the current database target from the environment/session."""
    return DatabaseTarget.from_env_and_flags(db_flag=None, warn_ignored=False)


def _raw_root() -> Path:
    from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

    return DEFAULT_RAW_BACKUP_DIR


async def _mint_admin_invite(db: AsyncSession, email: str, base_url: str) -> str:
    """Insert a new admin invite row and return its redemption URL.

    Called after a full factory reset (include_accounts=True) when all rows
    have been deleted.  At that point there are no users or pending invites,
    so the duplicate-guard queries are elided.
    """
    raw = secrets.token_urlsafe(32)
    token_hash = hash_invite_token(raw)
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=7)

    await db.execute(
        insert(models.Invite).values(
            email=email,
            token_hash=token_hash,
            role="admin",
            created_by=None,
            expires_at=expires_at,
            created_at=now,
        )
    )

    url_base = base_url.rstrip("/") if base_url else ""
    return f"{url_base}/invite#{raw}" if url_base else f"/invite#{raw}"


@router.get("/stats", response_model=DatabaseStatsPublic)
async def get_stats(
    service: DatabaseServiceDep,
    target: Annotated[DatabaseTarget, Depends(_get_target)],
    _actor: RequireAdmin,
) -> DatabaseStats:
    db_path = target.sqlite_path if target.dialect == "sqlite" else ""
    return await service.get_stats(db_path)


@router.post("/vacuum", response_model=VacuumResult)
def vacuum_db(
    service: DatabaseServiceDep,
    target: Annotated[DatabaseTarget, Depends(_get_target)],
    _actor: RequireAdmin,
) -> VacuumResult:
    """Vacuum the SQLite database to reclaim space after deletions.

    Requires a SQLite file target; raises 422 for non-SQLite databases.
    """
    if (
        target.dialect != "sqlite"
        or not target.location
        or target.location == ":memory:"
    ):
        raise HTTPException(
            status_code=422,
            detail="VACUUM is only available for SQLite file databases.",
        )
    return service.vacuum_sqlite(target.sqlite_path)


@router.post("/reset", response_model=ResetResult)
async def reset_db(
    target: Annotated[DatabaseTarget, Depends(_get_target)],
    actor: RequireAdmin,
    # _lock must precede db — see require_reset_lock docstring.
    _lock: ResetLockDep,
    db: ImmediateDbDep,
    background_tasks: BackgroundTasks,
    body: Annotated[ResetRequest | None, Body()] = None,
) -> JSONResponse:
    """Delete database rows and vacuum.

    Two modes selected via the request body:

    ``include_accounts=false`` (default):
        Delete all sleep data (devices, sessions, waveforms, events, statistics,
        settings, analysis results, breaths, detected patterns, days, Apple Health
        samples and summaries, mask log, validation runs, analysis and import job
        records) and purge every ``raw/<profile_id>/`` backup directory.
        User accounts, auth identities, 2FA recovery codes, invites, and profile
        containers are preserved, so users keep their accounts and can re-import
        whatever is still on their SD cards.

    ``include_accounts=true``:
        Full factory reset (every row in every table) plus raw-dir purge and
        vacuum.  A fresh bootstrap admin invite is created for
        the calling admin's email and returned in ``bootstrap_invite_url``.  The
        caller's session is immediately dead; they must redeem that URL to regain
        access.

    The response carries ``Cache-Control: no-store`` so the one-time
    ``bootstrap_invite_url`` is never cached by a proxy or browser.

    VACUUM runs as a post-response background task.  ``size_after_mb`` is null
    and ``vacuum_scheduled`` is true in the response when VACUUM is queued.
    A persistent marker file is written before the commit so that a container
    restart between commit and VACUUM causes startup to reschedule the VACUUM.

    Crash-safe cleanup (quarantine-before-commit pattern): raw backup dirs are
    renamed into ``.quarantine/`` BEFORE the commit.  A crash between quarantine
    and commit leaves dirs in ``.quarantine/`` for ``DeletionSaga.recover()``
    case 2 to sweep on next boot; DB rows survive so the state is consistent.
    New imports proceed normally, but the raw archive is lost and cannot be
    rebuilt from the database.

    Note: this handler holds the SQLite write lock for the entire delete+commit,
    so on large databases concurrent import/analysis writers may hit their 5 s
    busy timeout — operators should quiesce imports before a factory reset.
    """
    from snore.api.config import get_config  # noqa: PLC0415

    cfg = get_config()
    req = body or ResetRequest()

    is_sqlite_file = target.dialect == "sqlite" and target.location not in (
        "",
        ":memory:",
    )
    db_path = target.sqlite_path if is_sqlite_file else ""

    size_before = await asyncio.to_thread(file_size_mb, db_path)

    raw_root = _raw_root()
    caller_email = ""

    # Write vacuum marker before commit so a container restart between commit
    # and VACUUM causes startup to reschedule the VACUUM.  Best-effort: a
    # marker-write failure is logged but never aborts the request.
    if is_sqlite_file:
        try:
            DEFAULT_VACUUM_PENDING_MARKER.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_VACUUM_PENDING_MARKER.write_text(db_path)
        except Exception:
            logger.warning(
                "Could not write vacuum pending marker; VACUUM may not resume after a crash"
            )

    # Quarantined dirs are populated inside the async-with block so they are
    # accessible for rmtree after the block.
    quarantined: list[tuple[int, Path]] = []

    async with db_busy_maps_to_409():
        if req.include_accounts:
            # Full factory reset.
            #
            # Fetch caller email BEFORE any deletes — this is a hard precondition.
            # If the calling user's row can't be found, abort rather than wiping
            # the database and leaving the system with no admin and no invite.
            row = (
                await db.execute(
                    select(models.User.canonical_email).where(
                        models.User.id == actor.user_id
                    )
                )
            ).first()
            caller_email = str(row[0]) if row and row[0] else ""

            if not caller_email:
                raise HTTPException(
                    status_code=500,
                    detail="Could not resolve caller email; reset aborted to prevent data loss.",
                )

        # Collect all profile IDs for raw-dir quarantine (Core SQL, avoids ORM
        # identity-map conflicts with the bulk delete that follows).
        profile_ids = list((await db.execute(select(models.Profile.id))).scalars())

        skip = frozenset() if req.include_accounts else _DATA_RESET_SKIP
        tables_cleared = {}
        for table in reversed(Base.metadata.sorted_tables):
            if table.name in skip:
                continue
            cursor = await db.execute(table.delete())
            tables_cleared[table.name] = cursor.rowcount or 0  # type: ignore[attr-defined]
        total = sum(tables_cleared.values())

        if req.include_accounts:
            # After bulk deletes, expunge stale ORM state (e.g. the acting user's
            # row, which was deleted) before inserting the invite so the subsequent
            # commit does not try to flush stale ORM objects.
            db.expunge_all()

            # Insert the bootstrap invite in the same transaction as the deletion.
            base_url = cfg.public_base_url or ""
            bootstrap_invite_url: str | None = await _mint_admin_invite(
                db, caller_email, base_url
            )
        else:
            bootstrap_invite_url = None

        # Quarantine raw dirs BEFORE committing (crash-safe).  A crash between
        # quarantine and commit leaves dirs in .quarantine/ for startup recovery;
        # DB rows survive so the state is consistent.
        for pid in profile_ids:
            dst = quarantine_profile_raw_dir(pid, raw_root)
            if dst is not None:
                quarantined.append((pid, dst))

        try:
            await db.commit()
        except Exception:
            # Commit failed — restore quarantined dirs so raw backups are not
            # orphaned alongside the uncommitted DB rows.
            for pid, dst in quarantined:
                src = raw_root / str(pid)
                try:
                    dst.rename(src)
                except Exception:
                    logger.warning(
                        "Could not restore quarantined raw dir for profile %d after "
                        "commit failure; dir remains in .quarantine/ for startup recovery",
                        pid,
                    )
            raise

        if req.include_accounts:
            logger.warning(
                "Full factory reset committed for caller %s; bootstrap invite URL "
                "is only in the response body and will not be recoverable afterward.",
                caller_email,
            )

    # The user asked for this data to be gone; see clear_waveform_array_cache.
    clear_waveform_array_cache()

    # Best-effort rmtree of quarantined dirs after a successful commit.
    # CancelledError may skip this; startup recovery (DeletionSaga case 2)
    # purges any .quarantine/ leftovers on next boot.
    def _rmtree_quarantined() -> None:
        for pid, dst in quarantined:
            try:
                shutil.rmtree(dst, ignore_errors=True)
                logger.info("Purged quarantine for profile %d (reset)", pid)
            except Exception:
                logger.warning(
                    "Post-commit rmtree for profile %d failed; startup recovery will purge",
                    pid,
                )

    try:
        await asyncio.to_thread(_rmtree_quarantined)
    except Exception:
        # The reset is committed; a cleanup failure must not withhold the
        # response (for factory resets the one-time bootstrap invite URL
        # exists nowhere else). Leftover quarantine dirs are logged for operators.
        logger.exception(
            "Quarantine cleanup failed after reset commit; startup recovery will purge on next boot"
        )

    # Schedule VACUUM as a post-response background task.  FastAPI runs sync
    # background tasks in a thread pool so the event loop is never blocked.
    if is_sqlite_file:
        background_tasks.add_task(_vacuum_background, db_path)

    return JSONResponse(
        content=ResetResult(
            status="success",
            tables_cleared=tables_cleared,
            total_rows_deleted=total,
            size_before_mb=size_before,
            size_after_mb=None,
            vacuum_scheduled=is_sqlite_file,
            bootstrap_invite_url=bootstrap_invite_url,
        ).model_dump(),
        headers=NO_STORE,
    )
