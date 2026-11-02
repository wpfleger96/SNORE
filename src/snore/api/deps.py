from collections.abc import AsyncGenerator, Callable
from datetime import date, datetime, time
from typing import Annotated

from fastapi import Depends, Path, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from snore.auth.actor import ActorContext, AuthMode
from snore.database.session import check_db_staleness, get_session, session_scope


async def get_db() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency that provides a committed/rolled-back AsyncSession."""
    async with session_scope() as session:
        yield session


async def get_db_immediate() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency providing a BEGIN IMMEDIATE session for writes.

    Acquires the SQLite write lock at transaction open (BEGIN IMMEDIATE) so
    contending writers queue on busy_timeout rather than failing instantly on a
    WAL snapshot-upgrade conflict.  Use for endpoints that read before writing
    to prevent SQLITE_BUSY bypassing the timeout that a deferred-BEGIN
    transaction would trigger.
    """
    async with session_scope(immediate=True) as session:
        yield session


# SQLite INTEGER is a signed 64-bit value; larger ids overflow the driver.
_MAX_DB_ID = 2**63 - 1
BoundedPathId = Annotated[int, Path(le=_MAX_DB_ID)]
BoundedQueryId = Annotated[int | None, Query(le=_MAX_DB_ID)]

ImmediateDbDep = Annotated[AsyncSession, Depends(get_db_immediate)]


async def get_raw_session() -> AsyncGenerator[AsyncSession]:
    """FastAPI dependency yielding an AsyncSession WITHOUT an open transaction.

    Use when a handler needs fine-grained transaction control (e.g., splitting
    a read transaction from network I/O from a write transaction).

    Staleness check: ``check_db_staleness`` is called once at dependency
    resolution (when the session is created), not per transaction.  Handlers
    that run multiple explicit transactions with network I/O between them may
    observe a DB-file swap between transactions; the swap will be detected and
    the engine rebuilt at the start of the next request.  Sessions already open
    when a swap occurs keep their connection to the OLD unlinked inode; any
    writes they commit are lost when the inode's last file descriptor closes.
    """
    await check_db_staleness()
    session = get_session()
    try:
        yield session
    finally:
        await session.close()


async def get_actor(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> ActorContext:
    """Return the actor for this request.

    In local mode: AuthMiddleware auto-provisions and sets request.state.actor.
    In multiuser mode: AuthMiddleware validates the session cookie and sets it.

    Raises 401 if no actor is present (unauthenticated in multiuser or
    middleware provisioning failed in local mode).
    """
    actor: ActorContext | None = getattr(request.state, "actor", None)
    if actor is not None:
        return actor

    # Fallback for test environments where middleware may not run (e.g. unit
    # tests that create a TestClient without lifespan and don't set state.actor).
    # In production multiuser mode the middleware always runs; this path is only
    # reached in test setups that bypass middleware.
    from snore.api.config import get_config  # noqa: PLC0415

    cfg = get_config()
    if cfg.is_multiuser:
        from fastapi import HTTPException  # noqa: PLC0415

        raise HTTPException(status_code=401, detail="Authentication required")

    from snore.auth.factory import ActorContextFactory  # noqa: PLC0415

    factory = ActorContextFactory(db)
    return await factory.make_local(mode=AuthMode.LOCAL)


ActorDep = Annotated[ActorContext, Depends(get_actor)]


def service_dep[T](cls: Callable[[AsyncSession, int], T]) -> Callable[..., T]:
    """Return a FastAPI dependency that constructs ``cls(db, profile_id)``."""

    async def _dep(
        db: Annotated[AsyncSession, Depends(get_db)],
        actor: ActorDep,
    ) -> T:
        return cls(db, actor.profile_id)

    return _dep  # type: ignore[return-value]


def service_dep_immediate[T](
    cls: Callable[[AsyncSession, int], T],
) -> Callable[..., T]:
    """Return a FastAPI dependency that constructs ``cls`` with a
    profile-scoped BEGIN IMMEDIATE session.

    Routes with a writable guard must declare that guard before this dependency
    so rejected requests do not acquire SQLite's writer lock.
    """

    async def _dep(
        actor: ActorDep,
        db: ImmediateDbDep,
    ) -> T:
        return cls(db, actor.profile_id)

    return _dep  # type: ignore[return-value]


class PaginationParams:
    def __init__(
        self,
        limit: int = Query(default=20, ge=1, le=1000),
        offset: int = Query(default=0, ge=0),
    ):
        self.limit = limit
        self.offset = offset


class DateRangeParams:
    def __init__(
        self,
        from_date: Annotated[date | None, Query()] = None,
        to_date: Annotated[date | None, Query()] = None,
    ):
        self.from_date = from_date
        self.to_date = to_date

    @property
    def start_datetime(self) -> datetime | None:
        """from_date as an inclusive datetime lower bound (midnight)."""
        return datetime.combine(self.from_date, time.min) if self.from_date else None

    @property
    def end_datetime(self) -> datetime | None:
        """to_date as an inclusive datetime upper bound (end of day)."""
        return datetime.combine(self.to_date, time.max) if self.to_date else None
