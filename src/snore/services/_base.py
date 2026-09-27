"""Shared base and ownership helpers for profile-scoped services."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, Select, func, select
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.exceptions import NotFoundError
from snore.utils.db_chunk import iter_id_chunks

__all__ = [
    "ProfileScopedService",
    "get_owned_session_ids",
    "paginate",
    "require_owned_session",
    "session_device_join",
]


def session_device_join[SelectT: Select[*tuple[Any, ...]]](stmt: SelectT) -> SelectT:
    """Join Session → Device: the ownership edge used by profile filtering."""
    return stmt.join(models.Device, models.Session.device_id == models.Device.id)


async def paginate[*Ts](
    db: AsyncSession,
    stmt: Select[*Ts],
    *,
    order_by: ColumnElement[Any],
    limit: int,
    offset: int,
) -> tuple[Result[*Ts], int]:
    """Count rows matching *stmt*, then execute one ordered page of it.

    The total is ``COUNT(*)`` over the filtered statement (before ordering and
    windowing), so it is independent of the page requested.  ``limit <= 0``
    means unlimited; a non-positive ``offset`` is omitted.  Row-shape mapping
    stays with the caller — the executed page ``Result`` is returned as-is.
    """
    # COUNT over stmt's subquery is correct only if the statement's joins are
    # row-preserving (1:1 / non-multiplying); a multiplying join inflates total.
    total = (
        await db.execute(select(func.count()).select_from(stmt.subquery()))
    ).scalar() or 0

    stmt = stmt.order_by(order_by)
    if limit > 0:
        stmt = stmt.limit(limit)
    if offset > 0:
        stmt = stmt.offset(offset)
    return await db.execute(stmt), total


class ProfileScopedService:
    """Base for services whose queries are scoped to a single profile."""

    def __init__(self, db_session: AsyncSession, profile_id: int) -> None:
        self.db_session = db_session
        self.profile_id = profile_id

    def _profile_filter(self) -> ColumnElement[bool]:
        """WHERE predicate: limit rows to this profile via device ownership."""
        return models.Device.profile_id == self.profile_id


async def get_owned_session_ids(
    db: AsyncSession, profile_id: int, session_ids: Sequence[int]
) -> set[int]:
    """Return the subset of session_ids that belong to this profile.

    Used by routes to validate ownership before mutation: any ID absent from
    the returned set is either missing or owned by a different profile.
    """
    if not session_ids:
        return set()
    owned: set[int] = set()
    for chunk in iter_id_chunks(session_ids):
        rows = (
            (
                await db.execute(
                    session_device_join(select(models.Session.id)).where(
                        models.Session.id.in_(chunk),
                        models.Device.profile_id == profile_id,
                    )
                )
            )
            .scalars()
            .all()
        )
        owned.update(rows)
    return owned


async def require_owned_session(
    db: AsyncSession, profile_id: int, session_id: int
) -> None:
    """Raise NotFoundError unless session_id belongs to this profile.

    Foreign ID → 404, not 403, to avoid oracle attacks.
    """
    row = (
        await db.execute(
            session_device_join(select(models.Session.id)).where(
                models.Session.id == session_id,
                models.Device.profile_id == profile_id,
            )
        )
    ).scalar_one_or_none()
    if row is None:
        raise NotFoundError(f"Session {session_id} not found")
