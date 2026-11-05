"""Store ``analysis_results`` windows as device wall-clock times.

``analysis_results.timestamp_start``/``timestamp_end`` are documented as device
wall-clock, but analysis wrote them as ``datetime.fromtimestamp(offset, tz=UTC)``
with ``offset`` in seconds from session start, so every stored value is
``1970-01-01 00:00:00 + offset``.  The offset is therefore recoverable, and the
upgrade rewrites each such row as ``sessions.start_time + offset``.  No
re-analysis is needed.

Only rows whose ``timestamp_start`` falls before 1971 are touched, so rows
already written as wall-clock are left alone.  The arithmetic runs in Python
rather than SQLite ``datetime()``, which drops microseconds; the lightweight
tables below let SQLAlchemy read and write the same storage format the ORM
uses.  A fresh install has no rows, so this works on any dialect.

Downgrade turns every wall-clock row back into ``1970-01-01 + offset``.

Revision ID: 020_analysis_result_wallclock
Revises: 019_oscar_event_parity
"""

from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa

from alembic import op

revision: str = "020_analysis_result_wallclock"
down_revision: str | Sequence[str] | None = "019_oscar_event_parity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EPOCH = datetime(1970, 1, 1)
_CUTOFF = datetime(1971, 1, 1)

_analysis_results = sa.table(
    "analysis_results",
    sa.column("id", sa.Integer),
    sa.column("session_id", sa.Integer),
    sa.column("timestamp_start", sa.DateTime),
    sa.column("timestamp_end", sa.DateTime),
)
_sessions = sa.table(
    "sessions",
    sa.column("id", sa.Integer),
    sa.column("start_time", sa.DateTime),
)


def _shift(where: sa.ColumnElement[bool], direction: int) -> None:
    """Move matching windows by ``direction * (sessions.start_time - 1970-01-01)``.

    ``+1`` turns epoch-based offsets into wall-clock times, ``-1`` reverses it.
    """
    bind = op.get_bind()
    rows = bind.execute(
        sa.select(
            _analysis_results.c.id,
            _analysis_results.c.timestamp_start,
            _analysis_results.c.timestamp_end,
            _sessions.c.start_time,
        )
        .join(_sessions, _sessions.c.id == _analysis_results.c.session_id)
        .where(where)
    ).all()
    if not rows:
        return

    updates = []
    for row in rows:
        delta = (row.start_time - _EPOCH) * direction
        updates.append(
            {
                "row_id": row.id,
                "new_start": row.timestamp_start + delta,
                "new_end": row.timestamp_end + delta,
            }
        )
    bind.execute(
        _analysis_results.update()
        .where(_analysis_results.c.id == sa.bindparam("row_id"))
        .values(
            timestamp_start=sa.bindparam("new_start"),
            timestamp_end=sa.bindparam("new_end"),
        ),
        updates,
    )


def upgrade() -> None:
    _shift(_analysis_results.c.timestamp_start < _CUTOFF, 1)


def downgrade() -> None:
    _shift(_analysis_results.c.timestamp_start >= _CUTOFF, -1)
