"""Store ``analysis_results`` windows as device wall-clock times.

``analysis_results.timestamp_start``/``timestamp_end`` are documented as device
wall-clock, but analysis wrote them as ``datetime.fromtimestamp(offset, tz=UTC)``
with ``offset`` in seconds from session start (of the first/last flow sample),
so a stored value is ``1970-01-01 00:00:00 + offset``.  ``snore db scrub-demo``
then copies rows into a demo profile shifted by a whole number of days ``D``
(positive or negative), giving ``1970-01-01 + offset + D`` against a demo
session that starts at ``start_time + D``.

In both cases the start offset is ``(timestamp_start - 1970-01-01) mod 1 day``:
the first flow sample sits well under a day after session start, and a
whole-day shift leaves the remainder unchanged (Python's ``timedelta`` modulo
is non-negative, so shifts back into 1969 work too).  The upgrade rewrites each
window as ``sessions.start_time + start_offset`` and keeps its duration, which
neither the bug nor the demo shift changes.  No re-analysis is needed.

The upgrade touches only rows whose ``timestamp_start`` is before their
session's ``start_time``: buggy rows of either kind are years earlier, while
correct wall-clock rows are at or after it because offsets are non-negative.
Rows whose session has no ``start_time`` are skipped.  The arithmetic runs in
Python rather than SQLite ``datetime()``, which drops microseconds, through
dialect-neutral Core statements that read and write the ORM's storage format.

Downgrade is lossy: it turns every row at or after its session's start back
into ``1970-01-01 + offset``, including rows written correctly after the
upgrade, and demo rows lose their day shift.

Revision ID: 020_analysis_result_wallclock
Revises: 019_oscar_event_parity
"""

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta

import sqlalchemy as sa

from alembic import op

revision: str = "020_analysis_result_wallclock"
down_revision: str | Sequence[str] | None = "019_oscar_event_parity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EPOCH = datetime(1970, 1, 1)
_ONE_DAY = timedelta(days=1)

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


def _move_windows(
    where: sa.ColumnElement[bool],
    new_start: Callable[[datetime, datetime], datetime],
) -> None:
    """Move each matching window to ``new_start(timestamp_start, session_start)``.

    The window's duration is kept.
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
        start = new_start(row.timestamp_start, row.start_time)
        updates.append(
            {
                "row_id": row.id,
                "new_start": start,
                "new_end": start + (row.timestamp_end - row.timestamp_start),
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
    _move_windows(
        _analysis_results.c.timestamp_start < _sessions.c.start_time,
        lambda start, session_start: session_start + (start - _EPOCH) % _ONE_DAY,
    )


def downgrade() -> None:
    _move_windows(
        _analysis_results.c.timestamp_start >= _sessions.c.start_time,
        lambda start, session_start: _EPOCH + (start - session_start),
    )
