"""Correct stored event durations and session AHI recounts to match OSCAR.

Adds ``unclassified_apneas`` to ``statistics`` and ``days``.  For a fresh Alembic
install (empty DB), ``001_baseline`` creates both tables from the current
``Base.metadata``, which already includes the column, so the add is a no-op and
the corrections below match no rows.

ResMed hypopneas and arousals carry no duration, but the parser used to invent
10 s and shift the start back by it.  Every ResMed ``H``/``RE`` row at exactly
10.0 s is that default (no real annotation of those types has a duration), so
the upgrade moves it forward to the device's annotation time and clears the
duration.  Demo sessions are copies of parsed ResMed data and get the same
correction.  OSCAR-imported events never had the shift and are left alone.

The session recount used to drop ``UA``, ``CAA`` and ``MA`` events.  Sessions
with at least one of those are recounted from ``events``: ``central_apneas`` now
includes clear airway, ``mixed_apneas`` and ``unclassified_apneas`` are filled,
and where ``usage_hours > 0``, ``ahi`` and ``cai`` follow.  Every other session
keeps its row unchanged.  The AHI event types are spelled out here so this
revision never moves with a later change to ``AHI_EVENT_TYPES``.

Run ``snore db recompute-days`` afterwards: ``days`` is not touched beyond the
new column, so ``*_computed`` and the day-level UA counts stay stale until then.
No re-parse is needed.

Downgrade restores the old durations and recount, then drops the columns.  A
later re-upgrade restores ``statistics``, but ``days.unclassified_apneas`` needs
``snore db recompute-days`` again.

Revision ID: 019_oscar_event_parity
Revises: 018_day_device_indices
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "019_oscar_event_parity"
down_revision: str | Sequence[str] | None = "018_day_device_indices"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_UNCLASSIFIED_APNEAS = "unclassified_apneas"
_TABLES = ("days", "statistics")

_RESMED_HYPOPNEA_OR_RERA = (
    "event_type IN ('H', 'RE') AND session_id IN "
    "(SELECT id FROM sessions WHERE import_source IN ('resmed_edf', 'demo'))"
)

# The sessions the old recount got wrong, with both recounts' event counts.
_COUNTS = """
    WITH counts AS (
        SELECT
            session_id,
            SUM(event_type = 'CA') AS old_central,
            SUM(event_type IN ('CA', 'CAA')) AS central,
            SUM(event_type = 'MA') AS mixed,
            SUM(event_type = 'UA') AS unclassified,
            SUM(event_type IN ('OA', 'CA', 'H')) AS old_ahi_events,
            SUM(event_type IN ('OA', 'CA', 'CAA', 'MA', 'UA', 'H')) AS ahi_events
        FROM events
        GROUP BY session_id
        HAVING SUM(event_type IN ('UA', 'CAA', 'MA')) > 0
    )
"""


def _shifted_start_time(offset: str) -> str:
    """SQL moving a stored 'YYYY-MM-DD HH:MM:SS.ffffff' by ``offset`` seconds.

    SQLite's datetime() would drop the fraction, so only the first 19 characters
    go through it.
    """
    return (
        f"datetime(substr(start_time, 1, 19), '{offset} seconds') "
        "|| substr(start_time, 20)"
    )


def upgrade() -> None:
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    insp = sa_inspect(op.get_bind())
    for table in _TABLES:
        if _UNCLASSIFIED_APNEAS not in {c["name"] for c in insp.get_columns(table)}:
            op.add_column(
                table,
                sa.Column(
                    _UNCLASSIFIED_APNEAS,
                    sa.Integer(),
                    nullable=False,
                    server_default="0",
                ),
            )

    op.execute(
        sa.text(
            f"UPDATE events SET start_time = {_shifted_start_time('+10')}, "
            f"duration_seconds = NULL "
            f"WHERE duration_seconds = 10.0 AND {_RESMED_HYPOPNEA_OR_RERA}"
        )
    )
    op.execute(
        sa.text(
            f"""
            {_COUNTS}
            UPDATE statistics SET
                central_apneas = counts.central,
                mixed_apneas = counts.mixed,
                unclassified_apneas = counts.unclassified,
                ahi = CASE WHEN usage_hours > 0
                    THEN 1.0 * counts.ahi_events / usage_hours ELSE ahi END,
                cai = CASE WHEN usage_hours > 0
                    THEN 1.0 * counts.central / usage_hours ELSE cai END
            FROM counts
            WHERE statistics.session_id = counts.session_id
            """
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    # The batch rebuild drops and recreates ``days``; with foreign keys enforced
    # that would cascade-delete every session (``ON DELETE CASCADE``).  SQLite
    # ignores PRAGMA foreign_keys inside a transaction, so refuse rather than
    # try to switch it off here.
    if (
        bind.dialect.name == "sqlite"
        and bind.exec_driver_sql("PRAGMA foreign_keys").scalar()
    ):
        raise RuntimeError(
            "Downgrading 019_oscar_event_parity rebuilds the days and statistics "
            "tables; run it with PRAGMA foreign_keys=OFF to avoid cascading "
            "deletes."
        )

    op.execute(
        sa.text(
            f"UPDATE events SET start_time = {_shifted_start_time('-10')}, "
            f"duration_seconds = 10.0 "
            f"WHERE duration_seconds IS NULL AND {_RESMED_HYPOPNEA_OR_RERA}"
        )
    )
    op.execute(
        sa.text(
            f"""
            {_COUNTS}
            UPDATE statistics SET
                central_apneas = counts.old_central,
                mixed_apneas = 0,
                ahi = CASE WHEN usage_hours > 0
                    THEN 1.0 * counts.old_ahi_events / usage_hours ELSE ahi END,
                cai = CASE WHEN usage_hours > 0
                    THEN 1.0 * counts.old_central / usage_hours ELSE cai END
            FROM counts
            WHERE statistics.session_id = counts.session_id
            """
        )
    )

    insp = sa_inspect(bind)
    for table in _TABLES:
        if _UNCLASSIFIED_APNEAS not in {c["name"] for c in insp.get_columns(table)}:
            continue
        # batch_alter_table is the SQLite-portable way to drop a column
        # (table-copy recreate).
        with op.batch_alter_table(table) as batch_op:
            batch_op.drop_column(_UNCLASSIFIED_APNEAS)

    from snore.database.models import DAY_INDEX_SOURCE_TRIGGERS  # noqa: PLC0415

    # The ``days`` rebuild dropped these triggers.
    if bind.dialect.name == "sqlite":
        for trigger_sql in DAY_INDEX_SOURCE_TRIGGERS.values():
            op.execute(sa.text(trigger_sql))
