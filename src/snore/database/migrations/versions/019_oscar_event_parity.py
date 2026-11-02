"""Correct stored event durations and session AHI recounts to match OSCAR.

Adds ``unclassified_apneas`` to ``statistics`` and ``days``, and an ``events``
index on ``(session_id, event_type)`` for ``device_event_types``.  For a fresh
Alembic install (empty DB), ``001_baseline`` creates all of them from the current
``Base.metadata``, so the adds are no-ops and the corrections below match no rows.
Those corrections are SQLite SQL, so any other dialect skips them: it can only be
a fresh install, with no rows to correct.

ResMed hypopneas and arousals carry no duration, but the parser used to invent
10 s and shift the start back by it.  Every ResMed ``H``/``RE`` row at exactly
10.0 s is that default (no real annotation of those types has a duration), so
the upgrade moves it forward to the device's annotation time and clears the
duration.  Demo sessions are normally clones of ResMed data and get the same
correction.  ``scrub_demo`` can clone an OSCAR source, and such a demo is
corrected wrongly here until ``scrub_demo`` rebuilds it.

OSCAR-imported events keep their start times.  The old OSCAR parser did apply its
10 s default and shift when a duration was missing, but that branch is unlikely
on real files and those rows can't be told from a real 10 s event.  A stored
duration of 0 is the old form of "none recorded", so it becomes NULL.

The session recount used to drop ``UA``, ``CAA`` and ``MA`` events.  Sessions
with at least one of those are recounted from ``events``: ``central_apneas`` now
includes clear airway, ``mixed_apneas`` and ``unclassified_apneas`` are filled,
and where ``usage_hours > 0``, ``ahi`` and ``cai`` follow.  Every other session
keeps its row unchanged.  The AHI event types are spelled out here so this
revision never moves with a later change to ``AHI_EVENT_TYPES``.
``days.unclassified_apneas`` is then backfilled the way day aggregation sums it.

Run ``snore db recompute-days`` afterwards: the rest of ``days`` is not touched,
so ``days.central_apneas``, ``*_computed`` and the headline ``ahi``/``cai`` on
derived days stay stale until then.  No re-parse is needed, except at session
edges: a migrated ``H``/``RE`` can sit up to 10 s past session end, and one the
old parser dropped from a session's first 10 s only returns with a re-parse.

Downgrade restores the old durations and recount, then drops the index and the
columns.

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
_EVENT_TYPE_INDEX = "ix_events_session_id_event_type"

_RESMED_HYPOPNEA_OR_RERA = (
    "event_type IN ('H', 'RE') AND session_id IN "
    "(SELECT id FROM sessions WHERE import_source IN ('resmed_edf', 'demo'))"
)
_OSCAR_EVENT = (
    "session_id IN (SELECT id FROM sessions WHERE import_source = 'oscar_binary')"
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
    bind = op.get_bind()
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    insp = sa_inspect(bind)
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
    if _EVENT_TYPE_INDEX not in {ix["name"] for ix in insp.get_indexes("events")}:
        op.create_index(_EVENT_TYPE_INDEX, "events", ["session_id", "event_type"])

    if bind.dialect.name != "sqlite":
        return

    op.execute(
        sa.text(
            f"UPDATE events SET start_time = {_shifted_start_time('+10')}, "
            f"duration_seconds = NULL "
            f"WHERE duration_seconds = 10.0 AND {_RESMED_HYPOPNEA_OR_RERA}"
        )
    )
    op.execute(
        sa.text(
            "UPDATE events SET duration_seconds = NULL "
            f"WHERE duration_seconds = 0 AND {_OSCAR_EVENT}"
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
    op.execute(
        sa.text(
            """
            UPDATE days SET unclassified_apneas = (
                SELECT COALESCE(SUM(statistics.unclassified_apneas), 0)
                FROM sessions
                JOIN statistics ON statistics.session_id = sessions.id
                WHERE sessions.day_id = days.id AND sessions.enabled = 1
            )
            """
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    if bind.dialect.name == "sqlite":
        # The batch rebuild drops and recreates ``days``; with foreign keys
        # enforced that would cascade-delete every session (``ON DELETE CASCADE``).
        # SQLite ignores PRAGMA foreign_keys inside a transaction, so refuse rather
        # than try to switch it off here.
        if bind.exec_driver_sql("PRAGMA foreign_keys").scalar():
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
                "UPDATE events SET duration_seconds = 0 "
                f"WHERE duration_seconds IS NULL AND {_OSCAR_EVENT}"
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
    if _EVENT_TYPE_INDEX in {ix["name"] for ix in insp.get_indexes("events")}:
        op.drop_index(_EVENT_TYPE_INDEX, table_name="events")
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
