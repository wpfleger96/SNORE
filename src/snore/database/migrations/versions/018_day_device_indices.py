"""Add the device-index headline columns to ``days`` and ``statistics``.

ResMed STR reports the device's own daily AHI/OAI/CAI/HI (stored per session in
``statistics.*_device`` by ``015_statistics_device_indices``).  STR values are
daily, so they are only meaningful at day level.  ``DayManager`` now keeps the
SNORE recount in ``*_computed`` and headlines either the trusted device value
or the recount in ``ahi/oai/cai/hi``, recording the choice in ``index_source``
(``"device"`` or ``"derived"``).  Trust requires the imported mask-on time to
match the device's daily mask-on time, stored per session in the new
``statistics.usage_hours_device`` column.

For a fresh Alembic install (empty DB), ``001_baseline`` creates both tables
from the current ``Base.metadata`` — which already includes these columns — so
the column adds are a no-op there.  For an existing install stamped before this
revision, it adds the columns and backfills ``days``: before this revision
``ahi/oai/cai/hi`` always held the recount, so copying them into ``*_computed``
with ``index_source = 'derived'`` is exact.  Days with no index stay NULL.
On SQLite, triggers reject any ``days.index_source`` outside ``IndexSource``
(see ``DAY_INDEX_SOURCE_TRIGGERS``); a CHECK would need a ``days`` rebuild.
``usage_hours_device`` stays NULL until sessions are re-imported, so days stay
derived until then; a forced re-import (``snore import --force``) fills
``usage_hours_device`` and re-aggregates the days itself, promoting days whose
device values are trusted to ``index_source = 'device'``.

Downgrade first copies the recount back into ``ahi/oai/cai/hi`` on device days,
so pre-018 code (which reads those columns as the recount) sees the recount and
a later re-upgrade backfills ``*_computed`` with the recount labelled
``derived``.  The re-upgrade cannot restore device days: the dropped
``usage_hours_device`` needs another forced re-import.

Revision ID: 018_day_device_indices
Revises: 017_sessions_day_id_index
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "018_day_device_indices"
down_revision: str | Sequence[str] | None = "017_sessions_day_id_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DAYS = "days"
_COMPUTED_COLUMNS = ("ahi_computed", "oai_computed", "cai_computed", "hi_computed")
_INDEX_SOURCE = "index_source"
_STATISTICS = "statistics"
_USAGE_HOURS_DEVICE = "usage_hours_device"


def upgrade() -> None:
    bind = op.get_bind()
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    insp = sa_inspect(bind)
    day_cols = {c["name"] for c in insp.get_columns(_DAYS)}
    for column in _COMPUTED_COLUMNS:
        if column not in day_cols:
            op.add_column(_DAYS, sa.Column(column, sa.Float(), nullable=True))
    if _INDEX_SOURCE not in day_cols:
        op.add_column(_DAYS, sa.Column(_INDEX_SOURCE, sa.String(16), nullable=True))

    stat_cols = {c["name"] for c in insp.get_columns(_STATISTICS)}
    if _USAGE_HOURS_DEVICE not in stat_cols:
        op.add_column(
            _STATISTICS, sa.Column(_USAGE_HOURS_DEVICE, sa.Float(), nullable=True)
        )

    from snore.database.models import DAY_INDEX_SOURCE_TRIGGERS  # noqa: PLC0415

    if bind.dialect.name == "sqlite":
        for trigger_sql in DAY_INDEX_SOURCE_TRIGGERS.values():
            op.execute(sa.text(trigger_sql))

    # Rows with index_source already set were aggregated by the new code.
    op.execute(
        sa.text(
            "UPDATE days SET ahi_computed = ahi, oai_computed = oai, "
            "cai_computed = cai, hi_computed = hi, index_source = 'derived' "
            "WHERE index_source IS NULL AND (ahi IS NOT NULL OR oai IS NOT NULL "
            "OR cai IS NOT NULL OR hi IS NOT NULL)"
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
            "Downgrading 018_day_device_indices rebuilds the days and "
            "statistics tables; run it with PRAGMA foreign_keys=OFF to avoid "
            "cascading deletes."
        )

    from snore.database.models import DAY_INDEX_SOURCE_TRIGGERS  # noqa: PLC0415

    if bind.dialect.name == "sqlite":
        for trigger_name in DAY_INDEX_SOURCE_TRIGGERS:
            op.execute(sa.text(f"DROP TRIGGER IF EXISTS {trigger_name}"))

    insp = sa_inspect(bind)
    if _INDEX_SOURCE in {c["name"] for c in insp.get_columns(_DAYS)}:
        op.execute(
            sa.text(
                "UPDATE days SET ahi = ahi_computed, oai = oai_computed, "
                "cai = cai_computed, hi = hi_computed WHERE index_source = 'device'"
            )
        )
    # batch_alter_table is the SQLite-portable way to drop a column
    # (table-copy recreate).
    for table, columns in (
        (_DAYS, (*_COMPUTED_COLUMNS, _INDEX_SOURCE)),
        (_STATISTICS, (_USAGE_HOURS_DEVICE,)),
    ):
        existing_cols = {c["name"] for c in insp.get_columns(table)}
        to_drop = [c for c in columns if c in existing_cols]
        if not to_drop:
            continue
        with op.batch_alter_table(table) as batch_op:
            for column in to_drop:
                batch_op.drop_column(column)
