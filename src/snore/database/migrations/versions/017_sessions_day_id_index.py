"""Index ``sessions.day_id`` and delete orphaned ``days`` rows.

``DayManager.aggregate_day_statistics`` and the orphan probe in
``DayManager.recalculate_day`` both look sessions up by ``day_id``.  Without an
index each is a full ``sessions`` scan, so ``snore db recompute-days`` costs
O(days x sessions).  The model-side ``Index`` covers fresh DBs via
``create_all``; this migration covers pre-existing DBs.  Both use the name
``ix_sessions_day_id`` so the ``test_migration_schema_drift`` parity check
stays green.

Before ``recalculate_day`` pruned orphans, deleting every session of a day
left a zero-session ``days`` shell behind.  The one-off DELETE below removes
those historical shells so the lifecycle rule holds for existing databases
without a manual ``recompute-days`` run.  Days whose sessions are all disabled
are still referenced and are kept.

Revision ID: 017_sessions_day_id_index
Revises: 016_validation_runs
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "017_sessions_day_id_index"
down_revision: str | Sequence[str] | None = "016_validation_runs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "ix_sessions_day_id"
_TABLE = "sessions"


def upgrade() -> None:
    bind = op.get_bind()
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    existing_indexes = {ix["name"] for ix in sa_inspect(bind).get_indexes(_TABLE)}
    if _INDEX not in existing_indexes:
        op.create_index(_INDEX, _TABLE, ["day_id"])

    op.execute(
        sa.text(
            "DELETE FROM days WHERE id NOT IN "
            "(SELECT day_id FROM sessions WHERE day_id IS NOT NULL)"
        )
    )


def downgrade() -> None:
    # The orphan-row DELETE is not reversible; only the index is dropped.
    bind = op.get_bind()
    from sqlalchemy import inspect as sa_inspect  # noqa: PLC0415

    existing_indexes = {ix["name"] for ix in sa_inspect(bind).get_indexes(_TABLE)}
    if _INDEX in existing_indexes:
        op.drop_index(_INDEX, table_name=_TABLE)
