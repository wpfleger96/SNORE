"""Rescale Apple Health SpO2 samples stored as fractions to percent.

Sources write SpO2 as a fraction (0.95) or a percent (95), both with unit ``%``,
and the parser used to store either as recorded.  It now scales fractions at
import (``_spo2_percent`` in ``parsers/apple_health/type_handlers.py``); this
revision applies the same rule to existing rows: a value in [0.5, 1.0] is a
fraction and becomes ``ROUND(value_num * 100, 4)``.  Values in neither plausible
range are left as recorded.  ``value_num`` is the only value column.

The dedup index includes ``value_num``, so an older fraction export and a newer
percent export of the same sample are both stored.  Rescaling would make the
pair collide, so the fraction copy of such a pair is deleted first.

The SQL is SQLite's, so any other dialect skips it: that can only be a fresh
install, with no rows to rescale.

Downgrade is a no-op: a rescaled row cannot be told from one recorded as a
percent, so the original scale is lost.

Revision ID: 022_health_spo2_percent
Revises: 021_analysis_results_session_index
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "022_health_spo2_percent"
down_revision: str | Sequence[str] | None = "021_analysis_results_session_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Spelled out so this revision never moves with a later parser change.
_SPO2_FRACTION = (
    "record_type = 'HKQuantityTypeIdentifierOxygenSaturation' "
    "AND value_num BETWEEN 0.5 AND 1.0"
)


def upgrade() -> None:
    if op.get_bind().dialect.name != "sqlite":
        return
    op.execute(
        sa.text(
            f"""
            DELETE FROM health_samples
            WHERE {_SPO2_FRACTION} AND EXISTS (
                SELECT 1 FROM health_samples AS pct
                WHERE pct.profile_id = health_samples.profile_id
                    AND pct.record_type = health_samples.record_type
                    AND pct.source_name = health_samples.source_name
                    AND pct.start_time = health_samples.start_time
                    AND pct.end_time = health_samples.end_time
                    AND pct.value_num = ROUND(health_samples.value_num * 100, 4)
            )
            """
        )
    )
    op.execute(
        sa.text(
            "UPDATE health_samples SET value_num = ROUND(value_num * 100, 4) "
            f"WHERE {_SPO2_FRACTION}"
        )
    )


def downgrade() -> None:
    # Lossy upgrade: rescaled rows are indistinguishable from percent rows.
    pass
