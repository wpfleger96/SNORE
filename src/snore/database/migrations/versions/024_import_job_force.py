"""Add ``force`` to ``import_job_records``.

Startup resume re-runs an upload whose spool directory survived a restart,
rebuilding the job from this row.  A forced re-parse upload must resume forced,
or it would skip every session it was meant to replace.  Existing rows are
unforced.

Revision ID: 024_import_job_force
Revises: 023_analysis_results_covering_index
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op
from sqlalchemy import inspect as sa_inspect

revision: str = "024_import_job_force"
down_revision: str | Sequence[str] | None = "023_analysis_results_covering_index"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "import_job_records"
_COLUMN = "force"


def upgrade() -> None:
    existing_cols = {c["name"] for c in sa_inspect(op.get_bind()).get_columns(_TABLE)}
    if _COLUMN not in existing_cols:
        op.add_column(
            _TABLE,
            sa.Column(
                _COLUMN, sa.Boolean(), nullable=False, server_default=sa.text("0")
            ),
        )


def downgrade() -> None:
    op.drop_column(_TABLE, _COLUMN)
