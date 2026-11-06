"""Index ``analysis_results`` on ``(session_id, created_at, id)``.

Every "latest analysis for a session" lookup (``snore.analysis.queries``)
filters by ``session_id`` and orders by ``created_at DESC, id DESC``.  Without
an index each lookup was a full ``analysis_results`` scan, and MCP tools issue
one per session.  The model-side ``Index`` covers fresh DBs via ``create_all``;
this migration covers pre-existing DBs under the same name.

Building the index reads every row once: ``created_at`` is stored after the
large ``programmatic_result_json`` column, so SQLite walks each row's overflow
chain.  The first startup after deploy is slower on a large table.

Revision ID: 021_analysis_results_session_index
Revises: 020_analysis_result_wallclock
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import inspect as sa_inspect

revision: str = "021_analysis_results_session_index"
down_revision: str | Sequence[str] | None = "020_analysis_result_wallclock"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX = "ix_analysis_results_session_id_created_at_id"
_TABLE = "analysis_results"


def _has_index() -> bool:
    indexes = sa_inspect(op.get_bind()).get_indexes(_TABLE)
    return _INDEX in {ix["name"] for ix in indexes}


def upgrade() -> None:
    if not _has_index():
        op.create_index(_INDEX, _TABLE, ["session_id", "created_at", "id"])


def downgrade() -> None:
    if _has_index():
        op.drop_index(_INDEX, table_name=_TABLE)
