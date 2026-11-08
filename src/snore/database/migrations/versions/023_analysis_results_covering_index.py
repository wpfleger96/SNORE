"""Cover ``engine_versions_json`` in the latest-analysis index.

Staleness checks (``snore analysis run --only stale``, the breath service's
STALE_VERSION classification) read each session's latest row's
``engine_versions_json``.  That column is stored after the large
``programmatic_result_json``, so reading it from the table walks the row's
overflow chain (often over a megabyte per row).  Indexing
``(session_id, created_at, id, engine_versions_json)`` lets the latest-row
ranking in ``snore.analysis.queries`` run from the index alone.  Its leading
columns are 021's index, which this revision drops as redundant.  The
model-side ``Index`` covers fresh DBs via ``create_all``; this migration covers
pre-existing DBs under the same name.

Building the index reads every row once, through the same overflow chains, so
the first startup after deploy is slower on a large table.

Revision ID: 023_analysis_results_covering_index
Revises: 022_health_spo2_percent
"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import inspect as sa_inspect

revision: str = "023_analysis_results_covering_index"
down_revision: str | Sequence[str] | None = "022_health_spo2_percent"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLE = "analysis_results"
_INDEX = "ix_analysis_results_session_latest"
_INDEX_021 = "ix_analysis_results_session_id_created_at_id"


def _index_names() -> set[str | None]:
    return {ix["name"] for ix in sa_inspect(op.get_bind()).get_indexes(_TABLE)}


def upgrade() -> None:
    names = _index_names()
    if _INDEX not in names:
        op.create_index(
            _INDEX,
            _TABLE,
            ["session_id", "created_at", "id", "engine_versions_json"],
        )
    if _INDEX_021 in names:
        op.drop_index(_INDEX_021, table_name=_TABLE)


def downgrade() -> None:
    names = _index_names()
    if _INDEX_021 not in names:
        op.create_index(_INDEX_021, _TABLE, ["session_id", "created_at", "id"])
    if _INDEX in names:
        op.drop_index(_INDEX, table_name=_TABLE)
