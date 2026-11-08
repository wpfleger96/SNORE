"""Migration 023 replaces 021's index with one covering engine_versions_json."""

import sqlite3

from pathlib import Path

from alembic import command as alembic_command
from alembic.config import Config

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_INDEX = "ix_analysis_results_session_latest"
_INDEX_021 = "ix_analysis_results_session_id_created_at_id"
_COLUMNS = ["session_id", "created_at", "id", "engine_versions_json"]
_COLUMNS_021 = ["session_id", "created_at", "id"]


def _indexes(db_path: str) -> dict[str, list[str]]:
    con = sqlite3.connect(db_path)
    try:
        names = [
            row[1]
            for row in con.execute("PRAGMA index_list(analysis_results)")
            if not row[1].startswith("sqlite_autoindex")
        ]
        return {
            name: [row[2] for row in con.execute(f"PRAGMA index_info({name})")]
            for name in names
        }
    finally:
        con.close()


async def _db_at_022(tmp_path: Path) -> tuple[str, Config]:
    db_path = str(tmp_path / "pre023.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, "022_health_spo2_percent")
    return db_path, cfg


async def test_upgrade_from_022_replaces_021_index_with_covering_index(tmp_path):
    db_path, cfg = await _db_at_022(tmp_path)
    assert _indexes(db_path) == {_INDEX_021: _COLUMNS_021}

    alembic_command.upgrade(cfg, "head")

    assert _indexes(db_path) == {_INDEX: _COLUMNS}


async def test_downgrade_restores_021_index(tmp_path):
    db_path, cfg = await _db_at_022(tmp_path)
    alembic_command.upgrade(cfg, "head")

    alembic_command.downgrade(cfg, "022_health_spo2_percent")

    assert _indexes(db_path) == {_INDEX_021: _COLUMNS_021}


async def test_upgrade_is_idempotent_when_covering_index_exists(tmp_path):
    db_path, cfg = await _db_at_022(tmp_path)
    con = sqlite3.connect(db_path)
    con.execute(f"CREATE INDEX {_INDEX} ON analysis_results ({', '.join(_COLUMNS)})")
    con.commit()
    con.close()

    alembic_command.upgrade(cfg, "head")

    assert _indexes(db_path) == {_INDEX: _COLUMNS}


async def test_fresh_database_has_only_covering_index(tmp_path: Path) -> None:
    db_path = str(tmp_path / "fresh.db")
    await init_database(db_path)
    await cleanup_database()

    assert _indexes(db_path) == {_INDEX: _COLUMNS}
