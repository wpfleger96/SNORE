"""Migration 018 adds the device-index headline columns and backfills days as derived."""

import sqlite3

from alembic import command as alembic_command

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_NEW_COLUMNS = {
    "ahi_computed",
    "oai_computed",
    "cai_computed",
    "hi_computed",
    "index_source",
}


def _statistics_columns(con: sqlite3.Connection) -> set[str]:
    return {row[1] for row in con.execute("PRAGMA table_info(statistics)")}


def _insert_day(con: sqlite3.Connection, day_id: int, indices: tuple) -> None:
    con.execute(
        "INSERT INTO days (id, device_id, date, session_count, "
        "total_therapy_hours, obstructive_apneas, central_apneas, hypopneas, "
        "reras, ahi, oai, cai, hi, created_at, updated_at) "
        "VALUES (?, 1, ?, 1, 6.0, 0, 0, 0, 0, ?, ?, ?, ?, "
        "'2025-01-01 00:00:00', '2025-01-01 00:00:00')",
        (day_id, f"2025-01-{day_id:02d}", *indices),
    )


async def test_upgrade_from_017_backfills_recount_as_derived(tmp_path):
    db_path = str(tmp_path / "pre018.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, "017_sessions_day_id_index")

    con = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(days)")}
        assert not columns & _NEW_COLUMNS
        assert "usage_hours_device" not in _statistics_columns(con)
        _insert_day(con, 1, (5.0, 2.5, 1.25, 1.25))
        _insert_day(con, 2, (None, 2.0, None, None))
        _insert_day(con, 3, (None, None, None, None))
        con.commit()
    finally:
        con.close()

    alembic_command.upgrade(cfg, "head")

    con = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(days)")}
        assert _NEW_COLUMNS <= columns
        assert "usage_hours_device" in _statistics_columns(con)
        rows = con.execute(
            "SELECT id, ahi_computed, oai_computed, cai_computed, hi_computed, "
            "index_source FROM days ORDER BY id"
        ).fetchall()
    finally:
        con.close()

    assert rows == [
        (1, 5.0, 2.5, 1.25, 1.25, "derived"),
        (2, None, 2.0, None, None, "derived"),
        (3, None, None, None, None, None),
    ]
