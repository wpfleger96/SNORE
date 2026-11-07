"""Migration 022 rescales Apple Health SpO2 fractions to percent."""

import sqlite3

from pathlib import Path

from alembic import command as alembic_command
from alembic.config import Config

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_PRE_022 = "021_analysis_results_session_index"
_SPO2 = "HKQuantityTypeIdentifierOxygenSaturation"
_RR = "HKQuantityTypeIdentifierRespiratoryRate"

# (id, record_type, source_name, start_time, value_num)
_FRACTION = (1, _SPO2, "Watch", "2024-01-16 02:00:00.000000", 0.95)
_FRACTION_RESIDUE = (2, _SPO2, "Watch", "2024-01-16 02:05:00.000000", 0.93)
_PERCENT = (3, _SPO2, "Oximeter", "2024-01-16 02:00:00.000000", 96.0)
_IMPLAUSIBLE = (4, _SPO2, "Watch", "2024-01-16 02:10:00.000000", 0.2)
_RR_FRACTION = (5, _RR, "Watch", "2024-01-16 02:00:00.000000", 0.95)
# The same sample from an older fraction export and a newer percent export.
_DUP_FRACTION = (6, _SPO2, "Watch", "2024-01-16 03:00:00.000000", 0.97)
_DUP_PERCENT = (7, _SPO2, "Watch", "2024-01-16 03:00:00.000000", 97.0)
# The rescale range is [0.5, 1.0] inclusive, matching the parser's rule.
_LOWER_BOUND = (8, _SPO2, "Watch", "2024-01-16 04:00:00.000000", 0.5)
_UPPER_BOUND = (9, _SPO2, "Watch", "2024-01-16 04:05:00.000000", 1.0)
_ABOVE_RANGE = (10, _SPO2, "Watch", "2024-01-16 04:10:00.000000", 1.5)

_ROWS = (
    _FRACTION,
    _FRACTION_RESIDUE,
    _PERCENT,
    _IMPLAUSIBLE,
    _RR_FRACTION,
    _DUP_FRACTION,
    _DUP_PERCENT,
    _LOWER_BOUND,
    _UPPER_BOUND,
    _ABOVE_RANGE,
)


async def _db_at_021(tmp_path: Path) -> tuple[str, Config]:
    db_path = str(tmp_path / "pre022.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, _PRE_022)
    return db_path, cfg


def _seed(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "INSERT INTO health_samples (id, profile_id, record_type, source_name, "
            "start_time, end_time, value_num, unit, night_date, ingest_channel, "
            "imported_at) VALUES (?, 1, ?, ?, ?, ?, ?, '%', '2024-01-15', "
            "'export_xml', '2024-02-01 00:00:00.000000')",
            [(i, t, src, start, start, v) for i, t, src, start, v in _ROWS],
        )
        con.commit()
    finally:
        con.close()


def _values(db_path: str) -> dict[int, float]:
    con = sqlite3.connect(db_path)
    try:
        return dict(con.execute("SELECT id, value_num FROM health_samples"))
    finally:
        con.close()


async def test_upgrade_rescales_spo2_fractions_only(tmp_path: Path) -> None:
    db_path, cfg = await _db_at_021(tmp_path)
    _seed(db_path)

    alembic_command.upgrade(cfg, "head")

    assert _values(db_path) == {
        _FRACTION[0]: 95.0,
        _FRACTION_RESIDUE[0]: 93.0,
        _PERCENT[0]: 96.0,
        _IMPLAUSIBLE[0]: 0.2,
        _RR_FRACTION[0]: 0.95,
        _DUP_PERCENT[0]: 97.0,
        _LOWER_BOUND[0]: 50.0,
        _UPPER_BOUND[0]: 100.0,
        _ABOVE_RANGE[0]: 1.5,
    }


async def test_rescaled_row_dedups_against_reimported_percent(tmp_path: Path) -> None:
    """After the upgrade, a re-import of the same sample is ignored by the dedup index."""
    db_path, cfg = await _db_at_021(tmp_path)
    _seed(db_path)
    alembic_command.upgrade(cfg, "head")

    con = sqlite3.connect(db_path)
    try:
        cur = con.execute(
            "INSERT OR IGNORE INTO health_samples (profile_id, record_type, "
            "source_name, start_time, end_time, value_num, unit, night_date, "
            "ingest_channel, imported_at) VALUES (1, ?, ?, ?, ?, 95.0, '%', "
            "'2024-01-15', 'export_xml', '2024-03-01 00:00:00.000000')",
            (_SPO2, _FRACTION[2], _FRACTION[3], _FRACTION[3]),
        )
        con.commit()
    finally:
        con.close()

    assert cur.rowcount == 0


async def test_downgrade_keeps_rescaled_values(tmp_path: Path) -> None:
    db_path, cfg = await _db_at_021(tmp_path)
    _seed(db_path)
    alembic_command.upgrade(cfg, "head")

    alembic_command.downgrade(cfg, _PRE_022)

    assert _values(db_path)[_FRACTION[0]] == 95.0
