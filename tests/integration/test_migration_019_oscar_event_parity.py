"""Migration 019 removes invented H/RE durations and recounts sessions like OSCAR."""

import sqlite3

import pytest

from alembic import command as alembic_command

from snore.database.session import (
    _build_alembic_config,
    cleanup_database,
    init_database,
)

_PRE_019 = "018_day_device_indices"
_RESMED_SESSION = 1
_OSCAR_SESSION = 2
_UNAFFECTED_SESSION = 3
_ZERO_USAGE_SESSION = 4
_DEMO_SESSION = 5

_RAW_START = "2025-01-10 22:30:15.123456"
_FLAG_TIME = "2025-01-10 22:30:25.123456"

# (id, session_id, event_type, start_time, duration_seconds)
_RESMED_H = (1, _RESMED_SESSION, "H", _RAW_START, 10.0)
_RESMED_RE = (2, _RESMED_SESSION, "RE", "2025-01-10 23:59:55.000000", 10.0)
_RESMED_REAL_OA = (3, _RESMED_SESSION, "OA", _RAW_START, 10.0)
_RESMED_LONG_H = (4, _RESMED_SESSION, "H", _RAW_START, 12.5)
_RESMED_UA = (5, _RESMED_SESSION, "UA", _RAW_START, 21.0)
_RESMED_CA = (6, _RESMED_SESSION, "CA", _RAW_START, 15.0)
_RESMED_CAA = (7, _RESMED_SESSION, "CAA", _RAW_START, 15.0)
_RESMED_MA = (8, _RESMED_SESSION, "MA", _RAW_START, 15.0)
_OSCAR_H = (9, _OSCAR_SESSION, "H", _RAW_START, 10.0)
_OSCAR_UA = (10, _OSCAR_SESSION, "UA", _RAW_START, 10.0)
_UNAFFECTED_H = (11, _UNAFFECTED_SESSION, "H", _RAW_START, 10.0)
_ZERO_USAGE_UA = (12, _ZERO_USAGE_SESSION, "UA", _RAW_START, 10.0)
_DEMO_H = (13, _DEMO_SESSION, "H", _RAW_START, 10.0)

_EVENTS = (
    _RESMED_H,
    _RESMED_RE,
    _RESMED_REAL_OA,
    _RESMED_LONG_H,
    _RESMED_UA,
    _RESMED_CA,
    _RESMED_CAA,
    _RESMED_MA,
    _OSCAR_H,
    _OSCAR_UA,
    _UNAFFECTED_H,
    _ZERO_USAGE_UA,
    _DEMO_H,
)

# session_id, central, mixed, ahi, cai, usage_hours, reras, hypopneas, obstructive.
# Sentinel 99 in columns the migration must not write.
_STAT_COLUMNS = (
    "session_id, central_apneas, mixed_apneas, ahi, cai, usage_hours, reras, "
    "hypopneas, obstructive_apneas, oai, hi, rei"
)
_RESMED_STATS = (_RESMED_SESSION, 1, 0, 1.0, 0.25, 4.0, 99, 99, 99, 9.5, 9.5, None)
_OSCAR_STATS = (_OSCAR_SESSION, 0, 0, 0.5, 0.0, 2.0, 99, 99, 99, 9.5, 9.5, None)
_UNAFFECTED_STATS = (
    _UNAFFECTED_SESSION,
    7,
    0,
    1.23,
    4.56,
    3.0,
    99,
    99,
    99,
    9.5,
    9.5,
    None,
)
_ZERO_USAGE_STATS = (
    _ZERO_USAGE_SESSION,
    0,
    0,
    None,
    None,
    0.0,
    99,
    99,
    99,
    None,
    None,
    None,
)


def _seed(con: sqlite3.Connection) -> None:
    con.execute(
        "INSERT INTO days (id, device_id, date, session_count, total_therapy_hours, "
        "obstructive_apneas, central_apneas, hypopneas, reras, ahi, created_at, "
        "updated_at) VALUES (1, 1, '2025-01-10', 1, 4.0, 1, 1, 1, 1, 1.0, "
        "'2025-01-01 00:00:00', '2025-01-01 00:00:00')"
    )
    for session_id, source in (
        (_RESMED_SESSION, "resmed_edf"),
        (_OSCAR_SESSION, "oscar_binary"),
        (_UNAFFECTED_SESSION, "resmed_edf"),
        (_ZERO_USAGE_SESSION, "resmed_edf"),
        (_DEMO_SESSION, "demo"),
    ):
        con.execute(
            "INSERT INTO sessions (id, device_id, device_session_id, start_time, "
            "end_time, duration_seconds, import_date, import_source, "
            "data_quality_notes, has_waveform_data, has_event_data, has_statistics, "
            "enabled) VALUES (?, 1, ?, '2025-01-10 22:00:00.000000', "
            "'2025-01-11 02:00:00.000000', 14400, '2025-01-11 03:00:00.000000', ?, "
            "'{}', 0, 1, 1, 1)",
            (session_id, str(session_id), source),
        )
    con.executemany(
        "INSERT INTO events (id, session_id, event_type, start_time, duration_seconds) "
        "VALUES (?, ?, ?, ?, ?)",
        _EVENTS,
    )
    con.executemany(
        f"INSERT INTO statistics ({_STAT_COLUMNS}, flow_limitations) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
        [_RESMED_STATS, _OSCAR_STATS, _UNAFFECTED_STATS, _ZERO_USAGE_STATS],
    )


def _events(db_path: str) -> dict[int, tuple]:
    con = sqlite3.connect(db_path)
    try:
        return {
            row[0]: row
            for row in con.execute(
                "SELECT id, session_id, event_type, start_time, duration_seconds "
                "FROM events"
            )
        }
    finally:
        con.close()


def _statistics(db_path: str) -> dict[int, tuple]:
    con = sqlite3.connect(db_path)
    try:
        return {
            row[0]: row
            for row in con.execute(f"SELECT {_STAT_COLUMNS} FROM statistics")
        }
    finally:
        con.close()


def _unclassified(db_path: str) -> dict[int, tuple[int, str]]:
    con = sqlite3.connect(db_path)
    try:
        return {
            row[0]: (row[1], row[2])
            for row in con.execute(
                "SELECT session_id, unclassified_apneas, typeof(unclassified_apneas) "
                "FROM statistics"
            )
        }
    finally:
        con.close()


@pytest.fixture
async def pre_019(tmp_path):
    """A seeded DB at 018, plus its Alembic config."""
    db_path = str(tmp_path / "pre019.db")
    cfg = _build_alembic_config(f"sqlite:///{db_path}")
    await init_database(db_path)
    await cleanup_database()
    alembic_command.downgrade(cfg, _PRE_019)
    con = sqlite3.connect(db_path)
    try:
        _seed(con)
        con.commit()
    finally:
        con.close()
    return db_path, cfg


async def test_upgrade_moves_invented_durations_back_to_annotation_time(pre_019):
    db_path, cfg = pre_019

    alembic_command.upgrade(cfg, "head")

    events = _events(db_path)
    assert events[1] == (1, _RESMED_SESSION, "H", _FLAG_TIME, None)
    assert events[2] == (2, _RESMED_SESSION, "RE", "2025-01-11 00:00:05.000000", None)
    assert events[_DEMO_H[0]] == (_DEMO_H[0], _DEMO_SESSION, "H", _FLAG_TIME, None)


async def test_upgrade_leaves_real_durations_and_oscar_events_alone(pre_019):
    db_path, cfg = pre_019

    alembic_command.upgrade(cfg, "head")

    events = _events(db_path)
    for untouched in (_RESMED_REAL_OA, _RESMED_LONG_H, _OSCAR_H):
        assert events[untouched[0]] == untouched


async def test_upgrade_recounts_sessions_with_uncounted_events(pre_019):
    db_path, cfg = pre_019

    alembic_command.upgrade(cfg, "head")

    stats = _statistics(db_path)
    # OA + 2 H + UA + CA + CAA + MA = 7 over 4.0 h; CA + CAA = 2.
    assert stats[_RESMED_SESSION] == (
        _RESMED_SESSION,
        2,
        1,
        pytest.approx(7 / 4.0),
        pytest.approx(2 / 4.0),
        4.0,
        99,
        99,
        99,
        9.5,
        9.5,
        None,
    )
    # OSCAR sessions are recounted too: H + UA = 2 over 2.0 h.
    assert stats[_OSCAR_SESSION][3] == pytest.approx(1.0)
    assert _unclassified(db_path) == {
        _RESMED_SESSION: (1, "integer"),
        _OSCAR_SESSION: (1, "integer"),
        _UNAFFECTED_SESSION: (0, "integer"),
        _ZERO_USAGE_SESSION: (1, "integer"),
    }


async def test_upgrade_leaves_sessions_without_those_events_identical(pre_019):
    db_path, cfg = pre_019

    alembic_command.upgrade(cfg, "head")

    assert _statistics(db_path)[_UNAFFECTED_SESSION] == _UNAFFECTED_STATS


async def test_upgrade_keeps_rates_null_without_usage(pre_019):
    db_path, cfg = pre_019

    alembic_command.upgrade(cfg, "head")

    assert _statistics(db_path)[_ZERO_USAGE_SESSION][3:5] == (None, None)


async def test_upgrade_adds_days_column_as_zero_and_leaves_day_values(pre_019):
    db_path, cfg = pre_019

    alembic_command.upgrade(cfg, "head")

    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT unclassified_apneas, central_apneas, ahi FROM days"
        ).fetchone()
    finally:
        con.close()
    assert row == (0, 1, 1.0)


async def test_downgrade_restores_events_and_statistics(pre_019):
    db_path, cfg = pre_019
    events_before = _events(db_path)
    alembic_command.upgrade(cfg, "head")

    alembic_command.downgrade(cfg, _PRE_019)

    assert _events(db_path) == events_before
    stats = _statistics(db_path)
    # Old recount: OA + H + H + CA = 4 over 4.0 h, CA = 1, mixed back to 0.
    assert stats[_RESMED_SESSION][:5] == (
        _RESMED_SESSION,
        1,
        0,
        pytest.approx(4 / 4.0),
        pytest.approx(1 / 4.0),
    )
    assert stats[_UNAFFECTED_SESSION] == _UNAFFECTED_STATS
    assert stats[_ZERO_USAGE_SESSION] == _ZERO_USAGE_STATS


async def test_downgrade_drops_columns_and_keeps_day_triggers(pre_019):
    db_path, cfg = pre_019
    alembic_command.upgrade(cfg, "head")

    alembic_command.downgrade(cfg, _PRE_019)

    con = sqlite3.connect(db_path)
    try:
        for table in ("days", "statistics"):
            columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
            assert "unclassified_apneas" not in columns
        with pytest.raises(sqlite3.IntegrityError, match="invalid days.index_source"):
            con.execute("UPDATE days SET index_source = 'bogus' WHERE id = 1")
    finally:
        con.close()


async def test_reupgrade_after_downgrade_reaches_the_same_state(pre_019):
    db_path, cfg = pre_019
    alembic_command.upgrade(cfg, "head")
    events = _events(db_path)
    stats = _statistics(db_path)

    alembic_command.downgrade(cfg, _PRE_019)
    alembic_command.upgrade(cfg, "head")

    assert _events(db_path) == events
    assert _statistics(db_path) == stats


async def test_checksum_replay_of_019_keeps_corrected_data(pre_019):
    """Startup replay of an edited 019 runs the on-disk downgrade, then upgrade."""
    db_path, cfg = pre_019
    alembic_command.upgrade(cfg, "head")
    events = _events(db_path)
    await init_database(db_path)
    await cleanup_database()
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            "UPDATE snore_migration_checksums SET checksum = 'cafebabe' "
            "WHERE revision = '019_oscar_event_parity'"
        )
        con.commit()
    finally:
        con.close()

    await init_database(db_path)
    await cleanup_database()

    assert _events(db_path) == events
