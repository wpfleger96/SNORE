"""Pytest configuration and fixtures for SNORE tests."""

import atexit
import os
import shutil
import sqlite3
import tempfile  # load-bearing: mkdtemp used for the module-level test HOME below

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

# --- Isolated HOME (module-level, before any snore import) ---
# snore.constants resolves DEFAULT_DATABASE_PATH and every other ~/.snore path
# (raw backups, logs, spool, VACUUM/deploy markers) via Path.home() at import
# time, and CLI commands run without --db open DEFAULT_DATABASE_PATH directly,
# ignoring SNORE_DB_PATH.  A per-test HOME patch is too late for those, so the
# whole process gets a throwaway HOME before anything imports snore.  conftest
# is imported in every xdist worker, so each worker gets its own.  Matplotlib's
# font cache is keyed off HOME and takes ~15 s to rebuild, so pin it to a
# stable temp location instead of rebuilding it on every run.
_TEST_HOME = Path(tempfile.mkdtemp(prefix="snore-test-home-"))
atexit.register(shutil.rmtree, _TEST_HOME, ignore_errors=True)
os.environ["HOME"] = os.environ["USERPROFILE"] = str(_TEST_HOME)
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "snore-test-mplconfig")
)

sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("DATETIME", lambda s: datetime.fromisoformat(s.decode()))

# --- Hermetic env baseline (module-level, before any config resolves) ---
# The suite must be hermetic w.r.t. the invoking developer shell, which may
# export any config var that ``snore.api.config.load_config`` reads straight
# from ``os.environ`` (e.g. SNORE_BOOTSTRAP_ADMIN_EMAIL, SNORE_DATABASE_URL).
# Scrub every SNORE_* var plus the OAuth/CORS vars so config resolves from a
# clean slate that matches CI, which sets none of these.  This runs at module
# level, not in a fixture, because module-/session-scoped fixtures and the
# os.environ.setdefault below resolve config before any function-scoped fixture
# can act.
_LEAKY_ENV = {"GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "CORS_ORIGINS"}
for _name in [k for k in os.environ if k.startswith("SNORE_") or k in _LEAKY_ENV]:
    del os.environ[_name]

# Default to local auth mode for all tests so create_app() works without
# SNORE_SESSION_SECRET.  Phase 2 auth tests override this per-test via
# monkeypatch + the reset_auth_config fixture.
os.environ.setdefault("SNORE_AUTH_MODE", "local")

# --- Suite-wide DB guard (module-level baseline) ---
# Code that resolves a database outside any function-scoped fixture window —
# module- or session-scoped fixture setup, conftest import side effects — runs
# before the per-test _block_real_db fixture below has a chance to act.  This
# baseline covers those windows unconditionally.  Per-test monkeypatch.setenv
# calls still win because they happen later; _block_real_db refines this
# baseline to a per-test tmp_path for stronger isolation.
os.environ["SNORE_DB_PATH"] = str(_TEST_HOME / "guard.db")  # unconditional override


@pytest.fixture(autouse=True)
def reset_auth_config():
    """Reset the cached AppConfig between tests.

    Ensures each test starts with a fresh config derived from its own env.
    Tests that need multiuser mode use ``monkeypatch.setenv`` together with
    this fixture (always active) to get a clean multiuser config.
    """
    from snore.api.config import reset_config

    reset_config()
    yield
    reset_config()


@pytest.fixture(autouse=True)
def _reset_waveform_array_cache():
    """Clear the process-global deserialized-waveform array cache between tests.

    The cache in ``waveform_service`` is keyed by ``Waveform.id``.  Each test
    uses its own database, but row ids restart per DB, so without this reset a
    cached array from one test's row id would be served for a different test's
    row that happens to reuse that id — an isolation artifact that never occurs
    in production (one process serves one database).
    """
    from snore.services.waveform_service import clear_waveform_array_cache as _reset

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _disable_background_vacuum(monkeypatch):
    """Disable background VACUUM in all tests.

    FastAPI runs sync background tasks before the session's dep generator exits
    (i.e., before ``session_scope()``'s ``finally: await session.close()``).
    The aiosqlite connection pool therefore keeps a connection to the test's
    ``temp_db`` file open when ``_vacuum_background`` tries to open a pysqlite
    connection for VACUUM.

    The vacuum attempt itself is caught and logged as a warning — no test
    assertion fails because of it — but the transient pysqlite open creates
    OS-level file-lock contention that intermittently prevents the explicit
    ``vacuum_db`` endpoint test from running its own VACUUM on the same file
    under high xdist parallelism (18 workers).

    Making this a no-op removes the contention without affecting any assertion
    in the test suite.  Production paths exercise the real implementation.
    """
    from snore.api.routers import db as _db_router
    from snore.api.routers import me as _me_router
    from snore.services import database_service

    noop = lambda db_path: None  # noqa: E731
    monkeypatch.setattr(database_service, "_vacuum_background", noop)
    monkeypatch.setattr(_db_router, "_vacuum_background", noop)
    monkeypatch.setattr(_me_router, "_vacuum_background", noop)


@pytest.fixture(autouse=True)
def _block_real_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Refine the module-level DB guard to a per-test throwaway path.

    The module-level baseline (above) sets ``SNORE_DB_PATH`` at import time
    so code that runs before any function-scoped fixture (module/session-scoped
    fixture setup, conftest side effects) never resolves the real user database
    (~/.snore/snore.db).  This fixture tightens that to a fresh ``tmp_path``
    per test for stronger isolation between tests.

    Protected paths: ``DatabaseTarget.from_env_and_flags(db_flag=None)``
    (env chain: ``SNORE_DATABASE_URL`` > ``SNORE_DB_PATH`` > default) and the
    app lifespan's own env read in ``app.py``.

    Paths that ignore the env and use ``DEFAULT_DATABASE_PATH`` directly
    (``init_database(None)``, i.e. any CLI command run without ``--db``, and
    the ``snore db`` subcommands) are covered by the module-level test HOME
    instead: they land in a throwaway database, never the real one.

    Tests that set ``SNORE_DATABASE_URL`` via monkeypatch still win — it
    outranks ``SNORE_DB_PATH`` by precedence — so MCP module fixtures that
    write ``SNORE_DATABASE_URL`` directly are unaffected.
    """
    monkeypatch.setenv("SNORE_DB_PATH", str(tmp_path / "guard.db"))


def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that do not require external dependencies"
    )
    config.addinivalue_line("markers", "parser: Tests for device parsers")
    config.addinivalue_line(
        "markers", "integration: Integration tests combining multiple components"
    )
    config.addinivalue_line(
        "markers", "integration_pipeline: Full end-to-end pipeline integration tests"
    )
    config.addinivalue_line(
        "markers", "integration_features: Feature extraction integration tests"
    )
    config.addinivalue_line(
        "markers", "real_data: Tests that process actual CPAP session data"
    )
    config.addinivalue_line(
        "markers", "recorded: Tests using recorded PAP session data from device"
    )
    config.addinivalue_line(
        "markers", "requires_fixtures: Tests that require real session fixtures"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take significant time (>5 seconds)"
    )


def pytest_collection_modifyitems(items):
    """Auto-apply unit/integration markers based on test location."""
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def resmed_fixture_path(fixtures_dir):
    """Return path to ResMed test data."""
    return fixtures_dir / "device_data" / "resmed"


@pytest.fixture
def resmed_parser():
    """Return a ResMed EDF parser instance."""
    from snore.parsers.resmed_edf import ResmedEDFParser

    return ResmedEDFParser()


@pytest.fixture
def parser_registry():
    """Return the global parser registry with parsers registered."""
    from snore.parsers.register_all import ensure_registered_parsers
    from snore.parsers.registry import parser_registry

    ensure_registered_parsers()

    return parser_registry


# =============================================================================
# Database Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database for testing with a collision-proof name.

    Uses ``tmp_path`` (pytest's per-test isolated directory) plus a UUID so
    parallel xdist workers never share or collide on the same file — even when
    their clocks have the same millisecond timestamp.
    """
    import uuid

    db_path = tmp_path / f"test_snore_{uuid.uuid4().hex}.db"

    yield db_path

    if db_path.exists():
        db_path.unlink(missing_ok=True)
    for ext in ["-wal", "-shm"]:
        wal_file = Path(str(db_path) + ext)
        if wal_file.exists():
            wal_file.unlink(missing_ok=True)


@pytest.fixture
def db_session(temp_db):
    """Sync seed-only session for tests that need a synchronous ORM handle.

    Intentionally uses the sync ORM (``Session``) — this is an isolated
    test-data seed helper, NOT an application code path.  Application tests
    that exercise transaction semantics must use ``async_db_session`` or the
    full FastAPI lifespan.  Do NOT use this fixture for new application-facing
    tests.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from snore.database.models import Base

    engine = create_engine(f"sqlite:///{temp_db}")

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    engine.dispose()


@pytest.fixture
async def async_db_session(temp_db):
    """Create fresh async database session for each test.

    Used by tests for services that have been converted to AsyncSession in PR-2.
    The underlying database is the same temporary SQLite file as ``db_session``.
    """
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from snore.database.models import Base

    async_url = f"sqlite+aiosqlite:///{temp_db}"
    engine = create_async_engine(async_url, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, class_=AsyncSession
    )
    session = factory()

    try:
        yield session
    finally:
        await session.close()
        await engine.dispose()


@pytest.fixture
def test_profile(db_session):
    """Create a test User + Profile (sync). Returns the Profile.

    Required by any test that seeds a Device — devices.profile_id is NOT NULL
    after the multiuser schema migration.
    """
    import uuid

    from snore.database.models import Profile, User

    user = User(
        canonical_email=f"test_{uuid.uuid4().hex[:8]}@example.com",
        role="admin",
    )
    db_session.add(user)
    db_session.flush()

    profile = Profile(user_id=user.id, name="Test Profile")
    db_session.add(profile)
    db_session.flush()
    return profile


@pytest.fixture
async def async_test_profile(async_db_session):
    """Create a test User + Profile (async). Returns the Profile.

    Required by any test that seeds a Device — devices.profile_id is NOT NULL
    after the multiuser schema migration.
    """
    import uuid

    from snore.database.models import Profile, User

    user = User(
        canonical_email=f"test_{uuid.uuid4().hex[:8]}@example.com",
        role="admin",
    )
    async_db_session.add(user)
    await async_db_session.flush()

    profile = Profile(user_id=user.id, name="Test Profile")
    async_db_session.add(profile)
    await async_db_session.flush()
    return profile


@pytest.fixture
def test_device(db_session, test_profile):
    """Create a test device owned by test_profile."""
    import uuid

    from snore.database.models import Device

    device = Device(
        profile_id=test_profile.id,
        manufacturer="Test Manufacturer",
        model="Test Model",
        serial_number=f"TEST_{uuid.uuid4().hex[:8]}",
    )
    db_session.add(device)
    db_session.flush()
    return device


@pytest.fixture
async def async_test_device(async_db_session, async_test_profile):
    """Create a test device using the async session fixture."""
    import uuid

    from snore.database.models import Device

    device = Device(
        profile_id=async_test_profile.id,
        manufacturer="Test Manufacturer",
        model="Test Model",
        serial_number=f"TEST_{uuid.uuid4().hex[:8]}",
    )
    async_db_session.add(device)
    await async_db_session.flush()
    return device


@pytest.fixture
def test_session_factory(db_session):
    """Factory for creating test sessions with statistics."""
    import uuid

    from snore.database.models import Session, Statistics

    def _create_session(device_id, start_time, duration_hours=8.0, **stats_kwargs):
        session = Session(
            device_id=device_id,
            device_session_id=f"test_{start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
            start_time=start_time,
            end_time=start_time + timedelta(hours=duration_hours),
            duration_seconds=duration_hours * 3600,
            has_statistics=bool(stats_kwargs),
        )
        db_session.add(session)
        db_session.flush()

        if stats_kwargs:
            stats = Statistics(session_id=session.id, **stats_kwargs)
            db_session.add(stats)
            db_session.flush()
            db_session.refresh(session)

        return session

    return _create_session


@pytest.fixture
def async_test_session_factory(async_db_session):
    """Async factory for creating test sessions with statistics.

    Returns a coroutine factory — callers must await each call:
        session = await async_test_session_factory(device_id, start_time)
    """
    import uuid

    from snore.database.models import Session, Statistics

    async def _create_session(
        device_id, start_time, duration_hours=8.0, **stats_kwargs
    ):
        session = Session(
            device_id=device_id,
            device_session_id=f"test_{start_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}",
            start_time=start_time,
            end_time=start_time + timedelta(hours=duration_hours),
            duration_seconds=duration_hours * 3600,
            has_statistics=bool(stats_kwargs),
        )
        async_db_session.add(session)
        await async_db_session.flush()

        if stats_kwargs:
            stats = Statistics(session_id=session.id, **stats_kwargs)
            async_db_session.add(stats)
            await async_db_session.flush()
            await async_db_session.refresh(session)

        return session

    return _create_session


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture
def async_recorded_session(async_db_session):
    """Async factory for loading recorded session fixtures by YYYYMMDD ID.

    Usage:
        async def test_something(self, async_recorded_session):
            db, session = await async_recorded_session("20250808")
            # db is the AsyncSession, session is the CPAPSession ORM object

    Available sessions:
        - 20250110: Early therapy session (January 2025)
        - 20250808: Baseline session (August 2025)
        - 20250910: Multi-segment session (September 2025, 4 therapy segments)
        - 20251025: Event detection test session (October 2025)
    """
    from tests.helpers.fixtures_loader import async_import_to_test_db

    async def _load(session_id: str) -> Any:
        try:
            session = await async_import_to_test_db(session_id, async_db_session)
            return async_db_session, session
        except (ValueError, FileNotFoundError) as e:
            pytest.skip(f"Fixture {session_id} not available: {e}")

    return _load
