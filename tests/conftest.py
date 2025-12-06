"""Pytest configuration and fixtures for SNORE tests."""

import sqlite3
import tempfile

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

# Register datetime adapters for SQLite (Python 3.12+)
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("DATETIME", lambda s: datetime.fromisoformat(s.decode()))


def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that do not require external dependencies"
    )
    config.addinivalue_line("markers", "parser: Tests for device parsers")
    config.addinivalue_line(
        "markers", "business_logic: Tests for core business logic and algorithms"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests combining multiple components"
    )
    config.addinivalue_line(
        "markers", "integration_pipeline: Full end-to-end pipeline integration tests"
    )
    config.addinivalue_line(
        "markers", "integration_metrics: Breath metrics calculation and validation"
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
    from snore.parsers.register_all import register_all_parsers
    from snore.parsers.registry import parser_registry

    # Explicitly register parsers for testing
    register_all_parsers()

    return parser_registry


# =============================================================================
# Database Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = Path(tempfile.gettempdir())
    db_path = temp_dir / f"test_oscar_{datetime.now().timestamp()}.db"

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()
    # Also clean up WAL files if they exist
    for ext in ["-wal", "-shm"]:
        wal_file = Path(str(db_path) + ext)
        if wal_file.exists():
            wal_file.unlink()


@pytest.fixture
def db_session(temp_db):
    """Create fresh database session for each test with proper isolation."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from snore.database.models import Base

    # Create engine with SQLite
    engine = create_engine(f"sqlite:///{temp_db}")

    # Create all tables fresh for this test
    Base.metadata.create_all(engine)

    # Create session factory
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    # Cleanup
    session.close()
    engine.dispose()


@pytest.fixture
def initialized_db(temp_db):
    """Create database initialized with global session factory (for validation tests)."""
    from snore.database.session import (
        cleanup_database,
        init_database,
        session_scope,
    )

    # Initialize database with global session factory
    init_database(str(temp_db))

    # Yield the session scope context manager
    with session_scope() as session:
        yield session

    # Cleanup: Dispose engine and reset global session factory
    cleanup_database()


@pytest.fixture
def test_profile_factory(db_session):
    """Factory for creating test profiles with auto-generated unique usernames."""
    import uuid

    from snore.database.models import Profile

    def _create_profile(username=None, day_split_time="12:00:00", **kwargs):
        if username is None:
            username = f"test_user_{uuid.uuid4().hex[:8]}"

        profile = Profile(
            username=username, settings={"day_split_time": day_split_time}, **kwargs
        )
        db_session.add(profile)
        db_session.flush()
        return profile

    return _create_profile


@pytest.fixture
def test_device(db_session, test_profile_factory):
    """Create a test device linked to a test profile."""
    import uuid

    from snore.database.models import Device

    profile = test_profile_factory()  # Will auto-generate unique username
    device = Device(
        profile_id=profile.id,
        manufacturer="Test Manufacturer",
        model="Test Model",
        serial_number=f"TEST_{uuid.uuid4().hex[:8]}",
    )
    db_session.add(device)
    db_session.flush()
    return device, profile


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

        # Add statistics if provided
        if stats_kwargs:
            stats = Statistics(session_id=session.id, **stats_kwargs)
            db_session.add(stats)
            db_session.flush()
            # Important: refresh session to load the relationship
            db_session.refresh(session)

        return session

    return _create_session


# =============================================================================
# Integration Test Fixtures (Real Session Data)
# =============================================================================


@pytest.fixture
def recorded_session(db_session):
    """Factory for loading recorded session fixtures by YYYYMMDD ID.

    Usage:
        def test_something(self, recorded_session):
            db = recorded_session("20250808")
            session = db.query(Session).first()

    Available sessions:
        - 20250110: Early therapy session (January 2025)
        - 20250808: Baseline session (August 2025)
        - 20250910: Multi-segment session (September 2025, 4 therapy segments)
        - 20251025: Event detection test session (October 2025)
    """
    from tests.helpers.fixtures_loader import import_to_test_db

    def _load(session_id: str) -> Any:
        try:
            import_to_test_db(session_id, db_session)
            return db_session
        except (ValueError, FileNotFoundError) as e:
            pytest.skip(f"Fixture {session_id} not available: {e}")

    return _load
