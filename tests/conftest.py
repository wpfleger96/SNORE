"""Pytest configuration and fixtures for OSCAR-MCP tests."""

from pathlib import Path
import tempfile
from datetime import datetime, timedelta
import pytest


def pytest_configure(config):
    """Register custom test markers."""
    config.addinivalue_line("markers", "unit: Unit tests that do not require external dependencies")
    config.addinivalue_line("markers", "parser: Tests for device parsers")
    config.addinivalue_line(
        "markers", "business_logic: Tests for core business logic and algorithms"
    )


@pytest.fixture
def fixtures_dir():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def resmed_fixture_path(fixtures_dir):
    """Return path to ResMed test data."""
    return fixtures_dir / "resmed_sample"


@pytest.fixture
def resmed_parser():
    """Return a ResMed EDF parser instance."""
    from oscar_mcp.parsers.resmed_edf import ResmedEDFParser

    return ResmedEDFParser()


@pytest.fixture
def parser_registry():
    """Return the global parser registry with parsers registered."""
    from oscar_mcp.parsers.registry import parser_registry
    from oscar_mcp.parsers.register_all import register_all_parsers

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
    from oscar_mcp.database.models import Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

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
    from oscar_mcp.database.session import init_database, session_scope
    import oscar_mcp.database.session as session_module

    # Initialize database with global session factory
    init_database(str(temp_db))

    # Yield the session scope context manager
    with session_scope() as session:
        yield session

    # Cleanup: Reset global session factory to allow next test to use different database
    session_module._SessionFactory = None


@pytest.fixture
def test_profile_factory(db_session):
    """Factory for creating test profiles with auto-generated unique usernames."""
    import uuid
    from oscar_mcp.database.models import Profile

    def _create_profile(username=None, day_split_time="12:00:00", **kwargs):
        if username is None:
            username = f"test_user_{uuid.uuid4().hex[:8]}"

        profile = Profile(username=username, settings={"day_split_time": day_split_time}, **kwargs)
        db_session.add(profile)
        db_session.flush()
        return profile

    return _create_profile


@pytest.fixture
def test_device(db_session, test_profile_factory):
    """Create a test device linked to a test profile."""
    import uuid
    from oscar_mcp.database.models import Device

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
    from oscar_mcp.database.models import Session, Statistics

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
