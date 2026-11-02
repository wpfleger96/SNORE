from contextlib import asynccontextmanager

import pytest

import snore.api.import_jobs as _import_job_store
import snore.api.reset_jobs as _reset_jobs
import snore.auth.lockout as _lockout

from tests.helpers.api_client import make_test_client
from tests.helpers.reset_jobs import clear_reset_jobs

# ---------------------------------------------------------------------------
# Shared multiuser test helpers — imported by test_auth.py and test_me.py
# ---------------------------------------------------------------------------

_SESSION_SECRET = "test-secret-at-least-32-chars-long-abcdef"


def _multiuser_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the env vars required for a minimal multiuser config."""
    monkeypatch.setenv("SNORE_AUTH_MODE", "multiuser")
    monkeypatch.setenv("SNORE_SESSION_SECRET", _SESSION_SECRET)
    monkeypatch.setenv("SNORE_PUBLIC_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture(autouse=True)
def reset_import_job_store():
    """Stop any running import worker and reset all import job globals before/after each test.

    The import worker is module-global.  Any test that starts one (via the
    ``import_worker`` fixture) or that accidentally triggers a worker-start must
    have the state cleaned up so subsequent tests start fresh.
    """
    # Stop any worker left over from a previous test BEFORE clearing globals.
    # Nulling without stopping first lets a still-running thread become a zombie.
    _import_job_store.stop_import_worker()
    _import_job_store._jobs.clear()
    _import_job_store._per_user_count.clear()
    _import_job_store._global_count = 0
    _import_job_store._import_queue.clear()
    yield
    # Teardown after the test (5 s to survive slow CI).
    _import_job_store.stop_import_worker(timeout=5.0)
    _import_job_store._jobs.clear()
    _import_job_store._per_user_count.clear()
    _import_job_store._global_count = 0
    _import_job_store._import_queue.clear()


@pytest.fixture(autouse=True)
def reset_reset_job_store():
    """Empty the module-global reset job store around each test.

    A leftover non-terminal job would make every later POST in this worker 409.
    """
    clear_reset_jobs()
    yield
    clear_reset_jobs()


@pytest.fixture
def reset_job_session(async_db_session, monkeypatch):
    """Point the reset job at this test's database.

    The job opens ``session_scope`` itself rather than taking a request session,
    so ``make_test_client``'s ``get_db`` override never reaches it.
    """

    @asynccontextmanager
    async def _scope(*, immediate=False):
        async with async_db_session.begin():
            yield async_db_session

    monkeypatch.setattr(_reset_jobs, "session_scope", _scope)


@pytest.fixture(autouse=True)
def reset_auth_throttle_stores():
    """Clear the module-global rate-limit and lockout stores before each test.

    All three are per-process singletons keyed by client IP (or email + IP),
    and every test request presents the same client IP — so auth-endpoint
    traffic from earlier tests in the same worker would otherwise count
    against later tests' sliding windows and lock-out counters.
    """
    _lockout.get_rate_limit_store()._entries.clear()
    _lockout.get_lockout_store()._entries.clear()
    _lockout.get_invite_lockout_store()._entries.clear()


@pytest.fixture(autouse=True)
def _hermetic_raw_root(monkeypatch, tmp_path):
    """Redirect DEFAULT_RAW_BACKUP_DIR to a tmp_path subdir for every API test.

    The reset job's _raw_root() imports the constant at call time, so patching
    snore.constants keeps every code path in any API test away from the real
    ~/.snore/raw/ directory.
    """
    fake_raw = tmp_path / "raw"
    monkeypatch.setattr("snore.constants.DEFAULT_RAW_BACKUP_DIR", fake_raw)


@pytest.fixture
def import_worker():
    """Start the import worker with the real _run_import callback for this test.

    Use this fixture in tests that exercise the full upload-then-execute flow
    (SSE progress streaming, backup=True verification, etc.).  The
    ``reset_import_job_store`` autouse fixture handles teardown.
    """
    from snore.api.import_jobs import start_import_worker  # noqa: PLC0415
    from snore.api.import_worker import _run_import  # noqa: PLC0415

    start_import_worker(_run_import)


@pytest.fixture
def db_session(temp_db, async_db_session):
    """Sync seed-only session for seeding test data.

    Intentionally uses a sync AUTOCOMMIT session — this is a test-data seed
    helper, NOT an application code path.  Every ``flush()`` is immediately
    visible to the ``async_db_session`` used by the API override below.

    Do NOT use this fixture for new application-facing tests; use the async
    fixtures or ``real_app`` with ``httpx.AsyncClient`` instead.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from snore.database.models import Base

    engine = create_engine(
        f"sqlite:///{temp_db}",
        connect_args={"check_same_thread": False},
        isolation_level="AUTOCOMMIT",
    )
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    session.close()
    engine.dispose()


@pytest.fixture
def api_client(temp_db, async_db_session, db_session):
    """TestClient with get_db and get_actor overridden for test isolation.

    Seeds are injected via the AUTOCOMMIT ``db_session`` so they are
    immediately visible inside the override session.
    """
    # Do NOT use 'with TestClient(app)' — that runs lifespan (init_database),
    # which we skip since we are overriding get_db entirely; lifespan tests use
    # real_app instead.
    # See make_test_client()'s docstring for the FastAPI dep-cache rationale
    # behind the Depends(get_db) declaration in the actor override.
    return make_test_client(async_db_session)


@pytest.fixture
def localhost_api_client(temp_db, async_db_session, db_session):
    """TestClient that appears to connect from 127.0.0.1 (for localhost-only endpoints)."""
    return make_test_client(async_db_session, localhost=True)
