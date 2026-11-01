"""Shared fixtures for all unit tests.

Includes:
- import job store reset (autouse) — safe no-op for tests that never touch
  the store; MCP tests run against an always-empty store with no overhead.
- MCP roundtrip fixtures (mcp_client_factory, mock_db_session) — provide an
  in-memory fastmcp.Client environment with the lifespan patched to a
  lightweight mock (no real database or filesystem access required).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import fastmcp
import pytest

import snore.api.import_jobs as _import_job_store

from snore.mcp.server import StaticRuntime


@pytest.fixture(autouse=True)
def reset_import_job_store():
    """Stop any running import worker and reset all import job globals before/after each test.

    The import worker is module-global.  Any test that starts one (via the
    ``import_worker`` fixture) or that accidentally triggers a worker-start must
    have the state cleaned up so subsequent tests start fresh.

    Clearing empty dicts is cheap and side-effect-free, so this fixture is safe
    to apply to all unit tests even those that never touch the import job store.
    """
    _import_job_store.stop_import_worker()
    _import_job_store._jobs.clear()
    _import_job_store._per_user_count.clear()
    _import_job_store._global_count = 0
    _import_job_store._import_queue.clear()
    yield
    _import_job_store.stop_import_worker(timeout=5.0)
    _import_job_store._jobs.clear()
    _import_job_store._per_user_count.clear()
    _import_job_store._global_count = 0
    _import_job_store._import_queue.clear()


@pytest.fixture
def mock_db_session() -> MagicMock:
    """Return a minimal AsyncSession mock that satisfies scalar queries.

    The mock models an empty database: every scalar query returns 0 or None,
    every collection query returns an empty list.  Tests that need specific
    return values should override ``session.execute`` with their own
    ``AsyncMock(side_effect=[...])`` after receiving this fixture.
    """
    session = MagicMock()
    result_mock = MagicMock()
    result_mock.scalar_one.return_value = 0
    result_mock.scalar_one_or_none.return_value = None
    result_mock.scalars.return_value.all.return_value = []
    result_mock.scalars.return_value.first.return_value = None
    result_mock.one.return_value = (0, None, None)
    result_mock.all.return_value = []
    session.execute = AsyncMock(return_value=result_mock)
    return session


@pytest.fixture
def mcp_client_factory() -> Any:
    """Return a factory that yields a connected fastmcp.Client for in-memory tool tests.

    The factory signature is::

        async def factory(
            session: MagicMock,
            extra_patches: list | None = None,
        ) -> AsyncContextManager[fastmcp.Client]

    Usage::

        async def test_something(mcp_client_factory, mock_db_session):
            async with mcp_client_factory(mock_db_session) as client:
                result = await client.call_tool("get_data_overview", {})

    ``_lifespan`` is replaced with a stub that yields a ``SNORERuntime``
    backed by ``session`` (profile_id=1).  Any additional
    ``unittest.mock.patch`` context managers can be passed via
    ``extra_patches`` and are entered alongside the lifespan patch.

    Important: the returned factory is an async context manager.  Use it with
    ``async with``, not as a regular call.
    """
    from snore.mcp.server import make_server  # noqa: PLC0415

    # make_server() imports the tool modules lazily. Import them here, before a
    # test's patch() blocks start, or a module first imported under a patch keeps
    # the mock in its from-imports for the rest of the worker.
    make_server()

    @asynccontextmanager
    async def _client_context(
        session: MagicMock,
        extra_patches: list[Any] | None = None,
    ) -> AsyncIterator[fastmcp.Client]:
        @asynccontextmanager
        async def _mock_scope(s: MagicMock = session) -> Any:  # noqa: RUF029
            yield s

        @asynccontextmanager
        async def _fake_lifespan(
            app: Any,
            db_flag: str | None = None,
            profile_name: str = "neutral",
            *,
            actor_scoped: bool = False,
            manage_database: bool = True,
        ) -> Any:  # noqa: RUF029
            yield StaticRuntime(
                base_scope_provider=lambda: _mock_scope(session),
                profile_id=1,
            )

        mcp = make_server()
        with patch("snore.mcp.server._lifespan", _fake_lifespan):
            with ExitStack() as stack:
                for p in extra_patches or []:
                    stack.enter_context(p)
                async with fastmcp.Client(mcp) as client:
                    yield client

    return _client_context
