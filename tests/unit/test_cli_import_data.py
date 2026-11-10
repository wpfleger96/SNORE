"""Unit tests for import-command profile lookup and dry-run display."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import click
import pytest

from click.testing import CliRunner

from snore.cli.commands.import_data import _resolve_profile_timezone, import_data
from snore.cli.decorators import CliCtx
from snore.database.models import Profile
from snore.services.schemas import ImportSource


@pytest.mark.unit
class TestResolveProfileTimezone:
    """Best-effort timezone resolution used by import dry runs."""

    async def test_resolution_failure_returns_none(self):
        """A missing actor profile falls back without querying the profile row."""

        @asynccontextmanager
        async def failed_profile_session(*args: Any, **kwargs: Any) -> Any:
            raise click.ClickException("no such profile")
            yield  # pragma: no cover

        with patch(
            "snore.cli.commands.import_data.profile_session",
            failed_profile_session,
        ):
            timezone = await _resolve_profile_timezone(
                "/tmp/snore.db", "user@example.com", "nightly"
            )

        assert timezone is None

    async def test_success_reads_timezone_from_resolved_profile(self):
        """Database and actor selectors reach the shared resolved session."""
        session = MagicMock()
        session.get = AsyncMock(
            return_value=SimpleNamespace(timezone="America/New_York")
        )
        received: list[tuple[str | None, str | None, str | None]] = []

        @asynccontextmanager
        async def tracked_profile_session(
            db: str | None,
            actor_user: str | None,
            actor_profile: str | None,
        ) -> Any:
            received.append((db, actor_user, actor_profile))
            yield CliCtx(db=session, profile_id=23)

        with patch(
            "snore.cli.commands.import_data.profile_session",
            tracked_profile_session,
        ):
            timezone = await _resolve_profile_timezone(
                "/tmp/snore.db", "user@example.com", "nightly"
            )

        assert timezone == "America/New_York"
        assert received == [("/tmp/snore.db", "user@example.com", "nightly")]
        session.get.assert_awaited_once_with(Profile, 23)


@pytest.mark.unit
class TestDryRunSummary:
    """The dry-run table and summary show night dates, as the range selects."""

    def test_after_midnight_session_shows_its_night_date(self, tmp_path):
        """Sessions at 22:00 on 06-20 and 01:34 on 06-21 are both the 06-20 night."""
        sessions = [
            SimpleNamespace(
                start_time=start, duration_seconds=3600, events=[], statistics=None
            )
            for start in (datetime(2024, 6, 20, 22, 0), datetime(2024, 6, 21, 1, 34))
        ]
        parser = MagicMock(parser_id="resmed", manufacturer="ResMed")
        parser.parse_sessions.return_value = sessions
        service = MagicMock()
        service.detect_sources.return_value = [
            ImportSource(parser_name="resmed", root_path=str(tmp_path))
        ]

        with (
            patch("snore.cli.commands.import_data.init_db"),
            patch("snore.cli.commands.import_data.ImportService", return_value=service),
            patch(
                "snore.cli.commands.import_data.parser_registry.list_parsers",
                return_value=[parser],
            ),
            patch(
                "snore.cli.commands.import_data._resolve_profile_timezone",
                AsyncMock(return_value=None),
            ),
        ):
            result = CliRunner().invoke(import_data, [str(tmp_path), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Nights: 2024-06-20 to 2024-06-20" in result.output
        assert "2024-06-21" not in result.output
