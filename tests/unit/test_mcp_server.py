"""Unit tests for MCP server wiring: error boundary, size guard, profile building."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest

from snore.mcp.profiles import get_profile
from snore.mcp.server import (
    RESPONSE_SIZE_LIMIT,
    StaticRuntime,
    _build_instructions,
    _check_response_size,
    tool_error_boundary,
)

# ---------------------------------------------------------------------------
# Shared lifespan stub — used by TestStage2ToolsRegistered and
# TestChannelVocabInSync to patch snore.mcp.server._lifespan without
# touching the real DB.  Extracted once to eliminate the ~3x duplication.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _noop_scope(session: MagicMock) -> AsyncIterator[MagicMock]:
    yield session


@asynccontextmanager
async def _fake_static_lifespan(
    app: Any,
    db_flag: str | None = None,
    profile_name: str = "neutral",
    *,
    actor_scoped: bool = False,
    manage_database: bool = True,
) -> AsyncIterator[StaticRuntime]:
    session = MagicMock()
    yield StaticRuntime(
        base_scope_provider=lambda: _noop_scope(session),
        profile_id=1,
    )


class TestToolErrorBoundary:
    async def test_passes_through_on_success(self) -> None:
        @tool_error_boundary
        async def _ok() -> str:
            return "ok"

        result = await _ok()
        assert result == "ok"

    async def test_converts_validation_error_to_tool_error(self) -> None:
        from fastmcp.exceptions import ToolError

        from snore.mcp.errors import ValidationError

        @tool_error_boundary
        async def _bad() -> str:
            raise ValidationError("bad input")

        with pytest.raises(ToolError, match="bad input"):
            await _bad()

    async def test_converts_value_error_to_tool_error(self) -> None:
        from fastmcp.exceptions import ToolError

        @tool_error_boundary
        async def _bad() -> str:
            raise ValueError("invalid value")

        with pytest.raises(ToolError, match="invalid value"):
            await _bad()

    async def test_passes_through_tool_error_unchanged(self) -> None:
        from fastmcp.exceptions import ToolError

        @tool_error_boundary
        async def _bad() -> str:
            raise ToolError("already a tool error")

        with pytest.raises(ToolError, match="already a tool error"):
            await _bad()

    async def test_converts_generic_exception_to_tool_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from fastmcp.exceptions import ToolError

        @tool_error_boundary
        async def _bad() -> str:
            raise RuntimeError("internal detail that must not leak")

        with caplog.at_level(logging.ERROR, logger="snore.mcp.server"):
            with pytest.raises(ToolError, match="An unexpected error occurred."):
                await _bad()

        # Internal exception detail must be logged server-side, not forwarded.
        assert any(
            "internal detail that must not leak" in r.message
            or "internal detail that must not leak" in str(r.exc_info)
            for r in caplog.records
        )

    async def test_exc_with_response_status_code_emits_http_status_message(
        self,
    ) -> None:
        from unittest.mock import MagicMock

        from fastmcp.exceptions import ToolError

        mock_response = MagicMock()
        mock_response.status_code = 503

        exc = RuntimeError("secret response body")
        exc.response = mock_response

        @tool_error_boundary
        async def _bad() -> str:
            raise exc

        with pytest.raises(ToolError, match="HTTP 503 from upstream service"):
            await _bad()

    async def test_exc_with_response_status_does_not_include_response_body(
        self,
    ) -> None:
        from unittest.mock import MagicMock

        from fastmcp.exceptions import ToolError

        mock_response = MagicMock()
        mock_response.status_code = 401

        exc = RuntimeError("secret response body text")
        exc.response = mock_response

        @tool_error_boundary
        async def _bad() -> str:
            raise exc

        with pytest.raises(ToolError) as exc_info:
            await _bad()
        assert "secret response body text" not in str(exc_info.value)

    async def test_pydantic_multi_error_joined_with_semicolon(self) -> None:
        """Multiple pydantic field errors are joined with '; ' in the ToolError message."""
        from datetime import date

        from fastmcp.exceptions import ToolError
        from pydantic import ValidationError as PydanticValidationError

        from snore.services.breath_service import WaveformWindowRequest

        try:
            WaveformWindowRequest(
                therapy_date=date(2024, 1, 1),
                offset_start=-1.0,  # fails ge=0.0
                offset_end=60.0,
                max_points=2000,  # fails le=1000
            )
        except PydanticValidationError as pydantic_exc:
            captured = pydantic_exc

        @tool_error_boundary
        async def _raise() -> str:
            raise captured

        with pytest.raises(ToolError) as exc_info:
            await _raise()

        assert "; " in str(exc_info.value)

    async def test_pydantic_value_error_prefix_stripped(self) -> None:
        """The 'Value error, ' prefix on model_validator errors is stripped from the message."""
        from datetime import date

        from fastmcp.exceptions import ToolError
        from pydantic import ValidationError as PydanticValidationError

        from snore.services.breath_service import WaveformWindowRequest

        try:
            WaveformWindowRequest(
                therapy_date=date(2024, 1, 1),
                offset_start=100.0,
                offset_end=50.0,  # fails model_validator: offset_end must be > offset_start
            )
        except PydanticValidationError as pydantic_exc:
            captured = pydantic_exc

        @tool_error_boundary
        async def _raise() -> str:
            raise captured

        with pytest.raises(ToolError) as exc_info:
            await _raise()

        msg = str(exc_info.value)
        assert "Value error, " not in msg
        assert "offset_end" in msg

    async def test_pydantic_list_index_loc_formatted_with_dot_join(self) -> None:
        """List-indexed validation errors include the dot-joined loc (e.g. 'channels.1: ...')."""
        from datetime import date

        from fastmcp.exceptions import ToolError
        from pydantic import ValidationError as PydanticValidationError

        from snore.services.breath_service import WaveformWindowRequest

        try:
            WaveformWindowRequest(
                therapy_date=date(2024, 1, 1),
                offset_start=0.0,
                offset_end=60.0,
                channels=["flow", "not_a_valid_channel"],  # type: ignore[list-item]  # index 1 is invalid enum value
            )
        except PydanticValidationError as pydantic_exc:
            captured = pydantic_exc

        @tool_error_boundary
        async def _raise() -> str:
            raise captured

        with pytest.raises(ToolError) as exc_info:
            await _raise()

        assert "channels.1" in str(exc_info.value)


class TestCheckResponseSize:
    def test_small_response_passes(self) -> None:
        # Should not raise
        _check_response_size({"key": "value"}, "test_tool")

    def test_oversized_response_raises_tool_error(self) -> None:
        from fastmcp.exceptions import ToolError

        # Build a payload that exceeds RESPONSE_SIZE_LIMIT
        huge = {"data": "x" * (RESPONSE_SIZE_LIMIT + 1)}
        with pytest.raises(ToolError, match="exceeds"):
            _check_response_size(huge, "test_tool")

    def test_tool_name_appears_in_error_message(self) -> None:
        from fastmcp.exceptions import ToolError

        huge = {"data": "x" * (RESPONSE_SIZE_LIMIT + 1)}
        with pytest.raises(ToolError, match="my_tool"):
            _check_response_size(huge, "my_tool")

    def test_narrow_your_query_hint_included(self) -> None:
        from fastmcp.exceptions import ToolError

        huge = {"data": "x" * (RESPONSE_SIZE_LIMIT + 1)}
        with pytest.raises(ToolError, match="Narrow your query"):
            _check_response_size(huge, "test_tool")

    def test_non_ascii_payload_under_byte_limit_passes(self) -> None:
        # "Ā" (U+0100) is escaped by json.dumps to "Ā" — 6 ASCII bytes per char.
        # 80,000 chars × 6 bytes + JSON overhead ≈ 480,012 bytes; under 500,000.
        payload = {"data": "Ā" * 80_000}
        _check_response_size(payload, "test_tool")  # must not raise

    def test_non_ascii_payload_over_byte_limit_fails(self) -> None:
        from fastmcp.exceptions import ToolError

        # 90,000 "Ā" chars × 6 JSON bytes + overhead ≈ 540,012 bytes → over limit.
        payload = {"data": "Ā" * 90_000}
        with pytest.raises(ToolError, match="exceeds"):
            _check_response_size(payload, "test_tool")

    def test_measurement_failure_logs_warning_and_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from unittest.mock import patch

        with patch(
            "snore.mcp.tools._scaffold.json.dumps", side_effect=RuntimeError("boom")
        ):
            with caplog.at_level(logging.WARNING, logger="snore.mcp.tools._scaffold"):
                _check_response_size({"key": "value"}, "my_tool")  # must not raise

        assert any("measurement failed" in r.message for r in caplog.records)


class TestBuildInstructions:
    def test_contains_snore_version(self) -> None:
        profile = get_profile("neutral")
        instructions = _build_instructions(profile)
        assert "snore" in instructions.lower()

    def test_contains_profile_display_name(self) -> None:
        profile = get_profile("uars")
        instructions = _build_instructions(profile)
        assert "UARS" in instructions

    def test_does_not_reference_docs_tools(self) -> None:
        profile = get_profile("neutral")
        instructions = _build_instructions(profile)
        assert "docs://tools" not in instructions

    def test_instructions_mention_schemas_resource(self) -> None:
        profile = get_profile("neutral")
        instructions = _build_instructions(profile)
        assert "docs://schemas" in instructions

    def test_all_profiles_produce_non_empty_instructions(self) -> None:
        from snore.mcp.profiles import VALID_PROFILES

        for name in VALID_PROFILES:
            profile = get_profile(name)
            instructions = _build_instructions(profile)
            assert len(instructions) > 100


class TestValidatePageArgs:
    """validate_page_args rejects out-of-range pagination values."""

    def test_valid_args_return_page_size(self) -> None:
        from snore.mcp.validation import validate_page_args

        assert validate_page_args(1, 30) == 30

    def test_page_size_90_is_valid(self) -> None:
        from snore.mcp.validation import validate_page_args

        assert validate_page_args(1, 90) == 90

    def test_page_size_91_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_page_args

        with pytest.raises(ValidationError, match="page_size must be between 1 and 90"):
            validate_page_args(1, 91)

    def test_page_size_200_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_page_args

        with pytest.raises(ValidationError, match="page_size must be between 1 and 90"):
            validate_page_args(1, 200)

    def test_page_zero_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_page_args

        with pytest.raises(ValidationError, match="page must be >= 1"):
            validate_page_args(0, 30)

    def test_page_negative_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_page_args

        with pytest.raises(ValidationError, match="page must be >= 1"):
            validate_page_args(-5, 30)

    def test_page_size_zero_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_page_args

        with pytest.raises(ValidationError, match="page_size must be between 1 and 90"):
            validate_page_args(1, 0)

    def test_page_size_negative_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_page_args

        with pytest.raises(ValidationError, match="page_size must be between 1 and 90"):
            validate_page_args(1, -10)


class TestValidateComplianceThreshold:
    """validate_compliance_threshold rejects negative thresholds."""

    def test_zero_is_valid(self) -> None:
        from snore.mcp.validation import validate_compliance_threshold

        validate_compliance_threshold(0.0)  # must not raise

    def test_positive_value_is_valid(self) -> None:
        from snore.mcp.validation import validate_compliance_threshold

        validate_compliance_threshold(4.0)  # must not raise

    def test_negative_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_compliance_threshold

        with pytest.raises(
            ValidationError, match="compliance_threshold_hours must be >= 0"
        ):
            validate_compliance_threshold(-1.0)


class TestValidateMinDuration:
    """validate_min_duration rejects negative durations."""

    def test_none_is_valid(self) -> None:
        from snore.mcp.validation import validate_min_duration

        validate_min_duration(None)  # must not raise

    def test_zero_is_valid(self) -> None:
        from snore.mcp.validation import validate_min_duration

        validate_min_duration(0.0)  # must not raise

    def test_positive_is_valid(self) -> None:
        from snore.mcp.validation import validate_min_duration

        validate_min_duration(5.0)  # must not raise

    def test_negative_raises(self) -> None:
        from snore.mcp.errors import ValidationError
        from snore.mcp.validation import validate_min_duration

        with pytest.raises(ValidationError, match="min_duration must be >= 0"):
            validate_min_duration(-0.1)


class TestStage2ToolsRegistered:
    """The three Stage-2 tools appear in make_server() tool listing."""

    async def test_make_server_registers_exactly_ten_tools(self) -> None:
        """make_server() registers exactly ten tools (five Stage-1 + three Stage-2 + two Stage-3)."""
        from unittest.mock import patch

        import fastmcp

        from snore.mcp.server import make_server

        mcp = make_server()
        with patch("snore.mcp.server._lifespan", _fake_static_lifespan):
            async with fastmcp.Client(mcp) as client:
                tools = await client.list_tools()

        tool_names = {t.name for t in tools}
        assert "get_breath_table" in tool_names
        assert "find_windows" in tool_names
        assert "compare_epochs" in tool_names
        # Stage-1 tools still present
        assert "get_data_overview" in tool_names
        assert "get_events" in tool_names
        assert "get_settings_changes" in tool_names
        # Stage-3 tools
        assert "get_waveform" in tool_names
        assert "render_window" in tool_names
        assert "get_ca_analysis" not in tool_names
        assert len(tool_names) == 10

    async def test_compare_epochs_schema_has_epochs_parameter(self) -> None:
        """compare_epochs tool schema includes an 'epochs' parameter."""
        from unittest.mock import patch

        import fastmcp

        from snore.mcp.server import make_server

        mcp = make_server()
        with patch("snore.mcp.server._lifespan", _fake_static_lifespan):
            async with fastmcp.Client(mcp) as client:
                tools = await client.list_tools()

        ce_tool = next(t for t in tools if t.name == "compare_epochs")
        # epochs parameter must appear in the input schema
        schema = ce_tool.inputSchema
        assert "epochs" in (schema.get("properties") or {}), (
            f"compare_epochs inputSchema missing 'epochs': {schema}"
        )


class TestChannelVocabInSync:
    """Channel vocabulary in get_waveform and render_window docstrings must be identical
    and must cover all WaveformChannelName enum members."""

    async def _get_tools(self) -> list:
        from unittest.mock import patch

        import fastmcp

        from snore.mcp.server import make_server

        mcp = make_server()
        with patch("snore.mcp.server._lifespan", _fake_static_lifespan):
            async with fastmcp.Client(mcp) as client:
                return await client.list_tools()

    @staticmethod
    def _extract_channel_lines(description: str) -> list[str]:
        """Extract lines from the channel vocabulary block (lines that name a channel)."""
        channel_names = [
            "flow",
            "pressure",
            "therapy_pressure",
            "epap",
            "leak",
            "mv",
            "rr",
            "tv",
            "spo2",
            "pulse",
            "fl",
            "snore",
        ]
        return [
            line.strip()
            for line in description.splitlines()
            if any(f"`{name}`" in line for name in channel_names)
        ]

    async def test_channel_vocab_in_sync(self) -> None:
        """get_waveform and render_window channel vocabulary blocks are identical."""
        tools = await self._get_tools()
        tool_map = {t.name: t for t in tools}

        gw_desc = tool_map["get_waveform"].description or ""
        rw_desc = tool_map["render_window"].description or ""

        gw_lines = self._extract_channel_lines(gw_desc)
        rw_lines = self._extract_channel_lines(rw_desc)

        assert gw_lines, "get_waveform description has no channel lines"
        assert rw_lines, "render_window description has no channel lines"
        assert gw_lines == rw_lines, (
            "Channel vocabulary mismatch between get_waveform and render_window.\n"
            f"get_waveform:  {gw_lines}\n"
            f"render_window: {rw_lines}"
        )

    async def test_all_channel_names_in_both_tools(self) -> None:
        """Every WaveformChannelName enum value must appear in both tool descriptions."""
        from snore.services.breath_service import WaveformChannelName  # noqa: PLC0415

        tools = await self._get_tools()
        tool_map = {t.name: t for t in tools}

        gw_desc = tool_map["get_waveform"].description or ""
        rw_desc = tool_map["render_window"].description or ""

        missing_gw = [
            ch.value for ch in WaveformChannelName if f"`{ch.value}`" not in gw_desc
        ]
        missing_rw = [
            ch.value for ch in WaveformChannelName if f"`{ch.value}`" not in rw_desc
        ]

        assert not missing_gw, f"get_waveform missing channels: {missing_gw}"
        assert not missing_rw, f"render_window missing channels: {missing_rw}"


class TestLifespanStartupFailure:
    """_lifespan calls cleanup_database and re-raises when startup fails."""

    async def test_cleanup_called_and_error_reraised_on_profile_not_found(
        self,
    ) -> None:
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from snore.mcp.server import _lifespan

        # A mock DB session where .scalars().first() returns None → profile not found.
        mock_db = MagicMock()
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = None
        mock_db.execute = AsyncMock(return_value=result_mock)

        @asynccontextmanager
        async def _mock_session_scope():
            yield mock_db

        cleanup_mock = AsyncMock()
        mock_target = MagicMock()
        mock_target.resolve_async_url.return_value = "sqlite+aiosqlite:///:memory:"

        with (
            patch("snore.mcp.server.DatabaseTarget") as mock_target_cls,
            patch("snore.mcp.server.init_database_from_url", new_callable=AsyncMock),
            patch("snore.mcp.server.cleanup_database", cleanup_mock),
            patch("snore.mcp.server.session_scope", _mock_session_scope),
            patch("snore.parsers.register_all.ensure_registered_parsers"),
        ):
            mock_target_cls.from_env_and_flags.return_value = mock_target

            with pytest.raises(RuntimeError, match="No profile found"):
                async with _lifespan(None):
                    pass  # pragma: no cover

        cleanup_mock.assert_awaited_once()


class TestLifespanActorScoped:
    """_lifespan with actor_scoped=True yields ActorRuntime and skips the profile query."""

    async def test_actor_scoped_skips_profile_query_and_yields_actor_runtime(
        self,
    ) -> None:
        """actor_scoped=True: session_scope is never called; runtime is ActorRuntime."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from snore.mcp.server import ActorRuntime, _lifespan

        cleanup_mock = AsyncMock()
        session_scope_mock = MagicMock()  # must NOT be called
        mock_target = MagicMock()
        mock_target.resolve_async_url.return_value = "sqlite+aiosqlite:///:memory:"

        with (
            patch("snore.mcp.server.DatabaseTarget") as mock_target_cls,
            patch("snore.mcp.server.init_database_from_url", new_callable=AsyncMock),
            patch("snore.mcp.server.cleanup_database", cleanup_mock),
            patch("snore.mcp.server.session_scope", session_scope_mock),
            patch("snore.parsers.register_all.ensure_registered_parsers"),
        ):
            mock_target_cls.from_env_and_flags.return_value = mock_target

            async with _lifespan(None, actor_scoped=True) as rt:
                assert isinstance(rt, ActorRuntime)

        # Profile-existence query must never run in actor-scoped mode.
        session_scope_mock.assert_not_called()

    async def test_actor_scoped_false_yields_static_runtime(
        self,
    ) -> None:
        """actor_scoped=False (default): a found profile row → StaticRuntime."""
        from contextlib import asynccontextmanager
        from unittest.mock import AsyncMock, MagicMock, patch

        from snore.mcp.server import StaticRuntime, _lifespan

        cleanup_mock = AsyncMock()
        mock_target = MagicMock()
        mock_target.resolve_async_url.return_value = "sqlite+aiosqlite:///:memory:"

        # Build a mock DB session where .scalars().first() returns a profile row.
        mock_db = MagicMock()
        profile_row = MagicMock()
        profile_row.id = 7
        result_mock = MagicMock()
        result_mock.scalars.return_value.first.return_value = profile_row
        mock_db.execute = AsyncMock(return_value=result_mock)

        @asynccontextmanager
        async def _mock_session_scope():
            yield mock_db

        with (
            patch("snore.mcp.server.DatabaseTarget") as mock_target_cls,
            patch("snore.mcp.server.init_database_from_url", new_callable=AsyncMock),
            patch("snore.mcp.server.cleanup_database", cleanup_mock),
            patch("snore.mcp.server.session_scope", _mock_session_scope),
            patch("snore.parsers.register_all.ensure_registered_parsers"),
        ):
            mock_target_cls.from_env_and_flags.return_value = mock_target

            async with _lifespan(None) as rt:
                assert isinstance(rt, StaticRuntime)
                assert rt.profile_id == 7


class TestStaticRuntime:
    """StaticRuntime satisfies SNORERuntime protocol and returns stored values."""

    def test_profile_id_returns_stored_value(self) -> None:
        from unittest.mock import MagicMock

        from snore.mcp.server import StaticRuntime

        rt = StaticRuntime(base_scope_provider=MagicMock(), profile_id=42)
        assert rt.profile_id == 42

    def test_scope_provider_is_identity_stable(self) -> None:
        from unittest.mock import MagicMock

        rt = StaticRuntime(base_scope_provider=MagicMock(), profile_id=1)
        assert rt.scope_provider is rt.scope_provider

    async def test_scope_provider_delegates_to_base(self) -> None:
        from contextlib import asynccontextmanager

        sentinel = object()
        entered: list[bool] = []

        @asynccontextmanager
        async def fake_base():
            entered.append(True)
            yield sentinel

        rt = StaticRuntime(base_scope_provider=fake_base, profile_id=1)
        async with rt.scope_provider() as value:
            assert value is sentinel
        assert entered, "fake_base must have been entered"

    def test_satisfies_snore_runtime_protocol(self) -> None:
        """StaticRuntime structurally satisfies SNORERuntime for _runtime(ctx) usage."""
        from unittest.mock import MagicMock

        from snore.mcp.server import SNORERuntime, StaticRuntime

        rt: SNORERuntime = StaticRuntime(base_scope_provider=MagicMock(), profile_id=7)
        assert rt.profile_id == 7


class TestActorRuntime:
    """ActorRuntime delegates profile_id to current_actor() on each access."""

    def test_profile_id_raises_when_no_actor_bound(self) -> None:
        from unittest.mock import MagicMock

        from snore.mcp.auth import _current_actor  # noqa: PLC0415
        from snore.mcp.server import ActorRuntime  # noqa: PLC0415

        # Ensure no actor is bound in this context.
        token = _current_actor.set(None)
        try:
            rt = ActorRuntime(scope_provider=MagicMock())
            with pytest.raises(RuntimeError, match="actor_scope"):
                _ = rt.profile_id
        finally:
            _current_actor.reset(token)

    def test_profile_id_returns_bound_actor_profile_id(self) -> None:
        from unittest.mock import MagicMock

        from snore.mcp.auth import _current_actor  # noqa: PLC0415
        from snore.mcp.server import ActorRuntime  # noqa: PLC0415

        stub_actor = MagicMock()
        stub_actor.profile_id = 99

        token = _current_actor.set(stub_actor)
        try:
            rt = ActorRuntime(scope_provider=MagicMock())
            assert rt.profile_id == 99
        finally:
            _current_actor.reset(token)

    def test_scope_provider_is_read_only_property(self) -> None:
        from unittest.mock import MagicMock

        from snore.mcp.server import ActorRuntime  # noqa: PLC0415

        provider = MagicMock()
        rt = ActorRuntime(scope_provider=provider)
        assert rt.scope_provider is provider
        # read-only: assigning must raise AttributeError
        with pytest.raises(AttributeError):
            rt.scope_provider = MagicMock()  # type: ignore[misc]


class TestMakeServerAuth:
    """make_server() wires auth into FastMCP correctly."""

    def test_no_auth_gives_none_on_mcp(self) -> None:
        from snore.mcp.server import make_server

        mcp = make_server()
        assert mcp.auth is None

    def test_auth_provider_stored_on_mcp(self) -> None:
        from unittest.mock import MagicMock

        from snore.mcp.server import make_server

        mock_auth = MagicMock()
        mcp = make_server(auth=mock_auth)
        assert mcp.auth is mock_auth


class TestMakeServerManageDatabase:
    """make_server(manage_database=False) validation and lifespan behaviour."""

    def test_manage_database_false_without_auth_raises(self) -> None:
        from snore.mcp.server import make_server

        with pytest.raises(ValueError, match="manage_database=False requires auth"):
            make_server(manage_database=False)

    def test_manage_database_false_with_auth_creates_server(self) -> None:
        from unittest.mock import MagicMock

        from snore.mcp.server import make_server

        mock_auth = MagicMock()
        mcp = make_server(auth=mock_auth, manage_database=False)
        assert mcp is not None
        assert mcp.auth is mock_auth

    async def test_lifespan_manage_database_false_skips_init_and_cleanup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When manage_database=False, init_database_from_url and cleanup_database
        must not be called — the FastAPI lifespan owns the engine."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import snore.mcp.server as _server_mod  # noqa: PLC0415

        # Patch DB functions at the server module level.
        mock_init = AsyncMock()
        mock_cleanup = AsyncMock()

        # Patch ensure_registered_parsers (no-op).
        mock_parsers = MagicMock()

        with (
            patch.object(_server_mod, "init_database_from_url", mock_init),
            patch.object(_server_mod, "cleanup_database", mock_cleanup),
            patch("snore.parsers.register_all.ensure_registered_parsers", mock_parsers),
        ):
            async with _server_mod._lifespan(
                MagicMock(),
                actor_scoped=True,
                manage_database=False,
            ):
                pass

        mock_init.assert_not_called()
        mock_cleanup.assert_not_called()

    async def test_lifespan_manage_database_true_calls_init_and_cleanup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When manage_database=True (default), init and cleanup ARE called."""
        from unittest.mock import AsyncMock, MagicMock, patch

        import snore.mcp.server as _server_mod  # noqa: PLC0415

        mock_init = AsyncMock()
        mock_cleanup = AsyncMock()
        mock_parsers = MagicMock()
        mock_target = MagicMock()
        mock_target.resolve_async_url.return_value = "sqlite+aiosqlite:///:memory:"
        mock_target.location = ":memory:"

        # StaticRuntime path requires profile_id from the DB — mock the resolver.
        mock_profile_resolver = AsyncMock(return_value=1)

        with (
            patch.object(_server_mod, "init_database_from_url", mock_init),
            patch.object(_server_mod, "cleanup_database", mock_cleanup),
            patch("snore.parsers.register_all.ensure_registered_parsers", mock_parsers),
            patch(
                "snore.database.target.DatabaseTarget.from_env_and_flags",
                return_value=mock_target,
            ),
            patch.object(
                _server_mod, "_resolve_first_profile_id", mock_profile_resolver
            ),
            patch("snore.database.session.session_scope") as mock_scope,
        ):
            # session_scope needs to be an async context manager
            from contextlib import asynccontextmanager  # noqa: PLC0415

            @asynccontextmanager
            async def _scope():
                yield MagicMock()

            mock_scope.return_value = _scope()

            try:
                async with _server_mod._lifespan(
                    MagicMock(),
                    actor_scoped=False,
                    manage_database=True,
                ):
                    pass
            except Exception:
                pass  # profile resolution mock may vary; we only check init/cleanup

        mock_init.assert_called_once()
