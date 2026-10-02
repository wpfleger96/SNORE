"""Unit tests for waveform MCP tool adapters.

Pure-function tests (TestPureFunctions, TestPureErrors, TestErrorMapping) run
immediately and must all pass.

Client-level roundtrip tests (TestGetWaveformClient, TestRenderWindowClient)
call ``get_waveform`` and ``render_window`` through an in-memory fastmcp.Client.
These FAIL with "Unknown tool" until the server agent wires the wrappers —
that is expected and does not indicate a defect in this file.
"""

from __future__ import annotations

import json

from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from fastmcp.exceptions import ToolError
from pydantic import ValidationError as PydanticValidationError

from snore.mcp.tools.waveform import (
    fetch_waveform_raw,
    render_png_from_raw,
    waveform_response_from_raw,
)

# ---------------------------------------------------------------------------
# Blob helper (local copy — cross-file test imports are forbidden)
# ---------------------------------------------------------------------------


def _make_blob(offsets: list[float], values: list[float]) -> tuple[bytes, int]:
    """Serialize (offsets, values) as a float32 (n, 2) array — matches deserialize_waveform_blob."""
    arr = np.column_stack([offsets, values]).astype(np.float32)
    return arr.tobytes(), len(offsets)


# ---------------------------------------------------------------------------
# Module-level DTO builders used by multiple test classes
# ---------------------------------------------------------------------------


def _flow_channel(
    n: int = 60,
    time_start: float = 0.0,
    time_end: float = 30.0,
    constant_value: float = 1.0,
    sample_rate: float = 2.0,
    unit: str = "L/min",
) -> object:
    """Build a RawWaveformChannel for FLOW with uniform data."""
    from snore.services.breath_service import (  # noqa: PLC0415
        RawWaveformChannel,
        WaveformChannelName,
    )

    offsets = np.linspace(time_start, time_end, n).tolist()
    values = [constant_value] * n
    blob, count = _make_blob(offsets, values)
    return RawWaveformChannel(
        waveform_type=WaveformChannelName.FLOW,
        unit=unit,
        sample_rate=sample_rate,
        sample_count=count,
        raw_bytes=blob,
    )


def _pressure_channel(
    n: int = 30,
    time_start: float = 0.0,
    time_end: float = 30.0,
    constant_value: float = 10.0,
    sample_rate: float = 1.0,
    unit: str = "cmH2O",
) -> object:
    """Build a RawWaveformChannel for PRESSURE with uniform data."""
    from snore.services.breath_service import (  # noqa: PLC0415
        RawWaveformChannel,
        WaveformChannelName,
    )

    offsets = np.linspace(time_start, time_end, n).tolist()
    values = [constant_value] * n
    blob, count = _make_blob(offsets, values)
    return RawWaveformChannel(
        waveform_type=WaveformChannelName.PRESSURE,
        unit=unit,
        sample_rate=sample_rate,
        sample_count=count,
        raw_bytes=blob,
    )


def _make_request(
    offset_start: float = 0.0,
    offset_end: float = 30.0,
    channels: list | None = None,
    max_points: int | None = None,
    window_cap_seconds: float = 120.0,
) -> object:
    """Build a WaveformWindowRequest with default parameters."""
    from snore.services.breath_service import (  # noqa: PLC0415
        WaveformChannelName,
        WaveformWindowRequest,
    )

    ch_list = channels or [WaveformChannelName.FLOW, WaveformChannelName.PRESSURE]
    return WaveformWindowRequest(
        therapy_date=date(2024, 1, 1),
        offset_start=offset_start,
        offset_end=offset_end,
        channels=ch_list,
        max_points=max_points,
        window_cap_seconds=window_cap_seconds,
    )


def _make_raw(
    *,
    session_id: int = 42,
    channels: list | None = None,
    missing: list | None = None,
    request: object | None = None,
) -> object:
    """Build a RawWaveformWindow with the given channels."""
    from snore.services.breath_service import RawWaveformWindow  # noqa: PLC0415

    req = request if request is not None else _make_request()
    return RawWaveformWindow(
        request=req,
        session_id=session_id,
        session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
        channels=channels if channels is not None else [],
        missing_channels=missing if missing is not None else [],
    )


def _make_empty_day_raw() -> object:
    """Build the sentinel RawWaveformWindow returned for a date with no sessions."""
    from snore.services.breath_service import (  # noqa: PLC0415
        RawWaveformWindow,
        WaveformChannelName,
        WaveformWindowRequest,
    )

    request = WaveformWindowRequest(
        therapy_date=date(2024, 1, 1),
        offset_start=0.0,
        offset_end=30.0,
        channels=[WaveformChannelName.FLOW, WaveformChannelName.PRESSURE],
        window_cap_seconds=120.0,
    )
    return RawWaveformWindow(
        request=request,
        session_id=0,
        session_start_wall_clock=datetime.min,
        channels=[],
        missing_channels=[WaveformChannelName.FLOW, WaveformChannelName.PRESSURE],
    )


# ---------------------------------------------------------------------------
# TestPureFunctions — waveform_response_from_raw + render_png_from_raw
# ---------------------------------------------------------------------------


class TestPureFunctions:
    def test_happy_path_two_channels(self) -> None:
        """Two-channel raw: flow+pressure → correct schema fields and timezone_status."""
        flow = _flow_channel(n=60, time_start=0.0, time_end=30.0, sample_rate=2.0)
        pressure = _pressure_channel(
            n=30, time_start=0.0, time_end=30.0, sample_rate=1.0
        )
        raw = _make_raw(session_id=42, channels=[flow, pressure])

        response = waveform_response_from_raw(raw)

        assert response.session_id == 42
        assert response.session_start_wall_clock == "2024-01-01T22:00:00"
        assert response.timezone_status == "unknown"
        assert response.window_start_offset_s == pytest.approx(0.0)
        assert response.window_end_offset_s == pytest.approx(30.0)

        channel_types = {ch.channel_type for ch in response.channels}
        assert "flow" in channel_types
        assert "pressure" in channel_types

        for ch in response.channels:
            assert ch.sample_rate_hz > 0
            assert len(ch.values) > 0
            assert len(ch.offset_seconds) == len(ch.values)

        assert response.missing_channels == []
        assert response.missing_channel_reason is None

    def test_empty_day_sentinel_scrub(self) -> None:
        """session_id=0 → session_id null, session_start_wall_clock null, reason channel_absent."""
        raw = _make_empty_day_raw()
        response = waveform_response_from_raw(raw)

        assert response.session_id is None
        assert response.session_start_wall_clock is None
        assert response.channels == []
        assert "flow" in response.missing_channels
        assert "pressure" in response.missing_channels
        assert response.missing_channel_reason == "channel_absent"

    def test_lttb_downsampling(self) -> None:
        """500-sample channel with max_points=100 → is_downsampled=True, len(values)==100."""
        from snore.services.breath_service import (  # noqa: PLC0415
            RawWaveformChannel,
            WaveformChannelName,
            WaveformWindowRequest,
        )

        n = 500
        offsets = np.linspace(0.0, 50.0, n).tolist()
        values = [float(i % 20) for i in range(n)]
        blob, count = _make_blob(offsets, values)

        ch = RawWaveformChannel(
            waveform_type=WaveformChannelName.FLOW,
            unit="L/min",
            sample_rate=10.0,
            sample_count=count,
            raw_bytes=blob,
        )
        request = WaveformWindowRequest(
            therapy_date=date(2024, 1, 1),
            offset_start=0.0,
            offset_end=50.0,
            channels=[WaveformChannelName.FLOW],
            max_points=100,
            window_cap_seconds=120.0,
        )
        from snore.services.breath_service import RawWaveformWindow  # noqa: PLC0415

        raw = RawWaveformWindow(
            request=request,
            session_id=7,
            session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
            channels=[ch],
            missing_channels=[],
        )

        response = waveform_response_from_raw(raw)

        assert len(response.channels) == 1
        result_ch = response.channels[0]
        assert result_ch.is_downsampled is True
        assert len(result_ch.values) == 100
        assert result_ch.original_sample_count == 500

    def test_render_png_returns_png_magic(self) -> None:
        """render_png_from_raw always starts with the 8-byte PNG magic signature."""
        flow = _flow_channel(n=30, time_start=0.0, time_end=30.0)
        raw = _make_raw(channels=[flow])

        png_bytes = render_png_from_raw(raw)

        assert isinstance(png_bytes, bytes)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_png_empty_day_still_returns_valid_png(self) -> None:
        """Empty-day window (no channels) produces a valid PNG rather than an error."""
        raw = _make_empty_day_raw()
        png_bytes = render_png_from_raw(raw)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_lttb_minimum_max_points_downsamples_to_two(self) -> None:
        """max_points=2 (the lower bound) on a 500-sample channel → 2 values, is_downsampled=True."""
        from snore.services.breath_service import (  # noqa: PLC0415
            RawWaveformChannel,
            RawWaveformWindow,
            WaveformChannelName,
        )

        n = 500
        offsets = np.linspace(0.0, 50.0, n).tolist()
        values = [float(i % 20) for i in range(n)]
        blob, count = _make_blob(offsets, values)

        ch = RawWaveformChannel(
            waveform_type=WaveformChannelName.FLOW,
            unit="L/min",
            sample_rate=10.0,
            sample_count=count,
            raw_bytes=blob,
        )
        raw = RawWaveformWindow(
            request=_make_request(
                offset_end=50.0, channels=[WaveformChannelName.FLOW], max_points=2
            ),
            session_id=7,
            session_start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
            channels=[ch],
            missing_channels=[],
        )

        response = waveform_response_from_raw(raw)

        result_ch = response.channels[0]
        assert result_ch.is_downsampled is True
        assert len(result_ch.values) == 2
        assert result_ch.original_sample_count == 500

    @pytest.mark.parametrize(
        ("n", "max_points"),
        [(2, 2), (5, 10)],
        ids=["equal_to_max_points", "below_max_points"],
    )
    def test_window_within_max_points_passes_through_raw(
        self, n: int, max_points: int
    ) -> None:
        """A slice with no more samples than max_points is returned raw, not downsampled."""
        from snore.services.breath_service import WaveformChannelName  # noqa: PLC0415

        flow = _flow_channel(n=n, time_start=0.0, time_end=10.0)
        raw = _make_raw(
            channels=[flow],
            request=_make_request(
                offset_end=10.0,
                channels=[WaveformChannelName.FLOW],
                max_points=max_points,
            ),
        )

        response = waveform_response_from_raw(raw)

        result_ch = response.channels[0]
        assert result_ch.is_downsampled is False
        assert len(result_ch.values) == n
        assert result_ch.original_sample_count == n


# ---------------------------------------------------------------------------
# TestPureErrors — fetch_waveform_raw input validation
# ---------------------------------------------------------------------------


class TestPureErrors:
    async def test_window_exceeds_cap_raises_before_service_call(
        self, mock_db_session: MagicMock
    ) -> None:
        """offset_end - offset_start > window_cap_seconds → pydantic error; service not called."""
        with patch(
            "snore.services.breath_service.BreathService.fetch_waveform_window",
            new_callable=AsyncMock,
        ) as mock_svc:
            with pytest.raises(PydanticValidationError):
                await fetch_waveform_raw(
                    mock_db_session,
                    date(2024, 1, 1),
                    profile_id=1,
                    offset_start=0.0,
                    offset_end=200.0,  # 200 s > 120 s cap
                    window_cap_seconds=120.0,
                )

        mock_svc.assert_not_called()

    async def test_unknown_channel_string_raises_before_service_call(
        self, mock_db_session: MagicMock
    ) -> None:
        """Unrecognised channel string → pydantic error; service not called."""
        with patch(
            "snore.services.breath_service.BreathService.fetch_waveform_window",
            new_callable=AsyncMock,
        ) as mock_svc:
            with pytest.raises(PydanticValidationError):
                await fetch_waveform_raw(
                    mock_db_session,
                    date(2024, 1, 1),
                    profile_id=1,
                    offset_start=0.0,
                    offset_end=30.0,
                    channels=["invalid_channel"],
                    window_cap_seconds=120.0,
                )

        mock_svc.assert_not_called()


# ---------------------------------------------------------------------------
# TestErrorMapping — service exceptions mapped to ValidationError
# ---------------------------------------------------------------------------


class TestErrorMapping:
    async def test_device_not_owned_maps_to_validation_error(
        self, mock_db_session: MagicMock
    ) -> None:
        """DeviceNotOwnedError → ValidationError naming device_id but never profile_id."""
        from snore.mcp.errors import ValidationError  # noqa: PLC0415
        from snore.services.breath_service import DeviceNotOwnedError  # noqa: PLC0415

        exc = DeviceNotOwnedError(device_id=42, profile_id=999)

        with patch(
            "snore.services.breath_service.BreathService.fetch_waveform_window",
            new_callable=AsyncMock,
            side_effect=exc,
        ):
            with pytest.raises(ValidationError) as exc_info:
                await fetch_waveform_raw(
                    mock_db_session,
                    date(2024, 1, 1),
                    profile_id=1,
                    offset_start=0.0,
                    offset_end=30.0,
                    device_id=42,
                    window_cap_seconds=120.0,
                )

        message = str(exc_info.value)
        assert "device_id=42" in message
        assert "is not available in this session" in message
        assert "999" not in message  # no profile id

    async def test_device_ambiguity_maps_to_validation_error_listing_ids(
        self, mock_db_session: MagicMock
    ) -> None:
        """DeviceAmbiguityError → ValidationError listing both device IDs."""
        from snore.mcp.errors import ValidationError  # noqa: PLC0415
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        exc = DeviceAmbiguityError(
            therapy_date=date(2024, 1, 1),
            profile_id=1,
            owned_device_ids=[10, 20],
            device_serials={10: "SN10", 20: "SN20"},
        )

        with patch(
            "snore.services.breath_service.BreathService.fetch_waveform_window",
            new_callable=AsyncMock,
            side_effect=exc,
        ):
            with pytest.raises(ValidationError) as exc_info:
                await fetch_waveform_raw(
                    mock_db_session,
                    date(2024, 1, 1),
                    profile_id=1,
                    offset_start=0.0,
                    offset_end=30.0,
                    window_cap_seconds=120.0,
                )

        message = str(exc_info.value)
        assert "device_id=10" in message
        assert "device_id=20" in message
        assert "pass device_id" in message

    async def test_multi_session_ambiguity_maps_to_validation_error_listing_sessions(
        self, mock_db_session: MagicMock
    ) -> None:
        """MultiSessionAmbiguityError → ValidationError listing session IDs."""
        from snore.mcp.errors import ValidationError  # noqa: PLC0415
        from snore.services.breath_service import (  # noqa: PLC0415
            MultiSessionAmbiguityError,
            SessionSummary,
        )

        exc = MultiSessionAmbiguityError(
            therapy_date=date(2024, 1, 1),
            device_id=5,
            sessions=[
                SessionSummary(
                    session_id=100,
                    start_wall_clock=datetime(2024, 1, 1, 22, 0, 0),
                    duration_seconds=3600.0,
                ),
                SessionSummary(
                    session_id=101,
                    start_wall_clock=datetime(2024, 1, 2, 1, 0, 0),
                    duration_seconds=7200.0,
                ),
            ],
        )

        with patch(
            "snore.services.breath_service.BreathService.fetch_waveform_window",
            new_callable=AsyncMock,
            side_effect=exc,
        ):
            with pytest.raises(ValidationError) as exc_info:
                await fetch_waveform_raw(
                    mock_db_session,
                    date(2024, 1, 1),
                    profile_id=1,
                    offset_start=0.0,
                    offset_end=30.0,
                    window_cap_seconds=120.0,
                )

        message = str(exc_info.value)
        assert "session_id=100" in message
        assert "session_id=101" in message
        assert "pass session_id" in message


# ---------------------------------------------------------------------------
# TestGetWaveformClient — expected-failing until server wires get_waveform
#
# These tests call the "get_waveform" tool through a real fastmcp.Client.
# They FAIL with "Unknown tool" (or similar) until the server agent adds the
# wrapper in server.py.  Once wired, all tests in this class must pass.
# ---------------------------------------------------------------------------


class TestGetWaveformClient:
    async def test_roundtrip_field_mapping(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """get_waveform returns JSON with correct field names incl. window_start_offset_s."""
        flow = _flow_channel(n=60, time_start=0.0, time_end=30.0)
        raw = _make_raw(session_id=42, channels=[flow])

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.fetch_waveform_window",
                    new_callable=AsyncMock,
                    return_value=raw,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "get_waveform",
                {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 30.0},
            )

        assert not result.is_error
        payload = json.loads(result.content[0].text)
        assert payload["session_id"] == 42
        assert payload["session_start_wall_clock"] == "2024-01-01T22:00:00"
        assert payload["timezone_status"] == "unknown"
        assert payload["window_start_offset_s"] == pytest.approx(0.0)
        assert payload["window_end_offset_s"] == pytest.approx(30.0)
        assert len(payload["channels"]) == 1
        assert payload["channels"][0]["channel_type"] == "flow"

    async def test_oversize_response_raises_tool_error(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """Response exceeding RESPONSE_SIZE_LIMIT → ToolError advising to narrow the query."""
        flow = _flow_channel(n=60, time_start=0.0, time_end=30.0)
        raw = _make_raw(session_id=42, channels=[flow])

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.fetch_waveform_window",
                    new_callable=AsyncMock,
                    return_value=raw,
                ),
                patch("snore.mcp.tools._scaffold.RESPONSE_SIZE_LIMIT", new=1),
            ],
        ) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "get_waveform",
                    {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 30.0},
                )

    async def test_invalid_date_string_raises_tool_error(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """Unparseable date string → ToolError before the service is called."""
        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "get_waveform",
                    {
                        "date": "not-a-date",
                        "offset_start": 0.0,
                        "offset_end": 30.0,
                    },
                )

    async def test_get_waveform_120s_cap_raises_tool_error(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """get_waveform: offset_end - offset_start > 120 s → ToolError (pydantic/ValueError)."""
        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError):
                await client.call_tool(
                    "get_waveform",
                    {
                        "date": "2024-01-01",
                        "offset_start": 0.0,
                        "offset_end": 200.0,  # 200 s > 120 s cap
                    },
                )

    async def test_get_waveform_cap_error_message_omits_pydantic_internals(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """120 s cap violation → ToolError contains the human message only.

        The error must include the relevant cap text and must NOT expose pydantic
        schema internals (class name, input_value repr, pydantic.dev URLs).
        """
        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "get_waveform",
                    {
                        "date": "2024-01-01",
                        "offset_start": 0.0,
                        "offset_end": 121.0,  # 121 s > 120 s cap
                    },
                )

        err = str(exc_info.value)
        assert "exceeds the 120 s cap" in err
        assert "pydantic" not in err.lower()
        assert "input_value" not in err
        assert "validation error for" not in err.lower()

    @pytest.mark.parametrize("tool", ["get_waveform", "render_window"])
    async def test_max_points_below_two_raises_clean_tool_error(
        self, mock_db_session: MagicMock, mcp_client_factory: object, tool: str
    ) -> None:
        """max_points=1 → ToolError naming max_points; service never reaches LTTB."""
        with patch(
            "snore.services.breath_service.BreathService.fetch_waveform_window",
            new_callable=AsyncMock,
        ) as mock_svc:
            async with mcp_client_factory(mock_db_session) as client:
                with pytest.raises(ToolError) as exc_info:
                    await client.call_tool(
                        tool,
                        {
                            "date": "2024-01-01",
                            "offset_start": 0.0,
                            "offset_end": 30.0,
                            "max_points": 1,
                        },
                    )

        err = str(exc_info.value)
        assert "max_points" in err
        assert "target_points" not in err
        assert "pydantic" not in err.lower()
        assert "input_value" not in err
        mock_svc.assert_not_called()

    @pytest.mark.parametrize("tool", ["get_waveform", "render_window"])
    async def test_max_points_bounds_advertised_in_input_schema(
        self, mock_db_session: MagicMock, mcp_client_factory: object, tool: str
    ) -> None:
        """Clients see the max_points range in the tool schema, not only the docstring."""
        from snore.services.breath.dtos import (  # noqa: PLC0415
            MAX_POINTS_MAX,
            MAX_POINTS_MIN,
        )

        async with mcp_client_factory(mock_db_session) as client:
            tools = await client.list_tools()

        schema = next(t for t in tools if t.name == tool).inputSchema
        int_branch = next(
            b
            for b in schema["properties"]["max_points"]["anyOf"]
            if b.get("type") == "integer"
        )
        assert int_branch["minimum"] == MAX_POINTS_MIN
        assert int_branch["maximum"] == MAX_POINTS_MAX


# ---------------------------------------------------------------------------
# TestRenderWindowClient — expected-failing until server wires render_window
#
# These tests call the "render_window" tool through a real fastmcp.Client.
# They FAIL with "Unknown tool" (or similar) until the server agent adds the
# wrapper in server.py.  Once wired, all tests in this class must pass.
# ---------------------------------------------------------------------------


class TestRenderWindowClient:
    async def test_render_window_returns_image_content_with_png_magic(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """render_window returns ImageContent whose decoded payload starts with PNG magic."""
        import base64  # noqa: PLC0415

        flow = _flow_channel(n=60, time_start=0.0, time_end=30.0)
        raw = _make_raw(session_id=42, channels=[flow])

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.fetch_waveform_window",
                    new_callable=AsyncMock,
                    return_value=raw,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "render_window",
                {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 30.0},
            )

        content = result.content[0]
        assert content.type == "image"
        assert content.mimeType == "image/png"
        img_bytes = base64.b64decode(content.data)
        assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    async def test_render_window_empty_day_returns_image_not_error(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """render_window on a date with no sessions → ImageContent (not ToolError)."""
        raw = _make_empty_day_raw()

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.fetch_waveform_window",
                    new_callable=AsyncMock,
                    return_value=raw,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "render_window",
                {"date": "2024-01-01", "offset_start": 0.0, "offset_end": 30.0},
            )

        assert not result.is_error
        assert result.content[0].type == "image"

    async def test_render_window_900s_cap_raises_tool_error(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """render_window: offset_end - offset_start > 900 s → ToolError mentioning 900."""
        async with mcp_client_factory(mock_db_session) as client:
            with pytest.raises(ToolError) as exc_info:
                await client.call_tool(
                    "render_window",
                    {
                        "date": "2024-01-01",
                        "offset_start": 0.0,
                        "offset_end": 1000.0,  # 1000 s > 900 s cap for render_window
                    },
                )

        assert "900" in str(exc_info.value)

    async def test_render_window_channels_passthrough_to_service(
        self, mock_db_session: MagicMock, mcp_client_factory: object
    ) -> None:
        """render_window with channels=['spo2'] → WaveformWindowRequest contains SPO2;
        result is still ImageContent."""
        import base64  # noqa: PLC0415

        from snore.services.breath_service import WaveformChannelName  # noqa: PLC0415

        flow = _flow_channel(n=60, time_start=0.0, time_end=30.0)
        raw = _make_raw(session_id=42, channels=[flow])

        mock_fetch = AsyncMock(return_value=raw)

        async with mcp_client_factory(
            mock_db_session,
            extra_patches=[
                patch(
                    "snore.services.breath_service.BreathService.fetch_waveform_window",
                    mock_fetch,
                ),
            ],
        ) as client:
            result = await client.call_tool(
                "render_window",
                {
                    "date": "2024-01-01",
                    "offset_start": 0.0,
                    "offset_end": 30.0,
                    "channels": ["spo2"],
                },
            )

        # Verify the service received a request with SPO2 channel
        mock_fetch.assert_called_once()
        call_request = mock_fetch.call_args[0][0]
        assert WaveformChannelName.SPO2 in call_request.channels

        # Result must still be ImageContent with valid PNG
        content = result.content[0]
        assert content.type == "image"
        img_bytes = base64.b64decode(content.data)
        assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n"
