"""get_waveform_window / render_window tool adapters.

Architecture (DB-fetch / pure-compute split):
- DB fetch runs INSIDE the server's scope_provider context: ``fetch_waveform_raw``
  holds the AsyncSession only during the query; the scope closes before CPU work begins.
- ``waveform_response_from_raw`` and ``render_png_from_raw`` are PURE — they call
  ``compute_waveform_window`` (deserialize/slice/LTTB) and the PNG renderer outside
  the DB scope.  No session access occurs after ``fetch_waveform_raw`` returns.

Timestamp contract (A6):
- Session wall-clock anchor (session_start_wall_clock) uses tier-2
  (offset-free ISO 8601 + timezone_status "unknown" | "user_declared", with
  timezone_name carrying the profile's declared IANA zone) for absolute times.
- In-session waveform positions are numeric offsets in seconds (tier-3).
"""

from __future__ import annotations

import asyncio

from datetime import date
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import Context
from fastmcp.utilities.types import Image
from pydantic import Field

if TYPE_CHECKING:
    from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.schemas import (
    WaveformChannelSchema,
    WaveformWindowResponse,
    localize_wall_clock,
    tz_fields,
)
from snore.mcp.tools._scaffold import _dump_response, tool_error_boundary
from snore.mcp.tools._service_errors import (
    MAPPED_SERVICE_ERRORS,
    raise_mapped_service_error,
)
from snore.services.breath.dtos import MAX_POINTS_MAX, MAX_POINTS_MIN

if TYPE_CHECKING:
    from snore.services.breath_service import RawWaveformWindow

# Declared in the tool schemas so clients see the bounds before calling.
_MaxPoints = Annotated[int | None, Field(ge=MAX_POINTS_MIN, le=MAX_POINTS_MAX)]
_MAX_POINTS_RANGE = f"{MAX_POINTS_MIN}–{MAX_POINTS_MAX}"

# Channel list for the get_waveform description; render_window points to it.
_CHANNEL_VOCAB_DOC = (
    "Channels (unit, rate):\n"
    "``flow``: flow (L/min), 25 Hz.\n"
    "``pressure``: mask pressure (cmH2O), 0.5 Hz duty-cycle average: sits between "
    "EPAP and IPAP, not the bilevel square wave.\n"
    "``therapy_pressure``: target pressure (cmH2O), averaged like ``pressure``.\n"
    "``epap``: EPAP (cmH2O).\n"
    "``leak``: unintentional leak (L/min).\n"
    "``mv``: minute ventilation (L/min).\n"
    "``rr``: respiratory rate (breaths/min).\n"
    "``tv``: tidal volume (mL).\n"
    "``spo2``: oxygen saturation (%).\n"
    "``pulse``: pulse rate (bpm).\n"
    "``fl``: device flow-limitation index (0–1).\n"
    "``snore``: snore signal (0–5).\n"
    "``pressure_hr``: mask pressure (cmH2O), 25 Hz; shows the instantaneous "
    "IPAP/EPAP swing.\n"
    "``ti``: inspiratory time (s), 0.5 Hz; VAuto/AirCurve only.\n"
    "``ie_ratio``: I:E in % as recorded, 0.5 Hz; scale unconfirmed, not Ti/Te; "
    "VAuto/AirCurve only.\n"
    "``trigger_cycle``: undecoded trigger/cycle codes 0–16, 25 Hz; VAuto only. "
    "Categorical: each value holds until the next offset. ``max_points`` keeps "
    "each run start, both edges of each recording gap and the last sample (never "
    "LTTB), so two equal consecutive values mark a gap or the window end, and it "
    "can return more than ``max_points`` values.\n"
    "The last four come only from direct ResMed SD-card imports. Samples are "
    "stored as recorded, sentinel values included."
)


async def fetch_waveform_raw(
    db_session: AsyncSession,
    therapy_date: date,
    profile_id: int,
    offset_start: float,
    offset_end: float,
    device_id: int | None = None,
    session_id: int | None = None,
    channels: list[str] | None = None,
    max_points: int | None = None,
    *,
    window_cap_seconds: float,
) -> RawWaveformWindow:
    """Resolve and fetch raw waveform blobs within the caller's DB scope.

    Builds a ``WaveformWindowRequest`` (Pydantic coerces channel strings to
    ``WaveformChannelName`` and enforces offset ordering, window cap, and
    max_points bounds), then delegates to ``BreathService.fetch_waveform_window``.

    ``DeviceNotOwnedError``, ``DeviceAmbiguityError``, and
    ``MultiSessionAmbiguityError`` are mapped to ``ValidationError`` via
    ``raise_mapped_service_error``.  ``ValueError`` (e.g. explicit session not
    found on a device) and pydantic ``ValidationError`` from request construction
    propagate unchanged to the server's ``tool_error_boundary``.

    Args:
        db_session: Async database session (used only inside this call).
        therapy_date: Therapy date to query.
        profile_id: Profile scope for all ownership checks.
        offset_start: Window start in seconds from session start (>= 0).
        offset_end: Window end in seconds from session start (> offset_start).
        device_id: Filter to a specific device; required when the profile has
            multiple devices with sessions on the date.
        session_id: Filter to a specific session; required when the device had
            multiple sessions on the date.
        channels: Waveform channel names to fetch (e.g. ["flow", "pressure"]).
            Defaults to [flow, pressure, leak] when empty or None.
        max_points: Per-channel target point count for LTTB thinning
            (2–1000); categorical channels (``trigger_cycle``) keep their run
            starts instead and may exceed it.
        window_cap_seconds: Maximum window width in seconds (enforced by caller).
    """
    from snore.services.breath_service import (  # noqa: PLC0415
        BreathService,
        WaveformWindowRequest,
    )

    # Pydantic coerces channel strings to WaveformChannelName and enforces
    # offset ordering, window cap, and max_points bounds. Validation errors
    # propagate to the server's tool_error_boundary → ToolError.
    request = WaveformWindowRequest(
        therapy_date=therapy_date,
        device_id=device_id,
        session_id=session_id,
        offset_start=offset_start,
        offset_end=offset_end,
        channels=channels or [],  # type: ignore[arg-type]  # pydantic coerces str→WaveformChannelName
        max_points=max_points,
        window_cap_seconds=window_cap_seconds,
    )

    bs = BreathService(db_session, profile_id)
    try:
        return await bs.fetch_waveform_window(request)
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)


def waveform_response_from_raw(raw: RawWaveformWindow) -> WaveformWindowResponse:
    """Build a ``WaveformWindowResponse`` from a raw fetch result.

    Pure — no DB access.  Calls ``compute_waveform_window`` (deserialize/slice/LTTB)
    then maps the ``WaveformWindow`` DTO to the MCP response schema.

    Session sentinel scrub: ``session_id == 0`` means no session existed on the
    queried date.  Both ``session_id`` and ``session_start_wall_clock`` are emitted
    as ``null`` in that case — the ``datetime.min`` sentinel is never exposed.
    """
    from snore.services.breath_service import compute_waveform_window  # noqa: PLC0415

    window = compute_waveform_window(raw)

    # Empty-day sentinel: session_id=0 means no session on the date.
    session_id = window.session_id if window.session_id > 0 else None
    session_start_wall_clock = (
        localize_wall_clock(
            window.session_start_wall_clock,
            window.timezone_status,
            window.timezone_name,
        )
        if window.session_id > 0
        else None
    )

    channels = [
        WaveformChannelSchema(
            channel_type=str(ch.channel_type),
            unit=ch.unit,
            sample_rate_hz=ch.sample_rate,
            offset_seconds=ch.offset_seconds,
            values=ch.values,
            original_sample_count=ch.original_sample_count,
            is_downsampled=ch.is_downsampled,
        )
        for ch in window.channels
    ]

    return WaveformWindowResponse(
        session_id=session_id,
        session_start_wall_clock=session_start_wall_clock,
        **tz_fields(window),
        window_start_offset_s=window.window_start_offset,
        window_end_offset_s=window.window_end_offset,
        channels=channels,
        missing_channels=[str(c) for c in window.missing_channels],
        missing_channel_reason=(
            str(window.missing_channel_reason)
            if window.missing_channel_reason
            else None
        ),
    )


def render_png_from_raw(raw: RawWaveformWindow) -> bytes:
    """Render a raw waveform window as a PNG and return the raw bytes.

    Pure — no DB access.  Calls ``compute_waveform_window`` then passes the
    ``WaveformWindow`` DTO to the renderer.  An empty-channel window (e.g. no
    session on the queried date) still produces a valid single-panel PNG with a
    "No waveform data" label.

    Returns:
        Raw PNG bytes; always starts with the PNG magic ``\\x89PNG\\r\\n\\x1a\\n``.
    """
    from snore.mcp.rendering import render_waveform_window  # noqa: PLC0415
    from snore.services.breath_service import compute_waveform_window  # noqa: PLC0415

    window = compute_waveform_window(raw)
    return render_waveform_window(window)


def register(mcp: FastMCP) -> None:
    from snore.mcp.server import _fetch_waveform_for_tool  # noqa: PLC0415

    async def get_waveform(
        ctx: Context,
        date: str,
        offset_start: float,
        offset_end: float,
        device_id: int | None = None,
        session_id: int | None = None,
        channels: list[str] | None = None,
        max_points: _MaxPoints = None,
    ) -> dict[str, Any]:
        from snore.mcp.tools.waveform import waveform_response_from_raw  # noqa: PLC0415

        raw = await _fetch_waveform_for_tool(
            ctx,
            date,
            offset_start=offset_start,
            offset_end=offset_end,
            device_id=device_id,
            session_id=session_id,
            channels=channels,
            max_points=max_points,
            window_cap_seconds=120.0,
        )
        # CPU-bound: deserialize blobs, LTTB, map to schema — off the event loop.
        response = await asyncio.to_thread(waveform_response_from_raw, raw)
        return _dump_response(response, "get_waveform")

    get_waveform.__doc__ = (
        "Raw per-sample waveform arrays for one therapy-night window. Window cap "
        "is 120 s; a wider window is an error.\n\n"
        + _CHANNEL_VOCAB_DOC
        + "\n\nChannels the session did not record are listed in "
        "``missing_channels`` with ``missing_channel_reason`` "
        '``"channel_absent"``. A many-channel raw request can exceed the '
        "500,000-byte response limit; set ``max_points`` or request fewer "
        "channels. A ``device_id`` with no enabled session on the date and no "
        "``session_id`` returns empty ``channels`` with null ``session_id``; "
        "without ``device_id`` such a date is an error.\n\n"
        "Args:\n"
        "    date: Session date (YYYY-MM-DD).\n"
        "    offset_start: Window start, seconds from session start (≥ 0).\n"
        "    offset_end: Window end, seconds from session start.\n"
        "    device_id: Required if several devices have data that date.\n"
        "    session_id: Required if the device had several sessions that date.\n"
        "    channels: Channels to return. Default [flow, pressure, leak].\n"
        f"    max_points: Per-channel LTTB target ({_MAX_POINTS_RANGE}); omit for "
        "raw samples. Channels with no more samples than ``max_points`` come back "
        "raw."
    )
    # Functional form (not @decorator) because __doc__ is built dynamically above
    # before decoration; switching to @mcp.tool() would lose the channel vocab.
    mcp.tool()(tool_error_boundary(get_waveform))

    async def render_window(
        ctx: Context,
        date: str,
        offset_start: float,
        offset_end: float,
        device_id: int | None = None,
        session_id: int | None = None,
        channels: list[str] | None = None,
        max_points: _MaxPoints = None,
    ) -> Image:
        from snore.mcp.tools.waveform import render_png_from_raw  # noqa: PLC0415

        raw = await _fetch_waveform_for_tool(
            ctx,
            date,
            offset_start=offset_start,
            offset_end=offset_end,
            device_id=device_id,
            session_id=session_id,
            channels=channels,
            max_points=max_points,
            window_cap_seconds=900.0,
        )
        # CPU-bound: deserialize blobs, LTTB, matplotlib render — off the event loop.
        png = await asyncio.to_thread(render_png_from_raw, raw)
        return Image(data=png, format="png")

    render_window.__doc__ = (
        "PNG chart of one therapy-night window, one panel per channel. Window cap "
        "is 900 s; a wider window is an error. Channels are named as in "
        "``get_waveform``; missing channels are noted in the image title. A "
        "``device_id`` with no enabled session on the date and no ``session_id`` "
        'returns a "No waveform data" PNG; without ``device_id`` such a date is '
        "an error.\n\n"
        "Args:\n"
        "    date: Session date (YYYY-MM-DD).\n"
        "    offset_start: Window start, seconds from session start (≥ 0).\n"
        "    offset_end: Window end, seconds from session start.\n"
        "    device_id: Required if several devices have data that date.\n"
        "    session_id: Required if the device had several sessions that date.\n"
        "    channels: Channels to plot. Default [flow, pressure, leak].\n"
        f"    max_points: Per-channel LTTB target ({_MAX_POINTS_RANGE}) to thin "
        "dense windows; omit for raw samples."
    )
    mcp.tool()(tool_error_boundary(render_window))
