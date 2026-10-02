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
from typing import TYPE_CHECKING, Any

from fastmcp import Context
from fastmcp.utilities.types import Image

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

if TYPE_CHECKING:
    from snore.services.breath_service import RawWaveformWindow

# Shared channel vocabulary block for get_waveform and render_window tool descriptions.
_CHANNEL_VOCAB_DOC = """\
Channel vocabulary (12 channels):
    ``flow``             — inspiratory/expiratory flow (L/min), 25 Hz from BRP.edf
    ``pressure``         — mask pressure (cmH2O), 0.5 Hz duty-cycle average from
                           PLD.edf.  Reads approximately the time-weighted mean of
                           IPAP/EPAP — NOT the instantaneous bilevel square wave —
                           so it sits between EPAP and IPAP.  No higher-resolution
                           pressure channel exists in ResMed source data.  Plot
                           alongside ``therapy_pressure`` for delivered-vs-target
                           comparison.
    ``therapy_pressure`` — therapy-algorithm target pressure (cmH2O), 0.5 Hz
                           duty-cycle average from PLD.edf; same averaging caveat
                           as ``pressure``.
    ``epap``             — expiratory positive airway pressure (cmH2O)
    ``leak``             — estimated unintentional leak (L/min)
    ``mv``               — minute ventilation derived from flow (L/min)
    ``rr``               — respiratory rate (breaths/min)
    ``tv``               — tidal volume derived from flow (mL)
    ``spo2``             — pulse-oximetry oxygen saturation (%)
    ``pulse``            — pulse rate (bpm)
    ``fl``               — flow-limitation index (dimensionless)
    ``snore``            — snore-intensity signal (arbitrary units)"""


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
        max_points: LTTB target point count for downsampling (1–1000).
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
        max_points: int | None = None,
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
        "Raw per-sample waveform arrays for a single therapy-night window (≤2 min).\n\n"
        "Use this tool for deep numerical inspection of a specific waveform window\n"
        "when ``render_window`` or ``get_breath_table`` is insufficient.  Window cap\n"
        "is 120 s — requesting a wider window is an error.\n\n"
        + _CHANNEL_VOCAB_DOC
        + "\n\n"
        "Default channels when ``channels`` is empty or omitted: flow, pressure, leak.\n\n"
        "``max_points`` (1–1000): when set, applies LTTB downsampling per channel so\n"
        "the visual shape is preserved while reducing data volume.  Omit for raw\n"
        "unmodified samples.  Many-channel raw requests often exceed the 500,000-byte\n"
        "response limit — use ``max_points`` or fewer channels if that happens.\n"
        "Windows whose slice has fewer than 3 samples are returned raw even if\n"
        "``max_points`` is smaller (LTTB needs ≥3 points); ``is_downsampled`` stays false.\n\n"
        "Args:\n"
        "    date: Session date in YYYY-MM-DD format.\n"
        "    offset_start: Window start in seconds from session start (≥ 0).\n"
        "    offset_end: Window end in seconds from session start (> offset_start).\n"
        "                Must be within 120 s of offset_start (enforced).\n"
        "    device_id: Filter to a specific device.  Required when multiple devices\n"
        "               have data for the same date.\n"
        "    session_id: Filter to a specific session.  Required when the device had\n"
        "                multiple sessions on the date.\n"
        "    channels: Waveform channels to return.  Defaults to [flow, pressure, leak].\n"
        "    max_points: LTTB target sample count per channel (1–1000).\n\n"
        "Returns:\n"
        "    WaveformWindowResponse.  ``session_id`` and ``session_start_wall_clock``\n"
        '    (tier-2 offset-free ISO 8601, ``timezone_status: "unknown" | "user_declared"``,\n'
        "    with ``timezone_name`` carrying the profile's declared IANA zone) are null\n"
        "    when the date has no session.  Each channel carries ``offset_seconds``\n"
        "    arrays (tier-3 positions from session start) and ``values``.\n"
        "    ``missing_channels`` lists channels the device did not record;\n"
        '    ``missing_channel_reason: "channel_absent"`` is set when any are absent.\n\n'
        "Refusal semantics (successful responses):\n"
        "    When ``device_id`` is provided (owned device) and the date has no sessions,\n"
        "    the tool returns SUCCESS with ``session_id: null``,\n"
        "    ``session_start_wall_clock: null``, empty ``channels``, and requested\n"
        "    channels in ``missing_channels`` — not an error.\n\n"
        "Error conditions:\n"
        "    - Window width > 120 s → tool error.\n"
        "    - No ``device_id`` provided and no sessions found for date → tool error\n"
        '      ("No sessions found in range <start> to <end>").\n'
        "    - Multiple devices on date and no ``device_id`` → tool error listing IDs.\n"
        "    - Multiple sessions on date and no ``session_id`` → tool error listing IDs.\n"
        "    - Response exceeds 500,000-byte limit → tool error; use ``max_points``\n"
        "      or request fewer channels."
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
        max_points: int | None = None,
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
        "PNG waveform chart for visual inspection of a therapy-night window (≤15 min).\n\n"
        "Returns a stacked-subplot PNG image — one panel per channel — suitable for\n"
        "visual inspection of breathing patterns, flow limitation, leak, and SpO2.\n"
        "Window cap is 900 s (15 min) — requesting a wider window is an error.\n\n"
        + _CHANNEL_VOCAB_DOC
        + "\n\n"
        "Default channels when ``channels`` is empty or omitted: flow, pressure, leak.\n\n"
        "``max_points`` (1–1000): thins dense windows before plotting, preserving visual\n"
        "shape while speeding rendering.  Omit for raw unmodified samples.\n\n"
        "Missing channels (not recorded by the device) are noted in the image title\n"
        "rather than causing an error.\n\n"
        'A date with no sessions returns a valid single-panel PNG with a "No waveform\n'
        'data" label rather than a tool error — but only when ``device_id`` is provided\n'
        "(owned device, empty date) or when a single device is auto-resolved.  When no\n"
        "device can be resolved (no sessions in range and no ``device_id`` supplied)\n"
        "the tool raises an error instead.\n\n"
        "Args:\n"
        "    date: Session date in YYYY-MM-DD format.\n"
        "    offset_start: Window start in seconds from session start (≥ 0).\n"
        "    offset_end: Window end in seconds from session start (> offset_start).\n"
        "                Must be within 900 s of offset_start (enforced).\n"
        "    device_id: Filter to a specific device.  Required when multiple devices\n"
        "               have data for the same date.\n"
        "    session_id: Filter to a specific session.  Required when the device had\n"
        "                multiple sessions on the date.\n"
        "    channels: Waveform channels to plot.  Defaults to [flow, pressure, leak].\n"
        "    max_points: LTTB target sample count per channel before plotting (1–1000).\n\n"
        "Returns:\n"
        "    PNG image (not JSON) — one stacked subplot per channel.\n\n"
        "Error conditions:\n"
        "    - Window width > 900 s → tool error mentioning 900.\n"
        "    - Multiple devices on date and no ``device_id`` → tool error listing IDs.\n"
        "    - Multiple sessions on date and no ``session_id`` → tool error listing IDs."
    )
    mcp.tool()(tool_error_boundary(render_window))
