"""get_events and experimental_get_event_context tools —
BreathService.get_contextual_events() adapters.

Timestamp contract (A6):
- Event positions are stored as device wall-clock (naive datetime).
- Output uses tier-2 (offset-free ISO 8601 + timezone_status
  "unknown" | "user_declared", with timezone_name carrying the profile's
  declared IANA zone) for absolute times and tier-3 (offset_seconds from
  Session.start_time) for in-session positions.  No UTC offsets are fabricated.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.shared.versioning import MV_FALLBACK_ALGO_VERSION
from snore.mcp.schemas import (
    EventContext,
    EventRow,
    EventRowBase,
    EventsResponse,
    EventsResponseBase,
    ExperimentalEventContextResponse,
    ExperimentalEventContextRow,
    localize_wall_clock,
    tz_fields,
)
from snore.mcp.tools._capabilities import (
    build_device_capabilities,
    get_device_id_for_session,
)
from snore.mcp.tools._helpers import str_or_none
from snore.mcp.tools._scaffold import (
    _scope_and_run,
    _with_experimental_disclaimer,
    tool_error_boundary,
)
from snore.mcp.tools._service_errors import (
    MAPPED_SERVICE_ERRORS,
    raise_mapped_service_error,
)
from snore.services.breath_service import ContextualEvent, MvSource


async def _fetch_event_page(
    db_session: AsyncSession,
    event_date: date,
    profile_id: int,
    *,
    device_id: int | None,
    types: list[str] | None,
    min_duration: float | None,
    max_events: int,
    include_context: bool,
    include_ventilatory: bool,
    include_device_context: bool = True,
) -> tuple[list[ContextualEvent], EventsResponseBase]:
    """Fetch a date's events and cut them to ``max_events``.

    Returns the kept events plus the response fields both tools share.
    ``total_events`` and the response-level session anchor describe the events
    before truncation.
    """
    from snore.services.breath_service import BreathService  # noqa: PLC0415

    bs = BreathService(db_session, profile_id)
    try:
        events = await bs.get_contextual_events(
            therapy_date=event_date,
            event_types=types,
            min_duration=min_duration,
            device_id=device_id,
            include_context=include_context,
            include_ventilatory=include_ventilatory,
            include_device_context=include_device_context,
        )
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)

    first = events[0] if events else None

    # Resolve device_id for capabilities: use the explicit arg when given;
    # otherwise query the session's device via the shared profile-scoped helper.
    resolved_device_id = device_id
    if resolved_device_id is None and first is not None:
        resolved_device_id = await get_device_id_for_session(
            db_session, first.session_id, profile_id
        )
    caps = (
        await build_device_capabilities(
            db_session,
            profile_id,
            resolved_device_id,
            date_start=event_date,
            date_end=event_date,
        )
        if resolved_device_id is not None
        else None
    )

    # Response-level session anchor: populated only when all events share one session.
    anchor = first if len({ev.session_id for ev in events}) == 1 else None
    return events[:max_events], EventsResponseBase(
        date=event_date.isoformat(),
        session_id=anchor.session_id if anchor else None,
        session_start_wall_clock=(
            localize_wall_clock(
                anchor.session_start_wall_clock,
                anchor.timezone_status,
                anchor.timezone_name,
            )
            if anchor
            else None
        ),
        **tz_fields(first if first is not None else await bs.resolve_timezone()),
        total_events=len(events),
        truncated=len(events) > max_events,
        device_capabilities=caps,
    )


def _row_base(ev: ContextualEvent) -> EventRowBase:
    """The per-event anchors and positions both tools' rows carry."""
    return EventRowBase(
        session_id=ev.session_id,
        session_start_wall_clock=localize_wall_clock(
            ev.session_start_wall_clock, ev.timezone_status, ev.timezone_name
        ),
        event_type=ev.event_type,
        start_time_wall_clock=localize_wall_clock(
            ev.event_start_wall_clock, ev.timezone_status, ev.timezone_name
        ),
        **tz_fields(ev),
        offset_seconds=ev.offset_seconds,
    )


async def get_events(
    db_session: AsyncSession,
    event_date: date,
    profile_id: int,
    device_id: int | None = None,
    types: list[str] | None = None,
    min_duration: float | None = None,
    include_context: bool = True,
    max_events: int = 500,
) -> EventsResponse:
    """Return device-scored respiratory events for a date with inline waveform context.

    Uses BreathService.get_contextual_events() to fetch events enriched with
    pressure/leak at event time, delivered PS, and the device MV channel's
    mean over the prior 120 s.

    Args:
        db_session: Async database session.
        event_date: The date to query (YYYY-MM-DD).
        profile_id: Profile scope for BreathService ownership checks (required).
        device_id: Optional device filter; required when multiple devices share a date.
        types: Optional list of event types to filter (e.g. ["OA", "CA", "H"]).
        min_duration: Minimum event duration in seconds (optional filter).
        include_context: Whether to compute and attach the per-event waveform
            context block; False skips all waveform I/O (context and the
            context ``*_reason`` fields are null).
        max_events: Maximum events to return (≥1, pre-validated by server wrapper).
            total_events reflects the untruncated count; truncated=True when cut.
    """
    events, page = await _fetch_event_page(
        db_session,
        event_date,
        profile_id,
        device_id=device_id,
        types=types,
        min_duration=min_duration,
        max_events=max_events,
        include_context=include_context,
        include_ventilatory=False,
    )

    rows: list[EventRow] = []
    for ev in events:
        context: EventContext | None = None
        if include_context:
            context = EventContext(
                pressure_at_event_cmh2o=ev.pressure_at_event_cmh2o,
                leak_at_event_lpm=ev.leak_at_event_lpm,
                mv_prior_120s_lpm=ev.mv_prior_120s_lpm,
                minutes_since_session_start=round(ev.minutes_since_session_start, 2),
                ps_delivered_cmh2o=ev.ps_delivered_cmh2o,
            )

        rows.append(
            EventRow(
                **_row_base(ev).model_dump(),
                duration_seconds=ev.duration_seconds,
                spo2_drop_pct=ev.spo2_drop,
                peak_flow_limitation=ev.peak_flow_limitation,
                pressure_reason=str_or_none(ev.pressure_reason),
                leak_reason=str_or_none(ev.leak_reason),
                mv_reason=str_or_none(ev.mv_reason),
                ps_reason=str_or_none(ev.ps_reason),
                context=context,
            )
        )

    return EventsResponse(**page.model_dump(), events=rows)


async def experimental_get_event_context(
    db_session: AsyncSession,
    event_date: date,
    profile_id: int,
    device_id: int | None = None,
    types: list[str] | None = None,
    min_duration: float | None = None,
    max_events: int = 500,
) -> ExperimentalEventContextResponse:
    """Return experimental ventilatory context for a date's device-scored events.

    Uses BreathService.get_contextual_events() with ``include_ventilatory`` to
    fetch the preceding MV slope, MV stability index, and MV in the prior 120 s,
    deriving MV from the flow waveform when the device MV channel is absent.

    Args:
        db_session: Async database session.
        event_date: The date to query (YYYY-MM-DD).
        profile_id: Profile scope for BreathService ownership checks (required).
        device_id: Optional device filter; required when multiple devices share a date.
        types: Optional list of event types to filter (e.g. ["OA", "CA", "H"]).
        min_duration: Minimum event duration in seconds (optional filter).
        max_events: Maximum events to return (≥1, pre-validated by server wrapper).
            total_events reflects the untruncated count; truncated=True when cut.
    """
    events, page = await _fetch_event_page(
        db_session,
        event_date,
        profile_id,
        device_id=device_id,
        types=types,
        min_duration=min_duration,
        max_events=max_events,
        include_context=True,
        include_ventilatory=True,
        include_device_context=False,
    )

    return ExperimentalEventContextResponse(
        **page.model_dump(),
        events=[
            ExperimentalEventContextRow(
                **_row_base(ev).model_dump(),
                mv_prior_120s_lpm=ev.mv_prior_120s_lpm,
                mv_reason=str_or_none(ev.mv_reason),
                preceding_mv_slope_lpm_per_min=ev.preceding_mv_slope_lpm_per_min,
                preceding_mv_slope_reason=str_or_none(ev.preceding_mv_slope_reason),
                stability_index=ev.stability_index,
                stability_reason=str_or_none(ev.stability_reason),
                mv_source=str_or_none(ev.mv_source),
            )
            for ev in events
        ],
        mv_fallback_version=(
            MV_FALLBACK_ALGO_VERSION
            if any(ev.mv_source == MvSource.FLOW_DERIVED for ev in events)
            else None
        ),
    )


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import (  # noqa: PLC0415
        parse_date,
        validate_max_events,
        validate_min_duration,
    )

    @mcp.tool()
    @tool_error_boundary
    async def get_events(
        ctx: Context,
        date: str,
        device_id: int | None = None,
        types: list[str] | None = None,
        min_duration: float | None = None,
        include_context: bool = True,
        max_events: int = 500,
    ) -> dict[str, Any]:
        """Return device-scored respiratory events for a single session date with inline waveform context.

        Each event carries the device's event type, start time and duration.
        When waveform data is available, ``context`` adds means of device
        channels around the event: ``pressure_at_event_cmh2o`` and
        ``leak_at_event_lpm`` (±5 s), ``ps_delivered_cmh2o`` (therapy pressure
        − EPAP over ±5 s), and ``mv_prior_120s_lpm`` (device MV channel over
        the prior 120 s), plus ``minutes_since_session_start``.

        Null values carry a ``*_reason`` on the event row (``pressure_reason``,
        ``leak_reason``, ``mv_reason``, ``ps_reason``), including when a device
        channel is absent or corrupt, which nulls only the fields that depend
        on it.
        ``include_context=false`` skips the waveform work entirely: ``context``
        and those reasons are null.

        For MV slope, MV stability, and flow-derived MV on devices with no MV
        channel, use ``experimental_get_event_context``.

        Args:
            date: Session date in YYYY-MM-DD format.
            device_id: Optional device ID filter. Required when multiple devices
                       have data for the same date.
            types: Optional event type filter, e.g. ["CA", "OA", "H", "RE"].
                   Common values: ``OA`` (obstructive apnea), ``CA`` (central
                   apnea), ``H`` (hypopnea), ``RE`` (respiratory effort-related
                   arousal), ``FL`` (flow limitation), ``VS`` (vibratory snore).
            min_duration: Minimum event duration in seconds (optional).
            include_context: Compute and attach the per-event waveform context
                block (default true); false is faster and leaves ``context``
                and the context ``*_reason`` fields null.
            max_events: Maximum number of events to return after filtering (default 500,
                        minimum 1). When the result is truncated, ``total_events`` still
                        reports the full count and ``truncated`` is set to true in the
                        response.

        Returns:
            EventsResponse with events list, total_events count, and truncated flag.
        """
        from snore.mcp.tools.events import get_events as _impl  # noqa: PLC0415

        event_date = parse_date(date, "date")
        validate_min_duration(min_duration)
        validate_max_events(max_events)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_events",
            event_date=event_date,
            device_id=device_id,
            types=types,
            min_duration=min_duration,
            include_context=include_context,
            max_events=max_events,
        )

    @mcp.tool()
    @_with_experimental_disclaimer
    @tool_error_boundary
    async def experimental_get_event_context(
        ctx: Context,
        date: str,
        device_id: int | None = None,
        types: list[str] | None = None,
        min_duration: float | None = None,
        max_events: int = 500,
    ) -> dict[str, Any]:
        """Return SNORE's ventilatory-control context for the device-scored events on one date.

        Each row identifies an event and carries ``preceding_mv_slope_lpm_per_min``
        (MV trend over the 60 s before the event, L/min per minute; negative =
        falling ventilation), ``stability_index`` (MV stdev/mean over the same
        60 s), and ``mv_prior_120s_lpm`` (mean MV over the prior 120 s).

        ``mv_source`` is ``"device"`` (device MV channel), ``"flow_derived"``
        (MV computed from the flow waveform when the device recorded no MV),
        or null. All three metrics use that MV series, so they may be
        flow-derived; the response-level ``mv_fallback_version`` labels the
        derivation and is non-null only when some returned event is
        ``"flow_derived"``. The flow fallback loads the full-session flow
        waveform, so it is slower than ``get_events``.

        Null values carry a ``*_reason`` on the row (``mv_reason``,
        ``preceding_mv_slope_reason``, ``stability_reason``), including when
        the MV or flow channel is corrupt, which nulls only the MV fields.

        Args:
            date: Session date in YYYY-MM-DD format.
            device_id: Optional device ID filter. Required when multiple devices
                       have data for the same date.
            types: Optional event type filter, e.g. ["CA", "OA", "H"].
            min_duration: Minimum event duration in seconds (optional).
            max_events: Maximum number of events to return after filtering (default 500,
                        minimum 1). When the result is truncated, ``total_events`` still
                        reports the full count and ``truncated`` is set to true in the
                        response.

        Returns:
            ExperimentalEventContextResponse with events list, total_events count,
            truncated flag, and mv_fallback_version.
        """
        from snore.mcp.tools.events import (  # noqa: PLC0415
            experimental_get_event_context as _impl,
        )

        event_date = parse_date(date, "date")
        validate_min_duration(min_duration)
        validate_max_events(max_events)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="experimental_get_event_context",
            event_date=event_date,
            device_id=device_id,
            types=types,
            min_duration=min_duration,
            max_events=max_events,
        )
