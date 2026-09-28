"""get_events tool — BreathService.get_contextual_events() adapter.

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

from snore.mcp.schemas import (
    EventContext,
    EventRow,
    EventsResponse,
    localize_wall_clock,
    tz_fields,
)
from snore.mcp.tools._capabilities import (
    build_device_capabilities,
    get_device_id_for_session,
)
from snore.mcp.tools._helpers import str_or_none
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.mcp.tools._service_errors import (
    MAPPED_SERVICE_ERRORS,
    raise_mapped_service_error,
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
    """Return respiratory events for a session date with inline waveform context.

    Uses BreathService.get_contextual_events() to fetch events enriched with
    pressure/leak at event time, MV in the prior 120 s, and ventilatory-control
    context (preceding MV slope, MV stability index, delivered PS).

    Args:
        db_session: Async database session.
        event_date: The date to query (YYYY-MM-DD).
        profile_id: Profile scope for BreathService ownership checks (required).
        device_id: Optional device filter; required when multiple devices share a date.
        types: Optional list of event types to filter (e.g. ["OA", "CA", "H"]).
        min_duration: Minimum event duration in seconds (optional filter).
        include_context: Whether to attach per-event waveform context block.
        max_events: Maximum events to return (≥1, pre-validated by server wrapper).
            total_events reflects the untruncated count; truncated=True when cut.
    """
    from snore.services.breath_service import BreathService  # noqa: PLC0415

    bs = BreathService(db_session, profile_id)
    try:
        contextual_events = await bs.get_contextual_events(
            therapy_date=event_date,
            event_types=types,
            min_duration=min_duration,
            device_id=device_id,
        )
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)

    if not contextual_events:
        caps = (
            await build_device_capabilities(
                db_session,
                profile_id,
                device_id,
                date_start=event_date,
                date_end=event_date,
            )
            if device_id is not None
            else None
        )
        tz = await bs.resolve_timezone()
        return EventsResponse(
            date=event_date.isoformat(),
            session_id=None,
            session_start_wall_clock=None,
            **tz_fields(tz),
            events=[],
            total_events=0,
            device_capabilities=caps,
        )

    total_events = len(contextual_events)
    truncated = total_events > max_events
    events_to_map = contextual_events[:max_events] if truncated else contextual_events

    rows: list[EventRow] = []
    for ev in events_to_map:
        context: EventContext | None = None
        if include_context:
            context = EventContext(
                pressure_at_event_cmh2o=ev.pressure_at_event_cmh2o,
                leak_at_event_lpm=ev.leak_at_event_lpm,
                mv_prior_120s_lpm=ev.mv_prior_120s_lpm,
                minutes_since_session_start=round(ev.minutes_since_session_start, 2),
                preceding_mv_slope_lpm_per_min=ev.preceding_mv_slope_lpm_per_min,
                stability_index=ev.stability_index,
                ps_delivered_cmh2o=ev.ps_delivered_cmh2o,
                mv_source=str_or_none(ev.mv_source),
            )

        rows.append(
            EventRow(
                session_id=ev.session_id,
                session_start_wall_clock=localize_wall_clock(
                    ev.session_start_wall_clock,
                    ev.timezone_status,
                    ev.timezone_name,
                ),
                event_type=ev.event_type,
                start_time_wall_clock=localize_wall_clock(
                    ev.event_start_wall_clock, ev.timezone_status, ev.timezone_name
                ),
                **tz_fields(ev),
                offset_seconds=ev.offset_seconds,
                duration_seconds=ev.duration_seconds,
                spo2_drop_pct=None,
                peak_flow_limitation=None,
                pressure_reason=str_or_none(ev.pressure_reason),
                leak_reason=str_or_none(ev.leak_reason),
                mv_reason=str_or_none(ev.mv_reason),
                preceding_mv_slope_reason=str_or_none(ev.preceding_mv_slope_reason),
                stability_reason=str_or_none(ev.stability_reason),
                ps_reason=str_or_none(ev.ps_reason),
                context=context,
            )
        )

    # Response-level session anchor: populated only when all events share one session.
    session_ids = {ev.session_id for ev in contextual_events}
    if len(session_ids) == 1:
        anchor_session_id: int | None = contextual_events[0].session_id
        anchor_session_start: str | None = localize_wall_clock(
            contextual_events[0].session_start_wall_clock,
            contextual_events[0].timezone_status,
            contextual_events[0].timezone_name,
        )
    else:
        anchor_session_id = None
        anchor_session_start = None

    # Resolve device_id for capabilities: use the explicit arg when given;
    # otherwise query the session's device via the shared profile-scoped helper.
    resolved_device_id = device_id
    if resolved_device_id is None:
        resolved_device_id = await get_device_id_for_session(
            db_session, contextual_events[0].session_id, profile_id
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

    return EventsResponse(
        date=event_date.isoformat(),
        session_id=anchor_session_id,
        session_start_wall_clock=anchor_session_start,
        **tz_fields(contextual_events[0]),
        events=rows,
        total_events=total_events,
        truncated=truncated,
        device_capabilities=caps,
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
        """Return respiratory events for a single session date with inline waveform context.

        Each event includes pressure/leak at the event time and MV in the prior
        120 s (when waveform data is available), plus minutes since session start.

        Every event type also carries ventilatory-control context in ``context``:
        ``preceding_mv_slope_lpm_per_min`` (MV trend over the 60 s before the
        event, L/min per minute; negative = falling ventilation),
        ``stability_index`` (MV stdev/mean over the same 60 s), and
        ``ps_delivered_cmh2o`` (therapy pressure − EPAP over ±5 s).
        ``mv_source`` is ``"device"`` (device MV channel), ``"flow_derived"``
        (MV computed from the flow waveform when the device recorded no MV),
        or null. Null values carry a ``*_reason`` on the event row
        (``preceding_mv_slope_reason``, ``stability_reason``, ``ps_reason``).

        Args:
            date: Session date in YYYY-MM-DD format.
            device_id: Optional device ID filter. Required when multiple devices
                       have data for the same date.
            types: Optional event type filter, e.g. ["CA", "OA", "H", "RERA"].
                   Common values: ``OA`` (obstructive apnea), ``CA`` (central
                   apnea), ``H`` (hypopnea), ``RERA`` (respiratory effort-related
                   arousal), ``FL`` (flow limitation), ``VS`` (vibratory snore).
            min_duration: Minimum event duration in seconds (optional).
            include_context: Attach per-event waveform context block (default true).
            max_events: Maximum number of events to return after filtering (default 500,
                        minimum 1). When the result is truncated, ``total_events`` still
                        reports the full unfiltered count and ``truncated`` is set to true
                        in the response.

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
