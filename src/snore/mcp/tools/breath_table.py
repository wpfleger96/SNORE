"""get_breath_table tool — BreathService.get_breath_table() adapter.

Timestamp contract (A6):
- Session wall-clock anchor (session_start_wall_clock) uses tier-2
  (offset-free ISO 8601 + timezone_status "unknown" | "user_declared", with
  timezone_name carrying the profile's declared IANA zone) for absolute times.
- In-session positions are numeric offsets in seconds (tier-3).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.errors import ValidationError
from snore.mcp.schemas import (
    BreathTableBin,
    BreathTableQuery,
    BreathTableResponse,
    BreathTableRow,
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


async def get_breath_table(
    db_session: AsyncSession,
    therapy_date: date,
    profile_id: int,
    device_id: int | None = None,
    session_id: int | None = None,
    offset_start: float = 0.0,
    offset_end: float = 900.0,
    page: int = 1,
    page_size: int = 500,
    bin_minutes: float | None = None,
) -> BreathTableResponse:
    """Return a paginated breath-level table for a single therapy night.

    Builds a BreathQueryRange (Pydantic validates offset bounds, page_size cap,
    and the 15-min raw window limit), calls BreathService.get_breath_table, and
    maps the typed DTO to BreathTableResponse. No domain computation is done here.

    Args:
        db_session: Async database session.
        therapy_date: Therapy date to query.
        profile_id: Profile scope for ownership checks (required).
        device_id: Filter to a specific device; required when multiple devices
            share the date.
        session_id: Filter to a specific session; required when the device had
            multiple sessions on the date.
        offset_start: Window start in seconds from session start (>= 0).
        offset_end: Window end in seconds from session start (> offset_start).
            Raw window must be <= 900 s unless bin_minutes is set.
        page: Page number for raw rows (1-based, default 1).
        page_size: Rows per page for raw fetch (default 500, max 2000).
        bin_minutes: When set (>= 1.0), aggregate into time bins instead of
            returning raw rows. Required for windows > 15 min.
    """
    from snore.services.breath_service import (  # noqa: PLC0415
        AnalysisStatus,
        BreathQueryRange,
        BreathService,
    )

    # Pydantic validation errors (offset ordering, page_size cap, 15-min raw cap)
    # propagate to the server error boundary, which maps them to ToolError.
    query = BreathQueryRange(
        therapy_date=therapy_date,
        device_id=device_id,
        session_id=session_id,
        offset_start=offset_start,
        offset_end=offset_end,
        page=page,
        page_size=page_size,
        bin_minutes=bin_minutes,
    )

    bs = BreathService(db_session, profile_id)
    try:
        dto = await bs.get_breath_table(query)
    except MAPPED_SERVICE_ERRORS as exc:
        raise_mapped_service_error(exc)
    except ValueError as exc:
        # This branch is load-bearing. NoSessionsInRangeError (device resolution,
        # "in range X to Y") routes through MAPPED_SERVICE_ERRORS above. But
        # breath_service.py raises plain ValueError("No sessions found for date X")
        # for an owned device with an empty day — that lands here and is NOT caught
        # by MAPPED_SERVICE_ERRORS. Do not remove this fallback.
        msg = str(exc)
        if "No sessions found" in msg:
            raise ValidationError(
                f"No therapy data found for date {therapy_date}. "
                "Use get_data_overview to check which dates have imported data."
            ) from exc
        raise ValidationError(msg) from exc

    # Echo the resolved query — therapy_date is a date object, convert to string.
    mapped_query = BreathTableQuery(
        therapy_date=dto.query.therapy_date.isoformat(),
        device_id=dto.query.device_id,
        session_id=dto.query.session_id,
        offset_start=dto.query.offset_start,
        offset_end=dto.query.offset_end,
        page=dto.query.page,
        page_size=dto.query.page_size,
        bin_minutes=dto.query.bin_minutes,
    )

    # Top-level session anchor: dto.session_id is populated by the service for all
    # return paths (raw, binned, not_run, stale_version); fallback only for legacy stubs.
    top_session_id: int | None = dto.session_id
    top_session_start: str | None
    if dto.rows:
        top_session_start = localize_wall_clock(
            dto.rows[0].session_start_wall_clock,
            dto.rows[0].timezone_status,
            dto.rows[0].timezone_name,
        )
    elif dto.bins:
        top_session_start = localize_wall_clock(
            dto.bins[0].session_start_wall_clock,
            dto.bins[0].timezone_status,
            dto.bins[0].timezone_name,
        )
    else:
        top_session_start = None

    # Map raw rows — field renames per contract §4a:
    #   ti → ti_s, te → te_s, ttot → ttot_s
    #   peak_insp_flow → peak_insp_flow_lpm, peak_exp_flow → peak_exp_flow_lpm
    #   tidal_volume → tidal_volume_ml
    # Enum values serialised to str; None stays None.
    rows = [
        BreathTableRow(
            analysis_result_id=r.analysis_result_id,
            session_id=r.session_id,
            breath_number=r.breath_number,
            session_start_wall_clock=localize_wall_clock(
                r.session_start_wall_clock, r.timezone_status, r.timezone_name
            ),
            **tz_fields(r),
            start_offset_seconds=r.start_offset_seconds,
            end_offset_seconds=r.end_offset_seconds,
            ti_s=r.ti,
            te_s=r.te,
            ttot_s=r.ttot,
            ie_ratio=r.ie_ratio,
            duty_cycle=r.duty_cycle,
            peak_insp_flow_lpm=r.peak_insp_flow,
            peak_exp_flow_lpm=r.peak_exp_flow,
            tidal_volume_ml=r.tidal_volume,
            flatness_index=r.flatness_index,
            mid_insp_flattening=r.mid_insp_flattening,
            flow_class=r.flow_class,
            flow_class_confidence=r.flow_class_confidence,
            is_recovery_breath=r.is_recovery_breath,
            trigger_type=str_or_none(r.trigger_type),
            cycle_type=str_or_none(r.cycle_type),
            trigger_cycle_confidence=r.trigger_cycle_confidence,
            trigger_cycle_applicability=str_or_none(r.trigger_cycle_applicability),
            trigger_cycle_reason=str_or_none(r.trigger_cycle_reason),
            leak_valid=r.leak_valid,
            leak_valid_reason=str_or_none(r.leak_valid_reason),
            ramp_active=r.ramp_active,
            ramp_active_reason=str_or_none(r.ramp_active_reason),
            mask_off=r.mask_off,
            mask_off_reason=str_or_none(r.mask_off_reason),
        )
        for r in dto.rows
    ]

    # Map aggregated bins — tidal_volume_median → tidal_volume_median_ml.
    bins = [
        BreathTableBin(
            session_start_wall_clock=localize_wall_clock(
                b.session_start_wall_clock, b.timezone_status, b.timezone_name
            ),
            **tz_fields(b),
            bin_start_offset=b.bin_start_offset,
            bin_end_offset=b.bin_end_offset,
            breath_count=b.breath_count,
            flatness_index_median=b.flatness_index_median,
            mid_insp_flattening_median=b.mid_insp_flattening_median,
            flow_class_mode=b.flow_class_mode,
            tidal_volume_median_ml=b.tidal_volume_median,
            ie_ratio_median=b.ie_ratio_median,
            leak_valid_fraction=b.leak_valid_fraction,
            analysis_status=str(b.analysis_status),
        )
        for b in dto.bins
    ]

    # Resolve device_id for the capabilities block.
    # Priority: explicit arg → dto.session_id via shared helper (fix 4 guarantees it's set).
    resolved_device_id: int | None = device_id
    if resolved_device_id is None and dto.session_id is not None:
        resolved_device_id = await get_device_id_for_session(
            db_session, dto.session_id, profile_id
        )

    caps = (
        await build_device_capabilities(
            db_session,
            profile_id,
            resolved_device_id,
            date_start=therapy_date,
            date_end=therapy_date,
            # An OK page proves analysis exists for this profile — skip 3-join count query.
            analysis_run=True if dto.analysis_status == AnalysisStatus.OK else None,
        )
        if resolved_device_id is not None
        else None
    )

    # Response-level anchor label — cached in the service instance, so this
    # re-uses the resolution already done inside get_breath_table.
    tz = await bs.resolve_timezone()

    return BreathTableResponse(
        query=mapped_query,
        session_id=top_session_id,
        session_start_wall_clock=top_session_start,
        **tz_fields(tz),
        analysis_status=str(dto.analysis_status),
        algo_versions=(
            dto.algo_versions.model_dump(mode="json")
            if dto.algo_versions is not None
            else None
        ),
        null_reason=str(dto.null_reason) if dto.null_reason is not None else None,
        is_binned=dto.is_binned,
        total_breaths=dto.total_breaths,
        page=dto.page,
        page_size=dto.page_size,
        rows=rows,
        bins=bins,
        device_capabilities=caps,
    )


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import parse_date  # noqa: PLC0415

    @mcp.tool()
    @_with_experimental_disclaimer
    @tool_error_boundary
    async def get_breath_table(
        ctx: Context,
        date: str,
        offset_start: float,
        offset_end: float,
        device_id: int | None = None,
        session_id: int | None = None,
        page: int = 1,
        page_size: int = 500,
        bin_minutes: float | None = None,
    ) -> dict[str, Any]:
        """Paginated breath-level table for a single therapy night.

        Use this tool to inspect individual breath waveform features (flow class,
        flattening index, timing, tidal volume) within a time window of a therapy session.
        Call ``get_data_overview`` first to confirm analysis has been run; this tool
        requires breath-level analysis results.

        Raw windows are capped at 15 minutes (offset_end - offset_start ≤ 900 s).
        For longer windows set ``bin_minutes`` to aggregate into time bins — the response
        then populates ``bins`` instead of ``rows``.  ``page_size`` must be between 1 and 2000.

        Args:
            date: Session date in YYYY-MM-DD format.
            offset_start: Window start in seconds from session start (≥ 0).
            offset_end: Window end in seconds from session start (> offset_start).
                        Raw window must be ≤ 900 s unless ``bin_minutes`` is set.
            device_id: Filter to a specific device.  Required when multiple devices
                       have data for the same date.
            session_id: Filter to a specific session.  Required when the device had
                        multiple sessions on the date.  Pass ``device_id`` too when
                        both are given to validate consistency.
            page: Page number for raw rows (1-based, default 1).
            page_size: Rows per page for raw fetch (default 500, must be between 1 and 2000).
            bin_minutes: When set (≥ 1.0), aggregate breaths into bins of this width
                         instead of returning raw rows.  Required for windows > 15 min.

        Returns:
            BreathTableResponse.  ``is_binned`` indicates raw vs binned mode.
            ``analysis_status`` / ``null_reason`` describe coverage.
            ``device_capabilities`` describes what the device records.

        Refusal semantics:
            ``analysis_status`` is one of ``"ok"`` (results present),
            ``"not_run"`` (no analysis run), or ``"stale"`` (source data
            changed since analysis ran). When ``"not_run"`` or ``"stale"``,
            ``null_reason`` explains why (``"analysis_not_run"``,
            ``"analysis_stale"``). These are successful responses with
            ``total_breaths=0``, not tool errors.

        Error conditions:
            - No sessions found for date → tool error; use ``get_data_overview``.
            - Multiple devices on date and no ``device_id`` → tool error listing device IDs.
            - Multiple sessions on date and no ``session_id`` → tool error listing session IDs.
            - Raw window > 15 min without ``bin_minutes`` → tool error; set ``bin_minutes``.
            - Breath-level tables missing → tool error; run ``snore analysis run``.
        """
        from snore.mcp.tools.breath_table import (
            get_breath_table as _impl,  # noqa: PLC0415
        )

        therapy_date = parse_date(date, "date")
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_breath_table",
            therapy_date=therapy_date,
            device_id=device_id,
            session_id=session_id,
            offset_start=offset_start,
            offset_end=offset_end,
            page=page,
            page_size=page_size,
            bin_minutes=bin_minutes,
        )
