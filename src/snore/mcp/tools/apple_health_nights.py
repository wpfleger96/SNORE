"""get_apple_health_nights: the nights that have Apple Health sleep data.

Serves ``HealthService.list_nights`` (``GET /health/nights``), the list behind
the web Apple Health page.  The query is profile-scoped by profile_id.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context
from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.schemas import AppleHealthNightsResponse
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.mcp.validation import parse_open_date_range, validate_page_args
from snore.services.health_service import HealthService

if TYPE_CHECKING:
    from fastmcp import FastMCP


async def get_apple_health_nights(
    db_session: AsyncSession,
    profile_id: int,
    start: date | None = None,
    end: date | None = None,
    page: int = 1,
    page_size: int = 30,
) -> AppleHealthNightsResponse:
    """Return one page of the profile's Apple Health nights in range, newest first."""
    nights, total = await HealthService(db_session, profile_id).list_nights(
        from_date=start,
        to_date=end,
        limit=page_size,
        offset=(page - 1) * page_size,
    )
    return AppleHealthNightsResponse(
        nights=nights, total_nights=total, page=page, page_size=page_size
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @tool_error_boundary
    async def get_apple_health_nights(
        ctx: Context,
        start: str | None = None,
        end: str | None = None,
        page: int = 1,
        page_size: int = 30,
    ) -> dict[str, Any]:
        """List the nights that have Apple Health sleep data, newest first, as the web Apple Health page does.

        Each row is the night's Apple sleep summary: ``night_date``, the
        ``preferred_source`` its totals come from, time in bed, total sleep,
        core, deep, REM, awake and unspecified-stage seconds, sleep efficiency
        and stage coverage, and ``computed_at``, when SNORE built the summary.
        A night is listed only when it has Apple Health sleep samples; a date
        with only other Apple data (such as SpO2) is not. Paginated at 30
        nights per call (adjustable); ``total_nights`` counts every night in
        range. ``get_data_overview``'s ``apple_health`` gives the first and
        last night dates and the night count.

        A night's SpO2, respiratory rate and breathing disturbances are in
        ``get_night_detail``'s ``apple_health``, and its sleep-stage timeline
        is ``get_apple_sleep_stages``. Night dates use the same noon split
        as CPAP nights (samples starting before noon belong to the previous
        date), so an Apple night pairs with the device night of the same date.

        Omit ``start`` and/or ``end`` for an open range (all nights by default).

        Args:
            start: Optional first night date (YYYY-MM-DD), inclusive.
            end: Optional last night date (YYYY-MM-DD), inclusive.
            page: Page number (1-based). Default 1.
            page_size: Nights per page (must be between 1 and 90). Default 30.

        Returns:
            AppleHealthNightsResponse with nights and pagination.
        """
        from snore.mcp.tools.apple_health_nights import (  # noqa: PLC0415
            get_apple_health_nights as _impl,
        )

        start_d, end_d = parse_open_date_range(start, end)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_apple_health_nights",
            start=start_d,
            end=end_d,
            page=page,
            page_size=validate_page_args(page, page_size),
        )
