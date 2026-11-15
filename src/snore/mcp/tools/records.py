"""get_records: best and worst nights per metric, as the web Stats page's Records.

Serves ``StatsService.get_records`` (``GET /stats/records``), which reads Day
rows (profile-scoped through Device.profile_id) and Apple Health nights (by
profile_id); no waveform, analysis or breath data.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Annotated, Any

from fastmcp import Context
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession

from snore.mcp.schemas import NightRecordsResponse
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.mcp.validation import parse_open_date_range
from snore.services.stats_service import StatsService

if TYPE_CHECKING:
    from fastmcp import FastMCP

MAX_TOP_N = 50


async def get_records(
    db_session: AsyncSession,
    profile_id: int,
    top_n: int = 5,
    start: date | None = None,
    end: date | None = None,
) -> NightRecordsResponse:
    """Return each metric's ``top_n`` best and worst nights in the range."""
    records = await StatsService(db_session, profile_id).get_records(
        top_n=top_n, from_date=start, to_date=end
    )
    return NightRecordsResponse.model_validate(records, from_attributes=True)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @tool_error_boundary
    async def get_records(
        ctx: Context,
        top_n: Annotated[int, Field(ge=1, le=MAX_TOP_N)] = 5,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Return the best and worst nights per metric, as the web Stats page's Records show them.

        Each list holds up to ``top_n`` ``[night date, value]`` pairs; a night can
        appear in both lists, and equal values have no guaranteed order. Lowest is best
        for ``ahi`` and ``leak``; highest is best for ``therapy_hours``, ``spo2_min``
        and ``total_sleep_hours``. ``ahi`` is the nightly headline AHI: device-reported
        when trusted, else SNORE's recount; the lists can mix both. Device metrics rank
        device nights with at least 1 h of therapy from enabled sessions, all devices
        together and unlabelled, so a date can appear twice. ``total_sleep_hours`` ranks
        every Apple Health night with a total sleep time. A metric with no qualifying
        night is null.

        Omit ``start`` and/or ``end`` for an open range.

        Args:
            top_n: Nights per list, 1-50. Default 5, as the Stats page shows.
            start: Optional first night date (YYYY-MM-DD), inclusive.
            end: Optional last night date (YYYY-MM-DD), inclusive.
        """
        from snore.mcp.tools.records import (  # noqa: PLC0415
            get_records as _impl,
        )

        start_d, end_d = parse_open_date_range(start, end)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_records",
            top_n=top_n,
            start=start_d,
            end=end_d,
        )
