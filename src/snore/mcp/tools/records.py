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

        Each metric holds ``best`` and ``worst`` lists of ``[date, value]``
        pairs, up to ``top_n`` each (fewer when fewer nights have the metric),
        best first and worst first; with fewer than ``2 * top_n`` nights a
        night can appear in both lists. Dates are night dates. Equal values
        come in no guaranteed order.

        - ``ahi``: lowest is best. The nightly headline AHI, device-reported
          when trusted, else SNORE's recount; the lists can mix both.
        - ``leak``: nightly median leak (L/min); lowest is best.
        - ``therapy_hours``: nightly therapy hours; highest is best.
        - ``spo2_min``: nightly minimum SpO2 (%); highest is best.
        - ``total_sleep_hours``: Apple Health total sleep per night, rounded
          to 0.01 h; highest is best.

        The device metrics rank only device nights with at least 1 h of
        therapy, each device night judged on its own therapy hours. Disabled
        sessions count toward no device night, so a device night with only
        disabled sessions never qualifies. Every device of the profile is
        ranked together, one entry per device night, so a date can appear
        twice when two devices recorded it; entries do not name the device.
        A night without a value for a metric is skipped for that metric only.
        ``total_sleep_hours`` ranks every Apple Health night that has a total
        sleep time, therapy or not. A metric is null when no night in range
        qualifies; with no data in range every metric is null. Values come
        from nightly summaries only; no analysis data is read.

        Omit ``start`` and/or ``end`` for an open range (all data by default).

        Args:
            top_n: Nights per list, 1-50. Default 5, as the Stats page shows.
            start: Optional first night date (YYYY-MM-DD), inclusive.
            end: Optional last night date (YYYY-MM-DD), inclusive.

        Returns:
            NightRecordsResponse with one best/worst entry per metric.
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
