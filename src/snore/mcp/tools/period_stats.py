"""get_period_stats: per-period therapy statistics as the web Stats page shows them.

Serves ``StatsService.get_period_statistics`` (``GET /stats/periods``), which
is profile-scoped through Device.profile_id (Apple Health by profile_id).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.calculations import PeriodType
from snore.mcp.errors import ValidationError
from snore.mcp.schemas import PeriodStatsResponse
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.mcp.validation import (
    parse_date_range,
    parse_open_date_range,
    validate_nightly_span,
)
from snore.services.stats_service import StatsService

if TYPE_CHECKING:
    from fastmcp import FastMCP


async def get_period_stats(
    db_session: AsyncSession,
    profile_id: int,
    period_type: PeriodType = "month",
    start: date | None = None,
    end: date | None = None,
) -> PeriodStatsResponse:
    """Return the statistics of each period with therapy data, oldest first."""
    periods = await StatsService(db_session, profile_id).get_period_statistics(
        period_type, from_date=start, to_date=end
    )
    return PeriodStatsResponse(periods=periods)


def _parse_range(
    period_type: PeriodType, start: str | None, end: str | None
) -> tuple[date | None, date | None]:
    """Parse the bounds; a ``day`` range needs both and a capped span."""
    if period_type != "day":
        return parse_open_date_range(start, end)
    if start is None or end is None:
        raise ValidationError('period_type "day" requires both start and end')
    start_d, end_d = parse_date_range(start, end)
    validate_nightly_span(start_d, end_d)
    return start_d, end_d


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @tool_error_boundary
    async def get_period_stats(
        ctx: Context,
        period_type: PeriodType = "month",
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Return therapy statistics per day, week, month, 6 months or year, as the web Stats page shows them.

        ``periods`` holds one entry per period that has therapy data, oldest
        first; periods without any device night are omitted, so
        ``days_used`` can be less than ``days_in_period``. Periods are whole
        calendar periods (weeks start on Monday); when ``start`` or ``end``
        falls inside one, only the nights in range count. Every value is
        derived from the nightly device values across every device of the
        profile: counts, means, medians, minimums and pooled percentiles.
        ``avg_ahi``, ``avg_oai``, ``avg_cai`` and ``avg_hi`` average the
        nightly headline indices, which
        can mix device-reported and recounted nights. ``avg_rera`` is null
        when no device in the period has ever scored a RERA.
        ``device_flg_95th`` and ``snore_95th`` are the 95th percentiles of the
        device flow-limitation and snore channels over the period's pooled
        samples; null when no session has the channel. An open or long range
        decodes every session's flow-limitation and snore channel data, so
        bound ``start``/``end`` when the 95th percentiles over a shorter span
        are enough.
        ``avg_total_sleep_hours`` and ``avg_sleep_efficiency_pct`` average
        the period's Apple Health nights; null when it has none.

        Omit ``start`` and/or ``end`` for an open range (all data by default).
        ``period_type="day"`` needs both, spanning at most 90 nights; for
        nightly detail prefer ``get_nightly_summary``. No analysis data is read;
        SNORE's experimental metrics are in the ``experimental_*`` tools.

        Args:
            period_type: One of "day", "week", "month", "6month", "year".
                Default "month".
            start: Optional first date (YYYY-MM-DD), inclusive.
            end: Optional last date (YYYY-MM-DD), inclusive.

        Returns:
            PeriodStatsResponse with the periods list.
        """
        from snore.mcp.tools.period_stats import (  # noqa: PLC0415
            get_period_stats as _impl,
        )

        start_d, end_d = _parse_range(period_type, start, end)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_period_stats",
            period_type=period_type,
            start=start_d,
            end=end_d,
        )
