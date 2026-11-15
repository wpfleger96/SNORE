"""get_apple_sleep_stages: one night's Apple Health sleep-stage samples.

Serves ``HealthService.get_night_samples`` (``GET /health/nights/{date}/samples``),
the samples behind the web Apple Health night page's sleep-stage chart.  The
query is profile-scoped by profile_id.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context
from sqlalchemy.ext.asyncio import AsyncSession

from snore.exceptions import NotFoundError
from snore.mcp.errors import ValidationError
from snore.mcp.schemas import AppleSleepStagesResponse
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.mcp.validation import parse_date
from snore.services.health_service import HealthService

if TYPE_CHECKING:
    from fastmcp import FastMCP


async def get_apple_sleep_stages(
    db_session: AsyncSession,
    night_date: date,
    profile_id: int,
) -> AppleSleepStagesResponse:
    """Return the night's preferred-source sleep-stage samples, ordered by start.

    Raises ValidationError when the date has no Apple Health sleep night.
    """
    try:
        stages = await HealthService(db_session, profile_id).get_night_samples(
            night_date
        )
    except NotFoundError:
        raise ValidationError(
            f"No Apple Health sleep data found for {night_date}"
        ) from None
    return AppleSleepStagesResponse(date=night_date, stages=stages)


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @tool_error_boundary
    async def get_apple_sleep_stages(
        ctx: Context,
        date: str,
    ) -> dict[str, Any]:
        """Return one night's Apple Health sleep-stage samples from its preferred source, by start time, as the web Apple Health night page charts them.

        ``value_text`` is the stage: ``InBed``, ``Awake``, ``AsleepCore``,
        ``AsleepDeep``, ``AsleepREM`` or ``AsleepUnspecified``. Samples can overlap: an
        ``InBed`` sample spans the stages within it. Times are local wall-clock with no
        UTC offset. Errors with "No Apple Health sleep data found" when the date has no
        sleep samples, even if it has other Apple Health data.

        Args:
            date: Night date (YYYY-MM-DD); samples before noon belong to the previous
                date.
        """
        from snore.mcp.tools.apple_sleep_stages import (  # noqa: PLC0415
            get_apple_sleep_stages as _impl,
        )

        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_apple_sleep_stages",
            night_date=parse_date(date, "date"),
        )
