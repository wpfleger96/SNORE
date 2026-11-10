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
        """Return one night's Apple Health sleep-stage timeline, as the web Apple Health night page charts it.

        ``stages`` lists the sleep-analysis samples of the night's preferred
        source (the ``apple_health.preferred_source`` value ``get_night_detail``
        reports for the same date) by start time, as recorded (device data).
        Each one's ``value_text`` is the stage: ``InBed``, ``Awake``,
        ``AsleepCore``, ``AsleepDeep``, ``AsleepREM`` or
        ``AsleepUnspecified``; ``start_time``/``end_time`` bound it and
        ``source_name`` names the app or device that wrote it. Samples can
        overlap: an ``InBed`` sample, when the source wrote one, spans the
        stages within it. Time in bed is ``get_night_detail``'s
        ``apple_health.time_in_bed_seconds``.

        The night's Apple summary (stage totals, efficiency, SpO2, respiratory
        rate, breathing disturbances) is in ``get_night_detail``'s
        ``apple_health``. Times are Apple Health local wall-clock with no UTC
        offset. The tool errors with "No Apple Health sleep data found" when
        the date has no Apple Health sleep samples, even if it has other
        Apple Health data such as SpO2.

        Args:
            date: Night date in YYYY-MM-DD format (samples before noon belong
                to the previous date).

        Returns:
            AppleSleepStagesResponse with date and stages.
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
