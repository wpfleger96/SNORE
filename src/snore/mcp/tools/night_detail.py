"""get_night_detail: one night as the web Day page shows it.

Combines ``DayService.get_day`` (the device night), each of its sessions'
Statistics row (what the Session page shows, without decoding waveforms) and
``HealthService.get_night_detail`` (the Apple Health night).  Every query is
profile-scoped through Device.profile_id (Apple Health by profile_id).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.exceptions import NotFoundError
from snore.mcp.errors import ValidationError
from snore.mcp.schemas import NightDetailResponse, NightSession
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.services.day_service import DayService
from snore.services.event_service import device_reported_types
from snore.services.health_service import HealthService
from snore.services.schemas import DayDetail
from snore.services.session_service import session_statistics

if TYPE_CHECKING:
    from fastmcp import FastMCP


async def get_night_detail(
    db_session: AsyncSession,
    night_date: date,
    profile_id: int,
    device_id: int | None = None,
) -> NightDetailResponse:
    """Return the device night, its sessions and the Apple Health night for a date.

    Raises ValidationError when the date has neither a device night nor an
    Apple Health night.
    """
    try:
        day = await DayService(db_session, profile_id).get_day(night_date, device_id)
    except NotFoundError:
        day = None
    try:
        apple = await HealthService(db_session, profile_id).get_night_detail(night_date)
    except NotFoundError:
        apple = None
    if day is None and apple is None:
        raise ValidationError(f"No therapy or Apple Health data found for {night_date}")

    return NightDetailResponse(
        date=night_date,
        # The Apple summary in health_sleep is a subset of apple_health.
        device=day.model_copy(update={"health_sleep": None}) if day else None,
        sessions=await _night_sessions(db_session, profile_id, day) if day else [],
        apple_health=apple,
    )


async def _night_sessions(
    db_session: AsyncSession, profile_id: int, day: DayDetail
) -> list[NightSession]:
    """The day's sessions with their Statistics rows, ascending by start."""
    rows = (
        await db_session.execute(
            select(models.Session, models.Statistics)
            .join(models.Device, models.Session.device_id == models.Device.id)
            .outerjoin(
                models.Statistics, models.Statistics.session_id == models.Session.id
            )
            .where(
                models.Session.id.in_(day.session_ids),
                models.Device.profile_id == profile_id,
            )
            .order_by(models.Session.start_time)
        )
    ).all()
    reported = await device_reported_types(db_session, day.device_id)
    return [
        NightSession(
            id=session.id,
            start_time=session.start_time,
            end_time=session.end_time,
            enabled=session.enabled,
            therapy_mode=session.therapy_mode,
            statistics=(
                session_statistics(stats, session.import_source, reported)
                if stats is not None
                else None
            ),
        )
        for session, stats in rows
    ]


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import parse_date  # noqa: PLC0415

    @mcp.tool()
    @tool_error_boundary
    async def get_night_detail(
        ctx: Context,
        date: str,
        device_id: int | None = None,
    ) -> dict[str, Any]:
        """Return one night as the web Day page shows it: the device night, its sessions' device statistics, and the Apple Health night.

        ``device`` is the device night (null when the date has none): headline
        ``ahi``/``oai``/``cai``/``hi`` are the device-reported daily value when
        trusted, otherwise SNORE's recount; ``index_source`` (``"device"`` or
        ``"derived"``) says which, and the recount, when computable, is in
        ``*_computed``. Event counts are device-scored; ``reras`` and
        ``unclassified_apneas`` are null, not zero, when the device has never
        scored one. Pressure, EPAP, leak and SpO2 values, ``channel_stats``
        (device flow-limitation and snore channels, pooled over enabled
        sessions) and ``mask_on_spans`` are derived from device data.
        ``device.health_sleep`` is always null: the Apple night is in
        ``apple_health``.

        ``sessions`` lists every session of that device night, enabled and
        disabled, by start time; only enabled sessions feed the night's values.
        Each session's ``statistics`` (null when it has none) is the stored
        device summary: device-reported values such as ``ahi_device``,
        ``rin``, ``csr_pct`` and the humidifier, tube and ambient-humidity
        readings, plus pressure, leak, SpO2, pulse, respiratory-rate,
        tidal-volume (mL) and minute-ventilation statistics, some recomputed
        from the device waveforms (derived). Session ``ahi``/``oai``/``cai``/
        ``hi`` are SNORE's count of device-scored events per mask-on hour
        (derived). The response's ``provenance`` block lists every derived
        path.

        ``apple_health`` is the Apple Health night (null when the date has
        none): sleep stages, efficiency, SpO2 and respiratory rate, derived
        from Apple Health samples.

        Times are device wall-clock with no UTC offset. When several devices
        have a night on this date and ``device_id`` is omitted, the night of
        the lowest ``device_id`` is returned (``device.device_id`` says
        which); pass ``device_id`` to choose. The tool errors only when the
        date has neither a device night nor an Apple Health night. No analysis
        data is read; SNORE's experimental metrics are in the
        ``experimental_*`` tools.

        Args:
            date: Night date in YYYY-MM-DD format (sessions before noon belong
                to the previous date).
            device_id: Optional device ID.

        Returns:
            NightDetailResponse with date, device, sessions and apple_health.
        """
        from snore.mcp.tools.night_detail import (  # noqa: PLC0415
            get_night_detail as _impl,
        )

        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_night_detail",
            night_date=parse_date(date, "date"),
            device_id=device_id,
        )
