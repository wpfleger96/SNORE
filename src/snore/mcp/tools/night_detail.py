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
from snore.mcp.tools._service_errors import (
    MAPPED_SERVICE_ERRORS,
    raise_mapped_service_error,
)
from snore.services.breath.dtos import DeviceAmbiguityError
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

    Raises ValidationError when ``device_id`` has no night on the date, when
    ``device_id`` is omitted and several devices have one, and when the date
    has neither a device night nor an Apple Health night.
    """
    day_service = DayService(db_session, profile_id)
    day: DayDetail | None = None
    if device_id is not None:
        try:
            day = await day_service.get_day(night_date, device_id)
        except NotFoundError:
            # Same message for a foreign and a nonexistent device_id.
            raise ValidationError(
                f"device_id={device_id} has no night on {night_date} in this session"
            ) from None
    else:
        try:
            night_device_id = await _only_night_device_id(
                db_session, profile_id, night_date
            )
        except MAPPED_SERVICE_ERRORS as exc:
            raise_mapped_service_error(exc)
        if night_device_id is not None:
            day = await day_service.get_day(night_date, night_device_id)
    try:
        apple = await HealthService(db_session, profile_id).get_night_detail(night_date)
    except NotFoundError:
        apple = None
    if day is None and apple is None:
        raise ValidationError(f"No therapy or Apple Health data found for {night_date}")

    return NightDetailResponse(
        date=night_date,
        device=day,
        sessions=await _night_sessions(db_session, profile_id, day) if day else [],
        apple_health=apple,
    )


async def _only_night_device_id(
    db_session: AsyncSession, profile_id: int, night_date: date
) -> int | None:
    """The one profile device with a Day on the date, or None when none has.

    Raises DeviceAmbiguityError when several have one.
    """
    rows = (
        await db_session.execute(
            select(models.Device.id, models.Device.serial_number)
            .join(models.Day, models.Day.device_id == models.Device.id)
            .where(
                models.Device.profile_id == profile_id, models.Day.date == night_date
            )
            .order_by(models.Device.id)
        )
    ).all()
    if len(rows) > 1:
        raise DeviceAmbiguityError(
            therapy_date=night_date,
            profile_id=profile_id,
            owned_device_ids=[row.id for row in rows],
            device_serials={row.id: row.serial_number or "" for row in rows},
        )
    return rows[0].id if rows else None


async def _night_sessions(
    db_session: AsyncSession, profile_id: int, day: DayDetail
) -> list[NightSession]:
    """The day's sessions with their Statistics rows, ascending by start.

    Only the served Session columns are selected: the JSON columns decode
    through ``ValidatedJSON``, which raises on an invalid stored value.
    """
    rows = (
        await db_session.execute(
            select(
                models.Session.id,
                models.Session.start_time,
                models.Session.end_time,
                models.Session.enabled,
                models.Session.therapy_mode,
                models.Session.import_source,
                models.Statistics,
            )
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
            id=row.id,
            start_time=row.start_time,
            end_time=row.end_time,
            enabled=row.enabled,
            therapy_mode=row.therapy_mode,
            statistics=(
                session_statistics(row.Statistics, row.import_source, reported)
                if row.Statistics is not None
                else None
            ),
        )
        for row in rows
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
        (median, 95th and max of the device flow-limitation, snore,
        respiratory-rate, tidal-volume (mL), minute-ventilation, Ti and I:E
        channels and the oximeter pulse, pooled over enabled sessions) and
        ``mask_on_spans`` are derived from device data.
        ``device`` matches ``GET /days/{date}``: ``device.health_sleep`` is the
        Day page's Apple Health summary, and ``apple_health`` is the fuller
        Apple night.

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
        none): the Day page's Apple summary (sleep stages, efficiency) plus
        SpO2 and respiratory rate derived from Apple Health samples, and
        Apple's breathing disturbances. The night's sleep-stage timeline is
        in ``get_apple_sleep_stages``.

        Times are device wall-clock with no UTC offset. The tool errors when
        several devices have a night on this date and ``device_id`` is omitted
        (the message lists them; pass ``device_id`` to choose), when
        ``device_id`` has no night on the date, and when the date has neither
        a device night nor an Apple Health night. No analysis data is read;
        SNORE's experimental metrics are in the ``experimental_*`` tools.

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
