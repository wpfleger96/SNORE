"""get_data_overview tool — cold-start orientation."""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.mcp.schemas import DataOverviewResponse, DeviceInfo, tz_fields
from snore.mcp.tools._capabilities import build_device_capabilities
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.services.device_service import DeviceService


async def get_data_overview(
    db_session: AsyncSession, profile_id: int
) -> DataOverviewResponse:
    """Return a comprehensive overview of all imported data.

    Called by the get_data_overview MCP tool. Provides device inventory,
    date ranges, available waveform channels, event types, and analysis status
    — everything an LLM needs to orient itself to a cold database.

    All queries are scoped through Device.profile_id so no cross-profile data
    leaks into the response.
    """
    from snore.services.breath_service import BreathService  # noqa: PLC0415

    device_svc = DeviceService(db_session, profile_id)
    raw_devices = await device_svc.list_devices()
    bs = BreathService(db_session, profile_id)
    tz = await bs.resolve_timezone()

    if not raw_devices:
        return DataOverviewResponse(devices=[], **tz_fields(tz))

    # Analysis status — run once; derive both the bool flag and the count from
    # the same query to avoid two round-trips over the same three-join path.
    analysis_session_count = (
        await db_session.execute(
            select(func.count(models.AnalysisResult.session_id.distinct()))
            .join(
                models.Session,
                models.AnalysisResult.session_id == models.Session.id,
            )
            .join(
                models.Device,
                models.Session.device_id == models.Device.id,
            )
            .where(
                models.Device.profile_id == profile_id,
                models.Session.enabled.is_(True),
            )
        )
    ).scalar_one()
    analysis_run = analysis_session_count > 0

    device_ids = [d.id for d in raw_devices]

    # Bulk session stats — one query over all devices instead of N per-device queries.
    stats_rows = (
        await db_session.execute(
            select(
                models.Session.device_id,
                func.count(models.Session.id),
                func.min(models.Session.start_time),
                func.max(models.Session.start_time),
            )
            .where(
                models.Session.device_id.in_(device_ids),
                models.Session.enabled.is_(True),
            )
            .group_by(models.Session.device_id)
        )
    ).all()
    stats_by_device: dict[int, tuple[int, datetime | None, datetime | None]] = {
        int(row[0]): (row[1], row[2], row[3]) for row in stats_rows
    }

    # Bulk therapy modes — one query; Python groups by device_id.
    modes_rows = (
        await db_session.execute(
            select(models.Session.device_id, models.Session.therapy_mode)
            .where(
                models.Session.device_id.in_(device_ids),
                models.Session.enabled.is_(True),
                models.Session.therapy_mode.is_not(None),
            )
            .distinct()
            .order_by(models.Session.device_id, models.Session.therapy_mode)
        )
    ).all()
    modes_by_device: dict[int, list[str]] = {}
    for dev_id, mode in modes_rows:
        if mode is not None:  # excluded by the is_not(None) filter
            modes_by_device.setdefault(int(dev_id), []).append(mode)

    device_infos: list[DeviceInfo] = []
    total_sessions = 0
    global_min_date: date | None = None
    global_max_date: date | None = None

    for d in raw_devices:
        count, min_dt, max_dt = stats_by_device.get(int(d.id), (0, None, None))

        first_date = min_dt.date() if min_dt else None
        last_date = max_dt.date() if max_dt else None

        if first_date and (global_min_date is None or first_date < global_min_date):
            global_min_date = first_date
        if last_date and (global_max_date is None or last_date > global_max_date):
            global_max_date = last_date

        total_sessions += count or 0

        therapy_modes = modes_by_device.get(int(d.id), [])

        # Device capabilities — profile-scoped; exceptions propagate (no swallowing).
        # analysis_run is pre-computed above to avoid re-running the three-join query
        # per device.  AsyncSession is not safe under asyncio.gather — sequential.
        dev_caps = await build_device_capabilities(
            db_session,
            profile_id,
            d.id,
            date_start=first_date,
            date_end=last_date,
            analysis_run=analysis_run,
        )

        device_infos.append(
            DeviceInfo(
                id=d.id,
                manufacturer=d.manufacturer,
                model=d.model,
                serial_number=d.serial_number,
                first_session_date=first_date,
                last_session_date=last_date,
                session_count=count or 0,
                therapy_modes=[m for m in therapy_modes if m],
                device_capabilities=dev_caps,
            )
        )

    # Available waveform channel types scoped to this profile via Session → Device
    waveform_types = (
        (
            await db_session.execute(
                select(models.Waveform.waveform_type)
                .join(
                    models.Session,
                    models.Waveform.session_id == models.Session.id,
                )
                .join(
                    models.Device,
                    models.Session.device_id == models.Device.id,
                )
                .where(
                    models.Device.profile_id == profile_id,
                    models.Session.enabled.is_(True),
                )
                .distinct()
                .order_by(models.Waveform.waveform_type)
            )
        )
        .scalars()
        .all()
    )

    # Available event types scoped to this profile via Session → Device
    event_types = (
        (
            await db_session.execute(
                select(models.Event.event_type)
                .join(
                    models.Session,
                    models.Event.session_id == models.Session.id,
                )
                .join(
                    models.Device,
                    models.Session.device_id == models.Device.id,
                )
                .where(
                    models.Device.profile_id == profile_id,
                    models.Session.enabled.is_(True),
                )
                .distinct()
                .order_by(models.Event.event_type)
            )
        )
        .scalars()
        .all()
    )

    return DataOverviewResponse(
        devices=device_infos,
        date_range_start=global_min_date,
        date_range_end=global_max_date,
        total_sessions=total_sessions,
        available_waveform_channels=list(waveform_types),
        available_event_types=list(event_types),
        analysis_run=analysis_run,
        analysis_session_count=analysis_session_count,
        **tz_fields(tz),
    )


def register(mcp: FastMCP) -> None:
    @mcp.tool()
    @tool_error_boundary
    async def get_data_overview(ctx: Context) -> dict[str, Any]:
        """Orient to the imported dataset: devices, date ranges, channels, analysis status.

        Call this first before any other tool. Returns everything needed to understand
        what data is available and which tools are applicable.

        Returns:
            DataOverviewResponse with devices, date ranges, waveform channels,
            event types, and analysis status.
        """
        from snore.mcp.tools.overview import get_data_overview as _impl  # noqa: PLC0415

        return await _scope_and_run(ctx, _impl, tool_name="get_data_overview")
