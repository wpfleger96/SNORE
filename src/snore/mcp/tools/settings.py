"""get_settings_timeline tool — RxTracker adapter."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.rx_tracker import TIMELINE_KEYS, RxTracker, changed_setting_keys
from snore.mcp.schemas import (
    DeviceCapabilities,
    SettingsEpoch,
    SettingsTimelineResponse,
)
from snore.mcp.tools._capabilities import _has_analysis, build_device_capabilities
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary


async def get_settings_timeline(
    db_session: AsyncSession,
    start: date,
    end: date,
    profile_id: int,
    device_id: int | None = None,
) -> SettingsTimelineResponse:
    """Return therapy settings epochs in [start, end].

    Each epoch covers a contiguous period of identical settings.  Changed keys
    are flagged on the epoch where the change first appears.  Uses only the
    generic RX_KEYS plus ``mask_type``; epochs split when the device-reported
    mask type changes; no vendor-specific branching (G4).

    Device capabilities are populated per distinct device_id across the filtered
    epochs, scoped through build_device_capabilities (profile-safe).
    """
    tracker = RxTracker(profile_id)
    all_periods = await tracker.get_history(db_session, keys=TIMELINE_KEYS)

    # Filter to requested date range and optional device
    filtered = [
        p
        for p in all_periods
        if p.end_date >= start
        and p.start_date <= end
        and (device_id is None or p.device_id == device_id)
    ]
    filtered.sort(
        key=lambda p: (p.device_id if p.device_id is not None else -1, p.start_date)
    )

    # Pre-compute analysis status once to avoid re-running the three-join query
    # per device inside build_device_capabilities.
    analysis_run = await _has_analysis(db_session, profile_id)

    # Build capabilities once per real device id across the filtered epochs
    caps_cache: dict[int, DeviceCapabilities | None] = {}
    for period in filtered:
        dev_id = period.device_id
        if dev_id is not None and dev_id not in caps_cache:
            caps_cache[dev_id] = await build_device_capabilities(
                db_session,
                profile_id,
                dev_id,
                date_start=start,
                date_end=end,
                analysis_run=analysis_run,
            )

    epochs: list[SettingsEpoch] = []
    prev_settings: dict[int, dict[str, str | None]] = {}

    for period in filtered:
        dev_id = period.device_id
        raw = period.settings

        # Restrict to the timeline keys; absent keys become None
        settings: dict[str, str | None] = {k: raw.get(k) for k in TIMELINE_KEYS}

        # Determine which keys changed vs. previous epoch for this device
        key = dev_id if dev_id is not None else -1
        prev = prev_settings.get(key, {})
        changed_keys = sorted(changed_setting_keys(prev, settings))
        prev_settings[key] = settings

        # Clip epoch boundaries to requested range
        epoch_start = max(period.start_date, start)
        epoch_end = min(period.end_date, end)
        nights = (epoch_end - epoch_start).days + 1

        epochs.append(
            SettingsEpoch(
                start_date=epoch_start,
                end_date=epoch_end,
                nights=nights,
                settings=settings,
                changed_keys=changed_keys if prev else [],
                device_id=dev_id,
            )
        )

    caps_by_device: dict[str, DeviceCapabilities] = {
        str(dev_id): caps for dev_id, caps in caps_cache.items() if caps is not None
    }
    return SettingsTimelineResponse(
        epochs=epochs,
        total_epochs=len(epochs),
        device_capabilities_by_device=caps_by_device,
    )


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import parse_date_range  # noqa: PLC0415

    @mcp.tool()
    @tool_error_boundary
    async def get_settings_timeline(
        ctx: Context,
        start: str,
        end: str,
        device_id: int | None = None,
    ) -> dict[str, Any]:
        """Return therapy settings epochs for a date range.

        Each epoch represents a contiguous period of identical settings.
        Changed keys are flagged on the epoch where the change first appears.
        Uses generic RX_KEYS (mode, epr_level, epr_mode, pressure_min,
        pressure_max, pressure_fixed, ipap, epap, ps) plus ``mask_type``:
        epochs split when the device-reported mask type changes.  No
        vendor-specific branching (G4).

        Each epoch's ``device_id`` is ``null`` (never ``0``) when no device is
        associated with the epoch.

        Args:
            start: Start date in YYYY-MM-DD format.
            end: End date in YYYY-MM-DD format.
            device_id: Optional device ID filter. Use get_data_overview to list devices.
        """
        from snore.mcp.tools.settings import (
            get_settings_timeline as _impl,  # noqa: PLC0415
        )

        start_d, end_d = parse_date_range(start, end)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_settings_timeline",
            start=start_d,
            end=end_d,
            device_id=device_id,
        )
