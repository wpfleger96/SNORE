"""get_settings_changes tool — merged device-settings + mask-log change log."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastmcp import Context

if TYPE_CHECKING:
    from fastmcp import FastMCP
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.rx_tracker import RxTracker, merge_changes_with_mask_log
from snore.mcp.schemas import SettingsChangeEntry, SettingsChangesResponse
from snore.mcp.tools._scaffold import _scope_and_run, tool_error_boundary
from snore.services.mask_log_service import MaskLogService


async def get_settings_changes(
    db_session: AsyncSession,
    start: date,
    end: date,
    profile_id: int,
    device_id: int | None = None,
) -> SettingsChangesResponse:
    """Return a merged, source-tagged log of settings changes in [start, end].

    Combines RxTracker.get_changes (per-key device settings diffs across ALL
    persisted keys) with the profile's user-entered mask equipment log
    (key="mask_equipment").  Mask log entries are profile-level, so the
    optional device_id filter narrows only the device_settings entries.
    """
    # Device history is loaded only up to end; later days cannot affect earlier diffs.
    device_changes = (
        await RxTracker(profile_id).get_changes(db_session, end=end)
    ).changes
    # Full ordered log (start_date, id) so the first in-range entry's old_value
    # can come from the latest entry before the range — start-windowing stays in Python.
    mask_entries = await MaskLogService(db_session, profile_id).list_entries()
    merged = merge_changes_with_mask_log(
        device_changes, mask_entries, start, end, device_id
    )
    entries = [
        SettingsChangeEntry.model_validate(c, from_attributes=True) for c in merged
    ]
    return SettingsChangesResponse(changes=entries, total_changes=len(entries))


def register(mcp: FastMCP) -> None:
    from snore.mcp.validation import parse_date_range  # noqa: PLC0415

    @mcp.tool()
    @tool_error_boundary
    async def get_settings_changes(
        ctx: Context,
        start: str,
        end: str,
        device_id: int | None = None,
    ) -> dict[str, Any]:
        """Return a merged, date-ordered log of every settings change in a date range.

        ``source="device_settings"`` entries diff each device's stored settings
        day-over-day across all keys, comfort settings such as mask_type and
        humidity_level included. ``source="mask_log"`` entries are user-logged mask
        equipment changes (``key="mask_equipment"``); ``old_value`` is null for the
        first one logged. Mask log entries are profile-level, so ``device_id`` never
        filters them out.

        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            device_id: Optional device ID filter for device_settings entries.
        """
        from snore.mcp.tools.changes import (
            get_settings_changes as _impl,  # noqa: PLC0415
        )

        start_d, end_d = parse_date_range(start, end)
        return await _scope_and_run(
            ctx,
            _impl,
            tool_name="get_settings_changes",
            start=start_d,
            end=end_d,
            device_id=device_id,
        )
