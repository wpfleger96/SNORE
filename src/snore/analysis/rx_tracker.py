"""RX (prescription) change tracking and period analysis."""

import itertools
import logging

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from statistics import median

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload, selectinload

from snore.analysis.calculations import calculate_average_ahi
from snore.database.models import Day, Device
from snore.database.models import Session as SessionModel
from snore.services.schemas import (
    MaskLogEntryResponse,
    MergedSettingsChange,
    RxAllResponse,
    RxChangesResponse,
    RxComparisonResponse,
    RxPeriodResponse,
    RxSettingChange,
)
from snore.utils.stats import weighted_mean

logger = logging.getLogger(__name__)

RX_KEYS = (
    "mode",
    "epr_level",
    "epr_mode",
    "pressure_min",
    "pressure_max",
    "pressure_fixed",
    "ipap",
    "epap",
    "ps",
)

# Keys that define a settings-timeline epoch: the prescription keys plus the
# device-reported mask type.  humidity_level deliberately stays out — comfort
# tweaks should not split epochs.
TIMELINE_KEYS = (*RX_KEYS, "mask_type")

_TAIL_WALK_BATCH_SIZE = 90


def changed_setting_keys(
    prev: Mapping[str, str | None],
    curr: Mapping[str, str | None],
) -> set[str]:
    """Return the set of keys whose value differs between prev and curr.

    Uses union semantics: a key present in only one mapping (i.e. added or
    removed) is also reported.  Return type is a set so callers can choose
    their own ordering (sorted, RX_KEYS order, etc.).
    """
    return {k for k in set(prev) | set(curr) if prev.get(k) != curr.get(k)}


def _diff_settings(
    prev: dict[str, str], curr: dict[str, str]
) -> list[tuple[str, str | None, str | None]]:
    """Return (key, old_value, new_value) for every key whose value changed."""
    all_keys = sorted(set(prev) | set(curr))
    return [
        (k, prev.get(k), curr.get(k)) for k in all_keys if prev.get(k) != curr.get(k)
    ]


def _describe_mask(entry: MaskLogEntryResponse) -> str:
    """Return a human-readable mask name, handling nullable brand/model/style.

    Builds the name from the non-null parts of brand and model; falls back to
    style, then "unspecified mask".  Appends "(size)" when size is set.
    """
    name = (
        " ".join(p for p in (entry.brand, entry.model) if p)
        or entry.style
        or "unspecified mask"
    )
    return f"{name} ({entry.size})" if entry.size else name


def merge_changes_with_mask_log(
    device_changes: list[RxSettingChange],
    mask_entries: list[MaskLogEntryResponse],
    start: date,
    end: date,
    device_id: int | None = None,
) -> list[MergedSettingsChange]:
    """Merge device settings changes with mask log entries in [start, end].

    Device changes are windowed to the range; the optional device_id filter
    narrows only them — mask log entries are profile-level, so they are always
    included.  ``mask_entries`` must be the FULL (start_date, id)-ordered log,
    not pre-windowed, so the first in-range entry's old_value can come from
    the latest entry before the range (None for the first entry ever logged).

    Mask log entries with a null start_date are skipped and do not appear in
    the output or affect the prev_desc running state.

    Returns entries sorted by (date, source, device_id, key).
    """
    merged = [
        MergedSettingsChange(
            date=c.date,
            source="device_settings",
            device_id=c.device_id,
            device_name=c.device_name,
            key=c.key,
            old_value=c.old_value,
            new_value=c.new_value,
        )
        for c in device_changes
        if start <= c.date <= end and (device_id is None or c.device_id == device_id)
    ]

    prev_desc: str | None = None
    for entry in mask_entries:
        # Entries without a start_date cannot be placed on the timeline; skip
        # entirely so they don't affect prev_desc either.
        if entry.start_date is None:
            continue
        desc = _describe_mask(entry)
        if start <= entry.start_date <= end:
            merged.append(
                MergedSettingsChange(
                    date=entry.start_date,
                    source="mask_log",
                    key="mask_equipment",
                    old_value=prev_desc,
                    new_value=desc,
                    mask_brand=entry.brand,
                    mask_model=entry.model,
                    mask_size=entry.size,
                    mask_style=entry.style,
                    notes=entry.notes,
                )
            )
        prev_desc = desc

    merged.sort(
        key=lambda e: (
            e.date,
            e.source,
            e.device_id if e.device_id is not None else -1,
            e.key,
        )
    )
    return merged


@dataclass
class RxPeriod:
    """Therapy prescription period with consistent settings."""

    settings: dict[str, str]
    start_date: date
    end_date: date
    days: list[Day]
    device_id: int
    device_name: str


@dataclass
class RxPeriodStats(RxPeriod):
    """RX period with computed statistics."""

    avg_ahi: float | None = None
    median_ahi: float | None = None
    avg_hours: float | None = None
    total_hours: float = 0.0
    avg_leak: float | None = None


class RxTracker:
    """Track and analyze therapy prescription changes."""

    def __init__(self, profile_id: int) -> None:
        """Initialize the tracker scoped to a single profile.

        Args:
            profile_id: Profile to scope all device and day queries — required.
        """
        self.profile_id = profile_id

    async def get_history(
        self, db_session: AsyncSession, keys: tuple[str, ...] = RX_KEYS
    ) -> list[RxPeriodResponse]:
        """Return all RX periods with stats.

        Args:
            keys: Settings keys that define a period fingerprint.  Defaults
                to RX_KEYS (prescription-only periods — the /rx/* and CLI
                behavior); pass TIMELINE_KEYS to also split periods when the
                device-reported mask_type changes.
        """
        periods = await self._compute_periods(db_session, keys)
        stats = self._compute_period_stats(periods)
        return [self._to_response(p) for p in stats]

    async def get_current(self, db_session: AsyncSession) -> RxPeriodResponse | None:
        """Return the current (most recent) RX period, or None if no data.

        Invariant: for any DB state, result equals get_history()[-1] when
        non-empty — the period with max (start_date, device_id) across all
        devices.  Empty DB / no periods → None.  The invariant holds for the
        default RX_KEYS call only, not get_history(keys=TIMELINE_KEYS).
        """
        candidates: list[RxPeriod] = []
        for device_id, device_name in await self._get_devices(db_session):
            period = await self._find_last_period_for_device(
                db_session, device_id, device_name
            )
            if period is not None:
                candidates.append(period)

        if not candidates:
            return None

        best = max(candidates, key=lambda p: (p.start_date, p.device_id))
        stats = self._compute_period_stats([best])[0]
        return self._to_response(stats)

    async def get_comparison(
        self, db_session: AsyncSession, min_days: int = 7
    ) -> RxComparisonResponse:
        """Return all periods with best/worst indices."""
        periods = await self._compute_periods(db_session)
        stats = self._compute_period_stats(periods)
        best_index, worst_index = self._best_worst_indices(stats, min_days)
        return RxComparisonResponse(
            periods=[self._to_response(p) for p in stats],
            best_index=best_index,
            worst_index=worst_index,
        )

    async def get_changes(
        self, db_session: AsyncSession, end: date | None = None
    ) -> RxChangesResponse:
        """Return a log of every per-key settings change across all devices.

        Intentionally diffs ALL persisted settings keys — including comfort
        settings such as mask_type and humidity_level — not just the
        prescription-defining RX_KEYS.  This gives a complete audit trail of
        every setting the clinician or patient touched.  Do not narrow this to
        _get_day_period_settings; use _get_day_settings (no key filter).

        end: when supplied, device history is loaded only up to this date (Days
        with date > end are excluded at the SQL level).  The end bound is safe
        because later days cannot affect earlier diffs.  A start bound must NOT
        be pushed into SQL — change detection diffs consecutive days-with-settings
        and the first in-window change needs the last pre-window value;
        start-windowing stays in Python.
        """
        return self._compute_changes(await self._days_by_device(db_session, end=end))

    async def get_all(
        self, db_session: AsyncSession, min_days: int = 7
    ) -> RxAllResponse:
        """Return all RX data from a single database query."""
        device_groups = await self._days_by_device(db_session)

        periods = self._compute_periods_from_groups(device_groups)
        stats = self._compute_period_stats(periods)
        history = [self._to_response(p) for p in stats]

        best_index, worst_index = self._best_worst_indices(stats, min_days)

        return RxAllResponse(
            history=history,
            current=history[-1] if history else None,
            best_index=best_index,
            worst_index=worst_index,
            changes=self._compute_changes(device_groups),
        )

    def _to_response(self, period: RxPeriodStats) -> RxPeriodResponse:
        """Convert RxPeriodStats dataclass to RxPeriodResponse Pydantic model."""
        return RxPeriodResponse(
            settings=period.settings,
            start_date=period.start_date,
            end_date=period.end_date,
            days_count=len(period.days),
            avg_ahi=period.avg_ahi,
            median_ahi=period.median_ahi,
            avg_hours=period.avg_hours,
            total_hours=period.total_hours,
            avg_leak=period.avg_leak,
            device_id=period.device_id,
            device_name=period.device_name,
        )

    async def _compute_periods(
        self, db_session: AsyncSession, keys: tuple[str, ...] = RX_KEYS
    ) -> list[RxPeriod]:
        """Query all days and group into RX periods."""
        return self._compute_periods_from_groups(
            await self._days_by_device(db_session), keys
        )

    def _compute_periods_from_groups(
        self,
        device_groups: list[tuple[int, str, list[Day]]],
        keys: tuple[str, ...] = RX_KEYS,
    ) -> list[RxPeriod]:
        """Group consecutive days per device by therapy settings into RX periods."""
        all_periods: list[RxPeriod] = []
        for device_id, device_name, device_days in device_groups:
            all_periods.extend(
                self._compute_device_periods(device_days, device_id, device_name, keys)
            )
        all_periods.sort(key=lambda p: (p.start_date, p.device_id))
        return all_periods

    def _compute_changes(
        self, device_groups: list[tuple[int, str, list[Day]]]
    ) -> RxChangesResponse:
        """Diff ALL settings keys day-over-day per device."""
        all_changes: list[RxSettingChange] = []
        for device_id, device_name, device_days in device_groups:
            prev_settings: dict[str, str] | None = None
            for day in device_days:
                curr_settings = self._get_day_settings(day)
                if curr_settings is None:
                    continue
                if prev_settings is not None:
                    for key, old_val, new_val in _diff_settings(
                        prev_settings, curr_settings
                    ):
                        all_changes.append(
                            RxSettingChange(
                                date=day.date,
                                device_id=device_id,
                                device_name=device_name,
                                key=key,
                                old_value=old_val,
                                new_value=new_val,
                            )
                        )
                prev_settings = curr_settings
        all_changes.sort(key=lambda c: (c.date, c.device_id, c.key))
        return RxChangesResponse(changes=all_changes)

    async def _days_by_device(
        self, db_session: AsyncSession, end: date | None = None
    ) -> list[tuple[int, str, list[Day]]]:
        """Query all days grouped by device, ordered by (device_id, date).

        When end is supplied, only days with date <= end are included at the
        SQL level.  Callers that need the full history (get_all, _compute_periods)
        omit end and receive all days unchanged.

        Returns a list of (device_id, device_name, days) tuples, one per
        device, preserving the query order.  Devices whose Day rows have no
        matching Device row (device is None) are skipped with a warning.
        """
        q = (
            select(Day)
            .join(Device, Day.device_id == Device.id)
            .where(Device.profile_id == self.profile_id)
            .order_by(Day.device_id, Day.date)
            .options(
                joinedload(Day.sessions).joinedload(SessionModel.settings),
                joinedload(Day.device),
            )
        )
        if end is not None:
            q = q.where(Day.date <= end)
        days = (await db_session.execute(q)).unique().scalars().all()

        result: list[tuple[int, str, list[Day]]] = []
        for device_id, device_days_iter in itertools.groupby(
            days, key=lambda d: d.device_id
        ):
            device_days = list(device_days_iter)
            device = device_days[0].device
            if device is None:
                logger.warning(
                    "Skipping device_id=%s: no matching Device row", device_id
                )
                continue
            device_name = f"{device.manufacturer} {device.model}"
            result.append((device_id, device_name, device_days))

        return result

    def _compute_device_periods(
        self,
        days: list[Day],
        device_id: int,
        device_name: str,
        keys: tuple[str, ...] = RX_KEYS,
    ) -> list[RxPeriod]:
        """Run the fingerprint-grouping loop for a single device's days in date order."""
        periods: list[RxPeriod] = []
        current_settings: dict[str, str] | None = None
        current_fingerprint: str | None = None
        current_period_days: list[Day] = []
        current_start_date: date | None = None

        for day in days:
            day_settings = self._get_day_period_settings(day, keys)

            if day_settings is None:
                continue

            fingerprint = self._build_fingerprint(day_settings)

            if fingerprint != current_fingerprint:
                if current_period_days and current_start_date and current_settings:
                    periods.append(
                        RxPeriod(
                            settings=current_settings,
                            start_date=current_start_date,
                            end_date=current_period_days[-1].date,
                            days=current_period_days,
                            device_id=device_id,
                            device_name=device_name,
                        )
                    )

                current_settings = day_settings
                current_fingerprint = fingerprint
                current_period_days = [day]
                current_start_date = day.date
            else:
                current_period_days.append(day)

        if current_period_days and current_start_date and current_settings:
            periods.append(
                RxPeriod(
                    settings=current_settings,
                    start_date=current_start_date,
                    end_date=current_period_days[-1].date,
                    days=current_period_days,
                    device_id=device_id,
                    device_name=device_name,
                )
            )

        return periods

    def _compute_period_stats(self, periods: list[RxPeriod]) -> list[RxPeriodStats]:
        """
        Compute statistics for each RX period.

        Returns:
            List of RxPeriodStats with computed metrics.
        """
        stats_periods: list[RxPeriodStats] = []

        for period in periods:
            ahi_values = [d.ahi for d in period.days if d.ahi is not None]
            median_ahi = median(ahi_values) if ahi_values else None
            avg_ahi = calculate_average_ahi(period.days)
            avg_leak = weighted_mean(
                (d.leak_median, d.total_therapy_hours)
                for d in period.days
                if d.leak_median is not None
                and d.total_therapy_hours is not None
                and d.total_therapy_hours > 0
            )

            total_hours = sum(d.total_therapy_hours or 0 for d in period.days)
            avg_hours = total_hours / len(period.days) if period.days else None

            stats_periods.append(
                RxPeriodStats(
                    settings=period.settings,
                    start_date=period.start_date,
                    end_date=period.end_date,
                    days=period.days,
                    device_id=period.device_id,
                    device_name=period.device_name,
                    avg_ahi=avg_ahi,
                    median_ahi=median_ahi,
                    avg_hours=avg_hours,
                    total_hours=total_hours,
                    avg_leak=avg_leak,
                )
            )

        return stats_periods

    def _best_worst_indices(
        self, periods: list[RxPeriodStats], min_days: int = 7
    ) -> tuple[int | None, int | None]:
        """Return (best_index, worst_index) by average AHI among eligible periods."""
        eligible = [
            (i, p.avg_ahi)
            for i, p in enumerate(periods)
            if len(p.days) >= min_days and p.avg_ahi is not None
        ]
        if not eligible:
            return (None, None)

        best_idx = min(eligible, key=lambda t: t[1])[0]
        worst_idx = max(eligible, key=lambda t: t[1])[0]
        return (best_idx, worst_idx)

    def _get_day_settings(
        self, day: Day, key_filter: tuple[str, ...] | None = None
    ) -> dict[str, str] | None:
        """Extract settings from the longest enabled session of a day.

        Args:
            day: The Day ORM object with sessions pre-loaded.
            key_filter: If given, only these keys are included in the result.
                        If None, all keys with non-null values are returned.
        """
        enabled_sessions = [s for s in day.sessions if s.enabled]
        if not enabled_sessions:
            return None

        longest_session = max(
            enabled_sessions,
            key=lambda s: s.duration_seconds if s.duration_seconds else 0.0,
        )

        if not longest_session.settings:
            return None

        settings_dict: dict[str, str] = {}
        for setting in longest_session.settings:
            if setting.value is not None:
                if key_filter is None or setting.key in key_filter:
                    settings_dict[setting.key] = setting.value

        return settings_dict if settings_dict else None

    def _get_day_period_settings(
        self, day: Day, keys: tuple[str, ...] = RX_KEYS
    ) -> dict[str, str] | None:
        """Extract period-defining settings from the longest enabled session of a day."""
        return self._get_day_settings(day, key_filter=keys)

    def _build_fingerprint(self, settings: dict[str, str]) -> str:
        """Build a fingerprint string from RX settings for comparison."""
        sorted_items = sorted(settings.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)

    async def _get_devices(self, db_session: AsyncSession) -> list[tuple[int, str]]:
        """Return (device_id, device_name) for all devices that have Day rows.

        Uses an inner join so Day rows with no matching Device are naturally
        excluded — same outcome as the missing-device guard in _days_by_device.
        """
        rows = (
            await db_session.execute(
                select(Day.device_id, Device.manufacturer, Device.model)
                .join(Device, Day.device_id == Device.id)
                .where(Device.profile_id == self.profile_id)
                .distinct()
            )
        ).all()
        return [
            (
                device_id,
                f"{manufacturer} {model}",
            )  # name format must match _days_by_device
            for device_id, manufacturer, model in rows
        ]

    async def _query_device_days_desc(
        self,
        db_session: AsyncSession,
        device_id: int,
        *,
        before: date | None,
        limit: int,
    ) -> list[Day]:
        """Return up to `limit` days for `device_id` ordered newest-first.

        If `before` is given, only days strictly older than that date are
        included (keyset pagination cursor).
        """
        q = select(Day).where(Day.device_id == device_id).order_by(Day.date.desc())
        if before is not None:
            q = q.where(Day.date < before)
        return list(
            (
                await db_session.execute(
                    q.options(
                        selectinload(Day.sessions).selectinload(SessionModel.settings),
                    ).limit(limit)
                )
            )
            .unique()
            .scalars()
            .all()
        )

    async def _find_last_period_for_device(
        self,
        db_session: AsyncSession,
        device_id: int,
        device_name: str,
    ) -> RxPeriod | None:
        """Walk batches newest→oldest to find the last RX period for a device.

        Returns the most recent contiguous run of days sharing the same RX
        fingerprint, skipping days with no valid settings.  Stops as soon as
        a day with a different fingerprint (a period boundary) is encountered.
        If all batches exhaust without finding a boundary, the entire device
        history is one period.
        """
        current_fingerprint: str | None = None
        current_settings: dict[str, str] | None = None
        collected_days: list[Day] = []  # accumulated newest-first
        cursor: date | None = None
        done = False

        while not done:
            batch = await self._query_device_days_desc(
                db_session, device_id, before=cursor, limit=_TAIL_WALK_BATCH_SIZE
            )
            if not batch:
                break

            for day in batch:
                day_settings = self._get_day_period_settings(day)
                if day_settings is None:
                    continue

                fingerprint = self._build_fingerprint(day_settings)

                if current_fingerprint is None:
                    current_fingerprint = fingerprint
                    current_settings = day_settings

                if fingerprint != current_fingerprint:
                    done = True
                    break

                collected_days.append(day)

            if not done:
                cursor = batch[-1].date

        if not collected_days or current_settings is None:
            return None

        collected_days.reverse()
        return RxPeriod(
            settings=current_settings,
            start_date=collected_days[0].date,
            end_date=collected_days[-1].date,
            days=collected_days,
            device_id=device_id,
            device_name=device_name,
        )
