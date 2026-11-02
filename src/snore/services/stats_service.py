"""Statistics service for therapy data aggregation and analysis."""

import asyncio

from bisect import bisect_right
from collections.abc import Callable
from datetime import date, timedelta

from sqlalchemy import Row, func, select

from snore.analysis.calculations import (
    PeriodType,
    assess_therapy_effectiveness,
    calculate_ahi_trend_direction,
    calculate_average_ahi,
    calculate_period_statistics,
    calculate_records,
    calculate_trends_extended,
)
from snore.database import models
from snore.services._base import ProfileScopedService
from snore.services.channel_stats import load_channel_samples, pool_channel_stats
from snore.services.event_service import device_event_types
from snore.services.schemas import (
    DataRange,
    DeviceChannelStats,
    EventTypeCount,
    ExperimentalStats,
    PeriodStatistics,
    RecordsResponse,
    TherapySummary,
    TrendsResponse,
)
from snore.services.tidal_volume import tidal_volume_ml
from snore.utils.db_chunk import iter_id_chunks
from snore.utils.stats import usage_weighted_means, weighted_mean

# ---------------------------------------------------------------------------
# Type alias for the health nights list used across multiple helpers.
# ---------------------------------------------------------------------------
_HealthNights = list[models.HealthNightlySummary]

__all__ = ["StatsService"]


def _period_start_lookup(
    period_stats: list[PeriodStatistics],
) -> Callable[[date], date | None]:
    """Return a function mapping a date to the start of the period containing it."""
    sorted_starts = sorted(ps.period_start for ps in period_stats)
    period_end_by_start = {ps.period_start: ps.period_end for ps in period_stats}

    def period_start_of(day_date: date) -> date | None:
        idx = bisect_right(sorted_starts, day_date) - 1
        if idx < 0 or day_date > period_end_by_start[sorted_starts[idx]]:
            return None
        return sorted_starts[idx]

    return period_start_of


class StatsService(ProfileScopedService):
    """Service for therapy statistics computation and analysis."""

    async def _query_days(
        self,
        days_limit: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[models.Day]:
        """Query Day records filtered to this profile, optionally by date range."""
        query = (
            select(models.Day)
            .join(models.Device, models.Day.device_id == models.Device.id)
            .where(self._profile_filter())
        )
        if days_limit:
            cutoff_date = date.today() - timedelta(days=days_limit)
            query = query.where(models.Day.date >= cutoff_date)
        if from_date is not None:
            query = query.where(models.Day.date >= from_date)
        if to_date is not None:
            query = query.where(models.Day.date <= to_date)
        return list((await self.db_session.execute(query)).scalars().all())

    async def _query_health_nights(
        self,
        days_limit: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> _HealthNights:
        """Query HealthNightlySummary records for this profile, optional date range."""
        query = select(models.HealthNightlySummary).where(
            models.HealthNightlySummary.profile_id == self.profile_id
        )
        if days_limit:
            cutoff_date = date.today() - timedelta(days=days_limit)
            query = query.where(models.HealthNightlySummary.night_date >= cutoff_date)
        if from_date is not None:
            query = query.where(models.HealthNightlySummary.night_date >= from_date)
        if to_date is not None:
            query = query.where(models.HealthNightlySummary.night_date <= to_date)
        return list((await self.db_session.execute(query)).scalars().all())

    @staticmethod
    def _bin_nights_by_period(
        nights: _HealthNights,
        period_stats: list[PeriodStatistics],
    ) -> dict[date, _HealthNights]:
        """Bin health nights into their corresponding period_start buckets."""
        binned: dict[date, _HealthNights] = {ps.period_start: [] for ps in period_stats}
        if not nights or not period_stats:
            return binned

        period_start_of = _period_start_lookup(period_stats)
        for night in nights:
            pstart = period_start_of(night.night_date)
            if pstart is not None:
                binned[pstart].append(night)

        return binned

    @staticmethod
    def _per_period_sleep_avgs(
        binned: dict[date, _HealthNights],
        periods: list[PeriodStatistics],
    ) -> dict[date, tuple[float | None, float | None]]:
        """Return (avg_sleep_hours, avg_sleep_efficiency_pct) keyed by period_start.

        Values are None when no nights with valid data fall in the period.
        Rounding: hours → 2 decimal places, efficiency → 1.
        """
        result: dict[date, tuple[float | None, float | None]] = {}
        for ps in periods:
            ns = binned.get(ps.period_start, [])
            h_vals = [
                n.total_sleep_seconds / 3600
                for n in ns
                if n.total_sleep_seconds is not None
            ]
            e_vals = [
                n.sleep_efficiency_pct for n in ns if n.sleep_efficiency_pct is not None
            ]
            result[ps.period_start] = (
                round(sum(h_vals) / len(h_vals), 2) if h_vals else None,
                round(sum(e_vals) / len(e_vals), 1) if e_vals else None,
            )
        return result

    @staticmethod
    def _augment_period_stats(
        period_stats: list[PeriodStatistics],
        binned: dict[date, _HealthNights],
    ) -> list[PeriodStatistics]:
        """Return period stats with avg_total_sleep_hours and avg_sleep_efficiency_pct populated."""
        avgs = StatsService._per_period_sleep_avgs(binned, period_stats)
        return [
            ps.model_copy(
                update={
                    "avg_total_sleep_hours": avgs[ps.period_start][0],
                    "avg_sleep_efficiency_pct": avgs[ps.period_start][1],
                }
            )
            for ps in period_stats
        ]

    async def get_summary(
        self,
        days_limit: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> TherapySummary | None:
        """Compute aggregated therapy summary statistics."""
        day_records = await self._query_days(
            days_limit, from_date=from_date, to_date=to_date
        )

        if not day_records:
            return None

        # Day.total_therapy_hours counts only enabled sessions, so days whose
        # sessions are all disabled contribute neither hours nor a data day.
        used_days = [
            d
            for d in day_records
            if d.total_therapy_hours is not None and d.total_therapy_hours > 0
        ]
        total_hours = sum(d.total_therapy_hours for d in used_days)
        days_with_data = len(used_days)
        avg_hours = total_hours / days_with_data if days_with_data else 0.0

        # The date range reports therapy use; an all-disabled range still
        # reports the dates it spans.
        dates = [d.date for d in used_days or day_records]
        first_date = min(dates)
        last_date = max(dates)
        days_since_last = (date.today() - last_date).days

        # Averages are usage-weighted by Day.total_therapy_hours; min/max stay
        # over the raw per-day values.
        day_means = usage_weighted_means(
            day_records,
            {
                "pressure": "pressure_median",
                "leak": "leak_median",
                "spo2": "spo2_mean",
            },
            lambda d: d.total_therapy_hours,
        )
        avg_pressure = day_means["pressure"]
        avg_leak = day_means["leak"]
        avg_spo2 = day_means["spo2"]

        pressure_values = [
            d.pressure_median for d in day_records if d.pressure_median is not None
        ]
        min_pressure = min(pressure_values) if pressure_values else None
        max_pressure = max(pressure_values) if pressure_values else None

        spo2_mins = [d.spo2_min for d in day_records if d.spo2_min is not None]
        min_spo2 = min(spo2_mins) if spo2_mins else None

        day_ids = [d.id for d in day_records]
        # event_type is the GROUP BY key, not the chunked column, so the same
        # type recurs across day-chunks: merge counts by addition, then re-sort.
        merged_events: dict[str, int] = {}
        for chunk in iter_id_chunks(day_ids):
            rows = (
                await self.db_session.execute(
                    select(
                        models.Event.event_type,
                        func.count(models.Event.id).label("count"),
                    )
                    .join(models.Session)
                    .join(models.Day)
                    .where(models.Day.id.in_(chunk))
                    .group_by(models.Event.event_type)
                )
            ).all()
            for event_type, count in rows:
                merged_events[event_type] = merged_events.get(event_type, 0) + count

        event_counts = sorted(merged_events.items(), key=lambda kv: kv[1], reverse=True)
        total_events = sum(count for _, count in event_counts)
        event_type_counts = [
            EventTypeCount(
                event_type=event_type,
                count=count,
                percentage=(count / total_events * 100) if total_events > 0 else 0,
            )
            for event_type, count in event_counts
        ]

        stats_rows: list[Row[models.Statistics, str | None]] = []
        for chunk in iter_id_chunks(day_ids):
            stats_rows.extend(
                (
                    await self.db_session.execute(
                        select(models.Statistics, models.Session.import_source)
                        .join(models.Session)
                        .join(models.Day)
                        .where(models.Day.id.in_(chunk))
                    )
                ).all()
            )
        stats_records = [stat for stat, _ in stats_rows]

        usage_means = usage_weighted_means(
            stats_records,
            {
                "rr": "respiratory_rate_mean",
                "mv": "minute_ventilation_mean",
                "pulse": "pulse_mean",
                "rei": "rei",
                "epap": "epap_mean",
            },
            lambda stat: stat.usage_hours,
        )
        # Stored tidal volume mixes units by import source; weight it in mL.
        avg_tidal_volume = weighted_mean(
            (tv, stat.usage_hours)
            for stat, import_source in stats_rows
            if stat.usage_hours
            and stat.usage_hours > 0
            and (tv := tidal_volume_ml(stat.tidal_volume_mean, import_source))
            is not None
        )
        total_spo2_time_below_90 = sum(
            stat.spo2_time_below_90
            for stat in stats_records
            if stat.usage_hours
            and stat.usage_hours > 0
            and stat.spo2_time_below_90 is not None
        )

        return TherapySummary(
            first_date=first_date,
            last_date=last_date,
            days_since_last=days_since_last,
            total_hours=total_hours,
            avg_hours=avg_hours,
            days_with_data=days_with_data,
            avg_ahi=calculate_average_ahi(day_records),
            avg_rei=usage_means["rei"],
            avg_pressure=avg_pressure,
            min_pressure=min_pressure,
            max_pressure=max_pressure,
            avg_epap=usage_means["epap"],
            avg_leak=avg_leak,
            avg_spo2=avg_spo2,
            min_spo2=min_spo2,
            total_spo2_time_below_90=total_spo2_time_below_90,
            avg_pulse=usage_means["pulse"],
            avg_respiratory_rate=usage_means["rr"],
            avg_tidal_volume=avg_tidal_volume,
            avg_minute_ventilation=usage_means["mv"],
            event_counts=event_type_counts,
        )

    async def get_experimental_stats(self) -> ExperimentalStats:
        """SNORE's effectiveness band and AHI trend call over all recorded days.

        No recorded days is ``effectiveness="unknown"`` with no trend direction.
        """
        days = await self._query_days()
        weekly_ahi = [
            week.avg_ahi
            for week in await self._period_statistics(days, "week")
            if week.avg_ahi is not None
        ]
        return ExperimentalStats(
            effectiveness=assess_therapy_effectiveness(calculate_average_ahi(days)),
            ahi_trend_direction=calculate_ahi_trend_direction(weekly_ahi),
        )

    async def _period_statistics(
        self, day_records: list[models.Day], period_type: PeriodType
    ) -> list[PeriodStatistics]:
        """Period statistics, telling RERA zeros from devices that never score one."""
        reported = await device_event_types(
            self.db_session, {day.device_id for day in day_records}
        )
        return calculate_period_statistics(day_records, period_type, reported)

    async def _with_channel_percentiles(
        self,
        day_records: list[models.Day],
        period_stats: list[PeriodStatistics],
    ) -> list[PeriodStatistics]:
        """Return period stats with the FL and snore 95ths of their pooled samples.

        Every enabled session of a period feeds one pool, so a period's value is
        not an average of its nights' percentiles.
        """
        if not period_stats:
            return period_stats

        period_start_of = _period_start_lookup(period_stats)
        sessions_by_period: dict[date, list[int]] = {
            ps.period_start: [] for ps in period_stats
        }
        for chunk in iter_id_chunks([day.id for day in day_records]):
            rows = await self.db_session.execute(
                select(models.Session.id, models.Day.date)
                .join(models.Day, models.Session.day_id == models.Day.id)
                .where(models.Day.id.in_(chunk), models.Session.enabled.is_(True))
            )
            for session_id, day_date in rows:
                pstart = period_start_of(day_date)
                if pstart is not None:
                    sessions_by_period[pstart].append(session_id)

        samples = await load_channel_samples(
            self.db_session,
            [sid for sids in sessions_by_period.values() for sid in sids],
        )

        def pool_by_period() -> dict[date, DeviceChannelStats]:
            return {
                pstart: pool_channel_stats(samples, sids) or DeviceChannelStats()
                for pstart, sids in sessions_by_period.items()
            }

        pooled = await asyncio.to_thread(pool_by_period)
        return [
            ps.model_copy(
                update=pooled[ps.period_start].model_dump(
                    include={"device_flg_95th", "snore_95th"}
                )
            )
            for ps in period_stats
        ]

    async def get_period_statistics(
        self,
        period_type: PeriodType,
        days_limit: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> list[PeriodStatistics]:
        """Calculate statistics grouped by time periods."""
        day_records = await self._query_days(
            days_limit, from_date=from_date, to_date=to_date
        )
        period_stats = await self._with_channel_percentiles(
            day_records, await self._period_statistics(day_records, period_type)
        )
        if not period_stats:
            return period_stats
        health_nights = await self._query_health_nights(
            days_limit, from_date=from_date, to_date=to_date
        )
        binned = self._bin_nights_by_period(health_nights, period_stats)
        return self._augment_period_stats(period_stats, binned)

    async def _aggregate_session_stats_per_period(
        self,
        day_records: list[models.Day],
        period_stats: list[PeriodStatistics],
    ) -> dict[date, dict[str, float | None]]:
        """Compute usage-weighted session-level means per period."""
        if not day_records or not period_stats:
            return {
                ps.period_start: {"epap": None, "rr": None, "pulse": None, "mv": None}
                for ps in period_stats
            }

        day_ids = [d.id for d in day_records]
        rows: list[Row[models.Statistics, date]] = []
        for chunk in iter_id_chunks(day_ids):
            rows.extend(
                (
                    await self.db_session.execute(
                        select(models.Statistics, models.Day.date)
                        .join(
                            models.Session,
                            models.Statistics.session_id == models.Session.id,
                        )
                        .join(models.Day, models.Session.day_id == models.Day.id)
                        .where(models.Day.id.in_(chunk))
                    )
                ).all()
            )

        field_map = {
            "epap": "epap_mean",
            "rr": "respiratory_rate_mean",
            "pulse": "pulse_mean",
            "mv": "minute_ventilation_mean",
        }

        period_start_of = _period_start_lookup(period_stats)
        grouped: dict[date, list[models.Statistics]] = {}
        for stat, day_date in rows:
            period_start = period_start_of(day_date)
            if period_start is not None:
                grouped.setdefault(period_start, []).append(stat)

        return {
            ps.period_start: usage_weighted_means(
                grouped.get(ps.period_start, []),
                field_map,
                lambda stat: stat.usage_hours,
            )
            for ps in period_stats
        }

    async def get_trends(
        self,
        period_type: PeriodType,
        days_limit: int | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> TrendsResponse:
        """Compute extended trend data for the requested period granularity.

        Apple Health series are left unset (not ``None``) when no night has data,
        so ``exclude_unset`` serialization omits their keys.
        """
        day_records = await self._query_days(
            days_limit, from_date=from_date, to_date=to_date
        )
        period_stats = await self._with_channel_percentiles(
            day_records, await self._period_statistics(day_records, period_type)
        )
        session_extras = await self._aggregate_session_stats_per_period(
            day_records, period_stats
        )
        trends = calculate_trends_extended(period_stats, session_extras)

        if period_stats:
            health_nights = await self._query_health_nights(
                days_limit, from_date=from_date, to_date=to_date
            )
            binned = self._bin_nights_by_period(health_nights, period_stats)
            avgs = self._per_period_sleep_avgs(binned, period_stats)
            sleep_hours_series: list[tuple[date, float | None]] = [
                (ps.period_start, avgs[ps.period_start][0]) for ps in period_stats
            ]
            sleep_eff_series: list[tuple[date, float | None]] = [
                (ps.period_start, avgs[ps.period_start][1]) for ps in period_stats
            ]

            # Only add health series when data exists; keeps response shape
            # backward-compatible for profiles with no Apple Health import.
            if any(v is not None for _, v in sleep_hours_series):
                trends["total_sleep_hours"] = sleep_hours_series
            if any(v is not None for _, v in sleep_eff_series):
                trends["sleep_efficiency"] = sleep_eff_series

        return TrendsResponse.model_validate(trends)

    async def get_records(
        self, days_limit: int | None = None, top_n: int = 5
    ) -> RecordsResponse:
        """Calculate top best/worst days for key metrics.

        Metrics without a qualifying day are left unset (not ``None``), so
        ``exclude_unset`` serialization omits their keys.
        """
        day_records = await self._query_days(days_limit)
        records = calculate_records(day_records, top_n)

        health_nights = await self._query_health_nights(days_limit)
        sleep_pairs = [
            (n.night_date, round(n.total_sleep_seconds / 3600, 2))
            for n in health_nights
            if n.total_sleep_seconds is not None
        ]
        if sleep_pairs:
            records["total_sleep_hours"] = {
                "best": sorted(sleep_pairs, key=lambda x: x[1], reverse=True)[:top_n],
                "worst": sorted(sleep_pairs, key=lambda x: x[1])[:top_n],
            }

        return RecordsResponse.model_validate(records)

    async def get_data_range(self) -> DataRange:
        """Return the profile's earliest and latest Day.date (all-time, ignores days_limit)."""
        row = (
            await self.db_session.execute(
                select(func.min(models.Day.date), func.max(models.Day.date))
                .join(models.Device, models.Day.device_id == models.Device.id)
                .where(self._profile_filter())
            )
        ).one()
        return DataRange(earliest_date=row[0], latest_date=row[1])
