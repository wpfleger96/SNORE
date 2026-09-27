"""Statistics service for therapy data aggregation and analysis."""

from bisect import bisect_right
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
from snore.services.schemas import (
    DataRange,
    EventTypeCount,
    PeriodStatistics,
    TherapySummary,
)
from snore.utils.db_chunk import iter_id_chunks
from snore.utils.stats import usage_weighted_means

# ---------------------------------------------------------------------------
# Type alias for the health nights list used across multiple helpers.
# ---------------------------------------------------------------------------
_HealthNights = list[models.HealthNightlySummary]

__all__ = ["StatsService"]


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
        if not nights or not period_stats:
            return {ps.period_start: [] for ps in period_stats}

        sorted_starts = sorted(ps.period_start for ps in period_stats)
        period_end_by_start: dict[date, date] = {
            ps.period_start: ps.period_end for ps in period_stats
        }
        binned: dict[date, _HealthNights] = {ps.period_start: [] for ps in period_stats}

        for night in nights:
            idx = bisect_right(sorted_starts, night.night_date) - 1
            if idx < 0:
                continue
            pstart = sorted_starts[idx]
            if night.night_date <= period_end_by_start[pstart]:
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

        dates = [d.date for d in day_records]
        first_date = min(dates)
        last_date = max(dates)
        days_since_last = (date.today() - last_date).days

        day_ids = [d.id for d in day_records]
        days_with_data = len(day_records)

        total_duration = 0.0
        for chunk in iter_id_chunks(day_ids):
            total_duration += (
                await self.db_session.execute(
                    select(func.sum(models.Session.duration_seconds))
                    .join(models.Day)
                    .where(models.Day.id.in_(chunk))
                )
            ).scalar() or 0
        total_hours = total_duration / 3600
        avg_hours = total_hours / days_with_data if days_with_data > 0 else 0

        avg_ahi = calculate_average_ahi(day_records)
        effectiveness = assess_therapy_effectiveness(avg_ahi) if avg_ahi else "unknown"

        weekly_periods = calculate_period_statistics(day_records, "week")
        weekly_ahi_values = [
            ps.avg_ahi for ps in weekly_periods if ps.avg_ahi is not None
        ]
        ahi_trend_direction = calculate_ahi_trend_direction(weekly_ahi_values)

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

        stats_records: list[models.Statistics] = []
        for chunk in iter_id_chunks(day_ids):
            stats_records.extend(
                (
                    await self.db_session.execute(
                        select(models.Statistics)
                        .join(models.Session)
                        .join(models.Day)
                        .where(models.Day.id.in_(chunk))
                    )
                )
                .scalars()
                .all()
            )

        usage_means = usage_weighted_means(
            stats_records,
            {
                "rr": "respiratory_rate_mean",
                "tv": "tidal_volume_mean",
                "mv": "minute_ventilation_mean",
                "pulse": "pulse_mean",
                "rei": "rei",
                "epap": "epap_mean",
            },
            lambda stat: stat.usage_hours,
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
            avg_ahi=avg_ahi,
            effectiveness=effectiveness,
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
            avg_tidal_volume=usage_means["tv"],
            avg_minute_ventilation=usage_means["mv"],
            ahi_trend_direction=ahi_trend_direction,
            event_counts=event_type_counts,
        )

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
        period_stats = calculate_period_statistics(day_records, period_type)
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

        sorted_starts = sorted(ps.period_start for ps in period_stats)
        period_end_by_start: dict[date, date] = {
            ps.period_start: ps.period_end for ps in period_stats
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

        grouped: dict[date, list[models.Statistics]] = {}
        for stat, day_date in rows:
            idx = bisect_right(sorted_starts, day_date) - 1
            if idx < 0:
                continue
            period_start = sorted_starts[idx]
            if day_date > period_end_by_start[period_start]:
                continue
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
    ) -> dict[str, list[tuple[date, float | None]]]:
        """Compute extended trend data for the requested period granularity."""
        day_records = await self._query_days(
            days_limit, from_date=from_date, to_date=to_date
        )
        period_stats = calculate_period_statistics(day_records, period_type)
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

        return trends

    async def get_records(
        self, days_limit: int | None = None, top_n: int = 5
    ) -> dict[str, dict[str, list[tuple[date, float]]]]:
        """Calculate top best/worst days for key metrics."""
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

        return records

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
