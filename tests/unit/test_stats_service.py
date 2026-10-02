"""Unit tests for StatsService."""

import uuid

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day, Device, Event, Profile, Session, Statistics, User
from snore.services.stats_service import StatsService


async def _create_day_with_session(
    db_session: AsyncSession,
    device: Device,
    day_date: date,
    duration_hours: float = 8.0,
    ahi: float = 2.5,
    **day_kwargs: Any,
) -> tuple[Day, Session]:
    """Helper to create a Day with associated Session."""
    day = Day(
        device_id=device.id,
        date=day_date,
        total_therapy_hours=duration_hours,
        ahi=ahi,
        **day_kwargs,
    )
    db_session.add(day)
    await db_session.flush()
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"test_{day_date.isoformat()}",
        start_time=datetime.combine(day_date, datetime.min.time()),
        end_time=datetime.combine(day_date, datetime.min.time())
        + timedelta(hours=duration_hours),
        duration_seconds=duration_hours * 3600,
    )
    db_session.add(sess)
    await db_session.flush()
    return day, sess


class TestStatsService:
    """Tests for StatsService methods."""

    async def test_empty_database_returns_none(self, async_db_session):
        """Empty database returns None for get_summary."""
        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is None

    async def test_summary_with_data(self, async_db_session, async_test_device):
        """get_summary computes basic metrics correctly."""
        today = date.today()
        day1, sess1 = await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=2),
            duration_hours=8.0,
            ahi=2.5,
        )
        day2, sess2 = await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=1),
            duration_hours=7.0,
            ahi=3.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        assert summary.days_with_data == 2
        assert summary.total_hours == 15.0
        assert summary.avg_hours == 7.5
        # Usage-weighted by total_therapy_hours: (2.5*8 + 3.0*7) / 15 = 41/15.
        assert summary.avg_ahi == pytest.approx(41.0 / 15.0)
        assert summary.effectiveness == "excellent"
        assert summary.first_date == day1.date
        assert summary.last_date == day2.date

    async def test_summary_days_limit(self, async_db_session, async_test_device):
        """days_limit parameter filters Day records correctly."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=10),
            duration_hours=8.0,
            ahi=2.5,
        )
        day2, _ = await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=2),
            duration_hours=7.0,
            ahi=3.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary(days_limit=5)

        assert summary is not None
        assert summary.days_with_data == 1
        assert summary.first_date == day2.date
        assert summary.last_date == day2.date

    async def test_summary_event_counts(self, async_db_session, async_test_device):
        """Event counts are aggregated correctly."""
        today = date.today()
        day, sess = await _create_day_with_session(
            async_db_session, async_test_device, today, duration_hours=8.0, ahi=2.5
        )

        for i in range(5):
            event = Event(
                session_id=sess.id,
                event_type="ObstructiveApnea",
                start_time=datetime.combine(today, datetime.min.time())
                + timedelta(minutes=i * 10),
                duration_seconds=10.0,
            )
            async_db_session.add(event)

        for i in range(3):
            event = Event(
                session_id=sess.id,
                event_type="Hypopnea",
                start_time=datetime.combine(today, datetime.min.time())
                + timedelta(minutes=i * 15),
                duration_seconds=12.0,
            )
            async_db_session.add(event)

        await async_db_session.flush()

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        assert len(summary.event_counts) == 2

        oa_count = next(
            e for e in summary.event_counts if e.event_type == "ObstructiveApnea"
        )
        assert oa_count.count == 5
        assert oa_count.percentage == 62.5

        h_count = next(e for e in summary.event_counts if e.event_type == "Hypopnea")
        assert h_count.count == 3
        assert h_count.percentage == 37.5

    async def test_weighted_stats(self, async_db_session, async_test_device):
        """Weighted averages computed correctly based on usage_hours."""
        today = date.today()
        day1, sess1 = await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=1),
            duration_hours=8.0,
            ahi=2.5,
        )
        day2, sess2 = await _create_day_with_session(
            async_db_session, async_test_device, today, duration_hours=4.0, ahi=3.0
        )

        stats1 = Statistics(
            session_id=sess1.id,
            usage_hours=8.0,
            pulse_mean=70.0,
            rei=1.5,
            epap_mean=5.0,
        )
        stats2 = Statistics(
            session_id=sess2.id,
            usage_hours=4.0,
            pulse_mean=80.0,
            rei=2.0,
            epap_mean=6.0,
        )
        async_db_session.add(stats1)
        async_db_session.add(stats2)
        await async_db_session.flush()

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        assert summary.avg_pulse is not None
        assert abs(summary.avg_pulse - 73.33) < 0.1
        assert summary.avg_rei is not None
        assert abs(summary.avg_rei - 1.67) < 0.1
        assert summary.avg_epap is not None
        assert abs(summary.avg_epap - 5.33) < 0.1

    async def test_period_statistics(self, async_db_session, async_test_device):
        """get_period_statistics returns correct list of PeriodStatistics."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=10),
            duration_hours=8.0,
            ahi=2.5,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=3),
            duration_hours=7.0,
            ahi=3.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        period_stats = await service.get_period_statistics("week")

        assert len(period_stats) > 0
        for stat in period_stats:
            assert stat.period_type == "week"
            assert stat.days_used >= 0

    async def test_get_records(self, async_db_session, async_test_device):
        """get_records returns best/worst days for metrics."""
        today = date.today()
        for i in range(7):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                today - timedelta(days=i),
                duration_hours=7.0 + i * 0.5,
                ahi=2.0 + i * 0.5,
                leak_median=10.0 + i,
                spo2_min=92.0 - i,
            )

        service = StatsService(async_db_session, profile_id=1)
        records = await service.get_records(top_n=5)

        assert records.model_fields_set >= {"ahi", "leak", "therapy_hours", "spo2_min"}
        assert records.ahi is not None

        assert len(records.ahi.best) <= 5
        assert len(records.ahi.worst) <= 5

        assert records.ahi.best[0][1] == 2.0
        assert records.ahi.worst[0][1] == 5.0

    async def test_pressure_aggregates(self, async_db_session, async_test_device):
        """Pressure avg is usage-weighted from Day.pressure_median; min/max unweighted."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=2),
            duration_hours=8.0,
            ahi=2.5,
            pressure_median=10.0,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=1),
            duration_hours=7.0,
            ahi=3.0,
            pressure_median=12.0,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today,
            duration_hours=7.5,
            ahi=2.0,
            pressure_median=9.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        assert summary.avg_pressure is not None
        # Usage-weighted: (10*8 + 12*7 + 9*7.5) / (8+7+7.5) = 231.5/22.5.
        assert summary.avg_pressure == pytest.approx(231.5 / 22.5)
        assert summary.min_pressure == 9.0
        assert summary.max_pressure == 12.0

    async def test_spo2_aggregates(self, async_db_session, async_test_device):
        """SpO2 min/avg computed correctly from Day records."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=2),
            duration_hours=8.0,
            ahi=2.5,
            spo2_mean=96.0,
            spo2_min=90,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=1),
            duration_hours=7.0,
            ahi=3.0,
            spo2_mean=95.0,
            spo2_min=88,
        )

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        assert summary.avg_spo2 is not None
        # Usage-weighted: (96*8 + 95*7) / 15 = 1433/15.
        assert summary.avg_spo2 == pytest.approx(1433.0 / 15.0)
        assert summary.min_spo2 == 88

    async def test_leak_avg_usage_weighted(self, async_db_session, async_test_device):
        """avg_leak weights each day's leak_median by total_therapy_hours."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=1),
            duration_hours=8.0,
            leak_median=10.0,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today,
            duration_hours=2.0,
            leak_median=30.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        # Weighted (10*8 + 30*2)/10 = 14.0, NOT the arithmetic mean 20.0.
        assert summary.avg_leak == pytest.approx(14.0)

    async def test_zero_hour_day_excluded_from_weighted_means(
        self, async_db_session, async_test_device
    ):
        """A day with total_therapy_hours == 0 drops out of the weighted means."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=1),
            duration_hours=6.0,
            pressure_median=11.0,
        )
        # Zero-hour day carries an extreme pressure that must not shift the mean.
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today,
            duration_hours=0.0,
            pressure_median=99.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        assert summary.avg_pressure == pytest.approx(11.0)

    async def test_all_disabled_day_excluded_from_hours_and_day_count(
        self, async_db_session, async_test_device
    ):
        """A day whose sessions are all disabled adds no hours and no data day."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=1),
            duration_hours=6.0,
        )
        # total_therapy_hours counts only enabled sessions, so it is 0 here even
        # though the disabled session itself lasted 4 hours.
        disabled_day = Day(
            device_id=async_test_device.id, date=today, total_therapy_hours=0.0
        )
        async_db_session.add(disabled_day)
        await async_db_session.flush()
        async_db_session.add(
            Session(
                device_id=async_test_device.id,
                day_id=disabled_day.id,
                device_session_id=f"disabled_{today.isoformat()}",
                start_time=datetime.combine(today, datetime.min.time()),
                end_time=datetime.combine(today, datetime.min.time())
                + timedelta(hours=4),
                duration_seconds=4 * 3600,
                enabled=False,
            )
        )
        await async_db_session.flush()

        summary = await StatsService(async_db_session, profile_id=1).get_summary()

        assert summary is not None
        assert summary.total_hours == pytest.approx(6.0)
        assert summary.days_with_data == 1
        assert summary.avg_hours == pytest.approx(6.0)

    async def test_zero_ahi_assessed_excellent(
        self, async_db_session, async_test_device
    ):
        """An average AHI of exactly 0.0 is a real value, not missing data."""
        await _create_day_with_session(
            async_db_session, async_test_device, date.today(), ahi=0.0
        )

        summary = await StatsService(async_db_session, profile_id=1).get_summary()

        assert summary is not None
        assert summary.avg_ahi == 0.0
        assert summary.effectiveness == "excellent"


class TestChunkedIdBinds:
    """Bulk ``Day.id.in_(...)`` binds are chunked under SQLite's param cap (#280)."""

    async def test_summary_event_counts_added_and_resorted_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Event counts are ADDED (not overwritten) across chunks, then re-sorted DESC."""
        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)
        today = date.today()
        # The same event types recur on all three days, which span a chunk
        # boundary, so an accidental per-chunk overwrite would drop the earlier
        # chunk's counts and yield the wrong totals and ordering.
        per_day = [
            {"ObstructiveApnea": 5, "Hypopnea": 1},
            {"ObstructiveApnea": 4, "Hypopnea": 2},
            {"ObstructiveApnea": 1, "Hypopnea": 5},
        ]
        for i, counts in enumerate(per_day):
            day_date = today - timedelta(days=i)
            _, sess = await _create_day_with_session(
                async_db_session, async_test_device, day_date
            )
            for event_type, n in counts.items():
                for j in range(n):
                    async_db_session.add(
                        Event(
                            session_id=sess.id,
                            event_type=event_type,
                            start_time=datetime.combine(day_date, datetime.min.time())
                            + timedelta(minutes=j),
                            duration_seconds=10.0,
                        )
                    )
        await async_db_session.flush()

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        # True totals: OA = 5+4+1 = 10, Hypopnea = 1+2+5 = 8 (18 total).
        # The list order also asserts the re-sort by count DESC.
        assert [(e.event_type, e.count) for e in summary.event_counts] == [
            ("ObstructiveApnea", 10),
            ("Hypopnea", 8),
        ]
        oa, hyp = summary.event_counts
        assert oa.percentage == pytest.approx(10 / 18 * 100)
        assert hyp.percentage == pytest.approx(8 / 18 * 100)

    async def test_summary_stats_records_concat_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Statistics rows are concatenated across chunks for summary aggregates."""
        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)
        today = date.today()
        for i, below in enumerate((30, 45, 25)):
            _, sess = await _create_day_with_session(
                async_db_session, async_test_device, today - timedelta(days=i)
            )
            async_db_session.add(
                Statistics(
                    session_id=sess.id,
                    usage_hours=6.0,
                    spo2_time_below_90=below,
                )
            )
        await async_db_session.flush()

        service = StatsService(async_db_session, profile_id=1)
        summary = await service.get_summary()

        assert summary is not None
        # Sum spans the chunk boundary: 30 + 45 + 25 = 100.
        assert summary.total_spo2_time_below_90 == 100

    async def test_aggregate_session_stats_concat_across_chunks(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """Per-period session stats use Statistics rows concatenated across chunks."""
        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)
        # Three days in one month → one period, spanning the chunk boundary.
        specs = [
            (date(2024, 1, 5), 8.0, 5.0),
            (date(2024, 1, 15), 4.0, 7.0),
            (date(2024, 1, 25), 2.0, 11.0),
        ]
        for day_date, usage, epap in specs:
            _, sess = await _create_day_with_session(
                async_db_session, async_test_device, day_date
            )
            async_db_session.add(
                Statistics(session_id=sess.id, usage_hours=usage, epap_mean=epap)
            )
        await async_db_session.flush()

        service = StatsService(async_db_session, profile_id=1)
        day_records = await service._query_days()
        period_stats = await service.get_period_statistics("month")
        extras = await service._aggregate_session_stats_per_period(
            day_records, period_stats
        )

        # Usage-weighted epap over ALL three sessions:
        # (5*8 + 7*4 + 11*2) / (8+4+2) = 90/14.
        assert extras[date(2024, 1, 1)]["epap"] == pytest.approx(90.0 / 14.0)


class TestGetDataRange:
    """Tests for StatsService.get_data_range()."""

    async def test_returns_none_fields_with_no_data(self, async_db_session):
        """Empty database returns DataRange with both dates None."""
        service = StatsService(async_db_session, profile_id=1)
        result = await service.get_data_range()

        assert result.earliest_date is None
        assert result.latest_date is None

    async def test_returns_both_bounds_with_multiple_days(
        self, async_db_session, async_test_device
    ):
        """Returns earliest and latest Day.date across all profile days."""
        today = date.today()
        oldest = today - timedelta(days=30)
        newest = today - timedelta(days=5)
        await _create_day_with_session(async_db_session, async_test_device, oldest)
        await _create_day_with_session(async_db_session, async_test_device, newest)
        await _create_day_with_session(
            async_db_session, async_test_device, today - timedelta(days=15)
        )

        service = StatsService(async_db_session, profile_id=1)
        result = await service.get_data_range()

        assert result.earliest_date == oldest
        assert result.latest_date == newest

    async def test_multi_device_spans_both_devices(self, async_db_session):
        """Min/max span days from two devices on the same profile."""
        user = User(
            canonical_email=f"u_{uuid.uuid4().hex[:8]}@test.com",
            role="admin",
        )
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="P")
        async_db_session.add(profile)
        await async_db_session.flush()

        dev_a = Device(
            profile_id=profile.id,
            manufacturer="Mfr",
            model="A",
            serial_number=f"SN_{uuid.uuid4().hex[:6]}",
        )
        dev_b = Device(
            profile_id=profile.id,
            manufacturer="Mfr",
            model="B",
            serial_number=f"SN_{uuid.uuid4().hex[:6]}",
        )
        async_db_session.add(dev_a)
        async_db_session.add(dev_b)
        await async_db_session.flush()

        today = date.today()
        earliest = today - timedelta(days=60)
        latest = today - timedelta(days=1)
        # dev_a owns the earliest day; dev_b owns the latest day
        await _create_day_with_session(async_db_session, dev_a, earliest)
        await _create_day_with_session(async_db_session, dev_b, latest)

        service = StatsService(async_db_session, profile_id=profile.id)
        result = await service.get_data_range()

        assert result.earliest_date == earliest
        assert result.latest_date == latest

    async def test_cross_profile_isolation(self, async_db_session):
        """Days on a different profile do not affect the result."""

        async def _make_profile_and_device() -> tuple[Profile, Device]:
            user = User(
                canonical_email=f"u_{uuid.uuid4().hex[:8]}@test.com",
                role="admin",
            )
            async_db_session.add(user)
            await async_db_session.flush()
            profile = Profile(user_id=user.id, name="P")
            async_db_session.add(profile)
            await async_db_session.flush()
            device = Device(
                profile_id=profile.id,
                manufacturer="Mfr",
                model="M",
                serial_number=f"SN_{uuid.uuid4().hex[:6]}",
            )
            async_db_session.add(device)
            await async_db_session.flush()
            return profile, device

        profile_a, dev_a = await _make_profile_and_device()
        profile_b, dev_b = await _make_profile_and_device()

        today = date.today()
        # profile_a has a day 10 days ago
        await _create_day_with_session(
            async_db_session, dev_a, today - timedelta(days=10)
        )
        # profile_b has a day 1 day ago — must NOT appear in profile_a's range
        await _create_day_with_session(
            async_db_session, dev_b, today - timedelta(days=1)
        )

        service = StatsService(async_db_session, profile_id=profile_a.id)
        result = await service.get_data_range()

        assert result.earliest_date == today - timedelta(days=10)
        assert result.latest_date == today - timedelta(days=10)


class TestQueryDaysFiltering:
    """Test _query_days with from_date and to_date parameters."""

    async def test_from_date_excludes_earlier_days(
        self, async_db_session, async_test_device
    ):
        """from_date filters out Day records before the cutoff."""
        await _create_day_with_session(
            async_db_session, async_test_device, date(2024, 1, 1), duration_hours=8.0
        )
        await _create_day_with_session(
            async_db_session, async_test_device, date(2024, 6, 1), duration_hours=8.0
        )

        service = StatsService(async_db_session, profile_id=1)
        result = await service._query_days(from_date=date(2024, 3, 1))

        assert len(result) == 1
        assert result[0].date == date(2024, 6, 1)

    async def test_to_date_excludes_later_days(
        self, async_db_session, async_test_device
    ):
        """to_date filters out Day records after the cutoff."""
        await _create_day_with_session(
            async_db_session, async_test_device, date(2024, 1, 1), duration_hours=8.0
        )
        await _create_day_with_session(
            async_db_session, async_test_device, date(2024, 6, 1), duration_hours=8.0
        )

        service = StatsService(async_db_session, profile_id=1)
        result = await service._query_days(to_date=date(2024, 3, 1))

        assert len(result) == 1
        assert result[0].date == date(2024, 1, 1)

    async def test_from_date_and_to_date_combined(
        self, async_db_session, async_test_device
    ):
        """Both from_date and to_date can be applied simultaneously."""
        for month in [1, 4, 7, 10]:
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                date(2024, month, 15),
                duration_hours=8.0,
            )

        service = StatsService(async_db_session, profile_id=1)
        result = await service._query_days(
            from_date=date(2024, 3, 1), to_date=date(2024, 8, 1)
        )

        result_dates = {d.date for d in result}
        assert result_dates == {date(2024, 4, 15), date(2024, 7, 15)}

    async def test_days_limit_and_from_date_stack(
        self, async_db_session, async_test_device
    ):
        """days_limit and from_date both apply as AND conditions."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=100),
            duration_hours=8.0,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=5),
            duration_hours=8.0,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=2),
            duration_hours=8.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        # days_limit=30 excludes the oldest day; from_date also excludes days before 4 days ago
        result = await service._query_days(
            days_limit=30, from_date=today - timedelta(days=4)
        )

        assert len(result) == 1
        assert result[0].date == today - timedelta(days=2)


class TestAggregateSessionStatsPerPeriod:
    """Test _aggregate_session_stats_per_period with hand-computed weighted means."""

    async def test_weighted_means_per_period(self, async_db_session, async_test_device):
        """Usage-weighted epap/rr/pulse/mv are computed correctly per period bucket."""
        # Jan: two sessions with different usage_hours
        jan_day, jan_sess1 = await _create_day_with_session(
            async_db_session, async_test_device, date(2024, 1, 10), duration_hours=8.0
        )
        jan_day2 = Day(
            device_id=async_test_device.id,
            date=date(2024, 1, 20),
            total_therapy_hours=4.0,
        )
        async_db_session.add(jan_day2)
        await async_db_session.flush()
        jan_sess2 = Session(
            device_id=async_test_device.id,
            day_id=jan_day2.id,
            device_session_id="test_jan20",
            start_time=datetime(2024, 1, 20, 22, 0),
            end_time=datetime(2024, 1, 21, 2, 0),
            duration_seconds=4 * 3600,
        )
        async_db_session.add(jan_sess2)
        await async_db_session.flush()

        stats1 = Statistics(
            session_id=jan_sess1.id,
            usage_hours=8.0,
            epap_mean=5.0,
            respiratory_rate_mean=14.0,
            pulse_mean=60.0,
            minute_ventilation_mean=7.0,
        )
        stats2 = Statistics(
            session_id=jan_sess2.id,
            usage_hours=4.0,
            epap_mean=7.0,
            respiratory_rate_mean=16.0,
            pulse_mean=72.0,
            minute_ventilation_mean=9.0,
        )
        async_db_session.add(stats1)
        async_db_session.add(stats2)
        await async_db_session.flush()

        service = StatsService(async_db_session, profile_id=1)
        day_records = await service._query_days()
        period_stats = await service.get_period_statistics("month")

        extras = await service._aggregate_session_stats_per_period(
            day_records, period_stats
        )

        jan_start = date(2024, 1, 1)
        assert jan_start in extras

        # hand-computed: weighted mean = (5.0*8 + 7.0*4) / (8+4) = (40+28)/12 = 68/12 ≈ 5.667
        assert extras[jan_start]["epap"] == pytest.approx(68.0 / 12.0, abs=0.01)
        # rr: (14*8 + 16*4) / 12 = (112 + 64)/12 = 176/12 ≈ 14.667
        assert extras[jan_start]["rr"] == pytest.approx(176.0 / 12.0, abs=0.01)
        # pulse: (60*8 + 72*4) / 12 = (480 + 288)/12 = 768/12 = 64.0
        assert extras[jan_start]["pulse"] == pytest.approx(64.0, abs=0.01)

    async def test_missing_usage_hours_skipped(
        self, async_db_session, async_test_device
    ):
        """Statistics rows with None or 0 usage_hours are excluded from weighting."""
        day, sess = await _create_day_with_session(
            async_db_session, async_test_device, date(2024, 2, 10), duration_hours=8.0
        )
        # usage_hours = 0: should be excluded
        stats = Statistics(
            session_id=sess.id,
            usage_hours=0.0,
            epap_mean=99.0,
        )
        async_db_session.add(stats)
        await async_db_session.flush()

        service = StatsService(async_db_session, profile_id=1)
        day_records = await service._query_days()
        period_stats = await service.get_period_statistics("month")

        extras = await service._aggregate_session_stats_per_period(
            day_records, period_stats
        )

        feb_start = date(2024, 2, 1)
        assert extras[feb_start]["epap"] is None

    async def test_empty_day_records_returns_none_dict(self, async_db_session):
        """Empty day_records yields None values for all periods."""
        from snore.services.schemas import PeriodStatistics

        ps = PeriodStatistics(
            period_type="month",
            period_start=date(2024, 3, 1),
            period_end=date(2024, 3, 31),
        )
        service = StatsService(async_db_session, profile_id=1)
        extras = await service._aggregate_session_stats_per_period([], [ps])

        assert extras[date(2024, 3, 1)] == {
            "epap": None,
            "rr": None,
            "pulse": None,
            "mv": None,
        }


class TestGetTrends:
    """Test StatsService.get_trends with the new (period_type, days_limit) signature."""

    async def test_get_trends_returns_13_keys(
        self, async_db_session, async_test_device
    ):
        """get_trends returns all 13 metric keys."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=3),
            duration_hours=8.0,
            ahi=2.5,
        )

        service = StatsService(async_db_session, profile_id=1)
        result = await service.get_trends("week")

        expected = {
            "ahi",
            "usage",
            "spo2",
            "leak",
            "pressure",
            "oai",
            "cai",
            "hi",
            "rera",
            "epap",
            "rr",
            "pulse",
            "mv",
        }
        assert result.model_fields_set == expected

    async def test_get_trends_day_granularity(
        self, async_db_session, async_test_device
    ):
        """get_trends with period_type='day' produces one entry per therapy day."""
        base = date.today() - timedelta(days=5)
        for i in range(3):
            await _create_day_with_session(
                async_db_session,
                async_test_device,
                base + timedelta(days=i),
                duration_hours=7.0,
                ahi=float(i + 1),
            )

        service = StatsService(async_db_session, profile_id=1)
        result = await service.get_trends("day")

        assert len(result.ahi) == 3

    async def test_get_trends_days_limit_filters(
        self, async_db_session, async_test_device
    ):
        """days_limit parameter passed to get_trends restricts the Day records used."""
        today = date.today()
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=60),
            duration_hours=8.0,
        )
        await _create_day_with_session(
            async_db_session,
            async_test_device,
            today - timedelta(days=2),
            duration_hours=8.0,
        )

        service = StatsService(async_db_session, profile_id=1)
        result = await service.get_trends("month", days_limit=30)

        # Only the recent day should appear
        assert len(result.ahi) == 1

    async def test_get_trends_empty_returns_13_empty_lists(self, async_db_session):
        """Empty database returns 13-key dict with empty lists."""
        service = StatsService(async_db_session, profile_id=1)
        result = await service.get_trends("week")

        assert len(result.model_fields_set) == 13
        for field in result.model_fields_set:
            assert getattr(result, field) == []
