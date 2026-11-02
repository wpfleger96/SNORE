"""
Tests for statistical calculations module.

Tests the calculation functions for AHI statistics, period-based grouping,
and trend analysis used in the stats CLI command.
"""

from datetime import date, datetime, timedelta

import pytest

from sqlalchemy import select

from snore.analysis.calculations import (
    PeriodType,
    calculate_ahi_trend_direction,
    calculate_average_ahi,
    calculate_median_ahi,
    calculate_period_statistics,
    calculate_trends,
    calculate_trends_extended,
)
from snore.database.day_manager import DayManager
from snore.database.models import Day, Device
from snore.services.schemas import PeriodStatistics


def _periods(
    days: list[Day], period_type: PeriodType, *reported: str
) -> list[PeriodStatistics]:
    """``calculate_period_statistics`` with every day's device scoring ``reported``."""
    return calculate_period_statistics(
        days, period_type, {day.device_id: set(reported) for day in days}
    )


class TestCalculateMedianAhi:
    """Test median AHI calculation."""

    async def test_median_odd_count(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Median with odd number of days returns middle value."""
        device = async_test_device

        days = []
        for i, ahi in enumerate([2.0, 5.0, 8.0]):
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(2024, 11, i + 1, 12, 0, 0),
                duration_hours=8.0,
                ahi=ahi,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = calculate_median_ahi(days)

        assert result == pytest.approx(5.0, abs=0.01)

    async def test_median_even_count(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Median with even number of days returns average of two middle values."""
        device = async_test_device

        days = []
        for i, ahi in enumerate([2.0, 4.0, 6.0, 8.0]):
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(2024, 11, i + 1, 12, 0, 0),
                duration_hours=8.0,
                ahi=ahi,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = calculate_median_ahi(days)

        assert result == pytest.approx(5.0, abs=0.01)

    def test_median_empty(self):
        """Median of empty list returns None."""
        result = calculate_median_ahi([])

        assert result is None

    async def test_median_single(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Median with single day returns that day's AHI."""
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 1, 12, 0, 0),
            duration_hours=8.0,
            ahi=7.5,
        )
        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        result = calculate_median_ahi([day])

        assert result == pytest.approx(7.5, abs=0.01)

    async def test_median_all_none(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Median when all days have no AHI returns None."""
        device = async_test_device

        days = []
        for i in range(3):
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(2024, 11, i + 1, 12, 0, 0),
                duration_hours=8.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = calculate_median_ahi(days)

        assert result is None


class TestCalculateAverageAhi:
    """Test usage-weighted average AHI calculation."""

    @staticmethod
    def _day(ahi: float | None, hours: float | None) -> Day:
        day = Day()
        day.date = date(2024, 1, 1)
        day.ahi = ahi
        day.total_therapy_hours = hours
        return day

    def test_empty_returns_none(self) -> None:
        assert calculate_average_ahi([]) is None

    def test_weighted_by_therapy_hours(self) -> None:
        """A short high-AHI night barely moves the usage-weighted average."""
        days = [self._day(10.0, 8.0), self._day(40.0, 0.5)]

        result = calculate_average_ahi(days)

        # Weighted: (10*8 + 40*0.5) / 8.5 = 100/8.5 ≈ 11.76, NOT the
        # unweighted arithmetic mean of 25.0.
        assert result == pytest.approx(100.0 / 8.5)

    def test_zero_and_none_hour_days_excluded(self) -> None:
        """Days with hours None or <= 0 drop out entirely."""
        days = [
            self._day(10.0, 8.0),
            self._day(40.0, 0.5),
            self._day(99.0, 0.0),  # zero hours → excluded
            self._day(99.0, None),  # None hours → excluded
        ]

        result = calculate_average_ahi(days)

        assert result == pytest.approx(100.0 / 8.5)

    def test_none_ahi_excluded(self) -> None:
        """Days without an AHI are excluded from the weighting."""
        days = [self._day(5.0, 8.0), self._day(None, 8.0)]

        result = calculate_average_ahi(days)

        assert result == pytest.approx(5.0)

    def test_all_none_hours_returns_none(self) -> None:
        """When no day carries positive therapy hours, result is None."""
        days = [self._day(5.0, None), self._day(6.0, 0.0)]

        assert calculate_average_ahi(days) is None


class TestCalculatePeriodStatistics:
    """Test period-based statistics grouping."""

    async def test_monthly_buckets(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Monthly bucketing creates correct number of periods."""
        device = async_test_device
        start_date = date(2024, 10, 1)

        days = []
        for i in range(60):
            session_date = start_date + timedelta(days=i)
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(
                    session_date.year, session_date.month, session_date.day, 12, 0, 0
                ),
                duration_hours=8.0,
                ahi=5.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = _periods(days, "month")

        assert len(result) == 2
        assert result[0].period_start == date(2024, 10, 1)
        assert result[0].period_end == date(2024, 10, 31)
        assert result[1].period_start == date(2024, 11, 1)
        assert result[1].period_end == date(2024, 11, 30)

    async def test_weekly_buckets(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Weekly bucketing aligns to week boundaries."""
        device = async_test_device
        start_date = date(2024, 11, 4)

        days = []
        for i in range(14):
            session_date = start_date + timedelta(days=i)
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(
                    session_date.year, session_date.month, session_date.day, 12, 0, 0
                ),
                duration_hours=8.0,
                ahi=5.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = _periods(days, "week")

        assert len(result) >= 2

    async def test_yearly_buckets(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Yearly bucketing spans multiple years."""
        device = async_test_device

        days = []
        for year in [2024, 2025]:
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(year, 6, 15, 12, 0, 0),
                duration_hours=8.0,
                ahi=5.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = _periods(days, "year")

        assert len(result) == 2
        assert result[0].period_start == date(2024, 1, 1)
        assert result[0].period_end == date(2024, 12, 31)
        assert result[1].period_start == date(2025, 1, 1)
        assert result[1].period_end == date(2025, 12, 31)

    async def test_6month_buckets(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """6-month bucketing creates half-year periods."""
        device = async_test_device

        days = []
        for month in [2, 8]:
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(2024, month, 15, 12, 0, 0),
                duration_hours=8.0,
                ahi=5.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = _periods(days, "6month")

        assert len(result) == 2
        assert result[0].period_start == date(2024, 1, 1)
        assert result[0].period_end == date(2024, 6, 30)
        assert result[1].period_start == date(2024, 7, 1)
        assert result[1].period_end == date(2024, 12, 31)

    def test_empty_days(self):
        """Empty day list returns empty result."""
        result = _periods([], "month")

        assert result == []

    async def test_period_metrics(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Period statistics compute correct metrics."""
        device = async_test_device
        start_date = date(2024, 11, 1)

        days = []
        for i in range(10):
            session_date = start_date + timedelta(days=i)
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(
                    session_date.year, session_date.month, session_date.day, 12, 0, 0
                ),
                duration_hours=7.5,
                ahi=4.0 + i * 0.5,
                pressure_median=10.0,
                leak_median=8.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        result = _periods(days, "month")

        assert len(result) == 1
        period = result[0]
        assert period.days_used == 10
        assert period.avg_hours_per_day == pytest.approx(7.5, abs=0.01)
        assert period.avg_ahi == pytest.approx(6.25, abs=0.5)
        assert period.median_ahi == pytest.approx(6.25, abs=0.5)
        assert period.avg_pressure == pytest.approx(10.0, abs=0.01)
        assert period.avg_leak == pytest.approx(8.0, abs=0.01)

    async def test_invalid_period_type(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Invalid period type raises ValueError."""
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 1, 12, 0, 0),
            duration_hours=8.0,
            ahi=5.0,
        )
        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        with pytest.raises(ValueError, match="Invalid period_type: invalid"):
            _periods([day], "invalid")


class TestDayGranularity:
    """Test 'day' period type: boundaries, bucketing, and multi-device merging."""

    async def test_day_boundaries_contiguous(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Day granularity emits contiguous (d, d) tuples, one per day, inclusive."""
        device = async_test_device
        days_orm = []
        for i in range(3):
            d = date(2024, 3, 1) + timedelta(days=i)
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(d.year, d.month, d.day, 22, 0, 0),
                duration_hours=7.0,
                ahi=3.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days_orm.append(day)

        result = _periods(days_orm, "day")

        assert len(result) == 3
        for i, period in enumerate(result):
            expected = date(2024, 3, 1) + timedelta(days=i)
            assert period.period_start == expected
            assert period.period_end == expected
            assert period.days_in_period == 1

    async def test_day_bucketing_skips_gap_days(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Gap days with no therapy records are absent from the results."""
        device = async_test_device
        # Create sessions on day 1 and day 3, leaving day 2 as a gap
        for d in [date(2024, 5, 1), date(2024, 5, 3)]:
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(d.year, d.month, d.day, 22, 0, 0),
                duration_hours=7.0,
                ahi=2.0,
            )
            await DayManager.link_session_to_day(session, device.id, async_db_session)

        day_records = (
            (
                await async_db_session.execute(
                    select(Day).where(Day.device_id == device.id)
                )
            )
            .scalars()
            .all()
        )
        result = _periods(day_records, "day")

        result_dates = [p.period_start for p in result]
        assert date(2024, 5, 1) in result_dates
        assert date(2024, 5, 3) in result_dates
        # The gap day must not appear as a zero-AHI bucket
        assert date(2024, 5, 2) not in result_dates
        assert len(result) == 2

    async def test_day_multidevice_same_date_merges(
        self, async_db_session, async_test_session_factory, async_test_profile
    ):
        """Multiple Day rows sharing a date (one per device) merge into one bucket."""
        import uuid

        from snore.database.models import Device

        # Create two distinct devices (both owned by the same test profile)
        for _ in range(2):
            dev = Device(
                profile_id=async_test_profile.id,
                manufacturer="Mfr",
                model="M1",
                serial_number=f"SN_{uuid.uuid4().hex[:8]}",
            )
            async_db_session.add(dev)
        await async_db_session.flush()
        devices = (await async_db_session.execute(select(Device))).scalars().all()

        target_date = date(2024, 6, 15)
        for dev in devices:
            session = await async_test_session_factory(
                device_id=dev.id,
                start_time=datetime(2024, 6, 15, 22, 0, 0),
                duration_hours=8.0,
                ahi=4.0,
            )
            await DayManager.link_session_to_day(session, dev.id, async_db_session)

        day_records = (await async_db_session.execute(select(Day))).scalars().all()
        # Sanity: two Day rows exist for the same date
        same_date_days = [d for d in day_records if d.date == target_date]
        assert len(same_date_days) == 2

        result = _periods(day_records, "day")

        matching = [p for p in result if p.period_start == target_date]
        assert len(matching) == 1, "two devices on same date must merge to one bucket"
        assert matching[0].days_used == 2


class TestNewPeriodFields:
    """Test avg_oai, avg_cai, avg_hi, avg_rera population."""

    def _make_day(
        self,
        db_session: object,
        device: object,
        day_date: date,
        oai: float | None = None,
        cai: float | None = None,
        hi: float | None = None,
        reras: int = 0,
        total_therapy_hours: float = 8.0,
    ) -> Day:
        from snore.database.models import Day as DayModel

        day = DayModel(
            device_id=device.id,
            date=day_date,
            total_therapy_hours=total_therapy_hours,
            oai=oai,
            cai=cai,
            hi=hi,
            reras=reras,
        )
        db_session.add(day)
        db_session.flush()
        return day

    def test_oai_cai_hi_populated(self, db_session, test_device):
        """avg_oai, avg_cai, avg_hi are averaged across days in the period."""
        days = [
            self._make_day(
                db_session,
                test_device,
                date(2024, 7, i + 1),
                oai=float(i + 1),
                cai=float(i + 2),
                hi=float(i + 3),
            )
            for i in range(3)
        ]

        result = _periods(days, "month")

        assert len(result) == 1
        p = result[0]
        assert p.avg_oai == pytest.approx(2.0, abs=0.01)  # mean of 1,2,3
        assert p.avg_cai == pytest.approx(3.0, abs=0.01)  # mean of 2,3,4
        assert p.avg_hi == pytest.approx(4.0, abs=0.01)  # mean of 3,4,5

    def test_avg_rera_is_usage_weighted(self, db_session, test_device):
        """avg_rera = total RERAs / total therapy hours, not a mean of daily rates."""
        days = [
            self._make_day(
                db_session,
                test_device,
                date(2024, 8, 1),
                reras=8,
                total_therapy_hours=8.0,  # rate = 1.0
            ),
            self._make_day(
                db_session,
                test_device,
                date(2024, 8, 2),
                reras=6,
                total_therapy_hours=6.0,  # rate = 1.0
            ),
            self._make_day(
                db_session,
                test_device,
                date(2024, 8, 3),
                reras=12,
                total_therapy_hours=4.0,  # rate = 3.0
            ),
        ]

        result = _periods(days, "month")

        assert len(result) == 1
        # (8 + 6 + 12) / (8 + 6 + 4); the mean of daily rates would be 5/3.
        assert result[0].avg_rera == pytest.approx(26.0 / 18.0)

    def test_avg_rera_none_when_no_qualifying_days(self, db_session, test_device):
        """avg_rera is None when all days in a bucket have total_therapy_hours == 0."""
        days = [
            self._make_day(
                db_session,
                test_device,
                date(2024, 9, i + 1),
                reras=5,
                total_therapy_hours=0.0,
            )
            for i in range(2)
        ]

        result = _periods(days, "month")

        # Days exist so the bucket is NOT skipped; only avg_hours_per_day and avg_rera are None.
        assert len(result) == 1
        assert result[0].avg_hours_per_day is None
        assert result[0].avg_rera is None

    def _zero_rera_days(self, db_session: object, test_device: object) -> list[Day]:
        return [
            self._make_day(
                db_session,
                test_device,
                date(2024, 10, i + 1),
                reras=0,
                total_therapy_hours=8.0,
            )
            for i in range(2)
        ]

    def test_avg_rera_zero_on_rera_reporting_device(self, db_session, test_device):
        """Zero RERAs on a device that scores them is a real 0.0."""
        days = self._zero_rera_days(db_session, test_device)

        result = _periods(days, "month", "RE")

        assert result[0].avg_rera == 0.0

    def test_avg_rera_none_without_reporting_device(self, db_session, test_device):
        """Zero RERAs on a device that never scores them is no measurement."""
        days = self._zero_rera_days(db_session, test_device)

        result = _periods(days, "month")

        assert result[0].avg_rera is None

    def test_avg_rera_skips_days_of_non_reporting_device(
        self, db_session, test_device, test_profile
    ):
        """In a mixed period only the reporting device's days are averaged."""
        other = Device(
            profile_id=test_profile.id,
            manufacturer="Test Manufacturer",
            model="Other",
            serial_number="OTHER",
        )
        db_session.add(other)
        db_session.flush()
        days = [
            self._make_day(
                db_session,
                test_device,
                date(2024, 10, 1),
                reras=4,
                total_therapy_hours=8.0,  # rate = 0.5
            ),
            self._make_day(
                db_session,
                test_device,
                date(2024, 10, 2),
                reras=0,
                total_therapy_hours=8.0,  # rate = 0.0
            ),
            self._make_day(
                db_session, other, date(2024, 10, 3), reras=0, total_therapy_hours=8.0
            ),
        ]

        result = calculate_period_statistics(
            days, "month", {test_device.id: {"RE"}, other.id: set()}
        )

        assert result[0].avg_rera == pytest.approx(0.25)

    def test_oai_cai_hi_usage_weighted(self, db_session, test_device):
        """avg_oai/cai/hi weight each day by total_therapy_hours, not equally."""
        days = [
            self._make_day(
                db_session,
                test_device,
                date(2024, 7, 1),
                oai=10.0,
                cai=20.0,
                hi=30.0,
                total_therapy_hours=8.0,
            ),
            self._make_day(
                db_session,
                test_device,
                date(2024, 7, 2),
                oai=40.0,
                cai=60.0,
                hi=80.0,
                total_therapy_hours=2.0,
            ),
        ]

        result = _periods(days, "month")

        assert len(result) == 1
        p = result[0]
        # Usage-weighted, not the unweighted means of 25/40/55.
        assert p.avg_oai == pytest.approx((10 * 8 + 40 * 2) / 10.0)  # 16.0
        assert p.avg_cai == pytest.approx((20 * 8 + 60 * 2) / 10.0)  # 28.0
        assert p.avg_hi == pytest.approx((30 * 8 + 80 * 2) / 10.0)  # 40.0

    def test_pressure_leak_spo2_usage_weighted(self):
        """avg_pressure/leak/spo2 weight each day by total_therapy_hours."""

        def _day(
            day_date: date,
            hours: float,
            pressure: float,
            leak: float,
            spo2: float,
        ) -> Day:
            day = Day()
            day.date = day_date
            day.total_therapy_hours = hours
            day.pressure_median = pressure
            day.leak_median = leak
            day.spo2_mean = spo2
            day.spo2_min = None
            day.ahi = None
            day.oai = None
            day.cai = None
            day.hi = None
            day.reras = None
            return day

        days = [
            _day(date(2024, 7, 1), 8.0, 10.0, 5.0, 95.0),
            _day(date(2024, 7, 2), 2.0, 15.0, 25.0, 90.0),
        ]

        result = _periods(days, "month")

        assert len(result) == 1
        p = result[0]
        assert p.avg_pressure == pytest.approx((10 * 8 + 15 * 2) / 10.0)  # 11.0
        assert p.avg_leak == pytest.approx((5 * 8 + 25 * 2) / 10.0)  # 9.0
        assert p.avg_spo2 == pytest.approx((95 * 8 + 90 * 2) / 10.0)  # 94.0

    def test_new_fields_none_when_no_oai_cai_hi(self, db_session, test_device):
        """avg_oai/cai/hi are None when the underlying Day fields are all None."""
        day = self._make_day(
            db_session,
            test_device,
            date(2024, 11, 1),
            oai=None,
            cai=None,
            hi=None,
            total_therapy_hours=8.0,
        )

        result = _periods([day], "month")

        assert len(result) == 1
        assert result[0].avg_oai is None
        assert result[0].avg_cai is None
        assert result[0].avg_hi is None


class TestCalculateTrends:
    """Test trend data extraction."""

    async def test_trends_structure(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Trends returns dict with all 15 expected keys."""
        device = async_test_device
        start_date = date(2024, 11, 1)

        days = []
        for i in range(10):
            session_date = start_date + timedelta(days=i)
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(
                    session_date.year, session_date.month, session_date.day, 12, 0, 0
                ),
                duration_hours=7.5,
                ahi=5.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        period_stats = _periods(days, "week")
        result = calculate_trends(period_stats)

        expected_keys = {
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
        assert set(result.keys()) == expected_keys
        assert isinstance(result["ahi"], list)
        assert isinstance(result["usage"], list)

    async def test_trends_values(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Trend values match period statistics."""
        device = async_test_device
        start_date = date(2024, 11, 1)

        days = []
        for i in range(10):
            session_date = start_date + timedelta(days=i)
            session = await async_test_session_factory(
                device_id=device.id,
                start_time=datetime(
                    session_date.year, session_date.month, session_date.day, 12, 0, 0
                ),
                duration_hours=7.5,
                ahi=5.0,
            )
            day = await DayManager.link_session_to_day(
                session, device.id, async_db_session
            )
            days.append(day)

        period_stats = _periods(days, "week")
        result = calculate_trends(period_stats)

        assert len(result["ahi"]) == len(period_stats)
        for i, (trend_date, trend_ahi) in enumerate(result["ahi"]):
            assert trend_date == period_stats[i].period_start
            assert trend_ahi == period_stats[i].avg_ahi

    def test_trends_empty(self):
        """Empty period statistics returns empty lists for all 13 keys."""
        result = calculate_trends([])

        expected_keys = {
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
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert result[key] == []


class TestCalculateTrendsExtended:
    """Test the extended trends function with session extras."""

    def _make_period_stats(self, db_session: object, test_device: object) -> list:
        """Create two monthly period stats for testing."""
        from snore.database.models import Day as DayModel

        for month in [1, 2]:
            day = DayModel(
                device_id=test_device.id,
                date=date(2024, month, 15),
                total_therapy_hours=8.0,
                ahi=float(month),
            )
            db_session.add(day)
        db_session.flush()

        day_records = (
            db_session.query(DayModel)
            .filter(DayModel.device_id == test_device.id)
            .all()
        )
        return _periods(day_records, "month")

    def test_every_series_key_always_present(self, db_session, test_device):
        """calculate_trends_extended always returns exactly 13 keys."""
        period_stats = self._make_period_stats(db_session, test_device)
        result = calculate_trends_extended(period_stats, {})

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
        assert set(result.keys()) == expected

    def test_extras_absent_session_keys_are_none(self, db_session, test_device):
        """When session_extras is empty, epap/rr/pulse/mv entries are all None."""
        period_stats = self._make_period_stats(db_session, test_device)
        result = calculate_trends_extended(period_stats, {})

        for key in ("epap", "rr", "pulse", "mv"):
            for _, value in result[key]:
                assert value is None

    def test_extras_values_land_on_correct_period_start(self, db_session, test_device):
        """Values from session_extras are associated with the correct period_start."""
        period_stats = self._make_period_stats(db_session, test_device)
        jan_start = date(2024, 1, 1)
        feb_start = date(2024, 2, 1)

        extras = {
            jan_start: {"epap": 5.0, "rr": 14.0, "pulse": 62.0, "mv": 7.5},
            feb_start: {"epap": 6.0, "rr": 15.0, "pulse": 65.0, "mv": 8.0},
        }

        result = calculate_trends_extended(period_stats, extras)

        epap_by_date = dict(result["epap"])
        assert epap_by_date[jan_start] == pytest.approx(5.0)
        assert epap_by_date[feb_start] == pytest.approx(6.0)

        pulse_by_date = dict(result["pulse"])
        assert pulse_by_date[jan_start] == pytest.approx(62.0)
        assert pulse_by_date[feb_start] == pytest.approx(65.0)

    def test_empty_period_stats_returns_empty_lists(self):
        """Empty period_stats yields empty lists for all 13 keys."""
        result = calculate_trends_extended([], {})

        assert all(v == [] for v in result.values())
        assert len(result) == 13


class TestCalculateAhiTrendDirection:
    def test_empty_returns_none(self) -> None:
        assert calculate_ahi_trend_direction([]) is None

    def test_single_value_returns_none(self) -> None:
        assert calculate_ahi_trend_direction([5.0]) is None

    def test_zero_prior_average_returns_none(self) -> None:
        assert calculate_ahi_trend_direction([0.0, 0.0, 3.0]) is None

    def test_improving(self) -> None:
        # latest = 1.0, prior_avg = 5.0 → 1.0 < 4.5 → improving
        result = calculate_ahi_trend_direction([5.0, 5.0, 1.0])
        assert result == "improving"

    def test_worsening(self) -> None:
        # latest = 10.0, prior_avg = 5.0 → 10.0 > 5.5 → worsening
        result = calculate_ahi_trend_direction([5.0, 5.0, 10.0])
        assert result == "worsening"

    def test_stable(self) -> None:
        # latest = 5.2, prior_avg = 5.0 → within ±10% → stable
        result = calculate_ahi_trend_direction([5.0, 5.0, 5.2])
        assert result == "stable"


class TestNullGuardInPeriodStatistics:
    """NULL total_therapy_hours/reras must not crash calculate_period_statistics.

    SQLite ALTER TABLE ADD COLUMN leaves existing rows NULL even when the ORM
    column is declared non-nullable.  These tests verify the guard condition
    safely excludes such rows from avg_rera rather than raising TypeError.
    """

    def test_null_total_therapy_hours_excluded_from_avg_rera(self):
        """Day with NULL total_therapy_hours is left out of avg_rera without error."""
        day = Day()
        day.date = date(2024, 1, 15)
        day.reras = 5
        day.total_therapy_hours = None  # simulates ALTER TABLE ADD COLUMN NULL
        day.ahi = None
        day.pressure_median = None
        day.leak_median = None
        day.spo2_mean = None
        day.spo2_min = None
        day.oai = None
        day.cai = None
        day.hi = None

        result = _periods([day], "month")

        assert len(result) == 1
        assert result[0].avg_rera is None

    def test_null_reras_excluded_from_avg_rera(self):
        """Day with NULL reras is left out of avg_rera without error."""
        day = Day()
        day.date = date(2024, 1, 15)
        day.reras = None  # simulates ALTER TABLE ADD COLUMN NULL
        day.total_therapy_hours = 8.0
        day.ahi = None
        day.pressure_median = None
        day.leak_median = None
        day.spo2_mean = None
        day.spo2_min = None
        day.oai = None
        day.cai = None
        day.hi = None

        result = _periods([day], "month")

        assert len(result) == 1
        assert result[0].avg_rera is None
