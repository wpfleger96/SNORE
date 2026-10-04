"""Unit tests for DayService."""

import logging

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Day, Device, Session
from snore.exceptions import NotFoundError
from snore.provenance import IndexSource
from snore.services.day_service import DayService


async def _create_day(
    db_session: AsyncSession,
    device: Device,
    day_date: date,
    ahi: float | None = None,
    total_therapy_hours: float = 8.0,
    session_count: int = 1,
) -> Day:
    """Helper to create a Day record with an optional linked Session."""
    day = Day(
        device_id=device.id,
        date=day_date,
        session_count=session_count,
        total_therapy_hours=total_therapy_hours,
        ahi=ahi,
        oai=0.1,
        cai=0.2,
        hi=0.3,
        pressure_median=10.0,
        leak_median=5.0,
        spo2_mean=96.5,
    )
    db_session.add(day)
    await db_session.flush()
    return day


async def _create_session_for_day(
    db_session: AsyncSession, device: Device, day: Day
) -> Session:
    """Create a Session linked to a Day."""
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"day_test_{day.date.isoformat()}",
        start_time=datetime.combine(day.date, datetime.min.time()),
        end_time=datetime.combine(day.date, datetime.min.time()) + timedelta(hours=8),
        duration_seconds=28800,
    )
    db_session.add(sess)
    await db_session.flush()
    return sess


class TestDayServiceList:
    async def test_list_empty(self, async_db_session, async_test_device):
        """Empty database returns empty list and zero total."""
        service = DayService(async_db_session, profile_id=1)
        items, total = await service.list_days()
        assert items == []
        assert total == 0

    async def test_list_returns_all_days(self, async_db_session, async_test_device):
        """All days returned when no filters applied."""
        await _create_day(
            async_db_session, async_test_device, date(2025, 1, 1), ahi=2.0
        )
        await _create_day(
            async_db_session, async_test_device, date(2025, 1, 2), ahi=3.0
        )
        await _create_day(
            async_db_session, async_test_device, date(2025, 1, 3), ahi=4.0
        )
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, total = await service.list_days(limit=0)

        assert total == 3
        assert len(items) == 3

    async def test_list_sorted_desc_by_date(self, async_db_session, async_test_device):
        """Results are sorted by date descending."""
        await _create_day(async_db_session, async_test_device, date(2025, 1, 1))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 3))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 2))
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, _ = await service.list_days(limit=0)

        assert items[0].date == date(2025, 1, 3)
        assert items[1].date == date(2025, 1, 2)
        assert items[2].date == date(2025, 1, 1)

    async def test_list_with_from_date_filter(
        self, async_db_session, async_test_device
    ):
        """from_date filter excludes older records."""
        await _create_day(async_db_session, async_test_device, date(2025, 1, 1))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 5))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 10))
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, total = await service.list_days(from_date=date(2025, 1, 5), limit=0)

        assert total == 2
        assert all(item.date >= date(2025, 1, 5) for item in items)

    async def test_list_with_to_date_filter(self, async_db_session, async_test_device):
        """to_date filter excludes newer records."""
        await _create_day(async_db_session, async_test_device, date(2025, 1, 1))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 5))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 10))
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, total = await service.list_days(to_date=date(2025, 1, 5), limit=0)

        assert total == 2
        assert all(item.date <= date(2025, 1, 5) for item in items)

    async def test_list_with_date_range(self, async_db_session, async_test_device):
        """Combined from_date/to_date range filters correctly."""
        await _create_day(async_db_session, async_test_device, date(2025, 1, 1))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 5))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 10))
        await _create_day(async_db_session, async_test_device, date(2025, 1, 15))
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, total = await service.list_days(
            from_date=date(2025, 1, 4), to_date=date(2025, 1, 11), limit=0
        )

        assert total == 2
        assert {item.date for item in items} == {date(2025, 1, 5), date(2025, 1, 10)}

    async def test_list_with_device_id_filter(
        self, async_db_session, async_test_device, async_test_profile
    ):
        """device_id filter restricts to that device only."""
        from snore.database.models import Device

        other_device = Device(
            profile_id=async_test_profile.id,
            manufacturer="Other",
            model="Model",
            serial_number="OTHER_RX_999",
        )
        async_db_session.add(other_device)
        await async_db_session.flush()

        await _create_day(async_db_session, async_test_device, date(2025, 1, 1))
        await _create_day(async_db_session, other_device, date(2025, 1, 2))
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, total = await service.list_days(device_id=async_test_device.id, limit=0)

        assert total == 1
        assert items[0].device_id == async_test_device.id

    async def test_list_limit_applied(self, async_db_session, async_test_device):
        """limit parameter restricts number of items returned."""
        for i in range(10):
            await _create_day(
                async_db_session,
                async_test_device,
                date(2025, 1, 1) + timedelta(days=i),
            )
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, total = await service.list_days(limit=3)

        assert total == 10
        assert len(items) == 3

    async def test_list_offset_applied(self, async_db_session, async_test_device):
        """offset parameter skips records."""
        for i in range(5):
            await _create_day(
                async_db_session,
                async_test_device,
                date(2025, 1, 1) + timedelta(days=i),
            )
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items_page1, total = await service.list_days(limit=2, offset=0)
        items_page2, _ = await service.list_days(limit=2, offset=2)

        assert total == 5
        assert len(items_page1) == 2
        assert len(items_page2) == 2
        # Pages should not overlap
        dates_page1 = {item.date for item in items_page1}
        dates_page2 = {item.date for item in items_page2}
        assert dates_page1.isdisjoint(dates_page2)

    async def test_list_item_fields(self, async_db_session, async_test_device):
        """DayListItem contains expected fields with correct values."""
        await _create_day(
            async_db_session,
            async_test_device,
            date(2025, 6, 15),
            ahi=2.5,
            total_therapy_hours=7.0,
        )
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        items, _ = await service.list_days()

        assert len(items) == 1
        item = items[0]
        assert item.date == date(2025, 6, 15)
        assert item.device_id == async_test_device.id
        assert item.session_count == 1
        assert item.total_therapy_hours == pytest.approx(7.0)
        assert item.ahi == pytest.approx(2.5)


class TestDayServiceGet:
    async def test_get_nonexistent(self, async_db_session):
        """Raises NotFoundError for a date with no Day record."""
        service = DayService(async_db_session, profile_id=1)
        with pytest.raises(NotFoundError):
            await service.get_day(date(2024, 1, 1))

    async def test_get_existing_day(self, async_db_session, async_test_device):
        """Returns full DayDetail with all stats fields."""
        await _create_day(
            async_db_session,
            async_test_device,
            date(2025, 4, 10),
            ahi=3.5,
            total_therapy_hours=8.0,
        )
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        result = await service.get_day(date(2025, 4, 10))

        assert result is not None
        assert result.date == date(2025, 4, 10)
        assert result.device_id == async_test_device.id
        assert result.ahi == pytest.approx(3.5)
        assert result.oai == pytest.approx(0.1)
        assert result.cai == pytest.approx(0.2)
        assert result.hi == pytest.approx(0.3)
        assert result.avg_pressure == pytest.approx(10.0)
        assert result.avg_leak == pytest.approx(5.0)
        assert result.avg_spo2 == pytest.approx(96.5)

    async def test_get_day_maps_renamed_stats_and_counts(
        self, async_db_session, async_test_device
    ):
        """Renamed Day columns, identity stats, and event counts reach DayDetail."""
        async_db_session.add(
            Day(
                device_id=async_test_device.id,
                date=date(2025, 4, 13),
                session_count=2,
                total_therapy_hours=7.5,
                obstructive_apneas=4,
                central_apneas=3,
                hypopneas=2,
                reras=1,
                pressure_median=9.5,
                pressure_95th=12.0,
                leak_median=4.0,
                leak_95th=20.0,
                epap_95th=8.0,
                spo2_mean=95.0,
                spo2_min=88.0,
                ahi_computed=1.3,
                hi_computed=0.2,
            )
        )
        await async_db_session.flush()

        result = await DayService(async_db_session, profile_id=1).get_day(
            date(2025, 4, 13)
        )

        assert (result.avg_pressure, result.avg_leak, result.avg_spo2) == (
            9.5,
            4.0,
            95.0,
        )
        assert (result.pressure_median, result.pressure_95th) == (9.5, 12.0)
        assert (result.leak_95th, result.epap_95th, result.spo2_min) == (
            20.0,
            8.0,
            88.0,
        )
        assert (
            result.obstructive_apneas,
            result.central_apneas,
            result.hypopneas,
            result.reras,
        ) == (4, 3, 2, 1)
        assert (result.session_count, result.total_therapy_hours) == (2, 7.5)
        assert (result.ahi_computed, result.hi_computed) == (1.3, 0.2)
        assert result.oai_computed is None
        assert result.session_ids == []

    async def test_get_device_headline_day_exposes_source_and_recount(
        self, async_db_session, async_test_device
    ):
        """DayDetail carries the headline indices, their source, and the recount."""
        async_db_session.add(
            Day(
                device_id=async_test_device.id,
                date=date(2025, 4, 11),
                session_count=1,
                total_therapy_hours=8.0,
                ahi=1.2,
                oai=0.4,
                cai=0.3,
                hi=0.5,
                index_source=IndexSource.DEVICE,
                ahi_computed=1.6,
                oai_computed=0.6,
                cai_computed=0.4,
                hi_computed=0.6,
            )
        )
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        result = await service.get_day(date(2025, 4, 11))
        items, _ = await service.list_days()

        assert result.index_source == IndexSource.DEVICE
        assert (result.ahi, result.oai, result.cai, result.hi) == pytest.approx(
            (1.2, 0.4, 0.3, 0.5)
        )
        assert (
            result.ahi_computed,
            result.oai_computed,
            result.cai_computed,
            result.hi_computed,
        ) == pytest.approx((1.6, 0.6, 0.4, 0.6))
        assert items[0].index_source == IndexSource.DEVICE
        assert items[0].ahi_computed == pytest.approx(1.6)

    async def test_get_derived_headline_day_reports_derived_source(
        self, async_db_session, async_test_device
    ):
        """A recount headline is labelled 'derived'."""
        async_db_session.add(
            Day(
                device_id=async_test_device.id,
                date=date(2025, 4, 12),
                session_count=1,
                total_therapy_hours=8.0,
                ahi=1.6,
                ahi_computed=1.6,
                index_source=IndexSource.DERIVED,
            )
        )
        await async_db_session.flush()

        result = await DayService(async_db_session, profile_id=1).get_day(
            date(2025, 4, 12)
        )

        assert result.index_source == IndexSource.DERIVED
        assert result.ahi == pytest.approx(result.ahi_computed)

    async def test_get_day_with_session_ids(self, async_db_session, async_test_device):
        """DayDetail includes linked session IDs."""
        day = await _create_day(async_db_session, async_test_device, date(2025, 5, 20))
        sess1 = await _create_session_for_day(async_db_session, async_test_device, day)
        # Second session same day
        sess2 = Session(
            device_id=async_test_device.id,
            day_id=day.id,
            device_session_id="day_test_2025-05-20_b",
            start_time=datetime(2025, 5, 20, 4, 0),
            end_time=datetime(2025, 5, 20, 10, 0),
            duration_seconds=21600,
        )
        async_db_session.add(sess2)
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        result = await service.get_day(date(2025, 5, 20))

        assert result is not None
        assert set(result.session_ids) == {sess1.id, sess2.id}

    async def test_get_day_empty_session_ids(self, async_db_session, async_test_device):
        """DayDetail has empty session_ids when no sessions linked."""
        await _create_day(async_db_session, async_test_device, date(2025, 7, 1))
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        result = await service.get_day(date(2025, 7, 1))

        assert result is not None
        assert result.session_ids == []

    async def test_get_day_two_devices_same_date_no_exception(
        self, async_db_session, async_test_device, async_test_profile, caplog
    ):
        """When two devices both have a Day on the same date, get_day returns the
        first (by device_id) rather than raising MultipleResultsFound."""
        second_device = Device(
            profile_id=async_test_profile.id,
            manufacturer="OtherCo",
            model="OtherModel",
            serial_number="DAY_SVC_TEST_002",
        )
        async_db_session.add(second_device)
        await async_db_session.flush()

        await _create_day(
            async_db_session, async_test_device, date(2025, 8, 1), ahi=2.0
        )
        await _create_day(async_db_session, second_device, date(2025, 8, 1), ahi=4.0)
        await async_db_session.flush()

        service = DayService(async_db_session, async_test_profile.id)
        with caplog.at_level(logging.WARNING, logger="snore.services.day_service"):
            result = await service.get_day(date(2025, 8, 1))

        assert result is not None
        # Should return the lower device_id deterministically
        assert result.device_id == min(async_test_device.id, second_device.id)
        assert "Multiple Day rows" in caplog.text

    async def test_get_day_device_id_filter_selects_correct_device(
        self, async_db_session, async_test_device, async_test_profile
    ):
        """device_id= parameter selects the specific device's Day row."""
        second_device = Device(
            profile_id=async_test_profile.id,
            manufacturer="OtherCo",
            model="OtherModel",
            serial_number="DAY_SVC_TEST_003",
        )
        async_db_session.add(second_device)
        await async_db_session.flush()

        await _create_day(
            async_db_session, async_test_device, date(2025, 8, 2), ahi=2.0
        )
        await _create_day(async_db_session, second_device, date(2025, 8, 2), ahi=4.0)
        await async_db_session.flush()

        service = DayService(async_db_session, async_test_profile.id)
        result = await service.get_day(date(2025, 8, 2), device_id=second_device.id)

        assert result is not None
        assert result.device_id == second_device.id
        assert result.ahi == pytest.approx(4.0)


class TestDayServiceFlReraProxy:
    async def test_get_day_populates_fl_rera_when_analysis_present(
        self, async_db_session, async_test_device, monkeypatch
    ):
        """FL/RERA proxy fields carry BreathService values onto DayDetail.

        ``rera_count_reason`` is aliased from the DTO's ``rera_reason``.
        """

        async def _fake_summary(self, therapy_date, device_id=None, **kwargs):
            return SimpleNamespace(
                fl_class_ge4_pct=12.5,
                fl_class_ge4_pct_reason=None,
                rera_index=3.2,
                rera_index_reason=None,
                rera_count=7,
                rera_reason=None,
            )

        monkeypatch.setattr(
            "snore.services.breath_service.BreathService.get_nightly_summary",
            _fake_summary,
        )

        day = await _create_day(async_db_session, async_test_device, date(2025, 9, 1))
        await _create_session_for_day(async_db_session, async_test_device, day)
        await async_db_session.flush()

        service = DayService(async_db_session, profile_id=1)
        result = await service.get_day(date(2025, 9, 1))

        assert result.fl_class_ge4_pct == pytest.approx(12.5)
        assert result.fl_class_ge4_pct_reason is None
        assert result.rera_index == pytest.approx(3.2)
        assert result.rera_index_reason is None
        assert result.rera_count == 7
        assert result.rera_count_reason is None

    async def test_get_day_nulls_fl_rera_when_analysis_absent(
        self, async_db_session, async_test_device
    ):
        """An un-analyzed night (sessions present, none OK) nulls with
        ``not_available`` — the success path, not the exception path."""
        day = await _create_day(async_db_session, async_test_device, date(2025, 9, 2))
        await _create_session_for_day(async_db_session, async_test_device, day)
        await async_db_session.flush()

        service = DayService(async_db_session, async_test_device.profile_id)
        result = await service.get_day(date(2025, 9, 2))

        assert result.fl_class_ge4_pct is None
        assert result.fl_class_ge4_pct_reason == "not_available"
        assert result.rera_index is None
        assert result.rera_index_reason == "not_available"
        assert result.rera_count is None
        assert result.rera_count_reason == "not_available"

    async def test_get_day_nulls_and_warns_when_day_has_no_sessions(
        self, async_db_session, async_test_device, caplog
    ):
        """A Day row with no sessions is anomalous: null with ``analysis_not_run``
        and a logged warning naming the date."""
        await _create_day(async_db_session, async_test_device, date(2025, 9, 3))
        await async_db_session.flush()

        service = DayService(async_db_session, async_test_device.profile_id)
        with caplog.at_level(logging.WARNING, logger="snore.services.day_service"):
            result = await service.get_day(date(2025, 9, 3))

        assert result.fl_class_ge4_pct_reason == "analysis_not_run"
        assert result.rera_index_reason == "analysis_not_run"
        assert result.rera_count_reason == "analysis_not_run"
        assert "no analyzable sessions" in caplog.text
        assert "2025-09-03" in caplog.text

    async def test_get_day_nulls_without_warning_when_all_sessions_disabled(
        self, async_db_session, async_test_device, caplog
    ):
        """Disabling every session of a night is a normal user choice: the
        FL/RERA fields null with ``analysis_not_run`` but nothing is warned."""
        day = await _create_day(async_db_session, async_test_device, date(2025, 9, 4))
        sess = await _create_session_for_day(async_db_session, async_test_device, day)
        sess.enabled = False
        await async_db_session.flush()

        service = DayService(async_db_session, async_test_device.profile_id)
        with caplog.at_level(logging.WARNING, logger="snore.services.day_service"):
            result = await service.get_day(date(2025, 9, 4))

        assert result.rera_index_reason == "analysis_not_run"
        assert result.fl_class_ge4_pct_reason == "analysis_not_run"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
