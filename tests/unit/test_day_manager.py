"""
Tests for DayManager business logic.

Tests the critical day-splitting algorithm and statistical aggregation logic
that determines which calendar day sessions belong to and how statistics
are aggregated across multiple sessions.
"""

from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta

import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.day_manager import DayManager
from snore.database.models import Day, Device, Session
from snore.provenance import IndexSource


class TestDaySplitLogic:
    """Test day-splitting algorithm (hardcoded noon boundary logic).

    These tests exercise a pure classmethod with no DB access — the
    async_db_session fixture is included for consistency but not used.
    """

    def test_default_noon_split_before(self):
        """Session before noon belongs to previous calendar day."""
        session_start = datetime(2024, 11, 5, 11, 59, 0)
        result = DayManager.get_day_for_session(session_start)
        assert result == date(2024, 11, 4)

    def test_default_noon_split_at_boundary(self):
        """Session exactly at noon belongs to same calendar day."""
        session_start = datetime(2024, 11, 5, 12, 0, 0)
        result = DayManager.get_day_for_session(session_start)
        assert result == date(2024, 11, 5)

    def test_default_noon_split_after(self):
        """Session after noon belongs to same calendar day."""
        session_start = datetime(2024, 11, 5, 12, 1, 0)
        result = DayManager.get_day_for_session(session_start)
        assert result == date(2024, 11, 5)

    def test_midnight_session_with_noon_split(self):
        """Midnight session (00:00) with noon split belongs to previous day."""
        session_start = datetime(2024, 11, 5, 0, 0, 0)
        result = DayManager.get_day_for_session(session_start)
        assert result == date(2024, 11, 4)

    def test_late_night_session_23_59(self):
        """Late night session (23:59) with noon split belongs to same day."""
        session_start = datetime(2024, 11, 5, 23, 59, 0)
        result = DayManager.get_day_for_session(session_start)
        assert result == date(2024, 11, 5)


class TestStatisticalAggregation:
    """Test statistical aggregation across multiple sessions."""

    async def test_single_session_aggregation(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Single session aggregation should copy statistics directly."""
        device = async_test_device

        session_start = datetime(2024, 11, 5, 12, 0, 0)
        session = await async_test_session_factory(
            device_id=device.id,
            start_time=session_start,
            duration_hours=8.0,
            obstructive_apneas=10,
            central_apneas=5,
            hypopneas=8,
            reras=3,
            ahi=5.0,
            oai=1.25,
            cai=0.625,
            hi=1.0,
            pressure_min=8.0,
            pressure_max=15.0,
            pressure_median=11.0,
            pressure_mean=11.5,
            leak_min=0.0,
            leak_max=24.0,
            leak_median=5.0,
        )

        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        assert day.session_count == 1
        assert day.total_therapy_hours == pytest.approx(8.0, abs=0.01)
        assert day.obstructive_apneas == 10
        assert day.central_apneas == 5
        assert day.hypopneas == 8
        assert day.reras == 3
        assert day.ahi == pytest.approx(5.0, abs=0.01)
        assert day.pressure_min == pytest.approx(8.0, abs=0.01)
        assert day.pressure_max == pytest.approx(15.0, abs=0.01)

    async def test_multi_session_event_counts_sum(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Event counts should sum across sessions."""
        device = async_test_device

        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            obstructive_apneas=10,
            central_apneas=5,
            hypopneas=8,
            reras=3,
        )

        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=6.0,
            obstructive_apneas=15,
            central_apneas=3,
            hypopneas=12,
            reras=5,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session2, device.id, async_db_session
        )

        assert day.session_count == 2
        assert day.obstructive_apneas == 25
        assert day.central_apneas == 8
        assert day.hypopneas == 20
        assert day.reras == 8

    async def test_weighted_average_ahi(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """AHI should be weighted by session duration."""
        device = async_test_device

        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            ahi=10.0,
        )

        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=2.0,
            ahi=4.0,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session2, device.id, async_db_session
        )

        assert day.ahi == pytest.approx(8.0, abs=0.01)

    async def test_pressure_min_max_across_sessions(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Pressure min/max should be extremes across all sessions."""
        device = async_test_device

        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            pressure_min=8.0,
            pressure_max=15.0,
            pressure_median=11.0,
        )

        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=4.0,
            pressure_min=6.0,
            pressure_max=12.0,
            pressure_median=9.0,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session2, device.id, async_db_session
        )

        assert day.pressure_min == pytest.approx(6.0, abs=0.01)
        assert day.pressure_max == pytest.approx(15.0, abs=0.01)

    async def test_empty_day_resets_statistics(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Day with no sessions should have reset statistics."""
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=8.0,
            obstructive_apneas=10,
            ahi=5.0,
        )

        day = await DayManager.link_session_to_day(session, device.id, async_db_session)
        assert day.session_count == 1
        assert day.obstructive_apneas == 10

        session.day_id = None
        await async_db_session.flush()
        await DayManager.aggregate_day_statistics(day, async_db_session)

        assert day.session_count == 0
        assert day.total_therapy_hours == 0.0
        assert day.obstructive_apneas == 0
        assert day.ahi is None

    async def test_empty_day_resets_epap_statistics(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Regression: epap fields must reset to None when a day loses all sessions.

        The reset block previously cleared pressure/leak/spo2 but left stale
        epap aggregates behind.
        """
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=8.0,
            epap_min=5.0,
            epap_max=9.0,
            epap_median=7.0,
            epap_mean=7.2,
            epap_95th=8.5,
        )

        day = await DayManager.link_session_to_day(session, device.id, async_db_session)
        assert day.epap_min == pytest.approx(5.0, abs=0.01)
        assert day.epap_max == pytest.approx(9.0, abs=0.01)
        assert day.epap_median == pytest.approx(7.0, abs=0.01)
        assert day.epap_mean == pytest.approx(7.2, abs=0.01)
        assert day.epap_95th == pytest.approx(8.5, abs=0.01)

        session.day_id = None
        await async_db_session.flush()
        await DayManager.aggregate_day_statistics(day, async_db_session)

        assert day.epap_min is None
        assert day.epap_max is None
        assert day.epap_median is None
        assert day.epap_mean is None
        assert day.epap_95th is None

    async def test_partial_data_null_values(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Sessions with missing statistics should be handled gracefully."""
        device = async_test_device

        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
            ahi=8.0,
        )

        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=4.0,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session2, device.id, async_db_session
        )

        assert day.ahi == pytest.approx(8.0, abs=0.01)

    async def test_zero_duration_session_handling(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Sessions with zero duration should not cause division by zero."""
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=0.0,
            ahi=5.0,
        )

        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        assert day.total_therapy_hours == pytest.approx(0.0, abs=0.01)

    async def test_total_therapy_hours_sums_durations(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Total therapy hours should sum all session durations."""
        device = async_test_device

        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=4.0,
        )

        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 18, 0, 0),
            duration_hours=2.5,
        )

        session3 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=1.5,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        await DayManager.link_session_to_day(session2, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session3, device.id, async_db_session
        )

        assert day.total_therapy_hours == pytest.approx(8.0, abs=0.01)
        assert day.session_count == 3

    async def test_total_therapy_hours_prefers_usage_hours(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """total_therapy_hours uses statistics.usage_hours when present.

        A gap-merged session may span 24h but only have 5.9h of mask-on time.
        The day total must reflect actual usage, not the inflated span.
        """
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=24.0,
            usage_hours=5.9,
        )

        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        assert day.total_therapy_hours == pytest.approx(5.9, abs=0.001)

    async def test_total_therapy_hours_falls_back_to_span(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """total_therapy_hours falls back to span/3600 when usage_hours is absent."""
        device = async_test_device

        # No stats_kwargs → no Statistics row created → fallback to span
        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=6.5,
        )

        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        assert day.total_therapy_hours == pytest.approx(6.5, abs=0.001)

    async def test_total_therapy_hours_mixed_sessions(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Mixed day: usage_hours present on one session, absent on the other."""
        device = async_test_device

        # Session with usage_hours: span 24h, actual usage 5.9h
        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=24.0,
            usage_hours=5.9,
        )

        # Session without statistics: falls back to span (2.0h)
        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 6, 2, 0, 0),
            duration_hours=2.0,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session2, device.id, async_db_session
        )

        assert day.total_therapy_hours == pytest.approx(7.9, abs=0.001)

    async def test_weighted_average_uses_usage_hours_not_span(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Weighted average weights follow usage_hours, not session span.

        Session 1: span 24h, usage 2h, ahi 10.0
        Session 2: span 2h,  usage 8h, ahi 2.0
        Span-weighted:  (10*24 + 2*2) / 26 ≈ 9.38  (wrong — dominated by gap span)
        Usage-weighted: (10*2  + 2*8) / 10 = 3.6   (correct)
        """
        device = async_test_device

        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=24.0,
            usage_hours=2.0,
            ahi=10.0,
        )

        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 6, 2, 0, 0),
            duration_hours=2.0,
            usage_hours=8.0,
            ahi=2.0,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session2, device.id, async_db_session
        )

        # usage-weighted: (10*2 + 2*8) / (2+8) = 36/10 = 3.6
        assert day.ahi == pytest.approx(3.6, abs=0.01)

    async def test_stat_session_pairing_regression(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Regression: stats/session pairs must align by ownership, not list position.

        Three sessions: only session 1 and session 3 have Statistics.
        Session 1: span 1h, usage_hours absent (None), ahi 10.0
        Session 2: span 5h, no Statistics row
        Session 3: span 6h, usage_hours absent (None), ahi 4.0

        With the old zip-misaligned code, stats3 was paired with session2 (span 5h),
        giving weighted avg = (10*1 + 4*5) / (1+5) = 5.0.
        With aligned pairs, stats3 is paired with session3 (span 6h),
        giving weighted avg = (10*1 + 4*6) / (1+6) = 34/7 ≈ 4.857.
        """
        device = async_test_device

        session1 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=1.0,
            ahi=10.0,
        )
        session2 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 14, 0, 0),
            duration_hours=5.0,
        )
        session3 = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 20, 0, 0),
            duration_hours=6.0,
            ahi=4.0,
        )

        await DayManager.link_session_to_day(session1, device.id, async_db_session)
        await DayManager.link_session_to_day(session2, device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            session3, device.id, async_db_session
        )

        # Correctly paired: (10*1 + 4*6) / (1+6) = 34/7 ≈ 4.857
        expected = (10.0 * 1.0 + 4.0 * 6.0) / (1.0 + 6.0)
        assert day.ahi == pytest.approx(expected, abs=0.001)
        # Confirm the old misaligned result is not returned
        misaligned = (10.0 * 1.0 + 4.0 * 5.0) / (1.0 + 5.0)
        assert day.ahi != pytest.approx(misaligned, abs=0.001)

    async def test_zero_usage_hours_contributes_nothing_to_total(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """usage_hours == 0.0 must contribute 0 to total_therapy_hours, not span.

        The old truthy check treated 0.0 as falsy and fell back to the session span.
        """
        device = async_test_device

        # usage_hours=0.0 means known-zero mask-on time; span (5h) must not bleed in.
        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 12, 0, 0),
            duration_hours=5.0,
            usage_hours=0.0,
        )

        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        assert day.total_therapy_hours == pytest.approx(0.0, abs=0.001)


class TestDayPruning:
    """Test DayManager.recalculate_day's orphan-pruning lifecycle rule."""

    async def test_recalculate_day_prunes_unreferenced_day(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """A day no Session row references is deleted, not reset in place."""
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=8.0,
        )
        day = await DayManager.link_session_to_day(session, device.id, async_db_session)
        day_id = day.id

        await async_db_session.delete(session)
        await async_db_session.flush()

        assert await DayManager.recalculate_day(day, async_db_session) is False
        assert await async_db_session.get(Day, day_id) is None

    async def test_recalculate_day_keeps_day_with_only_disabled_sessions(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """A disabled session still references its day, so the row survives
        with zeroed aggregates rather than being pruned."""
        device = async_test_device

        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=8.0,
            ahi=5.0,
        )
        day = await DayManager.link_session_to_day(session, device.id, async_db_session)
        day_id = day.id

        session.enabled = False
        await async_db_session.flush()

        assert await DayManager.recalculate_day(day, async_db_session) is True

        await async_db_session.flush()
        async_db_session.expire_all()
        stored = (
            await async_db_session.execute(select(Day).where(Day.id == day_id))
        ).scalar_one()
        assert stored.session_count == 0
        assert stored.ahi is None

    async def test_recalculate_day_sees_pending_session_before_pruning(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """A Session added but not yet flushed still counts as a reference.

        The existence probe is a Core statement that bypasses autoflush; without
        an explicit flush the pending session would be invisible and the day
        deleted out from under it.
        """
        device = async_test_device

        first = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=8.0,
        )
        day = await DayManager.link_session_to_day(first, device.id, async_db_session)
        day_id = day.id

        await async_db_session.delete(first)
        await async_db_session.flush()

        start = datetime(2024, 11, 6, 2, 0, 0)
        pending = Session(
            device_id=device.id,
            device_session_id="pending_session",
            start_time=start,
            end_time=start + timedelta(hours=4),
            duration_seconds=4 * 3600,
            day_id=day_id,
        )
        async_db_session.add(pending)

        assert await DayManager.recalculate_day(day, async_db_session) is True
        assert await async_db_session.get(Day, day_id) is day
        assert day.session_count == 1


# usage_hours_device matches the two-session day's 6 h + 2 h of imported time.
_DEVICE = {
    "ahi_device": 2.5,
    "oai_device": 1.0,
    "cai_device": 0.5,
    "hi_device": 1.0,
    "usage_hours_device": 8.0,
}


class TestDeviceIndexHeadline:
    """Day headline prefers the device-reported daily AHI when it is trusted.

    The two-session day has a recount AHI of (4*6 + 8*2) / 8 = 5.0, distinct
    from the device value 2.5, so each assertion shows which source won.  Its
    imported mask-on time is 8 h unless a test overrides ``usage_hours``.
    """

    async def _two_session_day(
        self,
        db: AsyncSession,
        device: Device,
        factory: Callable[..., Awaitable[Session]],
        first_device: dict[str, float] = _DEVICE,
        second_device: dict[str, float] = _DEVICE,
    ) -> tuple[Day, Session, Session]:
        first = await factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=6.0,
            ahi=4.0,
            oai=2.0,
            cai=1.0,
            hi=1.0,
            **first_device,
        )
        second = await factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 6, 4, 0, 0),
            duration_hours=2.0,
            ahi=8.0,
            oai=4.0,
            cai=2.0,
            hi=2.0,
            **second_device,
        )
        await DayManager.link_session_to_day(first, device.id, db)
        day = await DayManager.link_session_to_day(second, device.id, db)
        return day, first, second

    async def test_identical_device_values_headline_device(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, _, _ = await self._two_session_day(
            async_db_session, async_test_device, async_test_session_factory
        )

        assert day.index_source == IndexSource.DEVICE
        assert (day.ahi, day.oai, day.cai, day.hi) == (2.5, 1.0, 0.5, 1.0)
        assert day.ahi_computed == pytest.approx(5.0)
        assert day.oai_computed == pytest.approx(2.5)

    async def test_partial_device_values_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            second_device={},
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    async def test_disabled_session_on_day_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        device = async_test_device
        day, _, _ = await self._two_session_day(
            async_db_session, device, async_test_session_factory
        )
        disabled = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 6, 7, 0, 0),
            duration_hours=1.0,
            ahi=0.0,
            **_DEVICE,
        )
        disabled.day_id = day.id
        disabled.enabled = False
        await async_db_session.flush()

        await DayManager.aggregate_day_statistics(day, async_db_session)

        assert day.session_count == 2
        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    async def test_no_device_values_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device={},
            second_device={},
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == day.ahi_computed == pytest.approx(5.0)
        assert day.hi == day.hi_computed == pytest.approx(1.25)

    async def test_disagreeing_device_values_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            second_device={**_DEVICE, "ahi_device": 3.0},
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    async def test_device_indices_disagree_on_oai_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            second_device={**_DEVICE, "oai_device": 1.5},
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)
        assert day.oai == pytest.approx(2.5)

    async def test_device_indices_disagree_on_hi_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            second_device={**_DEVICE, "hi_device": 0.0},
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.hi == pytest.approx(1.25)

    async def test_device_oai_missing_on_every_session_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        no_oai = {**_DEVICE}
        del no_oai["oai_device"]
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device=no_oai,
            second_device=no_oai,
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)
        assert day.oai == pytest.approx(2.5)

    async def test_enabled_session_without_statistics_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        device = async_test_device
        day, _, _ = await self._two_session_day(
            async_db_session, device, async_test_session_factory
        )
        bare = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 6, 7, 0, 0),
            duration_hours=1.0,
        )
        await DayManager.link_session_to_day(bare, device.id, async_db_session)

        assert day.session_count == 3
        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    async def test_trusted_device_values_without_therapy_hours_headline_device(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        # Zero mask-on time leaves no recount, but the device reported its
        # daily indices over a mask-on time within tolerance (3 min), so the
        # headline shows them.
        device = {**_DEVICE, "usage_hours": 0.0, "usage_hours_device": 0.05}
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device=device,
            second_device=device,
        )

        assert day.total_therapy_hours == 0.0
        assert day.index_source == IndexSource.DEVICE
        assert (day.ahi, day.oai, day.cai, day.hi) == (2.5, 1.0, 0.5, 1.0)
        for field in ("ahi", "oai", "cai", "hi"):
            assert getattr(day, f"{field}_computed") is None

    async def test_imported_hours_within_fraction_tolerance_headline_device(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        # 8.0 h imported vs 8.4 h reported: 24 min short, inside 5% (25.2 min).
        device = {**_DEVICE, "usage_hours_device": 8.4}
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device=device,
            second_device=device,
        )

        assert day.index_source == IndexSource.DEVICE
        assert day.ahi == 2.5

    async def test_imported_hours_short_of_device_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        # 8.0 h imported vs 9.0 h reported: 60 min short, beyond 5% (27 min).
        device = {**_DEVICE, "usage_hours_device": 9.0}
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device=device,
            second_device=device,
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    async def test_imported_hours_beyond_device_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        # 8.0 h imported vs 7.0 h reported: 60 min over, beyond 5% (21 min).
        device = {**_DEVICE, "usage_hours_device": 7.0}
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device=device,
            second_device=device,
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    @pytest.mark.parametrize(
        ("device_hours", "expected"),
        [
            # 4.5 min short: over 5% (3.2 min) but inside the 5 min floor.
            (1.075, IndexSource.DEVICE),
            # 6 min short: beyond the 5 min floor and 5% (3.3 min).
            (1.1, IndexSource.DERIVED),
        ],
    )
    async def test_short_day_uses_minimum_coverage_tolerance(
        self,
        async_db_session,
        async_test_device,
        async_test_session_factory,
        device_hours,
        expected,
    ):
        device = {**_DEVICE, "usage_hours": 0.5, "usage_hours_device": device_hours}
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device=device,
            second_device=device,
        )

        assert day.total_therapy_hours == pytest.approx(1.0)
        assert day.index_source == expected

    async def test_device_usage_hours_missing_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        no_usage = {**_DEVICE}
        del no_usage["usage_hours_device"]
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            first_device=no_usage,
            second_device=no_usage,
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    async def test_device_usage_hours_disagree_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, _, _ = await self._two_session_day(
            async_db_session,
            async_test_device,
            async_test_session_factory,
            second_device={**_DEVICE, "usage_hours_device": 8.05},
        )

        assert day.index_source == IndexSource.DERIVED
        assert day.ahi == pytest.approx(5.0)

    async def test_recount_without_ahi_headline_derived(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        device = async_test_device
        session = await async_test_session_factory(
            device_id=device.id,
            start_time=datetime(2024, 11, 5, 22, 0, 0),
            duration_hours=6.0,
            oai=2.0,
        )
        day = await DayManager.link_session_to_day(session, device.id, async_db_session)

        assert day.ahi_computed is None
        assert day.oai_computed == pytest.approx(2.0)
        assert day.index_source == IndexSource.DERIVED

    async def test_empty_day_clears_device_and_computed_indices(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        day, first, second = await self._two_session_day(
            async_db_session, async_test_device, async_test_session_factory
        )
        first.day_id = None
        second.day_id = None
        await async_db_session.flush()

        await DayManager.aggregate_day_statistics(day, async_db_session)

        for field in ("ahi", "oai", "cai", "hi"):
            assert getattr(day, field) is None
            assert getattr(day, f"{field}_computed") is None
        assert day.index_source is None
