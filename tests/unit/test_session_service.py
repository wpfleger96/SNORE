"""Unit tests for SessionService."""

from datetime import date, datetime, timedelta

import pytest

from snore.database.models import Day, Session
from snore.services.session_service import SessionService


class TestSessionServiceList:
    """Tests for SessionService.list_sessions()."""

    async def test_list_sessions_empty(self, async_db_session):
        """Empty database returns empty list."""
        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions()

        assert len(result.sessions) == 0
        assert result.total_count == 0
        assert result.limit == 20

    async def test_list_sessions_with_data(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Returns correct session data."""
        now = datetime.now()
        await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0, ahi=2.5, usage_hours=8.0
        )
        await async_test_session_factory(
            async_test_device.id,
            now + timedelta(days=1),
            duration_hours=7.0,
            ahi=3.2,
            usage_hours=7.0,
        )

        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions()

        assert len(result.sessions) == 2
        assert result.total_count == 2
        assert result.sessions[0].ahi == 3.2
        assert result.sessions[1].ahi == 2.5

    async def test_list_sessions_filter_by_device(
        self,
        async_db_session,
        async_test_device,
        async_test_profile,
        async_test_session_factory,
    ):
        """Device filter works correctly."""
        now = datetime.now()
        await async_test_session_factory(async_test_device.id, now, duration_hours=8.0)

        from snore.database.models import Device

        other_device = Device(
            profile_id=async_test_profile.id,
            manufacturer="Other",
            model="Model",
            serial_number="OTHER123",
        )
        async_db_session.add(other_device)
        await async_db_session.flush()
        await async_test_session_factory(
            other_device.id, now + timedelta(days=1), duration_hours=7.0
        )

        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions(device=async_test_device.serial_number)

        assert len(result.sessions) == 1
        assert result.sessions[0].serial_number == async_test_device.serial_number

    async def test_list_sessions_date_range(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """From/to date filtering works."""
        base = datetime(2025, 1, 1, 12, 0, 0)
        await async_test_session_factory(async_test_device.id, base, duration_hours=8.0)
        await async_test_session_factory(
            async_test_device.id, base + timedelta(days=5), duration_hours=8.0
        )
        await async_test_session_factory(
            async_test_device.id, base + timedelta(days=10), duration_hours=8.0
        )

        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions(
            from_date=base + timedelta(days=2), to_date=base + timedelta(days=8)
        )

        assert len(result.sessions) == 1
        assert result.sessions[0].start_time.date() == (base + timedelta(days=5)).date()

    async def test_list_sessions_excludes_disabled(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Default excludes disabled sessions."""
        now = datetime.now()
        s1 = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )
        s2 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=1), duration_hours=8.0
        )
        s2.enabled = False
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions()

        assert len(result.sessions) == 1
        assert result.sessions[0].id == s1.id

    async def test_list_sessions_includes_disabled(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """include_disabled=True shows disabled sessions."""
        now = datetime.now()
        await async_test_session_factory(async_test_device.id, now, duration_hours=8.0)
        s2 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=1), duration_hours=8.0
        )
        s2.enabled = False
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions(include_disabled=True)

        assert len(result.sessions) == 2

    async def test_list_sessions_sorting(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Sort order works correctly."""
        now = datetime.now()
        await async_test_session_factory(async_test_device.id, now, duration_hours=6.0)
        await async_test_session_factory(
            async_test_device.id, now + timedelta(days=1), duration_hours=8.0
        )
        await async_test_session_factory(
            async_test_device.id, now + timedelta(days=2), duration_hours=7.0
        )

        service = SessionService(async_db_session, profile_id=1)
        result_asc = await service.list_sessions(sort_by="date-asc")
        assert result_asc.sessions[0].start_time < result_asc.sessions[1].start_time

        result_duration = await service.list_sessions(sort_by="duration")
        assert result_duration.sessions[0].duration_hours == 8.0


class TestSessionServiceDetail:
    """Tests for SessionService.get_session_detail()."""

    async def test_get_session_detail_found(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Returns full session detail."""
        now = datetime.now()
        session = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0, ahi=2.5, usage_hours=8.0
        )

        service = SessionService(async_db_session, profile_id=1)
        detail = await service.get_session_detail(session.id)

        assert detail.id == session.id
        assert detail.device_manufacturer == async_test_device.manufacturer
        assert detail.duration_hours == 8.0
        assert detail.statistics is not None
        assert detail.statistics.ahi == 2.5

    async def test_get_session_detail_not_found(self, async_db_session):
        """Raises ValueError if session not found."""
        service = SessionService(async_db_session, profile_id=1)

        with pytest.raises(ValueError, match="Session 999 not found"):
            await service.get_session_detail(999)

    async def test_get_session_detail_with_settings(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Includes settings when requested."""
        now = datetime.now()
        session = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )

        from snore.database.models import Setting

        setting = Setting(session_id=session.id, key="test_key", value="test_value")
        async_db_session.add(setting)
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        detail = await service.get_session_detail(session.id, include_settings=True)

        assert detail.settings is not None
        assert len(detail.settings) == 1
        assert detail.settings[0].key == "test_key"


class TestSessionDetailActiveMask:
    """Tests that get_session_detail populates active_mask correctly."""

    async def test_active_mask_populated_when_entry_exists(
        self,
        async_db_session,
        async_test_device,
        async_test_profile,
        async_test_session_factory,
    ):
        """active_mask is set when a mask entry has start_date <= session date."""
        from datetime import date  # noqa: PLC0415

        from snore.database.models import MaskLogEntry  # noqa: PLC0415

        session_start = datetime(2025, 6, 15, 22, 0)
        session = await async_test_session_factory(
            async_test_device.id, session_start, duration_hours=8.0
        )

        entry = MaskLogEntry(
            profile_id=async_test_profile.id,
            brand="ResMed",
            model="AirFit P10",
            style="pillows",
            start_date=date(2025, 6, 1),
        )
        async_db_session.add(entry)
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=async_test_profile.id)
        detail = await service.get_session_detail(session.id)

        assert detail.active_mask is not None
        assert detail.active_mask.brand == "ResMed"
        assert detail.active_mask.model == "AirFit P10"

    async def test_active_mask_none_when_no_entries(
        self,
        async_db_session,
        async_test_device,
        async_test_profile,
        async_test_session_factory,
    ):
        """active_mask is None when there are no mask log entries."""
        session_start = datetime(2025, 6, 15, 22, 0)
        session = await async_test_session_factory(
            async_test_device.id, session_start, duration_hours=8.0
        )

        service = SessionService(async_db_session, profile_id=async_test_profile.id)
        detail = await service.get_session_detail(session.id)

        assert detail.active_mask is None


class TestSessionServiceDelete:
    """Tests for SessionService delete operations."""

    async def test_delete_preview(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Returns correct preview counts."""
        now = datetime.now()
        s1 = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )
        s2 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=1), duration_hours=7.0
        )

        from snore.database.models import Event, Waveform

        event = Event(
            session_id=s1.id,
            event_type="OA",
            start_time=now,
            duration_seconds=12.0,
        )
        waveform = Waveform(
            session_id=s1.id,
            waveform_type="flow",
            sample_rate=25.0,
            sample_count=1000,
            data_blob=b"test",
        )
        async_db_session.add(event)
        async_db_session.add(waveform)
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(session_ids=[s1.id, s2.id])

        assert len(preview.sessions) == 2
        assert preview.event_count == 1
        assert preview.waveform_count == 1
        assert preview.stats_count == 0

    async def test_delete_preview_no_filters(self, async_db_session):
        """Raises ValueError when no filters specified."""
        service = SessionService(async_db_session, profile_id=1)

        with pytest.raises(ValueError, match="At least one filter must be specified"):
            await service.get_delete_preview()

    async def test_delete_sessions(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Actually deletes sessions."""
        now = datetime.now()
        s1 = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )
        s2 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=1), duration_hours=7.0
        )
        s1_id = s1.id
        s2_id = s2.id

        from sqlalchemy import func, select

        from snore.database.models import Event

        event = Event(
            session_id=s1_id,
            event_type="OA",
            start_time=now,
            duration_seconds=12.0,
        )
        async_db_session.add(event)
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        deleted = await service.delete_sessions([s1_id])

        assert deleted == 1

        remaining = (
            await async_db_session.execute(
                select(func.count()).where(Session.id == s2_id)
            )
        ).scalar()
        assert remaining == 1

        deleted_session = (
            await async_db_session.execute(
                select(func.count()).where(Session.id == s1_id)
            )
        ).scalar()
        assert deleted_session == 0

    async def test_delete_one_session_recomputes_day(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Deleting one of a day's sessions re-aggregates the day from the rest."""
        from snore.database.day_manager import DayManager

        base = datetime(2025, 3, 1, 22, 0, 0)
        s1 = await async_test_session_factory(
            async_test_device.id,
            base,
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=8.0,
            obstructive_apneas=16,
            pressure_mean=10.0,
        )
        s2 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(hours=5),
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=4.0,
            obstructive_apneas=8,
            pressure_mean=12.0,
        )

        await DayManager.link_session_to_day(s1, async_test_device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            s2, async_test_device.id, async_db_session
        )
        assert day.session_count == 2
        assert day.ahi == pytest.approx(6.0)
        assert day.obstructive_apneas == 24

        service = SessionService(async_db_session, profile_id=1)
        deleted = await service.delete_sessions([s1.id])
        assert deleted == 1

        await async_db_session.flush()
        await async_db_session.refresh(day)
        assert day.session_count == 1
        assert day.ahi == pytest.approx(4.0)
        assert day.obstructive_apneas == 8
        assert day.pressure_mean == pytest.approx(12.0)

    async def test_delete_all_sessions_prunes_day(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Deleting every session of a day removes the orphaned Day row."""
        from snore.database.day_manager import DayManager

        base = datetime(2025, 3, 1, 22, 0, 0)
        s1 = await async_test_session_factory(
            async_test_device.id,
            base,
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=8.0,
            obstructive_apneas=16,
            epap_mean=7.0,
            pressure_mean=10.0,
            leak_median=2.0,
            spo2_mean=94.0,
        )
        s2 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(hours=5),
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=4.0,
            obstructive_apneas=8,
        )

        await DayManager.link_session_to_day(s1, async_test_device.id, async_db_session)
        day = await DayManager.link_session_to_day(
            s2, async_test_device.id, async_db_session
        )
        assert day.session_count == 2
        assert day.epap_mean is not None

        day_id = day.id

        service = SessionService(async_db_session, profile_id=1)
        deleted = await service.delete_sessions([s1.id, s2.id])
        assert deleted == 2

        await async_db_session.flush()
        assert await async_db_session.get(Day, day_id) is None

    async def test_delete_last_enabled_session_keeps_day_with_disabled_sibling(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Deleting a day's only enabled session while a disabled sibling
        remains keeps the Day row at session_count 0 and the sibling intact."""
        from snore.database.day_manager import DayManager

        base = datetime(2025, 3, 1, 22, 0, 0)
        enabled = await async_test_session_factory(
            async_test_device.id, base, duration_hours=4.0, usage_hours=4.0, ahi=8.0
        )
        disabled = await async_test_session_factory(
            async_test_device.id, base + timedelta(hours=5), duration_hours=2.0
        )
        disabled.enabled = False
        await async_db_session.flush()

        await DayManager.link_session_to_day(
            enabled, async_test_device.id, async_db_session
        )
        day = await DayManager.link_session_to_day(
            disabled, async_test_device.id, async_db_session
        )
        assert day.session_count == 1
        day_id, disabled_id = day.id, disabled.id

        service = SessionService(async_db_session, profile_id=1)
        assert await service.delete_sessions([enabled.id]) == 1

        await async_db_session.flush()
        async_db_session.expire_all()
        stored = await async_db_session.get(Day, day_id)
        assert stored is not None
        assert stored.session_count == 0
        assert stored.ahi is None
        assert await async_db_session.get(Session, disabled_id) is not None

    async def test_delete_spanning_multiple_days_recomputes_each(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """One delete call spanning two days recomputes every affected day."""
        from snore.database.day_manager import DayManager

        base = datetime(2025, 3, 1, 22, 0, 0)
        # Day A: two sessions — deleting one leaves a recomputed partial day.
        a1 = await async_test_session_factory(
            async_test_device.id,
            base,
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=8.0,
        )
        a2 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(hours=5),
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=4.0,
        )
        # Day B: one session — deleting it orphans the day, which is pruned.
        b1 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(days=2),
            duration_hours=6.0,
            usage_hours=6.0,
            ahi=2.0,
        )

        await DayManager.link_session_to_day(a1, async_test_device.id, async_db_session)
        day_a = await DayManager.link_session_to_day(
            a2, async_test_device.id, async_db_session
        )
        day_b = await DayManager.link_session_to_day(
            b1, async_test_device.id, async_db_session
        )
        assert day_a.id != day_b.id
        assert day_a.session_count == 2
        assert day_b.session_count == 1
        day_b_id = day_b.id

        service = SessionService(async_db_session, profile_id=1)
        deleted = await service.delete_sessions([a1.id, b1.id])
        assert deleted == 2

        await async_db_session.flush()
        await async_db_session.refresh(day_a)
        assert day_a.session_count == 1
        assert day_a.ahi == pytest.approx(4.0)
        assert await async_db_session.get(Day, day_b_id) is None

    async def test_delete_across_chunk_boundary_recomputes_days(
        self,
        async_db_session,
        async_test_device,
        async_test_session_factory,
        monkeypatch,
    ):
        """Deletes every listed session and recomputes each affected day exactly
        once, even when a single day's sessions land in different ID chunks."""
        from sqlalchemy import func, select

        from snore.database.day_manager import DayManager

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = datetime(2025, 3, 1, 22, 0, 0)
        # Day A: four sessions.  The three deleted ones span two chunks; one survives.
        a1 = await async_test_session_factory(
            async_test_device.id, base, duration_hours=4.0, usage_hours=4.0, ahi=8.0
        )
        a2 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(hours=1),
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=6.0,
        )
        a3 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(hours=2),
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=4.0,
        )
        a_keep = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(hours=3),
            duration_hours=4.0,
            usage_hours=4.0,
            ahi=2.0,
        )
        # Day B and Day C: one session each; deleting it empties the day.
        b1 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(days=2),
            duration_hours=6.0,
            usage_hours=6.0,
            ahi=10.0,
        )
        c1 = await async_test_session_factory(
            async_test_device.id,
            base + timedelta(days=4),
            duration_hours=5.0,
            usage_hours=5.0,
            ahi=5.0,
        )

        day_a = None
        for s in (a1, a2, a3, a_keep):
            day_a = await DayManager.link_session_to_day(
                s, async_test_device.id, async_db_session
            )
        day_b = await DayManager.link_session_to_day(
            b1, async_test_device.id, async_db_session
        )
        day_c = await DayManager.link_session_to_day(
            c1, async_test_device.id, async_db_session
        )
        assert day_a.session_count == 4
        day_b_id, day_c_id = day_b.id, day_c.id

        service = SessionService(async_db_session, profile_id=1)
        deleted = await service.delete_sessions([a1.id, a2.id, a3.id, b1.id, c1.id])
        assert deleted == 5

        await async_db_session.flush()
        remaining = (
            await async_db_session.execute(
                select(func.count())
                .select_from(Session)
                .where(Session.id.in_([a1.id, a2.id, a3.id, b1.id, c1.id]))
            )
        ).scalar()
        assert remaining == 0
        survivor = (
            await async_db_session.execute(
                select(func.count()).select_from(Session).where(Session.id == a_keep.id)
            )
        ).scalar()
        assert survivor == 1

        await async_db_session.refresh(day_a)
        assert day_a.session_count == 1
        assert day_a.ahi == pytest.approx(2.0)
        assert await async_db_session.get(Day, day_b_id) is None
        assert await async_db_session.get(Day, day_c_id) is None

    async def test_delete_preview_sums_counts_and_sorts_across_chunks(
        self,
        async_db_session,
        async_test_device,
        async_test_session_factory,
        monkeypatch,
    ):
        """Preview counts are summed across chunks and sessions are re-sorted
        start_time DESC after the chunked concatenation."""
        from snore.database.models import Event, Waveform

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = datetime(2025, 5, 1, 22, 0, 0)
        # Start times deliberately out of id order so the global DESC ordering is
        # only correct after the post-concat re-sort.
        s1 = await async_test_session_factory(
            async_test_device.id, base + timedelta(days=0), ahi=1.0
        )
        s2 = await async_test_session_factory(
            async_test_device.id, base + timedelta(days=4), ahi=2.0
        )
        s3 = await async_test_session_factory(
            async_test_device.id, base + timedelta(days=1), ahi=3.0
        )
        s4 = await async_test_session_factory(
            async_test_device.id, base + timedelta(days=3), ahi=4.0
        )
        s5 = await async_test_session_factory(
            async_test_device.id, base + timedelta(days=2), ahi=5.0
        )

        # Events on s1, s3, s5 (span three chunks): 1 + 2 + 1 = 4.
        for sid, n in ((s1.id, 1), (s3.id, 2), (s5.id, 1)):
            for _ in range(n):
                async_db_session.add(
                    Event(
                        session_id=sid,
                        event_type="OA",
                        start_time=base,
                        duration_seconds=10.0,
                    )
                )
        # Waveforms on s2, s4 (span two chunks): 1 + 2 = 3.  Distinct types per
        # session — (session_id, waveform_type) is unique.
        for sid, wtype in ((s2.id, "flow"), (s4.id, "flow"), (s4.id, "pressure")):
            async_db_session.add(
                Waveform(
                    session_id=sid,
                    waveform_type=wtype,
                    sample_rate=25.0,
                    sample_count=100,
                    data_blob=b"x",
                )
            )
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(
            session_ids=[s1.id, s2.id, s3.id, s4.id, s5.id]
        )

        assert preview.event_count == 4
        assert preview.waveform_count == 3
        assert preview.stats_count == 5
        assert [s.id for s in preview.sessions] == [s2.id, s4.id, s5.id, s3.id, s1.id]
        times = [s.start_time for s in preview.sessions]
        assert times == sorted(times, reverse=True)

    async def test_get_owned_ids_across_chunks_returns_only_owned(
        self,
        async_db_session,
        async_test_device,
        async_test_session_factory,
        monkeypatch,
    ):
        """Ownership filtering unions correctly across chunks: interleaved owned
        and foreign IDs yield exactly the owned subset."""
        from snore.database.models import Device, Profile, User

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        now = datetime(2025, 7, 1, 22, 0, 0)
        o1 = await async_test_session_factory(async_test_device.id, now)
        o2 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=1)
        )
        o3 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=2)
        )

        user = User(canonical_email="foreign_a@example.com", role="admin")
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="Foreign")
        async_db_session.add(profile)
        await async_db_session.flush()
        foreign_device = Device(
            profile_id=profile.id,
            manufacturer="Other",
            model="Model",
            serial_number="FOREIGN_A",
        )
        async_db_session.add(foreign_device)
        await async_db_session.flush()
        f1 = await async_test_session_factory(foreign_device.id, now)
        f2 = await async_test_session_factory(
            foreign_device.id, now + timedelta(days=1)
        )
        f3 = await async_test_session_factory(
            foreign_device.id, now + timedelta(days=2)
        )

        service = SessionService(async_db_session, profile_id=1)
        owned = await service.get_owned_ids([o1.id, f1.id, o2.id, f2.id, o3.id, f3.id])

        assert owned == {o1.id, o2.id, o3.id}

    async def test_delete_preview_dedupes_duplicate_ids_across_chunks(
        self,
        async_db_session,
        async_test_device,
        async_test_session_factory,
        monkeypatch,
    ):
        """A duplicate id split across chunks must not double-count related rows
        or list the session twice.

        Without the entry-point dedup, [s1, s2, s1] chunks to [s1, s2] then [s1]
        at size 2, so s1's events/stats are counted twice and s1 appears twice in
        the session list.
        """
        from snore.database.models import Event

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        base = datetime(2025, 10, 1, 22, 0, 0)
        s1 = await async_test_session_factory(async_test_device.id, base, ahi=1.0)
        s2 = await async_test_session_factory(
            async_test_device.id, base + timedelta(days=1), ahi=2.0
        )
        # Two events on s1 — double-counting would report four.
        for _ in range(2):
            async_db_session.add(
                Event(
                    session_id=s1.id,
                    event_type="OA",
                    start_time=base,
                    duration_seconds=10.0,
                )
            )
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(session_ids=[s1.id, s2.id, s1.id])

        # s1 listed once; DESC by start_time puts the later s2 first.
        assert [s.id for s in preview.sessions] == [s2.id, s1.id]
        assert preview.event_count == 2  # s1's two events, counted once
        assert preview.stats_count == 2  # one Statistics row per distinct session

    async def test_delete_enforces_ownership_across_chunks(
        self,
        async_db_session,
        async_test_device,
        async_test_session_factory,
        monkeypatch,
    ):
        """The DELETE's ownership predicate holds per chunk: foreign IDs mixed
        into the list are never deleted."""
        from sqlalchemy import func, select

        from snore.database.models import Device, Profile, User

        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        now = datetime(2025, 9, 1, 22, 0, 0)
        o1 = await async_test_session_factory(async_test_device.id, now)
        o2 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=1)
        )
        o3 = await async_test_session_factory(
            async_test_device.id, now + timedelta(days=2)
        )

        user = User(canonical_email="foreign_b@example.com", role="admin")
        async_db_session.add(user)
        await async_db_session.flush()
        profile = Profile(user_id=user.id, name="Foreign")
        async_db_session.add(profile)
        await async_db_session.flush()
        foreign_device = Device(
            profile_id=profile.id,
            manufacturer="Other",
            model="Model",
            serial_number="FOREIGN_B",
        )
        async_db_session.add(foreign_device)
        await async_db_session.flush()
        f1 = await async_test_session_factory(foreign_device.id, now)
        f2 = await async_test_session_factory(
            foreign_device.id, now + timedelta(days=1)
        )

        service = SessionService(async_db_session, profile_id=1)
        deleted = await service.delete_sessions([o1.id, f1.id, o2.id, f2.id, o3.id])
        assert deleted == 3

        await async_db_session.flush()
        owned_remaining = (
            await async_db_session.execute(
                select(func.count())
                .select_from(Session)
                .where(Session.id.in_([o1.id, o2.id, o3.id]))
            )
        ).scalar()
        assert owned_remaining == 0
        foreign_remaining = (
            await async_db_session.execute(
                select(func.count())
                .select_from(Session)
                .where(Session.id.in_([f1.id, f2.id]))
            )
        ).scalar()
        assert foreign_remaining == 2


class TestSessionServiceEnable:
    """Tests for SessionService.set_session_enabled()."""

    async def test_set_session_enabled_toggle(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Toggles enabled flag."""
        now = datetime.now()
        session = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )
        assert session.enabled is True

        service = SessionService(async_db_session, profile_id=1)
        await service.set_session_enabled(session.id, False)

        await async_db_session.refresh(session)
        assert session.enabled is False

    async def test_disable_last_session_keeps_day_and_reenable_restores_stats(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Disabling a day's only session keeps its Day row (the disabled
        session still references it) with session_count reset to 0, and
        re-enabling it restores the aggregates."""
        from snore.database.day_manager import DayManager

        session = await async_test_session_factory(
            async_test_device.id,
            datetime(2025, 3, 1, 22, 0, 0),
            duration_hours=6.0,
            usage_hours=6.0,
            ahi=5.0,
        )
        day = await DayManager.link_session_to_day(
            session, async_test_device.id, async_db_session
        )
        assert day.session_count == 1

        service = SessionService(async_db_session, profile_id=1)
        await service.set_session_enabled(session.id, False)

        await async_db_session.flush()
        await async_db_session.refresh(day)
        assert day.session_count == 0
        assert day.total_therapy_hours == 0.0
        assert day.ahi is None

        await service.set_session_enabled(session.id, True)

        await async_db_session.flush()
        await async_db_session.refresh(day)
        assert day.session_count == 1
        assert day.ahi == pytest.approx(5.0)

    async def test_set_session_enabled_not_found(self, async_db_session):
        """Raises ValueError if session not found."""
        service = SessionService(async_db_session, profile_id=1)

        with pytest.raises(ValueError, match="Session 999 not found"):
            await service.set_session_enabled(999, False)

    async def test_set_session_enabled_idempotent(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Idempotent when already in desired state."""
        now = datetime.now()
        session = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )

        service = SessionService(async_db_session, profile_id=1)
        await service.set_session_enabled(session.id, True)

        await async_db_session.refresh(session)
        assert session.enabled is True


class TestSessionTherapyDay:
    """Tests that therapy_day (noon-cutoff) is returned correctly."""

    async def test_list_session_late_evening_stays_same_day(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Session starting at 23:57 has therapy_day == its calendar date."""
        start = datetime(2025, 8, 9, 23, 57, 0)
        await async_test_session_factory(
            async_test_device.id, start, duration_hours=7.5
        )

        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions()

        assert len(result.sessions) == 1
        assert result.sessions[0].therapy_day == date(2025, 8, 9)

    async def test_list_session_early_morning_rolls_back_one_day(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Session starting at 01:12 has therapy_day == previous calendar date (noon cutoff)."""
        start = datetime(2025, 8, 10, 1, 12, 0)
        await async_test_session_factory(
            async_test_device.id, start, duration_hours=7.5
        )

        service = SessionService(async_db_session, profile_id=1)
        result = await service.list_sessions()

        assert len(result.sessions) == 1
        assert result.sessions[0].therapy_day == date(2025, 8, 9)

    async def test_session_detail_therapy_day_early_morning(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """SessionDetail.therapy_day is previous day for sessions starting before noon."""
        start = datetime(2025, 8, 10, 0, 12, 0)
        session = await async_test_session_factory(
            async_test_device.id, start, duration_hours=8.0
        )

        service = SessionService(async_db_session, profile_id=1)
        detail = await service.get_session_detail(session.id)

        assert detail.therapy_day == date(2025, 8, 9)
        assert detail.start_time == start

    async def test_delete_preview_therapy_day_early_morning(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """DeletePreview sessions carry correct therapy_day for early-morning sessions."""
        start = datetime(2025, 8, 10, 2, 0, 0)
        session = await async_test_session_factory(
            async_test_device.id, start, duration_hours=7.0
        )

        service = SessionService(async_db_session, profile_id=1)
        preview = await service.get_delete_preview(session_ids=[session.id])

        assert len(preview.sessions) == 1
        assert preview.sessions[0].therapy_day == date(2025, 8, 9)


class TestSessionServiceResolve:
    """Tests for SessionService.resolve_session_id()."""

    async def test_resolve_session_id_by_id(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Resolving by explicit ID validates ownership and returns the same ID."""
        now = datetime(2025, 1, 15, 12, 0, 0)
        session = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )
        service = SessionService(
            async_db_session, profile_id=async_test_device.profile_id
        )
        resolved = await service.resolve_session_id(session_id=session.id, date=None)

        assert resolved == session.id

    async def test_resolve_session_id_by_date(
        self, async_db_session, async_test_device, async_test_session_factory
    ):
        """Resolves via Day join when date provided."""
        now = datetime(2025, 1, 15, 12, 0, 0)
        session = await async_test_session_factory(
            async_test_device.id, now, duration_hours=8.0
        )

        day = Day(
            device_id=async_test_device.id,
            date=now.date(),
            session_count=1,
            total_therapy_hours=8.0,
        )
        async_db_session.add(day)
        await async_db_session.flush()

        session.day_id = day.id
        await async_db_session.flush()

        service = SessionService(async_db_session, profile_id=1)
        resolved = await service.resolve_session_id(session_id=None, date=now)

        assert resolved == session.id

    async def test_resolve_session_id_not_found(self, async_db_session):
        """Raises ValueError when no session found for date."""
        service = SessionService(async_db_session, profile_id=1)

        with pytest.raises(ValueError, match="No session found for date"):
            await service.resolve_session_id(session_id=None, date=datetime(2025, 1, 1))

    async def test_resolve_session_id_no_params(self, async_db_session):
        """Raises ValueError when neither ID nor date provided."""
        service = SessionService(async_db_session, profile_id=1)

        with pytest.raises(
            ValueError, match="Either session_id or date must be provided"
        ):
            await service.resolve_session_id(session_id=None, date=None)
