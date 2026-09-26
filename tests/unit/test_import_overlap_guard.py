"""
Tests for the import-time overlap guard in SessionImporter._import_single_session.

Covers:
- New session fully covers one existing session → replacement
- New session fully covers two existing segment sessions → both replaced
- Partial overlap (neither covers the other) → incoming skipped
- Adjacent non-overlapping (a.end == b.start) → both kept
- Force re-import of same device_session_id does not false-trigger the guard
- Replaced row on a different day → extra_day_ids contains its day_id
"""

from __future__ import annotations

import logging

from datetime import datetime

import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database import models
from snore.database.day_manager import DayManager
from snore.database.importers import SessionImporter
from snore.parsers.unified import DeviceInfo, SessionStatistics, UnifiedSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SERIAL = "OVERLAP_TEST_001"


def _dt(day: int, hour: int, minute: int = 0) -> datetime:
    """Wall-clock datetime on 2026-01-{day} at HH:MM (naive, no tz)."""
    return datetime(2026, 1, day, hour, minute)


def _make_unified(
    serial_number: str,
    device_session_id: str,
    start: datetime,
    end: datetime,
    import_source: str = "resmed_edf",
) -> UnifiedSession:
    """Build a minimal UnifiedSession for import testing."""
    return UnifiedSession(
        device_info=DeviceInfo(
            manufacturer="TestCo",
            model="TestModel",
            serial_number=serial_number,
        ),
        device_session_id=device_session_id,
        start_time=start,
        end_time=end,
        import_source=import_source,
        statistics=SessionStatistics(),
    )


async def _seed_session(
    db: AsyncSession,
    device: models.Device,
    device_session_id: str,
    start: datetime,
    end: datetime,
    import_source: str = "resmed_edf",
) -> models.Session:
    """Insert a bare Session row linked to the correct Day."""
    day_date = DayManager.get_day_for_session(start)
    day = await DayManager.get_or_create_day(device.id, day_date, db)

    session = models.Session(
        device_id=device.id,
        device_session_id=device_session_id,
        start_time=start,
        end_time=end,
        duration_seconds=(end - start).total_seconds(),
        day_id=day.id,
        import_source=import_source,
    )
    db.add(session)
    await db.flush()
    return session


async def _count_sessions(db: AsyncSession, device_id: int) -> int:
    result = await db.execute(
        select(models.Session).where(models.Session.device_id == device_id)
    )
    return len(result.scalars().all())


async def _session_exists(db: AsyncSession, device_session_id: str) -> bool:
    result = await db.execute(
        select(models.Session).where(
            models.Session.device_session_id == device_session_id
        )
    )
    return result.scalars().first() is not None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def profile(async_db_session):
    import uuid

    from snore.database.models import Profile, User

    user = User(
        canonical_email=f"overlap_{uuid.uuid4().hex[:8]}@test.local",
        role="admin",
    )
    async_db_session.add(user)
    await async_db_session.flush()
    profile = Profile(user_id=user.id, name="Overlap Test Profile")
    async_db_session.add(profile)
    await async_db_session.flush()
    return profile


@pytest.fixture
async def device(async_db_session, profile):
    dev = models.Device(
        profile_id=profile.id,
        manufacturer="TestCo",
        model="TestModel",
        serial_number=_SERIAL,
    )
    async_db_session.add(dev)
    await async_db_session.flush()
    return dev


@pytest.fixture
def importer(profile):
    return SessionImporter(profile_id=profile.id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOverlapGuardReplacement:
    async def test_new_covers_existing_single_replaces_it(
        self, async_db_session, importer, device
    ):
        """Incoming session that fully envelops an existing one replaces it."""
        await _seed_session(
            async_db_session,
            device,
            "20260105_010000",
            _dt(5, 1),
            _dt(5, 7),
        )
        await async_db_session.flush()

        incoming = _make_unified(
            _SERIAL, "20260105_merged", _dt(5, 0, 30), _dt(5, 7, 30)
        )

        async with async_db_session.begin_nested():
            (
                was_imported,
                day_id,
                new_id,
                extra_day_ids,
                _deleted,
            ) = await importer._import_single_session(async_db_session, incoming)

        assert was_imported is True
        assert new_id is not None
        assert not await _session_exists(async_db_session, "20260105_010000")
        assert await _session_exists(async_db_session, "20260105_merged")
        assert await _count_sessions(async_db_session, device.id) == 1

    async def test_new_covers_two_segments_replaces_both(
        self, async_db_session, importer, device
    ):
        """Incoming merged session that covers two segments deletes both."""
        await _seed_session(
            async_db_session, device, "20260106_010000", _dt(6, 1), _dt(6, 4)
        )
        await _seed_session(
            async_db_session, device, "20260106_040000", _dt(6, 4), _dt(6, 8)
        )
        await async_db_session.flush()

        incoming = _make_unified(_SERIAL, "20260106_merged", _dt(6, 1), _dt(6, 8))

        async with async_db_session.begin_nested():
            (
                was_imported,
                day_id,
                new_id,
                extra_day_ids,
                _deleted,
            ) = await importer._import_single_session(async_db_session, incoming)

        assert was_imported is True
        assert not await _session_exists(async_db_session, "20260106_010000")
        assert not await _session_exists(async_db_session, "20260106_040000")
        assert await _session_exists(async_db_session, "20260106_merged")
        assert await _count_sessions(async_db_session, device.id) == 1

    async def test_partial_overlap_skips_incoming(
        self, async_db_session, importer, device, caplog
    ):
        """Partial overlap (neither fully contains the other) → incoming skipped."""
        await _seed_session(
            async_db_session, device, "20260107_010000", _dt(7, 1), _dt(7, 5)
        )
        await async_db_session.flush()

        # 03:00–08:00 overlaps 01:00–05:00, but neither covers the other
        incoming = _make_unified(_SERIAL, "20260107_030000", _dt(7, 3), _dt(7, 8))

        with caplog.at_level(logging.WARNING, logger="snore.database.importers"):
            async with async_db_session.begin_nested():
                (
                    was_imported,
                    day_id,
                    new_id,
                    extra_day_ids,
                    _deleted,
                ) = await importer._import_single_session(async_db_session, incoming)

        assert was_imported is False
        assert day_id is None
        assert new_id is None
        assert await _session_exists(async_db_session, "20260107_010000")
        assert not await _session_exists(async_db_session, "20260107_030000")
        assert "Overlap guard: skipping" in caplog.text

    async def test_adjacent_non_overlapping_keeps_both(
        self, async_db_session, importer, device
    ):
        """Adjacent sessions (a.end == b.start) are not overlapping — both kept."""
        await _seed_session(
            async_db_session, device, "20260108_010000", _dt(8, 1), _dt(8, 4)
        )
        await async_db_session.flush()

        # Exactly adjacent: new session starts exactly where existing ends
        incoming = _make_unified(_SERIAL, "20260108_040000", _dt(8, 4), _dt(8, 8))

        async with async_db_session.begin_nested():
            (
                was_imported,
                day_id,
                new_id,
                extra_day_ids,
                _deleted,
            ) = await importer._import_single_session(async_db_session, incoming)

        assert was_imported is True
        assert await _session_exists(async_db_session, "20260108_010000")
        assert await _session_exists(async_db_session, "20260108_040000")
        assert await _count_sessions(async_db_session, device.id) == 2

    async def test_force_reimport_same_id_does_not_trigger_guard(
        self, async_db_session, importer, device
    ):
        """Force-reimporting the same device_session_id deletes it before the guard
        runs, so the guard sees no overlapping rows."""
        await _seed_session(
            async_db_session, device, "20260109_010000", _dt(9, 1), _dt(9, 7)
        )
        await async_db_session.flush()

        incoming = _make_unified(_SERIAL, "20260109_010000", _dt(9, 1), _dt(9, 7))

        async with async_db_session.begin_nested():
            (
                was_imported,
                day_id,
                new_id,
                extra_day_ids,
                _deleted,
            ) = await importer._import_single_session(
                async_db_session, incoming, force=True
            )

        assert was_imported is True
        assert await _session_exists(async_db_session, "20260109_010000")
        assert await _count_sessions(async_db_session, device.id) == 1

    async def test_replaced_row_different_day_yields_extra_day_id(
        self, async_db_session, importer, device
    ):
        """A session on therapy day N replaced by a session that lands on therapy day
        N-1 must result in day N being re-aggregated, then deleted as an empty orphan.

        The noon split: start < 12:00 → previous calendar day; start >= 12:00 → same day.
        Existing starts at 13:00 on day 20 → therapy day 20.
        Incoming covering session starts at 11:00 on day 20 → therapy day 19 (before noon).
        After replacement day 20 is empty and must be cleaned up.
        """
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        from snore.database.importers import SessionImporter  # noqa: PLC0415
        from snore.database.models import Day  # noqa: PLC0415

        full_importer = SessionImporter(profile_id=importer.profile_id)

        # Import session A (therapy day 20) via the full batch path to get proper Day setup.
        session_a = _make_unified(_SERIAL, "20260120_130000", _dt(20, 13), _dt(20, 19))
        await full_importer.import_sessions_batch([session_a], db=async_db_session)

        days_before = (
            (
                await async_db_session.execute(
                    sa_select(Day).where(Day.device_id == device.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(days_before) == 1
        assert days_before[0].session_count == 1

        # Import session B: 11:00–21:00 → therapy day 19.  B fully covers A.
        session_b = _make_unified(_SERIAL, "20260120_merged", _dt(20, 11), _dt(20, 21))
        imported, skipped, failed, ids = await full_importer.import_sessions_batch(
            [session_b], db=async_db_session
        )

        assert imported == 1
        assert skipped == 0
        # Session A must be gone, session B present.
        assert not await _session_exists(async_db_session, "20260120_130000")
        assert await _session_exists(async_db_session, "20260120_merged")

        # Therapy day 20 (orphaned after A was deleted) must have been cleaned up.
        # Therapy day 19 (B's day) must remain with session_count == 1.
        days_after = (
            (
                await async_db_session.execute(
                    sa_select(Day).where(Day.device_id == device.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(days_after) == 1
        assert days_after[0].session_count == 1

    async def test_force_with_partial_overlap_skips_and_preserves_existing(
        self, async_db_session, importer, device, caplog
    ):
        """Critical safety: force-import A' with same ID as A but partial overlap
        with B must not import A' AND must leave A untouched.

        Before the fix, the force path deleted A before the overlap check ran;
        when the guard returned False (partial overlap with B), A was gone but
        A' was never inserted — silent data loss.
        """
        # A: 01:00–07:00
        await _seed_session(
            async_db_session, device, "20260112_010000", _dt(12, 1), _dt(12, 7)
        )
        # B: 07:30–09:00  (does NOT overlap A; adjacent gap at 07:00–07:30)
        await _seed_session(
            async_db_session, device, "20260112_073000", _dt(12, 7, 30), _dt(12, 9)
        )
        await async_db_session.flush()

        # A': same device_session_id as A, wider range 01:00–08:00.
        # Covers A entirely but only partially overlaps B (08:00 < 09:00).
        incoming = _make_unified(_SERIAL, "20260112_010000", _dt(12, 1), _dt(12, 8))

        with caplog.at_level(logging.WARNING, logger="snore.database.importers"):
            async with async_db_session.begin_nested():
                (
                    was_imported,
                    day_id,
                    new_id,
                    extra_day_ids,
                    _deleted,
                ) = await importer._import_single_session(
                    async_db_session, incoming, force=True
                )

        # A' must NOT be imported (partial overlap with B)
        assert was_imported is False
        assert day_id is None
        assert new_id is None
        # A must still exist — nothing was deleted
        assert await _session_exists(async_db_session, "20260112_010000")
        assert await _count_sessions(async_db_session, device.id) == 2
        assert "Overlap guard: skipping" in caplog.text

    async def test_replacement_deletes_orphan_day_row(
        self, async_db_session, importer, device
    ):
        """When a replaced session is the sole occupant of its Day, the Day row
        is deleted after re-aggregation so it doesn't appear in day listings."""
        from snore.database.importers import SessionImporter  # noqa: PLC0415

        # Seed a session on day 15 as the ONLY session for that day.
        await _seed_session(
            async_db_session, device, "20260115_010000", _dt(15, 1), _dt(15, 7)
        )
        await async_db_session.flush()

        # Wider session that fully covers the existing one and belongs to the same day.
        incoming = _make_unified(
            _SERIAL, "20260115_merged", _dt(15, 0, 30), _dt(15, 7, 30)
        )

        # import_sessions_batch owns day re-aggregation and orphan cleanup.
        full_importer = SessionImporter(profile_id=importer.profile_id)
        imported, skipped, failed, _ = await full_importer.import_sessions_batch(
            [incoming], db=async_db_session
        )

        assert imported == 1
        assert skipped == 0

        # The old session must be gone, the new one present.
        assert not await _session_exists(async_db_session, "20260115_010000")
        assert await _session_exists(async_db_session, "20260115_merged")

        # The Day row must still exist (the new session occupies it).
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        from snore.database.models import Day  # noqa: PLC0415

        days = (
            (
                await async_db_session.execute(
                    sa_select(Day).where(Day.device_id == device.id)
                )
            )
            .scalars()
            .all()
        )
        # Exactly one Day row — the original orphan was not left behind.
        assert len(days) == 1
        assert days[0].session_count == 1

    async def test_replacement_keeps_day_with_disabled_sibling(
        self, async_db_session, importer, device
    ):
        """A disabled session still references its Day, so replacing the day's
        only enabled session must not prune the row.

        The old ``session_count == 0`` check counted enabled sessions only, so
        the ORM delete would have tried to null the sibling's FK and aborted the
        import with an IntegrityError."""
        # Enabled session on therapy day 20, replaced below by one on day 19.
        await _seed_session(
            async_db_session, device, "20260120_130000", _dt(20, 13), _dt(20, 19)
        )
        # Disabled sibling on therapy day 20 that survives the replacement.
        disabled = await _seed_session(
            async_db_session, device, "20260120_210000", _dt(20, 21), _dt(20, 22)
        )
        disabled.enabled = False
        await async_db_session.flush()

        # 11:00-20:00 covers the 13:00-19:00 session and lands on therapy day 19.
        incoming = _make_unified(_SERIAL, "20260120_merged", _dt(20, 11), _dt(20, 20))
        full_importer = SessionImporter(profile_id=importer.profile_id)
        imported, skipped, failed, _ = await full_importer.import_sessions_batch(
            [incoming], db=async_db_session
        )
        assert (imported, skipped, failed) == (1, 0, 0)
        assert not await _session_exists(async_db_session, "20260120_130000")
        assert await _session_exists(async_db_session, "20260120_210000")

        days = (
            (
                await async_db_session.execute(
                    select(models.Day)
                    .where(models.Day.device_id == device.id)
                    .order_by(models.Day.date)
                )
            )
            .scalars()
            .all()
        )
        assert [(d.date.day, d.session_count) for d in days] == [(19, 1), (20, 0)]


class TestOldFormatPurge:
    """Old noon-bucket rows (e.g. ``20260130_merged``) are purged unconditionally.

    The new proximity-chained format (``20260130_225000_merged``) must NOT be
    treated as an old-format row — it flows through the normal all_covered check.
    """

    async def test_old_format_row_replaced_by_contained_new_format(
        self, async_db_session, importer, device
    ):
        """Stale noon-bucket row is deleted unconditionally; incoming new-format
        session that falls inside its time range is imported (previously skipped)."""
        # Wide stale row: 22:00 on day 29 through 07:00 on day 30.
        await _seed_session(
            async_db_session,
            device,
            "20260130_merged",
            _dt(29, 22),
            _dt(30, 7),
        )
        await async_db_session.flush()

        # New-format chain: 22:30–06:30 — fully contained within the old row.
        incoming = _make_unified(
            _SERIAL, "20260129_223000_merged", _dt(29, 22, 30), _dt(30, 6, 30)
        )

        async with async_db_session.begin_nested():
            (
                was_imported,
                day_id,
                new_id,
                extra_day_ids,
                deleted_ids,
            ) = await importer._import_single_session(async_db_session, incoming)

        assert was_imported is True
        assert new_id is not None
        assert not await _session_exists(async_db_session, "20260130_merged")
        assert await _session_exists(async_db_session, "20260129_223000_merged")
        assert await _count_sessions(async_db_session, device.id) == 1

    async def test_new_format_partial_overlap_still_skips(
        self, async_db_session, importer, device, caplog
    ):
        """New-format partial overlap still triggers a skip — the old-format purge
        path does not interfere when no old-format rows are present."""
        await _seed_session(
            async_db_session,
            device,
            "20260125_010000",
            _dt(25, 1),
            _dt(25, 5),
        )
        await async_db_session.flush()

        # 03:00–08:00 overlaps 01:00–05:00 but neither covers the other.
        incoming = _make_unified(
            _SERIAL, "20260125_030000_merged", _dt(25, 3), _dt(25, 8)
        )

        with caplog.at_level(logging.WARNING, logger="snore.database.importers"):
            async with async_db_session.begin_nested():
                was_imported, *_ = await importer._import_single_session(
                    async_db_session, incoming
                )

        assert was_imported is False
        assert await _session_exists(async_db_session, "20260125_010000")
        assert not await _session_exists(async_db_session, "20260125_030000_merged")
        assert "Overlap guard: skipping" in caplog.text

    async def test_mixed_old_format_and_covered_new_format_both_deleted(
        self, async_db_session, importer, device
    ):
        """Mixed overlap: old-format row deleted unconditionally; new-format row
        that is fully covered by the incoming session is also deleted.  Import
        proceeds."""
        # Stale wide row: 22:00 day 19 to 09:00 day 20.
        await _seed_session(
            async_db_session,
            device,
            "20260120_merged",
            _dt(19, 22),
            _dt(20, 9),
        )
        # New-format segment: 23:30–04:00 (incoming will fully cover it).
        await _seed_session(
            async_db_session,
            device,
            "20260119_233000",
            _dt(19, 23, 30),
            _dt(20, 4),
        )
        await async_db_session.flush()

        # Incoming: 22:30–07:00 — covers both the old-format row and the new-format row.
        incoming = _make_unified(
            _SERIAL, "20260119_223000_merged", _dt(19, 22, 30), _dt(20, 7)
        )

        async with async_db_session.begin_nested():
            (
                was_imported,
                day_id,
                new_id,
                extra_day_ids,
                deleted_ids,
            ) = await importer._import_single_session(async_db_session, incoming)

        assert was_imported is True
        assert new_id is not None
        assert not await _session_exists(async_db_session, "20260120_merged")
        assert not await _session_exists(async_db_session, "20260119_233000")
        assert await _session_exists(async_db_session, "20260119_223000_merged")
        assert await _count_sessions(async_db_session, device.id) == 1

    async def test_mixed_old_format_deleted_new_format_partial_overlap_skips(
        self, async_db_session, importer, device, caplog
    ):
        """Mixed overlap: old-format row is deleted unconditionally, but the
        remaining new-format row causes a skip because the incoming session only
        partially covers it."""
        # Stale wide row: 22:00 day 21 to 09:00 day 22.
        await _seed_session(
            async_db_session,
            device,
            "20260122_merged",
            _dt(21, 22),
            _dt(22, 9),
        )
        # New-format segment that incoming will only partially cover: 05:00–10:00.
        await _seed_session(
            async_db_session,
            device,
            "20260122_050000",
            _dt(22, 5),
            _dt(22, 10),
        )
        await async_db_session.flush()

        # Incoming: 22:30–07:00 — covers old-format entirely, but only partial
        # overlap with the new-format segment (07:00 < 10:00).
        incoming = _make_unified(
            _SERIAL, "20260121_223000_merged", _dt(21, 22, 30), _dt(22, 7)
        )

        with caplog.at_level(logging.WARNING, logger="snore.database.importers"):
            async with async_db_session.begin_nested():
                (
                    was_imported,
                    day_id,
                    new_id,
                    extra_day_ids,
                    deleted_ids,
                ) = await importer._import_single_session(async_db_session, incoming)

        assert was_imported is False
        # Old-format row was unconditionally purged.
        assert not await _session_exists(async_db_session, "20260122_merged")
        # New-format row is untouched — skip path does not delete it.
        assert await _session_exists(async_db_session, "20260122_050000")
        # Incoming was not inserted.
        assert not await _session_exists(async_db_session, "20260121_223000_merged")
        assert "Overlap guard: skipping" in caplog.text


class TestBatchSessionIdPruning:
    async def test_intra_batch_replacement_prunes_stale_id(
        self, async_db_session, importer, device
    ):
        """If session B in a batch fully covers and replaces session A (also in the
        same batch), the returned session_ids must contain only B's ID — not A's
        now-deleted ID, which would cause downstream consumers to hit a missing row.
        """
        # A: 01:00–07:00 on day 17
        session_a = _make_unified(_SERIAL, "20260117_010000", _dt(17, 1), _dt(17, 7))
        # B: 00:30–07:30 on day 17 — fully covers A (new.start ≤ A.start, new.end ≥ A.end)
        session_b = _make_unified(
            _SERIAL, "20260117_merged", _dt(17, 0, 30), _dt(17, 7, 30)
        )

        from snore.database.importers import SessionImporter  # noqa: PLC0415

        full_importer = SessionImporter(profile_id=importer.profile_id)
        (
            imported,
            skipped,
            failed,
            session_ids,
        ) = await full_importer.import_sessions_batch(
            [session_a, session_b], db=async_db_session
        )

        assert imported == 2  # both "imported" (A counted when it first imported)
        assert skipped == 0
        assert failed == 0
        # A's ID must be pruned; only B's ID survives
        assert len(session_ids) == 1
        # The surviving session must be B (the merger)
        assert await _session_exists(async_db_session, "20260117_merged")
        assert not await _session_exists(async_db_session, "20260117_010000")
        # Verify the returned ID matches the surviving row
        from sqlalchemy import select as sa_select  # noqa: PLC0415

        from snore.database import models as m  # noqa: PLC0415

        survivor = (
            (
                await async_db_session.execute(
                    sa_select(m.Session).where(m.Session.device_id == device.id)
                )
            )
            .scalars()
            .first()
        )
        assert survivor is not None
        assert session_ids[0] == survivor.id
