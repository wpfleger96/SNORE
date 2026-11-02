"""Tests for mask-on segment persistence (validity flags v1).

Covers:
1. ResMed night merge records one [start, end) interval per EDF segment in
   merged-session offset seconds.
2. A single-segment night stores [(0.0, duration)] — "known, no gaps" stays
   distinguishable from None = unknown.
3. Importer round-trip: UnifiedSession.mask_on_segments lands in the
   sessions.mask_on_segments JSON column (tuples serialized to lists).
4. Sessions without segment info (e.g. OSCAR imports) persist NULL.
5. Analysis-side validation: parse_mask_on_segments rejects malformed or
   mis-ordered stored JSON with a warning + None instead of crashing.
"""

from __future__ import annotations

import uuid

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from snore.database import models
from snore.database.importers import SessionImporter
from snore.database.models import Base
from snore.parsers.resmed_edf import ResmedEDFParser
from snore.parsers.unified import DeviceInfo, UnifiedSession
from snore.therapy_hours import parse_mask_on_segments

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_info() -> DeviceInfo:
    return DeviceInfo(
        manufacturer="ResMed",
        model="AirSense 10",
        serial_number=f"SN_{uuid.uuid4().hex[:8]}",
    )


def _segment_session(start: datetime, duration_s: float) -> UnifiedSession:
    """Minimal per-segment UnifiedSession as _parse_session_group would build."""
    return UnifiedSession(
        device_info=_device_info(),
        device_session_id=start.strftime("%Y%m%d_%H%M%S"),
        start_time=start,
        end_time=start + timedelta(seconds=duration_s),
    )


def _parse_night(
    segment_sessions: list[UnifiedSession], monkeypatch: pytest.MonkeyPatch
) -> UnifiedSession:
    """Drive _parse_night_session with synthetic pre-parsed segment sessions."""
    parser = ResmedEDFParser()
    queue = list(segment_sessions)
    monkeypatch.setattr(
        parser,
        "_parse_session_group",
        lambda *args, **kwargs: queue.pop(0),
    )
    segments = {
        s.device_session_id: {"BRP": Path(f"/nonexistent/{s.device_session_id}.edf")}
        for s in segment_sessions
    }
    night = parser._parse_night_session(
        night_date="20250910",
        chain_id=min(segments),
        segments=segments,
        device_info=_device_info(),
        base_path=Path("/nonexistent"),
    )
    assert night is not None
    return night


# ---------------------------------------------------------------------------
# Parser: segment recording
# ---------------------------------------------------------------------------


class TestParserMaskOnSegments:
    def test_multi_segment_merge_records_one_interval_per_segment(self, monkeypatch):
        """Each EDF segment yields one [start, end) interval at its merged offset."""
        base = datetime(2025, 9, 10, 22, 0, 0)
        night = _parse_night(
            [
                _segment_session(base, 3600.0),  # 22:00–23:00
                # 10-min mask-off gap, then 23:10–01:10 (7200 s)
                _segment_session(base + timedelta(seconds=4200), 7200.0),
            ],
            monkeypatch,
        )

        assert night.mask_on_segments == [(0.0, 3600.0), (4200.0, 11400.0)]
        # Prose data-quality notes stay unchanged.
        assert any("2 segment(s)" in note for note in night.data_quality_notes)

    def test_three_segments_record_three_ascending_intervals(self, monkeypatch):
        """Intervals stay ascending and per-segment across multiple gaps."""
        base = datetime(2025, 9, 10, 22, 0, 0)
        night = _parse_night(
            [
                _segment_session(base, 1800.0),
                _segment_session(base + timedelta(seconds=2400), 1800.0),
                _segment_session(base + timedelta(seconds=4800), 1800.0),
            ],
            monkeypatch,
        )

        assert night.mask_on_segments == [
            (0.0, 1800.0),
            (2400.0, 4200.0),
            (4800.0, 6600.0),
        ]

    def test_single_segment_records_full_duration_interval(self, monkeypatch):
        """A single-segment night stores [(0.0, duration)], not None."""
        base = datetime(2025, 9, 10, 22, 0, 0)
        night = _parse_night([_segment_session(base, 3600.0)], monkeypatch)

        assert night.mask_on_segments == [(0.0, 3600.0)]

    def test_unified_session_defaults_to_unknown(self):
        """Sessions built without segment info (e.g. OSCAR) default to None."""
        session = _segment_session(datetime(2025, 9, 10, 22, 0, 0), 3600.0)
        assert session.mask_on_segments is None


# ---------------------------------------------------------------------------
# Overlapping segment coalescing (ResMed straddle fix)
# ---------------------------------------------------------------------------


class TestOverlappingSegmentCoalescing:
    """ResMed sometimes writes one mask-on period's EDF files across two adjacent-second
    timestamps (e.g. PLD at :59, BRP/SA2 at :00). The parser sees two full-length
    segments 1 s apart; strictly-overlapping intervals must be coalesced so the stored
    list is disjoint. Touching intervals (next start == prev end) stay separate.
    """

    def test_straddled_file_type_groups_coalesce_to_one_interval(self, monkeypatch):
        """PLD at :59 + BRP at :00 — two 36420 s segments 1 s apart merge to one.

        Mirrors the real 2025-08-11 night: union span is 36421 s, stored as a
        single interval starting at offset 0.
        """
        base = datetime(2025, 8, 11, 23, 52, 59)
        night = _parse_night(
            [
                _segment_session(base, 36420.0),  # PLD group at :59
                _segment_session(base + timedelta(seconds=1), 36420.0),  # BRP at :00
            ],
            monkeypatch,
        )

        assert night.mask_on_segments == [(0.0, 36421.0)]
        assert any("Coalesced" in note for note in night.data_quality_notes)

    def test_straddle_pair_plus_later_real_segment_keeps_real_gap(self, monkeypatch):
        """Straddle pair coalesces; a later real segment (big gap) stays separate.

        Mirrors the real 2025-05-06 night: two overlapping segments followed by
        a second genuine mask-on period after a 21 000 s gap.
        """
        base = datetime(2025, 5, 6, 22, 0, 0)
        night = _parse_night(
            [
                _segment_session(base, 21240.0),
                _segment_session(base + timedelta(seconds=1), 21240.0),
                _segment_session(base + timedelta(seconds=21876), 11700.0),
            ],
            monkeypatch,
        )

        assert night.mask_on_segments == [(0.0, 21241.0), (21876.0, 33576.0)]

    def test_fully_contained_segment_is_absorbed(self, monkeypatch):
        """A short segment that starts and ends inside a longer one is absorbed.

        Merged end is the max of both ends, not the last segment's end.
        """
        base = datetime(2025, 9, 10, 22, 0, 0)
        night = _parse_night(
            [
                _segment_session(base, 7200.0),  # 22:00–00:00
                _segment_session(
                    base + timedelta(seconds=3600), 1800.0
                ),  # 23:00–23:30 (contained)
            ],
            monkeypatch,
        )

        assert night.mask_on_segments == [(0.0, 7200.0)]

    def test_touching_intervals_stay_separate(self, monkeypatch):
        """next start == prev end is touching, not overlapping — must NOT coalesce."""
        base = datetime(2025, 9, 10, 22, 0, 0)
        night = _parse_night(
            [
                _segment_session(base, 3600.0),
                _segment_session(base + timedelta(seconds=3600), 3600.0),
            ],
            monkeypatch,
        )

        assert night.mask_on_segments == [(0.0, 3600.0), (3600.0, 7200.0)]
        assert not any("Coalesced" in note for note in night.data_quality_notes)

    def test_straddled_night_usage_hours_equals_union(self, monkeypatch):
        """After finalize_statistics(), usage_hours reflects the coalesced union span."""
        base = datetime(2025, 8, 11, 23, 52, 59)
        night = _parse_night(
            [
                _segment_session(base, 36420.0),
                _segment_session(base + timedelta(seconds=1), 36420.0),
            ],
            monkeypatch,
        )

        night.finalize_statistics()

        assert night.statistics.usage_hours == pytest.approx(36421.0 / 3600)


# ---------------------------------------------------------------------------
# Analysis-side validation of the stored JSON (trust boundary)
# ---------------------------------------------------------------------------


class TestParseMaskOnSegments:
    """parse_mask_on_segments must degrade to None on any malformed value."""

    def test_none_stays_none(self):
        assert parse_mask_on_segments(None, session_id=1) is None

    def test_valid_segments_return_tuples(self):
        result = parse_mask_on_segments([[0.0, 3600.0], [4200.0, 7800.0]], session_id=1)
        assert result == [(0.0, 3600.0), (4200.0, 7800.0)]
        assert all(isinstance(seg, tuple) for seg in result)

    def test_adjacent_segments_are_valid(self):
        """next start == previous end (touching intervals) is allowed."""
        result = parse_mask_on_segments([[0.0, 100.0], [100.0, 200.0]], session_id=1)
        assert result == [(0.0, 100.0), (100.0, 200.0)]

    def test_integer_values_are_coerced_to_float(self):
        assert parse_mask_on_segments([[0, 3600]], session_id=1) == [(0.0, 3600.0)]

    def test_wrong_length_item_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_mask_on_segments([[0.0, 1.0, 2.0]], session_id=1) is None
            assert parse_mask_on_segments([[0.0]], session_id=1) is None
        assert "mask_on_segments" in caplog.text

    def test_non_numeric_item_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_mask_on_segments([[0.0, "3600"]], session_id=1) is None
            assert parse_mask_on_segments([[None, 3600.0]], session_id=1) is None
            assert parse_mask_on_segments([[True, 3600.0]], session_id=1) is None
        assert "mask_on_segments" in caplog.text

    def test_non_finite_item_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_mask_on_segments([[0.0, float("inf")]], session_id=1) is None
            assert parse_mask_on_segments([[float("-inf"), 1.0]], session_id=1) is None
            assert parse_mask_on_segments([[float("nan"), 1.0]], session_id=1) is None
        assert "mask_on_segments" in caplog.text

    def test_non_sequence_item_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_mask_on_segments([42], session_id=1) is None
            assert parse_mask_on_segments(["0.0,3600.0"], session_id=1) is None

    def test_non_list_value_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert parse_mask_on_segments({"start": 0.0}, session_id=1) is None
            assert parse_mask_on_segments("[[0.0, 3600.0]]", session_id=1) is None

    def test_out_of_order_segments_return_none(self, caplog):
        with caplog.at_level("WARNING"):
            assert (
                parse_mask_on_segments([[4200.0, 7800.0], [0.0, 3600.0]], session_id=1)
                is None
            )
        assert "out of order" in caplog.text

    def test_overlapping_segments_return_none(self):
        assert (
            parse_mask_on_segments([[0.0, 3600.0], [3599.0, 7800.0]], session_id=1)
            is None
        )

    def test_inverted_interval_returns_none(self):
        assert parse_mask_on_segments([[3600.0, 0.0]], session_id=1) is None

    def test_empty_interval_returns_none(self):
        assert parse_mask_on_segments([[100.0, 100.0]], session_id=1) is None


# ---------------------------------------------------------------------------
# Importer round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
async def mem_db():
    """In-memory async SQLite session with schema created and a single profile."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, class_=AsyncSession
    )
    session = factory()

    user = models.User(
        canonical_email=f"user_{uuid.uuid4().hex[:8]}@example.com",
        role="admin",
    )
    session.add(user)
    await session.flush()
    profile = models.Profile(user_id=user.id, name="Default")
    session.add(profile)
    await session.flush()
    await session.commit()

    yield session, profile.id

    await session.close()
    await engine.dispose()


class TestImporterMaskOnSegments:
    async def test_segments_round_trip_to_json_column(self, mem_db):
        """UnifiedSession.mask_on_segments lands in the DB as list-of-lists."""
        db, profile_id = mem_db
        unified = UnifiedSession(
            device_info=_device_info(),
            device_session_id=f"sess_{uuid.uuid4().hex[:8]}",
            start_time=datetime(2025, 9, 10, 22, 0, 0, tzinfo=UTC),
            end_time=datetime(2025, 9, 11, 1, 10, 0, tzinfo=UTC),
            mask_on_segments=[(0.0, 3600.0), (4200.0, 11400.0)],
        )

        importer = SessionImporter(profile_id=profile_id)
        imported, _, _, session_ids = await importer.import_sessions_batch(
            [unified], db=db
        )
        await db.commit()
        assert imported == 1

        row = (
            await db.execute(
                select(models.Session).where(models.Session.id == session_ids[0])
            )
        ).scalar_one()
        assert row.mask_on_segments == [[0.0, 3600.0], [4200.0, 11400.0]]

    async def test_unknown_segments_persist_null(self, mem_db):
        """mask_on_segments=None (unknown) stays NULL in the DB."""
        db, profile_id = mem_db
        unified = UnifiedSession(
            device_info=_device_info(),
            device_session_id=f"sess_{uuid.uuid4().hex[:8]}",
            start_time=datetime(2025, 9, 10, 22, 0, 0, tzinfo=UTC),
            end_time=datetime(2025, 9, 11, 6, 0, 0, tzinfo=UTC),
        )

        importer = SessionImporter(profile_id=profile_id)
        imported, _, _, session_ids = await importer.import_sessions_batch(
            [unified], db=db
        )
        await db.commit()
        assert imported == 1

        row = (
            await db.execute(
                select(models.Session).where(models.Session.id == session_ids[0])
            )
        ).scalar_one()
        assert row.mask_on_segments is None
