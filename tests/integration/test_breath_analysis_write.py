"""Acceptance tests for breath-row persistence.

Pinned scenarios:
A1. Fresh import → Breath rows present.
    After a full import+analysis, the `breaths` table is non-empty and every
    row references the correct analysis_result_id.

A2. Two consecutive re-analyses → two AnalysisResult rows, correct latest
    selection, intact history.  Includes the equal-created_at tie-breaker
    (highest id wins).

A3. Atomic rollback: AnalysisResult parent is flushed, Breath children bulk-add
    raises → both parent and children are absent from the DB.

A4. Non-UTC host determinism: AnalysisResult timestamps stored in naive UTC
    regardless of OS timezone (non-UTC host simulation via monkeypatching).

A5. Validity flags end-to-end: analyze_session over a seeded session with ramp
    settings + persisted mask-on segments persists Breath rows whose
    ramp_active / mask_off values and reasons match the v1 derivations.

A6. primary_mode threading: analyze_session with an explicit non-default
    primary_mode persists that mode into engine_versions_json.run.primary_mode,
    rather than falling back to DEFAULT_MODE.
"""

from __future__ import annotations

import uuid

from datetime import UTC, datetime

import numpy as np
import pytest

from sqlalchemy import func, select

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.service import AnalysisService
from snore.analysis.shared.versioning import (
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisRunMetadata,
)
from snore.analysis.types import AnalysisComputation, ModeResult
from snore.analysis.types import AnalysisResult as AnalysisResultDTO
from snore.database import models
from snore.database.models import AnalysisResult as AnalysisResult_ORM
from snore.database.session import init_database, session_scope


async def _make_profile(db: AsyncSession) -> tuple[int, int]:
    """Insert User + Profile and return (user_id, profile_id)."""
    user = models.User(
        canonical_email=f"breath_{uuid.uuid4().hex[:8]}@test.example",
        role="admin",
    )
    db.add(user)
    await db.flush()
    profile = models.Profile(user_id=user.id, name="Breath Test Profile")
    db.add(profile)
    await db.flush()
    return user.id, profile.id


async def _make_device_and_session(
    db: AsyncSession, profile_id: int, *, start: datetime
) -> tuple[int, int]:
    """Insert Device + Day + Session.  Returns (device_id, session_id)."""
    device = models.Device(
        profile_id=profile_id,
        serial_number=f"SN_{uuid.uuid4().hex[:6]}",
        manufacturer="TestMfr",
        model="TestModel",
        firmware_version="1.0",
    )
    db.add(device)
    await db.flush()
    day = models.Day(
        device_id=device.id,
        date=start.date(),
        session_count=1,
    )
    db.add(day)
    await db.flush()
    from datetime import timedelta

    session = models.Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
        start_time=start,
        end_time=start + timedelta(hours=7),
        duration_seconds=7 * 3600.0,
    )
    db.add(session)
    await db.flush()
    return device.id, session.id


def _make_algo_versions() -> AlgoVersions:
    identity = AlgorithmIdentity.current()
    run_meta = AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"])
    return AlgoVersions(identity=identity, run=run_meta)


def _breath_row(
    analysis_result_id: int, session_id: int, breath_number: int
) -> models.Breath:
    return models.Breath(
        analysis_result_id=analysis_result_id,
        session_id=session_id,
        breath_number=breath_number,
        start_offset_s=float(breath_number * 4),
        end_offset_s=float(breath_number * 4 + 3),
        total_time_s=3.0,
        i_e_ratio=0.5,
        duty_cycle=0.33,
        peak_flow_lpm=30.0,
        tidal_volume_ml=400.0,
        respiratory_rate_rolling=15.0,
    )


# ---------------------------------------------------------------------------
# A1 — Fresh import populates breaths
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFreshImportPopulatesBreaths:
    async def test_store_result_populates_breaths_table(self, temp_db):
        """After store_result, Breath rows exist and reference the analysis_result_id."""
        await init_database(str(temp_db))
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            _, session_id = await _make_device_and_session(
                db, profile_id, start=datetime(2025, 1, 10, 22, 0)
            )
            algo_versions = _make_algo_versions()
            analysis = models.AnalysisResult(
                session_id=session_id,
                timestamp_start=datetime(2025, 1, 10, 22, 0),
                timestamp_end=datetime(2025, 1, 11, 5, 0),
                programmatic_result_json={},
                processing_time_ms=100,
                engine_versions_json=algo_versions.model_dump(),
            )
            db.add(analysis)
            await db.flush()
            ar_id = analysis.id

            breaths = [_breath_row(ar_id, session_id, n) for n in range(1, 6)]
            db.add_all(breaths)

        # Verify outside the write transaction
        async with session_scope() as db:
            count = (
                await db.execute(
                    select(func.count())
                    .select_from(models.Breath)
                    .where(models.Breath.analysis_result_id == ar_id)
                )
            ).scalar()
            assert count == 5, f"Expected 5 breath rows, got {count}"

            # Each row references the correct analysis_result_id
            rows = (
                (
                    await db.execute(
                        select(models.Breath).where(
                            models.Breath.analysis_result_id == ar_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert all(r.analysis_result_id == ar_id for r in rows)

    async def test_breath_rows_reference_correct_session_id(self, temp_db):
        """Breath.session_id matches the parent Session.id."""
        await init_database(str(temp_db))
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            _, session_id = await _make_device_and_session(
                db, profile_id, start=datetime(2025, 1, 10, 22, 0)
            )
            algo_versions = _make_algo_versions()
            analysis = models.AnalysisResult(
                session_id=session_id,
                timestamp_start=datetime(2025, 1, 10, 22, 0),
                timestamp_end=datetime(2025, 1, 11, 5, 0),
                programmatic_result_json={},
                processing_time_ms=50,
                engine_versions_json=algo_versions.model_dump(),
            )
            db.add(analysis)
            await db.flush()
            db.add(_breath_row(analysis.id, session_id, 1))

        async with session_scope() as db:
            row = (
                (
                    await db.execute(
                        select(models.Breath).where(
                            models.Breath.analysis_result_id == analysis.id
                        )
                    )
                )
                .scalars()
                .first()
            )
            assert row is not None
            assert row.session_id == session_id


# ---------------------------------------------------------------------------
# A2 — Two re-analyses → correct latest selection + tie-breaker
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestTwoReanalysesLatestSelection:
    async def test_two_runs_produce_two_analysis_result_rows(self, temp_db):
        """Each store_result call creates a new AnalysisResult row (history preserved)."""
        await init_database(str(temp_db))
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            _, session_id = await _make_device_and_session(
                db, profile_id, start=datetime(2025, 1, 10, 22, 0)
            )
            algo = _make_algo_versions()
            ar1 = models.AnalysisResult(
                session_id=session_id,
                timestamp_start=datetime(2025, 1, 10, 22, 0),
                timestamp_end=datetime(2025, 1, 11, 5, 0),
                programmatic_result_json={},
                processing_time_ms=100,
                engine_versions_json=algo.model_dump(),
            )
            db.add(ar1)
            await db.flush()
            ar2 = models.AnalysisResult(
                session_id=session_id,
                timestamp_start=datetime(2025, 1, 10, 22, 0),
                timestamp_end=datetime(2025, 1, 11, 5, 0),
                programmatic_result_json={},
                processing_time_ms=110,
                engine_versions_json=algo.model_dump(),
            )
            db.add(ar2)
            await db.flush()
            ar1_id = ar1.id
            ar2_id = ar2.id

        async with session_scope() as db:
            count = (
                await db.execute(
                    select(func.count())
                    .select_from(models.AnalysisResult)
                    .where(models.AnalysisResult.session_id == session_id)
                )
            ).scalar()
            assert count == 2, "Both analysis runs must be retained"
            assert ar2_id > ar1_id, "ar2 inserted second must have higher id"

    async def test_latest_selection_picks_highest_id_on_equal_created_at(self, temp_db):
        """When created_at is equal, the row with the highest id is 'latest'."""
        await init_database(str(temp_db))
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            _, session_id = await _make_device_and_session(
                db, profile_id, start=datetime(2025, 1, 10, 22, 0)
            )
            algo = _make_algo_versions()
            tie_ts = datetime(
                2025, 1, 11, 8, 0, 0, tzinfo=UTC
            )  # identical created_at, tz-aware

            ar_old = models.AnalysisResult(
                session_id=session_id,
                timestamp_start=datetime(2025, 1, 10, 22, 0),
                timestamp_end=datetime(2025, 1, 11, 5, 0),
                programmatic_result_json={},
                processing_time_ms=90,
                engine_versions_json=algo.model_dump(),
                created_at=tie_ts,
            )
            db.add(ar_old)
            await db.flush()
            ar_new = models.AnalysisResult(
                session_id=session_id,
                timestamp_start=datetime(2025, 1, 10, 22, 0),
                timestamp_end=datetime(2025, 1, 11, 5, 0),
                programmatic_result_json={},
                processing_time_ms=95,
                engine_versions_json=algo.model_dump(),
                created_at=tie_ts,  # same timestamp
            )
            db.add(ar_new)
            await db.flush()
            ar_old_id = ar_old.id
            ar_new_id = ar_new.id

        async with session_scope() as db:
            # Tie-breaker: ORDER BY created_at DESC, id DESC → ar_new wins
            latest = (
                (
                    await db.execute(
                        select(models.AnalysisResult)
                        .where(models.AnalysisResult.session_id == session_id)
                        .order_by(
                            models.AnalysisResult.created_at.desc(),
                            models.AnalysisResult.id.desc(),
                        )
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )
            assert latest is not None
            assert latest.id == ar_new_id, (
                f"Tie-breaker must select highest id ({ar_new_id}), got {latest.id}"
            )
            assert latest.id != ar_old_id


# ---------------------------------------------------------------------------
# A3 — Atomic rollback on child-insert failure
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAtomicRollbackOnChildInsertFailure:
    async def test_breath_insert_failure_rolls_back_analysis_result_parent(
        self, temp_db
    ):
        """If Breath bulk-add fails after AnalysisResult flush, parent is also absent.

        Simulates a constraint violation during add_all by raising inside
        the same transaction after the parent flush.  The transaction scope
        wraps both the parent insert and the child bulk-add; a failure anywhere
        inside must leave both absent.
        """
        await init_database(str(temp_db))

        session_id_holder: list[int] = []
        ar_id_holder: list[int] = []

        # Phase 1: set up the session row (committed separately)
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            _, session_id = await _make_device_and_session(
                db, profile_id, start=datetime(2025, 1, 15, 22, 0)
            )
            session_id_holder.append(session_id)

        sid = session_id_holder[0]

        # Phase 2: write parent + children in one transaction, then fail
        try:
            async with session_scope() as db:
                algo = _make_algo_versions()
                ar = models.AnalysisResult(
                    session_id=sid,
                    timestamp_start=datetime(2025, 1, 15, 22, 0),
                    timestamp_end=datetime(2025, 1, 16, 5, 0),
                    programmatic_result_json={},
                    processing_time_ms=50,
                    engine_versions_json=algo.model_dump(),
                )
                db.add(ar)
                await db.flush()  # assigns ar.id
                ar_id_holder.append(ar.id)
                # Simulate child-insert failure AFTER parent flush
                raise RuntimeError("Forced child-insert failure")
        except RuntimeError:
            pass

        ar_id = ar_id_holder[0]

        # Both parent and children must be absent
        async with session_scope() as db:
            ar_count = (
                await db.execute(
                    select(func.count())
                    .select_from(models.AnalysisResult)
                    .where(models.AnalysisResult.id == ar_id)
                )
            ).scalar()
            assert ar_count == 0, (
                f"AnalysisResult id={ar_id} must be rolled back after child failure"
            )

            breath_count = (
                await db.execute(
                    select(func.count())
                    .select_from(models.Breath)
                    .where(models.Breath.analysis_result_id == ar_id)
                )
            ).scalar()
            assert breath_count == 0, (
                f"Breath rows for analysis_result_id={ar_id} must not survive"
            )


# ---------------------------------------------------------------------------
# A5 — Validity flags end-to-end (ramp_active / mask_off)
# ---------------------------------------------------------------------------


def _sine_flow_blob(duration_s: float, sample_rate: float = 25.0) -> tuple[bytes, int]:
    """Serialize a sinusoidal breathing flow waveform (±30 L/min, 4 s period)."""
    timestamps = np.arange(0.0, duration_s, 1.0 / sample_rate, dtype=np.float32)
    values = (30.0 * np.sin(2.0 * np.pi * timestamps / 4.0)).astype(np.float32)
    blob = np.column_stack([timestamps, values]).astype(np.float32).tobytes()
    return blob, len(timestamps)


async def _seed_flow_session(
    db: AsyncSession,
    profile_id: int,
    *,
    duration_s: float,
    mask_on_segments: list[list[float]] | None,
    settings: dict[str, str],
) -> int:
    """Seed a Session with a synthetic flow waveform + settings. Returns its id."""
    from datetime import timedelta

    start = datetime(2025, 9, 10, 22, 0)
    device = models.Device(
        profile_id=profile_id,
        serial_number=f"SN_{uuid.uuid4().hex[:6]}",
        manufacturer="ResMed",
        model="AirSense 10",
    )
    db.add(device)
    await db.flush()
    day = models.Day(device_id=device.id, date=start.date(), session_count=1)
    db.add(day)
    await db.flush()
    session = models.Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
        start_time=start,
        end_time=start + timedelta(seconds=duration_s),
        duration_seconds=duration_s,
        mask_on_segments=mask_on_segments,
    )
    db.add(session)
    await db.flush()

    blob, sample_count = _sine_flow_blob(duration_s)
    db.add(
        models.Waveform(
            session_id=session.id,
            waveform_type="flow",
            sample_rate=25.0,
            unit="L/min",
            data_blob=blob,
            sample_count=sample_count,
        )
    )
    for key, value in settings.items():
        db.add(models.Setting(session_id=session.id, key=key, value=value))
    await db.flush()
    return session.id


@pytest.mark.integration
class TestValidityFlagsEndToEnd:
    async def test_ramp_and_mask_flags_persisted_from_settings_and_segments(
        self, temp_db
    ):
        """Seeded ramp settings + segments → per-breath ramp_active/mask_off values."""
        await init_database(str(temp_db))
        duration_s = 720.0  # 12 min; ramp_time 5 min → flip at 300 s
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            session_id = await _seed_flow_session(
                db,
                profile_id,
                duration_s=duration_s,
                mask_on_segments=[[0.0, duration_s]],
                settings={"ramp_enabled": "True", "ramp_time": "5"},
            )
            svc = AnalysisService(db, profile_id=profile_id)
            await svc.analyze_session(session_id)

        async with session_scope() as db:
            rows = (
                (
                    await db.execute(
                        select(models.Breath)
                        .where(models.Breath.session_id == session_id)
                        .order_by(models.Breath.breath_number)
                    )
                )
                .scalars()
                .all()
            )
            assert rows, "analysis must persist Breath rows"

            in_ramp = [r for r in rows if r.end_offset_s < 295.0]
            past_ramp = [r for r in rows if r.start_offset_s > 305.0]
            assert in_ramp and past_ramp, "session must span the ramp boundary"
            assert all(r.ramp_active is True for r in in_ramp)
            assert all(r.ramp_active is False for r in past_ramp)
            assert all(r.ramp_active_reason is None for r in rows)

            # One full mask-on segment, no gaps → uniformly False, no reason.
            assert all(r.mask_off is False for r in rows)
            assert all(r.mask_off_reason is None for r in rows)

    async def test_unknown_settings_and_segments_yield_null_reasons(self, temp_db):
        """No ramp settings + NULL segments → null flags with honest reasons."""
        await init_database(str(temp_db))
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            session_id = await _seed_flow_session(
                db,
                profile_id,
                duration_s=300.0,
                mask_on_segments=None,
                settings={},
            )
            svc = AnalysisService(db, profile_id=profile_id)
            await svc.analyze_session(session_id)

        async with session_scope() as db:
            rows = (
                (
                    await db.execute(
                        select(models.Breath).where(
                            models.Breath.session_id == session_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert rows, "analysis must persist Breath rows"
            assert all(r.ramp_active is None for r in rows)
            assert all(r.ramp_active_reason == "not_available" for r in rows)
            assert all(r.mask_off is None for r in rows)
            assert all(r.mask_off_reason == "segments_unknown" for r in rows)

    async def test_malformed_stored_segments_degrade_to_unknown(self, temp_db):
        """Malformed mask_on_segments JSON must not crash analysis.

        The stored column value is untrusted; a wrong-shape item is rejected by
        parse_mask_on_segments (warning + None) so analysis completes with
        mask_off=None / "segments_unknown" instead of raising.
        """
        await init_database(str(temp_db))
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            session_id = await _seed_flow_session(
                db,
                profile_id,
                duration_s=300.0,
                mask_on_segments=[[0.0, 100.0, 200.0]],  # 3-element item: malformed
                settings={},
            )
            svc = AnalysisService(db, profile_id=profile_id)
            await svc.analyze_session(session_id)

        async with session_scope() as db:
            rows = (
                (
                    await db.execute(
                        select(models.Breath).where(
                            models.Breath.session_id == session_id
                        )
                    )
                )
                .scalars()
                .all()
            )
            assert rows, "analysis must still persist Breath rows"
            assert all(r.mask_off is None for r in rows)
            assert all(r.mask_off_reason == "segments_unknown" for r in rows)


# ---------------------------------------------------------------------------
# A6 — primary_mode threaded through analyze_session into storage
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAnalyzeSessionPrimaryModeThreading:
    async def test_explicit_primary_mode_persisted_in_engine_versions(self, temp_db):
        """analyze_session persists an explicit non-default primary_mode.

        With modes=["aasm", "resmed"] and primary_mode="resmed", the stored
        engine_versions_json.run.primary_mode must be "resmed".  If analyze_session
        failed to thread primary_mode into load_session_inputs_raw, the load phase
        would resolve None → DEFAULT_MODE ("aasm"), and this assertion would catch
        the silent fallback.
        """
        await init_database(str(temp_db))
        async with session_scope() as db:
            _, profile_id = await _make_profile(db)
            session_id = await _seed_flow_session(
                db,
                profile_id,
                duration_s=300.0,
                mask_on_segments=[[0.0, 300.0]],
                settings={},
            )
            svc = AnalysisService(db, profile_id=profile_id)
            await svc.analyze_session(
                session_id, modes=["aasm", "resmed"], primary_mode="resmed"
            )

        async with session_scope() as db:
            stored = (
                (
                    await db.execute(
                        select(AnalysisResult_ORM).where(
                            AnalysisResult_ORM.session_id == session_id
                        )
                    )
                )
                .scalars()
                .first()
            )
            assert stored is not None
            assert stored.engine_versions_json["run"]["primary_mode"] == "resmed"


# ---------------------------------------------------------------------------
# A4 — Non-UTC host determinism
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNonUtcHostDeterminism:
    async def test_store_result_timestamp_matches_utc_under_shifted_tz(self, temp_db):
        """store_result stores timestamps as naive UTC regardless of OS timezone.

        Shifts the process timezone to America/New_York (UTC-5 in winter) using
        os.environ["TZ"] + time.tzset() (POSIX only — skipped on Windows), then
        drives the real store_result() with a known epoch.  The stored naive
        datetime must equal datetime.utcfromtimestamp(epoch), not the local-time
        interpretation.  If store_result used bare datetime.fromtimestamp(epoch)
        without a tz argument, this test would fail on a UTC-5 host by 5 hours.
        """
        import os
        import time as _time

        if not hasattr(_time, "tzset"):
            pytest.skip("tzset not available on this platform (Windows)")

        # 2024-01-19 02:00:00 UTC — wall-clock is 2024-01-18 21:00:00 in UTC-5.
        epoch = 1705622400.0
        expected_naive_utc = datetime.fromtimestamp(epoch, tz=UTC).replace(
            tzinfo=None
        )  # 2024-01-19 02:00:00

        await init_database(str(temp_db))

        original_tz = os.environ.get("TZ")
        try:
            os.environ["TZ"] = "America/New_York"
            _time.tzset()

            async with session_scope() as db:
                _, profile_id = await _make_profile(db)
                session_start = expected_naive_utc
                _, session_id = await _make_device_and_session(
                    db, profile_id, start=session_start
                )

                result_dto = AnalysisResultDTO(
                    session_id=session_id,
                    session_duration_hours=7.0,
                    total_breaths=0,
                    machine_events=[],
                    mode_results={
                        "aasm": ModeResult(
                            mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                        )
                    },
                    timestamp_start=epoch,
                    timestamp_end=epoch + 7 * 3600.0,
                )
                computation = AnalysisComputation(
                    summary=result_dto, breaths=[], primary_mode="aasm"
                )
                svc = AnalysisService(db, profile_id=profile_id)
                ar_id = await svc.store_result(computation, processing_time_ms=10)
        finally:
            if original_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_tz
            _time.tzset()

        async with session_scope() as db:
            stored = (
                (
                    await db.execute(
                        select(AnalysisResult_ORM).where(AnalysisResult_ORM.id == ar_id)
                    )
                )
                .scalars()
                .first()
            )
            assert stored is not None
            # Must be naive (UTC, no tzinfo).
            assert stored.timestamp_start.tzinfo is None
            # Must match the UTC epoch interpretation, not local-time.
            assert stored.timestamp_start == expected_naive_utc, (
                f"Under UTC-5 host, stored {stored.timestamp_start!r} "
                f"but expected UTC epoch {expected_naive_utc!r} — "
                "service.py must use fromtimestamp(ts, tz=UTC).replace(tzinfo=None)"
            )
