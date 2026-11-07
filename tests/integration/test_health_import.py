"""Integration tests for the Apple Health import pipeline.

Tests the full HealthImportService flow: file import, payload import, idempotency,
dry_run, and nightly summary generation.  Uses a real DB initialized via
init_database() — write_gate / run_txn / session_scope all resolve to the same file.
"""

from __future__ import annotations

import uuid

from datetime import date
from pathlib import Path

import pytest

from sqlalchemy import func, select

from snore.database.models import HealthNightlySummary, HealthSample, Profile, User
from snore.database.session import init_database, session_scope
from snore.services.health_import_service import HealthImportService
from snore.services.health_service import HealthService

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "health_data"
# Pass the directory to import_file — AppleHealthParser.detect() requires a directory
# or zip, not a bare XML file (xml_reader handles bare XML but detect() does not).
EXPORT_DIR = FIXTURE_DIR

# Sleep nights present in export.xml (from parser/fixture analysis):
# - 2024-01-14: Watch AsleepREM (11:30–11:45 on Jan 15, noon-split → Jan 14)
# - 2024-01-15: Watch stages + iPhone InBed (various records from Jan 15–16)
EXPECTED_NIGHTS = {date(2024, 1, 14), date(2024, 1, 15)}
WATCH_SOURCE = "Will's Apple Watch"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_profile_id() -> int:
    async with session_scope() as db:
        user = User(
            canonical_email=f"health_{uuid.uuid4().hex[:8]}@test.local",
            role="member",
        )
        db.add(user)
        await db.flush()
        profile = Profile(user_id=user.id, name="Health Test Profile")
        db.add(profile)
        await db.flush()
        return profile.id


async def _sample_count(profile_id: int) -> int:
    async with session_scope() as db:
        result = await db.execute(
            select(func.count())
            .select_from(HealthSample)
            .where(HealthSample.profile_id == profile_id)
        )
        return result.scalar() or 0


async def _summary_nights(profile_id: int) -> set[date]:
    async with session_scope() as db:
        result = await db.execute(
            select(HealthNightlySummary.night_date).where(
                HealthNightlySummary.profile_id == profile_id
            )
        )
        return {row[0] for row in result.fetchall()}


# ---------------------------------------------------------------------------
# File import tests
# ---------------------------------------------------------------------------


class TestImportFile:
    async def test_file_import_populates_samples_and_summaries(self, temp_db):
        """Basic file import: samples land in DB and sleep summaries are created."""
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        svc = HealthImportService()
        result = await svc.import_file(EXPORT_DIR, profile_id)

        assert result.dry_run is False
        assert result.inserted > 0, "At least some records should be inserted"
        assert result.skipped == 0, "First import should skip nothing"
        # StepCount is the one unhandled HK type in the fixture.
        assert "HKQuantityTypeIdentifierStepCount" in result.unknown_metrics

        # Health samples in DB.
        count = await _sample_count(profile_id)
        assert count == result.inserted

        # Nightly summaries exist for both sleep nights in the fixture.
        nights = await _summary_nights(profile_id)
        assert EXPECTED_NIGHTS.issubset(nights), (
            f"Expected summary nights {EXPECTED_NIGHTS!r} but got {nights!r}"
        )
        assert result.nights_recomputed == len(nights)

    async def test_watch_preferred_on_overlapping_source_night(self, temp_db):
        """Night 2024-01-15 has Watch stage records and iPhone InBed → Watch preferred."""
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        svc = HealthImportService()
        await svc.import_file(EXPORT_DIR, profile_id)

        night = date(2024, 1, 15)
        async with session_scope() as db:
            result = await db.execute(
                select(HealthNightlySummary).where(
                    HealthNightlySummary.profile_id == profile_id,
                    HealthNightlySummary.night_date == night,
                )
            )
            summary = result.scalars().first()

        assert summary is not None
        assert summary.preferred_source == WATCH_SOURCE

    async def test_idempotent_reimport(self, temp_db):
        """Second import of the same file inserts 0 records and leaves summaries unchanged."""
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        svc = HealthImportService()
        first = await svc.import_file(EXPORT_DIR, profile_id)

        # Snapshot the summaries after first import.
        async with session_scope() as db:
            r1 = await db.execute(
                select(HealthNightlySummary).where(
                    HealthNightlySummary.profile_id == profile_id
                )
            )
            summaries_before = {
                s.night_date: s.total_sleep_seconds for s in r1.scalars().all()
            }

        second = await svc.import_file(EXPORT_DIR, profile_id)

        assert second.inserted == 0, "Re-import must insert nothing"
        assert second.skipped == first.inserted, (
            "All records must be skipped on re-import"
        )

        # Summaries should be unchanged.
        async with session_scope() as db:
            r2 = await db.execute(
                select(HealthNightlySummary).where(
                    HealthNightlySummary.profile_id == profile_id
                )
            )
            summaries_after = {
                s.night_date: s.total_sleep_seconds for s in r2.scalars().all()
            }

        assert summaries_before == summaries_after, (
            "Summaries must not change on idempotent re-import"
        )

    async def test_dry_run_reports_counts_without_writing(self, temp_db):
        """dry_run=True counts would-be inserts but writes no rows."""
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        svc = HealthImportService()
        result = await svc.import_file(EXPORT_DIR, profile_id, dry_run=True)

        assert result.dry_run is True
        assert result.inserted > 0, "dry_run must report would-be inserts"
        assert result.nights_recomputed == 0

        # No data written.
        count = await _sample_count(profile_id)
        assert count == 0, "dry_run must not write any samples"
        nights = await _summary_nights(profile_id)
        assert len(nights) == 0, "dry_run must not write any summaries"

    async def test_dry_run_after_first_import_reports_all_skipped(self, temp_db):
        """After a real import, dry_run must report all records as would-be-skipped."""
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        svc = HealthImportService()
        first = await svc.import_file(EXPORT_DIR, profile_id)
        dry = await svc.import_file(EXPORT_DIR, profile_id, dry_run=True)

        assert dry.inserted == 0
        assert dry.skipped == first.inserted

    async def test_invalid_path_raises(self, temp_db, tmp_path):
        """Non-Apple-Health path raises ValueError."""
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        svc = HealthImportService()
        not_health = tmp_path / "random.xml"
        not_health.write_text("<HealthData/>")

        with pytest.raises(ValueError, match="not a supported Apple Health export"):
            await svc.import_file(not_health, profile_id)

    async def test_progress_callback_receives_increments(self, temp_db):
        """progress_callback is called with increasing processed counts."""
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        calls: list[int] = []
        svc = HealthImportService()
        await svc.import_file(EXPORT_DIR, profile_id, progress_callback=calls.append)

        assert len(calls) >= 1
        assert calls[-1] > 0
        # Counts must be monotonically non-decreasing.
        assert calls == sorted(calls)


# ---------------------------------------------------------------------------
# Cancellation tests
# ---------------------------------------------------------------------------


class TestCancellation:
    async def test_cancel_predicate_yields_partial_counts_and_committed_summaries(
        self, temp_db
    ):
        """cancel_predicate stops import after the first batch; committed data survives.

        With batch_size=2 the fixture's 10 records split into 5 batches.  A
        closure that returns True after the first call (i.e., before the second
        batch) means:
          - Batch 1 (2 records) is committed.
          - The predicate fires True before batch 2 → loop breaks.
          - Nightly summaries are recomputed for nights touched by batch 1.
          - result.inserted < 10 and result.inserted > 0.
          - result.nights_recomputed > 0 (at least one night committed).
        """
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        batch_calls: list[int] = [0]

        def cancel_after_first_batch() -> bool:
            batch_calls[0] += 1
            # First call (before batch 1): allow. Second call (before batch 2): cancel.
            return batch_calls[0] > 1

        svc = HealthImportService()
        result = await svc.import_file(
            EXPORT_DIR,
            profile_id,
            batch_size=2,
            cancel_predicate=cancel_after_first_batch,
        )

        # Partial import: some records inserted, not all.
        assert result.inserted > 0, "At least one batch must have committed"
        assert result.inserted < 10, (
            "Cancel must stop before all 10 records are imported"
        )
        assert result.dry_run is False

        # Summaries are recomputed for every night touched by committed batches.
        assert result.nights_recomputed > 0, (
            "At least one nightly summary must be recomputed for committed nights"
        )

        # At least one nightly summary must be present in the DB.
        # nights_recomputed counts all nights where recompute ran; nights that
        # had only quantity samples (SpO2, RR) end up with no sleep summary
        # after recompute (recompute deletes the row when no sleep stages remain).
        nights_in_db = await _summary_nights(profile_id)
        assert len(nights_in_db) > 0, (
            "At least one nightly summary must exist for the committed nights"
        )
        assert len(nights_in_db) <= result.nights_recomputed, (
            "nights_in_db must be a subset of the nights that were recomputed"
        )


# ---------------------------------------------------------------------------
# nights_recomputed semantics
# ---------------------------------------------------------------------------


class TestNightsRecomputed:
    async def test_sleep_nights_counted_quantity_only_nights_excluded(
        self, temp_db, tmp_path
    ):
        """nights_recomputed counts only nights that received SLEEP records.

        The fixture has 2 sleep nights (2024-03-10, 2024-03-11) and 1 night
        with only a SpO2 record (2024-03-12, because the record starts at
        04:00 on the 13th which is before noon → prior night).  The SpO2-only
        night produces no nightly summary (recompute is a no-op), so
        nights_recomputed must equal 2, not 3.
        """
        # Build a minimal export.xml: 2 sleep nights + 1 SpO2-only night.
        (tmp_path / "export.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<HealthData>"
            ' <ExportDate value="2024-03-15 09:00:00 +0000"/>'
            " <Me/>"
            # Night 2024-03-10: sleep record (start 22:00 ≥ noon → same day)
            ' <Record type="HKCategoryTypeIdentifierSleepAnalysis"'
            ' sourceName="Watch" sourceVersion="1.0" device=""'
            ' creationDate="2024-03-11 06:00:00 +0000"'
            ' startDate="2024-03-10 22:00:00 +0000"'
            ' endDate="2024-03-11 06:00:00 +0000"'
            ' value="HKCategoryValueSleepAnalysisAsleepREM" unit=""/>'
            # Night 2024-03-11: sleep record (start 22:00 ≥ noon → same day)
            ' <Record type="HKCategoryTypeIdentifierSleepAnalysis"'
            ' sourceName="Watch" sourceVersion="1.0" device=""'
            ' creationDate="2024-03-12 06:00:00 +0000"'
            ' startDate="2024-03-11 22:00:00 +0000"'
            ' endDate="2024-03-12 06:00:00 +0000"'
            ' value="HKCategoryValueSleepAnalysisAsleepREM" unit=""/>'
            # Night 2024-03-12: SpO2 ONLY (start 04:00 < noon → previous day)
            ' <Record type="HKQuantityTypeIdentifierOxygenSaturation"'
            ' sourceName="Watch" sourceVersion="1.0" device=""'
            ' creationDate="2024-03-13 04:00:00 +0000"'
            ' startDate="2024-03-13 04:00:00 +0000"'
            ' endDate="2024-03-13 04:00:00 +0000"'
            ' value="0.95" unit="%"/>'
            "</HealthData>"
        )
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        svc = HealthImportService()
        result = await svc.import_file(tmp_path, profile_id)

        # 2 sleep nights, not 3 — quantity-only night excluded from count.
        assert result.nights_recomputed == 2
        nights = await _summary_nights(profile_id)
        assert date(2024, 3, 10) in nights
        assert date(2024, 3, 11) in nights
        assert date(2024, 3, 12) not in nights, (
            "SpO2-only night must not produce a nightly summary"
        )


# ---------------------------------------------------------------------------
# SpO2 scale normalisation through to the night aggregate
# ---------------------------------------------------------------------------


def _spo2_record(source: str, start: str, value: str) -> str:
    return (
        ' <Record type="HKQuantityTypeIdentifierOxygenSaturation"'
        f' sourceName="{source}" sourceVersion="1.0" device=""'
        f' creationDate="{start}" startDate="{start}" endDate="{start}"'
        f' value="{value}" unit="%"/>'
    )


class TestSpo2Scale:
    async def test_fraction_and_percent_sources_same_night_aggregate_in_percent(
        self, temp_db, tmp_path
    ):
        """A fraction source and a percent source on one night average on one scale."""
        (tmp_path / "export.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<HealthData>"
            ' <ExportDate value="2024-03-15 09:00:00 +0000"/>'
            " <Me/>"
            ' <Record type="HKCategoryTypeIdentifierSleepAnalysis"'
            ' sourceName="Watch A" sourceVersion="1.0" device=""'
            ' creationDate="2024-03-11 06:00:00 +0000"'
            ' startDate="2024-03-10 22:00:00 +0000"'
            ' endDate="2024-03-11 06:00:00 +0000"'
            ' value="HKCategoryValueSleepAnalysisAsleepCore" unit=""/>'
            + _spo2_record("Watch A", "2024-03-11 02:00:00 +0000", "0.95")
            + _spo2_record("Oximeter B", "2024-03-11 02:05:00 +0000", "92")
            + "</HealthData>"
        )
        await init_database(str(temp_db))
        profile_id = await _create_profile_id()

        await HealthImportService().import_file(tmp_path, profile_id)
        async with session_scope() as db:
            detail = await HealthService(db, profile_id).get_night_detail(
                date(2024, 3, 10)
            )

        assert detail.avg_spo2_pct == pytest.approx(93.5)
        assert detail.min_spo2_pct == pytest.approx(92)
