"""Integration tests for the Apple Health read API (/api/v1/health/*)."""

from __future__ import annotations

import uuid

from datetime import UTC, date, datetime

import pytest

from sqlalchemy.orm import Session

from snore.database.models import HealthNightlySummary, HealthSample, Profile, User

_SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
_SPO2_TYPE = "HKQuantityTypeIdentifierOxygenSaturation"
_RR_TYPE = "HKQuantityTypeIdentifierRespiratoryRate"
_WATCH = "Will's Apple Watch"
_IPHONE = "Will's iPhone"


class TestListNights:
    def test_empty_profile_returns_empty_list(self, api_client):
        """GET /health/nights with no data returns {items: [], total: 0}."""
        resp = api_client.get("/api/v1/health/nights")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0

    def test_seeded_summary_appears_in_list(self, api_client, db_session, test_profile):
        """A seeded HealthNightlySummary is visible in GET /health/nights with correct fields."""
        db_session.add(
            HealthNightlySummary(
                profile_id=test_profile.id,
                night_date=date(2024, 1, 15),
                total_sleep_seconds=7 * 3600,
                time_in_bed_seconds=8 * 3600,
                sleep_efficiency_pct=87.5,
                computed_at=datetime.now(UTC),
            )
        )
        db_session.flush()

        resp = api_client.get("/api/v1/health/nights")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        item = data["items"][0]
        assert item["night_date"] == "2024-01-15"
        assert item["total_sleep_seconds"] == pytest.approx(7 * 3600)
        assert item["time_in_bed_seconds"] == pytest.approx(8 * 3600)
        assert item["sleep_efficiency_pct"] == pytest.approx(87.5)


class TestGetNightDetail:
    def test_unknown_night_returns_404(self, api_client, test_profile):
        """GET /health/nights/{night_date} with no summary returns 404."""
        resp = api_client.get("/api/v1/health/nights/2030-01-01")
        assert resp.status_code == 404

    def test_profile_isolation_other_profile_night_is_404(
        self, api_client, db_session, test_profile
    ):
        """A summary that belongs to another profile is invisible to this actor."""
        other_user = User(
            canonical_email=f"other_{uuid.uuid4().hex[:8]}@test.local",
            role="member",
        )
        db_session.add(other_user)
        db_session.flush()
        other_profile = Profile(user_id=other_user.id, name="Other")
        db_session.add(other_profile)
        db_session.flush()

        # Seed a summary only for the OTHER profile.
        db_session.add(
            HealthNightlySummary(
                profile_id=other_profile.id,
                night_date=date(2024, 1, 15),
                total_sleep_seconds=6 * 3600,
                computed_at=datetime.now(UTC),
            )
        )
        db_session.flush()

        # The actor's own profile has no data for this date.
        resp = api_client.get("/api/v1/health/nights/2024-01-15")
        assert resp.status_code == 404

    def test_detail_returns_stage_seconds_and_efficiency(
        self, api_client, db_session, test_profile
    ):
        """Seeded summary returns correct stage seconds and sleep efficiency."""
        db_session.add(
            HealthNightlySummary(
                profile_id=test_profile.id,
                night_date=date(2024, 1, 15),
                total_sleep_seconds=7 * 3600,
                time_in_bed_seconds=8 * 3600,
                core_seconds=2 * 3600,
                deep_seconds=1 * 3600,
                rem_seconds=4 * 3600,
                sleep_efficiency_pct=87.5,
                stage_coverage_pct=100.0,
                computed_at=datetime.now(UTC),
            )
        )
        db_session.flush()

        resp = api_client.get("/api/v1/health/nights/2024-01-15")
        assert resp.status_code == 200
        data = resp.json()
        assert data["night_date"] == "2024-01-15"
        assert data["core_seconds"] == pytest.approx(2 * 3600)
        assert data["deep_seconds"] == pytest.approx(1 * 3600)
        assert data["rem_seconds"] == pytest.approx(4 * 3600)
        assert data["sleep_efficiency_pct"] == pytest.approx(87.5)
        assert data["stage_coverage_pct"] == pytest.approx(100.0)
        # No SpO2/RR samples seeded → null in response.
        assert data["avg_spo2_pct"] is None
        assert data["min_spo2_pct"] is None
        assert data["avg_rr"] is None

    def test_spo2_fraction_normalized_to_percent(
        self, api_client, db_session, test_profile
    ):
        """SpO2 values stored as fractions (≤ 1.5) are multiplied by 100 on read."""
        db_session.add(
            HealthNightlySummary(
                profile_id=test_profile.id,
                night_date=date(2024, 1, 15),
                total_sleep_seconds=7 * 3600,
                computed_at=datetime.now(UTC),
            )
        )
        # Fraction values: 0.95 and 0.99 → avg 0.97 → 97.0%
        for val, h in [(0.95, 2), (0.99, 3)]:
            db_session.add(
                HealthSample(
                    profile_id=test_profile.id,
                    record_type=_SPO2_TYPE,
                    source_name=_WATCH,
                    start_time=datetime(2024, 1, 16, h),
                    end_time=datetime(2024, 1, 16, h),
                    value_num=val,
                    unit="%",
                    night_date=date(2024, 1, 15),
                    ingest_channel="export_xml",
                )
            )
        db_session.flush()

        resp = api_client.get("/api/v1/health/nights/2024-01-15")
        assert resp.status_code == 200
        data = resp.json()
        # each sample ≤ 1.5 → multiply by 100
        assert data["avg_spo2_pct"] == pytest.approx(97.0, abs=0.1)
        assert data["min_spo2_pct"] == pytest.approx(95.0, abs=0.1)

    def test_spo2_percent_stored_unchanged(self, api_client, db_session, test_profile):
        """SpO2 values already in percent (> 1.5) are not re-multiplied."""
        db_session.add(
            HealthNightlySummary(
                profile_id=test_profile.id,
                night_date=date(2024, 1, 15),
                total_sleep_seconds=7 * 3600,
                computed_at=datetime.now(UTC),
            )
        )
        # Percent values: 95.0 and 99.0 → avg 97.0 (no multiplication)
        for val, h in [(95.0, 2), (99.0, 3)]:
            db_session.add(
                HealthSample(
                    profile_id=test_profile.id,
                    record_type=_SPO2_TYPE,
                    source_name=_WATCH,
                    start_time=datetime(2024, 1, 16, h),
                    end_time=datetime(2024, 1, 16, h),
                    value_num=val,
                    unit="%",
                    night_date=date(2024, 1, 15),
                    ingest_channel="export_xml",
                )
            )
        db_session.flush()

        resp = api_client.get("/api/v1/health/nights/2024-01-15")
        assert resp.status_code == 200
        data = resp.json()
        # each sample > 1.5 → left as-is (just rounded)
        assert data["avg_spo2_pct"] == pytest.approx(97.0, abs=0.1)
        assert data["min_spo2_pct"] == pytest.approx(95.0, abs=0.1)

    def test_rr_avg_returned_from_seeded_samples(
        self, api_client, db_session, test_profile
    ):
        """avg_rr is correctly aggregated from RespiratoryRate samples."""
        db_session.add(
            HealthNightlySummary(
                profile_id=test_profile.id,
                night_date=date(2024, 1, 15),
                total_sleep_seconds=7 * 3600,
                computed_at=datetime.now(UTC),
            )
        )
        for val, h in [(14.0, 2), (16.0, 3)]:
            db_session.add(
                HealthSample(
                    profile_id=test_profile.id,
                    record_type=_RR_TYPE,
                    source_name=_WATCH,
                    start_time=datetime(2024, 1, 16, h),
                    end_time=datetime(2024, 1, 16, h),
                    value_num=val,
                    unit="count/min",
                    night_date=date(2024, 1, 15),
                    ingest_channel="export_xml",
                )
            )
        db_session.flush()

        resp = api_client.get("/api/v1/health/nights/2024-01-15")
        assert resp.status_code == 200
        data = resp.json()
        assert data["avg_rr"] == pytest.approx(15.0, abs=0.01)


class TestListNightsPaginationAndFiltering:
    """Pagination, ordering, and date-range filtering for GET /health/nights."""

    def _seed_five_nights(self, db_session: Session, profile_id: int) -> None:
        """Seed summaries for 2024-01-11 through 2024-01-15 (five consecutive nights)."""
        for d in [
            date(2024, 1, 11),
            date(2024, 1, 12),
            date(2024, 1, 13),
            date(2024, 1, 14),
            date(2024, 1, 15),
        ]:
            db_session.add(
                HealthNightlySummary(
                    profile_id=profile_id,
                    night_date=d,
                    total_sleep_seconds=7 * 3600,
                    computed_at=datetime.now(UTC),
                )
            )
        db_session.flush()

    def test_results_ordered_most_recent_first(
        self, api_client, db_session, test_profile
    ):
        """GET /health/nights returns nights in descending date order."""
        self._seed_five_nights(db_session, test_profile.id)

        resp = api_client.get("/api/v1/health/nights")

        assert resp.status_code == 200
        data = resp.json()
        dates = [item["night_date"] for item in data["items"]]
        assert dates == sorted(dates, reverse=True)

    def test_limit_and_offset_return_middle_slice(
        self, api_client, db_session, test_profile
    ):
        """limit=2 offset=2 on 5 nights yields the expected middle slice; total=5."""
        self._seed_five_nights(db_session, test_profile.id)

        resp = api_client.get("/api/v1/health/nights?limit=2&offset=2")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2
        # Most-recent-first order: 01-15, 01-14, [01-13, 01-12], 01-11
        dates = [item["night_date"] for item in data["items"]]
        assert dates == ["2024-01-13", "2024-01-12"]

    def test_from_date_is_inclusive_excludes_earlier_nights(
        self, api_client, db_session, test_profile
    ):
        """from_date boundary night is included; nights before it are excluded."""
        self._seed_five_nights(db_session, test_profile.id)

        resp = api_client.get("/api/v1/health/nights?from_date=2024-01-13")

        assert resp.status_code == 200
        data = resp.json()
        dates = [item["night_date"] for item in data["items"]]
        assert "2024-01-13" in dates  # boundary is included
        assert "2024-01-12" not in dates  # one night before is excluded
        assert data["total"] == 3  # 01-13, 01-14, 01-15

    def test_to_date_is_inclusive_excludes_later_nights(
        self, api_client, db_session, test_profile
    ):
        """to_date boundary night is included; nights after it are excluded."""
        self._seed_five_nights(db_session, test_profile.id)

        resp = api_client.get("/api/v1/health/nights?to_date=2024-01-13")

        assert resp.status_code == 200
        data = resp.json()
        dates = [item["night_date"] for item in data["items"]]
        assert "2024-01-13" in dates  # boundary is included
        assert "2024-01-14" not in dates  # one night after is excluded
        assert data["total"] == 3  # 01-11, 01-12, 01-13


class TestGetNightSamples:
    def _seed_two_source_night(self, db_session: Session, profile_id: int) -> None:
        """Seed a night with Watch stage rows (preferred) and one iPhone InBed."""
        db_session.add(
            HealthNightlySummary(
                profile_id=profile_id,
                night_date=date(2024, 1, 15),
                preferred_source=_WATCH,
                total_sleep_seconds=6 * 3600,
                computed_at=datetime.now(UTC),
            )
        )
        # Three Watch sleep-stage records at different start times.
        for i, stage in enumerate(["AsleepCore", "AsleepDeep", "AsleepREM"]):
            db_session.add(
                HealthSample(
                    profile_id=profile_id,
                    record_type=_SLEEP_TYPE,
                    source_name=_WATCH,
                    start_time=datetime(2024, 1, 16, i),
                    end_time=datetime(2024, 1, 16, i + 1),
                    value_text=stage,
                    night_date=date(2024, 1, 15),
                    ingest_channel="export_xml",
                )
            )
        # One iPhone InBed record.
        db_session.add(
            HealthSample(
                profile_id=profile_id,
                record_type=_SLEEP_TYPE,
                source_name=_IPHONE,
                start_time=datetime(2024, 1, 15, 23),
                end_time=datetime(2024, 1, 16, 7),
                value_text="InBed",
                night_date=date(2024, 1, 15),
                ingest_channel="export_xml",
            )
        )
        # SpO2 quantity record — must not appear in the samples response.
        db_session.add(
            HealthSample(
                profile_id=profile_id,
                record_type=_SPO2_TYPE,
                source_name=_WATCH,
                start_time=datetime(2024, 1, 16, 2),
                end_time=datetime(2024, 1, 16, 2),
                value_num=0.97,
                unit="%",
                night_date=date(2024, 1, 15),
                ingest_channel="export_xml",
            )
        )
        db_session.flush()

    def test_default_returns_only_preferred_source_stage_rows(
        self, api_client, db_session, test_profile
    ):
        """Default call returns only Watch sleep-stage rows ordered by start_time."""
        self._seed_two_source_night(db_session, test_profile.id)

        resp = api_client.get("/api/v1/health/nights/2024-01-15/samples")
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) == 3
        assert all(s["source_name"] == _WATCH for s in samples)
        assert all(s["record_type"] == _SLEEP_TYPE for s in samples)
        # Ordered by start_time ascending.
        times = [s["start_time"] for s in samples]
        assert times == sorted(times)

    def test_explicit_source_name_overrides_preferred(
        self, api_client, db_session, test_profile
    ):
        """?source_name=iPhone returns only iPhone rows regardless of preferred_source."""
        self._seed_two_source_night(db_session, test_profile.id)

        resp = api_client.get(
            f"/api/v1/health/nights/2024-01-15/samples?source_name={_IPHONE}"
        )
        assert resp.status_code == 200
        samples = resp.json()
        assert len(samples) == 1
        assert samples[0]["source_name"] == _IPHONE
        assert samples[0]["value_text"] == "InBed"

    def test_quantity_rows_never_appear_in_samples(
        self, api_client, db_session, test_profile
    ):
        """SpO2 and RR quantity rows are excluded from the samples endpoint."""
        self._seed_two_source_night(db_session, test_profile.id)

        # Fetch watch samples — SpO2 row is Watch-sourced but must not appear.
        resp = api_client.get(
            f"/api/v1/health/nights/2024-01-15/samples?source_name={_WATCH}"
        )
        assert resp.status_code == 200
        samples = resp.json()
        assert all(s["record_type"] == _SLEEP_TYPE for s in samples)

    def test_missing_night_returns_404(self, api_client, test_profile):
        """GET /health/nights/{date}/samples with no summary returns 404."""
        resp = api_client.get("/api/v1/health/nights/2030-01-01/samples")
        assert resp.status_code == 404


class TestListNightDates:
    def test_returns_all_seeded_dates_ascending(
        self, api_client, db_session, test_profile
    ):
        """GET /health/nights/dates returns all seeded dates in ascending order."""
        for d in [date(2024, 1, 14), date(2024, 1, 15)]:
            db_session.add(
                HealthNightlySummary(
                    profile_id=test_profile.id,
                    night_date=d,
                    total_sleep_seconds=6 * 3600,
                    computed_at=datetime.now(UTC),
                )
            )
        db_session.flush()

        resp = api_client.get("/api/v1/health/nights/dates")
        assert resp.status_code == 200
        data = resp.json()
        assert "2024-01-14" in data["dates"]
        assert "2024-01-15" in data["dates"]
        assert data["dates"] == sorted(data["dates"])
