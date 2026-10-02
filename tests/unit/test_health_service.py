"""Unit tests for HealthService.

Uses the global async_db_session fixture (real in-memory SQLite DB) — the
repo pattern for service unit tests that require a database, matching
test_health_importers.py's approach.

Service methods tested:
- get_night_detail: SpO2 fraction→percent normalization, RR avg aggregation
- get_night_samples: preferred-source filtering, explicit source_name override
- list_night_dates: ascending sort order
"""

from __future__ import annotations

import uuid

from datetime import UTC, date, datetime

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import HealthNightlySummary, HealthSample, Profile, User
from snore.services.health_service import HealthService, spo2_display_pct

_SLEEP_TYPE = "HKCategoryTypeIdentifierSleepAnalysis"
_SPO2_TYPE = "HKQuantityTypeIdentifierOxygenSaturation"
_RR_TYPE = "HKQuantityTypeIdentifierRespiratoryRate"
_WATCH = "Will's Apple Watch"
_IPHONE = "Will's iPhone"
_NIGHT = date(2024, 1, 15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dt(h: int, mi: int = 0, day: int = 16) -> datetime:
    """Shorthand for a 2024-01-{day} datetime."""
    return datetime(2024, 1, day, h, mi)


async def _make_profile(db: AsyncSession) -> int:
    user = User(
        canonical_email=f"svc_{uuid.uuid4().hex[:8]}@test.local",
        role="member",
    )
    db.add(user)
    await db.flush()
    profile = Profile(user_id=user.id, name="SvcTest")
    db.add(profile)
    await db.flush()
    return profile.id


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _add_spo2_samples(
    db: AsyncSession, profile_id: int, values: tuple[float, ...]
) -> None:
    """Add one SpO2 sample per value, an hour apart starting at 02:00."""
    for h, val in enumerate(values, start=2):
        db.add(
            HealthSample(
                profile_id=profile_id,
                record_type=_SPO2_TYPE,
                source_name=_WATCH,
                start_time=_dt(h),
                end_time=_dt(h),
                value_num=val,
                unit="%",
                night_date=_NIGHT,
                ingest_channel="export_xml",
            )
        )


@pytest.fixture
async def profile_id(async_db_session):
    return await _make_profile(async_db_session)


@pytest.fixture
async def night_summary(async_db_session, profile_id):
    """Minimal HealthNightlySummary for 2024-01-15 with no preferred_source."""
    s = HealthNightlySummary(
        profile_id=profile_id,
        night_date=_NIGHT,
        total_sleep_seconds=7 * 3600,
        computed_at=datetime.now(UTC),
    )
    async_db_session.add(s)
    await async_db_session.flush()
    return s


# ---------------------------------------------------------------------------
# get_night_detail — SpO2 + RR aggregation
# ---------------------------------------------------------------------------


class TestGetNightDetailAggregates:
    async def test_fraction_spo2_normalized_to_percent(
        self, async_db_session, profile_id, night_summary
    ):
        """SpO2 stored as fractions (≤ 1.5) is multiplied by 100 on read.

        Fixture: 0.95 and 0.99 → avg 0.97, min 0.95 → both become × 100.
        """
        for val, h in [(0.95, 2), (0.99, 3)]:
            async_db_session.add(
                HealthSample(
                    profile_id=profile_id,
                    record_type=_SPO2_TYPE,
                    source_name=_WATCH,
                    start_time=_dt(h),
                    end_time=_dt(h),
                    value_num=val,
                    unit="%",
                    night_date=_NIGHT,
                    ingest_channel="export_xml",
                )
            )
        await async_db_session.flush()

        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_spo2_pct == pytest.approx(97.0, abs=0.1)
        assert detail.min_spo2_pct == pytest.approx(95.0, abs=0.1)

    async def test_percent_spo2_not_re_multiplied(
        self, async_db_session, profile_id, night_summary
    ):
        """SpO2 stored as percents (> 1.5) is left unchanged (just rounded).

        Fixture: 95.0 and 99.0 → avg 97.0, min 95.0 — no multiplication.
        """
        for val, h in [(95.0, 2), (99.0, 3)]:
            async_db_session.add(
                HealthSample(
                    profile_id=profile_id,
                    record_type=_SPO2_TYPE,
                    source_name=_WATCH,
                    start_time=_dt(h),
                    end_time=_dt(h),
                    value_num=val,
                    unit="%",
                    night_date=_NIGHT,
                    ingest_channel="export_xml",
                )
            )
        await async_db_session.flush()

        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_spo2_pct == pytest.approx(97.0, abs=0.1)
        assert detail.min_spo2_pct == pytest.approx(95.0, abs=0.1)

    async def test_rr_avg_computed_from_samples(
        self, async_db_session, profile_id, night_summary
    ):
        """avg_rr is the mean of all RespiratoryRate samples for the night."""
        for val, h in [(14.0, 2), (16.0, 3)]:
            async_db_session.add(
                HealthSample(
                    profile_id=profile_id,
                    record_type=_RR_TYPE,
                    source_name=_WATCH,
                    start_time=_dt(h),
                    end_time=_dt(h),
                    value_num=val,
                    unit="count/min",
                    night_date=_NIGHT,
                    ingest_channel="export_xml",
                )
            )
        await async_db_session.flush()

        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_rr == pytest.approx(15.0, abs=0.01)

    @pytest.mark.parametrize(
        ("values", "expected_avg", "expected_min"),
        [
            ((0.95, 95.0), 95.0, 95.0),
            ((0.99, 92.0), 95.5, 92.0),
        ],
    )
    async def test_mixed_unit_spo2_normalized_before_aggregating(
        self,
        async_db_session,
        profile_id,
        night_summary,
        values,
        expected_avg,
        expected_min,
    ):
        """Mixed fraction+percent SpO2 sources: each sample is normalized first.

        Aggregating raw values gives avg 47.975 for 0.95 + 95 and min 0.99 (→ 99%)
        for 0.99 + 92 — both wrong scales.
        """
        _add_spo2_samples(async_db_session, profile_id, values)
        await async_db_session.flush()

        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_spo2_pct == pytest.approx(expected_avg, abs=0.1)
        assert detail.min_spo2_pct == pytest.approx(expected_min, abs=0.1)

    @pytest.mark.parametrize("implausible", [0.0, 1.5, -1.0, 20.0, 101.0])
    async def test_implausible_spo2_sample_ignored(
        self, async_db_session, profile_id, night_summary, implausible
    ):
        """A sample outside 50–100% after normalization is dropped from avg/min.

        Without the filter: 0.0 → 0%, 1.5 → 150%, -1 → -100%, 20 → 20%.
        """
        _add_spo2_samples(async_db_session, profile_id, (implausible, 95.0))
        await async_db_session.flush()

        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_spo2_pct == pytest.approx(95.0, abs=0.1)
        assert detail.min_spo2_pct == pytest.approx(95.0, abs=0.1)

    async def test_only_implausible_spo2_returns_none(
        self, async_db_session, profile_id, night_summary
    ):
        """A night whose SpO2 samples are all implausible has no SpO2 aggregates."""
        _add_spo2_samples(async_db_session, profile_id, (0.0, 1.5, 20.0))
        await async_db_session.flush()

        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_spo2_pct is None
        assert detail.min_spo2_pct is None

    async def test_rr_samples_excluded_from_spo2_aggregates(
        self, async_db_session, profile_id, night_summary
    ):
        """RR samples on the same night feed avg_rr only, never the SpO2 stats.

        An RR of 60 /min sits inside the plausible SpO2 percent range, so it would
        drag avg/min SpO2 down if it leaked into those aggregates.
        """
        _add_spo2_samples(async_db_session, profile_id, (0.95, 97.0))
        for val, h in [(60.0, 4), (14.0, 5)]:
            async_db_session.add(
                HealthSample(
                    profile_id=profile_id,
                    record_type=_RR_TYPE,
                    source_name=_WATCH,
                    start_time=_dt(h),
                    end_time=_dt(h),
                    value_num=val,
                    unit="count/min",
                    night_date=_NIGHT,
                    ingest_channel="export_xml",
                )
            )
        await async_db_session.flush()

        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_spo2_pct == pytest.approx(96.0, abs=0.1)
        assert detail.min_spo2_pct == pytest.approx(95.0, abs=0.1)
        assert detail.avg_rr == pytest.approx(37.0, abs=0.01)

    async def test_no_spo2_or_rr_returns_none(
        self, async_db_session, profile_id, night_summary
    ):
        """avg_spo2_pct, min_spo2_pct, and avg_rr are None when no samples exist."""
        detail = await HealthService(async_db_session, profile_id).get_night_detail(
            _NIGHT
        )

        assert detail.avg_spo2_pct is None
        assert detail.min_spo2_pct is None
        assert detail.avg_rr is None


# ---------------------------------------------------------------------------
# get_night_samples — source filtering
# ---------------------------------------------------------------------------


class TestGetNightSamplesFiltering:
    async def _seed_two_source_night(
        self, db: AsyncSession, pid: int, preferred: str = _WATCH
    ) -> None:
        """Seed a summary + Watch stage records + iPhone InBed for 2024-01-15."""
        db.add(
            HealthNightlySummary(
                profile_id=pid,
                night_date=_NIGHT,
                preferred_source=preferred,
                total_sleep_seconds=6 * 3600,
                computed_at=datetime.now(UTC),
            )
        )
        # Watch stage records.
        for i, stage in enumerate(["AsleepCore", "AsleepDeep"]):
            db.add(
                HealthSample(
                    profile_id=pid,
                    record_type=_SLEEP_TYPE,
                    source_name=_WATCH,
                    start_time=_dt(i + 1),
                    end_time=_dt(i + 2),
                    value_text=stage,
                    night_date=_NIGHT,
                    ingest_channel="export_xml",
                )
            )
        # iPhone InBed.
        db.add(
            HealthSample(
                profile_id=pid,
                record_type=_SLEEP_TYPE,
                source_name=_IPHONE,
                start_time=datetime(2024, 1, 15, 23),
                end_time=datetime(2024, 1, 16, 7),
                value_text="InBed",
                night_date=_NIGHT,
                ingest_channel="export_xml",
            )
        )
        await db.flush()

    async def test_default_uses_preferred_source(self, async_db_session, profile_id):
        """Without explicit source_name, only preferred_source sleep rows are returned."""
        await self._seed_two_source_night(
            async_db_session, profile_id, preferred=_WATCH
        )

        samples = await HealthService(async_db_session, profile_id).get_night_samples(
            _NIGHT
        )

        assert len(samples) == 2
        assert all(s.source_name == _WATCH for s in samples)

    async def test_explicit_source_name_overrides_preferred(
        self, async_db_session, profile_id
    ):
        """Explicit source_name returns that source's rows regardless of preferred_source."""
        await self._seed_two_source_night(
            async_db_session, profile_id, preferred=_WATCH
        )

        samples = await HealthService(async_db_session, profile_id).get_night_samples(
            _NIGHT, source_name=_IPHONE
        )

        assert len(samples) == 1
        assert samples[0].source_name == _IPHONE

    async def test_only_sleep_type_returned(self, async_db_session, profile_id):
        """Quantity record types (SpO2) are excluded even when source matches."""
        db = async_db_session
        db.add(
            HealthNightlySummary(
                profile_id=profile_id,
                night_date=_NIGHT,
                preferred_source=_WATCH,
                total_sleep_seconds=6 * 3600,
                computed_at=datetime.now(UTC),
            )
        )
        # Sleep stage row (should appear).
        db.add(
            HealthSample(
                profile_id=profile_id,
                record_type=_SLEEP_TYPE,
                source_name=_WATCH,
                start_time=_dt(1),
                end_time=_dt(2),
                value_text="AsleepCore",
                night_date=_NIGHT,
                ingest_channel="export_xml",
            )
        )
        # SpO2 quantity row (should NOT appear).
        db.add(
            HealthSample(
                profile_id=profile_id,
                record_type=_SPO2_TYPE,
                source_name=_WATCH,
                start_time=_dt(2),
                end_time=_dt(2),
                value_num=0.97,
                unit="%",
                night_date=_NIGHT,
                ingest_channel="export_xml",
            )
        )
        await db.flush()

        samples = await HealthService(db, profile_id).get_night_samples(_NIGHT)

        assert len(samples) == 1
        assert samples[0].record_type == _SLEEP_TYPE


# ---------------------------------------------------------------------------
# list_night_dates — sort order
# ---------------------------------------------------------------------------


class TestListNightDatesSorted:
    async def test_dates_returned_ascending(self, async_db_session, profile_id):
        """list_night_dates returns dates sorted oldest-first regardless of insert order."""
        db = async_db_session
        # Insert in reverse order.
        for d in [date(2024, 1, 16), date(2024, 1, 14), date(2024, 1, 15)]:
            db.add(
                HealthNightlySummary(
                    profile_id=profile_id,
                    night_date=d,
                    total_sleep_seconds=6 * 3600,
                    computed_at=datetime.now(UTC),
                )
            )
        await db.flush()

        dates = await HealthService(db, profile_id).list_night_dates()

        assert dates == [date(2024, 1, 14), date(2024, 1, 15), date(2024, 1, 16)]
        assert dates == sorted(dates)

    async def test_empty_profile_returns_empty_list(self, async_db_session, profile_id):
        """list_night_dates returns [] when there are no summaries for the profile."""
        dates = await HealthService(async_db_session, profile_id).list_night_dates()
        assert dates == []


# ---------------------------------------------------------------------------
# spo2_display_pct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (0.95, 95.0),  # fraction → percent
        (95.0, 95.0),  # already percent
        (0.5, 50.0),  # lower fraction boundary
        (1.0, 100.0),  # upper fraction boundary
        (50.0, 50.0),  # lower percent boundary
        (100.0, 100.0),  # upper percent boundary
        (20.0, 20.0),  # implausible: shown as stored
        (1.5, 1.5),  # implausible: shown as stored
    ],
)
def test_spo2_display_pct_scales_fractions_and_keeps_others(
    stored: float, expected: float
) -> None:
    assert spo2_display_pct(stored) == pytest.approx(expected)
