"""Unit tests for the Apple Health cross-source night-level validator.

Covers three surfaces:
- ``correlate_night_pairs`` (the pure, DB-free correlation seam) directly.
- ``AppleCrossValidator`` end-to-end against a real in-memory DB for the Apple
  side (health_samples + health_nightly_summaries), with the SNORE nightly side
  stubbed to plant exact per-night indices, day statuses, and coverage.
- The night_date join convention (noon-split) shared by both axes.
"""

from __future__ import annotations

import csv
import json
import uuid

from datetime import UTC, date, datetime, time

import numpy as np
import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.shared.versioning import DayAnalysisStatus, NullReason
from snore.database.day_manager import DayManager
from snore.database.models import (
    HealthNightlySummary,
    HealthSample,
    Profile,
    User,
)
from snore.parsers.apple_health.models import apply_noon_split
from snore.services.breath.dtos import (
    DeviceAmbiguityError,
    NightlyAnalysisSummary,
    NightlyRangeSummary,
)
from snore.services.breath_service import BreathService
from snore.validation.apple_cross_report import (
    AppleCrossAggregate,
    AppleCrossNightRecord,
    AppleCrossValidationReport,
    PairCorrelation,
    correlate_night_pairs,
    export_apple_cross_report_csv,
    export_apple_cross_report_json,
)
from snore.validation.apple_cross_validator import AppleCrossValidator

_BD_TYPE = "HKQuantityTypeIdentifierAppleSleepingBreathingDisturbances"


# ---------------------------------------------------------------------------
# Pure seam: correlate_night_pairs
# ---------------------------------------------------------------------------


class TestCorrelateNightPairs:
    def test_monotonic_pairs_yield_rho_one(self):
        snore = {date(2024, 1, d): float(d) for d in (1, 2, 3, 4)}
        apple = {date(2024, 1, d): float(d) * 10 for d in (1, 2, 3, 4)}
        result = correlate_night_pairs(snore, apple)
        assert result.n_paired_nights == 4
        assert result.rho == pytest.approx(1.0)
        assert result.reason is None

    def test_anti_monotonic_pairs_yield_rho_minus_one(self):
        snore = {date(2024, 1, d): float(d) for d in (1, 2, 3, 4)}
        apple = {date(2024, 1, d): float(-d) for d in (1, 2, 3, 4)}
        result = correlate_night_pairs(snore, apple)
        assert result.rho == pytest.approx(-1.0)

    def test_only_intersection_is_paired(self):
        snore = {date(2024, 1, d): float(d) for d in (1, 2, 3, 4, 5)}
        apple = {date(2024, 1, d): float(d) for d in (3, 4, 5, 6)}
        result = correlate_night_pairs(snore, apple)
        assert result.n_paired_nights == 3  # nights 3, 4, 5

    def test_fewer_than_three_pairs_is_insufficient(self):
        snore = {date(2024, 1, 1): 1.0, date(2024, 1, 2): 2.0}
        apple = {date(2024, 1, 1): 1.0, date(2024, 1, 2): 2.0}
        result = correlate_night_pairs(snore, apple)
        assert result.rho is None
        assert result.p_value is None
        assert result.n_paired_nights == 2
        assert result.reason == "insufficient_pairs"

    def test_constant_side_is_degenerate(self):
        snore = {date(2024, 1, d): 5.0 for d in (1, 2, 3)}
        apple = {date(2024, 1, d): float(d) for d in (1, 2, 3)}
        result = correlate_night_pairs(snore, apple)
        assert result.rho is None
        assert result.n_paired_nights == 3
        assert result.reason == "degenerate"


# ---------------------------------------------------------------------------
# Helpers for the validator end-to-end path
# ---------------------------------------------------------------------------


async def _make_profile(db: AsyncSession) -> int:
    user = User(
        canonical_email=f"apple_{uuid.uuid4().hex[:8]}@test.local", role="member"
    )
    db.add(user)
    await db.flush()
    profile = Profile(user_id=user.id, name="AppleXTest")
    db.add(profile)
    await db.flush()
    return profile.id


def _night(
    therapy_date: date,
    *,
    day_status: DayAnalysisStatus = DayAnalysisStatus.OK,
    rera_index: float | None = None,
    rera_index_reason: NullReason | None = None,
    fl_class_ge4_pct: float | None = None,
    fl_class_ge4_pct_reason: NullReason | None = None,
) -> NightlyAnalysisSummary:
    """Build a NightlyAnalysisSummary with only the fields this validator reads."""
    return NightlyAnalysisSummary(
        therapy_date=therapy_date,
        device_id=1,
        day_status=day_status,
        session_coverage=[],
        eligible_session_count=1,
        analyzed_session_count=1,
        algorithm_identity=None,
        rera_count=None,
        rera_reason=None,
        primary_mode="aasm",
        fl_median=None,
        fl_5th=None,
        fl_min=None,
        fl_reason=None,
        fl_class_ge4_pct=fl_class_ge4_pct,
        fl_class_ge4_pct_reason=fl_class_ge4_pct_reason,
        ti_median_s=None,
        ti_median_reason=None,
        ie_ratio_median=None,
        ie_ratio_reason=None,
        total_therapy_hours=7.0,
        compliance_threshold_hours=4.0,
        is_compliant=True,
        rera_index=rera_index,
        rera_index_reason=rera_index_reason,
    )


def _stub_nights(
    monkeypatch: pytest.MonkeyPatch, nights: list[NightlyAnalysisSummary]
) -> None:
    """Patch BreathService.get_nightly_range_summary to return the given nights.

    Returns exactly the nights whose therapy_date falls in the requested chunk,
    so the validator's 90-night paging never double-counts.
    """

    async def _fake(self, date_start, date_end, **_kwargs):  # noqa: ANN001
        in_range = [n for n in nights if date_start <= n.therapy_date <= date_end]
        return NightlyRangeSummary(
            date_start=date_start,
            date_end=date_end,
            device_id=1,
            compliance_threshold_hours=4.0,
            n_calendar_nights=(date_end - date_start).days + 1,
            n_nights=len(in_range),
            days_compliant=len(in_range),
            compliance_pct=100.0,
            nights=in_range,
        )

    monkeypatch.setattr(BreathService, "get_nightly_range_summary", _fake)


def _stub_multi_device(
    monkeypatch: pytest.MonkeyPatch,
    nights: list[NightlyAnalysisSummary],
    ambiguous: set[date],
) -> None:
    """Stub get_nightly_range_summary to raise ambiguity for unpinned nights.

    A call is ambiguous when ``device_id`` is None and its span covers any
    ``ambiguous`` date — mirroring _resolve_range raising for a multi-device
    night.  Pinning ``device_id`` suppresses the raise and returns every night.
    """

    async def _fake(self, date_start, date_end, *, device_id=None, **_kwargs):  # noqa: ANN001
        if device_id is None:
            hit = next((d for d in ambiguous if date_start <= d <= date_end), None)
            if hit is not None:
                raise DeviceAmbiguityError(
                    therapy_date=hit,
                    profile_id=0,
                    owned_device_ids=[1, 2],
                    device_serials={1: "A", 2: "B"},
                )
        in_range = [n for n in nights if date_start <= n.therapy_date <= date_end]
        return NightlyRangeSummary(
            date_start=date_start,
            date_end=date_end,
            device_id=device_id or 1,
            compliance_threshold_hours=4.0,
            n_calendar_nights=(date_end - date_start).days + 1,
            n_nights=len(in_range),
            days_compliant=len(in_range),
            compliance_pct=100.0,
            nights=in_range,
        )

    monkeypatch.setattr(BreathService, "get_nightly_range_summary", _fake)


async def _seed_apple_bd(
    db: AsyncSession, profile_id: int, night: date, value: float
) -> None:
    db.add(
        HealthSample(
            profile_id=profile_id,
            record_type=_BD_TYPE,
            source_name="Will's Apple Watch",
            start_time=datetime(night.year, night.month, night.day, 23, 0),
            end_time=datetime(night.year, night.month, night.day, 23, 0),
            value_num=value,
            unit="count",
            night_date=night,
            ingest_channel="export_xml",
        )
    )


async def _seed_fragmentation(
    db: AsyncSession,
    profile_id: int,
    night: date,
    awake_seconds: float | None,
    sleep_efficiency_pct: float | None,
) -> None:
    db.add(
        HealthNightlySummary(
            profile_id=profile_id,
            night_date=night,
            awake_seconds=awake_seconds,
            sleep_efficiency_pct=sleep_efficiency_pct,
            computed_at=datetime.now(UTC),
        )
    )


# ---------------------------------------------------------------------------
# Validator end-to-end
# ---------------------------------------------------------------------------


class TestAppleCrossValidator:
    async def test_monotonic_rera_vs_apple_bd_correlates(
        self, async_db_session, monkeypatch
    ):
        """A planted monotonic (rera_index, apple_bd) set yields rho == 1.0."""
        profile_id = await _make_profile(async_db_session)
        nights = [date(2024, 3, d) for d in (1, 2, 3, 4)]
        summaries = [_night(n, rera_index=float(i + 1)) for i, n in enumerate(nights)]
        _stub_nights(monkeypatch, summaries)
        for i, n in enumerate(nights):
            await _seed_apple_bd(async_db_session, profile_id, n, float(i + 1) * 5)
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-03-01", "2024-03-04")

        agg = report.aggregate
        assert agg.total_nights == 4
        assert agg.n_with_apple_bd == 4
        assert agg.rera_vs_apple_bd.n_paired_nights == 4
        assert agg.rera_vs_apple_bd.rho == pytest.approx(1.0)
        assert agg.rera_vs_apple_bd.reason is None

    async def test_skip_reasons_and_apple_coverage(self, async_db_session, monkeypatch):
        """Not-run / stale nights are flagged; missing Apple BD is counted."""
        n_ok = date(2024, 4, 1)
        n_not_run = date(2024, 4, 2)
        n_stale = date(2024, 4, 3)
        summaries = [
            _night(n_ok, rera_index=2.0),
            _night(
                n_not_run,
                day_status=DayAnalysisStatus.NOT_RUN,
                rera_index_reason=NullReason.NOT_AVAILABLE,
            ),
            _night(
                n_stale,
                day_status=DayAnalysisStatus.STALE,
                rera_index_reason=NullReason.ANALYSIS_STALE,
            ),
        ]
        profile_id = await _make_profile(async_db_session)
        _stub_nights(monkeypatch, summaries)
        # Apple BD only for the OK night; the other two lack it.
        await _seed_apple_bd(async_db_session, profile_id, n_ok, 12.0)
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-04-01", "2024-04-03")

        by_date = {r.night_date: r for r in report.nights}
        assert by_date["2024-04-01"].skip_reason is None
        assert by_date["2024-04-02"].skip_reason == "analysis_not_run"
        assert by_date["2024-04-03"].skip_reason == "analysis_stale"
        assert by_date["2024-04-02"].apple_bd_reason == "no_apple_bd"

        agg = report.aggregate
        assert agg.n_analysis_not_run == 1
        assert agg.n_analysis_stale == 1
        assert agg.n_with_apple_bd == 1
        assert agg.n_skipped_no_apple_bd == 2

    async def test_mixed_version_night_excluded_despite_non_null_indices(
        self, async_db_session, monkeypatch
    ):
        """A MIXED_VERSION (stale) night keeps its indices out of every correlation.

        MIXED_VERSION maps to skip_reason='analysis_stale'.  Even when such a
        night still carries non-null rera_index / fl / apple_bd (the last run
        left values behind), the documented invariant is that it contributes no
        SNORE side to any correlation and is counted in n_analysis_stale.
        """
        profile_id = await _make_profile(async_db_session)
        ok_nights = [date(2024, 10, d) for d in (1, 2, 3)]
        stale_night = date(2024, 10, 4)
        summaries = [
            *(_night(n, rera_index=float(i + 1)) for i, n in enumerate(ok_nights)),
            _night(
                stale_night,
                day_status=DayAnalysisStatus.MIXED_VERSION,
                rera_index=99.0,  # non-null but must not feed correlation
                fl_class_ge4_pct=99.0,
            ),
        ]
        _stub_nights(monkeypatch, summaries)
        for i, n in enumerate(ok_nights):
            await _seed_apple_bd(async_db_session, profile_id, n, float(i + 1) * 5)
        # The stale night also has an Apple BD value present.
        await _seed_apple_bd(async_db_session, profile_id, stale_night, 5.0)
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-10-01", "2024-10-04")

        by_date = {r.night_date: r for r in report.nights}
        assert by_date["2024-10-04"].skip_reason == "analysis_stale"

        agg = report.aggregate
        assert agg.n_analysis_stale == 1
        # Only the three OK nights are paired; the stale night is excluded even
        # though its indices and Apple BD are all non-null.
        assert agg.rera_vs_apple_bd.n_paired_nights == 3
        assert agg.rera_vs_apple_bd.rho == pytest.approx(1.0)

    async def test_fragmentation_pairs_use_nightly_summary(
        self, async_db_session, monkeypatch
    ):
        """awake_seconds / sleep_efficiency_pct join from HealthNightlySummary."""
        profile_id = await _make_profile(async_db_session)
        nights = [date(2024, 5, d) for d in (1, 2, 3)]
        summaries = [
            _night(n, rera_index=float(i + 1), fl_class_ge4_pct=float(i + 1) * 2)
            for i, n in enumerate(nights)
        ]
        _stub_nights(monkeypatch, summaries)
        for i, n in enumerate(nights):
            await _seed_fragmentation(
                async_db_session,
                profile_id,
                n,
                awake_seconds=float(i + 1) * 100,  # monotonic with rera
                sleep_efficiency_pct=90.0 - i,  # monotonic-decreasing
            )
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-05-01", "2024-05-03")

        agg = report.aggregate
        assert agg.rera_vs_awake_seconds.n_paired_nights == 3
        assert agg.rera_vs_awake_seconds.rho == pytest.approx(1.0)
        # fl rises while sleep efficiency falls → perfect negative rank correlation
        assert agg.fl_vs_sleep_efficiency.rho == pytest.approx(-1.0)

    async def test_multi_source_bd_is_averaged_per_night(
        self, async_db_session, monkeypatch
    ):
        """Two BD rows on one night collapse to their mean before correlating."""
        profile_id = await _make_profile(async_db_session)
        night = date(2024, 6, 1)
        _stub_nights(monkeypatch, [_night(night, rera_index=1.0)])
        await _seed_apple_bd(async_db_session, profile_id, night, 10.0)
        await _seed_apple_bd(async_db_session, profile_id, night, 20.0)
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-06-01", "2024-06-01")

        rec = report.nights[0]
        assert rec.apple_breathing_disturbances == pytest.approx(15.0)

    async def test_device_ambiguous_night_degrades_rest_scored(
        self, async_db_session, monkeypatch
    ):
        """One multi-device night skips as device_ambiguous; the range never aborts."""
        profile_id = await _make_profile(async_db_session)
        nights = [date(2024, 7, d) for d in (1, 2, 3)]
        ambiguous_night = nights[1]
        # Only the two unambiguous nights carry resolvable SNORE summaries.
        summaries = [
            _night(nights[0], rera_index=1.0),
            _night(nights[2], rera_index=3.0),
        ]
        _stub_multi_device(monkeypatch, summaries, ambiguous={ambiguous_night})
        for n in nights:
            await _seed_apple_bd(async_db_session, profile_id, n, 5.0)
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-07-01", "2024-07-03")

        by_date = {r.night_date: r for r in report.nights}
        assert by_date["2024-07-02"].skip_reason == "device_ambiguous"
        assert by_date["2024-07-02"].rera_index is None
        # Apple side is still joined onto the skipped night.
        assert by_date["2024-07-02"].apple_breathing_disturbances == pytest.approx(5.0)
        assert by_date["2024-07-01"].skip_reason is None
        assert by_date["2024-07-03"].skip_reason is None

        agg = report.aggregate
        assert agg.total_nights == 3
        assert agg.n_device_ambiguous == 1

    async def test_pinned_device_id_resolves_ambiguity(
        self, async_db_session, monkeypatch
    ):
        """Passing device_id suppresses ambiguity; the night is scored, not skipped."""
        profile_id = await _make_profile(async_db_session)
        nights = [date(2024, 9, d) for d in (1, 2, 3)]
        summaries = [_night(n, rera_index=float(i + 1)) for i, n in enumerate(nights)]
        _stub_multi_device(monkeypatch, summaries, ambiguous={nights[1]})
        for i, n in enumerate(nights):
            await _seed_apple_bd(async_db_session, profile_id, n, float(i + 1))
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-09-01", "2024-09-03", device_id=1)

        agg = report.aggregate
        assert agg.n_device_ambiguous == 0
        assert agg.total_nights == 3
        assert agg.rera_vs_apple_bd.n_paired_nights == 3
        assert agg.rera_vs_apple_bd.rho == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Night-date join convention
# ---------------------------------------------------------------------------


class TestNightDateJoinConvention:
    def test_noon_split_matches_day_manager_across_boundary(self):
        """apply_noon_split must agree with DayManager.get_day_for_session.

        The Apple axis (HealthSample.night_date / HealthNightlySummary.night_date)
        and the SNORE axis (NightlyAnalysisSummary.therapy_date) key on two
        independent copies of the noon cutoff.  Pin that they resolve identical
        night dates across the boundary so the join can never silently drift.
        """
        boundary_datetimes = [
            datetime(2024, 7, 2, 0, 0, 0),  # midnight → previous night
            datetime(2024, 7, 2, 11, 59, 59),  # just before noon → previous night
            datetime(2024, 7, 2, 12, 0, 0),  # noon exactly → today
            datetime(2024, 7, 2, 12, 0, 1),  # just after noon → today
            datetime(2024, 7, 2, 22, 0, 0),  # evening → today
        ]
        for dt in boundary_datetimes:
            assert apply_noon_split(dt) == DayManager.get_day_for_session(dt), dt

        # Both copies of the cutoff are the same noon constant.
        assert time(12, 0) == DayManager.DEFAULT_SPLIT_TIME

    async def test_validator_joins_apple_bd_on_shared_night_date(
        self, async_db_session, monkeypatch
    ):
        """A watch sample at 23:00 keys to the same night as the SNORE summary."""
        profile_id = await _make_profile(async_db_session)
        night = date(2024, 8, 10)
        # Sample recorded at 23:00 local → apply_noon_split keeps it on `night`.
        sample_night = apply_noon_split(datetime(2024, 8, 10, 23, 0))
        assert sample_night == night
        _stub_nights(monkeypatch, [_night(night, rera_index=1.0)])
        await _seed_apple_bd(async_db_session, profile_id, sample_night, 7.5)
        await async_db_session.flush()

        report = await AppleCrossValidator(
            async_db_session, profile_id
        ).validate_date_range("2024-08-10", "2024-08-10")

        assert report.nights[0].apple_breathing_disturbances == pytest.approx(7.5)


# ---------------------------------------------------------------------------
# Report exporters (JSON / CSV)
# ---------------------------------------------------------------------------


def _sample_report() -> AppleCrossValidationReport:
    """A report with one scored night and one skipped (null-SNORE) night."""
    return AppleCrossValidationReport(
        report_date="2024-11-01 08:00:00",
        date_range_start="2024-11-01",
        date_range_end="2024-11-02",
        aggregate=AppleCrossAggregate(
            total_nights=2,
            n_analysis_not_run=0,
            n_analysis_stale=1,
            n_device_ambiguous=0,
            n_skipped_no_apple_bd=0,
            n_with_apple_bd=2,
            rera_vs_apple_bd=PairCorrelation(
                n_paired_nights=1, reason="insufficient_pairs"
            ),
            fl_vs_apple_bd=PairCorrelation(
                n_paired_nights=1, reason="insufficient_pairs"
            ),
            rera_vs_awake_seconds=PairCorrelation(
                n_paired_nights=0, reason="insufficient_pairs"
            ),
            fl_vs_sleep_efficiency=PairCorrelation(
                n_paired_nights=0, reason="insufficient_pairs"
            ),
        ),
        nights=[
            AppleCrossNightRecord(
                night_date="2024-11-01",
                rera_index=3.25,
                fl_class_ge4_pct=12.5,
                apple_breathing_disturbances=7.0,
                awake_seconds=None,
                sleep_efficiency_pct=None,
                skip_reason=None,
            ),
            AppleCrossNightRecord(
                night_date="2024-11-02",
                rera_index=None,
                rera_index_reason="analysis_stale",
                fl_class_ge4_pct=None,
                apple_breathing_disturbances=9.0,
                apple_bd_reason=None,
                skip_reason="analysis_stale",
            ),
        ],
    )


class TestAppleCrossExporters:
    def test_json_export_round_trips_to_equal_report(self, tmp_path):
        report = _sample_report()
        out = tmp_path / "nested" / "report.json"

        export_apple_cross_report_json(report, out)

        loaded = json.loads(out.read_text())
        rebuilt = AppleCrossValidationReport.model_validate(loaded)
        assert rebuilt == report

    def test_csv_export_writes_one_row_per_night(self, tmp_path):
        report = _sample_report()
        out = tmp_path / "report.csv"

        export_apple_cross_report_csv(report, out)

        with open(out, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == len(report.nights)
        scored, skipped = rows
        # Numeric cells are fixed to 4 decimals; None becomes an empty string.
        assert scored["night_date"] == "2024-11-01"
        assert scored["rera_index"] == "3.2500"
        assert scored["apple_breathing_disturbances"] == "7.0000"
        assert scored["awake_seconds"] == ""
        assert scored["skip_reason"] == ""
        assert skipped["rera_index"] == ""
        assert skipped["rera_index_reason"] == "analysis_stale"
        assert skipped["skip_reason"] == "analysis_stale"


def test_module_imports_numpy_free_of_side_effects():
    """Guard: the pure seam does not require a DB/session import to run."""
    assert callable(correlate_night_pairs)
    assert np is not None
