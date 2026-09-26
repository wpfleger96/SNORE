"""Unit tests for BreathService seams (plan v3.8 — all seam unit-tested against
fixture data, independent of MCP).

Coverage:
- get_analysis_status: NOT_RUN, OK, STALE_VERSION
- get_breath_table: empty → NOT_RUN, paginated BreathRow, time-binned BreathBin
- find_windows: empty → NOT_RUN result, WORST_FLATTENING_LEAK_VALID criterion
- compare_epochs: empty range → NO_DATA_IN_RANGE + null algorithm_identity;
                  populated → nights_with_data > 0
- get_nightly_summary: ValueError no sessions, OK day with analyzed_session_count
- get_nightly_range_summary: empty range returns n_nights=0
- get_device_capabilities: no sessions → null_reason; with sessions → date range
- get_contextual_events: no events → empty; with events → ContextualEvent list
- get_ca_analysis: no CA events → empty; with CA events → CaAnalysisResult
- get_waveform_window: no waveform data → empty channels
- fetch_waveform_window_raw (module-level): missing session raises ValueError
- compute_waveform_window (module-level): empty raw data round-trip
- A-suite rework: A1/A2/A3 drive store_result() production path
- vendor_applicability: non-ResMed device → unvalidated_device in breath rows
- deletion cascade: Breath rows deleted when AnalysisResult parent is deleted
- missing breaths table: actionable RuntimeError on no such table
"""

from __future__ import annotations

import uuid

from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.modes.types import ModeResult
from snore.analysis.service import AnalysisService
from snore.analysis.shared.versioning import (
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisRunMetadata,
    AnalysisStatus,
    NullReason,
)
from snore.analysis.types import AnalysisComputation
from snore.analysis.types import AnalysisResult as AnalysisResultDTO
from snore.constants import FlowLimitationConstants as FLC
from snore.database import models
from snore.services.breath_service import (
    BreathQueryRange,
    BreathService,
    DayAnalysisStatus,
    EpochRequest,
    WaveformChannelName,
    WaveformWindowRequest,
    WindowCriterion,
    compute_waveform_window,
    derive_mv_from_flow,
    fetch_waveform_window_raw,
)

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


async def _make_profile(db: AsyncSession) -> tuple[int, int]:
    """Return (user_id, profile_id)."""
    user = models.User(
        canonical_email=f"svc_{uuid.uuid4().hex[:8]}@test.example",
        role="admin",
    )
    db.add(user)
    await db.flush()
    profile = models.Profile(user_id=user.id, name="SeamTest")
    db.add(profile)
    await db.flush()
    return user.id, profile.id


async def _make_device(
    db: AsyncSession,
    profile_id: int,
    manufacturer: str = "ResMed",
) -> models.Device:
    device = models.Device(
        profile_id=profile_id,
        serial_number=f"SN_{uuid.uuid4().hex[:6]}",
        manufacturer=manufacturer,
        model="AirCurve 11 VAuto",
        firmware_version="1.0",
    )
    db.add(device)
    await db.flush()
    return device


async def _make_day_and_session(
    db: AsyncSession,
    device_id: int,
    therapy_date: date,
    *,
    duration_hours: float = 7.0,
) -> tuple[models.Day, models.Session]:
    start_dt = datetime(therapy_date.year, therapy_date.month, therapy_date.day, 22, 0)
    day = models.Day(device_id=device_id, date=therapy_date, session_count=1)
    db.add(day)
    await db.flush()
    session = models.Session(
        device_id=device_id,
        day_id=day.id,
        device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
        start_time=start_dt,
        end_time=start_dt + timedelta(hours=duration_hours),
        duration_seconds=duration_hours * 3600.0,
    )
    db.add(session)
    await db.flush()
    return day, session


def _make_algo_versions(primary_mode: str = "aasm") -> AlgoVersions:
    identity = AlgorithmIdentity.current()
    run_meta = AnalysisRunMetadata(primary_mode=primary_mode, modes=[primary_mode])
    return AlgoVersions(identity=identity, run=run_meta)


async def _store_analysis_with_breaths(
    db: AsyncSession,
    session: models.Session,
    profile_id: int,
    n_breaths: int = 5,
    flow_class: int | None = 1,
    is_recovery: bool = False,
    flow_specs: list[tuple[int | None, float | None]] | None = None,
) -> models.AnalysisResult:
    """Write an AnalysisResult + Breath rows via AnalysisService.store_result.

    ``flow_specs`` overrides the uniform (flow_class, flow_confidence) with an
    explicit per-breath list, used to exercise the rule-matched / fallback split.
    """
    if flow_specs is not None:
        n_breaths = len(flow_specs)
    result_dto = AnalysisResultDTO(
        session_id=session.id,
        session_duration_hours=session.duration_seconds / 3600.0
        if session.duration_seconds
        else 7.0,
        total_breaths=n_breaths,
        machine_events=[],
        mode_results={
            "aasm": ModeResult(
                mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
            )
        },
        timestamp_start=session.start_time.timestamp(),
        timestamp_end=session.end_time.timestamp()
        if session.end_time
        else (session.start_time + timedelta(hours=7)).timestamp(),
    )

    from snore.analysis.types import ComputedBreath  # noqa: PLC0415

    breaths = [
        ComputedBreath(
            breath_number=i + 1,
            start_offset_s=float(i * 4),
            end_offset_s=float(i * 4 + 3),
            inspiration_time_s=1.2,
            expiration_time_s=1.8,
            total_time_s=3.0,
            i_e_ratio=0.67,
            duty_cycle=0.4,
            peak_flow_lpm=30.0,
            peak_exp_flow_lpm=20.0,
            tidal_volume_ml=400.0,
            respiratory_rate_rolling=15.0,
            flatness_index=0.2,
            mid_insp_flattening=0.35,
            flow_class=flow_specs[i][0] if flow_specs is not None else flow_class,
            flow_confidence=flow_specs[i][1] if flow_specs is not None else 0.9,
            is_recovery_breath=is_recovery if i == n_breaths - 1 else False,
            inferred_trigger_type="normal",
            trigger_confidence=0.8,
            inferred_cycle_type="normal",
            cycle_confidence=0.75,
            trigger_cycle_applicable=True,
            trigger_cycle_reason=None,
            leak_valid=True,
            leak_valid_reason=None,
            ramp_active=None,
            ramp_active_reason="not_available",
            mask_off=False,
            mask_off_reason=None,
        )
        for i in range(n_breaths)
    ]

    computation = AnalysisComputation(
        summary=result_dto, breaths=breaths, primary_mode="aasm"
    )
    svc = AnalysisService(db, profile_id=profile_id)
    await svc.store_result(computation, processing_time_ms=42)
    await db.flush()
    # Re-query to get the committed row
    ar = (
        (
            await db.execute(
                select(models.AnalysisResult)
                .where(models.AnalysisResult.session_id == session.id)
                .order_by(models.AnalysisResult.id.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    assert ar is not None
    return ar


# ---------------------------------------------------------------------------
# get_analysis_status
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetAnalysisStatus:
    async def test_not_run_when_no_analysis_result(self, async_db_session):
        """get_analysis_status returns NOT_RUN when no AnalysisResult exists."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 1, 10)
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        status, algo = await svc.get_analysis_status(session.id)

        assert status == AnalysisStatus.NOT_RUN
        assert algo is None

    async def test_ok_when_current_version_result_exists(self, async_db_session):
        """get_analysis_status returns OK when latest result matches current identity."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 1, 10)
        )
        await _store_analysis_with_breaths(async_db_session, session, profile_id)

        svc = BreathService(async_db_session, profile_id=profile_id)
        status, algo = await svc.get_analysis_status(session.id)

        assert status == AnalysisStatus.OK
        assert algo is not None

    async def test_stale_version_when_engine_versions_missing(self, async_db_session):
        """get_analysis_status returns STALE_VERSION for a legacy flat engine_versions_json."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 1, 10)
        )

        # Insert a bare (legacy) AnalysisResult with no "identity" key
        ar = models.AnalysisResult(
            session_id=session.id,
            timestamp_start=session.start_time,
            timestamp_end=session.end_time or session.start_time + timedelta(hours=7),
            programmatic_result_json={},
            processing_time_ms=10,
            engine_versions_json={"version": "0.1.0"},  # legacy flat shape
        )
        async_db_session.add(ar)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        status, algo = await svc.get_analysis_status(session.id)

        assert status == AnalysisStatus.STALE_VERSION

    async def test_stale_version_for_format_version_3_identity(self, async_db_session):
        """A stored identity from before the validity-flags bump classifies STALE.

        Legacy format_version=3 rows lack the ``validity_flags`` field; pydantic
        back-fills it from the default, so the ``format_version`` bump to 4 is
        what makes the comparison fail — this test pins that behavior.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 1, 10)
        )

        stored = _make_algo_versions().model_dump()
        stored["identity"]["format_version"] = 3
        del stored["identity"]["validity_flags"]  # legacy rows never stored it
        ar = models.AnalysisResult(
            session_id=session.id,
            timestamp_start=session.start_time,
            timestamp_end=session.end_time or session.start_time + timedelta(hours=7),
            programmatic_result_json={},
            processing_time_ms=10,
            engine_versions_json=stored,
        )
        async_db_session.add(ar)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        status, algo = await svc.get_analysis_status(session.id)

        assert status == AnalysisStatus.STALE_VERSION
        assert algo is not None  # the row parses; it is stale, not corrupt


# ---------------------------------------------------------------------------
# get_breath_table
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetBreathTable:
    async def test_not_run_returns_not_run_status(self, async_db_session):
        """get_breath_table returns AnalysisStatus.NOT_RUN when no analysis exists."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 1, 10)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        query = BreathQueryRange(
            therapy_date=therapy_date,
            device_id=dev.id,
            offset_start=0.0,
            offset_end=900.0,
        )
        page = await svc.get_breath_table(query)

        assert page.analysis_status == AnalysisStatus.NOT_RUN
        assert page.rows == []
        assert page.bins == []

    async def test_paginated_breath_rows_returned_when_analysis_exists(
        self, async_db_session
    ):
        """get_breath_table returns BreathRow list for a short window."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 1, 10)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=10
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        # Window spanning first ~15s of session — under 15-min raw cap
        query = BreathQueryRange(
            therapy_date=therapy_date,
            device_id=dev.id,
            offset_start=0.0,
            offset_end=60.0,
        )
        page = await svc.get_breath_table(query)

        assert page.analysis_status == AnalysisStatus.OK
        # Some breath rows should be in the window
        assert len(page.rows) > 0 or len(page.bins) > 0

    async def test_no_data_in_range_when_window_outside_session(self, async_db_session):
        """get_breath_table returns empty rows for an out-of-range window."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 1, 10)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=3
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        # Window far beyond session end — use bin_minutes to bypass 15-min raw cap
        query = BreathQueryRange(
            therapy_date=therapy_date,
            device_id=dev.id,
            offset_start=9000.0,
            offset_end=9900.0,
            bin_minutes=15.0,
        )
        page = await svc.get_breath_table(query)

        # Either empty rows, empty bins, or null_reason set
        assert page.rows == [] or page.null_reason == NullReason.NO_DATA_IN_RANGE


# ---------------------------------------------------------------------------
# BreathQueryRange validation — non-finite offsets (DoS guard)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBreathQueryRangeValidation:
    def test_infinite_offset_end_with_bin_minutes_is_rejected(self):
        """offset_end=inf with bin_minutes set is rejected before service call.

        Without this guard the binned loop (while bin_start < offset_end) becomes
        infinite once float addition saturates.
        """
        with pytest.raises(ValueError, match="finite"):
            BreathQueryRange(
                therapy_date=date(2025, 1, 1),
                offset_start=0.0,
                offset_end=float("inf"),
                bin_minutes=5.0,
            )

    def test_infinite_offset_start_is_rejected(self):
        """offset_start=inf is rejected by validate_window."""
        with pytest.raises(ValueError, match="finite"):
            BreathQueryRange(
                therapy_date=date(2025, 1, 1),
                offset_start=float("inf"),
                offset_end=float("inf"),
                bin_minutes=5.0,
            )

    def test_nan_offset_end_is_rejected(self):
        """NaN offset_end is rejected (caught by field gt=0 constraint before validate_window)."""
        with pytest.raises(ValueError):
            BreathQueryRange(
                therapy_date=date(2025, 1, 1),
                offset_start=0.0,
                offset_end=float("nan"),
                bin_minutes=5.0,
            )


# ---------------------------------------------------------------------------
# find_windows
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindWindows:
    async def test_not_run_result_when_no_analysis(self, async_db_session):
        """find_windows returns NOT_RUN day_status when no analysis exists."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 1, 10)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date=therapy_date,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=3,
            device_id=dev.id,
        )

        assert result.windows == []
        from snore.services.breath_service import DayAnalysisStatus  # noqa: PLC0415

        # plan §1 line 864 rule 3: all-not-run → NOT_RUN
        assert result.day_status == DayAnalysisStatus.NOT_RUN

    async def test_worst_flattening_criterion_returns_result(self, async_db_session):
        """find_windows returns a FindWindowsResult for WORST_FLATTENING_LEAK_VALID."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 1, 10)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=20, flow_class=3
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date=therapy_date,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=3,
            device_id=dev.id,
        )

        from snore.services.breath_service import DayAnalysisStatus  # noqa: PLC0415

        # plan §1 line 864 rule 2: all-OK → OK
        assert result.day_status == DayAnalysisStatus.OK
        assert len(result.windows) > 0
        assert result.windows[0].worst_mid_insp_flattening == pytest.approx(0.35)

    async def test_worst_flattening_none_flattening_breaths_skipped(
        self, async_db_session
    ):
        """Breaths with mid_insp_flattening=None are never selected as anchors.

        Real fragment breaths skip the classifier and arrive with
        mid_insp_flattening=None. With default options (no threshold), the
        eligibility filter must exclude them unconditionally so the f-string
        format in reason_summary never sees a None value.
        """
        from snore.analysis.types import ComputedBreath  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 2, 1)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)

        # Build a mix: two breaths with real flattening values, one with None.
        base_breath = {
            "breath_number": 0,
            "start_offset_s": 0.0,
            "end_offset_s": 3.0,
            "inspiration_time_s": 1.2,
            "expiration_time_s": 1.8,
            "total_time_s": 3.0,
            "i_e_ratio": 0.67,
            "duty_cycle": 0.4,
            "peak_flow_lpm": 30.0,
            "peak_exp_flow_lpm": 20.0,
            "tidal_volume_ml": 400.0,
            "respiratory_rate_rolling": 15.0,
            "flatness_index": 0.2,
            "flow_class": 3,
            "flow_confidence": 0.9,
            "is_recovery_breath": False,
            "inferred_trigger_type": "normal",
            "trigger_confidence": 0.8,
            "inferred_cycle_type": "normal",
            "cycle_confidence": 0.75,
            "trigger_cycle_applicable": True,
            "trigger_cycle_reason": None,
            "leak_valid": True,
            "leak_valid_reason": None,
            "ramp_active": None,
            "ramp_active_reason": "not_available",
            "mask_off": False,
            "mask_off_reason": None,
        }

        breaths = []
        for i, mid_fl in enumerate([0.8, None, 0.6, 0.7, 0.5, 0.4, 0.3]):
            b = ComputedBreath(
                **{
                    **base_breath,
                    "breath_number": i + 1,
                    "mid_insp_flattening": mid_fl,
                    "start_offset_s": float(i * 4),
                    "end_offset_s": float(i * 4 + 3),
                }
            )
            breaths.append(b)

        from snore.analysis.modes.types import ModeResult  # noqa: PLC0415
        from snore.analysis.service import AnalysisService  # noqa: PLC0415
        from snore.analysis.types import AnalysisComputation  # noqa: PLC0415
        from snore.analysis.types import (
            AnalysisResult as AnalysisResultDTO,  # noqa: PLC0415
        )

        result_dto = AnalysisResultDTO(
            session_id=session.id,
            session_duration_hours=7.0,
            total_breaths=len(breaths),
            machine_events=[],
            mode_results={
                "aasm": ModeResult(
                    mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                )
            },
            timestamp_start=session.start_time.timestamp(),
            timestamp_end=(session.end_time or session.start_time).timestamp(),
        )
        computation = AnalysisComputation(
            summary=result_dto, breaths=breaths, primary_mode="aasm"
        )
        await AnalysisService(async_db_session, profile_id=profile_id).store_result(
            computation, processing_time_ms=10
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date=therapy_date,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=5,
            device_id=dev.id,
        )

        assert result.day_status == DayAnalysisStatus.OK
        assert len(result.windows) > 0
        for w in result.windows:
            assert w.worst_mid_insp_flattening is not None

    async def test_single_primary_mode_session_reports_primary_mode(
        self, async_db_session
    ):
        """find_windows populates primary_mode when all sessions share one primary mode.

        A day with a single WORST_FLATTENING_LEAK_VALID session analysed with
        primary_mode='aasm' must set result.primary_mode == 'aasm'.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 2, 10)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=10
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date=therapy_date,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=3,
            device_id=dev.id,
        )

        assert result.day_status == DayAnalysisStatus.OK
        assert result.primary_mode == "aasm"

    async def test_mixed_primary_modes_yields_none_primary_mode(self, async_db_session):
        """find_windows returns primary_mode=None when sessions differ in primary mode.

        WORST_FLATTENING_LEAK_VALID does not refuse on primary_mode mismatch
        (only FL_RUN_ENDING_IN_RECOVERY does), so windows may still be returned,
        but result.primary_mode must be None.
        """
        from snore.analysis.modes.types import ModeResult  # noqa: PLC0415
        from snore.analysis.service import AnalysisService  # noqa: PLC0415
        from snore.analysis.types import AnalysisComputation  # noqa: PLC0415
        from snore.analysis.types import (
            AnalysisResult as AnalysisResultDTO,  # noqa: PLC0415
        )

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 2, 11)

        start_a = datetime(
            therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
        )
        day = models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        async_db_session.add(day)
        await async_db_session.flush()

        session_a = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a,
            end_time=start_a + timedelta(hours=3),
            duration_seconds=3 * 3600.0,
        )
        session_b = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a + timedelta(hours=4),
            end_time=start_a + timedelta(hours=7),
            duration_seconds=3 * 3600.0,
        )
        async_db_session.add_all([session_a, session_b])
        await async_db_session.flush()

        # Store analysis for each session with a different primary_mode
        for sess, pm in [(session_a, "aasm"), (session_b, "aasm_relaxed")]:
            result_dto = AnalysisResultDTO(
                session_id=sess.id,
                session_duration_hours=3.0,
                total_breaths=10,
                machine_events=[],
                mode_results={
                    pm: ModeResult(
                        mode_name=pm, apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                    )
                },
                timestamp_start=sess.start_time.timestamp(),
                timestamp_end=(sess.end_time or sess.start_time).timestamp(),
            )
            computation = AnalysisComputation(
                summary=result_dto, breaths=[], primary_mode=pm
            )
            await AnalysisService(async_db_session, profile_id=profile_id).store_result(
                computation, processing_time_ms=10
            )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date=therapy_date,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=3,
            device_id=dev.id,
        )

        # Mixed primary modes → primary_mode=None; no early-return for WORST_FLATTENING
        assert result.primary_mode is None
        assert result.day_status != DayAnalysisStatus.NOT_RUN


# ---------------------------------------------------------------------------
# compare_epochs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompareEpochs:
    async def test_empty_range_yields_no_data_in_range_and_null_identity(
        self, async_db_session
    ):
        """compare_epochs with no sessions → null algorithm_identity + NO_DATA_IN_RANGE."""
        _, profile_id = await _make_profile(async_db_session)
        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.compare_epochs(
            epochs=[
                EpochRequest(
                    label="empty",
                    date_start=date(2025, 1, 1),
                    date_end=date(2025, 1, 7),
                )
            ]
        )

        assert len(result.epochs) == 1
        es = result.epochs[0]
        assert es.algorithm_identity is None
        assert es.null_reason == NullReason.NO_DATA_IN_RANGE
        assert es.nights_with_data == 0

    async def test_analyzed_session_count_positive_when_analysis_exists(
        self, async_db_session
    ):
        """compare_epochs sets nights_with_data > 0 when a session has an OK analysis."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 2, 1)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(async_db_session, session, profile_id)

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.compare_epochs(
            epochs=[
                EpochRequest(
                    label="feb",
                    date_start=therapy_date,
                    date_end=therapy_date,
                    device_id=dev.id,
                )
            ]
        )

        assert len(result.epochs) == 1
        es = result.epochs[0]
        assert es.nights_with_data > 0
        assert es.algorithm_identity is not None
        assert es.null_reason is None

    async def test_mixed_confidence_splits_rule_matched_from_fallback_guesses(
        self, async_db_session
    ):
        """flow_class_distribution counts only rule-matched breaths (confidence >
        FL_DEFAULT_CONFIDENCE); fallback guesses stamped at exactly the default
        confidence land in flow_class_distribution_fallback.  Missing and
        below-default confidence values are excluded from both.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 4, 1)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)

        default_conf = FLC.FL_DEFAULT_CONFIDENCE
        # 3 rule-matched (confidence above the gate): classes 3, 4, 4.
        # 2 fallback guesses (confidence exactly at the gate): classes 1, 4.
        await _store_analysis_with_breaths(
            async_db_session,
            session,
            profile_id,
            flow_specs=[
                (3, 0.9),
                (4, 0.8),
                (4, 0.75),
                (1, default_conf),
                (4, default_conf),
                (7, None),
                (6, 0.4),
            ],
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.compare_epochs(
            epochs=[
                EpochRequest(
                    label="apr",
                    date_start=therapy_date,
                    date_end=therapy_date,
                    device_id=dev.id,
                )
            ]
        )

        es = result.epochs[0]
        assert es.null_reason is None
        assert es.flow_class_distribution == {3: 1, 4: 2}
        assert es.flow_class_distribution_fallback == {1: 1, 4: 1}
        # Only the five classifier-canonical rows land in a distribution.  Missing
        # and below-default confidence cannot be identified as fallback guesses.
        rule_total = sum(es.flow_class_distribution.values())
        fallback_total = sum(es.flow_class_distribution_fallback.values())
        assert rule_total + fallback_total == 5


# ---------------------------------------------------------------------------
# get_nightly_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetNightlySummary:
    async def test_not_run_when_no_sessions_for_date(self, async_db_session):
        """get_nightly_summary raises ValueError when no sessions exist for date."""
        _, profile_id = await _make_profile(async_db_session)
        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError, match="No sessions found"):
            await svc.get_nightly_summary(date(2025, 3, 1))

    async def test_ok_day_returns_analyzed_session_count(self, async_db_session):
        """get_nightly_summary returns analyzed_session_count >= 1 for an analyzed day."""
        from snore.services.breath_service import DayAnalysisStatus  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 5)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=10
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_summary(therapy_date, device_id=dev.id)

        assert summary.therapy_date == therapy_date
        assert summary.device_id == dev.id
        assert summary.day_status == DayAnalysisStatus.OK
        assert summary.analyzed_session_count >= 1

    async def test_not_run_day_returns_not_run_status(self, async_db_session):
        """get_nightly_summary returns NOT_RUN when session has no analysis."""
        from snore.services.breath_service import DayAnalysisStatus  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 6)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_summary(therapy_date, device_id=dev.id)

        # plan §1 line 864 rule 3: all-not-run → NOT_RUN
        assert summary.day_status == DayAnalysisStatus.NOT_RUN
        assert summary.analyzed_session_count == 0
        assert summary.fl_reason == NullReason.NOT_AVAILABLE

    async def test_waveform_opt_out_preserves_fl_rera_fields(self, async_db_session):
        """``include_waveform_stats=False`` leaves the breath-derived FL/RERA
        fields byte-identical; only the waveform ``device_flg_*`` fields differ.

        Pins the day-detail hot-path optimization: skipping the waveform blobs
        must not perturb the metrics day detail actually consumes.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 12)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        # Rule-matched high-confidence class-5 breaths → non-zero
        # fl_class_ge4_pct, so the equality check compares a real value.
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=8, flow_class=5
        )
        # A real fl waveform blob so the with-stats path aggregates device_flg.
        samples = np.array([[float(i), 0.4] for i in range(10)], dtype=np.float32)
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="fl",
                sample_rate=1.0,
                sample_count=10,
                data_blob=samples.tobytes(),
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        with_stats = await svc.get_nightly_summary(
            therapy_date, device_id=dev.id, include_waveform_stats=True
        )
        without_stats = await svc.get_nightly_summary(
            therapy_date, device_id=dev.id, include_waveform_stats=False
        )

        assert with_stats.fl_class_ge4_pct is not None
        assert with_stats.fl_class_ge4_pct == without_stats.fl_class_ge4_pct
        assert (
            with_stats.fl_class_ge4_pct_reason == without_stats.fl_class_ge4_pct_reason
        )
        assert with_stats.rera_index == without_stats.rera_index
        assert with_stats.rera_index_reason == without_stats.rera_index_reason
        assert with_stats.rera_count == without_stats.rera_count
        assert with_stats.rera_reason == without_stats.rera_reason

        # The opt-out actually skipped waveform aggregation.
        assert with_stats.device_flg_median is not None
        assert without_stats.device_flg_median is None


# ---------------------------------------------------------------------------
# get_nightly_range_summary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetNightlyRangeSummary:
    async def test_empty_range_returns_zero_nights(self, async_db_session):
        """get_nightly_range_summary over a range with no sessions returns n_nights=0."""
        _, profile_id = await _make_profile(async_db_session)
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=date(2025, 4, 1),
            date_end=date(2025, 4, 3),
        )

        assert summary.n_nights == 0
        assert summary.nights == []

    async def test_range_with_one_session_returns_one_night(self, async_db_session):
        """get_nightly_range_summary over a range with one session returns n_nights=1."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 4, 2)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=date(2025, 4, 1),
            date_end=date(2025, 4, 3),
            device_id=dev.id,
        )

        assert summary.n_nights == 1


# ---------------------------------------------------------------------------
# get_device_capabilities
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetDeviceCapabilities:
    async def test_no_sessions_returns_null_reason(self, async_db_session):
        """get_device_capabilities returns a null_reason when no sessions exist."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        svc = BreathService(async_db_session, profile_id=profile_id)
        caps = await svc.get_device_capabilities(device_id=dev.id)

        assert caps.device_id == dev.id
        assert caps.null_reason is not None  # NO_DATA_IN_RANGE or similar

    async def test_with_sessions_returns_date_range(self, async_db_session):
        """get_device_capabilities returns actual_date_start/end when sessions exist."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 5, 1)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        caps = await svc.get_device_capabilities(device_id=dev.id)

        assert caps.device_id == dev.id
        assert caps.actual_date_start == therapy_date
        assert caps.actual_date_end == therapy_date

    async def test_day_with_no_sessions_returns_no_data_in_range(
        self, async_db_session
    ):
        """get_device_capabilities returns NO_DATA_IN_RANGE for a Day row with no sessions.

        Empty Day cache rows (session_count=0, no Session children) must not inflate
        nights_with_data or produce a non-null actual date range.  DayManager now
        prunes such rows, so this models legacy or hand-edited data.
        plan §13 lines 949-961: actual endpoints derived from dates with ≥1 session.
        """
        from snore.database import models as _m  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        # Create a Day row with session_count=0 and NO Session children
        empty_day = _m.Day(
            device_id=dev.id,
            date=date(2025, 6, 1),
            session_count=0,
        )
        async_db_session.add(empty_day)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        caps = await svc.get_device_capabilities(device_id=dev.id)

        # Day row with no sessions must NOT count as imported data
        assert caps.null_reason == NullReason.NO_DATA_IN_RANGE
        assert caps.actual_date_start is None
        assert caps.actual_date_end is None
        assert caps.nights_with_data == 0


# ---------------------------------------------------------------------------
# get_contextual_events
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetContextualEvents:
    async def test_empty_when_no_events(self, async_db_session):
        """get_contextual_events returns [] when no machine events exist."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 5, 10)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        events = await svc.get_contextual_events(
            therapy_date=therapy_date, device_id=dev.id
        )

        assert events == []

    async def test_returns_event_with_context_fields(self, async_db_session):
        """get_contextual_events returns ContextualEvent list for sessions with events."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 5, 10)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        # Insert a machine event
        event = models.Event(
            session_id=session.id,
            event_type="OA",
            start_time=session.start_time + timedelta(minutes=30),
            duration_seconds=10.0,
        )
        async_db_session.add(event)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        events = await svc.get_contextual_events(
            therapy_date=therapy_date, device_id=dev.id
        )

        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "OA"
        assert ev.duration_seconds == 10.0


# ---------------------------------------------------------------------------
# get_ca_analysis
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetCaAnalysis:
    async def test_empty_when_no_ca_events(self, async_db_session):
        """get_ca_analysis returns empty ca_events when no CA events exist."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 6, 1)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(async_db_session, session, profile_id)

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert result.ca_events == []

    async def test_returns_ca_events_when_ca_present(self, async_db_session):
        """get_ca_analysis populates ca_events list when CA events are present."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 6, 1)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(async_db_session, session, profile_id)

        ca_event = models.Event(
            session_id=session.id,
            event_type="CA",
            start_time=session.start_time + timedelta(minutes=10),
            duration_seconds=20.0,
        )
        async_db_session.add(ca_event)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert len(result.ca_events) == 1
        assert result.ca_events[0].duration_seconds == pytest.approx(20.0)
        assert result.ca_events[0].session_id == session.id
        # periodic_breathing_pct is a float or None
        assert result.periodic_breathing_pct is None or isinstance(
            result.periodic_breathing_pct, float
        )


# ---------------------------------------------------------------------------
# get_waveform_window
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetWaveformWindow:
    async def test_no_waveform_data_returns_empty_channels(self, async_db_session):
        """get_waveform_window returns empty channels when no waveform blobs exist."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 6, 10)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        req = WaveformWindowRequest(
            therapy_date=therapy_date,
            session_id=session.id,
            offset_start=0.0,
            offset_end=120.0,
        )
        window = await svc.get_waveform_window(req)

        # No waveform blobs stored → empty channels
        assert window.channels == []

    async def test_explicit_session_id_on_empty_date_raises(self, async_db_session):
        """get_waveform_window raises ValueError for explicit session_id on an empty date.

        plan §9 lines 822-825: a non-None session_id must raise if it doesn't belong
        to the requested date/device.  An owned device with no sessions on the
        requested date and an explicit session_id must NOT return a synthetic empty window.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        # No sessions created for this date — empty day
        therapy_date = date(2025, 6, 11)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError):
            await svc.get_waveform_window(
                WaveformWindowRequest(
                    therapy_date=therapy_date,
                    device_id=dev.id,
                    session_id=99999,  # non-None, bogus
                    offset_start=0.0,
                    offset_end=60.0,
                )
            )

    async def test_no_session_id_on_empty_date_returns_empty_window(
        self, async_db_session
    ):
        """get_waveform_window with session_id=None on empty date returns empty window.

        Regression guard: the session_id validation fix must not raise when session_id
        is None — only the synthetic empty window is returned.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 6, 12)

        svc = BreathService(async_db_session, profile_id=profile_id)
        window = await svc.get_waveform_window(
            WaveformWindowRequest(
                therapy_date=therapy_date,
                device_id=dev.id,
                # session_id=None (default)
                offset_start=0.0,
                offset_end=60.0,
            )
        )
        assert window.session_id == 0
        assert window.channels == []


# ---------------------------------------------------------------------------
# fetch_waveform_window_raw (module-level)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchWaveformWindowRaw:
    async def test_foreign_session_id_raises_value_error(self, async_db_session):
        """fetch_waveform_window_raw raises ValueError for a foreign session_id.

        plan §9 lines 720-735: the public seam must verify Device.profile_id and
        raise ValueError when the session is not owned by profile_id.
        """
        _, profile_a_id = await _make_profile(async_db_session)
        _, profile_b_id = await _make_profile(async_db_session)
        dev_b = await _make_device(async_db_session, profile_b_id)
        therapy_date = date(2025, 2, 1)
        _, session_b = await _make_day_and_session(
            async_db_session, dev_b.id, therapy_date
        )
        # Give session_b a waveform row so the difference from "no data" is detectable
        async_db_session.add(
            models.Waveform(
                session_id=session_b.id,
                waveform_type="flow",
                sample_rate=1.0,
                sample_count=10,
                data_blob=b"\x00" * 80,
            )
        )
        await async_db_session.flush()

        req = WaveformWindowRequest(
            therapy_date=therapy_date,
            session_id=session_b.id,
            offset_start=0.0,
            offset_end=10.0,
            channels=[WaveformChannelName.FLOW],
        )
        # Profile A must not receive profile B's waveform bytes
        with pytest.raises(ValueError):
            await fetch_waveform_window_raw(async_db_session, profile_a_id, req)

    async def test_owned_session_returns_waveform_data(self, async_db_session):
        """fetch_waveform_window_raw returns data for an owned session."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 2, 2)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        req = WaveformWindowRequest(
            therapy_date=therapy_date,
            session_id=session.id,
            offset_start=0.0,
            offset_end=10.0,
            channels=[WaveformChannelName.FLOW],
        )
        # No Waveform row → empty channels but no error
        raw = await fetch_waveform_window_raw(async_db_session, profile_id, req)
        assert raw.session_id == session.id
        assert WaveformChannelName.FLOW in raw.missing_channels

    async def test_wrong_therapy_date_raises_value_error(self, async_db_session):
        """fetch_waveform_window_raw raises ValueError when therapy_date does not match.

        An owned session on the correct profile is paired with a different therapy_date
        in the request.  The Day join must detect the mismatch and reject.
        plan §9 lines 822-825: full tuple (session, date, device) must match.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        correct_date = date(2025, 3, 1)
        wrong_date = date(2025, 3, 2)  # different date
        _, session = await _make_day_and_session(async_db_session, dev.id, correct_date)

        req = WaveformWindowRequest(
            therapy_date=wrong_date,  # does not match the session's date
            session_id=session.id,
            offset_start=0.0,
            offset_end=10.0,
            channels=[WaveformChannelName.FLOW],
        )
        with pytest.raises(ValueError):
            await fetch_waveform_window_raw(async_db_session, profile_id, req)

    async def test_wrong_device_id_raises_value_error(self, async_db_session):
        """fetch_waveform_window_raw raises ValueError when device_id doesn't match session.

        Same profile, two devices.  Session belongs to device_a; request specifies
        device_b.  The device_id predicate must reject the mismatch.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev_a = await _make_device(async_db_session, profile_id)
        dev_b = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 3)
        _, session_a = await _make_day_and_session(
            async_db_session, dev_a.id, therapy_date
        )

        req = WaveformWindowRequest(
            therapy_date=therapy_date,
            session_id=session_a.id,
            device_id=dev_b.id,  # session belongs to dev_a, not dev_b
            offset_start=0.0,
            offset_end=10.0,
            channels=[WaveformChannelName.FLOW],
        )
        with pytest.raises(ValueError):
            await fetch_waveform_window_raw(async_db_session, profile_id, req)


# ---------------------------------------------------------------------------
# compute_waveform_window (module-level)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeWaveformWindow:
    def test_empty_raw_returns_empty_channels(self):
        """compute_waveform_window on empty raw channels returns window with no channels."""
        from snore.services.breath_service import RawWaveformWindow  # noqa: PLC0415

        req = WaveformWindowRequest(
            therapy_date=date(2025, 1, 1),
            session_id=1,
            offset_start=0.0,
            offset_end=60.0,
        )
        raw = RawWaveformWindow(
            request=req,
            session_id=1,
            session_start_wall_clock=datetime(2025, 1, 1, 22, 0, 0),
            channels=[],
            missing_channels=[],
        )
        window = compute_waveform_window(raw)
        assert window.channels == []
        assert window.missing_channel_reason is None


# ---------------------------------------------------------------------------
# fetch_ca_analysis / compute_ca_analysis seam (module-level)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchCaAnalysis:
    async def test_fetch_returns_dto_free_of_orm_handles(self, async_db_session):
        """fetch_ca_analysis returns a RawCaAnalysis carrying no ORM/DB objects.

        Verifies that RawCaAnalysis, its session_data, pre_waveform, and
        ca_events fields all contain only plain Python/Pydantic values so that
        compute_ca_analysis can run outside the DB scope safely.
        """
        from snore.services.breath_service import (  # noqa: PLC0415
            BreathService,
            RawCaAnalysis,
            RawCaSessionData,
        )

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 6, 20)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(async_db_session, session, profile_id)
        # Add a CA event so session_data is populated
        ca_event = models.Event(
            session_id=session.id,
            event_type="CA",
            start_time=session.start_time + timedelta(minutes=5),
            duration_seconds=10.0,
        )
        async_db_session.add(ca_event)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        raw = await svc.fetch_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert isinstance(raw, RawCaAnalysis)
        assert len(raw.session_data) == 1

        sd = raw.session_data[0]
        assert isinstance(sd, RawCaSessionData)
        # No ORM model types anywhere in the DTO
        assert isinstance(sd.session_id, int)
        assert isinstance(sd.session_start, datetime)
        assert len(sd.ca_events) == 1
        assert sd.ca_events[0].duration_seconds == pytest.approx(10.0)
        # pre_waveform carries bytes, not SQLAlchemy rows
        from snore.services.breath_service import RawWaveformWindow  # noqa: PLC0415

        assert isinstance(sd.pre_waveform, RawWaveformWindow)
        for ch in sd.pre_waveform.channels:
            assert isinstance(ch.raw_bytes, bytes)

    async def test_compute_is_deterministic_on_fixed_raw(self, async_db_session):
        """compute_ca_analysis produces identical output when called twice with the same raw.

        Verifies that the pure function has no mutable state — identical inputs
        always yield identical CaAnalysisResult values.
        """
        from snore.services.breath_service import (  # noqa: PLC0415
            BreathService,
            compute_ca_analysis,
        )

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 6, 21)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(async_db_session, session, profile_id)
        ca_event = models.Event(
            session_id=session.id,
            event_type="CA",
            start_time=session.start_time + timedelta(minutes=15),
            duration_seconds=8.0,
        )
        async_db_session.add(ca_event)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        raw = await svc.fetch_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        result1 = compute_ca_analysis(raw)
        result2 = compute_ca_analysis(raw)

        assert result1.model_dump() == result2.model_dump()

    async def test_orchestrator_equals_fetch_plus_compute(self, async_db_session):
        """get_ca_analysis(date) == compute_ca_analysis(fetch_ca_analysis(date)).

        Verifies that the convenience orchestrator produces bit-identical output
        to the explicit fetch+compute composition.
        """
        from snore.services.breath_service import (  # noqa: PLC0415
            BreathService,
            compute_ca_analysis,
        )

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 6, 22)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)
        await _store_analysis_with_breaths(async_db_session, session, profile_id)
        ca_event = models.Event(
            session_id=session.id,
            event_type="CA",
            start_time=session.start_time + timedelta(minutes=15),
            duration_seconds=8.0,
        )
        async_db_session.add(ca_event)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        raw = await svc.fetch_ca_analysis(therapy_date=therapy_date, device_id=dev.id)
        composed = compute_ca_analysis(raw)
        orchestrated = await svc.get_ca_analysis(
            therapy_date=therapy_date, device_id=dev.id
        )

        assert composed.model_dump() == orchestrated.model_dump()


# ---------------------------------------------------------------------------
# A-suite rework — drives store_result() production path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAsuiteViaStoreResult:
    """A1-A3 reworked to drive the production store_result() path."""

    async def test_a1_store_result_writes_breath_rows_observable(
        self, async_db_session
    ):
        """A1: store_result() populates Breath rows; asserted via separate query."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 7, 1)
        )
        ar = await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=5
        )

        count = (
            await async_db_session.execute(
                select(func.count())
                .select_from(models.Breath)
                .where(models.Breath.analysis_result_id == ar.id)
            )
        ).scalar()

        assert count == 5, f"Expected 5 Breath rows, got {count}"

    async def test_a1_breath_rows_reference_correct_session_id(self, async_db_session):
        """A1: each Breath.session_id matches the parent Session.id."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 7, 2)
        )
        ar = await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=3
        )

        rows = (
            (
                await async_db_session.execute(
                    select(models.Breath).where(
                        models.Breath.analysis_result_id == ar.id
                    )
                )
            )
            .scalars()
            .all()
        )
        assert all(r.session_id == session.id for r in rows), (
            "All Breath rows must reference the correct session_id"
        )

    async def test_a2_two_store_result_calls_produce_two_rows(self, async_db_session):
        """A2: two store_result() calls → two AnalysisResult rows preserved."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 7, 3)
        )

        ar1 = await _store_analysis_with_breaths(async_db_session, session, profile_id)
        ar2 = await _store_analysis_with_breaths(async_db_session, session, profile_id)

        count = (
            await async_db_session.execute(
                select(func.count())
                .select_from(models.AnalysisResult)
                .where(models.AnalysisResult.session_id == session.id)
            )
        ).scalar()

        assert count == 2, f"Two store_result calls must produce 2 rows; got {count}"
        assert ar2.id > ar1.id, "Second result must have higher id"

    async def test_a2_production_selector_returns_latest_by_id_on_equal_created_at(
        self, async_db_session
    ):
        """A2 tie-breaker: BreathService._latest_analysis_for_session returns higher id."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 7, 4)
        )

        tie_ts = datetime(2025, 7, 4, 8, 0, 0, tzinfo=UTC)
        algo = _make_algo_versions()

        ar_old = models.AnalysisResult(
            session_id=session.id,
            timestamp_start=session.start_time,
            timestamp_end=session.end_time or session.start_time + timedelta(hours=7),
            programmatic_result_json={},
            processing_time_ms=50,
            engine_versions_json=algo.model_dump(),
            created_at=tie_ts,
        )
        async_db_session.add(ar_old)
        await async_db_session.flush()

        ar_new = models.AnalysisResult(
            session_id=session.id,
            timestamp_start=session.start_time,
            timestamp_end=session.end_time or session.start_time + timedelta(hours=7),
            programmatic_result_json={},
            processing_time_ms=55,
            engine_versions_json=algo.model_dump(),
            created_at=tie_ts,  # same timestamp
        )
        async_db_session.add(ar_new)
        await async_db_session.flush()
        ar_new_id = ar_new.id

        # Call the production selector
        svc = BreathService(async_db_session, profile_id=profile_id)
        _status, _algo, ar_id = await svc._latest_analysis_for_session(session.id)

        assert ar_id == ar_new_id, (
            f"Tie-breaker must select highest id ({ar_new_id}), got {ar_id}"
        )

    async def test_a3_rollback_when_store_result_raises_after_flush(
        self, async_db_session
    ):
        """A3: AnalysisResult is absent after store_result raises mid-transaction."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 7, 5)
        )

        from snore.analysis.types import ComputedBreath  # noqa: PLC0415

        result_dto = AnalysisResultDTO(
            session_id=session.id,
            session_duration_hours=7.0,
            total_breaths=1,
            machine_events=[],
            mode_results={
                "aasm": ModeResult(
                    mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                )
            },
            timestamp_start=session.start_time.timestamp(),
            timestamp_end=(
                session.end_time or session.start_time + timedelta(hours=7)
            ).timestamp(),
        )
        good_breath = ComputedBreath(
            breath_number=1,
            start_offset_s=0.0,
            end_offset_s=3.0,
            inspiration_time_s=1.2,
            expiration_time_s=1.8,
            total_time_s=3.0,
            i_e_ratio=0.67,
            duty_cycle=0.4,
            peak_flow_lpm=30.0,
            peak_exp_flow_lpm=20.0,
            tidal_volume_ml=400.0,
            respiratory_rate_rolling=15.0,
            flatness_index=0.2,
            mid_insp_flattening=0.35,
            flow_class=1,
            flow_confidence=0.9,
            is_recovery_breath=False,
            inferred_trigger_type="normal",
            trigger_confidence=0.8,
            inferred_cycle_type="normal",
            cycle_confidence=0.75,
            trigger_cycle_applicable=True,
            trigger_cycle_reason=None,
            leak_valid=True,
            leak_valid_reason=None,
            ramp_active=None,
            ramp_active_reason="not_available",
            mask_off=False,
            mask_off_reason=None,
        )
        computation = AnalysisComputation(
            summary=result_dto, breaths=[good_breath], primary_mode="aasm"
        )

        # Monkeypatch add_all to raise after flush
        original_add_all = async_db_session.add_all

        def _failing_add_all(rows: Any) -> None:
            raise RuntimeError("Induced child-insert failure")

        async_db_session.add_all = _failing_add_all

        try:
            svc = AnalysisService(async_db_session, profile_id=profile_id)
            with pytest.raises((RuntimeError, Exception)):
                await svc.store_result(computation, processing_time_ms=1)
        finally:
            async_db_session.add_all = original_add_all
            await async_db_session.rollback()

        # After rollback, no AnalysisResult should exist
        count = (
            await async_db_session.execute(
                select(func.count())
                .select_from(models.AnalysisResult)
                .where(models.AnalysisResult.session_id == session.id)
            )
        ).scalar()
        assert count == 0, "AnalysisResult must be rolled back when child-insert fails"


# ---------------------------------------------------------------------------
# Vendor applicability — non-ResMed device gets unvalidated_device
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVendorApplicability:
    async def test_non_resmed_device_gets_unvalidated_trigger_cycle(
        self, async_db_session
    ):
        """Breaths computed for a non-ResMed device have unvalidated_device applicability."""
        from snore.analysis.service import _build_computed_breaths  # noqa: PLC0415
        from snore.analysis.shared.breath_segmenter import (
            BreathMetrics,  # noqa: PLC0415
        )
        from snore.analysis.shared.feature_extractors import (  # noqa: PLC0415
            largest_inspiratory_segment,
        )
        from snore.analysis.shared.trigger_cycle import (  # noqa: PLC0415
            APPLICABILITY_UNVALIDATED_DEVICE,
        )

        t = np.linspace(0.0, 30.0, 750)  # 25Hz, 30s
        flow = np.abs(np.sin(np.linspace(0.0, 10 * np.pi, 750))) * 30.0

        bm = BreathMetrics(
            breath_number=1,
            start_time=0.0,
            middle_time=1.2,
            end_time=3.0,
            inspiration_time=1.2,
            expiration_time=1.8,
            duration=3.0,
            respiratory_rate=20.0,
            respiratory_rate_rolling=15.0,
            tidal_volume=400.0,
            tidal_volume_smoothed=400.0,
            peak_inspiratory_flow=30.0,
            peak_expiratory_flow=20.0,
            i_e_ratio=0.67,
            minute_ventilation=6.0,
            amplitude=50.0,
            is_complete=True,
        )

        insp_flow = largest_inspiratory_segment(
            flow[np.searchsorted(t, bm.start_time) : np.searchsorted(t, bm.end_time)]
        )
        results = _build_computed_breaths(
            breaths=[bm],
            flow_pattern_by_number={},
            breath_shape_by_number={},
            insp_flow_by_number={bm.breath_number: insp_flow},
            recovery_breath_indices=set(),
            leak_timestamps=None,
            leak_values=None,
            device_manufacturer="Philips",
        )

        assert len(results) == 1
        cb = results[0]
        # Non-ResMed: trigger/cycle heuristic is not applicable
        assert (
            cb.trigger_cycle_applicable is False
            or cb.trigger_cycle_reason == APPLICABILITY_UNVALIDATED_DEVICE
        )

    async def test_resmed_device_gets_validated_trigger_cycle(self, async_db_session):
        """Breaths computed for a ResMed device do NOT carry unvalidated_device reason."""
        from snore.analysis.service import _build_computed_breaths  # noqa: PLC0415
        from snore.analysis.shared.breath_segmenter import (
            BreathMetrics,  # noqa: PLC0415
        )
        from snore.analysis.shared.feature_extractors import (  # noqa: PLC0415
            largest_inspiratory_segment,
        )

        t = np.linspace(0.0, 30.0, 750)
        flow = np.abs(np.sin(np.linspace(0.0, 10 * np.pi, 750))) * 30.0

        bm = BreathMetrics(
            breath_number=1,
            start_time=0.0,
            middle_time=1.2,
            end_time=3.0,
            inspiration_time=1.2,
            expiration_time=1.8,
            duration=3.0,
            respiratory_rate=20.0,
            respiratory_rate_rolling=15.0,
            tidal_volume=400.0,
            tidal_volume_smoothed=400.0,
            peak_inspiratory_flow=30.0,
            peak_expiratory_flow=20.0,
            i_e_ratio=0.67,
            minute_ventilation=6.0,
            amplitude=50.0,
            is_complete=True,
        )

        insp_flow = largest_inspiratory_segment(
            flow[np.searchsorted(t, bm.start_time) : np.searchsorted(t, bm.end_time)]
        )
        results = _build_computed_breaths(
            breaths=[bm],
            flow_pattern_by_number={},
            breath_shape_by_number={},
            insp_flow_by_number={bm.breath_number: insp_flow},
            recovery_breath_indices=set(),
            leak_timestamps=None,
            leak_values=None,
            device_manufacturer="ResMed",
        )

        assert len(results) == 1
        cb = results[0]
        # ResMed: trigger/cycle heuristic is applicable (may be None on short breath)
        assert cb.trigger_cycle_reason != "unvalidated_device"


# ---------------------------------------------------------------------------
# Deletion cascade — Breath rows cascade-delete with AnalysisResult
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeletionCascade:
    async def test_breath_rows_deleted_when_analysis_result_deleted(
        self, async_db_session
    ):
        """Breath rows are cascade-deleted when their AnalysisResult parent is deleted."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, date(2025, 9, 1)
        )
        ar = await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=5
        )
        ar_id = ar.id

        # Verify breaths exist
        before_count = (
            await async_db_session.execute(
                select(func.count())
                .select_from(models.Breath)
                .where(models.Breath.analysis_result_id == ar_id)
            )
        ).scalar()
        assert before_count == 5

        # Delete the AnalysisResult
        await async_db_session.delete(ar)
        await async_db_session.flush()

        # Breaths must be gone
        after_count = (
            await async_db_session.execute(
                select(func.count())
                .select_from(models.Breath)
                .where(models.Breath.analysis_result_id == ar_id)
            )
        ).scalar()
        assert after_count == 0, (
            f"Breath rows must cascade-delete with AnalysisResult; {after_count} remain"
        )


# ---------------------------------------------------------------------------
# Missing breaths table — actionable RuntimeError (real-path, no monkeypatching)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMissingBreathsTable:
    async def test_missing_breaths_table_raises_actionable_error(
        self, async_db_session, tmp_path
    ):
        """store_result raises a RuntimeError with actionable message when breaths table absent.

        Uses a separate fresh SQLite database (created via Base.metadata.create_all)
        then executes ``DROP TABLE breaths`` before calling store_result().  When
        store_result flushes the breath children, SQLAlchemy raises
        OperationalError("no such table: breaths") from the real SQLite engine —
        no monkeypatching.  The handler must convert this to a RuntimeError with
        the drop-and-reimport guidance message.
        """
        import sqlite3  # noqa: PLC0415

        from sqlalchemy.ext.asyncio import (  # noqa: PLC0415
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        from snore.analysis.types import ComputedBreath  # noqa: PLC0415
        from snore.database.models import Base  # noqa: PLC0415

        db_path = tmp_path / "missing_breaths.db"
        async_url = f"sqlite+aiosqlite:///{db_path}"
        engine = create_async_engine(async_url, echo=False)

        # Create full schema (including breaths table).
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(
            bind=engine, expire_on_commit=False, class_=AsyncSession
        )

        # Insert the prerequisite rows (profile, device, session) using the full schema.
        async with factory() as setup_db:
            _, profile_id = await _make_profile(setup_db)
            dev = await _make_device(setup_db, profile_id)
            _, session = await _make_day_and_session(
                setup_db, dev.id, date(2025, 10, 1)
            )
            await setup_db.commit()
            session_id = session.id

        await engine.dispose()

        # Drop the breaths table directly via sqlite3 (synchronous, no ORM).
        con = sqlite3.connect(str(db_path))
        con.execute("DROP TABLE breaths")
        con.commit()
        con.close()

        # Re-open the engine — now the breaths table is gone.
        engine2 = create_async_engine(async_url, echo=False)
        factory2 = async_sessionmaker(
            bind=engine2, expire_on_commit=False, class_=AsyncSession
        )

        result_dto = AnalysisResultDTO(
            session_id=session_id,
            session_duration_hours=7.0,
            total_breaths=1,
            machine_events=[],
            mode_results={
                "aasm": ModeResult(
                    mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                )
            },
            timestamp_start=1000.0,
            timestamp_end=26200.0,
        )
        breath = ComputedBreath(
            breath_number=1,
            start_offset_s=0.0,
            end_offset_s=3.0,
            inspiration_time_s=1.2,
            expiration_time_s=1.8,
            total_time_s=3.0,
            i_e_ratio=0.67,
            duty_cycle=0.4,
            peak_flow_lpm=30.0,
            peak_exp_flow_lpm=20.0,
            tidal_volume_ml=400.0,
            respiratory_rate_rolling=15.0,
            flatness_index=0.2,
            mid_insp_flattening=0.35,
            flow_class=1,
            flow_confidence=0.9,
            is_recovery_breath=False,
            inferred_trigger_type=None,
            trigger_confidence=None,
            inferred_cycle_type=None,
            cycle_confidence=None,
            trigger_cycle_applicable=None,
            trigger_cycle_reason=None,
            leak_valid=None,
            leak_valid_reason="channel_absent",
            ramp_active=None,
            ramp_active_reason="not_available",
            mask_off=None,
            mask_off_reason=None,
        )
        computation = AnalysisComputation(
            summary=result_dto, breaths=[breath], primary_mode="aasm"
        )

        try:
            async with factory2() as db:
                svc = AnalysisService(db, profile_id=profile_id)
                with pytest.raises(
                    RuntimeError,
                    match="breaths.*table.*missing|drop.*database|re-import",
                ):
                    await svc.store_result(computation, processing_time_ms=1)
        finally:
            await engine2.dispose()


# ---------------------------------------------------------------------------
# Two-profile adversarial matrix
# ---------------------------------------------------------------------------
#
# Every seam category is tested for profile isolation: a session/device owned
# by profile B is invisible to a BreathService scoped to profile A.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTwoProfileIsolation:
    """Two-profile adversarial matrix per Thufir review finding (pass 1).

    Profile A owns device_a/session_a; profile B owns device_b/session_b.
    BreathService scoped to profile A must never return data owned by B.
    """

    async def _setup_two_profiles(
        self, db: AsyncSession
    ) -> tuple[int, int, models.Session, int, int, models.Session]:
        """Create two profiles with independent devices and sessions.

        Returns (profile_a_id, device_a_id, session_a, profile_b_id, device_b_id, session_b).
        """
        therapy_date = date(2025, 6, 15)
        _, profile_a_id = await _make_profile(db)
        dev_a = await _make_device(db, profile_a_id)
        _, session_a = await _make_day_and_session(db, dev_a.id, therapy_date)

        _, profile_b_id = await _make_profile(db)
        dev_b = await _make_device(db, profile_b_id, manufacturer="Other")
        _, session_b = await _make_day_and_session(db, dev_b.id, therapy_date)

        return profile_a_id, dev_a.id, session_a, profile_b_id, dev_b.id, session_b

    async def test_get_analysis_status_foreign_session_returns_not_run(
        self, async_db_session
    ):
        """get_analysis_status for a session owned by B returns NOT_RUN for profile A."""
        (
            profile_a_id,
            _dev_a_id,
            _session_a,
            _profile_b_id,
            _dev_b_id,
            session_b,
        ) = await self._setup_two_profiles(async_db_session)

        # Populate an analysis result for session_b
        await _store_analysis_with_breaths(
            async_db_session, session_b, _profile_b_id, n_breaths=5
        )

        # Profile A must not see session_b's analysis
        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        status, algo = await svc_a.get_analysis_status(session_b.id)
        assert status == AnalysisStatus.NOT_RUN
        assert algo is None

    async def test_get_breath_table_foreign_explicit_session_raises(
        self, async_db_session
    ):
        """get_breath_table with an explicit session_id owned by B raises ValueError."""
        (
            profile_a_id,
            _,
            _,
            profile_b_id,
            _dev_b_id,
            session_b,
        ) = await self._setup_two_profiles(async_db_session)
        await _store_analysis_with_breaths(
            async_db_session, session_b, profile_b_id, n_breaths=3
        )

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        with pytest.raises(ValueError, match="profile|not owned|not found"):
            await svc_a.get_breath_table(
                BreathQueryRange(
                    therapy_date=date(2025, 6, 15),
                    session_id=session_b.id,
                    offset_start=0.0,
                    offset_end=300.0,  # 5-min window within cap
                )
            )

    async def test_resolve_range_foreign_profile_sessions_invisible(
        self, async_db_session
    ):
        """_resolve_range does not expose sessions from profile B when scoped to A.

        When two profiles each have a session on the same date, profile A's
        _resolve_range only finds A's session; profile B's session is invisible.
        """
        therapy_date = date(2025, 6, 15)
        (
            profile_a_id,
            dev_a_id,
            session_a,
            _profile_b_id,
            _dev_b_id,
            session_b,
        ) = await self._setup_two_profiles(async_db_session)

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        # Should resolve to A's device without ambiguity error
        resolved_device_id, sessions_by_date = await svc_a._resolve_range(
            therapy_date, therapy_date, None
        )
        assert resolved_device_id == dev_a_id
        day_sessions = sessions_by_date.get(therapy_date, [])
        session_ids = [s.id for s in day_sessions]
        assert session_a.id in session_ids
        # B's session must not be returned
        assert session_b.id not in session_ids

    async def test_get_device_capabilities_foreign_device_returns_empty(
        self, async_db_session
    ):
        """get_device_capabilities for device_b returns null_reason for profile A."""
        (
            profile_a_id,
            _dev_a_id,
            _session_a,
            _profile_b_id,
            dev_b_id,
            _session_b,
        ) = await self._setup_two_profiles(async_db_session)

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        caps = await svc_a.get_device_capabilities(device_id=dev_b_id)
        assert caps.null_reason is not None
        assert caps.session_count == 0
        assert caps.nights_with_data == 0

    async def test_find_windows_foreign_date_returns_not_run(self, async_db_session):
        """find_windows on a date only populated by profile B returns NOT_RUN for A."""
        therapy_date = date(2025, 7, 20)
        _, profile_a_id = await _make_profile(async_db_session)
        # A has no session on this date

        _, profile_b_id = await _make_profile(async_db_session)
        dev_b = await _make_device(async_db_session, profile_b_id, manufacturer="Other")
        _, session_b = await _make_day_and_session(
            async_db_session, dev_b.id, therapy_date
        )
        await _store_analysis_with_breaths(
            async_db_session, session_b, profile_b_id, n_breaths=5
        )

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        result = await svc_a.find_windows(
            therapy_date=therapy_date,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=5,
        )
        assert result.day_status == DayAnalysisStatus.NOT_RUN
        assert result.windows == []

    async def test_compare_epochs_foreign_device_returns_not_available(
        self, async_db_session
    ):
        """compare_epochs with a foreign device_id returns NOT_AVAILABLE (not NO_DATA_IN_RANGE).

        A device owned by profile B is not accessible by profile A's service.
        The structured null_reason must be NOT_AVAILABLE, distinguishing it from
        an owned device that simply has no data in the requested range.
        """
        (
            profile_a_id,
            _dev_a_id,
            _session_a,
            _profile_b_id,
            dev_b_id,
            _session_b,
        ) = await self._setup_two_profiles(async_db_session)

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        result = await svc_a.compare_epochs(
            epochs=[
                EpochRequest(
                    label="b_epoch",
                    date_start=date(2025, 6, 15),
                    date_end=date(2025, 6, 15),
                    device_id=dev_b_id,
                )
            ]
        )
        assert len(result.epochs) == 1
        # Foreign device → NOT_AVAILABLE (plan §13: explicit foreign device_id)
        assert result.epochs[0].null_reason == NullReason.NOT_AVAILABLE

    async def test_get_waveform_window_foreign_device_raises(self, async_db_session):
        """get_waveform_window with a foreign device_id raises DeviceNotOwnedError.

        Resolution now lives in get_waveform_window via _resolve_range; the raw
        fetch no longer performs resolution or profile checks.
        """
        from snore.services.breath_service import DeviceNotOwnedError  # noqa: PLC0415

        (
            profile_a_id,
            _,
            _,
            _profile_b_id,
            dev_b_id,
            session_b,
        ) = await self._setup_two_profiles(async_db_session)

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        with pytest.raises(DeviceNotOwnedError):
            await svc_a.get_waveform_window(
                WaveformWindowRequest(
                    therapy_date=date(2025, 6, 15),
                    session_id=session_b.id,
                    device_id=dev_b_id,
                    channels=[WaveformChannelName.FLOW],
                    offset_start=0.0,
                    offset_end=60.0,
                )
            )

    async def test_ambiguity_payload_excludes_foreign_profile_session(
        self, async_db_session
    ):
        """Ambiguity error when A has 2 sessions lists only A's sessions, not B's.

        Before the Item-1 fix, the resolution query had no profile predicate, so
        the payload could include profile B's session IDs.  This test fails if
        the profile predicate is removed from fetch_waveform_window_raw.
        """
        therapy_date = date(2025, 6, 15)
        _, profile_a_id = await _make_profile(async_db_session)
        dev_a = await _make_device(async_db_session, profile_a_id)

        # Profile A: two sessions on the same date (share one Day row)
        day_a = models.Day(device_id=dev_a.id, date=therapy_date, session_count=2)
        async_db_session.add(day_a)
        await async_db_session.flush()
        session_a1 = models.Session(
            device_id=dev_a.id,
            day_id=day_a.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
            ),
            end_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 23, 0
            ),
            duration_seconds=7200.0,
        )
        session_a2 = models.Session(
            device_id=dev_a.id,
            day_id=day_a.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 23, 30
            ),
            end_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day + 1, 1, 0
            ),
            duration_seconds=5400.0,
        )
        async_db_session.add_all([session_a1, session_a2])
        await async_db_session.flush()

        # Profile B: one session on the same date
        _, profile_b_id = await _make_profile(async_db_session)
        dev_b = await _make_device(async_db_session, profile_b_id)
        _, session_b = await _make_day_and_session(
            async_db_session, dev_b.id, therapy_date
        )

        from snore.services.breath_service import (  # noqa: PLC0415
            MultiSessionAmbiguityError,
        )

        # Resolution now lives in get_waveform_window via _resolve_range.
        # Profile A has 1 device with 2 sessions → MultiSessionAmbiguityError;
        # profile B's session must not appear in the payload.
        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        with pytest.raises(MultiSessionAmbiguityError) as exc_info:
            await svc_a.get_waveform_window(
                WaveformWindowRequest(
                    therapy_date=therapy_date,
                    channels=[WaveformChannelName.FLOW],
                    offset_start=0.0,
                    offset_end=60.0,
                )
            )
        err = exc_info.value
        session_ids_in_payload = {s.session_id for s in err.sessions}
        assert session_b.id not in session_ids_in_payload, (
            "foreign profile B session must not appear in ambiguity payload"
        )
        assert {session_a1.id, session_a2.id} == session_ids_in_payload

    async def test_foreign_session_on_same_date_does_not_cause_ambiguity(
        self, async_db_session
    ):
        """Profile A (1 session) + profile B (1 session) on same date → no ambiguity.

        Resolution now lives in get_waveform_window via _resolve_range.  Profile A
        has one device with one session; profile B's session is scoped out during
        _resolve_range, so no multi-session error fires.
        """
        therapy_date = date(2025, 6, 15)
        (
            profile_a_id,
            dev_a_id,
            session_a,
            _profile_b_id,
            _dev_b_id,
            _session_b,
        ) = await self._setup_two_profiles(async_db_session)

        # Should resolve cleanly to profile A's session (profile B's is invisible)
        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        window = await svc_a.get_waveform_window(
            WaveformWindowRequest(
                therapy_date=therapy_date,
                channels=[WaveformChannelName.FLOW],
                offset_start=0.0,
                offset_end=60.0,
            )
        )
        # Returns empty channels (no waveform data) but resolved to A's session
        assert window.session_id == session_a.id

    async def test_foreign_analysis_result_not_returned_via_breath_table_by_date(
        self, async_db_session
    ):
        """get_breath_table by date returns only profile A's breath rows, not B's.

        Profile B has a larger analysis (10 breaths) on the same date as A (3 breaths).
        Profile A's breath table resolved by date must return exactly 3 rows.
        """
        therapy_date = date(2025, 6, 15)
        (
            profile_a_id,
            dev_a_id,
            session_a,
            profile_b_id,
            _dev_b_id,
            session_b,
        ) = await self._setup_two_profiles(async_db_session)

        await _store_analysis_with_breaths(
            async_db_session, session_a, profile_a_id, n_breaths=3
        )
        await _store_analysis_with_breaths(
            async_db_session, session_b, profile_b_id, n_breaths=10
        )

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        page = await svc_a.get_breath_table(
            BreathQueryRange(
                therapy_date=therapy_date,
                device_id=dev_a_id,
                offset_start=0.0,
                offset_end=900.0,
            )
        )
        assert len(page.rows) == 3, "must return only profile A's 3 breath rows"

    async def test_get_contextual_events_foreign_session_not_returned(
        self, async_db_session
    ):
        """get_contextual_events for profile A's session returns A's events, not B's."""
        therapy_date = date(2025, 6, 15)
        (
            profile_a_id,
            dev_a_id,
            session_a,
            _profile_b_id,
            _dev_b_id,
            session_b,
        ) = await self._setup_two_profiles(async_db_session)

        # Profile A: one OA event
        ev_a = models.Event(
            session_id=session_a.id,
            event_type="OA",
            start_time=session_a.start_time + timedelta(minutes=30),
            duration_seconds=12.0,
        )
        # Profile B: two CA events (must not appear in A's results)
        ev_b1 = models.Event(
            session_id=session_b.id,
            event_type="CA",
            start_time=session_b.start_time + timedelta(minutes=10),
            duration_seconds=8.0,
        )
        ev_b2 = models.Event(
            session_id=session_b.id,
            event_type="CA",
            start_time=session_b.start_time + timedelta(minutes=20),
            duration_seconds=6.0,
        )
        async_db_session.add_all([ev_a, ev_b1, ev_b2])
        await async_db_session.flush()

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        events = await svc_a.get_contextual_events(
            therapy_date=therapy_date, device_id=dev_a_id
        )

        event_types = {e.event_type for e in events}
        assert "CA" not in event_types, "profile B's CA events must not appear"
        assert len(events) == 1
        assert events[0].event_type == "OA"

    async def test_get_ca_analysis_foreign_session_ca_events_not_returned(
        self, async_db_session
    ):
        """get_ca_analysis for profile A's session returns A's CA events, not B's."""
        therapy_date = date(2025, 6, 15)
        (
            profile_a_id,
            dev_a_id,
            session_a,
            _profile_b_id,
            _dev_b_id,
            session_b,
        ) = await self._setup_two_profiles(async_db_session)

        # Profile A: no CA events
        # Profile B: two CA events
        for i in range(2):
            async_db_session.add(
                models.Event(
                    session_id=session_b.id,
                    event_type="CA",
                    start_time=session_b.start_time + timedelta(minutes=10 + i * 15),
                    duration_seconds=10.0,
                )
            )
        await async_db_session.flush()

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        result = await svc_a.get_ca_analysis(
            therapy_date=therapy_date, device_id=dev_a_id
        )

        assert result.ca_events == [], (
            "profile B's CA events must not appear in profile A's get_ca_analysis"
        )


# ---------------------------------------------------------------------------
# Stale-session coverage / status-precedence tests (Thufir pass-1 finding #3)
# ---------------------------------------------------------------------------


async def _store_stale_analysis_with_breaths(
    db: AsyncSession, session: models.Session, n_breaths: int = 5
) -> models.AnalysisResult:
    """Insert a stale AnalysisResult (legacy engine_versions_json) + Breath rows.

    Uses the legacy flat shape to guarantee STALE_VERSION from _latest_analysis_for_session.
    """
    start = session.start_time
    end = session.end_time or (start + timedelta(hours=7))
    ar = models.AnalysisResult(
        session_id=session.id,
        timestamp_start=start,
        timestamp_end=end,
        programmatic_result_json={},
        processing_time_ms=10,
        engine_versions_json={"version": "0.0.1"},  # legacy flat → STALE_VERSION
    )
    db.add(ar)
    await db.flush()

    breaths = [
        models.Breath(
            analysis_result_id=ar.id,
            session_id=session.id,
            breath_number=i + 1,
            start_offset_s=float(i * 4),
            end_offset_s=float(i * 4 + 3),
            leak_valid=True,
            flow_class=4,  # flow-limited → would rank in WORST_FLATTENING
            mid_insp_flattening=0.8,  # high — would pollute results if included
        )
        for i in range(n_breaths)
    ]
    db.add_all(breaths)
    await db.flush()
    return ar


@pytest.mark.unit
class TestStaleCoverageStateMachine:
    """Status-precedence and stale-row exclusion tests for finding #3."""

    async def test_all_stale_nightly_summary_is_stale(self, async_db_session):
        """get_nightly_summary returns STALE (not NOT_RUN) when analysis exists but is stale."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 1)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)

        # Stale analysis — old engine_versions_json
        await _store_stale_analysis_with_breaths(async_db_session, session, n_breaths=3)

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_nightly_summary(therapy_date)

        # plan §1 line 864 rule 4: all-stale → STALE, not NOT_RUN
        assert result.day_status == DayAnalysisStatus.STALE

    async def test_stale_session_excluded_from_find_windows_breath_rows(
        self, async_db_session
    ):
        """find_windows returns empty windows when the only analysis is stale.

        Stale breath rows must not contribute to WORST_FLATTENING results even
        though ar_id is not None.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 2)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)

        # Stale analysis with highly-ranked breaths that would appear if leaking
        await _store_stale_analysis_with_breaths(async_db_session, session, n_breaths=5)

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date,
            WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=10,
        )

        # plan §1 line 864 rule 4: all-stale → STALE
        assert result.day_status == DayAnalysisStatus.STALE
        assert result.windows == [], "stale breath rows must not appear in windows"

    async def test_stale_and_not_run_find_windows_status_is_partial(
        self, async_db_session
    ):
        """find_windows returns PARTIAL for a stale+not-run mixed day.

        One session has a stale analysis; a second session on the same day has
        no analysis at all.  plan §1 line 864 rule 5: any mixed state → PARTIAL.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 3)

        # Session A: stale
        day, session_a = await _make_day_and_session(
            async_db_session, dev.id, therapy_date
        )
        await _store_stale_analysis_with_breaths(
            async_db_session, session_a, n_breaths=2
        )

        # Session B: not run — share the same Day
        session_b = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 2, 0
            ),
            end_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 5, 0
            ),
            duration_seconds=3 * 3600.0,
        )
        async_db_session.add(session_b)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date, WindowCriterion.WORST_FLATTENING_LEAK_VALID, n=5
        )

        # plan §1 line 864 rule 5: stale + not-run → PARTIAL (not STALE)
        assert result.day_status == DayAnalysisStatus.PARTIAL

    async def test_ca_events_returned_when_analysis_stale(self, async_db_session):
        """get_ca_analysis returns CA events even when analysis is stale.

        CA events are stored at import time (event-anchored) and must be available
        regardless of analysis version.  day_status must be STALE, not NOT_RUN.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 3, 4)
        _, session = await _make_day_and_session(async_db_session, dev.id, therapy_date)

        # Stale analysis
        await _store_stale_analysis_with_breaths(async_db_session, session, n_breaths=2)

        # CA event stored at import time (independent of analysis)
        ca_event = models.Event(
            session_id=session.id,
            event_type="CA",
            start_time=session.start_time + timedelta(seconds=120),
            duration_seconds=15.0,
        )
        async_db_session.add(ca_event)
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date, device_id=dev.id)

        # plan §1 line 864 rule 4: all-stale → STALE
        assert result.day_status == DayAnalysisStatus.STALE
        assert len(result.ca_events) == 1, "CA events must be returned on stale days"


# ---------------------------------------------------------------------------
# Same-profile, two-device adversarial tests (Thufir pass-2 finding CRITICAL)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSameProfileTwoDevice:
    """One profile owns two devices with sessions on the same date.

    Without device_id, methods must raise DeviceAmbiguityError.
    With device_id, methods must use only that device's data.
    """

    async def _setup_two_devices(
        self, db: AsyncSession, therapy_date: date
    ) -> tuple[int, models.Device, models.Session, models.Device, models.Session]:
        """Create one profile with two devices, each with a session on therapy_date."""
        _, profile_id = await _make_profile(db)
        dev_a = await _make_device(db, profile_id, manufacturer="ResMed")
        _, session_a = await _make_day_and_session(db, dev_a.id, therapy_date)
        dev_b = await _make_device(db, profile_id, manufacturer="Philips")
        _, session_b = await _make_day_and_session(db, dev_b.id, therapy_date)
        return profile_id, dev_a, session_a, dev_b, session_b

    async def test_two_device_same_profile_nightly_summary_raises_device_ambiguity(
        self, async_db_session
    ):
        """get_nightly_summary without device_id raises DeviceAmbiguityError."""
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        therapy_date = date(2025, 11, 1)
        profile_id, dev_a, _, dev_b, _ = await self._setup_two_devices(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(DeviceAmbiguityError) as exc_info:
            await svc.get_nightly_summary(therapy_date)

        err = exc_info.value
        assert dev_a.id in err.owned_device_ids
        assert dev_b.id in err.owned_device_ids

    async def test_two_device_same_profile_find_windows_with_explicit_device(
        self, async_db_session
    ):
        """find_windows(device_id=dev_a) only uses dev_a's sessions."""
        therapy_date = date(2025, 11, 2)
        profile_id, dev_a, session_a, dev_b, session_b = await self._setup_two_devices(
            async_db_session, therapy_date
        )
        # Add OA events to dev_a's session only
        async_db_session.add(
            models.Event(
                session_id=session_a.id,
                event_type="OA",
                start_time=session_a.start_time + timedelta(minutes=30),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.find_windows(
            therapy_date,
            criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
            n=3,
            device_id=dev_a.id,
        )

        # Should resolve to dev_a without ambiguity
        assert result.device_id == dev_a.id
        # Verify dev_b sessions are not in coverage
        session_ids_in_result = {c.session_id for c in result.session_coverage}
        assert session_b.id not in session_ids_in_result

    async def test_two_device_same_profile_find_windows_raises_without_device_id(
        self, async_db_session
    ):
        """find_windows without device_id raises DeviceAmbiguityError."""
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        therapy_date = date(2025, 11, 3)
        profile_id, _, _, _, _ = await self._setup_two_devices(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(DeviceAmbiguityError):
            await svc.find_windows(
                therapy_date,
                criterion=WindowCriterion.WORST_FLATTENING_LEAK_VALID,
                n=3,
            )

    async def test_two_device_explicit_session_wrong_device_raises(
        self, async_db_session
    ):
        """get_breath_table(session_id=session_a, device_id=dev_b) raises ValueError."""
        therapy_date = date(2025, 11, 4)
        profile_id, dev_a, session_a, dev_b, session_b = await self._setup_two_devices(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError):
            await svc.get_breath_table(
                BreathQueryRange(
                    therapy_date=therapy_date,
                    session_id=session_a.id,
                    device_id=dev_b.id,
                    offset_start=0.0,
                    offset_end=300.0,
                )
            )

    async def test_two_device_get_contextual_events_raises_without_device_id(
        self, async_db_session
    ):
        """get_contextual_events without device_id raises DeviceAmbiguityError."""
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        therapy_date = date(2025, 11, 5)
        profile_id, _, _, _, _ = await self._setup_two_devices(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(DeviceAmbiguityError):
            await svc.get_contextual_events(therapy_date=therapy_date)

    async def test_two_device_get_ca_analysis_raises_without_device_id(
        self, async_db_session
    ):
        """get_ca_analysis without device_id raises DeviceAmbiguityError."""
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        therapy_date = date(2025, 11, 6)
        profile_id, _, _, _, _ = await self._setup_two_devices(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(DeviceAmbiguityError):
            await svc.get_ca_analysis(therapy_date=therapy_date)

    async def test_two_device_waveform_window_raises_device_ambiguity(
        self, async_db_session
    ):
        """get_waveform_window with two devices and no device_id raises DeviceAmbiguityError.

        Resolution now lives in get_waveform_window via _resolve_range so that
        multi-device ambiguity produces DeviceAmbiguityError (not MultiSessionAmbiguityError).
        """
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        therapy_date = date(2025, 11, 7)
        profile_id, dev_a, _, dev_b, _ = await self._setup_two_devices(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(DeviceAmbiguityError) as exc_info:
            await svc.get_waveform_window(
                WaveformWindowRequest(
                    therapy_date=therapy_date,
                    channels=[WaveformChannelName.FLOW],
                    offset_start=0.0,
                    offset_end=60.0,
                )
            )
        err = exc_info.value
        assert dev_a.id in err.owned_device_ids
        assert dev_b.id in err.owned_device_ids


# ---------------------------------------------------------------------------
# Split-night tests: one device, two sessions on same date
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSplitNight:
    """One device, two sessions on the same night.

    Events and CA analysis must aggregate across both sessions.
    """

    async def _setup_split_night(
        self, db: AsyncSession, therapy_date: date
    ) -> tuple[int, models.Device, models.Session, models.Session]:
        """Create a profile with one device and two sessions on therapy_date."""
        _, profile_id = await _make_profile(db)
        dev = await _make_device(db, profile_id)

        start_a = datetime(
            therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
        )
        day = models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        db.add(day)
        await db.flush()

        session_a = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a,
            end_time=start_a + timedelta(hours=3),
            duration_seconds=3 * 3600.0,
        )
        session_b = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a + timedelta(hours=4),
            end_time=start_a + timedelta(hours=7),
            duration_seconds=3 * 3600.0,
        )
        db.add_all([session_a, session_b])
        await db.flush()
        return profile_id, dev, session_a, session_b

    async def test_split_night_contextual_events_returns_events_from_both_sessions(
        self, async_db_session
    ):
        """get_contextual_events returns OA events from both split-night sessions."""
        therapy_date = date(2025, 12, 1)
        profile_id, dev, session_a, session_b = await self._setup_split_night(
            async_db_session, therapy_date
        )

        # Add one OA event to each session
        async_db_session.add_all(
            [
                models.Event(
                    session_id=session_a.id,
                    event_type="OA",
                    start_time=session_a.start_time + timedelta(minutes=30),
                    duration_seconds=10.0,
                ),
                models.Event(
                    session_id=session_b.id,
                    event_type="OA",
                    start_time=session_b.start_time + timedelta(minutes=30),
                    duration_seconds=8.0,
                ),
            ]
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        events = await svc.get_contextual_events(
            therapy_date=therapy_date, device_id=dev.id
        )

        # Both sessions' events must be returned
        assert len(events) == 2
        session_ids_seen = {e.session_id for e in events}
        assert session_a.id in session_ids_seen
        assert session_b.id in session_ids_seen

    async def test_split_night_ca_analysis_returns_ca_events_from_both_sessions(
        self, async_db_session
    ):
        """get_ca_analysis returns CA events from both split-night sessions."""
        therapy_date = date(2025, 12, 2)
        profile_id, dev, session_a, session_b = await self._setup_split_night(
            async_db_session, therapy_date
        )

        # Add one CA event to each session
        async_db_session.add_all(
            [
                models.Event(
                    session_id=session_a.id,
                    event_type="CA",
                    start_time=session_a.start_time + timedelta(minutes=15),
                    duration_seconds=20.0,
                ),
                models.Event(
                    session_id=session_b.id,
                    event_type="CA",
                    start_time=session_b.start_time + timedelta(minutes=20),
                    duration_seconds=15.0,
                ),
            ]
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        # Both sessions' CA events must be returned
        assert len(result.ca_events) == 2
        session_ids_seen = {e.session_id for e in result.ca_events}
        assert session_a.id in session_ids_seen
        assert session_b.id in session_ids_seen


# ---------------------------------------------------------------------------
# Input validation tests (Thufir pass-2 finding IMPORTANT-9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContextualEventsInputValidation:
    """Input validation for get_contextual_events."""

    async def test_contextual_events_invalid_event_types_raises(self, async_db_session):
        """get_contextual_events with event_types=123 raises ValueError."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 12, 10)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError, match="event_types"):
            await svc.get_contextual_events(
                therapy_date=therapy_date,
                device_id=dev.id,
                event_types=123,  # noqa: PGH003  # type mismatch is intentional for test
            )

    async def test_contextual_events_empty_string_in_event_types_raises(
        self, async_db_session
    ):
        """get_contextual_events with event_types=[""] raises ValueError."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 12, 11)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError, match="event_types"):
            await svc.get_contextual_events(
                therapy_date=therapy_date,
                device_id=dev.id,
                event_types=[""],
            )

    async def test_contextual_events_negative_min_duration_raises(
        self, async_db_session
    ):
        """get_contextual_events with min_duration=-1 raises ValueError."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 12, 12)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError, match="min_duration"):
            await svc.get_contextual_events(
                therapy_date=therapy_date,
                device_id=dev.id,
                min_duration=-1.0,
            )


# ---------------------------------------------------------------------------
# Nightly range date validation (Thufir pass-2 finding IMPORTANT-5)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNightlyRangeDateValidation:
    async def test_reversed_date_range_raises_value_error(self, async_db_session):
        """get_nightly_range_summary with date_end < date_start raises ValueError."""
        _, profile_id = await _make_profile(async_db_session)
        svc = BreathService(async_db_session, profile_id=profile_id)

        with pytest.raises(ValueError, match="date_end"):
            await svc.get_nightly_range_summary(
                date_start=date(2025, 4, 10),
                date_end=date(2025, 4, 1),  # reversed
            )


# ---------------------------------------------------------------------------
# Waveform blob builder helpers (shared with numeric provenance tests)
# ---------------------------------------------------------------------------


def _make_waveform_blob_from_arrays(
    timestamps: np.ndarray, values: np.ndarray
) -> bytes:
    """Build a valid waveform blob from caller-supplied timestamp/value arrays."""
    data = np.column_stack([timestamps.astype(np.float32), values.astype(np.float32)])
    return data.tobytes()


def _make_corrupt_waveform_blob() -> bytes:
    """Return a byte string that cannot be parsed as float32 pairs."""
    return b"\x00" * 3  # 3 bytes is not divisible by 8 (2×float32)


# ---------------------------------------------------------------------------
# Corrupt-blob propagation through public seams (Thufir pass-2 IMPORTANT-8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCorruptBlobThroughPublicSeams:
    """Corrupt waveform blobs must propagate as ValueError, not silently become
    NOT_AVAILABLE, through the get_contextual_events and get_ca_analysis seams."""

    async def test_corrupt_pressure_blob_raises_in_contextual_events(
        self, async_db_session
    ):
        """get_contextual_events re-raises ValueError when pressure blob is corrupt.

        plan IMPORTANT-8: sanitized invalid-waveform errors must propagate through
        public seams, not be swallowed as NOT_AVAILABLE.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 7, 1)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="pressure",
                sample_rate=1.0,
                sample_count=100,
                data_blob=_make_corrupt_waveform_blob(),
            )
        )
        # OA event at 120s → triggers ±5 s pressure waveform fetch
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="OA",
                start_time=session.start_time + timedelta(seconds=120),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError):
            await svc.get_contextual_events(therapy_date=therapy_date, device_id=dev.id)

    async def test_corrupt_mv_blob_raises_in_ca_analysis(self, async_db_session):
        """get_ca_analysis re-raises ValueError when MV blob is corrupt.

        plan IMPORTANT-8: same propagation requirement for CA analysis.
        CA event at offset > 0 triggers the per-event MV slope fetch.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 7, 2)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="mv",
                sample_rate=1.0,
                sample_count=100,
                data_blob=_make_corrupt_waveform_blob(),
            )
        )
        # CA at 300s offset → code enters `if offset_s > 0` and fetches MV
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=300),
                duration_seconds=12.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError):
            await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)


# ---------------------------------------------------------------------------
# Numeric provenance: CA fields must be nonzero when real data is present
# (Thufir pass-2 IMPORTANT-2: "test numeric fixtures whose expected values are nonzero")
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaNumericProvenance:
    """CA per-event and night-level fields must be numerically correct, not just
    structurally non-null, when real waveform/analysis data is present."""

    async def test_ca_pb_pct_nonzero_from_known_episodes(self, async_db_session):
        """periodic_breathing_pct is nonzero when analysis has persisted PB episodes.

        Session = 3600 s.  One episode: start_time=600 s, end_time=960 s (360 s).
        Expected: pb_pct = 360 / 3600 * 100 = 10.0 %.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 7, 10)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        episodes = [
            {
                "start_time": 600.0,
                "end_time": 960.0,
                "cycle_length": 30.0,
                "regularity_score": 0.9,
                "confidence": 0.95,
                "has_apneas": False,
            }
        ]
        result_dto = AnalysisResultDTO(
            session_id=session.id,
            session_duration_hours=1.0,
            total_breaths=0,
            machine_events=[],
            mode_results={
                "aasm": ModeResult(
                    mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                )
            },
            timestamp_start=session.start_time.timestamp(),
            timestamp_end=(session.start_time + timedelta(hours=1)).timestamp(),
            periodic_breathing_episodes=episodes,
        )
        from snore.analysis.service import AnalysisService as _AS  # noqa: PLC0415

        await _AS(async_db_session, profile_id=profile_id).store_result(
            AnalysisComputation(summary=result_dto, breaths=[], primary_mode="aasm"),
            processing_time_ms=10,
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=100),
                duration_seconds=8.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert result.periodic_breathing_pct is not None
        assert result.periodic_breathing_pct > 0.0
        # 360 / 3600 * 100 = 10.0 %  (±0.1 for float rounding)
        assert abs(result.periodic_breathing_pct - 10.0) < 0.1

    async def test_ca_mv_slope_nonzero_from_linear_ramp(self, async_db_session):
        """preceding_mv_slope ≈ 60.0 (L/min per minute) when MV = t (unit ramp).

        CA at offset 300 s.  MV window = [240, 300] s (60 s; plan §12 line 976).
        Linear regression of y=t on x=t → per-second slope = 1.0 L/min per s.
        After unit conversion (×60 s/min): preceding_mv_slope = 60.0 L/min per minute.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 7, 11)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        n = 3600
        ts = np.arange(n, dtype=np.float32)
        vals = np.arange(n, dtype=np.float32)  # y = x → slope = 1.0
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="mv",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(ts, vals),
            )
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=300),
                duration_seconds=12.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert len(result.ca_events) == 1
        ev = result.ca_events[0]
        assert ev.preceding_mv_slope is not None
        assert ev.preceding_mv_slope > 0.0
        # plan §12 line 976: slope in L/min per minute; unit ramp → 60.0 (tolerance ±1.0)
        assert abs(ev.preceding_mv_slope - 60.0) < 1.0

    async def test_ca_ps_nonzero_from_known_pressures(self, async_db_session):
        """ps_delivered_cmh2o ≈ 12.0 when THERAPY_PRESSURE=20.0, EPAP=8.0.

        PS = mean(THERAPY_PRESSURE − EPAP) over ±5 s window = 20.0 − 8.0 = 12.0.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 7, 12)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        n = 3600
        ts = np.arange(n, dtype=np.float32)
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="therapy_pressure",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(
                    ts, np.full(n, 20.0, dtype=np.float32)
                ),
            )
        )
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="epap",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(
                    ts, np.full(n, 8.0, dtype=np.float32)
                ),
            )
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=300),
                duration_seconds=12.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert len(result.ca_events) == 1
        ev = result.ca_events[0]
        assert ev.ps_delivered_cmh2o is not None
        assert abs(ev.ps_delivered_cmh2o - 12.0) < 0.5

    async def test_ca_mv_variance_nonzero_from_two_distinct_bins(
        self, async_db_session
    ):
        """mv_rolling_variance is nonzero when MV bins have different means.

        Session = 1200 s (exactly two 600-s bins, no tail bin).
        Bin 1 [0-600 s]: MV = 5.0.  Bin 2 [600-1200 s): MV = 15.0.
        variance([5.0, 15.0]) = 50.0.

        Session must have an OK AnalysisResult — eligibility gate skips NOT_RUN sessions.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2025, 7, 13)
        _, session = await _make_day_and_session(
            async_db_session,
            dev.id,
            therapy_date,
            duration_hours=1200.0 / 3600.0,
        )
        # Store an OK analysis result so this session passes the eligibility gate
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=1
        )

        n = 1200
        ts = np.arange(n, dtype=np.float32)
        vals = np.where(ts < 600.0, 5.0, 15.0).astype(np.float32)
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="mv",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(ts, vals),
            )
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=100),
                duration_seconds=8.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert result.mv_rolling_variance is not None
        assert result.mv_rolling_variance > 0.0
        # variance([5.0, 15.0]) = 50.0  (±1.0 tolerance for bin boundary effects)
        assert abs(result.mv_rolling_variance - 50.0) < 1.0


# ---------------------------------------------------------------------------
# Additional TestSameProfileTwoDevice tests (two-phase resolver)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSameProfileTwoDeviceExtended:
    """Extended two-device tests exercising the _resolve_range two-phase design."""

    async def _setup_two_devices_on_date(
        self, db: AsyncSession, therapy_date: date
    ) -> tuple[int, models.Device, models.Session, models.Device, models.Session]:
        """One profile with two devices, both having sessions on therapy_date."""
        _, profile_id = await _make_profile(db)
        dev_a = await _make_device(db, profile_id, manufacturer="ResMed")
        _, session_a = await _make_day_and_session(db, dev_a.id, therapy_date)
        dev_b = await _make_device(db, profile_id, manufacturer="Philips")
        _, session_b = await _make_day_and_session(db, dev_b.id, therapy_date)
        return profile_id, dev_a, session_a, dev_b, session_b

    async def test_two_device_compare_epochs_raises_device_ambiguity(
        self, async_db_session
    ):
        """compare_epochs with same-profile two devices and no device_id raises DeviceAmbiguityError."""
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        therapy_date = date(2026, 1, 15)
        profile_id, dev_a, _, dev_b, _ = await self._setup_two_devices_on_date(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(DeviceAmbiguityError) as exc_info:
            await svc.compare_epochs(
                epochs=[
                    EpochRequest(
                        label="ambiguous",
                        date_start=therapy_date,
                        date_end=therapy_date,
                        # device_id=None → ambiguity since two owned devices have sessions
                    )
                ]
            )
        err = exc_info.value
        assert dev_a.id in err.owned_device_ids
        assert dev_b.id in err.owned_device_ids

    async def test_two_device_disjoint_date_range_raises_device_ambiguity(
        self, async_db_session
    ):
        """Two devices with sessions on disjoint date ranges → DeviceAmbiguityError in compare_epochs."""
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev_a = await _make_device(async_db_session, profile_id, manufacturer="ResMed")
        dev_b = await _make_device(async_db_session, profile_id, manufacturer="Philips")
        date_a = date(2026, 1, 10)  # device A only
        date_b = date(2026, 2, 10)  # device B only
        await _make_day_and_session(async_db_session, dev_a.id, date_a)
        await _make_day_and_session(async_db_session, dev_b.id, date_b)

        svc = BreathService(async_db_session, profile_id=profile_id)
        # Range spans both dates — two distinct devices found → DeviceAmbiguityError
        with pytest.raises(DeviceAmbiguityError) as exc_info:
            await svc.compare_epochs(
                epochs=[
                    EpochRequest(
                        label="cross_month",
                        date_start=date_a,
                        date_end=date_b,
                    )
                ]
            )
        err = exc_info.value
        assert dev_a.id in err.owned_device_ids
        assert dev_b.id in err.owned_device_ids

    async def test_explicit_foreign_device_id_raises_in_epochs(self, async_db_session):
        """Foreign device_id in compare_epochs returns NOT_AVAILABLE (not NO_DATA_IN_RANGE)."""
        _, profile_a_id = await _make_profile(async_db_session)
        _, profile_b_id = await _make_profile(async_db_session)
        dev_b = await _make_device(async_db_session, profile_b_id)
        therapy_date = date(2026, 1, 20)
        await _make_day_and_session(async_db_session, dev_b.id, therapy_date)

        # Profile A asks for device_b (owned by profile B) → structured error, not NO_DATA_IN_RANGE
        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        result = await svc_a.compare_epochs(
            epochs=[
                EpochRequest(
                    label="foreign_device",
                    date_start=therapy_date,
                    date_end=therapy_date,
                    device_id=dev_b.id,
                )
            ]
        )
        assert len(result.epochs) == 1
        assert result.epochs[0].null_reason == NullReason.NOT_AVAILABLE

    async def test_two_device_nightly_range_raises_device_ambiguity(
        self, async_db_session
    ):
        """get_nightly_range_summary with two devices and no device_id raises DeviceAmbiguityError.

        Before the fix, the per-date loop resolved independently and mixed both devices.
        """
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        therapy_date = date(2026, 3, 10)
        profile_id, dev_a, _, dev_b, _ = await self._setup_two_devices_on_date(
            async_db_session, therapy_date
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(DeviceAmbiguityError) as exc_info:
            await svc.get_nightly_range_summary(
                date_start=therapy_date,
                date_end=therapy_date,
            )
        err = exc_info.value
        assert dev_a.id in err.owned_device_ids
        assert dev_b.id in err.owned_device_ids

    async def test_foreign_device_id_in_contextual_events_propagates(
        self, async_db_session
    ):
        """get_contextual_events with a foreign device_id propagates (not []).

        Before the fix, the ValueError was caught and silently returned [].
        """
        from snore.services.breath_service import DeviceNotOwnedError  # noqa: PLC0415

        _, profile_a_id = await _make_profile(async_db_session)
        _, profile_b_id = await _make_profile(async_db_session)
        dev_b = await _make_device(async_db_session, profile_b_id)
        therapy_date = date(2026, 3, 11)
        await _make_day_and_session(async_db_session, dev_b.id, therapy_date)

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        # profile A asking about profile B's device → DeviceNotOwnedError, not []
        with pytest.raises(DeviceNotOwnedError):
            await svc_a.get_contextual_events(
                therapy_date=therapy_date, device_id=dev_b.id
            )

    async def test_foreign_device_id_in_nightly_range_raises_device_not_owned(
        self, async_db_session
    ):
        """get_nightly_range_summary with a foreign device_id raises DeviceNotOwnedError.

        Before the fix, the ValueError was caught and a valid-looking empty summary
        was returned with device_id=<foreign>, n_nights=0, compliance_pct=0.0.
        """
        from snore.services.breath_service import DeviceNotOwnedError  # noqa: PLC0415

        _, profile_a_id = await _make_profile(async_db_session)
        _, profile_b_id = await _make_profile(async_db_session)
        dev_b = await _make_device(async_db_session, profile_b_id)
        therapy_date = date(2026, 3, 12)
        await _make_day_and_session(async_db_session, dev_b.id, therapy_date)

        svc_a = BreathService(async_db_session, profile_id=profile_a_id)
        with pytest.raises(DeviceNotOwnedError):
            await svc_a.get_nightly_range_summary(
                date_start=therapy_date,
                date_end=therapy_date,
                device_id=dev_b.id,
            )

    async def test_valid_device_with_no_sessions_in_range_returns_empty_summary(
        self, async_db_session
    ):
        """get_nightly_range_summary for an owned device with no sessions returns n_nights=0."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_nightly_range_summary(
            date_start=date(2026, 3, 1),
            date_end=date(2026, 3, 7),
            device_id=dev.id,
        )
        assert result.n_nights == 0
        assert result.nights == []

    async def test_two_separate_disjoint_epoch_requests_raise_device_ambiguity(
        self, async_db_session
    ):
        """Two separate EpochRequests spanning different devices raise DeviceAmbiguityError.

        This is the specific regression case from pass 5: two SEPARATE EpochRequest
        objects (not one spanning both dates) where device A owns epoch 1 dates and
        device B owns epoch 2 dates.  With no device_id, the union-resolve fires.
        """
        from snore.services.breath_service import DeviceAmbiguityError  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev_a = await _make_device(async_db_session, profile_id, manufacturer="ResMed")
        dev_b = await _make_device(async_db_session, profile_id, manufacturer="Philips")
        date_a = date(2026, 4, 1)  # device A only on this date
        date_b = date(2026, 5, 1)  # device B only on this date
        await _make_day_and_session(async_db_session, dev_a.id, date_a)
        await _make_day_and_session(async_db_session, dev_b.id, date_b)

        svc = BreathService(async_db_session, profile_id=profile_id)
        # Two separate, disjoint-date EpochRequests with no device_id
        with pytest.raises(DeviceAmbiguityError) as exc_info:
            await svc.compare_epochs(
                epochs=[
                    EpochRequest(
                        label="epoch_a",
                        date_start=date_a,
                        date_end=date_a,
                        # device_id=None → auto-select from union
                    ),
                    EpochRequest(
                        label="epoch_b",
                        date_start=date_b,
                        date_end=date_b,
                        # device_id=None → auto-select from union
                    ),
                ]
            )
        err = exc_info.value
        assert dev_a.id in err.owned_device_ids
        assert dev_b.id in err.owned_device_ids


# ---------------------------------------------------------------------------
# compare_epochs refusal tests (two-phase: RX + identity checks before breath queries)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompareEpochsRefusal:
    """Verify metadata-check semantics run before any breath queries.

    A mid-epoch RX change nulls only its own epoch (null_reason=RX_CHANGED_WITHIN_EPOCH);
    clean epochs still compute and the top-level null_reason is set only when EVERY
    epoch was nulled for an RX change.
    Cross-epoch algorithm identity mismatch is a non-blocking warning: distributions
    are still computed and version_warnings is populated instead of nulling everything.
    """

    async def _make_setting(
        self,
        db: AsyncSession,
        session_id: int,
        key: str,
        value: str,
    ) -> None:
        from snore.database import models as _models  # noqa: PLC0415

        db.add(_models.Setting(session_id=session_id, key=key, value=value))
        await db.flush()

    async def test_same_night_rx_divergence_refuses_with_null_distributions(
        self, async_db_session
    ):
        """Within-epoch RX change → null distributions for that epoch (RX_CHANGED_WITHIN_EPOCH).

        One device, two OK sessions on the same night with different PS values.
        """
        from snore.analysis.rx_tracker import RX_KEYS  # noqa: PLC0415
        from snore.database import models as _models  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 3, 1)

        start_a = datetime(
            therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
        )
        day = _models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        async_db_session.add(day)
        await async_db_session.flush()

        session_a = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a,
            end_time=start_a + timedelta(hours=3),
            duration_seconds=3 * 3600.0,
        )
        session_b = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a + timedelta(hours=4),
            end_time=start_a + timedelta(hours=7),
            duration_seconds=3 * 3600.0,
        )
        async_db_session.add_all([session_a, session_b])
        await async_db_session.flush()

        await _store_analysis_with_breaths(
            async_db_session, session_a, profile_id, n_breaths=2
        )
        await _store_analysis_with_breaths(
            async_db_session, session_b, profile_id, n_breaths=2
        )

        # Pick a key from RX_KEYS to vary between sessions
        rx_key = next(iter(RX_KEYS))
        await self._make_setting(async_db_session, session_a.id, rx_key, "8.0")
        await self._make_setting(
            async_db_session, session_b.id, rx_key, "12.0"
        )  # differs

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.compare_epochs(
            epochs=[
                EpochRequest(
                    label="same_night_rx",
                    date_start=therapy_date,
                    date_end=therapy_date,
                    device_id=dev.id,
                )
            ]
        )

        assert len(result.epochs) == 1
        epoch_result = result.epochs[0]
        # RX changed within epoch → refusal with null distributions
        assert epoch_result.null_reason == NullReason.RX_CHANGED_WITHIN_EPOCH
        assert epoch_result.mid_insp_flattening.median is None
        assert epoch_result.flatness_index.median is None
        assert result.null_reason == NullReason.RX_CHANGED_WITHIN_EPOCH

    async def test_one_violating_epoch_among_three_nulls_only_itself(
        self, async_db_session, monkeypatch
    ):
        """A mid-epoch RX change nulls only its own epoch; the two clean epochs
        still compute, and the top-level null_reason stays None."""
        from snore.analysis.rx_tracker import RX_KEYS  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        rx_key = next(iter(RX_KEYS))

        # Clean epoch 1: single session with breaths → computes normally.
        d1 = date(2026, 5, 1)
        _, s1 = await _make_day_and_session(async_db_session, dev.id, d1)
        await _store_analysis_with_breaths(
            async_db_session, s1, profile_id, n_breaths=3
        )

        # Violating epoch: two sessions on different days with divergent RX.
        d2, d3 = date(2026, 5, 10), date(2026, 5, 11)
        _, s2 = await _make_day_and_session(async_db_session, dev.id, d2)
        _, s3 = await _make_day_and_session(async_db_session, dev.id, d3)
        await _store_analysis_with_breaths(
            async_db_session, s2, profile_id, n_breaths=3
        )
        await _store_analysis_with_breaths(
            async_db_session, s3, profile_id, n_breaths=3
        )
        await self._make_setting(async_db_session, s2.id, rx_key, "8.0")
        await self._make_setting(async_db_session, s3.id, rx_key, "12.0")

        # Clean epoch 2: single session with breaths → computes normally.
        d4 = date(2026, 5, 20)
        _, s4 = await _make_day_and_session(async_db_session, dev.id, d4)
        await _store_analysis_with_breaths(
            async_db_session, s4, profile_id, n_breaths=3
        )

        waveform_session_ids: list[int] = []

        async def _record_waveform_session_ids(
            _db: AsyncSession, session_ids: list[int]
        ) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
            waveform_session_ids.extend(session_ids)
            return {}, {}

        monkeypatch.setattr(
            BreathService,
            "_fetch_waveform_channel_vals",
            staticmethod(_record_waveform_session_ids),
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.compare_epochs(
            epochs=[
                EpochRequest(
                    label="clean1", date_start=d1, date_end=d1, device_id=dev.id
                ),
                EpochRequest(
                    label="violating", date_start=d2, date_end=d3, device_id=dev.id
                ),
                EpochRequest(
                    label="clean2", date_start=d4, date_end=d4, device_id=dev.id
                ),
            ]
        )

        by_label = {e.label: e for e in result.epochs}
        # Only the violating epoch is nulled, with the RX reason on both axes.
        assert by_label["violating"].null_reason == NullReason.RX_CHANGED_WITHIN_EPOCH
        assert by_label["violating"].mid_insp_flattening.median is None
        assert by_label["violating"].rera_reason == NullReason.RX_CHANGED_WITHIN_EPOCH
        # Clean epochs compute normally.
        for lbl in ("clean1", "clean2"):
            assert by_label[lbl].null_reason is None
            assert by_label[lbl].mid_insp_flattening.median is not None
            assert by_label[lbl].flow_class_distribution
        # Not a whole-comparison refusal; violation still reported.
        assert result.null_reason is None
        assert len(result.rx_violations) == 1
        assert result.rx_violations[0].epoch_label == "violating"
        # Nulled epochs must not load and deserialize full-night waveform blobs.
        assert set(waveform_session_ids) == {s1.id, s4.id}

    async def test_all_epochs_violating_sets_top_level_refusal(self, async_db_session):
        """When every epoch has a mid-epoch RX change, the top-level null_reason is
        RX_CHANGED_WITHIN_EPOCH and every epoch is nulled."""
        from snore.analysis.rx_tracker import RX_KEYS  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        rx_key = next(iter(RX_KEYS))

        da1, da2 = date(2026, 6, 1), date(2026, 6, 2)
        _, sa1 = await _make_day_and_session(async_db_session, dev.id, da1)
        _, sa2 = await _make_day_and_session(async_db_session, dev.id, da2)
        await _store_analysis_with_breaths(
            async_db_session, sa1, profile_id, n_breaths=2
        )
        await _store_analysis_with_breaths(
            async_db_session, sa2, profile_id, n_breaths=2
        )
        await self._make_setting(async_db_session, sa1.id, rx_key, "8.0")
        await self._make_setting(async_db_session, sa2.id, rx_key, "12.0")

        db1, db2 = date(2026, 6, 10), date(2026, 6, 11)
        _, sb1 = await _make_day_and_session(async_db_session, dev.id, db1)
        _, sb2 = await _make_day_and_session(async_db_session, dev.id, db2)
        await _store_analysis_with_breaths(
            async_db_session, sb1, profile_id, n_breaths=2
        )
        await _store_analysis_with_breaths(
            async_db_session, sb2, profile_id, n_breaths=2
        )
        await self._make_setting(async_db_session, sb1.id, rx_key, "9.0")
        await self._make_setting(async_db_session, sb2.id, rx_key, "13.0")

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.compare_epochs(
            epochs=[
                EpochRequest(label="A", date_start=da1, date_end=da2, device_id=dev.id),
                EpochRequest(label="B", date_start=db1, date_end=db2, device_id=dev.id),
            ]
        )

        assert result.null_reason == NullReason.RX_CHANGED_WITHIN_EPOCH
        assert all(
            e.null_reason == NullReason.RX_CHANGED_WITHIN_EPOCH for e in result.epochs
        )
        assert len(result.rx_violations) == 2

    async def test_cross_epoch_identity_mismatch_warns_instead_of_refusing(
        self, async_db_session
    ):
        """Cross-epoch algorithm identity mismatch → version_warnings populated, NOT a hard refusal.

        Because _latest_analysis_for_session only returns OK when the stored identity
        matches the current runtime identity, genuine cross-epoch mismatches cannot be
        created through DB fixtures alone.  We mock _latest_analysis_for_session to
        inject different AlgoVersions (each individually OK) for two sessions in two
        different epochs, so the cross-epoch identity check fires.

        New semantics: result.null_reason is None (not ALGO_VERSION_MISMATCH);
        epoch null_reasons are None; version_warnings lists the differing fields.
        """
        import copy  # noqa: PLC0415

        from unittest.mock import patch  # noqa: PLC0415

        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AnalysisRunMetadata,
        )
        from snore.database import models as _models  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        date_a = date(2026, 4, 1)
        date_b = date(2026, 5, 1)

        _, session_a = await _make_day_and_session(async_db_session, dev.id, date_a)
        _, session_b = await _make_day_and_session(async_db_session, dev.id, date_b)

        # Construct two AlgoVersions with different identities on CROSS_VERSION_REFUSAL_KEYS
        current_id = AlgorithmIdentity.current()
        run_meta = AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"])
        algo_a = AlgoVersions(identity=current_id, run=run_meta)

        alt_id_dict = copy.deepcopy(current_id.model_dump())
        # Use "segmenter" — always a string field in CROSS_VERSION_REFUSAL_KEYS
        old_segmenter = alt_id_dict.get("segmenter", "v0")
        alt_id_dict["segmenter"] = (
            "v999.999.999" if old_segmenter != "v999.999.999" else "v998.0.0"
        )
        algo_b = AlgoVersions(
            identity=AlgorithmIdentity.model_validate(alt_id_dict), run=run_meta
        )

        # Insert real AnalysisResult rows so _resolve_range finds sessions; we'll
        # override the status via mock
        for sess in (session_a, session_b):
            async_db_session.add(
                _models.AnalysisResult(
                    session_id=sess.id,
                    timestamp_start=sess.start_time,
                    timestamp_end=sess.end_time or sess.start_time + timedelta(hours=7),
                    programmatic_result_json={},
                    processing_time_ms=5,
                    engine_versions_json=algo_a.model_dump(),  # both "current" in DB
                )
            )
        await async_db_session.flush()

        # Query real ar_ids to feed into the mock
        from sqlalchemy import select  # noqa: PLC0415

        ar_a_id = (
            await async_db_session.execute(
                select(_models.AnalysisResult.id)
                .where(_models.AnalysisResult.session_id == session_a.id)
                .limit(1)
            )
        ).scalar()
        ar_b_id = (
            await async_db_session.execute(
                select(_models.AnalysisResult.id)
                .where(_models.AnalysisResult.session_id == session_b.id)
                .limit(1)
            )
        ).scalar()

        async def _mocked_latest(
            session_id: int,
        ) -> tuple[AnalysisStatus, AlgoVersions | None, int | None]:
            if session_id == session_a.id:
                return (AnalysisStatus.OK, algo_a, ar_a_id)
            if session_id == session_b.id:
                return (AnalysisStatus.OK, algo_b, ar_b_id)
            return (AnalysisStatus.NOT_RUN, None, None)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with patch.object(
            svc, "_latest_analysis_for_session", side_effect=_mocked_latest
        ):
            result = await svc.compare_epochs(
                epochs=[
                    EpochRequest(
                        label="epoch_a",
                        date_start=date_a,
                        date_end=date_a,
                        device_id=dev.id,
                    ),
                    EpochRequest(
                        label="epoch_b",
                        date_start=date_b,
                        date_end=date_b,
                        device_id=dev.id,
                    ),
                ]
            )

        # Cross-epoch identity mismatch is now a warning, not a hard refusal.
        assert result.null_reason is None
        for es in result.epochs:
            assert es.null_reason is None
        # version_warnings must mention the differing field and both version strings
        assert len(result.version_warnings) > 0
        assert any("segmenter" in w for w in result.version_warnings)
        segmenter_warn = next(w for w in result.version_warnings if "segmenter" in w)
        assert "v999.999.999" in segmenter_warn

    async def test_primary_mode_mismatch_nulls_rera_fields(self, async_db_session):
        """Same identity but different primary_mode → rera_reason=PRIMARY_MODE_MISMATCH, rera_proxy_count=None.

        Two OK sessions in one epoch share the same algorithm identity (so
        distributions are computed) but have different primary_mode values.
        The RERA fields must be null with PRIMARY_MODE_MISMATCH while other
        distribution fields remain populated.
        """
        from datetime import timedelta  # noqa: PLC0415
        from unittest.mock import patch  # noqa: PLC0415

        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AnalysisRunMetadata,
        )
        from snore.database import models as _models  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 1)

        start_a = datetime(
            therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
        )
        day = _models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        async_db_session.add(day)
        await async_db_session.flush()

        session_a = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a,
            end_time=start_a + timedelta(hours=3),
            duration_seconds=3 * 3600.0,
        )
        session_b = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a + timedelta(hours=4),
            end_time=start_a + timedelta(hours=7),
            duration_seconds=3 * 3600.0,
        )
        async_db_session.add_all([session_a, session_b])
        await async_db_session.flush()

        # Insert real AnalysisResult rows so sessions are found; status injected via mock
        current_id = AlgorithmIdentity.current()
        run_meta_a = AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"])
        run_meta_b = AnalysisRunMetadata(
            primary_mode="aasm_relaxed", modes=["aasm_relaxed"]
        )
        algo_a = AlgoVersions(identity=current_id, run=run_meta_a)
        algo_b = AlgoVersions(identity=current_id, run=run_meta_b)

        for sess in (session_a, session_b):
            async_db_session.add(
                _models.AnalysisResult(
                    session_id=sess.id,
                    timestamp_start=sess.start_time,
                    timestamp_end=sess.end_time or sess.start_time + timedelta(hours=7),
                    programmatic_result_json={},
                    processing_time_ms=5,
                    engine_versions_json=algo_a.model_dump(),
                )
            )
        await async_db_session.flush()

        from sqlalchemy import select as _sel  # noqa: PLC0415

        ar_a_id = (
            await async_db_session.execute(
                _sel(_models.AnalysisResult.id)
                .where(_models.AnalysisResult.session_id == session_a.id)
                .limit(1)
            )
        ).scalar()
        ar_b_id = (
            await async_db_session.execute(
                _sel(_models.AnalysisResult.id)
                .where(_models.AnalysisResult.session_id == session_b.id)
                .limit(1)
            )
        ).scalar()

        async def _mocked_latest(
            session_id: int,
        ) -> tuple[AnalysisStatus, AlgoVersions | None, int | None]:
            if session_id == session_a.id:
                return (AnalysisStatus.OK, algo_a, ar_a_id)
            if session_id == session_b.id:
                return (AnalysisStatus.OK, algo_b, ar_b_id)
            return (AnalysisStatus.NOT_RUN, None, None)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with patch.object(
            svc, "_latest_analysis_for_session", side_effect=_mocked_latest
        ):
            result = await svc.compare_epochs(
                epochs=[
                    EpochRequest(
                        label="mixed_mode",
                        date_start=therapy_date,
                        date_end=therapy_date,
                        device_id=dev.id,
                    )
                ]
            )

        assert len(result.epochs) == 1
        epoch_result = result.epochs[0]
        # Distributions are computed (no identity mismatch)
        assert result.null_reason is None
        # RERA fields are nulled because primary_mode differs across sessions
        assert epoch_result.rera_reason == NullReason.PRIMARY_MODE_MISMATCH
        assert epoch_result.rera_proxy_count is None


# ---------------------------------------------------------------------------
# Additional CA numeric provenance tests (eligibility gate + variance)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaNumericProvenanceExtended:
    """Additional CA numeric tests exercising the eligibility gate and cross-session variance."""

    async def test_ca_stability_index_over_60s_window(self, async_db_session):
        """stability_index uses a 60-second window (plan §12 line 980).

        CA at 90 s.  The 60-second window covers [30, 90] s.
        Signal design: [0, 30) s = 0.0 (constant-zero region);
                       [30, 90] s = alternating 8.0/12.0 → mean≈10, stdev≈2, CV≈0.2.
        The old 120-second window [−30→0, 30, 90] includes the zero region, producing
        a very different CV (the zero values drag the mean down toward 0, causing
        stdev/mean → large or undefined).  This fixture falsifies the 120-second path.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 6, 1)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        # [0, 30) = 0.0; [30, 90] = alternating 8/12; rest = 0.0
        n = 3600
        ts = np.arange(n, dtype=np.float32)
        vals = np.where(
            (ts >= 30) & (ts <= 90),
            np.where(ts % 2 == 0, 8.0, 12.0),
            0.0,
        ).astype(np.float32)
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="mv",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(ts, vals),
            )
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=90),
                duration_seconds=8.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert len(result.ca_events) == 1
        ev = result.ca_events[0]
        # plan §12 line 980: stability_index = stdev / mean (CV) over 60-s window
        assert ev.stability_index is not None
        assert ev.stability_reason is None
        # alternating 8/12 in [30,90]: mean≈10, stdev≈2, CV≈0.2
        assert abs(ev.stability_index - 0.2) < 0.05

    async def test_ca_pb_pct_over_eligible_sessions_only(self, async_db_session):
        """PB% uses only OK-session durations in denominator (eligibility gate).

        Split night: 1 OK session (1800 s, 360 s of PB) + 1 NOT_RUN session (1800 s).
        pb_pct = 360 / 1800 * 100 = 20.0 % (NOT 10%, which would dilute with NOT_RUN).
        """
        from snore.database import models as _models  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 6, 2)

        start_a = datetime(
            therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
        )
        day = _models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        async_db_session.add(day)
        await async_db_session.flush()

        # session_a: 1800 s, gets an OK analysis with PB episodes (360 s total)
        session_a = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a,
            end_time=start_a + timedelta(seconds=1800),
            duration_seconds=1800.0,
        )
        # session_b: 1800 s, NOT_RUN (no AnalysisResult)
        session_b = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a + timedelta(hours=3),
            end_time=start_a + timedelta(hours=3) + timedelta(seconds=1800),
            duration_seconds=1800.0,
        )
        async_db_session.add_all([session_a, session_b])
        await async_db_session.flush()

        # Store OK analysis with PB episode for session_a
        from snore.analysis.modes.types import ModeResult  # noqa: PLC0415
        from snore.analysis.service import AnalysisService as _AS  # noqa: PLC0415
        from snore.analysis.types import AnalysisComputation  # noqa: PLC0415
        from snore.analysis.types import (
            AnalysisResult as AnalysisResultDTO,  # noqa: PLC0415
        )

        episodes = [
            {
                "start_time": 100.0,
                "end_time": 460.0,  # 360 s of PB
                "cycle_length": 30.0,
                "regularity_score": 0.9,
                "confidence": 0.95,
                "has_apneas": False,
            }
        ]
        result_dto = AnalysisResultDTO(
            session_id=session_a.id,
            session_duration_hours=1800.0 / 3600.0,
            total_breaths=0,
            machine_events=[],
            mode_results={
                "aasm": ModeResult(
                    mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                )
            },
            timestamp_start=session_a.start_time.timestamp(),
            timestamp_end=(session_a.start_time + timedelta(seconds=1800)).timestamp(),
            periodic_breathing_episodes=episodes,
        )
        await _AS(async_db_session, profile_id=profile_id).store_result(
            AnalysisComputation(summary=result_dto, breaths=[], primary_mode="aasm"),
            processing_time_ms=5,
        )

        # Add a CA event so we can call get_ca_analysis
        async_db_session.add(
            _models.Event(
                session_id=session_a.id,
                event_type="CA",
                start_time=session_a.start_time + timedelta(seconds=50),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert result.periodic_breathing_pct is not None
        # plan §12: pb_pct = 360 / 1800 * 100 = 20.0 % (NOT_RUN session excluded from denominator)
        assert abs(result.periodic_breathing_pct - 20.0) < 0.1

    async def test_ca_mv_variance_over_all_eligible_sessions(self, async_db_session):
        """Cross-session MV variance combines bin means from ALL OK sessions.

        Two OK sessions: session_a MV=5.0 (one 600-s bin), session_b MV=15.0 (one 600-s bin).
        Combined bin_means = [5.0, 15.0]; variance = 50.0.
        """
        from snore.database import models as _models  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 6, 3)

        start_a = datetime(
            therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
        )
        day = _models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        async_db_session.add(day)
        await async_db_session.flush()

        session_a = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a,
            end_time=start_a + timedelta(seconds=600),
            duration_seconds=600.0,
        )
        session_b = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a + timedelta(hours=2),
            end_time=start_a + timedelta(hours=2) + timedelta(seconds=600),
            duration_seconds=600.0,
        )
        async_db_session.add_all([session_a, session_b])
        await async_db_session.flush()

        # Both sessions get OK analysis
        await _store_analysis_with_breaths(
            async_db_session, session_a, profile_id, n_breaths=1
        )
        await _store_analysis_with_breaths(
            async_db_session, session_b, profile_id, n_breaths=1
        )

        # session_a MV = constant 5.0 for 600 s
        n = 600
        ts_a = np.arange(n, dtype=np.float32)
        async_db_session.add(
            _models.Waveform(
                session_id=session_a.id,
                waveform_type="mv",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(
                    ts_a, np.full(n, 5.0, dtype=np.float32)
                ),
            )
        )
        # session_b MV = constant 15.0 for 600 s
        ts_b = np.arange(n, dtype=np.float32)
        async_db_session.add(
            _models.Waveform(
                session_id=session_b.id,
                waveform_type="mv",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(
                    ts_b, np.full(n, 15.0, dtype=np.float32)
                ),
            )
        )
        # CA event in session_a for get_ca_analysis to find
        async_db_session.add(
            _models.Event(
                session_id=session_a.id,
                event_type="CA",
                start_time=session_a.start_time + timedelta(seconds=100),
                duration_seconds=8.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert result.mv_rolling_variance is not None
        # Combined bin_means = [5.0, 15.0] from both OK sessions → variance = 50.0
        assert abs(result.mv_rolling_variance - 50.0) < 1.0


# ---------------------------------------------------------------------------
# event_types cap tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestContextualEventsInputValidationExtended:
    """event_types cap tests (plan §13: 50-item limit)."""

    async def test_event_types_at_cap_does_not_raise(self, async_db_session):
        """get_contextual_events with exactly 50 unique event_types succeeds."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 1)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        event_types_50 = [f"TYPE_{i:03d}" for i in range(50)]
        svc = BreathService(async_db_session, profile_id=profile_id)
        # Should not raise — empty result since no events of those types exist
        result = await svc.get_contextual_events(
            therapy_date=therapy_date,
            device_id=dev.id,
            event_types=event_types_50,
        )
        assert result == []

    async def test_event_types_over_cap_raises_value_error(self, async_db_session):
        """get_contextual_events with 51 unique event_types raises ValueError.

        Silent truncation would drop valid requested types — the service must reject.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 2)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        event_types_51 = [f"TYPE_{i:03d}" for i in range(51)]
        svc = BreathService(async_db_session, profile_id=profile_id)
        with pytest.raises(ValueError, match="50"):
            await svc.get_contextual_events(
                therapy_date=therapy_date,
                device_id=dev.id,
                event_types=event_types_51,
            )


# ---------------------------------------------------------------------------
# Unexpected runtime error propagation tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnexpectedErrorPropagation:
    """Non-ValueError compute failures must propagate, not be swallowed."""

    async def test_unexpected_runtime_error_propagates_through_contextual_events(
        self, async_db_session
    ):
        """A RuntimeError from compute_waveform_window propagates out of get_contextual_events.

        Absent channels use missing_channels (no exception).  Corrupt blobs raise
        ValueError.  Any other unexpected exception must propagate rather than being
        swallowed by a catch-all.
        """
        from unittest.mock import patch  # noqa: PLC0415

        from snore.services import breath_service as bs_mod  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 8, 1)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="OA",
                start_time=session.start_time + timedelta(minutes=10),
                duration_seconds=5.0,
            )
        )
        await async_db_session.flush()

        def _raise_runtime(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected compute failure")

        svc = BreathService(async_db_session, profile_id=profile_id)
        with patch.object(
            bs_mod, "compute_waveform_window", side_effect=_raise_runtime
        ):
            with pytest.raises(RuntimeError, match="injected compute failure"):
                await svc.get_contextual_events(
                    therapy_date=therapy_date, device_id=dev.id
                )

    async def test_unexpected_runtime_error_propagates_through_ca_analysis(
        self, async_db_session
    ):
        """A RuntimeError from compute_waveform_window propagates out of get_ca_analysis."""
        from unittest.mock import patch  # noqa: PLC0415

        from snore.services import breath_service as bs_mod  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 8, 2)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=300),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        def _raise_runtime(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected CA compute failure")

        svc = BreathService(async_db_session, profile_id=profile_id)
        with patch.object(
            bs_mod, "compute_waveform_window", side_effect=_raise_runtime
        ):
            with pytest.raises(RuntimeError, match="injected CA compute failure"):
                await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)


# ---------------------------------------------------------------------------
# CA MIXED_VERSION refusal test (Thufir pass-4 acceptance item)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaMixedVersionRefusal:
    """get_ca_analysis refuses night-level fields on MIXED_VERSION day coverage."""

    async def test_mixed_version_refuses_ca_night_level_fields(self, async_db_session):
        """MIXED_VERSION day_status nulls pb_pct and mv_rolling_variance.

        Two sessions on the same night with different algorithm identities
        → day_status=MIXED_VERSION → periodic_breathing_pct is None
        with pb_reason=ALGO_VERSION_MISMATCH.

        plan §1 line 185: MIXED_VERSION is the first-wins state.
        """
        import copy  # noqa: PLC0415

        from unittest.mock import patch  # noqa: PLC0415

        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AnalysisRunMetadata,
        )
        from snore.database import models as _models  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 20)

        day = _models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        async_db_session.add(day)
        await async_db_session.flush()

        session_a = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
            ),
            end_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 23, 0
            ),
            duration_seconds=7200.0,
        )
        session_b = _models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 23, 30
            ),
            end_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day + 1, 1, 30
            ),
            duration_seconds=7200.0,
        )
        async_db_session.add_all([session_a, session_b])
        await async_db_session.flush()

        # Build two distinct algorithm identities
        current_id = AlgorithmIdentity.current()
        run_meta = AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"])
        algo_a = AlgoVersions(identity=current_id, run=run_meta)

        alt_id_dict = copy.deepcopy(current_id.model_dump())
        old_seg = alt_id_dict.get("segmenter", "v0")
        alt_id_dict["segmenter"] = "v999.0.0" if old_seg != "v999.0.0" else "v998.0.0"
        algo_b = AlgoVersions(
            identity=AlgorithmIdentity.model_validate(alt_id_dict), run=run_meta
        )

        for sess, algo in ((session_a, algo_a), (session_b, algo_b)):
            async_db_session.add(
                _models.AnalysisResult(
                    session_id=sess.id,
                    timestamp_start=sess.start_time,
                    timestamp_end=sess.end_time,
                    programmatic_result_json={},
                    processing_time_ms=5,
                    engine_versions_json=algo.model_dump(),
                )
            )
        await async_db_session.flush()

        from sqlalchemy import select as _select  # noqa: PLC0415

        ar_a_id = (
            await async_db_session.execute(
                _select(_models.AnalysisResult.id)
                .where(_models.AnalysisResult.session_id == session_a.id)
                .limit(1)
            )
        ).scalar()
        ar_b_id = (
            await async_db_session.execute(
                _select(_models.AnalysisResult.id)
                .where(_models.AnalysisResult.session_id == session_b.id)
                .limit(1)
            )
        ).scalar()

        async def _mocked_latest(
            session_id: int,
        ) -> tuple[AnalysisStatus, AlgoVersions | None, int | None]:
            if session_id == session_a.id:
                return (AnalysisStatus.OK, algo_a, ar_a_id)
            if session_id == session_b.id:
                return (AnalysisStatus.OK, algo_b, ar_b_id)
            return (AnalysisStatus.NOT_RUN, None, None)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with patch.object(
            svc, "_latest_analysis_for_session", side_effect=_mocked_latest
        ):
            result = await svc.get_ca_analysis(
                therapy_date=therapy_date, device_id=dev.id
            )

        # plan §1 line 185: MIXED_VERSION when sessions have distinct identities
        assert result.day_status == DayAnalysisStatus.MIXED_VERSION
        # plan §12 lines 984-993: MIXED_VERSION requires algorithm_identity=None
        assert result.algorithm_identity is None
        # plan §12 lines 984-993: top-level provenance is ALGO_VERSION_MISMATCH, not STALE
        assert result.null_reason == NullReason.ALGO_VERSION_MISMATCH
        # Night-level fields refused on MIXED_VERSION (plan §12 line 980)
        assert result.periodic_breathing_pct is None
        assert result.pb_reason == NullReason.ALGO_VERSION_MISMATCH
        assert result.mv_rolling_variance is None


# ---------------------------------------------------------------------------
# Code-review follow-up fixes (5 findings from parallel /code-review pass)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCodeReviewFixes:
    """Tests for the 5 code-review findings applied in one commit."""

    # --- Fix 1: unguarded window_end in get_contextual_events ---

    async def test_event_before_session_start_does_not_crash_contextual_events(
        self, async_db_session
    ):
        """get_contextual_events succeeds for events with offset_s < 0 (clock skew).

        Before the fix, window_end = offset_s + 5.0 had no lower bound.  When
        offset_s = -10.0, window_end = -5.0 fails Pydantic Field(gt=0.0) and
        aborted the entire call.  The fix guards window_end > 0; the event still
        appears in results with pressure_reason/leak_reason = NOT_AVAILABLE.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 9, 1)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        # Event 10 s BEFORE session start → offset_s = -10.0
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="OA",
                start_time=session.start_time - timedelta(seconds=10),
                duration_seconds=5.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        # Must not raise; old code would crash with Pydantic ValidationError
        results = await svc.get_contextual_events(
            therapy_date=therapy_date, device_id=dev.id
        )

        assert len(results) == 1
        ev = results[0]
        # Window guard: pressure/leak unavailable because window_end <= 0
        assert ev.pressure_reason == NullReason.NOT_AVAILABLE
        assert ev.leak_reason == NullReason.NOT_AVAILABLE

    # --- Fix 2: unguarded ps_win_end in get_ca_analysis ---

    async def test_ca_event_before_session_start_does_not_crash_ca_analysis(
        self, async_db_session
    ):
        """get_ca_analysis succeeds for CA events with offset_s < 0 (clock skew).

        Before the fix, ps_win_end = offset_s + 5.0 with no guard caused the
        same Pydantic crash when offset_s = -10.0.  The fix guards ps_win_end > 0
        and returns ps_delivered=None, ps_reason=NOT_AVAILABLE for that CA.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 9, 2)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        # CA event 10 s BEFORE session start
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time - timedelta(seconds=10),
                duration_seconds=8.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert len(result.ca_events) == 1
        ca = result.ca_events[0]
        # PS window guard: ps_win_end <= 0 → null with NOT_AVAILABLE
        assert ca.ps_delivered_cmh2o is None
        assert ca.ps_reason == NullReason.NOT_AVAILABLE

    # --- Fix 3: pre-loaded waveform blobs (single DB fetch per session) ---

    async def test_contextual_events_single_waveform_fetch_per_session(
        self, async_db_session
    ):
        """get_contextual_events calls _fetch_waveform_blobs exactly once per session.

        Before the fix, the function called _fetch_waveform_blobs twice per event
        (pressure/leak + MV), so N events = 2N blob reads.  The fix pre-loads
        once before the loop and slices in Python.
        """
        from unittest.mock import patch  # noqa: PLC0415

        import snore.services.breath_service as bs_mod  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 9, 3)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        # Two events in the session
        for i in range(2):
            async_db_session.add(
                models.Event(
                    session_id=session.id,
                    event_type="OA",
                    start_time=session.start_time + timedelta(minutes=10 + i * 10),
                    duration_seconds=5.0,
                )
            )
        await async_db_session.flush()

        call_count = 0
        original_fetch = bs_mod._fetch_waveform_blobs  # noqa: SLF001

        async def counting_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_fetch(*args, **kwargs)

        svc = BreathService(async_db_session, profile_id=profile_id)
        with patch.object(bs_mod, "_fetch_waveform_blobs", side_effect=counting_fetch):
            await svc.get_contextual_events(therapy_date=therapy_date, device_id=dev.id)

        # 1 pre-load per session (not 2 × N_events)
        assert call_count == 1, (
            f"Expected 1 _fetch_waveform_blobs call (pre-load) but got {call_count}; "
            "old code would call 2 × N_events"
        )

    # --- Fix 4: 90-night cap on get_nightly_range_summary ---

    async def test_nightly_range_91_nights_raises_value_error(self, async_db_session):
        """get_nightly_range_summary raises ValueError for ranges > 90 nights."""
        _, profile_id = await _make_profile(async_db_session)
        svc = BreathService(async_db_session, profile_id=profile_id)

        with pytest.raises(ValueError, match="90"):
            await svc.get_nightly_range_summary(
                date_start=date(2025, 1, 1),
                date_end=date(2025, 4, 2),  # 91 days
            )

    async def test_nightly_range_90_nights_does_not_raise(self, async_db_session):
        """get_nightly_range_summary succeeds for exactly 90 nights (cap boundary)."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        svc = BreathService(async_db_session, profile_id=profile_id)

        # 90-night range on an owned device → should not raise (may return empty)
        result = await svc.get_nightly_range_summary(
            date_start=date(2025, 1, 1),
            date_end=date(2025, 3, 31),  # exactly 90 days inclusive (90-1+1=90)
            device_id=dev.id,
        )
        assert result.n_calendar_nights == 90

    # --- Fix 5: pb_pct=0.0 false-positive when duration_seconds is NULL ---

    async def test_pb_pct_is_null_when_session_duration_is_null(self, async_db_session):
        """periodic_breathing_pct is null (not 0.0) when session.duration_seconds is NULL.

        Before the fix: total_eligible_s=0 but pb_seen_any=True → pb_pct=0.0, reason=None.
        After the fix: pb_pct=None, pb_reason=NOT_AVAILABLE.
        """
        from snore.analysis.service import AnalysisService as _AS  # noqa: PLC0415
        from snore.analysis.types import AnalysisComputation  # noqa: PLC0415

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 9, 4)

        # Session with NULL duration_seconds
        day = models.Day(device_id=dev.id, date=therapy_date, session_count=1)
        async_db_session.add(day)
        await async_db_session.flush()
        session = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day, 22, 0
            ),
            end_time=datetime(
                therapy_date.year, therapy_date.month, therapy_date.day + 1, 5, 0
            ),
            duration_seconds=None,  # NULL — the bug trigger
        )
        async_db_session.add(session)
        await async_db_session.flush()

        # Analysis result with known PB episodes
        episodes = [
            {
                "start_time": 100.0,
                "end_time": 400.0,
                "cycle_length": 30.0,
                "regularity_score": 0.9,
                "confidence": 0.95,
                "has_apneas": False,
            }
        ]
        result_dto = AnalysisResultDTO(
            session_id=session.id,
            session_duration_hours=7.0,
            total_breaths=0,
            machine_events=[],
            mode_results={
                "aasm": ModeResult(
                    mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                )
            },
            timestamp_start=session.start_time.timestamp(),
            timestamp_end=(session.start_time + timedelta(hours=7)).timestamp(),
            periodic_breathing_episodes=episodes,
        )
        await _AS(async_db_session, profile_id=profile_id).store_result(
            AnalysisComputation(summary=result_dto, breaths=[], primary_mode="aasm"),
            processing_time_ms=10,
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=200),
                duration_seconds=8.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        # NULL duration → cannot compute %; must be null not 0.0
        assert result.periodic_breathing_pct is None, (
            "pb_pct=0.0 (old bug) when session.duration_seconds is NULL; "
            "must be None with NOT_AVAILABLE reason"
        )
        assert result.pb_reason == NullReason.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# Ti / I:E ratio fields on NightlyAnalysisSummary
# ---------------------------------------------------------------------------


async def _store_night_with_breath_specs(
    db: AsyncSession,
    therapy_date: date,
    breath_specs: list[dict],
) -> tuple[int, int]:
    """Seed one night and return (profile_id, device_id).

    Each dict in ``breath_specs`` may set any ``ComputedBreath`` field;
    unset keys fall back to sensible defaults.  Supported overrides:
    ``inspiration_time_s``, ``i_e_ratio``, ``leak_valid``, ``flow_class``,
    ``flow_confidence``, ``peak_flow_lpm``, ``is_recovery_breath``.
    """
    from snore.analysis.types import ComputedBreath  # noqa: PLC0415

    _, profile_id = await _make_profile(db)
    dev = await _make_device(db, profile_id)
    _, session = await _make_day_and_session(db, dev.id, therapy_date)

    computed_breaths = []
    for i, spec in enumerate(breath_specs):
        computed_breaths.append(
            ComputedBreath(
                breath_number=i + 1,
                start_offset_s=float(i * 4),
                end_offset_s=float(i * 4 + 3),
                inspiration_time_s=spec.get("inspiration_time_s", 1.2),
                expiration_time_s=1.8,
                total_time_s=3.0,
                i_e_ratio=spec.get("i_e_ratio", 0.67),
                duty_cycle=0.4,
                peak_flow_lpm=spec.get("peak_flow_lpm", 30.0),
                peak_exp_flow_lpm=20.0,
                tidal_volume_ml=400.0,
                respiratory_rate_rolling=15.0,
                flatness_index=0.2,
                mid_insp_flattening=0.35,
                flow_class=spec.get("flow_class", 1),
                flow_confidence=spec.get("flow_confidence", 0.9),
                is_recovery_breath=spec.get("is_recovery_breath", False),
                inferred_trigger_type="normal",
                trigger_confidence=0.8,
                inferred_cycle_type="normal",
                cycle_confidence=0.75,
                trigger_cycle_applicable=True,
                trigger_cycle_reason=None,
                leak_valid=spec.get("leak_valid", True),
                leak_valid_reason=None,
                ramp_active=None,
                ramp_active_reason="not_available",
                mask_off=False,
                mask_off_reason=None,
            )
        )

    result_dto = AnalysisResultDTO(
        session_id=session.id,
        session_duration_hours=7.0,
        total_breaths=len(computed_breaths),
        machine_events=[],
        mode_results={
            "aasm": ModeResult(
                mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
            )
        },
        timestamp_start=session.start_time.timestamp(),
        timestamp_end=(session.start_time + timedelta(hours=7)).timestamp(),
    )
    computation = AnalysisComputation(
        summary=result_dto, breaths=computed_breaths, primary_mode="aasm"
    )
    await AnalysisService(db, profile_id=profile_id).store_result(
        computation, processing_time_ms=42
    )
    await db.flush()
    return profile_id, dev.id


@pytest.mark.unit
class TestNightlyRangeSummaryTiIe:
    """Ti (inspiration time) and I:E ratio fields on per-night NightlyAnalysisSummary."""

    async def test_ti_and_ie_median_odd_count_equals_middle_value(
        self, async_db_session
    ):
        """ti_median_s and ie_ratio_median are the exact middle value for odd-count leak-valid breaths."""
        therapy_date = date(2025, 7, 1)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"inspiration_time_s": 1.0, "i_e_ratio": 0.5, "leak_valid": True},
                {"inspiration_time_s": 1.2, "i_e_ratio": 0.6, "leak_valid": True},
                {"inspiration_time_s": 1.4, "i_e_ratio": 0.7, "leak_valid": True},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.ti_median_s == pytest.approx(1.2)
        assert night.ti_median_reason is None
        assert night.ie_ratio_median == pytest.approx(0.6)
        assert night.ie_ratio_reason is None

    async def test_ti_and_ie_median_even_count_averages_two_middle_values(
        self, async_db_session
    ):
        """ti_median_s and ie_ratio_median average the two middle values for even-count breaths."""
        therapy_date = date(2025, 7, 2)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"inspiration_time_s": 1.0, "i_e_ratio": 0.4, "leak_valid": True},
                {"inspiration_time_s": 1.2, "i_e_ratio": 0.6, "leak_valid": True},
                {"inspiration_time_s": 1.4, "i_e_ratio": 0.8, "leak_valid": True},
                {"inspiration_time_s": 1.6, "i_e_ratio": 1.0, "leak_valid": True},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.ti_median_s == pytest.approx(1.3)  # (1.2 + 1.4) / 2
        assert night.ti_median_reason is None
        assert night.ie_ratio_median == pytest.approx(0.7)  # (0.6 + 0.8) / 2
        assert night.ie_ratio_reason is None

    async def test_leak_invalid_breaths_excluded_from_ti_and_ie_medians(
        self, async_db_session
    ):
        """Breaths with leak_valid=False or leak_valid=None do not shift ti_median_s or ie_ratio_median."""
        therapy_date = date(2025, 7, 3)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                # Three qualifying breaths; median Ti=1.2, I:E=0.6
                {"inspiration_time_s": 1.0, "i_e_ratio": 0.5, "leak_valid": True},
                {"inspiration_time_s": 1.2, "i_e_ratio": 0.6, "leak_valid": True},
                {"inspiration_time_s": 1.4, "i_e_ratio": 0.7, "leak_valid": True},
                # Would shift the median if not filtered:
                {"inspiration_time_s": 9.9, "i_e_ratio": 9.9, "leak_valid": False},
                {"inspiration_time_s": 9.9, "i_e_ratio": 9.9, "leak_valid": None},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.ti_median_s == pytest.approx(1.2)
        assert night.ie_ratio_median == pytest.approx(0.6)

    async def test_null_ti_or_ie_within_leak_valid_breaths_excluded_independently(
        self, async_db_session
    ):
        """Null inspiration_time_s or i_e_ratio within leak-valid breaths are skipped per-field."""
        therapy_date = date(2025, 7, 4)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                # Ti present, I:E null → Ti contributes; I:E does not
                {"inspiration_time_s": 1.0, "i_e_ratio": None, "leak_valid": True},
                {"inspiration_time_s": 1.4, "i_e_ratio": None, "leak_valid": True},
                # Ti null, I:E present → I:E contributes; Ti does not
                {"inspiration_time_s": None, "i_e_ratio": 0.6, "leak_valid": True},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        # Ti from breaths 1 & 2 only: (1.0 + 1.4) / 2 = 1.2
        assert night.ti_median_s == pytest.approx(1.2)
        assert night.ti_median_reason is None
        # I:E from breath 3 only: 0.6
        assert night.ie_ratio_median == pytest.approx(0.6)
        assert night.ie_ratio_reason is None

    async def test_no_leak_valid_breaths_ti_and_ie_null_with_not_available_reason(
        self, async_db_session
    ):
        """All breaths leak_valid=False/None → ti_median_s and ie_ratio_median are None with NOT_AVAILABLE."""
        therapy_date = date(2025, 7, 5)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"inspiration_time_s": 1.2, "i_e_ratio": 0.6, "leak_valid": False},
                {"inspiration_time_s": 1.5, "i_e_ratio": 0.7, "leak_valid": None},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.ti_median_s is None
        assert night.ti_median_reason == NullReason.NOT_AVAILABLE
        assert night.ie_ratio_median is None
        assert night.ie_ratio_reason == NullReason.NOT_AVAILABLE

    async def test_zero_breaths_ti_and_ie_null_with_not_available_reason(
        self, async_db_session
    ):
        """A session with no breaths at all → ti_median_s and ie_ratio_median are None with NOT_AVAILABLE."""
        therapy_date = date(2025, 7, 6)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.ti_median_s is None
        assert night.ti_median_reason == NullReason.NOT_AVAILABLE
        assert night.ie_ratio_median is None
        assert night.ie_ratio_reason == NullReason.NOT_AVAILABLE


@pytest.mark.unit
class TestNightlySummaryFlClassGe4Pct:
    """fl_class_ge4_pct on per-night NightlyAnalysisSummary."""

    async def test_pct_counts_leak_valid_classified_breaths_only(
        self, async_db_session
    ):
        """Denominator = leak-valid breaths with a class; leak-invalid and unclassified breaths excluded."""
        therapy_date = date(2025, 7, 10)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                # Leak-valid classified: [5, 5, 1] → 2/3 with class >= 4
                {"flow_class": 5, "leak_valid": True},
                {"flow_class": 5, "leak_valid": True},
                {"flow_class": 1, "leak_valid": True},
                # Leak-valid but unclassified → excluded from denominator
                {"flow_class": None, "leak_valid": True},
                # Leak-INVALID class 6 → excluded entirely
                {"flow_class": 6, "leak_valid": False},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.fl_class_ge4_pct == pytest.approx(100.0 * 2 / 3)
        assert night.fl_class_ge4_pct_reason is None

    async def test_all_breaths_leak_invalid_pct_null_with_not_available_reason(
        self, async_db_session
    ):
        """All breaths leak_valid=False/None → fl_class_ge4_pct is None with NOT_AVAILABLE."""
        therapy_date = date(2025, 7, 11)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"flow_class": 5, "leak_valid": False},
                {"flow_class": 6, "leak_valid": None},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.fl_class_ge4_pct is None
        assert night.fl_class_ge4_pct_reason == NullReason.NOT_AVAILABLE

    async def test_fallback_confidence_breath_excluded_from_numerator_and_denominator(
        self, async_db_session
    ):
        """A fallback-confidence breath (== FL_DEFAULT_CONFIDENCE) is dropped from
        both numerator and denominator; only rule-matched breaths count."""
        therapy_date = date(2025, 7, 12)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                # Rule-matched class >= 4 → numerator and denominator
                {"flow_class": 5, "flow_confidence": 0.9},
                # Rule-matched class < 4 → denominator only
                {"flow_class": 1, "flow_confidence": 0.9},
                # Fallback guess sitting at the gate boundary → excluded from both
                {"flow_class": 5, "flow_confidence": FLC.FL_DEFAULT_CONFIDENCE},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        # Denominator 2 (the two 0.9-confidence breaths), numerator 1.  Counting
        # the fallback class-5 breath would give 2/3 instead.
        assert night.fl_class_ge4_pct == pytest.approx(50.0)
        assert night.fl_class_ge4_pct_reason is None

    async def test_null_confidence_breath_excluded_from_numerator_and_denominator(
        self, async_db_session
    ):
        """A breath with flow_confidence=None is dropped from both numerator and
        denominator (the gate requires a determinate confidence)."""
        therapy_date = date(2025, 7, 13)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"flow_class": 5, "flow_confidence": 0.9},
                {"flow_class": 1, "flow_confidence": 0.9},
                # No confidence recorded → excluded from both
                {"flow_class": 5, "flow_confidence": None},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.fl_class_ge4_pct == pytest.approx(50.0)
        assert night.fl_class_ge4_pct_reason is None


# ---------------------------------------------------------------------------
# leak_above_24_pct on NightlyAnalysisSummary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNightlySummaryLeakAbove24Pct:
    """leak_above_24_pct on per-night NightlyAnalysisSummary.

    Denominator = breaths with determinate leak_valid (True or False).
    Numerator   = breaths with leak_valid is False (leak > 24 L/min).
    None breaths are excluded from both.
    Result is rounded to 1 decimal.
    """

    async def test_mixed_leak_valid_values_pct_excludes_none(self, async_db_session):
        """3 True, 1 False, 2 None → denominator=4, numerator=1, pct=25.0."""
        therapy_date = date(2025, 7, 20)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"leak_valid": True},
                {"leak_valid": True},
                {"leak_valid": True},
                {"leak_valid": False},
                {"leak_valid": None},
                {"leak_valid": None},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.leak_above_24_pct == 25.0
        assert night.leak_above_24_pct_reason is None

    async def test_all_leak_valid_true_pct_is_zero(self, async_db_session):
        """All breaths leak_valid=True → 0 numerator → leak_above_24_pct == 0.0."""
        therapy_date = date(2025, 7, 21)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"leak_valid": True},
                {"leak_valid": True},
                {"leak_valid": True},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.leak_above_24_pct == 0.0
        assert night.leak_above_24_pct_reason is None

    async def test_all_leak_valid_none_pct_is_null_with_not_available_reason(
        self, async_db_session
    ):
        """All breaths leak_valid=None → denominator=0 → leak_above_24_pct is None with NOT_AVAILABLE."""
        therapy_date = date(2025, 7, 22)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                {"leak_valid": None},
                {"leak_valid": None},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=therapy_date,
            date_end=therapy_date,
            device_id=device_id,
        )

        night = summary.nights[0]
        assert night.leak_above_24_pct is None
        assert night.leak_above_24_pct_reason == NullReason.NOT_AVAILABLE


# ---------------------------------------------------------------------------
# rera_index, rdi, and DURATION_ZERO fields on NightlyAnalysisSummary
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNightlySummaryReraRdi:
    """rera_index, rdi, and NullReason.DURATION_ZERO on NightlyAnalysisSummary."""

    async def test_rera_index_computed_correctly_with_ahi_and_hours(
        self, async_db_session
    ):
        """rera_index = round(rera_count / hours, 2); rdi = round(ahi + rera_index, 2)."""
        from snore.analysis.types import ComputedBreath  # noqa: PLC0415

        therapy_date = date(2025, 8, 1)
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        start = datetime(therapy_date.year, therapy_date.month, therapy_date.day, 22, 0)
        day = models.Day(
            device_id=dev.id,
            date=therapy_date,
            session_count=1,
            total_therapy_hours=8.0,
            ahi=5.0,
        )
        async_db_session.add(day)
        await async_db_session.flush()
        session = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start,
            end_time=start + timedelta(hours=8),
            duration_seconds=8 * 3600.0,
        )
        async_db_session.add(session)
        await async_db_session.flush()

        # Two FL breaths (flow_class=5, run_len=2) + recovery breath → rera_count=1
        breaths = [
            ComputedBreath(
                breath_number=i + 1,
                start_offset_s=float(i * 4),
                end_offset_s=float(i * 4 + 3),
                inspiration_time_s=1.2,
                expiration_time_s=1.8,
                total_time_s=3.0,
                i_e_ratio=0.67,
                duty_cycle=0.4,
                peak_flow_lpm=30.0,
                peak_exp_flow_lpm=20.0,
                tidal_volume_ml=400.0,
                respiratory_rate_rolling=15.0,
                flatness_index=0.2,
                mid_insp_flattening=0.35,
                flow_class=5 if i < 2 else 1,
                flow_confidence=0.9,
                is_recovery_breath=(i == 2),
                inferred_trigger_type="normal",
                trigger_confidence=0.8,
                inferred_cycle_type="normal",
                cycle_confidence=0.75,
                trigger_cycle_applicable=True,
                trigger_cycle_reason=None,
                leak_valid=True,
                leak_valid_reason=None,
                ramp_active=None,
                ramp_active_reason="not_available",
                mask_off=False,
                mask_off_reason=None,
            )
            for i in range(3)
        ]
        result_dto = AnalysisResultDTO(
            session_id=session.id,
            session_duration_hours=8.0,
            total_breaths=3,
            machine_events=[],
            mode_results={
                "aasm": ModeResult(
                    mode_name="aasm", apneas=[], hypopneas=[], ahi=0.0, rdi=0.0
                )
            },
            timestamp_start=start.timestamp(),
            timestamp_end=(start + timedelta(hours=8)).timestamp(),
        )
        await AnalysisService(async_db_session, profile_id=profile_id).store_result(
            AnalysisComputation(
                summary=result_dto, breaths=breaths, primary_mode="aasm"
            ),
            processing_time_ms=42,
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_summary(therapy_date, device_id=dev.id)

        assert summary.rera_count == 1
        expected_rera_index = round(1 / 8.0, 2)
        assert summary.rera_index == pytest.approx(expected_rera_index)
        assert summary.rera_index_reason is None
        expected_rdi = round(5.0 + expected_rera_index, 2)
        assert summary.rdi == pytest.approx(expected_rdi)
        assert summary.rdi_reason is None

    async def test_rera_index_none_when_therapy_hours_zero(self, async_db_session):
        """rera_index is None with DURATION_ZERO reason when total_therapy_hours == 0."""
        therapy_date = date(2025, 8, 2)
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        start = datetime(therapy_date.year, therapy_date.month, therapy_date.day, 22, 0)
        day = models.Day(
            device_id=dev.id,
            date=therapy_date,
            session_count=1,
            total_therapy_hours=0.0,
        )
        async_db_session.add(day)
        await async_db_session.flush()
        session = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start,
            end_time=start + timedelta(hours=1),
            duration_seconds=3600.0,
        )
        async_db_session.add(session)
        await async_db_session.flush()
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=5
        )

        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_summary(therapy_date, device_id=dev.id)

        assert summary.rera_index is None
        assert summary.rera_index_reason == NullReason.DURATION_ZERO
        assert summary.rdi is None
        assert summary.rdi_reason == NullReason.DURATION_ZERO

    async def test_rera_proxy_version_null_when_analysis_not_run(
        self, async_db_session
    ):
        """Session without analysis → no RERA scan ran, so rera_count is None
        and rera_proxy_version must be None (not stamped 'v2')."""
        therapy_date = date(2025, 8, 3)
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        await _make_day_and_session(async_db_session, dev.id, therapy_date)

        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_summary(therapy_date, device_id=dev.id)

        assert summary.rera_count is None
        assert summary.rera_proxy_version is None

    async def test_range_summary_bulk_matches_per_night_summary(self, async_db_session):
        """get_nightly_range_summary over N nights produces identical per-night fields as get_nightly_summary."""
        therapy_dates = [date(2025, 8, 10), date(2025, 8, 11)]
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        for d in therapy_dates:
            _, session = await _make_day_and_session(async_db_session, dev.id, d)
            await _store_analysis_with_breaths(
                async_db_session, session, profile_id, n_breaths=10
            )

        svc = BreathService(async_db_session, profile_id=profile_id)
        range_summary = await svc.get_nightly_range_summary(
            date_start=therapy_dates[0],
            date_end=therapy_dates[-1],
            device_id=dev.id,
        )

        assert range_summary.n_nights == 2
        for i, d in enumerate(therapy_dates):
            per_night = await svc.get_nightly_summary(d, device_id=dev.id)
            bulk = range_summary.nights[i]
            assert bulk.therapy_date == per_night.therapy_date
            assert bulk.day_status == per_night.day_status
            assert bulk.analyzed_session_count == per_night.analyzed_session_count
            assert bulk.eligible_session_count == per_night.eligible_session_count
            assert bulk.rera_count == per_night.rera_count
            assert bulk.rera_reason == per_night.rera_reason
            assert bulk.rera_proxy_version == per_night.rera_proxy_version == "v2"
            assert bulk.fl_median == per_night.fl_median
            assert bulk.fl_95th == per_night.fl_95th
            assert bulk.fl_max == per_night.fl_max
            assert bulk.fl_class_ge4_pct == per_night.fl_class_ge4_pct
            assert bulk.fl_class_ge4_pct_reason == per_night.fl_class_ge4_pct_reason
            assert bulk.ti_median_s == per_night.ti_median_s
            assert bulk.ie_ratio_median == per_night.ie_ratio_median
            assert bulk.rera_index == per_night.rera_index
            assert bulk.rera_index_reason == per_night.rera_index_reason
            assert bulk.rdi == per_night.rdi
            assert bulk.rdi_reason == per_night.rdi_reason
            assert bulk.total_therapy_hours == per_night.total_therapy_hours
            assert bulk.is_compliant == per_night.is_compliant

    async def test_range_summary_skips_nights_without_sessions(self, async_db_session):
        """3-night range with empty middle night → n_nights == 2, middle absent from nights."""
        d1 = date(2025, 8, 20)
        d2 = date(2025, 8, 21)
        d3 = date(2025, 8, 22)
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        for d in (d1, d3):
            await _make_day_and_session(async_db_session, dev.id, d)
        # No session on d2

        svc = BreathService(async_db_session, profile_id=profile_id)
        summary = await svc.get_nightly_range_summary(
            date_start=d1, date_end=d3, device_id=dev.id
        )

        assert summary.n_nights == 2
        night_dates = {n.therapy_date for n in summary.nights}
        assert d1 in night_dates
        assert d3 in night_dates
        assert d2 not in night_dates


# ---------------------------------------------------------------------------
# RERA-proxy v2 — iter_fl_run_recoveries / _count_fl_run_reras
# ---------------------------------------------------------------------------


def _fl_row(
    flow_class: int | None,
    peak_flow_lpm: float | None = None,
    is_recovery_breath: bool = False,
) -> Any:
    """Minimal breath-row stub for the module-level FL-run scanner."""
    from types import SimpleNamespace  # noqa: PLC0415

    return SimpleNamespace(
        flow_class=flow_class,
        peak_flow_lpm=peak_flow_lpm,
        is_recovery_breath=is_recovery_breath,
    )


@pytest.mark.unit
class TestReraProxyV2Scanner:
    """Self-contained recovery criterion (b) + analysis-flag criterion (a)."""

    def test_amplitude_recovery_without_flag_counted(self):
        """Class-1 follower 30% above the run's mean peak flow counts without the flag."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=26.0, is_recovery_breath=False),
        ]
        assert _count_fl_run_reras(rows) == 1

    def test_follower_only_10pct_above_mean_not_counted(self):
        """Follower amplitude below the 20% margin does not count."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=22.0),
        ]
        assert _count_fl_run_reras(rows) == 0

    def test_follower_at_exact_margin_boundary_counted(self):
        """Follower at exactly (1 + margin) * run mean counts (>= comparison)."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=(1.0 + 0.20) * 20.0),
        ]
        assert _count_fl_run_reras(rows) == 1

    def test_custom_margin_kwarg_respected(self):
        """A smaller recovery_amplitude_margin admits a smaller amplitude rise."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=22.0),
        ]
        assert _count_fl_run_reras(rows, recovery_amplitude_margin=0.05) == 1

    def test_follower_class_3_high_amplitude_not_counted(self):
        """Class must drop to <= 2; a class-3 follower never satisfies criterion (b)."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(3, peak_flow_lpm=40.0),
        ]
        assert _count_fl_run_reras(rows) == 0

    def test_run_at_end_of_rows_not_counted(self):
        """A qualifying run with no follower cannot end in recovery."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
        ]
        assert _count_fl_run_reras(rows) == 0

    def test_fl_class_follower_extends_run_no_double_count(self):
        """A class >= threshold breath extends the run; the event counts once."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(4, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=26.0),
        ]
        assert _count_fl_run_reras(rows) == 1

    def test_null_follower_peak_flow_skips_criterion_b(self):
        """Follower with null peak_flow_lpm can only count via the recovery flag."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=None),
        ]
        assert _count_fl_run_reras(rows) == 0
        rows[-1].is_recovery_breath = True
        assert _count_fl_run_reras(rows) == 1

    def test_all_null_run_peaks_skip_criterion_b(self):
        """A run with no non-null peak_flow_lpm can only count via the recovery flag."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=None),
            _fl_row(5, peak_flow_lpm=None),
            _fl_row(1, peak_flow_lpm=100.0),
        ]
        assert _count_fl_run_reras(rows) == 0
        rows[-1].is_recovery_breath = True
        assert _count_fl_run_reras(rows) == 1

    def test_recovery_flag_still_counts_without_amplitude(self):
        """Criterion (a): the analysis-time flag counts regardless of amplitude."""
        from snore.services.breath_service import _count_fl_run_reras  # noqa: PLC0415

        rows = [
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=10.0, is_recovery_breath=True),
        ]
        assert _count_fl_run_reras(rows) == 1

    def test_iterator_yields_run_and_recovery_indices(self):
        """The scanner yields (run_start_idx, run_last_idx, recovery_idx)."""
        from snore.services.breath_service import (  # noqa: PLC0415
            iter_fl_run_recoveries,
        )

        rows = [
            _fl_row(1, peak_flow_lpm=30.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(5, peak_flow_lpm=20.0),
            _fl_row(1, peak_flow_lpm=26.0),
        ]
        assert list(iter_fl_run_recoveries(rows)) == [(1, 2, 3)]

    async def test_windows_and_count_identify_same_events(self, async_db_session):
        """_find_fl_run_windows and _count_fl_run_reras agree on the same rows."""
        therapy_date = date(2025, 9, 1)
        profile_id, device_id = await _store_night_with_breath_specs(
            async_db_session,
            therapy_date,
            breath_specs=[
                # Run 1 (amplitude recovery, no flag)
                {"flow_class": 5, "peak_flow_lpm": 20.0},
                {"flow_class": 5, "peak_flow_lpm": 20.0},
                {"flow_class": 1, "peak_flow_lpm": 26.0},
                # Separator
                {"flow_class": 1, "peak_flow_lpm": 20.0},
                # Run 2 (flagged recovery, low amplitude)
                {"flow_class": 4, "peak_flow_lpm": 20.0},
                {"flow_class": 4, "peak_flow_lpm": 20.0},
                {"flow_class": 4, "peak_flow_lpm": 20.0},
                {
                    "flow_class": 1,
                    "peak_flow_lpm": 10.0,
                    "is_recovery_breath": True,
                },
                # Run 3 — too short (length 1), never counts
                {"flow_class": 5, "peak_flow_lpm": 20.0},
                {"flow_class": 1, "peak_flow_lpm": 26.0},
            ],
        )
        svc = BreathService(async_db_session, profile_id=profile_id)

        windows_result = await svc.find_windows(
            therapy_date=therapy_date,
            criterion=WindowCriterion.FL_RUN_ENDING_IN_RECOVERY,
            n=50,
            device_id=device_id,
        )
        summary = await svc.get_nightly_summary(therapy_date, device_id=device_id)

        assert summary.rera_count == 2
        assert len(windows_result.windows) == summary.rera_count


@pytest.mark.unit
class TestDeriveMvFromFlow:
    """Pure-function tests for the flow-derived MV fallback."""

    def test_constant_positive_flow_yields_constant_mv(self):
        """Constant 12 L/min flow → every derived MV sample ≈ 12; first output
        lands exactly at offsets[0] + window_s."""
        offsets = np.arange(10.0, 310.0, 1.0)
        values = np.full_like(offsets, 12.0)

        out_t, out_v = derive_mv_from_flow(offsets, values)

        assert out_t.size > 0
        assert out_t[0] == pytest.approx(10.0 + 60.0)
        assert out_t[-1] <= offsets[-1]
        assert np.allclose(out_v, 12.0)

    def test_negative_flow_clipped_to_zero(self):
        """Flow alternating +10/−10 → negatives clip to 0 → derived MV ≈ 5."""
        offsets = np.arange(0.0, 200.0, 1.0)
        values = np.where(offsets % 2 == 0, 10.0, -10.0)

        _, out_v = derive_mv_from_flow(offsets, values)

        assert out_v.size > 0
        assert np.allclose(out_v, 5.0, atol=0.5)

    def test_timestamp_gap_omits_empty_window_samples(self):
        """Merged-session gap [101, 400) → output samples whose trailing window
        contains zero input samples are omitted, then resume after the gap."""
        offsets = np.concatenate([np.arange(0.0, 101.0), np.arange(400.0, 501.0)])
        values = np.full_like(offsets, 6.0)

        out_t, out_v = derive_mv_from_flow(offsets, values)

        # Grid is every 2 s from 60; windows [t-60, t] for t in [162, 398]
        # fall entirely inside the gap and must be omitted.
        in_gap = (out_t >= 162.0) & (out_t < 400.0)
        assert not in_gap.any()
        # Samples resume once the window reaches post-gap data
        assert (out_t >= 400.0).any()
        assert np.allclose(out_v, 6.0)

    def test_input_shorter_than_window_returns_empty(self):
        """Input spanning < window_s seconds cannot fill one window → empty."""
        offsets = np.arange(0.0, 30.0, 1.0)
        values = np.full_like(offsets, 12.0)

        out_t, out_v = derive_mv_from_flow(offsets, values)

        assert out_t.size == 0
        assert out_v.size == 0

    def test_empty_input_returns_empty(self):
        out_t, out_v = derive_mv_from_flow(np.array([]), np.array([]))

        assert out_t.size == 0
        assert out_v.size == 0

    def test_non_monotonic_offsets_return_empty(self):
        """Offsets with an inversion would produce garbage windows via
        searchsorted → empty arrays (downstream metrics go null)."""
        offsets = np.concatenate([np.arange(0.0, 100.0), np.arange(50.0, 200.0)])
        values = np.full_like(offsets, 8.0)

        out_t, out_v = derive_mv_from_flow(offsets, values)

        assert out_t.size == 0
        assert out_v.size == 0

    def test_nan_values_treated_as_zero(self):
        """NaN flow samples count as 0.0 flow instead of poisoning the
        cumulative sum: alternating 10/NaN → derived MV ≈ 5, all finite."""
        offsets = np.arange(0.0, 200.0, 1.0)
        values = np.full_like(offsets, 10.0)
        values[::2] = np.nan

        out_t, out_v = derive_mv_from_flow(offsets, values)

        assert out_t.size > 0
        assert np.all(np.isfinite(out_v))
        assert np.allclose(out_v, 5.0, atol=0.5)


@pytest.mark.unit
class TestCaMvFlowFallback:
    """CA analysis derives MV from flow when no device MV channel exists."""

    async def test_flow_only_session_derives_mv_metrics(self, async_db_session):
        """No mv waveform, flow seeded → slope/stability/variance non-null and
        mv_source == 'flow_derived' at both event and night level.

        Flow: 5 L/min for [0, 600) s, 15 L/min for [600, 1200) s (all positive,
        so derived MV mirrors the flow levels) → two 600-s bins with distinct
        means → mv_rolling_variance > 0.  CA at 300 s: derived MV in the
        window [240, 300] is constant 5.0 → slope 0.0, CV 0.0 (non-null).
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 1)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1200.0 / 3600.0
        )
        # OK analysis → session passes the night-level eligibility gate
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=1
        )

        n = 1200
        ts = np.arange(n, dtype=np.float32)
        vals = np.where(ts < 600.0, 5.0, 15.0).astype(np.float32)
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="flow",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(ts, vals),
            )
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=300),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        raw = await svc.fetch_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        # Fetch side: MV absent → FLOW blobs fetched for the fallback
        assert raw.session_data[0].flow_waveform is not None

        from snore.services.breath_service import compute_ca_analysis  # noqa: PLC0415

        result = compute_ca_analysis(raw)

        assert len(result.ca_events) == 1
        ev = result.ca_events[0]
        assert ev.mv_source == "flow_derived"
        assert ev.preceding_mv_slope is not None
        assert ev.preceding_mv_reason is None
        assert ev.stability_index is not None
        assert ev.stability_reason is None

        assert result.mv_source == "flow_derived"
        assert result.mv_fallback_version == "v1"
        assert result.mv_rolling_variance is not None
        assert result.mv_rolling_variance > 0.0
        assert result.mv_variance_reason is None

    async def test_device_mv_session_skips_flow_fetch(self, async_db_session):
        """Device mv waveform present → mv_source == 'device' and the FLOW
        blob is never fetched (flow_waveform is None on the raw data)."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 2)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )

        n = 3600
        ts = np.arange(n, dtype=np.float32)
        async_db_session.add(
            models.Waveform(
                session_id=session.id,
                waveform_type="mv",
                sample_rate=1.0,
                sample_count=n,
                data_blob=_make_waveform_blob_from_arrays(
                    ts, np.full(n, 8.0, dtype=np.float32)
                ),
            )
        )
        async_db_session.add(
            models.Event(
                session_id=session.id,
                event_type="CA",
                start_time=session.start_time + timedelta(seconds=300),
                duration_seconds=10.0,
            )
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        raw = await svc.fetch_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        # Device MV present → no needless FLOW fetch
        assert raw.session_data[0].flow_waveform is None

        from snore.services.breath_service import compute_ca_analysis  # noqa: PLC0415

        result = compute_ca_analysis(raw)

        assert len(result.ca_events) == 1
        assert result.ca_events[0].mv_source == "device"
        assert result.mv_source == "device"

    async def test_mixed_mv_sources_yield_mixed_night_source(self, async_db_session):
        """Two sessions on one night — one with a device mv waveform, one with
        only flow → per-event mv_source is 'device' / 'flow_derived' per
        session and the night-level mv_source is 'mixed'."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 4)

        start_a = datetime(
            therapy_date.year, therapy_date.month, therapy_date.day, 21, 0
        )
        day = models.Day(device_id=dev.id, date=therapy_date, session_count=2)
        async_db_session.add(day)
        await async_db_session.flush()
        session_a = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a,
            end_time=start_a + timedelta(seconds=1200),
            duration_seconds=1200.0,
        )
        session_b = models.Session(
            device_id=dev.id,
            day_id=day.id,
            device_session_id=f"SESS_{uuid.uuid4().hex[:8]}",
            start_time=start_a + timedelta(hours=2),
            end_time=start_a + timedelta(hours=2, seconds=1200),
            duration_seconds=1200.0,
        )
        async_db_session.add_all([session_a, session_b])
        await async_db_session.flush()

        n = 1200
        ts = np.arange(n, dtype=np.float32)
        # Session A: device mv channel; session B: flow only (fallback path)
        async_db_session.add_all(
            [
                models.Waveform(
                    session_id=session_a.id,
                    waveform_type="mv",
                    sample_rate=1.0,
                    sample_count=n,
                    data_blob=_make_waveform_blob_from_arrays(
                        ts, np.full(n, 8.0, dtype=np.float32)
                    ),
                ),
                models.Waveform(
                    session_id=session_b.id,
                    waveform_type="flow",
                    sample_rate=1.0,
                    sample_count=n,
                    data_blob=_make_waveform_blob_from_arrays(
                        ts, np.full(n, 12.0, dtype=np.float32)
                    ),
                ),
                models.Event(
                    session_id=session_a.id,
                    event_type="CA",
                    start_time=session_a.start_time + timedelta(seconds=300),
                    duration_seconds=10.0,
                ),
                models.Event(
                    session_id=session_b.id,
                    event_type="CA",
                    start_time=session_b.start_time + timedelta(seconds=300),
                    duration_seconds=10.0,
                ),
            ]
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert len(result.ca_events) == 2
        source_by_session = {ev.session_id: ev.mv_source for ev in result.ca_events}
        assert source_by_session[session_a.id] == "device"
        assert source_by_session[session_b.id] == "flow_derived"
        assert result.mv_source == "mixed"

    async def test_pb_zero_episodes_yields_zero_pct(self, async_db_session):
        """Analyzed-OK session with pb_json present but zero PB episodes →
        periodic_breathing_pct == 0.0 with reason None (not null+not_available)."""
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)
        therapy_date = date(2026, 7, 3)
        _, session = await _make_day_and_session(
            async_db_session, dev.id, therapy_date, duration_hours=1.0
        )
        # store_result persists programmatic_result_json (no PB episodes)
        await _store_analysis_with_breaths(
            async_db_session, session, profile_id, n_breaths=1
        )
        await async_db_session.flush()

        svc = BreathService(async_db_session, profile_id=profile_id)
        result = await svc.get_ca_analysis(therapy_date=therapy_date, device_id=dev.id)

        assert result.periodic_breathing_pct == 0.0
        assert result.pb_reason is None


# ---------------------------------------------------------------------------
# _fetch_waveform_channel_vals — chunked ID binding (#280)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFetchWaveformChannelValsChunked:
    async def test_fetch_waveform_channel_vals_chunked(
        self, async_db_session, monkeypatch
    ):
        """fl/snore values survive the ID-chunk boundary in _fetch_waveform_channel_vals.

        With ID_CHUNK_SIZE=2 and three sessions each carrying an fl AND a snore
        waveform, the session_ids span two chunks.  Both per-session dicts must
        return every session on both channels — a bug that dropped any non-final
        chunk would leave a session out.
        """
        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)

        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        session_ids: list[int] = []
        expected_fl: dict[int, list[float]] = {}
        expected_snore: dict[int, list[float]] = {}
        ts = np.arange(3, dtype=np.float32)
        for i in range(3):
            _, session = await _make_day_and_session(
                async_db_session, dev.id, date(2025, 8, 1) + timedelta(days=i)
            )
            fl_vals = np.array([1.0 + i, 2.0 + i, 3.0 + i], dtype=np.float32)
            snore_vals = np.array([10.0 + i, 20.0 + i, 30.0 + i], dtype=np.float32)
            async_db_session.add_all(
                [
                    models.Waveform(
                        session_id=session.id,
                        waveform_type="fl",
                        sample_rate=1.0,
                        sample_count=3,
                        data_blob=_make_waveform_blob_from_arrays(ts, fl_vals),
                    ),
                    models.Waveform(
                        session_id=session.id,
                        waveform_type="snore",
                        sample_rate=1.0,
                        sample_count=3,
                        data_blob=_make_waveform_blob_from_arrays(ts, snore_vals),
                    ),
                ]
            )
            session_ids.append(session.id)
            expected_fl[session.id] = [float(v) for v in fl_vals]
            expected_snore[session.id] = [float(v) for v in snore_vals]
        await async_db_session.flush()

        fl_by_sess, snore_by_sess = await BreathService._fetch_waveform_channel_vals(
            async_db_session, session_ids
        )

        assert set(fl_by_sess) == set(session_ids)
        assert set(snore_by_sess) == set(session_ids)
        for sid in session_ids:
            assert fl_by_sess[sid] == pytest.approx(expected_fl[sid])
            assert snore_by_sess[sid] == pytest.approx(expected_snore[sid])


# ---------------------------------------------------------------------------
# get_nightly_range_summary — chunked Day / AnalysisResult / Breath binds (#280)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNightlyRangeSummaryChunked:
    async def test_multichunk_range_summary_equals_single_chunk(
        self, async_db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunking the nightly range binds does not change the summary.

        Seeds five analysed nights so the date IN-list (Day query), the
        AnalysisResult-id IN-list (_classify_sessions_bulk), and the ok_ar_id
        IN-list (Breath query) each span three chunks at ID_CHUNK_SIZE=2.  The
        summary produced at size 2 must equal the single-chunk (default) run —
        proving the chunk boundary drops no night, metric, or Day linkage.
        """
        _, profile_id = await _make_profile(async_db_session)
        dev = await _make_device(async_db_session, profile_id)

        date_start = date(2025, 9, 1)
        n_days = 5  # 5 dates / 5 ar ids / 5 ok ar ids → 3 chunks each at size 2
        for i in range(n_days):
            day, session = await _make_day_and_session(
                async_db_session, dev.id, date_start + timedelta(days=i)
            )
            # Distinct day-level total so day linkage (not the session-duration
            # fallback of 7.0) is observable per night — a dropped Day chunk
            # would surface the fallback instead.
            day.total_therapy_hours = 5.5
            await _store_analysis_with_breaths(
                async_db_session, session, profile_id, n_breaths=6
            )
        await async_db_session.flush()

        date_end = date_start + timedelta(days=n_days - 1)

        # Single-chunk baseline first (default ID_CHUNK_SIZE).
        baseline = await BreathService(
            async_db_session, profile_id=profile_id
        ).get_nightly_range_summary(
            date_start=date_start, date_end=date_end, device_id=dev.id
        )

        # Force multi-chunk execution and re-run against the same seeded data.
        monkeypatch.setattr("snore.utils.db_chunk.ID_CHUNK_SIZE", 2)
        chunked = await BreathService(
            async_db_session, profile_id=profile_id
        ).get_nightly_range_summary(
            date_start=date_start, date_end=date_end, device_id=dev.id
        )

        assert chunked.model_dump() == baseline.model_dump()

        # Concrete behaviour: every seeded night is present, analysed, compliant,
        # carries breath-derived metrics, and reflects the linked Day total.
        assert chunked.n_nights == n_days
        assert chunked.days_compliant == n_days
        assert chunked.compliance_pct == pytest.approx(100.0)
        assert len(chunked.nights) == n_days
        for night in chunked.nights:
            assert night.day_status == DayAnalysisStatus.OK
            assert night.analyzed_session_count == 1
            assert night.fl_median == pytest.approx(0.35)
            assert night.total_therapy_hours == pytest.approx(5.5)
