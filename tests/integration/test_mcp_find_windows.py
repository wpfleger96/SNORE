"""Integration tests for the find_windows MCP tool.

Calls ``snore.mcp.tools.windows.find_windows`` directly (not via the MCP
client) against a real SQLite database populated with seeded rows.  Covers:

- worst_flattening_leak_valid: windows ordered worst-first; leak-invalid
  breaths never anchor a window.
- Mixed algorithm identities: FL-ranked criteria refuse, CA criterion still
  returns windows.
- ca_centered with no analysis: windows anchored at event offsets, per-window
  analysis_status="not_run".
- fl_run_ending_in_recovery: FL run ending in recovery breath → window found;
  mixed primary_mode → null_reason primary_mode_mismatch.
- Criterion-irrelevant option: ValueError naming the offending field.
- Isolation (two profiles, adversarial): profile A sees empty result for B's
  date; A passing B's device_id → ValidationError.
"""

from __future__ import annotations

import uuid

from datetime import date, datetime, timedelta
from typing import Any

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import (
    AnalysisResult,
    Breath,
    Day,
    Device,
    Event,
    Session,
)
from snore.mcp.errors import ValidationError
from tests.integration.conftest import (
    _make_day_session,
    _make_device,
    _make_profile,
)

# ---------------------------------------------------------------------------
# _make_analysis_result is defined locally: it accepts a ``segmenter`` override
# for mixed-version tests, which the shared conftest version does not support.
# ---------------------------------------------------------------------------


async def _add_session_to_day(
    db: AsyncSession,
    device: Device,
    day: Day,
    start_hour: int = 2,
    duration_hours: float = 2.0,
) -> Session:
    """Add a second Session to an existing Day (avoids Day UNIQUE constraint)."""
    day_date = day.date
    start_dt = datetime(day_date.year, day_date.month, day_date.day, start_hour, 0, 0)
    sess = Session(
        device_id=device.id,
        day_id=day.id,
        device_session_id=f"fw2_{day_date.isoformat()}_{uuid.uuid4().hex[:6]}",
        start_time=start_dt,
        end_time=start_dt + timedelta(hours=duration_hours),
        duration_seconds=duration_hours * 3600,
        enabled=True,
    )
    db.add(sess)
    await db.flush()
    return sess


async def _make_analysis_result(
    db: AsyncSession,
    session: Session,
    primary_mode: str = "aasm",
    segmenter: str | None = None,
) -> AnalysisResult:
    """Create an AnalysisResult with current algorithm identity.

    Pass ``segmenter`` to override the identity's segmenter field, creating a
    distinct identity for mixed-version tests.
    """
    from snore.analysis.shared.versioning import (  # noqa: PLC0415
        AlgorithmIdentity,
        AlgoVersions,
        AnalysisRunMetadata,
    )

    identity = AlgorithmIdentity.current()
    if segmenter is not None:
        identity = AlgorithmIdentity(
            format_version=identity.format_version,
            segmenter=segmenter,
            fl_classifier=identity.fl_classifier,
            flattening=identity.flattening,
            trigger_cycle=identity.trigger_cycle,
            leak_valid=identity.leak_valid,
            recovery_detector=identity.recovery_detector,
        )

    algo_versions = AlgoVersions(
        identity=identity,
        run=AnalysisRunMetadata(primary_mode=primary_mode, modes=[primary_mode]),
    )
    ar = AnalysisResult(
        session_id=session.id,
        timestamp_start=session.start_time,
        timestamp_end=session.end_time,
        engine_versions_json=algo_versions.model_dump(),
    )
    db.add(ar)
    await db.flush()
    return ar


async def _make_breath(
    db: AsyncSession,
    ar: AnalysisResult,
    session: Session,
    breath_number: int,
    mid_insp_flattening: float | None = None,
    leak_valid: bool | None = True,
    flow_class: int | None = None,
    is_recovery_breath: bool | None = None,
) -> Breath:
    start_offset = float(breath_number) * 5.0
    breath = Breath(
        analysis_result_id=ar.id,
        session_id=session.id,
        breath_number=breath_number,
        start_offset_s=start_offset,
        end_offset_s=start_offset + 4.0,
        mid_insp_flattening=mid_insp_flattening,
        leak_valid=leak_valid,
        flow_class=flow_class,
        is_recovery_breath=is_recovery_breath,
    )
    db.add(breath)
    return breath


# ---------------------------------------------------------------------------
# TestWorstFlatteningLeakValid
# ---------------------------------------------------------------------------


class TestWorstFlatteningLeakValid:
    async def test_windows_ordered_worst_first_leak_invalid_excluded(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """worst_flattening_leak_valid windows sorted by flattening ascending
        (lower = more flow-limited); a breath with leak_valid=False cannot anchor
        a window even if its mid_insp_flattening is the lowest."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 1, 15)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)

        # Breath 6 (index 6): lowest flattening BUT leak_valid=False → not an anchor
        # Breath 4 (index 4): second-lowest → anchor for window 1
        # Breath 9 (index 9): third-lowest → anchor for window 2
        for i in range(10):
            if i == 6:
                mid_fl, lv = 0.20, False
            elif i == 4:
                mid_fl, lv = 0.35, True
            elif i == 9:
                mid_fl, lv = 0.55, True
            else:
                mid_fl, lv = 1.00, True
            await _make_breath(
                async_db_session,
                ar,
                sess,
                breath_number=i,
                mid_insp_flattening=mid_fl,
                leak_valid=lv,
            )

        await async_db_session.flush()
        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="worst_flattening_leak_valid",
            n=5,
        )

        assert result.null_reason is None
        assert result.day_status == "ok"
        assert len(result.windows) == 2

        # Windows sorted worst-first by mid_insp_flattening
        assert result.windows[0].worst_mid_insp_flattening == pytest.approx(0.35)
        assert result.windows[1].worst_mid_insp_flattening == pytest.approx(0.55)

        # Every window has OK analysis status
        for w in result.windows:
            assert w.analysis_status == "ok"

    async def test_leak_invalid_breath_not_in_anchor_set(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """The leak-invalid breath with the lowest flattening is absent from
        the anchor set."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 1, 16)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)

        for i in range(10):
            mid_fl = 0.20 if i == 6 else (0.40 if i == 4 else 1.00)
            lv: bool | None = False if i == 6 else True
            await _make_breath(
                async_db_session,
                ar,
                sess,
                breath_number=i,
                mid_insp_flattening=mid_fl,
                leak_valid=lv,
            )

        await async_db_session.flush()
        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="worst_flattening_leak_valid",
            n=5,
        )

        # With breath 6 excluded, only breath 4 has a notably low flattening.
        # Other breaths have flattening=1.00 and also anchor windows when eligible.
        assert result.null_reason is None
        assert any(
            w.worst_mid_insp_flattening == pytest.approx(0.40) for w in result.windows
        ), "Breath 4 (lowest valid-leak anchor) must appear in windows"
        assert not any(
            w.worst_mid_insp_flattening == pytest.approx(0.20) for w in result.windows
        ), "Breath 6 (leak_valid=False) must not anchor any window"

    async def test_flattening_threshold_excludes_breaths_above_it(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """flattening_threshold is a ceiling: only breaths at or below it anchor."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 1, 17)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)

        flattening = {2: 0.45, 12: 0.30, 22: 0.60}
        for i in range(25):
            await _make_breath(
                async_db_session,
                ar,
                sess,
                breath_number=i,
                mid_insp_flattening=flattening.get(i, 1.00),
            )

        await async_db_session.flush()
        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="worst_flattening_leak_valid",
            n=5,
            flattening_threshold=0.5,
        )

        assert [w.worst_mid_insp_flattening for w in result.windows] == [
            pytest.approx(0.30),
            pytest.approx(0.45),
        ]


# ---------------------------------------------------------------------------
# TestMixedAlgorithmIdentities
# ---------------------------------------------------------------------------


class TestMixedAlgorithmIdentities:
    async def test_fl_criterion_refuses_mixed_version_day(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two sessions classified as OK but with different algorithm identities
        → worst_flattening_leak_valid returns empty windows with
        null_reason='algo_version_mismatch' and day_status='mixed_version'.

        Uses patch.object on _latest_analysis_for_session so both sessions appear
        OK with different identities — the same technique as test_breath_service_seams
        and test_mcp_compare_epochs (both the unit and integration suites do this to
        exercise a scenario that requires code-level version bumps in production).
        """
        import copy  # noqa: PLC0415

        from unittest.mock import patch  # noqa: PLC0415

        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AlgoVersions,
            AnalysisRunMetadata,
        )
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415
        from snore.services.breath_service import (  # noqa: PLC0415
            AnalysisStatus,
            BreathService,
        )

        target_date = date(2024, 2, 1)
        device = await _make_device(async_db_session, async_test_profile.id)

        # Two sessions on the SAME Day (avoids UNIQUE constraint on device+date)
        day, sess1 = await _make_day_session(async_db_session, device, target_date)
        ar1 = await _make_analysis_result(async_db_session, sess1)
        await _make_breath(
            async_db_session,
            ar1,
            sess1,
            breath_number=0,
            mid_insp_flattening=0.8,
            leak_valid=True,
        )
        sess2 = await _add_session_to_day(async_db_session, device, day)
        ar2 = await _make_analysis_result(async_db_session, sess2)
        await _make_breath(
            async_db_session,
            ar2,
            sess2,
            breath_number=0,
            mid_insp_flattening=0.9,
            leak_valid=True,
        )

        # Build two distinct identities differing on a CROSS_VERSION_REFUSAL_KEY
        identity_a = AlgorithmIdentity.current()
        alt_dict = copy.deepcopy(identity_a.model_dump())
        alt_dict["segmenter"] = "v_old_incompatible"
        identity_b = AlgorithmIdentity.model_validate(alt_dict)

        run_meta = AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"])
        algo_a = AlgoVersions(identity=identity_a, run=run_meta)
        algo_b = AlgoVersions(identity=identity_b, run=run_meta)

        async def _mock_latest(self_svc: Any, session_id: int) -> Any:
            if session_id == sess1.id:
                return (AnalysisStatus.OK, algo_a, ar1.id)
            if session_id == sess2.id:
                return (AnalysisStatus.OK, algo_b, ar2.id)
            return (AnalysisStatus.NOT_RUN, None, None)

        await async_db_session.flush()
        with patch.object(BreathService, "_latest_analysis_for_session", _mock_latest):
            result = await find_windows(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                criterion="worst_flattening_leak_valid",
                n=5,
                device_id=device.id,
            )

        assert result.windows == []
        assert result.null_reason == "algo_version_mismatch"
        assert result.day_status == "mixed_version"

    async def test_ca_centered_works_on_same_mixed_version_day(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """ca_centered ignores algorithm version differences; it still returns
        windows anchored at CA events on the same mixed-identity day."""
        import copy  # noqa: PLC0415

        from unittest.mock import patch  # noqa: PLC0415

        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AlgoVersions,
            AnalysisRunMetadata,
        )
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415
        from snore.services.breath_service import (  # noqa: PLC0415
            AnalysisStatus,
            BreathService,
        )

        target_date = date(2024, 2, 2)
        device = await _make_device(async_db_session, async_test_profile.id)

        day, sess1 = await _make_day_session(async_db_session, device, target_date)
        ar1 = await _make_analysis_result(async_db_session, sess1)

        # CA event anchored 5 min into sess1
        ca_offset_s = 300.0
        async_db_session.add(
            Event(
                session_id=sess1.id,
                event_type="CA",
                start_time=sess1.start_time + timedelta(seconds=ca_offset_s),
                duration_seconds=15.0,
            )
        )
        await async_db_session.flush()

        # Second session on same Day with a different algorithm identity
        sess2 = await _add_session_to_day(async_db_session, device, day)
        ar2 = await _make_analysis_result(async_db_session, sess2)

        identity_a = AlgorithmIdentity.current()
        alt_dict = copy.deepcopy(identity_a.model_dump())
        alt_dict["segmenter"] = "v_old_incompatible"
        identity_b = AlgorithmIdentity.model_validate(alt_dict)

        run_meta = AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"])
        algo_a = AlgoVersions(identity=identity_a, run=run_meta)
        algo_b = AlgoVersions(identity=identity_b, run=run_meta)

        async def _mock_latest(self_svc: Any, session_id: int) -> Any:
            if session_id == sess1.id:
                return (AnalysisStatus.OK, algo_a, ar1.id)
            if session_id == sess2.id:
                return (AnalysisStatus.OK, algo_b, ar2.id)
            return (AnalysisStatus.NOT_RUN, None, None)

        with patch.object(BreathService, "_latest_analysis_for_session", _mock_latest):
            result = await find_windows(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                criterion="ca_centered",
                n=5,
                device_id=device.id,
            )

        # CA criterion works despite mixed identities — anchored at the CA event offset
        assert len(result.windows) == 1
        assert result.windows[0].anchor_event_offset == pytest.approx(ca_offset_s)
        assert result.windows[0].criterion == "ca_centered"


# ---------------------------------------------------------------------------
# TestCaCenteredNoAnalysis
# ---------------------------------------------------------------------------


class TestCaCenteredNoAnalysis:
    async def test_ca_centered_no_analysis_anchors_at_event_offsets(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """ca_centered with no analysis run at all: windows anchored at CA event
        offsets, per-window analysis_status='not_run',
        analysis_reason='analysis_not_run'."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 3, 1)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)

        # Seed two CA events — no AnalysisResult created for this session
        offset1_s = 300.0  # 5 min into session
        offset2_s = 600.0  # 10 min into session
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="CA",
                start_time=sess.start_time + timedelta(seconds=offset1_s),
                duration_seconds=18.0,
            )
        )
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="CA",
                start_time=sess.start_time + timedelta(seconds=offset2_s),
                duration_seconds=22.0,
            )
        )
        await async_db_session.flush()

        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="ca_centered",
            n=5,
            device_id=device.id,
        )

        assert result.null_reason is None
        assert len(result.windows) == 2

        anchor_offsets = {w.anchor_event_offset for w in result.windows}
        assert anchor_offsets == {offset1_s, offset2_s}

        for w in result.windows:
            assert w.analysis_status == "not_run"
            assert w.analysis_reason == "analysis_not_run"
            assert w.analysis_result_id is None
            assert w.criterion == "ca_centered"

    async def test_ca_window_offsets_match_context_seconds(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Window start/end offsets are event_offset ± context_seconds (default 120s)."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 3, 2)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)

        ev_offset_s = 300.0
        async_db_session.add(
            Event(
                session_id=sess.id,
                event_type="CA",
                start_time=sess.start_time + timedelta(seconds=ev_offset_s),
                duration_seconds=15.0,
            )
        )
        await async_db_session.flush()

        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="ca_centered",
            n=5,
            device_id=device.id,
        )

        assert len(result.windows) == 1
        w = result.windows[0]
        assert w.window_start_offset == pytest.approx(ev_offset_s - 120.0)
        assert w.window_end_offset == pytest.approx(ev_offset_s + 120.0)

    async def test_ca_centered_skips_disabled_sessions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """CA events of a disabled session on the same night yield no windows."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 3, 3)
        device = await _make_device(async_db_session, async_test_profile.id)
        day, enabled = await _make_day_session(async_db_session, device, target_date)
        disabled = await _add_session_to_day(async_db_session, device, day)
        disabled.enabled = False
        for sess in (enabled, disabled):
            async_db_session.add(
                Event(
                    session_id=sess.id,
                    event_type="CA",
                    start_time=sess.start_time + timedelta(seconds=300.0),
                    duration_seconds=15.0,
                )
            )
        await async_db_session.flush()

        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="ca_centered",
            n=5,
            device_id=device.id,
        )

        assert [w.session_id for w in result.windows] == [enabled.id]


# ---------------------------------------------------------------------------
# TestFlRunEndingInRecovery
# ---------------------------------------------------------------------------


class TestFlRunEndingInRecovery:
    async def test_fl_run_followed_by_recovery_breath_yields_window(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """FL run of 2+ breaths (flow_class >= 4) immediately followed by
        is_recovery_breath=True → one window returned with fl_run_length=2."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 4, 1)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)

        # Breaths: normal, FL, FL, recovery, normal
        await _make_breath(async_db_session, ar, sess, breath_number=0, flow_class=1)
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=1,
            flow_class=5,
            mid_insp_flattening=0.40,
        )
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=2,
            flow_class=6,
            mid_insp_flattening=0.55,
        )
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=3,
            flow_class=1,
            is_recovery_breath=True,
        )
        await _make_breath(async_db_session, ar, sess, breath_number=4, flow_class=1)

        await async_db_session.flush()
        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="fl_run_ending_in_recovery",
            n=5,
        )

        assert result.null_reason is None
        assert len(result.windows) == 1
        w = result.windows[0]
        assert w.fl_run_length == 2
        assert w.criterion == "fl_run_ending_in_recovery"
        assert w.worst_mid_insp_flattening == pytest.approx(0.40)
        assert w.analysis_status == "ok"

    async def test_mixed_primary_mode_refuses_fl_run_criterion(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two OK sessions with the same identity but different primary_mode
        → fl_run_ending_in_recovery returns null_reason='primary_mode_mismatch'.

        Uses patch.object on _latest_analysis_for_session so both sessions appear OK
        but with different primary_mode metadata (aasm vs aasm_relaxed).
        """
        from unittest.mock import patch  # noqa: PLC0415

        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AlgoVersions,
            AnalysisRunMetadata,
        )
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415
        from snore.services.breath_service import (  # noqa: PLC0415
            AnalysisStatus,
            BreathService,
        )

        target_date = date(2024, 4, 2)
        device = await _make_device(async_db_session, async_test_profile.id)

        # Two sessions on the same Day with FL breaths + recovery
        day, sess1 = await _make_day_session(async_db_session, device, target_date)
        ar1 = await _make_analysis_result(async_db_session, sess1)
        await _make_breath(async_db_session, ar1, sess1, breath_number=0, flow_class=5)
        await _make_breath(
            async_db_session,
            ar1,
            sess1,
            breath_number=1,
            flow_class=1,
            is_recovery_breath=True,
        )

        sess2 = await _add_session_to_day(async_db_session, device, day)
        ar2 = await _make_analysis_result(async_db_session, sess2)
        await _make_breath(async_db_session, ar2, sess2, breath_number=0, flow_class=5)
        await _make_breath(
            async_db_session,
            ar2,
            sess2,
            breath_number=1,
            flow_class=1,
            is_recovery_breath=True,
        )

        identity = AlgorithmIdentity.current()
        algo_a = AlgoVersions(
            identity=identity,
            run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
        )
        algo_b = AlgoVersions(
            identity=identity,
            run=AnalysisRunMetadata(
                primary_mode="aasm_relaxed", modes=["aasm_relaxed"]
            ),
        )

        async def _mock_latest(self_svc: Any, session_id: int) -> Any:
            if session_id == sess1.id:
                return (AnalysisStatus.OK, algo_a, ar1.id)
            if session_id == sess2.id:
                return (AnalysisStatus.OK, algo_b, ar2.id)
            return (AnalysisStatus.NOT_RUN, None, None)

        await async_db_session.flush()
        with patch.object(BreathService, "_latest_analysis_for_session", _mock_latest):
            result = await find_windows(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                criterion="fl_run_ending_in_recovery",
                n=5,
                device_id=device.id,
            )

        assert result.windows == []
        assert result.null_reason == "primary_mode_mismatch"


# ---------------------------------------------------------------------------
# TestCriterionIrrelevantOption
# ---------------------------------------------------------------------------


class TestCriterionIrrelevantOption:
    async def test_context_seconds_non_default_for_fl_criterion_raises_value_error(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Passing context_seconds (relevant only to ca_centered) with
        worst_flattening_leak_valid raises ValueError naming the field."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 5, 1)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=0,
            mid_insp_flattening=0.8,
            leak_valid=True,
        )

        await async_db_session.flush()
        with pytest.raises(ValueError) as exc_info:
            await find_windows(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                criterion="worst_flattening_leak_valid",
                n=5,
                context_seconds=60.0,  # non-default; irrelevant to this criterion
            )

        assert "context_seconds" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TestIsolation
# ---------------------------------------------------------------------------


class TestIsolation:
    async def test_profile_a_sees_empty_result_for_profile_b_date(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Profile A querying a date where only Profile B has data → empty NOT_RUN
        result, zero windows from B's sessions."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2024, 6, 1)
        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        ar_b = await _make_analysis_result(async_db_session, sess_b)

        # Seed breaths for B — profile A must not see them
        for i in range(5):
            await _make_breath(
                async_db_session,
                ar_b,
                sess_b,
                breath_number=i,
                mid_insp_flattening=0.9,
                leak_valid=True,
            )

        # Profile A calls find_windows for the date that only B has data for
        await async_db_session.flush()
        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=profile_a.id,
            criterion="worst_flattening_leak_valid",
            n=5,
        )

        # Should be an empty NOT_RUN refusal, NOT B's windows
        assert result.windows == []
        assert result.device_id is None  # 0-sentinel mapped to None
        assert result.day_status == "not_run"

    async def test_profile_a_with_profile_b_device_id_raises_validation_error(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Profile A passing Profile B's device_id → DeviceNotOwnedError mapped to
        ValidationError with 'not available in this session' message."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2024, 6, 2)
        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        await _make_analysis_result(async_db_session, sess_b)

        with pytest.raises(ValidationError) as exc_info:
            await find_windows(
                async_db_session,
                target_date,
                profile_id=profile_a.id,
                criterion="worst_flattening_leak_valid",
                n=5,
                device_id=device_b.id,  # B's device, forbidden for A
            )

        err_msg = str(exc_info.value)
        assert "not available in this session" in err_msg
        # Error message must name the device_id but must not mention "profile"
        # (profile_id is server-internal; leaking it aids enumeration)
        assert f"device_id={device_b.id}" in err_msg
        assert "profile" not in err_msg.lower()


# ---------------------------------------------------------------------------
# TestReraProxyCentered
# ---------------------------------------------------------------------------


class TestReraProxyCentered:
    async def test_rera_proxy_centered_window_anchor_and_clamping(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """rera_proxy_centered returns a window centered on the recovery breath's
        start_offset_s (±context_seconds), clamped to [0, last_breath.end_offset_s].

        Fixture: breaths 0–4 with breath spacing of 5s (each breath spans 5s).
          breath 0: start=0.0,  end=4.0  (normal)
          breath 1: start=5.0,  end=9.0  (FL, flow_class=5)
          breath 2: start=10.0, end=14.0 (FL, flow_class=6)
          breath 3: start=15.0, end=19.0 (recovery, is_recovery_breath=True)
          breath 4: start=20.0, end=24.0 (normal)

        With default context_seconds=120.0:
          win_start = max(0.0, 15.0 - 120.0) = 0.0  [clamped]
          win_end   = min(24.0, 15.0 + 120.0) = 24.0 [clamped]
          anchor_event_offset = 15.0
          fl_run_length = 2
        """
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 7, 1)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)

        # Flattening outside the FL run (breaths 0, 3, 4) is lower than inside
        # it, so worst_mid_insp_flattening proves it reads the run's minimum.
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=0,
            flow_class=1,
            mid_insp_flattening=0.05,
        )
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=1,
            flow_class=5,
            mid_insp_flattening=0.4,
        )
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=2,
            flow_class=6,
            mid_insp_flattening=0.2,
        )
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=3,
            flow_class=1,
            is_recovery_breath=True,
            mid_insp_flattening=0.1,
        )
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=4,
            flow_class=1,
            mid_insp_flattening=0.05,
        )

        await async_db_session.flush()
        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="rera_proxy_centered",
            n=5,
        )

        assert result.null_reason is None
        assert len(result.windows) == 1
        w = result.windows[0]
        assert w.criterion == "rera_proxy_centered"
        assert w.fl_run_length == 2
        # anchor is the recovery breath's start_offset_s (breath_number=3 → 15.0)
        assert w.anchor_event_offset == pytest.approx(15.0)
        # clamped to session bounds
        assert w.window_start_offset == pytest.approx(0.0)
        assert w.window_end_offset == pytest.approx(24.0)
        assert w.analysis_status == "ok"
        assert "rera_proxy" in w.reason_summary
        assert w.worst_mid_insp_flattening == pytest.approx(0.2)

    async def test_rera_proxy_centered_no_clamping_mid_session(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Window is not clamped when the recovery breath is far from both ends.

        Fixture: 60 breaths (spacing 5s each), FL run at breaths 30-31,
        recovery at breath 32. Last breath ends at 299.0.
          rec_offset = 32 * 5 = 160.0
          context_seconds = 120.0 (default)
          win_start = max(0.0, 160.0 - 120.0) = 40.0  [not clamped]
          win_end   = min(299.0, 160.0 + 120.0) = 280.0 [not clamped]
        """
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 7, 2)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)

        for i in range(60):
            if i in (30, 31):
                fc = 5
                is_rec = None
            elif i == 32:
                fc = 1
                is_rec = True
            else:
                fc = 1
                is_rec = None
            await _make_breath(
                async_db_session,
                ar,
                sess,
                breath_number=i,
                flow_class=fc,
                is_recovery_breath=is_rec,
            )

        await async_db_session.flush()
        result = await find_windows(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            criterion="rera_proxy_centered",
            n=5,
        )

        assert result.null_reason is None
        assert len(result.windows) == 1
        w = result.windows[0]
        rec_offset = 32 * 5.0  # breath_number=32, start_offset_s = 160.0
        assert w.anchor_event_offset == pytest.approx(rec_offset)
        assert w.window_start_offset == pytest.approx(rec_offset - 120.0)
        assert w.window_end_offset == pytest.approx(rec_offset + 120.0)

    async def test_rera_proxy_centered_irrelevant_option_rejected(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Passing an option irrelevant to rera_proxy_centered raises ValueError
        naming the offending field."""
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 7, 3)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        ar = await _make_analysis_result(async_db_session, sess)
        await _make_breath(async_db_session, ar, sess, breath_number=0, flow_class=5)
        await _make_breath(
            async_db_session,
            ar,
            sess,
            breath_number=1,
            flow_class=1,
            is_recovery_breath=True,
        )

        await async_db_session.flush()
        with pytest.raises(ValueError) as exc_info:
            await find_windows(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                criterion="rera_proxy_centered",
                n=5,
                include_unknown_leak=True,  # irrelevant to rera_proxy_centered
            )

        assert "include_unknown_leak" in str(exc_info.value)

    async def test_unknown_criterion_error_includes_rera_proxy_centered(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Unknown criterion error message must list rera_proxy_centered."""
        from snore.mcp.errors import ValidationError  # noqa: PLC0415
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415

        target_date = date(2024, 7, 4)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)

        await async_db_session.flush()
        with pytest.raises(ValidationError) as exc_info:
            await find_windows(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                criterion="totally_bogus_criterion",
                n=5,
            )

        assert "rera_proxy_centered" in str(exc_info.value)

    async def test_mixed_primary_mode_refuses_rera_proxy_centered(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two OK sessions with the same identity but different primary_mode
        → rera_proxy_centered returns null_reason='primary_mode_mismatch'.

        Uses patch.object on _latest_analysis_for_session so both sessions appear OK
        but with different primary_mode metadata (aasm vs aasm_relaxed).
        """
        from unittest.mock import patch  # noqa: PLC0415

        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AlgoVersions,
            AnalysisRunMetadata,
        )
        from snore.mcp.tools.windows import find_windows  # noqa: PLC0415
        from snore.services.breath_service import (  # noqa: PLC0415
            AnalysisStatus,
            BreathService,
        )

        target_date = date(2024, 7, 5)
        device = await _make_device(async_db_session, async_test_profile.id)

        day, sess1 = await _make_day_session(async_db_session, device, target_date)
        ar1 = await _make_analysis_result(async_db_session, sess1)
        await _make_breath(async_db_session, ar1, sess1, breath_number=0, flow_class=5)
        await _make_breath(
            async_db_session,
            ar1,
            sess1,
            breath_number=1,
            flow_class=1,
            is_recovery_breath=True,
        )

        sess2 = await _add_session_to_day(async_db_session, device, day)
        ar2 = await _make_analysis_result(async_db_session, sess2)
        await _make_breath(async_db_session, ar2, sess2, breath_number=0, flow_class=5)
        await _make_breath(
            async_db_session,
            ar2,
            sess2,
            breath_number=1,
            flow_class=1,
            is_recovery_breath=True,
        )

        identity = AlgorithmIdentity.current()
        algo_a = AlgoVersions(
            identity=identity,
            run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
        )
        algo_b = AlgoVersions(
            identity=identity,
            run=AnalysisRunMetadata(
                primary_mode="aasm_relaxed", modes=["aasm_relaxed"]
            ),
        )

        async def _mock_latest(self_svc: Any, session_id: int) -> Any:
            if session_id == sess1.id:
                return (AnalysisStatus.OK, algo_a, ar1.id)
            if session_id == sess2.id:
                return (AnalysisStatus.OK, algo_b, ar2.id)
            return (AnalysisStatus.NOT_RUN, None, None)

        await async_db_session.flush()
        with patch.object(BreathService, "_latest_analysis_for_session", _mock_latest):
            result = await find_windows(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,
                criterion="rera_proxy_centered",
                n=5,
                device_id=device.id,
            )

        assert result.windows == []
        assert result.null_reason == "primary_mode_mismatch"
