"""Integration tests for the experimental_compare_epochs MCP tool.

Exercises the full stack: experimental_compare_epochs adapter → BreathService → SQLite.
The default compare_epochs tool is covered in test_mcp_device_waveform_aggregates.py.

Scenarios covered:
  1. Two homogeneous epochs → per-epoch distributions populated.
  2. RX change mid-epoch → all epochs nulled with rx_changed_within_epoch.
  3. Cross-epoch algorithm identity mismatch → distributions still populated;
     version_warnings non-empty (non-blocking warning, not a hard refusal).
  4. Mixed primary_mode within one epoch → rera_proxy_count null + rera_reason
     primary_mode_mismatch; FL distributions still populated.
  5. Epoch over a range with no data → no_data_in_range; night without OK
     analysis counts in nights_missing_analysis, not nights_with_data.
  6. Only leak-valid breaths contribute to distributions.
  7. Profile isolation (adversarial): B's data never leaks into A's distributions;
     explicit foreign device_id → not_available (no exception).
  8. Sessions with unknown settings are not an RX change.
"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import (
    AnalysisResult,
    Breath,
    Session,
    Setting,
)
from snore.mcp.schemas import EpochSpec
from tests.integration.conftest import (
    _make_analysis_result,
    _make_day_session,
    _make_device,
    _make_profile,
)


async def _make_breath(
    db: AsyncSession,
    ar: AnalysisResult,
    session: Session,
    breath_number: int,
    leak_valid: bool = True,
    i_e_ratio: float | None = 0.4,
    mid_insp_flattening: float | None = None,
    flatness_index: float | None = None,
    tidal_volume_ml: float | None = None,
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
        i_e_ratio=i_e_ratio,
        leak_valid=leak_valid,
        mid_insp_flattening=mid_insp_flattening,
        flatness_index=flatness_index,
        tidal_volume_ml=tidal_volume_ml,
        flow_class=flow_class,
        is_recovery_breath=is_recovery_breath,
    )
    db.add(breath)
    return breath


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCompareEpochsTwoHomogeneousEpochs:
    async def test_two_homogeneous_epochs_return_populated_distributions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Two epochs with consistent RX and identical algorithm identity:
        distributions are populated, nights_with_data correct, rx_settings echoed."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)

        # Epoch A: two nights in January
        _, sess_a1 = await _make_day_session(async_db_session, device, date(2025, 1, 5))
        ar_a1 = await _make_analysis_result(async_db_session, sess_a1)
        async_db_session.add(
            Setting(session_id=sess_a1.id, key="pressure_min", value="4.0")
        )
        await _make_breath(
            async_db_session,
            ar_a1,
            sess_a1,
            1,
            mid_insp_flattening=0.6,
            flatness_index=0.7,
            tidal_volume_ml=400.0,
        )
        await _make_breath(
            async_db_session,
            ar_a1,
            sess_a1,
            2,
            mid_insp_flattening=0.4,
            flatness_index=0.5,
            tidal_volume_ml=380.0,
        )

        _, sess_a2 = await _make_day_session(
            async_db_session, device, date(2025, 1, 20)
        )
        ar_a2 = await _make_analysis_result(async_db_session, sess_a2)
        async_db_session.add(
            Setting(session_id=sess_a2.id, key="pressure_min", value="4.0")
        )
        await _make_breath(
            async_db_session,
            ar_a2,
            sess_a2,
            1,
            mid_insp_flattening=0.5,
            flatness_index=0.6,
            tidal_volume_ml=390.0,
        )
        await async_db_session.flush()

        # Epoch B: two nights in February
        _, sess_b1 = await _make_day_session(async_db_session, device, date(2025, 2, 5))
        ar_b1 = await _make_analysis_result(async_db_session, sess_b1)
        async_db_session.add(
            Setting(session_id=sess_b1.id, key="pressure_min", value="6.0")
        )
        await _make_breath(
            async_db_session,
            ar_b1,
            sess_b1,
            1,
            mid_insp_flattening=0.3,
            flatness_index=0.4,
            tidal_volume_ml=360.0,
        )
        await _make_breath(
            async_db_session,
            ar_b1,
            sess_b1,
            2,
            mid_insp_flattening=0.2,
            flatness_index=0.3,
            tidal_volume_ml=350.0,
        )

        _, sess_b2 = await _make_day_session(
            async_db_session, device, date(2025, 2, 20)
        )
        ar_b2 = await _make_analysis_result(async_db_session, sess_b2)
        async_db_session.add(
            Setting(session_id=sess_b2.id, key="pressure_min", value="6.0")
        )
        await _make_breath(
            async_db_session,
            ar_b2,
            sess_b2,
            1,
            mid_insp_flattening=0.25,
            flatness_index=0.35,
            tidal_volume_ml=370.0,
        )
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(
                    label="EpochA", date_start="2025-01-01", date_end="2025-01-31"
                ),
                EpochSpec(
                    label="EpochB", date_start="2025-02-01", date_end="2025-02-28"
                ),
            ],
        )

        assert result.null_reason is None
        assert len(result.epochs) == 2

        epoch_a = result.epochs[0]
        assert epoch_a.label == "EpochA"
        assert epoch_a.null_reason is None
        assert epoch_a.nights_with_data == 2
        assert epoch_a.nights_missing_analysis == 0
        assert epoch_a.rera_proxy_version == "v2"
        # mid_insp_flattening median of [0.6, 0.4, 0.5] = 0.5
        assert epoch_a.mid_insp_flattening.median == pytest.approx(0.5, abs=0.01)
        assert epoch_a.mid_insp_flattening.n_breaths == 3
        # flow_class keys are strings
        assert all(isinstance(k, str) for k in epoch_a.flow_class_distribution)
        # rx_settings echoed from first session's snapshot
        assert epoch_a.rx_settings.get("pressure_min") == "4.0"

        epoch_b = result.epochs[1]
        assert epoch_b.label == "EpochB"
        assert epoch_b.null_reason is None
        assert epoch_b.nights_with_data == 2
        assert epoch_b.rx_settings.get("pressure_min") == "6.0"
        # mid_insp_flattening median of [0.3, 0.2, 0.25] = 0.25
        assert epoch_b.mid_insp_flattening.median == pytest.approx(0.25, abs=0.01)


class TestCompareEpochsRxViolation:
    async def test_rx_change_within_epoch_nulls_all_epochs(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """RX change between two sessions inside one epoch nulls ALL epoch distributions
        and returns rx_violations listing the epoch label, changed key, and change date."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)

        # Night 1: pressure_min = 4.0
        _, sess1 = await _make_day_session(async_db_session, device, date(2025, 1, 5))
        ar1 = await _make_analysis_result(async_db_session, sess1)
        async_db_session.add(
            Setting(session_id=sess1.id, key="pressure_min", value="4.0")
        )
        await _make_breath(async_db_session, ar1, sess1, 1, mid_insp_flattening=0.5)
        await async_db_session.flush()

        # Night 2: pressure_min = 6.0 (changed!)
        _, sess2 = await _make_day_session(async_db_session, device, date(2025, 1, 20))
        ar2 = await _make_analysis_result(async_db_session, sess2)
        async_db_session.add(
            Setting(session_id=sess2.id, key="pressure_min", value="6.0")
        )
        await _make_breath(async_db_session, ar2, sess2, 1, mid_insp_flattening=0.3)
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(
                    label="Epoch1", date_start="2025-01-01", date_end="2025-01-31"
                )
            ],
        )

        assert result.null_reason == "rx_changed_within_epoch"
        assert len(result.epochs) == 1
        assert result.epochs[0].null_reason == "rx_changed_within_epoch"

        assert len(result.rx_violations) == 1
        v = result.rx_violations[0]
        assert v.epoch_label == "Epoch1"
        assert "pressure_min" in v.changed_keys
        # change_dates contains the date when the change was detected (isoformat str)
        assert len(v.change_dates) == 1
        assert v.change_dates[0] == "2025-01-20"

    async def _seed_nights(
        self, db: AsyncSession, profile_id: int, rx_values: list[str | None]
    ) -> None:
        """One analyzed session per night from 2025-02-01; None seeds no settings."""
        device = await _make_device(db, profile_id)
        for offset, value in enumerate(rx_values, start=1):
            _, sess = await _make_day_session(db, device, date(2025, 2, offset))
            ar = await _make_analysis_result(db, sess)
            await _make_breath(db, ar, sess, 1, mid_insp_flattening=0.5)
            if value is not None:
                db.add(Setting(session_id=sess.id, key="pressure_min", value=value))
        await db.flush()

    @pytest.mark.parametrize(
        ("rx_values", "expected_rx"),
        [
            pytest.param([None, "8.0"], {"pressure_min": "8.0"}, id="unknown_first"),
            pytest.param(["8.0", None], {"pressure_min": "8.0"}, id="unknown_last"),
            pytest.param([None, None], {}, id="all_unknown"),
        ],
    )
    async def test_unknown_settings_are_not_an_rx_change(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        rx_values: list[str | None],
        expected_rx: dict[str, str],
    ) -> None:
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        await self._seed_nights(async_db_session, async_test_profile.id, rx_values)

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(label="A", date_start="2025-02-01", date_end="2025-02-28")
            ],
        )

        epoch = result.epochs[0]
        assert result.null_reason is None
        assert result.rx_violations == []
        assert epoch.null_reason is None
        assert epoch.rx_settings == expected_rx
        assert epoch.mid_insp_flattening.median == pytest.approx(0.5)

    async def test_change_across_unknown_session_reports_later_known_date(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        await self._seed_nights(
            async_db_session, async_test_profile.id, ["8.0", None, "12.0"]
        )

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(label="A", date_start="2025-02-01", date_end="2025-02-28")
            ],
        )

        assert result.epochs[0].null_reason == "rx_changed_within_epoch"
        assert result.epochs[0].rx_settings == {"pressure_min": "8.0"}
        violation = result.rx_violations[0]
        assert violation.changed_keys == ["pressure_min"]
        assert violation.change_dates == ["2025-02-03"]


class TestCompareEpochsAlgoMismatch:
    async def test_cross_epoch_algo_mismatch_warns_but_populates_distributions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Sessions in different epochs that classify OK but with different algorithm
        identities on a CROSS_VERSION_REFUSAL_KEY produce non-null distributions AND
        non-empty version_warnings — algo mismatch is no longer a hard refusal."""
        from snore.analysis.shared.versioning import (  # noqa: PLC0415
            AlgorithmIdentity,
            AlgoVersions,
            AnalysisRunMetadata,
        )
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415
        from snore.services.breath_service import (  # noqa: PLC0415
            AnalysisStatus,
            BreathService,
        )

        device = await _make_device(async_db_session, async_test_profile.id)

        _, sess1 = await _make_day_session(async_db_session, device, date(2025, 1, 5))
        ar1 = await _make_analysis_result(async_db_session, sess1)
        await _make_breath(async_db_session, ar1, sess1, 1, mid_insp_flattening=0.5)
        await async_db_session.flush()

        _, sess2 = await _make_day_session(async_db_session, device, date(2025, 2, 5))
        ar2 = await _make_analysis_result(async_db_session, sess2)
        await _make_breath(async_db_session, ar2, sess2, 1, mid_insp_flattening=0.4)
        await async_db_session.flush()

        # Build two distinct identities differing on a CROSS_VERSION_REFUSAL_KEY
        identity_current = AlgorithmIdentity.current()
        identity_stale_dict = identity_current.model_dump()
        identity_stale_dict["segmenter"] = "old_segmenter_v0"
        identity_stale = AlgorithmIdentity.model_validate(identity_stale_dict)

        run_meta = AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"])
        results_iter = iter(
            [
                (
                    AnalysisStatus.OK,
                    AlgoVersions(identity=identity_current, run=run_meta),
                    ar1.id,
                ),
                (
                    AnalysisStatus.OK,
                    AlgoVersions(identity=identity_stale, run=run_meta),
                    ar2.id,
                ),
            ]
        )

        async def _mock_latest(self_svc: Any, session_id: int) -> Any:
            return next(results_iter)

        with patch.object(BreathService, "_latest_analysis_for_session", _mock_latest):
            result = await experimental_compare_epochs(
                async_db_session,
                profile_id=async_test_profile.id,
                epochs=[
                    EpochSpec(
                        label="EpochA", date_start="2025-01-01", date_end="2025-01-31"
                    ),
                    EpochSpec(
                        label="EpochB", date_start="2025-02-01", date_end="2025-02-28"
                    ),
                ],
            )

        # No longer a hard refusal: distributions are populated
        assert result.null_reason is None
        assert len(result.epochs) == 2
        assert all(s.null_reason is None for s in result.epochs)
        # Distributions contain the breath data
        assert result.epochs[0].mid_insp_flattening.n_breaths == 1
        assert result.epochs[1].mid_insp_flattening.n_breaths == 1
        # version_warnings is non-empty with a message about segmenter
        assert len(result.version_warnings) > 0
        assert any("segmenter" in w for w in result.version_warnings)
        # The warning mentions both versions
        segmenter_warning = next(w for w in result.version_warnings if "segmenter" in w)
        assert "old_segmenter_v0" in segmenter_warning


class TestCompareEpochsMixedPrimaryMode:
    async def test_mixed_primary_mode_nulls_rera_but_fl_distributions_populated(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Sessions within one epoch with different primary_mode values degrade only
        RERA fields (rera_proxy_count=None, rera_reason=primary_mode_mismatch) while
        mid_insp_flattening distribution is still populated."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)

        # Session 1: primary_mode = "aasm"
        _, sess1 = await _make_day_session(async_db_session, device, date(2025, 1, 5))
        ar1 = await _make_analysis_result(async_db_session, sess1, primary_mode="aasm")
        await _make_breath(
            async_db_session,
            ar1,
            sess1,
            1,
            mid_insp_flattening=0.6,
            flatness_index=0.7,
        )
        await _make_breath(
            async_db_session,
            ar1,
            sess1,
            2,
            mid_insp_flattening=0.4,
            flatness_index=0.5,
        )
        await async_db_session.flush()

        # Session 2: primary_mode = "resmed" (different from "aasm")
        _, sess2 = await _make_day_session(async_db_session, device, date(2025, 1, 20))
        ar2 = await _make_analysis_result(
            async_db_session, sess2, primary_mode="resmed"
        )
        await _make_breath(
            async_db_session,
            ar2,
            sess2,
            1,
            mid_insp_flattening=0.5,
            flatness_index=0.6,
        )
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(label="Mixed", date_start="2025-01-01", date_end="2025-01-31")
            ],
        )

        assert result.null_reason is None
        assert len(result.epochs) == 1
        ep = result.epochs[0]
        assert ep.null_reason is None
        # RERA fields degraded; version is null because the scan never ran
        assert ep.rera_proxy_count is None
        assert ep.rera_reason == "primary_mode_mismatch"
        assert ep.rera_proxy_version is None
        # FL distributions still populated
        assert ep.mid_insp_flattening.median is not None
        assert ep.mid_insp_flattening.n_breaths == 3
        assert ep.mid_insp_flattening.median == pytest.approx(0.5, abs=0.01)


class TestCompareEpochsNoData:
    async def test_epoch_with_no_sessions_returns_no_data_in_range(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Epoch over a date range with no sessions returns no_data_in_range.
        A night that has a session but no OK analysis counts in nights_missing_analysis,
        not nights_with_data."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)

        # Seed one session with NO analysis result (nights_missing_analysis += 1)
        _, sess_no_ar = await _make_day_session(
            async_db_session, device, date(2025, 1, 10)
        )
        # No AnalysisResult created for this session
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(label="Empty", date_start="2025-01-01", date_end="2025-01-31")
            ],
        )

        # The top-level null_reason is None when only individual epochs fail (no
        # cross-epoch refusal); the per-epoch null_reason carries the actual reason.
        assert result.null_reason is None
        assert len(result.epochs) == 1
        ep = result.epochs[0]
        assert ep.null_reason == "no_data_in_range"
        assert ep.nights_with_data == 0
        # Session exists but has no OK analysis
        assert ep.nights_missing_analysis == 1
        # No RERA scan ran → no criterion version to stamp
        assert ep.rera_proxy_version is None

    async def test_epoch_with_no_sessions_at_all_returns_no_data_in_range(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Epoch date range that has no sessions at all (not even missing-analysis ones):
        no_data_in_range, nights_with_data=0, nights_missing_analysis=0."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        # Seed data in January, query February
        _, sess = await _make_day_session(async_db_session, device, date(2025, 1, 5))
        await _make_analysis_result(async_db_session, sess)
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(label="Feb", date_start="2025-02-01", date_end="2025-02-28")
            ],
        )

        assert result.null_reason == "no_data_in_range"
        ep = result.epochs[0]
        assert ep.nights_with_data == 0
        assert ep.nights_missing_analysis == 0

    async def test_explicit_owned_device_without_sessions_is_top_level_no_data(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """An owned device with no sessions in range gets the same top-level answer
        as an auto-selected device: the shared union check, not a per-epoch fallback."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(
                    label="Feb",
                    date_start="2025-02-01",
                    date_end="2025-02-28",
                    device_id=device.id,
                )
            ],
        )

        assert result.null_reason == "no_data_in_range"
        assert result.epochs[0].null_reason == "no_data_in_range"
        assert result.epochs[0].rera_reason == "no_data_in_range"


class TestCompareEpochsLeakValid:
    async def test_only_leak_valid_breaths_contribute_to_distributions(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Breaths with leak_valid=False or None are excluded from distributions;
        seeding extreme values for invalid breaths must not affect the median."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, date(2025, 1, 5))
        ar = await _make_analysis_result(async_db_session, sess)

        # Two valid breaths with mid_insp_flattening = 0.5 and 0.5
        await _make_breath(
            async_db_session, ar, sess, 1, leak_valid=True, mid_insp_flattening=0.5
        )
        await _make_breath(
            async_db_session, ar, sess, 2, leak_valid=True, mid_insp_flattening=0.5
        )
        # Two invalid breaths with extreme values
        await _make_breath(
            async_db_session, ar, sess, 3, leak_valid=False, mid_insp_flattening=100.0
        )
        await _make_breath(
            async_db_session, ar, sess, 4, leak_valid=False, mid_insp_flattening=200.0
        )
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=async_test_profile.id,
            epochs=[
                EpochSpec(label="Ep", date_start="2025-01-01", date_end="2025-01-31")
            ],
        )

        assert result.null_reason is None
        ep = result.epochs[0]
        assert ep.null_reason is None
        # Only the 2 leak-valid breaths contribute
        assert ep.mid_insp_flattening.n_breaths == 2
        assert ep.mid_insp_flattening.median == pytest.approx(0.5, abs=0.01)


class TestCompareEpochsProfileIsolation:
    async def test_profile_a_over_dates_with_only_profile_b_data_returns_no_data(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Profile A comparing epochs over dates where only profile B has sessions
        → no_data_in_range (B's breaths never leak into A's distributions)."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2025, 3, 10)

        # Only profile B has data on target_date
        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        ar_b = await _make_analysis_result(async_db_session, sess_b)
        await _make_breath(async_db_session, ar_b, sess_b, 1, mid_insp_flattening=0.9)
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=profile_a.id,
            epochs=[
                EpochSpec(
                    label="A_range",
                    date_start="2025-03-01",
                    date_end="2025-03-31",
                )
            ],
        )

        # Profile A sees no data — B's sessions are not accessible
        assert result.null_reason == "no_data_in_range"
        assert result.epochs[0].nights_with_data == 0

    async def test_profile_a_passing_profile_b_device_id_returns_not_available(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Profile A explicitly passing profile B's device_id → all epochs
        not_available; no exception is raised (handled inside the service)."""
        from snore.mcp.tools.epochs import experimental_compare_epochs  # noqa: PLC0415

        profile_a = async_test_profile
        profile_b = await _make_profile(async_db_session)

        target_date = date(2025, 3, 10)

        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        await _make_analysis_result(async_db_session, sess_b)
        await async_db_session.flush()

        result = await experimental_compare_epochs(
            async_db_session,
            profile_id=profile_a.id,
            epochs=[
                EpochSpec(
                    label="Attempt",
                    date_start="2025-03-01",
                    date_end="2025-03-31",
                    device_id=device_b.id,  # Profile A does not own this device
                )
            ],
        )

        # Service returns not_available (no exception) for foreign device_id
        assert result.null_reason == "not_available"
        assert all(ep.null_reason == "not_available" for ep in result.epochs)
