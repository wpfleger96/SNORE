"""
Unit tests for the FL validation module.

Tests alignment helper, metric computation, and validator skip paths.
All DB interactions use the mock_db_session fixture from conftest.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from snore.constants import FLOW_LIMITATION_CLASSES
from snore.constants import FlowLimitationConstants as FLC
from snore.database.waveform_blobs import deserialize_waveform_blob
from snore.validation.alignment import average_waveform_over_breaths
from snore.validation.fl_validator import (
    FlowLimitationValidator,
    _auc_mwu,
    score_fl_arrays,
)
from snore.validation.stats import spearman_or_none

# ---------------------------------------------------------------------------
# alignment helper
# ---------------------------------------------------------------------------


class TestAverageWaveformOverBreaths:
    def test_normal_averaging(self):
        """Each breath window picks up its samples and averages them."""
        timestamps = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
        values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        starts = np.array([0.0, 4.0])
        ends = np.array([3.0, 9.0])

        result = average_waveform_over_breaths(starts, ends, timestamps, values)

        assert result.shape == (2,)
        # Breath 0: samples at t=0 (0.1) and t=2 (0.3) → mean = 0.2
        np.testing.assert_allclose(result[0], 0.2)
        # Breath 1: samples at t=4 (0.5), t=6 (0.7), t=8 (0.9) → mean = 0.7
        np.testing.assert_allclose(result[1], 0.7)

    def test_window_with_no_samples_is_nan(self):
        """A breath window containing zero waveform samples yields NaN."""
        timestamps = np.array([10.0, 12.0])
        values = np.array([0.5, 0.6])
        # Breath window is entirely before any waveform samples
        starts = np.array([0.0])
        ends = np.array([5.0])

        result = average_waveform_over_breaths(starts, ends, timestamps, values)
        assert np.isnan(result[0])

    def test_gap_mid_signal(self):
        """Breaths spanning a gap in the waveform get NaN for the gapped breath."""
        timestamps = np.array([0.0, 1.0, 100.0, 101.0])
        values = np.array([0.2, 0.4, 0.6, 0.8])
        starts = np.array([0.0, 50.0, 100.0])
        ends = np.array([2.0, 90.0, 102.0])

        result = average_waveform_over_breaths(starts, ends, timestamps, values)

        np.testing.assert_allclose(result[0], 0.3)  # (0.2+0.4)/2
        assert np.isnan(result[1])  # no samples in [50, 90)
        np.testing.assert_allclose(result[2], 0.7)  # (0.6+0.8)/2

    def test_empty_inputs(self):
        """Empty arrays return empty result without error."""
        result = average_waveform_over_breaths(
            np.array([]), np.array([]), np.array([]), np.array([])
        )
        assert result.shape == (0,)

    def test_empty_waveform(self):
        """Non-empty breaths but empty waveform → all NaN."""
        result = average_waveform_over_breaths(
            np.array([0.0, 5.0]),
            np.array([4.0, 9.0]),
            np.array([]),
            np.array([]),
        )
        assert result.shape == (2,)
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# Spearman helper (now lives in stats.py)
# ---------------------------------------------------------------------------


class TestSpearmanOrNone:
    def test_monotone_increasing_gives_r_approx_1(self):
        x = np.arange(10, dtype=float)
        y = x * 2 + 1
        r, p = spearman_or_none(x, y)
        assert r is not None
        assert abs(r - 1.0) < 1e-6

    def test_too_few_samples_returns_none(self):
        r, p = spearman_or_none(np.array([0.1, 0.9]), np.array([0.2, 0.8]))
        assert r is None
        assert p is None

    def test_constant_input_returns_none(self):
        """scipy returns NaN for constant input; wrapper must return (None, None)."""
        x = np.ones(10)
        y = np.arange(10, dtype=float)
        r, p = spearman_or_none(x, y)
        assert r is None
        assert p is None

    def test_n_equals_3_is_allowed(self):
        r, p = spearman_or_none(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]))
        assert r is not None
        assert abs(r - (-1.0)) < 1e-6


# ---------------------------------------------------------------------------
# AUC helper
# ---------------------------------------------------------------------------


class TestAucMwu:
    def test_perfect_separator_auc_is_1(self):
        """When positives all score higher than negatives, AUC = 1.0."""
        scores = np.array([0.9, 0.8, 0.7, 0.1, 0.2, 0.3])
        labels = np.array([True, True, True, False, False, False])
        auc = _auc_mwu(scores, labels)
        assert auc is not None
        assert abs(auc - 1.0) < 1e-6

    def test_anti_separator_auc_is_0(self):
        """When positives all score lower than negatives, AUC = 0.0."""
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        labels = np.array([True, True, True, False, False, False])
        auc = _auc_mwu(scores, labels)
        assert auc is not None
        assert abs(auc - 0.0) < 1e-6

    def test_no_positives_returns_none(self):
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([False, False, False])
        assert _auc_mwu(scores, labels) is None

    def test_no_negatives_returns_none(self):
        scores = np.array([0.8, 0.9])
        labels = np.array([True, True])
        assert _auc_mwu(scores, labels) is None


# ---------------------------------------------------------------------------
# Validator skip paths (using mock DB)
# ---------------------------------------------------------------------------


def _make_mock_session_row(
    session_id: int = 1, parser_version: str | None = "1.1.0"
) -> MagicMock:
    """Minimal Session ORM mock."""
    row = MagicMock()
    row.id = session_id
    row.start_time.strftime = lambda fmt: "2025-01-01"
    row.duration_seconds = 28800.0
    row.parser_version = parser_version
    return row


def _make_analysis_result_mock(ar_id: int = 99) -> MagicMock:
    """Analysis row mock with current AlgoVersions so BreathService classifies it OK."""
    from snore.analysis.shared.versioning import (  # noqa: PLC0415
        AlgorithmIdentity,
        AlgoVersions,
        AnalysisRunMetadata,
    )

    algo_versions = AlgoVersions(
        identity=AlgorithmIdentity.current(),
        run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
    )
    row = MagicMock()
    row.engine_versions_json = algo_versions.model_dump()
    row.id = ar_id
    return row


def _make_waveform_blob(timestamps: list[float], values: list[float]) -> bytes:
    """Build a float32 waveform blob matching deserialize_waveform_blob format."""
    ts = np.array(timestamps, dtype=np.float32)
    vs = np.array(values, dtype=np.float32)
    return np.column_stack([ts, vs]).astype(np.float32).tobytes()


def _make_breath_mock(
    start: float,
    end: float,
    mid_insp_flattening: float,
    flatness_index: float,
    breath_number: int = 0,
    flow_class: int | None = None,
    flow_confidence: float | None = None,
) -> MagicMock:
    b = MagicMock()
    b.start_offset_s = start
    b.end_offset_s = end
    b.mid_insp_flattening = mid_insp_flattening
    b.flatness_index = flatness_index
    b.breath_number = breath_number
    b.leak_valid = True
    b.flow_class = flow_class
    b.flow_confidence = flow_confidence
    return b


class TestFlowLimitationValidatorSkipPaths:
    """Tests for skip-reason logic — no real DB required."""

    @pytest.mark.asyncio
    async def test_skip_no_flg_waveform(self, mock_db_session):
        """Session without FLG waveform row → skipped_reason = 'no_flg_waveform'."""
        session_row = _make_mock_session_row(1)

        # First execute → sessions list
        # Second execute → waveform lookup → empty (no FLG waveform)
        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = None

        mock_db_session.execute = AsyncMock(
            side_effect=[sessions_result, waveform_result]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        assert len(report.sessions) == 1
        s = report.sessions[0]
        assert s.skipped_reason == "no_flg_waveform"
        assert s.has_flg_waveform is False
        assert s.parser_version == "1.1.0"

    @pytest.mark.asyncio
    async def test_parser_version_null_falls_back_to_unknown(self, mock_db_session):
        """Session with NULL parser_version → report shows 'unknown'."""
        session_row = _make_mock_session_row(1, parser_version=None)

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = None

        mock_db_session.execute = AsyncMock(
            side_effect=[sessions_result, waveform_result]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        assert report.sessions[0].parser_version == "unknown"

    @pytest.mark.asyncio
    async def test_skip_no_analysis(self, mock_db_session):
        """Session with FLG waveform but no analysis → skipped_reason = 'no_analysis'."""

        session_row = _make_mock_session_row(2)

        waveform_mock = MagicMock()
        waveform_mock.data_blob = b"\x00" * 16  # placeholder
        waveform_mock.sample_count = 2

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        # AnalysisResult query for _latest_analysis_for_session → None (NOT_RUN)
        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = None

        mock_db_session.execute = AsyncMock(
            side_effect=[sessions_result, waveform_result, analysis_result]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        assert len(report.sessions) == 1
        s = report.sessions[0]
        assert s.skipped_reason == "no_analysis"
        assert s.has_flg_waveform is True

    @pytest.mark.asyncio
    async def test_skip_no_valid_breaths(self, mock_db_session):
        """Session with FLG waveform and analysis but no leak-valid breaths → 'no_valid_breaths'."""
        session_row = _make_mock_session_row(3)

        timestamps = np.array([0.0, 2.0], dtype=np.float32)
        values = np.array([0.3, 0.5], dtype=np.float32)
        data = np.column_stack([timestamps, values]).astype(np.float32)

        waveform_mock = MagicMock()
        waveform_mock.data_blob = data.tobytes()
        waveform_mock.sample_count = 2

        analysis_row = _make_analysis_result_mock(99)

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        # Breath query → empty list
        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = []

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        assert len(report.sessions) == 1
        s = report.sessions[0]
        assert s.skipped_reason == "no_valid_breaths"
        assert s.has_flg_waveform is True

    @pytest.mark.asyncio
    async def test_skip_no_flg_samples_null_blob(self, mock_db_session):
        """Waveform row with data_blob=None → skipped_reason = 'no_flg_samples'."""
        session_row = _make_mock_session_row(4)

        waveform_mock = MagicMock()
        waveform_mock.data_blob = None
        waveform_mock.sample_count = 10

        analysis_row = _make_analysis_result_mock(99)

        breath_mock = _make_breath_mock(0.0, 4.0, 0.8, 0.2)

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = [breath_mock]

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        s = report.sessions[0]
        assert s.skipped_reason == "no_flg_samples"
        assert s.has_flg_waveform is True

    @pytest.mark.asyncio
    async def test_skip_no_flg_samples_zero_count(self, mock_db_session):
        """Waveform row with sample_count=0 → skipped_reason = 'no_flg_samples'."""
        session_row = _make_mock_session_row(5)

        waveform_mock = MagicMock()
        waveform_mock.data_blob = b"\x00" * 16
        waveform_mock.sample_count = 0

        analysis_row = _make_analysis_result_mock(99)

        breath_mock = _make_breath_mock(0.0, 4.0, 0.8, 0.2)

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = [breath_mock]

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        s = report.sessions[0]
        assert s.skipped_reason == "no_flg_samples"
        assert s.has_flg_waveform is True

    @pytest.mark.asyncio
    async def test_skip_no_aligned_pairs(self, mock_db_session):
        """All breath windows have no FLG samples → skipped_reason = 'no_aligned_pairs'."""
        session_row = _make_mock_session_row(6)

        # FLG samples at t=100–102; breath windows at t=0–10 → no overlap
        blob = _make_waveform_blob([100.0, 101.0, 102.0], [0.5, 0.5, 0.5])
        waveform_mock = MagicMock()
        waveform_mock.data_blob = blob
        waveform_mock.sample_count = 3

        analysis_row = _make_analysis_result_mock(99)

        breaths = [_make_breath_mock(0.0, 4.0, 0.8, 0.2, i) for i in range(3)]

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = breaths

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        s = report.sessions[0]
        assert s.skipped_reason == "no_aligned_pairs"
        assert s.has_flg_waveform is True

    @pytest.mark.asyncio
    async def test_exception_creates_error_session_and_counts_in_total(
        self, mock_db_session
    ):
        """Unhandled exception during _validate_session → error record in results, total_sessions correct."""
        session_row = _make_mock_session_row(7)

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        mock_db_session.execute = AsyncMock(return_value=sessions_result)

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)

        with patch.object(
            validator,
            "_validate_session",
            side_effect=RuntimeError("boom"),
        ):
            report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        assert len(report.sessions) == 1
        s = report.sessions[0]
        assert s.skipped_reason == "error"
        assert s.spearman_flattening_r is None
        # total_sessions must equal the number the query returned
        assert report.aggregate.total_sessions == 1

    @pytest.mark.asyncio
    async def test_inf_in_flg_blob_is_filtered_out(self, mock_db_session):
        """Inf values in FLG blob are masked before metrics are computed; session still compared."""
        session_row = _make_mock_session_row(8)

        # 5 good breaths; FLG blob has 1 Inf sample among valid ones
        ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        vs = [0.1, float("inf"), 0.3, 0.5, 0.7, 0.9]
        blob = _make_waveform_blob(ts, vs)
        waveform_mock = MagicMock()
        waveform_mock.data_blob = blob
        waveform_mock.sample_count = 6

        analysis_row = _make_analysis_result_mock(99)

        # Breaths that align with non-Inf samples
        breaths = [
            _make_breath_mock(0.0, 1.5, 0.9, 0.1, 0),  # t=0 → 0.1
            _make_breath_mock(4.0, 5.5, 0.7, 0.3, 1),  # t=4 → 0.3
            _make_breath_mock(6.0, 7.5, 0.5, 0.5, 2),  # t=6 → 0.5
            _make_breath_mock(8.0, 9.5, 0.3, 0.7, 3),  # t=8 → 0.7
            _make_breath_mock(10.0, 11.5, 0.1, 0.9, 4),  # t=10 → 0.9
        ]

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = breaths

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        s = report.sessions[0]
        # Session must be compared (not skipped)
        assert s.skipped_reason is None
        # spearman_flattening_r must be finite (Inf did not leak into metrics)
        assert s.spearman_flattening_r is not None
        assert np.isfinite(s.spearman_flattening_r)
        # device_flg_95th must be finite (Inf was masked before computing percentile)
        assert s.device_flg_95th is not None
        assert np.isfinite(s.device_flg_95th)

    @pytest.mark.asyncio
    async def test_empty_date_range_returns_empty_report(self, mock_db_session):
        """No sessions in range → aggregate totals are all 0."""
        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = []
        mock_db_session.execute = AsyncMock(return_value=sessions_result)

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        assert report.aggregate.total_sessions == 0
        assert report.aggregate.sessions_compared == 0
        assert report.sessions == []


# ---------------------------------------------------------------------------
# End-to-end numeric pipeline test
# ---------------------------------------------------------------------------


class TestNumericPipeline:
    """Hand-computable end-to-end fixture that catches orientation inversion."""

    @pytest.mark.asyncio
    async def test_known_fixture_produces_exact_metrics(self, mock_db_session):
        """
        5 breaths with perfectly correlated flattening_severity and FLG averages.

        Breath layout (start, end, mid_insp_flat, flatness_index):
          0: [0, 4)   0.9  0.1  → flattening_severity=0.1
          1: [4, 8)   0.7  0.3  → flattening_severity=0.3
          2: [8, 12)  0.5  0.5  → flattening_severity=0.5
          3: [12, 16) 0.3  0.7  → flattening_severity=0.7
          4: [16, 20) 0.1  0.9  → flattening_severity=0.9

        FLG at 0.5 Hz (every 2s, values repeat in pairs):
          t=[ 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
          v=[0.1,0.1,0.3,0.3,0.5, 0.5, 0.7, 0.7, 0.9, 0.9]

        Per-window averages (half-open [start, end)):
          breath 0 → mean([0.1,0.1]) = 0.1
          breath 1 → mean([0.3,0.3]) = 0.3
          breath 2 → mean([0.5,0.5]) = 0.5
          breath 3 → mean([0.7,0.7]) = 0.7
          breath 4 → mean([0.9,0.9]) = 0.9

        Both series are identical and monotone → spearman_flattening_r = 1.0.

        labels_t25 = flg >= 0.25 → [F, T, T, T, T]
        positives(flat_sev) = [0.3,0.5,0.7,0.9], negatives = [0.1]
        All positives > negative → AUC = 1.0.
        """
        session_row = _make_mock_session_row(10)

        ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
        vs = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9]
        blob = _make_waveform_blob(ts, vs)
        waveform_mock = MagicMock()
        waveform_mock.data_blob = blob
        waveform_mock.sample_count = 10

        analysis_row = _make_analysis_result_mock(100)

        breaths = [
            _make_breath_mock(0.0, 4.0, 0.9, 0.1, 0),
            _make_breath_mock(4.0, 8.0, 0.7, 0.3, 1),
            _make_breath_mock(8.0, 12.0, 0.5, 0.5, 2),
            _make_breath_mock(12.0, 16.0, 0.3, 0.7, 3),
            _make_breath_mock(16.0, 20.0, 0.1, 0.9, 4),
        ]

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = breaths

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        assert len(report.sessions) == 1
        s = report.sessions[0]
        assert s.skipped_reason is None
        assert s.n_breaths_compared == 5

        assert s.spearman_flattening_r is not None
        assert abs(s.spearman_flattening_r - 1.0) < 1e-9

        assert s.auc_t25 is not None
        assert abs(s.auc_t25 - 1.0) < 1e-9


class TestFlowClassValidation:
    """flow_class weight vs FLG metrics, including fallback-confidence exclusion."""

    @pytest.mark.asyncio
    async def test_class_weight_metrics_exclude_fallback_breaths(self, mock_db_session):
        """
        Five rule-matched breaths whose class weight tracks FLG perfectly, plus a
        sixth fallback-confidence breath (flow_confidence == 0.5) whose FLG would
        break the correlation if it were not excluded.

        Rule-matched breaths (class → weight, breath-averaged FLG):
          class 1 (w=0.0) FLG 0.1
          class 4 (w=0.6) FLG 0.3
          class 5 (w=0.7) FLG 0.5
          class 6 (w=0.9) FLG 0.7
          class 7 (w=1.0) FLG 0.9
        Monotone → spearman_class_weight_r = 1.0; positives (FLG>=0.25) all
        outscore the single negative → auc_class_t25 = 1.0.

        Excluded breath: class 7 (w=1.0) conf 0.5 FLG 0.05 — included, it would
        pair the max weight with the min FLG and drop Spearman below 1.0.
        """
        session_row = _make_mock_session_row(11)

        ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
        vs = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 0.05, 0.05]
        blob = _make_waveform_blob(ts, vs)
        waveform_mock = MagicMock()
        waveform_mock.data_blob = blob
        waveform_mock.sample_count = 12

        analysis_row = _make_analysis_result_mock(101)

        breaths = [
            _make_breath_mock(0.0, 4.0, 0.9, 0.1, 0, flow_class=1, flow_confidence=0.8),
            _make_breath_mock(4.0, 8.0, 0.7, 0.3, 1, flow_class=4, flow_confidence=0.8),
            _make_breath_mock(
                8.0, 12.0, 0.5, 0.5, 2, flow_class=5, flow_confidence=0.8
            ),
            _make_breath_mock(
                12.0, 16.0, 0.3, 0.7, 3, flow_class=6, flow_confidence=0.8
            ),
            _make_breath_mock(
                16.0, 20.0, 0.1, 0.9, 4, flow_class=7, flow_confidence=0.8
            ),
            _make_breath_mock(
                20.0, 24.0, 0.5, 0.5, 5, flow_class=7, flow_confidence=0.5
            ),
        ]

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = breaths

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        s = report.sessions[0]
        assert s.skipped_reason is None
        # Only the five rule-matched breaths enter the class-weight metrics.
        assert s.n_class_breaths_compared == 5
        assert s.spearman_class_weight_r is not None
        assert abs(s.spearman_class_weight_r - 1.0) < 1e-9
        assert s.auc_class_t25 is not None
        assert abs(s.auc_class_t25 - 1.0) < 1e-9
        # t50 positives (FLG>=0.50: weights 0.7/0.9/1.0) all outscore the two
        # negatives (0.0/0.6) → perfect separation.
        assert s.auc_class_t50 is not None
        assert abs(s.auc_class_t50 - 1.0) < 1e-9

        # Aggregate means propagate the per-session class-weight metrics.
        assert report.aggregate.mean_spearman_class_weight_r is not None
        assert abs(report.aggregate.mean_spearman_class_weight_r - 1.0) < 1e-9

    @pytest.mark.asyncio
    async def test_no_classified_breaths_yields_none_class_metrics(
        self, mock_db_session
    ):
        """Breaths without flow_class → class-weight metrics stay None, flattening unaffected."""
        session_row = _make_mock_session_row(12)

        ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        vs = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5]
        blob = _make_waveform_blob(ts, vs)
        waveform_mock = MagicMock()
        waveform_mock.data_blob = blob
        waveform_mock.sample_count = 6

        analysis_row = _make_analysis_result_mock(102)

        # flow_class defaults to None (unclassified)
        breaths = [
            _make_breath_mock(0.0, 4.0, 0.9, 0.1, 0),
            _make_breath_mock(4.0, 8.0, 0.7, 0.3, 1),
            _make_breath_mock(8.0, 12.0, 0.5, 0.5, 2),
        ]

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]

        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock

        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row

        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = breaths

        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")

        s = report.sessions[0]
        assert s.skipped_reason is None
        assert s.spearman_class_weight_r is None
        assert s.auc_class_t25 is None
        assert s.auc_class_t50 is None


class TestScoreFlArraysExtractionEquivalence:
    """Pin the extraction: the values ``_validate_session`` reports are exactly
    what the pure ``score_fl_arrays`` core computes for the same input arrays."""

    @pytest.mark.asyncio
    async def test_report_fields_match_score_fl_arrays(self, mock_db_session):
        # Rule-matched breaths across the class ladder plus one fallback breath,
        # exercising both the flattening and class-weight AUC paths.
        ts = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
        vs = [0.1, 0.1, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.9, 0.9, 0.05, 0.05]
        # (start, end, mid_insp, flatness, flow_class, flow_confidence)
        specs = [
            (0.0, 4.0, 0.9, 0.1, 1, 0.8),
            (4.0, 8.0, 0.7, 0.3, 4, 0.8),
            (8.0, 12.0, 0.5, 0.5, 5, 0.8),
            (12.0, 16.0, 0.3, 0.7, 6, 0.8),
            (16.0, 20.0, 0.1, 0.9, 7, 0.8),
            (20.0, 24.0, 0.5, 0.5, 7, 0.5),  # fallback confidence — excluded
        ]
        breaths = [
            _make_breath_mock(s, e, mi, fi, i, flow_class=fc, flow_confidence=cf)
            for i, (s, e, mi, fi, fc, cf) in enumerate(specs)
        ]

        # Rebuild the arrays score_fl_arrays receives inside _validate_session.
        starts = np.array([b.start_offset_s for b in breaths], dtype=np.float64)
        ends = np.array([b.end_offset_s for b in breaths], dtype=np.float64)
        mid = np.array([b.mid_insp_flattening for b in breaths], dtype=np.float64)
        flatness = np.array([b.flatness_index for b in breaths], dtype=np.float64)
        class_weight = np.array(
            [
                FLOW_LIMITATION_CLASSES[b.flow_class]["weight"]
                if b.flow_class in FLOW_LIMITATION_CLASSES
                else np.nan
                for b in breaths
            ],
            dtype=np.float64,
        )
        rule_matched = np.array(
            [b.flow_confidence > FLC.FL_DEFAULT_CONFIDENCE for b in breaths],
            dtype=bool,
        )
        # Deserialize through the same loader so float32 rounding matches exactly.
        blob = _make_waveform_blob(ts, vs)
        flg_ts, flg_raw = deserialize_waveform_blob(blob, len(ts))
        valid_mask = (flg_raw >= 0.0) & np.isfinite(flg_raw)
        flg_vs = np.clip(flg_raw[valid_mask], 0.0, 1.0)
        breath_flg = average_waveform_over_breaths(
            starts,
            ends,
            flg_ts[valid_mask].astype(np.float64),
            flg_vs.astype(np.float64),
        )
        scores = score_fl_arrays(
            mid, flatness, class_weight, rule_matched, breath_flg, flg_vs
        )
        assert scores is not None

        session_row = _make_mock_session_row(20)
        waveform_mock = MagicMock()
        waveform_mock.data_blob = blob
        waveform_mock.sample_count = len(ts)
        analysis_row = _make_analysis_result_mock(120)

        sessions_result = MagicMock()
        sessions_result.scalars.return_value.all.return_value = [session_row]
        waveform_result = MagicMock()
        waveform_result.scalars.return_value.first.return_value = waveform_mock
        analysis_result = MagicMock()
        analysis_result.scalars.return_value.first.return_value = analysis_row
        breaths_result = MagicMock()
        breaths_result.scalars.return_value.all.return_value = breaths
        mock_db_session.execute = AsyncMock(
            side_effect=[
                sessions_result,
                waveform_result,
                analysis_result,
                breaths_result,
            ]
        )

        validator = FlowLimitationValidator(mock_db_session, profile_id=1)
        report = await validator.validate_date_range("2025-01-01", "2025-01-31")
        s = report.sessions[0]

        # The wrapper maps score_fl_arrays' threshold-agnostic fields back to the
        # report's published t25/t50 names — every metric must match exactly.
        assert s.n_breaths_compared == scores.n_breaths_compared
        assert s.n_class_breaths_compared == scores.n_class_breaths_compared
        assert s.spearman_flattening_r == scores.spearman_flattening_r
        assert s.spearman_flatness_r == scores.spearman_flatness_r
        assert s.spearman_class_weight_r == scores.spearman_class_weight_r
        assert s.auc_t25 == scores.auc_low
        assert s.auc_t50 == scores.auc_high
        assert s.auc_class_t25 == scores.auc_class_low
        assert s.auc_class_t50 == scores.auc_class_high
        assert s.snore_fl_95th == scores.snore_fl_95th
        assert s.device_flg_95th == scores.device_flg_95th
