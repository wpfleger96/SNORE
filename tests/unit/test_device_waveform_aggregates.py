"""Unit tests for device channel aggregation in _device_night_stats.

Tests cover:
- Correct median/95th/max computation for device FL channel
- Negative sentinel filter for FL (values < 0 dropped; zeros retained)
- snore_pct_time threshold (> 0.5, not >= 0.5)
- Leak time over the 24 L/min redline (inclusive boundary, sentinels excluded)
- Empty channel → null fields + CHANNEL_ABSENT reason
- Multi-session night aggregation (values merged across sessions)
- Compliance from Day hours, falling back to session durations
"""

from __future__ import annotations

import math
import statistics as stats_lib

from datetime import date
from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_mock_session(session_id: int) -> Any:
    """Minimal mock for a DB Session ORM row."""
    s = MagicMock()
    s.id = session_id
    s.duration_seconds = 28800.0  # 8 hours
    return s


def _call_build(
    *,
    day_sessions: list[Any],
    fl_by_session: dict[int, list[float]] | None = None,
    snore_by_session: dict[int, list[float]] | None = None,
    leak_by_session: dict[int, list[float]] | None = None,
    day_row: Any | None = None,
) -> Any:
    """Call _device_night_stats with the given per-session channel samples."""
    from snore.services.breath.nightly import _device_night_stats  # noqa: PLC0415

    return _device_night_stats(
        therapy_date=date(2025, 6, 1),
        device_id=1,
        day_sessions=day_sessions,
        day_row=day_row,
        compliance_threshold_hours=4.0,
        channel_vals={
            "fl": fl_by_session or {},
            "snore": snore_by_session or {},
            "leak": leak_by_session or {},
        },
    )


@pytest.mark.unit
class TestDeviceFLAggregation:
    def test_fl_median_95th_max_computed(self) -> None:
        """Median, 95th percentile, and max are computed from FL samples."""
        sess = _make_mock_session(1)
        fl_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = _call_build(
            day_sessions=[sess],
            fl_by_session={1: fl_vals},
        )
        assert result.device_flg_median is not None
        assert abs(result.device_flg_median - stats_lib.median(fl_vals)) < 1e-6
        assert result.device_flg_95th is not None
        assert result.device_flg_95th <= 1.0
        assert result.device_flg_max == pytest.approx(1.0)
        assert result.device_flg_reason is None

    def test_negative_sentinel_filtered_out(self) -> None:
        """FL values < 0 (mask-off sentinel −0.01) are excluded from aggregation."""
        sess = _make_mock_session(1)
        # Two negative sentinels, three legitimate values
        fl_vals = [-0.01, -0.01, 0.3, 0.5, 0.7]
        result = _call_build(
            day_sessions=[sess],
            fl_by_session={1: fl_vals},
        )
        # Only [0.3, 0.5, 0.7] contribute
        assert result.device_flg_median == pytest.approx(0.5)
        assert result.device_flg_max == pytest.approx(0.7)

    def test_zeros_are_retained(self) -> None:
        """FL value 0.0 is legitimate therapy data and must not be filtered."""
        sess = _make_mock_session(1)
        fl_vals = [0.0, 0.0, 0.5]
        result = _call_build(
            day_sessions=[sess],
            fl_by_session={1: fl_vals},
        )
        # All three values contribute; median of [0, 0, 0.5] = 0
        assert result.device_flg_median == pytest.approx(0.0)
        assert result.device_flg_max == pytest.approx(0.5)

    def test_empty_fl_channel_gives_null_reason(self) -> None:
        """No FL values → all FL fields null, reason = channel_absent."""
        from snore.analysis.shared.versioning import NullReason  # noqa: PLC0415

        sess = _make_mock_session(1)
        result = _call_build(
            day_sessions=[sess],
            fl_by_session=None,
        )
        assert result.device_flg_median is None
        assert result.device_flg_95th is None
        assert result.device_flg_max is None
        assert result.device_flg_reason == NullReason.CHANNEL_ABSENT

    def test_all_negatives_treated_as_absent(self) -> None:
        """If all FL values are negative sentinels, the result is null + channel_absent."""
        from snore.analysis.shared.versioning import NullReason  # noqa: PLC0415

        sess = _make_mock_session(1)
        result = _call_build(
            day_sessions=[sess],
            fl_by_session={1: [-0.01, -0.01, -0.01]},
        )
        assert result.device_flg_median is None
        assert result.device_flg_reason == NullReason.CHANNEL_ABSENT


@pytest.mark.unit
class TestSnoreAggregation:
    def test_snore_median_95th_computed(self) -> None:
        """Snore median and 95th percentile computed from all samples."""
        sess = _make_mock_session(1)
        snore_vals = [0.0, 0.0, 0.2, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 5.0]
        result = _call_build(
            day_sessions=[sess],
            snore_by_session={1: snore_vals},
        )
        assert result.snore_median is not None
        assert abs(result.snore_median - stats_lib.median(snore_vals)) < 1e-6
        assert result.snore_95th is not None
        assert result.snore_reason is None

    def test_snore_pct_time_above_0_5(self) -> None:
        """snore_pct_time is the fraction of samples strictly > 0.5."""
        sess = _make_mock_session(1)
        # 3 out of 10 samples are > 0.5: 0.6, 0.7, 0.8 (0.5 does not count)
        snore_vals = [0.0, 0.0, 0.5, 0.5, 0.6, 0.7, 0.8, 0.0, 0.0, 0.0]
        result = _call_build(
            day_sessions=[sess],
            snore_by_session={1: snore_vals},
        )
        assert result.snore_pct_time == pytest.approx(0.3)  # 3/10

    def test_snore_threshold_is_exclusive(self) -> None:
        """Exactly 0.5 does NOT count as snoring (threshold is > 0.5)."""
        sess = _make_mock_session(1)
        snore_vals = [0.5, 0.5, 0.5, 0.51]
        result = _call_build(
            day_sessions=[sess],
            snore_by_session={1: snore_vals},
        )
        # Only 0.51 qualifies
        assert result.snore_pct_time == pytest.approx(0.25)

    def test_snore_zeros_retained(self) -> None:
        """Zero snore values are legitimate and contribute to statistics."""
        sess = _make_mock_session(1)
        snore_vals = [0.0, 0.0, 0.0, 0.0]
        result = _call_build(
            day_sessions=[sess],
            snore_by_session={1: snore_vals},
        )
        assert result.snore_median == pytest.approx(0.0)
        assert result.snore_pct_time == pytest.approx(0.0)
        assert result.snore_reason is None

    def test_empty_snore_channel_gives_null_reason(self) -> None:
        """No snore values → all snore fields null, reason = channel_absent."""
        from snore.analysis.shared.versioning import NullReason  # noqa: PLC0415

        sess = _make_mock_session(1)
        result = _call_build(
            day_sessions=[sess],
            snore_by_session=None,
        )
        assert result.snore_median is None
        assert result.snore_95th is None
        assert result.snore_pct_time is None
        assert result.snore_reason == NullReason.CHANNEL_ABSENT


@pytest.mark.unit
class TestMultiSessionNightAggregation:
    def test_values_merged_across_sessions(self) -> None:
        """FL and snore values from multiple sessions of one night are aggregated together."""
        sess1 = _make_mock_session(1)
        sess2 = _make_mock_session(2)
        # Session 1: FL [0.1, 0.2], Session 2: FL [0.3, 0.4]
        fl_by = {1: [0.1, 0.2], 2: [0.3, 0.4]}
        result = _call_build(
            day_sessions=[sess1, sess2],
            fl_by_session=fl_by,
        )
        # All 4 values merged: median of [0.1, 0.2, 0.3, 0.4] = 0.25
        assert result.device_flg_median == pytest.approx(0.25)
        assert result.device_flg_max == pytest.approx(0.4)

    def test_partial_session_coverage(self) -> None:
        """Night with two sessions where only one has waveform data: uses available data."""

        sess1 = _make_mock_session(1)
        sess2 = _make_mock_session(2)
        # Only sess1 has FL data
        result = _call_build(
            day_sessions=[sess1, sess2],
            fl_by_session={1: [0.2, 0.4, 0.6]},
        )
        assert result.device_flg_median == pytest.approx(0.4)
        assert result.device_flg_reason is None

    def test_leak_samples_merged_across_sessions(self) -> None:
        """Leak percent is pooled over samples, not averaged per session."""
        sess1 = _make_mock_session(1)
        sess2 = _make_mock_session(2)
        result = _call_build(
            day_sessions=[sess1, sess2],
            leak_by_session={1: [30.0, 0.0, 0.0, 0.0], 2: [0.0] * 6},
        )
        assert result.leak_above_24_pct == pytest.approx(10.0)  # 1 of 10 samples


@pytest.mark.unit
class TestLeakAbove24Pct:
    def test_boundary_sample_counts_as_above(self) -> None:
        """Exactly 24 L/min counts: the redline is inclusive."""
        result = _call_build(
            day_sessions=[_make_mock_session(1)],
            leak_by_session={1: [23.99, 24.0, 10.0, 0.0]},
        )
        assert result.leak_above_24_pct == pytest.approx(25.0)
        assert result.leak_above_24_pct_reason is None

    def test_rounded_to_one_decimal(self) -> None:
        result = _call_build(
            day_sessions=[_make_mock_session(1)],
            leak_by_session={1: [30.0, 0.0, 0.0]},
        )
        assert result.leak_above_24_pct == 33.3

    @pytest.mark.parametrize("sentinel", [-1.0, -0.02, math.nan, math.inf, -math.inf])
    def test_sentinels_excluded_from_numerator_and_denominator(
        self, sentinel: float
    ) -> None:
        """Negative and non-finite samples count on neither side of the percent."""
        result = _call_build(
            day_sessions=[_make_mock_session(1)],
            leak_by_session={1: [sentinel, sentinel, 30.0, 0.0]},
        )
        assert result.leak_above_24_pct == pytest.approx(50.0)

    def test_zero_leak_is_zero_percent(self) -> None:
        """A recorded night with no leak over the redline is 0.0, not null."""
        result = _call_build(
            day_sessions=[_make_mock_session(1)],
            leak_by_session={1: [0.0, 5.0]},
        )
        assert result.leak_above_24_pct == 0.0
        assert result.leak_above_24_pct_reason is None

    @pytest.mark.parametrize("samples", [None, {1: [-1.0, math.nan]}])
    def test_no_usable_samples_gives_null_reason(
        self, samples: dict[int, list[float]] | None
    ) -> None:
        """No samples, or only sentinels → null + channel_absent."""
        from snore.analysis.shared.versioning import NullReason  # noqa: PLC0415

        result = _call_build(
            day_sessions=[_make_mock_session(1)], leak_by_session=samples
        )
        assert result.leak_above_24_pct is None
        assert result.leak_above_24_pct_reason == NullReason.CHANNEL_ABSENT


@pytest.mark.unit
class TestDeviceNightCompliance:
    def test_uses_day_therapy_hours_when_present(self) -> None:
        """Day.total_therapy_hours wins over the 8 h session span."""
        day = MagicMock()
        day.total_therapy_hours = 3.5
        result = _call_build(day_sessions=[_make_mock_session(1)], day_row=day)
        assert result.total_therapy_hours == pytest.approx(3.5)
        assert result.is_compliant is False

    def test_falls_back_to_session_durations_without_day_hours(self) -> None:
        result = _call_build(day_sessions=[_make_mock_session(1)])
        assert result.total_therapy_hours == pytest.approx(8.0)
        assert result.is_compliant is True
