"""Median, 95th and max of device channels, pooling, and mask-on span arithmetic."""

import statistics

from datetime import datetime

import numpy as np
import pytest

from snore.services.channel_stats import (
    ChannelSamples,
    mask_on_spans,
    pool_channel_stats,
    sample_distribution,
    usable_samples,
)
from snore.utils.stats import percentile_nearest_rank

pytestmark = pytest.mark.unit


def _samples(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _pool(
    fl: dict[int, np.ndarray] | None = None,
    snore: dict[int, np.ndarray] | None = None,
) -> ChannelSamples:
    return {"fl": fl or {}, "snore": snore or {}}


def _distribution(samples: np.ndarray) -> tuple[float, float, float]:
    result = sample_distribution(samples)
    assert result is not None
    return result


def test_sample_distribution_empty_is_none() -> None:
    assert sample_distribution(_samples()) is None


def test_sample_distribution_single_sample_is_every_statistic() -> None:
    assert sample_distribution(_samples(0.5)) == (0.5, 0.5, 0.5)


def test_sample_distribution_95th_is_nearest_rank_of_sorted_values() -> None:
    twenty = np.arange(20, 0, -1, dtype=np.float32)

    _, p95, maximum = _distribution(twenty)

    # min(int(20 * 0.95), 19) = 19, the top value, where interpolation would not be.
    assert p95 == 20.0
    assert maximum == 20.0
    _, p95_of_21, _ = _distribution(np.arange(1, 22, dtype=np.float32))
    assert p95_of_21 == 20.0


@pytest.mark.parametrize(
    "values",
    [
        [0.1, 0.2, 0.4],
        [0.1, 0.2, 0.3, 0.7],
        [0.0, 0.0, 0.0, 0.9],
    ],
)
def test_sample_distribution_matches_python_definitions(values: list[float]) -> None:
    samples = _samples(*values)
    as_python = sorted(float(v) for v in samples)

    median, p95, maximum = _distribution(samples)

    assert median == statistics.median(as_python)
    assert p95 == percentile_nearest_rank(as_python, 0.95)
    assert maximum == as_python[-1]


def test_usable_samples_drops_fl_sentinel_and_keeps_zeros() -> None:
    raw = _samples(-0.01, 0.0, 0.25, float("nan"), float("inf"))

    assert usable_samples("fl", raw).tolist() == [0.0, 0.25]


def test_usable_samples_keeps_negative_snore_like_mcp() -> None:
    raw = _samples(-0.01, 0.0, 1.5, float("nan"), float("-inf"))

    assert usable_samples("snore", raw).tolist() == pytest.approx([-0.01, 0.0, 1.5])


def test_pool_differs_from_averaging_session_percentiles() -> None:
    quiet = np.zeros(19, dtype=np.float32)
    loud = np.ones(1, dtype=np.float32)
    samples = _pool(fl={1: quiet, 2: loud})

    pooled = pool_channel_stats(samples, [1, 2])

    # Averaging the per-session 95ths would read (0 + 1) / 2 = 0.5.
    assert pooled.device_flg_95th == 1.0
    assert pooled.device_flg_median == 0.0


def test_pool_uses_only_the_requested_sessions() -> None:
    samples = _pool(fl={1: _samples(0.2), 2: _samples(0.9)})

    assert pool_channel_stats(samples, [1]).device_flg_max == pytest.approx(0.2)


def test_pool_rounds_to_four_places() -> None:
    samples = _pool(snore={1: _samples(1.23456789)})

    assert pool_channel_stats(samples, [1]).snore_max == 1.2346


def test_pool_channel_absent_is_null_for_that_channel_only() -> None:
    pooled = pool_channel_stats(_pool(fl={1: _samples(0.2)}), [1])

    assert pooled.device_flg_max == pytest.approx(0.2)
    assert pooled.snore_median is None
    assert pooled.snore_95th is None
    assert pooled.snore_max is None


@pytest.mark.parametrize(
    "samples",
    [_pool(), _pool(fl={1: _samples()})],
    ids=["no-rows", "only-sentinels-after-filtering"],
)
def test_pool_with_no_usable_samples_is_none(samples: ChannelSamples) -> None:
    assert pool_channel_stats(samples, [1]) is None


_SESSION_START = datetime(2025, 3, 1, 22, 30, 0)


def test_mask_on_spans_add_offsets_to_the_wall_clock_start() -> None:
    spans = mask_on_spans(7, _SESSION_START, [[0.0, 3600.0], [5400.5, 9000.5]])

    assert [(s.session_id, s.start, s.end, s.duration_seconds) for s in spans] == [
        (7, _SESSION_START, datetime(2025, 3, 1, 23, 30, 0), 3600.0),
        (
            7,
            datetime(2025, 3, 2, 0, 0, 0, 500000),
            datetime(2025, 3, 2, 1, 0, 0, 500000),
            3600.0,
        ),
    ]


def test_mask_on_spans_keep_a_naive_wall_clock() -> None:
    spans = mask_on_spans(7, _SESSION_START, [[0.0, 60.0]])

    assert spans is not None
    span = spans[0]
    assert span.start.tzinfo is None
    assert span.end.tzinfo is None


def test_mask_on_spans_null_segments_are_unknown() -> None:
    assert mask_on_spans(7, _SESSION_START, None) is None


def test_mask_on_spans_empty_segments_are_known_zero() -> None:
    assert mask_on_spans(7, _SESSION_START, []) == []


@pytest.mark.parametrize(
    "segments",
    [[[100.0, 0.0]], [[0.0, float("inf")]], "garbage"],
    ids=["inverted", "infinite", "not-a-list"],
)
def test_mask_on_spans_malformed_segments_are_unknown(segments: object) -> None:
    assert mask_on_spans(7, _SESSION_START, segments) is None
