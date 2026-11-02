"""Median, 95th and max of device channels, pooling, and mask-on span arithmetic."""

import statistics

from datetime import datetime

import numpy as np
import pytest

from snore.services.channel_stats import (
    ChannelHistograms,
    channel_histogram,
    histogram_distribution,
    mask_on_spans,
    pool_channel_stats,
)
from snore.utils.stats import percentile_nearest_rank

pytestmark = pytest.mark.unit


def _samples(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _histogram(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Histogram with every finite sample kept: the snore channel has no floor."""
    return channel_histogram("snore", samples)


def _pool(
    fl: dict[int, np.ndarray] | None = None,
    snore: dict[int, np.ndarray] | None = None,
) -> ChannelHistograms:
    return {
        "fl": {sid: channel_histogram("fl", v) for sid, v in (fl or {}).items()},
        "snore": {sid: _histogram(v) for sid, v in (snore or {}).items()},
    }


def _distribution(*sessions: np.ndarray) -> tuple[float, float, float]:
    result = histogram_distribution([_histogram(session) for session in sessions])
    assert result is not None
    return result


def _sorted_sample_maths(*sessions: np.ndarray) -> tuple[float, float, float]:
    """The definitions, computed the slow way from every sample as a Python float."""
    ordered = sorted(float(v) for session in sessions for v in session)
    p95 = percentile_nearest_rank(ordered, 0.95)
    assert p95 is not None
    return statistics.median(ordered), p95, ordered[-1]


def test_distribution_of_no_sessions_is_none() -> None:
    assert histogram_distribution([]) is None


def test_distribution_of_sessions_with_no_samples_is_none() -> None:
    assert histogram_distribution([_histogram(_samples())]) is None


def test_distribution_skips_an_empty_session() -> None:
    assert _distribution(_samples(), _samples(0.5)) == (0.5, 0.5, 0.5)


def test_distribution_single_sample_is_every_statistic() -> None:
    assert _distribution(_samples(0.5)) == (0.5, 0.5, 0.5)


def test_distribution_95th_is_nearest_rank_of_sorted_values() -> None:
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
def test_distribution_matches_python_definitions(values: list[float]) -> None:
    samples = _samples(*values)

    assert _distribution(samples) == _sorted_sample_maths(samples)


def test_distribution_merges_a_value_shared_by_sessions() -> None:
    first = _samples(0.5, 0.5, 1.0)
    second = _samples(0.5, 2.0)

    assert _distribution(first, second) == _sorted_sample_maths(first, second)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "session_sizes",
    [(1,), (2,), (501,), (500,), (7, 13, 1), (7, 13, 2), (250, 1, 300, 49)],
    ids=lambda sizes: f"{sum(sizes)}-samples-{len(sizes)}-sessions",
)
@pytest.mark.parametrize("distinct_values", [8, None], ids=["repeating", "continuous"])
def test_histograms_equal_sorted_sample_maths(
    seed: int, session_sizes: tuple[int, ...], distinct_values: int | None
) -> None:
    rng = np.random.default_rng(seed)
    if distinct_values is None:
        sessions = [rng.random(size, dtype=np.float32) for size in session_sizes]
    else:
        levels = rng.random(distinct_values, dtype=np.float32)
        sessions = [rng.choice(levels, size) for size in session_sizes]

    expected = _sorted_sample_maths(*sessions)

    assert _distribution(*sessions) == expected
    pooled = pool_channel_stats(_pool(snore=dict(enumerate(sessions))), range(9))
    assert pooled is not None
    assert (pooled.snore_median, pooled.snore_95th, pooled.snore_max) == tuple(
        round(value, 4) for value in expected
    )


def test_fl_histogram_drops_the_sentinel_and_keeps_zeros() -> None:
    raw = _samples(-0.01, 0.0, 0.0, 0.25, float("nan"), float("inf"))

    values, counts = channel_histogram("fl", raw)

    assert values.tolist() == [0.0, 0.25]
    assert counts.tolist() == [2, 1]


def test_snore_histogram_keeps_negative_values_like_mcp() -> None:
    raw = _samples(-0.01, 0.0, 1.5, float("nan"), float("-inf"))

    values, counts = channel_histogram("snore", raw)

    assert values.tolist() == pytest.approx([-0.01, 0.0, 1.5])
    assert counts.tolist() == [1, 1, 1]


def test_pool_differs_from_averaging_session_percentiles() -> None:
    quiet = np.zeros(19, dtype=np.float32)
    loud = np.ones(1, dtype=np.float32)
    histograms = _pool(fl={1: quiet, 2: loud})

    pooled = pool_channel_stats(histograms, [1, 2])

    # Averaging the per-session 95ths would read (0 + 1) / 2 = 0.5.
    assert pooled.device_flg_95th == 1.0
    assert pooled.device_flg_median == 0.0


def test_pool_uses_only_the_requested_sessions() -> None:
    histograms = _pool(fl={1: _samples(0.2), 2: _samples(0.9)})

    assert pool_channel_stats(histograms, [1]).device_flg_max == pytest.approx(0.2)


def test_pool_rounds_to_four_places() -> None:
    histograms = _pool(snore={1: _samples(1.23456789)})

    assert pool_channel_stats(histograms, [1]).snore_max == 1.2346


def test_pool_channel_absent_is_null_for_that_channel_only() -> None:
    pooled = pool_channel_stats(_pool(fl={1: _samples(0.2)}), [1])

    assert pooled.device_flg_max == pytest.approx(0.2)
    assert pooled.snore_median is None
    assert pooled.snore_95th is None
    assert pooled.snore_max is None


@pytest.mark.parametrize(
    "histograms",
    [_pool(), _pool(fl={1: _samples()}), _pool(fl={1: _samples(-0.01)})],
    ids=["no-rows", "empty-blob", "only-sentinels-after-filtering"],
)
def test_pool_with_no_usable_samples_is_none(histograms: ChannelHistograms) -> None:
    assert pool_channel_stats(histograms, [1]) is None


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


@pytest.mark.parametrize(
    "segments",
    [[[0, 10**400]], [[-1.0, 60.0]], [[0, 1e12]], [[0, 1e15]]],
    ids=[
        "integer-too-large-for-a-float",
        "negative-start",
        "past-the-last-datetime",
        "too-large-for-a-timedelta",
    ],
)
def test_mask_on_spans_out_of_range_offsets_are_unknown(
    segments: object, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING"):
        assert mask_on_spans(7, _SESSION_START, segments) is None

    assert "Session 7" in caplog.text
