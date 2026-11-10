"""Large Leak span detection and the time over the leak redline."""

import numpy as np
import pytest

from snore.services.channel_stats import channel_histogram, pool_histograms
from snore.services.leak_redline import large_leak_offsets, leak_redline_pct

# Leak is recorded at 0.5 Hz: one sample every 2 seconds.
_RATE = 0.5


def _spans(
    values: list[float], timestamps: list[float] | None = None
) -> list[list[float]]:
    """Large Leak spans of samples taken every 2 seconds unless ``timestamps``."""
    offsets = (
        timestamps if timestamps is not None else [2.0 * i for i in range(len(values))]
    )
    return large_leak_offsets(
        np.array(offsets, dtype=np.float32), np.array(values, dtype=np.float32), _RATE
    ).tolist()


def test_span_closes_at_first_sample_below_redline() -> None:
    assert _spans([10.0, 30.0, 40.0, 10.0, 10.0]) == [[2.0, 6.0]]


def test_span_open_at_session_end_closes_one_period_after_last_sample() -> None:
    assert _spans([10.0, 30.0, 30.0]) == [[2.0, 6.0]]


def test_sample_exactly_at_redline_opens_a_span() -> None:
    assert _spans([23.9, 24.0, 23.9]) == [[2.0, 4.0]]


def test_span_ending_on_a_sample_exactly_at_redline_is_kept() -> None:
    # OSCAR drops this span (it closes only after a sample strictly above).
    assert _spans([30.0, 24.0, 10.0]) == [[0.0, 4.0]]


def test_lone_sample_at_recording_end_lasts_one_period() -> None:
    assert _spans([10.0, 10.0, 30.0]) == [[4.0, 6.0]]


def test_several_spans_in_one_segment_are_separate() -> None:
    assert _spans([30.0, 10.0, 30.0, 30.0, 10.0]) == [[0.0, 2.0], [4.0, 8.0]]


def test_span_never_crosses_a_recording_gap() -> None:
    # Two mask-on segments, leak above the redline across the mask-off gap.
    timestamps = [0.0, 2.0, 4.0, 600.0, 602.0, 604.0]
    values = [10.0, 30.0, 30.0, 30.0, 30.0, 10.0]

    assert _spans(values, timestamps) == [[2.0, 6.0], [600.0, 604.0]]


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        # 1.5 sample periods (3 s) is still one recording; wider is a gap.
        (3.0, [[0.0, 5.0]]),
        (3.5, [[0.0, 2.0], [3.5, 5.5]]),
    ],
)
def test_gap_starts_past_one_and_a_half_sample_periods(
    step: float, expected: list[list[float]]
) -> None:
    assert _spans([30.0, 30.0], [0.0, step]) == expected


def test_sentinel_and_nan_samples_close_a_span() -> None:
    assert _spans([30.0, -1.0, 30.0, float("nan")]) == [[0.0, 2.0], [4.0, 6.0]]


def test_no_samples_have_no_spans() -> None:
    assert _spans([]) == []


def test_leak_below_redline_has_no_spans() -> None:
    assert _spans([0.0, 12.0, 23.9]) == []


@pytest.mark.parametrize(
    "timestamps",
    [
        [0.0, float("nan"), 4.0],
        [0.0, float("inf"), 4.0],
        [0.0, 4.0, 2.0],
        [0.0, 2.0, 2.0],
        [-2.0, 0.0, 2.0],
    ],
)
def test_unusable_sample_times_are_rejected(timestamps: list[float]) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _spans([30.0, 30.0, 30.0], timestamps)


@pytest.mark.parametrize("rate", [0.0, -0.5, float("nan")])
def test_unusable_sample_rate_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="sample rate"):
        large_leak_offsets(np.array([0.0]), np.array([30.0]), rate)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # Sentinels are not samples: 2 of the 3 valid samples reach 24.
        ([-1.0, 10.0, 24.0, 30.0], 66.7),
        ([10.0, 23.99], 0.0),
        ([-1.0], None),
    ],
)
def test_leak_redline_pct_counts_valid_samples_at_or_above_redline(
    values: list[float], expected: float | None
) -> None:
    pooled = pool_histograms([channel_histogram("leak", np.array(values))])

    assert leak_redline_pct(pooled) == expected
