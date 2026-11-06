"""count_fl_run_reras_arrays counts exactly what _count_fl_run_reras counts.

The array counter reads nulls as NaN (flow_class, peak_flow_lpm) and the
recovery flag as a bool array; every case here builds both inputs from the
same random breath sequence and compares the counts.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from snore.services.breath.algorithms import (
    _count_fl_run_reras,
    count_fl_run_reras_arrays,
)

type _Breath = tuple[int | None, bool | None, float | None]


def _count_both(breaths: list[_Breath], **tunables: float) -> tuple[int, int]:
    """(row-based count, array-based count) over the same breaths."""
    rows = [
        SimpleNamespace(flow_class=fc, is_recovery_breath=rec, peak_flow_lpm=peak)
        for fc, rec, peak in breaths
    ]
    flow_class = np.array([fc for fc, _, _ in breaths], dtype=np.float64)
    is_recovery = np.array([rec is True for _, rec, _ in breaths], dtype=bool)
    peaks = np.array([peak for _, _, peak in breaths], dtype=np.float64)
    return (
        _count_fl_run_reras(rows, **tunables),
        count_fl_run_reras_arrays(flow_class, is_recovery, peaks, **tunables),
    )


def _random_breaths(rng: np.random.Generator) -> list[_Breath]:
    n = int(rng.integers(0, 40))
    # A few run peaks plus followers that land near (1 + margin) x mean, so
    # threshold ties and near-ties come up often.
    peak_choices = [20.0, 24.0, 25.0, 26.0, 30.0, 0.0, 1e-9]
    breaths: list[_Breath] = []
    for _ in range(n):
        fc = None if rng.random() < 0.12 else int(rng.integers(0, 8))
        rec_draw = rng.random()
        rec = None if rec_draw < 0.1 else bool(rec_draw < 0.25)
        peak_draw = rng.random()
        if peak_draw < 0.15:
            peak = None
        elif peak_draw < 0.6:
            peak = float(rng.choice(peak_choices))
        else:
            peak = float(rng.normal(25.0, 6.0))
        breaths.append((fc, rec, peak))
    return breaths


@pytest.mark.unit
class TestArrayCounterMatchesRowCounter:
    def test_random_sequences_with_default_tunables_match(self) -> None:
        rng = np.random.default_rng(468)
        for _ in range(5000):
            breaths = _random_breaths(rng)
            row_count, array_count = _count_both(breaths)
            assert array_count == row_count, breaths

    def test_random_sequences_with_custom_tunables_match(self) -> None:
        rng = np.random.default_rng(469)
        for _ in range(5000):
            breaths = _random_breaths(rng)
            tunables: dict[str, float] = {
                "fl_class_threshold": int(rng.integers(1, 7)),
                "min_fl_run_length": int(rng.integers(0, 5)),
                "recovery_amplitude_margin": float(rng.choice([0.0, 0.05, 0.2, 1.0])),
            }
            row_count, array_count = _count_both(breaths, **tunables)
            assert array_count == row_count, (breaths, tunables)

    @pytest.mark.parametrize(
        ("breaths", "expected"),
        [
            pytest.param([], 0, id="empty"),
            pytest.param([(5, None, 20.0), (5, None, 20.0)], 0, id="run-at-end"),
            pytest.param(
                [(5, None, None), (5, None, None), (1, False, 100.0)],
                0,
                id="all-null-run-peaks",
            ),
            pytest.param(
                [(5, None, None), (5, None, None), (1, True, None)],
                1,
                id="flagged-follower-needs-no-peaks",
            ),
            pytest.param(
                [(5, None, 20.0), (None, None, 20.0), (5, None, 20.0), (1, None, 26.0)],
                0,
                id="null-class-splits-run",
            ),
            pytest.param(
                [(5, None, 20.0), (4, None, 20.0), (1, None, 24.0)],
                1,
                id="follower-at-exact-margin",
            ),
            pytest.param(
                [(5, None, 20.0), (5, None, 20.0), (3, None, 40.0)],
                0,
                id="class-3-follower",
            ),
        ],
    )
    def test_edge_cases_match_and_count(
        self, breaths: list[_Breath], expected: int
    ) -> None:
        assert _count_both(breaths) == (expected, expected)
