"""Equivalence tests for the windowed ``match_events_by_start_time``.

The matcher also runs at analysis time (``validate_event_type``), whose output
is stored, so the windowed rewrite must pick exactly the same pairs as the
original quadratic scan for every input — sorted or not, with duplicates.
"""

import random

from collections.abc import Sequence

import pytest

from pydantic import BaseModel

from snore.analysis.modes.postprocess import match_events_by_start_time


class _Ev(BaseModel, frozen=True):
    start_time: float
    tag: int


def _reference_match(
    programmatic: Sequence[_Ev], machine: Sequence[_Ev], tolerance: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """The original O(n*m) algorithm, kept verbatim as the oracle."""
    matched: list[tuple[int, int]] = []
    matched_machine_indices: set[int] = set()
    false_positives: list[int] = []
    for prog_event in programmatic:
        match_found = False
        for m_idx, mach_event in enumerate(machine):
            if m_idx in matched_machine_indices:
                continue
            if abs(prog_event.start_time - mach_event.start_time) <= tolerance:
                matched.append((prog_event.tag, mach_event.tag))
                matched_machine_indices.add(m_idx)
                match_found = True
                break
        if not match_found:
            false_positives.append(prog_event.tag)
    false_negatives = [
        e.tag for i, e in enumerate(machine) if i not in matched_machine_indices
    ]
    return matched, false_positives, false_negatives


def _random_events(rng: random.Random, n: int, base: float, tag0: int) -> list[_Ev]:
    # Coarse grid → many exact duplicates and exact-tolerance boundary hits;
    # jitter → generic floats.  Shuffled half the time to exercise unsorted input.
    events = [
        _Ev(
            start_time=base
            + (
                rng.randrange(0, 60) * 2.5
                if rng.random() < 0.5
                else rng.uniform(0.0, 150.0)
            ),
            tag=tag0 + i,
        )
        for i in range(n)
    ]
    if rng.random() < 0.5:
        events.sort(key=lambda e: e.start_time)
    return events


@pytest.mark.parametrize("base", [0.0, 1_736_467_626.0])
def test_windowed_matcher_equals_reference_on_random_inputs(base):
    rng = random.Random(20261002)
    for _ in range(3000):
        tolerance = rng.choice([0.0, 2.5, 5.0, rng.uniform(0.0, 12.0)])
        prog = _random_events(rng, rng.randrange(0, 25), base, 0)
        mach = _random_events(rng, rng.randrange(0, 25), base, 1000)

        result = match_events_by_start_time(prog, mach, tolerance)

        assert (
            [(p.tag, m.tag) for p, m in result.matched],
            [e.tag for e in result.false_positives],
            [e.tag for e in result.false_negatives],
        ) == _reference_match(prog, mach, tolerance)


def test_unsorted_machine_picks_first_in_input_order_not_nearest():
    prog = [_Ev(start_time=10.0, tag=0)]
    mach = [_Ev(start_time=14.0, tag=1), _Ev(start_time=10.0, tag=2)]

    result = match_events_by_start_time(prog, mach, 5.0)

    assert [(p.tag, m.tag) for p, m in result.matched] == [(0, 1)]
    assert [e.tag for e in result.false_negatives] == [2]
