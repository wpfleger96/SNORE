"""The experimental night's array reduction matches the row-based one it replaced.

``_build_nightly_summary`` used to loop over each OK session's breath rows;
it now pools per-run ``_reduce_night_run`` results built from float arrays
(nulls as NaN, booleans as 1.0/0.0).  The reference below is the old row loop,
kept verbatim in spirit: leak-valid FL / Ti / I:E lists, the rule-matched
FL-class gate (``flow_confidence > FL_DEFAULT_CONFIDENCE``) and the RERA proxy
over every breath of a run.  Each case builds both from the same random rows
and compares the night's fields exactly.
"""

from __future__ import annotations

import random
import statistics

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from pydantic import BaseModel

from snore.analysis.shared.versioning import (
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisRunMetadata,
    AnalysisStatus,
)
from snore.constants import FlowLimitationConstants as FLC
from snore.database import models
from snore.services.breath._core import SessionClassification
from snore.services.breath.algorithms import _count_fl_run_reras
from snore.services.breath.dtos import NightlyAnalysisSummary
from snore.services.breath.nightly import (
    _NIGHT_BREATH_COLUMNS,
    NightlyMixin,
    _NightRunBreaths,
    _reduce_night_run,
)
from snore.utils.stats import percentile_nearest_rank

# One breath row in ``_NIGHT_BREATH_COLUMNS`` order: leak_valid,
# mid_insp_flattening, inspiration_time_s, i_e_ratio, flow_class,
# flow_confidence, is_recovery_breath, peak_flow_lpm.
type _Row = tuple[
    bool | None,
    float | None,
    float | None,
    float | None,
    int | None,
    float | None,
    bool | None,
    float | None,
]

_COMPARED_FIELDS = (
    "fl_median",
    "fl_5th",
    "fl_min",
    "fl_class_ge4_pct",
    "ti_median_s",
    "ie_ratio_median",
    "rera_count",
)

_ALGO = AlgoVersions(
    identity=AlgorithmIdentity.current(),
    run=AnalysisRunMetadata(primary_mode="aasm", modes=["aasm"]),
)
_DEFAULT = FLC.FL_DEFAULT_CONFIDENCE


class _NightSession(BaseModel):
    """One session of a night: its latest run's status, run id and breaths."""

    session_id: int
    status: AnalysisStatus
    ar_id: int | None
    breaths: list[_Row]


def _old_low_tail(
    vals: list[float],
) -> tuple[float | None, float | None, float | None]:
    if not vals:
        return None, None, None
    sorted_v = sorted(vals)
    return (
        float(statistics.median(sorted_v)),
        percentile_nearest_rank(sorted_v, 0.05),
        sorted_v[0],
    )


def _old_reference(sessions: list[_NightSession]) -> dict[str, float | int | None]:
    """The night's compared fields as the old row-based loop computed them."""
    ok = [s for s in sessions if s.status == AnalysisStatus.OK]
    if not ok:
        return dict.fromkeys(_COMPARED_FIELDS)
    fl_vals: list[float] = []
    ti_vals: list[float] = []
    ie_vals: list[float] = []
    fl_class_ge4_num = 0
    fl_class_den = 0
    rera_count = 0
    for s in ok:
        if s.ar_id is None:
            continue
        rows = [
            SimpleNamespace(
                leak_valid=lv,
                mid_insp_flattening=fl,
                inspiration_time_s=ti,
                i_e_ratio=ie,
                flow_class=fc,
                flow_confidence=conf,
                is_recovery_breath=rec,
                peak_flow_lpm=peak,
            )
            for lv, fl, ti, ie, fc, conf, rec, peak in s.breaths
        ]
        for b in rows:
            if b.leak_valid is True:
                if b.mid_insp_flattening is not None:
                    fl_vals.append(b.mid_insp_flattening)
                if b.inspiration_time_s is not None:
                    ti_vals.append(b.inspiration_time_s)
                if b.i_e_ratio is not None:
                    ie_vals.append(b.i_e_ratio)
                if (
                    b.flow_class is not None
                    and b.flow_confidence is not None
                    and b.flow_confidence > FLC.FL_DEFAULT_CONFIDENCE
                ):
                    fl_class_den += 1
                    if b.flow_class >= 4:
                        fl_class_ge4_num += 1
        rera_count += _count_fl_run_reras(rows)
    fl_median, fl_5th, fl_min = _old_low_tail(fl_vals)
    return {
        "fl_median": fl_median,
        "fl_5th": fl_5th,
        "fl_min": fl_min,
        "fl_class_ge4_pct": (
            100.0 * fl_class_ge4_num / fl_class_den if fl_class_den > 0 else None
        ),
        "ti_median_s": float(statistics.median(ti_vals)) if ti_vals else None,
        "ie_ratio_median": float(statistics.median(ie_vals)) if ie_vals else None,
        "rera_count": rera_count,
    }


def _run_array(rows: list[_Row]) -> np.ndarray:
    """Rows as the breath stream hands them to a reducer."""
    return np.array(list(zip(*rows, strict=True)), dtype=np.float64)


def _new_summary(sessions: list[_NightSession]) -> NightlyAnalysisSummary:
    """The night as ``_build_nightly_summary`` builds it from reduced runs.

    Every run with breaths is reduced, non-OK ones included, so the builder's
    own OK gate decides what counts.
    """
    runs: dict[int, _NightRunBreaths] = {
        s.ar_id: _reduce_night_run(_run_array(s.breaths))
        for s in sessions
        if s.ar_id is not None and s.breaths
    }
    ar_classification: dict[int, SessionClassification] = {
        s.session_id: (
            s.status,
            _ALGO if s.status == AnalysisStatus.OK else None,
            s.ar_id,
        )
        for s in sessions
    }
    return NightlyMixin._build_nightly_summary(
        therapy_date=date(2025, 3, 1),
        device_id=1,
        day_sessions=[
            models.Session(id=s.session_id, duration_seconds=3600.0) for s in sessions
        ],
        day_row=None,
        ar_classification=ar_classification,
        runs=runs,
    )


def _compared(summary: NightlyAnalysisSummary) -> dict[str, float | int | None]:
    return {field: getattr(summary, field) for field in _COMPARED_FIELDS}


def _assert_python_scalars(fields: dict[str, float | int | None]) -> None:
    for field, value in fields.items():
        expected_type = int if field == "rera_count" else float
        assert value is None or type(value) is expected_type, (field, type(value))


def _maybe(rng: random.Random, p_null: float, value: float) -> float | None:
    return None if rng.random() < p_null else value


def _random_value(
    rng: random.Random, pool: list[float], low: float, high: float
) -> float:
    # Pool draws repeat values (and integer-valued floats) so ties come up.
    return rng.choice(pool) if rng.random() < 0.5 else rng.uniform(low, high)


def _random_row(rng: random.Random) -> _Row:
    confidence = rng.choice(
        [None, rng.uniform(0.0, _DEFAULT), _DEFAULT, rng.uniform(_DEFAULT, 1.0), 1.0]
    )
    return (
        rng.choice([True, True, True, False, None]),
        _maybe(rng, 0.15, _random_value(rng, [0.0, 0.25, 0.5, 1.0], 0.0, 1.0)),
        _maybe(rng, 0.15, _random_value(rng, [1.0, 1.5, 2.0], 0.5, 3.0)),
        _maybe(rng, 0.15, _random_value(rng, [0.5, 1.0, 2.0], 0.2, 2.5)),
        None if rng.random() < 0.12 else rng.randint(1, 7),
        confidence,
        rng.choice([True, False, False, None]),
        _maybe(rng, 0.15, _random_value(rng, [20.0, 24.0, 25.0, 30.0], 10.0, 40.0)),
    )


def _random_night(rng: random.Random, first_id: int) -> list[_NightSession]:
    sessions: list[_NightSession] = []
    for k in range(rng.randint(1, 3)):
        sid = first_id + k
        status_draw = rng.random()
        if status_draw < 0.1:
            status, ar_id = AnalysisStatus.NOT_RUN, None
        elif status_draw < 0.2:
            # A stale run keeps its id; its breaths must not count either.
            status, ar_id = AnalysisStatus.STALE_VERSION, sid
        else:
            status, ar_id = AnalysisStatus.OK, sid
        # Lengths 0 (an OK run with no breaths, absent from ``runs``), 1, more.
        n = rng.choice([0, 1, rng.randint(2, 30)])
        breaths = [] if ar_id is None else [_random_row(rng) for _ in range(n)]
        sessions.append(
            _NightSession(session_id=sid, status=status, ar_id=ar_id, breaths=breaths)
        )
    return sessions


def _breath(
    *,
    leak_valid: bool | None = True,
    flattening: float | None = None,
    ti: float | None = None,
    ie: float | None = None,
    flow_class: int | None = None,
    confidence: float | None = None,
    recovery: bool | None = False,
    peak: float | None = None,
) -> _Row:
    return (leak_valid, flattening, ti, ie, flow_class, confidence, recovery, peak)


def _ok_night(*runs: list[_Row]) -> list[_NightSession]:
    return [
        _NightSession(
            session_id=k, status=AnalysisStatus.OK, ar_id=100 + k, breaths=breaths
        )
        for k, breaths in enumerate(runs, start=1)
    ]


@pytest.mark.unit
class TestNightArraysMatchRows:
    def test_breath_columns_match_row_layout(self) -> None:
        assert [c.key for c in _NIGHT_BREATH_COLUMNS] == [
            "leak_valid",
            "mid_insp_flattening",
            "inspiration_time_s",
            "i_e_ratio",
            "flow_class",
            "flow_confidence",
            "is_recovery_breath",
            "peak_flow_lpm",
        ]

    def test_random_nights_match_row_reduction(self) -> None:
        rng = random.Random(20261107)
        fl_parities: set[int] = set()
        for night in range(3000):
            sessions = _random_night(rng, first_id=10 * night)
            expected = _old_reference(sessions)
            actual = _compared(_new_summary(sessions))
            assert actual == expected, sessions
            _assert_python_scalars(actual)
            fl_count = sum(
                row[0] is True and row[1] is not None
                for s in sessions
                if s.status == AnalysisStatus.OK
                for row in s.breaths
            )
            if fl_count:
                fl_parities.add(fl_count % 2)
        # Both median branches ran.
        assert fl_parities == {0, 1}

    def test_night_without_breaths_has_no_distributions(self) -> None:
        sessions = _ok_night([])

        actual = _compared(_new_summary(sessions))

        assert actual == {**dict.fromkeys(_COMPARED_FIELDS), "rera_count": 0}
        assert actual == _old_reference(sessions)

    def test_single_breath_is_every_statistic(self) -> None:
        sessions = _ok_night(
            [
                _breath(
                    flattening=0.3,
                    ti=1.2,
                    ie=0.5,
                    flow_class=5,
                    confidence=0.9,
                    peak=20.0,
                )
            ]
        )

        actual = _compared(_new_summary(sessions))

        assert actual == {
            "fl_median": 0.3,
            "fl_5th": 0.3,
            "fl_min": 0.3,
            "fl_class_ge4_pct": 100.0,
            "ti_median_s": 1.2,
            "ie_ratio_median": 0.5,
            "rera_count": 0,
        }
        assert actual == _old_reference(sessions)
        _assert_python_scalars(actual)

    def test_even_count_across_sessions_averages_middle_pair(self) -> None:
        sessions = _ok_night(
            [_breath(flattening=0.4, ti=2.0)],
            [
                _breath(flattening=0.1, ti=1.0),
                _breath(leak_valid=False, flattening=0.0),
            ],
        )

        actual = _compared(_new_summary(sessions))

        assert (actual["fl_median"], actual["fl_min"], actual["ti_median_s"]) == (
            0.25,
            0.1,
            1.5,
        )
        assert actual == _old_reference(sessions)

    def test_confidence_at_default_is_excluded_from_fl_class_share(self) -> None:
        at_default = _breath(flow_class=5, confidence=_DEFAULT)
        rule_matched = _breath(flow_class=1, confidence=_DEFAULT + 0.01)

        only_default = _compared(_new_summary(_ok_night([at_default, at_default])))
        mixed = _compared(_new_summary(_ok_night([at_default, rule_matched])))

        assert only_default["fl_class_ge4_pct"] is None
        assert mixed["fl_class_ge4_pct"] == 0.0

    def test_leak_invalid_breath_still_counts_for_rera_proxy(self) -> None:
        # Breaths 1-2 form a flow-limited run (breath 2 leak-invalid) and
        # breath 3 is a flagged recovery breath.
        sessions = _ok_night(
            [
                _breath(flow_class=4, confidence=0.8, flattening=0.1, peak=20.0),
                _breath(leak_valid=False, flow_class=5, peak=20.0),
                _breath(flow_class=1, recovery=True, flattening=0.9, peak=30.0),
            ]
        )

        actual = _compared(_new_summary(sessions))

        assert actual["rera_count"] == 1
        assert actual == _old_reference(sessions)
