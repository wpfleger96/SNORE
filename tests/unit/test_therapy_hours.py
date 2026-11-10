"""Unit tests for the therapy-hours denominator contract in snore.therapy_hours."""

from __future__ import annotations

import pytest

from snore.parsers.unified import (
    DeviceInfo,
    RespiratoryEvent,
    RespiratoryEventType,
    UnifiedSession,
)
from snore.therapy_hours import (
    TherapyHoursBasis,
    compliance_pct,
    is_compliant_night,
    therapy_hours,
)

# ---------------------------------------------------------------------------
# Compliance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("hours", "compliant"), [(4.0, True), (3.999, False)])
def test_compliance_threshold_is_inclusive(hours: float, compliant: bool) -> None:
    assert is_compliant_night(hours) is compliant


def test_compliance_pct_is_share_of_calendar_days() -> None:
    assert compliance_pct(3, 4) == 75.0


# ---------------------------------------------------------------------------
# MASK_ON
# ---------------------------------------------------------------------------


def test_mask_on_none_segments_returns_none():
    assert therapy_hours(TherapyHoursBasis.MASK_ON, mask_on_segments=None) is None


def test_mask_on_empty_segments_returns_known_zero():
    assert therapy_hours(TherapyHoursBasis.MASK_ON, mask_on_segments=[]) == 0.0


def test_mask_on_single_hour_segment_returns_one():
    assert (
        therapy_hours(TherapyHoursBasis.MASK_ON, mask_on_segments=[(0.0, 3600.0)])
        == 1.0
    )


def test_mask_on_multi_segment_sums_durations():
    # (3600) + (3600) = 7200s = 2.0h; the gap between segments is excluded.
    segments = [(0.0, 3600.0), (4200.0, 7800.0)]
    assert therapy_hours(
        TherapyHoursBasis.MASK_ON, mask_on_segments=segments
    ) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SESSION_SPAN
# ---------------------------------------------------------------------------


def test_session_span_none_returns_none():
    assert therapy_hours(TherapyHoursBasis.SESSION_SPAN, span_seconds=None) is None


def test_session_span_zero_returns_zero():
    assert therapy_hours(TherapyHoursBasis.SESSION_SPAN, span_seconds=0.0) == 0.0


def test_session_span_one_hour_returns_one():
    assert therapy_hours(TherapyHoursBasis.SESSION_SPAN, span_seconds=3600.0) == 1.0


# ---------------------------------------------------------------------------
# WAVEFORM_COVERAGE
# ---------------------------------------------------------------------------


def test_waveform_coverage_happy_path():
    # 36000 samples at 10 Hz = 3600s = 1.0h.
    assert (
        therapy_hours(
            TherapyHoursBasis.WAVEFORM_COVERAGE, sample_count=36000, sample_rate=10.0
        )
        == 1.0
    )


def test_waveform_coverage_missing_sample_count_returns_none():
    assert (
        therapy_hours(
            TherapyHoursBasis.WAVEFORM_COVERAGE, sample_count=None, sample_rate=10.0
        )
        is None
    )


def test_waveform_coverage_missing_sample_rate_returns_none():
    assert (
        therapy_hours(
            TherapyHoursBasis.WAVEFORM_COVERAGE, sample_count=36000, sample_rate=None
        )
        is None
    )


def test_waveform_coverage_zero_sample_rate_returns_none():
    assert (
        therapy_hours(
            TherapyHoursBasis.WAVEFORM_COVERAGE, sample_count=36000, sample_rate=0.0
        )
        is None
    )


def test_waveform_coverage_negative_sample_rate_returns_none():
    assert (
        therapy_hours(
            TherapyHoursBasis.WAVEFORM_COVERAGE, sample_count=36000, sample_rate=-10.0
        )
        is None
    )


# ---------------------------------------------------------------------------
# No hidden fallback between bases: foreign kwargs are rejected, not ignored
# ---------------------------------------------------------------------------


def test_mask_on_rejects_span_input():
    # A span supplied alongside MASK_ON is a mis-routed kwarg, not a fallback.
    with pytest.raises(ValueError, match="span_seconds"):
        therapy_hours(
            TherapyHoursBasis.MASK_ON, mask_on_segments=[], span_seconds=7200.0
        )


def test_session_span_rejects_mask_on_input():
    with pytest.raises(ValueError, match="mask_on_segments"):
        therapy_hours(
            TherapyHoursBasis.SESSION_SPAN,
            span_seconds=3600.0,
            mask_on_segments=[(0.0, 3600.0)],
        )


# Every basis paired with each kwarg that belongs to a different basis; a
# None-valued foreign kwarg is still accepted (it is the parameter default).
_FOREIGN_KWARGS = {
    TherapyHoursBasis.MASK_ON: [
        {"span_seconds": 3600.0},
        {"sample_count": 36000},
        {"sample_rate": 10.0},
    ],
    TherapyHoursBasis.SESSION_SPAN: [
        {"mask_on_segments": [(0.0, 3600.0)]},
        {"sample_count": 36000},
        {"sample_rate": 10.0},
    ],
    TherapyHoursBasis.WAVEFORM_COVERAGE: [
        {"mask_on_segments": [(0.0, 3600.0)]},
        {"span_seconds": 3600.0},
    ],
}


@pytest.mark.parametrize(
    ("basis", "foreign_kwarg"),
    [(basis, kw) for basis, kws in _FOREIGN_KWARGS.items() for kw in kws],
)
def test_basis_rejects_foreign_kwarg(basis, foreign_kwarg):
    (foreign_name,) = foreign_kwarg
    with pytest.raises(ValueError, match=foreign_name):
        therapy_hours(basis, **foreign_kwarg)


def test_error_names_all_offending_kwargs():
    with pytest.raises(ValueError, match="sample_count.*sample_rate"):
        therapy_hours(
            TherapyHoursBasis.MASK_ON,
            mask_on_segments=[],
            sample_count=36000,
            sample_rate=10.0,
        )


def test_none_valued_foreign_kwarg_is_accepted():
    # SESSION_SPAN with mask_on_segments=None is fine — None is the default and
    # signals "not supplied", so it is not treated as a mis-routed argument.
    assert (
        therapy_hours(
            TherapyHoursBasis.SESSION_SPAN,
            span_seconds=3600.0,
            mask_on_segments=None,
        )
        == 1.0
    )


# ---------------------------------------------------------------------------
# Cross-parser denominator parity through finalize_statistics
# ---------------------------------------------------------------------------

_DEVICE = DeviceInfo(manufacturer="ResMed", model="AirSense 10", serial_number="SN123")


def _session_with_events(
    duration_s: float,
    mask_on_segments: list[tuple[float, float]] | None,
) -> UnifiedSession:
    from datetime import UTC, datetime, timedelta

    base = datetime(2025, 9, 10, 22, 0, 0, tzinfo=UTC)
    session = UnifiedSession(
        device_info=_DEVICE,
        device_session_id="parity",
        start_time=base,
        end_time=base + timedelta(seconds=duration_s),
        mask_on_segments=mask_on_segments,
    )
    for _ in range(6):
        session.add_event(
            RespiratoryEvent(
                event_type=RespiratoryEventType.OBSTRUCTIVE_APNEA,
                start_time=base,
                duration_seconds=10.0,
            )
        )
    return session


def test_mask_on_full_span_matches_span_only_ahi():
    """One hour of therapy via mask-on segments vs span-only yields identical AHI."""
    mask_session = _session_with_events(
        duration_s=3600.0, mask_on_segments=[(0.0, 3600.0)]
    )
    span_session = _session_with_events(duration_s=3600.0, mask_on_segments=None)
    mask_session.finalize_statistics()
    span_session.finalize_statistics()

    assert mask_session.statistics.ahi == pytest.approx(span_session.statistics.ahi)
    assert mask_session.statistics.usage_hours == pytest.approx(
        span_session.statistics.usage_hours
    )
