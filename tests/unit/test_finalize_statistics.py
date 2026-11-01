"""Tests for UnifiedSession.finalize_statistics usage-based AHI computation.

Covers:
1. usage_hours is computed from mask_on_segments (mask-on time) when present,
   not from the session span (start/end difference).
2. All per-hour indices (AHI, OAI, CAI, HI) are computed over usage hours.
3. Span fallback: when mask_on_segments is None, duration_seconds is used.
4. Zero-division guard: therapy_seconds == 0 leaves statistics as None.
5. Waveform statistics replace the STR summary values; without a waveform the
   device value stays.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from snore.parsers.unified import (
    DeviceInfo,
    RespiratoryEvent,
    RespiratoryEventType,
    UnifiedSession,
    WaveformData,
    WaveformType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEVICE = DeviceInfo(
    manufacturer="ResMed",
    model="AirSense 10",
    serial_number="SN123",
)

_BASE = datetime(2025, 9, 10, 22, 0, 0, tzinfo=UTC)


def _session(
    duration_s: float,
    mask_on_segments: list[tuple[float, float]] | None = None,
    oa: int = 0,
    ca: int = 0,
    h: int = 0,
) -> UnifiedSession:
    """Build a minimal UnifiedSession with synthetic respiratory events."""
    end = _BASE + timedelta(seconds=duration_s)
    session = UnifiedSession(
        device_info=_DEVICE,
        device_session_id="test-sess",
        start_time=_BASE,
        end_time=end,
        mask_on_segments=mask_on_segments,
    )
    for _ in range(oa):
        session.add_event(
            RespiratoryEvent(
                event_type=RespiratoryEventType.OBSTRUCTIVE_APNEA,
                start_time=_BASE,
                duration_seconds=10.0,
            )
        )
    for _ in range(ca):
        session.add_event(
            RespiratoryEvent(
                event_type=RespiratoryEventType.CENTRAL_APNEA,
                start_time=_BASE,
                duration_seconds=10.0,
            )
        )
    for _ in range(h):
        session.add_event(
            RespiratoryEvent(
                event_type=RespiratoryEventType.HYPOPNEA,
                start_time=_BASE,
                duration_seconds=10.0,
            )
        )
    return session


# ---------------------------------------------------------------------------
# usage_hours from mask_on_segments
# ---------------------------------------------------------------------------


class TestWaveformStatisticsOverStrSummary:
    @pytest.mark.parametrize(
        ("waveform_type", "prefix"),
        [
            (WaveformType.THERAPY_PRESSURE, "pressure"),
            (WaveformType.EPAP, "epap"),
            (WaveformType.LEAK_RATE, "leak"),
        ],
    )
    def test_waveform_replaces_str_value(
        self, waveform_type: WaveformType, prefix: str
    ) -> None:
        session = _session(duration_s=3600)
        for stat in ("max", "median", "95th"):
            setattr(session.statistics, f"{prefix}_{stat}", 99.0)
        session.add_waveform(
            WaveformData(
                waveform_type=waveform_type,
                sample_rate=1.0,
                unit="",
                timestamps=[0.0, 1.0, 2.0],
                values=[1.0, 2.0, 3.0],
                max_value=3.0,
            )
        )

        session.finalize_statistics()

        assert getattr(session.statistics, f"{prefix}_max") == 3.0
        assert getattr(session.statistics, f"{prefix}_median") == 2.0
        assert getattr(session.statistics, f"{prefix}_95th") < 3.0

    def test_str_value_stays_without_a_waveform(self) -> None:
        session = _session(duration_s=3600)
        session.statistics.leak_median = 12.0

        session.finalize_statistics()

        assert session.statistics.leak_median == 12.0


class TestUsageHoursFromMaskOnSegments:
    def test_two_segments_with_gap_yields_mask_on_time_only(self):
        """Segments [(0,3600), (4200,7800)] on a 7800s span → 2.0 usage hours, not 2.167."""
        # span = 7800s = 2.167h; mask-on = 3600 + 3600 = 7200s = 2.0h
        session = _session(
            duration_s=7800,
            mask_on_segments=[(0, 3600), (4200, 7800)],
        )
        session.finalize_statistics()

        assert session.statistics.usage_hours == pytest.approx(2.0, abs=1e-6)

    def test_single_full_segment_equals_span(self):
        """A single segment covering the full span should equal the span."""
        session = _session(
            duration_s=3600,
            mask_on_segments=[(0, 3600)],
        )
        session.finalize_statistics()

        assert session.statistics.usage_hours == pytest.approx(1.0, abs=1e-6)

    def test_three_segments_summed_correctly(self):
        """Three segments: (0,1800) + (2400,4200) + (4800,6600) → 5400s = 1.5h."""
        session = _session(
            duration_s=6600,
            mask_on_segments=[(0, 1800), (2400, 4200), (4800, 6600)],
        )
        session.finalize_statistics()

        assert session.statistics.usage_hours == pytest.approx(1.5, abs=1e-6)


# ---------------------------------------------------------------------------
# AHI over usage hours, not span hours
# ---------------------------------------------------------------------------


class TestAhiUsesUsageHours:
    def test_ahi_computed_over_mask_on_time_not_span(self):
        """With a gap, AHI based on mask-on time is higher than span-based AHI.

        Setup: span = 7800s (~2.167h), mask-on = 7200s (2.0h), total events = 10.
        Span-AHI  = 10 / 2.167 ≈ 4.615
        Usage-AHI = 10 / 2.0   = 5.0
        """
        session = _session(
            duration_s=7800,
            mask_on_segments=[(0, 3600), (4200, 7800)],
            oa=5,
            ca=3,
            h=2,
        )
        session.finalize_statistics()

        expected_usage_ahi = 10 / 2.0
        span_ahi = 10 / (7800 / 3600)
        assert session.statistics.ahi == pytest.approx(expected_usage_ahi, abs=1e-6)
        assert session.statistics.ahi != pytest.approx(span_ahi, abs=0.01)

    def test_oai_cai_hi_all_use_usage_hours(self):
        """Each per-type index divides by usage hours, not span hours."""
        session = _session(
            duration_s=7800,
            mask_on_segments=[(0, 3600), (4200, 7800)],  # 2.0 usage hours
            oa=4,
            ca=2,
            h=6,
        )
        session.finalize_statistics()

        usage_hours = 2.0
        assert session.statistics.oai == pytest.approx(4 / usage_hours, abs=1e-6)
        assert session.statistics.cai == pytest.approx(2 / usage_hours, abs=1e-6)
        assert session.statistics.hi == pytest.approx(6 / usage_hours, abs=1e-6)


# ---------------------------------------------------------------------------
# Fallback: mask_on_segments is None → use span
# ---------------------------------------------------------------------------


class TestSpanFallbackWhenMaskOnSegmentsNone:
    def test_usage_hours_equals_span_when_segments_none(self):
        """No segment info (e.g., OSCAR import) → usage_hours = session span."""
        session = _session(
            duration_s=3600,
            mask_on_segments=None,
            oa=3,
        )
        session.finalize_statistics()

        assert session.statistics.usage_hours == pytest.approx(1.0, abs=1e-6)

    def test_ahi_uses_span_when_segments_none(self):
        """AHI = total_events / span_hours when mask_on_segments is None."""
        # 6 events over 2h span → AHI = 3.0
        session = _session(
            duration_s=7200,
            mask_on_segments=None,
            oa=3,
            ca=2,
            h=1,
        )
        session.finalize_statistics()

        assert session.statistics.ahi == pytest.approx(3.0, abs=1e-6)

    def test_duration_seconds_property_unchanged(self):
        """duration_seconds always reflects session span regardless of segments."""
        session = _session(
            duration_s=7800,
            mask_on_segments=[(0, 3600), (4200, 7800)],
        )
        assert session.duration_seconds == pytest.approx(7800.0, abs=1e-6)

    def test_duration_hours_property_unchanged(self):
        """duration_hours always reflects session span regardless of segments."""
        session = _session(
            duration_s=7800,
            mask_on_segments=[(0, 3600), (4200, 7800)],
        )
        assert session.duration_hours == pytest.approx(7800 / 3600, abs=1e-6)


# ---------------------------------------------------------------------------
# Zero-division guard
# ---------------------------------------------------------------------------


class TestZeroDivisionGuard:
    def test_none_segments_falls_back_to_span(self):
        """mask_on_segments=None means unknown — fall back to duration_seconds."""
        session = _session(
            duration_s=3600,
            mask_on_segments=None,
        )
        session.finalize_statistics()

        assert session.statistics.usage_hours == pytest.approx(1.0, abs=1e-6)


class TestEmptyMaskOnSegments:
    def test_empty_segments_yields_zero_usage_hours_no_rates(self):
        """mask_on_segments=[] means known-zero mask-on time, not span fallback.

        Events are recorded but no denominator exists, so rates must be None.
        """
        session = _session(
            duration_s=3600,
            mask_on_segments=[],
            oa=5,
            ca=2,
            h=3,
        )
        session.finalize_statistics()

        assert session.statistics.usage_hours == 0.0
        assert session.statistics.ahi is None
        assert session.statistics.oai is None
        assert session.statistics.cai is None
        assert session.statistics.hi is None

    def test_empty_segments_does_not_use_span(self):
        """Span must not bleed into usage_hours when segments is an explicit empty list."""
        session = _session(
            duration_s=7200,  # 2h span that must not appear in usage_hours
            mask_on_segments=[],
        )
        session.finalize_statistics()

        assert session.statistics.usage_hours == 0.0
