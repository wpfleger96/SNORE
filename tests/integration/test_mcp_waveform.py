"""Integration tests for the waveform MCP tool adapters.

Covers:
- Happy path: seeded flow waveform sliced to the requested window.
- Empty day: no sessions → sentinel response (session_id null, channels empty).
- Missing channel: only flow seeded, pressure absent → channel_absent reason.
- Two-profile isolation: profile A cannot access profile B's session or waveform;
  auto-resolve as A returns only A's data.
- PNG render: seeded data → valid PNG magic bytes.

Calls adapter functions directly (no MCP server involved).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pytest

from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.models import Session, Waveform
from tests.integration.conftest import (
    _make_day_session,
    _make_device,
    _make_profile,
)


def _make_blob(offsets: list[float], values: list[float]) -> tuple[bytes, int]:
    """Serialize (offsets, values) as a float32 (n, 2) array — matches deserialize_waveform_blob."""
    arr = np.column_stack([offsets, values]).astype(np.float32)
    return arr.tobytes(), len(offsets)


async def _make_waveform(
    db: AsyncSession,
    session: Session,
    waveform_type: str,
    sample_rate: float,
    unit: str,
    n_samples: int,
    time_start: float = 0.0,
    constant_value: float = 1.0,
) -> Waveform:
    """Seed a Waveform row with a uniform constant signal."""
    dt = 1.0 / sample_rate
    offsets = [time_start + i * dt for i in range(n_samples)]
    values = [constant_value] * n_samples
    blob_bytes, count = _make_blob(offsets, values)
    wf = Waveform(
        session_id=session.id,
        waveform_type=waveform_type,
        sample_rate=sample_rate,
        unit=unit,
        data_blob=blob_bytes,
        sample_count=count,
    )
    db.add(wf)
    await db.flush()
    return wf


# ---------------------------------------------------------------------------
# TestHappyPath — flow waveform sliced to the requested window
# ---------------------------------------------------------------------------


class TestHappyPath:
    async def test_slice_returns_only_samples_in_window(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Seed 0–60 s flow waveform; fetch 10–20 s → all offset_seconds within [10, 20]."""
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            waveform_response_from_raw,
        )

        target_date = date(2024, 3, 1)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)

        # 25 Hz, 60 s → 1500 samples covering 0–59.96 s
        await _make_waveform(
            async_db_session, sess, "flow", 25.0, "L/min", 1500, time_start=0.0
        )

        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            offset_start=10.0,
            offset_end=20.0,
            window_cap_seconds=120.0,
        )
        response = waveform_response_from_raw(raw)

        assert response.session_id == sess.id
        assert response.session_start_wall_clock is not None
        assert response.timezone_status == "unknown"

        # flow channel must be present
        flow_channels = [ch for ch in response.channels if ch.channel_type == "flow"]
        assert len(flow_channels) == 1, "flow channel missing from response"
        flow = flow_channels[0]

        assert len(flow.offset_seconds) > 0
        # Every reported offset must be within the requested window
        assert all(10.0 - 1e-3 <= t <= 20.0 + 1e-3 for t in flow.offset_seconds)


# ---------------------------------------------------------------------------
# TestEmptyDay — no sessions on the queried date
# ---------------------------------------------------------------------------


class TestEmptyDay:
    async def test_no_sessions_returns_sentinel_response(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Profile has a device but no sessions on the date → sentinel (session_id null)."""
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            waveform_response_from_raw,
        )

        target_date = date(2024, 3, 2)
        device = await _make_device(async_db_session, async_test_profile.id)
        # No sessions seeded on target_date

        # device_id required: auto-resolve would raise ValueError (no sessions in range).
        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            offset_start=0.0,
            offset_end=30.0,
            device_id=device.id,
            channels=["flow", "pressure"],
            window_cap_seconds=120.0,
        )
        response = waveform_response_from_raw(raw)

        assert response.session_id is None
        assert response.session_start_wall_clock is None
        assert response.channels == []
        assert "flow" in response.missing_channels
        assert "pressure" in response.missing_channels
        assert response.missing_channel_reason == "channel_absent"


# ---------------------------------------------------------------------------
# TestMissingChannel — requested channel absent in the database
# ---------------------------------------------------------------------------


class TestMissingChannel:
    async def test_absent_channel_appears_in_missing_with_reason(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Seed flow only; request flow+pressure → pressure in missing_channels, reason channel_absent."""
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            waveform_response_from_raw,
        )

        target_date = date(2024, 3, 3)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)

        # Seed flow only — no pressure waveform
        await _make_waveform(async_db_session, sess, "flow", 25.0, "L/min", 750)

        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            offset_start=0.0,
            offset_end=30.0,
            channels=["flow", "pressure"],
            window_cap_seconds=120.0,
        )
        response = waveform_response_from_raw(raw)

        assert len(response.channels) == 1
        assert response.channels[0].channel_type == "flow"
        assert "pressure" in response.missing_channels
        assert response.missing_channel_reason == "channel_absent"


# ---------------------------------------------------------------------------
# TestIsolation — adversarial two-profile isolation
# ---------------------------------------------------------------------------


class TestIsolation:
    async def test_explicit_b_session_id_raises_value_error(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Profile A passing profile B's session_id → ValueError (not found or not owned)."""
        from snore.mcp.tools.waveform import fetch_waveform_raw  # noqa: PLC0415

        target_date = date(2024, 3, 4)

        # Profile A: device + session + waveform
        device_a = await _make_device(async_db_session, async_test_profile.id)
        _, sess_a = await _make_day_session(async_db_session, device_a, target_date)
        await _make_waveform(async_db_session, sess_a, "flow", 25.0, "L/min", 750)

        # Profile B: separate device + session
        profile_b = await _make_profile(async_db_session)
        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        await _make_waveform(
            async_db_session, sess_b, "flow", 25.0, "L/min", 750, constant_value=999.0
        )

        with pytest.raises(ValueError):
            await fetch_waveform_raw(
                async_db_session,
                target_date,
                profile_id=async_test_profile.id,  # profile A
                offset_start=0.0,
                offset_end=30.0,
                session_id=sess_b.id,  # B's session
                window_cap_seconds=120.0,
            )

    async def test_auto_resolve_as_a_excludes_b_sentinel(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Auto-resolve as profile A returns only A's data; B's 999.0 sentinel never appears."""
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            waveform_response_from_raw,
        )

        target_date = date(2024, 3, 5)

        # Profile A: constant value 2.0
        device_a = await _make_device(async_db_session, async_test_profile.id)
        _, sess_a = await _make_day_session(async_db_session, device_a, target_date)
        await _make_waveform(
            async_db_session, sess_a, "flow", 25.0, "L/min", 750, constant_value=2.0
        )

        # Profile B: unique sentinel 999.0
        profile_b = await _make_profile(async_db_session)
        device_b = await _make_device(async_db_session, profile_b.id)
        _, sess_b = await _make_day_session(async_db_session, device_b, target_date)
        await _make_waveform(
            async_db_session, sess_b, "flow", 25.0, "L/min", 750, constant_value=999.0
        )

        # Auto-resolve as profile A — must not find session B
        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            offset_start=0.0,
            offset_end=30.0,
            channels=["flow"],
            window_cap_seconds=120.0,
        )
        response = waveform_response_from_raw(raw)

        assert response.session_id == sess_a.id
        for ch in response.channels:
            # B's 999.0 sentinel must not appear in A's values
            assert all(v < 500.0 for v in ch.values), (
                f"B's sentinel appeared in A's channel {ch.channel_type}: {ch.values[:5]}"
            )


# ---------------------------------------------------------------------------
# TestRenderPng — PNG magic bytes with real seeded data
# ---------------------------------------------------------------------------


class TestRenderPng:
    async def test_seeded_waveform_renders_valid_png(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """render_png_from_raw on real seeded data returns bytes with PNG magic."""
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            render_png_from_raw,
        )

        target_date = date(2024, 3, 6)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_waveform(async_db_session, sess, "flow", 25.0, "L/min", 750)

        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            offset_start=0.0,
            offset_end=30.0,
            channels=["flow"],
            window_cap_seconds=120.0,
        )
        png_bytes = render_png_from_raw(raw)

        assert isinstance(png_bytes, bytes)
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# TestResmedOnlyChannels — pressure_hr, ti, ie_ratio, trigger_cycle
# ---------------------------------------------------------------------------

_RESMED_ONLY_CHANNELS = ["pressure_hr", "ti", "ie_ratio", "trigger_cycle"]


class TestResmedOnlyChannels:
    @pytest.mark.parametrize(
        ("channel", "sample_rate", "unit", "value"),
        [
            ("pressure_hr", 25.0, "cmH2O", 9.5),
            ("ti", 0.5, "seconds", 1.25),
            ("ie_ratio", 0.5, "%", 50.0),
            ("trigger_cycle", 25.0, "", 3.0),
        ],
    )
    async def test_stored_channel_fetch_returns_its_samples(
        self,
        async_db_session: AsyncSession,
        async_test_profile: Any,
        channel: str,
        sample_rate: float,
        unit: str,
        value: float,
    ) -> None:
        """Each channel is returned with its stored unit, rate and samples."""
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            waveform_response_from_raw,
        )

        target_date = date(2024, 3, 7)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_waveform(
            async_db_session,
            sess,
            channel,
            sample_rate,
            unit,
            int(30 * sample_rate),
            constant_value=value,
        )

        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            offset_start=0.0,
            offset_end=30.0,
            channels=[channel],
            window_cap_seconds=120.0,
        )
        response = waveform_response_from_raw(raw)

        assert [ch.channel_type for ch in response.channels] == [channel]
        returned = response.channels[0]
        assert returned.unit == unit
        assert returned.sample_rate_hz == sample_rate
        assert len(returned.values) == int(30 * sample_rate)
        assert all(v == pytest.approx(value) for v in returned.values)
        assert response.missing_channels == []
        assert response.missing_channel_reason is None

    async def test_session_without_them_lists_them_as_missing(
        self, async_db_session: AsyncSession, async_test_profile: Any
    ) -> None:
        """Flow-only session → the four channels are missing, flow is returned."""
        from snore.mcp.tools.waveform import (  # noqa: PLC0415
            fetch_waveform_raw,
            waveform_response_from_raw,
        )

        target_date = date(2024, 3, 8)
        device = await _make_device(async_db_session, async_test_profile.id)
        _, sess = await _make_day_session(async_db_session, device, target_date)
        await _make_waveform(async_db_session, sess, "flow", 25.0, "L/min", 750)

        raw = await fetch_waveform_raw(
            async_db_session,
            target_date,
            profile_id=async_test_profile.id,
            offset_start=0.0,
            offset_end=30.0,
            channels=["flow", *_RESMED_ONLY_CHANNELS],
            window_cap_seconds=120.0,
        )
        response = waveform_response_from_raw(raw)

        assert [ch.channel_type for ch in response.channels] == ["flow"]
        assert response.missing_channels == _RESMED_ONLY_CHANNELS
        assert response.missing_channel_reason == "channel_absent"
