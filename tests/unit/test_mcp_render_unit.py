"""Unit tests for the MCP waveform renderer.

Exercises ``render_waveform_window`` directly with real ``WaveformWindow``
DTOs — no server wiring, no database.  Verifies PNG output correctness,
no-pyplot guarantee, and DTO immutability.
"""

from __future__ import annotations

import math
import sys

from datetime import datetime

# ---------------------------------------------------------------------------
# Module-level helpers — build real service DTOs
# ---------------------------------------------------------------------------

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_channel(
    *,
    channel_type: str = "flow",
    unit: str | None = "L/min",
    n_points: int = 100,
    is_downsampled: bool = False,
    original_sample_count: int | None = None,
) -> object:
    """Build a single ``WaveformChannel`` DTO with synthetic values."""
    from snore.services.breath_service import (  # noqa: PLC0415
        WaveformChannel,
        WaveformChannelName,
    )

    offsets = [i * 0.1 for i in range(n_points)]
    values = [math.sin(o) for o in offsets]
    osc = original_sample_count if original_sample_count is not None else n_points

    return WaveformChannel(
        channel_type=WaveformChannelName(channel_type),
        unit=unit,
        sample_rate=10.0,
        offset_seconds=offsets,
        values=values,
        original_sample_count=osc,
        is_downsampled=is_downsampled,
    )


def _make_window(
    *,
    session_id: int = 42,
    channels: list | None = None,
    missing_channels: list | None = None,
) -> object:
    """Build a ``WaveformWindow`` DTO."""
    from snore.services.breath_service import (  # noqa: PLC0415
        NullReason,
        TimezoneStatus,
        WaveformWindow,
    )

    return WaveformWindow(
        session_id=session_id,
        session_start_wall_clock=datetime(2024, 1, 15, 22, 0, 0),
        timezone_status=TimezoneStatus.UNKNOWN,
        window_start_offset=0.0,
        window_end_offset=10.0,
        channels=channels or [],
        missing_channels=missing_channels or [],
        missing_channel_reason=None
        if not missing_channels
        else NullReason.NOT_AVAILABLE,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empty_channels_window_produces_valid_png() -> None:
    """Empty-channel window (session_id=0) renders a valid PNG placeholder."""
    from snore.mcp.rendering import render_waveform_window  # noqa: PLC0415

    window = _make_window(session_id=0, channels=[])
    result = render_waveform_window(window)
    assert isinstance(result, bytes)
    assert result[:8] == PNG_MAGIC
    assert len(result) > 0


def test_single_channel_produces_valid_png() -> None:
    """Single-channel window with 100 synthetic data points renders a valid PNG."""
    from snore.mcp.rendering import render_waveform_window  # noqa: PLC0415

    channel = _make_channel(n_points=100)
    window = _make_window(channels=[channel])
    result = render_waveform_window(window)
    assert result[:8] == PNG_MAGIC
    assert len(result) > 0


def test_three_channels_produces_valid_png() -> None:
    """Three-channel window renders a stacked multi-panel PNG."""
    from snore.mcp.rendering import render_waveform_window  # noqa: PLC0415

    channels = [
        _make_channel(channel_type="flow", unit="L/min"),
        _make_channel(channel_type="pressure", unit="cmH2O"),
        _make_channel(channel_type="leak", unit="L/min"),
    ]
    window = _make_window(channels=channels)
    result = render_waveform_window(window)
    assert result[:8] == PNG_MAGIC
    assert len(result) > 0


def test_downsampled_channel_produces_valid_png() -> None:
    """Downsampled channel (LTTB annotation) renders a valid PNG."""
    from snore.mcp.rendering import render_waveform_window  # noqa: PLC0415

    channel = _make_channel(
        n_points=50,
        is_downsampled=True,
        original_sample_count=500,
    )
    window = _make_window(channels=[channel])
    result = render_waveform_window(window)
    assert result[:8] == PNG_MAGIC
    assert len(result) > 0


def test_render_does_not_import_pyplot() -> None:
    """render_waveform_window must not import matplotlib.pyplot (no global state)."""
    from snore.mcp.rendering import render_waveform_window  # noqa: PLC0415

    # Remove pyplot from cache so we have a clean baseline
    sys.modules.pop("matplotlib.pyplot", None)

    channel = _make_channel(n_points=20)
    window = _make_window(channels=[channel])
    render_waveform_window(window)
    assert "matplotlib.pyplot" not in sys.modules


def test_render_does_not_mutate_input_dto() -> None:
    """render_waveform_window must not modify the WaveformWindow DTO."""
    from snore.mcp.rendering import render_waveform_window  # noqa: PLC0415

    channel = _make_channel(n_points=30)
    window = _make_window(channels=[channel])

    snapshot_before = window.model_dump()
    render_waveform_window(window)
    snapshot_after = window.model_dump()
    assert snapshot_before == snapshot_after


def test_trigger_cycle_panel_draws_steps_with_integer_ticks() -> None:
    """Codes are drawn as held steps on an integer axis; magnitudes stay lines."""
    from snore.mcp.rendering import _build_figure  # noqa: PLC0415

    window = _make_window(
        channels=[
            _make_channel(channel_type="flow"),
            _make_channel(channel_type="trigger_cycle", unit=""),
        ]
    )

    flow_ax, codes_ax = _build_figure(window).axes

    assert flow_ax.get_lines()[0].get_drawstyle() == "default"
    assert codes_ax.get_lines()[0].get_drawstyle() == "steps-post"
    assert all(float(tick).is_integer() for tick in codes_ax.get_yticks())
    assert codes_ax.get_ylabel() == "trigger/cycle code (raw)"


def test_downsampled_trigger_cycle_title_names_run_starts() -> None:
    """A reduced trigger_cycle panel says it kept run starts, not LTTB points."""
    from snore.mcp.rendering import _build_figure  # noqa: PLC0415

    channel = _make_channel(
        channel_type="trigger_cycle",
        unit="",
        n_points=50,
        is_downsampled=True,
        original_sample_count=500,
    )

    (ax,) = _build_figure(_make_window(channels=[channel])).axes

    assert ax.get_title(loc="right") == "run starts 50/500"


# ---------------------------------------------------------------------------
# Tests for _build_suptitle — pure helper, no PNG rendering needed
# ---------------------------------------------------------------------------


class TestBuildSuptitle:
    def test_no_session_branch_contains_no_session_and_not_datetime_min(
        self,
    ) -> None:
        """session_id=0 → title contains 'no session' and must not expose datetime.min."""
        from snore.mcp.rendering import _build_suptitle  # noqa: PLC0415

        window = _make_window(session_id=0)
        title = _build_suptitle(window)

        assert "no session" in title
        # datetime.min is 0001-01-01; it must never leak into the title
        assert "0001-01-01" not in title

    def test_happy_path_contains_wall_clock_and_missing_channels(self) -> None:
        """session_id > 0 with missing channels → title has wall-clock ISO string
        and 'missing:' list."""
        from snore.mcp.rendering import _build_suptitle  # noqa: PLC0415
        from snore.services.breath_service import WaveformChannelName  # noqa: PLC0415

        window = _make_window(
            session_id=42,
            missing_channels=[WaveformChannelName.SPO2, WaveformChannelName.PULSE],
        )
        title = _build_suptitle(window)

        # Wall-clock should appear (the DTO has datetime(2024, 1, 15, 22, 0, 0))
        assert "2024-01-15T22:00:00" in title
        # Missing channels listed
        assert "missing:" in title
        assert "spo2" in title
        assert "pulse" in title
