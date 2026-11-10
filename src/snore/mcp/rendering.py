"""Presentation-only renderer for MCP waveform responses.

Converts an already-computed ``WaveformWindow`` DTO into a PNG byte string.
No database access, no domain computation — pure visual presentation over
an in-memory DTO.  Uses matplotlib's OO API exclusively (no pyplot) so there
is no global figure state and the module is safe to import in any thread or
async context.
"""

from __future__ import annotations

import io

from typing import TYPE_CHECKING

from snore.mcp.schemas import localize_wall_clock
from snore.services.breath.dtos import CATEGORICAL_WAVEFORM_CHANNELS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from snore.services.breath_service import WaveformWindow


def _build_suptitle(window: WaveformWindow) -> str:
    """Build the figure suptitle string for a waveform window.

    Pure helper — no side effects, no imports.  Extracts the title construction
    so it can be tested directly without rendering a full PNG.
    """
    if window.session_id > 0:
        wall_clock_str = localize_wall_clock(
            window.session_start_wall_clock,
            window.timezone_status,
            window.timezone_name,
        )
        title_parts = [
            f"Session {window.session_id}",
            wall_clock_str,
            f"{window.window_start_offset:.0f}–{window.window_end_offset:.0f} s",
        ]
    else:
        title_parts = [
            "no session",
            f"{window.window_start_offset:.0f}–{window.window_end_offset:.0f} s",
        ]

    if window.missing_channels:
        title_parts.append(
            "missing: " + ", ".join(str(ch) for ch in window.missing_channels)
        )

    return "  |  ".join(title_parts)


# Descriptive y-axis labels for channels where the default "type (unit)" string
# would omit important signal-processing context.  Unmapped channels fall back to
# the default f"{channel_type} ({unit})" format.
_CHANNEL_LABELS: dict[str, str] = {
    # Both pressure channels are 0.5 Hz duty-cycle averages from PLD.edf — the
    # rendered value reads ~time-weighted mean of IPAP/EPAP, not the instantaneous
    # bilevel square wave.  The label makes this explicit so readers don't mistake
    # the trace for the instantaneous delivered pressure.
    "pressure": "mask pressure (cmH2O, 0.5 Hz avg)",
    "therapy_pressure": "therapy pressure (cmH2O, 0.5 Hz avg)",
    "pressure_hr": "mask pressure (cmH2O, 25 Hz)",
    "trigger_cycle": "trigger/cycle code (raw)",
}


def _build_figure(window: WaveformWindow) -> Figure:
    """Lay out a ``WaveformWindow`` as a matplotlib figure.

    Layout: one subplot per channel, stacked vertically.  An empty-channel
    window (``session_id == 0`` or no channels) produces a single panel with a
    centered "No waveform data" label.  Categorical channels are drawn as
    steps holding each code until the next sample, with integer y ticks.
    """
    from matplotlib.figure import Figure  # noqa: PLC0415
    from matplotlib.ticker import MaxNLocator  # noqa: PLC0415

    channels = window.channels
    n = len(channels)

    if n == 0:
        fig: Figure = Figure(figsize=(12.0, 4.0), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            "No waveform data",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
    else:
        fig = Figure(figsize=(12.0, max(4.0, 2.5 * n)), dpi=100)
        for idx, ch in enumerate(channels, start=1):
            ax = fig.add_subplot(n, 1, idx)
            categorical = ch.channel_type in CATEGORICAL_WAVEFORM_CHANNELS
            ax.plot(
                ch.offset_seconds,
                ch.values,
                linewidth=0.8,
                drawstyle="steps-post" if categorical else "default",
            )
            if categorical:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            y_label = _CHANNEL_LABELS.get(
                ch.channel_type,
                f"{ch.channel_type} ({ch.unit})" if ch.unit else ch.channel_type,
            )
            ax.set_ylabel(y_label, fontsize=8)
            ax.tick_params(labelsize=7)

            if ch.is_downsampled:
                method = "run starts" if categorical else "LTTB"
                ax.set_title(
                    f"{method} {len(ch.values)}/{ch.original_sample_count}",
                    loc="right",
                    fontsize=7,
                )

            if idx == n:
                ax.set_xlabel("Offset from session start (s)", fontsize=8)

    fig.suptitle(_build_suptitle(window), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def render_waveform_window(window: WaveformWindow) -> bytes:
    """Render a ``WaveformWindow`` DTO as a PNG and return the raw bytes.

    Args:
        window: A fully-computed ``WaveformWindow`` from ``compute_waveform_window``.

    Returns:
        Raw PNG bytes (always starts with the PNG magic ``\\x89PNG\\r\\n\\x1a\\n``).
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: PLC0415

    buf = io.BytesIO()
    canvas = FigureCanvasAgg(_build_figure(window))
    canvas.print_png(buf)  # type: ignore[no-untyped-call]
    return buf.getvalue()
