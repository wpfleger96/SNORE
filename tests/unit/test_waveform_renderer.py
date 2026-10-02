"""Smoke tests that draw waveform charts through real (unmocked) plotext."""

import io

import numpy as np
import pytest

from rich.console import Console

from snore.waveform.renderer import WaveformRenderer
from tests.helpers.terminal_charts import has_drawn_braille


@pytest.fixture(autouse=True)
def _pin_terminal_size(monkeypatch):
    # plotext clamps plot_size to the terminal size; a small terminal shrinks the
    # charts and drops subplot labels, so pin one larger than any chart here.
    monkeypatch.setenv("LINES", "50")
    monkeypatch.setenv("COLUMNS", "120")


def _sine_window() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.linspace(3600.0, 3630.0, 300)
    return timestamps, np.sin(timestamps)


def _renderer(
    console_out: io.StringIO, *, width: int, height: int, show_events: bool = True
) -> WaveformRenderer:
    return WaveformRenderer(
        console=Console(file=console_out),
        width=width,
        height=height,
        show_events=show_events,
    )


def test_render_draws_single_waveform_chart():
    timestamps, values = _sine_window()
    console_out = io.StringIO()

    _renderer(console_out, width=60, height=12, show_events=False).render(
        timestamps, values, session_id=7
    )

    out = console_out.getvalue()
    assert "Session 7 - Flow Waveform" in out
    assert has_drawn_braille(out)
    assert "01:00:00" in out
    assert "Sample rate" in out


def test_render_with_infinite_values_draws_chart():
    timestamps, values = _sine_window()
    values[10] = np.inf
    values[20] = -np.inf
    console_out = io.StringIO()

    _renderer(console_out, width=60, height=12, show_events=False).render(
        timestamps, values
    )

    assert has_drawn_braille(console_out.getvalue())


def test_render_multi_draws_stacked_charts():
    timestamps, values = _sine_window()
    console_out = io.StringIO()

    _renderer(console_out, width=60, height=24).render_multi(
        [(timestamps, values, "flow"), (timestamps, values + 10, "pressure")],
        session_id=7,
    )

    out = console_out.getvalue()
    assert "Session 7 - Multi-waveform" in out
    assert "Pressure (cmH2O)" in out
    assert "01:00:00" in out
    assert "Sample rates:" in out
    panels = out.split("┌")[1:]
    assert len(panels) == 2
    assert all(has_drawn_braille(panel) for panel in panels)


def test_render_multi_with_one_waveform_draws_chart():
    timestamps, values = _sine_window()
    console_out = io.StringIO()

    _renderer(console_out, width=60, height=12).render_multi(
        [(timestamps, values, "flow"), (np.array([]), np.array([]), "spo2")]
    )

    out = console_out.getvalue()
    assert out.count("┌") == 1
    assert has_drawn_braille(out)
    assert "Flow (L/min)" in out
