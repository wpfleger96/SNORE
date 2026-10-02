"""Smoke tests that draw waveform charts through real (unmocked) plotext."""

import io

import numpy as np
import pytest

from rich.console import Console

from snore.waveform.renderer import WaveformRenderer


@pytest.fixture(autouse=True)
def _pin_terminal_size(monkeypatch):
    # plotext sizes figures from the terminal; a small one drops the title.
    monkeypatch.setenv("LINES", "50")
    monkeypatch.setenv("COLUMNS", "120")


def _sine_window() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.linspace(3600.0, 3630.0, 300)
    return timestamps, np.sin(timestamps)


def test_render_draws_single_waveform_chart(capsys):
    timestamps, values = _sine_window()
    console_out = io.StringIO()

    WaveformRenderer(
        console=Console(file=console_out), width=60, height=12, show_events=False
    ).render(timestamps, values, session_id=7)

    assert "Session 7 - Flow Waveform" in capsys.readouterr().out
    assert "Sample rate" in console_out.getvalue()


def test_render_multi_draws_stacked_charts(capsys):
    timestamps, values = _sine_window()
    console_out = io.StringIO()

    WaveformRenderer(
        console=Console(file=console_out), width=60, height=24
    ).render_multi(
        [(timestamps, values, "flow"), (timestamps, values + 10, "pressure")],
        session_id=7,
    )

    out = capsys.readouterr().out
    assert "Session 7 - Multi-waveform" in out
    assert "Pressure (cmH2O)" in out
    assert "Sample rates:" in console_out.getvalue()
