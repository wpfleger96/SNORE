"""Smoke tests that draw waveform charts through real (unmocked) plotext."""

import numpy as np

import snore.cli  # noqa: F401  # load before snore.waveform to avoid an import cycle

from snore.waveform.renderer import WaveformRenderer


def _sine_window() -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.linspace(3600.0, 3630.0, 300)
    return timestamps, np.sin(timestamps)


def test_render_draws_single_waveform_chart(capsys):
    timestamps, values = _sine_window()

    WaveformRenderer(width=60, height=12, show_events=False).render(
        timestamps, values, session_id=7
    )

    assert "Session 7 - Flow Waveform" in capsys.readouterr().out


def test_render_multi_draws_stacked_charts(capsys):
    timestamps, values = _sine_window()

    WaveformRenderer(width=60, height=24).render_multi(
        [(timestamps, values, "flow"), (timestamps, values + 10, "pressure")],
        session_id=7,
    )

    out = capsys.readouterr().out
    assert "Session 7 - Multi-waveform" in out
    assert "Pressure (cmH2O)" in out
