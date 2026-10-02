"""ASCII and high-resolution waveform rendering for terminal display."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import plotext as plt

from rich.console import Console
from rich.markup import escape

if TYPE_CHECKING:
    from snore.analysis.shared.types import ApneaEvent, HypopneaEvent
    from snore.analysis.types import AnalysisEvent

    EventType = AnalysisEvent | ApneaEvent | HypopneaEvent

WAVEFORM_UNITS = {
    "flow": "L/min",
    "pressure": "cmH2O",
    "therapy_pressure": "cmH2O",
    "epap": "cmH2O",
    "leak": "L/min",
    "mv": "L/min",
    "rr": "breaths/min",
    "tv": "mL",
    "spo2": "%",
    "pulse": "BPM",
    "fl": "a.u.",
    "snore": "a.u.",
}

WAVEFORM_LABELS = {
    "flow": "Flow",
    "pressure": "Pressure",
    "therapy_pressure": "Therapy Pressure",
    "epap": "EPAP",
    "leak": "Leak",
    "mv": "Minute Ventilation",
    "rr": "Respiratory Rate",
    "tv": "Tidal Volume",
    "spo2": "SpO2",
    "pulse": "Pulse",
    "fl": "Flow Limitation",
    "snore": "Snore",
}


def format_time_offset(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class WaveformRenderer:
    """Render flow waveform using plotext for high-resolution terminal display."""

    def __init__(
        self,
        *,
        console: Console,
        width: int = 80,
        height: int = 20,
        show_events: bool = True,
    ):
        """
        Initialize renderer.

        Args:
            console: Rich console that receives all text output
            width: Chart width in characters (default: 80)
            height: Chart height in lines (default: 20)
            show_events: Whether to show event annotations (default: True)
        """
        self.console = console
        self.width = width
        self.height = height
        self.show_events = show_events

    def render(
        self,
        timestamps: np.ndarray,
        values: np.ndarray,
        machine_events: Sequence[EventType] | None = None,
        programmatic_events: Sequence[EventType] | None = None,
        session_id: int | None = None,
        center_time: str | None = None,
        waveform_type: str = "flow",
    ) -> None:
        """
        Generate high-resolution waveform visualization using plotext.

        Args:
            timestamps: Timestamp array in seconds
            values: Waveform value array
            machine_events: Machine-detected events in window
            programmatic_events: Programmatically-detected events in window
            session_id: Session ID for title
            center_time: Center time for title
            waveform_type: Type of waveform (default: "flow")

        Note:
            Text goes to self.console; plotext writes the chart to stdout. Returns None.
        """
        if len(timestamps) < 2 or len(values) < 2 or timestamps[-1] == timestamps[0]:
            self.console.print("No data in window")
            return

        label = WAVEFORM_LABELS.get(waveform_type, waveform_type.capitalize())
        unit = WAVEFORM_UNITS.get(waveform_type, "?")

        if session_id is not None:
            window_size = timestamps[-1] - timestamps[0]
            if center_time:
                title = f"Session {session_id} - {label} at {center_time} ({window_size:.0f}s)"
            else:
                title = f"Session {session_id} - {label} Waveform"
        else:
            title = f"{label} Waveform"

        sample_rate = len(timestamps) / (timestamps[-1] - timestamps[0])
        self.console.print(
            f"Sample rate: {sample_rate:.0f}Hz | Samples: {len(timestamps)}"
        )
        self.console.print()

        fig = plt.figure
        fig.clear()
        fig.theme("colorless")

        start_time = timestamps[0]
        relative_timestamps = timestamps - start_time
        window_duration = timestamps[-1] - start_time

        fig.draw(fig.signal(relative_timestamps, values, marker="braille").lines())
        fig.title(title)
        fig.label(unit, axis="y")

        tick_interval = (
            10 if window_duration <= 60 else (15 if window_duration <= 120 else 30)
        )
        tick_positions = list(range(0, int(window_duration) + 1, tick_interval))
        tick_labels = [format_time_offset(start_time + t) for t in tick_positions]
        fig.ruler("x").ticks(tick_positions, tick_labels)

        fig.plot_size(self.width, self.height)
        # fig.show() writes to fd 1 from native code, bypassing sys.stdout.
        print(fig.build())

        if self.show_events:
            self.console.print()
            self.console.print("Events in window:")

            if machine_events and len(machine_events) > 0:
                for event in machine_events:
                    time_str = format_time_offset(event.start_time)
                    event_type = getattr(event, "event_type", "Unknown")
                    self.console.print(
                        f"  Machine:      {escape(str(event_type))} at {time_str} ({event.duration:.1f}s)"
                    )
            else:
                self.console.print("  Machine:      (none)")

            if programmatic_events and len(programmatic_events) > 0:
                for event in programmatic_events:
                    time_str = format_time_offset(event.start_time)

                    if hasattr(event, "event_type"):
                        event_type = event.event_type
                    else:
                        event_type = "H"

                    flow_red = getattr(event, "flow_reduction", None)
                    if flow_red is not None:
                        self.console.print(
                            f"  Programmatic: {escape(str(event_type))} at {time_str} ({event.duration:.1f}s, {flow_red * 100:.0f}% flow reduction)"
                        )
                    else:
                        self.console.print(
                            f"  Programmatic: {escape(str(event_type))} at {time_str} ({event.duration:.1f}s)"
                        )
            else:
                self.console.print("  Programmatic: (none)")

    def render_multi(
        self,
        waveform_data: list[tuple[np.ndarray, np.ndarray, str]],
        session_id: int | None = None,
        center_time: str | None = None,
    ) -> None:
        """
        Generate multi-waveform visualization with stacked subplots.

        Args:
            waveform_data: List of (timestamps, values, waveform_type) tuples
            session_id: Session ID for title
            center_time: Center time for title

        Note:
            Text goes to self.console; plotext writes the chart to stdout. Returns None.
            Maximum 4 waveforms supported.
        """
        if not waveform_data:
            self.console.print("No waveform data provided")
            return

        if len(waveform_data) > 4:
            self.console.print("Warning: Maximum 4 waveforms supported, using first 4")
            waveform_data = waveform_data[:4]

        num_plots = len(waveform_data)
        plot_height = max(8, self.height // num_plots)

        fig = plt.figure
        fig.clear()
        fig.subplots(num_plots, 1)

        for idx, (timestamps, values, waveform_type) in enumerate(waveform_data):
            if len(timestamps) == 0 or len(values) == 0:
                continue

            label = WAVEFORM_LABELS.get(waveform_type, waveform_type.capitalize())
            unit = WAVEFORM_UNITS.get(waveform_type, "?")

            start_time = timestamps[0]
            relative_timestamps = timestamps - start_time
            window_duration = timestamps[-1] - start_time

            subplot = fig.subplot(idx + 1, 1)
            subplot.theme("colorless")
            subplot.draw(
                subplot.signal(relative_timestamps, values, marker="braille").lines()
            )

            if idx == 0 and session_id is not None:
                window_size = timestamps[-1] - timestamps[0]
                if center_time:
                    title = f"Session {session_id} - Multi-waveform at {center_time} ({window_size:.0f}s)"
                else:
                    title = f"Session {session_id} - Multi-waveform"
                subplot.title(title)

            subplot.label(f"{label} ({unit})", axis="y")

            if idx == num_plots - 1:
                tick_interval = (
                    10
                    if window_duration <= 60
                    else (15 if window_duration <= 120 else 30)
                )
                tick_positions = list(range(0, int(window_duration) + 1, tick_interval))
                tick_labels = [
                    format_time_offset(start_time + t) for t in tick_positions
                ]
                subplot.ruler("x").ticks(tick_positions, tick_labels)

            subplot.plot_size(self.width, plot_height)

        print(fig.build())

        sample_rates = []
        for timestamps, _values, waveform_type in waveform_data:
            if len(timestamps) > 1 and (timestamps[-1] - timestamps[0]) > 0:
                rate = len(timestamps) / (timestamps[-1] - timestamps[0])
                label = WAVEFORM_LABELS.get(waveform_type, waveform_type)
                sample_rates.append(f"{label}: {rate:.0f}Hz")

        if sample_rates:
            self.console.print()
            self.console.print("Sample rates: " + " | ".join(sample_rates))
