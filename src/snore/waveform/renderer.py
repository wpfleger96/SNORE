"""ASCII waveform rendering for terminal display."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from snore.analysis.shared.types import ApneaEvent, HypopneaEvent
    from snore.analysis.types import AnalysisEvent

    EventType = AnalysisEvent | ApneaEvent | HypopneaEvent


def _format_time_offset(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class AsciiWaveformRenderer:
    """Render flow waveform as ASCII art for terminal display."""

    def __init__(
        self,
        width: int = 80,
        height: int = 20,
        show_events: bool = True,
    ):
        """
        Initialize renderer.

        Args:
            width: Chart width in characters (default: 80)
            height: Chart height in lines (default: 20)
            show_events: Whether to show event annotations (default: True)
        """
        self.width = width
        self.height = height
        self.show_events = show_events

    def render(
        self,
        timestamps: np.ndarray,
        flow_values: np.ndarray,
        machine_events: Sequence["EventType"] | None = None,
        programmatic_events: Sequence["EventType"] | None = None,
        session_id: int | None = None,
        center_time: str | None = None,
    ) -> str:
        """
        Generate ASCII representation of waveform.

        Args:
            timestamps: Timestamp array in seconds
            flow_values: Flow value array in L/min
            machine_events: Machine-detected events in window
            programmatic_events: Programmatically-detected events in window
            session_id: Session ID for title
            center_time: Center time for title

        Returns:
            ASCII art string
        """
        if len(timestamps) == 0 or len(flow_values) == 0:
            return "No data in window"

        lines = []

        if session_id is not None:
            window_size = timestamps[-1] - timestamps[0]

            if center_time:
                title = f"Session {session_id} - Flow Waveform at {center_time} (window: {window_size:.0f}s)"
            else:
                title = f"Session {session_id} - Flow Waveform"

            lines.append(title)
            lines.append(
                f"Sample rate: {len(timestamps) / window_size:.0f}Hz | Samples: {len(timestamps)}"
            )
            lines.append("")

        y_label_width = 6
        chart_width = self.width - y_label_width - 1

        step = max(1, len(flow_values) // chart_width)
        sampled_values = flow_values[::step][:chart_width]

        min_flow = float(np.min(sampled_values))
        max_flow = float(np.max(sampled_values))
        flow_range = max_flow - min_flow if max_flow != min_flow else 1.0

        for row in range(self.height):
            row_flow = max_flow - (row / (self.height - 1)) * flow_range

            line = f"{row_flow:>5.0f} │"

            for col in range(min(chart_width, len(sampled_values))):
                val = sampled_values[col]

                normalized = (val - min_flow) / flow_range
                point_row = int((1 - normalized) * (self.height - 1))

                if point_row == row:
                    line += "●"
                elif row == self.height // 2 and min_flow < 0 < max_flow:
                    line += "─"
                else:
                    line += " "

            lines.append(line)

        x_axis = " " * y_label_width + "└" + "─" * min(chart_width, len(sampled_values))
        lines.append(x_axis)

        start_time_str = _format_time_offset(timestamps[0])
        end_time_str = _format_time_offset(timestamps[-1])
        spacing = max(0, chart_width - len(start_time_str) - len(end_time_str))
        time_line = (
            f"{' ' * y_label_width} {start_time_str}{' ' * spacing}{end_time_str}"
        )
        lines.append(time_line)

        if self.show_events:
            lines.append("")
            lines.append("Events in window:")

            if machine_events and len(machine_events) > 0:
                for event in machine_events:
                    time_str = _format_time_offset(event.start_time)
                    event_type = getattr(event, "event_type", "Unknown")
                    lines.append(
                        f"  Machine:      {event_type} at {time_str} ({event.duration:.1f}s)"
                    )
            else:
                lines.append("  Machine:      (none)")

            if programmatic_events and len(programmatic_events) > 0:
                for event in programmatic_events:
                    time_str = _format_time_offset(event.start_time)

                    if hasattr(event, "event_type"):
                        event_type = event.event_type
                    else:
                        event_type = "H"

                    flow_red = getattr(event, "flow_reduction", None)
                    if flow_red is not None:
                        lines.append(
                            f"  Programmatic: {event_type} at {time_str} ({event.duration:.1f}s, {flow_red * 100:.0f}% flow reduction)"
                        )
                    else:
                        lines.append(
                            f"  Programmatic: {event_type} at {time_str} ({event.duration:.1f}s)"
                        )
            else:
                lines.append("  Programmatic: (none)")

        return "\n".join(lines)
