"""Analysis utility functions."""

from snore.analysis.shared.types import ApneaEvent, HypopneaEvent
from snore.analysis.types import AnalysisEvent
from snore.constants import EVENT_TYPE_HYPOPNEA, get_apnea_type, is_apnea_type


def convert_machine_events(
    machine_events: list[AnalysisEvent],
) -> tuple[list[ApneaEvent], list[HypopneaEvent], float]:
    """
    Convert raw machine AnalysisEvents to typed ApneaEvent/HypopneaEvent lists.

    Args:
        machine_events: List of machine-detected events with Unix timestamps

    Returns:
        Tuple of (apnea_events, hypopnea_events, session_start_time)
        where session_start_time is the earliest event timestamp
    """
    if machine_events:
        session_start = min(event.start_time for event in machine_events)
    else:
        session_start = 0.0

    apneas: list[ApneaEvent] = []
    hypopneas: list[HypopneaEvent] = []

    for event in machine_events:
        relative_time = event.start_time - session_start

        if is_apnea_type(event.event_type):
            apnea_type = get_apnea_type(event.event_type)
            if apnea_type:
                apneas.append(
                    ApneaEvent(
                        start_time=relative_time,
                        end_time=relative_time + event.duration,
                        duration=event.duration,
                        event_type=apnea_type,
                        flow_reduction=event.flow_reduction or 0.0,
                        confidence=event.confidence or 1.0,
                        classification_confidence=1.0,
                        baseline_flow=event.baseline_flow or 0.0,
                        detection_method="machine",
                    )
                )
        elif event.event_type == EVENT_TYPE_HYPOPNEA:
            hypopneas.append(
                HypopneaEvent(
                    start_time=relative_time,
                    end_time=relative_time + event.duration,
                    duration=event.duration,
                    flow_reduction=event.flow_reduction or 0.0,
                    confidence=event.confidence or 1.0,
                    baseline_flow=event.baseline_flow or 0.0,
                    has_desaturation=event.has_desaturation,
                    has_arousal=False,
                )
            )

    return apneas, hypopneas, session_start
