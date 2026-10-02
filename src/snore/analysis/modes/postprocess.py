"""Event post-processing: validation, deduplication, merging and matching."""

from __future__ import annotations

import logging

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from snore.analysis.modes.config import DetectionModeConfig
from snore.analysis.shared.types import ApneaEvent, HypopneaEvent, RERAEvent

if TYPE_CHECKING:
    from snore.services.schemas import EventValidationResult

logger = logging.getLogger(__name__)

# Single source of truth for machine vs programmatic event matching tolerance
EVENT_MATCH_TOLERANCE_SECONDS = 5.0

# Events comparable by start-time tolerance matching. Apneas, hypopneas and
# RERAs all expose ``start_time``; matching never inspects type-specific fields.
MatchableEvent = ApneaEvent | HypopneaEvent | RERAEvent


class HasStartTime(Protocol):
    """Anything one-to-one matching can pair: only ``start_time`` is read."""

    @property
    def start_time(self) -> float: ...


def _calculate_event_overlap(event1: ApneaEvent, event2: ApneaEvent) -> float:
    """
    Calculate overlap ratio between two events.

    Args:
        event1: First apnea event
        event2: Second apnea event

    Returns:
        Overlap ratio (0.0-1.0) relative to shorter event duration
    """
    overlap_start = max(event1.start_time, event2.start_time)
    overlap_end = min(event1.end_time, event2.end_time)

    if overlap_start >= overlap_end:
        return 0.0

    overlap_duration = overlap_end - overlap_start
    shorter_duration = min(event1.duration, event2.duration)

    return overlap_duration / shorter_duration


def _validate_event(
    config: DetectionModeConfig,
    reductions: np.ndarray,
    start_idx: int,
    end_idx: int,
    threshold: float | None = None,
) -> bool:
    """
    Validate that event contains at least one breath meeting the threshold.

    Uses configured validation threshold.

    Args:
        config: Detection mode configuration
        reductions: Array of reduction values per breath (0.0-1.0)
        start_idx: Event start index in breaths array
        end_idx: Event end index in breaths array
        threshold: Override threshold (if None, uses config.apnea_validation_threshold)

    Returns:
        True if at least one breath meets the threshold
    """
    event_reductions = reductions[start_idx:end_idx]
    if len(event_reductions) == 0:
        return False

    max_reduction = float(np.max(event_reductions))

    validation_threshold = (
        threshold if threshold is not None else config.apnea_validation_threshold
    )
    return max_reduction >= validation_threshold


def _deduplicate_events(
    config: DetectionModeConfig,
    events: list[ApneaEvent],
    overlap_threshold: float = 0.5,
) -> list[ApneaEvent]:
    """
    Remove duplicate/overlapping events, keeping highest confidence.

    When multiple detection methods find the same event, keep the
    detection with highest confidence. Merge events that overlap
    by more than overlap_threshold (50% default).

    Args:
        config: Detection mode configuration
        events: List of apnea events (potentially overlapping)
        overlap_threshold: Minimum overlap ratio to consider duplicates (0.0-1.0)

    Returns:
        List of deduplicated events
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.start_time)

    deduplicated = []
    current = sorted_events[0]

    for next_event in sorted_events[1:]:
        overlap = _calculate_event_overlap(current, next_event)

        if overlap > overlap_threshold:
            if next_event.confidence > current.confidence:
                logger.debug(
                    f"  Replacing {current.detection_method} event at {current.start_time:.1f}s "
                    f"(conf={current.confidence:.2f}) with {next_event.detection_method} "
                    f"(conf={next_event.confidence:.2f})"
                )
                current = next_event
            else:
                logger.debug(
                    f"  Keeping {current.detection_method} event at {current.start_time:.1f}s "
                    f"(conf={current.confidence:.2f}), dropping {next_event.detection_method} "
                    f"(conf={next_event.confidence:.2f})"
                )
        else:
            deduplicated.append(current)
            current = next_event

    deduplicated.append(current)

    if len(events) > len(deduplicated):
        logger.info(
            f"{config.name}: Deduplicated {len(events)} events to {len(deduplicated)}"
        )

    return deduplicated


# The merge helpers stay on the narrower ApneaEvent | HypopneaEvent union (unlike
# the MatchableEvent matching helpers): RERAs are never merged, and merging reads
# type-specific fields (flow_reduction, classification_confidence, has_arousal).
def _merge_adjacent_events(
    events: Sequence[ApneaEvent | HypopneaEvent],
    max_gap: float,
) -> list[ApneaEvent | HypopneaEvent]:
    """
    Merge events that are close together in time AND of the same type.

    Per AASM standards, only events of the same type should be merged.

    Args:
        events: List of ApneaEvent or HypopneaEvent objects
        max_gap: Maximum gap in seconds to merge

    Returns:
        List of merged events
    """
    if len(events) <= 1:
        return list(events)

    merged = []
    current = events[0]

    for next_event in events[1:]:
        gap = next_event.start_time - current.end_time
        same_type = type(next_event) == type(current)

        if gap <= max_gap and same_type:
            current = _merge_two_events(current, next_event)
        else:
            merged.append(current)
            current = next_event

    merged.append(current)
    return merged


def _merge_two_events(
    event1: ApneaEvent | HypopneaEvent,
    event2: ApneaEvent | HypopneaEvent,
) -> ApneaEvent | HypopneaEvent:
    """
    Merge two adjacent events of the same type.

    Args:
        event1: First event
        event2: Second event

    Returns:
        Merged event
    """
    merged_duration = event2.end_time - event1.start_time

    if isinstance(event1, ApneaEvent) and isinstance(event2, ApneaEvent):
        return ApneaEvent(
            start_time=event1.start_time,
            end_time=event2.end_time,
            duration=merged_duration,
            event_type=(
                event1.event_type
                if event1.classification_confidence >= event2.classification_confidence
                else event2.event_type
            ),
            flow_reduction=(event1.flow_reduction + event2.flow_reduction) / 2,
            confidence=min(event1.confidence, event2.confidence),
            classification_confidence=min(
                event1.classification_confidence, event2.classification_confidence
            ),
            baseline_flow=event1.baseline_flow,
            detection_method=event1.detection_method,
        )
    elif isinstance(event1, HypopneaEvent) and isinstance(event2, HypopneaEvent):
        return HypopneaEvent(
            start_time=event1.start_time,
            end_time=event2.end_time,
            duration=merged_duration,
            flow_reduction=(event1.flow_reduction + event2.flow_reduction) / 2,
            confidence=min(event1.confidence, event2.confidence),
            baseline_flow=event1.baseline_flow,
            has_arousal=event1.has_arousal or event2.has_arousal,
            has_desaturation=event1.has_desaturation or event2.has_desaturation,
        )

    event_typed: ApneaEvent | HypopneaEvent = event1
    return event_typed


@dataclass(frozen=True)
class MatchedEvents[E: HasStartTime]:
    """
    Outcome of greedy one-to-one matching of programmatic vs machine events.

    Attributes:
        matched: (programmatic, machine) event pairs matched within tolerance
        false_positives: Programmatic events with no matching machine event
        false_negatives: Machine events with no matching programmatic event
    """

    matched: list[tuple[E, E]]
    false_positives: list[E]
    false_negatives: list[E]


def match_events_by_start_time[E: HasStartTime](
    programmatic: Sequence[E],
    machine: Sequence[E],
    tolerance_seconds: float = EVENT_MATCH_TOLERANCE_SECONDS,
) -> MatchedEvents[E]:
    """
    Greedily match programmatic events to machine events by start time.

    Each machine event is matched at most once; each programmatic event
    matches the first unmatched machine event within tolerance.

    Args:
        programmatic: Events detected by our algorithm
        machine: Events reported by the CPAP machine
        tolerance_seconds: Max start-time difference for a match

    Returns:
        MatchedEvents with matched pairs and unmatched events per side
    """
    matched: list[tuple[E, E]] = []
    matched_machine_indices: set[int] = set()
    false_positives: list[E] = []

    for prog_event in programmatic:
        match_found = False
        for m_idx, mach_event in enumerate(machine):
            if m_idx in matched_machine_indices:
                continue

            time_diff = abs(prog_event.start_time - mach_event.start_time)
            if time_diff <= tolerance_seconds:
                matched.append((prog_event, mach_event))
                matched_machine_indices.add(m_idx)
                match_found = True
                break

        if not match_found:
            false_positives.append(prog_event)

    false_negatives: list[E] = [
        mach_event
        for m_idx, mach_event in enumerate(machine)
        if m_idx not in matched_machine_indices
    ]

    return MatchedEvents(
        matched=matched,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def split_by_tolerance_match(
    events: Sequence[MatchableEvent],
    references: Sequence[MatchableEvent],
    tolerance_seconds: float = EVENT_MATCH_TOLERANCE_SECONDS,
) -> tuple[list[MatchableEvent], list[MatchableEvent]]:
    """
    Split events into (matched, unmatched) by start-time proximity.

    An event is matched if ANY reference event starts within tolerance
    (references may match multiple events).

    Args:
        events: Events to classify
        references: Reference events to match against
        tolerance_seconds: Max start-time difference for a match

    Returns:
        Tuple of (matched events, unmatched events), preserving input order
    """
    matched: list[MatchableEvent] = []
    unmatched: list[MatchableEvent] = []

    for event in events:
        is_matched = any(
            abs(event.start_time - ref.start_time) <= tolerance_seconds
            for ref in references
        )
        if is_matched:
            matched.append(event)
        else:
            unmatched.append(event)

    return matched, unmatched


def validate_event_type(
    programmatic: Sequence[MatchableEvent],
    machine: Sequence[MatchableEvent],
    tolerance_seconds: float = EVENT_MATCH_TOLERANCE_SECONDS,
) -> tuple[EventValidationResult, MatchedEvents[MatchableEvent]]:
    """
    Validate a single event type against machine events.

    Args:
        programmatic: Events detected by our algorithm
        machine: Events reported by the CPAP machine
        tolerance_seconds: Max start-time difference for a match

    Returns:
        Tuple of (validation statistics, matched/unmatched event lists)
    """
    from snore.services.schemas import EventValidationResult

    match_result = match_events_by_start_time(programmatic, machine, tolerance_seconds)

    matched = len(match_result.matched)
    machine_count = len(machine)
    programmatic_count = len(programmatic)
    false_positives = programmatic_count - matched
    false_negatives = machine_count - matched

    if machine_count == 0:
        sensitivity = 1.0
    elif matched + false_negatives > 0:
        sensitivity = matched / (matched + false_negatives)
    else:
        sensitivity = 0.0

    if matched + false_positives > 0:
        precision = matched / (matched + false_positives)
    else:
        precision = 0.0 if machine_count == 0 else 1.0

    if precision + sensitivity > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0.0

    total_unique = machine_count + programmatic_count - matched
    agreement_percentage = (matched / total_unique * 100) if total_unique > 0 else 100.0

    validation = EventValidationResult(
        machine_event_count=machine_count,
        programmatic_event_count=programmatic_count,
        matched_events=matched,
        false_positives=false_positives,
        false_negatives=false_negatives,
        sensitivity=sensitivity,
        precision=precision,
        f1_score=f1_score,
        agreement_percentage=agreement_percentage,
    )

    return validation, match_result
