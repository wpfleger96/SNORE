"""
Event reconciliation module for merging machine and programmatic events.

This module handles the hybrid approach of combining machine-flagged events
(from CPAP device) with programmatically detected events, managing conflicts
and tracking event sources.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from oscar_mcp.constants import (
    EVENT_TYPE_OBSTRUCTIVE_APNEA,
    EVENT_TYPE_CENTRAL_APNEA,
    EVENT_TYPE_CLEAR_AIRWAY,
    EVENT_TYPE_MIXED_APNEA,
    EVENT_TYPE_HYPOPNEA,
    EVENT_TYPE_RERA,
)

logger = logging.getLogger(__name__)


@dataclass
class RespiratoryEvent:
    """Unified respiratory event from either machine or programmatic source."""

    event_type: str
    start_time: float
    duration: float
    source: str
    confidence: Optional[float] = None
    flow_reduction: Optional[float] = None
    has_desaturation: Optional[bool] = None
    baseline_flow: Optional[float] = None


class EventReconciliation:
    """
    Reconcile machine-flagged and programmatically detected events.

    This class implements the hybrid approach where machine events are the
    primary source of truth (matching OSCAR's behavior) while programmatic
    events provide validation and additional insights.
    """

    # Time window (seconds) to consider events as overlapping
    OVERLAP_THRESHOLD = 5.0

    def __init__(self):
        """Initialize event reconciliation."""
        pass

    def reconcile_events(
        self,
        machine_events: List[RespiratoryEvent],
        programmatic_events: List[RespiratoryEvent],
    ) -> Tuple[List[RespiratoryEvent], Dict]:
        """
        Merge and reconcile machine and programmatic events.

        Args:
            machine_events: Events flagged by CPAP device
            programmatic_events: Events detected by analysis engine

        Returns:
            Tuple of:
            - Merged event list (prioritizing machine events)
            - Reconciliation statistics and discrepancies
        """
        merged_events = []
        reconciliation_stats = {
            "machine_event_count": len(machine_events),
            "programmatic_event_count": len(programmatic_events),
            "matched_events": 0,
            "machine_only": 0,
            "programmatic_only": 0,
            "discrepancies": [],
        }

        machine_events_sorted = sorted(machine_events, key=lambda e: e.start_time)
        programmatic_events_sorted = sorted(programmatic_events, key=lambda e: e.start_time)

        used_programmatic_indices = set()

        for machine_event in machine_events_sorted:
            merged_events.append(machine_event)

            matching_prog_idx = self._find_overlapping_event(
                machine_event, programmatic_events_sorted, used_programmatic_indices
            )

            if matching_prog_idx is not None:
                used_programmatic_indices.add(matching_prog_idx)
                reconciliation_stats["matched_events"] += 1

                prog_event = programmatic_events_sorted[matching_prog_idx]
                if machine_event.event_type != prog_event.event_type:
                    reconciliation_stats["discrepancies"].append(
                        {
                            "start_time": machine_event.start_time,
                            "machine_type": machine_event.event_type,
                            "programmatic_type": prog_event.event_type,
                            "reason": "type_mismatch",
                        }
                    )
            else:
                reconciliation_stats["machine_only"] += 1

        for idx, prog_event in enumerate(programmatic_events_sorted):
            if idx not in used_programmatic_indices:
                merged_events.append(prog_event)
                reconciliation_stats["programmatic_only"] += 1

        merged_events_sorted = sorted(merged_events, key=lambda e: e.start_time)

        return merged_events_sorted, reconciliation_stats

    def _find_overlapping_event(
        self,
        target_event: RespiratoryEvent,
        candidate_events: List[RespiratoryEvent],
        excluded_indices: set,
    ) -> Optional[int]:
        """
        Find event in candidates that overlaps with target event.

        Args:
            target_event: Event to match
            candidate_events: List of candidate events to search
            excluded_indices: Indices already matched (skip these)

        Returns:
            Index of matching event or None if no match found
        """
        target_start = target_event.start_time
        target_end = target_start + target_event.duration

        for idx, candidate in enumerate(candidate_events):
            if idx in excluded_indices:
                continue

            candidate_start = candidate.start_time
            candidate_end = candidate_start + candidate.duration

            time_diff = abs(target_start - candidate_start)
            if time_diff <= self.OVERLAP_THRESHOLD:
                return idx

            overlap_start = max(target_start, candidate_start)
            overlap_end = min(target_end, candidate_end)
            if overlap_start < overlap_end:
                return idx

        return None

    def calculate_indices(
        self, events: List[RespiratoryEvent], session_hours: float
    ) -> Dict[str, float]:
        """
        Calculate AHI and RDI from event list.

        Args:
            events: List of respiratory events
            session_hours: Session duration in hours

        Returns:
            Dictionary with AHI, RDI, and event counts
        """
        if session_hours <= 0:
            logger.warning("Invalid session hours for index calculation")
            return {"ahi": 0.0, "rdi": 0.0, "total_events": 0}

        event_counts = {
            "obstructive_apneas": 0,
            "central_apneas": 0,
            "clear_airway": 0,
            "mixed_apneas": 0,
            "hypopneas": 0,
            "reras": 0,
            "total_events": 0,
        }

        for event in events:
            event_counts["total_events"] += 1

            if event.event_type == EVENT_TYPE_OBSTRUCTIVE_APNEA:
                event_counts["obstructive_apneas"] += 1
            elif event.event_type == EVENT_TYPE_CENTRAL_APNEA:
                event_counts["central_apneas"] += 1
            elif event.event_type == EVENT_TYPE_CLEAR_AIRWAY:
                event_counts["clear_airway"] += 1
            elif event.event_type == EVENT_TYPE_MIXED_APNEA:
                event_counts["mixed_apneas"] += 1
            elif event.event_type == EVENT_TYPE_HYPOPNEA:
                event_counts["hypopneas"] += 1
            elif event.event_type == EVENT_TYPE_RERA:
                event_counts["reras"] += 1

        ahi_count = sum(
            event_counts[k.replace("_", "_").lower()]
            for k in [
                "obstructive_apneas",
                "central_apneas",
                "clear_airway",
                "mixed_apneas",
                "hypopneas",
            ]
        )
        rdi_count = ahi_count + event_counts["reras"]

        ahi = ahi_count / session_hours
        rdi = rdi_count / session_hours

        return {
            "ahi": ahi,
            "rdi": rdi,
            **event_counts,
        }
