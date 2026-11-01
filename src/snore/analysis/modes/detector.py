"""Unified respiratory event detector using configuration."""

import logging

from typing import Any, cast

import numpy as np

from snore.analysis.modes.baseline import _calculate_baseline
from snore.analysis.modes.classification import (
    _calculate_apnea_confidence,
    _calculate_hypopnea_confidence,
    _calculate_rera_confidence,
    _check_desaturation,
    _classify_apnea_type,
)
from snore.analysis.modes.config import DetectionModeConfig
from snore.analysis.modes.postprocess import (
    EVENT_MATCH_TOLERANCE_SECONDS,
    MatchableEvent,
    _deduplicate_events,
    _merge_adjacent_events,
    _validate_event,
    match_events_by_start_time,
    validate_event_type,
)
from snore.analysis.modes.types import HypopneaMode, ModeResult
from snore.analysis.shared.types import (
    ApneaEvent,
    BreathMetrics,
    HypopneaEvent,
    RERAEvent,
)
from snore.constants import EventDetectionConstants as EDC

logger = logging.getLogger(__name__)


def _mark_breaths_in_events(
    breaths: list[BreathMetrics],
    events: list[ApneaEvent] | list[HypopneaEvent],
) -> None:
    """Flag breaths fully contained in any event as ``in_event``.

    Contained breaths are dropped from the rolling baselines used by downstream
    detection. Breaths are time-ordered, so each event's scan stops once a
    breath starts after the event ends.
    """
    for event in events:
        for breath in breaths:
            if breath.start_time > event.end_time:
                break  # breaths are time-ordered
            if (
                breath.start_time >= event.start_time
                and breath.end_time <= event.end_time
            ):
                breath.in_event = True


class EventDetector:
    """
    Configurable respiratory event detector.

    Uses DetectionModeConfig to parameterize detection behavior.
    Supports both AASM-compliant and relaxed detection modes.
    """

    def __init__(self, config: DetectionModeConfig):
        """
        Initialize detector with configuration.

        Args:
            config: Detection mode configuration
        """
        self.config = config

    @property
    def name(self) -> str:
        """Get mode name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get mode description."""
        return self.config.description

    def _detect_events_resmed(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray],
        sample_rate: float,
    ) -> list[ApneaEvent]:
        """
        ResMed-style detection combining multiple strategies.

        1. Gap detection - finds periods with no breaths
        2. Near-zero flow - finds sustained low flow in raw signal
        3. Amplitude reduction - lower threshold (50% vs 90%)

        Deduplicates overlapping detections from different methods.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Tuple of (timestamps, flow_values)
            sample_rate: Sampling rate in Hz

        Returns:
            List of deduplicated and merged apnea events
        """
        logger.info(f"{self.config.name}: Running multi-strategy detection")

        all_events = []

        gap_events = self._detect_breath_gaps(breaths, min_gap_seconds=10.0)
        all_events.extend(gap_events)

        timestamps, flow_values = flow_data
        if len(timestamps) > 1:
            zero_events = self._detect_near_zero_flow(
                flow_values, timestamps, zero_threshold=2.0, min_duration=10.0
            )
            all_events.extend(zero_events)

        amplitude_events = self._detect_apneas(breaths, flow_data, sample_rate)
        all_events.extend(amplitude_events)

        logger.info(
            f"{self.config.name}: Combined {len(all_events)} events from all strategies"
        )

        deduplicated = _deduplicate_events(
            self.config, all_events, overlap_threshold=0.5
        )

        merged = cast(
            list[ApneaEvent],
            _merge_adjacent_events(deduplicated, self.config.merge_gap),
        )

        return merged

    def detect_events(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray],
        sample_rate: float,
        session_duration_hours: float,
    ) -> ModeResult:
        """
        Run detection algorithm and return results.

        Branches to mode-specific detection:
        - ResMed: Multi-strategy (gap + near-zero + amplitude)
        - AASM/aasm_relaxed: Amplitude-based only

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Tuple of (timestamps, flow_values)
            sample_rate: Sampling rate in Hz (from waveform metadata)
            session_duration_hours: Total session duration in hours

        Returns:
            ModeResult with detected events and metrics
        """
        # Reset per-call breath state so each detect_events call is
        # self-contained. detect_events runs once per mode over the SAME breaths
        # list, and the apnea and hypopnea loops set in_event=True; without this
        # reset a prior mode's flags would contaminate this mode's baselines and
        # make results mode-order-dependent. in_event is the only breath field
        # detect_events mutates.
        for breath in breaths:
            breath.in_event = False

        if self.config.name == "resmed":
            apneas = self._detect_events_resmed(breaths, flow_data, sample_rate)
        else:
            apneas = self._detect_apneas(breaths, flow_data, sample_rate)

        hypopneas = self._detect_hypopneas(
            breaths, flow_data, spo2_signal=None, exclude_events=apneas
        )

        reras: list[RERAEvent] = []
        if self.config.rera_detection_enabled:
            reras = self._detect_reras(breaths, apneas, hypopneas)

        total_events = len(apneas) + len(hypopneas)
        ahi = (
            total_events / session_duration_hours if session_duration_hours > 0 else 0.0
        )

        rdi = (
            (total_events + len(reras)) / session_duration_hours
            if session_duration_hours > 0
            else 0.0
        )

        return ModeResult(
            mode_name=self.config.name,
            apneas=apneas,
            hypopneas=hypopneas,
            reras=reras,
            ahi=ahi,
            rdi=rdi,
            metadata={
                "config": self.config.name,
                "baseline_method": self.config.baseline_method.value,
                "apnea_threshold": self.config.apnea_threshold,
                "validation_threshold": self.config.apnea_validation_threshold,
                "rera_count": len(reras),
            },
        )

    # ========================================================================
    # Event Detection
    # ========================================================================

    def _detect_apneas(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray],
        sample_rate: float,
    ) -> list[ApneaEvent]:
        """
        Detect apnea events using configured thresholds.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Tuple of (timestamps, flow_values)
            sample_rate: Sampling rate in Hz

        Returns:
            List of detected apnea events
        """
        if not breaths:
            logger.warning("No breaths provided for apnea detection")
            return []

        logger.info(
            f"{self.config.name}: Detecting apneas (threshold={self.config.apnea_threshold * 100}%, "
            f"validation={self.config.apnea_validation_threshold * 100}%)"
        )

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = _calculate_baseline(self.config, breaths, i)
            baselines[i] = baseline

            if baseline > 0:
                if self.config.metric == "amplitude":
                    value = breath.amplitude
                else:
                    value = breath.tidal_volume
                reduction = 1.0 - (value / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        logger.debug(
            f"Baseline range: {np.min(baselines):.1f} - {np.max(baselines):.1f}, mean: {np.mean(baselines):.1f}"
        )
        logger.debug(
            f"Reduction range: {np.min(reductions) * 100:.1f}% - {np.max(reductions) * 100:.1f}%, mean: {np.mean(reductions) * 100:.1f}%"
        )

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            self.config.apnea_threshold,
            self.config.min_event_duration,
        )

        logger.debug(f"Found {len(regions)} potential apnea events (before merging)")

        apneas = []
        for start_idx, end_idx, duration in regions:
            event_breaths = breaths[start_idx:end_idx]
            event_reductions = reductions[start_idx:end_idx]
            event_baselines = baselines[start_idx:end_idx]

            if (
                len(event_breaths) == 0
                or len(event_reductions) == 0
                or len(event_baselines) == 0
            ):
                continue

            if not _validate_event(self.config, reductions, start_idx, end_idx):
                logger.debug(
                    f"  Rejecting apnea {start_idx}-{end_idx}: fails validation"
                )
                continue

            start_time = event_breaths[0].start_time
            end_time = event_breaths[-1].end_time
            avg_reduction = float(np.mean(event_reductions))
            avg_baseline = float(np.mean(event_baselines))

            flow_signal = None
            timestamps, flow_values = flow_data
            mask = (timestamps >= start_time) & (timestamps <= end_time)
            flow_signal = flow_values[mask]

            event_type, classification_confidence = _classify_apnea_type(
                flow_signal=flow_signal, sample_rate=sample_rate
            )
            confidence = _calculate_apnea_confidence(
                avg_reduction, duration, avg_baseline
            )

            logger.debug(
                f"  Apnea at {start_time:.1f}s: type={event_type}, duration={duration:.1f}s, "
                f"reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f}, "
                f"confidence={confidence:.2f}, classification_confidence={classification_confidence:.2f}"
            )

            apneas.append(
                ApneaEvent(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    duration=float(duration),
                    event_type=event_type,
                    flow_reduction=float(avg_reduction),
                    confidence=float(confidence),
                    classification_confidence=float(classification_confidence),
                    baseline_flow=float(avg_baseline),
                )
            )

        apneas = cast(
            list[ApneaEvent], _merge_adjacent_events(apneas, self.config.merge_gap)
        )

        oa = sum(1 for a in apneas if a.event_type == "OA")
        ca = sum(1 for a in apneas if a.event_type == "CA")
        ma = sum(1 for a in apneas if a.event_type == "MA")
        ua = sum(1 for a in apneas if a.event_type == "UA")

        logger.info(
            f"{self.config.name}: Detected {len(apneas)} apneas: {oa} OA, {ca} CA, {ma} MA, {ua} UA"
        )

        _mark_breaths_in_events(breaths, apneas)

        return apneas

    def _detect_hypopneas(
        self,
        breaths: list[BreathMetrics],
        flow_data: tuple[np.ndarray, np.ndarray] | None = None,
        spo2_signal: np.ndarray | None = None,
        exclude_events: list[ApneaEvent] | None = None,
    ) -> list[HypopneaEvent]:
        """
        Detect hypopnea events using configured mode.

        Supports multiple detection modes:
        - AASM_3PCT/4PCT: Requires SpO2 desaturation (3% or 4%)
        - FLOW_ONLY: 40% flow reduction without SpO2
        - DISABLED: Skip detection

        Falls back to FLOW_ONLY if SpO2 unavailable and fallback enabled.

        Args:
            breaths: List of BreathMetrics objects
            flow_data: Optional tuple of (timestamps, flow_values)
            spo2_signal: SpO2 data for desaturation detection
            exclude_events: List of apnea events to exclude from hypopnea detection

        Returns:
            List of detected hypopnea events
        """
        if not breaths:
            return []

        if self.config.hypopnea_mode == HypopneaMode.DISABLED:
            logger.info(f"{self.config.name}: Hypopnea detection disabled")
            return []

        has_spo2 = spo2_signal is not None
        actual_mode = self.config.hypopnea_mode

        if not has_spo2:
            if self.config.hypopnea_mode in (
                HypopneaMode.AASM_3PCT,
                HypopneaMode.AASM_4PCT,
            ):
                if self.config.hypopnea_flow_only_fallback:
                    logger.info(
                        f"{self.config.name}: No SpO2 data, falling back to flow-only hypopnea detection"
                    )
                    actual_mode = HypopneaMode.FLOW_ONLY
                else:
                    logger.info(
                        f"{self.config.name}: Skipping hypopnea detection - no SpO2 data and fallback disabled"
                    )
                    return []

        logger.info(
            f"{self.config.name}: Detecting hypopneas (mode: {actual_mode.value})"
        )

        min_threshold = self.config.hypopnea_min_threshold

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = _calculate_baseline(self.config, breaths, i)
            baselines[i] = baseline

            if baseline > 0:
                if self.config.metric == "amplitude":
                    value = breath.amplitude
                else:
                    value = breath.tidal_volume
                reduction = 1.0 - (value / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        if exclude_events:
            excluded_count = 0
            for i, breath in enumerate(breaths):
                for apnea in exclude_events:
                    if (
                        breath.start_time < apnea.end_time
                        and breath.end_time > apnea.start_time
                    ):
                        reductions[i] = 0.0
                        excluded_count += 1
                        break
            if excluded_count > 0:
                logger.debug(
                    f"Excluded {excluded_count} breaths overlapping with apnea events"
                )

        breaths_in_range = np.sum(
            (reductions >= min_threshold)
            & (reductions < EDC.APNEA_FLOW_REDUCTION_THRESHOLD)
        )
        logger.debug(
            f"Breaths in hypopnea range ({min_threshold * 100:.0f}-89%): {breaths_in_range}"
        )

        regions = self._find_consecutive_reduced_breaths(
            breaths,
            reductions,
            min_threshold,
            self.config.min_event_duration,
        )

        hypopneas = []
        for start_idx, end_idx, duration in regions:
            event_reductions = reductions[start_idx:end_idx]
            if len(event_reductions) == 0:
                continue
            avg_reduction = float(np.mean(event_reductions))

            if avg_reduction >= EDC.APNEA_FLOW_REDUCTION_THRESHOLD:
                logger.debug(
                    f"  Skipping region {start_idx}-{end_idx}: avg reduction {avg_reduction * 100:.1f}% >= 90% (apnea)"
                )
                continue

            if not _validate_event(
                self.config,
                reductions,
                start_idx,
                end_idx,
                threshold=min_threshold,
            ):
                logger.debug(
                    f"  Rejecting hypopnea {start_idx}-{end_idx}: fails validation"
                )
                continue

            event_baselines = baselines[start_idx:end_idx]
            if len(event_baselines) == 0:
                continue
            avg_baseline = float(np.mean(event_baselines))

            start_time = breaths[start_idx].start_time
            end_time = breaths[end_idx - 1].end_time

            has_desaturation = None
            if spo2_signal is not None and flow_data is not None:
                timestamps, _ = flow_data
                if len(spo2_signal) != len(timestamps):
                    logger.warning(
                        f"SpO2/flow timestamp mismatch: {len(spo2_signal)} vs {len(timestamps)} - "
                        "skipping desaturation check"
                    )
                    has_desaturation = None
                else:
                    mask = (timestamps >= start_time) & (timestamps <= end_time)
                    if np.any(mask):
                        if actual_mode == HypopneaMode.AASM_4PCT:
                            has_desaturation = _check_desaturation(
                                spo2_signal[mask], threshold=4.0
                            )
                        else:
                            has_desaturation = _check_desaturation(spo2_signal[mask])

            confidence = _calculate_hypopnea_confidence(
                avg_reduction, duration, has_desaturation, detection_mode=actual_mode
            )

            logger.debug(
                f"  Hypopnea at {start_time:.1f}s: duration={duration:.1f}s, "
                f"reduction={avg_reduction * 100:.1f}%, baseline={avg_baseline:.1f}, confidence={confidence:.2f}"
            )

            hypopneas.append(
                HypopneaEvent(
                    start_time=float(start_time),
                    end_time=float(end_time),
                    duration=float(duration),
                    flow_reduction=float(avg_reduction),
                    confidence=float(confidence),
                    baseline_flow=float(avg_baseline),
                    has_desaturation=has_desaturation,
                )
            )

        hypopneas = cast(
            list[HypopneaEvent],
            _merge_adjacent_events(hypopneas, self.config.merge_gap),
        )

        # Mark contained breaths as in-event so they are excluded from the
        # baselines used by downstream detection (RERAs run after hypopneas).
        # Applied after merging so it does not perturb the hypopneas computed in
        # this same call.
        _mark_breaths_in_events(breaths, hypopneas)

        logger.info(f"{self.config.name}: Detected {len(hypopneas)} hypopneas")

        return hypopneas

    def _detect_reras(
        self,
        breaths: list[BreathMetrics],
        apneas: list[ApneaEvent],
        hypopneas: list[HypopneaEvent],
    ) -> list[RERAEvent]:
        """
        Detect RERA-like events using FLOW event algorithm.

        Detects sequences of flow-limited breaths ending with recovery breath,
        without EEG arousal detection. Uses amplitude reduction as proxy for
        flow limitation.

        This is the analysis-time amplitude-crescendo RERA detector and feeds
        ModeResult.rdi. It is a DIFFERENT RERA definition from the query-time
        FL-run proxy (breath_service.py::_count_fl_run_reras, versioned as
        RERA_PROXY_ALGO_VERSION) that feeds the nightly rera_index/rdi; the two
        disagree by construction.

        Algorithm (thresholds from DetectionModeConfig):
        1. Find sequences of ≥2 breaths with amplitude reduction in
           [rera_reduction_min, rera_reduction_max)
        2. Look for recovery breath with reduction < rera_recovery_reduction_max
           and amplitude increase ≥ rera_recovery_increase_min over the run mean
        3. Ensure ≥2-breath separation from apneas/hypopneas

        Args:
            breaths: List of BreathMetrics objects
            apneas: Detected apnea events (to avoid overlap)
            hypopneas: Detected hypopnea events (to avoid overlap)

        Returns:
            List of detected RERA events
        """
        if not breaths or len(breaths) < 3:
            return []

        logger.info(f"{self.config.name}: Detecting RERA events")

        baselines = np.zeros(len(breaths))
        reductions = np.zeros(len(breaths))

        for i, breath in enumerate(breaths):
            baseline = _calculate_baseline(self.config, breaths, i)
            baselines[i] = baseline

            if baseline > 0:
                reduction = 1.0 - (breath.amplitude / baseline)
                reductions[i] = max(0.0, min(1.0, reduction))
            else:
                reductions[i] = 0.0

        excluded = np.zeros(len(breaths), dtype=bool)
        for event in list(apneas) + list(hypopneas):
            for i, breath in enumerate(breaths):
                if breath.start_time > event.end_time:
                    break  # breaths are time-ordered
                if (
                    breath.start_time >= event.start_time
                    and breath.end_time <= event.end_time
                ):
                    excluded[i] = True

        reduction_min = self.config.rera_reduction_min
        reduction_max = self.config.rera_reduction_max
        recovery_reduction_max = self.config.rera_recovery_reduction_max
        recovery_increase_min = self.config.rera_recovery_increase_min

        reras = []
        i = 0
        while i < len(breaths) - 2:
            if excluded[i]:
                i += 1
                continue

            if reduction_min <= reductions[i] < reduction_max:
                seq_start = i
                seq_count = 0
                while (
                    i < len(breaths)
                    and not excluded[i]
                    and reduction_min <= reductions[i] < reduction_max
                ):
                    seq_count += 1
                    i += 1

                if seq_count >= 2 and i < len(breaths):
                    seq_end_time = breaths[i - 1].end_time
                    min_duration = seq_end_time - breaths[seq_start].start_time
                    if min_duration < self.config.min_event_duration * 0.5:
                        continue

                    recovery_found = False
                    recovery_idx = -1

                    for j in range(i, min(i + 2, len(breaths))):
                        if excluded[j]:
                            continue

                        if reductions[j] < recovery_reduction_max:
                            seq_avg_amplitude = np.mean(
                                [breaths[k].amplitude for k in range(seq_start, i)]
                            )
                            recovery_amplitude = breaths[j].amplitude

                            if seq_avg_amplitude > 0:
                                increase_pct = (
                                    recovery_amplitude - seq_avg_amplitude
                                ) / seq_avg_amplitude

                                if increase_pct >= recovery_increase_min:
                                    recovery_found = True
                                    recovery_idx = j
                                    break

                    if recovery_found and recovery_idx >= 0:
                        start_time = breaths[seq_start].start_time
                        end_time = breaths[recovery_idx].end_time
                        duration = end_time - start_time

                        if duration >= self.config.min_event_duration:
                            seq_baseline = np.mean(baselines[seq_start:i])
                            recovery_amplitude = breaths[recovery_idx].amplitude
                            seq_avg_amplitude = np.mean(
                                [breaths[k].amplitude for k in range(seq_start, i)]
                            )

                            amplitude_increase = (
                                (recovery_amplitude - seq_avg_amplitude)
                                / seq_avg_amplitude
                                if seq_avg_amplitude > 0
                                else 0.0
                            )

                            confidence = _calculate_rera_confidence(
                                seq_count, float(amplitude_increase), duration
                            )

                            logger.debug(
                                f"  RERA at {start_time:.1f}s: duration={duration:.1f}s, "
                                f"obstructed_breaths={seq_count}, recovery_increase={amplitude_increase * 100:.1f}%, "
                                f"confidence={confidence:.2f}"
                            )

                            reras.append(
                                RERAEvent(
                                    start_time=float(start_time),
                                    end_time=float(end_time),
                                    duration=float(duration),
                                    obstructed_breath_count=seq_count,
                                    recovery_amplitude_increase_pct=float(
                                        amplitude_increase * 100
                                    ),
                                    confidence=float(confidence),
                                    baseline_flow=float(seq_baseline),
                                )
                            )

                            i = recovery_idx + 1
                            continue

            i += 1

        logger.info(f"{self.config.name}: Detected {len(reras)} RERA events")

        return reras

    # ========================================================================
    # Shared Utilities (no duplication - single implementation)
    # ========================================================================

    def _detect_breath_gaps(
        self,
        breaths: list[BreathMetrics],
        min_gap_seconds: float = 10.0,
    ) -> list[ApneaEvent]:
        """
        Detect apneas based on absence of breaths (gap detection).

        Finds periods ≥min_gap_seconds between consecutive breaths.
        Gaps indicate breathing cessation - classified as Central Apnea
        since no respiratory effort is detectable.

        This complements amplitude-based detection for events where
        breath segmentation fails entirely (no breaths to measure).

        Args:
            breaths: List of BreathMetrics objects
            min_gap_seconds: Minimum gap duration to qualify as apnea (default 10.0s)

        Returns:
            List of detected gap-based apnea events
        """
        events = []
        for i in range(1, len(breaths)):
            prev_end = breaths[i - 1].end_time
            curr_start = breaths[i].start_time
            gap = curr_start - prev_end

            if gap >= min_gap_seconds:
                events.append(
                    ApneaEvent(
                        start_time=float(prev_end),
                        end_time=float(curr_start),
                        duration=float(gap),
                        event_type="CA",  # Gap = no effort = central
                        flow_reduction=1.0,  # 100% - no breathing at all
                        confidence=0.85,
                        classification_confidence=0.9,  # High confidence: gap = no effort = CA
                        baseline_flow=0.0,  # No flow during gap
                        detection_method="gap",
                    )
                )

        if events:
            logger.info(f"{self.config.name}: Detected {len(events)} gap-based apneas")

        return events

    def _detect_near_zero_flow(
        self,
        flow_signal: np.ndarray,
        timestamps: np.ndarray,
        zero_threshold: float = 2.0,
        min_duration: float = 10.0,
    ) -> list[ApneaEvent]:
        """
        Detect apneas based on sustained near-zero flow in raw signal.

        Complements breath-based detection for cases where:
        - Breath segmentation misses the event entirely
        - Flow is too low to segment into distinct breaths

        Uses contiguous region detection to find periods where |flow| < threshold.

        Note: This is DIFFERENT from flow limitation flatness, which
        measures time at PEAK flow. This measures time at ZERO flow.

        Args:
            flow_signal: Raw flow signal values (L/min)
            timestamps: Timestamp array corresponding to flow samples
            zero_threshold: Flow threshold for "near-zero" (default 2.0 L/min)
            min_duration: Minimum event duration in seconds (default 10.0s)

        Returns:
            List of detected near-zero flow apnea events
        """
        total_duration = timestamps[-1] - timestamps[0]
        sample_rate = len(timestamps) / total_duration
        min_samples = int(min_duration * sample_rate)

        near_zero_mask = np.abs(flow_signal) < zero_threshold

        padded = np.empty(len(near_zero_mask) + 2, dtype=np.int8)
        padded[0] = 0
        padded[1:-1] = near_zero_mask
        padded[-1] = 0
        delta = np.diff(padded)
        starts = np.flatnonzero(delta == 1)  # index of first in-run sample
        ends = np.flatnonzero(delta == -1)  # one past last in-run sample

        events = []
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            if e - s >= min_samples:
                start_time = float(timestamps[s])
                end_time = float(timestamps[e - 1])
                events.append(
                    ApneaEvent(
                        start_time=start_time,
                        end_time=end_time,
                        duration=end_time - start_time,
                        event_type="CA",
                        flow_reduction=1.0,
                        confidence=0.80,
                        classification_confidence=0.85,  # High confidence: near-zero = CA
                        baseline_flow=0.0,
                        detection_method="near_zero_flow",
                    )
                )

        if events:
            logger.info(
                f"{self.config.name}: Detected {len(events)} near-zero flow apneas"
            )

        return events

    def _find_consecutive_reduced_breaths(
        self,
        breaths: list[BreathMetrics],
        reductions: np.ndarray,
        threshold: float,
        min_duration: float,
    ) -> list[tuple[int, int, float]]:
        """
        Find runs of consecutive breaths meeting reduction threshold.

        Per AASM/ResMed standards, events terminate when 2+ consecutive breaths
        fall below the recovery threshold (50% of baseline = 50% reduction).

        Args:
            breaths: List of BreathMetrics objects
            reductions: Array of reduction values (0.0-1.0) per breath
            threshold: Minimum reduction to qualify (e.g., 0.9 for 90%)
            min_duration: Minimum total duration in seconds

        Returns:
            List of (start_idx, end_idx, total_duration) tuples
        """
        regions = []
        in_region = False
        region_start = 0
        recovery_count = 0
        recovery_threshold = EDC.EVENT_TERMINATION_RECOVERY
        min_recovery_breaths = EDC.EVENT_TERMINATION_MIN_BREATHS

        breaths_meeting_threshold = np.sum(reductions >= threshold)
        logger.debug(
            f"Finding consecutive reduced breaths: {breaths_meeting_threshold} breaths meet threshold >= {threshold * 100:.1f}%"
        )

        for i, reduction in enumerate(reductions):
            if reduction >= threshold and not in_region:
                in_region = True
                region_start = i
                recovery_count = 0
            elif in_region:
                if reduction < recovery_threshold:
                    recovery_count += 1
                    if recovery_count >= min_recovery_breaths:
                        end_idx = i - min_recovery_breaths + 1
                        start_time = breaths[region_start].start_time
                        end_time = breaths[end_idx - 1].end_time
                        duration = end_time - start_time
                        if duration >= min_duration:
                            regions.append((region_start, end_idx, duration))
                        in_region = False
                        recovery_count = 0
                else:
                    recovery_count = 0

        if in_region:
            start_time = breaths[region_start].start_time
            end_time = breaths[-1].end_time
            duration = end_time - start_time
            if duration >= min_duration:
                regions.append((region_start, len(breaths), duration))

        return regions

    def validate_against_machine_events(
        self,
        programmatic_apneas: list[ApneaEvent],
        programmatic_hypopneas: list[HypopneaEvent],
        machine_apneas: list[ApneaEvent],
        machine_hypopneas: list[HypopneaEvent],
        tolerance_seconds: float = EVENT_MATCH_TOLERANCE_SECONDS,
        programmatic_reras: list[RERAEvent] | None = None,
        machine_reras: list[RERAEvent] | None = None,
    ) -> dict[str, Any]:
        """
        Validate programmatic event detection against machine-detected events.

        Compares timing of detected events with machine events and calculates
        agreement statistics (sensitivity, precision, F1 score).

        RERAs are validated only when the device flagged at least one machine RE
        event: many ResMed configurations never emit RE, and their absence must
        not read as an algorithm failure. When no machine RERAs are supplied the
        RERA validation is excluded from the averaged agreement and from the
        combined event lists, and reported via
        ``rera_validation_status = "no_machine_re_events"`` so overall agreement
        stays comparable to the apnea/hypopnea-only baseline.

        Args:
            programmatic_apneas: Apneas detected by our algorithm
            programmatic_hypopneas: Hypopneas detected by our algorithm
            machine_apneas: Apneas reported by the CPAP machine
            machine_hypopneas: Hypopneas reported by the CPAP machine
            tolerance_seconds: Max time difference for event matching (default 5s)
            programmatic_reras: RERAs detected by our algorithm (optional)
            machine_reras: RERAs reported by the CPAP machine (optional)

        Returns:
            Dictionary with validation metrics for apneas, hypopneas and RERAs,
            the per-type matched/unmatched event lists ("apnea_matches",
            "hypopnea_matches", "rera_matches"), the combined cross-type
            "false_negative_events" / "false_positive_events" lists, and
            "rera_validation_status" ("ok" or "no_machine_re_events").
        """
        programmatic_reras = programmatic_reras or []
        machine_reras = machine_reras or []
        has_machine_reras = len(machine_reras) > 0

        apnea_validation, apnea_matches = validate_event_type(
            programmatic_apneas, machine_apneas, tolerance_seconds
        )
        hypopnea_validation, hypopnea_matches = validate_event_type(
            programmatic_hypopneas, machine_hypopneas, tolerance_seconds
        )
        rera_validation, rera_matches = validate_event_type(
            programmatic_reras, machine_reras, tolerance_seconds
        )

        all_programmatic: list[MatchableEvent] = [
            *programmatic_apneas,
            *programmatic_hypopneas,
        ]
        all_machine: list[MatchableEvent] = [*machine_apneas, *machine_hypopneas]
        # Fold RERAs into the combined lists only when the device provides RERAs
        # to match against; otherwise unmatched programmatic RERAs would inflate
        # the cross-type false-positive list against absent ground truth.
        if has_machine_reras:
            all_programmatic.extend(programmatic_reras)
            all_machine.extend(machine_reras)
        all_programmatic.sort(key=lambda e: e.start_time)
        all_machine.sort(key=lambda e: e.start_time)

        combined = match_events_by_start_time(
            all_programmatic, all_machine, tolerance_seconds
        )

        event_validations = [apnea_validation, hypopnea_validation]
        total_machine = len(machine_apneas) + len(machine_hypopneas)
        total_programmatic = len(programmatic_apneas) + len(programmatic_hypopneas)
        if has_machine_reras:
            event_validations.append(rera_validation)
            total_machine += len(machine_reras)
            total_programmatic += len(programmatic_reras)

        n = len(event_validations)

        return {
            "apnea_validation": apnea_validation,
            "hypopnea_validation": hypopnea_validation,
            "rera_validation": rera_validation,
            "rera_validation_status": (
                "ok" if has_machine_reras else "no_machine_re_events"
            ),
            "apnea_matches": apnea_matches,
            "hypopnea_matches": hypopnea_matches,
            "rera_matches": rera_matches,
            "matched_events": [programmatic for programmatic, _ in combined.matched],
            "false_negative_events": combined.false_negatives,
            "false_positive_events": combined.false_positives,
            "overall_agreement": {
                "total_machine_events": total_machine,
                "total_programmatic_events": total_programmatic,
                "average_sensitivity": sum(v.sensitivity for v in event_validations)
                / n,
                "average_precision": sum(v.precision for v in event_validations) / n,
                "average_f1": sum(v.f1_score for v in event_validations) / n,
            },
        }
