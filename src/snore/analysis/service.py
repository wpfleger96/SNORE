"""
Analysis service for orchestrating programmatic analysis.

This module provides the main interface for running comprehensive CPAP session
analysis, loading data from the database, and storing results.

I/O–compute separation
----------------------
``AnalysisService.load_session_inputs_raw()`` performs **only** database reads,
returning raw bytes in a ``RawSessionBlobs`` dataclass.

``AnalysisService.prepare_inputs()`` deserialises blobs into numpy arrays
(compute phase — no DB session required).

``AnalysisService.load_session_inputs()`` is a convenience wrapper that calls
both methods in sequence; the session remains open across the NumPy work.
Callers that need the session closed before compute should call the two methods
separately — see ``AnalysisFacade.run_analysis`` and ``BatchAnalysisCoordinator``.

``AnalysisService.compute_analysis()`` runs the full analysis pipeline on a
detached ``AnalysisInputs`` DTO.  No session is held.

``AnalysisService.analyze_session()`` orchestrates the three phases
(raw-fetch + deserialise + compute + persist) with the caller-provided session.
"""

import logging
import time

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.data.waveform_loader import (
    WaveformLoader,
    deserialize_waveform_blob,
    detect_and_mark_artifacts,
    fetch_waveform_blob,
)
from snore.analysis.modes import DEFAULT_MODE, get_mode
from snore.analysis.shared.breath_segmenter import BreathSegmenter
from snore.analysis.shared.feature_extractors import (
    WaveformFeatureExtractor,
    largest_inspiratory_segment,
)
from snore.analysis.shared.flow_limitation import FlowLimitationClassifier
from snore.analysis.shared.pattern_detector import ComplexPatternDetector
from snore.analysis.shared.pulse_detector import PulseChangeDetector
from snore.analysis.shared.versioning import (
    LEAK_VALID_MAX_ALIGNMENT_GAP_S,
    LEAK_VALID_THRESHOLD_LPM,
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisRunMetadata,
)
from snore.analysis.types import (
    AnalysisComputation,
    AnalysisEvent,
    AnalysisResult,
    ComputedBreath,
    _machine_ahi,
)
from snore.constants import BreathSegmentationConstants as BSC
from snore.constants import PatternDetectionConstants as PDC
from snore.constants import PulseChangeConstants as PCC
from snore.database import models
from snore.therapy_hours import TherapyHoursBasis, therapy_hours

logger = logging.getLogger(__name__)

__all__ = [
    "AnalysisInputs",
    "AnalysisComputation",
    "RawSessionBlobs",
    "AnalysisService",
    "AnalysisResult",
]

_RAMP_KEYS: frozenset[str] = frozenset({"ramp_enabled", "ramp_time", "smart_ramp"})


# Optional waveform channels fetched by ``load_session_inputs_raw``:
# (waveform_type, log level on absence, log message template).
_OPTIONAL_CHANNELS: tuple[tuple[str, int, str], ...] = (
    ("spo2", logging.INFO, "No SpO2 data available: %s"),
    ("pulse", logging.DEBUG, "Pulse waveform not available: %s"),
    ("leak", logging.DEBUG, "Leak waveform not available: %s"),
)


@dataclass
class RawSessionBlobs:
    """Raw DB-fetched bytes for one session — no NumPy, no ORM references.

    Created by ``AnalysisService.load_session_inputs_raw()`` (I/O phase only).
    Passed to ``AnalysisService.prepare_inputs()`` outside the ORM session so
    that all NumPy work (deserialization, artifact detection) runs without a
    held session lock.
    """

    session_id: int
    flow_blob: bytes
    flow_sample_count: int
    flow_metadata: dict[str, Any]
    machine_events: list[AnalysisEvent]
    spo2_blob: bytes | None = None
    spo2_sample_count: int = 0
    spo2_metadata: dict[str, Any] = field(default_factory=dict)
    pulse_blob: bytes | None = None
    pulse_sample_count: int = 0
    pulse_metadata: dict[str, Any] = field(default_factory=dict)
    # Quality-flag inputs: leak channel for leak_valid derivation.
    leak_blob: bytes | None = None
    leak_sample_count: int = 0
    leak_metadata: dict[str, Any] = field(default_factory=dict)
    modes: list[str] = field(default_factory=lambda: [DEFAULT_MODE])
    # Primary mode for RERA/recovery-marker storage.
    primary_mode: str = DEFAULT_MODE
    # Device manufacturer string from the Device row — used to gate
    # vendor-specific heuristics (e.g. trigger/cycle inference).
    device_manufacturer: str = "ResMed"
    # Validity-flag inputs: mask-on segments + ramp settings.
    mask_on_segments: list[tuple[float, float]] | None = None
    session_duration_s: float | None = None
    ramp_enabled: bool | None = None
    ramp_time_minutes: int | None = None
    smart_ramp: bool = False


@dataclass
class AnalysisInputs:
    """Detached DTO carrying all DB-loaded inputs needed for analysis compute.

    Created by ``AnalysisService.load_session_inputs()``.  Contains only plain
    Python types and numpy arrays — no ORM objects, no live session references.
    Safe to pass across thread boundaries or between executor tasks.
    """

    session_id: int
    flow_timestamps: np.ndarray
    flow_values: np.ndarray
    sample_rate: float
    machine_events: list[AnalysisEvent]
    spo2_values: np.ndarray | None = None
    pulse_timestamps: np.ndarray | None = None
    pulse_values: np.ndarray | None = None
    # Quality-flag inputs: leak waveform (timestamps, values) for leak_valid derivation.
    leak_timestamps: np.ndarray | None = None
    leak_values: np.ndarray | None = None
    modes: list[str] = field(default_factory=lambda: [DEFAULT_MODE])
    # Primary mode for RERA/recovery-marker storage.
    primary_mode: str = DEFAULT_MODE
    # Device manufacturer string — gates vendor-specific heuristics.
    device_manufacturer: str = "ResMed"
    # Validity-flag inputs: mask-on segments + ramp settings.
    mask_on_segments: list[tuple[float, float]] | None = None
    session_duration_s: float | None = None
    ramp_enabled: bool | None = None
    ramp_time_minutes: int | None = None
    smart_ramp: bool = False


def _resolve_primary_mode(modes: list[str], primary_mode: str | None) -> str:
    """Resolve and validate primary_mode against modes list.

    Rules:
    - If primary_mode is supplied, it MUST be in modes → ValueError if not.
    - If not supplied and DEFAULT_MODE is in modes → DEFAULT_MODE.
    - If not supplied and DEFAULT_MODE is NOT in modes → ValueError (caller must
      supply primary_mode explicitly for non-default mode sets).
    """
    if primary_mode is not None:
        if primary_mode not in modes:
            raise ValueError(
                f"primary_mode {primary_mode!r} must be a member of modes {modes}"
            )
        return primary_mode
    # No primary_mode supplied: default only when DEFAULT_MODE is present.
    if DEFAULT_MODE in modes:
        return DEFAULT_MODE
    raise ValueError(
        f"primary_mode must be supplied explicitly when modes {modes!r} "
        f"exclude the DEFAULT_MODE {DEFAULT_MODE!r}"
    )


class AnalysisService:
    """
    Service for running programmatic analysis on CPAP sessions.

    This service handles:
    - Loading waveform data from database
    - Running the programmatic analysis engine
    - Storing results in the database
    - Providing structured results for consumption

    Two construction modes:
    - **DB mode**: ``AnalysisService(db_session, profile_id=n)`` — ``profile_id`` is
      required when a session is provided; all DB queries are scoped to that profile.
    - **Compute mode**: ``AnalysisService()`` — ``db_session=None``; only
      ``compute_analysis()`` and ``prepare_inputs()`` may be called.

    Example:
        >>> service = AnalysisService(db_session, profile_id=42)
        >>> result = await service.analyze_session(session_id=123)
        >>> print(f"AHI: {result['event_timeline']['ahi']}")
    """

    def __init__(
        self,
        db_session: AsyncSession | None = None,
        profile_id: int | None = None,
        min_breath_duration: float = BSC.MIN_BREATH_DURATION,
    ):
        """
        Initialize analysis service.

        Args:
            db_session: SQLAlchemy async database session.  Pass ``None`` for a
                compute-only instance (``compute_analysis``/``prepare_inputs`` only).
            profile_id: Required when ``db_session`` is provided.  All DB queries
                are scoped to this profile — sessions not owned by it raise or
                return ``None`` exactly as if missing.
            min_breath_duration: Minimum breath duration for segmentation (seconds)
        """
        if db_session is not None and profile_id is None:
            raise ValueError(
                "AnalysisService requires profile_id when db_session is provided"
            )
        self.db_session = db_session
        self.profile_id = profile_id
        self.waveform_loader = (
            WaveformLoader(db_session) if db_session is not None else None
        )
        self.breath_segmenter = BreathSegmenter(min_breath_duration=min_breath_duration)
        self.feature_extractor = WaveformFeatureExtractor()
        self.flow_classifier = FlowLimitationClassifier()
        self.pattern_detector = ComplexPatternDetector()
        self.pulse_detector = PulseChangeDetector(
            bpm_threshold=PCC.BPM_THRESHOLD,
            duration_threshold=PCC.DURATION_THRESHOLD,
        )

    async def _load_machine_events(
        self, session_id: int, session_start_ts: float
    ) -> list[AnalysisEvent]:
        """Load machine-flagged events from database.

        Args:
            session_id: Database session ID
            session_start_ts: Session start Unix timestamp, supplied by the caller
                (already fetched in load_session_inputs_raw) to avoid a redundant
                Session + Device query.

        Returns:
            List of respiratory events with session-relative timestamps
        """
        assert self.db_session is not None, "_load_machine_events requires a DB session"

        events = (
            (
                await self.db_session.execute(
                    select(models.Event)
                    .filter_by(session_id=session_id)
                    .order_by(models.Event.start_time)
                )
            )
            .scalars()
            .all()
        )

        respiratory_events = []
        for event in events:
            offset_seconds = event.start_time.timestamp() - session_start_ts

            respiratory_events.append(
                AnalysisEvent(
                    event_type=event.event_type,
                    start_time=offset_seconds,
                    duration=event.duration_seconds or 10.0,
                    source="machine",
                    confidence=1.0,
                )
            )

        return respiratory_events

    async def load_session_inputs(
        self,
        session_id: int,
        modes: list[str] | None = None,
    ) -> AnalysisInputs:
        """Load all DB inputs for a session and return them as a detached DTO.

        Convenience wrapper: calls ``load_session_inputs_raw()`` (DB I/O only)
        then ``prepare_inputs()`` (NumPy deserialization + artifact detection).
        Both phases run while the ORM session is still held.  Callers that need
        the session closed *before* NumPy work should call those two methods
        directly — see ``AnalysisFacade.run_analysis`` and the batch coordinator.

        Args:
            session_id: Database session ID
            modes: Detection modes to include in the DTO (None = default mode)

        Returns:
            ``AnalysisInputs`` DTO with copied numpy arrays.

        Raises:
            ValueError: If session not found or has no flow waveform data.
        """
        raw = await self.load_session_inputs_raw(session_id, modes=modes)
        return AnalysisService.prepare_inputs(raw)

    async def _fetch_optional_channel(
        self,
        session_id: int,
        waveform_type: str,
        log_level: int,
        unavailable_msg: str,
    ) -> tuple[bytes | None, int, dict[str, Any]]:
        """Fetch one optional waveform channel as raw bytes.

        Absence (or any fetch failure) is logged at ``log_level`` and mapped to
        the empty ``(None, 0, {})`` triple — optional channels never raise.
        """
        assert self.db_session is not None, (
            "_fetch_optional_channel requires a DB session"
        )
        try:
            return await fetch_waveform_blob(self.db_session, session_id, waveform_type)
        except Exception as e:
            logger.log(log_level, unavailable_msg, e)
            return None, 0, {}

    async def load_session_inputs_raw(
        self,
        session_id: int,
        modes: list[str] | None = None,
        primary_mode: str | None = None,
    ) -> RawSessionBlobs:
        """Fetch all DB inputs for a session as raw bytes — **I/O phase only**.

        No NumPy, no deserialization, no artifact detection.  The ORM session is
        NOT needed after this call returns.  Call ``AnalysisService.prepare_inputs()``
        outside the session scope to convert blobs to numpy arrays.

        Args:
            session_id: Database session ID.
            modes: Detection modes (``None`` = default mode).
            primary_mode: Mode whose recovery markers are persisted (must be in
                ``modes``; defaults to ``DEFAULT_MODE`` when it is present in
                ``modes``).

        Returns:
            ``RawSessionBlobs`` DTO with raw bytes and scalar metadata.

        Raises:
            ValueError: If session not found or has no flow waveform data, or
                if ``primary_mode`` is not a member of ``modes``.
        """
        assert self.db_session is not None, (
            "load_session_inputs_raw requires a DB session"
        )
        modes_list = list(modes) if modes is not None else [DEFAULT_MODE]
        resolved_primary = _resolve_primary_mode(modes_list, primary_mode)

        # Single query: Session + Device (inner join — device_id is NOT NULL) +
        # Setting (outer join — 0-3 rows for ramp keys).  The outer join fans
        # out to one row per matching Setting; the Device/Session columns are
        # identical across all rows.  When no Setting rows match, the outer join
        # produces one row with Setting.key/value = NULL.
        stmt = (
            select(
                models.Session,
                models.Device.manufacturer.label("manufacturer"),
                models.Setting.key.label("setting_key"),
                models.Setting.value.label("setting_value"),
            )
            .join(models.Device, models.Session.device_id == models.Device.id)
            .outerjoin(
                models.Setting,
                (models.Setting.session_id == models.Session.id)
                & models.Setting.key.in_(_RAMP_KEYS),
            )
            .where(models.Session.id == session_id)
            .where(models.Device.profile_id == self.profile_id)
        )

        rows = (await self.db_session.execute(stmt)).all()
        if not rows:
            raise ValueError(f"Session {session_id} not found")

        session = rows[0].Session
        device_manufacturer: str = (
            rows[0].manufacturer if rows[0].manufacturer is not None else "unknown"
        )
        settings_by_key: dict[str, str | None] = {
            row.setting_key: row.setting_value
            for row in rows
            if row.setting_key is not None
        }

        # Validity-flag inputs: persisted mask-on segments + ramp settings.
        mask_on_segments = _parse_mask_on_segments(session.mask_on_segments, session_id)
        session_duration_s = (
            float(session.duration_seconds)
            if session.duration_seconds is not None
            else None
        )

        ramp_enabled = _parse_bool_setting(settings_by_key.get("ramp_enabled"))
        ramp_time_minutes = _parse_int_setting(settings_by_key.get("ramp_time"))
        smart_ramp = _parse_bool_setting(settings_by_key.get("smart_ramp")) is True

        try:
            flow_blob, flow_sample_count, flow_metadata = await fetch_waveform_blob(
                self.db_session, session_id, "flow"
            )
        except Exception as e:
            logger.error(f"Failed to load flow waveform for session {session_id}: {e}")
            raise ValueError(
                f"No flow waveform data available for session {session_id}"
            ) from e

        if flow_sample_count == 0:
            raise ValueError(f"Empty flow waveform data for session {session_id}")

        machine_events = await self._load_machine_events(
            session_id, session.start_time.timestamp()
        )

        # Optional channels: absence is logged (per-channel level), never raised.
        # Leak feeds quality-flag derivation.
        optional: dict[str, tuple[bytes | None, int, dict[str, Any]]] = {}
        for waveform_type, log_level, unavailable_msg in _OPTIONAL_CHANNELS:
            optional[waveform_type] = await self._fetch_optional_channel(
                session_id, waveform_type, log_level, unavailable_msg
            )
        spo2_blob, spo2_sample_count, spo2_metadata = optional["spo2"]
        pulse_blob, pulse_sample_count, pulse_metadata = optional["pulse"]
        leak_blob, leak_sample_count, leak_metadata = optional["leak"]

        return RawSessionBlobs(
            session_id=session_id,
            flow_blob=flow_blob,
            flow_sample_count=flow_sample_count,
            flow_metadata=flow_metadata,
            machine_events=machine_events,
            spo2_blob=spo2_blob,
            spo2_sample_count=spo2_sample_count,
            spo2_metadata=spo2_metadata,
            pulse_blob=pulse_blob,
            pulse_sample_count=pulse_sample_count,
            pulse_metadata=pulse_metadata,
            leak_blob=leak_blob,
            leak_sample_count=leak_sample_count,
            leak_metadata=leak_metadata,
            modes=modes_list,
            primary_mode=resolved_primary,
            device_manufacturer=device_manufacturer,
            mask_on_segments=mask_on_segments,
            session_duration_s=session_duration_s,
            ramp_enabled=ramp_enabled,
            ramp_time_minutes=ramp_time_minutes,
            smart_ramp=smart_ramp,
        )

    @staticmethod
    def prepare_inputs(raw: RawSessionBlobs) -> AnalysisInputs:
        """Convert a ``RawSessionBlobs`` DTO into a compute-ready ``AnalysisInputs`` DTO.

        **Compute phase only — no DB access.**  Performs deserialization
        (``np.frombuffer``) and artifact detection (NumPy).  Safe to call after
        the ORM session has been closed.

        Args:
            raw: Raw blobs from ``load_session_inputs_raw()``.

        Returns:
            ``AnalysisInputs`` with copied numpy arrays.
        """
        # Deserialise flow waveform (compute — no session).
        flow_timestamps, flow_values = deserialize_waveform_blob(
            raw.flow_blob, raw.flow_sample_count
        )
        # Artifact detection (NumPy — no session).
        artifact_mask = detect_and_mark_artifacts(flow_values, "flow")
        _ = artifact_mask  # mask available to callers via metadata if needed

        sample_rate = float(raw.flow_metadata.get("sample_rate", 25.0))

        spo2_values: np.ndarray | None = None
        if raw.spo2_blob is not None and raw.spo2_sample_count > 0:
            try:
                _, spo2_raw = deserialize_waveform_blob(
                    raw.spo2_blob, raw.spo2_sample_count
                )
                if len(spo2_raw) > 0 and len(spo2_raw) == len(flow_timestamps):
                    spo2_values = spo2_raw.copy()
                elif len(spo2_raw) > 0:
                    logger.warning(
                        f"SpO2 length mismatch ({len(spo2_raw)} vs {len(flow_timestamps)}), skipping"
                    )
            except Exception as e:
                logger.warning(f"Failed to deserialise SpO2 blob: {e}")

        pulse_timestamps: np.ndarray | None = None
        pulse_values: np.ndarray | None = None
        if raw.pulse_blob is not None and raw.pulse_sample_count > 0:
            try:
                pt, pv = deserialize_waveform_blob(
                    raw.pulse_blob, raw.pulse_sample_count
                )
                pulse_timestamps = pt.copy()
                pulse_values = pv.copy()
            except Exception as e:
                logger.debug(f"Failed to deserialise pulse blob: {e}")

        # Leak waveform for quality-flag derivation.
        leak_timestamps: np.ndarray | None = None
        leak_values: np.ndarray | None = None
        if raw.leak_blob is not None and raw.leak_sample_count > 0:
            try:
                lt, lv = deserialize_waveform_blob(raw.leak_blob, raw.leak_sample_count)
                leak_timestamps = lt.copy()
                leak_values = lv.copy()
            except Exception as e:
                logger.debug(f"Failed to deserialise leak blob: {e}")

        return AnalysisInputs(
            session_id=raw.session_id,
            flow_timestamps=flow_timestamps.copy(),
            flow_values=flow_values.copy(),
            sample_rate=sample_rate,
            machine_events=raw.machine_events,
            spo2_values=spo2_values,
            pulse_timestamps=pulse_timestamps,
            pulse_values=pulse_values,
            leak_timestamps=leak_timestamps,
            leak_values=leak_values,
            modes=raw.modes,
            primary_mode=raw.primary_mode,
            device_manufacturer=raw.device_manufacturer,
            mask_on_segments=raw.mask_on_segments,
            session_duration_s=raw.session_duration_s,
            ramp_enabled=raw.ramp_enabled,
            ramp_time_minutes=raw.ramp_time_minutes,
            smart_ramp=raw.smart_ramp,
        )

    def compute_analysis(self, inputs: AnalysisInputs) -> AnalysisComputation:
        """Run pure analysis compute on a detached ``AnalysisInputs`` DTO.

        **No database access** — all inputs come from the DTO.  Safe to call
        after the ORM session has been closed.

        Returns an ``AnalysisComputation`` private envelope containing:
        - ``summary``: the public ``AnalysisResult`` (stored in
          ``programmatic_result_json``) — unchanged shape, no breath rows.
        - ``breaths``: per-breath ``ComputedBreath`` list (persisted as
          ``models.Breath`` children of the ``AnalysisResult`` row).

        Breaths NEVER enter ``summary``; ``programmatic_result_json`` stays
        exactly the same size as before.

        Args:
            inputs: Detached DTO from ``load_session_inputs()``.

        Returns:
            ``AnalysisComputation`` envelope.

        Raises:
            ValueError: If no breaths can be segmented.
        """
        timestamps = inputs.flow_timestamps
        flow_values = inputs.flow_values
        sample_rate = inputs.sample_rate
        session_id = inputs.session_id
        primary_mode = inputs.primary_mode

        session_duration_hours = (
            therapy_hours(
                TherapyHoursBasis.WAVEFORM_COVERAGE,
                sample_count=len(timestamps),
                sample_rate=sample_rate,
            )
            or 0.0
        )
        logger.info(
            f"Loaded {len(timestamps)} flow samples at {sample_rate}Hz "
            f"({session_duration_hours:.1f} hours)"
        )
        logger.info(f"Loaded {len(inputs.machine_events)} machine-flagged events")

        breaths = self.breath_segmenter.segment_breaths(
            timestamps, flow_values, sample_rate
        )
        logger.info(f"Segmented {len(breaths)} breaths")

        if not breaths:
            raise ValueError(f"No breaths segmented for session {session_id}")

        breath_features = []
        breath_shape_by_number: dict[int, Any] = {}
        # Memoize the extracted inspiratory flow so _build_computed_breaths reuses
        # it instead of repeating the searchsorted + slice + segment extraction.
        insp_flow_by_number: dict[int, np.ndarray] = {}
        for breath in breaths:
            breath_start_idx = np.searchsorted(timestamps, breath.start_time)
            breath_end_idx = np.searchsorted(timestamps, breath.end_time)

            breath_flow = flow_values[breath_start_idx:breath_end_idx]
            insp_flow = largest_inspiratory_segment(breath_flow)
            insp_flow_by_number[breath.breath_number] = insp_flow

            if len(insp_flow) > 10:
                shape = self.feature_extractor.extract_shape_features(
                    insp_flow, sample_rate
                )
                peaks = self.feature_extractor.extract_peak_features(
                    insp_flow, sample_rate
                )
                breath_features.append((breath.breath_number, shape, peaks))
                breath_shape_by_number[breath.breath_number] = shape

        flow_analysis = self.flow_classifier.analyze_session(breath_features)
        logger.info(f"Flow limitation index: {flow_analysis.flow_limitation_index:.3f}")

        # Map breath_number → FlowPattern for per-breath ComputedBreath assembly.
        flow_pattern_by_number: dict[int, Any] = {
            p.breath_number: p for p in flow_analysis.patterns
        }

        tidal_volumes = np.array([b.tidal_volume for b in breaths])
        breath_timestamps = np.array([b.start_time for b in breaths])
        respiratory_rates = np.array([b.respiratory_rate_rolling for b in breaths])

        csr_detection = self.pattern_detector.detect_csr(
            breath_timestamps, tidal_volumes, window_minutes=10.0
        )

        csr_episodes_list = self.pattern_detector.detect_csr_episodes(
            breath_timestamps,
            tidal_volumes,
            window_minutes=PDC.CSR_WINDOW_MINUTES,
            step_minutes=PDC.CSR_WINDOW_STEP_MINUTES,
        )

        periodic_breathing = self.pattern_detector.detect_periodic_breathing(
            breath_timestamps, tidal_volumes, respiratory_rates
        )

        pb_episodes_list = self.pattern_detector.detect_periodic_breathing_episodes(
            breath_timestamps,
            tidal_volumes,
            respiratory_rates,
            window_minutes=PDC.CSR_WINDOW_MINUTES,
            step_minutes=PDC.CSR_WINDOW_STEP_MINUTES,
        )

        pulse_change_count = None
        pulse_change_index = None
        if inputs.pulse_timestamps is not None and inputs.pulse_values is not None:
            try:
                pulse_events = self.pulse_detector.detect(
                    inputs.pulse_timestamps, inputs.pulse_values
                )
                pulse_change_count = len(pulse_events)
                pulse_change_index = (
                    pulse_change_count / session_duration_hours
                    if session_duration_hours > 0
                    else 0.0
                )
                logger.info(
                    f"Pulse changes: {pulse_change_count} total, "
                    f"{pulse_change_index:.1f} per hour"
                )
            except Exception as e:
                logger.debug(f"Pulse detection failed: {e}")

        mode_results = {}
        for mode_name in inputs.modes:
            try:
                mode = get_mode(mode_name)
                mode_result = mode.detect_events(
                    breaths=breaths,
                    flow_data=(timestamps, flow_values),
                    sample_rate=sample_rate,
                    session_duration_hours=session_duration_hours,
                )
                mode_results[mode_name] = mode_result
                logger.info(
                    f"Mode '{mode_name}': Detected {len(mode_result.apneas)} apneas, "
                    f"{len(mode_result.hypopneas)} hypopneas, AHI={mode_result.ahi:.1f}"
                )
            except Exception as e:
                logger.error(f"Failed to run mode '{mode_name}': {e}")
                continue

        machine_ahi = _machine_ahi(inputs.machine_events, session_duration_hours)

        summary = AnalysisResult(
            session_id=session_id,
            session_duration_hours=session_duration_hours,
            total_breaths=len(breaths),
            machine_events=inputs.machine_events,
            machine_ahi=machine_ahi,
            machine_rdi=machine_ahi,
            mode_results=mode_results,
            flow_analysis=flow_analysis.model_dump(),
            csr_detection=csr_detection.model_dump() if csr_detection else None,
            periodic_breathing=periodic_breathing.model_dump()
            if periodic_breathing
            else None,
            csr_episodes=[ep.model_dump() for ep in csr_episodes_list],
            periodic_breathing_episodes=[ep.model_dump() for ep in pb_episodes_list],
            pulse_change_count=pulse_change_count,
            pulse_change_index=pulse_change_index,
            timestamp_start=float(timestamps[0]) if len(timestamps) > 0 else 0.0,
            timestamp_end=float(timestamps[-1]) if len(timestamps) > 0 else 0.0,
        )

        # Build per-breath ComputedBreath objects.
        # Recovery breaths are sourced from the primary mode's RERA detector only.
        recovery_breath_indices = _collect_recovery_breath_indices(
            breaths, mode_results.get(primary_mode)
        )

        computed_breaths = _build_computed_breaths(
            breaths=breaths,
            flow_pattern_by_number=flow_pattern_by_number,
            breath_shape_by_number=breath_shape_by_number,
            insp_flow_by_number=insp_flow_by_number,
            recovery_breath_indices=recovery_breath_indices,
            leak_timestamps=inputs.leak_timestamps,
            leak_values=inputs.leak_values,
            device_manufacturer=inputs.device_manufacturer,
            mask_on_segments=inputs.mask_on_segments,
            session_duration_s=inputs.session_duration_s,
            ramp_enabled=inputs.ramp_enabled,
            ramp_time_minutes=inputs.ramp_time_minutes,
            smart_ramp=inputs.smart_ramp,
        )

        return AnalysisComputation(
            summary=summary, breaths=computed_breaths, primary_mode=primary_mode
        )

    async def analyze_session(
        self,
        session_id: int,
        modes: list[str] | None = None,
        primary_mode: str | None = None,
        store_results: bool = True,
    ) -> AnalysisResult:
        """Analyze session with specified detection mode(s).

        The production entry point is ``analysis.runner.run_one``, which owns a
        short read scope closed before compute and dispatches per policy; this
        method runs all three phases on the injected session (same-transaction
        write path relied on by tests and in-process callers).

        Uses the same three-phase pipeline as batch analysis:

        1. **I/O phase** (``load_session_inputs_raw``): DB reads only — raw blobs
           and scalar metadata.  ORM session not needed after this returns.
        2. **Compute phase** (``prepare_inputs`` + ``compute_analysis``): NumPy /
           scipy work on detached data.  No session held.
        3. **Write phase** (``store_result``): short INSERT via the injected session.

        Args:
            session_id: Database session ID
            modes: Detection modes to run (None = default mode)
            primary_mode: Mode whose recovery markers are persisted.  Must be a
                member of ``modes`` when supplied; defaults to DEFAULT_MODE when
                DEFAULT_MODE is in modes; required otherwise (ValueError).
            store_results: Whether to persist results

        Returns:
            AnalysisResult with results from all modes

        Raises:
            ValueError: If session not found or has no waveform data
        """
        modes_list = list(modes) if modes is not None else [DEFAULT_MODE]
        logger.info(
            f"Starting analysis for session {session_id} with modes: {modes_list}"
        )
        start_time = time.time()

        # I/O phase: fetch raw blobs while the session is held.
        raw = await self.load_session_inputs_raw(
            session_id, modes=modes_list, primary_mode=primary_mode
        )
        # Compute phase: deserialization + NumPy work — no session needed.
        inputs = AnalysisService.prepare_inputs(raw)
        computation = self.compute_analysis(inputs)

        processing_time_ms = int((time.time() - start_time) * 1000)
        logger.info(f"Analysis complete in {processing_time_ms}ms")

        if store_results:
            await self.store_result(computation, processing_time_ms)

        return computation.summary

    async def get_analysis_result(self, session_id: int) -> AnalysisResult | None:
        """
        Retrieve stored analysis result for a session.

        Scoped to this profile — returns None for foreign session IDs (treats
        "not owned" and "not analyzed" identically to avoid oracle attacks).

        Args:
            session_id: Database session ID

        Returns:
            AnalysisResult dataclass or None if not found
        """
        assert self.db_session is not None, "get_analysis_result requires a DB session"
        # Validate ownership before fetching analysis rows.
        if self.profile_id is not None:
            owned = (
                await self.db_session.execute(
                    select(models.Session.id)
                    .join(models.Device, models.Session.device_id == models.Device.id)
                    .where(
                        models.Session.id == session_id,
                        models.Device.profile_id == self.profile_id,
                    )
                )
            ).scalar_one_or_none()
            if owned is None:
                return None

        from snore.analysis.queries import latest_analysis_row  # noqa: PLC0415

        analysis = await latest_analysis_row(self.db_session, session_id)

        if not analysis:
            return None

        return AnalysisResult.from_stored_json(analysis.programmatic_result_json)

    async def store_result(
        self, computation: AnalysisComputation, processing_time_ms: int
    ) -> int:
        """Store analysis result + breath children to database atomically.

        Public write seam — callers (including ``BatchAnalysisCoordinator``) use
        this method so the write phase is not tied to a private implementation detail.

        Enforces profile ownership: raises ``NotFoundError`` when the target
        session does not belong to ``self.profile_id``, preventing cross-profile
        writes.

        Persistence contract:
        - ``AnalysisResult`` parent row is added and flushed to assign its PK.
        - ``Breath`` children are added with the flushed parent ID in one transaction.
        - ``programmatic_result_json`` contains only the public ``AnalysisResult``
          summary — breaths are NEVER stored there (they exist only as ``Breath``
          rows; no duplication).
        - ``engine_versions_json`` uses the nested
          ``{"identity": ..., "run": ...}`` shape — static algorithm identity
          kept separate from per-run metadata.

        Args:
            computation: AnalysisComputation envelope (summary + breaths).
            processing_time_ms: Processing time in milliseconds.

        Returns:
            Database analysis result ID.

        Raises:
            NotFoundError: If the session does not belong to this profile.
        """
        from snore.exceptions import NotFoundError as _NotFoundError  # noqa: PLC0415

        assert self.db_session is not None, "store_result requires a DB session"

        result = computation.summary

        # Verify the target session is owned by this profile before writing.
        owned = (
            await self.db_session.execute(
                select(models.Session.id)
                .join(models.Device, models.Session.device_id == models.Device.id)
                .where(
                    models.Session.id == result.session_id,
                    models.Device.profile_id == self.profile_id,
                )
            )
        ).scalar_one_or_none()
        if owned is None:
            raise _NotFoundError(
                f"Session {result.session_id} not found or not owned by profile {self.profile_id}"
            )

        # Build the nested engine_versions_json ({"identity": ..., "run": ...}).
        mode_keys = list(result.mode_results.keys())
        identity = AlgorithmIdentity.current()
        run_metadata = AnalysisRunMetadata(
            primary_mode=computation.primary_mode,
            modes=mode_keys,
        )
        algo_versions = AlgoVersions(identity=identity, run=run_metadata)

        analysis = models.AnalysisResult(
            session_id=result.session_id,
            timestamp_start=datetime.fromtimestamp(
                result.timestamp_start, tz=UTC
            ).replace(tzinfo=None),
            timestamp_end=datetime.fromtimestamp(result.timestamp_end, tz=UTC).replace(
                tzinfo=None
            ),
            programmatic_result_json=result.model_dump(),
            processing_time_ms=processing_time_ms,
            engine_versions_json=algo_versions.model_dump(),
        )

        self.db_session.add(analysis)
        # Flush to assign analysis.id so Breath children can reference it.
        await self.db_session.flush()

        # Persist breath children atomically in the same transaction.
        if computation.breaths:
            breath_rows = [
                models.Breath(
                    analysis_result_id=analysis.id,
                    session_id=result.session_id,
                    breath_number=cb.breath_number,
                    start_offset_s=cb.start_offset_s,
                    end_offset_s=cb.end_offset_s,
                    inspiration_time_s=cb.inspiration_time_s,
                    expiration_time_s=cb.expiration_time_s,
                    total_time_s=cb.total_time_s,
                    i_e_ratio=cb.i_e_ratio,
                    duty_cycle=cb.duty_cycle,
                    peak_flow_lpm=cb.peak_flow_lpm,
                    peak_exp_flow_lpm=cb.peak_exp_flow_lpm,
                    tidal_volume_ml=cb.tidal_volume_ml,
                    respiratory_rate_rolling=cb.respiratory_rate_rolling,
                    flatness_index=cb.flatness_index,
                    mid_insp_flattening=cb.mid_insp_flattening,
                    flow_class=cb.flow_class,
                    flow_confidence=cb.flow_confidence,
                    is_recovery_breath=cb.is_recovery_breath,
                    inferred_trigger_type=cb.inferred_trigger_type,
                    trigger_confidence=cb.trigger_confidence,
                    inferred_cycle_type=cb.inferred_cycle_type,
                    cycle_confidence=cb.cycle_confidence,
                    trigger_cycle_applicable=cb.trigger_cycle_applicable,
                    trigger_cycle_reason=cb.trigger_cycle_reason,
                    leak_valid=cb.leak_valid,
                    leak_valid_reason=cb.leak_valid_reason,
                    ramp_active=cb.ramp_active,
                    ramp_active_reason=cb.ramp_active_reason,
                    mask_off=cb.mask_off,
                    mask_off_reason=cb.mask_off_reason,
                )
                for cb in computation.breaths
            ]
            try:
                self.db_session.add_all(breath_rows)
                await self.db_session.flush()
            except Exception as exc:
                exc_msg = str(exc).lower()
                if "no such table" in exc_msg and "breath" in exc_msg:
                    raise RuntimeError(
                        "The 'breaths' table is missing from the database. "
                        "This happens when connecting to a database created before "
                        "breath-feature support was added. Drop the database and "
                        "re-import to populate it with the current schema."
                    ) from exc
                raise

        logger.info(
            f"Stored analysis result {analysis.id} with {len(computation.breaths)} breath rows"
        )
        return analysis.id


# ---------------------------------------------------------------------------
# Module-level compute helpers (private to this module)
# ---------------------------------------------------------------------------


def _parse_mask_on_segments(
    raw: Any, session_id: int
) -> list[tuple[float, float]] | None:
    """Validate and parse the persisted ``sessions.mask_on_segments`` JSON value.

    The column stores ascending, non-overlapping ``[start, end)`` intervals in
    session offset seconds as a JSON list of 2-element number lists.  The value
    is untrusted at this boundary (written by older code versions, or edited
    outside the app), so it is validated instead of consumed blindly:

    - the value must be a list whose items are each a 2-element sequence of
      real numbers (bools rejected), and
    - intervals must have positive measure (start < end) and be ascending and
      non-overlapping (each start >= the previous end).

    On any violation a warning is logged and None is returned, so downstream
    validity flags degrade to ``segments_unknown`` / ``not_available`` instead
    of crashing the analysis run.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        logger.warning(
            f"Session {session_id}: mask_on_segments is not a list "
            f"({type(raw).__name__}) — treating segments as unknown"
        )
        return None

    parsed: list[tuple[float, float]] = []
    previous_end = float("-inf")
    for seg in raw:
        if (
            not isinstance(seg, (list, tuple))
            or len(seg) != 2
            or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in seg
            )
        ):
            logger.warning(
                f"Session {session_id}: malformed mask_on_segments item {seg!r} "
                "(expected a 2-sequence of numbers) — treating segments as unknown"
            )
            return None
        start, end = float(seg[0]), float(seg[1])
        if not start < end or start < previous_end:
            logger.warning(
                f"Session {session_id}: mask_on_segments interval "
                f"[{start}, {end}) is empty, out of order, or overlaps the "
                "previous interval — treating segments as unknown"
            )
            return None
        parsed.append((start, end))
        previous_end = end
    return parsed


def _collect_recovery_breath_indices(
    breaths: list[Any],
    primary_mode_result: Any | None,
) -> set[int]:
    """Return breath_numbers that are recovery breaths in the primary mode's RERAs.

    A recovery breath ends a RERA — it is the last breath in the sequence,
    identified by end_time proximity to the RERA's end_time.
    """
    if primary_mode_result is None:
        return set()
    recovery_numbers: set[int] = set()
    for rera in getattr(primary_mode_result, "reras", []):
        for b in breaths:
            if abs(b.end_time - rera.end_time) < 0.5:
                recovery_numbers.add(b.breath_number)
    return recovery_numbers


def _build_computed_breaths(
    *,
    breaths: list[Any],
    flow_pattern_by_number: dict[int, Any],
    breath_shape_by_number: dict[int, Any],
    insp_flow_by_number: dict[int, np.ndarray],
    recovery_breath_indices: set[int],
    leak_timestamps: Any | None,
    leak_values: Any | None,
    device_manufacturer: str = "ResMed",
    mask_on_segments: list[tuple[float, float]] | None = None,
    session_duration_s: float | None = None,
    ramp_enabled: bool | None = None,
    ramp_time_minutes: int | None = None,
    smart_ramp: bool = False,
) -> list[ComputedBreath]:
    """Build the list of ComputedBreath for one session's analysis.

    Args:
        breaths: List of BreathMetrics from the segmenter.
        flow_pattern_by_number: Maps breath_number → FlowPattern (from classifier).
        breath_shape_by_number: Pre-computed ShapeFeatures keyed by breath_number.
            Only present for breaths where len(insp_flow) > 10; absent entries
            produce None flatness_index (same as the prior per-breath extraction).
        insp_flow_by_number: Pre-extracted inspiratory flow keyed by breath_number,
            memoized by compute_analysis so this loop skips re-extraction.
        recovery_breath_indices: Set of breath_numbers identified as recovery breaths.
        leak_timestamps: Leak waveform timestamps (may be None).
        leak_values: Leak waveform values (may be None).
        device_manufacturer: Manufacturer string from the Device row.  Trigger/cycle
            inference is only validated on ResMed devices; other vendors receive
            ``vendor_applicability="unvalidated_device"``.
        mask_on_segments: Persisted [start, end) mask-on intervals in session
            offset seconds (None = unknown).
        session_duration_s: Session duration in seconds (None = unknown).
        ramp_enabled: Parsed ``ramp_enabled`` setting (None = not recorded).
        ramp_time_minutes: Parsed ``ramp_time`` setting in minutes.
        smart_ramp: Parsed ``smart_ramp`` setting (SmartRamp active).

    Returns:
        list of ComputedBreath (same length as breaths).
    """
    from snore.analysis.shared.feature_extractors import (  # noqa: PLC0415
        compute_mid_insp_flattening,
    )
    from snore.analysis.shared.trigger_cycle import (  # noqa: PLC0415
        APPLICABILITY_UNVALIDATED_DEVICE,
        APPLICABILITY_VALIDATED,
        infer_trigger_cycle,
    )

    # Gate vendor-specific heuristic: trigger/cycle inference is tuned on ResMed
    # flow waveforms only.  Other vendors get confidence=null + unvalidated_device.
    tc_vendor_applicability = (
        APPLICABILITY_VALIDATED
        if device_manufacturer == "ResMed"
        else APPLICABILITY_UNVALIDATED_DEVICE
    )

    # Precompute per-session structures once, outside the ~10k-breath loop:
    # the mask-off gap list and the mask-on segment starts for the ramp clock.
    mask_off_gaps = _mask_off_gaps(mask_on_segments, session_duration_s)
    ramp_segment_starts = (
        [start for start, _end in mask_on_segments]
        if mask_on_segments is not None
        else None
    )

    computed: list[ComputedBreath] = []
    for idx, breath in enumerate(breaths):
        # Inspiratory flow (for mid-insp flattening and trigger inference) was
        # extracted once in compute_analysis; reuse the memoized slice.
        insp_flow: np.ndarray = insp_flow_by_number[breath.breath_number]

        flatness_idx: float | None = None
        mid_insp: float | None = None
        if len(insp_flow) > 10:
            shape = breath_shape_by_number.get(breath.breath_number)
            flatness_idx = shape.flatness_index if shape is not None else None
            mid_insp = compute_mid_insp_flattening(insp_flow)

        fl = flow_pattern_by_number.get(breath.breath_number)
        flow_cls = fl.flow_class if fl is not None else None
        flow_conf = fl.confidence if fl is not None else None

        gap_before: float | None = None
        if idx > 0:
            gap_before = breath.start_time - breaths[idx - 1].end_time

        insp_arr = insp_flow if len(insp_flow) > 0 else None
        tc = infer_trigger_cycle(
            inspiration_time_s=breath.inspiration_time,
            gap_before_s=gap_before,
            insp_flow_array=insp_arr,
            vendor_applicability=tc_vendor_applicability,
        )

        leak_valid, leak_valid_reason = _compute_leak_valid(
            breath_start=breath.start_time,
            breath_end=breath.end_time,
            leak_timestamps=leak_timestamps,
            leak_values=leak_values,
        )

        ramp_active, ramp_active_reason = _compute_ramp_active(
            breath_start_s=float(breath.start_time),
            segment_starts=ramp_segment_starts,
            ramp_enabled=ramp_enabled,
            ramp_time_minutes=ramp_time_minutes,
            smart_ramp=smart_ramp,
        )

        mask_off, mask_off_reason = _compute_mask_off(
            breath_start_s=float(breath.start_time),
            breath_end_s=float(breath.end_time),
            gaps=mask_off_gaps,
        )

        duty_cycle: float | None = None
        if (
            breath.duration is not None
            and breath.duration > 0
            and breath.inspiration_time is not None
        ):
            duty_cycle = float(breath.inspiration_time / breath.duration)

        computed.append(
            ComputedBreath(
                breath_number=breath.breath_number,
                start_offset_s=float(breath.start_time),
                end_offset_s=float(breath.end_time),
                inspiration_time_s=(
                    float(breath.inspiration_time)
                    if breath.inspiration_time is not None
                    else None
                ),
                expiration_time_s=(
                    float(breath.expiration_time)
                    if breath.expiration_time is not None
                    else None
                ),
                total_time_s=float(breath.duration),
                i_e_ratio=float(breath.i_e_ratio),
                duty_cycle=duty_cycle,
                peak_flow_lpm=float(breath.peak_inspiratory_flow),
                peak_exp_flow_lpm=float(breath.peak_expiratory_flow),
                tidal_volume_ml=float(breath.tidal_volume),
                respiratory_rate_rolling=float(breath.respiratory_rate_rolling),
                flatness_index=flatness_idx,
                mid_insp_flattening=mid_insp,
                flow_class=flow_cls,
                flow_confidence=flow_conf,
                is_recovery_breath=(breath.breath_number in recovery_breath_indices),
                inferred_trigger_type=tc.inferred_trigger_type,
                trigger_confidence=tc.trigger_confidence,
                inferred_cycle_type=tc.inferred_cycle_type,
                cycle_confidence=tc.cycle_confidence,
                trigger_cycle_applicable=tc.trigger_cycle_applicable,
                trigger_cycle_reason=tc.trigger_cycle_reason,
                leak_valid=leak_valid,
                leak_valid_reason=leak_valid_reason,
                ramp_active=ramp_active,
                ramp_active_reason=ramp_active_reason,
                mask_off=mask_off,
                mask_off_reason=mask_off_reason,
            )
        )
    return computed


def _compute_leak_valid(
    *,
    breath_start: float,
    breath_end: float,
    leak_timestamps: Any | None,
    leak_values: Any | None,
) -> tuple[bool | None, str | None]:
    """Derive the ``leak_valid`` quality flag for one breath.

    Logic (leak_valid derivation, v1):
    - No leak channel present  →  (None, "channel_absent")
    - Overlapping samples found →  mean of overlapping samples < threshold
    - No overlap, nearest neighbour ≤ 5 s away  →  that sample's value < threshold
    - No overlap, nearest neighbour > 5 s away  →  (None, "channel_unaligned")

    Returns:
        (leak_valid: bool | None, reason: str | None)
    """

    if leak_timestamps is None or leak_values is None or len(leak_timestamps) == 0:
        return None, "channel_absent"

    # Find overlapping samples: leak_timestamps within [breath_start, breath_end).
    mask = (leak_timestamps >= breath_start) & (leak_timestamps < breath_end)
    if mask.any():
        mean_leak = float(np.mean(leak_values[mask]))
        return mean_leak < LEAK_VALID_THRESHOLD_LPM, None

    # No overlap — nearest neighbour.
    breath_mid = (breath_start + breath_end) / 2.0
    dists = np.abs(leak_timestamps - breath_mid)
    nearest_idx = int(np.argmin(dists))
    nearest_dist = float(dists[nearest_idx])

    if nearest_dist <= LEAK_VALID_MAX_ALIGNMENT_GAP_S:
        leak_sample = float(leak_values[nearest_idx])
        return leak_sample < LEAK_VALID_THRESHOLD_LPM, None

    return None, "channel_unaligned"


def _parse_bool_setting(value: str | None) -> bool | None:
    """Parse a stored KV-settings boolean ("True"/"False"/"1"/"0", any case)."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("true", "1"):
        return True
    if normalized in ("false", "0"):
        return False
    return None


def _parse_int_setting(value: str | None) -> int | None:
    """Parse a stored KV-settings integer, tolerating float strings ("20.0")."""
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _compute_ramp_active(
    *,
    breath_start_s: float,
    segment_starts: list[float] | None,
    ramp_enabled: bool | None,
    ramp_time_minutes: int | None,
    smart_ramp: bool,
) -> tuple[bool | None, str | None]:
    """Derive the ``ramp_active`` validity flag for one breath.

    Settings-driven timed heuristic (VALIDITY_FLAGS_ALGO_VERSION v1):

    - ``ramp_enabled`` unknown          → (None, "not_available")
    - SmartRamp active                  → (None, "smart_ramp_indeterminate") —
      SmartRamp ends on sleep detection, not a timer; a timed heuristic would
      fabricate data.
    - Ramp disabled                     → (False, None)
    - ``ramp_time`` unknown             → (None, "not_available")
    - Else: the breath is in ramp iff its offset from the start of its
      CONTAINING mask-on segment (ResMed re-runs ramp on every mask-on
      restart) is below ``ramp_time_minutes * 60``.  When segments are
      unknown (``segment_starts`` is None), the offset is measured from
      session start (offset 0).

    ``segment_starts`` holds the ascending mask-on segment start offsets,
    precomputed once per session by ``_build_computed_breaths``.

    Gap-interior breaths (deliberate semantic): a breath whose start lies
    inside a mask-off gap has no containing segment, so its ramp offset is
    measured against the PRECEDING mask-on segment's start.  This is a seam
    artifact already flagged by ``mask_off=True``, and the slightly stale
    offset is intentional and harmless — consumers gate on ``mask_off`` first.

    Returns:
        (ramp_active: bool | None, reason: str | None)
    """
    if ramp_enabled is None:
        return None, "not_available"
    if smart_ramp:
        return None, "smart_ramp_indeterminate"
    if ramp_enabled is False:
        return False, None
    if ramp_time_minutes is None:
        return None, "not_available"

    segment_start = 0.0
    if segment_starts is not None:
        for start in segment_starts:
            if start <= breath_start_s:
                segment_start = start
            else:
                break
    offset_in_segment = breath_start_s - segment_start
    return offset_in_segment < ramp_time_minutes * 60, None


def _mask_off_gaps(
    mask_on_segments: list[tuple[float, float]] | None,
    session_duration_s: float | None,
) -> list[tuple[float, float]] | None:
    """Precompute the mask-off gap list for one session.

    Gaps are the complement of the persisted mask-on segments within
    ``[0, session_duration]`` — before, between, and after the segments.
    When ``session_duration_s`` is None the trailing gap is omitted (leading
    and inter-segment gaps are still detected).  Returns None when segments
    are unknown.

    Called once per session by ``_build_computed_breaths`` so the per-breath
    ``_compute_mask_off`` does not rebuild the list ~10k times.
    """
    if mask_on_segments is None:
        return None

    gaps: list[tuple[float, float]] = []
    previous_end = 0.0
    for start, end in mask_on_segments:
        if start > previous_end:
            gaps.append((previous_end, start))
        previous_end = max(previous_end, end)
    if session_duration_s is not None and session_duration_s > previous_end:
        gaps.append((previous_end, session_duration_s))
    return gaps


def _compute_mask_off(
    *,
    breath_start_s: float,
    breath_end_s: float,
    gaps: list[tuple[float, float]] | None,
) -> tuple[bool | None, str | None]:
    """Derive the ``mask_off`` validity flag for one breath.

    ``gaps`` is the per-session precomputed gap list from ``_mask_off_gaps``
    (None = segments unknown); the flag is True iff the breath interval
    overlaps any gap with positive measure (VALIDITY_FLAGS_ALGO_VERSION v1).

    Breaths cannot lie wholly inside a gap — there are no flow samples there
    (the merge concatenates arrays across a timestamp jump) — so True marks
    the rare seam-spanning artifact breath segmented across a mask-off gap.
    The dominant honest value is False, replacing the previous uniform null.

    Returns:
        (mask_off: bool | None, reason: str | None)
    """
    if gaps is None:
        return None, "segments_unknown"

    for gap_start, gap_end in gaps:
        overlap = min(breath_end_s, gap_end) - max(breath_start_s, gap_start)
        if overlap > 0:
            return True, None
    return False, None


def _compute_session_in_process(raw: RawSessionBlobs) -> AnalysisComputation:
    """Pure-compute phase suitable for a subprocess worker.

    Replicates the closure body from ``BatchAnalysisCoordinator._run_one``
    at module level so it can be pickled and dispatched to the shared
    ``ProcessPoolExecutor``.  No DB access — all inputs arrive as
    ``RawSessionBlobs`` (bytes + plain Python); all outputs are plain
    Python / NumPy (``AnalysisComputation`` is a picklable dataclass).

    Args:
        raw: DB-fetched blobs from the I/O phase.

    Returns:
        ``AnalysisComputation`` envelope ready for the write phase.
    """
    inputs = AnalysisService.prepare_inputs(raw)
    return AnalysisService().compute_analysis(inputs)
