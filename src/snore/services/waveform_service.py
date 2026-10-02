"""Waveform service for listing and loading waveform data."""

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from typing import Any

import numpy as np

from sqlalchemy import select

from snore.analysis.data.waveform_loader import deserialize_waveform_blob
from snore.database import models
from snore.exceptions import NotFoundError
from snore.services._base import ProfileScopedService, require_owned_session
from snore.services.lttb import lttb_downsample
from snore.services.schemas import (
    EventComparisonDetail,
    EventComparisonResult,
    WaveformInfo,
)

__all__ = ["WaveformService", "clear_waveform_array_cache"]

# Byte cap for the deserialized-array cache below.  A full night of 25 Hz flow
# is ~5.8 MB (float32 timestamps + values); 64 MB holds ~11 such channels.
# Exposed as a module constant so tests can monkeypatch it (read fresh on every
# insert, so a patched value takes effect immediately).
WAVEFORM_ARRAY_CACHE_MAX_BYTES = 64 * 1024 * 1024


class _WaveformArrayCache:
    """Module-level, byte-capped LRU of deserialized waveform arrays.

    Keyed by ``Waveform.id`` (the row primary key), which gives automatic
    staleness invalidation for free: row ids are never mutated in place, and a
    session delete + re-import produces a *new* row id.  (SQLite can reuse a
    rowid after deleting the max row — which is exactly why the key must be the
    id fetched fresh from the DB on every request, never a ``(session_id,
    type)`` pair.)

    Confinement: FastAPI serves requests on a single event loop, and no ``await``
    occurs between the ``OrderedDict`` operations below, so they execute
    atomically under the GIL.  No lock is required or used.
    """

    def __init__(self) -> None:
        self._entries: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._total_bytes = 0

    def get(self, waveform_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Return cached ``(timestamps, values)`` for a row id, or ``None``."""
        entry = self._entries.get(waveform_id)
        if entry is not None:
            self._entries.move_to_end(waveform_id)
        return entry

    def put(self, waveform_id: int, timestamps: np.ndarray, values: np.ndarray) -> None:
        """Insert arrays for a row id, evicting least-recently-used past the cap.

        ``get`` is called before ``put`` on the miss path, so a repeat key is not
        expected; the guard makes a double insert idempotent rather than
        double-counting bytes.  The just-inserted entry is never evicted, so a
        single oversized channel is still served (all others make way for it).
        """
        if waveform_id in self._entries:
            return
        self._entries[waveform_id] = (timestamps, values)
        self._total_bytes += timestamps.nbytes + values.nbytes
        while (
            self._total_bytes > WAVEFORM_ARRAY_CACHE_MAX_BYTES
            and len(self._entries) > 1
        ):
            _, (old_ts, old_vals) = self._entries.popitem(last=False)
            self._total_bytes -= old_ts.nbytes + old_vals.nbytes

    def clear(self) -> None:
        """Drop all entries (used to reset state between tests)."""
        self._entries.clear()
        self._total_bytes = 0


_waveform_array_cache = _WaveformArrayCache()


def clear_waveform_array_cache() -> None:
    """Drop every entry in the module-level deserialized-array cache.

    The cache is keyed by ``Waveform.id``, but that row id is a bare SQLite
    rowid alias (no ``sqlite_autoincrement``), so SQLite *reuses* an id after the
    max row is deleted or the table is emptied.  A reused id would otherwise
    serve the deleted row's arrays — even to a different profile, since the
    per-request ownership check validates session→profile, not cache-entry→row.
    Id-keying alone therefore cannot detect a delete→re-import; every code path
    that deletes waveform rows MUST call this so a reused id starts cold.
    """
    _waveform_array_cache.clear()


class WaveformService(ProfileScopedService):
    """Service for waveform listing and loading operations."""

    async def _assert_session_owned(self, session_id: int) -> None:
        """Raise NotFoundError if session_id doesn't belong to this profile."""
        await require_owned_session(self.db_session, self.profile_id, session_id)

    async def list_waveforms(self, session_id: int) -> list[WaveformInfo]:
        """
        List available waveform types for a session.

        Returns empty list if no waveforms found.

        Args:
            session_id: Database session ID

        Returns:
            List of WaveformInfo objects with metadata
        """
        await self._assert_session_owned(session_id)
        waveforms = (
            (
                await self.db_session.execute(
                    select(models.Waveform)
                    .where(models.Waveform.session_id == session_id)
                    .order_by(models.Waveform.waveform_type)
                )
            )
            .scalars()
            .all()
        )

        result = []
        for wf in waveforms:
            sample_count = wf.sample_count or 0
            duration_seconds = (
                sample_count / wf.sample_rate if wf.sample_rate > 0 else 0
            )
            result.append(
                WaveformInfo(
                    waveform_type=wf.waveform_type,
                    sample_rate=wf.sample_rate,
                    sample_count=sample_count,
                    unit=wf.unit,
                    duration_hours=duration_seconds / 3600,
                )
            )
        return result

    async def _fetch_waveform_metadata(
        self, session_id: int, waveform_type: str
    ) -> dict[str, Any]:
        """Light query for a Waveform row's scalar metadata, WITHOUT the blob.

        Selects only scalar columns so warm-cache requests never page the ~MB
        ``data_blob``.  The returned dict mirrors ``fetch_waveform_blob``'s
        ``metadata_scalars`` key-for-key (including ``waveform_id`` and
        ``sample_count``), keeping :meth:`get_waveform_data`'s return contract
        identical whether arrays come from the cache or a fresh deserialize.

        Raises:
            ValueError: If the waveform row is not found.
        """
        row = (
            await self.db_session.execute(
                select(
                    models.Waveform.id,
                    models.Waveform.session_id,
                    models.Waveform.waveform_type,
                    models.Waveform.sample_rate,
                    models.Waveform.unit,
                    models.Waveform.min_value,
                    models.Waveform.max_value,
                    models.Waveform.mean_value,
                    models.Waveform.sample_count,
                ).filter_by(session_id=session_id, waveform_type=waveform_type)
            )
        ).first()

        if row is None:
            raise ValueError(
                f"Waveform not found: session_id={session_id}, type={waveform_type}"
            )

        (
            waveform_id,
            wf_session_id,
            wf_type,
            sample_rate,
            unit,
            min_value,
            max_value,
            mean_value,
            sample_count,
        ) = row
        metadata = {
            "waveform_id": waveform_id,
            "session_id": wf_session_id,
            "waveform_type": wf_type,
            "sample_rate": sample_rate,
            "unit": unit,
            "min_value": min_value,
            "max_value": max_value,
            "mean_value": mean_value,
            "sample_count": sample_count,
        }
        return metadata

    async def get_waveform_data(
        self,
        session_id: int,
        waveform_type: str,
        max_points: int | None = None,
        start_seconds: float | None = None,
        end_seconds: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        Load waveform data with optional windowing and LTTB downsampling.

        Structured as two explicit phases (§7 I/O–compute split):

        1. **I/O phase**: ``_assert_session_owned`` (ownership check, always
           first — the cache never bypasses it) then a light metadata-only query.
           On a cache miss this phase also reads the ``data_blob`` by row id (the
           metadata query already resolved it).  The injected session is used
           only here.
        2. **Compute phase** (``deserialize_waveform_blob``): converts raw bytes to
           numpy arrays.  No DB session access occurs here.

        Deserialized arrays are cached module-level by row id (see
        :class:`_WaveformArrayCache`), so a warm request skips both the ~MB blob
        fetch and the deserialize.  Metadata always comes from the fresh light
        query, so unit/sample_rate/etc. stay current.  Cached arrays are marked
        read-only; the windowing slice below is a read-only view of them, and the
        no-window/no-downsample path returns the read-only arrays directly (the
        router only reads them via ``.tolist()``; LTTB allocates its own output).

        Args:
            session_id: Database session ID
            waveform_type: Type of waveform to load
            max_points: If set, downsample to this many points using LTTB
            start_seconds: If set, filter timestamps >= this value
            end_seconds: If set, filter timestamps <= this value

        Returns:
            Tuple of (timestamps, values, metadata)

        Raises:
            ValueError: If waveform not found
        """
        # --- I/O phase: DB access only ---
        # Ownership check stays FIRST: the cache must never serve a row to a
        # profile that does not own its session.
        await self._assert_session_owned(session_id)
        try:
            metadata = await self._fetch_waveform_metadata(session_id, waveform_type)
        except ValueError as e:
            raise NotFoundError(str(e)) from e

        # --- Compute phase: no DB session needed unless the blob must be read ---
        # Do NOT close self.db_session here: the caller may hold it open for
        # subsequent queries (e.g. loading analysis overlays in waveform show).
        waveform_id = metadata["waveform_id"]
        cached = _waveform_array_cache.get(waveform_id)
        if cached is None:
            # Fetch only the blob, keyed by the id the metadata query already
            # resolved — no second scalar-column round trip.  The row can vanish
            # between the two queries (concurrent delete); surface that as the
            # same 404 as a metadata miss above.
            blob_row = (
                await self.db_session.execute(
                    select(models.Waveform.data_blob).where(
                        models.Waveform.id == waveform_id
                    )
                )
            ).first()
            if blob_row is None:
                raise NotFoundError(
                    f"Waveform not found: session_id={session_id}, type={waveform_type}"
                )
            timestamps, values = deserialize_waveform_blob(
                blob_row[0], metadata["sample_count"] or 0
            )
            # Freeze so a caller (or a windowing view, which shares this buffer)
            # can never mutate the shared cached arrays.
            timestamps.flags.writeable = False
            values.flags.writeable = False
            _waveform_array_cache.put(waveform_id, timestamps, values)
        else:
            timestamps, values = cached

        if start_seconds is not None or end_seconds is not None:
            # Timestamps are ascending, so bound the window with binary search
            # instead of a full-length boolean mask.  ``left``/``right`` reproduce
            # the old inclusive mask exactly: lo is the first ts >= start, hi is
            # one past the last ts <= end.  Both bounds one-sided or absent (0 /
            # len) and empty results (lo >= hi) fall out naturally.
            lo = (
                int(np.searchsorted(timestamps, start_seconds, side="left"))
                if start_seconds is not None
                else 0
            )
            hi = (
                int(np.searchsorted(timestamps, end_seconds, side="right"))
                if end_seconds is not None
                else len(timestamps)
            )
            timestamps = timestamps[lo:hi]
            values = values[lo:hi]

        if max_points and len(timestamps) > max_points:
            timestamps, values = lttb_downsample(timestamps, values, max_points)

        return timestamps, values, metadata

    async def _load_analysis_result(self, session_id: int) -> Any:
        """Load the latest AnalysisResult row for a session, returning a validated result object."""
        from sqlalchemy import select as _select

        from snore.analysis.types import AnalysisResult as _AnalysisResult
        from snore.database import models as _models

        analysis_row = (
            (
                await self.db_session.execute(
                    _select(_models.AnalysisResult)
                    .filter_by(session_id=session_id)
                    .order_by(_models.AnalysisResult.created_at.desc())
                )
            )
            .scalars()
            .first()
        )

        if analysis_row is None:
            return None
        return _AnalysisResult.from_stored_json(analysis_row.programmatic_result_json)

    async def compare_events(
        self,
        session_id: int,
        mode: str = "aasm",
        tolerance_seconds: float = 5.0,
    ) -> EventComparisonResult:
        """
        Compare machine vs programmatic events for a session.

        Args:
            session_id: Database session ID
            mode: Detection mode to compare
            tolerance_seconds: Time tolerance in seconds for matching events

        Returns:
            EventComparisonResult with false negatives and false positives

        Raises:
            NotFoundError: If no analysis result found or mode not available
        """
        from snore.analysis.utils import convert_machine_events

        await self._assert_session_owned(session_id)
        result = await self._load_analysis_result(session_id)

        if result is None:
            raise NotFoundError("No analysis results found for this session")

        if mode not in result.mode_results:
            raise NotFoundError(f"Mode '{mode}' not found in analysis results")

        mode_result = result.mode_results[mode]

        machine_events_raw = result.machine_events or []
        machine_apneas, machine_hypopneas = convert_machine_events(machine_events_raw)
        all_machine = machine_apneas + machine_hypopneas

        prog_apneas = list(mode_result.apneas)
        prog_hypopneas = list(mode_result.hypopneas)
        all_prog = prog_apneas + prog_hypopneas

        false_negatives: list[EventComparisonDetail] = []
        false_positives_apnea: list[EventComparisonDetail] = []
        false_positives_hypopnea: list[EventComparisonDetail] = []

        all_machine_sorted = sorted(all_machine, key=lambda e: e.start_time)
        prog_times = sorted(e.start_time for e in all_prog)
        machine_times = [e.start_time for e in all_machine_sorted]

        for m_event in all_machine_sorted:
            lo = bisect_left(prog_times, m_event.start_time - tolerance_seconds)
            hi = bisect_right(prog_times, m_event.start_time + tolerance_seconds)
            if lo >= hi:
                false_negatives.append(
                    EventComparisonDetail(
                        event_type=getattr(m_event, "event_type", "unknown"),
                        start_time=m_event.start_time,
                        duration=getattr(m_event, "duration", 0.0),
                        source="machine",
                        confidence=None,
                        flow_reduction=None,
                    )
                )

        for apnea_event in prog_apneas:
            lo = bisect_left(machine_times, apnea_event.start_time - tolerance_seconds)
            hi = bisect_right(machine_times, apnea_event.start_time + tolerance_seconds)
            if lo >= hi:
                false_positives_apnea.append(
                    EventComparisonDetail(
                        event_type=apnea_event.event_type,
                        start_time=apnea_event.start_time,
                        duration=apnea_event.duration,
                        source="programmatic",
                        confidence=getattr(apnea_event, "confidence", None),
                        flow_reduction=getattr(apnea_event, "flow_reduction", None),
                    )
                )

        for hypopnea_event in prog_hypopneas:
            lo = bisect_left(
                machine_times, hypopnea_event.start_time - tolerance_seconds
            )
            hi = bisect_right(
                machine_times, hypopnea_event.start_time + tolerance_seconds
            )
            if lo >= hi:
                false_positives_hypopnea.append(
                    EventComparisonDetail(
                        event_type="H",
                        start_time=hypopnea_event.start_time,
                        duration=hypopnea_event.duration,
                        source="programmatic",
                        confidence=hypopnea_event.confidence,
                        flow_reduction=hypopnea_event.flow_reduction,
                    )
                )

        return EventComparisonResult(
            session_id=session_id,
            mode=mode,
            machine_event_count=len(machine_events_raw),
            programmatic_event_count=len(prog_apneas) + len(prog_hypopneas),
            false_negatives=false_negatives,
            false_positives_apnea=false_positives_apnea,
            false_positives_hypopnea=false_positives_hypopnea,
        )
