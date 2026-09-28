"""Shared service core: DB session state, range resolution, analysis classification."""

from __future__ import annotations

import json

from datetime import date
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.data.waveform_loader import deserialize_waveform_blob
from snore.analysis.queries import latest_analysis_ids, latest_analysis_row
from snore.analysis.shared.versioning import (
    AlgorithmIdentity,
    AlgoVersions,
    AnalysisStatus,
    DayAnalysisStatus,
    TimezoneStatus,
)
from snore.database import models
from snore.utils.db_chunk import iter_id_chunks

from .dtos import (
    DeviceAmbiguityError,
    DeviceNotOwnedError,
    NoSessionsInRangeError,
    SessionCoverage,
)

# JSON path of the persisted PB episode list inside programmatic_result_json.
_PB_PATH = "$.periodic_breathing_episodes"

# (status, algo_versions, analysis_result_id) for a session's latest run.
SessionClassification = tuple[AnalysisStatus, AlgoVersions | None, int | None]


def _decode_json_extract(value: object) -> object:
    """Decode a SQLite ``json_extract`` result: arrays/objects come back as JSON
    text, scalars as native values, a missing key or JSON null as ``None``.
    Undecodable text is returned unchanged (callers treat it as malformed)."""
    if not isinstance(value, str):
        return value
    try:
        decoded: object = json.loads(value)
    except ValueError:
        return value
    return decoded


async def _resolve_timezone(
    db: AsyncSession, profile_id: int
) -> tuple[TimezoneStatus, str | None]:
    """Resolve the profile's user-declared timezone (A6 labeling metadata).

    Returns ``(USER_DECLARED, iana_name)`` when the profile declares a
    timezone, else ``(UNKNOWN, None)``.  Timestamps are never rewritten and
    no UTC offset is ever fabricated — this labels interpretation only.
    """
    tz_name = (
        await db.execute(
            select(models.Profile.timezone).where(models.Profile.id == profile_id)
        )
    ).scalar_one_or_none()
    if tz_name:
        return TimezoneStatus.USER_DECLARED, tz_name
    return TimezoneStatus.UNKNOWN, None


class _BreathServiceCore:
    """Shared state and helpers for all BreathService method groups."""

    def __init__(self, db_session: AsyncSession, profile_id: int) -> None:
        self._db = db_session
        self._profile_id = profile_id
        self._tz_cache: tuple[TimezoneStatus, str | None] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def resolve_timezone(self) -> tuple[TimezoneStatus, str | None]:
        """Profile timezone label (A6), cached per service instance."""
        if self._tz_cache is None:
            self._tz_cache = await _resolve_timezone(self._db, self._profile_id)
        return self._tz_cache

    async def _latest_analysis_for_session(
        self, session_id: int
    ) -> SessionClassification:
        """Return (status, algo_versions, analysis_result_id) for latest run.

        Ownership is assumed: callers are responsible for verifying the
        session belongs to ``self._profile_id`` via ``_resolve_range``
        or an explicit profile-scoped query before calling this helper.

        Returns (NOT_RUN, None, None) when no run exists.
        Returns (STALE_VERSION, algo|None, id) when engine_versions_json is stale.
        """
        row = await latest_analysis_row(self._db, session_id)
        if row is None:
            return AnalysisStatus.NOT_RUN, None, None
        status, algo = self._classify_analysis_row(row)
        return status, algo, row.id

    @staticmethod
    def _classify_analysis_row(
        row: models.AnalysisResult,
    ) -> tuple[AnalysisStatus, AlgoVersions | None]:
        """Classify an AnalysisResult ORM row: (status, algo|None)."""
        return _BreathServiceCore._classify_engine_versions(row.engine_versions_json)

    @staticmethod
    def _classify_engine_versions(
        engine_versions_json: dict[str, Any],
    ) -> tuple[AnalysisStatus, AlgoVersions | None]:
        """Classify a stored ``engine_versions_json`` payload: (status, algo|None).

        Shared by the single-row and bulk (column-limited) lookups so both
        classify identically.
        """
        from snore.services.breath_service import BreathService  # noqa: PLC0415

        algo = AlgoVersions.from_stored(engine_versions_json)
        if algo is None:
            return AnalysisStatus.STALE_VERSION, None
        current = BreathService._current_algorithm_identity()
        if algo.identity.model_dump() != current.model_dump():
            return AnalysisStatus.STALE_VERSION, algo
        return AnalysisStatus.OK, algo

    async def _latest_analysis_bulk(
        self, session_ids: list[int]
    ) -> tuple[dict[int, SessionClassification], dict[int, object]]:
        """Bulk variant of ``_latest_analysis_for_session``.

        Returns ``(classification, pb_episodes)``:

        - ``classification``: session_id → ``SessionClassification`` for every
          id in ``session_ids``; sessions without a run map to
          ``(NOT_RUN, None, None)``.
        - ``pb_episodes``: session_id → the latest run's persisted
          ``periodic_breathing_episodes`` (decoded JSON; ``[]`` for an explicit
          JSON null; ``None`` when the key is absent, i.e. PB never ran), for
          sessions with a run.

        Column-limited: the (large) result JSON is never loaded — SQLite's
        ``json_extract`` pulls only the episode list.  One window-function
        query resolves the latest AnalysisResult ids, then one chunked query
        loads their columns.  Ownership is assumed, exactly as in
        ``_latest_analysis_for_session``.
        """
        classification: dict[int, SessionClassification] = {
            sid: (AnalysisStatus.NOT_RUN, None, None) for sid in session_ids
        }
        pb_episodes: dict[int, object] = {}
        ar_id_by_session = await latest_analysis_ids(self._db, session_ids)
        ar = models.AnalysisResult
        # Chunk the unbounded AnalysisResult-id IN-list (SQLite bound-param cap).
        for chunk in iter_id_chunks(list(ar_id_by_session.values())):
            rows = await self._db.execute(
                select(
                    ar.id,
                    ar.session_id,
                    ar.engine_versions_json,
                    func.json_extract(ar.programmatic_result_json, _PB_PATH).label(
                        "pb_episodes"
                    ),
                    func.json_type(ar.programmatic_result_json, _PB_PATH).label(
                        "pb_type"
                    ),
                ).where(ar.id.in_(chunk))
            )
            for row in rows:
                status, algo = self._classify_engine_versions(row.engine_versions_json)
                classification[row.session_id] = (status, algo, row.id)
                # A persisted result always carries the key; an explicit JSON
                # null means PB ran with no episodes, a missing key (e.g. an
                # empty result JSON) means it never ran.
                pb_episodes[row.session_id] = (
                    []
                    if row.pb_type == "null"
                    else _decode_json_extract(row.pb_episodes)
                )
        return classification, pb_episodes

    async def latest_analysis_for_session(
        self, session_id: int
    ) -> SessionClassification:
        """Supported public lookup for validation modules.

        Returns (status, algo_versions, analysis_result_id) for the latest run.
        Ownership is assumed: the session must belong to ``self._profile_id`` —
        callers are responsible for ensuring that via a profile-scoped query.

        Returns (NOT_RUN, None, None) when no run exists.
        """
        return await self._latest_analysis_for_session(session_id)

    # ------------------------------------------------------------------
    # Single range-aware resolver (replaces _resolve_device, _resolve_session_for_date,
    # and _fetch_day_sessions — all callers must use _resolve_range)
    # ------------------------------------------------------------------

    async def _resolve_range(
        self,
        date_start: date,
        date_end: date,
        device_id: int | None,
    ) -> tuple[int, dict[date, list[Any]]]:
        """Return (resolved_device_id, sessions_by_date) for [date_start, date_end].

        device_id given and owned by this profile:
            Validate ownership independent of data presence.
            Return (device_id, sessions_by_date) — sessions_by_date may be empty.
        device_id given and NOT owned:
            Raise DeviceNotOwnedError(device_id, profile_id).
        device_id None, 0 owned devices with sessions in range:
            Raise ValueError("No sessions found in range").
        device_id None, 1 distinct owned device in range:
            Auto-select it; return (device_id, sessions_by_date).
        device_id None, ≥2 distinct owned devices in range:
            Raise DeviceAmbiguityError with all owned device_ids.

        For a single-date point query call: _resolve_range(d, d, device_id).
        """
        if device_id is not None:
            # Validate ownership independent of data presence
            owned = (
                await self._db.execute(
                    select(models.Device.id).where(
                        models.Device.id == device_id,
                        models.Device.profile_id == self._profile_id,
                    )
                )
            ).scalar_one_or_none()
            if owned is None:
                raise DeviceNotOwnedError(
                    device_id=device_id, profile_id=self._profile_id
                )
            # Fetch sessions in range (ownership already verified above)
            stmt = (
                select(models.Session, models.Day)
                .join(models.Day, models.Session.day_id == models.Day.id)
                .where(
                    models.Session.device_id == device_id,
                    models.Day.date >= date_start,
                    models.Day.date <= date_end,
                )
                .order_by(models.Day.date, models.Session.start_time)
            )
            rows = (await self._db.execute(stmt)).all()
            sessions_by_date: dict[date, list[Any]] = {}
            for r in rows:
                d = r.Day.date
                if d not in sessions_by_date:
                    sessions_by_date[d] = []
                sessions_by_date[d].append(r.Session)
            return device_id, sessions_by_date

        # device_id is None — auto-select from owned sessions in range
        stmt = (
            select(models.Session, models.Day)
            .join(models.Day, models.Session.day_id == models.Day.id)
            .join(models.Device, models.Session.device_id == models.Device.id)
            .where(
                models.Day.date >= date_start,
                models.Day.date <= date_end,
                models.Device.profile_id == self._profile_id,
            )
            .order_by(models.Day.date, models.Session.start_time)
        )
        rows = (await self._db.execute(stmt)).all()
        if not rows:
            raise NoSessionsInRangeError(date_start, date_end)
        # Distinct device_ids, order-preserving
        device_ids_seen: list[int] = list(
            dict.fromkeys(r.Session.device_id for r in rows)
        )
        if len(device_ids_seen) > 1:
            serial_rows = (
                await self._db.execute(
                    select(models.Device.id, models.Device.serial_number).where(
                        models.Device.id.in_(device_ids_seen)
                    )
                )
            ).all()
            device_serials = {int(r[0]): (r[1] or "") for r in serial_rows}
            raise DeviceAmbiguityError(
                therapy_date=date_start,
                profile_id=self._profile_id,
                owned_device_ids=device_ids_seen,
                device_serials=device_serials,
            )
        resolved_device_id = device_ids_seen[0]
        sessions_by_date = {}
        for r in rows:
            d = r.Day.date
            if d not in sessions_by_date:
                sessions_by_date[d] = []
            sessions_by_date[d].append(r.Session)
        return resolved_device_id, sessions_by_date

    @staticmethod
    def _reduce_day_status(
        coverages: list[SessionCoverage],
        identities: list[AlgorithmIdentity],
    ) -> DayAnalysisStatus:
        """Reduce per-session coverage to a day-level DayAnalysisStatus.

        Precedence:
        1. Multiple distinct algorithm identities among OK sessions → MIXED_VERSION
        2. All OK → OK
        3. All NOT_RUN → NOT_RUN
        4. All STALE_VERSION → STALE
        5. Anything else (stale+not-run, ok+stale, ok+not-run, …) → PARTIAL
        """
        if not coverages:
            return DayAnalysisStatus.NOT_RUN

        # Rule 1: multiple distinct identities → MIXED_VERSION
        if len(identities) > 1:
            id_strs = {str(i.model_dump()) for i in identities}
            if len(id_strs) > 1:
                return DayAnalysisStatus.MIXED_VERSION

        statuses = {c.analysis_status for c in coverages}

        # Rule 2: all OK → OK
        if statuses == {AnalysisStatus.OK}:
            return DayAnalysisStatus.OK

        # Rule 3: all NOT_RUN → NOT_RUN
        if statuses == {AnalysisStatus.NOT_RUN}:
            return DayAnalysisStatus.NOT_RUN

        # Rule 4: all STALE_VERSION → STALE
        if statuses == {AnalysisStatus.STALE_VERSION}:
            return DayAnalysisStatus.STALE

        # Rule 5: any other mix → PARTIAL
        return DayAnalysisStatus.PARTIAL

    @staticmethod
    async def _fetch_waveform_channel_vals(
        db: AsyncSession,
        session_ids: list[int],
    ) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
        """Fetch and deserialize fl and snore waveform values for a set of sessions.

        Returns (fl_vals_by_session, snore_vals_by_session).  Each map goes
        session_id → list of raw float values including any negative sentinel values;
        callers are responsible for applying filters (e.g. the fl >= 0 guard in
        compare_epochs and _build_nightly_summary).  Snore zeros are legitimate
        data and are retained.  Sessions without a channel row are absent from the
        respective dict.

        This is pure I/O + light compute (numpy deserialization); no analysis
        state is consulted.
        """
        fl_by_sess: dict[int, list[float]] = {}
        snore_by_sess: dict[int, list[float]] = {}

        if not session_ids:
            return fl_by_sess, snore_by_sess

        # Chunk the unbounded session_id list (SQLite bound-param cap); the
        # waveform_type IN-list is a constant kept in every chunk.  Each session
        # falls in exactly one chunk, so the per-session dicts never collide.
        for chunk in iter_id_chunks(session_ids):
            wf_rows = (
                (
                    await db.execute(
                        select(models.Waveform).where(
                            models.Waveform.session_id.in_(chunk),
                            models.Waveform.waveform_type.in_(["fl", "snore"]),
                        )
                    )
                )
                .scalars()
                .all()
            )

            for wf in wf_rows:
                if not wf.data_blob or not wf.sample_count:
                    continue
                try:
                    _ts, vals = deserialize_waveform_blob(wf.data_blob, wf.sample_count)
                except ValueError:
                    continue
                sid = int(wf.session_id)
                if wf.waveform_type == "fl":
                    # Retain raw values including any negative sentinels.
                    # The caller (_build_nightly_summary) applies the >= 0 filter.
                    fl_by_sess[sid] = [float(v) for v in vals]
                elif wf.waveform_type == "snore":
                    # Zeros are legitimate snore data — retain all values.
                    snore_by_sess[sid] = [float(v) for v in vals]

        return fl_by_sess, snore_by_sess

    @staticmethod
    def _current_algorithm_identity() -> AlgorithmIdentity:
        """Current algorithm identity for STALE_VERSION detection."""
        return AlgorithmIdentity.current()
