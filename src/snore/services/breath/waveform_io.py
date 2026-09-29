"""Waveform-window fetch seams (DB-touching)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.analysis.shared.versioning import TimezoneStatus
from snore.database import models

from ._core import _BreathServiceCore, _resolve_timezone
from .dtos import (
    MultiSessionAmbiguityError,
    RawWaveformChannel,
    RawWaveformWindow,
    SessionSummary,
    WaveformChannelName,
    WaveformWindow,
    WaveformWindowRequest,
)


async def _fetch_waveform_blobs(
    db: AsyncSession,
    request: WaveformWindowRequest,
    session_id: int,
    session_start: datetime,
    timezone_status: TimezoneStatus = TimezoneStatus.UNKNOWN,
    timezone_name: str | None = None,
) -> RawWaveformWindow:
    """PRIVATE — fetch waveform blobs for a pre-resolved, already-owned session.

    Trusted internal helper: ownership has already been verified by the caller
    (via ``_resolve_range`` or ``fetch_waveform_window_raw``).  No ownership
    check or Session query is performed here.
    """

    requested_types = [ch.value for ch in request.channels]
    wf_stmt = select(models.Waveform).where(
        models.Waveform.session_id == session_id,
        models.Waveform.waveform_type.in_(requested_types),
    )
    wf_rows = (await db.execute(wf_stmt)).scalars().all()
    wf_by_type = {w.waveform_type: w for w in wf_rows}

    channels: list[RawWaveformChannel] = []
    missing: list[WaveformChannelName] = []
    for ch in request.channels:
        wf = wf_by_type.get(ch.value)
        if wf is None:
            missing.append(ch)
        else:
            channels.append(
                RawWaveformChannel(
                    waveform_type=ch,
                    unit=getattr(wf, "unit", None),
                    sample_rate=wf.sample_rate or 1.0,
                    sample_count=getattr(wf, "sample_count", 0),
                    raw_bytes=wf.data_blob or b"",
                )
            )

    return RawWaveformWindow(
        request=request,
        session_id=session_id,
        session_start_wall_clock=session_start,
        timezone_status=timezone_status,
        timezone_name=timezone_name,
        channels=channels,
        missing_channels=missing,
    )


async def fetch_waveform_window_raw(
    db: AsyncSession,
    profile_id: int,
    request: WaveformWindowRequest,
) -> RawWaveformWindow:
    """PUBLIC — fetch waveform blobs with profile-level ownership enforcement.

    Never closes db: the scope owner opens and closes the scope around this call.

    ``request.session_id`` must be set (direct callers must have a resolved session).
    Verifies ``Device.profile_id == profile_id`` via a join; raises ``ValueError``
    when the session is not found or is not owned by ``profile_id``.  Derives
    ``session_start`` from the DB row — never from caller-supplied data, so a
    forged anchor cannot shift window offsets.
    """

    if request.session_id is None:
        raise ValueError(
            "request.session_id must be set; direct callers of fetch_waveform_window_raw "
            "must resolve a session before calling this function"
        )

    # Full-tuple ownership query: Session + Device (profile) + Day (date) + optional device.
    # Ownership contract: the session must match profile_id, therapy_date, AND device_id.
    stmt = (
        select(models.Session.start_time)
        .join(models.Device, models.Session.device_id == models.Device.id)
        .join(models.Day, models.Session.day_id == models.Day.id)
        .where(
            models.Session.id == request.session_id,
            models.Device.profile_id == profile_id,
            models.Day.date == request.therapy_date,
        )
    )
    if request.device_id is not None:
        stmt = stmt.where(models.Session.device_id == request.device_id)
    row = (await db.execute(stmt)).one_or_none()

    if row is None:
        raise ValueError(
            f"Session {request.session_id} not found or not owned by "
            f"profile {profile_id} for date {request.therapy_date}"
        )

    session_start: datetime = row[0]
    tz_status, tz_name = await _resolve_timezone(db, profile_id)
    return await _fetch_waveform_blobs(
        db,
        request,
        request.session_id,
        session_start,
        timezone_status=tz_status,
        timezone_name=tz_name,
    )


class WaveformMixin(_BreathServiceCore):
    """Waveform-window service methods."""

    async def fetch_waveform_window(
        self, request: WaveformWindowRequest
    ) -> RawWaveformWindow:
        """Resolve, validate, and fetch raw waveform blobs for a window request.

        MCP raw/render seam: the fetch step runs
        inside the caller's DB scope while ``compute_waveform_window`` (pure, CPU-only)
        runs after the scope closes.  Direct callers that need the raw bytes or want
        to render a PNG call this method, then pass the returned ``RawWaveformWindow``
        to ``compute_waveform_window`` independently.

        Raises ``DeviceAmbiguityError`` for multi-device profiles with no device_id,
        ``DeviceNotOwnedError`` for a foreign device_id, ``ValueError`` when an
        explicit session_id is provided but the date has no sessions, and
        ``MultiSessionAmbiguityError`` when the date has multiple sessions and no
        session_id was specified.
        """
        from snore.services.breath_service import (  # noqa: PLC0415
            _fetch_waveform_blobs,
        )

        resolved_device_id, sessions_by_date = await self._resolve_range(
            request.therapy_date, request.therapy_date, request.device_id
        )
        day_sessions = sessions_by_date.get(request.therapy_date, [])

        tz_status, tz_name = await self.resolve_timezone()

        # Validate explicit session_id BEFORE the empty-day return.
        # An owned device on an empty date with an explicit session_id must raise,
        # not silently return a synthetic empty window.
        if not day_sessions:
            if request.session_id is not None:
                raise ValueError(
                    f"Session {request.session_id} not found for date "
                    f"{request.therapy_date} on device {resolved_device_id}"
                )
            return RawWaveformWindow(
                request=request,
                session_id=0,
                session_start_wall_clock=datetime.min,
                timezone_status=tz_status,
                timezone_name=tz_name,
                channels=[],
                missing_channels=list(request.channels),
            )

        if request.session_id is not None:
            # Verify the provided session_id belongs to the resolved device
            session_ids = {s.id for s in day_sessions}
            if request.session_id not in session_ids:
                raise ValueError(
                    f"Session {request.session_id} not found for date "
                    f"{request.therapy_date} on device {resolved_device_id}"
                )
            session_row = next(s for s in day_sessions if s.id == request.session_id)
        elif len(day_sessions) > 1:
            raise MultiSessionAmbiguityError(
                therapy_date=request.therapy_date,
                device_id=resolved_device_id,
                sessions=[
                    SessionSummary(
                        session_id=s.id,
                        start_wall_clock=s.start_time,
                        timezone_status=tz_status,
                        timezone_name=tz_name,
                        duration_seconds=s.duration_seconds or 0.0,
                    )
                    for s in day_sessions
                ],
            )
        else:
            session_row = day_sessions[0]

        resolved_request = request.model_copy(
            update={"device_id": resolved_device_id, "session_id": session_row.id}
        )
        return await _fetch_waveform_blobs(
            self._db,
            resolved_request,
            session_row.id,
            session_row.start_time,
            timezone_status=tz_status,
            timezone_name=tz_name,
        )

    async def get_waveform_window(
        self, request: WaveformWindowRequest
    ) -> WaveformWindow:
        """Convenience orchestrator: resolve → fetch blobs → compute. Never closes self._db.

        Uses ``_resolve_range`` for device validation and session selection (raises
        ``DeviceAmbiguityError`` for multi-device, ``DeviceNotOwnedError`` for foreign
        device_id), then delegates to ``fetch_waveform_window`` (the MCP seam) and
        applies ``compute_waveform_window`` (pure) to produce the final DTO.
        """
        from snore.services.breath_service import (  # noqa: PLC0415
            compute_waveform_window,
        )

        raw = await self.fetch_waveform_window(request)
        return compute_waveform_window(raw)
