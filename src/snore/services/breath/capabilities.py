"""Device capabilities and contextual events."""

from __future__ import annotations

from datetime import date

from sqlalchemy import exists, select
from sqlalchemy import func as sqlfunc

from snore.analysis.rx_tracker import RX_KEYS as _RX_KEYS
from snore.analysis.shared.versioning import NullReason
from snore.database import models
from snore.parsers.register_all import ensure_registered_parsers
from snore.parsers.registry import parser_registry
from snore.utils.db_chunk import iter_id_chunks

from ._core import _BreathServiceCore
from .algorithms import _extract_window_mean
from .dtos import (
    ContextualEvent,
    DeviceCapabilities,
    WaveformChannelName,
    WaveformWindowRequest,
)


class CapabilitiesMixin(_BreathServiceCore):
    """Device-capability and contextual-event methods."""

    async def get_device_capabilities(
        self,
        device_id: int,
        date_start: date | None = None,
        date_end: date | None = None,
    ) -> DeviceCapabilities:
        """Actual covered range + channels, event types, setting keys present."""
        ensure_registered_parsers()

        # Verify device ownership before querying (fetch full row for identity fields)
        owned_device = (
            (
                await self._db.execute(
                    select(models.Device).where(
                        models.Device.id == device_id,
                        models.Device.profile_id == self._profile_id,
                    )
                )
            )
            .scalars()
            .first()
        )
        if owned_device is None:
            return DeviceCapabilities(
                device_id=device_id,
                requested_date_start=date_start,
                requested_date_end=date_end,
                actual_date_start=None,
                actual_date_end=None,
                null_reason=NullReason.NOT_AVAILABLE,
                channels_present=[],
                all_setting_keys_present=[],
                rx_keys_present=[],
                event_types_present=[],
                session_count=0,
                nights_with_data=0,
                supported_vendor_models=[],
                manufacturer=None,
                model=None,
                serial_number=None,
            )

        # Date range of actual data — only days with at least one Session count
        # as "imported nights".  DayManager.recalculate_day prunes orphaned Day
        # rows, so this predicate is defence-in-depth against hand-edited data.
        day_stmt = select(models.Day).where(
            models.Day.device_id == device_id,
            exists().where(models.Session.day_id == models.Day.id),
        )
        if date_start is not None:
            day_stmt = day_stmt.where(models.Day.date >= date_start)
        if date_end is not None:
            day_stmt = day_stmt.where(models.Day.date <= date_end)
        days = (await self._db.execute(day_stmt)).scalars().all()

        null_reason: NullReason | None = None
        actual_start: date | None = None
        actual_end: date | None = None
        session_count = 0
        nights_with_data = 0

        if not days:
            # Owned device exists but has no data in range
            null_reason = NullReason.NO_DATA_IN_RANGE
        else:
            actual_start = min(d.date for d in days)
            actual_end = max(d.date for d in days)
            nights_with_data = len(days)
            day_ids = [d.id for d in days]

            # Chunk the unbounded day_id list (SQLite bound-param cap); each day
            # falls in exactly one chunk, so per-chunk counts sum to the total.
            for chunk in iter_id_chunks(day_ids):
                chunk_count = (
                    await self._db.execute(
                        select(sqlfunc.count())
                        .select_from(models.Session)
                        .where(models.Session.day_id.in_(chunk))
                    )
                ).scalar()
                session_count += chunk_count or 0

        # Session IDs for this device in range
        sess_stmt = (
            select(models.Session.id)
            .join(models.Day, models.Session.day_id == models.Day.id)
            .where(models.Day.device_id == device_id)
        )
        if date_start is not None:
            sess_stmt = sess_stmt.where(models.Day.date >= date_start)
        if date_end is not None:
            sess_stmt = sess_stmt.where(models.Day.date <= date_end)
        session_ids = list((await self._db.execute(sess_stmt)).scalars().all())

        channels_present: list[str] = []
        event_types_present: list[str] = []
        all_setting_keys: list[str] = []

        if session_ids:
            # Chunk the unbounded session_id list (SQLite bound-param cap); each
            # session falls in exactly one chunk, so the DISTINCT sets union
            # cleanly across chunks with no double-counting.
            channel_set: set[str] = set()
            event_type_set: set[str] = set()
            setting_key_set: set[str] = set()
            for chunk in iter_id_chunks(session_ids):
                wf_rows = (
                    (
                        await self._db.execute(
                            select(models.Waveform.waveform_type)
                            .where(models.Waveform.session_id.in_(chunk))
                            .distinct()
                        )
                    )
                    .scalars()
                    .all()
                )
                channel_set.update(str(w) for w in wf_rows)

                ev_rows = (
                    (
                        await self._db.execute(
                            select(models.Event.event_type)
                            .where(models.Event.session_id.in_(chunk))
                            .distinct()
                        )
                    )
                    .scalars()
                    .all()
                )
                event_type_set.update(str(e) for e in ev_rows)

                setting_rows = (
                    (
                        await self._db.execute(
                            select(models.Setting.key)
                            .where(models.Setting.session_id.in_(chunk))
                            .distinct()
                        )
                    )
                    .scalars()
                    .all()
                )
                setting_key_set.update(str(k) for k in setting_rows)
            channels_present = sorted(channel_set)
            event_types_present = sorted(event_type_set)
            all_setting_keys = sorted(setting_key_set)

        # rx_keys_present: only keys that actually have non-null values
        rx_keys: list[str] = []
        if session_ids:
            # Chunk only the unbounded session_id list; the _RX_KEYS filter is a
            # small constant IN-list kept in every chunk (union across chunks).
            rx_key_set: set[str] = set()
            for chunk in iter_id_chunks(session_ids):
                rx_key_rows = (
                    (
                        await self._db.execute(
                            select(models.Setting.key)
                            .where(
                                models.Setting.session_id.in_(chunk),
                                models.Setting.key.in_(list(_RX_KEYS)),
                                models.Setting.value.is_not(None),
                            )
                            .distinct()
                        )
                    )
                    .scalars()
                    .all()
                )
                rx_key_set.update(str(k) for k in rx_key_rows)
            rx_keys = sorted(rx_key_set)

        # Supported vendor models from parsers registry — let real exceptions propagate
        supported_models: list[str] = list(parser_registry.list_supported_models())

        return DeviceCapabilities(
            device_id=device_id,
            requested_date_start=date_start,
            requested_date_end=date_end,
            actual_date_start=actual_start,
            actual_date_end=actual_end,
            null_reason=null_reason,
            channels_present=channels_present,
            all_setting_keys_present=all_setting_keys,
            rx_keys_present=rx_keys,
            event_types_present=event_types_present,
            session_count=session_count,
            nights_with_data=nights_with_data,
            supported_vendor_models=supported_models,
            manufacturer=owned_device.manufacturer,
            model=owned_device.model,
            serial_number=owned_device.serial_number,
        )

    async def get_contextual_events(
        self,
        therapy_date: date,
        event_types: list[str] | None = None,
        min_duration: float | None = None,
        device_id: int | None = None,
    ) -> list[ContextualEvent]:
        """Machine events enriched with waveform context.

        Returns events from ALL sessions on the resolved device.
        Pressure and leak values are sampled at the event start (±5 s window).
        MV is the mean over the 120 s preceding the event.
        All values are ``null`` + ``NOT_AVAILABLE`` when the relevant channel is absent.
        """
        from snore.services.breath_service import (  # noqa: PLC0415
            _fetch_waveform_blobs,
            compute_waveform_window,
        )

        # Input validation
        if event_types is not None:
            if not isinstance(event_types, list) or not all(
                isinstance(et, str) and et for et in event_types
            ):
                raise ValueError(
                    "event_types must be None or a list of non-empty strings"
                )
            # Deduplicate (order-preserving), then enforce the 50-item cap.
            event_types = list(dict.fromkeys(event_types))
            if len(event_types) > 50:
                raise ValueError("event_types must contain at most 50 unique values")
        if min_duration is not None and min_duration < 0:
            raise ValueError("min_duration must be None or >= 0")

        # Resolve device via _resolve_range — DeviceAmbiguityError and ownership
        # errors propagate to the caller; a foreign/unknown device is not []
        resolved_device_id, sessions_by_date = await self._resolve_range(
            therapy_date, therapy_date, device_id
        )
        day_sessions = sessions_by_date.get(therapy_date, [])

        tz_status, tz_name = await self.resolve_timezone()
        results: list[ContextualEvent] = []
        for session_row in day_sessions:
            session_id = session_row.id
            session_start = session_row.start_time
            session_start_f = session_start.timestamp()

            # Fetch machine events for this session
            ev_stmt = select(models.Event).where(models.Event.session_id == session_id)
            if event_types:
                ev_stmt = ev_stmt.where(models.Event.event_type.in_(event_types))
            if min_duration is not None:
                ev_stmt = ev_stmt.where(models.Event.duration_seconds >= min_duration)
            ev_stmt = ev_stmt.order_by(models.Event.start_time)
            events = (await self._db.execute(ev_stmt)).scalars().all()

            # Pre-load all needed channels for this session ONCE — one DB fetch for
            # all events rather than two per event (fix: per-event blob read N+1).
            # Corrupt blobs still raise ValueError — never silently skipped.
            session_duration_s = session_row.duration_seconds or 32400.0
            pre_raw = await _fetch_waveform_blobs(
                self._db,
                WaveformWindowRequest(
                    therapy_date=therapy_date,
                    session_id=session_id,
                    device_id=resolved_device_id,
                    channels=[
                        WaveformChannelName.PRESSURE,
                        WaveformChannelName.LEAK,
                        WaveformChannelName.MV,
                    ],
                    offset_start=0.0,
                    offset_end=session_duration_s,
                    window_cap_seconds=session_duration_s,
                ),
                session_id,
                session_start,
            )
            pre_window = compute_waveform_window(pre_raw)
            # Map channel_type → (offsets, values) for O(1) per-event slicing
            pre_ch: dict[WaveformChannelName, tuple[list[float], list[float]]] = {
                ch.channel_type: (ch.offset_seconds, ch.values)
                for ch in pre_window.channels
            }

            for ev in events:
                ev_start_f = ev.start_time.timestamp()
                offset_s = ev_start_f - session_start_f
                minutes_since = offset_s / 60.0

                pressure_at: float | None = None
                pressure_reason: NullReason | None = NullReason.NOT_AVAILABLE
                leak_at: float | None = None
                leak_reason: NullReason | None = NullReason.NOT_AVAILABLE
                mv_prior: float | None = None
                mv_reason: NullReason | None = NullReason.NOT_AVAILABLE

                window_start = max(0.0, offset_s - 5.0)
                window_end = offset_s + 5.0
                mv_window_start = max(0.0, offset_s - 120.0)

                # Slice pre-loaded arrays — no DB access per event.
                # Guard window_end > 0 (fix: event before session start crash).
                if window_end > 0.0:
                    for ch_type in (
                        WaveformChannelName.PRESSURE,
                        WaveformChannelName.LEAK,
                    ):
                        if ch_type in pre_ch:
                            offsets, vals = pre_ch[ch_type]
                            val = _extract_window_mean(
                                offsets, vals, window_start, window_end
                            )
                            if val is not None:
                                if ch_type == WaveformChannelName.PRESSURE:
                                    pressure_at = val
                                    pressure_reason = None
                                else:
                                    leak_at = val
                                    leak_reason = None

                # MV window (prior 120 s)
                if offset_s > 0.0 and WaveformChannelName.MV in pre_ch:
                    offsets, vals = pre_ch[WaveformChannelName.MV]
                    val = _extract_window_mean(offsets, vals, mv_window_start, offset_s)
                    if val is not None:
                        mv_prior = val
                        mv_reason = None

                results.append(
                    ContextualEvent(
                        session_id=session_id,
                        session_start_wall_clock=session_start,
                        event_type=ev.event_type,
                        event_start_wall_clock=ev.start_time,
                        timezone_status=tz_status,
                        timezone_name=tz_name,
                        offset_seconds=offset_s,
                        duration_seconds=ev.duration_seconds,
                        pressure_at_event_cmh2o=pressure_at,
                        pressure_reason=pressure_reason,
                        leak_at_event_lpm=leak_at,
                        leak_reason=leak_reason,
                        mv_prior_120s_lpm=mv_prior,
                        mv_reason=mv_reason,
                        minutes_since_session_start=minutes_since,
                    )
                )
        return results
