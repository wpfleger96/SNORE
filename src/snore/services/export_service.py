"""Unified export service for CPAP data in multiple formats.

Supports:
- raw: OSCAR-compatible SD card directory reconstruction from backups
- csv: Parsed data as CSV files (sessions, events, settings, columns provenance
  sidecar, optional waveforms)
- json: Parsed data as a single JSON document
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
import zipfile

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from pydantic import BaseModel
from sqlalchemy import ColumnElement, select
from sqlalchemy.ext.asyncio import AsyncSession

from snore.constants import DEFAULT_RAW_BACKUP_DIR
from snore.database import models
from snore.metrics import EXPORT_STAT_KEYS
from snore.parsers.base import RawFileManifest
from snore.provenance import Provenance, field_description, field_provenance
from snore.services.schemas import SessionDetail, SessionStatistics

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Exported session measurement columns, each read from the response-schema
# field that carries the same value.  Identifier columns (session id, dates,
# device serial/model, therapy mode, timezone) carry no tier.
_EXPORT_COLUMN_FIELDS: dict[str, type[BaseModel]] = {
    "duration_hours": SessionDetail,
    **dict.fromkeys(EXPORT_STAT_KEYS, SessionStatistics),
}

EXPORT_COLUMN_PROVENANCE: dict[str, Provenance] = {
    name: field_provenance(model, name) for name, model in _EXPORT_COLUMN_FIELDS.items()
}


def _columns_csv_rows() -> list[list[str]]:
    """Rows of the ``columns.csv`` provenance sidecar.

    One row per ``sessions.csv`` measurement column; ``events.csv`` and
    ``settings.csv`` get one whole-file row each since every value in them is
    device data.
    """
    return [
        ["name", "provenance", "description"],
        *(
            [name, tier, field_description(_EXPORT_COLUMN_FIELDS[name], name)]
            for name, tier in EXPORT_COLUMN_PROVENANCE.items()
        ),
        ["events.csv", Provenance.DEVICE, "Device-scored events, verbatim."],
        ["settings.csv", Provenance.DEVICE, "Device settings, verbatim."],
    ]


@dataclass
class ExportResult:
    """Result of an export operation."""

    format: str
    output_path: Path
    nights_exported: int = 0
    files_written: int = 0
    total_bytes: int = 0
    warnings: list[str] = field(default_factory=list)


class ExportService:
    """Unified export service for CPAP data."""

    def __init__(self, profile_id: int, backup_root: Path | None = None) -> None:
        self.profile_id = profile_id
        # Default backup root is namespaced by profile so raw files from
        # different profiles never share the same directory.
        self.backup_root = backup_root or DEFAULT_RAW_BACKUP_DIR / str(profile_id)

    def _profile_filters(self) -> list[ColumnElement[bool]]:
        """Return profile isolation predicates (always applied — never global)."""
        return [models.Device.profile_id == self.profile_id]

    # ------------------------------------------------------------------
    # Raw export (filesystem only, no DB)
    # ------------------------------------------------------------------

    def export_raw(
        self,
        output: Path,
        date_from: date | None = None,
        date_to: date | None = None,
        device_serial: str | None = None,
        as_zip: bool = False,
        dry_run: bool = False,
        trim_str: bool = False,
    ) -> ExportResult:
        """Export raw backup files as an OSCAR-compatible directory.

        Works entirely from the filesystem — no database needed.
        """
        from snore.parsers.register_all import ensure_registered_parsers
        from snore.parsers.registry import parser_registry

        ensure_registered_parsers()

        serial = self._resolve_device_serial(device_serial)
        device_root = self.backup_root / serial

        if not device_root.is_dir():
            raise FileNotFoundError(
                f"No backup found for device '{serial}' at {device_root}.\n"
                "Run 'snore import /path/to/sd' to create a backup first."
            )

        parser = parser_registry.detect_parser(device_root)
        if parser is None:
            raise RuntimeError(f"No parser recognized the backup at {device_root}")

        if not parser.supports_raw_backup:
            raise RuntimeError(
                f"Parser '{parser.parser_id}' does not support raw export"
            )

        manifest = parser.get_raw_file_manifest(device_root, date_from, date_to)

        if dry_run:
            return self._dry_run_raw(manifest, output, date_from, date_to)

        warnings: list[str] = []
        if (date_from or date_to) and manifest.device_files and not trim_str:
            warnings.append(
                "Device-level files may contain data outside the exported date range. "
                "Use --trim-str to trim STR.edf to the date range."
            )

        if as_zip or str(output).endswith(".zip"):
            result = self._export_raw_zip(manifest, output, warnings)
        else:
            result = self._export_raw_dir(manifest, output, warnings)

        if (
            trim_str
            and date_from
            and date_to
            and not (as_zip or str(output).endswith(".zip"))
        ):
            parser.trim_device_summary(output, date_from, date_to)

        return result

    def _dry_run_raw(
        self,
        manifest: RawFileManifest,
        output: Path,
        date_from: date | None,
        date_to: date | None,
    ) -> ExportResult:
        """Return what would be exported without copying."""
        return ExportResult(
            format="raw",
            output_path=output,
            nights_exported=len(manifest.nights),
            files_written=manifest.total_files,
            total_bytes=manifest.total_bytes,
        )

    def _export_raw_dir(
        self,
        manifest: RawFileManifest,
        output: Path,
        warnings: list[str],
    ) -> ExportResult:
        """Copy raw files to a directory."""
        output.mkdir(parents=True, exist_ok=True)
        files_written = 0

        for src in manifest.device_files:
            dest = output / src.relative_to(manifest.source_root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            files_written += 1

        for _night_date, files in sorted(manifest.nights.items()):
            for src in files:
                dest = output / src.relative_to(manifest.source_root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                files_written += 1

        return ExportResult(
            format="raw",
            output_path=output,
            nights_exported=len(manifest.nights),
            files_written=files_written,
            warnings=warnings,
        )

    def _export_raw_zip(
        self,
        manifest: RawFileManifest,
        output: Path,
        warnings: list[str],
    ) -> ExportResult:
        """Write raw files to a zip archive."""
        output.parent.mkdir(parents=True, exist_ok=True)
        files_written = 0

        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
            for src in manifest.device_files:
                arcname = str(src.relative_to(manifest.source_root))
                zf.write(src, arcname)
                files_written += 1

            for _night_date, files in sorted(manifest.nights.items()):
                for src in files:
                    arcname = str(src.relative_to(manifest.source_root))
                    zf.write(src, arcname)
                    files_written += 1

        return ExportResult(
            format="raw",
            output_path=output,
            nights_exported=len(manifest.nights),
            files_written=files_written,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # CSV export (from database)
    # ------------------------------------------------------------------

    async def export_csv(
        self,
        db_session: AsyncSession,
        output: Path,
        date_from: date | None = None,
        date_to: date | None = None,
        device_serial: str | None = None,
        include_waveforms: bool = False,
    ) -> ExportResult:
        """Export parsed data as CSV files.

        All three output files (sessions, events, settings) are written in a
        single generator pass — no full materialisation of the result set.
        """
        from snore.database import models  # noqa: PLC0415

        output.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        files_written = 0
        nights: set[date] = set()

        if include_waveforms:
            warnings.append(
                "Waveform export enabled. Files may be very large "
                "(~720K rows per 8-hour session at 25Hz)."
            )

        session_header = [
            "device_session_id",
            "date",
            "start_time",
            "end_time",
            "duration_hours",
            "device_serial",
            "device_model",
            "therapy_mode",
            "timezone",
            *EXPORT_STAT_KEYS,
        ]
        event_header = [
            "device_session_id",
            "session_date",
            "event_type",
            "start_time",
            "duration_seconds",
            "timezone",
        ]
        setting_header = ["device_session_id", "session_date", "key", "value"]

        sessions_path = output / "sessions.csv"
        events_path = output / "events.csv"
        settings_path = output / "settings.csv"

        with open(output / "columns.csv", "w", newline="") as cf:
            csv.writer(cf).writerows(_columns_csv_rows())

        found_any = False
        with (
            open(sessions_path, "w", newline="") as sf,
            open(events_path, "w", newline="") as ef,
            open(settings_path, "w", newline="") as stf,
        ):
            sw = csv.writer(sf)
            ew = csv.writer(ef)
            stw = csv.writer(stf)
            sw.writerow(session_header)
            ew.writerow(event_header)
            stw.writerow(setting_header)

            async for s, evs, stts in self._build_export_rows(
                db_session, date_from, date_to, device_serial
            ):
                found_any = True
                stats = s.get("statistics") or {}
                night = (
                    s["start_time"].date()
                    if s["start_time"].hour >= 12
                    else (s["start_time"] - timedelta(days=1)).date()
                )
                nights.add(night)

                sw.writerow(
                    [
                        s["device_session_id"],
                        s["start_time"].strftime("%Y-%m-%d"),
                        s["start_time"].isoformat(),
                        s["end_time"].isoformat(),
                        round(s["duration_seconds"] / 3600, 2)
                        if s["duration_seconds"]
                        else "",
                        s["serial_number"],
                        s["model"],
                        s["therapy_mode"],
                        "local",
                        *[stats.get(k, "") for k in EXPORT_STAT_KEYS],
                    ]
                )

                for e in evs:
                    ew.writerow(
                        [
                            s["device_session_id"],
                            s["start_time"].strftime("%Y-%m-%d"),
                            e.event_type,
                            e.start_time.isoformat() if e.start_time else "",
                            e.duration_seconds or "",
                            "local",
                        ]
                    )

                for st in stts:
                    stw.writerow(
                        [
                            s["device_session_id"],
                            s["start_time"].strftime("%Y-%m-%d"),
                            st.key,
                            st.value,
                        ]
                    )

                if include_waveforms:
                    sid = s["id"]
                    waveforms_dir = output / "waveforms"
                    waveforms_dir.mkdir(exist_ok=True)
                    waveforms = (
                        (
                            await db_session.execute(
                                select(models.Waveform).filter_by(session_id=sid)
                            )
                        )
                        .scalars()
                        .all()
                    )
                    for w in waveforms:
                        if not w.data_blob:
                            continue
                        fname = (
                            f"{s['serial_number']}_{s['start_time']:%Y%m%d}_"
                            f"{s['start_time']:%H%M%S}_{w.waveform_type}.csv"
                        )
                        wpath = waveforms_dir / fname
                        flat = np.frombuffer(w.data_blob, dtype=np.float32)
                        if len(flat) % 2 != 0:
                            continue
                        wf_rows = flat.reshape(-1, 2)
                        with open(wpath, "w", newline="") as wf:
                            writer = csv.writer(wf)
                            writer.writerow(["offset_seconds", "value"])
                            for wrow in wf_rows:
                                writer.writerow([f"{wrow[0]:.3f}", f"{wrow[1]:.3f}"])
                        files_written += 1

        if not found_any:
            warnings.append("No sessions found for the specified filters.")
            return ExportResult(format="csv", output_path=output, warnings=warnings)

        files_written += 4

        return ExportResult(
            format="csv",
            output_path=output,
            nights_exported=len(nights),
            files_written=files_written,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # JSON export (from database)
    # ------------------------------------------------------------------

    async def export_json(
        self,
        db_session: AsyncSession,
        output: Path,
        date_from: date | None = None,
        date_to: date | None = None,
        device_serial: str | None = None,
    ) -> ExportResult:
        """Export parsed data as a single JSON document.

        Writes the JSON array incrementally — one object per session — so the
        full session list is never held in memory simultaneously.
        """
        output.parent.mkdir(parents=True, exist_ok=True)
        warnings: list[str] = []
        nights: set[date] = set()
        session_count = 0

        def _session_obj(
            s: dict[str, Any], events: list[Any], settings: list[Any]
        ) -> str:
            """Serialise one session to a compact JSON object string."""
            obj = {
                "device_session_id": s["device_session_id"],
                "date": s["start_time"].strftime("%Y-%m-%d"),
                "start_time": s["start_time"].isoformat(),
                "end_time": s["end_time"].isoformat(),
                "duration_hours": round(s["duration_seconds"] / 3600, 2)
                if s["duration_seconds"]
                else None,
                "timezone": "local",
                "device": {
                    "serial": s["serial_number"],
                    "model": s["model"],
                    "manufacturer": s["manufacturer"],
                },
                "statistics": s.get("statistics") or {},
                "settings": {st.key: st.value for st in settings},
                "events": [
                    {
                        "event_type": e.event_type,
                        "start_time": e.start_time.isoformat()
                        if e.start_time
                        else None,
                        "duration_seconds": e.duration_seconds,
                    }
                    for e in events
                ],
            }
            return json.dumps(obj, indent=2, default=str)

        with open(output, "w") as f:
            header = {
                "exported_at": datetime.now().isoformat(),
                "snore_export_format": "1.1",
                "date_range": {
                    "from": date_from.isoformat() if date_from else None,
                    "to": date_to.isoformat() if date_to else None,
                },
                "provenance": {
                    "columns": EXPORT_COLUMN_PROVENANCE,
                    "sections": {
                        "events": Provenance.DEVICE,
                        "settings": Provenance.DEVICE,
                    },
                },
            }
            f.write("{\n")
            for k, v in header.items():
                f.write(f"  {json.dumps(k)}: {json.dumps(v, default=str)},\n")
            f.write('  "sessions": [\n')

            first_session = True
            async for s, events, settings in self._build_export_rows(
                db_session, date_from, date_to, device_serial
            ):
                night = (
                    s["start_time"].date()
                    if s["start_time"].hour >= 12
                    else (s["start_time"] - timedelta(days=1)).date()
                )
                nights.add(night)
                session_count += 1

                if not first_session:
                    f.write(",\n")
                first_session = False
                obj_str = _session_obj(s, events, settings)
                indented = "\n".join("    " + line for line in obj_str.splitlines())
                f.write(indented)

            f.write("\n  ],\n")
            f.write(f'  "session_count": {session_count}\n')
            f.write("}\n")

        if session_count == 0:
            warnings.append("No sessions found for the specified filters.")

        return ExportResult(
            format="json",
            output_path=output,
            nights_exported=len(nights),
            files_written=1,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    _EXPORT_CHUNK_SIZE: int = 500  # Sessions per DB fetch window

    async def _build_export_rows(
        self,
        db_session: AsyncSession,
        date_from: date | None,
        date_to: date | None,
        device_serial: str | None,
    ) -> AsyncIterator[tuple[dict[str, Any], list[Any], list[Any]]]:
        """Yield ``(session_dict, events, settings)`` tuples in bounded chunks.

        Fetches sessions in windows of ``_EXPORT_CHUNK_SIZE`` rows and bulk-loads
        events/settings for each window.  No full materialisation of the entire
        result set occurs; memory is bounded to one chunk at a time.
        """

        chunk: list[dict[str, Any]] = []
        async for row in self._query_sessions_chunked(
            db_session, date_from, date_to, device_serial
        ):
            chunk.append(row)
            if len(chunk) >= self._EXPORT_CHUNK_SIZE:
                session_ids = [s["id"] for s in chunk]
                events_by_session = await self._bulk_load_events(
                    db_session, session_ids
                )
                settings_by_session = await self._bulk_load_settings(
                    db_session, session_ids
                )
                for s in chunk:
                    yield (
                        s,
                        events_by_session.get(s["id"], []),
                        settings_by_session.get(s["id"], []),
                    )
                chunk = []

        if chunk:
            session_ids = [s["id"] for s in chunk]
            events_by_session = await self._bulk_load_events(db_session, session_ids)
            settings_by_session = await self._bulk_load_settings(
                db_session, session_ids
            )
            for s in chunk:
                yield (
                    s,
                    events_by_session.get(s["id"], []),
                    settings_by_session.get(s["id"], []),
                )

    def _resolve_device_serial(self, device_serial: str | None) -> str:
        """Resolve device serial, failing fast on ambiguity."""
        if device_serial:
            return device_serial

        if not self.backup_root.is_dir():
            raise FileNotFoundError(
                f"No backup directory found at {self.backup_root}.\n"
                "Run 'snore import /path/to/sd' to create backups first."
            )

        devices = [
            d.name
            for d in self.backup_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        if len(devices) == 0:
            raise FileNotFoundError(f"No device backups found in {self.backup_root}.")
        if len(devices) == 1:
            return devices[0]

        device_list = "\n  ".join(devices)
        raise ValueError(
            f"Multiple devices found. Specify one with --device:\n  {device_list}"
        )

    async def _query_sessions(
        self,
        db_session: AsyncSession,
        date_from: date | None,
        date_to: date | None,
        device_serial: str | None,
    ) -> list[dict[str, Any]]:
        """Query sessions with optional filters, returning dicts with joined data."""
        result = []
        async for row in self._query_sessions_chunked(
            db_session, date_from, date_to, device_serial
        ):
            result.append(row)
        return result

    async def _query_sessions_chunked(
        self,
        db_session: AsyncSession,
        date_from: date | None,
        date_to: date | None,
        device_serial: str | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield session dicts one-by-one using async streaming.

        Memory is bounded to one database page at a time regardless of result set
        size.  Callers that need all rows at once should use ``_query_sessions``.
        """
        from snore.database import models  # noqa: PLC0415

        stats_alias = models.Statistics
        stmt = (
            select(
                models.Session.id,
                models.Session.device_session_id,
                models.Session.start_time,
                models.Session.end_time,
                models.Session.duration_seconds,
                models.Session.therapy_mode,
                models.Device.serial_number,
                models.Device.model,
                models.Device.manufacturer,
                *[getattr(stats_alias, k) for k in EXPORT_STAT_KEYS],
            )
            .join(models.Device, models.Session.device_id == models.Device.id)
            .outerjoin(stats_alias, models.Session.id == stats_alias.session_id)
            .where(models.Session.enabled.is_(True), *self._profile_filters())
            .order_by(models.Session.start_time)
        )

        if device_serial:
            stmt = stmt.where(models.Device.serial_number == device_serial)
        if date_from:
            stmt = stmt.where(
                models.Session.start_time
                >= datetime(date_from.year, date_from.month, date_from.day)
            )
        if date_to:
            stmt = stmt.where(
                models.Session.start_time
                <= datetime(date_to.year, date_to.month, date_to.day, 23, 59, 59)
            )

        result = await db_session.execute(stmt)
        for row in result:
            r = dict(row._mapping)
            r["statistics"] = {
                k: r.pop(k) for k in EXPORT_STAT_KEYS if r.get(k) is not None
            }
            for k in EXPORT_STAT_KEYS:
                r.pop(k, None)
            yield r

    @staticmethod
    async def _bulk_load_events(
        db_session: AsyncSession, session_ids: list[int]
    ) -> dict[int, list[Any]]:
        """Load all events for the given session IDs in one query."""
        from collections import defaultdict

        from snore.database import models

        if not session_ids:
            return {}

        events = (
            (
                await db_session.execute(
                    select(models.Event)
                    .filter(models.Event.session_id.in_(session_ids))
                    .order_by(models.Event.session_id, models.Event.start_time)
                )
            )
            .scalars()
            .all()
        )
        grouped: dict[int, list[Any]] = defaultdict(list)
        for e in events:
            grouped[e.session_id].append(e)
        return dict(grouped)

    @staticmethod
    async def _bulk_load_settings(
        db_session: AsyncSession, session_ids: list[int]
    ) -> dict[int, list[Any]]:
        """Load all settings for the given session IDs in one query."""
        from collections import defaultdict

        from snore.database import models

        if not session_ids:
            return {}

        settings = (
            (
                await db_session.execute(
                    select(models.Setting)
                    .filter(models.Setting.session_id.in_(session_ids))
                    .order_by(models.Setting.session_id, models.Setting.key)
                )
            )
            .scalars()
            .all()
        )
        grouped: dict[int, list[Any]] = defaultdict(list)
        for s in settings:
            grouped[s.session_id].append(s)
        return dict(grouped)
