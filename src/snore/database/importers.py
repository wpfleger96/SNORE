"""
Session import functionality for converting UnifiedSession to database records.

Handles the complete import process including waveforms, events, and statistics.
"""

import json
import logging

from datetime import UTC, datetime

import numpy as np

from sqlalchemy.orm import Session

from snore.constants import DEFAULT_PROFILE_NAME
from snore.database import models
from snore.database.day_manager import DayManager
from snore.database.session import session_scope
from snore.models.unified import UnifiedSession, WaveformData

logger = logging.getLogger(__name__)


def serialize_waveform(waveform: WaveformData) -> bytes:
    """
    Serialize waveform data to bytes for database storage.

    Stores timestamps and values as float32 numpy arrays.
    No compression - SQLite and filesystem handle that efficiently.

    Args:
        waveform: WaveformData object

    Returns:
        Serialized bytes
    """
    if isinstance(waveform.timestamps, list):
        timestamps = np.array(waveform.timestamps, dtype=np.float32)
    else:
        timestamps = waveform.timestamps.astype(np.float32)

    if isinstance(waveform.values, list):
        values = np.array(waveform.values, dtype=np.float32)
    else:
        values = waveform.values.astype(np.float32)

    data = np.column_stack([timestamps, values])
    return data.tobytes()


class SessionImporter:
    """Handles importing UnifiedSession objects to database using SQLAlchemy."""

    def __init__(self, profile_id: int | None = None):
        """
        Initialize importer.

        Args:
            profile_id: Optional profile_id to link devices to
        """
        self.profile_id = profile_id

    @staticmethod
    def get_or_create_default_profile(db: Session) -> int:
        """
        Get or create the Default profile for SD card imports.

        Returns:
            Profile ID of the Default profile
        """
        profile = (
            db.query(models.Profile).filter_by(username=DEFAULT_PROFILE_NAME).first()
        )
        if not profile:
            profile = models.Profile(
                username=DEFAULT_PROFILE_NAME, settings={"day_split_time": "12:00:00"}
            )
            db.add(profile)
            db.flush()
            logger.info(f"Created default profile '{DEFAULT_PROFILE_NAME}'")
        return profile.id

    @staticmethod
    def cleanup_orphaned_records(db: Session) -> int:
        """
        Remove orphaned records from child tables that reference non-existent sessions.

        This can happen if CASCADE delete is not enabled or if a database was corrupted.

        Args:
            db: SQLAlchemy session

        Returns:
            Number of orphaned records removed
        """
        from sqlalchemy import text

        tables = ["settings", "events", "waveforms", "statistics"]
        total_cleaned = 0

        for table in tables:
            result = db.execute(
                text(
                    f"DELETE FROM {table} WHERE session_id NOT IN (SELECT id FROM sessions)"
                )
            )
            count = result.rowcount if hasattr(result, "rowcount") else 0
            if count > 0:
                logger.info(f"Cleaned {count} orphaned records from {table}")
                total_cleaned += count

        if total_cleaned > 0:
            db.commit()

        return total_cleaned

    def import_session(self, session_data: UnifiedSession, force: bool = False) -> bool:
        """
        Import a complete session to database.

        Args:
            session_data: UnifiedSession to import
            force: If True, re-import existing sessions

        Returns:
            True if imported, False if skipped (already exists)
        """
        with session_scope() as db:
            # Auto-assign Default profile for SD card imports without profile
            if self.profile_id is None:
                self.profile_id = self.get_or_create_default_profile(db)

            device = (
                db.query(models.Device)
                .filter_by(serial_number=session_data.device_info.serial_number)
                .first()
            )

            if device:
                device.manufacturer = session_data.device_info.manufacturer
                device.model = session_data.device_info.model
                device.firmware_version = session_data.device_info.firmware_version
                device.hardware_version = session_data.device_info.hardware_version
                device.product_code = session_data.device_info.product_code
                if self.profile_id:
                    device.profile_id = self.profile_id
                device.last_import = datetime.now(UTC).replace(tzinfo=None)
            else:
                device = models.Device(
                    manufacturer=session_data.device_info.manufacturer,
                    model=session_data.device_info.model,
                    serial_number=session_data.device_info.serial_number,
                    firmware_version=session_data.device_info.firmware_version,
                    hardware_version=session_data.device_info.hardware_version,
                    product_code=session_data.device_info.product_code,
                    profile_id=self.profile_id,
                )
                db.add(device)
                db.flush()

            existing = (
                db.query(models.Session)
                .filter_by(
                    device_id=device.id,
                    device_session_id=session_data.device_session_id,
                )
                .first()
            )

            if existing and not force:
                logger.debug(
                    f"Session {session_data.device_session_id} already exists, skipping"
                )
                return False

            if existing and force:
                logger.info(
                    f"Force re-importing session {session_data.device_session_id}"
                )
                db.delete(existing)
                db.flush()

            notes_json = (
                json.dumps(session_data.data_quality_notes)
                if session_data.data_quality_notes
                else None
            )

            new_session = models.Session(
                device_id=device.id,
                device_session_id=session_data.device_session_id,
                start_time=session_data.start_time,
                end_time=session_data.end_time,
                duration_seconds=session_data.duration_seconds,
                therapy_mode=session_data.settings.mode.value
                if session_data.settings
                else None,
                import_source=session_data.import_source,
                parser_version=session_data.parser_version,
                data_quality_notes=notes_json,
                has_waveform_data=session_data.has_waveform_data,
                has_event_data=session_data.has_event_data,
                has_statistics=session_data.has_statistics,
            )
            db.add(new_session)
            db.flush()

            if device.profile_id:
                profile = db.query(models.Profile).get(device.profile_id)
                if profile:
                    day_date = DayManager.get_day_for_session(
                        session_data.start_time, profile
                    )
                    day = DayManager.create_or_update_day(
                        device.profile_id, day_date, db
                    )
                    new_session.day_id = day.id

            if session_data.has_waveform_data:
                self._import_waveforms(db, new_session.id, session_data)

            if session_data.has_event_data:
                self._import_events(db, new_session.id, session_data)

            if session_data.has_statistics:
                self._import_statistics(db, new_session.id, session_data)

            if session_data.settings:
                self._import_settings(db, new_session.id, session_data)

        logger.info(
            f"Imported session {session_data.device_session_id} from {session_data.start_time}"
        )
        return True

    def _import_waveforms(
        self, db: Session, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import all waveforms for session."""
        if not session_data.waveforms:
            return

        waveform_records = []
        for waveform_type, waveform in session_data.waveforms.items():
            data_blob = serialize_waveform(waveform)
            sample_count = (
                len(waveform.values)
                if isinstance(waveform.values, list)
                else len(waveform.values)
            )

            waveform_records.append(
                models.Waveform(
                    session_id=session_id,
                    waveform_type=waveform_type.value,
                    sample_rate=waveform.sample_rate,
                    unit=waveform.unit,
                    min_value=waveform.min_value,
                    max_value=waveform.max_value,
                    mean_value=waveform.mean_value,
                    data_blob=data_blob,
                    sample_count=sample_count,
                )
            )

        db.bulk_save_objects(waveform_records)
        logger.debug(f"Bulk imported {len(waveform_records)} waveforms")

    def _import_events(
        self, db: Session, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import all respiratory events for session."""
        if not session_data.events:
            return

        event_records = [
            models.Event(
                session_id=session_id,
                event_type=event.event_type.value,
                start_time=event.start_time,
                duration_seconds=event.duration_seconds,
                spo2_drop=event.spo2_drop,
                peak_flow_limitation=event.peak_flow_limitation,
            )
            for event in session_data.events
        ]
        db.bulk_save_objects(event_records)

        logger.debug(f"Bulk imported {len(event_records)} events")

    def _import_statistics(
        self, db: Session, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import session statistics."""
        stats = session_data.statistics

        stats_record = models.Statistics(
            session_id=session_id,
            obstructive_apneas=stats.obstructive_apneas,
            central_apneas=stats.central_apneas,
            mixed_apneas=stats.mixed_apneas,
            hypopneas=stats.hypopneas,
            reras=stats.reras,
            flow_limitations=stats.flow_limitations,
            ahi=stats.ahi,
            oai=stats.oai,
            cai=stats.cai,
            hi=stats.hi,
            rei=stats.rei,
            pressure_min=stats.pressure_min,
            pressure_max=stats.pressure_max,
            pressure_median=stats.pressure_median,
            pressure_mean=stats.pressure_mean,
            pressure_95th=stats.pressure_95th,
            leak_min=stats.leak_min,
            leak_max=stats.leak_max,
            leak_median=stats.leak_median,
            leak_mean=stats.leak_mean,
            leak_95th=stats.leak_95th,
            leak_percentile_70=stats.leak_percentile_70,
            respiratory_rate_min=stats.respiratory_rate_min,
            respiratory_rate_max=stats.respiratory_rate_max,
            respiratory_rate_mean=stats.respiratory_rate_mean,
            tidal_volume_min=stats.tidal_volume_min,
            tidal_volume_max=stats.tidal_volume_max,
            tidal_volume_mean=stats.tidal_volume_mean,
            minute_ventilation_min=stats.minute_ventilation_min,
            minute_ventilation_max=stats.minute_ventilation_max,
            minute_ventilation_mean=stats.minute_ventilation_mean,
            spo2_min=stats.spo2_min,
            spo2_max=stats.spo2_max,
            spo2_mean=stats.spo2_mean,
            spo2_time_below_90=stats.spo2_time_below_90,
            pulse_min=stats.pulse_min,
            pulse_max=stats.pulse_max,
            pulse_mean=stats.pulse_mean,
            usage_hours=stats.usage_hours,
        )
        db.add(stats_record)

        logger.debug("Imported session statistics")

    def _import_settings(
        self, db: Session, session_id: int, session_data: UnifiedSession
    ) -> None:
        """Import session settings."""
        settings = session_data.settings

        if not settings:
            return

        settings_dict = {
            "mode": settings.mode.value,
            "pressure_min": settings.pressure_min,
            "pressure_max": settings.pressure_max,
            "pressure_fixed": settings.pressure_fixed,
            "ipap": settings.ipap,
            "epap": settings.epap,
            "epr_level": settings.epr_level,
            "ramp_time": settings.ramp_time,
            "ramp_start_pressure": settings.ramp_start_pressure,
            "humidity_level": settings.humidity_level,
            "tube_temp": settings.tube_temp,
            "mask_type": settings.mask_type,
        }

        if settings.other_settings:
            settings_dict.update(settings.other_settings)

        for key, value in settings_dict.items():
            if value is not None:
                setting_record = models.Setting(
                    session_id=session_id, key=key, value=str(value)
                )
                db.add(setting_record)

        logger.debug(f"Imported {len(settings_dict)} settings")


def import_session(
    session_data: UnifiedSession, profile_id: int | None = None, force: bool = False
) -> bool:
    """
    Convenience function to import a session.

    Args:
        session_data: UnifiedSession to import
        profile_id: Optional profile_id to link device to
        force: Force re-import if exists

    Returns:
        True if imported, False if skipped
    """
    importer = SessionImporter(profile_id=profile_id)
    return importer.import_session(session_data, force=force)
