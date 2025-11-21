"""
Fixture loading utilities for test data.

Provides functions to load real session fixtures and import them to test databases.
"""

from pathlib import Path
from typing import List, Tuple

from sqlalchemy.orm import Session

from oscar_mcp.parsers.resmed_edf import ResmedEDFParser
from oscar_mcp.database.models import Session as CPAPSession


# Path to fixtures directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "real_sessions"


def get_available_fixtures() -> List[str]:
    """
    Get list of available real session fixtures.

    Returns:
        List of fixture names (directory names in real_sessions/)
    """
    if not FIXTURES_DIR.exists():
        return []

    return [d.name for d in FIXTURES_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]


def get_fixture_path(fixture_name: str) -> Path:
    """
    Get path to a specific fixture.

    Args:
        fixture_name: Name of the fixture

    Returns:
        Path to fixture directory

    Raises:
        ValueError: If fixture doesn't exist
    """
    fixture_path = FIXTURES_DIR / fixture_name

    if not fixture_path.exists():
        available = get_available_fixtures()
        raise ValueError(
            f"Fixture '{fixture_name}' not found. Available fixtures: {', '.join(available)}"
        )

    return fixture_path


def get_fixture_files(fixture_name: str) -> dict:
    """
    Get paths to all files in a fixture.

    Args:
        fixture_name: Name of the fixture

    Returns:
        Dict mapping file types to paths (e.g., {"BRP": Path(...), "EVE": Path(...)})
    """
    fixture_path = get_fixture_path(fixture_name)

    files = {}
    for file_path in fixture_path.glob("*.edf"):
        # Extract file type from filename (e.g., "20250215_032456_BRP.edf" -> "BRP")
        parts = file_path.stem.split("_")
        if len(parts) >= 3:
            file_type = parts[2]
            files[file_type] = file_path

    return files


def load_real_session(fixture_name: str) -> Tuple[Path, dict]:
    """
    Load a real session fixture.

    Args:
        fixture_name: Name of the fixture to load

    Returns:
        Tuple of (fixture_directory_path, file_paths_dict)

    Example:
        >>> path, files = load_real_session("2025_baseline")
        >>> print(files.keys())  # ['BRP', 'PLD', 'SA2', 'EVE', 'CSL']
    """
    fixture_path = get_fixture_path(fixture_name)
    files = get_fixture_files(fixture_name)

    return fixture_path, files


def import_to_test_db(
    fixture_name: str,
    db_session: Session,
    profile_name: str = "TestProfile",
    machine_model: str = "AirSense 10",
) -> CPAPSession:
    """
    Import a fixture session to test database.

    Args:
        fixture_name: Name of the fixture to import
        db_session: SQLAlchemy database session
        profile_name: Profile name for the session (not currently used)
        machine_model: Machine model identifier (not currently used)

    Returns:
        Imported CPAPSession object from database

    Note:
        This function extracts the database path from the SQLAlchemy session
        to initialize the database and import the session using SessionImporter.

    Example:
        >>> session = import_to_test_db("2025_baseline", test_db)
        >>> print(session.session_date)
    """
    fixture_path, files = load_real_session(fixture_name)

    # Find BRP file (main data file)
    if "BRP" not in files:
        raise ValueError(f"Fixture {fixture_name} missing BRP file")

    # Extract session_id from BRP filename (e.g., "20250215_032456_BRP.edf" -> "20250215_032456")
    brp_filename = files["BRP"].stem
    session_id = "_".join(brp_filename.split("_")[:2])

    # Create device info (minimal for test fixtures)
    from oscar_mcp.models.unified import DeviceInfo

    device_info = DeviceInfo(
        manufacturer="ResMed",
        model="AirSense 10",
        serial_number="TEST_FIXTURE",
    )

    # Parse using ResMed parser's internal method
    parser = ResmedEDFParser()
    unified_session = parser._parse_session_group(
        session_id=session_id, files=files, device_info=device_info, base_path=fixture_path
    )

    # Import directly into the provided test session instead of using SessionImporter
    # which relies on the global session factory
    from oscar_mcp.database import models
    from oscar_mcp.database.importers import serialize_waveform
    import json

    # Get or create device
    device = (
        db_session.query(models.Device)
        .filter_by(serial_number=unified_session.device_info.serial_number)
        .first()
    )

    if device:
        device.manufacturer = unified_session.device_info.manufacturer
        device.model = unified_session.device_info.model
        device.firmware_version = unified_session.device_info.firmware_version
        device.hardware_version = unified_session.device_info.hardware_version
        device.product_code = unified_session.device_info.product_code
    else:
        device = models.Device(
            manufacturer=unified_session.device_info.manufacturer,
            model=unified_session.device_info.model,
            serial_number=unified_session.device_info.serial_number,
            firmware_version=unified_session.device_info.firmware_version,
            hardware_version=unified_session.device_info.hardware_version,
            product_code=unified_session.device_info.product_code,
        )
        db_session.add(device)
        db_session.flush()

    # Check if session already exists
    existing = (
        db_session.query(models.Session)
        .filter_by(device_id=device.id, device_session_id=unified_session.device_session_id)
        .first()
    )

    if existing:
        raise RuntimeError(
            f"Session already exists in database: {unified_session.device_session_id}"
        )

    # Create session
    notes_json = (
        json.dumps(unified_session.data_quality_notes)
        if unified_session.data_quality_notes
        else None
    )

    new_session = models.Session(
        device_id=device.id,
        device_session_id=unified_session.device_session_id,
        start_time=unified_session.start_time,
        end_time=unified_session.end_time,
        duration_seconds=unified_session.duration_seconds,
        therapy_mode=unified_session.settings.mode.value if unified_session.settings else None,
        import_source=unified_session.import_source,
        parser_version=unified_session.parser_version,
        data_quality_notes=notes_json,
        has_waveform_data=unified_session.has_waveform_data,
        has_event_data=unified_session.has_event_data,
        has_statistics=unified_session.has_statistics,
    )
    db_session.add(new_session)
    db_session.flush()

    # Import waveforms
    if unified_session.has_waveform_data:
        for waveform_type, waveform in unified_session.waveforms.items():
            data_blob = serialize_waveform(waveform)
            sample_count = (
                len(waveform.values) if isinstance(waveform.values, list) else len(waveform.values)
            )

            waveform_record = models.Waveform(
                session_id=new_session.id,
                waveform_type=waveform_type.value,
                sample_rate=waveform.sample_rate,
                unit=waveform.unit,
                min_value=waveform.min_value,
                max_value=waveform.max_value,
                mean_value=waveform.mean_value,
                data_blob=data_blob,
                sample_count=sample_count,
            )
            db_session.add(waveform_record)

    # Import events
    if unified_session.has_event_data:
        for event in unified_session.events:
            event_record = models.Event(
                session_id=new_session.id,
                event_type=event.event_type.value,
                start_time=event.start_time,
                duration_seconds=event.duration_seconds,
                spo2_drop=event.spo2_drop,
                peak_flow_limitation=event.peak_flow_limitation,
            )
            db_session.add(event_record)

    # Import statistics
    if unified_session.has_statistics:
        stats = unified_session.statistics
        stats_record = models.Statistics(
            session_id=new_session.id,
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
        db_session.add(stats_record)

    # Import settings
    if unified_session.settings:
        settings = unified_session.settings
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
                    session_id=new_session.id, key=key, value=str(value)
                )
                db_session.add(setting_record)

    db_session.commit()

    # Query the imported session from SQLAlchemy to return it
    cpap_session = (
        db_session.query(CPAPSession)
        .filter(CPAPSession.device_session_id == unified_session.device_session_id)
        .first()
    )

    if cpap_session is None:
        raise RuntimeError(
            f"Session was imported but not found in database: {unified_session.device_session_id}"
        )

    return cpap_session


def get_fixture_metadata(fixture_name: str) -> dict:
    """
    Get metadata about a fixture.

    Args:
        fixture_name: Name of the fixture

    Returns:
        Dict with metadata (date, file count, file types, etc.)
    """
    fixture_path, files = load_real_session(fixture_name)

    # Extract date from BRP filename if available
    session_date = None
    if "BRP" in files:
        # Filename format: YYYYMMDD_HHMMSS_BRP.edf
        filename = files["BRP"].stem
        parts = filename.split("_")
        if len(parts) >= 2:
            date_str = parts[0]
            time_str = parts[1]
            session_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:4]}:{time_str[4:6]}"

    # Calculate total size
    total_size = sum(f.stat().st_size for f in files.values())

    return {
        "name": fixture_name,
        "path": str(fixture_path),
        "session_date": session_date,
        "file_count": len(files),
        "file_types": list(files.keys()),
        "total_size_bytes": total_size,
        "total_size_kb": total_size / 1024,
    }


def list_fixtures_with_metadata() -> List[dict]:
    """
    List all available fixtures with their metadata.

    Returns:
        List of metadata dicts for each fixture
    """
    fixtures = get_available_fixtures()
    return [get_fixture_metadata(name) for name in fixtures]
