"""Database management commands."""

from __future__ import annotations

import asyncio

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import click

from snore.auth.factory import resolve_local_profile_id
from snore.cli.decorators import CliCtx, db_option, db_session, profile_scoped_command
from snore.cli.display import (
    ICON_STATS,
    console,
    print_error,
    print_footer,
    print_header,
    print_kv,
    print_subsection,
    print_success,
    print_warning,
)
from snore.constants import DEFAULT_DATABASE_PATH
from snore.database.day_manager import DayManager
from snore.database.models import Base
from snore.database.session import cleanup_database, init_database, session_scope
from snore.services.database_service import DatabaseService
from snore.utils.db_chunk import iter_id_chunks


@click.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command()
@db_option
def init(db: str | None) -> None:
    """Initialize database schema (create_all; migrations resume post-1.0)."""
    db_path = Path(db).expanduser() if db else Path(DEFAULT_DATABASE_PATH)
    db_existed = db_path.exists()

    asyncio.run(init_database(str(db_path)))

    table_names = sorted(Base.metadata.tables.keys())

    if db_existed:
        print_success(f"Database already initialized at {db_path}")
        console.print("\nVerified tables:")
    else:
        print_success(f"Created new database at {db_path}")
        console.print("\nInitialized tables:")

    for table_name in table_names:
        console.print(f"    - {table_name}")

    if db_existed:
        console.print("\nDatabase is up to date")


@db.command("stats")
@profile_scoped_command
async def db_stats(ctx: CliCtx) -> None:
    """Show database statistics."""
    from snore.database.session import get_db_path  # noqa: PLC0415

    service = DatabaseService(ctx.db, ctx.profile_id)
    stats = await service.get_stats(get_db_path())

    print_header("Database Statistics", ICON_STATS)
    print_kv("Database", str(stats.db_path), indent=0)
    print_kv("Size", f"{stats.size_mb:.1f} MB", indent=0)

    print_subsection("Row Counts")
    print_kv("Profiles", str(stats.profile_count))
    print_kv("Devices", str(stats.device_count))
    print_kv("Sessions", str(stats.session_count))
    print_kv("Days", str(stats.day_count))
    print_kv("Events", str(stats.event_count))
    print_kv("Waveforms", str(stats.waveform_count))
    print_kv("Analysis Results", str(stats.analysis_count))
    print_kv("Detected Patterns", str(stats.pattern_count))

    print_subsection("Data Coverage")
    print_kv(
        "Sessions with waveforms",
        f"{stats.sessions_with_waveforms}/{stats.session_count} ({stats.waveform_coverage_pct:.1f}%)",
    )
    print_kv(
        "Sessions with events",
        f"{stats.sessions_with_events}/{stats.session_count} ({stats.event_coverage_pct:.1f}%)",
    )
    print_kv(
        "Sessions analyzed",
        f"{stats.analysis_count}/{stats.session_count} ({stats.analysis_coverage_pct:.1f}%)",
    )

    if stats.first_session and stats.last_session:
        console.print(
            f"\nDate range: {stats.first_session:%Y-%m-%d} to {stats.last_session:%Y-%m-%d}"
        )

    print_footer()


@db.command()
@db_option
@click.confirmation_option(prompt="Are you sure you want to vacuum the database?")
def vacuum(db: str | None) -> None:
    """Optimize database (reclaim space after deletions)."""
    db_path = Path(db) if db else Path(DEFAULT_DATABASE_PATH)

    console.print("Vacuuming database...")

    async def _run() -> None:
        async with db_session(db) as session:
            service = DatabaseService(session, await resolve_local_profile_id(session))
            result = service.vacuum_sqlite(str(db_path))
            print_success(
                f"Database vacuumed successfully ({result.size_before_mb:.1f} MB → {result.size_after_mb:.1f} MB)"
            )

    asyncio.run(_run())


@db.command("cleanup-orphans")
@db_option
@click.confirmation_option(
    prompt="Remove orphaned records (events, waveforms, statistics, settings) not linked to any session?"
)
def cleanup_orphans(db: str | None) -> None:
    """Remove orphaned child records not linked to any session.

    Orphaned rows can accumulate when CASCADE delete is not enforced or after
    a database corruption event.  This command only removes rows whose
    ``session_id`` references no existing session.  Concurrent imports from
    another process may briefly block this command; rerun if it reports the
    database is locked.

    Suggest running 'snore db vacuum' afterwards to reclaim freed disk space.
    """

    async def _run() -> None:
        from snore.database.importers import SessionImporter  # noqa: PLC0415
        from snore.database.write_gate import write_gate  # noqa: PLC0415

        async with write_gate():
            async with session_scope(immediate=True) as sess:
                counts = await SessionImporter.cleanup_orphaned_records(sess)

        total = sum(counts.values())
        if total == 0:
            print_success("No orphaned records found — database is clean")
            return

        print_subsection("Deleted orphaned records")
        for table, count in counts.items():
            if count > 0:
                print_kv(table, str(count))
        print_success(f"Removed {total} orphaned record(s)")
        console.print("\nTip: run 'snore db vacuum' to reclaim freed space")

    asyncio.run(_run())


@db.command("recompute-days")
@db_option
@click.confirmation_option(
    prompt=(
        "Re-derive every Day's metrics from stored session statistics "
        "and delete Day rows no session references?"
    )
)
def recompute_days(db: str | None) -> None:
    """Recompute all Day aggregates from stored session statistics (no reparse).

    Re-runs day aggregation over the ``Statistics`` already stored for each
    session, refreshing every Day row across all devices and profiles.  Use
    after an aggregation-formula change (e.g. a new weighting) to update
    historical Day rows without re-importing raw device data.  Day rows that
    no session references are deleted rather than recomputed.

    Days are processed in chunks, each committed in its own gated write
    transaction (like ``cleanup-orphans``), so the SQLite write lock is
    released between chunks instead of held for the whole run — concurrent
    import/analysis workers are not starved.  Recomputation is idempotent, so
    unlike an all-or-nothing pass a run interrupted after some chunks have
    committed can simply be re-run: re-processed chunks yield identical values.
    """

    async def _run() -> None:
        from rich.progress import (  # noqa: PLC0415
            BarColumn,
            MofNCompleteColumn,
            Progress,
            TextColumn,
        )
        from sqlalchemy import select  # noqa: PLC0415

        from snore.database.models import Day  # noqa: PLC0415
        from snore.database.session import (  # noqa: PLC0415
            init_database,
            session_scope,
        )
        from snore.database.write_gate import write_gate  # noqa: PLC0415

        await init_database(str(Path(db).expanduser()) if db else None)

        async with session_scope() as session:
            day_ids = (
                (await session.execute(select(Day.id).order_by(Day.id))).scalars().all()
            )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Recomputing days", total=len(day_ids))
            pruned = 0
            for chunk in iter_id_chunks(day_ids):
                async with write_gate(), session_scope(immediate=True) as session:
                    days = (
                        (await session.execute(select(Day).where(Day.id.in_(chunk))))
                        .scalars()
                        .all()
                    )
                    for day in days:
                        if not await DayManager.recalculate_day(day, session):
                            pruned += 1
                progress.update(task, advance=len(chunk))

        summary = f"Recomputed {len(day_ids) - pruned} day(s) from session statistics"
        if pruned:
            summary += f"; pruned {pruned} orphaned day(s)"
        print_success(summary)

    asyncio.run(_run())


@db.command()
@db_option
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def drop(db: str | None, force: bool) -> None:
    """Drop database (permanently delete all CPAP data)."""
    db_path = Path(db).expanduser() if db else Path(DEFAULT_DATABASE_PATH)

    if not db_path.exists():
        console.print(f"Database does not exist at {db_path}")
        return

    try:
        asyncio.run(init_database(str(db_path)))

        async def _show_stats() -> None:
            async with session_scope() as session:
                profile_id = await resolve_local_profile_id(session)
                service = DatabaseService(session, profile_id)
                stats = await service.get_stats(str(db_path))

                console.print(f"\nDatabase: {db_path}")
                console.print(f"Size: {stats.size_mb:.1f} MB")
                console.print(f"Devices: {stats.device_count}")
                console.print(f"Sessions: {stats.session_count}")
                console.print(f"Events: {stats.event_count:,}")

                if stats.first_session and stats.last_session:
                    console.print(
                        f"Date range: {stats.first_session:%Y-%m-%d} to {stats.last_session:%Y-%m-%d}"
                    )

        asyncio.run(_show_stats())

    except Exception as e:
        print_warning(f"Could not read database stats: {e}")

    if not force:
        print_warning("WARNING: This will permanently delete all CPAP data!")
        if not click.confirm(
            "Are you sure you want to drop the database?", default=False
        ):
            console.print("Database drop cancelled")
            return

    try:
        asyncio.run(cleanup_database())
    except Exception as e:
        print_warning(f"Warning during cleanup: {e}")

    try:
        if db_path.exists():
            db_path.unlink()
            print_success(f"Deleted database: {db_path}")

        for ext in ["-wal", "-shm"]:
            wal_file = Path(str(db_path) + ext)
            if wal_file.exists():
                wal_file.unlink()
                print_success(f"Deleted: {wal_file.name}")

        console.print("\nDatabase dropped successfully")

    except Exception as e:
        raise click.ClickException(f"Error dropping database: {e}") from e


@db.command("scrub-demo")
@db_option
@click.option(
    "--source-profile",
    "source_profile_id",
    required=True,
    type=int,
    help="Profile ID to copy data from",
)
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def scrub_demo(db: str | None, source_profile_id: int, yes: bool) -> None:
    """Populate the demo account from a source profile.

    Reproducible, idempotent scrub: running twice replaces prior demo data.

    PII scrubs applied:
    - profiles: name='Demo', username/first_name/last_name -> None,
      date_of_birth -> None, height_cm -> None, settings -> {}
    - devices: serial_number -> 'DEMO-NNN', firmware/hardware/product_code -> None;
      manufacturer and model are kept (device product names, not PII).
    - sessions: import_source -> 'demo'
    - settings: all keys and values are copied unchanged. Device settings are
      an enum-like fixed set (pressure, ramp, EPR, etc.) — therapy configuration
      data, not identity PII. Preserving them lets demo viewers see realistic
      therapy settings.
    - detected_patterns: notes -> None
    - Date rotation: all dates/datetimes shifted by a consistent whole-day
      offset so the most recent source day lands exactly 7 days before today.
    - Raw backup files are NOT copied (filesystem isolation).
    """
    if not yes:
        click.confirm(
            f"Scrub demo data from source profile {source_profile_id}? "
            "This will replace all existing demo data.",
            abort=True,
        )

    async def _run() -> None:
        async with db_session(db) as session:
            await _do_scrub_demo(session, source_profile_id)

    asyncio.run(_run())


# Columns the demo scrub must NOT copy verbatim: identity/FK columns get
# remapped values, ``date`` is shifted, and audit instants regenerate via
# defaults.  test_demo imports these so its column guards stay in lockstep
# with the clone. Statistics has no surrogate ``id`` — session_id is its PK.
_DAY_CLONE_EXCLUDED_COLUMNS = ("id", "device_id", "date", "created_at", "updated_at")
_STATS_CLONE_EXCLUDED_COLUMNS = ("session_id",)


async def _do_scrub_demo(session: Any, source_profile_id: int) -> None:
    """Async implementation of the scrub-demo command.

    Algorithm (single transaction — atomicity beats WAL-peak concerns at this scale;
    peak disk usage during scrub is roughly 2–3× the dataset size while old and new
    data co-exist before the cascade-delete of existing demo devices completes):

    Step 1 — verify source profile exists.
    Step 2 — find-or-create demo user and profile (via DemoService).
    Step 3 — compute whole-day date offset (most recent source day → today-7).
    Step 4 — copy device/day/session chain with PII scrubs and date shift:
              - waveforms copied via INSERT...SELECT at the SQL layer so blobs never
                enter Python memory;
              - events, statistics, settings, analysis_results fetched in one batch
                per device (WHERE session_id IN (...)) to avoid per-session round-trips;
              - detected_patterns and breaths batched per analysis result batch.
    Step 5 — post-scrub integrity checks (ClickException, not bare assert).
    Step 6 — print summary.

    To revoke a demo session: bump ``users.session_version`` for the demo user;
    all outstanding session cookies will be invalidated on the next request.

    See the Click command docstring for full semantics and PII scrub policy.
    """
    from datetime import datetime  # noqa: PLC0415

    from sqlalchemy import select, text  # noqa: PLC0415
    from sqlalchemy.ext.asyncio import AsyncSession  # noqa: PLC0415

    from snore.database.models import (  # noqa: PLC0415
        AnalysisResult,
        Breath,
        Day,
        DetectedPattern,
        Device,
        Event,
        Profile,
        Setting,
        Statistics,
    )
    from snore.database.models import (  # noqa: PLC0415
        Session as DbSession,
    )
    from snore.services.demo_service import DemoService  # noqa: PLC0415

    db: AsyncSession = session

    # ---- 1. Verify source profile exists ----
    source_profile = await db.get(Profile, source_profile_id)
    if source_profile is None:
        raise click.ClickException(f"Source profile {source_profile_id} not found")

    # ---- 2. Find-or-create demo user + profile ----
    demo_user, demo_profile, created = await DemoService(db).ensure_user_and_profile()
    if created:
        console.print(
            f"Created demo user (id={demo_user.id}), profile id={demo_profile.id}"
        )
    else:
        console.print(
            f"Using existing demo user (id={demo_user.id}), profile id={demo_profile.id}"
        )

    # ---- 3. Compute date shift ----
    # Find the most recent date across all source days.
    max_date_q = (
        select(Day.date)
        .join(Device, Day.device_id == Device.id)
        .where(Device.profile_id == source_profile_id)
        .order_by(Day.date.desc())
        .limit(1)
    )
    max_date_row = (await db.execute(max_date_q)).scalars().first()

    if max_date_row is None:
        console.print("[yellow]Source profile has no days — nothing to copy.[/yellow]")
        return

    max_date: date = max_date_row
    today = date.today()
    target_date = today - timedelta(days=7)
    day_offset = target_date - max_date
    console.print(
        f"Date shift: {day_offset.days:+d} days "
        f"(most recent source day {max_date} -> {target_date})"
    )

    def shift_date(d: date | None) -> date | None:
        return d + day_offset if d is not None else None

    def shift_dt(dt: datetime | None) -> datetime | None:
        if dt is None:
            return None
        return dt + day_offset

    # ---- 4. Copy device/day/session chain ----
    source_dev_q = select(Device).where(Device.profile_id == source_profile_id)
    source_devices = (await db.execute(source_dev_q)).scalars().all()

    counts: dict[str, int] = {
        "devices": 0,
        "days": 0,
        "sessions": 0,
        "waveforms": 0,
        "events": 0,
        "statistics": 0,
        "settings": 0,
        "analysis_results": 0,
        "detected_patterns": 0,
        "breaths": 0,
    }

    # Map source device id → demo device id (needed for session.device_id).
    device_id_map: dict[int, int] = {}
    # Map source day id → demo day id (needed for session.day_id composite FK).
    day_id_map: dict[int, int] = {}
    # Map source session id → demo session id (needed for child table copies).
    src_to_demo_session: dict[int, int] = {}

    # --- Phase A: devices and days ---
    for dev_idx, src_dev in enumerate(source_devices):
        demo_serial = f"DEMO-{dev_idx + 1:03d}"
        demo_dev = Device(
            profile_id=demo_profile.id,
            manufacturer=src_dev.manufacturer,
            model=src_dev.model,
            serial_number=demo_serial,
            firmware_version=None,
            hardware_version=None,
            product_code=None,
        )
        db.add(demo_dev)
        await db.flush()
        device_id_map[src_dev.id] = demo_dev.id
        counts["devices"] += 1

        days_q = select(Day).where(Day.device_id == src_dev.id)
        src_days = (await db.execute(days_q)).scalars().all()
        for src_day in src_days:
            # Reflect all aggregate columns so new metrics are never silently
            # dropped (test_demo asserts every copied column stays numeric).
            day_payload = {
                c.name: getattr(src_day, c.name)
                for c in Day.__table__.columns
                if c.name not in _DAY_CLONE_EXCLUDED_COLUMNS
            }
            demo_day = Day(
                device_id=demo_dev.id,
                date=shift_date(src_day.date),
                **day_payload,
            )
            db.add(demo_day)
            await db.flush()
            day_id_map[src_day.id] = demo_day.id
            counts["days"] += 1

    # --- Phase B: sessions ---
    # Fetch all source sessions in one query per device, create demo sessions,
    # and copy waveforms via INSERT...SELECT (SQL-layer; blobs never enter Python).
    source_dev_ids = [d.id for d in source_devices]
    all_sessions_q = select(DbSession).where(DbSession.device_id.in_(source_dev_ids))
    all_src_sessions = (await db.execute(all_sessions_q)).scalars().all()

    for src_sess in all_src_sessions:
        demo_device_id = device_id_map[src_sess.device_id]
        demo_day_id = day_id_map.get(src_sess.day_id) if src_sess.day_id else None

        demo_sess = DbSession(
            device_id=demo_device_id,
            day_id=demo_day_id,
            device_session_id=src_sess.device_session_id,
            start_time=shift_dt(src_sess.start_time),
            end_time=shift_dt(src_sess.end_time),
            duration_seconds=src_sess.duration_seconds,
            therapy_mode=src_sess.therapy_mode,
            import_source="demo",
            parser_version=src_sess.parser_version,
            data_quality_notes=src_sess.data_quality_notes,
            mask_on_segments=src_sess.mask_on_segments,
            has_waveform_data=src_sess.has_waveform_data,
            has_event_data=src_sess.has_event_data,
            has_statistics=src_sess.has_statistics,
            enabled=src_sess.enabled,
        )
        db.add(demo_sess)
        await db.flush()
        src_to_demo_session[src_sess.id] = demo_sess.id
        counts["sessions"] += 1

        # Waveforms have no date columns and no PII — copy at the SQL layer
        # so LargeBinary blobs never enter Python memory.
        wf_result = await db.execute(
            text(
                "INSERT INTO waveforms "
                "(session_id, waveform_type, sample_rate, unit, "
                "min_value, max_value, mean_value, data_blob, sample_count) "
                "SELECT :new_sid, waveform_type, sample_rate, unit, "
                "min_value, max_value, mean_value, data_blob, sample_count "
                "FROM waveforms WHERE session_id = :old_sid"
            ),
            {"new_sid": demo_sess.id, "old_sid": src_sess.id},
        )
        counts["waveforms"] += wf_result.rowcount  # type: ignore[attr-defined]

    # --- Phase C: batch-fetch scalar child tables, insert with session mapping ---
    source_session_ids = list(src_to_demo_session.keys())

    if source_session_ids:
        # Events — date-shifted.
        for chunk in iter_id_chunks(source_session_ids):
            events_q = select(Event).where(Event.session_id.in_(chunk))
            for src_evt in (await db.execute(events_q)).scalars().all():
                db.add(
                    Event(
                        session_id=src_to_demo_session[src_evt.session_id],
                        event_type=src_evt.event_type,
                        start_time=shift_dt(src_evt.start_time),
                        duration_seconds=src_evt.duration_seconds,
                        spo2_drop=src_evt.spo2_drop,
                        peak_flow_limitation=src_evt.peak_flow_limitation,
                    )
                )
                counts["events"] += 1

        # Statistics — no date/PII columns.
        for chunk in iter_id_chunks(source_session_ids):
            stats_q = select(Statistics).where(Statistics.session_id.in_(chunk))
            for src_stats in (await db.execute(stats_q)).scalars().all():
                # Reflect all metric columns so new metrics are never silently
                # dropped (test_demo asserts every copied column stays numeric).
                stats_payload = {
                    c.name: getattr(src_stats, c.name)
                    for c in Statistics.__table__.columns
                    if c.name not in _STATS_CLONE_EXCLUDED_COLUMNS
                }
                db.add(
                    Statistics(
                        session_id=src_to_demo_session[src_stats.session_id],
                        **stats_payload,
                    )
                )
                counts["statistics"] += 1

        # Settings — device therapy config (enum-like constants, not identity PII).
        for chunk in iter_id_chunks(source_session_ids):
            settings_q = select(Setting).where(Setting.session_id.in_(chunk))
            for src_setting in (await db.execute(settings_q)).scalars().all():
                db.add(
                    Setting(
                        session_id=src_to_demo_session[src_setting.session_id],
                        key=src_setting.key,
                        value=src_setting.value,
                    )
                )
                counts["settings"] += 1

        # --- Phase D: analysis results ---
        src_to_demo_ar: dict[int, int] = {}
        for chunk in iter_id_chunks(source_session_ids):
            ar_q = select(AnalysisResult).where(AnalysisResult.session_id.in_(chunk))
            for src_ar in (await db.execute(ar_q)).scalars().all():
                demo_ar = AnalysisResult(
                    session_id=src_to_demo_session[src_ar.session_id],
                    timestamp_start=shift_dt(src_ar.timestamp_start),
                    timestamp_end=shift_dt(src_ar.timestamp_end),
                    programmatic_result_json=src_ar.programmatic_result_json,
                    processing_time_ms=src_ar.processing_time_ms,
                    engine_versions_json=src_ar.engine_versions_json,
                )
                db.add(demo_ar)
                await db.flush()
                src_to_demo_ar[src_ar.id] = demo_ar.id
                counts["analysis_results"] += 1

        # --- Phase E: detected_patterns and breaths (batched by analysis result IDs) ---
        source_ar_ids = list(src_to_demo_ar.keys())
        if source_ar_ids:
            for chunk in iter_id_chunks(source_ar_ids):
                patterns_q = select(DetectedPattern).where(
                    DetectedPattern.analysis_result_id.in_(chunk)
                )
                for src_pat in (await db.execute(patterns_q)).scalars().all():
                    db.add(
                        DetectedPattern(
                            analysis_result_id=src_to_demo_ar[
                                src_pat.analysis_result_id
                            ],
                            pattern_id=src_pat.pattern_id,
                            start_time=shift_dt(src_pat.start_time),
                            duration=src_pat.duration,
                            confidence=src_pat.confidence,
                            detected_by=src_pat.detected_by,
                            metrics_json=src_pat.metrics_json,
                            notes=None,  # PII scrub
                        )
                    )
                    counts["detected_patterns"] += 1

            # Breaths use session-relative offsets (seconds) — no date shift needed.
            # Both FKs (analysis_result_id, session_id) must be remapped.
            for chunk in iter_id_chunks(source_ar_ids):
                breaths_q = select(Breath).where(Breath.analysis_result_id.in_(chunk))
                for src_breath in (await db.execute(breaths_q)).scalars().all():
                    db.add(
                        Breath(
                            analysis_result_id=src_to_demo_ar[
                                src_breath.analysis_result_id
                            ],
                            session_id=src_to_demo_session[src_breath.session_id],
                            breath_number=src_breath.breath_number,
                            start_offset_s=src_breath.start_offset_s,
                            end_offset_s=src_breath.end_offset_s,
                            inspiration_time_s=src_breath.inspiration_time_s,
                            expiration_time_s=src_breath.expiration_time_s,
                            total_time_s=src_breath.total_time_s,
                            i_e_ratio=src_breath.i_e_ratio,
                            duty_cycle=src_breath.duty_cycle,
                            peak_flow_lpm=src_breath.peak_flow_lpm,
                            peak_exp_flow_lpm=src_breath.peak_exp_flow_lpm,
                            tidal_volume_ml=src_breath.tidal_volume_ml,
                            respiratory_rate_rolling=(
                                src_breath.respiratory_rate_rolling
                            ),
                            flatness_index=src_breath.flatness_index,
                            mid_insp_flattening=src_breath.mid_insp_flattening,
                            flow_class=src_breath.flow_class,
                            flow_confidence=src_breath.flow_confidence,
                            is_recovery_breath=src_breath.is_recovery_breath,
                            inferred_trigger_type=src_breath.inferred_trigger_type,
                            trigger_confidence=src_breath.trigger_confidence,
                            inferred_cycle_type=src_breath.inferred_cycle_type,
                            cycle_confidence=src_breath.cycle_confidence,
                            trigger_cycle_applicable=(
                                src_breath.trigger_cycle_applicable
                            ),
                            trigger_cycle_reason=src_breath.trigger_cycle_reason,
                            leak_valid=src_breath.leak_valid,
                            leak_valid_reason=src_breath.leak_valid_reason,
                            ramp_active=src_breath.ramp_active,
                            ramp_active_reason=src_breath.ramp_active_reason,
                            mask_off=src_breath.mask_off,
                            mask_off_reason=src_breath.mask_off_reason,
                        )
                    )
                    counts["breaths"] += 1

    await db.flush()

    # ---- 5. Post-scrub integrity checks ----
    # Using explicit ClickException (not bare assert) so checks survive python -O.

    # a. No source serial numbers remain in demo devices.
    assert_device_stmt = select(Device).where(Device.profile_id == demo_profile.id)
    demo_devices_check = (await db.execute(assert_device_stmt)).scalars().all()
    source_serials = {d.serial_number for d in source_devices}
    for dev in demo_devices_check:
        if dev.serial_number in source_serials:
            raise click.ClickException(
                f"Integrity check failed: demo device still has source serial "
                f"{dev.serial_number!r}"
            )

    # b. Demo profile has no PII fields.
    await db.refresh(demo_profile)
    for field, val in [
        ("first_name", demo_profile.first_name),
        ("last_name", demo_profile.last_name),
        ("date_of_birth", demo_profile.date_of_birth),
        ("username", demo_profile.username),
        ("height_cm", demo_profile.height_cm),
    ]:
        if val is not None:
            raise click.ClickException(
                f"Integrity check failed: demo profile.{field} is not None"
            )
    if demo_profile.settings:
        raise click.ClickException(
            "Integrity check failed: demo profile.settings is not empty"
        )

    # c. No detected_patterns.notes survive the scrub.
    assert_pattern_stmt = (
        select(DetectedPattern)
        .join(AnalysisResult, DetectedPattern.analysis_result_id == AnalysisResult.id)
        .join(DbSession, AnalysisResult.session_id == DbSession.id)
        .join(Device, DbSession.device_id == Device.id)
        .where(Device.profile_id == demo_profile.id)
        .where(DetectedPattern.notes.is_not(None))
    )
    bad_patterns = (await db.execute(assert_pattern_stmt)).scalars().all()
    if bad_patterns:
        raise click.ClickException(
            f"Integrity check failed: {len(bad_patterns)} demo detected_patterns "
            "still have notes"
        )

    # d. No raw backup files for demo profile (scrub must never copy filesystem data).
    from snore.constants import DEFAULT_RAW_BACKUP_DIR  # noqa: PLC0415

    raw_dir = Path(DEFAULT_RAW_BACKUP_DIR) / str(demo_profile.id)
    if raw_dir.exists():
        raise click.ClickException(
            f"Integrity check failed: raw backup dir exists for demo profile: {raw_dir}"
        )

    # ---- 6. Summary ----
    print_success("Scrub-demo complete")
    print_subsection("Rows copied")
    for table, count in counts.items():
        print_kv(table, str(count))
    print_kv("day offset", f"{day_offset.days:+d} days")


@db.command("purge-quarantine")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt")
def db_purge_quarantine(yes: bool) -> None:
    """Purge all quarantined profile raw data (offline operator command).

    Quarantined directories are left behind when a profile deletion saga is
    interrupted (e.g. crash after rename but before cascade-delete).  This
    command removes them permanently.

    The API server must not be running when this command is used.
    """
    if not yes:
        click.confirm(
            "Purge all quarantined profile raw data? This cannot be undone. "
            "Ensure the API server is not running.",
            abort=True,
        )

    from snore.services.profile_service import DeletionSaga  # noqa: PLC0415
    from snore.services.writer_lease import WriterLeaseError  # noqa: PLC0415

    saga = DeletionSaga()
    try:
        saga.purge_quarantine()
        print_success("Quarantine purged successfully")
    except WriterLeaseError as e:
        print_error(str(e))
