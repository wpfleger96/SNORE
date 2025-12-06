"""
Command-line interface for SNORE.

Provides commands for importing CPAP data, querying sessions, and database management.
"""

import logging
import sys

from datetime import datetime
from pathlib import Path
from typing import Any

import click

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from snore.analysis.service import AnalysisService
from snore.config import (
    get_config_path,
    get_default_profile,
    set_default_profile,
    unset_default_profile,
)
from snore.constants import (
    DEFAULT_LIST_SESSIONS_LIMIT,
    EVENT_TYPE_CENTRAL_APNEA,
    EVENT_TYPE_CLEAR_AIRWAY,
    EVENT_TYPE_HYPOPNEA,
    EVENT_TYPE_MIXED_APNEA,
    EVENT_TYPE_OBSTRUCTIVE_APNEA,
    EVENT_TYPE_RERA,
)
from snore.database import models
from snore.database.importers import SessionImporter
from snore.database.session import init_database, session_scope
from snore.parsers.register_all import register_all_parsers
from snore.parsers.registry import parser_registry

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def ensure_profile(username: str) -> int:
    """Get or create profile by username, return profile_id."""
    with session_scope() as session:
        profile = session.query(models.Profile).filter_by(username=username).first()
        if not profile:
            profile = models.Profile(
                username=username, settings={"day_split_time": "12:00:00"}
            )
            session.add(profile)
            session.flush()
        return profile.id


def resolve_profile(explicit_profile: str | None, db_session: Session) -> str:
    """
    Resolve profile using precedence: CLI > config > auto-detect.

    Args:
        explicit_profile: Value from --profile flag (None if not provided)
        db_session: Active database session

    Returns:
        Username to use

    Raises:
        click.ClickException: If profile cannot be resolved
    """
    if explicit_profile:
        return explicit_profile

    config_profile = get_default_profile()
    if config_profile:
        prof = (
            db_session.query(models.Profile).filter_by(username=config_profile).first()
        )
        if prof:
            return config_profile
        else:
            click.echo(
                f"Warning: Default profile '{config_profile}' not found in database.",
                err=True,
            )
            click.echo(
                "Update with: snore config set-default-profile <name>",
                err=True,
            )

    profiles = db_session.query(models.Profile).all()
    if len(profiles) == 1:
        username: str = profiles[0].username
        return username

    if len(profiles) == 0:
        raise click.ClickException(
            "No profiles found. Import data first: snore import-data <path>"
        )
    else:
        profile_list = ", ".join([p.username for p in profiles])
        raise click.ClickException(
            f"Multiple profiles found ({profile_list}). "
            "Specify --profile <name> or set default: "
            "snore config set-default-profile <name>"
        )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose: bool) -> None:
    """SNORE: CPAP Data Management Tool"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Re-import existing sessions")
@click.option(
    "--db", type=click.Path(), help="Database path (default: ~/snore/snore.db)"
)
@click.option("--limit", "-n", type=int, help="Limit to first N sessions")
@click.option(
    "--sort-by",
    type=click.Choice(["date-asc", "date-desc", "filesystem"]),
    default="filesystem",
    help="Session sort order (default: filesystem)",
)
@click.option(
    "--date-from",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Import sessions from this date (YYYY-MM-DD)",
)
@click.option(
    "--date-to",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Import sessions up to this date (YYYY-MM-DD)",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be imported without importing"
)
def import_data(
    path: str,
    force: bool,
    db: str | None,
    limit: int | None,
    sort_by: str,
    date_from: datetime | None,
    date_to: datetime | None,
    dry_run: bool,
) -> int:
    """Import CPAP data from device SD card or directory."""
    data_path = Path(path)

    # Register parsers
    register_all_parsers()

    # Auto-detect all available parsers and data roots
    click.echo(f"📂 Scanning {data_path}...")
    results = parser_registry.detect_all_parsers(data_path)

    if not results:
        click.echo("❌ Error: No compatible parser found for this data", err=True)
        click.echo("\nSupported devices:")
        for p in parser_registry.list_parsers():
            click.echo(f"  - {p.manufacturer}: {p.parser_id}")
        return 1

    # Handle multiple data roots (e.g., multiple OSCAR profiles)
    selected_results = []
    if len(results) > 1:
        click.echo(f"\nFound {len(results)} data sources:\n")
        for i, (parser, detection) in enumerate(results, 1):
            meta = detection.metadata or {}
            if meta.get("profile_name"):
                desc = f"{meta['profile_name']} ({meta.get('structure_type', 'unknown').replace('_', ' ')})"
            else:
                structure = meta.get("structure_type", "raw SD card").replace("_", " ")
                serial = meta.get("device_serial", "unknown")
                desc = f"{structure} - S/N: {serial}"

            click.echo(f"  {i}. {parser.manufacturer} - {desc}")
            if meta.get("data_root"):
                click.echo(f"     Path: {meta['data_root']}")

        click.echo(f"  {len(results) + 1}. Import all")

        choice = click.prompt("\nSelect which to import", type=int, default=1)

        if choice == len(results) + 1:
            selected_results = results
        elif 1 <= choice <= len(results):
            selected_results = [results[choice - 1]]
        else:
            click.echo(f"❌ Invalid choice: {choice}", err=True)
            return 1
    else:
        selected_results = results

    # Initialize database once before processing any data sources
    init_database(str(Path(db)) if db else None)

    # Clean up any orphaned records from previous incomplete operations
    with session_scope() as session:
        orphaned_count = SessionImporter.cleanup_orphaned_records(session)
        if orphaned_count > 0:
            click.echo(f"⚠️  Cleaned up {orphaned_count} orphaned records from database")
    # Process each selected data source
    total_imported = 0
    total_skipped = 0
    total_failed = 0

    for parser, detection in selected_results:
        meta = detection.metadata or {}
        source_desc = (
            meta.get("profile_name") or f"S/N {meta.get('device_serial', 'unknown')}"
        )

        if len(selected_results) > 1:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"Processing: {source_desc}")
            click.echo(f"{'=' * 60}")

        click.echo(f"✓ Detected: {parser.manufacturer} ({parser.parser_id})")
        click.echo(
            f"  Structure: {meta.get('structure_type', 'unknown').replace('_', ' ')}"
        )
        if meta.get("data_root"):
            click.echo(f"  Data root: {meta['data_root']}")

        # Handle profile if present
        profile_id = None
        if meta.get("profile_name"):
            profile_id = ensure_profile(meta["profile_name"])
            click.echo(f"  Profile: {meta['profile_name']}")

        # Format date parameters for parser
        date_from_str = date_from.strftime("%Y-%m-%d") if date_from else None
        date_to_str = date_to.strftime("%Y-%m-%d") if date_to else None

        # Show filter summary if any filters are active
        if limit or date_from or date_to or sort_by != "filesystem":
            click.echo("\n📋 Import filters:")
            if limit:
                click.echo(f"  • Limit: {limit} sessions")
            if sort_by != "filesystem":
                order_desc = "oldest first" if sort_by == "date-asc" else "newest first"
                click.echo(f"  • Sort: {order_desc}")
            if date_from:
                click.echo(f"  • From: {date_from:%Y-%m-%d}")
            if date_to:
                click.echo(f"  • To: {date_to:%Y-%m-%d}")

        # Parse sessions
        click.echo("\n📋 Parsing sessions...")
        try:
            sessions = list(
                parser.parse_sessions(
                    data_path,
                    date_from=date_from_str,
                    date_to=date_to_str,
                    limit=limit,
                    sort_by=sort_by if sort_by != "filesystem" else None,
                )
            )
        except Exception as e:
            click.echo(f"❌ Error parsing sessions: {e}", err=True)
            if logging.getLogger().level == logging.DEBUG:
                raise
            if len(selected_results) > 1:
                continue
            return 1

        if not sessions:
            click.echo("⚠️  No sessions found")
            if len(selected_results) > 1:
                continue
            return 0

        click.echo(f"✓ Found {len(sessions)} sessions")

        # Dry-run mode: show what would be imported
        if dry_run:
            click.echo("\n🔍 DRY RUN MODE - No data will be imported\n")
            click.echo(
                f"{'Date':<12} {'Time':<8} {'Duration':<10} {'AHI':<6} {'Events':<8}"
            )
            click.echo("=" * 55)

            total_duration = 0.0
            total_events = 0

            # Sort sessions by date descending for display
            sorted_sessions = sorted(sessions, key=lambda s: s.start_time, reverse=True)

            for unified_session in sorted_sessions:
                duration_hours = (
                    unified_session.duration_seconds / 3600
                    if unified_session.duration_seconds
                    else 0
                )
                total_duration += duration_hours

                # Count events
                num_events = (
                    len(unified_session.events) if unified_session.events else 0
                )
                total_events += num_events

                # Get AHI from statistics if available
                ahi_str = "N/A"
                if (
                    hasattr(unified_session, "statistics")
                    and unified_session.statistics
                ):
                    if unified_session.statistics.ahi is not None:
                        ahi_str = f"{unified_session.statistics.ahi:.1f}"

                click.echo(
                    f"{unified_session.start_time:%Y-%m-%d}   {unified_session.start_time:%H:%M:%S}  "
                    f"{duration_hours:>6.1f}h    "
                    f"{ahi_str:>5}  "
                    f"{num_events:>6}"
                )

            click.echo("=" * 55)
            click.echo("\n📊 Summary:")
            click.echo(f"  • Total sessions: {len(sessions)}")
            click.echo(f"  • Total duration: {total_duration:.1f} hours")
            click.echo(f"  • Total events: {total_events}")
            if sessions:
                # Calculate actual date range using min/max
                first_date = min(s.start_time for s in sessions)
                last_date = max(s.start_time for s in sessions)
                click.echo(
                    f"  • Date range: {first_date:%Y-%m-%d} to {last_date:%Y-%m-%d}"
                )
            if len(selected_results) == 1:
                click.echo("\n✓ Dry run complete. Use without --dry-run to import.")
            continue

        # Create importer with profile_id
        importer = SessionImporter(profile_id=profile_id)

        # Import sessions
        imported = 0
        skipped = 0
        failed = 0

        with click.progressbar(
            sessions,
            label="Importing sessions",
            show_pos=True,
            item_show_func=lambda s: f"{s.start_time:%Y-%m-%d}" if s else "",
        ) as bar:
            for unified_session in bar:
                try:
                    if importer.import_session(unified_session, force=force):
                        imported += 1
                    else:
                        skipped += 1
                except Exception as e:
                    failed += 1
                    logger.error(
                        f"Failed to import session {unified_session.device_session_id}: {e}"
                    )

        # Update totals
        total_imported += imported
        total_skipped += skipped
        total_failed += failed

        # Show per-source summary if processing multiple sources
        if len(selected_results) > 1:
            click.echo(f"\n{'=' * 50}")
            click.echo(f"📊 Summary for {source_desc}")
            click.echo(f"{'=' * 50}")
            click.echo(f"✓ Imported: {imported} sessions")
            if skipped > 0:
                click.echo(f"⊝ Skipped:  {skipped} sessions")
            if failed > 0:
                click.echo(f"❌ Failed:   {failed} sessions")

    # Final summary for dry-run mode
    if dry_run and len(selected_results) > 1:
        click.echo(f"\n{'=' * 50}")
        click.echo("📊 Overall Dry Run Summary")
        click.echo(f"{'=' * 50}")
        click.echo(f"✓ Total data sources: {len(selected_results)}")
        click.echo("\n✓ Dry run complete. Use without --dry-run to import.")
        return 0
    elif dry_run:
        return 0

    # Final summary for actual imports
    click.echo(f"\n{'=' * 50}")
    click.echo("📊 Overall Import Summary")
    click.echo(f"{'=' * 50}")
    click.echo(f"✓ Imported: {total_imported} sessions")
    if total_skipped > 0:
        click.echo(
            f"⊝ Skipped:  {total_skipped} sessions (already exist, use --force to re-import)"
        )
    if total_failed > 0:
        click.echo(f"❌ Failed:   {total_failed} sessions")

    click.echo(f"{'=' * 50}")

    if total_failed > 0:
        return 1
    return 0


@cli.command("list-profiles")
@click.option("--db", type=click.Path(), help="Database path")
def list_profiles(db: str | None) -> None:
    """List all available profiles in the database."""
    if db:
        init_database(str(Path(db)))
    else:
        init_database()

    with session_scope() as session:
        profiles = session.query(models.Profile).all()

        if not profiles:
            click.echo("No profiles found in database")
            return

        click.echo("\nAvailable Profiles:\n")

        for profile in profiles:
            click.echo(f"Profile: {profile.username}")

            if profile.first_name or profile.last_name:
                name_parts = [profile.first_name, profile.last_name]
                full_name = " ".join(part for part in name_parts if part)
                click.echo(f"  Name: {full_name}")

            session_count = (
                session.query(models.Session)
                .join(models.Day)
                .filter(models.Day.profile_id == profile.id)
                .count()
            )

            day_count = (
                session.query(models.Day)
                .filter(models.Day.profile_id == profile.id)
                .count()
            )

            click.echo(f"  Sessions: {session_count}")
            click.echo(f"  Days with data: {day_count}")

            if day_count > 0:
                days = (
                    session.query(models.Day)
                    .filter(models.Day.profile_id == profile.id)
                    .order_by(models.Day.date)
                    .all()
                )
                first_date = days[0].date
                last_date = days[-1].date
                click.echo(f"  Date range: {first_date} to {last_date}")

            click.echo()


@cli.command("list-sessions")
@click.option(
    "--from-date",
    "from_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--to-date",
    "to_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--limit",
    type=int,
    default=DEFAULT_LIST_SESSIONS_LIMIT,
    help="Max sessions to show (use 0 for all)",
)
@click.option("--db", type=click.Path(), help="Database path")
def list_sessions(
    from_date: datetime | None, to_date: datetime | None, limit: int, db: str | None
) -> None:
    """List imported sessions."""
    if db:
        init_database(str(Path(db)))
    else:
        init_database()

    with session_scope() as session:
        # Build base WHERE clause for both count and data queries
        where_clause = "WHERE 1=1"
        params: dict[str, Any] = {}

        if from_date:
            where_clause += " AND sessions.start_time >= :from_date"
            params["from_date"] = from_date

        if to_date:
            where_clause += " AND sessions.start_time <= :to_date"
            params["to_date"] = to_date

        # Count total matching sessions
        count_query = f"""
            SELECT COUNT(*) as total
            FROM sessions
            JOIN devices ON sessions.device_id = devices.id
            LEFT JOIN profiles ON devices.profile_id = profiles.id
            {where_clause}
        """
        total_sessions = session.execute(text(count_query), params).scalar() or 0

        # Build main query with profile information
        query = f"""
            SELECT
                sessions.id as session_id,
                sessions.start_time,
                sessions.duration_seconds,
                devices.manufacturer,
                devices.model,
                profiles.username as profile_name
            FROM sessions
            JOIN devices ON sessions.device_id = devices.id
            LEFT JOIN profiles ON devices.profile_id = profiles.id
            {where_clause}
            ORDER BY sessions.start_time DESC
        """

        # Add limit if specified (0 means no limit)
        if limit > 0:
            query += " LIMIT :limit"
            params["limit"] = limit

        # Execute query
        result = session.execute(text(query), params)
        sessions = result.fetchall()

        if not sessions:
            click.echo("No sessions found")
            return

        # Display sessions
        click.echo(
            f"\n{'Date':<12} {'ID':<6} {'Time':<8} {'Duration':<10} {'Profile':<15} {'Device':<20} {'AHI':<6}"
        )
        click.echo("=" * 91)

        for sess in sessions:
            start = (
                datetime.fromisoformat(sess.start_time)
                if isinstance(sess.start_time, str)
                else sess.start_time
            )
            duration_hours = (
                sess.duration_seconds / 3600 if sess.duration_seconds else 0
            )
            device_name = f"{sess.manufacturer} {sess.model}"
            profile_name = sess.profile_name or "N/A"

            # Get AHI from statistics if available
            stats_result = session.execute(
                text("SELECT ahi FROM statistics WHERE session_id = :session_id"),
                {"session_id": sess.session_id},
            )
            stats = stats_result.fetchone()
            ahi = f"{stats.ahi:.1f}" if stats and stats.ahi is not None else "N/A"

            click.echo(
                f"{start:%Y-%m-%d}   {sess.session_id:<6} {start:%H:%M:%S}  "
                f"{duration_hours:>6.1f}h    "
                f"{profile_name:<15} "
                f"{device_name:<20} "
                f"{ahi:>5}"
            )

        # Show truncation info if results were limited
        displayed_count = len(sessions)
        if limit > 0 and displayed_count < total_sessions:
            click.echo(
                f"\nShowing {displayed_count} of {total_sessions} sessions (most recent first)"
            )
            click.echo(
                "💡 Tip: Use --limit <number> to see more sessions, or --limit 0 to see all"
            )
        elif limit == 0 and total_sessions > DEFAULT_LIST_SESSIONS_LIMIT:
            click.echo(f"\nShowing all {total_sessions} sessions")


@cli.command("delete-sessions")
@click.option(
    "--session-id",
    "session_ids",
    type=str,
    help="Comma-separated session IDs to delete (e.g., '1,2,3')",
)
@click.option(
    "--from-date",
    "from_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Delete sessions from this date (YYYY-MM-DD)",
)
@click.option(
    "--to-date",
    "to_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Delete sessions up to this date (YYYY-MM-DD)",
)
@click.option("--all", "delete_all", is_flag=True, help="Delete all sessions")
@click.option(
    "--dry-run", is_flag=True, help="Preview what would be deleted without deleting"
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option("--db", type=click.Path(), help="Database path")
def delete_sessions(
    session_ids: str | None,
    from_date: datetime | None,
    to_date: datetime | None,
    delete_all: bool,
    dry_run: bool,
    force: bool,
    db: str | None,
) -> int | None:
    """Delete sessions from the database."""
    if db:
        init_database(str(Path(db)))
    else:
        init_database()

    # Validate that at least one filter is provided
    if not any([session_ids, from_date, to_date, delete_all]):
        click.echo("❌ Error: You must specify at least one filter:", err=True)
        click.echo("  • --session-id <ids>")
        click.echo("  • --from-date <date>")
        click.echo("  • --to-date <date>")
        click.echo("  • --all")
        return 1

    with session_scope() as session:
        # Build query to select sessions
        query = """
            SELECT
                sessions.id,
                sessions.device_session_id,
                sessions.start_time,
                sessions.duration_seconds,
                devices.manufacturer,
                devices.model,
                devices.serial_number
            FROM sessions
            JOIN devices ON sessions.device_id = devices.id
            WHERE 1=1
        """
        params: dict[str, Any] = {}

        # Apply filters
        if session_ids:
            # Parse comma-separated IDs
            try:
                id_list = [int(sid.strip()) for sid in session_ids.split(",")]
                query += " AND sessions.id IN :session_ids"
                params["session_ids"] = id_list
            except ValueError:
                click.echo(
                    "❌ Error: Invalid session ID format. Use comma-separated integers (e.g., '1,2,3')",
                    err=True,
                )
                return 1

        if from_date:
            query += " AND sessions.start_time >= :from_date"
            params["from_date"] = from_date

        if to_date:
            query += " AND sessions.start_time <= :to_date"
            params["to_date"] = to_date

        query += " ORDER BY sessions.start_time DESC"

        # Execute query to get sessions
        if "session_ids" in params:
            result = session.execute(
                text(query).bindparams(bindparam("session_ids", expanding=True)), params
            )
        else:
            result = session.execute(text(query), params)
        sessions = result.fetchall()

        if not sessions:
            click.echo("⚠️  No sessions found matching the specified criteria")
            return 0

        # Count related data
        session_ids_to_delete = [s.id for s in sessions]

        event_count = session.execute(
            text(
                "SELECT COUNT(*) as count FROM events WHERE session_id IN :session_ids"
            ).bindparams(bindparam("session_ids", expanding=True)),
            {"session_ids": session_ids_to_delete},
        ).scalar()

        waveform_count = session.execute(
            text(
                "SELECT COUNT(*) as count FROM waveforms WHERE session_id IN :session_ids"
            ).bindparams(bindparam("session_ids", expanding=True)),
            {"session_ids": session_ids_to_delete},
        ).scalar()

        stats_count = session.execute(
            text(
                "SELECT COUNT(*) as count FROM statistics WHERE session_id IN :session_ids"
            ).bindparams(bindparam("session_ids", expanding=True)),
            {"session_ids": session_ids_to_delete},
        ).scalar()

        # Display sessions to be deleted
        click.echo(f"\n{'=' * 70}")
        if dry_run:
            click.echo("🔍 DRY RUN MODE - No data will be deleted")
        else:
            click.echo("⚠️  Sessions to be DELETED")
        click.echo(f"{'=' * 70}\n")

        click.echo(
            f"{'ID':<5} {'Date':<12} {'Time':<8} {'Duration':<10} {'Device':<25}"
        )
        click.echo("-" * 70)

        for sess in sessions:
            start = (
                datetime.fromisoformat(sess.start_time)
                if isinstance(sess.start_time, str)
                else sess.start_time
            )
            duration_hours = (
                sess.duration_seconds / 3600 if sess.duration_seconds else 0
            )
            device_name = f"{sess.manufacturer} {sess.model}"

            click.echo(
                f"{sess.id:<5} "
                f"{start:%Y-%m-%d}   {start:%H:%M:%S}  "
                f"{duration_hours:>6.1f}h    "
                f"{device_name:<25}"
            )

        # Display summary
        click.echo("\n" + "=" * 70)
        click.echo("📊 Deletion Summary")
        click.echo("=" * 70)
        click.echo(f"Sessions:    {len(sessions)}")
        click.echo(f"Events:      {event_count}")
        click.echo(f"Waveforms:   {waveform_count}")
        click.echo(f"Statistics:  {stats_count}")
        click.echo("=" * 70 + "\n")

        # Dry-run mode: exit without deleting
        if dry_run:
            click.echo("✓ Dry run complete. Use without --dry-run to delete.")
            return 0

        # Confirmation prompt (unless --force)
        if not force:
            click.echo("⚠️  WARNING: This action cannot be undone!")
            if not click.confirm("Are you sure you want to delete these sessions?"):
                click.echo("Deletion cancelled")
                return 0

        # Perform deletion (CASCADE should handle related data)
        session.execute(
            text("DELETE FROM sessions WHERE id IN :session_ids").bindparams(
                bindparam("session_ids", expanding=True)
            ),
            {"session_ids": session_ids_to_delete},
        )
        session.commit()

        click.echo(
            f"\n✓ Successfully deleted {len(sessions)} session(s) and related data"
        )

        # Suggest vacuum for large deletions
        if len(sessions) > 10:
            click.echo("\n💡 Tip: Run 'snore db vacuum' to reclaim disk space")

        return 0


@cli.command("delete-analysis")
@click.option(
    "--session-id",
    "session_ids",
    type=str,
    help="Comma-separated session IDs to delete analysis for (e.g., '1,2,3')",
)
@click.option(
    "--from-date",
    "from_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Delete analysis for sessions from this date (YYYY-MM-DD)",
)
@click.option(
    "--to-date",
    "to_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Delete analysis for sessions up to this date (YYYY-MM-DD)",
)
@click.option("--all", "delete_all", is_flag=True, help="Delete all analysis results")
@click.option(
    "--all-versions",
    is_flag=True,
    help="Delete all analysis versions (default: only latest)",
)
@click.option(
    "--dry-run", is_flag=True, help="Preview what would be deleted without deleting"
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
@click.option("--db", type=click.Path(), help="Database path")
def delete_analysis(
    session_ids: str | None,
    from_date: datetime | None,
    to_date: datetime | None,
    delete_all: bool,
    all_versions: bool,
    dry_run: bool,
    force: bool,
    db: str | None,
) -> int | None:
    """Delete analysis results without deleting the sessions themselves."""
    if db:
        init_database(str(Path(db)))
    else:
        init_database()

    # Validate that at least one filter is provided
    if not any([session_ids, from_date, to_date, delete_all]):
        raise click.ClickException(
            "You must specify at least one filter:\n"
            "  • --session-id <ids>\n"
            "  • --from-date <date>\n"
            "  • --to-date <date>\n"
            "  • --all"
        )

    with session_scope() as session:
        # Build query to select sessions with analysis
        query = """
            SELECT DISTINCT
                sessions.id,
                sessions.device_session_id,
                sessions.start_time,
                devices.manufacturer,
                devices.model
            FROM sessions
            JOIN devices ON sessions.device_id = devices.id
            JOIN analysis_results ON sessions.id = analysis_results.session_id
            WHERE 1=1
        """
        params: dict[str, Any] = {}

        # Apply filters
        if session_ids:
            # Parse comma-separated IDs
            try:
                id_list = [int(sid.strip()) for sid in session_ids.split(",")]
                query += " AND sessions.id IN :session_ids"
                params["session_ids"] = id_list
            except ValueError:
                click.echo(
                    "❌ Error: Invalid session ID format. Use comma-separated integers (e.g., '1,2,3')",
                    err=True,
                )
                return 1

        if from_date:
            query += " AND sessions.start_time >= :from_date"
            params["from_date"] = from_date

        if to_date:
            query += " AND sessions.start_time <= :to_date"
            params["to_date"] = to_date

        query += " ORDER BY sessions.start_time DESC"

        # Execute query to get sessions with analysis
        if "session_ids" in params:
            result = session.execute(
                text(query).bindparams(bindparam("session_ids", expanding=True)), params
            )
        else:
            result = session.execute(text(query), params)
        sessions_with_analysis = result.fetchall()

        if not sessions_with_analysis:
            click.echo(
                "⚠️  No sessions with analysis results found matching the specified criteria"
            )
            return 0

        # Get session IDs
        session_ids_list = [s.id for s in sessions_with_analysis]

        # Count analysis records per session
        analysis_counts = session.execute(
            text(
                """
                SELECT session_id, COUNT(*) as count
                FROM analysis_results
                WHERE session_id IN :session_ids
                GROUP BY session_id
            """
            ).bindparams(bindparam("session_ids", expanding=True)),
            {"session_ids": session_ids_list},
        ).fetchall()

        analysis_count_dict = {row[0]: int(row[1]) for row in analysis_counts}

        total_analysis_records = sum(analysis_count_dict.values())
        records_to_delete = (
            total_analysis_records if all_versions else len(sessions_with_analysis)
        )

        # Get detected patterns count (for display)
        patterns_count = session.execute(
            text(
                """
                SELECT COUNT(*) as count
                FROM detected_patterns
                WHERE analysis_result_id IN (
                    SELECT id FROM analysis_results WHERE session_id IN :session_ids
                )
            """
            ).bindparams(bindparam("session_ids", expanding=True)),
            {"session_ids": session_ids_list},
        ).scalar()

        # Display analysis to be deleted
        click.echo(f"\n{'=' * 80}")
        if dry_run:
            click.echo("🔍 DRY RUN MODE - No data will be deleted")
        else:
            click.echo("⚠️  Analysis Results to be DELETED")
        click.echo(f"{'=' * 80}\n")

        click.echo(
            f"{'Sess ID':<8} {'Date':<12} {'Time':<8} {'Versions':<10} {'Device':<25}"
        )
        click.echo("-" * 80)

        for sess in sessions_with_analysis:
            start = (
                datetime.fromisoformat(sess.start_time)
                if isinstance(sess.start_time, str)
                else sess.start_time
            )
            device_name = f"{sess.manufacturer} {sess.model}"
            version_count = analysis_count_dict.get(sess.id, 0)

            click.echo(
                f"{sess.id:<8} "
                f"{start:%Y-%m-%d}   {start:%H:%M:%S}  "
                f"{version_count:<10} "
                f"{device_name:<25}"
            )

        # Display summary
        click.echo("\n" + "=" * 80)
        click.echo("📊 Deletion Summary")
        click.echo("=" * 80)
        click.echo(f"Sessions with analysis:          {len(sessions_with_analysis)}")
        click.echo(
            f"Total analysis records:          {total_analysis_records}"
            + (
                " (all versions)"
                if all_versions or total_analysis_records == len(sessions_with_analysis)
                else ""
            )
        )
        click.echo(
            f"Analysis records to delete:      {records_to_delete}"
            + (
                " (latest only)"
                if not all_versions
                and total_analysis_records > len(sessions_with_analysis)
                else ""
            )
        )
        click.echo(
            f"Detected patterns to delete:     {patterns_count} (cascade delete)"
        )
        click.echo("=" * 80 + "\n")

        # Dry-run mode: exit without deleting
        if dry_run:
            click.echo("✓ Dry run complete. Use without --dry-run to delete.")
            return 0

        # Confirmation prompt (unless --force)
        if not force:
            click.echo(
                "⚠️  WARNING: This will delete analysis results but keep the sessions!"
            )
            if not click.confirm(
                "Are you sure you want to delete these analysis results?"
            ):
                click.echo("Deletion cancelled")
                return 0

        # Perform deletion
        if all_versions:
            # Delete all analysis records for matching sessions
            session.execute(
                text(
                    "DELETE FROM analysis_results WHERE session_id IN :session_ids"
                ).bindparams(bindparam("session_ids", expanding=True)),
                {"session_ids": session_ids_list},
            )
            deleted_count = records_to_delete
        else:
            # Delete only the latest analysis record per session
            deleted_count = 0
            for session_id in session_ids_list:
                # Get the latest analysis ID for this session
                latest_result = session.execute(
                    text(
                        """
                        SELECT id FROM analysis_results
                        WHERE session_id = :session_id
                        ORDER BY created_at DESC
                        LIMIT 1
                    """
                    ),
                    {"session_id": session_id},
                ).fetchone()

                if latest_result:
                    session.execute(
                        text("DELETE FROM analysis_results WHERE id = :analysis_id"),
                        {"analysis_id": latest_result.id},
                    )
                    deleted_count += 1

        session.commit()

        click.echo(
            f"\n✓ Successfully deleted {deleted_count} analysis record(s) for {len(sessions_with_analysis)} session(s)"
        )

        # Suggest vacuum for large deletions
        if deleted_count > 10:
            click.echo("\n💡 Tip: Run 'snore db vacuum' to reclaim disk space")

        return 0


@cli.group()
def db() -> None:
    """Database management commands."""
    pass


@db.command()
@click.option("--db", type=click.Path(), help="Database path")
def init(db: str | None) -> int | None:
    """Initialize database (creates tables if needed)."""
    from snore.constants import DEFAULT_DATABASE_PATH

    if db:
        db_path = str(Path(db))
    else:
        db_path = DEFAULT_DATABASE_PATH

    init_database(db_path)
    click.echo(f"✓ Database initialized at {db_path}")
    return None


@db.command()
@click.option("--db", type=click.Path(), help="Database path")
def stats(db: str | None) -> None:
    """Show database statistics."""
    import os

    from snore.constants import DEFAULT_DATABASE_PATH

    if db:
        init_database(str(Path(db)))
        db_path = Path(db)
    else:
        init_database()
        db_path = Path(DEFAULT_DATABASE_PATH)

    with session_scope() as session:
        device_count = session.query(models.Device).count()
        session_count = session.query(models.Session).count()
        event_count = session.execute(text("SELECT COUNT(*) FROM events")).scalar()

        first_session = session.execute(
            text("SELECT MIN(start_time) as first FROM sessions")
        ).scalar()

        last_session = session.execute(
            text("SELECT MAX(start_time) as last FROM sessions")
        ).scalar()

        size_bytes = os.path.getsize(db_path) if db_path.exists() else 0
        size_mb = size_bytes / (1024 * 1024)

        click.echo("\n📊 Database Statistics")
        click.echo(f"{'=' * 50}")
        click.echo(f"Database: {db_path}")
        click.echo(f"Size: {size_mb:.1f} MB")
        click.echo(f"\nDevices: {device_count}")
        click.echo(f"Sessions: {session_count}")
        click.echo(f"Events: {event_count}")

        if first_session and last_session:
            first_dt = (
                datetime.fromisoformat(first_session)
                if isinstance(first_session, str)
                else first_session
            )
            last_dt = (
                datetime.fromisoformat(last_session)
                if isinstance(last_session, str)
                else last_session
            )
            click.echo(f"\nDate range: {first_dt:%Y-%m-%d} to {last_dt:%Y-%m-%d}")

        click.echo(f"{'=' * 50}\n")


@db.command()
@click.option("--db", type=click.Path(), help="Database path")
@click.confirmation_option(prompt="Are you sure you want to vacuum the database?")
def vacuum(db: str | None) -> None:
    """Optimize database (reclaim space after deletions)."""
    if db:
        init_database(str(Path(db)))
    else:
        init_database()

    click.echo("Vacuuming database...")

    with session_scope() as session:
        session.execute(text("VACUUM"))
        session.commit()

    click.echo("✓ Database vacuumed successfully")


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@config.command("set-default-profile")
@click.argument("username")
@click.option("--db", type=click.Path(), help="Database path")
def set_default_profile_cmd(username: str, db: str | None) -> None:
    """Set default profile for CLI commands (must exist in database)."""

    if db:
        init_database(str(Path(db)))
    else:
        init_database()

    with session_scope() as session:
        profile = session.query(models.Profile).filter_by(username=username).first()
        if not profile:
            all_profiles = session.query(models.Profile).all()
            if all_profiles:
                available = ", ".join([p.username for p in all_profiles])
                click.echo(f"Error: Profile '{username}' not found.", err=True)
                click.echo(f"Available profiles: {available}", err=True)
            else:
                click.echo("Error: No profiles in database.", err=True)
                click.echo("Import data first: snore import-data <path>", err=True)
            sys.exit(1)

    set_default_profile(username)
    click.echo(f"✓ Default profile: {username}")
    click.echo(f"  Config: {get_config_path()}")


@config.command("get-default-profile")
def get_default_profile_cmd() -> None:
    """Show current default profile."""
    default = get_default_profile()
    if default:
        click.echo(f"Default profile: {default}")
        click.echo(f"  (from {get_config_path()})")
    else:
        click.echo("No default profile configured.")
        click.echo("Set with: snore config set-default-profile <name>")


@config.command("unset-default-profile")
def unset_default_profile_cmd() -> None:
    """Remove default profile setting."""
    default = get_default_profile()
    if default:
        unset_default_profile()
        click.echo(f"✓ Removed default profile: {default}")
    else:
        click.echo("No default profile was configured.")


@config.command("show")
def show_config_cmd() -> None:
    """Show all configuration settings."""
    from snore.config import load_config

    config_path = get_config_path()
    if not config_path.exists():
        click.echo(f"No config file: {config_path}")
        return

    click.echo(f"Config file: {config_path}\n")
    config_data = load_config()
    if not config_data:
        click.echo("Configuration is empty.")
        return

    click.echo("Settings:")
    if "profile" in config_data:
        click.echo("  [profile]")
        for key, value in config_data["profile"].items():
            click.echo(f'    {key} = "{value}"')


@cli.command()
@click.option(
    "--profile", required=False, help="Profile username (optional if default set)"
)
@click.option("--session-id", type=int, help="Analyze single session by ID")
@click.option(
    "--date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Analyze single session by date (YYYY-MM-DD)",
)
@click.option(
    "--start",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Start date for batch analysis (YYYY-MM-DD)",
)
@click.option(
    "--end",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="End date for batch analysis (YYYY-MM-DD)",
)
@click.option(
    "--limit",
    type=int,
    default=DEFAULT_LIST_SESSIONS_LIMIT,
    help="Max sessions to show in list mode (use 0 for all)",
)
@click.option(
    "--list",
    "list_mode",
    is_flag=True,
    help="Show analysis status instead of running analysis",
)
@click.option("--db", type=click.Path(), help="Database path")
@click.option("--no-store", is_flag=True, help="Don't store results in database")
@click.option(
    "--analyzed-only", is_flag=True, help="Show only analyzed sessions (with --list)"
)
def analyze(
    profile: str | None,
    session_id: int | None,
    date: datetime | None,
    start: datetime | None,
    end: datetime | None,
    limit: int,
    list_mode: bool,
    db: str | None,
    no_store: bool,
    analyzed_only: bool,
) -> int | None:
    """Run programmatic analysis on CPAP sessions."""
    # Initialize database
    if db:
        init_database(str(Path(db)))
    else:
        init_database()

    # Resolve profile using precedence: CLI > config > auto-detect
    with session_scope() as temp_session:
        resolved_profile = resolve_profile(profile, temp_session)

    # Validate mutually exclusive options
    single_session_flags = [session_id is not None, date is not None]
    batch_flags = [start is not None, end is not None]

    single_count = sum(single_session_flags)
    batch_count = sum(batch_flags)

    if single_count > 1:
        click.echo("Error: --session-id and --date are mutually exclusive", err=True)
        sys.exit(1)

    if single_count > 0 and batch_count > 0:
        click.echo(
            "Error: Single session flags (--session-id, --date) cannot be used with batch flags (--start, --end)",
            err=True,
        )
        sys.exit(1)

    if single_count == 0 and batch_count == 0:
        # In list mode, no flags means "show recent sessions (limited by --limit)"
        # In analyze mode, require selection flags for safety
        if not list_mode:
            click.echo(
                "Error: Must provide at least one selection flag (--session-id, --date, --start, or --end)",
                err=True,
            )
            sys.exit(1)

    with session_scope() as session:
        # Lookup profile (use resolved_profile)
        prof = (
            session.query(models.Profile).filter_by(username=resolved_profile).first()
        )
        if not prof:
            click.echo(f"Error: Profile '{resolved_profile}' not found", err=True)
            sys.exit(1)

        # Route to appropriate mode
        if list_mode:
            _list_sessions(session, prof, start, end, limit, analyzed_only)
        elif single_count > 0:
            _analyze_single_session(session, prof, session_id, date, no_store)
        else:
            _analyze_batch(
                session, prof, start, end, start is None and end is None, no_store
            )

    return None


def _analyze_single_session(
    session: Any,
    prof: Any,
    session_id: int | None,
    date: datetime | None,
    no_store: bool,
) -> None:
    """Analyze a single session and display detailed report."""
    if date:
        db_session = (
            session.query(models.Session)
            .join(models.Day)
            .filter(models.Day.profile_id == prof.id, models.Day.date == date.date())
            .first()
        )
        if not db_session:
            click.echo(f"Error: No session found for {date.date()}", err=True)
            sys.exit(1)
        session_id = db_session.id
        session_date_str = date.date().isoformat()
    else:
        db_session = session.query(models.Session).filter_by(id=session_id).first()
        if not db_session:
            click.echo(f"Error: Session {session_id} not found", err=True)
            sys.exit(1)
        # Use Day.date for consistency
        day_date = (
            db_session.day.date if db_session.day else db_session.start_time.date()
        )
        session_date_str = day_date.isoformat()

    click.echo(f"\nAnalyzing session {session_date_str} (ID: {session_id})...")

    analysis_service = AnalysisService(session)

    assert session_id is not None, "session_id should not be None"

    try:
        result = analysis_service.analyze_session(
            session_id=session_id, store_results=not no_store
        )

        click.echo(f"✓ Analysis complete in {result.processing_time_ms}ms\n")

        click.echo("=" * 60)
        click.echo("ANALYSIS SUMMARY")
        click.echo("=" * 60)

        flow_analysis = result.flow_analysis
        event_timeline = result.event_timeline

        click.echo(f"\nSession Duration: {result.duration_hours:.1f} hours")
        click.echo(f"Total Breaths: {result.total_breaths:,}")

        machine_events = getattr(result, "machine_events", [])

        if machine_events:
            machine_event_counts: dict[str, int] = {}
            for event in machine_events:
                machine_event_counts[event.event_type] = (
                    machine_event_counts.get(event.event_type, 0) + 1
                )

            total_machine = len(machine_events)
            oa_count = machine_event_counts.get(EVENT_TYPE_OBSTRUCTIVE_APNEA, 0)
            ca_count = machine_event_counts.get(EVENT_TYPE_CENTRAL_APNEA, 0)
            caa_count = machine_event_counts.get(EVENT_TYPE_CLEAR_AIRWAY, 0)
            ma_count = machine_event_counts.get(EVENT_TYPE_MIXED_APNEA, 0)
            h_count = machine_event_counts.get(EVENT_TYPE_HYPOPNEA, 0)
            re_count = machine_event_counts.get(EVENT_TYPE_RERA, 0)

            machine_ahi_count = oa_count + ca_count + caa_count + ma_count + h_count
            machine_rdi_count = machine_ahi_count + re_count
            machine_ahi = machine_ahi_count / result.duration_hours
            machine_rdi = machine_rdi_count / result.duration_hours

            click.echo("\nMACHINE-DETECTED EVENTS (from CPAP device)")
            click.echo(f"  AHI: {machine_ahi:.1f} events/hour")
            click.echo(f"  RDI: {machine_rdi:.1f} events/hour")
            click.echo(f"  Total Events: {total_machine}")
            if oa_count > 0:
                click.echo(f"    - Obstructive Apneas (OA): {oa_count}")
            if caa_count > 0 or ca_count > 0:
                clear_airway_total = caa_count + ca_count
                click.echo(
                    f"    - Clear Airway / Central Apnea (CA): {clear_airway_total}"
                )
            if ma_count > 0:
                click.echo(f"    - Mixed Apneas (MA): {ma_count}")
            if h_count > 0:
                click.echo(f"    - Hypopneas (H): {h_count}")
            if re_count > 0:
                click.echo(f"    - RERAs (RE): {re_count}")

        click.echo("\nPROGRAMMATIC ANALYSIS (from flow waveform)")
        click.echo(f"  AHI: {event_timeline['ahi']:.1f} events/hour")
        click.echo(f"  RDI: {event_timeline['rdi']:.1f} events/hour")
        click.echo(f"  Flow Limitation Index: {flow_analysis['fl_index']:.2f}")
        click.echo(f"  Total Events: {event_timeline['total_events']}")
        click.echo(f"    - Apneas: {len(event_timeline['apneas'])}")
        for apnea_type in ["OA", "CA", "MA", "UA"]:
            count = sum(
                1 for a in event_timeline["apneas"] if a["event_type"] == apnea_type
            )
            if count > 0:
                click.echo(f"      • {apnea_type}: {count}")
        click.echo(f"    - Hypopneas: {len(event_timeline['hypopneas'])}")
        click.echo(f"    - RERAs: {len(event_timeline['reras'])}")

        if machine_events and event_timeline["total_events"] != total_machine:
            discrepancy = abs(total_machine - event_timeline["total_events"])
            click.echo(
                f"\n⚠️  Discrepancy: Machine detected {total_machine} events, "
                f"programmatic found {event_timeline['total_events']} "
                f"({discrepancy} difference)"
            )

        click.echo("\nFLOW LIMITATION CLASSES")
        for fl_class, count in sorted(flow_analysis["class_distribution"].items()):
            if count > 0:
                pct = (count / result.total_breaths) * 100
                click.echo(f"  Class {fl_class}: {count:,} breaths ({pct:.1f}%)")

        if result.csr_detection:
            csr = result.csr_detection
            click.echo("\nCHEYNE-STOKES RESPIRATION")
            click.echo(f"  Detected: Yes (confidence: {csr['confidence']:.2f})")
            click.echo(f"  Cycle Length: {csr['cycle_length']:.0f}s")
            click.echo(f"  CSR Index: {csr['csr_index']:.1%}")

        if result.periodic_breathing:
            periodic = result.periodic_breathing
            click.echo("\nPERIODIC BREATHING")
            click.echo(f"  Detected: Yes (confidence: {periodic['confidence']:.2f})")
            click.echo(f"  Cycle Length: {periodic['cycle_length']:.0f}s")
            click.echo(f"  Regularity: {periodic['regularity_score']:.2f}")

        if result.positional_analysis:
            positional = result.positional_analysis
            click.echo("\nPOSITIONAL ANALYSIS")
            click.echo(f"  Event Clustering: {positional['total_clusters']} clusters")
            click.echo(
                f"  Positional Likelihood: {positional['positional_likelihood']:.2f}"
            )

        if not no_store:
            stored = analysis_service.get_analysis_result(session_id)
            if stored:
                click.echo(
                    f"\nResults stored with analysis ID: {stored['analysis_id']}"
                )

        click.echo("\n" + "=" * 60)

    except Exception as e:
        click.echo(f"\nAnalysis failed: {e}", err=True)
        logger.error("Analysis error", exc_info=True)
        sys.exit(1)


def _analyze_batch(
    session: Any,
    prof: Any,
    start: datetime | None,
    end: datetime | None,
    analyze_all: bool,
    no_store: bool,
) -> None:
    """Analyze multiple sessions with progress bar."""
    query = (
        session.query(models.Session)
        .join(models.Day)
        .filter(models.Day.profile_id == prof.id)
    )

    if not analyze_all:
        if start:
            query = query.filter(models.Day.date >= start.date())
        if end:
            query = query.filter(models.Day.date <= end.date())

    sessions = query.order_by(models.Day.date).all()

    if not sessions:
        click.echo("No sessions found for the specified criteria")
        return

    click.echo(f"\nAnalyzing {len(sessions)} sessions...")

    analysis_service = AnalysisService(session)
    successful = 0
    failed = 0

    with click.progressbar(sessions, label="Analyzing") as bar:
        for db_session in bar:
            try:
                analysis_service.analyze_session(
                    session_id=db_session.id, store_results=not no_store
                )
                successful += 1
            except Exception as e:
                failed += 1
                logger.debug(f"Failed to analyze session {db_session.id}: {e}")

    click.echo("\n✓ Analysis complete")
    click.echo(f"  Successful: {successful}")
    click.echo(f"  Failed: {failed}")


def _list_sessions(
    session: Any,
    prof: Any,
    start: datetime | None,
    end: datetime | None,
    limit: int,
    analyzed_only: bool,
) -> None:
    """List sessions and their analysis status.

    Args:
        limit: Maximum sessions to show (0 for unlimited)
    """
    # Build query
    query = (
        session.query(models.Session)
        .join(models.Day)
        .filter(models.Day.profile_id == prof.id)
    )

    # Apply date filters if specified
    if start:
        query = query.filter(models.Day.date >= start.date())
    if end:
        query = query.filter(models.Day.date <= end.date())

    # Order by most recent first
    query = query.order_by(models.Day.date.desc())

    # Count total before limiting
    total_sessions = query.count()

    # Apply limit if specified (0 means no limit)
    if limit > 0:
        query = query.limit(limit)

    sessions = query.all()

    if not sessions:
        click.echo("No sessions found")
        return

    click.echo(
        f"{'Date':<12} {'ID':<6} {'Duration':<10} {'Analyzed':<10} {'Analysis ID':<12}"
    )
    click.echo("-" * 60)

    # Track displayed count (may be less than len(sessions) with analyzed_only)
    displayed_count = 0

    for db_session in sessions:
        analysis = (
            session.query(models.AnalysisResult)
            .filter_by(session_id=db_session.id)
            .order_by(models.AnalysisResult.created_at.desc())
            .first()
        )

        has_analysis = analysis is not None

        if analyzed_only and not has_analysis:
            continue

        displayed_count += 1

        duration = (
            f"{db_session.duration_seconds / 3600:.1f}h"
            if db_session.duration_seconds
            else "N/A"
        )
        analyzed_str = "✓" if has_analysis else "✗"
        analysis_id_str = str(analysis.id) if analysis else "-"

        # Use Day.date for consistency with --date queries
        # Fallback to start_time.date() if session somehow lacks day assignment
        day_date = (
            db_session.day.date if db_session.day else db_session.start_time.date()
        )
        click.echo(
            f"{day_date!s:<12} {db_session.id:<6} {duration:<10} "
            f"{analyzed_str:<10} {analysis_id_str:<12}"
        )

    # Show helpful messages
    if analyzed_only and displayed_count > 0:
        click.echo(f"\nShowing {displayed_count} analyzed session(s)")
    elif limit > 0 and total_sessions > limit:
        click.echo(
            f"\nShowing {limit} of {total_sessions} sessions (most recent first)"
        )
        click.echo(
            "Tip: Use --limit <number> to see more sessions, or --limit 0 to see all"
        )
    elif limit == 0 and total_sessions > DEFAULT_LIST_SESSIONS_LIMIT:
        click.echo(f"\nShowing all {total_sessions} sessions")


@cli.group()
def completions() -> None:
    """Manage shell tab completion."""
    pass


from snore.completions import get_supported_shells

_SUPPORTED_SHELLS = list(get_supported_shells())


@completions.command(name="bash")
def completions_bash() -> None:
    """Output bash completion script for manual installation."""
    from snore.completions import generate_completion_script

    try:
        script = generate_completion_script("bash")
        click.echo(script)
        click.echo("\nTo install: Add the above to your ~/.bashrc or run:")
        click.echo("  snore completions install")
    except Exception as e:
        click.echo(f"Error generating completion script: {e}", err=True)
        sys.exit(1)


@completions.command(name="zsh")
def completions_zsh() -> None:
    """Output zsh completion script for manual installation."""
    from snore.completions import generate_completion_script

    try:
        script = generate_completion_script("zsh")
        click.echo(script)
        click.echo("\nTo install: Add the above to your ~/.zshrc or run:")
        click.echo("  snore completions install")
    except Exception as e:
        click.echo(f"Error generating completion script: {e}", err=True)
        sys.exit(1)


@completions.command(name="install")
@click.option(
    "--shell",
    type=click.Choice(_SUPPORTED_SHELLS, case_sensitive=False),
    help="Shell type (auto-detected if not specified)",
)
def completions_install(shell: str | None) -> None:
    """Install shell completion to config file."""
    from snore.completions import detect_shell, install_completion

    if shell is None:
        shell = detect_shell()
        if shell is None:
            click.echo(
                "Error: Could not detect shell. Please specify with --shell", err=True
            )
            sys.exit(1)
        click.echo(f"Detected shell: {shell}")

    success, message = install_completion(shell, dry_run=False)

    if success:
        click.echo(f"✓ {message}")
    else:
        click.echo(f"Error: {message}", err=True)
        sys.exit(1)


@completions.command(name="uninstall")
@click.option(
    "--shell",
    type=click.Choice(_SUPPORTED_SHELLS, case_sensitive=False),
    help="Shell type (auto-detected if not specified)",
)
def completions_uninstall(shell: str | None) -> None:
    """Remove shell completion from config file."""
    from snore.completions import (
        detect_shell,
        find_config_file,
        uninstall_completion,
    )

    if shell is None:
        shell = detect_shell()
        if shell is None:
            click.echo(
                "Error: Could not detect shell. Please specify with --shell", err=True
            )
            sys.exit(1)

    config_path = find_config_file(shell)
    if config_path is None:
        click.echo(f"Error: No {shell} config file found", err=True)
        sys.exit(1)

    success, message = uninstall_completion(config_path)

    if success:
        click.echo(f"✓ {message}")
    else:
        click.echo(f"Error: {message}", err=True)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
