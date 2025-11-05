"""
CLI utility for importing OSCAR data into the MCP database.

This tool scans OSCAR profile directories and imports the data
into a SQLite database for use by the OSCAR-MCP server.
"""

import logging
import os
import sys

import click

from oscar_mcp.constants import DEFAULT_DATABASE_PATH, COMPLIANCE_MIN_HOURS
from oscar_mcp.database.session import init_database, session_scope
from oscar_mcp.database import models
from oscar_mcp.parsers.oscar_format import ProfileScanner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("oscar-import")


@click.group()
@click.option("--database", default=DEFAULT_DATABASE_PATH, help="Path to database file")
@click.pass_context
def cli(ctx, database):
    """OSCAR data import utility for OSCAR-MCP."""
    ctx.ensure_object(dict)
    ctx.obj["database"] = database

    # Initialize database
    logger.info(f"Using database: {database}")
    init_database(database)


@cli.command()
@click.argument("oscar_profile_path")
@click.pass_context
def scan(ctx, oscar_profile_path):
    """
    Scan an OSCAR profile directory and display available data.

    OSCAR_PROFILE_PATH: Path to OSCAR profile directory
    (e.g., ~/.local/share/OSCAR/Profiles/ProfileName/)
    """
    if not os.path.exists(oscar_profile_path):
        click.echo(f"Error: Profile path does not exist: {oscar_profile_path}", err=True)
        sys.exit(1)

    click.echo(f"Scanning OSCAR profile: {oscar_profile_path}")
    click.echo("")

    try:
        scanner = ProfileScanner(oscar_profile_path)
        machines = scanner.scan_machines()

        if not machines:
            click.echo("No machines found in this profile.")
            return

        click.echo(f"Found {len(machines)} machine(s):")
        click.echo("")

        total_sessions = 0
        for machine_id in machines:
            machine_scanner = scanner.get_machine_scanner(machine_id)
            sessions = machine_scanner.scan_sessions()
            total_sessions += len(sessions)

            click.echo(f"  Machine ID: {machine_id}")
            click.echo(f"    Sessions: {len(sessions)}")
            if sessions:
                click.echo(f"    First: {sessions[0]}")
                click.echo(f"    Last:  {sessions[-1]}")
            click.echo("")

        click.echo(f"Total: {total_sessions} session(s) across {len(machines)} machine(s)")
        click.echo("")
        click.echo("To import this data, run:")
        click.echo(f"  oscar-import import '{oscar_profile_path}' --profile-name 'Your Name'")

    except Exception as e:
        logger.error(f"Error scanning profile: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("oscar_profile_path")
@click.option("--profile-name", required=True, help="Name for this profile in the database")
@click.option("--first-name", help="User first name")
@click.option("--last-name", help="User last name")
@click.pass_context
def import_data(ctx, oscar_profile_path, profile_name, first_name, last_name):
    """
    Import OSCAR data from a profile directory into the database.

    OSCAR_PROFILE_PATH: Path to OSCAR profile directory
    """
    if not os.path.exists(oscar_profile_path):
        click.echo(f"Error: Profile path does not exist: {oscar_profile_path}", err=True)
        sys.exit(1)

    click.echo(f"Importing OSCAR profile: {oscar_profile_path}")
    click.echo(f"Profile name: {profile_name}")
    click.echo("")

    try:
        with session_scope() as session:
            # Check if profile already exists
            existing_profile = (
                session.query(models.Profile).filter_by(username=profile_name).first()
            )
            if existing_profile:
                click.echo(f"Profile '{profile_name}' already exists. Updating...")
                profile = existing_profile
            else:
                # Create new profile
                profile = models.Profile(
                    username=profile_name, first_name=first_name, last_name=last_name
                )
                session.add(profile)
                session.flush()  # Get profile ID
                click.echo(f"Created profile: {profile_name}")

            # Scan OSCAR directory
            scanner = ProfileScanner(oscar_profile_path)
            machines = scanner.scan_machines()

            click.echo(f"Found {len(machines)} machine(s)")
            click.echo("")

            # TODO: Implement full import logic
            # For each machine:
            #   1. Create/update Device record (link to profile.id)
            #   2. Scan sessions from OSCAR directory
            #   3. For each session:
            #      a. Parse summary (.000) and event (.001) files
            #      b. Create Session record (with device_id)
            #      c. Create Event records in database
            #      d. Create Statistics record (1-to-1 with Session)
            #      e. Use DayManager.link_session_to_day() to:
            #         - Determine day date using day-splitting logic
            #         - Create/update Day record
            #         - Link session to day (session.day_id = day.id)
            #         - Recalculate day statistics
            #
            # Example code structure:
            #   from oscar_mcp.database.day_manager import DayManager
            #
            #   for machine_data in machines:
            #       device = get_or_create_device(profile_id, machine_data)
            #       for session_data in parse_sessions(machine_data):
            #           session = create_session(device.id, session_data)
            #           DayManager.link_session_to_day(session, profile, session_obj)

            click.echo("NOTE: Full import functionality is not yet implemented.")
            click.echo("This is a skeleton implementation showing the structure.")
            click.echo("")
            click.echo("TODO: Implement:")
            click.echo("  - Parse OSCAR .000/.001 files (summary and event data)")
            click.echo("  - Extract session metadata and statistics")
            click.echo("  - Store Events and Statistics in database")
            click.echo("  - Use DayManager to aggregate sessions into days")
            click.echo("  - Handle incremental imports (only new sessions)")
            click.echo("")
            click.echo("See DayManager in oscar_mcp.database.day_manager for day aggregation.")

    except Exception as e:
        logger.error(f"Error importing data: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def status(ctx):
    """Display status of imported data in the database."""
    try:
        with session_scope() as session:
            profiles = session.query(models.Profile).all()

            if not profiles:
                click.echo("No profiles found in database.")
                click.echo("")
                click.echo("To import OSCAR data, run:")
                click.echo("  oscar-import scan <path-to-oscar-profile>")
                return

            click.echo(f"Database: {ctx.obj['database']}")
            click.echo("")
            click.echo(f"Found {len(profiles)} profile(s):")
            click.echo("")

            for profile in profiles:
                click.echo(f"  Profile: {profile.username}")
                if profile.first_name or profile.last_name:
                    full_name = f"{profile.first_name or ''} {profile.last_name or ''}".strip()
                    click.echo(f"    Name: {full_name}")

                # Count devices
                device_count = len(profile.devices)
                click.echo(f"    Devices: {device_count}")

                # Count days and sessions
                day_count = len(profile.days)
                total_sessions = sum(len(day.sessions) for day in profile.days)
                total_hours = sum(day.total_therapy_hours or 0 for day in profile.days)

                click.echo(f"    Days: {day_count}")
                click.echo(f"    Sessions: {total_sessions}")
                click.echo(f"    Total hours: {total_hours:.1f}")

                if profile.days:
                    dates = [day.date for day in profile.days]
                    click.echo(f"    Date range: {min(dates)} to {max(dates)}")

                # Calculate compliance
                if profile.days:
                    compliant_days = sum(
                        1
                        for day in profile.days
                        if day.total_therapy_hours
                        and day.total_therapy_hours >= COMPLIANCE_MIN_HOURS
                    )
                    compliance_rate = (compliant_days / len(profile.days)) * 100
                    click.echo(
                        f"    Compliance: {compliance_rate:.1f}% ({compliant_days}/{day_count} days >= 4 hours)"
                    )

                click.echo("")

    except Exception as e:
        logger.error(f"Error retrieving status: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the oscar-import CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
