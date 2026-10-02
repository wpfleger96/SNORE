"""Session management commands."""

from __future__ import annotations

from datetime import datetime

import click

from snore.cli.decorators import (
    CliCtx,
    date_range_options,
    parse_id_list,
    profile_scoped_command,
)
from snore.cli.display import (
    ICON_STATS,
    Column,
    console,
    print_dry_run_complete,
    print_dry_run_header,
    print_footer,
    print_header,
    print_success,
    print_table,
    print_tip,
    print_warning,
)
from snore.constants import DEFAULT_LIST_SESSIONS_LIMIT
from snore.provenance import field_provenance
from snore.services.schemas import SessionListItem


@click.group()
def session() -> None:
    """Session management commands."""
    pass


@session.command("list")
@click.option("--device", "-d", help="Filter by device serial number")
@date_range_options
@click.option(
    "--limit",
    type=int,
    default=DEFAULT_LIST_SESSIONS_LIMIT,
    help="Max sessions to show (use 0 for all)",
)
@click.option(
    "--sort-by",
    type=click.Choice(["date-asc", "date-desc", "session-id", "duration"]),
    default="date-desc",
    help="Sort order for results (default: date-desc)",
)
@click.option("--all", "show_all", is_flag=True, help="Include disabled sessions")
@profile_scoped_command
async def session_list(
    ctx: CliCtx,
    device: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    limit: int,
    sort_by: str,
    show_all: bool,
) -> None:
    """List imported sessions."""
    from snore.services.session_service import SessionService  # noqa: PLC0415

    service = SessionService(ctx.db, ctx.profile_id)
    result = await service.list_sessions(
        device=device,
        from_date=date_from,
        to_date=date_to,
        limit=limit,
        sort_by=sort_by,
        include_disabled=show_all,
    )

    if not result.sessions:
        console.print("No sessions found")
        return

    rows = []
    for sess in result.sessions:
        device_name = f"{sess.manufacturer} {sess.model}"
        ahi_str = f"{sess.ahi:.1f}" if sess.ahi is not None else "N/A"
        status_marker = "" if sess.enabled else "[disabled]"
        rows.append(
            (
                str(sess.id),
                f"{sess.therapy_day}",
                f"{sess.start_time:%H:%M:%S}",
                f"{sess.duration_hours:>6.1f}h",
                device_name,
                sess.serial_number,
                f"{ahi_str:<8} {status_marker}",
            )
        )

    print_table(
        [
            Column("ID", 5),
            Column("Therapy Day", 12),
            Column("Time", 8),
            Column("Duration", 10, field_provenance(SessionListItem, "duration_hours")),
            Column("Device", 30),
            Column("Serial", 15),
            Column("AHI", 8, field_provenance(SessionListItem, "ahi")),
        ],
        rows,
    )

    if result.total_count > 0 and limit > 0 and result.total_count > limit:
        console.print(
            f"\nShowing {len(result.sessions)} of {result.total_count} sessions"
        )
        print_tip(f"Use '--limit {result.total_count}' or '--limit 0' to show all")
    else:
        console.print(f"\nShowing all {len(result.sessions)} sessions")


@session.command("show")
@click.argument("session_id", type=int)
@click.option("--settings", "show_settings", is_flag=True, help="Show device settings")
@profile_scoped_command
async def session_show(
    ctx: CliCtx,
    session_id: int,
    show_settings: bool,
) -> None:
    """Show details for a specific session."""
    from snore.cli.display.analysis import display_session_detail  # noqa: PLC0415
    from snore.services.session_service import SessionService  # noqa: PLC0415

    service = SessionService(ctx.db, ctx.profile_id)

    try:
        detail = await service.get_session_detail(
            session_id, include_settings=show_settings
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    display_session_detail(detail, show_settings)


@session.command("delete")
@click.option("--device", "-d", help="Filter by device serial number")
@click.option(
    "--session-id",
    "session_ids",
    type=str,
    help="Comma-separated session IDs to delete (e.g., '1,2,3')",
)
@date_range_options
@click.option("--all", "delete_all", is_flag=True, help="Delete all sessions")
@click.option(
    "--dry-run", is_flag=True, help="Preview what would be deleted without deleting"
)
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@profile_scoped_command
async def session_delete(
    ctx: CliCtx,
    device: str | None,
    session_ids: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    delete_all: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Delete sessions from the database."""
    from snore.services.session_service import SessionService  # noqa: PLC0415

    id_list = None
    if session_ids:
        id_list = parse_id_list(session_ids)

    service = SessionService(ctx.db, ctx.profile_id)

    try:
        preview = await service.get_delete_preview(
            device=device,
            session_ids=id_list,
            from_date=date_from,
            to_date=date_to,
            delete_all=delete_all,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    if not preview.sessions:
        print_warning("No sessions found matching the specified criteria")
        return

    print_footer(wide=True)
    if dry_run:
        print_dry_run_header("deleted")
    else:
        print_warning("Sessions to be DELETED")
    print_footer(wide=True)
    console.print()

    print_table(
        [
            Column("ID", 5),
            Column("Therapy Day", 12),
            Column("Time", 8),
            Column("Duration", 10),
            Column("Device", 30),
            Column("Serial", 15),
        ],
        (
            (
                str(sess.id),
                f"{sess.therapy_day}",
                f"{sess.start_time:%H:%M:%S}",
                f"{sess.duration_hours:>6.1f}h",
                f"{sess.manufacturer} {sess.model}",
                sess.serial_number,
            )
            for sess in preview.sessions
        ),
    )

    print_header("Deletion Summary", ICON_STATS, wide=True)
    console.print(f"Sessions:    {len(preview.sessions)}")
    console.print(f"Events:      {preview.event_count}")
    console.print(f"Waveforms:   {preview.waveform_count}")
    console.print(f"Statistics:  {preview.stats_count}")
    print_footer(wide=True)
    console.print()

    if dry_run:
        print_dry_run_complete("delete")
        return

    if not force:
        print_warning("WARNING: This action cannot be undone!")
        if not click.confirm("Are you sure you want to delete these sessions?"):
            console.print("Deletion cancelled")
            return

    session_ids_to_delete = [s.id for s in preview.sessions]
    deleted_count = await service.delete_sessions(session_ids_to_delete)

    print_success(f"Successfully deleted {deleted_count} session(s) and related data")

    if deleted_count > 10:
        print_tip("Run 'snore db vacuum' to reclaim disk space")


async def _toggle_session_async(ctx: CliCtx, session_id: int, enabled: bool) -> None:
    """Enable or disable a session and recalculate day statistics."""
    from snore.services.session_service import SessionService  # noqa: PLC0415

    service = SessionService(ctx.db, ctx.profile_id)

    try:
        detail = await service.get_session_detail(session_id)
        if detail.enabled == enabled:
            status = "enabled" if enabled else "disabled"
            console.print(f"Session {session_id} is already {status}")
            return

        await service.set_session_enabled(session_id, enabled)

        status = "enabled" if enabled else "disabled"
        console.print(f"Session {session_id} {status} and day statistics recalculated")
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@session.command("enable")
@click.argument("session_id", type=int)
@profile_scoped_command
async def session_enable(ctx: CliCtx, session_id: int) -> None:
    """Enable a session and recalculate day statistics."""
    await _toggle_session_async(ctx, session_id, enabled=True)


@session.command("disable")
@click.argument("session_id", type=int)
@profile_scoped_command
async def session_disable(ctx: CliCtx, session_id: int) -> None:
    """Disable a session and recalculate day statistics."""
    await _toggle_session_async(ctx, session_id, enabled=False)
