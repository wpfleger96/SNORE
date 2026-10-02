"""Waveform inspection and visualization commands."""

from __future__ import annotations

import asyncio
import csv

from datetime import datetime

import click

from rich.markup import escape

from snore.cli.decorators import (
    CliCtx,
    profile_scoped_command,
    session_id_date_options,
)
from snore.cli.display import (
    Column,
    console,
    mark_field,
    print_table,
    print_warning,
)
from snore.services.schemas import EventComparisonResult
from snore.waveform import format_time_offset
from snore.waveform.inspector import parse_time_offset


async def _resolve_session_id(
    ctx: CliCtx,
    session_id: int | None,
    date: datetime | None,
) -> int:
    """
    Resolve session ID from either explicit ID or date.

    Args:
        ctx: CLI runtime context (db session + profile id)
        session_id: Explicit session ID (takes precedence)
        date: Date to look up session

    Returns:
        Resolved session ID

    Raises:
        click.ClickException: If session cannot be resolved
    """
    from snore.services.session_service import SessionService  # noqa: PLC0415

    service = SessionService(ctx.db, ctx.profile_id)

    try:
        return await service.resolve_session_id(session_id, date)
    except ValueError as e:
        raise click.ClickException(str(e)) from e


@click.group()
def waveform() -> None:
    """Waveform inspection and visualization commands."""
    pass


@waveform.command("list")
@session_id_date_options
@profile_scoped_command
async def list_waveforms(
    ctx: CliCtx,
    session_id: int | None,
    date: datetime | None,
) -> None:
    """
    List available waveform types for a session.

    Shows all waveform data available for the specified session, including
    sample rates, sample counts, units, and durations.

    Examples:
        snore waveform list --session-id 37
        snore waveform list --date 2025-10-25
    """
    from snore.services.waveform_service import WaveformService  # noqa: PLC0415

    if session_id is None and date is None:
        raise click.ClickException("Either --session-id or --date must be provided")

    resolved_id = await _resolve_session_id(ctx, session_id, date)

    service = WaveformService(ctx.db, ctx.profile_id)
    waveforms = await service.list_waveforms(resolved_id)

    if not waveforms:
        console.print(f"No waveforms found for session {resolved_id}")
        return

    console.print(f"Available waveforms for session {resolved_id}:")
    print_table(
        [
            Column("TYPE", 12),
            Column("RATE", 12),
            Column("SAMPLES", 10),
            Column("UNIT", 10),
            Column("DURATION", 0),
        ],
        (
            (
                wf.waveform_type,
                f"{wf.sample_rate:.1f}Hz",
                str(wf.sample_count),
                wf.unit or "?",
                f"{wf.duration_hours:.1f}h",
            )
            for wf in waveforms
        ),
        header_separator=False,
        indent=1,
    )


@waveform.command("show")
@session_id_date_options
@click.option("--time", required=True, help="Time offset (HH:MM:SS)")
@click.option(
    "--window", type=int, default=60, help="Window size in seconds (default: 60)"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["plot", "csv"]),
    default="plot",
    help="Output format (plot=interactive graph, csv=data export)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (required for csv format)",
)
@click.option(
    "--mode", "-m", default="aasm", help="Detection mode to compare (default: aasm)"
)
@click.option(
    "--type",
    "waveform_type",
    default="flow",
    help="Waveform type to display (default: flow)",
)
@profile_scoped_command
async def show_waveform(
    ctx: CliCtx,
    session_id: int | None,
    date: datetime | None,
    time: str,
    window: int,
    output_format: str,
    output: str | None,
    mode: str,
    waveform_type: str,
) -> None:
    """
    Display waveform at a specific time.

    View waveform data centered on a specific time offset to visually
    inspect detected respiratory events (for flow waveforms).

    Examples:
        snore waveform show --session-id 37 --time 05:56:22 --window 30
        snore waveform show --date 2025-10-25 --time 01:25:16 --type pressure
        snore waveform show --session-id 37 --time 01:25:16 --format csv --output waveform.csv
    """
    from snore.analysis.service import AnalysisService  # noqa: PLC0415
    from snore.waveform import WaveformInspector, WaveformRenderer  # noqa: PLC0415

    if session_id is None and date is None:
        raise click.ClickException("Either --session-id or --date must be provided")

    if output_format == "csv" and output is None:
        raise click.ClickException("--output is required for csv format")

    waveform_types = [t.strip() for t in waveform_type.split(",")]

    if output_format == "csv" and len(waveform_types) > 1:
        raise click.ClickException("CSV export only supports single waveform type")

    if len(waveform_types) > 4:
        raise click.ClickException("Maximum 4 waveform types supported")

    try:
        center_seconds = parse_time_offset(time)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    resolved_id = await _resolve_session_id(ctx, session_id, date)
    profile_id = ctx.profile_id

    inspector = WaveformInspector(ctx.db, profile_id)

    if len(waveform_types) == 1:
        waveform_type_single = waveform_types[0]
        try:
            timestamps, values, metadata = await inspector.get_window(
                session_id=resolved_id,
                center_seconds=center_seconds,
                window_seconds=float(window),
                waveform_type=waveform_type_single,
            )
        except Exception as e:
            raise click.ClickException(f"Error loading waveform: {e}") from e

        if len(timestamps) == 0:
            raise click.ClickException("No data in window")

        machine_events = []
        programmatic_events = []

        if waveform_type_single == "flow":
            analysis_service = AnalysisService(ctx.db, profile_id=profile_id)
            try:
                result = await analysis_service.get_analysis_result(resolved_id)
            except Exception:
                result = None

            if result:
                start_time = center_seconds - window / 2
                end_time = center_seconds + window / 2

                if result.machine_events:
                    machine_events = inspector.find_events_in_window(
                        result.machine_events, start_time, end_time
                    )

                if mode in result.mode_results:
                    mode_result = result.mode_results[mode]
                    all_prog_events = list(mode_result.apneas) + list(
                        mode_result.hypopneas
                    )
                    programmatic_events = inspector.find_events_in_window(
                        all_prog_events, start_time, end_time
                    )

        if output_format == "plot":
            show_events = waveform_type_single == "flow"
            renderer = WaveformRenderer(
                console=console, width=80, height=20, show_events=show_events
            )
            renderer.render(
                timestamps=timestamps,
                values=values,
                machine_events=machine_events,
                programmatic_events=programmatic_events,
                session_id=resolved_id,
                center_time=time,
                waveform_type=waveform_type_single,
            )

        elif output_format == "csv":
            assert output is not None
            column_name = f"{waveform_type_single}_value"
            with open(output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp_seconds", column_name])
                for ts, value in zip(timestamps, values, strict=True):
                    writer.writerow([f"{ts:.3f}", f"{value:.3f}"])

            console.print(f"Exported {len(timestamps)} samples to {output}")

    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        from snore.database.session import session_scope  # noqa: PLC0415

        async def _load_waveform_async(
            wf_type: str,
        ) -> tuple[np.ndarray, np.ndarray, str] | None:
            async with session_scope() as thread_session:
                thread_inspector = WaveformInspector(thread_session, profile_id)
                ts, vals, _meta = await thread_inspector.get_window(
                    session_id=resolved_id,
                    center_seconds=center_seconds,
                    window_seconds=float(window),
                    waveform_type=wf_type,
                )
            if len(ts) > 0:
                return (ts, vals, wf_type)
            return None

        def load_waveform(
            wf_type: str,
        ) -> tuple[np.ndarray, np.ndarray, str] | None:
            return asyncio.run(_load_waveform_async(wf_type))

        waveform_data = []
        warnings: list[str] = []
        max_workers = min(4, len(waveform_types))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(load_waveform, wf_type): wf_type
                for wf_type in waveform_types
            }

            for future in as_completed(futures):
                wf_type = futures[future]
                try:
                    wf_result = future.result()
                    if wf_result is not None:
                        waveform_data.append(wf_result)
                    else:
                        warnings.append(f"No data for waveform type '{wf_type}'")
                except Exception as e:
                    warnings.append(f"Failed to load waveform type '{wf_type}': {e}")

        for w in warnings:
            print_warning(w)

        if not waveform_data:
            raise click.ClickException("No waveform data loaded")

        type_order = {t: i for i, t in enumerate(waveform_types)}
        waveform_data.sort(key=lambda x: type_order.get(x[2], 999))

        renderer = WaveformRenderer(
            console=console, width=80, height=20, show_events=False
        )
        renderer.render_multi(
            waveform_data=waveform_data,
            session_id=resolved_id,
            center_time=time,
        )


@waveform.command("compare")
@session_id_date_options
@click.option(
    "--mode", "-m", default="aasm", help="Detection mode to compare (default: aasm)"
)
@click.option("--show-unmatched", is_flag=True, help="Only show unmatched events")
@profile_scoped_command
async def compare_events(
    ctx: CliCtx,
    session_id: int | None,
    date: datetime | None,
    mode: str,
    show_unmatched: bool,
) -> None:
    """
    Compare machine vs programmatic events with waveform inspection commands.

    Lists false positives and false negatives with commands to inspect each event.

    Examples:
        snore waveform compare --session-id 37 --mode aasm
        snore waveform compare --date 2025-10-25 --mode resmed --show-unmatched
    """
    from snore.exceptions import NotFoundError  # noqa: PLC0415
    from snore.services.waveform_service import WaveformService  # noqa: PLC0415

    if session_id is None and date is None:
        raise click.ClickException("Either --session-id or --date must be provided")

    resolved_id = await _resolve_session_id(ctx, session_id, date)

    service = WaveformService(ctx.db, ctx.profile_id)
    try:
        comparison = await service.compare_events(
            resolved_id, mode, tolerance_seconds=5.0
        )
    except NotFoundError as e:
        raise click.ClickException(str(e)) from e

    programmatic = mark_field(
        "Programmatic", EventComparisonResult, "programmatic_event_count"
    )
    console.print(f"Session {resolved_id} - Event Comparison ({mode} mode)")
    console.print(
        f"Machine: {comparison.machine_event_count} events | {programmatic}: {comparison.programmatic_event_count} events"
    )
    console.print("")

    if not show_unmatched or len(comparison.false_negatives) > 0:
        console.print(
            f"{mark_field('FALSE NEGATIVES', EventComparisonResult, 'false_negatives')} (machine events missed by programmatic): {len(comparison.false_negatives)}"
        )
        for event in comparison.false_negatives:
            time_str = format_time_offset(event.start_time)
            console.print(
                f"  {escape(str(event.event_type))} at {time_str} ({event.duration:.1f}s)"
            )
            console.print(
                f"    → View: snore waveform show --session-id {resolved_id} --time {time_str}"
            )
        console.print("")

    if (
        not show_unmatched
        or len(comparison.false_positives_apnea)
        + len(comparison.false_positives_hypopnea)
        > 0
    ):
        console.print(
            f"{mark_field('FALSE POSITIVES', EventComparisonResult, 'false_positives_apnea')} (programmatic events not in machine): {len(comparison.false_positives_apnea) + len(comparison.false_positives_hypopnea)}"
        )

        for event in comparison.false_positives_apnea:
            time_str = format_time_offset(event.start_time)
            conf = event.confidence or 0
            flow_red = event.flow_reduction or 0
            console.print(
                f"  {escape(str(event.event_type))} at {time_str} (conf: {conf:.2f}, flow_red: {flow_red * 100:.0f}%)"
            )
            console.print(
                f"    → View: snore waveform show --session-id {resolved_id} --time {time_str}"
            )

        for event in comparison.false_positives_hypopnea:
            time_str = format_time_offset(event.start_time)
            conf = event.confidence or 0
            flow_red = event.flow_reduction or 0
            console.print(
                f"  H at {time_str} (conf: {conf:.2f}, flow_red: {flow_red * 100:.0f}%)"
            )
            console.print(
                f"    → View: snore waveform show --session-id {resolved_id} --time {time_str}"
            )
