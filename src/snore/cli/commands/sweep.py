"""sweep-thresholds command — offline FL/RERA query-time tuning sweep.

Loads stored data once and re-scores a Cartesian grid of query-time tunables
against one of three reference targets, ranking the results.  It measures only:
no algorithm or threshold in the codebase is changed, and nothing is written to
the database.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from rich.table import Table

from snore.cli.decorators import (
    CliCtx,
    cli_error_boundary,
    date_range_options_required,
    profile_scoped_command,
)
from snore.cli.display import console, fmt_sig, mark_provenance
from snore.provenance import Provenance
from snore.validation.sweep import SweepResult


def _parse_floats(raw: str | None) -> list[float] | None:
    """Parse a comma-separated list of numbers into floats (None when absent)."""
    if raw is None:
        return None
    try:
        return [float(v.strip()) for v in raw.split(",") if v.strip()]
    except ValueError as e:
        raise click.BadParameter(
            f"Invalid numeric list: {raw!r}. Expected comma-separated numbers."
        ) from e


# CLI option name -> grid knob name.  Options irrelevant to the chosen target
# are silently ignored (the knob is simply absent from that target's grid).
_OPTION_TO_KNOB = {
    "fl_class_threshold_raw": "fl_class_threshold",
    "min_fl_run_length_raw": "min_fl_run_length",
    "recovery_margin_raw": "recovery_amplitude_margin",
    "include_fallback_raw": "include_fallback",
    "flg_low_raw": "flg_low_threshold",
    "flg_high_raw": "flg_high_threshold",
}


def _fmt(v: float | int | None) -> str:
    """Format a cell; floats use adaptive significant digits so tiny magnitudes
    (the chance floor ~4e-5, precision ~1e-3) stay visible instead of collapsing
    to ``0.0000``.  Integers render plainly (no scientific notation)."""
    if v is None:
        return "N/A"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return fmt_sig(v, na="N/A")


def _render_table(result: SweepResult, top: int) -> None:
    """Render the ranked grid as a Rich table, current-defaults row highlighted."""
    knob_cols = list(result.rows[0].knobs.keys())
    title = (
        f"Top {min(top, len(result.rows))} of {len(result.rows)} "
        f"(ranked by {result.objective_label})"
    )
    table = Table(title=mark_provenance(title, Provenance.EXPERIMENTAL))
    table.add_column("#", justify="right")
    for knob in knob_cols:
        table.add_column(knob, justify="right")
    table.add_column("objective", justify="right")
    for col in result.metric_columns:
        table.add_column(col, justify="right")

    for rank, row in enumerate(result.rows[:top], start=1):
        marker = " (default)" if row.is_default else ""
        cells = [f"{rank}{marker}"]
        cells += [_fmt(row.knobs[k]) for k in knob_cols]
        cells.append(_fmt(row.objective))
        cells += [_fmt(row.metrics.get(c)) for c in result.metric_columns]
        table.add_row(*cells, style="bold cyan" if row.is_default else None)

    console.print(table)
    console.print("[bold cyan]Bold cyan[/bold cyan] row = current defaults.")


@click.command()
@date_range_options_required
@click.option(
    "--target",
    type=click.Choice(["flg", "re", "apple"]),
    required=True,
    help="Reference target: flg (device FLG AUC), re (machine RE), apple (Apple BD)",
)
@click.option(
    "--export",
    type=click.Path(),
    help="Write the FULL ranked grid to a .csv file",
)
@click.option(
    "--fl-class-threshold",
    "fl_class_threshold_raw",
    help="RERA-proxy flow_class threshold values (comma-separated; re/apple)",
)
@click.option(
    "--min-fl-run-length",
    "min_fl_run_length_raw",
    help="RERA-proxy minimum FL run length values (comma-separated; re/apple)",
)
@click.option(
    "--recovery-margin",
    "recovery_margin_raw",
    help="RERA-proxy recovery amplitude margin values (comma-separated; re/apple)",
)
@click.option(
    "--include-fallback",
    "include_fallback_raw",
    help="RERA-proxy fallback inclusion: 1=all breaths, 0=exclude low-confidence "
    "fallback guesses from FL-run detection (comma-separated; re/apple)",
)
@click.option(
    "--flg-low",
    "flg_low_raw",
    help="FLG AUC low-breakpoint values (comma-separated; flg)",
)
@click.option(
    "--flg-high",
    "flg_high_raw",
    help="FLG AUC high-breakpoint values (comma-separated; flg)",
)
@click.option(
    "--top",
    type=click.IntRange(min=1),
    default=15,
    show_default=True,
    help="Number of top-ranked rows to display (export always writes all)",
)
@profile_scoped_command
async def sweep_thresholds(
    ctx: CliCtx,
    date_from: datetime,
    date_to: datetime,
    target: str,
    export: str | None,
    fl_class_threshold_raw: str | None,
    min_fl_run_length_raw: str | None,
    recovery_margin_raw: str | None,
    include_fallback_raw: str | None,
    flg_low_raw: str | None,
    flg_high_raw: str | None,
    top: int,
) -> None:
    """
    Offline threshold-sweep harness for FL/RERA tuning.

    Loads stored breaths (and, per target, FLG waveforms / machine RE / Apple
    nightly signals) once, then re-scores a grid of query-time tunables and
    ranks the results.  Around current defaults the near-chance FL/RERA numbers
    reproduce — the point is that future tuning becomes data-driven, not that
    this run improves anything.
    """
    if date_from > date_to:
        raise click.ClickException("--from date must be before or equal to --to date")
    if export and Path(export).suffix != ".csv":
        raise click.ClickException("--export path must end in .csv")

    overrides = {
        "fl_class_threshold_raw": fl_class_threshold_raw,
        "min_fl_run_length_raw": min_fl_run_length_raw,
        "recovery_margin_raw": recovery_margin_raw,
        "include_fallback_raw": include_fallback_raw,
        "flg_low_raw": flg_low_raw,
        "flg_high_raw": flg_high_raw,
    }

    from snore.validation.sweep import (  # noqa: PLC0415
        DEFAULT_GRIDS,
        evaluate_grid,
        export_sweep_csv,
        load_sweep_data,
    )

    grid = {k: list(v) for k, v in DEFAULT_GRIDS[target].items()}
    for opt_name, raw in overrides.items():
        values = _parse_floats(raw)
        if values is None:
            continue
        knob = _OPTION_TO_KNOB[opt_name]
        if knob in grid:
            grid[knob] = values

    empty_knobs = [knob for knob, vals in grid.items() if not vals]
    if empty_knobs:
        raise click.ClickException(
            f"Empty parameter grid: no values for {', '.join(empty_knobs)}. "
            "Provide at least one value per swept knob."
        )

    async with cli_error_boundary("Sweep error"):
        console.print(
            f"Sweeping target={target} from {date_from.date()} to {date_to.date()}..."
        )
        data = await load_sweep_data(
            ctx.db,
            ctx.profile_id,
            date_from.strftime("%Y-%m-%d"),
            date_to.strftime("%Y-%m-%d"),
            target,
        )
        result = evaluate_grid(data, grid)

        console.print(
            f"\nLoaded {result.n_units_loaded} {result.unit_label}; "
            f"evaluated {len(result.rows)} grid combinations."
        )
        console.print(f"[dim]{result.notice}[/dim]")

        if result.reference:
            ref = ", ".join(f"{k}={_fmt(v)}" for k, v in result.reference.items())
            console.print(f"[dim]Reference: {ref}[/dim]")

        if not result.rows or result.n_units_loaded == 0:
            console.print("\n[yellow]No data to sweep in this range.[/yellow]")
        else:
            _render_table(result, top)

        if export:
            export_path = Path(export)
            export_sweep_csv(result, export_path)
            console.print(
                f"\nFull ranked grid ({len(result.rows)} rows) exported to "
                f"{export_path}"
            )
