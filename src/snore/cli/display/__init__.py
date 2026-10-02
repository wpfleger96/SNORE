"""Centralized CLI display helpers — the single source of truth for all terminal output."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import NamedTuple

from rich.console import Console
from rich.markup import escape

from snore.provenance import Provenance

console = Console()
err_console = Console(stderr=True)

SEP_WIDE = 80
SEP_NARROW = 60

ICON_CHECK = "[green]✓[/green]"
ICON_WARN = "[yellow]⚠[/yellow]"
ICON_ERROR = "[red]✗[/red]"
ICON_SKIP = "[dim]⊙[/dim]"
ICON_DRY_RUN = "\U0001f50d"
ICON_SCAN = "\U0001f4c2"
ICON_STATS = "\U0001f4ca"
ICON_IMPORT = "\U0001f4e5"
ICON_BACKUP = "\U0001f4e6"
ICON_FILTERS = "\U0001f4cb"
ICON_TIP = "[yellow]\U0001f4a1[/yellow]"
ICON_CHART = "\U0001f4c8"


_PROVENANCE_MARKERS: dict[Provenance, str] = {
    Provenance.EXPERIMENTAL: "*",
    Provenance.DERIVED: "†",
    Provenance.DEVICE: "",
}
_PROVENANCE_LEGEND: dict[Provenance, str] = {
    Provenance.EXPERIMENTAL: "* experimental (SNORE heuristic)",
    Provenance.DERIVED: "† derived (computed by SNORE from device data)",
}

# Tiers marked during the current command; None outside provenance_legend().
_used_provenance: ContextVar[set[Provenance] | None] = ContextVar(
    "_used_provenance", default=None
)


def mark_provenance(label: str, provenance: Provenance | None) -> str:
    """Append the tier's marker to ``label`` and record it for the legend."""
    if provenance is None or provenance is Provenance.DEVICE:
        return label
    used = _used_provenance.get()
    if used is not None:
        used.add(provenance)
    return f"{label}{_PROVENANCE_MARKERS[provenance]}"


@contextmanager
def provenance_legend() -> Iterator[None]:
    """Collect markers emitted inside the block; print their legend on clean exit."""
    used: set[Provenance] = set()
    token = _used_provenance.set(used)
    try:
        yield
    finally:
        _used_provenance.reset(token)
    parts = [text for tier, text in _PROVENANCE_LEGEND.items() if tier in used]
    if parts:
        console.print(f"[dim]{'  '.join(parts)}[/dim]", highlight=False)


def fmt_sig(v: float | None, *, na: str = "N/A") -> str:
    """Adaptive formatting so tiny magnitudes stay visible.

    Values such as a chance-precision floor (~4e-5) and proxy precision (~1e-3)
    collapse to ``0.000`` at fixed 3 decimals — comparing floor vs precision is
    the whole point of those fields, so render small values with more decimals
    or in scientific notation.  Callers keep the full float in their models and
    JSON/CSV exports; this only affects terminal display.
    """
    if v is None:
        return na
    if v == 0.0:
        return "0"
    a = abs(v)
    if a >= 0.1:
        return f"{v:.3f}"
    if a >= 1e-3:
        return f"{v:.4f}"
    return f"{v:.2e}"


def _indent_prefix(indent: int) -> str:
    return "  " * indent


def print_success(message: str, *, indent: int = 0) -> None:
    console.print(f"{_indent_prefix(indent)}{ICON_CHECK} {message}")


def print_warning(message: str, *, indent: int = 0) -> None:
    err_console.print(f"{_indent_prefix(indent)}{ICON_WARN} {message}")


def print_error(message: str, *, indent: int = 0) -> None:
    err_console.print(f"{_indent_prefix(indent)}{ICON_ERROR} {message}")


def print_skip(message: str, *, indent: int = 0) -> None:
    console.print(f"{_indent_prefix(indent)}{ICON_SKIP} {message}")


def print_info(message: str, *, indent: int = 0) -> None:
    console.print(f"{_indent_prefix(indent)}{message}")


def print_tip(message: str) -> None:
    console.print(f"{ICON_TIP} Tip: {message}")


def print_raw(message: str, *, indent: int = 0) -> None:
    console.print(f"{_indent_prefix(indent)}{message}", markup=False, highlight=False)


def print_header(title: str, icon: str = "", *, wide: bool = False) -> None:
    width = SEP_WIDE if wide else SEP_NARROW
    prefix = f"{icon} " if icon else ""
    console.print(f"\n{prefix}{title}")
    console.print("=" * width)


def print_footer(*, wide: bool = False) -> None:
    width = SEP_WIDE if wide else SEP_NARROW
    console.print("=" * width)


def print_separator(*, wide: bool = False) -> None:
    width = SEP_WIDE if wide else SEP_NARROW
    console.print("-" * width)


class Column(NamedTuple):
    """A ``print_table`` column; a width of 0 leaves the cell unpadded."""

    header: str
    width: int
    provenance: Provenance | None = None


def print_table(
    columns: Sequence[Column],
    rows: Iterable[Sequence[str]],
    *,
    header_separator: bool = True,
    wide: bool = True,
    indent: int = 0,
) -> None:
    """Print a plain-text table of left-aligned, space-separated columns.

    A column's provenance marks its header; a width of 0 suits a ragged last
    column. Cells longer than their column width are not truncated.
    """
    prefix = _indent_prefix(indent)
    widths = [col.width for col in columns]
    headers = [mark_provenance(col.header, col.provenance) for col in columns]

    def _format_row(cells: Sequence[str]) -> str:
        return prefix + " ".join(
            f"{cell:<{width}}" if width else cell
            for cell, width in zip(cells, widths, strict=True)
        )

    console.print(_format_row(headers), markup=False, highlight=False)
    if header_separator:
        print_separator(wide=wide)
    for row in rows:
        console.print(_format_row(row), markup=False, highlight=False)


def print_subsection(title: str) -> None:
    console.print(f"\n{title}")


def print_kv(
    key: str,
    value: str,
    *,
    indent: int = 1,
    provenance: Provenance | None = None,
) -> None:
    key = mark_provenance(key, provenance)
    console.print(f"{_indent_prefix(indent)}[dim]{escape(key)}:[/dim] {escape(value)}")


def print_dry_run_header(action: str = "imported") -> None:
    console.print(f"\n{ICON_DRY_RUN} DRY RUN MODE - No data will be {action}\n")


def print_dry_run_complete(action_verb: str = "run") -> None:
    print_success(f"Dry run complete. Use without --dry-run to {action_verb}.")
