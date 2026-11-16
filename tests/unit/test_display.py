"""Unit tests for cli/display helpers."""

import asyncio

from io import StringIO
from unittest.mock import patch

import pytest

from rich.console import Console

from snore.analysis.modes.types import ModeResult
from snore.cli.display import (
    SEP_NARROW,
    SEP_WIDE,
    Column,
    _indent_prefix,
    experimental_legend,
    mark_experimental,
    mark_field,
    print_dry_run_complete,
    print_dry_run_header,
    print_header,
    print_kv,
    print_raw,
    print_success,
    print_table,
    print_warning,
    use_plain_legend,
)
from snore.provenance import EXPERIMENTAL_NOTE
from snore.services.schemas import SessionStatistics

LEGEND = f"* experimental: {EXPERIMENTAL_NOTE}"


@pytest.fixture()
def capture_stdout():
    buf = StringIO()
    # force_terminal=True so Rich renders markup and icons (tests check plain-text substrings)
    test_console = Console(file=buf, force_terminal=True, width=120)
    with patch("snore.cli.display.console", test_console):
        yield buf


@pytest.fixture()
def capture_stderr():
    buf = StringIO()
    # force_terminal=True so Rich renders markup and icons (tests check plain-text substrings)
    test_console = Console(file=buf, stderr=True, force_terminal=True, width=120)
    with patch("snore.cli.display.err_console", test_console):
        yield buf


class TestConsoleRouting:
    def test_print_success_routes_to_stdout(self, capture_stdout):
        print_success("done")
        assert "done" in capture_stdout.getvalue()
        assert "✓" in capture_stdout.getvalue()

    def test_print_warning_routes_to_stderr(self, capture_stderr):
        print_warning("watch out")
        assert "watch out" in capture_stderr.getvalue()
        assert "⚠" in capture_stderr.getvalue()


class TestIndentation:
    def test_indent_prefix_one(self):
        assert _indent_prefix(1) == "  "

    def test_print_success_with_indent(self, capture_stdout):
        print_success("msg", indent=2)
        output = capture_stdout.getvalue()
        assert output.startswith("    ")


class TestPrintRaw:
    def test_brackets_not_parsed_as_markup(self, capture_stdout):
        print_raw("[disabled] session")
        output = capture_stdout.getvalue()
        assert "[disabled]" in output

    def test_rich_markup_not_applied(self, capture_stdout):
        print_raw("[bold]not bold[/bold]")
        output = capture_stdout.getvalue()
        assert "[bold]" in output
        assert "[/bold]" in output


class TestSeparators:
    def test_header_contains_title_and_separator(self, capture_stdout):
        print_header("My Title")
        output = capture_stdout.getvalue()
        assert "My Title" in output
        assert "=" * SEP_NARROW in output

    def test_header_wide_uses_wide_separator(self, capture_stdout):
        print_header("Wide Title", wide=True)
        output = capture_stdout.getvalue()
        assert "Wide Title" in output
        assert "=" * SEP_WIDE in output


class TestKeyValue:
    def test_kv_contains_key_and_value(self, capture_stdout):
        print_kv("Name", "Alice")
        output = capture_stdout.getvalue()
        assert "Name" in output
        assert "Alice" in output


class TestTable:
    def test_table_prints_padded_headers_then_rows(self, capture_stdout):
        print_table(
            [Column("Day", 6), Column("AHI", 6), Column("FLI", 0)],
            [("d1", "1.0", "0.2")],
        )
        header, _separator, row = capture_stdout.getvalue().splitlines()
        assert header == "Day    AHI    FLI"
        assert row == "d1     1.0    0.2"


class TestExperimentalMarker:
    def test_mark_experimental_appends_marker(self):
        assert mark_experimental("x") == "x*"

    def test_mark_field_experimental_field_is_marked(self):
        assert mark_field("AHI", ModeResult, "ahi") == "AHI*"

    def test_mark_field_non_experimental_field_is_plain(self):
        assert mark_field("AHI", SessionStatistics, "ahi") == "AHI"

    def test_mark_field_non_experimental_field_needs_no_legend(self, capture_stdout):
        with experimental_legend():
            mark_field("AHI", SessionStatistics, "ahi")
        assert capture_stdout.getvalue() == ""

    def test_legend_printed_once_for_several_markers(self, capture_stdout):
        with experimental_legend():
            print_kv(mark_experimental("A"), "1")
            print_kv(mark_experimental("B"), "2")
            print_kv("C", "3")
        output = capture_stdout.getvalue()
        assert "A*:" in output
        assert "C:" in output
        assert output.count("* experimental") == 1
        assert LEGEND in output.splitlines()[-1]

    def test_legend_without_markers_prints_nothing(self, capture_stdout):
        with experimental_legend():
            print_kv("OA", "3")
        assert "experimental" not in capture_stdout.getvalue()

    def test_legend_printed_when_block_raises_after_markers(self, capture_stdout):
        with pytest.raises(SystemExit), experimental_legend():
            mark_experimental("x")
            raise SystemExit(1)
        assert LEGEND in capture_stdout.getvalue()

    def test_legend_skipped_when_block_raises_before_markers(self, capture_stdout):
        with pytest.raises(RuntimeError), experimental_legend():
            raise RuntimeError
        assert capture_stdout.getvalue() == ""

    def test_legend_collects_marker_emitted_inside_asyncio_run(self, capture_stdout):
        async def _emit() -> None:
            mark_experimental("x")

        with experimental_legend():
            asyncio.run(_emit())
        assert LEGEND in capture_stdout.getvalue()

    def test_legend_is_dim_by_default(self, capture_stdout):
        with experimental_legend():
            mark_experimental("x")
        assert "\x1b[2m" in capture_stdout.getvalue()

    def test_plain_legend_has_no_styling(self, capture_stdout):
        with experimental_legend():
            use_plain_legend()
            mark_experimental("x")
        assert capture_stdout.getvalue() == f"{LEGEND}\n"

    def test_markers_do_not_leak_into_next_legend(self, capture_stdout):
        with experimental_legend():
            mark_experimental("x")
        mark_experimental("outside")
        with experimental_legend():
            pass
        assert capture_stdout.getvalue().count("* experimental") == 1


class TestDryRun:
    def test_dry_run_header_contains_mode_label(self, capture_stdout):
        print_dry_run_header()
        output = capture_stdout.getvalue()
        assert "DRY RUN MODE" in output

    def test_dry_run_complete_custom_verb(self, capture_stdout):
        print_dry_run_complete("import")
        output = capture_stdout.getvalue()
        assert "Dry run complete" in output
        assert "import" in output
