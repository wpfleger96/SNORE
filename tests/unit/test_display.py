"""Unit tests for cli/display helpers."""

from io import StringIO
from unittest.mock import patch

import pytest

from rich.console import Console

from snore.cli.display import (
    SEP_NARROW,
    SEP_WIDE,
    Column,
    _indent_prefix,
    mark_field,
    mark_provenance,
    print_dry_run_complete,
    print_dry_run_header,
    print_header,
    print_kv,
    print_raw,
    print_success,
    print_table,
    print_warning,
    provenance_legend,
    use_plain_legend,
)
from snore.provenance import PROVENANCE_NOTES, Provenance
from snore.services.schemas import SessionStatistics


def _legend_part(tier: Provenance) -> str:
    marker = {Provenance.DERIVED: "†", Provenance.EXPERIMENTAL: "*"}[tier]
    return f"{marker} {tier.value}: {PROVENANCE_NOTES[tier]}"


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

    def test_kv_with_derived_provenance_marks_key(self, capture_stdout):
        print_kv("Avg AHI", "2.1", provenance=Provenance.DERIVED)
        assert "Avg AHI†:" in capture_stdout.getvalue()

    def test_kv_with_device_provenance_is_unmarked(self, capture_stdout):
        print_kv("OA", "3", provenance=Provenance.DEVICE)
        assert "OA:" in capture_stdout.getvalue()


class TestProvenance:
    def test_marker_per_tier(self):
        assert mark_provenance("x", Provenance.EXPERIMENTAL) == "x*"
        assert mark_provenance("x", Provenance.DERIVED) == "x†"
        assert mark_provenance("x", Provenance.DEVICE) == "x"
        assert mark_provenance("x", None) == "x"

    def test_table_column_provenance_marks_only_that_header(self, capture_stdout):
        print_table(
            [Column("Day", 6), Column("AHI", 6, Provenance.DERIVED), Column("FLI", 0)],
            [("d1", "1.0", "0.2")],
        )
        header = capture_stdout.getvalue().splitlines()[0]
        assert header.split() == ["Day", "AHI†", "FLI"]

    def test_mark_field_reads_tier_from_model_tag(self):
        assert mark_field("AHI", SessionStatistics, "ahi") == "AHI†"

    def test_legend_lists_only_used_markers_once(self, capture_stdout):
        with provenance_legend():
            print_kv("A", "1", provenance=Provenance.DERIVED)
            print_kv("B", "2", provenance=Provenance.DERIVED)
            print_kv("C", "3", provenance=Provenance.DEVICE)
        output = capture_stdout.getvalue()
        assert output.count("† derived") == 1
        assert "* experimental" not in output
        assert _legend_part(Provenance.DERIVED) in output.splitlines()[-1]

    def test_legend_lists_derived_before_experimental(self, capture_stdout):
        with provenance_legend():
            mark_provenance("y", Provenance.EXPERIMENTAL)
            mark_provenance("x", Provenance.DERIVED)
        expected = (
            f"{_legend_part(Provenance.DERIVED)}  "
            f"{_legend_part(Provenance.EXPERIMENTAL)}"
        )
        assert expected in capture_stdout.getvalue().splitlines()[-1]

    def test_legend_without_markers_prints_nothing(self, capture_stdout):
        with provenance_legend():
            print_kv("OA", "3", provenance=Provenance.DEVICE)
        assert "derived" not in capture_stdout.getvalue()

    def test_legend_printed_when_block_raises_after_markers(self, capture_stdout):
        with pytest.raises(SystemExit), provenance_legend():
            mark_provenance("x", Provenance.EXPERIMENTAL)
            raise SystemExit(1)
        assert _legend_part(Provenance.EXPERIMENTAL) in capture_stdout.getvalue()

    def test_legend_skipped_when_block_raises_before_markers(self, capture_stdout):
        with pytest.raises(RuntimeError), provenance_legend():
            raise RuntimeError
        assert capture_stdout.getvalue() == ""

    def test_legend_is_dim_by_default(self, capture_stdout):
        with provenance_legend():
            mark_provenance("x", Provenance.DERIVED)
        assert "\x1b[2m" in capture_stdout.getvalue()

    def test_plain_legend_has_no_styling(self, capture_stdout):
        with provenance_legend():
            use_plain_legend()
            mark_provenance("x", Provenance.DERIVED)
        assert capture_stdout.getvalue() == f"{_legend_part(Provenance.DERIVED)}\n"

    def test_markers_do_not_leak_into_next_legend(self, capture_stdout):
        with provenance_legend():
            mark_provenance("x", Provenance.EXPERIMENTAL)
        mark_provenance("outside", Provenance.DERIVED)
        with provenance_legend():
            pass
        assert capture_stdout.getvalue().count("* experimental") == 1
        assert "derived" not in capture_stdout.getvalue()


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
