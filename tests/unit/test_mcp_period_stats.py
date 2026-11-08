"""get_period_stats argument validation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from snore.mcp.errors import ValidationError
from snore.mcp.tools.period_stats import _validate_range
from snore.mcp.tools.summary import MAX_NIGHTLY_RANGE

START = date(2026, 1, 1)


def test_reversed_range_is_rejected() -> None:
    with pytest.raises(ValidationError, match="on or before"):
        _validate_range("month", START, START - timedelta(days=1))


@pytest.mark.parametrize("bounds", [(None, None), (START, None), (None, START)])
def test_open_range_is_allowed_for_coarse_periods(
    bounds: tuple[date | None, date | None],
) -> None:
    _validate_range("week", *bounds)


@pytest.mark.parametrize("bounds", [(None, None), (START, None), (None, START)])
def test_day_periods_require_both_bounds(
    bounds: tuple[date | None, date | None],
) -> None:
    with pytest.raises(ValidationError, match="requires both"):
        _validate_range("day", *bounds)


def test_day_periods_accept_the_nightly_range_cap() -> None:
    _validate_range("day", START, START + timedelta(days=MAX_NIGHTLY_RANGE - 1))


def test_day_periods_reject_ranges_over_the_cap() -> None:
    with pytest.raises(ValidationError, match=f"at most {MAX_NIGHTLY_RANGE}"):
        _validate_range("day", START, START + timedelta(days=MAX_NIGHTLY_RANGE))
