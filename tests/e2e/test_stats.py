"""Therapy-statistics views over the multi-night dataset.

`stats` aggregates across days, so it needs more than one night to be meaningful.
Numeric values on the recorded nights are sparse, so these assert command-level
success plus the structural output (period rows, month labels) — enough to catch
a stats/aggregation regression without coupling to exact figures.
"""

from __future__ import annotations

# Months present in the multi-night dataset (by Day date / noon-to-noon).
EXPECTED_MONTHS = ["Jun 2024", "Jan 2025", "Aug 2025", "Sep 2025", "Oct 2025"]


def test_stats_monthly_breakdown_lists_every_month(snore, multi_night_db):
    result = snore("stats", "--period", "month", db=multi_night_db)
    assert result.returncode == 0, result.stderr or result.stdout
    for month in EXPECTED_MONTHS:
        assert month in result.stdout, f"missing month row {month!r}"
    # The fixed device night's AHI is deterministic in its month row.  The
    # 2024-06-20 night imports 34 min of the 92 min the device's STR record
    # covers, so its device AHI (0.6) fails the coverage check and the row
    # shows the recount (10 events / 0.567 h = 17.6).
    jun_row = next(
        line for line in result.stdout.splitlines() if line.startswith("Jun 2024")
    )
    _month, _year, _days, _hours, avg_ahi, _med_ahi = jun_row.split()
    assert avg_ahi == "17.6", jun_row
