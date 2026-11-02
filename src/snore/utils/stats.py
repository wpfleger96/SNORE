"""Small numeric helpers shared across statistics code."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


def nearest_rank_index(n: int, q: float) -> int:
    """Index of the nearest-rank ``q`` quantile among ``n`` ascending values."""
    return min(int(n * q), n - 1)


def percentile_nearest_rank(sorted_vals: Sequence[float], q: float) -> float | None:
    """Return the nearest-rank percentile of a pre-sorted sequence.

    Args:
        sorted_vals: Values sorted in ascending order.
        q: Quantile in ``[0, 1]``.

    Returns:
        ``sorted_vals[nearest_rank_index(len(sorted_vals), q)]``, or ``None``
        when the sequence is empty.
    """
    if not sorted_vals:
        return None
    return sorted_vals[nearest_rank_index(len(sorted_vals), q)]


def fraction_at_or_above(vals: Sequence[float], threshold: float) -> float | None:
    """Return the fraction of ``vals`` at or above ``threshold``, ``None`` when empty."""
    if not vals:
        return None
    return sum(1 for v in vals if v >= threshold) / len(vals)


def weighted_mean(pairs: Iterable[tuple[float, float]]) -> float | None:
    """Return the weighted mean of ``(value, weight)`` pairs.

    Returns ``None`` when the total weight is zero or negative (including the
    empty case).
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for value, weight in pairs:
        weighted_sum += value * weight
        total_weight += weight
    return weighted_sum / total_weight if total_weight > 0 else None


def usage_weighted_means(
    rows: Iterable[Any],
    field_map: Mapping[str, str],
    get_hours: Callable[[Any], float | None],
) -> dict[str, float | None]:
    """Compute usage-hours-weighted means of several row attributes at once.

    Rows whose hours are ``None`` or ``<= 0`` are skipped entirely.  For each
    output key, rows where ``getattr(row, field)`` is ``None`` are excluded
    from both that key's weighted sum and its accumulated hours, so each key
    is averaged only over the hours that actually carried a value.

    Args:
        rows: Objects exposing the attributes named in ``field_map``.
        field_map: Output key -> row attribute name.
        get_hours: Extracts the weight (usage hours) from a row.

    Returns:
        Output key -> weighted mean, ``None`` for keys with no accumulated
        hours.
    """
    weighted_sums = {key: 0.0 for key in field_map}
    hours_for = {key: 0.0 for key in field_map}
    for row in rows:
        hours = get_hours(row)
        if not hours or hours <= 0:
            continue
        for key, field in field_map.items():
            val = getattr(row, field)
            if val is not None:
                weighted_sums[key] += val * hours
                hours_for[key] += hours
    return {
        key: weighted_sums[key] / hours_for[key] if hours_for[key] > 0 else None
        for key in field_map
    }
