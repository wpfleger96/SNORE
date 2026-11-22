"""Tidal volume unit normalization for values read from ``Statistics``."""

from collections.abc import Iterable

from snore.constants import OSCAR_BINARY_PARSER_ID
from snore.utils.stats import weighted_mean


def tidal_volume_ml(value: float | None, import_source: str | None) -> float | None:
    """Convert a stored ``Statistics.tidal_volume_*`` value to mL.

    OSCAR stores tidal volume in mL (OSCAR schema.cpp:244).  ResMed STR ``TidVol``
    is stored as read, in litres (OSCAR scales it, resmed_loader.cpp :1990-1998).
    """
    if value is None:
        return None
    return round(value if import_source == OSCAR_BINARY_PARSER_ID else value * 1000, 1)


def usage_weighted_tidal_volume_ml(
    sessions: Iterable[tuple[float | None, str | None, float | None]],
) -> float | None:
    """Usage-weighted mean tidal volume (mL) of sessions' stored values.

    Each item is ``(Statistics.tidal_volume_mean, Session.import_source,
    Statistics.usage_hours)``; sessions without a value or positive usage hours
    are skipped, and None means none qualifies.
    """
    return weighted_mean(
        (tv, hours)
        for value, import_source, hours in sessions
        if hours
        and hours > 0
        and (tv := tidal_volume_ml(value, import_source)) is not None
    )
