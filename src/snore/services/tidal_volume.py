"""Tidal volume unit normalization for values read from ``Statistics``."""

from snore.constants import OSCAR_BINARY_PARSER_ID


def tidal_volume_ml(value: float | None, import_source: str | None) -> float | None:
    """Convert a stored ``Statistics.tidal_volume_*`` value to mL.

    OSCAR stores tidal volume in mL (OSCAR schema.cpp:244).  ResMed STR ``TidVol``
    is stored as read, in litres (OSCAR scales it, resmed_loader.cpp :1990-1998).
    """
    if value is None:
        return None
    return round(value if import_source == OSCAR_BINARY_PARSER_ID else value * 1000, 1)
