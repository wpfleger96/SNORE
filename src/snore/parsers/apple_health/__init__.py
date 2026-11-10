"""Apple Health parser package.

Public surface consumed by the importer layer::

    from snore.parsers.apple_health import (
        AppleHealthParser,
        RawHealthRecord,
        iter_records,
    )
"""

from __future__ import annotations

from snore.parsers.apple_health.models import RawHealthRecord
from snore.parsers.apple_health.parser import AppleHealthParser
from snore.parsers.apple_health.xml_reader import iter_records

__all__ = [
    "AppleHealthParser",
    "RawHealthRecord",
    "iter_records",
]
