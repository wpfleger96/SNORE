"""Chunked ID binding for SQLite's bound-parameter cap."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import batched

ID_CHUNK_SIZE: int = 500
"""Max IDs bound into a single ``col.in_(...)`` predicate.

SQLite caps bound parameters per statement at SQLITE_MAX_VARIABLE_NUMBER: 999 on builds
before 3.32, 32766 after.  Chunking to the 999 floor means an unbounded caller ID list can
never trip ``sqlite3.OperationalError: too many SQL variables`` on any build.  500 matches
the export precedent (``export_service._EXPORT_CHUNK_SIZE``) and leaves headroom for a
handful of *other* bound params in the same statement.  A statement that binds more than one
risky IN-list must chunk each list separately or be split (see the delete-preview count in
session_service).
"""


def iter_id_chunks[T](
    ids: Sequence[T], size: int | None = None
) -> Iterator[tuple[T, ...]]:
    """Yield ``ids`` in tuples of at most ``size`` (default ``ID_CHUNK_SIZE``) for IN-binding.

    Empty input yields nothing; a list already within the cap yields exactly one chunk (no
    behavioural or performance change for the common small case).  SQLAlchemy accepts a
    tuple in ``.in_()``.  The element type is preserved, so non-integer bind values (e.g.
    ``date`` objects) are chunked against the same parameter cap too.

    A smaller ``size`` is for callers that bound the work done per chunk (e.g. one write
    transaction per chunk).  It must not exceed ``ID_CHUNK_SIZE``: a larger chunk defeats the
    bound-parameter cap this helper exists to enforce.
    """
    yield from batched(ids, ID_CHUNK_SIZE if size is None else size, strict=False)
