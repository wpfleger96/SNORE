"""Shared SessionCoverage → SessionCoverageEntry mapping helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from snore.mcp.schemas import SessionCoverageEntry

if TYPE_CHECKING:
    from snore.services.breath_service import SessionCoverage


def map_session_coverage(
    coverage: list[SessionCoverage],
) -> list[SessionCoverageEntry]:
    """Map service SessionCoverage DTOs to MCP SessionCoverageEntry schema objects.

    Keeps the algo_versions model_dump branch out of the tool adapters.
    """
    return [
        SessionCoverageEntry(
            session_id=c.session_id,
            analysis_status=str(c.analysis_status),
            algo_versions=c.algo_versions.model_dump(mode="json")
            if c.algo_versions is not None
            else None,
        )
        for c in coverage
    ]
