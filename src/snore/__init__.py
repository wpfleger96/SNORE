"""
SNORE: Sleep eNvironment Observation & Respiratory Evaluation

MCP server for analyzing OSCAR CPAP/APAP therapy data.
"""

from typing import Any

__all__ = ["server"]


def __getattr__(name: str) -> Any:
    """Lazy load server to avoid circular imports at module level."""
    if name == "server":
        from snore.server import server

        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
