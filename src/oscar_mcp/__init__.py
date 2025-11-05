"""
OSCAR-MCP: MCP server for analyzing OSCAR CPAP/APAP therapy data.
"""

__all__ = ["server"]


def __getattr__(name):
    """Lazy load server to avoid circular imports at module level."""
    if name == "server":
        from oscar_mcp.server import server

        return server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
