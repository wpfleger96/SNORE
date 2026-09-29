"""Tool scaffold: shared helpers imported by every MCP tool module.

Lives here (not in server.py) so tool modules can import these at module level
without creating an import cycle.  The cycle would be:
  server.py → (registers) tool modules → tool modules → server.py

By moving the shared pieces here, tool modules depend on _scaffold (no cycle),
and server.py imports from _scaffold for its own use and to re-export for
callers that import them from server.py.

``SNORERuntime`` is referenced only under TYPE_CHECKING so no runtime import
from server.py occurs.
"""

from __future__ import annotations

import inspect
import json
import logging
import re

from collections.abc import Awaitable, Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import ValidationError as PydanticValidationError

from snore.constants import FL_RERA_EXPERIMENTAL_DISCLAIMER
from snore.mcp.errors import ValidationError

if TYPE_CHECKING:
    from snore.mcp.server import SNORERuntime

logger = logging.getLogger(__name__)

RESPONSE_SIZE_LIMIT = 500_000  # bytes; tools return narrow-your-query guidance

# Pydantic v2 wraps validator-raised ValueErrors with this prefix in the "msg" field.
# Stripping it produces cleaner user-visible messages. If a future pydantic upgrade
# changes this prefix, the strip becomes a no-op and messages will include the prefix
# again — that breakage should be caught by TestToolErrorBoundary.test_pydantic_value_error_prefix_stripped.
_PYDANTIC_VALUE_ERROR_PREFIX = "Value error, "


def tool_error_boundary(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Convert common tool failures into ToolError so FastMCP sets isError=true."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except ToolError:
            raise
        except PydanticValidationError as exc:
            parts: list[str] = []
            for err in exc.errors():
                msg = err["msg"].removeprefix(_PYDANTIC_VALUE_ERROR_PREFIX)
                loc = err.get("loc", ())
                if loc:
                    parts.append(f"{'.'.join(str(p) for p in loc)}: {msg}")
                else:
                    parts.append(msg)
            raise ToolError("; ".join(parts)) from exc
        except (ValidationError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        except Exception as exc:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            if status is not None:
                raise ToolError(f"HTTP {status} from upstream service") from exc
            logger.exception("Unexpected error in tool call")
            raise ToolError("An unexpected error occurred.") from exc

    return wrapper


# A Google-style docstring section header on its own line ("Args:", "Returns:",
# "Refusal semantics:", ...). FastMCP keeps only the leading prose as the tool
# description and turns everything from the first such header onward into
# parameter/return metadata — so the disclaimer must be injected BEFORE it, not
# appended to the end of the docstring where it would be discarded.
_DOC_SECTION_HEADER = re.compile(r"^[A-Z][A-Za-z ]*:$")


def _with_fl_rera_disclaimer(
    func: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Inject the FL/RERA experimental disclaimer into a tool's description.

    ``@mcp.tool()`` reads ``__doc__`` at decoration time and docstrings are static
    string literals, so the shared constant cannot be interpolated into the
    literal itself. Apply this decorator BELOW ``@mcp.tool()`` (decorators run
    bottom-up) so the appended text is present when FastMCP captures the tool
    description that clients see. The disclaimer is placed as its own paragraph
    at the end of the leading prose (before the first ``Args:``-style section),
    which is the region FastMCP surfaces as the client-visible description.
    """
    if not func.__doc__:
        return func
    lines = inspect.cleandoc(func.__doc__).splitlines()
    disclaimer = f"FL/RERA proxy metrics: {FL_RERA_EXPERIMENTAL_DISCLAIMER}"
    insert_at = next(
        (i for i, line in enumerate(lines) if _DOC_SECTION_HEADER.match(line)),
        len(lines),
    )
    prefix = [] if insert_at > 0 and lines[insert_at - 1] == "" else [""]
    suffix = [""] if insert_at < len(lines) else []
    block = prefix + [disclaimer] + suffix
    func.__doc__ = "\n".join(lines[:insert_at] + block + lines[insert_at:])
    return func


def _check_response_size(result: Any, tool_name: str) -> None:
    """Raise ToolError if the serialized result exceeds RESPONSE_SIZE_LIMIT.

    Cost note: this re-serializes the already-model_dumped payload purely to
    measure its byte length — O(response size) work on every successful call,
    bounded by the page/window/epoch caps and the size limit itself.
    """
    try:
        size = len(json.dumps(result, default=str).encode("utf-8"))
    except Exception:
        logger.warning(
            "_check_response_size: measurement failed for %s; skipping size gate",
            tool_name,
        )
        return
    if size > RESPONSE_SIZE_LIMIT:
        raise ToolError(
            f"Response from {tool_name} exceeds the {RESPONSE_SIZE_LIMIT:,}-byte limit. "
            "Narrow your query: use a shorter date range, smaller page_size, or add "
            "device/type filters."
        )


def _runtime(ctx: Context) -> SNORERuntime:
    """Extract the SNORERuntime from the FastMCP context.

    fastmcp types ``ctx.lifespan_context`` as ``dict[str, Any]``; cast narrows
    it to the actual yielded type without changing runtime behaviour.
    """
    return cast("SNORERuntime", ctx.lifespan_context)


async def _scope_and_run(
    ctx: Context,
    impl: Callable[..., Awaitable[Any]],
    *,
    tool_name: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Open a DB scope, call impl, model_dump the result, and size-check it.

    Shared scaffold for the 8 standard-pattern tools (all except the waveform pair,
    which has non-standard return paths).  Each tool module's
    ``register`` closure calls this with the resolved date/validation kwargs.
    """
    runtime = _runtime(ctx)
    async with runtime.scope_provider() as db:
        result = await impl(db, profile_id=runtime.profile_id, **kwargs)
    payload: dict[str, Any] = result.model_dump(mode="json")
    _check_response_size(payload, tool_name)
    return payload
