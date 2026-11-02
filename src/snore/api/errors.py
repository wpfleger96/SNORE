import logging

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

from snore.api.middleware import _AUTH_PATH_PREFIX
from snore.exceptions import NotFoundError

logger = logging.getLogger(__name__)

__all__ = [
    "NotFoundError",
    "auth_validation_error_handler",
    "db_busy_maps_to_409",
    "not_found_handler",
    "server_error_handler",
]


@asynccontextmanager
async def db_busy_maps_to_409() -> AsyncGenerator[None]:
    """Map SQLite write-lock contention to 409 instead of a generic 500.

    Covers writers the reset job intentionally does not serialize
    (imports, analysis jobs).
    """
    try:
        yield
    except OperationalError as exc:
        if "database is locked" in str(exc).lower():
            logger.warning("SQLite write-lock contention mapped to 409: %s", exc)
            raise HTTPException(
                status_code=409,
                detail="Database is busy (an import or analysis may be running) — try again shortly",
            ) from None
        raise


class ErrorResponse(BaseModel):
    error: str
    message: str


async def auth_validation_error_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Return 422 errors on auth routes without echoing submitted credential values.

    Pydantic's default ``RequestValidationError`` serialization includes an
    ``input`` field containing the rejected value verbatim.  On auth endpoints
    this exposes tokens and passwords in error bodies, contradicting the
    credential-hygiene invariant.  This handler strips ``input`` from each
    error and applies ``Cache-Control: no-store``.

    Non-auth paths fall through to FastAPI's default handler.
    """
    if not request.url.path.startswith(_AUTH_PATH_PREFIX):
        if isinstance(exc, RequestValidationError):
            return await request_validation_exception_handler(request, exc)
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    sanitized = []
    if isinstance(exc, RequestValidationError):
        sanitized = [
            {k: v for k, v in error.items() if k != "input"} for error in exc.errors()
        ]
    return JSONResponse(
        status_code=422,
        content={"detail": sanitized},
        headers={"Cache-Control": "no-store"},
    )


async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(error="not_found", message=str(exc)).model_dump(),
    )


async def server_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled server error on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="server_error", message="Internal server error"
        ).model_dump(),
    )
