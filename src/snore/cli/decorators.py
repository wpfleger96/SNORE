"""Shared CLI decorators and utilities."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class CliCtx:
    """Runtime context injected into ``@profile_scoped_command`` bodies."""

    db: AsyncSession
    profile_id: int


def init_db(db: str | None) -> None:
    """Initialize the database, resolving the path if provided.

    Bridges the sync Click command boundary to the async ``init_database``
    coroutine by running it in a new event loop.
    """
    import asyncio  # noqa: PLC0415

    from snore.database.session import init_database  # noqa: PLC0415

    asyncio.run(init_database(str(Path(db).expanduser()) if db else None))


@asynccontextmanager
async def db_session(db: str | None) -> AsyncIterator[AsyncSession]:
    """Async CLI bridge: initialise the database and yield an AsyncSession.

    Each CLI command wraps its body in ``asyncio.run`` and uses this context
    manager to obtain a session.  The session is committed on clean exit and
    rolled back on exception.
    """
    from snore.database.session import init_database, session_scope  # noqa: PLC0415

    await init_database(str(Path(db).expanduser()) if db else None)
    async with session_scope() as session:
        yield session


@asynccontextmanager
async def profile_session(
    db: str | None,
    actor_user: str | None,
    actor_profile: str | None,
) -> AsyncIterator[CliCtx]:
    """Yield a CLI database session with its actor profile resolved."""
    from snore.auth.factory import resolve_cli_profile_id  # noqa: PLC0415

    async with db_session(db) as session:
        profile_id = await resolve_cli_profile_id(session, actor_user, actor_profile)
        yield CliCtx(db=session, profile_id=profile_id)


async def resolve_profile_id_once(
    db: str | None,
    actor_user: str | None,
    actor_profile: str | None,
) -> int:
    """Resolve a CLI actor profile in a short-lived database session."""
    async with profile_session(db, actor_user, actor_profile) as ctx:
        return ctx.profile_id


@asynccontextmanager
async def cli_error_boundary(label: str) -> AsyncIterator[None]:
    """Standard CLI error boundary: ClickException passes through; any other
    exception is reported to stderr (with a traceback at DEBUG) and re-raised
    as a ClickException."""
    import logging  # noqa: PLC0415
    import traceback  # noqa: PLC0415

    from snore.cli.display import err_console  # noqa: PLC0415

    try:
        yield
    except click.ClickException:
        raise
    except Exception as e:
        err_console.print(f"{label}: {e}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            traceback.print_exc()
        raise click.ClickException(str(e)) from e


def db_option(f: Any) -> Any:
    """Shared --db option for commands that access the database."""
    return click.option(
        "--db",
        default=None,
        type=click.Path(),
        help="Path to SQLite database file",
    )(f)


def device_option(f: Any) -> Any:
    """Shared --device/-d option for filtering by device serial number."""
    return click.option("--device", "-d", help="Device serial number")(f)


def session_id_date_options(f: Any) -> Any:
    """Shared --session-id/--date options for selecting a session."""
    f = click.option(
        "--date",
        type=click.DateTime(formats=["%Y-%m-%d"]),
        help="Session date (YYYY-MM-DD)",
    )(f)
    f = click.option("--session-id", type=int, help="Session ID")(f)
    return f


def date_range_options(f: Any) -> Any:
    """Shared --from/--to date range options."""
    f = click.option(
        "--to",
        "date_to",
        default=None,
        type=click.DateTime(formats=["%Y-%m-%d"]),
        help="End date (YYYY-MM-DD)",
    )(f)
    f = click.option(
        "--from",
        "date_from",
        default=None,
        type=click.DateTime(formats=["%Y-%m-%d"]),
        help="Start date (YYYY-MM-DD)",
    )(f)
    return f


def date_range_options_required(f: Any) -> Any:
    """Shared --from/--to date range options (required, not optional)."""
    f = click.option(
        "--to",
        "date_to",
        required=True,
        type=click.DateTime(formats=["%Y-%m-%d"]),
        help="End date (YYYY-MM-DD)",
    )(f)
    f = click.option(
        "--from",
        "date_from",
        required=True,
        type=click.DateTime(formats=["%Y-%m-%d"]),
        help="Start date (YYYY-MM-DD)",
    )(f)
    return f


def parse_id_list(raw: str) -> list[int]:
    """Parse a comma-separated string of IDs into a list of ints.

    Raises click.BadParameter on invalid input.
    """
    try:
        return [int(sid.strip()) for sid in raw.split(",")]
    except ValueError as e:
        raise click.BadParameter(
            f"Invalid ID list: {raw!r}. Expected comma-separated integers."
        ) from e


def actor_options(f: Any) -> Any:
    """Shared --user/--profile options for profile-scoped data commands.

    Adds ``--user`` (``SNORE_USER`` env var) and ``--profile``
    (``SNORE_PROFILE`` env var) to the decorated command.  The resolved
    parameter names are ``actor_user`` and ``actor_profile`` — distinct from
    the ``--user`` option used by ``snore user`` / ``snore profile`` operator
    commands.

    Most commands should prefer ``profile_scoped_command``, which bundles these
    options with the async runtime preamble.  Use ``actor_options`` directly only
    when a command needs custom control over session/profile resolution.
    """
    f = click.option(
        "--profile",
        "actor_profile",
        default=None,
        envvar="SNORE_PROFILE",
        help="Profile name or ID (default: user's default; env: SNORE_PROFILE)",
    )(f)
    f = click.option(
        "--user",
        "actor_user",
        default=None,
        envvar="SNORE_USER",
        help="User email (default: local admin user; env: SNORE_USER)",
    )(f)
    return f


def profile_scoped_command(f: Any) -> Any:
    """Bundle ``--db`` + ``--user``/``--profile`` options with the async runtime preamble.

    The decorated body must be ``async def body(ctx: CliCtx, **command_kwargs)``.
    At call time this opens a DB session, resolves the profile id, builds a
    ``CliCtx``, runs the body under ``asyncio.run``, and returns its value.
    ``click.ClickException`` raised inside the body propagates unchanged.

    Provenance markers emitted by the body are collected and their legend is
    printed once after it returns.

    An explicit ``--db`` must already exist: since ``init_database`` silently
    creates a missing SQLite file, the path is checked before the session opens.
    """
    import asyncio  # noqa: PLC0415
    import functools  # noqa: PLC0415

    from snore.cli.display import provenance_legend  # noqa: PLC0415

    @functools.wraps(f)
    def wrapper(
        *args: Any,
        db: str | None,
        actor_user: str | None,
        actor_profile: str | None,
        **kwargs: Any,
    ) -> Any:
        if db and not Path(db).expanduser().exists():
            raise click.ClickException(f"Database not found: {db}")

        async def _run() -> Any:
            async with profile_session(db, actor_user, actor_profile) as ctx:
                return await f(ctx, *args, **kwargs)

        with provenance_legend():
            return asyncio.run(_run())

    return db_option(actor_options(wrapper))
