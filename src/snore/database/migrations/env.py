import os
import sys

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from snore.database.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
# Naming convention is baked into Base.metadata via NAMING_CONVENTION in models.py.


def _resolve_url() -> str:
    """Return the configured URL via ``DatabaseTarget``, falling back to DEFAULT_DATABASE_PATH.

    Routes through the same ``DatabaseTarget`` resolver used by the app and CLI
    so Alembic and the runtime always see identical URL derivation logic.

    Uses ``resolve_migration_url()`` (not ``resolve_sync_url()``) so that the
    driver mapping is correct for Alembic's needs — today both are identical for
    SQLite, but they will diverge at the hosted milestone (postgresql → psycopg
    for migrations vs. asyncpg for runtime).
    """
    url = config.get_main_option("sqlalchemy.url") or ""
    if url not in ("", "sqlite:///"):
        # An explicit URL was provided in alembic.ini or via env override.
        # Let DatabaseTarget parse and normalise it.
        from snore.database.target import DatabaseTarget  # noqa: PLC0415

        target = DatabaseTarget.from_url(url)
        return target.resolve_migration_url()

    # No explicit URL — use the environment/default resolution chain.
    from snore.database.target import DatabaseTarget  # noqa: PLC0415

    target = DatabaseTarget.from_env_and_flags(db_flag=None, warn_ignored=False)
    return target.resolve_migration_url()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = _resolve_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Escape "%" for configparser interpolation (URL-escaped characters).
    config.set_main_option("sqlalchemy.url", _resolve_url().replace("%", "%%"))
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # render_as_batch is required for SQLite ALTER TABLE support; it is
            # harmless (but unnecessary) on PostgreSQL.  We gate it on the dialect
            # so generated migration SQL stays portable and readable on Postgres.
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
