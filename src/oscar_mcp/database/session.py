"""Database session management for OSCAR-MCP."""

import os
import threading

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from oscar_mcp.constants import DEFAULT_DATABASE_PATH
from oscar_mcp.database.models import Base

# Global engine and session factory
_engine = None
_SessionFactory = None
_init_lock = threading.Lock()


def init_database(database_path: str | None = None) -> None:
    """
    Initialize the database connection in a thread-safe manner.

    Args:
        database_path: Path to the SQLite database file.
                      Defaults to DEFAULT_DATABASE_PATH.

    Raises:
        PermissionError: If directory cannot be created
        ValueError: If database path is invalid
    """
    global _engine, _SessionFactory

    with _init_lock:
        if _engine is not None and _SessionFactory is not None:
            return

        if database_path is None:
            database_path = DEFAULT_DATABASE_PATH

        if not database_path or not isinstance(database_path, str):
            raise ValueError(f"Invalid database path: {database_path}")

        db_dir = os.path.dirname(database_path)
        if db_dir:
            try:
                os.makedirs(db_dir, exist_ok=True)
            except PermissionError as e:
                raise PermissionError(
                    f"Cannot create database directory {db_dir}: {e}"
                ) from e

        database_url = f"sqlite:///{database_path}"

        _engine = create_engine(
            database_url,
            echo=False,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )

        @event.listens_for(_engine, "connect")
        def set_sqlite_pragma(dbapi_conn: Any, connection_record: Any) -> None:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        _SessionFactory = sessionmaker(bind=_engine)

        Base.metadata.create_all(_engine)


def get_session() -> Session:
    """
    Get a new database session.

    Returns:
        A new SQLAlchemy session.

    Raises:
        RuntimeError: If database has not been initialized.
    """
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    return _SessionFactory()


@contextmanager
def session_scope() -> Generator[Session]:
    """
    Provide a transactional scope for database operations.

    Usage:
        with session_scope() as session:
            session.add(obj)
            # Automatically commits on success, rolls back on error

    Yields:
        A database session.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_engine() -> Engine:
    """Get the database engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def cleanup_database() -> None:
    """
    Clean up database connections and reset global state.

    This function should be called during test cleanup to prevent resource warnings.
    It properly disposes of the SQLAlchemy engine and resets global variables.
    """
    global _engine, _SessionFactory

    with _init_lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None
        _SessionFactory = None
