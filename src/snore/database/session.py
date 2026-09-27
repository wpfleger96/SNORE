"""Database session management for SNORE.

Async SQLite transaction recipe (§4, PR-2)
-------------------------------------------
The async engine uses ``aiosqlite`` as the DBAPI.

**PRAGMA setup (``"connect"`` event):**
The ``"connect"`` event fires on every new DBAPI connection.  aiosqlite wraps
the raw ``sqlite3.Connection``; we access it via
``dbapi_conn.driver_connection._conn``.  We set ``isolation_level = None``
(autocommit) *permanently* on the raw connection so PRAGMAs like
``journal_mode=WAL`` can run outside a transaction.  The connection stays in
autocommit mode — SQLAlchemy's logical transaction control takes over via the
``"begin"`` event (see below).

**Transaction control (``"begin"`` event):**
With aiosqlite in autocommit mode, SQLAlchemy's logical ``BEGIN`` never emits
an actual ``BEGIN`` statement to SQLite.  That means a released savepoint
(``RELEASE SAVEPOINT sp``) escapes the outer rollback — the row is committed to
the file before the outer ``ROLLBACK`` fires.

To fix this, we attach a ``"begin"`` event listener to
``async_engine.sync_engine`` that executes an explicit ``BEGIN`` statement
whenever SQLAlchemy starts a new logical transaction.  This restores the
expected two-layer semantics: SQLite's outer ``BEGIN`` contains all
``SAVEPOINT`` / ``RELEASE`` / ``ROLLBACK TO`` operations so that an outer
``ROLLBACK`` always undoes any released savepoints.

VACUUM uses a separate sync AUTOCOMMIT connection and is unaffected.

**Migrations:**
Alembic stays fully synchronous and uses the pysqlite URL.  The async
``init_database`` / ``init_database_from_url`` entry points run
``_apply_migrations_sync`` via ``asyncio.to_thread`` so they never block the
event loop.

expire_on_commit=False is set on all sessions so that ORM attributes remain
accessible after a commit without triggering implicit I/O — required for
async contexts where lazy loads raise MissingGreenlet.
"""

import asyncio
import logging
import os

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import Engine, MetaData, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.schema import CreateColumn

from snore.constants import DEFAULT_DATABASE_PATH
from snore.database.models import Base

logger = logging.getLogger(__name__)

_engine: AsyncEngine | None = None
_AsyncSessionFactory: async_sessionmaker[AsyncSession] | None = None
_db_path: str | None = None

# Staleness-detection globals — set/cleared alongside the engine globals above.
#
# _db_identity: (st_dev, st_ino) of the DB file recorded at engine-init time.
#   A changed identity means the file was atomically replaced on disk (e.g. by
#   SNORE's import pipeline, which does an atomic rename(2) to a new inode).
#   None disables detection (in-memory DB, OSError during stat, or not inited).
#
# _sync_url / _async_url: cached at init so check_db_staleness can reinitialize
#   the engine without re-resolving URLs from the path alone.
#
# _engine_generation: monotonically increments on every successful _do_init call.
#   NEVER reset by cleanup_database or initialization failure.  External callers
#   snapshot this value to detect that the engine was rebuilt (e.g. after a
#   swap-triggered reinit in check_db_staleness).
#
# _pending_reinit: (sync_url, async_url, db_path) of a swap-triggered reinit
#   that has not yet completed.  Written in check_db_staleness before teardown
#   starts; cleared inside _do_init's success publish block.  NOT cleared by
#   _do_cleanup — it must survive cleanup so the reinit can be retried.  When
#   _pending_reinit is not None and _db_path is None, check_db_staleness retries
#   the reinit instead of silently returning with a broken/None engine.
#   Because _pending_reinit survives cleanup_database, a request slipping in
#   after lifespan teardown (with a failed swap-reinit pending) would
#   re-initialize the swap target instead of raising "not initialized" —
#   acceptable in the current server lifecycle where connection acceptance stops
#   before teardown.
_db_identity: tuple[int, int] | None = None
_sync_url: str | None = None
_async_url: str | None = None
_engine_generation: int = 0
_pending_reinit: tuple[str, str, str] | None = None

# Shared state machine: three linearized states under one stable lock.
#
#   initialized:       _engine is not None
#   init-in-flight:    _init_task is not None and not done
#   cleanup-in-flight: _cleanup_task is not None and not done
#
# Protected by _init_lock (created lazily, never replaced).
# Public initialization/cleanup entry points read/mutate these globals
# inside ONE lock acquisition — no check-then-reacquire gaps.
# Read-only accessors (get_session, get_engine, get_db_path) remain
# lock-free; they check single published values, not state transitions.
_init_lock: asyncio.Lock | None = None  # Created lazily inside an event loop.
_init_task: asyncio.Task[None] | None = None  # The in-flight once-init-task.
_cleanup_task: asyncio.Task[None] | None = None  # The in-flight once-cleanup-task.

# Execution-option key/value for BEGIN IMMEDIATE transactions.
# Used by session_scope(immediate=True) at connection checkout and by the
# "begin" event listener that reads them to choose the BEGIN variant.
# Module-level constants prevent a typo from silently downgrading
# BEGIN IMMEDIATE to plain BEGIN at either site.
_TXN_OPT_KEY = "sqlite_txn_mode"
_TXN_OPT_IMMEDIATE = "IMMEDIATE"

# Public mapping for callers that need to escalate a session to IMMEDIATE mode
# by calling ``await session.connection(execution_options=TXN_OPT_IMMEDIATE)``
# at the start of a handler body before any SQL is issued.  The ``begin``
# event listener reads this option and issues ``BEGIN IMMEDIATE`` instead of
# a plain ``BEGIN``, preventing SQLITE_BUSY on the first write in a deferred
# transaction when another writer has committed since the transaction opened.
TXN_OPT_IMMEDIATE: dict[str, str] = {_TXN_OPT_KEY: _TXN_OPT_IMMEDIATE}


def _get_init_lock() -> asyncio.Lock:
    """Return the module-level asyncio.Lock, creating it on first call.

    Must be called from within a running event loop.  The lock is never
    replaced once created so all callers always share the same domain.
    """
    global _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    return _init_lock


def _build_alembic_config(database_url: str) -> AlembicConfig:
    migrations_dir = str(Path(__file__).parent / "migrations")
    cfg = AlembicConfig()
    cfg.set_main_option("script_location", migrations_dir)
    # Alembic's configparser treats "%" as interpolation; URL-escaped
    # characters (e.g. SQLAlchemy 2.1 renders ":memory:" as "%3Amemory%3A")
    # must be doubled.
    cfg.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return cfg


def _sync_additive_schema(metadata: MetaData, engine: Engine) -> None:
    """Additively sync columns and indexes from *metadata* onto an existing database.

    Called in empty-chain mode immediately after ``create_all``, which is a
    no-op for tables that already exist.  The work here is for tables that
    pre-dated this startup — i.e., rows already in the file when the process
    started:

    - For each table in *metadata* that exists in the database, compute the
      model columns absent from the live table and emit
      ``ALTER TABLE … ADD COLUMN`` for each one.  Column DDL is compiled from
      the SQLAlchemy column object via ``CreateColumn`` so type strings and
      defaults are always correct for the engine's dialect.
    - Guard: a missing column that is ``NOT NULL`` with no ``server_default``
      raises ``RuntimeError`` with an actionable message.  SQLite cannot
      ``ADD COLUMN NOT NULL`` without a default — every new non-nullable column
      must carry a ``server_default`` in its model definition.  A non-constant
      ``server_default`` (e.g. ``text("CURRENT_TIMESTAMP")``) passes this guard
      but SQLite rejects ``ADD COLUMN`` with a non-constant default on populated
      tables ("Cannot add a column with non-constant default") — new columns on
      existing tables need constant server defaults.
    - For each such table, compare model indexes against live indexes and call
      ``Index.create(engine)`` for each missing one.

    Idempotent: a second call on an already-synced database is a no-op.
    Also a no-op on a fresh database (``create_all`` has built everything).

    Out of scope: dropped columns, renamed columns, changed types, dropped
    indexes — those require Alembic migration files once post-1.0 is in use.
    Two additional silent gaps: (a) foreign-key clauses on new columns —
    SQLite ``ADD COLUMN`` cannot carry ``REFERENCES``, so a new FK column is
    added without its constraint; (b) new ``UniqueConstraint``s and
    ``unique=True`` backing indexes — ``table.indexes`` excludes them, so
    they are never created on existing tables by this helper.
    """
    # The inspector caches per-table reflection results; each get_columns /
    # get_indexes call is issued exactly once per table per run.
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    preparer = engine.dialect.identifier_preparer

    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            continue

        live_col_names = {col["name"] for col in inspector.get_columns(table.name)}
        for col in table.columns:
            if col.name in live_col_names:
                continue
            if not col.nullable and col.server_default is None:
                raise RuntimeError(
                    f"Cannot add column '{col.name}' to table '{table.name}': "
                    f"the column is NOT NULL with no server_default.  "
                    f"SQLite cannot ADD COLUMN NOT NULL without a default — "
                    f"add a server_default= to the column definition."
                )
            col_ddl = str(CreateColumn(col).compile(dialect=engine.dialect))
            table_id = preparer.quote_identifier(table.name)
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE {table_id} ADD COLUMN {col_ddl}"))
            except OperationalError as exc:
                # Guard: another process added this column concurrently (rolling
                # restart).  SQLite (and MySQL, lowercased) emit "duplicate column
                # name"; the repo is SQLite-only.
                if "duplicate column name" in str(exc).lower():
                    logger.info(
                        "Column %s.%s added concurrently; skipping",
                        table.name,
                        col.name,
                    )
                else:
                    raise
            else:
                logger.info(
                    "Added column %s.%s to existing table", table.name, col.name
                )

        live_index_names = {i["name"] for i in inspector.get_indexes(table.name)}
        for idx in table.indexes:
            if idx.name in live_index_names:
                continue
            try:
                idx.create(engine, checkfirst=True)
            except OperationalError as exc:
                # Guard: checkfirst=True emits a PRAGMA check then plain CREATE
                # INDEX (not IF NOT EXISTS); two racing processes can both pass
                # the check.  The loser gets "already exists".
                if "already exists" in str(exc).lower():
                    logger.info("Index %s created concurrently; skipping", idx.name)
                else:
                    raise
            else:
                logger.info(
                    "Created index %s on existing table %s", idx.name, table.name
                )


def _compute_migration_checksums(script_dir: Any) -> dict[str, str]:
    """Compute SHA-256 checksums over raw file bytes for every revision script.

    Returns a ``{revision_id: hexdigest}`` mapping for all revisions with a
    non-falsy ``Script.path``.  Revisions without a path (e.g. synthetic base
    stubs) are skipped silently.

    Cost: O(N) file reads per startup — every migration file is hashed on every
    startup call.  An unreadable migration file (``PermissionError``,
    ``FileNotFoundError``, etc.) propagates immediately and aborts startup; this
    is intentional fail-fast behaviour so a corrupt migration directory cannot
    silently skip the replay check.
    """
    import hashlib  # noqa: PLC0415

    checksums: dict[str, str] = {}
    for script in script_dir.walk_revisions():
        if not script.path:
            continue
        raw = Path(script.path).read_bytes()
        checksums[script.revision] = hashlib.sha256(raw).hexdigest()
    return checksums


def _read_stored_checksums(conn: Any) -> dict[str, str] | None:
    """Read checksums from the ``snore_migration_checksums`` table.

    Returns a ``{revision: checksum}`` mapping, ``{}`` if the table exists but
    is empty, or ``None`` if the table is absent (pre-feature DB — caller should
    treat this as a backfill case rather than a mismatch).
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    try:
        rows = conn.execute(
            "SELECT revision, checksum FROM snore_migration_checksums"
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    except _sqlite3.OperationalError as exc:
        # Table missing → pre-feature DB; signal backfill with None.
        logger.debug("Checksum table read failed (%s); treating as pre-feature DB", exc)
        return None


def _ensure_checksum_table(engine: Engine) -> None:
    """Create the ``snore_migration_checksums`` table if it does not yet exist.

    Uses plain DDL rather than an Alembic migration for the same reason
    ``alembic_version`` itself is not a migration: the table must exist before
    checksums can be checked, so it cannot be managed by the very system it
    bootstraps.
    """
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS snore_migration_checksums "
                "(revision TEXT PRIMARY KEY, checksum TEXT NOT NULL)"
            )
        )


def _write_checksums(engine: Engine, checksums: dict[str, str]) -> None:
    """Persist *checksums* to the database, replacing any prior rows.

    Runs DELETE + INSERT in one transaction so orphaned rows from renamed or
    deleted revision IDs are always cleaned up.  Parameterised binds are used
    throughout — never string interpolation.
    """
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM snore_migration_checksums"))
        if checksums:
            conn.execute(
                text(
                    "INSERT INTO snore_migration_checksums (revision, checksum)"
                    " VALUES (:revision, :checksum)"
                ),
                [{"revision": r, "checksum": c} for r, c in checksums.items()],
            )


def _first_parent_revision(down: Any) -> str | None:
    """Return the first parent revision ID from a ``down_revision`` value.

    Handles ``None`` (no parent), tuples/lists (merge points — only the first
    parent is followed), and plain string values.
    """
    if down is None:
        return None
    if isinstance(down, (tuple, list)):
        return str(down[0])
    return str(down)


def _find_earliest_edited_revision(
    script_dir: Any,
    stamp: str,
    stored: dict[str, str],
    current: dict[str, str],
) -> Any | None:
    """Walk the applied migration chain and return the earliest edited revision.

    Traverses from *stamp* back to the chain base following ``down_revision``
    links.  A revision is considered *edited* only when ``stored`` has a
    checksum for it AND that checksum differs from ``current``; revisions
    absent from ``stored`` are new migrations, not edits.

    Only the first parent of a merge point is walked — edits on a
    non-first-parent branch are not detected (the chain is linear today).

    Returns the ``Script`` object closest to the base with a mismatch, or
    ``None`` when the chain is clean or *stamp* cannot be resolved.
    """
    earliest: Any | None = None
    try:
        rev = script_dir.get_revision(stamp)
        visited: set[str] = set()
        while rev is not None:
            rev_id = rev.revision
            # Guard: a malformed or cyclic down_revision chain must degrade to
            # "no replay", never hang startup.
            if rev_id in visited:
                break
            visited.add(rev_id)
            if rev_id in stored and stored[rev_id] != current.get(rev_id):
                earliest = rev
            next_rev_id = _first_parent_revision(rev.down_revision)
            if next_rev_id is None:
                break
            try:
                rev = script_dir.get_revision(next_rev_id)
            except Exception:
                break
    except Exception:
        # stamp not resolvable (stale stamp, missing file, ResolutionError) →
        # Alembic's own upgrade handles it; no replay needed.
        return None
    return earliest


def _replay_migration_edits(
    cfg: AlembicConfig,
    earliest_edited: Any,
    engine: Engine,
    current_checksums: dict[str, str],
) -> None:
    """Downgrade to before *earliest_edited* then upgrade to head.

    If ``earliest_edited.down_revision`` is ``None`` (the baseline was edited),
    downgrade target is ``"base"``, which drops all Alembic-managed tables.
    This path is gated behind the ``SNORE_ALLOW_BASELINE_REPLAY=1`` environment
    variable; without it a ``RuntimeError`` is raised to prevent accidental data
    loss.  Non-base targets are always applied automatically.

    Writes updated checksums after a successful replay.
    """
    target = _first_parent_revision(earliest_edited.down_revision) or "base"
    rev_id = earliest_edited.revision

    if target == "base":
        logger.warning(
            "Migration checksum mismatch: revision %s was edited and has no"
            " predecessor — downgrading to base DROPS ALL Alembic-managed tables"
            " (data loss).",
            rev_id,
        )
        if os.environ.get("SNORE_ALLOW_BASELINE_REPLAY") != "1":
            raise RuntimeError(
                f"Migration checksum mismatch on baseline revision {rev_id}: "
                "replaying it would downgrade to 'base' and DROP ALL "
                "Alembic-managed tables. Set SNORE_ALLOW_BASELINE_REPLAY=1 to "
                "allow, or restore the original file contents."
            )
    else:
        logger.warning(
            "Migration checksum mismatch: revision %s was edited."
            " Downgrading to %s and re-applying to head.",
            rev_id,
            target,
        )

    alembic_command.downgrade(cfg, target)
    alembic_command.upgrade(cfg, "head")
    _write_checksums(engine, current_checksums)
    logger.info("Migration checksum replay complete for revision %s.", rev_id)


def _apply_migrations_sync(sync_url: str) -> None:
    """Apply pending migrations, or skip when the schema is already at head.

    Called from within ``asyncio.to_thread`` so it never blocks the event loop.
    Uses the sync pysqlite URL for Alembic.

    **Empty-chain mode (pre-1.0):** When ``versions/`` contains zero migration
    files, ``ScriptDirectory.get_heads()`` returns ``[]``.  In this mode the
    function skips all Alembic machinery and ensures the schema in two steps:

    1. ``Base.metadata.create_all(checkfirst=True)`` — creates any tables that
       do not yet exist.  Idempotent; never alters existing tables.
    2. ``_sync_additive_schema(Base.metadata, engine)`` — for each table that
       already existed before step 1, adds any model columns or indexes absent
       from the live table.  New ``NOT NULL`` columns must carry a
       ``server_default`` in the model definition; without one a ``RuntimeError``
       is raised with an actionable message (SQLite cannot ``ADD COLUMN NOT NULL``
       without a default value).

    No ``alembic_version`` table is created on this path.  Stale
    ``alembic_version`` rows left by a pre-flatten DB are silently ignored
    (the owner drops incompatible DBs manually).

    **Fast-path skip:** The database stamp is read first (before computing heads)
    via a read-only connection.  When the stamp matches the current head, all
    Alembic machinery is skipped.  Reading the stamp first means a new migration
    file landing between the two reads can only force the slow path, never
    accidentally skip a brand-new migration.  Any failure of the stamp read
    (missing file, missing table, OperationalError) falls through safely.

    **Unstamped-existing-DB path:** When the database has application tables but
    no ``alembic_version`` table (created in zero-migration mode before this
    migration chain existed), the DB is stamped at the no-op ``001_baseline``
    revision and upgraded to head, then ``_sync_additive_schema`` runs to add
    any model columns/indexes the pre-chain database is missing — migrations
    only cover schema changes made after the chain was introduced, so the
    additive sync is what restores earlier drift (e.g. a pre-#200 database
    missing ``users.google_link_disabled``).

    **Checksum tracking:** Every migration file's SHA-256 checksum is stored in
    ``snore_migration_checksums`` after it is applied.  On each startup, computed
    checksums are compared against the stored ones.  When an already-applied
    migration file has been edited (checksum mismatch), the function downgrades
    to just before the earliest edited revision and re-applies to head, then
    rewrites all checksums.  The fast-path skip only fires when the stamp matches
    head AND the stored checksum table is non-empty AND all checksums match; an
    empty table (e.g. from a crash between table creation and the first write)
    falls through for backfill.  Editing ``001_baseline`` (the no-op stamp
    revision) would force a downgrade to ``"base"`` (drops all Alembic-managed
    tables); this is gated behind ``SNORE_ALLOW_BASELINE_REPLAY=1`` to prevent
    accidental data loss.
    """
    import sqlite3 as _sqlite3  # noqa: PLC0415

    from urllib.parse import quote as _quote  # noqa: PLC0415

    from alembic.script import ScriptDirectory  # noqa: PLC0415
    from sqlalchemy import create_engine  # noqa: PLC0415
    from sqlalchemy.engine import make_url as _make_url  # noqa: PLC0415

    url_obj = _make_url(sync_url)
    # Build the Alembic config once; reused by both the fast-path head check
    # and the slow-path upgrade/stamp call below.
    alembic_cfg = _build_alembic_config(sync_url)

    # Step 1: Read the stamped version first, before computing heads.
    # The read-only open avoids acquiring any write lock.
    stored_version: str | None = None
    stored_checksums: dict[str, str] | None = None
    if url_obj.get_backend_name() == "sqlite":
        db_file = url_obj.database
        if db_file and db_file != ":memory:":
            try:
                # Percent-encode the path so spaces and special characters in
                # directory names do not corrupt the URI query string.
                encoded_path = _quote(db_file, safe="/")
                ro_conn = _sqlite3.connect(f"file:{encoded_path}?mode=ro", uri=True)
                try:
                    row = ro_conn.execute(
                        "SELECT version_num FROM alembic_version"
                    ).fetchone()
                    stored_version = row[0] if row else None
                    stored_checksums = _read_stored_checksums(ro_conn)
                finally:
                    ro_conn.close()
            except Exception as exc:
                # Any failure (missing file, missing alembic_version table,
                # OperationalError) → fall through to the normal migration path
                # so startup is always correct.
                logger.debug(
                    "Migration fast-path check failed; falling through to full"
                    " migration: %s",
                    exc,
                )

    # Step 2: Compute heads once, after reading the stamp.
    # Reading the stamp before computing heads means a migration file landing
    # between the two reads can only make the comparison fail (falling through
    # to upgrade), never fast-path-skip a brand-new migration.
    script_dir = ScriptDirectory.from_config(alembic_cfg)
    heads = set(script_dir.get_heads())
    current_checksums = _compute_migration_checksums(script_dir)

    # Hoist the edited-revision scan so the result can be reused by both the
    # fast-path guard and the slow-path else branch without a second walk.
    earliest_edited = (
        _find_earliest_edited_revision(
            script_dir, stored_version, stored_checksums, current_checksums
        )
        if stored_version is not None and stored_checksums
        else None
    )

    # Fast-path: skip migrations when the database is already at the current
    # Alembic head AND the checksum table is non-empty AND all checksums match.
    # An empty stored table (stored_checksums == {}) means the table exists but
    # has no rows — e.g. crash between CREATE TABLE and first INSERT — so we
    # fall through to the slow path for a backfill rather than skipping forever.
    if (
        stored_version is not None
        and {stored_version} == heads
        and stored_checksums
        and earliest_edited is None
    ):
        logger.debug(
            "Schema at head %s with matching checksums; skipping migrations",
            stored_version,
        )
        return
    elif stored_version is not None and {stored_version} == heads:
        if not stored_checksums:
            logger.debug(
                "Schema at head %s but checksum table absent or empty;"
                " will backfill checksums",
                stored_version,
            )
        else:
            logger.debug(
                "Schema at head %s but checksum mismatch detected; will replay edits",
                stored_version,
            )
        # fall through to the slow path for backfill or replay

    # Empty-chain mode: no migration files → manage schema via create_all then
    # additive sync.  No alembic_version table is written; stale rows from
    # pre-flatten DBs are ignored — the owner drops incompatible DBs manually.
    # New NOT NULL columns must carry a server_default or _sync_additive_schema
    # raises RuntimeError; SQLite cannot ADD COLUMN NOT NULL without a default.
    if not heads:
        engine = create_engine(sync_url, connect_args={"check_same_thread": False})
        try:
            Base.metadata.create_all(engine, checkfirst=True)
            _sync_additive_schema(Base.metadata, engine)
        finally:
            engine.dispose()
        logger.info(
            "Empty migration chain: schema ensured via create_all + additive sync"
        )
        return

    # Slow path: apply migrations. Wrap the entire block in try/finally so
    # the engine is always disposed even if alembic_command.upgrade raises.
    engine = create_engine(sync_url, connect_args={"check_same_thread": False})
    try:
        _ensure_checksum_table(engine)
        table_names = set(inspect(engine).get_table_names())

        if "sessions" not in table_names:
            Base.metadata.create_all(engine)
            alembic_command.stamp(alembic_cfg, "head")
            _write_checksums(engine, current_checksums)
            logger.info("Fresh database created and stamped at head")
        elif "alembic_version" not in table_names:
            # Existing DB from before the migration chain existed (created in
            # zero-migration mode via create_all with no Alembic stamp).  Stamp
            # at the no-op baseline so the subsequent upgrade applies only the
            # schema changes that are genuinely missing, rather than replaying
            # the full chain onto already-existing tables.  Migrations only
            # cover post-chain schema changes, so additive sync must run once
            # here to restore pre-chain drift (missing columns/indexes).
            alembic_command.stamp(alembic_cfg, "001_baseline")
            alembic_command.upgrade(alembic_cfg, "head")
            _sync_additive_schema(Base.metadata, engine)
            _write_checksums(engine, current_checksums)
            logger.info(
                "Stamped existing DB at 001_baseline, upgraded to head, and"
                " synced pre-chain schema drift"
            )
        else:
            # When the fast-path RO read failed or returned empty, re-read
            # checksums through the engine — the table is guaranteed to exist
            # now that _ensure_checksum_table has run.  This prevents a
            # transient read failure from masking an edit that needs replay.
            if not stored_checksums:
                raw = engine.raw_connection()
                try:
                    refreshed = _read_stored_checksums(raw.driver_connection)
                finally:
                    raw.close()
                stored_checksums = refreshed or {}
                earliest_edited = (
                    _find_earliest_edited_revision(
                        script_dir, stored_version, stored_checksums, current_checksums
                    )
                    if stored_version is not None and stored_checksums
                    else None
                )
            if earliest_edited is not None:
                _replay_migration_edits(
                    alembic_cfg, earliest_edited, engine, current_checksums
                )
            else:
                alembic_command.upgrade(alembic_cfg, "head")
                _write_checksums(engine, current_checksums)
                logger.info("Database migrations applied (upgrade to head)")
    finally:
        engine.dispose()


def _register_sqlite_pragmas_on_async_engine(async_engine: AsyncEngine) -> None:
    """Attach the integrated SQLite connection recipe to *async_engine*.

    Two event listeners are registered on ``async_engine.sync_engine``:

    **"connect" — PRAGMA setup:**
    Fires on every new DBAPI connection.  For aiosqlite connections,
    ``dbapi_conn.driver_connection._conn`` is the raw ``sqlite3.Connection``.
    We set ``isolation_level = None`` (autocommit) *permanently* so that
    PRAGMAs like ``journal_mode=WAL`` run outside a transaction and so that
    SQLAlchemy's logical transaction control is the sole source of truth.
    The connection is left in autocommit mode after the event returns.

    **"begin" — explicit BEGIN or BEGIN IMMEDIATE:**
    With the DBAPI in autocommit mode, SQLAlchemy's logical transaction does
    not emit a real ``BEGIN`` to SQLite.  Without an explicit ``BEGIN``, a
    released savepoint (``RELEASE SAVEPOINT sp``) writes directly to the
    database file and cannot be undone by a later outer rollback.

    The ``"begin"`` listener executes ``BEGIN`` (or ``BEGIN IMMEDIATE`` when
    the ``"sqlite_txn_mode"`` execution option is ``"IMMEDIATE"``) via a raw
    cursor whenever SQLAlchemy opens a new logical transaction.  This
    re-establishes the two-layer semantics: SQLite's outer ``BEGIN`` contains
    all ``SAVEPOINT`` / ``RELEASE`` / ``ROLLBACK TO`` operations, so an outer
    rollback undoes every released savepoint within it.

    ``BEGIN IMMEDIATE`` acquires the write lock at transaction open rather
    than at first write, so contending writers queue on ``busy_timeout``
    instead of failing instantly on a WAL snapshot-upgrade conflict.  Pass
    ``execution_options={"sqlite_txn_mode": "IMMEDIATE"}`` to the connection
    before SQL is issued (via ``session_scope(immediate=True)``) to activate
    this mode.
    """
    from sqlalchemy import event  # noqa: PLC0415

    @event.listens_for(async_engine.sync_engine, "connect")
    def set_sqlite_pragmas(dbapi_conn: Any, connection_record: Any) -> None:
        # aiosqlite wraps the raw connection; unwrap to access isolation_level.
        raw_conn = getattr(
            getattr(dbapi_conn, "driver_connection", None), "_conn", None
        )
        if raw_conn is None:
            # Fallback: try direct raw connection (sync pysqlite path).
            raw_conn = dbapi_conn

        # Set autocommit permanently — the "begin" listener emits explicit BEGIN.
        raw_conn.isolation_level = None

        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=15000")
            cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
            cursor.execute("PRAGMA temp_store=MEMORY")
            # Cap WAL file growth so checkpoints shrink an oversized WAL back
            # down to this limit after each checkpoint completes.
            cursor.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
        finally:
            cursor.close()

    @event.listens_for(async_engine.sync_engine, "begin")
    def emit_begin(conn: Any) -> None:
        # With DBAPI in autocommit mode, emit an explicit BEGIN so that SQLite's
        # outer transaction wraps all SAVEPOINTs and outer ROLLBACK works correctly.
        # When session_scope(immediate=True) is used, emit BEGIN IMMEDIATE instead:
        # this acquires the write lock at transaction open so contending writers
        # queue on busy_timeout rather than failing instantly on a WAL snapshot
        # upgrade (SQLite returns SQLITE_BUSY immediately on a deferred→write
        # upgrade when another connection has committed, bypassing busy_timeout).
        txn_mode = conn.get_execution_options().get(_TXN_OPT_KEY)
        if txn_mode == _TXN_OPT_IMMEDIATE:
            conn.exec_driver_sql("BEGIN IMMEDIATE")
        else:
            conn.exec_driver_sql("BEGIN")


async def _do_init(sync_url: str, async_url: str, db_path: str | None) -> None:
    """Build the engine, run migrations, then publish globals atomically.

    Called exactly once per initialization lifecycle.  On failure or
    cancellation, disposes any partially-built engine and clears all globals
    (including ``_init_task``) so a retry starts fresh.

    ``CancelledError`` is caught as ``BaseException`` so a cancelled driver
    tears down cleanly and lets the next caller retry rather than hanging.
    """
    global \
        _engine, \
        _AsyncSessionFactory, \
        _db_path, \
        _db_identity, \
        _sync_url, \
        _async_url, \
        _engine_generation, \
        _pending_reinit

    engine: AsyncEngine | None = None
    try:
        if db_path and db_path != ":memory:":
            db_dir = os.path.dirname(db_path)
            if db_dir:
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except PermissionError as e:
                    raise PermissionError(
                        f"Cannot create database directory {db_dir}: {e}"
                    ) from e

        engine = create_async_engine(
            async_url,
            echo=False,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )

        from sqlalchemy.engine import make_url as _make_url  # noqa: PLC0415

        dialect = _make_url(async_url).get_backend_name()
        if dialect == "sqlite":
            _register_sqlite_pragmas_on_async_engine(engine)

        # Run migrations off the event loop — never blocks uvicorn.
        await asyncio.to_thread(_apply_migrations_sync, sync_url)

        # Publish only after successful migration.
        _db_path = db_path
        _engine = engine
        _AsyncSessionFactory = async_sessionmaker(
            bind=engine,
            expire_on_commit=False,
            class_=AsyncSession,
        )

        # Record the file identity (dev, inode) for atomic-swap detection.
        # Statting after migrations is safe — the pool is lazy, so if the file
        # was replaced mid-init, both this stat and the first connection observe
        # the new file consistently.
        if db_path and db_path != ":memory:":
            try:
                st = os.stat(db_path)
                _db_identity = (st.st_dev, st.st_ino)
            except OSError:
                # File absent between directory creation and stat (e.g. racing
                # test teardown).  Swap detection is disabled until next init.
                _db_identity = None
        else:
            _db_identity = None
        _sync_url = sync_url
        _async_url = async_url
        _pending_reinit = None  # Reinit completed successfully.
        _engine_generation += 1
    except BaseException:
        # Atomic teardown on any failure including CancelledError.
        # Use synchronous disposal only — awaiting inside an except-BaseException
        # block after catching CancelledError can re-raise CancelledError at the
        # next await in Python 3.11+ cancellation semantics.
        # _init_task does NOT need explicit clearing here: _init_with_once_task
        # checks ``_init_task.done()`` under the lock, which returns True once
        # this task is cancelled or failed — so retries always create a fresh task.
        if engine is not None:
            engine.sync_engine.dispose()
        _engine = None
        _AsyncSessionFactory = None
        _db_path = None
        _db_identity = None
        _sync_url = None
        _async_url = None
        # _engine_generation deliberately survives — see module-level comment.
        raise


async def _init_with_once_task(
    sync_url: str, async_url: str, db_path: str | None
) -> None:
    """Coordinate initialization via a shared once-task.

    Single-acquisition state machine: one lock acquisition decides everything
    — no check-then-reacquire gap exists.

    State transitions (under _init_lock):
    - cleanup-in-flight → capture cleanup ref, release lock, await shield,
      then RETRY the whole acquisition from the top.
    - initialized → return immediately.
    - init-in-flight → take the existing task ref.
    - idle → create a new init task with the T2 done-callback.

    Await via ``asyncio.shield()`` so cancelling a caller never cancels the
    shared work.

    Cancellation semantics (init-survives-caller-cancellation):
    - Cancelling a **waiter** raises ``CancelledError`` in the waiter only; the
      shared task keeps running and eventually publishes globals for any other
      concurrent caller.
    - If the sole surviving waiter is cancelled and the task later fails,
      a done-callback on the task logs the failure once and consumes the
      exception so asyncio does not emit "Task exception was never retrieved".
    """
    global _init_task

    lock = _get_init_lock()

    while True:
        async with lock:
            # --- Single-acquisition decision ---
            if _cleanup_task is not None and not _cleanup_task.done():
                # Cleanup is in flight: wait for it, then retry.
                cleanup_ref = _cleanup_task
            elif _engine is not None and _AsyncSessionFactory is not None:
                # Already initialized — fast path (inside the lock).
                return
            elif _init_task is None or _init_task.done():
                # No task in flight (first call or previous task done/failed).
                new_task: asyncio.Task[None] = asyncio.create_task(
                    _do_init(sync_url, async_url, db_path)
                )
                # T2: done-callback observes terminal failure when no waiter
                # remains.  If _do_init fails after all callers have been
                # cancelled, this callback logs it once and marks it retrieved —
                # preventing asyncio's "Task exception was never retrieved"
                # warning.  It does not affect propagation to live waiters.
                new_task.add_done_callback(_observe_init_task_exception)
                _init_task = new_task
                task = new_task
                cleanup_ref = None
            else:
                # init-in-flight: join the existing task.
                task = _init_task
                cleanup_ref = None

        if cleanup_ref is not None:
            # Wait for cleanup outside the lock so _do_cleanup can acquire it.
            await asyncio.shield(cleanup_ref)
            # Cleanup finished — retry the state-machine acquisition.
            continue

        # Await the shared init task through shield() so cancelling this
        # coroutine never cancels the shared work.
        await asyncio.shield(task)
        return


def _observe_init_task_exception(task: asyncio.Task[None]) -> None:
    """Done-callback: log a terminal init failure if no waiter consumed it.

    Called by asyncio when the shared ``_init_task`` completes.  If the task
    failed (not cancelled), calling ``task.exception()`` here marks the
    exception as retrieved, preventing asyncio's "Task exception was never
    retrieved" warning.  A proper log line is emitted instead.

    This callback does NOT affect exception propagation to live waiters — they
    receive the exception through their own ``await asyncio.shield(task)`` path
    before this callback fires if they are still present.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Database initialization failed (no active waiters retrieved the error): %s",
            exc,
            exc_info=exc,
        )


def _observe_cleanup_task_exception(task: asyncio.Task[None]) -> None:
    """Done-callback: log a terminal cleanup failure if no waiter consumed it.

    Called by asyncio when the shared ``_cleanup_task`` completes.  If the task
    failed (not cancelled), calling ``task.exception()`` here marks the
    exception as retrieved, preventing asyncio's "Task exception was never
    retrieved" warning.  A proper log line is emitted instead.

    This callback mirrors the T2 pattern for ``_observe_init_task_exception``.
    It does NOT affect exception propagation to live shield()-waiters.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Database cleanup failed (no active waiters retrieved the error): %s",
            exc,
            exc_info=exc,
        )


async def init_database(database_path: str | None = None) -> None:
    """Initialize the database connection using a once-task state machine.

    Concurrent callers await the SAME in-flight initialization — no caller
    returns before migrations complete.  On failure, globals are cleared
    atomically so a retry re-runs migrations from a clean state.

    Migrations run via ``asyncio.to_thread`` so this coroutine never blocks
    the event loop.  Safe to ``await`` directly inside a FastAPI lifespan or
    any async CLI bridge.

    Args:
        database_path: Path to the SQLite database file.
                      Defaults to DEFAULT_DATABASE_PATH.

    Raises:
        PermissionError: If directory cannot be created
        ValueError: If database path is invalid
    """
    from snore.database.target import DatabaseTarget  # noqa: PLC0415

    if database_path is None:
        database_path = DEFAULT_DATABASE_PATH

    if not database_path or not isinstance(database_path, str):
        raise ValueError(f"Invalid database path: {database_path}")

    target = DatabaseTarget.from_url(database_path)
    sync_url = target.resolve_sync_url()
    async_url = target.resolve_async_url()
    db_path = target.sqlite_path if target.is_sqlite else None

    await _init_with_once_task(sync_url, async_url, db_path)


async def init_database_from_url(database_url: str) -> None:
    """Initialise the database from a fully-formed SQLAlchemy URL.

    Concurrent callers await the SAME in-flight initialization — no caller
    returns before migrations complete.  Migrations run via
    ``asyncio.to_thread`` so this coroutine never blocks the event loop.

    Used by ``serve`` after the parent has resolved the canonical URL from the
    ``DatabaseTarget`` precedence chain and exported it as ``SNORE_DATABASE_URL``.

    Args:
        database_url: A fully-formed SQLAlchemy URL string.
    """
    from snore.database.target import DatabaseTarget  # noqa: PLC0415

    if not database_url:
        raise ValueError(f"Invalid database URL: {database_url!r}")

    target = DatabaseTarget.from_url(database_url)
    sync_url = target.resolve_sync_url()
    async_url = target.resolve_async_url()
    db_path = target.sqlite_path if target.is_sqlite else None

    await _init_with_once_task(sync_url, async_url, db_path)


def get_session() -> AsyncSession:
    """Return a new (unstarted) async database session.

    Raises:
        RuntimeError: If database has not been initialized.
    """
    if _AsyncSessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _AsyncSessionFactory()


async def _reinit_after_swap(sync_url: str, async_url: str, db_path: str) -> None:
    """Tear down the stale engine and reinitialize from the given URLs.

    Called by ``check_db_staleness`` via ``asyncio.shield`` when an atomic
    DB-file swap is detected.  Using shield ensures this coroutine runs to
    completion even if the triggering request is cancelled mid-swap.

    On success, ``_do_init``'s publish block clears ``_pending_reinit``.
    On failure or post-teardown cancellation, ``_pending_reinit`` is NOT
    cleared by ``_do_cleanup``, so subsequent calls to ``check_db_staleness``
    find it set and retry the reinit via the recovery path at the top of that
    function.
    """
    await cleanup_database()
    await _init_with_once_task(sync_url, async_url, db_path)


async def check_db_staleness() -> None:
    """Detect an atomic DB-file swap and transparently reinitialize the engine.

    SNORE's import pipeline atomically replaces the SQLite file with a
    ``rename(2)`` to a new inode.  The singleton async engine would otherwise
    keep serving the old unlinked inode forever.  This function compares the
    engine's recorded ``(st_dev, st_ino)`` against the live filesystem on every
    session open and triggers a transparent reinit when they differ.

    Hot path: one ``os.stat`` call, no lock.  On mismatch (cold path), the
    function sets ``_pending_reinit`` to record the intended reinit URLs before
    any await, then calls ``asyncio.shield(_reinit_after_swap(...))`` so the
    teardown and rebuild run to completion even if the calling request is
    cancelled.  On success ``_do_init`` clears ``_pending_reinit``; on failure
    or post-teardown cancellation it remains set.

    Recovery path: if ``_pending_reinit`` is set and ``_db_path`` is None, a
    previous swap-reinit was interrupted after teardown.  The recorded URLs are
    used to retry the reinit immediately via ``asyncio.shield``, so the engine
    is always restored regardless of which request first detects the failure.

    Concurrency safety:
    - Concurrent detections are harmless — both converge on the same
      once-cleanup-task and once-init-task so the engine is replaced exactly
      once regardless of how many callers detect the mismatch simultaneously.
    - No deadlock is possible: ``_init_lock`` is never held across an ``await``
      and ``session_scope`` never runs under it.
    - Module globals are captured into locals before the first ``await`` so a
      concurrent ``cleanup_database()`` cannot race between the mismatch check
      and the reinit call to produce a ``None`` dereference.

    Write-loss at swap time:
    Sessions already open when a swap occurs keep their connection to the OLD
    unlinked inode — reads succeed against the old data and any writes they
    commit land on the unlinked inode and are lost when its last file descriptor
    closes.  Detection happens at scope/session open only.

    Backend-agnostic contract:
    Detection applies ONLY to file-backed SQLite targets; for ``:memory:`` and
    non-SQLite backends (e.g. a future PostgreSQL target, where ``_db_path`` is
    ``None``) the function is a no-op costing one ``is None`` comparison — the
    storage layer stays dialect-agnostic.

    WAL/SHM note:
    After a swap, stale ``-wal``/``-shm`` files from the old inode may remain
    beside the new file; SQLite validates the WAL header salt on open and
    silently ignores an incompatible WAL, so this is safe — noted here so
    operators do not chase it.
    """
    global _pending_reinit

    # Recovery path: a previous swap-reinit was cancelled or failed after
    # teardown — _db_path is None but _pending_reinit carries the URLs needed
    # to complete the rebuild.  asyncio.shield ensures the reinit runs to
    # completion even if this caller is also cancelled.
    if _pending_reinit is not None and _db_path is None:
        pending = _pending_reinit
        await asyncio.shield(_reinit_after_swap(*pending))
        return

    # Fast exits: not a real on-disk file, or identity not recorded.
    if _db_path is None or _db_path == ":memory:" or _db_identity is None:
        return

    # Capture _db_path into a local so the stat and all subsequent reads use
    # the same value — a concurrent cleanup_database() cannot null it mid-function.
    db_path = _db_path

    try:
        st = os.stat(db_path)
    except FileNotFoundError:
        # File momentarily absent mid-swap — recheck on the next call.
        return
    except OSError as exc:
        logger.warning(
            "Unexpected error stat-ing database path %s; staleness detection "
            "skipped this call: %s",
            db_path,
            exc,
            exc_info=True,
        )
        return

    if (st.st_dev, st.st_ino) == _db_identity:
        return  # Hot path: identity unchanged.

    # Cold path: inode mismatch detected.  Capture remaining globals into locals
    # before the first await so a concurrent cleanup_database() cannot race
    # between the mismatch check and the reinit call.
    captured_sync_url = _sync_url
    captured_async_url = _async_url
    captured_identity = _db_identity

    if captured_sync_url is None or captured_async_url is None:
        # A concurrent cleanup cleared the URLs between our stat and here.
        # The engine is already being torn down; nothing left to do.
        return

    logger.warning(
        "Database file replaced on disk (inode %s -> %s) at path %s; "
        "disposing stale engine and reconnecting.",
        captured_identity,
        (st.st_dev, st.st_ino),
        db_path,
    )
    # Record the pending reinit BEFORE the first await so cancellation or a
    # failed reinit leaves _pending_reinit set — enabling the recovery path
    # above to retry on the next call.  _do_init clears it on success.
    _pending_reinit = (captured_sync_url, captured_async_url, db_path)
    await asyncio.shield(
        _reinit_after_swap(captured_sync_url, captured_async_url, db_path)
    )


@asynccontextmanager
async def session_scope(*, immediate: bool = False) -> AsyncGenerator[AsyncSession]:
    """Provide a transactional scope for async database operations.

    Args:
        immediate: When ``True``, emit ``BEGIN IMMEDIATE`` instead of plain
            ``BEGIN``.  Gated and bulk write scopes should pass
            ``immediate=True`` so contending writers queue on
            ``busy_timeout`` rather than failing instantly on a WAL snapshot
            upgrade conflict.  SQLite returns ``SQLITE_BUSY`` immediately
            (bypassing ``busy_timeout``) when a deferred transaction attempts
            its first write and another connection has committed since the
            deferred ``BEGIN`` — ``BEGIN IMMEDIATE`` avoids this by acquiring
            the write lock upfront.

    Usage::

        async with session_scope() as session:              # read or best-effort write
            result = await session.execute(...)

        async with session_scope(immediate=True) as session:  # write with lock queuing
            session.add(obj)

    Calls ``check_db_staleness()`` before opening the session and may
    transparently rebuild the engine if the database file was replaced on disk
    since the last initialization.  Sessions already open when a swap occurs
    keep their connection to the OLD unlinked inode; any writes they commit are
    lost when the inode's last file descriptor closes.

    Commits on success; rolls back on any exception.

    Yields:
        An async database session.
    """
    await check_db_staleness()
    session = get_session()
    try:
        async with session.begin():
            if immediate:
                # Force connection checkout with the IMMEDIATE option before any
                # SQL is executed.  The "begin" event listener reads this option
                # and emits BEGIN IMMEDIATE, so contending writers queue on
                # busy_timeout rather than fail instantly on a WAL snapshot
                # upgrade.  SQLAlchemy applies execution options at checkout
                # before conn.begin() fires the "begin" event, so the option
                # is visible to the listener when it runs.
                await session.connection(
                    execution_options={_TXN_OPT_KEY: _TXN_OPT_IMMEDIATE}
                )
            yield session
    except Exception:
        # begin() rolls back automatically on __aexit__ with an exception.
        raise
    finally:
        await session.close()


def get_engine() -> AsyncEngine:
    """Get the database engine."""
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _engine


def get_db_path() -> str:
    """Get the path to the initialized database."""
    if _db_path is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _db_path


def get_engine_generation() -> int:
    """Return the current engine generation counter.

    Increments on every successful ``_do_init`` and is never reset by
    ``cleanup_database`` or initialization failure.  Callers can snapshot this
    value before an operation and compare it afterwards to detect that the engine
    was rebuilt — for example after a swap-triggered reinit in
    ``check_db_staleness``.
    """
    return _engine_generation


async def _do_cleanup(owned_init_task: asyncio.Task[None] | None) -> None:
    """Owned teardown coroutine — runs as the shared ``_cleanup_task``.

    Accepts the init task that was in-flight when cleanup was requested (already
    detached from ``_init_task`` under the lock by the caller).  Cancels + awaits
    it to quiescence, disposes the published engine, and clears all globals.

    Two nested finalization layers:
    - Inner ``finally`` (inside the lock): clears ``_engine``, ``_AsyncSessionFactory``,
      and ``_db_path`` regardless of disposal outcome so a failed dispose never
      leaves stale published state.  Any disposal exception is re-raised after
      the clear so the cleanup task fails with the real error.
    - Outer ``finally``: clears ``_cleanup_task`` under the lock on EVERY exit
      (normal, exception, or cancellation) — state machine exits cleanup-in-flight
      with no stuck-barrier.
    """
    global \
        _engine, \
        _AsyncSessionFactory, \
        _db_path, \
        _cleanup_task, \
        _db_identity, \
        _sync_url, \
        _async_url

    lock = _get_init_lock()

    try:
        # Cancel and await the captured init task outside the lock so _do_init's
        # BaseException handler can run without deadlocking on the lock.
        if owned_init_task is not None and not owned_init_task.done():
            owned_init_task.cancel()
            try:
                await owned_init_task
            except (asyncio.CancelledError, Exception):
                pass

        # Dispose published state (engine.dispose() is an awaitable).
        # The inner finally guarantees all three globals are cleared even if
        # disposal raises, so a subsequent init always starts from a clean state.
        async with lock:
            try:
                if _engine is not None:
                    await _engine.dispose()
            finally:
                _engine = None
                _AsyncSessionFactory = None
                _db_path = None
                _db_identity = None
                _sync_url = None
                _async_url = None
                # _engine_generation deliberately survives — see module-level comment.
                # _pending_reinit also deliberately survives cleanup — the
                # swap-recovery path in check_db_staleness depends on it.
    finally:
        # Terminal state transition: always exit cleanup-in-flight,
        # whether we completed normally, raised, or were cancelled.
        async with lock:
            _cleanup_task = None


async def cleanup_database() -> None:
    """Clean up database connections and reset global state.

    Concurrent callers all await the SAME cleanup operation via
    ``asyncio.shield()``.  If a caller is cancelled, the underlying teardown
    task continues to completion so state is always left clean.

    After this coroutine returns, ``_engine``, ``_AsyncSessionFactory``,
    ``_db_path``, ``_db_identity``, ``_sync_url``, ``_async_url``,
    ``_init_task``, and ``_cleanup_task`` are all ``None``.
    Subsequent ``init_database()`` calls create a fresh engine.

    This function should be called during test teardown or application shutdown.
    """
    global _init_task, _cleanup_task

    lock = _get_init_lock()

    # Under the lock: create the cleanup task if none is running, detach
    # _init_task so it becomes the cleanup task's owned input, then take a
    # local ref regardless.  Concurrent callers all get the same task ref.
    async with lock:
        if _cleanup_task is None or _cleanup_task.done():
            owned_init = _init_task
            _init_task = None
            _cleanup_task = asyncio.create_task(_do_cleanup(owned_init))
            # V2: observe terminal failure when all callers have been cancelled.
            # Same pattern as T2 / _observe_init_task_exception.
            _cleanup_task.add_done_callback(_observe_cleanup_task_exception)
        cleanup_ref = _cleanup_task

    # Await via shield: caller cancellation does not cancel the teardown task.
    await asyncio.shield(cleanup_ref)
