"""Tests for DatabaseTarget — three-state resolver (§2).

Three states:
- Unrecognised dialect: parse raises ValueError.
- Recognised target: parse succeeds; resolution capability-gates PostgreSQL.
- Driver installed / operation supported: SQLite resolution returns a valid URL.

Also covers the full precedence chain, driver-qualified input stripping, and
the ``serve --db`` env-collision scenario.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sqlalchemy.engine import make_url

from snore.database.target import DatabaseTarget


class TestDatabaseTargetParsing:
    """Parsing is dialect-complete: sqlite and postgresql recognised; unknown dialects error."""

    def test_parse_unrecognised_dialect_raises_error(self):
        """Unrecognised dialect raises ValueError at parse time."""
        with pytest.raises(ValueError, match="Unrecognised database dialect"):
            DatabaseTarget.from_url("mysql://localhost/db")

    def test_parse_sqlite_url_succeeds(self):
        """sqlite:/// URL is recognised and parses successfully."""
        target = DatabaseTarget.from_url("sqlite:///path/to/db.sqlite")
        assert target.dialect == "sqlite"
        assert target.is_sqlite

    def test_parse_postgresql_url_succeeds(self):
        """postgresql:// URL is recognised at parse time."""
        target = DatabaseTarget.from_url("postgresql://localhost/snore")
        assert target.dialect == "postgresql"
        assert not target.is_sqlite

    def test_parse_bare_path_treated_as_sqlite(self):
        """A bare file path without a scheme is treated as sqlite."""
        target = DatabaseTarget.from_url("/tmp/test.db")
        assert target.dialect == "sqlite"

    def test_parse_driver_qualified_sqlite_strips_driver(self):
        """sqlite+pysqlite:// is stripped to sqlite and the URL is normalised."""
        target = DatabaseTarget.from_url("sqlite+pysqlite:///path/to/db.db")
        assert target.dialect == "sqlite"

    def test_parse_driver_qualified_postgresql_strips_driver(self):
        """postgresql+psycopg:// is stripped to postgresql at parse time."""
        target = DatabaseTarget.from_url("postgresql+psycopg://localhost/snore")
        assert target.dialect == "postgresql"


class TestDatabaseTargetResolution:
    """Resolution is capability-gated: sqlite resolves; postgresql is blocked."""

    def test_sqlite_sync_url_resolves_successfully(self):
        """sqlite target produces a valid sync SQLAlchemy URL."""
        target = DatabaseTarget.from_url("sqlite:///test.db")
        url = target.resolve_sync_url()
        assert url.startswith("sqlite+pysqlite:///")
        assert "test.db" in url

    def test_sqlite_migration_url_resolves_successfully(self):
        """sqlite target produces a valid migration URL."""
        target = DatabaseTarget.from_url("sqlite:///test.db")
        url = target.resolve_migration_url()
        assert url.startswith("sqlite+pysqlite:///")

    def test_postgresql_sync_url_capability_gated(self):
        """postgresql target raises a sanitised RuntimeError — no credentials in message."""
        target = DatabaseTarget.from_url("postgresql://user:secret@localhost/db")
        with pytest.raises(RuntimeError) as exc_info:
            target.resolve_sync_url()
        error_text = str(exc_info.value)
        assert "PostgreSQL support requires a driver" in error_text
        # Credentials must not appear in the error message.
        assert "secret" not in error_text
        assert "user" not in error_text

    def test_postgresql_migration_url_capability_gated(self):
        """postgresql migration URL also raises a sanitised RuntimeError."""
        target = DatabaseTarget.from_url("postgresql://localhost/snore")
        with pytest.raises(RuntimeError, match="PostgreSQL support requires a driver"):
            target.resolve_migration_url()


class TestDatabaseTargetPrecedenceChain:
    """Precedence: --db > SNORE_DATABASE_URL > SNORE_DB_PATH > default."""

    def test_db_flag_wins_over_env_vars(self, monkeypatch, tmp_path):
        """--db flag wins over both env vars; warnings are logged for ignored inputs."""
        monkeypatch.setenv("SNORE_DATABASE_URL", "sqlite:///env_url.db")
        monkeypatch.setenv("SNORE_DB_PATH", "/some/path.db")

        db_path = str(tmp_path / "flag.db")
        target = DatabaseTarget.from_env_and_flags(db_flag=db_path, warn_ignored=False)
        assert "flag.db" in target.location

    def test_database_url_wins_over_db_path(self, monkeypatch):
        """SNORE_DATABASE_URL wins over SNORE_DB_PATH."""
        monkeypatch.setenv("SNORE_DATABASE_URL", "sqlite:///url_wins.db")
        monkeypatch.setenv("SNORE_DB_PATH", "/ignored.db")
        monkeypatch.delenv("SNORE_DATABASE_URL", raising=False)
        monkeypatch.setenv("SNORE_DATABASE_URL", "sqlite:///url_wins.db")

        target = DatabaseTarget.from_env_and_flags(warn_ignored=False)
        assert "url_wins.db" in target.raw_url

    def test_db_path_used_when_no_url(self, monkeypatch):
        """SNORE_DB_PATH is used when SNORE_DATABASE_URL is not set."""
        monkeypatch.delenv("SNORE_DATABASE_URL", raising=False)
        monkeypatch.setenv("SNORE_DB_PATH", "/my/path.db")

        target = DatabaseTarget.from_env_and_flags(warn_ignored=False)
        assert "path.db" in target.location

    def test_default_path_used_when_no_inputs(self, monkeypatch):
        """Default SQLite path is used when no flags or env vars are set."""
        monkeypatch.delenv("SNORE_DATABASE_URL", raising=False)
        monkeypatch.delenv("SNORE_DB_PATH", raising=False)

        target = DatabaseTarget.from_env_and_flags(warn_ignored=False)
        assert target.dialect == "sqlite"

    def test_serve_db_flag_with_both_env_vars_preloads_child_with_canonical_url(
        self, monkeypatch
    ):
        """serve --db scenario: --db wins over both env vars; canonical URL is exported.

        Simulates a parent process that has SNORE_DATABASE_URL and SNORE_DB_PATH
        inherited from the environment. The --db flag must override both.
        """
        monkeypatch.setenv("SNORE_DATABASE_URL", "sqlite:///inherited_url.db")
        monkeypatch.setenv("SNORE_DB_PATH", "/inherited/path.db")

        db_flag_path = "/explicit/db.db"
        target = DatabaseTarget.from_env_and_flags(
            db_flag=db_flag_path, warn_ignored=True
        )
        canonical_url = target.resolve_sync_url()

        # The resolved URL points to the --db flag value, not the env vars.
        assert "db.db" in canonical_url
        assert "inherited" not in canonical_url


class TestDatabaseTargetExactURLAssertions:
    """Exact URL/path assertions covering all documented edge cases (§2).

    These tests pin the exact make_url/URL.set behaviour to prevent regressions
    from any future hand-rolled parser creeping back in.
    """

    # --- SQLite three-slash relative ---

    def test_sqlite_relative_three_slash_location_is_filename_only(self):
        """sqlite:///rel.db → location is 'rel.db' (not '/rel.db')."""
        target = DatabaseTarget.from_url("sqlite:///rel.db")
        assert target.location == "rel.db"
        assert target.sqlite_path == "rel.db"

    def test_sqlite_relative_three_slash_sync_url_is_three_slash(self):
        """sqlite:///rel.db resolves to sqlite+pysqlite:///rel.db (not ////rel.db)."""
        target = DatabaseTarget.from_url("sqlite:///rel.db")
        url = target.resolve_sync_url()
        assert url == "sqlite+pysqlite:///rel.db"

    # --- SQLite four-slash absolute ---

    def test_sqlite_absolute_four_slash_location_is_absolute_path(self):
        """sqlite:////abs/path.db → location is '/abs/path.db'."""
        target = DatabaseTarget.from_url("sqlite:////abs/path.db")
        assert target.location == "/abs/path.db"
        assert target.sqlite_path == "/abs/path.db"

    def test_sqlite_absolute_four_slash_sync_url_is_four_slash(self):
        """sqlite:////abs/path.db resolves to sqlite+pysqlite:////abs/path.db."""
        target = DatabaseTarget.from_url("sqlite:////abs/path.db")
        url = target.resolve_sync_url()
        assert url == "sqlite+pysqlite:////abs/path.db"

    # --- SQLite nested relative path ---

    def test_sqlite_nested_relative_path_preserved(self):
        """sqlite:///sub/dir/db.sqlite → location is 'sub/dir/db.sqlite'."""
        target = DatabaseTarget.from_url("sqlite:///sub/dir/db.sqlite")
        assert target.location == "sub/dir/db.sqlite"
        url = target.resolve_sync_url()
        assert url == "sqlite+pysqlite:///sub/dir/db.sqlite"

    # --- :memory: ---

    def test_sqlite_memory_url_location_is_memory_sentinel(self):
        """sqlite:///:memory: → location is ':memory:'."""
        target = DatabaseTarget.from_url("sqlite:///:memory:")
        assert target.location == ":memory:"
        assert target.sqlite_path == ":memory:"

    def test_sqlite_memory_sync_url_is_exact(self):
        """sqlite:///:memory: resolves to sqlite+pysqlite:///:memory:."""
        target = DatabaseTarget.from_url("sqlite:///:memory:")
        url = target.resolve_sync_url()
        # Compare parsed URLs: SQLAlchemy 2.1 renders ":" as "%3A" in the path.
        assert make_url(url) == make_url("sqlite+pysqlite:///:memory:")

    # --- Query parameters preserved ---

    def test_sqlite_query_parameters_preserved_in_resolved_url(self):
        """Query parameters (e.g. ?timeout=5000) survive resolution."""
        target = DatabaseTarget.from_url("sqlite:///my.db?timeout=5000")
        url = target.resolve_sync_url()
        assert "timeout=5000" in url

    # --- Driver-qualified input normalised ---

    def test_sqlite_pysqlite_qualified_relative_resolves_correctly(self):
        """sqlite+pysqlite:///rel.db resolves to the correct relative path."""
        target = DatabaseTarget.from_url("sqlite+pysqlite:///rel.db")
        assert target.dialect == "sqlite"
        assert target.location == "rel.db"
        url = target.resolve_sync_url()
        assert url == "sqlite+pysqlite:///rel.db"

    def test_sqlite_pysqlite_qualified_absolute_resolves_correctly(self):
        """sqlite+pysqlite:////abs/path.db resolves to the correct absolute path."""
        target = DatabaseTarget.from_url("sqlite+pysqlite:////abs/path.db")
        assert target.dialect == "sqlite"
        assert target.location == "/abs/path.db"
        url = target.resolve_sync_url()
        assert url == "sqlite+pysqlite:////abs/path.db"

    # --- Bare path (no scheme) ---

    def test_bare_relative_path_location_equals_input(self):
        """A bare relative path is treated as sqlite with location == input."""
        target = DatabaseTarget.from_url("my/data.db")
        assert target.dialect == "sqlite"
        assert target.location == "my/data.db"

    def test_bare_absolute_path_location_equals_input(self):
        """A bare absolute path is treated as sqlite with location == input."""
        target = DatabaseTarget.from_url("/tmp/test.db")
        assert target.dialect == "sqlite"
        assert target.location == "/tmp/test.db"


class TestServeDbCollisionScenario:
    """serve --db scenario: the resolved URL must point to the flag-specified DB.

    The key property: when snore serve exports SNORE_DATABASE_URL to the child
    process, that URL must open the correct file — not whatever SNORE_DB_PATH
    or SNORE_DATABASE_URL the child may have inherited.
    """

    def test_serve_db_flag_resolves_correct_path(self, monkeypatch, tmp_path):
        """--db /explicit/db.db → resolved URL contains the exact path, not inherited vars."""
        monkeypatch.setenv("SNORE_DATABASE_URL", "sqlite:///inherited.db")
        monkeypatch.setenv("SNORE_DB_PATH", "/also/inherited.db")

        explicit_path = str(tmp_path / "serve.db")
        target = DatabaseTarget.from_env_and_flags(
            db_flag=explicit_path, warn_ignored=False
        )
        canonical_url = target.resolve_sync_url()

        # Exact check: the URL must contain the flag-specified path.
        assert "serve.db" in canonical_url
        assert "inherited" not in canonical_url
        # The URL must be a valid sqlite+pysqlite:/// URL.
        assert canonical_url.startswith("sqlite+pysqlite://")

    def test_serve_db_flag_sqlite_path_attribute_is_flag_value(
        self, monkeypatch, tmp_path
    ):
        """target.sqlite_path must equal the flag value (not a mangled version)."""
        monkeypatch.delenv("SNORE_DATABASE_URL", raising=False)
        monkeypatch.delenv("SNORE_DB_PATH", raising=False)

        explicit_path = str(tmp_path / "exact.db")
        target = DatabaseTarget.from_env_and_flags(
            db_flag=explicit_path, warn_ignored=False
        )
        assert target.sqlite_path == explicit_path


# ---------------------------------------------------------------------------
# Serve command: real invocation verifies env export (§2 serve --db test)
# ---------------------------------------------------------------------------


class TestServeCommandEnvExport:
    """The ``serve --db`` command must export SNORE_DATABASE_URL before handing
    off to uvicorn, overriding any previously-set SNORE_DATABASE_URL or
    SNORE_DB_PATH that the process inherited.

    This class invokes the real ``snore serve`` Click command (with uvicorn.run
    mocked so no server actually starts) and asserts the env state seen by the
    would-be child process.
    """

    def test_serve_db_exports_canonical_url_overrides_inherited_env(
        self, monkeypatch, tmp_path
    ):
        """``snore serve --db <path>`` exports canonical URL; SNORE_DB_PATH removed.

        Invokes the real serve Click command via Click's test runner.
        ``uvicorn.run`` is replaced by a spy so no server starts; the test
        inspects the environment state captured at the moment uvicorn would run.
        """
        import os

        from unittest.mock import patch

        from click.testing import CliRunner

        from snore.cli.commands.serve import serve

        explicit_db = str(tmp_path / "serve_test.db")
        inherited_url = "sqlite:///should_not_appear.db"
        inherited_path = "/should/also/not/appear.db"

        captured_env: dict[str, str] = {}

        def spy_uvicorn(*args: object, **kwargs: object) -> None:  # noqa: ARG001
            captured_env.update(os.environ)

        runner = CliRunner()
        with patch("snore.cli.commands.serve.uvicorn.run", spy_uvicorn):
            result = runner.invoke(
                serve,
                ["--db", explicit_db, "--port", "19999"],
                env={
                    "SNORE_DATABASE_URL": inherited_url,
                    "SNORE_DB_PATH": inherited_path,
                },
                catch_exceptions=False,
            )

        assert result.exit_code == 0, (
            f"serve exited {result.exit_code}: {result.output}"
        )
        assert "SNORE_DATABASE_URL" in captured_env, (
            "serve must export SNORE_DATABASE_URL before handing off to uvicorn"
        )
        exported_url = captured_env["SNORE_DATABASE_URL"]
        assert "serve_test.db" in exported_url, (
            f"Exported URL {exported_url!r} must reference the --db flag path"
        )
        assert "should_not_appear" not in exported_url, (
            f"Exported URL {exported_url!r} must not reference the inherited SNORE_DATABASE_URL"
        )
        assert "SNORE_DB_PATH" not in captured_env, (
            "serve must remove SNORE_DB_PATH before handing off to uvicorn"
        )
        assert exported_url.startswith("sqlite+pysqlite://"), (
            f"Exported URL must be a valid SQLAlchemy SQLite URL; got {exported_url!r}"
        )

    def test_lifespan_opens_flagged_database(self, monkeypatch, tmp_path):
        """The app lifespan opens the flagged database, not inherited env vars.

        Invokes the real ``snore serve`` Click command to capture the exported
        URL, then runs the FastAPI lifespan with the real ``init_database_from_url``
        (not patched).  Proves the flagged DB file is created on disk — not just
        that the env variable is set.
        """
        import asyncio  # noqa: PLC0415
        import os  # noqa: PLC0415

        from unittest.mock import patch  # noqa: PLC0415

        from click.testing import CliRunner  # noqa: PLC0415

        from snore.cli.commands.serve import serve  # noqa: PLC0415

        explicit_db = str(tmp_path / "lifespan_test.db")
        captured_env: dict[str, str] = {}

        def spy_uvicorn(*args: object, **kwargs: object) -> None:  # noqa: ARG001
            captured_env.update(os.environ)

        runner = CliRunner()
        with patch("snore.cli.commands.serve.uvicorn.run", spy_uvicorn):
            result = runner.invoke(
                serve,
                ["--db", explicit_db, "--port", "19999"],
                # Passing env={"SNORE_DATABASE_URL": None} tells Click to save and
                # restore this key.  serve() writes it to os.environ directly; without
                # this entry CliRunner's finally block would not clean it up and the
                # canonical URL would leak into the next test.
                env={"SNORE_DATABASE_URL": None},
                catch_exceptions=False,
            )

        assert result.exit_code == 0, (
            f"serve exited {result.exit_code}: {result.output}"
        )
        exported_url = captured_env.get("SNORE_DATABASE_URL", "")
        assert "lifespan_test.db" in exported_url, (
            f"Exported URL must reference the flag DB; got {exported_url!r}"
        )

        # Run the real lifespan with the exported URL.  init_database_from_url
        # is called WITHOUT patching so the SQLite file is actually created on disk.
        # This proves the lifespan opens the flagged DB, not an inherited path.
        from snore.api.app import create_app  # noqa: PLC0415
        from snore.database.session import cleanup_database  # noqa: PLC0415

        app = create_app()
        # Click's finally block deleted SNORE_DATABASE_URL (it was saved as
        # "not set" via env={"SNORE_DATABASE_URL": None} above).  Set it back
        # for the lifespan call; monkeypatch records NOTSET and cleans it up
        # on teardown, so this does not leak.
        monkeypatch.setenv("SNORE_DATABASE_URL", exported_url)

        async def run_lifespan() -> None:
            import threading as _threading  # noqa: PLC0415

            _dummy_stop = _threading.Event()
            _dummy_thread = _threading.Thread(target=_dummy_stop.wait, daemon=True)
            _dummy_thread.start()
            with (
                patch(
                    "snore.api.app._start_import_reaper",
                    return_value=(_dummy_thread, _dummy_stop),
                ),
                patch("snore.api.app._shutdown_import_jobs", return_value=[]),
            ):
                async with app.router.lifespan_context(app):
                    pass  # Startup + immediate shutdown

        asyncio.run(run_lifespan())
        asyncio.run(cleanup_database())

        # The flagged database file must exist on disk — the lifespan opened it.
        assert (tmp_path / "lifespan_test.db").exists(), (
            "The flagged SQLite DB file must be created by the lifespan. "
            "If absent, the lifespan opened a different database."
        )


# ---------------------------------------------------------------------------
# Suite-wide DB guard observability
# ---------------------------------------------------------------------------


class TestSuiteDbGuard:
    """Observes that the suite-wide _block_real_db guard in tests/conftest.py is active.

    This test performs NO env manipulation of its own — it relies entirely on
    the conftest guard being in place.  If both the module-level baseline and
    the ``_block_real_db`` fixture body are removed from conftest.py, this test
    fails first, surfacing the regression before any test can accidentally write
    to the real user database.
    """

    def test_env_chain_resolves_to_guard_db_not_real_db(self):
        """Without any env manipulation, DB resolution lands in the guard tmp path.

        This test observes the suite-wide ``_block_real_db`` guard from
        tests/conftest.py; if the guard is removed, this test fails first.
        """
        from snore.constants import DEFAULT_DATABASE_PATH

        target = DatabaseTarget.from_env_and_flags(warn_ignored=False)

        assert target.is_sqlite
        assert Path(target.sqlite_path).name == "guard.db"
        assert (
            Path(target.sqlite_path).resolve()
            != Path(DEFAULT_DATABASE_PATH).expanduser().resolve()
        )
