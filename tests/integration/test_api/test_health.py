"""Integration tests for the /health and /health/busy endpoints and SPA fallback serving."""

from __future__ import annotations

from pathlib import Path

import pytest

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_200_in_multiuser_mode_without_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /health returns 200 with {"status": "ok"} in multiuser mode, no session."""
        monkeypatch.setenv("SNORE_AUTH_MODE", "multiuser")
        monkeypatch.setenv(
            "SNORE_SESSION_SECRET",
            "test-secret-at-least-32-chars-long-abcdef",
        )
        monkeypatch.setenv("SNORE_PUBLIC_BASE_URL", "http://127.0.0.1:8000")

        from snore.api.app import create_app

        app = create_app()
        # No 'with' — skips lifespan (no DB needed for /health).
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_absent_from_openapi_schema(self) -> None:
        """/health must not appear in the OpenAPI schema (include_in_schema=False)."""
        from snore.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        schema = client.get("/openapi.json").json()
        assert "/health" not in schema.get("paths", {})


class TestHealthBusyEndpoint:
    """Tests for the GET /health/busy watchtower pre-update gate endpoint."""

    def _make_client(self) -> TestClient:
        from snore.api.app import create_app

        app = create_app()
        # No 'with' — skips lifespan; no DB needed for /health/busy.
        return TestClient(app, raise_server_exceptions=True)

    def test_health_busy_idle_returns_not_busy(self) -> None:
        """Baseline: no active jobs → {"busy": false}."""
        resp = self._make_client().get("/health/busy")
        assert resp.status_code == 200
        assert resp.json() == {"busy": False}

    def test_health_busy_with_active_import_job(self) -> None:
        """A PENDING_UPLOAD import job → busy=true."""
        import snore.api.import_jobs as ij

        # reserve_slot() creates a PENDING_UPLOAD job (active state).
        # The reset_import_job_store autouse fixture cleans this up after the test.
        job = ij.reserve_slot(None)
        assert job is not None

        resp = self._make_client().get("/health/busy")
        assert resp.status_code == 200
        assert resp.json() == {"busy": True}

    def test_health_busy_with_queued_reset_job(self) -> None:
        """A QUEUED reset job → busy=true, unlike queued analysis or validation.

        The user has been told their data is being deleted, so a container swap
        must not silently drop the job.  The store fixture cleans it up.
        """
        import snore.api.reset_jobs as rj

        job = rj.enqueue(kind=rj.ResetJobKind.RESET, user_id=1, db_path="")
        assert job is not None

        resp = self._make_client().get("/health/busy")
        assert resp.status_code == 200
        assert resp.json() == {"busy": True}

    def test_health_busy_with_running_reset_job(self) -> None:
        """A RUNNING reset job, which includes its VACUUM → busy=true."""
        import snore.api.reset_jobs as rj

        job = rj.enqueue(kind=rj.ResetJobKind.DELETE_DATA, user_id=1, db_path="")
        assert job is not None
        assert job.try_start()

        resp = self._make_client().get("/health/busy")
        assert resp.status_code == 200
        assert resp.json() == {"busy": True}

    def test_health_busy_after_reset_job_finishes_is_not_busy(self) -> None:
        """A terminal reset job lingers for its status poll but is not busy."""
        import snore.api.reset_jobs as rj

        job = rj.enqueue(kind=rj.ResetJobKind.RESET, user_id=1, db_path="")
        assert job is not None
        assert job.try_start()
        job.finish(succeeded=True)

        resp = self._make_client().get("/health/busy")
        assert resp.status_code == 200
        assert resp.json() == {"busy": False}

    def test_health_busy_queued_analysis_job_is_not_busy(self) -> None:
        """A QUEUED (not yet started) analysis job → busy=false.

        QUEUED jobs have no in-progress data writes; interrupting them is safe
        because sessions are already committed and analysis re-triggers on the
        next import or manual batch run.
        """
        import snore.api.analysis_jobs as aj

        job = aj.enqueue(
            profile_id=1, session_ids=[1], source=aj.AnalysisJobSource.BATCH
        )
        assert job is not None
        # Deliberately do NOT call try_start() — job stays QUEUED.

        try:
            resp = self._make_client().get("/health/busy")
            assert resp.status_code == 200
            assert resp.json() == {"busy": False}
        finally:
            aj._all_jobs.clear()
            aj._queue.clear()

    def test_health_busy_with_running_analysis_job(self) -> None:
        """A RUNNING analysis job → busy=true."""
        import snore.api.analysis_jobs as aj

        job = aj.enqueue(
            profile_id=1, session_ids=[1], source=aj.AnalysisJobSource.BATCH
        )
        assert job is not None
        started = job.try_start()  # QUEUED → RUNNING (no real worker started)
        assert started

        try:
            resp = self._make_client().get("/health/busy")
            assert resp.status_code == 200
            assert resp.json() == {"busy": True}
        finally:
            # Clean up: mark finished so the job doesn't leak into other tests.
            job.finish(succeeded=False, error_message="test cleanup")
            aj._all_jobs.clear()
            aj._queue.clear()

    def test_health_busy_multiple_signals(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Active import + running analysis simultaneously → busy=true.

        Also verifies the debug log contains the reason tokens so operators
        can diagnose why a deploy was deferred via container logs.
        """
        import logging

        import snore.api.analysis_jobs as aj
        import snore.api.import_jobs as ij

        import_job = ij.reserve_slot(None)
        assert import_job is not None

        analysis_job = aj.enqueue(
            profile_id=1, session_ids=[1], source=aj.AnalysisJobSource.BATCH
        )
        assert analysis_job is not None
        analysis_job.try_start()

        try:
            with caplog.at_level(logging.DEBUG, logger="snore.api.app"):
                resp = self._make_client().get("/health/busy")
            assert resp.status_code == 200
            assert resp.json() == {"busy": True}
            # Reasons are internal-only; verify they appear in the debug log.
            assert any(
                "imports" in r.message and "analysis" in r.message
                for r in caplog.records
            )
        finally:
            analysis_job.finish(succeeded=False, error_message="test cleanup")
            aj._all_jobs.clear()
            aj._queue.clear()

    def test_health_busy_absent_from_openapi_schema(self) -> None:
        """/health/busy must not appear in the OpenAPI schema (include_in_schema=False)."""
        resp = self._make_client().get("/openapi.json")
        assert "/health/busy" not in resp.json().get("paths", {})


class TestSPAFallback:
    """Verify that the SPA fallback serves index.html for unknown paths but not
    for /api/ paths, which must always return their own error responses."""

    def _make_dist(self, tmp_path: Path) -> Path:
        """Create a minimal dist tree: assets/ dir + index.html."""
        (tmp_path / "assets").mkdir()
        (tmp_path / "index.html").write_text(
            "<html><body>SNORE SPA</body></html>", encoding="utf-8"
        )
        return tmp_path

    def test_root_returns_spa_index_html(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET / serves index.html (200 HTML) when ui/dist is present."""
        dist = self._make_dist(tmp_path)

        import snore.api.app as _app_module

        monkeypatch.setattr(_app_module, "_resolve_spa_dist", lambda: dist)

        from snore.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert resp.headers.get("cache-control") == "no-cache, must-revalidate"

    def test_api_404_not_overridden_by_spa(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GET /api/v1/nonexistent returns 404 JSON, not the SPA index.html."""
        dist = self._make_dist(tmp_path)

        import snore.api.app as _app_module

        monkeypatch.setattr(_app_module, "_resolve_spa_dist", lambda: dist)

        from snore.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/api/v1/nonexistent")
        assert resp.status_code == 404
        assert "text/html" not in resp.headers.get("content-type", "")

    def test_spa_fallback_sets_no_cache_header(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SPA fallback response carries Cache-Control: no-cache, must-revalidate."""
        dist = self._make_dist(tmp_path)

        import snore.api.app as _app_module

        monkeypatch.setattr(_app_module, "_resolve_spa_dist", lambda: dist)

        from snore.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache, must-revalidate"

    def test_assets_serve_immutable_cache_header(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Static assets under /assets get Cache-Control: public, max-age=31536000, immutable."""
        dist = self._make_dist(tmp_path)
        (dist / "assets" / "main.abc123.js").write_text(
            "console.log(1)", encoding="utf-8"
        )

        import snore.api.app as _app_module

        monkeypatch.setattr(_app_module, "_resolve_spa_dist", lambda: dist)

        from snore.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/assets/main.abc123.js")
        assert resp.status_code == 200
        assert (
            resp.headers.get("cache-control") == "public, max-age=31536000, immutable"
        )

        head_resp = client.head("/assets/main.abc123.js")
        assert head_resp.status_code == 200
        assert (
            head_resp.headers.get("cache-control")
            == "public, max-age=31536000, immutable"
        )

    def test_gzip_middleware_compresses_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GZip middleware compresses SPA fallback responses that exceed minimum_size."""
        dist = self._make_dist(tmp_path)
        # Override index.html with content > 1000 bytes so GZipMiddleware (minimum_size=1000)
        # compresses it.
        (dist / "index.html").write_text(
            "<html><body>" + ("SNORE " * 200) + "</body></html>", encoding="utf-8"
        )

        import snore.api.app as _app_module

        monkeypatch.setattr(_app_module, "_resolve_spa_dist", lambda: dist)

        from snore.api.app import create_app

        app = create_app()
        client = TestClient(app, raise_server_exceptions=True)
        resp = client.get("/", headers={"Accept-Encoding": "gzip"})
        assert resp.status_code == 200
        assert resp.headers.get("content-encoding") == "gzip"
