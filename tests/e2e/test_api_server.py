"""Live REST API exercised against a real `snore serve` process.

Unlike the in-process TestClient integration tests (which override `get_db`
and skip the lifespan), this boots the actual server subprocess against an
imported database. That covers the real uvicorn boot, the app lifespan, the
`/api/v1` routing, the `NotFoundError`→404 convention, and the OpenAPI schema
that the UI's generated types are derived from.
"""

from __future__ import annotations

import json

import pytest


@pytest.fixture
def running_api(server, imported_db):
    """A live server bound to the imported database for the duration of a test."""
    with server(imported_db) as srv:
        yield srv


def test_openapi_schema_is_served(running_api):
    resp = running_api.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["openapi"].startswith("3.")
    # The routes the UI depends on are all present under /api/v1.
    paths = schema["paths"]
    assert "/api/v1/sessions/" in paths
    assert any(p.startswith("/api/v1/stats/") for p in paths)


def test_sessions_listing_includes_imported_night(running_api):
    body = running_api.get("/api/v1/sessions/").json()
    items = body["items"] if isinstance(body, dict) else body
    assert any(item.get("id") == 1 for item in items)


def test_missing_session_returns_structured_404(running_api):
    """The `None`→`NotFoundError`→404 convention: a clean 404, not a 500."""
    resp = running_api.get("/api/v1/sessions/9999")
    assert resp.status_code == 404
    body = resp.json()
    # Structured error envelope, not an unhandled exception.
    assert body.get("error") == "not_found" or "not found" in json.dumps(body).lower()


def test_openapi_matches_committed_generated_types(running_api, request):
    """Guard against schema drift between the live API and the UI codegen.

    Three directions are checked:
    1. Paths: generated types must match live API paths bidirectionally.
    2. Components: every Pydantic schema component must appear in generated.ts
       — catches new fields (e.g. ``cancelled`` on ``BatchAnalysisResult``)
       that were added to Python but never regenerated.

    Skips cleanly when the generated file is not present.
    """
    import re

    repo_root = request.config.rootpath
    generated = repo_root / "ui" / "src" / "types" / "generated.ts"
    if not generated.exists():
        pytest.skip("ui/src/types/generated.ts not present in this checkout")

    schema = running_api.get("/openapi.json").json()
    live_paths = set(schema["paths"].keys())

    generated_text = generated.read_text()
    referenced = set(re.findall(r"['\"](?P<p>/api/v1/[^\'\"]+)[\'\"]", generated_text))
    if not referenced:
        pytest.skip("generated types declare no /api/v1 path keys to compare")

    # Direction 1a: generated references paths that no longer exist.
    stale = referenced - live_paths
    assert not stale, (
        f"ui generated types reference paths the live API no longer serves: {sorted(stale)}"
    )

    # Direction 1b: live API paths not yet in generated types.
    ungenerated = live_paths - referenced
    assert not ungenerated, (
        "live API paths missing from ui generated types "
        "(regenerate with pnpm generate:types): "
        f"{sorted(ungenerated)}"
    )

    # Direction 2: live schema components and their properties must be reflected in
    # the committed openapi.json artifact.  This catches new/changed fields (like
    # `cancelled` on BatchAnalysisResult) that were never regenerated.
    committed_openapi = repo_root / "ui" / "openapi.json"
    if committed_openapi.exists():
        import json as _json  # noqa: PLC0415

        committed_schema = _json.loads(committed_openapi.read_text())
        live_comps = schema.get("components", {}).get("schemas", {})
        committed_comps = committed_schema.get("components", {}).get("schemas", {})

        # Check every live component's required fields are in the committed artifact.
        # A missing or changed required field means regeneration is needed.
        for comp_name, live_comp in live_comps.items():
            committed_comp = committed_comps.get(comp_name)
            assert committed_comp is not None, (
                f"Schema component {comp_name!r} not in committed openapi.json "
                "(regenerate ui/openapi.json with pnpm generate:types)"
            )
            live_props = set(live_comp.get("properties", {}).keys())
            committed_props = set(committed_comp.get("properties", {}).keys())
            missing_props = live_props - committed_props
            assert not missing_props, (
                f"Component {comp_name!r} has properties in live API not in committed "
                f"openapi.json: {sorted(missing_props)}. "
                "Regenerate ui/openapi.json and ui/src/types/generated.ts."
            )
