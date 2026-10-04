# Local development overrides for SNORE.
# Imported by the main justfile via `import? 'local.just'`.

# Start the API in LOCAL auth mode (single-user, no login, identical to pre-multiuser UX).
dev: web-install web-build
    SNORE_AUTH_MODE=local uv run uvicorn snore.api.app:create_app --factory --reload --reload-dir src --host 127.0.0.1 --port 8000

# Start the API in MULTIUSER auth mode with a throwaway secret and a seeded admin invite.
# Prints the signup URL to stdout so you can paste it into a browser.
dev-auth: (_dev-auth "0")

# Like dev-auth, but requires TOTP enrollment for password-login users.
dev-auth-mfa: (_dev-auth "1")

_dev-auth require_totp: web-install web-build (dev-invite "dev@localhost" "admin")
    SNORE_AUTH_MODE=multiuser \
    SNORE_SESSION_SECRET=dev-throwaway-secret-not-for-production-1234567890ab \
    SNORE_PUBLIC_BASE_URL=http://127.0.0.1:8000 \
    SNORE_BIND_HOST=127.0.0.1 \
    SNORE_DEV_ORIGINS=http://localhost:5173,http://127.0.0.1:5173 \
    SNORE_REQUIRE_TOTP={{require_totp}} \
    uv run snore serve --host 127.0.0.1 --port 8000 --reload

# Seed an invite into the dev DB and print the signup URL — unless the
# account already exists (invites are signup-only; redeeming one for a
# registered email fails with a deliberately generic error).
# Run from a second terminal while dev-auth is up to add more users.
dev-invite email="member@localhost" role="member":
    @if uv run snore user list | grep -qF "] {{email}} "; then \
        echo "--- {{email}} is already registered — sign in at http://127.0.0.1:8000 ---"; \
    else \
        echo "--- Invite: {{email}} ({{role}}) ---"; \
        SNORE_PUBLIC_BASE_URL=http://127.0.0.1:8000 uv run snore user invite {{email}} --role {{role}}; \
    fi

# Seed fake Google bindings into the dev DB so the /admin/mcp reset flows are
# testable under `just dev-auth` (no real Google OAuth needed). Idempotent.
dev-seed-bindings:
    uv run python scripts/seed_dev_google_bindings.py

# Fully reset local dev state: stop stray dev servers (orphaned uvicorn
# reload workers are the usual "database is locked" culprit), drop the DB,
# and clear staged data. `snore db drop` skips sidecar cleanup when the main
# DB file is already gone, so -wal/-shm are removed explicitly. Logs are
# kept. The next `just dev` recreates the schema via startup migrations and
# re-provisions the local user; re-import your data afterwards.
reset:
    -pkill -f "[u]vicorn snore.api.app" 2>/dev/null
    -pkill -f "[s]nore serve" 2>/dev/null
    -uv run snore db drop --force
    rm -f ~/.snore/snore.db ~/.snore/snore.db-wal ~/.snore/snore.db-shm ~/.snore/vacuum.pending ~/.snore/writers.lock
    rm -rf ~/.snore/raw ~/.snore/spool
    @echo "Local dev state reset — run 'just dev' and re-import your data."

# Build the Docker image locally for smoke-testing.
docker-build:
    docker build -t snore .

# Regenerate ui/src/types/generated.ts from the FastAPI OpenAPI spec.
# Local auth mode: building the app requires no session secret, and all
# routers/schemas are mounted regardless of mode.
ui-generate-types:
    SNORE_AUTH_MODE=local uv run python scripts/export_openapi.py ui/openapi.json
    cd ui && pnpm run generate:types
