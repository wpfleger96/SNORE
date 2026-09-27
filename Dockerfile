# Stage 1: build the Vue SPA
FROM node:24-slim AS ui-builder

WORKDIR /app/ui

RUN corepack enable && corepack prepare pnpm@12.5.1 --activate

COPY ui/package.json ui/pnpm-lock.yaml ui/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile

COPY ui/ ./
RUN pnpm run build

# Stage 2: compile Python dependencies (full image has gcc for native extensions)
FROM python:3.14 AS python-builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# README.md is required by hatchling to build the editable install.
COPY src/ ./src/
COPY README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 3: slim runtime — copy the pre-built venv, no compiler needed
FROM python:3.14-slim AS runtime

WORKDIR /app

COPY --from=python-builder /app/.venv ./.venv
COPY --from=python-builder /app/src ./src/
COPY --from=python-builder /app/pyproject.toml ./

# Copy the built SPA.  _resolve_spa_dist() in app.py falls back to
# Path(__file__).parents[3] / "ui" / "dist", which resolves to /app/ui/dist
# for an editable install where __file__ == /app/src/snore/api/app.py.
COPY --from=ui-builder /app/ui/dist ./ui/dist/

# Lifecycle hook — named explicitly so dev-only scripts (export_openapi.py,
# post-screenshots.sh) are not shipped into the prod image.
COPY --chmod=755 scripts/watchtower-pre-update.sh ./scripts/watchtower-pre-update.sh

# Non-root user; HOME=/data is load-bearing — the app resolves all state
# ($HOME/.snore/snore.db, $HOME/.snore/raw/, $HOME/.snore/logs/) relative to
# $HOME.  Changing HOME requires a matching change to the volume mount path.
RUN groupadd -g 1000 snore && \
    useradd -u 1000 -g snore -d /data -s /usr/sbin/nologin -M snore && \
    mkdir -p /data && \
    chown snore:snore /data

# HOME=/data is load-bearing (see comment above).
ENV HOME=/data

ARG GIT_SHA=dev
ARG BUILD_TIME=
ENV SNORE_GIT_SHA=$GIT_SHA
ENV SNORE_BUILD_TIME=$BUILD_TIME

# /data persists the database, raw backups, and logs across container restarts.
# Bind-mount requirement: the host directory must be owned by uid/gid 1000
# (e.g. `chown 1000:1000 /opt/snore/data`).  Docker-managed volumes are chowned
# automatically; host-path mounts are not.
VOLUME /data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

USER snore

# Auth mode and secrets are supplied at runtime via env_file; the image bakes
# no secrets and assumes no auth mode.
CMD ["/app/.venv/bin/snore", "serve", "--host", "0.0.0.0", "--port", "8000"]
