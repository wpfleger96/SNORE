"""SNORE MCP Server.

Third presentation layer over the async service layer — a peer of the CLI and
FastAPI, never a place where analysis logic lives.

Design doctrine:
  - Tiered data access: overview → summary → events → breath table → raw waveform
  - Compute server-side, return compact JSON; units on every field
  - Data-quality flags / null + reason everywhere (G2)
  - Stateless: no module-global state; DB access injected via SNORERuntime (G3)
  - Profile-parameterized: profiles shape instructions only, not data (G1)
  - Vendor dispatch stays in the parser/service layer (G4)

DB-access pattern (M2 / Thufir MINOR):
  ``_lifespan`` builds a ``SNORERuntime`` (Protocol) and yields it as the
  FastMCP lifespan context.  Every tool and resource receives a ``ctx: Context``
  parameter; FastMCP injects it by type annotation and excludes it from the
  client-facing schema.  Tools call ``runtime = _runtime(ctx)`` to extract it
  and then ``runtime.scope_provider()`` instead of ``session_scope()`` directly.

  Two concrete implementations exist:
    ``StaticRuntime`` — used by the stdio path; profile_id is looked up at
      startup from the first live profile row, and refreshed automatically
      when the underlying SQLite file is replaced (new inode → new engine
      generation detected inside ``scope_provider``).
    ``ActorRuntime`` — used by the OAuth HTTP path; scope_provider is
      ``actor_scope`` from ``snore.mcp.auth``; profile_id is read from the
      per-request ``ActorContext`` on each property access.
"""

from __future__ import annotations

import json
import logging

from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from importlib.metadata import version
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from fastmcp.server.auth import AuthProvider

    from snore.services.breath_service import RawWaveformWindow

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from sqlalchemy.ext.asyncio import AsyncSession

from snore.database.session import (
    cleanup_database,
    get_engine_generation,
    init_database_from_url,
    session_scope,
)
from snore.database.target import DatabaseTarget
from snore.mcp.profiles import ClinicalProfile, get_profile
from snore.mcp.schemas import SCHEMA_MODEL_MAP, model_to_schema
from snore.mcp.tools._scaffold import (  # noqa: F401  (re-exported; tests import from here)
    RESPONSE_SIZE_LIMIT,
    ArgumentValidationMiddleware,
    _check_response_size,
    _runtime,
    _scope_and_run,
    tool_error_boundary,
)
from snore.mcp.validation import parse_date
from snore.provenance import PROVENANCE_NOTES, response_provenance

logger = logging.getLogger(__name__)

# DB-access seam (M2): tools call runtime.scope_provider(), never session_scope()
# directly.  ``SNORERuntime`` is a Protocol (two implementations: StaticRuntime
# for stdio, ActorRuntime for OAuth HTTP).  Built in _lifespan and injected via
# FastMCP's lifespan context; no tool code needs changing for auth.
# Type: a zero-arg callable returning an async context manager that yields AsyncSession.
_ScopeProvider = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class SNORERuntime(Protocol):
    """Protocol implemented by StaticRuntime (stdio) and ActorRuntime (OAuth HTTP).

    FastMCP yields a conforming instance from ``_lifespan`` and makes it
    available as ``ctx.lifespan_context``.  Tools access ``scope_provider``
    and ``profile_id`` through this interface without knowing which
    implementation is active.
    """

    @property
    def scope_provider(self) -> _ScopeProvider: ...
    @property
    def profile_id(self) -> int: ...


class StaticRuntime:
    """Generation-aware stdio-path runtime.

    ``profile_id`` is resolved from the live profile with the lowest ID at
    server startup and re-resolved automatically whenever the SQLite file
    underneath the engine is replaced (detected by a change in the value
    returned by ``get_engine_generation()`` between consecutive requests).

    Retry-until-found: when the new DB contains no live profile at the moment
    the generation change is detected, ``_known_generation`` is intentionally
    NOT advanced so every subsequent scope entry re-attempts resolution until
    a live profile is found.  The cost is one extra ``SELECT … LIMIT 1`` per
    scope entry while in this transient, pathological state; once a live
    profile is found the generation is anchored and the overhead reverts to zero.

    Refresh safety — why mutation is safe here:
      - The refresh query is idempotent: ``SELECT … WHERE deleting_at IS NULL
        ORDER BY id LIMIT 1`` always returns the same row for a given DB state.
      - Concurrent tool calls that both detect a generation change will run the
        same query and write the same value; last-write-wins produces no
        inconsistency.
      - All tools read ``runtime.profile_id`` *inside* an already-open scope
        (i.e. after ``async with runtime.scope_provider() as db``), so
        ``_profile_id`` is always fully updated before it is consumed.

    Designed for the stdio transport where tool calls are effectively serial;
    the unsynchronized ``_profile_id`` mutation relies on that assumption —
    multiuser HTTP uses ``ActorRuntime`` with per-request context instead.
    """

    def __init__(self, base_scope_provider: _ScopeProvider, profile_id: int) -> None:
        self._base_scope_provider = base_scope_provider
        self._profile_id = profile_id
        self._known_generation: int = get_engine_generation()
        self._scope_provider: _ScopeProvider = self._scoped_provider

    @asynccontextmanager
    async def _scoped_provider(self) -> AsyncGenerator[AsyncSession]:
        async with self._base_scope_provider() as db:
            current_gen = get_engine_generation()
            if current_gen != self._known_generation:
                new_profile_id = await _resolve_first_profile_id(db)
                if new_profile_id is not None:
                    self._profile_id = new_profile_id
                    self._known_generation = current_gen
                    logger.info(
                        "StaticRuntime: refreshed profile_id=%d after database file replacement",
                        self._profile_id,
                    )
                else:
                    logger.warning(
                        "StaticRuntime: new DB has no live profile yet; retaining "
                        "profile_id=%d — will retry on next scope entry",
                        self._profile_id,
                    )
            yield db

    @property
    def scope_provider(self) -> _ScopeProvider:
        return self._scope_provider

    @property
    def profile_id(self) -> int:
        return self._profile_id


class ActorRuntime:
    """OAuth HTTP runtime: per-request actor provides scope and profile_id.

    ``scope_provider`` is ``actor_scope`` from snore.mcp.auth (imported lazily
    so stdio startup never loads OAuth extras).  ``profile_id`` delegates to
    ``current_actor()`` on each access, reflecting whichever actor is bound to
    the current request context var.
    """

    def __init__(self, scope_provider: _ScopeProvider) -> None:
        self._scope_provider = scope_provider

    @property
    def scope_provider(self) -> _ScopeProvider:
        return self._scope_provider

    @property
    def profile_id(self) -> int:
        from snore.mcp.auth import current_actor  # noqa: PLC0415

        return current_actor().profile_id


def _provenance_legend() -> str:
    return "\n".join(
        f"  {tier.value.upper()}: {note}" for tier, note in PROVENANCE_NOTES.items()
    )


def _build_instructions(profile: ClinicalProfile) -> str:
    return f"""\
SNORE MCP Server v{version("snore")}
Sleep eNvironment Observation & Respiratory Evaluation

Each tool's description documents its parameters, refusal semantics, and error
conditions. Use `docs://schemas/{{type}}` for full response JSON schemas.

Active clinical profile: {profile.display_name}
{profile.priority_hint}

Clinical context:
{profile.clinical_context}

WORKFLOW:
1. get_data_overview  → discover devices, date ranges, channels
2. get_nightly_summary → identify nights of interest (30 nights/page; incl. periodic-breathing %)
3. get_settings_timeline → understand settings epochs
4. get_settings_changes → full audit log of device settings changes + user-logged mask equipment changes
5. get_events (date) → event-level detail for a night (incl. MV slope/stability, delivered PS)
6. get_breath_table, find_windows, compare_epochs for breath morphology tuning
7. render_window → PNG chart for visual inspection (≤15 min window)
8. get_waveform → raw per-sample arrays for deep inspection (≤2 min window)

DATA TIERS (progressive disclosure):
  Tier 1 (primary):  computed metrics — indices, percentiles, aggregates
  Tier 2 (secondary): PNG charts — render_window (≤15 min window)
  Tier 3 (escape hatch): raw arrays — get_waveform ≤2 min / ≤1000 pts

NULL FIELDS: When data is absent, fields are null + a companion *_reason field
explains why (e.g. rera_index_reason: "analysis_not_run"). Never infer from null.

PROVENANCE: every metric has one provenance class:
{_provenance_legend()}
Field descriptions in docs://schemas start with [DEVICE] / [DERIVED] /
[EXPERIMENTAL]. Responses with derived or experimental fields carry a top-level
`provenance` block listing those field paths; metric fields it does not list
are device data, identifiers and bookkeeping fields (ids, offsets, pagination,
coverage counts) are not classified, and responses with only device data carry
no block. The block's `source_dependent` map takes a field path to the sibling
field whose value decides that value's tier (e.g. MV fields ->
`events[].context.mv_source`, where `flow_derived` means experimental). Device-scored events are the reference
standard; never present experimental metrics as device-scored or clinically
validated.

See docs://capabilities for dataset-specific channel availability.
"""


# ---------------------------------------------------------------------------
# Profile resolution helper
# ---------------------------------------------------------------------------


async def _resolve_first_profile_id(db: AsyncSession) -> int | None:
    """Return the id of the first live profile, or None when none exists.

    Selects the ``Profile`` row with ``deleting_at IS NULL``, ordered by id
    ascending, limit 1.  Used at startup (inside ``_lifespan``) and on every
    engine-generation change detected by ``StaticRuntime._scoped_provider``.
    """
    from sqlalchemy import select  # noqa: PLC0415

    from snore.database import models  # noqa: PLC0415

    row = (
        (
            await db.execute(
                select(models.Profile)
                .where(models.Profile.deleting_at.is_(None))
                .order_by(models.Profile.id)
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    return int(row.id) if row is not None else None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(
    app: Any,
    db_flag: str | None = None,
    profile_name: str = "neutral",
    *,
    actor_scoped: bool = False,
    manage_database: bool = True,
) -> AsyncGenerator[SNORERuntime]:
    """FastMCP lifespan: initialize DB, build runtime, yield to tools.

    Yields a ``SNORERuntime`` implementation as the lifespan context so every
    tool and resource can access the scope-provider seam and profile_id without
    module globals.  Teardown calls ``cleanup_database()`` in a ``finally``
    block so it runs even if tool errors occur during shutdown.

    When ``actor_scoped=True`` (OAuth HTTP), the per-request actor carries its
    own profile_id and scope; startup skips the first-live-profile query so an
    empty database does not prevent server start.

    When ``manage_database=False`` (embedded into FastAPI), the FastAPI lifespan
    owns the module-global engine; this lifespan must not initialize or clean up
    the database — doing so would dispose the shared engine mid-flight.
    """
    from snore.parsers.register_all import ensure_registered_parsers  # noqa: PLC0415

    if manage_database:
        target = DatabaseTarget.from_env_and_flags(db_flag=db_flag, warn_ignored=True)
        async_url = target.resolve_async_url()
        await init_database_from_url(async_url)

    try:
        # Register vendor parsers once at startup (idempotent; tools must not call
        # ensure_registered_parsers() themselves — this is the single call site).
        ensure_registered_parsers()
        # Warm the per-model provenance cache so first tool calls skip schema generation.
        for model in SCHEMA_MODEL_MAP.values():
            response_provenance(model)

        db_location = target.location if manage_database else "embedded"
        runtime: SNORERuntime
        if actor_scoped:
            from snore.mcp.auth import actor_scope  # noqa: PLC0415

            # profile_name shapes only the instructions text; per-request actors
            # carry their own profile_id, so no profile row is resolved here.
            runtime = ActorRuntime(scope_provider=actor_scope)
            logger.info(
                "SNORE MCP server started — db=%r profile=actor-scoped (OAuth)",
                db_location,
            )
        else:
            # Resolve the active profile — required by BreathService, DeviceService,
            # and RxTracker (all scoped to a profile_id since multiuser Phase 1).
            async with session_scope() as _db:
                profile_id = await _resolve_first_profile_id(_db)
                if profile_id is None:
                    raise RuntimeError(
                        "No profile found in database — run 'snore db init' first "
                        "or import CPAP data to create a profile."
                    )

            runtime = StaticRuntime(
                base_scope_provider=session_scope, profile_id=profile_id
            )
            logger.info(
                "SNORE MCP server started — db=%r profile=%s profile_id=%d",
                db_location,
                profile_name,
                runtime.profile_id,
            )
    except Exception:
        if manage_database:
            await cleanup_database()
        raise

    try:
        yield runtime
    finally:
        if manage_database:
            await cleanup_database()
        logger.info("SNORE MCP server stopped")


def make_server(
    db_flag: str | None = None,
    profile_name: str = "neutral",
    auth: AuthProvider | None = None,
    *,
    manage_database: bool = True,
) -> FastMCP:
    """Construct and return a configured FastMCP instance.

    When ``auth`` is provided (OAuth HTTP path), the server uses actor-scoped
    sessions — each request's actor carries its own profile_id, and the
    database startup check requiring at least one profile is skipped.

    When ``manage_database=False`` (embedded into FastAPI via ``mcp_embed``),
    the FastAPI lifespan owns the module-global engine; this server must not
    initialize or clean up the database.  ``auth`` must be provided in this
    case — embedded mode always uses actor-scoped auth.

    Raises:
        ValueError: If ``manage_database=False`` and ``auth`` is None.
    """
    if not manage_database and auth is None:
        raise ValueError(
            "manage_database=False requires auth — embedded mode requires "
            "actor-scoped auth so every request resolves its own DB scope"
        )

    profile = get_profile(profile_name)

    @asynccontextmanager
    async def _bound_lifespan(app: Any) -> AsyncGenerator[SNORERuntime]:
        async with _lifespan(
            app,
            db_flag=db_flag,
            profile_name=profile_name,
            actor_scoped=auth is not None,
            manage_database=manage_database,
        ) as rt:
            yield rt

    mcp = FastMCP(
        name="snore",
        instructions=_build_instructions(profile),
        lifespan=_bound_lifespan,
        auth=auth,
        middleware=[ArgumentValidationMiddleware()],
    )

    _register_resources(mcp)
    _register_tools(mcp)

    return mcp


async def _fetch_waveform_for_tool(
    ctx: Context,
    date: str,
    offset_start: float,
    offset_end: float,
    device_id: int | None,
    session_id: int | None,
    channels: list[str] | None,
    max_points: int | None,
    window_cap_seconds: float,
) -> RawWaveformWindow:
    """Parse date, extract runtime, fetch raw waveform within a scoped DB session.

    Shared by ``get_waveform`` and ``render_window``; the only difference between
    the two callers is ``window_cap_seconds`` (120 s vs. 900 s).  Post-processing
    (JSON serialisation or PNG render) runs in each wrapper after this returns,
    outside the DB transaction.
    """
    from snore.mcp.tools.waveform import fetch_waveform_raw  # noqa: PLC0415

    runtime = _runtime(ctx)
    therapy_date = parse_date(date, "date")

    async with runtime.scope_provider() as db:
        return await fetch_waveform_raw(
            db,
            therapy_date,
            profile_id=runtime.profile_id,
            offset_start=offset_start,
            offset_end=offset_end,
            device_id=device_id,
            session_id=session_id,
            channels=channels,
            max_points=max_points,
            window_cap_seconds=window_cap_seconds,
        )


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


def _register_resources(mcp: FastMCP) -> None:

    @mcp.resource("docs://schemas/{schema_type}")
    def get_schema(schema_type: str) -> str:
        """JSON schema for a named response type.

        Available schema_types: device_capabilities, device_info, data_overview,
        settings_epoch, settings_timeline, settings_change_entry, settings_changes,
        nightly_row, compliance_fields,
        nightly_summary, event_context, event_row, events_response, capability_entry,
        breath_table_query, breath_table_row, breath_table_bin, breath_table_response,
        window_row, session_coverage_entry, find_windows_response, epoch_spec,
        epoch_distribution, epoch_stats, epoch_rx_violation, compare_epochs_response,
        waveform_channel, waveform_window.
        """
        model = SCHEMA_MODEL_MAP.get(schema_type)
        if model is None:
            available = sorted(SCHEMA_MODEL_MAP.keys())
            raise ToolError(
                f"Unknown schema type {schema_type!r}. Available: {available}"
            )
        return json.dumps(model_to_schema(model), indent=2)

    @mcp.resource("docs://capabilities")
    async def get_capabilities(ctx: Context) -> str:
        """Dataset capabilities — dynamically generated from imported data.

        Lists which waveform channels, event types, and analysis features are
        present in the imported dataset.  Use this to understand what is and is
        not available before calling tools.

        Channels/settings are derived from DB rows (G2 — capability-honest).
        Parser registry is consulted for supported-vendor context only.
        """
        from snore.mcp.tools.overview import get_data_overview
        from snore.parsers.registry import parser_registry

        runtime = _runtime(ctx)

        async with runtime.scope_provider() as db:
            overview = await get_data_overview(db, profile_id=runtime.profile_id)

        supported_parsers = [
            f"{p.manufacturer} ({p.parser_id})" for p in parser_registry.list_parsers()
        ]

        caps = {
            "description": (
                "Available data channels and features in the imported SNORE dataset. "
                "Channels listed as present=false are not available — tool fields "
                "for absent channels return null with a reason. "
                "supported_parsers lists registered device parsers (supplementary context only)."
            ),
            "devices": [
                {
                    "id": d.id,
                    "manufacturer": d.manufacturer,
                    "model": d.model,
                    "date_range": {
                        "start": d.first_session_date.isoformat()
                        if d.first_session_date
                        else None,
                        "end": d.last_session_date.isoformat()
                        if d.last_session_date
                        else None,
                    },
                    "session_count": d.session_count,
                }
                for d in overview.devices
            ],
            "waveform_channels": [
                {"channel": ch, "present": True}
                for ch in overview.available_waveform_channels
            ],
            "event_types": overview.available_event_types,
            "analysis": {
                "run": overview.analysis_run,
                "session_count": overview.analysis_session_count,
                "note": (
                    "Run 'snore analysis run' or re-import with analysis enabled "
                    "to populate analysis-derived fields (RERA index, RDI, breath table)."
                    if not overview.analysis_run
                    else "Analysis results are available. RERA index and RDI fields "
                    "are populated; both are experimental (RERA proxy from SNORE's "
                    "flow-limitation classifier), not device-scored."
                ),
            },
            "provenance": {str(tier): note for tier, note in PROVENANCE_NOTES.items()},
            "supported_parsers": supported_parsers,
        }

        return json.dumps(caps, indent=2, default=str)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _register_tools(mcp: FastMCP) -> None:
    from snore.mcp.tools import (  # noqa: PLC0415
        breath_table,
        changes,
        epochs,
        events,
        overview,
        settings,
        summary,
        waveform,
        windows,
    )

    overview.register(mcp)
    settings.register(mcp)
    changes.register(mcp)
    summary.register(mcp)
    events.register(mcp)
    breath_table.register(mcp)
    windows.register(mcp)
    epochs.register(mcp)
    waveform.register(mcp)
