# AGENTS.md

SNORE (Sleep eNvironment Observation & Respiratory Evaluation) is a CLI tool for analyzing CPAP/APAP therapy data with commands for importing, querying, and analyzing sleep therapy sessions.

## Quick Commands

```bash
# Development workflow
just                # Quick check: sync, type-check, lint-check, format-check
just test           # Run all tests
just check-all      # Full quality check + tests
just check          # Full quality check (Python + UI)
just ci             # CI workflow (same as check + test)
just pre-commit     # Pre-commit: sync, type-check, lint, format (Python + UI)
just lint           # Ruff lint with auto-fix
just format         # Ruff format
just sync           # Install dependencies (uv sync)
just dev-api        # Start REST API dev server (with reload)
just dev-ui         # Start Vue UI dev server
just ui-install     # Install UI pnpm dependencies
just ui-build       # Build UI for production

# CLI (local development - always use `uv run snore`)
uv run snore import <path>                     # Import device data
uv run snore session list                      # List sessions
uv run snore session enable/disable <id>       # Toggle session inclusion
uv run snore analysis run --session-id <id>    # Analyze session
uv run snore analysis list                     # List sessions with analysis status
uv run snore analysis show --date YYYY-MM-DD   # Show analysis results
uv run snore waveform show --date YYYY-MM-DD --time HH:MM:SS  # Visualize waveform
uv run snore waveform compare --session-id <id> --mode aasm   # Compare detection
uv run snore stats                             # Therapy statistics summary
uv run snore stats --period week --trend       # Period breakdown with trends
uv run snore validate --from YYYY-MM-DD --to YYYY-MM-DD  # Batch validation
uv run snore export raw --from YYYY-MM-DD      # Raw file backup
uv run snore export csv --from YYYY-MM-DD      # CSV export
uv run snore export json --from YYYY-MM-DD --to YYYY-MM-DD  # JSON export
uv run snore serve                             # Start REST API (localhost:8000)
uv run snore rx history                        # Prescription settings history
uv run snore rx current                        # Current prescription settings
uv run snore rx compare                        # Compare prescription periods
uv run snore db drop                           # Drop database (with confirmation)
uv run snore db init                           # Initialize database
uv run snore setup --github                    # Install as uv tool from GitHub
uv run snore upgrade                           # Upgrade to latest version
uv run snore --version                         # Show version and check for updates
uv run snore completions install               # Install shell completions
uv run snore logs show/clear/path              # Log management
```

## Project Structure

```
src/snore/
├── cli/                # CLI commands (Click): groups/, commands/, display/, decorators.py
├── constants.py        # Channel IDs, mappings, flow limitation classes, unit constants
├── completions.py      # Shell completion generation and installation
├── exceptions.py       # Domain exceptions (NotFoundError)
├── metrics.py          # Session metric registry (MetricSpec: day aggregation), export keys
├── provenance.py       # Provenance tiers + provenance_field / field_provenance / response_provenance
├── logging_config.py   # Logging configuration with rotation
├── types.py            # Shared type definitions
├── analysis/           # Analysis algorithms
│   ├── calculations.py # Metric calculations
│   ├── rx_tracker.py   # Prescription change tracking
│   ├── utils.py        # Analysis utilities
│   ├── service.py      # AnalysisService orchestrator
│   ├── types.py        # AnalysisResult, AnalysisEvent
│   ├── data/
│   │   └── waveform_loader.py  # Database waveform loading
│   ├── shared/         # Core algorithms
│   │   ├── breath_segmenter.py     # Breath segmentation
│   │   ├── feature_extractors.py   # Waveform feature extraction
│   │   ├── flow_limitation.py      # Flow limitation classification
│   │   ├── pattern_detector.py     # CSR, periodic breathing detection
│   │   ├── pulse_detector.py       # Pulse change detection
│   │   └── types.py                # BreathMetrics, ApneaEvent, etc.
│   └── modes/          # Detection modes
│       ├── config.py   # AASM_CONFIG, AASM_RELAXED_CONFIG, RESMED_CONFIG
│       ├── detector.py # EventDetector (detection core)
│       ├── baseline.py # Baseline flow computation
│       ├── postprocess.py # Validate/dedupe/merge + event matching (tolerance)
│       ├── classification.py # Apnea-type classification + confidence
│       └── types.py    # ModeResult, DetectionModeConfig
├── api/                # FastAPI REST API
│   ├── app.py          # FastAPI application factory
│   ├── deps.py         # Dependency injection (get_db)
│   ├── errors.py       # Exception handlers
│   ├── middleware.py    # Auth, CSRF/body-cap, and rate-limit middleware
│   ├── schemas.py      # API request/response schemas
│   └── routers/        # Route handlers
│       ├── auth/       # Auth package: routes_session, routes_invites, routes_google
│       ├── admin.py, me.py, profiles.py, import_data.py
│       ├── analysis.py, days.py, devices.py, events.py
│       ├── rx.py, sessions.py, stats.py, waveforms.py
├── bootstrap/          # Installation and updates
│   ├── core.py         # Shared constants + uv subprocess runner
│   ├── installer.py    # Global uv tool installation
│   ├── updater.py      # Version upgrade logic
│   └── version.py      # Version management
├── database/           # SQLAlchemy ORM layer
│   ├── models.py       # DB models (10 tables)
│   ├── session.py      # session_scope() context manager
│   ├── day_manager.py  # Day splitting and aggregation
│   ├── importers.py    # Session import pipeline
│   ├── types.py        # Database type extensions
│   └── migrations/     # Alembic migration infrastructure
│       └── env.py
├── parsers/            # Device parsers
│   ├── base.py         # DeviceParser abstract class
│   ├── registry.py     # parser_registry singleton
│   ├── register_all.py # register_all_parsers() — call at startup
│   ├── resmed_edf.py   # ResMed EDF+ parser
│   ├── unified.py      # UnifiedSession, WaveformData, RespiratoryEvent
│   ├── event_labels.py # Annotation label → RespiratoryEventType mapping
│   ├── compression.py  # Data compression utilities
│   ├── discovery.py    # Data source discovery
│   ├── oscar_mappings.py # OSCAR channel mappings
│   ├── resmed_file_index.py # ResMed file indexing
│   ├── types.py        # Parser type definitions
│   └── formats/
│       ├── edf.py      # Generic EDF/EDF+ reader
│       └── types.py    # Format type definitions
├── mcp/                # MCP server (third presentation layer, peer of CLI and api/)
│   ├── server.py       # make_server, SNORERuntime protocol, lifespan, tool/resource registration
│   ├── auth.py         # OAuth 2.1 GoogleProvider integration (HTTP transport only)
│   ├── errors.py       # ValidationError (mapped to ToolError at the boundary)
│   ├── profiles.py     # Clinical profile loading (shapes instructions text only)
│   ├── schemas.py      # MCP response Pydantic models (SCHEMA_MODEL_MAP for docs://)
│   ├── validation.py   # parse_date, parse_date_range, validate_* helpers
│   └── tools/
│       ├── _helpers.py         # _str_or_none shared helper
│       ├── _scaffold.py        # _scope_and_run, tool_error_boundary, _check_response_size
│       ├── _service_errors.py  # MAPPED_SERVICE_ERRORS, raise_mapped_service_error
│       ├── _capabilities.py    # build_device_capabilities, get_device_id_for_session
│       ├── _coverage.py        # map_session_coverage
│       ├── overview.py         # get_data_overview tool
│       ├── settings.py         # get_settings_timeline tool
│       ├── changes.py          # get_settings_changes tool
│       ├── summary.py          # get_nightly_summary tool
│       ├── events.py           # get_events tool
│       ├── breath_table.py     # get_breath_table tool
│       ├── windows.py          # find_windows tool
│       ├── epochs.py           # compare_epochs tool
│       └── waveform.py         # get_waveform + render_window tools
├── services/           # Business logic layer (between CLI/API and database)
│   ├── schemas.py      # Service response schemas (Pydantic)
│   ├── analysis_facade.py   # Analysis orchestration
│   ├── backup_service.py    # Raw file backup
│   ├── database_service.py  # DB stats + device listing (system/metadata)
│   ├── day_service.py       # Day aggregation
│   ├── event_service.py     # Event queries + matching
│   ├── export_service.py    # Data export (CSV, JSON)
│   ├── lttb.py              # LTTB downsampling algorithm
│   ├── session_service.py   # Session management
│   ├── stats_service.py     # Statistics calculations
│   └── waveform_service.py  # Waveform data access (single entry point)
├── validation/         # Data validation
│   ├── batch.py        # BatchValidator
│   └── report.py       # ValidationReport
└── waveform/           # Waveform visualization
    ├── inspector.py    # Data loading and inspection
    └── renderer.py     # ASCII/plotext terminal rendering
ui/                     # Vue 3 + TypeScript + Tailwind v4 + shadcn-vue frontend
├── src/
│   ├── api/            # API client (createApiEndpoint wrappers, 8 modules)
│   ├── components/     # Reusable UI components (10+)
│   ├── composables/    # useApiLoad and other composition functions
│   ├── utils/          # formatting.ts (shared date/time formatters)
│   ├── views/          # Page views (7 views)
│   ├── router/         # Vue Router configuration
│   └── types/          # index.ts re-exports generated.ts (OpenAPI codegen)
docs/
├── apnea_detection_reference.md  # Algorithm guide with Vancouver-style citations
├── manufacturers/resmed.md        # ResMed-specific documentation
├── roadmap.md                     # Project roadmap
└── references/                    # Research papers, clinical guidelines (9 PDFs)
    ├── README.md                  # Index with full citations
    └── PMC*.pdf                   # Open-access research papers
tests/
├── conftest.py         # Main fixtures
├── fixtures/           # Test data (recorded sessions, device data)
├── helpers/            # synthetic_data.py, validation_helpers.py
├── unit/               # Unit tests (auto-marked)
└── integration/        # Integration tests (auto-marked)
    └── test_api/       # API integration tests
```

## OSCAR Relationship

**OSCAR** (Open Source CPAP Analysis Reporter) is the upstream OSS project (Qt/C++ desktop GUI, 18+ manufacturers).

**Import paths:**
- **Direct (ResMed):** SD Card → SNORE → Database
- **Via OSCAR (all manufacturers):** SD Card → OSCAR → OSCAR Profiles dir → SNORE → Database

**What SNORE borrows from OSCAR:**
- Channel ID system (0x1000=Pressure, 0x1200=Flow, etc.) in `constants.py`
- Binary format parsers for `.000/.001` session files

**OSCAR parsers:** `parsers/oscar_device.py` (main), `oscar_summary.py`, `oscar_events.py`, `qdatastream.py`

**Golden reference for ResMed STR decoding:** local OSCAR clone at `/home/will/Development/OSCAR-code`,
file `oscar/SleepLib/loader_plugins/resmed_loader.cpp`. Comments in `resmed_edf.py` cite it as
`OSCAR :NNNN` (line numbers in that file). When changing STR decode logic, verify against the OSCAR
source first and keep the citations current.

## Tech Stack

- Python 3.13+ with UV package manager
- SQLAlchemy 2.0 ORM with SQLite (~/.snore/snore.db)
- Click CLI framework
- Pydantic for validation
- FastAPI + uvicorn (REST API)
- Vue 3 + TypeScript + Tailwind CSS v4 + shadcn-vue/reka-ui (frontend)
- uPlot (UI charts)
- Rich (terminal formatting)
- Plotext (terminal charts)
- scipy (signal processing)
- Alembic (database migrations)
- pytest with coverage

## Production Deployment

Production runs as a Docker stack on a homelab host, provisioned from the private [`wpfleger96/homelabconfigs`](https://github.com/wpfleger96/homelabconfigs) repo (Ansible playbook `ansible/playbooks/install_and_configure_snore.yml`, compose file under `ansible/playbooks/files/snore/`). Host details, env templates, and public URL live there, not here.

- **Deploys are automatic.** Merging to `main` triggers `.github/workflows/docker.yml`, which pushes `ghcr.io/wpfleger96/snore:latest`; Watchtower on the host polls every 5 minutes and swaps the container (a pre-update lifecycle hook defers the swap while an import/analysis/reset is in flight). There is no manual deploy step for app code. Env/infra changes deploy via `just config snore` from homelabconfigs.
- **Data layout:** the stack lives in `/opt/snore` on the host; `/opt/snore/data` mounts to `/data` in the container, and `HOME=/data`, so the production DB is `/data/.snore/snore.db`.
- **Running the CLI in production:** SSH to the host, then `docker exec snore /app/.venv/bin/snore <command>` — the venv is not on the container's `PATH`, so the absolute path is required (add `-it` for commands that prompt). Production is multiuser — most commands need `--user <email>` (and `--profile <name>` when the user has several).
- **Re-analysis after algorithm version bumps is manual.** Bumping any `*_ALGO_VERSION` constant (`src/snore/analysis/shared/versioning.py`) marks every stored analysis row `STALE_VERSION` on the next deploy; aggregates null stale sessions rather than serving old numbers, but nothing re-analyzes automatically. Re-run per profile with an open-ended range — `snore analysis run --from 2000-01-01 --user <email> --profile <name>` — which processes every session in range (no already-analyzed skip) and stores new version rows. Optionally drop the superseded rows afterward: `snore analysis delete --all --stale-versions`, then `snore db vacuum`.
- **Day rows are not recomputed on deploy.** After any change to day aggregation, backfill historical Day rows with `docker exec -it snore /app/.venv/bin/snore db recompute-days`. It re-derives every Day across all users, profiles, and devices from stored session statistics (no reparse), so it takes no `--user`/`--profile`; it prompts for confirmation (`--yes` skips it) and is safe to re-run if interrupted. Migration 018 backfills existing days as derived (old values copied into `*_computed`), and `recompute-days` alone cannot promote them to the device-reported AHI headline: the trust rule needs the per-session `statistics.usage_hours_device` (STR daily mask-on time), which is NULL on sessions imported before 018. Only a forced re-parse fills it: `docker exec -it snore /app/.venv/bin/snore import --force --all --no-backup --user <email> --profile <name> /data/.snore/raw/<profile_id>` (the profile's raw archive). A plain import or web upload/rescan skips already-imported sessions. `--force` deletes and recreates each session it finds (statistics, events, waveforms, settings, and cascaded analysis/breath rows; post-import analysis re-runs unless `--no-analyze`), keeps disabled sessions disabled (so does any import whose session replaces only disabled overlapping rows; replacing a disabled/enabled mix imports enabled and logs a warning), and re-aggregates the affected days itself, so no `recompute-days` is needed afterwards.

## Key Patterns

**Parser Plugin Architecture:**
```python
# Inherit DeviceParser, implement 4 methods, register with parser_registry
class MyParser(DeviceParser):
    def get_metadata(self) -> ParserMetadata: ...
    def detect(self, path: Path) -> ParserDetectionResult: ...
    def get_device_info(self, path: Path) -> DeviceInfo: ...
    def parse_sessions(self, path, ...) -> Iterator[UnifiedSession]: ...
parser_registry.register(MyParser)
```

**Database Context Manager:**
```python
with session_scope() as session:
    # Auto-commit on success, rollback on exception
```
In CLI commands, use the combined helper from `cli/decorators.py` instead of
calling `init_db()` + `session_scope()` manually:
```python
with db_session(db) as session:
    ...
```

**Service Layer Pattern:**
Services sit between CLI/API and database. Constructor injection with SQLAlchemy session, typed Pydantic returns:
```python
class StatsService:
    def __init__(self, db_session: Session): ...
    def get_summary(self, days: int | None) -> StatsSummary: ...
```
- Queries use the SQLAlchemy ORM (no raw/f-string SQL).
- Missing resources raise `NotFoundError` (`snore/exceptions.py`); the API maps it
  to 404 via the registered handler — routers should NOT re-check for `None`.

**Crossing schema boundaries:** the unified parser models, services schemas, and ORM
models align by field name — do NOT write field-by-field mappings:
```python
models.Statistics(session_id=sid, **stats.model_dump())  # Pydantic → ORM
SessionStatistics.model_validate(orm_row)  # ORM → Pydantic (from_attributes)
```
`tests/unit/test_schema_alignment.py` enforces the alignment; if you add a field,
add it to both sides.

**API Router Pattern:**
FastAPI routers get services via the `service_dep` factory and dates via
`DateRangeParams` (`api/deps.py`):
```python
router = APIRouter(prefix="/sessions", tags=["sessions"])
SessionServiceDep = Annotated[SessionService, Depends(service_dep(SessionService))]

@router.get("/")
def list_sessions(svc: SessionServiceDep, dates: DateRangeParams = Depends()) -> ...:
    svc.list_sessions(from_date=dates.start_datetime, ...)
```

**CLI display:** use the helpers in `cli/display/` (`print_table`, `print_header`,
`print_kv`, ...) instead of hand-rolled f-string alignment. Repeated Click options
get composite decorators in `cli/decorators.py` (e.g. `device_option`).

**Parser helpers:** ResMed EDF signal reading goes through
`ResmedEDFParser._read_waveform` (valid-range masking, unit conversion, stats);
basic min/max/mean via `extract_basic_stats`; unit strings come from the `UNIT_*`
constants in `constants.py`; annotation labels map via `parsers/event_labels.py`.

**ResMed STR settings decoding** (`parsers/resmed_edf.py`, verified against OSCAR — see OSCAR Relationship):
- Series detection is ProductCode-only: `_detect_series11` reads `Identification.json`
  (`ProductCode >= 39000` → Series 11; values can be JSON floats, hence `int(float(str(code)))`)
  with `Identification.tgt` key=value fallback for S9. The model-name regex `_is_eleven_series`
  has no production callers — do not reintroduce model-string series checks.
- Mode decode is two-step: S11 raw → S10 basis via `_S11_MODE_TO_S10`, then the shared `_MODE_MAP`.
  Unknown modes warn + skip the record (S11 raw 0/5; S10 raw 10 "PAC").
- S11 emits enum-valued signals one higher than S10 ("−1 family"): normalize via `_norm()`
  (subtracts 1 on S11; NaN → `None` so booleans read unknown, not False). All downstream lookup
  maps are keyed on the S10 basis — never add an S11-keyed map.
- Signal rosters are per-mode maps merged into `ALL_STR_SIGNAL_MAPS`: APAP `S.A.*`/`S.AFH.*`,
  S11 vAuto `S.VA.*` / S11 bilevel `S.S.*`, S10 bilevel pressures `S.BL.*` with timing/comfort from
  bare `S.*` signals (`S.Cycle`, `S.Trigger`, `S.TiMax`, `S.TiMin`, `S.RiseEnable`, `S.EasyBreathe`),
  ASV `S.AV.*`/`S.AA.*`, iVAPS `S.i.*`. Add new signals to the per-mode map, not the merged map.
- `STR.edf` pre-allocates ~a year of daily record slots (unused days are all-NaN rows), and the
  device periodically rolls it into `STR_Backup/STR-YYYYMMDD.edf` snapshots (~183-day cadence
  observed). `_load_str_caches` merges primary + backups longer-file-wins per start-date, probing
  EDF headers before full loads (`_preload_str_file`) and falling back to the next candidate when
  a winner is corrupt.

**Event matching tolerance** is single-sourced:
`EVENT_MATCH_TOLERANCE_SECONDS` in `analysis/modes/postprocess.py`. Never hardcode 5.0.

**Provenance tagging:** every reported metric carries one of three tiers (`snore/provenance.py`):
- **Device**: verbatim device data (device-scored events and their counts, waveforms, STR values, settings).
- **Derived**: deterministic math on device data (index recounts over SNORE mask-on hours, percentiles, usage, roll-ups).
- **Experimental**: SNORE's own detection/classification heuristics (FL classes, RERA proxy, breath
  segmentation features, programmatic event detection, SNORE-vs-device agreement scores).

Aggregates inherit the weakest input: anything built on an experimental input stays Experimental;
aggregates of Device/Derived inputs are Derived. Exception: a sum of device-scored event counts
(across sessions, days, or a validation run) stays Device; rates, means, percentiles and recounts
are Derived. A field with a preferred source and a fallback is tagged with the preferred source's
tier and its description names the fallback, in both directions: pressure/EPAP/leak percentiles
recomputed from the waveform when present are Derived (falling back to the device's summary value);
`respiratory_rate_mean/max` and tidal volume / minute ventilation mean/max prefer the device STR
value and are Device (falling back to the OSCAR session summary).
Day-level `ahi/oai/cai/hi` (`Day` rows, `DayListItem`/`DayDetail`, MCP `NightlyRow`) are
source-dependent headlines: the device-reported daily value when trusted, otherwise SNORE's recount;
`index_source` (`IndexSource.DEVICE`/`DERIVED`) says which. They are tagged Device with
`source_field="index_source"` — copy that pattern for any new source-dependent field. Aggregates over
them (period, Rx, trends, records) stay Derived and can mix device-reported and recounted days. The
recount, when computable, is in `*_computed` (`DayListItem` carries only `ahi_computed`;
`DayDetail` and MCP `NightlyRow` carry all four).

The Pydantic schemas are the single source of tiers. Every metric field in MCP/REST schemas uses
`provenance_field(Provenance.X, "...")` (adds `x-provenance` + a `[TIER]` description prefix);
`SessionStatistics` (`services/schemas.py`) is canonical for the Statistics columns. Read a tier
back with `field_provenance(<Model>, "<field>")` and the prefix-free description with
`field_description(<Model>, "<field>")`, from the response model that carries the value. Any value
or column label backed by a tagged field must read its tier this way (or via `mark_field`);
hardcoding `Provenance.X` is allowed only for section/report titles with no single backing field.
MCP responses get a top-level `provenance` block from `response_provenance` (`snore/provenance.py`).
`tests/unit/test_provenance_tags.py` enforces tagging on every numeric/boolean (or list-of-number)
response field, with bookkeeping allowlists: `_NON_METRIC_FIELDS` for names that are never metrics
(ids, positions, pagination) and `_NON_METRIC_MODEL_FIELDS` for coverage counts keyed by
`(model, field)`; it also requires a tag on fields holding a model whose tier depends on the parent.

CLI entry points (`cli/display`): `mark_field(label, Model, "field")` returns the label with the
field's tier marker; `mark_provenance(label, tier)` does the same for a tier in hand;
`print_kv(..., provenance=...)` and `Column(header, width, provenance)` for `print_table` mark keys
and headers. The legend line prints automatically only for commands wrapped in
`profile_scoped_command` (`cli/decorators.py`), including when the command exits with an error
after printing markers; elsewhere markers appear with no legend. Exports carry a JSON `provenance`
header and a CSV `columns.csv` sidecar.

The web UI does not mark individual fields. A section or card whose content is an Experimental
metric gets one `<ExperimentalBanner>` (pass a short `body` on small cards; `ValidationPanelShell`
takes `experimental`/`experimental-note`); Device and Derived data stay unflagged. Day-level AHI
names its source (`index_source`) in text: device-reported or SNORE recount.

**UI:** API types are generated — run `just ui-generate-types` after changing API
schemas (`ui/src/types/generated.ts`; `types/index.ts` re-exports them). New API
wrappers use `createApiEndpoint` in `ui/src/api/client.ts`; plain view loaders use
the `useApiLoad` composable; date/time formatting comes from `ui/src/utils/formatting.ts`.

**Analysis Architecture:** Direct orchestration in `service.py`:
- BreathSegmenter → feature extraction → FlowLimitationClassifier → ComplexPatternDetector → PulseChangeDetector → EventDetector
- Event detection via modes with `DetectionModeConfig`:
  - `aasm` - AASM Scoring Manual v2.6 compliant (default, time-based baseline 120s, AASM_3PCT hypopneas with 3% SpO2 drop, RERA enabled, flow-only fallback)
  - `aasm_relaxed` - AASM with relaxed thresholds for machine matching (breath-based baseline 30 breaths, 85% validation, AASM_3PCT hypopneas, RERA enabled)
  - `resmed` - ResMed device algorithm matching (time-based baseline 120s, gap + low-flow detection, 50% apnea threshold, FLOW_ONLY hypopneas at 20% threshold, RERA enabled)
- Detected events: Obstructive Apnea (OA), Central Apnea (CA), Mixed Apnea (MA), Hypopnea (mode-dependent), RERA (flow-based without EEG)
- All apneas include `classification_confidence` (OA/CA/MA from flow-only is approximation)
- RDI calculation includes RERAs (AHI + RERAs/hour)
- All types use Pydantic models (validation, serialization)

**Unified Data Model:** All device data converts to `UnifiedSession` → `WaveformData` → `RespiratoryEvent`

## Code Style

- Type hints: `str | None` (not Optional), `list[str]` (not List), avoid `Any` types
- Imports: stdlib, third-party, then `snore.` absolute imports
- Naming: snake_case functions, PascalCase classes, UPPER_SNAKE constants
- All data types use Pydantic models (no dataclasses). One exception: display-only value tuples in
  `cli/display` (e.g. `Column`) may be `NamedTuple`.

## Testing

**Standard workflow:**
```bash
just test                # All tests (standard pytest invocation)
just check-all           # Quality checks + tests
```

**Advanced pytest options (when needed):**
```bash
# Specific test subsets
uv run pytest tests/unit/                    # Unit only
uv run pytest tests/integration/             # Integration only
uv run pytest tests/integration/test_api/    # API integration tests
uv run pytest -m recorded                    # Tests using real device data
uv run pytest tests/unit/test_file.py -v    # Single file verbose
uv run pytest tests/ --cov=snore             # With coverage
```

Markers: `unit`, `integration`, `parser`, `recorded`, `real_data`, `slow`

Key fixtures: `db_session`, `test_device`, `test_session_factory`, `recorded_session("YYYYMMDD")`

## PR Screenshots

PR screenshots are **ad hoc**: capture only what demonstrates THIS PR's UI changes — one
focused shot per change, each with a caption. Never post the regression battery
(`ui/screenshot.spec.ts` via `pnpm run test:screenshot`) to a PR; that suite is for local
regression eyeballing only.

**Capture:** the UI renders against a route-mocked API (no backend needed). Write a
throwaway Playwright spec patterned on `ui/screenshot.spec.ts` — import the fixtures
from `ui/tests/fixtures/api-fixtures`, spread-and-override them inline so every new
field/state in your diff is exercised, and use `locator.screenshot()` to crop to the
relevant section when a full page would bury the change. Gotcha:
`ui/playwright.config.ts` pins `testMatch: 'screenshot.spec.ts'`, so pair your spec
with a throwaway config (same `webServer`/`baseURL`/viewport, `testMatch` pointing at
your spec):

```bash
just web-build
cd ui && pnpm exec playwright test --config=pr-shots.config.ts
```

Delete the throwaway spec/config before committing. Dark mode: click the sidebar
toggle (`page.getByText('Dark Mode').click()`); include dark variants only where the
change is theme-sensitive. Name files with numeric prefixes to control order
(`01-devices-overview.png`).

**Post:** `scripts/post-screenshots.sh` hosts PNGs on a per-developer branch
(`agent-screenshots/<github-username>`) and comments on the PR with immutable
commit-SHA image URLs:

```bash
bash scripts/post-screenshots.sh <pr-number> <png-dir> body.md
```

`body.md` uses `{{filename}}` placeholders (without `.png`); images not referenced by
a placeholder are appended at the end. One section per change:

```markdown
## Screenshots

### Settings history
Consecutive-session diff; added keys show "(new)", removed keys "(removed)".

{{03-settings-history}}
```

**Re-posting:** the script appends a new comment — it never edits or deletes old ones.
After re-posting, delete the superseded comment so reviewers only see the current set:

```bash
gh pr view <pr> --json comments --jq '.comments[] | select(.body | test("pr-<pr>--")) | {id, url}'
gh api -X DELETE repos/<owner>/<repo>/issues/comments/<stale-comment-id>
```

## Common Gotchas

1. **OSCAR day-splitting logic:** Sessions before noon belong to previous day (e.g., 01:50 AM Dec 8 = Dec 7's sleep). Use `Day.date` for display/queries, not `session.start_time.date()`. See `DayManager.get_day_for_session()` in `src/snore/database/day_manager.py` and the `session.day.date` fallback in `src/snore/services/analysis_facade.py`
2. **Refresh after relationship changes:** `db_session.refresh(session)` after adding statistics
3. **Integration test isolation:** Use `reset_database_state()` autouse fixture pattern
4. **WAL cleanup:** Temp databases need `-wal` and `-shm` file cleanup
5. **Profile management removed:** Profiles are now optional (device-centric model). Days link directly to devices via `device_id`, not `profile_id`
6. **Type safety:** Use proper types (`list[BreathMetrics]` not `list[Any]`) - mypy strict mode enabled
7. **Pydantic validation:** Use `model_construct()` to bypass validation when testing invalid data
8. **Local development vs installed tool** - **CRITICAL**: Always use `uv run snore` when developing locally:
   - **Local dev (from repo)**: `uv run snore <command>` → runs YOUR local code changes directly
   - **Installed tool (any directory)**: `snore <command>` → runs installed version from `~/.local/share/uv/tools/`
   - Running `snore` without `uv run` will NOT reflect your local changes
   - **NEVER use editable install** (`uv pip install -e .`) - risks conflicts with installed version, unnecessary complexity
9. **RERA confidence capped at 0.7:** Flow-based RERA detection without EEG cannot exceed 0.7 confidence (true RERAs require EEG arousal). Detection uses ≥2 flow-limited breaths + recovery breath ≥50% amplitude increase.
10. **Apnea classification confidence:** OA vs CA vs MA classification from flow-only data is approximation (true classification needs thoracic/abdominal effort bands). All apneas include `classification_confidence` field (0-1) based on effort score distinctiveness.
11. **SpO2/Flow timestamp alignment:** `_detect_hypopneas()` validates SpO2 and flow signal lengths match before indexing. Mismatch logs warning and skips desaturation check to prevent IndexError with external oximeters at different sample rates.
12. **Documentation citations:** Use Vancouver-style numbered citations [1], [2] in `docs/apnea_detection_reference.md`. Add PDF to `docs/references/` and update both inline citation and References section. See existing format: author list, journal, year, volume, pages, DOI, PMCID, local path, URL.
13. **Parser parity verification:** import real data into a scratch DB via the global `--db` option (`uv run snore import --all --no-backup --db <path> "<folder>"`), dump `RxTracker().get_changes()` rows, and diff against a known-good baseline before/after parser changes. Put large scratch DBs on real disk — `/tmp` is a 16G tmpfs.
14. **Never add unlayered global CSS resets in `App.vue`** (e.g. a bare `* { margin: 0 }`): unlayered styles outrank every Tailwind v4 layered utility and silently break component styling. The same rule cuts the other way: a Tailwind utility (`max-md:*`, `md:*`, `w-*`, `h-*`) can NEVER override a property that a Vue scoped `<style>` block or `layout.css` already sets — scoped styles compile to unlayered CSS and always win. Mobile overrides of scoped-set properties must live in a `@media (max-width: 767.98px)` block inside the same scoped style. shadcn `data-[...]:` variant utilities also out-specify plain utilities.
15. **Fresh worktree setup:** run `uv sync` and `just web-install` once per new worktree — the managed pre-commit hook (`.hooks/pre-commit` → `just pre-commit`) auto-fixes and re-stages lint/format issues including the web legs, and `just check-all` runs the check-mode web variants.
16. **Two-phase import / background worker location:** `POST /import` accepts uploads and returns a `job_id` immediately; the actual import runs in a background thread driven by the persistent worker in `api/import_worker.py` (not inside the router). The worker emits a non-terminal `phase_complete` SSE event after the import commits, then enqueues a downstream `AnalysisJob`, then terminates the import job. Every terminal payload carries `import_committed=True` and the full import result if the import phase committed to the DB — this durability guarantee applies to success, analysis-failure, and cancellation payloads alike. The worker is passed to `start_import_worker()` during app lifespan startup (`app.py`); tests import it from `snore.api.import_worker`.
17. **Mobile responsiveness conventions:** single breakpoint — mobile is `< 768px` (`@media (max-width: 767.98px)` in CSS, matching Tailwind's `max-md`; the JS side is the `useIsMobile()` singleton in `ui/src/composables/useIsMobile.ts`). Use CSS for styling differences; use `useIsMobile` + `v-if`/`v-else` only when the rendered structure or props must differ (e.g. table ↔ card list). Wide tables get a mobile card branch built from the shared unlayered classes in `layout.css` (`.card-list`, `.data-card`, `.data-card-header`, `.data-card-row`, `.data-card-label`, `.data-card-value`, `.data-card-actions`) — do not invent per-view header/action classes; extract a per-view card child component (see `SessionCard.vue`, `UserCard.vue`) when the branch grows interactive state. Touch targets use `var(--tap-target)` (44px) from `layout.css`, never a hardcoded size. In unit tests, jsdom's `matchMedia` mock (`ui/src/tests/matchMedia.ts`) defaults to desktop; flip a test to the mobile branch with `setMediaMatches(true)`.
18. **FL confidence gate and the RERA fallback:** FL classifications carry a confidence gate. Fallback flatness-triage guesses sit at exactly `FL_DEFAULT_CONFIDENCE` and are excluded from nightly `fl_class_ge4_pct` and from `compare_epochs`' `flow_class_distribution` (they land in `flow_class_distribution_fallback` instead), but the RERA proxy's FL-run scan deliberately includes them — removing them zeroes the proxy (probed 2026-08-23). FL/RERA metrics are experimental relative-trend instruments, not validated absolute measurements; read `docs/fl-rera-validation.md` before touching these paths.

## MCP Development

### Adding or modifying an MCP tool

Each tool lives in `src/snore/mcp/tools/<name>.py` and follows a two-part structure:

1. A module-level async function that accepts an `AsyncSession` and returns a Pydantic model — this is what tests call directly.
2. A `register(mcp: FastMCP) -> None` function at the bottom of the module that defines the `@mcp.tool()` closure and wires it to the common scaffold.

The common scaffold for the eight standard-pattern tools lives in `_scope_and_run` (`tools/_scaffold.py`, re-exported from `server.py`): open scope → `await impl(db, profile_id=..., **kwargs)` → `model_dump(mode="json")` → add a top-level `provenance` block (`response_provenance` in `snore/provenance.py`, cache warmed in the lifespan) → `_check_response_size`. Tools that expose experimental metrics wrap their description with `_with_experimental_disclaimer` (`tools/_scaffold.py`). Tools with non-standard return paths (`get_waveform`, `render_window`) handle the scope themselves inside their `register` closures.

There is deliberately no per-event-type tool: event-type-specific context (e.g. MV slope/stability/PS before a central apnea) belongs in `get_events`' per-event context block, and night-level metrics (e.g. `periodic_breathing_pct`) belong in `get_nightly_summary` rows.

### Service error mapping seam

`tools/_service_errors.py` maps `BreathService` exceptions to `ValidationError` (which the server's `tool_error_boundary` converts to `ToolError`):

- `DeviceNotOwnedError` → "device_id=X is not available in this session"
- `DeviceAmbiguityError` → "Multiple devices have sessions on DATE: ... Devices: device_id=X (serial=Y), ..."
- `MultiSessionAmbiguityError` → "Multiple sessions on DATE for device_id=X: ..."
- `NoSessionsInRangeError` → "No therapy data found for date/range ..."
- `OperationalError("no such table")` → "Breath-level data tables are missing ..."

Any tool that calls `BreathService` should use `except MAPPED_SERVICE_ERRORS as exc: raise_mapped_service_error(exc)`. Do not write per-tool except branches for these exceptions.

### `tool_error_boundary` contract

`tool_error_boundary` in `tools/_scaffold.py` (re-exported from `server.py`) wraps every registered tool closure. It converts:
- `ToolError` → passes through unchanged
- `PydanticValidationError` → `ToolError` with cleaned field-path messages
- `ValidationError` or `ValueError` → `ToolError(str(exc))`
- HTTP errors (upstream `response.status_code`) → `ToolError("HTTP N from upstream service")`
- All other exceptions → logged + `ToolError("An unexpected error occurred.")`

Internal detail never reaches the client. Never raise `ToolError` with raw exception text that might include profile IDs, file paths, or SQL fragments.

### Running MCP tests

```bash
# Unit tests (in-memory, fast — run while iterating)
uv run pytest tests/unit/test_mcp_server.py tests/unit/test_mcp_tools_roundtrip.py tests/unit/test_mcp_validation.py

# Per-tool unit tests
uv run pytest tests/unit/test_mcp_breath_table.py tests/unit/test_mcp_compare_epochs.py tests/unit/test_mcp_find_windows.py

# Integration tests (real SQLite in-memory DB)
uv run pytest tests/integration/ -k mcp
```

Integration test seed helpers (`_make_profile`, `_make_device`, `_make_day_session`, `_make_analysis_result`) live in `tests/integration/conftest.py` as plain functions — import them from there, do not copy them into new test files.

## Key Files by Task

| Task | Files |
|------|-------|
| Add MCP tool | `src/snore/mcp/tools/<name>.py` (impl + register); `src/snore/mcp/schemas.py` (response schema); `tests/unit/test_mcp_<name>.py` + `tests/integration/test_mcp_<name>.py` |
| Add CLI command | `src/snore/cli/groups/` (subcommand group) or `cli/commands/` (standalone); helpers in `cli/decorators.py`, `cli/display/` |
| Add device parser | `src/snore/parsers/base.py`, `src/snore/parsers/register_all.py`, create new parser file |
| Modify ResMed STR settings decoding | `src/snore/parsers/resmed_edf.py` (per-mode signal maps, `_parse_str_settings`), `tests/unit/test_resmed_str_converter.py`, `tests/unit/test_resmed_str_backup.py` |
| Prescription (rx) history | `src/snore/analysis/rx_tracker.py`, `src/snore/cli/groups/rx.py`, UI labels in `ui/src/utils/deviceSettings.ts` |
| Add analysis algorithm | `src/snore/analysis/shared/` (breath/feature algorithms) or `modes/` (event detection) |
| Add detection mode | `src/snore/analysis/modes/config.py` (add `DetectionModeConfig`), update `detector.py` |
| Modify event detection | `src/snore/analysis/modes/detector.py` (detection core); `baseline.py`/`postprocess.py`/`classification.py` (supporting algorithms) |
| Add event type | `src/snore/analysis/shared/types.py` (event models), update `detector.py`, add to `EventTimeline` |
| Tune detection thresholds | `src/snore/analysis/modes/config.py` (DetectionModeConfig fields), validate with `validate_against_machine_events()` |
| Modify data models | `src/snore/parsers/unified.py` (data), `database/models.py` (ORM), use Pydantic; keep `test_schema_alignment.py` passing |
| Add waveform visualization | `src/snore/waveform/renderer.py` (ASCII/plotext), `inspector.py` (data loading) |
| Add API endpoint | `src/snore/api/routers/`, `api/schemas.py`, `api/deps.py`; then `just ui-generate-types` |
| Add/modify service | `src/snore/services/`, `services/schemas.py` |
| Frontend development | `ui/src/views/`, `ui/src/components/`, `ui/src/api/`, `ui/src/composables/`, `ui/src/utils/` |
| Regenerate UI API types | `just ui-generate-types` (`scripts/export_openapi.py` → `ui/src/types/generated.ts`) |
| Export functionality | `src/snore/services/export_service.py`, `services/backup_service.py` |
| Data validation | `src/snore/validation/batch.py`, `validation/report.py` |
| Add test fixture | `tests/conftest.py`, `tests/helpers/` |
| Modify channel IDs | `src/snore/constants.py` (must align with OSCAR's schema.h) |
| Update algorithm documentation | `docs/apnea_detection_reference.md` (add inline citations [#]) |
| Add research reference | `docs/references/` (add PDF + update README.md index) |
