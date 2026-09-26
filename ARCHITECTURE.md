# SNORE Architecture

Technical documentation for the SNORE system architecture, components, and design decisions.

---

## System Overview

### Modular, Parser-Agnostic Platform

```
┌─────────────────────────────────────────────┐
│        Vue 3 Frontend (ui/)                │
│  Dashboard, Session Explorer, Waveforms     │
│  PrimeVue components + uPlot charts         │
├──────────────┬──────────────┬──────────────┤
│ FastAPI REST │   CLI (cli/) │  MCP Server  │
│ API (api/)   │  Click cmds  │  (mcp/)      │
│ 18 routers   │  snore cmds  │  11 tools    │
├──────────────┴──────────────┴──────────────┤
│        Service Layer (services/)            │
│  19 services: business logic between        │
│  CLI/API and database                       │
├─────────────────────────────────────────────┤
│        Analysis Layer (Parser Agnostic)     │
│  Breath segmentation, event detection,      │
│  flow limitation, pulse change detection    │
├─────────────────────────────────────────────┤
│         SQLite Database ✅                  │
│  Universal schema, direct BLOB storage      │
│  create_all schema; Alembic kept (pre-1.0)  │
├─────────────────────────────────────────────┤
│         CLI Import Tool ✅                  │
│  snore import (auto-detection)              │
│  Raw backup, parallel parsing               │
├─────────────────────────────────────────────┤
│         Unified Data Model                  │
│  UnifiedSession, WaveformData, etc.         │
│  All parsers output to this format          │
├─────────────────────────────────────────────┤
│         Parser Registry & Detection         │
│  Auto-detects device from file structure    │
│  Confidence-based selection                 │
├─────────────────────────────────────────────┤
│           Device Parser Plugins             │
│ ┌──────────┬──────────┬──────────┬────────┐│
│ │ ResMed   │ OSCAR    │ Philips  │ Future ││
│ │ EDF+     │ Binary   │ (TODO)   │        ││
│ │   ✅     │  ✅      │          │        ││
│ └──────────┴──────────┴──────────┴────────┘│
└─────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Analysis layer never knows about parser formats
2. **Extensibility**: Add new device support without touching existing code
3. **Single Source of Truth**: SQLite database stores unified format
4. **Auto-Detection**: Users just point to data, it "just works"
5. **Testability**: Each parser tested independently with real data

---

## Core Components

### Unified Data Model

**File:** `src/snore/parsers/unified.py`

All parsers convert to these universal Pydantic structures:

**UnifiedSession**
- Device-agnostic session container
- Start/end times, duration, mode
- Waveforms dict (by type)
- Events list
- Statistics
- Quality notes

**WaveformData**
- Time-series data (Flow, Pressure, Leak, SpO2, Pulse)
- Timestamps as numpy array (seconds offset from session start)
- Values as numpy float32 array
- Sample rate, unit, min/max/mean statistics

**RespiratoryEvent**
- Event types: OA, CA, H, UA, RERA, FL, etc.
- Start time, duration
- Optional annotations (from EVE files)

**DeviceInfo**
- Manufacturer, model, serial number
- Firmware version
- Metadata

**SessionStatistics**
- Event counts (OA, CA, H, etc.)
- Indices (AHI, OAI, CAI, HI, REI)
- Pressure stats (min, max, median, p95)
- Leak stats
- Respiratory stats
- SpO2 stats (min, max, mean, time below 90%)
- Pulse stats

### Parser Infrastructure

**File:** `src/snore/parsers/base.py`

**DeviceParser (Abstract Base Class)**
```python
class DeviceParser(ABC):
    @abstractmethod
    def detect(self, path: Path) -> ParserDetectionResult:
        """Check if this parser can handle the data"""

    @abstractmethod
    def get_device_info(self, path: Path) -> DeviceInfo:
        """Extract device metadata"""

    @abstractmethod
    def parse_sessions(self, path: Path, ...) -> Iterator[UnifiedSession]:
        """Parse all sessions to unified format"""

    @abstractmethod
    def get_metadata(self) -> ParserMetadata:
        """Parser identification and capabilities"""
```

**Adding New Parser**:
```python
# File: src/snore/parsers/philips.py
class PhilipsParser(DeviceParser):
    def detect(self, path):
        return (path / "PXXXXXX").exists()

    def get_device_info(self, path):
        return DeviceInfo(manufacturer="Philips", ...)

    def parse_sessions(self, path):
        # Convert Philips format → UnifiedSession
        yield unified_session

    def get_metadata(self):
        return ParserMetadata(parser_id="philips", ...)

# Auto-register
parser_registry.register(PhilipsParser())
```

### Parser Registry

**File:** `src/snore/parsers/registry.py`

- Auto-detects device type from file structure
- Confidence-based parser selection
- Manufacturer hints (optional)
- Global singleton: `parser_registry`

```python
# Usage
parser = parser_registry.detect_parser(path)
for session in parser.parse_sessions(path):
    # session is UnifiedSession
    print(f"{session.start_time}: {session.duration_hours}h")
```

### EDF+ Reader Library

**File:** `src/snore/parsers/formats/edf.py`

Generic EDF/EDF+ file reader for medical devices:
- Signal extraction with proper unit conversion
- Annotation parsing (for events)
- Header information extraction
- **Reusable** by any EDF-based parser

---

## Analysis System

### Overview

The analysis system performs programmatic respiratory event detection on imported CPAP sessions using configurable detection modes.

**Architecture:**
```
analysis/
├── service.py          # Orchestration
├── types.py            # AnalysisResult, AnalysisEvent
├── calculations.py     # Metric calculations
├── rx_tracker.py       # Prescription change tracking
├── utils.py            # Analysis utilities
├── data/
│   └── waveform_loader.py  # Database waveform loading
├── shared/             # Core algorithms
│   ├── breath_segmenter.py
│   ├── feature_extractors.py
│   ├── flow_limitation.py
│   ├── pattern_detector.py
│   ├── pulse_detector.py       # Pulse change detection (NEW)
│   └── types.py
└── modes/
    ├── config.py       # AASM_CONFIG, AASM_RELAXED_CONFIG, RESMED_CONFIG
    ├── detector.py
    └── types.py
```

### Detection Modes

**Config-Based Strategy Pattern:**
- Single `EventDetector` class configurable via `DetectionModeConfig`
- No class hierarchy - behavior controlled by configuration
- Easily extensible by adding new configs

**Available Modes:**

1. **AASM Mode** (default)
   - AASM Scoring Manual v2.6 compliant
   - Time-based baseline (120 seconds, 2 minutes)
   - 90% validation threshold
   - Strict apnea detection (≥90% flow reduction)
   - Hypopnea detection (30% flow + 3% SpO2 desaturation)
   - RERA detection enabled

2. **AASM Relaxed Mode**
   - AASM-based with relaxed thresholds
   - Breath-based baseline (30 breaths)
   - 85% validation threshold
   - Hypopnea detection (30% flow + 3% SpO2 desaturation)
   - RERA detection enabled
   - Better for matching machine-detected events

3. **ResMed Mode**
   - Approximates ResMed machine detection logic (gap + low-flow approach)
   - Time-based baseline (120 seconds, 2 minutes)
   - 50% apnea threshold (vs 90% AASM) — detects flow gaps, not near-complete cessation
   - Flow-only hypopnea detection (20% reduction, no SpO2 required)
   - RERA detection enabled
   - Designed to match ResMed AirSense/AirCurve event counts

**Configuration Parameters:**
```python
DetectionModeConfig(
    name="mode_name",
    baseline_method=BaselineMethod.TIME | BaselineMethod.BREATH,
    baseline_window=120.0,  # seconds or breath count
    apnea_threshold=0.90,  # 90% flow reduction
    apnea_validation_threshold=0.90,
    hypopnea_min_threshold=0.30,
    hypopnea_max_threshold=0.89,
    min_event_duration=10.0,
    merge_gap=3.0,
    metric="amplitude",
    hypopnea_mode=HypopneaMode.AASM_3PCT | AASM_4PCT | FLOW_ONLY | DISABLED,
    hypopnea_flow_only_fallback=True,  # Fallback if no SpO2 data
    rera_detection_enabled=True,  # Detect RERA-like events
)
```

**Hypopnea Detection Modes:**
- `AASM_3PCT` - 30% flow + 3% SpO2 drop (AASM recommended)
- `AASM_4PCT` - 30% flow + 4% SpO2 drop (CMS/Medicare)
- `FLOW_ONLY` - Flow reduction, no SpO2 required (threshold config-controlled; 20% in ResMed mode)
- `DISABLED` - Skip hypopnea detection

### Analysis Pipeline

```
1. Load waveform data (timestamps, flow values)
   ↓
2. BreathSegmenter.segment_breaths()
   → Identifies individual breaths from flow signal
   ↓
3. WaveformFeatureExtractor (per breath)
   → Extracts shape features, spectral features, waveform features
   ↓
4. FlowLimitationClassifier.analyze_session()
   → Classifies flow limitation severity
   ↓
5. ComplexPatternDetector
   → Detects CSR (Cheyne-Stokes Respiration)
   → Detects periodic breathing
   ↓
5.5. PulseChangeDetector
   → Detects pulse rate changes ≥5 BPM within 8-second sliding windows
   → Outputs pulse_change_count and pulse_change_index (per hour)
   ↓
6. EventDetector.detect_events() (per mode)
   → Detects apneas (obstructive, central, mixed, unspecified) with confidence levels
   → Detects hypopneas (mode-dependent: SpO2-based or flow-only)
   → Detects RERAs (Respiratory Effort-Related Arousals)
   → Calculates AHI, RDI (includes RERAs)
   ↓
7. AnalysisResult (stored in database)
   → mode_results: {mode_name: ModeResult}
   → flow_analysis, csr_detection, periodic_breathing
```

### Type System

**All analysis types use Pydantic models:**
- Validation at construction time
- Automatic JSON serialization via `model_dump()`
- No manual `to_dict()`/`from_dict()` methods needed

**Key Types:**
- `BreathMetrics` - Individual breath measurements (Pydantic)
- `ApneaEvent`, `HypopneaEvent`, `RERAEvent` - Detected events (Pydantic)
- `ModeResult` - Per-mode detection results (Pydantic)
- `AnalysisResult` - Complete analysis output (Pydantic)
- `DetectionModeConfig` - Mode configuration (frozen Pydantic)

---

## REST API

### Overview

FastAPI application serving the same data as the CLI through HTTP endpoints. Launched via `snore serve`.

**Application:** `src/snore/api/app.py`

**Routers:**
| Router | Prefix | Endpoints |
|--------|--------|-----------|
| auth | `/auth` | Login/logout/demo-login/status, active profile, invite lookup/redeem, Google OAuth (login + invite signup, single callback) |
| me | `/auth/me` | Self-service account: display name, password, preferences |
| admin | `/admin` | User management, invite lifecycle (admin-only) |
| profiles | `/profiles` | Profile CRUD |
| sessions | `/sessions` | List, detail, enable/disable, delete, bulk delete-preview |
| waveforms | `/sessions` | List types, get data (LTTB downsampling), compare events |
| events | `/sessions` | List, match machine vs programmatic |
| analysis | (v1 prefix) | List status, get result, run, delete, batch analysis |
| stats | `/stats` | Summary, periods, trends, records |
| devices | `/devices` | List, detail |
| days | `/days` | List, detail |
| rx | `/rx` | History, current, compare |
| equipment | `/equipment` | Mask log CRUD (`GET/POST /masks`, `PATCH/DELETE /masks/{id}`); device mask-type epochs (`GET /masks/epochs`); write ops guarded by `RequireWritable` |
| reports | `/reports` | Report generation |
| import | `/import` | Upload files, SSE progress, cancel, pipeline job list; detect + path-import (local mode only) |
| export | `/export` | CSV, JSON, raw file download |
| db | `/db` | Stats, vacuum; reset (local mode only) |
| validation | `/validate` | Batch validation report |

**Key patterns:**
- Dependency injection: `db: Session = Depends(get_db)` for database sessions
- Caller-controlled transactions: `get_db()` handles commit/rollback
- Domain exceptions: `NotFoundError` → 404, genuine `ValueError` → 500
- LTTB downsampling: 720k-point waveforms served in <100ms via `max_points` param
- CORS: Configured for Vue dev server (`localhost:5173`)
- OpenAPI: Auto-generated docs at `/docs`
- Auth middleware (`api/middleware.py`): `AuthMiddleware` resolves the actor from the signed session cookie on every request (auto-provisions the local actor in local mode); `AuthPathMiddleware` enforces the CSRF Origin/Referer check on unsafe methods, a 16 KiB body cap on `/api/v1/auth/*`, and blanket `Cache-Control: no-store` on auth responses; `RateLimitMiddleware` applies a per-IP 30 req/60 s window to `/api/v1/auth/*` in multiuser mode

### Background Job Pipeline

Imports and analysis run as serialised background jobs to avoid SQLite write-lock contention.

**Two-phase import flow:**
1. `POST /import` — accepts multipart upload, reserves an admission slot, writes files to a temp directory, and returns a `job_id` immediately (HTTP 202).
2. The persistent import worker thread (`api/import_worker.py`) picks up the job from the FIFO queue and runs `ImportService.import_sources()`. On success it emits a non-terminal `phase_complete` SSE event carrying the import result, then enqueues a downstream `AnalysisJob`.
3. The import job terminates with `complete` (or `error`/cancel). Every terminal payload carries `import_committed=True` and the full import result whenever the import phase committed to the database, so late or stalled SSE observers always learn that data landed.
4. `GET /import/{id}/progress` — SSE stream; connecting after the job is terminal delivers the stored terminal event immediately.

**Admission caps** — `SNORE_MAX_JOBS_PER_USER` (default 3) and `SNORE_MAX_JOBS_GLOBAL` (default 10) gate new reservations before any request body bytes are read. The slot counts until `release_capacity()` is called after temp-file cleanup.

**TTL reaper** — `api/import_jobs.py` runs both a dedicated background reaper thread (every 60 s) and an eager reap-on-read inside `get_job` / `list_jobs` / `reserve_slot` (because HTTP handlers poll the store on every request). `api/analysis_jobs.py` reaps inline after each job completes inside the worker loop — sufficient there because the analysis store is accessed only by the worker thread and a low-frequency list endpoint.

---

## Service Layer

19 service modules in `src/snore/services/` form the business logic layer between CLI/API and database:

| Service | Responsibility |
|---------|---------------|
| AnalysisFacade | Analysis orchestration, result retrieval, batch analysis |
| BackupService | Raw SD card file backup to `~/.snore/raw/` |
| BreathService | Per-session breath-by-breath data retrieval; stores and queries segmented breath records with shape features and flow-limitation scores |
| DatabaseService | Database operations (stats, vacuum, init) |
| DayService | Day aggregation and lookup |
| DemoService | Demo mode bootstrap — provisions the demo user, profile, and fixture session data for local evaluation |
| DeviceService | Device listing and per-device detail (usage summary, current settings, settings history) |
| EventService | Event queries and matching |
| ExportService | Data export (CSV, JSON, raw files) |
| ImportService | CPAP data import: source detection, file upload, backup, parse orchestration |
| lttb (module) | Largest-Triangle-Three-Buckets downsampling via `lttb_downsample()` |
| MaskEpochService | Contiguous device-reported mask-type epochs via `RxTracker.get_history(keys=("mask_type",))`, normalized to the mask style vocabulary |
| MaskLogService | Profile-scoped CRUD for user-entered mask equipment log; all identity fields optional; 404-not-403 for foreign profile IDs |
| ProfileService | Profile CRUD, timezone validation, and raw-directory quarantine/purge for offline deletion |
| ReportService | Nightly and comparison CPAP report generation from session and statistics data |
| SessionService | Session CRUD and filtering |
| StatsService | Statistics calculations and summaries |
| WaveformService | Waveform data access, formatting, event comparison |
| WriterLeaseManager | Shared/exclusive SQLite writer lease acquired at startup; serialises import + analysis writes |

Prescription/therapy settings tracking lives in `analysis/rx_tracker.py` (RxTracker),
which returns the Pydantic responses from `services/schemas.py` directly.

**Pattern:** Constructor injection with SQLAlchemy session, typed Pydantic returns via `services/schemas.py`.

---

## Database Schema

### Tables

**profiles**
```sql
id, username (UNIQUE), first_name, last_name, date_of_birth, height_cm,
settings (JSON), created_at, updated_at
```

**devices**
```sql
id, manufacturer, model, serial_number, firmware_version,
hardware_version, product_code, first_seen, last_import
UNIQUE(serial_number)
```

**days**
```sql
id, device_id (FK devices), date,
session_count, total_therapy_hours,
obstructive_apneas, central_apneas, hypopneas, reras,
ahi, oai, cai, hi,
pressure_min/max/median/mean/95th, epap_min/max/median/mean/95th,
leak_min/max/median/mean/95th, spo2_min/max/mean,
created_at, updated_at
UNIQUE(device_id, date)
```

After any membership change a `days` row survives only if at least one `sessions` row (enabled or disabled) still references it; `DayManager.recalculate_day` prunes orphaned rows after session deletion, import-time replacement, and `snore db recompute-days` (migration `017` removed shells left by earlier versions). `session_count` counts enabled sessions, so a day whose sessions are all disabled persists with `session_count = 0` and reset aggregates. Requesting a fully deleted day by date therefore returns not-found rather than a zeroed row.

**sessions**
```sql
id, device_id (FK devices), day_id (FK days),
device_session_id, start_time, end_time, duration_seconds,
therapy_mode, import_date, import_source, parser_version,
data_quality_notes (JSON),
has_waveform_data, has_event_data, has_statistics, enabled
UNIQUE(device_id, device_session_id)
```

**waveforms**
```sql
id, session_id, waveform_type, sample_rate, unit,
min_value, max_value, mean_value, data_blob, sample_count
UNIQUE(session_id, waveform_type)
```
- `data_blob`: Numpy array as bytes (timestamps and values stacked)
- **No compression** - SQLite/filesystem handles that efficiently
- Simplified from original design for performance

**events**
```sql
id, session_id (FK sessions), event_type, start_time, duration_seconds,
spo2_drop, peak_flow_limitation
```

**statistics**
```sql
session_id (PK, FK sessions), [45+ metric columns]
AHI, OAI, CAI, HI, REI, pressure stats, leak stats, SpO2 stats, etc.
```

**settings**
```sql
id, session_id (FK sessions), key, value
UNIQUE(session_id, key)
```
Key-value pairs for extensibility across device types

**analysis_results**
```sql
id, session_id (FK sessions),
timestamp_start, timestamp_end,
programmatic_result_json (JSON), processing_time_ms,
engine_versions_json (JSON), created_at
```
Stores programmatic analysis results (detection mode, events, AHI/RDI, metadata)

**detected_patterns**
```sql
id, analysis_result_id (FK analysis_results),
pattern_id, start_time, duration, confidence,
detected_by, metrics_json (JSON), notes
```

**mask_log**
```sql
id, profile_id (FK profiles),
brand (nullable), model (nullable), size (nullable), style (nullable),
start_date (nullable), notes (nullable),
created_at, updated_at
```
All identity fields (`brand`, `model`, `size`, `style`, `start_date`) are nullable to support partial entries saved before the user has complete information. CHECK constraints enforce non-empty when provided (`brand IS NULL OR length(brand) > 0`, same for `model`) and style within the vocabulary (`style IS NULL OR style IN ('pillows','nasal','full_face')`). Index `ix_mask_log_profile_start_date` on `(profile_id, start_date)`. Entries with NULL `start_date` are excluded from active-mask resolution and from the merged settings timeline. Added by migrations 008 and 009.

---

## Data Flow

### Current ResMed Flow (Working)
```
ResMed SD Card
    ↓
DATALOG/YYYY/*.edf files
    ↓
ResMed EDF+ Parser (src/snore/parsers/resmed_edf.py)
    ↓
UnifiedSession objects
    ↓
Session Importer (src/snore/database/importers.py)
    ↓
SQLite Database (~/.snore/snore.db)
    ↓
Analysis Tools
    ↓
CLI/Reports
```

### Other Devices Flow (Working)
```
Any CPAP Device
    ↓
OSCAR Desktop App
    ↓
.000/.001 binary files
    ↓
OSCAR Binary Parser ✅
    ↓
UnifiedSession objects
    ↓
[Same pipeline as ResMed]
```

### API Data Flow
```
SQLite Database
    ↓
Service Layer (services/)
    ↓
FastAPI Routers (api/routers/)
    ↓
JSON Response → Vue Frontend (ui/)
```

---

## MCP Server

`src/snore/mcp/` implements the third presentation layer — a peer of the CLI and the FastAPI REST API. It exposes SNORE data to LLM clients via the Model Context Protocol.

### Transports

Two transports are available, controlled by the deployment mode:

- **stdio** (`snore mcp`): local tool use; no authentication. A `StaticRuntime` resolves the active profile at startup from the first live profile row and watches for database file replacement (inode change → auto-refresh). Use this for Claude Desktop / Claude Code integration.
- **streamable-HTTP** (embedded in `snore serve`): multi-user, OAuth 2.1 via FastMCP's `GoogleProvider`. When `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` are set in multiuser mode, `snore.api.mcp_embed.build_mcp_app` mounts the FastMCP server at `{SNORE_PUBLIC_BASE_URL}/mcp` inside the FastAPI app — one uvicorn process serves both, same origin, no separate MCP base URL needed. Each request carries a per-user `ActorRuntime` that reads `profile_id` from the request context var rather than a startup-time lookup. The FastMCP sub-app lifespan is chained inside the FastAPI lifespan so the database engine is shared and never double-initialized.

Both transports share the same tool and resource registrations; the only difference is which `SNORERuntime` implementation is yielded by the FastMCP lifespan.

### SNORERuntime Protocol Seam

Tools never import a database session factory directly. Instead, `server.py` yields a `SNORERuntime` instance from its FastMCP lifespan context, and every tool accesses `runtime.scope_provider()` and `runtime.profile_id` through this interface:

```
StaticRuntime     — stdio path; profile_id resolved at startup
ActorRuntime      — OAuth HTTP path; profile_id from per-request context var
```

`_scope_and_run(ctx, impl, *, tool_name, **kwargs)` in `server.py` captures the common scaffold shared by eight of the eleven tools: open scope → call impl with `(db, profile_id=..., **kwargs)` → `model_dump(mode="json")` → `_check_response_size`.

### Tool Modules

Each tool is defined in `src/snore/mcp/tools/<name>.py` and owns a `register(mcp: FastMCP) -> None` function. `make_server` calls every module's `register` so the tool registration logic is co-located with the tool's DB-fetch and mapping code.

| Module | Tool name | Pattern |
|--------|-----------|---------|
| `overview.py` | `get_data_overview` | `_scope_and_run` |
| `settings.py` | `get_settings_timeline` | `_scope_and_run` |
| `changes.py` | `get_settings_changes` | `_scope_and_run` |
| `summary.py` | `get_nightly_summary` | `_scope_and_run` |
| `events.py` | `get_events` | `_scope_and_run` |
| `breath_table.py` | `get_breath_table` | `_scope_and_run` |
| `windows.py` | `find_windows` | `_scope_and_run` |
| `epochs.py` | `compare_epochs` | `_scope_and_run` |
| `ca_analysis.py` | `get_ca_analysis` | fetch inside scope, compute outside |
| `waveform.py` | `get_waveform` | `_fetch_waveform_for_tool` helper |
| `waveform.py` | `render_window` | `_fetch_waveform_for_tool` helper |

### Tiered Data Model

Tools implement progressive disclosure across three tiers:

| Tier | Tools | Description |
|------|-------|-------------|
| 1 (primary) | overview, summary, settings, changes, events, epochs, ca_analysis | Computed metrics — indices, percentiles, aggregates |
| 2 (secondary) | render_window | PNG charts — visual inspection of waveform windows ≤15 min |
| 3 (escape hatch) | get_waveform, breath_table | Raw arrays and per-breath rows for deep inspection |

---

## File Locations

- **Database:** `~/.snore/snore.db`
- **ResMed data:** `DATALOG/YYYY/*.edf` on SD card or `~/Downloads/OSCAR/Profiles/<Profile>/<Device>/Backup/`
- **OSCAR cache:** `~/Downloads/OSCAR/Profiles/<Profile>/<Device>/Summaries/*.000` and `Events/*.001`

---

## Dependencies

**Core:** Python 3.13+, pydantic, sqlalchemy, alembic, click, numpy, pyedflib, scipy, mne, python-dateutil, pytz, requests

**API:** fastapi, uvicorn, httpx

**CLI:** rich, plotext, pint, jinja2, packaging

**Development:** pytest, pytest-cov, ruff, mypy

**All managed via** `pyproject.toml` with uv

---

## Device Support

**ResMed Devices (Direct Import)** ✅
- AirSense 10/11, AirCurve 10/11, S9 series
- Native EDF+ format → Direct import from SD card
- ~40% of CPAP market

**All Other Manufacturers (via OSCAR)** ✅
- Philips, Fisher & Paykel, Löwenstein, Weinmann, DeVilbiss, BMC, and 12+ others
- Import via OSCAR desktop app → Binary cache files (.000/.001)
- OSCAR binary parser fully implemented
- 100% device coverage (all 18 OSCAR-supported manufacturers)
