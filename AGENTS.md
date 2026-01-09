# AGENTS.md

SNORE (Sleep eNvironment Observation & Respiratory Evaluation) is a CLI tool for analyzing CPAP/APAP therapy data with commands for importing, querying, and analyzing sleep therapy sessions.

## Quick Commands

```bash
# Development workflow
just                # Quick check: sync, type-check, lint-check, format-check
just test           # Run pytest
just check-all      # Full quality check + tests
just pre-commit     # Pre-commit: sync, type-check, lint, format, test
just ci             # CI workflow (same as pre-commit but checks only)
just lint           # Ruff lint with auto-fix
just format         # Ruff format
uv sync             # Install dependencies

# Testing
uv run pytest tests/unit/test_file.py  # Single test file

# CLI (local development - use `uv run`)
uv run snore import-data <path>                # Import device data
uv run snore list-sessions                     # List sessions
uv run snore analyze run --session-id <id>     # Analyze session
uv run snore analyze list                      # List sessions with analysis status
uv run snore analyze show --date YYYY-MM-DD    # Show analysis results
uv run snore config set-default-profile <name> # Set default profile
uv run snore db drop                           # Drop database (with confirmation)
uv run snore db init                           # Initialize database
uv run snore setup --github                    # Install as uv tool from GitHub
uv run snore upgrade                           # Upgrade to latest version
uv run snore --version                         # Show version and check for updates
```

## Project Structure

```
src/snore/
├── cli.py              # CLI commands (Click)
├── config.py           # Configuration management
├── constants.py        # Channel IDs, mappings
├── analysis/           # Analysis algorithms
│   ├── shared/         # Breath segmentation, feature extraction, flow limitation
│   ├── modes/          # Event detection modes (AASM, AASM Relaxed)
│   └── service.py      # AnalysisService orchestrator (direct component calls)
├── database/           # SQLAlchemy ORM layer
│   ├── models.py       # DB models
│   └── session.py      # session_scope() context manager
├── models/             # Data models
│   └── unified.py      # UnifiedSession, WaveformData, RespiratoryEvent
└── parsers/            # Device parsers
    ├── base.py         # DeviceParser abstract class
    ├── registry.py     # parser_registry singleton
    └── resmed_edf.py   # ResMed EDF+ parser
tests/
├── conftest.py         # Main fixtures
├── fixtures/           # Test data (recorded sessions, device data)
├── helpers/            # synthetic_data.py, validation_helpers.py
├── unit/               # Unit tests (auto-marked)
└── integration/        # Integration tests (auto-marked)
```

## OSCAR Relationship

**OSCAR (Open Source CPAP Analysis Reporter)** is the upstream OSS project that inspired SNORE's design. OSCAR is a mature Qt/C++ desktop GUI supporting 18+ CPAP manufacturers.

**Dual Import Strategy:**
```
Direct Import:    Device SD Card → SNORE → Database (ResMed only, faster)
Via OSCAR:        Device SD Card → OSCAR Desktop → .000/.001 files → SNORE → Database (all manufacturers)
```

**What SNORE borrows from OSCAR:**

1. **Channel ID System:** Uses OSCAR's exact channel IDs (0x1000=Pressure, 0x1100=ObstructiveApnea, 0x1200=Flow, 0x2000=SpO2) from `constants.py`
2. **Binary Format Parsers:** Complete Qt binary format parsers in `parsers/oscar_summary.py`, `oscar_events.py`, `qdatastream.py`
3. **Flow Limitation Classes:** OSCAR's 7-class system (Sinusoidal→Normal, DoublePeak→Mild, etc.)
4. **Algorithm Validation:** Tests compare against OSCAR's breath segmentation and metrics

**OSCAR File Format (.000 summary, .001 events):**
- Magic number: 0xC73216AB
- Qt QDataStream serialization with optional qCompress
- Delta-encoded timestamps for events
- Pre-calculated statistics (AHI, pressure stats, etc.)
- Settings stored as Qt QVariant types

**Key OSCAR parsers:**
- `oscar_summary.py` - Parse .000 files (session metadata, statistics) - **has settings skip hack** due to custom Qt types
- `oscar_events.py` - Parse .001 files (waveforms, respiratory events) - **uses bit masking** on event types
- `qdatastream.py` - Qt binary serialization (little-endian, type markers)
- `compression.py` - Qt qCompress/qUncompress, CRC16, delta-time decoding

**SNORE vs OSCAR:**
- SNORE: CLI-first, Python 3.13+, SQLite, direct device import (ResMed only)
- OSCAR: Desktop GUI, Qt/C++, proprietary binary storage, broad device support

## Tech Stack

- Python 3.13+ with UV package manager
- SQLAlchemy 2.0 ORM with SQLite (~/.snore/snore.db)
- Click CLI framework
- Pydantic for validation
- pytest with coverage

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

**Analysis Architecture:** Direct orchestration in `service.py`:
- BreathSegmenter → feature extraction → FlowLimitationClassifier → ComplexPatternDetector
- Event detection via modes with `DetectionModeConfig`:
  - `aasm` - AASM Scoring Manual v2.6 compliant (default)
  - `aasm_relaxed` - AASM with relaxed thresholds for machine matching
  - `resmed` - ResMed device algorithm matching
- All types use Pydantic models (validation, serialization)

**Unified Data Model:** All device data converts to `UnifiedSession` → `WaveformData` → `RespiratoryEvent`

## Code Style

- Type hints: `str | None` (not Optional), `list[str]` (not List), avoid `Any` types
- Imports: stdlib, third-party, then `snore.` absolute imports
- Naming: snake_case functions, PascalCase classes, UPPER_SNAKE constants
- All data types use Pydantic models (no dataclasses)

## Testing

```bash
just test                                    # All tests
uv run pytest tests/unit/                    # Unit only
uv run pytest tests/integration/             # Integration only
uv run pytest -m recorded                    # Tests using real device data
uv run pytest tests/unit/test_file.py -v    # Single file verbose
```

Markers: `unit`, `integration`, `parser`, `recorded`, `real_data`, `slow`

Key fixtures: `db_session`, `test_profile_factory`, `test_session_factory`, `recorded_session("YYYYMMDD")`

## Common Gotchas

1. **OSCAR day-splitting logic:** Sessions before noon belong to previous day (e.g., 01:50 AM Dec 8 = Dec 7's sleep). Use `Day.date` for display/queries, not `session.start_time.date()`. See `day_manager.py:45-47` and `cli.py:664` (list-sessions uses day_date with fallback)
2. **Refresh after relationship changes:** `db_session.refresh(session)` after adding statistics
3. **Integration test isolation:** Use `reset_database_state()` autouse fixture pattern
4. **WAL cleanup:** Temp databases need `-wal` and `-shm` file cleanup
5. **Profile resolution:** CLI flag > config > auto-detect fallback chain
6. **Type safety:** Use proper types (`list[BreathMetrics]` not `list[Any]`) - mypy strict mode enabled
7. **Pydantic validation:** Use `model_construct()` to bypass validation when testing invalid data
8. **Local development vs installed tool** - **CRITICAL**: Always use `uv run snore` when developing locally:
   - **Local dev (from repo)**: `uv run snore <command>` → runs YOUR local code changes directly
   - **Installed tool (any directory)**: `snore <command>` → runs installed version from `~/.local/share/uv/tools/`
   - Running `snore` without `uv run` will NOT reflect your local changes
   - **NEVER use editable install** (`uv pip install -e .`) - risks conflicts with installed version, unnecessary complexity

## Key Files by Task

| Task | Files |
|------|-------|
| Add CLI command | `src/snore/cli.py` |
| Add device parser | `src/snore/parsers/base.py`, `registry.py`, create new parser file |
| Add analysis algorithm | `src/snore/analysis/shared/` (breath/feature algorithms) or `modes/` (event detection) |
| Add detection mode | `src/snore/analysis/modes/config.py` (add `DetectionModeConfig`), update `detector.py` |
| Modify data models | `src/snore/models/unified.py` (data), `database/models.py` (ORM), use Pydantic |
| Add configuration setting | `src/snore/config.py` (stored in ~/.snore/config.toml) |
| Add test fixture | `tests/conftest.py`, `tests/helpers/` |
| Modify channel IDs | `src/snore/constants.py` (must align with OSCAR's schema.h) |
