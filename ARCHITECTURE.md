# SNORE Architecture

Technical documentation for the SNORE system architecture, components, and design decisions.

---

## System Overview

### Modular, Parser-Agnostic Platform

```
┌─────────────────────────────────────────────┐
│        Analysis Layer (Parser Agnostic)     │
│  get_sessions(), analyze_therapy(), etc.    │
│         Zero knowledge of formats           │
├─────────────────────────────────────────────┤
│         SQLite Database ✅                  │
│  Universal schema, direct BLOB storage      │
│  Auto-creates at ~/.snore/snore.db │
├─────────────────────────────────────────────┤
│         CLI Import Tool ✅                  │
│  snore import (auto-detection)          │
│  Progress bars, duplicate prevention        │
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
│ │   ✅     │  🔧      │          │        ││
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

**File:** `src/snore/models/unified.py`

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
├── service.py          # Orchestration (BreathSegmenter → FeatureExtractor → Classifier → Detector)
├── types.py            # AnalysisResult, AnalysisEvent (Pydantic models)
├── shared/             # Core algorithms
│   ├── breath_segmenter.py      # Breath segmentation from flow data
│   ├── feature_extractors.py   # Waveform feature extraction
│   ├── flow_limitation.py       # Flow limitation classification
│   ├── pattern_detector.py     # Complex pattern detection (CSR, periodic breathing)
│   └── types.py                 # BreathMetrics, ApneaEvent, HypopneaEvent (Pydantic)
└── modes/              # Detection modes
    ├── config.py       # AASM_CONFIG, AASM_RELAXED_CONFIG
    ├── detector.py     # EventDetector (configurable)
    └── types.py        # ModeResult, DetectionModeConfig (Pydantic)
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
   - Approximates ResMed machine detection logic
   - Breath-based baseline (40 breaths)
   - 85% validation threshold
   - Flow-only hypopnea detection (40% reduction, no SpO2 required)
   - RERA detection enabled
   - Designed to match ResMed AirSense/AirCurve event counts

**Configuration Parameters:**
```python
DetectionModeConfig(
    name="mode_name",
    baseline_method=BaselineMethod.TIME | BaselineMethod.BREATH,
    baseline_window=120.0,  # seconds or breath count
    apnea_threshold=0.90,   # 90% flow reduction
    apnea_validation_threshold=0.90,
    hypopnea_min_threshold=0.30,
    hypopnea_max_threshold=0.89,
    min_event_duration=10.0,
    merge_gap=3.0,
    metric="amplitude",
    hypopnea_mode=HypopneaMode.AASM_3PCT | AASM_4PCT | FLOW_ONLY | DISABLED,
    hypopnea_flow_only_fallback=True,  # Fallback if no SpO2 data
    rera_detection_enabled=True  # Detect RERA-like events
)
```

**Hypopnea Detection Modes:**
- `AASM_3PCT` - 30% flow + 3% SpO2 drop (AASM recommended)
- `AASM_4PCT` - 30% flow + 4% SpO2 drop (CMS/Medicare)
- `FLOW_ONLY` - 40% flow reduction (ResMed-style, no SpO2)
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

## Database Schema

### Tables

**devices**
```sql
id, manufacturer, model, serial_number, firmware_version, metadata_json
UNIQUE(manufacturer, serial_number)
```

**sessions**
```sql
id, device_id, device_session_id, start_time, end_time, duration_seconds,
therapy_mode, source_format, data_quality_notes
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
id, session_id, event_type, start_time, duration_seconds, annotation
```

**statistics**
```sql
id, session_id, [45+ metric columns]
AHI, OAI, CAI, HI, REI, pressure stats, leak stats, SpO2 stats, etc.
```

**settings**
```sql
id, session_id, setting_key, setting_value
```
Key-value pairs for extensibility across device types

**analyses**
```sql
id, session_id, mode_name, analysis_timestamp, result_json
```
Stores programmatic analysis results (detection mode, events, AHI/RDI, metadata)

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

### Other Devices Flow (Partial)
```
Any CPAP Device
    ↓
OSCAR Desktop App
    ↓
.000/.001 binary files
    ↓
OSCAR Binary Parser (🔧 partial)
    ↓
UnifiedSession objects
    ↓
[Same pipeline as ResMed]
```

---

## File Locations

### Production Data
```
~/Downloads/OSCAR/Profiles/<ProfileName>/
├── <Manufacturer>_<Serial>/
│   ├── Backup/              # ResMed EDF+ files (direct import ✅)
│   │   ├── STR.edf
│   │   ├── Identification.json
│   │   └── DATALOG/YYYY/
│   │       └── YYYYMMDD_HHMMSS_*.edf
│   ├── Summaries/           # OSCAR binary (partial parser 🔧)
│   │   └── *.000
│   └── Events/              # OSCAR binary (partial parser 🔧)
│       └── *.001
```

### Database Location
```
~/.snore/
└── snore.db            # SQLite database
    └── snore.db-wal    # Write-ahead log
```

---

## Dependencies

**Core:**
- Python 3.13+
- pyedflib (EDF file reading)
- numpy (efficient array operations)
- click (CLI)
- pydantic (data validation)

**Development:**
- pytest (testing)
- pytest-cov (coverage)
- ruff (linting)

**All managed via** `pyproject.toml` with uv

---

## CLI Commands Reference

### Import Data

```bash
# Import ResMed data from SD card
uv run snore import-data ~/path/to/ResMed/Backup/

# Import with verbose logging
uv run snore -v import-data ~/path/to/data/

# Import options (NEW)
uv run snore import-data ~/path/to/data/ \
  --limit 10 \
  --sort-by date-desc \
  --date-from 2024-10-01 \
  --date-to 2024-10-31 \
  --dry-run \
  --force
```

### Query Data

```bash
# List all imported sessions
uv run snore list-sessions

# List sessions in date range
uv run snore list-sessions --from-date 2024-01-01 --to-date 2024-12-31

# Show more sessions
uv run snore list-sessions --limit 50
```

### Analyze Sessions

```bash
# Analyze specific date
uv run snore analyze --date 2024-12-05

# Analyze with specific mode
uv run snore analyze --date 2024-12-05 --mode aasm_relaxed

# Analyze with all modes
uv run snore analyze --date 2024-12-05 --all-modes

# List analyzed sessions
uv run snore analyze --list
```

### Database Management

```bash
# Show database statistics
uv run snore db stats

# Initialize database (rarely needed - auto-created)
uv run snore db init

# Optimize database (reclaim space)
uv run snore db vacuum

# Direct SQL queries
sqlite3 ~/.snore/snore.db
```

### Run Tests

```bash
# All tests
uv run pytest tests/ -v

# Parser tests only
uv run pytest tests/test_parsers.py -v

# Integration tests only
uv run pytest tests/test_import_pipeline.py -v

# With coverage
uv run pytest tests/ --cov=snore

# Check linter
uv run ruff check .
```
---

## Device Support & Coverage

### Supported CPAP/BiPAP Manufacturers

OSCAR supports **18 different device manufacturers**, each with unique native formats:

#### 1. ResMed (Most Popular) - ✅ FULLY SUPPORTED
- **Market Share:** ~40%
- **Models**: S9, AirSense 10/11, AirCurve 10/11
- **Native Format**: EDF+ (European Data Format Plus)
- **Files**: STR.edf, BRP, PLD, EVE, CSL, SAD
- **Status**: Direct import via ResMed EDF+ parser

#### 2. Philips Respironics - 🔧 Via OSCAR Desktop
- **Market Share:** ~30%
- **Models**: DreamStation 1/2/Go, System One, BiPAP AutoSV
- **Native Format**: Proprietary Binary (.000, .001, .002, .005, .006)
- **Status**: Via OSCAR desktop app → .000/.001 export

#### 3. Fisher & Paykel - 🔧 Via OSCAR Desktop
- **Market Share:** ~10%
- **Models**: ICON series, SleepStyle series
- **Native Format**: Proprietary binary (ICON) or EDF+ with extensions
- **Status**: Via OSCAR desktop app

#### 4-18. Other Manufacturers - 🔧 Via OSCAR Desktop
- Löwenstein (WMEDF format)
- Weinmann (WMDATA files)
- DeVilbiss (DV5/DV6)
- BMC, Resvent / Hoffrichter, vREM, and others

### Parser Strategy

**Option 1: Direct Import (ResMed)** ✅
```
ResMed SD Card → SNORE → SQLite Database
```
- Currently working for ResMed devices
- 40% market coverage

**Option 2: OSCAR Desktop Export (All Others)** 🔧
```
Device SD Card → OSCAR Desktop → .000/.001 Files → SNORE → Database
```
- Requires completing OSCAR binary parser
- 100% device coverage

**Option 3: Additional Native Parsers** (Future)
```
Philips SD Card → SNORE → Database
```
- Add parsers for Philips, F&P, etc. as needed
- Each parser ~200-300 lines

---

## Architecture Benefits

1. ✅ **Extensibility**
   - New parsers require zero analysis code changes
   - Add device support by creating one file
   - Database schema supports all device types
   - New analysis modes via simple config addition

2. ✅ **Maintainability**
   - Each parser completely independent
   - Clear separation of concerns
   - Easy to debug and test in isolation
   - Simplified codebase (~2,400 lines removed in recent refactor)
   - Single type system (Pydantic throughout)
   - Config-based detection (no class hierarchy)

3. ✅ **User Experience**
   - Auto-detection "just works"
   - Same tools for all device types
   - Consistent data format
   - Fast imports (no compression overhead)

4. ✅ **Future-Proof**
   - Can support any new CPAP device
   - Can switch parser implementations
   - Database schema is universal

5. ✅ **Performance**
   - Direct BLOB storage (no compression overhead)
   - Efficient numpy serialization
   - Strategic database indexes
   - Fast queries on unified schema

---
