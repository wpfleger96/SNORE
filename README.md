# SNORE
**S**leep e**N**vironment **O**bservation & **R**espiratory **E**valuation

![CI Status](https://github.com/wpfleger96/SNORE/actions/workflows/ci.yml/badge.svg)
[![GitHub Contributors](https://img.shields.io/github/contributors/wpfleger96/snore.svg)](https://github.com/wpfleger96/snore/graphs/contributors)
[![Lines of Code](https://aschey.tech/tokei/github/wpfleger96/snore?category=code)](https://github.com/wpfleger96/snore)
[![License](https://img.shields.io/github/license/wpfleger96/snore.svg)](https://github.com/wpfleger96/snore/blob/main/LICENSE)

MCP (Model Context Protocol) server for analyzing and inspecting CPAP/APAP therapy data.

## Overview

SNORE provides an MCP interface for CPAP therapy data analysis, enabling LLMs like Claude to analyze sleep therapy data, generate reports, and answer questions about treatment effectiveness.

### Features

- **Direct ResMed Import**: Import ResMed AirSense/AirCurve data directly from SD card
- **Universal Database**: SQLite storage for CPAP data
- **Auto-Detection**: Automatically detects ResMed device type
- **CLI Tool**: Import, query, delete, and manage CPAP data from command line
- **Comprehensive Parsing**: Waveforms, events, statistics, and device metadata
- **Default Profile Configuration**: Save default profile to avoid repeated `--profile` flags

**Supported Devices:** ResMed AirSense 10/11, AirCurve 10/11, S9 series

### Architecture

- **Modular Parser System**: Abstract base class + registry for device detection
- **Universal Data Model**: All devices convert to unified format
- **SQLite Database**: Universal schema supporting all device types
- **Parser-Agnostic MCP Server**: Tools work with any device
- **UV Package Management**: Modern Python dependency management

## Installation

### Prerequisites

- Python 3.13 or later
- UV package manager
- OSCAR desktop application with existing data

### Global Installation (Recommended)

```bash
# From PyPI (TODO)
uv tool install snore

# From GitHub (development)
uv tool install git+ssh://git@github.com/wpfleger96/SNORE.git

# Or from cloned repo
cd SNORE
uv run snore setup              # PyPI (TODO)
uv run snore setup --github     # GitHub
```

### Development Installation

```bash
# Clone the repository
git clone git@github.com:wpfleger96/SNORE.git
cd SNORE

# Install in editable mode
uv pip install -e .
```

## Quick Start

### 1. Import ResMed CPAP Data

For **ResMed AirSense/AirCurve users**, import directly from your SD card:

```bash
# Install dependencies
uv sync

# Import ResMed data from SD card
uv run snore import-data /path/to/ResMed/Backup/

# Or if you've already imported to OSCAR desktop app
uv run snore import-data ~/Downloads/OSCAR/Profiles/<Profile>/ResMed_*/Backup/

# Advanced import options
uv run snore import-data /path/to/data/ \
  --limit 10 \                    # Import only first 10 sessions
  --sort-by date-desc \           # Newest first
  --date-from 2024-01-01 \        # Filter by date range
  --date-to 2024-12-31 \
  --dry-run                       # Preview without importing
```

The import will:
- Auto-detect your ResMed device
- Parse all session files (BRP, PLD, SA2, EVE)
- Store waveforms, events, and statistics in SQLite
- Show progress bar during import

### 2. View Imported Data

```bash
# List all imported sessions
uv run snore list-sessions

# List sessions in date range
uv run snore list-sessions --from-date 2024-01-01 --to-date 2024-12-31

# Show database statistics
uv run snore db stats
```

### 3. Configure Default Profile (Optional)

To avoid passing `--profile` every time, set a default profile:

```bash
# Set default profile
uv run snore config set-default-profile <username>

# View current default
uv run snore config get-default-profile

# Remove default
uv run snore config unset-default-profile

# Show all configuration
uv run snore config show
```

The default profile is stored in `~/.snore/config.toml`. Once set, the `analyze` command will use it automatically:

```bash
# Before: required --profile flag
uv run snore analyze --profile john_doe --all

# After: profile auto-detected
uv run snore analyze --all
```

**Profile Resolution:**
1. Explicit `--profile` flag takes precedence
2. Falls back to configured default
3. Auto-detects if only one profile exists in database
4. Shows helpful error if multiple profiles and no default set

### 4. Analyze CPAP Sessions

Run programmatic respiratory event detection on imported sessions:

```bash
# Analyze specific date (uses default profile)
uv run snore analyze --date 2024-12-05

# Analyze specific session by ID
uv run snore analyze --session-id 123

# Analyze date range
uv run snore analyze --start 2024-12-01 --end 2024-12-31

# Run specific detection mode
uv run snore analyze --date 2024-12-05 --mode aasm_relaxed

# Run all available modes
uv run snore analyze --date 2024-12-05 --all-modes

# List analyzed sessions
uv run snore analyze --list
```

**Available Detection Modes:**
- `aasm` (default) - AASM Scoring Manual v2.6 compliant detection
- `aasm_relaxed` - AASM-based with relaxed thresholds for machine matching

**Analysis Output:**
- Detected apneas (obstructive, central, mixed, unspecified)
- Detected hypopneas
- AHI (Apnea-Hypopnea Index) and RDI (Respiratory Disturbance Index)
- Flow limitation analysis
- Complex pattern detection (CSR, periodic breathing)

### 5. Manage Sessions

```bash
# Delete sessions by date range (with preview)
uv run snore delete-sessions --from-date 2024-01-01 --to-date 2024-01-31 --dry-run

# Delete specific sessions by ID
uv run snore delete-sessions --session-id "1,2,3"

# Delete all sessions (with confirmation)
uv run snore delete-sessions --all

# Force delete without confirmation prompt
uv run snore delete-sessions --session-id "5" --force

# Database maintenance after large deletions
uv run snore db vacuum
```

### 6. Direct Database Access

Query the SQLite database directly:

```bash
sqlite3 ~/.snore/snore.db

# Example queries
SELECT COUNT(*) FROM sessions;
SELECT * FROM devices;
SELECT date(start_time), duration_seconds FROM sessions ORDER BY start_time DESC LIMIT 10;
```

## Database Schema

The SQLite database stores all parsed CPAP data:

**Tables:**
- `devices` - Device metadata (manufacturer, model, serial, firmware)
- `sessions` - Session records (start, end, duration, mode)
- `waveforms` - Time-series data (Flow, Pressure, Leak, SpO2, Pulse)
- `events` - Respiratory events (Apneas, Hypopneas, RERA, etc.)
- `statistics` - Pre-calculated metrics (AHI, pressure stats, leak stats, SpO2)
- `settings` - Therapy configuration (key-value pairs)
- `analyses` - Programmatic analysis results (detection mode, events, metrics)

Database location: `~/.snore/snore.db`

## Project Structure

```
SNORE/
├── src/snore/
│   ├── cli.py                   # CLI commands (import, analyze, list, db)
│   ├── constants.py             # Channel IDs and mappings
│   ├── models/
│   │   └── unified.py           # Universal data model (Pydantic)
│   ├── database/
│   │   ├── schema.sql           # SQLite schema
│   │   ├── manager.py           # Database operations
│   │   └── importers.py         # Session import pipeline
│   ├── parsers/
│   │   ├── base.py              # Abstract parser interface
│   │   ├── registry.py          # Parser auto-detection
│   │   ├── resmed_edf.py        # ResMed EDF+ parser
│   │   └── formats/
│   │       └── edf.py           # Generic EDF reader
│   ├── analysis/
│   │   ├── service.py           # Analysis orchestration
│   │   ├── shared/              # Core algorithms (breath segmentation, features)
│   │   └── modes/               # Detection modes (AASM, AASM Relaxed)
│   └── utils/
├── tests/
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
└── pyproject.toml
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=snore

# Check linting
uv run ruff check .
```

## Clinical Disclaimer

This tool is for informational purposes only. Always consult with your sleep medicine physician or healthcare provider for medical advice regarding your CPAP therapy.

## License

MIT License

## Acknowledgments

- OSCAR (Open Source CPAP Analysis Reporter) project for the desktop application
- MCP (Model Context Protocol) for the server framework
- The sleep apnea community for supporting open-source analysis tools

## Support

For issues, questions, or contributions, please visit the GitHub repository.
