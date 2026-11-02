# SNORE
**S**leep e**N**vironment **O**bservation & **R**espiratory **E**valuation

![CI Status](https://github.com/wpfleger96/SNORE/actions/workflows/ci.yml/badge.svg)
[![GitHub Contributors](https://img.shields.io/github/contributors/wpfleger96/snore.svg)](https://github.com/wpfleger96/snore/graphs/contributors)
[![Lines of Code](https://aschey.tech/tokei/github/wpfleger96/snore?category=code)](https://github.com/wpfleger96/snore)
[![License](https://img.shields.io/github/license/wpfleger96/snore.svg)](https://github.com/wpfleger96/snore/blob/main/LICENSE)

CLI tool for analyzing and inspecting CPAP/APAP therapy data.

## Overview

SNORE analyzes CPAP therapy data, generates reports, and provides insights about treatment effectiveness. Import data directly from your CPAP device's SD card.

### Features

- **Direct ResMed Import**: Import ResMed AirSense/AirCurve data directly from SD card
- **OSCAR Parser Support**: Import data from all OSCAR-cached manufacturers
- **Universal Database**: SQLite storage for CPAP data
- **Auto-Detection**: Automatically detects ResMed device type
- **CLI Tool**: Import, query, delete, and manage CPAP data from command line
- **Comprehensive Parsing**: Waveforms, events, statistics, and device metadata
- **Waveform Visualization**: Terminal charts with plotext
- **Prescription/Therapy Tracking**: View and compare therapy settings over time
- **Data Export**: Raw backup, CSV, and JSON export formats
- **REST API**: FastAPI server launched via `snore serve`
- **Batch Validation**: Compare programmatic vs machine event detection

**Supported Devices:** ResMed AirSense 10/11, AirCurve 10/11, S9 series, and all OSCAR-cached manufacturers

## Installation

### Prerequisites

- Python 3.13 or later
- UV package manager
- ResMed SD card OR OSCAR desktop application (for non-ResMed devices)

### Global Installation

**From GitHub (Development)**

Install from GitHub to get the latest development code:

```bash
uv tool install git+ssh://git@github.com/wpfleger96/SNORE.git
```

Or use the `setup` command with `--github`:

```bash
cd SNORE
uv run snore setup --github
```

### Development Installation

```bash
# Clone the repository
git clone git@github.com:wpfleger96/SNORE.git
cd SNORE

# Install dependencies (never use editable install — see AGENTS.md gotcha #8)
uv sync
```

## Quick Start

### 1. Import CPAP Data

For **ResMed AirSense/AirCurve users**, import directly from your SD card:

```bash
# Install dependencies
uv sync

# Import ResMed data from SD card
uv run snore import /path/to/ResMed/Backup/

# Or if you've already imported to OSCAR desktop app
uv run snore import ~/Downloads/OSCAR/Profiles/<Profile>/ResMed_*/Backup/

# Advanced import options
# --limit N               Import only first N sessions
# --sort-by date-desc     Newest first (date-asc, date-desc, filesystem)
# --from / --to           Filter by date range
# --dry-run               Preview without importing
# --force                 Re-import existing sessions
# --no-parallel           Disable parallel parsing (for debugging)
# --batch-size N          Sessions per transaction (default: 50)
# --no-backup             Skip raw file backup
# --backup-dir PATH       Raw backup directory (default: ~/.snore/raw/)
# --all                   Import all detected sources without prompting
uv run snore import /path/to/data/ \
  --limit 10 \
  --sort-by date-desc \
  --from 2024-01-01 \
  --to 2024-12-31 \
  --dry-run
```

The import will:
- Auto-detect your ResMed device
- Parse all session files (BRP, PLD, SA2, EVE)
- Store waveforms, events, and statistics in SQLite
- Show progress bar during import

### 2. View Imported Data

```bash
# List all imported sessions
uv run snore session list

# List sessions for specific device
uv run snore session list --device 22231974465

# List sessions in date range
uv run snore session list --from 2024-01-01 --to 2024-12-31

# Show database statistics
uv run snore db stats

# Show session details with therapy settings
uv run snore session show 123 --settings

# Visualize waveform data
uv run snore waveform list --date 2024-12-05
uv run snore waveform show --date 2024-12-05 --time 01:30:00
uv run snore waveform show --date 2024-12-05 --time 01:30:00 --type flow,pressure
uv run snore waveform compare --session-id 123 --mode aasm

# View prescription/therapy settings
uv run snore rx history
uv run snore rx current
uv run snore rx compare
```

### 3. Analyze CPAP Sessions

Run programmatic respiratory event detection on imported sessions:

```bash
# Analyze specific session by ID
uv run snore analysis run --session-id 123

# Analyze specific date
uv run snore analysis run --date 2024-12-05

# Analyze date range
uv run snore analysis run --from 2024-12-01 --to 2024-12-31

# Run specific detection mode
uv run snore analysis run --date 2024-12-05 --mode aasm_relaxed

# Run all available modes
uv run snore analysis run --date 2024-12-05 --all-modes

# Analysis run options
uv run snore analysis run --date 2024-12-05 --no-store       # Don't save to DB
uv run snore analysis run --date 2024-12-05 --debug-events   # Detailed machine vs programmatic comparison
uv run snore analysis run --date 2024-12-05 --plain          # Plain output without colors

# List analyzed sessions
uv run snore analysis list

# Show stored analysis results
uv run snore analysis show --session-id 123

# Delete analysis results
uv run snore analysis delete --session-id 123
```

**Available Detection Modes:**
- `aasm` (default) - AASM Scoring Manual v2.6 compliant detection with 3% SpO2 desaturation
- `aasm_relaxed` - AASM-based with relaxed thresholds for machine matching
- `resmed` - ResMed machine approximation using flow-only hypopnea detection

**Analysis Output:**
- Detected apneas (obstructive, central, mixed, unspecified) with confidence levels
- Detected hypopneas (configurable modes: AASM 3%/4% SpO2, flow-only, disabled)
- Detected RERAs (Respiratory Effort-Related Arousals)
- AHI (Apnea-Hypopnea Index) and RDI (Respiratory Disturbance Index including RERAs)
- Flow limitation analysis
- Complex pattern detection (CSR, periodic breathing)

### 4. Manage Sessions

```bash
# Delete sessions by date range (with preview)
uv run snore session delete --from 2024-01-01 --to 2024-01-31 --dry-run

# Delete sessions for specific device
uv run snore session delete --device 22231974465 --dry-run

# Delete specific sessions by ID
uv run snore session delete --session-id "1,2,3"

# Delete all sessions (with confirmation)
uv run snore session delete --all                        # DESTRUCTIVE: deletes all session data

# Force delete without confirmation prompt
uv run snore session delete --session-id "5" --force

# Enable/disable sessions (exclude from statistics)
uv run snore session enable 123
uv run snore session disable 123

# Database maintenance after large deletions
uv run snore db vacuum

# Initialize database
uv run snore db init
```

### 5. Statistics

```bash
# Show therapy statistics
uv run snore stats                    # Summary statistics
uv run snore stats --period week      # Weekly breakdown
uv run snore stats --period month     # Monthly breakdown
uv run snore stats --trend            # Trend analysis with charts
uv run snore stats --records          # Top 5 best/worst days
uv run snore stats --days 30          # Last 30 days only
```

### 6. Export Data

```bash
# Export data
uv run snore export raw --from 2024-01-01 --to 2024-12-31  # Raw file backup
uv run snore export csv --from 2024-01-01                   # CSV export
uv run snore export json --from 2024-01-01 --to 2024-12-31 # JSON export
```

### 7. Validate

```bash
# Batch validation (compare programmatic vs machine detection)
uv run snore validate --from 2024-01-01 --to 2024-12-31
```

### 8. Web UI & REST API

```bash
# Start API in single-user local mode (no authentication, loopback only)
just dev

# Start API in multiuser mode with a seeded test invite (dev config in local.just)
just dev-auth

# Production / network deployment — requires SNORE_SESSION_SECRET and SNORE_PUBLIC_BASE_URL
SNORE_AUTH_MODE=multiuser \
SNORE_SESSION_SECRET=<32-char-secret> \
SNORE_PUBLIC_BASE_URL=https://snore.example.com \
uv run snore serve --host 0.0.0.0 --port 8000

# Web UI at http://localhost:8000, API docs at http://localhost:8000/docs
```

> **Deployment note:** `snore serve` is the supported network launcher.
> The **default auth mode is `multiuser`** (fail-closed) — running
> `uv run snore serve` without setting `SNORE_AUTH_MODE=local` will exit
> immediately if `SNORE_SESSION_SECRET` or `SNORE_PUBLIC_BASE_URL` is not
> configured.  Use `just dev` for explicit single-user local mode (loopback
> only), `just dev-auth` for multiuser development, or the full env block
> above for network deployment.

**First-admin bootstrap (`SNORE_BOOTSTRAP_ADMIN_EMAIL`):** Set this optional
env var to an email address and, on startup, SNORE will automatically create a
7-day admin invite for that address if no active admin user exists yet.  The
redemption URL is printed to `docker logs` (search for "Bootstrap admin invite
created").  Once redeemed, the variable has no further effect.  If you prefer
to create the invite manually instead, run:

```bash
snore user invite <email> --role admin
```

The redemption URL contains the one-time invite token and is written to the
server log as the delivery channel, so treat your logs as sensitive until the
invite is redeemed.

#### Deploy gating (watchtower)

SNORE ships a pre-update lifecycle hook for `nickfedor/watchtower` that defers container replacement while the app is mid-operation (active import, running analysis, or database reset job).

**Endpoint:** `GET /health/busy` — unauthenticated, not in the OpenAPI schema. Returns `{"busy": <bool>}`. Active signals: in-flight import jobs (PENDING_UPLOAD, PENDING, or RUNNING), running analysis jobs, and a queued or running database reset or data deletion job (including its VACUUM). Reasons are logged at DEBUG level for operator diagnostics and omitted from the response to avoid leaking activity timing on a tunnel-exposed endpoint.

**Hook script:** `/app/scripts/watchtower-pre-update.sh` (absolute path inside the container). Exits `75` (EX_TEMPFAIL) when the app is busy, `0` otherwise. Fail-open: any error — connection refused, timeout, missing python, malformed response — exits `0` so an unhealthy app never blocks its own replacement.

**Known limitation:** CLI imports run via `docker exec` inside the container are not reflected in the busy signal; only API-initiated imports are tracked.

**Compose configuration** the operator must add to the `snore` service (add these only after an image containing the script has been deployed — a missing script exec-fails and aborts the whole watchtower update run under this fork's semantics):

```yaml
services:
  snore:
    labels:
      com.centurylinklabs.watchtower.lifecycle.pre-update: /app/scripts/watchtower-pre-update.sh
      com.centurylinklabs.watchtower.lifecycle.pre-update-timeout: "2"  # minutes (= 120 s); must exceed the hook's 5 s fetch timeout
    stop_grace_period: 60s

  watchtower:
    environment:
      WATCHTOWER_LIFECYCLE_HOOKS: "true"
```

### 9. Shell Completions

```bash
uv run snore completions install    # Auto-detect and install shell completions
uv run snore completions bash       # Show bash completion script
uv run snore completions zsh        # Show zsh completion script
```

## Development

```bash
# Install dependencies
just sync

# Quick quality check (no tests)
just

# Run all tests
just test

# Full quality check (Python + UI, no tests)
just check

# Full quality check + tests
just check-all

# Pre-commit checks (type-check, lint, format — Python + UI)
just pre-commit

# CI workflow (type-check, lint, format, test)
just ci

# Start API in single-user local mode (no auth, loopback only)
just dev

# Start API in multiuser mode with a seeded invite
just dev-auth

# Build UI for production
just ui-build
```

## Clinical Disclaimer

This tool is for informational purposes only. Always consult with your sleep medicine physician or healthcare provider for medical advice regarding your CPAP therapy.

## License

MIT License

## Acknowledgments

- OSCAR (Open Source CPAP Analysis Reporter) project for the desktop application
- The sleep apnea community for supporting open-source analysis tools

## Support

For issues, questions, or contributions, please visit the GitHub repository.
