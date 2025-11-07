# ResMed CPAP Data Reference

Quick reference for ResMed CPAP device data files and formats.

## Directory Structure

```
Backup/
├── STR.edf                    # Device settings/configuration
├── Identification.json        # Device metadata (optional)
└── DATALOG/
    └── YYYY/                  # Year subdirectories (e.g., 2024/)
        ├── YYYYMMDD_HHMMSS_BRP.edf
        ├── YYYYMMDD_HHMMSS_PLD.edf
        ├── YYYYMMDD_HHMMSS_SA2.edf
        ├── YYYYMMDD_HHMMSS_EVE.edf
        └── YYYYMMDD_HHMMSS_CSL.edf
```

## File Types

| Type | Purpose | Format | Sample Rate | Per-Segment |
|------|---------|--------|-------------|-------------|
| **STR** | Device settings | EDF | N/A | No (all-time) |
| **BRP** | Breathing flow waveform | EDF+C | ~25 Hz | Yes |
| **PLD** | Pressure & leak waveforms | EDF+C | ~0.5 Hz | Yes |
| **SA2** | SpO2 & pulse waveforms | EDF+C | 1 Hz | Yes |
| **EVE** | Respiratory events | EDF+C/D | N/A | Yes* |
| **CSL** | Compliance summary | EDF+ | N/A | Yes |

*EVE files contain all-day events; must filter by session timestamps after parsing.

### BRP - Breathing Waveforms
- **Signals**: Flow Rate ("Flow", "Flow.40ms")
- **Units**: L/min (auto-converted from L/s)
- **Typical Rate**: 25 Hz (40ms intervals)

### PLD - Pressure & Leak
- **Signals**:
  - Mask Pressure ("Press.2s", "MaskPress.2s")
  - Leak Rate ("Leak.2s")
- **Units**: cmH2O (pressure), L/min (leak)
- **Typical Rate**: 0.5 Hz (2-second intervals, noted by ".2s" suffix)

### SA2 - Oximetry Statistics
- **Signals**:
  - SpO2 (oxygen saturation %)
  - Pulse (heart rate bpm)
- **Special**: -1 or 0 = no oximeter connected
- **Note**: File may be empty if oximeter not attached

### EVE - Events
- **Contains**: EDF+ annotations for:
  - OA (Obstructive Apnea)
  - CA (Central Apnea)
  - H (Hypopnea)
  - RERA (Arousal)
  - FL (Flow Limitation)
  - LL (Large Leak)
  - VS (Vibratory Snore)
  - PB (Periodic Breathing)
- **Format**: EDF+C (continuous) or EDF+D (discontinuous)
- **Important**: Stores all-day events; filter by session time range after parsing

## File Naming

Format: `YYYYMMDD_HHMMSS_TYPE.edf`

Example: `20240621_013454_BRP.edf`
- Date: June 21, 2024
- Time: 1:34:54 AM
- Type: Breathing waveform

## Segments and Sessions

### What is a Segment?
A segment = one continuous mask-on period. New files created when:
- User puts mask on
- After mask removal (bathroom, water, etc.)

### Night Grouping (Noon Cutoff)
Sessions grouped into "nights" using noon boundary:
- Before noon → previous day's night
- At/after noon → current day's night

**Example**:
```
Night of June 21, 2024:
  20240621_013454_BRP.edf    # 1:34 AM - 5:30 AM
  20240621_053022_BRP.edf    # 5:30 AM - 7:15 AM
  → Single session with gap between segments
```

## Special File Behaviors

### Zero-Record Files
- **When**: Device powered on briefly but not used
- **Size**: Small stub files (1-3 KB)
- **Header**: Valid EDF header, `num_data_records = 0`
- **Handling**: Gracefully skip, normal occurrence

### Discontinuous Files (EDF+D)
- **Applies to**: EVE files only
- **Format marker**: Header reserved field contains "EDF+D"
- **Cause**: Mask removal detected during recording
- **Limitation**: pyedflib cannot read; requires direct annotation parsing

### EVE All-Day Events
Unlike BRP/PLD/SA2 which contain only segment data, EVE files store all events from the entire day. Must filter by session timestamp range.

## EDF+ Annotation Format

Used in EVE files for event markers:

```
+offset\x15duration\x14Event Text\x14\x00
```

**Delimiters**:
- `\x14` (0x14): Field separator
- `\x15` (0x15): Duration marker
- `\x00` (0x00): End of annotation

**Example**: `+120.5\x1512.5\x14Obstructive apnea\x14\x00`
- Offset: 120.5s from recording start
- Duration: 12.5s
- Event: Obstructive apnea

## Device Detection

ResMed identified by:
1. `STR.edf` file exists in backup root
2. `DATALOG/` directory present
3. (Optional) `Identification.json` contains ResMed model

## Supported Models

- AirSense 10 (AutoSet, Elite, CPAP)
- AirSense 11 AutoSet
- AirCurve 10 (S, VAuto, ASV)
- AirCurve 11 VAuto
- S9 (AutoSet, Elite, VPAP Auto)

## Implementation Notes

- **Parser**: `src/oscar_mcp/parsers/resmed_edf.py`
- **EDF Reader**: `src/oscar_mcp/parsers/formats/edf.py`
- **Signal files** (BRP/PLD/SA2): Always EDF+C (continuous)
- **Event files** (EVE): May be EDF+D (discontinuous)
- **Duration calculation**: `num_data_records × record_duration`
