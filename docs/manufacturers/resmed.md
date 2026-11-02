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
| **BRP** | Breathing flow + pressure waveforms | EDF+C | ~25 Hz | Yes |
| **PLD** | Pressure, leak & respiratory trends | EDF+C | ~0.5 Hz | Yes |
| **SA2** | SpO2 & pulse waveforms | EDF+C | 1 Hz | Yes |
| **EVE** | Respiratory events | EDF+C/D | N/A | Yes* |
| **CSL** | Compliance summary | EDF+ | N/A | Yes |
| **TCV** | Trigger/cycle event codes (VAuto only) | EDF+C | ~25 Hz | Yes |

*EVE files contain all-day events; must filter by session timestamps after parsing.
†TCV files are absent on AirSense (APAP/CPAP) devices; silently skipped.

### BRP - Breathing Waveforms
- **Signals**:
  - Flow Rate ("Flow.40ms", "Flow") — L/s on device, auto-converted to L/min
  - Mask Pressure hi-rate ("Press.40ms", "Press") — cmH2O at 25 Hz
- **Typical Rate**: 25 Hz (40ms intervals)

### PLD - Pressure & Leak
- **Signals**:
  - Therapy Pressure ("Press.2s")
  - Mask Pressure ("MaskPress.2s")
  - EPAP ("EPRPress.2s", "Exp Pres" on older models)
  - Leak Rate ("Leak.2s") — L/s on device, auto-converted to L/min
  - Minute Ventilation ("MinVent.2s") — L/min
  - Flow Limitation Index ("FlowLim.2s", "FlowLim", "FFL Index") — dimensionless 0–1
  - Snore Index ("Snore.2s", "Snore") — dimensionless
  - Respiratory Rate ("RespRate.2s", "RespRate") — bpm
  - Tidal Volume ("TidVol.2s", "TidVol") — L on device, stored as mL
  - I:E Ratio ("IERatio.2s", "IERatio") — % (VAuto only)
  - Inspiratory Time ("Ti.2s", "Ti") — seconds (VAuto only)
- **Typical Rate**: 0.5 Hz (2-second intervals, noted by ".2s" suffix)

### TCV - Trigger/Cycle Events (VAuto only)
- **Signals**: TrigCycEvt.40ms — proprietary integer codes 0–16
- **Note**: Codes are undecoded (OSCAR never decoded them); imported verbatim for research
- **Typical Rate**: 25 Hz (40ms intervals)

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
- **Timing**: Annotations mark the end of an event. Apneas carry a real duration, so SNORE subtracts it to store the start. Hypopneas and arousals carry none (`resmed_loader.cpp:3808`), so they keep the annotation time and a null duration

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

ResMed's reporting day runs noon-to-noon. A session's night is determined by its start time:

- Start before noon → attributed to the **previous** calendar day's night
- Start at/after noon → attributed to the **current** calendar day's night

**Example**:
```
Night of June 21, 2024:
  20240621_013454_BRP.edf    # 1:34 AM - 5:30 AM
  20240621_053022_BRP.edf    # 5:30 AM - 7:15 AM
```

#### Noon Mid-Session Rollover

When therapy is physically in progress at 12:00:00 local time, the device closes the active DATALOG files at ~11:59:xx and opens a new file group at ~12:00:0x. One continuous physical session becomes two file groups with a 36–47 s gap, whose filename timestamps fall on opposite sides of the noon boundary.

#### Segment Chaining

`chain_session_segments()` in `src/snore/parsers/resmed_file_index.py` groups DATALOG segments into sessions. It sorts all segments chronologically and chains consecutive segments while the inter-segment gap is ≤ 4 hours (`OSCAR_COMBINE_CLOSE_SECONDS`, anchored to OSCAR's "Combine Close Sessions" default of 240 min). Each chain becomes one session, filed under the night of its first segment's start time (noon rule).

Gap measurement: a segment's end time is derived from its EDF header (`num_records × record_duration`, read from the first readable of `BRP`/`PLD`/`SA2`). Each physical segment also writes an annotation-only group (`CSL` + `EVE` files, stamped seconds before its waveform group) that carries no duration; such groups advance the chain's clock to their own start time (a lower bound) instead of breaking the chain, so a session's stub and waveform groups always chain together.

Discovery I/O cost: gap measurement requires opening each segment's EDF header (one 256-byte read per file) during the discovery phase, before any waveform data is parsed. On a local SSD this is negligible — even a multi-year DATALOG tree with thousands of segments completes in under a second. Importing directly from an SD card over USB with a cold filesystem cache can add roughly 30–60 seconds for a multi-year archive, because the thousands of small header opens happen sequentially. This cost is inherent to measuring inter-segment gaps and cannot be avoided without a separate pre-recorded duration index.

This replaces an earlier unconditional noon-bucket merge that had no gap cutoff. The old approach produced phantom 18–24 h sessions in two cases:

- A noon rollover put the post-noon half of a session into the next night's bucket, where it merged with that evening's sleep across an ~18 h gap.
- Isolated Device Diagnostic or mask-fit blips (segments with actual data) anchored merges across multi-hour gaps.

Chaining fixes both: rollover halves chain across the 36–47 s gap and file under the pre-noon start's night; blips that exceed the 4 h threshold become separate (tiny) sessions.

Multi-segment chains use `device_session_id = "<YYYYMMDD_HHMMSS>_merged"` (first segment's timestamp); single-segment chains keep the segment's `YYYYMMDD_HHMMSS`. The legacy format was `YYYYMMDD_merged`.

## Special File Behaviors

### Zero-Record Files
- **When**: Device powered on briefly but not used; also the dominant output of Air11 Device Diagnostic self-tests (see [Device Diagnostic Self-Test](#device-diagnostic-self-test-airsense-11--aircurve-11) below)
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

### Device Diagnostic Self-Test (AirSense 11 / AirCurve 11)

AirSense 11 and AirCurve 11 devices include a scheduled "Device Diagnostic" self-test, configurable to daily / weekly / every 2 weeks / off. It runs after therapy has stopped, briefly spinning the motor to assess internal acoustics via an acoustic sensor (source: ResMed product support, 2026). Each run writes a full `BRP` / `PLD` / `SA2` DATALOG file group.

Empirically ~97% of diagnostic runs produce zero-record stubs (see [Zero-Record Files](#zero-record-files) above). The remaining ~3% contain 60–240 s of real records and parse as genuine short segments, indistinguishable from mask-fit checks or brief SmartStart mask-fiddling. With the daily setting enabled, expect approximately 2 diagnostic file groups per day.

The sibling **AcousticSignal** feature captures 0.2 s acoustic samples only while therapy is running and writes no DATALOG groups — it cannot create phantom sessions.

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
3. (Optional) `Identification.json` or `Identification.tgt` contains ResMed model

## Supported Models

- AirSense 10 (AutoSet, Elite, CPAP)
- AirSense 11 AutoSet
- AirCurve 10 (S, VAuto, ASV)
- AirCurve 11 VAuto
- S9 (AutoSet, Elite, VPAP Auto)

## Known Divergences from OSCAR

### PLD computed-channel initialization samples

OSCAR skips the first 10 samples (20 s at 0.5 Hz) of the PLD computed channels
(RespRate, TidVol, Ti, IERatio) as device-initialization artifacts
(OSCAR `resmed_loader.cpp:4293`).  SNORE imports all samples without trimming and
leaves artifact handling to downstream analysis.

### TrigCycEvt on older firmware (BRP vs TCV)

On some older firmware (per commented-out OSCAR code, `resmed_loader.cpp:3902`),
`TrigCycEvt.40ms` may reside in the BRP file rather than a separate TCV file.
SNORE currently reads `TrigCycEvt.40ms` only from TCV files; sessions from such
devices will not have a `trigger_cycle` waveform.  This is a known coverage gap.
