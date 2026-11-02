# OSCAR Feature Reference

Developer reference for understanding OSCAR's implementation to inform SNORE development.

**Repository:** ~/Development/OSCAR-code
**Language:** C++ (Qt5/Qt6)
**Architecture:** Desktop GUI with OpenGL rendering

---

## Data Architecture

### Session/Day/Profile Hierarchy

```
Profile (user account)
  └─ Machine (CPAP device)
      └─ Day (calendar date, noon-to-noon)
          └─ Session[] (individual therapy periods)
              ├─ EventList[] (waveforms, events)
              └─ Summary statistics
```

**Files:**
- `oscar/SleepLib/profiles.h` (lines 73-253) - Profile class
- `oscar/SleepLib/machine.h` (lines 115-406) - Machine class
- `oscar/SleepLib/day.h` (lines 50-374) - Day class
- `oscar/SleepLib/session.h` (lines 44-341) - Session class

### EventList Types

**EVL_Waveform** - Regularly sampled time-series data (flow rate, pressure, SpO2)
- Sample rate + gain/offset encoding
- Compressed storage with run-length encoding

**EVL_Event** - Timestamped discrete events (apneas, hypopneas, flags)
- Start time + optional duration
- Event type code (ChannelID)

**File:** `oscar/SleepLib/event.h` (lines 137-229)

### Channel System

Channels are 32-bit identifiers for all data types. Defined in `oscar/SleepLib/schema.cpp`:

| Range | Category | Examples |
|-------|----------|----------|
| 0x1000-0x11FF | CPAP waveforms | Flow (0x1004), Pressure (0x1001), Leak (0x1009) |
| 0x1200-0x13FF | CPAP events | Obstructive Apnea (0x1201), Hypopnea (0x1203), RERA (0x120A) |
| 0x1400-0x14FF | CPAP settings | Pressure Min/Max, EPR, Ramp |
| 0x2100-0x21FF | Oximetry | SpO2 (0x2100), Pulse (0x2101) |
| 0x2300-0x23FF | Sleep staging | Sleep Stage (0x2300) |
| 0x2990-0x2992 | Position | Orientation, Inclination, Movement |

**Key channels:**
```cpp
CPAP_FlowRate     = 0x1004  // Flow waveform (L/min)
CPAP_Pressure     = 0x1001  // Therapy pressure (cmH2O)
CPAP_Leak         = 0x1009  // Unintentional leak (L/min)
CPAP_Obstructive  = 0x1201  // Obstructive apnea flag
CPAP_Hypopnea     = 0x1203  // Hypopnea flag
CPAP_RERA         = 0x120A  // RERA flag
OXI_SPO2          = 0x2100  // SpO2 percentage
OXI_Pulse         = 0x2101  // Pulse rate (BPM)
```

### Summary Types

Pre-calculated statistics per session/day. Enum `SummaryType` in `oscar/SleepLib/machine_common.h` (line 50):

```cpp
ST_CNT    // Event count
ST_AVG    // Simple average
ST_WAVG   // Time-weighted average
ST_MIN    // Minimum value
ST_MAX    // Maximum value
ST_PERC   // Percentile (usually 95th)
ST_CPH    // Count per hour
ST_SPH    // Sum per hour (percentage of time)
```

---

## Binary File Format

### File Types and Magic Number

All OSCAR binary files start with magic number `0xC73216AB`.

| Extension | File Type | Content |
|-----------|-----------|---------|
| .000 | Summary (type=0) | Session statistics, settings, channel summaries |
| .001 | Events (type=1) | Waveforms, respiratory events, time-series data |

### Header Structure

**Base header (32 bytes, all files):**
```
Offset  Size  Type      Field
0       4     uint32    Magic number (0xC73216AB)
4       2     uint16    Version
6       2     uint16    File type (0=summary, 1=events)
8       4     uint32    Machine ID
12      4     uint32    Session ID
16      8     int64     First timestamp (ms since epoch)
24      8     int64     Last timestamp (ms since epoch)
```

**Events file extended header (10 bytes, version ≥10):**
```
Offset  Size  Type      Field
32      2     uint16    Compression (0=none, 1=qCompress)
34      2     uint16    Machine type
36      4     int32     Uncompressed data size
40      2     uint16    CRC16 checksum
```

### Qt QDataStream Format

OSCAR uses Qt 4.6 QDataStream format with **little-endian** byte order.

**QString:**
- 4-byte length (in bytes), 0xFFFFFFFF = null
- UTF-16-LE encoded data

**QVariant:**
- 4-byte type code + 1-byte null flag + type-specific data
- Type codes: Bool=1, Int=2, UInt=3, LongLong=4, Double=6, String=10, ByteArray=12, DateTime=16

**QHash<K,V>:**
- 4-byte count, then count × (key, value) pairs

**QVector<T>:**
- 4-byte count, then packed array of T values

### Compression

**qCompress format:**
- 4 bytes: Uncompressed size (big-endian uint32)
- N bytes: zlib-compressed data

**CRC16:** CRC-16-CCITT polynomial (0x1021)

### EventList Storage

**Waveforms (EVL_Waveform):**
- Regularly sampled (sample_rate Hz)
- Data stored as int16 with gain/offset: `actual = (stored × gain) + offset`

**Events (EVL_Event):**
- Delta-encoded timestamps (uint32 array)
- Event type in low 4 bits (mask with 0x0F)

### Summary File Statistics

Version 18 summary files contain pre-calculated statistics per channel:
- counts, sums, averages, weighted_averages
- minimums, maximums, physical_minimums, physical_maximums
- counts_per_hour, sums_per_hour
- first_channel_time, last_channel_time
- value_summaries, time_summaries (histogram data)
- gains, available_channels

### Key Implementation Files

| File | Purpose |
|------|---------|
| `parsers/qdatastream.py` | Qt QDataStream binary format reader |
| `parsers/compression.py` | qCompress/qUncompress, CRC16, delta encoding |
| `parsers/oscar_summary.py` | .000 summary file parser |
| `parsers/oscar_events.py` | .001 events file parser |
| `constants.py:285` | `OSCAR_MAGIC_NUMBER = 0xC73216AB` |

---

## Analysis Implementation

### FlowParser - Breath Segmentation

**File:** `oscar/SleepLib/calcs.cpp` (lines 104-892)

**Algorithm:**
1. `openFlow()` (line 278) - Load flow waveform, apply gain, run filter chain
2. `calcPeaks()` (line 320) - Zero-crossing detection with hysteresis to identify breath boundaries
3. `calc()` (line 422) - Per-breath calculations:
   - Tidal volume: Integration of inspiratory flow
   - Ti/Te: Inspiration/expiration duration
   - Respiratory rate: Breath count in 60s sliding window
   - Minute ventilation: TV × RR

**Filters:**
- `percentileFilter()` (line 128) - Smoothing via percentile ranking
- `xpassFilter()` (line 188) - Configurable low/high-pass

### AHI Calculation

**Per-session AHI:** `oscar/SleepLib/calcs.cpp:981`
```cpp
float calcAHI(Session *session, qint64 start, qint64 end) {
    int count = countEvents(session, AHI_channels, start, end);
    float hours = (end - start) / 3600000.0;
    return count / hours;
}
```

**Sliding window AHI graph:** `oscar/SleepLib/calcs.cpp:1023`
- Configurable window (default 60 minutes)
- Recalculates AHI at each time step
- Output: Waveform channel for visualization

**Contributing channels:** `schema.cpp:420-424`
- CPAP_ClearAirway (central apnea)
- CPAP_AllApnea
- CPAP_Obstructive
- CPAP_Hypopnea
- CPAP_Apnea (unclassified)

SNORE's `AHI_EVENT_TYPES` (`constants.py`) counts the event types these channels map to on the ResMed EDF path, `CA`/`CAA`, `UA`, `OA` and `H`, and adds `MA`, which no parser emits today. OSCAR-binary import does not read the `CPAP_Apnea` or `CPAP_AllApnea` channels yet (#432). RERAs are not an AHI channel.

### Leak Calculation

**File:** `oscar/SleepLib/calcs.cpp` (lines 1288-1405)

**Classes:**
- `LeakCalculator` - Abstract base (line 1177)
- `LinearInterpolateLeak` - Linear interpolation between 4 and 20 cmH2O mask leak specs
- `ProfileLeakCalculator` - Uses user-configured leak rates

**Algorithm:**
```
Unintentional Leak = Total Leak - Mask Intentional Leak(pressure)
```

**Large leak flagging:** Threshold-based (redline setting, default 24 L/min)

### SpO2 Drop Detection

**File:** `oscar/SleepLib/calcs.cpp:1483`

**Algorithm:**
1. Calculate baseline SpO2 (rolling average, configurable window)
2. Detect drops ≥ threshold % from baseline
3. Flag events with duration ≥ minimum seconds

**Settings:** `oscar/SleepLib/profiles.h:515-523`
- `spO2DropPercentage` - Default 3%
- `spO2DropDuration` - Default 8 seconds
- `oxiDesaturationThreshold` - Absolute floor (default 88%)

### Pulse Change Detection

**File:** `oscar/SleepLib/calcs.cpp:1407`

Similar to SpO2 drop but for pulse rate changes (default: ≥5 BPM for ≥8 seconds).

### User Event Flagging

**File:** `oscar/SleepLib/calcs.cpp:737-880`

Allows users to define flow restriction events with configurable thresholds:
- Flow restriction percentage (e.g., 30%)
- Minimum duration (e.g., 10 seconds)
- Option to flag over existing machine-detected events

---

## Device Loaders

### Plugin Architecture

**Base class:** `MachineLoader` in `oscar/SleepLib/machine_loader.h`

**Key methods:**
```cpp
virtual int Detect(const QString &path) = 0;      // Confidence 0-100
virtual Machine *OpenMachine(QString &path) = 0;  // Create machine instance
virtual bool OpenSession(Session *session) = 0;   // Parse session data
```

**Registration:** `main.cpp:738-751` calls `Loader::Register()` for each manufacturer

### ResMed Implementation

**File:** `oscar/SleepLib/loader_plugins/resmed_loader.cpp`

**EDF files parsed:**
| File | Content | Sample Rate |
|------|---------|-------------|
| STR.edf | Settings/configuration | - |
| BRP.edf | Flow Rate waveform | 25 Hz |
| PLD.edf | Pressure, Leak, RR, MV, TV | 0.5 Hz (2s samples) |
| SA2.edf | SpO2, Pulse (if oximeter attached) | 1 Hz |
| EVE.edf | Events (apneas, hypopneas) | Annotation format |
| CSL.edf | Compliance/summary log | - |

**Detection:** Looks for DATALOG folder with STR.edf files (line 385)

**EDF parser:** `oscar/SleepLib/loader_plugins/edfparser.cpp`
- `EDFInfo` class handles EDF+ format
- Annotation parsing for events
- Signal conversion with gain/offset

### PRS1 Implementation (Philips)

**File:** `oscar/SleepLib/loader_plugins/prs1_loader.cpp`

**Binary formats:**
- .000 - Compliance summary
- .001 - Session summary
- .002 - Events
- .005/.006 - High-resolution waveforms
- PROP.TXT - Device configuration

**Event parsing:** Family/version-specific parsers (lines 567-594)
- `ParseEventsF0V23()` - 50-Series CPAP
- `ParseEventsF0V4()` - 60-Series CPAP
- `ParseEventsF0V6()` - DreamStation CPAP
- `ParseEventsF5V*()` - BiPAP AutoSV devices

**DreamStation 2 encryption:** `PRDS2File` class (line 267)
- AES-GCM decryption
- Salt-based key derivation with caching

---

## Graphing System

### Architecture

**Container hierarchy:**
```
gGraphView (QGraphicsView - OpenGL widget)
  └─ gGraph (QGraphicsScene - container)
      └─ Layer[] (gLayer subclasses - drawing primitives)
          ├─ gLineChart (time-series waveform)
          ├─ gSummaryChart (overview bars)
          ├─ gYAxis / gXAxis
          └─ EventFlags (event overlays)
```

**Files:**
- `oscar/Graphs/gGraphView.cpp` - OpenGL rendering widget
- `oscar/Graphs/gGraph.cpp` - Graph container and zoom/pan logic
- `oscar/Graphs/gLineChart.cpp` - Waveform line rendering
- `oscar/Graphs/gSummaryChart.cpp` - Bar chart rendering

### Zoom/Pan Implementation

**Mouse wheel zoom:** `gGraphView.cpp:688`
- Default: 0.75x (out) or 1.5x (in)
- Ctrl+wheel: 0.25x or 4x (fast zoom)
- 50-item zoom history with undo/redo

**Click-drag zoom:** Select region to zoom into

**Linked graphs:** All graphs on a page zoom/pan together

### Layer Types

**gLineChart** - Continuous waveform
- Adaptive rendering (skip points when zoomed out)
- Min/max/mean shading when downsampled
- Configurable line color, fill, thickness

**gSummaryChart** - Bar/candlestick charts
- Daily overview bars
- Stacked bars for event breakdowns
- Median/percentile markers

**EventFlags** - Event overlay markers
- Colored flags at event timestamps
- Configurable shapes (triangle, square, circle)
- Tooltip on hover

### Statistics Visualization

**gAHIChart** - `oscar/Graphs/gAHIChart.cpp`
- Per-day AHI bars with event type breakdown
- Median/average reference lines
- Tooltip shows per-event contributions

**gUsageChart** - Session duration bars

**Minutes at Pressure** - `oscar/Graphs/MinutesAtPressure.cpp`
- Pressure distribution histogram
- Time spent at each 0.5 cmH2O increment

---

## Statistics & Reporting

### Session-Level Statistics

**File:** `oscar/SleepLib/session.cpp`

**Methods:**
```cpp
Min(ChannelID id)                     // Minimum value (line 1371)
Max(ChannelID id)                     // Maximum value (line 1412)
avg(ChannelID id)                     // Simple average (line 2045)
wavg(ChannelID id)                    // Time-weighted average (line 2293)
percentile(ChannelID id, float p)     // Percentile (line 2234)
cph(ChannelID id)                     // Count per hour (line 2086)
timeAboveThreshold(...)               // Time above value (line 2114)
```

**UpdateSummaries():** `session.cpp:1208`
- Runs all analysis functions
- Builds value/time histograms
- Calculates min/max/avg/wavg for all channels

### Day-Level Statistics

**File:** `oscar/SleepLib/day.cpp`

Aggregates across multiple sessions in a day:
```cpp
Min() / Max()           // Min/max across sessions (line 941)
avg() / wavg()          // Averages (line 618, 656)
percentile()            // With time-weighting support (line 345)
cph() / sph()           // Count/sum per hour (line 1096, 1109)
total_time()            // Total with overlap handling (line 681)
```

**Configurable calculations:**
- Middle value: median, wavg, or avg (user preference)
- Max value: 99.5th percentile or absolute max

### Profile-Level Statistics

**File:** `oscar/SleepLib/profiles.cpp` (lines 1289-1718)

**Methods:**
```cpp
calcCount(ChannelID, date_from, date_to)
calcHours(ChannelID, date_from, date_to)
calcAvg(ChannelID, date_from, date_to)
calcWavg(ChannelID, date_from, date_to)
calcPercentile(ChannelID, percent, date_from, date_to)
```

Supports multi-period queries (week, month, 6 months, year).

### HTML Report Generation

**File:** `oscar/statistics.cpp`

**Statistics class:** (lines 206-252)
```cpp
GenerateHTML()              // Full statistics report
GenerateCPAPUsage()         // Usage statistics section
GenerateRXChanges()         // Settings change history
GenerateMachineList()       // Device list section
```

**Report includes:**
- Multi-period summaries (configurable date ranges)
- Event counts and indices (AHI, RDI, REI)
- Pressure/leak/SpO2 statistics
- Settings change timeline
- Days in/out of compliance

### CSV Export

**Structure:**
- Summary: Per-day aggregated statistics
- Sessions: Per-session data
- Detailed: Events with timestamps and session context

**Export dialog:** `oscar/reports.cpp` - User selects fields and date range

### Compliance Tracking

**Class:** `SummaryInfo` in `statistics.cpp:251-301`

**Metrics:**
- `daysInCompliance` - Days ≥ threshold hours (default: 4 hours)
- `daysOutOfCompliance` - Days < threshold
- `numDisabledsessions` - Sessions excluded from analysis

---

## Key Algorithms (Code References)

| Function | File:Line | Purpose |
|----------|-----------|---------|
| `calcAHI()` | oscar/SleepLib/calcs.cpp:981 | Per-session AHI calculation |
| `calcAHIGraph()` | oscar/SleepLib/calcs.cpp:1023 | Sliding window AHI waveform |
| `calcRespRate()` | oscar/SleepLib/calcs.cpp:894 | Orchestrates flow analysis pipeline |
| `FlowParser::calc()` | oscar/SleepLib/calcs.cpp:422 | Breath-by-breath metrics (TV, RR, MV) |
| `FlowParser::calcPeaks()` | oscar/SleepLib/calcs.cpp:320 | Zero-crossing breath segmentation |
| `calcLeaks()` | oscar/SleepLib/calcs.cpp:1288 | Unintentional leak calculation |
| `flagLargeLeaks()` | oscar/SleepLib/calcs.cpp:1342 | Large leak event detection |
| `calcSPO2Drop()` | oscar/SleepLib/calcs.cpp:1483 | SpO2 desaturation detection |
| `calcPulseChange()` | oscar/SleepLib/calcs.cpp:1407 | Pulse rate change detection |
| `flagUserEvents()` | oscar/SleepLib/calcs.cpp:737 | User-configurable event flagging |
| `Day::total_time()` | oscar/SleepLib/day.cpp:681 | Session overlap handling |
| `Session::UpdateSummaries()` | oscar/SleepLib/session.cpp:1208 | Run all analysis calculations |

---

## Notes for SNORE Development

**Architecture differences:**
- OSCAR: File-based storage with binary caching
- SNORE: SQLite database with ORM

**Data model mapping:**
- OSCAR Session → SNORE Session (1:1)
- OSCAR Day → SNORE Day (aggregation layer)
- OSCAR EventList → SNORE Waveform/Event tables
- OSCAR Summary types → SNORE Statistics table
- OSCAR Statistics page `FLG` row (`Profile::calcPercentile`) → FL and snore 95th percentiles in `GET /stats/periods` and `GET /stats/trends`, computed over the pooled waveform samples of a period's enabled sessions (`services/channel_stats.py`). Per-session and per-day median, 95th and max are in `GET /sessions/{id}` and `GET /days/{date}` as `channel_stats`.
- OSCAR mask-on slices → `mask_on_spans` in the same session and day responses, built from `Session.mask_on_segments`; null or empty when segments are unknown (OSCAR imports).

**Algorithm reuse opportunities:**
- FlowParser breath segmentation → SNORE BreathSegmenter (already implemented)
- calcSPO2Drop() → SNORE SpO2 drop detection (already implemented)
- calcLeaks() → SNORE could add unintentional leak calculation
- User event flagging → SNORE could add configurable detection

**What SNORE does differently:**
- Multiple detection modes (AASM, AASM Relaxed, ResMed) vs OSCAR's single configurable mode
- 7-class flow limitation classification vs OSCAR's basic flagging
- CSR/periodic breathing detection vs device-reported only
- Programmatic event detection vs primarily device-reported
