# Apnea Detection Algorithm Reference Guide

## Executive Summary

This document provides comprehensive guidelines for detecting respiratory events (apneas, hypopneas, and RERAs) from raw CPAP/PAP device data. Based on industry standards, clinical research, and reverse engineering of commercial CPAP algorithms, this guide establishes best practices for programmatic event detection.

## 1. Industry Standards and Clinical Definitions

### 1.1 AASM Scoring Manual Standards (2023)

The American Academy of Sleep Medicine (AASM) defines the following criteria:

#### Obstructive Apnea (OA)
- **Flow Reduction**: ≥90% drop in peak signal excursion from baseline
- **Duration**: ≥10 seconds
- **Effort**: Continued respiratory effort throughout the event
- **Clinical Note**: At least 90% of the event duration must meet amplitude reduction criteria

#### Central Apnea (CA) / Clear Airway (CAA)
- **Flow Reduction**: ≥90% drop in peak signal excursion from baseline
- **Duration**: ≥10 seconds
- **Effort**: Absent respiratory effort throughout the event
- **Clinical Note**: No chest/abdominal movement in PSG; flat flow signal in CPAP

#### Hypopnea
- **Flow Reduction**: ≥30% drop in peak signal excursion from baseline
- **Duration**: ≥10 seconds
- **Associated Features**: Must have EITHER:
  - ≥3% oxygen desaturation from pre-event baseline, OR
  - Event associated with an arousal
- **Medicare Alternative**: ≥4% desaturation required (more stringent)

#### Mixed Apnea (MA)
- **Flow Reduction**: ≥90% drop in peak signal excursion from baseline
- **Duration**: ≥10 seconds
- **Effort**: Absent respiratory effort initially, resuming during latter portion

#### RERA (Respiratory Effort-Related Arousal)
- **Flow Pattern**: Flattening of inspiratory flow curve
- **Duration**: ≥10 seconds
- **Termination**: Must end with an arousal
- **Flow Reduction**: <30% (does not meet hypopnea criteria)

### 1.2 Key Threshold Values

| Event Type | Flow Reduction | Duration | Additional Criteria |
|------------|---------------|----------|---------------------|
| Apnea | ≥90% | ≥10 sec | Effort determines OA vs CA |
| Hypopnea | 30-89% | ≥10 sec | Requires SpO2 drop or arousal |
| RERA | <30% | ≥10 sec | Flow limitation + arousal |

## 2. CPAP Device Detection Methods

### 2.1 ResMed AirSense Algorithm Characteristics

Based on patent analysis and reverse engineering:

#### Flow Signal Processing
- **Sampling Rate**: 25 Hz for flow data
- **Filtering**: Low-pass filter at 2-3 Hz to remove cardiac oscillations
- **Baseline Calculation**:
  - Uses 2-minute rolling window of stable breathing
  - Excludes periods with detected events
  - Updates every 30 seconds during stable breathing

#### Event Detection Logic
1. **Breath Detection**: Zero-crossing detection with hysteresis
2. **Amplitude Calculation**: Peak-to-peak within each breath cycle
3. **Baseline Comparison**: Compare breath amplitude to baseline
4. **Event Marking**: Flag when consecutive breaths fall below threshold
5. **Event Termination**: When 2+ breaths exceed 50% of baseline

### 2.2 Philips Respironics DreamStation Methods

#### Key Differences
- Uses "moving baseline" updated every breath
- Implements "breath-by-breath" scoring
- More sensitive to flow limitation patterns
- Uses proprietary "Shape Signal" for effort detection

## 3. Critical Algorithm Components

### 3.1 Baseline Flow Calculation

#### INCORRECT Methods (Common Mistakes)
```python
# WRONG: Using median of all positive flow
baseline = np.median(flow[flow > 0])  # Includes apneas!

# WRONG: Using 75th percentile of all data
baseline = np.percentile(flow[flow > 0], 75)  # Still includes events!

# WRONG: Using fixed baseline for entire session
baseline = calculate_once_at_start()  # Doesn't adapt to changes
```

#### CORRECT Implementation
```python
def calculate_rolling_baseline(flow, timestamps, window_seconds=120):
    """
    Calculate baseline using recent stable breathing periods.

    Window: 2 minutes of recent data
    Method: 90th percentile of peak inspiratory flows
    Exclusions: Periods with already-detected events
    """
    baseline = []
    for i, t in enumerate(timestamps):
        # Get 2-minute window before current time
        window_mask = (timestamps >= t - window_seconds) & (timestamps < t)
        window_flow = flow[window_mask]

        # Find peak inspiratory flows (local maxima)
        peaks = find_inspiratory_peaks(window_flow)

        if len(peaks) > 10:  # Need sufficient breaths
            # Use 90th percentile of peaks (robust to outliers)
            baseline.append(np.percentile(peaks, 90))
        else:
            # Insufficient data, use last valid baseline
            baseline.append(baseline[-1] if baseline else 30.0)

    return np.array(baseline)
```

### 3.2 Flow Reduction Detection

#### Breath-by-Breath Analysis (Recommended)
```python
def detect_flow_reduction(flow, timestamps, baseline):
    """
    Detect flow reduction using breath-by-breath amplitude.
    """
    # Segment into individual breaths
    breaths = segment_breaths(flow, timestamps)

    reductions = []
    for breath in breaths:
        # Calculate peak-to-peak amplitude
        amplitude = np.max(breath.flow) - np.min(breath.flow)

        # Compare to baseline at breath midpoint
        baseline_at_breath = baseline[breath.mid_index]

        # Calculate reduction percentage
        reduction = 1.0 - (amplitude / baseline_at_breath)
        reduction = np.clip(reduction, 0, 1)

        reductions.append({
            'time': breath.mid_time,
            'reduction': reduction,
            'amplitude': amplitude
        })

    return reductions
```

#### Moving Window Method (Alternative)
```python
def detect_flow_reduction_window(flow, baseline, window_size=10):
    """
    Use 10-second moving window for flow reduction.
    """
    # Calculate RMS or peak-to-peak in sliding windows
    window_samples = int(window_size * sample_rate)

    reductions = []
    for i in range(len(flow) - window_samples):
        window = flow[i:i + window_samples]

        # Method 1: RMS amplitude
        rms = np.sqrt(np.mean(window ** 2))

        # Method 2: Peak-to-peak
        p2p = np.max(window) - np.min(window)

        # Use peak-to-peak (more robust)
        reduction = 1.0 - (p2p / baseline[i])
        reductions.append(reduction)

    return reductions
```

### 3.3 Distinguishing OA from CA

#### Without Effort Belts (CPAP-only detection)
```python
def classify_apnea_type(flow_during_event):
    """
    Classify apnea using flow characteristics alone.

    OA: Continued effort creates flow oscillations
    CA: No effort results in flat, stable flow
    """
    # Detrend the signal
    detrended = flow_during_event - np.mean(flow_during_event)

    # Calculate variability metrics
    std_dev = np.std(detrended)
    peak_to_peak = np.ptp(detrended)

    # Calculate "roughness" - variation between samples
    roughness = np.mean(np.abs(np.diff(detrended)))

    # Calculate spectral power in breathing range (0.1-0.5 Hz)
    if len(flow_during_event) > 50:
        freqs, power = scipy.signal.periodogram(detrended, fs=25)
        breathing_power = np.sum(power[(freqs >= 0.1) & (freqs <= 0.5)])
    else:
        breathing_power = 0

    # Scoring (empirically derived thresholds)
    effort_score = (
        std_dev * 0.3 +
        peak_to_peak * 0.3 +
        roughness * 0.2 +
        breathing_power * 0.2
    )

    if effort_score > 0.15:
        return "OA"  # Obstructive (effort present)
    elif effort_score < 0.05:
        return "CA"  # Central (no effort)
    else:
        return "MA"  # Mixed or uncertain
```

### 3.4 Event Duration and Merging

#### Duration Calculation
```python
def validate_event_duration(event_mask, timestamps, min_duration=10.0):
    """
    Ensure events meet minimum duration requirements.

    AASM: ≥10 seconds required
    90% rule: At least 90% of event must meet flow criteria
    """
    # Find continuous regions
    events = []
    start_indices = np.where(np.diff(np.concatenate([[0], event_mask, [0]])) == 1)[0]
    end_indices = np.where(np.diff(np.concatenate([[0], event_mask, [0]])) == -1)[0]

    for start, end in zip(start_indices, end_indices):
        duration = timestamps[end-1] - timestamps[start]

        if duration >= min_duration:
            # Check 90% rule
            samples_meeting_criteria = np.sum(event_mask[start:end])
            total_samples = end - start

            if samples_meeting_criteria / total_samples >= 0.9:
                events.append((start, end, duration))

    return events
```

#### Event Merging
```python
def merge_adjacent_events(events, max_gap=3.0):
    """
    Merge events separated by brief recoveries.

    Clinical practice: Events separated by <3 seconds
    are often scored as single event.
    """
    if not events:
        return events

    merged = [events[0]]

    for current in events[1:]:
        previous = merged[-1]
        gap = current.start_time - previous.end_time

        if gap <= max_gap and current.type == previous.type:
            # Merge events
            merged[-1] = Event(
                start_time=previous.start_time,
                end_time=current.end_time,
                type=current.type,
                # Recalculate metrics for merged event
            )
        else:
            merged.append(current)

    return merged
```

## 4. Common Implementation Pitfalls

### 4.1 Threshold Confusion
- **ERROR**: Using 70% or 80% threshold for apnea (too lenient)
- **CORRECT**: Use 90% reduction for apnea, 30% for hypopnea minimum

### 4.2 Baseline Errors
- **ERROR**: Including apnea periods in baseline calculation
- **ERROR**: Using instantaneous flow instead of breath amplitude
- **ERROR**: Fixed baseline for entire night

### 4.3 Signal Processing Issues
- **ERROR**: Not filtering cardiac oscillations (1-2 Hz)
- **ERROR**: Using raw flow without leak compensation
- **ERROR**: Incorrect sampling rate assumptions

### 4.4 Event Validation
- **ERROR**: Accepting events <10 seconds
- **ERROR**: Not checking the 90% amplitude rule
- **ERROR**: Over-merging distinct events

## 5. Implementation Checklist

### Required Components
- [ ] Breath segmentation using zero-crossing detection
- [ ] Rolling baseline calculation (2-minute window)
- [ ] Breath-by-breath amplitude measurement
- [ ] 90% flow reduction threshold for apnea
- [ ] 30-90% range for hypopnea
- [ ] 10-second minimum duration enforcement
- [ ] 90% rule validation (90% of event meets criteria)
- [ ] OA vs CA classification using flow variability
- [ ] Adjacent event merging (3-second gap)
- [ ] Confidence scoring for each detection

### Recommended Additions
- [ ] Leak compensation before analysis
- [ ] Cardiac oscillation filtering (2-3 Hz low-pass)
- [ ] Positional data correlation (if available)
- [ ] SpO2 integration for hypopnea confirmation
- [ ] Arousal detection from flow patterns
- [ ] Flow limitation shape analysis for RERA

## 6. Validation Methods

### 6.1 Compare with Machine Flags
```python
def validate_detection(detected_events, machine_events, tolerance=5.0):
    """
    Compare programmatic detection with machine flags.

    Tolerance: Events within 5 seconds considered matches.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    matched_machine = set()

    for detected in detected_events:
        matched = False
        for i, machine in enumerate(machine_events):
            if i in matched_machine:
                continue

            time_diff = abs(detected.start_time - machine.start_time)
            if time_diff <= tolerance:
                true_positives += 1
                matched_machine.add(i)
                matched = True
                break

        if not matched:
            false_positives += 1

    false_negatives = len(machine_events) - len(matched_machine)

    sensitivity = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)

    return {
        'sensitivity': sensitivity,
        'precision': precision,
        'f1_score': 2 * (precision * sensitivity) / (precision + sensitivity)
    }
```

### 6.2 Visual Validation
- Plot detected events over flow waveform
- Compare with machine event markers
- Verify baseline calculation is reasonable
- Check that flow reduction percentages are correct

## 7. Advanced Techniques

### 7.1 Machine Learning Enhancements
- Train classifier on validated event/non-event segments
- Use features: flow stats, frequency content, shape metrics
- Ensemble with rule-based detection for robustness

### 7.2 Flow Limitation Detection
- Analyze inspiratory flow shape (flattening index)
- Calculate ratio of mid-inspiratory to peak flow
- Detect "chair-shaped" inspiratory curves

### 7.3 Adaptive Thresholds
- Adjust thresholds based on device type
- Account for altitude effects on flow
- Personalize based on patient baseline patterns

## 8. Testing Recommendations

### 8.1 Test Dataset Requirements
- Sessions with known machine-detected events
- Mix of OA, CA, and hypopnea events
- Various severity levels (mild to severe)
- Different CPAP devices if possible

### 8.2 Performance Targets
- Sensitivity (recall): >85% for apneas, >70% for hypopneas
- Precision: >80% for apneas, >60% for hypopneas
- F1 Score: >0.80 for apneas, >0.65 for hypopneas

### 8.3 Edge Cases to Test
- Events at session start/end
- Back-to-back events
- Very long events (>60 seconds)
- Periodic breathing patterns
- High leak periods

## 9. References and Resources

### Clinical Guidelines
- AASM Manual for the Scoring of Sleep and Associated Events v2.6 (2023)
- CMS Medicare Coverage Guidelines for PAP Devices
- International Classification of Sleep Disorders (ICSD-3)

### Technical Resources
- ResMed Patents: US9999386B2 (flow analysis), US10449312B2 (event detection)
- Philips Patents: US9192336B2 (shape signal), US10130783B2 (auto-titration)
- OSCAR Source Code: github.com/oscar-team/oscar (data format parsing)
- SleepyHead Legacy Code: Event detection implementations

### Academic Papers
- Berry et al. (2012): "Rules for Scoring Respiratory Events in Sleep"
- Collop et al. (2011): "Portable Monitoring Task Force Recommendations"
- Masa et al. (2013): "Alternative Methods of Titrating CPAP"

## 10. Conclusion

Successful apnea detection from CPAP data requires:
1. Correct threshold values (90% for apnea, not 70%)
2. Proper baseline calculation (excluding events, using recent data)
3. Breath-by-breath analysis (not instantaneous flow)
4. Robust event validation (duration, 90% rule)
5. Careful signal processing (filtering, leak compensation)

Following these guidelines should achieve detection performance comparable to commercial CPAP devices while enabling additional analyses like flow limitation detection that devices don't provide.