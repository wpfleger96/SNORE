export interface GlossaryEntry {
    label: string
    short: string // one-sentence explanation
    long?: string // optional fuller detail
}

export const GLOSSARY: Record<string, GlossaryEntry> = {
    session_duration_hours: {
        label: 'Duration',
        short: 'Total hours of CPAP therapy recorded in this session.',
    },
    total_breaths: {
        label: 'Total Breaths',
        short: 'Number of detected complete breath cycles during the session.',
    },
    machine_events: {
        label: 'Device-scored Events',
        short: 'Respiratory events scored by the CPAP device firmware in real time.',
        long: 'Device-scored events use proprietary device algorithms and may differ from SNORE-detected events.',
    },
    pulse_change_count: {
        label: 'Pulse Changes',
        short: 'Number of pulse-rate change events detected, used as arousal markers.',
    },
    programmatic_events: {
        label: 'SNORE-detected Events',
        short: "Respiratory events detected by SNORE's own analysis algorithms from the raw flow signal.",
    },
    ahi: {
        label: 'AHI',
        short: 'Apnea-Hypopnea Index: total apneas and hypopneas per hour of therapy.',
        long: "A night's headline AHI (and OAI, CAI, HI) is the device-reported daily value when SNORE can trust it: the night has no disabled sessions, every enabled session carries the same device indices and device mask-on time, and SNORE's imported hours are within the larger of 5 minutes or 5% of the device's mask-on time. Otherwise it is SNORE's recount: device-scored events divided by SNORE's mask-on hours. A session's AHI is always the recount. Period and trend AHIs are usage-hours-weighted averages of the nightly headline values, so they can mix device-reported and recounted nights. Common clinical thresholds: <5 normal, 5–15 mild, 15–30 moderate, >30 severe.",
    },
    mode_ahi: {
        label: 'Mode AHI',
        short: "SNORE-detected apneas and hypopneas per hour, from this detection mode's own analysis of the flow waveform.",
        long: "An experimental SNORE heuristic, not the device's AHI; compare it against the device-scored events to judge the mode.",
    },
    rdi: {
        label: 'RDI',
        short: "Respiratory Disturbance Index: this detection mode's SNORE-detected apneas, hypopneas, and RERAs per hour.",
        long: "RDI is the mode's AHI plus its RERAs per hour, so it is never below that mode's AHI. A large gap suggests airway effort and arousals without frank apneas.",
    },
    rei: {
        label: 'REI',
        short: 'Respiratory Event Index: respiratory events per hour of recorded device time (used when sleep time is not measured).',
    },
    oai: {
        label: 'OAI',
        short: 'Obstructive Apnea Index: obstructive apneas per hour of therapy.',
    },
    cai: {
        label: 'CAI',
        short: 'Central Apnea Index: central apneas per hour of therapy.',
    },
    hi: {
        label: 'HI',
        short: 'Hypopnea Index: hypopneas per hour of therapy.',
    },
    rera: {
        label: 'RERA Index',
        short: 'Respiratory Effort-Related Arousals per hour of therapy.',
    },
    apneas: {
        label: 'Apneas',
        short: 'Total apnea events: obstructive, central, and mixed combined.',
    },
    obstructive_apneas: {
        label: 'Obstructive Apneas',
        short: 'Airway blocked by soft-tissue collapse while breathing effort continues; airflow drops ≥90% for ≥10 s.',
    },
    central_apneas: {
        label: 'Central Apneas',
        short: 'Breathing pauses where the brain stops signaling breathing effort; no airflow and no effort for ≥10 s.',
    },
    mixed_apneas: {
        label: 'Mixed Apneas',
        short: 'Apneas that begin central (no effort) and end obstructive (effort against a blocked airway).',
    },
    hypopneas: {
        label: 'Hypopneas',
        short: 'Partial airway obstructions with roughly ≥30% flow reduction lasting ≥10 s.',
    },
    reras: {
        label: 'RERAs',
        short: 'Respiratory Effort-Related Arousals: flow-limited breathing that causes an arousal without meeting apnea or hypopnea criteria.',
    },
    flow_limitations: {
        label: 'Flow Limitations',
        short: 'Breaths or periods where the inspiratory airflow shape is flattened by partial airway narrowing.',
    },
    fl_class_ge4_pct: {
        label: 'FL Class ≥4',
        short: 'Share of confidently-classified breaths with a flow-limitation class of 4 or higher (more flattened inspiratory shapes).',
        long: "From SNORE's experimental breath analysis, not the device. Useful for night-to-night trends, not a clinically validated measurement.",
    },
    rera_index: {
        label: 'RERA Proxy Index',
        short: "Estimated respiratory effort-related arousals per hour, from SNORE's flow-limitation-run RERA proxy.",
        long: "SNORE's experimental breath analysis; useful for night-to-night trends, not a clinically validated measurement. Distinct from the device RERA count and the analysis-time RERA detector.",
    },
    rera_count: {
        label: 'RERA Proxy Count',
        short: "Total respiratory effort-related arousals detected across the night by SNORE's flow-limitation-run RERA proxy (a count, not a per-hour rate).",
        long: "SNORE's experimental breath analysis; useful for night-to-night trends, not a clinically validated measurement. Distinct from the device RERA count and the analysis-time RERA detector.",
    },
    event_type_oa: {
        label: 'OA — Obstructive Apnea',
        short: 'Airway blocked by soft-tissue collapse; airflow drops ≥90% for ≥10 s while breathing effort continues.',
    },
    event_type_ca: {
        label: 'CA — Central Apnea',
        short: 'Brain stops signaling breathing effort; no airflow and no effort for ≥10 s.',
    },
    event_type_ma: {
        label: 'MA — Mixed Apnea',
        short: 'Begins as a central apnea (no effort) and ends obstructive (effort against a blocked airway).',
    },
    event_type_h: {
        label: 'H — Hypopnea',
        short: 'Partial airway obstruction with roughly ≥30% flow reduction lasting ≥10 s.',
    },
    event_type_re: {
        label: 'RE — RERA',
        short: 'Flow-limited breathing that causes an arousal without meeting apnea or hypopnea criteria.',
    },
    event_type_fl: {
        label: 'FL — Flow Limitation',
        short: 'Inspiratory airflow shape flattened by partial airway narrowing without reaching apnea/hypopnea thresholds.',
    },
    pressure: {
        label: 'Pressure',
        short: 'Therapy air pressure delivered by the device, in cmH₂O.',
        long: 'Min/median/95th-percentile/max variants summarize the pressure distribution across the night.',
    },
    epap: {
        label: 'EPAP',
        short: 'Expiratory Positive Airway Pressure: pressure delivered during exhalation, in cmH₂O.',
    },
    ipap: {
        label: 'IPAP',
        short: 'Inspiratory Positive Airway Pressure: pressure delivered during inhalation, in cmH₂O (bilevel modes).',
    },
    leak: {
        label: 'Leak',
        short: 'Unintentional air leak rate from the mask, in L/min.',
        long: 'Sustained large leaks reduce therapy effectiveness and can hide events from detection. Percentile variants summarize the night’s leak distribution.',
    },
    spo2: {
        label: 'SpO₂',
        short: 'Blood oxygen saturation from the pulse oximeter, in percent.',
    },
    spo2_drop: {
        label: 'SpO₂ Drop',
        short: 'Decrease in blood oxygen saturation during the event window.',
    },
    spo2_below_90: {
        label: 'SpO₂ Below 90%',
        short: 'Total time with oxygen saturation under 90%.',
    },
    pulse: {
        label: 'Pulse',
        short: 'Heart rate from the pulse oximeter, in beats per minute.',
    },
    resp_rate: {
        label: 'Respiratory Rate',
        short: 'Breaths per minute.',
    },
    tidal_volume: {
        label: 'Tidal Volume',
        short: 'Estimated air volume moved per breath, in mL.',
    },
    mv: {
        label: 'Minute Ventilation',
        short: 'Total air volume breathed per minute (tidal volume × respiratory rate), in L/min.',
    },
    peak_fl: {
        label: 'Peak Flow Limitation',
        short: 'Highest flow-limitation score recorded during the event window (0–1 scale).',
    },
    usage: {
        label: 'Usage',
        short: 'Hours of therapy use.',
    },
    flow_reduction: {
        label: 'Flow Reduction',
        short: 'Estimated fraction of baseline inspiratory flow lost during the event.',
    },
    confidence: {
        label: 'Confidence',
        short: 'Algorithm confidence that this event meets its classification criteria.',
        long: 'Events near classification thresholds get lower confidence and warrant more skepticism.',
    },
    avg_confidence: {
        label: 'Avg Confidence',
        short: 'Average breath-classification confidence across all breaths in the session.',
    },
    csr: {
        label: 'Cheyne-Stokes Respiration',
        short: 'Cyclical waxing-and-waning breathing with central pauses, associated with cardiac or neurological conditions.',
    },
    periodic_breathing: {
        label: 'Periodic Breathing',
        short: 'Repeating cycles of breathing and pauses; broader than CSR and also seen at altitude.',
    },
    flow_limitation_index: {
        label: 'Flow Limitation Index',
        short: 'Severity-weighted share of breaths showing flow limitation.',
        long: "Each breath's class weight (0.0 for Class 1 up to 1.0 for Class 7) is multiplied by its classification confidence, then averaged across all breaths.",
    },
    false_negatives: {
        label: 'False Negatives',
        short: "Device-scored events that SNORE's analysis did not detect.",
    },
    false_positives: {
        label: 'False Positives',
        short: "SNORE-detected events absent from the device's event log.",
        long: 'May be real events the device missed, or over-detections by the algorithm.',
    },
    days_with_data: {
        label: 'Days with Data',
        short: 'Number of days in the period with therapy hours from enabled sessions.',
    },
    effectiveness: {
        label: 'Effectiveness',
        short: 'Overall therapy quality rating derived from AHI: excellent, good, fair, or poor.',
    },
    ahi_trend: {
        label: 'AHI Trend',
        short: 'Direction of recent AHI change: improving, worsening, or stable.',
    },
    sensitivity: {
        label: 'Sensitivity',
        short: 'Share of device-scored events that SNORE also detected (true-positive rate / recall).',
    },
    precision: {
        label: 'Precision',
        short: 'Share of SNORE-detected events that match a device-scored event.',
    },
    f1: {
        label: 'F1 Score',
        short: 'Harmonic mean of sensitivity and precision; balances missed events against over-detection.',
    },

    // ── Validation: signal-correlation & experimental-metric axes ─────────────
    spearman_r: {
        label: 'Spearman r',
        short: 'Rank correlation between a SNORE per-breath metric and the device signal it is validated against, from -1 to +1.',
        long: 'Spearman r measures monotonic agreement of rankings rather than absolute values, so it is robust to the smoothing and unit differences between SNORE and device channels. Values near +1 mean the two rank nights or breaths the same way.',
    },
    auc: {
        label: 'AUC',
        short: 'Area under the ROC curve: how well a SNORE score separates device-flagged flow-limited breaths from the rest (0.5 = chance, 1.0 = perfect).',
        long: 'AUC25 and AUC50 are the same measure taken at two device FLG operating points — discriminating breaths at device FLG ≥ 0.25 and ≥ 0.50 respectively. Higher thresholds isolate more severely flow-limited breaths.',
    },
    chance_floor: {
        label: 'Chance Precision Floor',
        short: 'The precision a random detector firing at the same density would reach by chance alone.',
        long: 'Computed as the pooled device-scored RE rate per second × (2 × match tolerance). Measured precision at or below this floor is indistinguishable from chance given how often the proxy fires — it is context, not a signal of failure.',
    },
    rera_proxy: {
        label: 'RERA Proxy',
        short: "SNORE's experimental FL-run RERA proxy: runs of ≥2 consecutive flow-limited breaths ending in a recovery breath.",
        long: 'Fires far more often than the device scores RE (which ResMed does very conservatively), so near-zero precision against device-scored RE is expected. Useful as an internally-consistent trend instrument, not a validated absolute count.',
    },
    apple_breathing_disturbances: {
        label: 'Apple Breathing Disturbances',
        short: "Apple Watch's sleeping breathing-disturbance metric — a genuinely independent second axis for the SNORE indices.",
        long: 'Derived from wrist sensors during sleep, independent of the ResMed device. A positive rank correlation with the SNORE RERA/FL indices is weak external evidence they track real respiratory disturbance.',
    },
    cross_night_spearman: {
        label: 'Cross-night Spearman',
        short: "Rank correlation of SNORE's nightly 95th-percentile FL against the device's nightly 95th-percentile FLG, across nights.",
        long: 'A night-level agreement check: even when per-breath alignment is noisy, nights the device ranks as more flow-limited should rank higher for SNORE too.',
    },

    // ── New device-channel labels ──────────────────────────────────────────
    fl_device: {
        label: 'Flow Limitation (device)',
        short: "ResMed's proprietary per-breath severity index for flow limitation, 0 (none) to 1 (severe).",
        long: "This is distinct from SNORE's computed flow-limitation classes. The device reports a continuous 0–1 score derived from its own internal algorithm; SNORE's FL classes are based on inspiratory flow-shape analysis.",
    },
    snore_device: {
        label: 'Snore (device)',
        short: 'ResMed device snore index, 0 (absent) to 5 (severe), sampled once per breath.',
        long: 'A unitless severity score derived from the high-frequency vibration component of mask pressure. Not equivalent to decibel snore measurements.',
    },
    ie_ratio_waveform: {
        label: 'I:E Ratio',
        short: 'Instantaneous inspiratory-to-expiratory time ratio (expressed as a percentage, where 100 = 1:1).',
    },
    ti_waveform: {
        label: 'Inspiratory Time',
        short: 'Duration of the inspiratory phase of each breath, in seconds.',
    },
    pressure_hr_waveform: {
        label: 'Mask Pressure (25 Hz)',
        short: '25 Hz mask-pressure signal providing finer time resolution than the standard 2 Hz pressure channel.',
        long: 'Useful for detecting brief snore vibrations and inspiratory flow-limitation shapes that are averaged away in the lower-resolution channel.',
    },
    trigger_cycle: {
        label: 'Trigger/Cycle (raw codes)',
        short: 'Raw numeric event codes (0–16) logged by the device firmware for breath trigger and cycle transitions.',
        long: 'These are undecoded manufacturer-internal event codes. They are stored as-is and have not been mapped to named states. Consult ResMed documentation or OSCAR source for code-to-state mappings.',
    },

    // ── STR daily statistics ───────────────────────────────────────────────
    uai: {
        label: 'UAI',
        short: 'Unintentional Apnea Index reported by the device firmware; apneas per hour (device-computed).',
    },
    ai_str: {
        label: 'AI',
        short: 'Apnea Index reported by the device; total apneas (OA + CA) per hour of therapy.',
    },
    rin: {
        label: 'RIN',
        short: 'Respiratory Intensity Index; device-reported measure of respiratory event intensity (events/hr).',
    },
    spont_cyc_pct: {
        label: 'Spont Cyc %',
        short: 'Percentage of breaths that cycled spontaneously (patient-triggered expiration) in VAuto/bilevel modes.',
    },
    mask_events_str: {
        label: 'Mask Events',
        short: 'Number of mask-on events logged by the device during the session.',
    },
    flow_5th: {
        label: 'Flow 5th Percentile',
        short: '5th percentile of instantaneous flow. Typically negative because expiratory flow is represented as a negative value.',
        long: 'This value is expected to be negative for healthy breathing — it represents the lower tail of the flow distribution where expiratory flow dominates. A large negative value is normal, not an error.',
    },
    ie_ratio_stat: {
        label: 'I:E Ratio',
        short: 'Daily I:E ratio percentile from the STR summary (percent; 100 = 1:1, 50 = 1:2). VAuto devices only.',
        long: 'Values above 100 indicate inspiration longer than expiration (I > E), which is normal on bilevel and VAuto devices.',
    },
    ti_stat: {
        label: 'Inspiratory Time (Ti)',
        short: 'Inspiratory phase duration percentile from the STR summary, in seconds. VAuto devices only.',
    },
    amb_humidity: {
        label: 'Ambient Humidity',
        short: 'Median ambient relative humidity recorded by the device humidifier sensor, in percent.',
    },
    hum_temp: {
        label: 'Humidifier Temperature',
        short: 'Median humidifier chamber temperature, in °C.',
    },

    // ── Blower-side flow & pressure (STR percentiles) ─────────────────────
    blow_press: {
        label: 'Blower Pressure',
        short: 'Pressure measured at the blower output, upstream of the humidifier and tubing, in cmH₂O.',
        long: 'Differs from mask pressure: blower pressure is higher because it has not yet dropped across the hose and humidifier. Percentile variants summarize the nightly distribution.',
    },
    blow_flow: {
        label: 'Blower Flow',
        short: 'Airflow measured at the blower output, in L/min. Includes both patient flow and intentional leak.',
    },

    // ── Climate & humidifier power ────────────────────────────────────────
    htube_temp: {
        label: 'Heated Tube Temperature',
        short: 'Median heated-tube (ClimateLine) temperature, in °C.',
    },
    htube_pow: {
        label: 'Heated Tube Power',
        short: 'Median power delivered to the heated tube, in watts.',
    },
    hum_pow: {
        label: 'Humidifier Power',
        short: 'Median power delivered to the humidifier heater plate, in watts.',
    },

    // ── Apple Health sleep metrics ────────────────────────────────────────
    time_in_bed: {
        label: 'Time in Bed',
        short: 'Total time in bed, in hours — from recorded InBed samples when present, or derived from the sleep stage session (asleep + awake) on exports where the OS no longer emits InBed records.',
    },
    total_sleep: {
        label: 'Total Sleep',
        short: 'Total time actually asleep (Core + Deep + REM stages combined), in hours.',
    },
    sleep_efficiency: {
        label: 'Sleep Efficiency',
        short: 'Total sleep divided by time in bed, expressed as a percentage.',
        long: 'Sleep efficiency = total sleep / time in bed × 100. Values below 85% may indicate difficulty falling or staying asleep.',
    },
    core_sleep: {
        label: 'Core Sleep',
        short: "Apple Health's Core stage corresponds to NREM N1 and N2 light sleep combined.",
        long: "Core sleep (N1 + N2) is the most common stage and forms the backbone of each sleep cycle. Apple Health labels light non-REM sleep as 'Core'.",
    },
    deep_sleep: {
        label: 'Deep Sleep',
        short: 'NREM N3 slow-wave sleep — the most restorative stage.',
        long: 'Deep sleep supports physical repair and immune function. It is most concentrated in the first half of the night and decreases with age.',
    },
    rem_sleep: {
        label: 'REM Sleep',
        short: 'Rapid Eye Movement sleep, associated with dreaming and memory consolidation.',
    },
    awake_time: {
        label: 'Awake',
        short: 'Time spent awake after initial sleep onset, as detected by Apple Health.',
    },

    // ── Primary waveform channels ─────────────────────────────────────────
    flow: {
        label: 'Flow Rate',
        short: 'Bidirectional inspiratory/expiratory airflow in L/min, sampled at 25 Hz.',
        long: 'Positive values are inspiratory; negative values are expiratory. This is the primary signal used for breath detection and event scoring.',
    },
    therapy_pressure: {
        label: 'Therapy Pressure',
        short: 'Therapy-algorithm target pressure in cmH₂O, reported as a 0.5 Hz duty-cycle average.',
        long: "Distinct from the mask-side 'Pressure' channel: this is the algorithm's commanded set point rather than the measured delivered pressure.",
    },

    // ── Device settings (Equipment page) ──────────────────────────────────
    // The setting_ prefix namespaces these entries so they never collide with the
    // measured-channel entries above (ipap, epap, pressure). Each label mirrors
    // SETTING_LABELS in deviceSettings.ts (kept in sync by a unit test).
    setting_mode: {
        label: 'Mode',
        short: 'The therapy mode the device runs, which determines how pressure is delivered.',
        long: 'CPAP holds a single fixed pressure, APAP auto-adjusts within a range, and the bilevel modes (S, VAuto, ST) alternate between two pressure levels. The mode decides which of the other pressure settings actually apply.',
    },
    setting_pressure_fixed: {
        label: 'Pressure',
        short: 'The single fixed pressure delivered all night in CPAP mode, in cmH₂O.',
    },
    setting_pressure_min: {
        label: 'Min Pressure',
        short: 'The lower bound of the auto-adjusting pressure range in APAP mode, in cmH₂O.',
        long: 'The device will not drop below this pressure; it is the baseline it settles to when your airway is stable.',
    },
    setting_pressure_max: {
        label: 'Max Pressure',
        short: 'The upper bound of the auto-adjusting pressure range in APAP mode, in cmH₂O.',
        long: 'The device raises pressure toward this ceiling in response to obstructive events but never exceeds it.',
    },
    setting_ipap: {
        label: 'IPAP',
        short: 'Inspiratory Positive Airway Pressure: the pressure delivered while breathing in, in cmH₂O.',
        long: 'The higher of the two bilevel pressures. It supports each breath and, together with EPAP, sets the amount of pressure support.',
    },
    setting_epap: {
        label: 'EPAP',
        short: 'Expiratory Positive Airway Pressure: the pressure delivered while breathing out, in cmH₂O.',
        long: 'The lower of the two bilevel pressures. It holds the airway open between breaths and is the main defense against obstructive apneas.',
    },
    setting_ps: {
        label: 'Pressure Support',
        short: 'The fixed difference between IPAP and EPAP, in cmH₂O.',
        long: 'Pressure support eases breathing effort and boosts tidal volume; a higher value means more ventilatory assistance on each breath.',
    },
    setting_min_epap: {
        label: 'Min EPAP',
        short: 'The lower bound of the auto-adjusting EPAP range in bilevel auto modes, in cmH₂O.',
    },
    setting_max_epap: {
        label: 'Max EPAP',
        short: 'The upper bound of the auto-adjusting EPAP range in bilevel auto modes, in cmH₂O.',
    },
    setting_min_ps: {
        label: 'Min PS',
        short: 'The lower bound of auto-adjusting pressure support, in cmH₂O.',
        long: 'Applies to modes that vary pressure support automatically, such as iVAPS or ASV-style ventilation.',
    },
    setting_max_ps: {
        label: 'Max PS',
        short: 'The upper bound of auto-adjusting pressure support, in cmH₂O.',
        long: 'Applies to modes that vary pressure support automatically, such as iVAPS or ASV-style ventilation.',
    },
    setting_epap_auto: {
        label: 'EPAP Auto',
        short: 'Whether EPAP auto-adjusts in response to obstructive events rather than staying fixed.',
    },
    setting_ramp_start_pressure: {
        label: 'Ramp Start Pressure',
        short: 'The reduced pressure the device starts at during the ramp period, in cmH₂O.',
        long: 'A gentler starting pressure makes falling asleep easier; the device rises from here to the prescribed pressure over the ramp time.',
    },
    setting_epr_level: {
        label: 'EPR Level',
        short: 'The Expiratory Pressure Relief level, from 1 to 3.',
        long: 'Each level drops exhalation pressure by that many cmH₂O in CPAP and APAP modes, making breathing out feel easier without changing the treatment pressure.',
    },
    setting_epr_mode: {
        label: 'EPR Mode',
        short: 'When Expiratory Pressure Relief applies, or whether it is off.',
        long: 'Common options are Full Time (relief on every breath all night) and Ramp Only (relief only during the ramp period).',
    },
    setting_response: {
        label: 'Response',
        short: 'How quickly APAP raises pressure after an event, set to Standard or Soft.',
        long: 'Soft responds more gently and gradually, which some users find more comfortable; Standard reacts faster to events.',
    },
    setting_ramp_enabled: {
        label: 'Ramp',
        short: 'Whether the device starts at a low pressure and gradually rises to the prescribed level while you fall asleep.',
    },
    setting_ramp_time: {
        label: 'Ramp Time',
        short: 'How long the ramp period lasts before full pressure is reached, in minutes.',
    },
    setting_smart_ramp: {
        label: 'Smart Ramp',
        short: 'AutoRamp holds the low ramp pressure until the device detects you have fallen asleep, then ramps up.',
    },
    setting_ti_min: {
        label: 'Ti Min',
        short: 'The shortest inspiratory time the device allows per breath, in seconds.',
        long: 'In bilevel modes this constrains how briefly pressure can stay at IPAP, keeping breaths from being cut too short.',
    },
    setting_ti_max: {
        label: 'Ti Max',
        short: 'The longest inspiratory time the device allows per breath, in seconds.',
        long: 'In bilevel modes this caps how long pressure stays at IPAP before cycling back to EPAP.',
    },
    setting_rise_time: {
        label: 'Rise Time',
        short: 'How quickly pressure transitions from EPAP to IPAP at the start of a breath.',
        long: 'A longer rise time makes the switch to inhalation pressure feel gentler and more gradual.',
    },
    setting_trigger: {
        label: 'Trigger',
        short: 'The sensitivity for detecting the start of inhalation and switching to IPAP.',
        long: 'A higher trigger sensitivity switches to inspiratory pressure on a smaller breathing effort.',
    },
    setting_cycle: {
        label: 'Cycle',
        short: 'The sensitivity for detecting the end of inhalation and switching back to EPAP.',
        long: 'This affects how well the device stays synchronized with the end of your natural breath.',
    },
    setting_humidity_enabled: {
        label: 'Humidity',
        short: 'Whether the heated humidifier is turned on.',
    },
    setting_humidity_level: {
        label: 'Humidity Level',
        short: 'The moisture output of the heated humidifier, from 1 to 8.',
        long: 'Higher levels add more moisture to the air, reducing dryness and nasal congestion.',
    },
    setting_climate_control: {
        label: 'Climate Control',
        short: 'Whether the device manages humidity and tube temperature automatically or you set them yourself.',
        long: 'In Auto mode the device coordinates humidity and tube temperature to prevent rainout; in Manual mode you control each setting directly.',
    },
    setting_tube_temp_enabled: {
        label: 'Heated Tube',
        short: 'Whether the heated tube is turned on.',
    },
    setting_tube_temp: {
        label: 'Tube Temperature',
        short: 'The target temperature of the air in the heated tube.',
        long: 'Warming the hose prevents condensation, or "rainout," that would otherwise pool and gurgle in the tubing.',
    },
    setting_smart_start: {
        label: 'Smart Start',
        short: 'Therapy starts automatically when you breathe into the mask.',
    },
    setting_smart_stop: {
        label: 'Smart Stop',
        short: 'Therapy stops automatically a short time after you remove the mask.',
    },
    setting_ab_filter: {
        label: 'Filter Type',
        short: 'The air filter type the device is configured for, Standard or antibacterial.',
        long: 'The setting adjusts the device airflow compensation to account for the filter fitted.',
    },
    setting_mask_type: {
        label: 'Mask Type',
        short: 'The mask category the device compensates for, such as Full Face, Nasal, or Pillows.',
        long: 'The device tunes its leak and pressure compensation curves to the selected mask category.',
    },
    setting_easy_breathe: {
        label: 'Easy-Breathe',
        short: 'The Easy-Breathe waveform smooths pressure changes to mirror natural breathing for comfort.',
    },
    setting_tube: {
        label: 'Tube',
        short: 'The hose diameter or model the device calibrates its flow against, such as SlimLine or Standard.',
    },
    setting_pt_access: {
        label: 'Patient Access',
        short: 'Whether the patient menu is allowed to change clinical settings on the device.',
    },
    setting_pt_view: {
        label: 'Patient View',
        short: 'Which on-device menu is shown: Simple, or the more detailed Advanced view.',
    },
}
