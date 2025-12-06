-- SNORE Database Schema
-- SQLite database for storing CPAP therapy data from all device types
-- Universal schema approach inspired by OSCAR's channel-based architecture

-- =============================================================================
-- DEVICES
-- =============================================================================

CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manufacturer TEXT NOT NULL,
    model TEXT NOT NULL,
    serial_number TEXT UNIQUE NOT NULL,
    firmware_version TEXT,
    hardware_version TEXT,
    product_code TEXT,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_import TIMESTAMP,
    CONSTRAINT chk_manufacturer CHECK (length(manufacturer) > 0),
    CONSTRAINT chk_serial CHECK (length(serial_number) > 0)
);

-- =============================================================================
-- SESSIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id INTEGER NOT NULL,
    device_session_id TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    duration_seconds REAL,
    therapy_mode TEXT,
    import_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    import_source TEXT,
    parser_version TEXT,
    data_quality_notes TEXT,  -- JSON array of quality issues
    has_waveform_data BOOLEAN DEFAULT 0,
    has_event_data BOOLEAN DEFAULT 0,
    has_statistics BOOLEAN DEFAULT 0,
    FOREIGN KEY (device_id) REFERENCES devices(id) ON DELETE CASCADE,
    UNIQUE(device_id, device_session_id),
    CONSTRAINT chk_time_range CHECK (end_time >= start_time),
    CONSTRAINT chk_duration CHECK (duration_seconds IS NULL OR duration_seconds >= 0)
);

-- =============================================================================
-- WAVEFORMS (Time-Series Data)
-- =============================================================================

CREATE TABLE IF NOT EXISTS waveforms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    waveform_type TEXT NOT NULL,
    sample_rate REAL NOT NULL,
    unit TEXT,
    min_value REAL,
    max_value REAL,
    mean_value REAL,
    data_blob BLOB NOT NULL,
    sample_count INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    UNIQUE(session_id, waveform_type),
    CONSTRAINT chk_sample_rate CHECK (sample_rate > 0)
);

-- =============================================================================
-- EVENTS (Respiratory Events, Flags)
-- =============================================================================

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    start_time TIMESTAMP NOT NULL,
    duration_seconds REAL,
    spo2_drop REAL,
    peak_flow_limitation REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    CONSTRAINT chk_duration CHECK (duration_seconds IS NULL OR duration_seconds >= 0)
);

-- =============================================================================
-- STATISTICS (Pre-calculated Summary Data)
-- =============================================================================

CREATE TABLE IF NOT EXISTS statistics (
    session_id INTEGER PRIMARY KEY,
    -- Respiratory event counts
    obstructive_apneas INTEGER DEFAULT 0,
    central_apneas INTEGER DEFAULT 0,
    mixed_apneas INTEGER DEFAULT 0,
    hypopneas INTEGER DEFAULT 0,
    reras INTEGER DEFAULT 0,
    flow_limitations INTEGER DEFAULT 0,
    -- Respiratory indices
    ahi REAL,
    oai REAL,
    cai REAL,
    hi REAL,
    rei REAL,
    -- Pressure statistics
    pressure_min REAL,
    pressure_max REAL,
    pressure_median REAL,
    pressure_mean REAL,
    pressure_95th REAL,
    -- Leak statistics
    leak_min REAL,
    leak_max REAL,
    leak_median REAL,
    leak_mean REAL,
    leak_95th REAL,
    leak_percentile_70 REAL,
    -- Respiratory rate statistics
    respiratory_rate_min REAL,
    respiratory_rate_max REAL,
    respiratory_rate_mean REAL,
    -- Tidal volume statistics
    tidal_volume_min REAL,
    tidal_volume_max REAL,
    tidal_volume_mean REAL,
    -- Minute ventilation statistics
    minute_ventilation_min REAL,
    minute_ventilation_max REAL,
    minute_ventilation_mean REAL,
    -- Oximetry statistics
    spo2_min REAL,
    spo2_max REAL,
    spo2_mean REAL,
    spo2_time_below_90 INTEGER,
    pulse_min REAL,
    pulse_max REAL,
    pulse_mean REAL,
    -- Usage
    usage_hours REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- =============================================================================
-- SETTINGS (Device Configuration)
-- =============================================================================

CREATE TABLE IF NOT EXISTS settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
    UNIQUE(session_id, key)
);


-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_device ON sessions(device_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON sessions(start_time);
CREATE INDEX IF NOT EXISTS idx_sessions_device_session ON sessions(device_id, device_session_id);

-- Events indexes
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_session_time ON events(session_id, start_time);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

-- Waveforms indexes
CREATE INDEX IF NOT EXISTS idx_waveforms_session ON waveforms(session_id);
CREATE INDEX IF NOT EXISTS idx_waveforms_session_type ON waveforms(session_id, waveform_type);

-- Settings indexes
CREATE INDEX IF NOT EXISTS idx_settings_session ON settings(session_id);
