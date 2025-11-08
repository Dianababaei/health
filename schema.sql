-- ============================================================================
-- Artemis Health - TimescaleDB Schema Initialization
-- ============================================================================
-- Database: artemis_health
-- Version: 1.0
-- Created: 2025-11-08
-- Description: Complete schema for cattle health monitoring system
--              with time-series optimization using TimescaleDB
-- ============================================================================

-- Prerequisites:
-- 1. PostgreSQL 14+ installed
-- 2. TimescaleDB extension installed
-- 3. Database created: CREATE DATABASE artemis_health;

-- Connect to database and enable TimescaleDB extension
\c artemis_health

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable PostGIS for future GPS integration (optional)
-- CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================================
-- TABLE 1: raw_sensor_readings
-- Purpose: Store raw sensor data from neck-mounted devices
-- Layer: 0 (Base data)
-- Retention: 180 days
-- ============================================================================

CREATE TABLE IF NOT EXISTS raw_sensor_readings (
    timestamp TIMESTAMPTZ NOT NULL,
    cow_id INTEGER NOT NULL,
    sensor_id TEXT NOT NULL,
    temperature DOUBLE PRECISION NOT NULL CHECK (temperature BETWEEN 30.0 AND 45.0),
    fxa DOUBLE PRECISION NOT NULL,  -- X-axis acceleration (forward/backward) in g
    mya DOUBLE PRECISION NOT NULL,  -- Y-axis acceleration (lateral) in g
    rza DOUBLE PRECISION NOT NULL,  -- Z-axis acceleration (vertical) in g
    sxg DOUBLE PRECISION NOT NULL,  -- X-axis angular velocity (roll) in deg/s
    lyg DOUBLE PRECISION NOT NULL,  -- Y-axis angular velocity (pitch) in deg/s
    dzg DOUBLE PRECISION NOT NULL,  -- Z-axis angular velocity (yaw) in deg/s
    data_quality TEXT DEFAULT 'good' CHECK (data_quality IN ('good', 'degraded', 'poor', 'sensor_error')),
    metadata JSONB,  -- Additional sensor metadata (battery, RSSI, etc.)

    PRIMARY KEY (cow_id, timestamp)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'raw_sensor_readings',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for raw_sensor_readings
CREATE INDEX IF NOT EXISTS idx_raw_sensor_timestamp_desc
    ON raw_sensor_readings (timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_raw_sensor_cow_time
    ON raw_sensor_readings (cow_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_raw_sensor_quality
    ON raw_sensor_readings (data_quality)
    WHERE data_quality != 'good';

CREATE INDEX IF NOT EXISTS idx_raw_sensor_device
    ON raw_sensor_readings (sensor_id, timestamp DESC);

-- Compression policy for raw_sensor_readings (compress after 7 days)
ALTER TABLE raw_sensor_readings
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id, sensor_id',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('raw_sensor_readings', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy for raw_sensor_readings (delete after 180 days)
SELECT add_retention_policy('raw_sensor_readings', INTERVAL '180 days', if_not_exists => TRUE);

-- ============================================================================
-- TABLE 2: behavioral_states
-- Purpose: Store Layer 1 behavior classification results
-- Layer: 1 (Behavioral)
-- Retention: 180 days
-- ============================================================================

CREATE TABLE IF NOT EXISTS behavioral_states (
    timestamp TIMESTAMPTZ NOT NULL,
    cow_id INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('lying', 'standing', 'walking', 'ruminating', 'feeding', 'unknown')),
    confidence DOUBLE PRECISION NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    duration_minutes INTEGER,  -- Duration of continuous state (null for ongoing)
    motion_intensity DOUBLE PRECISION,  -- Overall motion level (0-1 scale)
    posture_context TEXT CHECK (posture_context IN ('lying', 'standing', NULL)),  -- For ruminating state
    metadata JSONB,  -- Detailed classification info (feature values, secondary predictions)

    PRIMARY KEY (cow_id, timestamp)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'behavioral_states',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for behavioral_states
CREATE INDEX IF NOT EXISTS idx_behavioral_cow_time
    ON behavioral_states (cow_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_behavioral_state
    ON behavioral_states (state, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_behavioral_cow_state_time
    ON behavioral_states (cow_id, state, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_behavioral_low_confidence
    ON behavioral_states (confidence)
    WHERE confidence < 0.7;

-- Compression policy for behavioral_states
ALTER TABLE behavioral_states
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id, state',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('behavioral_states', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy for behavioral_states
SELECT add_retention_policy('behavioral_states', INTERVAL '180 days', if_not_exists => TRUE);

-- ============================================================================
-- TABLE 3: physiological_metrics
-- Purpose: Store Layer 2 physiological analysis results
-- Layer: 2 (Physiological)
-- Retention: 180 days
-- ============================================================================

CREATE TABLE IF NOT EXISTS physiological_metrics (
    timestamp TIMESTAMPTZ NOT NULL,
    cow_id INTEGER NOT NULL,
    baseline_temp DOUBLE PRECISION NOT NULL,  -- Individual baseline temperature
    current_temp DOUBLE PRECISION NOT NULL,   -- Current measured temperature
    temp_deviation DOUBLE PRECISION,          -- Deviation from baseline (°C)
    circadian_phase DOUBLE PRECISION CHECK (circadian_phase BETWEEN 0 AND 6.283185),  -- Radians (0-2π)
    circadian_amplitude DOUBLE PRECISION,     -- Expected circadian variation amplitude
    temp_anomaly_score DOUBLE PRECISION CHECK (temp_anomaly_score BETWEEN 0.0 AND 1.0),
    circadian_rhythm_stability DOUBLE PRECISION CHECK (circadian_rhythm_stability BETWEEN 0.0 AND 1.0),
    activity_level DOUBLE PRECISION,          -- Recent activity level (from behavioral_states)
    temp_activity_correlation DOUBLE PRECISION,  -- Temperature-activity relationship
    metadata JSONB,  -- Analysis parameters, statistical measures

    PRIMARY KEY (cow_id, timestamp)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'physiological_metrics',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for physiological_metrics
CREATE INDEX IF NOT EXISTS idx_physiological_cow_time
    ON physiological_metrics (cow_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_physiological_anomalies
    ON physiological_metrics (temp_anomaly_score DESC, timestamp DESC)
    WHERE temp_anomaly_score > 0.7;

CREATE INDEX IF NOT EXISTS idx_physiological_rhythm_instability
    ON physiological_metrics (circadian_rhythm_stability, timestamp DESC)
    WHERE circadian_rhythm_stability < 0.5;

CREATE INDEX IF NOT EXISTS idx_physiological_temp_deviation
    ON physiological_metrics (cow_id, temp_deviation, timestamp DESC);

-- Compression policy for physiological_metrics
ALTER TABLE physiological_metrics
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('physiological_metrics', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy for physiological_metrics
SELECT add_retention_policy('physiological_metrics', INTERVAL '180 days', if_not_exists => TRUE);

-- ============================================================================
-- TABLE 4: alerts
-- Purpose: Store Layer 3 alert events with status tracking
-- Layer: 3 (Health Intelligence)
-- Retention: 365 days (extended for audit trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    cow_id INTEGER NOT NULL,
    alert_type TEXT NOT NULL CHECK (alert_type IN (
        'fever', 'heat_stress', 'prolonged_inactivity', 'estrus',
        'pregnancy_detected', 'sensor_malfunction', 'abnormal_rumination',
        'feeding_anomaly', 'circadian_disruption', 'health_score_critical'
    )),
    severity TEXT NOT NULL CHECK (severity IN ('critical', 'warning', 'info')),
    title TEXT NOT NULL,  -- Human-readable alert title
    details JSONB NOT NULL,  -- Alert-specific details (thresholds, values, duration)
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'acknowledged', 'resolved', 'false_positive')),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by TEXT,
    resolution_notes TEXT,
    sensor_values JSONB,  -- Snapshot of sensor data at alert time
    related_metrics JSONB,  -- Related physiological/behavioral metrics

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Convert to hypertable with 7-day chunks (longer chunks for alerts)
SELECT create_hypertable(
    'alerts',
    'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- Indexes for alerts
CREATE INDEX IF NOT EXISTS idx_alerts_active
    ON alerts (status, timestamp DESC)
    WHERE status = 'active';

CREATE INDEX IF NOT EXISTS idx_alerts_cow_time
    ON alerts (cow_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_type
    ON alerts (alert_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_severity
    ON alerts (severity, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_dashboard
    ON alerts (cow_id, alert_type, status, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_alerts_id
    ON alerts (alert_id);

-- Compression policy for alerts (compress after 30 days)
ALTER TABLE alerts
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id, alert_type, severity',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('alerts', INTERVAL '30 days', if_not_exists => TRUE);

-- Retention policy for alerts (delete after 365 days)
SELECT add_retention_policy('alerts', INTERVAL '365 days', if_not_exists => TRUE);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_alerts_updated_at
    BEFORE UPDATE ON alerts
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- TABLE 5: health_scores
-- Purpose: Store Layer 3 health scoring (0-100 scale)
-- Layer: 3 (Health Intelligence)
-- Retention: 180 days
-- ============================================================================

CREATE TABLE IF NOT EXISTS health_scores (
    timestamp TIMESTAMPTZ NOT NULL,
    cow_id INTEGER NOT NULL,
    health_score INTEGER NOT NULL CHECK (health_score BETWEEN 0 AND 100),
    temperature_component DOUBLE PRECISION CHECK (temperature_component BETWEEN 0.0 AND 1.0),
    activity_component DOUBLE PRECISION CHECK (activity_component BETWEEN 0.0 AND 1.0),
    behavior_component DOUBLE PRECISION CHECK (behavior_component BETWEEN 0.0 AND 1.0),
    rumination_component DOUBLE PRECISION CHECK (rumination_component BETWEEN 0.0 AND 1.0),
    alert_penalty DOUBLE PRECISION CHECK (alert_penalty BETWEEN 0.0 AND 1.0),
    trend_direction TEXT CHECK (trend_direction IN ('improving', 'stable', 'deteriorating')),
    trend_rate DOUBLE PRECISION,  -- Points per day change rate
    days_since_baseline INTEGER,  -- Days since last "normal" health score
    contributing_factors JSONB,  -- Detailed breakdown of score components

    PRIMARY KEY (cow_id, timestamp)
);

-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'health_scores',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Indexes for health_scores
CREATE INDEX IF NOT EXISTS idx_health_cow_time
    ON health_scores (cow_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_health_score_low
    ON health_scores (health_score, timestamp DESC)
    WHERE health_score < 50;

CREATE INDEX IF NOT EXISTS idx_health_deteriorating
    ON health_scores (trend_direction, timestamp DESC)
    WHERE trend_direction = 'deteriorating';

CREATE INDEX IF NOT EXISTS idx_health_ranking
    ON health_scores (timestamp DESC, health_score DESC);

CREATE INDEX IF NOT EXISTS idx_health_components
    ON health_scores (cow_id, temperature_component, activity_component, timestamp DESC);

-- Compression policy for health_scores
ALTER TABLE health_scores
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('health_scores', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy for health_scores
SELECT add_retention_policy('health_scores', INTERVAL '180 days', if_not_exists => TRUE);

-- ============================================================================
-- CONTINUOUS AGGREGATES
-- Pre-computed rollups for fast dashboard queries
-- ============================================================================

-- Hourly sensor data rollups
CREATE MATERIALIZED VIEW IF NOT EXISTS sensor_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS hour,
    cow_id,
    AVG(temperature) AS avg_temp,
    MAX(temperature) AS max_temp,
    MIN(temperature) AS min_temp,
    STDDEV(fxa) AS fxa_std,
    STDDEV(mya) AS mya_std,
    STDDEV(rza) AS rza_std,
    COUNT(*) AS reading_count
FROM raw_sensor_readings
GROUP BY hour, cow_id;

-- Refresh policy for sensor_hourly (update every hour)
SELECT add_continuous_aggregate_policy('sensor_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Daily behavioral summaries
CREATE MATERIALIZED VIEW IF NOT EXISTS behavior_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    cow_id,
    state,
    COUNT(*) AS state_count,
    SUM(COALESCE(duration_minutes, 1)) AS total_minutes,
    AVG(confidence) AS avg_confidence
FROM behavioral_states
GROUP BY day, cow_id, state;

-- Refresh policy for behavior_daily (update every day)
SELECT add_continuous_aggregate_policy('behavior_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Daily health score trends
CREATE MATERIALIZED VIEW IF NOT EXISTS health_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    cow_id,
    AVG(health_score) AS avg_health_score,
    MIN(health_score) AS min_health_score,
    MAX(health_score) AS max_health_score,
    AVG(temperature_component) AS avg_temp_component,
    AVG(activity_component) AS avg_activity_component,
    AVG(behavior_component) AS avg_behavior_component,
    AVG(rumination_component) AS avg_rumination_component
FROM health_scores
GROUP BY day, cow_id;

-- Refresh policy for health_daily (update every day)
SELECT add_continuous_aggregate_policy('health_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ============================================================================
-- MATERIALIZED VIEWS (Non-Continuous)
-- For less frequently updated aggregates
-- ============================================================================

-- Active alerts summary (refresh manually or via cron)
CREATE MATERIALIZED VIEW IF NOT EXISTS active_alerts_summary AS
SELECT
    cow_id,
    alert_type,
    severity,
    COUNT(*) AS alert_count,
    MIN(timestamp) AS first_alert,
    MAX(timestamp) AS latest_alert
FROM alerts
WHERE status = 'active'
GROUP BY cow_id, alert_type, severity;

CREATE INDEX IF NOT EXISTS idx_active_alerts_cow ON active_alerts_summary(cow_id);

-- ============================================================================
-- DATABASE CONFIGURATION
-- Performance optimization settings
-- ============================================================================

-- Shared buffers (25% of RAM - adjust based on your system)
-- ALTER SYSTEM SET shared_buffers = '4GB';

-- Work memory (for sorting/joins)
-- ALTER SYSTEM SET work_mem = '64MB';

-- Maintenance work memory (for vacuuming)
-- ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Effective cache size (50-75% of RAM)
-- ALTER SYSTEM SET effective_cache_size = '12GB';

-- Auto-vacuum settings
ALTER SYSTEM SET autovacuum = on;
ALTER SYSTEM SET autovacuum_max_workers = 4;
ALTER SYSTEM SET autovacuum_naptime = '30s';

-- Parallel query settings (adjust based on CPU cores)
-- ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
-- ALTER SYSTEM SET parallel_tuple_cost = 0.01;
-- ALTER SYSTEM SET parallel_setup_cost = 100;

-- ============================================================================
-- SAMPLE DATA (Optional - for testing)
-- Uncomment to insert sample data for development/testing
-- ============================================================================

/*
-- Insert sample sensor readings
INSERT INTO raw_sensor_readings (
    timestamp, cow_id, sensor_id, temperature,
    fxa, mya, rza, sxg, lyg, dzg, data_quality
) VALUES
    (NOW() - INTERVAL '5 minutes', 1042, 'SENSOR-A3F2', 38.6, 0.15, 0.08, 0.72, 5.2, 12.3, 2.1, 'good'),
    (NOW() - INTERVAL '4 minutes', 1042, 'SENSOR-A3F2', 38.6, 0.14, 0.09, 0.71, 5.0, 11.8, 2.3, 'good'),
    (NOW() - INTERVAL '3 minutes', 1042, 'SENSOR-A3F2', 38.7, 0.16, 0.10, 0.73, 5.5, 12.1, 2.0, 'good'),
    (NOW() - INTERVAL '2 minutes', 1042, 'SENSOR-A3F2', 38.6, 0.15, 0.11, 0.72, 5.3, 12.0, 2.2, 'good'),
    (NOW() - INTERVAL '1 minute', 1042, 'SENSOR-A3F2', 38.7, 0.17, 0.09, 0.74, 5.4, 12.5, 2.1, 'good');

-- Insert sample behavioral states
INSERT INTO behavioral_states (
    timestamp, cow_id, state, confidence, motion_intensity, posture_context
) VALUES
    (NOW() - INTERVAL '5 minutes', 1042, 'standing', 0.92, 0.15, NULL),
    (NOW() - INTERVAL '4 minutes', 1042, 'standing', 0.91, 0.14, NULL),
    (NOW() - INTERVAL '3 minutes', 1042, 'feeding', 0.87, 0.35, 'standing'),
    (NOW() - INTERVAL '2 minutes', 1042, 'feeding', 0.89, 0.37, 'standing'),
    (NOW() - INTERVAL '1 minute', 1042, 'standing', 0.93, 0.12, NULL);

-- Insert sample health score
INSERT INTO health_scores (
    timestamp, cow_id, health_score, temperature_component, activity_component,
    behavior_component, rumination_component, alert_penalty, trend_direction
) VALUES
    (NOW(), 1042, 78, 0.85, 0.92, 0.88, 0.75, 0.10, 'stable');
*/

-- ============================================================================
-- VERIFICATION QUERIES
-- Run these to verify schema installation
-- ============================================================================

-- Check hypertables
SELECT * FROM timescaledb_information.hypertables;

-- Check compression policies
SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_compression';

-- Check retention policies
SELECT * FROM timescaledb_information.jobs WHERE proc_name = 'policy_retention';

-- Check continuous aggregates
SELECT * FROM timescaledb_information.continuous_aggregates;

-- ============================================================================
-- SCHEMA INSTALLATION COMPLETE
-- ============================================================================

-- Next steps:
-- 1. Review and adjust memory settings in DATABASE CONFIGURATION section
-- 2. Restart PostgreSQL to apply system settings: sudo systemctl restart postgresql
-- 3. Test with sample data (uncomment SAMPLE DATA section)
-- 4. Set up application connection pooling (PgBouncer recommended)
-- 5. Configure monitoring (pg_stat_statements, TimescaleDB telemetry)

COMMENT ON DATABASE artemis_health IS 'Artemis Health - Cattle Health Monitoring System';
