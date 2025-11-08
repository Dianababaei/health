# TimescaleDB Schema Documentation - Artemis Health

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Database System:** TimescaleDB (PostgreSQL extension)
**Purpose:** Time-series data storage and analysis for cattle health monitoring system

---

## Table of Contents

1. [Overview](#overview)
2. [Database Architecture](#database-architecture)
3. [Schema Definitions](#schema-definitions)
   - [raw_sensor_readings](#1-raw_sensor_readings)
   - [behavioral_states](#2-behavioral_states)
   - [physiological_metrics](#3-physiological_metrics)
   - [alerts](#4-alerts)
   - [health_scores](#5-health_scores)
4. [Retention Policies](#retention-policies)
5. [Query Optimization Strategies](#query-optimization-strategies)
6. [Data Ingestion Patterns](#data-ingestion-patterns)
7. [Sample Queries](#sample-queries)
8. [Performance Considerations](#performance-considerations)
9. [References](#references)

---

## Overview

### System Context

The Artemis Health system monitors cattle using neck-mounted sensors that transmit data every minute. The database schema supports:

- **Layer 1 (Behavioral):** Physical activity and posture classification
- **Layer 2 (Physiological):** Temperature analysis and circadian rhythm monitoring
- **Layer 3 (Health Intelligence):** Alert generation and health scoring

### Key Requirements

- **Data Volume:** ~1,440 sensor readings per cow per day (1-minute intervals)
- **Retention Period:** 90-180 days for reproductive cycle tracking (estrus: 21 days, pregnancy: 60+ days)
- **Query Patterns:** Real-time dashboards, rolling window analysis, multi-day trend detection
- **Write Performance:** Batch ingestion of minute-level data
- **Read Performance:** Sub-second queries for dashboard and alert generation

### TimescaleDB Benefits

- **Hypertables:** Automatic time-based partitioning for efficient queries
- **Continuous Aggregates:** Pre-computed rollups for common aggregations
- **Compression:** Native compression for historical data (7+ days old)
- **Retention Policies:** Automatic data lifecycle management
- **PostgreSQL Compatibility:** Full SQL support with time-series optimizations

---

## Database Architecture

### Hypertable Strategy

All time-series tables are implemented as TimescaleDB hypertables with:
- **Partitioning:** Time-based chunks (1-7 day intervals)
- **Indexing:** Optimized for time-descending queries and cow_id filters
- **Compression:** Enabled for chunks older than 7 days
- **Retention:** Automatic deletion based on data age policies

### Data Flow

```
Sensor Data (1-minute intervals)
    ↓
raw_sensor_readings (Layer 0 - Base data)
    ↓
behavioral_states (Layer 1 - Behavior classification)
    ↓
physiological_metrics (Layer 2 - Temperature/circadian analysis)
    ↓
alerts + health_scores (Layer 3 - Health intelligence)
```

---

## Schema Definitions

### 1. raw_sensor_readings

**Purpose:** Store raw sensor data from neck-mounted devices (Layer 0 - base data).

#### Table Schema

```sql
CREATE TABLE raw_sensor_readings (
    timestamp TIMESTAMPTZ NOT NULL,
    cow_id INTEGER NOT NULL,
    sensor_id TEXT NOT NULL,
    temperature DOUBLE PRECISION NOT NULL CHECK (temperature BETWEEN 30.0 AND 45.0),
    fxa DOUBLE PRECISION NOT NULL,  -- X-axis acceleration (forward/backward)
    mya DOUBLE PRECISION NOT NULL,  -- Y-axis acceleration (lateral)
    rza DOUBLE PRECISION NOT NULL,  -- Z-axis acceleration (vertical)
    sxg DOUBLE PRECISION NOT NULL,  -- X-axis angular velocity (roll)
    lyg DOUBLE PRECISION NOT NULL,  -- Y-axis angular velocity (pitch)
    dzg DOUBLE PRECISION NOT NULL,  -- Z-axis angular velocity (yaw)
    data_quality TEXT DEFAULT 'good' CHECK (data_quality IN ('good', 'degraded', 'poor', 'sensor_error')),
    metadata JSONB,  -- Additional sensor metadata (battery level, signal strength, etc.)

    PRIMARY KEY (cow_id, timestamp)
);
```

#### Hypertable Configuration

```sql
-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'raw_sensor_readings',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
```

#### Indexes

```sql
-- Time-descending index for recent data queries
CREATE INDEX idx_raw_sensor_timestamp_desc
    ON raw_sensor_readings (timestamp DESC);

-- Cow-specific queries (most common pattern)
CREATE INDEX idx_raw_sensor_cow_time
    ON raw_sensor_readings (cow_id, timestamp DESC);

-- Data quality filtering
CREATE INDEX idx_raw_sensor_quality
    ON raw_sensor_readings (data_quality)
    WHERE data_quality != 'good';

-- Sensor ID for device-specific diagnostics
CREATE INDEX idx_raw_sensor_device
    ON raw_sensor_readings (sensor_id, timestamp DESC);
```

#### Compression Policy

```sql
-- Compress chunks older than 7 days
ALTER TABLE raw_sensor_readings
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id, sensor_id',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('raw_sensor_readings', INTERVAL '7 days');
```

#### Column Descriptions

| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| `timestamp` | TIMESTAMPTZ | UTC timestamp of sensor reading | `2025-11-08 14:23:00+00` |
| `cow_id` | INTEGER | Unique cattle identifier | `1042` |
| `sensor_id` | TEXT | Unique sensor device identifier | `SENSOR-A3F2-9821` |
| `temperature` | DOUBLE PRECISION | Body temperature in °C | `38.6` |
| `fxa` | DOUBLE PRECISION | Forward/backward acceleration (g-force) | `0.15` |
| `mya` | DOUBLE PRECISION | Lateral acceleration (g-force) | `0.08` |
| `rza` | DOUBLE PRECISION | Vertical acceleration (g-force) | `0.72` |
| `sxg` | DOUBLE PRECISION | Roll angular velocity (degrees/second) | `5.2` |
| `lyg` | DOUBLE PRECISION | Pitch angular velocity (degrees/second) | `12.3` |
| `dzg` | DOUBLE PRECISION | Yaw angular velocity (degrees/second) | `2.1` |
| `data_quality` | TEXT | Sensor data quality indicator | `good` |
| `metadata` | JSONB | Additional sensor info (battery, RSSI, etc.) | `{"battery": 87, "rssi": -65}` |

#### Constraints & Validation

- **Temperature Check:** 30.0°C - 45.0°C (physiological range + error margin)
- **Data Quality Enum:** Ensures only valid quality values
- **Primary Key:** Prevents duplicate readings for same cow at same timestamp
- **NOT NULL:** All sensor readings required (missing data = sensor_error quality)

---

### 2. behavioral_states

**Purpose:** Store Layer 1 behavior classification results (lying, standing, walking, ruminating, feeding).

#### Table Schema

```sql
CREATE TABLE behavioral_states (
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
```

#### Hypertable Configuration

```sql
-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'behavioral_states',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
```

#### Indexes

```sql
-- Cow-specific time-descending queries (most common)
CREATE INDEX idx_behavioral_cow_time
    ON behavioral_states (cow_id, timestamp DESC);

-- State-based filtering for behavior analysis
CREATE INDEX idx_behavioral_state
    ON behavioral_states (state, timestamp DESC);

-- State duration queries (e.g., prolonged lying detection)
CREATE INDEX idx_behavioral_cow_state_time
    ON behavioral_states (cow_id, state, timestamp DESC);

-- Low-confidence predictions for model retraining
CREATE INDEX idx_behavioral_low_confidence
    ON behavioral_states (confidence)
    WHERE confidence < 0.7;
```

#### Compression Policy

```sql
ALTER TABLE behavioral_states
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id, state',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('behavioral_states', INTERVAL '7 days');
```

#### Column Descriptions

| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| `timestamp` | TIMESTAMPTZ | UTC timestamp of classification | `2025-11-08 14:23:00+00` |
| `cow_id` | INTEGER | Unique cattle identifier | `1042` |
| `state` | TEXT | Classified behavioral state | `ruminating` |
| `confidence` | DOUBLE PRECISION | Model confidence score (0.0-1.0) | `0.92` |
| `duration_minutes` | INTEGER | Continuous state duration (updated retroactively) | `35` |
| `motion_intensity` | DOUBLE PRECISION | Normalized motion level (0.0-1.0) | `0.15` |
| `posture_context` | TEXT | Posture during activity (for ruminating/feeding) | `lying` |
| `metadata` | JSONB | Classification details (features, probabilities) | `{"rza": -0.58, "mya_freq": 0.92}` |

#### State Definitions (from behavioral_sensor_signatures.md)

- **lying:** Rza < -0.5g, low motion
- **standing:** Rza > 0.7g, low motion
- **walking:** Rhythmic Fxa patterns, 0.8-1.2 Hz
- **ruminating:** Mya/Lyg frequency 0.67-1.0 Hz (40-60 cycles/min)
- **feeding:** Lyg head-down position, Mya lateral movements
- **unknown:** Ambiguous or transition states

---

### 3. physiological_metrics

**Purpose:** Store Layer 2 physiological analysis results (temperature patterns, circadian rhythm).

#### Table Schema

```sql
CREATE TABLE physiological_metrics (
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
```

#### Hypertable Configuration

```sql
-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'physiological_metrics',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
```

#### Indexes

```sql
-- Cow-specific trend queries
CREATE INDEX idx_physiological_cow_time
    ON physiological_metrics (cow_id, timestamp DESC);

-- High anomaly detection for alerts
CREATE INDEX idx_physiological_anomalies
    ON physiological_metrics (temp_anomaly_score DESC, timestamp DESC)
    WHERE temp_anomaly_score > 0.7;

-- Circadian rhythm disruption detection
CREATE INDEX idx_physiological_rhythm_instability
    ON physiological_metrics (circadian_rhythm_stability, timestamp DESC)
    WHERE circadian_rhythm_stability < 0.5;

-- Temperature deviation queries
CREATE INDEX idx_physiological_temp_deviation
    ON physiological_metrics (cow_id, temp_deviation, timestamp DESC);
```

#### Compression Policy

```sql
ALTER TABLE physiological_metrics
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('physiological_metrics', INTERVAL '7 days');
```

#### Column Descriptions

| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| `timestamp` | TIMESTAMPTZ | UTC timestamp of analysis | `2025-11-08 14:23:00+00` |
| `cow_id` | INTEGER | Unique cattle identifier | `1042` |
| `baseline_temp` | DOUBLE PRECISION | Individual baseline temperature (°C) | `38.5` |
| `current_temp` | DOUBLE PRECISION | Current body temperature (°C) | `38.8` |
| `temp_deviation` | DOUBLE PRECISION | Deviation from baseline (°C) | `+0.3` |
| `circadian_phase` | DOUBLE PRECISION | Current circadian phase (radians, 0-2π) | `3.14` |
| `circadian_amplitude` | DOUBLE PRECISION | Expected circadian variation (°C) | `0.4` |
| `temp_anomaly_score` | DOUBLE PRECISION | Temperature anomaly severity (0.0-1.0) | `0.25` |
| `circadian_rhythm_stability` | DOUBLE PRECISION | Rhythm regularity score (0.0-1.0) | `0.85` |
| `activity_level` | DOUBLE PRECISION | Recent activity intensity (0.0-1.0) | `0.62` |
| `temp_activity_correlation` | DOUBLE PRECISION | Temp-activity relationship indicator | `0.15` |
| `metadata` | JSONB | Analysis details (window size, statistical params) | `{"window_hours": 24}` |

#### Physiological Analysis Context

- **Baseline Temperature:** Individualized per cow (typically 38.0-39.0°C)
- **Circadian Phase:** 0 = midnight, π/2 = 6 AM, π = noon, 3π/2 = 6 PM
- **Anomaly Detection:** Fever (>39.5°C), Estrus (+0.3-0.6°C), Heat stress
- **Rhythm Stability:** Loss of circadian rhythm indicates illness

---

### 4. alerts

**Purpose:** Store Layer 3 alert events with status tracking and resolution.

#### Table Schema

```sql
CREATE TABLE alerts (
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
```

#### Hypertable Configuration

```sql
-- Convert to hypertable with 7-day chunks (longer retention)
SELECT create_hypertable(
    'alerts',
    'timestamp',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);
```

#### Indexes

```sql
-- Active alerts for dashboard (most critical query)
CREATE INDEX idx_alerts_active
    ON alerts (status, timestamp DESC)
    WHERE status = 'active';

-- Cow-specific alert history
CREATE INDEX idx_alerts_cow_time
    ON alerts (cow_id, timestamp DESC);

-- Alert type analysis
CREATE INDEX idx_alerts_type
    ON alerts (alert_type, timestamp DESC);

-- Severity-based filtering
CREATE INDEX idx_alerts_severity
    ON alerts (severity, timestamp DESC);

-- Multi-criteria dashboard query
CREATE INDEX idx_alerts_dashboard
    ON alerts (cow_id, alert_type, status, timestamp DESC);

-- Alert ID for quick lookups
CREATE INDEX idx_alerts_id
    ON alerts (alert_id);
```

#### Retention Policy (Extended)

```sql
-- Keep alerts for 365 days (audit trail)
SELECT add_retention_policy('alerts', INTERVAL '365 days');
```

#### Compression Policy

```sql
ALTER TABLE alerts
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id, alert_type, severity',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('alerts', INTERVAL '30 days');
```

#### Column Descriptions

| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| `alert_id` | SERIAL | Unique alert identifier | `12847` |
| `timestamp` | TIMESTAMPTZ | UTC timestamp when alert triggered | `2025-11-08 14:23:00+00` |
| `cow_id` | INTEGER | Affected cattle identifier | `1042` |
| `alert_type` | TEXT | Alert category | `fever` |
| `severity` | TEXT | Alert severity level | `critical` |
| `title` | TEXT | Human-readable alert message | `High Fever Detected - Cow 1042` |
| `details` | JSONB | Alert-specific information | `{"temp": 40.2, "duration_min": 120}` |
| `status` | TEXT | Current alert status | `active` |
| `acknowledged_at` | TIMESTAMPTZ | When alert was acknowledged | `2025-11-08 14:30:00+00` |
| `acknowledged_by` | TEXT | User who acknowledged | `john.doe@farm.com` |
| `resolved_at` | TIMESTAMPTZ | When alert was resolved | `2025-11-08 18:00:00+00` |
| `resolved_by` | TEXT | User who resolved | `vet.smith@clinic.com` |
| `resolution_notes` | TEXT | Resolution details | `Treated with antibiotics` |
| `sensor_values` | JSONB | Sensor data snapshot | `{"temp": 40.2, "rza": -0.6}` |
| `related_metrics` | JSONB | Associated analysis data | `{"anomaly_score": 0.95}` |

#### Alert Type Definitions (from description.md)

- **fever:** Temperature > 39.5°C + reduced motion
- **heat_stress:** High temperature + high activity
- **prolonged_inactivity:** Extended lying without rumination (>4 hours)
- **estrus:** Temperature rise (+0.3-0.6°C) + increased activity
- **pregnancy_detected:** Stable temperature post-estrus, reduced activity (60+ days)
- **sensor_malfunction:** Missing data, poor quality readings, battery low
- **abnormal_rumination:** <4 hours/day or absent patterns
- **feeding_anomaly:** Reduced feeding time or irregular patterns
- **circadian_disruption:** Loss of normal temperature rhythm
- **health_score_critical:** Health score < 30

---

### 5. health_scores

**Purpose:** Store Layer 3 health scoring (0-100 scale) with component breakdown.

#### Table Schema

```sql
CREATE TABLE health_scores (
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
```

#### Hypertable Configuration

```sql
-- Convert to hypertable with 1-day chunks
SELECT create_hypertable(
    'health_scores',
    'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);
```

#### Indexes

```sql
-- Cow-specific trend queries
CREATE INDEX idx_health_cow_time
    ON health_scores (cow_id, timestamp DESC);

-- Low health score alerts
CREATE INDEX idx_health_score_low
    ON health_scores (health_score, timestamp DESC)
    WHERE health_score < 50;

-- Deteriorating trend detection
CREATE INDEX idx_health_deteriorating
    ON health_scores (trend_direction, timestamp DESC)
    WHERE trend_direction = 'deteriorating';

-- Herd-wide health ranking
CREATE INDEX idx_health_ranking
    ON health_scores (timestamp DESC, health_score DESC);

-- Component-based analysis
CREATE INDEX idx_health_components
    ON health_scores (cow_id, temperature_component, activity_component, timestamp DESC);
```

#### Compression Policy

```sql
ALTER TABLE health_scores
    SET (timescaledb.compress,
         timescaledb.compress_segmentby = 'cow_id',
         timescaledb.compress_orderby = 'timestamp DESC');

SELECT add_compression_policy('health_scores', INTERVAL '7 days');
```

#### Column Descriptions

| Column | Type | Description | Example Value |
|--------|------|-------------|---------------|
| `timestamp` | TIMESTAMPTZ | UTC timestamp of score calculation | `2025-11-08 14:23:00+00` |
| `cow_id` | INTEGER | Unique cattle identifier | `1042` |
| `health_score` | INTEGER | Overall health score (0-100) | `78` |
| `temperature_component` | DOUBLE PRECISION | Temperature stability score (0.0-1.0) | `0.85` |
| `activity_component` | DOUBLE PRECISION | Activity level appropriateness (0.0-1.0) | `0.92` |
| `behavior_component` | DOUBLE PRECISION | Behavioral pattern normality (0.0-1.0) | `0.88` |
| `rumination_component` | DOUBLE PRECISION | Rumination adequacy score (0.0-1.0) | `0.75` |
| `alert_penalty` | DOUBLE PRECISION | Penalty for active alerts (0.0-1.0) | `0.10` |
| `trend_direction` | TEXT | Score trajectory | `stable` |
| `trend_rate` | DOUBLE PRECISION | Rate of change (points/day) | `-1.2` |
| `days_since_baseline` | INTEGER | Days since normal health | `0` |
| `contributing_factors` | JSONB | Detailed component breakdown | `{"fever_risk": 0.05}` |

#### Health Score Calculation Logic

```
health_score = 100 * (
    0.25 * temperature_component +
    0.20 * activity_component +
    0.25 * behavior_component +
    0.20 * rumination_component +
    0.10 * (1 - alert_penalty)
)

Interpretation:
- 80-100: Healthy (green)
- 60-79: Monitor (yellow)
- 40-59: Concerning (orange)
- 0-39: Critical (red)
```

---

## Retention Policies

### Data Lifecycle Management

TimescaleDB retention policies automatically delete data older than specified intervals:

```sql
-- raw_sensor_readings: 180 days (full reproductive cycle)
SELECT add_retention_policy('raw_sensor_readings', INTERVAL '180 days');

-- behavioral_states: 180 days
SELECT add_retention_policy('behavioral_states', INTERVAL '180 days');

-- physiological_metrics: 180 days
SELECT add_retention_policy('physiological_metrics', INTERVAL '180 days');

-- health_scores: 180 days
SELECT add_retention_policy('health_scores', INTERVAL '180 days');

-- alerts: 365 days (extended for audit trail)
SELECT add_retention_policy('alerts', INTERVAL '365 days');
```

### Compression Timeline

All hypertables compress data older than 7 days to reduce storage:

| Data Age | Status | Storage Efficiency |
|----------|--------|-------------------|
| 0-7 days | Uncompressed | 100% (fast writes) |
| 7-180/365 days | Compressed | ~10-20% (read-optimized) |
| >180/365 days | Deleted | 0% (automatic cleanup) |

### Storage Estimates

**Per Cow Daily Storage (Uncompressed):**
- raw_sensor_readings: ~60 KB/day (1,440 rows × ~42 bytes)
- behavioral_states: ~30 KB/day
- physiological_metrics: ~25 KB/day
- health_scores: ~20 KB/day
- alerts: ~1-5 KB/day (variable)

**Total:** ~135-140 KB/cow/day uncompressed
**Compressed (7+ days):** ~20-25 KB/cow/day

**100 Cows, 180 Days:**
- Uncompressed (7 days): 100 × 140 KB × 7 = ~98 MB
- Compressed (173 days): 100 × 25 KB × 173 = ~432 MB
- **Total: ~530 MB**

---

## Query Optimization Strategies

### 1. Continuous Aggregates

Pre-compute common aggregations for fast dashboard queries:

```sql
-- Hourly sensor data rollups
CREATE MATERIALIZED VIEW sensor_hourly
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

-- Refresh policy (update every hour)
SELECT add_continuous_aggregate_policy('sensor_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');
```

```sql
-- Daily behavioral summaries
CREATE MATERIALIZED VIEW behavior_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    cow_id,
    state,
    COUNT(*) AS state_count,
    SUM(duration_minutes) AS total_minutes,
    AVG(confidence) AS avg_confidence
FROM behavioral_states
GROUP BY day, cow_id, state;

SELECT add_continuous_aggregate_policy('behavior_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');
```

```sql
-- Daily health score trends
CREATE MATERIALIZED VIEW health_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', timestamp) AS day,
    cow_id,
    AVG(health_score) AS avg_health_score,
    MIN(health_score) AS min_health_score,
    MAX(health_score) AS max_health_score,
    AVG(temperature_component) AS avg_temp_component,
    AVG(activity_component) AS avg_activity_component
FROM health_scores
GROUP BY day, cow_id;

SELECT add_continuous_aggregate_policy('health_daily',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');
```

### 2. Materialized Views (Non-Continuous)

For less frequently updated aggregates:

```sql
-- Active alerts summary (refreshed every 5 minutes)
CREATE MATERIALIZED VIEW active_alerts_summary AS
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

CREATE INDEX idx_active_alerts_cow ON active_alerts_summary(cow_id);
```

### 3. Index-Only Scans

Design indexes to answer queries without table access:

```sql
-- Dashboard "last 10 minutes" query optimization
CREATE INDEX idx_recent_sensor_data
    ON raw_sensor_readings (timestamp DESC, cow_id)
    INCLUDE (temperature, fxa, mya, rza);
```

### 4. Parallel Query Configuration

Enable parallel queries for multi-day analysis:

```sql
-- PostgreSQL parallel query settings
ALTER DATABASE artemis_health SET max_parallel_workers_per_gather = 4;
ALTER DATABASE artemis_health SET parallel_tuple_cost = 0.01;
ALTER DATABASE artemis_health SET parallel_setup_cost = 100;
```

### 5. Query Planning Hints

For complex analytical queries:

```sql
-- Force parallel scan for large time ranges
/*+ Parallel(raw_sensor_readings 4) */
SELECT ...
FROM raw_sensor_readings
WHERE timestamp >= NOW() - INTERVAL '30 days';
```

---

## Data Ingestion Patterns

### 1. Batch Insert Strategy

Insert sensor data in batches for optimal performance:

```sql
-- Batch insert with ON CONFLICT (handle duplicate timestamps)
INSERT INTO raw_sensor_readings (
    timestamp, cow_id, sensor_id, temperature,
    fxa, mya, rza, sxg, lyg, dzg, data_quality
)
VALUES
    ('2025-11-08 14:01:00+00', 1042, 'SENSOR-A3F2', 38.6, 0.15, 0.08, 0.72, 5.2, 12.3, 2.1, 'good'),
    ('2025-11-08 14:02:00+00', 1042, 'SENSOR-A3F2', 38.6, 0.14, 0.09, 0.71, 5.0, 11.8, 2.3, 'good'),
    ('2025-11-08 14:03:00+00', 1042, 'SENSOR-A3F2', 38.7, 0.16, 0.10, 0.73, 5.5, 12.1, 2.0, 'good')
    -- ... up to 1000 rows
ON CONFLICT (cow_id, timestamp)
DO UPDATE SET
    temperature = EXCLUDED.temperature,
    fxa = EXCLUDED.fxa,
    mya = EXCLUDED.mya,
    rza = EXCLUDED.rza,
    sxg = EXCLUDED.sxg,
    lyg = EXCLUDED.lyg,
    dzg = EXCLUDED.dzg,
    data_quality = EXCLUDED.data_quality,
    metadata = EXCLUDED.metadata;
```

### 2. Transaction Batching

Recommended batch sizes:

```sql
BEGIN;
-- Insert 1000-5000 rows per transaction
INSERT INTO raw_sensor_readings VALUES (...);
COMMIT;
```

**Performance Guidelines:**
- **Optimal Batch Size:** 1,000-5,000 rows per transaction
- **Maximum Batch Size:** 10,000 rows (larger batches increase lock contention)
- **Commit Frequency:** Every 1-5 minutes for minute-level data

### 3. COPY Command for Bulk Loading

For large historical data imports:

```sql
COPY raw_sensor_readings (
    timestamp, cow_id, sensor_id, temperature,
    fxa, mya, rza, sxg, lyg, dzg, data_quality
)
FROM '/path/to/sensor_data.csv'
WITH (FORMAT CSV, HEADER true);
```

### 4. Connection Pooling

Use connection pooling (PgBouncer, pgpool-II) for concurrent writes:

```ini
# PgBouncer configuration
[databases]
artemis_health = host=localhost port=5432 dbname=artemis_health

[pgbouncer]
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
```

### 5. Write-Ahead Logging (WAL) Optimization

For high-throughput ingestion:

```sql
-- Increase WAL buffer size
ALTER SYSTEM SET wal_buffers = '16MB';

-- Asynchronous commit for non-critical writes (optional)
ALTER DATABASE artemis_health SET synchronous_commit = off;
```

---

## Sample Queries

### 1. Retrieve Last 10 Minutes of Sensor Data (Dashboard)

```sql
SELECT
    timestamp,
    cow_id,
    temperature,
    fxa, mya, rza,
    sxg, lyg, dzg,
    data_quality
FROM raw_sensor_readings
WHERE timestamp >= NOW() - INTERVAL '10 minutes'
    AND cow_id = 1042
ORDER BY timestamp DESC;
```

**Optimization:** Uses `idx_raw_sensor_cow_time` index for fast retrieval.

---

### 2. Calculate 5-Minute Rolling Average (Alert Generation)

```sql
SELECT
    time_bucket('1 minute', timestamp) AS minute,
    cow_id,
    AVG(temperature) OVER (
        PARTITION BY cow_id
        ORDER BY time_bucket('1 minute', timestamp)
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_temp,
    AVG(fxa) OVER (
        PARTITION BY cow_id
        ORDER BY time_bucket('1 minute', timestamp)
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS rolling_avg_fxa
FROM raw_sensor_readings
WHERE timestamp >= NOW() - INTERVAL '1 hour'
    AND cow_id = 1042
ORDER BY minute DESC;
```

---

### 3. Query Behavioral State Durations (Last 24 Hours)

```sql
SELECT
    state,
    COUNT(*) AS occurrences,
    SUM(COALESCE(duration_minutes, 1)) AS total_minutes,
    AVG(confidence) AS avg_confidence,
    MAX(timestamp) AS last_occurrence
FROM behavioral_states
WHERE cow_id = 1042
    AND timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY state
ORDER BY total_minutes DESC;
```

**Expected Output:**
```
state       | occurrences | total_minutes | avg_confidence | last_occurrence
------------|-------------|---------------|----------------|------------------
lying       | 520         | 520           | 0.94           | 2025-11-08 14:20
ruminating  | 380         | 380           | 0.89           | 2025-11-08 14:15
standing    | 180         | 180           | 0.92           | 2025-11-08 14:23
feeding     | 120         | 120           | 0.87           | 2025-11-08 13:45
walking     | 65          | 65            | 0.91           | 2025-11-08 14:10
```

---

### 4. Retrieve Active Alerts (Dashboard)

```sql
SELECT
    alert_id,
    timestamp,
    cow_id,
    alert_type,
    severity,
    title,
    details,
    (NOW() - timestamp) AS duration
FROM alerts
WHERE status = 'active'
ORDER BY severity DESC, timestamp DESC
LIMIT 50;
```

**Optimization:** Uses `idx_alerts_active` partial index.

---

### 5. Multi-Day Health Score Trend (7/14/30/90 Days)

```sql
WITH daily_scores AS (
    SELECT
        DATE_TRUNC('day', timestamp) AS day,
        cow_id,
        AVG(health_score) AS avg_score,
        MIN(health_score) AS min_score,
        MAX(health_score) AS max_score
    FROM health_scores
    WHERE cow_id = 1042
        AND timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY DATE_TRUNC('day', timestamp), cow_id
)
SELECT
    day,
    avg_score,
    min_score,
    max_score,
    LAG(avg_score, 1) OVER (ORDER BY day) AS prev_day_score,
    avg_score - LAG(avg_score, 1) OVER (ORDER BY day) AS day_over_day_change
FROM daily_scores
ORDER BY day DESC;
```

**Alternative (using continuous aggregate):**

```sql
SELECT *
FROM health_daily
WHERE cow_id = 1042
    AND day >= NOW() - INTERVAL '30 days'
ORDER BY day DESC;
```

---

### 6. Detect Prolonged Lying (Inactivity Alert)

```sql
WITH lying_periods AS (
    SELECT
        cow_id,
        timestamp,
        state,
        LAG(state) OVER (PARTITION BY cow_id ORDER BY timestamp) AS prev_state,
        CASE
            WHEN state = 'lying' AND LAG(state) OVER (PARTITION BY cow_id ORDER BY timestamp) != 'lying'
            THEN 1
            ELSE 0
        END AS lying_start
    FROM behavioral_states
    WHERE cow_id = 1042
        AND timestamp >= NOW() - INTERVAL '12 hours'
),
lying_sessions AS (
    SELECT
        cow_id,
        timestamp,
        SUM(lying_start) OVER (PARTITION BY cow_id ORDER BY timestamp) AS session_id
    FROM lying_periods
    WHERE state = 'lying'
)
SELECT
    cow_id,
    session_id,
    MIN(timestamp) AS session_start,
    MAX(timestamp) AS session_end,
    EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))/60 AS duration_minutes
FROM lying_sessions
GROUP BY cow_id, session_id
HAVING EXTRACT(EPOCH FROM (MAX(timestamp) - MIN(timestamp)))/60 > 240  -- >4 hours
ORDER BY session_start DESC;
```

---

### 7. Temperature-Activity Correlation Analysis

```sql
SELECT
    r.timestamp,
    r.cow_id,
    r.temperature,
    b.state,
    b.motion_intensity,
    CASE
        WHEN r.temperature > 39.5 AND b.motion_intensity < 0.2 THEN 'FEVER_RISK'
        WHEN r.temperature > 39.0 AND b.motion_intensity > 0.7 THEN 'HEAT_STRESS_RISK'
        WHEN r.temperature BETWEEN 38.0 AND 39.0 AND b.motion_intensity BETWEEN 0.3 AND 0.7 THEN 'NORMAL'
        ELSE 'MONITOR'
    END AS health_status
FROM raw_sensor_readings r
JOIN behavioral_states b
    ON r.cow_id = b.cow_id AND r.timestamp = b.timestamp
WHERE r.cow_id = 1042
    AND r.timestamp >= NOW() - INTERVAL '2 hours'
ORDER BY r.timestamp DESC;
```

---

### 8. Rumination Time Calculation (Daily)

```sql
SELECT
    DATE_TRUNC('day', timestamp) AS day,
    cow_id,
    SUM(CASE WHEN state = 'ruminating' THEN 1 ELSE 0 END) AS rumination_minutes,
    COUNT(*) AS total_minutes,
    ROUND(100.0 * SUM(CASE WHEN state = 'ruminating' THEN 1 ELSE 0 END) / COUNT(*), 2) AS rumination_percent
FROM behavioral_states
WHERE cow_id = 1042
    AND timestamp >= NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('day', timestamp), cow_id
ORDER BY day DESC;
```

**Expected Output:**
```
day         | cow_id | rumination_minutes | total_minutes | rumination_percent
------------|--------|-------------------|---------------|-------------------
2025-11-08  | 1042   | 485               | 1440          | 33.68
2025-11-07  | 1042   | 520               | 1440          | 36.11
2025-11-06  | 1042   | 490               | 1440          | 34.03
```

**Alert Trigger:** rumination_minutes < 240 (4 hours/day) → `abnormal_rumination` alert

---

### 9. Herd-Wide Health Ranking (Top 10 Lowest Scores)

```sql
SELECT
    cow_id,
    health_score,
    trend_direction,
    temperature_component,
    activity_component,
    behavior_component,
    timestamp
FROM health_scores
WHERE timestamp >= NOW() - INTERVAL '1 hour'
ORDER BY health_score ASC
LIMIT 10;
```

---

### 10. Alert Summary Report (Last 30 Days)

```sql
SELECT
    alert_type,
    severity,
    COUNT(*) AS total_alerts,
    COUNT(*) FILTER (WHERE status = 'active') AS active_alerts,
    COUNT(*) FILTER (WHERE status = 'resolved') AS resolved_alerts,
    AVG(EXTRACT(EPOCH FROM (COALESCE(resolved_at, NOW()) - timestamp))/3600) AS avg_resolution_hours
FROM alerts
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY alert_type, severity
ORDER BY total_alerts DESC;
```

---

## Performance Considerations

### Connection Pooling

**Recommended Setup (PgBouncer):**

```ini
[pgbouncer]
pool_mode = transaction
max_client_conn = 200
default_pool_size = 25
reserve_pool_size = 5
reserve_pool_timeout = 3
```

### Concurrent Access

**Read Concurrency:**
- Dashboard queries: 10-50 concurrent connections
- Alert generation: 5-10 background workers
- Analytics/reporting: 2-5 long-running queries

**Write Concurrency:**
- Sensor data ingestion: 1-5 concurrent writers (batched)
- Layer 1/2/3 processing: 5-10 concurrent writers

### Memory Configuration

```sql
-- Shared buffers (25% of RAM)
ALTER SYSTEM SET shared_buffers = '4GB';

-- Work memory (for sorting/joins)
ALTER SYSTEM SET work_mem = '64MB';

-- Maintenance work memory (for vacuuming)
ALTER SYSTEM SET maintenance_work_mem = '1GB';

-- Effective cache size (50-75% of RAM)
ALTER SYSTEM SET effective_cache_size = '12GB';
```

### Vacuum and Analyze

```sql
-- Auto-vacuum settings
ALTER SYSTEM SET autovacuum = on;
ALTER SYSTEM SET autovacuum_max_workers = 4;
ALTER SYSTEM SET autovacuum_naptime = '30s';

-- Manual analyze for better query planning
ANALYZE raw_sensor_readings;
ANALYZE behavioral_states;
ANALYZE physiological_metrics;
ANALYZE alerts;
ANALYZE health_scores;
```

### Monitoring Queries

```sql
-- Check hypertable chunk sizes
SELECT
    hypertable_name,
    chunk_name,
    range_start,
    range_end,
    pg_size_pretty(total_bytes) AS total_size
FROM timescaledb_information.chunks
ORDER BY hypertable_name, range_start DESC;

-- Check compression status
SELECT
    hypertable_name,
    compression_status,
    uncompressed_total_bytes,
    compressed_total_bytes,
    pg_size_pretty(compressed_total_bytes) AS compressed_size
FROM timescaledb_information.compression_settings;

-- Active queries
SELECT
    pid,
    now() - query_start AS duration,
    state,
    query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;
```

---

## References

### TimescaleDB Documentation
- Official Docs: https://docs.timescale.com/
- Best Practices: https://docs.timescale.com/timescaledb/latest/best-practices/
- Compression Guide: https://docs.timescale.com/timescaledb/latest/how-to-guides/compression/

### Project Documentation
- [description.md](../description.md) - Sensor parameters and system requirements
- [behavioral_sensor_signatures.md](behavioral_sensor_signatures.md) - Behavioral state definitions
- [schema.sql](../schema.sql) - Executable DDL statements

### Database Administration
- PostgreSQL Performance Tuning: https://wiki.postgresql.org/wiki/Performance_Optimization
- PgBouncer Setup: https://www.pgbouncer.org/config.html

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-08 | Artemis Health Team | Initial schema design with complete DDL and optimization strategies |

---

**End of Document**
