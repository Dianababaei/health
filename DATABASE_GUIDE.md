# SQLite Database Guide

## Database Location
```
i:\livestock\health\data\alert_state.db
```

## Quick Access Methods

### 1. GUI Tool (Easiest)
**DB Browser for SQLite** - https://sqlitebrowser.org/
- Open → Browse to `data/alert_state.db`
- Click tables to view data
- Run SQL queries in "Execute SQL" tab

### 2. Python Script (This Project)
```bash
python inspect_database.py
```
Interactive menu to:
- List tables
- View data
- Export to CSV
- Run queries
- Clear tables

### 3. Command Line
```bash
sqlite3 data/alert_state.db
```
Then use SQL commands:
```sql
.tables                    -- List all tables
.schema alerts             -- Show table structure
SELECT * FROM alerts;      -- View data
```

---

## Database Tables

### Table 1: `alerts`
**Purpose:** Store health alerts with lifecycle tracking

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `alert_id` | TEXT | Unique alert identifier (PRIMARY KEY) |
| `cow_id` | TEXT | Cow identifier (e.g., "COW_001") |
| `alert_type` | TEXT | Type: fever, heat_stress, inactivity, etc. |
| `severity` | TEXT | critical, high, medium, low |
| `confidence` | REAL | Detection confidence (0.0-1.0) |
| `status` | TEXT | active, acknowledged, resolved, false_positive |
| `created_at` | TEXT | When alert was created (ISO8601) |
| `updated_at` | TEXT | Last update timestamp (ISO8601) |
| `resolution_notes` | TEXT | Notes when resolved |
| `sensor_values` | TEXT | JSON of sensor values at detection |
| `detection_details` | TEXT | JSON of detection details |
| `timestamp` | TEXT | Alert detection timestamp (ISO8601) |

**Example Data:**
```sql
SELECT alert_id, cow_id, alert_type, severity, status, timestamp
FROM alerts
LIMIT 5;
```

**Sample Row:**
```
alert_id: COW_001_fever_0_1731642345.123
cow_id: COW_001
alert_type: fever
severity: critical
status: active
timestamp: 2025-11-15T01:05:45.123
sensor_values: {"temperature": 40.2, "motion": 0.03}
```

**Common Queries:**
```sql
-- Get all active alerts for COW_001
SELECT * FROM alerts
WHERE cow_id = 'COW_001' AND status = 'active'
ORDER BY timestamp DESC;

-- Count alerts by type
SELECT alert_type, COUNT(*) as count
FROM alerts
GROUP BY alert_type;

-- Get fever alerts from last 24 hours
SELECT * FROM alerts
WHERE alert_type = 'fever'
  AND datetime(timestamp) >= datetime('now', '-1 day')
ORDER BY timestamp DESC;
```

---

### Table 2: `state_history`
**Purpose:** Track alert status changes over time

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment ID (PRIMARY KEY) |
| `alert_id` | TEXT | Reference to alerts table |
| `old_status` | TEXT | Previous status (NULL for new alerts) |
| `new_status` | TEXT | New status |
| `changed_at` | TEXT | When status changed (ISO8601) |
| `notes` | TEXT | Notes about the change |

**Example Data:**
```sql
SELECT * FROM state_history
WHERE alert_id = 'COW_001_fever_0_1731642345.123'
ORDER BY changed_at;
```

**Sample Rows:**
```
id: 1
alert_id: COW_001_fever_0_1731642345.123
old_status: NULL
new_status: active
changed_at: 2025-11-15T01:05:45.123
notes: Alert created

id: 2
alert_id: COW_001_fever_0_1731642345.123
old_status: active
new_status: acknowledged
changed_at: 2025-11-15T01:15:00.000
notes: Alert acknowledged
```

**Common Queries:**
```sql
-- Get status change history for an alert
SELECT old_status, new_status, changed_at, notes
FROM state_history
WHERE alert_id = 'COW_001_fever_0_1731642345.123'
ORDER BY changed_at;

-- Count status transitions
SELECT old_status, new_status, COUNT(*) as count
FROM state_history
GROUP BY old_status, new_status;
```

---

### Table 3: `health_scores`
**Purpose:** Store calculated health scores over time

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-increment ID (PRIMARY KEY) |
| `timestamp` | TEXT | Score calculation time (ISO8601) |
| `cow_id` | TEXT | Cow identifier |
| `total_score` | REAL | Overall health score (0-100) |
| `temperature_score` | REAL | Temperature component (0-1) |
| `activity_score` | REAL | Activity component (0-1) |
| `behavioral_score` | REAL | Behavioral component (0-1) |
| `alert_score` | REAL | Alert penalty component (0-1) |
| `health_category` | TEXT | excellent, good, moderate, poor |
| `confidence` | REAL | Overall confidence (0-1) |
| `metadata` | TEXT | JSON with full details |
| `created_at` | TEXT | When record was created (ISO8601) |

**Example Data:**
```sql
SELECT timestamp, cow_id, total_score, health_category
FROM health_scores
ORDER BY timestamp DESC
LIMIT 5;
```

**Sample Row:**
```
id: 1
timestamp: 2025-11-15T01:05:45.123
cow_id: COW_001
total_score: 65.3
temperature_score: 0.75
activity_score: 0.60
behavioral_score: 0.70
alert_score: 0.50
health_category: good
confidence: 0.85
```

**Common Queries:**
```sql
-- Get latest health score for COW_001
SELECT * FROM health_scores
WHERE cow_id = 'COW_001'
ORDER BY timestamp DESC
LIMIT 1;

-- Get health score trend (last 7 days)
SELECT timestamp, total_score, health_category
FROM health_scores
WHERE cow_id = 'COW_001'
  AND datetime(timestamp) >= datetime('now', '-7 days')
ORDER BY timestamp;

-- Average health score by day
SELECT date(timestamp) as day, AVG(total_score) as avg_score
FROM health_scores
WHERE cow_id = 'COW_001'
GROUP BY date(timestamp)
ORDER BY day DESC;

-- Count by health category
SELECT health_category, COUNT(*) as count
FROM health_scores
GROUP BY health_category;
```

---

## Maintenance Tasks

### View Database Size
```bash
# Windows
dir data\alert_state.db

# Linux/Mac
ls -lh data/alert_state.db
```

### Backup Database
```bash
# Copy the file
cp data/alert_state.db data/alert_state_backup_2025-11-15.db
```

### Clear All Data (Reset)
```python
# Using inspect_database.py
python inspect_database.py
# Choose option 8: Clear entire database

# OR manually
sqlite3 data/alert_state.db
DELETE FROM alerts;
DELETE FROM state_history;
DELETE FROM health_scores;
.quit
```

### Export to CSV
```python
# Using inspect_database.py
python inspect_database.py
# Choose option 5: Export table to CSV

# OR using pandas
import pandas as pd
import sqlite3

conn = sqlite3.connect('data/alert_state.db')
df = pd.read_sql_query("SELECT * FROM alerts", conn)
df.to_csv('alerts_export.csv', index=False)
conn.close()
```

---

## Database Relationships

```
┌─────────────────────┐
│ alerts              │
│─────────────────────│
│ alert_id (PK)       │◄──┐
│ cow_id              │   │
│ alert_type          │   │
│ severity            │   │
│ status              │   │
│ timestamp           │   │
│ ...                 │   │
└─────────────────────┘   │
                          │
                          │
┌─────────────────────┐   │
│ state_history       │   │
│─────────────────────│   │
│ id (PK)             │   │
│ alert_id (FK)       │───┘
│ old_status          │
│ new_status          │
│ changed_at          │
│ notes               │
└─────────────────────┘

┌─────────────────────┐
│ health_scores       │
│─────────────────────│
│ id (PK)             │
│ timestamp           │
│ cow_id              │
│ total_score         │
│ component scores... │
│ health_category     │
│ metadata            │
└─────────────────────┘
```

---

## Usage Examples

### Get Current Health Status
```sql
SELECT
    h.cow_id,
    h.total_score,
    h.health_category,
    COUNT(a.alert_id) as active_alerts
FROM health_scores h
LEFT JOIN alerts a ON h.cow_id = a.cow_id AND a.status = 'active'
WHERE h.cow_id = 'COW_001'
GROUP BY h.cow_id, h.total_score, h.health_category
ORDER BY h.timestamp DESC
LIMIT 1;
```

### Find Degrading Health Trends
```sql
SELECT
    cow_id,
    date(timestamp) as day,
    AVG(total_score) as avg_score
FROM health_scores
GROUP BY cow_id, date(timestamp)
HAVING AVG(total_score) < 60  -- Below "good" threshold
ORDER BY day DESC;
```

### Alert Summary Report
```sql
SELECT
    alert_type,
    severity,
    status,
    COUNT(*) as count,
    MIN(timestamp) as first_seen,
    MAX(timestamp) as last_seen
FROM alerts
WHERE cow_id = 'COW_001'
GROUP BY alert_type, severity, status
ORDER BY severity, alert_type;
```

---

## Troubleshooting

### Database Locked Error
```
# Close all connections, then:
sqlite3 data/alert_state.db
.timeout 5000
# Try your query again
```

### Corrupted Database
```bash
# Check integrity
sqlite3 data/alert_state.db "PRAGMA integrity_check;"

# If corrupted, restore from backup
cp data/alert_state_backup.db data/alert_state.db
```

### Reset Database Schema
```python
# Delete database and let it recreate
import os
os.remove('data/alert_state.db')

# Restart dashboard and upload data
# Database will be recreated automatically
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| Open database | `sqlite3 data/alert_state.db` |
| List tables | `.tables` |
| Show schema | `.schema alerts` |
| View data | `SELECT * FROM alerts LIMIT 10;` |
| Count rows | `SELECT COUNT(*) FROM alerts;` |
| Export CSV | `.mode csv`<br>`.output alerts.csv`<br>`SELECT * FROM alerts;` |
| Quit | `.quit` |
| Python GUI | `python inspect_database.py` |

---

## Integration with Dashboard

### Where Data Comes From:
1. **Upload CSV** → Dashboard processes
2. **Layer 3** → Creates alerts → Saves to `alerts` table
3. **Dashboard Metrics** → Calculates health score → Saves to `health_scores` table
4. **User Actions** → Status changes → Saves to `state_history` table

### Where Data Goes:
1. **Home Page** → Loads latest health score from `health_scores`
2. **Health Analysis Page** → Queries `health_scores` for trends
3. **Alerts Page** → Queries `alerts` and `state_history` for display

### Data Flow:
```
CSV Upload
    ↓
Processing (3 Layers)
    ↓
SQLite Database
    ↓
Dashboard Display
```
