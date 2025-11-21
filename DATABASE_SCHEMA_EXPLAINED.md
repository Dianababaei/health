# Database Schema Documentation

Complete guide to the SQLite database structure, table relationships, and data lifecycle.

## Database Overview

**Database File:** `data/alert_state.db`
**Type:** SQLite 3
**Purpose:** Store health scores, alerts, and alert state history for livestock health monitoring

**Tables:**
1. `health_scores` - Health score records (one per upload)
2. `alerts` - Active and historical alerts
3. `state_history` - Alert lifecycle audit trail
4. `sqlite_sequence` - Auto-increment tracking (SQLite internal)

---

## Table 1: `health_scores`

### Purpose
Stores calculated health scores from uploaded sensor data. Each CSV upload creates **ONE** health score record representing the overall health at that point in time.

### Schema

```sql
CREATE TABLE health_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,              -- When the data was collected
    cow_id TEXT NOT NULL,                 -- Animal identifier (currently "COW_001")
    total_score REAL NOT NULL,            -- Overall health score (0-100)
    temperature_score REAL NOT NULL,      -- Temperature component (0-1)
    activity_score REAL NOT NULL,         -- Activity component (0-1)
    behavioral_score REAL NOT NULL,       -- Behavioral component (0-1)
    alert_score REAL NOT NULL,            -- Alert penalty component (0-1)
    health_category TEXT NOT NULL,        -- Category: excellent/good/moderate/poor
    confidence REAL NOT NULL,             -- Confidence in calculation (0-1)
    metadata TEXT,                        -- Full JSON data with all details
    created_at TEXT NOT NULL              -- When record was inserted
)
```

### Indexes
- `idx_health_scores_cow_id` - Fast queries by cow_id
- `idx_health_scores_timestamp` - Fast queries by time range
- `idx_health_scores_cow_timestamp` - Fast queries by cow + time

### When Records Are Created

**Frequency:** Once per CSV upload
**Trigger:** User uploads CSV through dashboard → Processing complete → Health score calculated and saved

**Timeline:**
```
Upload CSV → Layer 1 (Classify) → Layer 2 (Temperature) → Layer 3 (Alerts)
→ Calculate Health Score → Save to health_scores table
```

**Example:**
- Upload 1 (Nov 15, 8:00 AM): Creates health score #1
- Upload 2 (Nov 15, 2:00 PM): Creates health score #2
- Upload 3 (Nov 16, 8:00 AM): Creates health score #3

### When Records Are Updated

**NEVER.** Health scores are **immutable**. Each upload creates a new record. This allows tracking health trends over time.

### Data Lifecycle

1. **Creation:** On CSV upload completion
2. **Storage:** Permanent (until manually deleted)
3. **Deletion:**
   - Manual: Delete database file or use `inspect_database.py`
   - Automatic: Could implement `delete_old_scores(days=90)` to remove old records

### Sample Data

```
id | timestamp                  | cow_id  | total_score | health_category | created_at
1  | 2025-11-15T08:00:00       | COW_001 | 100.0       | excellent       | 2025-11-15T08:05:23
2  | 2025-11-15T14:00:00       | COW_001 | 72.3        | good            | 2025-11-15T14:12:45
3  | 2025-11-16T08:00:00       | COW_001 | 85.5        | excellent       | 2025-11-16T08:03:11
```

### Component Scores (0-1 scale)

All component scores are normalized to 0-1 range:
- **1.0** = Perfect (no issues)
- **0.5** = Moderate issues
- **0.0** = Severe issues

**Total Score Calculation:**
```
total_score (0-100) =
    temperature_score × 30 +
    activity_score × 25 +
    behavioral_score × 25 +
    alert_score × 20
```

### Health Categories

| Score Range | Category   | Color  | Meaning                |
|-------------|------------|--------|------------------------|
| 80-100      | excellent  | Green  | Healthy, normal care   |
| 60-79       | good       | Yellow | Minor concerns         |
| 40-59       | moderate   | Orange | Increased monitoring   |
| 0-39        | poor       | Red    | Immediate attention    |

### Metadata JSON Structure

The `metadata` column contains the complete health score calculation details:

```json
{
    "timestamp": "2025-11-15T08:00:00",
    "cow_id": "COW_001",
    "total_score": 72.3,
    "temperature_component": 0.91,
    "activity_component": 1.00,
    "behavioral_component": 0.48,
    "alert_component": 0.40,
    "health_category": "good",
    "confidence": 0.85,
    "weights": {
        "temperature_stability": 0.30,
        "activity_level": 0.25,
        "behavioral_patterns": 0.25,
        "alert_frequency": 0.20
    },
    "metadata": {
        "calculation_timestamp": "2025-11-15T08:05:23",
        "warnings": [
            "No rumination detected",
            "Elevated temperature in 15.2% of readings"
        ],
        "baseline_temp": 38.5,
        "data_points": 20160,
        "active_alerts": 2
    }
}
```

---

## Table 2: `alerts`

### Purpose
Stores health alerts detected by the system. Tracks alert lifecycle from creation through resolution.

### Schema

```sql
CREATE TABLE alerts (
    alert_id TEXT PRIMARY KEY,           -- Unique ID: COW_001_fever_2025-11-15T10-30-00
    cow_id TEXT NOT NULL,                -- Animal identifier
    alert_type TEXT NOT NULL,            -- Type: fever, inactivity, heat_stress
    severity TEXT NOT NULL,              -- Severity: critical, warning, info
    confidence REAL,                     -- Detection confidence (0-1)
    status TEXT NOT NULL,                -- Lifecycle: active, acknowledged, resolved
    created_at TEXT NOT NULL,            -- When alert was first created
    updated_at TEXT NOT NULL,            -- Last status change
    resolution_notes TEXT,               -- Notes when resolved/dismissed
    sensor_values TEXT,                  -- JSON: Sensor readings that triggered alert
    detection_details TEXT,              -- JSON: Detection algorithm details
    timestamp TEXT NOT NULL              -- When the condition was detected
)
```

### Indexes
- `idx_alerts_cow_id` - Fast queries by cow
- `idx_alerts_status` - Fast queries by status
- `idx_alerts_cow_status` - Fast queries by cow + status

### Alert Types

| Alert Type   | Severity | Trigger Condition                    |
|--------------|----------|--------------------------------------|
| fever        | critical | Temperature > 39.5°C sustained       |
| inactivity   | warning  | Low movement for extended period     |
| heat_stress  | warning  | High temp + environmental factors    |

### Alert Lifecycle (Status Field)

```
active → acknowledged → resolved
   ↓           ↓
   └─────> false_positive
```

**States:**
1. **active** - Newly detected, requires attention
2. **acknowledged** - Seen by user, being monitored
3. **resolved** - Issue addressed or cleared
4. **false_positive** - Incorrectly detected

### When Records Are Created

**Frequency:** Variable (only when conditions are detected)
**Trigger:** Layer 3 alert detection during CSV upload

**Example Timeline:**
```
Upload 1 (Normal data):
  - Temperature: 38.5°C (normal)
  - Activity: Good
  - Result: 0 alerts created

Upload 2 (Fever data):
  - Temperature: 40.0°C (high)
  - Activity: Low
  - Result: 2 alerts created
    - fever (critical)
    - inactivity (warning)

Upload 3 (Recovery data):
  - Temperature: 38.5°C (normal)
  - Activity: Good
  - Result: 0 new alerts
  - Existing alerts can be marked as "resolved"
```

### When Records Are Updated

**Status Changes:**
1. **User acknowledges alert** → status: active → acknowledged
2. **User resolves alert** → status: acknowledged → resolved
3. **User marks false positive** → status: active → false_positive

**Updated Fields:**
- `status` - Changed to new status
- `updated_at` - Set to current timestamp
- `resolution_notes` - User's notes (optional)

**State History:** Every status change is logged to `state_history` table

### Sample Data

```
alert_id                              | cow_id  | alert_type | severity | status        | timestamp
COW_001_fever_2025-11-15T10-30-00    | COW_001 | fever      | critical | active        | 2025-11-15T10:30:00
COW_001_inactivity_2025-11-15T14-00  | COW_001 | inactivity | warning  | acknowledged  | 2025-11-15T14:00:00
COW_001_fever_2025-11-14T09-15-00    | COW_001 | fever      | critical | resolved      | 2025-11-14T09:15:00
```

### Sensor Values JSON

The `sensor_values` column contains the readings that triggered the alert:

```json
{
    "temperature": 40.2,
    "fxa": 0.05,
    "mya": 0.03,
    "rza": 0.02,
    "state": "lying"
}
```

### Detection Details JSON

The `detection_details` column contains algorithm-specific information:

```json
{
    "source": "upload",
    "detector": "ImmediateAlertDetector",
    "threshold_exceeded": "temperature > 39.5",
    "duration": "sustained for 30 minutes",
    "baseline": 38.5
}
```

---

## Table 3: `state_history`

### Purpose
Audit trail for alert lifecycle changes. Tracks every status transition with timestamp and notes.

### Schema

```sql
CREATE TABLE state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT NOT NULL,              -- Foreign key to alerts table
    old_status TEXT,                     -- Previous status (NULL for creation)
    new_status TEXT NOT NULL,            -- New status
    changed_at TEXT NOT NULL,            -- When change occurred
    notes TEXT                           -- Reason for change (user notes)
)
```

### Index
- `idx_state_history_alert_id` - Fast queries for alert's history

### When Records Are Created

**Frequency:** Every time an alert status changes
**Trigger:** Alert creation or status update

**Timeline:**
```
Alert Created:
  - old_status: NULL
  - new_status: active
  - notes: "Alert created"

User Acknowledges:
  - old_status: active
  - new_status: acknowledged
  - notes: "Vet called, monitoring"

User Resolves:
  - old_status: acknowledged
  - new_status: resolved
  - notes: "Temperature normalized, recovery confirmed"
```

### When Records Are Updated

**NEVER.** State history is **append-only** for audit purposes.

### Sample Data

```
id | alert_id                          | old_status    | new_status   | changed_at           | notes
1  | COW_001_fever_2025-11-15T10-30   | NULL          | active       | 2025-11-15T10:30:00 | Alert created
2  | COW_001_fever_2025-11-15T10-30   | active        | acknowledged | 2025-11-15T10:35:12 | Vet called
3  | COW_001_fever_2025-11-15T10-30   | acknowledged  | resolved     | 2025-11-15T18:22:45 | Temp normal
```

### Complete Alert Lifecycle Example

**Alert:** Fever detected at 10:30 AM

**state_history entries:**
1. **10:30:00** - Created (NULL → active)
2. **10:35:12** - User acknowledged (active → acknowledged) - "Vet notified"
3. **18:22:45** - User resolved (acknowledged → resolved) - "Temperature back to normal"

This provides complete traceability of alert handling.

---

## Table 4: `sqlite_sequence`

### Purpose
SQLite internal table for tracking auto-increment values.

### Schema

```sql
CREATE TABLE sqlite_sequence (
    name TEXT,      -- Table name
    seq INTEGER     -- Next auto-increment value
)
```

### When Records Are Created/Updated

**Automatic.** Managed by SQLite when tables with `AUTOINCREMENT` columns insert rows.

### Sample Data

```
name             | seq
health_scores    | 15
state_history    | 42
```

**Meaning:**
- Next `health_scores.id` will be 16
- Next `state_history.id` will be 43

### Do Not Modify
This table is managed by SQLite. **Never manually edit** this table.

---

## Data Relationships

### Entity Relationship Diagram

```
┌─────────────────┐
│  health_scores  │
│  (one per upload)
│                 │
│  id (PK)        │
│  timestamp      │
│  cow_id         │
│  total_score    │
│  ...            │
└─────────────────┘
         │
         │ (indirect relationship via cow_id + timestamp)
         │
         ▼
┌─────────────────┐          ┌──────────────────┐
│     alerts      │◄─────────│  state_history   │
│ (0-N per upload)│          │  (audit trail)   │
│                 │          │                  │
│  alert_id (PK)  │          │  id (PK)         │
│  cow_id         │          │  alert_id (FK)   │
│  alert_type     │          │  old_status      │
│  severity       │          │  new_status      │
│  status         │          │  changed_at      │
│  timestamp      │          │  notes           │
│  ...            │          │                  │
└─────────────────┘          └──────────────────┘
```

### Relationships

1. **health_scores ↔ alerts**: Implicit relationship via `cow_id` and `timestamp`
   - Health score references alerts that were active during calculation
   - Not a database foreign key (for flexibility)

2. **alerts → state_history**: One-to-many
   - Each alert can have multiple state history entries
   - Foreign key: `state_history.alert_id` → `alerts.alert_id`

---

## Data Queries

### Common Queries

**Get Latest Health Score:**
```sql
SELECT * FROM health_scores
WHERE cow_id = 'COW_001'
ORDER BY timestamp DESC
LIMIT 1;
```

**Get Active Alerts:**
```sql
SELECT * FROM alerts
WHERE cow_id = 'COW_001' AND status = 'active'
ORDER BY timestamp DESC;
```

**Get Alert History:**
```sql
SELECT a.alert_type, a.severity, sh.old_status, sh.new_status, sh.changed_at, sh.notes
FROM alerts a
JOIN state_history sh ON a.alert_id = sh.alert_id
WHERE a.alert_id = 'COW_001_fever_2025-11-15T10-30-00'
ORDER BY sh.changed_at;
```

**Get Health Trend (Last 30 Days):**
```sql
SELECT timestamp, total_score, health_category
FROM health_scores
WHERE cow_id = 'COW_001'
  AND timestamp >= date('now', '-30 days')
ORDER BY timestamp;
```

**Count Alerts by Type:**
```sql
SELECT alert_type, COUNT(*) as count
FROM alerts
WHERE cow_id = 'COW_001'
GROUP BY alert_type;
```

---

## Data Volume Estimates

### Expected Growth Rates

**Scenario:** 1 cow, daily uploads for 1 year

| Table          | Records/Upload | Daily | Monthly | Yearly |
|----------------|----------------|-------|---------|--------|
| health_scores  | 1              | 1     | 30      | 365    |
| alerts         | 0-5 (avg 0.5)  | 0.5   | 15      | 180    |
| state_history  | 0-15 (avg 2)   | 2     | 60      | 730    |

**Database Size:**
- 1 year: ~5 MB
- 5 years: ~25 MB
- 10 years: ~50 MB

**Conclusion:** SQLite is perfectly suitable for this use case.

---

## Maintenance Operations

### Backup Database

```bash
# Windows
copy data\alert_state.db data\backups\alert_state_2025-11-15.db

# Linux/Mac
cp data/alert_state.db data/backups/alert_state_2025-11-15.db
```

### Clear All Data

```bash
# Delete database file
del data\alert_state.db  # Windows
rm data/alert_state.db   # Linux/Mac

# Next upload will recreate empty database
```

### Clear Old Health Scores

```python
from src.health_intelligence.logging.health_score_manager import HealthScoreManager

manager = HealthScoreManager()
deleted = manager.delete_old_scores(days=90)  # Delete scores older than 90 days
print(f"Deleted {deleted} old records")
```

### Export to CSV

```bash
python inspect_database.py
# Choose option 5: Export table to CSV
# Enter table name: health_scores
```

---

## Best Practices

### When to Create Records

**health_scores:**
- ✅ After every CSV upload
- ❌ Don't update existing records (create new ones)

**alerts:**
- ✅ When Layer 3 detects anomalies
- ❌ Don't create duplicates for same condition

**state_history:**
- ✅ Every alert status change
- ✅ Include meaningful notes
- ❌ Don't delete history (it's an audit trail)

### Data Retention

**Recommended:**
- Keep last 90 days of health scores for trending
- Keep all critical alerts (fever, emergency) indefinitely
- Keep resolved alerts for 30 days
- Keep state history indefinitely (small size)

### Performance Tips

1. **Use indexes** - Already created, don't remove them
2. **Query by cow_id + timestamp** - Uses composite index
3. **Limit result sets** - Use `LIMIT` for recent data
4. **Vacuum database** - Run `VACUUM;` occasionally to reclaim space

---

## Migration Notes

### From PostgreSQL (Deprecated)

The system previously used PostgreSQL but was migrated to SQLite for simplicity.

**Key Changes:**
- Single file database (no server needed)
- Same schema and column names
- Connections created per-query (no connection pooling)
- All data migrated to `data/alert_state.db`

### Schema Versioning

**Current Version:** 1.0
**Created:** November 2025

**Future Schema Changes:**
- Add migration scripts in `migrations/` folder
- Document version changes
- Provide upgrade path for existing databases
