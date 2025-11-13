# Data Storage - Why Alerts Are in Two Places

## The Issue You Found

**Problem:**
- Home page shows 1 alert ✓
- Alerts page shows 0 alerts ✗

**Reason:** Different data sources!

---

## Two Storage Locations for Alerts

### 1. JSON Files (for Home Page)
**Location:** `data/simulation/DEMO_COW_001_alerts.json`

**Format:**
```json
[
  {
    "timestamp": "2025-10-31 15:30:00",
    "cow_id": "DEMO_COW_001",
    "alert_type": "fever",
    "severity": "critical"
  }
]
```

**Who reads this:**
- Home page (`0_Home.py`)
- Health Analysis page (`3_Health_Analysis.py`)

**Why:** Simple file-based storage for demo/simulation data

---

### 2. SQLite Database (for Alerts Page)
**Location:** `data/alert_state.db`

**Table:** `alerts`

**Columns:**
- alert_id
- cow_id
- alert_type
- severity
- status (active/acknowledged/resolved)
- created_at
- updated_at
- etc.

**Who reads this:**
- Alerts page (`2_Alerts.py`)

**Why:** Alerts page needs advanced features:
- Acknowledge alerts
- Resolve alerts
- Track status changes
- Alert history
- These require database queries

---

## Why Both?

**Flexibility:**
- JSON files: Easy for testing/demo (no database setup)
- Database: Required for production features

**Page Requirements:**
- **Home page:** Just needs to DISPLAY alerts → JSON is enough
- **Alerts page:** Needs to MANAGE alerts → Database required

---

## The Fix Applied

When generating demo data, I now save to BOTH:

**In `generate_demo_data.py`:**

```python
# Save to JSON (for Home page)
with open('alerts.json', 'w') as f:
    json.dump(alerts, f)

# ALSO save to database (for Alerts page)
state_manager = AlertStateManager(db_path="data/alert_state.db")
for alert in alerts:
    state_manager.create_alert(alert)
```

---

## How Upload Works

**When you upload alerts JSON in Home page:**

```python
# dashboard/pages/0_Home.py, line 102-135

# 1. Save to JSON file
with open('data/simulation/alerts.json', 'w') as f:
    json.dump(alerts_data, f)

# 2. ALSO save to database
state_manager = AlertStateManager(db_path="data/alert_state.db")
for alert in alerts_data:
    state_manager.create_alert(alert)
```

This ensures BOTH pages can see the alerts!

---

## Current Demo Data Status

✅ **JSON file:** `data/simulation/DEMO_COW_001_alerts.json` (2 alerts)
✅ **Database:** `data/alert_state.db` (2 alerts)

**Both have:**
- 1 fever alert (critical)
- 1 inactivity alert (warning)

---

## Verify Both Sources

### Check JSON:
```bash
cat data/simulation/DEMO_COW_001_alerts.json
```

### Check Database:
```bash
cd src
python -c "from health_intelligence.logging import AlertStateManager; sm = AlertStateManager(db_path='../data/alert_state.db'); alerts = sm.query_alerts(); print(f'{len(alerts)} alerts in database')"
```

---

## Production Scenario

**In production with real sensors:**

1. **Sensor data flows in** → Saved to TimescaleDB

2. **Alert detection runs** → Detects fever/heat stress/etc.

3. **Alerts saved to:**
   - Database (`alert_state.db`) ✓ Always
   - Optional: Also export to JSON for backup

4. **Both pages work:**
   - Home page: Can read from database OR JSON
   - Alerts page: Always reads from database

---

## Summary

| Page | Data Source | Why |
|------|------------|-----|
| Home | JSON files first, then DB | Simple display |
| Alerts | Database only | Needs management features |

**Solution:** Save to BOTH when generating/uploading demo data.

**Now:** Both pages show the same 2 alerts! ✓
