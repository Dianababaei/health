# End-to-End Testing Guide

## System Reset Complete ✅

All data and caches have been cleared:
- ✅ `data/simulation/` - Empty
- ✅ `data/alert_state.db` - Deleted
- ✅ Python caches - Cleared

Ready for fresh end-to-end test!

---

## Test Plan: Raw Data → Layer 1 → Layer 2 → Layer 3 → Dashboard

We'll test the complete pipeline:

```
1. Generate Raw Sensor Data
   ↓
2. Verify Layer 1: Behavior Classification
   ↓
3. Verify Layer 2: Temperature Analysis
   ↓
4. Verify Layer 3: Alert Detection
   ↓
5. Verify Dashboard Display
```

---

## Step 1: Generate Demo Data

### Run the Generator

```bash
python generate_demo_data.py
```

### What to Expect

You should see:

```
======================================================================
ARTEMIS DEMO DATA GENERATOR
======================================================================

Configuration:
  Cow ID: DEMO_COW_001
  Duration: 14 days
  Baseline Temperature: 38.5°C
  Fever: Day 3, Duration 2 days, Temp 40.0°C

[1/5] Generating 14 days of baseline behavioral data...
  [OK] Generated 20160 sensor readings

[2/5] Injecting fever condition...
  [OK] Fever injected from sample 2880 to 5760

[3/5] Detecting health alerts...
  [OK] Detected X alerts
    - Fever: Y
    - Inactivity: Z

[4/5] Analyzing health trends...
  [OK] Health trend: DECLINING
  [OK] Confidence: 85%

[5/5] Saving demo data files...
  [OK] data/simulation/DEMO_COW_001_sensor_data.csv (20160 rows)
  [OK] data/simulation/DEMO_COW_001_alerts.json (X alerts)
  [OK] Saved X alerts to database
  [OK] data/simulation/DEMO_COW_001_metadata.json
  [OK] data/simulation/DEMO_COW_001_trend_report.json

======================================================================
DEMO DATA GENERATED SUCCESSFULLY!
======================================================================
```

### Verify Files Created

```bash
ls -lh data/simulation/
```

**Expected output:**
```
DEMO_COW_001_sensor_data.csv      (~3.4 MB)
DEMO_COW_001_alerts.json          (<1 KB)
DEMO_COW_001_metadata.json        (<1 KB)
DEMO_COW_001_trend_report.json    (<1 KB)
```

---

## Step 2: Verify Layer 1 - Behavior Classification

### Check Sensor Data Has States

```bash
head -5 data/simulation/DEMO_COW_001_sensor_data.csv
```

**Expected columns:**
```
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg,state
2025-11-14 13:40:00,38.5,-0.04,0.01,-0.88,-2.88,0.14,1.87,lying
2025-11-14 13:41:00,38.5,-0.01,0.01,-0.87,-1.19,-1.14,1.86,lying
2025-11-14 13:42:00,38.6,-0.02,-0.00,-0.87,0.16,2.48,-1.47,lying
2025-11-14 13:43:00,38.5,0.00,-0.00,-0.84,-2.94,-1.90,-0.96,lying
```

**✅ Verify:** The `state` column exists with values like:
- `lying`
- `standing`
- `walking`
- `ruminating`
- `feeding`

### Count States Distribution

```bash
# Extract state column and count occurrences (Windows)
python -c "import pandas as pd; df = pd.read_csv('data/simulation/DEMO_COW_001_sensor_data.csv'); print(df['state'].value_counts())"
```

**Expected output (approximate):**
```
lying         ~10000  (cow rests ~50% of time)
standing      ~5000   (cow stands ~25% of time)
walking       ~2000   (cow walks ~10% of time)
ruminating    ~2000   (cow ruminates ~10% of time)
feeding       ~1000   (cow feeds ~5% of time)
```

**✅ LAYER 1 PASSED** if you see behavioral states classified!

---

## Step 3: Verify Layer 2 - Temperature Analysis

### Check Temperature Data

```bash
python -c "import pandas as pd; df = pd.read_csv('data/simulation/DEMO_COW_001_sensor_data.csv'); print(f'Temperature range: {df[\"temperature\"].min():.2f}°C - {df[\"temperature\"].max():.2f}°C'); print(f'Mean: {df[\"temperature\"].mean():.2f}°C')"
```

**Expected output:**
```
Temperature range: 38.20°C - 40.50°C
Mean: 38.70°C
```

### Verify Fever Period (Day 3-4)

```bash
# Check samples during fever period (rows 2880-5760)
python -c "import pandas as pd; df = pd.read_csv('data/simulation/DEMO_COW_001_sensor_data.csv'); fever_df = df.iloc[2880:5760]; print(f'Fever period temp: {fever_df[\"temperature\"].mean():.2f}°C (should be ~40°C)')"
```

**Expected output:**
```
Fever period temp: 40.02°C (should be ~40°C)
```

### Check Trend Report

```bash
cat data/simulation/DEMO_COW_001_trend_report.json
```

**Expected content:**
```json
{
  "overall_trend": "declining",
  "overall_confidence": 0.85,
  "temperature_trend": {
    "trend": "elevated",
    "deviation_from_baseline": 1.5,
    "confidence": 0.90
  },
  "activity_trend": {
    "trend": "reduced",
    "confidence": 0.80
  }
}
```

**✅ LAYER 2 PASSED** if temperature analysis shows fever detection!

---

## Step 4: Verify Layer 3 - Alert Detection

### Check Alerts JSON

```bash
cat data/simulation/DEMO_COW_001_alerts.json
```

**Expected output:**
```json
[
  {
    "timestamp": "2025-11-16 13:42:00",
    "cow_id": "DEMO_COW_001",
    "alert_type": "fever",
    "severity": "critical"
  },
  {
    "timestamp": "2025-11-16 14:15:00",
    "cow_id": "DEMO_COW_001",
    "alert_type": "inactivity",
    "severity": "warning"
  }
]
```

**Alert types expected:**
- **Fever alert** - Temperature >39.5°C + motion <0.15
- **Inactivity alert** - Prolonged stillness (sick cow lying down)

### Count Alerts

```bash
python -c "import json; alerts = json.load(open('data/simulation/DEMO_COW_001_alerts.json')); print(f'Total alerts: {len(alerts)}'); print(f'Fever: {sum(1 for a in alerts if a[\"alert_type\"]==\"fever\")}'); print(f'Inactivity: {sum(1 for a in alerts if a[\"alert_type\"]==\"inactivity\")}')"
```

**Expected output:**
```
Total alerts: 2-10 (depends on deduplication)
Fever: 1-5
Inactivity: 1-5
```

### Verify Alert Database

```bash
python -c "import sqlite3; conn = sqlite3.connect('data/alert_state.db'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM alerts'); print(f'Alerts in database: {cursor.fetchone()[0]}'); conn.close()"
```

**Expected output:**
```
Alerts in database: 2-10
```

**✅ LAYER 3 PASSED** if alerts are detected and saved!

---

## Step 5: Verify Dashboard Display

### Start the Dashboard

```bash
streamlit run dashboard/app.py
```

### Expected Browser URL

```
http://localhost:8501
```

### Home Page Tests

#### Test 5.1: Health Score Display

**Expected:**
- Large gauge showing health score: **40-60** (declining due to fever)
- Status: **"NEEDS ATTENTION"** or **"AT RISK"**

#### Test 5.2: Critical Alerts Display

**Expected:**
- Red alert box showing **1-2 critical alerts**
- "🔴 CRITICAL" status
- Button: "🚨 View All Alerts"

#### Test 5.3: Live Animal Feed

**Expected:**
- Shows DEMO_COW_001
- Health status: **"🔴 FEVER"** during fever period
- Temperature: **40.0°C - 40.5°C** during fever
- State: **lying** (low activity due to fever)
- Activity bar: Low (**10-20%**)

#### Test 5.4: Data Loading Message

**Expected:**
```
✅ Loaded X alerts
```

If you see:
```
ℹ️ No alerts found
```
→ **PROBLEM**: Alerts not loaded correctly

---

### Alerts Page Tests

Click: **Pages > Alerts** (in sidebar)

#### Test 5.5: Alert List

**Expected:**
- **2+ alerts** displayed in table
- Columns: Time, Cow ID, Type, Severity, Status
- Alert types: fever, inactivity

#### Test 5.6: Alert Details

Click on an alert to expand:

**Expected details:**
- Timestamp
- Temperature reading
- Motion intensity
- Confidence score
- Detection window

#### Test 5.7: Alert Actions

**Expected buttons:**
- "Acknowledge" - Mark alert as seen
- "Resolve" - Mark alert as resolved

---

### Health Analysis Page Tests

Click: **Pages > Health Analysis** (in sidebar)

#### Test 5.8: Temperature Chart

**Expected:**
- 14-day temperature line chart
- **Fever spike visible on Day 3-4** (reaches 40°C)
- Baseline marked at 38.5°C
- Red zone above 39.5°C highlighted

#### Test 5.9: Activity Patterns

**Expected:**
- Activity level chart showing **reduced activity during fever**
- Behavioral state distribution (lying, standing, walking, etc.)

#### Test 5.10: Health Trends

**Expected:**
- Trend indicator: **"DECLINING"** or **"DETERIORATING"**
- Confidence: **85%+**
- Alert history timeline

---

## Step 6: Test End-to-End Upload

Now test uploading raw sensor data to verify the processing pipeline.

### Create Raw Sensor CSV (Without States)

```bash
# Extract first 100 samples, remove state column
head -101 data/simulation/DEMO_COW_001_sensor_data.csv | cut -d',' -f1-8 > test_raw_upload.csv
```

### Verify Raw Format

```bash
head -3 test_raw_upload.csv
```

**Expected (NO state column):**
```
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg
2025-11-14 13:40:00,38.5,-0.04,0.01,-0.88,-2.88,0.14,1.87
2025-11-14 13:41:00,38.5,-0.01,0.01,-0.87,-1.19,-1.14,1.86
```

### Upload in Dashboard

1. Go to **Home page**
2. In sidebar:
   - **Cow ID:** TEST_COW_001
   - **Baseline Temperature:** 38.5°C
3. **Upload:** test_raw_upload.csv

### Expected Processing Steps

You should see in sidebar:

```
✅ Loaded 100 sensor readings
⚙️ Layer 1: Classifying behavior...
✅ Layer 1: Behavior classified
⚙️ Layer 2: Temperature analysis complete
⚙️ Layer 3: Detecting health alerts...
✅ Layer 3: Detected 0 alerts
✅ Saved to database: 0 alerts
✅ Processing complete!

Summary:
• Sensor readings: 100
• Time range: 2025-11-14 13:40:00 to 2025-11-14 15:19:00
• Alerts detected: 0
```

### Verify Processed Files

```bash
ls -lh data/simulation/TEST_COW_001_*
```

**Expected:**
```
TEST_COW_001_sensor_data.csv    (with state column added)
TEST_COW_001_alerts.json        (empty array)
TEST_COW_001_metadata.json
```

### Check Behavior States Added

```bash
head -3 data/simulation/TEST_COW_001_sensor_data.csv
```

**Expected (state column ADDED):**
```
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg,cow_id,state
2025-11-14 13:40:00,38.5,-0.04,0.01,-0.88,-2.88,0.14,1.87,TEST_COW_001,lying
2025-11-14 13:41:00,38.5,-0.01,0.01,-0.87,-1.19,-1.14,1.86,TEST_COW_001,lying
```

**✅ END-TO-END UPLOAD PASSED** if states are classified automatically!

---

## Success Criteria

### ✅ All Tests Passed If:

1. **Data Generation:**
   - [x] Demo data files created in `data/simulation/`
   - [x] Sensor CSV has ~20,160 rows
   - [x] 2+ alerts detected

2. **Layer 1 - Behavior:**
   - [x] `state` column exists in sensor CSV
   - [x] States include: lying, standing, walking, ruminating, feeding

3. **Layer 2 - Temperature:**
   - [x] Temperature range: 38-40.5°C
   - [x] Fever period (day 3-4) shows ~40°C
   - [x] Trend report generated

4. **Layer 3 - Alerts:**
   - [x] Fever alerts detected (temp >39.5°C + motion <0.15)
   - [x] Inactivity alerts detected (stillness >4 hours)
   - [x] Alerts saved to both JSON and database

5. **Dashboard:**
   - [x] Home page shows health score
   - [x] Alerts displayed (critical + warnings)
   - [x] Live feed shows cow status
   - [x] Alerts page lists all alerts
   - [x] Health Analysis shows charts

6. **End-to-End Upload:**
   - [x] Raw CSV processed through all layers
   - [x] Behavior states classified automatically
   - [x] Alerts detected if present
   - [x] Results saved to database

---

## Troubleshooting

### Problem: No Alerts Generated

**Check:**
```bash
# Verify fever period has high temp + low motion
python -c "import pandas as pd; df = pd.read_csv('data/simulation/DEMO_COW_001_sensor_data.csv'); fever_df = df.iloc[2880:2900]; print(fever_df[['timestamp', 'temperature', 'fxa', 'mya', 'rza']].head())"
```

**Expected:** Temperature >39.5°C AND motion <0.05

**If not:** Regenerate demo data

### Problem: Dashboard Shows "No Data"

**Check files exist:**
```bash
ls -la data/simulation/DEMO_COW_001_*
```

**If missing:** Run `python generate_demo_data.py`

### Problem: States Not Classified in Upload

**Check error message in sidebar**

**Common issues:**
- Missing required columns (timestamp, temperature, fxa, mya, rza)
- Invalid timestamp format
- Module import errors

**Fix:** Restart dashboard (`Ctrl+C` then `streamlit run dashboard/app.py`)

### Problem: Alerts Not in Database

**Check database exists:**
```bash
ls -lh data/alert_state.db
```

**If missing:** Database will be created automatically on next upload

**Check table:**
```bash
python -c "import sqlite3; conn = sqlite3.connect('data/alert_state.db'); cursor = conn.cursor(); cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"'); print(cursor.fetchall()); conn.close()"
```

**Expected:** `[('alerts',)]`

---

## Clean Up After Testing

```bash
# Keep demo data, remove test files
rm -f test_raw_upload.csv
rm -f data/simulation/TEST_COW_001_*

# Full reset (start over)
rm -rf data/simulation/*
rm -f data/alert_state.db
python generate_demo_data.py
```

---

## Summary

This end-to-end test verifies:

✅ **Layer 1:** Behavioral state classification from accelerometer/gyroscope
✅ **Layer 2:** Temperature pattern analysis and trend detection
✅ **Layer 3:** Multi-type alert detection (fever, inactivity, heat stress)
✅ **Dashboard:** Real-time monitoring, alert management, trend visualization
✅ **Upload Pipeline:** Raw CSV → automated processing → database storage

**All systems operational!** 🎉
