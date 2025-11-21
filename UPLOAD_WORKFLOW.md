# Upload Workflow - End-to-End Processing

## Overview

The dashboard now accepts **ONLY raw sensor CSV files** and automatically processes them through all 3 intelligence layers. No pre-processed alerts or metadata files needed!

---

## What You Need

### Required: Raw Sensor CSV

**Format:**
```csv
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg
2025-11-13 16:10:00,38.5,-0.04,0.01,-0.88,-2.88,0.14,1.87
2025-11-13 16:11:00,38.6,-0.01,0.01,-0.87,-1.19,-1.14,1.86
...
```

**Required columns:**
- `timestamp` - ISO format datetime (e.g., "2025-11-13 16:10:00")
- `temperature` - Body temperature in °C
- `fxa` - Forward/aft acceleration (g-force)
- `mya` - Medial/lateral acceleration (g-force)
- `rza` - Vertical acceleration (g-force)
- `sxg` - X-axis gyroscope (degrees/sec) - optional
- `lyg` - Y-axis gyroscope (degrees/sec) - optional
- `dzg` - Z-axis gyroscope (degrees/sec) - optional

**Optional columns:**
- `state` - Behavioral state (if already classified)
- `cow_id` - Animal identifier (can be entered in UI)

---

## How to Upload

### Step 1: Open Dashboard

```bash
streamlit run dashboard/app.py
```

### Step 2: Configure Parameters

In the sidebar:
- **Cow ID**: Enter animal identifier (default: COW_001)
- **Baseline Temperature**: Enter normal body temp (default: 38.5°C)

### Step 3: Upload Raw Sensor CSV

Click "📊 Raw Sensor Data (CSV)" and select your file.

### Step 4: Automatic Processing

The system will:

1. **Validate** sensor data format
2. **Layer 1**: Classify behavioral states (lying, standing, walking, ruminating, feeding)
3. **Layer 2**: Analyze temperature patterns
4. **Layer 3**: Detect health alerts (fever, heat stress, inactivity)
5. **Save** results to both JSON and database

### Step 5: View Results

Click "🔄 Refresh to View Data" to see:
- Health score
- Detected alerts
- Behavioral patterns
- Temperature trends

---

## Processing Details

### Layer 1: Behavior Classification

Uses accelerometer and gyroscope data to classify:
- **Lying**: Low vertical acceleration, minimal movement
- **Standing**: High vertical acceleration, low motion
- **Walking**: Rhythmic motion patterns
- **Ruminating**: Specific head movement patterns
- **Feeding**: Forward head motion with chewing

### Layer 2: Temperature Analysis

Analyzes:
- Temperature deviations from baseline
- Circadian rhythm patterns
- Temperature-activity correlation
- Multi-day trends

### Layer 3: Alert Detection

Detects 4 types of alerts:

1. **Fever Alert**
   - Temperature > 39.5°C
   - Motion intensity < 0.15
   - Duration ≥ 2 minutes
   - Severity: Critical if temp > 40.0°C

2. **Heat Stress Alert**
   - Temperature > 39.0°C
   - Activity level > 0.60
   - Duration ≥ 2 minutes
   - Severity: Critical if temp > 39.8°C

3. **Inactivity Alert**
   - All motion axes < 0.05
   - Duration ≥ 4 hours
   - Excludes normal rest periods
   - Severity: Critical if > 8 hours

4. **Sensor Malfunction Alert**
   - No data received
   - Out-of-range values
   - Stuck sensor values

---

## Output Files

After processing, the following files are saved:

### 1. Sensor Data with States
**Location:** `data/simulation/{COW_ID}_sensor_data.csv`

Includes original sensor data PLUS classified behavioral states.

### 2. Alerts JSON
**Location:** `data/simulation/{COW_ID}_alerts.json`

```json
[
  {
    "timestamp": "2025-11-13 16:15:00",
    "cow_id": "COW_001",
    "alert_type": "fever",
    "severity": "critical",
    "confidence": 0.92,
    "sensor_values": {
      "temperature": 40.2,
      "motion_intensity": 0.04
    },
    "details": {
      "max_temperature": 40.2,
      "avg_motion": 0.05,
      "threshold_exceeded_by": 0.7
    }
  }
]
```

### 3. Metadata JSON
**Location:** `data/simulation/{COW_ID}_metadata.json`

```json
{
  "cow_id": "COW_001",
  "baseline_temp": 38.5,
  "total_samples": 20160,
  "start_time": "2025-11-13 16:10:00",
  "end_time": "2025-11-27 16:10:00",
  "num_alerts": 2,
  "processed_at": "2025-11-14 10:30:00",
  "processing": "end-to-end (Layer 1 + Layer 2 + Layer 3)"
}
```

### 4. Database Storage
**Location:** `data/alert_state.db` (SQLite)

Alerts are also saved to database for management features:
- Acknowledge alerts
- Resolve alerts
- Track status changes
- Alert history

---

## Example Workflow

### Generate Test Data

```bash
# Option 1: Use pre-generated demo
# (Already includes 14 days with fever scenario)
streamlit run dashboard/app.py
# Upload: data/demo/DEMO_COW_001_sensor_data.csv

# Option 2: Generate custom scenario
streamlit run simulation_app.py
# Configure parameters, generate, download CSV
```

### Test Raw Upload

```bash
# Create test file (first 20 samples without state column)
head -20 data/demo/DEMO_COW_001_sensor_data.csv | cut -d',' -f1-8 > data/test_raw_sensor.csv

# Upload test_raw_sensor.csv in dashboard
# System will classify behavior and detect alerts automatically
```

---

## Troubleshooting

### Error: Missing Required Columns

**Problem:** CSV missing required columns

**Solution:** Ensure CSV has: timestamp, temperature, fxa, mya, rza

```bash
# Check columns
head -1 your_sensor_data.csv
```

### Error: Processing Failed

**Problem:** Data format issues

**Solution:** Check timestamp format
```python
# Correct format
2025-11-13 16:10:00

# Incorrect
Nov 13, 2025 4:10 PM
```

### No Alerts Detected

**Problem:** Data might be normal (no health issues)

**Solution:** Check data:
- Temperature should have fever (>39.5°C) for fever alert
- Motion should be low (<0.15) during fever
- Duration should be ≥2 minutes

### Database Save Failed

**Problem:** Alert database not accessible

**Solution:**
- Alerts still saved to JSON (works fine)
- Database save is optional for upload workflow
- Alerts page might not show alerts (use Home page instead)

---

## Production Workflow

### Real Sensor Integration

1. **Sensor Data Collection**
   - Neck-mounted sensor on cattle
   - Samples every minute: temp, accel (3-axis), gyro (3-axis)
   - Stream to central server

2. **Real-time Processing**
   - Process data as it arrives
   - Run through all 3 layers continuously
   - Generate alerts within 1-2 minutes

3. **Alert Management**
   - Display in dashboard
   - Send notifications (email, SMS)
   - Log to database
   - Track acknowledgment/resolution

---

## API Integration (Future)

For programmatic uploads:

```python
import requests
import pandas as pd

# Prepare data
df = pd.read_csv('sensor_data.csv')

# Upload to dashboard API
response = requests.post(
    'http://dashboard-api/upload',
    json={
        'cow_id': 'COW_001',
        'baseline_temp': 38.5,
        'sensor_data': df.to_dict('records')
    }
)

# Get results
alerts = response.json()['alerts']
```

---

## Summary

✅ **Upload ONLY raw sensor CSV**
✅ **No pre-processing needed**
✅ **Automatic end-to-end processing**
✅ **Alerts detected in 1-2 minutes**
✅ **Results saved to JSON + database**
✅ **View in dashboard immediately**

This is the true end-to-end test the system was designed for!
