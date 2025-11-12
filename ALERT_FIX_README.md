# Alert Detection Fix - Important Update

## Issues Fixed

### 1. Auto-Refresh Too Frequent (FIXED ✅)
**Problem**: Home page was refreshing every 30 seconds, making the app feel unstable.

**Solution**: Disabled auto-refresh by default. Set `AUTO_REFRESH_ENABLED = False` in [Home page](dashboard/pages/0_Home.py:37).

To re-enable: Change line 37 to `AUTO_REFRESH_ENABLED = True`

### 2. Alerts Not Being Detected (FIXED ✅)
**Problem**: Multiple issues prevented alert detection:
- Simulation was calling wrong method (`check_alerts()` instead of `detect_alerts()`)
- Wrong data type being passed (dict instead of DataFrame)
- **ROOT CAUSE**: Fever motion reduction was insufficient (multiplied by 0.7 instead of setting to low absolute value)

**Solution**: Fixed in [Simulation page](dashboard/pages/99_Simulation_Testing.py:261)
- Now calls `detect_alerts()` with DataFrame and sliding window
- **Critical fix (line 200-206)**: Sets motion to 0.02-0.05 during fever (below 0.15 threshold)
- Fever now properly detected with high temperature + low motion

### 3. Alerts Not Loading in Home Page (FIXED ✅)
**Problem**: Data loader only checked `logs/malfunction_alerts.json`, but simulation saves to `data/simulation/*_alerts.json`.

**Solution**: Updated [data_loader.py](dashboard/utils/data_loader.py:157) to check simulation directory first, with better error handling for empty files.

---

## Root Cause Explanation

### Why Fever Wasn't Detected

The fever alert requires TWO conditions to be met simultaneously:
1. Temperature > 39.5°C ✅ (This was working)
2. Motion intensity < 0.15 ❌ (This was NOT working)

**The Problem**:
```python
# OLD CODE (broken)
df.loc[fever_start:fever_end, activity_cols] *= 0.7  # Only 30% reduction
```

When normal motion is ~0.6-1.2, multiplying by 0.7 gives:
- 0.6 × 0.7 = 0.42 (still way above 0.15 threshold!)
- 1.2 × 0.7 = 0.84 (3x above threshold!)

**The Fix**:
```python
# NEW CODE (working)
df.loc[fever_start:fever_end, 'fxa'] = np.random.uniform(0.02, 0.05, N)  # Absolute low value
df.loc[fever_start:fever_end, 'mya'] = np.random.uniform(0.02, 0.05, N)
df.loc[fever_start:fever_end, 'rza'] = np.random.uniform(0.02, 0.05, N)
```

This sets motion to 0.02-0.05, which gives:
- Motion intensity = √(0.05² + 0.05² + 0.05²) ≈ 0.087 ✅ (below 0.15!)

---

## How to See Alerts Now

### Step 1: Clear Old Simulation Data
```bash
# Delete old empty alert file
rm i:/livestock/health/data/simulation/SIM_COW_001_alerts.json
```

Or just regenerate with a new cow ID.

### Step 2: Generate New Simulation with Fever
1. Open the app: `streamlit run dashboard/app.py`
2. Go to **Simulation Testing** page
3. Configure:
   - **Cow ID**: `FEVER_TEST_001` (or any name)
   - **Duration**: `7 days`
   - **Baseline temp**: `38.5°C`
   - ✅ Check **Fever**
     - Start on day: `3`
     - Duration: `2 days`
     - Fever temp: `40.0°C`
4. Click **Generate Simulation Data**

### Step 3: Check Alerts
After generation completes:

1. **In Simulation Page**:
   - Go to "Alerts" tab
   - You should see fever alerts detected during days 3-4
   - Alert type: `fever`
   - Severity: `critical` (if temp >40°C) or `warning` (if 39.5-40°C)

2. **In Home Page**:
   - Navigate to Home page
   - You should see:
     - Alert count in the top metrics
     - Critical alerts highlighted in red
     - Animals at risk count

---

## Understanding Alert Detection

### Fever Alert Triggers When:
- Temperature > 39.5°C **AND**
- Motion intensity < 0.15 **AND**
- Condition persists for ≥ 2 minutes (2 consecutive samples)

### Heat Stress Alert Triggers When:
- Temperature > 39.0°C **AND**
- Activity level > 60% **AND**
- Condition persists for ≥ 2 minutes

### Inactivity Alert Triggers When:
- Cow is motionless for > 4 hours (excluding normal lying/rest states)

### Sensor Malfunction Alert Triggers When:
- No data received
- Stuck sensor values
- Out-of-range readings

---

## Verification

To verify the fix worked:

```bash
# Check that alerts file is no longer empty
cat i:/livestock/health/data/simulation/FEVER_TEST_001_alerts.json

# Should show JSON array with alerts like:
# [
#   {
#     "timestamp": "2025-11-12T15:30:00",
#     "cow_id": "FEVER_TEST_001",
#     "alert_type": "fever",
#     "severity": "critical"
#   },
#   ...
# ]
```

---

## Technical Details

### What Was Wrong

**Before** (broken):
```python
# Wrong method name and wrong data type
detected = detector.check_alerts(  # ❌ Method doesn't exist
    cow_id=cow_id,
    sensor_reading=reading,  # ❌ Should be DataFrame, not dict
    current_behavioral_state=row['state']
)
```

**After** (fixed):
```python
# Correct method with DataFrame
detected = detector.detect_alerts(  # ✅ Correct method
    sensor_data=window_df,  # ✅ DataFrame with 10-minute window
    cow_id=cow_id,
    behavioral_state=df.iloc[idx]['state'],
    baseline_temp=baseline_temp
)
```

### Why Window is Needed

The alert detector needs a rolling window of recent data to:
1. Confirm the condition persists (not just a momentary spike)
2. Calculate confidence scores based on data quality
3. Check for the minimum required samples (2 minutes = 2 samples)

### Data Loader Fix

**Before** (only checked production logs):
```python
alert_log_file = 'logs/malfunction_alerts.json'  # ❌ Hardcoded
```

**After** (checks simulation first):
```python
# ✅ Check simulation directory
if sim_files:
    latest_sim = sorted(sim_files, key=lambda x: x.stat().st_mtime)[-1]
    alert_log_file = str(latest_sim)
```

---

## Quick Test Script

To test alert detection works:

```python
cd i:/livestock/health
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
from simulation import SimulationEngine
from simulation.health_conditions import FeverSimulator
from health_intelligence.alerts.immediate_detector import ImmediateAlertDetector
from datetime import datetime, timedelta

# Generate 1 hour of data with fever
engine = SimulationEngine(baseline_temperature=38.5)
df = engine.generate_continuous_data(duration_hours=1, start_datetime=datetime.now() - timedelta(hours=1))

# Inject fever in last 20 minutes
df.loc[40:59, 'temperature'] = 40.5
df.loc[40:59, ['fxa', 'mya', 'rza']] = 0.01

# Detect alerts
detector = ImmediateAlertDetector()
window = df.iloc[40:51]  # 11 samples
alerts = detector.detect_alerts(sensor_data=window, cow_id='TEST', baseline_temp=38.5)

print(f'Alerts: {len(alerts)}')
for a in alerts:
    print(f'  {a.alert_type} - {a.severity} - {a.confidence:.2f}')
"
```

Expected output:
```
Alerts: 1
  fever - critical - 1.00
```

---

## If Still No Alerts

1. **Check simulation generated fever data**:
   ```python
   import pandas as pd
   df = pd.read_csv('data/simulation/FEVER_TEST_001_sensor_data.csv')
   print(f"Fever readings (>39.5): {(df['temperature'] > 39.5).sum()}")
   ```

2. **Check alert file exists and is not empty**:
   ```bash
   ls -lh data/simulation/*_alerts.json
   cat data/simulation/FEVER_TEST_001_alerts.json
   ```

3. **Check logs for errors**:
   - Look for "Error" or "Warning" messages in terminal where Streamlit is running

4. **Verify config file**:
   ```bash
   cat config/alert_thresholds.yaml
   # Should have fever_alert section with temperature_threshold: 39.5
   ```

---

## Summary

✅ **Auto-refresh**: Disabled (page no longer refreshes every 30 seconds)
✅ **Alert detection**: Fixed (now uses correct method and data format)
✅ **Alert loading**: Fixed (checks simulation directory)
✅ **Empty file handling**: Fixed (returns empty array instead of error)

**Next step**: Regenerate simulation with fever condition to see alerts!
