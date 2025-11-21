# Comprehensive Testing Guide - Artemis Livestock Health Monitoring System

**Version:** 2.0
**Date:** November 17, 2025
**Purpose:** Complete testing guide for all features and scenarios

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Pre-Testing Setup](#pre-testing-setup)
3. [Test Scenarios](#test-scenarios)
4. [Feature Testing Matrix](#feature-testing-matrix)
5. [Expected Results](#expected-results)
6. [Troubleshooting](#troubleshooting)

---

## System Overview

### Architecture

```
Artemis Health Monitoring System
├── Layer 1: Behavioral Classification (rule_based_classifier.py)
│   ├── Lying/Standing/Walking detection
│   ├── Rumination detection (FFT-based)
│   └── Transition states
├── Layer 2: Physiological Analysis
│   ├── Temperature analysis (circadian_rhythm.py)
│   └── Trend analysis
├── Layer 3: Health Intelligence
│   ├── Immediate Alert Detection (fever, heat stress, inactivity)
│   ├── Reproductive Health (estrus, pregnancy)
│   └── Long-term trend analysis
└── Dashboard Metrics
    ├── Health Score Calculation (simple_health_scorer.py)
    └── Alert State Management (SQLite database)
```

### Data Flow

```
Raw Sensor Data (CSV)
    ↓
Layer 1: Behavioral Classification → States (lying, standing, ruminating, etc.)
    ↓
Layer 2: Physiological Analysis → Temperature trends, activity patterns
    ↓
Layer 3: Alert Detection → Immediate alerts + Reproductive events
    ↓
Health Score Calculation → Total score (0-100) with components
    ↓
Database Storage → SQLite (alerts + health_scores tables)
    ↓
Dashboard Display → Real-time visualization
```

---

## Pre-Testing Setup

### 1. Environment Preparation

```bash
# Verify Python installation
python --version  # Should be 3.8+

# Install dependencies
pip install -r requirements.txt

# Verify Streamlit
streamlit --version
```

### 2. Clean Database State

```bash
# Backup existing data (if needed)
cp data/alert_state.db data/alert_state.db.backup

# Clear database (optional - for fresh start)
rm data/alert_state.db

# Clear dashboard data
rm -rf data/dashboard/*
```

### 3. Verify Test Data Generators

```bash
# List available generators
ls -la generate_raw_sensor_data*.py

# You should see:
# - generate_raw_sensor_data.py (basic)
# - generate_raw_sensor_data_2.py (general health)
# - generate_raw_sensor_data_3.py (reproductive focus)
# - generate_raw_sensor_data_4.py (rumination focus)
```

---

## Test Scenarios

### Scenario 1: Basic Health Monitoring (General Purpose)

**Purpose:** Test basic health monitoring with fever, heat stress, and inactivity detection.

**Dataset:** `generate_raw_sensor_data_2.py`

**Steps:**

1. **Generate Test Data**
```bash
python generate_raw_sensor_data_2.py
```

**Expected Output:**
```
Generated sensor data saved to: data/raw_test/COW_001_sensor_data.csv
Total samples: 10,080 (7 days, 1-minute intervals)
```

2. **Start Dashboard**
```bash
streamlit run dashboard/pages/0_Home.py
```

3. **Upload Data**
   - Open sidebar in Home page
   - Set Baseline Temperature: 38.5°C
   - Upload: `data/raw_test/COW_001_sensor_data.csv`
   - Wait for processing (10-30 seconds)

4. **Verify Processing**

Check for these messages in the sidebar:
```
✅ Layer 1: Behavior classified (with rumination detection)
⚙️ Layer 2: Temperature analysis complete
✅ Layer 3: Detected X immediate health alerts
✅ Health score: XX/100 (category) - Saved to database
```

5. **Inspect Results**

**Home Page:**
- Health Score: Should show calculated score (0-100)
- Active Alerts: Count of detected alerts
- Live Animal Feed: Shows current state

**Alerts Page (2_Alerts.py):**
- Active Alerts: List of all alerts with proper timestamps
- Alert types: fever, heat_stress, inactivity
- Severity: Critical and Warning levels
- Timestamps: Should show different times (not all the same)

**Health Analysis Page (3_Health_Analysis.py):**
- Contributing Factors:
  - 🌡️ Temperature Stability: 0-100%
  - 🏃 Activity Level: 0-100%
  - 🎯 Behavioral Consistency: 0-100%
  - 🐄 Rumination Quality: X% (should not be 0 if rumination detected)
  - ⚠️ Alert Status: 0%=many alerts, 100%=no alerts

6. **Expected Results**

| Metric | Expected Range | Pass Criteria |
|--------|---------------|---------------|
| Health Score | 40-90 | ✅ Score calculated |
| Active Alerts | 5-25 | ✅ Alerts detected |
| Rumination % | 0-10% | ✅ Shows actual % (may be low) |
| Alert Types | fever, heat_stress, inactivity | ✅ At least 2 types |

**Pass/Fail:**
- ✅ PASS if all alerts visible with different timestamps
- ✅ PASS if health score < 100 (reflects alerts)
- ✅ PASS if rumination shows actual percentage
- ❌ FAIL if alerts show "Future date" or all same time
- ❌ FAIL if rumination = 0% when data has rumination states

---

### Scenario 2: Reproductive Health Detection (Estrus & Pregnancy)

**Purpose:** Test estrus and pregnancy detection capabilities.

**Dataset:** `generate_raw_sensor_data_3.py`

**Steps:**

1. **Generate Reproductive-Focused Data**
```bash
python generate_raw_sensor_data_3.py
```

**Expected Output:**
```
Generated reproductive health data: data/raw_test/COW_001_sensor_data.csv
Duration: 21 days
Estrus events: 1 (Day 7-8)
Temperature patterns: Baseline 38.5°C, Estrus peak 39.3°C
Activity patterns: Baseline 0.30, Estrus peak 0.48
```

2. **Upload and Process**
   - Start dashboard: `streamlit run dashboard/pages/0_Home.py`
   - Upload the generated CSV
   - Baseline Temperature: 38.5°C

3. **Check Reproductive Detection**

Sidebar should show:
```
⚙️ Layer 3: Detecting reproductive events...
✅ Layer 3: Detected X reproductive event(s)
```

4. **Verify Reproductive Alerts**

Navigate to **Alerts Page (2_Alerts.py)**

Look for:
- Alert type: `estrus` or `pregnancy`
- Severity: `info`
- Details: Temperature rise, activity increase, indicators

**Database Verification:**
```bash
# Use recalculate_health_score.py to check
python recalculate_health_score.py
```

Look for alert types in output.

5. **Expected Results**

| Metric | Expected | Pass Criteria |
|--------|----------|---------------|
| Estrus Alerts | 1-2 | ✅ At least 1 estrus event |
| Pregnancy Alerts | 0-1 | ⚠️ May be 0 (requires 21+ day observation) |
| Alert Details | Temp rise, activity increase | ✅ Detection details present |
| Confidence | 0.6-0.8 | ✅ Medium to high confidence |

**Pass/Fail:**
- ✅ PASS if estrus event detected
- ⚠️ PARTIAL if no pregnancy (may need longer observation)
- ❌ FAIL if no reproductive alerts at all

---

### Scenario 3: Rumination Detection

**Purpose:** Test rumination detection and display.

**Dataset:** `generate_raw_sensor_data_4.py`

**Steps:**

1. **Generate Rumination-Focused Data**
```bash
python generate_raw_sensor_data_4.py
```

**Expected Output:**
```
Generated rumination-focused data: data/raw_test/COW_001_sensor_data.csv
Duration: 7 days
Rumination periods: Enhanced patterns with 15-20% rumination time
States: ruminating_lying, ruminating_standing
```

2. **Upload and Process**
   - Upload generated CSV through Home page
   - Baseline Temperature: 38.5°C

3. **Verify Rumination Detection**

**During Processing:**
```
✅ Layer 1: Behavior classified (with rumination detection)
```

**After Processing:**

Go to **Health Analysis Page (3_Health_Analysis.py)**

Check **Contributing Factors Breakdown**:
- 🐄 Rumination Quality: Should show 10-20%

**Database Check:**
```bash
# Create quick check script
python -c "
import pandas as pd
df = pd.read_csv('data/dashboard/COW_001_sensor_data.csv')
ruminat = df['state'].str.contains('ruminat', case=False, na=False).sum()
print(f'Rumination samples: {ruminat} ({ruminat/len(df)*100:.1f}%)')
"
```

4. **Expected Results**

| Metric | Expected | Pass Criteria |
|--------|----------|---------------|
| Rumination % | 10-20% | ✅ Within healthy range |
| Rumination States | ruminating_lying, ruminating_standing | ✅ Both states detected |
| Health Score | Higher (70-95) | ✅ Better score due to good rumination |
| Behavioral Component | 0.6-0.9 | ✅ Good behavioral patterns |

**Pass/Fail:**
- ✅ PASS if rumination > 10%
- ✅ PASS if "Rumination Quality" displays actual %
- ❌ FAIL if rumination = 0%
- ❌ FAIL if rumination not reflected in behavioral score

---

### Scenario 4: Alert Management Workflow

**Purpose:** Test alert acknowledgment, resolution, and state management.

**Dataset:** Any dataset with alerts (use Scenario 1)

**Steps:**

1. **Generate Alerts** (if not already done)
```bash
python generate_raw_sensor_data_2.py
# Upload through dashboard
```

2. **Navigate to Alerts Page**
```
Dashboard → Alerts (2_Alerts.py)
```

3. **Test Alert Actions**

**For each active alert:**

a. **Expand Alert Card**
   - Click on alert to expand
   - Verify details visible:
     - Alert ID
     - Cow ID
     - Type, Severity, Confidence
     - Detected timestamp
     - Sensor values

b. **Acknowledge Alert**
   - Click "✓ Acknowledge" button
   - Verify success message
   - Alert moves to "Acknowledged Alerts" panel
   - Status changes from "active" to "acknowledged"

c. **Resolve Alert**
   - Click "✓ Resolve" button
   - Verify success message
   - Alert removed from active list
   - Status changes to "resolved"

d. **Add Notes**
   - Enter notes in text area
   - Click "Save Notes"
   - Verify notes saved

e. **Mark False Positive**
   - Click "✗ False Positive"
   - Verify alert marked appropriately

4. **Verify State Transitions**

Check database metrics:
```
🔴 Active: X
👁️ Acknowledged: Y
✅ Resolved: Z
📊 Total: X + Y + Z
```

5. **Expected Results**

| Action | Expected Behavior | Pass Criteria |
|--------|------------------|---------------|
| Acknowledge | Moves to acknowledged panel | ✅ State updated |
| Resolve | Removes from active list | ✅ Resolved count increases |
| Add Notes | Notes saved to database | ✅ Notes persist |
| False Positive | Alert flagged | ✅ Status updated |

**Pass/Fail:**
- ✅ PASS if all state transitions work
- ✅ PASS if metrics update correctly
- ❌ FAIL if buttons don't work
- ❌ FAIL if states don't persist

---

### Scenario 5: Health Score Accuracy

**Purpose:** Verify health score calculation reflects all components correctly.

**Steps:**

1. **Generate Test Data with Known Patterns**

Create a CSV with specific patterns:
- High temperature (40°C) for fever
- Low activity for inactivity
- Excessive lying (90% of time)
- No rumination

2. **Upload and Calculate**
   - Upload through Home page
   - Note the health score

3. **Verify Components**

Use the recalculation script:
```bash
python recalculate_health_score.py
```

**Expected Output:**
```
Component Scores (normalized 0-1):
  Temperature: X.XXX (low if fever)
  Activity: X.XXX (low if inactive)
  Behavioral: X.XXX (low if excessive lying)
  Alert: X.XXX (low if many alerts)
```

4. **Component Validation**

| Condition | Expected Component Score | Penalty |
|-----------|------------------------|---------|
| Fever (temp > 39.5°C) | Temperature < 0.5 | -15 points |
| Low activity | Activity < 0.5 | -10 points |
| Excessive lying (>70%) | Behavioral < 0.4 | -10 points |
| No rumination | Behavioral -5 points | -5 points |
| 10 critical alerts | Alert = 0.0 | -25 points (max) |

5. **Total Score Formula**

```
Total = (Temperature × 30%) + (Activity × 25%) + (Behavioral × 25%) + (Alert × 20%)
All scores are 0-1 normalized, result is 0-100
```

**Example Calculation:**
```
Temperature: 0.5 → 0.5 × 30 = 15 points
Activity: 0.8 → 0.8 × 25 = 20 points
Behavioral: 0.2 → 0.2 × 25 = 5 points
Alert: 0.0 → 0.0 × 20 = 0 points
Total: 15 + 20 + 5 + 0 = 40/100 (Moderate)
```

6. **Expected Results**

| Test Case | Expected Score Range | Category |
|-----------|---------------------|----------|
| Perfect health | 90-100 | Excellent |
| Minor issues | 70-89 | Good |
| Multiple alerts | 50-69 | Moderate |
| Severe issues | 0-49 | Poor |

**Pass/Fail:**
- ✅ PASS if score matches calculated components
- ✅ PASS if category matches score range
- ❌ FAIL if score doesn't reflect alerts
- ❌ FAIL if components don't sum correctly

---

### Scenario 6: Multi-Day Trend Analysis

**Purpose:** Test historical trend tracking and baseline comparisons.

**Steps:**

1. **Generate Multi-Day Data**
```bash
# Generate 21 days of data
python generate_raw_sensor_data_2.py  # Modify to extend duration
```

2. **Upload Multiple Times**

Upload data in batches to simulate multiple health score entries:
- Day 1-7: Upload first week
- Day 8-14: Upload second week
- Day 15-21: Upload third week

3. **View Health Analysis**

Navigate to **Health Analysis Page (3_Health_Analysis.py)**

Check:
- **Health Score History Chart**: Line graph showing trends
- **Average Score**: Mean across all records
- **Highest/Lowest Score**: Range of scores
- **Data Points**: Number of health score entries

4. **Verify Baseline Calculation**

The page calculates baseline as:
```
Baseline = Average of last 30 days (or all available data if < 30 days)
```

5. **Expected Results**

| Metric | Expected | Pass Criteria |
|--------|----------|---------------|
| Historical Data | Multiple data points | ✅ Chart shows trend line |
| Baseline Score | Average of scores | ✅ Baseline calculated |
| Trend Direction | Improving/Stable/Deteriorating | ✅ Trend indicator shown |
| Date Range | X days | ✅ Matches uploaded data |

**Pass/Fail:**
- ✅ PASS if multiple health scores tracked
- ✅ PASS if trends visualized correctly
- ❌ FAIL if only one data point shown
- ❌ FAIL if chart doesn't render

---

### Scenario 7: Edge Cases and Error Handling

**Purpose:** Test system robustness with unusual inputs.

**Test Cases:**

#### 7.1 Empty/Invalid Data
```bash
# Create empty CSV
echo "timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg" > empty.csv

# Upload through dashboard
```

**Expected:** Error message, no crash

#### 7.2 Missing Columns
```bash
# Create CSV without required columns
echo "timestamp,temperature" > incomplete.csv
echo "2025-11-17 10:00:00,38.5" >> incomplete.csv

# Upload through dashboard
```

**Expected:** "Missing required columns: fxa, mya, rza"

#### 7.3 Extreme Values
```bash
# Create data with extreme temperatures
# Temp = 45°C (way above normal)
# Temp = 30°C (way below normal)
```

**Expected:**
- Critical fever alerts
- Low health score
- System handles extreme values

#### 7.4 Future Timestamps
```bash
# Data with timestamps in the future
# (Already tested - should show absolute dates)
```

**Expected:** Displays "Nov XX, HH:MM" format

#### 7.5 Very Large Dataset
```bash
# Generate 30 days at 1-second intervals
# ~2.6 million samples
```

**Expected:**
- Processing may take longer
- System should handle gracefully
- May show memory/performance warnings

**Pass/Fail:**
- ✅ PASS if errors handled gracefully
- ✅ PASS if no crashes
- ❌ FAIL if system crashes on invalid input

---

## Feature Testing Matrix

### Complete Feature Checklist

| Feature | Test Scenario | Status | Notes |
|---------|--------------|--------|-------|
| **Layer 1: Behavioral Classification** ||||
| Lying detection | Scenario 1 | ✅ | |
| Standing detection | Scenario 1 | ✅ | |
| Walking detection | Scenario 1 | ✅ | |
| Rumination detection | Scenario 3 | ✅ | Requires FFT |
| Transition states | Scenario 1 | ✅ | |
| **Layer 2: Physiological Analysis** ||||
| Temperature trends | Scenario 1 | ✅ | |
| Circadian rhythm | Scenario 6 | ✅ | |
| Activity patterns | Scenario 1 | ✅ | |
| **Layer 3: Immediate Alerts** ||||
| Fever detection | Scenario 1 | ✅ | Temp > 39.5°C |
| Heat stress detection | Scenario 1 | ✅ | High temp + high activity |
| Inactivity detection | Scenario 1 | ✅ | Low movement |
| **Layer 3: Reproductive Health** ||||
| Estrus detection | Scenario 2 | ✅ | Requires specific patterns |
| Pregnancy detection | Scenario 2 | ⚠️ | Requires 21+ days |
| **Dashboard Metrics** ||||
| Health score calculation | Scenario 5 | ✅ | 0-100 scale |
| Component scoring | Scenario 5 | ✅ | 4 components |
| Alert impact on score | Scenario 5 | ✅ | Penalty applied |
| **Alert Management** ||||
| Alert creation | Scenario 1 | ✅ | Auto-created |
| Alert acknowledgment | Scenario 4 | ✅ | State management |
| Alert resolution | Scenario 4 | ✅ | State management |
| False positive marking | Scenario 4 | ✅ | State management |
| **Data Visualization** ||||
| Health score gauge | All scenarios | ✅ | Visual gauge |
| Alert timeline | Scenario 6 | ✅ | Historical view |
| Contributing factors | Scenario 1 | ✅ | Component breakdown |
| Trend indicators | Scenario 6 | ✅ | Improving/declining |
| **Database & Storage** ||||
| SQLite persistence | All scenarios | ✅ | Data saved |
| Health score history | Scenario 6 | ✅ | Multiple entries |
| Alert state tracking | Scenario 4 | ✅ | Status changes |

---

## Expected Results Summary

### Healthy Cow (Baseline)

```
Health Score: 85-95/100 (Good to Excellent)
Components:
  - Temperature: 0.9-1.0 (38.0-39.0°C)
  - Activity: 0.8-1.0 (Normal movement)
  - Behavioral: 0.7-0.9 (Balanced states, 10-20% rumination)
  - Alerts: 0.8-1.0 (0-2 minor alerts)

Active Alerts: 0-3
States: 40-60% lying, 10-20% standing, 5-10% walking, 10-20% ruminating
```

### Sick Cow (Multiple Issues)

```
Health Score: 30-60/100 (Poor to Moderate)
Components:
  - Temperature: 0.0-0.5 (Fever present)
  - Activity: 0.2-0.6 (Reduced activity)
  - Behavioral: 0.1-0.4 (Excessive lying, no rumination)
  - Alerts: 0.0-0.2 (10+ active alerts)

Active Alerts: 10-25
Alert Types: fever (critical), inactivity (warning), heat_stress (warning)
States: 80-90% lying, 5-10% standing, 0-5% walking, 0% ruminating
```

### Cow in Estrus

```
Health Score: 70-85/100 (Good)
Components:
  - Temperature: 0.6-0.8 (Slight elevation)
  - Activity: 0.9-1.0 (Increased activity)
  - Behavioral: 0.7-0.9 (Active states)
  - Alerts: 0.8-1.0 (Estrus alert is "info" severity)

Active Alerts: 1-3
Special: Estrus alert with temp rise + activity increase indicators
States: More standing/walking, normal rumination
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Rumination shows 0%

**Symptoms:** Health Analysis page shows "Rumination Quality: 0.0%"

**Causes:**
1. Test data doesn't have rumination patterns
2. Rumination detection disabled
3. Display bug (fixed in latest version)

**Solutions:**
```bash
# Solution 1: Use rumination-focused dataset
python generate_raw_sensor_data_4.py

# Solution 2: Verify rumination in data
python -c "
import pandas as pd
df = pd.read_csv('data/dashboard/COW_001_sensor_data.csv')
print(df['state'].value_counts())
"

# Solution 3: Check if fixed applied
# Should see ruminating_lying and ruminating_standing in state counts
```

#### Issue 2: All alerts show same timestamp

**Symptoms:** Alerts display "2 hours ago" for all

**Cause:** Using `created_at` instead of `timestamp` field

**Solution:** Fixed in [notification_panel.py:110](dashboard/components/notification_panel.py#L110)
- Verify fix applied
- Restart Streamlit dashboard

#### Issue 3: Health score doesn't reflect alerts

**Symptoms:** Score is 91 even with 19 alerts

**Causes:**
1. Health score calculated before alerts added
2. Stale health score in database

**Solution:**
```bash
# Recalculate health score
python recalculate_health_score.py

# Or re-upload data through dashboard
```

#### Issue 4: No reproductive alerts detected

**Symptoms:** No estrus or pregnancy alerts

**Causes:**
1. Data doesn't have reproductive patterns
2. Patterns not strong enough for detection
3. Detection code not running

**Solutions:**
```bash
# Use reproductive-focused dataset
python generate_raw_sensor_data_3.py

# Check detection logs in sidebar during upload
# Should see: "✅ Layer 3: Detected X reproductive event(s)"
```

#### Issue 5: Dashboard won't start

**Symptoms:** Error when running `streamlit run`

**Solutions:**
```bash
# Check Streamlit installation
pip install --upgrade streamlit

# Check Python path
python --version

# Check for port conflicts
streamlit run dashboard/pages/0_Home.py --server.port 8502
```

#### Issue 6: Database locked error

**Symptoms:** "database is locked" error

**Solution:**
```bash
# Close all dashboard instances
# Wait 5 seconds
# Restart dashboard
streamlit run dashboard/pages/0_Home.py
```

---

## Performance Benchmarks

### Expected Processing Times

| Data Size | Classification | Alert Detection | Total Time |
|-----------|---------------|----------------|------------|
| 1 day (1,440 samples) | 2-5 sec | 1-2 sec | 5-10 sec |
| 7 days (10,080 samples) | 10-20 sec | 5-10 sec | 20-40 sec |
| 21 days (30,240 samples) | 30-60 sec | 15-30 sec | 60-120 sec |

*Times are approximate and depend on system performance*

### Memory Usage

| Data Size | Expected RAM Usage |
|-----------|-------------------|
| 1 day | 50-100 MB |
| 7 days | 200-400 MB |
| 21 days | 500-800 MB |

---

## Success Criteria

### Overall System Pass

System is considered fully functional if:

1. ✅ All 7 test scenarios pass
2. ✅ All features in matrix are working
3. ✅ No critical bugs in error handling
4. ✅ Health scores accurately reflect components
5. ✅ Alerts display with correct timestamps
6. ✅ Rumination detected when present in data
7. ✅ Alert state management works
8. ✅ Database persistence functional

### Minimum Viable Product (MVP)

System meets MVP criteria if:

1. ✅ Scenarios 1, 4, 5 pass (basic health monitoring)
2. ✅ Immediate alerts working (fever, heat stress, inactivity)
3. ✅ Health score calculation functional
4. ✅ Dashboard displays data correctly

---

## Test Report Template

Use this template to document test results:

```markdown
# Test Report - [Date]

## Environment
- Python Version: X.X.X
- Streamlit Version: X.X.X
- OS: [Windows/Linux/Mac]

## Test Results

### Scenario 1: Basic Health Monitoring
- Status: ✅ PASS / ❌ FAIL
- Health Score: XX/100
- Alerts Detected: XX
- Notes: [Any observations]

### Scenario 2: Reproductive Health
- Status: ✅ PASS / ⚠️ PARTIAL / ❌ FAIL
- Estrus Alerts: XX
- Pregnancy Alerts: XX
- Notes: [Any observations]

[Continue for all scenarios...]

## Issues Found
1. [Issue description]
2. [Issue description]

## Overall Assessment
- MVP Criteria: ✅ MET / ❌ NOT MET
- Full System Pass: ✅ PASS / ❌ FAIL
- Recommendation: [APPROVE FOR PRODUCTION / NEEDS FIXES]
```

---

## Next Steps After Testing

1. **If all tests pass:**
   - Review PRODUCTION_GUIDE.md for deployment
   - Set up monitoring and alerts
   - Train users on dashboard

2. **If tests fail:**
   - Document failures in test report
   - Review TROUBLESHOOTING section
   - Check recent fixes in FINAL_DASHBOARD_FIXES.md
   - Re-run specific failing scenarios

3. **For production deployment:**
   - Use real sensor data instead of simulations
   - Configure baseline temperatures for each cow
   - Set up automated data ingestion
   - Configure alert notifications (email, SMS, etc.)

---

## Contact & Support

- **Documentation:** Check remaining .md files in project root
- **Issues:** Document in test report
- **Updates:** Review git commit history for latest changes

---

**End of Comprehensive Testing Guide**
