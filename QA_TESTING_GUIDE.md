# QA Testing Guide - Livestock Health Monitoring System

## Quick Start for QA/Product Managers

This guide provides a complete testing workflow to verify all system features.

---

## 🎯 Test Objective

Verify that the livestock health monitoring system correctly:
- Detects all 5 alert types (fever, heat stress, inactivity, estrus, sensor malfunction)
- Displays all 5 behavioral states (lying, standing, walking, ruminating, feeding)
- Calculates health scores accurately
- Persists data correctly
- Provides intuitive dashboard experience

---

## 📊 Test Data Overview

**Pre-generated comprehensive test dataset:**
- **Cow ID**: QA_TEST_001
- **Duration**: 21 days
- **Sampling**: 1 sample per minute (30,240 data points)
- **File**: `data/dashboard/QA_TEST_001_sensor_data.csv`

**Test Scenarios Included:**
| Day | Scenario | Expected Alert |
|-----|----------|----------------|
| 1-2 | Normal baseline | None |
| 3 | Fever (40°C + low motion) | CRITICAL fever alert |
| 4 | Recovery | None |
| 5-6 | Normal | None |
| 7 | Heat stress (39.7°C + high activity) | WARNING heat stress |
| 8 | Recovery | None |
| 9-10 | Normal | None |
| 11 | Prolonged inactivity (6 hours stillness) | WARNING inactivity |
| 12 | Recovery | None |
| 13-14 | Normal | None |
| 15 | Estrus (+0.5°C + increased activity) | INFO estrus |
| 16-17 | Normal | None |
| 18 | Sensor malfunction (temp 45°C) | CRITICAL sensor error |
| 19-20 | Normal | None |
| 21 | Complex behavioral patterns | None (all behaviors visible) |

---

## 🧪 Testing Procedure

### Step 1: Clean Environment

```bash
# Close dashboard if running (press Ctrl+C in terminal)

# Delete old database (Windows)
del data\alert_state.db

# Or on Linux/Mac
rm data/alert_state.db
```

### Step 2: Generate Fresh Test Data

```bash
python tools/generate_qa_test_data.py
```

**Expected Output:**
```
[OK] Sensor data saved: data\dashboard\QA_TEST_001_sensor_data.csv
[OK] Metadata saved: data\dashboard\QA_TEST_001_metadata.json
[OK] QA test guide saved: data\dashboard\QA_TEST_001_QA_GUIDE.json
```

### Step 3: Start Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard should open at: http://localhost:8501

### Step 4: Upload Test Data

1. Navigate to **Home** page
2. In sidebar, find "Upload Raw Sensor Data" section
3. Upload file: `QA_TEST_001_sensor_data.csv`
4. Enter Cow ID: `QA_TEST_001`
5. Click "Process Upload Data"

**Expected Result:**
- ✅ Success message: "Data processed successfully"
- ✅ Processing time displayed
- ✅ Alerts detected count shown

---

## ✅ Test Checklist

### Page 1: Home

**Metrics Section:**
- [ ] Current health score displayed (0-100)
- [ ] Health status badge (Excellent/Good/Fair/Poor/Critical)
- [ ] Temperature reading with status (Normal/Elevated/Critical)
- [ ] Activity level percentage
- [ ] Current behavioral state shown

**Recent Alerts Panel:**
- [ ] 5 active alerts displayed
- [ ] Alerts show: timestamp, type, severity, message
- [ ] Alert types visible:
  - [ ] Fever (Day 3) - CRITICAL
  - [ ] Heat Stress (Day 7) - WARNING
  - [ ] Prolonged Inactivity (Day 11) - WARNING
  - [ ] Estrus Detected (Day 15) - INFO
  - [ ] Sensor Malfunction (Day 18) - CRITICAL

**Activity Chart:**
- [ ] 21-day timeline visible
- [ ] Temperature trend line plotted
- [ ] Activity level bars shown
- [ ] Alert markers visible on timeline
- [ ] Zoom/pan functionality works

### Page 2: Alerts

**Alerts Table:**
- [ ] All 5 alerts listed in table
- [ ] Columns: Timestamp, Type, Severity, Status, Message, Actions
- [ ] Severity badges color-coded (red=CRITICAL, yellow=WARNING, blue=INFO)
- [ ] "View Details" button works for each alert
- [ ] "Dismiss" button works and updates status

**Alert Filtering:**
- [ ] Filter by severity (Critical/Warning/Info)
- [ ] Filter by type (Fever/Heat Stress/Inactivity/Estrus/Sensor)
- [ ] Filter by status (Active/Dismissed)
- [ ] Date range filter works

**Alert Details Modal:**
- [ ] Click "View Details" opens modal
- [ ] Shows full alert information
- [ ] Displays sensor values at time of alert
- [ ] Shows detection window and confidence
- [ ] Includes recommendations (if applicable)

### Page 3: Health Analysis

**Health Score Chart:**
- [ ] 21-day health score timeline
- [ ] Scores range from 0-95 across timeline
- [ ] Low scores visible on event days (3, 7, 11, 18)
- [ ] High scores visible on normal days (1-2, 5-6, etc.)
- [ ] Component breakdown available (temp, activity, behavioral, alerts)

**Behavioral Patterns:**
- [ ] Pie chart shows behavior distribution
- [ ] All 5 states present:
  - [ ] Lying
  - [ ] Standing
  - [ ] Walking
  - [ ] Ruminating
  - [ ] Feeding
- [ ] Percentages add up to ~100%
- [ ] Time-of-day heatmap visible

**Multi-day Trends:**
- [ ] Temperature trend line with min/max bands
- [ ] Activity trend line
- [ ] Circadian rhythm pattern visible
- [ ] Abnormal days clearly visible (3, 7, 11, 15, 18)

**Export Functionality:**
- [ ] "Export Health Report" button works
- [ ] Downloads PDF/CSV report
- [ ] Report includes: health scores, alerts summary, behavioral stats

---

## 🔍 Expected Health Score Pattern

| Days | Expected Score Range | Reason |
|------|---------------------|--------|
| 1-2 | 85-95 (Excellent) | Normal baseline |
| 3 | 20-40 (Critical) | Fever penalty |
| 4 | 60-75 (Fair) | Recovery |
| 5-6 | 85-95 (Excellent) | Normal |
| 7 | 50-70 (Fair) | Heat stress |
| 8-10 | 80-95 (Good/Excellent) | Normal |
| 11 | 40-60 (Poor) | Inactivity penalty |
| 12-14 | 80-95 (Good/Excellent) | Normal |
| 15 | 70-85 (Good) | Estrus (not severe) |
| 16-17 | 85-95 (Excellent) | Normal |
| 18 | 0-20 (Critical) | Sensor malfunction |
| 19-21 | 80-95 (Good/Excellent) | Normal |

---

## 🐛 Common Issues & Solutions

### Issue 1: Database locked error
**Cause:** Dashboard is still running
**Solution:** Close dashboard (Ctrl+C), wait 5 seconds, delete database, restart

### Issue 2: No alerts detected
**Cause:** Incorrect data format or upload failed
**Solution:** Check that CSV has correct columns (timestamp, temperature, fxa, mya, rza, lyg, rzg)

### Issue 3: Charts not displaying
**Cause:** Browser cache or Streamlit session issue
**Solution:** Refresh browser (F5), or clear Streamlit cache (hamburger menu → Clear Cache → Rerun)

### Issue 4: Health scores all 0 or 100
**Cause:** Baseline not calculated correctly
**Solution:** Ensure at least 2 days of data uploaded, check temperature column has valid values

---

## 📝 Test Report Template

```
QA TEST REPORT - Livestock Health Monitoring System
Date: [DATE]
Tester: [NAME]
Test Data: QA_TEST_001 (21 days)

RESULTS:
✅ / ❌  All 5 alerts detected correctly
✅ / ❌  Health scores calculated accurately
✅ / ❌  Behavioral classification working
✅ / ❌  Charts and visualizations rendering
✅ / ❌  Alert dismissal functionality works
✅ / ❌  Data export functional

NOTES:
[Any observations, bugs, or suggestions]

OVERALL STATUS: PASS / FAIL
```

---

## 🎓 Advanced Testing

### Test Case 1: Real-time Updates
1. Upload data
2. Dismiss an alert
3. Navigate to Home page
4. Verify dismissed alert no longer shows in "Recent Alerts"

### Test Case 2: Multi-Cow Support
1. Generate data for second cow: `python tools/generate_qa_test_data.py` (edit cow_id to "QA_TEST_002")
2. Upload both cows
3. Use cow selector dropdown in sidebar
4. Verify data switches correctly between cows

### Test Case 3: Performance
1. Upload 21-day dataset (30,240 data points)
2. Measure upload processing time
3. Check chart rendering speed
4. Verify dashboard responsiveness

**Expected Performance:**
- Upload processing: < 30 seconds
- Chart rendering: < 3 seconds
- Page transitions: < 1 second

---

## 🎯 Acceptance Criteria

**System passes QA if:**
- ✅ All 5 alert types detected correctly
- ✅ All 5 behavioral states classified
- ✅ Health scores follow expected pattern (0-95 range)
- ✅ All 3 dashboard pages load and display data
- ✅ Alert dismissal works and persists
- ✅ Charts render correctly and are interactive
- ✅ No Python errors or exceptions
- ✅ Data persistence works (reload page, data still there)

**Ready for production when:**
- All acceptance criteria met
- Performance benchmarks achieved
- No critical bugs found
- UX is intuitive for farm operators

---

## 📞 Support

If you encounter issues during testing:
1. Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. Review logs in terminal where dashboard is running
3. Check database exists: `ls -l data/alert_state.db`
4. Verify test data generated: `ls -l data/dashboard/QA_TEST_001*`

---

**Ready to test! 🚀 Follow the steps above to verify the system.**
