# Simulation Testing Dashboard - User Guide

## Overview

The **Simulation Testing Dashboard** (`99_Simulation_Testing.py`) is a complete simulation environment built into your Streamlit app. It generates realistic cow data for all 3 layers so you can test your entire system end-to-end.

## Access

```bash
streamlit run dashboard/app.py
```

Then navigate to: **🧪 Simulation Testing** (page 99 in sidebar)

---

## Features

### 🎮 Full Control
- **Choose any cow ID** (simulates individual animals)
- **Set duration**: 1 to 90 days of data
- **Adjust baseline temperature** for each cow
- **Inject health conditions** at any time during simulation

### 🌡️ Health Conditions You Can Simulate
1. **Fever**
   - Start on any day
   - Choose duration (1-7 days)
   - Set fever temperature (39.5-41.5°C)
   - Automatic activity reduction

2. **Estrus (Breeding)**
   - Occurs on specific day
   - Temperature spike + activity increase
   - 8-hour duration

3. **Pregnancy**
   - Start from any day
   - Stable temperature pattern
   - Progressive activity reduction

4. **Heat Stress**
   - Occurs on specific day
   - High temperature + panting behavior
   - 1-24 hour duration

### 📊 5 Analysis Tabs

**Tab 1: Overview**
- Key health metrics
- Behavioral state distribution
- Active conditions summary

**Tab 2: Temperature Analysis**
- Temperature timeline chart
- Fever threshold detection
- Statistics (min/max/std)

**Tab 3: Activity Analysis**
- Daily activity levels
- Behavioral state timeline
- Movement patterns

**Tab 4: Alerts**
- Real-time alert detection
- Alert timeline
- Type and severity breakdown

**Tab 5: Health Trends**
- 7/14/30/90-day trend analysis
- Overall health trajectory
- Recommendations
- Significant changes

### 💾 Export Options
- **CSV**: Raw sensor data
- **JSON**: Health trend reports
- **CSV**: Alert logs

---

## How to Use

### Basic Testing (No Health Conditions)

1. Open Simulation Testing page
2. Enter Cow ID: `TEST_COW_001`
3. Set duration: `14 days`
4. Click **Generate Simulation Data**
5. Explore the 5 tabs

**Result**: You'll see 14 days of healthy cow behavior

---

### Test Fever Detection

1. Cow ID: `SICK_COW_001`
2. Duration: `14 days`
3. ✅ Check **Fever**
   - Start on day: `3`
   - Duration: `2 days`
   - Fever temp: `40.0°C`
4. Click **Generate**

**Result**:
- Days 1-2: Normal
- Days 3-4: High temperature, reduced activity
- Days 5-14: Recovery
- **Alerts tab**: Shows fever alerts detected
- **Trends tab**: Shows deteriorating trend during illness

---

### Test Estrus Detection

1. Cow ID: `BREEDING_COW_001`
2. Duration: `21 days`
3. ✅ Check **Estrus**
   - Estrus on day: `11` (mid-cycle)
4. Click **Generate**

**Result**:
- Day 11: Temperature spike + activity increase
- **Temperature tab**: Shows clear spike in chart
- Can repeat every 21 days to simulate natural cycle

---

### Test Pregnancy

1. Cow ID: `PREGNANT_COW_001`
2. Duration: `90 days`
3. ✅ Check **Pregnancy**
   - Start from day: `1`
4. Click **Generate**

**Result**:
- Very stable temperature throughout
- Gradual activity reduction over time
- **Trends tab**: Shows stable temperature trend

---

### Test Multiple Conditions (Realistic Scenario)

**Scenario**: Cow recovers from fever, then enters estrus

1. Cow ID: `REAL_SCENARIO_001`
2. Duration: `30 days`
3. ✅ **Fever**
   - Start: day `3`
   - Duration: `3 days`
   - Temp: `40.5°C`
4. ✅ **Estrus**
   - On day: `25`
5. Click **Generate**

**Result**: Complete health journey:
- Days 1-2: Normal
- Days 3-5: Sick (fever)
- Days 6-24: Recovery
- Day 25: Estrus event
- Days 26-30: Normal

---

## Use Cases

### 1. Before Real Data Arrives

**Test all your dashboard pages:**

```
1. Generate cow: DEMO_COW_001 (14 days, with fever)
2. Open "Health Overview" page → See health metrics
3. Open "Alerts Dashboard" page → See fever alerts
4. Open "Trend Analysis" page → See deteriorating trend
5. Open "Behavioral Patterns" page → See activity reduction
```

All pages will work with this simulated data!

### 2. Testing New Features

**Example**: You add a new alert type

```
1. Generate test data with heat stress
2. Verify your new alert appears in Alerts tab
3. Export alert CSV to verify format
4. Test acknowledgment workflow
```

### 3. Training & Demos

**Show stakeholders how system works:**

```
1. Generate healthy cow (7 days)
   → Show normal patterns

2. Generate sick cow (fever, 7 days)
   → Show how alerts trigger
   → Show trend analysis detects decline

3. Generate recovering cow (14 days, fever days 1-3)
   → Show improving trend
   → Demonstrate recovery tracking
```

### 4. ML Model Training

**Generate labeled training data:**

```
1. Generate cow with estrus
2. Export CSV
3. Use temperature spike as labeled estrus event
4. Train your estrus detection model
5. Repeat for different patterns
```

### 5. Algorithm Validation

**Test your analysis algorithms:**

```
1. Generate cow with known conditions
2. Run your algorithm
3. Compare to ground truth (you know when fever starts/ends)
4. Tune parameters
5. Re-generate and test again
```

---

## Integration with Rest of App

### Generated Data Works Everywhere

The simulation generates data in the **exact same format** as real cows:

**Sensor Data Columns:**
```
timestamp, temperature, fxa, mya, rza, sxg, lyg, dzg, state
```

**Alert Format:**
```
timestamp, cow_id, alert_type, severity
```

**Trend Report:**
- Same JSON structure as real analysis
- All metrics calculated identically
- Can be saved to database

### Save to Database (Optional)

To make simulated data available to other dashboard pages:

1. Generate simulation data
2. Export CSV
3. Load into your database table
4. Now all dashboard pages see this data

**Or** modify dashboard pages to check `st.session_state.simulation_data` first before querying database.

---

## Advanced Tips

### 1. Reproducible Testing

Set the same baseline and conditions to get consistent results for regression testing.

### 2. Stress Testing

Generate 90 days of data to test:
- Performance with large datasets
- Long-term trend accuracy
- Memory usage

### 3. Edge Cases

Test unusual scenarios:
- Fever that lasts 7 days (severe illness)
- Estrus + Heat stress same day
- Very low baseline temp (36.5°C)
- Very high baseline temp (39.3°C)

### 4. Batch Testing

Generate multiple cows:
```
Loop:
  - COW_001 (healthy)
  - COW_002 (fever)
  - COW_003 (pregnant)
  - COW_004 (estrus)
  - COW_005 (heat stress)
```

Now you have a full herd simulation!

---

## Troubleshooting

**Q: Simulation takes a long time**
→ Reduce duration or uncheck alert detection

**Q: No alerts detected even with fever**
→ Check that fever temp is >39.5°C and duration >2 minutes

**Q: Can't see data in other dashboard pages**
→ Simulation data is session-specific. Export CSV and reload, or modify other pages to use `st.session_state.simulation_data`

**Q: Want to add more cows**
→ Change Cow ID and generate again. Each cow stored separately in session state

**Q: Graphs are slow**
→ For 90-day simulations, charts have 130k+ points. Streamlit handles this but it takes a moment

---

## Technical Details

### Data Generation Process

1. **Baseline behavioral data** (Layer 1)
   - Uses `SimulationEngine`
   - Generates all 5 behavioral states
   - Creates realistic sensor signatures

2. **Health condition injection** (Layer 2)
   - Uses condition-specific simulators
   - Modifies temperature and activity
   - Preserves realistic patterns

3. **Alert detection** (Layer 3)
   - Runs `ImmediateAlertDetector`
   - Checks fever, heat stress, inactivity
   - Records all alerts

4. **Trend analysis** (Layer 3)
   - Uses `MultiDayHealthTrendTracker`
   - Calculates 7/14/30/90-day trends
   - Generates recommendations

### Performance

- **1 day**: ~2 seconds
- **7 days**: ~5 seconds
- **30 days**: ~15 seconds
- **90 days**: ~40 seconds

(Includes data generation + alert detection + trend analysis)

---

## Examples

### Example 1: Test Fever Detection Algorithm

```
Goal: Verify fever detection works correctly

1. Generate: 7 days, fever on day 3-4 (40°C)
2. Check Alerts tab → Should see fever alerts on days 3-4
3. Check Temperature tab → Should see red line above 39.5°C
4. Export alert CSV → Verify timestamps match fever period
```

### Example 2: Validate Trend Analysis

```
Goal: Ensure trend correctly identifies deterioration

1. Generate: 14 days, fever on days 5-7 (40.5°C)
2. Go to Health Trends tab
3. Check 7-day trend → Should show "deteriorating"
4. Check 14-day trend → Should show "deteriorating" or "stable"
5. Check recommendations → Should suggest veterinary review
```

### Example 3: Demo Estrus Detection

```
Goal: Show stakeholders how breeding detection works

1. Generate: 21 days, estrus on day 11
2. Show Temperature tab → Point out spike
3. Show Activity tab → Point out activity increase
4. Explain: "This pattern indicates breeding readiness"
5. Export data → "This gets logged for breeding management"
```

---

## Next Steps

After testing with simulation:

1. **Integrate with Real Data**
   - Use same data format
   - Real sensors → Database → Dashboard

2. **Keep Simulation Available**
   - Use for testing new features
   - Use for training
   - Use for demos

3. **Extend Simulation**
   - Add more health conditions
   - Simulate herd dynamics
   - Add seasonal patterns

---

## Summary

The Simulation Testing Dashboard provides:
- ✅ Complete end-to-end testing environment
- ✅ Realistic data for all 3 layers
- ✅ Interactive configuration
- ✅ Comprehensive visualization
- ✅ Export capabilities
- ✅ Integration with rest of app

**It's a complete virtual dairy farm in your dashboard!** 🐄📊

Use it to thoroughly test everything before real cows connect to your system.
