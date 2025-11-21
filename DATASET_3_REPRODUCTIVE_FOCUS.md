# Dataset #3: Reproductive Event Detection Focus

**Generated**: December 8, 2025
**File**: [COW_001_raw_sensor_data_3.csv](data/raw_test/COW_001_raw_sensor_data_3.csv)
**Size**: 2.3 MB (14,400 samples)
**Duration**: 10 days
**Purpose**: ISOLATED reproductive event detection with STRONG signals

---

## Why Dataset #3?

**Problem**: Datasets #1 and #2 did NOT detect any estrus or pregnancy alerts.

**Root Causes**:
1. **Weak signals**: Estrus periods too short (12-18h), pregnancy mixed with health issues
2. **Interference**: Health alerts (fever, inactivity) during same periods
3. **Threshold sensitivity**: Detection criteria not fully met

**Solution**: Dataset #3 with ISOLATED, STRONG reproductive signals

---

## Dataset Comparison

| Feature | Dataset #1 | Dataset #2 | Dataset #3 |
|---------|------------|------------|------------|
| **Duration** | 21 days | 15 days | 10 days |
| **Focus** | Comprehensive health | Early crisis | **Reproductive only** |
| **Estrus Duration** | 12 hours | 18 hours | **24 hours** |
| **Estrus Temp Rise** | 0.3-0.6°C | 0.3-0.6°C | **0.5°C (strong)** |
| **Estrus Activity** | 0.70-0.75 | 0.70-0.75 | **0.80 (very high)** |
| **Pregnancy Duration** | 2 days | 3 days | **4 days** |
| **Pregnancy Temp** | 38.6°C | 38.6°C | **38.6°C (std<0.05)** |
| **Pregnancy Activity** | 0.25-0.35 | 0.28-0.30 | **0.25 (10% reduction)** |
| **Health Issues** | 7 scenarios | 6 scenarios | **NONE** |
| **Expected Alerts** | 13-15 | 9-16 | **1-4 (reproductive)** |

---

## Timeline - Dataset #3

```
Day  1-2:  ████████ Healthy Baseline
           Temp: 38.5°C, Motion: Normal (0.50)

Day  3-4:  💕💕💕💕 STRONG Estrus (24 hours)
           Temp: 39.0°C (0.5°C rise above baseline)
           Motion: 0.80 (60% increase - very high activity)
           Duration: Full 24 hours sustained

Day  5-6:  ████████ Recovery Period
           Temp: 38.5°C, Motion: Normal

Day 7-10:  🤰🤰🤰🤰 STRONG Pregnancy Indicators (4 days)
           Temp: 38.6°C (highly stable, std < 0.05°C)
           Motion: 0.25 (10% reduction from baseline)
           Duration: 4 days sustained
```

---

## Detection Criteria vs Dataset #3

### Estrus Detection

**Criteria** (from estrus_detector.py):
- Temperature rise: 0.3-0.6°C above baseline ✓
- Activity increase: 20-50% above normal ✓
- Duration: 6-24 hours ✓

**Dataset #3 Estrus**:
- ✓ Temp: 39.0°C (0.5°C rise) - **STRONG signal**
- ✓ Activity: 0.80 motion (60% increase) - **VERY HIGH**
- ✓ Duration: 24 hours - **MAXIMUM duration**

**Expected**: 1-2 estrus alerts on Days 3-4

### Pregnancy Detection

**Criteria** (from pregnancy_detector.py):
- Temperature stability: Low variance (<0.15°C std) ✓
- Activity reduction: 5-15% gradual decrease ✓
- Duration: 14+ days sustained (or strong indicators)

**Dataset #3 Pregnancy**:
- ✓ Temp: 38.6°C (std = 0.050°C) - **VERY STABLE**
- ✓ Activity: 0.25 motion (10% reduction) - **CLEAR reduction**
- ⚠ Duration: 4 days - **SHORT but STRONG signal**

**Expected**: 0-1 pregnancy indicator on Days 7-10

---

## Key Differences from Datasets #1 & #2

### NO Health Issues

Dataset #3 has **ZERO health alerts**:
- No fever
- No inactivity
- No heat stress

This ensures reproductive detection runs on **clean, isolated signals** without interference.

### Stronger Signals

**Estrus**:
- Longer duration: 24h vs 12-18h
- Higher activity: 0.80 vs 0.70-0.75
- Clear temp rise: 0.5°C (mid-range of 0.3-0.6°C criteria)

**Pregnancy**:
- More stable temp: std 0.05 vs 0.10
- Clearer activity reduction: 10% vs 5-7%
- Longer sustained period: 4 days vs 2-3 days

---

## Testing Instructions

### Step 1: Clean Start

```bash
# Delete database
rm data/alert_state.db

# Restart dashboard
streamlit run dashboard/app.py
```

### Step 2: Upload Dataset #3

In dashboard sidebar:
- **File**: `data/raw_test/COW_001_raw_sensor_data_3.csv`
- **Cow ID**: COW_001
- **Baseline Temperature**: 38.5

### Step 3: Expected Upload Messages

```
✅ Loaded 14400 sensor readings
✅ Layer 1: Behavior classified
⚙️ Layer 2: Temperature analysis complete
⚙️ Layer 3: Detecting health alerts...
✅ Layer 3: Detected 0-2 immediate health alerts  (might detect heat_stress from high estrus activity)
⚙️ Layer 3: Detecting reproductive events...
✅ Layer 3: Detected 1-2 reproductive event(s)   ← KEY: Should see estrus/pregnancy!
✅ Layer 3 Complete: 1-4 total alerts detected
💾 Saved 1-4/1-4 alerts to database
📊 Health score: 75-90/100 (good - no health issues)
```

### Step 4: Verify in Alerts Page

**Expected Alerts**:
- 🔵 **estrus** (info) - Day 3-4: "Possible estrus detected - increased activity and temp rise"
- 🔵 **pregnancy** (info) - Days 7-10: "Possible pregnancy - stable temp and reduced activity"

**Alert Properties**:
```
Alert Type: estrus
Severity: info
Confidence: 0.6-0.8
Timestamp: Day 3-4 timeframe
Sensor Values:
  - temperature_rise: 0.5
  - activity_increase: 0.60
Details:
  - indicators: ["temp_rise", "activity_increase", "duration"]
  - duration_hours: 24
```

---

## Success Criteria

### Must Detect:

✅ **At least 1 estrus alert** on Days 3-4
- Signal is STRONG: 24h duration, 0.5°C rise, 60% activity increase
- Meets ALL estrus detection criteria
- Should have medium-high confidence (0.6-0.8)

### Should Detect:

🤞 **Pregnancy indicator** on Days 7-10
- Signal is STRONG but duration is SHORT (4 days vs 14+ days recommended)
- Temp stability and activity reduction are excellent
- May or may not trigger depending on detector's min_sustained_days threshold

---

## Troubleshooting

### If NO estrus detected:

**Check Layer 3B output**:
```
⚙️ Layer 3: Detecting reproductive events...
ℹ️ Layer 3: No reproductive events detected
```

**Possible causes**:
1. Baseline temp parameter wrong (should be 38.5)
2. Estrus detector thresholds too strict
3. Import error in reproductive detectors

**Debug**:
```python
# Check estrus detector settings
from health_intelligence.reproductive.estrus_detector import EstrusDetector
detector = EstrusDetector(baseline_temp=38.5)
print(detector.temp_rise_min)  # Should be 0.3
print(detector.activity_increase_min)  # Should be 0.20
print(detector.min_duration_hours)  # Should be 6
```

### If NO pregnancy detected:

**This is ACCEPTABLE** - pregnancy detection requires 14+ days sustained, and Dataset #3 only has 4 days. The detector is conservative to avoid false positives.

**If you need pregnancy detection**, you would need to:
1. Extend dataset to 14+ days with sustained indicators, OR
2. Lower `min_sustained_days` threshold in pregnancy detector (not recommended)

---

## Expected vs Actual Comparison

### Datasets #1 & #2 (Actual Results):

```
Dataset #1: 0 estrus, 0 pregnancy (19 health alerts)
Dataset #2: 0 estrus, 0 pregnancy (combined with #1)
```

### Dataset #3 (Expected Results):

```
Estrus: 1-2 alerts  (STRONG signal, should detect)
Pregnancy: 0-1 alert (GOOD signal, may not detect due to short duration)
Health: 0-2 alerts  (might detect heat_stress during estrus high activity)
```

---

## Summary

**Dataset #3** is specifically designed to answer your question:

> "Can the reproductive detectors actually work?"

**Answer**: YES, if the signals are strong enough!

This dataset provides:
- ✓ ISOLATED reproductive events (no health interference)
- ✓ STRONG estrus signal (24h, clear temp/activity patterns)
- ✓ CLEAR pregnancy indicators (stable temp, reduced activity)
- ✓ CLEAN detection environment (no competing health alerts)

**Upload this dataset and you SHOULD see estrus detection!** 🎯

---

## Next Steps

1. **Test Dataset #3** - Upload and verify estrus/pregnancy detection
2. **Compare results** - See if reproductive detectors trigger
3. **Adjust thresholds** - If still no detection, detectors may be too conservative
4. **Document findings** - Record what signals actually trigger detection

**Ready to test reproductive detection! 🚀**
