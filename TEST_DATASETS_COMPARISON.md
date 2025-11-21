# Test Datasets Comparison

**Generated**: December 8, 2025

You now have TWO test datasets for comprehensive validation:

---

## Dataset #1: 21-Day Comprehensive Test

**File**: [data/raw_test/COW_001_raw_sensor_data.csv](data/raw_test/COW_001_raw_sensor_data.csv)

**Size**: 4.8 MB (30,240 samples)
**Duration**: 21 days
**Purpose**: Comprehensive testing with gradual condition progression

### Scenario Timeline

| Day | Scenario | Temperature | Motion | Severity | Notes |
|-----|----------|-------------|--------|----------|-------|
| 1-2 | Healthy baseline | 38.5°C | Normal | - | Baseline period |
| 3-4 | Mild fever | 39.5-39.8°C | Low (0.08-0.10) | Warning | First health event |
| 5 | Recovery | 38.5°C | Normal | - | Brief recovery |
| 6 | Estrus (12h) | 38.8-39.1°C | Very high (0.70-0.75) | Info | Reproductive event |
| 7-8 | Healthy | 38.5°C | Normal | - | Recovery period |
| 9 | Heat stress | 39.0-39.4°C | High (0.70) | Warning | Environmental stress |
| 10 | Recovery | 38.5°C | Normal | - | Brief recovery |
| 11-12 | Moderate fever | 39.8-40.5°C | Very low (0.05-0.07) | Critical | Acute illness |
| 13 | Recovery | 38.5°C | Normal | - | Brief recovery |
| 14-15 | Prolonged inactivity | 38.5°C | Extremely low (0.02) | Warning | Standalone inactivity |
| 16 | Recovery | 38.5°C | Normal | - | Brief recovery |
| 17-18 | Pregnancy indicator | 38.6°C | Reduced (0.25-0.35) | Info | Reproductive indicator |
| 19-20 | Severe fever | 41.0°C (peak 42.0°C) | Extremely low (0.03-0.05) | Critical | Worst scenario |
| 21 | Recovery/End | 38.5°C | Normal | - | Dataset ends |

### Expected Alerts: **13-15 total**

**Immediate Health Alerts** (11):
- Fever: 5 alerts (1 warning on day 3, 2 warning on days 11-12, 2 critical on days 19-20)
- Heat stress: 2 alerts (days 6, 9)
- Inactivity: 4 alerts (days 14-15, 19)

**Reproductive Alerts** (2-4):
- Estrus: 0-1 alert (day 6, 12-hour event)
- Pregnancy: 0-1 alert (days 17-18)

### Key Characteristics

- **Pattern**: Gradual escalation (mild → moderate → severe fever)
- **Recovery periods**: Multiple healthy days between events
- **Severe condition**: At END of dataset (days 19-20)
- **Final health score**: Low (35-55) due to severe fever at end

---

## Dataset #2: 15-Day Validation Test

**File**: [data/raw_test/COW_001_raw_sensor_data_2.csv](data/raw_test/COW_001_raw_sensor_data_2.csv)

**Size**: 3.4 MB (21,600 samples)
**Duration**: 15 days
**Purpose**: Validation testing with early critical condition

### Scenario Timeline

| Day | Scenario | Temperature | Motion | Severity | Notes |
|-----|----------|-------------|--------|----------|-------|
| 1 | Healthy baseline | 38.5°C | Normal | - | Baseline period |
| 2-3 | Severe fever | 40.8°C (peak 41.9°C) | Extremely low (0.04-0.05) | Critical | EARLY critical event |
| 4 | Recovery | 38.5°C | Normal | - | Brief recovery |
| 5 | Estrus (18h) | 38.8-39.1°C | Very high (0.70-0.75) | Info | Longer estrus period |
| 6 | Recovery | 38.5°C | Normal | - | Recovery period |
| 7-8 | Heat stress | 39.1-39.5°C | High (0.65-0.70) | Warning | SUSTAINED 2 days |
| 9 | Recovery | 38.5°C | Normal | - | Brief recovery |
| 10 | Moderate fever | 39.8-40.0°C | Very low (0.06-0.07) | Critical | Single-day fever |
| 11-12 | Prolonged inactivity | 38.6°C | Extremely low (0.02) | Warning | Post-fever lethargy |
| 13-15 | Pregnancy indicator | 38.6°C | Reduced (0.28-0.30) | Info | LONGER period (3 days) |

### Expected Alerts: **9-16 total**

**Immediate Health Alerts** (9-14):
- Fever: 4-6 alerts (2-3 critical on days 2-3, 1-2 critical on day 10)
- Heat stress: 2-4 alerts (days 7-8, sustained period)
- Inactivity: 3-6 alerts (days 2-3 with fever, days 11-12 standalone)

**Reproductive Alerts** (0-2):
- Estrus: 0-1 alert (day 5, 18-hour event - longer than dataset #1)
- Pregnancy: 0-1 alert (days 13-15, 3-day period)

### Key Characteristics

- **Pattern**: Immediate crisis (severe fever on day 2-3)
- **Sustained conditions**: Heat stress spans 2 full days
- **Severe condition**: At START of dataset (days 2-3)
- **Final health score**: Moderate (40-65) - pregnancy indicator at end

---

## Comparison Table

| Feature | Dataset #1 | Dataset #2 |
|---------|------------|------------|
| **Duration** | 21 days | 15 days |
| **Samples** | 30,240 | 21,600 |
| **File Size** | 4.8 MB | 3.4 MB |
| **Scenarios** | 7 conditions | 6 conditions |
| **Expected Alerts** | 13-15 | 9-16 |
| **Fever Severity** | 3 levels (mild/moderate/severe) | 2 levels (moderate/severe) |
| **Severe Fever Timing** | Days 19-20 (END) | Days 2-3 (START) |
| **Estrus Duration** | 12 hours | 18 hours |
| **Heat Stress Duration** | 1 day | 2 days |
| **Pregnancy Period** | 2 days | 3 days |
| **Recovery Periods** | Multiple throughout | Fewer, shorter |
| **Final Health Score** | Low (35-55) | Moderate (40-65) |
| **Testing Focus** | Gradual escalation | Early crisis + recovery |

---

## Testing Strategy

### Dataset #1: Primary QA Testing

**Use for:**
- Complete system validation
- Long-term trend analysis (21 days)
- Multiple severity level detection
- Recovery pattern analysis
- Final low health score validation

**Upload first** to establish baseline expectations.

### Dataset #2: Validation Testing

**Use for:**
- Confirm detection consistency
- Test early critical alert response
- Validate sustained condition detection (2-day heat stress)
- Verify reproductive detector with longer periods
- Test recovery scoring (moderate score at end)

**Upload second** to validate fixes work across different scenarios.

---

## Expected Results Comparison

### Alert Count

| Alert Type | Dataset #1 | Dataset #2 |
|------------|------------|------------|
| Fever (warning) | 3 | 0 |
| Fever (critical) | 4 | 4-6 |
| Heat stress | 2 | 2-4 |
| Inactivity | 4 | 3-6 |
| Estrus | 0-1 | 0-1 |
| Pregnancy | 0-1 | 0-1 |
| **TOTAL** | **13-15** | **9-16** |

### Health Score Trajectory

**Dataset #1**: Gradual decline
- Days 1-2: Excellent (85-100)
- Days 3-4: Good (70-80) - mild fever
- Days 11-12: Moderate (50-65) - moderate fever
- Days 19-20: Poor (35-55) - severe fever
- **Final score**: Low due to severe fever at end

**Dataset #2**: Recovery pattern
- Day 1: Excellent (85-100)
- Days 2-3: Poor (35-50) - severe fever
- Days 5-9: Good (65-80) - recovery with estrus/heat stress
- Day 10: Moderate (50-65) - moderate fever
- Days 13-15: Moderate (40-65) - pregnancy indicator
- **Final score**: Moderate due to stable pregnancy indicator

---

## Which Dataset to Test First?

### Start with Dataset #1 (21-day)

**Reasons:**
1. More comprehensive (7 scenarios vs 6)
2. Shows full severity progression (mild → moderate → severe)
3. Multiple recovery periods demonstrate scoring recovery
4. Established as primary QA dataset
5. Matches all documentation references

### Then Test Dataset #2 (15-day)

**Reasons:**
1. Validates system consistency with different patterns
2. Tests early critical alert detection
3. Confirms sustained condition handling (2-day heat stress)
4. Verifies final score reflects recovery (moderate vs low)
5. Shorter dataset = faster testing iterations

---

## Testing Commands

### Dataset #1 (21-day)

```bash
# Delete database
rm data/alert_state.db

# Start dashboard
streamlit run dashboard/app.py

# Upload in dashboard:
# - File: data/raw_test/COW_001_raw_sensor_data.csv
# - Cow ID: COW_001
# - Baseline Temp: 38.5
```

**Expected output:**
```
✅ Layer 3: Detected 11 immediate health alerts
✅ Layer 3: Detected 2 reproductive event(s)
✅ Layer 3 Complete: 13 total alerts detected
📊 Health score: 35-55/100 (poor/moderate)
```

### Dataset #2 (15-day)

```bash
# Delete database
rm data/alert_state.db

# Start dashboard
streamlit run dashboard/app.py

# Upload in dashboard:
# - File: data/raw_test/COW_001_raw_sensor_data_2.csv
# - Cow ID: COW_001
# - Baseline Temp: 38.5
```

**Expected output:**
```
✅ Layer 3: Detected 9-14 immediate health alerts
✅ Layer 3: Detected 0-2 reproductive event(s)
✅ Layer 3 Complete: 9-16 total alerts detected
📊 Health score: 40-65/100 (moderate)
```

---

## Success Criteria

### Both Datasets Should Pass:

✅ **Alert detection works**: 9-16 alerts detected
✅ **Critical severity present**: Critical alerts for severe/moderate fever
✅ **Multiple alert types**: fever, heat_stress, inactivity
✅ **Severity differentiation**: Warning vs Critical properly assigned
✅ **Timeline spread**: Alerts across multiple days
✅ **Health score calculation**: Score reflects condition severity
✅ **Reproductive detection**: 0-2 estrus/pregnancy alerts

### Validation Points:

- Dataset #1 final score (35-55) < Dataset #2 final score (40-65)
- Dataset #2 has more critical alerts early (days 2-3 vs days 19-20)
- Dataset #2 heat stress alerts span 2 days (days 7-8)
- Both datasets detect estrus and pregnancy (if detectors enabled)

---

## Summary

You now have two complementary test datasets:

1. **Dataset #1**: 21-day comprehensive test with gradual escalation
2. **Dataset #2**: 15-day validation test with early crisis

Both datasets test the fixed alert detection system thoroughly with different temporal patterns and severity distributions.

**Ready to test!** 🚀

Start with Dataset #1, then validate with Dataset #2.
