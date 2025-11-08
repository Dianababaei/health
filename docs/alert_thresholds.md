# Alert Threshold Codification Document - Artemis Health

**Document Version:** 1.0
**Last Updated:** 2025-11-08
**Audience:** Alert system implementers (Layer 3) and system testers
**Purpose:** Exact numerical thresholds for all alert types with codified rules and decision logic

---

## Table of Contents

1. [Overview](#overview)
2. [Alert Classification](#alert-classification)
3. [Alert Type Definitions](#alert-type-definitions)
   - [1. Fever Alert](#1-fever-alert)
   - [2. Heat Stress Alert](#2-heat-stress-alert)
   - [3. Prolonged Inactivity Alert](#3-prolonged-inactivity-alert)
   - [4. Estrus Alert](#4-estrus-alert)
   - [5. Pregnancy Indication Alert](#5-pregnancy-indication-alert)
   - [6. Sensor Malfunction Alert](#6-sensor-malfunction-alert)
4. [Alert Priority Levels](#alert-priority-levels)
5. [False Positive Mitigation Strategies](#false-positive-mitigation-strategies)
6. [Alert Decision Trees](#alert-decision-trees)
7. [Implementation Guidelines](#implementation-guidelines)
8. [References](#references)

---

## Overview

### Purpose

This document provides exact, testable thresholds for the Artemis Health alert system. All variables are fully specified with numerical values derived from veterinary literature and sensor behavior patterns.

### Alert Latency Categories

| Category | Latency | Alert Types | Use Case |
|----------|---------|-------------|----------|
| **Immediate Alerts** | 1-5 minutes | Fever, Heat Stress, Sensor Malfunction | Critical health emergencies |
| **Short-Term Alerts** | 5-10 minutes | Prolonged Inactivity (initial detection) | Health monitoring |
| **Pattern Alerts** | 30-60 minutes | Estrus | Reproductive management |
| **Trend Alerts** | 21-60 days | Pregnancy Indication | Long-term reproductive tracking |

### Data Sources

- **Layer 0:** `raw_sensor_readings` table (1-minute sensor data)
- **Layer 1:** `behavioral_states` table (behavior classification)
- **Layer 2:** `physiological_metrics` table (temperature analysis, circadian rhythm)
- **Individual Baselines:** Per-cow historical data (7-30 day rolling averages)

---

## Alert Classification

### Alert Categories

```
CRITICAL (Red)
├─ Fever
├─ Heat Stress
└─ Prolonged Inactivity (>8 hours)

WARNING (Yellow)
├─ Prolonged Inactivity (6-8 hours)
└─ Sensor Malfunction

INFO (Blue)
├─ Estrus
└─ Pregnancy Indication
```

### Alert Lifecycle

1. **Detection:** Threshold conditions met
2. **Confirmation:** Sustained for confirmation window
3. **Trigger:** Alert created in `alerts` table (status = 'active')
4. **Acknowledgment:** User/system marks as acknowledged
5. **Resolution:** Conditions normalized or manually resolved
6. **Cooldown:** Prevent duplicate alerts for same event (24-hour default)

---

## Alert Type Definitions

### 1. Fever Alert

**Alert Type:** `fever`
**Priority:** `CRITICAL` (red)
**Latency:** 5-10 minutes
**Database Table:** `alerts.alert_type = 'fever'`

#### Threshold Conditions

All conditions must be met simultaneously:

| Condition | Threshold | Source |
|-----------|-----------|--------|
| **Temperature (Absolute)** | > 39.5°C | Veterinary standard [1,2] |
| **Temperature (Relative)** | > baseline + 0.8°C | Individual cow baseline |
| **Motion Magnitude** | < 0.15 m/s² | Reduced activity indicator |
| **Duration** | ≥ 6 minutes sustained | Confirmation window [2] |

#### Detailed Specifications

**Temperature Thresholds:**
- **Absolute Threshold:** `current_temp > 39.5°C`
- **Relative Threshold:** `current_temp > (baseline_temp + 0.8°C)`
- **Rationale:** Combine absolute veterinary fever threshold (39.5°C) with individual variation detection (+0.8°C from baseline)

**Motion Calculation:**
```
motion_magnitude = sqrt(fxa² + mya² + rza²)
reduced_motion = motion_magnitude < 0.15 g  (where 1g ≈ 9.81 m/s²)
                ≈ 1.47 m/s²
```

**Confirmation Window:**
- **Duration:** 6 minutes sustained (6 consecutive 1-minute readings)
- **Rationale:** Avoids false positives from brief temperature spikes or momentary inactivity
- **Literature:** Studies show sustained fever >6 hours is best indicator, but we detect earlier for rapid response [2]

#### Alert Content

**Title:** `"High Fever Detected - Cow {cow_id}"`

**Details (JSONB):**
```json
{
  "current_temp": 40.2,
  "baseline_temp": 38.5,
  "temp_deviation": 1.7,
  "motion_magnitude": 0.12,
  "duration_minutes": 6,
  "first_detection": "2025-11-08T14:23:00Z"
}
```

**Sensor Values Snapshot:**
```json
{
  "temp": 40.2,
  "fxa": 0.05,
  "mya": 0.03,
  "rza": -0.58,
  "behavioral_state": "lying",
  "motion_intensity": 0.08
}
```

#### False Positive Mitigation

1. **Exclude Standing Rumination Near Ground:**
   - If `behavioral_state = 'ruminating'` AND `posture_context = 'standing'` AND `lyg` indicates head-down position
   - Collar may be near warm ground (false elevated temperature)
   - **Mitigation:** Require fever to persist when animal changes posture

2. **Baseline Comparison:**
   - Use individual cow baseline (not herd average)
   - Calculate baseline from 7-day rolling average during healthy periods
   - **Rationale:** Individual variation in normal temperature (38.0-39.0°C range)

3. **Circadian Rhythm Exclusion:**
   - Normal circadian temperature variation: ±0.5°C
   - If `circadian_rhythm_stability > 0.8` (stable rhythm), check if elevation is within expected circadian pattern
   - **Mitigation:** Only alert if deviation exceeds circadian expectations

4. **Activity Context:**
   - Ensure motion is truly reduced (not just brief rest)
   - Check behavioral state: lying + minimal rumination suggests illness
   - **Expected:** Sick animals show <4 hours rumination/day vs. normal 7-10 hours

#### Escalation Rules

- **Initial:** Priority = `CRITICAL`
- **If Persists >12 hours:** Send secondary alert with increased urgency
- **If Temp >40.5°C:** Immediate escalation (hyperthermia risk)

#### Cooldown Period

- **Duration:** 24 hours after resolution
- **Rationale:** Prevent duplicate alerts if fever is being treated
- **Override:** New alert if temperature increases by additional 0.5°C

---

### 2. Heat Stress Alert

**Alert Type:** `heat_stress`
**Priority:** `CRITICAL` (red)
**Latency:** 10 minutes
**Database Table:** `alerts.alert_type = 'heat_stress'`

#### Threshold Conditions

All conditions must be met simultaneously:

| Condition | Threshold | Source |
|-----------|-----------|--------|
| **Temperature** | > 40.0°C | Hyperthermia threshold [3,4] |
| **Activity Level** | > 0.50 (normalized 0-1 scale) | High sustained activity |
| **Duration** | ≥ 10 minutes sustained | Confirmation window |
| **Temperature Rise Rate** | > 0.5°C in 30 minutes | Rapid rise indicator |

#### Detailed Specifications

**Temperature Threshold:**
- **Absolute:** `current_temp > 40.0°C`
- **Rationale:** Hyperthermia begins above 40°C (105°F) [3,4]
- **Critical:** >40.5°C indicates emergency (>5-10% of herd with this temp requires immediate action) [4]

**Activity Level Calculation:**
```
activity_level = (motion_magnitude - baseline_motion) / baseline_motion
high_activity = activity_level > 0.50  (50% above baseline)

OR using behavioral states:
high_activity = behavioral_state IN ('walking', 'feeding')
                AND motion_intensity > 0.50
```

**Confirmation Window:**
- **Duration:** 10 minutes sustained (10 consecutive 1-minute readings)
- **Rationale:** Ensures persistent condition, not brief exertion

**Temperature Rise Rate:**
- Calculate over 30-minute rolling window
- `temp_rise_rate = (current_temp - temp_30min_ago) / 30`
- **Threshold:** `temp_rise_rate > 0.5°C per 30 minutes`

#### Alert Content

**Title:** `"Heat Stress Detected - Cow {cow_id}"`

**Details (JSONB):**
```json
{
  "current_temp": 40.3,
  "baseline_temp": 38.6,
  "temp_rise_rate_per_30min": 0.7,
  "activity_level": 0.62,
  "behavioral_state": "walking",
  "duration_minutes": 10,
  "ambient_temp": 35.0,
  "risk_level": "high"
}
```

#### False Positive Mitigation

1. **Differentiate from Normal Walking:**
   - Normal walking: Rhythmic Fxa patterns (0.8-1.2 Hz), moderate temp rise (<39.5°C)
   - Heat stress: Erratic high activity + high temp (>40°C)
   - **Check:** If `behavioral_state = 'walking'` is rhythmic and temp <40°C → Not heat stress

2. **Exclude Brief Exercise:**
   - Require sustained high activity + high temp for 10 minutes
   - Brief running or stress behavior may cause temporary elevation

3. **Ambient Temperature Context (if available):**
   - Heat stress more likely when `ambient_temp > 25°C (77°F)` [4]
   - If ambient sensor available, use THI (Temperature-Humidity Index)
   - **THI > 68:** Heat stress risk for high-producing dairy cows [4]

4. **Behavioral Pattern:**
   - Heat stress: Erratic, non-rhythmic high activity
   - Normal activity: Regular behavioral patterns
   - **Check:** Use `motion_intensity` variance - high variance suggests stress

#### Escalation Rules

- **Initial:** Priority = `CRITICAL`
- **If Temp >40.5°C:** Immediate escalation (risk of collapse)
- **If Duration >30 minutes:** Secondary alert for urgent intervention
- **If Multiple Cows Affected:** Herd-level heat stress alert (environmental issue)

#### Cooldown Period

- **Duration:** 12 hours after resolution
- **Override:** New alert if temperature rises again above 40°C

---

### 3. Prolonged Inactivity Alert

**Alert Type:** `prolonged_inactivity`
**Priority:** `WARNING` (yellow, escalates to `CRITICAL` if >8 hours)
**Latency:** 6-8 hours
**Database Table:** `alerts.alert_type = 'prolonged_inactivity'`

#### Threshold Conditions

| Condition | Threshold | Source |
|-----------|-----------|--------|
| **Lying Duration** | > 6 hours continuous | Initial warning threshold |
| **Motion Magnitude** | < 0.05 m/s² across all axes | Minimal movement |
| **Rumination Activity** | < 50% of normal | Reduced rumination |
| **Critical Threshold** | > 8 hours continuous | Escalation to CRITICAL |

#### Detailed Specifications

**Lying Detection:**
- **Posture:** `behavioral_state = 'lying'` (or `rza < -0.5g`)
- **Duration:** Continuous lying for >6 hours without standing
- **Rationale:** Normal rest bouts: 30 min - 4 hours; >6 hours suggests issue [5]

**Motion Threshold:**
```
motion_magnitude = sqrt(fxa² + mya² + rza²)
minimal_motion = motion_magnitude < 0.05 g  (≈ 0.49 m/s²)
```

**Rumination Check:**
- Calculate rumination minutes in last 6 hours
- **Normal:** 180-300 minutes ruminating per 6-hour period (assuming 7-10 hrs/day total)
- **Alert:** <90 minutes ruminating in 6-hour period (<50% of minimum normal)

**Exception Windows (Normal Rest Periods):**
- **Nighttime:** 10:00 PM - 6:00 AM (exclude from immediate alert)
- **Daytime:** 6:00 AM - 10:00 PM (alert if >6 hours continuous lying)
- **Rationale:** Increased lying at night is normal; prolonged daytime lying is concerning

#### Alert Content

**Title (WARNING):** `"Prolonged Inactivity - Cow {cow_id}"`
**Title (CRITICAL):** `"Severe Inactivity Alert - Downer Cow Risk - Cow {cow_id}"`

**Details (JSONB):**
```json
{
  "lying_duration_hours": 6.5,
  "motion_magnitude_avg": 0.03,
  "rumination_minutes_6hr": 45,
  "rumination_percent_normal": 25,
  "current_temp": 38.3,
  "temp_status": "normal",
  "time_of_day": "14:30",
  "alert_level": "WARNING"
}
```

#### Differentiation: Sick vs. Normal Rest

| Indicator | Normal Rest | Sick/Inactive |
|-----------|-------------|---------------|
| **Duration** | 30 min - 4 hours | >6 hours continuous |
| **Rumination** | Normal (Mya 40-60 cycles/min) | Absent or <50% normal |
| **Temperature** | 38.0-39.0°C | Often elevated (>39.5°C fever) or depressed |
| **Time of Day** | Nighttime (10 PM - 6 AM) | Daytime or excessive nighttime |
| **Motion** | Position changes, rumination | Minimal movement |

**Decision Logic:**
```
IF lying_duration > 6 hours AND rumination < 50% normal:
    IF temperature > 39.5°C:
        ALERT: "Prolonged Inactivity with Fever (possible illness)"
    ELSE IF temperature < 38.0°C:
        ALERT: "Prolonged Inactivity with Hypothermia (shock risk)"
    ELSE:
        ALERT: "Prolonged Inactivity (monitor for downer cow syndrome)"
```

#### Escalation Rules

- **6-8 hours:** Priority = `WARNING`
- **>8 hours:** Priority = `CRITICAL` (downer cow syndrome risk)
- **>12 hours:** Immediate escalation - secondary tissue damage risk [5]
- **With fever (>39.5°C):** Immediate `CRITICAL` regardless of duration

#### False Positive Mitigation

1. **Nighttime Exception:**
   - If lying period overlaps significantly with 10 PM - 6 AM, reduce alert severity
   - **Mitigation:** Only alert if lying extends >2 hours past 6 AM

2. **Normal Rumination Pattern:**
   - If rumination is normal (>180 min in 6 hours), reduce alert priority
   - **Interpretation:** Animal is resting normally, not sick

3. **Recent Calving:**
   - Post-calving cows may lie longer (recovery period)
   - **Mitigation:** If cow calved in last 7 days, increase threshold to 8 hours before alerting

4. **Behavioral Context:**
   - Check if animal stands when approached (if camera/visual data available)
   - **True Downer:** Unable to rise even with stimulation
   - **Resting:** Stands when motivated

#### Cooldown Period

- **Duration:** 24 hours after standing and resuming normal activity
- **Resolution Criteria:** Standing for >1 hour continuously AND rumination resumes (>40 cycles/min)

---

### 4. Estrus Alert

**Alert Type:** `estrus`
**Priority:** `INFO` (blue)
**Latency:** 30-60 minutes
**Database Table:** `alerts.alert_type = 'estrus'`

#### Threshold Conditions

Both conditions must be met:

| Condition | Threshold | Source |
|-----------|-----------|--------|
| **Temperature Rise** | +0.3 to +0.6°C above baseline | Estrus indicator [6,7] |
| **Activity Increase** | >40% above 7-day average | Activity surge [6,7] |
| **Pattern Duration** | Sustained 30-60 minutes | Confirmation window |
| **Temperature Duration** | Elevated for >3 hours | Estrus window indicator [7] |

#### Detailed Specifications

**Temperature Rise:**
- **Baseline:** Individual cow's 7-day rolling average during non-estrus periods
- **Rise Amount:** `current_temp - baseline_temp` = +0.3°C to +0.6°C
- **Duration:** Elevated for >3 hours (vaginal temp remains >+0.3°C for 6.8±4.6 hours) [7]
- **Rationale:** Estrus-specific temperature rise; excludes fever (>39.5°C absolute)

**Activity Increase Calculation:**
```
activity_7day_avg = AVG(motion_magnitude) over last 7 days
activity_current = motion_magnitude (1-hour rolling average)
activity_increase_percent = ((activity_current - activity_7day_avg) / activity_7day_avg) * 100

estrus_activity = activity_increase_percent > 40%
```

**Alternative Activity Metrics:**
- **Step Count:** 1.5-fold increase over monthly average [7]
- **Standing Time:** Increased standing and walking vs. lying
- **Behavioral:** Increased walking, reduced rumination (short-term)

**Confirmation Window:**
- **Initial Detection:** 30 minutes (temp rise + activity increase both present)
- **Sustained Pattern:** Confirm over 60 minutes
- **Estrus Duration:** Typically 12-18 hours total (peak breeding window)

**Peak Detection:**
- **Optimal Breeding:** 10-12 hours after estrus onset
- **Alert Timing:** Trigger alert at onset for timely insemination planning

#### Alert Content

**Title:** `"Estrus Detected - Cow {cow_id}"`

**Details (JSONB):**
```json
{
  "temp_rise_celsius": 0.45,
  "baseline_temp": 38.5,
  "current_temp": 38.95,
  "activity_increase_percent": 65,
  "activity_7day_avg": 0.25,
  "activity_current": 0.41,
  "duration_hours_elevated": 4.2,
  "detection_time": "2025-11-08T06:30:00Z",
  "estimated_breeding_window": "2025-11-08T16:30:00Z to 2025-11-08T22:30:00Z"
}
```

**Behavioral Correlation (if Layer 1 available):**
```json
{
  "standing_increase_percent": 35,
  "walking_increase_percent": 180,
  "rumination_minutes_12hr": 200,
  "rumination_reduction_percent": 25
}
```

#### False Positive Mitigation

1. **Exclude Fever:**
   - **Check:** If `current_temp > 39.5°C` (absolute fever threshold)
   - **Interpretation:** Fever (illness), NOT estrus
   - **Mitigation:** Suppress estrus alert, trigger fever alert instead

2. **Temperature Rise Magnitude:**
   - **Estrus:** +0.3°C to +0.6°C (moderate, specific rise)
   - **Fever:** >+0.8°C (large rise above baseline)
   - **Mitigation:** If rise >0.7°C, classify as potential fever, not estrus

3. **Activity Pattern:**
   - **Estrus:** Sustained elevated activity (daytime peak), 2.8-fold increase during turnout [7]
   - **Stress:** Erratic, brief high activity spikes
   - **Mitigation:** Check for sustained pattern over hours, not minutes

4. **Circadian Alignment:**
   - **Estrus:** Typically peaks during daytime hours (better detection with daylight)
   - **Nighttime:** Reduced detection accuracy
   - **Context:** Weight daytime activity increases more heavily

5. **Previous Estrus Cycle:**
   - **Normal Cycle:** 21 days (18-24 days)
   - **Check:** Last estrus alert timestamp
   - **Mitigation:** If <15 days since last estrus, reduce alert confidence (potential false positive)
   - **Validation:** If ~21 days since last estrus, high confidence

#### Estrus Window Calculation

```
estrus_onset = timestamp of initial detection
breeding_window_start = estrus_onset + 10 hours
breeding_window_end = estrus_onset + 18 hours
optimal_AI_time = estrus_onset + 12 hours
```

#### Cooldown Period

- **Duration:** 15 days (prevent false repeat alerts within cycle)
- **Next Expected Estrus:** 21 days from current alert
- **Override:** If new temperature rise + activity increase after 18 days, allow new alert

---

### 5. Pregnancy Indication Alert

**Alert Type:** `pregnancy_detected`
**Priority:** `INFO` (blue)
**Latency:** 21-28 days (early indication) or 60+ days (confirmation)
**Database Table:** `alerts.alert_type = 'pregnancy_detected'`

#### Threshold Conditions

All conditions must be met over multi-day analysis:

| Condition | Threshold | Source |
|-----------|-----------|--------|
| **Temperature Stability** | Variance < 0.20°C over 7 days | Stable post-conception |
| **Activity Reduction** | >20% below pre-estrus baseline | Gradual decline |
| **Time Post-Estrus** | 21-28 days (early) or 60+ days (confirm) | Pregnancy timeline |
| **Previous Estrus** | Estrus alert in system history | Required precondition |
| **No Subsequent Estrus** | No estrus alert in 21-28 day window | Non-return to estrus |

#### Detailed Specifications

**Temperature Stability:**
```
temp_variance_7day = VARIANCE(daily_avg_temp) over last 7 days
stable_temp = temp_variance_7day < 0.20°C
```

**Rationale:**
- Pregnant cows maintain stable baseline temperature post-conception
- No estrus-related temperature spikes in subsequent cycles
- **Variance Threshold:** <0.20°C indicates stable hormonal state

**Activity Reduction:**
```
pre_estrus_activity_baseline = AVG(motion_magnitude) 7 days BEFORE last estrus
current_activity = AVG(motion_magnitude) over last 7 days
activity_reduction_percent = ((pre_estrus_activity_baseline - current_activity) / pre_estrus_activity_baseline) * 100

pregnancy_indicator = activity_reduction_percent > 20%
```

**Gradual Activity Decline:**
- **Not sudden drop** (sudden = illness)
- **Gradual reduction** over weeks post-estrus
- **Behavioral:** Reduced walking, increased lying/rumination time

**Time Windows:**

| Window | Days Post-Estrus | Confidence | Method |
|--------|------------------|------------|--------|
| **Early Indication** | 21-28 days | Medium | Non-return to estrus + stable temp |
| **Intermediate** | 28-45 days | Medium-High | Continued stability + activity reduction |
| **Confirmation** | 60+ days | High | Strong behavioral/physiological indicators |

**Required Precondition:**
- **Previous Estrus Alert:** Must exist in `alerts` table with `alert_type = 'estrus'`
- **Timing:** 21-60+ days ago
- **Validation:** Check `SELECT * FROM alerts WHERE cow_id = X AND alert_type = 'estrus' AND timestamp BETWEEN NOW() - INTERVAL '65 days' AND NOW() - INTERVAL '18 days'`

**No Subsequent Estrus:**
- Expected estrus cycle: 21 days (±3 days)
- **Check:** No estrus alert in 21-28 day window post-conception
- **Non-Return:** Strong pregnancy indicator (80-90% accuracy) [8]

#### Alert Content

**Title (21-28 days):** `"Early Pregnancy Indication - Cow {cow_id}"`
**Title (60+ days):** `"Pregnancy Confirmation - Cow {cow_id}"`

**Details (JSONB):**
```json
{
  "days_post_estrus": 25,
  "last_estrus_date": "2025-10-14",
  "temp_variance_7day": 0.15,
  "temp_stability_score": 0.92,
  "activity_reduction_percent": 28,
  "pre_estrus_activity": 0.32,
  "current_activity": 0.23,
  "no_subsequent_estrus": true,
  "confidence_level": "medium",
  "estimated_calving_date": "2025-07-22",
  "recommendation": "Confirm with veterinary examination"
}
```

#### False Positive Mitigation

1. **Differentiate from Illness:**
   - **Pregnancy:** Gradual activity decline, stable normal temperature (38.0-39.0°C)
   - **Illness:** Sudden activity drop, fever (>39.5°C) or hypothermia
   - **Check:** Temperature must remain in normal range
   - **Mitigation:** If any fever alerts in analysis window, suppress pregnancy alert

2. **Require Gradual Decline:**
   - **Week 1 post-estrus:** Baseline activity
   - **Week 2-3:** 10-15% reduction
   - **Week 4+:** 20-30% reduction
   - **Mitigation:** Check for smooth decline, not abrupt changes

3. **Estrus Cycle Validation:**
   - **Check:** Previous estrus alert exists and timing aligns with 21-day cycle
   - **Mitigation:** If no previous estrus in system, reduce confidence or suppress alert

4. **Temperature Anomaly Exclusion:**
   - **Check:** No high anomaly scores in `physiological_metrics.temp_anomaly_score`
   - **Threshold:** `temp_anomaly_score < 0.3` (low anomaly)
   - **Mitigation:** Stable = healthy pregnancy; instability = potential issue

5. **Behavioral Consistency:**
   - **Check:** Rumination time remains normal (7-10 hours/day)
   - **Mitigation:** Pregnancy doesn't drastically reduce rumination; illness does

#### Pregnancy Timeline

```
Day 0: Estrus detection (breeding window)
Day 21-24: Expected return to estrus IF NOT pregnant
Day 21-28: Early pregnancy indication (non-return + stability)
Day 30-45: Intermediate confirmation (continued stability)
Day 60+: Strong pregnancy confirmation
Day 280-283: Expected calving (gestation: ~283 days)
```

#### Cooldown Period

- **Duration:** None (one-time alert per pregnancy)
- **Update Alerts:** Can issue updated confidence alerts at 30, 60, 90 days
- **Resolution:** After calving or pregnancy loss (return to estrus)

---

### 6. Sensor Malfunction Alert

**Alert Type:** `sensor_malfunction`
**Priority:** `WARNING` (yellow)
**Latency:** Immediate (no confirmation window)
**Database Table:** `alerts.alert_type = 'sensor_malfunction'`

#### Threshold Conditions

Any ONE of the following triggers immediate alert:

| Condition | Threshold | Description |
|-----------|-----------|-------------|
| **No Data** | >5 minutes without readings | Connectivity/power issue |
| **Stuck Values** | Identical values >2 hours | Sensor hardware failure |
| **Out-of-Range Temp** | <35°C or >42°C | Beyond physiological limits |
| **Out-of-Range Accel** | \|Fxa\|, \|Mya\|, \|Rza\| > 5g | Unrealistic acceleration |
| **Out-of-Range Gyro** | \|Sxg\|, \|Lyg\|, \|Dzg\| > 300°/s | Unrealistic angular velocity |
| **Contradictory Signals** | Lying posture + walking motion | Logic conflict |
| **Data Quality Flag** | `data_quality` = 'sensor_error' | Explicit error signal |

#### Detailed Specifications

**1. No Data Received:**
```
time_since_last_reading = NOW() - MAX(timestamp) FROM raw_sensor_readings WHERE cow_id = X
alert_trigger = time_since_last_reading > 5 minutes
```

**Rationale:**
- Data transmitted every 1 minute normally
- >5 minutes = connectivity issue, battery failure, or sensor offline

**2. Stuck Values:**
```
SELECT COUNT(DISTINCT (temperature, fxa, mya, rza, sxg, lyg, dzg))
FROM raw_sensor_readings
WHERE cow_id = X
  AND timestamp >= NOW() - INTERVAL '2 hours'
```

**Alert if:** Count = 1 (all readings identical for 2 hours)

**Rationale:**
- Animal movement ensures sensor values change continuously
- Identical values >2 hours = sensor frozen or hardware failure

**3. Out-of-Range Temperature:**
```
alert_temp_low = temperature < 35.0°C  (hypothermia beyond survival)
alert_temp_high = temperature > 42.0°C (fatal hyperthermia)
```

**Physiological Limits:**
- **Normal:** 38.0-39.3°C
- **Fever:** 39.5-40.5°C
- **Hyperthermia:** 40.5-42.0°C
- **Lethal:** >42.0°C (sensor error more likely than actual)
- **Hypothermia:** <37.0°C (severe), <35.0°C (sensor error likely)

**4. Out-of-Range Accelerations:**
```
alert_fxa = ABS(fxa) > 5.0 g
alert_mya = ABS(mya) > 5.0 g
alert_rza = ABS(rza) > 5.0 g
```

**Rationale:**
- Cattle movement typically: -2g to +2g range
- **Walking:** 0.3-1.5g peak-to-peak
- **Running:** Up to 2-3g
- **>5g:** Sensor impact, fall, or malfunction (not normal cattle movement)

**5. Out-of-Range Angular Velocities:**
```
alert_sxg = ABS(sxg) > 300 degrees/second
alert_lyg = ABS(lyg) > 300 degrees/second
alert_dzg = ABS(dzg) > 300 degrees/second
```

**Rationale:**
- Normal head movements: 5-30 degrees/second
- **Feeding:** Lyg up to 15-25 degrees/second
- **>300 degrees/second:** Sensor error or extreme event (collar loose/falling)

**6. Contradictory Signals:**

**Example 1: Lying posture + High walking motion**
```
contradiction_1 = (rza < -0.5 AND fxa_std > 0.3)
   → Rza says lying, but Fxa variance says walking
```

**Example 2: Standing posture + No motion for hours**
```
contradiction_2 = (rza > 0.7 AND motion_magnitude < 0.05 FOR >4 hours)
   → Rza says standing, but no movement (unrealistic)
```

**Example 3: Temperature anomaly + All other sensors zero**
```
contradiction_3 = (temperature > 40.0 AND fxa = 0 AND mya = 0 AND rza = 0)
   → Only temperature reading, all motion sensors flatlined
```

**7. Data Quality Flag:**
```
alert_quality = data_quality IN ('poor', 'sensor_error')
```

**Quality Levels (from schema):**
- `good` - Normal operation
- `degraded` - Weak signal or minor issues
- `poor` - Significant issues, data unreliable
- `sensor_error` - Explicit sensor hardware error

#### Alert Content

**Title:** `"Sensor Malfunction - Cow {cow_id} - {malfunction_type}"`

**Malfunction Types:**
- `no_data` - No readings received
- `stuck_values` - Sensor frozen
- `temp_out_of_range` - Impossible temperature
- `accel_out_of_range` - Impossible acceleration
- `gyro_out_of_range` - Impossible angular velocity
- `contradictory_signals` - Logic conflict in data
- `data_quality_error` - Explicit error flag

**Details (JSONB):**
```json
{
  "malfunction_type": "stuck_values",
  "last_reading_time": "2025-11-08T14:23:00Z",
  "time_since_last_reading_minutes": 8,
  "stuck_values": {
    "temperature": 38.5,
    "fxa": 0.12,
    "mya": 0.08,
    "rza": 0.72,
    "duration_hours": 2.5
  },
  "sensor_id": "SENSOR-A3F2-9821",
  "battery_level": 12,
  "signal_strength_rssi": -85,
  "recommendation": "Check sensor battery and collar placement"
}
```

#### False Positive Mitigation

1. **Stuck Values - Allow Brief Periods:**
   - **Threshold:** 2 hours (not 30 minutes)
   - **Rationale:** Very still lying cow might have minimal changes for 30-60 min
   - **Mitigation:** Require extended period to confirm true malfunction

2. **Out-of-Range - Single Reading Grace:**
   - **Threshold:** 2 consecutive out-of-range readings (2 minutes)
   - **Rationale:** Brief electrical noise might cause single spurious reading
   - **Mitigation:** Require sustained out-of-range before alerting

3. **No Data - Connectivity Check:**
   - **Check:** Other sensors in same barn/area
   - **If multiple sensors offline:** Network issue, not individual sensor
   - **Mitigation:** Issue "network malfunction" alert instead of per-sensor alerts

4. **Battery Level Context:**
   - **Check:** `metadata->>'battery'` in last reading
   - **If battery <15%:** Flag as likely power issue vs. hardware failure
   - **Mitigation:** Differentiate "battery_low" vs "sensor_error" alerts

#### Escalation Rules

- **Initial:** Priority = `WARNING`
- **If >30 min no data:** Escalate to `CRITICAL` (animal unmonitored)
- **If multiple sensors fail:** Escalate to `CRITICAL` (system-wide issue)

#### Cooldown Period

- **Duration:** None (immediate re-alert if issue persists or recurs)
- **Resolution:** After sensor data returns to normal for 10 minutes

---

## Alert Priority Levels

### Priority Definitions

| Priority | Color | Dashboard Display | Notification | Example Alert Types |
|----------|-------|-------------------|--------------|---------------------|
| **CRITICAL** | Red | Top of list, flashing | Immediate push notification | Fever, Heat Stress, Prolonged Inactivity (>8hr) |
| **WARNING** | Yellow | Standard display | Dashboard badge | Prolonged Inactivity (6-8hr), Sensor Malfunction |
| **INFO** | Blue | Summary section | Log only | Estrus, Pregnancy Indication |

### Priority Assignment Logic

```sql
-- Alert priority determination
CASE
    WHEN alert_type = 'fever' THEN 'CRITICAL'
    WHEN alert_type = 'heat_stress' THEN 'CRITICAL'
    WHEN alert_type = 'prolonged_inactivity' AND duration_hours > 8 THEN 'CRITICAL'
    WHEN alert_type = 'prolonged_inactivity' AND duration_hours BETWEEN 6 AND 8 THEN 'WARNING'
    WHEN alert_type = 'sensor_malfunction' AND time_since_last_reading > 30 THEN 'CRITICAL'
    WHEN alert_type = 'sensor_malfunction' THEN 'WARNING'
    WHEN alert_type = 'estrus' THEN 'INFO'
    WHEN alert_type = 'pregnancy_detected' THEN 'INFO'
    ELSE 'WARNING'
END AS severity
```

### Escalation Rules

```
Initial Alert → Check Duration/Severity → Escalate if Worsens

Examples:
1. Prolonged Inactivity:
   6 hours → WARNING
   8 hours → CRITICAL (auto-escalate)

2. Fever:
   39.5°C → CRITICAL
   40.5°C → CRITICAL + Secondary Alert (hyperthermia)

3. Sensor Malfunction:
   No data 5 min → WARNING
   No data 30 min → CRITICAL (animal unmonitored)
```

---

## False Positive Mitigation Strategies

### General Principles

1. **Multi-Condition Logic**
   - Require 2+ independent signals for confirmation
   - Example: Fever = high temp AND low motion (not just high temp)

2. **Individual Baselines**
   - Use per-cow historical data (7-30 day rolling averages)
   - Avoid herd-wide averages (individual variation 38.0-39.0°C normal range)

3. **Behavioral Context Integration**
   - Leverage Layer 1 behavioral states when available
   - Example: Lying + minimal rumination = sick vs. Lying + normal rumination = resting

4. **Circadian Rhythm Awareness**
   - Normal temperature variation: ±0.5°C over 24 hours
   - Exclude alerts during expected fluctuations
   - Check `physiological_metrics.circadian_rhythm_stability`

5. **Confirmation Windows**
   - Require sustained conditions (5-10 min for immediate alerts)
   - Avoid single-minute anomalies triggering alerts

6. **Alert Cooldown Periods**
   - Prevent duplicate alerts for same event (24-hour default)
   - Only override cooldown if conditions significantly worsen

### Specific Strategies by Alert Type

| Alert Type | Primary Mitigation | Secondary Mitigation |
|------------|-------------------|---------------------|
| **Fever** | Baseline + duration (6 min) | Circadian exclusion, posture context |
| **Heat Stress** | Duration (10 min) | Differentiate from normal walking |
| **Prolonged Inactivity** | Nighttime exception, rumination check | Temperature context (fever vs. normal) |
| **Estrus** | Exclude fever (>39.5°C), cycle timing | Activity pattern validation |
| **Pregnancy** | Gradual decline validation, temperature stability | Previous estrus required |
| **Sensor Malfunction** | 2 consecutive readings for out-of-range | Grace period for stuck values (2 hours) |

---

## Alert Decision Trees

### Decision Tree 1: High Temperature Detection

```
Temperature > 38.0°C
├─ Temperature > 39.5°C?
│  ├─ YES → Fever or Heat Stress?
│  │  ├─ Motion < 0.15 m/s²?
│  │  │  ├─ YES → FEVER (temp + low motion)
│  │  │  └─ NO → Check activity level
│  │  │     ├─ Activity > 0.50 (high)?
│  │  │     │  ├─ YES → HEAT STRESS (temp + high activity)
│  │  │     │  └─ NO → FEVER (default if temp >39.5°C)
│  │  │     └─ Sustained >6-10 min? → CONFIRM and ALERT
│  └─ NO → Temperature in 38.0-39.5°C range
│     ├─ Rise from baseline > 0.3°C?
│     │  ├─ YES → Estrus or Illness?
│     │  │  ├─ Rise = 0.3-0.6°C AND Activity +40%?
│     │  │  │  ├─ YES → ESTRUS
│     │  │  │  └─ NO → Monitor (potential early fever)
│     │  │  └─ Rise > 0.7°C?
│     │  │     └─ YES → Early FEVER warning
│     │  └─ NO → Normal variation
│     └─ Normal circadian fluctuation
└─ Temperature < 38.0°C
   ├─ Temperature < 35.0°C? → SENSOR MALFUNCTION (out of range)
   └─ Temperature 35.0-38.0°C
      ├─ Prolonged <37.0°C? → HYPOTHERMIA alert (shock risk)
      └─ Monitor for recovery
```

### Decision Tree 2: Prolonged Lying Detection

```
Behavioral State = 'lying' OR Rza < -0.5g
├─ Duration > 4 hours?
│  ├─ YES → Continue to detailed check
│  │  ├─ Duration > 6 hours?
│  │  │  ├─ YES → Inactivity concern
│  │  │  │  ├─ Time of day?
│  │  │  │  │  ├─ Nighttime (10 PM - 6 AM)?
│  │  │  │  │  │  ├─ YES → Normal rest (suppress alert)
│  │  │  │  │  │  └─ NO → Daytime lying >6 hr → Check rumination
│  │  │  │  │  └─ Daytime → Check rumination
│  │  │  │  │     ├─ Rumination < 50% normal?
│  │  │  │  │     │  ├─ YES → Sick behavior
│  │  │  │  │     │  │  ├─ Temperature > 39.5°C?
│  │  │  │  │     │  │  │  ├─ YES → CRITICAL: Inactivity + Fever
│  │  │  │  │     │  │  │  └─ NO → WARNING: Prolonged Inactivity
│  │  │  │  │     │  └─ NO → Normal rest
│  │  │  │  │     └─ Duration > 8 hours?
│  │  │  │  │        ├─ YES → CRITICAL: Downer Cow Risk
│  │  │  │  │        └─ NO (6-8 hr) → WARNING: Monitor
│  │  │  └─ NO (4-6 hr) → Normal long rest
│  └─ NO (<4 hours) → Normal rest period
└─ Not lying → No alert
```

### Decision Tree 3: Estrus vs. Fever Differentiation

```
Temperature Rise Detected (>+0.3°C from baseline)
├─ Absolute Temperature?
│  ├─ >39.5°C (Fever threshold)
│  │  └─ FEVER (not estrus) → Check motion for confirmation
│  └─ <39.5°C → Continue estrus check
│     ├─ Temperature Rise = +0.3 to +0.6°C?
│     │  ├─ YES → Estrus range
│     │  │  ├─ Activity increase >40%?
│     │  │  │  ├─ YES → ESTRUS (temp + activity match)
│     │  │  │  │  ├─ Last estrus ~21 days ago?
│     │  │  │  │  │  ├─ YES → High confidence ESTRUS
│     │  │  │  │  │  └─ NO → Medium confidence ESTRUS
│     │  │  │  │  └─ Sustained 3+ hours? → CONFIRM ESTRUS
│     │  │  │  └─ NO → Ambiguous (temp rise without activity)
│     │  │  │     └─ Monitor for fever development
│     │  └─ NO → Check if rise >+0.7°C
│     │     ├─ YES → Early FEVER (large rise)
│     │     └─ NO → Inconclusive, monitor
│     └─ Temperature Rise >+0.6°C?
│        └─ YES → Likely early FEVER (too large for estrus)
└─ Temperature stable or declining → No alert
```

---

## Implementation Guidelines

### Database Integration

**Alert Creation SQL:**
```sql
INSERT INTO alerts (
    timestamp, cow_id, alert_type, severity, title, details, status, sensor_values, related_metrics
)
VALUES (
    NOW(),
    1042,
    'fever',
    'critical',
    'High Fever Detected - Cow 1042',
    '{"current_temp": 40.2, "baseline_temp": 38.5, "temp_deviation": 1.7, "motion_magnitude": 0.12, "duration_minutes": 6}'::jsonb,
    'active',
    '{"temp": 40.2, "fxa": 0.05, "mya": 0.03, "rza": -0.58, "behavioral_state": "lying"}'::jsonb,
    '{"temp_anomaly_score": 0.95, "circadian_rhythm_stability": 0.65}'::jsonb
)
RETURNING alert_id;
```

**Alert Check Query (Fever Example):**
```sql
WITH cow_baseline AS (
    SELECT
        cow_id,
        AVG(temperature) AS baseline_temp,
        AVG(SQRT(fxa*fxa + mya*mya + rza*rza)) AS baseline_motion
    FROM raw_sensor_readings
    WHERE cow_id = 1042
      AND timestamp >= NOW() - INTERVAL '7 days'
      AND data_quality = 'good'
    GROUP BY cow_id
),
recent_readings AS (
    SELECT
        timestamp,
        temperature,
        SQRT(fxa*fxa + mya*mya + rza*rza) AS motion_magnitude
    FROM raw_sensor_readings
    WHERE cow_id = 1042
      AND timestamp >= NOW() - INTERVAL '10 minutes'
    ORDER BY timestamp DESC
)
SELECT
    COUNT(*) AS fever_minutes,
    MAX(r.temperature) AS max_temp,
    AVG(r.motion_magnitude) AS avg_motion
FROM recent_readings r
CROSS JOIN cow_baseline b
WHERE r.temperature > 39.5  -- Absolute threshold
  AND r.temperature > (b.baseline_temp + 0.8)  -- Relative threshold
  AND r.motion_magnitude < 0.15  -- Reduced motion
HAVING COUNT(*) >= 6;  -- 6 minutes sustained
```

### Polling Frequency

| Alert Type | Check Frequency | Rationale |
|------------|----------------|-----------|
| Fever | Every 1-2 minutes | Immediate detection needed |
| Heat Stress | Every 2-5 minutes | Fast response critical |
| Prolonged Inactivity | Every 15-30 minutes | Long-duration condition |
| Estrus | Every 10-15 minutes | Pattern emerges over hours |
| Pregnancy | Daily (batch job) | Long-term trend analysis |
| Sensor Malfunction | Every 1 minute | Immediate detection of data loss |

### Performance Optimization

1. **Use Continuous Aggregates:**
   - Pre-compute hourly/daily metrics in `sensor_hourly`, `behavior_daily`
   - Reduce real-time computation load

2. **Index Utilization:**
   - Queries should use `idx_raw_sensor_cow_time`, `idx_physiological_cow_time`
   - Filter by `cow_id` and recent `timestamp` ranges

3. **Batch Processing:**
   - Process multiple cows in single query when possible
   - Use CTEs and window functions for efficiency

4. **Alert Deduplication:**
   - Check for existing active alerts before creating new ones
   - Use cooldown periods to prevent spam

**Cooldown Check:**
```sql
SELECT EXISTS(
    SELECT 1
    FROM alerts
    WHERE cow_id = 1042
      AND alert_type = 'fever'
      AND status IN ('active', 'acknowledged')
      AND timestamp >= NOW() - INTERVAL '24 hours'
) AS alert_exists;
```

### Alert Resolution

**Auto-Resolution Conditions:**

| Alert Type | Resolution Criteria | Check Frequency |
|------------|---------------------|-----------------|
| Fever | Temp <39.0°C for 30 min | Every 5 min |
| Heat Stress | Temp <39.5°C AND normal activity for 20 min | Every 5 min |
| Prolonged Inactivity | Standing for >30 min AND rumination resumed | Every 15 min |
| Estrus | Temp returns to baseline AND activity normalizes | Every 30 min |
| Pregnancy | N/A (manual resolution or calving event) | Manual |
| Sensor Malfunction | Normal data for 10 min | Every 1 min |

**Resolution SQL:**
```sql
UPDATE alerts
SET status = 'resolved',
    resolved_at = NOW(),
    resolved_by = 'AUTO_SYSTEM',
    resolution_notes = 'Temperature returned to normal (<39.0°C) for 30 minutes'
WHERE cow_id = 1042
  AND alert_type = 'fever'
  AND status = 'active'
  AND alert_id = 12847;
```

---

## References

### Veterinary Literature

1. **Barker, Z. E., et al. (2018).** "Behavioral changes associated with fever in transition dairy cows." *Journal of Dairy Science.* Fever threshold: 39.4-39.7°C (39.5°C commonly used).

2. **Vickers, L. A., et al. (2010).** "Comparison of rectal and vaginal temperatures in lactating dairy cows." *Journal of Dairy Science.* Duration-based monitoring: body temperature >39.7°C for ≥6 hours is the best early indicator of illness.

3. **Gaughan, J. B., et al. (2019).** "Non-Invasive Physiological Indicators of Heat Stress in Cattle." *Animals.* Normal core body temperature: 38.0-39.3°C; hyperthermia begins above 40°C.

4. **West, J. W. (2003).** "Effects of heat-stress on production in dairy cattle." *Journal of Dairy Science.* Heat stress threshold: ambient >25°C (77°F), body temp >40.5°C is emergency (>5-10% of herd).

5. **Weary, D. M., & Tucker, C. B. (2017).** "Bovine Secondary Recumbency and Downer Cow Syndrome." *Merck Veterinary Manual.* Prolonged recumbency: ≥12-24 hours; secondary tissue damage after >24 hours.

6. **Saint-Dizier, M., & Chastant-Maillard, S. (2012).** "Towards an automated detection of oestrus in dairy cattle." *Reproduction in Domestic Animals.* Activity increase: 2.8-fold at estrus (daytime); step count 1.5-fold over average.

7. **Sakatani, M., et al. (2016).** "The efficiency of vaginal temperature measurement for detection of estrus in Japanese Black cows." *Theriogenology.* Vaginal temp rise: +0.6±0.3°C at estrus, elevated ≥0.3°C for 6.8±4.6 hours.

8. **Fricke, P. M., et al. (2014).** "Fertility of dairy cows after resynchronization of ovulation at three intervals following first timed insemination." *Journal of Dairy Science.* Non-return to estrus: 80-90% accuracy for pregnancy indication at 21-24 days post-breeding.

### Project Documentation

9. [description.md](../description.md) - System overview and alert type specifications
10. [behavioral_sensor_signatures.md](behavioral_sensor_signatures.md) - Behavioral state patterns for alert context
11. [database_schema.md](database_schema.md) - TimescaleDB schema for alert storage

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-08 | Artemis Health Team | Initial codification with exact thresholds for all 6 alert types |

---

**End of Document**
