# Livestock Behavior Pattern Specifications

## Document Overview

This document provides comprehensive research-based specifications for the 6 behavioral patterns used in the Artemis Health livestock monitoring system. All parameter ranges are derived from livestock physiology research, accelerometer/gyroscope sensor characteristics, and field data analysis.

**Version:** 1.0  
**Last Updated:** 2024  
**System:** Neck-mounted sensor collar with 3-axis accelerometer, 3-axis gyroscope, temperature sensor  
**Sampling Rate:** 1 sample per minute  

---

## Table of Contents

1. [Research Foundation & Methodology](#research-foundation--methodology)
2. [Sensor Coordinate System](#sensor-coordinate-system)
3. [Behavior Definitions](#behavior-definitions)
4. [Parameter Rationale by Behavior](#parameter-rationale-by-behavior)
5. [Distinguishing Features Matrix](#distinguishing-features-matrix)
6. [Transition Patterns](#transition-patterns)
7. [Validation Rules](#validation-rules)
8. [References & Sources](#references--sources)

---

## Research Foundation & Methodology

### Physiological Basis

**Cattle Body Temperature:**
- Normal range: 38.3-39.1°C (101.0-102.5°F)
- Average: 38.6°C (101.5°F)
- Fever threshold: >39.5°C (103°F)
- Hypothermia: <37.5°C (99.5°F)
- Diurnal variation: ±0.5°C throughout day
- Heat stress threshold: >39.2°C sustained

**Biomechanical Principles:**
- Gravity constant: 9.8 m/s²
- Cattle walking speed: 0.8-1.3 m/s (typical)
- Walking gait frequency: 1-2 Hz (60-120 steps/minute)
- Ruminating frequency: 60-90 jaw cycles/minute (1-1.5 Hz)
- Neck angle during feeding: ~45-60° downward
- Resting metabolic rate: Lower temperature during lying

### Sensor Placement Considerations

**Neck-Mounted Collar:**
- Accelerometer measures neck orientation relative to gravity
- Gyroscope measures angular velocity of head/neck movements
- Z-axis (Rza) critically important for posture detection:
  - Standing: Neck vertical → Rza ≈ 9.8 m/s² (gravity)
  - Lying: Neck horizontal → Rza ≈ 0-2 m/s²
  - Feeding: Neck tilted down → Rza ≈ 3-5 m/s²

### Data Sources

1. **Peer-Reviewed Research:**
   - Accelerometer-based behavior classification in dairy cattle (Robert et al., 2009)
   - Validation of automated behavior monitoring systems (Borchers et al., 2016)
   - Rumination detection using neck-mounted sensors (Schirmann et al., 2009)
   - Walking gait analysis in cattle (Van Nuffel et al., 2015)

2. **Manufacturer Specifications:**
   - Typical accelerometer range: ±16g (±156.8 m/s²)
   - Typical gyroscope range: ±2000°/s
   - Temperature sensor accuracy: ±0.1°C

3. **Field Observations:**
   - Typical lying time: 10-14 hours/day
   - Ruminating time: 7-9 hours/day (usually while lying)
   - Feeding time: 3-5 hours/day in multiple bouts
   - Standing idle: 2-3 hours/day

---

## Sensor Coordinate System

### Axis Definitions (Neck-Mounted)

```
Animal facing forward, standing upright:

       Head
        ↑
        | +Y (Mya - lateral, side-to-side)
        |
        |_____ +X (Fxa - forward-backward)
       /
      / +Z (Rza - vertical, up-down)
     ↓
    Body

Angular Velocities (Gyroscope):
- Sxg (Roll): Rotation around X-axis (tilting head side-to-side)
- Lyg (Pitch): Rotation around Y-axis (nodding head up-down)
- Dzg (Yaw): Rotation around Z-axis (turning head left-right)
```

### Key Orientation Values

| Posture | Rza Value (m/s²) | Interpretation |
|---------|------------------|----------------|
| Standing upright | ~9.8 | Neck vertical, Z-axis aligned with gravity |
| Feeding (head down) | ~3-5 | Neck tilted ~45-60° downward |
| Lying (side) | ~0-2 | Neck horizontal, minimal Z-axis gravity component |
| Walking | ~6-10 (variable) | Generally upright with vertical oscillations |

---

## Behavior Definitions

### 1. LYING

**Description:** Animal resting in lateral recumbency (lying on side), minimal movement, horizontal body orientation.

**Typical Duration:** 30-120 minute bouts, 10-14 hours total per day

**Key Indicators:**
- **Rza < 4 m/s²** (horizontal neck orientation - DIAGNOSTIC)
- Very low angular velocity (std < 50°/s)
- Minimal movement variance
- Slightly lower body temperature

**Biological Context:**
- Essential for rest and rumination
- Cattle spend 50-60% of day lying
- Optimal for digestive processes
- Reduced metabolic rate

---

### 2. STANDING

**Description:** Animal standing still or with minimal movement, alert or resting while upright.

**Typical Duration:** Variable, from seconds to hours

**Key Indicators:**
- **Rza > 7 m/s²** (vertical neck orientation - DIAGNOSTIC)
- Low to moderate movement variance
- Random, non-rhythmic head movements
- Normal baseline temperature

**Biological Context:**
- Transitional state between activities
- Social interaction posture
- Alert observation of environment
- Thermoregulation (standing to cool, lying to warm)

---

### 3. WALKING

**Description:** Active locomotion with characteristic rhythmic gait pattern.

**Typical Duration:** 1-10 minute bouts, 2-3 hours total per day

**Key Indicators:**
- **Rhythmic patterns at 1-2 Hz** (DIAGNOSTIC)
- High acceleration variance (std > 3 m/s²)
- Periodic oscillations in all axes
- Generally upright posture (Rza ~8.5)
- Slightly elevated temperature

**Biological Context:**
- Movement between locations (feed, water, shade)
- Grazing on pasture
- Increased metabolic heat production
- Distinct from standing by rhythmic component

---

### 4. RUMINATING

**Description:** Regurgitation and re-chewing of cud, highly characteristic rhythmic jaw movements.

**Typical Duration:** 5-60 minute bouts, 7-9 hours total per day

**Key Indicators:**
- **Lyg rhythmic at 1-1.5 Hz (60-90 cycles/min)** (DIAGNOSTIC)
- **Dzg rhythmic at 1-1.5 Hz** (DIAGNOSTIC)
- Can be lying or standing (Rza variable)
- Low body movement
- Distinct frequency signature

**Biological Context:**
- Essential digestive process for ruminants
- Usually occurs while lying (70-80% of time)
- Each chew cycle: 60-70 jaw movements
- Indicator of good digestive health
- **Most distinctive behavior pattern in sensor data**

**Frequency Analysis:**
- Jaw cycle: 60-90 cycles per minute
- Converted to Hz: 1.0-1.5 Hz
- Appears in both Lyg (vertical jaw motion) and Dzg (lateral chewing)
- Very regular, sinusoidal pattern

---

### 5. FEEDING

**Description:** Active eating or grazing with head in lowered position, repetitive reaching and pulling motions.

**Typical Duration:** 10-40 minute bouts, 3-5 hours total per day

**Key Indicators:**
- **Rza ~3-4 m/s² (head-down position)** (DIAGNOSTIC)
- **Lyg negative bias** (head tilted downward)
- Moderate movement variance
- Semi-rhythmic but irregular patterns
- Can have brief pauses for chewing

**Biological Context:**
- Grazing or eating from trough/feeder
- Head lowered 45-60° from vertical
- Bite-chew-swallow cycles (less regular than ruminating)
- May alternate with short ruminating periods
- Social synchronization (herd feeding together)

---

### 6. STRESS

**Description:** Agitated, anxious, or distressed behavior with erratic, unpredictable movements.

**Typical Duration:** Variable, from minutes to hours depending on stressor

**Key Indicators:**
- **Very high variance across ALL channels** (DIAGNOSTIC)
- Angular velocity std > 250°/s
- No rhythmic patterns (chaotic)
- Elevated body temperature (38.8-39.5°C)
- Rapid, unpredictable movements

**Biological Context:**
- Response to fear, pain, social conflict, heat stress
- Physiological stress response (elevated cortisol)
- Increased heart rate and respiration
- May include pacing, head tossing, tail swishing
- Health indicator - prolonged stress impacts welfare

**Stressors:**
- Environmental (heat, cold, flies)
- Social (aggression, isolation)
- Physical (pain, illness)
- Management (handling, transportation)

---

## Parameter Rationale by Behavior

### LYING - Detailed Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Fxa** | mean=0, std=0.8 | Minimal forward-backward movement while resting |
| **Mya** | mean=0, std=0.7 | Minimal lateral movement, slight shifting |
| **Rza** | mean=0.5, std=1.2 | **CRITICAL:** Horizontal orientation, neck not vertical |
| **Sxg** | std=25 | Small rolling adjustments of head position |
| **Lyg** | std=30 | Occasional head lifts or adjustments |
| **Dzg** | std=35 | Small head rotations for environmental awareness |
| **temp** | 38.5°C | Slightly lower than standing due to reduced metabolic rate |

**Key Distinction:** Rza < 4 m/s² separates lying from all upright behaviors.

---

### STANDING - Detailed Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Fxa** | std=1.2 | Small weight shifts, postural adjustments |
| **Mya** | std=1.1 | Minimal lateral sway |
| **Rza** | mean=9.0, std=1.5 | **CRITICAL:** Near-vertical orientation (~gravity) |
| **Sxg** | std=40 | Moderate head movements for scanning |
| **Lyg** | std=50 | Head up/down movements while alert |
| **Dzg** | std=55 | Head rotations for environmental awareness |
| **temp** | 38.6°C | Normal baseline temperature |

**Key Distinction:** Rza > 7 m/s² indicates upright posture; low variance separates from walking.

---

### WALKING - Detailed Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Fxa** | mean=0.5, std=3.5 | Forward motion with rhythmic gait oscillations |
| **Mya** | std=3.2 | Side-to-side sway from walking gait |
| **Rza** | mean=8.5, std=2.8 | Generally upright with vertical bouncing from steps |
| **Sxg** | std=120 | Rhythmic rolling motion from gait cycle |
| **Lyg** | std=140 | Rhythmic pitch changes from head movement |
| **Dzg** | std=110 | Forward directional movement |
| **temp** | 38.7°C | Elevated 0.1-0.3°C from physical activity |
| **Frequency** | 1.5 Hz | Gait cycle frequency (90 steps/min typical) |

**Key Distinction:** High variance + rhythmic 1-2 Hz component separates from standing.

---

### RUMINATING - Detailed Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Fxa** | std=0.9 | Minimal body movement during rumination |
| **Mya** | std=1.2 | Slight lateral movement from chewing |
| **Rza** | mean=5.0, std=3.5 | Variable (can be lying ~2 or standing ~9) |
| **Sxg** | std=35 | Minimal rolling, body stationary |
| **Lyg** | std=180 | **CRITICAL:** Rhythmic vertical jaw movements (150-300°/s range) |
| **Dzg** | std=130 | **CRITICAL:** Rhythmic lateral chewing motion (100-200°/s range) |
| **temp** | 38.65°C | Baseline to slightly elevated from digestion |
| **Frequency** | 1.25 Hz | Chewing frequency (75 cycles/min typical) |

**Key Distinction:** Unique rhythmic signature in Lyg + Dzg at 1-1.5 Hz is DIAGNOSTIC for rumination. This is the most reliably identifiable behavior pattern.

**Why This Pattern is Unique:**
- Jaw movement creates consistent up-down (Lyg) and side-to-side (Dzg) angular velocities
- Frequency is slower and more regular than walking
- Higher amplitude in angular velocity than lying/standing
- Pattern persists for extended periods (minutes to hours)

---

### FEEDING - Detailed Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Fxa** | std=2.5 | Forward reaching and pulling feed motions |
| **Mya** | std=2.3 | Lateral movements reaching for feed |
| **Rza** | mean=3.5, std=2.2 | **CRITICAL:** Head-down position (45-60° tilt) |
| **Sxg** | std=65 | Moderate rolling from head movements |
| **Lyg** | mean=-15, std=100 | **Negative bias** from downward head angle |
| **Dzg** | std=80 | Head rotations selecting and reaching feed |
| **temp** | 38.65°C | Baseline to slightly elevated |
| **Frequency** | 0.5 Hz | Semi-rhythmic bite-chew cycles (irregular) |

**Key Distinction:** Rza ~3-4 m/s² (lower than standing, higher than lying) + Lyg negative bias identifies head-down feeding posture. Less rhythmic than ruminating.

---

### STRESS - Detailed Rationale

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Fxa** | std=5.5 | Very high variance - pacing, restlessness |
| **Mya** | std=5.2 | Unpredictable lateral movements |
| **Rza** | std=4.5 | Highly variable - rapid posture changes |
| **Sxg** | std=280 | **VERY HIGH** - sudden rolling movements |
| **Lyg** | std=300 | **VERY HIGH** - rapid head movements |
| **Dzg** | std=290 | **VERY HIGH** - rapid head rotations |
| **temp** | mean=39.0, std=0.35 | Elevated from stress response |
| **Frequency** | None | No rhythmic pattern - chaos is the signature |

**Key Distinction:** Extremely high variance across all channels (especially angular velocities std > 250°/s) combined with elevated temperature. Absence of rhythm distinguishes from walking.

---

## Distinguishing Features Matrix

This table shows the PRIMARY features that distinguish each behavior:

| Behavior | Primary Indicator | Secondary Indicator | Tertiary Indicator |
|----------|-------------------|---------------------|-------------------|
| **Lying** | Rza < 4 m/s² | Angular vel. std < 50°/s | Low temp |
| **Standing** | Rza > 7 m/s² | Non-rhythmic | Moderate angular vel. |
| **Walking** | Rhythmic 1-2 Hz | Accel. std > 3 m/s² | Rza ~8.5 |
| **Ruminating** | Lyg+Dzg rhythmic 1-1.5 Hz | Low body movement | Duration > 5 min |
| **Feeding** | Rza ~3-4 m/s² | Lyg negative bias | Semi-rhythmic |
| **Stress** | Angular vel. std > 250°/s | High variance all channels | Elevated temp |

### Decision Tree for Classification

```
1. Check Rza (Z-axis acceleration):
   - Rza < 4 → LYING (unless high angular velocity → STRESS)
   - Rza > 7 → UPRIGHT (standing/walking/ruminating)
   - Rza 3-5 → Likely FEEDING

2. If UPRIGHT, check for rhythmic patterns:
   - Rhythmic at 1-2 Hz in acceleration → WALKING
   - Rhythmic at 1-1.5 Hz in Lyg+Dzg → RUMINATING
   - No rhythm, low variance → STANDING
   - No rhythm, high variance → STRESS

3. Check angular velocity variance:
   - Std > 250°/s → STRESS (overrides other indicators)
   
4. Check temperature:
   - > 39.0°C + high variance → STRESS
   - Normal + rhythmic Lyg/Dzg → RUMINATING
```

---

## Transition Patterns

Understanding how animals move between behaviors is critical for realistic data generation and validation.

### Lying ↔ Standing

**Lying to Standing:**
- Duration: 10-30 seconds
- Pattern: Gradual increase in Rza from ~0-2 to ~8-10 m/s²
- Stages: Push up with front legs → raise hindquarters → stabilize
- Angular velocities show brief spike during transition
- Most common after ruminating bout

**Standing to Lying:**
- Duration: 5-20 seconds
- Pattern: Gradual decrease in Rza from ~8-10 to ~0-2 m/s²
- Stages: Fold front legs → lower body → settle on side
- Brief high variance in angular velocity during lowering
- Often preceded by circling behavior

**Frequency:** 6-12 times per day

---

### Standing ↔ Walking

**Standing to Walking:**
- Duration: 2-5 seconds
- Pattern: Acceleration variance increases, rhythmic components emerge
- Smooth transition, no dramatic Rza change (both upright)
- Frequency components appear in FFT analysis

**Walking to Standing:**
- Duration: 2-5 seconds
- Pattern: Rhythmic components fade, variance decreases
- Gradual deceleration
- May include brief higher variance as animal stops

**Frequency:** 20-50 times per day (many short walking bouts)

---

### Any → Feeding

**Transition to Feeding:**
- Duration: 3-10 seconds
- Pattern: Rza decreases to ~3-4 m/s² (head lowering)
- Lyg shows negative shift (downward head angle)
- May occur from standing (common) or walking (approaching feed)
- Angular velocity increases as head moves to feed

**Feeding to Other:**
- Can transition directly to ruminating (begin chewing cud)
- Can lift head to standing (Rza increases to ~9)
- Walking away from feed bunk

**Frequency:** 10-20 feeding bouts per day

---

### Any → Ruminating

**Transition to Ruminating:**
- Duration: 2-5 seconds
- Pattern: Rhythmic Lyg/Dzg pattern emerges at 1-1.5 Hz
- Body movement decreases
- Can begin from lying (most common) or standing
- Often follows feeding bout (30-60 min after eating)

**Ruminating to Other:**
- Pattern gradually fades over 5-10 seconds
- May transition to lying (if was standing)
- May transition to standing (if was lying)
- Brief pause between ruminating bouts is normal

**Frequency:** 15-20 ruminating bouts per day

---

### Any → Stress

**Transition to Stress:**
- Duration: 1-3 seconds (RAPID onset)
- Pattern: Sudden increase in variance across all channels
- Angular velocities spike to > 250°/s std
- Temperature may rise (but this takes minutes)
- Triggered by external stressor

**Stress to Other:**
- Gradual return to normal as stressor removed
- May take 5-30 minutes for full recovery
- Temperature returns to baseline slowly
- Often transitions to standing (alert) or walking (escape)

**Frequency:** Ideally rare (< 5 times per day), indicates welfare issue if frequent

---

## Validation Rules

### Physical Plausibility Constraints

```python
# Acceleration limits (beyond this suggests sensor error)
acceleration_absolute_max = 15.0  # m/s² (beyond typical animal movement)

# Angular velocity limits
angular_velocity_absolute_max = 800.0  # °/s (beyond normal head movements)

# Temperature limits
temperature_min = 36.0  # °C (hypothermia threshold)
temperature_max = 42.0  # °C (life-threatening hyperthermia)
temperature_normal_range = (38.3, 39.1)  # °C
```

### Behavior-Specific Validation

**Lying Detection:**
```python
def validate_lying(data):
    return (
        data['Rza'] < 4.0 and
        data['angular_velocity_std'] < 50.0 and
        data['temperature'] < 39.0
    )
```

**Standing Detection:**
```python
def validate_standing(data):
    return (
        data['Rza'] > 7.0 and
        data['Rza'] < 11.0 and
        not has_rhythmic_pattern(data)
    )
```

**Walking Detection:**
```python
def validate_walking(data):
    return (
        data['acceleration_std'] > 3.0 and
        has_frequency_component(data, 1.0, 2.0) and
        data['Rza'] > 5.0
    )
```

**Ruminating Detection:**
```python
def validate_ruminating(data):
    return (
        has_frequency_component(data['Lyg'], 1.0, 1.5) and
        has_frequency_component(data['Dzg'], 1.0, 1.5) and
        data['body_movement_std'] < 3.0 and
        data['bout_duration'] > 5 * 60  # 5 minutes minimum
    )
```

**Feeding Detection:**
```python
def validate_feeding(data):
    return (
        data['Rza'] > 2.0 and
        data['Rza'] < 6.0 and
        data['Lyg_mean'] < -5.0 and  # Negative bias
        data['bout_duration'] > 5 * 60  # 5 minutes minimum
    )
```

**Stress Detection:**
```python
def validate_stress(data):
    return (
        data['angular_velocity_std'] > 250.0 or
        (data['acceleration_std'] > 5.0 and 
         data['temperature'] > 39.0 and
         not has_rhythmic_pattern(data))
    )
```

### Inter-Behavior Validation

**Mutual Exclusivity Rules:**
- Cannot be lying AND standing simultaneously (Rza conflict)
- Cannot be walking AND lying simultaneously (movement conflict)
- Can be ruminating AND lying (common combination)
- Can be ruminating AND standing (less common but valid)
- Stress can override other classifications if extreme variance

**Temporal Validation:**
- Minimum bout duration: 1 minute (sampling rate limit)
- Lying bouts typically > 30 minutes
- Ruminating bouts typically > 5 minutes
- Walking bouts typically 1-10 minutes
- Feeding bouts typically 5-30 minutes
- Maximum transitions per hour: ~20 (more suggests classification error)

### Statistical Validation

**Z-Score Anomaly Detection:**
```python
def detect_anomaly(value, mean, std):
    z_score = abs(value - mean) / std
    return z_score > 4.0  # Beyond 4 standard deviations
```

**Frequency Component Validation:**
```python
def validate_frequency(fft_result, expected_hz):
    peak_freq = get_dominant_frequency(fft_result)
    tolerance = 0.3  # Hz
    return abs(peak_freq - expected_hz) < tolerance
```

---

## References & Sources

### Peer-Reviewed Literature

1. **Robert, B., White, B. J., Renter, D. G., & Larson, R. L. (2009).** "Evaluation of three-dimensional accelerometers to monitor and classify behavior patterns in cattle." *Computers and Electronics in Agriculture, 67*(1-2), 80-84.
   - Established accelerometer ranges for cattle behaviors
   - Validated Z-axis for posture detection

2. **Borchers, M. R., Chang, Y. M., Tsai, I. C., Wadsworth, B. A., & Bewley, J. M. (2016).** "A validation of technologies monitoring dairy cow feeding, ruminating, and lying behaviors." *Journal of Dairy Science, 99*(9), 7458-7466.
   - Confirmed ruminating frequency ranges (60-90 cycles/min)
   - Validated lying time patterns

3. **Schirmann, K., von Keyserlingk, M. A., Weary, D. M., Veira, D. M., & Heuwieser, W. (2009).** "Technical note: Validation of a system for monitoring rumination in dairy cows." *Journal of Dairy Science, 92*(12), 6052-6055.
   - Established gyroscope patterns for rumination detection
   - Confirmed pitch and yaw angular velocity signatures

4. **Van Nuffel, A., Zwertvaegher, I., Van Weyenberg, S., Pastell, M., Thorup, V. M., Bahr, C., ... & Tuyttens, F. A. M. (2015).** "Lameness detection in dairy cows: Part 2. Use of sensors to automatically register changes in locomotion or behavior." *Animals, 5*(3), 861-885.
   - Walking gait frequency analysis (1-2 Hz)
   - Acceleration patterns during locomotion

5. **Kilgour, R. J., Uetake, K., Ishiwata, T., & Melville, G. J. (2012).** "The behaviour of beef cattle at pasture." *Applied Animal Behaviour Science, 138*(1-2), 12-17.
   - Time budgets for different behaviors
   - Transition patterns between activities

### Technical References

6. **Alvarenga, F. A. P., Borges, I., Palkovic, L., Rodina, J., Oddy, V. H., & Dobos, R. C. (2016).** "Using a three-axis accelerometer to identify and classify sheep behaviour at pasture." *Applied Animal Behaviour Science, 181*, 91-99.
   - Livestock accelerometer methodology
   - Signal processing techniques

7. **Martiskainen, P., Järvinen, M., Skön, J. P., Tiirikainen, J., Kolehmainen, M., & Mononen, J. (2009).** "Cow behaviour pattern recognition using a three-dimensional accelerometer and support vector machines." *Applied Animal Behaviour Science, 119*(1-2), 32-38.
   - Machine learning classification approaches
   - Feature extraction from sensor data

### Physiological References

8. **Burfeind, O., Suthar, V. S., Voigtsberger, R., Bonk, S., & Heuwieser, W. (2014).** "Body temperature in early postpartum dairy cows." *Theriogenology, 82*(1), 121-131.
   - Normal temperature ranges
   - Diurnal temperature patterns

9. **West, J. W. (2003).** "Effects of heat-stress on production in dairy cattle." *Journal of Dairy Science, 86*(6), 2131-2144.
   - Temperature elevation during stress
   - Heat stress indicators

10. **Dado, R. G., & Allen, M. S. (1994).** "Variation in and relationships among feeding, chewing, and drinking variables for lactating dairy cows." *Journal of Dairy Science, 77*(1), 132-144.
    - Feeding and ruminating time budgets
    - Chewing patterns and frequencies

### Industry Standards

11. **Precision Livestock Farming '15** - Conference proceedings on sensor-based livestock monitoring
12. **American Dairy Science Association** - Guidelines for behavior monitoring systems
13. **International Society for Applied Ethology** - Animal welfare and behavior assessment standards

---

## Assumptions & Limitations

### Assumptions Made

1. **Sensor Placement:** Assumes collar is properly fitted and positioned on neck (not loose or rotated)
2. **Calibration:** Assumes sensors are properly calibrated before deployment
3. **Sampling Rate:** 1-minute sampling is sufficient to capture behavior patterns (may miss very brief events)
4. **Individual Variation:** Parameters represent population averages; individual animals may vary ±20%
5. **Environmental Conditions:** Assumes typical conditions (not extreme weather, proper facility design)
6. **Health Status:** Parameters based on healthy animals; illness may alter patterns

### Known Limitations

1. **Transition Ambiguity:** Brief transitions between behaviors may be difficult to classify accurately
2. **Combined Behaviors:** Ruminating while lying creates mixed signal (addressed by parameter ranges)
3. **Brief Events:** Events < 1 minute may be missed or misclassified due to sampling rate
4. **Temperature Lag:** Body temperature changes lag behind behavior changes (thermal inertia)
5. **Individual Differences:** Breed, age, size affect exact parameter values (±10-20% variation expected)

### Future Improvements

1. **Breed-Specific Parameters:** Develop separate parameter sets for different breeds
2. **Age Adjustment:** Scale parameters for calves, heifers, mature cows
3. **Environmental Corrections:** Adjust temperature thresholds for ambient conditions
4. **Individual Calibration:** Learn individual baselines for improved accuracy
5. **Context Integration:** Incorporate time of day, feeding schedule, social factors

---

## Appendix: Quick Reference Table

### Complete Parameter Summary

| Behavior | Rza | Accel Std | Angular Std | Temp | Rhythm | Key Feature |
|----------|-----|-----------|-------------|------|--------|-------------|
| Lying | <4 | <2 | <50 | 38.5 | No | Horizontal orientation |
| Standing | >7 | ~1-2 | ~50 | 38.6 | No | Vertical orientation |
| Walking | ~8.5 | >3 | 100-150 | 38.7 | 1-2 Hz | Rhythmic gait |
| Ruminating | 2-10 | ~1 | Lyg/Dzg>100 | 38.65 | 1-1.5 Hz | Chewing rhythm |
| Feeding | 3-5 | ~2-3 | ~70 | 38.65 | No | Head down |
| Stress | Variable | >5 | >250 | >39.0 | No | High variance |

---

**Document End**

*For technical implementation details, see: `config/behavior_patterns.py`*

*For questions or updates, contact the Artemis Health data science team.*
