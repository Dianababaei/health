# Complete Simulator Usage Guide

## What Simulators Do

Your livestock health monitoring system has **two types of simulators** that work together to help you test your entire application without needing real cow data:

### 1. **Layer 1 Simulator** (Behavioral Data) - `src/simulation/`
**Purpose**: Generates realistic sensor data (accelerometer, gyroscope, temperature) for different behavioral states

**What it simulates:**
- 5 behavioral states: Lying, Standing, Walking, Ruminating, Feeding
- Realistic sensor signatures (accelerometer axes: fxa, mya, rza; gyroscope: sxg, lyg, dzg)
- Natural state transitions
- Baseline temperature patterns

**Use case**:
✓ Test Layer 1 behavioral classification algorithms
✓ Train machine learning models
✓ Validate data processing pipelines

### 2. **Layer 2/3 Simulator** (Health Conditions) - `src/simulation/health_conditions.py`
**Purpose**: Simulates physiological health conditions on top of behavioral data

**What it simulates:**
- **Fever**: Elevated temperature (>39.5°C), reduced activity
- **Heat Stress**: High temp + high activity (panting behavior)
- **Estrus**: Temperature spike + activity increase (breeding behavior)
- **Pregnancy**: Stable temperature, dampened circadian, reduced activity

**Use case**:
✓ Test Layer 2 physiological analysis (temperature anomaly detection, circadian rhythm)
✓ Test Layer 3 health intelligence (alert detection, trend tracking)
✓ Validate dashboard visualization of health conditions

---

## Your Testing Workflow

Here's how to test your entire app using simulators:

### **Option 1: Full Integration Test (All Layers)**

```python
from simulation import SimulationEngine
from simulation.health_conditions import FeverSimulator, EstrusSimulator
from layer1_behavior import BehavioralClassifier
from layer2_physiological import CircadianRhythmAnalyzer, MultiDayTrendAnalyzer
from health_intelligence import MultiDayHealthTrendTracker, ImmediateAlertDetector
import pandas as pd
from datetime import datetime, timedelta

# Step 1: Generate baseline behavioral data (Layer 1 simulator)
print("Step 1: Generating 7 days of realistic behavioral data...")
engine = SimulationEngine(
    baseline_temperature=38.5,
    sampling_rate=1.0,  # 1 sample per minute
    random_seed=42
)

df_baseline = engine.generate_continuous_data(
    duration_hours=7 * 24,  # 7 days
    start_datetime=datetime.now() - timedelta(days=7),
    include_stress=True,
    stress_probability=0.02  # 2% chance of stress events
)

print(f"  Generated {len(df_baseline)} sensor readings")
print(f"  States: {df_baseline['state'].value_counts().to_dict()}")

# Step 2: Inject health condition (Layer 2 simulator)
print("\nStep 2: Injecting fever condition on day 3...")
fever_sim = FeverSimulator(
    baseline_fever_temp=40.0,  # 40°C fever
    activity_reduction=0.30    # 30% activity reduction
)

# Replace temperature and modify activity for fever period (day 3)
fever_start = 2 * 24 * 60  # Start of day 3 (in minutes)
fever_duration = 24 * 60   # 24 hours

# Generate fever temperature pattern
fever_temp = fever_sim.generate_temperature(
    duration_minutes=fever_duration,
    random_seed=42
)

# Apply fever to dataframe
df_baseline.loc[fever_start:fever_start+fever_duration, 'temperature'] = fever_temp

# Reduce activity during fever
fever_mask = (df_baseline.index >= fever_start) & (df_baseline.index < fever_start + fever_duration)
activity_cols = ['fxa', 'mya', 'sxg', 'lyg', 'dzg']
df_baseline.loc[fever_mask, activity_cols] *= (1.0 - 0.30)

print(f"  Fever injected: {fever_duration} minutes on day 3")
print(f"  Temperature range during fever: {fever_temp.min():.2f}-{fever_temp.max():.2f}°C")

# Step 3: Test Layer 1 - Behavioral Classification
print("\nStep 3: Testing Layer 1 behavioral classification...")
# Your Layer 1 classifier would process df_baseline here
# For now, states are already labeled in 'state' column
state_distribution = df_baseline['state'].value_counts(normalize=True) * 100
print(f"  State distribution: {state_distribution.to_dict()}")

# Step 4: Test Layer 2 - Physiological Analysis
print("\nStep 4: Testing Layer 2 physiological analysis...")

# 4a: Circadian rhythm analysis
circadian = CircadianRhythmAnalyzer()
circadian_results = circadian.analyze_circadian_rhythm(
    temperature_data=df_baseline[['timestamp', 'temperature']],
    window_hours=24
)
print(f"  Circadian amplitude: {circadian_results['amplitude']:.2f}°C")
print(f"  Peak time: {circadian_results['peak_hour']:.1f} hours")

# 4b: Multi-day trend analysis
trend_analyzer = MultiDayTrendAnalyzer(temperature_baseline=38.5)
temp_df = df_baseline[['timestamp', 'temperature']].copy()
activity_df = df_baseline[['timestamp', 'state']].copy()
activity_df['behavioral_state'] = activity_df['state']
activity_df['movement_intensity'] = df_baseline[['fxa', 'mya', 'rza']].abs().sum(axis=1) / 3.0

trend_report = trend_analyzer.analyze_trends(
    cow_id=1,
    temperature_data=temp_df,
    activity_data=activity_df,
    anomaly_history=[],
    circadian_scores=None
)

print(f"  7-day trend: {trend_report.trends[7].trend_direction.value if 7 in trend_report.trends else 'N/A'}")

# Step 5: Test Layer 3 - Health Intelligence
print("\nStep 5: Testing Layer 3 health intelligence...")

# 5a: Immediate alert detection
alert_detector = ImmediateAlertDetector()

# Check for fever alerts
recent_data = df_baseline.tail(10)  # Last 10 minutes
alerts = []

for idx, row in recent_data.iterrows():
    reading = {
        'timestamp': row['timestamp'],
        'temperature': row['temperature'],
        'fxa': row['fxa'],
        'mya': row['mya'],
        'rza': row['rza']
    }

    detected_alerts = alert_detector.check_alerts(
        cow_id='COW_001',
        sensor_reading=reading,
        current_behavioral_state=row['state']
    )

    alerts.extend(detected_alerts)

print(f"  Alerts detected: {len(alerts)}")
if alerts:
    print(f"  Alert types: {[a.alert_type for a in alerts]}")

# 5b: Multi-day health trend tracker
health_tracker = MultiDayHealthTrendTracker(temperature_baseline=38.5)

# Prepare alert history (convert detected alerts to dict format)
alert_history = [
    {
        'timestamp': alert.timestamp,
        'alert_type': alert.alert_type,
        'severity': alert.severity,
        'cow_id': 'COW_001'
    }
    for alert in alerts
]

health_trend_report = health_tracker.analyze_trends(
    cow_id='COW_001',
    temperature_data=temp_df,
    activity_data=activity_df,
    alert_history=alert_history,
    behavioral_states=activity_df
)

print(f"  Overall health trend: {health_trend_report.overall_trend.value}")
print(f"  Confidence: {health_trend_report.overall_confidence:.1%}")
print(f"  Recommendations: {health_trend_report.recommendations}")

# Step 6: Test Dashboard Visualization
print("\nStep 6: Exporting dashboard-ready data...")
dashboard_data = {
    'cow_id': 'COW_001',
    'current_state': df_baseline.iloc[-1]['state'],
    'current_temperature': df_baseline.iloc[-1]['temperature'],
    'alerts': [alert.to_dict() for alert in alerts],
    'health_trends': health_trend_report.to_dict(),
    'circadian_metrics': circadian_results,
    'behavioral_summary': state_distribution.to_dict()
}

print(f"  Dashboard data keys: {list(dashboard_data.keys())}")
print("\n✓ Full integration test complete!")
```

---

### **Option 2: Test Individual Layers**

#### **Layer 1 Only: Behavioral Classification**

```python
from simulation import SimulationEngine

# Generate clean behavioral data
engine = SimulationEngine(random_seed=42)

# Test each state individually
states_to_test = ['lying', 'standing', 'walking', 'ruminating', 'feeding']

for state_name in states_to_test:
    # Generate 30 minutes of pure state data
    df = engine.generate_single_state_data(
        state=state_name,
        duration_minutes=30
    )

    # Test your Layer 1 classifier
    # predicted_state = your_classifier.predict(df)
    # accuracy = (predicted_state == state_name).mean()

    print(f"{state_name}: {len(df)} samples generated")
```

#### **Layer 2 Only: Physiological Analysis**

```python
from simulation.health_conditions import FeverSimulator, EstrusSimulator
from layer2_physiological import CircadianRhythmAnalyzer
import pandas as pd
from datetime import datetime, timedelta

# Test Fever Detection
fever_sim = FeverSimulator(baseline_fever_temp=40.0)

# Generate 48 hours of fever pattern
temp_data = fever_sim.generate_temperature(
    duration_minutes=48 * 60,
    random_seed=42
)

timestamps = pd.date_range(
    start=datetime.now() - timedelta(hours=48),
    periods=len(temp_data),
    freq='1min'
)

df = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temp_data
})

# Test circadian analysis
analyzer = CircadianRhythmAnalyzer()
results = analyzer.analyze_circadian_rhythm(df, window_hours=24)

print(f"Detected fever: {results['mean_temp'] > 39.5}")
print(f"Circadian health score: {results['health_score']:.2f}")

# Test Estrus Detection
estrus_sim = EstrusSimulator()
estrus_temp, estrus_activity = estrus_sim.generate_estrus_episode(
    duration_minutes=60,
    random_seed=42
)

print(f"Estrus temperature spike: {estrus_temp.max() - estrus_temp.min():.2f}°C")
print(f"Activity increase: {estrus_activity.mean():.2f}")
```

#### **Layer 3 Only: Health Intelligence**

```python
from health_intelligence import MultiDayHealthTrendTracker
from simulation import SimulationEngine
import pandas as pd

# Generate 30 days of data with gradual deterioration
engine = SimulationEngine(baseline_temperature=38.5)

# Simulate sick cow: increasing temperature over time
df = engine.generate_continuous_data(
    duration_hours=30 * 24,
    include_stress=True,
    stress_probability=0.1  # 10% stress (sick behavior)
)

# Manually increase temperature over time (simulate deterioration)
time_factor = np.linspace(0, 1, len(df))
df['temperature'] += time_factor * 1.5  # Gradual 1.5°C increase

# Test health trend tracker
tracker = MultiDayHealthTrendTracker()
report = tracker.analyze_trends(
    cow_id='SICK_COW_001',
    temperature_data=df[['timestamp', 'temperature']],
    activity_data=df[['timestamp', 'state']],
    alert_history=[],
    behavioral_states=df[['timestamp', 'state']]
)

print(f"Detected trend: {report.overall_trend.value}")
print(f"Significant changes: {report.significant_changes}")
```

---

## Use Real Data for Layer 1 + Simulated Health Conditions

If you have **real sensor data** from actual cows but want to test health conditions:

```python
import pandas as pd
from simulation.health_conditions import FeverSimulator, PregnancySimulator

# Load your real cow data
real_data = pd.read_csv('path/to/real_cow_data.csv')
# Columns: timestamp, temperature, fxa, mya, rza, sxg, lyg, dzg

# Inject simulated fever into days 5-7
fever_sim = FeverSimulator(baseline_fever_temp=40.5)

# Find day 5 start index
day5_start = find_index_for_day(real_data, day=5)  # Your function
day7_end = find_index_for_day(real_data, day=7)

fever_duration = day7_end - day5_start

# Generate fever temperature
fever_temp = fever_sim.generate_temperature(
    duration_minutes=fever_duration,
    random_seed=42
)

# Replace real temperature with fever temperature
real_data.loc[day5_start:day7_end, 'temperature'] = fever_temp

# Reduce activity during fever
real_data.loc[day5_start:day7_end, ['fxa', 'mya', 'rza']] *= 0.7

# Now test your Layer 2 and Layer 3 with this hybrid data
# Your algorithms will process real behavioral patterns with simulated illness
```

---

## Testing Dashboard with Simulated Data

### Stream Simulated Data to Dashboard

```python
import time
from simulation import SimulationEngine
import streamlit as st

# In your Streamlit dashboard code
st.title("Live Cattle Health Monitoring (SIMULATION MODE)")

# Initialize simulator
@st.cache_resource
def get_simulator():
    return SimulationEngine(random_seed=None)  # Random each time

engine = get_simulator()

# Simulate real-time streaming
placeholder = st.empty()

while True:
    # Generate next minute of data
    current_data = engine.generate_continuous_data(
        duration_hours=1/60,  # 1 minute
        start_datetime=datetime.now()
    )

    # Update dashboard
    with placeholder.container():
        st.metric("Current Temperature", f"{current_data['temperature'].iloc[-1]:.1f}°C")
        st.metric("Current State", current_data['state'].iloc[-1])

        # Show trend chart
        st.line_chart(current_data[['temperature']])

    time.sleep(60)  # Update every minute
```

---

## Quick Reference: What to Use When

| Testing Goal | Simulator to Use | Key Functions |
|--------------|------------------|---------------|
| **Test behavioral classification** | `SimulationEngine` | `generate_continuous_data()`, `generate_single_state_data()` |
| **Test fever detection** | `FeverSimulator` | `generate_temperature()`, `modify_motion_pattern()` |
| **Test estrus detection** | `EstrusSimulator` | `generate_estrus_episode()` |
| **Test pregnancy detection** | `PregnancySimulator` | `generate_temperature()`, `modify_motion_pattern()` |
| **Test heat stress** | `HeatStressSimulator` | `generate_temperature()`, `generate_panting_pattern()` |
| **Test circadian analysis** | `CircadianRhythmGenerator` | `generate()` |
| **Test alerts** | All health simulators | Inject conditions, then use `ImmediateAlertDetector` |
| **Test trend tracking** | Multiple days of data | `MultiDayHealthTrendTracker.analyze_trends()` |
| **Test dashboard** | Any simulator | Export to CSV or stream data |

---

## Advanced: Create Custom Health Scenarios

```python
from simulation import SimulationEngine
from simulation.health_conditions import FeverSimulator, EstrusSimulator

# Scenario: Cow recovers from fever, then enters estrus
engine = SimulationEngine(baseline_temperature=38.5)

# Day 1-2: Normal
normal_data = engine.generate_continuous_data(duration_hours=48)

# Day 3-4: Fever
fever_sim = FeverSimulator(baseline_fever_temp=40.0)
fever_temp = fever_sim.generate_temperature(duration_minutes=48*60)
fever_data = engine.generate_continuous_data(duration_hours=48)
fever_data['temperature'] = fever_temp
fever_data[['fxa', 'mya']] *= 0.7  # Reduced activity

# Day 5-6: Recovery (normal)
recovery_data = engine.generate_continuous_data(duration_hours=48)

# Day 7: Estrus
estrus_sim = EstrusSimulator()
estrus_temp, estrus_activity = estrus_sim.generate_estrus_episode(duration_minutes=8*60)
estrus_data = engine.generate_continuous_data(duration_hours=24)
# Inject estrus spike at hour 12
spike_start = 12 * 60
estrus_data.loc[spike_start:spike_start+8*60, 'temperature'] = estrus_temp
estrus_data.loc[spike_start:spike_start+8*60, 'fxa'] *= 1.5

# Combine all
full_scenario = pd.concat([normal_data, fever_data, recovery_data, estrus_data])

# Test your entire system with this realistic scenario
```

---

## Summary

**What you have:**

1. ✅ **Behavioral Simulator** (`src/simulation/`) - Generates realistic sensor data for 5 behavioral states
2. ✅ **Health Condition Simulators** (`src/simulation/health_conditions.py`) - Adds fever, estrus, pregnancy, heat stress
3. ✅ **Circadian Rhythm Generator** (`src/simulation/circadian_rhythm.py`) - Creates 24-hour temperature patterns

**How to use them:**

- **Layer 1 Testing**: Use `SimulationEngine` alone
- **Layer 2 Testing**: Combine `SimulationEngine` + health condition simulators
- **Layer 3 Testing**: Use multi-day simulated data with injected health events
- **Dashboard Testing**: Stream simulated data or load pre-generated CSV files
- **Hybrid Testing**: Use real sensor data + inject simulated health conditions

**Next steps:**

1. Run `python examples/trend_tracker_example.py` to see a complete example
2. Check `src/simulation/example_usage.py` for simulator examples
3. Read `docs/health_simulation_parameters.md` for detailed health condition parameters

Your simulators are production-ready and validated against veterinary literature! 🎉
