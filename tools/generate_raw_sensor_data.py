"""
Generate RAW sensor data ONLY - for testing end-to-end dashboard processing

This script generates ONLY the raw sensor CSV without:
- NO behavioral states (state column)
- NO alerts.json
- NO metadata.json
- NO processing

The dashboard will process this raw data through all 3 layers.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from simulation import SimulationEngine
from simulation.health_conditions import FeverSimulator

print("="*70)
print("RAW SENSOR DATA GENERATOR (No Processing)")
print("="*70)

# Configuration - Single cow mode
cow_id = "COW_001"
duration_days = 21  # Extended to 21 days for comprehensive testing
baseline_temp = 38.5

# Multiple alert scenarios for QA testing
alert_scenarios = [
    # Scenario 1: Mild fever (Days 3-4) - Temp 39.5-39.8°C
    {'type': 'mild_fever', 'start_day': 3, 'duration_days': 2, 'temp': 39.6, 'motion_level': 0.08, 'severity': 'warning'},

    # Scenario 2: Estrus/Heat (Day 6) - Elevated temp + high activity
    {'type': 'estrus', 'start_day': 6, 'duration_days': 0.5, 'temp': 38.8, 'motion_level': 0.75, 'severity': 'info'},

    # Scenario 3: Heat stress (Day 9) - Moderate temp + very high activity
    {'type': 'heat_stress', 'start_day': 9, 'duration_days': 1, 'temp': 39.3, 'motion_level': 0.70, 'severity': 'warning'},

    # Scenario 4: Moderate fever (Days 11-12) - Temp 39.8-40.5°C
    {'type': 'moderate_fever', 'start_day': 11, 'duration_days': 2, 'temp': 40.0, 'motion_level': 0.05, 'severity': 'critical'},

    # Scenario 5: Prolonged inactivity (Days 14-15) - Normal temp, very low motion
    {'type': 'inactivity', 'start_day': 14, 'duration_days': 2, 'temp': 38.5, 'motion_level': 0.02, 'severity': 'warning'},

    # Scenario 6: Pregnancy indicator (Day 17) - Slightly reduced activity
    {'type': 'pregnancy', 'start_day': 17, 'duration_days': 2, 'temp': 38.6, 'motion_level': 0.30, 'severity': 'info'},

    # Scenario 7: Severe fever (Days 19-20) - Temp >40.5°C
    {'type': 'severe_fever', 'start_day': 19, 'duration_days': 1.5, 'temp': 41.0, 'motion_level': 0.03, 'severity': 'critical'},
]

print(f"\nConfiguration:")
print(f"  Cow ID: {cow_id}")
print(f"  Duration: {duration_days} days")
print(f"  Baseline Temperature: {baseline_temp}°C")
print(f"  Alert Scenarios: {len(alert_scenarios)} different conditions")
for i, scenario in enumerate(alert_scenarios, 1):
    print(f"    {i}. {scenario['type']}: Day {scenario['start_day']}, {scenario['duration_days']} days, Temp {scenario['temp']}°C")

# Create output directory
output_dir = Path('data/raw_test')
output_dir.mkdir(parents=True, exist_ok=True)

# Keep only raw sensor columns (NO 'state' column)
raw_columns = ['timestamp', 'temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']

# Step 1: Generate baseline data
print(f"\n[1/3] Generating {duration_days} days of sensor data...")

engine = SimulationEngine(
    baseline_temperature=baseline_temp,
    sampling_rate=1.0  # 1 sample per minute
)

df = engine.generate_continuous_data(
    duration_hours=duration_days * 24
)

print(f"  [OK] Generated {len(df)} sensor readings")

# Step 2: Inject multiple alert scenarios
print(f"\n[2/3] Injecting {len(alert_scenarios)} alert scenarios...")

for i, scenario in enumerate(alert_scenarios, 1):
    scenario_type = scenario['type']
    start_day = scenario['start_day']
    duration_days = scenario['duration_days']
    target_temp = scenario['temp']
    motion_level = scenario['motion_level']

    # Calculate sample indices
    start_idx = int((start_day - 1) * 24 * 60)
    end_idx = int(start_idx + (duration_days * 24 * 60))
    N = end_idx - start_idx

    # Inject temperature based on scenario
    if scenario_type in ['moderate_fever', 'severe_fever']:
        # Use fever simulator for realistic high fever pattern (39.5-42.0°C)
        fever_sim = FeverSimulator(
            baseline_fever_temp=target_temp,
            activity_reduction=0.30
        )
        temp_data = fever_sim.generate_temperature(duration_minutes=N)
        df.loc[start_idx:end_idx-1, 'temperature'] = temp_data
    elif scenario_type == 'mild_fever':
        # Mild fever: sustained elevated temperature just above threshold (39.5-39.8°C)
        df.loc[start_idx:end_idx-1, 'temperature'] = np.random.uniform(39.5, 39.8, N)
    elif scenario_type == 'heat_stress':
        # Heat stress: moderately elevated temperature (39.0-39.5°C)
        df.loc[start_idx:end_idx-1, 'temperature'] = np.random.uniform(39.0, 39.4, N)
    elif scenario_type == 'estrus':
        # Estrus: slight temperature rise (0.3-0.6°C above baseline)
        df.loc[start_idx:end_idx-1, 'temperature'] = np.random.uniform(38.8, 39.1, N)
    elif scenario_type == 'pregnancy':
        # Pregnancy: slightly elevated baseline (0.1-0.3°C above normal)
        df.loc[start_idx:end_idx-1, 'temperature'] = np.random.normal(38.6, 0.1, N)
    else:
        # Normal temperature with slight variation
        df.loc[start_idx:end_idx-1, 'temperature'] = np.random.normal(target_temp, 0.1, N)

    # Inject motion patterns based on scenario type
    if scenario_type in ['heat_stress', 'estrus']:
        # High activity (estrus: 20-50% increase, heat stress: sustained high)
        df.loc[start_idx:end_idx-1, 'fxa'] = np.random.uniform(0.60, 0.80, N)
        df.loc[start_idx:end_idx-1, 'mya'] = np.random.uniform(0.60, 0.80, N)
        df.loc[start_idx:end_idx-1, 'rza'] = np.random.uniform(0.60, 0.80, N)
        df.loc[start_idx:end_idx-1, 'sxg'] = np.random.uniform(0.60, 0.80, N)
        df.loc[start_idx:end_idx-1, 'lyg'] = np.random.uniform(0.60, 0.80, N)
        df.loc[start_idx:end_idx-1, 'dzg'] = np.random.uniform(0.60, 0.80, N)
    elif scenario_type == 'inactivity':
        # Very low motion (below 0.05 for all axes)
        df.loc[start_idx:end_idx-1, 'fxa'] = np.random.uniform(0.01, 0.03, N)
        df.loc[start_idx:end_idx-1, 'mya'] = np.random.uniform(0.01, 0.03, N)
        df.loc[start_idx:end_idx-1, 'rza'] = np.random.uniform(0.01, 0.03, N)
        df.loc[start_idx:end_idx-1, 'sxg'] = np.random.uniform(0.01, 0.03, N)
        df.loc[start_idx:end_idx-1, 'lyg'] = np.random.uniform(0.01, 0.03, N)
        df.loc[start_idx:end_idx-1, 'dzg'] = np.random.uniform(0.01, 0.03, N)
    elif scenario_type == 'pregnancy':
        # Moderately reduced activity (20-30% decrease)
        df.loc[start_idx:end_idx-1, 'fxa'] = np.random.uniform(0.25, 0.35, N)
        df.loc[start_idx:end_idx-1, 'mya'] = np.random.uniform(0.25, 0.35, N)
        df.loc[start_idx:end_idx-1, 'rza'] = np.random.uniform(0.25, 0.35, N)
        df.loc[start_idx:end_idx-1, 'sxg'] = np.random.uniform(0.25, 0.35, N)
        df.loc[start_idx:end_idx-1, 'lyg'] = np.random.uniform(0.25, 0.35, N)
        df.loc[start_idx:end_idx-1, 'dzg'] = np.random.uniform(0.25, 0.35, N)
    else:
        # Low motion (fever scenarios - sickness behavior)
        base = motion_level
        df.loc[start_idx:end_idx-1, 'fxa'] = np.random.uniform(base, base + 0.02, N)
        df.loc[start_idx:end_idx-1, 'mya'] = np.random.uniform(base, base + 0.02, N)
        df.loc[start_idx:end_idx-1, 'rza'] = np.random.uniform(base, base + 0.02, N)
        df.loc[start_idx:end_idx-1, 'sxg'] = np.random.uniform(base, base + 0.02, N)
        df.loc[start_idx:end_idx-1, 'lyg'] = np.random.uniform(base, base + 0.02, N)
        df.loc[start_idx:end_idx-1, 'dzg'] = np.random.uniform(base, base + 0.02, N)

    print(f"  [{i}/{len(alert_scenarios)}] {scenario_type}: Samples {start_idx}-{end_idx} (Day {start_day})")

# Step 3: Save RAW sensor data ONLY (remove 'state' column if exists)
print(f"\n[3/3] Saving RAW sensor data...")

df_raw = df[raw_columns].copy()

# Save raw sensor data
raw_file = output_dir / f'{cow_id}_raw_sensor_data.csv'
df_raw.to_csv(raw_file, index=False)

print(f"  [OK] {raw_file} ({len(df_raw)} rows)")

# Verify alert scenarios
print(f"\n  Verification of alert scenarios:")
for scenario in alert_scenarios:
    start_idx = int((scenario['start_day'] - 1) * 24 * 60)
    end_idx = int(start_idx + (scenario['duration_days'] * 24 * 60))
    period = df_raw.iloc[start_idx:end_idx]
    print(f"    • {scenario['type']}: Temp={period['temperature'].mean():.2f}°C, Motion={period['fxa'].mean():.4f}")

print("\n" + "="*70)
print("RAW SENSOR DATA GENERATED!")
print("="*70)

print(f"\nSaved to: {raw_file.absolute()}")

print(f"\nData Summary:")
print(f"  • Cow ID: {cow_id}")
print(f"  • Total samples: {len(df_raw):,}")
print(f"  • Duration: {duration_days} days")
print(f"  • Time range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
print(f"  • Temperature range: {df_raw['temperature'].min():.2f}°C - {df_raw['temperature'].max():.2f}°C")
print(f"  • Alert scenarios: {len(alert_scenarios)} conditions across {duration_days} days")

print(f"\n" + "="*70)
print("NEXT STEPS - Test End-to-End Processing:")
print("="*70)
print(f"\n1. Start dashboard:")
print(f"   streamlit run dashboard/app.py")
print(f"\n2. Upload raw sensor CSV:")
print(f"   • File: {raw_file}")
print(f"   • Cow ID: {cow_id} (fixed in single-cow mode)")
print(f"   • Baseline Temp: {baseline_temp}°C")
print(f"\n3. Dashboard will process through 3-Layer Intelligence System:")
print(f"   Layer 1: Classify behavioral states (lying, standing, etc.)")
print(f"   Layer 2: Analyze temperature patterns")
print(f"   Layer 3: Detect health alerts (fever, inactivity)")
print(f"   + Dashboard Metrics: Calculate health score (0-100)")
print(f"\n4. Expected results (21-day comprehensive test):")
print(f"   • Home page shows single cow live feed with health score")
print(f"   • Health Analysis page shows 21-day health score trend")
print(f"   • Alerts page shows multiple alerts from 7 different scenarios:")
print(f"")
print(f"   FEVER TYPES (3 levels):")
print(f"     - Mild fever (Days 3-4): 39.5-39.8°C, Warning severity")
print(f"     - Moderate fever (Days 11-12): 39.8-40.5°C, Critical severity")
print(f"     - Severe fever (Days 19-20): >40.5°C, Critical severity")
print(f"")
print(f"   REPRODUCTIVE EVENTS:")
print(f"     - Estrus/Heat (Day 6): Elevated temp + high activity, Info")
print(f"     - Pregnancy (Day 17): Reduced activity, Info")
print(f"")
print(f"   OTHER CONDITIONS:")
print(f"     - Heat stress (Day 9): High temp + high activity, Warning")
print(f"     - Inactivity (Days 14-15): Very low motion, Warning")
print(f"")
print(f"   • Health scores will vary across the 21 days (showing declining/recovering patterns)")
print(f"   • Total expected alerts: 15-20+ (across all 7 scenarios)")
print(f"   • Database should contain: 1 health score, 15+ alerts, state history entries")

print("\n" + "="*70)
