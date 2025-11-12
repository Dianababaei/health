"""
Quick Demo - See Your Simulators in Action!

This script demonstrates:
1. Generating behavioral data
2. Simulating health conditions
3. Testing trend analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("="*70)
print("LIVESTOCK HEALTH MONITORING - QUICK DEMO")
print("="*70)

# ============================================================================
# DEMO 1: Generate Behavioral Data
# ============================================================================
print("\n" + "="*70)
print("DEMO 1: Behavioral Data Simulation")
print("="*70)

from simulation import SimulationEngine

engine = SimulationEngine(baseline_temperature=38.5, random_seed=42)

print("\nGenerating 24 hours of realistic cow behavior...")
df = engine.generate_continuous_data(duration_hours=24)

print(f"\nOK Generated {len(df)} sensor readings (1 per minute)")
print(f"\nBehavioral states detected:")
for state, count in df['state'].value_counts().items():
    percentage = (count / len(df)) * 100
    print(f"  {state:12s}: {count:4d} samples ({percentage:5.1f}%)")

print(f"\nTemperature range: {df['temperature'].min():.2f}°C - {df['temperature'].max():.2f}°C")
print(f"Average activity (fxa): {df['fxa'].mean():.3f} m/s²")

# Save sample
sample_file = "demo_behavioral_data.csv"
df.head(100).to_csv(sample_file, index=False)
print(f"\nOK Saved first 100 samples to: {sample_file}")

# ============================================================================
# DEMO 2: Simulate Health Conditions
# ============================================================================
print("\n" + "="*70)
print("DEMO 2: Health Condition Simulation")
print("="*70)

from simulation.health_conditions import (
    FeverSimulator,
    EstrusSimulator,
    PregnancySimulator,
    HeatStressSimulator
)

# Test each health condition
print("\n1. Fever Simulation (40°C elevated temperature)")
fever_sim = FeverSimulator(baseline_fever_temp=40.0, activity_reduction=0.30)
fever_temp = fever_sim.generate_temperature(duration_minutes=1440, random_seed=42)
print(f"   Temperature range: {fever_temp.min():.2f}°C - {fever_temp.max():.2f}°C")
print(f"   Above fever threshold (>39.5°C): {(fever_temp > 39.5).sum()} / {len(fever_temp)} samples")

print("\n2. Estrus Simulation (breeding behavior)")
estrus_sim = EstrusSimulator()
estrus_temp = estrus_sim.generate_temperature_spike(duration_minutes=60, random_seed=42)
spike_magnitude = estrus_temp.max() - estrus_temp.min()
print(f"   Temperature spike: {spike_magnitude:.2f}°C (typical estrus indicator)")

print("\n3. Pregnancy Simulation (stable temperature)")
pregnancy_sim = PregnancySimulator()
preg_temp = pregnancy_sim.generate_temperature(duration_minutes=1440, random_seed=42)
print(f"   Temperature: {preg_temp.mean():.2f}°C (std: {preg_temp.std():.3f}°C)")
print(f"   Stability: Very stable (pregnancy dampens circadian rhythm)")

print("\n4. Heat Stress Simulation (high temp + panting)")
heat_sim = HeatStressSimulator()
heat_temp = heat_sim.generate_temperature(duration_minutes=1440, random_seed=42)
panting = heat_sim.generate_panting_pattern(duration_minutes=60, random_seed=42)
print(f"   Temperature: {heat_temp.mean():.2f}°C (elevated)")
print(f"   Panting pattern: {panting.mean():.2f} (rapid breathing indicator)")

# ============================================================================
# DEMO 3: Multi-Day Trend Analysis
# ============================================================================
print("\n" + "="*70)
print("DEMO 3: Multi-Day Health Trend Analysis")
print("="*70)

from health_intelligence import MultiDayHealthTrendTracker

# Generate 14 days of data with gradual fever onset
print("\nScenario: Cow develops fever gradually over 14 days")
print("Generating 14 days of sensor data...")

dates = pd.date_range(end=datetime.now(), periods=14*1440, freq='1min')

# Temperature increases gradually (simulates illness onset)
day_of_illness = np.arange(len(dates)) / 1440  # Days since start
temp_increase = np.minimum(day_of_illness * 0.15, 1.5)  # Max 1.5°C increase
base_temp = 38.5
noise = np.random.normal(0, 0.1, len(dates))
temperatures = base_temp + temp_increase + noise

temp_df = pd.DataFrame({
    'timestamp': dates,
    'temperature': temperatures
})

# Activity decreases as illness progresses
activity_decrease = np.minimum(day_of_illness * 0.05, 0.3)
activity = 0.5 - activity_decrease + np.random.normal(0, 0.05, len(dates))

activity_df = pd.DataFrame({
    'timestamp': dates,
    'behavioral_state': ['lying'] * len(dates),
    'movement_intensity': activity
})

# Run trend analysis
print("Analyzing health trends over 7 and 14-day periods...")
tracker = MultiDayHealthTrendTracker(temperature_baseline=38.5)

report = tracker.analyze_trends(
    cow_id='DEMO_COW_001',
    temperature_data=temp_df,
    activity_data=activity_df,
    alert_history=[],
    behavioral_states=activity_df
)

print(f"\nOK Analysis complete!")
print(f"\nRESULTS:")
print(f"  Overall Health Trend: {report.overall_trend.value.upper()}")
print(f"  Confidence: {report.overall_confidence:.1%}")

if report.trend_7day:
    print(f"\n  7-Day Period:")
    print(f"    Temperature: {report.trend_7day.temperature_mean:.2f}°C (drift: {report.trend_7day.temperature_baseline_drift:+.2f}°C)")
    print(f"    Activity: {report.trend_7day.activity_level_mean:.2f}")
    print(f"    Trend: {report.trend_7day.trend_indicator.value}")

if report.trend_14day:
    print(f"\n  14-Day Period:")
    print(f"    Temperature: {report.trend_14day.temperature_mean:.2f}°C (drift: {report.trend_14day.temperature_baseline_drift:+.2f}°C)")
    print(f"    Activity: {report.trend_14day.activity_level_mean:.2f}")
    print(f"    Trend: {report.trend_14day.trend_indicator.value}")

print(f"\n  Significant Changes Detected:")
for change in report.significant_changes:
    print(f"    - {change}")

print(f"\n  Recommendations:")
for rec in report.recommendations:
    print(f"    - {rec}")

# ============================================================================
# DEMO 4: Export for Dashboard
# ============================================================================
print("\n" + "="*70)
print("DEMO 4: Dashboard Data Export")
print("="*70)

dashboard_data = report.to_dict()

print("\nOK Health trend report converted to JSON-ready format")
print(f"\nDashboard data structure:")
print(f"  Keys: {list(dashboard_data.keys())}")
print(f"\n  7-day trend available: {dashboard_data['trend_7day'] is not None}")

if dashboard_data['trend_7day']:
    print(f"  7-day metrics available:")
    print(f"    - temperature: {list(dashboard_data['trend_7day']['temperature'].keys())}")
    print(f"    - activity: {list(dashboard_data['trend_7day']['activity'].keys())}")
    print(f"    - alerts: {list(dashboard_data['trend_7day']['alerts'].keys())}")

# Save dashboard data
import json
dashboard_file = "demo_dashboard_data.json"
with open(dashboard_file, 'w') as f:
    # Convert datetime to string for JSON
    export_data = {
        'cow_id': dashboard_data['cow_id'],
        'overall_trend': dashboard_data['overall_trend'],
        'overall_confidence': dashboard_data['overall_confidence'],
        'recommendations': dashboard_data['recommendations']
    }
    json.dump(export_data, f, indent=2)

print(f"\nOK Saved dashboard data to: {dashboard_file}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("DEMO COMPLETE!")
print("="*70)
print("\nYou just saw:")
print("  OK Behavioral data simulation (lying, walking, etc.)")
print("  OK Health condition simulation (fever, estrus, pregnancy, heat stress)")
print("  OK Multi-day trend analysis (7-day and 14-day periods)")
print("  OK Dashboard data export (JSON format)")
print("\nFiles created:")
print(f"  - {sample_file} - Sample behavioral data")
print(f"  - {dashboard_file} - Dashboard-ready health trends")
print("\nNext steps:")
print("  - Run: python test_full_system.py (complete system test)")
print("  - Read: docs/SIMULATOR_USAGE_GUIDE.md (detailed guide)")
print("  - Read: EXAMPLES_GUIDE.md (all available examples)")
print("="*70)
