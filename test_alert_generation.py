"""
Test script to verify alert generation with the fixed fever simulation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from simulation import SimulationEngine
from simulation.health_conditions import FeverSimulator
from health_intelligence.alerts.immediate_detector import ImmediateAlertDetector
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

print("=" * 60)
print("ALERT GENERATION TEST")
print("=" * 60)

# Parameters (matching your simulation settings)
cow_id = 'TEST_ALERT_001'
baseline_temp = 38.5
duration_days = 7
fever_day = 3
fever_duration = 2
fever_temp = 40.0

print(f"\nGenerating {duration_days} days of data with fever on day {fever_day}...")

# Step 1: Generate baseline data
engine = SimulationEngine(
    baseline_temperature=baseline_temp,
    sampling_rate=1.0
)

df = engine.generate_continuous_data(
    duration_hours=duration_days * 24,
    start_datetime=datetime.now() - timedelta(days=duration_days)
)

print(f"[OK] Generated {len(df)} samples")

# Step 2: Inject fever (USING THE NEW FIXED CODE)
fever_start_idx = (fever_day - 1) * 24 * 60
fever_end_idx = fever_start_idx + (fever_duration * 24 * 60)

fever_sim = FeverSimulator(
    baseline_fever_temp=fever_temp,
    activity_reduction=0.30
)

fever_temp_data = fever_sim.generate_temperature(
    duration_minutes=fever_duration * 24 * 60
)

df.loc[fever_start_idx:fever_end_idx-1, 'temperature'] = fever_temp_data

# NEW FIXED CODE - Set absolute low motion values
df.loc[fever_start_idx:fever_end_idx-1, 'fxa'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
df.loc[fever_start_idx:fever_end_idx-1, 'mya'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
df.loc[fever_start_idx:fever_end_idx-1, 'rza'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
df.loc[fever_start_idx:fever_end_idx-1, 'sxg'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
df.loc[fever_start_idx:fever_end_idx-1, 'lyg'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
df.loc[fever_start_idx:fever_end_idx-1, 'dzg'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)

print(f"[OK] Injected fever on day {fever_day} (indices {fever_start_idx}-{fever_end_idx})")

# Verify fever conditions
fever_df = df.iloc[fever_start_idx:fever_end_idx]
fever_df['motion'] = np.sqrt(fever_df['fxa']**2 + fever_df['mya']**2 + fever_df['rza']**2)

print(f"\nFever Period Verification:")
print(f"  Temperature: {fever_df['temperature'].min():.2f} - {fever_df['temperature'].max():.2f}°C")
print(f"  Motion: {fever_df['motion'].min():.3f} - {fever_df['motion'].max():.3f}")
print(f"  Motion avg: {fever_df['motion'].mean():.3f}")
print(f"  Threshold: 0.150")
print(f"  [OK] All motion below threshold: {fever_df['motion'].max() < 0.15}")

# Step 3: Detect alerts (USING THE NEW FIXED CODE)
print(f"\nDetecting alerts...")
alerts = []
detector = ImmediateAlertDetector()

window_size = 10
detected_count = 0

for idx in range(window_size, len(df), 10):
    window_start = max(0, idx - window_size)
    window_df = df.iloc[window_start:idx+1].copy()

    detected = detector.detect_alerts(
        sensor_data=window_df,
        cow_id=cow_id,
        behavioral_state=df.iloc[idx]['state'],
        baseline_temp=baseline_temp
    )

    if detected:
        detected_count += len(detected)
        alerts.extend([{
            'timestamp': str(a.timestamp),
            'alert_type': a.alert_type,
            'severity': a.severity,
            'cow_id': cow_id
        } for a in detected])

print(f"[OK] Alert detection complete")
print(f"\n{'='*60}")
print(f"RESULTS:")
print(f"{'='*60}")
print(f"Total alerts detected: {len(alerts)}")

if len(alerts) > 0:
    print(f"\n[SUCCESS] Alerts were detected!\n")

    # Group by type
    alert_types = {}
    for alert in alerts:
        alert_type = alert['alert_type']
        if alert_type not in alert_types:
            alert_types[alert_type] = []
        alert_types[alert_type].append(alert)

    print("Alert breakdown:")
    for alert_type, type_alerts in alert_types.items():
        print(f"  {alert_type}: {len(type_alerts)} alerts")
        severities = {}
        for a in type_alerts:
            sev = a['severity']
            severities[sev] = severities.get(sev, 0) + 1
        for sev, count in severities.items():
            print(f"    - {sev}: {count}")

    print(f"\nFirst few alerts:")
    for i, alert in enumerate(alerts[:5]):
        print(f"  {i+1}. {alert['alert_type']} - {alert['severity']} at {alert['timestamp']}")

    # Save to file
    output_file = Path(__file__).parent / 'data' / 'simulation' / f'{cow_id}_alerts.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(alerts, f, indent=2)

    print(f"\n[OK] Alerts saved to: {output_file}")
    print(f"\n{'='*60}")
    print("THE FIX WORKS! Alerts are being generated correctly.")
    print("Now regenerate the simulation in the Streamlit app.")
    print(f"{'='*60}")

else:
    print(f"\n[FAILED] No alerts detected\n")
    print("This shouldn't happen. Check:")
    print("1. Fever temperature is high enough")
    print("2. Motion is low enough")
    print("3. Alert detector is working")
