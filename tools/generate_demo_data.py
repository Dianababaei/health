"""
Generate demo data for Artemis Livestock Health Monitoring
Standalone script - no Streamlit needed
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from simulation import SimulationEngine
from simulation.health_conditions import FeverSimulator
from health_intelligence import MultiDayHealthTrendTracker
from health_intelligence.alerts.immediate_detector import ImmediateAlertDetector

print("="*70)
print("ARTEMIS DEMO DATA GENERATOR")
print("="*70)

# Configuration
cow_id = "DEMO_COW_001"
duration_days = 14
baseline_temp = 38.5
fever_day = 3
fever_duration = 2
fever_temp = 40.0

print(f"\nConfiguration:")
print(f"  Cow ID: {cow_id}")
print(f"  Duration: {duration_days} days")
print(f"  Baseline Temperature: {baseline_temp}°C")
print(f"  Fever: Day {fever_day}, Duration {fever_duration} days, Temp {fever_temp}°C")

# Step 1: Generate baseline data
print(f"\n[1/5] Generating {duration_days} days of baseline behavioral data...")

engine = SimulationEngine(
    baseline_temperature=baseline_temp,
    sampling_rate=1.0  # 1 sample per minute
)

df = engine.generate_continuous_data(
    duration_hours=duration_days * 24
)

print(f"  [OK] Generated {len(df)} sensor readings")

# Step 2: Inject fever condition
print(f"\n[2/5] Injecting fever condition...")

fever_sim = FeverSimulator(
    baseline_fever_temp=fever_temp,
    activity_reduction=0.30
)

fever_start_idx = (fever_day - 1) * 24 * 60
fever_end_idx = fever_start_idx + (fever_duration * 24 * 60)

# Generate fever temperature
fever_temp_data = fever_sim.generate_temperature(
    duration_minutes=fever_duration * 24 * 60
)

df.loc[fever_start_idx:fever_end_idx-1, 'temperature'] = fever_temp_data

# Set motion to very low (below 0.15 threshold for fever detection)
N = fever_end_idx - fever_start_idx
df.loc[fever_start_idx:fever_end_idx-1, 'fxa'] = np.random.uniform(0.02, 0.05, N)
df.loc[fever_start_idx:fever_end_idx-1, 'mya'] = np.random.uniform(0.02, 0.05, N)
df.loc[fever_start_idx:fever_end_idx-1, 'rza'] = np.random.uniform(0.02, 0.05, N)
df.loc[fever_start_idx:fever_end_idx-1, 'sxg'] = np.random.uniform(0.02, 0.05, N)
df.loc[fever_start_idx:fever_end_idx-1, 'lyg'] = np.random.uniform(0.02, 0.05, N)
df.loc[fever_start_idx:fever_end_idx-1, 'dzg'] = np.random.uniform(0.02, 0.05, N)

print(f"  [OK] Fever injected from sample {fever_start_idx} to {fever_end_idx}")

# Step 3: Detect alerts
print(f"\n[3/5] Detecting health alerts...")

alerts = []
detector = ImmediateAlertDetector()

window_size = 10
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
        for a in detected:
            alerts.append({
                'timestamp': str(a.timestamp),
                'cow_id': cow_id,
                'alert_type': str(a.alert_type),
                'severity': str(a.severity)
            })

print(f"  [OK] Detected {len(alerts)} alerts")

# Count by type
fever_alerts = sum(1 for a in alerts if a['alert_type'] == 'fever')
inactivity_alerts = sum(1 for a in alerts if a['alert_type'] == 'inactivity')
print(f"    - Fever: {fever_alerts}")
print(f"    - Inactivity: {inactivity_alerts}")

# Step 4: Calculate trends
print(f"\n[4/5] Analyzing health trends...")

tracker = MultiDayHealthTrendTracker(temperature_baseline=baseline_temp)

temp_df = df[['timestamp', 'temperature']].copy()
activity_df = df[['timestamp', 'state']].copy()
activity_df['behavioral_state'] = activity_df['state']
activity_df['movement_intensity'] = df[['fxa', 'mya', 'rza']].abs().sum(axis=1) / 3.0

trend_report = tracker.analyze_trends(
    cow_id=cow_id,
    temperature_data=temp_df,
    activity_data=activity_df,
    alert_history=alerts,
    behavioral_states=activity_df
)

print(f"  [OK] Health trend: {trend_report.overall_trend.value}")
print(f"  [OK] Confidence: {trend_report.overall_confidence:.0%}")

# Step 5: Save files
print(f"\n[5/5] Saving demo data files...")

# Create dashboard directory
demo_dir = Path('data/dashboard')
demo_dir.mkdir(parents=True, exist_ok=True)

# Save sensor data
sensor_file = demo_dir / f'{cow_id}_sensor_data.csv'
df.to_csv(sensor_file, index=False)
print(f"  [OK] {sensor_file} ({len(df)} rows)")

# Save alerts to JSON
alerts_file = demo_dir / f'{cow_id}_alerts.json'
with open(alerts_file, 'w') as f:
    json.dump(alerts, indent=2, fp=f)
print(f"  [OK] {alerts_file} ({len(alerts)} alerts)")

# ALSO save alerts to database (for Alerts page)
try:
    sys.path.insert(0, str(Path(__file__).parent / 'src'))
    from health_intelligence.logging import AlertStateManager

    state_manager = AlertStateManager(db_path="data/alert_state.db")

    for i, alert in enumerate(alerts):
        alert_data = {
            'alert_id': f"DEMO_{alert['cow_id']}_{alert['alert_type']}_{i}_{datetime.now().timestamp()}",
            'cow_id': alert['cow_id'],
            'alert_type': alert['alert_type'],
            'severity': alert['severity'],
            'timestamp': alert['timestamp'],
            'confidence': 0.95,
            'sensor_values': {},
            'detection_details': {'source': 'demo_data'}
        }
        state_manager.create_alert(alert_data)

    print(f"  [OK] Saved {len(alerts)} alerts to database")
except Exception as e:
    print(f"  [WARNING] Could not save to database: {e}")

# Save metadata
metadata = {
    'cow_id': cow_id,
    'baseline_temp': baseline_temp,
    'duration_days': duration_days,
    'total_samples': len(df),
    'start_time': str(df['timestamp'].min()),
    'end_time': str(df['timestamp'].max()),
    'num_alerts': len(alerts),
    'conditions': {
        'fever': True,
        'estrus': False,
        'pregnancy': False,
        'heat_stress': False
    },
    'fever_details': {
        'start_day': fever_day,
        'duration_days': fever_duration,
        'temperature': fever_temp
    },
    'generated_at': str(datetime.now())
}

metadata_file = demo_dir / f'{cow_id}_metadata.json'
with open(metadata_file, 'w') as f:
    json.dump(metadata, indent=2, fp=f)
print(f"  [OK] {metadata_file}")

# Save trend report
trend_dict = trend_report.to_dict()
if 'analysis_timestamp' in trend_dict:
    trend_dict['analysis_timestamp'] = str(trend_dict['analysis_timestamp'])

trend_file = demo_dir / f'{cow_id}_trend_report.json'
with open(trend_file, 'w') as f:
    json.dump(trend_dict, indent=2, fp=f)
print(f"  [OK] {trend_file}")

print("\n" + "="*70)
print("DEMO DATA GENERATED SUCCESSFULLY!")
print("="*70)

print(f"\nFiles saved to: {demo_dir.absolute()}")
print(f"\nDemo Data Summary:")
print(f"  • Sensor readings: {len(df):,}")
print(f"  • Total alerts: {len(alerts)}")
print(f"  • Fever alerts: {fever_alerts}")
print(f"  • Inactivity alerts: {inactivity_alerts}")
print(f"  • Health trend: {trend_report.overall_trend.value.upper()}")
print(f"  • Confidence: {trend_report.overall_confidence:.0%}")

print(f"\nNext Steps:")
print(f"  1. Run main app: streamlit run dashboard/app.py")
print(f"  2. Demo data is already loaded in data/dashboard/")
print(f"  3. Dashboard will automatically show the demo data")
print(f"  4. Explore the dashboard!")
print(f"\nOptional - Test end-to-end processing:")
print(f"  • Upload raw sensor CSV: data/dashboard/{cow_id}_sensor_data.csv")
print(f"  • Dashboard will re-process through all 3 layers")

print("\n" + "="*70)
