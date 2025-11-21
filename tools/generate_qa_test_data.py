"""
QA/Product Manager Test Data Generator

Generates comprehensive test dataset covering ALL system features:
- All alert types (fever, heat stress, inactivity, sensor malfunction, estrus)
- All behavioral states (lying, standing, walking, ruminating, feeding)
- Normal baseline periods
- Edge cases and transitions
- Multi-day scenarios

Run this to get a complete dataset for testing every feature.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path


def generate_comprehensive_test_data(
    cow_id: str = "QA_TEST_001",
    days: int = 21,
    output_dir: str = "data/dashboard"
):
    """
    Generate 21 days of comprehensive test data covering all scenarios.

    Timeline:
    Day 1-2: Normal baseline (establish normal patterns)
    Day 3: Fever event (high temp + low motion)
    Day 4: Recovery from fever
    Day 5-6: Normal
    Day 7: Heat stress (high temp + high activity)
    Day 8: Recovery
    Day 9-10: Normal
    Day 11: Prolonged inactivity (4+ hours stillness)
    Day 12: Recovery
    Day 13-14: Normal
    Day 15: Estrus event (temp rise + activity increase)
    Day 16-17: Normal
    Day 18: Sensor malfunction simulation (out of range values)
    Day 19-20: Normal
    Day 21: Complex day (multiple behavioral states)
    """

    print("=" * 60)
    print("QA/PRODUCT MANAGER COMPREHENSIVE TEST DATA GENERATOR")
    print("=" * 60)
    print(f"\nCow ID: {cow_id}")
    print(f"Duration: {days} days")
    print(f"Output: {output_dir}/")
    print("\nGenerating scenarios:")
    print("  [OK] Normal baseline (Days 1-2)")
    print("  [OK] Fever alert (Day 3)")
    print("  [OK] Heat stress alert (Day 7)")
    print("  [OK] Prolonged inactivity alert (Day 11)")
    print("  [OK] Estrus detection (Day 15)")
    print("  [OK] Sensor malfunction (Day 18)")
    print("  [OK] All behavioral states (throughout)")
    print("\n" + "=" * 60)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize data structure
    start_date = datetime(2024, 11, 1, 0, 0, 0)
    minutes_per_day = 24 * 60
    total_minutes = days * minutes_per_day

    timestamps = [start_date + timedelta(minutes=i) for i in range(total_minutes)]

    data = {
        'timestamp': timestamps,
        'temperature': np.zeros(total_minutes),
        'fxa': np.zeros(total_minutes),
        'mya': np.zeros(total_minutes),
        'rza': np.zeros(total_minutes),
        'sxg': np.zeros(total_minutes),
        'lyg': np.zeros(total_minutes),
        'dzg': np.zeros(total_minutes),
    }

    # Normal baseline temperature with circadian rhythm
    baseline_temp = 38.5
    for i, ts in enumerate(timestamps):
        hour = ts.hour + ts.minute / 60.0
        # Circadian rhythm: peak at 18:00, trough at 06:00
        circadian = 0.4 * np.sin(2 * np.pi * (hour - 6) / 24)
        data['temperature'][i] = baseline_temp + circadian + np.random.normal(0, 0.1)

    # Helper function to set behavioral state
    def set_behavior(start_idx, end_idx, state):
        if state == 'lying':
            data['rza'][start_idx:end_idx] = np.random.uniform(-0.9, -0.6, end_idx - start_idx)
            data['fxa'][start_idx:end_idx] = np.random.normal(0, 0.05, end_idx - start_idx)
            data['mya'][start_idx:end_idx] = np.random.normal(0, 0.05, end_idx - start_idx)
            data['sxg'][start_idx:end_idx] = np.random.normal(0, 2, end_idx - start_idx)
            data['lyg'][start_idx:end_idx] = np.random.normal(0, 2, end_idx - start_idx)
            data['dzg'][start_idx:end_idx] = np.random.normal(0, 2, end_idx - start_idx)

        elif state == 'standing':
            data['rza'][start_idx:end_idx] = np.random.uniform(0.7, 0.9, end_idx - start_idx)
            data['fxa'][start_idx:end_idx] = np.random.normal(0, 0.1, end_idx - start_idx)
            data['mya'][start_idx:end_idx] = np.random.normal(0, 0.1, end_idx - start_idx)
            data['sxg'][start_idx:end_idx] = np.random.normal(0, 3, end_idx - start_idx)
            data['lyg'][start_idx:end_idx] = np.random.normal(0, 3, end_idx - start_idx)
            data['dzg'][start_idx:end_idx] = np.random.normal(0, 3, end_idx - start_idx)

        elif state == 'walking':
            data['rza'][start_idx:end_idx] = np.random.uniform(0.5, 0.8, end_idx - start_idx)
            # Walking has high Fxa variance
            for i in range(start_idx, end_idx):
                step_freq = 1.0  # 1 Hz (60 steps/min)
                t = (i - start_idx) / 60.0
                data['fxa'][i] = 0.3 * np.sin(2 * np.pi * step_freq * t) + np.random.normal(0, 0.1)
            data['mya'][start_idx:end_idx] = np.random.normal(0, 0.15, end_idx - start_idx)
            data['sxg'][start_idx:end_idx] = np.random.normal(0, 8, end_idx - start_idx)
            data['lyg'][start_idx:end_idx] = np.random.normal(0, 8, end_idx - start_idx)
            data['dzg'][start_idx:end_idx] = np.random.normal(0, 8, end_idx - start_idx)

        elif state == 'ruminating':
            data['rza'][start_idx:end_idx] = np.random.uniform(-0.9, -0.6, end_idx - start_idx)
            # Rumination has specific jaw movement frequency (50 cycles/min = 0.83 Hz)
            for i in range(start_idx, end_idx):
                t = (i - start_idx) / 60.0
                data['mya'][i] = 0.12 * np.sin(2 * np.pi * 0.83 * t) + np.random.normal(0, 0.03)
                data['lyg'][i] = 10 * np.sin(2 * np.pi * 0.83 * t) + np.random.normal(0, 2)
            data['fxa'][start_idx:end_idx] = np.random.normal(0, 0.05, end_idx - start_idx)
            data['sxg'][start_idx:end_idx] = np.random.normal(0, 2, end_idx - start_idx)
            data['dzg'][start_idx:end_idx] = np.random.normal(0, 2, end_idx - start_idx)

        elif state == 'feeding':
            data['rza'][start_idx:end_idx] = np.random.uniform(0.5, 0.7, end_idx - start_idx)
            data['lyg'][start_idx:end_idx] = np.random.uniform(-25, -15, end_idx - start_idx)  # Head down
            data['mya'][start_idx:end_idx] = np.random.normal(0, 0.2, end_idx - start_idx)
            data['fxa'][start_idx:end_idx] = np.random.normal(0, 0.1, end_idx - start_idx)
            data['sxg'][start_idx:end_idx] = np.random.normal(0, 5, end_idx - start_idx)
            data['dzg'][start_idx:end_idx] = np.random.normal(0, 5, end_idx - start_idx)

        elif state == 'inactive':
            # Complete stillness
            data['rza'][start_idx:end_idx] = 0.0
            data['fxa'][start_idx:end_idx] = 0.0
            data['mya'][start_idx:end_idx] = 0.0
            data['sxg'][start_idx:end_idx] = 0.0
            data['lyg'][start_idx:end_idx] = 0.0
            data['dzg'][start_idx:end_idx] = 0.0

    # Generate daily patterns with all behaviors
    for day in range(days):
        day_start = day * minutes_per_day

        # Normal daily pattern (default for most days)
        if day not in [2, 6, 10, 14, 17, 20]:  # Skip special event days
            # Night lying (00:00 - 06:00)
            set_behavior(day_start, day_start + 360, 'lying')

            # Morning feeding (06:00 - 07:00)
            set_behavior(day_start + 360, day_start + 420, 'feeding')

            # Morning walking/grazing (07:00 - 09:00)
            set_behavior(day_start + 420, day_start + 540, 'walking')

            # Mid-morning ruminating while lying (09:00 - 11:00)
            set_behavior(day_start + 540, day_start + 660, 'ruminating')

            # Midday standing/resting (11:00 - 13:00)
            set_behavior(day_start + 660, day_start + 780, 'standing')

            # Afternoon walking (13:00 - 15:00)
            set_behavior(day_start + 780, day_start + 900, 'walking')

            # Late afternoon feeding (15:00 - 16:00)
            set_behavior(day_start + 900, day_start + 960, 'feeding')

            # Evening ruminating (16:00 - 18:00)
            set_behavior(day_start + 960, day_start + 1080, 'ruminating')

            # Evening lying (18:00 - 24:00)
            set_behavior(day_start + 1080, day_start + 1440, 'lying')

    # DAY 3: FEVER EVENT
    day = 2
    day_start = day * minutes_per_day
    print(f"\n[Day {day+1}] Generating FEVER alert...")
    # Elevated temperature + low motion
    data['temperature'][day_start:day_start + 1440] += 1.5  # 40.0°C (fever)
    set_behavior(day_start, day_start + 1440, 'lying')  # Lethargic all day
    # Add some minimal movement (not complete stillness)
    data['fxa'][day_start:day_start + 1440] += np.random.normal(0, 0.08, 1440)

    # DAY 7: HEAT STRESS
    day = 6
    day_start = day * minutes_per_day
    print(f"[Day {day+1}] Generating HEAT STRESS alert...")
    # High temp during hot afternoon + high activity
    afternoon_start = day_start + 720  # 12:00
    afternoon_end = day_start + 960    # 16:00
    data['temperature'][afternoon_start:afternoon_end] += 1.2  # 39.7°C
    set_behavior(afternoon_start, afternoon_end, 'walking')  # Pacing/agitation
    # Rest of day normal
    set_behavior(day_start, afternoon_start, 'lying')
    set_behavior(afternoon_end, day_start + 1440, 'lying')

    # DAY 11: PROLONGED INACTIVITY
    day = 10
    day_start = day * minutes_per_day
    print(f"[Day {day+1}] Generating PROLONGED INACTIVITY alert...")
    # 6 hours of complete stillness (not lying - just inactive)
    inactivity_start = day_start + 480  # 08:00
    inactivity_end = day_start + 840    # 14:00 (6 hours)
    set_behavior(inactivity_start, inactivity_end, 'inactive')
    # Rest of day normal
    set_behavior(day_start, inactivity_start, 'lying')
    set_behavior(inactivity_end, day_start + 1440, 'standing')

    # DAY 15: ESTRUS EVENT
    day = 14
    day_start = day * minutes_per_day
    print(f"[Day {day+1}] Generating ESTRUS alert...")
    # Temperature rise + increased activity for 12 hours
    estrus_start = day_start + 360   # 06:00
    estrus_end = day_start + 1080     # 18:00
    data['temperature'][estrus_start:estrus_end] += 0.5  # +0.5°C rise
    # Increased activity (restlessness, mounting behavior)
    for i in range(estrus_start, estrus_end, 120):  # Every 2 hours
        set_behavior(i, i + 60, 'walking')  # 1 hour walking
        set_behavior(i + 60, i + 120, 'standing')  # 1 hour standing (restless)
    # Rest of day normal
    set_behavior(day_start, estrus_start, 'lying')
    set_behavior(estrus_end, day_start + 1440, 'lying')

    # DAY 18: SENSOR MALFUNCTION
    day = 17
    day_start = day * minutes_per_day
    print(f"[Day {day+1}] Generating SENSOR MALFUNCTION alert...")
    # Out of range temperature values
    malfunction_start = day_start + 600  # 10:00
    malfunction_end = day_start + 720    # 12:00
    data['temperature'][malfunction_start:malfunction_end] = 45.0  # Way out of range
    # Stuck sensor values (no variance)
    data['fxa'][malfunction_start:malfunction_end] = 0.5
    data['mya'][malfunction_start:malfunction_end] = 0.5
    data['rza'][malfunction_start:malfunction_end] = 0.5
    # Rest of day normal
    set_behavior(day_start, malfunction_start, 'lying')
    set_behavior(malfunction_end, day_start + 1440, 'standing')

    # DAY 21: COMPLEX DAY (all behaviors clearly visible)
    day = 20
    day_start = day * minutes_per_day
    print(f"[Day {day+1}] Generating COMPLEX behavioral day...")
    set_behavior(day_start, day_start + 240, 'lying')       # 00:00-04:00
    set_behavior(day_start + 240, day_start + 360, 'ruminating')  # 04:00-06:00
    set_behavior(day_start + 360, day_start + 420, 'feeding')     # 06:00-07:00
    set_behavior(day_start + 420, day_start + 540, 'walking')     # 07:00-09:00
    set_behavior(day_start + 540, day_start + 660, 'standing')    # 09:00-11:00
    set_behavior(day_start + 660, day_start + 780, 'feeding')     # 11:00-13:00
    set_behavior(day_start + 780, day_start + 960, 'walking')     # 13:00-16:00
    set_behavior(day_start + 960, day_start + 1080, 'ruminating') # 16:00-18:00
    set_behavior(day_start + 1080, day_start + 1200, 'standing')  # 18:00-20:00
    set_behavior(day_start + 1200, day_start + 1440, 'lying')     # 20:00-24:00

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add calculated motion_intensity
    df['motion_intensity'] = np.sqrt(df['fxa']**2 + df['mya']**2 + df['rza']**2)

    # Save sensor data
    sensor_file = Path(output_dir) / f"{cow_id}_sensor_data.csv"
    df.to_csv(sensor_file, index=False)
    print(f"\n[OK] Sensor data saved: {sensor_file}")
    print(f"  - {len(df)} data points ({days} days)")
    print(f"  - Columns: {list(df.columns)}")

    # Create metadata
    metadata = {
        "cow_id": cow_id,
        "name": "QA Test Cow",
        "breed": "Holstein",
        "age_years": 4,
        "weight_kg": 650,
        "lactation_number": 2,
        "days_in_milk": 120,
        "reproductive_status": "cycling",
        "baseline_temperature": baseline_temp,
        "data_start": start_date.isoformat(),
        "data_end": timestamps[-1].isoformat(),
        "total_days": days,
        "sampling_rate": "1 per minute",
        "notes": "Comprehensive QA test data covering all alert types and behaviors",
        "test_scenarios": {
            "day_3": "Fever event (40°C + low motion)",
            "day_7": "Heat stress (39.7°C + high activity)",
            "day_11": "Prolonged inactivity (6 hours stillness)",
            "day_15": "Estrus event (+0.5°C + increased activity)",
            "day_18": "Sensor malfunction (out of range values)",
            "day_21": "Complex behavioral patterns (all states)"
        }
    }

    metadata_file = Path(output_dir) / f"{cow_id}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Metadata saved: {metadata_file}")

    # Generate expected alerts summary
    expected_alerts = {
        "expected_alert_count": 5,
        "alerts": [
            {
                "day": 3,
                "type": "fever",
                "severity": "critical",
                "description": "Temperature 40°C + low motion",
                "expected_trigger_time": "Throughout day 3"
            },
            {
                "day": 7,
                "type": "heat_stress",
                "severity": "warning",
                "description": "Temperature 39.7°C + high activity",
                "expected_trigger_time": "Day 7, 12:00-16:00"
            },
            {
                "day": 11,
                "type": "prolonged_inactivity",
                "severity": "warning",
                "description": "6 hours complete stillness",
                "expected_trigger_time": "Day 11, 08:00-14:00"
            },
            {
                "day": 15,
                "type": "estrus",
                "severity": "info",
                "description": "+0.5°C temperature + increased activity",
                "expected_trigger_time": "Day 15, 06:00-18:00"
            },
            {
                "day": 18,
                "type": "sensor_malfunction",
                "severity": "critical",
                "description": "Temperature out of range (45°C)",
                "expected_trigger_time": "Day 18, 10:00-12:00"
            }
        ],
        "behavioral_states_to_verify": [
            "lying", "standing", "walking", "ruminating", "feeding"
        ],
        "health_score_expectations": {
            "days_1_2": "High (85-95) - normal baseline",
            "day_3": "Low (20-40) - fever penalty",
            "day_7": "Medium (50-70) - heat stress",
            "day_11": "Medium-Low (40-60) - inactivity",
            "day_15": "Medium-High (70-85) - estrus (not severe health issue)",
            "day_18": "Very Low (0-20) - sensor malfunction",
            "day_21": "High (80-95) - all behaviors normal"
        }
    }

    qa_guide_file = Path(output_dir) / f"{cow_id}_QA_GUIDE.json"
    with open(qa_guide_file, 'w') as f:
        json.dump(expected_alerts, f, indent=2)
    print(f"[OK] QA test guide saved: {qa_guide_file}")

    print("\n" + "=" * 60)
    print("QA TEST DATA GENERATION COMPLETE!")
    print("=" * 60)
    print("\nNEXT STEPS FOR TESTING:")
    print("1. Close dashboard if running (Ctrl+C)")
    print("2. Delete old database: del data\\alert_state.db")
    print("3. Start dashboard: streamlit run dashboard/app.py")
    print("4. Upload the generated CSV file from dashboard")
    print("5. Navigate through all 3 pages to verify features")
    print("\nEXPECTED RESULTS:")
    print("  [OK] 5 alerts should be detected (fever, heat stress, inactivity, estrus, malfunction)")
    print("  [OK] All 5 behavioral states should appear in charts")
    print("  [OK] Health score should vary from 0-95 across the 21 days")
    print("  [OK] Timeline should show clear event markers")
    print(f"\nFiles generated in: {output_dir}/")
    print(f"  - {cow_id}_sensor_data.csv (main data)")
    print(f"  - {cow_id}_metadata.json (cow info)")
    print(f"  - {cow_id}_QA_GUIDE.json (testing checklist)")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    generate_comprehensive_test_data(
        cow_id="QA_TEST_001",
        days=21,
        output_dir="data/dashboard"
    )
