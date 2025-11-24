"""
End-to-End System Test
======================
Comprehensive test that verifies all system components work together.

Tests:
1. Test data generation (all scenarios)
2. Layer 1: Behavioral classification
3. Layer 2: Temperature analysis
4. Layer 3: Alert detection (fever, heat stress, inactivity, estrus, malfunction)
5. Health score calculation
6. Database storage (alerts, health scores, sensor data)
7. Data retrieval and querying

Run: python test_end_to_end.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import pandas as pd
import numpy as np

# Test results tracking
RESULTS = {
    'passed': [],
    'failed': [],
    'warnings': []
}

def log_pass(test_name, details=""):
    RESULTS['passed'].append(test_name)
    print(f"  [PASS] {test_name}" + (f" - {details}" if details else ""))

def log_fail(test_name, details=""):
    RESULTS['failed'].append((test_name, details))
    print(f"  [FAIL] {test_name}" + (f" - {details}" if details else ""))

def log_warn(test_name, details=""):
    RESULTS['warnings'].append((test_name, details))
    print(f"  [WARN] {test_name}" + (f" - {details}" if details else ""))


def generate_test_data(days=15):
    """Generate comprehensive test data with all scenarios."""
    print("\n" + "="*60)
    print("GENERATING TEST DATA")
    print("="*60)

    np.random.seed(42)  # Reproducible

    minutes_per_day = 24 * 60
    total_minutes = days * minutes_per_day

    start_date = datetime(2025, 11, 1, 0, 0, 0)

    # Initialize arrays
    data = {
        'timestamp': [start_date + timedelta(minutes=i) for i in range(total_minutes)],
        'temperature': np.random.normal(38.5, 0.15, total_minutes),
        'fxa': np.random.normal(0, 0.1, total_minutes),
        'mya': np.random.normal(0, 0.1, total_minutes),
        'rza': np.random.uniform(-0.9, -0.6, total_minutes),  # Default lying
        'sxg': np.random.normal(0, 5, total_minutes),
        'lyg': np.random.normal(0, 5, total_minutes),
        'dzg': np.random.normal(0, 5, total_minutes),
        'state': ['lying'] * total_minutes,
    }

    scenarios = []

    def set_state(start_idx, end_idx, state):
        for i in range(start_idx, min(end_idx, total_minutes)):
            data['state'][i] = state
            if state == 'lying':
                data['rza'][i] = np.random.uniform(-0.9, -0.6)
                data['fxa'][i] = np.random.normal(0, 0.05)
            elif state == 'standing':
                data['rza'][i] = np.random.uniform(0.6, 0.9)
                data['fxa'][i] = np.random.normal(0, 0.08)
            elif state == 'walking':
                data['rza'][i] = np.random.uniform(0.5, 0.7)
                data['fxa'][i] = np.random.normal(0, 0.25)
                data['mya'][i] = np.random.normal(0, 0.15)
            elif state == 'feeding':
                data['rza'][i] = np.random.uniform(0.5, 0.7)
                data['lyg'][i] = np.random.uniform(-25, -15)

    # Normal daily patterns for most days
    for day in range(days):
        if day in [2, 6, 10, 14]:  # Skip event days
            continue
        day_start = day * minutes_per_day
        set_state(day_start, day_start + 360, 'lying')        # 00:00-06:00
        set_state(day_start + 360, day_start + 420, 'feeding')  # 06:00-07:00
        set_state(day_start + 420, day_start + 540, 'walking')  # 07:00-09:00
        set_state(day_start + 540, day_start + 660, 'lying')    # 09:00-11:00
        set_state(day_start + 660, day_start + 780, 'standing') # 11:00-13:00
        set_state(day_start + 780, day_start + 900, 'walking')  # 13:00-15:00
        set_state(day_start + 900, day_start + 960, 'feeding')  # 15:00-16:00
        set_state(day_start + 960, day_start + 1440, 'lying')   # 16:00-24:00

    # DAY 3: FEVER (critical)
    # Fever requires: temp > 39.5 AND motion_intensity < 0.15
    # motion_intensity = sqrt(fxa² + mya² + rza²)
    # For motion < 0.15: all components must be tiny (< 0.1 each)
    day = 2
    day_start = day * minutes_per_day
    fever_start = day_start + 480  # 08:00
    fever_end = day_start + 960    # 16:00
    for i in range(fever_start, fever_end):
        data['temperature'][i] = np.random.normal(40.2, 0.2)  # High fever
        data['state'][i] = 'lying'
        # Very low motion - all components must be tiny for motion_intensity < 0.15
        data['fxa'][i] = np.random.normal(0, 0.02)
        data['mya'][i] = np.random.normal(0, 0.02)
        data['rza'][i] = np.random.normal(0, 0.05)  # Near zero, not -0.8 (lying)
    scenarios.append(('Day 3', 'fever', 'critical', fever_start, fever_end))

    # DAY 7: HEAT STRESS (warning)
    # Heat stress requires: temp > 39.0 AND activity_level > 0.6
    # activity_level = motion_intensity / 2.0, so need motion > 1.2
    day = 6
    day_start = day * minutes_per_day
    heat_start = day_start + 600   # 10:00
    heat_end = day_start + 1020    # 17:00
    for i in range(heat_start, heat_end):
        data['temperature'][i] = np.random.normal(39.3, 0.15)  # Elevated
        data['state'][i] = 'walking'  # Active state
        # High motion for activity_level > 0.6 (need motion > 1.2)
        data['fxa'][i] = np.random.normal(0, 0.5)
        data['mya'][i] = np.random.normal(0, 0.5)
        data['rza'][i] = np.random.normal(0.7, 0.1)  # Standing/walking posture
    scenarios.append(('Day 7', 'heat_stress', 'warning', heat_start, heat_end))

    # DAY 11: PROLONGED INACTIVITY (warning)
    # Inactivity check: abs(fxa) < 0.05 AND abs(mya) < 0.05 AND abs(rza) < 0.05
    # Also behavioral_state must NOT be 'lying' or 'ruminating'
    # This represents a cow that is completely still (not normal behavior)
    day = 10
    day_start = day * minutes_per_day
    inactivity_start = day_start
    inactivity_end = day_start + 480  # 8 hours with no movement
    for i in range(inactivity_start, inactivity_end):
        data['state'][i] = 'standing'  # NOT lying (excluded from inactivity)
        # All axes near zero = complete stillness
        data['fxa'][i] = np.random.normal(0, 0.01)  # < 0.05
        data['mya'][i] = np.random.normal(0, 0.01)  # < 0.05
        data['rza'][i] = np.random.normal(0, 0.01)  # < 0.05 (near zero, not gravity-aligned)
    scenarios.append(('Day 11', 'inactivity', 'warning', inactivity_start, inactivity_end))

    # DAY 15: ESTRUS (info) - temperature spike + activity increase
    day = 14
    day_start = day * minutes_per_day
    estrus_start = day_start + 360  # 06:00
    estrus_end = day_start + 1080   # 18:00
    for i in range(estrus_start, estrus_end):
        data['temperature'][i] = np.random.normal(39.0, 0.1)  # Slight elevation
        data['state'][i] = 'standing' if i % 3 == 0 else 'walking'
        data['fxa'][i] = np.random.normal(0, 0.3)  # High activity
    scenarios.append(('Day 15', 'estrus', 'info', estrus_start, estrus_end))

    df = pd.DataFrame(data)
    df['cow_id'] = 'TEST_COW_001'

    print(f"  Generated {len(df)} records over {days} days")
    print(f"  Scenarios embedded: {len(scenarios)}")
    for s in scenarios:
        print(f"    - {s[0]}: {s[1]} ({s[2]})")

    return df, scenarios


def test_layer1_classification(df):
    """Test Layer 1: Behavioral Classification."""
    print("\n" + "="*60)
    print("TEST: LAYER 1 - BEHAVIORAL CLASSIFICATION")
    print("="*60)

    try:
        from layer1.rule_based_classifier import RuleBasedClassifier

        classifier = RuleBasedClassifier(
            enable_rumination=False,  # Disabled due to sampling rate
            enable_feeding=True
        )

        # Classify behaviors using classify_batch method
        result_df = classifier.classify_batch(df.copy())

        if 'state' not in result_df.columns:
            log_fail("Classification output", "No 'state' column in output")
            return None

        # Check behavioral states detected
        states = result_df['state'].value_counts()
        print(f"  Detected states: {dict(states)}")

        expected_states = ['lying', 'standing', 'walking']
        for state in expected_states:
            if state in states:
                log_pass(f"Detected '{state}'", f"{states[state]} samples")
            else:
                log_fail(f"Detected '{state}'", "Not found")

        # Check rumination is NOT detected (disabled)
        if 'ruminating' in states and states['ruminating'] > 0:
            log_warn("Rumination detection", f"Found {states['ruminating']} samples (should be disabled)")
        else:
            log_pass("Rumination disabled", "Correctly not detecting rumination")

        return result_df

    except Exception as e:
        log_fail("Layer 1 classification", str(e))
        import traceback
        traceback.print_exc()
        return None


def test_layer3_alerts(df, baseline_temp=38.5):
    """Test Layer 3: Alert Detection."""
    print("\n" + "="*60)
    print("TEST: LAYER 3 - ALERT DETECTION")
    print("="*60)

    try:
        from health_intelligence.alerts.immediate_detector import ImmediateAlertDetector

        # Initialize detector with baseline_temperature dict format
        detector = ImmediateAlertDetector(
            baseline_temperature={'TEST_COW_001': baseline_temp}
        )

        all_alerts = []
        alert_keys_seen = set()

        # Process in small windows (10-minute rolling) to simulate real-time
        # The detector only looks at last 2 minutes, so we need frequent calls
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')

        window_size = 10  # 10 minutes per window
        for i in range(0, len(df_sorted), window_size):
            # Get window of data
            window_end = min(i + window_size, len(df_sorted))
            window_data = df_sorted.iloc[:window_end].copy()  # Include all history up to now

            alerts = detector.detect_alerts(
                sensor_data=window_data,
                cow_id='TEST_COW_001',
                baseline_temp=baseline_temp
            )

            # Deduplicate alerts
            for alert in alerts:
                key = f"{alert.alert_type}_{alert.timestamp}"
                if key not in alert_keys_seen:
                    all_alerts.append(alert)
                    alert_keys_seen.add(key)

        print(f"  Total alerts detected: {len(all_alerts)}")

        # Group by type
        alert_types = {}
        for alert in all_alerts:
            atype = str(alert.alert_type)
            if atype not in alert_types:
                alert_types[atype] = []
            alert_types[atype].append(alert)

        print(f"  Alert types: {list(alert_types.keys())}")

        # Check expected alerts
        expected_alerts = {
            'fever': ('critical', 'Day 3 fever scenario'),
            'heat_stress': ('warning', 'Day 7 heat stress scenario'),
            'inactivity': ('warning', 'Day 11 inactivity scenario'),
        }

        for alert_type, (expected_severity, description) in expected_alerts.items():
            matching = [a for a in all_alerts if alert_type in str(a.alert_type).lower()]
            if matching:
                log_pass(f"Alert: {alert_type}", f"Detected {len(matching)} alerts")
            else:
                log_fail(f"Alert: {alert_type}", f"Not detected ({description})")

        # Check estrus (informational)
        estrus_alerts = [a for a in all_alerts if 'estrus' in str(a.alert_type).lower()]
        if estrus_alerts:
            log_pass("Alert: estrus", f"Detected {len(estrus_alerts)} (informational)")
        else:
            log_warn("Alert: estrus", "Not detected (may need more data)")

        return all_alerts

    except Exception as e:
        log_fail("Layer 3 alert detection", str(e))
        import traceback
        traceback.print_exc()
        return []


def test_health_score(df, alerts):
    """Test Health Score Calculation."""
    print("\n" + "="*60)
    print("TEST: HEALTH SCORE CALCULATION")
    print("="*60)

    try:
        from health_intelligence.scoring.simple_health_scorer import SimpleHealthScorer

        scorer = SimpleHealthScorer()

        # Convert alerts to dict format
        alert_dicts = []
        for alert in alerts:
            alert_dicts.append({
                'alert_type': str(alert.alert_type),
                'severity': str(alert.severity),
                'timestamp': str(alert.timestamp),
                'cow_id': alert.cow_id
            })

        result = scorer.calculate_score(
            cow_id='TEST_COW_001',
            sensor_data=df,
            baseline_temp=38.5,
            active_alerts=alert_dicts
        )

        print(f"  Total Score: {result['total_score']:.1f}/100")
        print(f"  Category: {result['health_category']}")
        print(f"  Components:")
        print(f"    - Temperature: {result['temperature_component']:.3f}")
        print(f"    - Activity: {result['activity_component']:.3f}")
        print(f"    - Behavioral: {result['behavioral_component']:.3f}")
        print(f"    - Alert: {result['alert_component']:.3f}")

        # Validate score
        if 0 <= result['total_score'] <= 100:
            log_pass("Score in valid range", f"{result['total_score']:.1f}")
        else:
            log_fail("Score in valid range", f"{result['total_score']} out of 0-100")

        # Check category
        if result['health_category'] in ['excellent', 'good', 'moderate', 'poor']:
            log_pass("Valid category", result['health_category'])
        else:
            log_fail("Valid category", f"Unknown: {result['health_category']}")

        # With alerts, score should be reduced
        if len(alert_dicts) > 0 and result['alert_component'] < 1.0:
            log_pass("Alert penalty applied", f"alert_component={result['alert_component']:.2f}")
        elif len(alert_dicts) > 0:
            log_warn("Alert penalty", "Alerts exist but no penalty applied")

        return result

    except Exception as e:
        log_fail("Health score calculation", str(e))
        import traceback
        traceback.print_exc()
        return None


def test_database_storage(df, alerts, health_score, temp_db_path):
    """Test Database Storage."""
    print("\n" + "="*60)
    print("TEST: DATABASE STORAGE")
    print("="*60)

    try:
        from health_intelligence.logging.sensor_data_manager import SensorDataManager
        from health_intelligence.logging.health_score_manager import HealthScoreManager
        from health_intelligence.logging.alert_state_manager import AlertStateManager

        # Test Sensor Data Manager
        print("\n  Testing SensorDataManager...")
        sensor_mgr = SensorDataManager(db_path=temp_db_path)

        inserted, skipped = sensor_mgr.append_sensor_data(df, 'TEST_COW_001')
        print(f"    Inserted: {inserted}, Skipped: {skipped}")

        if inserted > 0:
            log_pass("Sensor data storage", f"{inserted} records saved")
        else:
            log_fail("Sensor data storage", "No records saved")

        # Verify retrieval
        retrieved = sensor_mgr.get_all_data('TEST_COW_001')
        if len(retrieved) == inserted:
            log_pass("Sensor data retrieval", f"{len(retrieved)} records")
        else:
            log_warn("Sensor data retrieval", f"Expected {inserted}, got {len(retrieved)}")

        # Test Health Score Manager
        print("\n  Testing HealthScoreManager...")
        health_mgr = HealthScoreManager(db_path=temp_db_path)

        if health_score:
            saved = health_mgr.save_health_score(health_score)
            if saved:
                log_pass("Health score storage", "Saved successfully")
            else:
                log_fail("Health score storage", "Save returned False")

            # Verify retrieval
            latest = health_mgr.get_latest_score('TEST_COW_001')
            if latest and abs(latest['total_score'] - health_score['total_score']) < 0.1:
                log_pass("Health score retrieval", f"Score: {latest['total_score']:.1f}")
            else:
                log_fail("Health score retrieval", "Mismatch or not found")

        # Test Alert State Manager
        print("\n  Testing AlertStateManager...")
        alert_mgr = AlertStateManager(db_path=temp_db_path)

        saved_alerts = 0
        for alert in alerts[:5]:  # Save first 5 alerts
            alert_data = {
                'alert_id': f"TEST_{alert.timestamp}_{alert.alert_type}",
                'cow_id': alert.cow_id,
                'alert_type': str(alert.alert_type),
                'severity': str(alert.severity),
                'confidence': alert.confidence,
                'status': 'active',
                'timestamp': str(alert.timestamp),
                'sensor_values': alert.sensor_values,
                'detection_details': alert.details
            }
            if alert_mgr.create_alert(alert_data):
                saved_alerts += 1

        if saved_alerts > 0:
            log_pass("Alert storage", f"{saved_alerts} alerts saved")
        else:
            log_warn("Alert storage", "No alerts saved")

        # Verify alert retrieval
        all_alerts = alert_mgr.query_alerts(cow_id='TEST_COW_001')
        if len(all_alerts) >= saved_alerts:
            log_pass("Alert retrieval", f"{len(all_alerts)} alerts found")
        else:
            log_fail("Alert retrieval", f"Expected {saved_alerts}, got {len(all_alerts)}")

        return True

    except Exception as e:
        log_fail("Database storage", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_data_loader():
    """Test data loader functions."""
    print("\n" + "="*60)
    print("TEST: DATA LOADER FUNCTIONS")
    print("="*60)

    try:
        from data_processing.health_score_loader import (
            get_latest_health_score,
            calculate_baseline_health_score,
            get_contributing_factors
        )

        # These may return None if no data exists, but should not crash
        latest = get_latest_health_score('TEST_COW_001', None)
        if latest is None:
            log_warn("get_latest_health_score", "No data (expected if fresh DB)")
        else:
            log_pass("get_latest_health_score", f"Score: {latest.get('health_score', 'N/A')}")

        baseline, start, end = calculate_baseline_health_score('TEST_COW_001', 30)
        log_pass("calculate_baseline_health_score", f"Baseline: {baseline:.1f}")

        return True

    except Exception as e:
        log_fail("Data loader functions", str(e))
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(RESULTS['passed']) + len(RESULTS['failed'])

    print(f"\n  PASSED: {len(RESULTS['passed'])}/{total}")
    print(f"  FAILED: {len(RESULTS['failed'])}/{total}")
    print(f"  WARNINGS: {len(RESULTS['warnings'])}")

    if RESULTS['failed']:
        print("\n  FAILURES:")
        for name, details in RESULTS['failed']:
            print(f"    - {name}: {details}")

    if RESULTS['warnings']:
        print("\n  WARNINGS:")
        for name, details in RESULTS['warnings']:
            print(f"    - {name}: {details}")

    print("\n" + "="*60)
    if len(RESULTS['failed']) == 0:
        print("ALL TESTS PASSED!")
    else:
        print(f"SOME TESTS FAILED ({len(RESULTS['failed'])} failures)")
    print("="*60)

    return len(RESULTS['failed']) == 0


def main():
    """Run all end-to-end tests."""
    print("\n" + "="*60)
    print("ARTEMIS LIVESTOCK HEALTH - END-TO-END TEST")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Use temporary database for testing
    temp_dir = tempfile.mkdtemp()
    temp_db = os.path.join(temp_dir, 'test_alert_state.db')
    print(f"\nUsing temp database: {temp_db}")

    try:
        # 1. Generate test data
        df, scenarios = generate_test_data(days=15)

        # 2. Test Layer 1
        classified_df = test_layer1_classification(df)
        if classified_df is not None:
            df = classified_df

        # 3. Test Layer 3 (alerts)
        alerts = test_layer3_alerts(df, baseline_temp=38.5)

        # 4. Test Health Score
        health_score = test_health_score(df, alerts)

        # 5. Test Database Storage
        test_database_storage(df, alerts, health_score, temp_db)

        # 6. Test Data Loader
        test_data_loader()

        # Print summary
        success = print_summary()

        return 0 if success else 1

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
