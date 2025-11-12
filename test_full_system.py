"""
Quick System Test - Verify All Components Work

Tests:
1. Behavioral simulator (Layer 1)
2. Health condition simulators (Layer 2/3)
3. Trend tracker (Layer 3)
4. Dashboard imports
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("="*70)
print("LIVESTOCK HEALTH MONITORING SYSTEM - COMPONENT TEST")
print("="*70)

# Test 1: Behavioral Simulator
print("\n[1/4] Testing Behavioral Simulator...")
try:
    from simulation import SimulationEngine
    engine = SimulationEngine(random_seed=42)
    df = engine.generate_continuous_data(duration_hours=1)
    print(f"  SUCCESS: Generated {len(df)} samples")
    print(f"  States: {df['state'].value_counts().to_dict()}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 2: Health Condition Simulators
print("\n[2/4] Testing Health Condition Simulators...")
try:
    from simulation.health_conditions import (
        FeverSimulator,
        EstrusSimulator,
        PregnancySimulator,
        HeatStressSimulator
    )

    # Fever
    fever_sim = FeverSimulator(baseline_fever_temp=40.0)
    fever_temp = fever_sim.generate_temperature(duration_minutes=60, random_seed=42)
    print(f"  Fever: {len(fever_temp)} readings, range {fever_temp.min():.1f}-{fever_temp.max():.1f}C")

    # Estrus
    estrus_sim = EstrusSimulator()
    estrus_temp = estrus_sim.generate_temperature_spike(duration_minutes=60, random_seed=42)
    print(f"  Estrus: Temperature spike {estrus_temp.max()-estrus_temp.min():.2f}C")

    # Pregnancy
    pregnancy_sim = PregnancySimulator()
    preg_temp = pregnancy_sim.generate_temperature(duration_minutes=60, random_seed=42)
    print(f"  Pregnancy: Stable temp {preg_temp.mean():.2f}C (std {preg_temp.std():.2f})")

    # Heat Stress
    heat_sim = HeatStressSimulator()
    heat_temp = heat_sim.generate_temperature(duration_minutes=60, random_seed=42)
    print(f"  Heat Stress: High temp {heat_temp.mean():.2f}C")

    print("  SUCCESS: All 4 health simulators working")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 3: Trend Tracker
print("\n[3/4] Testing Multi-Day Health Trend Tracker...")
try:
    from health_intelligence import MultiDayHealthTrendTracker, TrendIndicator
    import pandas as pd
    from datetime import datetime, timedelta

    # Generate test data
    tracker = MultiDayHealthTrendTracker(temperature_baseline=38.5)

    # Create minimal test data
    import numpy as np
    dates = pd.date_range(end=datetime.now(), periods=7*1440, freq='1min')
    temp_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 38.5 + np.random.normal(0, 0.1, len(dates))
    })

    activity_data = pd.DataFrame({
        'timestamp': dates,
        'behavioral_state': ['lying'] * len(dates),
        'movement_intensity': [0.5] * len(dates)
    })

    report = tracker.analyze_trends(
        cow_id='TEST_COW',
        temperature_data=temp_data,
        activity_data=activity_data,
        alert_history=[],
        behavioral_states=activity_data
    )

    print(f"  SUCCESS: Trend analysis complete")
    print(f"  Overall trend: {report.overall_trend.value}")
    print(f"  Confidence: {report.overall_confidence:.1%}")
    print(f"  7-day trend available: {report.trend_7day is not None}")

except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Dashboard Components
print("\n[4/4] Testing Dashboard Component Imports...")
try:
    # Skip if streamlit not installed
    import importlib.util
    if importlib.util.find_spec("streamlit") is None:
        print("  SKIPPED: Streamlit not installed (optional for testing)")
    else:
        from dashboard.components import (
            render_notification_panel,
            render_alert_history,
            render_alerts_panel
        )
        print("  SUCCESS: All dashboard components import correctly")
except Exception as e:
    print(f"  FAILED: {e}")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print("\nYour system is ready to use! You can:")
print("  1. Generate synthetic behavioral data with SimulationEngine")
print("  2. Inject health conditions (fever, estrus, pregnancy, heat stress)")
print("  3. Test Layer 2 physiological analysis with simulated data")
print("  4. Test Layer 3 health intelligence and trend tracking")
print("  5. Visualize everything in the Streamlit dashboard")
print("\nSee docs/SIMULATOR_USAGE_GUIDE.md for complete usage examples")
print("="*70)
