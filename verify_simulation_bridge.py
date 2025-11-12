"""
Simple Verification - Simulation Bridge Utility

Tests that the simulation_data_bridge.py utility exists and can be imported.
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / 'dashboard'))

print("=" * 70)
print("SIMULATION BRIDGE VERIFICATION")
print("=" * 70)

# Test 1: Check file exists
print("\n[1/3] Checking bridge utility file exists...")
bridge_file = Path(__file__).parent / 'dashboard' / 'utils' / 'simulation_data_bridge.py'
if bridge_file.exists():
    print(f"  OK File exists: {bridge_file}")
else:
    print(f"  ERROR File not found: {bridge_file}")
    sys.exit(1)

# Test 2: Import the module
print("\n[2/3] Importing bridge utility...")
try:
    from utils import simulation_data_bridge
    print("  OK Module imported successfully")
except ImportError as e:
    print(f"  ERROR Import failed: {e}")
    sys.exit(1)

# Test 3: Check functions exist
print("\n[3/3] Checking bridge utility functions...")
required_functions = [
    'is_using_simulation',
    'get_data_source',
    'get_alerts_source',
    'get_simulation_sensor_data',
    'get_simulation_alerts',
    'get_simulation_trend_report',
    'get_simulation_cow_id',
    'get_simulation_baseline_temp',
    'render_data_source_indicator',
    'get_latest_reading_from_simulation',
    'get_time_range_from_simulation',
    'export_simulation_to_csv',
    'clear_simulation_data',
    'get_simulation_metadata',
]

missing = []
for func_name in required_functions:
    if hasattr(simulation_data_bridge, func_name):
        print(f"  OK {func_name}")
    else:
        print(f"  ERROR Missing: {func_name}")
        missing.append(func_name)

if missing:
    print(f"\n  ERROR Missing {len(missing)} functions")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
print("\nOK All checks passed!")
print("\nYour simulation bridge utility is ready to use.")
print("\nNext steps:")
print("  1. Run: streamlit run dashboard/app.py")
print("  2. Go to: Simulation Testing page")
print("  3. Generate simulation data")
print("  4. Navigate to other pages to see it in action")
print("\nSee: HOW_TO_TEST_APP_WITH_SIMULATION.md for complete guide")
print("=" * 70)
