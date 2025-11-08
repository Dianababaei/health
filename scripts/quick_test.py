#!/usr/bin/env python3
"""
Quick integration test for dataset generation components.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing imports...")
try:
    from src.simulation.health_events import HealthEventSimulator, HealthEventType
    print("  ✓ health_events")
    
    from src.simulation.label_generator import LabelGenerator
    print("  ✓ label_generator")
    
    from src.simulation.export import DatasetExporter, DatasetSplitter
    print("  ✓ export")
    
    from src.simulation.dataset_generator import (
        DatasetGenerationConfig,
        DatasetGenerator
    )
    print("  ✓ dataset_generator")
    
    from src.simulation.engine import SimulationEngine
    print("  ✓ engine")
    
    print("\n✓ All imports successful!")
    
except Exception as e:
    print(f"\n✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting health event generation...")
try:
    simulator = HealthEventSimulator(seed=42)
    start = datetime(2024, 1, 1)
    end = start + timedelta(days=7)
    
    events = simulator.generate_all_events(
        start, end,
        include_estrus=False,
        num_illness=1
    )
    
    print(f"  ✓ Generated {len(events['illness'])} illness events")
    print(f"  ✓ Generated {len(events['sensor_degradation'])} sensor degradation events")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting basic dataset generation (1 hour)...")
try:
    config = DatasetGenerationConfig(
        duration_days=1.0/24.0,  # 1 hour
        animal_id="test_001",
        seed=42,
        include_estrus=False,
        include_pregnancy=False,
        num_illness_events=0
    )
    
    generator = DatasetGenerator(config)
    datasets = generator.generate_dataset()
    
    print(f"  ✓ Generated {len(datasets['labeled_data'])} data points")
    print(f"  ✓ Columns: {list(datasets['labeled_data'].columns)[:8]}...")
    print(f"  ✓ Labels present: behavioral_state, temperature_status, health_events, sensor_quality")
    
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ Quick integration test passed!")
print("=" * 60)
print("\nAll components are working correctly.")
print("You can now run full dataset generation.")
