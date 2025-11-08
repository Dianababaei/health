#!/usr/bin/env python3
"""
Quick test script for dataset generation.

Tests that all components work together correctly by generating
small sample datasets.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.dataset_generator import (
    DatasetGenerationConfig,
    DatasetGenerator
)


def test_short_dataset():
    """Test short-term dataset generation (reduced to 1 day for speed)."""
    print("=" * 60)
    print("Testing Short-Term Dataset Generation (1 day sample)")
    print("=" * 60)
    
    config = DatasetGenerationConfig(
        duration_days=1,
        animal_id="test_cow_001",
        seed=42,
        include_estrus=False,
        include_pregnancy=False,
        num_illness_events=1,
        num_heat_stress_events=0
    )
    
    generator = DatasetGenerator(config)
    
    print("Generating dataset...")
    start_time = datetime.now()
    datasets = generator.generate_dataset()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    print(f"✓ Generation completed in {duration:.2f} seconds")
    print(f"  Data points: {len(datasets['labeled_data'])}")
    print(f"  Expected: {1 * 24 * 60} (1440)")
    
    # Validate
    validation = generator.validate_dataset(datasets['labeled_data'])
    print(f"\nValidation:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Errors: {len(validation['errors'])}")
    print(f"  Warnings: {len(validation['warnings'])}")
    
    if validation['errors']:
        for error in validation['errors']:
            print(f"    ERROR: {error}")
    
    if validation['warnings']:
        for warning in validation['warnings'][:3]:  # Show first 3
            print(f"    WARNING: {warning}")
    
    # Check columns
    print(f"\nColumns present:")
    print(f"  Sensor data: {all(c in datasets['labeled_data'].columns for c in ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg'])}")
    print(f"  Labels: {all(c in datasets['labeled_data'].columns for c in ['behavioral_state', 'temperature_status', 'health_events', 'sensor_quality'])}")
    
    # Check behavioral states
    state_dist = datasets['labeled_data']['behavioral_state'].value_counts()
    print(f"\nBehavioral state distribution:")
    for state, count in state_dist.items():
        pct = (count / len(datasets['labeled_data'])) * 100
        print(f"  {state}: {pct:.1f}%")
    
    # Check health events
    health_dist = datasets['labeled_data']['health_events'].value_counts()
    print(f"\nHealth events:")
    for event, count in health_dist.items():
        print(f"  {event}: {count} minutes")
    
    # Check daily aggregates
    print(f"\nDaily aggregates:")
    print(f"  Days: {len(datasets['daily_aggregates'])}")
    if len(datasets['daily_aggregates']) > 0:
        day = datasets['daily_aggregates'].iloc[0]
        print(f"  Health score: {day['health_score']:.1f}")
        print(f"  Mean temp: {day['mean_temperature']:.2f}°C")
    
    return datasets


def test_medium_dataset():
    """Test medium-term dataset generation (reduced to 3 days for speed)."""
    print("\n" + "=" * 60)
    print("Testing Medium-Term Dataset Generation (3 day sample)")
    print("=" * 60)
    
    config = DatasetGenerationConfig(
        duration_days=3,
        animal_id="test_cow_002",
        seed=123,
        include_estrus=False,  # Too short
        include_pregnancy=False,
        num_illness_events=0,
        num_heat_stress_events=1
    )
    
    generator = DatasetGenerator(config)
    
    print("Generating dataset...")
    start_time = datetime.now()
    datasets = generator.generate_dataset()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    print(f"✓ Generation completed in {duration:.2f} seconds")
    print(f"  Data points: {len(datasets['labeled_data'])}")
    print(f"  Expected: {3 * 24 * 60} (4320)")
    
    # Validate
    validation = generator.validate_dataset(datasets['labeled_data'])
    print(f"\nValidation:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Errors: {len(validation['errors'])}")
    print(f"  Warnings: {len(validation['warnings'])}")
    
    # Check for heat stress
    heat_stress_count = (datasets['labeled_data']['health_events'] == 'heat_stress').sum()
    print(f"\nHeat stress events: {heat_stress_count} minutes")
    
    return datasets


def test_with_estrus():
    """Test with estrus cycle (30 days minimum)."""
    print("\n" + "=" * 60)
    print("Testing Dataset with Estrus Cycle (30 days)")
    print("=" * 60)
    
    config = DatasetGenerationConfig(
        duration_days=30,
        animal_id="test_cow_003",
        seed=456,
        include_estrus=True,
        include_pregnancy=True,
        num_illness_events=1,
        num_heat_stress_events=2
    )
    
    generator = DatasetGenerator(config)
    
    print("Generating dataset...")
    start_time = datetime.now()
    datasets = generator.generate_dataset()
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds()
    
    print(f"✓ Generation completed in {duration:.2f} seconds")
    print(f"  Data points: {len(datasets['labeled_data'])}")
    
    # Check for estrus and pregnancy
    estrus_count = (datasets['labeled_data']['health_events'] == 'estrus').sum()
    pregnancy_count = (datasets['labeled_data']['health_events'] == 'pregnancy_indication').sum()
    illness_count = (datasets['labeled_data']['health_events'] == 'illness').sum()
    
    print(f"\nHealth events:")
    print(f"  Estrus: {estrus_count} minutes")
    print(f"  Pregnancy indication: {pregnancy_count} minutes")
    print(f"  Illness: {illness_count} minutes")
    
    # Check daily aggregates
    daily = datasets['daily_aggregates']
    estrus_days = daily['estrus_day'].sum()
    pregnancy_days = daily['pregnancy_day'].sum()
    
    print(f"\nDaily aggregates:")
    print(f"  Estrus days: {estrus_days}")
    print(f"  Pregnancy days: {pregnancy_days}")
    print(f"  Mean health score: {daily['health_score'].mean():.1f}")
    
    return datasets


def main():
    """Run all tests."""
    print("Dataset Generation Test Suite")
    print("Testing dataset generation components...")
    print()
    
    try:
        # Test 1: Short dataset
        datasets_short = test_short_dataset()
        
        # Test 2: Medium dataset
        datasets_medium = test_medium_dataset()
        
        # Test 3: With estrus
        datasets_estrus = test_with_estrus()
        
        print("\n" + "=" * 60)
        print("✓ All tests completed successfully!")
        print("=" * 60)
        print("\nDataset generation system is working correctly.")
        print("You can now generate full datasets using:")
        print("  python scripts/generate_datasets.py")
        
        return 0
    
    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ Test failed with error:")
        print("=" * 60)
        print(f"{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
