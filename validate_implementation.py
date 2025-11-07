"""
Validation script to verify the dataset generator implementation.
Run this before generating full datasets to ensure everything is configured correctly.
"""

import sys
from pathlib import Path

# Add src/data to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "data"))

from datetime import datetime
import numpy as np
from synthetic_generator import SyntheticDataGenerator, BEHAVIOR_PATTERNS


def validate_behavior_patterns():
    """Verify all behavior patterns are defined."""
    print("\n✓ Checking behavior patterns...")
    
    expected_behaviors = ['lying', 'standing', 'walking', 'ruminating', 'feeding', 'stress']
    actual_behaviors = list(BEHAVIOR_PATTERNS.keys())
    
    assert set(actual_behaviors) == set(expected_behaviors), \
        f"Missing behaviors: {set(expected_behaviors) - set(actual_behaviors)}"
    
    print(f"  ✓ All 6 behaviors defined: {', '.join(expected_behaviors)}")
    
    # Check each pattern has required attributes
    for behavior, pattern in BEHAVIOR_PATTERNS.items():
        assert hasattr(pattern, 'name')
        assert hasattr(pattern, 'temp_mean')
        assert hasattr(pattern, 'fxa_range')
        print(f"  ✓ {behavior}: temp={pattern.temp_mean}°C")


def validate_generator_basic():
    """Test basic generator functionality."""
    print("\n✓ Testing basic generator...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate small sample
    df = generator.generate_behavior_sample(
        behavior='lying',
        duration_minutes=5,
        start_time=datetime(2024, 1, 1),
    )
    
    assert len(df) == 5, f"Expected 5 samples, got {len(df)}"
    assert 'timestamp' in df.columns
    assert 'temp' in df.columns
    assert 'behavior_label' in df.columns
    
    required_columns = ['timestamp', 'temp', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg', 'behavior_label']
    assert all(col in df.columns for col in required_columns), \
        f"Missing columns: {set(required_columns) - set(df.columns)}"
    
    print(f"  ✓ Generated {len(df)} samples with all required columns")
    print(f"  ✓ Columns: {', '.join(df.columns)}")


def validate_circadian_patterns():
    """Test circadian rhythm functions."""
    print("\n✓ Testing circadian patterns...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Test activity varies by time of day
    noon_activity = generator.circadian.get_hour_factor(12)
    night_activity = generator.circadian.get_hour_factor(2)
    
    assert noon_activity > night_activity, \
        f"Daytime activity ({noon_activity}) should be higher than night ({night_activity})"
    
    print(f"  ✓ Activity factor: noon={noon_activity}, night={night_activity}")
    
    # Test temperature varies by time of day
    afternoon_temp = generator.circadian.get_temperature_adjustment(16)
    morning_temp = generator.circadian.get_temperature_adjustment(4)
    
    assert afternoon_temp > morning_temp, \
        f"Afternoon temp ({afternoon_temp}) should be higher than morning ({morning_temp})"
    
    print(f"  ✓ Temperature adjustment: afternoon={afternoon_temp:.2f}°C, morning={morning_temp:.2f}°C")


def validate_transitions():
    """Test behavior transition generation."""
    print("\n✓ Testing behavior transitions...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate transition
    df = generator.generate_transition(
        from_behavior='lying',
        to_behavior='standing',
        start_time=datetime(2024, 1, 1),
        transition_minutes=3,
    )
    
    assert len(df) == 3, f"Expected 3 transition samples, got {len(df)}"
    assert (df['behavior_label'] == 'standing').all()
    
    print(f"  ✓ Generated {len(df)}-minute transition (lying → standing)")


def validate_sequence_generation():
    """Test behavior sequence generation."""
    print("\n✓ Testing behavior sequences...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    behaviors = [
        ('lying', 5),
        ('standing', 3),
        ('walking', 4),
    ]
    
    df = generator.generate_behavior_sequence(
        behaviors=behaviors,
        start_time=datetime(2024, 1, 1),
        smooth_transitions=True,
    )
    
    # Should have behavior samples + smooth transitions
    expected_min = 5 + 3 + 4  # Base durations
    assert len(df) >= expected_min, \
        f"Expected at least {expected_min} samples, got {len(df)}"
    
    print(f"  ✓ Generated sequence with {len(df)} samples")
    
    # Check all behaviors present
    unique_behaviors = df['behavior_label'].unique()
    print(f"  ✓ Behaviors in sequence: {', '.join(unique_behaviors)}")


def validate_daily_schedule():
    """Test daily schedule generation."""
    print("\n✓ Testing daily schedule generation...")
    
    generator = SyntheticDataGenerator(seed=42)
    schedule = generator.generate_daily_schedule(datetime(2024, 1, 1))
    
    assert len(schedule) > 0, "Schedule should not be empty"
    
    total_minutes = sum(duration for _, duration in schedule)
    
    # Should be reasonably close to 24 hours
    assert 1000 <= total_minutes <= 1600, \
        f"Daily schedule total ({total_minutes} min) out of reasonable range"
    
    print(f"  ✓ Generated schedule: {len(schedule)} activities, {total_minutes} minutes")
    print(f"  ✓ ~{total_minutes/60:.1f} hours (target: ~24 hours)")


def validate_sensor_ranges():
    """Test that sensor values are within expected ranges."""
    print("\n✓ Testing sensor value ranges...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate samples for each behavior
    all_valid = True
    for behavior in ['lying', 'standing', 'walking', 'ruminating', 'feeding', 'stress']:
        df = generator.generate_behavior_sample(
            behavior=behavior,
            duration_minutes=20,
            start_time=datetime(2024, 1, 1),
        )
        
        # Check ranges
        temp_ok = (df['temp'].min() >= 37.0) and (df['temp'].max() <= 41.0)
        acc_ok = (df[['Fxa', 'Mya', 'Rza']].abs().max().max() <= 3.0)
        gyro_ok = (df[['Sxg', 'Lyg', 'Dzg']].abs().max().max() <= 100)
        
        if not (temp_ok and acc_ok and gyro_ok):
            all_valid = False
            print(f"  ✗ {behavior}: Range check failed")
        else:
            print(f"  ✓ {behavior}: All sensor values in valid ranges")
    
    assert all_valid, "Some behaviors have sensor values out of range"


def validate_no_missing_data():
    """Test that generated data has no missing values."""
    print("\n✓ Testing for missing values...")
    
    generator = SyntheticDataGenerator(seed=42)
    
    df = generator.generate_behavior_sample(
        behavior='walking',
        duration_minutes=30,
        start_time=datetime(2024, 1, 1),
    )
    
    has_nulls = df.isnull().any().any()
    assert not has_nulls, "Generated data contains null values"
    
    print(f"  ✓ No missing values in {len(df)} samples")


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("DATASET GENERATOR IMPLEMENTATION VALIDATION")
    print("=" * 70)
    
    try:
        validate_behavior_patterns()
        validate_generator_basic()
        validate_circadian_patterns()
        validate_transitions()
        validate_sequence_generation()
        validate_daily_schedule()
        validate_sensor_ranges()
        validate_no_missing_data()
        
        print("\n" + "=" * 70)
        print("✅ ALL VALIDATION CHECKS PASSED")
        print("=" * 70)
        print("\nThe implementation is ready to generate datasets.")
        print("Run: python src/data/dataset_generator.py")
        print("Or: ./generate_datasets.sh (Linux/macOS)")
        print("Or: generate_datasets.bat (Windows)")
        print()
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
