"""
Tests for synthetic data generator.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data"))

from datetime import datetime
import pandas as pd
from synthetic_generator import SyntheticDataGenerator, BEHAVIOR_PATTERNS


def test_behavior_patterns():
    """Test that all behavior patterns are defined."""
    expected_behaviors = ['lying', 'standing', 'walking', 'ruminating', 'feeding', 'stress']
    assert set(BEHAVIOR_PATTERNS.keys()) == set(expected_behaviors)
    print("✓ All 6 behavior patterns defined")


def test_generate_single_behavior():
    """Test generating data for a single behavior."""
    generator = SyntheticDataGenerator(seed=42)
    
    df = generator.generate_behavior_sample(
        behavior='lying',
        duration_minutes=10,
        start_time=datetime(2024, 1, 1),
        apply_circadian=True,
    )
    
    assert len(df) == 10
    assert 'timestamp' in df.columns
    assert 'temp' in df.columns
    assert 'behavior_label' in df.columns
    assert (df['behavior_label'] == 'lying').all()
    print(f"✓ Generated 10 minutes of 'lying' behavior")


def test_generate_transition():
    """Test generating behavior transition."""
    generator = SyntheticDataGenerator(seed=42)
    
    df = generator.generate_transition(
        from_behavior='lying',
        to_behavior='standing',
        start_time=datetime(2024, 1, 1),
        transition_minutes=2,
    )
    
    assert len(df) == 2
    assert (df['behavior_label'] == 'standing').all()
    print(f"✓ Generated transition from lying to standing")


def test_generate_sequence():
    """Test generating behavior sequence."""
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
    
    # Should have behavior samples + transitions
    assert len(df) > 12  # 5+3+4 + 2 transitions
    print(f"✓ Generated behavior sequence ({len(df)} samples)")


def test_generate_daily_schedule():
    """Test generating realistic daily schedule."""
    generator = SyntheticDataGenerator(seed=42)
    
    schedule = generator.generate_daily_schedule(datetime(2024, 1, 1))
    
    assert len(schedule) > 0
    assert all(isinstance(item, tuple) and len(item) == 2 for item in schedule)
    
    total_minutes = sum(duration for _, duration in schedule)
    # Should be close to 24 hours
    assert 1200 <= total_minutes <= 1600  # Reasonable range
    print(f"✓ Generated daily schedule ({len(schedule)} activities, {total_minutes} min)")


def test_circadian_patterns():
    """Test circadian pattern functions."""
    generator = SyntheticDataGenerator(seed=42)
    
    # Test activity factors
    day_factor = generator.circadian.get_hour_factor(12)  # Noon
    night_factor = generator.circadian.get_hour_factor(2)  # 2am
    
    assert day_factor > night_factor
    print(f"✓ Circadian activity: day={day_factor}, night={night_factor}")
    
    # Test temperature adjustment
    afternoon_temp = generator.circadian.get_temperature_adjustment(16)  # 4pm
    early_morning_temp = generator.circadian.get_temperature_adjustment(4)  # 4am
    
    assert afternoon_temp > early_morning_temp
    print(f"✓ Circadian temperature: afternoon={afternoon_temp:.2f}, morning={early_morning_temp:.2f}")


def test_sensor_value_ranges():
    """Test that generated sensor values are within expected ranges."""
    generator = SyntheticDataGenerator(seed=42)
    
    # Generate samples for each behavior
    for behavior in BEHAVIOR_PATTERNS.keys():
        df = generator.generate_behavior_sample(
            behavior=behavior,
            duration_minutes=50,
            start_time=datetime(2024, 1, 1),
        )
        
        # Check temperature range
        assert df['temp'].min() >= 37.0
        assert df['temp'].max() <= 41.0
        
        # Check acceleration range
        assert df['Fxa'].abs().max() <= 3.0
        assert df['Mya'].abs().max() <= 3.0
        assert df['Rza'].abs().max() <= 3.0
        
        # Check gyroscope range
        assert df['Sxg'].abs().max() <= 100
        assert df['Lyg'].abs().max() <= 100
        assert df['Dzg'].abs().max() <= 100
        
        print(f"✓ Sensor values for '{behavior}' within valid ranges")


def test_no_missing_values():
    """Test that generated data has no missing values."""
    generator = SyntheticDataGenerator(seed=42)
    
    df = generator.generate_behavior_sample(
        behavior='walking',
        duration_minutes=20,
        start_time=datetime(2024, 1, 1),
    )
    
    assert not df.isnull().any().any()
    print("✓ No missing values in generated data")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RUNNING DATA GENERATOR TESTS")
    print("=" * 60 + "\n")
    
    test_behavior_patterns()
    test_generate_single_behavior()
    test_generate_transition()
    test_generate_sequence()
    test_generate_daily_schedule()
    test_circadian_patterns()
    test_sensor_value_ranges()
    test_no_missing_values()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_tests()
