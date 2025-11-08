"""
Example usage of the cattle behavior simulation system.

This script demonstrates how to use the SimulationEngine to generate
realistic cattle sensor data for various purposes.
"""

from datetime import datetime
from simulation.engine import SimulationEngine, BatchSimulator
from simulation.transitions import BehaviorState


def example_continuous_simulation():
    """Example: Generate continuous sensor data for 24 hours."""
    print("=" * 60)
    print("Example 1: Continuous 24-hour simulation")
    print("=" * 60)
    
    # Initialize engine
    engine = SimulationEngine(
        baseline_temperature=38.5,
        sampling_rate=1.0,  # 1 sample per minute
        random_seed=42  # For reproducibility
    )
    
    # Generate 24 hours of data
    df = engine.generate_continuous_data(
        duration_hours=24,
        start_datetime=datetime(2024, 1, 1, 0, 0, 0),
        include_stress=True,
        stress_probability=0.05
    )
    
    print(f"\nGenerated {len(df)} samples")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nSensor statistics:")
    print(df[['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']].describe())
    
    # Export to CSV
    engine.export_to_csv(df, 'data/simulated/continuous_24h.csv')
    
    return df


def example_single_state():
    """Example: Generate data for a single behavioral state."""
    print("\n" + "=" * 60)
    print("Example 2: Single state generation (Walking)")
    print("=" * 60)
    
    engine = SimulationEngine(random_seed=42)
    
    # Generate 30 minutes of walking data
    df = engine.generate_single_state_data(
        state=BehaviorState.WALKING,
        duration_minutes=30,
        start_datetime=datetime.now()
    )
    
    print(f"\nGenerated {len(df)} samples of walking behavior")
    print(f"\nWalking data characteristics:")
    print(f"  Average Fxa (should show rhythmic pattern): {df['fxa'].mean():.3f}")
    print(f"  Fxa std dev (should be moderate): {df['fxa'].std():.3f}")
    print(f"  Average Rza (should be 0.7-0.9g): {df['rza'].mean():.3f}")
    
    return df


def example_labeled_dataset():
    """Example: Generate labeled dataset for ML training."""
    print("\n" + "=" * 60)
    print("Example 3: Labeled dataset for ML training")
    print("=" * 60)
    
    engine = SimulationEngine(random_seed=42)
    
    # Generate balanced dataset
    df = engine.generate_labeled_dataset(
        samples_per_state=50,  # 50 samples per state
        duration_per_sample_minutes=10,  # 10 minutes per sample
        include_stress=True
    )
    
    print(f"\nGenerated labeled dataset with {len(df)} total samples")
    print(f"\nState distribution:")
    print(df.groupby('state').size())
    
    # Export
    engine.export_to_csv(df, 'data/simulated/labeled_training_data.csv')
    
    return df


def example_multi_animal():
    """Example: Generate data for multiple animals."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-animal simulation")
    print("=" * 60)
    
    engine = SimulationEngine()
    batch = BatchSimulator(engine)
    
    # Generate data for 5 animals, 12 hours each
    df = batch.generate_multi_animal_dataset(
        num_animals=5,
        hours_per_animal=12,
        output_dir='data/simulated/multi_animal',
        individual_files=True
    )
    
    print(f"\nGenerated data for multiple animals")
    print(f"Animals: {df['animal_id'].nunique()}")
    print(f"Total samples: {len(df)}")
    
    return df


def example_state_comparison():
    """Example: Compare sensor signatures across different states."""
    print("\n" + "=" * 60)
    print("Example 5: State signature comparison")
    print("=" * 60)
    
    engine = SimulationEngine(random_seed=42)
    
    states_to_compare = [
        BehaviorState.LYING,
        BehaviorState.STANDING,
        BehaviorState.WALKING,
        BehaviorState.FEEDING,
    ]
    
    print("\nSensor signatures by state:")
    print("-" * 60)
    
    for state in states_to_compare:
        df = engine.generate_single_state_data(
            state=state,
            duration_minutes=30
        )
        
        print(f"\n{state.value.upper()}:")
        print(f"  Rza (vertical): {df['rza'].mean():.3f} ± {df['rza'].std():.3f}")
        print(f"  Fxa (forward):  {df['fxa'].mean():.3f} ± {df['fxa'].std():.3f}")
        print(f"  Mya (lateral):  {df['mya'].mean():.3f} ± {df['mya'].std():.3f}")
        print(f"  Lyg (pitch):    {df['lyg'].mean():.3f} ± {df['lyg'].std():.3f}")
        print(f"  Temp (°C):      {df['temperature'].mean():.2f} ± {df['temperature'].std():.2f}")


def example_validation():
    """Example: Validate generated data quality."""
    print("\n" + "=" * 60)
    print("Example 6: Data validation")
    print("=" * 60)
    
    engine = SimulationEngine(random_seed=42)
    
    # Generate some data
    df = engine.generate_continuous_data(
        duration_hours=6,
        include_stress=True
    )
    
    # Validate
    is_valid, warnings = engine.validate_generated_data(df)
    
    print(f"\nValidation result: {'PASSED' if is_valid else 'FAILED'}")
    if warnings:
        print("\nWarnings/Errors:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nNo warnings - data quality looks good!")
    
    # Get state statistics
    stats = engine.get_state_statistics(df)
    
    print("\nState statistics:")
    for state, state_stats in stats.items():
        print(f"\n{state}:")
        for key, value in state_stats.items():
            print(f"  {key}: {value:.2f}")


if __name__ == "__main__":
    """Run all examples."""
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "CATTLE BEHAVIOR SIMULATION EXAMPLES" + " " * 13 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run examples
    example_continuous_simulation()
    example_single_state()
    example_labeled_dataset()
    example_state_comparison()
    example_validation()
    
    # Optionally run multi-animal (takes longer)
    run_multi_animal = input("\nRun multi-animal simulation? (y/n): ")
    if run_multi_animal.lower() == 'y':
        example_multi_animal()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
