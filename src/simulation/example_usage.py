"""
Example Usage of Behavioral State Simulation

This script demonstrates various ways to use the simulation engine
to generate realistic cattle sensor data.
"""

from datetime import datetime
import os

# Import simulation components
from engine import SimulationEngine, SimulationConfig
from state_params import AnimalProfile, BehavioralState
from transitions import StateTransitionConfig
from noise import NoiseParameters


def example_basic_simulation():
    """Example 1: Basic 24-hour simulation."""
    print("=" * 60)
    print("Example 1: Basic 24-hour Simulation")
    print("=" * 60)
    
    # Create engine with default settings
    engine = SimulationEngine(
        animal_id="cow_001",
        seed=42  # For reproducibility
    )
    
    # Run simulation
    print("Running 24-hour simulation...")
    data = engine.run_simulation(
        duration_hours=24,
        start_time=datetime(2024, 1, 1, 0, 0)
    )
    
    print(f"Generated {len(data)} data points")
    print("\nFirst few rows:")
    print(data.head())
    
    print("\nSummary statistics:")
    stats = engine.get_summary_statistics()
    print(f"  State distribution: {stats['state_distribution']}")
    print(f"  Transition percentage: {stats['transition_percentage']:.2f}%")
    
    # Save to CSV
    output_dir = "../../data/simulated"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/basic_simulation_24h.csv"
    engine.export_to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    
    return data


def example_multi_day_simulation():
    """Example 2: Multi-day simulation showing circadian patterns."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-day Simulation (7 days)")
    print("=" * 60)
    
    engine = SimulationEngine(
        animal_id="cow_002",
        seed=123
    )
    
    print("Running 7-day simulation...")
    data = engine.run_multi_day_simulation(
        num_days=7,
        start_time=datetime(2024, 1, 1, 0, 0)
    )
    
    print(f"Generated {len(data)} data points ({len(data)/1440:.1f} days)")
    
    # Analyze daily patterns
    data['hour'] = data['timestamp'].dt.hour
    daily_temp = data.groupby('hour')['temperature'].mean()
    
    print("\nAverage temperature by hour:")
    print(f"  Lowest (4 AM): {daily_temp[4]:.2f}°C")
    print(f"  Highest (4 PM): {daily_temp[16]:.2f}°C")
    
    # Save to CSV
    output_dir = "../../data/simulated"
    output_file = f"{output_dir}/multi_day_simulation_7d.csv"
    engine.export_to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    
    return data


def example_custom_animal_profile():
    """Example 3: Simulation with custom animal profile."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Animal Profile")
    print("=" * 60)
    
    # Create custom profile for a more active, larger animal
    profile = AnimalProfile(
        animal_id="cow_003_active",
        baseline_temperature=38.7,  # Slightly higher baseline
        activity_multiplier=1.3,    # 30% more active
        body_size_factor=1.15,      # 15% larger
        age_category="adult"
    )
    
    print(f"Animal profile:")
    print(f"  Baseline temperature: {profile.baseline_temperature:.2f}°C")
    print(f"  Activity level: {profile.activity_multiplier * 100:.0f}%")
    print(f"  Body size: {profile.body_size_factor * 100:.0f}%")
    
    engine = SimulationEngine(
        animal_profile=profile,
        seed=456
    )
    
    print("\nRunning 48-hour simulation...")
    data = engine.run_simulation(duration_hours=48)
    
    print(f"\nGenerated statistics:")
    print(f"  Mean temperature: {data['temperature'].mean():.2f}°C")
    print(f"  Mean |Fxa|: {abs(data['fxa']).mean():.3f}g")
    
    # Save to CSV
    output_dir = "../../data/simulated"
    output_file = f"{output_dir}/custom_profile_48h.csv"
    engine.export_to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    
    return data


def example_sick_animal():
    """Example 4: Simulating a sick animal with fever."""
    print("\n" + "=" * 60)
    print("Example 4: Sick Animal Simulation")
    print("=" * 60)
    
    # Create profile for sick animal
    sick_profile = AnimalProfile(
        animal_id="cow_004_sick",
        baseline_temperature=38.5,
        fever_offset=1.5,        # +1.5°C fever
        lethargy_factor=0.5,     # 50% reduced activity
        age_category="adult"
    )
    
    print(f"Sick animal profile:")
    print(f"  Fever: +{sick_profile.fever_offset}°C")
    print(f"  Activity level: {sick_profile.lethargy_factor * 100:.0f}%")
    
    engine = SimulationEngine(
        animal_profile=sick_profile,
        seed=789
    )
    
    print("\nRunning 24-hour simulation...")
    data = engine.run_simulation(duration_hours=24)
    
    print(f"\nGenerated statistics:")
    print(f"  Mean temperature: {data['temperature'].mean():.2f}°C (elevated)")
    print(f"  Mean |Fxa|: {abs(data['fxa']).mean():.3f}g (reduced)")
    
    # Compare state distribution
    stats = engine.get_summary_statistics()
    print(f"\nState distribution:")
    for state, pct in stats['state_distribution'].items():
        print(f"  {state}: {pct:.1f}%")
    
    # Save to CSV
    output_dir = "../../data/simulated"
    output_file = f"{output_dir}/sick_animal_24h.csv"
    engine.export_to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    
    return data


def example_custom_configuration():
    """Example 5: Using custom simulation configuration."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)
    
    # Custom noise parameters (lower noise)
    noise_params = NoiseParameters(
        temperature_std=0.05,      # Half the default noise
        accelerometer_std=0.025,
        gyroscope_std=1.0
    )
    
    # Custom simulation config
    sim_config = SimulationConfig(
        time_step_minutes=1.0,
        include_validation=True,
        include_noise=True,
        include_temporal_effects=True,
        ambient_temperature=25.0  # 25°C ambient
    )
    
    print("Custom configuration:")
    print(f"  Temperature noise: ±{noise_params.temperature_std}°C")
    print(f"  Ambient temperature: {sim_config.ambient_temperature}°C")
    
    engine = SimulationEngine(
        animal_id="cow_005",
        noise_params=noise_params,
        sim_config=sim_config,
        seed=101
    )
    
    print("\nRunning 12-hour simulation...")
    data = engine.run_simulation(duration_hours=12)
    
    print(f"Generated {len(data)} data points")
    print(f"Validation warnings: {len(engine.validation_warnings)}")
    
    # Save to CSV
    output_dir = "../../data/simulated"
    output_file = f"{output_dir}/custom_config_12h.csv"
    engine.export_to_csv(output_file)
    print(f"\nData saved to: {output_file}")
    
    return data


def example_compare_healthy_vs_sick():
    """Example 6: Generate data for both healthy and sick animals."""
    print("\n" + "=" * 60)
    print("Example 6: Healthy vs Sick Animal Comparison")
    print("=" * 60)
    
    # Healthy animal
    healthy_profile = AnimalProfile(
        animal_id="cow_006_healthy",
        baseline_temperature=38.5,
        activity_multiplier=1.0
    )
    
    # Sick animal
    sick_profile = AnimalProfile(
        animal_id="cow_006_sick",
        baseline_temperature=38.5,
        fever_offset=1.2,
        lethargy_factor=0.6
    )
    
    # Run both simulations with same seed for comparison
    print("Generating healthy animal data...")
    healthy_engine = SimulationEngine(animal_profile=healthy_profile, seed=999)
    healthy_data = healthy_engine.run_simulation(duration_hours=24)
    
    print("Generating sick animal data...")
    sick_engine = SimulationEngine(animal_profile=sick_profile, seed=999)
    sick_data = sick_engine.run_simulation(duration_hours=24)
    
    # Compare statistics
    print("\nComparison:")
    print(f"  Healthy - Mean temp: {healthy_data['temperature'].mean():.2f}°C, "
          f"Mean |Fxa|: {abs(healthy_data['fxa']).mean():.3f}g")
    print(f"  Sick    - Mean temp: {sick_data['temperature'].mean():.2f}°C, "
          f"Mean |Fxa|: {abs(sick_data['fxa']).mean():.3f}g")
    
    # Save both datasets
    output_dir = "../../data/simulated"
    healthy_engine.export_to_csv(f"{output_dir}/healthy_animal_24h.csv")
    sick_engine.export_to_csv(f"{output_dir}/sick_animal_comparison_24h.csv")
    print(f"\nData saved to: {output_dir}/")
    
    return healthy_data, sick_data


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Behavioral State Simulation - Example Usage")
    print("=" * 60)
    
    # Run examples
    example_basic_simulation()
    example_multi_day_simulation()
    example_custom_animal_profile()
    example_sick_animal()
    example_custom_configuration()
    example_compare_healthy_vs_sick()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("Check the data/simulated/ directory for output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
