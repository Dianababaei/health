"""
Example Usage: Baseline Temperature Calculation System

Demonstrates the complete baseline calculation pipeline including:
- Circadian rhythm extraction
- Multi-window baseline calculation
- Dynamic updates with adaptive windowing
- Drift detection
- History management
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yaml
from pathlib import Path

from baseline_calculator import BaselineCalculator
from circadian_extractor import CircadianExtractor
from baseline_updater import (
    BaselineUpdater,
    BaselineDriftDetector,
    BaselineHistoryManager,
)


def generate_example_data(
    cow_id: int = 1,
    baseline_temp: float = 38.5,
    n_days: int = 30,
) -> pd.DataFrame:
    """
    Generate example temperature data with circadian pattern.
    
    Args:
        cow_id: Cow identifier
        baseline_temp: Baseline temperature (°C)
        n_days: Number of days to generate
        
    Returns:
        DataFrame with temperature data
    """
    print(f"Generating {n_days} days of synthetic temperature data...")
    
    # Generate minute-level data
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [
        start_time + timedelta(minutes=i)
        for i in range(n_days * 24 * 60)
    ]
    
    temperatures = []
    for ts in timestamps:
        hour_of_day = ts.hour + ts.minute / 60.0
        
        # Circadian component (±0.5°C, peak at 4 PM)
        phase = 2 * np.pi * (hour_of_day - 16.0) / 24.0
        circadian = -0.4 * np.cos(phase)
        
        # Add noise
        noise = np.random.normal(0, 0.1)
        
        temp = baseline_temp + circadian + noise
        temperatures.append(temp)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'cow_id': cow_id,
    })
    
    print(f"Generated {len(df)} temperature readings")
    return df


def example_1_basic_baseline_calculation():
    """Example 1: Basic baseline calculation with circadian extraction."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Baseline Calculation")
    print("=" * 70)
    
    # Generate data
    df = generate_example_data(cow_id=1, baseline_temp=38.6, n_days=7)
    
    # Initialize calculator
    calculator = BaselineCalculator(
        window_days=7,
        robust_method="trimmed_mean",
        fever_threshold=39.5,
    )
    
    # Calculate baseline
    print("\nCalculating baseline...")
    result = calculator.calculate_baseline(df, cow_id=1)
    
    # Print results
    print("\n--- Baseline Calculation Results ---")
    print(f"Cow ID: {result.cow_id}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Baseline Temperature: {result.baseline_temp:.3f}°C")
    print(f"Circadian Amplitude: {result.circadian_amplitude:.3f}°C")
    print(f"Calculation Window: {result.calculation_window_days} days")
    print(f"Samples Used: {result.samples_used:,}")
    print(f"Outliers Excluded: {result.outliers_excluded}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Method: {result.method}")
    
    # Validate baseline
    is_valid, warnings = calculator.validate_baseline(result.baseline_temp)
    print(f"\nBaseline Valid: {is_valid}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return result


def example_2_multi_window_calculation():
    """Example 2: Calculate baselines with multiple window sizes."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Window Baseline Calculation")
    print("=" * 70)
    
    # Generate 30 days of data
    df = generate_example_data(cow_id=1, baseline_temp=38.5, n_days=30)
    
    # Calculate baselines with different window sizes
    calculator = BaselineCalculator()
    results = calculator.calculate_baseline_multi_window(
        df, cow_id=1, window_days_list=[7, 14, 30]
    )
    
    print("\n--- Multi-Window Results ---")
    for window_days, result in results.items():
        print(f"\n{window_days}-day window:")
        print(f"  Baseline: {result.baseline_temp:.3f}°C")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Samples: {result.samples_used:,}")
    
    return results


def example_3_circadian_extraction_and_detrending():
    """Example 3: Detailed circadian extraction and detrending."""
    print("\n" + "=" * 70)
    print("Example 3: Circadian Rhythm Extraction")
    print("=" * 70)
    
    # Generate data
    df = generate_example_data(cow_id=1, baseline_temp=38.5, n_days=7)
    
    # Extract circadian profile
    extractor = CircadianExtractor(method="fourier", fourier_components=2)
    profile = extractor.extract_circadian_profile(df)
    
    print("\n--- Circadian Profile ---")
    print(f"Amplitude: {profile.amplitude:.3f}°C")
    print(f"Peak Hour: {profile.peak_hour:.1f}")
    print(f"Trough Hour: {profile.trough_hour:.1f}")
    print(f"Mean Temperature: {profile.mean_temp:.3f}°C")
    print(f"Confidence: {profile.confidence:.2f}")
    
    # Validate profile
    is_valid, warnings = extractor.validate_circadian_profile(profile)
    print(f"\nProfile Valid: {is_valid}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Detrend temperatures
    detrended_df = extractor.detrend_temperatures(df, profile)
    
    print("\n--- Detrending Results ---")
    print(f"Original Std Dev: {df['temperature'].std():.3f}°C")
    print(f"Detrended Std Dev: {detrended_df['detrended_temp'].std():.3f}°C")
    print(f"Variability Reduction: {(1 - detrended_df['detrended_temp'].std() / df['temperature'].std()) * 100:.1f}%")
    
    # Show hourly means
    print("\n--- Hourly Temperature Profile ---")
    print("Hour | Mean Temp | Samples")
    print("-----|-----------|--------")
    for i in range(0, 24, 3):  # Show every 3 hours
        print(f" {i:2d}  | {profile.hourly_means[i]:6.3f}°C | {profile.hourly_counts[i]:6d}")
    
    return profile


def example_4_drift_detection():
    """Example 4: Baseline drift detection."""
    print("\n" + "=" * 70)
    print("Example 4: Baseline Drift Detection")
    print("=" * 70)
    
    # Generate data with gradual drift (0.1°C per day)
    print("\nGenerating data with baseline drift (0.1°C/day)...")
    
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = []
    temperatures = []
    
    for day in range(14):
        for minute in range(24 * 60):
            ts = start_time + timedelta(days=day, minutes=minute)
            hour_of_day = ts.hour + ts.minute / 60.0
            
            # Drifting baseline
            baseline = 38.5 + 0.1 * day
            
            # Circadian component
            phase = 2 * np.pi * (hour_of_day - 16.0) / 24.0
            circadian = -0.4 * np.cos(phase)
            
            temp = baseline + circadian + np.random.normal(0, 0.1)
            
            timestamps.append(ts)
            temperatures.append(temp)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'cow_id': 1,
    })
    
    # Calculate baselines at different time points
    calculator = BaselineCalculator(window_days=7)
    
    baseline_history = []
    for day in range(7, 14):
        current_time = start_time + timedelta(days=day)
        result = calculator.calculate_baseline(df, cow_id=1, current_time=current_time)
        baseline_history.append({
            'timestamp': result.timestamp,
            'baseline_temp': result.baseline_temp,
        })
    
    baseline_df = pd.DataFrame(baseline_history)
    
    # Detect drift
    detector = BaselineDriftDetector(
        drift_threshold=0.5,
        drift_window_days=7,
        drift_method="linear_regression",
    )
    
    drift_detected, magnitude, confidence = detector.detect_drift(
        baseline_df, baseline_df['timestamp'].max()
    )
    
    print("\n--- Drift Detection Results ---")
    print(f"Drift Detected: {drift_detected}")
    print(f"Drift Magnitude: {magnitude:+.3f}°C over 7 days")
    print(f"Confidence: {confidence:.2f}")
    
    if drift_detected:
        print(f"\n⚠️  ALERT: Baseline drift exceeds threshold!")
        print(f"   Magnitude: {magnitude:+.3f}°C (threshold: ±0.5°C)")
        print(f"   This may indicate chronic illness or environmental change.")
    
    return drift_detected, magnitude


def example_5_dynamic_updates_with_history():
    """Example 5: Dynamic baseline updates with history management."""
    print("\n" + "=" * 70)
    print("Example 5: Dynamic Baseline Updates with History")
    print("=" * 70)
    
    # Setup history manager
    history_manager = BaselineHistoryManager(
        storage_backend="json",
        storage_path="data/baseline_history_example",
        retain_days=180,
    )
    
    # Setup updater
    updater = BaselineUpdater(
        history_manager=history_manager,
        adaptive_windowing=True,
        initial_window_days=7,
        expand_after_days=14,
        smoothing_alpha=0.3,
    )
    
    # Generate data
    df = generate_example_data(cow_id=1, baseline_temp=38.5, n_days=30)
    
    # Perform updates at different time points
    print("\nPerforming baseline updates...")
    
    update_days = [7, 10, 14, 18, 22, 26, 30]
    results = []
    
    for day in update_days:
        current_time = df['timestamp'].min() + timedelta(days=day)
        result = updater.update_baseline(
            df, cow_id=1, current_time=current_time, force_update=True
        )
        results.append(result)
        
        print(f"\nDay {day}:")
        print(f"  Window: {result.calculation_window_days} days")
        print(f"  Baseline: {result.baseline_temp:.3f}°C")
        print(f"  Confidence: {result.confidence_score:.2f}")
    
    # Retrieve and display history
    print("\n--- Baseline History ---")
    history = history_manager.retrieve_history(cow_id=1)
    print(f"Total stored baselines: {len(history)}")
    
    if not history.empty:
        print("\nRecent history:")
        for idx, row in history.tail(5).iterrows():
            print(f"  {row['timestamp']}: {row['baseline_temp']:.3f}°C "
                  f"(window={row['calculation_window_days']}d)")
    
    # Get current baseline
    current_baseline = updater.get_current_baseline(cow_id=1)
    print(f"\nCurrent Baseline: {current_baseline:.3f}°C")
    
    return results


def example_6_load_configuration():
    """Example 6: Load configuration from YAML file."""
    print("\n" + "=" * 70)
    print("Example 6: Loading Configuration from YAML")
    print("=" * 70)
    
    config_path = Path(__file__).parent.parent.parent / "config" / "baseline_config.yaml"
    
    if not config_path.exists():
        print(f"\nConfiguration file not found: {config_path}")
        return None
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n--- Configuration Loaded ---")
    print(f"Rolling Windows: {config['rolling_windows']['short_window_days']}, "
          f"{config['rolling_windows']['medium_window_days']}, "
          f"{config['rolling_windows']['long_window_days']} days")
    print(f"Circadian Bins: {config['circadian']['hourly_bins']}")
    print(f"Robust Method: {config['robust_statistics']['method']}")
    print(f"Fever Threshold: {config['anomaly_exclusion']['fever_threshold']}°C")
    print(f"Drift Threshold: {config['drift_detection']['drift_threshold']}°C")
    print(f"Update Frequency: {config['dynamic_updates']['update_frequency_hours']} hours")
    
    # Initialize components with config
    calculator = BaselineCalculator(
        window_days=config['rolling_windows']['short_window_days'],
        robust_method=config['robust_statistics']['method'],
        fever_threshold=config['anomaly_exclusion']['fever_threshold'],
        trim_percentage=config['robust_statistics']['trim_percentage'],
    )
    
    print("\n✓ Calculator initialized with configuration")
    
    return config


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Baseline Temperature Calculation System - Examples")
    print("=" * 70)
    
    # Run examples
    example_1_basic_baseline_calculation()
    example_2_multi_window_calculation()
    example_3_circadian_extraction_and_detrending()
    example_4_drift_detection()
    example_5_dynamic_updates_with_history()
    example_6_load_configuration()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
