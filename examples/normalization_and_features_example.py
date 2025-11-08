"""
Normalization and Feature Engineering Pipeline Example

This example demonstrates the complete pipeline for normalizing sensor data
and engineering features for machine learning models.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processing import (
    normalize_temperature,
    standardize_acceleration,
    standardize_angular_velocity,
    normalize_sensor_data,
    calculate_motion_intensity,
    calculate_pitch_angle,
    calculate_roll_angle,
    calculate_activity_score,
    calculate_postural_stability,
    calculate_head_movement_intensity,
    extract_rhythmic_features,
    engineer_features,
    create_feature_vector
)


def example_1_basic_normalization():
    """Example 1: Basic normalization of individual sensors."""
    print("=" * 70)
    print("Example 1: Basic Sensor Normalization")
    print("=" * 70)
    
    # Temperature normalization (35-42°C -> 0-1)
    print("\n1. Temperature Normalization:")
    temps = [35.0, 38.5, 42.0, 37.0, 40.5]
    for temp in temps:
        norm_temp = normalize_temperature(temp)
        print(f"  {temp}°C -> {norm_temp:.3f}")
    
    # Acceleration standardization (z-score)
    print("\n2. Acceleration Standardization:")
    accels = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for accel in accels:
        std_accel = standardize_acceleration(accel)
        print(f"  {accel:+.1f}g -> {std_accel:+.3f} (z-score)")
    
    # Angular velocity standardization
    print("\n3. Angular Velocity Standardization:")
    gyros = [-40.0, -20.0, 0.0, 20.0, 40.0]
    for gyro in gyros:
        std_gyro = standardize_angular_velocity(gyro)
        print(f"  {gyro:+.1f}°/s -> {std_gyro:+.3f} (z-score)")


def example_2_batch_normalization():
    """Example 2: Normalize complete sensor dataset."""
    print("\n" + "=" * 70)
    print("Example 2: Batch Normalization of Sensor Data")
    print("=" * 70)
    
    # Create sample sensor data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
        'animal_id': 'cow_001',
        'temperature': [38.5, 38.7, 38.3, 38.9, 38.6, 38.4, 38.8, 38.5, 38.7, 38.6],
        'fxa': [0.0, 0.4, -0.2, 0.5, 0.1, -0.1, 0.3, 0.0, 0.2, -0.3],
        'mya': [0.0, 0.2, -0.1, 0.3, 0.05, -0.05, 0.15, 0.0, 0.1, -0.15],
        'rza': [0.9, 0.85, 0.88, 0.82, 0.9, 0.87, 0.85, 0.9, 0.88, 0.86],
        'sxg': [0.0, 5.0, -3.0, 8.0, 2.0, -2.0, 6.0, 0.0, 4.0, -4.0],
        'lyg': [0.0, 10.0, -5.0, 12.0, 3.0, -3.0, 8.0, 0.0, 6.0, -6.0],
        'dzg': [0.0, 3.0, -2.0, 5.0, 1.0, -1.0, 4.0, 0.0, 2.0, -2.0]
    })
    
    print("\nOriginal data (first 5 rows):")
    print(data[['temperature', 'fxa', 'mya', 'rza']].head())
    
    # Normalize all sensors
    normalized_data = normalize_sensor_data(data)
    
    print("\nNormalized data (first 5 rows):")
    print(normalized_data[['temperature_norm', 'fxa_std', 'mya_std', 'rza_std']].head())
    
    print("\nNormalized temperature statistics:")
    print(f"  Min: {normalized_data['temperature_norm'].min():.3f}")
    print(f"  Max: {normalized_data['temperature_norm'].max():.3f}")
    print(f"  Mean: {normalized_data['temperature_norm'].mean():.3f}")


def example_3_motion_features():
    """Example 3: Calculate motion-based features."""
    print("\n" + "=" * 70)
    print("Example 3: Motion-Based Feature Engineering")
    print("=" * 70)
    
    # Simulate different behavioral states
    behaviors = {
        'Lying': {'fxa': 0.05, 'mya': 0.05, 'rza': -0.8},
        'Standing': {'fxa': 0.0, 'mya': 0.0, 'rza': 0.9},
        'Walking': {'fxa': 0.4, 'mya': 0.2, 'rza': 0.85},
        'Feeding': {'fxa': 0.2, 'mya': 0.1, 'rza': 0.65}
    }
    
    print("\nMotion Intensity by Behavior:")
    for behavior, accel in behaviors.items():
        intensity = calculate_motion_intensity(
            accel['fxa'], accel['mya'], accel['rza']
        )
        print(f"  {behavior:12s}: {intensity:.3f}g")
    
    print("\nActivity Score by Behavior (weighted):")
    for behavior, accel in behaviors.items():
        score = calculate_activity_score(
            accel['fxa'], accel['mya'], accel['rza']
        )
        print(f"  {behavior:12s}: {score:.3f}")


def example_4_orientation_features():
    """Example 4: Calculate orientation angles."""
    print("\n" + "=" * 70)
    print("Example 4: Orientation Angle Calculation")
    print("=" * 70)
    
    behaviors = {
        'Lying': {'rza': -0.8, 'fxa': 0.05, 'mya': 0.05},
        'Standing': {'rza': 0.9, 'fxa': 0.0, 'mya': 0.0},
        'Walking': {'rza': 0.85, 'fxa': 0.4, 'mya': 0.2},
        'Feeding (head down)': {'rza': 0.65, 'fxa': 0.2, 'mya': 0.1}
    }
    
    print("\nPitch Angle by Behavior:")
    for behavior, accel in behaviors.items():
        pitch_rad = calculate_pitch_angle(accel['rza'])
        pitch_deg = np.degrees(pitch_rad)
        print(f"  {behavior:20s}: {pitch_deg:+6.1f}°")
    
    print("\nRoll Angle by Behavior:")
    for behavior, accel in behaviors.items():
        roll_rad = calculate_roll_angle(accel['fxa'], accel['mya'])
        roll_deg = np.degrees(roll_rad)
        print(f"  {behavior:20s}: {roll_deg:+6.1f}°")


def example_5_rhythmic_features():
    """Example 5: Extract rhythmic features for rumination detection."""
    print("\n" + "=" * 70)
    print("Example 5: Rhythmic Pattern Feature Extraction")
    print("=" * 70)
    
    # Simulate rumination pattern (50 cycles/min = 0.83 Hz)
    print("\nSimulating rumination pattern (50 cycles/min)...")
    t = np.linspace(0, 60, 60)  # 60 seconds at 1 Hz sampling
    rumination_signal = np.sin(2 * np.pi * 0.83 * t) * 0.12  # Mya oscillations
    
    features = extract_rhythmic_features(
        rumination_signal,
        sampling_rate=1.0,
        target_freq_range=(0.67, 1.0)  # 40-60 cycles/min
    )
    
    print("\nExtracted Features:")
    print(f"  Dominant Frequency: {features['dominant_frequency']:.3f} Hz ({features['dominant_frequency']*60:.0f} cycles/min)")
    print(f"  Spectral Power: {features['spectral_power']:.3f}")
    print(f"  Zero Crossing Rate: {features['zero_crossing_rate']:.3f}")
    print(f"  Peak Count: {features['peak_count']}")
    print(f"  Regularity Score: {features['regularity_score']:.3f}")
    
    # Compare with non-rhythmic signal (standing)
    print("\nSimulating standing (no rhythmic pattern)...")
    standing_signal = np.random.normal(0, 0.04, 60)  # Random noise
    
    features_standing = extract_rhythmic_features(
        standing_signal,
        sampling_rate=1.0,
        target_freq_range=(0.67, 1.0)
    )
    
    print("\nExtracted Features (Standing):")
    print(f"  Dominant Frequency: {features_standing['dominant_frequency']:.3f} Hz")
    print(f"  Spectral Power: {features_standing['spectral_power']:.3f}")
    print(f"  Regularity Score: {features_standing['regularity_score']:.3f}")


def example_6_complete_pipeline():
    """Example 6: Complete pipeline from raw data to ML-ready features."""
    print("\n" + "=" * 70)
    print("Example 6: Complete Feature Engineering Pipeline")
    print("=" * 70)
    
    # Create realistic sensor data (simulated walking behavior)
    print("\nGenerating simulated walking behavior data...")
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'temperature': np.random.normal(38.7, 0.2, n_samples),
        'fxa': np.random.normal(0.4, 0.15, n_samples),
        'mya': np.random.normal(0.0, 0.1, n_samples),
        'rza': np.random.normal(0.85, 0.08, n_samples),
        'sxg': np.random.normal(0.0, 6.0, n_samples),
        'lyg': np.random.normal(0.0, 5.0, n_samples),
        'dzg': np.random.normal(0.0, 4.0, n_samples)
    })
    
    print(f"Generated {len(data)} samples")
    
    # Step 1: Normalize sensor data
    print("\nStep 1: Normalizing sensor data...")
    normalized = normalize_sensor_data(data)
    print(f"  Created {len([c for c in normalized.columns if c.endswith('_norm') or c.endswith('_std')])} normalized features")
    
    # Step 2: Engineer features
    print("\nStep 2: Engineering derived features...")
    features = engineer_features(normalized, window_size=10, include_rhythmic=False)
    print(f"  Total columns: {len(features.columns)}")
    print(f"  Engineered features: {[c for c in features.columns if c not in data.columns and not c.endswith('_norm') and not c.endswith('_std')]}")
    
    # Step 3: Create ML-ready feature vector
    print("\nStep 3: Creating ML-ready feature vectors...")
    feature_vector = create_feature_vector(features, include_raw_normalized=True)
    print(f"  Feature vector shape: {feature_vector.shape}")
    print(f"  Features: {list(feature_vector.columns)}")
    
    # Step 4: Display feature statistics
    print("\nFeature Statistics:")
    print(feature_vector.describe().T[['mean', 'std', 'min', 'max']])
    
    # Verify scikit-learn compatibility
    print("\nVerifying scikit-learn compatibility...")
    X = feature_vector.values
    print(f"  Array shape: {X.shape}")
    print(f"  Data type: {X.dtype}")
    print(f"  Contains NaN: {np.any(np.isnan(X))}")
    print(f"  Contains Inf: {np.any(np.isinf(X))}")
    print("  ✓ Ready for ML models!")
    
    return feature_vector


def example_7_behavioral_comparison():
    """Example 7: Compare features across different behaviors."""
    print("\n" + "=" * 70)
    print("Example 7: Feature Comparison Across Behaviors")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 50
    
    behaviors = ['Lying', 'Standing', 'Walking']
    behavior_params = {
        'Lying': {
            'temp': (38.5, 0.2), 'fxa': (0.0, 0.05), 'mya': (0.0, 0.04),
            'rza': (-0.8, 0.1), 'sxg': (0.0, 2.0), 'lyg': (0.0, 2.0), 'dzg': (0.0, 1.5)
        },
        'Standing': {
            'temp': (38.6, 0.2), 'fxa': (0.0, 0.04), 'mya': (0.0, 0.04),
            'rza': (0.9, 0.08), 'sxg': (0.0, 3.0), 'lyg': (0.0, 3.0), 'dzg': (0.0, 2.5)
        },
        'Walking': {
            'temp': (38.7, 0.2), 'fxa': (0.4, 0.15), 'mya': (0.0, 0.1),
            'rza': (0.85, 0.08), 'sxg': (0.0, 6.0), 'lyg': (0.0, 5.0), 'dzg': (0.0, 4.0)
        }
    }
    
    results = {}
    
    for behavior, params in behavior_params.items():
        data = pd.DataFrame({
            'temperature': np.random.normal(params['temp'][0], params['temp'][1], n_samples),
            'fxa': np.random.normal(params['fxa'][0], params['fxa'][1], n_samples),
            'mya': np.random.normal(params['mya'][0], params['mya'][1], n_samples),
            'rza': np.random.normal(params['rza'][0], params['rza'][1], n_samples),
            'sxg': np.random.normal(params['sxg'][0], params['sxg'][1], n_samples),
            'lyg': np.random.normal(params['lyg'][0], params['lyg'][1], n_samples),
            'dzg': np.random.normal(params['dzg'][0], params['dzg'][1], n_samples)
        })
        
        features = engineer_features(data, include_rhythmic=False)
        results[behavior] = features
    
    # Compare key features
    print("\nKey Feature Comparison:")
    print(f"{'Feature':25s} {'Lying':>12s} {'Standing':>12s} {'Walking':>12s}")
    print("-" * 65)
    
    feature_names = [
        'motion_intensity',
        'pitch_angle',
        'activity_score',
        'postural_stability',
        'head_movement_intensity'
    ]
    
    for feat in feature_names:
        values = [results[b][feat].mean() for b in behaviors]
        print(f"{feat:25s} {values[0]:12.4f} {values[1]:12.4f} {values[2]:12.4f}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 8 + "NORMALIZATION AND FEATURE ENGINEERING EXAMPLES" + " " * 14 + "║")
    print("╚" + "=" * 68 + "╝")
    
    try:
        example_1_basic_normalization()
        example_2_batch_normalization()
        example_3_motion_features()
        example_4_orientation_features()
        example_5_rhythmic_features()
        feature_vector = example_6_complete_pipeline()
        example_7_behavioral_comparison()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Use feature vectors for training ML models (Random Forest, SVM)")
        print("  2. Evaluate feature importance for behavior classification")
        print("  3. Optimize feature selection based on model performance")
        print("  4. Integrate with real-time monitoring pipeline")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
