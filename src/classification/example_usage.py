"""
Hybrid Classification Pipeline - Example Usage

This script demonstrates various use cases for the integrated behavioral
classification pipeline.

Examples:
1. Basic batch classification
2. Custom configuration
3. Stress detection calibration
4. Real-time processing simulation
5. Export and visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from classification.hybrid_pipeline import HybridClassificationPipeline
from classification.stress_detector import StressDetector
from classification.state_transition_smoother import StateTransitionSmoother


def generate_sample_data(n_samples=60, behavior='mixed'):
    """
    Generate sample sensor data for testing.
    
    Args:
        n_samples: Number of 1-minute samples to generate
        behavior: Type of behavior ('lying', 'standing', 'walking', 'mixed')
    
    Returns:
        DataFrame with sensor readings
    """
    np.random.seed(42)
    
    # Base patterns for different behaviors
    if behavior == 'lying':
        fxa_pattern = np.random.normal(0, 0.05, n_samples)
        mya_pattern = np.random.normal(0, 0.05, n_samples)
        rza_pattern = np.random.normal(-0.7, 0.1, n_samples)
    elif behavior == 'standing':
        fxa_pattern = np.random.normal(0, 0.1, n_samples)
        mya_pattern = np.random.normal(0, 0.1, n_samples)
        rza_pattern = np.random.normal(0.8, 0.05, n_samples)
    elif behavior == 'walking':
        fxa_pattern = np.random.normal(0.3, 0.15, n_samples)
        mya_pattern = np.random.normal(0, 0.12, n_samples)
        rza_pattern = np.random.normal(0.6, 0.15, n_samples)
    else:  # mixed
        # Create segments of different behaviors
        segment_size = n_samples // 3
        fxa_pattern = np.concatenate([
            np.random.normal(0, 0.05, segment_size),      # Lying
            np.random.normal(0, 0.1, segment_size),       # Standing
            np.random.normal(0.3, 0.15, n_samples - 2*segment_size)  # Walking
        ])
        mya_pattern = np.random.normal(0, 0.1, n_samples)
        rza_pattern = np.concatenate([
            np.random.normal(-0.7, 0.1, segment_size),    # Lying
            np.random.normal(0.8, 0.05, segment_size),    # Standing
            np.random.normal(0.6, 0.15, n_samples - 2*segment_size)  # Walking
        ])
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'temperature': np.random.uniform(37.5, 39.0, n_samples),
        'fxa': fxa_pattern,
        'mya': mya_pattern,
        'rza': rza_pattern,
        'sxg': np.random.normal(0, 10, n_samples),
        'lyg': np.random.normal(0, 10, n_samples),
        'dzg': np.random.normal(0, 10, n_samples)
    })
    
    return data


def example_1_basic_classification():
    """Example 1: Basic batch classification."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Batch Classification")
    print("="*60)
    
    # Generate sample data
    sensor_data = generate_sample_data(n_samples=30, behavior='mixed')
    print(f"\nGenerated {len(sensor_data)} samples of sensor data")
    
    # Initialize pipeline with defaults
    pipeline = HybridClassificationPipeline()
    print("Pipeline initialized with default configuration")
    
    # Classify
    print("\nClassifying behavioral states...")
    results = pipeline.classify_batch(sensor_data)
    
    # Display results
    print("\nClassification Results:")
    print(results[['timestamp', 'state', 'confidence', 'is_stressed', 'classification_source']].head(10))
    
    # Summary statistics
    print("\nState Distribution:")
    print(results['state'].value_counts())
    
    print("\nAverage Confidence by State:")
    print(results.groupby('state')['confidence'].mean())
    
    # Pipeline statistics
    stats = pipeline.get_statistics()
    print(f"\nProcessing Performance:")
    print(f"  Total samples: {stats['total_classifications']}")
    print(f"  Avg time per sample: {stats['avg_time_per_sample_ms']:.2f}ms")


def example_2_custom_configuration():
    """Example 2: Using custom configuration."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Create custom config
    config_path = Path(__file__).parent / 'pipeline_config.yaml'
    
    # Initialize with custom config
    pipeline = HybridClassificationPipeline(config_path=str(config_path))
    print(f"Pipeline initialized with config: {config_path.name}")
    
    # Generate data
    sensor_data = generate_sample_data(n_samples=20, behavior='standing')
    
    # Classify
    results = pipeline.classify_batch(sensor_data)
    
    print("\nResults with custom configuration:")
    print(results[['state', 'confidence', 'smoothing_applied']].head(10))
    
    # Show config details
    print("\nActive Configuration:")
    print(f"  Min duration: {pipeline.config['smoother']['min_duration']}")
    print(f"  Window size: {pipeline.config['smoother']['window_size']}")
    print(f"  Confidence threshold: {pipeline.config['smoother']['confidence_threshold']}")


def example_3_stress_detection():
    """Example 3: Stress detection with calibration."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Stress Detection with Calibration")
    print("="*60)
    
    # Generate normal behavior baseline
    print("\nGenerating normal behavior baseline data...")
    normal_data = generate_sample_data(n_samples=100, behavior='lying')
    
    # Initialize pipeline
    pipeline = HybridClassificationPipeline()
    
    # Calibrate stress detector
    print("Calibrating stress detector on normal behavior...")
    pipeline.stress_detector.calibrate(normal_data)
    
    # Generate test data with stress
    print("\nGenerating test data with stress patterns...")
    np.random.seed(123)
    stressed_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-02', periods=20, freq='1min'),
        'temperature': np.random.uniform(38.5, 40.0, 20),  # Elevated temp
        'fxa': np.random.normal(0, 0.8, 20),  # High variance
        'mya': np.random.normal(0, 0.8, 20),  # High variance
        'rza': np.random.normal(0.5, 0.6, 20),  # Erratic posture
        'sxg': np.random.normal(0, 80, 20),  # High angular velocity
        'lyg': np.random.normal(0, 80, 20),
        'dzg': np.random.normal(0, 80, 20)
    })
    
    # Classify
    results = pipeline.classify_batch(stressed_data)
    
    # Show stress detection results
    print("\nStress Detection Results:")
    print(results[['timestamp', 'state', 'is_stressed', 'stress_score']].head(15))
    
    stressed_count = results['is_stressed'].sum()
    print(f"\nStressed samples detected: {stressed_count}/{len(results)} ({100*stressed_count/len(results):.1f}%)")


def example_4_state_smoothing():
    """Example 4: State transition smoothing comparison."""
    print("\n" + "="*60)
    print("EXAMPLE 4: State Transition Smoothing")
    print("="*60)
    
    # Generate jittery data
    np.random.seed(456)
    n_samples = 20
    
    # Create alternating states (artificial jitter)
    states_raw = ['lying' if i % 3 == 0 else 'standing' for i in range(n_samples)]
    
    sensor_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'temperature': np.random.uniform(37.5, 39.0, n_samples),
        'fxa': [0.05 if s == 'lying' else 0.1 for s in states_raw],
        'mya': np.random.normal(0, 0.1, n_samples),
        'rza': [-0.7 if s == 'lying' else 0.8 for s in states_raw],
        'sxg': np.random.normal(0, 10, n_samples),
        'lyg': np.random.normal(0, 10, n_samples),
        'dzg': np.random.normal(0, 10, n_samples)
    })
    
    # Classify WITHOUT smoothing
    pipeline_no_smooth = HybridClassificationPipeline()
    pipeline_no_smooth.config['pipeline']['enable_smoothing'] = False
    results_no_smooth = pipeline_no_smooth.classify_batch(sensor_data)
    
    # Classify WITH smoothing
    pipeline_smooth = HybridClassificationPipeline()
    results_smooth = pipeline_smooth.classify_batch(sensor_data)
    
    # Compare
    print("\nComparison of Raw vs Smoothed States:")
    comparison = pd.DataFrame({
        'timestamp': sensor_data['timestamp'],
        'raw_state': results_no_smooth['state'],
        'smoothed_state': results_smooth['state'],
        'smoothing_applied': results_smooth['smoothing_applied']
    })
    print(comparison.head(15))
    
    # Count state transitions
    transitions_raw = (results_no_smooth['state'] != results_no_smooth['state'].shift()).sum()
    transitions_smooth = (results_smooth['state'] != results_smooth['state'].shift()).sum()
    
    print(f"\nState transitions (jitter):")
    print(f"  Without smoothing: {transitions_raw}")
    print(f"  With smoothing: {transitions_smooth}")
    print(f"  Reduction: {100*(transitions_raw-transitions_smooth)/transitions_raw:.1f}%")


def example_5_export_results():
    """Example 5: Export and save results."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Export Results")
    print("="*60)
    
    # Generate data
    sensor_data = generate_sample_data(n_samples=50, behavior='mixed')
    
    # Initialize pipeline
    pipeline = HybridClassificationPipeline()
    
    # Classify
    results = pipeline.classify_batch(sensor_data)
    
    # Export to CSV
    output_dir = Path(__file__).parent.parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    csv_path = output_dir / 'example_behavioral_states.csv'
    pipeline.export_results(results, str(csv_path), format='csv')
    print(f"\nResults exported to: {csv_path}")
    
    # Export to JSON
    json_path = output_dir / 'example_behavioral_states.json'
    pipeline.export_results(results, str(json_path), format='json')
    print(f"Results exported to: {json_path}")
    
    # Show file sizes
    print(f"\nFile sizes:")
    print(f"  CSV: {csv_path.stat().st_size / 1024:.2f} KB")
    print(f"  JSON: {json_path.stat().st_size / 1024:.2f} KB")


def example_6_real_time_simulation():
    """Example 6: Simulate real-time processing."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Real-Time Processing Simulation")
    print("="*60)
    
    # Initialize pipeline
    pipeline = HybridClassificationPipeline()
    
    # Simulate streaming data (1-minute intervals)
    print("\nSimulating real-time data stream...")
    print("Processing samples as they arrive...\n")
    
    for minute in range(10):
        # Generate single sample
        sample = generate_sample_data(n_samples=1, behavior='mixed')
        sample['timestamp'] = pd.Timestamp.now() + timedelta(minutes=minute)
        
        # Classify
        result = pipeline.classify_batch(sample)
        
        # Display
        state = result['state'].iloc[0]
        confidence = result['confidence'].iloc[0]
        stressed = result['is_stressed'].iloc[0]
        
        print(f"Minute {minute+1:2d}: {state:12s} (conf={confidence:.2f}, stressed={stressed})")
    
    # Final statistics
    stats = pipeline.get_statistics()
    print(f"\nTotal processing time: {stats['avg_processing_time_seconds']:.3f}s")
    print(f"Average per sample: {stats['avg_time_per_sample_ms']:.2f}ms")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("HYBRID CLASSIFICATION PIPELINE - EXAMPLE USAGE")
    print("="*60)
    
    try:
        example_1_basic_classification()
        example_2_custom_configuration()
        example_3_stress_detection()
        example_4_state_smoothing()
        example_5_export_results()
        example_6_real_time_simulation()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
