"""
Example Usage of Activity Metrics Module

Demonstrates various use cases and features of the ActivityTracker class.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from activity_metrics import ActivityTracker


def example_1_basic_usage():
    """Example 1: Basic behavioral log generation."""
    print("\n" + "="*60)
    print("Example 1: Basic Behavioral Log Generation")
    print("="*60)
    
    # Create sample data
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=20, freq='1min')
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': ['lying'] * 5 + ['standing'] * 5 + ['walking'] * 5 + ['feeding'] * 5,
        'fxa': np.random.randn(20) * 0.2,
        'mya': np.random.randn(20) * 0.1,
        'rza': np.random.randn(20) * 0.1 - 0.8
    })
    
    # Initialize tracker
    tracker = ActivityTracker()
    
    # Generate behavioral log
    log = tracker.generate_behavioral_log(data)
    
    print("\nBehavioral Log (first 5 rows):")
    print(log.head())
    
    print(f"\nTotal records: {len(log)}")
    print(f"Columns: {list(log.columns)}")


def example_2_duration_tracking():
    """Example 2: Track duration for consecutive states."""
    print("\n" + "="*60)
    print("Example 2: Duration Tracking")
    print("="*60)
    
    # Create data with known durations
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=30, freq='1min')
    states = ['lying'] * 10 + ['standing'] * 5 + ['walking'] * 8 + ['feeding'] * 7
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'fxa': np.random.randn(30) * 0.2,
        'mya': np.random.randn(30) * 0.1,
        'rza': np.random.randn(30) * 0.1 - 0.8
    })
    
    tracker = ActivityTracker()
    
    # Calculate durations
    result = tracker.calculate_durations(data)
    
    print("\nDuration Summary:")
    for state in tracker.BEHAVIORAL_STATES:
        state_data = result[result['behavioral_state'] == state]
        if not state_data.empty:
            duration = state_data['duration_minutes'].iloc[0]
            print(f"  {state.capitalize()}: {duration:.1f} minutes")


def example_3_hourly_aggregation():
    """Example 3: Hourly aggregation for 24 hours."""
    print("\n" + "="*60)
    print("Example 3: Hourly Aggregation")
    print("="*60)
    
    # Generate 24 hours of minute-level data
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=24*60, freq='1min')
    
    # Simulate circadian patterns (more lying at night)
    states = []
    for ts in timestamps:
        hour = ts.hour
        if hour >= 22 or hour < 6:  # Night
            state = np.random.choice(['lying', 'standing'], p=[0.8, 0.2])
        else:  # Day
            state = np.random.choice(
                ['lying', 'standing', 'walking', 'feeding', 'ruminating'],
                p=[0.2, 0.3, 0.2, 0.15, 0.15]
            )
        states.append(state)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'fxa': np.random.randn(len(timestamps)) * 0.2,
        'mya': np.random.randn(len(timestamps)) * 0.1,
        'rza': np.random.randn(len(timestamps)) * 0.1 - 0.8
    })
    
    tracker = ActivityTracker()
    
    # Aggregate by hour
    hourly = tracker.aggregate_hourly(data)
    
    print(f"\nHourly aggregation for {len(hourly)} hours")
    print("\nSample hours:")
    print(hourly[['hour', 'lying_minutes', 'standing_minutes', 'walking_minutes', 
                  'total_minutes', 'state_transitions']].head(5))
    
    # Find peak activity hour
    hourly['activity_score'] = (hourly['walking_minutes'] + 
                                 hourly['feeding_minutes'] + 
                                 hourly['ruminating_minutes'])
    peak_hour = hourly.loc[hourly['activity_score'].idxmax()]
    print(f"\nPeak activity hour: {peak_hour['hour']}")
    print(f"  Activity score: {peak_hour['activity_score']:.1f} minutes")


def example_4_daily_aggregation():
    """Example 4: Daily aggregation for multi-day dataset."""
    print("\n" + "="*60)
    print("Example 4: Daily Aggregation")
    print("="*60)
    
    # Generate 7 days of data
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=7*24*60, freq='1min')
    
    states = np.random.choice(
        ['lying', 'standing', 'walking', 'feeding', 'ruminating'],
        size=len(timestamps),
        p=[0.45, 0.20, 0.12, 0.10, 0.13]
    )
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'fxa': np.random.randn(len(timestamps)) * 0.2,
        'mya': np.random.randn(len(timestamps)) * 0.1,
        'rza': np.random.randn(len(timestamps)) * 0.1 - 0.8
    })
    
    tracker = ActivityTracker()
    
    # Aggregate by day
    daily = tracker.aggregate_daily(data)
    
    print(f"\nDaily aggregation for {len(daily)} days")
    print("\nDaily summary:")
    print(daily[['day', 'lying_minutes', 'standing_minutes', 'walking_minutes',
                 'total_minutes', 'rest_percentage', 'active_percentage']])
    
    # Validate daily totals
    validation = tracker.validate_daily_totals(daily)
    print(f"\nValidation: {'PASSED' if validation['all_valid'] else 'FAILED'}")
    if not validation['all_valid']:
        print(f"Invalid days: {len(validation['invalid_days'])}")


def example_5_movement_intensity():
    """Example 5: Movement intensity calculation and analysis."""
    print("\n" + "="*60)
    print("Example 5: Movement Intensity Analysis")
    print("="*60)
    
    # Generate data with varying intensity by state
    n_samples = 100
    states = []
    fxa_vals = []
    mya_vals = []
    rza_vals = []
    
    for _ in range(n_samples):
        state = np.random.choice(['lying', 'standing', 'walking'])
        states.append(state)
        
        if state == 'lying':
            fxa = np.random.normal(0.02, 0.01)
            mya = np.random.normal(0.01, 0.01)
            rza = np.random.normal(-0.85, 0.05)
        elif state == 'standing':
            fxa = np.random.normal(0.1, 0.05)
            mya = np.random.normal(0.05, 0.02)
            rza = np.random.normal(-0.80, 0.05)
        else:  # walking
            fxa = np.random.normal(0.5, 0.15)
            mya = np.random.normal(0.3, 0.1)
            rza = np.random.normal(-0.5, 0.1)
        
        fxa_vals.append(fxa)
        mya_vals.append(mya)
        rza_vals.append(rza)
    
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'behavioral_state': states,
        'fxa': fxa_vals,
        'mya': mya_vals,
        'rza': rza_vals
    })
    
    tracker = ActivityTracker()
    
    # Calculate movement intensity
    result = tracker.calculate_movement_intensity(data)
    
    # Analyze by state
    intensity_by_state = result.groupby('behavioral_state')['movement_intensity'].agg(['mean', 'std'])
    
    print("\nMovement Intensity by State:")
    print(intensity_by_state)
    
    print("\nIntensity correlation with activity level:")
    print(f"  Walking > Standing: {intensity_by_state.loc['walking', 'mean'] > intensity_by_state.loc['standing', 'mean']}")
    print(f"  Standing > Lying: {intensity_by_state.loc['standing', 'mean'] > intensity_by_state.loc['lying', 'mean']}")


def example_6_activity_rest_ratio():
    """Example 6: Calculate activity/rest ratios."""
    print("\n" + "="*60)
    print("Example 6: Activity/Rest Ratio Analysis")
    print("="*60)
    
    # Generate data with known distribution
    n_samples = 1000
    states = (
        ['lying'] * 450 +  # 45% rest
        ['standing'] * 200 +  # 20% active
        ['walking'] * 120 +  # 12% active
        ['feeding'] * 130 +  # 13% active
        ['ruminating'] * 100  # 10% active
    )
    np.random.shuffle(states)
    
    data = pd.DataFrame({
        'behavioral_state': states
    })
    
    tracker = ActivityTracker()
    
    # Calculate ratio
    ratio = tracker.calculate_activity_rest_ratio(data)
    
    print("\nActivity/Rest Ratio:")
    print(f"  Rest percentage: {ratio['rest_percentage']:.1f}%")
    print(f"  Active percentage: {ratio['active_percentage']:.1f}%")
    print(f"  Rest count: {ratio['rest_count']}")
    print(f"  Active count: {ratio['active_count']}")
    
    # Expected: 45% rest, 55% active
    print(f"\nExpected vs Actual:")
    print(f"  Rest: 45.0% vs {ratio['rest_percentage']:.1f}%")
    print(f"  Active: 55.0% vs {ratio['active_percentage']:.1f}%")


def example_7_state_transitions():
    """Example 7: Analyze state transition patterns."""
    print("\n" + "="*60)
    print("Example 7: State Transition Analysis")
    print("="*60)
    
    # Create data with known transition pattern
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1min')
    
    # Pattern: lying -> standing -> walking -> feeding -> ruminating -> lying (repeat)
    states = []
    pattern = ['lying'] * 20 + ['standing'] * 15 + ['walking'] * 10 + ['feeding'] * 15 + ['ruminating'] * 20
    states.extend(pattern)
    states.extend(pattern[:20])  # Add partial repeat
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states
    })
    
    tracker = ActivityTracker()
    
    # Count transitions
    transitions = tracker.count_state_transitions(data)
    
    print(f"\nTotal transitions: {transitions['total_transitions']}")
    
    print("\nTransition patterns:")
    for pattern, count in sorted(transitions['transition_matrix'].items(), 
                                 key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count}")


def example_8_full_pipeline():
    """Example 8: Complete pipeline with exports."""
    print("\n" + "="*60)
    print("Example 8: Full Pipeline Execution")
    print("="*60)
    
    # Generate realistic 3-day dataset
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=3*24*60, freq='1min')
    
    states = np.random.choice(
        ['lying', 'standing', 'walking', 'feeding', 'ruminating'],
        size=len(timestamps),
        p=[0.45, 0.20, 0.12, 0.10, 0.13]
    )
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'fxa': np.random.randn(len(timestamps)) * 0.2,
        'mya': np.random.randn(len(timestamps)) * 0.1,
        'rza': np.random.randn(len(timestamps)) * 0.1 - 0.8,
        'confidence_score': np.random.uniform(0.85, 0.99, len(timestamps))
    })
    
    tracker = ActivityTracker()
    
    # Run full pipeline
    results = tracker.process_full_pipeline(
        data,
        export_logs=True,
        export_hourly=True,
        export_daily=True,
        csv_format=True,
        json_format=True
    )
    
    print("\nPipeline Results:")
    print(f"  Behavioral log records: {len(results['behavioral_log'])}")
    print(f"  Hourly aggregations: {len(results['hourly_aggregation'])}")
    print(f"  Daily aggregations: {len(results['daily_aggregation'])}")
    
    print("\nActivity/Rest Ratio:")
    ratio = results['activity_rest_ratio']
    print(f"  Rest: {ratio['rest_percentage']:.1f}%")
    print(f"  Active: {ratio['active_percentage']:.1f}%")
    
    print("\nState Transitions:")
    trans = results['state_transitions']
    print(f"  Total transitions: {trans['total_transitions']}")
    
    print("\nExported Files:")
    for key, path in results['export_paths'].items():
        print(f"  {key}: {path}")


def example_9_edge_cases():
    """Example 9: Handle edge cases (missing data, single state)."""
    print("\n" + "="*60)
    print("Example 9: Edge Case Handling")
    print("="*60)
    
    tracker = ActivityTracker()
    
    # Case 1: Single state day (all lying)
    print("\nCase 1: Single-state day (sick animal)")
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': ['lying'] * 1440,
        'fxa': [0.01] * 1440,
        'mya': [0.01] * 1440,
        'rza': [-0.85] * 1440
    })
    
    ratio = tracker.calculate_activity_rest_ratio(data)
    print(f"  Rest percentage: {ratio['rest_percentage']:.1f}%")
    print(f"  Active percentage: {ratio['active_percentage']:.1f}%")
    
    # Case 2: Missing data windows
    print("\nCase 2: Missing data windows")
    timestamps = list(pd.date_range('2024-01-01 00:00:00', periods=10, freq='1min'))
    timestamps.extend(pd.date_range('2024-01-01 00:20:00', periods=10, freq='1min'))  # 10 min gap
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': ['lying'] * 10 + ['standing'] * 10
    })
    
    handled = tracker.handle_missing_data(data, fill_method='forward')
    print(f"  Original records: {len(data)}")
    print(f"  After handling: {len(handled)}")
    
    # Case 3: Empty dataframe
    print("\nCase 3: Empty dataframe")
    empty_df = pd.DataFrame(columns=['timestamp', 'behavioral_state', 'fxa', 'mya', 'rza'])
    ratio = tracker.calculate_activity_rest_ratio(empty_df)
    print(f"  Rest percentage: {ratio['rest_percentage']:.1f}%")
    print(f"  Total count: {ratio['total_count']}")


def example_10_performance_test():
    """Example 10: Performance test with 30 days of data."""
    print("\n" + "="*60)
    print("Example 10: Performance Test (30 days)")
    print("="*60)
    
    import time
    
    # Generate 30 days of minute-level data (43,200 records)
    print("\nGenerating 43,200 records (30 days)...")
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=30*24*60, freq='1min')
    
    states = np.random.choice(
        ['lying', 'standing', 'walking', 'feeding', 'ruminating'],
        size=len(timestamps),
        p=[0.45, 0.20, 0.12, 0.10, 0.13]
    )
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'fxa': np.random.randn(len(timestamps)) * 0.2,
        'mya': np.random.randn(len(timestamps)) * 0.1,
        'rza': np.random.randn(len(timestamps)) * 0.1 - 0.8
    })
    
    print(f"Dataset size: {len(data)} records")
    
    tracker = ActivityTracker()
    
    # Measure processing time
    start_time = time.time()
    
    results = tracker.process_full_pipeline(
        data,
        export_logs=False,
        export_hourly=False,
        export_daily=False
    )
    
    duration = time.time() - start_time
    
    print(f"\nProcessing time: {duration:.2f} seconds")
    print(f"Records per second: {len(data)/duration:.0f}")
    
    if duration < 10.0:
        print("✓ Performance criterion met (<10 seconds)")
    else:
        print("✗ Performance criterion not met (≥10 seconds)")
    
    print(f"\nResults:")
    print(f"  Behavioral log: {len(results['behavioral_log'])} records")
    print(f"  Hourly aggregations: {len(results['hourly_aggregation'])}")
    print(f"  Daily aggregations: {len(results['daily_aggregation'])}")


if __name__ == '__main__':
    """Run all examples."""
    print("\n" + "="*60)
    print("Activity Metrics Module - Example Usage")
    print("="*60)
    
    examples = [
        example_1_basic_usage,
        example_2_duration_tracking,
        example_3_hourly_aggregation,
        example_4_daily_aggregation,
        example_5_movement_intensity,
        example_6_activity_rest_ratio,
        example_7_state_transitions,
        example_8_full_pipeline,
        example_9_edge_cases,
        example_10_performance_test
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
