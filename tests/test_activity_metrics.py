"""
Unit tests for activity_metrics module.

Tests duration tracking, aggregations, movement intensity calculations,
activity ratios, state transitions, and export functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from layer1_behavior.activity_metrics import ActivityTracker


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def activity_tracker(temp_output_dir):
    """Create ActivityTracker instance with temporary output directory."""
    return ActivityTracker(output_dir=temp_output_dir)


@pytest.fixture
def simple_behavioral_data():
    """Create simple synthetic behavioral data."""
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=10, freq='1min')
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': ['lying', 'lying', 'standing', 'standing', 'walking',
                            'walking', 'walking', 'feeding', 'feeding', 'ruminating'],
        'fxa': [0.01, 0.02, 0.1, 0.15, 0.5, 0.6, 0.55, 0.3, 0.25, 0.1],
        'mya': [0.01, 0.02, 0.05, 0.08, 0.3, 0.4, 0.35, 0.2, 0.15, 0.05],
        'rza': [-0.85, -0.84, -0.82, -0.80, -0.5, -0.45, -0.48, -0.7, -0.72, -0.80]
    })
    return data


@pytest.fixture
def multi_day_behavioral_data():
    """Create multi-day synthetic behavioral data."""
    # Generate 3 days of minute-level data
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=3*24*60, freq='1min')
    
    # Create realistic behavioral patterns
    states = []
    for i in range(len(timestamps)):
        hour = timestamps[i].hour
        minute = timestamps[i].minute
        
        # Nighttime (22:00 - 06:00): mostly lying
        if hour >= 22 or hour < 6:
            if np.random.random() < 0.8:
                state = 'lying'
            else:
                state = 'standing'
        # Daytime: mixed activities
        else:
            rand = np.random.random()
            if rand < 0.3:
                state = 'standing'
            elif rand < 0.5:
                state = 'walking'
            elif rand < 0.7:
                state = 'feeding'
            elif rand < 0.85:
                state = 'ruminating'
            else:
                state = 'lying'
        
        states.append(state)
    
    # Generate accelerometer data based on state
    fxa_values = []
    mya_values = []
    rza_values = []
    
    for state in states:
        if state == 'lying':
            fxa = np.random.normal(0.02, 0.01)
            mya = np.random.normal(0.01, 0.01)
            rza = np.random.normal(-0.85, 0.05)
        elif state == 'standing':
            fxa = np.random.normal(0.1, 0.05)
            mya = np.random.normal(0.05, 0.02)
            rza = np.random.normal(-0.80, 0.05)
        elif state == 'walking':
            fxa = np.random.normal(0.5, 0.15)
            mya = np.random.normal(0.3, 0.1)
            rza = np.random.normal(-0.5, 0.1)
        elif state == 'feeding':
            fxa = np.random.normal(0.3, 0.1)
            mya = np.random.normal(0.2, 0.08)
            rza = np.random.normal(-0.7, 0.08)
        else:  # ruminating
            fxa = np.random.normal(0.15, 0.05)
            mya = np.random.normal(0.1, 0.04)
            rza = np.random.normal(-0.75, 0.05)
        
        fxa_values.append(fxa)
        mya_values.append(mya)
        rza_values.append(rza)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'fxa': fxa_values,
        'mya': mya_values,
        'rza': rza_values
    })
    
    return data


@pytest.fixture
def known_duration_data():
    """Create data with known consecutive state durations for validation."""
    # Create known pattern: 5 min lying, 3 min standing, 4 min walking, 2 min feeding
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=14, freq='1min')
    states = (['lying'] * 5 + ['standing'] * 3 + ['walking'] * 4 + ['feeding'] * 2)
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'fxa': np.random.randn(14) * 0.1,
        'mya': np.random.randn(14) * 0.1,
        'rza': np.random.randn(14) * 0.1 - 0.8
    })
    
    return data


class TestDurationCalculator:
    """Test duration calculation functionality."""
    
    def test_calculate_durations_basic(self, activity_tracker, simple_behavioral_data):
        """Test basic duration calculation."""
        result = activity_tracker.calculate_durations(simple_behavioral_data)
        
        assert 'duration_minutes' in result.columns
        assert len(result) == len(simple_behavioral_data)
        
        # Verify durations are positive
        assert (result['duration_minutes'] > 0).all()
    
    def test_calculate_durations_known_values(self, activity_tracker, known_duration_data):
        """Test duration calculation with known consecutive states."""
        result = activity_tracker.calculate_durations(known_duration_data)
        
        # First group (lying): 5 minutes
        lying_rows = result[result['behavioral_state'] == 'lying']
        assert all(lying_rows['duration_minutes'] == 5.0)
        
        # Second group (standing): 3 minutes
        standing_rows = result[result['behavioral_state'] == 'standing']
        assert all(standing_rows['duration_minutes'] == 3.0)
        
        # Third group (walking): 4 minutes
        walking_rows = result[result['behavioral_state'] == 'walking']
        assert all(walking_rows['duration_minutes'] == 4.0)
        
        # Fourth group (feeding): 2 minutes
        feeding_rows = result[result['behavioral_state'] == 'feeding']
        assert all(feeding_rows['duration_minutes'] == 2.0)
    
    def test_calculate_durations_single_state(self, activity_tracker):
        """Test duration calculation when all readings are same state."""
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=60, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'behavioral_state': ['lying'] * 60,
            'fxa': [0.01] * 60,
            'mya': [0.01] * 60,
            'rza': [-0.85] * 60
        })
        
        result = activity_tracker.calculate_durations(data)
        
        # All rows should have same duration (60 minutes)
        assert all(result['duration_minutes'] == 60.0)


class TestMovementIntensity:
    """Test movement intensity calculation."""
    
    def test_calculate_movement_intensity(self, activity_tracker, simple_behavioral_data):
        """Test basic movement intensity calculation."""
        result = activity_tracker.calculate_movement_intensity(simple_behavioral_data)
        
        assert 'movement_intensity' in result.columns
        assert len(result) == len(simple_behavioral_data)
        
        # Verify intensities are non-negative
        assert (result['movement_intensity'] >= 0).all()
    
    def test_movement_intensity_magnitude(self, activity_tracker):
        """Test that movement intensity correctly calculates magnitude."""
        data = pd.DataFrame({
            'fxa': [3.0, 0.0],
            'mya': [4.0, 1.0],
            'rza': [0.0, 0.0]
        })
        
        result = activity_tracker.calculate_movement_intensity(data)
        
        # First row: sqrt(3^2 + 4^2 + 0^2) = 5.0
        assert np.isclose(result['movement_intensity'].iloc[0], 5.0)
        
        # Second row: sqrt(0^2 + 1^2 + 0^2) = 1.0
        assert np.isclose(result['movement_intensity'].iloc[1], 1.0)
    
    def test_movement_intensity_correlates_with_activity(self, activity_tracker, multi_day_behavioral_data):
        """Test that movement intensity correlates with expected activity levels."""
        result = activity_tracker.calculate_movement_intensity(multi_day_behavioral_data)
        
        # Calculate average intensity per state
        state_intensities = result.groupby('behavioral_state')['movement_intensity'].mean()
        
        # Walking should have higher intensity than lying
        assert state_intensities['walking'] > state_intensities['lying']
        
        # Standing should have higher intensity than lying
        assert state_intensities['standing'] > state_intensities['lying']


class TestHourlyAggregation:
    """Test hourly aggregation functionality."""
    
    def test_aggregate_hourly_basic(self, activity_tracker, multi_day_behavioral_data):
        """Test basic hourly aggregation."""
        result = activity_tracker.aggregate_hourly(multi_day_behavioral_data)
        
        assert 'hour' in result.columns
        assert len(result) == 3 * 24  # 3 days * 24 hours
        
        # Check that required columns exist
        for state in activity_tracker.BEHAVIORAL_STATES:
            assert f'{state}_minutes' in result.columns
        
        assert 'total_minutes' in result.columns
        assert 'state_transitions' in result.columns
    
    def test_hourly_aggregation_60_minutes(self, activity_tracker, multi_day_behavioral_data):
        """Test that hourly aggregations partition 60-minute windows correctly."""
        result = activity_tracker.aggregate_hourly(multi_day_behavioral_data)
        
        # Each hour should account for approximately 60 minutes (within tolerance)
        for _, row in result.iterrows():
            total = row['total_minutes']
            assert 58 <= total <= 62, f"Hour total {total} not within 58-62 minute range"
    
    def test_hourly_all_minutes_accounted(self, activity_tracker, multi_day_behavioral_data):
        """Test that all minutes are accounted for in state breakdowns."""
        result = activity_tracker.aggregate_hourly(multi_day_behavioral_data)
        
        for _, row in result.iterrows():
            state_sum = sum(row[f'{state}_minutes'] for state in activity_tracker.BEHAVIORAL_STATES)
            total = row['total_minutes']
            
            # State minutes should approximately sum to total
            assert np.isclose(state_sum, total, atol=0.1)


class TestDailyAggregation:
    """Test daily aggregation functionality."""
    
    def test_aggregate_daily_basic(self, activity_tracker, multi_day_behavioral_data):
        """Test basic daily aggregation."""
        result = activity_tracker.aggregate_daily(multi_day_behavioral_data)
        
        assert 'day' in result.columns
        assert len(result) == 3  # 3 days
        
        # Check that required columns exist
        for state in activity_tracker.BEHAVIORAL_STATES:
            assert f'{state}_minutes' in result.columns
        
        assert 'total_minutes' in result.columns
        assert 'state_transitions' in result.columns
    
    def test_daily_aggregation_1440_minutes(self, activity_tracker, multi_day_behavioral_data):
        """Test that daily aggregations sum to 1440 minutes (24 hours)."""
        result = activity_tracker.aggregate_daily(multi_day_behavioral_data)
        
        # Each day should account for approximately 1440 minutes
        for _, row in result.iterrows():
            total = row['total_minutes']
            assert 1435 <= total <= 1445, f"Day total {total} not within 1435-1445 minute range"
    
    def test_daily_all_minutes_accounted(self, activity_tracker, multi_day_behavioral_data):
        """Test that all minutes are accounted for in daily state breakdowns."""
        result = activity_tracker.aggregate_daily(multi_day_behavioral_data)
        
        for _, row in result.iterrows():
            state_sum = sum(row[f'{state}_minutes'] for state in activity_tracker.BEHAVIORAL_STATES)
            total = row['total_minutes']
            
            # State minutes should approximately sum to total
            assert np.isclose(state_sum, total, atol=0.1)


class TestActivityRestRatio:
    """Test activity/rest ratio calculation."""
    
    def test_calculate_activity_rest_ratio(self, activity_tracker, simple_behavioral_data):
        """Test basic activity/rest ratio calculation."""
        result = activity_tracker.calculate_activity_rest_ratio(simple_behavioral_data)
        
        assert 'rest_percentage' in result
        assert 'active_percentage' in result
        assert 'rest_count' in result
        assert 'active_count' in result
        
        # Percentages should sum to 100
        assert np.isclose(result['rest_percentage'] + result['active_percentage'], 100.0)
    
    def test_activity_rest_ratio_known_values(self, activity_tracker):
        """Test activity/rest ratio with known distribution."""
        # Create data: 30 lying (rest), 70 standing (active)
        data = pd.DataFrame({
            'behavioral_state': ['lying'] * 30 + ['standing'] * 70
        })
        
        result = activity_tracker.calculate_activity_rest_ratio(data)
        
        assert np.isclose(result['rest_percentage'], 30.0)
        assert np.isclose(result['active_percentage'], 70.0)
        assert result['rest_count'] == 30
        assert result['active_count'] == 70
    
    def test_activity_rest_ratio_all_rest(self, activity_tracker):
        """Test activity/rest ratio when all states are rest."""
        data = pd.DataFrame({
            'behavioral_state': ['lying'] * 100
        })
        
        result = activity_tracker.calculate_activity_rest_ratio(data)
        
        assert np.isclose(result['rest_percentage'], 100.0)
        assert np.isclose(result['active_percentage'], 0.0)


class TestStateTransitions:
    """Test state transition counting."""
    
    def test_count_state_transitions_basic(self, activity_tracker, simple_behavioral_data):
        """Test basic state transition counting."""
        result = activity_tracker.count_state_transitions(simple_behavioral_data)
        
        assert 'total_transitions' in result
        assert 'transition_matrix' in result
        
        # Should have positive transitions
        assert result['total_transitions'] > 0
    
    def test_count_state_transitions_known_pattern(self, activity_tracker, known_duration_data):
        """Test transition counting with known pattern."""
        result = activity_tracker.count_state_transitions(known_duration_data)
        
        # Pattern has 3 transitions: lying->standing, standing->walking, walking->feeding
        assert result['total_transitions'] == 3
        
        # Check specific transitions
        matrix = result['transition_matrix']
        assert 'lying_to_standing' in matrix
        assert 'standing_to_walking' in matrix
        assert 'walking_to_feeding' in matrix
    
    def test_count_state_transitions_no_change(self, activity_tracker):
        """Test transition counting when state doesn't change."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'behavioral_state': ['lying'] * 10
        })
        
        result = activity_tracker.count_state_transitions(data)
        
        # Should have 0 transitions
        assert result['total_transitions'] == 0


class TestBehavioralLogGeneration:
    """Test behavioral log generation."""
    
    def test_generate_behavioral_log_basic(self, activity_tracker, simple_behavioral_data):
        """Test basic behavioral log generation."""
        result = activity_tracker.generate_behavioral_log(simple_behavioral_data)
        
        # Check required fields
        assert 'timestamp' in result.columns
        assert 'behavioral_state' in result.columns
        assert 'confidence_score' in result.columns
        assert 'duration_minutes' in result.columns
        assert 'movement_intensity' in result.columns
        
        assert len(result) == len(simple_behavioral_data)
    
    def test_generate_behavioral_log_with_confidence(self, activity_tracker, simple_behavioral_data):
        """Test behavioral log with custom confidence scores."""
        simple_behavioral_data['confidence'] = [0.9, 0.95, 0.88, 0.92, 0.85, 
                                                0.90, 0.93, 0.87, 0.91, 0.94]
        
        result = activity_tracker.generate_behavioral_log(
            simple_behavioral_data, 
            confidence_col='confidence'
        )
        
        assert 'confidence_score' in result.columns
        assert (result['confidence_score'] > 0.8).all()
    
    def test_generate_behavioral_log_required_fields(self, activity_tracker, multi_day_behavioral_data):
        """Test that behavioral log includes all required fields with proper units."""
        result = activity_tracker.generate_behavioral_log(multi_day_behavioral_data)
        
        # Verify data types
        assert pd.api.types.is_datetime64_any_dtype(result['timestamp'])
        assert result['behavioral_state'].dtype == object
        assert pd.api.types.is_numeric_dtype(result['confidence_score'])
        assert pd.api.types.is_numeric_dtype(result['duration_minutes'])
        assert pd.api.types.is_numeric_dtype(result['movement_intensity'])


class TestExportFunctions:
    """Test CSV and JSON export functionality."""
    
    def test_export_to_csv(self, activity_tracker, simple_behavioral_data):
        """Test CSV export."""
        log = activity_tracker.generate_behavioral_log(simple_behavioral_data)
        output_path = activity_tracker.export_to_csv(log, 'test_export.csv', include_timestamp=False)
        
        assert output_path.exists()
        assert output_path.suffix == '.csv'
        
        # Verify CSV is valid
        loaded = pd.read_csv(output_path)
        assert len(loaded) == len(log)
    
    def test_export_to_json(self, activity_tracker, simple_behavioral_data):
        """Test JSON export."""
        log = activity_tracker.generate_behavioral_log(simple_behavioral_data)
        output_path = activity_tracker.export_to_json(log, 'test_export.json', include_timestamp=False)
        
        assert output_path.exists()
        assert output_path.suffix == '.json'
        
        # Verify JSON is valid
        with open(output_path, 'r') as f:
            loaded = json.load(f)
        assert len(loaded) == len(log)
    
    def test_export_with_timestamp(self, activity_tracker, simple_behavioral_data):
        """Test export with timestamp in filename."""
        log = activity_tracker.generate_behavioral_log(simple_behavioral_data)
        output_path = activity_tracker.export_to_csv(log, 'test_export.csv', include_timestamp=True)
        
        assert output_path.exists()
        assert 'test_export' in output_path.stem
        assert output_path.suffix == '.csv'


class TestFullPipeline:
    """Test full pipeline processing."""
    
    def test_process_full_pipeline(self, activity_tracker, multi_day_behavioral_data):
        """Test full pipeline execution."""
        results = activity_tracker.process_full_pipeline(
            multi_day_behavioral_data,
            export_logs=True,
            export_hourly=True,
            export_daily=True,
            csv_format=True,
            json_format=True
        )
        
        # Check all results are present
        assert results['behavioral_log'] is not None
        assert results['hourly_aggregation'] is not None
        assert results['daily_aggregation'] is not None
        assert results['activity_rest_ratio'] is not None
        assert results['state_transitions'] is not None
        
        # Check export paths
        assert 'log_csv' in results['export_paths']
        assert 'log_json' in results['export_paths']
        assert 'hourly_csv' in results['export_paths']
        assert 'daily_csv' in results['export_paths']
    
    def test_process_large_dataset(self, activity_tracker):
        """Test processing 30 days of minute-level data (43,200 records)."""
        # Generate 30 days of data
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=30*24*60, freq='1min')
        
        states = np.random.choice(
            activity_tracker.BEHAVIORAL_STATES, 
            size=len(timestamps), 
            p=[0.45, 0.20, 0.10, 0.10, 0.15]  # Realistic distribution
        )
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'behavioral_state': states,
            'fxa': np.random.randn(len(timestamps)) * 0.2,
            'mya': np.random.randn(len(timestamps)) * 0.1,
            'rza': np.random.randn(len(timestamps)) * 0.1 - 0.8
        })
        
        # Measure processing time
        import time
        start_time = time.time()
        
        results = activity_tracker.process_full_pipeline(
            data,
            export_logs=False,
            export_hourly=False,
            export_daily=False
        )
        
        duration = time.time() - start_time
        
        # Should process in < 10 seconds
        assert duration < 10.0, f"Processing took {duration:.2f} seconds, expected < 10"
        
        # Verify results
        assert len(results['behavioral_log']) == 43200
        assert len(results['daily_aggregation']) == 30


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_handle_missing_data(self, activity_tracker):
        """Test handling of missing data windows."""
        # Create data with gaps
        timestamps = list(pd.date_range('2024-01-01 00:00:00', periods=5, freq='1min'))
        timestamps.extend(pd.date_range('2024-01-01 00:10:00', periods=5, freq='1min'))  # 5 minute gap
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'behavioral_state': ['lying'] * 10
        })
        
        result = activity_tracker.handle_missing_data(data)
        
        assert len(result) == 10
        assert 'behavioral_state' in result.columns
    
    def test_single_state_day(self, activity_tracker):
        """Test handling of single-state days (e.g., all lying if sick)."""
        # Create full day of lying state
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'behavioral_state': ['lying'] * 1440,
            'fxa': [0.01] * 1440,
            'mya': [0.01] * 1440,
            'rza': [-0.85] * 1440
        })
        
        results = activity_tracker.process_full_pipeline(
            data,
            export_logs=False,
            export_hourly=False,
            export_daily=False
        )
        
        # Should complete without errors
        assert results['behavioral_log'] is not None
        assert results['daily_aggregation'] is not None
        
        # Should show 100% rest
        activity_rest = results['activity_rest_ratio']
        assert np.isclose(activity_rest['rest_percentage'], 100.0)
    
    def test_validate_daily_totals(self, activity_tracker, multi_day_behavioral_data):
        """Test validation of daily totals."""
        daily_agg = activity_tracker.aggregate_daily(multi_day_behavioral_data)
        validation = activity_tracker.validate_daily_totals(daily_agg)
        
        assert 'all_valid' in validation
        assert 'invalid_days' in validation
        
        # Should be valid (within tolerance)
        assert validation['all_valid'] is True
    
    def test_empty_dataframe(self, activity_tracker):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['timestamp', 'behavioral_state', 'fxa', 'mya', 'rza'])
        
        result = activity_tracker.calculate_activity_rest_ratio(empty_df)
        
        assert result['rest_percentage'] == 0.0
        assert result['active_percentage'] == 0.0
        assert result['total_count'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
