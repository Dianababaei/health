"""
Unit tests for temperature-activity correlation engine.

Tests correlation analysis, fever pattern detection, heat stress detection,
confidence scoring, and event generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from layer2.baseline import BaselineCalculator
from layer2.temperature_anomaly import TemperatureAnomalyDetector, AnomalyType
from layer2.temp_activity_correlation import (
    TemperatureActivityCorrelator,
    HealthPattern,
    CorrelationEvent
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def baseline_calculator():
    """Create BaselineCalculator instance."""
    return BaselineCalculator(window_hours=24, min_samples=30)


@pytest.fixture
def anomaly_detector():
    """Create TemperatureAnomalyDetector instance."""
    return TemperatureAnomalyDetector()


@pytest.fixture
def correlator():
    """Create TemperatureActivityCorrelator instance."""
    return TemperatureActivityCorrelator()


@pytest.fixture
def normal_temperature_data():
    """Generate normal temperature data (38.5°C ± 0.2°C)."""
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
    temperatures = np.random.normal(38.5, 0.1, len(timestamps))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures
    })


@pytest.fixture
def fever_temperature_data():
    """Generate fever temperature data (40.0°C sustained)."""
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
    
    # Normal for first 6 hours, fever for next 6 hours, recovery for rest
    temperatures = []
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        if hour < 6:
            temp = np.random.normal(38.5, 0.1)
        elif hour < 12:
            temp = np.random.normal(40.0, 0.2)  # Fever
        else:
            temp = np.random.normal(38.8, 0.15)  # Recovery
        temperatures.append(temp)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures
    })


@pytest.fixture
def heat_stress_temperature_data():
    """Generate heat stress temperature data (39.8°C during day)."""
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
    
    temperatures = []
    for ts in timestamps:
        hour = ts.hour
        if 10 <= hour < 16:  # Heat stress during midday
            temp = np.random.normal(39.8, 0.2)
        else:
            temp = np.random.normal(38.5, 0.1)
        temperatures.append(temp)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures
    })


@pytest.fixture
def normal_behavioral_data():
    """Generate normal behavioral data with typical day/night patterns."""
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
    
    states = []
    intensities = []
    
    for ts in timestamps:
        hour = ts.hour
        
        # Night (22:00-06:00): mostly lying
        if hour >= 22 or hour < 6:
            state = 'lying' if np.random.random() < 0.9 else 'standing'
            intensity = np.random.normal(0.15, 0.05)
        # Day: mixed activities
        else:
            rand = np.random.random()
            if rand < 0.3:
                state = 'lying'
                intensity = np.random.normal(0.15, 0.05)
            elif rand < 0.5:
                state = 'standing'
                intensity = np.random.normal(0.30, 0.10)
            elif rand < 0.7:
                state = 'walking'
                intensity = np.random.normal(0.80, 0.15)
            else:
                state = 'feeding'
                intensity = np.random.normal(0.50, 0.10)
        
        states.append(state)
        intensities.append(max(0.05, intensity))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'movement_intensity': intensities,
        'duration_minutes': 1.0  # Assume 1-minute sampling
    })


@pytest.fixture
def fever_behavioral_data():
    """Generate behavioral data showing reduced activity (fever pattern)."""
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
    
    states = []
    intensities = []
    durations = []
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        
        # During fever (6:00-12:00): mostly lying, very low activity
        if 6 <= hour < 12:
            state = 'lying'
            intensity = np.random.normal(0.10, 0.03)
            duration = 120.0 if i % 120 == 0 else 0.0  # Mark long lying bouts
        else:
            # Normal pattern otherwise
            if hour >= 22 or hour < 6:
                state = 'lying' if np.random.random() < 0.9 else 'standing'
                intensity = np.random.normal(0.15, 0.05)
            else:
                rand = np.random.random()
                if rand < 0.4:
                    state = 'lying'
                    intensity = np.random.normal(0.15, 0.05)
                elif rand < 0.6:
                    state = 'standing'
                    intensity = np.random.normal(0.30, 0.10)
                else:
                    state = 'walking'
                    intensity = np.random.normal(0.70, 0.15)
            duration = 1.0
        
        states.append(state)
        intensities.append(max(0.05, intensity))
        durations.append(duration if duration > 0 else 1.0)
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'movement_intensity': intensities,
        'duration_minutes': durations
    })


@pytest.fixture
def heat_stress_behavioral_data():
    """Generate behavioral data showing elevated activity (heat stress pattern)."""
    timestamps = pd.date_range('2024-01-01 00:00:00', periods=1440, freq='1min')
    
    states = []
    intensities = []
    
    for ts in timestamps:
        hour = ts.hour
        
        # During heat stress (10:00-16:00): restless, high activity
        if 10 <= hour < 16:
            rand = np.random.random()
            if rand < 0.5:
                state = 'walking'
                intensity = np.random.normal(0.90, 0.10)
            elif rand < 0.8:
                state = 'standing'
                intensity = np.random.normal(0.60, 0.15)
            else:
                state = 'feeding'
                intensity = np.random.normal(0.70, 0.10)
        else:
            # Normal pattern
            if hour >= 22 or hour < 6:
                state = 'lying' if np.random.random() < 0.9 else 'standing'
                intensity = np.random.normal(0.15, 0.05)
            else:
                rand = np.random.random()
                if rand < 0.4:
                    state = 'lying'
                    intensity = np.random.normal(0.15, 0.05)
                elif rand < 0.6:
                    state = 'standing'
                    intensity = np.random.normal(0.30, 0.10)
                else:
                    state = 'walking'
                    intensity = np.random.normal(0.70, 0.15)
        
        states.append(state)
        intensities.append(max(0.05, intensity))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'behavioral_state': states,
        'movement_intensity': intensities,
        'duration_minutes': 1.0
    })


# ============================================================================
# Baseline Calculator Tests
# ============================================================================

class TestBaselineCalculator:
    """Test baseline temperature calculation."""
    
    def test_calculate_baseline_basic(self, baseline_calculator, normal_temperature_data):
        """Test basic baseline calculation."""
        result = baseline_calculator.calculate_baseline(normal_temperature_data)
        
        assert 'baseline_temp' in result.columns
        assert 'baseline_lower' in result.columns
        assert 'baseline_upper' in result.columns
        
        # Check that baseline values are reasonable
        valid_baselines = result['baseline_temp'].dropna()
        assert len(valid_baselines) > 0
        assert valid_baselines.mean() > 38.0
        assert valid_baselines.mean() < 39.0
    
    def test_individual_baseline(self, baseline_calculator, normal_temperature_data):
        """Test individual baseline calculation."""
        baseline_stats = baseline_calculator.calculate_individual_baseline(
            normal_temperature_data,
            lookback_days=1
        )
        
        assert baseline_stats['baseline'] is not None
        assert 38.0 < baseline_stats['baseline'] < 39.0
        assert baseline_stats['std'] > 0
        assert baseline_stats['sample_count'] > 0
    
    def test_hourly_baselines(self, baseline_calculator, normal_temperature_data):
        """Test hour-specific baseline calculation."""
        hourly_baselines = baseline_calculator.calculate_hourly_baselines(
            normal_temperature_data
        )
        
        assert len(hourly_baselines) == 24
        
        # Check that most hours have valid baselines
        valid_hours = sum(1 for h in hourly_baselines.values() if h['baseline'] is not None)
        assert valid_hours >= 20  # At least 20 out of 24 hours


# ============================================================================
# Anomaly Detector Tests
# ============================================================================

class TestAnomalyDetector:
    """Test temperature anomaly detection."""
    
    def test_detect_normal_temperature(self, anomaly_detector, normal_temperature_data):
        """Test that normal temperatures are not flagged as anomalies."""
        result = anomaly_detector.detect_anomalies(normal_temperature_data)
        
        assert 'anomaly_type' in result.columns
        
        # Most readings should be normal
        normal_count = (result['anomaly_type'] == AnomalyType.NORMAL.value).sum()
        assert normal_count / len(result) > 0.95
    
    def test_detect_fever(self, anomaly_detector, fever_temperature_data):
        """Test fever detection (>39.0°C)."""
        result = anomaly_detector.detect_anomalies(fever_temperature_data)
        
        # Should detect fever during hours 6-12
        fever_period = result[(result.index >= 360) & (result.index < 720)]
        fever_count = (fever_period['anomaly_type'].isin([
            AnomalyType.FEVER.value,
            AnomalyType.HEAT_STRESS.value
        ])).sum()
        
        # At least 80% of fever period should be detected
        assert fever_count / len(fever_period) > 0.8
    
    def test_detect_heat_stress(self, anomaly_detector, heat_stress_temperature_data):
        """Test heat stress detection (>39.5°C)."""
        result = anomaly_detector.detect_anomalies(heat_stress_temperature_data)
        
        # Should detect heat stress during hours 10-16
        stress_period = result[(result.index >= 600) & (result.index < 960)]
        stress_count = (stress_period['anomaly_type'] == AnomalyType.HEAT_STRESS.value).sum()
        
        # At least 70% of stress period should be detected
        assert stress_count / len(stress_period) > 0.7
    
    def test_anomaly_severity_scoring(self, anomaly_detector, fever_temperature_data):
        """Test that severity scores are reasonable."""
        result = anomaly_detector.detect_anomalies(fever_temperature_data)
        
        # Severity should be between 0 and 1
        assert (result['anomaly_severity'] >= 0).all()
        assert (result['anomaly_severity'] <= 1).all()
        
        # Higher temperatures should have higher severity
        fever_rows = result[result['anomaly_type'] != AnomalyType.NORMAL.value]
        if len(fever_rows) > 0:
            assert fever_rows['anomaly_severity'].mean() > 0.3


# ============================================================================
# Correlator Tests
# ============================================================================

class TestCorrelator:
    """Test temperature-activity correlation engine."""
    
    def test_load_behavioral_data(self, correlator, normal_behavioral_data):
        """Test loading and preparing behavioral data."""
        result = correlator.load_behavioral_data(normal_behavioral_data)
        
        assert 'activity_level' in result.columns
        assert 'is_rest' in result.columns
        assert 'is_active' in result.columns
        
        # Activity levels should be between 0 and 1
        assert (result['activity_level'] >= 0).all()
        assert (result['activity_level'] <= 1).all()
    
    def test_merge_temperature_activity(self, correlator, normal_temperature_data, normal_behavioral_data):
        """Test time-aligned merging of temperature and behavioral data."""
        behavioral_prepared = correlator.load_behavioral_data(normal_behavioral_data)
        
        merged = correlator.merge_temperature_activity(
            normal_temperature_data,
            behavioral_prepared
        )
        
        assert len(merged) > 0
        assert 'temperature' in merged.columns
        assert 'behavioral_state' in merged.columns
        assert 'activity_level' in merged.columns
    
    def test_detect_fever_pattern(self, correlator, fever_temperature_data, fever_behavioral_data):
        """Test fever pattern detection (high temp + low activity)."""
        # Prepare data
        behavioral_prepared = correlator.load_behavioral_data(fever_behavioral_data)
        merged = correlator.merge_temperature_activity(
            fever_temperature_data,
            behavioral_prepared
        )
        
        # Calculate baseline
        merged = correlator.calculate_baseline_activity(merged)
        
        # Detect fever patterns
        result = correlator.detect_fever_pattern(merged)
        
        assert 'fever_pattern' in result.columns
        assert 'fever_confidence' in result.columns
        
        # Should detect fever during hours 6-12
        fever_detections = result['fever_pattern'].sum()
        assert fever_detections > 0
        
        # Fever detections should have reasonable confidence
        fever_rows = result[result['fever_pattern']]
        if len(fever_rows) > 0:
            assert fever_rows['fever_confidence'].mean() > 30  # At least 30/100
    
    def test_detect_heat_stress_pattern(self, correlator, heat_stress_temperature_data, heat_stress_behavioral_data):
        """Test heat stress pattern detection (high temp + high activity)."""
        # Prepare data
        behavioral_prepared = correlator.load_behavioral_data(heat_stress_behavioral_data)
        merged = correlator.merge_temperature_activity(
            heat_stress_temperature_data,
            behavioral_prepared
        )
        
        # Calculate baseline
        merged = correlator.calculate_baseline_activity(merged)
        
        # Detect heat stress patterns
        result = correlator.detect_heat_stress_pattern(merged)
        
        assert 'heat_stress_pattern' in result.columns
        assert 'heat_stress_confidence' in result.columns
        
        # Should detect heat stress during hours 10-16
        stress_detections = result['heat_stress_pattern'].sum()
        assert stress_detections > 0
        
        # Heat stress detections should have reasonable confidence
        stress_rows = result[result['heat_stress_pattern']]
        if len(stress_rows) > 0:
            assert stress_rows['heat_stress_confidence'].mean() > 30
    
    def test_distinguish_fever_from_normal_rest(self, correlator, normal_temperature_data, normal_behavioral_data):
        """Test that normal nighttime rest is not flagged as fever."""
        behavioral_prepared = correlator.load_behavioral_data(normal_behavioral_data)
        merged = correlator.merge_temperature_activity(
            normal_temperature_data,
            behavioral_prepared
        )
        
        merged = correlator.calculate_baseline_activity(merged)
        result = correlator.detect_fever_pattern(merged)
        
        # Should have very few or no fever detections
        fever_rate = result['fever_pattern'].sum() / len(result)
        assert fever_rate < 0.1  # Less than 10% false positive rate
    
    def test_correlation_metrics(self, correlator, normal_temperature_data, normal_behavioral_data):
        """Test correlation coefficient calculation."""
        behavioral_prepared = correlator.load_behavioral_data(normal_behavioral_data)
        merged = correlator.merge_temperature_activity(
            normal_temperature_data,
            behavioral_prepared
        )
        
        correlation_results = correlator.calculate_correlation_metrics(
            merged,
            window_hours=[1, 4, 24]
        )
        
        assert '1h' in correlation_results
        assert '4h' in correlation_results
        assert '24h' in correlation_results
        
        # Check that correlation values are in valid range
        for window, df in correlation_results.items():
            corr_col = f'correlation_{window}'
            valid_corrs = df[corr_col].dropna()
            if len(valid_corrs) > 0:
                assert (valid_corrs >= -1).all()
                assert (valid_corrs <= 1).all()
    
    def test_lag_analysis(self, correlator, normal_temperature_data, normal_behavioral_data):
        """Test lag analysis between temperature and activity."""
        behavioral_prepared = correlator.load_behavioral_data(normal_behavioral_data)
        merged = correlator.merge_temperature_activity(
            normal_temperature_data,
            behavioral_prepared
        )
        
        lag_results = correlator.calculate_lag_analysis(merged)
        
        assert 'best_lag_minutes' in lag_results
        assert 'best_correlation' in lag_results
        assert 'lag_correlations' in lag_results
        
        # Best lag should be within tested range
        assert -30 <= lag_results['best_lag_minutes'] <= 30
    
    def test_generate_correlation_events(self, correlator, fever_temperature_data, fever_behavioral_data):
        """Test correlation event generation."""
        behavioral_prepared = correlator.load_behavioral_data(fever_behavioral_data)
        merged = correlator.merge_temperature_activity(
            fever_temperature_data,
            behavioral_prepared
        )
        
        merged = correlator.calculate_baseline_activity(merged)
        merged = correlator.detect_fever_pattern(merged)
        
        events = correlator.generate_correlation_events(merged)
        
        # Should generate at least one fever event
        assert len(events) > 0
        
        # Events should have required attributes
        for event in events:
            assert isinstance(event, CorrelationEvent)
            assert event.pattern_type in [HealthPattern.FEVER, HealthPattern.HEAT_STRESS]
            assert 0 <= event.confidence <= 100
            assert event.temperature > 0
            assert event.duration_minutes > 0
    
    def test_full_correlation_pipeline(self, correlator, fever_temperature_data, fever_behavioral_data):
        """Test complete correlation analysis pipeline."""
        merged, events, metrics = correlator.process_full_correlation(
            fever_temperature_data,
            fever_behavioral_data
        )
        
        assert len(merged) > 0
        assert len(events) > 0
        assert 'total_records' in metrics
        assert 'fever_detections' in metrics
        assert 'heat_stress_detections' in metrics
        assert 'events_generated' in metrics
        
        # Verify fever was detected
        assert metrics['fever_detections'] > 0
    
    def test_pattern_confidence_scoring(self, correlator):
        """Test pattern confidence scoring system."""
        # Test fever confidence
        fever_confidence = correlator.calculate_pattern_confidence(
            temperature=40.0,
            activity_level=0.1,
            correlation=-0.7,
            duration_minutes=120.0,
            pattern_type=HealthPattern.FEVER
        )
        
        assert 0 <= fever_confidence <= 100
        assert fever_confidence > 50  # Should be high for clear fever pattern
        
        # Test heat stress confidence
        stress_confidence = correlator.calculate_pattern_confidence(
            temperature=40.0,
            activity_level=0.9,
            correlation=0.7,
            duration_minutes=120.0,
            pattern_type=HealthPattern.HEAT_STRESS
        )
        
        assert 0 <= stress_confidence <= 100
        assert stress_confidence > 50  # Should be high for clear heat stress pattern
    
    def test_edge_case_missing_data(self, correlator):
        """Test handling of missing data."""
        # Create data with gaps
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1min')
        temp_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.normal(38.5, 0.1, 100)
        })
        
        # Behavioral data with different timestamps (gaps)
        behavior_timestamps = pd.date_range('2024-01-01 00:00:00', periods=90, freq='1min')
        behavior_data = pd.DataFrame({
            'timestamp': behavior_timestamps,
            'behavioral_state': ['lying'] * 90,
            'movement_intensity': np.random.normal(0.15, 0.05, 90)
        })
        
        # Should handle gracefully
        behavioral_prepared = correlator.load_behavioral_data(behavior_data)
        merged = correlator.merge_temperature_activity(temp_data, behavioral_prepared)
        
        # Should have some merged data despite gaps
        assert len(merged) > 0
        assert len(merged) <= min(len(temp_data), len(behavior_data))
    
    def test_edge_case_transitional_states(self, correlator):
        """Test handling of transitional behavioral states."""
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=100, freq='1min')
        
        behavior_data = pd.DataFrame({
            'timestamp': timestamps,
            'behavioral_state': ['transition'] * 100,
            'movement_intensity': np.random.normal(0.4, 0.1, 100)
        })
        
        # Should handle transition states
        behavioral_prepared = correlator.load_behavioral_data(behavior_data)
        
        assert 'activity_level' in behavioral_prepared.columns
        # Transition states should have moderate activity
        assert 0.2 < behavioral_prepared['activity_level'].mean() < 0.6
    
    def test_performance_large_dataset(self, correlator):
        """Test processing performance on larger dataset."""
        import time
        
        # Generate 7 days of minute-level data
        n_records = 7 * 24 * 60
        timestamps = pd.date_range('2024-01-01 00:00:00', periods=n_records, freq='1min')
        
        temp_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.normal(38.5, 0.2, n_records)
        })
        
        behavior_data = pd.DataFrame({
            'timestamp': timestamps,
            'behavioral_state': np.random.choice(['lying', 'standing', 'walking'], n_records),
            'movement_intensity': np.random.normal(0.3, 0.2, n_records)
        })
        
        # Measure processing time
        start_time = time.time()
        merged, events, metrics = correlator.process_full_correlation(
            temp_data,
            behavior_data
        )
        elapsed_time = time.time() - start_time
        
        # Should process in reasonable time (<5 seconds for 7 days)
        assert elapsed_time < 5.0
        
        # Should process all records
        assert len(merged) > 0
        
        # Calculate per-record processing time
        time_per_record = elapsed_time / n_records * 1000  # Convert to ms
        assert time_per_record < 0.05  # Less than 50ms per data point


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_fever_detection_accuracy(self, correlator, fever_temperature_data, fever_behavioral_data):
        """Test fever detection accuracy on simulated data."""
        merged, events, metrics = correlator.process_full_correlation(
            fever_temperature_data,
            fever_behavioral_data
        )
        
        # Should detect fever events
        fever_events = [e for e in events if e.pattern_type == HealthPattern.FEVER]
        assert len(fever_events) > 0
        
        # Fever events should be during the fever period (hours 6-12)
        for event in fever_events:
            hour = event.timestamp.hour
            # Allow some tolerance
            assert 5 <= hour <= 13
        
        # Average confidence should be reasonable
        avg_confidence = np.mean([e.confidence for e in fever_events])
        assert avg_confidence > 40  # At least 40/100
    
    def test_heat_stress_detection_accuracy(self, correlator, heat_stress_temperature_data, heat_stress_behavioral_data):
        """Test heat stress detection accuracy on simulated data."""
        merged, events, metrics = correlator.process_full_correlation(
            heat_stress_temperature_data,
            heat_stress_behavioral_data
        )
        
        # Should detect heat stress events
        stress_events = [e for e in events if e.pattern_type == HealthPattern.HEAT_STRESS]
        assert len(stress_events) > 0
        
        # Heat stress events should be during the stress period (hours 10-16)
        for event in stress_events:
            hour = event.timestamp.hour
            # Allow some tolerance
            assert 9 <= hour <= 17
        
        # Average confidence should be reasonable
        avg_confidence = np.mean([e.confidence for e in stress_events])
        assert avg_confidence > 40
    
    def test_event_serialization(self, correlator, fever_temperature_data, fever_behavioral_data):
        """Test that correlation events can be serialized."""
        merged, events, metrics = correlator.process_full_correlation(
            fever_temperature_data,
            fever_behavioral_data
        )
        
        # Should be able to convert events to dictionaries
        for event in events:
            event_dict = event.to_dict()
            
            assert isinstance(event_dict, dict)
            assert 'timestamp' in event_dict
            assert 'pattern_type' in event_dict
            assert 'confidence' in event_dict
            assert 'temperature' in event_dict
            assert 'contributing_factors' in event_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
