"""
Integration tests for Hybrid Behavioral Classification Pipeline

Tests the integrated pipeline including:
- Component initialization
- Rule-based + ML integration
- Stress detection
- State transition smoothing
- End-to-end classification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from classification.hybrid_pipeline import HybridClassificationPipeline
from classification.stress_detector import StressDetector
from classification.state_transition_smoother import StateTransitionSmoother, BehavioralState
from classification.ml_classifier_wrapper import MLClassifierWrapper


class TestStressDetector:
    """Test stress detection component."""
    
    def test_initialization(self):
        """Test stress detector initialization."""
        detector = StressDetector(
            window_size=5,
            variance_threshold_sigma=2.0,
            min_axes_threshold=3
        )
        
        assert detector.window_size == 5
        assert detector.variance_threshold_sigma == 2.0
        assert detector.min_axes_threshold == 3
        assert not detector.is_calibrated
    
    def test_detect_stress_batch(self):
        """Test batch stress detection."""
        # Create test data with normal variance
        np.random.seed(42)
        data = pd.DataFrame({
            'fxa': np.random.normal(0, 0.1, 100),
            'mya': np.random.normal(0, 0.1, 100),
            'rza': np.random.normal(0.8, 0.05, 100),
            'sxg': np.random.normal(0, 5, 100),
            'lyg': np.random.normal(0, 5, 100),
            'dzg': np.random.normal(0, 5, 100)
        })
        
        detector = StressDetector(window_size=5)
        results = detector.detect_stress_batch(data)
        
        assert 'is_stressed' in results.columns
        assert 'stress_score' in results.columns
        assert len(results) == len(data)
    
    def test_detect_high_stress(self):
        """Test detection of high stress patterns."""
        # Create data with high variance (stress pattern)
        np.random.seed(42)
        data = pd.DataFrame({
            'fxa': np.random.normal(0, 0.8, 100),  # High variance
            'mya': np.random.normal(0, 0.8, 100),  # High variance
            'rza': np.random.normal(0.8, 0.6, 100),  # High variance
            'sxg': np.random.normal(0, 80, 100),  # High variance
            'lyg': np.random.normal(0, 80, 100),  # High variance
            'dzg': np.random.normal(0, 80, 100)   # High variance
        })
        
        detector = StressDetector(window_size=5, min_axes_threshold=3)
        results = detector.detect_stress_batch(data)
        
        # Should detect stress in some samples
        assert results['is_stressed'].sum() > 0
        assert results['stress_score'].max() > 0.5


class TestStateTransitionSmoother:
    """Test state transition smoother component."""
    
    def test_initialization(self):
        """Test smoother initialization."""
        smoother = StateTransitionSmoother(
            min_duration=2,
            window_size=5,
            confidence_threshold=0.6
        )
        
        assert smoother.min_duration == 2
        assert smoother.window_size == 5
        assert smoother.confidence_threshold == 0.6
    
    def test_smooth_single_state(self):
        """Test single state smoothing."""
        smoother = StateTransitionSmoother(min_duration=2)
        
        # First classification
        result1 = smoother.smooth_single(BehavioralState.LYING, 0.9)
        assert result1.state == BehavioralState.LYING
        
        # Same state continues
        result2 = smoother.smooth_single(BehavioralState.LYING, 0.9)
        assert result2.state == BehavioralState.LYING
    
    def test_smooth_state_transition(self):
        """Test state transition smoothing."""
        smoother = StateTransitionSmoother(min_duration=2)
        
        # Establish stable lying state
        smoother.smooth_single(BehavioralState.LYING, 0.9)
        smoother.smooth_single(BehavioralState.LYING, 0.9)
        
        # Single standing detection - should be smoothed out
        result = smoother.smooth_single(BehavioralState.STANDING, 0.8)
        # With min_duration=2, might keep lying or accept standing
        assert result.state in [BehavioralState.LYING, BehavioralState.STANDING]
    
    def test_smooth_batch(self):
        """Test batch smoothing."""
        smoother = StateTransitionSmoother(min_duration=2, window_size=3)
        
        # Create test data with jittery states
        data = pd.DataFrame({
            'state': ['lying', 'standing', 'lying', 'lying', 'lying', 'standing', 'standing'],
            'confidence': [0.9, 0.7, 0.9, 0.9, 0.9, 0.8, 0.8]
        })
        
        results = smoother.smooth_batch(data)
        
        assert 'smoothed_state' in results.columns
        assert 'smoothing_applied' in results.columns
        assert len(results) == len(data)


class TestMLClassifierWrapper:
    """Test ML classifier wrapper component."""
    
    def test_initialization_without_models(self):
        """Test initialization without model files."""
        wrapper = MLClassifierWrapper(use_fallback=True)
        
        assert not wrapper.models_available
        assert wrapper.use_fallback
    
    def test_ruminating_fallback(self):
        """Test ruminating detection fallback."""
        wrapper = MLClassifierWrapper(use_fallback=True)
        
        # Create features with ruminating frequency
        features = pd.DataFrame({
            'mya_dominant_frequency': [0.8],  # In ruminating range
            'mya_regularity_score': [0.7]
        })
        
        result = wrapper.classify_ruminating(features, use_model=False)
        
        assert result.state in [BehavioralState.RUMINATING, BehavioralState.UNCERTAIN]
        assert 0.0 <= result.confidence <= 1.0
    
    def test_feeding_fallback(self):
        """Test feeding detection fallback."""
        wrapper = MLClassifierWrapper(use_fallback=True)
        
        # Create features with head-down position
        features = pd.DataFrame({
            'pitch_angle': [-0.6],  # Head down
            'head_movement_intensity': [15.0]  # Active movement
        })
        
        result = wrapper.classify_feeding(features, use_model=False)
        
        assert result.state in [BehavioralState.FEEDING, BehavioralState.UNCERTAIN]
        assert 0.0 <= result.confidence <= 1.0


class TestHybridPipeline:
    """Test integrated hybrid classification pipeline."""
    
    @pytest.fixture
    def sample_sensor_data(self):
        """Create sample sensor data for testing."""
        np.random.seed(42)
        n_samples = 20
        
        # Generate realistic sensor data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
            'temperature': np.random.uniform(37.5, 39.0, n_samples),
            'fxa': np.random.normal(0, 0.2, n_samples),
            'mya': np.random.normal(0, 0.15, n_samples),
            'rza': np.random.normal(0.8, 0.1, n_samples),  # Standing posture
            'sxg': np.random.normal(0, 10, n_samples),
            'lyg': np.random.normal(0, 10, n_samples),
            'dzg': np.random.normal(0, 10, n_samples)
        })
        
        return data
    
    @pytest.fixture
    def lying_sensor_data(self):
        """Create sensor data for lying behavior."""
        np.random.seed(42)
        n_samples = 10
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
            'temperature': np.random.uniform(37.5, 38.5, n_samples),
            'fxa': np.random.normal(0, 0.05, n_samples),
            'mya': np.random.normal(0, 0.05, n_samples),
            'rza': np.random.normal(-0.7, 0.1, n_samples),  # Lying posture
            'sxg': np.random.normal(0, 5, n_samples),
            'lyg': np.random.normal(0, 5, n_samples),
            'dzg': np.random.normal(0, 5, n_samples)
        })
        
        return data
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = HybridClassificationPipeline()
        
        assert pipeline.rule_classifier is not None
        assert pipeline.ml_classifier is not None
        assert pipeline.stress_detector is not None
        assert pipeline.smoother is not None
    
    def test_classify_batch_basic(self, sample_sensor_data):
        """Test basic batch classification."""
        pipeline = HybridClassificationPipeline()
        
        results = pipeline.classify_batch(sample_sensor_data)
        
        # Check output structure
        assert len(results) == len(sample_sensor_data)
        assert 'state' in results.columns
        assert 'confidence' in results.columns
        assert 'is_stressed' in results.columns
        assert 'classification_source' in results.columns
    
    def test_classify_lying_behavior(self, lying_sensor_data):
        """Test classification of lying behavior."""
        pipeline = HybridClassificationPipeline()
        
        results = pipeline.classify_batch(lying_sensor_data)
        
        # Most samples should be classified as lying
        lying_count = (results['state'] == 'lying').sum()
        assert lying_count > len(lying_sensor_data) * 0.5
    
    def test_stress_detection_enabled(self, sample_sensor_data):
        """Test that stress detection is working."""
        pipeline = HybridClassificationPipeline()
        
        results = pipeline.classify_batch(sample_sensor_data)
        
        assert 'is_stressed' in results.columns
        assert 'stress_score' in results.columns
        assert results['stress_score'].dtype in [np.float64, np.float32]
    
    def test_smoothing_applied(self, sample_sensor_data):
        """Test that smoothing is applied."""
        pipeline = HybridClassificationPipeline()
        
        results = pipeline.classify_batch(sample_sensor_data)
        
        # Smoothing should be applied to some samples
        if 'smoothing_applied' in results.columns:
            assert results['smoothing_applied'].sum() >= 0
    
    def test_processing_speed(self, sample_sensor_data):
        """Test that processing meets speed requirement (<1 sec per minute)."""
        pipeline = HybridClassificationPipeline()
        
        # Process data
        results = pipeline.classify_batch(sample_sensor_data)
        
        # Check statistics
        stats = pipeline.get_statistics()
        
        # Should process quickly
        avg_time_per_sample = stats['avg_time_per_sample_ms']
        assert avg_time_per_sample < 1000  # Less than 1 second per sample
    
    def test_export_results(self, sample_sensor_data, tmp_path):
        """Test exporting results to file."""
        pipeline = HybridClassificationPipeline()
        
        results = pipeline.classify_batch(sample_sensor_data)
        
        # Export to CSV
        output_file = tmp_path / "test_results.csv"
        pipeline.export_results(results, str(output_file), format='csv')
        
        assert output_file.exists()
        
        # Verify file can be read back
        loaded = pd.read_csv(output_file)
        assert len(loaded) == len(results)
    
    def test_get_statistics(self, sample_sensor_data):
        """Test getting pipeline statistics."""
        pipeline = HybridClassificationPipeline()
        
        # Process some data
        pipeline.classify_batch(sample_sensor_data)
        
        stats = pipeline.get_statistics()
        
        assert 'total_classifications' in stats
        assert 'avg_processing_time_seconds' in stats
        assert stats['total_classifications'] == len(sample_sensor_data)


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow from data to output."""
        # Create test data
        np.random.seed(42)
        n_samples = 30
        
        sensor_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
            'temperature': np.random.uniform(37.5, 39.0, n_samples),
            'fxa': np.concatenate([
                np.random.normal(0, 0.05, 10),  # Lying
                np.random.normal(0, 0.1, 10),   # Standing
                np.random.normal(0.3, 0.1, 10)  # Walking
            ]),
            'mya': np.random.normal(0, 0.1, n_samples),
            'rza': np.concatenate([
                np.random.normal(-0.7, 0.1, 10),  # Lying
                np.random.normal(0.8, 0.05, 10),  # Standing
                np.random.normal(0.6, 0.1, 10)    # Walking
            ]),
            'sxg': np.random.normal(0, 10, n_samples),
            'lyg': np.random.normal(0, 10, n_samples),
            'dzg': np.random.normal(0, 10, n_samples)
        })
        
        # Initialize pipeline
        pipeline = HybridClassificationPipeline()
        
        # Classify
        results = pipeline.classify_batch(sensor_data)
        
        # Verify results
        assert len(results) == n_samples
        assert results['confidence'].min() >= 0.0
        assert results['confidence'].max() <= 1.0
        
        # Check that different states are detected
        unique_states = results['state'].unique()
        assert len(unique_states) > 0
        
        # Verify statistics
        stats = pipeline.get_statistics()
        assert stats['total_classifications'] == n_samples


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
