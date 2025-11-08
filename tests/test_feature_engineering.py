"""
Unit Tests for Feature Engineering Module

Tests all feature engineering functions with known sensor patterns and expected outputs.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_processing.feature_engineering import (
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


class TestMotionIntensity(unittest.TestCase):
    """Test motion intensity calculation."""
    
    def test_motion_intensity_scalar(self):
        """Test with known values."""
        # Test with equal components
        result = calculate_motion_intensity(1.0, 1.0, 1.0)
        expected = np.sqrt(3.0)
        self.assertAlmostEqual(result, expected, places=5)
        
        # Test with zeros
        result = calculate_motion_intensity(0.0, 0.0, 0.0)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_motion_intensity_array(self):
        """Test with numpy arrays."""
        fxa = np.array([1.0, 0.0, -1.0])
        mya = np.array([0.0, 1.0, 0.0])
        rza = np.array([0.0, 0.0, 1.0])
        
        result = calculate_motion_intensity(fxa, mya, rza)
        expected = np.array([1.0, 1.0, np.sqrt(2.0)])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_motion_intensity_series(self):
        """Test with pandas Series."""
        fxa = pd.Series([0.5, 0.0, -0.5])
        mya = pd.Series([0.5, 0.0, -0.5])
        rza = pd.Series([0.0, 1.0, 0.0])
        
        result = calculate_motion_intensity(fxa, mya, rza)
        
        self.assertIsInstance(result, pd.Series)
        self.assertAlmostEqual(result.iloc[1], 1.0)
    
    def test_motion_intensity_behavioral_patterns(self):
        """Test with realistic behavioral patterns."""
        # Lying: minimal movement
        lying_intensity = calculate_motion_intensity(0.05, 0.05, -0.8)
        self.assertLess(lying_intensity, 1.0)
        
        # Walking: moderate movement
        walking_intensity = calculate_motion_intensity(0.4, 0.2, 0.85)
        self.assertGreater(walking_intensity, 0.8)


class TestOrientationAngles(unittest.TestCase):
    """Test pitch and roll angle calculations."""
    
    def test_pitch_angle_upright(self):
        """Test pitch for upright posture (standing)."""
        # Rza = 1.0g (upright) -> pitch ≈ π/2
        result = calculate_pitch_angle(1.0)
        self.assertAlmostEqual(result, np.pi / 2, places=5)
    
    def test_pitch_angle_lying(self):
        """Test pitch for lying posture."""
        # Rza = -0.8g (lying) -> negative pitch
        result = calculate_pitch_angle(-0.8)
        self.assertLess(result, 0)
        self.assertGreater(result, -np.pi / 2)
    
    def test_pitch_angle_horizontal(self):
        """Test pitch for horizontal orientation."""
        # Rza = 0.0g (horizontal) -> pitch ≈ 0
        result = calculate_pitch_angle(0.0)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_pitch_angle_array(self):
        """Test pitch with array input."""
        rza = np.array([1.0, 0.0, -1.0])
        result = calculate_pitch_angle(rza)
        
        expected = np.array([np.pi / 2, 0.0, -np.pi / 2])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_pitch_angle_out_of_range(self):
        """Test pitch with physically impossible values."""
        # Values > 1.0g should be clipped
        result = calculate_pitch_angle(2.0)
        self.assertAlmostEqual(result, np.pi / 2)  # Clipped to 1.0
        
        result = calculate_pitch_angle(-2.0)
        self.assertAlmostEqual(result, -np.pi / 2)  # Clipped to -1.0
    
    def test_roll_angle_basic(self):
        """Test roll angle calculation."""
        # Test cardinal directions
        result = calculate_roll_angle(1.0, 0.0)
        self.assertAlmostEqual(result, 0.0, places=5)
        
        result = calculate_roll_angle(0.0, 1.0)
        self.assertAlmostEqual(result, np.pi / 2, places=5)
        
        result = calculate_roll_angle(-1.0, 0.0)
        self.assertAlmostEqual(abs(result), np.pi, places=5)
    
    def test_roll_angle_array(self):
        """Test roll with array input."""
        fxa = np.array([1.0, 0.0, -1.0])
        mya = np.array([0.0, 1.0, 0.0])
        
        result = calculate_roll_angle(fxa, mya)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)


class TestActivityScore(unittest.TestCase):
    """Test activity score calculation."""
    
    def test_activity_score_basic(self):
        """Test with known values and default weights."""
        result = calculate_activity_score(1.0, 1.0, 1.0)
        # Default weights: (0.4, 0.3, 0.3), sum = 1.0
        expected = 0.4 + 0.3 + 0.3
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_activity_score_custom_weights(self):
        """Test with custom weights."""
        result = calculate_activity_score(1.0, 1.0, 1.0, weights=(0.5, 0.3, 0.2))
        expected = 0.5 + 0.3 + 0.2
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_activity_score_array(self):
        """Test with array inputs."""
        fxa = np.array([0.5, 0.0, -0.5])
        mya = np.array([0.3, 0.0, -0.3])
        rza = np.array([0.2, 1.0, 0.2])
        
        result = calculate_activity_score(fxa, mya, rza)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), 3)
        
        # Second element should have highest score (rza=1.0)
        self.assertGreater(result[1], result[0])
    
    def test_activity_score_behavioral_patterns(self):
        """Test with realistic behavioral patterns."""
        # Lying: low activity
        lying_score = calculate_activity_score(0.05, 0.05, -0.8)
        
        # Walking: high activity
        walking_score = calculate_activity_score(0.4, 0.2, 0.85)
        
        self.assertGreater(walking_score, lying_score)


class TestPosturalStability(unittest.TestCase):
    """Test postural stability calculation."""
    
    def test_postural_stability_stable(self):
        """Test with stable posture (low variance)."""
        # Constant Rza = stable
        rza = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        result = calculate_postural_stability(rza)
        self.assertAlmostEqual(result, 0.0, places=5)
    
    def test_postural_stability_unstable(self):
        """Test with unstable posture (high variance)."""
        # Varying Rza = unstable
        rza = np.array([0.5, 0.9, 0.3, 0.8, 0.4])
        result = calculate_postural_stability(rza)
        self.assertGreater(result, 0.01)
    
    def test_postural_stability_rolling_window(self):
        """Test with rolling window calculation."""
        rza = pd.Series([0.9, 0.9, 0.5, 0.5, 0.9, 0.9])
        result = calculate_postural_stability(rza, window_size=3)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), len(rza))
    
    def test_postural_stability_series(self):
        """Test with pandas Series."""
        rza = pd.Series([0.9, 0.8, 0.85, 0.9, 0.88])
        result = calculate_postural_stability(rza)
        
        self.assertIsInstance(result, (float, np.floating))
        self.assertGreaterEqual(result, 0.0)


class TestHeadMovementIntensity(unittest.TestCase):
    """Test head movement intensity calculation."""
    
    def test_head_movement_intensity_basic(self):
        """Test with known values."""
        result = calculate_head_movement_intensity(3.0, 4.0)
        expected = 5.0  # 3-4-5 triangle
        self.assertAlmostEqual(result, expected, places=5)
    
    def test_head_movement_intensity_array(self):
        """Test with array inputs."""
        lyg = np.array([0.0, 10.0, -10.0])
        dzg = np.array([10.0, 0.0, 10.0])
        
        result = calculate_head_movement_intensity(lyg, dzg)
        expected = np.array([10.0, 10.0, np.sqrt(200)])
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_head_movement_intensity_series(self):
        """Test with pandas Series."""
        lyg = pd.Series([5.0, 0.0, -5.0])
        dzg = pd.Series([0.0, 5.0, 0.0])
        
        result = calculate_head_movement_intensity(lyg, dzg)
        
        self.assertIsInstance(result, pd.Series)
        self.assertAlmostEqual(result.iloc[0], 5.0)
        self.assertAlmostEqual(result.iloc[1], 5.0)
    
    def test_head_movement_behavioral_patterns(self):
        """Test with realistic behavioral patterns."""
        # Ruminating: moderate head movement
        ruminating_intensity = calculate_head_movement_intensity(6.0, 2.0)
        
        # Feeding: high head movement
        feeding_intensity = calculate_head_movement_intensity(-15.0, 3.5)
        
        self.assertGreater(feeding_intensity, 10.0)
        self.assertGreater(ruminating_intensity, 5.0)


class TestRhythmicFeatures(unittest.TestCase):
    """Test rhythmic pattern feature extraction."""
    
    def test_rhythmic_features_constant_signal(self):
        """Test with constant signal (no rhythm)."""
        signal_data = np.ones(100)
        result = extract_rhythmic_features(signal_data, sampling_rate=1.0)
        
        # Should have all required keys
        self.assertIn('dominant_frequency', result)
        self.assertIn('spectral_power', result)
        self.assertIn('zero_crossing_rate', result)
        self.assertIn('peak_count', result)
        self.assertIn('regularity_score', result)
        
        # Constant signal has no zero crossings
        self.assertAlmostEqual(result['zero_crossing_rate'], 0.0)
    
    def test_rhythmic_features_sinusoidal(self):
        """Test with known sinusoidal signal."""
        # Create 1 Hz sinusoid
        t = np.linspace(0, 10, 100)  # 10 seconds at 10 Hz sampling
        signal_data = np.sin(2 * np.pi * 1.0 * t)
        
        result = extract_rhythmic_features(
            signal_data,
            sampling_rate=10.0,
            target_freq_range=(0.5, 1.5)
        )
        
        # Should detect ~1 Hz frequency
        self.assertIsNotNone(result['dominant_frequency'])
        if not np.isnan(result['dominant_frequency']):
            self.assertGreater(result['dominant_frequency'], 0.5)
            self.assertLess(result['dominant_frequency'], 1.5)
    
    def test_rhythmic_features_rumination_pattern(self):
        """Test with simulated rumination pattern (50 cycles/min = 0.83 Hz)."""
        # Create signal at 0.83 Hz
        t = np.linspace(0, 60, 60)  # 60 seconds at 1 Hz sampling
        signal_data = np.sin(2 * np.pi * 0.83 * t)
        
        result = extract_rhythmic_features(
            signal_data,
            sampling_rate=1.0,
            target_freq_range=(0.67, 1.0)
        )
        
        # Should detect frequency in rumination range
        if not np.isnan(result['dominant_frequency']):
            self.assertGreater(result['dominant_frequency'], 0.6)
            self.assertLess(result['dominant_frequency'], 1.1)
        
        # Should have reasonable peak count
        self.assertGreater(result['peak_count'], 0)
    
    def test_rhythmic_features_with_nan(self):
        """Test with NaN values in signal."""
        signal_data = np.array([1.0, 2.0, np.nan, 3.0, 2.0, 1.0, np.nan, 2.0, 3.0, 2.0])
        result = extract_rhythmic_features(signal_data, sampling_rate=1.0)
        
        # Should handle NaNs gracefully
        self.assertIsNotNone(result)
        self.assertIn('zero_crossing_rate', result)
    
    def test_rhythmic_features_insufficient_data(self):
        """Test with insufficient data points."""
        signal_data = np.array([1.0, 2.0, 1.0])
        result = extract_rhythmic_features(signal_data, sampling_rate=1.0)
        
        # Should return NaN for most features
        self.assertTrue(np.isnan(result['dominant_frequency']))
        self.assertTrue(np.isnan(result['spectral_power']))


class TestEngineerFeatures(unittest.TestCase):
    """Test comprehensive feature engineering function."""
    
    def setUp(self):
        """Create sample sensor data."""
        self.data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'fxa': np.random.randn(100) * 0.3,
            'mya': np.random.randn(100) * 0.2,
            'rza': np.random.randn(100) * 0.1 + 0.8,  # Mostly upright
            'sxg': np.random.randn(100) * 5.0,
            'lyg': np.random.randn(100) * 5.0,
            'dzg': np.random.randn(100) * 3.0
        })
    
    def test_engineer_features_basic(self):
        """Test basic feature engineering."""
        result = engineer_features(self.data, include_rhythmic=False)
        
        # Check that basic features are created
        self.assertIn('motion_intensity', result.columns)
        self.assertIn('pitch_angle', result.columns)
        self.assertIn('roll_angle', result.columns)
        self.assertIn('activity_score', result.columns)
        self.assertIn('postural_stability', result.columns)
        self.assertIn('head_movement_intensity', result.columns)
    
    def test_engineer_features_with_rhythmic(self):
        """Test feature engineering with rhythmic features."""
        result = engineer_features(self.data, include_rhythmic=True)
        
        # Should include rhythmic features
        self.assertIn('mya_dominant_frequency', result.columns)
        self.assertIn('lyg_dominant_frequency', result.columns)
        self.assertIn('mya_zero_crossing_rate', result.columns)
    
    def test_engineer_features_with_window(self):
        """Test feature engineering with rolling window."""
        result = engineer_features(self.data, window_size=10, include_rhythmic=False)
        
        # Postural stability should be rolling
        self.assertIn('postural_stability', result.columns)
        # Check that it's not a single value
        self.assertGreater(result['postural_stability'].nunique(), 1)
    
    def test_engineer_features_preserves_original(self):
        """Test that original columns are preserved."""
        result = engineer_features(self.data, include_rhythmic=False)
        
        # Original sensor columns should still exist
        self.assertIn('fxa', result.columns)
        self.assertIn('rza', result.columns)
        self.assertIn('lyg', result.columns)
    
    def test_engineer_features_partial_data(self):
        """Test with partial sensor data."""
        partial_data = pd.DataFrame({
            'fxa': [0.5, 0.3, 0.4],
            'mya': [0.2, 0.1, 0.3],
            'rza': [0.9, 0.85, 0.88]
        })
        
        result = engineer_features(partial_data, include_rhythmic=False)
        
        # Should create features from available sensors
        self.assertIn('motion_intensity', result.columns)
        self.assertIn('pitch_angle', result.columns)


class TestCreateFeatureVector(unittest.TestCase):
    """Test feature vector creation for ML models."""
    
    def setUp(self):
        """Create sample engineered data."""
        self.data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='1min'),
            'temperature_norm': np.random.rand(50),
            'fxa_std': np.random.randn(50),
            'mya_std': np.random.randn(50),
            'rza_std': np.random.randn(50),
            'motion_intensity': np.random.rand(50) * 2,
            'pitch_angle': np.random.randn(50) * 0.5,
            'roll_angle': np.random.randn(50) * 0.5,
            'activity_score': np.random.rand(50),
            'postural_stability': np.random.rand(50) * 0.1,
            'head_movement_intensity': np.random.rand(50) * 10
        })
    
    def test_create_feature_vector_auto_detect(self):
        """Test automatic feature detection."""
        result = create_feature_vector(self.data, include_raw_normalized=True)
        
        # Should include normalized and engineered features
        self.assertGreater(len(result.columns), 5)
        
        # Should be all numeric
        self.assertTrue(all(result.dtypes.apply(lambda x: np.issubdtype(x, np.number))))
    
    def test_create_feature_vector_specific_columns(self):
        """Test with specific feature columns."""
        feature_cols = ['motion_intensity', 'pitch_angle', 'activity_score']
        result = create_feature_vector(self.data, feature_columns=feature_cols)
        
        # Should only include specified columns
        self.assertEqual(len(result.columns), 3)
        self.assertIn('motion_intensity', result.columns)
        self.assertIn('pitch_angle', result.columns)
    
    def test_create_feature_vector_no_nan(self):
        """Test that output has no NaN values."""
        # Add some NaN values
        data_with_nan = self.data.copy()
        data_with_nan.loc[0, 'motion_intensity'] = np.nan
        data_with_nan.loc[5, 'pitch_angle'] = np.nan
        
        result = create_feature_vector(data_with_nan)
        
        # Should have no NaNs after processing
        self.assertEqual(result.isnull().sum().sum(), 0)
    
    def test_create_feature_vector_sklearn_compatible(self):
        """Test that output is compatible with scikit-learn."""
        result = create_feature_vector(self.data)
        
        # Should be able to convert to numpy array
        X = result.values
        self.assertIsInstance(X, np.ndarray)
        
        # Should have 2D shape
        self.assertEqual(len(X.shape), 2)
        
        # Should have samples and features
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)
    
    def test_create_feature_vector_exclude_raw(self):
        """Test excluding raw normalized values."""
        result = create_feature_vector(self.data, include_raw_normalized=False)
        
        # Should not include _norm or _std columns
        for col in result.columns:
            self.assertFalse(col.endswith('_norm'))
            self.assertFalse(col.endswith('_std'))


class TestIntegration(unittest.TestCase):
    """Integration tests for complete pipeline."""
    
    def test_full_pipeline_lying_behavior(self):
        """Test full pipeline with simulated lying behavior."""
        # Simulate lying behavior (from behavioral signatures)
        data = pd.DataFrame({
            'temperature': np.random.normal(38.5, 0.2, 100),
            'fxa': np.random.normal(0.0, 0.05, 100),
            'mya': np.random.normal(0.0, 0.04, 100),
            'rza': np.random.normal(-0.8, 0.1, 100),  # Lying orientation
            'sxg': np.random.normal(0.0, 2.0, 100),
            'lyg': np.random.normal(0.0, 2.0, 100),
            'dzg': np.random.normal(0.0, 1.5, 100)
        })
        
        # Engineer features
        features = engineer_features(data, include_rhythmic=False)
        
        # Check that motion intensity is low (lying)
        mean_motion = features['motion_intensity'].mean()
        self.assertLess(mean_motion, 1.0)
        
        # Check that pitch angle indicates lying (negative)
        mean_pitch = features['pitch_angle'].mean()
        self.assertLess(mean_pitch, 0)
    
    def test_full_pipeline_walking_behavior(self):
        """Test full pipeline with simulated walking behavior."""
        # Simulate walking behavior
        data = pd.DataFrame({
            'temperature': np.random.normal(38.7, 0.2, 100),
            'fxa': np.random.normal(0.4, 0.15, 100),  # Forward acceleration
            'mya': np.random.normal(0.0, 0.1, 100),
            'rza': np.random.normal(0.85, 0.08, 100),  # Upright
            'sxg': np.random.normal(0.0, 6.0, 100),
            'lyg': np.random.normal(0.0, 5.0, 100),
            'dzg': np.random.normal(0.0, 4.0, 100)
        })
        
        # Engineer features
        features = engineer_features(data, include_rhythmic=False)
        
        # Check that motion intensity is higher (walking)
        mean_motion = features['motion_intensity'].mean()
        self.assertGreater(mean_motion, 0.5)
        
        # Check that pitch angle indicates upright (positive)
        mean_pitch = features['pitch_angle'].mean()
        self.assertGreater(mean_pitch, 0.5)


if __name__ == '__main__':
    unittest.main()
