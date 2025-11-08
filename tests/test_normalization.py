"""
Unit Tests for Normalization Module

Tests all normalization functions with various input types and edge cases.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_processing.normalization import (
    normalize_temperature,
    standardize_acceleration,
    standardize_angular_velocity,
    normalize_sensor_data,
    inverse_normalize_temperature,
    inverse_standardize_acceleration,
    inverse_standardize_angular_velocity,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    ACCELERATION_EXPECTED_MEAN,
    ACCELERATION_EXPECTED_STD,
    GYROSCOPE_EXPECTED_MEAN,
    GYROSCOPE_EXPECTED_STD
)


class TestTemperatureNormalization(unittest.TestCase):
    """Test temperature normalization functions."""
    
    def test_normalize_temperature_scalar(self):
        """Test normalization with scalar values."""
        # Test midpoint
        result = normalize_temperature(38.5)
        self.assertAlmostEqual(result, 0.5, places=5)
        
        # Test minimum
        result = normalize_temperature(35.0)
        self.assertAlmostEqual(result, 0.0, places=5)
        
        # Test maximum
        result = normalize_temperature(42.0)
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_normalize_temperature_array(self):
        """Test normalization with numpy arrays."""
        temps = np.array([35.0, 38.5, 42.0])
        expected = np.array([0.0, 0.5, 1.0])
        result = normalize_temperature(temps)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_normalize_temperature_series(self):
        """Test normalization with pandas Series."""
        temps = pd.Series([35.0, 38.5, 42.0])
        result = normalize_temperature(temps)
        expected = pd.Series([0.0, 0.5, 1.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_normalize_temperature_with_nan(self):
        """Test normalization with NaN values."""
        # Scalar
        result = normalize_temperature(np.nan)
        self.assertTrue(np.isnan(result))
        
        # Series
        temps = pd.Series([35.0, np.nan, 42.0])
        result = normalize_temperature(temps)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[0], 0.0)
        self.assertAlmostEqual(result.iloc[2], 1.0)
    
    def test_inverse_normalize_temperature(self):
        """Test inverse transformation."""
        # Round trip test
        original = 38.5
        normalized = normalize_temperature(original)
        recovered = inverse_normalize_temperature(normalized)
        self.assertAlmostEqual(recovered, original, places=5)
        
        # Array test
        original_arr = np.array([35.0, 38.5, 42.0])
        normalized_arr = normalize_temperature(original_arr)
        recovered_arr = inverse_normalize_temperature(normalized_arr)
        np.testing.assert_array_almost_equal(recovered_arr, original_arr)


class TestAccelerationStandardization(unittest.TestCase):
    """Test acceleration standardization functions."""
    
    def test_standardize_acceleration_scalar(self):
        """Test standardization with scalar values."""
        # Test mean value
        result = standardize_acceleration(0.0)
        self.assertAlmostEqual(result, 0.0, places=5)
        
        # Test one std above mean
        result = standardize_acceleration(1.0)
        self.assertAlmostEqual(result, 1.0, places=5)
        
        # Test one std below mean
        result = standardize_acceleration(-1.0)
        self.assertAlmostEqual(result, -1.0, places=5)
    
    def test_standardize_acceleration_array(self):
        """Test standardization with numpy arrays."""
        accel = np.array([-2.0, 0.0, 2.0])
        expected = np.array([-2.0, 0.0, 2.0])
        result = standardize_acceleration(accel)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_standardize_acceleration_series(self):
        """Test standardization with pandas Series."""
        accel = pd.Series([-1.0, 0.0, 1.0])
        result = standardize_acceleration(accel)
        expected = pd.Series([-1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_standardize_acceleration_with_nan(self):
        """Test standardization with NaN values."""
        # Scalar
        result = standardize_acceleration(np.nan)
        self.assertTrue(np.isnan(result))
        
        # Series
        accel = pd.Series([-1.0, np.nan, 1.0])
        result = standardize_acceleration(accel)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[0], -1.0)
        self.assertAlmostEqual(result.iloc[2], 1.0)
    
    def test_inverse_standardize_acceleration(self):
        """Test inverse transformation."""
        # Round trip test
        original = 0.5
        standardized = standardize_acceleration(original)
        recovered = inverse_standardize_acceleration(standardized)
        self.assertAlmostEqual(recovered, original, places=5)
        
        # Array test
        original_arr = np.array([-1.0, 0.0, 1.0])
        standardized_arr = standardize_acceleration(original_arr)
        recovered_arr = inverse_standardize_acceleration(standardized_arr)
        np.testing.assert_array_almost_equal(recovered_arr, original_arr)


class TestAngularVelocityStandardization(unittest.TestCase):
    """Test angular velocity standardization functions."""
    
    def test_standardize_angular_velocity_scalar(self):
        """Test standardization with scalar values."""
        # Test mean value
        result = standardize_angular_velocity(0.0)
        self.assertAlmostEqual(result, 0.0, places=5)
        
        # Test one std above mean (20 deg/s)
        result = standardize_angular_velocity(20.0)
        self.assertAlmostEqual(result, 1.0, places=5)
        
        # Test two std below mean (-40 deg/s)
        result = standardize_angular_velocity(-40.0)
        self.assertAlmostEqual(result, -2.0, places=5)
    
    def test_standardize_angular_velocity_array(self):
        """Test standardization with numpy arrays."""
        gyro = np.array([-40.0, 0.0, 20.0])
        expected = np.array([-2.0, 0.0, 1.0])
        result = standardize_angular_velocity(gyro)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_standardize_angular_velocity_series(self):
        """Test standardization with pandas Series."""
        gyro = pd.Series([-20.0, 0.0, 20.0])
        result = standardize_angular_velocity(gyro)
        expected = pd.Series([-1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_standardize_angular_velocity_with_nan(self):
        """Test standardization with NaN values."""
        # Scalar
        result = standardize_angular_velocity(np.nan)
        self.assertTrue(np.isnan(result))
        
        # Series
        gyro = pd.Series([-20.0, np.nan, 20.0])
        result = standardize_angular_velocity(gyro)
        self.assertTrue(pd.isna(result.iloc[1]))
        self.assertAlmostEqual(result.iloc[0], -1.0)
        self.assertAlmostEqual(result.iloc[2], 1.0)
    
    def test_inverse_standardize_angular_velocity(self):
        """Test inverse transformation."""
        # Round trip test
        original = 15.0
        standardized = standardize_angular_velocity(original)
        recovered = inverse_standardize_angular_velocity(standardized)
        self.assertAlmostEqual(recovered, original, places=5)
        
        # Array test
        original_arr = np.array([-20.0, 0.0, 20.0])
        standardized_arr = standardize_angular_velocity(original_arr)
        recovered_arr = inverse_standardize_angular_velocity(standardized_arr)
        np.testing.assert_array_almost_equal(recovered_arr, original_arr)


class TestNormalizeSensorData(unittest.TestCase):
    """Test the comprehensive normalize_sensor_data function."""
    
    def setUp(self):
        """Create sample sensor data."""
        self.data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [35.0, 38.5, 42.0, 37.0, 40.0],
            'fxa': [-2.0, 0.0, 2.0, 0.5, -0.5],
            'mya': [-1.0, 0.0, 1.0, 0.3, -0.3],
            'rza': [-1.0, 0.0, 1.0, 0.8, -0.8],
            'sxg': [-40.0, 0.0, 20.0, 10.0, -10.0],
            'lyg': [-20.0, 0.0, 20.0, 15.0, -15.0],
            'dzg': [0.0, 10.0, -10.0, 5.0, -5.0]
        })
    
    def test_normalize_all_sensors(self):
        """Test normalization of all sensor types."""
        result = normalize_sensor_data(self.data)
        
        # Check that normalized columns exist
        self.assertIn('temperature_norm', result.columns)
        self.assertIn('fxa_std', result.columns)
        self.assertIn('mya_std', result.columns)
        self.assertIn('rza_std', result.columns)
        self.assertIn('sxg_std', result.columns)
        self.assertIn('lyg_std', result.columns)
        self.assertIn('dzg_std', result.columns)
        
        # Check specific values
        self.assertAlmostEqual(result['temperature_norm'].iloc[1], 0.5)  # 38.5 -> 0.5
        self.assertAlmostEqual(result['fxa_std'].iloc[1], 0.0)  # 0.0 -> 0.0
        self.assertAlmostEqual(result['sxg_std'].iloc[2], 1.0)  # 20.0 -> 1.0
    
    def test_selective_normalization(self):
        """Test selective normalization of sensor types."""
        result = normalize_sensor_data(
            self.data,
            normalize_temp=True,
            standardize_accel=False,
            standardize_gyro=False
        )
        
        # Only temperature should be normalized
        self.assertIn('temperature_norm', result.columns)
        self.assertNotIn('fxa_std', result.columns)
        self.assertNotIn('sxg_std', result.columns)
    
    def test_preserve_original_data(self):
        """Test that original data is preserved."""
        original_data = self.data.copy()
        result = normalize_sensor_data(self.data)
        
        # Original columns should still exist
        pd.testing.assert_frame_equal(self.data, original_data)
        self.assertIn('temperature', result.columns)
        self.assertIn('fxa', result.columns)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_out_of_range_temperature(self):
        """Test normalization with out-of-range temperatures."""
        # Below minimum
        result = normalize_temperature(30.0)
        self.assertLess(result, 0.0)
        
        # Above maximum
        result = normalize_temperature(45.0)
        self.assertGreater(result, 1.0)
    
    def test_extreme_accelerations(self):
        """Test standardization with extreme accelerations."""
        # Well beyond expected range
        result = standardize_acceleration(5.0)
        self.assertAlmostEqual(result, 5.0)  # Should still work
    
    def test_extreme_angular_velocities(self):
        """Test standardization with extreme angular velocities."""
        # Beyond sensor range
        result = standardize_angular_velocity(100.0)
        self.assertAlmostEqual(result, 5.0)  # 100/20 = 5.0
    
    def test_empty_dataframe(self):
        """Test normalization with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = normalize_sensor_data(empty_df)
        self.assertEqual(len(result), 0)
    
    def test_missing_columns(self):
        """Test normalization with missing sensor columns."""
        partial_df = pd.DataFrame({
            'temperature': [38.5],
            'fxa': [0.5]
        })
        result = normalize_sensor_data(partial_df)
        
        # Should still work with available columns
        self.assertIn('temperature_norm', result.columns)
        self.assertIn('fxa_std', result.columns)
        # Missing columns shouldn't cause errors
        self.assertNotIn('sxg_std', result.columns)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability and precision."""
    
    def test_round_trip_precision(self):
        """Test that round-trip conversions maintain precision."""
        original_temps = np.linspace(35.0, 42.0, 100)
        
        for temp in original_temps:
            normalized = normalize_temperature(temp)
            recovered = inverse_normalize_temperature(normalized)
            self.assertAlmostEqual(recovered, temp, places=10)
    
    def test_large_arrays(self):
        """Test with large arrays for performance and stability."""
        large_array = np.random.randn(10000) * 10  # Random accelerations
        result = standardize_acceleration(large_array)
        
        # Should maintain array size
        self.assertEqual(len(result), len(large_array))
        
        # Should not contain inf or unexpected NaN
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_very_small_values(self):
        """Test with very small values near zero."""
        small_values = np.array([1e-10, -1e-10, 1e-15])
        result = standardize_acceleration(small_values)
        
        # Should handle without underflow
        self.assertFalse(np.any(np.isnan(result)))


if __name__ == '__main__':
    unittest.main()
