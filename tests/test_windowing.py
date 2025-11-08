"""
Unit tests for windowing module.

Tests window generation, rolling statistics, buffer management,
edge cases, and integration with validation module.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.windowing import (
    WindowGenerator,
    WindowStatistics,
    create_window_summary,
)
from data_processing.ingestion import DataIngestionModule


class TestWindowStatistics(unittest.TestCase):
    """Test WindowStatistics container class."""
    
    def test_initialization(self):
        """Test WindowStatistics initialization."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 5, 0)
        stats = {
            'temperature': {'mean': 38.5, 'variance': 0.1, 'min': 38.4, 'max': 38.6},
        }
        
        ws = WindowStatistics(
            window_start=start,
            window_end=end,
            statistics=stats,
            sample_count=5,
        )
        
        self.assertEqual(ws.window_start, start)
        self.assertEqual(ws.window_end, end)
        self.assertEqual(ws.sample_count, 5)
        self.assertEqual(ws.statistics['temperature']['mean'], 38.5)
    
    def test_to_dict(self):
        """Test conversion to flat dictionary."""
        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 0, 5, 0)
        stats = {
            'temperature': {'mean': 38.5, 'variance': 0.1, 'min': 38.4, 'max': 38.6},
            'fxa': {'mean': 0.0, 'variance': 0.01, 'min': -0.1, 'max': 0.1},
        }
        
        ws = WindowStatistics(
            window_start=start,
            window_end=end,
            statistics=stats,
            sample_count=5,
        )
        
        result = ws.to_dict()
        
        self.assertIn('window_start', result)
        self.assertIn('window_end', result)
        self.assertIn('sample_count', result)
        self.assertIn('temperature_mean', result)
        self.assertIn('temperature_variance', result)
        self.assertIn('fxa_mean', result)
        self.assertEqual(result['temperature_mean'], 38.5)
        self.assertEqual(result['sample_count'], 5)


class TestWindowGenerator(unittest.TestCase):
    """Test WindowGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data (30 minutes at 1-minute intervals)
        timestamps = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=30,
            freq='1min',
        )
        
        self.sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': np.random.uniform(38.0, 39.0, 30),
            'fxa': np.random.uniform(-0.1, 0.1, 30),
            'mya': np.random.uniform(-0.1, 0.1, 30),
            'rza': np.random.uniform(-1.0, -0.7, 30),
            'sxg': np.random.uniform(1.5, 2.5, 30),
            'lyg': np.random.uniform(-2.0, -1.0, 30),
            'dzg': np.random.uniform(0.5, 1.5, 30),
        })
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        self.assertEqual(generator.window_size_minutes, 5)
        self.assertEqual(generator.slide_interval_minutes, 1)
    
    def test_initialization_invalid_window_size(self):
        """Test initialization with invalid window size."""
        with self.assertRaises(ValueError):
            WindowGenerator(window_size_minutes=3)  # Too small
        
        with self.assertRaises(ValueError):
            WindowGenerator(window_size_minutes=15)  # Too large
    
    def test_initialization_invalid_slide_interval(self):
        """Test initialization with invalid slide interval."""
        with self.assertRaises(ValueError):
            WindowGenerator(window_size_minutes=5, slide_interval_minutes=0)
    
    def test_generate_sliding_windows_basic(self):
        """Test basic sliding window generation."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        windows = generator.generate_sliding_windows(self.sample_data)
        
        # Should have multiple overlapping windows
        self.assertGreater(len(windows), 0)
        
        # Check first window
        self.assertIsInstance(windows[0], WindowStatistics)
        self.assertEqual(windows[0].sample_count, 5)  # 5 minutes at 1/min
        
        # Verify statistics calculated for all sensors
        for sensor in WindowGenerator.SENSOR_COLUMNS:
            self.assertIn(sensor, windows[0].statistics)
            self.assertIn('mean', windows[0].statistics[sensor])
            self.assertIn('variance', windows[0].statistics[sensor])
            self.assertIn('min', windows[0].statistics[sensor])
            self.assertIn('max', windows[0].statistics[sensor])
    
    def test_generate_sliding_windows_overlap(self):
        """Test that sliding windows overlap correctly."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        windows = generator.generate_sliding_windows(self.sample_data)
        
        # Windows should overlap - check first two windows
        if len(windows) >= 2:
            # Second window should start 1 minute after first
            time_diff = (windows[1].window_start - windows[0].window_start).total_seconds()
            self.assertEqual(time_diff, 60)  # 1 minute in seconds
    
    def test_generate_fixed_windows_basic(self):
        """Test basic fixed window generation."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        windows = generator.generate_fixed_windows(self.sample_data)
        
        # Should have non-overlapping windows
        self.assertGreater(len(windows), 0)
        
        # Check first window
        self.assertIsInstance(windows[0], WindowStatistics)
        self.assertEqual(windows[0].sample_count, 5)  # 5 minutes at 1/min
    
    def test_generate_fixed_windows_non_overlapping(self):
        """Test that fixed windows don't overlap."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        windows = generator.generate_fixed_windows(self.sample_data)
        
        # Windows should not overlap - check consecutive windows
        if len(windows) >= 2:
            # Second window should start where first ends
            self.assertEqual(windows[1].window_start, windows[0].window_end)
    
    def test_generate_windows_with_10_minute_window(self):
        """Test window generation with 10-minute window size."""
        generator = WindowGenerator(window_size_minutes=10, slide_interval_minutes=1)
        windows = generator.generate_sliding_windows(self.sample_data)
        
        # Should have windows with 10 samples each
        self.assertGreater(len(windows), 0)
        self.assertEqual(windows[0].sample_count, 10)
    
    def test_calculate_statistics_accuracy(self):
        """Test accuracy of calculated statistics."""
        # Create data with known statistics
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=5, freq='1min')
        test_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.0, 38.5, 39.0, 38.5, 38.0],  # Mean should be 38.4
            'fxa': [0.0, 0.0, 0.0, 0.0, 0.0],
            'mya': [0.0, 0.0, 0.0, 0.0, 0.0],
            'rza': [-1.0, -1.0, -1.0, -1.0, -1.0],
            'sxg': [2.0, 2.0, 2.0, 2.0, 2.0],
            'lyg': [-1.5, -1.5, -1.5, -1.5, -1.5],
            'dzg': [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=5)
        windows = generator.generate_sliding_windows(test_data)
        
        self.assertEqual(len(windows), 1)
        
        # Check temperature statistics
        temp_stats = windows[0].statistics['temperature']
        self.assertAlmostEqual(temp_stats['mean'], 38.4, places=5)
        self.assertAlmostEqual(temp_stats['min'], 38.0, places=5)
        self.assertAlmostEqual(temp_stats['max'], 39.0, places=5)
        
        # Check fxa statistics (all zeros)
        fxa_stats = windows[0].statistics['fxa']
        self.assertAlmostEqual(fxa_stats['mean'], 0.0, places=5)
        self.assertAlmostEqual(fxa_stats['variance'], 0.0, places=5)
    
    def test_handle_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        generator = WindowGenerator(window_size_minutes=5)
        empty_df = pd.DataFrame()
        
        windows = generator.generate_sliding_windows(empty_df)
        self.assertEqual(len(windows), 0)
    
    def test_handle_missing_timestamp_column(self):
        """Test handling of missing timestamp column."""
        generator = WindowGenerator(window_size_minutes=5)
        df = pd.DataFrame({'temperature': [38.5, 38.6]})
        
        with self.assertRaises(ValueError):
            generator.generate_sliding_windows(df)
    
    def test_handle_insufficient_data_for_window(self):
        """Test handling when data is insufficient for window size."""
        # Create data with only 3 samples
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=3, freq='1min')
        small_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5, 38.6, 38.5],
            'fxa': [0.0, 0.0, 0.0],
            'mya': [0.0, 0.0, 0.0],
            'rza': [-1.0, -1.0, -1.0],
            'sxg': [2.0, 2.0, 2.0],
            'lyg': [-1.5, -1.5, -1.5],
            'dzg': [1.0, 1.0, 1.0],
        })
        
        generator = WindowGenerator(window_size_minutes=5, min_samples_per_window=4)
        windows = generator.generate_sliding_windows(small_data)
        
        # Should not create windows with insufficient data
        self.assertEqual(len(windows), 0)
    
    def test_handle_missing_sensor_values(self):
        """Test handling of missing sensor values (NaN)."""
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=5, freq='1min')
        data_with_nan = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5, np.nan, 38.6, 38.5, np.nan],
            'fxa': [0.0, 0.0, 0.0, 0.0, 0.0],
            'mya': [0.0, 0.0, 0.0, 0.0, 0.0],
            'rza': [-1.0, -1.0, -1.0, -1.0, -1.0],
            'sxg': [2.0, 2.0, 2.0, 2.0, 2.0],
            'lyg': [-1.5, -1.5, -1.5, -1.5, -1.5],
            'dzg': [1.0, 1.0, 1.0, 1.0, 1.0],
        })
        
        generator = WindowGenerator(window_size_minutes=5)
        windows = generator.generate_sliding_windows(data_with_nan)
        
        self.assertEqual(len(windows), 1)
        
        # Statistics should be calculated from valid values only
        temp_stats = windows[0].statistics['temperature']
        # Mean of [38.5, 38.6, 38.5] = 38.533...
        self.assertAlmostEqual(temp_stats['mean'], 38.533, places=2)
    
    def test_to_dataframe_conversion(self):
        """Test conversion of windows to DataFrame."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        windows = generator.generate_sliding_windows(self.sample_data)
        
        df = generator.to_dataframe(windows)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('window_start', df.columns)
        self.assertIn('window_end', df.columns)
        self.assertIn('sample_count', df.columns)
        
        # Check that all sensor statistics are present
        for sensor in WindowGenerator.SENSOR_COLUMNS:
            self.assertIn(f'{sensor}_mean', df.columns)
            self.assertIn(f'{sensor}_variance', df.columns)
            self.assertIn(f'{sensor}_min', df.columns)
            self.assertIn(f'{sensor}_max', df.columns)
    
    def test_to_dataframe_empty_windows(self):
        """Test to_dataframe with empty window list."""
        generator = WindowGenerator(window_size_minutes=5)
        df = generator.to_dataframe([])
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
    
    def test_buffer_management_update(self):
        """Test buffer update functionality."""
        generator = WindowGenerator(window_size_minutes=5)
        
        # Add first batch
        timestamps1 = pd.date_range(start='2024-01-01 00:00:00', periods=5, freq='1min')
        data1 = pd.DataFrame({
            'timestamp': timestamps1,
            'temperature': [38.5] * 5,
            'fxa': [0.0] * 5,
            'mya': [0.0] * 5,
            'rza': [-1.0] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [1.0] * 5,
        })
        
        generator.update_buffer(data1)
        self.assertEqual(len(generator.buffer), 5)
        
        # Add second batch
        timestamps2 = pd.date_range(start='2024-01-01 00:05:00', periods=5, freq='1min')
        data2 = pd.DataFrame({
            'timestamp': timestamps2,
            'temperature': [38.6] * 5,
            'fxa': [0.0] * 5,
            'mya': [0.0] * 5,
            'rza': [-1.0] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [1.0] * 5,
        })
        
        generator.update_buffer(data2)
        self.assertEqual(len(generator.buffer), 10)
    
    def test_buffer_management_trimming(self):
        """Test that buffer is trimmed to max size."""
        generator = WindowGenerator(window_size_minutes=5)
        
        # Add 20 minutes of data (should be trimmed to ~10 minutes)
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=20, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * 20,
            'fxa': [0.0] * 20,
            'mya': [0.0] * 20,
            'rza': [-1.0] * 20,
            'sxg': [2.0] * 20,
            'lyg': [-1.5] * 20,
            'dzg': [1.0] * 20,
        })
        
        generator.update_buffer(data)
        
        # Buffer should be trimmed to buffer_max_size (10 minutes for 5-min window)
        self.assertLessEqual(len(generator.buffer), generator.buffer_max_size + 1)
    
    def test_process_buffer_sliding(self):
        """Test processing buffer in sliding mode."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        
        # Add 10 minutes of data
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=10, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * 10,
            'fxa': [0.0] * 10,
            'mya': [0.0] * 10,
            'rza': [-1.0] * 10,
            'sxg': [2.0] * 10,
            'lyg': [-1.5] * 10,
            'dzg': [1.0] * 10,
        })
        
        generator.update_buffer(data)
        windows = generator.process_buffer(mode='sliding')
        
        # Should generate overlapping windows
        self.assertGreater(len(windows), 0)
    
    def test_process_buffer_fixed(self):
        """Test processing buffer in fixed mode."""
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        
        # Add 10 minutes of data
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=10, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * 10,
            'fxa': [0.0] * 10,
            'mya': [0.0] * 10,
            'rza': [-1.0] * 10,
            'sxg': [2.0] * 10,
            'lyg': [-1.5] * 10,
            'dzg': [1.0] * 10,
        })
        
        generator.update_buffer(data)
        windows = generator.process_buffer(mode='fixed')
        
        # Should generate 2 non-overlapping windows
        self.assertEqual(len(windows), 2)
    
    def test_clear_buffer(self):
        """Test buffer clearing."""
        generator = WindowGenerator(window_size_minutes=5)
        
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=5, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * 5,
            'fxa': [0.0] * 5,
            'mya': [0.0] * 5,
            'rza': [-1.0] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [1.0] * 5,
        })
        
        generator.update_buffer(data)
        self.assertIsNotNone(generator.buffer)
        
        generator.clear_buffer()
        self.assertIsNone(generator.buffer)
    
    def test_get_buffer_info(self):
        """Test getting buffer information."""
        generator = WindowGenerator(window_size_minutes=5)
        
        # Empty buffer
        info = generator.get_buffer_info()
        self.assertEqual(info['size'], 0)
        
        # Add data
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=10, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * 10,
            'fxa': [0.0] * 10,
            'mya': [0.0] * 10,
            'rza': [-1.0] * 10,
            'sxg': [2.0] * 10,
            'lyg': [-1.5] * 10,
            'dzg': [1.0] * 10,
        })
        
        generator.update_buffer(data)
        info = generator.get_buffer_info()
        
        self.assertEqual(info['size'], 10)
        self.assertIsNotNone(info['time_range'])
        self.assertAlmostEqual(info['duration_minutes'], 9.0, places=1)
    
    def test_optional_statistics_median(self):
        """Test inclusion of median statistic."""
        generator = WindowGenerator(
            window_size_minutes=5,
            include_median=True,
        )
        
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=5, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.0, 38.5, 39.0, 38.5, 38.0],
            'fxa': [0.0] * 5,
            'mya': [0.0] * 5,
            'rza': [-1.0] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [1.0] * 5,
        })
        
        windows = generator.generate_sliding_windows(data)
        
        self.assertEqual(len(windows), 1)
        self.assertIn('median', windows[0].statistics['temperature'])
        self.assertAlmostEqual(
            windows[0].statistics['temperature']['median'],
            38.5,
            places=5,
        )
    
    def test_optional_statistics_std(self):
        """Test inclusion of standard deviation statistic."""
        generator = WindowGenerator(
            window_size_minutes=5,
            include_std=True,
        )
        
        timestamps = pd.date_range(start='2024-01-01 00:00:00', periods=5, freq='1min')
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.0, 38.5, 39.0, 38.5, 38.0],
            'fxa': [0.0] * 5,
            'mya': [0.0] * 5,
            'rza': [-1.0] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [1.0] * 5,
        })
        
        windows = generator.generate_sliding_windows(data)
        
        self.assertEqual(len(windows), 1)
        self.assertIn('std', windows[0].statistics['temperature'])
        self.assertGreater(windows[0].statistics['temperature']['std'], 0)


class TestIntegrationWithIngestion(unittest.TestCase):
    """Test integration with data ingestion module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures_dir = Path(__file__).parent / 'fixtures'
    
    def test_windowing_after_ingestion(self):
        """Test windowing on data loaded via ingestion module."""
        # Load data using ingestion module
        ingestion = DataIngestionModule(validate_data=True)
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df, summary = ingestion.load_batch(str(csv_file))
        
        self.assertGreater(len(df), 0)
        
        # Apply windowing
        generator = WindowGenerator(window_size_minutes=5, slide_interval_minutes=1)
        windows = generator.generate_sliding_windows(df)
        
        # Should generate windows from validated data
        self.assertGreater(len(windows), 0)
        
        # Check that all sensor statistics are present
        for window in windows:
            for sensor in WindowGenerator.SENSOR_COLUMNS:
                self.assertIn(sensor, window.statistics)
    
    def test_windowing_with_validated_data(self):
        """Test that windowing works with validated data."""
        from data_processing.validators import DataValidator
        
        # Load and validate data
        ingestion = DataIngestionModule(validate_data=True)
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df, summary = ingestion.load_batch(str(csv_file))
        
        # Generate windows
        generator = WindowGenerator(window_size_minutes=5)
        windows = generator.generate_fixed_windows(df)
        
        # Convert to DataFrame for ML consumption
        result_df = generator.to_dataframe(windows)
        
        # Verify output format
        self.assertIsInstance(result_df, pd.DataFrame)
        self.assertIn('window_start', result_df.columns)
        self.assertIn('sample_count', result_df.columns)
        
        # Verify all sensor statistics are present
        for sensor in WindowGenerator.SENSOR_COLUMNS:
            self.assertIn(f'{sensor}_mean', result_df.columns)


class TestWindowSummary(unittest.TestCase):
    """Test window summary functionality."""
    
    def test_create_window_summary(self):
        """Test creating summary of windows."""
        # Create sample windows
        start = datetime(2024, 1, 1, 0, 0, 0)
        windows = [
            WindowStatistics(
                window_start=start,
                window_end=start + timedelta(minutes=5),
                statistics={'temperature': {'mean': 38.5}},
                sample_count=5,
            ),
            WindowStatistics(
                window_start=start + timedelta(minutes=1),
                window_end=start + timedelta(minutes=6),
                statistics={'temperature': {'mean': 38.6}},
                sample_count=5,
            ),
        ]
        
        summary = create_window_summary(windows)
        
        self.assertEqual(summary['total_windows'], 2)
        self.assertIsNotNone(summary['time_range'])
        self.assertEqual(summary['avg_samples_per_window'], 5.0)
        self.assertEqual(summary['total_samples'], 10)
    
    def test_create_window_summary_empty(self):
        """Test creating summary with no windows."""
        summary = create_window_summary([])
        
        self.assertEqual(summary['total_windows'], 0)
        self.assertIsNone(summary['time_range'])
        self.assertEqual(summary['avg_samples_per_window'], 0)


if __name__ == '__main__':
    unittest.main()
