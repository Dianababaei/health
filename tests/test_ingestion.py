"""
Unit tests for data ingestion module.

Tests batch loading, incremental loading, validation, error handling,
and various edge cases.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.ingestion import DataIngestionModule
from data_processing.parsers import TimestampParser, CSVParser
from data_processing.validators import DataValidator


class TestTimestampParser(unittest.TestCase):
    """Test timestamp parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = TimestampParser()
    
    def test_detect_iso8601_format(self):
        """Test detection of ISO 8601 timestamp format."""
        samples = ['2024-01-01T00:00:00', '2024-01-01T00:01:00']
        fmt = self.parser.detect_format(samples)
        self.assertEqual(fmt, '%Y-%m-%dT%H:%M:%S')
    
    def test_detect_unix_timestamp_seconds(self):
        """Test detection of Unix timestamp (seconds)."""
        samples = ['1704067200', '1704067260']
        fmt = self.parser.detect_format(samples)
        self.assertEqual(fmt, 'unix_seconds')
        self.assertTrue(self.parser.is_unix_timestamp)
    
    def test_parse_iso8601(self):
        """Test parsing ISO 8601 timestamps."""
        samples = ['2024-01-01T00:00:00']
        self.parser.detect_format(samples)
        dt = self.parser.parse('2024-01-01T00:00:00')
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 1)
    
    def test_parse_unix_timestamp(self):
        """Test parsing Unix timestamps."""
        samples = ['1704067200']
        self.parser.detect_format(samples)
        dt = self.parser.parse('1704067200')
        self.assertIsInstance(dt, datetime)
    
    def test_parse_invalid_timestamp(self):
        """Test handling of invalid timestamp."""
        samples = ['2024-01-01T00:00:00']
        self.parser.detect_format(samples)
        with self.assertRaises(ValueError):
            self.parser.parse('invalid_timestamp')


class TestCSVParser(unittest.TestCase):
    """Test CSV parsing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = CSVParser()
        self.fixtures_dir = Path(__file__).parent / 'fixtures'
    
    def test_detect_comma_delimiter(self):
        """Test detection of comma delimiter."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        delimiter = self.parser.detect_delimiter(str(csv_file))
        self.assertEqual(delimiter, ',')
    
    def test_read_valid_csv(self):
        """Test reading valid CSV file."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df = self.parser.read_csv(str(csv_file))
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        self.assertIn('temperature', df.columns)
    
    def test_read_csv_incremental(self):
        """Test incremental CSV reading."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        
        # First read (from beginning)
        df1, pos1 = self.parser.read_csv_incremental(str(csv_file), last_position=0)
        self.assertGreater(len(df1), 0)
        self.assertGreater(pos1, 0)
        
        # Second read (no new data)
        df2, pos2 = self.parser.read_csv_incremental(str(csv_file), last_position=pos1)
        self.assertEqual(len(df2), 0)
        self.assertEqual(pos2, pos1)


class TestDataValidator(unittest.TestCase):
    """Test data validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataValidator()
        self.fixtures_dir = Path(__file__).parent / 'fixtures'
    
    def test_validate_columns_valid(self):
        """Test column validation with valid data."""
        parser = CSVParser()
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df = parser.read_csv(str(csv_file))
        result = self.validator.validate_columns(df)
        self.assertTrue(result)
    
    def test_validate_columns_missing(self):
        """Test column validation with missing columns."""
        import pandas as pd
        df = pd.DataFrame({'timestamp': [1, 2, 3], 'temperature': [38.5, 38.6, 38.5]})
        result = self.validator.validate_columns(df)
        self.assertFalse(result)
    
    def test_validate_data_types(self):
        """Test data type validation."""
        parser = CSVParser()
        csv_file = self.fixtures_dir / 'sample_malformed.csv'
        df = parser.read_csv(str(csv_file))
        
        df_clean = self.validator.validate_data_types(df)
        # Should have some NaN values from invalid entries
        self.assertTrue(df_clean['fxa'].isna().any() or df_clean['mya'].isna().any())
    
    def test_validate_sensor_ranges(self):
        """Test sensor range validation."""
        parser = CSVParser()
        csv_file = self.fixtures_dir / 'sample_malformed.csv'
        df = parser.read_csv(str(csv_file))
        
        # First convert to numeric
        df = self.validator.validate_data_types(df)
        
        # Then validate ranges
        df_clean = self.validator.validate_sensor_ranges(df)
        
        # Should have errors for out-of-range values
        self.assertGreater(len(self.validator.summary.errors), 0)


class TestDataIngestionModule(unittest.TestCase):
    """Test data ingestion module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures_dir = Path(__file__).parent / 'fixtures'
        self.temp_dir = tempfile.mkdtemp()
        self.module = DataIngestionModule(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_batch_valid_file(self):
        """Test batch loading of valid CSV file."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df, summary = self.module.load_batch(str(csv_file))
        
        self.assertGreater(len(df), 0)
        self.assertEqual(summary.total_rows_read, len(df))
        self.assertGreater(summary.valid_rows, 0)
        self.assertEqual(len(summary.errors), 0)
    
    def test_load_batch_malformed_file(self):
        """Test batch loading of malformed CSV file."""
        csv_file = self.fixtures_dir / 'sample_malformed.csv'
        df, summary = self.module.load_batch(str(csv_file))
        
        # Should still load some data
        self.assertGreater(len(df), 0)
        
        # Should have validation errors/warnings
        self.assertGreater(len(summary.errors) + len(summary.warnings), 0)
        
        # Some rows should be skipped
        self.assertGreater(summary.skipped_rows, 0)
    
    def test_load_batch_nonexistent_file(self):
        """Test batch loading of non-existent file."""
        df, summary = self.module.load_batch('nonexistent_file.csv')
        
        self.assertEqual(len(df), 0)
        self.assertGreater(len(summary.errors), 0)
        self.assertIn('not found', summary.errors[0].lower())
    
    def test_load_batch_with_unix_timestamps(self):
        """Test batch loading with Unix timestamps."""
        csv_file = self.fixtures_dir / 'sample_edge_cases.csv'
        df, summary = self.module.load_batch(str(csv_file))
        
        self.assertGreater(len(df), 0)
        self.assertIn('timestamp', df.columns)
        
        # Check that timestamps were parsed
        if len(df) > 0:
            self.assertTrue(hasattr(df['timestamp'].iloc[0], 'year'))
    
    def test_load_batch_with_optional_columns(self):
        """Test batch loading with optional columns (cow_id, sensor_id)."""
        csv_file = self.fixtures_dir / 'sample_edge_cases.csv'
        df, summary = self.module.load_batch(str(csv_file))
        
        self.assertGreater(len(df), 0)
        self.assertIn('cow_id', df.columns)
        self.assertIn('sensor_id', df.columns)
    
    def test_load_incremental_first_read(self):
        """Test incremental loading (first read)."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df, summary = self.module.load_incremental(str(csv_file))
        
        self.assertGreater(len(df), 0)
        self.assertGreater(summary.valid_rows, 0)
        
        # File position should be saved
        pos = self.module.get_file_position(str(csv_file))
        self.assertGreater(pos, 0)
    
    def test_load_incremental_subsequent_read(self):
        """Test incremental loading (subsequent read with no new data)."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        
        # First read
        df1, summary1 = self.module.load_incremental(str(csv_file))
        self.assertGreater(len(df1), 0)
        
        # Second read (no new data)
        df2, summary2 = self.module.load_incremental(str(csv_file))
        self.assertEqual(len(df2), 0)
    
    def test_reset_file_position(self):
        """Test resetting file position."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        
        # Read file
        df1, _ = self.module.load_incremental(str(csv_file))
        self.assertGreater(len(df1), 0)
        
        # Reset position
        self.module.reset_file_position(str(csv_file))
        pos = self.module.get_file_position(str(csv_file))
        self.assertEqual(pos, 0)
        
        # Read again (should get data again)
        df2, _ = self.module.load_incremental(str(csv_file))
        self.assertGreater(len(df2), 0)
    
    def test_load_batch_with_chunk_size(self):
        """Test batch loading with chunk size (for large files)."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df, summary = self.module.load_batch(str(csv_file), chunk_size=10)
        
        self.assertGreater(len(df), 0)
        self.assertEqual(summary.total_rows_read, len(df))
    
    def test_export_summary(self):
        """Test exporting ingestion summary."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df, summary = self.module.load_batch(str(csv_file))
        
        output_file = os.path.join(self.temp_dir, 'summary.txt')
        self.module.export_summary(summary, output_file)
        
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
    
    def test_monitor_file_with_max_iterations(self):
        """Test file monitoring with max iterations."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        
        # Track callback invocations
        callback_count = [0]
        
        def test_callback(df, summary):
            callback_count[0] += 1
        
        # Monitor for 2 iterations (should only get data on first iteration)
        self.module.monitor_file(
            str(csv_file),
            interval_seconds=1,
            callback=test_callback,
            max_iterations=2
        )
        
        # Callback should be called at least once
        self.assertGreaterEqual(callback_count[0], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete ingestion pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fixtures_dir = Path(__file__).parent / 'fixtures'
        self.temp_dir = tempfile.mkdtemp()
        self.module = DataIngestionModule(log_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_valid_file(self):
        """Test complete pipeline with valid file."""
        csv_file = self.fixtures_dir / 'sample_valid.csv'
        df, summary = self.module.load_batch(str(csv_file))
        
        # Verify data loaded
        self.assertGreater(len(df), 0)
        
        # Verify all required columns present
        required_cols = ['timestamp', 'temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        for col in required_cols:
            self.assertIn(col, df.columns)
        
        # Verify timestamps parsed
        self.assertTrue(hasattr(df['timestamp'].iloc[0], 'year'))
        
        # Verify no critical errors
        self.assertEqual(len(summary.errors), 0)
        
        # Verify validation was performed
        self.assertIsNotNone(summary.validation_summary)
    
    def test_end_to_end_malformed_file(self):
        """Test complete pipeline with malformed file (should handle gracefully)."""
        csv_file = self.fixtures_dir / 'sample_malformed.csv'
        df, summary = self.module.load_batch(str(csv_file))
        
        # Should still load some data
        self.assertGreater(len(df), 0)
        
        # Should have validation issues
        self.assertGreater(len(summary.errors) + len(summary.warnings), 0)
        
        # Should identify specific error types
        if summary.validation_summary:
            error_types = summary.validation_summary.get_error_counts()
            self.assertGreater(len(error_types), 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
