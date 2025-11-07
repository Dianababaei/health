#!/usr/bin/env python3
"""
Unit tests for data_loader module.

This test suite validates the CSV reading functionality for sensor data,
including proper handling of different timestamp formats, error conditions,
and directory batch loading.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from src.data_loader import (
    read_sensor_csv,
    read_sensor_directory,
    _parse_timestamp_column,
    _validate_columns,
    _convert_sensor_columns_to_numeric,
    REQUIRED_COLUMNS,
    SENSOR_COLUMNS
)


class TestTimestampParsing:
    """Tests for timestamp parsing functionality."""
    
    def test_parse_iso8601_format(self):
        """Test parsing of ISO 8601 timestamp format."""
        timestamps = pd.Series([
            '2024-01-15T10:00:00',
            '2024-01-15T10:01:00',
            '2024-01-15T10:02:00'
        ])
        result = _parse_timestamp_column(timestamps)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3
        assert result[0] == pd.Timestamp('2024-01-15 10:00:00')
    
    def test_parse_common_format(self):
        """Test parsing of common YYYY-MM-DD HH:MM:SS format."""
        timestamps = pd.Series([
            '2024-01-15 10:00:00',
            '2024-01-15 10:01:00',
            '2024-01-15 10:02:00'
        ])
        result = _parse_timestamp_column(timestamps)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3
        assert result[0] == pd.Timestamp('2024-01-15 10:00:00')
    
    def test_parse_unix_epoch_seconds(self):
        """Test parsing of Unix epoch timestamp in seconds."""
        timestamps = pd.Series([1705315200, 1705315260, 1705315320])  # Jan 15, 2024
        result = _parse_timestamp_column(timestamps)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3
    
    def test_parse_unix_epoch_milliseconds(self):
        """Test parsing of Unix epoch timestamp in milliseconds."""
        timestamps = pd.Series([1705315200000, 1705315260000, 1705315320000])
        result = _parse_timestamp_column(timestamps)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3
    
    def test_parse_iso8601_with_z(self):
        """Test parsing of ISO 8601 with Z timezone."""
        timestamps = pd.Series([
            '2024-01-15T10:00:00Z',
            '2024-01-15T10:01:00Z',
            '2024-01-15T10:02:00Z'
        ])
        result = _parse_timestamp_column(timestamps)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3
    
    def test_parse_with_microseconds(self):
        """Test parsing of timestamps with microseconds."""
        timestamps = pd.Series([
            '2024-01-15 10:00:00.123456',
            '2024-01-15 10:01:00.234567',
            '2024-01-15 10:02:00.345678'
        ])
        result = _parse_timestamp_column(timestamps)
        assert isinstance(result, pd.DatetimeIndex)
        assert len(result) == 3
    
    def test_parse_invalid_timestamps(self):
        """Test that invalid timestamps raise appropriate error."""
        timestamps = pd.Series(['invalid', 'not_a_date', 'xyz'])
        with pytest.raises(ValueError, match="Unable to parse timestamp column"):
            _parse_timestamp_column(timestamps)


class TestColumnValidation:
    """Tests for column validation functionality."""
    
    def test_validate_all_columns_present(self):
        """Test validation passes with all required columns."""
        df = pd.DataFrame(columns=REQUIRED_COLUMNS)
        # Should not raise any exception
        _validate_columns(df, 'test.csv')
    
    def test_validate_missing_columns(self):
        """Test validation fails with missing columns."""
        df = pd.DataFrame(columns=['timestamp', 'temperature', 'Fxa'])
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_columns(df, 'test.csv')
    
    def test_validate_extra_columns_allowed(self):
        """Test validation passes with extra columns."""
        columns = REQUIRED_COLUMNS + ['extra_column']
        df = pd.DataFrame(columns=columns)
        # Should not raise any exception
        _validate_columns(df, 'test.csv')


class TestNumericConversion:
    """Tests for numeric conversion functionality."""
    
    def test_convert_valid_numeric_data(self):
        """Test conversion of valid numeric sensor data."""
        data = {
            'temperature': ['38.5', '38.6', '38.7'],
            'Fxa': ['1.2', '1.3', '1.4'],
            'Mya': ['0.5', '0.6', '0.7'],
            'Rza': ['2.1', '2.2', '2.3'],
            'Sxg': ['0.1', '0.2', '0.3'],
            'Lyg': ['0.4', '0.5', '0.6'],
            'Dzg': ['1.0', '1.1', '1.2']
        }
        df = pd.DataFrame(data)
        result = _convert_sensor_columns_to_numeric(df, 'test.csv')
        
        for col in SENSOR_COLUMNS:
            assert pd.api.types.is_numeric_dtype(result[col])
    
    def test_convert_with_invalid_values(self):
        """Test conversion handles non-numeric values by setting to NaN."""
        data = {
            'temperature': ['38.5', 'invalid', '38.7'],
            'Fxa': ['1.2', '1.3', 'NaN'],
            'Mya': ['0.5', '0.6', '0.7'],
            'Rza': ['2.1', 'error', '2.3'],
            'Sxg': ['0.1', '0.2', '0.3'],
            'Lyg': ['0.4', '0.5', '0.6'],
            'Dzg': ['1.0', '1.1', '1.2']
        }
        df = pd.DataFrame(data)
        result = _convert_sensor_columns_to_numeric(df, 'test.csv')
        
        # Check that invalid values became NaN
        assert pd.isna(result.loc[1, 'temperature'])
        assert pd.isna(result.loc[1, 'Rza'])
        
        # Check that valid values are preserved
        assert result.loc[0, 'temperature'] == 38.5
        assert result.loc[2, 'temperature'] == 38.7


class TestReadSensorCSV:
    """Tests for read_sensor_csv function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def create_test_csv(self, filepath, num_rows=10, timestamp_format='standard'):
        """Helper to create a test CSV file."""
        data = {
            'timestamp': [],
            'temperature': [],
            'Fxa': [],
            'Mya': [],
            'Rza': [],
            'Sxg': [],
            'Lyg': [],
            'Dzg': []
        }
        
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        for i in range(num_rows):
            current_time = base_time + timedelta(minutes=i)
            
            if timestamp_format == 'standard':
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            elif timestamp_format == 'iso8601':
                timestamp_str = current_time.strftime('%Y-%m-%dT%H:%M:%S')
            elif timestamp_format == 'epoch':
                timestamp_str = str(int(current_time.timestamp()))
            else:
                timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
            
            data['timestamp'].append(timestamp_str)
            data['temperature'].append(38.5 + i * 0.1)
            data['Fxa'].append(1.0 + i * 0.05)
            data['Mya'].append(0.5 + i * 0.02)
            data['Rza'].append(2.0 + i * 0.03)
            data['Sxg'].append(0.1 + i * 0.01)
            data['Lyg'].append(0.4 + i * 0.02)
            data['Dzg'].append(1.0 + i * 0.04)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def test_read_valid_csv(self, temp_dir):
        """Test reading a valid CSV file."""
        csv_path = temp_dir / 'test_data.csv'
        self.create_test_csv(csv_path, num_rows=10)
        
        result = read_sensor_csv(csv_path)
        
        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert list(result.columns) == SENSOR_COLUMNS
        assert isinstance(result.index, pd.DatetimeIndex)
        
        # Check data is sorted
        assert result.index.is_monotonic_increasing
        
        # Check data types
        for col in SENSOR_COLUMNS:
            assert pd.api.types.is_numeric_dtype(result[col])
    
    def test_read_iso8601_format(self, temp_dir):
        """Test reading CSV with ISO 8601 timestamp format."""
        csv_path = temp_dir / 'test_iso.csv'
        self.create_test_csv(csv_path, num_rows=5, timestamp_format='iso8601')
        
        result = read_sensor_csv(csv_path)
        
        assert len(result) == 5
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_read_epoch_format(self, temp_dir):
        """Test reading CSV with Unix epoch timestamp format."""
        csv_path = temp_dir / 'test_epoch.csv'
        self.create_test_csv(csv_path, num_rows=5, timestamp_format='epoch')
        
        result = read_sensor_csv(csv_path)
        
        assert len(result) == 5
        assert isinstance(result.index, pd.DatetimeIndex)
    
    def test_read_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_sensor_csv('nonexistent_file.csv')
    
    def test_read_empty_file(self, temp_dir):
        """Test handling of empty CSV file."""
        csv_path = temp_dir / 'empty.csv'
        # Create empty file
        csv_path.write_text('')
        
        with pytest.raises((ValueError, pd.errors.EmptyDataError)):
            read_sensor_csv(csv_path)
    
    def test_read_missing_columns(self, temp_dir):
        """Test error handling for missing required columns."""
        csv_path = temp_dir / 'incomplete.csv'
        # Create CSV with only some columns
        data = {
            'timestamp': ['2024-01-15 10:00:00', '2024-01-15 10:01:00'],
            'temperature': [38.5, 38.6],
            'Fxa': [1.2, 1.3]
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="missing required columns"):
            read_sensor_csv(csv_path)
    
    def test_read_unsorted_timestamps(self, temp_dir):
        """Test that unsorted timestamps are properly sorted."""
        csv_path = temp_dir / 'unsorted.csv'
        data = {
            'timestamp': ['2024-01-15 10:02:00', '2024-01-15 10:00:00', '2024-01-15 10:01:00'],
            'temperature': [38.7, 38.5, 38.6],
            'Fxa': [1.4, 1.2, 1.3],
            'Mya': [0.7, 0.5, 0.6],
            'Rza': [2.3, 2.1, 2.2],
            'Sxg': [0.3, 0.1, 0.2],
            'Lyg': [0.6, 0.4, 0.5],
            'Dzg': [1.2, 1.0, 1.1]
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        result = read_sensor_csv(csv_path)
        
        # Check that data is sorted
        assert result.index.is_monotonic_increasing
        assert result.index[0] == pd.Timestamp('2024-01-15 10:00:00')
        assert result.index[-1] == pd.Timestamp('2024-01-15 10:02:00')


class TestReadSensorDirectory:
    """Tests for read_sensor_directory function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def create_test_csv(self, filepath, start_time, num_rows=10):
        """Helper to create a test CSV file."""
        data = {
            'timestamp': [],
            'temperature': [],
            'Fxa': [],
            'Mya': [],
            'Rza': [],
            'Sxg': [],
            'Lyg': [],
            'Dzg': []
        }
        
        for i in range(num_rows):
            current_time = start_time + timedelta(minutes=i)
            data['timestamp'].append(current_time.strftime('%Y-%m-%d %H:%M:%S'))
            data['temperature'].append(38.5 + i * 0.1)
            data['Fxa'].append(1.0 + i * 0.05)
            data['Mya'].append(0.5 + i * 0.02)
            data['Rza'].append(2.0 + i * 0.03)
            data['Sxg'].append(0.1 + i * 0.01)
            data['Lyg'].append(0.4 + i * 0.02)
            data['Dzg'].append(1.0 + i * 0.04)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def test_read_directory_multiple_files(self, temp_dir):
        """Test reading multiple CSV files from a directory."""
        # Create 3 CSV files with sequential data
        self.create_test_csv(temp_dir / 'data1.csv', datetime(2024, 1, 15, 10, 0), 5)
        self.create_test_csv(temp_dir / 'data2.csv', datetime(2024, 1, 15, 10, 10), 5)
        self.create_test_csv(temp_dir / 'data3.csv', datetime(2024, 1, 15, 10, 20), 5)
        
        result = read_sensor_directory(temp_dir)
        
        # Should have combined all files
        assert len(result) == 15
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.is_monotonic_increasing
    
    def test_read_directory_with_pattern(self, temp_dir):
        """Test reading directory with file pattern."""
        # Create CSV files with different names
        self.create_test_csv(temp_dir / 'animal_A.csv', datetime(2024, 1, 15, 10, 0), 5)
        self.create_test_csv(temp_dir / 'animal_B.csv', datetime(2024, 1, 15, 11, 0), 5)
        self.create_test_csv(temp_dir / 'other.csv', datetime(2024, 1, 15, 12, 0), 5)
        
        # Read only files matching pattern
        result = read_sensor_directory(temp_dir, pattern='animal_*.csv')
        
        # Should have only loaded animal files
        assert len(result) == 10
    
    def test_read_directory_removes_duplicates(self, temp_dir):
        """Test that duplicate timestamps are removed."""
        # Create files with overlapping timestamps
        self.create_test_csv(temp_dir / 'data1.csv', datetime(2024, 1, 15, 10, 0), 10)
        self.create_test_csv(temp_dir / 'data2.csv', datetime(2024, 1, 15, 10, 5), 10)
        
        result = read_sensor_directory(temp_dir)
        
        # Should have removed duplicates
        assert len(result) == len(result.index.unique())
    
    def test_read_directory_not_found(self):
        """Test that FileNotFoundError is raised for missing directory."""
        with pytest.raises(FileNotFoundError, match="does not exist"):
            read_sensor_directory('nonexistent_directory')
    
    def test_read_directory_empty(self, temp_dir):
        """Test error handling for directory with no CSV files."""
        # Create empty directory
        with pytest.raises(ValueError, match="No CSV files found"):
            read_sensor_directory(temp_dir)
    
    def test_read_directory_not_a_directory(self, temp_dir):
        """Test error handling when path is not a directory."""
        # Create a file instead of directory
        file_path = temp_dir / 'not_a_dir.txt'
        file_path.write_text('test')
        
        with pytest.raises(ValueError, match="not a directory"):
            read_sensor_directory(file_path)
    
    def test_read_directory_with_some_invalid_files(self, temp_dir):
        """Test that valid files are still loaded even if some files fail."""
        # Create one valid and one invalid CSV
        self.create_test_csv(temp_dir / 'valid.csv', datetime(2024, 1, 15, 10, 0), 5)
        
        # Create invalid CSV (missing columns)
        invalid_data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        pd.DataFrame(invalid_data).to_csv(temp_dir / 'invalid.csv', index=False)
        
        # Should still load the valid file
        result = read_sensor_directory(temp_dir)
        assert len(result) == 5


class TestIntegration:
    """Integration tests for complete workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_full_workflow_single_file(self, temp_dir):
        """Test complete workflow with a single file."""
        # Create realistic sensor data
        csv_path = temp_dir / 'sensor_data.csv'
        
        # Generate 24 hours of minute-by-minute data (1440 rows)
        timestamps = []
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        for i in range(1440):
            timestamps.append((base_time + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'))
        
        data = {
            'timestamp': timestamps,
            'temperature': np.random.normal(38.5, 0.3, 1440),
            'Fxa': np.random.normal(1.0, 0.2, 1440),
            'Mya': np.random.normal(0.5, 0.15, 1440),
            'Rza': np.random.normal(2.0, 0.25, 1440),
            'Sxg': np.random.normal(0.1, 0.05, 1440),
            'Lyg': np.random.normal(0.4, 0.1, 1440),
            'Dzg': np.random.normal(1.0, 0.2, 1440),
        }
        
        pd.DataFrame(data).to_csv(csv_path, index=False)
        
        # Load and verify
        result = read_sensor_csv(csv_path)
        
        assert len(result) == 1440
        assert result.index[0].date() == datetime(2024, 1, 15).date()
        assert all(col in result.columns for col in SENSOR_COLUMNS)
        
        # Check temperature is in reasonable range
        assert 37.0 < result['temperature'].mean() < 40.0
    
    def test_full_workflow_directory(self, temp_dir):
        """Test complete workflow with directory of files."""
        # Create multiple days of data
        for day in range(1, 4):  # 3 days
            csv_path = temp_dir / f'day_{day}.csv'
            
            timestamps = []
            base_time = datetime(2024, 1, day, 0, 0, 0)
            for i in range(100):  # 100 minutes per file for speed
                timestamps.append((base_time + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'))
            
            data = {
                'timestamp': timestamps,
                'temperature': np.random.normal(38.5, 0.3, 100),
                'Fxa': np.random.normal(1.0, 0.2, 100),
                'Mya': np.random.normal(0.5, 0.15, 100),
                'Rza': np.random.normal(2.0, 0.25, 100),
                'Sxg': np.random.normal(0.1, 0.05, 100),
                'Lyg': np.random.normal(0.4, 0.1, 100),
                'Dzg': np.random.normal(1.0, 0.2, 100),
            }
            
            pd.DataFrame(data).to_csv(csv_path, index=False)
        
        # Load directory
        result = read_sensor_directory(temp_dir)
        
        assert len(result) == 300  # 3 files * 100 rows each
        assert result.index.is_monotonic_increasing
        assert result.index[0].day == 1
        assert result.index[-1].day == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
