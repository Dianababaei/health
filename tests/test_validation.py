"""
Unit Tests for Data Validation Module

Tests cover:
- Complete valid data
- Missing parameters
- Out-of-range temperatures (hypothermia, severe fever)
- Extreme accelerations
- Timestamp gaps
- Data type validation
- Edge cases
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.validation import (
    DataValidator,
    ValidationSeverity,
    ValidationReport,
    ValidationIssue,
    validate_sensor_data
)


class TestValidationIssue(unittest.TestCase):
    """Test ValidationIssue class."""
    
    def test_create_issue(self):
        """Test creating a validation issue."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category='completeness',
            message='Missing parameter',
            row_index=10,
            column='temperature'
        )
        
        self.assertEqual(issue.severity, ValidationSeverity.ERROR)
        self.assertEqual(issue.category, 'completeness')
        self.assertEqual(issue.row_index, 10)
        self.assertEqual(issue.column, 'temperature')
    
    def test_issue_to_dict(self):
        """Test converting issue to dictionary."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            category='range',
            message='Out of range',
            value=43.0
        )
        
        issue_dict = issue.to_dict()
        self.assertEqual(issue_dict['severity'], 'WARNING')
        self.assertEqual(issue_dict['category'], 'range')
        self.assertEqual(issue_dict['value'], 43.0)


class TestValidationReport(unittest.TestCase):
    """Test ValidationReport class."""
    
    def test_add_issue(self):
        """Test adding issues to report."""
        report = ValidationReport()
        
        issue1 = ValidationIssue(ValidationSeverity.ERROR, 'completeness', 'Missing data')
        issue2 = ValidationIssue(ValidationSeverity.WARNING, 'range', 'Out of range')
        
        report.add_issue(issue1)
        report.add_issue(issue2)
        
        self.assertEqual(len(report.issues), 2)
    
    def test_get_summary(self):
        """Test getting report summary."""
        report = ValidationReport()
        
        report.add_issue(ValidationIssue(ValidationSeverity.ERROR, 'completeness', 'Missing'))
        report.add_issue(ValidationIssue(ValidationSeverity.ERROR, 'type', 'Invalid type'))
        report.add_issue(ValidationIssue(ValidationSeverity.WARNING, 'range', 'Out of range'))
        report.add_issue(ValidationIssue(ValidationSeverity.INFO, 'info', 'Edge case'))
        
        report.finalize(total_records=100, clean_records=96)
        
        summary = report.get_summary()
        
        self.assertEqual(summary['total_records'], 100)
        self.assertEqual(summary['clean_records'], 96)
        self.assertEqual(summary['flagged_records'], 4)
        self.assertEqual(summary['error_count'], 2)
        self.assertEqual(summary['warning_count'], 1)
        self.assertEqual(summary['info_count'], 1)


class TestDataValidatorCompleteness(unittest.TestCase):
    """Test completeness validation."""
    
    def test_complete_valid_data(self):
        """Test validation with complete valid data."""
        # Create valid dataset
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'temperature': [38.5] * 10,
            'fxa': [0.1] * 10,
            'mya': [0.2] * 10,
            'rza': [-0.85] * 10,
            'sxg': [2.0] * 10,
            'lyg': [-1.5] * 10,
            'dzg': [0.5] * 10
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        self.assertEqual(len(clean_data), 10)
        self.assertEqual(len(flagged_data), 0)
        self.assertEqual(report.get_summary()['error_count'], 0)
    
    def test_missing_column(self):
        """Test detection of missing required column."""
        # Missing 'temperature' column
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'fxa': [0.1] * 5,
            'mya': [0.2] * 5,
            'rza': [-0.85] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [0.5] * 5
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should report missing column as ERROR
        errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR]
        self.assertTrue(any('temperature' in str(i.message).lower() for i in errors))
    
    def test_missing_values(self):
        """Test detection of missing values in rows."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [38.5, np.nan, 38.6, 38.7, np.nan],
            'fxa': [0.1] * 5,
            'mya': [0.2] * 5,
            'rza': [-0.85] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [0.5] * 5
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should flag rows with missing values
        self.assertEqual(len(flagged_data), 2)
        self.assertEqual(len(clean_data), 3)
        self.assertGreater(report.get_summary()['error_count'], 0)


class TestDataValidatorTypes(unittest.TestCase):
    """Test data type validation."""
    
    def test_future_timestamp(self):
        """Test detection of future timestamps."""
        future_date = datetime.now() + timedelta(days=10)
        data = pd.DataFrame({
            'timestamp': [future_date, datetime.now(), datetime.now()],
            'temperature': [38.5] * 3,
            'fxa': [0.1] * 3,
            'mya': [0.2] * 3,
            'rza': [-0.85] * 3,
            'sxg': [2.0] * 3,
            'lyg': [-1.5] * 3,
            'dzg': [0.5] * 3
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should flag future timestamp
        self.assertGreater(len(flagged_data), 0)
        errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR]
        self.assertTrue(any('future' in i.message.lower() for i in errors))
    
    def test_invalid_numeric_conversion(self):
        """Test handling of invalid numeric values."""
        # This test simulates data that would fail numeric conversion
        # Note: Since we use pd.to_numeric with errors='coerce', 
        # invalid values become NaN and trigger missing value errors
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'temperature': [38.5, 38.6, 38.7],
            'fxa': [0.1, np.nan, 0.3],  # NaN simulates conversion failure
            'mya': [0.2] * 3,
            'rza': [-0.85] * 3,
            'sxg': [2.0] * 3,
            'lyg': [-1.5] * 3,
            'dzg': [0.5] * 3
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should handle invalid values
        self.assertGreater(len(report.issues), 0)


class TestDataValidatorRanges(unittest.TestCase):
    """Test range validation."""
    
    def test_temperature_out_of_normal_range(self):
        """Test detection of temperatures outside normal range."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [38.5, 34.5, 42.5, 38.6, 38.7],  # 34.5 and 42.5 out of range
            'fxa': [0.1] * 5,
            'mya': [0.2] * 5,
            'rza': [-0.85] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [0.5] * 5
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should flag out-of-range temperatures
        self.assertEqual(len(flagged_data), 2)
        warnings = [i for i in report.issues if i.severity == ValidationSeverity.WARNING]
        self.assertTrue(any('temperature' in i.message.lower() for i in warnings))
    
    def test_acceleration_out_of_typical_range(self):
        """Test detection of accelerations outside typical range."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='1min'),
            'temperature': [38.5] * 4,
            'fxa': [0.1, 3.0, -3.0, 0.2],  # 3.0 and -3.0 out of typical range
            'mya': [0.2] * 4,
            'rza': [-0.85] * 4,
            'sxg': [2.0] * 4,
            'lyg': [-1.5] * 4,
            'dzg': [0.5] * 4
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should flag out-of-range accelerations
        self.assertEqual(len(flagged_data), 2)
        warnings = [i for i in report.issues if i.severity == ValidationSeverity.WARNING]
        self.assertTrue(any('acceleration' in i.message.lower() for i in warnings))
    
    def test_angular_velocity_out_of_range(self):
        """Test detection of angular velocities outside sensor range."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=3, freq='1min'),
            'temperature': [38.5] * 3,
            'fxa': [0.1] * 3,
            'mya': [0.2] * 3,
            'rza': [-0.85] * 3,
            'sxg': [2.0, 300.0, 2.0],  # 300.0 out of sensor range
            'lyg': [-1.5] * 3,
            'dzg': [0.5] * 3
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should flag out-of-range angular velocity
        self.assertEqual(len(flagged_data), 1)


class TestDataValidatorTimestampContinuity(unittest.TestCase):
    """Test timestamp continuity validation."""
    
    def test_no_gaps(self):
        """Test continuous timestamps with no gaps."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'temperature': [38.5] * 10,
            'fxa': [0.1] * 10,
            'mya': [0.2] * 10,
            'rza': [-0.85] * 10,
            'sxg': [2.0] * 10,
            'lyg': [-1.5] * 10,
            'dzg': [0.5] * 10
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should have no timestamp continuity issues
        continuity_issues = [i for i in report.issues if i.category == 'timestamp_continuity']
        self.assertEqual(len(continuity_issues), 0)
    
    def test_gap_detection(self):
        """Test detection of gaps > 5 minutes."""
        timestamps = list(pd.date_range('2024-01-01 00:00', periods=5, freq='1min'))
        # Add a gap of 10 minutes
        timestamps.append(timestamps[-1] + timedelta(minutes=10))
        timestamps.extend(pd.date_range(timestamps[-1] + timedelta(minutes=1), periods=4, freq='1min'))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * len(timestamps),
            'fxa': [0.1] * len(timestamps),
            'mya': [0.2] * len(timestamps),
            'rza': [-0.85] * len(timestamps),
            'sxg': [2.0] * len(timestamps),
            'lyg': [-1.5] * len(timestamps),
            'dzg': [0.5] * len(timestamps)
        })
        
        validator = DataValidator(gap_threshold_minutes=5)
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should detect the gap
        continuity_issues = [i for i in report.issues 
                           if i.category == 'timestamp_continuity' and 'gap' in i.message.lower()]
        self.assertGreater(len(continuity_issues), 0)
    
    def test_duplicate_timestamps(self):
        """Test detection of duplicate timestamps."""
        timestamps = list(pd.date_range('2024-01-01', periods=5, freq='1min'))
        timestamps.append(timestamps[2])  # Duplicate
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * len(timestamps),
            'fxa': [0.1] * len(timestamps),
            'mya': [0.2] * len(timestamps),
            'rza': [-0.85] * len(timestamps),
            'sxg': [2.0] * len(timestamps),
            'lyg': [-1.5] * len(timestamps),
            'dzg': [0.5] * len(timestamps)
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should detect duplicate
        dup_issues = [i for i in report.issues if 'duplicate' in i.message.lower()]
        self.assertGreater(len(dup_issues), 0)


class TestDataValidatorCriticalOutOfRange(unittest.TestCase):
    """Test critical out-of-range detection."""
    
    def test_hypothermia_detection(self):
        """Test detection of hypothermia risk (<35°C)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [38.5, 34.5, 34.0, 38.6, 38.7],  # 34.5 and 34.0 are critical
            'fxa': [0.1] * 5,
            'mya': [0.2] * 5,
            'rza': [-0.85] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [0.5] * 5
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should detect hypothermia risk
        critical_issues = [i for i in report.issues 
                          if i.severity == ValidationSeverity.ERROR and 
                          'hypothermia' in i.message.lower()]
        self.assertEqual(len(critical_issues), 2)
    
    def test_severe_fever_detection(self):
        """Test detection of severe fever (>42°C)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='1min'),
            'temperature': [38.5, 42.5, 43.0, 38.6],  # 42.5 and 43.0 are critical
            'fxa': [0.1] * 4,
            'mya': [0.2] * 4,
            'rza': [-0.85] * 4,
            'sxg': [2.0] * 4,
            'lyg': [-1.5] * 4,
            'dzg': [0.5] * 4
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should detect severe fever
        critical_issues = [i for i in report.issues 
                          if i.severity == ValidationSeverity.ERROR and 
                          'fever' in i.message.lower()]
        self.assertEqual(len(critical_issues), 2)
    
    def test_extreme_acceleration_detection(self):
        """Test detection of extreme accelerations (>5g)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [38.5] * 5,
            'fxa': [0.1, 6.0, -7.0, 0.2, 0.3],  # 6.0 and -7.0 are extreme
            'mya': [0.2] * 5,
            'rza': [-0.85] * 5,
            'sxg': [2.0] * 5,
            'lyg': [-1.5] * 5,
            'dzg': [0.5] * 5
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should detect extreme accelerations
        critical_issues = [i for i in report.issues 
                          if i.severity == ValidationSeverity.ERROR and 
                          'extreme' in i.message.lower()]
        self.assertEqual(len(critical_issues), 2)


class TestConvenienceFunction(unittest.TestCase):
    """Test the convenience function."""
    
    def test_validate_sensor_data_function(self):
        """Test the validate_sensor_data convenience function."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'temperature': [38.5] * 10,
            'fxa': [0.1] * 10,
            'mya': [0.2] * 10,
            'rza': [-0.85] * 10,
            'sxg': [2.0] * 10,
            'lyg': [-1.5] * 10,
            'dzg': [0.5] * 10
        })
        
        result = validate_sensor_data(data)
        
        self.assertIn('clean_data', result)
        self.assertIn('flagged_data', result)
        self.assertIn('report', result)
        self.assertIn('summary', result)
        self.assertEqual(len(result['clean_data']), 10)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        data = pd.DataFrame(columns=['timestamp', 'temperature', 'fxa', 'mya', 
                                    'rza', 'sxg', 'lyg', 'dzg'])
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        self.assertEqual(len(clean_data), 0)
        self.assertEqual(len(flagged_data), 0)
    
    def test_single_record(self):
        """Test validation with single record."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')],
            'temperature': [38.5],
            'fxa': [0.1],
            'mya': [0.2],
            'rza': [-0.85],
            'sxg': [2.0],
            'lyg': [-1.5],
            'dzg': [0.5]
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        self.assertEqual(len(clean_data), 1)
    
    def test_boundary_values(self):
        """Test with boundary values (exactly at thresholds)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=6, freq='1min'),
            'temperature': [35.0, 42.0, 38.5, 38.5, 38.5, 38.5],  # Exactly at boundaries
            'fxa': [-2.0, 2.0, 0.1, 0.1, 0.1, 0.1],  # Exactly at boundaries
            'mya': [0.2] * 6,
            'rza': [-0.85] * 6,
            'sxg': [-250.0, 250.0, 2.0, 2.0, 2.0, 2.0],  # Exactly at boundaries
            'lyg': [-1.5] * 6,
            'dzg': [0.5] * 6
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Boundary values should be flagged (inclusive ranges)
        # Temperature: <35 or >42 are critical, so 35.0 and 42.0 are NOT critical but might be flagged as warnings
        self.assertGreater(len(flagged_data), 0)
    
    def test_multiple_issues_single_record(self):
        """Test record with multiple validation issues."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')],
            'temperature': [43.0],  # Severe fever
            'fxa': [6.0],  # Extreme acceleration
            'mya': [0.2],
            'rza': [-0.85],
            'sxg': [2.0],
            'lyg': [-1.5],
            'dzg': [0.5]
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should flag multiple issues for the same record
        self.assertEqual(len(flagged_data), 1)
        self.assertGreater(len(report.issues), 1)


class TestBatchValidation(unittest.TestCase):
    """Test validation with larger batches."""
    
    def test_large_dataset(self):
        """Test validation with larger dataset."""
        # Create 1 day of data (1440 records)
        num_records = 1440
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=num_records, freq='1min'),
            'temperature': np.random.uniform(37.5, 39.5, num_records),
            'fxa': np.random.uniform(-1.0, 1.0, num_records),
            'mya': np.random.uniform(-1.0, 1.0, num_records),
            'rza': np.random.uniform(-1.0, 0.0, num_records),
            'sxg': np.random.uniform(-50.0, 50.0, num_records),
            'lyg': np.random.uniform(-50.0, 50.0, num_records),
            'dzg': np.random.uniform(-50.0, 50.0, num_records)
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should process all records
        self.assertEqual(len(clean_data) + len(flagged_data), num_records)
        self.assertEqual(report.get_summary()['total_records'], num_records)
    
    def test_mixed_quality_batch(self):
        """Test batch with mix of valid and invalid records."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=20, freq='1min'),
            'temperature': [38.5] * 15 + [43.0] * 5,  # Last 5 have severe fever
            'fxa': [0.1] * 18 + [6.0] * 2,  # Last 2 have extreme acceleration
            'mya': [0.2] * 20,
            'rza': [-0.85] * 20,
            'sxg': [2.0] * 20,
            'lyg': [-1.5] * 20,
            'dzg': [0.5] * 20
        })
        
        validator = DataValidator()
        clean_data, flagged_data, report = validator.validate(data)
        
        # Should separate clean from flagged
        self.assertGreater(len(clean_data), 0)
        self.assertGreater(len(flagged_data), 0)
        self.assertEqual(len(clean_data) + len(flagged_data), 20)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
