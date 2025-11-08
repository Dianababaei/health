"""
Data Validation Module

This module provides comprehensive validation for livestock sensor data,
including completeness checks, data type validation, range validation,
timestamp continuity checks, and out-of-range detection.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "ERROR"  # Missing or invalid data
    WARNING = "WARNING"  # Suspicious but possible
    INFO = "INFO"  # Edge cases


class ValidationIssue:
    """Represents a validation issue."""
    
    def __init__(self, severity: ValidationSeverity, category: str, 
                 message: str, row_index: Optional[int] = None,
                 column: Optional[str] = None, value: Optional[Any] = None):
        """
        Initialize a validation issue.
        
        Args:
            severity: Severity level of the issue
            category: Category of validation (e.g., 'completeness', 'range')
            message: Description of the issue
            row_index: Row index where issue occurred
            column: Column name where issue occurred
            value: Value that caused the issue
        """
        self.severity = severity
        self.category = category
        self.message = message
        self.row_index = row_index
        self.column = column
        self.value = value
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            'severity': self.severity.value,
            'category': self.category,
            'message': self.message,
            'row_index': self.row_index,
            'column': self.column,
            'value': self.value,
            'timestamp': self.timestamp.isoformat()
        }


class ValidationReport:
    """Comprehensive validation report."""
    
    def __init__(self):
        """Initialize validation report."""
        self.issues: List[ValidationIssue] = []
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.total_records = 0
        self.clean_records = 0
        self.flagged_records = 0
    
    def add_issue(self, issue: ValidationIssue):
        """Add a validation issue to the report."""
        self.issues.append(issue)
        
        # Log the issue
        logger = logging.getLogger(__name__)
        log_msg = f"[{issue.severity.value}] {issue.category}: {issue.message}"
        if issue.row_index is not None:
            log_msg += f" (row {issue.row_index})"
        if issue.column:
            log_msg += f" (column: {issue.column})"
        
        if issue.severity == ValidationSeverity.ERROR:
            logger.error(log_msg)
        elif issue.severity == ValidationSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def finalize(self, total_records: int, clean_records: int):
        """Finalize the report with summary statistics."""
        self.end_time = datetime.now()
        self.total_records = total_records
        self.clean_records = clean_records
        self.flagged_records = total_records - clean_records
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the validation."""
        error_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for i in self.issues if i.severity == ValidationSeverity.INFO)
        
        return {
            'total_records': self.total_records,
            'clean_records': self.clean_records,
            'flagged_records': self.flagged_records,
            'total_issues': len(self.issues),
            'error_count': error_count,
            'warning_count': warning_count,
            'info_count': info_count,
            'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            'issues_by_category': self._count_by_category(),
            'issues_by_severity': {
                'ERROR': error_count,
                'WARNING': warning_count,
                'INFO': info_count
            }
        }
    
    def _count_by_category(self) -> Dict[str, int]:
        """Count issues by category."""
        categories = {}
        for issue in self.issues:
            categories[issue.category] = categories.get(issue.category, 0) + 1
        return categories
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'summary': self.get_summary(),
            'issues': [issue.to_dict() for issue in self.issues]
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to DataFrame."""
        if not self.issues:
            return pd.DataFrame()
        return pd.DataFrame([issue.to_dict() for issue in self.issues])


class DataValidator:
    """
    Comprehensive data validator for livestock sensor data.
    
    Validates:
    - Data completeness (all 7 parameters present)
    - Data types and ranges
    - Timestamp continuity
    - Out-of-range detection
    """
    
    # Expected columns for sensor data
    REQUIRED_COLUMNS = ['timestamp', 'temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
    
    # Validation ranges
    TEMP_RANGE = (35.0, 42.0)  # Normal cattle range in °C
    TEMP_CRITICAL_LOW = 35.0  # Hypothermia risk
    TEMP_CRITICAL_HIGH = 42.0  # Severe fever
    
    ACCELERATION_RANGE = (-2.0, 2.0)  # Typical range for cattle in g
    ACCELERATION_EXTREME = 5.0  # Physically impossible for neck-mounted sensor
    
    ANGULAR_VELOCITY_RANGE = (-250.0, 250.0)  # Typical sensor range in deg/s
    
    DEFAULT_GAP_THRESHOLD_MINUTES = 5  # Maximum allowed gap between readings
    EXPECTED_INTERVAL_MINUTES = 1  # Expected interval between readings
    
    def __init__(self, gap_threshold_minutes: int = DEFAULT_GAP_THRESHOLD_MINUTES,
                 log_level: int = logging.INFO):
        """
        Initialize the data validator.
        
        Args:
            gap_threshold_minutes: Maximum allowed gap between consecutive readings
            log_level: Logging level for validation messages
        """
        self.gap_threshold_minutes = gap_threshold_minutes
        self.report = ValidationReport()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def validate(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, ValidationReport]:
        """
        Perform complete validation on the dataset.
        
        Args:
            data: Input DataFrame with sensor data
            
        Returns:
            Tuple of (clean_data, flagged_data, validation_report)
            - clean_data: Records that passed all validations
            - flagged_data: Records with validation issues
            - validation_report: Detailed validation report
        """
        self.logger.info(f"Starting validation of {len(data)} records")
        
        # Reset report
        self.report = ValidationReport()
        
        # Create a copy to avoid modifying original
        df = data.copy()
        
        # Add a flag column to track issues
        df['_validation_flags'] = ''
        
        # Run validation checks
        self._validate_completeness(df)
        self._validate_data_types(df)
        self._validate_ranges(df)
        self._validate_timestamp_continuity(df)
        self._detect_out_of_range(df)
        
        # Separate clean and flagged data
        clean_mask = df['_validation_flags'] == ''
        clean_data = df[clean_mask].drop(columns=['_validation_flags']).copy()
        flagged_data = df[~clean_mask].copy()
        
        # Finalize report
        self.report.finalize(total_records=len(data), clean_records=len(clean_data))
        
        self.logger.info(f"Validation complete: {len(clean_data)} clean records, "
                        f"{len(flagged_data)} flagged records")
        
        return clean_data, flagged_data, self.report
    
    def _validate_completeness(self, df: pd.DataFrame):
        """
        Check that all 7 required parameters are present for every record.
        
        Args:
            df: DataFrame to validate
        """
        self.logger.debug("Checking data completeness...")
        
        # Check for missing columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category='completeness',
                message=f"Missing required columns: {', '.join(missing_cols)}"
            )
            self.report.add_issue(issue)
        
        # Check for null values in required columns
        for col in self.REQUIRED_COLUMNS:
            if col in df.columns:
                null_mask = df[col].isnull()
                if null_mask.any():
                    null_indices = df[null_mask].index.tolist()
                    for idx in null_indices:
                        issue = ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category='completeness',
                            message=f"Missing value in required column",
                            row_index=idx,
                            column=col
                        )
                        self.report.add_issue(issue)
                        df.loc[idx, '_validation_flags'] += 'MISSING_VALUE;'
    
    def _validate_data_types(self, df: pd.DataFrame):
        """
        Validate data types and attempt appropriate type conversion/rejection.
        
        Args:
            df: DataFrame to validate
        """
        self.logger.debug("Validating data types...")
        
        # Timestamp validation
        if 'timestamp' in df.columns:
            try:
                # Try to convert to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Check for future timestamps
                now = pd.Timestamp.now()
                future_mask = df['timestamp'] > now
                if future_mask.any():
                    future_indices = df[future_mask].index.tolist()
                    for idx in future_indices:
                        issue = ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category='data_type',
                            message=f"Timestamp is in the future",
                            row_index=idx,
                            column='timestamp',
                            value=str(df.loc[idx, 'timestamp'])
                        )
                        self.report.add_issue(issue)
                        df.loc[idx, '_validation_flags'] += 'FUTURE_TIMESTAMP;'
            except Exception as e:
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='data_type',
                    message=f"Invalid timestamp format: {str(e)}"
                )
                self.report.add_issue(issue)
        
        # Numeric columns validation
        numeric_cols = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        for col in numeric_cols:
            if col in df.columns:
                try:
                    # Try to convert to float
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Check for conversion failures (NaN after conversion)
                    invalid_mask = df[col].isnull()
                    if invalid_mask.any():
                        invalid_indices = df[invalid_mask].index.tolist()
                        for idx in invalid_indices:
                            issue = ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category='data_type',
                                message=f"Invalid numeric value",
                                row_index=idx,
                                column=col
                            )
                            self.report.add_issue(issue)
                            df.loc[idx, '_validation_flags'] += f'INVALID_TYPE_{col.upper()};'
                except Exception as e:
                    issue = ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category='data_type',
                        message=f"Type conversion error for {col}: {str(e)}"
                    )
                    self.report.add_issue(issue)
    
    def _validate_ranges(self, df: pd.DataFrame):
        """
        Validate that sensor values are within expected ranges.
        
        Args:
            df: DataFrame to validate
        """
        self.logger.debug("Validating value ranges...")
        
        # Temperature range validation
        if 'temperature' in df.columns:
            # Normal range
            out_of_range = (df['temperature'] < self.TEMP_RANGE[0]) | \
                          (df['temperature'] > self.TEMP_RANGE[1])
            if out_of_range.any():
                out_indices = df[out_of_range].index.tolist()
                for idx in out_indices:
                    temp_val = df.loc[idx, 'temperature']
                    if pd.notna(temp_val):
                        issue = ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            category='range',
                            message=f"Temperature outside normal range ({self.TEMP_RANGE[0]}-{self.TEMP_RANGE[1]}°C)",
                            row_index=idx,
                            column='temperature',
                            value=temp_val
                        )
                        self.report.add_issue(issue)
                        df.loc[idx, '_validation_flags'] += 'OUT_OF_RANGE_TEMP;'
        
        # Acceleration range validation
        acc_cols = ['fxa', 'mya', 'rza']
        for col in acc_cols:
            if col in df.columns:
                out_of_range = (df[col] < self.ACCELERATION_RANGE[0]) | \
                              (df[col] > self.ACCELERATION_RANGE[1])
                if out_of_range.any():
                    out_indices = df[out_of_range].index.tolist()
                    for idx in out_indices:
                        acc_val = df.loc[idx, col]
                        if pd.notna(acc_val):
                            issue = ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category='range',
                                message=f"Acceleration outside typical range ({self.ACCELERATION_RANGE[0]}-{self.ACCELERATION_RANGE[1]}g)",
                                row_index=idx,
                                column=col,
                                value=acc_val
                            )
                            self.report.add_issue(issue)
                            df.loc[idx, '_validation_flags'] += f'OUT_OF_RANGE_{col.upper()};'
        
        # Angular velocity range validation
        gyro_cols = ['sxg', 'lyg', 'dzg']
        for col in gyro_cols:
            if col in df.columns:
                out_of_range = (df[col] < self.ANGULAR_VELOCITY_RANGE[0]) | \
                              (df[col] > self.ANGULAR_VELOCITY_RANGE[1])
                if out_of_range.any():
                    out_indices = df[out_of_range].index.tolist()
                    for idx in out_indices:
                        gyro_val = df.loc[idx, col]
                        if pd.notna(gyro_val):
                            issue = ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                category='range',
                                message=f"Angular velocity outside sensor range ({self.ANGULAR_VELOCITY_RANGE[0]}-{self.ANGULAR_VELOCITY_RANGE[1]}°/s)",
                                row_index=idx,
                                column=col,
                                value=gyro_val
                            )
                            self.report.add_issue(issue)
                            df.loc[idx, '_validation_flags'] += f'OUT_OF_RANGE_{col.upper()};'
    
    def _validate_timestamp_continuity(self, df: pd.DataFrame):
        """
        Detect gaps > threshold minutes between consecutive readings.
        
        Args:
            df: DataFrame to validate
        """
        self.logger.debug("Checking timestamp continuity...")
        
        if 'timestamp' not in df.columns or len(df) < 2:
            return
        
        # Ensure timestamps are sorted
        df_sorted = df.sort_values('timestamp')
        
        # Calculate time differences
        time_diffs = df_sorted['timestamp'].diff()
        
        # Find gaps larger than threshold
        gap_threshold = pd.Timedelta(minutes=self.gap_threshold_minutes)
        gaps = time_diffs > gap_threshold
        
        if gaps.any():
            gap_indices = df_sorted[gaps].index.tolist()
            for idx in gap_indices:
                gap_minutes = time_diffs.loc[idx].total_seconds() / 60.0
                issue = ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category='timestamp_continuity',
                    message=f"Gap of {gap_minutes:.1f} minutes detected (threshold: {self.gap_threshold_minutes} min)",
                    row_index=idx,
                    column='timestamp'
                )
                self.report.add_issue(issue)
                df.loc[idx, '_validation_flags'] += 'TIMESTAMP_GAP;'
        
        # Check for duplicate timestamps
        duplicates = df_sorted['timestamp'].duplicated()
        if duplicates.any():
            dup_indices = df_sorted[duplicates].index.tolist()
            for idx in dup_indices:
                issue = ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category='timestamp_continuity',
                    message=f"Duplicate timestamp detected",
                    row_index=idx,
                    column='timestamp',
                    value=str(df.loc[idx, 'timestamp'])
                )
                self.report.add_issue(issue)
                df.loc[idx, '_validation_flags'] += 'DUPLICATE_TIMESTAMP;'
    
    def _detect_out_of_range(self, df: pd.DataFrame):
        """
        Detect critical out-of-range values (hypothermia, severe fever, extreme accelerations).
        
        Args:
            df: DataFrame to validate
        """
        self.logger.debug("Detecting critical out-of-range values...")
        
        # Critical temperature thresholds
        if 'temperature' in df.columns:
            # Hypothermia risk (<35°C)
            hypothermia = df['temperature'] < self.TEMP_CRITICAL_LOW
            if hypothermia.any():
                hypo_indices = df[hypothermia].index.tolist()
                for idx in hypo_indices:
                    temp_val = df.loc[idx, 'temperature']
                    if pd.notna(temp_val):
                        issue = ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category='critical_out_of_range',
                            message=f"Critical: Hypothermia risk (<{self.TEMP_CRITICAL_LOW}°C)",
                            row_index=idx,
                            column='temperature',
                            value=temp_val
                        )
                        self.report.add_issue(issue)
                        df.loc[idx, '_validation_flags'] += 'CRITICAL_LOW_TEMP;'
            
            # Severe fever (>42°C)
            severe_fever = df['temperature'] > self.TEMP_CRITICAL_HIGH
            if severe_fever.any():
                fever_indices = df[severe_fever].index.tolist()
                for idx in fever_indices:
                    temp_val = df.loc[idx, 'temperature']
                    if pd.notna(temp_val):
                        issue = ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            category='critical_out_of_range',
                            message=f"Critical: Severe fever (>{self.TEMP_CRITICAL_HIGH}°C)",
                            row_index=idx,
                            column='temperature',
                            value=temp_val
                        )
                        self.report.add_issue(issue)
                        df.loc[idx, '_validation_flags'] += 'CRITICAL_HIGH_TEMP;'
        
        # Extreme accelerations (>5g - physically impossible)
        acc_cols = ['fxa', 'mya', 'rza']
        for col in acc_cols:
            if col in df.columns:
                extreme = df[col].abs() > self.ACCELERATION_EXTREME
                if extreme.any():
                    extreme_indices = df[extreme].index.tolist()
                    for idx in extreme_indices:
                        acc_val = df.loc[idx, col]
                        if pd.notna(acc_val):
                            issue = ValidationIssue(
                                severity=ValidationSeverity.ERROR,
                                category='critical_out_of_range',
                                message=f"Critical: Extreme acceleration (>{self.ACCELERATION_EXTREME}g) - physically impossible",
                                row_index=idx,
                                column=col,
                                value=acc_val
                            )
                            self.report.add_issue(issue)
                            df.loc[idx, '_validation_flags'] += f'EXTREME_ACCELERATION_{col.upper()};'


def validate_sensor_data(data: pd.DataFrame, 
                        gap_threshold_minutes: int = 5,
                        log_level: int = logging.INFO) -> Dict[str, Any]:
    """
    Convenience function to validate sensor data and return a comprehensive report.
    
    Args:
        data: Input DataFrame with sensor data
        gap_threshold_minutes: Maximum allowed gap between consecutive readings
        log_level: Logging level for validation messages
        
    Returns:
        Dictionary containing:
        - 'clean_data': DataFrame with records that passed all validations
        - 'flagged_data': DataFrame with records that have validation issues
        - 'report': ValidationReport object
        - 'summary': Summary statistics dictionary
    """
    validator = DataValidator(gap_threshold_minutes=gap_threshold_minutes, log_level=log_level)
    clean_data, flagged_data, report = validator.validate(data)
    
    return {
        'clean_data': clean_data,
        'flagged_data': flagged_data,
        'report': report,
        'summary': report.get_summary()
    }
