"""
Data validation functions for cattle sensor data ingestion.

Validates column presence, data types, timestamp chronology, and sensor value ranges.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ValidationError:
    """Container for validation error information."""
    
    def __init__(
        self,
        error_type: str,
        message: str,
        row_number: Optional[int] = None,
        column: Optional[str] = None,
        value: Optional[Any] = None,
    ):
        """
        Initialize validation error.
        
        Args:
            error_type: Type of error (e.g., 'missing_column', 'invalid_type')
            message: Error message
            row_number: Row number where error occurred (if applicable)
            column: Column name where error occurred (if applicable)
            value: Problematic value (if applicable)
        """
        self.error_type = error_type
        self.message = message
        self.row_number = row_number
        self.column = column
        self.value = value
    
    def __repr__(self):
        """String representation of error."""
        parts = [f"[{self.error_type}]", self.message]
        if self.row_number is not None:
            parts.append(f"(row {self.row_number})")
        if self.column is not None:
            parts.append(f"(column: {self.column})")
        return " ".join(parts)


class ValidationSummary:
    """Summary of validation results."""
    
    def __init__(self):
        """Initialize validation summary."""
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        self.rows_processed = 0
        self.rows_valid = 0
        self.rows_skipped = 0
    
    def add_error(self, error: ValidationError):
        """Add an error to the summary."""
        self.errors.append(error)
        logger.error(str(error))
    
    def add_warning(self, warning: ValidationError):
        """Add a warning to the summary."""
        self.warnings.append(warning)
        logger.warning(str(warning))
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0
    
    def get_error_counts(self) -> Dict[str, int]:
        """Get counts of each error type."""
        counts = {}
        for error in self.errors:
            counts[error.error_type] = counts.get(error.error_type, 0) + 1
        return counts
    
    def get_warning_counts(self) -> Dict[str, int]:
        """Get counts of each warning type."""
        counts = {}
        for warning in self.warnings:
            counts[warning.error_type] = counts.get(warning.error_type, 0) + 1
        return counts
    
    def __repr__(self):
        """String representation of summary."""
        lines = [
            "Validation Summary:",
            f"  Rows processed: {self.rows_processed}",
            f"  Rows valid: {self.rows_valid}",
            f"  Rows skipped: {self.rows_skipped}",
            f"  Errors: {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        
        if self.errors:
            lines.append("\nError counts by type:")
            for error_type, count in self.get_error_counts().items():
                lines.append(f"  {error_type}: {count}")
        
        if self.warnings:
            lines.append("\nWarning counts by type:")
            for warning_type, count in self.get_warning_counts().items():
                lines.append(f"  {warning_type}: {count}")
        
        return "\n".join(lines)


class DataValidator:
    """
    Validator for cattle sensor data.
    
    Performs column validation, data type checks, timestamp chronology validation,
    and sensor value range checks.
    """
    
    # Required sensor columns
    REQUIRED_COLUMNS = [
        'timestamp',
        'temperature',
        'fxa',
        'mya',
        'rza',
        'sxg',
        'lyg',
        'dzg',
    ]
    
    # Optional columns
    OPTIONAL_COLUMNS = [
        'cow_id',
        'sensor_id',
        'state',
        'sample_id',
        'animal_id',
    ]
    
    # Sensor value ranges (min, max)
    SENSOR_RANGES = {
        'temperature': (35.0, 42.0),    # °C - reasonable range for cattle
        'fxa': (-5.0, 5.0),              # m/s² or g-units
        'mya': (-5.0, 5.0),              # m/s² or g-units
        'rza': (-2.0, 2.0),              # m/s² or g-units (mostly affected by gravity)
        'sxg': (-200.0, 200.0),          # °/s or rad/s
        'lyg': (-200.0, 200.0),          # °/s or rad/s
        'dzg': (-200.0, 200.0),          # °/s or rad/s
    }
    
    # Extreme value ranges (trigger warnings, not errors)
    EXTREME_RANGES = {
        'temperature': (37.0, 40.0),    # Outside normal range but not impossible
        'fxa': (-3.0, 3.0),
        'mya': (-3.0, 3.0),
        'rza': (-1.5, 1.5),
        'sxg': (-100.0, 100.0),
        'lyg': (-100.0, 100.0),
        'dzg': (-100.0, 100.0),
    }
    
    def __init__(self):
        """Initialize data validator."""
        self.summary = ValidationSummary()
    
    def validate_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that all required columns are present.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if all required columns present, False otherwise
        """
        missing_columns = []
        
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                missing_columns.append(col)
                self.summary.add_error(ValidationError(
                    'missing_column',
                    f"Required column '{col}' not found in CSV",
                    column=col
                ))
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("All required columns present")
        return True
    
    def validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and convert data types for sensor columns.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataFrame with converted data types (with invalid rows marked)
        """
        df_clean = df.copy()
        
        # Validate numeric columns
        numeric_columns = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        
        for col in numeric_columns:
            if col not in df_clean.columns:
                continue
            
            # Try to convert to numeric
            original_values = df_clean[col].copy()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Find rows that failed conversion
            invalid_mask = df_clean[col].isna() & original_values.notna()
            invalid_rows = df_clean.index[invalid_mask].tolist()
            
            for row_idx in invalid_rows:
                self.summary.add_error(ValidationError(
                    'invalid_type',
                    f"Non-numeric value in column '{col}'",
                    row_number=row_idx,
                    column=col,
                    value=original_values.iloc[row_idx]
                ))
        
        return df_clean
    
    def validate_sensor_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate sensor values are within acceptable ranges.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            DataFrame with out-of-range values marked as invalid
        """
        df_clean = df.copy()
        
        for sensor, (min_val, max_val) in self.SENSOR_RANGES.items():
            if sensor not in df_clean.columns:
                continue
            
            # Check for out-of-range values
            out_of_range = (df_clean[sensor] < min_val) | (df_clean[sensor] > max_val)
            out_of_range_rows = df_clean.index[out_of_range].tolist()
            
            for row_idx in out_of_range_rows:
                value = df_clean.loc[row_idx, sensor]
                if pd.notna(value):
                    self.summary.add_error(ValidationError(
                        'value_out_of_range',
                        f"Value {value:.2f} for '{sensor}' outside valid range [{min_val}, {max_val}]",
                        row_number=row_idx,
                        column=sensor,
                        value=value
                    ))
        
        # Check for extreme but not impossible values (warnings)
        for sensor, (min_val, max_val) in self.EXTREME_RANGES.items():
            if sensor not in df_clean.columns:
                continue
            
            extreme = (df_clean[sensor] < min_val) | (df_clean[sensor] > max_val)
            # Only warn if not already flagged as out of range
            sensor_min, sensor_max = self.SENSOR_RANGES[sensor]
            in_range = (df_clean[sensor] >= sensor_min) & (df_clean[sensor] <= sensor_max)
            extreme_rows = df_clean.index[extreme & in_range].tolist()
            
            for row_idx in extreme_rows:
                value = df_clean.loc[row_idx, sensor]
                if pd.notna(value):
                    self.summary.add_warning(ValidationError(
                        'extreme_value',
                        f"Extreme value {value:.2f} for '{sensor}' (unusual but within limits)",
                        row_number=row_idx,
                        column=sensor,
                        value=value
                    ))
        
        return df_clean
    
    def validate_timestamp_chronology(self, df: pd.DataFrame) -> bool:
        """
        Validate that timestamps are in chronological order.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            True if timestamps are chronological, False otherwise
        """
        if 'timestamp' not in df.columns:
            return True  # Already handled by column validation
        
        if len(df) < 2:
            return True  # Nothing to check
        
        # Check for non-chronological timestamps
        for i in range(1, len(df)):
            if df['timestamp'].iloc[i] < df['timestamp'].iloc[i-1]:
                self.summary.add_error(ValidationError(
                    'non_chronological',
                    f"Timestamp at row {i} is earlier than previous timestamp",
                    row_number=i,
                    column='timestamp',
                    value=df['timestamp'].iloc[i]
                ))
        
        # Check for duplicate timestamps
        duplicates = df['timestamp'].duplicated()
        duplicate_rows = df.index[duplicates].tolist()
        
        for row_idx in duplicate_rows:
            self.summary.add_warning(ValidationError(
                'duplicate_timestamp',
                f"Duplicate timestamp at row {row_idx}",
                row_number=row_idx,
                column='timestamp',
                value=df['timestamp'].iloc[row_idx]
            ))
        
        # Check time intervals (should be approximately 1 minute for 1Hz sampling)
        if len(df) > 1:
            time_diffs = df['timestamp'].diff().dt.total_seconds()
            
            # Expected interval is 60 seconds (1 minute)
            expected_interval = 60.0
            tolerance = 5.0  # Allow 5 seconds tolerance
            
            irregular = (time_diffs < (expected_interval - tolerance)) | \
                       (time_diffs > (expected_interval + tolerance))
            irregular_rows = df.index[irregular].tolist()[1:]  # Skip first (NaN)
            
            # Only warn about a sample of irregular intervals (not all)
            if irregular_rows:
                sample_size = min(10, len(irregular_rows))
                for row_idx in irregular_rows[:sample_size]:
                    interval = time_diffs.iloc[row_idx]
                    if pd.notna(interval):
                        self.summary.add_warning(ValidationError(
                            'irregular_interval',
                            f"Time interval {interval:.1f}s is irregular (expected ~{expected_interval}s)",
                            row_number=row_idx,
                            column='timestamp',
                            value=interval
                        ))
                
                if len(irregular_rows) > sample_size:
                    self.summary.add_warning(ValidationError(
                        'irregular_interval',
                        f"Total {len(irregular_rows)} irregular time intervals found (showing first {sample_size})",
                    ))
        
        return len([e for e in self.summary.errors if e.error_type == 'non_chronological']) == 0
    
    def validate_timestamp_ranges(self, df: pd.DataFrame) -> bool:
        """
        Validate that timestamps are within reasonable ranges.
        
        Args:
            df: DataFrame with 'timestamp' column
            
        Returns:
            True if timestamps are reasonable, False otherwise
        """
        if 'timestamp' not in df.columns or len(df) == 0:
            return True
        
        # Check for timestamps in the distant past or future
        now = datetime.now()
        min_reasonable = datetime(2020, 1, 1)  # No data before 2020
        max_reasonable = now + timedelta(days=1)  # Allow 1 day in future
        
        too_old = df['timestamp'] < min_reasonable
        too_new = df['timestamp'] > max_reasonable
        
        old_rows = df.index[too_old].tolist()
        new_rows = df.index[too_new].tolist()
        
        for row_idx in old_rows:
            self.summary.add_warning(ValidationError(
                'old_timestamp',
                f"Timestamp {df['timestamp'].iloc[row_idx]} is before {min_reasonable}",
                row_number=row_idx,
                column='timestamp',
                value=df['timestamp'].iloc[row_idx]
            ))
        
        for row_idx in new_rows:
            self.summary.add_warning(ValidationError(
                'future_timestamp',
                f"Timestamp {df['timestamp'].iloc[row_idx]} is in the future",
                row_number=row_idx,
                column='timestamp',
                value=df['timestamp'].iloc[row_idx]
            ))
        
        return len(old_rows) == 0 and len(new_rows) == 0
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationSummary]:
        """
        Perform complete validation on DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (cleaned DataFrame, validation summary)
        """
        self.summary = ValidationSummary()
        self.summary.rows_processed = len(df)
        
        logger.info(f"Validating DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # 1. Validate columns
        if not self.validate_columns(df):
            self.summary.rows_skipped = len(df)
            return df, self.summary
        
        # 2. Validate and convert data types
        df_clean = self.validate_data_types(df)
        
        # 3. Validate sensor ranges
        df_clean = self.validate_sensor_ranges(df_clean)
        
        # 4. Validate timestamp chronology (if timestamp column exists and is parsed)
        if 'timestamp' in df_clean.columns and pd.api.types.is_datetime64_any_dtype(df_clean['timestamp']):
            self.validate_timestamp_chronology(df_clean)
            self.validate_timestamp_ranges(df_clean)
        
        # Count valid rows (rows with no NaN in required numeric columns)
        numeric_cols = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        numeric_cols = [c for c in numeric_cols if c in df_clean.columns]
        
        valid_mask = df_clean[numeric_cols].notna().all(axis=1)
        self.summary.rows_valid = valid_mask.sum()
        self.summary.rows_skipped = len(df_clean) - self.summary.rows_valid
        
        logger.info(f"Validation complete: {self.summary.rows_valid} valid rows, "
                   f"{self.summary.rows_skipped} rows with issues")
        
        return df_clean, self.summary
