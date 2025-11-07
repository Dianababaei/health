#!/usr/bin/env python3
"""
Data Loader Module for Artemis Health Monitoring System

This module provides functions to load sensor data from CSV files containing
neck-mounted sensor measurements. The data includes timestamp and 7 sensor
channels: temperature, Fxa, Mya, Rza, Sxg, Lyg, and Dzg.

Functions:
    read_sensor_csv: Load a single CSV file with sensor data
    read_sensor_directory: Load and concatenate multiple CSV files from a directory

Usage Example:
    >>> from src.data_loader import read_sensor_csv, read_sensor_directory
    >>> 
    >>> # Load a single CSV file
    >>> df = read_sensor_csv('data/sensor_data.csv')
    >>> 
    >>> # Load all CSV files from a directory
    >>> df = read_sensor_directory('data/sensor_readings/')
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List
from datetime import datetime
import warnings

from src.utils.logger import get_logger

# Initialize logger for data loading
logger = get_logger('artemis.data.loader')

# Expected column names for sensor CSV files
REQUIRED_COLUMNS = ['timestamp', 'temperature', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg']
SENSOR_COLUMNS = ['temperature', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg']

# Timestamp formats to try in order
TIMESTAMP_FORMATS = [
    '%Y-%m-%d %H:%M:%S',      # YYYY-MM-DD HH:MM:SS
    '%Y-%m-%dT%H:%M:%S',      # ISO 8601 without timezone
    '%Y-%m-%dT%H:%M:%S.%f',   # ISO 8601 with microseconds
    '%Y-%m-%dT%H:%M:%SZ',     # ISO 8601 with Z timezone
    '%Y-%m-%d %H:%M:%S.%f',   # With microseconds
    '%d/%m/%Y %H:%M:%S',      # DD/MM/YYYY HH:MM:SS
    '%m/%d/%Y %H:%M:%S',      # MM/DD/YYYY HH:MM:SS
    '%Y/%m/%d %H:%M:%S',      # YYYY/MM/DD HH:MM:SS
]


def _parse_timestamp_column(timestamp_series: pd.Series) -> pd.DatetimeIndex:
    """
    Parse timestamp column with multiple format attempts.
    
    This function tries to parse timestamps in the following order:
    1. Unix epoch (numeric values)
    2. ISO 8601 and common date formats
    3. Pandas automatic inference as last resort
    
    Args:
        timestamp_series: Series containing timestamp strings or numbers
        
    Returns:
        DatetimeIndex with parsed timestamps
        
    Raises:
        ValueError: If timestamps cannot be parsed
    """
    # First, check if timestamps are numeric (Unix epoch)
    try:
        # Try to convert to numeric, if successful it might be Unix epoch
        numeric_timestamps = pd.to_numeric(timestamp_series, errors='coerce')
        if numeric_timestamps.notna().sum() == len(timestamp_series):
            # All values are numeric, assume Unix epoch
            logger.debug("Parsing timestamps as Unix epoch")
            # Check if values are in seconds or milliseconds
            if numeric_timestamps.min() > 1e10:  # Likely milliseconds
                return pd.to_datetime(numeric_timestamps, unit='ms')
            else:  # Likely seconds
                return pd.to_datetime(numeric_timestamps, unit='s')
    except Exception:
        pass
    
    # Try specific datetime formats
    for fmt in TIMESTAMP_FORMATS:
        try:
            parsed = pd.to_datetime(timestamp_series, format=fmt, errors='raise')
            logger.debug(f"Successfully parsed timestamps using format: {fmt}")
            return parsed
        except (ValueError, TypeError):
            continue
    
    # Last resort: let pandas infer the format
    try:
        logger.debug("Attempting pandas automatic timestamp inference")
        parsed = pd.to_datetime(timestamp_series, errors='raise', infer_datetime_format=True)
        logger.debug("Successfully parsed timestamps using automatic inference")
        return parsed
    except Exception as e:
        # Log some sample values to help debugging
        sample_values = timestamp_series.head(3).tolist()
        logger.error(f"Failed to parse timestamps. Sample values: {sample_values}")
        raise ValueError(
            f"Unable to parse timestamp column. Tried multiple formats but failed. "
            f"Sample values: {sample_values}. "
            f"Expected formats: Unix epoch, ISO 8601, or common date strings like 'YYYY-MM-DD HH:MM:SS'"
        ) from e


def _validate_columns(df: pd.DataFrame, filepath: str) -> None:
    """
    Validate that DataFrame has all required columns.
    
    Args:
        df: DataFrame to validate
        filepath: Path to the file being validated (for error messages)
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    
    if missing_columns:
        logger.error(f"Missing required columns in {filepath}: {missing_columns}")
        raise ValueError(
            f"CSV file '{filepath}' is missing required columns: {sorted(missing_columns)}. "
            f"Expected columns: {REQUIRED_COLUMNS}"
        )
    
    logger.debug(f"Column validation passed for {filepath}")


def _convert_sensor_columns_to_numeric(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Convert sensor columns to numeric values, coercing errors to NaN.
    
    Args:
        df: DataFrame with sensor columns
        filepath: Path to the file being processed (for logging)
        
    Returns:
        DataFrame with numeric sensor columns
    """
    df_copy = df.copy()
    rows_with_errors = set()
    
    for col in SENSOR_COLUMNS:
        # Convert to numeric, coercing errors to NaN
        original_values = df_copy[col].copy()
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Find rows where conversion failed
        failed_mask = original_values.notna() & df_copy[col].isna()
        if failed_mask.any():
            failed_indices = df_copy.index[failed_mask].tolist()
            rows_with_errors.update(failed_indices)
            failed_values = original_values[failed_mask].tolist()
            logger.warning(
                f"Non-numeric values in column '{col}' at rows {failed_indices[:5]} "
                f"(showing first 5). Sample values: {failed_values[:3]}. "
                f"These values have been set to NaN."
            )
    
    if rows_with_errors:
        logger.warning(
            f"Total of {len(rows_with_errors)} rows with non-numeric values in {filepath}. "
            f"These values have been coerced to NaN."
        )
    
    return df_copy


def read_sensor_csv(filepath: Union[str, Path], chunksize: int = None) -> pd.DataFrame:
    """
    Load a single CSV file containing sensor data.
    
    This function reads a CSV file with sensor measurements from neck-mounted
    devices. The file must contain 8 columns: timestamp and 7 sensor measurements
    (temperature, Fxa, Mya, Rza, Sxg, Lyg, Dzg).
    
    The function performs the following operations:
    1. Validates that all required columns are present
    2. Parses timestamps in multiple formats (Unix epoch, ISO 8601, common formats)
    3. Converts sensor columns to numeric values (invalid values become NaN)
    4. Sets the timestamp as the DatetimeIndex
    5. Sorts the data chronologically
    
    Args:
        filepath: Path to the CSV file (string or Path object)
        chunksize: Optional parameter for reading large files in chunks.
                   If specified, returns an iterator instead of a DataFrame.
                   For future scalability. Currently not implemented.
    
    Returns:
        DataFrame with DatetimeIndex and 7 numeric sensor columns,
        sorted chronologically
    
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If required columns are missing or timestamps cannot be parsed
        pd.errors.EmptyDataError: If the file is empty
    
    Example:
        >>> # Load a CSV file with sensor data
        >>> df = read_sensor_csv('data/animal_A12345_20240115.csv')
        >>> print(df.shape)
        (1440, 7)  # 24 hours of minute-by-minute data
        >>> 
        >>> print(df.columns.tolist())
        ['temperature', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg']
        >>> 
        >>> print(df.index)
        DatetimeIndex(['2024-01-15 00:00:00', '2024-01-15 00:01:00', ...], name='timestamp')
    """
    filepath = Path(filepath)
    
    # Check if file exists
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(
            f"The file '{filepath}' does not exist. "
            f"Please check the file path and try again."
        )
    
    logger.info(f"Loading sensor data from: {filepath}")
    
    # Handle chunksize parameter (for future scalability)
    if chunksize is not None:
        logger.warning(
            "The 'chunksize' parameter is not yet implemented. "
            "Loading entire file into memory."
        )
    
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Check if file is empty
        if df.empty:
            logger.error(f"CSV file is empty: {filepath}")
            raise ValueError(f"The file '{filepath}' is empty. No data to process.")
        
        logger.debug(f"Loaded {len(df)} rows from {filepath}")
        
        # Validate required columns
        _validate_columns(df, str(filepath))
        
        # Parse timestamp column
        try:
            df['timestamp'] = _parse_timestamp_column(df['timestamp'])
        except ValueError as e:
            logger.error(f"Timestamp parsing failed for {filepath}: {e}")
            raise
        
        # Convert sensor columns to numeric
        df = _convert_sensor_columns_to_numeric(df, str(filepath))
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        df.index.name = 'timestamp'
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        logger.debug(f"Data sorted chronologically from {df.index[0]} to {df.index[-1]}")
        
        logger.info(
            f"Successfully loaded {len(df)} records from {filepath} "
            f"(time range: {df.index[0]} to {df.index[-1]})"
        )
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty or has no data: {filepath}")
        raise ValueError(f"The file '{filepath}' is empty or contains no valid data.")
    except Exception as e:
        logger.error(f"Error loading CSV file {filepath}: {e}", exc_info=True)
        raise


def read_sensor_directory(dirpath: Union[str, Path], pattern: str = '*.csv') -> pd.DataFrame:
    """
    Load and concatenate all CSV files from a directory.
    
    This function finds all CSV files in the specified directory, reads each
    using read_sensor_csv(), and concatenates them into a single DataFrame.
    The result is sorted chronologically and duplicate timestamps are removed
    (keeping the first occurrence).
    
    Args:
        dirpath: Path to the directory containing CSV files
        pattern: Glob pattern for finding CSV files (default: '*.csv')
    
    Returns:
        DataFrame with DatetimeIndex and 7 numeric sensor columns,
        sorted chronologically with duplicates removed
    
    Raises:
        FileNotFoundError: If the directory does not exist
        ValueError: If no CSV files are found or directory is empty
    
    Example:
        >>> # Load all CSV files from a directory
        >>> df = read_sensor_directory('data/january_2024/')
        >>> print(df.shape)
        (44640, 7)  # 31 days of minute-by-minute data
        >>> 
        >>> # Load with custom pattern
        >>> df = read_sensor_directory('data/', pattern='animal_A*.csv')
    """
    dirpath = Path(dirpath)
    
    # Check if directory exists
    if not dirpath.exists():
        logger.error(f"Directory not found: {dirpath}")
        raise FileNotFoundError(
            f"The directory '{dirpath}' does not exist. "
            f"Please check the directory path and try again."
        )
    
    if not dirpath.is_dir():
        logger.error(f"Path is not a directory: {dirpath}")
        raise ValueError(f"The path '{dirpath}' is not a directory.")
    
    logger.info(f"Loading CSV files from directory: {dirpath} (pattern: {pattern})")
    
    # Find all CSV files in directory
    csv_files = sorted(dirpath.glob(pattern))
    
    if not csv_files:
        logger.error(f"No CSV files found in {dirpath} matching pattern '{pattern}'")
        raise ValueError(
            f"No CSV files found in directory '{dirpath}' matching pattern '{pattern}'. "
            f"Please check the directory contains CSV files."
        )
    
    logger.info(f"Found {len(csv_files)} CSV files to load")
    
    # Read and concatenate all files
    dataframes = []
    successful_files = 0
    failed_files = []
    
    for csv_file in csv_files:
        try:
            df = read_sensor_csv(csv_file)
            dataframes.append(df)
            successful_files += 1
        except Exception as e:
            failed_files.append(csv_file.name)
            logger.error(f"Failed to load {csv_file.name}: {e}")
            # Continue with other files instead of failing completely
    
    if not dataframes:
        logger.error(f"Failed to load any CSV files from {dirpath}")
        raise ValueError(
            f"Could not load any valid CSV files from '{dirpath}'. "
            f"All {len(csv_files)} files failed to load."
        )
    
    if failed_files:
        logger.warning(
            f"Successfully loaded {successful_files}/{len(csv_files)} files. "
            f"Failed files: {failed_files}"
        )
    
    # Concatenate all dataframes
    logger.debug(f"Concatenating {len(dataframes)} dataframes")
    combined_df = pd.concat(dataframes, axis=0)
    
    # Sort by timestamp
    combined_df.sort_index(inplace=True)
    
    # Remove duplicate timestamps (keep first occurrence)
    duplicate_count = combined_df.index.duplicated().sum()
    if duplicate_count > 0:
        logger.warning(
            f"Found {duplicate_count} duplicate timestamps. "
            f"Keeping first occurrence and removing duplicates."
        )
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    logger.info(
        f"Successfully loaded and combined {len(combined_df)} records "
        f"from {successful_files} files "
        f"(time range: {combined_df.index[0]} to {combined_df.index[-1]})"
    )
    
    return combined_df


if __name__ == '__main__':
    # Example usage demonstration
    print("Artemis Health - Data Loader Module")
    print("=" * 50)
    print("\nThis module provides functions to load sensor CSV data.")
    print("\nUsage:")
    print("  from src.data_loader import read_sensor_csv, read_sensor_directory")
    print("\n  # Load a single file")
    print("  df = read_sensor_csv('data/sensor_data.csv')")
    print("\n  # Load directory of files")
    print("  df = read_sensor_directory('data/sensor_readings/')")
    print("\nExpected CSV format:")
    print("  Columns:", REQUIRED_COLUMNS)
