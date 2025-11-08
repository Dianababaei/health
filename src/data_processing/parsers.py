"""
Timestamp and CSV parsing utilities for cattle sensor data ingestion.

Handles various timestamp formats (ISO 8601, Unix, custom) and CSV
parsing with auto-detection of delimiters and formats.
"""

import csv
import pandas as pd
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from dateutil import parser as date_parser
import logging

logger = logging.getLogger(__name__)


class TimestampParser:
    """
    Parse timestamps from various formats.
    
    Supports ISO 8601, Unix timestamps (seconds/milliseconds), and custom formats.
    """
    
    # Common timestamp formats to try
    COMMON_FORMATS = [
        "%Y-%m-%dT%H:%M:%S",           # ISO 8601 basic
        "%Y-%m-%dT%H:%M:%S.%f",        # ISO 8601 with microseconds
        "%Y-%m-%d %H:%M:%S",           # Space-separated
        "%Y/%m/%d %H:%M:%S",           # Slash-separated
        "%d-%m-%Y %H:%M:%S",           # European format
        "%m/%d/%Y %H:%M:%S",           # US format
        "%Y-%m-%d",                    # Date only
    ]
    
    def __init__(self):
        """Initialize timestamp parser."""
        self.detected_format = None
        self.is_unix_timestamp = False
        self.unix_scale = 1  # 1 for seconds, 1000 for milliseconds
    
    def detect_format(self, sample_timestamps: List[str]) -> str:
        """
        Detect timestamp format from sample values.
        
        Args:
            sample_timestamps: List of sample timestamp strings
            
        Returns:
            Detected format string or 'unix' for Unix timestamps
        """
        if not sample_timestamps:
            raise ValueError("No sample timestamps provided for format detection")
        
        # Try to parse first non-empty sample
        sample = None
        for ts in sample_timestamps:
            if ts and str(ts).strip():
                sample = str(ts).strip()
                break
        
        if not sample:
            raise ValueError("All sample timestamps are empty")
        
        # Check if it's a Unix timestamp (numeric)
        try:
            numeric_value = float(sample)
            # Unix timestamps are typically 10 digits (seconds) or 13 digits (milliseconds)
            if numeric_value > 1e9 and numeric_value < 1e10:
                self.is_unix_timestamp = True
                self.unix_scale = 1
                self.detected_format = 'unix_seconds'
                logger.info("Detected Unix timestamp format (seconds)")
                return 'unix_seconds'
            elif numeric_value > 1e12 and numeric_value < 1e13:
                self.is_unix_timestamp = True
                self.unix_scale = 1000
                self.detected_format = 'unix_milliseconds'
                logger.info("Detected Unix timestamp format (milliseconds)")
                return 'unix_milliseconds'
        except (ValueError, TypeError):
            pass
        
        # Try common formats
        for fmt in self.COMMON_FORMATS:
            try:
                datetime.strptime(sample, fmt)
                self.detected_format = fmt
                logger.info(f"Detected timestamp format: {fmt}")
                return fmt
            except ValueError:
                continue
        
        # Try dateutil parser as fallback (very flexible)
        try:
            date_parser.parse(sample)
            self.detected_format = 'dateutil'
            logger.info("Using flexible dateutil parser for timestamps")
            return 'dateutil'
        except Exception as e:
            raise ValueError(f"Could not detect timestamp format from sample: {sample}") from e
    
    def parse(self, timestamp_str: str) -> datetime:
        """
        Parse a timestamp string to datetime object.
        
        Args:
            timestamp_str: Timestamp string to parse
            
        Returns:
            Parsed datetime object
        """
        if not timestamp_str or (isinstance(timestamp_str, str) and not timestamp_str.strip()):
            raise ValueError("Empty timestamp string")
        
        timestamp_str = str(timestamp_str).strip()
        
        # Handle Unix timestamps
        if self.is_unix_timestamp:
            try:
                numeric_value = float(timestamp_str)
                return datetime.fromtimestamp(numeric_value / self.unix_scale)
            except Exception as e:
                raise ValueError(f"Invalid Unix timestamp: {timestamp_str}") from e
        
        # Handle formatted timestamps
        if self.detected_format and self.detected_format != 'dateutil':
            try:
                return datetime.strptime(timestamp_str, self.detected_format)
            except Exception as e:
                raise ValueError(
                    f"Timestamp '{timestamp_str}' does not match format '{self.detected_format}'"
                ) from e
        
        # Use dateutil as fallback
        try:
            return date_parser.parse(timestamp_str)
        except Exception as e:
            raise ValueError(f"Could not parse timestamp: {timestamp_str}") from e
    
    def parse_series(self, timestamp_series: pd.Series) -> pd.Series:
        """
        Parse a pandas Series of timestamps.
        
        Args:
            timestamp_series: Series of timestamp strings
            
        Returns:
            Series of datetime objects
        """
        if self.is_unix_timestamp:
            # For Unix timestamps, use pandas' efficient conversion
            try:
                numeric_series = pd.to_numeric(timestamp_series)
                if self.unix_scale == 1000:
                    return pd.to_datetime(numeric_series, unit='ms')
                else:
                    return pd.to_datetime(numeric_series, unit='s')
            except Exception as e:
                raise ValueError("Failed to parse Unix timestamps") from e
        
        # For string timestamps
        if self.detected_format and self.detected_format != 'dateutil':
            try:
                return pd.to_datetime(timestamp_series, format=self.detected_format)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse timestamps with format '{self.detected_format}'"
                ) from e
        
        # Use dateutil parser (slower but flexible)
        try:
            return pd.to_datetime(timestamp_series, infer_datetime_format=True)
        except Exception as e:
            raise ValueError("Failed to parse timestamps") from e


class CSVParser:
    """
    Parse CSV files with auto-detection of delimiter and encoding.
    """
    
    COMMON_DELIMITERS = [',', ';', '\t', '|']
    COMMON_ENCODINGS = ['utf-8', 'ascii', 'latin-1', 'cp1252']
    
    def __init__(self):
        """Initialize CSV parser."""
        self.detected_delimiter = None
        self.detected_encoding = None
        self.has_header = True
    
    def detect_delimiter(self, file_path: str, sample_lines: int = 5) -> str:
        """
        Detect CSV delimiter from file.
        
        Args:
            file_path: Path to CSV file
            sample_lines: Number of lines to sample
            
        Returns:
            Detected delimiter character
        """
        # Try to read with different encodings
        for encoding in self.COMMON_ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    sample = ''.join([f.readline() for _ in range(sample_lines)])
                
                # Use csv.Sniffer to detect delimiter
                try:
                    sniffer = csv.Sniffer()
                    dialect = sniffer.sniff(sample, delimiters=''.join(self.COMMON_DELIMITERS))
                    self.detected_delimiter = dialect.delimiter
                    self.detected_encoding = encoding
                    logger.info(f"Detected delimiter: '{self.detected_delimiter}', encoding: {encoding}")
                    return self.detected_delimiter
                except csv.Error:
                    # If sniffer fails, try counting delimiters
                    delimiter_counts = {delim: sample.count(delim) for delim in self.COMMON_DELIMITERS}
                    if max(delimiter_counts.values()) > 0:
                        self.detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
                        self.detected_encoding = encoding
                        logger.info(f"Detected delimiter by counting: '{self.detected_delimiter}', encoding: {encoding}")
                        return self.detected_delimiter
            
            except UnicodeDecodeError:
                continue
        
        # Default to comma if detection fails
        logger.warning("Could not detect delimiter, defaulting to comma")
        self.detected_delimiter = ','
        self.detected_encoding = 'utf-8'
        return ','
    
    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Detected encoding
        """
        for encoding in self.COMMON_ENCODINGS:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try reading first 1KB
                self.detected_encoding = encoding
                logger.info(f"Detected encoding: {encoding}")
                return encoding
            except UnicodeDecodeError:
                continue
        
        # Default to utf-8
        logger.warning("Could not detect encoding, defaulting to UTF-8")
        self.detected_encoding = 'utf-8'
        return 'utf-8'
    
    def read_csv(
        self,
        file_path: str,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
        nrows: Optional[int] = None,
        skiprows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Read CSV file with auto-detection.
        
        Args:
            file_path: Path to CSV file
            delimiter: Delimiter (auto-detected if None)
            encoding: Encoding (auto-detected if None)
            nrows: Number of rows to read (None for all)
            skiprows: Number of rows to skip at start
            
        Returns:
            DataFrame with CSV data
        """
        # Auto-detect delimiter and encoding if not provided
        if delimiter is None:
            delimiter = self.detect_delimiter(file_path)
        
        if encoding is None:
            if self.detected_encoding is None:
                encoding = self.detect_encoding(file_path)
            else:
                encoding = self.detected_encoding
        
        # Read CSV
        try:
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                nrows=nrows,
                skiprows=skiprows,
            )
            logger.info(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
            return df
        
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {file_path}") from e
    
    def read_csv_incremental(
        self,
        file_path: str,
        last_position: int = 0,
        delimiter: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Read new data from CSV file starting from last position.
        
        Args:
            file_path: Path to CSV file
            last_position: Last file position (byte offset)
            delimiter: Delimiter (auto-detected if None)
            encoding: Encoding (auto-detected if None)
            
        Returns:
            Tuple of (new data DataFrame, new file position)
        """
        # Auto-detect delimiter and encoding if not provided
        if delimiter is None and self.detected_delimiter is None:
            delimiter = self.detect_delimiter(file_path)
        elif delimiter is None:
            delimiter = self.detected_delimiter
        
        if encoding is None and self.detected_encoding is None:
            encoding = self.detect_encoding(file_path)
        elif encoding is None:
            encoding = self.detected_encoding
        
        # Read from last position
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                # If this is first read, save header
                if last_position == 0:
                    header_line = f.readline()
                    column_names = header_line.strip().split(delimiter)
                    last_position = f.tell()
                else:
                    # Read header to get column names
                    f.seek(0)
                    header_line = f.readline()
                    column_names = header_line.strip().split(delimiter)
                    # Move to last position
                    f.seek(last_position)
                
                # Read new lines
                new_lines = []
                for line in f:
                    if line.strip():  # Skip empty lines
                        new_lines.append(line.strip().split(delimiter))
                
                new_position = f.tell()
            
            # Create DataFrame from new lines
            if new_lines:
                df = pd.DataFrame(new_lines, columns=column_names)
                logger.info(f"Read {len(df)} new rows from position {last_position} to {new_position}")
                return df, new_position
            else:
                logger.info("No new data found")
                return pd.DataFrame(columns=column_names), last_position
        
        except Exception as e:
            raise ValueError(f"Failed to read CSV incrementally from {file_path}") from e
