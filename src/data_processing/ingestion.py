"""
Data ingestion module for cattle sensor data.

Provides batch and incremental loading modes with comprehensive error handling,
validation, and logging.
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List
import pandas as pd

from .parsers import TimestampParser, CSVParser
from .validators import DataValidator, ValidationSummary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IngestionSummary:
    """Summary of data ingestion operation."""
    
    def __init__(self):
        """Initialize ingestion summary."""
        self.file_path = None
        self.start_time = None
        self.end_time = None
        self.total_rows_read = 0
        self.valid_rows = 0
        self.skipped_rows = 0
        self.validation_summary: Optional[ValidationSummary] = None
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def __repr__(self):
        """String representation of summary."""
        duration = 0
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        lines = [
            "=" * 60,
            "Data Ingestion Summary",
            "=" * 60,
            f"File: {self.file_path}",
            f"Duration: {duration:.2f} seconds",
            f"Total rows read: {self.total_rows_read}",
            f"Valid rows: {self.valid_rows}",
            f"Skipped rows: {self.skipped_rows}",
            f"Errors: {len(self.errors)}",
            f"Warnings: {len(self.warnings)}",
        ]
        
        if self.errors:
            lines.append("\nTop Errors (first 5):")
            for error in self.errors[:5]:
                lines.append(f"  - {error}")
            if len(self.errors) > 5:
                lines.append(f"  ... and {len(self.errors) - 5} more")
        
        if self.warnings:
            lines.append("\nTop Warnings (first 5):")
            for warning in self.warnings[:5]:
                lines.append(f"  - {warning}")
            if len(self.warnings) > 5:
                lines.append(f"  ... and {len(self.warnings) - 5} more")
        
        if self.validation_summary:
            lines.append("\n" + str(self.validation_summary))
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class DataIngestionModule:
    """
    Main data ingestion module for cattle sensor data.
    
    Supports batch and incremental loading modes with comprehensive error handling,
    validation, and logging.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        log_file: str = "ingestion_errors.log",
        validate_data: bool = True,
    ):
        """
        Initialize data ingestion module.
        
        Args:
            log_dir: Directory for log files
            log_file: Name of error log file
            validate_data: Whether to validate data during ingestion
        """
        self.log_dir = Path(log_dir)
        self.log_file = self.log_dir / log_file
        self.validate_data = validate_data
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self._setup_file_logging()
        
        # Initialize parsers and validators
        self.csv_parser = CSVParser()
        self.timestamp_parser = TimestampParser()
        self.validator = DataValidator()
        
        # Track file positions for incremental reading
        self.file_positions: Dict[str, int] = {}
        
        logger.info("Data ingestion module initialized")
    
    def _setup_file_logging(self):
        """Setup file-based logging for errors."""
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    def load_batch(
        self,
        file_path: str,
        parse_timestamps: bool = True,
        chunk_size: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, IngestionSummary]:
        """
        Load entire CSV file in batch mode.
        
        Args:
            file_path: Path to CSV file
            parse_timestamps: Whether to parse timestamp column
            chunk_size: If specified, process file in chunks to save memory
            
        Returns:
            Tuple of (DataFrame with sensor data, ingestion summary)
        """
        summary = IngestionSummary()
        summary.file_path = file_path
        summary.start_time = datetime.now()
        
        logger.info(f"Starting batch load of {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                summary.errors.append(error_msg)
                logger.error(error_msg)
                summary.end_time = datetime.now()
                return pd.DataFrame(), summary
            
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                error_msg = f"File is empty: {file_path}"
                summary.errors.append(error_msg)
                logger.error(error_msg)
                summary.end_time = datetime.now()
                return pd.DataFrame(), summary
            
            # Read CSV file
            if chunk_size:
                # Process in chunks for large files
                df_list = []
                for chunk in pd.read_csv(
                    file_path,
                    delimiter=self.csv_parser.detect_delimiter(file_path),
                    encoding=self.csv_parser.detected_encoding or 'utf-8',
                    chunksize=chunk_size,
                ):
                    df_list.append(chunk)
                df = pd.concat(df_list, ignore_index=True)
            else:
                df = self.csv_parser.read_csv(file_path)
            
            summary.total_rows_read = len(df)
            logger.info(f"Read {len(df)} rows from CSV")
            
            # Parse timestamps if requested
            if parse_timestamps and 'timestamp' in df.columns:
                df = self._parse_timestamps(df, summary)
            
            # Validate data if requested
            if self.validate_data:
                df, validation_summary = self.validator.validate_dataframe(df)
                summary.validation_summary = validation_summary
                summary.valid_rows = validation_summary.rows_valid
                summary.skipped_rows = validation_summary.rows_skipped
                
                # Collect errors and warnings
                for error in validation_summary.errors:
                    summary.errors.append(str(error))
                for warning in validation_summary.warnings:
                    summary.warnings.append(str(warning))
            else:
                summary.valid_rows = len(df)
            
            summary.end_time = datetime.now()
            logger.info(f"Batch load complete: {summary.valid_rows} valid rows")
            
            return df, summary
        
        except Exception as e:
            error_msg = f"Failed to load file {file_path}: {str(e)}"
            summary.errors.append(error_msg)
            logger.exception(error_msg)
            summary.end_time = datetime.now()
            return pd.DataFrame(), summary
    
    def load_incremental(
        self,
        file_path: str,
        parse_timestamps: bool = True,
    ) -> Tuple[pd.DataFrame, IngestionSummary]:
        """
        Load new data from CSV file since last read (incremental mode).
        
        This simulates real-time data ingestion by tracking file position
        and only reading new rows.
        
        Args:
            file_path: Path to CSV file
            parse_timestamps: Whether to parse timestamp column
            
        Returns:
            Tuple of (DataFrame with new sensor data, ingestion summary)
        """
        summary = IngestionSummary()
        summary.file_path = file_path
        summary.start_time = datetime.now()
        
        logger.info(f"Starting incremental load of {file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                summary.errors.append(error_msg)
                logger.error(error_msg)
                summary.end_time = datetime.now()
                return pd.DataFrame(), summary
            
            # Get last file position
            last_position = self.file_positions.get(file_path, 0)
            
            # Read new data from file
            df, new_position = self.csv_parser.read_csv_incremental(
                file_path,
                last_position=last_position,
            )
            
            # Update file position
            self.file_positions[file_path] = new_position
            
            summary.total_rows_read = len(df)
            
            if len(df) == 0:
                logger.info("No new data found")
                summary.end_time = datetime.now()
                return df, summary
            
            logger.info(f"Read {len(df)} new rows from CSV")
            
            # Parse timestamps if requested
            if parse_timestamps and 'timestamp' in df.columns:
                df = self._parse_timestamps(df, summary)
            
            # Validate data if requested
            if self.validate_data:
                df, validation_summary = self.validator.validate_dataframe(df)
                summary.validation_summary = validation_summary
                summary.valid_rows = validation_summary.rows_valid
                summary.skipped_rows = validation_summary.rows_skipped
                
                # Collect errors and warnings
                for error in validation_summary.errors:
                    summary.errors.append(str(error))
                for warning in validation_summary.warnings:
                    summary.warnings.append(str(warning))
            else:
                summary.valid_rows = len(df)
            
            summary.end_time = datetime.now()
            logger.info(f"Incremental load complete: {summary.valid_rows} valid rows")
            
            return df, summary
        
        except Exception as e:
            error_msg = f"Failed to load file incrementally {file_path}: {str(e)}"
            summary.errors.append(error_msg)
            logger.exception(error_msg)
            summary.end_time = datetime.now()
            return pd.DataFrame(), summary
    
    def monitor_file(
        self,
        file_path: str,
        interval_seconds: int = 60,
        callback=None,
        max_iterations: Optional[int] = None,
    ):
        """
        Monitor CSV file for new data and load incrementally.
        
        This simulates real-time data ingestion by checking file periodically
        for new rows.
        
        Args:
            file_path: Path to CSV file to monitor
            interval_seconds: Check interval in seconds (default: 60 for 1 minute)
            callback: Optional callback function to call with new data
            max_iterations: Maximum number of check iterations (None for infinite)
        """
        logger.info(f"Starting file monitoring: {file_path} (interval: {interval_seconds}s)")
        
        iteration = 0
        while True:
            if max_iterations and iteration >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations}), stopping monitor")
                break
            
            # Load new data
            df, summary = self.load_incremental(file_path)
            
            # Call callback if provided and data was loaded
            if callback and len(df) > 0:
                callback(df, summary)
            
            # Wait for next interval
            iteration += 1
            time.sleep(interval_seconds)
    
    def monitor_directory(
        self,
        directory: str,
        file_pattern: str = "*.csv",
        interval_seconds: int = 60,
        callback=None,
        max_iterations: Optional[int] = None,
    ):
        """
        Monitor directory for new CSV files and load them.
        
        Args:
            directory: Directory to monitor
            file_pattern: File pattern to match (default: *.csv)
            interval_seconds: Check interval in seconds
            callback: Optional callback function to call with new data
            max_iterations: Maximum number of check iterations
        """
        logger.info(f"Starting directory monitoring: {directory} (pattern: {file_pattern})")
        
        processed_files = set()
        iteration = 0
        
        while True:
            if max_iterations and iteration >= max_iterations:
                logger.info(f"Reached max iterations ({max_iterations}), stopping monitor")
                break
            
            # Find CSV files in directory
            dir_path = Path(directory)
            csv_files = list(dir_path.glob(file_pattern))
            
            # Process new files
            for csv_file in csv_files:
                file_str = str(csv_file)
                
                if file_str not in processed_files:
                    logger.info(f"Found new file: {file_str}")
                    
                    # Load file in batch mode
                    df, summary = self.load_batch(file_str)
                    
                    # Call callback if provided and data was loaded
                    if callback and len(df) > 0:
                        callback(df, summary)
                    
                    processed_files.add(file_str)
            
            # Wait for next interval
            iteration += 1
            time.sleep(interval_seconds)
    
    def _parse_timestamps(
        self,
        df: pd.DataFrame,
        summary: IngestionSummary,
    ) -> pd.DataFrame:
        """
        Parse timestamp column to datetime.
        
        Args:
            df: DataFrame with timestamp column
            summary: Ingestion summary to update with errors
            
        Returns:
            DataFrame with parsed timestamps
        """
        try:
            # Detect timestamp format from sample
            sample_size = min(10, len(df))
            sample_timestamps = df['timestamp'].head(sample_size).tolist()
            
            self.timestamp_parser.detect_format(sample_timestamps)
            
            # Parse all timestamps
            df['timestamp'] = self.timestamp_parser.parse_series(df['timestamp'])
            
            logger.info("Successfully parsed timestamps")
            
        except Exception as e:
            error_msg = f"Failed to parse timestamps: {str(e)}"
            summary.errors.append(error_msg)
            logger.error(error_msg)
        
        return df
    
    def reset_file_position(self, file_path: str):
        """
        Reset file position for incremental reading.
        
        Args:
            file_path: Path to file to reset
        """
        if file_path in self.file_positions:
            del self.file_positions[file_path]
            logger.info(f"Reset file position for {file_path}")
    
    def get_file_position(self, file_path: str) -> int:
        """
        Get current file position for incremental reading.
        
        Args:
            file_path: Path to file
            
        Returns:
            Current file position (byte offset)
        """
        return self.file_positions.get(file_path, 0)
    
    def export_summary(self, summary: IngestionSummary, output_path: str):
        """
        Export ingestion summary to file.
        
        Args:
            summary: Ingestion summary to export
            output_path: Path to output file
        """
        try:
            with open(output_path, 'w') as f:
                f.write(str(summary))
            logger.info(f"Exported summary to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export summary: {e}")
