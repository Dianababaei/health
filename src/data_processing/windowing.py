"""
Sliding window implementation for cattle sensor data.

Provides configurable window sizes with rolling statistics (mean, variance, min, max)
for all 7 sensor parameters. Supports both fixed and sliding window modes,
with buffer management for real-time streaming scenarios.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union, Any
import logging

logger = logging.getLogger(__name__)


class WindowStatistics:
    """Container for window statistics."""
    
    def __init__(
        self,
        window_start: datetime,
        window_end: datetime,
        statistics: Dict[str, Dict[str, float]],
        sample_count: int,
    ):
        """
        Initialize window statistics.
        
        Args:
            window_start: Start timestamp of window
            window_end: End timestamp of window
            statistics: Dictionary of sensor->statistic->value mappings
            sample_count: Number of samples in window
        """
        self.window_start = window_start
        self.window_end = window_end
        self.statistics = statistics
        self.sample_count = sample_count
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary format for ML consumption.
        
        Returns:
            Dictionary with flattened statistics
        """
        result = {
            'window_start': self.window_start,
            'window_end': self.window_end,
            'sample_count': self.sample_count,
        }
        
        # Flatten nested statistics
        for sensor, stats in self.statistics.items():
            for stat_name, value in stats.items():
                result[f'{sensor}_{stat_name}'] = value
        
        return result
    
    def __repr__(self):
        """String representation."""
        return (
            f"WindowStatistics({self.window_start} to {self.window_end}, "
            f"{self.sample_count} samples)"
        )


class WindowGenerator:
    """
    Generate windows with rolling statistics for sensor data.
    
    Supports configurable window sizes (5-10 minutes), both sliding and fixed
    window modes, and real-time streaming with buffer management.
    """
    
    # Required sensor columns
    SENSOR_COLUMNS = [
        'temperature',
        'fxa',
        'mya',
        'rza',
        'sxg',
        'lyg',
        'dzg',
    ]
    
    def __init__(
        self,
        window_size_minutes: int = 5,
        slide_interval_minutes: int = 1,
        min_samples_per_window: int = 1,
        include_median: bool = False,
        include_std: bool = False,
    ):
        """
        Initialize window generator.
        
        Args:
            window_size_minutes: Window size in minutes (5-10)
            slide_interval_minutes: Slide interval for sliding windows (minutes)
            min_samples_per_window: Minimum samples required for valid window
            include_median: Whether to include median in statistics
            include_std: Whether to include standard deviation in statistics
        """
        if window_size_minutes < 5 or window_size_minutes > 10:
            raise ValueError("Window size must be between 5 and 10 minutes")
        
        if slide_interval_minutes < 1:
            raise ValueError("Slide interval must be at least 1 minute")
        
        if slide_interval_minutes > window_size_minutes:
            logger.warning(
                f"Slide interval ({slide_interval_minutes}min) > window size "
                f"({window_size_minutes}min). Windows will not overlap."
            )
        
        self.window_size_minutes = window_size_minutes
        self.slide_interval_minutes = slide_interval_minutes
        self.min_samples_per_window = min_samples_per_window
        self.include_median = include_median
        self.include_std = include_std
        
        # Buffer for real-time streaming
        self.buffer: Optional[pd.DataFrame] = None
        self.buffer_max_size = window_size_minutes + 5  # Keep extra for edge cases
        
        logger.info(
            f"WindowGenerator initialized: window={window_size_minutes}min, "
            f"slide={slide_interval_minutes}min"
        )
    
    def _calculate_statistics(
        self,
        window_data: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate rolling statistics for all sensor columns in window.
        
        Args:
            window_data: DataFrame containing sensor data for window
            
        Returns:
            Dictionary mapping sensor->statistic->value
        """
        statistics = {}
        
        for sensor in self.SENSOR_COLUMNS:
            if sensor not in window_data.columns:
                logger.warning(f"Sensor '{sensor}' not found in data")
                continue
            
            sensor_data = window_data[sensor].dropna()
            
            if len(sensor_data) == 0:
                # No valid data for this sensor
                statistics[sensor] = {
                    'mean': np.nan,
                    'variance': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                }
                if self.include_median:
                    statistics[sensor]['median'] = np.nan
                if self.include_std:
                    statistics[sensor]['std'] = np.nan
                continue
            
            # Calculate basic statistics
            statistics[sensor] = {
                'mean': float(sensor_data.mean()),
                'variance': float(sensor_data.var()),
                'min': float(sensor_data.min()),
                'max': float(sensor_data.max()),
            }
            
            # Optional statistics
            if self.include_median:
                statistics[sensor]['median'] = float(sensor_data.median())
            if self.include_std:
                statistics[sensor]['std'] = float(sensor_data.std())
        
        return statistics
    
    def generate_sliding_windows(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp',
    ) -> List[WindowStatistics]:
        """
        Generate sliding (overlapping) windows with statistics.
        
        Args:
            df: DataFrame with sensor data and timestamps
            timestamp_column: Name of timestamp column
            
        Returns:
            List of WindowStatistics objects
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return []
        
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_column).reset_index(drop=True)
        
        windows = []
        window_duration = timedelta(minutes=self.window_size_minutes)
        slide_duration = timedelta(minutes=self.slide_interval_minutes)
        
        # Get time range
        start_time = df[timestamp_column].iloc[0]
        end_time = df[timestamp_column].iloc[-1]
        
        # Generate sliding windows
        current_start = start_time
        
        while current_start + window_duration <= end_time + timedelta(seconds=30):
            current_end = current_start + window_duration
            
            # Extract data in window
            mask = (
                (df[timestamp_column] >= current_start) &
                (df[timestamp_column] < current_end)
            )
            window_data = df[mask]
            
            # Check if window has enough samples
            if len(window_data) >= self.min_samples_per_window:
                statistics = self._calculate_statistics(window_data)
                
                window_stat = WindowStatistics(
                    window_start=current_start,
                    window_end=current_end,
                    statistics=statistics,
                    sample_count=len(window_data),
                )
                windows.append(window_stat)
            else:
                logger.debug(
                    f"Window {current_start} to {current_end} has only "
                    f"{len(window_data)} samples (min: {self.min_samples_per_window})"
                )
            
            # Slide window
            current_start += slide_duration
        
        logger.info(f"Generated {len(windows)} sliding windows")
        return windows
    
    def generate_fixed_windows(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp',
    ) -> List[WindowStatistics]:
        """
        Generate fixed (non-overlapping) windows with statistics.
        
        Args:
            df: DataFrame with sensor data and timestamps
            timestamp_column: Name of timestamp column
            
        Returns:
            List of WindowStatistics objects
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return []
        
        if timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_column).reset_index(drop=True)
        
        windows = []
        window_duration = timedelta(minutes=self.window_size_minutes)
        
        # Get time range
        start_time = df[timestamp_column].iloc[0]
        end_time = df[timestamp_column].iloc[-1]
        
        # Generate fixed windows
        current_start = start_time
        
        while current_start < end_time:
            current_end = current_start + window_duration
            
            # Extract data in window
            mask = (
                (df[timestamp_column] >= current_start) &
                (df[timestamp_column] < current_end)
            )
            window_data = df[mask]
            
            # Check if window has enough samples
            if len(window_data) >= self.min_samples_per_window:
                statistics = self._calculate_statistics(window_data)
                
                window_stat = WindowStatistics(
                    window_start=current_start,
                    window_end=current_end,
                    statistics=statistics,
                    sample_count=len(window_data),
                )
                windows.append(window_stat)
            else:
                logger.debug(
                    f"Fixed window {current_start} to {current_end} has only "
                    f"{len(window_data)} samples (min: {self.min_samples_per_window})"
                )
            
            # Move to next window (non-overlapping)
            current_start = current_end
        
        logger.info(f"Generated {len(windows)} fixed windows")
        return windows
    
    def to_dataframe(
        self,
        windows: List[WindowStatistics],
    ) -> pd.DataFrame:
        """
        Convert list of WindowStatistics to DataFrame format.
        
        Args:
            windows: List of WindowStatistics objects
            
        Returns:
            DataFrame with flattened statistics for ML consumption
        """
        if not windows:
            return pd.DataFrame()
        
        records = [w.to_dict() for w in windows]
        return pd.DataFrame(records)
    
    def update_buffer(
        self,
        new_data: pd.DataFrame,
        timestamp_column: str = 'timestamp',
    ) -> None:
        """
        Update internal buffer with new streaming data.
        
        Args:
            new_data: New sensor data to add to buffer
            timestamp_column: Name of timestamp column
        """
        if new_data.empty:
            return
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(new_data[timestamp_column]):
            new_data = new_data.copy()
            new_data[timestamp_column] = pd.to_datetime(new_data[timestamp_column])
        
        # Initialize or append to buffer
        if self.buffer is None:
            self.buffer = new_data.copy()
        else:
            self.buffer = pd.concat([self.buffer, new_data], ignore_index=True)
            self.buffer = self.buffer.sort_values(timestamp_column).reset_index(drop=True)
        
        # Trim buffer to max size (based on time)
        if len(self.buffer) > 0:
            max_age = timedelta(minutes=self.buffer_max_size)
            latest_time = self.buffer[timestamp_column].iloc[-1]
            cutoff_time = latest_time - max_age
            
            self.buffer = self.buffer[
                self.buffer[timestamp_column] >= cutoff_time
            ].reset_index(drop=True)
        
        logger.debug(f"Buffer updated: {len(self.buffer)} samples")
    
    def process_buffer(
        self,
        mode: str = 'sliding',
        timestamp_column: str = 'timestamp',
    ) -> List[WindowStatistics]:
        """
        Process current buffer to generate windows.
        
        Args:
            mode: 'sliding' or 'fixed'
            timestamp_column: Name of timestamp column
            
        Returns:
            List of WindowStatistics objects
        """
        if self.buffer is None or self.buffer.empty:
            logger.warning("Buffer is empty")
            return []
        
        if mode == 'sliding':
            return self.generate_sliding_windows(self.buffer, timestamp_column)
        elif mode == 'fixed':
            return self.generate_fixed_windows(self.buffer, timestamp_column)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'sliding' or 'fixed'")
    
    def clear_buffer(self) -> None:
        """Clear the internal buffer."""
        self.buffer = None
        logger.debug("Buffer cleared")
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """
        Get information about current buffer state.
        
        Returns:
            Dictionary with buffer information
        """
        if self.buffer is None or self.buffer.empty:
            return {
                'size': 0,
                'time_range': None,
                'duration_minutes': 0,
            }
        
        return {
            'size': len(self.buffer),
            'time_range': (
                self.buffer['timestamp'].iloc[0],
                self.buffer['timestamp'].iloc[-1],
            ),
            'duration_minutes': (
                (self.buffer['timestamp'].iloc[-1] - 
                 self.buffer['timestamp'].iloc[0]).total_seconds() / 60
            ),
        }


def create_window_summary(
    windows: List[WindowStatistics],
) -> Dict[str, Any]:
    """
    Create a summary of window statistics.
    
    Args:
        windows: List of WindowStatistics objects
        
    Returns:
        Dictionary with summary information
    """
    if not windows:
        return {
            'total_windows': 0,
            'time_range': None,
            'avg_samples_per_window': 0,
        }
    
    total_samples = sum(w.sample_count for w in windows)
    
    return {
        'total_windows': len(windows),
        'time_range': (windows[0].window_start, windows[-1].window_end),
        'avg_samples_per_window': total_samples / len(windows),
        'total_samples': total_samples,
    }
