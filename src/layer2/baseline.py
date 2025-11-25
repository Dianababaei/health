"""
Baseline Temperature Calculation Module

Calculates individual animal temperature baselines for detecting anomalies.
Supports rolling window calculations and percentile-based approaches.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BaselineCalculator:
    """
    Calculates temperature baselines for individual animals.
    
    Features:
    - Rolling window baseline calculation
    - Percentile-based robust estimation
    - Time-of-day specific baselines
    - Circadian rhythm consideration
    """
    
    def __init__(
        self,
        window_hours: int = 336,  # 14 days = 14 * 24 = 336 hours
        percentile_lower: float = 25.0,
        percentile_upper: float = 75.0,
        min_samples: int = 60
    ):
        """
        Initialize baseline calculator.

        Args:
            window_hours: Window size for rolling baseline (default 336 hours = 14 days)
            percentile_lower: Lower percentile for baseline range (default 25th)
            percentile_upper: Upper percentile for baseline range (default 75th)
            min_samples: Minimum samples required for valid baseline (default 60)
        """
        self.window_hours = window_hours
        self.percentile_lower = percentile_lower
        self.percentile_upper = percentile_upper
        self.min_samples = min_samples

        logger.info(f"BaselineCalculator initialized: window={window_hours}h, "
                   f"percentiles=[{percentile_lower}, {percentile_upper}]")
    
    def calculate_baseline(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature'
    ) -> pd.DataFrame:
        """
        Calculate rolling baseline temperature.
        
        Args:
            data: DataFrame with timestamp and temperature columns
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            
        Returns:
            DataFrame with added baseline columns:
            - baseline_temp: median temperature in window
            - baseline_lower: lower bound (percentile_lower)
            - baseline_upper: upper bound (percentile_upper)
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Calculate rolling statistics
        window_size = f'{self.window_hours}H'
        
        df = df.set_index(timestamp_col)
        
        # Calculate rolling baseline (median)
        df['baseline_temp'] = df[temperature_col].rolling(
            window=window_size,
            min_periods=self.min_samples
        ).median()
        
        # Calculate rolling percentile bounds
        df['baseline_lower'] = df[temperature_col].rolling(
            window=window_size,
            min_periods=self.min_samples
        ).quantile(self.percentile_lower / 100.0)
        
        df['baseline_upper'] = df[temperature_col].rolling(
            window=window_size,
            min_periods=self.min_samples
        ).quantile(self.percentile_upper / 100.0)
        
        # Reset index
        df = df.reset_index()
        
        return df
    
    def calculate_individual_baseline(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        lookback_days: int = 7
    ) -> Dict[str, float]:
        """
        Calculate a single individual baseline from historical data.
        
        Uses data from the past N days to establish a stable baseline.
        
        Args:
            data: DataFrame with temperature data
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            lookback_days: Days of historical data to use
            
        Returns:
            Dictionary with baseline statistics:
            - baseline: median temperature
            - std: standard deviation
            - lower_bound: 25th percentile
            - upper_bound: 75th percentile
            - sample_count: number of samples used
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Filter to lookback period
        if len(df) > 0:
            latest_time = df[timestamp_col].max()
            cutoff_time = latest_time - timedelta(days=lookback_days)
            df = df[df[timestamp_col] >= cutoff_time]
        
        temps = df[temperature_col].dropna()
        
        if len(temps) < self.min_samples:
            logger.warning(f"Insufficient samples for baseline: {len(temps)} < {self.min_samples}")
            return {
                'baseline': None,
                'std': None,
                'lower_bound': None,
                'upper_bound': None,
                'sample_count': len(temps)
            }
        
        return {
            'baseline': float(temps.median()),
            'std': float(temps.std()),
            'lower_bound': float(temps.quantile(self.percentile_lower / 100.0)),
            'upper_bound': float(temps.quantile(self.percentile_upper / 100.0)),
            'sample_count': len(temps)
        }
    
    def calculate_hourly_baselines(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature'
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate hour-specific baselines for circadian patterns.
        
        Args:
            data: DataFrame with temperature data
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            
        Returns:
            Dictionary mapping hour (0-23) to baseline statistics
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        df['hour'] = df[timestamp_col].dt.hour
        
        hourly_baselines = {}
        
        for hour in range(24):
            hour_data = df[df['hour'] == hour][temperature_col].dropna()
            
            if len(hour_data) >= 10:  # Need at least 10 samples per hour
                hourly_baselines[hour] = {
                    'baseline': float(hour_data.median()),
                    'std': float(hour_data.std()),
                    'lower_bound': float(hour_data.quantile(self.percentile_lower / 100.0)),
                    'upper_bound': float(hour_data.quantile(self.percentile_upper / 100.0)),
                    'sample_count': len(hour_data)
                }
            else:
                hourly_baselines[hour] = {
                    'baseline': None,
                    'std': None,
                    'lower_bound': None,
                    'upper_bound': None,
                    'sample_count': len(hour_data)
                }
        
        return hourly_baselines
    
    def get_baseline_at_time(
        self,
        hourly_baselines: Dict[int, Dict[str, float]],
        timestamp: datetime
    ) -> Optional[float]:
        """
        Get baseline temperature for a specific timestamp using hourly baselines.
        
        Args:
            hourly_baselines: Dictionary from calculate_hourly_baselines
            timestamp: Timestamp to get baseline for
            
        Returns:
            Baseline temperature or None if not available
        """
        hour = timestamp.hour
        
        if hour in hourly_baselines:
            return hourly_baselines[hour].get('baseline')
        
        return None
