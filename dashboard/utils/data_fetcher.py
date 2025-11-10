"""
Data Fetcher Utilities for Real-Time Metrics Panel

Provides helper functions to retrieve and process latest sensor data
for the real-time metrics display panel.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def get_latest_sensor_readings(data_loader, time_window_minutes: int = 5) -> Optional[Dict[str, Any]]:
    """
    Get the latest sensor readings for all 7 parameters.
    
    Args:
        data_loader: DataLoader instance
        time_window_minutes: Time window to consider for latest reading (default: 5 minutes)
    
    Returns:
        Dictionary with latest sensor readings or None if no data available
    """
    try:
        # Load recent sensor data
        df = data_loader.load_sensor_data(
            time_range_hours=1,  # Load last hour to ensure we have data
            max_rows=1000
        )
        
        if df.empty:
            logger.warning("No sensor data available")
            return None
        
        # Get the most recent reading
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
            latest_row = df.iloc[-1]
            timestamp = latest_row['timestamp']
        else:
            latest_row = df.iloc[-1]
            timestamp = datetime.now()
        
        # Extract all 7 sensor parameters
        readings = {
            'timestamp': timestamp,
            'temperature': latest_row.get('temperature', None),
            'fxa': latest_row.get('fxa', None),
            'mya': latest_row.get('mya', None),
            'rza': latest_row.get('rza', None),
            'sxg': latest_row.get('sxg', None),
            'lyg': latest_row.get('lyg', None),
            'dzg': latest_row.get('dzg', None),
            'behavioral_state': latest_row.get('behavioral_state', 'unknown'),
        }
        
        # Calculate data freshness
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        
        time_diff = (datetime.now() - timestamp).total_seconds()
        readings['freshness_seconds'] = time_diff
        readings['is_stale'] = time_diff > (time_window_minutes * 60)
        
        return readings
        
    except Exception as e:
        logger.error(f"Error getting latest sensor readings: {e}")
        return None


def get_previous_readings(data_loader, lookback_minutes: int = 5) -> Optional[Dict[str, Any]]:
    """
    Get previous sensor readings for delta calculations.
    
    Args:
        data_loader: DataLoader instance
        lookback_minutes: How many minutes back to look for comparison
    
    Returns:
        Dictionary with previous sensor readings or None if not available
    """
    try:
        # Load recent sensor data
        df = data_loader.load_sensor_data(
            time_range_hours=1,
            max_rows=1000
        )
        
        if df.empty or len(df) < 2:
            return None
        
        # Sort by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Get reading from lookback_minutes ago (or closest available)
        if len(df) >= 2:
            # Try to find reading approximately lookback_minutes ago
            if 'timestamp' in df.columns:
                target_time = df.iloc[-1]['timestamp'] - timedelta(minutes=lookback_minutes)
                # Find closest reading to target time
                time_diffs = abs(df['timestamp'] - target_time)
                prev_idx = time_diffs.idxmin()
                prev_row = df.loc[prev_idx]
            else:
                # If no timestamp, just use second-to-last reading
                prev_row = df.iloc[-2]
            
            return {
                'temperature': prev_row.get('temperature', None),
                'fxa': prev_row.get('fxa', None),
                'mya': prev_row.get('mya', None),
                'rza': prev_row.get('rza', None),
                'sxg': prev_row.get('sxg', None),
                'lyg': prev_row.get('lyg', None),
                'dzg': prev_row.get('dzg', None),
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting previous readings: {e}")
        return None


def calculate_movement_intensity(fxa: float, mya: float, rza: float) -> Tuple[float, str]:
    """
    Calculate movement intensity from acceleration magnitudes.
    
    Args:
        fxa: Forward acceleration (g)
        mya: Lateral acceleration (g)
        rza: Vertical acceleration (g)
    
    Returns:
        Tuple of (intensity_value, intensity_label)
        intensity_value: 0-100 scale
        intensity_label: 'Low', 'Medium', or 'High'
    """
    try:
        # Calculate magnitude of acceleration vector
        magnitude = np.sqrt(fxa**2 + mya**2 + rza**2)
        
        # Scale to 0-100 (typical range for livestock is 0-2g)
        # Using exponential scaling to make differences more visible
        intensity_raw = magnitude * 50  # 2g would give 100
        intensity_value = min(100, max(0, intensity_raw))
        
        # Categorize intensity
        if intensity_value < 20:
            intensity_label = 'Low'
        elif intensity_value < 50:
            intensity_label = 'Medium'
        else:
            intensity_label = 'High'
        
        return intensity_value, intensity_label
        
    except Exception as e:
        logger.error(f"Error calculating movement intensity: {e}")
        return 0.0, 'Unknown'


def calculate_baseline_temperature_delta(
    current_temp: float,
    baseline_temp: float = 38.5
) -> Tuple[float, str]:
    """
    Calculate temperature difference from baseline and determine status.
    
    Args:
        current_temp: Current temperature reading (°C)
        baseline_temp: Baseline/normal temperature (°C, default: 38.5)
    
    Returns:
        Tuple of (delta, status)
        delta: Difference from baseline
        status: 'normal', 'fever', 'hypothermia'
    """
    try:
        delta = current_temp - baseline_temp
        
        # Determine status based on thresholds
        if current_temp >= 39.5:
            status = 'fever'
        elif current_temp <= 37.5:
            status = 'hypothermia'
        else:
            status = 'normal'
        
        return delta, status
        
    except Exception as e:
        logger.error(f"Error calculating baseline temperature delta: {e}")
        return 0.0, 'unknown'


def format_freshness_display(freshness_seconds: float) -> str:
    """
    Format data freshness for display.
    
    Args:
        freshness_seconds: Number of seconds since last update
    
    Returns:
        Human-readable string like "23 seconds ago" or "2 minutes ago"
    """
    try:
        if freshness_seconds < 60:
            return f"{int(freshness_seconds)} seconds ago"
        elif freshness_seconds < 3600:
            minutes = int(freshness_seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            hours = int(freshness_seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
    except:
        return "Unknown"


def get_sensor_deltas(current: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate deltas for all sensor parameters.
    
    Args:
        current: Current sensor readings
        previous: Previous sensor readings (or None)
    
    Returns:
        Dictionary with delta values for each sensor
    """
    deltas = {}
    
    if previous is None:
        return deltas
    
    sensor_params = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
    
    for param in sensor_params:
        curr_val = current.get(param)
        prev_val = previous.get(param)
        
        if curr_val is not None and prev_val is not None:
            try:
                deltas[param] = float(curr_val) - float(prev_val)
            except:
                deltas[param] = None
        else:
            deltas[param] = None
    
    return deltas


def is_value_concerning(param_name: str, value: float, delta: Optional[float] = None) -> bool:
    """
    Determine if a sensor value is concerning and should trigger visual alert.
    
    Args:
        param_name: Name of the sensor parameter
        value: Current value
        delta: Change from previous reading (optional)
    
    Returns:
        True if value is concerning, False otherwise
    """
    try:
        # Temperature thresholds
        if param_name == 'temperature':
            if value >= 39.5 or value <= 37.5:
                return True
            if delta is not None and abs(delta) > 0.5:  # Rapid temperature change
                return True
        
        # Acceleration thresholds (extreme values)
        if param_name in ['fxa', 'mya', 'rza']:
            if abs(value) > 1.5:  # Very high acceleration
                return True
        
        # Gyroscope thresholds (extreme rotation)
        if param_name in ['sxg', 'lyg', 'dzg']:
            if abs(value) > 100:  # Very high angular velocity
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking if value is concerning: {e}")
        return False


def get_5min_average_readings(data_loader) -> Optional[Dict[str, Any]]:
    """
    Get 5-minute average readings for smoother delta calculations.
    
    Args:
        data_loader: DataLoader instance
    
    Returns:
        Dictionary with averaged sensor readings or None
    """
    try:
        # Load last 5 minutes of data
        df = data_loader.load_sensor_data(
            time_range_hours=1,
            max_rows=1000
        )
        
        if df.empty:
            return None
        
        # Filter to last 5 minutes
        if 'timestamp' in df.columns:
            cutoff_time = datetime.now() - timedelta(minutes=5)
            df_recent = df[df['timestamp'] >= cutoff_time]
            
            if df_recent.empty:
                return None
            
            # Calculate averages
            averages = {
                'temperature': df_recent['temperature'].mean() if 'temperature' in df_recent.columns else None,
                'fxa': df_recent['fxa'].mean() if 'fxa' in df_recent.columns else None,
                'mya': df_recent['mya'].mean() if 'mya' in df_recent.columns else None,
                'rza': df_recent['rza'].mean() if 'rza' in df_recent.columns else None,
                'sxg': df_recent['sxg'].mean() if 'sxg' in df_recent.columns else None,
                'lyg': df_recent['lyg'].mean() if 'lyg' in df_recent.columns else None,
                'dzg': df_recent['dzg'].mean() if 'dzg' in df_recent.columns else None,
            }
            
            return averages
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting 5-minute average readings: {e}")
        return None
