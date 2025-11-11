"""
Trend Calculations Utility
==========================
Utilities for calculating trend metrics, aggregations, and multi-period comparisons
for the historical trends analysis dashboard.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.signal import find_peaks


def calculate_trend_metrics(
    data: pd.DataFrame,
    value_column: str,
    time_column: str = 'timestamp'
) -> Dict[str, Any]:
    """
    Calculate comprehensive trend metrics for a time series.
    
    Args:
        data: DataFrame with time series data
        value_column: Column name containing the values to analyze
        time_column: Column name containing timestamps
        
    Returns:
        Dictionary with trend metrics including slope, variance, direction, etc.
    """
    if data.empty or value_column not in data.columns:
        return {
            'slope': 0.0,
            'variance': 0.0,
            'direction': 'stable',
            'mean': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'trend_strength': 0.0,
            'p_value': 1.0,
        }
    
    # Sort by time
    df = data.sort_values(time_column).copy()
    values = df[value_column].dropna()
    
    if len(values) < 2:
        return {
            'slope': 0.0,
            'variance': values.var() if len(values) > 0 else 0.0,
            'direction': 'stable',
            'mean': values.mean() if len(values) > 0 else 0.0,
            'std': values.std() if len(values) > 0 else 0.0,
            'min': values.min() if len(values) > 0 else 0.0,
            'max': values.max() if len(values) > 0 else 0.0,
            'trend_strength': 0.0,
            'p_value': 1.0,
        }
    
    # Calculate linear regression
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    
    # Determine trend direction
    if p_value < 0.05:  # Statistically significant
        if slope > 0:
            direction = 'improving' if value_column not in ['temperature', 'temp_deviation'] else 'deteriorating'
        elif slope < 0:
            direction = 'deteriorating' if value_column not in ['temperature', 'temp_deviation'] else 'improving'
        else:
            direction = 'stable'
    else:
        direction = 'stable'
    
    return {
        'slope': float(slope),
        'variance': float(values.var()),
        'direction': direction,
        'mean': float(values.mean()),
        'std': float(values.std()),
        'min': float(values.min()),
        'max': float(values.max()),
        'trend_strength': float(abs(r_value)),
        'p_value': float(p_value),
        'r_squared': float(r_value ** 2),
    }


def aggregate_by_period(
    data: pd.DataFrame,
    time_column: str = 'timestamp',
    period: str = 'D',
    agg_functions: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    """
    Aggregate data by time period (day, week, etc.).
    
    Args:
        data: DataFrame with time series data
        time_column: Column name containing timestamps
        period: Pandas period string ('D' for day, 'W' for week, 'H' for hour)
        agg_functions: Dictionary mapping column names to aggregation functions
        
    Returns:
        Aggregated DataFrame
    """
    if data.empty or time_column not in data.columns:
        return pd.DataFrame()
    
    df = data.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.set_index(time_column)
    
    # Default aggregation functions
    if agg_functions is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg_functions = {col: ['mean', 'min', 'max', 'std'] for col in numeric_cols}
    
    # Perform aggregation
    aggregated = df.groupby(pd.Grouper(freq=period)).agg(agg_functions)
    
    # Flatten multi-level columns
    if isinstance(aggregated.columns, pd.MultiIndex):
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    aggregated = aggregated.reset_index()
    return aggregated


def compare_periods(
    data: pd.DataFrame,
    periods: List[int],
    value_column: str,
    time_column: str = 'timestamp'
) -> Dict[str, pd.DataFrame]:
    """
    Compare data across multiple time periods.
    
    Args:
        data: DataFrame with complete time series
        periods: List of periods in days (e.g., [7, 14, 30, 90])
        value_column: Column to compare
        time_column: Column with timestamps
        
    Returns:
        Dictionary mapping period labels to DataFrames
    """
    if data.empty or time_column not in data.columns:
        return {}
    
    df = data.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df = df.sort_values(time_column)
    
    current_time = df[time_column].max()
    period_data = {}
    
    for period in periods:
        start_time = current_time - timedelta(days=period)
        period_df = df[df[time_column] >= start_time].copy()
        
        if not period_df.empty:
            period_data[f"{period}_days"] = period_df
    
    return period_data


def detect_recovery_deterioration(
    current_data: pd.DataFrame,
    historical_data: pd.DataFrame,
    value_column: str,
    threshold: float = 0.1
) -> Dict[str, Any]:
    """
    Detect recovery or deterioration by comparing current to historical trends.
    
    Args:
        current_data: Recent data (e.g., last 7 days)
        historical_data: Historical baseline data
        value_column: Column to analyze
        threshold: Minimum change threshold to consider significant
        
    Returns:
        Dictionary with recovery/deterioration indicators
    """
    if current_data.empty or historical_data.empty:
        return {
            'status': 'unknown',
            'change_pct': 0.0,
            'is_significant': False,
        }
    
    current_mean = current_data[value_column].mean()
    historical_mean = historical_data[value_column].mean()
    
    change_pct = ((current_mean - historical_mean) / historical_mean) * 100
    is_significant = abs(change_pct) >= (threshold * 100)
    
    if is_significant:
        if change_pct > 0:
            status = 'improving' if value_column == 'health_score' else 'deteriorating'
        else:
            status = 'deteriorating' if value_column == 'health_score' else 'improving'
    else:
        status = 'stable'
    
    return {
        'status': status,
        'change_pct': float(change_pct),
        'is_significant': is_significant,
        'current_mean': float(current_mean),
        'historical_mean': float(historical_mean),
    }


def calculate_rolling_statistics(
    data: pd.DataFrame,
    value_column: str,
    window: int = 7,
    time_column: str = 'timestamp'
) -> pd.DataFrame:
    """
    Calculate rolling statistics (moving average, std, etc.).
    
    Args:
        data: DataFrame with time series data
        value_column: Column to calculate statistics for
        window: Rolling window size (number of periods)
        time_column: Column with timestamps
        
    Returns:
        DataFrame with original data plus rolling statistics
    """
    if data.empty or value_column not in data.columns:
        return data
    
    df = data.copy()
    df = df.sort_values(time_column)
    
    # Calculate rolling statistics
    df[f'{value_column}_rolling_mean'] = df[value_column].rolling(window=window, min_periods=1).mean()
    df[f'{value_column}_rolling_std'] = df[value_column].rolling(window=window, min_periods=1).std()
    df[f'{value_column}_rolling_min'] = df[value_column].rolling(window=window, min_periods=1).min()
    df[f'{value_column}_rolling_max'] = df[value_column].rolling(window=window, min_periods=1).max()
    
    return df


def detect_patterns(
    data: pd.DataFrame,
    value_column: str,
    time_column: str = 'timestamp',
    pattern_type: str = 'peaks'
) -> List[Dict[str, Any]]:
    """
    Detect patterns in time series data (peaks, troughs, anomalies).
    
    Args:
        data: DataFrame with time series data
        value_column: Column to analyze
        time_column: Column with timestamps
        pattern_type: Type of pattern to detect ('peaks', 'troughs', 'anomalies')
        
    Returns:
        List of detected patterns with timestamps and values
    """
    if data.empty or value_column not in data.columns:
        return []
    
    df = data.sort_values(time_column).copy()
    values = df[value_column].values
    
    patterns = []
    
    if pattern_type == 'peaks':
        # Find peaks
        peaks, properties = find_peaks(values, prominence=0.5)
        for idx in peaks:
            patterns.append({
                'timestamp': df.iloc[idx][time_column],
                'value': float(values[idx]),
                'type': 'peak',
                'prominence': float(properties['prominences'][list(peaks).index(idx)]),
            })
    
    elif pattern_type == 'troughs':
        # Find troughs (peaks in inverted signal)
        troughs, properties = find_peaks(-values, prominence=0.5)
        for idx in troughs:
            patterns.append({
                'timestamp': df.iloc[idx][time_column],
                'value': float(values[idx]),
                'type': 'trough',
                'prominence': float(properties['prominences'][list(troughs).index(idx)]),
            })
    
    elif pattern_type == 'anomalies':
        # Detect anomalies using Z-score
        mean = np.mean(values)
        std = np.std(values)
        z_scores = np.abs((values - mean) / std) if std > 0 else np.zeros_like(values)
        
        anomaly_indices = np.where(z_scores > 3)[0]
        for idx in anomaly_indices:
            patterns.append({
                'timestamp': df.iloc[idx][time_column],
                'value': float(values[idx]),
                'type': 'anomaly',
                'z_score': float(z_scores[idx]),
            })
    
    return patterns


def calculate_multi_period_summary(
    data: pd.DataFrame,
    periods: List[int] = [7, 14, 30, 90],
    time_column: str = 'timestamp'
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate summary statistics for multiple time periods.
    
    Args:
        data: DataFrame with time series data
        periods: List of periods in days
        time_column: Column with timestamps
        
    Returns:
        Dictionary mapping period labels to summary statistics
    """
    if data.empty or time_column not in data.columns:
        return {}
    
    df = data.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    current_time = df[time_column].max()
    
    summaries = {}
    
    for period in periods:
        start_time = current_time - timedelta(days=period)
        period_df = df[df[time_column] >= start_time]
        
        if not period_df.empty:
            numeric_cols = period_df.select_dtypes(include=[np.number]).columns
            summary = {
                'period_days': period,
                'data_points': len(period_df),
                'start_time': period_df[time_column].min(),
                'end_time': period_df[time_column].max(),
            }
            
            # Add statistics for numeric columns
            for col in numeric_cols:
                summary[f'{col}_mean'] = float(period_df[col].mean())
                summary[f'{col}_std'] = float(period_df[col].std())
                summary[f'{col}_min'] = float(period_df[col].min())
                summary[f'{col}_max'] = float(period_df[col].max())
            
            summaries[f"{period}_days"] = summary
    
    return summaries
