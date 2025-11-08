"""
Sensor Data Normalization Module

This module provides sensor-specific normalization methods for cattle monitoring data.
Each sensor type has its own normalization approach based on expected ranges and 
physiological constraints.

Normalization Methods:
- Temperature: Min-max scaling (35-42°C → 0-1 range)
- Accelerations (Fxa, Mya, Rza): Z-score standardization (-2g to +2g expected range)
- Angular velocities (Sxg, Lyg, Dzg): Z-score standardization (sensor-dependent range)

All methods handle NaN values gracefully and support both single values and arrays/Series.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional

# Define expected ranges for each sensor type based on cattle physiology and sensor specs
TEMPERATURE_MIN = 35.0  # °C - lower bound for cattle body temperature
TEMPERATURE_MAX = 42.0  # °C - upper bound for cattle body temperature

# Acceleration expected ranges (in g units)
ACCELERATION_EXPECTED_MIN = -2.0  # g
ACCELERATION_EXPECTED_MAX = 2.0   # g
ACCELERATION_EXPECTED_MEAN = 0.0  # g (centered)
ACCELERATION_EXPECTED_STD = 1.0   # g (for standardization)

# Angular velocity expected ranges (in degrees/second)
GYROSCOPE_EXPECTED_MIN = -50.0   # deg/s
GYROSCOPE_EXPECTED_MAX = 50.0    # deg/s
GYROSCOPE_EXPECTED_MEAN = 0.0    # deg/s (centered)
GYROSCOPE_EXPECTED_STD = 20.0    # deg/s (for standardization)


def normalize_temperature(
    temperature: Union[float, np.ndarray, pd.Series],
    min_val: float = TEMPERATURE_MIN,
    max_val: float = TEMPERATURE_MAX
) -> Union[float, np.ndarray, pd.Series]:
    """
    Normalize temperature using min-max scaling to 0-1 range.
    
    Formula: (T - T_min) / (T_max - T_min)
    
    Args:
        temperature: Temperature value(s) in degrees Celsius
        min_val: Minimum temperature for normalization (default: 35°C)
        max_val: Maximum temperature for normalization (default: 42°C)
        
    Returns:
        Normalized temperature in range [0, 1]
        
    Examples:
        >>> normalize_temperature(38.5)  # Mid-range temperature
        0.5
        >>> normalize_temperature(35.0)  # Minimum
        0.0
        >>> normalize_temperature(42.0)  # Maximum
        1.0
    """
    if isinstance(temperature, pd.Series):
        return temperature.apply(lambda x: (x - min_val) / (max_val - min_val) if pd.notna(x) else np.nan)
    elif isinstance(temperature, np.ndarray):
        result = (temperature - min_val) / (max_val - min_val)
        return result
    else:
        if pd.isna(temperature) or temperature is None or np.isnan(temperature):
            return np.nan
        return (temperature - min_val) / (max_val - min_val)


def standardize_acceleration(
    acceleration: Union[float, np.ndarray, pd.Series],
    expected_mean: float = ACCELERATION_EXPECTED_MEAN,
    expected_std: float = ACCELERATION_EXPECTED_STD
) -> Union[float, np.ndarray, pd.Series]:
    """
    Standardize acceleration using z-score normalization.
    
    Formula: (a - mean) / std
    
    Uses expected mean and std based on typical cattle movement patterns
    (-2g to +2g range with mean=0, std=1).
    
    Args:
        acceleration: Acceleration value(s) in g units (Fxa, Mya, or Rza)
        expected_mean: Expected mean for standardization (default: 0.0g)
        expected_std: Expected standard deviation (default: 1.0g)
        
    Returns:
        Standardized acceleration (z-score)
        
    Examples:
        >>> standardize_acceleration(0.0)  # At mean
        0.0
        >>> standardize_acceleration(1.0)  # One std above mean
        1.0
        >>> standardize_acceleration(-2.0)  # Two std below mean
        -2.0
    """
    if isinstance(acceleration, pd.Series):
        return acceleration.apply(
            lambda x: (x - expected_mean) / expected_std if pd.notna(x) else np.nan
        )
    elif isinstance(acceleration, np.ndarray):
        result = (acceleration - expected_mean) / expected_std
        return result
    else:
        if pd.isna(acceleration) or acceleration is None or np.isnan(acceleration):
            return np.nan
        return (acceleration - expected_mean) / expected_std


def standardize_angular_velocity(
    angular_velocity: Union[float, np.ndarray, pd.Series],
    expected_mean: float = GYROSCOPE_EXPECTED_MEAN,
    expected_std: float = GYROSCOPE_EXPECTED_STD
) -> Union[float, np.ndarray, pd.Series]:
    """
    Standardize angular velocity using z-score normalization.
    
    Formula: (ω - mean) / std
    
    Uses expected mean and std based on typical cattle movement patterns.
    
    Args:
        angular_velocity: Angular velocity value(s) in deg/s (Sxg, Lyg, or Dzg)
        expected_mean: Expected mean for standardization (default: 0.0 deg/s)
        expected_std: Expected standard deviation (default: 20.0 deg/s)
        
    Returns:
        Standardized angular velocity (z-score)
        
    Examples:
        >>> standardize_angular_velocity(0.0)   # At mean
        0.0
        >>> standardize_angular_velocity(20.0)  # One std above mean
        1.0
        >>> standardize_angular_velocity(-40.0) # Two std below mean
        -2.0
    """
    if isinstance(angular_velocity, pd.Series):
        return angular_velocity.apply(
            lambda x: (x - expected_mean) / expected_std if pd.notna(x) else np.nan
        )
    elif isinstance(angular_velocity, np.ndarray):
        result = (angular_velocity - expected_mean) / expected_std
        return result
    else:
        if pd.isna(angular_velocity) or angular_velocity is None or np.isnan(angular_velocity):
            return np.nan
        return (angular_velocity - expected_mean) / expected_std


def normalize_sensor_data(
    data: pd.DataFrame,
    normalize_temp: bool = True,
    standardize_accel: bool = True,
    standardize_gyro: bool = True
) -> pd.DataFrame:
    """
    Normalize all sensor parameters in a DataFrame.
    
    This is a convenience function that applies appropriate normalization
    to each sensor column based on its type.
    
    Args:
        data: DataFrame with sensor columns (temperature, fxa, mya, rza, sxg, lyg, dzg)
        normalize_temp: Whether to normalize temperature (default: True)
        standardize_accel: Whether to standardize accelerations (default: True)
        standardize_gyro: Whether to standardize angular velocities (default: True)
        
    Returns:
        DataFrame with normalized sensor values
        
    Note:
        Creates a copy of the input DataFrame to avoid modifying original data.
    """
    result = data.copy()
    
    # Normalize temperature
    if normalize_temp and 'temperature' in result.columns:
        result['temperature_norm'] = normalize_temperature(result['temperature'])
    
    # Standardize accelerations
    if standardize_accel:
        if 'fxa' in result.columns:
            result['fxa_std'] = standardize_acceleration(result['fxa'])
        if 'mya' in result.columns:
            result['mya_std'] = standardize_acceleration(result['mya'])
        if 'rza' in result.columns:
            result['rza_std'] = standardize_acceleration(result['rza'])
    
    # Standardize angular velocities
    if standardize_gyro:
        if 'sxg' in result.columns:
            result['sxg_std'] = standardize_angular_velocity(result['sxg'])
        if 'lyg' in result.columns:
            result['lyg_std'] = standardize_angular_velocity(result['lyg'])
        if 'dzg' in result.columns:
            result['dzg_std'] = standardize_angular_velocity(result['dzg'])
    
    return result


def inverse_normalize_temperature(
    normalized_temp: Union[float, np.ndarray, pd.Series],
    min_val: float = TEMPERATURE_MIN,
    max_val: float = TEMPERATURE_MAX
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert normalized temperature back to original Celsius scale.
    
    Args:
        normalized_temp: Normalized temperature value(s) in [0, 1] range
        min_val: Minimum temperature used in normalization (default: 35°C)
        max_val: Maximum temperature used in normalization (default: 42°C)
        
    Returns:
        Temperature in degrees Celsius
    """
    if isinstance(normalized_temp, pd.Series):
        return normalized_temp.apply(
            lambda x: x * (max_val - min_val) + min_val if pd.notna(x) else np.nan
        )
    elif isinstance(normalized_temp, np.ndarray):
        return normalized_temp * (max_val - min_val) + min_val
    else:
        if pd.isna(normalized_temp) or normalized_temp is None or np.isnan(normalized_temp):
            return np.nan
        return normalized_temp * (max_val - min_val) + min_val


def inverse_standardize_acceleration(
    standardized_accel: Union[float, np.ndarray, pd.Series],
    expected_mean: float = ACCELERATION_EXPECTED_MEAN,
    expected_std: float = ACCELERATION_EXPECTED_STD
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert standardized acceleration back to original g units.
    
    Args:
        standardized_accel: Standardized acceleration value(s) (z-score)
        expected_mean: Expected mean used in standardization (default: 0.0g)
        expected_std: Expected std used in standardization (default: 1.0g)
        
    Returns:
        Acceleration in g units
    """
    if isinstance(standardized_accel, pd.Series):
        return standardized_accel.apply(
            lambda x: x * expected_std + expected_mean if pd.notna(x) else np.nan
        )
    elif isinstance(standardized_accel, np.ndarray):
        return standardized_accel * expected_std + expected_mean
    else:
        if pd.isna(standardized_accel) or standardized_accel is None or np.isnan(standardized_accel):
            return np.nan
        return standardized_accel * expected_std + expected_mean


def inverse_standardize_angular_velocity(
    standardized_gyro: Union[float, np.ndarray, pd.Series],
    expected_mean: float = GYROSCOPE_EXPECTED_MEAN,
    expected_std: float = GYROSCOPE_EXPECTED_STD
) -> Union[float, np.ndarray, pd.Series]:
    """
    Convert standardized angular velocity back to original deg/s units.
    
    Args:
        standardized_gyro: Standardized angular velocity value(s) (z-score)
        expected_mean: Expected mean used in standardization (default: 0.0 deg/s)
        expected_std: Expected std used in standardization (default: 20.0 deg/s)
        
    Returns:
        Angular velocity in degrees/second
    """
    if isinstance(standardized_gyro, pd.Series):
        return standardized_gyro.apply(
            lambda x: x * expected_std + expected_mean if pd.notna(x) else np.nan
        )
    elif isinstance(standardized_gyro, np.ndarray):
        return standardized_gyro * expected_std + expected_mean
    else:
        if pd.isna(standardized_gyro) or standardized_gyro is None or np.isnan(standardized_gyro):
            return np.nan
        return standardized_gyro * expected_std + expected_mean
