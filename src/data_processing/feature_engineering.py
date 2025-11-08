"""
Feature Engineering Module

This module provides feature engineering functions for cattle behavioral monitoring.
Derives meaningful features from raw sensor data for machine learning models.

Features include:
- Motion intensity: Combined acceleration magnitude
- Orientation angles: Pitch and roll from accelerometer
- Rhythmic patterns: FFT, zero-crossings, peak detection (for rumination)
- Activity scores: Weighted combinations of sensor values
- Postural stability: Variance-based metrics
- Head movement intensity: Angular velocity magnitudes

All functions handle NaN values gracefully and are compatible with scikit-learn pipelines.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Optional, List
from scipy import signal
from scipy.fft import rfft, rfftfreq


def calculate_motion_intensity(
    fxa: Union[np.ndarray, pd.Series],
    mya: Union[np.ndarray, pd.Series],
    rza: Union[np.ndarray, pd.Series]
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate motion intensity as combined acceleration magnitude.
    
    Formula: sqrt(Fxa² + Mya² + Rza²)
    
    This represents the overall movement magnitude regardless of direction.
    Higher values indicate more intense movement (e.g., walking vs. lying).
    
    Args:
        fxa: Forward-backward acceleration (g)
        mya: Lateral acceleration (g)
        rza: Vertical acceleration (g)
        
    Returns:
        Motion intensity (g)
        
    Examples:
        >>> calculate_motion_intensity(0.5, 0.5, 0.5)
        0.866...
    """
    if isinstance(fxa, pd.Series):
        result = np.sqrt(fxa**2 + mya**2 + rza**2)
        return result
    else:
        return np.sqrt(fxa**2 + mya**2 + rza**2)


def calculate_pitch_angle(
    rza: Union[float, np.ndarray, pd.Series],
    g: float = 1.0
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate pitch angle from vertical acceleration (Rza).
    
    Formula: arcsin(Rza / g) in radians
    
    Pitch indicates head position:
    - Positive: Head up (standing)
    - Negative: Head down (lying, feeding)
    - Near zero: Horizontal orientation
    
    Args:
        rza: Vertical acceleration (g)
        g: Gravitational constant (default: 1.0g)
        
    Returns:
        Pitch angle in radians
        
    Note:
        Returns NaN for |Rza/g| > 1 (physically impossible)
    """
    normalized = rza / g
    
    if isinstance(normalized, pd.Series):
        # Clip to valid range for arcsin
        clipped = normalized.clip(-1.0, 1.0)
        return np.arcsin(clipped)
    elif isinstance(normalized, np.ndarray):
        clipped = np.clip(normalized, -1.0, 1.0)
        return np.arcsin(clipped)
    else:
        if pd.isna(normalized) or abs(normalized) > 1.0:
            return np.nan
        return np.arcsin(normalized)


def calculate_roll_angle(
    fxa: Union[float, np.ndarray, pd.Series],
    mya: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate roll angle from lateral and forward accelerations.
    
    Formula: arctan2(Mya, Fxa) in radians
    
    Roll indicates lateral tilt:
    - Useful for detecting lying on different sides
    - Combined with pitch for full posture estimation
    
    Args:
        fxa: Forward-backward acceleration (g)
        mya: Lateral acceleration (g)
        
    Returns:
        Roll angle in radians
    """
    if isinstance(fxa, pd.Series):
        return np.arctan2(mya, fxa)
    else:
        return np.arctan2(mya, fxa)


def calculate_activity_score(
    fxa: Union[np.ndarray, pd.Series],
    mya: Union[np.ndarray, pd.Series],
    rza: Union[np.ndarray, pd.Series],
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate weighted activity score from accelerations.
    
    Formula: w1*|Fxa| + w2*|Mya| + w3*|Rza|
    
    Default weights emphasize forward movement (indicative of walking/feeding).
    
    Args:
        fxa: Forward-backward acceleration (g)
        mya: Lateral acceleration (g)
        rza: Vertical acceleration (g)
        weights: Tuple of (w_fxa, w_mya, w_rza), must sum to ~1.0
        
    Returns:
        Activity score (weighted sum of absolute accelerations)
    """
    w_fxa, w_mya, w_rza = weights
    
    if isinstance(fxa, pd.Series):
        return w_fxa * np.abs(fxa) + w_mya * np.abs(mya) + w_rza * np.abs(rza)
    else:
        return w_fxa * np.abs(fxa) + w_mya * np.abs(mya) + w_rza * np.abs(rza)


def calculate_postural_stability(
    rza: Union[np.ndarray, pd.Series],
    window_size: Optional[int] = None
) -> Union[float, np.ndarray, pd.Series]:
    """
    Calculate postural stability from variance in vertical acceleration (Rza).
    
    Lower variance indicates stable posture (standing, lying).
    Higher variance indicates unstable posture or movement (walking, transitions).
    
    Args:
        rza: Vertical acceleration values (g)
        window_size: Number of samples for rolling window (None for whole series)
        
    Returns:
        Variance of Rza (lower = more stable)
        
    Note:
        If window_size is provided, returns rolling variance (for time series).
        Otherwise, returns single variance value for the entire input.
    """
    if isinstance(rza, pd.Series):
        if window_size is not None:
            return rza.rolling(window=window_size, center=True).var()
        else:
            return rza.var()
    else:
        if window_size is not None:
            # Calculate rolling variance for numpy array
            result = np.full_like(rza, np.nan, dtype=float)
            for i in range(len(rza)):
                start = max(0, i - window_size // 2)
                end = min(len(rza), i + window_size // 2 + 1)
                if end - start >= 2:  # Need at least 2 points for variance
                    result[i] = np.nanvar(rza[start:end])
            return result
        else:
            return np.nanvar(rza)


def calculate_head_movement_intensity(
    lyg: Union[np.ndarray, pd.Series],
    dzg: Union[np.ndarray, pd.Series]
) -> Union[np.ndarray, pd.Series]:
    """
    Calculate head movement intensity from pitch and yaw angular velocities.
    
    Formula: sqrt(Lyg² + Dzg²)
    
    Captures overall head movement regardless of direction.
    High values during feeding (head down motion) and ruminating (chewing).
    
    Args:
        lyg: Pitch angular velocity (deg/s)
        dzg: Yaw angular velocity (deg/s)
        
    Returns:
        Head movement intensity (deg/s)
    """
    if isinstance(lyg, pd.Series):
        return np.sqrt(lyg**2 + dzg**2)
    else:
        return np.sqrt(lyg**2 + dzg**2)


def extract_rhythmic_features(
    signal_data: Union[np.ndarray, pd.Series],
    sampling_rate: float = 1.0,
    target_freq_range: Tuple[float, float] = (0.67, 1.0)
) -> Dict[str, float]:
    """
    Extract rhythmic pattern features from time series data.
    
    Designed for detecting rumination patterns (40-60 cycles/min = 0.67-1.0 Hz).
    
    Features extracted:
    - dominant_frequency: Frequency with highest power in FFT
    - spectral_power: Total power in target frequency range
    - zero_crossing_rate: Rate of signal crossing zero
    - peak_count: Number of peaks detected
    - regularity_score: Measure of pattern consistency
    
    Args:
        signal_data: Time series data (e.g., Mya or Lyg for rumination)
        sampling_rate: Sampling rate in Hz (default: 1.0 Hz = 1 sample/sec)
        target_freq_range: Target frequency range in Hz (default: 0.67-1.0 Hz for rumination)
        
    Returns:
        Dictionary of rhythmic features
        
    Note:
        Returns NaN values if signal is too short or contains too many NaNs.
    """
    # Convert to numpy array and handle NaNs
    if isinstance(signal_data, pd.Series):
        data = signal_data.values
    else:
        data = np.array(signal_data)
    
    # Remove NaNs
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) < 10:  # Need minimum data points
        return {
            'dominant_frequency': np.nan,
            'spectral_power': np.nan,
            'zero_crossing_rate': np.nan,
            'peak_count': np.nan,
            'regularity_score': np.nan
        }
    
    # Zero-mean the signal
    data_centered = valid_data - np.mean(valid_data)
    
    # 1. FFT analysis
    n = len(data_centered)
    yf = rfft(data_centered)
    xf = rfftfreq(n, 1.0 / sampling_rate)
    power = np.abs(yf) ** 2
    
    # Find dominant frequency in target range
    mask = (xf >= target_freq_range[0]) & (xf <= target_freq_range[1])
    if np.any(mask):
        dominant_idx = np.argmax(power[mask])
        dominant_frequency = xf[mask][dominant_idx]
        spectral_power = np.sum(power[mask])
    else:
        dominant_frequency = np.nan
        spectral_power = 0.0
    
    # 2. Zero-crossing rate
    zero_crossings = np.where(np.diff(np.sign(data_centered)))[0]
    zero_crossing_rate = len(zero_crossings) / len(data_centered)
    
    # 3. Peak detection
    # Use scipy.signal.find_peaks with reasonable parameters
    peaks, _ = signal.find_peaks(data_centered, distance=int(sampling_rate * 0.5))
    peak_count = len(peaks)
    
    # 4. Regularity score (coefficient of variation of peak intervals)
    if len(peaks) > 2:
        peak_intervals = np.diff(peaks)
        regularity_score = 1.0 - (np.std(peak_intervals) / np.mean(peak_intervals))
        regularity_score = max(0.0, regularity_score)  # Clip to [0, 1]
    else:
        regularity_score = 0.0
    
    return {
        'dominant_frequency': dominant_frequency,
        'spectral_power': spectral_power,
        'zero_crossing_rate': zero_crossing_rate,
        'peak_count': peak_count,
        'regularity_score': regularity_score
    }


def engineer_features(
    data: pd.DataFrame,
    window_size: Optional[int] = None,
    sampling_rate: float = 1.0,
    include_rhythmic: bool = True
) -> pd.DataFrame:
    """
    Engineer all features from raw sensor data.
    
    Creates a feature-rich dataset suitable for machine learning models.
    
    Args:
        data: DataFrame with sensor columns (fxa, mya, rza, sxg, lyg, dzg)
        window_size: Window size for rolling statistics (None = no rolling)
        sampling_rate: Sampling rate in Hz (default: 1.0 Hz)
        include_rhythmic: Whether to extract rhythmic features (computationally expensive)
        
    Returns:
        DataFrame with engineered features
        
    Features created:
    - motion_intensity: Combined acceleration magnitude
    - pitch_angle: Orientation angle from Rza
    - roll_angle: Orientation angle from Fxa/Mya
    - activity_score: Weighted activity metric
    - postural_stability: Rza variance
    - head_movement_intensity: Combined Lyg/Dzg magnitude
    - rhythmic_* (if include_rhythmic=True): Rhythmic pattern features
    """
    result = data.copy()
    
    # Motion intensity
    if all(col in result.columns for col in ['fxa', 'mya', 'rza']):
        result['motion_intensity'] = calculate_motion_intensity(
            result['fxa'], result['mya'], result['rza']
        )
    
    # Orientation angles
    if 'rza' in result.columns:
        result['pitch_angle'] = calculate_pitch_angle(result['rza'])
    
    if 'fxa' in result.columns and 'mya' in result.columns:
        result['roll_angle'] = calculate_roll_angle(result['fxa'], result['mya'])
    
    # Activity score
    if all(col in result.columns for col in ['fxa', 'mya', 'rza']):
        result['activity_score'] = calculate_activity_score(
            result['fxa'], result['mya'], result['rza']
        )
    
    # Postural stability
    if 'rza' in result.columns:
        result['postural_stability'] = calculate_postural_stability(
            result['rza'], window_size=window_size
        )
    
    # Head movement intensity
    if 'lyg' in result.columns and 'dzg' in result.columns:
        result['head_movement_intensity'] = calculate_head_movement_intensity(
            result['lyg'], result['dzg']
        )
    
    # Rhythmic features (if requested and data is suitable)
    if include_rhythmic and len(result) >= 60:  # Need at least 1 min of data
        # Extract rhythmic features for Mya (rumination jaw movements)
        if 'mya' in result.columns:
            rhythmic_mya = extract_rhythmic_features(
                result['mya'], sampling_rate=sampling_rate
            )
            for key, value in rhythmic_mya.items():
                result[f'mya_{key}'] = value
        
        # Extract rhythmic features for Lyg (rumination head bobbing)
        if 'lyg' in result.columns:
            rhythmic_lyg = extract_rhythmic_features(
                result['lyg'], sampling_rate=sampling_rate
            )
            for key, value in rhythmic_lyg.items():
                result[f'lyg_{key}'] = value
    
    return result


def create_feature_vector(
    data: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    include_raw_normalized: bool = True
) -> pd.DataFrame:
    """
    Create ML-compatible feature vectors from engineered features.
    
    Assembles features into a structured format suitable for scikit-learn models.
    Handles missing values by forward-filling or dropping.
    
    Args:
        data: DataFrame with engineered features
        feature_columns: List of feature columns to include (None = auto-detect)
        include_raw_normalized: Whether to include normalized raw sensor values
        
    Returns:
        DataFrame with feature vectors, ready for ML models
        
    Note:
        The output is optimized for scikit-learn:
        - All numeric columns
        - No NaN values (filled or dropped)
        - Consistent column order
    """
    result = data.copy()
    
    # Auto-detect feature columns if not specified
    if feature_columns is None:
        feature_columns = []
        
        # Include normalized raw values if requested
        if include_raw_normalized:
            normalized_cols = [col for col in result.columns 
                             if col.endswith('_norm') or col.endswith('_std')]
            feature_columns.extend(normalized_cols)
        
        # Include engineered features
        engineered_cols = [
            'motion_intensity', 'pitch_angle', 'roll_angle',
            'activity_score', 'postural_stability', 'head_movement_intensity'
        ]
        feature_columns.extend([col for col in engineered_cols if col in result.columns])
        
        # Include rhythmic features
        rhythmic_cols = [col for col in result.columns 
                        if any(col.startswith(prefix) for prefix in ['mya_', 'lyg_'])
                        and any(col.endswith(suffix) for suffix in 
                               ['dominant_frequency', 'spectral_power', 
                                'zero_crossing_rate', 'peak_count', 'regularity_score'])]
        feature_columns.extend(rhythmic_cols)
    
    # Select only feature columns
    available_features = [col for col in feature_columns if col in result.columns]
    
    if not available_features:
        raise ValueError("No valid feature columns found in data")
    
    feature_data = result[available_features].copy()
    
    # Handle NaN values
    # For rolling window features, forward-fill then backward-fill
    feature_data = feature_data.ffill().bfill()
    
    # Drop any remaining rows with NaNs (edge cases)
    feature_data = feature_data.dropna()
    
    return feature_data
