"""
Feature Engineering for Cattle Behavior Classification

This module implements feature extraction functions for detecting ruminating
and feeding behaviors from neck-mounted accelerometer and gyroscope sensor data.

Features are based on literature-backed behavioral signatures:
- Ruminating: 40-60 cycles/min chewing pattern in Mya, synchronized Lyg patterns
- Feeding: Head-down Lyg pitch angle, lateral Mya variance, sustained head-down periods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import signal, stats
from scipy.fft import fft, fftfreq


class BehaviorFeatureExtractor:
    """
    Extract behavioral features from sensor data for cattle behavior classification.
    
    Focuses on ruminating and feeding detection based on literature-backed
    sensor signatures from neck-mounted accelerometers and gyroscopes.
    """
    
    def __init__(self, sampling_rate: float = 1.0, window_minutes: int = 10):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Samples per minute (default: 1.0)
            window_minutes: Window size for rolling statistics in minutes (default: 10)
        """
        self.sampling_rate = sampling_rate
        self.window_minutes = window_minutes
        self.window_samples = int(window_minutes * sampling_rate)
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all behavioral features from sensor data.
        
        Args:
            df: DataFrame with columns: timestamp, temperature, fxa, mya, rza, sxg, lyg, dzg, state
            
        Returns:
            DataFrame with extracted features
        """
        features = []
        
        # Process data in rolling windows
        for i in range(len(df)):
            # Define window boundaries
            start_idx = max(0, i - self.window_samples + 1)
            end_idx = i + 1
            window_data = df.iloc[start_idx:end_idx]
            
            # Skip if window too small
            if len(window_data) < self.window_samples // 2:
                continue
            
            # Extract features for this window
            feature_dict = {}
            
            # Add raw sensor values at current timestep
            feature_dict.update(self._extract_current_values(df.iloc[i]))
            
            # Extract ruminating features
            feature_dict.update(self._extract_ruminating_features(window_data))
            
            # Extract feeding features
            feature_dict.update(self._extract_feeding_features(window_data))
            
            # Extract rolling window statistics
            feature_dict.update(self._extract_rolling_statistics(window_data))
            
            # Extract motion intensity metrics
            feature_dict.update(self._extract_motion_intensity(window_data))
            
            # Extract temporal features
            feature_dict.update(self._extract_temporal_features(df.iloc[i], window_data))
            
            # Add label if available
            if 'state' in df.columns:
                feature_dict['state'] = df.iloc[i]['state']
            
            # Add timestamp
            if 'timestamp' in df.columns:
                feature_dict['timestamp'] = df.iloc[i]['timestamp']
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _extract_current_values(self, row: pd.Series) -> Dict[str, float]:
        """Extract current sensor values."""
        return {
            'current_temperature': row.get('temperature', np.nan),
            'current_fxa': row.get('fxa', np.nan),
            'current_mya': row.get('mya', np.nan),
            'current_rza': row.get('rza', np.nan),
            'current_sxg': row.get('sxg', np.nan),
            'current_lyg': row.get('lyg', np.nan),
            'current_dzg': row.get('dzg', np.nan),
        }
    
    def _extract_ruminating_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features for ruminating detection.
        
        Key indicators:
        - Mya frequency analysis targeting 40-60 cycles/min (0.67-1.0 Hz)
        - Lyg pitch variance showing rhythmic head movement
        - Cross-correlation between Mya and Lyg signals
        - Spectral power in chewing frequency bands
        
        Args:
            window_data: Window of sensor data
            
        Returns:
            Dictionary of ruminating features
        """
        features = {}
        
        mya = window_data['mya'].values
        lyg = window_data['lyg'].values
        
        # Mya frequency analysis (chewing pattern detection)
        features.update(self._extract_frequency_features(
            mya, 'mya', target_freq_range=(0.67, 1.0)
        ))
        
        # Lyg frequency analysis (head bobbing)
        features.update(self._extract_frequency_features(
            lyg, 'lyg', target_freq_range=(0.67, 1.0)
        ))
        
        # Lyg pitch variance (rhythmic head movement)
        features['ruminating_lyg_variance'] = np.var(lyg)
        features['ruminating_lyg_std'] = np.std(lyg)
        features['ruminating_lyg_range'] = np.ptp(lyg)
        
        # Cross-correlation between Mya and Lyg
        if len(mya) > 1 and len(lyg) > 1:
            # Normalize signals
            mya_norm = (mya - np.mean(mya)) / (np.std(mya) + 1e-8)
            lyg_norm = (lyg - np.mean(lyg)) / (np.std(lyg) + 1e-8)
            
            # Compute cross-correlation at zero lag
            xcorr = np.correlate(mya_norm, lyg_norm, mode='valid')[0] / len(mya_norm)
            features['ruminating_mya_lyg_xcorr'] = xcorr
            
            # Compute correlation coefficient
            features['ruminating_mya_lyg_corr'] = np.corrcoef(mya, lyg)[0, 1] if len(mya) > 1 else 0.0
        else:
            features['ruminating_mya_lyg_xcorr'] = 0.0
            features['ruminating_mya_lyg_corr'] = 0.0
        
        # Spectral power in chewing frequency bands (40-60 cycles/min = 0.67-1.0 Hz)
        mya_power = self._compute_spectral_power(mya, freq_range=(0.67, 1.0))
        lyg_power = self._compute_spectral_power(lyg, freq_range=(0.67, 1.0))
        
        features['ruminating_mya_spectral_power'] = mya_power
        features['ruminating_lyg_spectral_power'] = lyg_power
        features['ruminating_combined_spectral_power'] = mya_power + lyg_power
        
        # Rhythmicity score (how regular is the pattern)
        features['ruminating_mya_rhythmicity'] = self._compute_rhythmicity(mya)
        features['ruminating_lyg_rhythmicity'] = self._compute_rhythmicity(lyg)
        
        return features
    
    def _extract_feeding_features(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract features for feeding detection.
        
        Key indicators:
        - Lyg mean pitch angle (head-down position, negative values)
        - Lyg pitch variance (head movement during feeding)
        - Mya lateral variance (side-to-side browsing)
        - Duration of sustained head-down periods
        
        Args:
            window_data: Window of sensor data
            
        Returns:
            Dictionary of feeding features
        """
        features = {}
        
        lyg = window_data['lyg'].values
        mya = window_data['mya'].values
        rza = window_data['rza'].values
        
        # Lyg mean pitch angle (head-down indicator)
        features['feeding_lyg_mean'] = np.mean(lyg)
        features['feeding_lyg_median'] = np.median(lyg)
        
        # Negative Lyg values indicate head-down position
        features['feeding_lyg_negative_ratio'] = np.sum(lyg < 0) / len(lyg) if len(lyg) > 0 else 0.0
        features['feeding_lyg_mean_negative'] = np.mean(lyg[lyg < 0]) if np.any(lyg < 0) else 0.0
        
        # Lyg pitch variance (head movement variability)
        features['feeding_lyg_variance'] = np.var(lyg)
        features['feeding_lyg_std'] = np.std(lyg)
        features['feeding_lyg_range'] = np.ptp(lyg)
        
        # Mya lateral variance (side-to-side browsing)
        features['feeding_mya_variance'] = np.var(mya)
        features['feeding_mya_std'] = np.std(mya)
        features['feeding_mya_range'] = np.ptp(mya)
        features['feeding_mya_mean_abs'] = np.mean(np.abs(mya))
        
        # Duration of sustained head-down periods
        # Head-down threshold: Lyg < -10 degrees/second or significantly negative
        head_down_threshold = -10.0
        head_down_mask = lyg < head_down_threshold
        
        if np.any(head_down_mask):
            # Find contiguous head-down periods
            head_down_periods = self._find_contiguous_periods(head_down_mask)
            features['feeding_head_down_duration'] = np.max(head_down_periods) if head_down_periods else 0.0
            features['feeding_head_down_ratio'] = np.sum(head_down_mask) / len(lyg)
            features['feeding_head_down_count'] = len(head_down_periods)
        else:
            features['feeding_head_down_duration'] = 0.0
            features['feeding_head_down_ratio'] = 0.0
            features['feeding_head_down_count'] = 0
        
        # Standing posture indicator (feeding occurs while standing)
        features['feeding_rza_mean'] = np.mean(rza)
        features['feeding_standing_ratio'] = np.sum(rza > 0.7) / len(rza) if len(rza) > 0 else 0.0
        
        # Bite frequency estimation (30-90 bites/min = 0.5-1.5 Hz)
        mya_freq_features = self._extract_frequency_features(
            mya, 'feeding_mya_bite', target_freq_range=(0.5, 1.5)
        )
        features.update(mya_freq_features)
        
        return features
    
    def _extract_rolling_statistics(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute rolling window statistics for all sensor axes.
        
        Statistics: mean, std, min, max, range
        
        Args:
            window_data: Window of sensor data
            
        Returns:
            Dictionary of rolling statistics
        """
        features = {}
        
        sensors = ['fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg', 'temperature']
        stats_funcs = {
            'mean': np.mean,
            'std': np.std,
            'min': np.min,
            'max': np.max,
            'range': np.ptp,
            'median': np.median,
            'q25': lambda x: np.percentile(x, 25),
            'q75': lambda x: np.percentile(x, 75),
        }
        
        for sensor in sensors:
            if sensor in window_data.columns:
                data = window_data[sensor].values
                for stat_name, stat_func in stats_funcs.items():
                    features[f'rolling_{sensor}_{stat_name}'] = stat_func(data)
        
        return features
    
    def _extract_motion_intensity(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute motion intensity metrics combining multiple axes.
        
        Args:
            window_data: Window of sensor data
            
        Returns:
            Dictionary of motion intensity features
        """
        features = {}
        
        # Overall acceleration magnitude
        fxa = window_data['fxa'].values
        mya = window_data['mya'].values
        rza = window_data['rza'].values
        
        accel_magnitude = np.sqrt(fxa**2 + mya**2 + rza**2)
        features['motion_accel_magnitude_mean'] = np.mean(accel_magnitude)
        features['motion_accel_magnitude_std'] = np.std(accel_magnitude)
        features['motion_accel_magnitude_max'] = np.max(accel_magnitude)
        
        # Overall gyroscope magnitude
        sxg = window_data['sxg'].values
        lyg = window_data['lyg'].values
        dzg = window_data['dzg'].values
        
        gyro_magnitude = np.sqrt(sxg**2 + lyg**2 + dzg**2)
        features['motion_gyro_magnitude_mean'] = np.mean(gyro_magnitude)
        features['motion_gyro_magnitude_std'] = np.std(gyro_magnitude)
        features['motion_gyro_magnitude_max'] = np.max(gyro_magnitude)
        
        # Combined motion intensity score
        features['motion_intensity_score'] = (
            features['motion_accel_magnitude_std'] + 
            features['motion_gyro_magnitude_std'] * 0.1  # Scale gyro to comparable range
        )
        
        # Signal vector magnitude (SVM) - common in activity recognition
        svm = np.mean(np.abs(fxa) + np.abs(mya) + np.abs(rza))
        features['motion_svm'] = svm
        
        return features
    
    def _extract_temporal_features(self, row: pd.Series, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extract temporal features.
        
        Args:
            row: Current data row
            window_data: Window of sensor data
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Time of day features
        if 'timestamp' in row.index and pd.notna(row['timestamp']):
            timestamp = pd.to_datetime(row['timestamp'])
            features['temporal_hour'] = timestamp.hour
            features['temporal_minute'] = timestamp.minute
            features['temporal_hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            features['temporal_hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            
            # Day/night indicator
            features['temporal_is_daytime'] = 1.0 if 6 <= timestamp.hour < 18 else 0.0
        else:
            features['temporal_hour'] = 0
            features['temporal_minute'] = 0
            features['temporal_hour_sin'] = 0.0
            features['temporal_hour_cos'] = 1.0
            features['temporal_is_daytime'] = 0.0
        
        # Window duration
        features['temporal_window_size'] = len(window_data)
        
        return features
    
    def _extract_frequency_features(
        self, 
        signal_data: np.ndarray, 
        prefix: str,
        target_freq_range: Tuple[float, float] = (0.67, 1.0)
    ) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT.
        
        Args:
            signal_data: Time series signal
            prefix: Prefix for feature names
            target_freq_range: Target frequency range (Hz) to analyze
            
        Returns:
            Dictionary of frequency features
        """
        features = {}
        
        if len(signal_data) < 4:
            features[f'{prefix}_dominant_freq'] = 0.0
            features[f'{prefix}_spectral_energy'] = 0.0
            features[f'{prefix}_freq_in_target_range'] = 0.0
            return features
        
        # Compute FFT
        N = len(signal_data)
        yf = fft(signal_data - np.mean(signal_data))  # Remove DC component
        xf = fftfreq(N, 1.0 / self.sampling_rate)[:N//2]
        power = 2.0/N * np.abs(yf[:N//2])
        
        # Dominant frequency
        if len(power) > 0:
            peak_idx = np.argmax(power)
            features[f'{prefix}_dominant_freq'] = xf[peak_idx] if peak_idx < len(xf) else 0.0
            features[f'{prefix}_dominant_power'] = power[peak_idx]
        else:
            features[f'{prefix}_dominant_freq'] = 0.0
            features[f'{prefix}_dominant_power'] = 0.0
        
        # Total spectral energy
        features[f'{prefix}_spectral_energy'] = np.sum(power**2)
        
        # Check if dominant frequency is in target range
        dom_freq = features[f'{prefix}_dominant_freq']
        features[f'{prefix}_freq_in_target_range'] = float(
            target_freq_range[0] <= dom_freq <= target_freq_range[1]
        )
        
        # Energy in target frequency range
        freq_mask = (xf >= target_freq_range[0]) & (xf <= target_freq_range[1])
        features[f'{prefix}_target_band_energy'] = np.sum(power[freq_mask]**2) if np.any(freq_mask) else 0.0
        
        return features
    
    def _compute_spectral_power(
        self, 
        signal_data: np.ndarray, 
        freq_range: Tuple[float, float]
    ) -> float:
        """
        Compute spectral power in a specific frequency range.
        
        Args:
            signal_data: Time series signal
            freq_range: Frequency range (Hz) to compute power
            
        Returns:
            Spectral power in the specified range
        """
        if len(signal_data) < 4:
            return 0.0
        
        # Compute FFT
        N = len(signal_data)
        yf = fft(signal_data - np.mean(signal_data))
        xf = fftfreq(N, 1.0 / self.sampling_rate)[:N//2]
        power = 2.0/N * np.abs(yf[:N//2])
        
        # Sum power in frequency range
        freq_mask = (xf >= freq_range[0]) & (xf <= freq_range[1])
        total_power = np.sum(power[freq_mask]**2) if np.any(freq_mask) else 0.0
        
        return total_power
    
    def _compute_rhythmicity(self, signal_data: np.ndarray) -> float:
        """
        Compute rhythmicity score (regularity of oscillations).
        
        Uses autocorrelation to measure periodicity.
        
        Args:
            signal_data: Time series signal
            
        Returns:
            Rhythmicity score (0-1, higher = more rhythmic)
        """
        if len(signal_data) < 4:
            return 0.0
        
        # Normalize signal
        signal_norm = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        
        # Compute autocorrelation
        autocorr = np.correlate(signal_norm, signal_norm, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation (excluding zero lag)
        if len(autocorr) > 2:
            # Look for first peak after zero lag
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            if len(peaks) > 0:
                # Rhythmicity is the height of the first peak
                return float(autocorr[peaks[0] + 1])
        
        return 0.0
    
    def _find_contiguous_periods(self, mask: np.ndarray) -> List[int]:
        """
        Find lengths of contiguous True periods in a boolean mask.
        
        Args:
            mask: Boolean mask array
            
        Returns:
            List of period lengths
        """
        periods = []
        current_length = 0
        
        for value in mask:
            if value:
                current_length += 1
            else:
                if current_length > 0:
                    periods.append(current_length)
                    current_length = 0
        
        # Add final period if ended on True
        if current_length > 0:
            periods.append(current_length)
        
        return periods


def extract_features_from_dataframe(
    df: pd.DataFrame, 
    sampling_rate: float = 1.0, 
    window_minutes: int = 10
) -> pd.DataFrame:
    """
    Convenience function to extract features from a DataFrame.
    
    Args:
        df: Input DataFrame with sensor data
        sampling_rate: Samples per minute
        window_minutes: Window size for rolling statistics
        
    Returns:
        DataFrame with extracted features
    """
    extractor = BehaviorFeatureExtractor(
        sampling_rate=sampling_rate,
        window_minutes=window_minutes
    )
    return extractor.extract_all_features(df)
