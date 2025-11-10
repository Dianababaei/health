"""
Stress Behavior Detection Module

This module implements multi-axis movement variance analysis to detect stress behaviors
in cattle based on erratic movement patterns across acceleration and gyroscope sensors.

Stress indicators include:
- High simultaneous variance in Fxa, Mya, Rza (>2σ above baseline)
- Erratic Sxg/Lyg/Dzg patterns (rapid direction changes)
- Lack of rhythmic patterns characteristic of normal behaviors

The stress flag supplements primary behavioral state (e.g., "Walking + Stress").
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class StressDetectionResult:
    """Result of stress detection analysis."""
    is_stressed: bool
    stress_score: float  # 0.0-1.0
    variance_metrics: Dict[str, float]
    timestamp: Optional[pd.Timestamp] = None
    details: Optional[str] = None


class StressDetector:
    """
    Multi-axis variance stress detector for cattle behavioral monitoring.
    
    Analyzes movement patterns across all sensor axes to detect stress behaviors
    characterized by erratic, non-rhythmic movements.
    
    Attributes:
        window_size (int): Rolling window size in minutes (default: 5)
        variance_threshold_sigma (float): Threshold in standard deviations (default: 2.0)
        min_axes_threshold (int): Minimum axes exceeding threshold (default: 3)
    """
    
    def __init__(
        self,
        window_size: int = 5,
        variance_threshold_sigma: float = 2.0,
        min_axes_threshold: int = 3,
        sampling_rate: float = 1.0  # samples per minute
    ):
        """
        Initialize stress detector.
        
        Args:
            window_size: Rolling window size in samples (default: 5 = 5 minutes)
            variance_threshold_sigma: Number of std deviations above mean (default: 2.0)
            min_axes_threshold: Minimum number of axes that must exceed threshold (default: 3)
            sampling_rate: Sampling rate in samples per minute (default: 1.0)
        """
        self.window_size = window_size
        self.variance_threshold_sigma = variance_threshold_sigma
        self.min_axes_threshold = min_axes_threshold
        self.sampling_rate = sampling_rate
        
        # Baseline statistics (learned from data)
        self.baseline_stats: Optional[Dict[str, Dict[str, float]]] = None
        self.is_calibrated = False
    
    def calibrate(self, sensor_data: pd.DataFrame, percentile: float = 95.0):
        """
        Calibrate baseline statistics from normal behavior data.
        
        Args:
            sensor_data: DataFrame with sensor columns (fxa, mya, rza, sxg, lyg, dzg)
            percentile: Percentile to use for upper baseline (default: 95.0)
        """
        required_cols = ['fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        
        if not all(col in sensor_data.columns for col in required_cols):
            raise ValueError(f"Missing required sensor columns. Need: {required_cols}")
        
        # Calculate rolling variance for each sensor
        self.baseline_stats = {}
        
        for col in required_cols:
            variance = sensor_data[col].rolling(
                window=self.window_size,
                min_periods=1
            ).var()
            
            self.baseline_stats[col] = {
                'mean': variance.mean(),
                'std': variance.std(),
                'percentile_95': variance.quantile(percentile / 100.0),
                'max': variance.max()
            }
        
        self.is_calibrated = True
    
    def detect_stress_single(
        self,
        recent_data: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None
    ) -> StressDetectionResult:
        """
        Detect stress from a recent window of sensor data.
        
        Args:
            recent_data: DataFrame with last N samples (where N >= window_size)
            timestamp: Current timestamp for the detection
            
        Returns:
            StressDetectionResult with stress flag and metrics
        """
        if len(recent_data) < self.window_size:
            # Not enough data for reliable detection
            return StressDetectionResult(
                is_stressed=False,
                stress_score=0.0,
                variance_metrics={},
                timestamp=timestamp,
                details="Insufficient data for stress detection"
            )
        
        # Use last window_size samples
        window_data = recent_data.tail(self.window_size)
        
        # Calculate variance for each axis
        variance_metrics = self._calculate_variance_metrics(window_data)
        
        # Detect stress based on variance patterns
        stress_score, is_stressed, details = self._evaluate_stress(variance_metrics)
        
        return StressDetectionResult(
            is_stressed=is_stressed,
            stress_score=stress_score,
            variance_metrics=variance_metrics,
            timestamp=timestamp,
            details=details
        )
    
    def detect_stress_batch(
        self,
        sensor_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect stress for a batch of sensor readings.
        
        Args:
            sensor_data: DataFrame with sensor columns and optional timestamp
            
        Returns:
            DataFrame with added stress detection columns:
            - is_stressed: Boolean stress flag
            - stress_score: Stress score (0.0-1.0)
            - stress_details: Description of stress indicators
        """
        required_cols = ['fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        
        if not all(col in sensor_data.columns for col in required_cols):
            raise ValueError(f"Missing required sensor columns. Need: {required_cols}")
        
        # Calculate rolling variance for each sensor
        variance_df = pd.DataFrame()
        for col in required_cols:
            variance_df[f'{col}_var'] = sensor_data[col].rolling(
                window=self.window_size,
                min_periods=1
            ).var()
        
        # Detect stress for each row
        results = []
        for idx in range(len(sensor_data)):
            variance_metrics = {
                col: variance_df[f'{col}_var'].iloc[idx]
                for col in required_cols
            }
            
            stress_score, is_stressed, details = self._evaluate_stress(variance_metrics)
            
            results.append({
                'is_stressed': is_stressed,
                'stress_score': stress_score,
                'stress_details': details
            })
        
        # Add results to output dataframe
        result_df = sensor_data.copy()
        result_df['is_stressed'] = [r['is_stressed'] for r in results]
        result_df['stress_score'] = [r['stress_score'] for r in results]
        result_df['stress_details'] = [r['stress_details'] for r in results]
        
        return result_df
    
    def _calculate_variance_metrics(self, window_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate variance metrics for all sensor axes.
        
        Args:
            window_data: DataFrame with sensor readings
            
        Returns:
            Dictionary of variance values for each sensor
        """
        sensors = ['fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        variance_metrics = {}
        
        for sensor in sensors:
            if sensor in window_data.columns:
                variance_metrics[sensor] = window_data[sensor].var()
            else:
                variance_metrics[sensor] = 0.0
        
        return variance_metrics
    
    def _evaluate_stress(
        self,
        variance_metrics: Dict[str, float]
    ) -> Tuple[float, bool, str]:
        """
        Evaluate stress level based on variance metrics.
        
        Args:
            variance_metrics: Dictionary of variance values
            
        Returns:
            Tuple of (stress_score, is_stressed, details)
        """
        if not variance_metrics or all(v == 0.0 for v in variance_metrics.values()):
            return 0.0, False, "No variance data"
        
        # If calibrated, use baseline statistics
        if self.is_calibrated and self.baseline_stats:
            return self._evaluate_with_baseline(variance_metrics)
        else:
            return self._evaluate_without_baseline(variance_metrics)
    
    def _evaluate_with_baseline(
        self,
        variance_metrics: Dict[str, float]
    ) -> Tuple[float, bool, str]:
        """
        Evaluate stress using calibrated baseline statistics.
        
        Args:
            variance_metrics: Current variance measurements
            
        Returns:
            Tuple of (stress_score, is_stressed, details)
        """
        exceeded_axes = []
        z_scores = []
        
        for sensor, variance in variance_metrics.items():
            if sensor in self.baseline_stats:
                baseline = self.baseline_stats[sensor]
                mean = baseline['mean']
                std = baseline['std']
                
                if std > 0:
                    z_score = (variance - mean) / std
                    z_scores.append(z_score)
                    
                    if z_score > self.variance_threshold_sigma:
                        exceeded_axes.append(sensor)
        
        # Stress detected if multiple axes exceed threshold
        is_stressed = len(exceeded_axes) >= self.min_axes_threshold
        
        # Calculate stress score (0.0-1.0)
        if z_scores:
            avg_z_score = np.mean([max(0, z) for z in z_scores])
            stress_score = min(1.0, avg_z_score / (self.variance_threshold_sigma * 2))
        else:
            stress_score = 0.0
        
        if is_stressed:
            details = f"High variance in {len(exceeded_axes)} axes: {', '.join(exceeded_axes)}"
        else:
            details = "Normal variance levels"
        
        return stress_score, is_stressed, details
    
    def _evaluate_without_baseline(
        self,
        variance_metrics: Dict[str, float]
    ) -> Tuple[float, bool, str]:
        """
        Evaluate stress using absolute thresholds (no calibration).
        
        Uses literature-based absolute thresholds for high variance.
        
        Args:
            variance_metrics: Current variance measurements
            
        Returns:
            Tuple of (stress_score, is_stressed, details)
        """
        # Absolute variance thresholds (from literature)
        thresholds = {
            'fxa': 0.5,  # g²
            'mya': 0.5,  # g²
            'rza': 0.3,  # g²
            'sxg': 100.0,  # (°/s)²
            'lyg': 100.0,  # (°/s)²
            'dzg': 100.0,  # (°/s)²
        }
        
        exceeded_axes = []
        normalized_values = []
        
        for sensor, variance in variance_metrics.items():
            if sensor in thresholds:
                threshold = thresholds[sensor]
                normalized_value = variance / threshold
                normalized_values.append(normalized_value)
                
                if variance > threshold:
                    exceeded_axes.append(sensor)
        
        # Stress detected if multiple axes exceed threshold
        is_stressed = len(exceeded_axes) >= self.min_axes_threshold
        
        # Calculate stress score
        if normalized_values:
            avg_normalized = np.mean([max(0, v - 1) for v in normalized_values])
            stress_score = min(1.0, avg_normalized)
        else:
            stress_score = 0.0
        
        if is_stressed:
            details = f"High variance in {len(exceeded_axes)} axes: {', '.join(exceeded_axes)}"
        else:
            details = "Normal variance levels"
        
        return stress_score, is_stressed, details
    
    def get_baseline_stats(self) -> Optional[Dict]:
        """Get baseline statistics if calibrated."""
        return self.baseline_stats
    
    def reset_calibration(self):
        """Reset calibration to use absolute thresholds."""
        self.baseline_stats = None
        self.is_calibrated = False
