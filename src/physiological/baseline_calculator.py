"""
Baseline Temperature Calculator Module

Calculates individual baseline temperatures using multi-day rolling windows,
circadian detrending, and robust statistics to exclude outliers and anomalies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass, field
from scipy import stats

from .circadian_extractor import CircadianExtractor, CircadianProfile

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Container for baseline calculation results."""
    
    cow_id: int
    timestamp: datetime
    baseline_temp: float
    calculation_window_days: int
    samples_used: int
    outliers_excluded: int
    circadian_amplitude: float
    circadian_confidence: float
    confidence_score: float
    method: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'cow_id': self.cow_id,
            'timestamp': self.timestamp,
            'baseline_temp': self.baseline_temp,
            'calculation_window_days': self.calculation_window_days,
            'samples_used': self.samples_used,
            'outliers_excluded': self.outliers_excluded,
            'circadian_amplitude': self.circadian_amplitude,
            'circadian_confidence': self.circadian_confidence,
            'confidence_score': self.confidence_score,
            'method': self.method,
            **self.metadata
        }


class BaselineCalculator:
    """
    Calculate baseline temperatures for individual cows.
    
    Implements multi-day rolling windows, circadian rhythm extraction,
    robust statistics (median/trimmed mean), and anomaly exclusion.
    """
    
    def __init__(
        self,
        window_days: int = 14,
        min_data_days: int = 7,
        min_samples_per_day: int = 720,
        robust_method: str = "trimmed_mean",
        trim_percentage: float = 5.0,
        fever_threshold: float = 39.5,
        hypothermia_threshold: float = 37.0,
        max_temp_change_per_minute: float = 0.1,
        circadian_extractor: Optional[CircadianExtractor] = None,
    ):
        """
        Initialize baseline calculator.
        
        Args:
            window_days: Rolling window size in days
            min_data_days: Minimum days of data required
            min_samples_per_day: Minimum samples per day
            robust_method: "median", "trimmed_mean", or "winsorized_mean"
            trim_percentage: Percentage to trim from each tail
            fever_threshold: Temperature threshold for fever exclusion (°C)
            hypothermia_threshold: Temperature threshold for low exclusion (°C)
            max_temp_change_per_minute: Max rate of change (°C/min)
            circadian_extractor: Custom CircadianExtractor instance
        """
        self.window_days = window_days
        self.min_data_days = min_data_days
        self.min_samples_per_day = min_samples_per_day
        self.robust_method = robust_method
        self.trim_percentage = trim_percentage
        self.fever_threshold = fever_threshold
        self.hypothermia_threshold = hypothermia_threshold
        self.max_temp_change_per_minute = max_temp_change_per_minute
        
        # Initialize circadian extractor
        if circadian_extractor is None:
            self.circadian_extractor = CircadianExtractor()
        else:
            self.circadian_extractor = circadian_extractor
        
        logger.info(
            f"BaselineCalculator initialized: window={window_days}d, "
            f"method={robust_method}, fever_threshold={fever_threshold}°C"
        )
    
    def calculate_baseline(
        self,
        df: pd.DataFrame,
        cow_id: int,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        current_time: Optional[datetime] = None,
    ) -> BaselineResult:
        """
        Calculate baseline temperature for a cow.
        
        Args:
            df: DataFrame with temperature data
            cow_id: Cow identifier
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            current_time: Current timestamp (defaults to latest in data)
            
        Returns:
            BaselineResult with calculated baseline and metadata
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        # Ensure timestamp is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Set current time to latest data point if not provided
        if current_time is None:
            current_time = df[timestamp_col].max()
        
        # Filter to window
        window_start = current_time - timedelta(days=self.window_days)
        window_df = df[
            (df[timestamp_col] >= window_start) &
            (df[timestamp_col] <= current_time)
        ].copy()
        
        if window_df.empty:
            raise ValueError(
                f"No data in window: {window_start} to {current_time}"
            )
        
        # Validate sufficient data
        self._validate_data_sufficiency(window_df, timestamp_col)
        
        # Step 1: Exclude anomalies
        clean_df, n_excluded = self._exclude_anomalies(
            window_df, timestamp_col, temperature_col
        )
        
        logger.info(
            f"Cow {cow_id}: Excluded {n_excluded} anomalous readings "
            f"({n_excluded / len(window_df) * 100:.1f}%)"
        )
        
        # Step 2: Extract circadian profile
        circadian_profile = self.circadian_extractor.extract_circadian_profile(
            clean_df, timestamp_col, temperature_col
        )
        
        # Step 3: Detrend temperatures (remove circadian component)
        detrended_df = self.circadian_extractor.detrend_temperatures(
            clean_df, circadian_profile, timestamp_col, temperature_col
        )
        
        # Step 4: Calculate baseline using robust statistics
        baseline_temp = self._calculate_robust_baseline(
            detrended_df['detrended_temp'].values
        )
        
        # Add back mean temperature (baseline is relative to overall mean)
        baseline_temp += circadian_profile.mean_temp
        
        # Step 5: Calculate confidence score
        confidence = self._calculate_confidence(
            clean_df,
            circadian_profile,
            len(window_df),
            n_excluded,
        )
        
        # Create result
        result = BaselineResult(
            cow_id=cow_id,
            timestamp=current_time,
            baseline_temp=baseline_temp,
            calculation_window_days=self.window_days,
            samples_used=len(clean_df),
            outliers_excluded=n_excluded,
            circadian_amplitude=circadian_profile.amplitude,
            circadian_confidence=circadian_profile.confidence,
            confidence_score=confidence,
            method=self.robust_method,
            metadata={
                'peak_hour': circadian_profile.peak_hour,
                'trough_hour': circadian_profile.trough_hour,
                'window_start': window_start,
                'window_end': current_time,
            }
        )
        
        logger.info(
            f"Cow {cow_id}: Baseline calculated: {baseline_temp:.3f}°C "
            f"(confidence={confidence:.2f})"
        )
        
        return result
    
    def _validate_data_sufficiency(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
    ) -> None:
        """
        Validate that DataFrame has sufficient data for baseline calculation.
        
        Args:
            df: DataFrame to validate
            timestamp_col: Name of timestamp column
            
        Raises:
            ValueError if insufficient data
        """
        # Check number of unique days
        df['date'] = df[timestamp_col].dt.date
        n_unique_days = df['date'].nunique()
        
        if n_unique_days < self.min_data_days:
            raise ValueError(
                f"Insufficient data: {n_unique_days} days < "
                f"{self.min_data_days} required"
            )
        
        # Check average samples per day
        avg_samples_per_day = len(df) / n_unique_days
        
        if avg_samples_per_day < self.min_samples_per_day:
            logger.warning(
                f"Low sampling rate: {avg_samples_per_day:.0f} samples/day < "
                f"{self.min_samples_per_day} recommended"
            )
        
        df.drop(columns=['date'], inplace=True)
    
    def _exclude_anomalies(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        temperature_col: str,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Exclude anomalous temperature readings.
        
        Excludes:
        - Fever temperatures (> threshold)
        - Hypothermic temperatures (< threshold)
        - Rapid temperature changes (artifacts)
        
        Args:
            df: DataFrame with temperature data
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            
        Returns:
            Tuple of (cleaned DataFrame, number of excluded readings)
        """
        df = df.copy().sort_values(timestamp_col)
        initial_count = len(df)
        
        # Exclude extreme temperatures
        temp_mask = (
            (df[temperature_col] >= self.hypothermia_threshold) &
            (df[temperature_col] <= self.fever_threshold)
        )
        df = df[temp_mask]
        
        # Exclude rapid changes (likely sensor artifacts)
        if len(df) > 1:
            df['time_diff'] = df[timestamp_col].diff().dt.total_seconds() / 60.0
            df['temp_diff'] = df[temperature_col].diff().abs()
            df['temp_rate'] = df['temp_diff'] / df['time_diff']
            
            # Keep readings with reasonable rate of change
            rate_mask = (
                df['temp_rate'].isna() |  # Keep first reading
                (df['temp_rate'] <= self.max_temp_change_per_minute)
            )
            df = df[rate_mask]
            
            # Clean up temporary columns
            df = df.drop(columns=['time_diff', 'temp_diff', 'temp_rate'])
        
        n_excluded = initial_count - len(df)
        
        return df, n_excluded
    
    def _calculate_robust_baseline(
        self,
        temperatures: np.ndarray,
    ) -> float:
        """
        Calculate baseline using robust statistics.
        
        Args:
            temperatures: Array of detrended temperatures
            
        Returns:
            Baseline temperature (mean of detrended values)
        """
        # Remove NaN values
        temps = temperatures[~np.isnan(temperatures)]
        
        if len(temps) == 0:
            raise ValueError("No valid temperatures after anomaly exclusion")
        
        if self.robust_method == "median":
            baseline = np.median(temps)
        
        elif self.robust_method == "trimmed_mean":
            # Trim percentage from each tail
            trim_fraction = self.trim_percentage / 100.0
            baseline = stats.trim_mean(temps, trim_fraction)
        
        elif self.robust_method == "winsorized_mean":
            # Winsorize (clip) extreme values
            from scipy.stats.mstats import winsorize
            limits = (self.trim_percentage / 100.0, self.trim_percentage / 100.0)
            winsorized = winsorize(temps, limits=limits)
            baseline = np.mean(winsorized)
        
        else:
            # Default to simple mean
            logger.warning(f"Unknown method '{self.robust_method}', using mean")
            baseline = np.mean(temps)
        
        return float(baseline)
    
    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        circadian_profile: CircadianProfile,
        original_count: int,
        excluded_count: int,
    ) -> float:
        """
        Calculate confidence score for baseline calculation.
        
        Based on:
        - Data completeness
        - Circadian profile quality
        - Exclusion rate
        
        Args:
            df: Cleaned DataFrame
            circadian_profile: Extracted circadian profile
            original_count: Original number of readings
            excluded_count: Number of excluded readings
            
        Returns:
            Confidence score (0-1)
        """
        # Data completeness score
        expected_samples = self.window_days * self.min_samples_per_day
        completeness = min(1.0, len(df) / expected_samples)
        
        # Circadian quality score
        circadian_quality = circadian_profile.confidence
        
        # Exclusion rate penalty (high exclusion reduces confidence)
        exclusion_rate = excluded_count / original_count if original_count > 0 else 0
        exclusion_penalty = max(0, 1.0 - 2 * exclusion_rate)  # Penalize >50% exclusion
        
        # Combined confidence
        confidence = (
            0.4 * completeness +
            0.4 * circadian_quality +
            0.2 * exclusion_penalty
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def calculate_baseline_multi_window(
        self,
        df: pd.DataFrame,
        cow_id: int,
        window_days_list: List[int] = [7, 14, 30],
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        current_time: Optional[datetime] = None,
    ) -> Dict[int, BaselineResult]:
        """
        Calculate baselines for multiple window sizes.
        
        Args:
            df: DataFrame with temperature data
            cow_id: Cow identifier
            window_days_list: List of window sizes to calculate
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            current_time: Current timestamp (defaults to latest in data)
            
        Returns:
            Dictionary mapping window_days -> BaselineResult
        """
        results = {}
        
        for window_days in window_days_list:
            # Temporarily change window size
            original_window = self.window_days
            original_min_days = self.min_data_days
            
            try:
                self.window_days = window_days
                # Adjust min_data_days proportionally
                self.min_data_days = max(3, int(window_days * 0.7))
                
                result = self.calculate_baseline(
                    df, cow_id, timestamp_col, temperature_col, current_time
                )
                results[window_days] = result
                
            except ValueError as e:
                logger.warning(
                    f"Could not calculate {window_days}-day baseline: {e}"
                )
            
            finally:
                # Restore original settings
                self.window_days = original_window
                self.min_data_days = original_min_days
        
        return results
    
    def validate_baseline(
        self,
        baseline_temp: float,
        min_baseline: float = 37.5,
        max_baseline: float = 39.2,
    ) -> Tuple[bool, List[str]]:
        """
        Validate calculated baseline against physiological constraints.
        
        Args:
            baseline_temp: Calculated baseline temperature
            min_baseline: Minimum valid baseline (°C)
            max_baseline: Maximum valid baseline (°C)
            
        Returns:
            Tuple of (is_valid, warning_messages)
        """
        warnings = []
        is_valid = True
        
        if baseline_temp < min_baseline:
            warnings.append(
                f"Baseline too low: {baseline_temp:.2f}°C < {min_baseline}°C"
            )
            is_valid = False
        
        if baseline_temp > max_baseline:
            warnings.append(
                f"Baseline too high: {baseline_temp:.2f}°C > {max_baseline}°C"
            )
            is_valid = False
        
        # Check if within typical cattle range (38.0-39.0°C)
        if not (38.0 <= baseline_temp <= 39.0):
            warnings.append(
                f"Baseline outside typical range: {baseline_temp:.2f}°C "
                f"(typical: 38.0-39.0°C)"
            )
        
        return is_valid, warnings
