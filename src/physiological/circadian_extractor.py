"""
Circadian Rhythm Extraction Module

Extracts and models circadian temperature patterns using time-of-day binning
and Fourier analysis. Separates normal daily variation from baseline shifts.
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CircadianProfile:
    """Container for circadian rhythm profile data."""
    
    hourly_means: np.ndarray  # Mean temperature for each hour (24 values)
    hourly_stds: np.ndarray   # Std deviation for each hour (24 values)
    hourly_counts: np.ndarray  # Sample count for each hour (24 values)
    amplitude: float  # Peak-to-trough amplitude (°C)
    peak_hour: float  # Hour of peak temperature (0-23)
    trough_hour: float  # Hour of trough temperature (0-23)
    mean_temp: float  # Overall mean temperature
    confidence: float  # Confidence score (0-1) based on data quality
    fourier_coefficients: Optional[np.ndarray] = None  # Fourier fit coefficients
    
    def get_circadian_component(self, hour: float) -> float:
        """
        Get circadian component for a specific hour.
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            Circadian temperature component (deviation from mean)
        """
        if self.fourier_coefficients is not None:
            # Use Fourier fit for smooth interpolation
            return self._fourier_component(hour)
        else:
            # Use binned means with linear interpolation
            return self._binned_component(hour)
    
    def _fourier_component(self, hour: float) -> float:
        """Calculate circadian component using Fourier series."""
        if self.fourier_coefficients is None:
            return 0.0
        
        # Convert hour to radians (2π cycle over 24 hours)
        theta = 2 * np.pi * hour / 24.0
        
        # Calculate Fourier series sum
        component = 0.0
        n_harmonics = len(self.fourier_coefficients) // 2
        
        for k in range(n_harmonics):
            a_k = self.fourier_coefficients[2 * k]
            b_k = self.fourier_coefficients[2 * k + 1]
            component += a_k * np.cos((k + 1) * theta) + b_k * np.sin((k + 1) * theta)
        
        return component
    
    def _binned_component(self, hour: float) -> float:
        """Calculate circadian component using binned means with interpolation."""
        # Get integer hour and fractional part
        hour_int = int(hour) % 24
        hour_frac = hour - hour_int
        
        # Linear interpolation between adjacent hours
        next_hour = (hour_int + 1) % 24
        
        curr_val = self.hourly_means[hour_int] - self.mean_temp
        next_val = self.hourly_means[next_hour] - self.mean_temp
        
        return curr_val * (1 - hour_frac) + next_val * hour_frac


class CircadianExtractor:
    """
    Extract circadian temperature patterns from time-series data.
    
    Uses time-of-day binning (24 hourly bins) to calculate circadian profile,
    then fits smooth curve to separate normal circadian variation from baseline shift.
    """
    
    def __init__(
        self,
        hourly_bins: int = 24,
        min_samples_per_bin: int = 10,
        method: str = "fourier",
        fourier_components: int = 2,
    ):
        """
        Initialize circadian extractor.
        
        Args:
            hourly_bins: Number of hourly bins (default 24)
            min_samples_per_bin: Minimum samples required per bin
            method: Extraction method ("fourier", "binned_mean", "spline")
            fourier_components: Number of harmonics for Fourier fit
        """
        self.hourly_bins = hourly_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.method = method
        self.fourier_components = fourier_components
        
        logger.info(
            f"CircadianExtractor initialized: bins={hourly_bins}, "
            f"method={method}, components={fourier_components}"
        )
    
    def extract_circadian_profile(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
    ) -> CircadianProfile:
        """
        Extract circadian profile from temperature data.
        
        Args:
            df: DataFrame with timestamp and temperature columns
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            
        Returns:
            CircadianProfile object with extracted pattern
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        if timestamp_col not in df.columns or temperature_col not in df.columns:
            raise ValueError(f"Required columns not found in DataFrame")
        
        # Ensure timestamp is datetime
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract hour of day
        df['hour_of_day'] = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60.0
        
        # Bin by hour
        hourly_means, hourly_stds, hourly_counts = self._bin_by_hour(
            df['hour_of_day'].values,
            df[temperature_col].values
        )
        
        # Calculate overall statistics
        mean_temp = np.nanmean(df[temperature_col].values)
        
        # Calculate amplitude and peak/trough hours
        amplitude = np.nanmax(hourly_means) - np.nanmin(hourly_means)
        peak_hour = float(np.nanargmax(hourly_means))
        trough_hour = float(np.nanargmin(hourly_means))
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(hourly_counts)
        
        # Fit Fourier series if requested
        fourier_coefficients = None
        if self.method == "fourier":
            fourier_coefficients = self._fit_fourier_series(
                hourly_means, self.fourier_components
            )
        
        profile = CircadianProfile(
            hourly_means=hourly_means,
            hourly_stds=hourly_stds,
            hourly_counts=hourly_counts,
            amplitude=amplitude,
            peak_hour=peak_hour,
            trough_hour=trough_hour,
            mean_temp=mean_temp,
            confidence=confidence,
            fourier_coefficients=fourier_coefficients,
        )
        
        logger.info(
            f"Circadian profile extracted: amplitude={amplitude:.3f}°C, "
            f"peak_hour={peak_hour:.1f}, confidence={confidence:.2f}"
        )
        
        return profile
    
    def _bin_by_hour(
        self,
        hours: np.ndarray,
        temperatures: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin temperatures by hour of day.
        
        Args:
            hours: Array of hour values (0-23.99)
            temperatures: Array of temperature values
            
        Returns:
            Tuple of (hourly_means, hourly_stds, hourly_counts)
        """
        hourly_means = np.full(self.hourly_bins, np.nan)
        hourly_stds = np.full(self.hourly_bins, np.nan)
        hourly_counts = np.zeros(self.hourly_bins, dtype=int)
        
        # Bin edges (0, 1, 2, ..., 24)
        bin_edges = np.linspace(0, 24, self.hourly_bins + 1)
        
        for i in range(self.hourly_bins):
            # Get temperatures in this hour bin
            mask = (hours >= bin_edges[i]) & (hours < bin_edges[i + 1])
            bin_temps = temperatures[mask]
            
            # Remove NaN values
            bin_temps = bin_temps[~np.isnan(bin_temps)]
            
            if len(bin_temps) > 0:
                hourly_means[i] = np.mean(bin_temps)
                hourly_stds[i] = np.std(bin_temps)
                hourly_counts[i] = len(bin_temps)
        
        # Handle bins with insufficient data using interpolation
        hourly_means = self._interpolate_missing_bins(hourly_means, hourly_counts)
        
        return hourly_means, hourly_stds, hourly_counts
    
    def _interpolate_missing_bins(
        self,
        values: np.ndarray,
        counts: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate missing or sparse bins using circular interpolation.
        
        Args:
            values: Array of binned values (may contain NaN)
            counts: Array of sample counts per bin
            
        Returns:
            Array with interpolated values
        """
        # Identify valid bins
        valid_mask = (counts >= self.min_samples_per_bin) & ~np.isnan(values)
        
        if np.sum(valid_mask) == 0:
            logger.warning("No valid bins for interpolation, returning zeros")
            return np.zeros_like(values)
        
        if np.sum(valid_mask) == len(values):
            # All bins are valid
            return values
        
        # Circular interpolation (wrap around 24 hours)
        valid_indices = np.where(valid_mask)[0]
        valid_values = values[valid_mask]
        
        # Create extended arrays for circular wrapping
        extended_indices = np.concatenate([
            valid_indices - len(values),
            valid_indices,
            valid_indices + len(values)
        ])
        extended_values = np.concatenate([valid_values, valid_values, valid_values])
        
        # Interpolate all positions
        result = np.interp(
            np.arange(len(values)),
            extended_indices,
            extended_values
        )
        
        return result
    
    def _fit_fourier_series(
        self,
        hourly_means: np.ndarray,
        n_components: int,
    ) -> np.ndarray:
        """
        Fit Fourier series to hourly means.
        
        Args:
            hourly_means: Array of 24 hourly mean temperatures
            n_components: Number of harmonic components
            
        Returns:
            Array of Fourier coefficients [a1, b1, a2, b2, ...]
        """
        # Remove mean (we model deviation from mean)
        y = hourly_means - np.nanmean(hourly_means)
        
        # Hour positions (0, 1, 2, ..., 23)
        x = np.arange(len(hourly_means))
        theta = 2 * np.pi * x / len(hourly_means)
        
        # Build design matrix for least squares
        # Each row: [cos(θ), sin(θ), cos(2θ), sin(2θ), ...]
        A = np.zeros((len(x), 2 * n_components))
        
        for k in range(n_components):
            A[:, 2 * k] = np.cos((k + 1) * theta)
            A[:, 2 * k + 1] = np.sin((k + 1) * theta)
        
        # Least squares fit
        coefficients, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        
        return coefficients
    
    def _calculate_confidence(self, hourly_counts: np.ndarray) -> float:
        """
        Calculate confidence score based on data coverage.
        
        Args:
            hourly_counts: Array of sample counts per bin
            
        Returns:
            Confidence score (0-1)
        """
        # Check how many bins have sufficient data
        valid_bins = np.sum(hourly_counts >= self.min_samples_per_bin)
        coverage_ratio = valid_bins / len(hourly_counts)
        
        # Check uniformity of distribution
        if np.sum(hourly_counts) > 0:
            expected_count = np.sum(hourly_counts) / len(hourly_counts)
            uniformity = 1.0 - np.std(hourly_counts) / (expected_count + 1e-6)
            uniformity = np.clip(uniformity, 0, 1)
        else:
            uniformity = 0.0
        
        # Combined confidence score
        confidence = 0.7 * coverage_ratio + 0.3 * uniformity
        
        return float(confidence)
    
    def detrend_temperatures(
        self,
        df: pd.DataFrame,
        profile: CircadianProfile,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
    ) -> pd.DataFrame:
        """
        Remove circadian component from temperature data.
        
        Args:
            df: DataFrame with timestamp and temperature columns
            profile: CircadianProfile to use for detrending
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            
        Returns:
            DataFrame with added 'detrended_temp' column
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract hour of day
        df['hour_of_day'] = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60.0
        
        # Calculate circadian component for each timestamp
        circadian_components = df['hour_of_day'].apply(
            lambda h: profile.get_circadian_component(h)
        )
        
        # Detrend by removing circadian component
        df['detrended_temp'] = df[temperature_col] - circadian_components
        
        # Clean up temporary column
        df = df.drop(columns=['hour_of_day'])
        
        logger.debug(f"Detrended {len(df)} temperature readings")
        
        return df
    
    def validate_circadian_profile(
        self,
        profile: CircadianProfile,
        expected_amplitude: float = 0.5,
        max_amplitude: float = 1.0,
        min_amplitude: float = 0.1,
    ) -> Tuple[bool, List[str]]:
        """
        Validate circadian profile against expected physiological patterns.
        
        Args:
            profile: CircadianProfile to validate
            expected_amplitude: Expected amplitude (°C)
            max_amplitude: Maximum valid amplitude (°C)
            min_amplitude: Minimum valid amplitude (°C)
            
        Returns:
            Tuple of (is_valid, warning_messages)
        """
        warnings = []
        is_valid = True
        
        # Check amplitude
        if profile.amplitude > max_amplitude:
            warnings.append(
                f"Amplitude too high: {profile.amplitude:.2f}°C > {max_amplitude}°C"
            )
            is_valid = False
        elif profile.amplitude < min_amplitude:
            warnings.append(
                f"Amplitude too low: {profile.amplitude:.2f}°C < {min_amplitude}°C"
            )
        
        # Check if amplitude is within reasonable range of expected
        if abs(profile.amplitude - expected_amplitude) > expected_amplitude:
            warnings.append(
                f"Amplitude deviation from expected: {profile.amplitude:.2f}°C "
                f"(expected ~{expected_amplitude:.2f}°C)"
            )
        
        # Check peak hour (should be in afternoon, typically 14-18)
        if not (12 <= profile.peak_hour <= 20):
            warnings.append(
                f"Unusual peak hour: {profile.peak_hour:.1f} "
                f"(expected 14-18, afternoon)"
            )
        
        # Check trough hour (should be in early morning, typically 2-6)
        if not ((0 <= profile.trough_hour <= 8) or (profile.trough_hour >= 22)):
            warnings.append(
                f"Unusual trough hour: {profile.trough_hour:.1f} "
                f"(expected 2-6, early morning)"
            )
        
        # Check confidence
        if profile.confidence < 0.5:
            warnings.append(
                f"Low confidence: {profile.confidence:.2f} "
                f"(insufficient or uneven data distribution)"
            )
            is_valid = False
        
        return is_valid, warnings
