"""
Circadian Rhythm Analysis Module

This module implements circadian rhythm analysis for cattle body temperature
using time-series decomposition techniques. It extracts daily temperature patterns,
detects rhythm loss, and prepares visualization data for dashboard overlay.

Key Features:
- Fourier Transform-based periodicity detection
- 24-hour pattern extraction with amplitude, phase, and baseline
- Rhythm loss detection (flattened rhythm, phase shifts, irregular patterns)
- Rhythm health score calculation (0-100 scale)
- Dashboard visualization data generation

Requirements:
- Minimum 3 days of temperature data for reliable rhythm extraction
- Rolling window approach for incremental updates
- Handles missing data and irregular sampling
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


@dataclass
class CircadianParameters:
    """
    Stores circadian rhythm parameters extracted from temperature data.
    
    Attributes:
        amplitude: Temperature variation amplitude (°C), expected ±0.5°C
        phase: Time of daily peak temperature (hours, 0-24)
        baseline: Mean temperature over the period (°C)
        period: Detected period in hours (should be ~24 for circadian)
        trough_time: Time of daily minimum temperature (hours, 0-24)
        confidence: Confidence in rhythm detection (0.0-1.0)
        last_updated: Timestamp of last parameter update
    """
    amplitude: float
    phase: float  # Peak time in hours
    baseline: float
    period: float  # Should be ~24 hours
    trough_time: float
    confidence: float
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'amplitude': self.amplitude,
            'phase': self.phase,
            'baseline': self.baseline,
            'period': self.period,
            'trough_time': self.trough_time,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class RhythmHealthMetrics:
    """
    Metrics for assessing circadian rhythm health.
    
    Attributes:
        health_score: Overall rhythm health (0-100, 100=perfect)
        is_rhythm_lost: Whether rhythm loss detected
        amplitude_stable: Whether amplitude is stable
        phase_stable: Whether phase timing is stable
        pattern_smoothness: How well data fits sinusoidal pattern (0.0-1.0)
        days_of_data: Number of days used in analysis
        rhythm_loss_reasons: List of reasons if rhythm is lost
    """
    health_score: float
    is_rhythm_lost: bool
    amplitude_stable: bool
    phase_stable: bool
    pattern_smoothness: float
    days_of_data: float
    rhythm_loss_reasons: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'health_score': self.health_score,
            'is_rhythm_lost': self.is_rhythm_lost,
            'amplitude_stable': self.amplitude_stable,
            'phase_stable': self.phase_stable,
            'pattern_smoothness': self.pattern_smoothness,
            'days_of_data': self.days_of_data,
            'rhythm_loss_reasons': self.rhythm_loss_reasons,
        }


class CircadianRhythmAnalyzer:
    """
    Analyze circadian rhythm patterns in cattle body temperature.
    
    Uses Fourier Transform and sinusoidal fitting to extract 24-hour
    temperature cycles. Detects rhythm disruptions that indicate illness or stress.
    
    Example:
        >>> analyzer = CircadianRhythmAnalyzer(
        ...     min_days=3,
        ...     expected_amplitude=0.5,
        ...     min_amplitude_threshold=0.3
        ... )
        >>> rhythm_params = analyzer.extract_circadian_rhythm(temperature_data)
        >>> health_metrics = analyzer.calculate_rhythm_health()
        >>> viz_data = analyzer.generate_visualization_data()
    """
    
    def __init__(
        self,
        min_days: float = 3.0,
        expected_amplitude: float = 0.5,
        min_amplitude_threshold: float = 0.3,
        max_phase_drift_hours: float = 2.0,
        max_missing_hours_per_day: int = 3,
        expected_period_hours: float = 24.0,
        period_tolerance_hours: float = 2.0,
    ):
        """
        Initialize circadian rhythm analyzer.
        
        Args:
            min_days: Minimum days of data required (default: 3.0)
            expected_amplitude: Expected normal amplitude in °C (default: 0.5)
            min_amplitude_threshold: Threshold for rhythm loss in °C (default: 0.3)
            max_phase_drift_hours: Maximum phase drift to consider stable (default: 2.0)
            max_missing_hours_per_day: Max missing hours tolerated per day (default: 3)
            expected_period_hours: Expected circadian period (default: 24.0)
            period_tolerance_hours: Tolerance for period detection (default: 2.0)
        """
        self.min_days = min_days
        self.expected_amplitude = expected_amplitude
        self.min_amplitude_threshold = min_amplitude_threshold
        self.max_phase_drift_hours = max_phase_drift_hours
        self.max_missing_hours_per_day = max_missing_hours_per_day
        self.expected_period_hours = expected_period_hours
        self.period_tolerance_hours = period_tolerance_hours
        
        # Store current rhythm parameters
        self.current_rhythm: Optional[CircadianParameters] = None
        self.previous_rhythm: Optional[CircadianParameters] = None
        
        # History for tracking stability
        self.rhythm_history: List[CircadianParameters] = []
        self.max_history_length = 30  # Keep 30 days of history
        
        logger.info(
            f"CircadianRhythmAnalyzer initialized: min_days={min_days}, "
            f"expected_amplitude={expected_amplitude}°C, "
            f"min_threshold={min_amplitude_threshold}°C"
        )
    
    def extract_circadian_rhythm(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
    ) -> Optional[CircadianParameters]:
        """
        Extract circadian rhythm parameters from temperature time series.
        
        Uses Fourier Transform to identify dominant 24-hour periodicity,
        then fits sinusoidal model to extract amplitude, phase, and baseline.
        
        Args:
            df: DataFrame with temperature time series
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            
        Returns:
            CircadianParameters object or None if insufficient data
        """
        # Validate input data
        if df.empty or timestamp_col not in df.columns or temperature_col not in df.columns:
            logger.warning("Invalid input data for circadian rhythm extraction")
            return None
        
        # Make a copy and sort by timestamp
        df = df.copy().sort_values(timestamp_col).reset_index(drop=True)
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Remove NaN values
        df = df.dropna(subset=[temperature_col])
        
        if len(df) == 0:
            logger.warning("No valid temperature data after removing NaN values")
            return None
        
        # Check if we have enough data
        time_span = (df[timestamp_col].max() - df[timestamp_col].min()).total_seconds() / 3600
        days_of_data = time_span / 24.0
        
        if days_of_data < self.min_days:
            logger.warning(
                f"Insufficient data for circadian analysis: {days_of_data:.1f} days "
                f"(minimum: {self.min_days} days)"
            )
            return None
        
        # Check for missing hours per day
        avg_samples_per_hour = len(df) / (time_span + 1e-6)
        if avg_samples_per_hour < (24 - self.max_missing_hours_per_day) / 24.0:
            logger.warning(
                f"Too many missing hours in data: {avg_samples_per_hour:.1f} samples/hour"
            )
        
        # Perform Fourier analysis to detect periodicity
        period, dominant_freq, fft_confidence = self._detect_periodicity_fft(
            df, timestamp_col, temperature_col
        )
        
        if period is None:
            logger.warning("Could not detect circadian periodicity using FFT")
            return None
        
        # Fit sinusoidal model to extract precise parameters
        params = self._fit_sinusoidal_model(
            df, timestamp_col, temperature_col, period
        )
        
        if params is None:
            logger.warning("Could not fit sinusoidal model to temperature data")
            return None
        
        amplitude, phase, baseline = params
        
        # Calculate trough time (opposite of peak)
        trough_time = (phase + 12.0) % 24.0
        
        # Calculate confidence based on FFT strength and fit quality
        confidence = min(1.0, fft_confidence)
        
        # Create CircadianParameters object
        circadian_params = CircadianParameters(
            amplitude=amplitude,
            phase=phase,
            baseline=baseline,
            period=period,
            trough_time=trough_time,
            confidence=confidence,
            last_updated=datetime.now(),
        )
        
        # Update stored parameters
        self.previous_rhythm = self.current_rhythm
        self.current_rhythm = circadian_params
        
        # Add to history
        self.rhythm_history.append(circadian_params)
        if len(self.rhythm_history) > self.max_history_length:
            self.rhythm_history.pop(0)
        
        logger.info(
            f"Circadian rhythm extracted: amplitude={amplitude:.2f}°C, "
            f"phase={phase:.1f}h, baseline={baseline:.2f}°C, "
            f"confidence={confidence:.2f}"
        )
        
        return circadian_params
    
    def _detect_periodicity_fft(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        temperature_col: str,
    ) -> Tuple[Optional[float], Optional[float], float]:
        """
        Detect dominant periodicity using Fast Fourier Transform.
        
        Args:
            df: DataFrame with time series
            timestamp_col: Timestamp column name
            temperature_col: Temperature column name
            
        Returns:
            Tuple of (period_hours, dominant_frequency, confidence)
        """
        # Resample to regular intervals (1 hour) to avoid FFT issues
        df = df.set_index(timestamp_col)
        df_resampled = df[temperature_col].resample('1H').mean().interpolate(method='linear')
        
        # Remove any remaining NaN
        df_resampled = df_resampled.dropna()
        
        if len(df_resampled) < 24:
            return None, None, 0.0
        
        # Detrend the signal (remove linear trend)
        temperature_values = df_resampled.values
        detrended = signal.detrend(temperature_values)
        
        # Apply window to reduce spectral leakage
        window = signal.windows.hann(len(detrended))
        windowed_signal = detrended * window
        
        # Compute FFT
        n = len(windowed_signal)
        fft_values = fft(windowed_signal)
        fft_freq = fftfreq(n, d=1.0)  # Sampling interval = 1 hour
        
        # Only consider positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_power = np.abs(fft_values[positive_freq_idx]) ** 2
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        # Look for dominant frequency near 1/24 Hz (24-hour period)
        target_freq = 1.0 / self.expected_period_hours
        freq_tolerance = 1.0 / (self.expected_period_hours - self.period_tolerance_hours)
        
        # Find frequencies in the target range
        freq_mask = (fft_freq_positive >= target_freq - freq_tolerance / 2) & \
                    (fft_freq_positive <= target_freq + freq_tolerance / 2)
        
        if not freq_mask.any():
            return None, None, 0.0
        
        # Find dominant frequency in range
        valid_power = fft_power[freq_mask]
        valid_freq = fft_freq_positive[freq_mask]
        
        if len(valid_power) == 0:
            return None, None, 0.0
        
        dominant_idx = np.argmax(valid_power)
        dominant_freq = valid_freq[dominant_idx]
        dominant_power = valid_power[dominant_idx]
        
        # Convert to period in hours
        period = 1.0 / dominant_freq if dominant_freq > 0 else None
        
        # Calculate confidence based on signal-to-noise ratio
        total_power = np.sum(fft_power)
        snr = dominant_power / (total_power + 1e-10)
        confidence = min(1.0, snr * 10.0)  # Scale SNR to 0-1 range
        
        return period, dominant_freq, confidence
    
    def _fit_sinusoidal_model(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        temperature_col: str,
        period: float,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Fit sinusoidal model to temperature data.
        
        Model: T(t) = baseline + amplitude * sin(2π * t / period + phase_offset)
        
        Args:
            df: DataFrame with time series
            timestamp_col: Timestamp column name
            temperature_col: Temperature column name
            period: Period in hours (from FFT analysis)
            
        Returns:
            Tuple of (amplitude, phase_in_hours, baseline) or None
        """
        # Convert timestamps to hours since start
        df = df.copy()
        start_time = df[timestamp_col].min()
        df['hours_since_start'] = (df[timestamp_col] - start_time).dt.total_seconds() / 3600
        
        # Extract data
        t = df['hours_since_start'].values
        temp = df[temperature_col].values
        
        if len(t) < 10:
            return None
        
        # Define sinusoidal function
        def sinusoid(t, amplitude, phase_offset, baseline):
            return baseline + amplitude * np.sin(2 * np.pi * t / period + phase_offset)
        
        # Initial parameter guess
        baseline_guess = np.mean(temp)
        amplitude_guess = (np.max(temp) - np.min(temp)) / 2.0
        phase_guess = 0.0
        
        try:
            # Fit the model
            params, _ = curve_fit(
                sinusoid,
                t,
                temp,
                p0=[amplitude_guess, phase_guess, baseline_guess],
                maxfev=5000,
            )
            
            amplitude, phase_offset, baseline = params
            
            # Convert phase offset to peak time in hours (0-24)
            # phase_offset is in radians, need to convert to hours
            phase_hours = (-phase_offset * period / (2 * np.pi)) % period
            
            # Normalize to 0-24 hour range
            if period > 20 and period < 28:  # Likely 24-hour cycle
                phase_hours = phase_hours % 24.0
            
            # Take absolute value of amplitude (can be negative from fit)
            amplitude = abs(amplitude)
            
            return amplitude, phase_hours, baseline
            
        except Exception as e:
            logger.warning(f"Failed to fit sinusoidal model: {e}")
            return None
    
    def calculate_rhythm_health(
        self,
        df: Optional[pd.DataFrame] = None,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
    ) -> Optional[RhythmHealthMetrics]:
        """
        Calculate rhythm health score and detect rhythm loss.
        
        Checks multiple criteria:
        - Flattened rhythm (amplitude < threshold)
        - Phase shifts (peak time drift > threshold)
        - Irregular patterns (poor fit to sinusoidal curve)
        - Amplitude stability over time
        
        Args:
            df: Optional DataFrame with recent temperature data
            timestamp_col: Timestamp column name
            temperature_col: Temperature column name
            
        Returns:
            RhythmHealthMetrics object or None if no rhythm data
        """
        if self.current_rhythm is None:
            logger.warning("No current rhythm data available for health calculation")
            return None
        
        rhythm_loss_reasons = []
        amplitude_stable = True
        phase_stable = True
        
        # Check amplitude
        if self.current_rhythm.amplitude < self.min_amplitude_threshold:
            rhythm_loss_reasons.append(
                f"Flattened rhythm: amplitude {self.current_rhythm.amplitude:.2f}°C "
                f"< {self.min_amplitude_threshold}°C"
            )
            amplitude_stable = False
        
        # Check phase stability if we have previous rhythm
        if self.previous_rhythm is not None:
            phase_diff = abs(self.current_rhythm.phase - self.previous_rhythm.phase)
            # Handle wrap-around (e.g., 23h to 1h)
            if phase_diff > 12.0:
                phase_diff = 24.0 - phase_diff
            
            if phase_diff > self.max_phase_drift_hours:
                rhythm_loss_reasons.append(
                    f"Phase shift: {phase_diff:.1f}h drift > {self.max_phase_drift_hours}h threshold"
                )
                phase_stable = False
        
        # Calculate pattern smoothness from confidence
        pattern_smoothness = self.current_rhythm.confidence
        
        if pattern_smoothness < 0.5:
            rhythm_loss_reasons.append(
                f"Irregular pattern: smoothness {pattern_smoothness:.2f} < 0.5"
            )
        
        # Check for erratic variations if recent data provided
        if df is not None and len(df) > 0:
            # Calculate residuals from expected pattern
            residuals = self._calculate_pattern_residuals(
                df, timestamp_col, temperature_col
            )
            if residuals is not None:
                residual_std = np.std(residuals)
                if residual_std > 0.5:  # High variation from expected pattern
                    rhythm_loss_reasons.append(
                        f"Erratic variations: std={residual_std:.2f}°C > 0.5°C"
                    )
        
        # Calculate overall health score (0-100)
        health_score = 100.0
        
        # Amplitude component (0-40 points)
        amplitude_score = min(40.0, (self.current_rhythm.amplitude / self.expected_amplitude) * 40.0)
        health_score = amplitude_score
        
        # Phase stability component (0-20 points)
        if phase_stable:
            health_score += 20.0
        
        # Pattern smoothness component (0-20 points)
        health_score += pattern_smoothness * 20.0
        
        # Confidence component (0-20 points)
        health_score += self.current_rhythm.confidence * 20.0
        
        # Penalties for rhythm loss
        if len(rhythm_loss_reasons) > 0:
            health_score *= 0.5  # 50% penalty for any rhythm issues
        
        health_score = max(0.0, min(100.0, health_score))
        
        # Determine if rhythm is lost
        is_rhythm_lost = len(rhythm_loss_reasons) > 0
        
        # Calculate days of data
        days_of_data = len(self.rhythm_history) if self.rhythm_history else 0
        
        metrics = RhythmHealthMetrics(
            health_score=health_score,
            is_rhythm_lost=is_rhythm_lost,
            amplitude_stable=amplitude_stable,
            phase_stable=phase_stable,
            pattern_smoothness=pattern_smoothness,
            days_of_data=days_of_data,
            rhythm_loss_reasons=rhythm_loss_reasons,
        )
        
        logger.info(
            f"Rhythm health calculated: score={health_score:.1f}, "
            f"rhythm_lost={is_rhythm_lost}, reasons={len(rhythm_loss_reasons)}"
        )
        
        return metrics
    
    def _calculate_pattern_residuals(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        temperature_col: str,
    ) -> Optional[np.ndarray]:
        """
        Calculate residuals between actual and expected circadian pattern.
        
        Args:
            df: DataFrame with temperature data
            timestamp_col: Timestamp column name
            temperature_col: Temperature column name
            
        Returns:
            Array of residuals or None
        """
        if self.current_rhythm is None:
            return None
        
        df = df.copy()
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract hour of day
        df['hour_of_day'] = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60.0
        
        # Calculate expected temperature based on circadian model
        df['expected_temp'] = self._calculate_expected_temperature(df['hour_of_day'].values)
        
        # Calculate residuals
        residuals = df[temperature_col].values - df['expected_temp'].values
        
        return residuals
    
    def _calculate_expected_temperature(self, hours: np.ndarray) -> np.ndarray:
        """
        Calculate expected temperature at given hours based on circadian model.
        
        Args:
            hours: Array of hours (0-24)
            
        Returns:
            Array of expected temperatures
        """
        if self.current_rhythm is None:
            return np.zeros_like(hours)
        
        # Sinusoidal model: T(h) = baseline + amplitude * sin(2π * (h - phase) / 24)
        # Phase represents peak time, so we shift accordingly
        phase_radians = 2 * np.pi * self.current_rhythm.phase / 24.0
        expected = self.current_rhythm.baseline + self.current_rhythm.amplitude * \
                   np.sin(2 * np.pi * hours / 24.0 - phase_radians + np.pi / 2)
        
        return expected
    
    def generate_visualization_data(
        self,
        num_points: int = 24,
        confidence_interval: float = 0.2,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate visualization-ready data for dashboard overlay.
        
        Creates 24-hour circadian curve with confidence bands and metadata.
        
        Args:
            num_points: Number of hourly points (default: 24)
            confidence_interval: Width of confidence band in °C (default: 0.2)
            
        Returns:
            Dictionary with visualization data or None if no rhythm
        """
        if self.current_rhythm is None:
            logger.warning("No rhythm data available for visualization")
            return None
        
        # Generate hourly points (0-23)
        hours = np.linspace(0, 23, num_points)
        
        # Calculate expected temperatures
        expected_temps = self._calculate_expected_temperature(hours)
        
        # Calculate confidence bands
        upper_band = expected_temps + confidence_interval
        lower_band = expected_temps - confidence_interval
        
        # Create visualization structure
        viz_data = {
            'hourly_values': [
                {
                    'hour': float(h),
                    'expected_temperature': float(temp),
                    'upper_confidence': float(upper),
                    'lower_confidence': float(lower),
                }
                for h, temp, upper, lower in zip(hours, expected_temps, upper_band, lower_band)
            ],
            'rhythm_parameters': self.current_rhythm.to_dict(),
            'metadata': {
                'num_points': num_points,
                'confidence_interval_celsius': confidence_interval,
                'peak_time_hour': self.current_rhythm.phase,
                'trough_time_hour': self.current_rhythm.trough_time,
                'amplitude_celsius': self.current_rhythm.amplitude,
                'baseline_celsius': self.current_rhythm.baseline,
            },
        }
        
        logger.info(f"Generated visualization data with {num_points} points")
        
        return viz_data
    
    def get_current_position(
        self,
        current_time: datetime,
        current_temperature: float,
    ) -> Dict[str, Any]:
        """
        Get current temperature position relative to circadian curve.
        
        Args:
            current_time: Current timestamp
            current_temperature: Current temperature value
            
        Returns:
            Dictionary with position information
        """
        if self.current_rhythm is None:
            return {
                'current_temperature': current_temperature,
                'expected_temperature': None,
                'deviation': None,
                'status': 'no_rhythm_data',
            }
        
        # Extract hour of day
        hour_of_day = current_time.hour + current_time.minute / 60.0
        
        # Calculate expected temperature
        expected_temp = self._calculate_expected_temperature(np.array([hour_of_day]))[0]
        
        # Calculate deviation
        deviation = current_temperature - expected_temp
        
        # Determine status
        if abs(deviation) <= 0.5:
            status = 'normal'
        elif deviation > 0.5:
            status = 'above_expected'
        else:
            status = 'below_expected'
        
        return {
            'current_temperature': current_temperature,
            'expected_temperature': expected_temp,
            'deviation': deviation,
            'status': status,
            'hour_of_day': hour_of_day,
        }
    
    def update_with_new_data(
        self,
        new_df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
    ) -> bool:
        """
        Update rhythm model incrementally with new data.
        
        Uses rolling window approach to incorporate new measurements
        while maintaining recent history.
        
        Args:
            new_df: DataFrame with new temperature measurements
            timestamp_col: Timestamp column name
            temperature_col: Temperature column name
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Extract rhythm from new data
            new_rhythm = self.extract_circadian_rhythm(
                new_df, timestamp_col, temperature_col
            )
            
            if new_rhythm is not None:
                logger.info("Successfully updated rhythm model with new data")
                return True
            else:
                logger.warning("Failed to extract rhythm from new data")
                return False
                
        except Exception as e:
            logger.error(f"Error updating rhythm model: {e}")
            return False
    
    def get_rhythm_history(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get historical rhythm parameters for trend analysis.
        
        Args:
            days: Number of days of history to return
            
        Returns:
            List of rhythm parameter dictionaries
        """
        recent_history = self.rhythm_history[-days:] if days < len(self.rhythm_history) else self.rhythm_history
        return [rhythm.to_dict() for rhythm in recent_history]
    
    def detect_rhythm_loss_over_period(
        self,
        hours: float = 48.0,
    ) -> bool:
        """
        Detect if rhythm has been lost for specified period.
        
        Args:
            hours: Number of hours to check (default: 48)
            
        Returns:
            True if rhythm lost for entire period
        """
        if len(self.rhythm_history) < 2:
            return False
        
        # Check recent history
        days_to_check = int(np.ceil(hours / 24.0))
        recent = self.rhythm_history[-days_to_check:]
        
        # Check if all recent rhythms have low amplitude
        low_amplitude_count = sum(
            1 for r in recent if r.amplitude < self.min_amplitude_threshold
        )
        
        return low_amplitude_count == len(recent)
