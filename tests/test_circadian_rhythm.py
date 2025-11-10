"""
Unit Tests for Circadian Rhythm Analysis Module

Tests cover:
- Fourier Transform-based periodicity detection
- 24-hour pattern extraction (amplitude, phase, baseline)
- Rhythm loss detection (flattened rhythm, phase shifts, irregular patterns)
- Rhythm health score calculation
- Visualization data generation
- Edge cases: insufficient data, missing hours, irregular sampling
- Historical rhythm tracking and updates
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.layer2_physiological.circadian_rhythm import (
    CircadianRhythmAnalyzer,
    CircadianParameters,
    RhythmHealthMetrics,
)


class TestCircadianRhythmAnalyzer(unittest.TestCase):
    """Test CircadianRhythmAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CircadianRhythmAnalyzer(
            min_days=3.0,
            expected_amplitude=0.5,
            min_amplitude_threshold=0.3,
        )
    
    def _generate_synthetic_temperature_data(
        self,
        days: int = 3,
        baseline: float = 38.5,
        amplitude: float = 0.5,
        peak_hour: float = 16.0,
        noise_std: float = 0.1,
        sampling_interval_minutes: int = 60,
    ) -> pd.DataFrame:
        """
        Generate synthetic temperature data with circadian rhythm.
        
        Args:
            days: Number of days to generate
            baseline: Mean temperature (°C)
            amplitude: Circadian amplitude (°C)
            peak_hour: Time of peak temperature (hour, 0-24)
            noise_std: Standard deviation of noise (°C)
            sampling_interval_minutes: Sampling interval
            
        Returns:
            DataFrame with timestamp and temperature columns
        """
        # Generate timestamps
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        num_samples = int(days * 24 * 60 / sampling_interval_minutes)
        timestamps = [start_time + timedelta(minutes=i * sampling_interval_minutes) 
                     for i in range(num_samples)]
        
        # Generate circadian pattern
        hours = np.array([(ts.hour + ts.minute / 60.0) for ts in timestamps])
        
        # Sinusoidal pattern: T(h) = baseline + amplitude * sin(2π * (h - peak_hour + 6) / 24)
        # +6 shifts so peak occurs at peak_hour
        phase_radians = 2 * np.pi * peak_hour / 24.0
        temperatures = baseline + amplitude * np.sin(2 * np.pi * hours / 24.0 - phase_radians + np.pi / 2)
        
        # Add noise
        if noise_std > 0:
            temperatures += np.random.normal(0, noise_std, len(temperatures))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
        })
        
        return df
    
    def test_extract_rhythm_normal_3_days(self):
        """Test extracting circadian rhythm from 3 days of normal data."""
        # Generate 3 days of data with clear circadian pattern
        df = self._generate_synthetic_temperature_data(
            days=3,
            baseline=38.5,
            amplitude=0.5,
            peak_hour=16.0,
            noise_std=0.05,
        )
        
        # Extract rhythm
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        # Verify extraction succeeded
        self.assertIsNotNone(rhythm)
        self.assertIsInstance(rhythm, CircadianParameters)
        
        # Check parameters are reasonable
        self.assertAlmostEqual(rhythm.baseline, 38.5, delta=0.2)
        self.assertAlmostEqual(rhythm.amplitude, 0.5, delta=0.2)
        self.assertAlmostEqual(rhythm.phase, 16.0, delta=2.0)  # Within 2 hours
        self.assertAlmostEqual(rhythm.period, 24.0, delta=2.0)
        self.assertGreater(rhythm.confidence, 0.3)
    
    def test_extract_rhythm_7_days(self):
        """Test with 7 days of data (better accuracy)."""
        df = self._generate_synthetic_temperature_data(
            days=7,
            baseline=38.7,
            amplitude=0.4,
            peak_hour=15.0,
            noise_std=0.05,
        )
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        self.assertIsNotNone(rhythm)
        self.assertAlmostEqual(rhythm.baseline, 38.7, delta=0.15)
        self.assertAlmostEqual(rhythm.amplitude, 0.4, delta=0.15)
        self.assertAlmostEqual(rhythm.phase, 15.0, delta=1.5)
    
    def test_insufficient_data_2_days(self):
        """Test with insufficient data (< 3 days)."""
        df = self._generate_synthetic_temperature_data(days=2)
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        # Should return None due to insufficient data
        self.assertIsNone(rhythm)
    
    def test_extract_rhythm_with_missing_hours(self):
        """Test with data containing missing hours."""
        # Generate 4 days of data
        df = self._generate_synthetic_temperature_data(days=4, sampling_interval_minutes=60)
        
        # Remove random hours (simulate missing data)
        # Remove about 10% of data
        indices_to_keep = np.random.choice(len(df), size=int(len(df) * 0.9), replace=False)
        df = df.iloc[sorted(indices_to_keep)].reset_index(drop=True)
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        # Should still work with some missing data
        self.assertIsNotNone(rhythm)
        self.assertGreater(rhythm.confidence, 0.2)
    
    def test_rhythm_loss_low_amplitude(self):
        """Test detection of rhythm loss due to low amplitude."""
        # Generate data with very low amplitude (flattened rhythm)
        df = self._generate_synthetic_temperature_data(
            days=3,
            baseline=38.5,
            amplitude=0.2,  # Below threshold of 0.3
            peak_hour=16.0,
            noise_std=0.1,
        )
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        health = self.analyzer.calculate_rhythm_health(df)
        
        self.assertIsNotNone(rhythm)
        self.assertIsNotNone(health)
        self.assertTrue(health.is_rhythm_lost)
        self.assertFalse(health.amplitude_stable)
        self.assertIn('Flattened rhythm', ' '.join(health.rhythm_loss_reasons))
    
    def test_phase_shift_detection(self):
        """Test detection of phase shift between periods."""
        # First period: peak at 16:00
        df1 = self._generate_synthetic_temperature_data(
            days=3,
            baseline=38.5,
            amplitude=0.5,
            peak_hour=16.0,
        )
        
        rhythm1 = self.analyzer.extract_circadian_rhythm(df1)
        self.assertIsNotNone(rhythm1)
        
        # Second period: peak shifted to 20:00 (4-hour shift)
        df2 = self._generate_synthetic_temperature_data(
            days=3,
            baseline=38.5,
            amplitude=0.5,
            peak_hour=20.0,
        )
        
        rhythm2 = self.analyzer.extract_circadian_rhythm(df2)
        health = self.analyzer.calculate_rhythm_health(df2)
        
        self.assertIsNotNone(rhythm2)
        self.assertIsNotNone(health)
        
        # Should detect phase shift
        self.assertFalse(health.phase_stable)
        # Phase shift should be detected in reasons
        phase_shift_detected = any('Phase shift' in reason for reason in health.rhythm_loss_reasons)
        self.assertTrue(phase_shift_detected)
    
    def test_rhythm_health_score_perfect_rhythm(self):
        """Test health score calculation for perfect circadian rhythm."""
        # Generate ideal circadian data
        df = self._generate_synthetic_temperature_data(
            days=5,
            baseline=38.5,
            amplitude=0.5,
            peak_hour=16.0,
            noise_std=0.02,  # Very low noise
        )
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        health = self.analyzer.calculate_rhythm_health(df)
        
        self.assertIsNotNone(health)
        self.assertGreater(health.health_score, 70.0)  # Should be high score
        self.assertFalse(health.is_rhythm_lost)
        self.assertTrue(health.amplitude_stable)
    
    def test_rhythm_health_score_poor_rhythm(self):
        """Test health score calculation for poor rhythm."""
        # Generate poor rhythm: low amplitude, high noise
        df = self._generate_synthetic_temperature_data(
            days=3,
            baseline=38.5,
            amplitude=0.15,  # Very low
            peak_hour=16.0,
            noise_std=0.3,  # High noise
        )
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        health = self.analyzer.calculate_rhythm_health(df)
        
        self.assertIsNotNone(health)
        self.assertLess(health.health_score, 50.0)  # Should be low score
        self.assertTrue(health.is_rhythm_lost)
    
    def test_visualization_data_generation(self):
        """Test generation of visualization data."""
        # Generate data and extract rhythm
        df = self._generate_synthetic_temperature_data(
            days=4,
            baseline=38.5,
            amplitude=0.5,
            peak_hour=16.0,
        )
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        self.assertIsNotNone(rhythm)
        
        # Generate visualization data
        viz_data = self.analyzer.generate_visualization_data(num_points=24)
        
        self.assertIsNotNone(viz_data)
        self.assertIn('hourly_values', viz_data)
        self.assertIn('rhythm_parameters', viz_data)
        self.assertIn('metadata', viz_data)
        
        # Check hourly values
        hourly_values = viz_data['hourly_values']
        self.assertEqual(len(hourly_values), 24)
        
        # Each hourly value should have required fields
        for hv in hourly_values:
            self.assertIn('hour', hv)
            self.assertIn('expected_temperature', hv)
            self.assertIn('upper_confidence', hv)
            self.assertIn('lower_confidence', hv)
        
        # Check metadata
        metadata = viz_data['metadata']
        self.assertEqual(metadata['num_points'], 24)
        self.assertAlmostEqual(metadata['peak_time_hour'], 16.0, delta=2.0)
    
    def test_current_position_calculation(self):
        """Test calculation of current position relative to circadian curve."""
        # Generate and extract rhythm
        df = self._generate_synthetic_temperature_data(days=3, peak_hour=16.0)
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        self.assertIsNotNone(rhythm)
        
        # Test position at peak time (16:00)
        current_time = datetime(2024, 1, 1, 16, 0, 0)
        current_temp = 39.0  # At peak
        
        position = self.analyzer.get_current_position(current_time, current_temp)
        
        self.assertIn('current_temperature', position)
        self.assertIn('expected_temperature', position)
        self.assertIn('deviation', position)
        self.assertIn('status', position)
        self.assertEqual(position['current_temperature'], current_temp)
    
    def test_update_with_new_data(self):
        """Test incremental update with new data."""
        # Initial data
        df1 = self._generate_synthetic_temperature_data(days=3, peak_hour=16.0)
        rhythm1 = self.analyzer.extract_circadian_rhythm(df1)
        self.assertIsNotNone(rhythm1)
        
        # New data (next day)
        start_time = df1['timestamp'].max() + timedelta(hours=1)
        df2 = self._generate_synthetic_temperature_data(days=1, peak_hour=16.0)
        df2['timestamp'] = df2['timestamp'].apply(lambda x: start_time + (x - df2['timestamp'].min()))
        
        # Update with new data
        success = self.analyzer.update_with_new_data(
            pd.concat([df1, df2]).reset_index(drop=True)
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(self.analyzer.current_rhythm)
    
    def test_rhythm_history_tracking(self):
        """Test tracking of rhythm history."""
        # Generate multiple periods and track history
        for i in range(5):
            df = self._generate_synthetic_temperature_data(days=3, peak_hour=16.0)
            self.analyzer.extract_circadian_rhythm(df)
        
        # Check history
        history = self.analyzer.get_rhythm_history(days=5)
        self.assertEqual(len(history), 5)
        
        # Each entry should be a dictionary
        for entry in history:
            self.assertIsInstance(entry, dict)
            self.assertIn('amplitude', entry)
            self.assertIn('phase', entry)
            self.assertIn('baseline', entry)
    
    def test_detect_rhythm_loss_over_period(self):
        """Test detection of sustained rhythm loss."""
        # Generate multiple days with low amplitude
        for i in range(3):
            df = self._generate_synthetic_temperature_data(
                days=1,
                amplitude=0.2,  # Below threshold
            )
            self.analyzer.extract_circadian_rhythm(df)
        
        # Check if rhythm loss detected over period
        rhythm_lost = self.analyzer.detect_rhythm_loss_over_period(hours=48.0)
        
        self.assertTrue(rhythm_lost)
    
    def test_irregular_sampling_intervals(self):
        """Test handling of irregular sampling intervals."""
        # Generate data with irregular intervals
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = []
        temperatures = []
        
        current_time = start_time
        for day in range(4):
            for hour in range(24):
                # Irregular intervals: 30-90 minutes
                interval = np.random.randint(30, 90)
                current_time += timedelta(minutes=interval)
                timestamps.append(current_time)
                
                # Generate temperature with circadian pattern
                hour_of_day = current_time.hour + current_time.minute / 60.0
                temp = 38.5 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 16.0 + 6) / 24.0)
                temperatures.append(temp + np.random.normal(0, 0.05))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
        })
        
        # Should still extract rhythm despite irregular sampling
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        self.assertIsNotNone(rhythm)
        self.assertGreater(rhythm.confidence, 0.2)
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['timestamp', 'temperature'])
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        self.assertIsNone(rhythm)
    
    def test_nan_values_handling(self):
        """Test handling of NaN values in temperature data."""
        df = self._generate_synthetic_temperature_data(days=4)
        
        # Introduce some NaN values
        nan_indices = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
        df.loc[nan_indices, 'temperature'] = np.nan
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        # Should still work after removing NaN values
        self.assertIsNotNone(rhythm)
    
    def test_extreme_temperature_values(self):
        """Test with extreme temperature values."""
        # Generate data with fever (high temperature, reduced amplitude)
        df = self._generate_synthetic_temperature_data(
            days=3,
            baseline=40.0,  # Fever temperature
            amplitude=0.2,  # Reduced amplitude
            peak_hour=16.0,
        )
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        health = self.analyzer.calculate_rhythm_health(df)
        
        self.assertIsNotNone(rhythm)
        self.assertAlmostEqual(rhythm.baseline, 40.0, delta=0.3)
        
        # Should detect rhythm issues due to low amplitude
        self.assertIsNotNone(health)
        self.assertTrue(health.is_rhythm_lost)
    
    def test_no_circadian_pattern(self):
        """Test with data that has no circadian pattern (constant temperature)."""
        # Generate constant temperature data
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        num_samples = 3 * 24  # 3 days, hourly
        timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
        temperatures = [38.5 + np.random.normal(0, 0.1) for _ in range(num_samples)]
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
        })
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        # May extract rhythm but with very low amplitude and confidence
        if rhythm is not None:
            self.assertLess(rhythm.amplitude, 0.2)
            self.assertLess(rhythm.confidence, 0.5)
    
    def test_visualization_without_rhythm(self):
        """Test visualization data generation without extracted rhythm."""
        # Don't extract rhythm first
        viz_data = self.analyzer.generate_visualization_data()
        
        # Should return None
        self.assertIsNone(viz_data)
    
    def test_health_calculation_without_rhythm(self):
        """Test health calculation without extracted rhythm."""
        health = self.analyzer.calculate_rhythm_health()
        
        # Should return None
        self.assertIsNone(health)
    
    def test_multiple_peaks_in_fft(self):
        """Test FFT with multiple frequency components."""
        # Generate data with 24h circadian + 12h harmonic
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        num_samples = 4 * 24  # 4 days, hourly
        timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
        
        temperatures = []
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            # 24-hour component (dominant)
            temp_24h = 0.5 * np.sin(2 * np.pi * hour / 24.0)
            # 12-hour component (weaker)
            temp_12h = 0.1 * np.sin(2 * np.pi * hour / 12.0)
            temp = 38.5 + temp_24h + temp_12h + np.random.normal(0, 0.05)
            temperatures.append(temp)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
        })
        
        rhythm = self.analyzer.extract_circadian_rhythm(df)
        
        # Should still detect 24-hour rhythm as dominant
        self.assertIsNotNone(rhythm)
        self.assertAlmostEqual(rhythm.period, 24.0, delta=2.0)
    
    def test_circadian_parameters_to_dict(self):
        """Test CircadianParameters to_dict method."""
        params = CircadianParameters(
            amplitude=0.5,
            phase=16.0,
            baseline=38.5,
            period=24.0,
            trough_time=4.0,
            confidence=0.85,
            last_updated=datetime.now(),
        )
        
        params_dict = params.to_dict()
        
        self.assertIsInstance(params_dict, dict)
        self.assertEqual(params_dict['amplitude'], 0.5)
        self.assertEqual(params_dict['phase'], 16.0)
        self.assertEqual(params_dict['baseline'], 38.5)
        self.assertIn('last_updated', params_dict)
    
    def test_rhythm_health_metrics_to_dict(self):
        """Test RhythmHealthMetrics to_dict method."""
        metrics = RhythmHealthMetrics(
            health_score=85.0,
            is_rhythm_lost=False,
            amplitude_stable=True,
            phase_stable=True,
            pattern_smoothness=0.9,
            days_of_data=5.0,
            rhythm_loss_reasons=[],
        )
        
        metrics_dict = metrics.to_dict()
        
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['health_score'], 85.0)
        self.assertFalse(metrics_dict['is_rhythm_lost'])
        self.assertTrue(metrics_dict['amplitude_stable'])


class TestCircadianEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_exactly_3_days_of_data(self):
        """Test with exactly 3 days (minimum requirement)."""
        analyzer = CircadianRhythmAnalyzer(min_days=3.0)
        
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        num_samples = 3 * 24  # Exactly 3 days, hourly
        timestamps = [start_time + timedelta(hours=i) for i in range(num_samples)]
        
        temperatures = []
        for ts in timestamps:
            hour = ts.hour + ts.minute / 60.0
            temp = 38.5 + 0.5 * np.sin(2 * np.pi * (hour - 16.0 + 6) / 24.0)
            temperatures.append(temp + np.random.normal(0, 0.05))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
        })
        
        rhythm = analyzer.extract_circadian_rhythm(df)
        
        self.assertIsNotNone(rhythm)
    
    def test_maximum_missing_hours_per_day(self):
        """Test with exactly max missing hours per day."""
        analyzer = CircadianRhythmAnalyzer(max_missing_hours_per_day=3)
        
        # Generate 4 days of data
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        timestamps = []
        temperatures = []
        
        for day in range(4):
            # 21 hours per day (3 missing)
            for hour in range(21):
                ts = start_time + timedelta(days=day, hours=hour)
                timestamps.append(ts)
                hour_of_day = ts.hour + ts.minute / 60.0
                temp = 38.5 + 0.5 * np.sin(2 * np.pi * (hour_of_day - 16.0 + 6) / 24.0)
                temperatures.append(temp + np.random.normal(0, 0.05))
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperatures,
        })
        
        rhythm = analyzer.extract_circadian_rhythm(df)
        
        # Should still work at the boundary
        self.assertIsNotNone(rhythm)
    
    def test_amplitude_at_threshold(self):
        """Test with amplitude exactly at loss threshold."""
        analyzer = CircadianRhythmAnalyzer(min_amplitude_threshold=0.3)
        
        # Generate data with amplitude exactly at threshold
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=72, freq='1H'),
            'temperature': 38.5 + 0.3 * np.sin(
                2 * np.pi * np.arange(72) / 24.0
            ),
        })
        
        rhythm = analyzer.extract_circadian_rhythm(df)
        health = analyzer.calculate_rhythm_health(df)
        
        self.assertIsNotNone(rhythm)
        self.assertIsNotNone(health)
        
        # At exact threshold, should not trigger loss
        # (threshold is <, not <=)
        if rhythm.amplitude >= 0.3:
            self.assertTrue(health.amplitude_stable)


if __name__ == '__main__':
    unittest.main()
