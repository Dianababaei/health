"""
Unit tests for baseline temperature calculation.

Tests circadian extraction, baseline calculation, drift detection,
and dynamic updates with synthetic data.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physiological.circadian_extractor import CircadianExtractor, CircadianProfile
from src.physiological.baseline_calculator import BaselineCalculator, BaselineResult
from src.physiological.baseline_updater import (
    BaselineUpdater,
    BaselineDriftDetector,
    BaselineHistoryManager,
)


# ============================================================
# Test Data Generators
# ============================================================

def generate_synthetic_temperature_data(
    baseline_temp: float = 38.5,
    circadian_amplitude: float = 0.4,
    peak_hour: float = 16.0,
    n_days: int = 7,
    samples_per_hour: int = 60,
    noise_std: float = 0.1,
    add_fever_spike: bool = False,
    fever_start_hour: float = 120.0,
    fever_duration_hours: float = 12.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic temperature data with known circadian pattern.
    
    Args:
        baseline_temp: True baseline temperature (°C)
        circadian_amplitude: Circadian variation amplitude (°C)
        peak_hour: Hour of peak temperature (0-23)
        n_days: Number of days to generate
        samples_per_hour: Samples per hour (default 60 = minute-level)
        noise_std: Noise standard deviation (°C)
        add_fever_spike: Whether to add fever spike
        fever_start_hour: Hour to start fever spike
        fever_duration_hours: Duration of fever spike
        seed: Random seed
        
    Returns:
        DataFrame with columns [timestamp, temperature, cow_id]
    """
    np.random.seed(seed)
    
    # Generate timestamps
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    total_samples = n_days * 24 * samples_per_hour
    timestamps = [
        start_time + timedelta(minutes=i / samples_per_hour * 60)
        for i in range(total_samples)
    ]
    
    # Calculate circadian component for each timestamp
    temperatures = []
    
    for ts in timestamps:
        hour_of_day = ts.hour + ts.minute / 60.0
        
        # Circadian component (sinusoidal with peak at peak_hour)
        phase = 2 * np.pi * (hour_of_day - peak_hour) / 24.0
        circadian = -circadian_amplitude * np.cos(phase)
        
        # Base temperature
        temp = baseline_temp + circadian
        
        # Add noise
        temp += np.random.normal(0, noise_std)
        
        temperatures.append(temp)
    
    # Add fever spike if requested
    if add_fever_spike:
        for i, ts in enumerate(timestamps):
            hours_since_start = (ts - start_time).total_seconds() / 3600
            
            if fever_start_hour <= hours_since_start < fever_start_hour + fever_duration_hours:
                # Add fever component
                fever_position = (hours_since_start - fever_start_hour) / fever_duration_hours
                # Trapezoidal fever spike
                if fever_position < 0.2:
                    fever_magnitude = 1.5 * (fever_position / 0.2)
                elif fever_position > 0.8:
                    fever_magnitude = 1.5 * ((1 - fever_position) / 0.2)
                else:
                    fever_magnitude = 1.5
                
                temperatures[i] += fever_magnitude
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'cow_id': 1,
    })
    
    return df


def generate_drifting_baseline_data(
    initial_baseline: float = 38.5,
    drift_rate: float = 0.1,  # °C per day
    n_days: int = 14,
    samples_per_hour: int = 60,
    circadian_amplitude: float = 0.4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate data with gradual baseline drift.
    
    Args:
        initial_baseline: Starting baseline (°C)
        drift_rate: Rate of baseline change (°C/day)
        n_days: Number of days
        samples_per_hour: Samples per hour
        circadian_amplitude: Circadian amplitude
        seed: Random seed
        
    Returns:
        DataFrame with drifting baseline
    """
    np.random.seed(seed)
    
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    total_samples = n_days * 24 * samples_per_hour
    timestamps = [
        start_time + timedelta(minutes=i / samples_per_hour * 60)
        for i in range(total_samples)
    ]
    
    temperatures = []
    
    for ts in timestamps:
        # Calculate days since start
        days_elapsed = (ts - start_time).total_seconds() / 86400
        
        # Drifting baseline
        baseline = initial_baseline + drift_rate * days_elapsed
        
        # Circadian component
        hour_of_day = ts.hour + ts.minute / 60.0
        phase = 2 * np.pi * (hour_of_day - 16.0) / 24.0
        circadian = -circadian_amplitude * np.cos(phase)
        
        # Total temperature
        temp = baseline + circadian + np.random.normal(0, 0.1)
        temperatures.append(temp)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': temperatures,
        'cow_id': 1,
    })
    
    return df


# ============================================================
# Circadian Extractor Tests
# ============================================================

class TestCircadianExtractor:
    """Tests for CircadianExtractor class."""
    
    def test_extract_circadian_profile(self):
        """Test extraction of circadian profile from synthetic data."""
        # Generate data with known circadian pattern
        df = generate_synthetic_temperature_data(
            baseline_temp=38.5,
            circadian_amplitude=0.4,
            peak_hour=16.0,
            n_days=7,
        )
        
        extractor = CircadianExtractor()
        profile = extractor.extract_circadian_profile(df)
        
        # Check profile attributes
        assert isinstance(profile, CircadianProfile)
        assert len(profile.hourly_means) == 24
        assert len(profile.hourly_stds) == 24
        assert len(profile.hourly_counts) == 24
        
        # Check amplitude is close to expected (within 20% tolerance)
        assert abs(profile.amplitude - 0.4) < 0.1, \
            f"Amplitude {profile.amplitude:.3f} not close to expected 0.4°C"
        
        # Check peak hour is close to expected (within 2 hours)
        peak_diff = min(
            abs(profile.peak_hour - 16.0),
            abs(profile.peak_hour - 16.0 + 24),
            abs(profile.peak_hour - 16.0 - 24),
        )
        assert peak_diff < 2.0, \
            f"Peak hour {profile.peak_hour:.1f} not close to expected 16.0"
        
        # Check mean temperature
        assert abs(profile.mean_temp - 38.5) < 0.1, \
            f"Mean temp {profile.mean_temp:.3f} not close to expected 38.5°C"
        
        # Check confidence is reasonable
        assert 0.7 <= profile.confidence <= 1.0, \
            f"Confidence {profile.confidence:.2f} too low"
    
    def test_detrend_temperatures(self):
        """Test temperature detrending."""
        df = generate_synthetic_temperature_data(
            baseline_temp=38.5,
            circadian_amplitude=0.4,
            n_days=7,
        )
        
        extractor = CircadianExtractor()
        profile = extractor.extract_circadian_profile(df)
        detrended_df = extractor.detrend_temperatures(df, profile)
        
        # Check detrended column exists
        assert 'detrended_temp' in detrended_df.columns
        
        # Detrended temperatures should have much lower std than raw
        raw_std = df['temperature'].std()
        detrended_std = detrended_df['detrended_temp'].std()
        
        assert detrended_std < raw_std / 2, \
            "Detrending did not reduce variability sufficiently"
        
        # Detrended mean should be close to 0 (or original mean)
        detrended_mean = detrended_df['detrended_temp'].mean()
        assert abs(detrended_mean) < 0.1, \
            f"Detrended mean {detrended_mean:.3f} not close to 0"
    
    def test_circadian_component_interpolation(self):
        """Test smooth interpolation of circadian component."""
        df = generate_synthetic_temperature_data(n_days=7)
        
        extractor = CircadianExtractor(method="fourier")
        profile = extractor.extract_circadian_profile(df)
        
        # Test component retrieval for various hours
        for hour in [0.0, 6.5, 12.25, 18.75, 23.5]:
            component = profile.get_circadian_component(hour)
            assert -1.0 <= component <= 1.0, \
                f"Component {component:.3f} out of reasonable range at hour {hour}"
    
    def test_validate_circadian_profile(self):
        """Test circadian profile validation."""
        df = generate_synthetic_temperature_data(
            circadian_amplitude=0.4,
            peak_hour=16.0,
            n_days=7,
        )
        
        extractor = CircadianExtractor()
        profile = extractor.extract_circadian_profile(df)
        
        is_valid, warnings = extractor.validate_circadian_profile(profile)
        
        assert is_valid, f"Valid profile flagged as invalid: {warnings}"
        assert len(warnings) <= 1, "Too many warnings for valid profile"


# ============================================================
# Baseline Calculator Tests
# ============================================================

class TestBaselineCalculator:
    """Tests for BaselineCalculator class."""
    
    def test_calculate_baseline_simple(self):
        """Test basic baseline calculation."""
        df = generate_synthetic_temperature_data(
            baseline_temp=38.6,
            n_days=7,
        )
        
        calculator = BaselineCalculator(window_days=7)
        result = calculator.calculate_baseline(df, cow_id=1)
        
        # Check result structure
        assert isinstance(result, BaselineResult)
        assert result.cow_id == 1
        assert result.calculation_window_days == 7
        
        # Check baseline is close to true value (within 0.2°C)
        assert abs(result.baseline_temp - 38.6) < 0.2, \
            f"Baseline {result.baseline_temp:.3f} not close to expected 38.6°C"
        
        # Check confidence
        assert result.confidence_score >= 0.5, \
            f"Confidence {result.confidence_score:.2f} too low"
    
    def test_baseline_stability_normal_conditions(self):
        """Test baseline remains stable during normal conditions."""
        # Generate multiple days of normal data
        df = generate_synthetic_temperature_data(
            baseline_temp=38.5,
            n_days=30,
        )
        
        calculator = BaselineCalculator(window_days=7)
        
        # Calculate baselines at different time points
        baselines = []
        for day in range(7, 30, 3):
            current_time = df['timestamp'].min() + timedelta(days=day)
            result = calculator.calculate_baseline(
                df, cow_id=1, current_time=current_time
            )
            baselines.append(result.baseline_temp)
        
        # Check stability (should be within ±0.2°C)
        baseline_std = np.std(baselines)
        assert baseline_std < 0.2, \
            f"Baseline variability {baseline_std:.3f} exceeds ±0.2°C threshold"
    
    def test_fever_exclusion(self):
        """Test that fever spikes are excluded from baseline calculation."""
        # Generate data with fever spike
        df = generate_synthetic_temperature_data(
            baseline_temp=38.5,
            n_days=7,
            add_fever_spike=True,
            fever_start_hour=72.0,  # Day 3
            fever_duration_hours=24.0,
        )
        
        calculator = BaselineCalculator(
            window_days=7,
            fever_threshold=39.5,
        )
        
        result = calculator.calculate_baseline(df, cow_id=1)
        
        # Baseline should still be close to 38.5 despite fever
        assert abs(result.baseline_temp - 38.5) < 0.3, \
            f"Baseline {result.baseline_temp:.3f} affected by fever (expected ~38.5°C)"
        
        # Check that some readings were excluded
        assert result.outliers_excluded > 0, "No outliers excluded despite fever"
    
    def test_robust_statistics_methods(self):
        """Test different robust statistics methods."""
        df = generate_synthetic_temperature_data(baseline_temp=38.5, n_days=7)
        
        methods = ["median", "trimmed_mean"]
        baselines = {}
        
        for method in methods:
            calculator = BaselineCalculator(
                window_days=7,
                robust_method=method,
            )
            result = calculator.calculate_baseline(df, cow_id=1)
            baselines[method] = result.baseline_temp
        
        # All methods should produce similar results (within 0.15°C)
        baseline_values = list(baselines.values())
        baseline_range = max(baseline_values) - min(baseline_values)
        
        assert baseline_range < 0.15, \
            f"Robust methods produce too different results: {baselines}"
    
    def test_multi_window_calculation(self):
        """Test baseline calculation with multiple window sizes."""
        df = generate_synthetic_temperature_data(baseline_temp=38.5, n_days=30)
        
        calculator = BaselineCalculator()
        results = calculator.calculate_baseline_multi_window(
            df, cow_id=1, window_days_list=[7, 14, 30]
        )
        
        # Check all windows calculated
        assert len(results) == 3
        assert 7 in results
        assert 14 in results
        assert 30 in results
        
        # All baselines should be similar
        baselines = [r.baseline_temp for r in results.values()]
        baseline_range = max(baselines) - min(baselines)
        
        assert baseline_range < 0.3, \
            f"Multi-window baselines vary too much: {baselines}"
    
    def test_validate_baseline(self):
        """Test baseline validation."""
        calculator = BaselineCalculator()
        
        # Valid baseline
        is_valid, warnings = calculator.validate_baseline(38.5)
        assert is_valid, "Valid baseline flagged as invalid"
        
        # Too low
        is_valid, warnings = calculator.validate_baseline(37.0)
        assert not is_valid, "Low baseline not flagged"
        
        # Too high
        is_valid, warnings = calculator.validate_baseline(40.0)
        assert not is_valid, "High baseline not flagged"


# ============================================================
# Drift Detection Tests
# ============================================================

class TestBaselineDriftDetector:
    """Tests for BaselineDriftDetector class."""
    
    def test_detect_drift_positive(self):
        """Test detection of upward baseline drift."""
        # Generate data with drift
        df = generate_drifting_baseline_data(
            initial_baseline=38.5,
            drift_rate=0.1,  # 0.1°C per day = 0.7°C over 7 days
            n_days=14,
        )
        
        detector = BaselineDriftDetector(
            drift_threshold=0.5,
            drift_window_days=7,
        )
        
        # Calculate baselines at different time points
        calculator = BaselineCalculator(window_days=7)
        baselines = []
        
        for day in range(7, 14):
            current_time = df['timestamp'].min() + timedelta(days=day)
            result = calculator.calculate_baseline(
                df, cow_id=1, current_time=current_time
            )
            baselines.append({
                'timestamp': current_time,
                'baseline_temp': result.baseline_temp,
            })
        
        baseline_df = pd.DataFrame(baselines)
        
        # Detect drift
        drift_detected, magnitude, confidence = detector.detect_drift(
            baseline_df, baseline_df['timestamp'].max()
        )
        
        assert drift_detected, "Drift not detected"
        assert magnitude > 0.5, f"Drift magnitude {magnitude:.3f} too small"
        assert confidence > 0.6, f"Drift confidence {confidence:.2f} too low"
    
    def test_no_drift_stable_baseline(self):
        """Test that stable baseline doesn't trigger drift detection."""
        df = generate_synthetic_temperature_data(
            baseline_temp=38.5,
            n_days=14,
        )
        
        detector = BaselineDriftDetector(drift_threshold=0.5)
        calculator = BaselineCalculator(window_days=7)
        
        baselines = []
        for day in range(7, 14):
            current_time = df['timestamp'].min() + timedelta(days=day)
            result = calculator.calculate_baseline(
                df, cow_id=1, current_time=current_time
            )
            baselines.append({
                'timestamp': current_time,
                'baseline_temp': result.baseline_temp,
            })
        
        baseline_df = pd.DataFrame(baselines)
        
        drift_detected, magnitude, confidence = detector.detect_drift(
            baseline_df, baseline_df['timestamp'].max()
        )
        
        assert not drift_detected, "False drift detection on stable baseline"
        assert abs(magnitude) < 0.5, f"Magnitude {magnitude:.3f} exceeds threshold"


# ============================================================
# History Manager Tests
# ============================================================

class TestBaselineHistoryManager:
    """Tests for BaselineHistoryManager class."""
    
    def test_store_and_retrieve_json(self, tmp_path):
        """Test storing and retrieving baseline history (JSON)."""
        manager = BaselineHistoryManager(
            storage_backend="json",
            storage_path=str(tmp_path / "baseline_history")
        )
        
        # Create test result
        result = BaselineResult(
            cow_id=1,
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            baseline_temp=38.5,
            calculation_window_days=7,
            samples_used=10000,
            outliers_excluded=50,
            circadian_amplitude=0.4,
            circadian_confidence=0.85,
            confidence_score=0.9,
            method="trimmed_mean",
        )
        
        # Store
        manager.store_baseline(result)
        
        # Retrieve
        history = manager.retrieve_history(cow_id=1)
        
        assert len(history) == 1
        assert history.iloc[0]['cow_id'] == 1
        assert abs(history.iloc[0]['baseline_temp'] - 38.5) < 0.001
    
    def test_retrieve_time_range(self, tmp_path):
        """Test retrieving history within time range."""
        manager = BaselineHistoryManager(
            storage_backend="json",
            storage_path=str(tmp_path / "baseline_history")
        )
        
        # Store multiple results
        for day in range(1, 8):
            result = BaselineResult(
                cow_id=1,
                timestamp=datetime(2025, 1, day, 12, 0, 0),
                baseline_temp=38.5 + day * 0.01,
                calculation_window_days=7,
                samples_used=10000,
                outliers_excluded=50,
                circadian_amplitude=0.4,
                circadian_confidence=0.85,
                confidence_score=0.9,
                method="trimmed_mean",
            )
            manager.store_baseline(result)
        
        # Retrieve specific range
        history = manager.retrieve_history(
            cow_id=1,
            start_time=datetime(2025, 1, 3, 0, 0, 0),
            end_time=datetime(2025, 1, 5, 23, 59, 59),
        )
        
        assert len(history) == 3  # Days 3, 4, 5


# ============================================================
# Baseline Updater Tests
# ============================================================

class TestBaselineUpdater:
    """Tests for BaselineUpdater class."""
    
    def test_update_baseline(self, tmp_path):
        """Test baseline update."""
        history_manager = BaselineHistoryManager(
            storage_backend="json",
            storage_path=str(tmp_path / "baseline_history")
        )
        
        updater = BaselineUpdater(
            history_manager=history_manager,
            update_frequency_hours=24,
        )
        
        df = generate_synthetic_temperature_data(baseline_temp=38.5, n_days=7)
        
        # First update
        result = updater.update_baseline(df, cow_id=1, force_update=True)
        
        assert result is not None
        assert abs(result.baseline_temp - 38.5) < 0.2
    
    def test_adaptive_windowing(self, tmp_path):
        """Test adaptive window sizing."""
        history_manager = BaselineHistoryManager(
            storage_backend="json",
            storage_path=str(tmp_path / "baseline_history")
        )
        
        updater = BaselineUpdater(
            history_manager=history_manager,
            adaptive_windowing=True,
            initial_window_days=7,
            expand_after_days=14,
        )
        
        # Generate 30 days of data
        df = generate_synthetic_temperature_data(baseline_temp=38.5, n_days=30)
        
        # Update at day 7 (should use 7-day window)
        current_time = df['timestamp'].min() + timedelta(days=7)
        result_7d = updater.update_baseline(
            df, cow_id=1, current_time=current_time, force_update=True
        )
        assert result_7d.calculation_window_days == 7
        
        # Update at day 20 (should use longer window)
        current_time = df['timestamp'].min() + timedelta(days=20)
        result_20d = updater.update_baseline(
            df, cow_id=1, current_time=current_time, force_update=True
        )
        assert result_20d.calculation_window_days > 7
    
    def test_smoothing(self, tmp_path):
        """Test exponential smoothing of baseline updates."""
        history_manager = BaselineHistoryManager(
            storage_backend="json",
            storage_path=str(tmp_path / "baseline_history")
        )
        
        updater = BaselineUpdater(
            history_manager=history_manager,
            smoothing_alpha=0.3,
        )
        
        df = generate_synthetic_temperature_data(baseline_temp=38.5, n_days=14)
        
        # First update
        result1 = updater.update_baseline(df, cow_id=1, force_update=True)
        baseline1 = result1.baseline_temp
        
        # Second update (should be smoothed)
        current_time = df['timestamp'].max()
        result2 = updater.update_baseline(
            df, cow_id=1, current_time=current_time, force_update=True
        )
        baseline2 = result2.baseline_temp
        
        # Smoothing should keep values close
        assert abs(baseline2 - baseline1) < 0.1


# ============================================================
# Performance Tests
# ============================================================

class TestPerformance:
    """Performance tests for baseline calculation."""
    
    def test_calculation_speed(self):
        """Test that baseline calculation completes in < 5 seconds."""
        import time
        
        # Generate 30 days of minute-level data (43,200 samples)
        df = generate_synthetic_temperature_data(
            baseline_temp=38.5,
            n_days=30,
            samples_per_hour=60,
        )
        
        calculator = BaselineCalculator(window_days=30)
        
        start_time = time.time()
        result = calculator.calculate_baseline(df, cow_id=1)
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 5.0, \
            f"Calculation took {elapsed_time:.2f}s, exceeds 5s threshold"
        
        # Verify result is valid
        assert result.baseline_temp is not None


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """Integration tests for complete baseline calculation pipeline."""
    
    def test_full_pipeline(self, tmp_path):
        """Test complete pipeline from data to baseline with history."""
        # Setup
        history_manager = BaselineHistoryManager(
            storage_backend="json",
            storage_path=str(tmp_path / "baseline_history")
        )
        
        drift_detector = BaselineDriftDetector()
        
        updater = BaselineUpdater(
            history_manager=history_manager,
            drift_detector=drift_detector,
            adaptive_windowing=True,
        )
        
        # Generate data with slight drift
        df = generate_drifting_baseline_data(
            initial_baseline=38.5,
            drift_rate=0.05,  # Slow drift
            n_days=21,
        )
        
        # Update baselines at different time points
        results = []
        for day in range(7, 21, 3):
            current_time = df['timestamp'].min() + timedelta(days=day)
            result = updater.update_baseline(
                df, cow_id=1, current_time=current_time, force_update=True
            )
            results.append(result)
        
        # Verify all updates succeeded
        assert len(results) == 5
        assert all(r is not None for r in results)
        
        # Verify history was stored
        history = history_manager.retrieve_history(cow_id=1)
        assert len(history) >= 5
        
        # Verify baselines show trend (but smoothed)
        baselines = [r.baseline_temp for r in results]
        assert baselines[-1] > baselines[0], "Drift not captured in baselines"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
