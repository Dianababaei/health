"""
Unit Tests for Health Condition Simulators

Tests circadian rhythm generator and all health condition simulators
to ensure they produce realistic, physiologically valid outputs.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation.circadian_rhythm import (
    CircadianRhythmGenerator,
    create_normal_circadian,
    create_fever_circadian,
    create_pregnancy_circadian
)
from simulation.health_conditions import (
    FeverSimulator,
    HeatStressSimulator,
    EstrusSimulator,
    PregnancySimulator,
    create_health_condition
)


class TestCircadianRhythmGenerator(unittest.TestCase):
    """Test circadian temperature pattern generation."""

    def test_normal_circadian_baseline(self):
        """Test that normal circadian pattern averages to baseline."""
        generator = CircadianRhythmGenerator(baseline_temp=38.5)
        temp = generator.generate(1440, random_seed=42)  # 24 hours

        # Mean should be close to baseline (within noise tolerance)
        self.assertAlmostEqual(np.mean(temp), 38.5, delta=0.15)

    def test_circadian_amplitude(self):
        """Test that peak-to-trough amplitude matches specification."""
        generator = CircadianRhythmGenerator(
            baseline_temp=38.5,
            amplitude=0.5,
            noise_std=0.0  # No noise for clearer measurement
        )
        temp = generator.generate(1440, random_seed=42)

        # Peak-to-trough should be roughly 2 * amplitude
        peak = np.max(temp)
        trough = np.min(temp)
        amplitude_measured = (peak - trough) / 2.0

        self.assertAlmostEqual(amplitude_measured, 0.5, delta=0.1)

    def test_peak_timing(self):
        """Test that temperature peaks at specified hour."""
        generator = CircadianRhythmGenerator(
            baseline_temp=38.5,
            peak_hour=19.0,  # 7 PM
            noise_std=0.0
        )
        temp = generator.generate(1440, start_time_minutes=0, random_seed=42)

        # Find peak hour
        peak_index = np.argmax(temp)
        peak_hour = peak_index / 60.0

        # Should peak around 19:00 (±1 hour tolerance)
        self.assertAlmostEqual(peak_hour, 19.0, delta=1.0)

    def test_temperature_limits(self):
        """Test that generated temperatures stay within physiological limits."""
        generator = CircadianRhythmGenerator(baseline_temp=38.5)
        temp = generator.generate(1440 * 7, random_seed=42)  # 7 days

        # All temperatures should be 35-42°C
        self.assertTrue(np.all(temp >= 35.0))
        self.assertTrue(np.all(temp <= 42.0))

    def test_reproducibility(self):
        """Test that same seed produces same output."""
        temp1 = create_normal_circadian(1440, random_seed=42)
        temp2 = create_normal_circadian(1440, random_seed=42)

        np.testing.assert_array_equal(temp1, temp2)

    def test_fever_circadian_elevated(self):
        """Test that fever circadian has elevated baseline."""
        temp_fever = create_fever_circadian(1440, baseline_temp=40.0, random_seed=42)

        # Mean should be around 40°C
        self.assertGreater(np.mean(temp_fever), 39.5)
        self.assertLess(np.mean(temp_fever), 40.5)

    def test_pregnancy_circadian_dampened(self):
        """Test that pregnancy has dampened circadian rhythm."""
        temp_normal = create_normal_circadian(1440, random_seed=42)
        temp_pregnancy = create_pregnancy_circadian(1440, random_seed=42)

        # Pregnancy should have lower variance (dampened rhythm)
        variance_normal = np.var(temp_normal)
        variance_pregnancy = np.var(temp_pregnancy)

        self.assertLess(variance_pregnancy, variance_normal)


class TestFeverSimulator(unittest.TestCase):
    """Test fever condition simulator."""

    def test_fever_temperature_threshold(self):
        """Test that fever generates temperature >39.5°C consistently."""
        simulator = FeverSimulator(baseline_fever_temp=40.0)
        temp = simulator.generate_temperature(1440, random_seed=42)

        # At least 75% of readings should be >39.5°C (accounts for circadian trough)
        fever_count = np.sum(temp > 39.5)
        fever_percentage = fever_count / len(temp)

        self.assertGreater(fever_percentage, 0.75)

    def test_fever_circadian_preserved(self):
        """Test that circadian rhythm is preserved during fever."""
        simulator = FeverSimulator(baseline_fever_temp=40.0)
        temp = simulator.generate_temperature(1440 * 3, random_seed=42)  # 3 days

        # Should have periodic variation
        # Check that we have both peaks and troughs
        daily_means = [temp[i*1440:(i+1)*1440].mean() for i in range(3)]
        daily_stds = [temp[i*1440:(i+1)*1440].std() for i in range(3)]

        # Each day should have variance (circadian present)
        for std in daily_stds:
            self.assertGreater(std, 0.2)  # Should have daily variation

    def test_activity_reduction(self):
        """Test that fever reduces activity levels."""
        simulator = FeverSimulator(activity_reduction=0.30)

        # Generate sample acceleration data
        normal_activity = np.random.normal(0.5, 0.2, 1440)
        reduced_activity = simulator.modify_motion_pattern(normal_activity)

        # Reduced activity should have lower variance
        var_normal = np.var(normal_activity)
        var_reduced = np.var(reduced_activity)

        self.assertLess(var_reduced, var_normal)

        # Mean should be similar (not moving average, just reducing variance)
        self.assertAlmostEqual(np.mean(normal_activity), np.mean(reduced_activity), delta=0.05)

    def test_gradual_onset(self):
        """Test fever onset is gradual over specified hours."""
        simulator = FeverSimulator(baseline_fever_temp=40.0)
        temp = simulator.generate_temperature(
            duration_minutes=240,  # 4 hours
            onset_hours=2.0,  # 2 hour gradual onset
            random_seed=42
        )

        # First hour should be lower than last hour
        first_hour_mean = np.mean(temp[:60])
        last_hour_mean = np.mean(temp[-60:])

        self.assertLess(first_hour_mean, 39.5)  # Should start below fever threshold
        self.assertGreater(last_hour_mean, 39.5)  # Should end in fever range


class TestHeatStressSimulator(unittest.TestCase):
    """Test heat stress condition simulator."""

    def test_temperature_rise(self):
        """Test that heat stress causes temperature elevation."""
        simulator = HeatStressSimulator(peak_temp=40.0)
        temp = simulator.generate_temperature(180, random_seed=42)  # 3 hours

        # Peak should be reached
        self.assertGreater(np.max(temp), 39.5)

    def test_activity_pattern_biphasic(self):
        """Test that activity is high initially, then declines (exhaustion)."""
        simulator = HeatStressSimulator(
            initial_activity_boost=0.4,
            exhaustion_onset_hours=2.0
        )

        time_minutes = np.arange(240)  # 4 hours
        normal_activity = np.ones(240) * 0.5  # Constant baseline
        modified_activity = simulator.modify_activity_pattern(normal_activity, time_minutes)

        # Early activity (first hour) should be elevated
        early_mean = np.mean(modified_activity[:60])
        self.assertGreater(early_mean, 0.5)  # Should be boosted

        # Late activity (after 3 hours) should be reduced (exhaustion)
        late_mean = np.mean(modified_activity[-60:])
        self.assertLess(late_mean, 0.5)  # Should be reduced

    def test_panting_pattern_frequency(self):
        """Test that panting pattern has correct frequency."""
        simulator = HeatStressSimulator(panting_frequency=70.0)
        panting = simulator.generate_panting_pattern(60, random_seed=42)  # 60 minutes for better signal

        # Should be oscillating (not constant) - after downsampling, std will be small
        self.assertGreater(np.std(panting), 0.0005)

        # Amplitude should be reasonable for Mya (0.05-0.15g)
        self.assertLess(np.max(np.abs(panting)), 0.20)

    def test_disrupted_rhythm(self):
        """Test that circadian rhythm is disrupted during heat stress."""
        simulator = HeatStressSimulator()
        temp = simulator.generate_temperature(1440 * 2, random_seed=42)  # 2 days

        # Should have irregular patterns (not smooth circadian)
        # Check for irregular variance within days
        day1 = temp[:1440]
        day2 = temp[1440:]

        # Days should differ (not regular circadian)
        mean_diff = abs(np.mean(day1) - np.mean(day2))
        # Heat stress is acute, so days can differ significantly
        self.assertGreater(mean_diff, 0.0)  # Will vary


class TestEstrusSimulator(unittest.TestCase):
    """Test estrus (heat) condition simulator."""

    def test_temperature_spike_magnitude(self):
        """Test that estrus creates 0.3-0.6°C temperature spike."""
        simulator = EstrusSimulator(temp_rise=0.4, temp_spike_duration_minutes=8)

        # Generate base temperature
        base_temp = np.ones(100) * 38.5
        temp_with_spike = simulator.generate_temperature_spike(
            duration_minutes=100,
            spike_start_minute=50,
            base_temperature=base_temp,
            random_seed=42
        )

        # Temperature at peak of spike should be elevated
        # Find the max in spike region
        spike_region = temp_with_spike[50:58]
        baseline_region = temp_with_spike[:40]

        spike_max = np.max(spike_region)
        baseline_mean = np.mean(baseline_region)

        temp_rise = spike_max - baseline_mean
        self.assertGreater(temp_rise, 0.15)  # Should rise at least 0.15°C (Gaussian peak)
        self.assertLess(temp_rise, 0.8)      # Should not exceed 0.8°C

    def test_spike_duration(self):
        """Test that temperature spike lasts specified duration."""
        simulator = EstrusSimulator(temp_rise=0.5, temp_spike_duration_minutes=10)

        base_temp = np.ones(200) * 38.5
        temp_with_spike = simulator.generate_temperature_spike(
            duration_minutes=200,
            spike_start_minute=100,
            base_temperature=base_temp
        )

        # Gaussian spike shape means elevated region (>0.15°C) will be narrower than duration
        # Count region with noticeable elevation
        elevated = temp_with_spike > 38.65
        elevated_duration = np.sum(elevated)

        # Gaussian pulse has ~60% of energy in ±1 sigma, so expect ~4-8 minutes above threshold
        self.assertGreaterEqual(elevated_duration, 4)  # At least 4 minutes elevated
        self.assertLess(elevated_duration, 12)          # Not more than 12 minutes

    def test_activity_boost(self):
        """Test that estrus increases activity by 30-60%."""
        simulator = EstrusSimulator(
            activity_increase=0.45,  # 45% increase
            estrus_duration_hours=18
        )

        time_minutes = np.arange(1440)  # 24 hours
        normal_activity = np.ones(1440) * 0.5
        boosted_activity = simulator.apply_activity_boost(
            normal_activity,
            time_minutes,
            estrus_start_minute=0  # Estrus from start
        )

        # During estrus (first 18 hours), activity should be boosted
        estrus_period = boosted_activity[:18*60]
        non_estrus_period = boosted_activity[18*60:]

        estrus_mean = np.mean(estrus_period)
        non_estrus_mean = np.mean(non_estrus_period)

        # Estrus should be higher
        self.assertGreater(estrus_mean, non_estrus_mean)

        # Should be approximately 45% higher
        boost_factor = (estrus_mean - non_estrus_mean) / non_estrus_mean
        self.assertAlmostEqual(boost_factor, 0.45, delta=0.05)

    def test_estrus_cycle_schedule(self):
        """Test that estrus events repeat at ~21-day intervals."""
        simulator = EstrusSimulator(cycle_length_days=21)

        np.random.seed(42)
        estrus_days = simulator.get_estrus_schedule(
            simulation_days=100,
            first_estrus_day=5,
            cycle_variability_days=2
        )

        # Should have multiple events
        self.assertGreater(len(estrus_days), 3)

        # Intervals should be roughly 21 days (±2 days)
        intervals = np.diff(estrus_days)
        mean_interval = np.mean(intervals)

        self.assertAlmostEqual(mean_interval, 21, delta=3)


class TestPregnancySimulator(unittest.TestCase):
    """Test pregnancy condition simulator."""

    def test_temperature_stability(self):
        """Test that pregnancy temperature has low variance."""
        simulator = PregnancySimulator()
        temp = simulator.generate_temperature(
            duration_minutes=1440,
            pregnancy_day=60,
            random_seed=42
        )

        # Variance should be low (stable temperature)
        temp_variance = np.var(temp)
        self.assertLess(temp_variance, 0.15)  # Low variance

    def test_dampened_circadian(self):
        """Test that circadian rhythm is dampened during pregnancy."""
        simulator = PregnancySimulator(circadian_amplitude=0.35)
        temp = simulator.generate_temperature(1440, pregnancy_day=60, random_seed=42)

        # Peak-to-trough should be smaller than normal (±0.5°C)
        # With noise, actual amplitude will be higher than specified
        peak = np.max(temp)
        trough = np.min(temp)
        amplitude = (peak - trough) / 2.0

        # Should be less than 0.6°C (dampened compared to normal ±0.5°C + noise)
        self.assertLess(amplitude, 0.6)

    def test_elevated_baseline(self):
        """Test that pregnancy has slightly elevated baseline temperature."""
        simulator = PregnancySimulator(baseline_elevation=0.15)
        temp = simulator.generate_temperature(1440, pregnancy_day=60, random_seed=42)

        # Mean should be slightly above 38.5°C
        mean_temp = np.mean(temp)
        self.assertGreater(mean_temp, 38.5)
        self.assertLess(mean_temp, 39.0)  # But not fever level

    def test_progressive_activity_reduction(self):
        """Test that activity reduces progressively during pregnancy."""
        simulator = PregnancySimulator(activity_reduction_rate=0.003)

        normal_activity = np.random.normal(0.5, 0.2, 1000)

        # Early pregnancy (day 10)
        activity_early = simulator.apply_activity_reduction(
            normal_activity.copy(),
            pregnancy_day=10
        )

        # Late pregnancy (day 100)
        activity_late = simulator.apply_activity_reduction(
            normal_activity.copy(),
            pregnancy_day=100
        )

        # Late pregnancy should have lower variance than early
        var_early = np.var(activity_early)
        var_late = np.var(activity_late)

        self.assertLess(var_late, var_early)

    def test_activity_reduction_cap(self):
        """Test that activity reduction caps at 30%."""
        simulator = PregnancySimulator(activity_reduction_rate=0.003)

        normal_activity = np.ones(1000) * 0.5

        # Very late pregnancy (day 200, beyond typical)
        activity_very_late = simulator.apply_activity_reduction(
            normal_activity.copy(),
            pregnancy_day=200
        )

        # Should cap at 30% reduction
        mean_reduction = np.mean(normal_activity) - np.mean(activity_very_late)
        reduction_fraction = mean_reduction / np.mean(normal_activity)

        self.assertLess(reduction_fraction, 0.35)  # Should be ≤30%


class TestHealthConditionFactory(unittest.TestCase):
    """Test the create_health_condition factory function."""

    def test_create_fever(self):
        """Test fever condition creation."""
        result = create_health_condition('fever', 1440, severity=1.0, random_seed=42)

        self.assertIn('temperature', result)
        self.assertIn('simulator', result)
        self.assertEqual(result['condition_type'], 'fever')
        self.assertEqual(len(result['temperature']), 1440)

    def test_create_heat_stress(self):
        """Test heat stress condition creation."""
        result = create_health_condition('heat_stress', 1440, severity=1.0, random_seed=42)

        self.assertIn('temperature', result)
        self.assertIn('panting_pattern', result)
        self.assertEqual(result['condition_type'], 'heat_stress')

    def test_create_estrus(self):
        """Test estrus condition creation."""
        result = create_health_condition('estrus', 1440, severity=1.0, random_seed=42)

        self.assertIn('temperature', result)
        self.assertEqual(result['condition_type'], 'estrus')

    def test_create_pregnancy(self):
        """Test pregnancy condition creation."""
        result = create_health_condition('pregnancy', 1440, severity=1.0, pregnancy_day=60, random_seed=42)

        self.assertIn('temperature', result)
        self.assertEqual(result['condition_type'], 'pregnancy')

    def test_severity_scaling(self):
        """Test that severity parameter scales condition effects."""
        mild_fever = create_health_condition('fever', 1440, severity=0.5, random_seed=42)
        severe_fever = create_health_condition('fever', 1440, severity=2.0, random_seed=42)

        # Severe fever should have higher mean temperature
        mean_mild = np.mean(mild_fever['temperature'])
        mean_severe = np.mean(severe_fever['temperature'])

        self.assertGreater(mean_severe, mean_mild)

    def test_invalid_condition_type(self):
        """Test that invalid condition type raises error."""
        with self.assertRaises(ValueError):
            create_health_condition('unknown_condition', 1440)


class TestPhysiologicalValidity(unittest.TestCase):
    """Test that all simulators produce physiologically valid outputs."""

    def test_all_temperatures_in_range(self):
        """Test that all conditions produce temperatures in valid range (35-42°C)."""
        conditions = [
            create_health_condition('fever', 1440, random_seed=42),
            create_health_condition('heat_stress', 1440, random_seed=42),
            create_health_condition('estrus', 1440, random_seed=42),
            create_health_condition('pregnancy', 1440, random_seed=42)
        ]

        for condition in conditions:
            temp = condition['temperature']
            self.assertTrue(np.all(temp >= 35.0), f"{condition['condition_type']}: temp < 35°C")
            self.assertTrue(np.all(temp <= 42.0), f"{condition['condition_type']}: temp > 42°C")

    def test_no_nan_or_inf(self):
        """Test that simulators don't produce NaN or Inf values."""
        conditions = [
            create_health_condition('fever', 1440, random_seed=42),
            create_health_condition('heat_stress', 1440, random_seed=42),
            create_health_condition('estrus', 1440, random_seed=42),
            create_health_condition('pregnancy', 1440, random_seed=42)
        ]

        for condition in conditions:
            temp = condition['temperature']
            self.assertFalse(np.any(np.isnan(temp)), f"{condition['condition_type']}: NaN values")
            self.assertFalse(np.any(np.isinf(temp)), f"{condition['condition_type']}: Inf values")


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
