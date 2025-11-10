"""
Health Condition Simulators for Cattle

Simulates various health conditions with realistic temperature and motion patterns:
- Fever
- Heat Stress
- Estrus
- Pregnancy

Each simulator modulates the baseline behavioral patterns with condition-specific
physiological changes based on veterinary literature.

References:
- Burfeind et al. (2014): Fever patterns in dairy cattle
- West (2003): Heat stress indicators
- Roelofs et al. (2005): Estrus detection using activity and temperature
- Suthar et al. (2011): Pregnancy-related behavioral changes
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from .circadian_rhythm import CircadianRhythmGenerator


@dataclass
class HealthConditionParams:
    """Parameters for health condition simulation."""
    condition_type: str  # 'fever', 'heat_stress', 'estrus', 'pregnancy'
    onset_minute: int  # When condition starts
    duration_minutes: int  # How long condition lasts
    severity: float = 1.0  # Severity multiplier (0.5-2.0)


class FeverSimulator:
    """
    Simulates fever condition in cattle.

    Fever characteristics:
    - Sustained temperature elevation >39.5°C
    - Reduced activity (20-40% decrease)
    - Preserved circadian rhythm with larger amplitude
    - Duration: 6-48 hours typical

    Literature:
    - Burfeind et al. (2014): Fever threshold 39.5°C
    - Elevated temperature maintained 6-48 hours
    - Activity reduction during sickness behavior
    """

    def __init__(
        self,
        baseline_fever_temp: float = 40.0,
        activity_reduction: float = 0.30,  # 30% reduction
        circadian_amplitude: float = 0.7
    ):
        """
        Initialize fever simulator.

        Args:
            baseline_fever_temp: Elevated baseline temperature (°C), default 40.0°C
            activity_reduction: Fraction to reduce activity (0.0-1.0), default 0.30
            circadian_amplitude: Circadian amplitude during fever (°C), default 0.7°C
        """
        self.baseline_fever_temp = baseline_fever_temp
        self.activity_reduction = activity_reduction
        self.circadian_amplitude = circadian_amplitude

        if not 39.5 <= baseline_fever_temp <= 42.0:
            raise ValueError("Fever baseline should be 39.5-42.0°C")
        if not 0.0 <= activity_reduction <= 1.0:
            raise ValueError("Activity reduction should be 0.0-1.0")

    def generate_temperature(
        self,
        duration_minutes: int,
        start_time_minutes: float = 0.0,
        onset_hours: float = 0.0,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Generate fever temperature pattern with gradual onset.

        Args:
            duration_minutes: Total duration
            start_time_minutes: Starting time offset
            onset_hours: Hours for gradual temperature rise (default 0 = immediate)
            random_seed: Random seed

        Returns:
            Temperature array (°C)
        """
        circadian_gen = CircadianRhythmGenerator(
            baseline_temp=self.baseline_fever_temp,
            amplitude=self.circadian_amplitude,
            peak_hour=19.0,
            noise_std=0.15
        )

        temperature = circadian_gen.generate(
            duration_minutes, start_time_minutes, random_seed
        )

        # Apply gradual onset if specified
        if onset_hours > 0:
            onset_minutes = int(onset_hours * 60)
            if onset_minutes < duration_minutes:
                # Create onset ramp: 0 to 1 over onset_minutes
                ramp = np.linspace(0, 1, onset_minutes)
                full = np.ones(duration_minutes - onset_minutes)
                fever_multiplier = np.concatenate([ramp, full])

                # Blend from normal (38.5°C) to fever baseline
                normal_baseline = 38.5
                temperature_offset = (self.baseline_fever_temp - normal_baseline) * fever_multiplier
                temperature = normal_baseline + temperature_offset + \
                             self.circadian_amplitude * np.sin(
                                 (2 * np.pi / (24 * 60)) * (np.arange(duration_minutes) + start_time_minutes) -
                                 19.0 * (2 * np.pi / 24.0) + (np.pi / 2)
                             )

                # Add noise
                if random_seed is not None:
                    np.random.seed(random_seed)
                temperature += np.random.normal(0, 0.15, size=duration_minutes)
                temperature = np.clip(temperature, 35.0, 42.0)

        return temperature

    def modify_motion_pattern(
        self,
        acceleration: np.ndarray,
        apply_reduction: bool = True
    ) -> np.ndarray:
        """
        Reduce motion patterns to simulate sickness behavior.

        Args:
            acceleration: Original acceleration values (any axis)
            apply_reduction: Whether to apply reduction (False for baseline)

        Returns:
            Modified acceleration values with reduced variance
        """
        if not apply_reduction:
            return acceleration

        # Reduce variance around mean (lethargy)
        mean_value = np.mean(acceleration)
        centered = acceleration - mean_value
        reduced = centered * (1.0 - self.activity_reduction)
        return mean_value + reduced


class HeatStressSimulator:
    """
    Simulates heat stress condition in cattle.

    Heat stress characteristics:
    - Elevated temperature (39.0-40.5°C)
    - Initial high activity (pacing, restlessness)
    - Progressive exhaustion (activity decline)
    - Panting behavior (rapid Mya oscillations 60-80/min)
    - Disrupted circadian rhythm

    Literature:
    - West (2003): Heat stress thresholds and responses
    - Gaughan et al. (2008): Panting frequency during heat stress
    - Hillman et al. (2009): Behavioral changes under heat load
    """

    def __init__(
        self,
        peak_temp: float = 40.0,
        panting_frequency: float = 70.0,  # cycles per minute
        initial_activity_boost: float = 0.4,  # 40% increase initially
        exhaustion_onset_hours: float = 2.0
    ):
        """
        Initialize heat stress simulator.

        Args:
            peak_temp: Peak temperature during heat stress (°C)
            panting_frequency: Panting rate (cycles/min), default 70/min
            initial_activity_boost: Initial activity increase fraction
            exhaustion_onset_hours: Hours until exhaustion sets in
        """
        self.peak_temp = peak_temp
        self.panting_frequency = panting_frequency
        self.initial_activity_boost = initial_activity_boost
        self.exhaustion_onset_hours = exhaustion_onset_hours

    def generate_temperature(
        self,
        duration_minutes: int,
        start_time_minutes: float = 0.0,
        ambient_temp: float = 30.0,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Generate heat stress temperature pattern.

        Temperature rises rapidly, peaks, then may decline with exhaustion.
        Circadian rhythm disrupted.

        Args:
            duration_minutes: Total duration
            start_time_minutes: Starting time offset
            ambient_temp: Ambient temperature (affects severity)
            random_seed: Random seed

        Returns:
            Temperature array (°C)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        time_minutes = np.arange(duration_minutes)
        time_hours = time_minutes / 60.0

        # Base temperature rise profile
        # Rapid rise, plateau, possible decline
        baseline = 38.5

        # Temperature rises to peak over first 30-60 minutes
        rise_phase = np.minimum(time_hours / 0.5, 1.0)  # 0.5 hour rise time
        temp_rise = (self.peak_temp - baseline) * rise_phase

        # After exhaustion onset, temperature may decline slightly
        exhaustion_phase = np.maximum(0, time_hours - self.exhaustion_onset_hours) / 2.0
        exhaustion_decline = 0.5 * np.tanh(exhaustion_phase)  # Gradual decline up to 0.5°C

        temperature = baseline + temp_rise - exhaustion_decline

        # Add irregular fluctuations (disrupted rhythm)
        irregular_variation = 0.3 * np.sin(2 * np.pi * time_hours / 3.0)  # 3-hour irregular cycle
        temperature += irregular_variation

        # Add noise
        temperature += np.random.normal(0, 0.2, size=duration_minutes)

        # Physiological limits
        temperature = np.clip(temperature, 38.0, 41.5)

        return temperature

    def generate_panting_pattern(
        self,
        duration_minutes: int,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Generate panting behavior pattern for Mya (lateral head movement).

        Panting creates rhythmic lateral head motion at 60-80 cycles/min.

        Args:
            duration_minutes: Duration
            random_seed: Random seed

        Returns:
            Mya pattern (lateral acceleration)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        time_seconds = np.arange(duration_minutes * 60)  # Second resolution
        frequency_hz = self.panting_frequency / 60.0  # Convert cycles/min to Hz

        # Panting pattern: rhythmic with some irregularity
        panting_amplitude = 0.08 + np.random.uniform(-0.02, 0.02, size=len(time_seconds))
        panting_signal = panting_amplitude * np.sin(2 * np.pi * frequency_hz * time_seconds)

        # Downsample to per-minute (take mean of each minute)
        panting_per_minute = panting_signal.reshape(duration_minutes, 60).mean(axis=1)

        return panting_per_minute

    def modify_activity_pattern(
        self,
        acceleration: np.ndarray,
        time_minutes: np.ndarray
    ) -> np.ndarray:
        """
        Modify activity: Initial increase, then exhaustion decline.

        Args:
            acceleration: Original acceleration values
            time_minutes: Time array (minutes)

        Returns:
            Modified acceleration with heat stress pattern
        """
        time_hours = time_minutes / 60.0

        # Activity multiplier over time
        # High initially (restlessness), then decline (exhaustion)
        initial_boost = 1.0 + self.initial_activity_boost
        exhaustion_factor = np.exp(-time_hours / self.exhaustion_onset_hours)
        activity_multiplier = 1.0 + (initial_boost - 1.0) * exhaustion_factor

        # After exhaustion, activity drops below normal
        post_exhaustion = np.maximum(0, time_hours - self.exhaustion_onset_hours)
        activity_multiplier -= 0.3 * np.tanh(post_exhaustion / 2.0)

        return acceleration * activity_multiplier


class EstrusSimulator:
    """
    Simulates estrus (heat) condition in cattle.

    Estrus characteristics:
    - Short-term temperature rise: 0.3-0.6°C over 5-10 minutes
    - Increased activity: 30-60% boost for 12-24 hours
    - Mounting behavior, restlessness, reduced feeding
    - Repeats every ~21 days (estrous cycle)

    Literature:
    - Roelofs et al. (2005): Estrus detection thresholds
    - Aungier et al. (2012): Temperature and activity changes
    - Palmer et al. (2010): Mounting behavior acceleration signatures
    """

    def __init__(
        self,
        temp_rise: float = 0.4,  # 0.4°C rise
        temp_spike_duration_minutes: int = 8,
        activity_increase: float = 0.45,  # 45% increase
        estrus_duration_hours: int = 18,
        cycle_length_days: int = 21
    ):
        """
        Initialize estrus simulator.

        Args:
            temp_rise: Temperature increase during spike (°C)
            temp_spike_duration_minutes: Duration of temperature spike
            activity_increase: Activity boost fraction (0.3-0.6)
            estrus_duration_hours: Duration of estrus period (hours)
            cycle_length_days: Days between estrus cycles
        """
        self.temp_rise = temp_rise
        self.temp_spike_duration_minutes = temp_spike_duration_minutes
        self.activity_increase = activity_increase
        self.estrus_duration_hours = estrus_duration_hours
        self.cycle_length_days = cycle_length_days

    def generate_temperature_spike(
        self,
        duration_minutes: int,
        spike_start_minute: int = 0,
        base_temperature: np.ndarray = None,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Add estrus temperature spike to base circadian pattern.

        Args:
            duration_minutes: Total duration
            spike_start_minute: When spike occurs (minute offset)
            base_temperature: Base circadian temperature pattern (if None, creates normal)
            random_seed: Random seed

        Returns:
            Temperature with estrus spike added
        """
        if base_temperature is None:
            # Generate normal circadian base
            from .circadian_rhythm import create_normal_circadian
            base_temperature = create_normal_circadian(duration_minutes, random_seed=random_seed)

        temperature = base_temperature.copy()

        # Create temperature spike (Gaussian-like rise and fall)
        if spike_start_minute < duration_minutes:
            spike_end = min(spike_start_minute + self.temp_spike_duration_minutes, duration_minutes)
            spike_duration = spike_end - spike_start_minute

            # Smooth rise and fall (half-Gaussian shape)
            t = np.linspace(-2, 2, spike_duration)
            spike_shape = self.temp_rise * np.exp(-t**2)  # Gaussian pulse

            temperature[spike_start_minute:spike_end] += spike_shape

        return temperature

    def apply_activity_boost(
        self,
        acceleration: np.ndarray,
        time_minutes: np.ndarray,
        estrus_start_minute: int = 0
    ) -> np.ndarray:
        """
        Increase activity during estrus period.

        Args:
            acceleration: Base acceleration values
            time_minutes: Time array
            estrus_start_minute: When estrus period starts

        Returns:
            Boosted acceleration
        """
        estrus_duration_minutes = self.estrus_duration_hours * 60

        # Activity multiplier: 1.0 normally, 1.0 + activity_increase during estrus
        activity_multiplier = np.ones_like(time_minutes, dtype=float)

        estrus_mask = (time_minutes >= estrus_start_minute) & \
                      (time_minutes < estrus_start_minute + estrus_duration_minutes)

        activity_multiplier[estrus_mask] = 1.0 + self.activity_increase

        return acceleration * activity_multiplier

    def get_estrus_schedule(
        self,
        simulation_days: int,
        first_estrus_day: int = 5,
        cycle_variability_days: int = 2
    ) -> list:
        """
        Generate schedule of estrus events over simulation period.

        Args:
            simulation_days: Total simulation duration (days)
            first_estrus_day: Day of first estrus event
            cycle_variability_days: Random variation in cycle length (±days)

        Returns:
            List of estrus start days
        """
        estrus_days = []
        current_day = first_estrus_day

        while current_day < simulation_days:
            estrus_days.append(current_day)
            # Next cycle with variation
            cycle_length = self.cycle_length_days + np.random.randint(
                -cycle_variability_days, cycle_variability_days + 1
            )
            current_day += cycle_length

        return estrus_days


class PregnancySimulator:
    """
    Simulates pregnancy condition in cattle.

    Pregnancy characteristics:
    - Stable temperature: Low variance (<0.15°C daily)
    - Slightly elevated baseline: +0.1-0.2°C
    - Dampened circadian rhythm: ±0.3-0.4°C instead of ±0.5°C
    - Progressive activity reduction: 10-30% decrease over 60+ days
    - Increased lying time, reduced walking

    Literature:
    - Suthar et al. (2011): Activity changes during gestation
    - Kendall et al. (2008): Temperature stability in pregnant cattle
    - Borchers et al. (2017): Lying behavior increases with gestation
    """

    def __init__(
        self,
        baseline_elevation: float = 0.15,  # +0.15°C above normal
        circadian_amplitude: float = 0.35,  # Dampened to ±0.35°C
        temp_noise_std: float = 0.08,  # Low noise (stable)
        activity_reduction_rate: float = 0.003  # 0.3% per day, reaches 18% after 60 days
    ):
        """
        Initialize pregnancy simulator.

        Args:
            baseline_elevation: Temperature increase above normal (°C)
            circadian_amplitude: Dampened circadian amplitude (°C)
            temp_noise_std: Reduced temperature noise (°C)
            activity_reduction_rate: Daily activity reduction rate (fraction per day)
        """
        self.baseline_elevation = baseline_elevation
        self.circadian_amplitude = circadian_amplitude
        self.temp_noise_std = temp_noise_std
        self.activity_reduction_rate = activity_reduction_rate

    def generate_temperature(
        self,
        duration_minutes: int,
        pregnancy_day: int = 30,  # Day of pregnancy (0-280)
        start_time_minutes: float = 0.0,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Generate pregnancy temperature pattern.

        Temperature is stable with dampened circadian rhythm.

        Args:
            duration_minutes: Duration in minutes
            pregnancy_day: Current day of pregnancy (affects baseline)
            start_time_minutes: Starting time offset
            random_seed: Random seed

        Returns:
            Temperature array (°C)
        """
        # Baseline increases slightly over pregnancy
        # Early pregnancy: +0.1°C, late pregnancy: +0.2°C
        progression_factor = min(pregnancy_day / 280.0, 1.0)
        current_baseline = 38.5 + self.baseline_elevation * (0.5 + 0.5 * progression_factor)

        circadian_gen = CircadianRhythmGenerator(
            baseline_temp=current_baseline,
            amplitude=self.circadian_amplitude,
            peak_hour=19.0,
            noise_std=self.temp_noise_std
        )

        temperature = circadian_gen.generate(
            duration_minutes, start_time_minutes, random_seed
        )

        return temperature

    def apply_activity_reduction(
        self,
        acceleration: np.ndarray,
        pregnancy_day: int = 30
    ) -> np.ndarray:
        """
        Apply progressive activity reduction during pregnancy.

        Activity decreases gradually: 0% at day 0, 18% at day 60, 30% at term.

        Args:
            acceleration: Base acceleration values
            pregnancy_day: Current day of pregnancy

        Returns:
            Reduced acceleration
        """
        # Calculate cumulative reduction
        cumulative_reduction = min(
            pregnancy_day * self.activity_reduction_rate,
            0.30  # Cap at 30% reduction
        )

        reduction_factor = 1.0 - cumulative_reduction

        # Apply to variance (maintain mean, reduce fluctuations)
        mean_value = np.mean(acceleration)
        centered = acceleration - mean_value
        reduced = centered * reduction_factor

        return mean_value + reduced


# Convenience function to create condition
def create_health_condition(
    condition_type: str,
    duration_minutes: int,
    severity: float = 1.0,
    **kwargs
) -> Dict:
    """
    Factory function to create health condition simulators.

    Args:
        condition_type: 'fever', 'heat_stress', 'estrus', or 'pregnancy'
        duration_minutes: Duration of simulation
        severity: Severity multiplier (0.5-2.0)
        **kwargs: Additional parameters specific to condition

    Returns:
        Dictionary with 'temperature' and 'activity_modifier' keys
    """
    if condition_type == 'fever':
        simulator = FeverSimulator(
            baseline_fever_temp=39.5 + severity * 0.5,
            activity_reduction=0.20 + severity * 0.15
        )
        temp = simulator.generate_temperature(duration_minutes, **kwargs)
        return {
            'simulator': simulator,
            'temperature': temp,
            'condition_type': 'fever'
        }

    elif condition_type == 'heat_stress':
        simulator = HeatStressSimulator(
            peak_temp=39.5 + severity * 0.5,
            initial_activity_boost=0.3 + severity * 0.2
        )
        temp = simulator.generate_temperature(duration_minutes, **kwargs)
        return {
            'simulator': simulator,
            'temperature': temp,
            'panting_pattern': simulator.generate_panting_pattern(duration_minutes),
            'condition_type': 'heat_stress'
        }

    elif condition_type == 'estrus':
        simulator = EstrusSimulator(
            temp_rise=0.3 + severity * 0.2,
            activity_increase=0.35 + severity * 0.15
        )
        temp = simulator.generate_temperature_spike(duration_minutes, **kwargs)
        return {
            'simulator': simulator,
            'temperature': temp,
            'condition_type': 'estrus'
        }

    elif condition_type == 'pregnancy':
        simulator = PregnancySimulator(
            activity_reduction_rate=0.002 + severity * 0.002
        )
        temp = simulator.generate_temperature(duration_minutes, **kwargs)
        return {
            'simulator': simulator,
            'temperature': temp,
            'condition_type': 'pregnancy'
        }

    else:
        raise ValueError(f"Unknown condition type: {condition_type}")


if __name__ == "__main__":
    # Demo: Generate and visualize health conditions
    import matplotlib.pyplot as plt

    duration = 1440  # 24 hours
    time_hours = np.arange(duration) / 60.0

    # Create conditions
    fever = create_health_condition('fever', duration, severity=1.0, random_seed=42)
    heat_stress = create_health_condition('heat_stress', duration, severity=1.0, random_seed=42)
    estrus = create_health_condition('estrus', duration, severity=1.0, spike_start_minute=300, random_seed=42)
    pregnancy = create_health_condition('pregnancy', duration, severity=1.0, pregnancy_day=60, random_seed=42)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(time_hours, fever['temperature'])
    axes[0, 0].axhline(y=39.5, color='red', linestyle='--', label='Fever threshold')
    axes[0, 0].set_title('Fever (40°C baseline)')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(time_hours, heat_stress['temperature'])
    axes[0, 1].set_title('Heat Stress (activity-dependent)')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(time_hours, estrus['temperature'])
    axes[1, 0].axvline(x=300/60, color='red', linestyle='--', alpha=0.5, label='Spike at 5h')
    axes[1, 0].set_title('Estrus (0.4°C spike at 5h)')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(time_hours, pregnancy['temperature'])
    axes[1, 1].set_title('Pregnancy (stable, dampened rhythm)')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
