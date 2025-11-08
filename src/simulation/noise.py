"""
Noise and Variability Generation

This module generates realistic sensor noise and individual animal variation
to make simulated data more realistic. It includes:
- Gaussian sensor noise appropriate for each sensor type
- Individual animal baseline variations
- Environmental effects (temperature variation)
- Measurement uncertainty

Noise levels based on typical sensor specifications:
- Temperature: ±0.1°C
- Accelerometers: ±0.05g
- Gyroscopes: ±2°/s
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class NoiseParameters:
    """Noise parameters for each sensor type."""
    temperature_std: float = 0.1  # °C
    accelerometer_std: float = 0.05  # g
    gyroscope_std: float = 2.0  # °/s
    
    # Environmental variation
    ambient_temperature_variation: float = 0.05  # Additional temp variation from environment
    
    # Measurement artifacts
    drift_rate: float = 0.001  # Slow sensor drift over time


class NoiseGenerator:
    """
    Generates realistic sensor noise and measurement artifacts.
    
    This class provides methods to add various types of noise to sensor readings,
    including Gaussian noise, drift, and environmental effects.
    """
    
    def __init__(self, params: Optional[NoiseParameters] = None, 
                 seed: Optional[int] = None):
        """
        Initialize the noise generator.
        
        Args:
            params: Noise parameters (uses defaults if None)
            seed: Random seed for reproducibility
        """
        self.params = params or NoiseParameters()
        self.rng = np.random.default_rng(seed)
        self._drift_accumulator = 0.0
        self._time_steps = 0
    
    def add_temperature_noise(self, temperature: float) -> float:
        """
        Add realistic noise to temperature reading.
        
        Args:
            temperature: Base temperature value in °C
            
        Returns:
            Temperature with added noise
        """
        noise = self.rng.normal(0, self.params.temperature_std)
        return temperature + noise
    
    def add_accelerometer_noise(self, acceleration: float) -> float:
        """
        Add realistic noise to accelerometer reading.
        
        Args:
            acceleration: Base acceleration value in g
            
        Returns:
            Acceleration with added noise
        """
        noise = self.rng.normal(0, self.params.accelerometer_std)
        return acceleration + noise
    
    def add_gyroscope_noise(self, angular_velocity: float) -> float:
        """
        Add realistic noise to gyroscope reading.
        
        Args:
            angular_velocity: Base angular velocity in °/s
            
        Returns:
            Angular velocity with added noise
        """
        noise = self.rng.normal(0, self.params.gyroscope_std)
        return angular_velocity + noise
    
    def add_all_sensor_noise(self, temperature: float, fxa: float, mya: float, 
                            rza: float, sxg: float, lyg: float, dzg: float) -> Dict[str, float]:
        """
        Add appropriate noise to all sensor values at once.
        
        Args:
            temperature: Temperature in °C
            fxa, mya, rza: Accelerometer values in g
            sxg, lyg, dzg: Gyroscope values in °/s
            
        Returns:
            Dictionary with noisy sensor values
        """
        return {
            'temperature': self.add_temperature_noise(temperature),
            'fxa': self.add_accelerometer_noise(fxa),
            'mya': self.add_accelerometer_noise(mya),
            'rza': self.add_accelerometer_noise(rza),
            'sxg': self.add_gyroscope_noise(sxg),
            'lyg': self.add_gyroscope_noise(lyg),
            'dzg': self.add_gyroscope_noise(dzg),
        }
    
    def apply_environmental_temperature_effect(self, temperature: float, 
                                              hour_of_day: float,
                                              ambient_temp_celsius: Optional[float] = None) -> float:
        """
        Apply environmental temperature effects (time of day, ambient temperature).
        
        Body temperature naturally varies slightly with ambient conditions and time of day.
        
        Args:
            temperature: Base temperature in °C
            hour_of_day: Hour of day (0-23.99)
            ambient_temp_celsius: Ambient temperature (optional)
            
        Returns:
            Temperature with environmental effects
        """
        # Circadian temperature variation (small effect)
        # Slightly lower at night, slightly higher during day
        circadian_effect = 0.15 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Ambient temperature effect (if provided)
        ambient_effect = 0.0
        if ambient_temp_celsius is not None:
            # Very mild effect - cattle are homeothermic
            # Only extreme temperatures have small effects
            if ambient_temp_celsius > 30:
                ambient_effect = 0.1 * (ambient_temp_celsius - 30) / 10
            elif ambient_temp_celsius < 0:
                ambient_effect = -0.05 * abs(ambient_temp_celsius) / 10
        
        # Random environmental variation
        env_noise = self.rng.normal(0, self.params.ambient_temperature_variation)
        
        return temperature + circadian_effect + ambient_effect + env_noise
    
    def apply_sensor_drift(self, value: float, is_new_session: bool = False) -> float:
        """
        Apply gradual sensor drift over time.
        
        Sensors can drift slightly over long periods. This is typically very small
        but adds realism to long simulations.
        
        Args:
            value: Sensor value
            is_new_session: If True, resets drift (simulating sensor recalibration)
            
        Returns:
            Value with drift applied
        """
        if is_new_session:
            self._drift_accumulator = 0.0
            self._time_steps = 0
        
        self._time_steps += 1
        # Drift accumulates slowly (random walk)
        drift_change = self.rng.normal(0, self.params.drift_rate)
        self._drift_accumulator += drift_change
        
        # Limit total drift to reasonable range
        self._drift_accumulator = np.clip(self._drift_accumulator, -0.1, 0.1)
        
        return value + self._drift_accumulator
    
    def add_rhythmic_variation(self, base_value: float, time_in_state: float,
                              frequency: float, amplitude_scale: float = 1.0) -> float:
        """
        Add rhythmic variation to sensor values (for walking, ruminating, etc.).
        
        Args:
            base_value: Base sensor value
            time_in_state: Time spent in current state (minutes)
            frequency: Frequency of rhythmic pattern (Hz)
            amplitude_scale: Scale factor for amplitude
            
        Returns:
            Value with rhythmic variation
        """
        # Convert time to seconds for frequency calculation
        time_seconds = time_in_state * 60
        
        # Create rhythmic pattern with some noise
        phase = 2 * np.pi * frequency * time_seconds
        phase_noise = self.rng.normal(0, 0.1)  # Small phase jitter
        
        # Amplitude varies slightly
        amplitude = amplitude_scale * (1.0 + self.rng.normal(0, 0.1))
        
        rhythmic_component = amplitude * np.sin(phase + phase_noise)
        
        return base_value + rhythmic_component
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the noise generator state.
        
        Args:
            seed: New random seed (uses existing if None)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._drift_accumulator = 0.0
        self._time_steps = 0


class IndividualVariationGenerator:
    """
    Generates individual animal variation parameters.
    
    This creates realistic differences between individual animals while
    maintaining values within physiologically realistic ranges.
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the variation generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
    
    def generate_baseline_temperature(self) -> float:
        """
        Generate individual baseline body temperature.
        
        Normal range for cattle: 38.0-39.0°C
        
        Returns:
            Baseline temperature in °C
        """
        return self.rng.uniform(38.0, 39.0)
    
    def generate_activity_multiplier(self) -> float:
        """
        Generate individual activity level multiplier.
        
        Some animals are naturally more or less active.
        Range: 0.7-1.3 (70% to 130% of standard activity)
        
        Returns:
            Activity multiplier
        """
        return self.rng.uniform(0.7, 1.3)
    
    def generate_body_size_factor(self) -> float:
        """
        Generate body size factor affecting acceleration magnitudes.
        
        Larger animals have different acceleration patterns.
        Range: 0.85-1.15
        
        Returns:
            Body size factor
        """
        return self.rng.uniform(0.85, 1.15)
    
    def generate_age_category(self) -> str:
        """
        Generate age category with realistic distribution.
        
        Returns:
            Age category: "calf", "juvenile", "adult", or "senior"
        """
        # Most animals are adults, fewer calves/juveniles/seniors
        categories = ["calf", "juvenile", "adult", "adult", "adult", "adult", "senior"]
        return self.rng.choice(categories)
    
    def generate_health_modifiers(self, is_healthy: bool = True) -> Dict[str, float]:
        """
        Generate health modifiers for simulating sick animals.
        
        Args:
            is_healthy: If False, generates parameters for a sick animal
            
        Returns:
            Dictionary with 'fever_offset' and 'lethargy_factor'
        """
        if is_healthy:
            return {
                'fever_offset': 0.0,
                'lethargy_factor': 1.0
            }
        else:
            # Sick animal: fever and reduced activity
            fever_offset = self.rng.uniform(0.5, 2.0)  # +0.5 to +2.0°C
            lethargy_factor = self.rng.uniform(0.3, 0.7)  # 30-70% of normal activity
            return {
                'fever_offset': fever_offset,
                'lethargy_factor': lethargy_factor
            }
