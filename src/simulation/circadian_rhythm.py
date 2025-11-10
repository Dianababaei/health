"""
Circadian Rhythm Generator for Cattle Body Temperature

This module generates realistic circadian (24-hour) temperature patterns
based on cattle physiology literature.

References:
- Bitman et al. (1984): Core body temperature rhythm in dairy cattle
- Kendall & Webster (2009): Temperature patterns in grazing animals
"""

import numpy as np
from typing import Tuple


class CircadianRhythmGenerator:
    """
    Generates circadian temperature patterns for cattle.

    Normal cattle temperature follows a ~24-hour sinusoidal pattern:
    - Baseline: 38.5°C (mean)
    - Peak: 6-8 PM (38.8-39.2°C)
    - Trough: 4-6 AM (38.3-38.7°C)
    - Amplitude: ±0.5°C for healthy animals
    """

    def __init__(
        self,
        baseline_temp: float = 38.5,
        amplitude: float = 0.5,
        peak_hour: float = 19.0,  # 7 PM
        noise_std: float = 0.1
    ):
        """
        Initialize circadian rhythm generator.

        Args:
            baseline_temp: Mean body temperature (°C), default 38.5°C
            amplitude: Half of peak-to-trough variation (°C), default ±0.5°C
            peak_hour: Hour of day for temperature peak (0-24), default 19.0 (7 PM)
            noise_std: Standard deviation of measurement noise (°C), default 0.1°C
        """
        self.baseline_temp = baseline_temp
        self.amplitude = amplitude
        self.peak_hour = peak_hour
        self.noise_std = noise_std

        # Validate parameters
        if not 35.0 <= baseline_temp <= 42.0:
            raise ValueError(f"Baseline temperature {baseline_temp}°C outside physiological range (35-42°C)")
        if amplitude < 0 or amplitude > 2.0:
            raise ValueError(f"Amplitude {amplitude}°C unrealistic (should be 0-2°C)")
        if not 0 <= peak_hour <= 24:
            raise ValueError(f"Peak hour {peak_hour} must be between 0-24")

    def generate(
        self,
        duration_minutes: int,
        start_time_minutes: float = 0.0,
        random_seed: int = None
    ) -> np.ndarray:
        """
        Generate circadian temperature pattern.

        Args:
            duration_minutes: Length of time series in minutes
            start_time_minutes: Starting time offset in minutes (0 = midnight)
            random_seed: Random seed for reproducibility

        Returns:
            Array of temperatures (°C) with shape (duration_minutes,)
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        # Time array in minutes
        time_minutes = np.arange(duration_minutes) + start_time_minutes

        # Convert to hours for circadian calculation
        time_hours = time_minutes / 60.0

        # Generate circadian pattern using sinusoidal function
        # Peak at peak_hour, trough 12 hours later
        # Phase shift: peak_hour corresponds to phase = 0 at maximum
        phase_shift = self.peak_hour * (2 * np.pi / 24.0)
        circadian_pattern = self.amplitude * np.sin(
            (2 * np.pi / 24.0) * time_hours - phase_shift + (np.pi / 2)
        )

        # Add baseline
        temperature = self.baseline_temp + circadian_pattern

        # Add measurement noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=duration_minutes)
            temperature += noise

        # Ensure physiological limits
        temperature = np.clip(temperature, 35.0, 42.0)

        return temperature

    def get_temperature_at_time(self, hour_of_day: float) -> float:
        """
        Get expected temperature at a specific hour of day (without noise).

        Args:
            hour_of_day: Hour (0-24, can be fractional)

        Returns:
            Expected temperature (°C)
        """
        phase_shift = self.peak_hour * (2 * np.pi / 24.0)
        circadian_value = self.amplitude * np.sin(
            (2 * np.pi / 24.0) * hour_of_day - phase_shift + (np.pi / 2)
        )
        return self.baseline_temp + circadian_value

    def get_peak_trough_times(self) -> Tuple[float, float]:
        """
        Get the times (hours) of peak and trough temperatures.

        Returns:
            Tuple of (peak_hour, trough_hour)
        """
        trough_hour = (self.peak_hour + 12.0) % 24.0
        return (self.peak_hour, trough_hour)


def create_normal_circadian(
    duration_minutes: int,
    start_time_minutes: float = 0.0,
    random_seed: int = None
) -> np.ndarray:
    """
    Convenience function to create normal circadian temperature pattern.

    Uses standard parameters for healthy cattle:
    - Baseline: 38.5°C
    - Amplitude: ±0.5°C
    - Peak: 7 PM
    - Noise: ±0.1°C

    Args:
        duration_minutes: Length of time series
        start_time_minutes: Starting time offset (0 = midnight)
        random_seed: Random seed for reproducibility

    Returns:
        Temperature array (°C)
    """
    generator = CircadianRhythmGenerator(
        baseline_temp=38.5,
        amplitude=0.5,
        peak_hour=19.0,
        noise_std=0.1
    )
    return generator.generate(duration_minutes, start_time_minutes, random_seed)


def create_fever_circadian(
    duration_minutes: int,
    start_time_minutes: float = 0.0,
    baseline_temp: float = 40.0,
    random_seed: int = None
) -> np.ndarray:
    """
    Create circadian pattern for fever condition.

    Fever maintains circadian rhythm but with:
    - Elevated baseline (>39.5°C)
    - Larger amplitude (±0.7°C)

    Args:
        duration_minutes: Length of time series
        start_time_minutes: Starting time offset
        baseline_temp: Elevated baseline temperature (default 40.0°C)
        random_seed: Random seed

    Returns:
        Temperature array (°C)
    """
    generator = CircadianRhythmGenerator(
        baseline_temp=baseline_temp,
        amplitude=0.7,  # Larger amplitude during fever
        peak_hour=19.0,
        noise_std=0.15  # Slightly higher noise
    )
    return generator.generate(duration_minutes, start_time_minutes, random_seed)


def create_pregnancy_circadian(
    duration_minutes: int,
    start_time_minutes: float = 0.0,
    random_seed: int = None
) -> np.ndarray:
    """
    Create circadian pattern for pregnancy condition.

    Pregnancy shows:
    - Slightly elevated baseline (+0.1-0.2°C)
    - Dampened rhythm (±0.3-0.4°C)
    - Lower noise (stable temperature)

    Args:
        duration_minutes: Length of time series
        start_time_minutes: Starting time offset
        random_seed: Random seed

    Returns:
        Temperature array (°C)
    """
    generator = CircadianRhythmGenerator(
        baseline_temp=38.6,  # Slightly elevated
        amplitude=0.35,  # Dampened rhythm
        peak_hour=19.0,
        noise_std=0.08  # Lower noise (stable)
    )
    return generator.generate(duration_minutes, start_time_minutes, random_seed)


if __name__ == "__main__":
    # Demo: Generate and display circadian patterns
    import matplotlib.pyplot as plt

    duration = 1440 * 3  # 3 days
    time_hours = np.arange(duration) / 60.0

    # Generate patterns
    normal = create_normal_circadian(duration, random_seed=42)
    fever = create_fever_circadian(duration, random_seed=42)
    pregnancy = create_pregnancy_circadian(duration, random_seed=42)

    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(time_hours, normal, label='Normal', alpha=0.7)
    plt.plot(time_hours, fever, label='Fever', alpha=0.7)
    plt.plot(time_hours, pregnancy, label='Pregnancy', alpha=0.7)

    plt.axhline(y=38.5, color='gray', linestyle='--', alpha=0.3, label='Normal baseline')
    plt.axhline(y=39.5, color='red', linestyle='--', alpha=0.3, label='Fever threshold')

    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.title('Circadian Temperature Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
