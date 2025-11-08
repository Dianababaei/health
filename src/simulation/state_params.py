"""
Behavioral State Parameter Definitions

This module defines the sensor signatures for each cattle behavioral state
based on research literature. Each state has characteristic ranges for all
7 sensor parameters: Temperature, Fxa, Mya, Rza, Sxg, Lyg, Dzg.

Literature-based sensor signatures:
- Lying: Rza < -0.5g (vertical orientation), minimal movement
- Standing: Rza > 0.7g, low motion, stable angular velocities
- Walking: Rhythmic patterns (0.5-1.5 Hz), moderate motion
- Ruminating: Mya oscillations 40-60 cycles/min, head bobbing
- Feeding: Negative pitch (head down), forward movement

References:
- Cattle posture and activity patterns (various research papers)
- Accelerometer-based behavior classification studies
- Gyroscope patterns in ruminant behavior
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Optional
import numpy as np


class BehavioralState(Enum):
    """Enumeration of cattle behavioral states."""
    LYING = "lying"
    STANDING = "standing"
    WALKING = "walking"
    RUMINATING = "ruminating"
    FEEDING = "feeding"


@dataclass
class SensorRange:
    """Defines min, max, and typical values for a sensor parameter."""
    min_value: float
    max_value: float
    mean: float
    std: float  # Standard deviation for normal distribution
    
    def sample(self, rng: np.random.Generator) -> float:
        """Sample a value from this range using normal distribution, clipped to min/max."""
        value = rng.normal(self.mean, self.std)
        return np.clip(value, self.min_value, self.max_value)


@dataclass
class SensorSignature:
    """
    Complete sensor signature for a behavioral state.
    
    All accelerometer values in g (gravitational units).
    All gyroscope values in degrees/second.
    Temperature in degrees Celsius.
    """
    temperature: SensorRange  # °C
    fxa: SensorRange  # Forward-backward acceleration (g)
    mya: SensorRange  # Lateral acceleration (g)
    rza: SensorRange  # Vertical acceleration (g)
    sxg: SensorRange  # Roll angular velocity (°/s)
    lyg: SensorRange  # Pitch angular velocity (°/s)
    dzg: SensorRange  # Yaw angular velocity (°/s)
    
    # Optional: rhythmic patterns (for walking, ruminating)
    rhythmic_frequency: Optional[float] = None  # Hz
    rhythmic_amplitude_scale: Optional[float] = None


# Literature-based sensor signatures for each behavioral state
BEHAVIORAL_SIGNATURES: Dict[BehavioralState, SensorSignature] = {
    
    BehavioralState.LYING: SensorSignature(
        temperature=SensorRange(38.0, 39.2, 38.5, 0.2),
        fxa=SensorRange(-0.2, 0.2, 0.0, 0.05),  # Minimal forward movement
        mya=SensorRange(-0.15, 0.15, 0.0, 0.04),  # Minimal lateral movement
        rza=SensorRange(-1.0, -0.5, -0.8, 0.1),  # Negative = lying orientation
        sxg=SensorRange(-5.0, 5.0, 0.0, 2.0),  # Low roll velocity
        lyg=SensorRange(-5.0, 5.0, 0.0, 2.0),  # Low pitch velocity
        dzg=SensorRange(-3.0, 3.0, 0.0, 1.5),  # Low yaw velocity
        rhythmic_frequency=None,
        rhythmic_amplitude_scale=None
    ),
    
    BehavioralState.STANDING: SensorSignature(
        temperature=SensorRange(38.2, 39.3, 38.6, 0.2),
        fxa=SensorRange(-0.15, 0.15, 0.0, 0.04),  # Minimal movement
        mya=SensorRange(-0.15, 0.15, 0.0, 0.04),
        rza=SensorRange(0.7, 1.0, 0.9, 0.08),  # Upright orientation
        sxg=SensorRange(-8.0, 8.0, 0.0, 3.0),  # Small stabilizing movements
        lyg=SensorRange(-8.0, 8.0, 0.0, 3.0),
        dzg=SensorRange(-6.0, 6.0, 0.0, 2.5),
        rhythmic_frequency=None,
        rhythmic_amplitude_scale=None
    ),
    
    BehavioralState.WALKING: SensorSignature(
        temperature=SensorRange(38.3, 39.4, 38.7, 0.2),
        fxa=SensorRange(0.1, 0.8, 0.4, 0.15),  # Forward acceleration
        mya=SensorRange(-0.3, 0.3, 0.0, 0.1),  # Lateral sway
        rza=SensorRange(0.7, 0.95, 0.85, 0.08),  # Mostly upright with variation
        sxg=SensorRange(-15.0, 15.0, 0.0, 6.0),  # Increased roll
        lyg=SensorRange(-12.0, 12.0, 0.0, 5.0),  # Head bobbing
        dzg=SensorRange(-10.0, 10.0, 0.0, 4.0),  # Turning movements
        rhythmic_frequency=1.0,  # ~1 Hz gait frequency
        rhythmic_amplitude_scale=1.5
    ),
    
    BehavioralState.RUMINATING: SensorSignature(
        temperature=SensorRange(38.0, 39.2, 38.5, 0.2),
        # Rumination can occur while lying or standing - using intermediate values
        fxa=SensorRange(-0.15, 0.15, 0.0, 0.04),
        mya=SensorRange(-0.25, 0.25, 0.0, 0.12),  # Jaw movements (lateral)
        rza=SensorRange(-0.3, 0.95, 0.5, 0.3),  # Can be lying or standing
        sxg=SensorRange(-8.0, 8.0, 0.0, 3.0),
        lyg=SensorRange(-15.0, 15.0, 0.0, 6.0),  # Head bobbing from chewing
        dzg=SensorRange(-5.0, 5.0, 0.0, 2.0),
        rhythmic_frequency=0.83,  # ~50 cycles/min = 0.83 Hz
        rhythmic_amplitude_scale=2.0  # Strong rhythmic pattern
    ),
    
    BehavioralState.FEEDING: SensorSignature(
        temperature=SensorRange(38.2, 39.3, 38.6, 0.2),
        fxa=SensorRange(0.0, 0.4, 0.2, 0.1),  # Moderate forward movement
        mya=SensorRange(-0.2, 0.2, 0.0, 0.08),
        rza=SensorRange(0.4, 0.85, 0.65, 0.12),  # Head down position
        sxg=SensorRange(-10.0, 10.0, 0.0, 4.0),
        lyg=SensorRange(-25.0, -5.0, -15.0, 5.0),  # Negative pitch (head down)
        dzg=SensorRange(-8.0, 8.0, 0.0, 3.5),  # Side-to-side while eating
        rhythmic_frequency=0.5,  # Slower than walking
        rhythmic_amplitude_scale=0.8
    ),
}


@dataclass
class AnimalProfile:
    """
    Individual animal characteristics that modify baseline sensor values.
    
    This allows simulation of individual variation between animals while
    maintaining realistic behavioral patterns.
    """
    animal_id: str
    baseline_temperature: float = 38.5  # °C, normal range 38.0-39.0
    activity_multiplier: float = 1.0  # 0.7-1.3 range for individual variation
    body_size_factor: float = 1.0  # Affects acceleration magnitudes
    age_category: str = "adult"  # "calf", "juvenile", "adult", "senior"
    
    # Health modifiers (for simulating non-healthy animals)
    fever_offset: float = 0.0  # Add to temperature (e.g., +1.0 for fever)
    lethargy_factor: float = 1.0  # Reduces activity levels (< 1.0 = lethargic)
    
    def apply_temperature_modification(self, base_temp: float) -> float:
        """Apply individual variation to temperature."""
        # Individual baseline variation
        temp_offset = self.baseline_temperature - 38.5
        return base_temp + temp_offset + self.fever_offset
    
    def apply_activity_modification(self, base_activity: float) -> float:
        """Apply individual variation to activity-related sensors."""
        return base_activity * self.activity_multiplier * self.lethargy_factor * self.body_size_factor


def create_default_profile(animal_id: str, rng: Optional[np.random.Generator] = None) -> AnimalProfile:
    """
    Create an animal profile with realistic random variation.
    
    Args:
        animal_id: Unique identifier for the animal
        rng: Random number generator for reproducibility
        
    Returns:
        AnimalProfile with randomized but realistic parameters
    """
    if rng is None:
        rng = np.random.default_rng()
    
    return AnimalProfile(
        animal_id=animal_id,
        baseline_temperature=rng.uniform(38.0, 39.0),
        activity_multiplier=rng.uniform(0.7, 1.3),
        body_size_factor=rng.uniform(0.85, 1.15),
        age_category=rng.choice(["adult", "adult", "adult", "juvenile", "senior"]),
        fever_offset=0.0,
        lethargy_factor=1.0
    )


def get_state_signature(state: BehavioralState) -> SensorSignature:
    """Get the sensor signature for a given behavioral state."""
    return BEHAVIORAL_SIGNATURES[state]


def validate_sensor_values(temperature: float, fxa: float, mya: float, rza: float,
                          sxg: float, lyg: float, dzg: float) -> bool:
    """
    Validate that sensor values are within physically realistic ranges.
    
    Returns:
        True if all values are valid, False otherwise
    """
    # Temperature should be in viable range for cattle
    if not (36.0 <= temperature <= 42.0):
        return False
    
    # Accelerometer values should be within ±2g for normal behavior
    if not all(-2.0 <= val <= 2.0 for val in [fxa, mya, rza]):
        return False
    
    # Gyroscope values should be within ±50 degrees/second for normal behavior
    if not all(-50.0 <= val <= 50.0 for val in [sxg, lyg, dzg]):
        return False
    
    return True
