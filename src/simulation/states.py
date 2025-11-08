"""
Behavioral state generators for cattle sensor data simulation.

Each generator produces realistic sensor signatures for specific behavioral
states based on documented cattle behavior patterns.

Sensor axes:
- Fxa: Forward-backward acceleration (m/s²)
- Mya: Lateral (side-to-side) acceleration (m/s²)
- Rza: Vertical acceleration / body orientation (g)
- Sxg: Roll angular velocity (°/s)
- Lyg: Pitch angular velocity (°/s)
- Dzg: Yaw angular velocity (°/s)
- Temperature: Body temperature (°C)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SensorReading:
    """Single timestep of sensor data."""
    timestamp: float
    temperature: float
    fxa: float  # Forward-backward acceleration
    mya: float  # Lateral acceleration
    rza: float  # Vertical acceleration / orientation
    sxg: float  # Roll angular velocity
    lyg: float  # Pitch angular velocity
    dzg: float  # Yaw angular velocity


class BehavioralStateGenerator(ABC):
    """Base class for behavioral state generators."""
    
    def __init__(self, baseline_temperature: float = 38.5, sampling_rate: float = 1.0):
        """
        Initialize state generator.
        
        Args:
            baseline_temperature: Normal body temperature in °C (default: 38.5°C)
            sampling_rate: Samples per minute (default: 1.0 for 1 sample/minute)
        """
        self.baseline_temperature = baseline_temperature
        self.sampling_rate = sampling_rate
        self.time_step = 60.0 / sampling_rate  # seconds per sample
        
    @abstractmethod
    def generate(self, duration_minutes: float, start_time: float = 0.0) -> list[SensorReading]:
        """
        Generate sensor readings for this behavioral state.
        
        Args:
            duration_minutes: Duration of the state in minutes
            start_time: Starting timestamp in seconds
            
        Returns:
            List of SensorReading objects
        """
        pass
    
    @abstractmethod
    def get_typical_duration_range(self) -> Tuple[float, float]:
        """Return (min, max) typical duration in minutes for this state."""
        pass
    
    def sample_duration(self) -> float:
        """Sample a realistic duration for this state."""
        min_dur, max_dur = self.get_typical_duration_range()
        # Use triangular distribution with mode at 60% between min and max
        mode = min_dur + 0.6 * (max_dur - min_dur)
        return np.random.triangular(min_dur, mode, max_dur)


class LyingStateGenerator(BehavioralStateGenerator):
    """
    Generator for lying/resting state.
    
    Characteristics:
    - Rza: -0.5g to -1.0g (body horizontal/tilted)
    - Fxa, Mya: Near zero (±0.05g) with occasional small movements
    - Sxg, Lyg, Dzg: Minimal angular velocities (±5°/s)
    - Temperature: Baseline ± 0.2°C
    - Duration: 30-120 minutes per lying bout
    """
    
    def get_typical_duration_range(self) -> Tuple[float, float]:
        return (30.0, 120.0)
    
    def generate(self, duration_minutes: float, start_time: float = 0.0) -> list[SensorReading]:
        """Generate sensor data for lying state."""
        n_samples = int(duration_minutes * self.sampling_rate)
        readings = []
        
        # Base Rza value for this lying bout (between -0.5 and -1.0g)
        base_rza = np.random.uniform(-1.0, -0.5)
        
        for i in range(n_samples):
            timestamp = start_time + i * self.time_step
            
            # Temperature: baseline with small random walk
            temp = self.baseline_temperature + np.random.normal(0, 0.1)
            temp = np.clip(temp, self.baseline_temperature - 0.2, self.baseline_temperature + 0.2)
            
            # Rza: mostly stable with tiny variations
            rza = base_rza + np.random.normal(0, 0.02)
            
            # Fxa, Mya: near zero with occasional small movements
            fxa = np.random.normal(0, 0.02)
            if np.random.random() < 0.05:  # 5% chance of small movement
                fxa += np.random.normal(0, 0.03)
            fxa = np.clip(fxa, -0.05, 0.05)
            
            mya = np.random.normal(0, 0.02)
            if np.random.random() < 0.05:
                mya += np.random.normal(0, 0.03)
            mya = np.clip(mya, -0.05, 0.05)
            
            # Gyroscopes: minimal angular velocities
            sxg = np.random.normal(0, 2.0)
            sxg = np.clip(sxg, -5.0, 5.0)
            
            lyg = np.random.normal(0, 2.0)
            lyg = np.clip(lyg, -5.0, 5.0)
            
            dzg = np.random.normal(0, 2.0)
            dzg = np.clip(dzg, -5.0, 5.0)
            
            readings.append(SensorReading(
                timestamp=timestamp,
                temperature=temp,
                fxa=fxa,
                mya=mya,
                rza=rza,
                sxg=sxg,
                lyg=lyg,
                dzg=dzg
            ))
        
        return readings


class StandingStateGenerator(BehavioralStateGenerator):
    """
    Generator for standing state.
    
    Characteristics:
    - Rza: 0.7g to 0.95g (upright posture)
    - Fxa, Mya: Low variance (±0.1g) with occasional weight shifting
    - Sxg, Lyg, Dzg: Low angular velocities (±3°/s)
    - Temperature: Baseline ± 0.1°C
    - Duration: 5-30 minutes per standing bout
    """
    
    def get_typical_duration_range(self) -> Tuple[float, float]:
        return (5.0, 30.0)
    
    def generate(self, duration_minutes: float, start_time: float = 0.0) -> list[SensorReading]:
        """Generate sensor data for standing state."""
        n_samples = int(duration_minutes * self.sampling_rate)
        readings = []
        
        # Base Rza value for this standing bout
        base_rza = np.random.uniform(0.7, 0.95)
        
        for i in range(n_samples):
            timestamp = start_time + i * self.time_step
            
            # Temperature: baseline with minimal variation
            temp = self.baseline_temperature + np.random.normal(0, 0.05)
            temp = np.clip(temp, self.baseline_temperature - 0.1, self.baseline_temperature + 0.1)
            
            # Rza: stable upright with small variations
            rza = base_rza + np.random.normal(0, 0.03)
            rza = np.clip(rza, 0.7, 0.95)
            
            # Fxa: low variance with occasional weight shifting
            fxa = np.random.normal(0, 0.04)
            if np.random.random() < 0.1:  # 10% chance of weight shift
                fxa += np.random.normal(0, 0.06)
            fxa = np.clip(fxa, -0.1, 0.1)
            
            # Mya: similar to Fxa
            mya = np.random.normal(0, 0.04)
            if np.random.random() < 0.1:
                mya += np.random.normal(0, 0.06)
            mya = np.clip(mya, -0.1, 0.1)
            
            # Gyroscopes: low angular velocities
            sxg = np.random.normal(0, 1.5)
            sxg = np.clip(sxg, -3.0, 3.0)
            
            lyg = np.random.normal(0, 1.5)
            lyg = np.clip(lyg, -3.0, 3.0)
            
            dzg = np.random.normal(0, 1.5)
            dzg = np.clip(dzg, -3.0, 3.0)
            
            readings.append(SensorReading(
                timestamp=timestamp,
                temperature=temp,
                fxa=fxa,
                mya=mya,
                rza=rza,
                sxg=sxg,
                lyg=lyg,
                dzg=dzg
            ))
        
        return readings


class WalkingStateGenerator(BehavioralStateGenerator):
    """
    Generator for walking state.
    
    Characteristics:
    - Fxa: Rhythmic variance 0.3-0.8 m/s² at 0.5-1.5 Hz frequency
    - Rza: 0.7-0.9g (slight forward tilt)
    - Mya: Side-to-side oscillation 0.2-0.5 m/s²
    - Sxg, Dzg: Moderate angular velocities (10-30°/s) with gait rhythm
    - Temperature: Slight increase (+0.1-0.3°C) during extended walking
    - Duration: 2-15 minutes per walking bout
    """
    
    def get_typical_duration_range(self) -> Tuple[float, float]:
        return (2.0, 15.0)
    
    def generate(self, duration_minutes: float, start_time: float = 0.0) -> list[SensorReading]:
        """Generate sensor data for walking state."""
        n_samples = int(duration_minutes * self.sampling_rate)
        readings = []
        
        # Gait frequency for this walking bout (Hz)
        gait_frequency = np.random.uniform(0.5, 1.5)
        
        # Base Rza during walking
        base_rza = np.random.uniform(0.7, 0.9)
        
        # Temperature increase due to activity
        temp_increase = np.random.uniform(0.1, 0.3)
        
        for i in range(n_samples):
            timestamp = start_time + i * self.time_step
            time_in_seconds = i * self.time_step
            
            # Temperature: gradual increase during walking
            progress = min(1.0, i / max(1, n_samples - 1))
            temp = self.baseline_temperature + temp_increase * progress + np.random.normal(0, 0.05)
            
            # Rhythmic Fxa (forward acceleration with gait pattern)
            phase = 2 * np.pi * gait_frequency * time_in_seconds
            fxa_amplitude = np.random.uniform(0.3, 0.8)
            fxa = fxa_amplitude * np.sin(phase) + np.random.normal(0, 0.1)
            
            # Mya: side-to-side oscillation at half the gait frequency
            mya_amplitude = np.random.uniform(0.2, 0.5)
            mya = mya_amplitude * np.sin(0.5 * phase + np.pi/4) + np.random.normal(0, 0.05)
            
            # Rza: slight forward tilt with small variations
            rza = base_rza + np.random.normal(0, 0.05)
            rza = np.clip(rza, 0.7, 0.9)
            
            # Sxg: moderate angular velocity with gait rhythm
            sxg_amplitude = np.random.uniform(10, 30)
            sxg = sxg_amplitude * np.sin(phase + np.pi/3) + np.random.normal(0, 5)
            
            # Lyg: moderate pitch variations
            lyg = np.random.uniform(-15, 15) + np.random.normal(0, 5)
            
            # Dzg: yaw with gait rhythm
            dzg_amplitude = np.random.uniform(10, 30)
            dzg = dzg_amplitude * np.sin(phase - np.pi/4) + np.random.normal(0, 5)
            
            readings.append(SensorReading(
                timestamp=timestamp,
                temperature=temp,
                fxa=fxa,
                mya=mya,
                rza=rza,
                sxg=sxg,
                lyg=lyg,
                dzg=dzg
            ))
        
        return readings


class RuminatingStateGenerator(BehavioralStateGenerator):
    """
    Generator for ruminating state.
    
    Characteristics:
    - Mya: Chewing oscillations at 40-60 cycles/minute (0.67-1.0 Hz)
    - Lyg: Head bobbing pattern synchronized with chewing (±10-15°/s)
    - Can occur during lying (Rza < -0.5g) or standing (Rza > 0.7g)
    - Other axes maintain base state values
    - Temperature: Baseline
    - Duration: 20-60 minutes per rumination session
    """
    
    def __init__(self, baseline_temperature: float = 38.5, sampling_rate: float = 1.0,
                 base_posture: str = 'lying'):
        """
        Initialize ruminating state generator.
        
        Args:
            baseline_temperature: Normal body temperature in °C
            sampling_rate: Samples per minute
            base_posture: 'lying' or 'standing' - the posture during rumination
        """
        super().__init__(baseline_temperature, sampling_rate)
        self.base_posture = base_posture
        
        # Create appropriate base posture generator
        if base_posture == 'lying':
            self.posture_generator = LyingStateGenerator(baseline_temperature, sampling_rate)
        else:
            self.posture_generator = StandingStateGenerator(baseline_temperature, sampling_rate)
    
    def get_typical_duration_range(self) -> Tuple[float, float]:
        return (20.0, 60.0)
    
    def generate(self, duration_minutes: float, start_time: float = 0.0) -> list[SensorReading]:
        """Generate sensor data for ruminating state."""
        # First generate base posture readings
        base_readings = self.posture_generator.generate(duration_minutes, start_time)
        
        # Chewing frequency for this rumination session (cycles per minute)
        chewing_cpm = np.random.uniform(40, 60)
        chewing_freq_hz = chewing_cpm / 60.0  # Convert to Hz
        
        # Overlay chewing patterns
        modified_readings = []
        for i, reading in enumerate(base_readings):
            time_in_seconds = i * self.time_step
            phase = 2 * np.pi * chewing_freq_hz * time_in_seconds
            
            # Mya: add chewing oscillations
            chewing_amplitude = np.random.uniform(0.15, 0.25)
            mya = reading.mya + chewing_amplitude * np.sin(phase) + np.random.normal(0, 0.02)
            
            # Lyg: head bobbing synchronized with chewing
            lyg_amplitude = np.random.uniform(10, 15)
            lyg = lyg_amplitude * np.sin(phase + np.pi/6) + np.random.normal(0, 2)
            
            modified_readings.append(SensorReading(
                timestamp=reading.timestamp,
                temperature=reading.temperature,
                fxa=reading.fxa,
                mya=mya,
                rza=reading.rza,
                sxg=reading.sxg,
                lyg=lyg,
                dzg=reading.dzg
            ))
        
        return modified_readings


class FeedingStateGenerator(BehavioralStateGenerator):
    """
    Generator for feeding state.
    
    Characteristics:
    - Lyg: Negative pitch angle indicating head-down position (-20° to -45°)
    - Fxa: Moderate forward movement (0.1-0.3 m/s²) as animal approaches feed
    - Mya: Chewing pattern similar to ruminating but with more head movement variability
    - Rza: Standing orientation (0.7-0.9g)
    - Sxg, Dzg: Moderate variability as head moves side-to-side
    - Temperature: Baseline to +0.2°C
    - Duration: 15-45 minutes per feeding session
    """
    
    def get_typical_duration_range(self) -> Tuple[float, float]:
        return (15.0, 45.0)
    
    def generate(self, duration_minutes: float, start_time: float = 0.0) -> list[SensorReading]:
        """Generate sensor data for feeding state."""
        n_samples = int(duration_minutes * self.sampling_rate)
        readings = []
        
        # Base pitch angle (head down)
        base_lyg = np.random.uniform(-45, -20)
        
        # Chewing frequency (slightly variable during feeding)
        chewing_freq_hz = np.random.uniform(0.7, 1.1)  # 42-66 cycles/min
        
        # Base Rza for standing
        base_rza = np.random.uniform(0.7, 0.9)
        
        # Temperature increase
        temp_increase = np.random.uniform(0.0, 0.2)
        
        for i in range(n_samples):
            timestamp = start_time + i * self.time_step
            time_in_seconds = i * self.time_step
            
            # Temperature: baseline to slightly elevated
            progress = min(1.0, i / max(1, n_samples - 1))
            temp = self.baseline_temperature + temp_increase * progress + np.random.normal(0, 0.05)
            
            # Fxa: moderate forward movement with some variability
            fxa = np.random.uniform(0.1, 0.3) + np.random.normal(0, 0.05)
            
            # Mya: chewing pattern with high variability
            phase = 2 * np.pi * chewing_freq_hz * time_in_seconds
            mya_amplitude = np.random.uniform(0.15, 0.3)
            mya = mya_amplitude * np.sin(phase) + np.random.normal(0, 0.08)
            
            # Rza: standing posture
            rza = base_rza + np.random.normal(0, 0.05)
            rza = np.clip(rza, 0.7, 0.9)
            
            # Sxg: moderate side-to-side head roll
            sxg = np.random.uniform(-15, 15) + np.random.normal(0, 5)
            
            # Lyg: head-down position with chewing motion
            lyg_variation = 10 * np.sin(phase + np.pi/4)
            lyg = base_lyg + lyg_variation + np.random.normal(0, 3)
            lyg = np.clip(lyg, -45, -20)
            
            # Dzg: head turning side to side while feeding
            dzg = np.random.uniform(-20, 20) + np.random.normal(0, 5)
            
            readings.append(SensorReading(
                timestamp=timestamp,
                temperature=temp,
                fxa=fxa,
                mya=mya,
                rza=rza,
                sxg=sxg,
                lyg=lyg,
                dzg=dzg
            ))
        
        return readings


class StressBehaviorOverlay:
    """
    Overlay stress behavior patterns on existing sensor data.
    
    Characteristics:
    - Erratic Movement: High variance across all axes simultaneously
      (Fxa, Mya: >1.0 m/s², all gyroscopes >40°/s)
    - Rapid State Changes: Frequent transitions without normal duration patterns
    - Irregular Patterns: Loss of rhythmic characteristics
    - Duration: 5-20 minutes of elevated stress indicators
    """
    
    @staticmethod
    def get_typical_duration_range() -> Tuple[float, float]:
        return (5.0, 20.0)
    
    @staticmethod
    def apply_stress(readings: list[SensorReading], 
                     stress_intensity: float = 1.0) -> list[SensorReading]:
        """
        Apply stress behavior overlay to existing sensor readings.
        
        Args:
            readings: Original sensor readings
            stress_intensity: Multiplier for stress effects (0.0 to 1.0+)
            
        Returns:
            Modified sensor readings with stress patterns
        """
        stressed_readings = []
        
        for reading in readings:
            # High variance in accelerations
            fxa_stress = np.random.uniform(-1.5, 1.5) * stress_intensity
            mya_stress = np.random.uniform(-1.2, 1.2) * stress_intensity
            
            # Erratic angular velocities
            sxg_stress = np.random.uniform(-60, 60) * stress_intensity
            lyg_stress = np.random.uniform(-60, 60) * stress_intensity
            dzg_stress = np.random.uniform(-60, 60) * stress_intensity
            
            # Slight temperature increase from stress
            temp_stress = np.random.uniform(0, 0.3) * stress_intensity
            
            stressed_readings.append(SensorReading(
                timestamp=reading.timestamp,
                temperature=reading.temperature + temp_stress,
                fxa=reading.fxa + fxa_stress,
                mya=reading.mya + mya_stress,
                rza=reading.rza + np.random.normal(0, 0.1) * stress_intensity,
                sxg=reading.sxg + sxg_stress,
                lyg=reading.lyg + lyg_stress,
                dzg=reading.dzg + dzg_stress
            ))
        
        return stressed_readings
    
    @staticmethod
    def sample_duration() -> float:
        """Sample a realistic duration for stress behavior."""
        min_dur, max_dur = StressBehaviorOverlay.get_typical_duration_range()
        mode = min_dur + 0.3 * (max_dur - min_dur)  # Stress tends to be shorter
        return np.random.triangular(min_dur, mode, max_dur)
