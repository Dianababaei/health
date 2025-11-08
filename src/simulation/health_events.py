"""
Health Event Simulators

This module simulates realistic health events and conditions over time:
- Estrus cycles (fertility periods every ~21 days)
- Pregnancy progression (60+ days)
- Fever/illness episodes
- Heat stress events
- Sensor quality degradation

These events modify animal profiles and sensor readings to create realistic
long-term dataset scenarios with ground truth labels.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
from enum import Enum
import numpy as np

from .state_params import AnimalProfile


class HealthEventType(Enum):
    """Types of health events that can occur."""
    NONE = "none"
    ESTRUS = "estrus"
    PREGNANCY_INDICATION = "pregnancy_indication"
    FEVER = "fever"
    ILLNESS = "illness"
    HEAT_STRESS = "heat_stress"


class TemperatureStatus(Enum):
    """Temperature status classifications."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    FEVER = "fever"
    HEAT_STRESS = "heat_stress"
    DROPPING = "dropping"


class SensorQuality(Enum):
    """Sensor quality status."""
    NORMAL = "normal"
    NOISY = "noisy"
    MALFUNCTION = "malfunction"


@dataclass
class HealthEvent:
    """Represents a health event occurring over a time period."""
    event_type: HealthEventType
    start_time: datetime
    duration_hours: float
    severity: float = 1.0  # 0.0-1.0 scale
    
    @property
    def end_time(self) -> datetime:
        """Calculate end time of the event."""
        return self.start_time + timedelta(hours=self.duration_hours)
    
    def is_active(self, current_time: datetime) -> bool:
        """Check if event is active at given time."""
        return self.start_time <= current_time < self.end_time


@dataclass
class EstrusEvent(HealthEvent):
    """Estrus (heat) event - fertility period."""
    peak_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Set peak time to middle of event if not specified."""
        if self.peak_time is None:
            self.peak_time = self.start_time + timedelta(hours=self.duration_hours / 2)
    
    def get_temperature_modifier(self, current_time: datetime) -> float:
        """
        Get temperature modification for estrus.
        Temperature rises 0.3-0.6°C during estrus peak.
        """
        if not self.is_active(current_time):
            return 0.0
        
        # Calculate position in estrus cycle (0.0 to 1.0)
        hours_elapsed = (current_time - self.start_time).total_seconds() / 3600
        position = hours_elapsed / self.duration_hours
        
        # Gaussian-like curve centered at peak
        peak_modifier = 0.3 + (self.severity * 0.3)  # 0.3-0.6°C
        distance_from_peak = abs(position - 0.5) * 2  # 0 at center, 1 at edges
        modifier = peak_modifier * np.exp(-2 * distance_from_peak**2)
        
        return modifier
    
    def get_activity_modifier(self, current_time: datetime) -> float:
        """
        Get activity modification for estrus.
        Activity increases 1.2-1.5x during estrus.
        """
        if not self.is_active(current_time):
            return 1.0
        
        hours_elapsed = (current_time - self.start_time).total_seconds() / 3600
        position = hours_elapsed / self.duration_hours
        
        # Peak activity increase
        peak_increase = 1.2 + (self.severity * 0.3)
        distance_from_peak = abs(position - 0.5) * 2
        modifier = 1.0 + (peak_increase - 1.0) * np.exp(-2 * distance_from_peak**2)
        
        return modifier


@dataclass
class PregnancyState:
    """Represents ongoing pregnancy state."""
    conception_date: datetime
    gestation_days: int = 0
    is_confirmed: bool = False  # Confirmed after ~30 days
    
    def update(self, current_time: datetime):
        """Update pregnancy state based on current time."""
        self.gestation_days = (current_time - self.conception_date).days
        if self.gestation_days >= 30:
            self.is_confirmed = True
    
    def get_temperature_modifier(self) -> float:
        """Pregnancy causes slight temperature elevation (0.1-0.2°C)."""
        if self.gestation_days < 30:
            return 0.0
        return 0.15  # Stable slight elevation
    
    def get_activity_modifier(self) -> float:
        """Pregnancy causes gradual activity reduction."""
        if self.gestation_days < 30:
            return 1.0
        
        # Gradual reduction over pregnancy (283 days gestation)
        if self.gestation_days < 150:
            return 1.0 - (self.gestation_days - 30) * 0.001  # Slow decrease
        else:
            return 0.85 - (self.gestation_days - 150) * 0.0005  # More pronounced


@dataclass
class IllnessEvent(HealthEvent):
    """Illness/fever event."""
    fever_magnitude: float = 1.5  # °C above baseline
    lethargy_factor: float = 0.5  # Activity multiplier
    
    def get_temperature_modifier(self, current_time: datetime) -> float:
        """Get temperature increase due to fever."""
        if not self.is_active(current_time):
            return 0.0
        
        # Fever ramps up and down at edges
        hours_elapsed = (current_time - self.start_time).total_seconds() / 3600
        position = hours_elapsed / self.duration_hours
        
        # Trapezoidal shape: ramp up, plateau, ramp down
        if position < 0.1:  # Ramp up
            return self.fever_magnitude * (position / 0.1)
        elif position > 0.9:  # Ramp down
            return self.fever_magnitude * ((1.0 - position) / 0.1)
        else:  # Plateau
            return self.fever_magnitude
    
    def get_activity_modifier(self, current_time: datetime) -> float:
        """Get activity reduction due to lethargy."""
        if not self.is_active(current_time):
            return 1.0
        return self.lethargy_factor


@dataclass
class HeatStressEvent(HealthEvent):
    """Heat stress event from environmental conditions."""
    temperature_increase: float = 0.8  # °C
    
    def get_temperature_modifier(self, current_time: datetime) -> float:
        """Get temperature increase due to heat stress."""
        if not self.is_active(current_time):
            return 0.0
        return self.temperature_increase
    
    def get_activity_modifier(self, current_time: datetime) -> float:
        """Animals may become more active (restless) during heat stress."""
        if not self.is_active(current_time):
            return 1.0
        return 1.15  # Slight increase due to restlessness


@dataclass
class SensorDegradationEvent(HealthEvent):
    """Sensor quality degradation event."""
    noise_multiplier: float = 2.5  # Multiply normal noise by this factor
    
    def get_noise_multiplier(self, current_time: datetime) -> float:
        """Get noise multiplication factor."""
        if not self.is_active(current_time):
            return 1.0
        return self.noise_multiplier


class HealthEventSimulator:
    """
    Manages health events over time for realistic dataset generation.
    
    Generates estrus cycles, pregnancy, illness events, and sensor issues
    according to realistic timelines and probabilities.
    """
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 estrus_cycle_days: float = 21.0,
                 estrus_duration_hours: float = 18.0):
        """
        Initialize health event simulator.
        
        Args:
            seed: Random seed for reproducibility
            estrus_cycle_days: Days between estrus cycles (default 21)
            estrus_duration_hours: Duration of each estrus event (default 18)
        """
        self.rng = np.random.default_rng(seed)
        self.estrus_cycle_days = estrus_cycle_days
        self.estrus_duration_hours = estrus_duration_hours
        
        self.events: List[HealthEvent] = []
        self.pregnancy_state: Optional[PregnancyState] = None
    
    def generate_estrus_cycles(self,
                               start_time: datetime,
                               end_time: datetime,
                               conception_probability: float = 0.3) -> List[EstrusEvent]:
        """
        Generate estrus cycles for the simulation period.
        
        Args:
            start_time: Simulation start
            end_time: Simulation end
            conception_probability: Probability of conception per cycle
            
        Returns:
            List of EstrusEvent objects
        """
        estrus_events = []
        
        # Start first cycle at random offset (0-21 days into simulation)
        current_time = start_time + timedelta(
            days=self.rng.uniform(0, self.estrus_cycle_days)
        )
        
        while current_time < end_time:
            # Create estrus event
            duration = self.rng.normal(self.estrus_duration_hours, 3.0)
            duration = np.clip(duration, 12.0, 30.0)
            
            severity = self.rng.uniform(0.7, 1.0)
            
            event = EstrusEvent(
                event_type=HealthEventType.ESTRUS,
                start_time=current_time,
                duration_hours=duration,
                severity=severity
            )
            estrus_events.append(event)
            
            # Check for conception
            if self.pregnancy_state is None and self.rng.random() < conception_probability:
                # Conception occurs at peak of estrus
                self.pregnancy_state = PregnancyState(
                    conception_date=event.peak_time
                )
                # No more estrus cycles after conception
                break
            
            # Next cycle with some variation (18-24 days)
            cycle_length = self.rng.normal(self.estrus_cycle_days, 2.0)
            cycle_length = np.clip(cycle_length, 18.0, 24.0)
            current_time += timedelta(days=cycle_length)
        
        return estrus_events
    
    def generate_illness_events(self,
                                start_time: datetime,
                                end_time: datetime,
                                num_events: int = 0) -> List[IllnessEvent]:
        """
        Generate random illness/fever events.
        
        Args:
            start_time: Simulation start
            end_time: Simulation end
            num_events: Number of illness events to generate (0 for random)
            
        Returns:
            List of IllnessEvent objects
        """
        if num_events == 0:
            # Randomly determine 0-2 illness events
            duration_days = (end_time - start_time).days
            if duration_days < 30:
                num_events = 0 if self.rng.random() > 0.2 else 1
            else:
                num_events = self.rng.choice([0, 1, 1, 2])  # Bias toward 1
        
        illness_events = []
        total_hours = (end_time - start_time).total_seconds() / 3600
        
        for _ in range(num_events):
            # Random start time
            offset_hours = self.rng.uniform(0, total_hours * 0.8)  # Not too close to end
            event_start = start_time + timedelta(hours=offset_hours)
            
            # Duration: 24-72 hours typically
            duration = self.rng.uniform(24, 72)
            
            # Fever magnitude
            fever = self.rng.uniform(1.0, 2.0)
            
            # Lethargy
            lethargy = self.rng.uniform(0.4, 0.7)
            
            event = IllnessEvent(
                event_type=HealthEventType.ILLNESS,
                start_time=event_start,
                duration_hours=duration,
                severity=self.rng.uniform(0.6, 1.0),
                fever_magnitude=fever,
                lethargy_factor=lethargy
            )
            illness_events.append(event)
        
        return illness_events
    
    def generate_heat_stress_events(self,
                                    start_time: datetime,
                                    end_time: datetime,
                                    num_events: int = 0) -> List[HeatStressEvent]:
        """
        Generate heat stress events (environmental).
        
        Args:
            start_time: Simulation start
            end_time: Simulation end
            num_events: Number of heat stress events (0 for random)
            
        Returns:
            List of HeatStressEvent objects
        """
        if num_events == 0:
            duration_days = (end_time - start_time).days
            if duration_days < 30:
                num_events = 0 if self.rng.random() > 0.3 else 1
            else:
                num_events = self.rng.choice([0, 1, 2, 3])
        
        heat_events = []
        total_hours = (end_time - start_time).total_seconds() / 3600
        
        for _ in range(num_events):
            # Heat stress typically during afternoon hours
            day_offset = self.rng.uniform(0, total_hours / 24)
            event_start = start_time + timedelta(days=day_offset)
            # Adjust to afternoon (12-16h)
            event_start = event_start.replace(
                hour=int(self.rng.uniform(12, 16)),
                minute=0
            )
            
            # Duration: 2-8 hours
            duration = self.rng.uniform(2, 8)
            
            event = HeatStressEvent(
                event_type=HealthEventType.HEAT_STRESS,
                start_time=event_start,
                duration_hours=duration,
                severity=self.rng.uniform(0.5, 1.0),
                temperature_increase=self.rng.uniform(0.5, 1.2)
            )
            heat_events.append(event)
        
        return heat_events
    
    def generate_sensor_degradation_events(self,
                                          start_time: datetime,
                                          end_time: datetime) -> List[SensorDegradationEvent]:
        """
        Generate sensor quality degradation events.
        
        Args:
            start_time: Simulation start
            end_time: Simulation end
            
        Returns:
            List of SensorDegradationEvent objects
        """
        degradation_events = []
        total_hours = (end_time - start_time).total_seconds() / 3600
        
        # 5-10% of time should have degraded sensor quality
        target_degraded_hours = total_hours * self.rng.uniform(0.05, 0.10)
        
        # Generate 2-5 degradation events
        num_events = self.rng.integers(2, 6)
        avg_duration = target_degraded_hours / num_events
        
        for _ in range(num_events):
            offset_hours = self.rng.uniform(0, total_hours)
            event_start = start_time + timedelta(hours=offset_hours)
            
            # Duration varies around average
            duration = self.rng.exponential(avg_duration)
            duration = np.clip(duration, 0.5, total_hours * 0.1)  # Max 10% of total
            
            event = SensorDegradationEvent(
                event_type=HealthEventType.NONE,  # Not a health event
                start_time=event_start,
                duration_hours=duration,
                noise_multiplier=self.rng.uniform(2.0, 4.0)
            )
            degradation_events.append(event)
        
        return degradation_events
    
    def generate_all_events(self,
                           start_time: datetime,
                           end_time: datetime,
                           include_estrus: bool = True,
                           include_pregnancy: bool = True,
                           num_illness: int = 0,
                           num_heat_stress: int = 0) -> Dict[str, List]:
        """
        Generate all health events for the simulation period.
        
        Args:
            start_time: Simulation start
            end_time: Simulation end
            include_estrus: Include estrus cycles
            include_pregnancy: Allow pregnancy to occur
            num_illness: Number of illness events (0=auto)
            num_heat_stress: Number of heat stress events (0=auto)
            
        Returns:
            Dictionary with event types as keys and lists of events
        """
        events = {}
        
        if include_estrus:
            conception_prob = 0.3 if include_pregnancy else 0.0
            events['estrus'] = self.generate_estrus_cycles(
                start_time, end_time, conception_prob
            )
        else:
            events['estrus'] = []
        
        events['illness'] = self.generate_illness_events(
            start_time, end_time, num_illness
        )
        
        events['heat_stress'] = self.generate_heat_stress_events(
            start_time, end_time, num_heat_stress
        )
        
        events['sensor_degradation'] = self.generate_sensor_degradation_events(
            start_time, end_time
        )
        
        # Store all events
        self.events = (
            events['estrus'] +
            events['illness'] +
            events['heat_stress'] +
            events['sensor_degradation']
        )
        
        return events
    
    def get_active_events(self, current_time: datetime) -> List[HealthEvent]:
        """Get all events active at the given time."""
        return [e for e in self.events if e.is_active(current_time)]
    
    def get_profile_modifiers(self, current_time: datetime) -> Dict[str, float]:
        """
        Get all profile modifications for the current time.
        
        Returns:
            Dictionary with 'temperature_offset', 'activity_multiplier', 'noise_multiplier'
        """
        temp_offset = 0.0
        activity_mult = 1.0
        noise_mult = 1.0
        
        # Apply pregnancy state
        if self.pregnancy_state is not None:
            self.pregnancy_state.update(current_time)
            temp_offset += self.pregnancy_state.get_temperature_modifier()
            activity_mult *= self.pregnancy_state.get_activity_modifier()
        
        # Apply all active events
        for event in self.get_active_events(current_time):
            if isinstance(event, EstrusEvent):
                temp_offset += event.get_temperature_modifier(current_time)
                activity_mult *= event.get_activity_modifier(current_time)
            elif isinstance(event, IllnessEvent):
                temp_offset += event.get_temperature_modifier(current_time)
                activity_mult *= event.get_activity_modifier(current_time)
            elif isinstance(event, HeatStressEvent):
                temp_offset += event.get_temperature_modifier(current_time)
                activity_mult *= event.get_activity_modifier(current_time)
            elif isinstance(event, SensorDegradationEvent):
                noise_mult *= event.get_noise_multiplier(current_time)
        
        return {
            'temperature_offset': temp_offset,
            'activity_multiplier': activity_mult,
            'noise_multiplier': noise_mult
        }
    
    def get_health_event_label(self, current_time: datetime) -> HealthEventType:
        """
        Get the primary health event label for the current time.
        
        Priority: pregnancy > estrus > illness > heat_stress > none
        """
        # Check pregnancy first
        if self.pregnancy_state is not None:
            self.pregnancy_state.update(current_time)
            if self.pregnancy_state.is_confirmed:
                return HealthEventType.PREGNANCY_INDICATION
        
        # Check active events
        active_events = self.get_active_events(current_time)
        
        # Priority order
        for event_type in [HealthEventType.ESTRUS, HealthEventType.ILLNESS, 
                          HealthEventType.HEAT_STRESS]:
            for event in active_events:
                if event.event_type == event_type:
                    return event_type
        
        return HealthEventType.NONE
    
    def get_temperature_status(self, temperature: float, current_time: datetime) -> TemperatureStatus:
        """
        Determine temperature status based on value and context.
        
        Args:
            temperature: Current temperature reading
            current_time: Current timestamp
            
        Returns:
            TemperatureStatus enum
        """
        # Check for active heat stress
        for event in self.get_active_events(current_time):
            if isinstance(event, HeatStressEvent):
                return TemperatureStatus.HEAT_STRESS
        
        # Check for fever
        if temperature >= 39.5:
            return TemperatureStatus.FEVER
        elif temperature >= 39.0:
            return TemperatureStatus.ELEVATED
        elif temperature < 37.5:
            return TemperatureStatus.DROPPING
        else:
            return TemperatureStatus.NORMAL
    
    def get_sensor_quality(self, current_time: datetime) -> SensorQuality:
        """
        Determine sensor quality at current time.
        
        Args:
            current_time: Current timestamp
            
        Returns:
            SensorQuality enum
        """
        for event in self.get_active_events(current_time):
            if isinstance(event, SensorDegradationEvent):
                if event.noise_multiplier > 3.0:
                    return SensorQuality.MALFUNCTION
                else:
                    return SensorQuality.NOISY
        
        return SensorQuality.NORMAL
