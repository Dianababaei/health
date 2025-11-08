"""
Temporal Pattern Management

This module manages circadian rhythms and time-of-day behavioral preferences
for realistic cattle behavior simulation. Cattle exhibit strong daily patterns:
- Increased lying during night (22:00-06:00)
- Feeding peaks morning (06:00-10:00) and evening (16:00-20:00)
- More activity during daylight hours
- Temperature follows circadian rhythm

References:
- Cattle circadian rhythm studies
- Time-budgets of cattle behavior
- Diurnal activity patterns in ruminants
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np
from .state_params import BehavioralState


class TemporalPatternManager:
    """
    Manages temporal patterns including circadian rhythms and time-of-day effects.
    
    This class provides methods to:
    - Calculate time-of-day modulation of state transition probabilities
    - Apply circadian rhythm effects to physiological parameters
    - Determine appropriate behavioral patterns for different times
    """
    
    def __init__(self):
        """Initialize the temporal pattern manager."""
        pass
    
    @staticmethod
    def get_hour_of_day(timestamp: datetime) -> float:
        """
        Extract hour of day from timestamp as float (0.0 to 23.99).
        
        Args:
            timestamp: DateTime object
            
        Returns:
            Hour of day as float (e.g., 13.5 = 1:30 PM)
        """
        return timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
    
    @staticmethod
    def is_night_time(hour: float) -> bool:
        """
        Check if given hour is considered night time.
        
        Night period: 22:00 to 06:00
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            True if night time
        """
        return hour >= 22.0 or hour < 6.0
    
    @staticmethod
    def is_feeding_time(hour: float) -> bool:
        """
        Check if given hour is a typical feeding period.
        
        Feeding periods: 06:00-10:00 (morning) and 16:00-20:00 (evening)
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            True if feeding time
        """
        return (6.0 <= hour < 10.0) or (16.0 <= hour < 20.0)
    
    @staticmethod
    def get_lying_preference(hour: float) -> float:
        """
        Get lying behavior preference multiplier for given time.
        
        Higher values during night (22:00-06:00), lower during day.
        Range: 0.3 (midday) to 2.5 (midnight)
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            Lying preference multiplier
        """
        # Peak lying at midnight (2.5x), minimum at noon (0.3x)
        # Using cosine function centered at noon
        preference = 1.4 + 1.1 * np.cos(2 * np.pi * (hour - 12) / 24)
        return preference
    
    @staticmethod
    def get_feeding_preference(hour: float) -> float:
        """
        Get feeding behavior preference multiplier for given time.
        
        Peaks during morning (06:00-10:00) and evening (16:00-20:00).
        Range: 0.3 to 2.0
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            Feeding preference multiplier
        """
        # Two peaks: morning (08:00) and evening (18:00)
        morning_peak = 1.0 + 0.9 * np.exp(-((hour - 8.0) ** 2) / 8.0)
        evening_peak = 1.0 + 0.9 * np.exp(-((hour - 18.0) ** 2) / 8.0)
        
        # Combined preference
        preference = max(morning_peak, evening_peak)
        
        # Minimum during night
        if TemporalPatternManager.is_night_time(hour):
            preference *= 0.3
        
        return np.clip(preference, 0.3, 2.0)
    
    @staticmethod
    def get_activity_preference(hour: float) -> float:
        """
        Get general activity preference multiplier for given time.
        
        Higher during day (06:00-20:00), lower at night.
        Range: 0.4 (night) to 1.5 (day)
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            Activity preference multiplier
        """
        # Higher activity during daylight hours
        if 6.0 <= hour < 20.0:
            # Peak activity around midday
            preference = 1.2 + 0.3 * np.sin(2 * np.pi * (hour - 6.0) / 14.0)
        else:
            # Low activity at night
            preference = 0.4 + 0.2 * np.random.random()
        
        return preference
    
    def get_state_preference_multipliers(self, hour: float) -> Dict[BehavioralState, float]:
        """
        Get preference multipliers for all behavioral states at given time.
        
        These multipliers modify the base transition probabilities to create
        realistic time-of-day patterns.
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            Dictionary mapping states to preference multipliers
        """
        return {
            BehavioralState.LYING: self.get_lying_preference(hour),
            BehavioralState.STANDING: 1.0,  # Standing is neutral
            BehavioralState.WALKING: self.get_activity_preference(hour),
            BehavioralState.RUMINATING: 1.0 + 0.3 * self.get_lying_preference(hour) / 2.0,
            BehavioralState.FEEDING: self.get_feeding_preference(hour),
        }
    
    @staticmethod
    def apply_circadian_temperature_effect(base_temperature: float, hour: float) -> float:
        """
        Apply circadian rhythm effect to body temperature.
        
        Body temperature naturally varies with circadian rhythm:
        - Lower during night (nadir around 04:00)
        - Higher during day (peak around 16:00)
        - Variation amplitude: ~0.3-0.5°C
        
        Args:
            base_temperature: Base temperature in °C
            hour: Hour of day (0-23.99)
            
        Returns:
            Temperature with circadian effect
        """
        # Peak at 16:00, nadir at 04:00
        # Using sine wave with 24-hour period
        amplitude = 0.4
        phase_shift = 16.0  # Peak at 16:00
        
        circadian_effect = amplitude * np.sin(2 * np.pi * (hour - phase_shift + 6) / 24)
        
        return base_temperature + circadian_effect
    
    @staticmethod
    def get_transition_time_scale(hour: float) -> float:
        """
        Get time scale factor for state transitions.
        
        During active periods (day), transitions happen more frequently.
        During rest periods (night), transitions are less frequent.
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            Time scale factor (1.0 = normal, <1.0 = slower transitions)
        """
        if TemporalPatternManager.is_night_time(hour):
            # Slower transitions at night
            return 0.5
        else:
            # Normal or faster transitions during day
            return 1.0 + 0.2 * np.sin(2 * np.pi * (hour - 12) / 24)
    
    @staticmethod
    def calculate_day_period(hour: float) -> str:
        """
        Categorize time of day into periods.
        
        Args:
            hour: Hour of day (0-23.99)
            
        Returns:
            Period name: "night", "early_morning", "morning", "midday", 
                        "afternoon", "evening"
        """
        if hour < 6.0 or hour >= 22.0:
            return "night"
        elif 6.0 <= hour < 8.0:
            return "early_morning"
        elif 8.0 <= hour < 12.0:
            return "morning"
        elif 12.0 <= hour < 14.0:
            return "midday"
        elif 14.0 <= hour < 18.0:
            return "afternoon"
        elif 18.0 <= hour < 22.0:
            return "evening"
        else:
            return "night"
    
    def get_period_statistics(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """
        Calculate statistics about time periods in a simulation range.
        
        Args:
            start_time: Simulation start timestamp
            end_time: Simulation end timestamp
            
        Returns:
            Dictionary with period statistics (hours in each period)
        """
        period_counts = {
            "night": 0.0,
            "early_morning": 0.0,
            "morning": 0.0,
            "midday": 0.0,
            "afternoon": 0.0,
            "evening": 0.0
        }
        
        # Sample at 1-hour intervals
        current = start_time
        while current < end_time:
            hour = self.get_hour_of_day(current)
            period = self.calculate_day_period(hour)
            period_counts[period] += 1.0
            current += timedelta(hours=1)
        
        return period_counts


class SeasonalPatternManager:
    """
    Manages seasonal variations in behavior patterns (optional, for extended simulations).
    
    Seasonal effects include:
    - Temperature variations
    - Day length effects on activity
    - Breeding season behavioral changes
    """
    
    def __init__(self, latitude: float = 40.0):
        """
        Initialize seasonal pattern manager.
        
        Args:
            latitude: Latitude for day length calculations (degrees)
        """
        self.latitude = latitude
    
    def get_day_length(self, date: datetime) -> float:
        """
        Calculate approximate day length for given date.
        
        Args:
            date: Date to calculate day length for
            
        Returns:
            Day length in hours
        """
        # Simplified day length calculation
        day_of_year = date.timetuple().tm_yday
        
        # Using simplified formula (not exact but sufficient for simulation)
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        lat_rad = np.radians(self.latitude)
        decl_rad = np.radians(declination)
        
        # Hour angle at sunrise/sunset
        try:
            cos_hour_angle = -np.tan(lat_rad) * np.tan(decl_rad)
            cos_hour_angle = np.clip(cos_hour_angle, -1, 1)
            hour_angle = np.arccos(cos_hour_angle)
            day_length = 2 * np.degrees(hour_angle) / 15.0
        except:
            day_length = 12.0  # Default to 12 hours if calculation fails
        
        return day_length
    
    def get_seasonal_activity_modifier(self, date: datetime) -> float:
        """
        Get seasonal activity level modifier.
        
        Longer days = more activity opportunities.
        
        Args:
            date: Current date
            
        Returns:
            Activity modifier (0.8 to 1.2)
        """
        day_length = self.get_day_length(date)
        # Normalize around 12 hours
        modifier = 0.8 + 0.4 * (day_length - 8) / 8
        return np.clip(modifier, 0.8, 1.2)
