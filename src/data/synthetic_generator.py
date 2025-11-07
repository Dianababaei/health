"""
Synthetic Data Generator for Livestock Health Monitoring

Generates realistic sensor data with circadian rhythms and daily activity patterns.
Supports temperature variation, behavior sequences, and sensor readings.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import sys
import os

# Add config to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.behavior_patterns import (
    BEHAVIORS, HOURLY_SCHEDULE, SEQUENCE_TEMPLATES,
    TRANSITION_MATRIX, MIN_BEHAVIOR_DURATION, MAX_BEHAVIOR_DURATION
)


class SyntheticDataGenerator:
    """
    Generator for synthetic livestock sensor data with realistic circadian patterns.
    
    Features:
    - Circadian temperature variation (sinusoidal 24-hour cycle)
    - Time-of-day dependent behavior patterns
    - Realistic daily activity sequences
    - Smooth behavior transitions
    - Sensor data generation (accelerometer, gyroscope)
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the synthetic data generator.
        
        Args:
            random_seed: Seed for reproducibility (optional)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Circadian rhythm parameters
        self.base_temp = 38.5  # Base body temperature in Celsius
        self.temp_amplitude = 0.75  # Temperature variation amplitude
        self.acrophase = 16  # Hour of peak temperature (typically 14-18h)
        
        # Sensor noise parameters
        self.temp_noise_std = 0.1
        self.accel_noise_std = 0.2
        self.gyro_noise_std = 0.15
        
        # Behavior-specific sensor characteristics
        self.behavior_profiles = self._initialize_behavior_profiles()
    
    def _initialize_behavior_profiles(self) -> Dict:
        """
        Define sensor characteristics for each behavior type.
        
        Returns:
            Dictionary mapping behaviors to sensor value ranges
        """
        return {
            'lying': {
                'accel_mean': {'x': 0.1, 'y': 0.1, 'z': 9.8},  # Z-axis high (vertical)
                'accel_std': {'x': 0.2, 'y': 0.2, 'z': 0.3},
                'gyro_mean': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'gyro_std': {'x': 0.1, 'y': 0.1, 'z': 0.1}
            },
            'standing': {
                'accel_mean': {'x': 0.2, 'y': 0.2, 'z': 9.5},
                'accel_std': {'x': 0.4, 'y': 0.4, 'z': 0.5},
                'gyro_mean': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'gyro_std': {'x': 0.3, 'y': 0.3, 'z': 0.3}
            },
            'walking': {
                'accel_mean': {'x': 2.0, 'y': 1.5, 'z': 9.0},
                'accel_std': {'x': 1.5, 'y': 1.2, 'z': 1.0},
                'gyro_mean': {'x': 0.5, 'y': 0.3, 'z': 0.2},
                'gyro_std': {'x': 0.8, 'y': 0.6, 'z': 0.5}
            },
            'feeding': {
                'accel_mean': {'x': 0.8, 'y': 1.2, 'z': 8.5},
                'accel_std': {'x': 0.8, 'y': 1.0, 'z': 0.7},
                'gyro_mean': {'x': 0.2, 'y': 0.8, 'z': 0.1},
                'gyro_std': {'x': 0.4, 'y': 1.0, 'z': 0.3}
            },
            'ruminating': {
                'accel_mean': {'x': 0.3, 'y': 0.6, 'z': 9.3},
                'accel_std': {'x': 0.3, 'y': 0.5, 'z': 0.4},
                'gyro_mean': {'x': 0.1, 'y': 0.4, 'z': 0.0},
                'gyro_std': {'x': 0.2, 'y': 0.5, 'z': 0.2}
            }
        }
    
    def calculate_circadian_temperature(
        self, 
        hour: float, 
        base_temp: Optional[float] = None,
        amplitude: Optional[float] = None,
        acrophase: Optional[float] = None
    ) -> float:
        """
        Calculate body temperature with circadian rhythm.
        
        Formula: temp_base + amplitude * sin(2π * (hour - acrophase) / 24)
        
        Args:
            hour: Hour of day (0-24, can be fractional)
            base_temp: Base temperature (default: self.base_temp)
            amplitude: Temperature variation amplitude (default: self.temp_amplitude)
            acrophase: Hour of peak temperature (default: self.acrophase)
        
        Returns:
            Body temperature in Celsius
        """
        base = base_temp if base_temp is not None else self.base_temp
        amp = amplitude if amplitude is not None else self.temp_amplitude
        acro = acrophase if acrophase is not None else self.acrophase
        
        # Sinusoidal variation over 24-hour period
        phase = 2 * np.pi * (hour - acro) / 24
        temp = base + amp * np.sin(phase)
        
        # Add small random noise
        temp += np.random.normal(0, self.temp_noise_std)
        
        return temp
    
    def add_circadian_rhythm(
        self, 
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temp_col: str = 'temperature'
    ) -> pd.DataFrame:
        """
        Apply circadian temperature rhythm to existing dataframe.
        
        Args:
            df: DataFrame with timestamps
            timestamp_col: Name of timestamp column
            temp_col: Name of temperature column to create/update
        
        Returns:
            DataFrame with circadian temperature applied
        """
        df = df.copy()
        
        # Convert timestamps to hours
        if pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            hours = df[timestamp_col].dt.hour + df[timestamp_col].dt.minute / 60
        else:
            # Assume timestamp is in minutes from start
            hours = (df[timestamp_col] / 60) % 24
        
        # Apply circadian temperature
        df[temp_col] = hours.apply(self.calculate_circadian_temperature)
        
        return df
    
    def generate_daily_sequence(
        self,
        template: str = 'typical',
        randomize: bool = True,
        randomization_factor: float = 0.2
    ) -> List[Tuple[int, int, str]]:
        """
        Generate a daily behavior sequence based on template.
        
        Args:
            template: Template name ('typical', 'high_activity', 'low_activity')
            randomize: Whether to add randomization to sequence
            randomization_factor: Amount of randomization (0-1)
        
        Returns:
            List of (start_minute, end_minute, behavior) tuples
        """
        if template not in SEQUENCE_TEMPLATES:
            raise ValueError(f"Template '{template}' not found. "
                           f"Available: {list(SEQUENCE_TEMPLATES.keys())}")
        
        base_sequence = SEQUENCE_TEMPLATES[template].copy()
        
        if not randomize:
            return base_sequence
        
        # Add randomization to make sequences more realistic
        randomized_sequence = []
        current_minute = 0
        
        for start, end, behavior in base_sequence:
            duration = end - start
            
            # Add random variation to duration
            variation = int(duration * randomization_factor * (np.random.random() - 0.5) * 2)
            new_duration = max(
                MIN_BEHAVIOR_DURATION[behavior],
                min(MAX_BEHAVIOR_DURATION[behavior], duration + variation)
            )
            
            # Ensure we don't exceed day boundary
            new_end = min(current_minute + new_duration, 1440)
            
            if new_end > current_minute:
                randomized_sequence.append((current_minute, new_end, behavior))
            
            current_minute = new_end
            
            if current_minute >= 1440:
                break
        
        # Fill any remaining time with appropriate behavior
        if current_minute < 1440:
            hour = current_minute // 60
            schedule = HOURLY_SCHEDULE[hour]
            behavior = self._sample_behavior_from_schedule(schedule)
            randomized_sequence.append((current_minute, 1440, behavior))
        
        return randomized_sequence
    
    def generate_probabilistic_sequence(
        self,
        duration_minutes: int = 1440,
        start_behavior: Optional[str] = None
    ) -> List[Tuple[int, int, str]]:
        """
        Generate behavior sequence using probabilistic sampling based on time-of-day.
        
        This method creates more varied sequences than template-based generation
        by sampling behaviors according to hourly schedules.
        
        Args:
            duration_minutes: Total duration in minutes (default: 1440 = 24 hours)
            start_behavior: Initial behavior (default: sample from schedule)
        
        Returns:
            List of (start_minute, end_minute, behavior) tuples
        """
        sequence = []
        current_minute = 0
        
        # Determine starting behavior
        if start_behavior is None:
            hour = 0
            schedule = HOURLY_SCHEDULE[hour]
            current_behavior = self._sample_behavior_from_schedule(schedule)
        else:
            current_behavior = start_behavior
        
        while current_minute < duration_minutes:
            hour = (current_minute // 60) % 24
            schedule = HOURLY_SCHEDULE[hour]
            
            # Determine behavior duration
            min_duration = MIN_BEHAVIOR_DURATION[current_behavior]
            max_duration = MAX_BEHAVIOR_DURATION[current_behavior]
            duration = np.random.randint(min_duration, max_duration + 1)
            
            # Adjust for time-of-day (longer behaviors during appropriate times)
            if schedule.get(current_behavior, 0) > 0.3:
                duration = int(duration * 1.2)  # Extend during peak times
            
            end_minute = min(current_minute + duration, duration_minutes)
            sequence.append((current_minute, end_minute, current_behavior))
            
            current_minute = end_minute
            
            if current_minute >= duration_minutes:
                break
            
            # Transition to next behavior
            next_hour = (current_minute // 60) % 24
            next_schedule = HOURLY_SCHEDULE[next_hour]
            
            # Use transition matrix weighted by time-of-day schedule
            current_behavior = self._transition_behavior(
                current_behavior, 
                next_schedule
            )
        
        return sequence
    
    def _sample_behavior_from_schedule(self, schedule: Dict[str, float]) -> str:
        """
        Sample a behavior from schedule probabilities.
        
        Args:
            schedule: Dictionary of behavior probabilities
        
        Returns:
            Sampled behavior
        """
        behaviors = list(schedule.keys())
        probabilities = list(schedule.values())
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        return np.random.choice(behaviors, p=probabilities)
    
    def _transition_behavior(
        self, 
        current_behavior: str, 
        time_schedule: Dict[str, float]
    ) -> str:
        """
        Determine next behavior using transition matrix and time-of-day schedule.
        
        Args:
            current_behavior: Current behavior
            time_schedule: Time-of-day behavior probabilities
        
        Returns:
            Next behavior
        """
        # Get transition probabilities
        transitions = TRANSITION_MATRIX.get(current_behavior, {})
        
        # Combine transition probabilities with time-of-day schedule
        combined_probs = {}
        for behavior in BEHAVIORS:
            trans_prob = transitions.get(behavior, 0.0)
            time_prob = time_schedule.get(behavior, 0.0)
            # Weight: 60% transition matrix, 40% time-of-day schedule
            combined_probs[behavior] = 0.6 * trans_prob + 0.4 * time_prob
        
        # Normalize and sample
        total = sum(combined_probs.values())
        if total == 0:
            return self._sample_behavior_from_schedule(time_schedule)
        
        behaviors = list(combined_probs.keys())
        probabilities = [combined_probs[b] / total for b in behaviors]
        
        return np.random.choice(behaviors, p=probabilities)
    
    def generate_sensor_data(
        self,
        behavior: str,
        num_samples: int = 1
    ) -> Dict[str, np.ndarray]:
        """
        Generate sensor readings for a specific behavior.
        
        Args:
            behavior: Behavior type
            num_samples: Number of samples to generate
        
        Returns:
            Dictionary with 'accel' and 'gyro' arrays
        """
        if behavior not in self.behavior_profiles:
            raise ValueError(f"Unknown behavior: {behavior}")
        
        profile = self.behavior_profiles[behavior]
        
        # Generate accelerometer data (Fxa, Mya, Rza)
        accel_x = np.random.normal(
            profile['accel_mean']['x'],
            profile['accel_std']['x'],
            num_samples
        )
        accel_y = np.random.normal(
            profile['accel_mean']['y'],
            profile['accel_std']['y'],
            num_samples
        )
        accel_z = np.random.normal(
            profile['accel_mean']['z'],
            profile['accel_std']['z'],
            num_samples
        )
        
        # Generate gyroscope data (Sxg, Lyg, Dzg)
        gyro_x = np.random.normal(
            profile['gyro_mean']['x'],
            profile['gyro_std']['x'],
            num_samples
        )
        gyro_y = np.random.normal(
            profile['gyro_mean']['y'],
            profile['gyro_std']['y'],
            num_samples
        )
        gyro_z = np.random.normal(
            profile['gyro_mean']['z'],
            profile['gyro_std']['z'],
            num_samples
        )
        
        return {
            'Fxa': accel_x,
            'Mya': accel_y,
            'Rza': accel_z,
            'Sxg': gyro_x,
            'Lyg': gyro_y,
            'Dzg': gyro_z
        }
    
    def generate_dataset(
        self,
        num_days: int = 1,
        start_date: Optional[datetime] = None,
        sampling_interval_minutes: int = 1,
        sequence_type: str = 'probabilistic',
        template: str = 'typical',
        animal_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a complete synthetic dataset with circadian patterns.
        
        Args:
            num_days: Number of days to generate
            start_date: Starting date/time (default: now)
            sampling_interval_minutes: Data sampling interval (default: 1 minute)
            sequence_type: 'probabilistic' or 'template'
            template: Template name if using template-based generation
            animal_id: Animal identifier (optional)
        
        Returns:
            DataFrame with complete synthetic sensor data
        """
        if start_date is None:
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        all_data = []
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            
            # Generate daily behavior sequence
            if sequence_type == 'probabilistic':
                sequence = self.generate_probabilistic_sequence()
            else:
                sequence = self.generate_daily_sequence(template=template)
            
            # Generate data for each behavior segment
            for start_min, end_min, behavior in sequence:
                duration_minutes = end_min - start_min
                num_samples = duration_minutes // sampling_interval_minutes
                
                if num_samples == 0:
                    continue
                
                # Generate timestamps
                timestamps = [
                    current_date + timedelta(minutes=start_min + i * sampling_interval_minutes)
                    for i in range(num_samples)
                ]
                
                # Generate sensor data
                sensor_data = self.generate_sensor_data(behavior, num_samples)
                
                # Generate temperature with circadian rhythm
                hours = np.array([
                    (start_min + i * sampling_interval_minutes) / 60
                    for i in range(num_samples)
                ])
                temperatures = np.array([
                    self.calculate_circadian_temperature(h) for h in hours
                ])
                
                # Compile segment data
                for i in range(num_samples):
                    row = {
                        'timestamp': timestamps[i],
                        'temperature': temperatures[i],
                        'Fxa': sensor_data['Fxa'][i],
                        'Mya': sensor_data['Mya'][i],
                        'Rza': sensor_data['Rza'][i],
                        'Sxg': sensor_data['Sxg'][i],
                        'Lyg': sensor_data['Lyg'][i],
                        'Dzg': sensor_data['Dzg'][i],
                        'behavior': behavior
                    }
                    
                    if animal_id is not None:
                        row['animal_id'] = animal_id
                    
                    all_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Reorder columns
        columns = ['timestamp', 'temperature', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg', 'behavior']
        if animal_id is not None:
            columns.insert(0, 'animal_id')
        
        df = df[columns]
        
        return df
    
    def set_circadian_parameters(
        self,
        base_temp: Optional[float] = None,
        amplitude: Optional[float] = None,
        acrophase: Optional[float] = None
    ):
        """
        Update circadian rhythm parameters.
        
        Args:
            base_temp: Base body temperature (~38.5°C)
            amplitude: Temperature variation amplitude (0.5-1.0°C)
            acrophase: Hour of peak temperature (typically 14-18h)
        """
        if base_temp is not None:
            self.base_temp = base_temp
        if amplitude is not None:
            self.temp_amplitude = amplitude
        if acrophase is not None:
            self.acrophase = acrophase


# Convenience function
def generate_synthetic_data(
    num_days: int = 1,
    animal_id: Optional[str] = None,
    template: str = 'typical',
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Quick function to generate synthetic data.
    
    Args:
        num_days: Number of days to generate
        animal_id: Animal identifier
        template: Sequence template to use
        random_seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic sensor data
    """
    generator = SyntheticDataGenerator(random_seed=random_seed)
    return generator.generate_dataset(
        num_days=num_days,
        animal_id=animal_id,
        sequence_type='probabilistic',
        template=template
    )
