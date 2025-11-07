"""
Synthetic data generator for animal behavior monitoring.

Generates realistic sensor data with:
- 6 behavior patterns: lying, standing, walking, ruminating, feeding, stress
- Circadian temperature rhythms
- Smooth behavior transitions
- Multi-day continuous sequences
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pandas as pd


class BehaviorPattern:
    """Defines sensor characteristics for each behavior."""
    
    def __init__(
        self,
        name: str,
        temp_mean: float,
        temp_std: float,
        fxa_range: Tuple[float, float],
        mya_range: Tuple[float, float],
        rza_range: Tuple[float, float],
        sxg_range: Tuple[float, float],
        lyg_range: Tuple[float, float],
        dzg_range: Tuple[float, float],
    ):
        self.name = name
        self.temp_mean = temp_mean
        self.temp_std = temp_std
        self.fxa_range = fxa_range
        self.mya_range = mya_range
        self.rza_range = rza_range
        self.sxg_range = sxg_range
        self.lyg_range = lyg_range
        self.dzg_range = dzg_range


# Define behavior patterns based on animal physiology
BEHAVIOR_PATTERNS = {
    'lying': BehaviorPattern(
        name='lying',
        temp_mean=38.2,
        temp_std=0.15,
        fxa_range=(-0.3, 0.3),
        mya_range=(-0.2, 0.2),
        rza_range=(-0.6, -0.3),  # Horizontal orientation
        sxg_range=(-5, 5),
        lyg_range=(-5, 5),
        dzg_range=(-5, 5),
    ),
    'standing': BehaviorPattern(
        name='standing',
        temp_mean=38.4,
        temp_std=0.12,
        fxa_range=(-0.2, 0.2),
        mya_range=(-0.15, 0.15),
        rza_range=(-0.2, 0.2),  # Near vertical
        sxg_range=(-8, 8),
        lyg_range=(-8, 8),
        dzg_range=(-10, 10),
    ),
    'walking': BehaviorPattern(
        name='walking',
        temp_mean=38.6,
        temp_std=0.18,
        fxa_range=(-0.8, 1.2),  # Forward acceleration
        mya_range=(-0.5, 0.5),
        rza_range=(-0.4, 0.3),
        sxg_range=(-25, 25),
        lyg_range=(-30, 30),
        dzg_range=(-20, 20),
    ),
    'ruminating': BehaviorPattern(
        name='ruminating',
        temp_mean=38.3,
        temp_std=0.13,
        fxa_range=(-0.2, 0.2),
        mya_range=(-0.4, 0.4),  # Jaw movements
        rza_range=(-0.3, 0.1),
        sxg_range=(-10, 10),
        lyg_range=(-15, 15),  # Up-down head motion
        dzg_range=(-8, 8),
    ),
    'feeding': BehaviorPattern(
        name='feeding',
        temp_mean=38.5,
        temp_std=0.14,
        fxa_range=(-0.3, 0.5),
        mya_range=(-0.3, 0.3),
        rza_range=(-0.5, -0.1),  # Head down
        sxg_range=(-15, 15),
        lyg_range=(-35, 35),  # Strong pitch changes
        dzg_range=(-20, 20),
    ),
    'stress': BehaviorPattern(
        name='stress',
        temp_mean=39.0,
        temp_std=0.25,
        fxa_range=(-1.0, 1.0),
        mya_range=(-0.8, 0.8),
        rza_range=(-0.6, 0.6),
        sxg_range=(-45, 45),  # Erratic movements
        lyg_range=(-50, 50),
        dzg_range=(-40, 40),
    ),
}


class CircadianPattern:
    """Generates circadian temperature and activity patterns."""
    
    @staticmethod
    def get_hour_factor(hour: int) -> float:
        """Get activity factor for given hour (0-23)."""
        # Peak activity 6am-8pm, low activity at night
        if 6 <= hour < 20:
            return 1.0  # Daytime - normal activity
        elif 20 <= hour < 22:
            return 0.7  # Evening - reduced activity
        elif 22 <= hour or hour < 4:
            return 0.3  # Night - minimal activity
        else:  # 4-6am
            return 0.5  # Early morning - increasing activity
    
    @staticmethod
    def get_temperature_adjustment(hour: int) -> float:
        """Get temperature adjustment for circadian rhythm."""
        # Temperature lower at night, peaks mid-afternoon
        # Using sinusoidal pattern with minimum at 4am, maximum at 4pm
        hour_angle = (hour - 4) * 2 * np.pi / 24
        adjustment = 0.35 * np.sin(hour_angle)  # ±0.35°C variation
        return adjustment
    
    @staticmethod
    def get_daily_variation() -> float:
        """Get random day-to-day variation."""
        return np.random.normal(0, 0.1)


class SyntheticDataGenerator:
    """Generates synthetic sensor data with realistic patterns."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            np.random.seed(seed)
        self.circadian = CircadianPattern()
    
    def generate_behavior_sample(
        self,
        behavior: str,
        duration_minutes: int,
        start_time: datetime,
        apply_circadian: bool = True,
        day_variation: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate sensor data for a specific behavior.
        
        Args:
            behavior: Behavior name (lying, standing, walking, etc.)
            duration_minutes: Duration in minutes
            start_time: Starting timestamp
            apply_circadian: Whether to apply circadian patterns
            day_variation: Day-to-day temperature variation
        
        Returns:
            DataFrame with timestamp and sensor columns
        """
        if behavior not in BEHAVIOR_PATTERNS:
            raise ValueError(f"Unknown behavior: {behavior}")
        
        pattern = BEHAVIOR_PATTERNS[behavior]
        data = []
        
        for i in range(duration_minutes):
            current_time = start_time + timedelta(minutes=i)
            hour = current_time.hour
            
            # Base temperature
            temp = np.random.normal(pattern.temp_mean, pattern.temp_std)
            
            # Apply circadian rhythm
            if apply_circadian:
                temp += self.circadian.get_temperature_adjustment(hour)
                temp += day_variation
                
                # Adjust sensor ranges based on time of day
                activity_factor = self.circadian.get_hour_factor(hour)
            else:
                activity_factor = 1.0
            
            # Generate sensor values with some temporal correlation
            # Add slight smoothing to make transitions more realistic
            fxa = np.random.uniform(*pattern.fxa_range) * activity_factor
            mya = np.random.uniform(*pattern.mya_range) * activity_factor
            rza = np.random.uniform(*pattern.rza_range)
            sxg = np.random.uniform(*pattern.sxg_range) * activity_factor
            lyg = np.random.uniform(*pattern.lyg_range) * activity_factor
            dzg = np.random.uniform(*pattern.dzg_range) * activity_factor
            
            data.append({
                'timestamp': current_time,
                'temp': round(temp, 1),
                'Fxa': round(fxa, 1),
                'Mya': round(mya, 1),
                'Rza': round(rza, 1),
                'Sxg': round(sxg, 1),
                'Lyg': round(lyg, 1),
                'Dzg': round(dzg, 1),
                'behavior_label': behavior,
            })
        
        return pd.DataFrame(data)
    
    def generate_transition(
        self,
        from_behavior: str,
        to_behavior: str,
        start_time: datetime,
        transition_minutes: int = 2,
        apply_circadian: bool = True,
        day_variation: float = 0.0,
    ) -> pd.DataFrame:
        """
        Generate smooth transition between two behaviors.
        
        Args:
            from_behavior: Starting behavior
            to_behavior: Ending behavior
            start_time: Starting timestamp
            transition_minutes: Duration of transition (default 2 minutes)
            apply_circadian: Whether to apply circadian patterns
            day_variation: Day-to-day temperature variation
        
        Returns:
            DataFrame with gradual sensor value changes
        """
        from_pattern = BEHAVIOR_PATTERNS[from_behavior]
        to_pattern = BEHAVIOR_PATTERNS[to_behavior]
        
        data = []
        
        for i in range(transition_minutes):
            current_time = start_time + timedelta(minutes=i)
            hour = current_time.hour
            
            # Linear interpolation factor
            alpha = (i + 1) / transition_minutes
            
            # Interpolate temperature
            temp_mean = (1 - alpha) * from_pattern.temp_mean + alpha * to_pattern.temp_mean
            temp_std = (1 - alpha) * from_pattern.temp_std + alpha * to_pattern.temp_std
            temp = np.random.normal(temp_mean, temp_std)
            
            # Apply circadian rhythm
            if apply_circadian:
                temp += self.circadian.get_temperature_adjustment(hour)
                temp += day_variation
                activity_factor = self.circadian.get_hour_factor(hour)
            else:
                activity_factor = 1.0
            
            # Interpolate sensor ranges
            def interpolate_range(from_range, to_range, alpha):
                min_val = (1 - alpha) * from_range[0] + alpha * to_range[0]
                max_val = (1 - alpha) * from_range[1] + alpha * to_range[1]
                return (min_val, max_val)
            
            fxa_range = interpolate_range(from_pattern.fxa_range, to_pattern.fxa_range, alpha)
            mya_range = interpolate_range(from_pattern.mya_range, to_pattern.mya_range, alpha)
            rza_range = interpolate_range(from_pattern.rza_range, to_pattern.rza_range, alpha)
            sxg_range = interpolate_range(from_pattern.sxg_range, to_pattern.sxg_range, alpha)
            lyg_range = interpolate_range(from_pattern.lyg_range, to_pattern.lyg_range, alpha)
            dzg_range = interpolate_range(from_pattern.dzg_range, to_pattern.dzg_range, alpha)
            
            # Generate interpolated values
            fxa = np.random.uniform(*fxa_range) * activity_factor
            mya = np.random.uniform(*mya_range) * activity_factor
            rza = np.random.uniform(*rza_range)
            sxg = np.random.uniform(*sxg_range) * activity_factor
            lyg = np.random.uniform(*lyg_range) * activity_factor
            dzg = np.random.uniform(*dzg_range) * activity_factor
            
            data.append({
                'timestamp': current_time,
                'temp': round(temp, 1),
                'Fxa': round(fxa, 1),
                'Mya': round(mya, 1),
                'Rza': round(rza, 1),
                'Sxg': round(sxg, 1),
                'Lyg': round(lyg, 1),
                'Dzg': round(dzg, 1),
                'behavior_label': to_behavior,  # Label as target behavior
            })
        
        return pd.DataFrame(data)
    
    def generate_behavior_sequence(
        self,
        behaviors: List[Tuple[str, int]],
        start_time: datetime,
        apply_circadian: bool = True,
        smooth_transitions: bool = True,
    ) -> pd.DataFrame:
        """
        Generate a sequence of behaviors with optional smooth transitions.
        
        Args:
            behaviors: List of (behavior_name, duration_minutes) tuples
            start_time: Starting timestamp
            apply_circadian: Whether to apply circadian patterns
            smooth_transitions: Whether to generate smooth transitions
        
        Returns:
            DataFrame with complete behavior sequence
        """
        all_data = []
        current_time = start_time
        day_variation = self.circadian.get_daily_variation()
        
        for i, (behavior, duration) in enumerate(behaviors):
            # Add transition if not first behavior and smooth transitions enabled
            if i > 0 and smooth_transitions:
                prev_behavior = behaviors[i - 1][0]
                transition_df = self.generate_transition(
                    prev_behavior,
                    behavior,
                    current_time,
                    transition_minutes=2,
                    apply_circadian=apply_circadian,
                    day_variation=day_variation,
                )
                all_data.append(transition_df)
                current_time += timedelta(minutes=2)
            
            # Generate behavior data
            behavior_df = self.generate_behavior_sample(
                behavior,
                duration,
                current_time,
                apply_circadian=apply_circadian,
                day_variation=day_variation,
            )
            all_data.append(behavior_df)
            current_time += timedelta(minutes=duration)
        
        return pd.concat(all_data, ignore_index=True)
    
    def generate_daily_schedule(
        self,
        date: datetime,
        animal_id: Optional[str] = None,
    ) -> List[Tuple[str, int]]:
        """
        Generate a realistic daily behavior schedule.
        
        Args:
            date: Date for the schedule
            animal_id: Optional animal identifier for variation
        
        Returns:
            List of (behavior, duration_minutes) tuples for 24 hours
        """
        schedule = []
        
        # Night (00:00-06:00): Mostly lying with some ruminating
        schedule.extend([
            ('lying', np.random.randint(120, 180)),
            ('ruminating', np.random.randint(30, 60)),
            ('lying', np.random.randint(90, 120)),
        ])
        
        # Early morning (06:00-09:00): Wake up, stand, feed
        schedule.extend([
            ('standing', np.random.randint(10, 20)),
            ('walking', np.random.randint(5, 15)),
            ('feeding', np.random.randint(45, 75)),
            ('ruminating', np.random.randint(30, 45)),
        ])
        
        # Mid-morning (09:00-12:00): Mixed activity
        schedule.extend([
            ('standing', np.random.randint(15, 30)),
            ('walking', np.random.randint(15, 30)),
            ('lying', np.random.randint(60, 90)),
            ('ruminating', np.random.randint(30, 45)),
        ])
        
        # Afternoon (12:00-17:00): Feeding, walking, rest
        schedule.extend([
            ('walking', np.random.randint(10, 20)),
            ('feeding', np.random.randint(60, 90)),
            ('ruminating', np.random.randint(45, 60)),
            ('lying', np.random.randint(60, 90)),
            ('standing', np.random.randint(15, 30)),
        ])
        
        # Evening (17:00-20:00): Final feeding, preparation for rest
        schedule.extend([
            ('feeding', np.random.randint(30, 60)),
            ('ruminating', np.random.randint(30, 45)),
            ('standing', np.random.randint(10, 20)),
        ])
        
        # Night prep (20:00-00:00): Winding down
        schedule.extend([
            ('lying', np.random.randint(90, 120)),
            ('ruminating', np.random.randint(20, 40)),
            ('lying', np.random.randint(60, 90)),
        ])
        
        # Randomly add stress events (5% chance per day)
        if np.random.random() < 0.05:
            stress_idx = np.random.randint(0, len(schedule))
            schedule.insert(stress_idx, ('stress', np.random.randint(5, 15)))
        
        return schedule
