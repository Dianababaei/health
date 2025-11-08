"""
Ground Truth Label Generator

This module generates comprehensive ground truth labels for simulated datasets:
- Per-minute labels: behavioral_state, temperature_status, health_events, sensor_quality
- Daily aggregate labels: estrus_day, pregnancy_day, health_score

Labels are generated based on simulation state and health events to provide
complete ground truth for algorithm validation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from .health_events import (
    HealthEventSimulator, 
    HealthEventType,
    TemperatureStatus,
    SensorQuality
)
from .state_params import BehavioralState


class LabelGenerator:
    """
    Generates ground truth labels for simulated sensor data.
    
    Works in conjunction with HealthEventSimulator and SimulationEngine
    to produce comprehensive, time-aligned labels for all data points.
    """
    
    def __init__(self, health_simulator: Optional[HealthEventSimulator] = None):
        """
        Initialize label generator.
        
        Args:
            health_simulator: HealthEventSimulator instance for health event labels
        """
        self.health_simulator = health_simulator
    
    def generate_per_minute_labels(self,
                                   simulation_data: pd.DataFrame,
                                   health_simulator: Optional[HealthEventSimulator] = None) -> pd.DataFrame:
        """
        Generate per-minute ground truth labels for simulation data.
        
        Args:
            simulation_data: DataFrame from SimulationEngine with columns:
                            timestamp, temperature, true_state, etc.
            health_simulator: Optional HealthEventSimulator (uses self.health_simulator if None)
            
        Returns:
            DataFrame with added label columns:
            - behavioral_state: lying | standing | walking | ruminating | feeding
            - temperature_status: normal | elevated | fever | heat_stress | dropping
            - health_events: none | estrus | pregnancy_indication | illness
            - sensor_quality: normal | noisy | malfunction
        """
        if health_simulator is None:
            health_simulator = self.health_simulator
        
        if health_simulator is None:
            raise ValueError("No HealthEventSimulator provided")
        
        labeled_data = simulation_data.copy()
        
        # Behavioral state (rename from 'true_state' if needed)
        if 'true_state' in labeled_data.columns:
            labeled_data['behavioral_state'] = labeled_data['true_state']
        elif 'behavioral_state' not in labeled_data.columns:
            raise ValueError("simulation_data must contain 'true_state' or 'behavioral_state'")
        
        # Generate health event labels
        labeled_data['health_events'] = labeled_data['timestamp'].apply(
            lambda t: health_simulator.get_health_event_label(t).value
        )
        
        # Generate temperature status labels
        labeled_data['temperature_status'] = labeled_data.apply(
            lambda row: health_simulator.get_temperature_status(
                row['temperature'], row['timestamp']
            ).value,
            axis=1
        )
        
        # Generate sensor quality labels
        labeled_data['sensor_quality'] = labeled_data['timestamp'].apply(
            lambda t: health_simulator.get_sensor_quality(t).value
        )
        
        return labeled_data
    
    def generate_daily_aggregates(self,
                                 labeled_data: pd.DataFrame,
                                 health_simulator: Optional[HealthEventSimulator] = None) -> pd.DataFrame:
        """
        Generate daily aggregate labels from per-minute data.
        
        Args:
            labeled_data: DataFrame with per-minute labels
            health_simulator: Optional HealthEventSimulator
            
        Returns:
            DataFrame with one row per day containing:
            - date: Date
            - estrus_day: boolean
            - pregnancy_day: boolean
            - health_score: 0-100
            - mean_temperature: Daily average temperature
            - activity_level: Relative activity score
        """
        if health_simulator is None:
            health_simulator = self.health_simulator
        
        # Extract date from timestamp
        labeled_data = labeled_data.copy()
        labeled_data['date'] = pd.to_datetime(labeled_data['timestamp']).dt.date
        
        # Group by date
        daily_aggregates = []
        
        for date, day_data in labeled_data.groupby('date'):
            # Estrus day: any estrus event during the day
            estrus_day = (day_data['health_events'] == 'estrus').any()
            
            # Pregnancy day: any pregnancy indication during the day
            pregnancy_day = (day_data['health_events'] == 'pregnancy_indication').any()
            
            # Health score calculation (0-100)
            health_score = self._calculate_health_score(day_data)
            
            # Mean temperature
            mean_temperature = day_data['temperature'].mean()
            
            # Activity level (based on movement sensors)
            activity_level = self._calculate_activity_level(day_data)
            
            # Behavioral state distribution
            state_distribution = day_data['behavioral_state'].value_counts(normalize=True).to_dict()
            
            daily_aggregates.append({
                'date': date,
                'estrus_day': estrus_day,
                'pregnancy_day': pregnancy_day,
                'health_score': health_score,
                'mean_temperature': mean_temperature,
                'activity_level': activity_level,
                'lying_percent': state_distribution.get('lying', 0.0) * 100,
                'standing_percent': state_distribution.get('standing', 0.0) * 100,
                'walking_percent': state_distribution.get('walking', 0.0) * 100,
                'ruminating_percent': state_distribution.get('ruminating', 0.0) * 100,
                'feeding_percent': state_distribution.get('feeding', 0.0) * 100,
            })
        
        return pd.DataFrame(daily_aggregates)
    
    def _calculate_health_score(self, day_data: pd.DataFrame) -> float:
        """
        Calculate health score (0-100) based on daily data.
        
        Higher scores indicate better health. Score considers:
        - Temperature stability
        - Activity levels
        - Behavioral patterns
        - Absence of illness indicators
        
        Args:
            day_data: DataFrame with one day of data
            
        Returns:
            Health score (0-100)
        """
        score = 100.0
        
        # Temperature penalties
        temp_std = day_data['temperature'].std()
        if temp_std > 0.5:  # High variability
            score -= 10
        elif temp_std > 0.3:
            score -= 5
        
        fever_count = (day_data['temperature_status'] == 'fever').sum()
        if fever_count > 0:
            score -= 30 * min(fever_count / len(day_data), 1.0)
        
        elevated_count = (day_data['temperature_status'] == 'elevated').sum()
        if elevated_count > 0:
            score -= 10 * min(elevated_count / len(day_data), 1.0)
        
        # Illness penalties
        illness_count = (day_data['health_events'] == 'illness').sum()
        if illness_count > 0:
            score -= 25 * min(illness_count / len(day_data), 1.0)
        
        # Heat stress penalty
        heat_stress_count = (day_data['health_events'] == 'heat_stress').sum()
        if heat_stress_count > 0:
            score -= 15 * min(heat_stress_count / len(day_data), 1.0)
        
        # Activity pattern check
        # Normal lying should be 40-50% of day
        lying_pct = (day_data['behavioral_state'] == 'lying').sum() / len(day_data)
        if lying_pct < 0.3 or lying_pct > 0.6:
            score -= 10
        
        # Rumination check (should be 10-20% of day)
        ruminating_pct = (day_data['behavioral_state'] == 'ruminating').sum() / len(day_data)
        if ruminating_pct < 0.05:
            score -= 15  # Too little rumination is concerning
        
        # Sensor quality penalty
        bad_sensor_count = (
            (day_data['sensor_quality'] == 'noisy').sum() +
            (day_data['sensor_quality'] == 'malfunction').sum() * 2
        )
        if bad_sensor_count > 0:
            score -= 5 * min(bad_sensor_count / len(day_data), 1.0)
        
        # Estrus is normal, not a health penalty
        # Pregnancy is normal, not a health penalty
        
        return max(0.0, min(100.0, score))
    
    def _calculate_activity_level(self, day_data: pd.DataFrame) -> float:
        """
        Calculate relative activity level from sensor data.
        
        Args:
            day_data: DataFrame with one day of data
            
        Returns:
            Activity level score (arbitrary units, higher = more active)
        """
        # Calculate mean absolute acceleration
        if 'fxa' in day_data.columns and 'mya' in day_data.columns:
            mean_fxa = day_data['fxa'].abs().mean()
            mean_mya = day_data['mya'].abs().mean()
            mean_sxg = day_data['sxg'].abs().mean() if 'sxg' in day_data.columns else 0
            
            # Weighted combination
            activity = (mean_fxa * 2.0 + mean_mya * 1.5 + mean_sxg * 0.5)
            return activity
        
        return 0.0
    
    def add_stress_labels(self,
                         labeled_data: pd.DataFrame,
                         stress_threshold_activity: float = 1.5) -> pd.DataFrame:
        """
        Add stress detection labels based on behavioral patterns.
        
        Identifies potential stress based on:
        - High sustained activity
        - Reduced rumination
        - Elevated temperature with high activity
        
        Args:
            labeled_data: DataFrame with existing labels
            stress_threshold_activity: Activity multiplier threshold for stress
            
        Returns:
            DataFrame with added 'stress_indicator' column
        """
        labeled_data = labeled_data.copy()
        
        # Calculate rolling activity (1-hour window)
        window_size = 60  # 60 minutes
        
        # Simplified stress indicator based on temperature and state
        labeled_data['stress_indicator'] = (
            (labeled_data['temperature_status'].isin(['elevated', 'fever', 'heat_stress'])) &
            (labeled_data['behavioral_state'].isin(['walking', 'standing']))
        )
        
        return labeled_data
    
    def export_labels_to_csv(self,
                            labeled_data: pd.DataFrame,
                            output_path: str,
                            include_sensor_data: bool = True):
        """
        Export labeled data to CSV in required format.
        
        Args:
            labeled_data: DataFrame with all labels
            output_path: Path to output CSV file
            include_sensor_data: Include all sensor columns (True) or just labels (False)
        """
        if include_sensor_data:
            # Full export with sensors and labels
            columns = [
                'timestamp', 'temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg',
                'behavioral_state', 'temperature_status', 'health_events', 'sensor_quality'
            ]
        else:
            # Labels only
            columns = [
                'timestamp', 'behavioral_state', 'temperature_status', 
                'health_events', 'sensor_quality'
            ]
        
        # Filter to existing columns
        available_columns = [c for c in columns if c in labeled_data.columns]
        export_data = labeled_data[available_columns]
        
        export_data.to_csv(output_path, index=False)
    
    def generate_complete_labels(self,
                                simulation_data: pd.DataFrame,
                                health_simulator: HealthEventSimulator) -> Dict[str, pd.DataFrame]:
        """
        Generate all label types (per-minute and daily) in one call.
        
        Args:
            simulation_data: Raw simulation data from SimulationEngine
            health_simulator: HealthEventSimulator instance
            
        Returns:
            Dictionary with 'per_minute' and 'daily' DataFrames
        """
        # Generate per-minute labels
        per_minute = self.generate_per_minute_labels(simulation_data, health_simulator)
        
        # Generate daily aggregates
        daily = self.generate_daily_aggregates(per_minute, health_simulator)
        
        return {
            'per_minute': per_minute,
            'daily': daily
        }


def create_ground_truth_labels(simulation_data: pd.DataFrame,
                               health_simulator: HealthEventSimulator) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to generate all ground truth labels.
    
    Args:
        simulation_data: Raw simulation data from SimulationEngine
        health_simulator: HealthEventSimulator instance
        
    Returns:
        Dictionary with 'per_minute' and 'daily' labeled DataFrames
    """
    generator = LabelGenerator(health_simulator)
    return generator.generate_complete_labels(simulation_data, health_simulator)
