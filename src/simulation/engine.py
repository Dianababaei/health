"""
Simulation Engine Core

Main orchestrator for the behavioral state simulation. This engine:
- Manages time progression at 1-minute intervals
- Coordinates state transitions through the transition model
- Generates sensor values based on current state and animal profile
- Applies noise and individual variation
- Handles temporal patterns (circadian rhythms)
- Validates generated data
- Exports data to CSV/DataFrame formats

Usage:
    engine = SimulationEngine(animal_id="cow_001", seed=42)
    data = engine.run_simulation(
        duration_hours=24,
        start_time=datetime(2024, 1, 1, 0, 0)
    )
    data.to_csv('simulated_data.csv', index=False)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import warnings

from .state_params import (
    BehavioralState, 
    AnimalProfile, 
    create_default_profile,
    validate_sensor_values
)
from .transitions import StateTransitionModel, StateTransitionConfig
from .noise import NoiseGenerator, NoiseParameters
from .temporal import TemporalPatternManager


@dataclass
class SimulationConfig:
    """Configuration for the simulation engine."""
    time_step_minutes: float = 1.0  # Simulation time step
    include_validation: bool = True  # Validate sensor values
    include_noise: bool = True  # Add sensor noise
    include_temporal_effects: bool = True  # Apply circadian rhythms
    include_rhythmic_patterns: bool = True  # Add rhythmic variations
    
    # Optional: environmental parameters
    ambient_temperature: Optional[float] = None  # Celsius


class SimulationEngine:
    """
    Main simulation engine for generating realistic cattle sensor data.
    
    This engine orchestrates all components to generate minute-by-minute sensor
    readings that accurately reflect cattle behavioral states and individual
    characteristics.
    """
    
    def __init__(self,
                 animal_id: str = "sim_animal_001",
                 animal_profile: Optional[AnimalProfile] = None,
                 transition_config: Optional[StateTransitionConfig] = None,
                 noise_params: Optional[NoiseParameters] = None,
                 sim_config: Optional[SimulationConfig] = None,
                 seed: Optional[int] = None):
        """
        Initialize the simulation engine.
        
        Args:
            animal_id: Unique identifier for the simulated animal
            animal_profile: Animal characteristics (auto-generated if None)
            transition_config: State transition configuration
            noise_params: Noise generation parameters
            sim_config: Simulation configuration
            seed: Random seed for reproducibility
        """
        self.animal_id = animal_id
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Create or use provided animal profile
        if animal_profile is None:
            self.animal_profile = create_default_profile(animal_id, self.rng)
        else:
            self.animal_profile = animal_profile
        
        # Initialize components
        self.temporal_manager = TemporalPatternManager()
        self.transition_model = StateTransitionModel(
            config=transition_config,
            temporal_manager=self.temporal_manager,
            seed=self.rng.integers(0, 2**31) if seed else None
        )
        self.noise_generator = NoiseGenerator(
            params=noise_params,
            seed=self.rng.integers(0, 2**31) if seed else None
        )
        
        # Configuration
        self.config = sim_config or SimulationConfig()
        
        # Tracking
        self.current_time: Optional[datetime] = None
        self.simulation_data: List[Dict] = []
        self.validation_warnings: List[str] = []
    
    def _generate_base_sensor_values(self, 
                                     state: BehavioralState,
                                     timestamp: datetime,
                                     time_in_state: float) -> Dict[str, float]:
        """
        Generate base sensor values for the current state.
        
        Args:
            state: Current behavioral state
            timestamp: Current timestamp
            time_in_state: Time spent in current state (minutes)
            
        Returns:
            Dictionary with base sensor values
        """
        # Get sensor signature (possibly interpolated if transitioning)
        signature = self.transition_model.get_interpolated_signature()
        
        # Sample values from ranges
        values = {
            'temperature': signature.temperature.sample(self.rng),
            'fxa': signature.fxa.sample(self.rng),
            'mya': signature.mya.sample(self.rng),
            'rza': signature.rza.sample(self.rng),
            'sxg': signature.sxg.sample(self.rng),
            'lyg': signature.lyg.sample(self.rng),
            'dzg': signature.dzg.sample(self.rng),
        }
        
        # Add rhythmic patterns if applicable
        if (self.config.include_rhythmic_patterns and 
            signature.rhythmic_frequency is not None):
            
            # Apply rhythmic variation to motion sensors
            for key in ['fxa', 'mya', 'sxg', 'lyg', 'dzg']:
                values[key] = self.noise_generator.add_rhythmic_variation(
                    values[key],
                    time_in_state,
                    signature.rhythmic_frequency,
                    signature.rhythmic_amplitude_scale or 1.0
                )
        
        return values
    
    def _apply_individual_modifications(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Apply individual animal variations to sensor values.
        
        Args:
            values: Base sensor values
            
        Returns:
            Modified sensor values
        """
        modified = values.copy()
        
        # Apply temperature modifications
        modified['temperature'] = self.animal_profile.apply_temperature_modification(
            values['temperature']
        )
        
        # Apply activity modifications to motion sensors
        for key in ['fxa', 'mya', 'sxg', 'lyg', 'dzg']:
            modified[key] = self.animal_profile.apply_activity_modification(
                values[key]
            )
        
        # Rza (vertical acceleration) also affected by body size
        modified['rza'] = values['rza'] * self.animal_profile.body_size_factor
        
        return modified
    
    def _apply_temporal_effects(self, values: Dict[str, float], 
                               timestamp: datetime) -> Dict[str, float]:
        """
        Apply temporal effects (circadian rhythms, time-of-day).
        
        Args:
            values: Sensor values
            timestamp: Current timestamp
            
        Returns:
            Modified sensor values
        """
        if not self.config.include_temporal_effects:
            return values
        
        modified = values.copy()
        
        # Apply circadian temperature effect
        hour = self.temporal_manager.get_hour_of_day(timestamp)
        modified['temperature'] = self.temporal_manager.apply_circadian_temperature_effect(
            values['temperature'], hour
        )
        
        # Apply environmental temperature effect
        modified['temperature'] = self.noise_generator.apply_environmental_temperature_effect(
            modified['temperature'],
            hour,
            self.config.ambient_temperature
        )
        
        return modified
    
    def _apply_noise(self, values: Dict[str, float]) -> Dict[str, float]:
        """
        Apply sensor noise to all values.
        
        Args:
            values: Sensor values
            
        Returns:
            Noisy sensor values
        """
        if not self.config.include_noise:
            return values
        
        return self.noise_generator.add_all_sensor_noise(**values)
    
    def _validate_values(self, values: Dict[str, float], timestamp: datetime) -> bool:
        """
        Validate sensor values are within realistic ranges.
        
        Args:
            values: Sensor values to validate
            timestamp: Current timestamp for warning messages
            
        Returns:
            True if valid, False otherwise
        """
        if not self.config.include_validation:
            return True
        
        is_valid = validate_sensor_values(
            values['temperature'],
            values['fxa'],
            values['mya'],
            values['rza'],
            values['sxg'],
            values['lyg'],
            values['dzg']
        )
        
        if not is_valid:
            warning_msg = f"Invalid sensor values at {timestamp}: {values}"
            self.validation_warnings.append(warning_msg)
            warnings.warn(warning_msg)
        
        return is_valid
    
    def _generate_data_point(self, timestamp: datetime) -> Dict:
        """
        Generate a complete data point for the current time step.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dictionary with all sensor values and metadata
        """
        # Update state machine
        current_state, is_transitioning = self.transition_model.update(
            timestamp, 
            self.config.time_step_minutes
        )
        
        # Generate base sensor values
        values = self._generate_base_sensor_values(
            current_state,
            timestamp,
            self.transition_model.time_in_state
        )
        
        # Apply individual animal modifications
        values = self._apply_individual_modifications(values)
        
        # Apply temporal effects
        values = self._apply_temporal_effects(values, timestamp)
        
        # Apply noise
        values = self._apply_noise(values)
        
        # Validate
        self._validate_values(values, timestamp)
        
        # Create complete data point
        data_point = {
            'timestamp': timestamp,
            'animal_id': self.animal_id,
            'temperature': values['temperature'],
            'fxa': values['fxa'],
            'mya': values['mya'],
            'rza': values['rza'],
            'sxg': values['sxg'],
            'lyg': values['lyg'],
            'dzg': values['dzg'],
            'true_state': current_state.value,
            'is_transitioning': is_transitioning,
            'time_in_state': self.transition_model.time_in_state,
        }
        
        return data_point
    
    def run_simulation(self,
                      duration_hours: float = 24.0,
                      start_time: Optional[datetime] = None,
                      initial_state: Optional[BehavioralState] = None) -> pd.DataFrame:
        """
        Run the simulation for specified duration.
        
        Args:
            duration_hours: Simulation duration in hours
            start_time: Starting timestamp (uses current time if None)
            initial_state: Initial behavioral state (auto-selected if None)
            
        Returns:
            DataFrame with simulated sensor data
        """
        # Initialize
        if start_time is None:
            start_time = datetime.now().replace(second=0, microsecond=0)
        
        self.current_time = start_time
        self.simulation_data = []
        self.validation_warnings = []
        
        # Initialize state machine
        self.transition_model.initialize_state(initial_state, start_time)
        
        # Calculate number of time steps
        num_steps = int(duration_hours * 60 / self.config.time_step_minutes)
        
        # Run simulation
        for step in range(num_steps):
            # Generate data point
            data_point = self._generate_data_point(self.current_time)
            self.simulation_data.append(data_point)
            
            # Advance time
            self.current_time += timedelta(minutes=self.config.time_step_minutes)
        
        # Convert to DataFrame
        df = pd.DataFrame(self.simulation_data)
        
        # Report validation warnings if any
        if self.validation_warnings:
            warnings.warn(f"Simulation completed with {len(self.validation_warnings)} validation warnings")
        
        return df
    
    def run_multi_day_simulation(self,
                                num_days: int = 7,
                                start_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Run a multi-day simulation with consistent patterns.
        
        Args:
            num_days: Number of days to simulate
            start_time: Starting timestamp
            
        Returns:
            DataFrame with multi-day simulated data
        """
        return self.run_simulation(
            duration_hours=num_days * 24,
            start_time=start_time
        )
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the simulation.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.simulation_data:
            return {}
        
        df = pd.DataFrame(self.simulation_data)
        
        # State distribution
        state_counts = df['true_state'].value_counts()
        state_percentages = (state_counts / len(df) * 100).to_dict()
        
        # Sensor statistics
        sensor_stats = {}
        for sensor in ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']:
            sensor_stats[sensor] = {
                'mean': df[sensor].mean(),
                'std': df[sensor].std(),
                'min': df[sensor].min(),
                'max': df[sensor].max(),
            }
        
        # Transition statistics
        transitions = (df['is_transitioning'].sum() / len(df) * 100)
        
        return {
            'animal_id': self.animal_id,
            'total_data_points': len(df),
            'duration_hours': len(df) * self.config.time_step_minutes / 60,
            'state_distribution': state_percentages,
            'sensor_statistics': sensor_stats,
            'transition_percentage': transitions,
            'validation_warnings': len(self.validation_warnings),
        }
    
    def export_to_csv(self, filepath: str, include_metadata: bool = True):
        """
        Export simulation data to CSV file.
        
        Args:
            filepath: Output file path
            include_metadata: Include true_state and other metadata columns
        """
        df = pd.DataFrame(self.simulation_data)
        
        if not include_metadata:
            # Export only sensor data
            columns = ['timestamp', 'animal_id', 'temperature', 
                      'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
            df = df[columns]
        
        df.to_csv(filepath, index=False)
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the simulation engine to initial state.
        
        Args:
            seed: New random seed (keeps existing if None)
        """
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
        
        self.current_time = None
        self.simulation_data = []
        self.validation_warnings = []
        self.noise_generator.reset()


def create_simulation_engine(animal_id: str = "sim_animal_001",
                            seed: Optional[int] = None,
                            **kwargs) -> SimulationEngine:
    """
    Convenience function to create a simulation engine with default settings.
    
    Args:
        animal_id: Animal identifier
        seed: Random seed
        **kwargs: Additional arguments passed to SimulationEngine
        
    Returns:
        Configured SimulationEngine instance
    """
    return SimulationEngine(animal_id=animal_id, seed=seed, **kwargs)
