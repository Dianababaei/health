"""
Dataset Generation Orchestrator

Main orchestrator for generating complete datasets at different time scales:
- Short-term (7 days): Rapid testing and basic validation
- Medium-term (30 days): Circadian patterns and estrus cycle validation
- Long-term (90-180 days): Reproductive cycles and long-term trends

Coordinates SimulationEngine, HealthEventSimulator, and LabelGenerator
to produce comprehensive, labeled datasets with ground truth.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
import pandas as pd
import numpy as np

from .engine import SimulationEngine, SimulationConfig
from .state_params import AnimalProfile, create_default_profile
from .health_events import HealthEventSimulator, HealthEventType
from .label_generator import LabelGenerator
from .export import DatasetExporter, DatasetSplitter, export_complete_dataset
from .noise import NoiseParameters


class DatasetGenerationConfig:
    """Configuration for dataset generation."""
    
    def __init__(self,
                 duration_days: int,
                 animal_id: str = "sim_cow_001",
                 seed: Optional[int] = None,
                 include_estrus: bool = True,
                 include_pregnancy: bool = True,
                 num_illness_events: int = 0,
                 num_heat_stress_events: int = 0,
                 start_time: Optional[datetime] = None):
        """
        Initialize dataset generation configuration.
        
        Args:
            duration_days: Number of days to simulate
            animal_id: Unique animal identifier
            seed: Random seed for reproducibility
            include_estrus: Include estrus cycles
            include_pregnancy: Allow pregnancy events
            num_illness_events: Number of illness events (0=auto)
            num_heat_stress_events: Number of heat stress events (0=auto)
            start_time: Simulation start time (defaults to 2024-01-01)
        """
        self.duration_days = duration_days
        self.animal_id = animal_id
        self.seed = seed
        self.include_estrus = include_estrus
        self.include_pregnancy = include_pregnancy
        self.num_illness_events = num_illness_events
        self.num_heat_stress_events = num_heat_stress_events
        self.start_time = start_time or datetime(2024, 1, 1, 0, 0, 0)


class DatasetGenerator:
    """
    Main orchestrator for generating complete datasets.
    
    Coordinates all simulation components to generate realistic,
    labeled datasets at various time scales.
    """
    
    def __init__(self, config: DatasetGenerationConfig):
        """
        Initialize dataset generator.
        
        Args:
            config: Dataset generation configuration
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
        # Create animal profile
        self.animal_profile = create_default_profile(
            config.animal_id,
            self.rng
        )
        
        # Initialize health event simulator
        self.health_simulator = HealthEventSimulator(
            seed=self.rng.integers(0, 2**31) if config.seed else None
        )
        
        # Initialize label generator
        self.label_generator = LabelGenerator(self.health_simulator)
        
        # Store generated events for reference
        self.generated_events: Dict[str, List] = {}
    
    def generate_dataset(self) -> Dict[str, pd.DataFrame]:
        """
        Generate complete dataset with labels.
        
        Returns:
            Dictionary with 'sensor_data', 'labeled_data', and 'daily_aggregates'
        """
        # Generate health events for the entire period
        end_time = self.config.start_time + timedelta(days=self.config.duration_days)
        
        self.generated_events = self.health_simulator.generate_all_events(
            self.config.start_time,
            end_time,
            include_estrus=self.config.include_estrus,
            include_pregnancy=self.config.include_pregnancy,
            num_illness=self.config.num_illness_events,
            num_heat_stress=self.config.num_heat_stress_events
        )
        
        # Create simulation engine with modified profile
        sim_engine = SimulationEngine(
            animal_id=self.config.animal_id,
            animal_profile=self.animal_profile,
            seed=self.rng.integers(0, 2**31) if self.config.seed else None
        )
        
        # Run simulation with health event modifications
        sensor_data = self._run_simulation_with_health_events(
            sim_engine,
            self.config.start_time,
            self.config.duration_days
        )
        
        # Generate labels
        labeled_data = self.label_generator.generate_per_minute_labels(
            sensor_data,
            self.health_simulator
        )
        
        # Generate daily aggregates
        daily_aggregates = self.label_generator.generate_daily_aggregates(
            labeled_data,
            self.health_simulator
        )
        
        return {
            'sensor_data': sensor_data,
            'labeled_data': labeled_data,
            'daily_aggregates': daily_aggregates
        }
    
    def _run_simulation_with_health_events(self,
                                          engine: SimulationEngine,
                                          start_time: datetime,
                                          duration_days: int) -> pd.DataFrame:
        """
        Run simulation while applying health event modifications.
        
        Args:
            engine: SimulationEngine instance
            start_time: Start timestamp
            duration_days: Duration in days
            
        Returns:
            DataFrame with simulated sensor data
        """
        # Initialize simulation
        engine.current_time = start_time
        engine.simulation_data = []
        engine.transition_model.initialize_state(None, start_time)
        
        num_minutes = duration_days * 24 * 60
        
        # Run minute by minute with health event modifications
        for minute in range(num_minutes):
            current_time = start_time + timedelta(minutes=minute)
            
            # Get health event modifiers
            modifiers = self.health_simulator.get_profile_modifiers(current_time)
            
            # Temporarily modify animal profile
            original_fever = self.animal_profile.fever_offset
            original_lethargy = self.animal_profile.lethargy_factor
            
            self.animal_profile.fever_offset = modifiers['temperature_offset']
            self.animal_profile.lethargy_factor = modifiers['activity_multiplier']
            
            # Apply noise multiplier if needed
            if modifiers['noise_multiplier'] > 1.0:
                # Temporarily increase noise
                original_noise_params = engine.noise_generator.params
                engine.noise_generator.params = NoiseParameters(
                    temperature_std=original_noise_params.temperature_std * modifiers['noise_multiplier'],
                    accelerometer_std=original_noise_params.accelerometer_std * modifiers['noise_multiplier'],
                    gyroscope_std=original_noise_params.gyroscope_std * modifiers['noise_multiplier']
                )
            
            # Generate data point
            data_point = engine._generate_data_point(current_time)
            engine.simulation_data.append(data_point)
            
            # Restore original values
            self.animal_profile.fever_offset = original_fever
            self.animal_profile.lethargy_factor = original_lethargy
            
            if modifiers['noise_multiplier'] > 1.0:
                engine.noise_generator.params = original_noise_params
        
        return pd.DataFrame(engine.simulation_data)
    
    def get_generation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the dataset generation.
        
        Returns:
            Dictionary with generation parameters and event summaries
        """
        metadata = {
            'generation_config': {
                'duration_days': self.config.duration_days,
                'animal_id': self.config.animal_id,
                'seed': self.config.seed,
                'start_time': self.config.start_time.isoformat(),
                'end_time': (self.config.start_time + timedelta(days=self.config.duration_days)).isoformat(),
            },
            'animal_profile': {
                'baseline_temperature': float(self.animal_profile.baseline_temperature),
                'activity_multiplier': float(self.animal_profile.activity_multiplier),
                'body_size_factor': float(self.animal_profile.body_size_factor),
                'age_category': self.animal_profile.age_category,
            },
            'health_events': self._summarize_health_events(),
            'simulation_parameters': {
                'time_step_minutes': 1,
                'include_noise': True,
                'include_temporal_effects': True,
                'include_circadian_rhythms': True,
            }
        }
        
        return metadata
    
    def _summarize_health_events(self) -> Dict[str, Any]:
        """Summarize generated health events."""
        summary = {}
        
        for event_type, events in self.generated_events.items():
            if events:
                summary[event_type] = {
                    'count': len(events),
                    'events': [
                        {
                            'start_time': e.start_time.isoformat(),
                            'duration_hours': float(e.duration_hours),
                            'severity': float(e.severity) if hasattr(e, 'severity') else 1.0,
                        }
                        for e in events
                    ]
                }
            else:
                summary[event_type] = {'count': 0, 'events': []}
        
        # Add pregnancy info if present
        if self.health_simulator.pregnancy_state is not None:
            summary['pregnancy'] = {
                'conception_date': self.health_simulator.pregnancy_state.conception_date.isoformat(),
                'confirmed': self.health_simulator.pregnancy_state.is_confirmed,
            }
        
        return summary
    
    def validate_dataset(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the generated dataset.
        
        Checks for:
        - Continuity (no timestamp gaps)
        - Value ranges
        - Label consistency
        - State distribution realism
        
        Args:
            labeled_data: Generated labeled dataset
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check continuity
        timestamps = pd.to_datetime(labeled_data['timestamp'])
        time_diffs = timestamps.diff().dt.total_seconds() / 60.0
        gaps = time_diffs[time_diffs > 1.1].index.tolist()
        
        if gaps:
            validation['errors'].append(f"Found {len(gaps)} timestamp gaps")
            validation['is_valid'] = False
        
        # Check value ranges
        if 'temperature' in labeled_data.columns:
            temp_out_of_range = (
                (labeled_data['temperature'] < 36.0) |
                (labeled_data['temperature'] > 42.0)
            ).sum()
            if temp_out_of_range > 0:
                validation['warnings'].append(
                    f"{temp_out_of_range} temperature values out of range"
                )
        
        # Check behavioral state distribution
        if 'behavioral_state' in labeled_data.columns:
            state_dist = labeled_data['behavioral_state'].value_counts(normalize=True)
            
            # Lying should be 30-60%
            lying_pct = state_dist.get('lying', 0) * 100
            if lying_pct < 30 or lying_pct > 60:
                validation['warnings'].append(
                    f"Lying percentage ({lying_pct:.1f}%) outside expected range (30-60%)"
                )
            
            # Ruminating should be 5-25%
            ruminating_pct = state_dist.get('ruminating', 0) * 100
            if ruminating_pct < 5 or ruminating_pct > 25:
                validation['warnings'].append(
                    f"Ruminating percentage ({ruminating_pct:.1f}%) outside expected range (5-25%)"
                )
        
        # Check label consistency
        if 'health_events' in labeled_data.columns:
            estrus_count = (labeled_data['health_events'] == 'estrus').sum()
            pregnancy_count = (labeled_data['health_events'] == 'pregnancy_indication').sum()
            
            # If pregnancy exists, there should have been estrus before it
            if pregnancy_count > 0 and estrus_count == 0:
                validation['warnings'].append(
                    "Pregnancy indication without prior estrus event"
                )
        
        validation['statistics'] = {
            'total_points': len(labeled_data),
            'duration_days': len(labeled_data) / 1440.0,
        }
        
        return validation


def generate_short_term_dataset(animal_id: str = "cow_short_001",
                                seed: Optional[int] = None,
                                output_dir: str = "data/simulated") -> Dict[str, Any]:
    """
    Generate 7-day short-term dataset.
    
    Purpose: Rapid algorithm testing and development iteration
    Scenarios: Basic behavioral patterns, 1-2 simple health events
    
    Args:
        animal_id: Animal identifier
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with paths and statistics
    """
    config = DatasetGenerationConfig(
        duration_days=7,
        animal_id=animal_id,
        seed=seed,
        include_estrus=False,  # Too short for full cycle
        include_pregnancy=False,
        num_illness_events=1,  # 1 simple illness event
        num_heat_stress_events=1,
    )
    
    generator = DatasetGenerator(config)
    datasets = generator.generate_dataset()
    
    # Validate
    validation = generator.validate_dataset(datasets['labeled_data'])
    
    # Export
    metadata = generator.get_generation_metadata()
    metadata['validation'] = validation
    
    result = export_complete_dataset(
        datasets['labeled_data'],
        datasets['daily_aggregates'],
        'short_term_7d',
        output_dir,
        metadata,
        create_splits=True
    )
    
    return result


def generate_medium_term_dataset(animal_id: str = "cow_medium_001",
                                 seed: Optional[int] = None,
                                 output_dir: str = "data/simulated") -> Dict[str, Any]:
    """
    Generate 30-day medium-term dataset.
    
    Purpose: Circadian rhythm validation and temperature pattern analysis
    Scenarios: Complete circadian cycles, 1-2 estrus cycles, activity trends
    
    Args:
        animal_id: Animal identifier
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with paths and statistics
    """
    config = DatasetGenerationConfig(
        duration_days=30,
        animal_id=animal_id,
        seed=seed,
        include_estrus=True,  # 1-2 estrus cycles
        include_pregnancy=False,  # Not enough time
        num_illness_events=0,  # Auto-determine
        num_heat_stress_events=0,  # Auto-determine
    )
    
    generator = DatasetGenerator(config)
    datasets = generator.generate_dataset()
    
    # Validate
    validation = generator.validate_dataset(datasets['labeled_data'])
    
    # Export
    metadata = generator.get_generation_metadata()
    metadata['validation'] = validation
    
    result = export_complete_dataset(
        datasets['labeled_data'],
        datasets['daily_aggregates'],
        'medium_term_30d',
        output_dir,
        metadata,
        create_splits=True
    )
    
    return result


def generate_long_term_dataset(duration_days: int = 90,
                               animal_id: str = "cow_long_001",
                               seed: Optional[int] = None,
                               output_dir: str = "data/simulated") -> Dict[str, Any]:
    """
    Generate 90-180 day long-term dataset.
    
    Purpose: Reproductive cycle tracking and long-term trends
    Scenarios: Multiple estrus cycles, pregnancy progression, seasonal variations
    
    Args:
        duration_days: Duration (90 or 180 days)
        animal_id: Animal identifier
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary with paths and statistics
    """
    if duration_days not in [90, 180]:
        raise ValueError("Long-term dataset should be 90 or 180 days")
    
    config = DatasetGenerationConfig(
        duration_days=duration_days,
        animal_id=animal_id,
        seed=seed,
        include_estrus=True,
        include_pregnancy=True,  # May occur
        num_illness_events=0,  # Auto-determine
        num_heat_stress_events=0,  # Auto-determine
    )
    
    generator = DatasetGenerator(config)
    datasets = generator.generate_dataset()
    
    # Validate
    validation = generator.validate_dataset(datasets['labeled_data'])
    
    # Export
    metadata = generator.get_generation_metadata()
    metadata['validation'] = validation
    
    dataset_name = f'long_term_{duration_days}d'
    result = export_complete_dataset(
        datasets['labeled_data'],
        datasets['daily_aggregates'],
        dataset_name,
        output_dir,
        metadata,
        create_splits=True
    )
    
    return result


def generate_all_datasets(output_dir: str = "data/simulated",
                         seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate all three dataset time scales.
    
    Args:
        output_dir: Output directory
        seed: Random seed (will be modified for each dataset)
        
    Returns:
        Dictionary with results for all datasets
    """
    results = {}
    
    # Generate seeds for each dataset if base seed provided
    if seed is not None:
        rng = np.random.default_rng(seed)
        seeds = {
            'short': rng.integers(0, 2**31),
            'medium': rng.integers(0, 2**31),
            'long': rng.integers(0, 2**31),
        }
    else:
        seeds = {'short': None, 'medium': None, 'long': None}
    
    print("Generating short-term dataset (7 days)...")
    results['short_term'] = generate_short_term_dataset(
        animal_id="cow_short_001",
        seed=seeds['short'],
        output_dir=output_dir
    )
    print(f"  ✓ Generated: {results['short_term']['paths']['csv_path']}")
    
    print("\nGenerating medium-term dataset (30 days)...")
    results['medium_term'] = generate_medium_term_dataset(
        animal_id="cow_medium_001",
        seed=seeds['medium'],
        output_dir=output_dir
    )
    print(f"  ✓ Generated: {results['medium_term']['paths']['csv_path']}")
    
    print("\nGenerating long-term dataset (90 days)...")
    results['long_term'] = generate_long_term_dataset(
        duration_days=90,
        animal_id="cow_long_001",
        seed=seeds['long'],
        output_dir=output_dir
    )
    print(f"  ✓ Generated: {results['long_term']['paths']['csv_path']}")
    
    print("\n" + "=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)
    
    return results
