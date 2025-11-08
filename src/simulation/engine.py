"""
Simulation engine for generating realistic cattle behavioral sensor data.

Integrates all behavioral state generators and manages state transitions
to produce continuous, realistic sensor data streams.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta

from .states import (
    SensorReading,
    BehavioralStateGenerator,
    LyingStateGenerator,
    StandingStateGenerator,
    WalkingStateGenerator,
    RuminatingStateGenerator,
    FeedingStateGenerator,
    StressBehaviorOverlay,
)
from .transitions import (
    BehaviorState,
    StateTransitionManager,
    TransitionValidator,
)


class SimulationEngine:
    """
    Main simulation engine for generating realistic cattle sensor data.
    
    Coordinates state generators, manages transitions, and produces
    continuous sensor data streams with realistic behavioral patterns.
    """
    
    def __init__(
        self,
        baseline_temperature: float = 38.5,
        sampling_rate: float = 1.0,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the simulation engine.
        
        Args:
            baseline_temperature: Normal body temperature in °C (default: 38.5°C)
            sampling_rate: Samples per minute (default: 1.0 for 1 sample/minute)
            random_seed: Random seed for reproducibility (optional)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.baseline_temperature = baseline_temperature
        self.sampling_rate = sampling_rate
        
        # Initialize state generators
        self.generators = {
            BehaviorState.LYING: LyingStateGenerator(baseline_temperature, sampling_rate),
            BehaviorState.STANDING: StandingStateGenerator(baseline_temperature, sampling_rate),
            BehaviorState.WALKING: WalkingStateGenerator(baseline_temperature, sampling_rate),
            BehaviorState.RUMINATING_LYING: RuminatingStateGenerator(
                baseline_temperature, sampling_rate, base_posture='lying'
            ),
            BehaviorState.RUMINATING_STANDING: RuminatingStateGenerator(
                baseline_temperature, sampling_rate, base_posture='standing'
            ),
            BehaviorState.FEEDING: FeedingStateGenerator(baseline_temperature, sampling_rate),
        }
        
        # Initialize transition manager
        self.transition_manager = StateTransitionManager(sampling_rate)
        
        # Initialize validator
        self.validator = TransitionValidator()
        
        # Stress overlay handler
        self.stress_overlay = StressBehaviorOverlay()
    
    def generate_continuous_data(
        self,
        duration_hours: float,
        start_datetime: Optional[datetime] = None,
        include_stress: bool = False,
        stress_probability: float = 0.05,
    ) -> pd.DataFrame:
        """
        Generate continuous sensor data for specified duration.
        
        Args:
            duration_hours: Total duration in hours
            start_datetime: Starting datetime (default: now)
            include_stress: Whether to include stress behavior overlays
            stress_probability: Probability of stress occurring per state transition
            
        Returns:
            DataFrame with columns: timestamp, temperature, fxa, mya, rza, sxg, lyg, dzg, state
        """
        if start_datetime is None:
            start_datetime = datetime.now()
        
        total_minutes = duration_hours * 60
        current_time = 0.0  # in seconds
        all_readings = []
        state_labels = []
        
        # Start with a common initial state
        current_state = np.random.choice([
            BehaviorState.LYING,
            BehaviorState.STANDING,
        ], p=[0.6, 0.4])
        
        print(f"Starting simulation: {duration_hours:.1f} hours, "
              f"Initial state: {current_state.value}")
        
        while current_time < (total_minutes * 60):
            # Get state generator
            generator = self.generators[current_state]
            
            # Sample duration for this state
            duration_minutes = generator.sample_duration()
            
            # Don't exceed total duration
            remaining_minutes = (total_minutes * 60 - current_time) / 60
            duration_minutes = min(duration_minutes, remaining_minutes)
            
            if duration_minutes < 0.5:  # Less than 30 seconds
                break
            
            # Generate readings for this state
            readings = generator.generate(duration_minutes, start_time=current_time)
            
            # Apply stress overlay if requested
            if include_stress and np.random.random() < stress_probability:
                stress_duration = min(
                    StressBehaviorOverlay.sample_duration(),
                    duration_minutes
                )
                stress_n_samples = int(stress_duration * self.sampling_rate)
                
                if stress_n_samples > 0 and stress_n_samples <= len(readings):
                    # Apply stress to a random portion of the readings
                    start_idx = np.random.randint(0, max(1, len(readings) - stress_n_samples))
                    end_idx = start_idx + stress_n_samples
                    
                    stressed = self.stress_overlay.apply_stress(
                        readings[start_idx:end_idx],
                        stress_intensity=np.random.uniform(0.5, 1.2)
                    )
                    readings[start_idx:end_idx] = stressed
            
            all_readings.extend(readings)
            state_labels.extend([current_state.value] * len(readings))
            
            current_time = readings[-1].timestamp + (60.0 / self.sampling_rate)
            
            # Determine next state
            next_state = self.transition_manager.get_next_state(current_state)
            
            # If state is changing, create transition
            if next_state != current_state and current_time < (total_minutes * 60):
                if self.transition_manager.is_valid_transition(current_state, next_state):
                    # Generate first reading of next state for interpolation
                    next_generator = self.generators[next_state]
                    next_sample = next_generator.generate(0.1, start_time=current_time)
                    
                    if next_sample:
                        # Create transition
                        transition = self.transition_manager.create_transition(
                            current_state,
                            next_state,
                            readings[-1],
                            next_sample[0]
                        )
                        
                        if transition:
                            all_readings.extend(transition)
                            state_labels.extend(['transition'] * len(transition))
                            current_time = transition[-1].timestamp + (60.0 / self.sampling_rate)
                
                current_state = next_state
        
        # Convert to DataFrame
        df = self._readings_to_dataframe(all_readings, state_labels, start_datetime)
        
        print(f"Simulation complete: {len(df)} samples generated")
        print(f"State distribution:\n{df['state'].value_counts()}")
        
        return df
    
    def generate_single_state_data(
        self,
        state: BehaviorState,
        duration_minutes: float,
        start_datetime: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Generate sensor data for a single behavioral state.
        
        Args:
            state: Behavioral state to generate
            duration_minutes: Duration in minutes
            start_datetime: Starting datetime (default: now)
            
        Returns:
            DataFrame with sensor readings
        """
        if start_datetime is None:
            start_datetime = datetime.now()
        
        generator = self.generators[state]
        readings = generator.generate(duration_minutes, start_time=0.0)
        state_labels = [state.value] * len(readings)
        
        return self._readings_to_dataframe(readings, state_labels, start_datetime)
    
    def generate_labeled_dataset(
        self,
        samples_per_state: int = 100,
        duration_per_sample_minutes: float = 10.0,
        include_stress: bool = True,
    ) -> pd.DataFrame:
        """
        Generate a labeled dataset with balanced samples from each state.
        
        Useful for training machine learning models.
        
        Args:
            samples_per_state: Number of samples to generate per state
            duration_per_sample_minutes: Duration of each sample in minutes
            include_stress: Whether to include stress-overlaid samples
            
        Returns:
            DataFrame with labeled sensor data
        """
        all_data = []
        
        print(f"Generating labeled dataset: {samples_per_state} samples per state")
        
        # Generate samples for each state
        for state in self.generators.keys():
            print(f"  Generating {state.value} samples...")
            
            for i in range(samples_per_state):
                start_datetime = datetime.now() + timedelta(minutes=i * duration_per_sample_minutes)
                df = self.generate_single_state_data(
                    state,
                    duration_per_sample_minutes,
                    start_datetime
                )
                df['sample_id'] = f"{state.value}_{i}"
                all_data.append(df)
        
        # Generate stress samples if requested
        if include_stress:
            print(f"  Generating stress samples...")
            stress_samples = samples_per_state // 2  # Half as many stress samples
            
            for i in range(stress_samples):
                # Apply stress to random base state
                base_state = np.random.choice(list(self.generators.keys()))
                start_datetime = datetime.now() + timedelta(minutes=i * duration_per_sample_minutes)
                
                generator = self.generators[base_state]
                readings = generator.generate(duration_per_sample_minutes, start_time=0.0)
                
                # Apply stress overlay
                stressed = self.stress_overlay.apply_stress(
                    readings,
                    stress_intensity=np.random.uniform(0.5, 1.5)
                )
                
                state_labels = ['stress'] * len(stressed)
                df = self._readings_to_dataframe(stressed, state_labels, start_datetime)
                df['sample_id'] = f"stress_{i}"
                all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"Dataset complete: {len(combined_df)} total samples")
        print(f"State distribution:\n{combined_df['state'].value_counts()}")
        
        return combined_df
    
    def validate_generated_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that generated data meets quality criteria.
        
        Args:
            df: DataFrame with generated sensor data
            
        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        warnings = []
        
        # Check for missing values
        if df.isnull().any().any():
            warnings.append("Dataset contains missing values")
        
        # Check sensor value ranges
        sensor_ranges = {
            'temperature': (36.0, 42.0),
            'fxa': (-3.0, 3.0),
            'mya': (-3.0, 3.0),
            'rza': (-1.5, 1.5),
            'sxg': (-100.0, 100.0),
            'lyg': (-100.0, 100.0),
            'dzg': (-100.0, 100.0),
        }
        
        for sensor, (min_val, max_val) in sensor_ranges.items():
            if sensor in df.columns:
                if df[sensor].min() < min_val or df[sensor].max() > max_val:
                    warnings.append(
                        f"{sensor} values outside expected range "
                        f"[{min_val}, {max_val}]: "
                        f"actual [{df[sensor].min():.2f}, {df[sensor].max():.2f}]"
                    )
        
        # Check state distribution
        if 'state' in df.columns:
            state_counts = df['state'].value_counts()
            if len(state_counts) < 3:
                warnings.append("Dataset contains fewer than 3 different states")
        
        return len(warnings) == 0, warnings
    
    def _readings_to_dataframe(
        self,
        readings: List[SensorReading],
        state_labels: List[str],
        start_datetime: datetime
    ) -> pd.DataFrame:
        """
        Convert sensor readings to DataFrame with proper timestamps.
        
        Args:
            readings: List of SensorReading objects
            state_labels: List of state labels for each reading
            start_datetime: Starting datetime for the first reading
            
        Returns:
            DataFrame with sensor data and timestamps
        """
        data = {
            'timestamp': [],
            'temperature': [],
            'fxa': [],
            'mya': [],
            'rza': [],
            'sxg': [],
            'lyg': [],
            'dzg': [],
            'state': state_labels,
        }
        
        for reading in readings:
            # Convert relative timestamp to absolute datetime
            dt = start_datetime + timedelta(seconds=reading.timestamp)
            data['timestamp'].append(dt)
            data['temperature'].append(reading.temperature)
            data['fxa'].append(reading.fxa)
            data['mya'].append(reading.mya)
            data['rza'].append(reading.rza)
            data['sxg'].append(reading.sxg)
            data['lyg'].append(reading.lyg)
            data['dzg'].append(reading.dzg)
        
        df = pd.DataFrame(data)
        return df
    
    def export_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        include_state_labels: bool = True
    ):
        """
        Export generated data to CSV file.
        
        Args:
            df: DataFrame with sensor data
            filepath: Output file path
            include_state_labels: Whether to include state labels in export
        """
        if not include_state_labels and 'state' in df.columns:
            df = df.drop(columns=['state'])
        
        df.to_csv(filepath, index=False)
        print(f"Data exported to: {filepath}")
    
    def get_state_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each behavioral state in the dataset.
        
        Args:
            df: DataFrame with sensor data and state labels
            
        Returns:
            Dictionary of state statistics
        """
        if 'state' not in df.columns:
            return {}
        
        stats = {}
        
        for state in df['state'].unique():
            state_data = df[df['state'] == state]
            
            stats[state] = {
                'count': len(state_data),
                'duration_minutes': len(state_data) / self.sampling_rate,
                'avg_temperature': state_data['temperature'].mean(),
                'avg_rza': state_data['rza'].mean(),
                'std_fxa': state_data['fxa'].std(),
                'std_mya': state_data['mya'].std(),
                'max_gyro': max(
                    state_data['sxg'].abs().max(),
                    state_data['lyg'].abs().max(),
                    state_data['dzg'].abs().max()
                ),
            }
        
        return stats


class BatchSimulator:
    """
    Utility for generating multiple simulation runs in batch.
    
    Useful for creating large training datasets or testing different scenarios.
    """
    
    def __init__(self, engine: SimulationEngine):
        """
        Initialize batch simulator.
        
        Args:
            engine: SimulationEngine instance to use
        """
        self.engine = engine
    
    def generate_multi_animal_dataset(
        self,
        num_animals: int,
        hours_per_animal: float,
        output_dir: str,
        individual_files: bool = True,
    ) -> pd.DataFrame:
        """
        Generate simulated data for multiple animals.
        
        Args:
            num_animals: Number of animals to simulate
            hours_per_animal: Hours of data per animal
            output_dir: Directory to save output files
            individual_files: Whether to save individual CSV files per animal
            
        Returns:
            Combined DataFrame with all animal data
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        all_data = []
        
        print(f"Generating data for {num_animals} animals...")
        
        for animal_id in range(1, num_animals + 1):
            print(f"\nAnimal {animal_id}/{num_animals}")
            
            # Vary baseline temperature slightly per animal
            temp_variation = np.random.uniform(-0.3, 0.3)
            self.engine.baseline_temperature = 38.5 + temp_variation
            
            # Generate data
            df = self.engine.generate_continuous_data(
                duration_hours=hours_per_animal,
                include_stress=True,
                stress_probability=0.05
            )
            
            df['animal_id'] = f"animal_{animal_id:03d}"
            all_data.append(df)
            
            # Save individual file if requested
            if individual_files:
                filepath = os.path.join(output_dir, f"animal_{animal_id:03d}.csv")
                self.engine.export_to_csv(df, filepath)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined file
        combined_path = os.path.join(output_dir, "combined_dataset.csv")
        self.engine.export_to_csv(combined_df, combined_path)
        
        print(f"\nBatch simulation complete!")
        print(f"Total samples: {len(combined_df)}")
        
        return combined_df
