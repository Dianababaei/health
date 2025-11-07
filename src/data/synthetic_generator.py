"""
Synthetic Data Generator for Animal Health Sensor Data

This module generates realistic synthetic sensor data for different animal behaviors
with configurable noise levels and smooth transitions between behaviors.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import sys
import os

# Add config directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.behavior_patterns import get_behavior_params, get_all_behaviors, validate_behavior


class SyntheticDataGenerator:
    """
    Generate synthetic time-series sensor data for animal behavior monitoring.
    
    Supports 7 sensor channels sampled at 1-minute intervals:
    - temp: Body temperature (°C)
    - Fxa, Mya, Rza: Acceleration on X, Y, Z axes (m/s²)
    - Sxg, Lyg, Dzg: Angular velocity on X, Y, Z axes (rad/s)
    """
    
    CHANNELS = ['temp', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg']
    SAMPLING_INTERVAL_MINUTES = 1
    
    def __init__(self):
        """Initialize the synthetic data generator."""
        self.behaviors = get_all_behaviors()
        
    def generate(
        self,
        behavior: str,
        duration_minutes: int,
        start_time: Union[str, datetime] = None,
        noise_level: float = 0.1,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data for a single behavior.
        
        Parameters
        ----------
        behavior : str
            Name of behavior to generate (e.g., 'lying', 'standing', 'walking')
        duration_minutes : int
            Duration of behavior in minutes
        start_time : str or datetime, optional
            Start timestamp (default: '2024-01-01 00:00:00')
        noise_level : float, optional
            Fraction of signal std to use as noise (default: 0.1)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: timestamp, temp, Fxa, Mya, Rza, Sxg, Lyg, Dzg
            Also includes metadata as attributes
        """
        # Input validation
        self._validate_inputs(behavior, duration_minutes, noise_level)
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
        
        # Parse start time
        if start_time is None:
            start_time = datetime(2024, 1, 1, 0, 0, 0)
        elif isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        
        # Generate timestamps
        timestamps = pd.date_range(
            start=start_time,
            periods=duration_minutes,
            freq=f'{self.SAMPLING_INTERVAL_MINUTES}min'
        )
        
        # Get behavior parameters
        params = get_behavior_params(behavior)
        
        # Generate base signals for each channel
        data = {}
        for channel in self.CHANNELS:
            channel_params = params['parameters'][channel]
            base_signal = self._generate_base_signal(
                channel,
                duration_minutes,
                channel_params,
                params.get('frequencies', {})
            )
            
            # Add Gaussian noise
            noise_std = channel_params['std'] * noise_level
            noise = np.random.normal(0, noise_std, size=duration_minutes)
            data[channel] = base_signal + noise
        
        # Apply physical constraints
        data = self._apply_constraints(data)
        
        # Create DataFrame
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = 'timestamp'
        df = df.reset_index()
        
        # Add metadata
        df.attrs['behavior'] = behavior
        df.attrs['duration_minutes'] = duration_minutes
        df.attrs['noise_level'] = noise_level
        df.attrs['seed'] = seed
        df.attrs['generation_time'] = datetime.now().isoformat()
        
        return df
    
    def generate_sequence(
        self,
        sequence: List[Dict],
        start_time: Union[str, datetime] = None,
        noise_level: float = 0.1,
        transition_duration: int = 3,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic data for a sequence of behaviors with smooth transitions.
        
        Parameters
        ----------
        sequence : list of dict
            List of behavior specifications, each containing:
            - 'behavior': behavior name
            - 'duration': duration in minutes
        start_time : str or datetime, optional
            Start timestamp
        noise_level : float, optional
            Fraction of signal std to use as noise
        transition_duration : int, optional
            Duration of transition between behaviors in minutes (default: 3)
        seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all behaviors and smooth transitions
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Parse start time
        if start_time is None:
            start_time = datetime(2024, 1, 1, 0, 0, 0)
        elif isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        
        # Generate each behavior segment
        segments = []
        current_time = start_time
        
        for i, segment_spec in enumerate(sequence):
            behavior = segment_spec['behavior']
            duration = segment_spec.get('duration', 30)
            
            # Generate this segment
            segment_df = self.generate(
                behavior=behavior,
                duration_minutes=duration,
                start_time=current_time,
                noise_level=noise_level,
                seed=None  # Don't reset seed for each segment
            )
            
            segments.append(segment_df)
            current_time += timedelta(minutes=duration)
        
        # Concatenate all segments
        combined_df = pd.concat(segments, ignore_index=True)
        
        # Apply smooth transitions between segments
        combined_df = self._apply_transitions(
            combined_df, 
            sequence, 
            transition_duration
        )
        
        # Add metadata
        combined_df.attrs['sequence'] = sequence
        combined_df.attrs['noise_level'] = noise_level
        combined_df.attrs['transition_duration'] = transition_duration
        combined_df.attrs['seed'] = seed
        combined_df.attrs['generation_time'] = datetime.now().isoformat()
        
        return combined_df
    
    def _generate_base_signal(
        self,
        channel: str,
        duration: int,
        params: Dict,
        frequencies: Dict
    ) -> np.ndarray:
        """
        Generate base signal for a channel including rhythmic components.
        
        Parameters
        ----------
        channel : str
            Channel name
        duration : int
            Number of time points
        params : dict
            Channel parameters with 'mean' and 'std'
        frequencies : dict
            Frequency components for rhythmic behaviors
            
        Returns
        -------
        np.ndarray
            Base signal
        """
        # Start with mean + random variation
        mean = params['mean']
        std = params['std']
        
        # Generate base signal with some random walk characteristics
        base = np.random.normal(mean, std * 0.3, size=duration)
        
        # Add frequency components if present
        if channel in frequencies:
            time_points = np.arange(duration) / 60.0  # Convert to hours
            
            for freq_spec in frequencies[channel]:
                freq_hz = freq_spec['freq'] / 60.0  # Convert from cycles/min to Hz
                amplitude = freq_spec['amplitude']
                
                # Add sinusoidal component
                phase = np.random.uniform(0, 2 * np.pi)  # Random phase
                sine_wave = amplitude * np.sin(2 * np.pi * freq_hz * time_points * 60 + phase)
                base += sine_wave
        
        return base
    
    def _apply_constraints(self, data: Dict) -> Dict:
        """
        Apply physical constraints to sensor data.
        
        Parameters
        ----------
        data : dict
            Dictionary of channel data
            
        Returns
        -------
        dict
            Constrained data
        """
        # Temperature must be positive and in reasonable range
        data['temp'] = np.clip(data['temp'], 35.0, 42.0)
        
        # Acceleration magnitudes should be reasonable (not exceeding 5g except Rza with gravity)
        for acc_channel in ['Fxa', 'Mya']:
            data[acc_channel] = np.clip(data[acc_channel], -49.0, 49.0)
        
        # Rza can include gravity component
        data['Rza'] = np.clip(data['Rza'], -15.0, 20.0)
        
        # Angular velocities should be reasonable (not exceeding 2 rad/s for neck movements)
        for gyro_channel in ['Sxg', 'Lyg', 'Dzg']:
            data[gyro_channel] = np.clip(data[gyro_channel], -2.0, 2.0)
        
        return data
    
    def _apply_transitions(
        self,
        df: pd.DataFrame,
        sequence: List[Dict],
        transition_duration: int
    ) -> pd.DataFrame:
        """
        Apply smooth transitions between behavior segments.
        
        Parameters
        ----------
        df : pd.DataFrame
            Combined DataFrame
        sequence : list
            Sequence specification
        transition_duration : int
            Duration of transition in minutes
            
        Returns
        -------
        pd.DataFrame
            DataFrame with smooth transitions
        """
        if len(sequence) <= 1:
            return df
        
        df = df.copy()
        
        # Find transition points
        current_idx = 0
        for i in range(len(sequence) - 1):
            segment_duration = sequence[i].get('duration', 30)
            transition_start = current_idx + segment_duration - transition_duration // 2
            transition_end = transition_start + transition_duration
            
            # Make sure indices are valid
            transition_start = max(0, min(transition_start, len(df) - 1))
            transition_end = max(0, min(transition_end, len(df)))
            
            if transition_end > transition_start:
                # Apply sigmoid transition
                for channel in self.CHANNELS:
                    if transition_start > 0 and transition_end < len(df):
                        start_val = df.loc[transition_start, channel]
                        end_val = df.loc[transition_end, channel]
                        
                        # Create sigmoid blend
                        t = np.linspace(-3, 3, transition_end - transition_start)
                        sigmoid = 1 / (1 + np.exp(-t))
                        
                        # Interpolate
                        blended = start_val * (1 - sigmoid) + end_val * sigmoid
                        df.loc[transition_start:transition_end-1, channel] = blended
            
            current_idx += segment_duration
        
        return df
    
    def _validate_inputs(
        self,
        behavior: str,
        duration_minutes: int,
        noise_level: float
    ):
        """Validate input parameters."""
        if not validate_behavior(behavior):
            raise ValueError(
                f"Unknown behavior: '{behavior}'. "
                f"Valid behaviors: {self.behaviors}"
            )
        
        if duration_minutes <= 0:
            raise ValueError(f"Duration must be positive, got {duration_minutes}")
        
        if noise_level < 0:
            raise ValueError(f"Noise level must be non-negative, got {noise_level}")
    
    def export_to_csv(
        self,
        df: pd.DataFrame,
        filepath: str,
        include_metadata: bool = True
    ):
        """
        Export DataFrame to CSV file.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to export
        filepath : str
            Output file path
        include_metadata : bool, optional
            Whether to include metadata as comments (default: True)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        if include_metadata and hasattr(df, 'attrs') and df.attrs:
            # Write metadata as comments
            with open(filepath, 'w') as f:
                f.write("# Synthetic Sensor Data\n")
                for key, value in df.attrs.items():
                    f.write(f"# {key}: {value}\n")
                f.write("\n")
            
            # Append data
            df.to_csv(filepath, mode='a', index=False, date_format='%Y-%m-%d %H:%M:%S')
        else:
            df.to_csv(filepath, index=False, date_format='%Y-%m-%d %H:%M:%S')
