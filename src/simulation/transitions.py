"""
State Transition Model

This module implements the probabilistic state machine for behavioral transitions.
It includes:
- Transition probability matrices between behavioral states
- Duration distributions for each state
- Smooth transition interpolation between states
- Time-of-day modulation of transition probabilities

Based on cattle behavior research:
- Lying duration: 30-120 minutes
- Standing duration: 5-30 minutes
- Walking duration: 2-15 minutes
- Ruminating duration: 20-60 minutes
- Feeding duration: 15-45 minutes

Transition probabilities (per hour):
- Lying → Standing: 15%
- Standing → Walking: 25%
- Walking → Feeding: 40%
- etc.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
from datetime import datetime

from .state_params import BehavioralState, SensorSignature, get_state_signature
from .temporal import TemporalPatternManager


@dataclass
class StateTransitionConfig:
    """Configuration for state transition probabilities and durations."""
    
    # Transition probability matrix (from_state -> to_state -> probability per hour)
    # These are BASE probabilities that get modified by time-of-day
    transition_matrix: Dict[BehavioralState, Dict[BehavioralState, float]]
    
    # Duration distributions (state -> (min_minutes, max_minutes, mean_minutes))
    duration_ranges: Dict[BehavioralState, Tuple[float, float, float]]
    
    # Smooth transition duration (seconds)
    transition_smoothing_time: float = 60.0  # 1 minute default


# Default transition probability matrix (per hour)
DEFAULT_TRANSITION_MATRIX = {
    BehavioralState.LYING: {
        BehavioralState.LYING: 0.85,      # Stay lying
        BehavioralState.STANDING: 0.10,   # Get up
        BehavioralState.WALKING: 0.02,    # Rare: get up and walk immediately
        BehavioralState.RUMINATING: 0.03, # Start ruminating while lying
        BehavioralState.FEEDING: 0.00,    # Can't feed while lying
    },
    BehavioralState.STANDING: {
        BehavioralState.LYING: 0.15,      # Lie down
        BehavioralState.STANDING: 0.50,   # Stay standing
        BehavioralState.WALKING: 0.25,    # Start walking
        BehavioralState.RUMINATING: 0.05, # Start ruminating while standing
        BehavioralState.FEEDING: 0.05,    # Start feeding
    },
    BehavioralState.WALKING: {
        BehavioralState.LYING: 0.05,      # Lie down after walking
        BehavioralState.STANDING: 0.30,   # Stop and stand
        BehavioralState.WALKING: 0.25,    # Keep walking
        BehavioralState.RUMINATING: 0.00, # Can't ruminate while walking
        BehavioralState.FEEDING: 0.40,    # Walk to food and start eating
    },
    BehavioralState.RUMINATING: {
        BehavioralState.LYING: 0.20,      # Lie down while ruminating
        BehavioralState.STANDING: 0.30,   # Stand up or keep standing
        BehavioralState.WALKING: 0.05,    # Stop ruminating and walk
        BehavioralState.RUMINATING: 0.40, # Continue ruminating
        BehavioralState.FEEDING: 0.05,    # Stop ruminating and eat
    },
    BehavioralState.FEEDING: {
        BehavioralState.LYING: 0.10,      # Finish eating and lie down
        BehavioralState.STANDING: 0.20,   # Finish eating and stand
        BehavioralState.WALKING: 0.15,    # Walk to new location
        BehavioralState.RUMINATING: 0.25, # Start ruminating after eating
        BehavioralState.FEEDING: 0.30,    # Continue feeding
    },
}


# Duration ranges for each state (min, max, mean) in minutes
DEFAULT_DURATION_RANGES = {
    BehavioralState.LYING: (30.0, 120.0, 60.0),
    BehavioralState.STANDING: (5.0, 30.0, 15.0),
    BehavioralState.WALKING: (2.0, 15.0, 5.0),
    BehavioralState.RUMINATING: (20.0, 60.0, 40.0),
    BehavioralState.FEEDING: (15.0, 45.0, 25.0),
}


class StateTransitionModel:
    """
    Manages behavioral state transitions with probabilistic model.
    
    This class handles:
    - Determining when to transition between states
    - Selecting next state based on probability matrix
    - Generating realistic state durations
    - Smooth interpolation between states
    """
    
    def __init__(self, 
                 config: Optional[StateTransitionConfig] = None,
                 temporal_manager: Optional[TemporalPatternManager] = None,
                 seed: Optional[int] = None):
        """
        Initialize the state transition model.
        
        Args:
            config: Transition configuration (uses defaults if None)
            temporal_manager: Temporal pattern manager for time-of-day effects
            seed: Random seed for reproducibility
        """
        if config is None:
            config = StateTransitionConfig(
                transition_matrix=DEFAULT_TRANSITION_MATRIX,
                duration_ranges=DEFAULT_DURATION_RANGES,
                transition_smoothing_time=60.0
            )
        
        self.config = config
        self.temporal_manager = temporal_manager or TemporalPatternManager()
        self.rng = np.random.default_rng(seed)
        
        # Current state tracking
        self.current_state: Optional[BehavioralState] = None
        self.time_in_state: float = 0.0  # minutes
        self.target_duration: float = 0.0  # minutes
        self.is_transitioning: bool = False
        self.transition_progress: float = 0.0  # 0.0 to 1.0
        self.previous_state: Optional[BehavioralState] = None
    
    def initialize_state(self, initial_state: Optional[BehavioralState] = None,
                        timestamp: Optional[datetime] = None) -> BehavioralState:
        """
        Initialize the simulation with a starting state.
        
        Args:
            initial_state: Starting state (random if None)
            timestamp: Current timestamp for time-of-day effects
            
        Returns:
            The initialized state
        """
        if initial_state is None:
            # Choose initial state based on time of day
            if timestamp:
                hour = self.temporal_manager.get_hour_of_day(timestamp)
                if self.temporal_manager.is_night_time(hour):
                    initial_state = BehavioralState.LYING
                elif self.temporal_manager.is_feeding_time(hour):
                    initial_state = self.rng.choice([BehavioralState.FEEDING, BehavioralState.STANDING])
                else:
                    initial_state = self.rng.choice([BehavioralState.STANDING, BehavioralState.WALKING])
            else:
                initial_state = BehavioralState.STANDING
        
        self.current_state = initial_state
        self.time_in_state = 0.0
        self.target_duration = self._sample_duration(initial_state)
        self.is_transitioning = False
        self.transition_progress = 0.0
        
        return initial_state
    
    def _sample_duration(self, state: BehavioralState) -> float:
        """
        Sample a duration for the given state from its distribution.
        
        Args:
            state: Behavioral state
            
        Returns:
            Duration in minutes
        """
        min_dur, max_dur, mean_dur = self.config.duration_ranges[state]
        
        # Use log-normal distribution for more realistic durations
        # (some states last much longer than others)
        std_dur = (max_dur - min_dur) / 4.0
        
        # Sample from truncated normal distribution
        duration = self.rng.normal(mean_dur, std_dur)
        duration = np.clip(duration, min_dur, max_dur)
        
        return duration
    
    def _get_modified_transition_probabilities(self, 
                                              from_state: BehavioralState,
                                              timestamp: datetime) -> Dict[BehavioralState, float]:
        """
        Get transition probabilities modified by time-of-day effects.
        
        Args:
            from_state: Current state
            timestamp: Current timestamp
            
        Returns:
            Dictionary of transition probabilities to each state
        """
        base_probs = self.config.transition_matrix[from_state].copy()
        
        # Get time-of-day preferences
        hour = self.temporal_manager.get_hour_of_day(timestamp)
        preferences = self.temporal_manager.get_state_preference_multipliers(hour)
        
        # Modify probabilities by preferences
        modified_probs = {}
        total = 0.0
        
        for to_state, base_prob in base_probs.items():
            # Apply preference multiplier
            modified_prob = base_prob * preferences[to_state]
            modified_probs[to_state] = modified_prob
            total += modified_prob
        
        # Normalize to sum to 1.0
        if total > 0:
            for state in modified_probs:
                modified_probs[state] /= total
        
        return modified_probs
    
    def should_transition(self, timestamp: datetime) -> bool:
        """
        Determine if a state transition should occur.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            True if should transition to new state
        """
        # Check if we've exceeded the target duration
        if self.time_in_state >= self.target_duration:
            return True
        
        # Small chance of early transition (adds variability)
        if self.time_in_state > self.target_duration * 0.5:
            early_transition_prob = 0.05  # 5% chance per minute
            if self.rng.random() < early_transition_prob:
                return True
        
        return False
    
    def select_next_state(self, timestamp: datetime) -> BehavioralState:
        """
        Select the next behavioral state based on transition probabilities.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Next behavioral state
        """
        probs = self._get_modified_transition_probabilities(self.current_state, timestamp)
        
        # Sample from probability distribution
        states = list(probs.keys())
        probabilities = [probs[s] for s in states]
        
        next_state = self.rng.choice(states, p=probabilities)
        return next_state
    
    def update(self, timestamp: datetime, time_step_minutes: float = 1.0) -> Tuple[BehavioralState, bool]:
        """
        Update the state machine for one time step.
        
        Args:
            timestamp: Current timestamp
            time_step_minutes: Time step size in minutes
            
        Returns:
            Tuple of (current_state, is_transitioning)
        """
        # Update time in current state
        self.time_in_state += time_step_minutes
        
        # Check if we should transition
        if not self.is_transitioning and self.should_transition(timestamp):
            # Start transition
            self.previous_state = self.current_state
            self.current_state = self.select_next_state(timestamp)
            self.is_transitioning = True
            self.transition_progress = 0.0
        
        # Update transition progress
        if self.is_transitioning:
            transition_minutes = self.config.transition_smoothing_time / 60.0
            self.transition_progress += time_step_minutes / transition_minutes
            
            if self.transition_progress >= 1.0:
                # Transition complete
                self.is_transitioning = False
                self.transition_progress = 0.0
                self.time_in_state = 0.0
                self.target_duration = self._sample_duration(self.current_state)
                self.previous_state = None
        
        return self.current_state, self.is_transitioning
    
    def get_interpolated_signature(self) -> SensorSignature:
        """
        Get interpolated sensor signature during transitions.
        
        During transitions, sensor values gradually change from previous state
        to current state.
        
        Returns:
            Interpolated sensor signature
        """
        if not self.is_transitioning or self.previous_state is None:
            return get_state_signature(self.current_state)
        
        # Get signatures for both states
        prev_sig = get_state_signature(self.previous_state)
        curr_sig = get_state_signature(self.current_state)
        
        # Interpolate using smooth transition function (ease-in-out)
        t = self.transition_progress
        # Smooth step function: 3t² - 2t³
        smooth_t = 3 * t * t - 2 * t * t * t
        
        # Create interpolated signature
        def interpolate_range(prev_range, curr_range):
            from .state_params import SensorRange
            return SensorRange(
                min_value=prev_range.min_value * (1 - smooth_t) + curr_range.min_value * smooth_t,
                max_value=prev_range.max_value * (1 - smooth_t) + curr_range.max_value * smooth_t,
                mean=prev_range.mean * (1 - smooth_t) + curr_range.mean * smooth_t,
                std=prev_range.std * (1 - smooth_t) + curr_range.std * smooth_t,
            )
        
        interpolated_sig = SensorSignature(
            temperature=interpolate_range(prev_sig.temperature, curr_sig.temperature),
            fxa=interpolate_range(prev_sig.fxa, curr_sig.fxa),
            mya=interpolate_range(prev_sig.mya, curr_sig.mya),
            rza=interpolate_range(prev_sig.rza, curr_sig.rza),
            sxg=interpolate_range(prev_sig.sxg, curr_sig.sxg),
            lyg=interpolate_range(prev_sig.lyg, curr_sig.lyg),
            dzg=interpolate_range(prev_sig.dzg, curr_sig.dzg),
            rhythmic_frequency=curr_sig.rhythmic_frequency,
            rhythmic_amplitude_scale=curr_sig.rhythmic_amplitude_scale,
        )
        
        return interpolated_sig
    
    def get_state_info(self) -> Dict:
        """
        Get current state information for debugging/logging.
        
        Returns:
            Dictionary with state information
        """
        return {
            'current_state': self.current_state.value if self.current_state else None,
            'previous_state': self.previous_state.value if self.previous_state else None,
            'time_in_state': self.time_in_state,
            'target_duration': self.target_duration,
            'is_transitioning': self.is_transitioning,
            'transition_progress': self.transition_progress,
        }
