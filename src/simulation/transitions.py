"""
State transition logic for behavioral state simulation.

Handles smooth transitions between behavioral states with gradual sensor value
interpolation to ensure realistic state changes.
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum

from .states import (
    SensorReading,
    LyingStateGenerator,
    StandingStateGenerator,
    WalkingStateGenerator,
    RuminatingStateGenerator,
    FeedingStateGenerator,
)


class BehaviorState(Enum):
    """Enumeration of possible behavioral states."""
    LYING = "lying"
    STANDING = "standing"
    WALKING = "walking"
    RUMINATING_LYING = "ruminating_lying"
    RUMINATING_STANDING = "ruminating_standing"
    FEEDING = "feeding"


class StateTransitionManager:
    """
    Manages transitions between behavioral states with smooth interpolation.
    
    Implements realistic transition logic:
    - Lying → Standing: Gradual Rza increase over 5-15 seconds
    - Standing → Walking: Fxa rhythm onset, increased angular velocities
    - Walking → Standing/Feeding: Deceleration of rhythmic patterns
    - Ruminating: Can begin/end during lying or standing without state change
    - Stress Overlays: Sudden onset but gradual return to normal patterns
    """
    
    def __init__(self, sampling_rate: float = 1.0):
        """
        Initialize state transition manager.
        
        Args:
            sampling_rate: Samples per minute (default: 1.0 for 1 sample/minute)
        """
        self.sampling_rate = sampling_rate
        self.time_step = 60.0 / sampling_rate  # seconds per sample
        
        # Define valid transitions and their transition times (in seconds)
        self.transition_times = {
            (BehaviorState.LYING, BehaviorState.STANDING): (5, 15),
            (BehaviorState.STANDING, BehaviorState.LYING): (3, 8),
            (BehaviorState.STANDING, BehaviorState.WALKING): (2, 5),
            (BehaviorState.WALKING, BehaviorState.STANDING): (2, 5),
            (BehaviorState.WALKING, BehaviorState.FEEDING): (3, 7),
            (BehaviorState.STANDING, BehaviorState.FEEDING): (2, 5),
            (BehaviorState.FEEDING, BehaviorState.STANDING): (2, 5),
            (BehaviorState.FEEDING, BehaviorState.WALKING): (2, 5),
            # Ruminating transitions
            (BehaviorState.LYING, BehaviorState.RUMINATING_LYING): (1, 3),
            (BehaviorState.RUMINATING_LYING, BehaviorState.LYING): (1, 3),
            (BehaviorState.STANDING, BehaviorState.RUMINATING_STANDING): (1, 3),
            (BehaviorState.RUMINATING_STANDING, BehaviorState.STANDING): (1, 3),
        }
        
        # Transition probability matrix (from_state -> to_state)
        self.transition_probabilities = self._build_transition_probabilities()
    
    def _build_transition_probabilities(self) -> dict:
        """
        Build realistic transition probability matrix.
        
        Returns:
            Dictionary of {from_state: {to_state: probability}}
        """
        return {
            BehaviorState.LYING: {
                BehaviorState.LYING: 0.7,  # Tend to stay lying
                BehaviorState.STANDING: 0.2,
                BehaviorState.RUMINATING_LYING: 0.1,
            },
            BehaviorState.STANDING: {
                BehaviorState.STANDING: 0.4,
                BehaviorState.LYING: 0.2,
                BehaviorState.WALKING: 0.2,
                BehaviorState.FEEDING: 0.1,
                BehaviorState.RUMINATING_STANDING: 0.1,
            },
            BehaviorState.WALKING: {
                BehaviorState.WALKING: 0.5,
                BehaviorState.STANDING: 0.3,
                BehaviorState.FEEDING: 0.2,
            },
            BehaviorState.RUMINATING_LYING: {
                BehaviorState.RUMINATING_LYING: 0.8,  # Long rumination sessions
                BehaviorState.LYING: 0.2,
            },
            BehaviorState.RUMINATING_STANDING: {
                BehaviorState.RUMINATING_STANDING: 0.7,
                BehaviorState.STANDING: 0.3,
            },
            BehaviorState.FEEDING: {
                BehaviorState.FEEDING: 0.6,
                BehaviorState.STANDING: 0.2,
                BehaviorState.WALKING: 0.2,
            },
        }
    
    def get_next_state(self, current_state: BehaviorState) -> BehaviorState:
        """
        Sample the next behavioral state based on transition probabilities.
        
        Args:
            current_state: Current behavioral state
            
        Returns:
            Next behavioral state
        """
        probs = self.transition_probabilities.get(current_state, {})
        if not probs:
            return current_state
        
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        return np.random.choice(states, p=probabilities)
    
    def create_transition(
        self,
        from_state: BehaviorState,
        to_state: BehaviorState,
        start_reading: SensorReading,
        end_reading: SensorReading,
    ) -> list[SensorReading]:
        """
        Create smooth transition between two states.
        
        Args:
            from_state: Starting behavioral state
            to_state: Target behavioral state
            start_reading: Last reading from the previous state
            end_reading: First reading from the next state
            
        Returns:
            List of interpolated sensor readings for the transition
        """
        # Get transition time range
        transition_key = (from_state, to_state)
        if transition_key not in self.transition_times:
            # No defined transition - use instant transition
            return []
        
        min_time, max_time = self.transition_times[transition_key]
        transition_duration = np.random.uniform(min_time, max_time)
        
        # Calculate number of samples for transition
        n_samples = max(2, int((transition_duration / 60.0) * self.sampling_rate))
        
        # Create interpolated readings
        transition_readings = []
        for i in range(1, n_samples):
            # Linear interpolation factor
            t = i / n_samples
            
            # Apply different interpolation curves for different transitions
            if from_state == BehaviorState.LYING and to_state == BehaviorState.STANDING:
                # Gradual acceleration in Rza change (ease-in-out)
                t = self._ease_in_out(t)
            elif from_state == BehaviorState.STANDING and to_state == BehaviorState.WALKING:
                # Quick ramp-up for walking (ease-in)
                t = self._ease_in(t)
            elif from_state == BehaviorState.WALKING:
                # Gradual deceleration (ease-out)
                t = self._ease_out(t)
            
            timestamp = start_reading.timestamp + (i * transition_duration / n_samples)
            
            # Interpolate each sensor value
            temp = self._interpolate(start_reading.temperature, end_reading.temperature, t)
            fxa = self._interpolate(start_reading.fxa, end_reading.fxa, t)
            mya = self._interpolate(start_reading.mya, end_reading.mya, t)
            rza = self._interpolate(start_reading.rza, end_reading.rza, t)
            sxg = self._interpolate(start_reading.sxg, end_reading.sxg, t)
            lyg = self._interpolate(start_reading.lyg, end_reading.lyg, t)
            dzg = self._interpolate(start_reading.dzg, end_reading.dzg, t)
            
            # Add small noise to make transition more natural
            noise_scale = 0.05
            fxa += np.random.normal(0, noise_scale)
            mya += np.random.normal(0, noise_scale)
            sxg += np.random.normal(0, 2.0)
            lyg += np.random.normal(0, 2.0)
            dzg += np.random.normal(0, 2.0)
            
            transition_readings.append(SensorReading(
                timestamp=timestamp,
                temperature=temp,
                fxa=fxa,
                mya=mya,
                rza=rza,
                sxg=sxg,
                lyg=lyg,
                dzg=dzg
            ))
        
        return transition_readings
    
    def is_valid_transition(self, from_state: BehaviorState, to_state: BehaviorState) -> bool:
        """
        Check if a transition between states is valid.
        
        Args:
            from_state: Starting state
            to_state: Target state
            
        Returns:
            True if transition is valid, False otherwise
        """
        if from_state == to_state:
            return True
        
        return (from_state, to_state) in self.transition_times
    
    @staticmethod
    def _interpolate(start: float, end: float, t: float) -> float:
        """Linear interpolation between two values."""
        return start + (end - start) * t
    
    @staticmethod
    def _ease_in_out(t: float) -> float:
        """Ease-in-out interpolation curve (smooth acceleration and deceleration)."""
        return t * t * (3 - 2 * t)
    
    @staticmethod
    def _ease_in(t: float) -> float:
        """Ease-in interpolation curve (gradual acceleration)."""
        return t * t
    
    @staticmethod
    def _ease_out(t: float) -> float:
        """Ease-out interpolation curve (gradual deceleration)."""
        return t * (2 - t)
    
    def get_state_sequence_probabilities(
        self,
        sequence_length: int = 10,
        start_state: Optional[BehaviorState] = None
    ) -> list[BehaviorState]:
        """
        Generate a realistic sequence of behavioral states.
        
        Args:
            sequence_length: Number of states in the sequence
            start_state: Starting state (random if None)
            
        Returns:
            List of behavioral states
        """
        if start_state is None:
            # Start with common states
            start_state = np.random.choice([
                BehaviorState.LYING,
                BehaviorState.STANDING,
            ], p=[0.6, 0.4])
        
        sequence = [start_state]
        current = start_state
        
        for _ in range(sequence_length - 1):
            next_state = self.get_next_state(current)
            sequence.append(next_state)
            current = next_state
        
        return sequence


class TransitionValidator:
    """Validates that state transitions meet realistic criteria."""
    
    @staticmethod
    def validate_transition_smoothness(
        readings: list[SensorReading],
        max_jump_threshold: dict = None
    ) -> Tuple[bool, list[str]]:
        """
        Validate that sensor transitions are smooth without unrealistic jumps.
        
        Args:
            readings: List of sensor readings to validate
            max_jump_threshold: Dictionary of maximum allowed jumps per sensor
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if max_jump_threshold is None:
            max_jump_threshold = {
                'temperature': 0.5,  # °C
                'fxa': 2.0,  # m/s²
                'mya': 2.0,  # m/s²
                'rza': 0.5,  # g
                'sxg': 50.0,  # °/s
                'lyg': 50.0,  # °/s
                'dzg': 50.0,  # °/s
            }
        
        errors = []
        
        for i in range(1, len(readings)):
            prev = readings[i - 1]
            curr = readings[i]
            
            # Check each sensor for unrealistic jumps
            for sensor, threshold in max_jump_threshold.items():
                prev_val = getattr(prev, sensor)
                curr_val = getattr(curr, sensor)
                jump = abs(curr_val - prev_val)
                
                if jump > threshold:
                    errors.append(
                        f"Unrealistic jump in {sensor} at index {i}: "
                        f"{prev_val:.2f} -> {curr_val:.2f} (jump: {jump:.2f})"
                    )
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_state_duration(
        state: BehaviorState,
        duration_minutes: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that state duration is realistic.
        
        Args:
            state: Behavioral state
            duration_minutes: Duration in minutes
            
        Returns:
            Tuple of (is_valid, error message if invalid)
        """
        # Define reasonable duration ranges (min, max) in minutes
        duration_ranges = {
            BehaviorState.LYING: (10, 180),
            BehaviorState.STANDING: (2, 60),
            BehaviorState.WALKING: (1, 30),
            BehaviorState.RUMINATING_LYING: (10, 90),
            BehaviorState.RUMINATING_STANDING: (10, 90),
            BehaviorState.FEEDING: (10, 90),
        }
        
        min_dur, max_dur = duration_ranges.get(state, (0, float('inf')))
        
        if duration_minutes < min_dur:
            return False, f"{state.value} duration {duration_minutes:.1f} min is too short (min: {min_dur})"
        
        if duration_minutes > max_dur:
            return False, f"{state.value} duration {duration_minutes:.1f} min is too long (max: {max_dur})"
        
        return True, None
