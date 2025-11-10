"""
State Transition Smoothing Module

This module implements temporal consistency filters to reduce classification jitter
in behavioral state detection. Includes:

- Minimum state duration: Require state to persist for N consecutive minutes
- Transition probability matrix: Use expected state transitions
- Sliding window voting: Majority vote over N-minute window
- Confidence thresholding: Reject low-confidence predictions

Reduces single-minute state flips and improves classification stability.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque, Counter
from dataclasses import dataclass


class BehavioralState(Enum):
    """Enumeration of cattle behavioral states."""
    LYING = "lying"
    STANDING = "standing"
    WALKING = "walking"
    RUMINATING = "ruminating"
    FEEDING = "feeding"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"


@dataclass
class SmoothedClassification:
    """Result of smoothed classification."""
    state: BehavioralState
    confidence: float
    original_state: BehavioralState
    original_confidence: float
    smoothing_applied: bool
    votes: Optional[Dict[str, int]] = None


class StateTransitionSmoother:
    """
    Temporal consistency filter for behavioral state classifications.
    
    Implements multiple smoothing strategies to reduce classification jitter:
    1. Minimum duration filter: State must persist N samples
    2. Sliding window voting: Majority vote over window
    3. Transition probability filtering: Penalize unlikely transitions
    4. Confidence thresholding: Reject low-confidence predictions
    
    Attributes:
        min_duration (int): Minimum samples for state confirmation
        window_size (int): Sliding window size for voting
        confidence_threshold (float): Minimum confidence to accept new state
        use_transition_matrix (bool): Enable transition probability filtering
    """
    
    # Expected transition probability matrix (from literature/domain knowledge)
    # Higher values = more likely transition
    TRANSITION_PROBABILITIES = {
        'lying': {
            'lying': 0.95,
            'standing': 0.04,
            'walking': 0.005,
            'ruminating': 0.003,
            'feeding': 0.001,
            'transition': 0.001,
            'uncertain': 0.0
        },
        'standing': {
            'lying': 0.02,
            'standing': 0.85,
            'walking': 0.08,
            'ruminating': 0.02,
            'feeding': 0.02,
            'transition': 0.01,
            'uncertain': 0.0
        },
        'walking': {
            'lying': 0.01,
            'standing': 0.15,
            'walking': 0.70,
            'ruminating': 0.01,
            'feeding': 0.12,
            'transition': 0.01,
            'uncertain': 0.0
        },
        'ruminating': {
            'lying': 0.05,
            'standing': 0.10,
            'walking': 0.01,
            'ruminating': 0.82,
            'feeding': 0.01,
            'transition': 0.01,
            'uncertain': 0.0
        },
        'feeding': {
            'lying': 0.01,
            'standing': 0.10,
            'walking': 0.05,
            'ruminating': 0.02,
            'feeding': 0.80,
            'transition': 0.02,
            'uncertain': 0.0
        },
        'transition': {
            'lying': 0.20,
            'standing': 0.40,
            'walking': 0.15,
            'ruminating': 0.05,
            'feeding': 0.10,
            'transition': 0.10,
            'uncertain': 0.0
        },
        'uncertain': {
            'lying': 0.15,
            'standing': 0.20,
            'walking': 0.10,
            'ruminating': 0.05,
            'feeding': 0.10,
            'transition': 0.20,
            'uncertain': 0.20
        }
    }
    
    def __init__(
        self,
        min_duration: int = 2,
        window_size: int = 5,
        confidence_threshold: float = 0.6,
        use_transition_matrix: bool = True,
        use_sliding_window: bool = True
    ):
        """
        Initialize state transition smoother.
        
        Args:
            min_duration: Minimum consecutive samples before confirming state change (default: 2)
            window_size: Sliding window size for majority voting (default: 5)
            confidence_threshold: Minimum confidence to accept prediction (default: 0.6)
            use_transition_matrix: Enable transition probability filtering (default: True)
            use_sliding_window: Enable sliding window voting (default: True)
        """
        self.min_duration = min_duration
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.use_transition_matrix = use_transition_matrix
        self.use_sliding_window = use_sliding_window
        
        # State history for smoothing
        self.state_history: deque = deque(maxlen=window_size * 2)
        self.confidence_history: deque = deque(maxlen=window_size * 2)
        
        # Current stable state
        self.current_stable_state: Optional[BehavioralState] = None
        self.state_duration: int = 0
    
    def smooth_single(
        self,
        state: BehavioralState,
        confidence: float,
        timestamp: Optional[pd.Timestamp] = None
    ) -> SmoothedClassification:
        """
        Smooth a single classification using temporal consistency filters.
        
        Args:
            state: Predicted behavioral state
            confidence: Confidence score (0.0-1.0)
            timestamp: Optional timestamp
            
        Returns:
            SmoothedClassification with potentially modified state
        """
        original_state = state
        original_confidence = confidence
        smoothing_applied = False
        
        # Add to history
        self.state_history.append(state)
        self.confidence_history.append(confidence)
        
        # Step 1: Confidence thresholding
        if confidence < self.confidence_threshold:
            # Low confidence - fall back to previous stable state
            if self.current_stable_state is not None:
                state = self.current_stable_state
                smoothing_applied = True
        
        # Step 2: Transition probability filtering
        if self.use_transition_matrix and self.current_stable_state is not None:
            transition_prob = self._get_transition_probability(
                self.current_stable_state,
                state
            )
            
            # Penalize unlikely transitions by requiring higher confidence
            required_confidence = 1.0 - transition_prob
            if confidence < required_confidence:
                state = self.current_stable_state
                smoothing_applied = True
        
        # Step 3: Minimum duration filter
        if state == self.current_stable_state:
            # Same state continues
            self.state_duration += 1
        else:
            # State change detected
            if self.state_duration < self.min_duration and self.current_stable_state is not None:
                # Previous state duration too short - keep it
                state = self.current_stable_state
                self.state_duration += 1
                smoothing_applied = True
            else:
                # Accept state change
                self.current_stable_state = state
                self.state_duration = 1
        
        # Step 4: Sliding window voting (if enough history)
        votes = None
        if self.use_sliding_window and len(self.state_history) >= self.window_size:
            recent_states = list(self.state_history)[-self.window_size:]
            votes = Counter([s.value for s in recent_states])
            majority_state_str = votes.most_common(1)[0][0]
            majority_state = BehavioralState(majority_state_str)
            
            # If majority differs significantly from current, use majority
            majority_count = votes[majority_state_str]
            if majority_count >= (self.window_size // 2 + 1):
                if state != majority_state:
                    state = majority_state
                    smoothing_applied = True
        
        # Update current stable state
        self.current_stable_state = state
        
        return SmoothedClassification(
            state=state,
            confidence=confidence,
            original_state=original_state,
            original_confidence=original_confidence,
            smoothing_applied=smoothing_applied,
            votes=votes
        )
    
    def smooth_batch(
        self,
        classifications: pd.DataFrame,
        state_column: str = 'state',
        confidence_column: str = 'confidence'
    ) -> pd.DataFrame:
        """
        Smooth a batch of classifications.
        
        Args:
            classifications: DataFrame with state and confidence columns
            state_column: Name of state column
            confidence_column: Name of confidence column
            
        Returns:
            DataFrame with smoothed states and additional columns:
            - smoothed_state: Final smoothed state
            - original_state: Original predicted state
            - smoothing_applied: Boolean flag
        """
        if state_column not in classifications.columns:
            raise ValueError(f"State column '{state_column}' not found")
        if confidence_column not in classifications.columns:
            raise ValueError(f"Confidence column '{confidence_column}' not found")
        
        # Reset smoother state
        self.reset()
        
        results = []
        for idx, row in classifications.iterrows():
            # Convert state string to enum if needed
            if isinstance(row[state_column], str):
                state = BehavioralState(row[state_column])
            else:
                state = row[state_column]
            
            confidence = row[confidence_column]
            
            # Smooth single classification
            smoothed = self.smooth_single(state, confidence)
            
            results.append({
                'smoothed_state': smoothed.state.value,
                'original_state': smoothed.original_state.value,
                'smoothing_applied': smoothed.smoothing_applied,
                'confidence': confidence
            })
        
        # Create output dataframe
        result_df = classifications.copy()
        result_df['smoothed_state'] = [r['smoothed_state'] for r in results]
        result_df['original_state'] = [r['original_state'] for r in results]
        result_df['smoothing_applied'] = [r['smoothing_applied'] for r in results]
        
        return result_df
    
    def _get_transition_probability(
        self,
        from_state: BehavioralState,
        to_state: BehavioralState
    ) -> float:
        """
        Get transition probability between two states.
        
        Args:
            from_state: Current state
            to_state: Target state
            
        Returns:
            Probability (0.0-1.0)
        """
        from_key = from_state.value
        to_key = to_state.value
        
        if from_key in self.TRANSITION_PROBABILITIES:
            if to_key in self.TRANSITION_PROBABILITIES[from_key]:
                return self.TRANSITION_PROBABILITIES[from_key][to_key]
        
        # Default low probability for unknown transitions
        return 0.01
    
    def reset(self):
        """Reset smoother state (clear history)."""
        self.state_history.clear()
        self.confidence_history.clear()
        self.current_stable_state = None
        self.state_duration = 0
    
    def get_statistics(self) -> Dict:
        """Get smoother statistics."""
        if len(self.state_history) == 0:
            return {
                'history_length': 0,
                'current_state': None,
                'state_duration': 0
            }
        
        return {
            'history_length': len(self.state_history),
            'current_state': self.current_stable_state.value if self.current_stable_state else None,
            'state_duration': self.state_duration,
            'avg_confidence': np.mean(list(self.confidence_history)) if self.confidence_history else 0.0
        }
