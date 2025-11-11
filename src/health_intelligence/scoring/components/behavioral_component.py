"""
Behavioral Patterns Score Component

Calculates health score based on behavioral patterns:
- Ruminating frequency and duration
- Feeding regularity
- Stress behavior detection

Score Range: 0-25 points
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base_component import BaseScoreComponent, ComponentScore


class BehavioralScoreComponent(BaseScoreComponent):
    """
    Behavioral patterns scoring component.
    
    Placeholder Formula:
        score = 25 - (rumination_deficit * 5) - (stress_behavior_count * 10)
    
    Where:
        - rumination_deficit: Hours below optimal rumination time
        - stress_behavior_count: Number of detected stress behaviors
    
    Future Integration Points:
        - Individual behavioral baselines
        - Time-of-day feeding patterns
        - Social interaction metrics
        - Estrus detection integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize behavioral component.
        
        Args:
            config: Configuration dict with keys:
                - optimal_rumination_min: Min optimal rumination (minutes, default 400)
                - optimal_rumination_max: Max optimal rumination (minutes, default 600)
                - rumination_penalty_per_hour: Penalty per hour deficit (default 5.0)
                - stress_behavior_penalty: Penalty per stress incident (default 10.0)
        """
        super().__init__(config)
        
        # Load configuration with defaults
        self.optimal_rumination_min = self.config.get('optimal_rumination_min', 400)
        self.optimal_rumination_max = self.config.get('optimal_rumination_max', 600)
        self.rumination_penalty_per_hour = self.config.get('rumination_penalty_per_hour', 5.0)
        self.stress_behavior_penalty = self.config.get('stress_behavior_penalty', 10.0)
        self.max_stress_penalty = self.config.get('max_stress_penalty', 20.0)
        
        # Component weights
        self.feeding_weight = self.config.get('feeding_regularity_weight', 0.4)
        self.rumination_weight = self.config.get('rumination_weight', 0.4)
        self.stress_weight = self.config.get('stress_behavior_weight', 0.2)
    
    def get_required_columns(self) -> list:
        """Get required DataFrame columns."""
        return ['timestamp', 'behavioral_state']
    
    def calculate_score(
        self,
        cow_id: str,
        data: pd.DataFrame,
        stress_events: Optional[list] = None,
        feeding_pattern_score: Optional[float] = None,
        **kwargs
    ) -> ComponentScore:
        """
        Calculate behavioral patterns score.
        
        Args:
            cow_id: Animal identifier
            data: DataFrame with columns ['timestamp', 'behavioral_state']
            stress_events: List of stress behavior event dicts
            feeding_pattern_score: Pre-calculated feeding regularity score (0-1)
            **kwargs: Additional parameters
        
        Returns:
            ComponentScore with behavioral patterns score (0-25 points)
        """
        # Validate data
        is_valid, warnings = self.validate_data(data)
        if not is_valid:
            return ComponentScore(
                score=0.0,
                normalized_score=0.0,
                confidence=0.0,
                details={'error': 'Invalid or insufficient data'},
                warnings=warnings
            )
        
        details = {}
        score = 25.0  # Start with perfect score
        confidence = 0.8  # Base confidence
        
        # Calculate rumination time
        rumination_minutes = self._calculate_rumination_time(data)
        details['rumination_minutes'] = round(rumination_minutes, 1)
        details['optimal_rumination_range'] = [
            self.optimal_rumination_min,
            self.optimal_rumination_max
        ]
        
        # Calculate rumination deficit (in hours)
        rumination_deficit_minutes = 0.0
        if rumination_minutes < self.optimal_rumination_min:
            rumination_deficit_minutes = self.optimal_rumination_min - rumination_minutes
        elif rumination_minutes > self.optimal_rumination_max:
            # Excessive rumination might also be concerning
            rumination_deficit_minutes = (rumination_minutes - self.optimal_rumination_max) * 0.5
        
        rumination_deficit_hours = rumination_deficit_minutes / 60.0
        details['rumination_deficit_hours'] = round(rumination_deficit_hours, 2)
        
        # Apply rumination penalty (placeholder formula: deficit * 5)
        rumination_penalty = rumination_deficit_hours * self.rumination_penalty_per_hour
        rumination_penalty = min(rumination_penalty, 15.0)  # Cap at 15 points
        score -= rumination_penalty
        details['rumination_penalty'] = round(rumination_penalty, 2)
        
        # Calculate feeding regularity score
        if feeding_pattern_score is not None:
            details['feeding_pattern_score'] = round(feeding_pattern_score, 2)
            # Good feeding = bonus, poor feeding = penalty
            feeding_adjustment = (feeding_pattern_score - 0.5) * 4.0  # -2 to +2 points
            score += feeding_adjustment
            details['feeding_adjustment'] = round(feeding_adjustment, 2)
        else:
            # Calculate basic feeding metrics
            feeding_minutes = self._calculate_feeding_time(data)
            details['feeding_minutes'] = round(feeding_minutes, 1)
            
            # Expected feeding: 3-5 hours per day
            if 180 <= feeding_minutes <= 300:  # 3-5 hours
                feeding_bonus = 2.0
                score += feeding_bonus
                details['feeding_bonus'] = feeding_bonus
        
        # Count stress behaviors
        stress_count = 0
        if stress_events:
            stress_count = len(stress_events)
            details['stress_events'] = stress_events
        else:
            # Detect stress from erratic motion patterns (simple heuristic)
            stress_count = self._detect_stress_behaviors(data)
        
        details['stress_behavior_count'] = stress_count
        
        # Apply stress penalty (placeholder formula: stress_count * 10)
        stress_penalty = min(
            stress_count * self.stress_behavior_penalty,
            self.max_stress_penalty
        )
        score -= stress_penalty
        details['stress_penalty'] = round(stress_penalty, 2)
        
        # Calculate behavioral diversity (Shannon entropy)
        diversity = self._calculate_behavioral_diversity(data)
        details['behavioral_diversity'] = round(diversity, 3)
        
        # Bonus for healthy behavioral diversity
        if 0.6 <= diversity <= 0.9:  # Good variety without chaos
            diversity_bonus = 1.0
            score += diversity_bonus
            details['diversity_bonus'] = diversity_bonus
        
        # Adjust confidence based on data quality
        data_completeness = 1.0 - (data['behavioral_state'].isna().sum() / len(data))
        confidence *= data_completeness
        details['data_completeness'] = round(data_completeness, 3)
        
        # Clamp score to valid range [0, 25]
        score = max(0.0, min(25.0, score))
        normalized_score = self.normalize_score(score)
        
        details['raw_score'] = round(score, 2)
        details['formula'] = 'placeholder: 25 - (rumination_deficit * 5) - (stress_behavior_count * 10)'
        
        return ComponentScore(
            score=score,
            normalized_score=normalized_score,
            confidence=confidence,
            details=details,
            warnings=warnings
        )
    
    def _calculate_rumination_time(self, data: pd.DataFrame) -> float:
        """
        Calculate total rumination time in minutes.
        
        Args:
            data: DataFrame with 'behavioral_state' column
        
        Returns:
            Total rumination minutes
        """
        if 'behavioral_state' not in data.columns:
            return 0.0
        
        rumination_mask = data['behavioral_state'] == 'ruminating'
        rumination_minutes = rumination_mask.sum()  # Assuming 1-minute intervals
        
        return rumination_minutes
    
    def _calculate_feeding_time(self, data: pd.DataFrame) -> float:
        """
        Calculate total feeding time in minutes.
        
        Args:
            data: DataFrame with 'behavioral_state' column
        
        Returns:
            Total feeding minutes
        """
        if 'behavioral_state' not in data.columns:
            return 0.0
        
        feeding_mask = data['behavioral_state'] == 'feeding'
        feeding_minutes = feeding_mask.sum()  # Assuming 1-minute intervals
        
        return feeding_minutes
    
    def _detect_stress_behaviors(self, data: pd.DataFrame) -> int:
        """
        Detect potential stress behaviors from state transitions.
        
        Args:
            data: DataFrame with 'behavioral_state' column
        
        Returns:
            Count of potential stress incidents
        """
        if 'behavioral_state' not in data.columns:
            return 0
        
        # Count rapid state changes (proxy for erratic behavior)
        state_changes = (data['behavioral_state'] != data['behavioral_state'].shift(1)).sum()
        
        # More than 1 transition per 10 minutes might indicate stress
        expected_transitions = len(data) / 10
        excess_transitions = max(0, state_changes - expected_transitions)
        
        # Convert to stress incident count (rough estimate)
        stress_count = int(excess_transitions / 5)
        
        return stress_count
    
    def _calculate_behavioral_diversity(self, data: pd.DataFrame) -> float:
        """
        Calculate behavioral diversity using Shannon entropy.
        
        Args:
            data: DataFrame with 'behavioral_state' column
        
        Returns:
            Shannon entropy (0-1 normalized)
        """
        if 'behavioral_state' not in data.columns:
            return 0.0
        
        # Calculate state proportions
        state_counts = data['behavioral_state'].value_counts()
        proportions = state_counts / len(data)
        
        # Calculate Shannon entropy
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))
        
        # Normalize by maximum possible entropy (log2 of number of states)
        max_entropy = np.log2(len(proportions))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
