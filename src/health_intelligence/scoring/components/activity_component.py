"""
Activity Level Score Component

Calculates health score based on activity metrics:
- Movement intensity compared to baseline
- Activity duration (adequate walking, standing time)
- Prolonged inactivity incidents

Score Range: 0-25 points
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base_component import BaseScoreComponent, ComponentScore


class ActivityScoreComponent(BaseScoreComponent):
    """
    Activity level scoring component.
    
    Placeholder Formula:
        score = 25 - (activity_deviation * 8) - (inactivity_count * 7)
    
    Where:
        - activity_deviation: Deviation from baseline activity level (0-1 scale)
        - inactivity_count: Number of prolonged inactivity incidents
    
    Future Integration Points:
        - Time-of-day activity patterns
        - Breed-specific activity baselines
        - Weather-adjusted activity expectations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize activity component.
        
        Args:
            config: Configuration dict with keys:
                - optimal_deviation: Activity deviation for full points (default 0.15)
                - max_deviation: Activity deviation for zero points (default 0.50)
                - inactivity_penalty_per_incident: Points per inactivity (default 7.0)
                - duration_bonus_enabled: Add bonus for adequate duration (default True)
        """
        super().__init__(config)
        
        # Load configuration with defaults
        self.optimal_deviation = self.config.get('optimal_deviation', 0.15)
        self.max_deviation = self.config.get('max_deviation', 0.50)
        self.inactivity_penalty = self.config.get('inactivity_penalty_per_incident', 7.0)
        self.max_inactivity_penalty = self.config.get('max_inactivity_penalty', 20.0)
        self.duration_bonus_enabled = self.config.get('duration_bonus_enabled', True)
        self.min_active_hours = self.config.get('min_active_hours_per_day', 8)
    
    def get_required_columns(self) -> list:
        """Get required DataFrame columns."""
        return ['timestamp', 'movement_intensity']
    
    def calculate_score(
        self,
        cow_id: str,
        data: pd.DataFrame,
        baseline_activity: Optional[float] = None,
        inactivity_events: Optional[list] = None,
        behavioral_states: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> ComponentScore:
        """
        Calculate activity level score.
        
        Args:
            cow_id: Animal identifier
            data: DataFrame with columns ['timestamp', 'movement_intensity']
            baseline_activity: Baseline activity level (0-1), defaults to mean if not provided
            inactivity_events: List of prolonged inactivity event dicts
            behavioral_states: Optional DataFrame with behavioral state data
            **kwargs: Additional parameters
        
        Returns:
            ComponentScore with activity level score (0-25 points)
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
        
        # Calculate baseline if not provided
        if baseline_activity is None:
            baseline_activity = data['movement_intensity'].mean()
            details['baseline_source'] = 'calculated_mean'
            confidence *= 0.9
        else:
            details['baseline_source'] = 'provided'
        
        details['baseline_activity'] = round(baseline_activity, 3)
        
        # Calculate activity deviation from baseline
        current_activity = data['movement_intensity'].mean()
        activity_deviation = abs(current_activity - baseline_activity)
        
        details['current_activity'] = round(current_activity, 3)
        details['activity_deviation'] = round(activity_deviation, 3)
        details['deviation_threshold_optimal'] = self.optimal_deviation
        details['deviation_threshold_max'] = self.max_deviation
        
        # Apply deviation penalty using placeholder formula
        # Linear interpolation between optimal and max deviation
        if activity_deviation <= self.optimal_deviation:
            deviation_penalty = 0.0
        elif activity_deviation >= self.max_deviation:
            deviation_penalty = 8.0  # Maximum penalty from deviation
        else:
            # Linear scale between optimal and max
            deviation_ratio = (activity_deviation - self.optimal_deviation) / (
                self.max_deviation - self.optimal_deviation
            )
            deviation_penalty = deviation_ratio * 8.0
        
        score -= deviation_penalty
        details['deviation_penalty'] = round(deviation_penalty, 2)
        
        # Count inactivity incidents
        inactivity_count = 0
        if inactivity_events:
            inactivity_count = len(inactivity_events)
            details['inactivity_events'] = inactivity_events
        
        details['inactivity_count'] = inactivity_count
        
        # Apply inactivity penalty (placeholder formula: inactivity_count * 7)
        inactivity_penalty = min(
            inactivity_count * self.inactivity_penalty,
            self.max_inactivity_penalty
        )
        score -= inactivity_penalty
        details['inactivity_penalty'] = round(inactivity_penalty, 2)
        
        # Calculate activity duration bonus if enabled
        duration_bonus = 0.0
        if self.duration_bonus_enabled and behavioral_states is not None:
            active_hours = self._calculate_active_hours(behavioral_states)
            details['active_hours'] = round(active_hours, 2)
            
            if active_hours >= self.min_active_hours:
                # Bonus for meeting minimum activity duration
                duration_bonus = 2.0
                details['duration_bonus'] = duration_bonus
                score += duration_bonus
        
        # Adjust confidence based on data quality
        data_completeness = 1.0 - (data['movement_intensity'].isna().sum() / len(data))
        confidence *= data_completeness
        details['data_completeness'] = round(data_completeness, 3)
        
        # Clamp score to valid range [0, 25]
        score = max(0.0, min(25.0, score))
        normalized_score = self.normalize_score(score)
        
        details['raw_score'] = round(score, 2)
        details['formula'] = 'placeholder: 25 - (activity_deviation * 8) - (inactivity_count * 7)'
        
        return ComponentScore(
            score=score,
            normalized_score=normalized_score,
            confidence=confidence,
            details=details,
            warnings=warnings
        )
    
    def _calculate_active_hours(self, behavioral_states: pd.DataFrame) -> float:
        """
        Calculate total active hours from behavioral state data.
        
        Args:
            behavioral_states: DataFrame with 'behavioral_state' column
        
        Returns:
            Total active hours (non-lying time)
        """
        if 'behavioral_state' not in behavioral_states.columns:
            return 0.0
        
        # Count active states (not lying)
        active_states = ['standing', 'walking', 'feeding', 'ruminating']
        active_mask = behavioral_states['behavioral_state'].isin(active_states)
        
        # Calculate total active time
        # Assuming each row represents 1 minute of data
        active_minutes = active_mask.sum()
        active_hours = active_minutes / 60.0
        
        return active_hours
