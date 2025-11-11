"""
Temperature Stability Score Component

Calculates health score based on temperature stability metrics:
- Deviation from individual baseline
- Circadian rhythm regularity
- Fever incidents

Score Range: 0-25 points
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .base_component import BaseScoreComponent, ComponentScore


class TemperatureScoreComponent(BaseScoreComponent):
    """
    Temperature stability scoring component.
    
    Placeholder Formula:
        score = 25 - (temp_deviation * 10) - (fever_count * 5)
    
    Where:
        - temp_deviation: Average deviation from baseline (°C)
        - fever_count: Number of fever incidents in period
    
    Future Integration Points:
        - Custom baseline calculation methods
        - Machine learning-based anomaly detection
        - Breed-specific temperature profiles
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize temperature component.
        
        Args:
            config: Configuration dict with keys:
                - optimal_deviation: Deviation threshold for full points (default 0.5°C)
                - max_deviation: Deviation threshold for zero points (default 2.0°C)
                - fever_penalty_per_incident: Points deducted per fever (default 5.0)
                - circadian_bonus_enabled: Whether to add circadian bonus (default True)
                - circadian_bonus_max: Maximum circadian bonus points (default 3.0)
        """
        super().__init__(config)
        
        # Load configuration with defaults
        self.optimal_deviation = self.config.get('optimal_deviation', 0.5)
        self.max_deviation = self.config.get('max_deviation', 2.0)
        self.fever_penalty = self.config.get('fever_penalty_per_incident', 5.0)
        self.max_fever_penalty = self.config.get('max_fever_penalty', 15.0)
        self.circadian_bonus_enabled = self.config.get('circadian_bonus_enabled', True)
        self.circadian_bonus_max = self.config.get('circadian_bonus_max', 3.0)
    
    def get_required_columns(self) -> list:
        """Get required DataFrame columns."""
        return ['timestamp', 'temperature']
    
    def calculate_score(
        self,
        cow_id: str,
        data: pd.DataFrame,
        baseline_temp: Optional[float] = None,
        fever_events: Optional[list] = None,
        circadian_score: Optional[float] = None,
        **kwargs
    ) -> ComponentScore:
        """
        Calculate temperature stability score.
        
        Args:
            cow_id: Animal identifier
            data: DataFrame with columns ['timestamp', 'temperature']
            baseline_temp: Individual baseline temperature (°C), defaults to mean if not provided
            fever_events: List of fever event dicts from alert system
            circadian_score: Circadian rhythm health score (0-1), if available
            **kwargs: Additional parameters
        
        Returns:
            ComponentScore with temperature stability score (0-25 points)
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
        if baseline_temp is None:
            baseline_temp = data['temperature'].mean()
            details['baseline_source'] = 'calculated_mean'
            confidence *= 0.9  # Slightly lower confidence without provided baseline
        else:
            details['baseline_source'] = 'provided'
        
        details['baseline_temperature'] = round(baseline_temp, 2)
        
        # Calculate temperature deviation from baseline
        temp_deviations = (data['temperature'] - baseline_temp).abs()
        mean_deviation = temp_deviations.mean()
        max_deviation_observed = temp_deviations.max()
        
        details['mean_deviation'] = round(mean_deviation, 3)
        details['max_deviation_observed'] = round(max_deviation_observed, 3)
        details['deviation_threshold_optimal'] = self.optimal_deviation
        details['deviation_threshold_max'] = self.max_deviation
        
        # Apply deviation penalty using placeholder formula
        # Linear interpolation between optimal and max deviation
        if mean_deviation <= self.optimal_deviation:
            deviation_penalty = 0.0
        elif mean_deviation >= self.max_deviation:
            deviation_penalty = 10.0  # Maximum penalty from deviation
        else:
            # Linear scale between optimal and max
            deviation_ratio = (mean_deviation - self.optimal_deviation) / (
                self.max_deviation - self.optimal_deviation
            )
            deviation_penalty = deviation_ratio * 10.0
        
        score -= deviation_penalty
        details['deviation_penalty'] = round(deviation_penalty, 2)
        
        # Count fever incidents
        fever_count = 0
        if fever_events:
            fever_count = len(fever_events)
            details['fever_events'] = fever_events
        
        details['fever_count'] = fever_count
        
        # Apply fever penalty (placeholder formula: fever_count * 5)
        fever_penalty = min(fever_count * self.fever_penalty, self.max_fever_penalty)
        score -= fever_penalty
        details['fever_penalty'] = round(fever_penalty, 2)
        
        # Add circadian rhythm bonus if enabled and available
        circadian_bonus = 0.0
        if self.circadian_bonus_enabled and circadian_score is not None:
            # Bonus proportional to circadian health (0-1 scale)
            circadian_bonus = circadian_score * self.circadian_bonus_max
            score += circadian_bonus
            details['circadian_bonus'] = round(circadian_bonus, 2)
            details['circadian_score'] = round(circadian_score, 2)
        
        # Adjust confidence based on data quality
        data_completeness = 1.0 - (data['temperature'].isna().sum() / len(data))
        confidence *= data_completeness
        details['data_completeness'] = round(data_completeness, 3)
        
        # Clamp score to valid range [0, 25]
        score = max(0.0, min(25.0, score))
        normalized_score = self.normalize_score(score)
        
        details['raw_score'] = round(score, 2)
        details['formula'] = 'placeholder: 25 - (deviation * 10) - (fever_count * 5) + circadian_bonus'
        
        return ComponentScore(
            score=score,
            normalized_score=normalized_score,
            confidence=confidence,
            details=details,
            warnings=warnings
        )
