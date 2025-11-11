"""
Health Scoring System

Modular health scoring system that calculates a 0-100 health score from
independent component scores (temperature, activity, behavioral, alerts).

Score Ranges:
  - 80-100: Excellent health (green)
  - 60-79: Good health with minor concerns (yellow)
  - 40-59: Moderate health issues requiring attention (orange)
  - 0-39: Poor health, critical intervention needed (red)

Architecture:
  - Component-based design allows easy replacement of scoring logic
  - Configurable weights via YAML
  - Each component contributes 0-25 points
  - Final score is weighted sum of components
"""

import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict

from .components import (
    BaseScoreComponent,
    ComponentScore,
    TemperatureScoreComponent,
    ActivityScoreComponent,
    BehavioralScoreComponent,
    AlertScoreComponent,
)

logger = logging.getLogger(__name__)


@dataclass
class HealthScore:
    """
    Complete health score result.
    
    Attributes:
        timestamp: When score was calculated
        cow_id: Animal identifier
        total_score: Overall health score (0-100)
        component_scores: Dict of component name -> ComponentScore
        weights: Component weights used
        health_category: Health category (excellent, good, moderate, poor)
        confidence: Overall confidence in score (0-1)
        metadata: Additional calculation details
    """
    timestamp: datetime
    cow_id: str
    total_score: float
    component_scores: Dict[str, ComponentScore]
    weights: Dict[str, float]
    health_category: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization or database storage."""
        result = {
            'timestamp': self.timestamp.isoformat(),
            'cow_id': self.cow_id,
            'total_score': round(self.total_score, 2),
            'health_category': self.health_category,
            'confidence': round(self.confidence, 3),
            'weights': self.weights,
            'metadata': self.metadata,
        }
        
        # Add individual component scores (normalized 0-1)
        result['temperature_component'] = self.component_scores['temperature_stability'].normalized_score
        result['activity_component'] = self.component_scores['activity_level'].normalized_score
        result['behavioral_component'] = self.component_scores['behavioral_patterns'].normalized_score
        result['alert_component'] = self.component_scores['alert_frequency'].normalized_score
        
        # Add detailed breakdown
        result['component_details'] = {
            name: {
                'score': round(comp_score.score, 2),
                'normalized': round(comp_score.normalized_score, 3),
                'confidence': round(comp_score.confidence, 3),
                'details': comp_score.details,
                'warnings': comp_score.warnings
            }
            for name, comp_score in self.component_scores.items()
        }
        
        return result
    
    def to_database_record(self) -> Dict[str, Any]:
        """Convert to database record format matching schema.sql."""
        return {
            'timestamp': self.timestamp,
            'cow_id': self.cow_id,
            'total_score': round(self.total_score, 2),
            'temperature_score': self.component_scores['temperature_stability'].normalized_score,
            'activity_score': self.component_scores['activity_level'].normalized_score,
            'behavioral_score': self.component_scores['behavioral_patterns'].normalized_score,
            'alert_score': self.component_scores['alert_frequency'].normalized_score,
            'metadata': self.to_dict()  # Store full details in JSONB
        }


class HealthScorer:
    """
    Modular health scoring system.
    
    Calculates health scores from independent component scorers with
    configurable weights. Designed for easy replacement of components
    or formulas without refactoring.
    
    Example:
        >>> scorer = HealthScorer()
        >>> score = scorer.calculate_score(
        ...     cow_id="COW_001",
        ...     temperature_data=temp_df,
        ...     activity_data=activity_df,
        ...     behavioral_data=behavior_df,
        ...     active_alerts=alerts
        ... )
        >>> print(f"Health Score: {score.total_score}/100 ({score.health_category})")
    """
    
    # Health category thresholds
    EXCELLENT_THRESHOLD = 80
    GOOD_THRESHOLD = 60
    MODERATE_THRESHOLD = 40
    
    def __init__(
        self,
        config_path: str = "config/health_score_weights.yaml",
        custom_components: Optional[Dict[str, BaseScoreComponent]] = None
    ):
        """
        Initialize health scorer.
        
        Args:
            config_path: Path to YAML configuration file
            custom_components: Optional dict of component_name -> custom component instance
                              for replacing default components
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Load component weights
        self.weights = self.config.get('component_weights', {})
        self._validate_weights()
        
        # Initialize components
        self.components = self._initialize_components(custom_components)
        
        # Calculation settings
        calc_config = self.config.get('calculation', {})
        self.update_interval = calc_config.get('update_interval_minutes', 60)
        self.rolling_window_hours = calc_config.get('rolling_window_hours', 24)
        self.min_data_completeness = calc_config.get('min_data_completeness', 0.70)
        self.smoothing_enabled = calc_config.get('smoothing_enabled', True)
        self.smoothing_factor = calc_config.get('smoothing_factor', 0.3)
        
        logger.info(f"HealthScorer initialized with config: {config_path}")
        logger.info(f"Component weights: {self.weights}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded health scoring config from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            'component_weights': {
                'temperature_stability': 0.30,
                'activity_level': 0.25,
                'behavioral_patterns': 0.25,
                'alert_frequency': 0.20,
            },
            'calculation': {
                'update_interval_minutes': 60,
                'rolling_window_hours': 24,
                'min_data_completeness': 0.70,
            }
        }
    
    def _validate_weights(self):
        """Validate that component weights sum to 1.0."""
        validation_config = self.config.get('validation', {})
        enforce_sum = validation_config.get('enforce_weight_sum', True)
        tolerance = validation_config.get('weight_sum_tolerance', 0.01)
        
        if not enforce_sum:
            return
        
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > tolerance:
            logger.warning(
                f"Component weights sum to {weight_sum:.3f}, expected 1.0. "
                f"Normalizing weights."
            )
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= weight_sum
    
    def _initialize_components(
        self,
        custom_components: Optional[Dict[str, BaseScoreComponent]] = None
    ) -> Dict[str, BaseScoreComponent]:
        """
        Initialize scoring components.
        
        Args:
            custom_components: Optional custom component implementations
        
        Returns:
            Dict of component_name -> component instance
        """
        components = {}
        
        # Default components
        default_components = {
            'temperature_stability': TemperatureScoreComponent(
                self.config.get('temperature_stability', {})
            ),
            'activity_level': ActivityScoreComponent(
                self.config.get('activity_level', {})
            ),
            'behavioral_patterns': BehavioralScoreComponent(
                self.config.get('behavioral_patterns', {})
            ),
            'alert_frequency': AlertScoreComponent(
                self.config.get('alert_frequency', {})
            ),
        }
        
        # Use custom components if provided, otherwise use defaults
        for name, default_component in default_components.items():
            if custom_components and name in custom_components:
                components[name] = custom_components[name]
                logger.info(f"Using custom component for '{name}'")
            else:
                components[name] = default_component
        
        return components
    
    def calculate_score(
        self,
        cow_id: str,
        temperature_data: Optional[pd.DataFrame] = None,
        activity_data: Optional[pd.DataFrame] = None,
        behavioral_data: Optional[pd.DataFrame] = None,
        active_alerts: Optional[List[Dict]] = None,
        resolved_alerts: Optional[List[Dict]] = None,
        baseline_temp: Optional[float] = None,
        baseline_activity: Optional[float] = None,
        fever_events: Optional[List[Dict]] = None,
        inactivity_events: Optional[List[Dict]] = None,
        stress_events: Optional[List[Dict]] = None,
        circadian_score: Optional[float] = None,
        previous_score: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ) -> HealthScore:
        """
        Calculate comprehensive health score.
        
        Args:
            cow_id: Animal identifier
            temperature_data: DataFrame with ['timestamp', 'temperature']
            activity_data: DataFrame with ['timestamp', 'movement_intensity']
            behavioral_data: DataFrame with ['timestamp', 'behavioral_state']
            active_alerts: List of active alert dicts
            resolved_alerts: List of recently resolved alert dicts
            baseline_temp: Individual baseline temperature (°C)
            baseline_activity: Individual baseline activity level
            fever_events: List of fever event dicts
            inactivity_events: List of prolonged inactivity event dicts
            stress_events: List of stress behavior event dicts
            circadian_score: Circadian rhythm health score (0-1)
            previous_score: Previous health score for smoothing
            timestamp: Score calculation timestamp (defaults to now)
            **kwargs: Additional parameters
        
        Returns:
            HealthScore object with total score and component breakdown
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        component_scores = {}
        warnings = []
        
        # Calculate temperature component score
        if temperature_data is not None and not temperature_data.empty:
            temp_score = self.components['temperature_stability'].calculate_score(
                cow_id=cow_id,
                data=temperature_data,
                baseline_temp=baseline_temp,
                fever_events=fever_events,
                circadian_score=circadian_score
            )
            component_scores['temperature_stability'] = temp_score
            warnings.extend(temp_score.warnings)
        else:
            warnings.append("No temperature data provided")
            component_scores['temperature_stability'] = ComponentScore(
                score=0.0, normalized_score=0.0, confidence=0.0,
                details={'error': 'No data'}, warnings=[]
            )
        
        # Calculate activity component score
        if activity_data is not None and not activity_data.empty:
            activity_score = self.components['activity_level'].calculate_score(
                cow_id=cow_id,
                data=activity_data,
                baseline_activity=baseline_activity,
                inactivity_events=inactivity_events,
                behavioral_states=behavioral_data
            )
            component_scores['activity_level'] = activity_score
            warnings.extend(activity_score.warnings)
        else:
            warnings.append("No activity data provided")
            component_scores['activity_level'] = ComponentScore(
                score=0.0, normalized_score=0.0, confidence=0.0,
                details={'error': 'No data'}, warnings=[]
            )
        
        # Calculate behavioral component score
        if behavioral_data is not None and not behavioral_data.empty:
            behavioral_score = self.components['behavioral_patterns'].calculate_score(
                cow_id=cow_id,
                data=behavioral_data,
                stress_events=stress_events
            )
            component_scores['behavioral_patterns'] = behavioral_score
            warnings.extend(behavioral_score.warnings)
        else:
            warnings.append("No behavioral data provided")
            component_scores['behavioral_patterns'] = ComponentScore(
                score=0.0, normalized_score=0.0, confidence=0.0,
                details={'error': 'No data'}, warnings=[]
            )
        
        # Calculate alert component score
        alert_score = self.components['alert_frequency'].calculate_score(
            cow_id=cow_id,
            data=pd.DataFrame(),  # Not used
            active_alerts=active_alerts,
            resolved_alerts=resolved_alerts
        )
        component_scores['alert_frequency'] = alert_score
        warnings.extend(alert_score.warnings)
        
        # Calculate weighted total score
        total_score = 0.0
        total_weight = 0.0
        overall_confidence = 0.0
        
        for component_name, component_score in component_scores.items():
            weight = self.weights.get(component_name, 0.0)
            # Score is already 0-25, weight scales it
            weighted_score = component_score.score * weight / 0.25  # Normalize weight
            total_score += weighted_score
            total_weight += weight * component_score.confidence
            overall_confidence += weight * component_score.confidence
        
        # Normalize confidence
        if sum(self.weights.values()) > 0:
            overall_confidence /= sum(self.weights.values())
        
        # Apply smoothing if enabled and previous score available
        if self.smoothing_enabled and previous_score is not None:
            smoothed_score = (
                self.smoothing_factor * total_score +
                (1 - self.smoothing_factor) * previous_score
            )
            metadata = {
                'raw_score': round(total_score, 2),
                'smoothed_score': round(smoothed_score, 2),
                'previous_score': round(previous_score, 2),
                'smoothing_applied': True
            }
            total_score = smoothed_score
        else:
            metadata = {'smoothing_applied': False}
        
        # Clamp to valid range
        total_score = max(0.0, min(100.0, total_score))
        
        # Determine health category
        health_category = self._classify_health_category(total_score)
        
        # Add metadata
        metadata.update({
            'calculation_timestamp': timestamp.isoformat(),
            'warnings': warnings,
            'data_sources': {
                'temperature': temperature_data is not None,
                'activity': activity_data is not None,
                'behavioral': behavioral_data is not None,
                'alerts': active_alerts is not None,
            }
        })
        
        return HealthScore(
            timestamp=timestamp,
            cow_id=cow_id,
            total_score=total_score,
            component_scores=component_scores,
            weights=self.weights,
            health_category=health_category,
            confidence=overall_confidence,
            metadata=metadata
        )
    
    def _classify_health_category(self, score: float) -> str:
        """
        Classify health score into category.
        
        Args:
            score: Health score (0-100)
        
        Returns:
            Category: 'excellent', 'good', 'moderate', or 'poor'
        """
        if score >= self.EXCELLENT_THRESHOLD:
            return 'excellent'
        elif score >= self.GOOD_THRESHOLD:
            return 'good'
        elif score >= self.MODERATE_THRESHOLD:
            return 'moderate'
        else:
            return 'poor'
    
    def get_score_breakdown(self, health_score: HealthScore) -> Dict[str, Any]:
        """
        Get detailed breakdown of score calculation.
        
        Args:
            health_score: HealthScore object
        
        Returns:
            Dict with detailed breakdown showing contribution of each component
        """
        breakdown = {
            'total_score': round(health_score.total_score, 2),
            'health_category': health_score.health_category,
            'confidence': round(health_score.confidence, 3),
            'components': {}
        }
        
        for component_name, component_score in health_score.component_scores.items():
            weight = health_score.weights.get(component_name, 0.0)
            contribution = component_score.score * weight / 0.25  # Scale by weight
            
            breakdown['components'][component_name] = {
                'raw_score': round(component_score.score, 2),
                'max_score': 25,
                'weight': weight,
                'contribution_to_total': round(contribution, 2),
                'confidence': round(component_score.confidence, 3),
                'details': component_score.details
            }
        
        return breakdown
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update component weights dynamically.
        
        Args:
            new_weights: Dict of component_name -> weight
        """
        self.weights.update(new_weights)
        self._validate_weights()
        logger.info(f"Updated component weights: {self.weights}")
    
    def replace_component(self, component_name: str, new_component: BaseScoreComponent):
        """
        Replace a scoring component with custom implementation.
        
        Args:
            component_name: Name of component to replace
            new_component: New component instance
        """
        if component_name not in self.components:
            raise ValueError(f"Unknown component: {component_name}")
        
        self.components[component_name] = new_component
        logger.info(f"Replaced component '{component_name}' with {new_component.__class__.__name__}")
