"""
Health Scoring Module

Modular health scoring system that calculates 0-100 health scores from
temperature, activity, behavioral, and alert data.

Main Components:
- HealthScorer: Main scoring orchestrator
- HealthScore: Score result data structure
- Component modules: Individual scoring components

Example Usage:
    >>> from src.health_intelligence.scoring import HealthScorer
    >>> scorer = HealthScorer()
    >>> score = scorer.calculate_score(
    ...     cow_id="COW_001",
    ...     temperature_data=temp_df,
    ...     activity_data=activity_df,
    ...     behavioral_data=behavior_df,
    ...     active_alerts=alerts
    ... )
    >>> print(f"Health Score: {score.total_score}/100")
"""

from .health_scorer import HealthScorer, HealthScore
from .components import (
    BaseScoreComponent,
    ComponentScore,
    TemperatureScoreComponent,
    ActivityScoreComponent,
    BehavioralScoreComponent,
    AlertScoreComponent,
)

__all__ = [
    'HealthScorer',
    'HealthScore',
    'BaseScoreComponent',
    'ComponentScore',
    'TemperatureScoreComponent',
    'ActivityScoreComponent',
    'BehavioralScoreComponent',
    'AlertScoreComponent',
]
