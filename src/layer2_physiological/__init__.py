"""
Layer 2 Physiological Analysis Module

This module provides physiological analysis tools for cattle health monitoring,
focusing on body temperature patterns, circadian rhythm analysis, and multi-day trend detection.

Key Components:
- CircadianRhythmAnalyzer: Extract and analyze 24-hour temperature cycles
- MultiDayTrendAnalyzer: Track temperature and activity trends over 7/14/30/90 days
- Rhythm health scoring and loss detection
- Trend classification (improving, stable, deteriorating)
- Combined health trajectory assessment
- Dashboard visualization data generation
"""

from .circadian_rhythm import (
    CircadianRhythmAnalyzer,
    CircadianParameters,
    RhythmHealthMetrics,
)

from .trend_analysis import (
    MultiDayTrendAnalyzer,
    TrendDirection,
    HealthTrajectory,
    TrendPeriodConfig,
    TemperatureTrendMetrics,
    ActivityTrendMetrics,
    CombinedHealthTrend,
    TrendReport,
)

__all__ = [
    # Circadian rhythm analysis
    'CircadianRhythmAnalyzer',
    'CircadianParameters',
    'RhythmHealthMetrics',
    # Multi-day trend analysis
    'MultiDayTrendAnalyzer',
    'TrendDirection',
    'HealthTrajectory',
    'TrendPeriodConfig',
    'TemperatureTrendMetrics',
    'ActivityTrendMetrics',
    'CombinedHealthTrend',
    'TrendReport',
]
