"""
Layer 2: Physiological Analysis

Modules for monitoring internal body changes using temperature
and motion data to assess health conditions.

Sub-analyses:
- Temperature pattern analysis (average, sudden changes)
- Circadian rhythm tracking
- Temperature-activity correlation
- Long-term health trend tracking
"""

from .baseline import BaselineCalculator
from .temperature_anomaly import (
    TemperatureAnomalyDetector,
    AnomalyType,
    TemperatureAnomaly
)
from .temp_activity_correlation import (
    TemperatureActivityCorrelator,
    HealthPattern,
    CorrelationEvent
)

__all__ = [
    'BaselineCalculator',
    'TemperatureAnomalyDetector',
    'AnomalyType',
    'TemperatureAnomaly',
    'TemperatureActivityCorrelator',
    'HealthPattern',
    'CorrelationEvent'
]

__version__ = '0.1.0'
