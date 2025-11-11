"""
Health Intelligence Alert System

Immediate detection of critical health conditions:
- Fever alerts
- Heat stress alerts
- Prolonged inactivity alerts
- Sensor malfunction alerts

Features:
- Real-time detection (1-2 minute latency)
- Configurable thresholds
- Alert deduplication
- Confidence scoring
- Integration with behavioral and physiological analysis
"""

from .immediate_detector import ImmediateAlertDetector, Alert

__all__ = [
    'ImmediateAlertDetector',
    'Alert',
]

__version__ = '1.0.0'
