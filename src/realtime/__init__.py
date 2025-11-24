"""
Real-time MQTT Service Module

This module provides real-time health monitoring capabilities via MQTT,
including detector pipeline orchestration and scheduled job execution.

Components:
- scheduler: APScheduler integration for detector execution
"""

from .scheduler import DetectorScheduler

__all__ = [
    'DetectorScheduler',
]
