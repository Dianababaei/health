"""
Health Intelligence Module

Provides alert logging, state management, trend tracking, and health intelligence
components for the Artemis Health livestock monitoring system.

Components:
- Alert logging and state management
- Multi-day health trend tracking (7/14/30/90 days)
- Dashboard data preparation and visualization
"""

__version__ = "1.0.0"

from .logging.alert_logger import AlertLogger
from .logging.alert_state_manager import AlertStateManager, AlertStatus
from .trend_tracker import (
    MultiDayHealthTrendTracker,
    TrendIndicator,
    TimeWindowMetrics,
    HealthTrendReport
)

__all__ = [
    'AlertLogger',
    'AlertStateManager',
    'AlertStatus',
    'MultiDayHealthTrendTracker',
    'TrendIndicator',
    'TimeWindowMetrics',
    'HealthTrendReport',
]
