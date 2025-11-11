"""
Health Intelligence Module

<<<<<<< HEAD
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
=======
Provides health scoring, alert detection, and intelligent health analysis
for the Artemis Health monitoring system.
"""

from .alert_system import AlertSystem, AlertState, AlertStatus, AlertPriority
>>>>>>> 27759dc60d8adda17a1542bda67e6dae6f27db9e

__all__ = [
    'AlertSystem',
    'AlertState',
    'AlertStatus',
<<<<<<< HEAD
    'MultiDayHealthTrendTracker',
    'TrendIndicator',
    'TimeWindowMetrics',
    'HealthTrendReport',
=======
    'AlertPriority',
>>>>>>> 27759dc60d8adda17a1542bda67e6dae6f27db9e
]
