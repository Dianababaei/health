"""
Health Intelligence Module

Provides alert logging, state management, health scoring, trend tracking, and
intelligent health analysis for the Artemis Health livestock monitoring system.

Components:
- Alert logging and state management
- Health scoring framework
- Alert detection system
- Multi-day health trend tracking (7/14/30/90 days)
- Dashboard data preparation and visualization
"""

__version__ = "1.0.0"

# Alert logging and state management
from .logging.alert_logger import AlertLogger
from .logging.alert_state_manager import AlertStateManager, AlertStatus

# Alert system
try:
    from .alert_system import AlertSystem, AlertState, AlertPriority
except ImportError:
    AlertSystem = None
    AlertState = None
    AlertPriority = None

# Trend tracking
from .trend_tracker import (
    MultiDayHealthTrendTracker,
    TrendIndicator,
    TimeWindowMetrics,
    HealthTrendReport
)

# Health scoring
from .health_scoring import (
    HealthScorer,
    HealthScore,
    HealthCategory
)

# Reproductive health detection
try:
    from .reproductive import (
        EstrusDetector,
        EstrusEvent,
        PregnancyDetector,
        PregnancyStatus
    )
except ImportError:
    EstrusDetector = None
    EstrusEvent = None
    PregnancyDetector = None
    PregnancyStatus = None

__all__ = [
    # Alert logging
    'AlertLogger',
    'AlertStateManager',
    'AlertStatus',
    # Alert system (if available)
    'AlertSystem',
    'AlertState',
    'AlertPriority',
    # Trend tracking
    'MultiDayHealthTrendTracker',
    'TrendIndicator',
    'TimeWindowMetrics',
    'HealthTrendReport',
    # Health scoring
    'HealthScorer',
    'HealthScore',
    'HealthCategory',
    # Reproductive health
    'EstrusDetector',
    'EstrusEvent',
    'PregnancyDetector',
    'PregnancyStatus',
]
