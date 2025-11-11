"""
Health Intelligence Module

Provides alert logging, state management, and health intelligence components
for the Artemis Health livestock monitoring system.
"""

__version__ = "1.0.0"

from .logging.alert_logger import AlertLogger
from .logging.alert_state_manager import AlertStateManager, AlertStatus

__all__ = [
    'AlertLogger',
    'AlertStateManager',
    'AlertStatus',
]
