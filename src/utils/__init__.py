"""
Utility modules for Artemis Health Monitoring System

This package contains utility functions and classes used across the system.
"""

from .logger import (
    setup_logging,
    get_logger,
    log_alert,
    log_training_metrics,
    ensure_logging_initialized
)

__all__ = [
    'setup_logging',
    'get_logger',
    'log_alert',
    'log_training_metrics',
    'ensure_logging_initialized'
]
