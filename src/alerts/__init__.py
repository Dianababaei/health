"""
Alert management module for malfunction detection alerts.

Provides alert generation, queuing, logging, and notification management.
"""

from .alert_generator import AlertGenerator

__all__ = [
    'AlertGenerator',
]

__version__ = '1.0.0'
