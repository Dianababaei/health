"""
Data processing module for cattle sensor data ingestion.

This module provides comprehensive data ingestion capabilities with support for
batch and incremental loading modes, extensive validation, error handling,
windowing for time-series analysis, and malfunction detection.
"""

from .ingestion import DataIngestionModule, IngestionSummary
from .parsers import TimestampParser, CSVParser
from .validators import DataValidator, ValidationSummary, ValidationError
from .windowing import WindowGenerator, WindowStatistics, create_window_summary
from .malfunction_detection import (
    MalfunctionDetector,
    MalfunctionAlert,
    MalfunctionType,
    AlertSeverity,
    ConnectivityLossDetector,
    StuckValueDetector,
    OutOfRangeDetector,
)

__all__ = [
    'DataIngestionModule',
    'IngestionSummary',
    'TimestampParser',
    'CSVParser',
    'DataValidator',
    'ValidationSummary',
    'ValidationError',
    'WindowGenerator',
    'WindowStatistics',
    'create_window_summary',
    'MalfunctionDetector',
    'MalfunctionAlert',
    'MalfunctionType',
    'AlertSeverity',
    'ConnectivityLossDetector',
    'StuckValueDetector',
    'OutOfRangeDetector',
]

__version__ = '1.2.0'
