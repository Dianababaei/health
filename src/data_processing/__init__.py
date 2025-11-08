"""
Data processing module for cattle sensor data ingestion.

This module provides comprehensive data ingestion capabilities with support for
batch and incremental loading modes, extensive validation, error handling,
and windowing for time-series analysis.
"""

from .ingestion import DataIngestionModule, IngestionSummary
from .parsers import TimestampParser, CSVParser
from .validators import DataValidator, ValidationSummary, ValidationError
from .windowing import WindowGenerator, WindowStatistics, create_window_summary

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
]

__version__ = '1.1.0'
