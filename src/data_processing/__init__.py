"""
Data processing module for cattle sensor data ingestion.

This module provides comprehensive data ingestion capabilities with support for
batch and incremental loading modes, extensive validation, and error handling.
"""

from .ingestion import DataIngestionModule, IngestionSummary
from .parsers import TimestampParser, CSVParser
from .validators import DataValidator, ValidationSummary, ValidationError

__all__ = [
    'DataIngestionModule',
    'IngestionSummary',
    'TimestampParser',
    'CSVParser',
    'DataValidator',
    'ValidationSummary',
    'ValidationError',
]

__version__ = '1.0.0'
