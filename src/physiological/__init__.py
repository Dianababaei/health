"""
Physiological Analysis Module

Temperature baseline calculation with circadian rhythm extraction,
dynamic updates, and drift detection for cattle health monitoring.

Sub-modules:
- baseline_calculator: Core baseline temperature calculation
- circadian_extractor: Circadian rhythm extraction and detrending
- baseline_updater: Dynamic baseline updates and history management
"""

from .baseline_calculator import BaselineCalculator
from .circadian_extractor import CircadianExtractor
from .baseline_updater import BaselineUpdater

__all__ = [
    'BaselineCalculator',
    'CircadianExtractor',
    'BaselineUpdater',
]
