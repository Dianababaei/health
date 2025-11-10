"""
Feature Engineering Module for Cattle Behavior Classification

This module provides feature extraction and dataset preparation tools for
detecting cattle behaviors (ruminating and feeding) from sensor data.
"""

from .behavior_features import BehaviorFeatureExtractor, extract_features_from_dataframe
from .dataset_builder import DatasetBuilder, load_prepared_dataset

__all__ = [
    'BehaviorFeatureExtractor',
    'extract_features_from_dataframe',
    'DatasetBuilder',
    'load_prepared_dataset'
]
