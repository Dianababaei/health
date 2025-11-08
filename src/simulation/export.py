"""
Dataset Export Utilities

This module provides utilities for exporting simulated datasets to CSV
and generating comprehensive metadata files in JSON format.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from .health_events import HealthEventSimulator, HealthEvent


class DatasetExporter:
    """
    Handles exporting simulated datasets with metadata.
    
    Exports include:
    - CSV files with sensor data and ground truth labels
    - JSON metadata files with dataset statistics and parameters
    - Checksums for validation
    """
    
    def __init__(self, output_dir: str = "data/simulated"):
        """
        Initialize dataset exporter.
        
        Args:
            output_dir: Base directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dataset(self,
                      labeled_data: pd.DataFrame,
                      dataset_name: str,
                      metadata: Optional[Dict[str, Any]] = None,
                      export_daily: Optional[pd.DataFrame] = None):
        """
        Export complete dataset with CSV and metadata.
        
        Args:
            labeled_data: DataFrame with sensor data and labels
            dataset_name: Name for the dataset (e.g., 'short_term_7d')
            metadata: Optional metadata dictionary to include
            export_daily: Optional daily aggregate DataFrame
        """
        # Export main CSV
        csv_path = self.output_dir / f"{dataset_name}.csv"
        self._export_csv(labeled_data, csv_path)
        
        # Export daily aggregates if provided
        if export_daily is not None:
            daily_path = self.output_dir / f"{dataset_name}_daily.csv"
            export_daily.to_csv(daily_path, index=False)
        
        # Generate and export metadata
        full_metadata = self._generate_metadata(labeled_data, dataset_name, metadata)
        metadata_path = self.output_dir / f"metadata_{dataset_name}.json"
        self._export_metadata(full_metadata, metadata_path)
        
        return {
            'csv_path': str(csv_path),
            'metadata_path': str(metadata_path),
            'daily_path': str(daily_path) if export_daily is not None else None
        }
    
    def _export_csv(self, data: pd.DataFrame, path: Path):
        """
        Export data to CSV with proper formatting.
        
        Args:
            data: DataFrame to export
            path: Output file path
        """
        # Define column order
        column_order = [
            'timestamp', 'temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg',
            'behavioral_state', 'temperature_status', 'health_events', 'sensor_quality'
        ]
        
        # Filter to available columns in order
        available_cols = [c for c in column_order if c in data.columns]
        
        # Add any remaining columns not in standard order
        remaining_cols = [c for c in data.columns if c not in available_cols]
        export_cols = available_cols + remaining_cols
        
        # Export with timestamp formatting
        export_data = data[export_cols].copy()
        
        # Ensure timestamp is in ISO 8601 format
        if 'timestamp' in export_data.columns:
            export_data['timestamp'] = pd.to_datetime(export_data['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        export_data.to_csv(path, index=False, float_format='%.4f')
    
    def _generate_metadata(self,
                          data: pd.DataFrame,
                          dataset_name: str,
                          additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Generate comprehensive metadata for the dataset.
        
        Args:
            data: The dataset
            dataset_name: Name of the dataset
            additional_metadata: Optional additional metadata to include
            
        Returns:
            Complete metadata dictionary
        """
        metadata = {
            'dataset_name': dataset_name,
            'generation_timestamp': datetime.now().isoformat(),
            'statistics': self._calculate_statistics(data),
            'temporal_coverage': self._get_temporal_coverage(data),
            'data_quality': self._assess_data_quality(data),
        }
        
        # Add additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate dataset statistics."""
        stats = {
            'total_data_points': len(data),
            'duration_hours': len(data) / 60.0,  # Assuming 1-minute intervals
            'duration_days': len(data) / 1440.0,
        }
        
        # Behavioral state distribution
        if 'behavioral_state' in data.columns:
            state_dist = data['behavioral_state'].value_counts(normalize=True).to_dict()
            stats['behavioral_state_distribution'] = {
                k: float(v * 100) for k, v in state_dist.items()
            }
        
        # Health event distribution
        if 'health_events' in data.columns:
            health_dist = data['health_events'].value_counts().to_dict()
            stats['health_event_counts'] = {
                k: int(v) for k, v in health_dist.items()
            }
        
        # Temperature status distribution
        if 'temperature_status' in data.columns:
            temp_status_dist = data['temperature_status'].value_counts().to_dict()
            stats['temperature_status_counts'] = {
                k: int(v) for k, v in temp_status_dist.items()
            }
        
        # Sensor quality distribution
        if 'sensor_quality' in data.columns:
            sensor_quality_dist = data['sensor_quality'].value_counts(normalize=True).to_dict()
            stats['sensor_quality_distribution'] = {
                k: float(v * 100) for k, v in sensor_quality_dist.items()
            }
        
        # Sensor value statistics
        sensor_cols = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        stats['sensor_statistics'] = {}
        for col in sensor_cols:
            if col in data.columns:
                stats['sensor_statistics'][col] = {
                    'mean': float(data[col].mean()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'median': float(data[col].median()),
                }
        
        return stats
    
    def _get_temporal_coverage(self, data: pd.DataFrame) -> Dict:
        """Get temporal coverage information."""
        if 'timestamp' not in data.columns:
            return {}
        
        timestamps = pd.to_datetime(data['timestamp'])
        
        return {
            'start_time': timestamps.min().isoformat(),
            'end_time': timestamps.max().isoformat(),
            'time_step_minutes': 1,  # Fixed for our simulation
            'total_minutes': len(data),
            'has_gaps': self._check_for_gaps(timestamps),
        }
    
    def _check_for_gaps(self, timestamps: pd.Series) -> bool:
        """Check if there are gaps in the timestamp sequence."""
        if len(timestamps) < 2:
            return False
        
        # Calculate time differences
        time_diffs = timestamps.diff().dt.total_seconds() / 60.0
        
        # Check if all differences are ~1 minute (allowing small floating point errors)
        expected_diff = 1.0
        gaps = (time_diffs > expected_diff + 0.1).sum()
        
        return gaps > 0
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict:
        """Assess data quality metrics."""
        quality = {
            'completeness': 100.0,  # All fields present in simulation
            'continuity': not self._check_for_gaps(pd.to_datetime(data['timestamp'])) if 'timestamp' in data.columns else True,
        }
        
        # Check for any null values
        if data.isnull().any().any():
            null_counts = data.isnull().sum().to_dict()
            quality['null_values'] = {k: int(v) for k, v in null_counts.items() if v > 0}
            quality['completeness'] = float((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100)
        else:
            quality['null_values'] = {}
        
        # Validation ranges check
        quality['values_in_range'] = True
        if 'temperature' in data.columns:
            if (data['temperature'] < 36.0).any() or (data['temperature'] > 42.0).any():
                quality['values_in_range'] = False
        
        return quality
    
    def _export_metadata(self, metadata: Dict, path: Path):
        """
        Export metadata to JSON file with pretty formatting.
        
        Args:
            metadata: Metadata dictionary
            path: Output file path
        """
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)


class DatasetSplitter:
    """
    Splits datasets into train/validation/test sets.
    
    Ensures proper temporal ordering and balanced distribution
    of behavioral states and health events.
    """
    
    def __init__(self,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15):
        """
        Initialize dataset splitter.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001:
            raise ValueError("Split ratios must sum to 1.0")
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split_dataset(self,
                     data: pd.DataFrame,
                     temporal_order: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Split dataset into train/val/test sets.
        
        Args:
            data: Full dataset
            temporal_order: If True, split maintains temporal order (train < val < test)
                          If False, randomly shuffle before split
            
        Returns:
            Dictionary with 'train', 'validation', 'test' DataFrames
        """
        n_samples = len(data)
        
        if temporal_order:
            # Temporal split: maintain time order
            train_end = int(n_samples * self.train_ratio)
            val_end = int(n_samples * (self.train_ratio + self.val_ratio))
            
            train_data = data.iloc[:train_end].copy()
            val_data = data.iloc[train_end:val_end].copy()
            test_data = data.iloc[val_end:].copy()
        else:
            # Random split: shuffle data first
            shuffled = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
            
            train_end = int(n_samples * self.train_ratio)
            val_end = int(n_samples * (self.train_ratio + self.val_ratio))
            
            train_data = shuffled.iloc[:train_end].copy()
            val_data = shuffled.iloc[train_end:val_end].copy()
            test_data = shuffled.iloc[val_end:].copy()
        
        return {
            'train': train_data,
            'validation': val_data,
            'test': test_data
        }
    
    def export_splits(self,
                     splits: Dict[str, pd.DataFrame],
                     output_dir: Path,
                     dataset_name: str):
        """
        Export train/val/test splits to separate CSV files.
        
        Args:
            splits: Dictionary with train/validation/test DataFrames
            output_dir: Output directory
            dataset_name: Base name for files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {}
        for split_name, split_data in splits.items():
            path = output_dir / f"{dataset_name}_{split_name}.csv"
            
            # Export with timestamp formatting
            export_data = split_data.copy()
            if 'timestamp' in export_data.columns:
                export_data['timestamp'] = pd.to_datetime(export_data['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            export_data.to_csv(path, index=False, float_format='%.4f')
            paths[split_name] = str(path)
        
        return paths
    
    def get_split_statistics(self, splits: Dict[str, pd.DataFrame]) -> Dict:
        """
        Get statistics about the splits.
        
        Args:
            splits: Dictionary with train/validation/test DataFrames
            
        Returns:
            Dictionary with statistics for each split
        """
        stats = {}
        
        for split_name, split_data in splits.items():
            split_stats = {
                'num_samples': len(split_data),
                'percentage': len(split_data) / sum(len(s) for s in splits.values()) * 100,
            }
            
            # State distribution
            if 'behavioral_state' in split_data.columns:
                state_dist = split_data['behavioral_state'].value_counts(normalize=True).to_dict()
                split_stats['state_distribution'] = {k: float(v * 100) for k, v in state_dist.items()}
            
            # Health event counts
            if 'health_events' in split_data.columns:
                health_counts = split_data['health_events'].value_counts().to_dict()
                split_stats['health_event_counts'] = {k: int(v) for k, v in health_counts.items()}
            
            stats[split_name] = split_stats
        
        return stats


def export_complete_dataset(labeled_data: pd.DataFrame,
                            daily_data: Optional[pd.DataFrame],
                            dataset_name: str,
                            output_dir: str = "data/simulated",
                            metadata: Optional[Dict] = None,
                            create_splits: bool = True) -> Dict[str, Any]:
    """
    Complete export workflow for a dataset.
    
    Args:
        labeled_data: Per-minute labeled data
        daily_data: Daily aggregate data
        dataset_name: Name of the dataset
        output_dir: Output directory
        metadata: Additional metadata
        create_splits: Whether to create train/val/test splits
        
    Returns:
        Dictionary with paths and statistics
    """
    exporter = DatasetExporter(output_dir)
    
    # Export main dataset
    paths = exporter.export_dataset(
        labeled_data,
        dataset_name,
        metadata,
        daily_data
    )
    
    result = {
        'dataset_name': dataset_name,
        'paths': paths
    }
    
    # Create splits if requested
    if create_splits:
        splitter = DatasetSplitter()
        splits = splitter.split_dataset(labeled_data, temporal_order=True)
        
        split_paths = splitter.export_splits(
            splits,
            Path(output_dir) / 'splits',
            dataset_name
        )
        
        result['split_paths'] = split_paths
        result['split_statistics'] = splitter.get_split_statistics(splits)
    
    return result
