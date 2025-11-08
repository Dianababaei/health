"""
Activity Metrics Module

Provides comprehensive behavioral activity tracking and analysis for livestock monitoring.
Calculates duration metrics, movement intensity, activity ratios, and state transitions.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ActivityTracker:
    """
    Tracks and analyzes behavioral activity metrics from classified state data.
    
    Features:
    - Duration tracking per behavioral state
    - Hourly and daily aggregations
    - Movement intensity calculation from accelerometer data
    - Activity/rest ratio computation
    - State transition counting
    - Behavioral state log generation
    - CSV and JSON export capabilities
    """
    
    # Define behavioral states
    BEHAVIORAL_STATES = ['lying', 'standing', 'walking', 'feeding', 'ruminating']
    
    # Rest vs active classification
    REST_STATES = ['lying']
    ACTIVE_STATES = ['standing', 'walking', 'feeding', 'ruminating']
    
    def __init__(self, output_dir: str = "data/outputs/behavioral_logs"):
        """
        Initialize activity tracker.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ActivityTracker initialized: output_dir={self.output_dir}")
    
    def calculate_durations(
        self, 
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        state_col: str = 'behavioral_state'
    ) -> pd.DataFrame:
        """
        Calculate duration for consecutive same-state periods.
        
        Args:
            data: DataFrame with timestamp and state columns
            timestamp_col: Name of timestamp column
            state_col: Name of behavioral state column
            
        Returns:
            DataFrame with added duration_minutes column
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Calculate time difference in minutes (sampling interval)
        df['time_diff'] = df[timestamp_col].diff().dt.total_seconds() / 60.0
        
        # For the first row, use median interval or default to 1 minute
        median_interval = df['time_diff'].median()
        if pd.isna(median_interval) or median_interval == 0:
            median_interval = 1.0
        df.loc[0, 'time_diff'] = median_interval
        
        # Detect state changes
        df['state_changed'] = (df[state_col] != df[state_col].shift(1)).fillna(True)
        
        # Create group ID for consecutive same states
        df['state_group'] = df['state_changed'].cumsum()
        
        # Calculate duration for each state group
        group_durations = df.groupby('state_group')['time_diff'].sum()
        df['duration_minutes'] = df['state_group'].map(group_durations)
        
        # Clean up temporary columns
        df = df.drop(columns=['time_diff', 'state_changed', 'state_group'])
        
        return df
    
    def calculate_movement_intensity(
        self,
        data: pd.DataFrame,
        fxa_col: str = 'fxa',
        mya_col: str = 'mya',
        rza_col: str = 'rza'
    ) -> pd.DataFrame:
        """
        Calculate movement intensity from accelerometer data.
        
        Computes magnitude: sqrt(Fxa² + Mya² + Rza²)
        
        Args:
            data: DataFrame with accelerometer columns
            fxa_col: Name of X-axis acceleration column
            mya_col: Name of Y-axis acceleration column
            rza_col: Name of Z-axis acceleration column
            
        Returns:
            DataFrame with added movement_intensity column
        """
        df = data.copy()
        
        # Calculate magnitude
        df['movement_intensity'] = np.sqrt(
            df[fxa_col]**2 + 
            df[mya_col]**2 + 
            df[rza_col]**2
        )
        
        return df
    
    def aggregate_hourly(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        state_col: str = 'behavioral_state',
        intensity_col: str = 'movement_intensity'
    ) -> pd.DataFrame:
        """
        Aggregate behavioral data by hour.
        
        Calculates:
        - Minutes spent in each state per hour
        - Average movement intensity per hour
        - State transition counts per hour
        - Activity/rest ratio per hour
        
        Args:
            data: DataFrame with behavioral data
            timestamp_col: Name of timestamp column
            state_col: Name of behavioral state column
            intensity_col: Name of movement intensity column
            
        Returns:
            DataFrame with hourly aggregations
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Create hour grouping column
        df['hour'] = df[timestamp_col].dt.floor('H')
        
        # Calculate state durations per hour
        hourly_stats = []
        
        for hour, hour_data in df.groupby('hour'):
            stats = {'hour': hour}
            
            # Calculate time difference for duration
            hour_data = hour_data.sort_values(timestamp_col).reset_index(drop=True)
            time_diffs = hour_data[timestamp_col].diff().dt.total_seconds() / 60.0
            
            # For first row, use median or 1 minute
            median_interval = time_diffs.median()
            if pd.isna(median_interval) or median_interval == 0:
                median_interval = 1.0
            time_diffs.iloc[0] = median_interval
            
            # Minutes per state
            for state in self.BEHAVIORAL_STATES:
                state_mask = hour_data[state_col] == state
                stats[f'{state}_minutes'] = time_diffs[state_mask].sum()
            
            # Total minutes (should be ~60)
            stats['total_minutes'] = time_diffs.sum()
            
            # Average movement intensity
            if intensity_col in hour_data.columns:
                stats['avg_movement_intensity'] = hour_data[intensity_col].mean()
            
            # State transitions
            transitions = (hour_data[state_col] != hour_data[state_col].shift(1)).sum()
            stats['state_transitions'] = transitions
            
            # Activity/rest ratio
            rest_minutes = sum(stats.get(f'{state}_minutes', 0) for state in self.REST_STATES)
            active_minutes = sum(stats.get(f'{state}_minutes', 0) for state in self.ACTIVE_STATES)
            total_minutes = stats['total_minutes']
            
            stats['rest_minutes'] = rest_minutes
            stats['active_minutes'] = active_minutes
            stats['rest_percentage'] = (rest_minutes / total_minutes * 100) if total_minutes > 0 else 0
            stats['active_percentage'] = (active_minutes / total_minutes * 100) if total_minutes > 0 else 0
            
            hourly_stats.append(stats)
        
        return pd.DataFrame(hourly_stats)
    
    def aggregate_daily(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        state_col: str = 'behavioral_state',
        intensity_col: str = 'movement_intensity'
    ) -> pd.DataFrame:
        """
        Aggregate behavioral data by day.
        
        Calculates:
        - Minutes spent in each state per day (should sum to 1440)
        - Average movement intensity per day
        - State transition counts per day
        - Activity/rest ratio per day
        
        Args:
            data: DataFrame with behavioral data
            timestamp_col: Name of timestamp column
            state_col: Name of behavioral state column
            intensity_col: Name of movement intensity column
            
        Returns:
            DataFrame with daily aggregations
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Create day grouping column
        df['day'] = df[timestamp_col].dt.date
        
        # Calculate state durations per day
        daily_stats = []
        
        for day, day_data in df.groupby('day'):
            stats = {'day': pd.Timestamp(day)}
            
            # Calculate time difference for duration
            day_data = day_data.sort_values(timestamp_col).reset_index(drop=True)
            time_diffs = day_data[timestamp_col].diff().dt.total_seconds() / 60.0
            
            # For first row, use median or 1 minute
            median_interval = time_diffs.median()
            if pd.isna(median_interval) or median_interval == 0:
                median_interval = 1.0
            time_diffs.iloc[0] = median_interval
            
            # Minutes per state
            for state in self.BEHAVIORAL_STATES:
                state_mask = day_data[state_col] == state
                stats[f'{state}_minutes'] = time_diffs[state_mask].sum()
            
            # Total minutes (should be ~1440 for full day)
            stats['total_minutes'] = time_diffs.sum()
            
            # Average movement intensity
            if intensity_col in day_data.columns:
                stats['avg_movement_intensity'] = day_data[intensity_col].mean()
            
            # State transitions
            transitions = (day_data[state_col] != day_data[state_col].shift(1)).sum()
            stats['state_transitions'] = transitions
            
            # Activity/rest ratio
            rest_minutes = sum(stats.get(f'{state}_minutes', 0) for state in self.REST_STATES)
            active_minutes = sum(stats.get(f'{state}_minutes', 0) for state in self.ACTIVE_STATES)
            total_minutes = stats['total_minutes']
            
            stats['rest_minutes'] = rest_minutes
            stats['active_minutes'] = active_minutes
            stats['rest_percentage'] = (rest_minutes / total_minutes * 100) if total_minutes > 0 else 0
            stats['active_percentage'] = (active_minutes / total_minutes * 100) if total_minutes > 0 else 0
            
            daily_stats.append(stats)
        
        return pd.DataFrame(daily_stats)
    
    def calculate_activity_rest_ratio(
        self,
        data: pd.DataFrame,
        state_col: str = 'behavioral_state'
    ) -> Dict[str, float]:
        """
        Calculate overall activity/rest ratio for the dataset.
        
        Args:
            data: DataFrame with behavioral state column
            state_col: Name of behavioral state column
            
        Returns:
            Dictionary with rest and active percentages
        """
        df = data.copy()
        
        total_rows = len(df)
        if total_rows == 0:
            return {
                'rest_percentage': 0.0,
                'active_percentage': 0.0,
                'rest_count': 0,
                'active_count': 0
            }
        
        rest_count = df[state_col].isin(self.REST_STATES).sum()
        active_count = df[state_col].isin(self.ACTIVE_STATES).sum()
        
        return {
            'rest_percentage': (rest_count / total_rows) * 100,
            'active_percentage': (active_count / total_rows) * 100,
            'rest_count': int(rest_count),
            'active_count': int(active_count),
            'total_count': total_rows
        }
    
    def count_state_transitions(
        self,
        data: pd.DataFrame,
        state_col: str = 'behavioral_state',
        timestamp_col: str = 'timestamp'
    ) -> Dict[str, Any]:
        """
        Count state transitions and analyze transition patterns.
        
        Args:
            data: DataFrame with behavioral state column
            state_col: Name of behavioral state column
            timestamp_col: Name of timestamp column
            
        Returns:
            Dictionary with transition statistics
        """
        df = data.copy()
        
        # Sort by timestamp
        if timestamp_col in df.columns:
            df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Detect transitions
        state_changed = df[state_col] != df[state_col].shift(1)
        total_transitions = state_changed.sum() - 1  # Exclude first row
        
        # Count transitions between specific states
        transition_matrix = {}
        for i in range(1, len(df)):
            if state_changed.iloc[i]:
                from_state = df[state_col].iloc[i-1]
                to_state = df[state_col].iloc[i]
                key = f"{from_state}_to_{to_state}"
                transition_matrix[key] = transition_matrix.get(key, 0) + 1
        
        return {
            'total_transitions': int(total_transitions),
            'transition_matrix': transition_matrix,
            'avg_transitions_per_day': None  # Will be calculated if timestamp available
        }
    
    def generate_behavioral_log(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        state_col: str = 'behavioral_state',
        confidence_col: Optional[str] = None,
        include_movement_intensity: bool = True
    ) -> pd.DataFrame:
        """
        Generate behavioral state log with all required fields.
        
        Output fields:
        - timestamp
        - behavioral_state
        - confidence_score
        - duration_minutes
        - movement_intensity
        
        Args:
            data: DataFrame with classification output and sensor data
            timestamp_col: Name of timestamp column
            state_col: Name of behavioral state column
            confidence_col: Name of confidence score column (optional)
            include_movement_intensity: Whether to calculate movement intensity
            
        Returns:
            DataFrame with behavioral log format
        """
        df = data.copy()
        
        # Calculate durations
        df = self.calculate_durations(df, timestamp_col, state_col)
        
        # Calculate movement intensity if requested and data available
        if include_movement_intensity and all(col in df.columns for col in ['fxa', 'mya', 'rza']):
            df = self.calculate_movement_intensity(df)
        
        # Build output dataframe
        log_df = pd.DataFrame()
        log_df['timestamp'] = df[timestamp_col]
        log_df['behavioral_state'] = df[state_col]
        
        # Add confidence score
        if confidence_col and confidence_col in df.columns:
            log_df['confidence_score'] = df[confidence_col]
        else:
            # Default confidence if not provided
            log_df['confidence_score'] = 1.0
        
        log_df['duration_minutes'] = df['duration_minutes']
        
        # Add movement intensity
        if 'movement_intensity' in df.columns:
            log_df['movement_intensity'] = df['movement_intensity']
        else:
            log_df['movement_intensity'] = 0.0
        
        return log_df
    
    def export_to_csv(
        self,
        data: pd.DataFrame,
        filename: str,
        include_timestamp: bool = True
    ) -> Path:
        """
        Export behavioral data to CSV format.
        
        Args:
            data: DataFrame to export
            filename: Output filename
            include_timestamp: Whether to add timestamp to filename
            
        Returns:
            Path to exported file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(filename).stem
            extension = Path(filename).suffix or '.csv'
            filename = f"{base_name}_{timestamp}{extension}"
        
        output_path = self.output_dir / filename
        data.to_csv(output_path, index=False)
        
        logger.info(f"Exported {len(data)} rows to CSV: {output_path}")
        return output_path
    
    def export_to_json(
        self,
        data: pd.DataFrame,
        filename: str,
        include_timestamp: bool = True,
        orient: str = 'records'
    ) -> Path:
        """
        Export behavioral data to JSON format.
        
        Args:
            data: DataFrame to export
            filename: Output filename
            include_timestamp: Whether to add timestamp to filename
            orient: JSON orientation ('records', 'index', 'columns', etc.)
            
        Returns:
            Path to exported file
        """
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(filename).stem
            extension = Path(filename).suffix or '.json'
            filename = f"{base_name}_{timestamp}{extension}"
        
        output_path = self.output_dir / filename
        
        # Convert datetime to string for JSON serialization
        df = data.copy()
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        df.to_json(output_path, orient=orient, indent=2)
        
        logger.info(f"Exported {len(data)} rows to JSON: {output_path}")
        return output_path
    
    def process_full_pipeline(
        self,
        data: pd.DataFrame,
        export_logs: bool = True,
        export_hourly: bool = True,
        export_daily: bool = True,
        csv_format: bool = True,
        json_format: bool = True
    ) -> Dict[str, Any]:
        """
        Run full activity metrics pipeline.
        
        Args:
            data: Input DataFrame with behavioral states and sensor data
            export_logs: Whether to export behavioral logs
            export_hourly: Whether to export hourly aggregations
            export_daily: Whether to export daily aggregations
            csv_format: Whether to export CSV files
            json_format: Whether to export JSON files
            
        Returns:
            Dictionary with all computed metrics and export paths
        """
        results = {
            'behavioral_log': None,
            'hourly_aggregation': None,
            'daily_aggregation': None,
            'activity_rest_ratio': None,
            'state_transitions': None,
            'export_paths': {}
        }
        
        logger.info("Starting full activity metrics pipeline...")
        
        # Generate behavioral log
        behavioral_log = self.generate_behavioral_log(data)
        results['behavioral_log'] = behavioral_log
        
        if export_logs:
            if csv_format:
                path = self.export_to_csv(behavioral_log, 'behavioral_log.csv')
                results['export_paths']['log_csv'] = str(path)
            if json_format:
                path = self.export_to_json(behavioral_log, 'behavioral_log.json')
                results['export_paths']['log_json'] = str(path)
        
        # Hourly aggregation
        hourly_agg = self.aggregate_hourly(behavioral_log)
        results['hourly_aggregation'] = hourly_agg
        
        if export_hourly:
            if csv_format:
                path = self.export_to_csv(hourly_agg, 'hourly_aggregation.csv')
                results['export_paths']['hourly_csv'] = str(path)
            if json_format:
                path = self.export_to_json(hourly_agg, 'hourly_aggregation.json')
                results['export_paths']['hourly_json'] = str(path)
        
        # Daily aggregation
        daily_agg = self.aggregate_daily(behavioral_log)
        results['daily_aggregation'] = daily_agg
        
        if export_daily:
            if csv_format:
                path = self.export_to_csv(daily_agg, 'daily_aggregation.csv')
                results['export_paths']['daily_csv'] = str(path)
            if json_format:
                path = self.export_to_json(daily_agg, 'daily_aggregation.json')
                results['export_paths']['daily_json'] = str(path)
        
        # Activity/rest ratio
        activity_rest = self.calculate_activity_rest_ratio(behavioral_log)
        results['activity_rest_ratio'] = activity_rest
        
        # State transitions
        transitions = self.count_state_transitions(behavioral_log)
        results['state_transitions'] = transitions
        
        logger.info("Full activity metrics pipeline complete")
        
        return results
    
    def handle_missing_data(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        state_col: str = 'behavioral_state',
        fill_method: str = 'forward'
    ) -> pd.DataFrame:
        """
        Handle missing data windows in behavioral state data.
        
        Args:
            data: DataFrame with potential missing data
            timestamp_col: Name of timestamp column
            state_col: Name of behavioral state column
            fill_method: Method to fill gaps ('forward', 'backward', 'interpolate', 'unknown')
            
        Returns:
            DataFrame with handled missing data
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Detect gaps (missing minutes)
        time_diffs = df[timestamp_col].diff()
        expected_interval = time_diffs.median()
        
        # Find gaps larger than expected
        gaps = time_diffs > (expected_interval * 1.5)
        
        if gaps.any():
            logger.warning(f"Found {gaps.sum()} time gaps in data")
            
            if fill_method == 'forward':
                df[state_col] = df[state_col].fillna(method='ffill')
            elif fill_method == 'backward':
                df[state_col] = df[state_col].fillna(method='bfill')
            elif fill_method == 'unknown':
                df[state_col] = df[state_col].fillna('unknown')
            # interpolate not applicable for categorical data
        
        # Handle any remaining NaN values
        df[state_col] = df[state_col].fillna('unknown')
        
        return df
    
    def validate_daily_totals(
        self,
        daily_agg: pd.DataFrame,
        expected_minutes: int = 1440,
        tolerance: float = 5.0
    ) -> Dict[str, Any]:
        """
        Validate that daily aggregations sum to expected total minutes.
        
        Args:
            daily_agg: Daily aggregation DataFrame
            expected_minutes: Expected total minutes per day (1440 for 24 hours)
            tolerance: Acceptable deviation in minutes
            
        Returns:
            Dictionary with validation results
        """
        validation = {
            'all_valid': True,
            'invalid_days': [],
            'details': []
        }
        
        for _, row in daily_agg.iterrows():
            total = row['total_minutes']
            deviation = abs(total - expected_minutes)
            
            if deviation > tolerance:
                validation['all_valid'] = False
                validation['invalid_days'].append({
                    'day': str(row['day']),
                    'total_minutes': total,
                    'deviation': deviation
                })
        
        if validation['all_valid']:
            logger.info(f"All {len(daily_agg)} days validated successfully")
        else:
            logger.warning(f"{len(validation['invalid_days'])} days have invalid totals")
        
        return validation
