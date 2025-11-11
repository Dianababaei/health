"""
Trend Aggregator Module
=======================
Module for aggregating multi-day trend data and exporting analysis results.
Supports CSV and JSON export formats with configurable date ranges and metrics.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class TrendAggregator:
    """
    Aggregates and exports trend analysis data from multiple sources.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the TrendAggregator.
        
        Args:
            data_dir: Directory for reading/writing data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path('data')
        
    def aggregate_daily_metrics(
        self,
        sensor_data: pd.DataFrame,
        behavioral_data: Optional[pd.DataFrame] = None,
        health_scores: Optional[pd.DataFrame] = None,
        time_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Aggregate metrics by day for long-term trend analysis.
        
        Args:
            sensor_data: Raw sensor data
            behavioral_data: Behavioral state data
            health_scores: Health score data
            time_column: Column name for timestamps
            
        Returns:
            DataFrame with daily aggregated metrics
        """
        if sensor_data.empty:
            return pd.DataFrame()
        
        df = sensor_data.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df['date'] = df[time_column].dt.date
        
        # Aggregate sensor data by day
        daily_agg = {}
        
        # Temperature metrics
        if 'temperature' in df.columns:
            daily_agg['temp_mean'] = df.groupby('date')['temperature'].mean()
            daily_agg['temp_min'] = df.groupby('date')['temperature'].min()
            daily_agg['temp_max'] = df.groupby('date')['temperature'].max()
            daily_agg['temp_std'] = df.groupby('date')['temperature'].std()
        
        # Activity metrics (from accelerometer)
        accel_cols = ['fxa', 'mya', 'rza']
        if all(col in df.columns for col in accel_cols):
            df['activity_magnitude'] = np.sqrt(
                df['fxa']**2 + df['mya']**2 + df['rza']**2
            )
            daily_agg['activity_mean'] = df.groupby('date')['activity_magnitude'].mean()
            daily_agg['activity_max'] = df.groupby('date')['activity_magnitude'].max()
        
        # Behavioral state distribution
        if behavioral_data is not None and not behavioral_data.empty:
            behavioral_data['date'] = pd.to_datetime(behavioral_data[time_column]).dt.date
            
            # Calculate time spent in each state per day
            state_counts = behavioral_data.groupby(['date', 'state']).size().unstack(fill_value=0)
            for state in state_counts.columns:
                daily_agg[f'time_{state}'] = state_counts[state]
        
        # Health scores
        if health_scores is not None and not health_scores.empty:
            health_scores['date'] = pd.to_datetime(health_scores[time_column]).dt.date
            daily_agg['health_score_mean'] = health_scores.groupby('date')['health_score'].mean()
            daily_agg['health_score_min'] = health_scores.groupby('date')['health_score'].min()
        
        # Combine all aggregations
        result = pd.DataFrame(daily_agg)
        result.index.name = 'date'
        result = result.reset_index()
        
        return result
    
    def aggregate_weekly_metrics(
        self,
        daily_data: pd.DataFrame,
        date_column: str = 'date'
    ) -> pd.DataFrame:
        """
        Aggregate daily metrics into weekly summaries.
        
        Args:
            daily_data: Daily aggregated data
            date_column: Column name for dates
            
        Returns:
            DataFrame with weekly aggregated metrics
        """
        if daily_data.empty:
            return pd.DataFrame()
        
        df = daily_data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        
        # Resample to weekly
        weekly = df.resample('W').agg({
            col: ['mean', 'min', 'max', 'std'] if df[col].dtype in [np.float64, np.int64] else 'first'
            for col in df.columns
        })
        
        # Flatten column names
        if isinstance(weekly.columns, pd.MultiIndex):
            weekly.columns = ['_'.join(col).strip() for col in weekly.columns.values]
        
        weekly = weekly.reset_index()
        return weekly
    
    def export_to_csv(
        self,
        data: pd.DataFrame,
        filename: str,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Export data to CSV file.
        
        Args:
            data: DataFrame to export
            filename: Output filename
            output_dir: Output directory (defaults to self.data_dir)
            
        Returns:
            Path to exported file
        """
        if data.empty:
            logger.warning("No data to export")
            return ""
        
        output_path = Path(output_dir) if output_dir else self.data_dir / 'exports'
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        data.to_csv(filepath, index=False)
        
        logger.info(f"Exported {len(data)} rows to {filepath}")
        return str(filepath)
    
    def export_to_json(
        self,
        data: Union[pd.DataFrame, Dict, List],
        filename: str,
        output_dir: Optional[str] = None,
        orient: str = 'records'
    ) -> str:
        """
        Export data to JSON file.
        
        Args:
            data: Data to export (DataFrame, dict, or list)
            filename: Output filename
            output_dir: Output directory (defaults to self.data_dir)
            orient: JSON orientation for DataFrames ('records', 'index', 'columns')
            
        Returns:
            Path to exported file
        """
        output_path = Path(output_dir) if output_dir else self.data_dir / 'exports'
        output_path.mkdir(parents=True, exist_ok=True)
        
        filepath = output_path / filename
        
        if isinstance(data, pd.DataFrame):
            if data.empty:
                logger.warning("No data to export")
                return ""
            
            # Convert datetime columns to string
            df = data.copy()
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = df[col].astype(str)
            
            with open(filepath, 'w') as f:
                df.to_json(f, orient=orient, indent=2)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Exported data to {filepath}")
        return str(filepath)
    
    def create_trend_export_package(
        self,
        sensor_data: pd.DataFrame,
        behavioral_data: Optional[pd.DataFrame] = None,
        health_scores: Optional[pd.DataFrame] = None,
        alerts: Optional[List[Dict]] = None,
        patterns: Optional[List[Dict]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        export_format: str = 'csv'
    ) -> Dict[str, str]:
        """
        Create a complete export package with all trend analysis data.
        
        Args:
            sensor_data: Raw sensor data
            behavioral_data: Behavioral state data
            health_scores: Health score data
            alerts: List of alerts
            patterns: Detected patterns
            start_date: Start date for export (optional)
            end_date: End date for export (optional)
            export_format: Format to export ('csv', 'json', or 'both')
            
        Returns:
            Dictionary mapping file types to file paths
        """
        exported_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Filter by date range if specified
        if start_date or end_date:
            if not sensor_data.empty and 'timestamp' in sensor_data.columns:
                sensor_data = self._filter_by_date_range(
                    sensor_data, start_date, end_date, 'timestamp'
                )
            if behavioral_data is not None and not behavioral_data.empty:
                behavioral_data = self._filter_by_date_range(
                    behavioral_data, start_date, end_date, 'timestamp'
                )
            if health_scores is not None and not health_scores.empty:
                health_scores = self._filter_by_date_range(
                    health_scores, start_date, end_date, 'timestamp'
                )
        
        # Export sensor data
        if not sensor_data.empty:
            if export_format in ['csv', 'both']:
                filepath = self.export_to_csv(
                    sensor_data,
                    f'sensor_data_{timestamp}.csv'
                )
                exported_files['sensor_csv'] = filepath
            
            if export_format in ['json', 'both']:
                filepath = self.export_to_json(
                    sensor_data,
                    f'sensor_data_{timestamp}.json'
                )
                exported_files['sensor_json'] = filepath
        
        # Export daily aggregated data
        daily_data = self.aggregate_daily_metrics(
            sensor_data, behavioral_data, health_scores
        )
        
        if not daily_data.empty:
            if export_format in ['csv', 'both']:
                filepath = self.export_to_csv(
                    daily_data,
                    f'daily_metrics_{timestamp}.csv'
                )
                exported_files['daily_csv'] = filepath
            
            if export_format in ['json', 'both']:
                filepath = self.export_to_json(
                    daily_data,
                    f'daily_metrics_{timestamp}.json'
                )
                exported_files['daily_json'] = filepath
        
        # Export weekly aggregated data
        if not daily_data.empty:
            weekly_data = self.aggregate_weekly_metrics(daily_data)
            
            if not weekly_data.empty:
                if export_format in ['csv', 'both']:
                    filepath = self.export_to_csv(
                        weekly_data,
                        f'weekly_metrics_{timestamp}.csv'
                    )
                    exported_files['weekly_csv'] = filepath
        
        # Export alerts
        if alerts:
            filepath = self.export_to_json(
                alerts,
                f'alerts_{timestamp}.json'
            )
            exported_files['alerts_json'] = filepath
        
        # Export patterns
        if patterns:
            filepath = self.export_to_json(
                patterns,
                f'patterns_{timestamp}.json'
            )
            exported_files['patterns_json'] = filepath
        
        # Create summary metadata
        metadata = {
            'export_timestamp': timestamp,
            'export_format': export_format,
            'date_range': {
                'start': str(start_date) if start_date else 'N/A',
                'end': str(end_date) if end_date else 'N/A',
            },
            'data_summary': {
                'sensor_records': len(sensor_data),
                'behavioral_records': len(behavioral_data) if behavioral_data is not None else 0,
                'health_scores': len(health_scores) if health_scores is not None else 0,
                'alerts': len(alerts) if alerts else 0,
                'patterns': len(patterns) if patterns else 0,
            },
            'exported_files': list(exported_files.values()),
        }
        
        filepath = self.export_to_json(
            metadata,
            f'export_metadata_{timestamp}.json'
        )
        exported_files['metadata'] = filepath
        
        logger.info(f"Created export package with {len(exported_files)} files")
        return exported_files
    
    def _filter_by_date_range(
        self,
        data: pd.DataFrame,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        time_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter DataFrame by date range.
        
        Args:
            data: DataFrame to filter
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            time_column: Column name for timestamps
            
        Returns:
            Filtered DataFrame
        """
        if data.empty or time_column not in data.columns:
            return data
        
        df = data.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        
        if start_date:
            df = df[df[time_column] >= start_date]
        
        if end_date:
            df = df[df[time_column] <= end_date]
        
        return df


def export_pattern_summary(
    patterns: List[Dict[str, Any]],
    output_file: str = 'pattern_summary.csv'
) -> str:
    """
    Export pattern detection summary to CSV.
    
    Args:
        patterns: List of detected patterns
        output_file: Output filename
        
    Returns:
        Path to exported file
    """
    if not patterns:
        logger.warning("No patterns to export")
        return ""
    
    df = pd.DataFrame(patterns)
    
    # Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    
    # Export
    aggregator = TrendAggregator()
    return aggregator.export_to_csv(df, output_file)


def create_comparison_export(
    period_data: Dict[str, pd.DataFrame],
    metric_name: str,
    output_file: str = 'period_comparison.csv'
) -> str:
    """
    Create a comparison export of metrics across different time periods.
    
    Args:
        period_data: Dictionary mapping period labels to DataFrames
        metric_name: Name of the metric being compared
        output_file: Output filename
        
    Returns:
        Path to exported file
    """
    if not period_data:
        logger.warning("No period data to export")
        return ""
    
    # Combine data from all periods
    combined_data = []
    
    for period_label, df in period_data.items():
        if not df.empty and metric_name in df.columns:
            temp_df = df[['timestamp', metric_name]].copy()
            temp_df['period'] = period_label
            combined_data.append(temp_df)
    
    if not combined_data:
        logger.warning(f"No data found for metric: {metric_name}")
        return ""
    
    result = pd.concat(combined_data, ignore_index=True)
    
    # Export
    aggregator = TrendAggregator()
    return aggregator.export_to_csv(result, output_file)
