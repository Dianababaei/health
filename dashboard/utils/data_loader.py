"""
Data Loader Utilities for Artemis Health Dashboard

Provides functions to load sensor data, alerts, and behavioral states
from various sources (CSV files, JSON logs, database).
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import glob

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Utility class for loading data for the dashboard.
    
    Features:
    - Load sensor data from CSV files
    - Load alerts from JSON logs
    - Load behavioral state data
    - Cache management
    - Error handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data loader with configuration.
        
        Args:
            config: Dashboard configuration dictionary
        """
        self.config = config
        self.data_sources = config.get('data_sources', {})
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = config.get('cache', {}).get('ttl_seconds', 300)
        
    def load_sensor_data(
        self,
        data_dir: Optional[str] = None,
        file_pattern: Optional[str] = None,
        time_range_hours: Optional[int] = None,
        max_rows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load sensor data from CSV files.
        
        Args:
            data_dir: Directory containing sensor data files
            file_pattern: Pattern for matching files (e.g., "*.csv")
            time_range_hours: Filter data to last N hours
            max_rows: Maximum number of rows to return
            
        Returns:
            DataFrame with sensor data
        """
        try:
            # Use default values from config if not provided
            if data_dir is None:
                data_dir = self.data_sources.get('simulated_data_dir', 'data/simulated')
            if file_pattern is None:
                file_pattern = self.data_sources.get('sensor_data_pattern', '*.csv')
            
            # Find all matching files
            search_path = Path(data_dir) / file_pattern
            files = glob.glob(str(search_path))
            
            if not files:
                logger.warning(f"No sensor data files found in {data_dir}")
                return self._create_empty_sensor_dataframe()
            
            # Load most recent file (or all files)
            dfs = []
            for file in sorted(files)[-1:]:  # Load most recent file
                try:
                    df = pd.read_csv(file)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Error loading file {file}: {e}")
            
            if not dfs:
                return self._create_empty_sensor_dataframe()
            
            # Combine dataframes
            df = pd.concat(dfs, ignore_index=True)
            
            # Parse timestamp column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Filter by time range
            if time_range_hours and 'timestamp' in df.columns:
                cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
                df = df[df['timestamp'] >= cutoff_time]
            
            # Limit rows
            if max_rows and len(df) > max_rows:
                df = df.tail(max_rows)
            
            logger.info(f"Loaded {len(df)} sensor data rows from {len(dfs)} files")
            return df
            
        except Exception as e:
            logger.error(f"Error loading sensor data: {e}")
            return self._create_empty_sensor_dataframe()
    
    def load_alerts(
        self,
        alert_log_file: Optional[str] = None,
        max_alerts: Optional[int] = None,
        filter_severity: Optional[str] = None,
        filter_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load alerts from JSON log file.
        
        Args:
            alert_log_file: Path to alert log file
            max_alerts: Maximum number of alerts to return
            filter_severity: Filter by severity (critical, high, medium, low)
            filter_type: Filter by alert type
            
        Returns:
            List of alert dictionaries
        """
        try:
            # Use default value from config if not provided
            if alert_log_file is None:
                alert_log_file = self.data_sources.get('alert_log_file', 'logs/malfunction_alerts.json')
            
            alert_path = Path(alert_log_file)
            
            if not alert_path.exists():
                logger.warning(f"Alert log file not found: {alert_log_file}")
                return []
            
            # Read alerts from JSON log file (one JSON object per line)
            alerts = []
            with open(alert_path, 'r') as f:
                for line in f:
                    try:
                        alert = json.loads(line.strip())
                        
                        # Apply filters
                        if filter_severity and alert.get('severity') != filter_severity:
                            continue
                        if filter_type and alert.get('malfunction_type') != filter_type:
                            continue
                        
                        alerts.append(alert)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing alert JSON: {e}")
            
            # Sort by timestamp (most recent first)
            if alerts and 'detection_time' in alerts[0]:
                alerts.sort(key=lambda x: x.get('detection_time', ''), reverse=True)
            
            # Limit number of alerts
            if max_alerts:
                alerts = alerts[:max_alerts]
            
            logger.info(f"Loaded {len(alerts)} alerts from {alert_log_file}")
            return alerts
            
        except Exception as e:
            logger.error(f"Error loading alerts: {e}")
            return []
    
    def load_behavioral_data(
        self,
        data_dir: Optional[str] = None,
        time_range_hours: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Load behavioral state data from sensor data files.
        
        Args:
            data_dir: Directory containing data files
            time_range_hours: Filter data to last N hours
            
        Returns:
            DataFrame with behavioral state data
        """
        try:
            # Load sensor data (which may include behavioral_state column)
            df = self.load_sensor_data(
                data_dir=data_dir,
                time_range_hours=time_range_hours
            )
            
            # Check if behavioral_state column exists
            if 'behavioral_state' not in df.columns:
                logger.warning("No behavioral_state column found in data")
                df['behavioral_state'] = 'unknown'
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading behavioral data: {e}")
            return self._create_empty_sensor_dataframe()
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get latest metrics for the overview dashboard.
        
        Returns:
            Dictionary with latest metrics
        """
        try:
            # Load most recent sensor data
            df = self.load_sensor_data(time_range_hours=1)
            
            if df.empty:
                return self._create_empty_metrics()
            
            # Calculate metrics from latest data
            latest_row = df.iloc[-1] if len(df) > 0 else {}
            
            metrics = {
                'timestamp': latest_row.get('timestamp', datetime.now()),
                'temperature': latest_row.get('temperature', None),
                'activity_level': self._calculate_activity_level(df),
                'current_state': latest_row.get('behavioral_state', 'unknown'),
                'data_points': len(df),
                'time_range': '1 hour',
            }
            
            # Add temperature status
            if metrics['temperature']:
                temp = metrics['temperature']
                temp_config = self.config.get('metrics', {}).get('temperature', {})
                if temp >= temp_config.get('fever_threshold', 39.5):
                    metrics['temperature_status'] = 'fever'
                elif temp <= temp_config.get('hypothermia_threshold', 37.5):
                    metrics['temperature_status'] = 'hypothermia'
                else:
                    metrics['temperature_status'] = 'normal'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting latest metrics: {e}")
            return self._create_empty_metrics()
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent alerts.
        
        Returns:
            Dictionary with alert summary
        """
        try:
            alerts = self.load_alerts(max_alerts=100)
            
            summary = {
                'total_alerts': len(alerts),
                'active_alerts': 0,
                'by_severity': {},
                'by_type': {},
                'latest_alert': None,
            }
            
            if not alerts:
                return summary
            
            # Count by severity and type
            for alert in alerts:
                severity = alert.get('severity', 'unknown')
                alert_type = alert.get('malfunction_type', 'unknown')
                
                summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
                summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
                
                # Check if alert is active (within last 24 hours)
                if 'detection_time' in alert:
                    try:
                        detection_time = pd.to_datetime(alert['detection_time'])
                        if datetime.now() - detection_time.to_pydatetime() < timedelta(hours=24):
                            summary['active_alerts'] += 1
                    except:
                        pass
            
            summary['latest_alert'] = alerts[0] if alerts else None
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {'total_alerts': 0, 'active_alerts': 0, 'by_severity': {}, 'by_type': {}}
    
    def _calculate_activity_level(self, df: pd.DataFrame) -> float:
        """Calculate activity level from sensor data."""
        try:
            if df.empty:
                return 0.0
            
            # Calculate average magnitude of accelerometer readings
            accel_cols = ['fxa', 'mya', 'rza']
            if all(col in df.columns for col in accel_cols):
                magnitude = (df[accel_cols] ** 2).sum(axis=1).apply(lambda x: x ** 0.5)
                return float(magnitude.mean())
            
            return 0.0
        except:
            return 0.0
    
    def _create_empty_sensor_dataframe(self) -> pd.DataFrame:
        """Create an empty sensor dataframe with proper columns."""
        return pd.DataFrame(columns=[
            'timestamp', 'temperature', 'fxa', 'mya', 'rza', 
            'sxg', 'lyg', 'dzg', 'behavioral_state'
        ])
    
    def _create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics dictionary."""
        return {
            'timestamp': None,
            'temperature': None,
            'activity_level': 0.0,
            'current_state': 'unknown',
            'data_points': 0,
            'time_range': 'N/A',
        }


def load_config(config_path: str = "dashboard/config.yaml") -> Dict[str, Any]:
    """
    Load dashboard configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        # Return default configuration
        return {
            'dashboard': {
                'title': 'Artemis Health',
                'auto_refresh_interval_seconds': 60,
            },
            'data_sources': {
                'simulated_data_dir': 'data/simulated',
                'alert_log_file': 'logs/malfunction_alerts.json',
            },
        }
