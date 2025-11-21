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

        NEW: Automatically checks simulation directory first if prefer_simulation is true.

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
                # Check dashboard directory first
                prefer_dashboard = self.data_sources.get('prefer_dashboard', True)
                if prefer_dashboard:
                    dashboard_dir = self.data_sources.get('dashboard_data_dir', 'data/dashboard')
                    dashboard_path = Path(dashboard_dir)
                    if dashboard_path.exists():
                        dashboard_files = list(dashboard_path.glob('*_sensor_data.csv'))
                        if dashboard_files:
                            # Use most recent dashboard data
                            latest_data = sorted(dashboard_files, key=lambda x: x.stat().st_mtime)[-1]
                            logger.info(f"Using dashboard data: {latest_data}")
                            data_dir = dashboard_dir
                            file_pattern = latest_data.name

                # Fallback to default directory
                if data_dir is None:
                    data_dir = self.data_sources.get('dashboard_data_dir', 'data/dashboard')

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

        NEW: Automatically checks simulation directory first if prefer_simulation is true.

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
                # Check dashboard directory first
                prefer_dashboard = self.data_sources.get('prefer_dashboard', True)
                if prefer_dashboard:
                    dashboard_dir = self.data_sources.get('dashboard_data_dir', 'data/dashboard')
                    dashboard_path = Path(dashboard_dir)
                    if dashboard_path.exists():
                        dashboard_files = list(dashboard_path.glob('*_alerts.json'))
                        if dashboard_files:
                            # Use most recent dashboard alerts
                            latest_alerts = sorted(dashboard_files, key=lambda x: x.stat().st_mtime)[-1]
                            logger.info(f"Using dashboard alerts: {latest_alerts}")
                            alert_log_file = str(latest_alerts)

                # Fallback to default alert log
                if alert_log_file is None:
                    alert_log_file = self.data_sources.get('alert_log_file', 'logs/malfunction_alerts.json')

            alert_path = Path(alert_log_file)

            if not alert_path.exists():
                logger.warning(f"Alert log file not found: {alert_log_file}")
                return []

            # Check if it's a simulation alert file (JSON array) or log file (JSONL)
            alerts = []
            with open(alert_path, 'r') as f:
                content = f.read().strip()

                if not content or content == '[]':
                    # Empty file or empty array
                    logger.info(f"Alert file is empty: {alert_log_file}")
                    return []

                if content.startswith('['):
                    # JSON array format (simulation alerts)
                    try:
                        alerts = json.loads(content)
                        logger.info(f"Loaded {len(alerts)} alerts from JSON array format")
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing alert JSON array: {e}")
                        return []
                else:
                    # JSONL format (one JSON object per line)
                    for line in content.split('\n'):
                        if not line.strip():
                            continue
                        try:
                            alert = json.loads(line.strip())
                            alerts.append(alert)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing alert JSON line: {e}")
                    logger.info(f"Loaded {len(alerts)} alerts from JSONL format")

            # Apply filters
            filtered_alerts = []
            for alert in alerts:
                # Normalize severity field (might be 'severity' or in alert data)
                severity = alert.get('severity', alert.get('malfunction_severity', ''))
                alert_type = alert.get('alert_type', alert.get('malfunction_type', ''))

                # Apply filters
                if filter_severity and str(severity).lower() != filter_severity.lower():
                    continue
                if filter_type and str(alert_type).lower() != filter_type.lower():
                    continue

                filtered_alerts.append(alert)

            # Sort by timestamp (most recent first)
            if filtered_alerts:
                # Try different timestamp fields
                timestamp_field = None
                for field in ['timestamp', 'detection_time', 'created_at']:
                    if field in filtered_alerts[0]:
                        timestamp_field = field
                        break

                if timestamp_field:
                    filtered_alerts.sort(key=lambda x: x.get(timestamp_field, ''), reverse=True)

            # Limit number of alerts
            if max_alerts:
                filtered_alerts = filtered_alerts[:max_alerts]

            logger.info(f"Loaded {len(filtered_alerts)} alerts from {alert_log_file}")
            return filtered_alerts

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
