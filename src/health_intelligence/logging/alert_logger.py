"""
Alert Logger Module

Provides comprehensive alert logging with JSON and CSV output formats,
daily log rotation, and retention policy management.
"""

import logging
import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid


logger = logging.getLogger(__name__)


class AlertLogger:
    """
    Manages alert logging to JSON and CSV formats with daily rotation
    and retention policy.
    
    Features:
    - Dual format logging (JSON and CSV)
    - Daily log file rotation
    - Configurable retention policy (90-180 days)
    - Structured alert schema
    - Automatic directory creation
    """
    
    # Alert schema fields
    ALERT_FIELDS = [
        'alert_id',
        'timestamp',
        'cow_id',
        'alert_type',
        'severity',
        'confidence',
        'sensor_values',
        'detection_details',
        'status',
        'status_updated_at',
        'resolution_notes'
    ]
    
    def __init__(
        self,
        log_dir: str = "logs/alerts",
        retention_days: int = 180,
        auto_cleanup: bool = True,
    ):
        """
        Initialize alert logger.
        
        Args:
            log_dir: Directory for alert logs
            retention_days: Number of days to retain logs (90-180)
            auto_cleanup: Automatically cleanup old logs
        """
        self.log_dir = Path(log_dir)
        self.retention_days = max(90, min(retention_days, 180))  # Enforce 90-180 range
        self.auto_cleanup = auto_cleanup
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"AlertLogger initialized: dir={self.log_dir}, "
            f"retention={self.retention_days} days"
        )
        
        # Perform initial cleanup if enabled
        if self.auto_cleanup:
            self.cleanup_old_logs()
    
    def log_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Log an alert to both JSON and CSV formats.
        
        Args:
            alert_data: Alert data dictionary
            
        Returns:
            True if logging succeeded, False otherwise
        """
        try:
            # Validate and enrich alert data
            alert = self._prepare_alert(alert_data)
            
            # Log to JSON
            json_success = self._log_to_json(alert)
            
            # Log to CSV
            csv_success = self._log_to_csv(alert)
            
            if json_success and csv_success:
                logger.debug(f"Alert logged: {alert['alert_id']}")
                return True
            else:
                logger.warning(f"Partial logging failure for alert: {alert['alert_id']}")
                return False
                
        except Exception as e:
            logger.error(f"Error logging alert: {e}")
            return False
    
    def log_alerts_batch(self, alerts: List[Dict[str, Any]]) -> int:
        """
        Log multiple alerts in batch.
        
        Args:
            alerts: List of alert data dictionaries
            
        Returns:
            Number of successfully logged alerts
        """
        success_count = 0
        
        for alert_data in alerts:
            if self.log_alert(alert_data):
                success_count += 1
        
        logger.info(f"Batch logged {success_count}/{len(alerts)} alerts")
        return success_count
    
    def _prepare_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and validate alert data with required fields.
        
        Args:
            alert_data: Raw alert data
            
        Returns:
            Prepared alert dictionary
        """
        # Generate alert_id if not provided
        if 'alert_id' not in alert_data:
            alert_data['alert_id'] = str(uuid.uuid4())
        
        # Ensure timestamp is ISO8601 string
        if 'timestamp' in alert_data:
            if isinstance(alert_data['timestamp'], datetime):
                alert_data['timestamp'] = alert_data['timestamp'].isoformat()
        else:
            alert_data['timestamp'] = datetime.now().isoformat()
        
        # Set status and status_updated_at if not provided
        if 'status' not in alert_data:
            alert_data['status'] = 'active'
        
        if 'status_updated_at' not in alert_data:
            alert_data['status_updated_at'] = alert_data['timestamp']
        elif isinstance(alert_data['status_updated_at'], datetime):
            alert_data['status_updated_at'] = alert_data['status_updated_at'].isoformat()
        
        # Ensure required fields have defaults
        alert_data.setdefault('cow_id', 'unknown')
        alert_data.setdefault('alert_type', 'unknown')
        alert_data.setdefault('severity', 'info')
        alert_data.setdefault('confidence', 0.0)
        alert_data.setdefault('sensor_values', {})
        alert_data.setdefault('detection_details', {})
        alert_data.setdefault('resolution_notes', '')
        
        return alert_data
    
    def _log_to_json(self, alert: Dict[str, Any]) -> bool:
        """
        Log alert to JSON file (one JSON object per line).
        
        Args:
            alert: Prepared alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get filename with cow_id and date
            filename = self._get_json_filename(alert)
            filepath = self.log_dir / filename
            
            # Append to JSON log file (JSONL format)
            with open(filepath, 'a', encoding='utf-8') as f:
                json.dump(alert, f, ensure_ascii=False)
                f.write('\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing JSON log: {e}")
            return False
    
    def _log_to_csv(self, alert: Dict[str, Any]) -> bool:
        """
        Log alert to CSV file.
        
        Args:
            alert: Prepared alert dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get filename with cow_id and date
            filename = self._get_csv_filename(alert)
            filepath = self.log_dir / filename
            
            # Check if file exists to determine if we need to write header
            file_exists = filepath.exists()
            
            # Flatten nested dictionaries for CSV
            csv_row = self._flatten_alert_for_csv(alert)
            
            # Write to CSV
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_row.keys())
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow(csv_row)
            
            return True
            
        except Exception as e:
            logger.error(f"Error writing CSV log: {e}")
            return False
    
    def _flatten_alert_for_csv(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten alert dictionary for CSV output.
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Flattened dictionary suitable for CSV
        """
        flattened = {}
        
        for key, value in alert.items():
            if isinstance(value, (dict, list)):
                # Convert complex types to JSON string
                flattened[key] = json.dumps(value)
            else:
                flattened[key] = value
        
        return flattened
    
    def _get_json_filename(self, alert: Dict[str, Any]) -> str:
        """
        Get JSON filename for alert.
        
        Format: alerts_{cow_id}_{date}.json
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Filename string
        """
        cow_id = alert.get('cow_id', 'unknown')
        timestamp_str = alert.get('timestamp', datetime.now().isoformat())
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            date_str = timestamp.strftime('%Y%m%d')
        except:
            date_str = datetime.now().strftime('%Y%m%d')
        
        return f"alerts_{cow_id}_{date_str}.json"
    
    def _get_csv_filename(self, alert: Dict[str, Any]) -> str:
        """
        Get CSV filename for alert.
        
        Format: alerts_{cow_id}_{date}.csv
        
        Args:
            alert: Alert dictionary
            
        Returns:
            Filename string
        """
        cow_id = alert.get('cow_id', 'unknown')
        timestamp_str = alert.get('timestamp', datetime.now().isoformat())
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            date_str = timestamp.strftime('%Y%m%d')
        except:
            date_str = datetime.now().strftime('%Y%m%d')
        
        return f"alerts_{cow_id}_{date_str}.csv"
    
    def cleanup_old_logs(self) -> int:
        """
        Remove log files older than retention period.
        
        Returns:
            Number of files deleted
        """
        if not self.log_dir.exists():
            return 0
        
        deleted_count = 0
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        try:
            # Iterate through all log files
            for filepath in self.log_dir.glob("alerts_*.json"):
                if self._is_file_expired(filepath, cutoff_date):
                    filepath.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted expired log: {filepath.name}")
            
            for filepath in self.log_dir.glob("alerts_*.csv"):
                if self._is_file_expired(filepath, cutoff_date):
                    filepath.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted expired log: {filepath.name}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired log files")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during log cleanup: {e}")
            return deleted_count
    
    def _is_file_expired(self, filepath: Path, cutoff_date: datetime) -> bool:
        """
        Check if a log file is expired based on its modification time.
        
        Args:
            filepath: Path to log file
            cutoff_date: Cutoff datetime
            
        Returns:
            True if file is expired, False otherwise
        """
        try:
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            return mtime < cutoff_date
        except Exception as e:
            logger.warning(f"Error checking file age {filepath}: {e}")
            return False
    
    def read_alerts_json(
        self,
        cow_id: Optional[str] = None,
        date: Optional[str] = None,
        max_alerts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read alerts from JSON log files.
        
        Args:
            cow_id: Filter by cow ID (None for all)
            date: Filter by date in YYYYMMDD format (None for all)
            max_alerts: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        try:
            # Build file pattern
            if cow_id and date:
                pattern = f"alerts_{cow_id}_{date}.json"
            elif cow_id:
                pattern = f"alerts_{cow_id}_*.json"
            elif date:
                pattern = f"alerts_*_{date}.json"
            else:
                pattern = "alerts_*.json"
            
            # Read matching files
            for filepath in sorted(self.log_dir.glob(pattern)):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            alert = json.loads(line.strip())
                            alerts.append(alert)
                            
                            if max_alerts and len(alerts) >= max_alerts:
                                return alerts
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON line in {filepath}: {e}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error reading JSON logs: {e}")
            return alerts
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about alert logs.
        
        Returns:
            Dictionary with log statistics
        """
        stats = {
            'total_json_files': 0,
            'total_csv_files': 0,
            'total_alerts': 0,
            'oldest_log_date': None,
            'newest_log_date': None,
            'disk_usage_mb': 0.0,
        }
        
        try:
            json_files = list(self.log_dir.glob("alerts_*.json"))
            csv_files = list(self.log_dir.glob("alerts_*.csv"))
            
            stats['total_json_files'] = len(json_files)
            stats['total_csv_files'] = len(csv_files)
            
            # Calculate disk usage
            total_bytes = 0
            for filepath in json_files + csv_files:
                total_bytes += filepath.stat().st_size
            stats['disk_usage_mb'] = total_bytes / (1024 * 1024)
            
            # Count total alerts from JSON files
            alert_count = 0
            for filepath in json_files:
                with open(filepath, 'r', encoding='utf-8') as f:
                    alert_count += sum(1 for _ in f)
            stats['total_alerts'] = alert_count
            
            # Get date range
            if json_files:
                dates = []
                for filepath in json_files:
                    try:
                        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                        dates.append(mtime)
                    except:
                        pass
                
                if dates:
                    stats['oldest_log_date'] = min(dates).isoformat()
                    stats['newest_log_date'] = max(dates).isoformat()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting log statistics: {e}")
            return stats
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AlertLogger(dir={self.log_dir}, "
            f"retention={self.retention_days} days)"
        )
