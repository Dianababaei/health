"""
Alert Generator Module

Provides alert queue management, notification handling, and logging
for malfunction detection alerts.
"""

import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from queue import Queue, Empty
from pathlib import Path


logger = logging.getLogger(__name__)


class AlertGenerator:
    """
    Manages alert generation, queuing, and logging.
    
    Features:
    - In-memory queue for dashboard notifications
    - JSON logging to file
    - Callback support for real-time notifications
    - Alert statistics tracking
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        alert_log_file: str = "malfunction_alerts.json",
        max_queue_size: int = 1000,
    ):
        """
        Initialize alert generator.
        
        Args:
            log_dir: Directory for alert logs
            alert_log_file: Name of JSON log file
            max_queue_size: Maximum size of in-memory alert queue
        """
        self.log_dir = Path(log_dir)
        self.alert_log_file = self.log_dir / alert_log_file
        self.max_queue_size = max_queue_size
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory queue for dashboard
        self.alert_queue: Queue = Queue(maxsize=max_queue_size)
        
        # Callback for real-time notifications
        self.notification_callback: Optional[Callable] = None
        
        # Statistics
        self.total_alerts_generated = 0
        self.alerts_by_type: Dict[str, int] = {}
        self.alerts_by_severity: Dict[str, int] = {}
        
        logger.info(f"AlertGenerator initialized: log_file={self.alert_log_file}")
    
    def generate_alerts(self, alerts: List[Any]) -> int:
        """
        Generate alerts from malfunction detection.
        
        Args:
            alerts: List of MalfunctionAlert objects
            
        Returns:
            Number of alerts generated
        """
        if not alerts:
            return 0
        
        generated_count = 0
        
        for alert in alerts:
            # Convert to dict for processing
            alert_dict = alert.to_dict()
            
            # Add to queue
            self._add_to_queue(alert_dict)
            
            # Log to JSON file
            self._log_to_json(alert_dict)
            
            # Call notification callback if set
            if self.notification_callback:
                try:
                    self.notification_callback(alert_dict)
                except Exception as e:
                    logger.error(f"Error in notification callback: {e}")
            
            # Update statistics
            self._update_statistics(alert_dict)
            
            generated_count += 1
            
            logger.info(
                f"Alert generated: {alert_dict['malfunction_type']} "
                f"({alert_dict['severity']}) - {alert_dict['affected_sensors']}"
            )
        
        return generated_count
    
    def _add_to_queue(self, alert_dict: Dict[str, Any]):
        """Add alert to in-memory queue."""
        try:
            if self.alert_queue.full():
                # Remove oldest alert if queue is full
                try:
                    self.alert_queue.get_nowait()
                    logger.warning("Alert queue full, removed oldest alert")
                except Empty:
                    pass
            
            self.alert_queue.put_nowait(alert_dict)
        except Exception as e:
            logger.error(f"Error adding alert to queue: {e}")
    
    def _log_to_json(self, alert_dict: Dict[str, Any]):
        """Log alert to JSON file."""
        try:
            # Append to JSON log file (one JSON object per line)
            with open(self.alert_log_file, 'a') as f:
                json.dump(alert_dict, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Error logging alert to JSON: {e}")
    
    def _update_statistics(self, alert_dict: Dict[str, Any]):
        """Update alert statistics."""
        self.total_alerts_generated += 1
        
        malfunction_type = alert_dict.get('malfunction_type', 'unknown')
        self.alerts_by_type[malfunction_type] = self.alerts_by_type.get(malfunction_type, 0) + 1
        
        severity = alert_dict.get('severity', 'unknown')
        self.alerts_by_severity[severity] = self.alerts_by_severity.get(severity, 0) + 1
    
    def get_queued_alerts(self, max_count: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get alerts from queue without removing them.
        
        Args:
            max_count: Maximum number of alerts to return (None for all)
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        temp_queue = Queue(maxsize=self.max_queue_size)
        
        # Extract alerts from queue
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
                temp_queue.put_nowait(alert)
                
                if max_count and len(alerts) >= max_count:
                    break
            except Empty:
                break
        
        # Restore queue
        self.alert_queue = temp_queue
        
        return alerts
    
    def pop_alert(self) -> Optional[Dict[str, Any]]:
        """
        Remove and return next alert from queue.
        
        Returns:
            Alert dictionary or None if queue is empty
        """
        try:
            return self.alert_queue.get_nowait()
        except Empty:
            return None
    
    def clear_queue(self):
        """Clear all alerts from queue."""
        while not self.alert_queue.empty():
            try:
                self.alert_queue.get_nowait()
            except Empty:
                break
        logger.info("Alert queue cleared")
    
    def set_notification_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Set callback function for real-time notifications.
        
        Args:
            callback: Function to call with alert dictionary
        """
        self.notification_callback = callback
        logger.info("Notification callback registered")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        return {
            'total_alerts_generated': self.total_alerts_generated,
            'alerts_by_type': self.alerts_by_type.copy(),
            'alerts_by_severity': self.alerts_by_severity.copy(),
            'queue_size': self.alert_queue.qsize(),
            'max_queue_size': self.max_queue_size,
        }
    
    def read_alert_log(
        self,
        max_lines: Optional[int] = None,
        filter_type: Optional[str] = None,
        filter_severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Read alerts from JSON log file.
        
        Args:
            max_lines: Maximum number of alerts to read (None for all)
            filter_type: Filter by malfunction type
            filter_severity: Filter by severity
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        if not self.alert_log_file.exists():
            logger.warning(f"Alert log file not found: {self.alert_log_file}")
            return alerts
        
        try:
            with open(self.alert_log_file, 'r') as f:
                for line_num, line in enumerate(f):
                    if max_lines and line_num >= max_lines:
                        break
                    
                    try:
                        alert = json.loads(line.strip())
                        
                        # Apply filters
                        if filter_type and alert.get('malfunction_type') != filter_type:
                            continue
                        if filter_severity and alert.get('severity') != filter_severity:
                            continue
                        
                        alerts.append(alert)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing alert line {line_num}: {e}")
        
        except Exception as e:
            logger.error(f"Error reading alert log: {e}")
        
        return alerts
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"AlertGenerator(total_alerts={self.total_alerts_generated}, "
            f"queue_size={self.alert_queue.qsize()})"
        )
