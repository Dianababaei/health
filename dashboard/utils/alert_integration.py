"""
Alert Integration Utilities

Provides integration between the legacy alert logging system and the new
AlertSystem with state management. Converts alerts from JSON logs into
AlertState objects.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import uuid

from src.health_intelligence.alert_system import AlertSystem, AlertState

logger = logging.getLogger(__name__)


class AlertIntegration:
    """
    Integrates legacy alert logs with new AlertSystem.
    
    Converts alerts from malfunction_alerts.json (AlertGenerator format)
    to AlertState objects managed by AlertSystem.
    """
    
    def __init__(
        self,
        alert_system: AlertSystem,
        legacy_log_path: str = "logs/malfunction_alerts.json",
    ):
        """
        Initialize alert integration.
        
        Args:
            alert_system: AlertSystem instance
            legacy_log_path: Path to legacy alert JSON log
        """
        self.alert_system = alert_system
        self.legacy_log_path = Path(legacy_log_path)
    
    def import_legacy_alerts(
        self,
        max_alerts: Optional[int] = None,
        only_recent_hours: Optional[int] = 24,
    ) -> int:
        """
        Import alerts from legacy JSON log into AlertSystem.
        
        Args:
            max_alerts: Maximum number of alerts to import
            only_recent_hours: Only import alerts from last N hours
            
        Returns:
            Number of alerts imported
        """
        if not self.legacy_log_path.exists():
            logger.warning(f"Legacy alert log not found: {self.legacy_log_path}")
            return 0
        
        imported_count = 0
        
        try:
            with open(self.legacy_log_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if max_alerts and imported_count >= max_alerts:
                        break
                    
                    try:
                        legacy_alert = json.loads(line.strip())
                        
                        # Convert legacy alert to AlertState
                        alert_state = self._convert_legacy_alert(legacy_alert)
                        
                        if alert_state is None:
                            continue
                        
                        # Filter by time if specified
                        if only_recent_hours:
                            cutoff = datetime.now().timestamp() - (only_recent_hours * 3600)
                            if alert_state.timestamp.timestamp() < cutoff:
                                continue
                        
                        # Check if alert already exists
                        if alert_state.alert_id not in self.alert_system.alerts:
                            # Add to alert system
                            self.alert_system.alerts[alert_state.alert_id] = alert_state
                            imported_count += 1
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing alert line {line_num}: {e}")
                    except Exception as e:
                        logger.error(f"Error converting alert line {line_num}: {e}")
            
            # Save imported alerts
            if imported_count > 0:
                self.alert_system._save_states()
            
            logger.info(f"Imported {imported_count} legacy alerts")
            
        except Exception as e:
            logger.error(f"Error importing legacy alerts: {e}")
        
        return imported_count
    
    def _convert_legacy_alert(self, legacy_alert: Dict[str, Any]) -> Optional[AlertState]:
        """
        Convert legacy alert format to AlertState.
        
        Legacy format (from malfunction detection):
        {
            "detection_time": "2025-01-08T14:23:00",
            "malfunction_type": "stuck_values",
            "severity": "warning",
            "affected_sensors": ["sensor_001"],
            "confidence": 0.85,
            "message": "...",
            ...
        }
        
        Args:
            legacy_alert: Legacy alert dictionary
            
        Returns:
            AlertState object or None if conversion fails
        """
        try:
            # Extract fields
            detection_time_str = legacy_alert.get('detection_time')
            malfunction_type = legacy_alert.get('malfunction_type', 'unknown')
            severity = legacy_alert.get('severity', 'warning')
            affected_sensors = legacy_alert.get('affected_sensors', [])
            confidence = legacy_alert.get('confidence', 0.0)
            message = legacy_alert.get('message', '')
            
            # Parse timestamp
            if detection_time_str:
                try:
                    timestamp = datetime.fromisoformat(detection_time_str)
                except:
                    timestamp = datetime.now()
            else:
                timestamp = datetime.now()
            
            # Generate alert ID (use hash of detection time and type)
            alert_id = f"{malfunction_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            
            # Determine cow ID (use first sensor if available)
            cow_id = affected_sensors[0] if affected_sensors else "unknown"
            
            # Map severity to priority
            priority_mapping = {
                'critical': 'critical',
                'high': 'critical',
                'medium': 'warning',
                'low': 'warning',
                'warning': 'warning',
                'info': 'info',
            }
            priority = priority_mapping.get(severity.lower(), 'warning')
            
            # Map malfunction type to alert type
            alert_type_mapping = {
                'stuck_values': 'sensor_malfunction',
                'connectivity_loss': 'sensor_malfunction',
                'out_of_range': 'sensor_malfunction',
                'data_gap': 'sensor_malfunction',
            }
            alert_type = alert_type_mapping.get(malfunction_type, malfunction_type)
            
            # Build sensor values
            sensor_values = {
                'affected_sensors': affected_sensors,
                'confidence': confidence,
            }
            
            # Add any additional sensor data from legacy alert
            for key in ['temperature', 'fxa', 'mya', 'rza', 'motion_magnitude']:
                if key in legacy_alert:
                    sensor_values[key] = legacy_alert[key]
            
            # Get recommended actions
            recommended_actions = self.alert_system.get_recommended_actions(alert_type)
            
            # Build description
            description = message or f"{malfunction_type.replace('_', ' ').title()} detected"
            
            # Create metadata
            metadata = {
                'source': 'legacy_import',
                'original_severity': severity,
                'confidence': confidence,
            }
            
            # Create AlertState
            alert_state = AlertState(
                alert_id=alert_id,
                timestamp=timestamp,
                cow_id=cow_id,
                alert_type=alert_type,
                priority=priority,
                status='active',  # Assume all legacy alerts are active
                sensor_values=sensor_values,
                description=description,
                recommended_actions=recommended_actions,
                metadata=metadata,
            )
            
            return alert_state
            
        except Exception as e:
            logger.error(f"Error converting legacy alert: {e}")
            return None


def sync_legacy_alerts(
    alert_system: Optional[AlertSystem] = None,
    auto_import: bool = True,
) -> AlertSystem:
    """
    Convenience function to sync legacy alerts with AlertSystem.
    
    Args:
        alert_system: Existing AlertSystem (creates new if None)
        auto_import: Automatically import legacy alerts
        
    Returns:
        AlertSystem instance
    """
    if alert_system is None:
        alert_system = AlertSystem()
    
    if auto_import:
        integration = AlertIntegration(alert_system)
        imported = integration.import_legacy_alerts(
            only_recent_hours=168  # Import last 7 days
        )
        logger.info(f"Synced {imported} legacy alerts")
    
    return alert_system
