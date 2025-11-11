"""
Alert System Module - State Management and Recommended Actions

Provides alert state management, acknowledgment tracking, and recommended
actions for different alert types in the Artemis Health system.

Features:
- Alert state tracking (active/acknowledged/resolved)
- Acknowledgment logging with timestamp and user
- Recommended actions mapping for each alert type
- Alert persistence to JSON log files
- Query and filtering capabilities
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum

logger = logging.getLogger(__name__)


class AlertStatus(Enum):
    """Alert status enumeration."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertPriority(Enum):
    """Alert priority levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class AlertState:
    """
    Alert state data structure.
    
    Attributes:
        alert_id: Unique alert identifier
        timestamp: Alert creation time
        cow_id: Animal identifier
        alert_type: Type of alert
        priority: Alert priority (critical/warning/info)
        status: Current status (active/acknowledged/resolved)
        sensor_values: Relevant sensor readings
        description: Alert description
        recommended_actions: List of recommended actions
        acknowledged_by: User who acknowledged (if applicable)
        acknowledged_at: Acknowledgment timestamp (if applicable)
        resolved_at: Resolution timestamp (if applicable)
        metadata: Additional alert metadata
    """
    alert_id: str
    timestamp: datetime
    cow_id: str
    alert_type: str
    priority: str
    status: str
    sensor_values: Dict[str, Any]
    description: str
    recommended_actions: List[str]
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert state to dictionary."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat() if self.timestamp else None
        d['acknowledged_at'] = self.acknowledged_at.isoformat() if self.acknowledged_at else None
        d['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertState':
        """Create AlertState from dictionary."""
        # Convert ISO format strings back to datetime
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data.get('acknowledged_at'), str):
            data['acknowledged_at'] = datetime.fromisoformat(data['acknowledged_at'])
        if isinstance(data.get('resolved_at'), str):
            data['resolved_at'] = datetime.fromisoformat(data['resolved_at'])
        return cls(**data)


class AlertSystem:
    """
    Alert state management and recommended actions system.
    
    Features:
    - Track alert states (active/acknowledged/resolved)
    - Provide recommended actions for each alert type
    - Persist alert states to JSON
    - Query and filter alerts
    - Acknowledgment tracking
    """
    
    # Recommended actions mapping for each alert type
    RECOMMENDED_ACTIONS = {
        'fever': [
            "Immediately isolate the animal from the herd",
            "Check rectal temperature with thermometer for confirmation",
            "Contact veterinarian for examination",
            "Ensure access to fresh water and shade",
            "Monitor temperature every 2-4 hours",
            "Document any additional symptoms (nasal discharge, coughing, lethargy)",
        ],
        'heat_stress': [
            "Move animal to shaded area immediately",
            "Provide cool, fresh water access",
            "Use fans or misters if available",
            "Reduce activity and handling",
            "Monitor temperature every 30 minutes",
            "If temperature exceeds 40.5°C, contact veterinarian urgently",
        ],
        'inactivity': [
            "Visually inspect the animal for signs of distress",
            "Check for injuries, lameness, or bloating",
            "Assess appetite and water intake",
            "Monitor rumination activity",
            "If lying for >8 hours, contact veterinarian (downer cow risk)",
            "Provide comfortable bedding and check for environmental stressors",
        ],
        'sensor_malfunction': [
            "Check sensor battery level and connection",
            "Inspect collar for damage or looseness",
            "Verify sensor placement on neck",
            "Reboot sensor device if possible",
            "Replace sensor if malfunction persists",
            "Document sensor ID and issue for maintenance tracking",
        ],
        'estrus': [
            "Observe for behavioral signs of estrus (mounting, restlessness)",
            "Confirm with secondary indicators (mucus discharge, vocalization)",
            "Schedule breeding or AI within 12-18 hours of detection",
            "Record estrus detection for reproductive tracking",
            "Monitor for successful breeding confirmation",
            "Track for pregnancy indication in 21-28 days",
        ],
        'pregnancy_indication': [
            "Schedule veterinary confirmation (ultrasound or palpation)",
            "Monitor for return to estrus (negative indicator)",
            "Adjust nutrition for early pregnancy needs",
            "Continue monitoring for pregnancy maintenance",
            "Record expected calving date (approximately 280 days)",
            "Implement pre-calving management protocols at day 240+",
        ],
        'sensor_reconnected': [
            "Verify sensor is transmitting normal data",
            "Check data quality for gaps or anomalies",
            "Monitor animal for any behavioral changes during disconnection",
            "Update sensor maintenance log",
        ],
        'default': [
            "Review alert details and sensor readings",
            "Visually inspect the animal",
            "Contact veterinarian if condition persists or worsens",
            "Document observations for health record",
        ],
    }
    
    # Alert priority mapping based on type and conditions
    PRIORITY_MAPPING = {
        'fever': AlertPriority.CRITICAL,
        'heat_stress': AlertPriority.CRITICAL,
        'inactivity_critical': AlertPriority.CRITICAL,  # >8 hours or with fever
        'inactivity_warning': AlertPriority.WARNING,   # 4-8 hours
        'sensor_malfunction_critical': AlertPriority.CRITICAL,  # >30 min no data
        'sensor_malfunction_warning': AlertPriority.WARNING,   # 5-30 min no data
        'estrus': AlertPriority.WARNING,
        'pregnancy_indication': AlertPriority.WARNING,
        'sensor_reconnected': AlertPriority.INFO,
        'system_notification': AlertPriority.INFO,
    }
    
    def __init__(self, state_file: str = "logs/alerts/alert_states.json"):
        """
        Initialize alert system.
        
        Args:
            state_file: Path to alert state persistence file
        """
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # In-memory alert states
        self.alerts: Dict[str, AlertState] = {}
        
        # Load existing states
        self._load_states()
        
        logger.info(f"AlertSystem initialized with {len(self.alerts)} existing alerts")
    
    def create_alert(
        self,
        alert_id: str,
        timestamp: datetime,
        cow_id: str,
        alert_type: str,
        sensor_values: Dict[str, Any],
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AlertState:
        """
        Create a new alert with recommended actions.
        
        Args:
            alert_id: Unique alert identifier
            timestamp: Alert creation time
            cow_id: Animal identifier
            alert_type: Type of alert
            sensor_values: Relevant sensor readings
            description: Alert description
            metadata: Additional metadata
            
        Returns:
            Created AlertState object
        """
        # Determine priority
        priority = self._determine_priority(alert_type, metadata)
        
        # Get recommended actions
        recommended_actions = self.get_recommended_actions(alert_type)
        
        # Create alert state
        alert_state = AlertState(
            alert_id=alert_id,
            timestamp=timestamp,
            cow_id=cow_id,
            alert_type=alert_type,
            priority=priority.value,
            status=AlertStatus.ACTIVE.value,
            sensor_values=sensor_values,
            description=description,
            recommended_actions=recommended_actions,
            metadata=metadata or {},
        )
        
        # Store in memory
        self.alerts[alert_id] = alert_state
        
        # Persist to file
        self._save_states()
        
        logger.info(f"Created alert {alert_id}: {alert_type} ({priority.value}) for cow {cow_id}")
        
        return alert_state
    
    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = "dashboard_user",
    ) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User who acknowledged
            
        Returns:
            True if successful, False otherwise
        """
        if alert_id not in self.alerts:
            logger.warning(f"Alert {alert_id} not found for acknowledgment")
            return False
        
        alert = self.alerts[alert_id]
        
        if alert.status == AlertStatus.RESOLVED.value:
            logger.warning(f"Cannot acknowledge resolved alert {alert_id}")
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED.value
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        # Persist changes
        self._save_states()
        
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        
        return True
    
    def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert identifier
            resolution_notes: Optional notes about resolution
            
        Returns:
            True if successful, False otherwise
        """
        if alert_id not in self.alerts:
            logger.warning(f"Alert {alert_id} not found for resolution")
            return False
        
        alert = self.alerts[alert_id]
        alert.status = AlertStatus.RESOLVED.value
        alert.resolved_at = datetime.now()
        
        if resolution_notes:
            alert.metadata['resolution_notes'] = resolution_notes
        
        # Persist changes
        self._save_states()
        
        logger.info(f"Alert {alert_id} resolved")
        
        return True
    
    def get_alert(self, alert_id: str) -> Optional[AlertState]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)
    
    def get_active_alerts(self, cow_id: Optional[str] = None) -> List[AlertState]:
        """
        Get all active alerts.
        
        Args:
            cow_id: Filter by cow ID (optional)
            
        Returns:
            List of active alerts
        """
        alerts = [
            alert for alert in self.alerts.values()
            if alert.status == AlertStatus.ACTIVE.value
        ]
        
        if cow_id:
            alerts = [alert for alert in alerts if alert.cow_id == cow_id]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alerts_by_status(
        self,
        status: str,
        cow_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AlertState]:
        """
        Get alerts by status.
        
        Args:
            status: Alert status (active/acknowledged/resolved)
            cow_id: Filter by cow ID (optional)
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        alerts = [
            alert for alert in self.alerts.values()
            if alert.status == status
        ]
        
        if cow_id:
            alerts = [alert for alert in alerts if alert.cow_id == cow_id]
        
        alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def get_alerts_by_priority(
        self,
        priority: str,
        include_resolved: bool = False,
    ) -> List[AlertState]:
        """
        Get alerts by priority level.
        
        Args:
            priority: Priority level (critical/warning/info)
            include_resolved: Include resolved alerts
            
        Returns:
            List of alerts
        """
        alerts = [
            alert for alert in self.alerts.values()
            if alert.priority == priority
        ]
        
        if not include_resolved:
            alerts = [
                alert for alert in alerts
                if alert.status != AlertStatus.RESOLVED.value
            ]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_recommended_actions(self, alert_type: str) -> List[str]:
        """
        Get recommended actions for an alert type.
        
        Args:
            alert_type: Type of alert
            
        Returns:
            List of recommended action strings
        """
        return self.RECOMMENDED_ACTIONS.get(alert_type, self.RECOMMENDED_ACTIONS['default'])
    
    def _determine_priority(
        self,
        alert_type: str,
        metadata: Optional[Dict[str, Any]],
    ) -> AlertPriority:
        """Determine alert priority based on type and metadata."""
        # Handle inactivity with duration-based priority
        if alert_type == 'inactivity':
            duration_hours = metadata.get('duration_hours', 0) if metadata else 0
            if duration_hours >= 8:
                return AlertPriority.CRITICAL
            else:
                return AlertPriority.WARNING
        
        # Handle sensor malfunction with duration-based priority
        if alert_type == 'sensor_malfunction':
            gap_minutes = metadata.get('gap_minutes', 0) if metadata else 0
            if gap_minutes >= 30:
                return AlertPriority.CRITICAL
            else:
                return AlertPriority.WARNING
        
        # Use predefined mapping
        return self.PRIORITY_MAPPING.get(alert_type, AlertPriority.WARNING)
    
    def _load_states(self):
        """Load alert states from JSON file."""
        if not self.state_file.exists():
            logger.info("No existing alert state file found")
            return
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
                
            for alert_id, alert_data in data.items():
                try:
                    self.alerts[alert_id] = AlertState.from_dict(alert_data)
                except Exception as e:
                    logger.error(f"Error loading alert {alert_id}: {e}")
            
            logger.info(f"Loaded {len(self.alerts)} alert states from {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error loading alert states: {e}")
    
    def _save_states(self):
        """Save alert states to JSON file."""
        try:
            data = {
                alert_id: alert.to_dict()
                for alert_id, alert in self.alerts.items()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.alerts)} alert states to {self.state_file}")
            
        except Exception as e:
            logger.error(f"Error saving alert states: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with alert statistics
        """
        active_alerts = [a for a in self.alerts.values() if a.status == AlertStatus.ACTIVE.value]
        acknowledged_alerts = [a for a in self.alerts.values() if a.status == AlertStatus.ACKNOWLEDGED.value]
        resolved_alerts = [a for a in self.alerts.values() if a.status == AlertStatus.RESOLVED.value]
        
        # Count by priority
        priority_counts = {
            'critical': len([a for a in active_alerts if a.priority == AlertPriority.CRITICAL.value]),
            'warning': len([a for a in active_alerts if a.priority == AlertPriority.WARNING.value]),
            'info': len([a for a in active_alerts if a.priority == AlertPriority.INFO.value]),
        }
        
        # Count by type
        type_counts = {}
        for alert in active_alerts:
            type_counts[alert.alert_type] = type_counts.get(alert.alert_type, 0) + 1
        
        return {
            'total_alerts': len(self.alerts),
            'active': len(active_alerts),
            'acknowledged': len(acknowledged_alerts),
            'resolved': len(resolved_alerts),
            'by_priority': priority_counts,
            'by_type': type_counts,
        }
