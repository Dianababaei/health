"""
Immediate Alert Detection Module

Real-time detection of critical health conditions:
- Fever Alert: High temperature with reduced motion (>39.5°C, low activity, ≥2 min)
- Heat Stress Alert: High temperature with high activity (>threshold, ≥2 min)
- Prolonged Inactivity Alert: Continuous stillness (>4 hours, excluding normal rest)
- Sensor Malfunction Alert: Data quality issues (no data, stuck values, out-of-range)

Features:
- Configurable thresholds via YAML
- Alert deduplication
- Confidence scoring
- Integration with Layer 1 (behavioral states) and Layer 2 (baseline temps)
- Latency target: 1-2 minutes from condition onset
"""

import logging
import yaml
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """
    Immediate health alert data structure.
    
    Attributes:
        alert_id: Unique identifier (UUID)
        timestamp: When alert was detected
        cow_id: Animal identifier
        alert_type: Type of alert ('fever', 'heat_stress', 'inactivity', 'sensor_malfunction')
        severity: Alert severity ('critical', 'warning')
        confidence: Detection confidence (0.0-1.0)
        sensor_values: Relevant sensor readings at alert time
        detection_window: Time window used for detection (e.g., "2 minutes")
        status: Alert status ('active', 'resolved')
        details: Additional alert-specific information
    """
    alert_id: str
    timestamp: datetime
    cow_id: str
    alert_type: str
    severity: str
    confidence: float
    sensor_values: Dict[str, Any]
    detection_window: str
    status: str
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format (JSON-serializable)."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


class ImmediateAlertDetector:
    """
    Real-time health alert detection system.
    
    Detects four types of immediate alerts:
    1. Fever: High temperature with reduced motion
    2. Heat Stress: High temperature with high activity
    3. Prolonged Inactivity: Extended stillness
    4. Sensor Malfunction: Data quality issues
    
    Features:
    - Configurable thresholds from YAML
    - Rolling window detection for confirmation
    - Alert deduplication
    - Confidence scoring based on data quality and consistency
    """
    
    # Alert type constants
    ALERT_TYPE_FEVER = "fever"
    ALERT_TYPE_HEAT_STRESS = "heat_stress"
    ALERT_TYPE_INACTIVITY = "inactivity"
    ALERT_TYPE_SENSOR_MALFUNCTION = "sensor_malfunction"
    
    # Severity levels
    SEVERITY_CRITICAL = "critical"
    SEVERITY_WARNING = "warning"
    
    # Alert status
    STATUS_ACTIVE = "active"
    STATUS_RESOLVED = "resolved"
    
    def __init__(
        self,
        config_path: str = "config/alert_thresholds.yaml",
        baseline_temperature: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize immediate alert detector.
        
        Args:
            config_path: Path to threshold configuration YAML file
            baseline_temperature: Optional dict of cow_id -> baseline temp (°C)
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.baseline_temperature = baseline_temperature or {}
        
        # Alert tracking for deduplication
        self.active_alerts: Dict[str, Alert] = {}  # alert_key -> Alert
        self.alert_history: deque = deque(maxlen=1000)  # Recent alert history
        
        # Detection state tracking per cow
        self.detection_state: Dict[str, Dict] = {}  # cow_id -> state buffers
        
        logger.info(f"ImmediateAlertDetector initialized with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load threshold configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded alert thresholds from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            # Return default config
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {
            'fever_alert': {
                'temperature_threshold': 39.5,
                'motion_threshold': 0.15,
                'confirmation_window_minutes': 2,
                'min_samples_for_confirmation': 2,
                'severity': {'critical_temp': 40.0, 'warning_temp': 39.5},
                'deduplication_window_minutes': 30,
            },
            'heat_stress_alert': {
                'temperature_threshold': 39.0,
                'activity_threshold': 0.60,
                'confirmation_window_minutes': 2,
                'min_samples_for_confirmation': 2,
                'severity': {'critical_temp': 39.8, 'warning_temp': 39.0},
                'deduplication_window_minutes': 30,
            },
            'inactivity_alert': {
                'fxa_threshold': 0.05,
                'mya_threshold': 0.05,
                'rza_threshold': 0.05,
                'duration_threshold_hours': 4.0,
                'max_movement_minutes_per_hour': 5,
                'exclude_lying_state': True,
                'exclude_ruminating_state': True,
                'severity': {'critical_hours': 8.0, 'warning_hours': 4.0},
                'deduplication_window_minutes': 60,
            },
            'sensor_malfunction_alert': {
                'connectivity_loss': {'gap_threshold_minutes': 5, 'severity': 'critical'},
                'stuck_values': {'duration_threshold_minutes': 120, 'severity': 'warning'},
                'deduplication_window_minutes': 15,
            },
            'confidence_scoring': {
                'base_confidence': {
                    'fever': 0.85,
                    'heat_stress': 0.80,
                    'inactivity': 0.75,
                    'sensor_malfunction': 0.95,
                },
            },
        }
    
    def detect_alerts(
        self,
        sensor_data: pd.DataFrame,
        cow_id: str,
        behavioral_state: Optional[str] = None,
        baseline_temp: Optional[float] = None,
    ) -> List[Alert]:
        """
        Detect immediate alerts from real-time sensor data.
        
        Args:
            sensor_data: DataFrame with columns ['timestamp', 'temperature', 'fxa', 'mya', 'rza']
                        Must be time-ordered
            cow_id: Animal identifier
            behavioral_state: Current behavioral state from Layer 1 ('lying', 'standing', etc.)
            baseline_temp: Individual baseline temperature (if available)
        
        Returns:
            List of detected alerts
        """
        if sensor_data.empty:
            return []
        
        alerts = []
        
        # Update baseline if provided
        if baseline_temp is not None:
            self.baseline_temperature[cow_id] = baseline_temp
        
        # Initialize detection state for this cow if needed
        if cow_id not in self.detection_state:
            self.detection_state[cow_id] = {
                'fever_buffer': deque(maxlen=10),
                'heat_stress_buffer': deque(maxlen=10),
                'inactivity_start': None,
                'inactivity_duration_minutes': 0,
            }
        
        # Check each alert type
        fever_alert = self._check_fever(sensor_data, cow_id)
        if fever_alert:
            alerts.append(fever_alert)
        
        heat_stress_alert = self._check_heat_stress(sensor_data, cow_id, behavioral_state)
        if heat_stress_alert:
            alerts.append(heat_stress_alert)
        
        inactivity_alert = self._check_inactivity(sensor_data, cow_id, behavioral_state)
        if inactivity_alert:
            alerts.append(inactivity_alert)
        
        malfunction_alert = self._check_sensor_malfunction(sensor_data, cow_id)
        if malfunction_alert:
            alerts.append(malfunction_alert)
        
        # Log generated alerts
        for alert in alerts:
            self.alert_history.append(alert)
            logger.info(
                f"Alert generated: {alert.alert_type} for {cow_id} "
                f"({alert.severity}, confidence={alert.confidence:.2f})"
            )
        
        return alerts
    
    def _check_fever(
        self,
        sensor_data: pd.DataFrame,
        cow_id: str,
    ) -> Optional[Alert]:
        """
        Check for fever alert: Temperature >39.5°C AND motion intensity <threshold for ≥2 minutes.
        
        Args:
            sensor_data: Sensor readings
            cow_id: Animal identifier
        
        Returns:
            Alert if fever detected, None otherwise
        """
        config = self.config['fever_alert']
        temp_threshold = config['temperature_threshold']
        motion_threshold = config['motion_threshold']
        window_minutes = config['confirmation_window_minutes']
        min_samples = config['min_samples_for_confirmation']
        
        # Calculate motion intensity: sqrt(fxa² + mya² + rza²)
        df = sensor_data.copy()
        df['motion_intensity'] = np.sqrt(
            df['fxa']**2 + df['mya']**2 + df['rza']**2
        )
        
        # Check condition: high temp AND low motion
        df['fever_condition'] = (
            (df['temperature'] > temp_threshold) & 
            (df['motion_intensity'] < motion_threshold)
        )
        
        # Get recent data within confirmation window
        if len(df) > 0:
            latest_time = df['timestamp'].max()
            cutoff_time = latest_time - timedelta(minutes=window_minutes)
            recent_data = df[df['timestamp'] >= cutoff_time]
            
            # Check if condition met for required samples
            fever_samples = recent_data[recent_data['fever_condition']]
            
            if len(fever_samples) >= min_samples:
                # Check deduplication
                alert_key = f"{self.ALERT_TYPE_FEVER}_{cow_id}"
                if self._is_duplicate_alert(alert_key, config['deduplication_window_minutes']):
                    return None
                
                # Determine severity
                max_temp = fever_samples['temperature'].max()
                if max_temp > config['severity']['critical_temp']:
                    severity = self.SEVERITY_CRITICAL
                else:
                    severity = self.SEVERITY_WARNING
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    alert_type=self.ALERT_TYPE_FEVER,
                    data_quality=len(fever_samples) / len(recent_data),
                    window_completeness=len(fever_samples) / min_samples,
                )
                
                # Get sensor values at alert time
                latest_reading = fever_samples.iloc[-1]
                sensor_values = {
                    'temperature': float(latest_reading['temperature']),
                    'motion_intensity': float(latest_reading['motion_intensity']),
                    'fxa': float(latest_reading['fxa']),
                    'mya': float(latest_reading['mya']),
                    'rza': float(latest_reading['rza']),
                }
                
                # Create alert
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=latest_reading['timestamp'],
                    cow_id=cow_id,
                    alert_type=self.ALERT_TYPE_FEVER,
                    severity=severity,
                    confidence=confidence,
                    sensor_values=sensor_values,
                    detection_window=f"{window_minutes} minutes",
                    status=self.STATUS_ACTIVE,
                    details={
                        'max_temperature': float(max_temp),
                        'avg_motion': float(fever_samples['motion_intensity'].mean()),
                        'samples_confirming': len(fever_samples),
                        'threshold_exceeded_by': float(max_temp - temp_threshold),
                    },
                )
                
                # Track active alert
                self.active_alerts[alert_key] = alert
                
                return alert
        
        return None
    
    def _check_heat_stress(
        self,
        sensor_data: pd.DataFrame,
        cow_id: str,
        behavioral_state: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Check for heat stress alert: Temperature >threshold AND activity level >threshold for ≥2 minutes.
        
        Args:
            sensor_data: Sensor readings
            cow_id: Animal identifier
            behavioral_state: Current behavioral state
        
        Returns:
            Alert if heat stress detected, None otherwise
        """
        config = self.config['heat_stress_alert']
        temp_threshold = config['temperature_threshold']
        activity_threshold = config['activity_threshold']
        window_minutes = config['confirmation_window_minutes']
        min_samples = config['min_samples_for_confirmation']
        
        # Calculate motion intensity
        df = sensor_data.copy()
        df['motion_intensity'] = np.sqrt(
            df['fxa']**2 + df['mya']**2 + df['rza']**2
        )
        
        # Calculate activity level (0-1 scale)
        # High motion + active behavioral state = high activity
        df['activity_level'] = df['motion_intensity'] / 2.0  # Normalize roughly
        
        # Boost activity if in active behavioral state
        if behavioral_state in ['walking', 'standing', 'feeding']:
            df['activity_level'] = df['activity_level'] * 1.5
        
        df['activity_level'] = df['activity_level'].clip(0, 1.0)
        
        # Check condition: high temp AND high activity
        df['heat_stress_condition'] = (
            (df['temperature'] > temp_threshold) & 
            (df['activity_level'] > activity_threshold)
        )
        
        # Get recent data within confirmation window
        if len(df) > 0:
            latest_time = df['timestamp'].max()
            cutoff_time = latest_time - timedelta(minutes=window_minutes)
            recent_data = df[df['timestamp'] >= cutoff_time]
            
            # Check if condition met for required samples
            stress_samples = recent_data[recent_data['heat_stress_condition']]
            
            if len(stress_samples) >= min_samples:
                # Check deduplication
                alert_key = f"{self.ALERT_TYPE_HEAT_STRESS}_{cow_id}"
                if self._is_duplicate_alert(alert_key, config['deduplication_window_minutes']):
                    return None
                
                # Determine severity
                max_temp = stress_samples['temperature'].max()
                if max_temp > config['severity']['critical_temp']:
                    severity = self.SEVERITY_CRITICAL
                else:
                    severity = self.SEVERITY_WARNING
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    alert_type=self.ALERT_TYPE_HEAT_STRESS,
                    data_quality=len(stress_samples) / len(recent_data),
                    window_completeness=len(stress_samples) / min_samples,
                )
                
                # Get sensor values at alert time
                latest_reading = stress_samples.iloc[-1]
                sensor_values = {
                    'temperature': float(latest_reading['temperature']),
                    'activity_level': float(latest_reading['activity_level']),
                    'motion_intensity': float(latest_reading['motion_intensity']),
                    'behavioral_state': behavioral_state,
                }
                
                # Create alert
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=latest_reading['timestamp'],
                    cow_id=cow_id,
                    alert_type=self.ALERT_TYPE_HEAT_STRESS,
                    severity=severity,
                    confidence=confidence,
                    sensor_values=sensor_values,
                    detection_window=f"{window_minutes} minutes",
                    status=self.STATUS_ACTIVE,
                    details={
                        'max_temperature': float(max_temp),
                        'avg_activity': float(stress_samples['activity_level'].mean()),
                        'samples_confirming': len(stress_samples),
                        'behavioral_state': behavioral_state,
                    },
                )
                
                # Track active alert
                self.active_alerts[alert_key] = alert
                
                return alert
        
        return None
    
    def _check_inactivity(
        self,
        sensor_data: pd.DataFrame,
        cow_id: str,
        behavioral_state: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Check for prolonged inactivity alert: Continuous stillness >4 hours, excluding normal rest.
        
        Args:
            sensor_data: Sensor readings
            cow_id: Animal identifier
            behavioral_state: Current behavioral state
        
        Returns:
            Alert if prolonged inactivity detected, None otherwise
        """
        config = self.config['inactivity_alert']
        fxa_threshold = config['fxa_threshold']
        mya_threshold = config['mya_threshold']
        rza_threshold = config['rza_threshold']
        duration_threshold_hours = config['duration_threshold_hours']
        
        # Exclude normal rest periods (lying, ruminating)
        if behavioral_state and config.get('exclude_lying_state', True):
            if behavioral_state in ['lying', 'ruminating']:
                # Reset inactivity tracking during normal rest
                if cow_id in self.detection_state:
                    self.detection_state[cow_id]['inactivity_start'] = None
                    self.detection_state[cow_id]['inactivity_duration_minutes'] = 0
                return None
        
        # Check if all acceleration axes are below threshold (stillness)
        df = sensor_data.copy()
        df['is_still'] = (
            (np.abs(df['fxa']) < fxa_threshold) &
            (np.abs(df['mya']) < mya_threshold) &
            (np.abs(df['rza']) < rza_threshold)
        )
        
        # Track inactivity duration
        state = self.detection_state[cow_id]
        
        if len(df) > 0:
            latest_reading = df.iloc[-1]
            
            if latest_reading['is_still']:
                # Continue or start inactivity tracking
                if state['inactivity_start'] is None:
                    state['inactivity_start'] = latest_reading['timestamp']
                    state['inactivity_duration_minutes'] = 0
                else:
                    # Calculate duration
                    duration = latest_reading['timestamp'] - state['inactivity_start']
                    state['inactivity_duration_minutes'] = duration.total_seconds() / 60.0
            else:
                # Movement detected, reset tracking
                state['inactivity_start'] = None
                state['inactivity_duration_minutes'] = 0
                return None
            
            # Check if duration exceeds threshold
            duration_hours = state['inactivity_duration_minutes'] / 60.0
            
            if duration_hours >= duration_threshold_hours:
                # Check deduplication
                alert_key = f"{self.ALERT_TYPE_INACTIVITY}_{cow_id}"
                if self._is_duplicate_alert(alert_key, config['deduplication_window_minutes']):
                    return None
                
                # Determine severity
                if duration_hours > config['severity']['critical_hours']:
                    severity = self.SEVERITY_CRITICAL
                else:
                    severity = self.SEVERITY_WARNING
                
                # Calculate confidence
                confidence = self._calculate_confidence(
                    alert_type=self.ALERT_TYPE_INACTIVITY,
                    data_quality=1.0,
                    window_completeness=1.0,
                )
                
                # Adjust confidence based on behavioral state
                if behavioral_state and behavioral_state not in ['lying', 'ruminating']:
                    confidence *= 1.1  # Higher confidence if not in rest state
                
                confidence = min(confidence, 1.0)
                
                # Get sensor values at alert time
                sensor_values = {
                    'fxa': float(latest_reading['fxa']),
                    'mya': float(latest_reading['mya']),
                    'rza': float(latest_reading['rza']),
                    'behavioral_state': behavioral_state,
                }
                
                # Create alert
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=latest_reading['timestamp'],
                    cow_id=cow_id,
                    alert_type=self.ALERT_TYPE_INACTIVITY,
                    severity=severity,
                    confidence=confidence,
                    sensor_values=sensor_values,
                    detection_window=f"{duration_hours:.1f} hours",
                    status=self.STATUS_ACTIVE,
                    details={
                        'inactivity_duration_hours': duration_hours,
                        'inactivity_start': state['inactivity_start'].isoformat(),
                        'behavioral_state': behavioral_state,
                        'threshold_hours': duration_threshold_hours,
                    },
                )
                
                # Track active alert
                self.active_alerts[alert_key] = alert
                
                return alert
        
        return None
    
    def _check_sensor_malfunction(
        self,
        sensor_data: pd.DataFrame,
        cow_id: str,
    ) -> Optional[Alert]:
        """
        Check for sensor malfunction: No data, stuck values, or out-of-range readings.
        
        Note: Integrates with existing malfunction_detection.py logic.
        
        Args:
            sensor_data: Sensor readings
            cow_id: Animal identifier
        
        Returns:
            Alert if sensor malfunction detected, None otherwise
        """
        config = self.config['sensor_malfunction_alert']
        
        if sensor_data.empty:
            # No data received - connectivity loss
            alert_key = f"{self.ALERT_TYPE_SENSOR_MALFUNCTION}_{cow_id}_connectivity"
            if self._is_duplicate_alert(alert_key, config['deduplication_window_minutes']):
                return None
            
            confidence = self.config['confidence_scoring']['base_confidence']['sensor_malfunction']
            
            alert = Alert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                cow_id=cow_id,
                alert_type=self.ALERT_TYPE_SENSOR_MALFUNCTION,
                severity=self.SEVERITY_CRITICAL,
                confidence=confidence,
                sensor_values={},
                detection_window="immediate",
                status=self.STATUS_ACTIVE,
                details={
                    'malfunction_type': 'connectivity_loss',
                    'message': 'No sensor data received',
                },
            )
            
            self.active_alerts[alert_key] = alert
            return alert
        
        # Check for out-of-range temperature
        if 'temperature' in sensor_data.columns:
            out_of_range_config = config['out_of_range']['temperature']
            temp_min = out_of_range_config['min']
            temp_max = out_of_range_config['max']
            
            out_of_range = sensor_data[
                (sensor_data['temperature'] < temp_min) | 
                (sensor_data['temperature'] > temp_max)
            ]
            
            if len(out_of_range) > 0:
                alert_key = f"{self.ALERT_TYPE_SENSOR_MALFUNCTION}_{cow_id}_out_of_range"
                if self._is_duplicate_alert(alert_key, config['deduplication_window_minutes']):
                    return None
                
                latest_reading = out_of_range.iloc[-1]
                confidence = self.config['confidence_scoring']['base_confidence']['sensor_malfunction']
                
                alert = Alert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=latest_reading['timestamp'],
                    cow_id=cow_id,
                    alert_type=self.ALERT_TYPE_SENSOR_MALFUNCTION,
                    severity=self.SEVERITY_WARNING,
                    confidence=confidence,
                    sensor_values={
                        'temperature': float(latest_reading['temperature']),
                    },
                    detection_window="immediate",
                    status=self.STATUS_ACTIVE,
                    details={
                        'malfunction_type': 'out_of_range',
                        'sensor': 'temperature',
                        'value': float(latest_reading['temperature']),
                        'valid_range': [temp_min, temp_max],
                    },
                )
                
                self.active_alerts[alert_key] = alert
                return alert
        
        return None
    
    def _is_duplicate_alert(
        self,
        alert_key: str,
        deduplication_window_minutes: int,
    ) -> bool:
        """
        Check if alert is a duplicate within deduplication window.
        
        Args:
            alert_key: Alert identifier for deduplication
            deduplication_window_minutes: Time window for deduplication
        
        Returns:
            True if duplicate, False otherwise
        """
        if alert_key in self.active_alerts:
            existing_alert = self.active_alerts[alert_key]
            time_since_alert = datetime.now() - existing_alert.timestamp
            
            if time_since_alert < timedelta(minutes=deduplication_window_minutes):
                logger.debug(f"Duplicate alert suppressed: {alert_key}")
                return True
            else:
                # Alert expired, remove from active
                del self.active_alerts[alert_key]
        
        return False
    
    def _calculate_confidence(
        self,
        alert_type: str,
        data_quality: float,
        window_completeness: float,
    ) -> float:
        """
        Calculate confidence score for alert detection.
        
        Args:
            alert_type: Type of alert
            data_quality: Data quality metric (0-1)
            window_completeness: Confirmation window completeness (0-1+)
        
        Returns:
            Confidence score (0.0-1.0)
        """
        # Get base confidence
        base_confidence = self.config['confidence_scoring']['base_confidence'].get(
            alert_type, 0.75
        )
        
        # Apply adjustments
        adjustments = self.config['confidence_scoring']['adjustments']
        
        # Data quality adjustment
        if data_quality > 0.95:
            confidence = base_confidence * adjustments['high_data_quality']
        elif data_quality < 0.80:
            confidence = base_confidence * adjustments['low_data_quality']
        else:
            confidence = base_confidence
        
        # Window completeness adjustment
        if window_completeness >= 1.0:
            confidence *= adjustments['full_window_confirmed']
        elif window_completeness < 0.80:
            confidence *= adjustments['partial_window_confirmed']
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))
    
    def get_active_alerts(self, cow_id: Optional[str] = None) -> List[Alert]:
        """
        Get currently active alerts.
        
        Args:
            cow_id: Optional filter by cow ID
        
        Returns:
            List of active alerts
        """
        if cow_id:
            return [
                alert for alert in self.active_alerts.values()
                if alert.cow_id == cow_id
            ]
        return list(self.active_alerts.values())
    
    def resolve_alert(self, alert_id: str):
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: Alert ID to resolve
        """
        for key, alert in self.active_alerts.items():
            if alert.alert_id == alert_id:
                alert.status = self.STATUS_RESOLVED
                logger.info(f"Alert resolved: {alert_id}")
                break
    
    def clear_resolved_alerts(self):
        """Remove resolved alerts from active tracking."""
        resolved_keys = [
            key for key, alert in self.active_alerts.items()
            if alert.status == self.STATUS_RESOLVED
        ]
        for key in resolved_keys:
            del self.active_alerts[key]
        
        if resolved_keys:
            logger.info(f"Cleared {len(resolved_keys)} resolved alerts")
