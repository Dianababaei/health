"""
Malfunction Detection Module

This module provides real-time detection of sensor malfunctions including:
- Connectivity loss (>5 minutes without data)
- Stuck values (identical readings for >2 hours)
- Out-of-range values (beyond physical sensor limits)

Each malfunction generates structured alerts with severity levels and deduplication.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
from collections import deque


logger = logging.getLogger(__name__)


class MalfunctionType(Enum):
    """Types of sensor malfunctions."""
    CONNECTIVITY_LOSS = "connectivity_loss"
    STUCK_VALUES = "stuck_values"
    OUT_OF_RANGE = "out_of_range"


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"


class MalfunctionAlert:
    """
    Represents a malfunction alert with all relevant details.
    
    Attributes:
        timestamp: When malfunction was detected
        malfunction_type: Type of malfunction
        affected_sensors: List of affected sensor parameters
        details: Additional information about the issue
        severity: Alert severity level
        alert_id: Unique identifier for deduplication
    """
    
    def __init__(
        self,
        timestamp: datetime,
        malfunction_type: MalfunctionType,
        affected_sensors: List[str],
        details: Dict[str, Any],
        severity: AlertSeverity,
    ):
        """
        Initialize malfunction alert.
        
        Args:
            timestamp: When malfunction was detected
            malfunction_type: Type of malfunction
            affected_sensors: List of affected sensor names
            details: Dictionary with issue details
            severity: Alert severity level
        """
        self.timestamp = timestamp
        self.malfunction_type = malfunction_type
        self.affected_sensors = affected_sensors
        self.details = details
        self.severity = severity
        self.alert_id = self._generate_alert_id()
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID for deduplication."""
        sensors_str = "_".join(sorted(self.affected_sensors))
        return f"{self.malfunction_type.value}_{sensors_str}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format (JSON-serializable)."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'malfunction_type': self.malfunction_type.value,
            'affected_sensors': self.affected_sensors,
            'details': self.details,
            'severity': self.severity.value,
            'alert_id': self.alert_id,
        }
    
    def to_json_string(self) -> str:
        """Convert alert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MalfunctionAlert({self.severity.value}, "
            f"{self.malfunction_type.value}, "
            f"sensors={self.affected_sensors})"
        )


class ConnectivityLossDetector:
    """
    Detect connectivity loss when no data received for >5 consecutive minutes.
    
    Tracks timestamp gaps between consecutive readings.
    """
    
    def __init__(self, gap_threshold_minutes: int = 5):
        """
        Initialize connectivity loss detector.
        
        Args:
            gap_threshold_minutes: Maximum allowed gap (default: 5 minutes)
        """
        self.gap_threshold_minutes = gap_threshold_minutes
        self.last_timestamp: Optional[datetime] = None
        self.gap_threshold = timedelta(minutes=gap_threshold_minutes)
    
    def check(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> List[MalfunctionAlert]:
        """
        Check for connectivity loss in data stream.
        
        Args:
            df: DataFrame with sensor data
            timestamp_column: Name of timestamp column
            
        Returns:
            List of alerts for connectivity loss
        """
        alerts = []
        
        if df.empty or timestamp_column not in df.columns:
            return alerts
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_column).reset_index(drop=True)
        
        # Calculate time differences
        time_diffs = df_sorted[timestamp_column].diff()
        
        # Find gaps larger than threshold
        gaps = time_diffs > self.gap_threshold
        
        if gaps.any():
            gap_indices = df_sorted[gaps].index.tolist()
            
            for idx in gap_indices:
                gap_minutes = time_diffs.iloc[idx].total_seconds() / 60.0
                prev_timestamp = df_sorted[timestamp_column].iloc[idx - 1]
                curr_timestamp = df_sorted[timestamp_column].iloc[idx]
                
                alert = MalfunctionAlert(
                    timestamp=curr_timestamp,
                    malfunction_type=MalfunctionType.CONNECTIVITY_LOSS,
                    affected_sensors=['all'],
                    details={
                        'gap_minutes': gap_minutes,
                        'last_known_timestamp': prev_timestamp.isoformat(),
                        'gap_threshold_minutes': self.gap_threshold_minutes,
                        'message': f"No data received for {gap_minutes:.1f} minutes"
                    },
                    severity=AlertSeverity.CRITICAL,
                )
                alerts.append(alert)
                
                logger.warning(
                    f"Connectivity loss detected: {gap_minutes:.1f} min gap "
                    f"from {prev_timestamp} to {curr_timestamp}"
                )
        
        # Update last timestamp
        if len(df_sorted) > 0:
            self.last_timestamp = df_sorted[timestamp_column].iloc[-1]
        
        return alerts


class StuckValueDetector:
    """
    Detect stuck sensor values when identical reading persists for >2 hours.
    
    Maintains rolling buffer per sensor to check for zero variance over time.
    """
    
    # Sensor columns to monitor
    SENSOR_COLUMNS = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
    
    # Tolerance for floating point comparison
    FLOAT_TOLERANCE = 0.001
    
    def __init__(
        self,
        stuck_threshold_minutes: int = 120,
        expected_interval_minutes: int = 1,
    ):
        """
        Initialize stuck value detector.
        
        Args:
            stuck_threshold_minutes: Duration to consider value stuck (default: 120 min)
            expected_interval_minutes: Expected sampling interval (default: 1 min)
        """
        self.stuck_threshold_minutes = stuck_threshold_minutes
        self.expected_interval_minutes = expected_interval_minutes
        self.min_samples = stuck_threshold_minutes // expected_interval_minutes
        
        # Rolling buffers per sensor: deque of (timestamp, value) tuples
        self.sensor_buffers: Dict[str, deque] = {
            sensor: deque(maxlen=self.min_samples + 10)
            for sensor in self.SENSOR_COLUMNS
        }
        
        logger.info(
            f"StuckValueDetector initialized: threshold={stuck_threshold_minutes}min, "
            f"min_samples={self.min_samples}"
        )
    
    def check(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> List[MalfunctionAlert]:
        """
        Check for stuck sensor values.
        
        Args:
            df: DataFrame with sensor data
            timestamp_column: Name of timestamp column
            
        Returns:
            List of alerts for stuck values
        """
        alerts = []
        
        if df.empty or timestamp_column not in df.columns:
            return alerts
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            df = df.copy()
            df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_column).reset_index(drop=True)
        
        # Update buffers and check each sensor
        for sensor in self.SENSOR_COLUMNS:
            if sensor not in df_sorted.columns:
                continue
            
            # Add new values to buffer
            for idx in df_sorted.index:
                timestamp = df_sorted.loc[idx, timestamp_column]
                value = df_sorted.loc[idx, sensor]
                
                if pd.notna(value):
                    self.sensor_buffers[sensor].append((timestamp, value))
            
            # Check if values are stuck
            alert = self._check_sensor_stuck(sensor)
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _check_sensor_stuck(self, sensor: str) -> Optional[MalfunctionAlert]:
        """
        Check if a specific sensor has stuck values.
        
        Args:
            sensor: Sensor name to check
            
        Returns:
            Alert if stuck, None otherwise
        """
        buffer = self.sensor_buffers[sensor]
        
        # Need enough samples to determine stuck status
        if len(buffer) < self.min_samples:
            return None
        
        # Extract values and timestamps
        timestamps, values = zip(*buffer)
        values_array = np.array(values)
        
        # Check if all values are identical (within tolerance)
        value_range = values_array.max() - values_array.min()
        
        if value_range <= self.FLOAT_TOLERANCE:
            # Values are stuck
            first_timestamp = timestamps[0]
            last_timestamp = timestamps[-1]
            duration_minutes = (last_timestamp - first_timestamp).total_seconds() / 60.0
            
            # Only alert if duration exceeds threshold
            if duration_minutes >= self.stuck_threshold_minutes:
                alert = MalfunctionAlert(
                    timestamp=last_timestamp,
                    malfunction_type=MalfunctionType.STUCK_VALUES,
                    affected_sensors=[sensor],
                    details={
                        'stuck_value': float(values[0]),
                        'duration_minutes': duration_minutes,
                        'sample_count': len(buffer),
                        'first_occurrence': first_timestamp.isoformat(),
                        'message': f"Sensor {sensor} stuck at {values[0]:.3f} for {duration_minutes:.1f} minutes"
                    },
                    severity=AlertSeverity.WARNING,
                )
                
                logger.warning(
                    f"Stuck value detected: {sensor} = {values[0]:.3f} "
                    f"for {duration_minutes:.1f} minutes"
                )
                
                return alert
        
        return None
    
    def reset_sensor(self, sensor: str):
        """Reset buffer for a specific sensor."""
        if sensor in self.sensor_buffers:
            self.sensor_buffers[sensor].clear()
    
    def reset_all(self):
        """Reset all sensor buffers."""
        for sensor in self.SENSOR_COLUMNS:
            self.sensor_buffers[sensor].clear()


class OutOfRangeDetector:
    """
    Detect values beyond physical sensor limits.
    
    Checks:
    - Temperature: <35°C or >42°C
    - Accelerations: >10g (hardware limit)
    - Angular velocities: >500 deg/s (sensor specification)
    """
    
    # Physical limits for temperature (viable cattle range)
    TEMP_MIN = 35.0
    TEMP_MAX = 42.0
    
    # Hardware limit for accelerations
    ACCELERATION_MAX = 10.0
    
    # Sensor specification for angular velocity
    ANGULAR_VELOCITY_MAX = 500.0
    
    # Sensor column mappings
    ACCELERATION_SENSORS = ['fxa', 'mya', 'rza']
    ANGULAR_VELOCITY_SENSORS = ['sxg', 'lyg', 'dzg']
    
    def check(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> List[MalfunctionAlert]:
        """
        Check for out-of-range sensor values.
        
        Args:
            df: DataFrame with sensor data
            timestamp_column: Name of timestamp column
            
        Returns:
            List of alerts for out-of-range values
        """
        alerts = []
        
        if df.empty:
            return alerts
        
        # Ensure timestamps are datetime
        if timestamp_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                df = df.copy()
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # Check temperature
        if 'temperature' in df.columns:
            temp_alerts = self._check_temperature(df, timestamp_column)
            alerts.extend(temp_alerts)
        
        # Check accelerations
        for sensor in self.ACCELERATION_SENSORS:
            if sensor in df.columns:
                acc_alerts = self._check_acceleration(df, sensor, timestamp_column)
                alerts.extend(acc_alerts)
        
        # Check angular velocities
        for sensor in self.ANGULAR_VELOCITY_SENSORS:
            if sensor in df.columns:
                gyro_alerts = self._check_angular_velocity(df, sensor, timestamp_column)
                alerts.extend(gyro_alerts)
        
        return alerts
    
    def _check_temperature(
        self,
        df: pd.DataFrame,
        timestamp_column: str
    ) -> List[MalfunctionAlert]:
        """Check temperature out-of-range."""
        alerts = []
        
        out_of_range = (df['temperature'] < self.TEMP_MIN) | (df['temperature'] > self.TEMP_MAX)
        
        if out_of_range.any():
            out_indices = df[out_of_range].index.tolist()
            
            for idx in out_indices:
                temp_val = df.loc[idx, 'temperature']
                
                if pd.notna(temp_val):
                    timestamp = df.loc[idx, timestamp_column] if timestamp_column in df.columns else datetime.now()
                    
                    if temp_val < self.TEMP_MIN:
                        reason = f"below viable range (<{self.TEMP_MIN}°C)"
                    else:
                        reason = f"above viable range (>{self.TEMP_MAX}°C)"
                    
                    alert = MalfunctionAlert(
                        timestamp=timestamp,
                        malfunction_type=MalfunctionType.OUT_OF_RANGE,
                        affected_sensors=['temperature'],
                        details={
                            'sensor': 'temperature',
                            'value': float(temp_val),
                            'min_limit': self.TEMP_MIN,
                            'max_limit': self.TEMP_MAX,
                            'message': f"Temperature {temp_val:.2f}°C {reason}"
                        },
                        severity=AlertSeverity.CRITICAL,
                    )
                    alerts.append(alert)
                    
                    logger.error(
                        f"Out-of-range temperature: {temp_val:.2f}°C at {timestamp}"
                    )
        
        return alerts
    
    def _check_acceleration(
        self,
        df: pd.DataFrame,
        sensor: str,
        timestamp_column: str
    ) -> List[MalfunctionAlert]:
        """Check acceleration out-of-range (>10g hardware limit)."""
        alerts = []
        
        out_of_range = df[sensor].abs() > self.ACCELERATION_MAX
        
        if out_of_range.any():
            out_indices = df[out_of_range].index.tolist()
            
            for idx in out_indices:
                acc_val = df.loc[idx, sensor]
                
                if pd.notna(acc_val):
                    timestamp = df.loc[idx, timestamp_column] if timestamp_column in df.columns else datetime.now()
                    
                    alert = MalfunctionAlert(
                        timestamp=timestamp,
                        malfunction_type=MalfunctionType.OUT_OF_RANGE,
                        affected_sensors=[sensor],
                        details={
                            'sensor': sensor,
                            'value': float(acc_val),
                            'limit': self.ACCELERATION_MAX,
                            'message': f"Acceleration {sensor}={acc_val:.2f}g exceeds hardware limit (>{self.ACCELERATION_MAX}g)"
                        },
                        severity=AlertSeverity.CRITICAL,
                    )
                    alerts.append(alert)
                    
                    logger.error(
                        f"Out-of-range acceleration: {sensor}={acc_val:.2f}g at {timestamp}"
                    )
        
        return alerts
    
    def _check_angular_velocity(
        self,
        df: pd.DataFrame,
        sensor: str,
        timestamp_column: str
    ) -> List[MalfunctionAlert]:
        """Check angular velocity out-of-range (>500 deg/s sensor spec)."""
        alerts = []
        
        out_of_range = df[sensor].abs() > self.ANGULAR_VELOCITY_MAX
        
        if out_of_range.any():
            out_indices = df[out_of_range].index.tolist()
            
            for idx in out_indices:
                gyro_val = df.loc[idx, sensor]
                
                if pd.notna(gyro_val):
                    timestamp = df.loc[idx, timestamp_column] if timestamp_column in df.columns else datetime.now()
                    
                    alert = MalfunctionAlert(
                        timestamp=timestamp,
                        malfunction_type=MalfunctionType.OUT_OF_RANGE,
                        affected_sensors=[sensor],
                        details={
                            'sensor': sensor,
                            'value': float(gyro_val),
                            'limit': self.ANGULAR_VELOCITY_MAX,
                            'message': f"Angular velocity {sensor}={gyro_val:.2f}°/s exceeds sensor specification (>{self.ANGULAR_VELOCITY_MAX}°/s)"
                        },
                        severity=AlertSeverity.CRITICAL,
                    )
                    alerts.append(alert)
                    
                    logger.error(
                        f"Out-of-range angular velocity: {sensor}={gyro_val:.2f}°/s at {timestamp}"
                    )
        
        return alerts


class MalfunctionDetector:
    """
    Main malfunction detection coordinator.
    
    Combines all detection methods with alert deduplication and management.
    """
    
    def __init__(
        self,
        connectivity_gap_minutes: int = 5,
        stuck_threshold_minutes: int = 120,
        enable_deduplication: bool = True,
        deduplication_window_minutes: int = 60,
    ):
        """
        Initialize malfunction detector.
        
        Args:
            connectivity_gap_minutes: Gap threshold for connectivity loss
            stuck_threshold_minutes: Duration threshold for stuck values
            enable_deduplication: Whether to deduplicate repeated alerts
            deduplication_window_minutes: Time window for deduplication
        """
        self.connectivity_detector = ConnectivityLossDetector(connectivity_gap_minutes)
        self.stuck_value_detector = StuckValueDetector(stuck_threshold_minutes)
        self.out_of_range_detector = OutOfRangeDetector()
        
        self.enable_deduplication = enable_deduplication
        self.deduplication_window = timedelta(minutes=deduplication_window_minutes)
        
        # Track recent alerts for deduplication: {alert_id: last_timestamp}
        self.recent_alerts: Dict[str, datetime] = {}
        
        logger.info("MalfunctionDetector initialized")
    
    def detect(
        self,
        df: pd.DataFrame,
        timestamp_column: str = 'timestamp'
    ) -> List[MalfunctionAlert]:
        """
        Detect all malfunction types in data.
        
        Args:
            df: DataFrame with sensor data
            timestamp_column: Name of timestamp column
            
        Returns:
            List of deduplicated malfunction alerts
        """
        all_alerts = []
        
        # Run all detectors
        connectivity_alerts = self.connectivity_detector.check(df, timestamp_column)
        stuck_alerts = self.stuck_value_detector.check(df, timestamp_column)
        range_alerts = self.out_of_range_detector.check(df, timestamp_column)
        
        all_alerts.extend(connectivity_alerts)
        all_alerts.extend(stuck_alerts)
        all_alerts.extend(range_alerts)
        
        # Apply deduplication
        if self.enable_deduplication:
            all_alerts = self._deduplicate_alerts(all_alerts)
        
        return all_alerts
    
    def _deduplicate_alerts(
        self,
        alerts: List[MalfunctionAlert]
    ) -> List[MalfunctionAlert]:
        """
        Remove duplicate alerts within deduplication window.
        
        Args:
            alerts: List of alerts to deduplicate
            
        Returns:
            Deduplicated list of alerts
        """
        deduplicated = []
        current_time = datetime.now()
        
        # Clean up old alerts from tracking
        expired_ids = [
            alert_id for alert_id, timestamp in self.recent_alerts.items()
            if current_time - timestamp > self.deduplication_window
        ]
        for alert_id in expired_ids:
            del self.recent_alerts[alert_id]
        
        # Check each alert
        for alert in alerts:
            alert_id = alert.alert_id
            
            # Check if we've seen this alert recently
            if alert_id in self.recent_alerts:
                last_seen = self.recent_alerts[alert_id]
                if alert.timestamp - last_seen < self.deduplication_window:
                    # Skip duplicate
                    logger.debug(f"Skipping duplicate alert: {alert_id}")
                    continue
            
            # Add to deduplicated list and update tracking
            deduplicated.append(alert)
            self.recent_alerts[alert_id] = alert.timestamp
        
        if len(alerts) != len(deduplicated):
            logger.info(
                f"Deduplicated {len(alerts)} alerts to {len(deduplicated)} "
                f"({len(alerts) - len(deduplicated)} duplicates removed)"
            )
        
        return deduplicated
    
    def get_alert_summary(self, alerts: List[MalfunctionAlert]) -> Dict[str, Any]:
        """
        Get summary statistics of alerts.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Dictionary with summary statistics
        """
        if not alerts:
            return {
                'total_alerts': 0,
                'by_type': {},
                'by_severity': {},
                'affected_sensors': []
            }
        
        # Count by type
        by_type = {}
        for malfunction_type in MalfunctionType:
            count = sum(1 for a in alerts if a.malfunction_type == malfunction_type)
            if count > 0:
                by_type[malfunction_type.value] = count
        
        # Count by severity
        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in alerts if a.severity == severity)
            if count > 0:
                by_severity[severity.value] = count
        
        # Collect all affected sensors
        affected_sensors = set()
        for alert in alerts:
            affected_sensors.update(alert.affected_sensors)
        
        return {
            'total_alerts': len(alerts),
            'by_type': by_type,
            'by_severity': by_severity,
            'affected_sensors': sorted(affected_sensors)
        }
    
    def reset(self):
        """Reset all detector states."""
        self.stuck_value_detector.reset_all()
        self.recent_alerts.clear()
        logger.info("MalfunctionDetector reset")
