"""
Temperature Anomaly Detection Module

Detects temperature anomalies based on baseline deviations and absolute thresholds.
Supports fever detection, heat stress identification, and hypothermia detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of temperature anomalies."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    FEVER = "fever"
    HEAT_STRESS = "heat_stress"
    HYPOTHERMIA = "hypothermia"


@dataclass
class TemperatureAnomaly:
    """Represents a detected temperature anomaly."""
    timestamp: datetime
    temperature: float
    baseline: Optional[float]
    deviation: Optional[float]
    anomaly_type: AnomalyType
    severity: float  # 0.0-1.0
    confidence: float  # 0.0-1.0


class TemperatureAnomalyDetector:
    """
    Detects temperature anomalies based on absolute thresholds and baseline deviations.
    
    Features:
    - Absolute threshold detection (fever >39.0°C, heat stress >39.5°C)
    - Baseline deviation detection
    - Severity scoring
    - Anomaly event generation
    """
    
    # Clinical thresholds for cattle (°C)
    NORMAL_RANGE = (37.5, 39.0)
    FEVER_THRESHOLD = 39.0
    HEAT_STRESS_THRESHOLD = 39.5
    HYPOTHERMIA_THRESHOLD = 37.0
    
    def __init__(
        self,
        baseline_deviation_threshold: float = 1.0,
        fever_threshold: float = 39.0,
        heat_stress_threshold: float = 39.5,
        hypothermia_threshold: float = 37.0
    ):
        """
        Initialize anomaly detector.
        
        Args:
            baseline_deviation_threshold: Standard deviations from baseline for anomaly
            fever_threshold: Temperature above which fever is detected (°C)
            heat_stress_threshold: Temperature above which heat stress is detected (°C)
            hypothermia_threshold: Temperature below which hypothermia is detected (°C)
        """
        self.baseline_deviation_threshold = baseline_deviation_threshold
        self.fever_threshold = fever_threshold
        self.heat_stress_threshold = heat_stress_threshold
        self.hypothermia_threshold = hypothermia_threshold
        
        logger.info(f"TemperatureAnomalyDetector initialized: "
                   f"fever={fever_threshold}°C, heat_stress={heat_stress_threshold}°C")
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        baseline_col: Optional[str] = 'baseline_temp',
        baseline_std_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect temperature anomalies in dataset.
        
        Args:
            data: DataFrame with temperature data
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            baseline_col: Name of baseline temperature column (optional)
            baseline_std_col: Name of baseline std deviation column (optional)
            
        Returns:
            DataFrame with added columns:
            - anomaly_type: Type of anomaly detected
            - anomaly_severity: Severity score (0.0-1.0)
            - anomaly_confidence: Confidence in detection (0.0-1.0)
            - deviation_from_baseline: Deviation in °C (if baseline available)
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Initialize columns
        df['anomaly_type'] = AnomalyType.NORMAL.value
        df['anomaly_severity'] = 0.0
        df['anomaly_confidence'] = 1.0
        df['deviation_from_baseline'] = None
        
        # Calculate deviation from baseline if available
        if baseline_col in df.columns:
            df['deviation_from_baseline'] = df[temperature_col] - df[baseline_col]
        
        # Detect anomalies row by row
        for idx, row in df.iterrows():
            temp = row[temperature_col]
            
            # Skip if temperature is NaN
            if pd.isna(temp):
                continue
            
            anomaly = self._classify_temperature(
                temperature=temp,
                baseline=row.get(baseline_col) if baseline_col in df.columns else None,
                baseline_std=row.get(baseline_std_col) if baseline_std_col and baseline_std_col in df.columns else None
            )
            
            df.at[idx, 'anomaly_type'] = anomaly['type'].value
            df.at[idx, 'anomaly_severity'] = anomaly['severity']
            df.at[idx, 'anomaly_confidence'] = anomaly['confidence']
        
        return df
    
    def _classify_temperature(
        self,
        temperature: float,
        baseline: Optional[float] = None,
        baseline_std: Optional[float] = None
    ) -> Dict:
        """
        Classify a single temperature reading.
        
        Args:
            temperature: Current temperature (°C)
            baseline: Baseline temperature (°C)
            baseline_std: Baseline standard deviation (°C)
            
        Returns:
            Dictionary with type, severity, and confidence
        """
        # Priority order: hypothermia > heat_stress > fever > elevated > normal
        
        # Check for hypothermia (most critical)
        if temperature < self.hypothermia_threshold:
            severity = min(1.0, (self.hypothermia_threshold - temperature) / 2.0)
            return {
                'type': AnomalyType.HYPOTHERMIA,
                'severity': severity,
                'confidence': 0.95
            }
        
        # Check for heat stress (>39.5°C)
        if temperature >= self.heat_stress_threshold:
            # Severity increases with temperature
            severity = min(1.0, (temperature - self.heat_stress_threshold) / 2.0 + 0.5)
            return {
                'type': AnomalyType.HEAT_STRESS,
                'severity': severity,
                'confidence': 0.90
            }
        
        # Check for fever (>39.0°C)
        if temperature >= self.fever_threshold:
            severity = min(1.0, (temperature - self.fever_threshold) / 1.5 + 0.3)
            return {
                'type': AnomalyType.FEVER,
                'severity': severity,
                'confidence': 0.85
            }
        
        # Check for elevation based on baseline deviation
        if baseline is not None and baseline_std is not None:
            deviation = temperature - baseline
            deviation_std = deviation / baseline_std if baseline_std > 0 else 0
            
            if deviation_std > self.baseline_deviation_threshold:
                severity = min(1.0, deviation_std / 3.0)
                return {
                    'type': AnomalyType.ELEVATED,
                    'severity': severity,
                    'confidence': 0.75
                }
        
        # Normal temperature
        return {
            'type': AnomalyType.NORMAL,
            'severity': 0.0,
            'confidence': 1.0
        }
    
    def extract_anomaly_events(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        min_duration_minutes: int = 5
    ) -> List[TemperatureAnomaly]:
        """
        Extract discrete anomaly events from detected anomalies.
        
        Groups consecutive anomalous readings into events.
        
        Args:
            data: DataFrame with detected anomalies
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            min_duration_minutes: Minimum duration to consider an event
            
        Returns:
            List of TemperatureAnomaly objects
        """
        df = data.copy()
        
        if 'anomaly_type' not in df.columns:
            logger.warning("No anomaly_type column found. Run detect_anomalies first.")
            return []
        
        # Filter to anomalous readings only
        anomalous = df[df['anomaly_type'] != AnomalyType.NORMAL.value].copy()
        
        if len(anomalous) == 0:
            return []
        
        # Sort by timestamp
        anomalous = anomalous.sort_values(timestamp_col).reset_index(drop=True)
        
        events = []
        
        # Group consecutive anomalies
        anomalous['group_changed'] = (anomalous['anomaly_type'] != anomalous['anomaly_type'].shift(1))
        anomalous['group_id'] = anomalous['group_changed'].cumsum()
        
        for group_id, group_data in anomalous.groupby('group_id'):
            # Calculate duration
            if len(group_data) >= 2:
                duration = (group_data[timestamp_col].iloc[-1] - 
                          group_data[timestamp_col].iloc[0]).total_seconds() / 60.0
            else:
                duration = 1.0  # Assume 1 minute for single reading
            
            # Skip if duration too short
            if duration < min_duration_minutes:
                continue
            
            # Create event from group
            event = TemperatureAnomaly(
                timestamp=group_data[timestamp_col].iloc[0],
                temperature=float(group_data[temperature_col].mean()),
                baseline=float(group_data['baseline_temp'].mean()) if 'baseline_temp' in group_data.columns else None,
                deviation=float(group_data['deviation_from_baseline'].mean()) if 'deviation_from_baseline' in group_data.columns else None,
                anomaly_type=AnomalyType(group_data['anomaly_type'].iloc[0]),
                severity=float(group_data['anomaly_severity'].mean()),
                confidence=float(group_data['anomaly_confidence'].mean())
            )
            
            events.append(event)
        
        logger.info(f"Extracted {len(events)} anomaly events from {len(anomalous)} anomalous readings")
        return events
