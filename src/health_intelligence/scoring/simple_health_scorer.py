"""
Simplified Health Score Calculator for Dashboard Upload Workflow

Calculates health scores directly from uploaded sensor data without requiring
complex component-based architecture. Designed for real-time dashboard use.

Score Components (0-100 scale):
- Temperature stability (30%): Deviation from baseline, fever detection
- Activity level (25%): Movement intensity from accelerometer data
- Behavioral patterns (25%): State distribution (lying, standing, ruminating, etc.)
- Alert frequency (20%): Number and severity of active alerts

Health Categories:
- 80-100: Excellent (green)
- 60-79: Good (yellow)
- 40-59: Moderate (orange)
- 0-39: Poor (red)
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List


logger = logging.getLogger(__name__)


class SimpleHealthScorer:
    """
    Simplified health scorer for dashboard use.

    Calculates health scores from sensor data without complex dependencies.
    """

    # Component weights (sum to 1.0)
    TEMPERATURE_WEIGHT = 0.30
    ACTIVITY_WEIGHT = 0.25
    BEHAVIORAL_WEIGHT = 0.25
    ALERT_WEIGHT = 0.20

    # Temperature thresholds
    NORMAL_TEMP_MIN = 38.0
    NORMAL_TEMP_MAX = 39.5
    FEVER_THRESHOLD = 39.5

    # Activity thresholds (normalized 0-1 scale)
    LOW_ACTIVITY_THRESHOLD = 0.15
    NORMAL_ACTIVITY_MIN = 0.2
    NORMAL_ACTIVITY_MAX = 0.8

    # Health category thresholds
    EXCELLENT_THRESHOLD = 80
    GOOD_THRESHOLD = 60
    MODERATE_THRESHOLD = 40

    def __init__(self):
        """Initialize simple health scorer."""
        logger.info("SimpleHealthScorer initialized")

    def calculate_score(
        self,
        cow_id: str,
        sensor_data: pd.DataFrame,
        baseline_temp: float = 38.5,
        active_alerts: Optional[List[Dict]] = None,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate health score from sensor data.

        Args:
            cow_id: Cow identifier
            sensor_data: DataFrame with columns: timestamp, temperature, fxa, mya, rza, state
            baseline_temp: Individual baseline temperature (°C)
            active_alerts: List of active alert dicts
            timestamp: Score calculation timestamp (defaults to now)

        Returns:
            Dictionary with health score and component breakdown
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Initialize scores
        temperature_score = 0.0
        activity_score = 0.0
        behavioral_score = 0.0
        alert_score = 0.0

        # Component confidence values
        temp_confidence = 0.0
        activity_confidence = 0.0
        behavioral_confidence = 0.0
        alert_confidence = 1.0  # Always have alert data (even if empty)

        warnings = []

        # Calculate temperature component (0-25 points, normalized to 0-1)
        if 'temperature' in sensor_data.columns and not sensor_data['temperature'].isna().all():
            temp_result = self._calculate_temperature_score(
                sensor_data, baseline_temp
            )
            temperature_score = temp_result['normalized_score']
            temp_confidence = temp_result['confidence']
            warnings.extend(temp_result['warnings'])
        else:
            warnings.append("No temperature data available")

        # Calculate activity component (0-25 points, normalized to 0-1)
        accel_cols = ['fxa', 'mya', 'rza']
        if all(col in sensor_data.columns for col in accel_cols):
            activity_result = self._calculate_activity_score(sensor_data)
            activity_score = activity_result['normalized_score']
            activity_confidence = activity_result['confidence']
            warnings.extend(activity_result['warnings'])
        else:
            warnings.append("No activity data available")

        # Calculate behavioral component (0-25 points, normalized to 0-1)
        if 'state' in sensor_data.columns and not sensor_data['state'].isna().all():
            behavioral_result = self._calculate_behavioral_score(sensor_data)
            behavioral_score = behavioral_result['normalized_score']
            behavioral_confidence = behavioral_result['confidence']
            warnings.extend(behavioral_result['warnings'])
        else:
            warnings.append("No behavioral state data available")

        # Calculate alert component (0-25 points, normalized to 0-1)
        alert_result = self._calculate_alert_score(active_alerts or [])
        alert_score = alert_result['normalized_score']
        warnings.extend(alert_result['warnings'])

        # Calculate weighted total score (0-100 scale)
        total_score = (
            temperature_score * self.TEMPERATURE_WEIGHT * 100 +
            activity_score * self.ACTIVITY_WEIGHT * 100 +
            behavioral_score * self.BEHAVIORAL_WEIGHT * 100 +
            alert_score * self.ALERT_WEIGHT * 100
        )

        # Calculate overall confidence
        overall_confidence = (
            temp_confidence * self.TEMPERATURE_WEIGHT +
            activity_confidence * self.ACTIVITY_WEIGHT +
            behavioral_confidence * self.BEHAVIORAL_WEIGHT +
            alert_confidence * self.ALERT_WEIGHT
        )

        # Clamp to valid range
        total_score = max(0.0, min(100.0, total_score))

        # Determine health category
        health_category = self._classify_health_category(total_score)

        # Build result dictionary
        result = {
            'timestamp': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'cow_id': cow_id,
            'total_score': round(total_score, 2),
            'temperature_component': round(temperature_score, 3),
            'activity_component': round(activity_score, 3),
            'behavioral_component': round(behavioral_score, 3),
            'alert_component': round(alert_score, 3),
            'health_category': health_category,
            'confidence': round(overall_confidence, 3),
            'weights': {
                'temperature_stability': self.TEMPERATURE_WEIGHT,
                'activity_level': self.ACTIVITY_WEIGHT,
                'behavioral_patterns': self.BEHAVIORAL_WEIGHT,
                'alert_frequency': self.ALERT_WEIGHT
            },
            'metadata': {
                'calculation_timestamp': timestamp.isoformat(),
                'warnings': warnings,
                'baseline_temp': baseline_temp,
                'data_points': len(sensor_data),
                'active_alerts': len(active_alerts or [])
            }
        }

        return result

    def _calculate_temperature_score(
        self,
        sensor_data: pd.DataFrame,
        baseline_temp: float
    ) -> Dict[str, Any]:
        """Calculate temperature component score (0-1 normalized)."""
        temps = sensor_data['temperature'].dropna()

        if len(temps) == 0:
            return {
                'normalized_score': 0.0,
                'confidence': 0.0,
                'warnings': ['No valid temperature readings']
            }

        warnings = []

        # Calculate average temperature
        avg_temp = temps.mean()

        # Calculate deviation from baseline
        deviation = abs(avg_temp - baseline_temp)

        # Detect fever periods (>39.5°C)
        fever_count = (temps > self.FEVER_THRESHOLD).sum()
        fever_ratio = fever_count / len(temps)

        # Calculate score (25 points max, normalized to 0-1)
        score = 25.0

        # Penalize deviation from baseline
        if deviation > 2.0:
            score = 0.0
            warnings.append(f"High temperature deviation: {deviation:.1f}°C from baseline")
        elif deviation > 0.5:
            # Linear penalty between 0.5°C and 2.0°C deviation
            penalty = (deviation - 0.5) / 1.5 * 15.0  # Up to 15 points penalty
            score -= penalty

        # Penalize fever periods
        if fever_ratio > 0.5:
            score -= 15.0
            warnings.append(f"Fever detected in {fever_ratio*100:.1f}% of readings")
        elif fever_ratio > 0.1:
            score -= fever_ratio * 20.0
            warnings.append(f"Elevated temperature in {fever_ratio*100:.1f}% of readings")

        # Clamp to valid range
        score = max(0.0, min(25.0, score))

        # Normalize to 0-1
        normalized_score = score / 25.0

        # Confidence based on data quality
        confidence = min(1.0, len(temps) / 100.0)  # Full confidence at 100+ readings

        return {
            'normalized_score': normalized_score,
            'confidence': confidence,
            'warnings': warnings,
            'avg_temp': avg_temp,
            'deviation': deviation,
            'fever_ratio': fever_ratio
        }

    def _calculate_activity_score(
        self,
        sensor_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate activity component score (0-1 normalized)."""
        # Calculate movement intensity from accelerometer data
        accel_cols = ['fxa', 'mya', 'rza']

        # Calculate magnitude of acceleration vector
        magnitude = np.sqrt(
            sensor_data[accel_cols[0]]**2 +
            sensor_data[accel_cols[1]]**2 +
            sensor_data[accel_cols[2]]**2
        )

        if len(magnitude) == 0:
            return {
                'normalized_score': 0.0,
                'confidence': 0.0,
                'warnings': ['No valid activity readings']
            }

        warnings = []

        # Calculate average activity level
        avg_activity = magnitude.mean()

        # Detect inactivity periods (very low movement)
        inactivity_count = (magnitude < self.LOW_ACTIVITY_THRESHOLD).sum()
        inactivity_ratio = inactivity_count / len(magnitude)

        # Calculate score (25 points max, normalized to 0-1)
        score = 25.0

        # Penalize low activity
        if avg_activity < 0.15:
            score = 5.0  # Very low score for prolonged inactivity
            warnings.append(f"Very low activity level: {avg_activity:.3f}")
        elif avg_activity < self.NORMAL_ACTIVITY_MIN:
            # Reduced activity
            penalty = (self.NORMAL_ACTIVITY_MIN - avg_activity) / self.NORMAL_ACTIVITY_MIN * 15.0
            score -= penalty
            warnings.append(f"Below normal activity: {avg_activity:.3f}")

        # Penalize inactivity periods
        if inactivity_ratio > 0.5:
            score -= 10.0
            warnings.append(f"Inactivity in {inactivity_ratio*100:.1f}% of time")
        elif inactivity_ratio > 0.3:
            score -= inactivity_ratio * 15.0

        # Clamp to valid range
        score = max(0.0, min(25.0, score))

        # Normalize to 0-1
        normalized_score = score / 25.0

        # Confidence based on data quality
        confidence = min(1.0, len(magnitude) / 100.0)

        return {
            'normalized_score': normalized_score,
            'confidence': confidence,
            'warnings': warnings,
            'avg_activity': avg_activity,
            'inactivity_ratio': inactivity_ratio
        }

    def _calculate_behavioral_score(
        self,
        sensor_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate behavioral component score (0-1 normalized)."""
        states = sensor_data['state'].dropna()

        if len(states) == 0:
            return {
                'normalized_score': 0.0,
                'confidence': 0.0,
                'warnings': ['No valid behavioral state data']
            }

        warnings = []

        # Calculate state distribution
        state_counts = states.value_counts()
        state_ratios = state_counts / len(states)

        # Calculate score (25 points max, normalized to 0-1)
        score = 25.0

        # Check for healthy behavioral patterns
        # Healthy cows should have balance of states

        # Penalize excessive lying (>70% of time)
        if 'lying' in state_ratios:
            lying_ratio = state_ratios['lying']
            if lying_ratio > 0.7:
                score -= 10.0
                warnings.append(f"Excessive lying: {lying_ratio*100:.1f}% of time")
            elif lying_ratio > 0.5:
                score -= (lying_ratio - 0.5) * 20.0

        # Rumination detection DISABLED at 1 sample/min (requires ≥10 Hz sampling)
        # No penalty for missing rumination - cannot be detected scientifically
        # See: Schirmann et al. 2009, Burfeind et al. 2011
        ruminating_ratio = 0.0
        for state in ['ruminating', 'ruminating_lying', 'ruminating_standing']:
            ruminating_ratio += state_ratios.get(state, 0)

        if ruminating_ratio > 0:
            # If rumination IS detected (e.g., higher sampling rate in future)
            if ruminating_ratio < 0.1:
                score -= 5.0
                warnings.append(f"Low rumination: {ruminating_ratio*100:.1f}% of time")
            # else: rumination is healthy, no penalty
        # else: No penalty - rumination detection disabled due to sampling rate limitation

        # Check for active states (walking, feeding)
        active_states = ['walking', 'feeding', 'standing']
        active_ratio = sum(state_ratios.get(s, 0) for s in active_states)

        if active_ratio < 0.2:
            score -= 8.0
            warnings.append(f"Low active time: {active_ratio*100:.1f}%")

        # Clamp to valid range
        score = max(0.0, min(25.0, score))

        # Normalize to 0-1
        normalized_score = score / 25.0

        # Confidence based on data quality
        confidence = min(1.0, len(states) / 100.0)

        return {
            'normalized_score': normalized_score,
            'confidence': confidence,
            'warnings': warnings,
            'state_distribution': state_ratios.to_dict()
        }

    def _calculate_alert_score(
        self,
        active_alerts: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate alert component score (0-1 normalized)."""
        warnings = []

        # Calculate score (25 points max, normalized to 0-1)
        score = 25.0

        if not active_alerts:
            # No alerts = perfect score
            return {
                'normalized_score': 1.0,
                'confidence': 1.0,
                'warnings': []
            }

        # Penalize based on alert severity
        for alert in active_alerts:
            severity = alert.get('severity', 'info').lower()

            if severity == 'critical':
                score -= 10.0
                warnings.append(f"Critical alert: {alert.get('alert_type', 'unknown')}")
            elif severity in ['high', 'warning']:
                score -= 5.0
                warnings.append(f"Warning alert: {alert.get('alert_type', 'unknown')}")
            elif severity in ['medium', 'info']:
                score -= 2.0
            else:
                score -= 1.0

        # Clamp to valid range
        score = max(0.0, min(25.0, score))

        # Normalize to 0-1
        normalized_score = score / 25.0

        return {
            'normalized_score': normalized_score,
            'confidence': 1.0,
            'warnings': warnings,
            'alert_count': len(active_alerts)
        }

    def _classify_health_category(self, score: float) -> str:
        """
        Classify health score into category.

        Args:
            score: Health score (0-100)

        Returns:
            Category: 'excellent', 'good', 'moderate', or 'poor'
        """
        if score >= self.EXCELLENT_THRESHOLD:
            return 'excellent'
        elif score >= self.GOOD_THRESHOLD:
            return 'good'
        elif score >= self.MODERATE_THRESHOLD:
            return 'moderate'
        else:
            return 'poor'
