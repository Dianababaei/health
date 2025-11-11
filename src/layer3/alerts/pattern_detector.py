"""
Pattern Alert Detection System

Detects estrus and pregnancy patterns using sliding window analysis with
multi-minute confirmation periods. Integrates temperature, activity, and
physiological trend data to identify reproductive patterns.

Features:
- Estrus detection: 5-10 minute windows, 0.3-0.6°C temp rise + activity increase
- Pregnancy detection: 7-14 day windows, stable temp + reduced activity post-estrus
- Sliding window management with configurable sizes
- Confidence scoring that increases over confirmation period
- Temperature rise validation (distinguish estrus from fever)
- Pattern confirmation state transitions (pending → confirmed)
"""

import logging
import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of pattern alerts."""
    ESTRUS = "estrus"
    PREGNANCY_INDICATION = "pregnancy_indication"


class AlertStatus(Enum):
    """Alert status values."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    RESOLVED = "resolved"


@dataclass
class PatternAlert:
    """
    Pattern alert data structure for estrus and pregnancy detection.
    
    Attributes:
        alert_id: Unique identifier
        timestamp: Detection timestamp
        cow_id: Animal identifier
        alert_type: Type of alert (estrus, pregnancy_indication)
        confidence: Detection confidence (0.0-1.0)
        detection_window: Window start, end, and duration
        pattern_metrics: Temperature rise, activity delta, stability scores
        supporting_data: Relevant sensor trends during detection window
        status: Alert status (pending, confirmed, resolved)
        related_events: Links to previous reproductive events
    """
    alert_id: str
    timestamp: datetime
    cow_id: str
    alert_type: str
    confidence: float
    detection_window: Dict[str, Any]
    pattern_metrics: Dict[str, Any]
    supporting_data: Dict[str, Any]
    status: str
    related_events: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with ISO format timestamps."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['detection_window']['start'] = self.detection_window['start'].isoformat()
        data['detection_window']['end'] = self.detection_window['end'].isoformat()
        return data


@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window analysis."""
    window_minutes: int  # Window size in minutes
    update_frequency_minutes: int  # How often to re-evaluate
    min_data_completeness: float = 0.80  # Require 80% data coverage


class PatternAlertDetector:
    """
    Detects estrus and pregnancy patterns using sliding window analysis.
    
    Analyzes temperature and activity data over configurable windows to identify:
    - Estrus: 0.3-0.6°C temperature rise + activity increase (5-10 min confirmation)
    - Pregnancy: Stable temperature + reduced activity post-estrus (7-14 days)
    
    Features:
    - Sliding window management
    - Multi-source data integration (temperature, activity, baselines, trends)
    - Confidence scoring that increases as patterns persist
    - Temperature rise validation (distinguish reproductive patterns from illness)
    - Alert state management (pending → confirmed)
    """
    
    # Default window configurations
    ESTRUS_WINDOW_CONFIG = SlidingWindowConfig(
        window_minutes=10,
        update_frequency_minutes=1,
        min_data_completeness=0.80
    )
    
    PREGNANCY_WINDOW_CONFIG = SlidingWindowConfig(
        window_minutes=14 * 24 * 60,  # 14 days in minutes
        update_frequency_minutes=60,  # Re-evaluate hourly
        min_data_completeness=0.75
    )
    
    # Detection thresholds
    ESTRUS_TEMP_RISE_MIN = 0.3  # °C
    ESTRUS_TEMP_RISE_MAX = 0.6  # °C
    ESTRUS_ACTIVITY_INCREASE_MIN = 0.15  # 15% increase
    ESTRUS_TEMP_RISE_RATE_MAX = 0.15  # °C per minute (gradual, not sudden fever)
    
    PREGNANCY_TEMP_STABILITY_CV_MAX = 0.05  # Coefficient of variation
    PREGNANCY_ACTIVITY_REDUCTION_MIN = 0.10  # 10% reduction
    PREGNANCY_POST_ESTRUS_MIN_DAYS = 7
    PREGNANCY_POST_ESTRUS_MAX_DAYS = 30
    
    # Confirmation thresholds
    ESTRUS_CONFIRMATION_MINUTES = 5
    PREGNANCY_CONFIRMATION_DAYS = 10
    
    def __init__(
        self,
        estrus_window_minutes: int = 10,
        pregnancy_window_days: int = 14,
        enable_cycle_tracking: bool = True
    ):
        """
        Initialize pattern alert detector.
        
        Args:
            estrus_window_minutes: Window size for estrus detection (default: 10)
            pregnancy_window_days: Window size for pregnancy detection (default: 14)
            enable_cycle_tracking: Enable reproductive cycle tracking
        """
        self.estrus_window_config = SlidingWindowConfig(
            window_minutes=estrus_window_minutes,
            update_frequency_minutes=1,
            min_data_completeness=0.80
        )
        
        self.pregnancy_window_config = SlidingWindowConfig(
            window_minutes=pregnancy_window_days * 24 * 60,
            update_frequency_minutes=60,
            min_data_completeness=0.75
        )
        
        self.enable_cycle_tracking = enable_cycle_tracking
        
        # Track active alerts per cow
        self.active_alerts: Dict[str, List[PatternAlert]] = {}
        
        # Track detection history for confidence scoring
        self.detection_history: Dict[str, List[Dict]] = {}
        
        logger.info(
            f"PatternAlertDetector initialized: "
            f"estrus_window={estrus_window_minutes}min, "
            f"pregnancy_window={pregnancy_window_days}days"
        )
    
    def detect_patterns(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        baseline_temp: float,
        activity_baseline: Optional[Dict[str, float]] = None,
        estrus_history: Optional[List[Dict]] = None
    ) -> List[PatternAlert]:
        """
        Detect estrus and pregnancy patterns in sensor data.
        
        Args:
            cow_id: Animal identifier
            temperature_data: DataFrame with columns ['timestamp', 'temperature']
            activity_data: DataFrame with columns ['timestamp', 'movement_intensity', 'behavioral_state']
            baseline_temp: Individual cow's baseline temperature
            activity_baseline: Baseline activity metrics (optional)
            estrus_history: Previous estrus detections for pregnancy linkage
            
        Returns:
            List of detected pattern alerts
        """
        alerts = []
        
        # Detect estrus patterns
        estrus_alerts = self._detect_estrus(
            cow_id=cow_id,
            temperature_data=temperature_data,
            activity_data=activity_data,
            baseline_temp=baseline_temp,
            activity_baseline=activity_baseline
        )
        alerts.extend(estrus_alerts)
        
        # Detect pregnancy patterns (only if estrus history available)
        if estrus_history and len(estrus_history) > 0:
            pregnancy_alerts = self._detect_pregnancy(
                cow_id=cow_id,
                temperature_data=temperature_data,
                activity_data=activity_data,
                baseline_temp=baseline_temp,
                activity_baseline=activity_baseline,
                estrus_history=estrus_history
            )
            alerts.extend(pregnancy_alerts)
        
        # Update active alerts
        if cow_id not in self.active_alerts:
            self.active_alerts[cow_id] = []
        self.active_alerts[cow_id].extend(alerts)
        
        return alerts
    
    def _detect_estrus(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        baseline_temp: float,
        activity_baseline: Optional[Dict[str, float]]
    ) -> List[PatternAlert]:
        """
        Detect estrus patterns using sliding window analysis.
        
        Estrus criteria:
        - Temperature rise of 0.3-0.6°C above baseline
        - Activity increase >15% above baseline
        - Gradual temperature rise (not sudden fever spike)
        - Pattern persists for 5-10 minutes
        
        Returns:
            List of estrus alerts
        """
        alerts = []
        
        if temperature_data.empty or activity_data.empty:
            return alerts
        
        # Merge temperature and activity data
        merged_data = self._merge_sensor_data(temperature_data, activity_data)
        
        if merged_data.empty:
            return alerts
        
        # Calculate sliding windows
        window_size = self.estrus_window_config.window_minutes
        current_time = merged_data['timestamp'].max()
        window_start = current_time - timedelta(minutes=window_size)
        
        window_data = merged_data[
            (merged_data['timestamp'] >= window_start) &
            (merged_data['timestamp'] <= current_time)
        ]
        
        if len(window_data) == 0:
            return alerts
        
        # Check data completeness
        expected_samples = window_size  # 1 sample per minute
        actual_samples = len(window_data)
        completeness = actual_samples / expected_samples if expected_samples > 0 else 0
        
        if completeness < self.estrus_window_config.min_data_completeness:
            logger.debug(f"Insufficient data completeness for estrus detection: {completeness:.2%}")
            return alerts
        
        # Calculate window metrics
        metrics = self._calculate_estrus_metrics(
            window_data=window_data,
            baseline_temp=baseline_temp,
            activity_baseline=activity_baseline
        )
        
        # Check if estrus pattern is detected
        if self._is_estrus_pattern(metrics):
            # Calculate confidence based on pattern strength and duration
            confidence = self._calculate_estrus_confidence(metrics, window_data)
            
            # Determine alert status
            pattern_duration = (current_time - window_start).total_seconds() / 60
            status = (AlertStatus.CONFIRMED.value 
                     if pattern_duration >= self.ESTRUS_CONFIRMATION_MINUTES 
                     else AlertStatus.PENDING.value)
            
            # Create alert
            alert = PatternAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=current_time,
                cow_id=cow_id,
                alert_type=AlertType.ESTRUS.value,
                confidence=confidence,
                detection_window={
                    'start': window_start,
                    'end': current_time,
                    'duration': f"{window_size} minutes"
                },
                pattern_metrics=metrics,
                supporting_data={
                    'temperature_trend': window_data['temperature'].tolist()[-10:],
                    'activity_trend': window_data.get('movement_intensity', pd.Series()).tolist()[-10:],
                    'baseline_temp': baseline_temp,
                    'data_completeness': completeness
                },
                status=status,
                related_events=[]
            )
            
            alerts.append(alert)
            logger.info(
                f"Estrus detected for cow {cow_id}: "
                f"temp_rise={metrics['temp_rise']:.2f}°C, "
                f"activity_increase={metrics['activity_increase']:.1%}, "
                f"confidence={confidence:.2f}"
            )
        
        return alerts
    
    def _detect_pregnancy(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        baseline_temp: float,
        activity_baseline: Optional[Dict[str, float]],
        estrus_history: List[Dict]
    ) -> List[PatternAlert]:
        """
        Detect pregnancy patterns using multi-day stability tracking.
        
        Pregnancy criteria:
        - Stable temperature (low variance, CV < 0.05)
        - Reduced activity compared to pre-estrus baseline
        - Prior estrus detection within 7-30 days
        - Pattern persists over 7-14 day window
        
        Returns:
            List of pregnancy indication alerts
        """
        alerts = []
        
        if temperature_data.empty or activity_data.empty:
            return alerts
        
        # Find most recent estrus event within valid timeframe
        current_time = temperature_data['timestamp'].max()
        recent_estrus = None
        
        for estrus in sorted(estrus_history, key=lambda x: x.get('timestamp', datetime.min), reverse=True):
            estrus_time = estrus.get('timestamp')
            if estrus_time is None:
                continue
            
            days_since_estrus = (current_time - estrus_time).days
            
            if self.PREGNANCY_POST_ESTRUS_MIN_DAYS <= days_since_estrus <= self.PREGNANCY_POST_ESTRUS_MAX_DAYS:
                recent_estrus = estrus
                break
        
        if recent_estrus is None:
            return alerts
        
        # Define pregnancy detection window (7-14 days post-estrus minimum)
        estrus_time = recent_estrus['timestamp']
        window_start = estrus_time + timedelta(days=self.PREGNANCY_POST_ESTRUS_MIN_DAYS)
        window_end = current_time
        
        # Filter data to pregnancy window
        temp_window = temperature_data[
            (temperature_data['timestamp'] >= window_start) &
            (temperature_data['timestamp'] <= window_end)
        ]
        
        activity_window = activity_data[
            (activity_data['timestamp'] >= window_start) &
            (activity_data['timestamp'] <= window_end)
        ]
        
        if temp_window.empty or activity_window.empty:
            return alerts
        
        # Check data completeness
        window_days = (window_end - window_start).days
        expected_samples = window_days * 1440  # 1 sample per minute
        actual_samples = len(temp_window)
        completeness = actual_samples / expected_samples if expected_samples > 0 else 0
        
        if completeness < self.pregnancy_window_config.min_data_completeness:
            logger.debug(f"Insufficient data completeness for pregnancy detection: {completeness:.2%}")
            return alerts
        
        # Calculate pregnancy metrics
        metrics = self._calculate_pregnancy_metrics(
            temp_window=temp_window,
            activity_window=activity_window,
            baseline_temp=baseline_temp,
            activity_baseline=activity_baseline,
            estrus_time=estrus_time
        )
        
        # Check if pregnancy pattern is detected
        if self._is_pregnancy_pattern(metrics):
            # Calculate confidence (increases over time)
            confidence = self._calculate_pregnancy_confidence(metrics, window_days)
            
            # Determine status (confirmed after sufficient observation period)
            status = (AlertStatus.CONFIRMED.value 
                     if window_days >= self.PREGNANCY_CONFIRMATION_DAYS 
                     else AlertStatus.PENDING.value)
            
            # Create alert
            alert = PatternAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=current_time,
                cow_id=cow_id,
                alert_type=AlertType.PREGNANCY_INDICATION.value,
                confidence=confidence,
                detection_window={
                    'start': window_start,
                    'end': window_end,
                    'duration': f"{window_days} days"
                },
                pattern_metrics=metrics,
                supporting_data={
                    'temperature_stability': metrics['temp_cv'],
                    'activity_reduction': metrics['activity_reduction'],
                    'days_post_estrus': (current_time - estrus_time).days,
                    'data_completeness': completeness
                },
                status=status,
                related_events=[recent_estrus.get('alert_id', 'unknown')]
            )
            
            alerts.append(alert)
            logger.info(
                f"Pregnancy indication detected for cow {cow_id}: "
                f"temp_cv={metrics['temp_cv']:.3f}, "
                f"activity_reduction={metrics['activity_reduction']:.1%}, "
                f"days_post_estrus={window_days}, "
                f"confidence={confidence:.2f}"
            )
        
        return alerts
    
    def _merge_sensor_data(
        self,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge temperature and activity data by timestamp."""
        # Ensure timestamps are datetime
        temp_df = temperature_data.copy()
        activity_df = activity_data.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(temp_df['timestamp']):
            temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        
        if not pd.api.types.is_datetime64_any_dtype(activity_df['timestamp']):
            activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])
        
        # Merge on timestamp
        merged = pd.merge(
            temp_df,
            activity_df,
            on='timestamp',
            how='inner'
        )
        
        return merged.sort_values('timestamp').reset_index(drop=True)
    
    def _calculate_estrus_metrics(
        self,
        window_data: pd.DataFrame,
        baseline_temp: float,
        activity_baseline: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Calculate estrus detection metrics.
        
        Returns:
            Dictionary with temp_rise, activity_increase, temp_rise_rate, etc.
        """
        temps = window_data['temperature'].values
        mean_temp = np.mean(temps)
        temp_rise = mean_temp - baseline_temp
        
        # Calculate temperature rise rate (°C per minute)
        if len(temps) > 1:
            time_span_minutes = len(temps)
            temp_range = np.max(temps) - np.min(temps)
            temp_rise_rate = temp_range / time_span_minutes if time_span_minutes > 0 else 0
        else:
            temp_rise_rate = 0
        
        # Calculate activity increase
        activity_increase = 0.0
        if 'movement_intensity' in window_data.columns:
            mean_activity = window_data['movement_intensity'].mean()
            baseline_activity = activity_baseline.get('mean', 0.5) if activity_baseline else 0.5
            
            if baseline_activity > 0:
                activity_increase = (mean_activity - baseline_activity) / baseline_activity
        
        # Check correlation between temperature and activity
        temp_activity_correlation = 0.0
        if 'movement_intensity' in window_data.columns and len(window_data) > 2:
            try:
                temp_activity_correlation = np.corrcoef(
                    window_data['temperature'].values,
                    window_data['movement_intensity'].values
                )[0, 1]
                if np.isnan(temp_activity_correlation):
                    temp_activity_correlation = 0.0
            except:
                temp_activity_correlation = 0.0
        
        return {
            'mean_temp': mean_temp,
            'temp_rise': temp_rise,
            'temp_rise_rate': temp_rise_rate,
            'activity_increase': activity_increase,
            'temp_activity_correlation': temp_activity_correlation,
            'sample_count': len(window_data)
        }
    
    def _calculate_pregnancy_metrics(
        self,
        temp_window: pd.DataFrame,
        activity_window: pd.DataFrame,
        baseline_temp: float,
        activity_baseline: Optional[Dict[str, float]],
        estrus_time: datetime
    ) -> Dict[str, Any]:
        """
        Calculate pregnancy detection metrics.
        
        Returns:
            Dictionary with temp_cv, activity_reduction, etc.
        """
        temps = temp_window['temperature'].values
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        temp_cv = std_temp / mean_temp if mean_temp > 0 else 0
        
        # Calculate activity reduction
        activity_reduction = 0.0
        if 'movement_intensity' in activity_window.columns:
            mean_activity = activity_window['movement_intensity'].mean()
            baseline_activity = activity_baseline.get('mean', 0.5) if activity_baseline else 0.5
            
            if baseline_activity > 0:
                activity_reduction = (baseline_activity - mean_activity) / baseline_activity
        
        # Calculate temperature stability (daily variance)
        daily_variance = 0.0
        if len(temp_window) > 1440:  # At least 1 day of data
            temp_window_copy = temp_window.copy()
            temp_window_copy['date'] = temp_window_copy['timestamp'].dt.date
            daily_means = temp_window_copy.groupby('date')['temperature'].mean()
            daily_variance = daily_means.var() if len(daily_means) > 1 else 0
        
        return {
            'mean_temp': mean_temp,
            'std_temp': std_temp,
            'temp_cv': temp_cv,
            'daily_variance': daily_variance,
            'activity_reduction': activity_reduction,
            'baseline_deviation': mean_temp - baseline_temp,
            'sample_count': len(temp_window)
        }
    
    def _is_estrus_pattern(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics indicate estrus pattern.
        
        Criteria:
        - Temperature rise between 0.3-0.6°C
        - Activity increase > 15%
        - Gradual temperature rise (not sudden fever)
        - Positive correlation between temp and activity
        """
        temp_rise = metrics['temp_rise']
        activity_increase = metrics['activity_increase']
        temp_rise_rate = metrics['temp_rise_rate']
        correlation = metrics['temp_activity_correlation']
        
        # Check temperature rise range
        if not (self.ESTRUS_TEMP_RISE_MIN <= temp_rise <= self.ESTRUS_TEMP_RISE_MAX):
            return False
        
        # Check activity increase
        if activity_increase < self.ESTRUS_ACTIVITY_INCREASE_MIN:
            return False
        
        # Check temperature rise rate (should be gradual, not sudden fever spike)
        if temp_rise_rate > self.ESTRUS_TEMP_RISE_RATE_MAX:
            logger.debug(f"Temperature rise too rapid for estrus: {temp_rise_rate:.3f}°C/min")
            return False
        
        # Check correlation (should be positive during estrus)
        if correlation < 0:
            logger.debug(f"Negative temp-activity correlation: {correlation:.2f}")
            return False
        
        return True
    
    def _is_pregnancy_pattern(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics indicate pregnancy pattern.
        
        Criteria:
        - Temperature stability (low CV)
        - Activity reduction > 10%
        """
        temp_cv = metrics['temp_cv']
        activity_reduction = metrics['activity_reduction']
        
        # Check temperature stability
        if temp_cv > self.PREGNANCY_TEMP_STABILITY_CV_MAX:
            return False
        
        # Check activity reduction
        if activity_reduction < self.PREGNANCY_ACTIVITY_REDUCTION_MIN:
            return False
        
        return True
    
    def _calculate_estrus_confidence(
        self,
        metrics: Dict[str, Any],
        window_data: pd.DataFrame
    ) -> float:
        """
        Calculate confidence score for estrus detection.
        
        Confidence increases with:
        - Pattern duration
        - Temperature rise within optimal range (0.4-0.5°C)
        - Strong temp-activity correlation
        - Consistent pattern
        """
        base_confidence = 0.7
        
        # Bonus for optimal temperature rise
        temp_rise = metrics['temp_rise']
        if 0.4 <= temp_rise <= 0.5:
            base_confidence += 0.1
        
        # Bonus for strong correlation
        correlation = metrics['temp_activity_correlation']
        if correlation > 0.5:
            base_confidence += 0.1
        
        # Bonus for pattern duration
        duration_minutes = len(window_data)
        if duration_minutes >= self.ESTRUS_CONFIRMATION_MINUTES:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _calculate_pregnancy_confidence(
        self,
        metrics: Dict[str, Any],
        window_days: int
    ) -> float:
        """
        Calculate confidence score for pregnancy detection.
        
        Confidence increases over time as pattern persists.
        """
        base_confidence = 0.5
        
        # Confidence increases with observation period
        if window_days >= 7:
            base_confidence += 0.1
        if window_days >= 10:
            base_confidence += 0.1
        if window_days >= 14:
            base_confidence += 0.1
        
        # Bonus for very stable temperature
        temp_cv = metrics['temp_cv']
        if temp_cv < 0.03:
            base_confidence += 0.1
        
        # Bonus for strong activity reduction
        activity_reduction = metrics['activity_reduction']
        if activity_reduction > 0.15:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def get_active_alerts(self, cow_id: Optional[str] = None) -> List[PatternAlert]:
        """
        Get active alerts for one or all cows.
        
        Args:
            cow_id: Specific cow ID or None for all cows
            
        Returns:
            List of active alerts
        """
        if cow_id:
            return self.active_alerts.get(cow_id, [])
        
        # Return all active alerts
        all_alerts = []
        for alerts in self.active_alerts.values():
            all_alerts.extend(alerts)
        return all_alerts
    
    def update_alert_status(self, alert_id: str, new_status: AlertStatus):
        """
        Update status of an existing alert.
        
        Args:
            alert_id: Alert identifier
            new_status: New status value
        """
        for cow_alerts in self.active_alerts.values():
            for alert in cow_alerts:
                if alert.alert_id == alert_id:
                    alert.status = new_status.value
                    logger.info(f"Alert {alert_id} status updated to {new_status.value}")
                    return
        
        logger.warning(f"Alert {alert_id} not found")
    
    def clear_resolved_alerts(self, cow_id: Optional[str] = None):
        """
        Remove resolved alerts from active tracking.
        
        Args:
            cow_id: Specific cow ID or None for all cows
        """
        if cow_id:
            if cow_id in self.active_alerts:
                self.active_alerts[cow_id] = [
                    a for a in self.active_alerts[cow_id]
                    if a.status != AlertStatus.RESOLVED.value
                ]
        else:
            for cow_id in self.active_alerts:
                self.active_alerts[cow_id] = [
                    a for a in self.active_alerts[cow_id]
                    if a.status != AlertStatus.RESOLVED.value
                ]
