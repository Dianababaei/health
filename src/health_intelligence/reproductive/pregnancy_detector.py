"""
Pregnancy Detection - Indicative Alerts

Provides early-stage indicative alerts for potential pregnancy.
This is a PoC-level detector that flags events for further observation.

Detection Criteria:
- Temperature stability: Low variance (< 0.15°C std dev)
- Activity reduction: Gradual decrease (5-15% reduction over weeks)
- Post-estrus timing: 21+ days after last estrus with no new estrus
- Duration: Sustained indicators for 14+ days

Note: These are indicative alerts, not final decisions.
Veterinary confirmation (ultrasound/blood test) required.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class PregnancyConfidence(Enum):
    """Confidence level for pregnancy indication"""
    LOW = "low"           # Weak indicators
    MEDIUM = "medium"     # Multiple indicators
    HIGH = "high"         # Strong indicators sustained


class PregnancyStatus(Enum):
    """Pregnancy status"""
    NOT_PREGNANT = "not_pregnant"
    POSSIBLY_PREGNANT = "possibly_pregnant"
    LIKELY_PREGNANT = "likely_pregnant"
    CONFIRMED_PREGNANT = "confirmed_pregnant"  # Requires vet confirmation


@dataclass
class PregnancyIndication:
    """
    Represents an indicative pregnancy detection (not confirmation).

    Attributes:
        timestamp: When indication was detected
        cow_id: Identifier for the cow
        status: Pregnancy status indication
        confidence: Confidence level
        days_since_estrus: Days since last estrus event
        temperature_stability: Temperature standard deviation
        activity_reduction: Activity reduction percentage
        indicators: Which indicators are present
        message: Human-readable description
        recommendation: Next steps for confirmation
    """
    timestamp: datetime
    cow_id: str
    status: PregnancyStatus
    confidence: PregnancyConfidence
    days_since_estrus: Optional[int]
    temperature_stability: float
    activity_reduction: float
    indicators: List[str]
    message: str
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cow_id': self.cow_id,
            'status': self.status.value,
            'confidence': self.confidence.value,
            'days_since_estrus': self.days_since_estrus,
            'temperature_stability': round(self.temperature_stability, 3),
            'activity_reduction': round(self.activity_reduction, 2),
            'indicators': self.indicators,
            'message': self.message,
            'recommendation': self.recommendation
        }


class PregnancyDetector:
    """
    Pregnancy indication using temperature stability and activity patterns.

    This is an indicative detector for PoC/research phase.
    It provides early indications that warrant veterinary confirmation.

    Usage:
        detector = PregnancyDetector()
        indication = detector.detect_pregnancy(
            cow_id='COW_001',
            temperature_data=temp_df,
            activity_data=activity_df,
            last_estrus_date=datetime(2025, 10, 1)
        )
    """

    def __init__(
        self,
        temp_stability_threshold: float = 0.15,  # Low std dev indicates stability
        activity_reduction_min: float = 0.05,    # 5% reduction
        activity_reduction_max: float = 0.15,    # 15% reduction
        min_days_post_estrus: int = 21,
        min_sustained_days: int = 14
    ):
        """
        Initialize pregnancy detector.

        Args:
            temp_stability_threshold: Max std dev for stable temperature
            activity_reduction_min: Minimum activity reduction to consider
            activity_reduction_max: Maximum activity reduction to consider
            min_days_post_estrus: Minimum days after estrus to check
            min_sustained_days: Minimum days indicators must be sustained
        """
        self.temp_stability_threshold = temp_stability_threshold
        self.activity_reduction_min = activity_reduction_min
        self.activity_reduction_max = activity_reduction_max
        self.min_days_post_estrus = min_days_post_estrus
        self.min_sustained_days = min_sustained_days

    def detect_pregnancy(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        last_estrus_date: Optional[datetime] = None,
        lookback_days: int = 30
    ) -> Optional[PregnancyIndication]:
        """
        Detect potential pregnancy indicators.

        Args:
            cow_id: Cow identifier
            temperature_data: DataFrame with ['timestamp', 'temperature']
            activity_data: DataFrame with ['timestamp', 'movement_intensity' or 'fxa']
            last_estrus_date: Date of last estrus event (if known)
            lookback_days: How many days to analyze

        Returns:
            PregnancyIndication if indicators present, None otherwise
        """
        if len(temperature_data) == 0 or len(activity_data) == 0:
            return None

        temp_df = temperature_data.copy()
        activity_df = activity_data.copy()

        # Ensure datetime
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])

        # Filter to analysis window
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        temp_df = temp_df[temp_df['timestamp'] >= cutoff_time]
        activity_df = activity_df[activity_df['timestamp'] >= cutoff_time]

        if len(temp_df) < self.min_sustained_days * 24 * 60:  # Need enough data
            return None

        # Calculate days since estrus
        days_since_estrus = None
        if last_estrus_date:
            days_since_estrus = (datetime.now() - last_estrus_date).days

            # Too soon to detect pregnancy
            if days_since_estrus < self.min_days_post_estrus:
                return None

        # Analyze temperature stability
        recent_temp = temp_df.tail(self.min_sustained_days * 24 * 60)
        temp_std = recent_temp['temperature'].std()
        temp_mean = recent_temp['temperature'].mean()

        is_temp_stable = temp_std < self.temp_stability_threshold

        # Analyze activity reduction
        # Compare recent to baseline (before potential pregnancy)
        if last_estrus_date:
            baseline_start = last_estrus_date - timedelta(days=14)
            baseline_end = last_estrus_date
        else:
            baseline_end = datetime.now() - timedelta(days=lookback_days // 2)
            baseline_start = baseline_end - timedelta(days=14)

        # Get activity column
        activity_col = None
        for col in ['movement_intensity', 'fxa', 'activity']:
            if col in activity_df.columns:
                activity_col = col
                break

        if activity_col is None:
            return None

        baseline_activity = activity_df[
            (activity_df['timestamp'] >= baseline_start) &
            (activity_df['timestamp'] <= baseline_end)
        ]

        recent_activity = activity_df.tail(self.min_sustained_days * 24 * 60)

        if len(baseline_activity) == 0 or len(recent_activity) == 0:
            return None

        baseline_avg = baseline_activity[activity_col].mean()
        recent_avg = recent_activity[activity_col].mean()

        activity_change = (baseline_avg - recent_avg) / baseline_avg
        is_activity_reduced = (
            self.activity_reduction_min <= activity_change <= self.activity_reduction_max
        )

        # Check for no new estrus events (no temp spikes)
        # Look for temperature spikes that would indicate estrus
        temp_df['temp_diff'] = temp_df['temperature'].diff()
        temp_spikes = temp_df[temp_df['temp_diff'] > 0.3]  # Estrus-like spike

        no_recent_estrus = len(temp_spikes) == 0

        # Collect indicators
        indicators = []
        if is_temp_stable:
            indicators.append('temperature_stability')
        if is_activity_reduced:
            indicators.append('activity_reduction')
        if no_recent_estrus:
            indicators.append('no_estrus_return')
        if days_since_estrus and days_since_estrus >= self.min_days_post_estrus:
            indicators.append('post_estrus_timing')

        # Determine status and confidence
        if len(indicators) == 0:
            return None

        if len(indicators) >= 3:
            status = PregnancyStatus.LIKELY_PREGNANT
            confidence = PregnancyConfidence.HIGH
            message = "Strong indicators of pregnancy detected. Veterinary confirmation recommended."
            recommendation = "Schedule ultrasound or blood test for confirmation within 1-2 weeks."
        elif len(indicators) == 2:
            status = PregnancyStatus.POSSIBLY_PREGNANT
            confidence = PregnancyConfidence.MEDIUM
            message = "Moderate indicators of pregnancy detected. Continue monitoring."
            recommendation = "Monitor for 7 more days, then consider veterinary confirmation."
        else:
            status = PregnancyStatus.POSSIBLY_PREGNANT
            confidence = PregnancyConfidence.LOW
            message = "Weak indicators of pregnancy detected. Observation recommended."
            recommendation = "Continue daily monitoring. Schedule check if indicators strengthen."

        # Create indication
        indication = PregnancyIndication(
            timestamp=datetime.now(),
            cow_id=cow_id,
            status=status,
            confidence=confidence,
            days_since_estrus=days_since_estrus,
            temperature_stability=temp_std,
            activity_reduction=activity_change * 100,  # Convert to percentage
            indicators=indicators,
            message=message,
            recommendation=recommendation
        )

        return indication

    def monitor_pregnancy_progression(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        pregnancy_start_date: datetime,
        current_day: int
    ) -> Dict[str, Any]:
        """
        Monitor ongoing pregnancy progression.

        Args:
            cow_id: Cow identifier
            temperature_data: Temperature data
            activity_data: Activity data
            pregnancy_start_date: When pregnancy was confirmed/indicated
            current_day: Current day of pregnancy

        Returns:
            Dictionary with progression metrics
        """
        days_pregnant = (datetime.now() - pregnancy_start_date).days

        # Expected activity reduction increases with pregnancy progression
        # Early: 5-10%, Mid: 10-15%, Late: 15-25%
        if days_pregnant < 90:
            stage = "early"
            expected_reduction = (0.05, 0.10)
        elif days_pregnant < 180:
            stage = "mid"
            expected_reduction = (0.10, 0.15)
        else:
            stage = "late"
            expected_reduction = (0.15, 0.25)

        # Get recent activity
        activity_col = None
        for col in ['movement_intensity', 'fxa', 'activity']:
            if col in activity_data.columns:
                activity_col = col
                break

        if activity_col is None:
            return {}

        recent_activity = activity_data.tail(7 * 24 * 60)  # Last 7 days
        baseline_before_pregnancy = activity_data[
            (activity_data['timestamp'] < pregnancy_start_date)
        ].tail(7 * 24 * 60)

        if len(baseline_before_pregnancy) == 0 or len(recent_activity) == 0:
            return {}

        baseline_avg = baseline_before_pregnancy[activity_col].mean()
        recent_avg = recent_activity[activity_col].mean()
        actual_reduction = (baseline_avg - recent_avg) / baseline_avg

        # Check if progression is normal
        is_normal = expected_reduction[0] <= actual_reduction <= expected_reduction[1]

        return {
            'cow_id': cow_id,
            'days_pregnant': days_pregnant,
            'stage': stage,
            'expected_reduction_min': expected_reduction[0] * 100,
            'expected_reduction_max': expected_reduction[1] * 100,
            'actual_reduction': actual_reduction * 100,
            'is_normal_progression': is_normal,
            'message': f"Pregnancy day {days_pregnant} ({stage} stage): "
                      f"Activity reduction {actual_reduction*100:.1f}% "
                      f"({'normal' if is_normal else 'abnormal'} for stage)"
        }
