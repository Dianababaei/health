"""
Estrus (Heat) Detection - Indicative Alerts

Provides early-stage indicative alerts for potential estrus events.
This is a PoC-level detector that flags events for further observation.

Detection Criteria:
- Temperature rise: 0.3-0.6°C above baseline
- Activity increase: 20-50% above normal
- Duration: 6-24 hours
- Typical cycle: 21 days

Note: These are indicative alerts, not final decisions.
Veterinary or manual confirmation recommended.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class EstrusConfidence(Enum):
    """Confidence level for estrus detection"""
    LOW = "low"           # Single indicator present
    MEDIUM = "medium"     # Two indicators present
    HIGH = "high"         # All indicators present


@dataclass
class EstrusEvent:
    """
    Represents a detected estrus event (indicative alert).

    Attributes:
        timestamp: When the event was detected
        cow_id: Identifier for the cow
        temperature_rise: Temperature increase above baseline (°C)
        activity_increase: Activity increase above baseline (%)
        duration_hours: How long indicators were present
        confidence: Confidence level of detection
        indicators: Which indicators triggered (temp, activity, duration)
        message: Human-readable description
    """
    timestamp: datetime
    cow_id: str
    temperature_rise: float
    activity_increase: float
    duration_hours: float
    confidence: EstrusConfidence
    indicators: List[str]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cow_id': self.cow_id,
            'temperature_rise': round(self.temperature_rise, 2),
            'activity_increase': round(self.activity_increase, 2),
            'duration_hours': round(self.duration_hours, 1),
            'confidence': self.confidence.value,
            'indicators': self.indicators,
            'message': self.message
        }


class EstrusDetector:
    """
    Estrus detection using temperature and activity patterns.

    This is an indicative detector for PoC/research phase.
    It flags potential estrus events for further observation.

    Usage:
        detector = EstrusDetector(baseline_temp=38.5)
        events = detector.detect_estrus(
            cow_id='COW_001',
            temperature_data=temp_df,
            activity_data=activity_df,
            lookback_hours=48
        )
    """

    def __init__(
        self,
        baseline_temp: float = 38.5,
        temp_rise_min: float = 0.3,
        temp_rise_max: float = 0.6,
        activity_increase_min: float = 0.20,  # 20%
        activity_increase_max: float = 0.50,  # 50%
        min_duration_hours: int = 6,
        max_duration_hours: int = 24
    ):
        """
        Initialize estrus detector.

        Args:
            baseline_temp: Normal body temperature baseline
            temp_rise_min: Minimum temperature rise to consider (°C)
            temp_rise_max: Maximum temperature rise to consider (°C)
            activity_increase_min: Minimum activity increase (fraction)
            activity_increase_max: Maximum activity increase (fraction)
            min_duration_hours: Minimum duration for event
            max_duration_hours: Maximum duration for event
        """
        self.baseline_temp = baseline_temp
        self.temp_rise_min = temp_rise_min
        self.temp_rise_max = temp_rise_max
        self.activity_increase_min = activity_increase_min
        self.activity_increase_max = activity_increase_max
        self.min_duration_hours = min_duration_hours
        self.max_duration_hours = max_duration_hours

    def detect_estrus(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        lookback_hours: int = 48
    ) -> List[EstrusEvent]:
        """
        Detect potential estrus events in recent data.

        Args:
            cow_id: Cow identifier
            temperature_data: DataFrame with columns ['timestamp', 'temperature']
            activity_data: DataFrame with columns ['timestamp', 'movement_intensity' or 'fxa']
            lookback_hours: How far back to search for events

        Returns:
            List of detected estrus events (may be empty)
        """
        if len(temperature_data) == 0 or len(activity_data) == 0:
            return []

        # Ensure timestamp column
        if 'timestamp' not in temperature_data.columns:
            return []

        temp_df = temperature_data.copy()
        activity_df = activity_data.copy()

        # Ensure datetime
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])
        activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])

        # Filter to lookback window
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        temp_df = temp_df[temp_df['timestamp'] >= cutoff_time]
        activity_df = activity_df[activity_df['timestamp'] >= cutoff_time]

        if len(temp_df) == 0 or len(activity_df) == 0:
            return []

        # Calculate baseline from older data (before lookback window)
        baseline_cutoff = datetime.now() - timedelta(hours=lookback_hours + 48)
        historical_temp = temperature_data[
            temperature_data['timestamp'] < baseline_cutoff
        ]

        if len(historical_temp) > 0:
            baseline_temp = historical_temp['temperature'].mean()
        else:
            baseline_temp = self.baseline_temp

        # Calculate activity baseline
        historical_activity = activity_data[
            activity_data['timestamp'] < baseline_cutoff
        ]

        # Get activity column
        activity_col = None
        for col in ['movement_intensity', 'fxa', 'activity']:
            if col in activity_df.columns:
                activity_col = col
                break

        if activity_col is None:
            return []

        if len(historical_activity) > 0 and activity_col in historical_activity.columns:
            baseline_activity = historical_activity[activity_col].mean()
        else:
            baseline_activity = activity_df[activity_col].mean()

        # Detect temperature rises
        temp_df['temp_rise'] = temp_df['temperature'] - baseline_temp
        temp_df['is_elevated'] = (
            (temp_df['temp_rise'] >= self.temp_rise_min) &
            (temp_df['temp_rise'] <= self.temp_rise_max)
        )

        # Detect activity increases
        activity_df['activity_pct_increase'] = (
            (activity_df[activity_col] - baseline_activity) / baseline_activity
        )
        activity_df['is_increased'] = (
            (activity_df['activity_pct_increase'] >= self.activity_increase_min) &
            (activity_df['activity_pct_increase'] <= self.activity_increase_max)
        )

        # Merge on timestamp (approximate)
        merged = pd.merge_asof(
            temp_df.sort_values('timestamp'),
            activity_df[['timestamp', 'activity_pct_increase', 'is_increased']].sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta(minutes=5)
        )

        # Find continuous periods where indicators are present
        events = []

        # Look for periods where both temp and activity are elevated
        merged['both_elevated'] = merged['is_elevated'] & merged['is_increased']

        if merged['both_elevated'].any():
            # Find continuous periods
            merged['event_group'] = (
                merged['both_elevated'] != merged['both_elevated'].shift()
            ).cumsum()

            event_periods = merged[merged['both_elevated']].groupby('event_group')

            for group_id, group in event_periods:
                if len(group) == 0:
                    continue

                # Calculate duration
                start_time = group['timestamp'].min()
                end_time = group['timestamp'].max()
                duration = (end_time - start_time).total_seconds() / 3600

                # Only consider if duration is reasonable
                if duration < self.min_duration_hours or duration > self.max_duration_hours:
                    continue

                # Calculate average indicators
                avg_temp_rise = group['temp_rise'].mean()
                avg_activity_increase = group['activity_pct_increase'].mean()

                # Determine confidence
                indicators = []
                if group['is_elevated'].mean() > 0.7:
                    indicators.append('temperature_rise')
                if group['is_increased'].mean() > 0.7:
                    indicators.append('activity_increase')
                if self.min_duration_hours <= duration <= self.max_duration_hours:
                    indicators.append('duration')

                if len(indicators) >= 3:
                    confidence = EstrusConfidence.HIGH
                elif len(indicators) == 2:
                    confidence = EstrusConfidence.MEDIUM
                else:
                    confidence = EstrusConfidence.LOW

                # Create event
                event = EstrusEvent(
                    timestamp=start_time,
                    cow_id=cow_id,
                    temperature_rise=avg_temp_rise,
                    activity_increase=avg_activity_increase * 100,  # Convert to percentage
                    duration_hours=duration,
                    confidence=confidence,
                    indicators=indicators,
                    message=f"Indicative estrus event detected: {confidence.value} confidence. "
                            f"Recommend observation for breeding readiness."
                )

                events.append(event)

        return events

    def predict_next_estrus(
        self,
        cow_id: str,
        last_estrus_date: datetime,
        cycle_days: int = 21
    ) -> datetime:
        """
        Predict next estrus based on typical 21-day cycle.

        Args:
            cow_id: Cow identifier
            last_estrus_date: Date of last detected estrus
            cycle_days: Typical cycle length (default 21 days)

        Returns:
            Predicted date of next estrus
        """
        return last_estrus_date + timedelta(days=cycle_days)

    def get_breeding_window(
        self,
        estrus_event: EstrusEvent
    ) -> tuple:
        """
        Get optimal breeding window for detected estrus event.

        Args:
            estrus_event: Detected estrus event

        Returns:
            Tuple of (start_time, end_time) for breeding window
        """
        # Optimal breeding is typically 12-18 hours after onset
        start = estrus_event.timestamp + timedelta(hours=12)
        end = estrus_event.timestamp + timedelta(hours=18)

        return (start, end)
