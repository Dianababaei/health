"""
Temperature-Activity Correlation Engine

Combines temperature data from Layer 2 with behavioral states from Layer 1
to identify health patterns including fever and heat stress.

Fever Pattern: Elevated temperature (>39.0°C) with reduced motion
Heat Stress Pattern: High temperature (>39.5°C) with elevated activity
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats

logger = logging.getLogger(__name__)


class HealthPattern(Enum):
    """Types of health patterns detected."""
    NORMAL = "normal"
    FEVER = "fever"
    HEAT_STRESS = "heat_stress"
    UNKNOWN = "unknown"


@dataclass
class CorrelationEvent:
    """Represents a detected temperature-activity correlation event."""
    timestamp: datetime
    pattern_type: HealthPattern
    confidence: float  # 0-100
    temperature: float
    activity_level: float
    behavioral_state: str
    duration_minutes: float
    correlation_coefficient: Optional[float] = None
    contributing_factors: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else str(self.timestamp),
            'pattern_type': self.pattern_type.value,
            'confidence': self.confidence,
            'temperature': self.temperature,
            'activity_level': self.activity_level,
            'behavioral_state': self.behavioral_state,
            'duration_minutes': self.duration_minutes,
            'correlation_coefficient': self.correlation_coefficient,
            'contributing_factors': self.contributing_factors
        }


class TemperatureActivityCorrelator:
    """
    Correlates temperature data with behavioral activity to detect health patterns.
    
    Features:
    - Time-aligned merging of temperature and behavioral data
    - Fever pattern detection (high temp + low activity)
    - Heat stress pattern detection (high temp + high activity)
    - Pearson correlation analysis
    - Lag analysis
    - Pattern confidence scoring
    """
    
    # Behavioral state activity levels (relative scale 0-1)
    ACTIVITY_LEVELS = {
        'lying': 0.1,
        'standing': 0.3,
        'walking': 0.8,
        'feeding': 0.5,
        'ruminating': 0.2,
        'transition': 0.4,
        'uncertain': 0.3
    }
    
    # Rest vs active state classification
    REST_STATES = ['lying']
    ACTIVE_STATES = ['walking', 'standing', 'feeding', 'ruminating']
    
    def __init__(
        self,
        fever_temp_threshold: float = 39.0,
        heat_stress_temp_threshold: float = 39.5,
        reduced_motion_threshold: float = 0.2,
        elevated_activity_threshold: float = 0.5,
        time_alignment_tolerance_minutes: int = 2,
        min_pattern_duration_minutes: int = 30
    ):
        """
        Initialize correlator.
        
        Args:
            fever_temp_threshold: Temperature threshold for fever detection (°C)
            heat_stress_temp_threshold: Temperature threshold for heat stress (°C)
            reduced_motion_threshold: Activity level below which motion is "reduced"
            elevated_activity_threshold: Activity level above which activity is "elevated"
            time_alignment_tolerance_minutes: Max time difference for merging data
            min_pattern_duration_minutes: Minimum duration for pattern detection
        """
        self.fever_temp_threshold = fever_temp_threshold
        self.heat_stress_temp_threshold = heat_stress_temp_threshold
        self.reduced_motion_threshold = reduced_motion_threshold
        self.elevated_activity_threshold = elevated_activity_threshold
        self.time_alignment_tolerance_minutes = time_alignment_tolerance_minutes
        self.min_pattern_duration_minutes = min_pattern_duration_minutes
        
        logger.info(f"TemperatureActivityCorrelator initialized: "
                   f"fever_threshold={fever_temp_threshold}°C, "
                   f"heat_stress_threshold={heat_stress_temp_threshold}°C")
    
    def load_behavioral_data(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        state_col: str = 'behavioral_state',
        intensity_col: Optional[str] = 'movement_intensity'
    ) -> pd.DataFrame:
        """
        Load and prepare behavioral data from Layer 1 output.
        
        Args:
            data: DataFrame with behavioral state data
            timestamp_col: Name of timestamp column
            state_col: Name of behavioral state column
            intensity_col: Name of movement intensity column (optional)
            
        Returns:
            Prepared DataFrame with activity levels
        """
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Map behavioral states to activity levels
        df['activity_level'] = df[state_col].map(self.ACTIVITY_LEVELS).fillna(0.3)
        
        # If movement intensity is available, use it to adjust activity level
        if intensity_col and intensity_col in df.columns:
            # Normalize intensity to 0-1 scale
            intensity_max = df[intensity_col].quantile(0.95)  # Use 95th percentile as max
            df['normalized_intensity'] = (df[intensity_col] / intensity_max).clip(0, 1)
            
            # Combine state-based and intensity-based activity
            df['activity_level'] = (df['activity_level'] * 0.6 + 
                                   df['normalized_intensity'] * 0.4)
        
        # Classify as rest or active
        df['is_rest'] = df[state_col].isin(self.REST_STATES)
        df['is_active'] = df[state_col].isin(self.ACTIVE_STATES)
        
        return df
    
    def merge_temperature_activity(
        self,
        temperature_data: pd.DataFrame,
        behavioral_data: pd.DataFrame,
        temp_timestamp_col: str = 'timestamp',
        temp_value_col: str = 'temperature',
        behavior_timestamp_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Time-aligned merge of temperature and behavioral data.
        
        Handles asynchronous data with tolerance for 1-2 minute lag.
        
        Args:
            temperature_data: DataFrame with temperature readings
            behavioral_data: DataFrame with behavioral states
            temp_timestamp_col: Timestamp column in temperature data
            temp_value_col: Temperature value column
            behavior_timestamp_col: Timestamp column in behavioral data
            
        Returns:
            Merged DataFrame with both temperature and behavioral data
        """
        temp_df = temperature_data.copy()
        behavior_df = behavioral_data.copy()
        
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(temp_df[temp_timestamp_col]):
            temp_df[temp_timestamp_col] = pd.to_datetime(temp_df[temp_timestamp_col])
        if not pd.api.types.is_datetime64_any_dtype(behavior_df[behavior_timestamp_col]):
            behavior_df[behavior_timestamp_col] = pd.to_datetime(behavior_df[behavior_timestamp_col])
        
        # Use merge_asof for time-aligned merge with tolerance
        tolerance = pd.Timedelta(minutes=self.time_alignment_tolerance_minutes)
        
        merged = pd.merge_asof(
            temp_df.sort_values(temp_timestamp_col),
            behavior_df.sort_values(behavior_timestamp_col),
            left_on=temp_timestamp_col,
            right_on=behavior_timestamp_col,
            direction='nearest',
            tolerance=tolerance,
            suffixes=('_temp', '_behavior')
        )
        
        # Drop rows where merge failed (no behavioral data within tolerance)
        merged = merged.dropna(subset=['behavioral_state'])
        
        logger.info(f"Merged {len(merged)} time-aligned records from "
                   f"{len(temp_df)} temperature and {len(behavior_df)} behavioral readings")
        
        return merged
    
    def calculate_baseline_activity(
        self,
        data: pd.DataFrame,
        window_hours: int = 24,
        activity_col: str = 'activity_level'
    ) -> pd.DataFrame:
        """
        Calculate individual baseline activity levels.
        
        Args:
            data: DataFrame with activity data
            window_hours: Rolling window size in hours
            activity_col: Name of activity level column
            
        Returns:
            DataFrame with baseline activity columns
        """
        df = data.copy()
        
        # Ensure we have a datetime index for rolling
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        window_size = f'{window_hours}H'
        
        # Calculate rolling statistics
        df['baseline_activity'] = df[activity_col].rolling(
            window=window_size,
            min_periods=30
        ).median()
        
        df['activity_percentile_25'] = df[activity_col].rolling(
            window=window_size,
            min_periods=30
        ).quantile(0.25)
        
        df['activity_percentile_75'] = df[activity_col].rolling(
            window=window_size,
            min_periods=30
        ).quantile(0.75)
        
        df = df.reset_index()
        
        return df
    
    def detect_fever_pattern(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        activity_col: str = 'activity_level',
        state_col: str = 'behavioral_state',
        duration_col: Optional[str] = 'duration_minutes'
    ) -> pd.DataFrame:
        """
        Detect fever patterns: elevated temperature with reduced motion.
        
        Criteria:
        - Temperature > fever_temp_threshold (default 39.0°C)
        - Reduced motion: lying >60 min OR activity < 20% baseline OR bottom 25th percentile
        
        Args:
            data: Merged temperature-activity DataFrame
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            activity_col: Name of activity level column
            state_col: Name of behavioral state column
            duration_col: Name of duration column (optional)
            
        Returns:
            DataFrame with fever_pattern column (boolean)
        """
        df = data.copy()
        
        # Initialize fever pattern column
        df['fever_pattern'] = False
        df['fever_confidence'] = 0.0
        
        # Check temperature threshold
        high_temp = df[temperature_col] > self.fever_temp_threshold
        
        # Check for reduced motion (multiple criteria)
        reduced_motion = pd.Series(False, index=df.index)
        
        # Criterion 1: Lying state with duration > 60 minutes
        if duration_col and duration_col in df.columns:
            long_lying = (df[state_col] == 'lying') & (df[duration_col] > 60)
            reduced_motion |= long_lying
        
        # Criterion 2: Activity level < 20% of baseline
        if 'baseline_activity' in df.columns:
            low_activity = df[activity_col] < (df['baseline_activity'] * 0.2)
            reduced_motion |= low_activity
        
        # Criterion 3: Activity in bottom 25th percentile
        if 'activity_percentile_25' in df.columns:
            bottom_quartile = df[activity_col] <= df['activity_percentile_25']
            reduced_motion |= bottom_quartile
        else:
            # Fallback: use absolute threshold
            reduced_motion |= df[activity_col] < self.reduced_motion_threshold
        
        # Combine conditions
        df['fever_pattern'] = high_temp & reduced_motion
        
        # Calculate confidence for fever patterns
        fever_mask = df['fever_pattern']
        if fever_mask.any():
            # Confidence based on:
            # 1. Temperature elevation above threshold
            # 2. Degree of activity reduction
            # 3. Time of day context (fever less likely during normal sleep hours)
            
            temp_excess = (df.loc[fever_mask, temperature_col] - self.fever_temp_threshold).clip(0, 2)
            temp_confidence = (temp_excess / 2.0 * 40).clip(0, 40)  # Max 40 points
            
            activity_reduction = 1.0 - df.loc[fever_mask, activity_col]
            activity_confidence = (activity_reduction * 40).clip(0, 40)  # Max 40 points
            
            # Time of day factor: reduce confidence during night (22:00-06:00)
            if timestamp_col in df.columns:
                hour = pd.to_datetime(df.loc[fever_mask, timestamp_col]).dt.hour
                time_factor = pd.Series(1.0, index=hour.index)
                night_hours = (hour >= 22) | (hour < 6)
                time_factor[night_hours] = 0.5  # 50% confidence reduction at night
                time_confidence = time_factor * 20  # Max 20 points
            else:
                time_confidence = 20
            
            df.loc[fever_mask, 'fever_confidence'] = (
                temp_confidence + activity_confidence + time_confidence
            ).clip(0, 100)
        
        logger.info(f"Detected {fever_mask.sum()} fever pattern instances")
        
        return df
    
    def detect_heat_stress_pattern(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        activity_col: str = 'activity_level',
        state_col: str = 'behavioral_state',
        transitions_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect heat stress patterns: high temperature with elevated activity.
        
        Criteria:
        - Temperature > heat_stress_temp_threshold (default 39.5°C)
        - Elevated activity: walking with high intensity OR top 50th percentile OR frequent transitions
        
        Args:
            data: Merged temperature-activity DataFrame
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            activity_col: Name of activity level column
            state_col: Name of behavioral state column
            transitions_col: Name of state transitions column (optional)
            
        Returns:
            DataFrame with heat_stress_pattern column (boolean)
        """
        df = data.copy()
        
        # Initialize heat stress pattern column
        df['heat_stress_pattern'] = False
        df['heat_stress_confidence'] = 0.0
        
        # Check temperature threshold
        high_temp = df[temperature_col] > self.heat_stress_temp_threshold
        
        # Check for elevated activity (multiple criteria)
        elevated_activity = pd.Series(False, index=df.index)
        
        # Criterion 1: Walking state with activity above baseline
        if 'baseline_activity' in df.columns:
            active_walking = (df[state_col] == 'walking') & (df[activity_col] > df['baseline_activity'])
            elevated_activity |= active_walking
        
        # Criterion 2: Activity in top 50th percentile
        activity_median = df[activity_col].median()
        elevated_activity |= df[activity_col] >= activity_median
        
        # Criterion 3: Use absolute threshold as fallback
        elevated_activity |= df[activity_col] > self.elevated_activity_threshold
        
        # Combine conditions
        df['heat_stress_pattern'] = high_temp & elevated_activity
        
        # Calculate confidence for heat stress patterns
        stress_mask = df['heat_stress_pattern']
        if stress_mask.any():
            # Confidence based on:
            # 1. Temperature elevation above threshold
            # 2. Degree of activity elevation
            # 3. Time of day context (heat stress more likely during day)
            
            temp_excess = (df.loc[stress_mask, temperature_col] - self.heat_stress_temp_threshold).clip(0, 2)
            temp_confidence = (temp_excess / 2.0 * 40).clip(0, 40)  # Max 40 points
            
            activity_elevation = df.loc[stress_mask, activity_col]
            activity_confidence = (activity_elevation * 40).clip(0, 40)  # Max 40 points
            
            # Time of day factor: increase confidence during day (08:00-20:00)
            if timestamp_col in df.columns:
                hour = pd.to_datetime(df.loc[stress_mask, timestamp_col]).dt.hour
                time_factor = pd.Series(0.5, index=hour.index)  # Base 50% confidence
                day_hours = (hour >= 8) & (hour < 20)
                time_factor[day_hours] = 1.0  # Full confidence during day
                time_confidence = time_factor * 20  # Max 20 points
            else:
                time_confidence = 15
            
            df.loc[stress_mask, 'heat_stress_confidence'] = (
                temp_confidence + activity_confidence + time_confidence
            ).clip(0, 100)
        
        logger.info(f"Detected {stress_mask.sum()} heat stress pattern instances")
        
        return df
    
    def calculate_correlation_metrics(
        self,
        data: pd.DataFrame,
        temperature_col: str = 'temperature',
        activity_col: str = 'activity_level',
        window_hours: List[int] = [1, 4, 24]
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate Pearson correlation coefficients over rolling windows.
        
        Args:
            data: Merged temperature-activity DataFrame
            temperature_col: Name of temperature column
            activity_col: Name of activity level column
            window_hours: List of window sizes in hours
            
        Returns:
            Dictionary mapping window size to DataFrame with correlation columns
        """
        df = data.copy()
        
        # Ensure we have a datetime index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        
        results = {}
        
        for window_h in window_hours:
            window_size = f'{window_h}H'
            col_name = f'correlation_{window_h}h'
            
            # Calculate rolling correlation
            df[col_name] = df[temperature_col].rolling(
                window=window_size,
                min_periods=10
            ).corr(df[activity_col])
            
            results[f'{window_h}h'] = df.reset_index()
        
        logger.info(f"Calculated correlations for windows: {window_hours}")
        
        return results
    
    def calculate_lag_analysis(
        self,
        data: pd.DataFrame,
        temperature_col: str = 'temperature',
        activity_col: str = 'activity_level',
        max_lag_minutes: int = 30
    ) -> Dict[str, float]:
        """
        Calculate lag analysis: does activity change precede or follow temperature?
        
        Args:
            data: Merged temperature-activity DataFrame
            temperature_col: Name of temperature column
            activity_col: Name of activity level column
            max_lag_minutes: Maximum lag to test (minutes)
            
        Returns:
            Dictionary with lag analysis results
        """
        df = data.copy()
        
        # Calculate correlations at different lags
        lag_correlations = {}
        
        for lag in range(-max_lag_minutes, max_lag_minutes + 1, 5):
            if lag == 0:
                corr = df[temperature_col].corr(df[activity_col])
            elif lag > 0:
                # Positive lag: temperature leads activity
                corr = df[temperature_col].corr(df[activity_col].shift(lag))
            else:
                # Negative lag: activity leads temperature
                corr = df[temperature_col].shift(-lag).corr(df[activity_col])
            
            lag_correlations[lag] = corr if not np.isnan(corr) else 0.0
        
        # Find lag with strongest correlation
        best_lag = max(lag_correlations.items(), key=lambda x: abs(x[1]))
        
        return {
            'best_lag_minutes': best_lag[0],
            'best_correlation': best_lag[1],
            'lag_correlations': lag_correlations
        }
    
    def calculate_pattern_confidence(
        self,
        temperature: float,
        activity_level: float,
        correlation: float,
        duration_minutes: float,
        pattern_type: HealthPattern
    ) -> float:
        """
        Calculate overall pattern confidence score (0-100).
        
        Combines:
        - Threshold exceedance strength
        - Correlation strength
        - Pattern duration
        
        Args:
            temperature: Current temperature (°C)
            activity_level: Current activity level (0-1)
            correlation: Correlation coefficient
            duration_minutes: Pattern duration
            pattern_type: Type of pattern
            
        Returns:
            Confidence score (0-100)
        """
        score = 0.0
        
        if pattern_type == HealthPattern.FEVER:
            # Temperature component (max 40 points)
            temp_excess = max(0, temperature - self.fever_temp_threshold)
            score += min(40, temp_excess / 2.0 * 40)
            
            # Activity component (max 30 points)
            activity_reduction = 1.0 - activity_level
            score += activity_reduction * 30
            
        elif pattern_type == HealthPattern.HEAT_STRESS:
            # Temperature component (max 40 points)
            temp_excess = max(0, temperature - self.heat_stress_temp_threshold)
            score += min(40, temp_excess / 2.0 * 40)
            
            # Activity component (max 30 points)
            score += activity_level * 30
        
        # Correlation component (max 15 points)
        if correlation is not None:
            score += abs(correlation) * 15
        
        # Duration component (max 15 points)
        # Longer patterns get higher confidence
        duration_factor = min(1.0, duration_minutes / 120.0)  # Cap at 2 hours
        score += duration_factor * 15
        
        return min(100, max(0, score))
    
    def generate_correlation_events(
        self,
        data: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        temperature_col: str = 'temperature',
        activity_col: str = 'activity_level',
        state_col: str = 'behavioral_state'
    ) -> List[CorrelationEvent]:
        """
        Generate structured correlation events from detected patterns.
        
        Args:
            data: DataFrame with detected patterns
            timestamp_col: Name of timestamp column
            temperature_col: Name of temperature column
            activity_col: Name of activity level column
            state_col: Name of behavioral state column
            
        Returns:
            List of CorrelationEvent objects
        """
        df = data.copy()
        events = []
        
        # Process fever patterns
        if 'fever_pattern' in df.columns:
            fever_data = df[df['fever_pattern']].copy()
            
            # Group consecutive fever patterns
            if len(fever_data) > 0:
                fever_data = fever_data.sort_values(timestamp_col).reset_index(drop=True)
                fever_data['group_changed'] = (
                    (fever_data[timestamp_col].diff() > pd.Timedelta(minutes=5)) |
                    (fever_data[timestamp_col].diff().isna())
                )
                fever_data['group_id'] = fever_data['group_changed'].cumsum()
                
                for group_id, group in fever_data.groupby('group_id'):
                    if len(group) < self.min_pattern_duration_minutes / 5:  # Assume 5-min sampling
                        continue
                    
                    duration = (group[timestamp_col].iloc[-1] - 
                              group[timestamp_col].iloc[0]).total_seconds() / 60.0
                    
                    event = CorrelationEvent(
                        timestamp=group[timestamp_col].iloc[0],
                        pattern_type=HealthPattern.FEVER,
                        confidence=float(group['fever_confidence'].mean()),
                        temperature=float(group[temperature_col].mean()),
                        activity_level=float(group[activity_col].mean()),
                        behavioral_state=group[state_col].mode()[0] if len(group[state_col].mode()) > 0 else group[state_col].iloc[0],
                        duration_minutes=duration,
                        correlation_coefficient=float(group['correlation_1h'].mean()) if 'correlation_1h' in group.columns else None,
                        contributing_factors={
                            'avg_temperature': float(group[temperature_col].mean()),
                            'max_temperature': float(group[temperature_col].max()),
                            'avg_activity': float(group[activity_col].mean()),
                            'min_activity': float(group[activity_col].min()),
                            'sample_count': len(group)
                        }
                    )
                    events.append(event)
        
        # Process heat stress patterns
        if 'heat_stress_pattern' in df.columns:
            stress_data = df[df['heat_stress_pattern']].copy()
            
            # Group consecutive heat stress patterns
            if len(stress_data) > 0:
                stress_data = stress_data.sort_values(timestamp_col).reset_index(drop=True)
                stress_data['group_changed'] = (
                    (stress_data[timestamp_col].diff() > pd.Timedelta(minutes=5)) |
                    (stress_data[timestamp_col].diff().isna())
                )
                stress_data['group_id'] = stress_data['group_changed'].cumsum()
                
                for group_id, group in stress_data.groupby('group_id'):
                    if len(group) < self.min_pattern_duration_minutes / 5:
                        continue
                    
                    duration = (group[timestamp_col].iloc[-1] - 
                              group[timestamp_col].iloc[0]).total_seconds() / 60.0
                    
                    event = CorrelationEvent(
                        timestamp=group[timestamp_col].iloc[0],
                        pattern_type=HealthPattern.HEAT_STRESS,
                        confidence=float(group['heat_stress_confidence'].mean()),
                        temperature=float(group[temperature_col].mean()),
                        activity_level=float(group[activity_col].mean()),
                        behavioral_state=group[state_col].mode()[0] if len(group[state_col].mode()) > 0 else group[state_col].iloc[0],
                        duration_minutes=duration,
                        correlation_coefficient=float(group['correlation_1h'].mean()) if 'correlation_1h' in group.columns else None,
                        contributing_factors={
                            'avg_temperature': float(group[temperature_col].mean()),
                            'max_temperature': float(group[temperature_col].max()),
                            'avg_activity': float(group[activity_col].mean()),
                            'max_activity': float(group[activity_col].max()),
                            'sample_count': len(group)
                        }
                    )
                    events.append(event)
        
        logger.info(f"Generated {len(events)} correlation events "
                   f"({sum(1 for e in events if e.pattern_type == HealthPattern.FEVER)} fever, "
                   f"{sum(1 for e in events if e.pattern_type == HealthPattern.HEAT_STRESS)} heat stress)")
        
        return events
    
    def process_full_correlation(
        self,
        temperature_data: pd.DataFrame,
        behavioral_data: pd.DataFrame,
        temp_timestamp_col: str = 'timestamp',
        temp_value_col: str = 'temperature',
        behavior_timestamp_col: str = 'timestamp',
        behavior_state_col: str = 'behavioral_state',
        intensity_col: Optional[str] = 'movement_intensity'
    ) -> Tuple[pd.DataFrame, List[CorrelationEvent], Dict]:
        """
        Run complete correlation analysis pipeline.
        
        Args:
            temperature_data: DataFrame with temperature readings
            behavioral_data: DataFrame with behavioral states
            temp_timestamp_col: Timestamp column in temperature data
            temp_value_col: Temperature value column
            behavior_timestamp_col: Timestamp column in behavioral data
            behavior_state_col: Behavioral state column
            intensity_col: Movement intensity column (optional)
            
        Returns:
            Tuple of (merged_data, events, metrics)
        """
        logger.info("Starting full correlation analysis")
        
        # Load and prepare behavioral data
        behavioral_prepared = self.load_behavioral_data(
            behavioral_data,
            timestamp_col=behavior_timestamp_col,
            state_col=behavior_state_col,
            intensity_col=intensity_col
        )
        
        # Merge temperature and activity data
        merged = self.merge_temperature_activity(
            temperature_data,
            behavioral_prepared,
            temp_timestamp_col=temp_timestamp_col,
            temp_value_col=temp_value_col,
            behavior_timestamp_col=behavior_timestamp_col
        )
        
        if len(merged) == 0:
            logger.warning("No data after merging. Check time alignment.")
            return merged, [], {}
        
        # Calculate baseline activity
        merged = self.calculate_baseline_activity(merged)
        
        # Detect fever patterns
        merged = self.detect_fever_pattern(merged)
        
        # Detect heat stress patterns
        merged = self.detect_heat_stress_pattern(merged)
        
        # Calculate correlation metrics
        correlation_results = self.calculate_correlation_metrics(merged)
        
        # Add 1-hour correlation to main dataframe
        if '1h' in correlation_results:
            merged = correlation_results['1h']
        
        # Calculate lag analysis
        lag_analysis = self.calculate_lag_analysis(merged)
        
        # Generate correlation events
        events = self.generate_correlation_events(merged)
        
        # Compile metrics
        metrics = {
            'total_records': len(merged),
            'fever_detections': merged['fever_pattern'].sum() if 'fever_pattern' in merged.columns else 0,
            'heat_stress_detections': merged['heat_stress_pattern'].sum() if 'heat_stress_pattern' in merged.columns else 0,
            'correlation_windows': correlation_results,
            'lag_analysis': lag_analysis,
            'events_generated': len(events)
        }
        
        logger.info(f"Correlation analysis complete: {len(events)} events generated")
        
        return merged, events, metrics
