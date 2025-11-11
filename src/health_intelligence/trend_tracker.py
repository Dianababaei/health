"""
Multi-Day Health Trend Tracker

Combines temperature trends, activity patterns, alert frequency, and behavioral changes
over 7, 14, 30, and 90-day periods to generate unified health trend indicators for
dashboard visualization.

This module aggregates data from:
- Layer 2: Temperature and activity trends
- Layer 1: Behavioral state classifications
- Health Intelligence: Alert logs and history

Output Format:
- JSON-serializable trend objects for dashboard consumption
- Trend indicators: improving/stable/deteriorating
- Confidence scores based on data completeness and consistency
- Comparative metrics between time windows
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# Import Layer 2 trend analysis
try:
    from ..layer2_physiological import (
        MultiDayTrendAnalyzer,
        TrendDirection,
        HealthTrajectory
    )
except ImportError:
    # Fallback for testing
    MultiDayTrendAnalyzer = None
    TrendDirection = None
    HealthTrajectory = None

logger = logging.getLogger(__name__)


class TrendIndicator(Enum):
    """Overall health trend indicator."""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TimeWindowMetrics:
    """Metrics aggregated over a specific time window."""
    period_days: int
    start_date: datetime
    end_date: datetime
    data_completeness: float  # 0.0-1.0

    # Temperature metrics
    temperature_mean: float
    temperature_std: float
    temperature_baseline_drift: float
    temperature_anomaly_count: int

    # Activity metrics
    total_activity_minutes: float
    activity_level_mean: float  # 0-1 normalized
    rest_minutes: float
    activity_diversity: float  # 0-1 Shannon entropy

    # Alert metrics
    alert_count: int
    alert_severity_distribution: Dict[str, int]  # {'critical': N, 'warning': M}
    alert_type_distribution: Dict[str, int]  # {'fever': N, 'heat_stress': M, ...}

    # Behavioral state metrics
    state_distribution: Dict[str, float]  # {'lying': 40%, 'standing': 20%, ...}
    state_changes_per_day: float

    # Computed trend
    trend_indicator: TrendIndicator
    confidence_score: float


@dataclass
class HealthTrendReport:
    """Complete health trend report for a single animal."""
    cow_id: str
    analysis_timestamp: datetime

    # Trend data for each period
    trend_7day: Optional[TimeWindowMetrics]
    trend_14day: Optional[TimeWindowMetrics]
    trend_30day: Optional[TimeWindowMetrics]
    trend_90day: Optional[TimeWindowMetrics]

    # Overall assessment
    overall_trend: TrendIndicator
    overall_confidence: float

    # Comparative analysis
    period_comparisons: Dict[str, Dict[str, Any]]  # e.g., {"7d_vs_14d": {...}}

    # Key insights
    significant_changes: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            'cow_id': self.cow_id,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'overall_trend': self.overall_trend.value,
            'overall_confidence': self.overall_confidence,
            'period_comparisons': self.period_comparisons,
            'significant_changes': self.significant_changes,
            'recommendations': self.recommendations,
        }

        # Add period data
        for period_name in ['trend_7day', 'trend_14day', 'trend_30day', 'trend_90day']:
            period_data = getattr(self, period_name)
            if period_data is not None:
                result[period_name] = {
                    'period_days': period_data.period_days,
                    'start_date': period_data.start_date.isoformat(),
                    'end_date': period_data.end_date.isoformat(),
                    'data_completeness': period_data.data_completeness,
                    'temperature': {
                        'mean': period_data.temperature_mean,
                        'std': period_data.temperature_std,
                        'baseline_drift': period_data.temperature_baseline_drift,
                        'anomaly_count': period_data.temperature_anomaly_count,
                    },
                    'activity': {
                        'total_minutes': period_data.total_activity_minutes,
                        'mean_level': period_data.activity_level_mean,
                        'rest_minutes': period_data.rest_minutes,
                        'diversity': period_data.activity_diversity,
                    },
                    'alerts': {
                        'count': period_data.alert_count,
                        'severity_distribution': period_data.alert_severity_distribution,
                        'type_distribution': period_data.alert_type_distribution,
                    },
                    'behavioral': {
                        'state_distribution': period_data.state_distribution,
                        'changes_per_day': period_data.state_changes_per_day,
                    },
                    'trend_indicator': period_data.trend_indicator.value,
                    'confidence': period_data.confidence_score,
                }
            else:
                result[period_name] = None

        return result


class MultiDayHealthTrendTracker:
    """
    Unified health trend tracker combining multiple data sources.

    Aggregates temperature, activity, alerts, and behavioral data over
    7, 14, 30, and 90-day periods to generate comprehensive trend reports.
    """

    def __init__(self, temperature_baseline: float = 38.5):
        """
        Initialize tracker.

        Args:
            temperature_baseline: Normal baseline temperature (default 38.5°C)
        """
        self.temperature_baseline = temperature_baseline
        self.periods = [7, 14, 30, 90]

        # Initialize Layer 2 trend analyzer if available
        if MultiDayTrendAnalyzer is not None:
            self.temp_activity_analyzer = MultiDayTrendAnalyzer(temperature_baseline)
        else:
            self.temp_activity_analyzer = None
            logger.warning("Layer 2 MultiDayTrendAnalyzer not available")

    def analyze_trends(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        alert_history: List[Dict[str, Any]],
        behavioral_states: pd.DataFrame,
        analysis_date: Optional[datetime] = None
    ) -> HealthTrendReport:
        """
        Generate comprehensive health trend report.

        Args:
            cow_id: Animal identifier
            temperature_data: DataFrame with columns ['timestamp', 'temperature']
            activity_data: DataFrame with columns ['timestamp', 'behavioral_state', 'movement_intensity']
            alert_history: List of alert dictionaries with 'timestamp', 'alert_type', 'severity'
            behavioral_states: DataFrame with columns ['timestamp', 'behavioral_state']
            analysis_date: Date to run analysis from (default: now)

        Returns:
            HealthTrendReport with all trend metrics and assessments
        """
        if analysis_date is None:
            analysis_date = datetime.now()

        # Analyze each time period
        trend_7day = self._analyze_period(
            cow_id, 7, analysis_date,
            temperature_data, activity_data, alert_history, behavioral_states
        )

        trend_14day = self._analyze_period(
            cow_id, 14, analysis_date,
            temperature_data, activity_data, alert_history, behavioral_states
        )

        trend_30day = self._analyze_period(
            cow_id, 30, analysis_date,
            temperature_data, activity_data, alert_history, behavioral_states
        )

        trend_90day = self._analyze_period(
            cow_id, 90, analysis_date,
            temperature_data, activity_data, alert_history, behavioral_states
        )

        # Calculate overall trend
        overall_trend, overall_confidence = self._calculate_overall_trend(
            trend_7day, trend_14day, trend_30day, trend_90day
        )

        # Generate period comparisons
        comparisons = self._compare_periods(
            trend_7day, trend_14day, trend_30day, trend_90day
        )

        # Identify significant changes
        significant_changes = self._identify_significant_changes(
            trend_7day, trend_14day, trend_30day, trend_90day
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_trend, trend_7day, trend_14day, significant_changes
        )

        return HealthTrendReport(
            cow_id=cow_id,
            analysis_timestamp=analysis_date,
            trend_7day=trend_7day,
            trend_14day=trend_14day,
            trend_30day=trend_30day,
            trend_90day=trend_90day,
            overall_trend=overall_trend,
            overall_confidence=overall_confidence,
            period_comparisons=comparisons,
            significant_changes=significant_changes,
            recommendations=recommendations
        )

    def _analyze_period(
        self,
        cow_id: str,
        period_days: int,
        analysis_date: datetime,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        alert_history: List[Dict[str, Any]],
        behavioral_states: pd.DataFrame
    ) -> Optional[TimeWindowMetrics]:
        """Analyze a single time period."""
        start_date = analysis_date - timedelta(days=period_days)
        end_date = analysis_date

        # Filter data for period
        temp_period = self._filter_by_date(temperature_data, start_date, end_date)
        activity_period = self._filter_by_date(activity_data, start_date, end_date)
        behavioral_period = self._filter_by_date(behavioral_states, start_date, end_date)

        # Check data completeness
        expected_samples = period_days * 1440  # 1 sample per minute
        temp_completeness = len(temp_period) / expected_samples if expected_samples > 0 else 0
        activity_completeness = len(activity_period) / expected_samples if expected_samples > 0 else 0
        data_completeness = (temp_completeness + activity_completeness) / 2.0

        # Require at least 60% data for any analysis
        if data_completeness < 0.60:
            logger.info(f"Insufficient data for {period_days}-day period: {data_completeness:.1%}")
            return None

        # Calculate temperature metrics
        temp_metrics = self._calculate_temperature_metrics(temp_period)

        # Calculate activity metrics
        activity_metrics = self._calculate_activity_metrics(activity_period)

        # Calculate alert metrics
        alert_metrics = self._calculate_alert_metrics(alert_history, start_date, end_date)

        # Calculate behavioral metrics
        behavioral_metrics = self._calculate_behavioral_metrics(behavioral_period, period_days)

        # Determine trend indicator
        trend_indicator, confidence = self._classify_trend(
            temp_metrics, activity_metrics, alert_metrics, behavioral_metrics, data_completeness
        )

        return TimeWindowMetrics(
            period_days=period_days,
            start_date=start_date,
            end_date=end_date,
            data_completeness=data_completeness,
            temperature_mean=temp_metrics['mean'],
            temperature_std=temp_metrics['std'],
            temperature_baseline_drift=temp_metrics['baseline_drift'],
            temperature_anomaly_count=temp_metrics['anomaly_count'],
            total_activity_minutes=activity_metrics['total_minutes'],
            activity_level_mean=activity_metrics['mean_level'],
            rest_minutes=activity_metrics['rest_minutes'],
            activity_diversity=activity_metrics['diversity'],
            alert_count=alert_metrics['count'],
            alert_severity_distribution=alert_metrics['severity_dist'],
            alert_type_distribution=alert_metrics['type_dist'],
            state_distribution=behavioral_metrics['state_dist'],
            state_changes_per_day=behavioral_metrics['changes_per_day'],
            trend_indicator=trend_indicator,
            confidence_score=confidence
        )

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Filter DataFrame by date range."""
        if df is None or len(df) == 0:
            return pd.DataFrame()

        if 'timestamp' not in df.columns:
            return pd.DataFrame()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]

    def _calculate_temperature_metrics(self, temp_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate temperature metrics for period."""
        if len(temp_data) == 0:
            return {
                'mean': np.nan,
                'std': np.nan,
                'baseline_drift': np.nan,
                'anomaly_count': 0
            }

        temps = temp_data['temperature'].values
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        baseline_drift = mean_temp - self.temperature_baseline

        # Count anomalies (>39.5°C or <37.5°C)
        anomaly_count = np.sum((temps > 39.5) | (temps < 37.5))

        return {
            'mean': mean_temp,
            'std': std_temp,
            'baseline_drift': baseline_drift,
            'anomaly_count': int(anomaly_count)
        }

    def _calculate_activity_metrics(self, activity_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate activity metrics for period."""
        if len(activity_data) == 0:
            return {
                'total_minutes': 0.0,
                'mean_level': 0.0,
                'rest_minutes': 0.0,
                'diversity': 0.0
            }

        # Calculate total active time (not lying/resting)
        active_states = ['walking', 'running', 'feeding', 'standing']
        rest_states = ['lying', 'ruminating']

        total_active = len(activity_data[activity_data['behavioral_state'].isin(active_states)])
        total_rest = len(activity_data[activity_data['behavioral_state'].isin(rest_states)])

        # Mean activity level from movement_intensity if available
        if 'movement_intensity' in activity_data.columns:
            mean_level = activity_data['movement_intensity'].mean()
        else:
            # Estimate from states
            state_to_intensity = {
                'lying': 0.1, 'ruminating': 0.2, 'standing': 0.3,
                'feeding': 0.4, 'walking': 0.7, 'running': 1.0
            }
            mean_level = activity_data['behavioral_state'].map(state_to_intensity).mean()

        # Calculate behavioral diversity (Shannon entropy)
        state_counts = activity_data['behavioral_state'].value_counts()
        total = len(activity_data)
        diversity = 0.0
        for count in state_counts.values:
            if count > 0:
                p = count / total
                diversity -= p * np.log2(p)

        # Normalize (max entropy for 6 states = log2(6) ≈ 2.58)
        max_entropy = np.log2(min(len(state_counts), 6))
        diversity = diversity / max_entropy if max_entropy > 0 else 0

        return {
            'total_minutes': float(total_active),
            'mean_level': float(mean_level),
            'rest_minutes': float(total_rest),
            'diversity': float(diversity)
        }

    def _calculate_alert_metrics(
        self,
        alert_history: List[Dict[str, Any]],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate alert metrics for period."""
        if not alert_history:
            return {
                'count': 0,
                'severity_dist': {},
                'type_dist': {}
            }

        # Filter alerts in period
        period_alerts = []
        for alert in alert_history:
            alert_time = alert.get('timestamp')
            if isinstance(alert_time, str):
                alert_time = pd.to_datetime(alert_time)
            if start_date <= alert_time <= end_date:
                period_alerts.append(alert)

        # Count by severity
        severity_dist = {}
        for alert in period_alerts:
            severity = alert.get('severity', 'unknown')
            severity_dist[severity] = severity_dist.get(severity, 0) + 1

        # Count by type
        type_dist = {}
        for alert in period_alerts:
            alert_type = alert.get('alert_type', 'unknown')
            type_dist[alert_type] = type_dist.get(alert_type, 0) + 1

        return {
            'count': len(period_alerts),
            'severity_dist': severity_dist,
            'type_dist': type_dist
        }

    def _calculate_behavioral_metrics(
        self,
        behavioral_data: pd.DataFrame,
        period_days: int
    ) -> Dict[str, Any]:
        """Calculate behavioral state metrics for period."""
        if len(behavioral_data) == 0:
            return {
                'state_dist': {},
                'changes_per_day': 0.0
            }

        # State distribution (percentages)
        state_counts = behavioral_data['behavioral_state'].value_counts()
        total = len(behavioral_data)
        state_dist = {state: (count / total * 100) for state, count in state_counts.items()}

        # Count state transitions
        states = behavioral_data['behavioral_state'].values
        transitions = np.sum(states[:-1] != states[1:])
        changes_per_day = transitions / period_days if period_days > 0 else 0

        return {
            'state_dist': state_dist,
            'changes_per_day': float(changes_per_day)
        }

    def _classify_trend(
        self,
        temp_metrics: Dict,
        activity_metrics: Dict,
        alert_metrics: Dict,
        behavioral_metrics: Dict,
        data_completeness: float
    ) -> Tuple[TrendIndicator, float]:
        """
        Classify overall trend based on all metrics.

        Returns:
            (trend_indicator, confidence_score)
        """
        # Score each component (0-1, higher is better/healthier)

        # Temperature score (lower drift and variance is better)
        temp_drift = abs(temp_metrics['baseline_drift'])
        temp_score = 1.0 - min(temp_drift / 1.0, 1.0)  # 1°C drift = 0 score
        temp_score *= 1.0 - min(temp_metrics['std'] / 0.5, 1.0)  # High std reduces score

        # Activity score (moderate activity is best)
        activity_score = min(activity_metrics['mean_level'] / 0.5, 1.0)
        activity_score *= activity_metrics['diversity']

        # Alert score (fewer alerts is better)
        alert_count = alert_metrics['count']
        critical_alerts = alert_metrics['severity_dist'].get('critical', 0)
        alert_score = 1.0 - min((alert_count + critical_alerts * 2) / 10, 1.0)

        # Behavioral score (good diversity and normal transitions)
        behavioral_score = behavioral_metrics['changes_per_day'] / 100  # ~100 is normal
        behavioral_score = min(behavioral_score, 1.0)

        # Combined health score (weighted average)
        health_score = (
            temp_score * 0.35 +
            activity_score * 0.30 +
            alert_score * 0.20 +
            behavioral_score * 0.15
        )

        # Classify trend
        if health_score > 0.7:
            trend = TrendIndicator.IMPROVING
        elif health_score > 0.5:
            trend = TrendIndicator.STABLE
        else:
            trend = TrendIndicator.DETERIORATING

        # Confidence based on data completeness and consistency
        confidence = data_completeness * health_score

        return trend, confidence

    def _calculate_overall_trend(
        self,
        trend_7day: Optional[TimeWindowMetrics],
        trend_14day: Optional[TimeWindowMetrics],
        trend_30day: Optional[TimeWindowMetrics],
        trend_90day: Optional[TimeWindowMetrics]
    ) -> Tuple[TrendIndicator, float]:
        """Calculate overall trend from all periods."""
        available_trends = [t for t in [trend_7day, trend_14day, trend_30day, trend_90day] if t is not None]

        if not available_trends:
            return TrendIndicator.INSUFFICIENT_DATA, 0.0

        # Weight shorter periods more heavily (more recent)
        weights = {7: 0.4, 14: 0.3, 30: 0.2, 90: 0.1}

        total_weight = 0.0
        weighted_score = 0.0

        for trend in available_trends:
            weight = weights.get(trend.period_days, 0.1)
            total_weight += weight

            # Convert trend to score (improving=1.0, stable=0.5, deteriorating=0.0)
            if trend.trend_indicator == TrendIndicator.IMPROVING:
                score = 1.0
            elif trend.trend_indicator == TrendIndicator.STABLE:
                score = 0.5
            else:  # DETERIORATING
                score = 0.0

            weighted_score += score * weight * trend.confidence_score

        # Normalize
        if total_weight > 0:
            weighted_score /= total_weight

        # Map back to trend
        if weighted_score > 0.7:
            overall_trend = TrendIndicator.IMPROVING
        elif weighted_score > 0.4:
            overall_trend = TrendIndicator.STABLE
        else:
            overall_trend = TrendIndicator.DETERIORATING

        # Overall confidence is average of available confidences
        overall_confidence = np.mean([t.confidence_score for t in available_trends])

        return overall_trend, overall_confidence

    def _compare_periods(
        self,
        trend_7day: Optional[TimeWindowMetrics],
        trend_14day: Optional[TimeWindowMetrics],
        trend_30day: Optional[TimeWindowMetrics],
        trend_90day: Optional[TimeWindowMetrics]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate period-over-period comparisons."""
        comparisons = {}

        pairs = [
            (trend_7day, trend_14day, "7d_vs_14d"),
            (trend_14day, trend_30day, "14d_vs_30d"),
            (trend_30day, trend_90day, "30d_vs_90d")
        ]

        for shorter, longer, name in pairs:
            if shorter is None or longer is None:
                continue

            comparisons[name] = {
                'temperature_delta': shorter.temperature_mean - longer.temperature_mean,
                'activity_delta': shorter.activity_level_mean - longer.activity_level_mean,
                'alert_delta': shorter.alert_count - longer.alert_count,
                'trend_change': f"{longer.trend_indicator.value} -> {shorter.trend_indicator.value}"
            }

        return comparisons

    def _identify_significant_changes(
        self,
        trend_7day: Optional[TimeWindowMetrics],
        trend_14day: Optional[TimeWindowMetrics],
        trend_30day: Optional[TimeWindowMetrics],
        trend_90day: Optional[TimeWindowMetrics]
    ) -> List[str]:
        """Identify significant changes worthy of attention."""
        changes = []

        if trend_7day is not None:
            # Recent alerts
            if trend_7day.alert_count > 3:
                changes.append(f"High alert frequency in past 7 days: {trend_7day.alert_count} alerts")

            # Temperature spike
            if abs(trend_7day.temperature_baseline_drift) > 0.5:
                changes.append(f"Temperature drift: {trend_7day.temperature_baseline_drift:+.2f}°C from baseline")

            # Low activity
            if trend_7day.activity_level_mean < 0.3:
                changes.append(f"Low activity level: {trend_7day.activity_level_mean:.2f}")

        # Compare short vs long term
        if trend_7day is not None and trend_30day is not None:
            temp_change = trend_7day.temperature_mean - trend_30day.temperature_mean
            if abs(temp_change) > 0.3:
                changes.append(f"Temperature change (7d vs 30d): {temp_change:+.2f}°C")

            activity_change = trend_7day.activity_level_mean - trend_30day.activity_level_mean
            if abs(activity_change) > 0.2:
                direction = "increased" if activity_change > 0 else "decreased"
                changes.append(f"Activity {direction} by {abs(activity_change):.2f} (7d vs 30d)")

        return changes

    def _generate_recommendations(
        self,
        overall_trend: TrendIndicator,
        trend_7day: Optional[TimeWindowMetrics],
        trend_14day: Optional[TimeWindowMetrics],
        significant_changes: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if overall_trend == TrendIndicator.DETERIORATING:
            recommendations.append("PRIORITY: Schedule veterinary examination")
            recommendations.append("Increase monitoring frequency")

        elif overall_trend == TrendIndicator.STABLE:
            recommendations.append("Continue routine monitoring")

        elif overall_trend == TrendIndicator.IMPROVING:
            recommendations.append("Positive health trend - maintain current care")

        # Specific recommendations
        if trend_7day is not None:
            if trend_7day.alert_count > 5:
                recommendations.append("Investigate root cause of frequent alerts")

            if trend_7day.activity_level_mean < 0.25:
                recommendations.append("Check for lameness or illness causing low activity")

            if trend_7day.activity_diversity < 0.4:
                recommendations.append("Low behavioral diversity - monitor for illness")

        return recommendations
