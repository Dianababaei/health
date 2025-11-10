"""
Multi-Day Trend Analysis System

Tracks temperature and activity trends over 7, 14, 30, and 90-day periods to identify
improving or deteriorating health patterns.

This module supports:
- Short-term (7 days): Acute condition recovery/decline
- Medium-term (14 days): Post-treatment progress, estrus cycle monitoring
- Long-term (30 days): Monthly health assessment, seasonal patterns
- Extended (90 days): Reproductive cycle tracking, pregnancy confirmation

Literature:
- Reproductive cycles: 21-day estrus cycle, 280-day gestation
- Recovery monitoring: 7-14 day post-treatment assessment
- Baseline drift detection over 30+ day periods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class TrendDirection(Enum):
    """Trend direction classification."""
    IMPROVING = "improving"
    STABLE = "stable"
    DETERIORATING = "deteriorating"
    INSUFFICIENT_DATA = "insufficient_data"


class HealthTrajectory(Enum):
    """Combined health trend assessment."""
    STRONG_IMPROVEMENT = "strong_improvement"
    MODERATE_IMPROVEMENT = "moderate_improvement"
    STABLE = "stable"
    MODERATE_DECLINE = "moderate_decline"
    SIGNIFICANT_DECLINE = "significant_decline"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TrendPeriodConfig:
    """Configuration for a trend analysis period."""
    name: str
    days: int
    min_data_completeness: float = 0.80  # Require 80% data
    temperature_stability_threshold: float = 0.15  # CV threshold for stability
    activity_change_threshold: float = 0.10  # 10% change threshold


@dataclass
class TemperatureTrendMetrics:
    """Temperature trend analysis metrics for a period."""
    period_days: int
    mean_temperature: float
    std_temperature: float
    coefficient_variation: float  # Stability index
    anomaly_count: int
    anomaly_frequency: float  # Anomalies per day
    baseline_drift: float  # Deviation from historical baseline
    circadian_health_score: float  # Average circadian rhythm quality
    trend_direction: TrendDirection
    confidence: float


@dataclass
class ActivityTrendMetrics:
    """Activity trend analysis metrics for a period."""
    period_days: int
    mean_daily_movement: float
    rest_activity_ratio: float  # Lying time / total time
    behavioral_diversity: float  # Shannon entropy of state distribution
    state_transition_frequency: float  # Transitions per day
    lying_percentage: float
    standing_percentage: float
    walking_percentage: float
    trend_direction: TrendDirection
    confidence: float


@dataclass
class CombinedHealthTrend:
    """Combined temperature and activity health trend."""
    period_days: int
    temperature_metrics: TemperatureTrendMetrics
    activity_metrics: ActivityTrendMetrics
    health_trajectory: HealthTrajectory
    overall_confidence: float
    significant_events: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TrendReport:
    """Complete trend report for all configured periods."""
    cow_id: int
    analysis_date: datetime
    trends: Dict[int, CombinedHealthTrend]  # period_days -> trend
    period_comparisons: Dict[Tuple[int, int], Dict] = field(default_factory=dict)  # (period1, period2) -> deltas


class MultiDayTrendAnalyzer:
    """
    Analyzes multi-day trends in temperature and activity data.

    Tracks patterns over 7, 14, 30, and 90-day periods to identify
    improving or deteriorating health trajectories.
    """

    DEFAULT_PERIODS = [
        TrendPeriodConfig("short_term", 7),
        TrendPeriodConfig("medium_term", 14),
        TrendPeriodConfig("long_term", 30),
        TrendPeriodConfig("extended", 90)
    ]

    def __init__(
        self,
        periods: Optional[List[TrendPeriodConfig]] = None,
        temperature_baseline: float = 38.5,
        temperature_normal_variance: float = 0.10
    ):
        """
        Initialize trend analyzer.

        Args:
            periods: List of period configurations (default: 7, 14, 30, 90 days)
            temperature_baseline: Normal baseline temperature (°C)
            temperature_normal_variance: Expected temperature variance for healthy animals
        """
        self.periods = periods if periods is not None else self.DEFAULT_PERIODS
        self.temperature_baseline = temperature_baseline
        self.temperature_normal_variance = temperature_normal_variance

    def analyze_trends(
        self,
        cow_id: int,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        anomaly_history: Optional[List[Dict]] = None,
        circadian_scores: Optional[pd.DataFrame] = None
    ) -> TrendReport:
        """
        Analyze trends across all configured periods.

        Args:
            cow_id: Animal ID
            temperature_data: DataFrame with columns ['timestamp', 'temperature']
            activity_data: DataFrame with columns ['timestamp', 'behavioral_state', 'movement_intensity']
            anomaly_history: List of anomaly events
            circadian_scores: DataFrame with circadian health scores

        Returns:
            Complete trend report with all periods
        """
        trends = {}

        for period_config in self.periods:
            # Calculate trends for this period
            temp_trend = self._analyze_temperature_trend(
                temperature_data,
                period_config,
                anomaly_history,
                circadian_scores
            )

            activity_trend = self._analyze_activity_trend(
                activity_data,
                period_config
            )

            # Combine into health trajectory
            combined_trend = self._assess_health_trajectory(
                period_config,
                temp_trend,
                activity_trend,
                anomaly_history
            )

            trends[period_config.days] = combined_trend

        # Calculate period-over-period comparisons
        comparisons = self._calculate_period_comparisons(trends)

        return TrendReport(
            cow_id=cow_id,
            analysis_date=datetime.now(),
            trends=trends,
            period_comparisons=comparisons
        )

    def _analyze_temperature_trend(
        self,
        temperature_data: pd.DataFrame,
        period_config: TrendPeriodConfig,
        anomaly_history: Optional[List[Dict]],
        circadian_scores: Optional[pd.DataFrame]
    ) -> TemperatureTrendMetrics:
        """Analyze temperature trends for specified period."""

        # Get data for this period
        cutoff_time = datetime.now() - timedelta(days=period_config.days)
        period_data = temperature_data[temperature_data['timestamp'] >= cutoff_time]

        # Check data completeness
        expected_samples = period_config.days * 1440  # 1 sample/min
        actual_samples = len(period_data)
        data_completeness = actual_samples / expected_samples if expected_samples > 0 else 0

        if data_completeness < period_config.min_data_completeness:
            return TemperatureTrendMetrics(
                period_days=period_config.days,
                mean_temperature=np.nan,
                std_temperature=np.nan,
                coefficient_variation=np.nan,
                anomaly_count=0,
                anomaly_frequency=0.0,
                baseline_drift=np.nan,
                circadian_health_score=np.nan,
                trend_direction=TrendDirection.INSUFFICIENT_DATA,
                confidence=0.0
            )

        # Calculate basic statistics
        temps = period_data['temperature'].values
        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        cv = std_temp / mean_temp if mean_temp > 0 else 0

        # Count anomalies in period
        anomaly_count = 0
        if anomaly_history:
            period_anomalies = [
                a for a in anomaly_history
                if a.get('timestamp', datetime.min) >= cutoff_time
            ]
            anomaly_count = len(period_anomalies)
        anomaly_frequency = anomaly_count / period_config.days

        # Calculate baseline drift (signed - positive means above baseline)
        baseline_drift = mean_temp - self.temperature_baseline

        # Average circadian health score
        circadian_score = 0.8  # Default if not provided
        if circadian_scores is not None and len(circadian_scores) > 0:
            period_scores = circadian_scores[circadian_scores['timestamp'] >= cutoff_time]
            if len(period_scores) > 0:
                circadian_score = period_scores['health_score'].mean()

        # Classify trend direction
        trend_direction = self._classify_temperature_trend(
            cv, anomaly_frequency, baseline_drift, period_config
        )

        # Calculate confidence
        confidence = self._calculate_temperature_confidence(
            data_completeness, cv, circadian_score
        )

        return TemperatureTrendMetrics(
            period_days=period_config.days,
            mean_temperature=mean_temp,
            std_temperature=std_temp,
            coefficient_variation=cv,
            anomaly_count=anomaly_count,
            anomaly_frequency=anomaly_frequency,
            baseline_drift=baseline_drift,
            circadian_health_score=circadian_score,
            trend_direction=trend_direction,
            confidence=confidence
        )

    def _classify_temperature_trend(
        self,
        cv: float,
        anomaly_freq: float,
        baseline_drift: float,
        config: TrendPeriodConfig
    ) -> TrendDirection:
        """
        Classify temperature trend direction.

        Improving: Low CV, few anomalies, near baseline (drift can be large if returning to baseline)
        Stable: Moderate CV, occasional anomalies, acceptable drift
        Deteriorating: High CV, frequent anomalies, large drift above baseline
        """
        # Score components (0-1, lower is better)
        stability_score = min(cv / config.temperature_stability_threshold, 1.0)
        anomaly_score = min(anomaly_freq / 2.0, 1.0)  # >2 anomalies/day is concerning

        # For drift: positive (above baseline) is concerning, near zero is good
        # Negative drift (below baseline) is less concerning if recovering
        if baseline_drift > 0:
            # Above baseline - concerning
            drift_score = min(baseline_drift / 0.5, 1.0)
        else:
            # Below baseline or at baseline - less concerning
            drift_score = min(abs(baseline_drift) / 1.0, 0.5)  # Cap at 0.5

        # Combined health score - weight drift more heavily as it's a key indicator
        health_score = (stability_score + anomaly_score + drift_score * 2.0) / 4.0

        # Immediate deterioration if high fever (>=0.8°C above baseline)
        if baseline_drift >= 0.8:
            return TrendDirection.DETERIORATING

        if health_score < 0.35:
            return TrendDirection.IMPROVING
        elif health_score < 0.65:
            return TrendDirection.STABLE
        else:
            return TrendDirection.DETERIORATING

    def _calculate_temperature_confidence(
        self,
        data_completeness: float,
        cv: float,
        circadian_score: float
    ) -> float:
        """Calculate confidence score for temperature trend."""
        # Base confidence on data completeness
        confidence = data_completeness

        # Reduce confidence if high variability (less reliable pattern)
        if cv > 0.15:
            confidence *= 0.8

        # Increase confidence if good circadian health
        if circadian_score > 0.8:
            confidence = min(confidence * 1.1, 1.0)

        return confidence

    def _analyze_activity_trend(
        self,
        activity_data: pd.DataFrame,
        period_config: TrendPeriodConfig
    ) -> ActivityTrendMetrics:
        """Analyze activity trends for specified period."""

        # Get data for this period
        cutoff_time = datetime.now() - timedelta(days=period_config.days)
        period_data = activity_data[activity_data['timestamp'] >= cutoff_time]

        # Check data completeness
        expected_samples = period_config.days * 1440
        actual_samples = len(period_data)
        data_completeness = actual_samples / expected_samples if expected_samples > 0 else 0

        if data_completeness < period_config.min_data_completeness:
            return ActivityTrendMetrics(
                period_days=period_config.days,
                mean_daily_movement=np.nan,
                rest_activity_ratio=np.nan,
                behavioral_diversity=np.nan,
                state_transition_frequency=np.nan,
                lying_percentage=np.nan,
                standing_percentage=np.nan,
                walking_percentage=np.nan,
                trend_direction=TrendDirection.INSUFFICIENT_DATA,
                confidence=0.0
            )

        # Calculate movement intensity
        if 'movement_intensity' in period_data.columns:
            mean_movement = period_data['movement_intensity'].mean()
        else:
            # Estimate from behavioral states if movement_intensity not available
            mean_movement = self._estimate_movement_from_states(period_data)

        # Calculate behavioral state percentages
        state_counts = period_data['behavioral_state'].value_counts()
        total = len(period_data)

        lying_pct = state_counts.get('lying', 0) / total * 100
        standing_pct = state_counts.get('standing', 0) / total * 100
        walking_pct = state_counts.get('walking', 0) / total * 100

        # Rest/activity ratio
        rest_time = state_counts.get('lying', 0) + state_counts.get('ruminating', 0)
        rest_activity_ratio = rest_time / total if total > 0 else 0

        # Behavioral diversity (Shannon entropy)
        diversity = self._calculate_behavioral_diversity(state_counts, total)

        # State transition frequency
        transitions = self._count_state_transitions(period_data)
        transition_freq = transitions / period_config.days

        # Classify trend direction
        trend_direction = self._classify_activity_trend(
            mean_movement, rest_activity_ratio, diversity, period_config
        )

        # Calculate confidence
        confidence = data_completeness * (1.0 if diversity > 0.5 else 0.8)

        return ActivityTrendMetrics(
            period_days=period_config.days,
            mean_daily_movement=mean_movement,
            rest_activity_ratio=rest_activity_ratio,
            behavioral_diversity=diversity,
            state_transition_frequency=transition_freq,
            lying_percentage=lying_pct,
            standing_percentage=standing_pct,
            walking_percentage=walking_pct,
            trend_direction=trend_direction,
            confidence=confidence
        )

    def _estimate_movement_from_states(self, data: pd.DataFrame) -> float:
        """Estimate movement intensity from behavioral states."""
        # Map states to approximate movement levels
        movement_map = {
            'lying': 0.1,
            'standing': 0.3,
            'ruminating': 0.2,
            'feeding': 0.4,
            'walking': 0.8,
            'running': 1.0
        }

        movements = data['behavioral_state'].map(movement_map).fillna(0.3)
        return movements.mean()

    def _calculate_behavioral_diversity(
        self,
        state_counts: pd.Series,
        total: int
    ) -> float:
        """
        Calculate Shannon entropy of behavioral state distribution.

        Higher diversity = more varied behaviors (healthy)
        Lower diversity = repetitive behaviors (potentially concerning)
        """
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in state_counts.values:
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        # Normalize to 0-1 (max entropy for 5 states = log2(5) ≈ 2.32)
        max_entropy = np.log2(min(len(state_counts), 5))
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        return normalized

    def _count_state_transitions(self, data: pd.DataFrame) -> int:
        """Count number of behavioral state transitions."""
        if len(data) < 2:
            return 0

        states = data['behavioral_state'].values
        transitions = np.sum(states[:-1] != states[1:])

        return int(transitions)

    def _classify_activity_trend(
        self,
        mean_movement: float,
        rest_ratio: float,
        diversity: float,
        config: TrendPeriodConfig
    ) -> TrendDirection:
        """
        Classify activity trend direction.

        Improving: Increasing movement, balanced rest/activity, high diversity
        Stable: Consistent moderate activity, normal rest ratio
        Deteriorating: Decreasing movement, excessive rest, low diversity
        """
        # Score components (0-1, higher is better for activity)
        movement_score = min(mean_movement / 0.5, 1.0)  # 0.5 is healthy baseline
        rest_score = 1.0 - abs(rest_ratio - 0.5) * 2.0  # Optimal rest ~50%, penalize deviation
        rest_score = max(rest_score, 0.0)  # Ensure non-negative
        diversity_score = diversity

        # Combined activity health score
        health_score = (movement_score + rest_score + diversity_score) / 3.0

        if health_score > 0.65:
            return TrendDirection.IMPROVING
        elif health_score > 0.45:
            return TrendDirection.STABLE
        else:
            return TrendDirection.DETERIORATING

    def _assess_health_trajectory(
        self,
        period_config: TrendPeriodConfig,
        temp_trend: TemperatureTrendMetrics,
        activity_trend: ActivityTrendMetrics,
        anomaly_history: Optional[List[Dict]]
    ) -> CombinedHealthTrend:
        """Combine temperature and activity trends into overall health assessment."""

        # Determine health trajectory
        trajectory = self._determine_trajectory(
            temp_trend.trend_direction,
            activity_trend.trend_direction
        )

        # Calculate overall confidence (average of both)
        overall_confidence = (temp_trend.confidence + activity_trend.confidence) / 2.0

        # Identify significant events
        significant_events = []
        if anomaly_history:
            cutoff = datetime.now() - timedelta(days=period_config.days)
            period_anomalies = [
                a for a in anomaly_history
                if a.get('timestamp', datetime.min) >= cutoff
            ]
            if len(period_anomalies) > 5:
                significant_events.append(
                    f"High anomaly frequency: {len(period_anomalies)} events in {period_config.days} days"
                )

        if temp_trend.baseline_drift > 0.5:
            significant_events.append(
                f"Temperature baseline drift: {temp_trend.baseline_drift:.2f}°C from normal"
            )

        if activity_trend.rest_activity_ratio > 0.7:
            significant_events.append(
                f"Excessive rest time: {activity_trend.rest_activity_ratio*100:.1f}% lying/ruminating"
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(trajectory, temp_trend, activity_trend)

        return CombinedHealthTrend(
            period_days=period_config.days,
            temperature_metrics=temp_trend,
            activity_metrics=activity_trend,
            health_trajectory=trajectory,
            overall_confidence=overall_confidence,
            significant_events=significant_events,
            recommendations=recommendations
        )

    def _determine_trajectory(
        self,
        temp_direction: TrendDirection,
        activity_direction: TrendDirection
    ) -> HealthTrajectory:
        """Determine overall health trajectory from component trends."""

        if temp_direction == TrendDirection.INSUFFICIENT_DATA or \
           activity_direction == TrendDirection.INSUFFICIENT_DATA:
            return HealthTrajectory.INSUFFICIENT_DATA

        # Map combinations to trajectories
        if temp_direction == TrendDirection.IMPROVING and \
           activity_direction == TrendDirection.IMPROVING:
            return HealthTrajectory.STRONG_IMPROVEMENT

        elif (temp_direction == TrendDirection.IMPROVING and activity_direction == TrendDirection.STABLE) or \
             (temp_direction == TrendDirection.STABLE and activity_direction == TrendDirection.IMPROVING):
            return HealthTrajectory.MODERATE_IMPROVEMENT

        elif temp_direction == TrendDirection.STABLE and activity_direction == TrendDirection.STABLE:
            return HealthTrajectory.STABLE

        elif (temp_direction == TrendDirection.DETERIORATING and activity_direction == TrendDirection.STABLE) or \
             (temp_direction == TrendDirection.STABLE and activity_direction == TrendDirection.DETERIORATING):
            return HealthTrajectory.MODERATE_DECLINE

        else:  # Both deteriorating
            return HealthTrajectory.SIGNIFICANT_DECLINE

    def _generate_recommendations(
        self,
        trajectory: HealthTrajectory,
        temp_trend: TemperatureTrendMetrics,
        activity_trend: ActivityTrendMetrics
    ) -> List[str]:
        """Generate actionable recommendations based on trends."""
        recommendations = []

        if trajectory == HealthTrajectory.SIGNIFICANT_DECLINE:
            recommendations.append("URGENT: Veterinary review recommended")
            recommendations.append("Consider immediate health assessment")

        elif trajectory == HealthTrajectory.MODERATE_DECLINE:
            recommendations.append("Monitor closely for further deterioration")
            recommendations.append("Review recent treatments or environmental changes")

        if temp_trend.anomaly_frequency > 2.0:
            recommendations.append(f"High fever frequency: {temp_trend.anomaly_frequency:.1f} events/day")

        if abs(temp_trend.baseline_drift) > 0.5:
            if temp_trend.baseline_drift > 0:
                recommendations.append(f"Elevated temperature: {temp_trend.baseline_drift:.2f}°C above baseline")
            else:
                recommendations.append(f"Low temperature: {abs(temp_trend.baseline_drift):.2f}°C below baseline")

        if temp_trend.coefficient_variation > 0.15:
            recommendations.append("High temperature variability - investigate environmental factors")

        if activity_trend.behavioral_diversity < 0.3:
            recommendations.append("Low behavioral diversity - check for restricted movement or illness")

        if activity_trend.rest_activity_ratio > 0.7:
            recommendations.append("Excessive rest time - monitor for lameness or illness")

        if activity_trend.rest_activity_ratio < 0.3:
            recommendations.append("Low rest time - check for stress or environmental disturbances")

        if trajectory == HealthTrajectory.STRONG_IMPROVEMENT:
            recommendations.append("Positive recovery trend - continue current treatment")

        return recommendations

    def _calculate_period_comparisons(
        self,
        trends: Dict[int, CombinedHealthTrend]
    ) -> Dict[Tuple[int, int], Dict]:
        """Calculate period-over-period comparison metrics."""
        comparisons = {}

        periods = sorted(trends.keys())
        for i in range(len(periods) - 1):
            shorter_period = periods[i]
            longer_period = periods[i + 1]

            shorter_trend = trends[shorter_period]
            longer_trend = trends[longer_period]

            # Calculate deltas
            temp_delta = (
                shorter_trend.temperature_metrics.mean_temperature -
                longer_trend.temperature_metrics.mean_temperature
            )

            activity_delta = (
                shorter_trend.activity_metrics.mean_daily_movement -
                longer_trend.activity_metrics.mean_daily_movement
            )

            # Percentage changes
            temp_pct_change = (
                temp_delta / longer_trend.temperature_metrics.mean_temperature * 100
                if longer_trend.temperature_metrics.mean_temperature > 0 else 0
            )

            activity_pct_change = (
                activity_delta / longer_trend.activity_metrics.mean_daily_movement * 100
                if longer_trend.activity_metrics.mean_daily_movement > 0 else 0
            )

            comparisons[(shorter_period, longer_period)] = {
                'temperature_delta': temp_delta,
                'temperature_pct_change': temp_pct_change,
                'activity_delta': activity_delta,
                'activity_pct_change': activity_pct_change,
                'trajectory_change': (
                    shorter_trend.health_trajectory != longer_trend.health_trajectory
                )
            }

        return comparisons

    def format_trend_report(self, report: TrendReport) -> Dict:
        """Format trend report as dictionary for export/API."""
        formatted = {
            'cow_id': report.cow_id,
            'analysis_date': report.analysis_date.isoformat(),
            'trends': {},
            'period_comparisons': {}
        }

        # Format each period's trends
        for period_days, trend in report.trends.items():
            formatted['trends'][f'{period_days}_day'] = {
                'temperature': {
                    'mean': trend.temperature_metrics.mean_temperature,
                    'std': trend.temperature_metrics.std_temperature,
                    'stability_index': trend.temperature_metrics.coefficient_variation,
                    'anomaly_frequency': trend.temperature_metrics.anomaly_frequency,
                    'baseline_drift': trend.temperature_metrics.baseline_drift,
                    'circadian_health': trend.temperature_metrics.circadian_health_score,
                    'trend': trend.temperature_metrics.trend_direction.value,
                    'confidence': trend.temperature_metrics.confidence
                },
                'activity': {
                    'mean_movement': trend.activity_metrics.mean_daily_movement,
                    'rest_ratio': trend.activity_metrics.rest_activity_ratio,
                    'behavioral_diversity': trend.activity_metrics.behavioral_diversity,
                    'transition_frequency': trend.activity_metrics.state_transition_frequency,
                    'lying_pct': trend.activity_metrics.lying_percentage,
                    'standing_pct': trend.activity_metrics.standing_percentage,
                    'walking_pct': trend.activity_metrics.walking_percentage,
                    'trend': trend.activity_metrics.trend_direction.value,
                    'confidence': trend.activity_metrics.confidence
                },
                'overall': {
                    'health_trajectory': trend.health_trajectory.value,
                    'confidence': trend.overall_confidence,
                    'significant_events': trend.significant_events,
                    'recommendations': trend.recommendations
                }
            }

        # Format comparisons
        for (period1, period2), deltas in report.period_comparisons.items():
            key = f'{period1}_vs_{period2}_day'
            formatted['period_comparisons'][key] = deltas

        return formatted


if __name__ == "__main__":
    # Demo usage
    print("Multi-Day Trend Analysis System")
    print("=" * 60)

    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=1440 * 30, freq='1min')

    # Simulated improving trend: temperature stabilizing
    temp_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': 39.5 - (np.arange(len(dates)) / len(dates)) * 1.0 + np.random.normal(0, 0.1, len(dates))
    })

    # Simulated improving trend: activity increasing
    activity_data = pd.DataFrame({
        'timestamp': dates,
        'behavioral_state': np.random.choice(['lying', 'standing', 'walking'], len(dates)),
        'movement_intensity': 0.3 + (np.arange(len(dates)) / len(dates)) * 0.3
    })

    # Run analysis
    analyzer = MultiDayTrendAnalyzer()
    report = analyzer.analyze_trends(
        cow_id=1042,
        temperature_data=temp_data,
        activity_data=activity_data
    )

    # Display results
    formatted = analyzer.format_trend_report(report)
    print(f"\nCow ID: {formatted['cow_id']}")
    print(f"Analysis Date: {formatted['analysis_date']}")
    print("\nTrends by Period:")
    for period, data in formatted['trends'].items():
        print(f"\n{period}:")
        print(f"  Health Trajectory: {data['overall']['health_trajectory']}")
        print(f"  Temperature Trend: {data['temperature']['trend']} (confidence: {data['temperature']['confidence']:.2f})")
        print(f"  Activity Trend: {data['activity']['trend']} (confidence: {data['activity']['confidence']:.2f})")
        if data['overall']['recommendations']:
            print(f"  Recommendations: {', '.join(data['overall']['recommendations'])}")
