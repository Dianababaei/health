"""
Comprehensive Health Scoring System (0-100 Scale)

Provides a composite health score that combines multiple factors:
- Temperature health (35%)
- Activity level (30%)
- Alert frequency (20%)
- Behavioral diversity (15%)

Score Ranges:
- 90-100: Excellent health
- 75-89: Good health
- 60-74: Fair health (monitor)
- 40-59: Poor health (intervention recommended)
- 0-39: Critical health (immediate action required)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import math


class HealthCategory(Enum):
    """Health category based on score"""
    EXCELLENT = "excellent"  # 90-100
    GOOD = "good"           # 75-89
    FAIR = "fair"           # 60-74
    POOR = "poor"           # 40-59
    CRITICAL = "critical"   # 0-39


@dataclass
class HealthScore:
    """
    Comprehensive health score with component breakdown.

    Attributes:
        timestamp: When score was calculated
        cow_id: Identifier for the cow
        overall_score: Composite score (0-100)
        category: Health category
        component_scores: Breakdown by component
        factors: Contributing factors and weights
        trend: Score trend (improving/stable/declining)
        recommendations: Action recommendations
    """
    timestamp: datetime
    cow_id: str
    overall_score: float
    category: HealthCategory
    component_scores: Dict[str, float]
    factors: Dict[str, Any]
    trend: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cow_id': self.cow_id,
            'overall_score': round(self.overall_score, 1),
            'category': self.category.value,
            'component_scores': {k: round(v, 1) for k, v in self.component_scores.items()},
            'factors': self.factors,
            'trend': self.trend,
            'recommendations': self.recommendations
        }


class HealthScorer:
    """
    Calculate comprehensive health scores from multi-source data.

    Usage:
        scorer = HealthScorer(baseline_temp=38.5)
        score = scorer.calculate_health_score(
            cow_id='COW_001',
            temperature_data=temp_df,
            activity_data=activity_df,
            alert_history=alerts,
            behavioral_states=states_df
        )
    """

    def __init__(
        self,
        baseline_temp: float = 38.5,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize health scorer.

        Args:
            baseline_temp: Normal body temperature baseline
            weights: Component weights (default: temp=0.35, activity=0.30,
                    alerts=0.20, behavioral=0.15)
        """
        self.baseline_temp = baseline_temp

        # Default weights
        if weights is None:
            self.weights = {
                'temperature': 0.35,
                'activity': 0.30,
                'alerts': 0.20,
                'behavioral': 0.15
            }
        else:
            self.weights = weights

        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            self.weights = {k: v/total_weight for k, v in self.weights.items()}

    def calculate_health_score(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        alert_history: List[Dict[str, Any]],
        behavioral_states: pd.DataFrame,
        lookback_days: int = 7,
        previous_score: Optional[float] = None
    ) -> HealthScore:
        """
        Calculate comprehensive health score.

        Args:
            cow_id: Cow identifier
            temperature_data: DataFrame with ['timestamp', 'temperature']
            activity_data: DataFrame with ['timestamp', 'movement_intensity']
            alert_history: List of alert dictionaries
            behavioral_states: DataFrame with ['timestamp', 'behavioral_state']
            lookback_days: Days to analyze
            previous_score: Previous score for trend calculation

        Returns:
            HealthScore object with complete breakdown
        """
        # Calculate component scores
        temp_score = self._calculate_temperature_score(temperature_data, lookback_days)
        activity_score = self._calculate_activity_score(activity_data, lookback_days)
        alert_score = self._calculate_alert_score(alert_history, lookback_days)
        behavioral_score = self._calculate_behavioral_score(behavioral_states, lookback_days)

        component_scores = {
            'temperature': temp_score,
            'activity': activity_score,
            'alerts': alert_score,
            'behavioral': behavioral_score
        }

        # Calculate weighted overall score
        overall = (
            temp_score * self.weights['temperature'] +
            activity_score * self.weights['activity'] +
            alert_score * self.weights['alerts'] +
            behavioral_score * self.weights['behavioral']
        )

        # Determine category
        if overall >= 90:
            category = HealthCategory.EXCELLENT
        elif overall >= 75:
            category = HealthCategory.GOOD
        elif overall >= 60:
            category = HealthCategory.FAIR
        elif overall >= 40:
            category = HealthCategory.POOR
        else:
            category = HealthCategory.CRITICAL

        # Determine trend
        if previous_score is not None:
            diff = overall - previous_score
            if diff > 5:
                trend = "improving"
            elif diff < -5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "baseline"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            category, component_scores, trend
        )

        # Collect contributing factors
        factors = {
            'weights': self.weights,
            'lookback_days': lookback_days,
            'component_contributions': {
                k: round(v * self.weights[k.replace('_health', '')], 1)
                for k, v in component_scores.items()
            }
        }

        return HealthScore(
            timestamp=datetime.now(),
            cow_id=cow_id,
            overall_score=overall,
            category=category,
            component_scores=component_scores,
            factors=factors,
            trend=trend,
            recommendations=recommendations
        )

    def _calculate_temperature_score(
        self,
        temperature_data: pd.DataFrame,
        lookback_days: int
    ) -> float:
        """
        Calculate temperature health score (0-100).

        Scoring criteria:
        - Baseline deviation: Lower is better
        - Variance: Lower is better
        - Anomalies: Fewer is better
        """
        if len(temperature_data) == 0:
            return 50.0  # Neutral score if no data

        temp_df = temperature_data.copy()
        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'])

        # Filter to lookback window
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = temp_df[temp_df['timestamp'] >= cutoff]

        if len(recent) == 0:
            return 50.0

        temps = recent['temperature']

        # Calculate metrics
        mean_temp = temps.mean()
        std_temp = temps.std()
        baseline_deviation = abs(mean_temp - self.baseline_temp)

        # Count anomalies (outside 2 std dev)
        lower_bound = self.baseline_temp - 0.5
        upper_bound = self.baseline_temp + 1.0
        anomalies = ((temps < lower_bound) | (temps > upper_bound)).sum()
        anomaly_rate = anomalies / len(temps)

        # Score components
        # 1. Baseline deviation (40 points) - exponential penalty
        deviation_score = 40 * math.exp(-2 * baseline_deviation)

        # 2. Variance (30 points) - lower variance is better
        variance_score = 30 * math.exp(-10 * std_temp)

        # 3. Anomaly rate (30 points) - fewer anomalies is better
        anomaly_score = 30 * (1 - min(anomaly_rate * 5, 1.0))

        total_score = deviation_score + variance_score + anomaly_score

        return min(max(total_score, 0), 100)

    def _calculate_activity_score(
        self,
        activity_data: pd.DataFrame,
        lookback_days: int
    ) -> float:
        """
        Calculate activity health score (0-100).

        Scoring criteria:
        - Moderate activity is best (not too high, not too low)
        - Consistency is valued
        - Daily variation patterns
        """
        if len(activity_data) == 0:
            return 50.0

        activity_df = activity_data.copy()
        activity_df['timestamp'] = pd.to_datetime(activity_df['timestamp'])

        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = activity_df[activity_df['timestamp'] >= cutoff]

        if len(recent) == 0:
            return 50.0

        # Get activity column
        activity_col = None
        for col in ['movement_intensity', 'fxa', 'activity']:
            if col in recent.columns:
                activity_col = col
                break

        if activity_col is None:
            return 50.0

        activity = recent[activity_col]

        # Calculate metrics
        mean_activity = activity.mean()
        std_activity = activity.std()

        # Ideal activity range: 0.3 - 0.7
        # Score based on how close to ideal
        ideal_center = 0.5
        ideal_range = 0.2

        # 1. Activity level score (50 points)
        # Peak at 0.5, decline on either side
        activity_distance = abs(mean_activity - ideal_center)
        activity_level_score = 50 * math.exp(-5 * (activity_distance / ideal_range)**2)

        # 2. Consistency score (30 points)
        # Lower std dev is better (but not zero)
        ideal_std = 0.1
        consistency_score = 30 * math.exp(-10 * abs(std_activity - ideal_std))

        # 3. Daily variation (20 points)
        # Check for healthy circadian pattern
        if len(recent) >= 24 * 60:  # At least 1 day
            recent['hour'] = recent['timestamp'].dt.hour
            hourly_activity = recent.groupby('hour')[activity_col].mean()

            # Healthy pattern: lower at night (22-4), higher during day (8-18)
            night_hours = list(range(22, 24)) + list(range(0, 5))
            day_hours = list(range(8, 19))

            night_activity = hourly_activity[hourly_activity.index.isin(night_hours)].mean()
            day_activity = hourly_activity[hourly_activity.index.isin(day_hours)].mean()

            # Healthy ratio: day/night should be > 1.2
            if not pd.isna(night_activity) and night_activity > 0:
                ratio = day_activity / night_activity
                variation_score = 20 * min(ratio / 1.5, 1.0)
            else:
                variation_score = 10
        else:
            variation_score = 15  # Default if not enough data

        total_score = activity_level_score + consistency_score + variation_score

        return min(max(total_score, 0), 100)

    def _calculate_alert_score(
        self,
        alert_history: List[Dict[str, Any]],
        lookback_days: int
    ) -> float:
        """
        Calculate alert-based health score (0-100).

        Scoring criteria:
        - Fewer alerts is better
        - Critical alerts penalize more
        - Recent alerts penalize more
        """
        # Convert to DataFrame
        if len(alert_history) == 0:
            return 100.0  # Perfect score if no alerts

        alerts_df = pd.DataFrame(alert_history)

        if 'timestamp' not in alerts_df.columns:
            return 100.0

        alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])

        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_alerts = alerts_df[alerts_df['timestamp'] >= cutoff]

        if len(recent_alerts) == 0:
            return 100.0

        # Count by severity
        severity_counts = {}
        if 'severity' in recent_alerts.columns:
            for severity in recent_alerts['severity']:
                severity_str = str(severity).lower()
                severity_counts[severity_str] = severity_counts.get(severity_str, 0) + 1
        else:
            severity_counts['unknown'] = len(recent_alerts)

        # Calculate penalty
        critical_count = severity_counts.get('critical', 0)
        high_count = severity_counts.get('high', 0)
        medium_count = severity_counts.get('medium', 0)
        low_count = severity_counts.get('low', 0)

        # Weighted penalty (critical alerts count more)
        total_penalty = (
            critical_count * 20 +
            high_count * 10 +
            medium_count * 5 +
            low_count * 2
        )

        # Score starts at 100 and decreases with alerts
        score = 100 - total_penalty

        return min(max(score, 0), 100)

    def _calculate_behavioral_score(
        self,
        behavioral_states: pd.DataFrame,
        lookback_days: int
    ) -> float:
        """
        Calculate behavioral health score (0-100).

        Scoring criteria:
        - Behavioral diversity (Shannon entropy)
        - Appropriate time in each state
        - State transition patterns
        """
        if len(behavioral_states) == 0:
            return 50.0

        states_df = behavioral_states.copy()
        states_df['timestamp'] = pd.to_datetime(states_df['timestamp'])

        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = states_df[states_df['timestamp'] >= cutoff]

        if len(recent) == 0:
            return 50.0

        # Get state column
        state_col = None
        for col in ['behavioral_state', 'state', 'behavior']:
            if col in recent.columns:
                state_col = col
                break

        if state_col is None:
            return 50.0

        states = recent[state_col]

        # 1. Behavioral diversity (50 points)
        # Using Shannon entropy (normalized)
        state_counts = states.value_counts()
        probabilities = state_counts / len(states)

        entropy = -sum(probabilities * np.log2(probabilities + 1e-10))
        max_entropy = np.log2(len(state_counts))  # Maximum possible entropy

        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0

        diversity_score = 50 * normalized_entropy

        # 2. Appropriate state distribution (50 points)
        # Ideal distribution (example):
        # Lying: 30-40%, Standing: 20-30%, Walking: 10-20%,
        # Ruminating: 15-25%, Feeding: 10-15%

        ideal_ranges = {
            'lying': (0.30, 0.40),
            'standing': (0.20, 0.30),
            'walking': (0.10, 0.20),
            'ruminating': (0.15, 0.25),
            'feeding': (0.10, 0.15)
        }

        distribution_score = 0
        for state, (min_pct, max_pct) in ideal_ranges.items():
            if state in state_counts.index:
                actual_pct = state_counts[state] / len(states)

                if min_pct <= actual_pct <= max_pct:
                    # Within ideal range
                    distribution_score += 10
                else:
                    # Outside range, partial credit
                    deviation = min(
                        abs(actual_pct - min_pct),
                        abs(actual_pct - max_pct)
                    )
                    distribution_score += 10 * math.exp(-10 * deviation)

        total_score = diversity_score + distribution_score

        return min(max(total_score, 0), 100)

    def _generate_recommendations(
        self,
        category: HealthCategory,
        component_scores: Dict[str, float],
        trend: str
    ) -> List[str]:
        """Generate action recommendations based on score"""
        recommendations = []

        # Overall recommendations
        if category == HealthCategory.CRITICAL:
            recommendations.append("URGENT: Immediate veterinary attention required")
            recommendations.append("Isolate cow and monitor closely")
        elif category == HealthCategory.POOR:
            recommendations.append("Schedule veterinary examination within 24 hours")
            recommendations.append("Increase monitoring frequency")
        elif category == HealthCategory.FAIR:
            recommendations.append("Monitor daily for changes")
            recommendations.append("Consider preventive care consultation")
        elif category == HealthCategory.GOOD:
            recommendations.append("Maintain current care routine")
        else:  # EXCELLENT
            recommendations.append("Continue excellent care practices")

        # Component-specific recommendations
        if component_scores['temperature'] < 60:
            recommendations.append("Temperature irregularities detected - check for fever or infection")

        if component_scores['activity'] < 60:
            recommendations.append("Low activity levels - check for lameness or pain")

        if component_scores['alerts'] < 70:
            recommendations.append("Multiple alerts detected - review alert history")

        if component_scores['behavioral'] < 60:
            recommendations.append("Abnormal behavioral patterns - investigate environmental factors")

        # Trend recommendations
        if trend == "declining":
            recommendations.append("Health declining - increase observation frequency")
        elif trend == "improving":
            recommendations.append("Health improving - continue current treatment")

        return recommendations
