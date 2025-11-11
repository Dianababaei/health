"""
Unit tests for Multi-Day Health Trend Tracker.

Tests time-window aggregation, trend classification, and dashboard output formatting.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from health_intelligence.trend_tracker import (
    MultiDayHealthTrendTracker,
    TrendIndicator,
    TimeWindowMetrics,
    HealthTrendReport
)


class TestMultiDayHealthTrendTracker(unittest.TestCase):
    """Test cases for health trend tracker."""

    def setUp(self):
        """Initialize tracker for tests."""
        self.tracker = MultiDayHealthTrendTracker(temperature_baseline=38.5)

    def _create_temperature_data(
        self,
        days: int,
        baseline: float = 38.5,
        pattern: str = 'stable'
    ) -> pd.DataFrame:
        """Create synthetic temperature data."""
        samples = days * 1440  # 1 per minute
        dates = pd.date_range(end=datetime.now(), periods=samples, freq='1min')

        if pattern == 'improving':
            # Temperature decreasing toward baseline
            temp_values = baseline + 0.5 - (np.arange(samples) / samples) * 0.5
            temp_values += np.random.normal(0, 0.08, samples)
        elif pattern == 'deteriorating':
            # Temperature rising with instability
            temp_values = baseline + (np.arange(samples) / samples) * 1.0
            temp_values += np.random.normal(0, 0.3, samples)
        else:  # stable
            temp_values = np.ones(samples) * baseline
            temp_values += np.random.normal(0, 0.1, samples)

        return pd.DataFrame({
            'timestamp': dates,
            'temperature': temp_values
        })

    def _create_activity_data(
        self,
        days: int,
        pattern: str = 'stable'
    ) -> pd.DataFrame:
        """Create synthetic activity data."""
        samples = days * 1440
        dates = pd.date_range(end=datetime.now(), periods=samples, freq='1min')

        if pattern == 'improving':
            movement = 0.3 + (np.arange(samples) / samples) * 0.3  # Increasing
            states = np.random.choice(
                ['lying', 'standing', 'walking', 'ruminating', 'feeding'],
                samples,
                p=[0.30, 0.30, 0.20, 0.10, 0.10]
            )
        elif pattern == 'deteriorating':
            movement = 0.6 - (np.arange(samples) / samples) * 0.4  # Decreasing
            states = np.random.choice(
                ['lying', 'standing', 'walking', 'ruminating', 'feeding'],
                samples,
                p=[0.70, 0.15, 0.05, 0.05, 0.05]  # Mostly lying
            )
        else:  # stable
            movement = np.ones(samples) * 0.5
            states = np.random.choice(
                ['lying', 'standing', 'walking', 'ruminating', 'feeding'],
                samples,
                p=[0.40, 0.25, 0.15, 0.10, 0.10]
            )

        return pd.DataFrame({
            'timestamp': dates,
            'behavioral_state': states,
            'movement_intensity': movement
        })

    def _create_alert_history(
        self,
        days: int,
        alerts_per_day: float = 1.0
    ) -> list:
        """Create synthetic alert history."""
        num_alerts = int(days * alerts_per_day)
        alerts = []

        end_date = datetime.now()
        for i in range(num_alerts):
            # Distribute alerts randomly across period
            days_back = np.random.uniform(0, days)
            alert_time = end_date - timedelta(days=days_back)

            alerts.append({
                'timestamp': alert_time,
                'alert_type': np.random.choice(['fever', 'heat_stress', 'inactivity', 'sensor']),
                'severity': np.random.choice(['critical', 'warning'], p=[0.3, 0.7]),
                'cow_id': 'test_cow_001'
            })

        return alerts

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        self.assertEqual(self.tracker.temperature_baseline, 38.5)
        self.assertEqual(self.tracker.periods, [7, 14, 30, 90])

    def test_analyze_trends_stable(self):
        """Test trend analysis with stable data."""
        temp_data = self._create_temperature_data(30, pattern='stable')
        activity_data = self._create_activity_data(30, pattern='stable')
        alert_history = self._create_alert_history(30, alerts_per_day=0.5)
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_001',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should have data for 7, 14, 30 day periods
        self.assertIsNotNone(report.trend_7day)
        self.assertIsNotNone(report.trend_14day)
        self.assertIsNotNone(report.trend_30day)

        # Overall trend should be stable or improving (both indicate health)
        self.assertIn(report.overall_trend, [TrendIndicator.STABLE, TrendIndicator.IMPROVING])

        # Confidence should be reasonable
        self.assertGreater(report.overall_confidence, 0.5)

    def test_analyze_trends_improving(self):
        """Test trend analysis with improving data."""
        temp_data = self._create_temperature_data(14, pattern='improving')
        activity_data = self._create_activity_data(14, pattern='improving')
        alert_history = self._create_alert_history(14, alerts_per_day=0.2)  # Few alerts
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_002',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should have improving trends
        self.assertIn(
            report.overall_trend,
            [TrendIndicator.IMPROVING, TrendIndicator.STABLE]
        )

    def test_analyze_trends_deteriorating(self):
        """Test trend analysis with deteriorating data."""
        temp_data = self._create_temperature_data(14, pattern='deteriorating')
        activity_data = self._create_activity_data(14, pattern='deteriorating')
        alert_history = self._create_alert_history(14, alerts_per_day=3.0)  # Many alerts
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_003',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should detect deterioration
        self.assertIn(
            report.overall_trend,
            [TrendIndicator.DETERIORATING, TrendIndicator.STABLE]
        )

        # Should generate recommendations
        self.assertGreater(len(report.recommendations), 0)

    def test_insufficient_data_handling(self):
        """Test graceful handling of insufficient data."""
        # Only 2 days of data
        temp_data = self._create_temperature_data(2, pattern='stable')
        activity_data = self._create_activity_data(2, pattern='stable')
        alert_history = []
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_004',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should have no data for longer periods
        self.assertIsNone(report.trend_7day)
        self.assertIsNone(report.trend_14day)
        self.assertIsNone(report.trend_30day)
        self.assertIsNone(report.trend_90day)

        # Overall trend should indicate insufficient data
        self.assertEqual(report.overall_trend, TrendIndicator.INSUFFICIENT_DATA)

    def test_temperature_metrics_calculation(self):
        """Test temperature metric calculations."""
        temp_data = self._create_temperature_data(7, baseline=38.5, pattern='stable')
        period_data = temp_data

        metrics = self.tracker._calculate_temperature_metrics(period_data)

        self.assertIn('mean', metrics)
        self.assertIn('std', metrics)
        self.assertIn('baseline_drift', metrics)
        self.assertIn('anomaly_count', metrics)

        # Mean should be close to baseline
        self.assertAlmostEqual(metrics['mean'], 38.5, delta=0.3)

    def test_activity_metrics_calculation(self):
        """Test activity metric calculations."""
        activity_data = self._create_activity_data(7, pattern='stable')

        metrics = self.tracker._calculate_activity_metrics(activity_data)

        self.assertIn('total_minutes', metrics)
        self.assertIn('mean_level', metrics)
        self.assertIn('rest_minutes', metrics)
        self.assertIn('diversity', metrics)

        # Diversity should be reasonable for varied states
        self.assertGreater(metrics['diversity'], 0.5)

    def test_alert_metrics_calculation(self):
        """Test alert metric calculations."""
        alert_history = self._create_alert_history(7, alerts_per_day=2.0)
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()

        metrics = self.tracker._calculate_alert_metrics(alert_history, start_date, end_date)

        self.assertIn('count', metrics)
        self.assertIn('severity_dist', metrics)
        self.assertIn('type_dist', metrics)

        # Should have approximately 14 alerts (7 days * 2/day)
        self.assertGreater(metrics['count'], 10)
        self.assertLess(metrics['count'], 20)

    def test_behavioral_metrics_calculation(self):
        """Test behavioral metric calculations."""
        behavioral_data = self._create_activity_data(7, pattern='stable')

        metrics = self.tracker._calculate_behavioral_metrics(behavioral_data, period_days=7)

        self.assertIn('state_dist', metrics)
        self.assertIn('changes_per_day', metrics)

        # State distribution should sum to ~100%
        total_pct = sum(metrics['state_dist'].values())
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

    def test_period_comparisons(self):
        """Test period-over-period comparisons."""
        temp_data = self._create_temperature_data(30, pattern='stable')
        activity_data = self._create_activity_data(30, pattern='stable')
        alert_history = self._create_alert_history(30, alerts_per_day=1.0)
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_005',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should have comparisons
        self.assertIn('7d_vs_14d', report.period_comparisons)
        self.assertIn('14d_vs_30d', report.period_comparisons)

        # Each comparison should have deltas
        comp = report.period_comparisons['7d_vs_14d']
        self.assertIn('temperature_delta', comp)
        self.assertIn('activity_delta', comp)
        self.assertIn('alert_delta', comp)

    def test_significant_changes_detection(self):
        """Test detection of significant changes."""
        # Create data with high alert frequency
        temp_data = self._create_temperature_data(14, pattern='stable')
        activity_data = self._create_activity_data(14, pattern='stable')
        alert_history = self._create_alert_history(14, alerts_per_day=5.0)  # High frequency
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_006',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should identify high alert frequency
        self.assertGreater(len(report.significant_changes), 0)
        self.assertTrue(any('alert' in change.lower() for change in report.significant_changes))

    def test_recommendations_generation(self):
        """Test recommendation generation."""
        temp_data = self._create_temperature_data(14, pattern='deteriorating')
        activity_data = self._create_activity_data(14, pattern='deteriorating')
        alert_history = self._create_alert_history(14, alerts_per_day=3.0)
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_007',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should have recommendations
        self.assertGreater(len(report.recommendations), 0)

        # For deteriorating trend, should recommend veterinary review
        if report.overall_trend == TrendIndicator.DETERIORATING:
            self.assertTrue(
                any('veterinary' in rec.lower() for rec in report.recommendations)
            )

    def test_dashboard_output_format(self):
        """Test that output is properly formatted for dashboard."""
        temp_data = self._create_temperature_data(14, pattern='stable')
        activity_data = self._create_activity_data(14, pattern='stable')
        alert_history = self._create_alert_history(14, alerts_per_day=1.0)
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_008',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Convert to dictionary
        output = report.to_dict()

        # Verify structure
        self.assertIn('cow_id', output)
        self.assertIn('analysis_timestamp', output)
        self.assertIn('overall_trend', output)
        self.assertIn('overall_confidence', output)
        self.assertIn('period_comparisons', output)
        self.assertIn('significant_changes', output)
        self.assertIn('recommendations', output)

        # Verify period data structure
        if output['trend_7day'] is not None:
            period = output['trend_7day']
            self.assertIn('temperature', period)
            self.assertIn('activity', period)
            self.assertIn('alerts', period)
            self.assertIn('behavioral', period)
            self.assertIn('trend_indicator', period)
            self.assertIn('confidence', period)

    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_df = pd.DataFrame()
        alert_history = []

        report = self.tracker.analyze_trends(
            cow_id='test_cow_009',
            temperature_data=empty_df,
            activity_data=empty_df,
            alert_history=alert_history,
            behavioral_states=empty_df
        )

        # Should handle gracefully
        self.assertEqual(report.overall_trend, TrendIndicator.INSUFFICIENT_DATA)
        self.assertEqual(report.overall_confidence, 0.0)

    def test_multiple_period_coverage(self):
        """Test that all 4 periods are analyzed when sufficient data exists."""
        # Create 90+ days of data
        temp_data = self._create_temperature_data(95, pattern='stable')
        activity_data = self._create_activity_data(95, pattern='stable')
        alert_history = self._create_alert_history(95, alerts_per_day=1.0)
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_010',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Should have all 4 periods
        self.assertIsNotNone(report.trend_7day)
        self.assertIsNotNone(report.trend_14day)
        self.assertIsNotNone(report.trend_30day)
        self.assertIsNotNone(report.trend_90day)

    def test_data_completeness_scoring(self):
        """Test that data completeness affects confidence scores."""
        # Create sparse data (only 70% complete)
        full_samples = 14 * 1440
        sparse_samples = int(full_samples * 0.7)

        dates = pd.date_range(end=datetime.now(), periods=sparse_samples, freq='1min')
        temp_data = pd.DataFrame({
            'timestamp': dates,
            'temperature': np.random.normal(38.5, 0.1, sparse_samples)
        })
        activity_data = pd.DataFrame({
            'timestamp': dates,
            'behavioral_state': np.random.choice(
                ['lying', 'standing', 'walking'],
                sparse_samples
            ),
            'movement_intensity': np.random.uniform(0.3, 0.7, sparse_samples)
        })

        alert_history = self._create_alert_history(14, alerts_per_day=1.0)
        behavioral_states = activity_data.copy()

        report = self.tracker.analyze_trends(
            cow_id='test_cow_011',
            temperature_data=temp_data,
            activity_data=activity_data,
            alert_history=alert_history,
            behavioral_states=behavioral_states
        )

        # Confidence should reflect data completeness
        if report.trend_14day is not None:
            self.assertLess(report.trend_14day.data_completeness, 0.8)


if __name__ == '__main__':
    unittest.main()
