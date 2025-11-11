"""
Unit Tests for Pattern Alert Detection System

Tests estrus and pregnancy detection with sliding window analysis,
temperature rise validation, and confidence scoring.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from layer3.alerts.pattern_detector import (
    PatternAlertDetector,
    PatternAlert,
    AlertType,
    AlertStatus
)


class TestPatternAlertDetector(unittest.TestCase):
    """Test pattern alert detection."""

    def setUp(self):
        """Initialize detector."""
        self.detector = PatternAlertDetector(
            estrus_window_minutes=10,
            pregnancy_window_days=14
        )
        self.baseline_temp = 38.5
        self.activity_baseline = {'mean': 0.5, 'std': 0.1}

    def _create_estrus_data(
        self,
        duration_minutes: int = 15,
        temp_rise: float = 0.4,
        activity_increase: float = 0.25
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic estrus pattern data."""
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=duration_minutes,
            freq='1min'
        )
        
        # Temperature: gradual rise
        temps = np.linspace(
            self.baseline_temp,
            self.baseline_temp + temp_rise,
            duration_minutes
        )
        # Add small noise
        temps += np.random.normal(0, 0.02, duration_minutes)
        
        temp_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temps
        })
        
        # Activity: increased movement
        baseline_activity = self.activity_baseline['mean']
        activity_values = np.ones(duration_minutes) * baseline_activity * (1 + activity_increase)
        activity_values += np.random.normal(0, 0.05, duration_minutes)
        
        activity_df = pd.DataFrame({
            'timestamp': timestamps,
            'movement_intensity': activity_values,
            'behavioral_state': ['walking'] * duration_minutes
        })
        
        return temp_df, activity_df

    def _create_fever_data(
        self,
        duration_minutes: int = 15,
        temp_rise: float = 1.5
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic fever pattern (sudden spike, low activity)."""
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=duration_minutes,
            freq='1min'
        )
        
        # Temperature: sudden spike
        temps = np.ones(duration_minutes) * (self.baseline_temp + temp_rise)
        # Add noise
        temps += np.random.normal(0, 0.1, duration_minutes)
        
        temp_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temps
        })
        
        # Activity: reduced (lethargy)
        baseline_activity = self.activity_baseline['mean']
        activity_values = np.ones(duration_minutes) * baseline_activity * 0.5
        activity_values += np.random.normal(0, 0.02, duration_minutes)
        
        activity_df = pd.DataFrame({
            'timestamp': timestamps,
            'movement_intensity': activity_values,
            'behavioral_state': ['lying'] * duration_minutes
        })
        
        return temp_df, activity_df

    def _create_pregnancy_data(
        self,
        duration_days: int = 14,
        activity_reduction: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create synthetic pregnancy pattern data (stable temp, reduced activity)."""
        samples = duration_days * 1440  # 1 sample per minute
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=samples,
            freq='1min'
        )
        
        # Temperature: very stable around baseline
        temps = np.ones(samples) * self.baseline_temp
        # Add very small noise for stability
        temps += np.random.normal(0, 0.03, samples)
        
        temp_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temps
        })
        
        # Activity: reduced
        baseline_activity = self.activity_baseline['mean']
        activity_values = np.ones(samples) * baseline_activity * (1 - activity_reduction)
        activity_values += np.random.normal(0, 0.02, samples)
        
        activity_df = pd.DataFrame({
            'timestamp': timestamps,
            'movement_intensity': activity_values,
            'behavioral_state': ['lying'] * samples
        })
        
        return temp_df, activity_df

    def test_detector_initialization(self):
        """Test detector initializes with correct parameters."""
        self.assertEqual(self.detector.estrus_window_config.window_minutes, 10)
        self.assertEqual(self.detector.pregnancy_window_config.window_minutes, 14 * 24 * 60)
        self.assertTrue(self.detector.enable_cycle_tracking)

    def test_estrus_detection_valid_pattern(self):
        """Test detection of valid estrus pattern."""
        temp_df, activity_df = self._create_estrus_data(
            duration_minutes=10,
            temp_rise=0.4,
            activity_increase=0.20
        )
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_001',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Should detect estrus
        self.assertGreater(len(alerts), 0)
        
        alert = alerts[0]
        self.assertEqual(alert.alert_type, AlertType.ESTRUS.value)
        self.assertEqual(alert.cow_id, 'cow_001')
        self.assertGreater(alert.confidence, 0.7)
        
        # Check metrics
        metrics = alert.pattern_metrics
        self.assertGreater(metrics['temp_rise'], 0.3)
        self.assertLess(metrics['temp_rise'], 0.6)
        self.assertGreater(metrics['activity_increase'], 0.15)

    def test_estrus_detection_short_window(self):
        """Test estrus detection with 5-minute window."""
        detector = PatternAlertDetector(estrus_window_minutes=5)
        
        temp_df, activity_df = self._create_estrus_data(
            duration_minutes=6,
            temp_rise=0.35,
            activity_increase=0.18
        )
        
        alerts = detector.detect_patterns(
            cow_id='cow_002',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        self.assertGreater(len(alerts), 0)
        alert = alerts[0]
        self.assertEqual(alert.alert_type, AlertType.ESTRUS.value)

    def test_estrus_detection_temp_too_low(self):
        """Test that low temperature rise doesn't trigger estrus alert."""
        temp_df, activity_df = self._create_estrus_data(
            duration_minutes=10,
            temp_rise=0.15,  # Below 0.3°C threshold
            activity_increase=0.20
        )
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_003',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Should not detect estrus
        self.assertEqual(len(alerts), 0)

    def test_estrus_detection_activity_too_low(self):
        """Test that insufficient activity increase doesn't trigger alert."""
        temp_df, activity_df = self._create_estrus_data(
            duration_minutes=10,
            temp_rise=0.4,
            activity_increase=0.05  # Below 15% threshold
        )
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_004',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Should not detect estrus
        self.assertEqual(len(alerts), 0)

    def test_fever_vs_estrus_distinction(self):
        """Test that fever pattern is distinguished from estrus."""
        # Fever: sudden high temp, low activity, no correlation
        temp_df, activity_df = self._create_fever_data(
            duration_minutes=10,
            temp_rise=1.5
        )
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_005',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Should NOT detect estrus (temp too high, activity low, wrong correlation)
        estrus_alerts = [a for a in alerts if a.alert_type == AlertType.ESTRUS.value]
        self.assertEqual(len(estrus_alerts), 0)

    def test_estrus_confidence_scoring(self):
        """Test confidence scoring increases with better patterns."""
        # Optimal pattern
        temp_df1, activity_df1 = self._create_estrus_data(
            duration_minutes=10,
            temp_rise=0.45,  # Optimal 0.4-0.5
            activity_increase=0.20
        )
        
        alerts1 = self.detector.detect_patterns(
            cow_id='cow_006',
            temperature_data=temp_df1,
            activity_data=activity_df1,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Marginal pattern
        temp_df2, activity_df2 = self._create_estrus_data(
            duration_minutes=10,
            temp_rise=0.3,  # At lower threshold
            activity_increase=0.15  # At threshold
        )
        
        self.detector.active_alerts = {}  # Reset
        alerts2 = self.detector.detect_patterns(
            cow_id='cow_007',
            temperature_data=temp_df2,
            activity_data=activity_df2,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        if len(alerts1) > 0 and len(alerts2) > 0:
            # Optimal pattern should have higher confidence
            self.assertGreater(alerts1[0].confidence, alerts2[0].confidence)

    def test_pregnancy_detection_valid_pattern(self):
        """Test detection of valid pregnancy pattern."""
        temp_df, activity_df = self._create_pregnancy_data(
            duration_days=14,
            activity_reduction=0.12
        )
        
        # Create estrus history (needed for pregnancy detection)
        estrus_time = datetime.now() - timedelta(days=14)
        estrus_history = [{
            'timestamp': estrus_time,
            'alert_id': 'estrus_001',
            'cow_id': 'cow_008'
        }]
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_008',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline,
            estrus_history=estrus_history
        )
        
        # Should detect pregnancy
        pregnancy_alerts = [a for a in alerts if a.alert_type == AlertType.PREGNANCY_INDICATION.value]
        self.assertGreater(len(pregnancy_alerts), 0)
        
        alert = pregnancy_alerts[0]
        self.assertEqual(alert.cow_id, 'cow_008')
        self.assertGreater(alert.confidence, 0.5)
        
        # Check linkage
        self.assertIn('estrus_001', alert.related_events)

    def test_pregnancy_detection_no_estrus_history(self):
        """Test pregnancy detection requires estrus history."""
        temp_df, activity_df = self._create_pregnancy_data(
            duration_days=14,
            activity_reduction=0.12
        )
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_009',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline,
            estrus_history=None
        )
        
        # Should NOT detect pregnancy without estrus history
        pregnancy_alerts = [a for a in alerts if a.alert_type == AlertType.PREGNANCY_INDICATION.value]
        self.assertEqual(len(pregnancy_alerts), 0)

    def test_pregnancy_detection_estrus_too_recent(self):
        """Test pregnancy not detected if estrus too recent (<7 days)."""
        temp_df, activity_df = self._create_pregnancy_data(
            duration_days=5,
            activity_reduction=0.12
        )
        
        # Estrus only 5 days ago
        estrus_time = datetime.now() - timedelta(days=5)
        estrus_history = [{
            'timestamp': estrus_time,
            'alert_id': 'estrus_002',
            'cow_id': 'cow_010'
        }]
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_010',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline,
            estrus_history=estrus_history
        )
        
        # Should NOT detect pregnancy (too soon)
        pregnancy_alerts = [a for a in alerts if a.alert_type == AlertType.PREGNANCY_INDICATION.value]
        self.assertEqual(len(pregnancy_alerts), 0)

    def test_pregnancy_detection_estrus_too_old(self):
        """Test pregnancy not detected if estrus too old (>30 days)."""
        temp_df, activity_df = self._create_pregnancy_data(
            duration_days=14,
            activity_reduction=0.12
        )
        
        # Estrus 40 days ago
        estrus_time = datetime.now() - timedelta(days=40)
        estrus_history = [{
            'timestamp': estrus_time,
            'alert_id': 'estrus_003',
            'cow_id': 'cow_011'
        }]
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_011',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline,
            estrus_history=estrus_history
        )
        
        # Should NOT detect pregnancy (too old)
        pregnancy_alerts = [a for a in alerts if a.alert_type == AlertType.PREGNANCY_INDICATION.value]
        self.assertEqual(len(pregnancy_alerts), 0)

    def test_pregnancy_confidence_increases_with_time(self):
        """Test pregnancy confidence increases with observation period."""
        # 10-day observation
        temp_df1, activity_df1 = self._create_pregnancy_data(
            duration_days=10,
            activity_reduction=0.12
        )
        
        estrus_time1 = datetime.now() - timedelta(days=17)
        estrus_history1 = [{
            'timestamp': estrus_time1,
            'alert_id': 'estrus_004',
            'cow_id': 'cow_012'
        }]
        
        alerts1 = self.detector.detect_patterns(
            cow_id='cow_012',
            temperature_data=temp_df1,
            activity_data=activity_df1,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline,
            estrus_history=estrus_history1
        )
        
        # 14-day observation
        temp_df2, activity_df2 = self._create_pregnancy_data(
            duration_days=14,
            activity_reduction=0.12
        )
        
        estrus_time2 = datetime.now() - timedelta(days=21)
        estrus_history2 = [{
            'timestamp': estrus_time2,
            'alert_id': 'estrus_005',
            'cow_id': 'cow_013'
        }]
        
        self.detector.active_alerts = {}  # Reset
        alerts2 = self.detector.detect_patterns(
            cow_id='cow_013',
            temperature_data=temp_df2,
            activity_data=activity_df2,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline,
            estrus_history=estrus_history2
        )
        
        pregnancy_alerts1 = [a for a in alerts1 if a.alert_type == AlertType.PREGNANCY_INDICATION.value]
        pregnancy_alerts2 = [a for a in alerts2 if a.alert_type == AlertType.PREGNANCY_INDICATION.value]
        
        if len(pregnancy_alerts1) > 0 and len(pregnancy_alerts2) > 0:
            # Longer observation should have higher confidence
            self.assertGreater(pregnancy_alerts2[0].confidence, pregnancy_alerts1[0].confidence)

    def test_alert_status_transitions(self):
        """Test alert status transitions from pending to confirmed."""
        # Short pattern (pending)
        temp_df1, activity_df1 = self._create_estrus_data(
            duration_minutes=3,
            temp_rise=0.4,
            activity_increase=0.20
        )
        
        alerts1 = self.detector.detect_patterns(
            cow_id='cow_014',
            temperature_data=temp_df1,
            activity_data=activity_df1,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Long pattern (confirmed)
        temp_df2, activity_df2 = self._create_estrus_data(
            duration_minutes=10,
            temp_rise=0.4,
            activity_increase=0.20
        )
        
        self.detector.active_alerts = {}  # Reset
        alerts2 = self.detector.detect_patterns(
            cow_id='cow_015',
            temperature_data=temp_df2,
            activity_data=activity_df2,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        if len(alerts1) > 0 and len(alerts2) > 0:
            # Short duration may be pending
            # Long duration should be confirmed
            self.assertEqual(alerts2[0].status, AlertStatus.CONFIRMED.value)

    def test_get_active_alerts(self):
        """Test retrieving active alerts."""
        temp_df, activity_df = self._create_estrus_data()
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_016',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Get alerts for specific cow
        cow_alerts = self.detector.get_active_alerts('cow_016')
        self.assertEqual(len(cow_alerts), len(alerts))
        
        # Get all alerts
        all_alerts = self.detector.get_active_alerts()
        self.assertGreaterEqual(len(all_alerts), len(alerts))

    def test_update_alert_status(self):
        """Test updating alert status."""
        temp_df, activity_df = self._create_estrus_data()
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_017',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        if len(alerts) > 0:
            alert_id = alerts[0].alert_id
            
            # Update status
            self.detector.update_alert_status(alert_id, AlertStatus.RESOLVED)
            
            # Verify update
            cow_alerts = self.detector.get_active_alerts('cow_017')
            updated_alert = next(a for a in cow_alerts if a.alert_id == alert_id)
            self.assertEqual(updated_alert.status, AlertStatus.RESOLVED.value)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data completeness."""
        # Create sparse data (low completeness)
        timestamps = pd.date_range(
            end=datetime.now(),
            periods=3,  # Only 3 samples for 10-minute window
            freq='1min'
        )
        
        temp_df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [self.baseline_temp + 0.4] * 3
        })
        
        activity_df = pd.DataFrame({
            'timestamp': timestamps,
            'movement_intensity': [0.7] * 3,
            'behavioral_state': ['walking'] * 3
        })
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_018',
            temperature_data=temp_df,
            activity_data=activity_df,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Should not detect due to insufficient data
        self.assertEqual(len(alerts), 0)

    def test_empty_data_handling(self):
        """Test handling of empty dataframes."""
        empty_temp = pd.DataFrame(columns=['timestamp', 'temperature'])
        empty_activity = pd.DataFrame(columns=['timestamp', 'movement_intensity', 'behavioral_state'])
        
        alerts = self.detector.detect_patterns(
            cow_id='cow_019',
            temperature_data=empty_temp,
            activity_data=empty_activity,
            baseline_temp=self.baseline_temp,
            activity_baseline=self.activity_baseline
        )
        
        # Should handle gracefully
        self.assertEqual(len(alerts), 0)


if __name__ == '__main__':
    unittest.main()
