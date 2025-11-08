"""
Unit Tests for Malfunction Detection Module

Tests cover:
- Connectivity loss detection (>5 min gaps)
- Stuck value detection (>2 hours identical readings)
- Out-of-range detection (temp, acceleration, angular velocity)
- Alert generation and deduplication
- Per-sensor tracking (7 parameters independently)
- Edge cases and boundary conditions
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.malfunction_detection import (
    MalfunctionDetector,
    MalfunctionAlert,
    MalfunctionType,
    AlertSeverity,
    ConnectivityLossDetector,
    StuckValueDetector,
    OutOfRangeDetector,
)


class TestMalfunctionAlert(unittest.TestCase):
    """Test MalfunctionAlert class."""
    
    def test_create_alert(self):
        """Test creating a malfunction alert."""
        alert = MalfunctionAlert(
            timestamp=datetime.now(),
            malfunction_type=MalfunctionType.CONNECTIVITY_LOSS,
            affected_sensors=['all'],
            details={'message': 'Test alert'},
            severity=AlertSeverity.CRITICAL,
        )
        
        self.assertEqual(alert.malfunction_type, MalfunctionType.CONNECTIVITY_LOSS)
        self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
        self.assertEqual(alert.affected_sensors, ['all'])
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        alert = MalfunctionAlert(
            timestamp=timestamp,
            malfunction_type=MalfunctionType.STUCK_VALUES,
            affected_sensors=['temperature'],
            details={'stuck_value': 38.5, 'duration_minutes': 150},
            severity=AlertSeverity.WARNING,
        )
        
        alert_dict = alert.to_dict()
        
        self.assertEqual(alert_dict['malfunction_type'], 'stuck_values')
        self.assertEqual(alert_dict['severity'], 'WARNING')
        self.assertEqual(alert_dict['affected_sensors'], ['temperature'])
        self.assertIn('details', alert_dict)
    
    def test_alert_id_generation(self):
        """Test unique alert ID generation."""
        alert1 = MalfunctionAlert(
            timestamp=datetime.now(),
            malfunction_type=MalfunctionType.STUCK_VALUES,
            affected_sensors=['temperature', 'fxa'],
            details={},
            severity=AlertSeverity.WARNING,
        )
        
        alert2 = MalfunctionAlert(
            timestamp=datetime.now(),
            malfunction_type=MalfunctionType.STUCK_VALUES,
            affected_sensors=['fxa', 'temperature'],  # Different order
            details={},
            severity=AlertSeverity.WARNING,
        )
        
        # Should generate same ID (sorted sensors)
        self.assertEqual(alert1.alert_id, alert2.alert_id)


class TestConnectivityLossDetector(unittest.TestCase):
    """Test connectivity loss detection."""
    
    def test_no_connectivity_loss(self):
        """Test with continuous data (no gaps)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'temperature': [38.5] * 10,
            'fxa': [0.1] * 10,
        })
        
        detector = ConnectivityLossDetector(gap_threshold_minutes=5)
        alerts = detector.check(data)
        
        self.assertEqual(len(alerts), 0)
    
    def test_connectivity_loss_6_minute_gap(self):
        """Test detection of 6-minute gap (exceeds 5-minute threshold)."""
        timestamps = list(pd.date_range('2024-01-01 00:00', periods=5, freq='1min'))
        # Add 6-minute gap
        timestamps.append(timestamps[-1] + timedelta(minutes=6))
        timestamps.extend(pd.date_range(timestamps[-1] + timedelta(minutes=1), periods=4, freq='1min'))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * len(timestamps),
            'fxa': [0.1] * len(timestamps),
        })
        
        detector = ConnectivityLossDetector(gap_threshold_minutes=5)
        alerts = detector.check(data)
        
        # Should detect the connectivity loss
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].malfunction_type, MalfunctionType.CONNECTIVITY_LOSS)
        self.assertEqual(alerts[0].severity, AlertSeverity.CRITICAL)
        self.assertIn('all', alerts[0].affected_sensors)
        self.assertGreater(alerts[0].details['gap_minutes'], 5.0)
    
    def test_multiple_connectivity_losses(self):
        """Test detection of multiple gaps."""
        timestamps = list(pd.date_range('2024-01-01 00:00', periods=5, freq='1min'))
        # First gap (6 minutes)
        timestamps.append(timestamps[-1] + timedelta(minutes=6))
        timestamps.extend(pd.date_range(timestamps[-1] + timedelta(minutes=1), periods=3, freq='1min'))
        # Second gap (10 minutes)
        timestamps.append(timestamps[-1] + timedelta(minutes=10))
        timestamps.extend(pd.date_range(timestamps[-1] + timedelta(minutes=1), periods=3, freq='1min'))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * len(timestamps),
        })
        
        detector = ConnectivityLossDetector(gap_threshold_minutes=5)
        alerts = detector.check(data)
        
        # Should detect both gaps
        self.assertEqual(len(alerts), 2)
    
    def test_exactly_5_minute_gap(self):
        """Test with exactly 5-minute gap (should not trigger)."""
        timestamps = list(pd.date_range('2024-01-01 00:00', periods=5, freq='1min'))
        # Exactly 5-minute gap
        timestamps.append(timestamps[-1] + timedelta(minutes=5))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * len(timestamps),
        })
        
        detector = ConnectivityLossDetector(gap_threshold_minutes=5)
        alerts = detector.check(data)
        
        # Should NOT detect (threshold is >, not >=)
        self.assertEqual(len(alerts), 0)


class TestStuckValueDetector(unittest.TestCase):
    """Test stuck value detection."""
    
    def test_no_stuck_values(self):
        """Test with varying sensor values."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=150, freq='1min'),
            'temperature': np.linspace(38.0, 39.0, 150),
            'fxa': np.random.uniform(-0.5, 0.5, 150),
        })
        
        detector = StuckValueDetector(stuck_threshold_minutes=120)
        alerts = detector.check(data)
        
        self.assertEqual(len(alerts), 0)
    
    def test_stuck_value_121_minutes(self):
        """Test detection of value stuck for 121 minutes (exceeds 120-minute threshold)."""
        # 121 data points at 1-minute intervals = 121 minutes
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=121, freq='1min'),
            'temperature': [38.5] * 121,  # Stuck at 38.5
            'fxa': np.random.uniform(-0.5, 0.5, 121),  # Not stuck
            'mya': [0.2] * 121,
            'rza': [-0.85] * 121,
            'sxg': [2.0] * 121,
            'lyg': [-1.5] * 121,
            'dzg': [0.5] * 121,
        })
        
        detector = StuckValueDetector(stuck_threshold_minutes=120)
        alerts = detector.check(data)
        
        # Should detect stuck temperature
        self.assertGreater(len(alerts), 0)
        temp_alerts = [a for a in alerts if 'temperature' in a.affected_sensors]
        self.assertEqual(len(temp_alerts), 1)
        self.assertEqual(temp_alerts[0].malfunction_type, MalfunctionType.STUCK_VALUES)
        self.assertEqual(temp_alerts[0].severity, AlertSeverity.WARNING)
        self.assertAlmostEqual(temp_alerts[0].details['stuck_value'], 38.5, places=2)
    
    def test_stuck_value_exactly_120_minutes(self):
        """Test with exactly 120 minutes (should trigger at threshold)."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=120, freq='1min'),
            'temperature': [38.5] * 120,
        })
        
        detector = StuckValueDetector(stuck_threshold_minutes=120)
        alerts = detector.check(data)
        
        # Should detect at exactly 120 minutes
        self.assertGreater(len(alerts), 0)
    
    def test_multiple_sensors_stuck(self):
        """Test detection when multiple sensors are stuck."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': [38.5] * 130,  # Stuck
            'fxa': [0.1] * 130,  # Stuck
            'mya': np.random.uniform(-0.5, 0.5, 130),  # Not stuck
            'rza': [-0.85] * 130,  # Stuck
            'sxg': [2.0] * 130,  # Stuck
            'lyg': [-1.5] * 130,  # Stuck
            'dzg': [0.5] * 130,  # Stuck
        })
        
        detector = StuckValueDetector(stuck_threshold_minutes=120)
        alerts = detector.check(data)
        
        # Should detect multiple stuck sensors (6 out of 7)
        self.assertGreaterEqual(len(alerts), 6)
        
        # Check each sensor independently
        stuck_sensors = [a.affected_sensors[0] for a in alerts]
        self.assertIn('temperature', stuck_sensors)
        self.assertIn('fxa', stuck_sensors)
        self.assertIn('rza', stuck_sensors)
        self.assertNotIn('mya', stuck_sensors)  # Not stuck
    
    def test_floating_point_tolerance(self):
        """Test that small floating point variations don't trigger stuck detection."""
        # Values with small variations (within 0.001 tolerance)
        temperatures = [38.5 + i * 0.0001 for i in range(130)]
        
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': temperatures,
        })
        
        detector = StuckValueDetector(stuck_threshold_minutes=120)
        alerts = detector.check(data)
        
        # Should detect as stuck (variations within tolerance)
        temp_alerts = [a for a in alerts if 'temperature' in a.affected_sensors]
        self.assertGreater(len(temp_alerts), 0)


class TestOutOfRangeDetector(unittest.TestCase):
    """Test out-of-range detection."""
    
    def test_no_out_of_range(self):
        """Test with all values in normal range."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
            'temperature': [38.5] * 10,
            'fxa': [0.1] * 10,
            'mya': [0.2] * 10,
            'rza': [-0.85] * 10,
            'sxg': [2.0] * 10,
            'lyg': [-1.5] * 10,
            'dzg': [0.5] * 10,
        })
        
        detector = OutOfRangeDetector()
        alerts = detector.check(data)
        
        self.assertEqual(len(alerts), 0)
    
    def test_temperature_out_of_range_low(self):
        """Test detection of temperature <35°C."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [38.5, 34.5, 34.0, 38.6, 38.7],
        })
        
        detector = OutOfRangeDetector()
        alerts = detector.check(data)
        
        # Should detect 2 out-of-range temperatures
        self.assertEqual(len(alerts), 2)
        for alert in alerts:
            self.assertEqual(alert.malfunction_type, MalfunctionType.OUT_OF_RANGE)
            self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
            self.assertIn('temperature', alert.affected_sensors)
            self.assertLess(alert.details['value'], 35.0)
    
    def test_temperature_out_of_range_high(self):
        """Test detection of temperature >42°C."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='1min'),
            'temperature': [38.5, 42.5, 43.0, 38.6],
        })
        
        detector = OutOfRangeDetector()
        alerts = detector.check(data)
        
        # Should detect 2 out-of-range temperatures
        self.assertEqual(len(alerts), 2)
        for alert in alerts:
            self.assertGreater(alert.details['value'], 42.0)
    
    def test_acceleration_out_of_range(self):
        """Test detection of acceleration >10g."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=5, freq='1min'),
            'temperature': [38.5] * 5,
            'fxa': [0.1, 11.0, 0.2, -12.0, 0.3],  # 11.0 and -12.0 exceed 10g
            'mya': [0.2] * 5,
            'rza': [-0.85] * 5,
        })
        
        detector = OutOfRangeDetector()
        alerts = detector.check(data)
        
        # Should detect 2 out-of-range accelerations
        self.assertEqual(len(alerts), 2)
        for alert in alerts:
            self.assertEqual(alert.malfunction_type, MalfunctionType.OUT_OF_RANGE)
            self.assertEqual(alert.severity, AlertSeverity.CRITICAL)
            self.assertIn('fxa', alert.affected_sensors)
            self.assertGreater(abs(alert.details['value']), 10.0)
    
    def test_angular_velocity_out_of_range(self):
        """Test detection of angular velocity >500 deg/s."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=4, freq='1min'),
            'temperature': [38.5] * 4,
            'sxg': [2.0, 501.0, 2.0, -600.0],  # 501.0 and -600.0 exceed 500
            'lyg': [-1.5] * 4,
            'dzg': [0.5] * 4,
        })
        
        detector = OutOfRangeDetector()
        alerts = detector.check(data)
        
        # Should detect 2 out-of-range angular velocities
        self.assertEqual(len(alerts), 2)
        for alert in alerts:
            self.assertIn('sxg', alert.affected_sensors)
            self.assertGreater(abs(alert.details['value']), 500.0)
    
    def test_boundary_values(self):
        """Test with values exactly at boundaries."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=6, freq='1min'),
            'temperature': [35.0, 42.0, 34.9, 42.1, 38.5, 38.5],
            'fxa': [10.0, -10.0, 10.1, -10.1, 0.1, 0.1],
            'sxg': [500.0, -500.0, 500.1, -500.1, 2.0, 2.0],
        })
        
        detector = OutOfRangeDetector()
        alerts = detector.check(data)
        
        # 35.0 and 42.0 are AT boundary (should NOT trigger)
        # 34.9 and 42.1 are OUT of range (should trigger)
        # Similarly for accelerations and angular velocities
        temp_alerts = [a for a in alerts if 'temperature' in a.affected_sensors]
        self.assertEqual(len(temp_alerts), 2)  # Only 34.9 and 42.1


class TestMalfunctionDetector(unittest.TestCase):
    """Test main malfunction detector with all detection types."""
    
    def test_detect_all_malfunction_types(self):
        """Test detection of all malfunction types together."""
        # Create data with multiple issues
        timestamps = list(pd.date_range('2024-01-01 00:00', periods=5, freq='1min'))
        # Add connectivity gap
        timestamps.append(timestamps[-1] + timedelta(minutes=6))
        timestamps.extend(pd.date_range(timestamps[-1] + timedelta(minutes=1), periods=130, freq='1min'))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * 5 + [38.5] + [38.5] * 130,  # Stuck after gap
            'fxa': [0.1] * 5 + [0.1] + [0.1] * 130,
            'mya': [0.2] * 5 + [15.0] + [0.2] * 130,  # One out-of-range value
        })
        
        detector = MalfunctionDetector()
        alerts = detector.detect(data)
        
        # Should detect connectivity loss, stuck values, and out-of-range
        self.assertGreater(len(alerts), 0)
        
        malfunction_types = [a.malfunction_type for a in alerts]
        self.assertIn(MalfunctionType.CONNECTIVITY_LOSS, malfunction_types)
        self.assertIn(MalfunctionType.OUT_OF_RANGE, malfunction_types)
    
    def test_alert_deduplication(self):
        """Test that duplicate alerts are filtered out."""
        # Create data that would generate same alert twice
        data1 = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': [38.5] * 130,  # Stuck
        })
        
        detector = MalfunctionDetector(enable_deduplication=True, deduplication_window_minutes=60)
        
        # First detection
        alerts1 = detector.detect(data1)
        self.assertGreater(len(alerts1), 0)
        
        # Second detection with same data (should be deduplicated)
        data2 = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 02:00', periods=130, freq='1min'),
            'temperature': [38.5] * 130,  # Still stuck
        })
        alerts2 = detector.detect(data2)
        
        # Should have fewer alerts due to deduplication
        self.assertLessEqual(len(alerts2), len(alerts1))
    
    def test_no_deduplication(self):
        """Test with deduplication disabled."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': [38.5] * 130,
        })
        
        detector = MalfunctionDetector(enable_deduplication=False)
        
        # First detection
        alerts1 = detector.detect(data)
        
        # Second detection (should generate same alerts)
        alerts2 = detector.detect(data)
        
        # Without deduplication, should get same number of alerts
        self.assertEqual(len(alerts1), len(alerts2))
    
    def test_get_alert_summary(self):
        """Test alert summary generation."""
        timestamps = list(pd.date_range('2024-01-01 00:00', periods=5, freq='1min'))
        timestamps.append(timestamps[-1] + timedelta(minutes=6))
        timestamps.extend(pd.date_range(timestamps[-1] + timedelta(minutes=1), periods=130, freq='1min'))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [38.5] * 5 + [34.0] + [38.5] * 130,  # Out of range + stuck
            'fxa': [0.1] * 136,
        })
        
        detector = MalfunctionDetector()
        alerts = detector.detect(data)
        summary = detector.get_alert_summary(alerts)
        
        self.assertIn('total_alerts', summary)
        self.assertIn('by_type', summary)
        self.assertIn('by_severity', summary)
        self.assertIn('affected_sensors', summary)
        self.assertGreater(summary['total_alerts'], 0)
    
    def test_reset_detector(self):
        """Test resetting detector state."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': [38.5] * 130,
        })
        
        detector = MalfunctionDetector()
        alerts1 = detector.detect(data)
        
        # Reset detector
        detector.reset()
        
        # Should be able to detect again after reset
        alerts2 = detector.detect(data)
        self.assertEqual(len(alerts1), len(alerts2))


class TestPerSensorTracking(unittest.TestCase):
    """Test that each of 7 parameters is tracked independently."""
    
    def test_independent_sensor_tracking(self):
        """Test that each sensor is monitored independently."""
        # Create data where different sensors have different issues
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': [38.5] * 130,  # Stuck
            'fxa': np.random.uniform(-0.5, 0.5, 130),  # Normal
            'mya': [0.2] * 130,  # Stuck
            'rza': np.random.uniform(-0.5, 0.5, 130),  # Normal
            'sxg': [2.0] * 130,  # Stuck
            'lyg': np.random.uniform(-50, 50, 130),  # Normal
            'dzg': [0.5] * 130,  # Stuck
        })
        
        detector = MalfunctionDetector()
        alerts = detector.detect(data)
        
        # Should detect stuck values for temperature, mya, sxg, dzg only
        stuck_alerts = [a for a in alerts if a.malfunction_type == MalfunctionType.STUCK_VALUES]
        self.assertEqual(len(stuck_alerts), 4)
        
        stuck_sensors = set()
        for alert in stuck_alerts:
            stuck_sensors.update(alert.affected_sensors)
        
        self.assertIn('temperature', stuck_sensors)
        self.assertIn('mya', stuck_sensors)
        self.assertIn('sxg', stuck_sensors)
        self.assertIn('dzg', stuck_sensors)
        self.assertNotIn('fxa', stuck_sensors)
        self.assertNotIn('rza', stuck_sensors)
        self.assertNotIn('lyg', stuck_sensors)
    
    def test_all_seven_sensors_monitored(self):
        """Test that all 7 sensor parameters are monitored."""
        # Create data where all sensors are stuck
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': [38.5] * 130,
            'fxa': [0.1] * 130,
            'mya': [0.2] * 130,
            'rza': [-0.85] * 130,
            'sxg': [2.0] * 130,
            'lyg': [-1.5] * 130,
            'dzg': [0.5] * 130,
        })
        
        detector = MalfunctionDetector()
        alerts = detector.detect(data)
        
        # Should detect stuck values for all 7 sensors
        stuck_alerts = [a for a in alerts if a.malfunction_type == MalfunctionType.STUCK_VALUES]
        self.assertEqual(len(stuck_alerts), 7)
        
        # Verify all 7 sensors are in alerts
        all_sensors = {'temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg'}
        detected_sensors = set()
        for alert in stuck_alerts:
            detected_sensors.update(alert.affected_sensors)
        
        self.assertEqual(detected_sensors, all_sensors)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame(columns=['timestamp', 'temperature', 'fxa'])
        
        detector = MalfunctionDetector()
        alerts = detector.detect(data)
        
        self.assertEqual(len(alerts), 0)
    
    def test_single_record(self):
        """Test with single record."""
        data = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01')],
            'temperature': [38.5],
        })
        
        detector = MalfunctionDetector()
        alerts = detector.detect(data)
        
        # Should not detect any issues with single record
        self.assertEqual(len(alerts), 0)
    
    def test_missing_timestamp_column(self):
        """Test handling of missing timestamp column."""
        data = pd.DataFrame({
            'temperature': [38.5] * 10,
        })
        
        detector = MalfunctionDetector()
        # Should handle gracefully without crashing
        alerts = detector.detect(data)
        
        # May or may not generate alerts, but should not crash
        self.assertIsInstance(alerts, list)
    
    def test_nan_values(self):
        """Test handling of NaN values in sensor data."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=130, freq='1min'),
            'temperature': [38.5] * 65 + [np.nan] * 65,  # Half NaN
        })
        
        detector = MalfunctionDetector()
        alerts = detector.detect(data)
        
        # Should handle NaN values without crashing
        self.assertIsInstance(alerts, list)


if __name__ == '__main__':
    unittest.main()
