"""
Unit Tests for Alert Logging System

Tests for AlertLogger class including JSON/CSV output,
log rotation, and retention policy.
"""

import unittest
import tempfile
import shutil
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_intelligence.logging.alert_logger import AlertLogger


class TestAlertLogger(unittest.TestCase):
    """Test cases for AlertLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for logs
        self.test_dir = tempfile.mkdtemp()
        self.logger = AlertLogger(
            log_dir=self.test_dir,
            retention_days=7,
            auto_cleanup=False
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test logger initialization."""
        self.assertTrue(Path(self.test_dir).exists())
        self.assertEqual(self.logger.retention_days, 7)
        self.assertEqual(self.logger.log_dir, Path(self.test_dir))
    
    def test_log_alert_json(self):
        """Test logging alert to JSON."""
        alert_data = {
            'alert_id': 'test-alert-001',
            'cow_id': 'COW001',
            'alert_type': 'fever',
            'severity': 'critical',
            'confidence': 0.95,
            'sensor_values': {'temperature': 40.5},
            'detection_details': {'threshold': 39.5},
            'timestamp': datetime.now().isoformat()
        }
        
        success = self.logger.log_alert(alert_data)
        self.assertTrue(success)
        
        # Check JSON file exists
        json_files = list(Path(self.test_dir).glob("alerts_COW001_*.json"))
        self.assertEqual(len(json_files), 1)
        
        # Read and verify content
        with open(json_files[0], 'r') as f:
            logged_alert = json.loads(f.readline())
            self.assertEqual(logged_alert['alert_id'], 'test-alert-001')
            self.assertEqual(logged_alert['cow_id'], 'COW001')
            self.assertEqual(logged_alert['alert_type'], 'fever')
    
    def test_log_alert_csv(self):
        """Test logging alert to CSV."""
        alert_data = {
            'alert_id': 'test-alert-002',
            'cow_id': 'COW002',
            'alert_type': 'heat_stress',
            'severity': 'high',
            'confidence': 0.85,
            'sensor_values': {'temperature': 39.8},
            'detection_details': {'duration_minutes': 120},
            'timestamp': datetime.now().isoformat()
        }
        
        success = self.logger.log_alert(alert_data)
        self.assertTrue(success)
        
        # Check CSV file exists
        csv_files = list(Path(self.test_dir).glob("alerts_COW002_*.csv"))
        self.assertEqual(len(csv_files), 1)
        
        # Read and verify content
        with open(csv_files[0], 'r') as f:
            reader = csv.DictReader(f)
            row = next(reader)
            self.assertEqual(row['alert_id'], 'test-alert-002')
            self.assertEqual(row['cow_id'], 'COW002')
    
    def test_log_multiple_alerts_same_cow_same_day(self):
        """Test logging multiple alerts for same cow on same day."""
        for i in range(3):
            alert_data = {
                'alert_id': f'test-alert-{i}',
                'cow_id': 'COW003',
                'alert_type': 'abnormal_behavior',
                'severity': 'warning',
                'confidence': 0.75,
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_alert(alert_data)
        
        # Should have only one JSON and one CSV file
        json_files = list(Path(self.test_dir).glob("alerts_COW003_*.json"))
        csv_files = list(Path(self.test_dir).glob("alerts_COW003_*.csv"))
        
        self.assertEqual(len(json_files), 1)
        self.assertEqual(len(csv_files), 1)
        
        # JSON file should have 3 lines
        with open(json_files[0], 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 3)
        
        # CSV file should have 4 lines (header + 3 data rows)
        with open(csv_files[0], 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 4)
    
    def test_log_alerts_batch(self):
        """Test batch logging of multiple alerts."""
        alerts = []
        for i in range(5):
            alerts.append({
                'alert_id': f'batch-alert-{i}',
                'cow_id': f'COW00{i}',
                'alert_type': 'test',
                'severity': 'info',
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            })
        
        success_count = self.logger.log_alerts_batch(alerts)
        self.assertEqual(success_count, 5)
    
    def test_prepare_alert_with_defaults(self):
        """Test alert preparation with default values."""
        minimal_alert = {'cow_id': 'COW004'}
        
        prepared = self.logger._prepare_alert(minimal_alert.copy())
        
        # Check defaults are added
        self.assertIn('alert_id', prepared)
        self.assertIn('timestamp', prepared)
        self.assertEqual(prepared['status'], 'active')
        self.assertEqual(prepared['severity'], 'info')
        self.assertEqual(prepared['confidence'], 0.0)
        self.assertIsInstance(prepared['sensor_values'], dict)
        self.assertIsInstance(prepared['detection_details'], dict)
    
    def test_prepare_alert_datetime_conversion(self):
        """Test datetime to ISO8601 string conversion."""
        alert_data = {
            'cow_id': 'COW005',
            'timestamp': datetime(2024, 1, 15, 12, 30, 0),
            'status_updated_at': datetime(2024, 1, 15, 12, 35, 0)
        }
        
        prepared = self.logger._prepare_alert(alert_data)
        
        # Check conversion to ISO8601
        self.assertIsInstance(prepared['timestamp'], str)
        self.assertIsInstance(prepared['status_updated_at'], str)
        self.assertIn('2024-01-15', prepared['timestamp'])
    
    def test_get_json_filename(self):
        """Test JSON filename generation."""
        alert = {
            'cow_id': 'COW006',
            'timestamp': '2024-01-15T12:00:00'
        }
        
        filename = self.logger._get_json_filename(alert)
        self.assertEqual(filename, 'alerts_COW006_20240115.json')
    
    def test_get_csv_filename(self):
        """Test CSV filename generation."""
        alert = {
            'cow_id': 'COW007',
            'timestamp': '2024-01-20T15:30:00'
        }
        
        filename = self.logger._get_csv_filename(alert)
        self.assertEqual(filename, 'alerts_COW007_20240120.csv')
    
    def test_flatten_alert_for_csv(self):
        """Test flattening nested dictionaries for CSV."""
        alert = {
            'alert_id': 'test',
            'sensor_values': {'temp': 40.0, 'hr': 80},
            'detection_details': {'method': 'threshold', 'confidence': 0.9}
        }
        
        flattened = self.logger._flatten_alert_for_csv(alert)
        
        # Complex types should be JSON strings
        self.assertIsInstance(flattened['sensor_values'], str)
        self.assertIsInstance(flattened['detection_details'], str)
        
        # Simple types should remain unchanged
        self.assertEqual(flattened['alert_id'], 'test')
    
    def test_read_alerts_json(self):
        """Test reading alerts from JSON logs."""
        # Log some test alerts
        for i in range(3):
            alert_data = {
                'alert_id': f'read-test-{i}',
                'cow_id': 'COW008',
                'alert_type': 'test',
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_alert(alert_data)
        
        # Read alerts
        alerts = self.logger.read_alerts_json(cow_id='COW008')
        
        self.assertEqual(len(alerts), 3)
        self.assertEqual(alerts[0]['cow_id'], 'COW008')
    
    def test_read_alerts_json_with_max_limit(self):
        """Test reading alerts with max limit."""
        # Log 5 alerts
        for i in range(5):
            alert_data = {
                'alert_id': f'limit-test-{i}',
                'cow_id': 'COW009',
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_alert(alert_data)
        
        # Read with limit
        alerts = self.logger.read_alerts_json(cow_id='COW009', max_alerts=3)
        
        self.assertEqual(len(alerts), 3)
    
    def test_cleanup_old_logs(self):
        """Test log cleanup based on retention policy."""
        # Create an old log file
        old_file = Path(self.test_dir) / 'alerts_COW010_20200101.json'
        old_file.write_text('{"test": "old"}')
        
        # Set file modification time to old date
        old_time = (datetime.now() - timedelta(days=365)).timestamp()
        old_file.touch()
        import os
        os.utime(old_file, (old_time, old_time))
        
        # Run cleanup
        deleted = self.logger.cleanup_old_logs()
        
        # Old file should be deleted
        self.assertGreater(deleted, 0)
        self.assertFalse(old_file.exists())
    
    def test_get_log_statistics(self):
        """Test log statistics retrieval."""
        # Log some alerts
        for i in range(3):
            alert_data = {
                'alert_id': f'stats-test-{i}',
                'cow_id': f'COW01{i}',
                'timestamp': datetime.now().isoformat()
            }
            self.logger.log_alert(alert_data)
        
        stats = self.logger.get_log_statistics()
        
        # Check statistics
        self.assertIn('total_json_files', stats)
        self.assertIn('total_csv_files', stats)
        self.assertIn('total_alerts', stats)
        self.assertIn('disk_usage_mb', stats)
        
        self.assertGreater(stats['total_alerts'], 0)
    
    def test_retention_days_bounds(self):
        """Test retention days is bounded between 90-180."""
        # Test lower bound
        logger1 = AlertLogger(log_dir=self.test_dir, retention_days=30)
        self.assertEqual(logger1.retention_days, 90)
        
        # Test upper bound
        logger2 = AlertLogger(log_dir=self.test_dir, retention_days=200)
        self.assertEqual(logger2.retention_days, 180)
        
        # Test valid range
        logger3 = AlertLogger(log_dir=self.test_dir, retention_days=120)
        self.assertEqual(logger3.retention_days, 120)
    
    def test_log_alert_invalid_data(self):
        """Test logging with invalid/missing data."""
        # Empty alert should still work (defaults will be added)
        success = self.logger.log_alert({})
        self.assertTrue(success)
    
    def test_concurrent_logging(self):
        """Test logging from multiple sources (same cow, same day)."""
        # Simulate concurrent alerts
        for i in range(10):
            alert_data = {
                'alert_id': f'concurrent-{i}',
                'cow_id': 'COW020',
                'alert_type': 'test_concurrent',
                'timestamp': datetime.now().isoformat()
            }
            success = self.logger.log_alert(alert_data)
            self.assertTrue(success)
        
        # Verify all alerts were logged
        alerts = self.logger.read_alerts_json(cow_id='COW020')
        self.assertEqual(len(alerts), 10)


class TestAlertLoggerIntegration(unittest.TestCase):
    """Integration tests for AlertLogger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = AlertLogger(log_dir=self.test_dir, retention_days=90)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_full_alert_lifecycle(self):
        """Test complete alert logging lifecycle."""
        # Create alert
        alert_data = {
            'alert_id': 'lifecycle-test',
            'cow_id': 'COW999',
            'alert_type': 'fever',
            'severity': 'critical',
            'confidence': 0.92,
            'sensor_values': {
                'temperature': 40.2,
                'heart_rate': 95
            },
            'detection_details': {
                'baseline_temp': 38.5,
                'threshold': 39.5,
                'duration_minutes': 30
            },
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        }
        
        # Log alert
        success = self.logger.log_alert(alert_data)
        self.assertTrue(success)
        
        # Verify JSON log
        json_files = list(Path(self.test_dir).glob("alerts_COW999_*.json"))
        self.assertEqual(len(json_files), 1)
        
        # Verify CSV log
        csv_files = list(Path(self.test_dir).glob("alerts_COW999_*.csv"))
        self.assertEqual(len(csv_files), 1)
        
        # Read back and verify
        alerts = self.logger.read_alerts_json(cow_id='COW999')
        self.assertEqual(len(alerts), 1)
        
        logged_alert = alerts[0]
        self.assertEqual(logged_alert['alert_id'], 'lifecycle-test')
        self.assertEqual(logged_alert['alert_type'], 'fever')
        self.assertEqual(logged_alert['severity'], 'critical')
        self.assertAlmostEqual(logged_alert['confidence'], 0.92, places=2)


if __name__ == '__main__':
    unittest.main()
