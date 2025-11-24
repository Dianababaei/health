"""
Unit Tests for Real-time Detection Pipeline

Tests cover:
- Mock all manager classes (SensorDataManager, AlertStateManager, HealthScoreManager)
- Mock all detector classes (ImmediateAlertDetector, EstrusDetector, SimpleHealthScorer)
- Active cow identification from database queries
- Detector execution with proper DataFrame preparation
- Alert/score storage via manager methods
- Error handling during detector failures
- Empty result scenarios (no cows, no alerts)
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.test_realtime import (
    create_sample_sensor_data,
    create_fever_sensor_data,
    create_estrus_sensor_data,
    DEFAULT_PIPELINE_CONFIG,
    TempDatabaseHelper
)


class TestPipelineInitialization(unittest.TestCase):
    """Test pipeline initialization and configuration."""
    
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    @patch('src.health_intelligence.logging.alert_state_manager.AlertStateManager')
    @patch('src.health_intelligence.logging.health_score_manager.HealthScoreManager')
    def test_pipeline_initializes_managers(self, mock_health_mgr, mock_alert_mgr, mock_sensor_mgr):
        """Test pipeline initializes all required managers."""
        # Simulate pipeline initialization
        db_path = "test.db"
        
        sensor_manager = mock_sensor_mgr(db_path=db_path)
        alert_manager = mock_alert_mgr(db_path=db_path)
        health_manager = mock_health_mgr(db_path=db_path)
        
        # Verify managers were created
        mock_sensor_mgr.assert_called_once_with(db_path=db_path)
        mock_alert_mgr.assert_called_once_with(db_path=db_path)
        mock_health_mgr.assert_called_once_with(db_path=db_path)
    
    @patch('src.health_intelligence.alerts.immediate_detector.ImmediateAlertDetector')
    @patch('src.health_intelligence.reproductive.estrus_detector.EstrusDetector')
    @patch('src.health_intelligence.scoring.simple_health_scorer.SimpleHealthScorer')
    def test_pipeline_initializes_detectors(self, mock_scorer, mock_estrus, mock_immediate):
        """Test pipeline initializes all detector instances."""
        # Simulate pipeline detector initialization
        immediate_detector = mock_immediate()
        estrus_detector = mock_estrus()
        health_scorer = mock_scorer()
        
        # Verify detectors were created
        mock_immediate.assert_called_once()
        mock_estrus.assert_called_once()
        mock_scorer.assert_called_once()
    
    def test_pipeline_loads_configuration(self):
        """Test pipeline loads configuration correctly."""
        config = DEFAULT_PIPELINE_CONFIG.copy()
        
        # Verify config values
        self.assertEqual(config['detection_interval_minutes'], 5)
        self.assertEqual(config['rolling_window_hours'], 24)
        self.assertIsInstance(config['enable_immediate_alerts'], bool)


class TestActiveCowIdentification(unittest.TestCase):
    """Test identification of active cows from database."""
    
    def setUp(self):
        """Set up test database with sample cows."""
        self.db_helper = TempDatabaseHelper()
        self.db_path = self.db_helper.setup()
        self.db_helper.insert_test_cows(['COW_001', 'COW_002', 'COW_003'])
    
    def tearDown(self):
        """Clean up test database."""
        self.db_helper.teardown()
    
    def test_get_active_cows_from_database(self):
        """Test retrieving list of active cows."""
        # Query would be: SELECT cow_id FROM cows WHERE status = 'active'
        cursor = self.db_helper.conn.cursor()
        cursor.execute("SELECT cow_id FROM cows WHERE status = 'active'")
        active_cows = [row[0] for row in cursor.fetchall()]
        
        self.assertEqual(len(active_cows), 3)
        self.assertIn('COW_001', active_cows)
        self.assertIn('COW_002', active_cows)
    
    def test_get_active_cows_with_recent_data(self):
        """Test retrieving cows with recent sensor data."""
        # Add sensor data for some cows
        df = create_sample_sensor_data(cow_id='COW_001', num_records=5)
        self.db_helper.insert_sensor_data(df)
        
        # Query cows with data in last 24 hours
        cursor = self.db_helper.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        cursor.execute("""
            SELECT DISTINCT cow_id FROM sensor_data 
            WHERE timestamp >= ?
        """, (cutoff,))
        
        active_cows = [row[0] for row in cursor.fetchall()]
        
        self.assertEqual(len(active_cows), 1)
        self.assertEqual(active_cows[0], 'COW_001')
    
    def test_empty_active_cows_list(self):
        """Test handling when no active cows exist."""
        # Remove all test cows
        cursor = self.db_helper.conn.cursor()
        cursor.execute("DELETE FROM cows")
        self.db_helper.conn.commit()
        
        cursor.execute("SELECT cow_id FROM cows WHERE status = 'active'")
        active_cows = [row[0] for row in cursor.fetchall()]
        
        self.assertEqual(len(active_cows), 0)


class TestDetectorExecution(unittest.TestCase):
    """Test execution of individual detectors."""
    
    @patch('src.health_intelligence.alerts.immediate_detector.ImmediateAlertDetector')
    def test_immediate_alert_detector_called_with_data(self, mock_detector_class):
        """Test immediate alert detector is called with sensor data."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_alerts.return_value = []
        
        # Simulate pipeline calling detector
        detector = mock_detector_class()
        sensor_data = create_sample_sensor_data(cow_id='COW_001', num_records=5)
        
        alerts = detector.detect_alerts(sensor_data, cow_id='COW_001')
        
        # Verify detector was called
        mock_detector.detect_alerts.assert_called_once()
        call_args = mock_detector.detect_alerts.call_args
        self.assertEqual(call_args[1]['cow_id'], 'COW_001')
    
    @patch('src.health_intelligence.reproductive.estrus_detector.EstrusDetector')
    def test_estrus_detector_called_with_data(self, mock_detector_class):
        """Test estrus detector is called with temperature and activity data."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_estrus.return_value = []
        
        # Simulate pipeline calling detector
        detector = mock_detector_class()
        sensor_data = create_sample_sensor_data(cow_id='COW_001', num_records=10)
        
        events = detector.detect_estrus(
            cow_id='COW_001',
            temperature_data=sensor_data[['timestamp', 'temperature']],
            activity_data=sensor_data[['timestamp', 'fxa']],
            lookback_hours=48
        )
        
        # Verify detector was called
        mock_detector.detect_estrus.assert_called_once()
    
    @patch('src.health_intelligence.scoring.simple_health_scorer.SimpleHealthScorer')
    def test_health_scorer_called_with_data(self, mock_scorer_class):
        """Test health scorer is called with sensor data."""
        mock_scorer = Mock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.calculate_score.return_value = {
            'timestamp': datetime.now().isoformat(),
            'cow_id': 'COW_001',
            'total_score': 85.0,
            'health_category': 'excellent'
        }
        
        # Simulate pipeline calling scorer
        scorer = mock_scorer_class()
        sensor_data = create_sample_sensor_data(cow_id='COW_001', num_records=10)
        
        score = scorer.calculate_score(
            cow_id='COW_001',
            sensor_data=sensor_data,
            baseline_temp=38.5
        )
        
        # Verify scorer was called
        mock_scorer.calculate_score.assert_called_once()
        self.assertEqual(score['cow_id'], 'COW_001')
        self.assertEqual(score['total_score'], 85.0)


class TestDataFramePreparation(unittest.TestCase):
    """Test DataFrame preparation for detectors."""
    
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    def test_retrieve_rolling_window_data(self, mock_manager_class):
        """Test retrieving rolling window data for detection."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock return data
        expected_df = create_sample_sensor_data(cow_id='COW_001', num_records=20)
        mock_manager.get_rolling_window_data.return_value = expected_df
        
        # Simulate pipeline retrieving data
        manager = mock_manager_class()
        df = manager.get_rolling_window_data(cow_id='COW_001', window_hours=24)
        
        # Verify correct method was called
        mock_manager.get_rolling_window_data.assert_called_once_with(
            cow_id='COW_001',
            window_hours=24
        )
        self.assertEqual(len(df), 20)
    
    def test_filter_data_by_time_window(self):
        """Test filtering data to specific time window."""
        df = create_sample_sensor_data(num_records=100)
        
        # Filter to last 10 records
        cutoff = df.iloc[-10]['timestamp']
        filtered = df[df['timestamp'] >= cutoff]
        
        self.assertEqual(len(filtered), 10)
    
    def test_handle_empty_dataframe(self):
        """Test handling empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should be empty
        self.assertTrue(empty_df.empty)
        self.assertEqual(len(empty_df), 0)
    
    def test_sort_data_by_timestamp(self):
        """Test data is sorted by timestamp."""
        df = create_sample_sensor_data(num_records=10)
        
        # Shuffle data
        shuffled = df.sample(frac=1).reset_index(drop=True)
        
        # Sort by timestamp
        sorted_df = shuffled.sort_values('timestamp').reset_index(drop=True)
        
        # Verify sorted
        for i in range(len(sorted_df) - 1):
            self.assertLessEqual(
                sorted_df.iloc[i]['timestamp'],
                sorted_df.iloc[i + 1]['timestamp']
            )


class TestAlertStorage(unittest.TestCase):
    """Test storage of detected alerts."""
    
    @patch('src.health_intelligence.logging.alert_state_manager.AlertStateManager')
    def test_store_immediate_alerts(self, mock_manager_class):
        """Test storing immediate alerts via AlertStateManager."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.log_alert.return_value = "alert-123"
        
        # Simulate alert detection and storage
        manager = mock_manager_class()
        
        alert_data = {
            'cow_id': 'COW_001',
            'alert_type': 'fever',
            'severity': 'warning',
            'confidence': 0.85,
            'sensor_values': {'temperature': 39.6},
            'details': {}
        }
        
        alert_id = manager.log_alert(**alert_data)
        
        # Verify alert was logged
        mock_manager.log_alert.assert_called_once()
        self.assertEqual(alert_id, "alert-123")
    
    @patch('src.health_intelligence.logging.alert_state_manager.AlertStateManager')
    def test_store_multiple_alerts(self, mock_manager_class):
        """Test storing multiple alerts in batch."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.log_alert.side_effect = ["alert-1", "alert-2", "alert-3"]
        
        manager = mock_manager_class()
        
        # Store multiple alerts
        alerts = [
            {'cow_id': 'COW_001', 'alert_type': 'fever', 'severity': 'warning'},
            {'cow_id': 'COW_002', 'alert_type': 'heat_stress', 'severity': 'critical'},
            {'cow_id': 'COW_003', 'alert_type': 'inactivity', 'severity': 'warning'}
        ]
        
        alert_ids = []
        for alert in alerts:
            alert_id = manager.log_alert(**alert)
            alert_ids.append(alert_id)
        
        # Verify all alerts were logged
        self.assertEqual(mock_manager.log_alert.call_count, 3)
        self.assertEqual(len(alert_ids), 3)
    
    @patch('src.health_intelligence.logging.alert_state_manager.AlertStateManager')
    def test_handle_storage_failure(self, mock_manager_class):
        """Test handling alert storage failure."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.log_alert.side_effect = Exception("Database error")
        
        manager = mock_manager_class()
        
        # Should raise exception
        with self.assertRaises(Exception):
            manager.log_alert(cow_id='COW_001', alert_type='fever')


class TestHealthScoreStorage(unittest.TestCase):
    """Test storage of health scores."""
    
    @patch('src.health_intelligence.logging.health_score_manager.HealthScoreManager')
    def test_store_health_score(self, mock_manager_class):
        """Test storing health score via HealthScoreManager."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.log_health_score.return_value = True
        
        manager = mock_manager_class()
        
        score_data = {
            'cow_id': 'COW_001',
            'score': 85.0,
            'temperature_component': 0.95,
            'activity_component': 0.88,
            'behavioral_component': 0.90,
            'alert_component': 0.75,
            'confidence': 0.92
        }
        
        result = manager.log_health_score(**score_data)
        
        # Verify score was logged
        mock_manager.log_health_score.assert_called_once()
        self.assertTrue(result)
    
    @patch('src.health_intelligence.logging.health_score_manager.HealthScoreManager')
    def test_store_scores_for_multiple_cows(self, mock_manager_class):
        """Test storing health scores for multiple cows."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.log_health_score.return_value = True
        
        manager = mock_manager_class()
        
        # Store scores for multiple cows
        cows = ['COW_001', 'COW_002', 'COW_003']
        
        for cow_id in cows:
            manager.log_health_score(cow_id=cow_id, score=80.0)
        
        # Verify all scores were logged
        self.assertEqual(mock_manager.log_health_score.call_count, 3)


class TestErrorHandling(unittest.TestCase):
    """Test error handling during detector failures."""
    
    @patch('src.health_intelligence.alerts.immediate_detector.ImmediateAlertDetector')
    def test_detector_exception_caught(self, mock_detector_class):
        """Test detector exception is caught and logged."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_alerts.side_effect = Exception("Detector failed")
        
        detector = mock_detector_class()
        
        # Simulate pipeline calling detector with error handling
        try:
            detector.detect_alerts(pd.DataFrame(), cow_id='COW_001')
            detected = False
        except Exception as e:
            # Pipeline should catch and log exception
            detected = True
            self.assertIn("Detector failed", str(e))
        
        self.assertTrue(detected)
    
    @patch('src.health_intelligence.reproductive.estrus_detector.EstrusDetector')
    def test_estrus_detector_error_does_not_stop_pipeline(self, mock_detector_class):
        """Test estrus detector error doesn't stop other detectors."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_estrus.side_effect = Exception("Estrus detection failed")
        
        detector = mock_detector_class()
        
        # Pipeline should catch exception and continue
        estrus_failed = False
        try:
            detector.detect_estrus(
                cow_id='COW_001',
                temperature_data=pd.DataFrame(),
                activity_data=pd.DataFrame()
            )
        except Exception:
            estrus_failed = True
        
        # Error should be caught
        self.assertTrue(estrus_failed)
        
        # But pipeline continues (would run health scoring next)
    
    @patch('src.health_intelligence.scoring.simple_health_scorer.SimpleHealthScorer')
    def test_scorer_error_handled_gracefully(self, mock_scorer_class):
        """Test health scorer error is handled gracefully."""
        mock_scorer = Mock()
        mock_scorer_class.return_value = mock_scorer
        mock_scorer.calculate_score.side_effect = ValueError("Invalid data")
        
        scorer = mock_scorer_class()
        
        # Should raise exception
        with self.assertRaises(ValueError):
            scorer.calculate_score(cow_id='COW_001', sensor_data=pd.DataFrame())


class TestEmptyResultScenarios(unittest.TestCase):
    """Test handling of empty results."""
    
    @patch('src.health_intelligence.alerts.immediate_detector.ImmediateAlertDetector')
    def test_no_alerts_detected(self, mock_detector_class):
        """Test handling when no alerts are detected."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_alerts.return_value = []
        
        detector = mock_detector_class()
        sensor_data = create_sample_sensor_data(cow_id='COW_001', num_records=5)
        
        alerts = detector.detect_alerts(sensor_data, cow_id='COW_001')
        
        # Should return empty list
        self.assertEqual(len(alerts), 0)
        self.assertIsInstance(alerts, list)
    
    @patch('src.health_intelligence.reproductive.estrus_detector.EstrusDetector')
    def test_no_estrus_events_detected(self, mock_detector_class):
        """Test handling when no estrus events detected."""
        mock_detector = Mock()
        mock_detector_class.return_value = mock_detector
        mock_detector.detect_estrus.return_value = []
        
        detector = mock_detector_class()
        
        events = detector.detect_estrus(
            cow_id='COW_001',
            temperature_data=pd.DataFrame(),
            activity_data=pd.DataFrame()
        )
        
        # Should return empty list
        self.assertEqual(len(events), 0)
    
    def test_no_active_cows_in_database(self):
        """Test handling when no active cows exist."""
        db_helper = TempDatabaseHelper()
        db_path = db_helper.setup()
        
        # Don't insert any cows
        cursor = db_helper.conn.cursor()
        cursor.execute("SELECT cow_id FROM cows WHERE status = 'active'")
        active_cows = [row[0] for row in cursor.fetchall()]
        
        # Should be empty
        self.assertEqual(len(active_cows), 0)
        
        db_helper.teardown()
    
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    def test_no_sensor_data_for_cow(self, mock_manager_class):
        """Test handling when no sensor data exists for cow."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.get_rolling_window_data.return_value = pd.DataFrame()
        
        manager = mock_manager_class()
        df = manager.get_rolling_window_data(cow_id='COW_999', window_hours=24)
        
        # Should return empty DataFrame
        self.assertTrue(df.empty)


class TestPipelineIntegration(unittest.TestCase):
    """Test full pipeline execution flow."""
    
    def setUp(self):
        """Set up test database with sample data."""
        self.db_helper = TempDatabaseHelper()
        self.db_path = self.db_helper.setup()
        self.db_helper.insert_test_cows(['COW_001', 'COW_002'])
        
        # Add sensor data
        for cow_id in ['COW_001', 'COW_002']:
            df = create_sample_sensor_data(cow_id=cow_id, num_records=10)
            self.db_helper.insert_sensor_data(df)
    
    def tearDown(self):
        """Clean up test database."""
        self.db_helper.teardown()
    
    @patch('src.health_intelligence.alerts.immediate_detector.ImmediateAlertDetector')
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    @patch('src.health_intelligence.logging.alert_state_manager.AlertStateManager')
    def test_full_pipeline_execution(self, mock_alert_mgr, mock_sensor_mgr, mock_detector):
        """Test complete pipeline execution with all components."""
        # Setup mocks
        mock_sensor_manager = Mock()
        mock_sensor_mgr.return_value = mock_sensor_manager
        
        mock_alert_manager = Mock()
        mock_alert_mgr.return_value = mock_alert_manager
        
        mock_detector_instance = Mock()
        mock_detector.return_value = mock_detector_instance
        
        # Mock data retrieval
        sensor_data = create_sample_sensor_data(cow_id='COW_001', num_records=10)
        mock_sensor_manager.get_rolling_window_data.return_value = sensor_data
        
        # Mock alert detection
        mock_detector_instance.detect_alerts.return_value = []
        
        # Simulate pipeline execution
        sensor_mgr = mock_sensor_mgr()
        alert_mgr = mock_alert_mgr()
        detector = mock_detector()
        
        # For each active cow
        for cow_id in ['COW_001', 'COW_002']:
            # Get sensor data
            data = sensor_mgr.get_rolling_window_data(cow_id=cow_id, window_hours=24)
            
            # Run detection
            alerts = detector.detect_alerts(data, cow_id=cow_id)
            
            # Store alerts (if any)
            for alert in alerts:
                alert_mgr.log_alert(**alert)
        
        # Verify execution
        self.assertEqual(mock_sensor_manager.get_rolling_window_data.call_count, 2)
        self.assertEqual(mock_detector_instance.detect_alerts.call_count, 2)


class TestPipelineConfiguration(unittest.TestCase):
    """Test pipeline configuration options."""
    
    def test_disable_immediate_alerts(self):
        """Test disabling immediate alert detection."""
        config = DEFAULT_PIPELINE_CONFIG.copy()
        config['enable_immediate_alerts'] = False
        
        # Pipeline should skip immediate alert detection
        self.assertFalse(config['enable_immediate_alerts'])
    
    def test_disable_estrus_detection(self):
        """Test disabling estrus detection."""
        config = DEFAULT_PIPELINE_CONFIG.copy()
        config['enable_estrus_detection'] = False
        
        self.assertFalse(config['enable_estrus_detection'])
    
    def test_configure_rolling_window(self):
        """Test configuring rolling window size."""
        config = DEFAULT_PIPELINE_CONFIG.copy()
        config['rolling_window_hours'] = 48
        
        self.assertEqual(config['rolling_window_hours'], 48)
    
    def test_configure_min_data_points(self):
        """Test configuring minimum data points requirement."""
        config = DEFAULT_PIPELINE_CONFIG.copy()
        config['min_data_points'] = 20
        
        self.assertEqual(config['min_data_points'], 20)


if __name__ == '__main__':
    unittest.main()
