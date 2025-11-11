"""
Unit Tests for Alert State Management System

Tests for AlertStateManager class including state transitions,
database operations, and query capabilities.
"""

import unittest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_intelligence.logging.alert_state_manager import (
    AlertStateManager,
    AlertStatus
)


class TestAlertStateManager(unittest.TestCase):
    """Test cases for AlertStateManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.manager = AlertStateManager(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary database
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_initialization(self):
        """Test manager initialization."""
        self.assertTrue(Path(self.temp_db.name).exists())
        self.assertIsInstance(self.manager, AlertStateManager)
    
    def test_create_alert(self):
        """Test creating a new alert."""
        alert_data = {
            'alert_id': 'test-001',
            'cow_id': 'COW001',
            'alert_type': 'fever',
            'severity': 'critical',
            'confidence': 0.95,
            'sensor_values': {'temperature': 40.5},
            'detection_details': {'threshold': 39.5},
            'timestamp': datetime.now().isoformat()
        }
        
        success = self.manager.create_alert(alert_data)
        self.assertTrue(success)
        
        # Verify alert was created
        alert = self.manager.get_alert('test-001')
        self.assertIsNotNone(alert)
        self.assertEqual(alert['alert_id'], 'test-001')
        self.assertEqual(alert['cow_id'], 'COW001')
        self.assertEqual(alert['status'], 'active')
    
    def test_create_duplicate_alert(self):
        """Test creating duplicate alert fails."""
        alert_data = {
            'alert_id': 'test-002',
            'cow_id': 'COW002',
            'alert_type': 'test',
            'timestamp': datetime.now().isoformat()
        }
        
        # First creation should succeed
        success1 = self.manager.create_alert(alert_data)
        self.assertTrue(success1)
        
        # Second creation should fail
        success2 = self.manager.create_alert(alert_data)
        self.assertFalse(success2)
    
    def test_get_alert(self):
        """Test retrieving an alert by ID."""
        alert_data = {
            'alert_id': 'test-003',
            'cow_id': 'COW003',
            'alert_type': 'heat_stress',
            'severity': 'high',
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat()
        }
        
        self.manager.create_alert(alert_data)
        alert = self.manager.get_alert('test-003')
        
        self.assertIsNotNone(alert)
        self.assertEqual(alert['alert_type'], 'heat_stress')
        self.assertEqual(alert['severity'], 'high')
    
    def test_get_nonexistent_alert(self):
        """Test retrieving non-existent alert returns None."""
        alert = self.manager.get_alert('nonexistent-id')
        self.assertIsNone(alert)
    
    def test_update_status_active_to_acknowledged(self):
        """Test updating status from active to acknowledged."""
        # Create active alert
        alert_data = {
            'alert_id': 'test-004',
            'cow_id': 'COW004',
            'alert_type': 'test',
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        # Update to acknowledged
        success = self.manager.update_status('test-004', 'acknowledged', 'Reviewed by vet')
        self.assertTrue(success)
        
        # Verify status changed
        alert = self.manager.get_alert('test-004')
        self.assertEqual(alert['status'], 'acknowledged')
        self.assertEqual(alert['resolution_notes'], 'Reviewed by vet')
    
    def test_update_status_active_to_resolved(self):
        """Test updating status from active to resolved."""
        alert_data = {
            'alert_id': 'test-005',
            'cow_id': 'COW005',
            'alert_type': 'test',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        success = self.manager.update_status('test-005', 'resolved', 'Issue fixed')
        self.assertTrue(success)
        
        alert = self.manager.get_alert('test-005')
        self.assertEqual(alert['status'], 'resolved')
    
    def test_update_status_acknowledged_to_resolved(self):
        """Test updating status from acknowledged to resolved."""
        alert_data = {
            'alert_id': 'test-006',
            'cow_id': 'COW006',
            'alert_type': 'test',
            'status': 'acknowledged',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        success = self.manager.update_status('test-006', 'resolved')
        self.assertTrue(success)
        
        alert = self.manager.get_alert('test-006')
        self.assertEqual(alert['status'], 'resolved')
    
    def test_invalid_state_transition(self):
        """Test invalid state transition is rejected."""
        # Create resolved alert
        alert_data = {
            'alert_id': 'test-007',
            'cow_id': 'COW007',
            'alert_type': 'test',
            'status': 'resolved',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        # Try to transition from resolved to active (invalid)
        success = self.manager.update_status('test-007', 'active')
        self.assertFalse(success)
        
        # Status should remain resolved
        alert = self.manager.get_alert('test-007')
        self.assertEqual(alert['status'], 'resolved')
    
    def test_acknowledge_alert(self):
        """Test acknowledge_alert convenience method."""
        alert_data = {
            'alert_id': 'test-008',
            'cow_id': 'COW008',
            'alert_type': 'test',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        success = self.manager.acknowledge_alert('test-008', 'Acknowledged by operator')
        self.assertTrue(success)
        
        alert = self.manager.get_alert('test-008')
        self.assertEqual(alert['status'], 'acknowledged')
    
    def test_resolve_alert(self):
        """Test resolve_alert convenience method."""
        alert_data = {
            'alert_id': 'test-009',
            'cow_id': 'COW009',
            'alert_type': 'test',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        success = self.manager.resolve_alert('test-009', 'Resolved successfully')
        self.assertTrue(success)
        
        alert = self.manager.get_alert('test-009')
        self.assertEqual(alert['status'], 'resolved')
    
    def test_mark_false_positive(self):
        """Test mark_false_positive convenience method."""
        alert_data = {
            'alert_id': 'test-010',
            'cow_id': 'COW010',
            'alert_type': 'test',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        success = self.manager.mark_false_positive('test-010', 'Sensor malfunction')
        self.assertTrue(success)
        
        alert = self.manager.get_alert('test-010')
        self.assertEqual(alert['status'], 'false_positive')
    
    def test_query_alerts_by_cow_id(self):
        """Test querying alerts by cow ID."""
        # Create alerts for different cows
        for i in range(3):
            alert_data = {
                'alert_id': f'test-cow-query-{i}',
                'cow_id': 'COW100',
                'alert_type': 'test',
                'timestamp': datetime.now().isoformat()
            }
            self.manager.create_alert(alert_data)
        
        # Create alert for different cow
        self.manager.create_alert({
            'alert_id': 'test-cow-query-other',
            'cow_id': 'COW200',
            'alert_type': 'test',
            'timestamp': datetime.now().isoformat()
        })
        
        # Query for COW100
        alerts = self.manager.query_alerts(cow_id='COW100')
        self.assertEqual(len(alerts), 3)
        self.assertTrue(all(a['cow_id'] == 'COW100' for a in alerts))
    
    def test_query_alerts_by_status(self):
        """Test querying alerts by status."""
        # Create alerts with different statuses
        self.manager.create_alert({
            'alert_id': 'test-status-1',
            'cow_id': 'COW101',
            'alert_type': 'test',
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        })
        
        self.manager.create_alert({
            'alert_id': 'test-status-2',
            'cow_id': 'COW102',
            'alert_type': 'test',
            'status': 'resolved',
            'timestamp': datetime.now().isoformat()
        })
        
        # Query for active alerts
        active_alerts = self.manager.query_alerts(status='active')
        self.assertTrue(all(a['status'] == 'active' for a in active_alerts))
    
    def test_query_alerts_by_severity(self):
        """Test querying alerts by severity."""
        # Create alerts with different severities
        for severity in ['critical', 'high', 'warning']:
            self.manager.create_alert({
                'alert_id': f'test-severity-{severity}',
                'cow_id': 'COW103',
                'alert_type': 'test',
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            })
        
        # Query for critical alerts
        critical_alerts = self.manager.query_alerts(severity='critical')
        self.assertTrue(all(a['severity'] == 'critical' for a in critical_alerts))
    
    def test_query_alerts_by_alert_type(self):
        """Test querying alerts by alert type."""
        # Create alerts of different types
        for alert_type in ['fever', 'heat_stress']:
            self.manager.create_alert({
                'alert_id': f'test-type-{alert_type}',
                'cow_id': 'COW104',
                'alert_type': alert_type,
                'timestamp': datetime.now().isoformat()
            })
        
        # Query for fever alerts
        fever_alerts = self.manager.query_alerts(alert_type='fever')
        self.assertTrue(all(a['alert_type'] == 'fever' for a in fever_alerts))
    
    def test_query_alerts_with_limit(self):
        """Test querying alerts with result limit."""
        # Create 10 alerts
        for i in range(10):
            self.manager.create_alert({
                'alert_id': f'test-limit-{i}',
                'cow_id': 'COW105',
                'alert_type': 'test',
                'timestamp': datetime.now().isoformat()
            })
        
        # Query with limit
        alerts = self.manager.query_alerts(cow_id='COW105', limit=5)
        self.assertEqual(len(alerts), 5)
    
    def test_query_alerts_with_date_range(self):
        """Test querying alerts within date range."""
        # Create alerts with different dates
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        self.manager.create_alert({
            'alert_id': 'test-date-1',
            'cow_id': 'COW106',
            'alert_type': 'test',
            'timestamp': yesterday.isoformat()
        })
        
        self.manager.create_alert({
            'alert_id': 'test-date-2',
            'cow_id': 'COW106',
            'alert_type': 'test',
            'timestamp': today.isoformat()
        })
        
        # Query for today's alerts
        start_of_today = today.replace(hour=0, minute=0, second=0, microsecond=0)
        alerts = self.manager.query_alerts(
            cow_id='COW106',
            start_date=start_of_today.isoformat()
        )
        
        self.assertGreaterEqual(len(alerts), 1)
    
    def test_get_state_history(self):
        """Test retrieving state change history."""
        # Create alert and update its status
        alert_data = {
            'alert_id': 'test-history',
            'cow_id': 'COW107',
            'alert_type': 'test',
            'timestamp': datetime.now().isoformat()
        }
        self.manager.create_alert(alert_data)
        
        # Update status multiple times
        self.manager.acknowledge_alert('test-history')
        self.manager.resolve_alert('test-history')
        
        # Get history
        history = self.manager.get_state_history('test-history')
        
        # Should have 3 entries (created, acknowledged, resolved)
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['new_status'], 'active')
        self.assertEqual(history[1]['new_status'], 'acknowledged')
        self.assertEqual(history[2]['new_status'], 'resolved')
    
    def test_get_statistics(self):
        """Test getting alert statistics."""
        # Create various alerts
        for i in range(5):
            status = 'active' if i < 2 else 'resolved'
            severity = 'critical' if i < 2 else 'warning'
            
            self.manager.create_alert({
                'alert_id': f'test-stats-{i}',
                'cow_id': f'COW20{i}',
                'alert_type': 'test',
                'status': status,
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            })
        
        stats = self.manager.get_statistics()
        
        self.assertIn('total_alerts', stats)
        self.assertIn('by_status', stats)
        self.assertIn('by_severity', stats)
        self.assertIn('by_type', stats)
        
        self.assertGreaterEqual(stats['total_alerts'], 5)
    
    def test_valid_transitions(self):
        """Test all valid state transitions."""
        transitions = [
            ('active', 'acknowledged'),
            ('active', 'resolved'),
            ('active', 'false_positive'),
            ('acknowledged', 'resolved'),
            ('acknowledged', 'false_positive'),
        ]
        
        for i, (from_status, to_status) in enumerate(transitions):
            alert_id = f'test-transition-{i}'
            self.manager.create_alert({
                'alert_id': alert_id,
                'cow_id': 'COW300',
                'alert_type': 'test',
                'status': from_status,
                'timestamp': datetime.now().isoformat()
            })
            
            success = self.manager.update_status(alert_id, to_status)
            self.assertTrue(success, f"Transition {from_status} -> {to_status} should be valid")
    
    def test_invalid_transitions_from_terminal_states(self):
        """Test that terminal states cannot transition."""
        terminal_states = ['resolved', 'false_positive']
        
        for i, status in enumerate(terminal_states):
            alert_id = f'test-terminal-{i}'
            self.manager.create_alert({
                'alert_id': alert_id,
                'cow_id': 'COW400',
                'alert_type': 'test',
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
            
            # Try to transition to any other state
            success = self.manager.update_status(alert_id, 'active')
            self.assertFalse(success, f"Terminal state {status} should not allow transitions")
    
    def test_json_field_parsing(self):
        """Test that JSON fields are properly parsed when retrieved."""
        alert_data = {
            'alert_id': 'test-json',
            'cow_id': 'COW500',
            'alert_type': 'test',
            'sensor_values': {'temp': 40.0, 'hr': 85},
            'detection_details': {'method': 'threshold', 'confidence': 0.9},
            'timestamp': datetime.now().isoformat()
        }
        
        self.manager.create_alert(alert_data)
        alert = self.manager.get_alert('test-json')
        
        # Check that complex fields are dictionaries, not JSON strings
        self.assertIsInstance(alert['sensor_values'], dict)
        self.assertIsInstance(alert['detection_details'], dict)
        self.assertEqual(alert['sensor_values']['temp'], 40.0)


class TestAlertStateManagerIntegration(unittest.TestCase):
    """Integration tests for AlertStateManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.manager = AlertStateManager(db_path=self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_full_alert_lifecycle(self):
        """Test complete alert lifecycle from creation to resolution."""
        # Create alert
        alert_data = {
            'alert_id': 'lifecycle-test',
            'cow_id': 'COW999',
            'alert_type': 'fever',
            'severity': 'critical',
            'confidence': 0.95,
            'sensor_values': {'temperature': 40.5},
            'detection_details': {'baseline': 38.5, 'threshold': 39.5},
            'timestamp': datetime.now().isoformat()
        }
        
        # Step 1: Create
        self.assertTrue(self.manager.create_alert(alert_data))
        alert = self.manager.get_alert('lifecycle-test')
        self.assertEqual(alert['status'], 'active')
        
        # Step 2: Acknowledge
        self.assertTrue(self.manager.acknowledge_alert('lifecycle-test', 'Vet notified'))
        alert = self.manager.get_alert('lifecycle-test')
        self.assertEqual(alert['status'], 'acknowledged')
        
        # Step 3: Resolve
        self.assertTrue(self.manager.resolve_alert('lifecycle-test', 'Treatment administered'))
        alert = self.manager.get_alert('lifecycle-test')
        self.assertEqual(alert['status'], 'resolved')
        
        # Verify history
        history = self.manager.get_state_history('lifecycle-test')
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['new_status'], 'active')
        self.assertEqual(history[1]['new_status'], 'acknowledged')
        self.assertEqual(history[2]['new_status'], 'resolved')
    
    def test_concurrent_alerts_multiple_cows(self):
        """Test handling multiple concurrent alerts for different cows."""
        # Create alerts for multiple cows
        for cow_num in range(10):
            for alert_num in range(3):
                alert_data = {
                    'alert_id': f'concurrent-cow{cow_num}-alert{alert_num}',
                    'cow_id': f'COW{cow_num:03d}',
                    'alert_type': 'test',
                    'severity': 'warning',
                    'timestamp': datetime.now().isoformat()
                }
                self.manager.create_alert(alert_data)
        
        # Query all alerts
        all_alerts = self.manager.query_alerts(limit=100)
        self.assertEqual(len(all_alerts), 30)
        
        # Query by specific cow
        cow_alerts = self.manager.query_alerts(cow_id='COW005')
        self.assertEqual(len(cow_alerts), 3)


if __name__ == '__main__':
    unittest.main()
