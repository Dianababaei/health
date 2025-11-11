"""
Unit Tests for Reproductive Cycle Tracking System

Tests estrus cycle tracking, pregnancy linkage, and cycle prediction.
"""

import unittest
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from layer3.reproductive_cycle_tracker import (
    ReproductiveCycleTracker,
    EstrusRecord,
    PregnancyRecord,
    ReproductiveCycleState
)


class TestReproductiveCycleTracker(unittest.TestCase):
    """Test reproductive cycle tracking."""

    def setUp(self):
        """Initialize tracker."""
        self.tracker = ReproductiveCycleTracker(
            cycle_length=21.0,
            cycle_std=2.0,
            retention_days=180
        )

    def test_tracker_initialization(self):
        """Test tracker initializes with correct parameters."""
        self.assertEqual(self.tracker.cycle_length, 21.0)
        self.assertEqual(self.tracker.cycle_std, 2.0)
        self.assertEqual(self.tracker.retention_days, 180)

    def test_record_first_estrus(self):
        """Test recording first estrus event."""
        timestamp = datetime.now()
        
        record = self.tracker.record_estrus(
            cow_id='cow_001',
            event_id='estrus_001',
            timestamp=timestamp,
            confidence=0.85
        )
        
        self.assertEqual(record.cow_id, 'cow_001')
        self.assertEqual(record.event_id, 'estrus_001')
        self.assertEqual(record.cycle_day, 0)
        self.assertEqual(record.cycle_number, 1)
        self.assertFalse(record.is_predicted)
        
        # Check state was created
        state = self.tracker.get_reproductive_state('cow_001')
        self.assertIsNotNone(state)
        self.assertEqual(state.cycle_number, 1)
        self.assertEqual(state.last_estrus.event_id, 'estrus_001')

    def test_record_multiple_estrus_cycles(self):
        """Test recording multiple estrus cycles."""
        base_time = datetime.now() - timedelta(days=50)
        
        # First estrus
        estrus1 = self.tracker.record_estrus(
            cow_id='cow_002',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Second estrus (21 days later)
        estrus2 = self.tracker.record_estrus(
            cow_id='cow_002',
            event_id='estrus_002',
            timestamp=base_time + timedelta(days=21),
            confidence=0.88
        )
        
        # Third estrus (20 days later - slight variation)
        estrus3 = self.tracker.record_estrus(
            cow_id='cow_002',
            event_id='estrus_003',
            timestamp=base_time + timedelta(days=41),
            confidence=0.90
        )
        
        # Check cycle numbers
        self.assertEqual(estrus1.cycle_number, 1)
        self.assertEqual(estrus2.cycle_number, 2)
        self.assertEqual(estrus3.cycle_number, 3)
        
        # Check state
        state = self.tracker.get_reproductive_state('cow_002')
        self.assertEqual(state.cycle_number, 3)
        self.assertEqual(len(state.estrus_history), 3)
        
        # Check cycle length was updated
        # Should be around 20.5 days ((21 + 20) / 2)
        self.assertGreater(state.cycle_length_mean, 20.0)
        self.assertLess(state.cycle_length_mean, 22.0)

    def test_estrus_cycle_prediction(self):
        """Test prediction of next estrus based on cycle history."""
        base_time = datetime.now() - timedelta(days=25)
        
        # Record estrus
        self.tracker.record_estrus(
            cow_id='cow_003',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Get prediction
        predicted = self.tracker.predict_next_estrus('cow_003')
        
        self.assertIsNotNone(predicted)
        
        # Should be approximately 21 days after last estrus
        expected = base_time + timedelta(days=21)
        delta = abs((predicted - expected).days)
        self.assertLess(delta, 1)  # Within 1 day

    def test_cycle_prediction_with_irregular_cycles(self):
        """Test prediction adapts to individual cow's cycle length."""
        base_time = datetime.now() - timedelta(days=70)
        
        # Cow with 19-day cycles
        self.tracker.record_estrus(
            cow_id='cow_004',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        self.tracker.record_estrus(
            cow_id='cow_004',
            event_id='estrus_002',
            timestamp=base_time + timedelta(days=19),
            confidence=0.87
        )
        
        self.tracker.record_estrus(
            cow_id='cow_004',
            event_id='estrus_003',
            timestamp=base_time + timedelta(days=38),
            confidence=0.86
        )
        
        # Get state
        state = self.tracker.get_reproductive_state('cow_004')
        
        # Cycle length should be around 19 days
        self.assertGreater(state.cycle_length_mean, 18.5)
        self.assertLess(state.cycle_length_mean, 19.5)
        
        # Prediction should use 19-day cycle
        predicted = self.tracker.predict_next_estrus('cow_004')
        self.assertIsNotNone(predicted)
        
        last_estrus_time = base_time + timedelta(days=38)
        expected = last_estrus_time + timedelta(days=19)
        delta = abs((predicted - expected).days)
        self.assertLess(delta, 2)

    def test_pregnancy_recording(self):
        """Test recording pregnancy event."""
        base_time = datetime.now() - timedelta(days=15)
        
        # Record estrus first
        self.tracker.record_estrus(
            cow_id='cow_005',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Record pregnancy
        pregnancy_time = base_time + timedelta(days=15)
        pregnancy = self.tracker.record_pregnancy(
            cow_id='cow_005',
            event_id='pregnancy_001',
            timestamp=pregnancy_time,
            confidence=0.75,
            linked_estrus_id='estrus_001'
        )
        
        self.assertEqual(pregnancy.cow_id, 'cow_005')
        self.assertEqual(pregnancy.linked_estrus_id, 'estrus_001')
        self.assertEqual(pregnancy.days_pregnant, 15)
        self.assertFalse(pregnancy.is_confirmed)  # Not yet 30 days
        
        # Check state
        state = self.tracker.get_reproductive_state('cow_005')
        self.assertTrue(state.is_pregnant)
        self.assertIsNotNone(state.pregnancy_record)

    def test_pregnancy_confirmation(self):
        """Test pregnancy confirmation after 30 days."""
        base_time = datetime.now() - timedelta(days=35)
        
        # Record estrus
        self.tracker.record_estrus(
            cow_id='cow_006',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Record pregnancy after 35 days
        pregnancy_time = base_time + timedelta(days=35)
        pregnancy = self.tracker.record_pregnancy(
            cow_id='cow_006',
            event_id='pregnancy_001',
            timestamp=pregnancy_time,
            confidence=0.85,
            linked_estrus_id='estrus_001'
        )
        
        # Should be confirmed
        self.assertTrue(pregnancy.is_confirmed)
        self.assertEqual(pregnancy.days_pregnant, 35)

    def test_pregnancy_prevents_estrus_prediction(self):
        """Test that pregnant cows don't get estrus predictions."""
        base_time = datetime.now() - timedelta(days=30)
        
        # Record estrus and pregnancy
        self.tracker.record_estrus(
            cow_id='cow_007',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        self.tracker.record_pregnancy(
            cow_id='cow_007',
            event_id='pregnancy_001',
            timestamp=datetime.now(),
            confidence=0.85,
            linked_estrus_id='estrus_001'
        )
        
        # Prediction should be None for pregnant cow
        predicted = self.tracker.predict_next_estrus('cow_007')
        self.assertIsNone(predicted)

    def test_estrus_pregnancy_linkage(self):
        """Test linking pregnancy to estrus event."""
        base_time = datetime.now() - timedelta(days=20)
        
        # Record estrus
        self.tracker.record_estrus(
            cow_id='cow_008',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Record pregnancy linked to estrus
        self.tracker.record_pregnancy(
            cow_id='cow_008',
            event_id='pregnancy_001',
            timestamp=datetime.now(),
            confidence=0.80,
            linked_estrus_id='estrus_001'
        )
        
        # Get linked events
        estrus, pregnancies = self.tracker.get_linked_events('cow_008', 'estrus_001')
        
        self.assertIsNotNone(estrus)
        self.assertEqual(estrus.event_id, 'estrus_001')
        self.assertEqual(len(pregnancies), 1)
        self.assertEqual(pregnancies[0].event_id, 'pregnancy_001')
        self.assertEqual(pregnancies[0].linked_estrus_id, 'estrus_001')

    def test_get_estrus_history(self):
        """Test retrieving estrus history."""
        base_time = datetime.now() - timedelta(days=60)
        
        # Record multiple estrus events
        for i in range(3):
            self.tracker.record_estrus(
                cow_id='cow_009',
                event_id=f'estrus_{i+1:03d}',
                timestamp=base_time + timedelta(days=i*21),
                confidence=0.85
            )
        
        # Get all history
        history = self.tracker.get_estrus_history('cow_009')
        self.assertEqual(len(history), 3)
        
        # Get recent history (30 days)
        recent_history = self.tracker.get_estrus_history('cow_009', days=30)
        self.assertLessEqual(len(recent_history), 2)

    def test_get_pregnancy_history(self):
        """Test retrieving pregnancy history."""
        base_time = datetime.now() - timedelta(days=40)
        
        # Record estrus and pregnancy
        self.tracker.record_estrus(
            cow_id='cow_010',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        self.tracker.record_pregnancy(
            cow_id='cow_010',
            event_id='pregnancy_001',
            timestamp=datetime.now(),
            confidence=0.85,
            linked_estrus_id='estrus_001'
        )
        
        # Get history
        history = self.tracker.get_pregnancy_history('cow_010')
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].event_id, 'pregnancy_001')

    def test_cycle_day_tracking(self):
        """Test tracking current cycle day."""
        base_time = datetime.now() - timedelta(days=10)
        
        # Record estrus
        self.tracker.record_estrus(
            cow_id='cow_011',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Update cycle day
        current_time = datetime.now()
        self.tracker.update_cycle_day('cow_011', current_time)
        
        # Check cycle day
        state = self.tracker.get_reproductive_state('cow_011')
        self.assertEqual(state.current_cycle_day, 10)

    def test_cycle_statistics(self):
        """Test cycle statistics calculation."""
        base_time = datetime.now() - timedelta(days=80)
        
        # Record multiple cycles
        cycle_days = [0, 21, 42, 63]
        for i, days in enumerate(cycle_days):
            self.tracker.record_estrus(
                cow_id='cow_012',
                event_id=f'estrus_{i+1:03d}',
                timestamp=base_time + timedelta(days=days),
                confidence=0.85
            )
        
        # Get statistics
        stats = self.tracker.get_cycle_statistics('cow_012')
        
        self.assertTrue(stats['tracked'])
        self.assertEqual(stats['cycle_count'], 4)
        self.assertIsNotNone(stats['cycle_length_mean'])
        self.assertEqual(stats['intervals'], [21, 21, 21])
        self.assertEqual(stats['interval_min'], 21)
        self.assertEqual(stats['interval_max'], 21)

    def test_irregular_cycle_detection(self):
        """Test detection of irregular cycle intervals."""
        base_time = datetime.now() - timedelta(days=50)
        
        # Record regular cycle
        self.tracker.record_estrus(
            cow_id='cow_013',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Record irregular cycle (35 days - outside normal range)
        self.tracker.record_estrus(
            cow_id='cow_013',
            event_id='estrus_002',
            timestamp=base_time + timedelta(days=35),
            confidence=0.80
        )
        
        # State should still be updated, but cycle length may not be
        state = self.tracker.get_reproductive_state('cow_013')
        self.assertEqual(state.cycle_number, 2)
        # Irregular cycle should log warning but not break tracking

    def test_cleanup_old_records(self):
        """Test cleanup of old records beyond retention period."""
        old_time = datetime.now() - timedelta(days=200)
        recent_time = datetime.now() - timedelta(days=50)
        
        # Record old estrus
        self.tracker.record_estrus(
            cow_id='cow_014',
            event_id='estrus_old',
            timestamp=old_time,
            confidence=0.85
        )
        
        # Record recent estrus
        self.tracker.record_estrus(
            cow_id='cow_014',
            event_id='estrus_recent',
            timestamp=recent_time,
            confidence=0.85
        )
        
        # Cleanup
        cutoff = datetime.now() - timedelta(days=180)
        self.tracker.cleanup_old_records(cutoff)
        
        # Check that old record was removed
        history = self.tracker.get_estrus_history('cow_014')
        event_ids = [e.event_id for e in history]
        
        self.assertNotIn('estrus_old', event_ids)
        self.assertIn('estrus_recent', event_ids)

    def test_untracked_cow_handling(self):
        """Test handling of untracked cows."""
        # Get state for untracked cow
        state = self.tracker.get_reproductive_state('cow_999')
        self.assertIsNone(state)
        
        # Get prediction for untracked cow
        predicted = self.tracker.predict_next_estrus('cow_999')
        self.assertIsNone(predicted)
        
        # Get statistics for untracked cow
        stats = self.tracker.get_cycle_statistics('cow_999')
        self.assertFalse(stats['tracked'])
        self.assertEqual(stats['cycle_count'], 0)

    def test_pregnancy_auto_linked_to_last_estrus(self):
        """Test pregnancy auto-links to last estrus if no ID provided."""
        base_time = datetime.now() - timedelta(days=20)
        
        # Record estrus
        self.tracker.record_estrus(
            cow_id='cow_015',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        # Record pregnancy without explicit link
        pregnancy = self.tracker.record_pregnancy(
            cow_id='cow_015',
            event_id='pregnancy_001',
            timestamp=datetime.now(),
            confidence=0.80,
            linked_estrus_id=None  # No explicit link
        )
        
        # Should auto-link to last estrus
        self.assertEqual(pregnancy.linked_estrus_id, 'estrus_001')
        self.assertIsNotNone(pregnancy.conception_date)

    def test_multiple_cows_tracking(self):
        """Test tracking multiple cows independently."""
        base_time = datetime.now() - timedelta(days=30)
        
        # Track cow 1 with 21-day cycle
        self.tracker.record_estrus(
            cow_id='cow_016',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        self.tracker.record_estrus(
            cow_id='cow_016',
            event_id='estrus_002',
            timestamp=base_time + timedelta(days=21),
            confidence=0.85
        )
        
        # Track cow 2 with 19-day cycle
        self.tracker.record_estrus(
            cow_id='cow_017',
            event_id='estrus_001',
            timestamp=base_time,
            confidence=0.85
        )
        
        self.tracker.record_estrus(
            cow_id='cow_017',
            event_id='estrus_002',
            timestamp=base_time + timedelta(days=19),
            confidence=0.85
        )
        
        # Check independent tracking
        state1 = self.tracker.get_reproductive_state('cow_016')
        state2 = self.tracker.get_reproductive_state('cow_017')
        
        self.assertAlmostEqual(state1.cycle_length_mean, 21.0, delta=0.5)
        self.assertAlmostEqual(state2.cycle_length_mean, 19.0, delta=0.5)


if __name__ == '__main__':
    unittest.main()
