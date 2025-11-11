"""
Unit Tests for Health Score Components

Tests cover:
- Base component interface
- Temperature stability scoring
- Activity level scoring
- Behavioral patterns scoring
- Alert frequency scoring
- Component score validation
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

from src.health_intelligence.scoring.components import (
    BaseScoreComponent,
    ComponentScore,
    TemperatureScoreComponent,
    ActivityScoreComponent,
    BehavioralScoreComponent,
    AlertScoreComponent,
)


class TestComponentScore(unittest.TestCase):
    """Test ComponentScore data structure."""
    
    def test_create_valid_score(self):
        """Test creating a valid ComponentScore."""
        score = ComponentScore(
            score=20.0,
            normalized_score=0.8,
            confidence=0.9,
            details={'test': 'data'}
        )
        
        self.assertEqual(score.score, 20.0)
        self.assertEqual(score.normalized_score, 0.8)
        self.assertEqual(score.confidence, 0.9)
        self.assertEqual(score.details, {'test': 'data'})
        self.assertEqual(score.warnings, [])
    
    def test_score_clamping(self):
        """Test that out-of-range scores are clamped."""
        score = ComponentScore(
            score=30.0,  # Above max of 25
            normalized_score=1.5,  # Above max of 1.0
            confidence=1.2,  # Above max of 1.0
            details={}
        )
        
        self.assertEqual(score.score, 25.0)
        self.assertEqual(score.normalized_score, 1.0)
        self.assertEqual(score.confidence, 1.0)
        self.assertGreater(len(score.warnings), 0)


class TestTemperatureScoreComponent(unittest.TestCase):
    """Test temperature stability scoring component."""
    
    def setUp(self):
        """Set up test component."""
        self.component = TemperatureScoreComponent()
    
    def test_perfect_temperature_score(self):
        """Test score with perfect temperature stability."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100  # Perfect stability
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_temp=38.5
        )
        
        self.assertGreater(score.score, 23)  # Should be near perfect
        self.assertGreater(score.confidence, 0.7)
        self.assertEqual(len(score.warnings), 0)
    
    def test_temperature_deviation_penalty(self):
        """Test score penalty with temperature deviation."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [39.5] * 100  # 1°C above baseline
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_temp=38.5
        )
        
        # Should have deviation penalty
        self.assertLess(score.score, 25)
        self.assertIn('deviation_penalty', score.details)
        self.assertGreater(score.details['deviation_penalty'], 0)
    
    def test_fever_incident_penalty(self):
        """Test score penalty with fever incidents."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100
        })
        
        fever_events = [
            {'timestamp': datetime.now(), 'severity': 'warning'},
            {'timestamp': datetime.now(), 'severity': 'critical'}
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_temp=38.5,
            fever_events=fever_events
        )
        
        # Should have fever penalty
        self.assertLess(score.score, 20)
        self.assertEqual(score.details['fever_count'], 2)
        self.assertGreater(score.details['fever_penalty'], 0)
    
    def test_circadian_bonus(self):
        """Test circadian rhythm bonus."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_temp=38.5,
            circadian_score=0.9  # Good circadian rhythm
        )
        
        # Should have circadian bonus
        self.assertIn('circadian_bonus', score.details)
        self.assertGreater(score.details['circadian_bonus'], 0)
    
    def test_no_data_handling(self):
        """Test handling of missing data."""
        data = pd.DataFrame()
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data
        )
        
        self.assertEqual(score.score, 0.0)
        self.assertEqual(score.confidence, 0.0)
        self.assertGreater(len(score.warnings), 0)


class TestActivityScoreComponent(unittest.TestCase):
    """Test activity level scoring component."""
    
    def setUp(self):
        """Set up test component."""
        self.component = ActivityScoreComponent()
    
    def test_perfect_activity_score(self):
        """Test score with optimal activity level."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'movement_intensity': [0.3] * 100  # Stable activity
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_activity=0.3
        )
        
        self.assertGreater(score.score, 23)  # Should be near perfect
        self.assertGreater(score.confidence, 0.7)
    
    def test_activity_deviation_penalty(self):
        """Test score penalty with activity deviation."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'movement_intensity': [0.1] * 100  # Low activity
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_activity=0.5  # Much higher baseline
        )
        
        # Should have deviation penalty
        self.assertLess(score.score, 25)
        self.assertGreater(score.details['deviation_penalty'], 0)
    
    def test_inactivity_incident_penalty(self):
        """Test score penalty with inactivity incidents."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'movement_intensity': [0.3] * 100
        })
        
        inactivity_events = [
            {'duration_hours': 5, 'timestamp': datetime.now()},
            {'duration_hours': 6, 'timestamp': datetime.now()}
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_activity=0.3,
            inactivity_events=inactivity_events
        )
        
        # Should have inactivity penalty
        self.assertLess(score.score, 20)
        self.assertEqual(score.details['inactivity_count'], 2)
    
    def test_activity_duration_bonus(self):
        """Test bonus for adequate activity duration."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'movement_intensity': [0.3] * 100
        })
        
        # Create behavioral states with adequate active time
        behavioral_states = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=480, freq='1min'),  # 8 hours
            'behavioral_state': ['standing'] * 480
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_activity=0.3,
            behavioral_states=behavioral_states
        )
        
        # Should have duration bonus
        self.assertIn('active_hours', score.details)
        self.assertGreaterEqual(score.details['active_hours'], 8.0)


class TestBehavioralScoreComponent(unittest.TestCase):
    """Test behavioral patterns scoring component."""
    
    def setUp(self):
        """Set up test component."""
        self.component = BehavioralScoreComponent()
    
    def test_optimal_rumination_score(self):
        """Test score with optimal rumination time."""
        # 500 minutes of rumination (within optimal range)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'behavioral_state': ['ruminating'] * 500 + ['standing'] * 940
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data
        )
        
        self.assertGreater(score.score, 20)
        self.assertGreater(score.confidence, 0.7)
        self.assertLessEqual(score.details['rumination_penalty'], 0)
    
    def test_rumination_deficit_penalty(self):
        """Test score penalty with insufficient rumination."""
        # Only 200 minutes of rumination (below optimal)
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'behavioral_state': ['ruminating'] * 200 + ['standing'] * 1240
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data
        )
        
        # Should have rumination penalty
        self.assertLess(score.score, 25)
        self.assertGreater(score.details['rumination_penalty'], 0)
    
    def test_stress_behavior_penalty(self):
        """Test score penalty with stress behaviors."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'behavioral_state': ['ruminating'] * 500 + ['standing'] * 940
        })
        
        stress_events = [
            {'type': 'erratic_motion', 'timestamp': datetime.now()},
            {'type': 'excessive_transitions', 'timestamp': datetime.now()}
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data,
            stress_events=stress_events
        )
        
        # Should have stress penalty
        self.assertLess(score.score, 23)
        self.assertEqual(score.details['stress_behavior_count'], 2)
    
    def test_behavioral_diversity(self):
        """Test behavioral diversity calculation."""
        # Diverse behavioral pattern
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
            'behavioral_state': (
                ['lying'] * 400 +
                ['standing'] * 300 +
                ['walking'] * 100 +
                ['ruminating'] * 150 +
                ['feeding'] * 50
            )
        })
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=data
        )
        
        # Should have good diversity
        self.assertIn('behavioral_diversity', score.details)
        self.assertGreater(score.details['behavioral_diversity'], 0.5)


class TestAlertScoreComponent(unittest.TestCase):
    """Test alert frequency scoring component."""
    
    def setUp(self):
        """Set up test component."""
        self.component = AlertScoreComponent()
    
    def test_no_alerts_perfect_score(self):
        """Test score with no active alerts."""
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=pd.DataFrame(),
            active_alerts=[]
        )
        
        self.assertEqual(score.score, 25.0)
        self.assertEqual(score.details['total_active_alerts'], 0)
    
    def test_critical_alert_penalty(self):
        """Test score penalty with critical alerts."""
        active_alerts = [
            {'severity': 'critical', 'type': 'fever'},
            {'severity': 'critical', 'type': 'sensor_malfunction'}
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=pd.DataFrame(),
            active_alerts=active_alerts
        )
        
        # Should have significant penalty
        self.assertLess(score.score, 10)
        self.assertEqual(score.details['critical_alerts'], 2)
        self.assertGreater(score.details['critical_penalty'], 0)
    
    def test_warning_alert_penalty(self):
        """Test score penalty with warning alerts."""
        active_alerts = [
            {'severity': 'warning', 'type': 'heat_stress'},
            {'severity': 'warning', 'type': 'inactivity'}
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=pd.DataFrame(),
            active_alerts=active_alerts
        )
        
        # Should have moderate penalty
        self.assertLess(score.score, 20)
        self.assertEqual(score.details['warning_alerts'], 2)
    
    def test_mixed_severity_alerts(self):
        """Test score with mixed severity alerts."""
        active_alerts = [
            {'severity': 'critical', 'type': 'fever'},
            {'severity': 'warning', 'type': 'heat_stress'},
            {'severity': 'info', 'type': 'maintenance'}
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=pd.DataFrame(),
            active_alerts=active_alerts
        )
        
        # Should have combined penalty
        self.assertLess(score.score, 15)
        self.assertEqual(score.details['critical_alerts'], 1)
        self.assertEqual(score.details['warning_alerts'], 1)
        self.assertEqual(score.details['info_alerts'], 1)
    
    def test_resolution_bonus(self):
        """Test bonus for resolved alerts."""
        resolved_alerts = [
            {'severity': 'warning', 'type': 'fever', 'resolved_at': datetime.now()},
            {'severity': 'warning', 'type': 'inactivity', 'resolved_at': datetime.now()}
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=pd.DataFrame(),
            active_alerts=[],
            resolved_alerts=resolved_alerts
        )
        
        # Should have resolution bonus
        self.assertGreater(score.score, 25)  # Can exceed base
        self.assertIn('resolution_bonus', score.details)
    
    def test_alert_trend_analysis(self):
        """Test alert trend detection."""
        # Improving trend: fewer recent alerts
        alert_history = [
            {'timestamp': datetime.now() - timedelta(days=3)},
            {'timestamp': datetime.now() - timedelta(days=3)},
            {'timestamp': datetime.now() - timedelta(days=3)},
            {'timestamp': datetime.now() - timedelta(hours=1)},
        ]
        
        score = self.component.calculate_score(
            cow_id="COW_001",
            data=pd.DataFrame(),
            active_alerts=[],
            alert_history=alert_history
        )
        
        # Should detect improving trend
        self.assertIn('alert_trend', score.details)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrames."""
        components = [
            TemperatureScoreComponent(),
            ActivityScoreComponent(),
            BehavioralScoreComponent(),
        ]
        
        for component in components:
            score = component.calculate_score(
                cow_id="COW_001",
                data=pd.DataFrame()
            )
            
            self.assertEqual(score.score, 0.0)
            self.assertEqual(score.confidence, 0.0)
            self.assertGreater(len(score.warnings), 0)
    
    def test_all_nan_data(self):
        """Test handling of all NaN values."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [np.nan] * 100,
            'movement_intensity': [np.nan] * 100,
            'behavioral_state': [np.nan] * 100
        })
        
        temp_component = TemperatureScoreComponent()
        score = temp_component.calculate_score(
            cow_id="COW_001",
            data=data
        )
        
        self.assertEqual(score.score, 0.0)
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [45.0] * 100,  # Extremely high
            'movement_intensity': [10.0] * 100  # Very high
        })
        
        temp_component = TemperatureScoreComponent()
        temp_score = temp_component.calculate_score(
            cow_id="COW_001",
            data=data,
            baseline_temp=38.5
        )
        
        # Should handle extreme values and still clamp score
        self.assertGreaterEqual(temp_score.score, 0.0)
        self.assertLessEqual(temp_score.score, 25.0)
    
    def test_single_data_point(self):
        """Test handling of single data point."""
        data = pd.DataFrame({
            'timestamp': [datetime.now()],
            'temperature': [38.5]
        })
        
        component = TemperatureScoreComponent()
        score = component.calculate_score(
            cow_id="COW_001",
            data=data
        )
        
        # Should work but with lower confidence
        self.assertGreater(score.confidence, 0.0)
        self.assertLessEqual(score.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()
