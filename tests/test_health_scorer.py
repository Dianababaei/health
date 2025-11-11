"""
Unit Tests for Health Scorer Module

Tests cover:
- Health score calculation with full data
- Component weight configuration
- Score aggregation and normalization
- Health category classification
- Score smoothing
- Missing data handling
- Configuration loading
- Custom component integration
- Edge cases and validation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import tempfile
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.health_intelligence.scoring import (
    HealthScorer,
    HealthScore,
    BaseScoreComponent,
    ComponentScore,
    TemperatureScoreComponent,
)


class TestHealthScore(unittest.TestCase):
    """Test HealthScore data structure."""
    
    def test_create_health_score(self):
        """Test creating a HealthScore object."""
        component_scores = {
            'temperature_stability': ComponentScore(
                score=20.0, normalized_score=0.8, confidence=0.9, details={}
            ),
            'activity_level': ComponentScore(
                score=18.0, normalized_score=0.72, confidence=0.85, details={}
            ),
            'behavioral_patterns': ComponentScore(
                score=22.0, normalized_score=0.88, confidence=0.8, details={}
            ),
            'alert_frequency': ComponentScore(
                score=25.0, normalized_score=1.0, confidence=0.95, details={}
            ),
        }
        
        health_score = HealthScore(
            timestamp=datetime.now(),
            cow_id="COW_001",
            total_score=85.0,
            component_scores=component_scores,
            weights={'temperature_stability': 0.3, 'activity_level': 0.25,
                    'behavioral_patterns': 0.25, 'alert_frequency': 0.2},
            health_category='excellent',
            confidence=0.875
        )
        
        self.assertEqual(health_score.cow_id, "COW_001")
        self.assertEqual(health_score.total_score, 85.0)
        self.assertEqual(health_score.health_category, 'excellent')
    
    def test_to_dict_conversion(self):
        """Test converting HealthScore to dictionary."""
        component_scores = {
            'temperature_stability': ComponentScore(
                score=20.0, normalized_score=0.8, confidence=0.9, details={}
            ),
            'activity_level': ComponentScore(
                score=18.0, normalized_score=0.72, confidence=0.85, details={}
            ),
            'behavioral_patterns': ComponentScore(
                score=22.0, normalized_score=0.88, confidence=0.8, details={}
            ),
            'alert_frequency': ComponentScore(
                score=25.0, normalized_score=1.0, confidence=0.95, details={}
            ),
        }
        
        health_score = HealthScore(
            timestamp=datetime.now(),
            cow_id="COW_001",
            total_score=85.0,
            component_scores=component_scores,
            weights={'temperature_stability': 0.3, 'activity_level': 0.25,
                    'behavioral_patterns': 0.25, 'alert_frequency': 0.2},
            health_category='excellent',
            confidence=0.875
        )
        
        result = health_score.to_dict()
        
        self.assertIn('total_score', result)
        self.assertIn('health_category', result)
        self.assertIn('component_details', result)
        self.assertEqual(result['cow_id'], "COW_001")


class TestHealthScorerInitialization(unittest.TestCase):
    """Test HealthScorer initialization and configuration."""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration."""
        scorer = HealthScorer()
        
        self.assertIsNotNone(scorer.components)
        self.assertIsNotNone(scorer.weights)
        self.assertEqual(len(scorer.components), 4)
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration file."""
        # Create temporary config
        config_data = {
            'component_weights': {
                'temperature_stability': 0.4,
                'activity_level': 0.3,
                'behavioral_patterns': 0.2,
                'alert_frequency': 0.1,
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            scorer = HealthScorer(config_path=config_path)
            self.assertEqual(scorer.weights['temperature_stability'], 0.4)
            self.assertEqual(scorer.weights['activity_level'], 0.3)
        finally:
            os.unlink(config_path)
    
    def test_weight_validation(self):
        """Test that weights are validated and normalized."""
        config_data = {
            'component_weights': {
                'temperature_stability': 0.5,
                'activity_level': 0.3,
                'behavioral_patterns': 0.2,
                'alert_frequency': 0.2,  # Sum = 1.2
            },
            'validation': {
                'enforce_weight_sum': True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            scorer = HealthScorer(config_path=config_path)
            # Weights should be normalized
            weight_sum = sum(scorer.weights.values())
            self.assertAlmostEqual(weight_sum, 1.0, places=2)
        finally:
            os.unlink(config_path)
    
    def test_custom_component_integration(self):
        """Test integration of custom scoring component."""
        
        class CustomTemperatureComponent(BaseScoreComponent):
            def get_required_columns(self):
                return ['timestamp', 'temperature']
            
            def calculate_score(self, cow_id, data, **kwargs):
                return ComponentScore(
                    score=15.0,
                    normalized_score=0.6,
                    confidence=0.9,
                    details={'custom': True}
                )
        
        custom_component = CustomTemperatureComponent()
        scorer = HealthScorer(
            custom_components={'temperature_stability': custom_component}
        )
        
        self.assertIsInstance(
            scorer.components['temperature_stability'],
            CustomTemperatureComponent
        )


class TestHealthScoreCalculation(unittest.TestCase):
    """Test health score calculation."""
    
    def setUp(self):
        """Set up test scorer."""
        self.scorer = HealthScorer()
    
    def test_calculate_score_with_full_data(self):
        """Test score calculation with complete data."""
        # Temperature data
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'temperature': [38.5 + np.random.normal(0, 0.1) for _ in range(1440)]
        })
        
        # Activity data
        activity_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'movement_intensity': [0.3 + np.random.normal(0, 0.05) for _ in range(1440)]
        })
        
        # Behavioral data
        behavioral_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'behavioral_state': (['lying'] * 400 + ['standing'] * 500 +
                                ['ruminating'] * 400 + ['walking'] * 140)
        })
        
        # No active alerts
        active_alerts = []
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            activity_data=activity_data,
            behavioral_data=behavioral_data,
            active_alerts=active_alerts,
            baseline_temp=38.5,
            baseline_activity=0.3
        )
        
        self.assertIsInstance(score, HealthScore)
        self.assertGreaterEqual(score.total_score, 0)
        self.assertLessEqual(score.total_score, 100)
        self.assertIn(score.health_category, ['excellent', 'good', 'moderate', 'poor'])
        self.assertGreater(score.confidence, 0)
    
    def test_calculate_score_with_partial_data(self):
        """Test score calculation with only some data available."""
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100
        })
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            baseline_temp=38.5
        )
        
        # Should still calculate score with available data
        self.assertIsInstance(score, HealthScore)
        self.assertGreaterEqual(score.total_score, 0)
        self.assertLessEqual(score.total_score, 100)
    
    def test_score_with_health_issues(self):
        """Test score calculation with various health issues."""
        # High temperature (fever)
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [40.0] * 100  # Fever
        })
        
        # Low activity
        activity_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'movement_intensity': [0.05] * 100  # Very low
        })
        
        # Critical alerts
        active_alerts = [
            {'severity': 'critical', 'type': 'fever'},
            {'severity': 'critical', 'type': 'inactivity'}
        ]
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            activity_data=activity_data,
            active_alerts=active_alerts,
            baseline_temp=38.5,
            baseline_activity=0.3
        )
        
        # Score should be low due to health issues
        self.assertLess(score.total_score, 60)
        self.assertIn(score.health_category, ['moderate', 'poor'])
    
    def test_perfect_health_score(self):
        """Test score calculation with perfect health metrics."""
        # Perfect temperature
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'temperature': [38.5] * 1440
        })
        
        # Optimal activity
        activity_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'movement_intensity': [0.3] * 1440
        })
        
        # Optimal behavioral pattern
        behavioral_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
            'behavioral_state': (['lying'] * 400 + ['standing'] * 540 +
                                ['ruminating'] * 450 + ['walking'] * 50)
        })
        
        # No alerts
        active_alerts = []
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            activity_data=activity_data,
            behavioral_data=behavioral_data,
            active_alerts=active_alerts,
            baseline_temp=38.5,
            baseline_activity=0.3
        )
        
        # Should get high score
        self.assertGreater(score.total_score, 75)
        self.assertIn(score.health_category, ['excellent', 'good'])


class TestHealthCategoryClassification(unittest.TestCase):
    """Test health category classification."""
    
    def setUp(self):
        """Set up test scorer."""
        self.scorer = HealthScorer()
    
    def test_excellent_category(self):
        """Test classification of excellent health."""
        category = self.scorer._classify_health_category(90)
        self.assertEqual(category, 'excellent')
    
    def test_good_category(self):
        """Test classification of good health."""
        category = self.scorer._classify_health_category(70)
        self.assertEqual(category, 'good')
    
    def test_moderate_category(self):
        """Test classification of moderate health."""
        category = self.scorer._classify_health_category(50)
        self.assertEqual(category, 'moderate')
    
    def test_poor_category(self):
        """Test classification of poor health."""
        category = self.scorer._classify_health_category(30)
        self.assertEqual(category, 'poor')
    
    def test_boundary_conditions(self):
        """Test category boundaries."""
        self.assertEqual(self.scorer._classify_health_category(80), 'excellent')
        self.assertEqual(self.scorer._classify_health_category(79.9), 'good')
        self.assertEqual(self.scorer._classify_health_category(60), 'good')
        self.assertEqual(self.scorer._classify_health_category(59.9), 'moderate')
        self.assertEqual(self.scorer._classify_health_category(40), 'moderate')
        self.assertEqual(self.scorer._classify_health_category(39.9), 'poor')


class TestScoreSmoothing(unittest.TestCase):
    """Test score smoothing functionality."""
    
    def setUp(self):
        """Set up test scorer with smoothing enabled."""
        config_data = {
            'component_weights': {
                'temperature_stability': 0.3,
                'activity_level': 0.25,
                'behavioral_patterns': 0.25,
                'alert_frequency': 0.2,
            },
            'calculation': {
                'smoothing_enabled': True,
                'smoothing_factor': 0.3
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            self.config_path = f.name
        
        self.scorer = HealthScorer(config_path=self.config_path)
    
    def tearDown(self):
        """Clean up config file."""
        os.unlink(self.config_path)
    
    def test_smoothing_applied(self):
        """Test that smoothing is applied when previous score provided."""
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100
        })
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            baseline_temp=38.5,
            previous_score=70.0  # Previous score
        )
        
        self.assertTrue(score.metadata['smoothing_applied'])
        self.assertIn('smoothed_score', score.metadata)
    
    def test_no_smoothing_without_previous(self):
        """Test that smoothing is not applied without previous score."""
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100
        })
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            baseline_temp=38.5
        )
        
        self.assertFalse(score.metadata['smoothing_applied'])


class TestScoreBreakdown(unittest.TestCase):
    """Test score breakdown functionality."""
    
    def setUp(self):
        """Set up test scorer."""
        self.scorer = HealthScorer()
    
    def test_get_score_breakdown(self):
        """Test getting detailed score breakdown."""
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100
        })
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            baseline_temp=38.5
        )
        
        breakdown = self.scorer.get_score_breakdown(score)
        
        self.assertIn('total_score', breakdown)
        self.assertIn('health_category', breakdown)
        self.assertIn('components', breakdown)
        self.assertIn('temperature_stability', breakdown['components'])
        
        # Each component should have weight and contribution
        for component_name, component_data in breakdown['components'].items():
            self.assertIn('weight', component_data)
            self.assertIn('contribution_to_total', component_data)
            self.assertIn('raw_score', component_data)


class TestWeightManagement(unittest.TestCase):
    """Test dynamic weight management."""
    
    def setUp(self):
        """Set up test scorer."""
        self.scorer = HealthScorer()
    
    def test_update_weights(self):
        """Test updating component weights."""
        original_weight = self.scorer.weights['temperature_stability']
        
        new_weights = {'temperature_stability': 0.5}
        self.scorer.update_weights(new_weights)
        
        self.assertNotEqual(
            self.scorer.weights['temperature_stability'],
            original_weight
        )
    
    def test_replace_component(self):
        """Test replacing a scoring component."""
        class CustomComponent(BaseScoreComponent):
            def get_required_columns(self):
                return []
            
            def calculate_score(self, cow_id, data, **kwargs):
                return ComponentScore(
                    score=20.0, normalized_score=0.8,
                    confidence=0.9, details={}
                )
        
        custom = CustomComponent()
        self.scorer.replace_component('temperature_stability', custom)
        
        self.assertIsInstance(
            self.scorer.components['temperature_stability'],
            CustomComponent
        )


class TestEdgeCasesAndValidation(unittest.TestCase):
    """Test edge cases and validation."""
    
    def setUp(self):
        """Set up test scorer."""
        self.scorer = HealthScorer()
    
    def test_no_data_provided(self):
        """Test score calculation with no data."""
        score = self.scorer.calculate_score(cow_id="COW_001")
        
        # Should still return valid HealthScore
        self.assertIsInstance(score, HealthScore)
        self.assertGreaterEqual(score.total_score, 0)
        self.assertLessEqual(score.total_score, 100)
    
    def test_score_range_validation(self):
        """Test that scores are always in valid range."""
        # Test with various extreme inputs
        test_cases = [
            {'temperature_data': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
                'temperature': [50.0] * 10  # Impossibly high
            })},
            {'activity_data': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=10, freq='1min'),
                'movement_intensity': [100.0] * 10  # Impossibly high
            })},
        ]
        
        for test_case in test_cases:
            score = self.scorer.calculate_score(
                cow_id="COW_001",
                **test_case
            )
            
            self.assertGreaterEqual(score.total_score, 0)
            self.assertLessEqual(score.total_score, 100)
    
    def test_confidence_calculation(self):
        """Test overall confidence calculation."""
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'temperature': [38.5] * 100
        })
        
        score = self.scorer.calculate_score(
            cow_id="COW_001",
            temperature_data=temp_data,
            baseline_temp=38.5
        )
        
        # Confidence should be in valid range
        self.assertGreaterEqual(score.confidence, 0.0)
        self.assertLessEqual(score.confidence, 1.0)


if __name__ == '__main__':
    unittest.main()
