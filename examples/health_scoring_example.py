"""
Health Scoring System - Example Usage

This example demonstrates how to use the health scoring system to calculate
comprehensive health scores for cattle from multiple data sources.

Features demonstrated:
1. Basic score calculation with sample data
2. Score interpretation and breakdown
3. Handling different health scenarios
4. Custom component integration (optional)
5. Weight adjustment
6. Score smoothing over time
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.health_intelligence.scoring import HealthScorer, HealthScore


def example_1_basic_scoring():
    """Example 1: Basic health score calculation."""
    print("=" * 60)
    print("Example 1: Basic Health Score Calculation")
    print("=" * 60)
    
    # Initialize scorer
    scorer = HealthScorer()
    
    # Create sample temperature data (normal temperature)
    temp_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'temperature': [38.5 + np.random.normal(0, 0.1) for _ in range(1440)]
    })
    
    # Create sample activity data (normal activity)
    activity_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'movement_intensity': [0.3 + np.random.normal(0, 0.05) for _ in range(1440)]
    })
    
    # Create sample behavioral data
    behavioral_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'behavioral_state': (
            ['lying'] * 400 +
            ['standing'] * 500 +
            ['ruminating'] * 450 +
            ['walking'] * 90
        )
    })
    
    # No active alerts (healthy animal)
    active_alerts = []
    
    # Calculate score
    score = scorer.calculate_score(
        cow_id="COW_001",
        temperature_data=temp_data,
        activity_data=activity_data,
        behavioral_data=behavioral_data,
        active_alerts=active_alerts,
        baseline_temp=38.5,
        baseline_activity=0.3
    )
    
    # Display results
    print(f"\nCow ID: {score.cow_id}")
    print(f"Health Score: {score.total_score:.1f}/100")
    print(f"Health Category: {score.health_category.upper()}")
    print(f"Confidence: {score.confidence:.2%}")
    print(f"Timestamp: {score.timestamp}")
    
    # Show component scores
    print("\nComponent Breakdown:")
    for component_name, component_score in score.component_scores.items():
        print(f"  {component_name}: {component_score.score:.1f}/25 "
              f"(confidence: {component_score.confidence:.2f})")
    
    print("\n")


def example_2_health_issues():
    """Example 2: Score calculation with health issues."""
    print("=" * 60)
    print("Example 2: Animal with Health Issues")
    print("=" * 60)
    
    scorer = HealthScorer()
    
    # High temperature (fever)
    temp_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'temperature': [40.0 + np.random.normal(0, 0.1) for _ in range(1440)]
    })
    
    # Low activity (lethargic)
    activity_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'movement_intensity': [0.1 + np.random.normal(0, 0.02) for _ in range(1440)]
    })
    
    # Reduced rumination
    behavioral_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'behavioral_state': (
            ['lying'] * 800 +      # Excessive lying
            ['standing'] * 400 +
            ['ruminating'] * 200 +  # Very low rumination
            ['walking'] * 40
        )
    })
    
    # Active health alerts
    active_alerts = [
        {'severity': 'critical', 'type': 'fever', 'timestamp': datetime.now()},
        {'severity': 'warning', 'type': 'inactivity', 'timestamp': datetime.now()}
    ]
    
    # Fever events
    fever_events = [
        {'timestamp': datetime.now() - timedelta(hours=2), 'max_temp': 40.1},
        {'timestamp': datetime.now() - timedelta(hours=1), 'max_temp': 40.2}
    ]
    
    # Calculate score
    score = scorer.calculate_score(
        cow_id="COW_002",
        temperature_data=temp_data,
        activity_data=activity_data,
        behavioral_data=behavioral_data,
        active_alerts=active_alerts,
        fever_events=fever_events,
        baseline_temp=38.5,
        baseline_activity=0.3
    )
    
    print(f"\nCow ID: {score.cow_id}")
    print(f"Health Score: {score.total_score:.1f}/100")
    print(f"Health Category: {score.health_category.upper()} ⚠️")
    print(f"Confidence: {score.confidence:.2%}")
    
    # Detailed breakdown
    breakdown = scorer.get_score_breakdown(score)
    print("\nDetailed Component Analysis:")
    for component_name, details in breakdown['components'].items():
        print(f"\n  {component_name}:")
        print(f"    Score: {details['raw_score']:.1f}/25")
        print(f"    Weight: {details['weight']}")
        print(f"    Contribution: {details['contribution_to_total']:.1f} points")
        
        # Show key details
        if 'deviation_penalty' in details['details']:
            print(f"    Deviation Penalty: {details['details']['deviation_penalty']:.1f}")
        if 'fever_penalty' in details['details']:
            print(f"    Fever Penalty: {details['details']['fever_penalty']:.1f}")
        if 'inactivity_penalty' in details['details']:
            print(f"    Inactivity Penalty: {details['details']['inactivity_penalty']:.1f}")
    
    print("\n")


def example_3_score_tracking():
    """Example 3: Track scores over time with smoothing."""
    print("=" * 60)
    print("Example 3: Score Tracking Over Time")
    print("=" * 60)
    
    scorer = HealthScorer()
    
    print("\nSimulating 5 hourly score calculations...")
    print(f"{'Hour':<6} {'Raw Score':<12} {'Smoothed':<12} {'Category':<12} {'Change'}")
    print("-" * 60)
    
    previous_score = None
    
    for hour in range(5):
        # Simulate gradually improving health
        health_factor = 0.6 + (hour * 0.1)  # 0.6 to 1.0
        
        temp_data = pd.DataFrame({
            'timestamp': pd.date_range(f'2024-01-01 {hour:02d}:00', periods=60, freq='1min'),
            'temperature': [38.5 + (1.0 - health_factor) * 1.5 + np.random.normal(0, 0.1) 
                          for _ in range(60)]
        })
        
        activity_data = pd.DataFrame({
            'timestamp': pd.date_range(f'2024-01-01 {hour:02d}:00', periods=60, freq='1min'),
            'movement_intensity': [0.3 * health_factor + np.random.normal(0, 0.02) 
                                  for _ in range(60)]
        })
        
        # Fewer alerts as health improves
        num_alerts = max(0, 3 - hour)
        active_alerts = [
            {'severity': 'warning', 'type': f'alert_{i}', 'timestamp': datetime.now()}
            for i in range(num_alerts)
        ]
        
        score = scorer.calculate_score(
            cow_id="COW_003",
            temperature_data=temp_data,
            activity_data=activity_data,
            active_alerts=active_alerts,
            baseline_temp=38.5,
            baseline_activity=0.3,
            previous_score=previous_score
        )
        
        raw = score.metadata.get('raw_score', score.total_score)
        smoothed = score.total_score
        change = f"+{smoothed - previous_score:.1f}" if previous_score else "N/A"
        
        print(f"{hour:<6} {raw:<12.1f} {smoothed:<12.1f} {score.health_category:<12} {change}")
        
        previous_score = score.total_score
    
    print("\n")


def example_4_weight_adjustment():
    """Example 4: Adjust component weights."""
    print("=" * 60)
    print("Example 4: Custom Component Weights")
    print("=" * 60)
    
    # Create sample data
    temp_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'temperature': [38.5] * 100
    })
    
    activity_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
        'movement_intensity': [0.3] * 100
    })
    
    # Calculate with default weights
    scorer_default = HealthScorer()
    score_default = scorer_default.calculate_score(
        cow_id="COW_004",
        temperature_data=temp_data,
        activity_data=activity_data,
        baseline_temp=38.5,
        baseline_activity=0.3
    )
    
    print("\nDefault Weights:")
    print(f"  {scorer_default.weights}")
    print(f"  Total Score: {score_default.total_score:.1f}/100")
    
    # Adjust weights to prioritize temperature
    scorer_custom = HealthScorer()
    scorer_custom.update_weights({
        'temperature_stability': 0.50,  # Increase temperature importance
        'activity_level': 0.20,
        'behavioral_patterns': 0.20,
        'alert_frequency': 0.10
    })
    
    score_custom = scorer_custom.calculate_score(
        cow_id="COW_004",
        temperature_data=temp_data,
        activity_data=activity_data,
        baseline_temp=38.5,
        baseline_activity=0.3
    )
    
    print("\nCustom Weights (Temperature-Focused):")
    print(f"  {scorer_custom.weights}")
    print(f"  Total Score: {score_custom.total_score:.1f}/100")
    print(f"  Difference: {score_custom.total_score - score_default.total_score:+.1f} points")
    
    print("\n")


def example_5_component_details():
    """Example 5: Examine component details."""
    print("=" * 60)
    print("Example 5: Component Details Analysis")
    print("=" * 60)
    
    scorer = HealthScorer()
    
    # Create data with some issues
    temp_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'temperature': [39.0 + np.random.normal(0, 0.2) for _ in range(1440)]
    })
    
    activity_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
        'movement_intensity': [0.2 + np.random.normal(0, 0.03) for _ in range(1440)]
    })
    
    score = scorer.calculate_score(
        cow_id="COW_005",
        temperature_data=temp_data,
        activity_data=activity_data,
        baseline_temp=38.5,
        baseline_activity=0.3
    )
    
    print(f"\nOverall Score: {score.total_score:.1f}/100 ({score.health_category})")
    print("\nComponent Details:")
    
    for component_name, component_score in score.component_scores.items():
        print(f"\n{component_name.upper()}:")
        print(f"  Score: {component_score.score:.1f}/25")
        print(f"  Formula: {component_score.details.get('formula', 'N/A')}")
        
        # Show relevant metrics
        details = component_score.details
        if 'mean_deviation' in details:
            print(f"  Mean Deviation: {details['mean_deviation']:.3f}°C")
            print(f"  Deviation Penalty: {details.get('deviation_penalty', 0):.1f}")
        if 'activity_deviation' in details:
            print(f"  Activity Deviation: {details['activity_deviation']:.3f}")
            print(f"  Current Activity: {details.get('current_activity', 0):.3f}")
        
        if component_score.warnings:
            print(f"  Warnings: {component_score.warnings}")
    
    print("\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("HEALTH SCORING SYSTEM - EXAMPLE USAGE")
    print("=" * 60 + "\n")
    
    try:
        example_1_basic_scoring()
        example_2_health_issues()
        example_3_score_tracking()
        example_4_weight_adjustment()
        example_5_component_details()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Review the configuration in config/health_score_weights.yaml")
        print("  2. Adjust component weights to match your priorities")
        print("  3. Integrate with your data pipeline")
        print("  4. Replace placeholder formulas with custom logic as needed")
        print("  5. Run tests: python -m pytest tests/test_health_scorer.py")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
