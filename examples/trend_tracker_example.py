"""
Example Usage: Multi-Day Health Trend Tracker

Demonstrates how to use the trend tracker to analyze health trends
over multiple time periods and generate dashboard-ready reports.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from health_intelligence import (
    MultiDayHealthTrendTracker,
    TrendIndicator,
    HealthTrendReport
)


def generate_sample_data(days: int, cow_id: str = "COW_001"):
    """Generate sample data for demonstration."""

    # Generate temperature data (1 sample per minute)
    samples = days * 1440
    dates = pd.date_range(end=datetime.now(), periods=samples, freq='1min')

    # Simulate a recovery pattern: high temperature decreasing over time
    base_temp = 39.0  # Starting with mild fever
    temp_drop = 0.5   # Dropping 0.5°C over the period
    temperatures = base_temp - (np.arange(samples) / samples) * temp_drop
    temperatures += np.random.normal(0, 0.15, samples)  # Add natural variation

    temperature_data = pd.DataFrame({
        'timestamp': dates,
        'temperature': temperatures
    })

    # Generate activity data
    # Simulate increasing activity over recovery period
    base_activity = 0.3
    activity_increase = 0.3
    movement = base_activity + (np.arange(samples) / samples) * activity_increase
    movement += np.random.normal(0, 0.05, samples)
    movement = np.clip(movement, 0, 1)

    # Simulate behavioral states - more diverse as animal recovers
    if days < 7:
        # Early recovery - mostly resting
        states = np.random.choice(
            ['lying', 'standing', 'walking', 'ruminating', 'feeding'],
            samples,
            p=[0.60, 0.20, 0.10, 0.05, 0.05]
        )
    else:
        # Later recovery - more normal behavior
        states = np.random.choice(
            ['lying', 'standing', 'walking', 'ruminating', 'feeding'],
            samples,
            p=[0.40, 0.25, 0.15, 0.10, 0.10]
        )

    activity_data = pd.DataFrame({
        'timestamp': dates,
        'behavioral_state': states,
        'movement_intensity': movement
    })

    # Generate alert history - decreasing frequency over recovery
    early_alerts = int(days * 2.0)  # High alert frequency early on
    late_alerts = max(1, int(days * 0.3))  # Lower frequency later
    total_alerts = (early_alerts + late_alerts) // 2

    alert_history = []
    end_date = datetime.now()

    for i in range(total_alerts):
        # More alerts in earlier days
        days_back = np.random.exponential(days / 4)  # Exponential distribution
        days_back = min(days_back, days)
        alert_time = end_date - timedelta(days=days_back)

        # Earlier alerts more likely to be critical
        if days_back > days / 2:
            severity = 'critical' if np.random.random() < 0.5 else 'warning'
        else:
            severity = 'warning' if np.random.random() < 0.7 else 'critical'

        alert_history.append({
            'timestamp': alert_time,
            'alert_type': np.random.choice(['fever', 'heat_stress', 'inactivity']),
            'severity': severity,
            'cow_id': cow_id
        })

    behavioral_states = activity_data.copy()

    return temperature_data, activity_data, alert_history, behavioral_states


def print_report_summary(report: HealthTrendReport):
    """Print a formatted summary of the trend report."""

    print("\n" + "="*70)
    print(f"HEALTH TREND REPORT - {report.cow_id}")
    print(f"Analysis Date: {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    print(f"\nOVERALL TREND: {report.overall_trend.value.upper()}")
    print(f"Confidence Score: {report.overall_confidence:.1%}")

    # Print each period's metrics
    periods = [
        ('7-Day Trend', report.trend_7day),
        ('14-Day Trend', report.trend_14day),
        ('30-Day Trend', report.trend_30day),
        ('90-Day Trend', report.trend_90day)
    ]

    print("\n" + "-"*70)
    print("PERIOD ANALYSIS")
    print("-"*70)

    for period_name, period_data in periods:
        if period_data is None:
            print(f"\n{period_name}: Insufficient data")
            continue

        print(f"\n{period_name}:")
        print(f"  Status: {period_data.trend_indicator.value}")
        print(f"  Confidence: {period_data.confidence_score:.1%}")
        print(f"  Data Completeness: {period_data.data_completeness:.1%}")

        print(f"\n  Temperature:")
        print(f"    Mean: {period_data.temperature_mean:.2f}°C")
        print(f"    Std Dev: {period_data.temperature_std:.2f}°C")
        print(f"    Baseline Drift: {period_data.temperature_baseline_drift:+.2f}°C")
        print(f"    Anomalies: {period_data.temperature_anomaly_count}")

        print(f"\n  Activity:")
        print(f"    Total Active Minutes: {period_data.total_activity_minutes:.0f}")
        print(f"    Mean Activity Level: {period_data.activity_level_mean:.2f}")
        print(f"    Rest Minutes: {period_data.rest_minutes:.0f}")
        print(f"    Behavioral Diversity: {period_data.activity_diversity:.2f}")

        print(f"\n  Alerts:")
        print(f"    Total Count: {period_data.alert_count}")
        if period_data.alert_severity_distribution:
            print(f"    Severity Distribution: {period_data.alert_severity_distribution}")

    # Print comparisons
    if report.period_comparisons:
        print("\n" + "-"*70)
        print("PERIOD COMPARISONS")
        print("-"*70)

        for comp_name, comp_data in report.period_comparisons.items():
            print(f"\n{comp_name}:")
            print(f"  Temperature Change: {comp_data['temperature_delta']:+.2f}°C")
            print(f"  Activity Change: {comp_data['activity_delta']:+.2f}")
            print(f"  Alert Change: {comp_data['alert_delta']:+d}")
            print(f"  Trend: {comp_data['trend_change']}")

    # Print significant changes
    if report.significant_changes:
        print("\n" + "-"*70)
        print("SIGNIFICANT CHANGES")
        print("-"*70)
        for change in report.significant_changes:
            print(f"  • {change}")

    # Print recommendations
    if report.recommendations:
        print("\n" + "-"*70)
        print("RECOMMENDATIONS")
        print("-"*70)
        for rec in report.recommendations:
            print(f"  • {rec}")

    print("\n" + "="*70 + "\n")


def main():
    """Main example demonstrating trend tracker usage."""

    print("Multi-Day Health Trend Tracker - Example Usage")
    print("=" * 70)

    # Initialize tracker
    tracker = MultiDayHealthTrendTracker(temperature_baseline=38.5)
    print(f"\nTracker initialized with baseline temperature: 38.5°C")
    print(f"Analysis periods: {tracker.periods} days")

    # Example 1: Animal recovering from illness (30 days of data)
    print("\n\nEXAMPLE 1: Animal Recovering from Illness")
    print("-" * 70)

    temp_data, activity_data, alert_history, behavioral_states = generate_sample_data(
        days=30,
        cow_id="COW_001_RECOVERING"
    )

    print(f"Generated 30 days of synthetic recovery data")
    print(f"  Temperature samples: {len(temp_data)}")
    print(f"  Activity samples: {len(activity_data)}")
    print(f"  Alert events: {len(alert_history)}")

    # Analyze trends
    report = tracker.analyze_trends(
        cow_id="COW_001_RECOVERING",
        temperature_data=temp_data,
        activity_data=activity_data,
        alert_history=alert_history,
        behavioral_states=behavioral_states
    )

    # Print summary
    print_report_summary(report)

    # Example 2: Export to JSON for dashboard
    print("\nEXAMPLE 2: Dashboard JSON Export")
    print("-" * 70)

    dashboard_data = report.to_dict()

    print(f"Report exported to JSON-serializable dictionary")
    print(f"  Top-level keys: {list(dashboard_data.keys())}")
    print(f"  Overall trend: {dashboard_data['overall_trend']}")
    print(f"  Overall confidence: {dashboard_data['overall_confidence']:.1%}")

    # Show sample of dashboard data structure
    if dashboard_data['trend_7day']:
        print(f"\n  7-day period structure:")
        print(f"    Keys: {list(dashboard_data['trend_7day'].keys())}")
        print(f"    Temperature metrics: {list(dashboard_data['trend_7day']['temperature'].keys())}")
        print(f"    Activity metrics: {list(dashboard_data['trend_7day']['activity'].keys())}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
