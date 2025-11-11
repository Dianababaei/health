"""
Example Usage: Alert Logging and State Management

Demonstrates how to use the AlertLogger and AlertStateManager
for comprehensive alert handling.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.health_intelligence.logging import AlertLogger, AlertStateManager, AlertStatus


def example_1_basic_logging():
    """Example 1: Basic alert logging to JSON and CSV."""
    print("=" * 60)
    print("Example 1: Basic Alert Logging")
    print("=" * 60)
    
    # Initialize logger
    logger = AlertLogger(
        log_dir="logs/alerts",
        retention_days=180,
        auto_cleanup=True
    )
    
    # Create a sample alert
    alert_data = {
        'alert_id': 'example-001',
        'cow_id': 'COW001',
        'alert_type': 'fever',
        'severity': 'critical',
        'confidence': 0.95,
        'sensor_values': {
            'temperature': 40.5,
            'heart_rate': 95
        },
        'detection_details': {
            'baseline_temp': 38.5,
            'threshold': 39.5,
            'duration_minutes': 30
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Log the alert
    success = logger.log_alert(alert_data)
    print(f"Alert logged: {success}")
    print(f"Alert ID: {alert_data['alert_id']}")
    print(f"Cow ID: {alert_data['cow_id']}")
    print(f"Type: {alert_data['alert_type']}")
    print(f"Severity: {alert_data['severity']}")
    print()


def example_2_batch_logging():
    """Example 2: Batch logging multiple alerts."""
    print("=" * 60)
    print("Example 2: Batch Alert Logging")
    print("=" * 60)
    
    logger = AlertLogger(log_dir="logs/alerts")
    
    # Create multiple alerts
    alerts = []
    for i in range(5):
        alerts.append({
            'alert_id': f'batch-{i}',
            'cow_id': f'COW00{i}',
            'alert_type': 'heat_stress',
            'severity': 'high',
            'confidence': 0.85 + i * 0.02,
            'sensor_values': {'temperature': 39.5 + i * 0.2},
            'detection_details': {'duration_minutes': 60 + i * 10},
            'timestamp': datetime.now().isoformat()
        })
    
    # Log batch
    success_count = logger.log_alerts_batch(alerts)
    print(f"Successfully logged {success_count}/{len(alerts)} alerts")
    print()


def example_3_state_management():
    """Example 3: Alert state lifecycle management."""
    print("=" * 60)
    print("Example 3: Alert State Management")
    print("=" * 60)
    
    # Initialize state manager
    manager = AlertStateManager(db_path="data/alert_state.db")
    
    # Create an alert
    alert_data = {
        'alert_id': 'state-example-001',
        'cow_id': 'COW010',
        'alert_type': 'abnormal_behavior',
        'severity': 'warning',
        'confidence': 0.78,
        'sensor_values': {'activity_level': 0.3},
        'detection_details': {'pattern': 'decreased_activity'},
        'timestamp': datetime.now().isoformat()
    }
    
    print("1. Creating alert...")
    manager.create_alert(alert_data)
    alert = manager.get_alert('state-example-001')
    print(f"   Status: {alert['status']}")
    
    print("\n2. Acknowledging alert...")
    manager.acknowledge_alert('state-example-001', 'Farmer notified')
    alert = manager.get_alert('state-example-001')
    print(f"   Status: {alert['status']}")
    
    print("\n3. Resolving alert...")
    manager.resolve_alert('state-example-001', 'Cow returned to normal behavior')
    alert = manager.get_alert('state-example-001')
    print(f"   Status: {alert['status']}")
    
    print("\n4. State history:")
    history = manager.get_state_history('state-example-001')
    for entry in history:
        print(f"   {entry['old_status'] or 'NEW'} → {entry['new_status']}")
        print(f"   Time: {entry['changed_at']}")
        print(f"   Notes: {entry['notes']}")
        print()


def example_4_querying_alerts():
    """Example 4: Querying and filtering alerts."""
    print("=" * 60)
    print("Example 4: Querying Alerts")
    print("=" * 60)
    
    manager = AlertStateManager(db_path="data/alert_state.db")
    
    # Create sample alerts with different attributes
    alert_types = ['fever', 'heat_stress', 'abnormal_behavior']
    severities = ['critical', 'high', 'warning']
    
    print("Creating sample alerts...")
    for i in range(10):
        alert_data = {
            'alert_id': f'query-example-{i}',
            'cow_id': f'COW{100 + i % 3}',
            'alert_type': alert_types[i % 3],
            'severity': severities[i % 3],
            'confidence': 0.7 + i * 0.02,
            'timestamp': (datetime.now() - timedelta(hours=i)).isoformat()
        }
        manager.create_alert(alert_data)
    
    # Query examples
    print("\n1. All active alerts:")
    active_alerts = manager.query_alerts(status='active', limit=5)
    print(f"   Found {len(active_alerts)} alerts")
    
    print("\n2. Critical severity alerts:")
    critical_alerts = manager.query_alerts(severity='critical')
    print(f"   Found {len(critical_alerts)} alerts")
    
    print("\n3. Alerts for specific cow:")
    cow_alerts = manager.query_alerts(cow_id='COW100')
    print(f"   Found {len(cow_alerts)} alerts for COW100")
    
    print("\n4. Fever alerts:")
    fever_alerts = manager.query_alerts(alert_type='fever')
    print(f"   Found {len(fever_alerts)} fever alerts")
    print()


def example_5_statistics():
    """Example 5: Getting alert statistics and analytics."""
    print("=" * 60)
    print("Example 5: Alert Statistics")
    print("=" * 60)
    
    manager = AlertStateManager(db_path="data/alert_state.db")
    
    # Get overall statistics
    stats = manager.get_statistics()
    
    print("Overall Statistics:")
    print(f"  Total Alerts: {stats.get('total_alerts', 0)}")
    print(f"  False Positive Rate: {stats.get('false_positive_rate', 0):.2f}%")
    print(f"  Avg Resolution Time: {stats.get('avg_resolution_time_minutes', 0):.1f} minutes")
    
    print("\nBy Status:")
    for status, count in stats.get('by_status', {}).items():
        print(f"  {status}: {count}")
    
    print("\nBy Severity:")
    for severity, count in stats.get('by_severity', {}).items():
        print(f"  {severity}: {count}")
    
    print("\nTop Alert Types:")
    for alert_type, count in list(stats.get('by_type', {}).items())[:5]:
        print(f"  {alert_type}: {count}")
    print()


def example_6_false_positive_handling():
    """Example 6: Handling false positive alerts."""
    print("=" * 60)
    print("Example 6: False Positive Handling")
    print("=" * 60)
    
    manager = AlertStateManager(db_path="data/alert_state.db")
    logger = AlertLogger(log_dir="logs/alerts")
    
    # Create and log an alert
    alert_data = {
        'alert_id': 'fp-example-001',
        'cow_id': 'COW020',
        'alert_type': 'fever',
        'severity': 'critical',
        'confidence': 0.88,
        'sensor_values': {'temperature': 39.8},
        'detection_details': {'threshold': 39.5},
        'timestamp': datetime.now().isoformat()
    }
    
    print("1. Creating alert...")
    logger.log_alert(alert_data)
    manager.create_alert(alert_data)
    
    print("2. Investigation revealed it was a false positive...")
    print("   (e.g., sensor temporarily affected by external heat source)")
    
    print("3. Marking as false positive...")
    manager.mark_false_positive(
        'fp-example-001',
        'Sensor placed near heating lamp, not actual fever'
    )
    
    alert = manager.get_alert('fp-example-001')
    print(f"   Status: {alert['status']}")
    print(f"   Notes: {alert['resolution_notes']}")
    print()


def example_7_log_cleanup():
    """Example 7: Log file cleanup and retention."""
    print("=" * 60)
    print("Example 7: Log Cleanup")
    print("=" * 60)
    
    logger = AlertLogger(
        log_dir="logs/alerts",
        retention_days=90,
        auto_cleanup=False  # Manual control
    )
    
    # Get statistics before cleanup
    stats_before = logger.get_log_statistics()
    print("Before cleanup:")
    print(f"  Total JSON files: {stats_before['total_json_files']}")
    print(f"  Total CSV files: {stats_before['total_csv_files']}")
    print(f"  Total alerts: {stats_before['total_alerts']}")
    print(f"  Disk usage: {stats_before['disk_usage_mb']:.2f} MB")
    
    # Run cleanup
    print("\nRunning cleanup...")
    deleted_count = logger.cleanup_old_logs()
    print(f"  Deleted {deleted_count} expired log files")
    
    # Get statistics after cleanup
    stats_after = logger.get_log_statistics()
    print("\nAfter cleanup:")
    print(f"  Total JSON files: {stats_after['total_json_files']}")
    print(f"  Total CSV files: {stats_after['total_csv_files']}")
    print(f"  Disk usage: {stats_after['disk_usage_mb']:.2f} MB")
    print()


def example_8_integration():
    """Example 8: Complete integration example."""
    print("=" * 60)
    print("Example 8: Complete Integration")
    print("=" * 60)
    
    # Initialize both systems
    logger = AlertLogger(log_dir="logs/alerts")
    manager = AlertStateManager(db_path="data/alert_state.db")
    
    print("Simulating alert detection and processing workflow...\n")
    
    # Simulate detected alerts from monitoring system
    detected_alerts = [
        {
            'alert_id': f'integration-{i}',
            'cow_id': f'COW{200 + i}',
            'alert_type': ['fever', 'heat_stress', 'abnormal_behavior'][i % 3],
            'severity': ['critical', 'high', 'warning'][i % 3],
            'confidence': 0.75 + i * 0.03,
            'sensor_values': {'temperature': 39.0 + i * 0.3},
            'detection_details': {'method': 'threshold'},
            'timestamp': datetime.now().isoformat()
        }
        for i in range(3)
    ]
    
    # Process each alert
    for alert_data in detected_alerts:
        print(f"Processing alert: {alert_data['alert_id']}")
        print(f"  Type: {alert_data['alert_type']}")
        print(f"  Severity: {alert_data['severity']}")
        print(f"  Cow: {alert_data['cow_id']}")
        
        # Log to file (JSON + CSV)
        logger.log_alert(alert_data)
        print("  ✓ Logged to file")
        
        # Track state in database
        manager.create_alert(alert_data)
        print("  ✓ State tracked in database")
        
        # Simulate acknowledgment for high severity
        if alert_data['severity'] in ['critical', 'high']:
            manager.acknowledge_alert(alert_data['alert_id'], 'Auto-acknowledged')
            print("  ✓ Auto-acknowledged")
        
        print()
    
    print("Summary:")
    stats = manager.get_statistics()
    print(f"  Total active alerts: {stats.get('by_status', {}).get('active', 0)}")
    print(f"  Total acknowledged: {stats.get('by_status', {}).get('acknowledged', 0)}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ALERT LOGGING AND STATE MANAGEMENT EXAMPLES")
    print("=" * 60 + "\n")
    
    examples = [
        example_1_basic_logging,
        example_2_batch_logging,
        example_3_state_management,
        example_4_querying_alerts,
        example_5_statistics,
        example_6_false_positive_handling,
        example_7_log_cleanup,
        example_8_integration,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
        
        input("Press Enter to continue...")
        print()


if __name__ == "__main__":
    main()
