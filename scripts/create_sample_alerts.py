#!/usr/bin/env python3
"""
Create Sample Alerts for Dashboard Testing

Generates sample alerts in the AlertSystem format for testing and demonstration
of the Alert Dashboard functionality.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_intelligence.alert_system import AlertSystem


def create_sample_alerts():
    """Create sample alerts for testing."""
    
    print("Creating sample alerts...")
    
    # Initialize alert system
    alert_system = AlertSystem()
    
    # Clear existing alerts for clean demo
    alert_system.alerts.clear()
    
    # Sample cow IDs
    cow_ids = ["1042", "1043", "1044", "1045", "1046"]
    
    # Create various alert types
    sample_alerts = []
    
    # 1. Critical Fever Alert
    sample_alerts.append({
        'alert_id': f"fever_{datetime.now().strftime('%Y%m%d_%H%M%S')}_001",
        'timestamp': datetime.now() - timedelta(hours=2),
        'cow_id': cow_ids[0],
        'alert_type': 'fever',
        'sensor_values': {
            'temperature': 40.2,
            'motion_magnitude': 0.12,
            'behavioral_state': 'lying',
            'fxa': 0.05,
            'mya': 0.03,
            'rza': -0.58,
        },
        'description': 'High fever detected with reduced activity - immediate attention required',
        'metadata': {
            'baseline_temp': 38.5,
            'temp_deviation': 1.7,
            'duration_minutes': 6,
        },
    })
    
    # 2. Heat Stress Alert
    sample_alerts.append({
        'alert_id': f"heat_stress_{datetime.now().strftime('%Y%m%d_%H%M%S')}_002",
        'timestamp': datetime.now() - timedelta(hours=1),
        'cow_id': cow_ids[1],
        'alert_type': 'heat_stress',
        'sensor_values': {
            'temperature': 40.5,
            'activity_level': 0.65,
            'behavioral_state': 'walking',
        },
        'description': 'Heat stress detected - high temperature with elevated activity',
        'metadata': {
            'temp_rise_rate': 0.7,
            'duration_minutes': 10,
            'ambient_temp': 35.0,
        },
    })
    
    # 3. Prolonged Inactivity (Critical)
    sample_alerts.append({
        'alert_id': f"inactivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_003",
        'timestamp': datetime.now() - timedelta(hours=8),
        'cow_id': cow_ids[2],
        'alert_type': 'inactivity',
        'sensor_values': {
            'motion_magnitude': 0.03,
            'behavioral_state': 'lying',
            'rumination_minutes': 15,
        },
        'description': 'Prolonged inactivity detected - animal lying for over 8 hours',
        'metadata': {
            'duration_hours': 8.5,
            'rumination_percent': 35,
        },
    })
    
    # 4. Sensor Malfunction
    sample_alerts.append({
        'alert_id': f"sensor_malfunction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_004",
        'timestamp': datetime.now() - timedelta(minutes=45),
        'cow_id': cow_ids[3],
        'alert_type': 'sensor_malfunction',
        'sensor_values': {
            'affected_sensors': ['temp_sensor', 'accelerometer'],
            'last_reading_time': (datetime.now() - timedelta(minutes=45)).isoformat(),
        },
        'description': 'Sensor connectivity lost - no data received for 45 minutes',
        'metadata': {
            'gap_minutes': 45,
            'malfunction_type': 'connectivity_loss',
        },
    })
    
    # 5. Estrus Detection (Warning)
    sample_alerts.append({
        'alert_id': f"estrus_{datetime.now().strftime('%Y%m%d_%H%M%S')}_005",
        'timestamp': datetime.now() - timedelta(hours=3),
        'cow_id': cow_ids[4],
        'alert_type': 'estrus',
        'sensor_values': {
            'temperature': 38.9,
            'activity_increase': 0.45,
            'behavioral_state': 'walking',
        },
        'description': 'Estrus detected - optimal breeding window',
        'metadata': {
            'temp_rise': 0.4,
            'activity_increase_percent': 45,
            'last_estrus_days_ago': 21,
        },
    })
    
    # 6. Pregnancy Indication
    sample_alerts.append({
        'alert_id': f"pregnancy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_006",
        'timestamp': datetime.now() - timedelta(days=1),
        'cow_id': cow_ids[0],
        'alert_type': 'pregnancy_indication',
        'sensor_values': {
            'temperature_stability': 0.92,
            'no_return_to_estrus': True,
        },
        'description': 'Early pregnancy indication - no return to estrus after 21 days',
        'metadata': {
            'days_since_breeding': 25,
            'confidence': 0.82,
        },
    })
    
    # 7. Moderate Inactivity (Warning)
    sample_alerts.append({
        'alert_id': f"inactivity_{datetime.now().strftime('%Y%m%d_%H%M%S')}_007",
        'timestamp': datetime.now() - timedelta(hours=5),
        'cow_id': cow_ids[1],
        'alert_type': 'inactivity',
        'sensor_values': {
            'motion_magnitude': 0.08,
            'behavioral_state': 'lying',
            'rumination_minutes': 45,
        },
        'description': 'Moderate inactivity detected - monitoring required',
        'metadata': {
            'duration_hours': 5.2,
            'rumination_percent': 65,
        },
    })
    
    # 8. Sensor Reconnected (Info)
    sample_alerts.append({
        'alert_id': f"sensor_reconnected_{datetime.now().strftime('%Y%m%d_%H%M%S')}_008",
        'timestamp': datetime.now() - timedelta(minutes=15),
        'cow_id': cow_ids[3],
        'alert_type': 'sensor_reconnected',
        'sensor_values': {
            'connection_status': 'active',
            'signal_strength': 0.85,
        },
        'description': 'Sensor reconnected - data transmission resumed',
        'metadata': {
            'downtime_minutes': 45,
            'data_quality': 'good',
        },
    })
    
    # Create all alerts in the system
    for alert_data in sample_alerts:
        alert = alert_system.create_alert(**alert_data)
        print(f"Created alert: {alert.alert_type} ({alert.priority}) for cow {alert.cow_id}")
    
    # Acknowledge some alerts for demonstration
    acknowledged_ids = [
        sample_alerts[4]['alert_id'],  # Estrus
        sample_alerts[7]['alert_id'],  # Sensor reconnected
    ]
    
    for alert_id in acknowledged_ids:
        alert_system.acknowledge_alert(alert_id, acknowledged_by="farm_manager")
        print(f"Acknowledged alert: {alert_id}")
    
    # Resolve one alert
    alert_system.resolve_alert(
        sample_alerts[7]['alert_id'],
        resolution_notes="Sensor back online, data quality verified"
    )
    print(f"Resolved alert: {sample_alerts[7]['alert_id']}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Sample Alerts Created Successfully!")
    print("="*60)
    
    stats = alert_system.get_statistics()
    print(f"\nAlert Statistics:")
    print(f"  Total Alerts: {stats['total_alerts']}")
    print(f"  Active: {stats['active']}")
    print(f"  Acknowledged: {stats['acknowledged']}")
    print(f"  Resolved: {stats['resolved']}")
    print(f"\nBy Priority:")
    print(f"  Critical: {stats['by_priority']['critical']}")
    print(f"  Warning: {stats['by_priority']['warning']}")
    print(f"  Info: {stats['by_priority']['info']}")
    print(f"\nBy Type:")
    for alert_type, count in stats['by_type'].items():
        print(f"  {alert_type}: {count}")
    
    print(f"\nAlert state saved to: {alert_system.state_file}")
    print("\nYou can now view these alerts in the dashboard at:")
    print("  http://localhost:8501/4_Alerts_Dashboard")


if __name__ == "__main__":
    create_sample_alerts()
