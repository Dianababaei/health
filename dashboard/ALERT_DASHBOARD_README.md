# Alert Dashboard & Notification Center

## Overview

The Alert Dashboard provides comprehensive alert monitoring and management for the Artemis Health system. It features priority-based color coding, interactive filtering and sorting, acknowledgment functionality, and actionable guidance for each alert type.

## Features

### ✨ Key Capabilities

- **Priority-Based Color Coding**: Visual indicators for Critical (🔴), Warning (🟠), and Info (🔵) alerts
- **Interactive Alert Cards**: Detailed view with sensor values, timestamps, and descriptions
- **Recommended Actions**: Context-specific guidance for each alert type
- **Filtering System**: Filter by priority, type, time range, status, and cow ID
- **Sorting Options**: Sort by time (newest/oldest), priority, or alert type
- **Acknowledgment**: Track who acknowledged alerts and when
- **State Management**: Persistent alert states (active/acknowledged/resolved)
- **Auto-Refresh**: Optional automatic data refresh at configurable intervals
- **Alert History**: Toggle between active and resolved alerts

### 📊 Alert Types & Priority Mapping

#### Critical (Red 🔴)
- **Fever**: Temperature >39.5°C with reduced motion
- **Heat Stress**: Temperature >40°C with high activity
- **Prolonged Inactivity**: >6 hours of continuous inactivity
- **Sensor Malfunction**: No data for >30 minutes

#### Warning (Yellow 🟠)
- **Estrus Detection**: Temperature rise with activity increase
- **Pregnancy Indication**: Long-term reproductive tracking
- **Moderate Inactivity**: 4-6 hours of reduced activity
- **Sensor Issues**: Minor connectivity or data quality problems

#### Info (Blue 🔵)
- **Sensor Reconnected**: Sensor back online
- **System Notifications**: Routine system messages

## Architecture

### Components

```
dashboard/
├── pages/
│   └── 4_Alerts_Dashboard.py      # Main alert dashboard page
├── components/
│   ├── __init__.py
│   └── alerts_panel.py             # Reusable alert panel component
└── utils/
    └── alert_integration.py        # Legacy alert integration

src/health_intelligence/
├── __init__.py
└── alert_system.py                 # Alert state management system
```

### Alert State Management

The `AlertSystem` class provides:
- Alert state tracking (active/acknowledged/resolved)
- Acknowledgment logging with user and timestamp
- Recommended actions mapping for each alert type
- Persistent storage in JSON format
- Query and filtering capabilities

### Alert Panel Component

The `AlertsPanel` class provides:
- Streamlit UI components for alert display
- Priority color scheme implementation
- Interactive filtering and sorting
- Acknowledgment UI with confirmation
- Auto-refresh logic

## Usage

### Basic Usage

```python
import streamlit as st
from dashboard.components import render_alerts_panel
from src.health_intelligence.alert_system import AlertSystem

# Initialize alert system
alert_system = AlertSystem()

# Render alerts panel
render_alerts_panel(
    alert_system=alert_system,
    show_resolved=False,
    auto_refresh_seconds=60,
)
```

### Creating Alerts Programmatically

```python
from datetime import datetime
from src.health_intelligence.alert_system import AlertSystem

# Initialize system
alert_system = AlertSystem()

# Create a new alert
alert = alert_system.create_alert(
    alert_id="fever_20250108_142300",
    timestamp=datetime.now(),
    cow_id="1042",
    alert_type="fever",
    sensor_values={
        "temperature": 40.2,
        "motion_magnitude": 0.12,
        "behavioral_state": "lying",
    },
    description="High fever detected with reduced activity",
    metadata={
        "baseline_temp": 38.5,
        "temp_deviation": 1.7,
        "duration_minutes": 6,
    },
)

# Alert is automatically assigned:
# - Priority (critical/warning/info)
# - Recommended actions
# - Active status
```

### Acknowledging Alerts

```python
# Acknowledge an alert
success = alert_system.acknowledge_alert(
    alert_id="fever_20250108_142300",
    acknowledged_by="farm_manager",
)

# Resolve an alert
success = alert_system.resolve_alert(
    alert_id="fever_20250108_142300",
    resolution_notes="Veterinary treatment administered",
)
```

### Querying Alerts

```python
# Get all active alerts
active_alerts = alert_system.get_active_alerts()

# Get active alerts for specific cow
cow_alerts = alert_system.get_active_alerts(cow_id="1042")

# Get alerts by status
acknowledged = alert_system.get_alerts_by_status("acknowledged")

# Get alerts by priority
critical_alerts = alert_system.get_alerts_by_priority("critical")

# Get statistics
stats = alert_system.get_statistics()
print(f"Active: {stats['active']}, Resolved: {stats['resolved']}")
```

## Recommended Actions

Each alert type has predefined recommended actions:

### Fever Alert
1. Immediately isolate the animal from the herd
2. Check rectal temperature with thermometer for confirmation
3. Contact veterinarian for examination
4. Ensure access to fresh water and shade
5. Monitor temperature every 2-4 hours
6. Document any additional symptoms

### Heat Stress Alert
1. Move animal to shaded area immediately
2. Provide cool, fresh water access
3. Use fans or misters if available
4. Reduce activity and handling
5. Monitor temperature every 30 minutes
6. If temperature exceeds 40.5°C, contact veterinarian urgently

### Inactivity Alert
1. Visually inspect the animal for signs of distress
2. Check for injuries, lameness, or bloating
3. Assess appetite and water intake
4. Monitor rumination activity
5. If lying for >8 hours, contact veterinarian (downer cow risk)
6. Provide comfortable bedding and check for environmental stressors

### Sensor Malfunction Alert
1. Check sensor battery level and connection
2. Inspect collar for damage or looseness
3. Verify sensor placement on neck
4. Reboot sensor device if possible
5. Replace sensor if malfunction persists
6. Document sensor ID and issue for maintenance tracking

### Estrus Alert
1. Observe for behavioral signs of estrus
2. Confirm with secondary indicators
3. Schedule breeding or AI within 12-18 hours
4. Record estrus detection for reproductive tracking
5. Monitor for successful breeding confirmation
6. Track for pregnancy indication in 21-28 days

### Pregnancy Indication Alert
1. Schedule veterinary confirmation
2. Monitor for return to estrus (negative indicator)
3. Adjust nutrition for early pregnancy needs
4. Continue monitoring for pregnancy maintenance
5. Record expected calving date
6. Implement pre-calving management protocols

## Integration with Legacy Systems

The `AlertIntegration` utility converts alerts from the legacy JSON log format:

```python
from dashboard.utils.alert_integration import sync_legacy_alerts

# Automatically import recent legacy alerts
alert_system = sync_legacy_alerts(auto_import=True)
```

## File Locations

- **Alert States**: `logs/alerts/alert_states.json`
- **Legacy Alerts**: `logs/malfunction_alerts.json` (from AlertGenerator)
- **Configuration**: Dashboard settings in session state

## Filtering & Sorting

### Available Filters
- **Priority**: All, Critical, Warning, Info
- **Alert Type**: Fever, Heat Stress, Inactivity, Sensor Malfunction, Estrus, Pregnancy
- **Time Range**: Last 24 Hours, Last 7 Days, Last 30 Days, All Time
- **Status**: Active Only, Acknowledged Only, All Unresolved, All Alerts
- **Cow ID**: Specific animal filter

### Sorting Options
- **Time (Newest First)**: Most recent alerts first
- **Time (Oldest First)**: Oldest alerts first
- **Priority (High to Low)**: Critical → Warning → Info
- **Type (A-Z)**: Alphabetical by alert type

## Auto-Refresh

Enable auto-refresh in the sidebar:
1. Check "Auto-Refresh" option
2. Set refresh interval (10-300 seconds)
3. Dashboard will automatically reload at specified interval

## Dashboard Settings

Accessible via sidebar:
- Show/hide resolved alerts
- Auto-refresh toggle and interval
- Alert priority guide reference

## Future Enhancements

Potential future features:
- Alert notification system (email/SMS)
- Alert correlation analysis
- Custom alert rule configuration
- Alert escalation workflows
- Time-series charts of alert frequency
- Export to PDF reports
- Mobile app notifications
- Multi-farm aggregation

## Support

For issues or questions:
1. Check logs at `logs/alerts/`
2. Verify alert state file exists
3. Ensure alert detection is running
4. Use "Retry" button on error page

---

**Version**: 2.0  
**Last Updated**: 2025-01-08  
**Component**: Alert Dashboard & Notification Center
