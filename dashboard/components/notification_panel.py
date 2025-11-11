"""
Alert Notification Panel Component

Real-time alert display with color-coded severity, expandable details,
and action buttons for alert management.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.health_intelligence.logging.alert_state_manager import AlertStateManager, AlertStatus


# Severity color mapping
SEVERITY_COLORS = {
    'critical': '#FF4444',  # Red
    'high': '#FF8C00',      # Dark Orange
    'warning': '#FFD700',   # Gold
    'medium': '#FFA500',    # Orange
    'info': '#4169E1',      # Royal Blue
    'low': '#90EE90'        # Light Green
}

SEVERITY_ICONS = {
    'critical': '🔴',
    'high': '🟠',
    'warning': '🟡',
    'medium': '🟡',
    'info': '🔵',
    'low': '🟢'
}


def render_notification_panel(
    state_manager: AlertStateManager,
    max_alerts: int = 10,
    auto_refresh: bool = True,
    refresh_interval: int = 30
):
    """
    Render the main notification panel with active alerts.
    
    Args:
        state_manager: AlertStateManager instance
        max_alerts: Maximum number of active alerts to display
        auto_refresh: Enable auto-refresh
        refresh_interval: Refresh interval in seconds
    """
    # Auto-refresh configuration
    if auto_refresh:
        st.markdown(
            f'<meta http-equiv="refresh" content="{refresh_interval}">',
            unsafe_allow_html=True
        )
    
    # Panel header
    st.markdown("### 🔔 Active Alerts")
    
    # Get active alerts
    active_alerts = state_manager.query_alerts(
        status=AlertStatus.ACTIVE.value,
        limit=max_alerts,
        sort_by="created_at",
        sort_order="DESC"
    )
    
    if not active_alerts:
        st.success("✅ No active alerts - All systems normal")
        return
    
    # Display alert count
    st.warning(f"⚠️ **{len(active_alerts)} Active Alert(s)**")
    
    # Sort by severity (critical first)
    severity_order = {'critical': 0, 'high': 1, 'warning': 2, 'medium': 3, 'info': 4, 'low': 5}
    sorted_alerts = sorted(
        active_alerts,
        key=lambda x: severity_order.get(x.get('severity', 'info'), 99)
    )
    
    # Render each alert
    for alert in sorted_alerts:
        render_alert_card(alert, state_manager)
    
    # Show refresh timestamp
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def render_alert_card(alert: Dict[str, Any], state_manager: AlertStateManager):
    """
    Render an individual alert card with expandable details.
    
    Args:
        alert: Alert data dictionary
        state_manager: AlertStateManager instance
    """
    alert_id = alert.get('alert_id', 'unknown')
    severity = alert.get('severity', 'info').lower()
    alert_type = alert.get('alert_type', 'Unknown')
    cow_id = alert.get('cow_id', 'unknown')
    confidence = alert.get('confidence', 0.0)
    created_at = alert.get('created_at', '')
    
    # Get severity icon and color
    icon = SEVERITY_ICONS.get(severity, '⚪')
    color = SEVERITY_COLORS.get(severity, '#808080')
    
    # Format created time
    try:
        created_dt = datetime.fromisoformat(created_at)
        time_ago = _format_time_ago(created_dt)
    except:
        time_ago = "Unknown time"
    
    # Create expandable alert card
    with st.expander(
        f"{icon} **{alert_type}** - Cow {cow_id} ({severity.upper()}) - {time_ago}",
        expanded=False
    ):
        # Create columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Alert details
            st.markdown(f"**Alert ID:** `{alert_id}`")
            st.markdown(f"**Cow ID:** {cow_id}")
            st.markdown(f"**Type:** {alert_type}")
            st.markdown(f"**Severity:** {severity.upper()}")
            st.markdown(f"**Confidence:** {confidence:.1%}")
            st.markdown(f"**Detected:** {created_at}")
            
            # Sensor values
            sensor_values = alert.get('sensor_values', {})
            if sensor_values:
                st.markdown("**Sensor Values:**")
                sensor_text = ", ".join([f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}" 
                                        for k, v in sensor_values.items()])
                st.text(sensor_text)
            
            # Detection details
            detection_details = alert.get('detection_details', {})
            if detection_details:
                st.markdown("**Detection Details:**")
                for key, value in detection_details.items():
                    st.text(f"  • {key}: {value}")
        
        with col2:
            # Action buttons
            st.markdown("**Actions:**")
            
            # Acknowledge button
            if st.button(
                "✓ Acknowledge",
                key=f"ack_{alert_id}",
                use_container_width=True
            ):
                if state_manager.acknowledge_alert(alert_id):
                    st.success("Alert acknowledged!")
                    st.rerun()
                else:
                    st.error("Failed to acknowledge alert")
            
            # Resolve button
            if st.button(
                "✓ Resolve",
                key=f"resolve_{alert_id}",
                use_container_width=True
            ):
                if state_manager.resolve_alert(alert_id):
                    st.success("Alert resolved!")
                    st.rerun()
                else:
                    st.error("Failed to resolve alert")
            
            # False positive button
            if st.button(
                "✗ False Positive",
                key=f"fp_{alert_id}",
                use_container_width=True
            ):
                if state_manager.mark_false_positive(alert_id):
                    st.warning("Marked as false positive")
                    st.rerun()
                else:
                    st.error("Failed to mark as false positive")
        
        # Add notes section
        st.markdown("---")
        notes = st.text_area(
            "Add Resolution Notes:",
            key=f"notes_{alert_id}",
            placeholder="Enter notes about this alert..."
        )
        
        if st.button("Save Notes", key=f"save_notes_{alert_id}"):
            # Update alert with notes (using resolve with notes)
            current_status = alert.get('status', AlertStatus.ACTIVE.value)
            if state_manager.update_status(alert_id, current_status, notes):
                st.success("Notes saved!")
            else:
                st.error("Failed to save notes")


def render_acknowledged_alerts_panel(
    state_manager: AlertStateManager,
    max_alerts: int = 5
):
    """
    Render panel showing recently acknowledged alerts.
    
    Args:
        state_manager: AlertStateManager instance
        max_alerts: Maximum number of alerts to display
    """
    st.markdown("### ⚠️ Acknowledged Alerts")
    
    # Get acknowledged alerts
    ack_alerts = state_manager.query_alerts(
        status=AlertStatus.ACKNOWLEDGED.value,
        limit=max_alerts,
        sort_by="updated_at",
        sort_order="DESC"
    )
    
    if not ack_alerts:
        st.info("No acknowledged alerts")
        return
    
    st.info(f"📋 {len(ack_alerts)} Acknowledged Alert(s)")
    
    # Display as simple list
    for alert in ack_alerts:
        severity = alert.get('severity', 'info')
        icon = SEVERITY_ICONS.get(severity, '⚪')
        alert_type = alert.get('alert_type', 'Unknown')
        cow_id = alert.get('cow_id', 'unknown')
        updated_at = alert.get('updated_at', '')
        
        with st.expander(f"{icon} {alert_type} - Cow {cow_id}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Alert ID:** `{alert.get('alert_id')}`")
                st.markdown(f"**Severity:** {severity.upper()}")
                st.markdown(f"**Acknowledged:** {updated_at}")
            
            with col2:
                # Resolve button
                if st.button(
                    "✓ Resolve",
                    key=f"resolve_ack_{alert.get('alert_id')}",
                    use_container_width=True
                ):
                    if state_manager.resolve_alert(alert.get('alert_id')):
                        st.success("Alert resolved!")
                        st.rerun()


def render_alert_summary_metrics(state_manager: AlertStateManager):
    """
    Render summary metrics for alerts.
    
    Args:
        state_manager: AlertStateManager instance
    """
    stats = state_manager.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = stats.get('total_alerts', 0)
        st.metric("Total Alerts", total)
    
    with col2:
        active = stats.get('by_status', {}).get(AlertStatus.ACTIVE.value, 0)
        st.metric("Active", active, delta=None)
    
    with col3:
        critical = stats.get('by_severity', {}).get('critical', 0)
        st.metric("Critical", critical, delta=None)
    
    with col4:
        avg_resolution = stats.get('avg_resolution_time_minutes', 0)
        st.metric("Avg Resolution", f"{avg_resolution:.1f} min")


def render_severity_distribution(state_manager: AlertStateManager):
    """
    Render severity distribution chart.
    
    Args:
        state_manager: AlertStateManager instance
    """
    stats = state_manager.get_statistics()
    severity_data = stats.get('by_severity', {})
    
    if not severity_data:
        st.info("No severity data available")
        return
    
    st.markdown("**Alert Distribution by Severity**")
    
    for severity, count in sorted(severity_data.items(), key=lambda x: x[1], reverse=True):
        icon = SEVERITY_ICONS.get(severity, '⚪')
        percentage = (count / sum(severity_data.values()) * 100) if severity_data else 0
        
        st.markdown(f"{icon} **{severity.upper()}**: {count} ({percentage:.1f}%)")


def _format_time_ago(dt: datetime) -> str:
    """
    Format datetime as relative time string.
    
    Args:
        dt: Datetime object
        
    Returns:
        Formatted string (e.g., "5 minutes ago")
    """
    now = datetime.now()
    diff = now - dt
    
    if diff < timedelta(minutes=1):
        return "Just now"
    elif diff < timedelta(hours=1):
        minutes = int(diff.total_seconds() / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    else:
        days = diff.days
        return f"{days} day{'s' if days != 1 else ''} ago"
