"""
Alert History Component

Provides comprehensive alert history viewing with filtering, search,
and analytics capabilities.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.health_intelligence.logging.alert_state_manager import AlertStateManager, AlertStatus


def render_alert_history(
    state_manager: AlertStateManager,
    default_days: int = 7
):
    """
    Render comprehensive alert history with filters and search.
    
    Args:
        state_manager: AlertStateManager instance
        default_days: Default number of days to show
    """
    st.markdown("## 📜 Alert History")
    
    # Filters section
    with st.expander("🔍 Filters & Search", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Date range filter
            date_option = st.selectbox(
                "Time Range",
                options=["Last 24 hours", "Last 7 days", "Last 30 days", "Custom"],
                index=1
            )
            
            if date_option == "Custom":
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date")
            else:
                end_date = datetime.now()
                if date_option == "Last 24 hours":
                    start_date = end_date - timedelta(days=1)
                elif date_option == "Last 7 days":
                    start_date = end_date - timedelta(days=7)
                else:  # Last 30 days
                    start_date = end_date - timedelta(days=30)
        
        with col2:
            # Cow ID filter
            cow_id_filter = st.text_input(
                "Cow ID",
                placeholder="e.g., COW001"
            )
        
        with col3:
            # Status filter
            status_filter = st.selectbox(
                "Status",
                options=["All", "Active", "Acknowledged", "Resolved", "False Positive"],
                index=0
            )
        
        with col4:
            # Severity filter
            severity_filter = st.selectbox(
                "Severity",
                options=["All", "Critical", "High", "Warning", "Medium", "Info", "Low"],
                index=0
            )
        
        # Alert type filter and search
        col5, col6 = st.columns(2)
        
        with col5:
            alert_type_filter = st.text_input(
                "Alert Type",
                placeholder="e.g., fever, heat_stress"
            )
        
        with col6:
            max_results = st.number_input(
                "Max Results",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
    
    # Build query parameters
    query_params = {
        'limit': max_results,
        'sort_by': 'created_at',
        'sort_order': 'DESC'
    }
    
    if cow_id_filter:
        query_params['cow_id'] = cow_id_filter.strip()
    
    if status_filter != "All":
        query_params['status'] = status_filter.lower()
    
    if severity_filter != "All":
        query_params['severity'] = severity_filter.lower()
    
    if alert_type_filter:
        query_params['alert_type'] = alert_type_filter.strip()
    
    # Date filters
    if isinstance(start_date, datetime):
        query_params['start_date'] = start_date.isoformat()
    else:
        query_params['start_date'] = datetime.combine(start_date, datetime.min.time()).isoformat()
    
    if isinstance(end_date, datetime):
        query_params['end_date'] = end_date.isoformat()
    else:
        query_params['end_date'] = datetime.combine(end_date, datetime.max.time()).isoformat()
    
    # Query alerts
    alerts = state_manager.query_alerts(**query_params)
    
    # Display results count
    st.markdown(f"**Found {len(alerts)} alert(s)**")
    
    if not alerts:
        st.info("No alerts found matching the filters")
        return
    
    # Display alerts in different formats
    tab1, tab2, tab3 = st.tabs(["📋 Table View", "📊 Details View", "📈 Analytics"])
    
    with tab1:
        render_alerts_table(alerts)
    
    with tab2:
        render_alerts_detailed(alerts, state_manager)
    
    with tab3:
        render_alerts_analytics(alerts, state_manager)


def render_alerts_table(alerts: List[Dict[str, Any]]):
    """
    Render alerts in table format.
    
    Args:
        alerts: List of alert dictionaries
    """
    if not alerts:
        st.info("No alerts to display")
        return
    
    # Convert to DataFrame
    table_data = []
    for alert in alerts:
        table_data.append({
            'Timestamp': alert.get('created_at', ''),
            'Cow ID': alert.get('cow_id', ''),
            'Type': alert.get('alert_type', ''),
            'Severity': alert.get('severity', '').upper(),
            'Status': alert.get('status', '').upper(),
            'Confidence': f"{alert.get('confidence', 0):.1%}",
            'Alert ID': alert.get('alert_id', '')
        })
    
    df = pd.DataFrame(table_data)
    
    # Display with color coding
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )


def render_alerts_detailed(alerts: List[Dict[str, Any]], state_manager: AlertStateManager):
    """
    Render alerts with full details in expandable cards.
    
    Args:
        alerts: List of alert dictionaries
        state_manager: AlertStateManager instance
    """
    if not alerts:
        st.info("No alerts to display")
        return
    
    for i, alert in enumerate(alerts):
        alert_id = alert.get('alert_id', 'unknown')
        severity = alert.get('severity', 'info')
        alert_type = alert.get('alert_type', 'Unknown')
        cow_id = alert.get('cow_id', 'unknown')
        status = alert.get('status', 'unknown')
        
        # Status icon
        status_icons = {
            'active': '🔴',
            'acknowledged': '🟡',
            'resolved': '🟢',
            'false_positive': '⚪'
        }
        status_icon = status_icons.get(status, '⚪')
        
        with st.expander(
            f"{status_icon} {i+1}. {alert_type} - Cow {cow_id} ({severity.upper()}) - {status.upper()}",
            expanded=False
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Basic info
                st.markdown(f"**Alert ID:** `{alert_id}`")
                st.markdown(f"**Cow ID:** {cow_id}")
                st.markdown(f"**Type:** {alert_type}")
                st.markdown(f"**Severity:** {severity.upper()}")
                st.markdown(f"**Status:** {status.upper()}")
                st.markdown(f"**Confidence:** {alert.get('confidence', 0):.1%}")
                
                # Timestamps
                st.markdown("---")
                st.markdown(f"**Created:** {alert.get('created_at', 'N/A')}")
                st.markdown(f"**Updated:** {alert.get('updated_at', 'N/A')}")
                
                # Sensor values
                sensor_values = alert.get('sensor_values', {})
                if sensor_values:
                    st.markdown("---")
                    st.markdown("**Sensor Values:**")
                    for key, value in sensor_values.items():
                        if isinstance(value, float):
                            st.text(f"  • {key}: {value:.2f}")
                        else:
                            st.text(f"  • {key}: {value}")
                
                # Detection details
                detection_details = alert.get('detection_details', {})
                if detection_details:
                    st.markdown("---")
                    st.markdown("**Detection Details:**")
                    for key, value in detection_details.items():
                        st.text(f"  • {key}: {value}")
                
                # Resolution notes
                notes = alert.get('resolution_notes', '')
                if notes:
                    st.markdown("---")
                    st.markdown("**Resolution Notes:**")
                    st.text(notes)
            
            with col2:
                # State history
                st.markdown("**State History:**")
                history = state_manager.get_state_history(alert_id)
                
                if history:
                    for entry in history:
                        old_status = entry.get('old_status', '')
                        new_status = entry.get('new_status', '')
                        changed_at = entry.get('changed_at', '')
                        
                        if old_status:
                            st.caption(f"{old_status} → {new_status}")
                        else:
                            st.caption(f"Created: {new_status}")
                        st.caption(f"  {changed_at}")
                else:
                    st.caption("No state history")


def render_alerts_analytics(alerts: List[Dict[str, Any]], state_manager: AlertStateManager):
    """
    Render analytics and statistics for alerts.
    
    Args:
        alerts: List of alert dictionaries
        state_manager: AlertStateManager instance
    """
    if not alerts:
        st.info("No alerts to analyze")
        return
    
    # Overall statistics
    st.markdown("### 📊 Alert Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts", len(alerts))
    
    with col2:
        active_count = sum(1 for a in alerts if a.get('status') == 'active')
        st.metric("Active", active_count)
    
    with col3:
        resolved_count = sum(1 for a in alerts if a.get('status') == 'resolved')
        st.metric("Resolved", resolved_count)
    
    with col4:
        fp_count = sum(1 for a in alerts if a.get('status') == 'false_positive')
        fp_rate = (fp_count / len(alerts) * 100) if alerts else 0
        st.metric("False Positive Rate", f"{fp_rate:.1f}%")
    
    st.markdown("---")
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Severity Distribution")
        severity_counts = {}
        for alert in alerts:
            severity = alert.get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            df_severity = pd.DataFrame([
                {'Severity': k.upper(), 'Count': v}
                for k, v in sorted(severity_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(df_severity, use_container_width=True, hide_index=True)
        else:
            st.info("No severity data")
    
    with col2:
        st.markdown("### 📈 Type Distribution")
        type_counts = {}
        for alert in alerts:
            alert_type = alert.get('alert_type', 'unknown')
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        if type_counts:
            df_types = pd.DataFrame([
                {'Alert Type': k, 'Count': v}
                for k, v in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            ])
            st.dataframe(df_types, use_container_width=True, hide_index=True)
        else:
            st.info("No type data")
    
    st.markdown("---")
    
    # Timeline analysis
    st.markdown("### 📅 Alert Frequency Over Time")
    
    # Group by date
    date_counts = {}
    for alert in alerts:
        try:
            created_at = alert.get('created_at', '')
            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            date_key = dt.strftime('%Y-%m-%d')
            date_counts[date_key] = date_counts.get(date_key, 0) + 1
        except:
            pass
    
    if date_counts:
        df_timeline = pd.DataFrame([
            {'Date': k, 'Alerts': v}
            for k, v in sorted(date_counts.items())
        ])
        st.line_chart(df_timeline.set_index('Date'))
    else:
        st.info("No timeline data available")
    
    st.markdown("---")
    
    # Resolution time analysis
    st.markdown("### ⏱️ Resolution Time Analysis")
    
    resolution_times = []
    for alert in alerts:
        if alert.get('status') == 'resolved':
            try:
                created = datetime.fromisoformat(alert.get('created_at', '').replace('Z', '+00:00'))
                updated = datetime.fromisoformat(alert.get('updated_at', '').replace('Z', '+00:00'))
                duration = (updated - created).total_seconds() / 60  # minutes
                resolution_times.append(duration)
            except:
                pass
    
    if resolution_times:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_time = sum(resolution_times) / len(resolution_times)
            st.metric("Average Resolution Time", f"{avg_time:.1f} min")
        
        with col2:
            min_time = min(resolution_times)
            st.metric("Fastest Resolution", f"{min_time:.1f} min")
        
        with col3:
            max_time = max(resolution_times)
            st.metric("Slowest Resolution", f"{max_time:.1f} min")
    else:
        st.info("No resolution time data available")
    
    st.markdown("---")
    
    # Cow-level analysis
    st.markdown("### 🐄 Alerts by Cow")
    
    cow_counts = {}
    for alert in alerts:
        cow_id = alert.get('cow_id', 'unknown')
        cow_counts[cow_id] = cow_counts.get(cow_id, 0) + 1
    
    if cow_counts:
        df_cows = pd.DataFrame([
            {'Cow ID': k, 'Alert Count': v}
            for k, v in sorted(cow_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        ])
        st.dataframe(df_cows, use_container_width=True, hide_index=True)
        
        # Highlight high-alert cows
        high_alert_threshold = 5
        high_alert_cows = [k for k, v in cow_counts.items() if v >= high_alert_threshold]
        if high_alert_cows:
            st.warning(f"⚠️ {len(high_alert_cows)} cow(s) with {high_alert_threshold}+ alerts")
    else:
        st.info("No cow data available")


def render_search_alerts(state_manager: AlertStateManager):
    """
    Render alert search interface.
    
    Args:
        state_manager: AlertStateManager instance
    """
    st.markdown("### 🔍 Search Alerts")
    
    search_term = st.text_input(
        "Search by Alert ID",
        placeholder="Enter full alert ID or UUID"
    )
    
    if search_term:
        alert = state_manager.get_alert(search_term.strip())
        
        if alert:
            st.success("✅ Alert found!")
            
            # Display alert details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.json(alert)
            
            with col2:
                st.markdown("**State History:**")
                history = state_manager.get_state_history(search_term.strip())
                for entry in history:
                    st.caption(f"{entry.get('old_status', 'New')} → {entry.get('new_status')}")
                    st.caption(f"  {entry.get('changed_at')}")
        else:
            st.error("❌ Alert not found")
