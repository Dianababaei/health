"""
Alert Panel Component - Artemis Health Dashboard

Provides comprehensive alert dashboard UI with:
- Priority-based color coding
- Alert cards with details and recommended actions
- Filtering by type, severity, time range
- Sorting by time, severity, alert type
- Acknowledgment functionality
- Auto-refresh capability
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.health_intelligence.alert_system import AlertSystem, AlertState, AlertPriority


class AlertsPanel:
    """
    Alert panel UI component for Streamlit dashboard.
    
    Features:
    - Color-coded priority indicators
    - Alert cards with full details
    - Interactive filtering and sorting
    - Acknowledgment with confirmation
    - Auto-refresh support
    - Empty state handling
    """
    
    # Priority color scheme
    PRIORITY_COLORS = {
        'critical': {
            'color': '#FF4444',
            'background': '#FFE8E8',
            'icon': '🔴',
            'label': 'CRITICAL',
        },
        'warning': {
            'color': '#FFA500',
            'background': '#FFF4E6',
            'icon': '🟠',
            'label': 'WARNING',
        },
        'info': {
            'color': '#4A90E2',
            'background': '#E8F4FF',
            'icon': '🔵',
            'label': 'INFO',
        },
    }
    
    # Alert type icons
    ALERT_TYPE_ICONS = {
        'fever': '🌡️',
        'heat_stress': '🔥',
        'inactivity': '😴',
        'sensor_malfunction': '⚠️',
        'estrus': '💕',
        'pregnancy_indication': '🤰',
        'sensor_reconnected': '✅',
        'default': '📢',
    }
    
    def __init__(self, alert_system: AlertSystem):
        """
        Initialize alerts panel.
        
        Args:
            alert_system: AlertSystem instance for state management
        """
        self.alert_system = alert_system
    
    def render(
        self,
        show_resolved: bool = False,
        auto_refresh_seconds: Optional[int] = None,
    ):
        """
        Render the complete alerts panel.
        
        Args:
            show_resolved: Include resolved alerts in display
            auto_refresh_seconds: Auto-refresh interval (None to disable)
        """
        # Auto-refresh logic
        if auto_refresh_seconds:
            st.empty()  # Placeholder for auto-refresh
        
        # Render controls
        filters = self._render_controls(show_resolved)
        
        st.markdown("---")
        
        # Get and filter alerts
        alerts = self._get_filtered_alerts(filters, show_resolved)
        
        # Render summary metrics
        self._render_summary_metrics(alerts)
        
        st.markdown("---")
        
        # Render alerts
        if alerts:
            self._render_alerts(alerts, filters['sort_by'])
        else:
            self._render_empty_state()
    
    def _render_controls(self, show_resolved: bool) -> Dict[str, Any]:
        """
        Render filter and control UI.
        
        Returns:
            Dictionary with filter selections
        """
        st.subheader("🎛️ Alert Controls")
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            priority_filter = st.selectbox(
                "Filter by Priority",
                options=["All", "Critical", "Warning", "Info"],
                index=0,
                key="alert_priority_filter",
            )
        
        with col2:
            alert_type_filter = st.selectbox(
                "Filter by Type",
                options=[
                    "All",
                    "Fever",
                    "Heat Stress",
                    "Inactivity",
                    "Sensor Malfunction",
                    "Estrus",
                    "Pregnancy Indication",
                ],
                index=0,
                key="alert_type_filter",
            )
        
        with col3:
            time_filter = st.selectbox(
                "Time Range",
                options=["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
                index=0,
                key="alert_time_filter",
            )
        
        with col4:
            if st.button("🔄 Refresh", use_container_width=True, key="alert_refresh"):
                st.rerun()
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            sort_by = st.selectbox(
                "Sort By",
                options=["Time (Newest First)", "Time (Oldest First)", "Priority (High to Low)", "Type (A-Z)"],
                index=0,
                key="alert_sort_by",
            )
        
        with col2:
            status_filter = st.selectbox(
                "Status",
                options=["Active Only", "Acknowledged Only", "All Unresolved", "All Alerts"],
                index=0 if not show_resolved else 3,
                key="alert_status_filter",
            )
        
        with col3:
            cow_filter = st.text_input(
                "Cow ID (optional)",
                placeholder="e.g., 1042",
                key="alert_cow_filter",
            )
        
        with col4:
            if st.button("🗑️ Clear", use_container_width=True, key="alert_clear_filters"):
                st.rerun()
        
        return {
            'priority': priority_filter.lower() if priority_filter != "All" else None,
            'alert_type': self._map_alert_type(alert_type_filter) if alert_type_filter != "All" else None,
            'time_range': time_filter,
            'sort_by': sort_by,
            'status': status_filter,
            'cow_id': cow_filter if cow_filter else None,
        }
    
    def _map_alert_type(self, display_name: str) -> str:
        """Map display name to alert type key."""
        mapping = {
            "Fever": "fever",
            "Heat Stress": "heat_stress",
            "Inactivity": "inactivity",
            "Sensor Malfunction": "sensor_malfunction",
            "Estrus": "estrus",
            "Pregnancy Indication": "pregnancy_indication",
        }
        return mapping.get(display_name, display_name.lower().replace(" ", "_"))
    
    def _get_filtered_alerts(
        self,
        filters: Dict[str, Any],
        show_resolved: bool,
    ) -> List[AlertState]:
        """
        Get alerts based on filters.
        
        Args:
            filters: Filter criteria
            show_resolved: Include resolved alerts
            
        Returns:
            List of filtered alerts
        """
        # Get base set of alerts
        if filters['status'] == "Active Only":
            alerts = self.alert_system.get_active_alerts(cow_id=filters['cow_id'])
        elif filters['status'] == "Acknowledged Only":
            alerts = self.alert_system.get_alerts_by_status(
                'acknowledged',
                cow_id=filters['cow_id']
            )
        elif filters['status'] == "All Unresolved":
            active = self.alert_system.get_alerts_by_status('active', cow_id=filters['cow_id'])
            acknowledged = self.alert_system.get_alerts_by_status('acknowledged', cow_id=filters['cow_id'])
            alerts = active + acknowledged
        else:  # All Alerts
            alerts = list(self.alert_system.alerts.values())
            if filters['cow_id']:
                alerts = [a for a in alerts if a.cow_id == filters['cow_id']]
        
        # Filter by priority
        if filters['priority']:
            alerts = [a for a in alerts if a.priority == filters['priority']]
        
        # Filter by alert type
        if filters['alert_type']:
            alerts = [a for a in alerts if a.alert_type == filters['alert_type']]
        
        # Filter by time range
        alerts = self._filter_by_time_range(alerts, filters['time_range'])
        
        # Sort alerts
        alerts = self._sort_alerts(alerts, filters['sort_by'])
        
        return alerts
    
    def _filter_by_time_range(
        self,
        alerts: List[AlertState],
        time_range: str,
    ) -> List[AlertState]:
        """Filter alerts by time range."""
        if time_range == "All Time":
            return alerts
        
        now = datetime.now()
        
        if time_range == "Last 24 Hours":
            cutoff = now - timedelta(hours=24)
        elif time_range == "Last 7 Days":
            cutoff = now - timedelta(days=7)
        elif time_range == "Last 30 Days":
            cutoff = now - timedelta(days=30)
        else:
            return alerts
        
        return [a for a in alerts if a.timestamp >= cutoff]
    
    def _sort_alerts(
        self,
        alerts: List[AlertState],
        sort_by: str,
    ) -> List[AlertState]:
        """Sort alerts based on criteria."""
        if sort_by == "Time (Newest First)":
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
        elif sort_by == "Time (Oldest First)":
            return sorted(alerts, key=lambda x: x.timestamp)
        elif sort_by == "Priority (High to Low)":
            priority_order = {'critical': 0, 'warning': 1, 'info': 2}
            return sorted(alerts, key=lambda x: (priority_order.get(x.priority, 3), x.timestamp), reverse=True)
        elif sort_by == "Type (A-Z)":
            return sorted(alerts, key=lambda x: (x.alert_type, x.timestamp), reverse=True)
        else:
            return alerts
    
    def _render_summary_metrics(self, alerts: List[AlertState]):
        """Render summary metrics for filtered alerts."""
        st.subheader("📊 Alert Summary")
        
        # Calculate metrics
        critical_count = len([a for a in alerts if a.priority == 'critical' and a.status == 'active'])
        warning_count = len([a for a in alerts if a.priority == 'warning' and a.status == 'active'])
        info_count = len([a for a in alerts if a.priority == 'info' and a.status == 'active'])
        acknowledged_count = len([a for a in alerts if a.status == 'acknowledged'])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Alerts", len(alerts))
        
        with col2:
            st.metric("🔴 Critical", critical_count)
        
        with col3:
            st.metric("🟠 Warning", warning_count)
        
        with col4:
            st.metric("🔵 Info", info_count)
        
        with col5:
            st.metric("✓ Acknowledged", acknowledged_count)
    
    def _render_alerts(self, alerts: List[AlertState], sort_by: str):
        """Render list of alert cards."""
        st.subheader(f"📋 Alerts ({len(alerts)} found)")
        
        for alert in alerts:
            self._render_alert_card(alert)
    
    def _render_alert_card(self, alert: AlertState):
        """
        Render individual alert card.
        
        Args:
            alert: AlertState object to render
        """
        priority_config = self.PRIORITY_COLORS[alert.priority]
        alert_icon = self.ALERT_TYPE_ICONS.get(alert.alert_type, self.ALERT_TYPE_ICONS['default'])
        
        # Create card with custom styling
        card_style = f"""
        <div style="
            background-color: {priority_config['background']};
            border-left: 5px solid {priority_config['color']};
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
        ">
        </div>
        """
        
        with st.container():
            st.markdown(card_style, unsafe_allow_html=True)
            
            # Header row
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(
                    f"### {alert_icon} {alert.alert_type.replace('_', ' ').title()}"
                )
                st.caption(f"Alert ID: {alert.alert_id} | Cow: {alert.cow_id}")
            
            with col2:
                st.markdown(
                    f"**{priority_config['icon']} {priority_config['label']}**",
                )
                st.caption(f"Status: {alert.status.upper()}")
            
            with col3:
                # Acknowledgment button
                if alert.status == 'active':
                    if st.button(
                        "✓ Acknowledge",
                        key=f"ack_{alert.alert_id}",
                        use_container_width=True,
                    ):
                        self._acknowledge_alert_with_confirmation(alert)
                elif alert.status == 'acknowledged':
                    st.success("✓ Acknowledged")
            
            # Timestamp
            st.markdown(f"**⏰ Detection Time:** {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if alert.acknowledged_at:
                st.markdown(
                    f"**✓ Acknowledged:** {alert.acknowledged_at.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"by {alert.acknowledged_by}"
                )
            
            # Description
            st.markdown(f"**📝 Description:** {alert.description}")
            
            # Sensor values
            with st.expander("📊 Sensor Values"):
                sensor_col1, sensor_col2 = st.columns(2)
                
                for i, (key, value) in enumerate(alert.sensor_values.items()):
                    with sensor_col1 if i % 2 == 0 else sensor_col2:
                        if isinstance(value, (int, float)):
                            st.metric(key.upper(), f"{value:.2f}" if isinstance(value, float) else value)
                        else:
                            st.markdown(f"**{key.upper()}:** {value}")
            
            # Recommended actions
            with st.expander("💡 Recommended Actions", expanded=alert.status == 'active'):
                for i, action in enumerate(alert.recommended_actions, 1):
                    st.markdown(f"{i}. {action}")
            
            # Metadata
            if alert.metadata:
                with st.expander("🔍 Additional Details"):
                    for key, value in alert.metadata.items():
                        st.markdown(f"**{key.replace('_', ' ').title()}:** {value}")
            
            st.markdown("---")
    
    def _acknowledge_alert_with_confirmation(self, alert: AlertState):
        """
        Acknowledge alert with confirmation dialog.
        
        Args:
            alert: AlertState to acknowledge
        """
        # Use session state for confirmation
        confirmation_key = f"confirm_ack_{alert.alert_id}"
        
        if confirmation_key not in st.session_state:
            st.session_state[confirmation_key] = False
        
        if not st.session_state[confirmation_key]:
            st.session_state[confirmation_key] = True
            st.rerun()
        else:
            # Perform acknowledgment
            success = self.alert_system.acknowledge_alert(alert.alert_id)
            
            if success:
                st.success(f"✅ Alert {alert.alert_id} acknowledged successfully!")
                # Reset confirmation state
                del st.session_state[confirmation_key]
                st.rerun()
            else:
                st.error(f"❌ Failed to acknowledge alert {alert.alert_id}")
    
    def _render_empty_state(self):
        """Render empty state when no alerts match filters."""
        st.info("""
        ### ✅ No Alerts Found
        
        There are currently no alerts matching your filter criteria.
        
        This could mean:
        - All animals are healthy
        - Alerts have been resolved
        - Your filters are too restrictive
        
        Try adjusting your filters or refreshing the data.
        """)
    
    def render_compact_summary(self):
        """Render a compact alert summary for overview dashboards."""
        stats = self.alert_system.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🔴 Critical",
                stats['by_priority']['critical'],
                delta=None,
            )
        
        with col2:
            st.metric(
                "🟠 Warning",
                stats['by_priority']['warning'],
                delta=None,
            )
        
        with col3:
            st.metric(
                "🔵 Info",
                stats['by_priority']['info'],
                delta=None,
            )
        
        with col4:
            st.metric(
                "📊 Active Total",
                stats['active'],
                delta=None,
            )
        
        # Quick link to full dashboard
        if stats['active'] > 0:
            st.markdown("[🔗 View Full Alert Dashboard →](4_Alerts_Dashboard)")


def render_alerts_panel(
    alert_system: Optional[AlertSystem] = None,
    show_resolved: bool = False,
    auto_refresh_seconds: Optional[int] = 60,
) -> AlertsPanel:
    """
    Convenience function to render alerts panel.
    
    Args:
        alert_system: AlertSystem instance (creates new if None)
        show_resolved: Include resolved alerts
        auto_refresh_seconds: Auto-refresh interval
        
    Returns:
        AlertsPanel instance
    """
    if alert_system is None:
        alert_system = AlertSystem()
    
    panel = AlertsPanel(alert_system)
    panel.render(show_resolved=show_resolved, auto_refresh_seconds=auto_refresh_seconds)
    
    return panel
