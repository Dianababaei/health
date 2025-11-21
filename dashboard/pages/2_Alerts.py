"""
Enhanced Alerts Dashboard Page - Artemis Health Dashboard

Displays active alerts with notification panel, alert history,
and comprehensive alert management capabilities.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.health_intelligence.logging import AlertStateManager
from dashboard.components import (
    render_notification_panel,
    render_acknowledged_alerts_panel,
    render_alert_summary_metrics,
    render_severity_distribution,
    render_alert_history,
    render_search_alerts,
)

# Page configuration
st.set_page_config(
    page_title="Alerts Dashboard - Artemis Health",
    page_icon="🚨",
    layout="wide",
)

# Initialize state manager
@st.cache_resource
def get_state_manager():
    """Get cached AlertStateManager instance."""
    return AlertStateManager(db_path="data/alert_state.db")

state_manager = get_state_manager()

# Page header
st.title("🚨 Alerts Dashboard")
st.markdown("*Real-time alert monitoring and management*")
st.markdown("---")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["🔔 Active Alerts", "📜 Alert History", "🔍 Search"])

with tab1:
    # Active Alerts Tab
    st.markdown("## Active Alerts & Notifications")

    # Summary metrics
    render_alert_summary_metrics(state_manager)

    st.markdown("---")

    # Two columns: Active alerts and acknowledged alerts
    col1, col2 = st.columns([2, 1])

    with col1:
        # Main notification panel
        render_notification_panel(
            state_manager=state_manager,
            max_alerts=50,
            auto_refresh=False,
            refresh_interval=60
        )

    with col2:
        # Severity distribution
        render_severity_distribution(state_manager)

        st.markdown("---")

        # Recently acknowledged alerts
        render_acknowledged_alerts_panel(
            state_manager=state_manager,
            max_alerts=5
        )

with tab2:
    # Alert History Tab
    render_alert_history(
        state_manager=state_manager,
        default_days=7
    )

with tab3:
    # Search Tab
    st.markdown("## 🔍 Search Alerts")
    render_search_alerts(state_manager)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <p><small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    <p><small>Alert Dashboard v2.0 - Powered by Artemis Health Intelligence</small></p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### 📊 Quick Statistics")
    stats = state_manager.get_statistics()

    st.metric("Total Alerts", stats.get('total_alerts', 0))

    active_count = stats.get('by_status', {}).get('active', 0)
    st.metric("Active Alerts", active_count)

    resolved_count = stats.get('by_status', {}).get('resolved', 0)
    st.metric("Resolved", resolved_count)

    st.markdown("---")

    st.markdown("### ⚙️ System Status")
    st.success("✅ Alert Logging: Active")
    st.success("✅ State Tracking: Active")
    st.info("💾 Database: SQLite")

    st.markdown("---")

    if st.button("🔄 Refresh Page", use_container_width=True):
        st.rerun()

    if st.button("📥 Export All Data", use_container_width=True):
        all_alerts = state_manager.query_alerts(limit=1000)
        st.success(f"Exported {len(all_alerts)} alerts")
