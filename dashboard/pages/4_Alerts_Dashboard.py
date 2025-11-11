"""
Alerts Dashboard Page - Artemis Health Dashboard

Comprehensive alert monitoring and notification center with:
- Priority-based color coding
- Interactive filtering and sorting
- Alert acknowledgment functionality
- Recommended actions
- Real-time updates
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config
from dashboard.components.alerts_panel import render_alerts_panel
from src.health_intelligence.alert_system import AlertSystem

# Page configuration
st.set_page_config(
    page_title="Alerts Dashboard - Artemis Health",
    page_icon="🚨",
    layout="wide",
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()

# Page header
st.title("🚨 Alert Dashboard & Notification Center")
st.markdown("*Comprehensive alert monitoring with priority-based management and actionable guidance*")
st.markdown("---")

# Main alert panel
try:
    # Sidebar options
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")
        
        show_resolved = st.checkbox(
            "Show Resolved Alerts",
            value=False,
            help="Include resolved alerts in the display",
        )
        
        auto_refresh = st.checkbox(
            "Auto-Refresh",
            value=False,
            help="Automatically refresh alert data",
        )
        
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                step=10,
            )
        else:
            refresh_interval = None
        
        st.markdown("---")
        
        st.markdown("### 📖 Alert Priority Guide")
        st.markdown("""
        **🔴 CRITICAL:**
        - Fever (>39.5°C)
        - Heat Stress (>40°C)
        - Prolonged Inactivity (>6hrs)
        - Sensor Malfunction (>30min)
        
        **🟠 WARNING:**
        - Estrus Detection
        - Pregnancy Indication
        - Moderate Inactivity (4-6hrs)
        - Minor Sensor Issues
        
        **🔵 INFO:**
        - Sensor Reconnected
        - System Notifications
        - Routine Updates
        """)
    
    # Render main alerts panel
    render_alerts_panel(
        alert_system=st.session_state.alert_system,
        show_resolved=show_resolved,
        auto_refresh_seconds=refresh_interval if auto_refresh else None,
    )
    
    # Additional statistics section
    st.markdown("---")
    st.subheader("📈 System Statistics")
    
    stats = st.session_state.alert_system.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Alerts (All Time)", stats['total_alerts'])
    
    with col2:
        st.metric("Currently Active", stats['active'])
    
    with col3:
        st.metric("Acknowledged", stats['acknowledged'])
    
    with col4:
        st.metric("Resolved", stats['resolved'])
    
    # Alert type breakdown
    if stats['by_type']:
        st.markdown("### 📊 Alert Type Distribution")
        
        type_df = pd.DataFrame([
            {'Alert Type': k.replace('_', ' ').title(), 'Count': v}
            for k, v in stats['by_type'].items()
        ]).sort_values('Count', ascending=False)
        
        st.dataframe(type_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"❌ Error loading alert dashboard: {str(e)}")
    st.info("""
    **Troubleshooting Steps:**
    1. Verify alert logging system is configured
    2. Check alert state file exists (logs/alerts/alert_states.json)
    3. Ensure alert detection is running
    4. Refresh the page
    """)
    
    if st.button("🔄 Retry"):
        st.rerun()

# Footer information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 10px;'>
    <p><small>Alert Dashboard v2.0 - Real-time monitoring with state management</small></p>
    <p><small>Last updated: Auto-refresh enabled</small></p>
</div>
""", unsafe_allow_html=True)
