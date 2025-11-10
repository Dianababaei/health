"""
Overview Page - Artemis Health Dashboard

Displays real-time metrics and system status overview.
This is a placeholder page that will be populated with detailed visualizations
by subsequent development tasks.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader

# Page configuration
st.set_page_config(
    page_title="Overview - Artemis Health",
    page_icon="📊",
    layout="wide",
)

# Initialize session state if needed
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Page header
st.title("📊 Overview Dashboard")
st.markdown("*Real-time livestock health monitoring and system status*")
st.markdown("---")

# Main content
with st.spinner("Loading overview data..."):
    try:
        data_loader = st.session_state.data_loader
        metrics = data_loader.get_latest_metrics()
        alert_summary = data_loader.get_alert_summary()
        
        # Key Metrics Row
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp = metrics.get('temperature')
            if temp:
                temp_status = metrics.get('temperature_status', 'normal')
                delta_color = "off" if temp_status == "normal" else "inverse"
                st.metric(
                    label="🌡️ Temperature",
                    value=f"{temp:.1f}°C",
                    delta=temp_status.upper() if temp_status != "normal" else "Normal",
                    delta_color=delta_color,
                )
            else:
                st.metric(label="🌡️ Temperature", value="N/A")
        
        with col2:
            activity = metrics.get('activity_level', 0)
            st.metric(
                label="📊 Activity Level",
                value=f"{activity:.2f}",
                delta=None,
            )
        
        with col3:
            current_state = metrics.get('current_state', 'unknown')
            st.metric(
                label="🐮 Current State",
                value=current_state.capitalize(),
                delta=None,
            )
        
        with col4:
            active_alerts = alert_summary.get('active_alerts', 0)
            delta_color = "off" if active_alerts == 0 else "inverse"
            st.metric(
                label="🚨 Active Alerts",
                value=str(active_alerts),
                delta=f"{alert_summary.get('total_alerts', 0)} total" if active_alerts > 0 else "None",
                delta_color=delta_color,
            )
        
        st.markdown("---")
        
        # Data Quality Section
        st.subheader("📊 Data Status")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            data_points = metrics.get('data_points', 0)
            if data_points > 0:
                st.success(f"✅ **{data_points}** data points loaded")
            else:
                st.warning("⚠️ No data available")
        
        with col2:
            last_update = metrics.get('timestamp')
            if last_update:
                st.info(f"🕐 Last update: **{last_update.strftime('%Y-%m-%d %H:%M:%S')}**")
            else:
                st.info("🕐 Last update: **N/A**")
        
        with col3:
            time_range = metrics.get('time_range', 'N/A')
            st.info(f"📅 Time range: **{time_range}**")
        
        st.markdown("---")
        
        # Recent Sensor Data Preview
        st.subheader("📋 Recent Sensor Data")
        
        sensor_data = data_loader.load_sensor_data(
            time_range_hours=1,
            max_rows=20
        )
        
        if not sensor_data.empty:
            st.dataframe(
                sensor_data,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.warning("⚠️ No sensor data available")
        
        st.markdown("---")
        
        # Alert Summary Section
        st.subheader("🚨 Alert Summary")
        
        if alert_summary.get('total_alerts', 0) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Alerts by Severity**")
                severity_data = alert_summary.get('by_severity', {})
                if severity_data:
                    for severity, count in severity_data.items():
                        st.markdown(f"- **{severity.capitalize()}**: {count}")
                else:
                    st.info("No severity data")
            
            with col2:
                st.markdown("**Alerts by Type**")
                type_data = alert_summary.get('by_type', {})
                if type_data:
                    for alert_type, count in type_data.items():
                        st.markdown(f"- **{alert_type}**: {count}")
                else:
                    st.info("No type data")
        else:
            st.success("✅ No alerts recorded")
        
    except Exception as e:
        st.error(f"❌ Error loading overview data: {str(e)}")
        st.info("💡 This page will be populated with detailed visualizations in subsequent tasks.")

# Placeholder notice
st.markdown("---")
st.info("""
**📝 Note**: This is a placeholder page. Detailed visualizations and interactive charts
will be added in subsequent development tasks as part of the "Build Real-Time Metrics Display Panel" subtask.
""")
