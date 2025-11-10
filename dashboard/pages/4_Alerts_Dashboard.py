"""
Alerts Dashboard Page - Artemis Health Dashboard

Displays active alerts, alert history, and alert statistics.
This is a placeholder page that will be populated with detailed visualizations
by subsequent development tasks.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader

# Page configuration
st.set_page_config(
    page_title="Alerts Dashboard - Artemis Health",
    page_icon="🚨",
    layout="wide",
)

# Initialize session state if needed
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Page header
st.title("🚨 Alerts Dashboard")
st.markdown("*Monitor and manage system alerts and notifications*")
st.markdown("---")

# Controls
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    severity_filter = st.selectbox(
        "Filter by Severity",
        options=["All", "Critical", "High", "Medium", "Low"],
        index=0,
    )

with col2:
    max_alerts = st.number_input(
        "Max Alerts",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

with col4:
    if st.button("🗑️ Clear Filters", use_container_width=True):
        st.rerun()

st.markdown("---")

# Main content
with st.spinner("Loading alert data..."):
    try:
        data_loader = st.session_state.data_loader
        
        # Load alerts with filters
        filter_severity = severity_filter.lower() if severity_filter != "All" else None
        alerts = data_loader.load_alerts(
            max_alerts=max_alerts,
            filter_severity=filter_severity,
        )
        
        # Get alert summary
        alert_summary = data_loader.get_alert_summary()
        
        # Alert Summary Cards
        st.subheader("📊 Alert Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_alerts = alert_summary.get('total_alerts', 0)
            st.metric("Total Alerts", total_alerts)
        
        with col2:
            active_alerts = alert_summary.get('active_alerts', 0)
            st.metric("Active Alerts", active_alerts, delta=None)
        
        with col3:
            critical_count = alert_summary.get('by_severity', {}).get('critical', 0)
            st.metric("Critical", critical_count, delta=None)
        
        with col4:
            high_count = alert_summary.get('by_severity', {}).get('high', 0)
            st.metric("High Priority", high_count, delta=None)
        
        st.markdown("---")
        
        # Alerts by Severity and Type
        st.subheader("📈 Alert Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Alerts by Severity**")
            severity_data = alert_summary.get('by_severity', {})
            
            if severity_data:
                for severity, count in sorted(severity_data.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_alerts * 100) if total_alerts > 0 else 0
                    
                    # Color code by severity
                    if severity == 'critical':
                        st.error(f"🔴 **{severity.capitalize()}**: {count} ({percentage:.1f}%)")
                    elif severity == 'high':
                        st.warning(f"🟠 **{severity.capitalize()}**: {count} ({percentage:.1f}%)")
                    elif severity == 'medium':
                        st.info(f"🟡 **{severity.capitalize()}**: {count} ({percentage:.1f}%)")
                    else:
                        st.success(f"🟢 **{severity.capitalize()}**: {count} ({percentage:.1f}%)")
            else:
                st.info("No severity data available")
        
        with col2:
            st.markdown("**Alerts by Type**")
            type_data = alert_summary.get('by_type', {})
            
            if type_data:
                for alert_type, count in sorted(type_data.items(), key=lambda x: x[1], reverse=True)[:5]:
                    percentage = (count / total_alerts * 100) if total_alerts > 0 else 0
                    st.markdown(f"- **{alert_type}**: {count} ({percentage:.1f}%)")
                
                if len(type_data) > 5:
                    st.caption(f"...and {len(type_data) - 5} more types")
            else:
                st.info("No type data available")
        
        st.markdown("---")
        
        # Active Alerts Section
        st.subheader("🔴 Active Alerts")
        
        if alerts:
            # Filter for active alerts (last 24 hours)
            active_alert_list = []
            for alert in alerts:
                if 'detection_time' in alert:
                    try:
                        detection_time = pd.to_datetime(alert['detection_time'])
                        if datetime.now() - detection_time.to_pydatetime() < timedelta(hours=24):
                            active_alert_list.append(alert)
                    except:
                        pass
            
            if active_alert_list:
                st.warning(f"⚠️ {len(active_alert_list)} active alert(s) in the last 24 hours")
                
                for i, alert in enumerate(active_alert_list[:10]):
                    severity = alert.get('severity', 'unknown')
                    alert_type = alert.get('malfunction_type', 'Unknown')
                    detection_time = alert.get('detection_time', 'Unknown')
                    affected_sensors = alert.get('affected_sensors', [])
                    confidence = alert.get('confidence', 0)
                    
                    with st.expander(f"Alert #{i+1}: {alert_type} ({severity.upper()})"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**Type:** {alert_type}")
                            st.markdown(f"**Severity:** {severity.upper()}")
                            st.markdown(f"**Detection Time:** {detection_time}")
                        
                        with col2:
                            st.markdown(f"**Affected Sensors:** {', '.join(affected_sensors) if affected_sensors else 'N/A'}")
                            st.markdown(f"**Confidence:** {confidence:.2%}")
                        
                        # Show alert details if available
                        if 'message' in alert:
                            st.markdown(f"**Message:** {alert['message']}")
            else:
                st.success("✅ No active alerts in the last 24 hours")
        else:
            st.success("✅ No alerts recorded")
        
        st.markdown("---")
        
        # Alert History
        st.subheader("📜 Alert History")
        
        if alerts:
            st.info(f"Showing {len(alerts)} most recent alerts")
            
            # Create DataFrame for better display
            alert_df_data = []
            for alert in alerts:
                alert_df_data.append({
                    'Time': alert.get('detection_time', 'Unknown'),
                    'Type': alert.get('malfunction_type', 'Unknown'),
                    'Severity': alert.get('severity', 'unknown').upper(),
                    'Sensors': ', '.join(alert.get('affected_sensors', [])) if alert.get('affected_sensors') else 'N/A',
                    'Confidence': f"{alert.get('confidence', 0):.1%}",
                })
            
            alert_df = pd.DataFrame(alert_df_data)
            
            # Display as table
            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True,
            )
            
            # Download button for alert history
            csv = alert_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Alert History (CSV)",
                data=csv,
                file_name=f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No alert history available")
        
        st.markdown("---")
        
        # Alert Statistics
        st.subheader("📊 Alert Statistics")
        
        if alerts:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Most Common Alert Type**")
                if type_data:
                    most_common = max(type_data.items(), key=lambda x: x[1])
                    st.info(f"{most_common[0]}: {most_common[1]} occurrences")
                else:
                    st.info("N/A")
            
            with col2:
                st.markdown("**Average Confidence**")
                confidences = [a.get('confidence', 0) for a in alerts if 'confidence' in a]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    st.info(f"{avg_confidence:.1%}")
                else:
                    st.info("N/A")
            
            with col3:
                st.markdown("**Latest Alert**")
                if alerts:
                    latest = alerts[0]
                    latest_time = latest.get('detection_time', 'Unknown')
                    st.info(f"{latest_time}")
                else:
                    st.info("N/A")
            
            # Placeholder for charts
            st.info("📊 Time-series chart showing alert frequency over time will be added here")
        else:
            st.info("No statistics available - no alerts recorded")
        
    except Exception as e:
        st.error(f"❌ Error loading alert data: {str(e)}")
        st.info("💡 Please check that alert log files are available")

# Placeholder notice
st.markdown("---")
st.info("""
**📝 Note**: This is a placeholder page. Detailed features including:
- Real-time alert notifications
- Alert acknowledgment and dismissal
- Alert escalation workflows
- Time-series charts of alert frequency
- Alert correlation analysis
- Custom alert rule configuration

...will be added in subsequent development tasks.
""")
