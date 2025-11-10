"""
Temperature Monitoring Page - Artemis Health Dashboard

Displays temperature trends, circadian rhythm analysis, and temperature alerts.
This is a placeholder page that will be populated with detailed visualizations
by subsequent development tasks.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader

# Page configuration
st.set_page_config(
    page_title="Temperature Monitoring - Artemis Health",
    page_icon="🌡️",
    layout="wide",
)

# Initialize session state if needed
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Page header
st.title("🌡️ Temperature Monitoring")
st.markdown("*Monitor body temperature trends and circadian rhythm patterns*")
st.markdown("---")

# Time range selector
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    time_range_options = {
        "Last 6 hours": 6,
        "Last 24 hours": 24,
        "Last 3 days": 72,
        "Last 7 days": 168,
        "Last 14 days": 336,
    }
    selected_label = st.selectbox(
        "Select Time Range",
        options=list(time_range_options.keys()),
        index=1,
    )
    selected_time_range = time_range_options[selected_label]

with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

st.markdown("---")

# Main content
with st.spinner("Loading temperature data..."):
    try:
        data_loader = st.session_state.data_loader
        sensor_data = data_loader.load_sensor_data(
            time_range_hours=selected_time_range
        )
        
        if not sensor_data.empty and 'temperature' in sensor_data.columns:
            # Temperature thresholds from config
            temp_config = st.session_state.config.get('metrics', {}).get('temperature', {})
            normal_min = temp_config.get('normal_min', 38.0)
            normal_max = temp_config.get('normal_max', 39.5)
            fever_threshold = temp_config.get('fever_threshold', 39.5)
            hypothermia_threshold = temp_config.get('hypothermia_threshold', 37.5)
            
            # Current Temperature Status
            st.subheader("🌡️ Current Temperature Status")
            
            latest_temp = sensor_data['temperature'].iloc[-1]
            avg_temp = sensor_data['temperature'].mean()
            min_temp = sensor_data['temperature'].min()
            max_temp = sensor_data['temperature'].max()
            std_temp = sensor_data['temperature'].std()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Current temperature with status
                if latest_temp >= fever_threshold:
                    st.metric("Current", f"{latest_temp:.2f}°C", delta="FEVER", delta_color="inverse")
                elif latest_temp <= hypothermia_threshold:
                    st.metric("Current", f"{latest_temp:.2f}°C", delta="LOW", delta_color="inverse")
                else:
                    st.metric("Current", f"{latest_temp:.2f}°C", delta="Normal")
            
            with col2:
                st.metric("Average", f"{avg_temp:.2f}°C")
            
            with col3:
                st.metric("Min / Max", f"{min_temp:.2f}°C / {max_temp:.2f}°C")
            
            with col4:
                st.metric("Std Dev", f"{std_temp:.3f}°C")
            
            st.markdown("---")
            
            # Temperature Range Analysis
            st.subheader("📊 Temperature Range Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Temperature Distribution**")
                
                # Count temperatures in different ranges
                normal_count = len(sensor_data[
                    (sensor_data['temperature'] >= normal_min) & 
                    (sensor_data['temperature'] <= normal_max)
                ])
                fever_count = len(sensor_data[sensor_data['temperature'] > fever_threshold])
                hypothermia_count = len(sensor_data[sensor_data['temperature'] < hypothermia_threshold])
                
                total_count = len(sensor_data)
                normal_pct = (normal_count / total_count) * 100
                fever_pct = (fever_count / total_count) * 100
                hypothermia_pct = (hypothermia_count / total_count) * 100
                
                st.markdown(f"- ✅ **Normal** ({normal_min}°C - {normal_max}°C): {normal_count} ({normal_pct:.1f}%)")
                st.markdown(f"- 🔥 **Fever** (>{fever_threshold}°C): {fever_count} ({fever_pct:.1f}%)")
                st.markdown(f"- ❄️ **Low** (<{hypothermia_threshold}°C): {hypothermia_count} ({hypothermia_pct:.1f}%)")
                
                if fever_count > 0:
                    st.warning(f"⚠️ Elevated temperature detected in {fever_pct:.1f}% of readings")
                elif hypothermia_count > 0:
                    st.warning(f"⚠️ Low temperature detected in {hypothermia_pct:.1f}% of readings")
                else:
                    st.success("✅ All temperatures within normal range")
            
            with col2:
                st.markdown("**Temperature Statistics**")
                st.info("📊 Histogram showing temperature distribution will be added here")
            
            st.markdown("---")
            
            # Temperature Trend
            st.subheader("📈 Temperature Trend Over Time")
            st.info("📈 Interactive line chart showing temperature over time with threshold bands will be added here")
            
            # Show basic trend info
            if 'timestamp' in sensor_data.columns:
                # Calculate hourly averages
                sensor_data_sorted = sensor_data.sort_values('timestamp')
                sensor_data_sorted['hour'] = pd.to_datetime(sensor_data_sorted['timestamp']).dt.hour
                
                hourly_avg = sensor_data_sorted.groupby('hour')['temperature'].mean()
                
                st.markdown("**Hourly Average Temperatures:**")
                col1, col2 = st.columns(2)
                
                hours_data = hourly_avg.to_dict()
                half_point = len(hours_data) // 2
                
                with col1:
                    for hour, temp in list(hours_data.items())[:half_point]:
                        st.markdown(f"- **{hour:02d}:00**: {temp:.2f}°C")
                
                with col2:
                    for hour, temp in list(hours_data.items())[half_point:]:
                        st.markdown(f"- **{hour:02d}:00**: {temp:.2f}°C")
            
            st.markdown("---")
            
            # Circadian Rhythm Analysis
            st.subheader("🌙 Circadian Rhythm Analysis")
            
            if selected_time_range >= 24:
                st.info("📊 24-hour temperature pattern visualization will be added here showing circadian rhythm")
                
                # Basic circadian analysis
                if 'timestamp' in sensor_data.columns:
                    sensor_data_sorted = sensor_data.sort_values('timestamp')
                    sensor_data_sorted['hour'] = pd.to_datetime(sensor_data_sorted['timestamp']).dt.hour
                    
                    hourly_stats = sensor_data_sorted.groupby('hour')['temperature'].agg(['mean', 'min', 'max'])
                    
                    peak_hour = hourly_stats['mean'].idxmax()
                    peak_temp = hourly_stats['mean'].max()
                    trough_hour = hourly_stats['mean'].idxmin()
                    trough_temp = hourly_stats['mean'].min()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Peak Temperature", f"{peak_temp:.2f}°C", f"at {peak_hour:02d}:00")
                    
                    with col2:
                        st.metric("Trough Temperature", f"{trough_temp:.2f}°C", f"at {trough_hour:02d}:00")
                    
                    with col3:
                        amplitude = peak_temp - trough_temp
                        st.metric("Circadian Amplitude", f"{amplitude:.2f}°C")
                    
                    # Health assessment
                    if amplitude >= 0.3 and amplitude <= 0.6:
                        st.success("✅ Normal circadian rhythm detected")
                    elif amplitude < 0.3:
                        st.warning("⚠️ Weak circadian rhythm - may indicate health issues")
                    else:
                        st.warning("⚠️ Strong circadian variation - monitor closely")
            else:
                st.info("ℹ️ Select a time range of at least 24 hours to view circadian rhythm analysis")
            
            st.markdown("---")
            
            # Temperature Alerts
            st.subheader("🚨 Temperature-Related Alerts")
            
            # Get alerts related to temperature
            alerts = data_loader.load_alerts(max_alerts=20)
            temp_alerts = [a for a in alerts if 'temperature' in a.get('malfunction_type', '').lower()]
            
            if temp_alerts:
                st.warning(f"⚠️ Found {len(temp_alerts)} temperature-related alerts")
                
                for alert in temp_alerts[:5]:
                    severity = alert.get('severity', 'unknown')
                    alert_type = alert.get('malfunction_type', 'Unknown')
                    detection_time = alert.get('detection_time', 'Unknown')
                    
                    with st.container():
                        if severity == 'critical':
                            st.error(f"**{alert_type}** - {detection_time}")
                        elif severity == 'high':
                            st.warning(f"**{alert_type}** - {detection_time}")
                        else:
                            st.info(f"**{alert_type}** - {detection_time}")
            else:
                st.success("✅ No temperature-related alerts")
            
            st.markdown("---")
            
            # Raw Data Table
            with st.expander("📋 View Raw Temperature Data"):
                temp_cols = ['timestamp', 'temperature'] if 'timestamp' in sensor_data.columns else ['temperature']
                if 'behavioral_state' in sensor_data.columns:
                    temp_cols.append('behavioral_state')
                
                st.dataframe(
                    sensor_data[temp_cols].tail(50),
                    use_container_width=True,
                    hide_index=True,
                )
        
        else:
            st.warning("⚠️ No temperature data available for the selected time range")
            st.info("💡 Please ensure sensor data files with temperature readings are available")
        
    except Exception as e:
        st.error(f"❌ Error loading temperature data: {str(e)}")

# Placeholder notice
st.markdown("---")
st.info("""
**📝 Note**: This is a placeholder page. Detailed visualizations including:
- Interactive temperature trend charts with threshold bands
- Circadian rhythm heatmaps
- Temperature correlation with activity
- Anomaly detection and alerts
- Multi-day temperature comparison

...will be added in subsequent development tasks.
""")
