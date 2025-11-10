"""
Health Trends Page - Artemis Health Dashboard

Displays multi-day health analysis, health scores, and long-term trends.
This is a placeholder page that will be populated with detailed visualizations
by subsequent development tasks.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader

# Page configuration
st.set_page_config(
    page_title="Health Trends - Artemis Health",
    page_icon="📈",
    layout="wide",
)

# Initialize session state if needed
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Page header
st.title("📈 Health Trends Analysis")
st.markdown("*Long-term health monitoring and trend analysis*")
st.markdown("---")

# Time range selector (multi-day analysis)
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    time_range_options = {
        "Last 3 days": 72,
        "Last 7 days": 168,
        "Last 14 days": 336,
        "Last 30 days": 720,
    }
    selected_label = st.selectbox(
        "Select Analysis Period",
        options=list(time_range_options.keys()),
        index=1,
    )
    selected_time_range = time_range_options[selected_label]

with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

st.markdown("---")

# Main content
with st.spinner("Analyzing health trends..."):
    try:
        data_loader = st.session_state.data_loader
        sensor_data = data_loader.load_sensor_data(
            time_range_hours=selected_time_range
        )
        
        if not sensor_data.empty and 'timestamp' in sensor_data.columns:
            # Calculate health score components
            temp_config = st.session_state.config.get('metrics', {}).get('temperature', {})
            
            # Health Score Overview
            st.subheader("🏥 Overall Health Score")
            
            # Calculate simple health score (placeholder algorithm)
            # Real implementation would use more sophisticated metrics
            health_score = 75  # Placeholder
            
            if 'temperature' in sensor_data.columns:
                # Temperature stability component
                temp_std = sensor_data['temperature'].std()
                temp_mean = sensor_data['temperature'].mean()
                
                normal_min = temp_config.get('normal_min', 38.0)
                normal_max = temp_config.get('normal_max', 39.5)
                
                # Simple scoring: penalize for out-of-range temps and high variability
                temp_in_range = len(sensor_data[
                    (sensor_data['temperature'] >= normal_min) & 
                    (sensor_data['temperature'] <= normal_max)
                ])
                temp_score = (temp_in_range / len(sensor_data)) * 100
                
                # Adjust for variability (lower is better)
                variability_penalty = min(temp_std * 20, 15)
                temp_score = max(0, temp_score - variability_penalty)
                
                health_score = temp_score
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                # Display health score with gauge
                if health_score >= 80:
                    st.success(f"### Health Score: {health_score:.0f}/100")
                    st.success("✅ **Excellent** - Animal is healthy")
                elif health_score >= 60:
                    st.info(f"### Health Score: {health_score:.0f}/100")
                    st.info("⚠️ **Good** - Minor concerns")
                elif health_score >= 40:
                    st.warning(f"### Health Score: {health_score:.0f}/100")
                    st.warning("⚠️ **Fair** - Monitor closely")
                else:
                    st.error(f"### Health Score: {health_score:.0f}/100")
                    st.error("🚨 **Poor** - Requires attention")
            
            with col2:
                st.metric("Data Points", len(sensor_data))
                st.metric("Time Span", f"{selected_time_range} hours")
            
            with col3:
                # Calculate trend
                if len(sensor_data) >= 2:
                    recent_score = health_score  # Simplified
                    st.metric("Trend", "Stable", delta=None)
                else:
                    st.info("Not enough data for trend")
            
            st.markdown("---")
            
            # Health Score Components
            st.subheader("📊 Health Score Components")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🌡️ Temperature Stability**")
                if 'temperature' in sensor_data.columns:
                    temp_score_display = min(100, temp_score)
                    st.progress(temp_score_display / 100)
                    st.metric("Score", f"{temp_score_display:.0f}/100")
                    
                    if temp_score_display >= 80:
                        st.success("Excellent temperature control")
                    elif temp_score_display >= 60:
                        st.info("Good temperature stability")
                    else:
                        st.warning("Temperature concerns detected")
                else:
                    st.info("N/A")
            
            with col2:
                st.markdown("**🏃 Activity Level**")
                # Calculate activity score
                if all(col in sensor_data.columns for col in ['fxa', 'mya', 'rza']):
                    movement_magnitude = (
                        sensor_data[['fxa', 'mya', 'rza']].pow(2).sum(axis=1).apply(lambda x: x**0.5)
                    )
                    avg_movement = movement_magnitude.mean()
                    
                    # Simple scoring (assumes 0.3-0.6 is normal)
                    if 0.3 <= avg_movement <= 0.6:
                        activity_score = 90
                    elif 0.2 <= avg_movement <= 0.8:
                        activity_score = 70
                    else:
                        activity_score = 50
                    
                    st.progress(activity_score / 100)
                    st.metric("Score", f"{activity_score:.0f}/100")
                    
                    if activity_score >= 80:
                        st.success("Normal activity levels")
                    else:
                        st.warning("Activity outside normal range")
                else:
                    st.info("N/A")
            
            with col3:
                st.markdown("**🎯 Behavioral Consistency**")
                # Check behavioral state consistency
                if 'behavioral_state' in sensor_data.columns:
                    state_distribution = sensor_data['behavioral_state'].value_counts(normalize=True)
                    
                    # Check for reasonable distribution
                    lying_pct = state_distribution.get('lying', 0)
                    
                    if 0.35 <= lying_pct <= 0.55:
                        behavior_score = 90
                    elif 0.25 <= lying_pct <= 0.65:
                        behavior_score = 70
                    else:
                        behavior_score = 50
                    
                    st.progress(behavior_score / 100)
                    st.metric("Score", f"{behavior_score:.0f}/100")
                    
                    if behavior_score >= 80:
                        st.success("Normal behavioral patterns")
                    else:
                        st.warning("Unusual behavioral patterns")
                else:
                    st.info("N/A")
            
            st.markdown("---")
            
            # Multi-Day Trends
            st.subheader("📈 Multi-Day Trends")
            
            if 'timestamp' in sensor_data.columns:
                # Group by day
                sensor_data_sorted = sensor_data.sort_values('timestamp')
                sensor_data_sorted['date'] = pd.to_datetime(sensor_data_sorted['timestamp']).dt.date
                
                daily_stats = sensor_data_sorted.groupby('date').agg({
                    'temperature': ['mean', 'min', 'max', 'std'] if 'temperature' in sensor_data.columns else []
                })
                
                if 'temperature' in sensor_data.columns:
                    st.markdown("**Daily Temperature Summary**")
                    st.info("📊 Multi-day line chart showing temperature trends will be added here")
                    
                    # Show daily averages
                    st.dataframe(
                        daily_stats,
                        use_container_width=True,
                    )
            
            st.markdown("---")
            
            # Health Events Timeline
            st.subheader("🕐 Health Events Timeline")
            
            # Get alerts
            alerts = data_loader.load_alerts(max_alerts=100)
            
            if alerts:
                # Filter alerts within time range
                cutoff_time = datetime.now() - timedelta(hours=selected_time_range)
                recent_alerts = []
                
                for alert in alerts:
                    if 'detection_time' in alert:
                        try:
                            detection_time = pd.to_datetime(alert['detection_time'])
                            if detection_time.to_pydatetime() >= cutoff_time:
                                recent_alerts.append(alert)
                        except:
                            pass
                
                if recent_alerts:
                    st.warning(f"⚠️ {len(recent_alerts)} health-related alerts in selected period")
                    
                    st.info("📊 Interactive timeline showing health events and alerts will be added here")
                    
                    # Show summary of alerts
                    alert_types = {}
                    for alert in recent_alerts:
                        alert_type = alert.get('malfunction_type', 'Unknown')
                        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
                    
                    st.markdown("**Alert Summary:**")
                    for alert_type, count in sorted(alert_types.items(), key=lambda x: x[1], reverse=True):
                        st.markdown(f"- **{alert_type}**: {count} occurrence(s)")
                else:
                    st.success("✅ No health alerts in selected period")
            else:
                st.success("✅ No health alerts recorded")
            
            st.markdown("---")
            
            # Long-term Recommendations
            st.subheader("💡 Health Recommendations")
            
            recommendations = []
            
            # Generate recommendations based on data
            if 'temperature' in sensor_data.columns:
                temp_std = sensor_data['temperature'].std()
                if temp_std > 0.3:
                    recommendations.append("⚠️ High temperature variability detected - monitor for fever or heat stress")
                
                avg_temp = sensor_data['temperature'].mean()
                if avg_temp > 39.2:
                    recommendations.append("🌡️ Elevated average temperature - check for signs of illness")
            
            if 'behavioral_state' in sensor_data.columns:
                state_dist = sensor_data['behavioral_state'].value_counts(normalize=True)
                lying_pct = state_dist.get('lying', 0)
                
                if lying_pct < 0.3:
                    recommendations.append("🐮 Low resting time - animal may be stressed or uncomfortable")
                elif lying_pct > 0.6:
                    recommendations.append("😴 Excessive resting time - monitor for lethargy or illness")
            
            if recommendations:
                for rec in recommendations:
                    st.warning(rec)
            else:
                st.success("✅ No concerns detected - animal appears healthy")
            
            st.markdown("---")
            
            # Comparative Analysis
            st.subheader("📊 Comparative Analysis")
            st.info("📊 Charts comparing current period with historical baseline will be added here")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Week-over-Week Comparison**")
                st.info("Chart placeholder: Compare metrics with previous week")
            
            with col2:
                st.markdown("**Health Score History**")
                st.info("Chart placeholder: Track health score over time")
        
        else:
            st.warning("⚠️ No data available for health trend analysis")
            st.info("💡 Select a longer time range or ensure data files are available")
        
    except Exception as e:
        st.error(f"❌ Error analyzing health trends: {str(e)}")

# Placeholder notice
st.markdown("---")
st.info("""
**📝 Note**: This is a placeholder page. Advanced features including:
- Predictive health modeling
- Anomaly detection algorithms
- Comparative herd analysis
- Historical baseline comparisons
- Export health reports (PDF)
- Custom health metrics configuration

...will be added in subsequent development tasks.
""")
