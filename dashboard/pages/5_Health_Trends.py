"""
Health Trends Page - Artemis Health Dashboard

Displays health score gauge, historical trends, and contributing factors breakdown.
Provides comprehensive health monitoring with baseline comparisons and trend analysis.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader
from dashboard.utils.db_connection import get_database_connection
from dashboard.utils.health_visualizations import (
    create_health_gauge,
    create_health_history_chart,
    display_contributing_factors_streamlit,
    create_trend_indicator,
    get_health_status_message,
)
from src.data_processing.health_score_loader import (
    query_health_scores,
    calculate_baseline_health_score,
    get_contributing_factors,
    get_latest_health_score,
)

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
st.title("📈 Health Score & Trends Dashboard")
st.markdown("*Real-time health monitoring with historical analysis and baseline comparisons*")
st.markdown("---")

# Control panel
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

with col1:
    # Time range selector with more granular options
    time_range_options = {
        "Last 7 days": 7,
        "Last 14 days": 14,
        "Last 30 days": 30,
        "Last 90 days": 90,
    }
    selected_label = st.selectbox(
        "Select Time Range",
        options=list(time_range_options.keys()),
        index=1,  # Default to 14 days
    )
    selected_days = time_range_options[selected_label]

with col2:
    # Cow ID selector (for multi-cow support)
    cow_id = st.number_input(
        "Cow ID",
        min_value=1000,
        max_value=9999,
        value=1042,
        step=1,
    )

with col3:
    # Baseline period selector
    baseline_days = st.selectbox(
        "Baseline Period",
        options=[7, 14, 30, 60],
        index=2,  # Default to 30 days
    )

with col4:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.markdown("---")

# Main content - Health Score Dashboard
with st.spinner("Loading health score data..."):
    try:
        # Get database connection
        conn = get_database_connection()
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=selected_days)
        
        # Query health scores
        health_data = query_health_scores(cow_id, start_time, end_time, conn)
        
        # Calculate baseline
        baseline_score, baseline_start, baseline_end = calculate_baseline_health_score(
            cow_id, baseline_days, end_time, conn
        )
        
        # Get latest health score
        latest_data = get_latest_health_score(cow_id, conn)
        
        if health_data.empty or latest_data is None:
            st.warning("⚠️ No health score data available for the selected cow and time range")
            st.info("💡 **Tips:**\n- Try a different cow ID\n- Extend the time range\n- Check if health score data is being generated")
        else:
            # Get current score
            current_score = latest_data['health_score']
            trend_direction = latest_data.get('trend_direction', 'stable')
            trend_rate = latest_data.get('trend_rate', 0.0)
            
            # Display health score gauge and key metrics
            st.subheader("🏥 Current Health Status")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Health Score Gauge
                gauge_fig = create_health_gauge(
                    current_score=current_score,
                    baseline_score=baseline_score,
                    show_delta=True
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Health status and metrics
                status_msg, icon, color = get_health_status_message(current_score)
                st.markdown(f"### {icon} Status")
                st.markdown(f"<p style='color: {color}; font-size: 16px;'><b>{status_msg}</b></p>", unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Display metrics
                st.metric(
                    "Current Score",
                    f"{current_score}/100",
                    delta=f"{current_score - baseline_score:+.1f} from baseline"
                )
                
                st.metric(
                    "Baseline Score",
                    f"{baseline_score:.1f}/100",
                    delta=None,
                    help=f"Average over last {baseline_days} days"
                )
                
                # Trend indicator
                st.markdown("**Trend:**")
                st.markdown(create_trend_indicator(trend_direction, trend_rate), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Health Score History Chart
            st.subheader("📈 Health Score History")
            
            if len(health_data) > 0:
                history_fig = create_health_history_chart(
                    health_data=health_data,
                    time_range_label=selected_label,
                    show_baseline=True,
                    baseline_score=baseline_score
                )
                st.plotly_chart(history_fig, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Score", f"{health_data['health_score'].mean():.1f}")
                
                with col2:
                    st.metric("Highest Score", f"{health_data['health_score'].max():.0f}")
                
                with col3:
                    st.metric("Lowest Score", f"{health_data['health_score'].min():.0f}")
                
                with col4:
                    st.metric("Data Points", len(health_data))
            else:
                st.info("Not enough data for historical chart")
            
            st.markdown("---")
            
            # Contributing Factors Breakdown
            st.subheader("📊 Contributing Factors Breakdown")
            
            # Get contributing factors for latest timestamp
            factors = get_contributing_factors(
                cow_id,
                latest_data['timestamp'],
                conn
            )
            
            if factors:
                # Display using Streamlit components
                display_contributing_factors_streamlit(factors)
                
                # Show component scores
                st.markdown("---")
                st.markdown("**Component Scores (0-1 scale):**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    temp_comp = latest_data.get('temperature_component', 0)
                    st.metric("🌡️ Temperature", f"{temp_comp:.2f}")
                
                with col2:
                    activity_comp = latest_data.get('activity_component', 0)
                    st.metric("🏃 Activity", f"{activity_comp:.2f}")
                
                with col3:
                    behavior_comp = latest_data.get('behavior_component', 0)
                    st.metric("🎯 Behavior", f"{behavior_comp:.2f}")
                
                with col4:
                    rumination_comp = latest_data.get('rumination_component', 0)
                    st.metric("🐄 Rumination", f"{rumination_comp:.2f}")
            else:
                st.info("Contributing factors data not available")
            
            st.markdown("---")
            
            # Historical Trend Analysis
            st.subheader("📉 Trend Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Score Distribution**")
                
                # Count scores in each zone
                if len(health_data) > 0:
                    excellent = len(health_data[health_data['health_score'] >= 80])
                    good = len(health_data[(health_data['health_score'] >= 70) & (health_data['health_score'] < 80)])
                    fair = len(health_data[(health_data['health_score'] >= 40) & (health_data['health_score'] < 70)])
                    poor = len(health_data[health_data['health_score'] < 40])
                    
                    total = len(health_data)
                    
                    st.markdown(f"- 🟢 **Excellent (80-100):** {excellent} ({excellent/total*100:.1f}%)")
                    st.markdown(f"- 🔵 **Good (70-79):** {good} ({good/total*100:.1f}%)")
                    st.markdown(f"- 🟡 **Fair (40-69):** {fair} ({fair/total*100:.1f}%)")
                    st.markdown(f"- 🔴 **Poor (0-39):** {poor} ({poor/total*100:.1f}%)")
            
            with col2:
                st.markdown("**Recommendations**")
                
                recommendations = []
                
                # Generate recommendations based on health score
                if current_score < 60:
                    recommendations.append("🚨 Schedule veterinary examination")
                    recommendations.append("📊 Review detailed behavioral and physiological data")
                elif current_score < 70:
                    recommendations.append("⚠️ Increase monitoring frequency")
                    recommendations.append("📋 Check for early warning signs")
                
                if trend_direction == 'deteriorating':
                    recommendations.append("📉 Health score declining - investigate causes")
                
                # Check component scores
                if latest_data.get('temperature_component', 1.0) < 0.7:
                    recommendations.append("🌡️ Temperature concerns detected")
                
                if latest_data.get('activity_component', 1.0) < 0.7:
                    recommendations.append("🏃 Activity level below normal")
                
                if recommendations:
                    for rec in recommendations:
                        st.warning(rec)
                else:
                    st.success("✅ No specific recommendations - maintain current care routine")
        
        # Close connection if it was opened
        if conn is not None and hasattr(conn, 'close'):
            conn.close()
        
    except Exception as e:
        st.error(f"❌ Error loading health score data: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

# Footer with information
st.markdown("---")
st.markdown("""
### 📊 About Health Scores

The health score (0-100) is calculated based on multiple factors:

- **Temperature Stability (25%)**: Body temperature patterns and circadian rhythm consistency
- **Activity Level (25%)**: Movement patterns and overall physical activity
- **Behavioral Consistency (25%)**: Normal behavioral patterns (lying, standing, feeding, ruminating)
- **Rumination Quality (20%)**: Rumination frequency and duration
- **Alert Impact (5%)**: Penalty for recent health alerts

**Color Zones:**
- 🟢 **Green (70-100)**: Healthy animal, normal monitoring
- 🟡 **Yellow (40-70)**: Fair health, increased monitoring recommended
- 🔴 **Red (0-40)**: Poor health, immediate attention required

**Baseline Comparison:**
The baseline score is calculated as the rolling average over the selected baseline period (default: 30 days).
The delta indicator shows how the current score compares to this historical baseline.
""")
