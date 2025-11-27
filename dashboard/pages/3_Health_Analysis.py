"""
Health Analysis Page - Artemis Health Dashboard

Comprehensive health analysis including:
- Health score trends over time
- Behavioral pattern analysis
- Temperature and activity correlation
- Reproductive cycle tracking
- Long-term health indicators
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Add parent directory to path for imports
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

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
    page_title="Health Analysis - Artemis Health",
    page_icon="📊",
    layout="wide",
)

# Initialize session state if needed
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Page header
st.title("📊 Health Analysis & Long-term Trends")
st.markdown("*Real-time health monitoring with historical analysis and baseline comparisons*")

# Database Status
try:
    import sqlite3
    from pathlib import Path

    db_path = Path("data/alert_state.db")

    if db_path.exists():
        conn_db = sqlite3.connect(str(db_path), timeout=30.0)
        cursor = conn_db.cursor()

        cursor.execute("SELECT COUNT(*) FROM health_scores")
        total_scores = cursor.fetchone()[0]

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM health_scores")
        date_range = cursor.fetchone()

        cursor.execute("SELECT AVG(total_score) FROM health_scores")
        avg_score = cursor.fetchone()[0]

        conn_db.close()

        # Show metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Total Records", total_scores)
        with col2:
            if avg_score:
                st.metric("📈 Average Score", f"{avg_score:.1f}")
            else:
                st.metric("📈 Average Score", "--")
        with col3:
            if date_range[0] and date_range[1]:
                from datetime import datetime
                start = datetime.fromisoformat(date_range[0])
                end = datetime.fromisoformat(date_range[1])
                days = (end - start).days + 1
                st.metric("📅 Data Range", f"{days} days")
            else:
                st.metric("📅 Data Range", "--")
    else:
        st.info("📂 No health score database found. Upload CSV data to generate health scores.")

except Exception as e:
    st.warning(f"Could not load database statistics: {e}")

st.markdown("---")

# Control panel
col1, col2 = st.columns([3, 1])

with col1:
    # Cow ID (fixed for single-cow mode)
    cow_id = "COW_001"
    st.info(f"**Cow ID:** {cow_id} | **Showing:** All available data")

with col2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Fixed settings - no user selection needed
baseline_days = 30  # Use 30 days for baseline calculation

st.markdown("---")

# Main content - Health Score Dashboard
with st.spinner("Loading all health score data..."):
    try:
        # Get ALL health scores (no time filtering)
        from src.health_intelligence.logging.health_score_manager import HealthScoreManager
        manager = HealthScoreManager(db_path="data/alert_state.db")

        # Query ALL health scores for this cow
        all_scores = manager.query_health_scores(cow_id=cow_id, sort_order="ASC")

        if not all_scores.empty:
            # Convert to expected format
            score_dates = pd.to_datetime(all_scores['timestamp'])
            oldest = score_dates.min()
            newest = score_dates.max()

            # Use the actual data range
            start_time = oldest.to_pydatetime()
            end_time = newest.to_pydatetime()

            # Map to expected column names
            health_data = all_scores.rename(columns={
                'total_score': 'health_score',
                'temperature_score': 'temperature_component',
                'activity_score': 'activity_component',
                'behavioral_score': 'behavior_component',
                'alert_score': 'alert_penalty'
            })

            # Add missing columns
            if 'rumination_component' not in health_data.columns:
                health_data['rumination_component'] = 0.0
            if 'trend_direction' not in health_data.columns:
                health_data['trend_direction'] = 'stable'
            if 'trend_rate' not in health_data.columns:
                health_data['trend_rate'] = 0.0

            # Get latest health score
            latest_data = get_latest_health_score(cow_id, None)

            # Calculate baseline using last 30 days from newest date
            baseline_score, baseline_start, baseline_end = calculate_baseline_health_score(
                cow_id, baseline_days, newest.to_pydatetime(), None
            )

            # Debug info (optional)
            with st.expander("🔍 Data Info", expanded=False):
                st.write(f"**Total Records:** {len(health_data)}")
                st.write(f"**Date Range:** {oldest.strftime('%Y-%m-%d %H:%M')} to {newest.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Days of Data:** {(newest - oldest).days + 1}")
                st.write(f"**Baseline Score:** {baseline_score:.1f}")
        else:
            # No health scores at all
            health_data = pd.DataFrame()
            latest_data = None

        if health_data.empty or latest_data is None:
            # No health scores at all
            st.warning("⚠️ No health score data available for COW_001")
            st.info("💡 **How to generate health scores:**\n\n"
                   "1. Go to the **Home** page\n"
                   "2. Upload CSV data using the sidebar\n"
                   "3. Wait for processing to complete\n"
                   "4. Health scores will be automatically saved to database\n"
                   "5. Return to this page to view trends")
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
                # Calculate days of data for label
                days_of_data = (newest - oldest).days + 1
                time_range_label = f"All data ({days_of_data} days)"

                history_fig = create_health_history_chart(
                    health_data=health_data,
                    time_range_label=time_range_label,
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
                None  # connection not needed (uses SQLite directly)
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

    except Exception as e:
        st.error(f"❌ Error loading health score data: {str(e)}")
        import traceback
        st.error(f"Details: {traceback.format_exc()}")

# Footer with information
st.markdown("---")
st.markdown("""
### 📊 About Health Scores

The health score (0-100) is calculated based on three core factors:

- **Temperature Stability (30%)**: Body temperature patterns and deviation from baseline
- **Activity Level (25%)**: Movement patterns and overall physical activity
- **Behavioral Consistency (25%)**: Normal behavioral patterns (lying, standing, walking, feeding)
- **Alert Penalty (20%)**: Deductions for active health alerts (see Alerts page for details)

> **Note:** Rumination detection is currently **disabled** due to sampling rate limitation.
> Accurate rumination detection requires ≥10 Hz sampling to detect jaw movement at 1.0-1.5 Hz.
> Current system uses 1 sample/minute. *(References: Schirmann et al. 2009, Burfeind et al. 2011)*

**Color Zones:**
- 🟢 **Green (80-100)**: Excellent health, normal monitoring
- 🟡 **Yellow (60-80)**: Good health, routine monitoring
- 🟠 **Orange (40-60)**: Moderate concern, increased monitoring recommended
- 🔴 **Red (0-40)**: Poor health, immediate attention required

**Baseline Comparison:**
The baseline score is calculated as the rolling average over the last 30 days of available data.
The delta indicator shows how the current score compares to this historical baseline.

**Health Score History Chart:**
Shows the time-series of health scores over all available data. The blue line tracks how the cow's health has changed over time, while the green dashed line shows the 30-day baseline average for comparison.
""")
