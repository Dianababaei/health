"""
Home - Herd Health Dashboard

Real-time monitoring with animal behavioral states and health status.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config, DataLoader
from components.modern_ui import render_section_header, render_health_score_gauge

# Page config
st.set_page_config(
    page_title="Home - Herd Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Auto-refresh configuration (disabled by default to prevent page flickering)
# Users can manually refresh using the browser refresh button or enable auto-refresh below
AUTO_REFRESH_ENABLED = False
AUTO_REFRESH_INTERVAL_SECONDS = 300  # 5 minutes if enabled

if AUTO_REFRESH_ENABLED:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since_refresh > AUTO_REFRESH_INTERVAL_SECONDS:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# ============================================================================
# HEADER WITH LIVE INDICATOR
# ============================================================================

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("""
    <h1 style="margin: 0; font-size: 32px; font-weight: 700;">🏠 Herd Dashboard</h1>
    <p style="margin: 4px 0 0 0; color: #7f8c8d; font-size: 14px;">
        Real-time health monitoring
    </p>
    """, unsafe_allow_html=True)

with col2:
    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="text-align: right; margin-top: 8px;">
        <div style="display: inline-block; width: 8px; height: 8px; background: #2ecc71; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite;"></div>
        <span style="color: #2ecc71; font-size: 13px; font-weight: 600;">LIVE</span>
        <div style="color: #7f8c8d; font-size: 12px;">{now}</div>
    </div>
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.3; }}
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# LOAD DATA
# ============================================================================

with st.spinner("Loading live data..."):
    try:
        data_loader = st.session_state.data_loader

        # Load sensor data (last 24 hours)
        df = data_loader.load_sensor_data(time_range_hours=24)

        # Load alerts
        alerts = data_loader.load_alerts(max_alerts=50)

        # Calculate health score if data exists
        if len(df) > 0:
            from health_intelligence import HealthScorer

            # Prepare data
            temp_df = df[['timestamp', 'temperature']].copy() if 'temperature' in df.columns else pd.DataFrame()
            activity_df = df[['timestamp']].copy()

            if 'fxa' in df.columns:
                activity_df['movement_intensity'] = np.sqrt(df['fxa']**2 + df.get('fya', 0)**2 + df.get('fza', 0)**2)
            elif 'movement_intensity' in df.columns:
                activity_df['movement_intensity'] = df['movement_intensity']
            else:
                activity_df['movement_intensity'] = 0.5

            behavioral_df = df[['timestamp']].copy()
            behavioral_df['behavioral_state'] = df['state'] if 'state' in df.columns else 'unknown'

            # Calculate health score
            scorer = HealthScorer()
            score_result = scorer.calculate_health_score(
                cow_id='HERD',
                temperature_data=temp_df,
                activity_data=activity_df,
                alert_history=alerts if isinstance(alerts, list) else [],
                behavioral_states=behavioral_df,
                lookback_days=1
            )

            herd_score = score_result.overall_score
            herd_category = score_result.category.value
        else:
            herd_score = None
            herd_category = "unknown"

    except Exception as e:
        st.error(f"Error loading data: {e}")
        herd_score = None
        herd_category = "unknown"
        df = pd.DataFrame()
        alerts = []

# ============================================================================
# TOP CRITICAL METRICS
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    # Health Score Gauge
    if herd_score is not None:
        render_health_score_gauge(herd_score, size="large")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: #f8f9fa; border-radius: 12px;">
            <div style="font-size: 48px; margin-bottom: 12px;">📊</div>
            <div style="color: #7f8c8d; font-size: 14px;">No data yet</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Critical Alerts
    if isinstance(alerts, list):
        alert_count = len(alerts)
        critical_count = sum(1 for a in alerts if str(a.get('severity', '')).lower() == 'critical')

        if critical_count > 0:
            color = "#e74c3c"
            icon = "🔴"
            status = "CRITICAL"
        elif alert_count > 0:
            color = "#f39c12"
            icon = "🟡"
            status = "WARNINGS"
        else:
            color = "#2ecc71"
            icon = "🟢"
            status = "ALL CLEAR"

        st.markdown(f"""
        <div style="text-align: center; padding: 40px 20px; background: #ffffff; border-radius: 12px; border: 3px solid {color};">
            <div style="font-size: 48px; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 32px; font-weight: 700; color: {color}; margin-bottom: 8px;">{critical_count}</div>
            <div style="color: #7f8c8d; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

        if alert_count > 0:
            if st.button("🚨 View All Alerts", use_container_width=True, type="primary"):
                st.switch_page("pages/2_Alerts.py")
    else:
        st.metric("🚨 Active Alerts", 0, "All Clear")

with col3:
    # Animals At Risk
    if len(df) > 0 and herd_score is not None:
        # Count animals needing attention (simplified)
        at_risk = critical_count if isinstance(alerts, list) else 0

        if at_risk > 0:
            color = "#e74c3c"
            icon = "⚠️"
        else:
            color = "#2ecc71"
            icon = "✅"

        st.markdown(f"""
        <div style="text-align: center; padding: 40px 20px; background: #ffffff; border-radius: 12px; border-left: 4px solid {color};">
            <div style="font-size: 48px; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 32px; font-weight: 700; color: {color}; margin-bottom: 8px;">{at_risk}</div>
            <div style="color: #7f8c8d; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">AT RISK</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.metric("🐮 Animals", "0", "No Data")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# LIVE ANIMAL FEED - Real-Time States
# ============================================================================

render_section_header(
    title="🐮 Live Animal Feed",
    subtitle="Current behavior and health status"
)

if len(df) > 0:
    # Get latest data point for each cow
    # Handle missing cow_id column
    if 'cow_id' not in df.columns:
        df['cow_id'] = 'COW_001'

    latest = df.sort_values('timestamp').groupby('cow_id', as_index=False).last()

    for idx, row in latest.iterrows():
        cow_id = row.get('cow_id', f'COW_{idx+1:03d}')
        state = row.get('state', 'unknown')
        temp = row.get('temperature', 0)

        # Calculate activity percentage
        if 'fxa' in row:
            activity = np.sqrt(row['fxa']**2 + row.get('fya', 0)**2 + row.get('fza', 0)**2)
            activity_pct = min(int(activity * 100), 100)
        elif 'movement_intensity' in row:
            activity_pct = min(int(row['movement_intensity'] * 100), 100)
        else:
            activity_pct = 50

        # Determine health status
        if temp > 39.5:
            health_status = "🔴 FEVER"
            health_color = "#e74c3c"
        elif temp > 39.0 and activity_pct > 80:
            health_status = "🟡 HEAT STRESS"
            health_color = "#f39c12"
        elif activity_pct < 20:
            health_status = "🟡 LOW ACTIVITY"
            health_color = "#f39c12"
        else:
            health_status = "🟢 HEALTHY"
            health_color = "#2ecc71"

        # State icon mapping
        state_icons = {
            'lying': '🛏️',
            'standing': '🧍',
            'walking': '🚶',
            'ruminating': '🐄',
            'feeding': '🍽️',
            'unknown': '❓'
        }
        state_icon = state_icons.get(state.lower(), '❓')

        # Activity bar (visual representation)
        filled_bars = int(activity_pct / 10)
        activity_bar = "█" * filled_bars + "░" * (10 - filled_bars)

        # Render animal card
        col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 3, 1])

        with col1:
            st.markdown(f"**{cow_id}**")

        with col2:
            st.markdown(f"<span style='color: {health_color}; font-weight: 600;'>{health_status}</span>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"{state_icon} {state.capitalize()}")

        with col4:
            st.markdown(f"{temp:.1f}°C  `{activity_bar}` {activity_pct}%")

        with col5:
            if health_status != "🟢 HEALTHY":
                if st.button("📋", key=f"details_{cow_id}", help="View Details"):
                    st.switch_page("pages/2_Alerts.py")

        st.markdown("---")

    st.markdown("<br>", unsafe_allow_html=True)

else:
    # No data state
    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 20px;
        background: #fff3cd;
        border-radius: 12px;
        border: 2px solid #f39c12;
    ">
        <div style="font-size: 64px; margin-bottom: 16px;">🐮</div>
        <h3 style="color: #856404; margin-bottom: 8px;">No Live Data</h3>
        <p style="color: #856404; margin-bottom: 20px;">
            Generate simulation data to test dashboard or connect real sensors
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🧪 Generate Simulation Data", use_container_width=True, type="primary"):
            st.switch_page("pages/99_Simulation_Testing.py")

# ============================================================================
# QUICK ACTIONS (Only if data exists)
# ============================================================================

if len(df) > 0:
    st.markdown("<br>", unsafe_allow_html=True)

    render_section_header(
        title="⚡ Quick Actions",
        subtitle="Navigate to detailed views"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚨 Alerts", use_container_width=True, type="secondary"):
            st.switch_page("pages/2_Alerts.py")

    with col2:
        if st.button("📈 Health Trends", use_container_width=True, type="secondary"):
            st.switch_page("pages/3_Health_Trends.py")

    with col3:
        if st.button("🌡️ Temperature", use_container_width=True, type="secondary"):
            st.switch_page("pages/4_Temperature.py")

    with col4:
        if st.button("🧪 Simulation", use_container_width=True, type="secondary"):
            st.switch_page("pages/99_Simulation_Testing.py")

# ============================================================================
# FOOTER WITH AUTO-REFRESH INFO
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)

refresh_status = "Auto-refresh enabled (5 min)" if AUTO_REFRESH_ENABLED else "Manual refresh (use browser refresh button)"
st.markdown(f"""
<div style="
    text-align: center;
    padding: 20px;
    color: #95a5a6;
    font-size: 12px;
    border-top: 1px solid #ecf0f1;
">
    Artemis Livestock Health Monitoring | {refresh_status} | Last updated: {now}
</div>
""", unsafe_allow_html=True)
