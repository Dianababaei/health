"""
Simulation Testing Dashboard - Real App Simulation Mode

Generate realistic data for all 3 layers and test the entire system.
This page simulates real cows with health conditions so you can test
your app before real data arrives.

Features:
- Generate behavioral data (Layer 1)
- Inject health conditions (Layer 2/3)
- Real-time trend tracking
- Alert detection
- Complete system integration test
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import plotly.express as px

# Add src and utils to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation import SimulationEngine
from utils.simulation_indicator import check_simulation_active, render_clear_button
from simulation.health_conditions import (
    FeverSimulator,
    EstrusSimulator,
    PregnancySimulator,
    HeatStressSimulator
)
from health_intelligence import MultiDayHealthTrendTracker
try:
    from health_intelligence.alerts.immediate_detector import ImmediateAlertDetector
except:
    ImmediateAlertDetector = None

# Page configuration
st.set_page_config(
    page_title="Simulation Testing",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'current_cow_id' not in st.session_state:
    st.session_state.current_cow_id = 'SIM_COW_001'
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# ============================================================================
# HEADER
# ============================================================================

st.title("🧪 Simulation Testing Dashboard")
st.markdown("""
**Test your entire livestock monitoring system with realistic simulated data.**

Generate data for all 3 layers:
- **Layer 1**: Behavioral states (lying, standing, walking, ruminating, feeding)
- **Layer 2**: Physiological analysis (temperature patterns, circadian rhythm)
- **Layer 3**: Health intelligence (trend tracking, alert detection)
""")

# ============================================================================
# SIDEBAR - SIMULATION CONTROLS
# ============================================================================

st.sidebar.header("🎮 Simulation Controls")

# Show current simulation status
is_active, metadata = check_simulation_active()
if is_active:
    st.sidebar.success(f"✅ Active: `{metadata.get('cow_id', 'Unknown')}`")
    render_clear_button()
else:
    st.sidebar.info("No simulation data yet")

st.sidebar.markdown("---")

# Cow selection/creation
st.sidebar.subheader("1. Select Cow")
cow_id = st.sidebar.text_input(
    "Cow ID",
    value=st.session_state.current_cow_id,
    help="Unique identifier for the simulated cow"
)
st.session_state.current_cow_id = cow_id

# Simulation duration
st.sidebar.subheader("2. Data Generation")
duration_days = st.sidebar.slider(
    "Simulation Duration (days)",
    min_value=1,
    max_value=90,
    value=14,
    help="How many days of data to generate"
)

baseline_temp = st.sidebar.slider(
    "Baseline Temperature (°C)",
    min_value=37.5,
    max_value=39.5,
    value=38.5,
    step=0.1,
    help="Normal body temperature for this cow"
)

# Health condition selection
st.sidebar.subheader("3. Health Conditions")
st.sidebar.markdown("**Select conditions to inject:**")

condition_fever = st.sidebar.checkbox("🌡️ Fever", value=False)
if condition_fever:
    fever_day = st.sidebar.slider("Start on day:", 1, duration_days, duration_days // 2, key='fever_day')
    fever_duration = st.sidebar.slider("Duration (days):", 1, 7, 2, key='fever_dur')
    fever_temp = st.sidebar.slider("Fever temp (°C):", 39.5, 41.5, 40.0, 0.1, key='fever_temp')

condition_estrus = st.sidebar.checkbox("🐄 Estrus", value=False)
if condition_estrus:
    estrus_day = st.sidebar.slider("Estrus on day:", 1, duration_days, duration_days // 2, key='estrus_day')

condition_pregnancy = st.sidebar.checkbox("🤰 Pregnancy", value=False)
if condition_pregnancy:
    pregnancy_start = st.sidebar.slider("Start from day:", 1, duration_days, 1, key='preg_start')

condition_heat_stress = st.sidebar.checkbox("🌞 Heat Stress", value=False)
if condition_heat_stress:
    heat_day = st.sidebar.slider("Heat stress on day:", 1, duration_days, duration_days // 2, key='heat_day')
    heat_duration = st.sidebar.slider("Duration (hours):", 1, 24, 6, key='heat_dur')

# Generate button
st.sidebar.markdown("---")
generate_button = st.sidebar.button(
    "🚀 Generate Simulation Data",
    type="primary",
    use_container_width=True
)

# ============================================================================
# GENERATE SIMULATION DATA
# ============================================================================

if generate_button:
    with st.spinner(f"Generating {duration_days} days of simulation data..."):

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Generate baseline behavioral data
        status_text.text("Step 1/5: Generating baseline behavioral data...")
        progress_bar.progress(20)

        engine = SimulationEngine(
            baseline_temperature=baseline_temp,
            sampling_rate=1.0,  # 1 sample per minute
            random_seed=None  # Random for variation
        )

        df = engine.generate_continuous_data(
            duration_hours=duration_days * 24,
            start_datetime=datetime.now() - timedelta(days=duration_days)
        )

        # Step 2: Inject health conditions
        status_text.text("Step 2/5: Injecting health conditions...")
        progress_bar.progress(40)

        # Fever
        if condition_fever:
            fever_sim = FeverSimulator(
                baseline_fever_temp=fever_temp,
                activity_reduction=0.30
            )
            fever_start_idx = (fever_day - 1) * 24 * 60
            fever_end_idx = fever_start_idx + (fever_duration * 24 * 60)

            fever_temp_data = fever_sim.generate_temperature(
                duration_minutes=fever_duration * 24 * 60
            )

            df.loc[fever_start_idx:fever_end_idx-1, 'temperature'] = fever_temp_data
            # Reduce activity to very low levels (sick cow is nearly motionless)
            # Motion intensity threshold for fever detection is 0.15
            # Set to 0.05-0.10 to ensure detection
            df.loc[fever_start_idx:fever_end_idx-1, 'fxa'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
            df.loc[fever_start_idx:fever_end_idx-1, 'mya'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
            df.loc[fever_start_idx:fever_end_idx-1, 'rza'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
            # Also reduce other accelerometer readings
            df.loc[fever_start_idx:fever_end_idx-1, 'sxg'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
            df.loc[fever_start_idx:fever_end_idx-1, 'lyg'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)
            df.loc[fever_start_idx:fever_end_idx-1, 'dzg'] = np.random.uniform(0.02, 0.05, fever_end_idx - fever_start_idx)

        # Estrus
        if condition_estrus:
            estrus_sim = EstrusSimulator()
            estrus_idx = (estrus_day - 1) * 24 * 60 + (12 * 60)  # Noon
            estrus_duration = 8 * 60  # 8 hours

            estrus_temp = estrus_sim.generate_temperature_spike(
                duration_minutes=estrus_duration
            )

            df.loc[estrus_idx:estrus_idx+estrus_duration-1, 'temperature'] = estrus_temp
            # Increase activity
            df.loc[estrus_idx:estrus_idx+estrus_duration-1, ['fxa', 'mya']] *= 1.5

        # Pregnancy
        if condition_pregnancy:
            pregnancy_sim = PregnancySimulator()
            preg_start_idx = (pregnancy_start - 1) * 24 * 60

            preg_temp = pregnancy_sim.generate_temperature(
                duration_minutes=len(df) - preg_start_idx
            )

            df.loc[preg_start_idx:, 'temperature'] = preg_temp[:len(df)-preg_start_idx]
            # Gradually reduce activity
            reduction_factor = np.linspace(1.0, 0.7, len(df) - preg_start_idx)
            df.loc[preg_start_idx:, ['fxa', 'mya']] *= reduction_factor[:, np.newaxis]

        # Heat stress
        if condition_heat_stress:
            heat_sim = HeatStressSimulator()
            heat_start_idx = (heat_day - 1) * 24 * 60 + (14 * 60)  # 2 PM
            heat_end_idx = heat_start_idx + (heat_duration * 60)

            heat_temp = heat_sim.generate_temperature(
                duration_minutes=heat_duration * 60
            )

            df.loc[heat_start_idx:heat_end_idx-1, 'temperature'] = heat_temp
            # High activity (panting)
            panting = heat_sim.generate_panting_pattern(duration_minutes=heat_duration * 60)
            df.loc[heat_start_idx:heat_end_idx-1, 'mya'] = panting

        # Step 3: Detect alerts
        status_text.text("Step 3/5: Detecting health alerts...")
        progress_bar.progress(60)

        alerts = []
        if ImmediateAlertDetector is not None:
            detector = ImmediateAlertDetector()

            # Process data in 10-minute windows (for efficiency)
            # The detector needs a rolling window of data to detect patterns
            window_size = 10  # 10 samples = 10 minutes
            for idx in range(window_size, len(df), 10):
                # Get sliding window of recent data
                window_start = max(0, idx - window_size)
                window_df = df.iloc[window_start:idx+1].copy()

                # Detect alerts using the window
                detected = detector.detect_alerts(
                    sensor_data=window_df,
                    cow_id=cow_id,
                    behavioral_state=df.iloc[idx]['state'],
                    baseline_temp=baseline_temp
                )

                if detected:
                    alerts.extend([{
                        'timestamp': a.timestamp,
                        'alert_type': a.alert_type,
                        'severity': a.severity,
                        'cow_id': cow_id
                    } for a in detected])

        # Step 4: Calculate trends
        status_text.text("Step 4/5: Analyzing health trends...")
        progress_bar.progress(80)

        tracker = MultiDayHealthTrendTracker(temperature_baseline=baseline_temp)

        temp_df = df[['timestamp', 'temperature']].copy()
        activity_df = df[['timestamp', 'state']].copy()
        activity_df['behavioral_state'] = activity_df['state']
        activity_df['movement_intensity'] = df[['fxa', 'mya', 'rza']].abs().sum(axis=1) / 3.0

        trend_report = tracker.analyze_trends(
            cow_id=cow_id,
            temperature_data=temp_df,
            activity_data=activity_df,
            alert_history=alerts,
            behavioral_states=activity_df
        )

        # Step 5: Save to FILES (for persistence across pages)
        status_text.text("Step 5/5: Saving data to files...")
        progress_bar.progress(90)

        # Create simulation data directory
        sim_dir = Path(__file__).parent.parent.parent / 'data' / 'simulation'
        sim_dir.mkdir(parents=True, exist_ok=True)

        # Save sensor data to CSV (this makes it available to all pages!)
        sensor_file = sim_dir / f'{cow_id}_sensor_data.csv'
        df.to_csv(sensor_file, index=False)

        # Save alerts to JSON
        import json
        alerts_file = sim_dir / f'{cow_id}_alerts.json'
        with open(alerts_file, 'w') as f:
            # Convert to serializable format
            alert_list = []
            for alert in alerts:
                alert_dict = {
                    'timestamp': str(alert['timestamp']),
                    'cow_id': alert['cow_id'],
                    'alert_type': str(alert['alert_type']),
                    'severity': str(alert['severity'])
                }
                alert_list.append(alert_dict)
            json.dump(alert_list, f, indent=2)

        # Save trend report to JSON
        trend_file = sim_dir / f'{cow_id}_trend_report.json'
        with open(trend_file, 'w') as f:
            trend_dict = trend_report.to_dict()
            # Convert datetime to string
            if 'analysis_timestamp' in trend_dict:
                trend_dict['analysis_timestamp'] = str(trend_dict['analysis_timestamp'])
            json.dump(trend_dict, f, indent=2)

        # Save metadata
        metadata_file = sim_dir / f'{cow_id}_metadata.json'
        metadata = {
            'cow_id': cow_id,
            'baseline_temp': baseline_temp,
            'duration_days': duration_days,
            'total_samples': len(df),
            'start_time': str(df['timestamp'].min()),
            'end_time': str(df['timestamp'].max()),
            'num_alerts': len(alerts),
            'conditions': {
                'fever': condition_fever,
                'estrus': condition_estrus,
                'pregnancy': condition_pregnancy,
                'heat_stress': condition_heat_stress
            },
            'generated_at': str(datetime.now())
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Also save to session state for current page
        progress_bar.progress(95)
        st.session_state.simulation_data = {
            'cow_id': cow_id,
            'df': df,
            'alerts': alerts,
            'trend_report': trend_report,
            'baseline_temp': baseline_temp,
            'duration_days': duration_days,
            'conditions': {
                'fever': condition_fever,
                'estrus': condition_estrus,
                'pregnancy': condition_pregnancy,
                'heat_stress': condition_heat_stress
            }
        }

        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()

        st.success(f"✅ Generated {len(df)} sensor readings for {cow_id} ({duration_days} days)")
        st.info(f"📁 Data saved to: `data/simulation/{cow_id}_*.csv/json`\n\n"
                f"✨ This data is now available to ALL dashboard pages!")

# ============================================================================
# DISPLAY SIMULATION RESULTS
# ============================================================================

if st.session_state.simulation_data is not None:
    data = st.session_state.simulation_data
    df = data['df']
    alerts = data['alerts']
    trend_report = data['trend_report']

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🌡️ Temperature Analysis",
        "🏃 Activity Analysis",
        "🚨 Alerts",
        "📈 Health Trends"
    ])

    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.header(f"Simulation Overview - {data['cow_id']}")

        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "Duration",
                f"{data['duration_days']} days",
                f"{len(df)} samples"
            )

        with col2:
            st.metric(
                "Avg Temperature",
                f"{df['temperature'].mean():.2f}°C",
                f"{df['temperature'].mean() - data['baseline_temp']:+.2f}°C"
            )

        with col3:
            st.metric(
                "Alerts Detected",
                len(alerts),
                "Active issues" if len(alerts) > 0 else "Healthy"
            )

        with col4:
            st.metric(
                "Health Trend",
                trend_report.overall_trend.value.title(),
                f"{trend_report.overall_confidence:.0%} confidence"
            )

        with col5:
            most_common_state = df['state'].mode()[0]
            state_pct = (df['state'] == most_common_state).sum() / len(df) * 100
            st.metric(
                "Primary State",
                most_common_state.title(),
                f"{state_pct:.0f}% of time"
            )

        st.markdown("---")

        # Behavioral state distribution
        st.subheader("Behavioral State Distribution")
        state_counts = df['state'].value_counts()

        fig_states = px.pie(
            values=state_counts.values,
            names=state_counts.index,
            title="Time Spent in Each State"
        )
        st.plotly_chart(fig_states, use_container_width=True)

        # Active conditions
        st.subheader("Injected Health Conditions")
        conditions = data['conditions']

        if any(conditions.values()):
            for condition, active in conditions.items():
                if active:
                    emoji_map = {
                        'fever': '🌡️',
                        'estrus': '🐄',
                        'pregnancy': '🤰',
                        'heat_stress': '🌞'
                    }
                    st.success(f"{emoji_map[condition]} {condition.title()} - ACTIVE")
        else:
            st.info("No health conditions injected - baseline healthy cow")

    # ========================================================================
    # TAB 2: TEMPERATURE ANALYSIS
    # ========================================================================
    with tab2:
        st.header("🌡️ Temperature Analysis")

        # Temperature timeline
        fig_temp = go.Figure()

        fig_temp.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['temperature'],
            mode='lines',
            name='Temperature',
            line=dict(color='#FF6B6B', width=1)
        ))

        # Add baseline
        fig_temp.add_hline(
            y=data['baseline_temp'],
            line_dash="dash",
            line_color="green",
            annotation_text=f"Baseline ({data['baseline_temp']}°C)"
        )

        # Add fever threshold
        fig_temp.add_hline(
            y=39.5,
            line_dash="dash",
            line_color="red",
            annotation_text="Fever Threshold (39.5°C)"
        )

        fig_temp.update_layout(
            title="Temperature Over Time",
            xaxis_title="Date/Time",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig_temp, use_container_width=True)

        # Temperature statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Min Temperature", f"{df['temperature'].min():.2f}°C")
        with col2:
            st.metric("Max Temperature", f"{df['temperature'].max():.2f}°C")
        with col3:
            st.metric("Std Deviation", f"{df['temperature'].std():.2f}°C")

        # Fever episodes
        fever_readings = df[df['temperature'] > 39.5]
        if len(fever_readings) > 0:
            st.warning(f"⚠️ Detected {len(fever_readings)} readings above fever threshold (39.5°C)")
            st.write(f"Fever percentage: {len(fever_readings)/len(df)*100:.1f}%")

    # ========================================================================
    # TAB 3: ACTIVITY ANALYSIS
    # ========================================================================
    with tab3:
        st.header("🏃 Activity Analysis")

        # Calculate daily activity
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        daily_activity = df.groupby('date').agg({
            'fxa': 'mean',
            'mya': 'mean',
            'state': lambda x: (x.isin(['walking', 'feeding'])).sum()
        }).reset_index()
        daily_activity.columns = ['date', 'forward_accel', 'lateral_accel', 'active_minutes']

        # Activity timeline
        fig_activity = go.Figure()

        fig_activity.add_trace(go.Bar(
            x=daily_activity['date'],
            y=daily_activity['active_minutes'],
            name='Active Minutes',
            marker_color='#4ECDC4'
        ))

        fig_activity.update_layout(
            title="Daily Activity (Walking + Feeding)",
            xaxis_title="Date",
            yaxis_title="Active Minutes",
            height=400
        )

        st.plotly_chart(fig_activity, use_container_width=True)

        # State timeline
        st.subheader("Behavioral States Over Time")

        # Sample every hour to make chart readable
        hourly_df = df.iloc[::60].copy()

        fig_states = px.scatter(
            hourly_df,
            x='timestamp',
            y='state',
            color='state',
            title="Behavioral State Timeline (hourly sampling)"
        )

        fig_states.update_layout(height=300)
        st.plotly_chart(fig_states, use_container_width=True)

    # ========================================================================
    # TAB 4: ALERTS
    # ========================================================================
    with tab4:
        st.header("🚨 Alert Detection")

        if len(alerts) > 0:
            st.warning(f"⚠️ {len(alerts)} alerts detected during simulation period")

            # Alert summary
            alert_df = pd.DataFrame(alerts)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Alerts by Type")
                type_counts = alert_df['alert_type'].value_counts()
                fig_types = px.bar(
                    x=type_counts.index,
                    y=type_counts.values,
                    labels={'x': 'Alert Type', 'y': 'Count'},
                    color=type_counts.index
                )
                st.plotly_chart(fig_types, use_container_width=True)

            with col2:
                st.subheader("Alerts by Severity")
                severity_counts = alert_df['severity'].value_counts()
                fig_severity = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    color=severity_counts.index,
                    color_discrete_map={
                        'critical': '#FF4444',
                        'warning': '#FFD700'
                    }
                )
                st.plotly_chart(fig_severity, use_container_width=True)

            # Alert timeline
            st.subheader("Alert Timeline")
            alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])

            fig_alert_timeline = px.scatter(
                alert_df,
                x='timestamp',
                y='alert_type',
                color='severity',
                title="Alerts Over Time",
                color_discrete_map={
                    'critical': '#FF4444',
                    'warning': '#FFD700'
                }
            )
            st.plotly_chart(fig_alert_timeline, use_container_width=True)

            # Alert details table
            st.subheader("Alert Details")
            st.dataframe(
                alert_df[['timestamp', 'alert_type', 'severity']].sort_values('timestamp', ascending=False),
                use_container_width=True,
                height=300
            )

        else:
            st.success("✅ No alerts detected - cow appears healthy throughout simulation period")

    # ========================================================================
    # TAB 5: HEALTH TRENDS
    # ========================================================================
    with tab5:
        st.header("📈 Multi-Day Health Trends")

        # Overall assessment
        trend_color = {
            'improving': 'success',
            'stable': 'info',
            'deteriorating': 'error'
        }

        getattr(st, trend_color.get(trend_report.overall_trend.value, 'info'))(
            f"**Overall Health Trend**: {trend_report.overall_trend.value.upper()} "
            f"(Confidence: {trend_report.overall_confidence:.1%})"
        )

        # Period analysis
        st.subheader("Trend Analysis by Period")

        periods = ['trend_7day', 'trend_14day', 'trend_30day', 'trend_90day']
        period_names = ['7-Day', '14-Day', '30-Day', '90-Day']

        for period_attr, period_name in zip(periods, period_names):
            period_data = getattr(trend_report, period_attr)

            if period_data is not None:
                with st.expander(f"📊 {period_name} Analysis", expanded=(period_attr == 'trend_7day')):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Trend Direction",
                            period_data.trend_indicator.value.title()
                        )

                    with col2:
                        st.metric(
                            "Avg Temperature",
                            f"{period_data.temperature_mean:.2f}°C",
                            f"{period_data.temperature_baseline_drift:+.2f}°C drift"
                        )

                    with col3:
                        st.metric(
                            "Activity Level",
                            f"{period_data.activity_level_mean:.2f}",
                            f"{period_data.activity_diversity:.2f} diversity"
                        )

                    st.write(f"**Data Completeness**: {period_data.data_completeness:.1%}")
                    st.write(f"**Confidence**: {period_data.confidence_score:.1%}")

                    if period_data.alert_count > 0:
                        st.warning(f"⚠️ {period_data.alert_count} alerts in this period")

        # Recommendations
        if trend_report.recommendations:
            st.subheader("🎯 Recommendations")
            for rec in trend_report.recommendations:
                st.info(f"• {rec}")

        # Significant changes
        if trend_report.significant_changes:
            st.subheader("⚡ Significant Changes Detected")
            for change in trend_report.significant_changes:
                st.warning(f"• {change}")

    # ========================================================================
    # EXPORT OPTIONS
    # ========================================================================
    st.markdown("---")
    st.subheader("💾 Export Simulation Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📥 Download Sensor Data (CSV)", use_container_width=True):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"simulation_{data['cow_id']}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("📥 Download Trend Report (JSON)", use_container_width=True):
            import json
            trend_json = json.dumps(trend_report.to_dict(), indent=2, default=str)
            st.download_button(
                label="Download JSON",
                data=trend_json,
                file_name=f"trends_{data['cow_id']}_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

    with col3:
        if st.button("📥 Download Alert Log (CSV)", use_container_width=True):
            if alerts:
                alert_csv = pd.DataFrame(alerts).to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=alert_csv,
                    file_name=f"alerts_{data['cow_id']}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No alerts to export")

else:
    # No data yet - show instructions
    st.info("👈 Use the sidebar controls to configure and generate simulation data")

    st.markdown("""
    ### How to Use This Simulation Dashboard

    1. **Configure Cow**: Enter a cow ID (or use default)
    2. **Set Duration**: Choose how many days to simulate (1-90 days)
    3. **Inject Conditions** (optional):
       - 🌡️ Fever: Elevated temperature + reduced activity
       - 🐄 Estrus: Temperature spike + increased activity
       - 🤰 Pregnancy: Stable temperature + gradual activity reduction
       - 🌞 Heat Stress: High temperature + panting behavior
    4. **Generate**: Click "Generate Simulation Data"
    5. **Analyze**: Explore the 5 tabs to see complete analysis

    ### What Gets Generated

    - **Layer 1**: Behavioral states (lying, standing, walking, ruminating, feeding)
    - **Layer 2**: Temperature patterns, circadian rhythm
    - **Layer 3**: Health trends (7/14/30/90 days), alert detection

    ### Testing Your App

    This simulation generates data exactly like real cows, so you can:
    - Test all dashboard pages with realistic data
    - Verify alert detection works correctly
    - Test trend analysis algorithms
    - Train machine learning models
    - Demo the system to stakeholders

    **After real data arrives**, this simulation can still be used for:
    - Testing new features before deployment
    - Training new staff
    - Demonstrating "what-if" scenarios
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.caption("🧪 Simulation Testing Dashboard | Powered by validated health simulators")
