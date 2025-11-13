"""
Standalone Simulation Data Generator

This is a SEPARATE app from the main dashboard.
Run with: streamlit run simulation_app.py

It only generates simulation data and provides downloads.
No connection to the main app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from simulation import SimulationEngine
from simulation.health_conditions import FeverSimulator, EstrusSimulator, PregnancySimulator, HeatStressSimulator
from health_intelligence import MultiDayHealthTrendTracker
from health_intelligence.alerts.immediate_detector import ImmediateAlertDetector

# Page config
st.set_page_config(
    page_title="Simulation Data Generator",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Livestock Simulation Data Generator")
st.markdown("Generate realistic simulation data for testing. **This is separate from the main app.**")
st.markdown("---")

# Sidebar configuration
st.sidebar.header("⚙️ Simulation Configuration")

# Basic settings
st.sidebar.subheader("1. Basic Settings")
cow_id = st.sidebar.text_input("Cow ID", value="SIM_COW_001")
duration_days = st.sidebar.slider("Duration (days)", 1, 30, 14)
baseline_temp = st.sidebar.number_input("Baseline Temperature (°C)", 36.0, 40.0, 38.5, 0.1)

# Health conditions
st.sidebar.subheader("2. Health Conditions")

condition_fever = st.sidebar.checkbox("🌡️ Fever", value=False)
if condition_fever:
    fever_day = st.sidebar.slider("Fever starts on day:", 1, duration_days, 3)
    fever_duration = st.sidebar.slider("Fever duration (days):", 1, 7, 2)
    fever_temp = st.sidebar.slider("Fever temperature (°C):", 39.5, 41.5, 40.0, 0.1)

condition_estrus = st.sidebar.checkbox("🐄 Estrus", value=False)
if condition_estrus:
    estrus_day = st.sidebar.slider("Estrus on day:", 1, duration_days, duration_days // 2)

condition_pregnancy = st.sidebar.checkbox("🤰 Pregnancy", value=False)
if condition_pregnancy:
    pregnancy_start = st.sidebar.slider("Pregnancy from day:", 1, duration_days, 1)

condition_heat_stress = st.sidebar.checkbox("🌞 Heat Stress", value=False)
if condition_heat_stress:
    heat_day = st.sidebar.slider("Heat stress on day:", 1, duration_days, duration_days // 2)
    heat_duration = st.sidebar.slider("Heat stress duration (hours):", 1, 24, 6)

# Generate button
st.sidebar.markdown("---")
generate_button = st.sidebar.button("🚀 Generate Data", type="primary", use_container_width=True)

# Main area
if generate_button:
    with st.spinner(f"Generating {duration_days} days of simulation data..."):

        # Step 1: Generate baseline data
        st.info("📊 Step 1/5: Generating baseline behavioral data...")

        engine = SimulationEngine(
            baseline_temperature=baseline_temp,
            sampling_rate=1.0  # 1 sample per minute
        )

        df = engine.generate_continuous_data(
            duration_hours=duration_days * 24
        )

        # Step 2: Inject health conditions
        st.info("💉 Step 2/5: Injecting health conditions...")

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

            # Set motion to very low (below 0.15 threshold)
            N = fever_end_idx - fever_start_idx
            df.loc[fever_start_idx:fever_end_idx-1, 'fxa'] = np.random.uniform(0.02, 0.05, N)
            df.loc[fever_start_idx:fever_end_idx-1, 'mya'] = np.random.uniform(0.02, 0.05, N)
            df.loc[fever_start_idx:fever_end_idx-1, 'rza'] = np.random.uniform(0.02, 0.05, N)
            df.loc[fever_start_idx:fever_end_idx-1, 'sxg'] = np.random.uniform(0.02, 0.05, N)
            df.loc[fever_start_idx:fever_end_idx-1, 'lyg'] = np.random.uniform(0.02, 0.05, N)
            df.loc[fever_start_idx:fever_end_idx-1, 'dzg'] = np.random.uniform(0.02, 0.05, N)

        if condition_estrus:
            estrus_sim = EstrusSimulator()
            estrus_idx = (estrus_day - 1) * 24 * 60 + (12 * 60)
            estrus_duration = 8 * 60

            temp_increase = estrus_sim.generate_temperature_pattern(
                duration_minutes=estrus_duration
            )
            activity_increase = estrus_sim.generate_activity_pattern(
                duration_minutes=estrus_duration
            )

            df.loc[estrus_idx:estrus_idx+estrus_duration-1, 'temperature'] += temp_increase[:estrus_duration]
            df.loc[estrus_idx:estrus_idx+estrus_duration-1, 'fxa'] *= activity_increase[:estrus_duration]

        if condition_pregnancy:
            preg_sim = PregnancySimulator()
            preg_start_idx = (pregnancy_start - 1) * 24 * 60
            preg_duration = (duration_days - pregnancy_start + 1) * 24 * 60

            temp_pattern = preg_sim.generate_temperature_pattern(
                duration_minutes=preg_duration
            )

            df.loc[preg_start_idx:preg_start_idx+preg_duration-1, 'temperature'] += temp_pattern[:preg_duration]

        if condition_heat_stress:
            heat_sim = HeatStressSimulator()
            heat_start_idx = (heat_day - 1) * 24 * 60 + (12 * 60)
            heat_end_idx = heat_start_idx + (heat_duration * 60)

            panting = heat_sim.generate_panting_pattern(
                duration_minutes=heat_duration * 60
            )

            df.loc[heat_start_idx:heat_end_idx-1, 'mya'] = panting

        # Step 3: Detect alerts
        st.info("🚨 Step 3/5: Detecting health alerts...")

        alerts = []
        detector = ImmediateAlertDetector()

        window_size = 10
        for idx in range(window_size, len(df), 10):
            window_start = max(0, idx - window_size)
            window_df = df.iloc[window_start:idx+1].copy()

            detected = detector.detect_alerts(
                sensor_data=window_df,
                cow_id=cow_id,
                behavioral_state=df.iloc[idx]['state'],
                baseline_temp=baseline_temp
            )

            if detected:
                alerts.extend([{
                    'timestamp': str(a.timestamp),
                    'cow_id': cow_id,
                    'alert_type': str(a.alert_type),
                    'severity': str(a.severity)
                } for a in detected])

        # Step 4: Calculate trends
        st.info("📈 Step 4/5: Analyzing health trends...")

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

        # Step 5: Prepare downloads
        st.info("📦 Step 5/5: Preparing downloads...")

        # Prepare metadata
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

        # Prepare trend report
        trend_dict = trend_report.to_dict()
        if 'analysis_timestamp' in trend_dict:
            trend_dict['analysis_timestamp'] = str(trend_dict['analysis_timestamp'])

        # Store in session state
        st.session_state.generated_data = {
            'sensor_df': df,
            'alerts': alerts,
            'metadata': metadata,
            'trend_report': trend_dict
        }

        st.success(f"✅ Generated {len(df)} sensor readings with {len(alerts)} alerts!")

# Display downloads if data exists
if 'generated_data' in st.session_state:
    data = st.session_state.generated_data

    st.markdown("---")
    st.header("📥 Download Generated Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Sensor data CSV
        csv_data = data['sensor_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Sensor Data (CSV)",
            data=csv_data,
            file_name=f"{data['metadata']['cow_id']}_sensor_data.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
        st.caption(f"📊 {len(data['sensor_df'])} samples")

    with col2:
        # Alerts JSON
        alerts_json = json.dumps(data['alerts'], indent=2).encode('utf-8')
        st.download_button(
            label="⬇️ Download Alerts (JSON)",
            data=alerts_json,
            file_name=f"{data['metadata']['cow_id']}_alerts.json",
            mime="application/json",
            use_container_width=True,
            type="primary"
        )
        st.caption(f"🚨 {len(data['alerts'])} alerts")

    with col3:
        # Metadata JSON
        metadata_json = json.dumps(data['metadata'], indent=2).encode('utf-8')
        st.download_button(
            label="⬇️ Download Metadata (JSON)",
            data=metadata_json,
            file_name=f"{data['metadata']['cow_id']}_metadata.json",
            mime="application/json",
            use_container_width=True,
            type="primary"
        )
        st.caption("📋 Configuration")

    st.markdown("---")
    st.info("""
    ### 📤 Next Steps:

    1. Download all 3 files above
    2. Open the **main dashboard** app (run: `streamlit run dashboard/app.py`)
    3. Go to Home page → Upload files in sidebar
    4. Click "Refresh to Load Data"

    **Note:** This simulation app is separate from the main dashboard app.
    """)

    # Preview
    st.markdown("---")
    st.header("👀 Preview")

    tab1, tab2, tab3 = st.tabs(["Sensor Data", "Alerts", "Metadata"])

    with tab1:
        st.dataframe(data['sensor_df'].head(100), use_container_width=True)

    with tab2:
        if len(data['alerts']) > 0:
            st.json(data['alerts'][:10])
        else:
            st.warning("No alerts generated")

    with tab3:
        st.json(data['metadata'])

else:
    st.info("👈 Configure settings in sidebar and click 'Generate Data' to start")
