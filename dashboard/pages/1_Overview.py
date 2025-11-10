"""
Overview Page - Artemis Health Dashboard

Displays comprehensive real-time metrics panel with all sensor readings,
behavioral state, temperature baseline comparison, and movement intensity.
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader
from dashboard.utils.data_fetcher import (
    get_latest_sensor_readings,
    get_previous_readings,
    calculate_movement_intensity,
    calculate_baseline_temperature_delta,
    format_freshness_display,
    get_sensor_deltas,
    is_value_concerning,
    get_5min_average_readings
)

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
st.title("📊 Real-Time Metrics Dashboard")
st.markdown("*Comprehensive livestock health monitoring with live sensor data*")
st.markdown("---")

# Main content
with st.spinner("Loading real-time metrics..."):
    try:
        data_loader = st.session_state.data_loader
        
        # Get latest sensor readings
        latest_readings = get_latest_sensor_readings(data_loader)
        
        if latest_readings is None:
            st.error("❌ No sensor data available")
            st.info("💡 Please ensure sensor data files are available in the configured directory")
        else:
            # Get previous readings for delta calculations
            previous_readings = get_previous_readings(data_loader, lookback_minutes=5)
            sensor_deltas = get_sensor_deltas(latest_readings, previous_readings)
            
            # Display data freshness indicator
            freshness_seconds = latest_readings.get('freshness_seconds', 0)
            is_stale = latest_readings.get('is_stale', False)
            freshness_text = format_freshness_display(freshness_seconds)
            
            if freshness_seconds < 60:
                freshness_class = "freshness-current"
                freshness_icon = "✅"
            elif freshness_seconds < 300:
                freshness_class = "freshness-stale"
                freshness_icon = "⚠️"
            else:
                freshness_class = "freshness-old"
                freshness_icon = "❌"
            
            st.markdown(
                f'<div class="{freshness_class}" style="padding: 0.5rem; border-radius: 8px; text-align: center; font-weight: 600;">'
                f'{freshness_icon} Data Updated: {freshness_text}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            
            # ========================================
            # BEHAVIORAL STATE DISPLAY
            # ========================================
            st.subheader("🐮 Current Behavioral State")
            
            behavioral_state = latest_readings.get('behavioral_state', 'unknown')
            state_colors = {
                'lying': 'lying',
                'standing': 'standing',
                'walking': 'walking',
                'ruminating': 'ruminating',
                'feeding': 'feeding',
                'unknown': 'unknown'
            }
            state_icons = {
                'lying': '🛏️',
                'standing': '🧍',
                'walking': '🚶',
                'ruminating': '🔄',
                'feeding': '🍽️',
                'unknown': '❓'
            }
            
            state_class = state_colors.get(behavioral_state.lower(), 'unknown')
            state_icon = state_icons.get(behavioral_state.lower(), '❓')
            
            st.markdown(
                f'<div style="text-align: center; margin: 1.5rem 0;">'
                f'<span class="state-badge state-badge-{state_class}" style="font-size: 1.2rem; padding: 0.8rem 2rem;">'
                f'{state_icon} {behavioral_state.upper()}'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            
            # ========================================
            # SENSOR READINGS GRID (3 columns)
            # ========================================
            st.subheader("📊 Sensor Readings")
            
            # Row 1: Temperature, Movement Intensity, Baseline Comparison
            col1, col2, col3 = st.columns(3)
            
            with col1:
                temp = latest_readings.get('temperature')
                if temp is not None:
                    temp_delta = sensor_deltas.get('temperature')
                    is_concerning = is_value_concerning('temperature', temp, temp_delta)
                    
                    if is_concerning:
                        st.markdown('<div class="metric-alert-fever">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="🌡️ Temperature",
                        value=f"{temp:.1f}°C",
                        delta=f"{temp_delta:+.1f}°C" if temp_delta is not None else None,
                        delta_color="normal"
                    )
                    
                    if is_concerning:
                        st.markdown('</div>', unsafe_allow_html=True)
                        if temp >= 39.5:
                            st.error("⚠️ FEVER DETECTED")
                        elif temp <= 37.5:
                            st.warning("⚠️ Low Temperature")
                else:
                    st.metric(label="🌡️ Temperature", value="N/A")
            
            with col2:
                # Calculate movement intensity
                fxa = latest_readings.get('fxa')
                mya = latest_readings.get('mya')
                rza = latest_readings.get('rza')
                
                if all(v is not None for v in [fxa, mya, rza]):
                    intensity_value, intensity_label = calculate_movement_intensity(fxa, mya, rza)
                    
                    st.metric(
                        label="🏃 Movement Intensity",
                        value=intensity_label,
                        delta=f"{intensity_value:.0f}/100"
                    )
                    
                    # Visual gauge
                    intensity_class = f"intensity-{intensity_label.lower()}"
                    st.markdown(
                        f'<div class="intensity-gauge">'
                        f'<div class="intensity-fill {intensity_class}" style="width: {intensity_value}%;"></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    if intensity_value > 70:
                        st.warning("⚠️ High Activity Detected")
                else:
                    st.metric(label="🏃 Movement Intensity", value="N/A")
            
            with col3:
                # Temperature vs Baseline
                temp = latest_readings.get('temperature')
                if temp is not None:
                    baseline_temp = st.session_state.config.get('metrics', {}).get('temperature', {}).get('normal_min', 38.5)
                    delta_baseline, temp_status = calculate_baseline_temperature_delta(temp, baseline_temp)
                    
                    status_icons = {
                        'normal': '✅',
                        'fever': '🔥',
                        'hypothermia': '🧊',
                        'unknown': '❓'
                    }
                    
                    st.metric(
                        label="📐 Baseline Comparison",
                        value=f"{status_icons.get(temp_status, '❓')} {temp_status.upper()}",
                        delta=f"{delta_baseline:+.1f}°C from {baseline_temp:.1f}°C",
                        delta_color="off" if temp_status == 'normal' else "inverse"
                    )
                else:
                    st.metric(label="📐 Baseline Comparison", value="N/A")
            
            st.markdown("---")
            
            # Row 2: Accelerometer readings (Fxa, Mya, Rza)
            st.markdown("**Accelerometer Data (g-force)**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fxa = latest_readings.get('fxa')
                if fxa is not None:
                    fxa_delta = sensor_deltas.get('fxa')
                    is_concerning = is_value_concerning('fxa', fxa, fxa_delta)
                    
                    if is_concerning:
                        st.markdown('<div class="metric-alert-warning">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="↔️ Fxa (Forward)",
                        value=f"{fxa:.3f} g",
                        delta=f"{fxa_delta:+.3f} g" if fxa_delta is not None else None
                    )
                    
                    if is_concerning:
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.metric(label="↔️ Fxa (Forward)", value="N/A")
            
            with col2:
                mya = latest_readings.get('mya')
                if mya is not None:
                    mya_delta = sensor_deltas.get('mya')
                    is_concerning = is_value_concerning('mya', mya, mya_delta)
                    
                    if is_concerning:
                        st.markdown('<div class="metric-alert-warning">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="↕️ Mya (Lateral)",
                        value=f"{mya:.3f} g",
                        delta=f"{mya_delta:+.3f} g" if mya_delta is not None else None
                    )
                    
                    if is_concerning:
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.metric(label="↕️ Mya (Lateral)", value="N/A")
            
            with col3:
                rza = latest_readings.get('rza')
                if rza is not None:
                    rza_delta = sensor_deltas.get('rza')
                    is_concerning = is_value_concerning('rza', rza, rza_delta)
                    
                    if is_concerning:
                        st.markdown('<div class="metric-alert-warning">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="⬆️ Rza (Vertical)",
                        value=f"{rza:.3f} g",
                        delta=f"{rza_delta:+.3f} g" if rza_delta is not None else None
                    )
                    
                    if is_concerning:
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.metric(label="⬆️ Rza (Vertical)", value="N/A")
            
            st.markdown("---")
            
            # Row 3: Gyroscope readings (Sxg, Lyg, Dzg)
            st.markdown("**Gyroscope Data (°/s)**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                sxg = latest_readings.get('sxg')
                if sxg is not None:
                    sxg_delta = sensor_deltas.get('sxg')
                    is_concerning = is_value_concerning('sxg', sxg, sxg_delta)
                    
                    if is_concerning:
                        st.markdown('<div class="metric-alert-warning">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="🔄 Sxg (Roll)",
                        value=f"{sxg:.2f} °/s",
                        delta=f"{sxg_delta:+.2f} °/s" if sxg_delta is not None else None
                    )
                    
                    if is_concerning:
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.metric(label="🔄 Sxg (Roll)", value="N/A")
            
            with col2:
                lyg = latest_readings.get('lyg')
                if lyg is not None:
                    lyg_delta = sensor_deltas.get('lyg')
                    is_concerning = is_value_concerning('lyg', lyg, lyg_delta)
                    
                    if is_concerning:
                        st.markdown('<div class="metric-alert-warning">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="↕️ Lyg (Pitch)",
                        value=f"{lyg:.2f} °/s",
                        delta=f"{lyg_delta:+.2f} °/s" if lyg_delta is not None else None
                    )
                    
                    if is_concerning:
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.metric(label="↕️ Lyg (Pitch)", value="N/A")
            
            with col3:
                dzg = latest_readings.get('dzg')
                if dzg is not None:
                    dzg_delta = sensor_deltas.get('dzg')
                    is_concerning = is_value_concerning('dzg', dzg, dzg_delta)
                    
                    if is_concerning:
                        st.markdown('<div class="metric-alert-warning">', unsafe_allow_html=True)
                    
                    st.metric(
                        label="🔄 Dzg (Yaw)",
                        value=f"{dzg:.2f} °/s",
                        delta=f"{dzg_delta:+.2f} °/s" if dzg_delta is not None else None
                    )
                    
                    if is_concerning:
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.metric(label="🔄 Dzg (Yaw)", value="N/A")
        
            st.markdown("---")
            
            # ========================================
            # SYSTEM STATUS
            # ========================================
            alert_summary = data_loader.get_alert_summary()
            
            st.subheader("🚨 System Status")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                active_alerts = alert_summary.get('active_alerts', 0)
                if active_alerts > 0:
                    st.error(f"⚠️ **{active_alerts}** active alert(s)")
                else:
                    st.success("✅ No active alerts")
            
            with col2:
                metrics = data_loader.get_latest_metrics()
                data_points = metrics.get('data_points', 0)
                if data_points > 0:
                    st.success(f"✅ **{data_points}** data points")
                else:
                    st.warning("⚠️ No data available")
            
            with col3:
                if latest_readings:
                    timestamp = latest_readings.get('timestamp')
                    if timestamp:
                        st.info(f"🕐 Last: **{timestamp.strftime('%H:%M:%S')}**")
                    else:
                        st.info("🕐 Timestamp: **N/A**")
                else:
                    st.warning("⚠️ No readings available")
            
            st.markdown("---")
            
            # Recent Sensor Data Preview
            st.subheader("📋 Recent Sensor Data")
            
            sensor_data = data_loader.load_sensor_data(
                time_range_hours=1,
                max_rows=20
            )
            
            if not sensor_data.empty:
                # Select relevant columns for display
                display_cols = []
                for col in ['timestamp', 'temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg', 'behavioral_state']:
                    if col in sensor_data.columns:
                        display_cols.append(col)
                
                st.dataframe(
                    sensor_data[display_cols].tail(10),
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
        st.error(f"❌ Error loading real-time metrics: {str(e)}")
        st.info("💡 Please ensure sensor data files are available in the configured data directory.")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details"):
            st.code(str(e))

# Information footer
st.markdown("---")
st.info("""
**ℹ️ About This Panel**: This Real-Time Metrics Panel displays live sensor data from neck-mounted devices. 
Data refreshes automatically every 60 seconds. Delta indicators show changes from the previous 5-minute reading. 
Color-coded alerts highlight concerning values requiring attention.
""")
