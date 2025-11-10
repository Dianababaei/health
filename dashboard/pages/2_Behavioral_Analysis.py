"""
Behavioral Analysis Page - Artemis Health Dashboard

Displays behavioral state timeline, activity patterns, and state transitions.
This is a placeholder page that will be populated with detailed visualizations
by subsequent development tasks.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader

# Page configuration
st.set_page_config(
    page_title="Behavioral Analysis - Artemis Health",
    page_icon="🐮",
    layout="wide",
)

# Initialize session state if needed
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

if 'selected_time_range' not in st.session_state:
    st.session_state.selected_time_range = 24

# Page header
st.title("🐮 Behavioral Analysis")
st.markdown("*Analyze cattle behavioral states and activity patterns*")
st.markdown("---")

# Time range selector
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    time_range_options = {
        "Last 1 hour": 1,
        "Last 6 hours": 6,
        "Last 24 hours": 24,
        "Last 3 days": 72,
        "Last 7 days": 168,
    }
    selected_label = st.selectbox(
        "Select Time Range",
        options=list(time_range_options.keys()),
        index=2,
    )
    selected_time_range = time_range_options[selected_label]

with col2:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

st.markdown("---")

# Main content
with st.spinner("Loading behavioral data..."):
    try:
        data_loader = st.session_state.data_loader
        behavioral_data = data_loader.load_behavioral_data(
            time_range_hours=selected_time_range
        )
        
        if not behavioral_data.empty:
            # Behavioral State Distribution
            st.subheader("📊 Behavioral State Distribution")
            
            if 'behavioral_state' in behavioral_data.columns:
                state_counts = behavioral_data['behavioral_state'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**State Counts**")
                    for state, count in state_counts.items():
                        percentage = (count / len(behavioral_data)) * 100
                        st.markdown(f"- **{state.capitalize()}**: {count} ({percentage:.1f}%)")
                
                with col2:
                    st.markdown("**State Distribution**")
                    # Placeholder for chart
                    st.info("📊 Pie chart will be added here showing state distribution")
            else:
                st.warning("⚠️ Behavioral state data not available in dataset")
            
            st.markdown("---")
            
            # Activity Patterns
            st.subheader("📈 Activity Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Accelerometer Activity**")
                if all(col in behavioral_data.columns for col in ['fxa', 'mya', 'rza']):
                    avg_fxa = behavioral_data['fxa'].mean()
                    avg_mya = behavioral_data['mya'].mean()
                    avg_rza = behavioral_data['rza'].mean()
                    
                    st.markdown(f"- **Forward (fxa)**: {avg_fxa:.3f} g")
                    st.markdown(f"- **Lateral (mya)**: {avg_mya:.3f} g")
                    st.markdown(f"- **Vertical (rza)**: {avg_rza:.3f} g")
                    
                    st.info("📈 Time series chart will be added here")
                else:
                    st.warning("⚠️ Accelerometer data not available")
            
            with col2:
                st.markdown("**Gyroscope Activity**")
                if all(col in behavioral_data.columns for col in ['sxg', 'lyg', 'dzg']):
                    avg_sxg = behavioral_data['sxg'].mean()
                    avg_lyg = behavioral_data['lyg'].mean()
                    avg_dzg = behavioral_data['dzg'].mean()
                    
                    st.markdown(f"- **Roll (sxg)**: {avg_sxg:.2f} °/s")
                    st.markdown(f"- **Pitch (lyg)**: {avg_lyg:.2f} °/s")
                    st.markdown(f"- **Yaw (dzg)**: {avg_dzg:.2f} °/s")
                    
                    st.info("📈 Time series chart will be added here")
                else:
                    st.warning("⚠️ Gyroscope data not available")
            
            st.markdown("---")
            
            # State Timeline
            st.subheader("🕐 Behavioral State Timeline")
            st.info("📊 Interactive timeline chart showing state transitions over time will be added here")
            
            # Show recent state changes
            if 'behavioral_state' in behavioral_data.columns and 'timestamp' in behavioral_data.columns:
                # Detect state changes
                behavioral_data_sorted = behavioral_data.sort_values('timestamp')
                state_changes = behavioral_data_sorted[
                    behavioral_data_sorted['behavioral_state'] != behavioral_data_sorted['behavioral_state'].shift(1)
                ]
                
                if len(state_changes) > 0:
                    st.markdown("**Recent State Changes:**")
                    recent_changes = state_changes.tail(10)[['timestamp', 'behavioral_state']]
                    st.dataframe(recent_changes, use_container_width=True, hide_index=True)
                else:
                    st.info("No state changes detected in selected time range")
            
            st.markdown("---")
            
            # Activity Metrics Summary
            st.subheader("📋 Activity Metrics Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Movement Intensity**")
                if all(col in behavioral_data.columns for col in ['fxa', 'mya', 'rza']):
                    movement_magnitude = (
                        behavioral_data[['fxa', 'mya', 'rza']].pow(2).sum(axis=1).apply(lambda x: x**0.5)
                    )
                    avg_movement = movement_magnitude.mean()
                    max_movement = movement_magnitude.max()
                    
                    st.metric("Average", f"{avg_movement:.3f}")
                    st.metric("Maximum", f"{max_movement:.3f}")
                else:
                    st.info("N/A")
            
            with col2:
                st.markdown("**Rotation Intensity**")
                if all(col in behavioral_data.columns for col in ['sxg', 'lyg', 'dzg']):
                    rotation_magnitude = (
                        behavioral_data[['sxg', 'lyg', 'dzg']].pow(2).sum(axis=1).apply(lambda x: x**0.5)
                    )
                    avg_rotation = rotation_magnitude.mean()
                    max_rotation = rotation_magnitude.max()
                    
                    st.metric("Average", f"{avg_rotation:.2f} °/s")
                    st.metric("Maximum", f"{max_rotation:.2f} °/s")
                else:
                    st.info("N/A")
            
            with col3:
                st.markdown("**Data Summary**")
                st.metric("Total Records", len(behavioral_data))
                if 'timestamp' in behavioral_data.columns:
                    time_span = (
                        behavioral_data['timestamp'].max() - behavioral_data['timestamp'].min()
                    ).total_seconds() / 3600
                    st.metric("Time Span", f"{time_span:.1f} hours")
            
            st.markdown("---")
            
            # Raw Data Table
            with st.expander("📋 View Raw Behavioral Data"):
                st.dataframe(
                    behavioral_data.tail(50),
                    use_container_width=True,
                    hide_index=True,
                )
        
        else:
            st.warning("⚠️ No behavioral data available for the selected time range")
            st.info("💡 Please ensure sensor data files are available in the configured directory")
        
    except Exception as e:
        st.error(f"❌ Error loading behavioral data: {str(e)}")

# Placeholder notice
st.markdown("---")
st.info("""
**📝 Note**: This is a placeholder page. Detailed visualizations including:
- Interactive state timeline charts
- Activity pattern heatmaps
- State transition diagrams
- Behavioral anomaly detection

...will be added in subsequent development tasks.
""")
