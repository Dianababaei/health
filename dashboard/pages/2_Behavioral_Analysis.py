"""
Behavioral Analysis Page - Artemis Health Dashboard

Displays behavioral state timeline, activity patterns, and state transitions.
Features comprehensive activity and behavior monitoring charts with movement
intensity analysis, activity vs rest comparisons, and historical baselines.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader
from dashboard.components.activity_charts import (
    calculate_movement_intensity,
    create_movement_intensity_chart,
    create_activity_rest_bar_chart,
    create_daily_activity_heatmap,
    create_historical_comparison_chart,
    get_activity_summary_stats,
    calculate_historical_baseline
)

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
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    time_range_options = {
        "Last 24 hours": 24,
        "Last 3 days": 72,
        "Last 7 days": 168,
        "Last 14 days": 336,
        "Last 30 days": 720,
    }
    selected_label = st.selectbox(
        "Select Time Range",
        options=list(time_range_options.keys()),
        index=0,
    )
    selected_time_range = time_range_options[selected_label]

with col2:
    baseline_options = {
        "7 days": 7,
        "14 days": 14,
        "30 days": 30,
    }
    baseline_label = st.selectbox(
        "Historical Baseline",
        options=list(baseline_options.keys()),
        index=0,
    )
    baseline_days = baseline_options[baseline_label]

with col3:
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
            # Calculate movement intensity
            if all(col in behavioral_data.columns for col in ['fxa', 'mya', 'rza']):
                behavioral_data = calculate_movement_intensity(behavioral_data)
            
            # Get summary statistics
            summary_stats = get_activity_summary_stats(behavioral_data)
            
            # ========================================================================
            # Activity Summary Cards
            # ========================================================================
            st.subheader("📊 Activity Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'avg_intensity' in summary_stats:
                    st.metric(
                        "Avg Movement Intensity",
                        f"{summary_stats['avg_intensity']:.3f}",
                        delta=None
                    )
                else:
                    st.metric("Avg Movement Intensity", "N/A")
            
            with col2:
                if 'active_percentage' in summary_stats:
                    st.metric(
                        "Active Time",
                        f"{summary_stats['active_percentage']:.1f}%",
                        delta=None
                    )
                else:
                    st.metric("Active Time", "N/A")
            
            with col3:
                if 'rest_percentage' in summary_stats:
                    st.metric(
                        "Rest Time",
                        f"{summary_stats['rest_percentage']:.1f}%",
                        delta=None
                    )
                else:
                    st.metric("Rest Time", "N/A")
            
            with col4:
                if 'stress_periods' in summary_stats:
                    st.metric(
                        "Elevated Activity Events",
                        f"{summary_stats['stress_periods']}",
                        delta=None
                    )
                else:
                    st.metric("Elevated Activity Events", "N/A")
            
            st.markdown("---")
            
            # ========================================================================
            # Movement Intensity Time-Series Chart
            # ========================================================================
            st.subheader("📈 Movement Intensity Over Time")
            
            if 'movement_intensity' in behavioral_data.columns:
                # Calculate baseline for comparison
                baseline = calculate_historical_baseline(
                    behavioral_data,
                    baseline_days=baseline_days
                )
                
                # Create chart
                intensity_fig = create_movement_intensity_chart(
                    behavioral_data,
                    title=f"Movement Intensity - Last {selected_label}",
                    show_stress_markers=True,
                    baseline=baseline
                )
                
                st.plotly_chart(intensity_fig, use_container_width=True)
                
                # Show baseline info
                if baseline and 'avg_intensity' in baseline:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            f"Baseline Avg ({baseline_days}d)",
                            f"{baseline['avg_intensity']:.3f}"
                        )
                    with col2:
                        st.metric(
                            f"Baseline Std Dev",
                            f"{baseline.get('std_intensity', 0):.3f}"
                        )
                    with col3:
                        st.metric(
                            f"Baseline Range",
                            f"{baseline.get('min_intensity', 0):.3f} - {baseline.get('max_intensity', 0):.3f}"
                        )
            else:
                st.warning("⚠️ Movement intensity data not available. Need fxa, mya, rza columns.")
            
            st.markdown("---")
            
            # ========================================================================
            # Activity vs Rest Duration Charts
            # ========================================================================
            st.subheader("⚖️ Activity vs Rest Duration")
            
            if 'behavioral_state' in behavioral_data.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Daily aggregation
                    daily_chart = create_activity_rest_bar_chart(
                        behavioral_data,
                        aggregation='daily',
                        title="Daily Activity vs Rest"
                    )
                    st.plotly_chart(daily_chart, use_container_width=True)
                
                with col2:
                    # Hourly aggregation (for shorter time ranges)
                    if selected_time_range <= 72:  # 3 days or less
                        hourly_chart = create_activity_rest_bar_chart(
                            behavioral_data,
                            aggregation='hourly',
                            title="Hourly Activity vs Rest"
                        )
                        st.plotly_chart(hourly_chart, use_container_width=True)
                    else:
                        st.info("💡 Hourly view available for time ranges up to 3 days. Daily view shown instead.")
                        daily_chart_2 = create_activity_rest_bar_chart(
                            behavioral_data,
                            aggregation='daily',
                            title="Daily Activity vs Rest"
                        )
                        st.plotly_chart(daily_chart_2, use_container_width=True)
            else:
                st.warning("⚠️ Behavioral state data not available for activity/rest analysis")
            
            st.markdown("---")
            
            # ========================================================================
            # Daily Activity Pattern Heatmap
            # ========================================================================
            st.subheader("🌡️ Daily Activity Pattern (24-Hour View)")
            
            if 'movement_intensity' in behavioral_data.columns and selected_time_range >= 24:
                heatmap_fig = create_daily_activity_heatmap(
                    behavioral_data,
                    title="Activity Intensity by Hour of Day"
                )
                st.plotly_chart(heatmap_fig, use_container_width=True)
                
                st.info("""
                📖 **How to read this chart:**
                - Each row represents a day
                - Each column represents an hour (0-23)
                - Darker colors indicate higher movement intensity
                - Look for patterns: cattle typically rest at night (darker on left/right) and are active during day
                """)
            elif selected_time_range < 24:
                st.info("ℹ️ Select a time range of at least 24 hours to view daily activity patterns")
            else:
                st.warning("⚠️ Movement intensity data not available for heatmap")
            
            st.markdown("---")
            
            # ========================================================================
            # Historical Comparison
            # ========================================================================
            st.subheader("📊 Historical Comparison")
            
            if 'movement_intensity' in behavioral_data.columns and selected_time_range >= 24:
                comparison_fig = create_historical_comparison_chart(
                    behavioral_data,
                    baseline_days=baseline_days,
                    title=f"Current Activity vs {baseline_days}-Day Historical Average"
                )
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                st.info("""
                📖 **How to read this chart:**
                - Gray dashed line: Historical average activity by hour of day
                - Blue solid line: Current (last 24h) activity by hour of day
                - Compare patterns to identify deviations from normal behavior
                - Significant deviations may indicate health issues or environmental changes
                """)
            elif selected_time_range < 24:
                st.info("ℹ️ Select a time range of at least 24 hours to view historical comparison")
            else:
                st.warning("⚠️ Movement intensity data not available for comparison")
            
            st.markdown("---")
            
            # ========================================================================
            # Behavioral State Distribution
            # ========================================================================
            st.subheader("📋 Behavioral State Distribution")
            
            if 'behavioral_state' in behavioral_data.columns:
                state_counts = behavioral_data['behavioral_state'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**State Breakdown**")
                    for state, count in state_counts.items():
                        percentage = (count / len(behavioral_data)) * 100
                        st.markdown(f"- **{state.capitalize()}**: {count} records ({percentage:.1f}%)")
                
                with col2:
                    st.markdown("**State Transitions**")
                    if 'state_transitions' in summary_stats:
                        st.metric("Total Transitions", summary_stats['state_transitions'])
                        
                        # Show recent state changes
                        if 'timestamp' in behavioral_data.columns:
                            behavioral_data_sorted = behavioral_data.sort_values('timestamp')
                            state_changes = behavioral_data_sorted[
                                behavioral_data_sorted['behavioral_state'] != behavioral_data_sorted['behavioral_state'].shift(1)
                            ]
                            
                            if len(state_changes) > 0:
                                st.markdown(f"**Recent Changes:** {len(state_changes.tail(10))} shown")
                            else:
                                st.info("No state changes detected")
            else:
                st.warning("⚠️ Behavioral state data not available in dataset")
            
            st.markdown("---")
            
            # ========================================================================
            # Detailed Statistics
            # ========================================================================
            with st.expander("📊 View Detailed Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Movement Statistics**")
                    if 'avg_intensity' in summary_stats:
                        st.write(f"- Average Intensity: {summary_stats['avg_intensity']:.3f}")
                        st.write(f"- Std Deviation: {summary_stats.get('std_intensity', 0):.3f}")
                        st.write(f"- Min Intensity: {summary_stats.get('min_intensity', 0):.3f}")
                        st.write(f"- Max Intensity: {summary_stats.get('max_intensity', 0):.3f}")
                    
                    st.markdown("**Activity Breakdown**")
                    if 'active_count' in summary_stats:
                        st.write(f"- Active Records: {summary_stats['active_count']}")
                        st.write(f"- Rest Records: {summary_stats['rest_count']}")
                        st.write(f"- Active %: {summary_stats['active_percentage']:.1f}%")
                        st.write(f"- Rest %: {summary_stats['rest_percentage']:.1f}%")
                
                with col2:
                    st.markdown("**Data Summary**")
                    st.write(f"- Total Records: {summary_stats.get('total_records', 0)}")
                    if 'time_span_hours' in summary_stats:
                        st.write(f"- Time Span: {summary_stats['time_span_hours']:.1f} hours")
                    if 'state_transitions' in summary_stats:
                        st.write(f"- State Transitions: {summary_stats['state_transitions']}")
                    
                    st.markdown("**Stress Indicators**")
                    if 'stress_periods' in summary_stats:
                        st.write(f"- Elevated Activity Events: {summary_stats['stress_periods']}")
                        st.write(f"- Stress %: {summary_stats.get('stress_percentage', 0):.1f}%")
            
            # Raw Data Table
            with st.expander("📋 View Raw Behavioral Data"):
                display_cols = ['timestamp', 'behavioral_state']
                if 'movement_intensity' in behavioral_data.columns:
                    display_cols.append('movement_intensity')
                if 'temperature' in behavioral_data.columns:
                    display_cols.append('temperature')
                
                display_data = behavioral_data[display_cols].tail(100) if all(c in behavioral_data.columns for c in display_cols) else behavioral_data.tail(100)
                st.dataframe(
                    display_data,
                    use_container_width=True,
                    hide_index=True,
                )
        
        else:
            st.warning("⚠️ No behavioral data available for the selected time range")
            st.info("💡 Please ensure sensor data files are available in the configured directory")
        
    except Exception as e:
        st.error(f"❌ Error loading behavioral data: {str(e)}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
