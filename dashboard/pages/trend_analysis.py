"""
Historical Trends and Pattern Analysis Dashboard
=================================================
Comprehensive trend analysis with multi-period comparisons, reproductive cycle
tracking, pattern detection, and data export capabilities.
"""

import streamlit as st
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader
from dashboard.utils.trend_calculations import (
    calculate_trend_metrics,
    aggregate_by_period,
    compare_periods,
    detect_recovery_deterioration,
    calculate_rolling_statistics,
    detect_patterns,
    calculate_multi_period_summary
)
from dashboard.utils.reproductive_cycle_viz import (
    detect_estrus_events,
    predict_next_estrus,
    detect_pregnancy,
    create_cycle_timeline,
    calculate_cycle_statistics
)
from src.data_processing.trend_aggregator import TrendAggregator, export_pattern_summary

# Page configuration
st.set_page_config(
    page_title="Trend Analysis - Artemis Health",
    page_icon="📊",
    layout="wide",
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = load_config()

if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Page header
st.title("📊 Historical Trends & Pattern Analysis")
st.markdown("*Multi-period trend comparison, reproductive cycle tracking, and pattern detection*")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Analysis Configuration")
    
    # Maximum time range selector
    max_time_range = st.selectbox(
        "Maximum Time Range",
        options=["90 days", "180 days", "270 days", "365 days"],
        index=1,
    )
    max_days = int(max_time_range.split()[0])
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trend Comparison",
    "🔄 Reproductive Cycle",
    "🔍 Pattern Detection",
    "💾 Export Data"
])

# Load data
with st.spinner("Loading historical data..."):
    try:
        data_loader = st.session_state.data_loader
        
        # Load sensor data for maximum time range
        sensor_data = data_loader.load_sensor_data(
            time_range_hours=max_days * 24
        )
        
        # Calculate activity level if not present
        if not sensor_data.empty and 'activity_level' not in sensor_data.columns:
            if all(col in sensor_data.columns for col in ['fxa', 'mya', 'rza']):
                sensor_data['activity_level'] = np.sqrt(
                    sensor_data['fxa']**2 + 
                    sensor_data['mya']**2 + 
                    sensor_data['rza']**2
                )
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        sensor_data = pd.DataFrame()

# ============================================================================
# TAB 1: TREND COMPARISON
# ============================================================================
with tab1:
    st.header("📈 Multi-Period Trend Comparison")
    
    if sensor_data.empty:
        st.warning("⚠️ No data available for trend analysis")
    else:
        # Period selector
        st.subheader("Select Time Periods to Compare")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            show_7d = st.checkbox("7 days", value=True)
        with col2:
            show_14d = st.checkbox("14 days", value=True)
        with col3:
            show_30d = st.checkbox("30 days", value=True)
        with col4:
            show_90d = st.checkbox("90 days", value=True)
        
        selected_periods = []
        if show_7d:
            selected_periods.append(7)
        if show_14d:
            selected_periods.append(14)
        if show_30d:
            selected_periods.append(30)
        if show_90d:
            selected_periods.append(90)
        
        if not selected_periods:
            st.warning("⚠️ Please select at least one time period")
        else:
            st.markdown("---")
            
            # Temperature Trends
            st.subheader("🌡️ Temperature Trends Comparison")
            
            if 'temperature' in sensor_data.columns:
                # Compare temperature across periods
                period_data = compare_periods(
                    sensor_data,
                    selected_periods,
                    'temperature'
                )
                
                # Create comparison chart
                fig_temp = go.Figure()
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                
                for idx, (period_label, df) in enumerate(period_data.items()):
                    if not df.empty and 'timestamp' in df.columns:
                        # Aggregate by day for cleaner visualization
                        daily_df = aggregate_by_period(df, period='D')
                        
                        if not daily_df.empty and 'temperature_mean' in daily_df.columns:
                            fig_temp.add_trace(go.Scatter(
                                x=daily_df['timestamp'],
                                y=daily_df['temperature_mean'],
                                name=period_label.replace('_', ' ').title(),
                                mode='lines',
                                line=dict(color=colors[idx % len(colors)], width=2),
                                hovertemplate=(
                                    "<b>%{fullData.name}</b><br>"
                                    "Date: %{x|%Y-%m-%d}<br>"
                                    "Temp: %{y:.2f}°C<br>"
                                    "<extra></extra>"
                                ),
                            ))
                
                # Add threshold lines
                temp_config = st.session_state.config.get('metrics', {}).get('temperature', {})
                normal_min = temp_config.get('normal_min', 38.0)
                normal_max = temp_config.get('normal_max', 39.5)
                
                fig_temp.add_hline(
                    y=normal_min,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Normal Min"
                )
                fig_temp.add_hline(
                    y=normal_max,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="Normal Max"
                )
                
                fig_temp.update_layout(
                    title="Temperature Trend Comparison",
                    xaxis_title="Date",
                    yaxis_title="Temperature (°C)",
                    height=500,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Trend metrics for each period
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Temperature Trend Metrics**")
                    
                    metrics_data = []
                    for period_label, df in period_data.items():
                        if not df.empty and 'temperature' in df.columns:
                            metrics = calculate_trend_metrics(df, 'temperature')
                            
                            metrics_data.append({
                                'Period': period_label.replace('_', ' ').title(),
                                'Mean (°C)': f"{metrics['mean']:.2f}",
                                'Std Dev': f"{metrics['std']:.3f}",
                                'Trend': metrics['direction'].title(),
                                'Slope': f"{metrics['slope']:.4f}",
                            })
                    
                    if metrics_data:
                        st.dataframe(
                            pd.DataFrame(metrics_data),
                            use_container_width=True,
                            hide_index=True
                        )
                
                with col2:
                    st.markdown("**Recovery/Deterioration Indicators**")
                    
                    # Compare most recent period to historical baseline
                    if len(period_data) >= 2:
                        shortest_period = min(selected_periods)
                        longest_period = max(selected_periods)
                        
                        current_data = period_data.get(f"{shortest_period}_days")
                        historical_data = period_data.get(f"{longest_period}_days")
                        
                        if current_data is not None and historical_data is not None:
                            recovery = detect_recovery_deterioration(
                                current_data,
                                historical_data,
                                'temperature',
                                threshold=0.05
                            )
                            
                            status = recovery['status']
                            change_pct = recovery['change_pct']
                            
                            if status == 'improving':
                                st.success(f"✅ **Improving Trend**")
                                st.metric(
                                    "Temperature Change",
                                    f"{change_pct:+.1f}%",
                                    delta="Improving",
                                    delta_color="normal"
                                )
                            elif status == 'deteriorating':
                                st.error(f"⚠️ **Deteriorating Trend**")
                                st.metric(
                                    "Temperature Change",
                                    f"{change_pct:+.1f}%",
                                    delta="Deteriorating",
                                    delta_color="inverse"
                                )
                            else:
                                st.info(f"➡️ **Stable Trend**")
                                st.metric(
                                    "Temperature Change",
                                    f"{change_pct:+.1f}%",
                                    delta="Stable"
                                )
                            
                            st.markdown(f"Comparing last {shortest_period} days to {longest_period}-day baseline")
            else:
                st.info("Temperature data not available")
            
            st.markdown("---")
            
            # Activity Trends
            st.subheader("🏃 Activity Level Trends Comparison")
            
            if 'activity_level' in sensor_data.columns:
                # Compare activity across periods
                period_data = compare_periods(
                    sensor_data,
                    selected_periods,
                    'activity_level'
                )
                
                # Create comparison chart
                fig_activity = go.Figure()
                
                for idx, (period_label, df) in enumerate(period_data.items()):
                    if not df.empty and 'timestamp' in df.columns:
                        # Aggregate by day
                        daily_df = aggregate_by_period(df, period='D')
                        
                        if not daily_df.empty and 'activity_level_mean' in daily_df.columns:
                            fig_activity.add_trace(go.Scatter(
                                x=daily_df['timestamp'],
                                y=daily_df['activity_level_mean'],
                                name=period_label.replace('_', ' ').title(),
                                mode='lines',
                                line=dict(color=colors[idx % len(colors)], width=2),
                                hovertemplate=(
                                    "<b>%{fullData.name}</b><br>"
                                    "Date: %{x|%Y-%m-%d}<br>"
                                    "Activity: %{y:.3f}<br>"
                                    "<extra></extra>"
                                ),
                            ))
                
                fig_activity.update_layout(
                    title="Activity Level Trend Comparison",
                    xaxis_title="Date",
                    yaxis_title="Activity Level",
                    height=500,
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_activity, use_container_width=True)
                
                # Activity metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Activity Trend Metrics**")
                    
                    metrics_data = []
                    for period_label, df in period_data.items():
                        if not df.empty and 'activity_level' in df.columns:
                            metrics = calculate_trend_metrics(df, 'activity_level')
                            
                            metrics_data.append({
                                'Period': period_label.replace('_', ' ').title(),
                                'Mean': f"{metrics['mean']:.3f}",
                                'Std Dev': f"{metrics['std']:.3f}",
                                'Trend': metrics['direction'].title(),
                                'Slope': f"{metrics['slope']:.5f}",
                            })
                    
                    if metrics_data:
                        st.dataframe(
                            pd.DataFrame(metrics_data),
                            use_container_width=True,
                            hide_index=True
                        )
                
                with col2:
                    st.markdown("**Recovery/Deterioration Indicators**")
                    
                    if len(period_data) >= 2:
                        shortest_period = min(selected_periods)
                        longest_period = max(selected_periods)
                        
                        current_data = period_data.get(f"{shortest_period}_days")
                        historical_data = period_data.get(f"{longest_period}_days")
                        
                        if current_data is not None and historical_data is not None:
                            recovery = detect_recovery_deterioration(
                                current_data,
                                historical_data,
                                'activity_level',
                                threshold=0.1
                            )
                            
                            status = recovery['status']
                            change_pct = recovery['change_pct']
                            
                            if status == 'improving':
                                st.success(f"✅ **Improving Trend**")
                                st.metric(
                                    "Activity Change",
                                    f"{change_pct:+.1f}%",
                                    delta="Improving",
                                    delta_color="normal"
                                )
                            elif status == 'deteriorating':
                                st.warning(f"⚠️ **Deteriorating Trend**")
                                st.metric(
                                    "Activity Change",
                                    f"{change_pct:+.1f}%",
                                    delta="Deteriorating",
                                    delta_color="inverse"
                                )
                            else:
                                st.info(f"➡️ **Stable Trend**")
                                st.metric(
                                    "Activity Change",
                                    f"{change_pct:+.1f}%",
                                    delta="Stable"
                                )
            else:
                st.info("Activity data not available")

# ============================================================================
# TAB 2: REPRODUCTIVE CYCLE TRACKING
# ============================================================================
with tab2:
    st.header("🔄 Reproductive Cycle Tracking")
    
    if sensor_data.empty:
        st.warning("⚠️ No data available for reproductive cycle analysis")
    else:
        st.markdown("*Estrus cycle detection and pregnancy tracking*")
        st.markdown("---")
        
        # Detect estrus events
        with st.spinner("Detecting estrus events..."):
            estrus_events = detect_estrus_events(
                sensor_data,
                temp_column='temperature',
                activity_column='activity_level'
            )
        
        # Display estrus event summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Detected Estrus Events", len(estrus_events))
        
        with col2:
            if estrus_events:
                latest_event = max(estrus_events, key=lambda x: x['start_time'])
                days_since = (datetime.now() - latest_event['start_time']).days
                st.metric("Days Since Last Estrus", days_since)
            else:
                st.metric("Days Since Last Estrus", "N/A")
        
        with col3:
            # Calculate cycle statistics
            if estrus_events:
                cycle_stats = calculate_cycle_statistics(estrus_events)
                if cycle_stats.get('avg_cycle_length'):
                    st.metric(
                        "Average Cycle Length",
                        f"{cycle_stats['avg_cycle_length']:.1f} days"
                    )
                else:
                    st.metric("Average Cycle Length", "N/A")
            else:
                st.metric("Average Cycle Length", "N/A")
        
        st.markdown("---")
        
        # Timeline visualization
        st.subheader("📅 Reproductive Cycle Timeline")
        
        if estrus_events:
            # Detect pregnancy
            pregnancy_status = detect_pregnancy(
                sensor_data,
                estrus_events,
                temp_column='temperature',
                activity_column='activity_level'
            )
            
            # Create timeline
            start_date = datetime.now() - timedelta(days=min(max_days, 180))
            end_date = datetime.now()
            
            fig_timeline = create_cycle_timeline(
                estrus_events,
                pregnancy_status,
                start_date,
                end_date,
                height=300
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Pregnancy status
            if pregnancy_status and pregnancy_status.get('detected'):
                st.markdown("---")
                st.subheader("🤰 Pregnancy Status")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    confidence = pregnancy_status['confidence']
                    if confidence >= 0.7:
                        st.success(f"**Pregnancy Detected**")
                    else:
                        st.warning(f"**Possible Pregnancy**")
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col2:
                    st.metric(
                        "Days Pregnant",
                        pregnancy_status['days_pregnant']
                    )
                
                with col3:
                    st.metric(
                        "Days Until Calving",
                        pregnancy_status['days_until_calving']
                    )
                
                with col4:
                    expected_date = pregnancy_status['expected_calving_date']
                    st.metric(
                        "Expected Calving",
                        expected_date.strftime('%Y-%m-%d')
                    )
                
                # Pregnancy indicators
                st.markdown("**Pregnancy Indicators:**")
                indicators = pregnancy_status.get('indicators', {})
                
                cols = st.columns(len(indicators))
                for idx, (indicator, confidence) in enumerate(indicators.items()):
                    with cols[idx]:
                        st.metric(
                            indicator.replace('_', ' ').title(),
                            f"{confidence:.1%}"
                        )
            
            # Predicted next estrus
            predicted_estrus = predict_next_estrus(estrus_events)
            
            if predicted_estrus:
                st.markdown("---")
                st.subheader("🔮 Next Estrus Prediction")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    predicted_date = predicted_estrus['predicted_date']
                    st.metric(
                        "Predicted Date",
                        predicted_date.strftime('%Y-%m-%d')
                    )
                
                with col2:
                    st.metric(
                        "Days Until",
                        predicted_estrus['days_until']
                    )
                
                with col3:
                    st.metric(
                        "Confidence",
                        f"{predicted_estrus['confidence']:.1%}"
                    )
                
                if 'prediction_range_days' in predicted_estrus:
                    st.info(
                        f"📊 Prediction range: ± {predicted_estrus['prediction_range_days']} days "
                        f"(based on cycle variability)"
                    )
        else:
            st.info("No estrus events detected in the selected time range. Need more data for analysis.")
        
        # Estrus event details table
        if estrus_events:
            st.markdown("---")
            st.subheader("📋 Detected Estrus Events")
            
            events_df = pd.DataFrame(estrus_events)
            events_df['start_time'] = pd.to_datetime(events_df['start_time']).dt.strftime('%Y-%m-%d %H:%M')
            events_df['end_time'] = pd.to_datetime(events_df['end_time']).dt.strftime('%Y-%m-%d %H:%M')
            events_df['duration_hours'] = events_df['duration_hours'].round(1)
            events_df['confidence'] = (events_df['confidence'] * 100).round(1).astype(str) + '%'
            
            st.dataframe(
                events_df[['start_time', 'end_time', 'duration_hours', 'confidence']],
                use_container_width=True,
                hide_index=True
            )

# ============================================================================
# TAB 3: PATTERN DETECTION
# ============================================================================
with tab3:
    st.header("🔍 Pattern Detection Summary")
    
    if sensor_data.empty:
        st.warning("⚠️ No data available for pattern detection")
    else:
        st.markdown("*Detected anomalies, peaks, and significant events*")
        st.markdown("---")
        
        # Pattern detection controls
        col1, col2 = st.columns(2)
        
        with col1:
            detect_temp_patterns = st.checkbox("Temperature Patterns", value=True)
        with col2:
            detect_activity_patterns = st.checkbox("Activity Patterns", value=True)
        
        all_patterns = []
        
        # Detect temperature patterns
        if detect_temp_patterns and 'temperature' in sensor_data.columns:
            with st.spinner("Detecting temperature patterns..."):
                temp_anomalies = detect_patterns(
                    sensor_data,
                    'temperature',
                    pattern_type='anomalies'
                )
                
                temp_peaks = detect_patterns(
                    sensor_data,
                    'temperature',
                    pattern_type='peaks'
                )
                
                for pattern in temp_anomalies:
                    pattern['metric'] = 'temperature'
                    pattern['type'] = 'anomaly'
                    all_patterns.append(pattern)
                
                for pattern in temp_peaks:
                    pattern['metric'] = 'temperature'
                    all_patterns.append(pattern)
        
        # Detect activity patterns
        if detect_activity_patterns and 'activity_level' in sensor_data.columns:
            with st.spinner("Detecting activity patterns..."):
                activity_anomalies = detect_patterns(
                    sensor_data,
                    'activity_level',
                    pattern_type='anomalies'
                )
                
                activity_peaks = detect_patterns(
                    sensor_data,
                    'activity_level',
                    pattern_type='peaks'
                )
                
                for pattern in activity_anomalies:
                    pattern['metric'] = 'activity_level'
                    pattern['type'] = 'anomaly'
                    all_patterns.append(pattern)
                
                for pattern in activity_peaks:
                    pattern['metric'] = 'activity_level'
                    all_patterns.append(pattern)
        
        # Display patterns
        if all_patterns:
            st.subheader(f"📊 Detected Patterns ({len(all_patterns)} total)")
            
            # Summary counts
            col1, col2, col3 = st.columns(3)
            
            with col1:
                anomaly_count = sum(1 for p in all_patterns if p['type'] == 'anomaly')
                st.metric("Anomalies", anomaly_count)
            
            with col2:
                peak_count = sum(1 for p in all_patterns if p['type'] == 'peak')
                st.metric("Peaks", peak_count)
            
            with col3:
                trough_count = sum(1 for p in all_patterns if p['type'] == 'trough')
                st.metric("Troughs", trough_count)
            
            st.markdown("---")
            
            # Pattern table
            st.subheader("📋 Pattern Details")
            
            patterns_df = pd.DataFrame(all_patterns)
            patterns_df['timestamp'] = pd.to_datetime(patterns_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            patterns_df['value'] = patterns_df['value'].round(3)
            
            # Add severity column for anomalies
            if 'z_score' in patterns_df.columns:
                patterns_df['severity'] = patterns_df['z_score'].apply(
                    lambda x: 'High' if x > 4 else 'Medium' if x > 3 else 'Low'
                )
            
            # Sort by timestamp
            patterns_df = patterns_df.sort_values('timestamp', ascending=False)
            
            # Display table
            display_cols = ['timestamp', 'metric', 'type', 'value']
            if 'severity' in patterns_df.columns:
                display_cols.append('severity')
            
            st.dataframe(
                patterns_df[display_cols],
                use_container_width=True,
                hide_index=True
            )
            
            # Store patterns for export
            st.session_state['detected_patterns'] = all_patterns
        else:
            st.info("No significant patterns detected in the selected time range.")

# ============================================================================
# TAB 4: DATA EXPORT
# ============================================================================
with tab4:
    st.header("💾 Export Trend Data")
    
    st.markdown("*Export historical data and analysis results*")
    st.markdown("---")
    
    # Export configuration
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.radio(
            "Export Format",
            options=["CSV", "JSON", "Both"],
            horizontal=True
        )
    
    with col2:
        # Date range selector for export
        include_date_filter = st.checkbox("Filter by Date Range", value=False)
    
    export_start_date = None
    export_end_date = None
    
    if include_date_filter:
        col1, col2 = st.columns(2)
        with col1:
            export_start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=30)
            )
            export_start_date = datetime.combine(export_start_date, datetime.min.time())
        
        with col2:
            export_end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            export_end_date = datetime.combine(export_end_date, datetime.max.time())
    
    st.markdown("---")
    
    # Export sections
    st.subheader("📦 Select Data to Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_sensor = st.checkbox("Sensor Data", value=True)
    with col2:
        export_daily = st.checkbox("Daily Aggregates", value=True)
    with col3:
        export_patterns = st.checkbox("Detected Patterns", value=True)
    
    st.markdown("---")
    
    # Export button
    if st.button("📥 Generate Export Package", type="primary", use_container_width=True):
        with st.spinner("Generating export package..."):
            try:
                aggregator = TrendAggregator()
                
                # Prepare data for export
                export_sensor_data = sensor_data if export_sensor else pd.DataFrame()
                export_patterns_data = st.session_state.get('detected_patterns', []) if export_patterns else []
                
                # Create export package
                exported_files = aggregator.create_trend_export_package(
                    sensor_data=export_sensor_data,
                    behavioral_data=None,
                    health_scores=None,
                    alerts=None,
                    patterns=export_patterns_data,
                    start_date=export_start_date,
                    end_date=export_end_date,
                    export_format=export_format.lower()
                )
                
                if exported_files:
                    st.success(f"✅ Export package created successfully!")
                    
                    st.markdown("**Exported Files:**")
                    for file_type, filepath in exported_files.items():
                        st.markdown(f"- `{filepath}`")
                    
                    # Provide download links (for small files)
                    if export_sensor and not sensor_data.empty:
                        st.markdown("---")
                        st.subheader("📥 Quick Downloads")
                        
                        # CSV download
                        if export_format.lower() in ['csv', 'both']:
                            csv_buffer = io.StringIO()
                            sensor_data.to_csv(csv_buffer, index=False)
                            st.download_button(
                                label="Download Sensor Data (CSV)",
                                data=csv_buffer.getvalue(),
                                file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        
                        # JSON download
                        if export_format.lower() in ['json', 'both']:
                            json_data = sensor_data.to_json(orient='records', indent=2)
                            st.download_button(
                                label="Download Sensor Data (JSON)",
                                data=json_data,
                                file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json"
                            )
                else:
                    st.warning("⚠️ No data to export")
                
            except Exception as e:
                st.error(f"❌ Error creating export: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Artemis Health Dashboard - Historical Trends & Pattern Analysis</small>
</div>
""", unsafe_allow_html=True)
