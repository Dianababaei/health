"""
Behavioral Timeline Visualization Page
======================================
Interactive timeline visualization showing cattle behavioral states over time
with statistics, zoom/pan controls, and export functionality.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BEHAVIOR_COLORS,
    BEHAVIOR_LABELS,
    BEHAVIOR_DISPLAY_ORDER,
    TIME_RANGES,
    DEFAULT_TIME_RANGE,
    TIMELINE_CHART_CONFIG,
    CHART_INTERACTION_CONFIG,
    get_time_range_timedelta,
    get_state_color,
    get_state_label,
    format_duration,
)
from utils.db_connection import cached_query_behavioral_states, cached_get_available_cows
from utils.behavior_stats import (
    generate_statistics_summary,
    prepare_timeline_data,
    format_duration_text,
    aggregate_by_hour,
)


def create_timeline_chart(df: pd.DataFrame, time_range_label: str) -> go.Figure:
    """
    Create an interactive timeline chart using Plotly.
    
    Args:
        df: DataFrame with columns: state, start, finish, confidence, duration_minutes
        time_range_label: Label for the selected time range
        
    Returns:
        Plotly Figure object
    """
    if df.empty:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No behavioral data available for the selected time range",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray"),
        )
        fig.update_layout(
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig
    
    # Prepare data for timeline
    timeline_df = df.copy()
    
    # Add display labels and colors
    timeline_df['State'] = timeline_df['state'].apply(get_state_label)
    timeline_df['color'] = timeline_df['state'].apply(get_state_color)
    
    # Create hover text
    timeline_df['hover_text'] = timeline_df.apply(
        lambda row: (
            f"<b>{get_state_label(row['state'])}</b><br>"
            f"Start: {row['start'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"End: {row['finish'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
            f"Duration: {format_duration_text(row['duration_minutes'])}<br>"
            f"Confidence: {row['confidence']:.1%}"
        ),
        axis=1
    )
    
    # Create timeline using Plotly timeline
    fig = px.timeline(
        timeline_df,
        x_start='start',
        x_end='finish',
        y='State',
        color='state',
        color_discrete_map=BEHAVIOR_COLORS,
        hover_data={'start': False, 'finish': False, 'state': False},
        custom_data=['hover_text'],
    )
    
    # Update hover template
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
    )
    
    # Update layout
    fig.update_layout(
        title=f"Behavioral Timeline - {time_range_label}",
        xaxis_title="Time",
        yaxis_title="Behavioral State",
        height=TIMELINE_CHART_CONFIG['height'],
        margin=TIMELINE_CHART_CONFIG['margin'],
        showlegend=True,
        legend=TIMELINE_CHART_CONFIG['legend'],
        hovermode='closest',
        plot_bgcolor=TIMELINE_CHART_CONFIG['plot_bgcolor'],
        paper_bgcolor=TIMELINE_CHART_CONFIG['paper_bgcolor'],
    )
    
    # Update x-axis
    fig.update_xaxes(
        showgrid=True,
        gridcolor='#E5E5E5',
        tickformat='%Y-%m-%d %H:%M',
    )
    
    # Update y-axis to show states in order
    fig.update_yaxes(
        categoryorder='array',
        categoryarray=[get_state_label(s) for s in reversed(BEHAVIOR_DISPLAY_ORDER) if s in timeline_df['state'].values],
        showgrid=False,
    )
    
    return fig


def display_statistics_panel(stats: dict, time_range_label: str):
    """
    Display statistics panel with state durations and transitions.
    
    Args:
        stats: Statistics dictionary from generate_statistics_summary
        time_range_label: Label for the selected time range
    """
    st.subheader("📊 Statistics Summary")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Duration by State")
        
        if not stats['durations']:
            st.info("No data available")
        else:
            # Sort by duration descending
            sorted_durations = sorted(
                stats['durations'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for state, duration in sorted_durations:
                percentage = stats['percentages'].get(state, 0)
                color = get_state_color(state)
                label = get_state_label(state)
                duration_text = format_duration_text(duration)
                
                st.markdown(
                    f"<div style='padding: 8px; margin: 4px 0; background-color: {color}20; "
                    f"border-left: 4px solid {color}; border-radius: 4px;'>"
                    f"<b>{label}:</b> {duration_text} ({percentage:.1f}%)"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    with col2:
        st.markdown("### State Transitions")
        st.metric("Total Transitions", stats['transitions'].get('total', 0))
        
        st.markdown("### Longest Continuous Periods")
        
        if not stats['longest_periods']:
            st.info("No significant periods found")
        else:
            # Sort by duration descending
            sorted_periods = sorted(
                stats['longest_periods'].items(),
                key=lambda x: x[1][0],
                reverse=True
            )
            
            # Show top 3 longest periods
            for state, (duration, start, end) in sorted_periods[:3]:
                if duration >= 30:  # Only show periods >= 30 minutes
                    color = get_state_color(state)
                    label = get_state_label(state)
                    duration_text = format_duration_text(duration)
                    
                    st.markdown(
                        f"<div style='padding: 8px; margin: 4px 0; background-color: {color}20; "
                        f"border-left: 4px solid {color}; border-radius: 4px;'>"
                        f"<b>{label}:</b> {duration_text}<br>"
                        f"<small>{start.strftime('%m/%d %H:%M')} - {end.strftime('%m/%d %H:%M')}</small>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


def export_timeline_data(df: pd.DataFrame, cow_id: int, time_range_label: str):
    """
    Create export button for timeline data as CSV.
    
    Args:
        df: Timeline DataFrame
        cow_id: Cow identifier
        time_range_label: Label for the selected time range
    """
    if df.empty:
        return
    
    # Prepare export data
    export_df = df.copy()
    export_df['cow_id'] = cow_id
    
    # Reorder columns
    export_columns = ['cow_id', 'state', 'start', 'finish', 'duration_minutes', 'confidence']
    export_df = export_df[[col for col in export_columns if col in export_df.columns]]
    
    # Format timestamps
    export_df['start'] = export_df['start'].dt.strftime('%Y-%m-%d %H:%M:%S')
    export_df['finish'] = export_df['finish'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Round numeric columns
    if 'duration_minutes' in export_df.columns:
        export_df['duration_minutes'] = export_df['duration_minutes'].round(2)
    if 'confidence' in export_df.columns:
        export_df['confidence'] = export_df['confidence'].round(4)
    
    # Convert to CSV
    csv = export_df.to_csv(index=False)
    
    # Create filename
    start_date = df['start'].min().strftime('%Y%m%d')
    end_date = df['finish'].max().strftime('%Y%m%d')
    filename = f"behavior_timeline_cow{cow_id}_{start_date}_{end_date}.csv"
    
    # Create download button
    st.download_button(
        label="📥 Export Timeline Data (CSV)",
        data=csv,
        file_name=filename,
        mime='text/csv',
    )


def main():
    """
    Main function for the behavioral timeline page.
    """
    # Page configuration
    st.set_page_config(
        page_title="Behavioral Timeline - Artemis Health",
        page_icon="🐄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Page title and description
    st.title("🐄 Behavioral Timeline Visualization")
    st.markdown(
        "Interactive timeline showing behavioral states (lying, standing, walking, ruminating, feeding) "
        "over time with detailed statistics and export capabilities."
    )
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Get available cows
    try:
        available_cows = cached_get_available_cows()
    except Exception as e:
        st.error(f"Error loading cow list: {e}")
        available_cows = [1001, 1002, 1003]
    
    # Cow selector
    selected_cow = st.sidebar.selectbox(
        "Select Cow ID",
        options=available_cows,
        index=0,
        help="Choose a cow to view behavioral timeline"
    )
    
    # Time range selector
    time_range_options = list(TIME_RANGES.keys())
    selected_time_range = st.sidebar.selectbox(
        "Time Range",
        options=time_range_options,
        index=time_range_options.index(DEFAULT_TIME_RANGE) if DEFAULT_TIME_RANGE in time_range_options else 0,
        help="Select the time range for the timeline visualization"
    )
    
    # Get time range info
    time_range_info = TIME_RANGES[selected_time_range]
    time_delta = get_time_range_timedelta(selected_time_range)
    
    # Calculate start and end times
    end_time = datetime.now()
    start_time = end_time - time_delta
    
    # Display time range info
    st.sidebar.info(
        f"**Time Range:** {time_range_info['label']}\n\n"
        f"**From:** {start_time.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"**To:** {end_time.strftime('%Y-%m-%d %H:%M')}"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Main content area
    with st.spinner("Loading behavioral data..."):
        try:
            # Query behavioral states
            df = cached_query_behavioral_states(selected_cow, start_time, end_time)
            
            if df.empty:
                st.warning(
                    f"No behavioral data found for Cow {selected_cow} in the selected time range. "
                    "This could indicate no sensor data or a database connection issue."
                )
                
                # Show mock data option
                st.info(
                    "💡 **Tip:** Set `USE_MOCK_DATA=true` in your environment to use simulated data "
                    "for demonstration purposes."
                )
                return
            
            # Aggregate data if needed for long time ranges
            if time_range_info['granularity'] == 'hour' and len(df) > 10000:
                st.info(f"Aggregating {len(df):,} data points to hourly view for better performance...")
                df = aggregate_by_hour(df)
            
            # Display data summary
            st.success(
                f"✅ Loaded {len(df):,} data points for Cow {selected_cow} "
                f"({time_range_info['label']})"
            )
            
            # Prepare timeline data
            timeline_df = prepare_timeline_data(df)
            
            # Create and display timeline chart
            st.subheader("📈 Behavioral Timeline")
            fig = create_timeline_chart(timeline_df, selected_time_range)
            st.plotly_chart(fig, use_container_width=True, config=CHART_INTERACTION_CONFIG)
            
            # Display statistics
            stats = generate_statistics_summary(df)
            display_statistics_panel(stats, selected_time_range)
            
            # Export functionality
            st.subheader("📤 Export Data")
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                export_timeline_data(timeline_df, selected_cow, selected_time_range)
            
            with col2:
                st.metric("Total Data Points", f"{stats['data_points']:,}")
            
            with col3:
                st.metric("Time Coverage", format_duration_text(stats['total_time']))
            
            # Additional information
            with st.expander("ℹ️ About This Visualization"):
                st.markdown("""
                ### Interactive Features
                - **Zoom:** Scroll or use the zoom tools to focus on specific time periods
                - **Pan:** Click and drag to move along the timeline
                - **Hover:** Hover over segments to see detailed information
                - **Download:** Use the camera icon to export the chart as PNG
                
                ### Behavioral States
                - 🔵 **Lying:** Resting or sleeping in recumbent position
                - 🟢 **Standing:** Upright position with minimal movement
                - 🟠 **Walking:** Active locomotion and movement
                - 🟣 **Ruminating:** Chewing cud (regurgitation and re-chewing)
                - 🟡 **Feeding:** Eating and grazing activities
                
                ### Statistics
                - **Duration:** Total time spent in each state
                - **Transitions:** Number of times the animal changed states
                - **Longest Periods:** Longest continuous duration for each state
                
                ### Data Quality
                Each segment includes a confidence score indicating the classification certainty.
                Higher confidence scores (>90%) indicate more reliable classifications.
                """)
        
        except Exception as e:
            st.error(f"Error loading or processing data: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
