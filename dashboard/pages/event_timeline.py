"""
Event Timeline Visualization Page
==================================
Interactive timeline visualization showing all event types (alerts, behavioral changes, 
health events, sensor issues) in chronological order with filters and detailed views.
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dashboard.utils.event_aggregator import EventAggregator
from dashboard.utils.timeline_viz import TimelineVizBuilder, create_timeline_visualization
from dashboard.utils.db_connection import get_database_connection, get_available_cows
from src.data_processing.event_query import query_all_events, get_event_date_range

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Event Timeline - Artemis Health",
    page_icon="📅",
    layout="wide",
)

# Initialize session state
if 'selected_event' not in st.session_state:
    st.session_state.selected_event = None
if 'event_aggregator' not in st.session_state:
    st.session_state.event_aggregator = EventAggregator()
if 'viz_builder' not in st.session_state:
    st.session_state.viz_builder = TimelineVizBuilder()


def display_event_details(event_row):
    """
    Display detailed information about a selected event in a sidebar or expandable section.
    
    Args:
        event_row: Pandas Series containing event data
    """
    st.markdown("### 📋 Event Details")
    
    # Event header
    st.markdown(f"**{event_row['title']}**")
    
    # Basic information
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Timestamp:**")
        st.text(event_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
        
        st.markdown(f"**Cow ID:**")
        st.text(event_row['cow_id'])
        
        st.markdown(f"**Category:**")
        st.text(event_row['category_label'])
    
    with col2:
        st.markdown(f"**Event Type:**")
        st.text(event_row['event_type'].replace('_', ' ').title())
        
        st.markdown(f"**Severity:**")
        severity = event_row['severity']
        severity_colors = {
            'critical': '🔴',
            'warning': '🟠',
            'info': '🔵',
        }
        st.text(f"{severity_colors.get(severity, '⚪')} {severity.upper()}")
        
        st.markdown(f"**Status:**")
        st.text(event_row['status'].title())
    
    # Description
    if event_row.get('description'):
        st.markdown("**Description:**")
        st.info(event_row['description'])
    
    # Sensor values
    if event_row.get('sensor_values'):
        st.markdown("**Sensor Data:**")
        st.code(event_row['sensor_values'])
    
    # Additional context
    st.markdown("**Source:**")
    st.text(event_row['source'])
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📍 Jump to Time", key="jump_to_time"):
            st.info("Timeline will zoom to this event")
    
    with col2:
        if st.button("🔗 Related Events", key="related_events"):
            st.info("Show related events for this cow")
    
    with col3:
        if st.button("📊 View Metrics", key="view_metrics"):
            st.info("View sensor metrics at this time")


def render_filter_panel():
    """
    Render the filter panel in the sidebar.
    
    Returns:
        Dictionary with filter settings
    """
    st.sidebar.header("🔍 Event Filters")
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    
    # Preset time ranges
    time_range_preset = st.sidebar.selectbox(
        "Quick Select",
        options=[
            "Last 24 Hours",
            "Last 7 Days",
            "Last 30 Days",
            "Last 90 Days",
            "Custom Range",
        ],
        index=1,  # Default to Last 7 Days
    )
    
    # Calculate date range
    now = datetime.now()
    if time_range_preset == "Last 24 Hours":
        start_time = now - timedelta(hours=24)
        end_time = now
    elif time_range_preset == "Last 7 Days":
        start_time = now - timedelta(days=7)
        end_time = now
    elif time_range_preset == "Last 30 Days":
        start_time = now - timedelta(days=30)
        end_time = now
    elif time_range_preset == "Last 90 Days":
        start_time = now - timedelta(days=90)
        end_time = now
    else:  # Custom Range
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=now - timedelta(days=7),
                max_value=now.date(),
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=now,
                max_value=now.date(),
            )
        start_time = datetime.combine(start_date, datetime.min.time())
        end_time = datetime.combine(end_date, datetime.max.time())
    
    st.sidebar.markdown("---")
    
    # Event type filters
    st.sidebar.subheader("📂 Event Categories")
    
    show_alerts = st.sidebar.checkbox("🚨 Alerts", value=True)
    show_behavioral = st.sidebar.checkbox("🔄 Behavioral Changes", value=True)
    show_health = st.sidebar.checkbox("💊 Health Events", value=True)
    show_sensor = st.sidebar.checkbox("📡 Sensor Issues", value=True)
    
    # Severity filters
    st.sidebar.subheader("⚠️ Severity Levels")
    
    show_critical = st.sidebar.checkbox("🔴 Critical", value=True)
    show_warning = st.sidebar.checkbox("🟠 Warning", value=True)
    show_info = st.sidebar.checkbox("🔵 Info", value=True)
    
    st.sidebar.markdown("---")
    
    # Cow filter
    st.sidebar.subheader("🐮 Cow Selection")
    
    # Get available cows
    try:
        conn = get_database_connection()
        available_cows = get_available_cows(conn)
    except:
        available_cows = [1001, 1002, 1003, 1004, 1005]
    
    cow_filter = st.sidebar.selectbox(
        "Filter by Cow",
        options=["All Cows"] + [f"Cow {cid}" for cid in available_cows],
        index=0,
    )
    
    # Parse cow ID
    if cow_filter == "All Cows":
        cow_id = None
    else:
        cow_id = int(cow_filter.split()[1])
    
    # Build filter settings
    filters = {
        'start_time': start_time,
        'end_time': end_time,
        'cow_id': cow_id,
        'event_types': [],
        'categories': [],
        'severities': [],
    }
    
    # Determine which event types to query
    if show_alerts:
        filters['event_types'].append('alerts')
        filters['categories'].extend(['alerts_critical', 'alerts_warning'])
    if show_behavioral:
        filters['event_types'].append('behavioral')
        filters['categories'].append('behavioral')
    if show_health:
        filters['event_types'].append('temperature')
        filters['categories'].append('health')
    if show_sensor:
        filters['event_types'].append('sensor')
        filters['categories'].append('sensor')
    
    # Determine which severities to show
    if show_critical:
        filters['severities'].append('critical')
    if show_warning:
        filters['severities'].append('warning')
    if show_info:
        filters['severities'].append('info')
    
    return filters


def main():
    """Main application logic."""
    
    # Page header
    st.title("📅 Event Timeline Visualization")
    st.markdown(
        "*Interactive timeline showing all system events: alerts, behavioral changes, "
        "health anomalies, and sensor issues*"
    )
    st.markdown("---")
    
    # Render filter panel
    filters = render_filter_panel()
    
    # Check if any event types selected
    if not filters['event_types']:
        st.warning("⚠️ Please select at least one event category from the sidebar filters.")
        return
    
    # Load data button
    if st.button("🔄 Load/Refresh Events", type="primary"):
        with st.spinner("Loading events from database..."):
            try:
                # Get database connection
                conn = get_database_connection()
                
                if conn is None:
                    st.error("❌ Unable to connect to database. Using mock data mode.")
                    # Generate mock events for demonstration
                    st.session_state.events_df = _generate_mock_events(filters)
                else:
                    # Query events
                    event_data = query_all_events(
                        connection=conn,
                        cow_id=filters['cow_id'],
                        start_time=filters['start_time'],
                        end_time=filters['end_time'],
                        event_types=filters['event_types'],
                    )
                    
                    # Aggregate events
                    aggregator = st.session_state.event_aggregator
                    events_df = aggregator.aggregate_events(event_data)
                    
                    # Apply filters
                    events_df = aggregator.filter_events(
                        events_df,
                        categories=filters['categories'],
                        severities=filters['severities'],
                    )
                    
                    st.session_state.events_df = events_df
                    
                    # Close connection
                    if hasattr(conn, 'close'):
                        conn.close()
                
                st.success(f"✅ Loaded {len(st.session_state.events_df)} events")
                
            except Exception as e:
                st.error(f"❌ Error loading events: {str(e)}")
                logger.error(f"Error loading events: {e}", exc_info=True)
                st.session_state.events_df = pd.DataFrame()
    
    # Display timeline if data is loaded
    if 'events_df' in st.session_state and not st.session_state.events_df.empty:
        df = st.session_state.events_df
        
        # Display statistics
        st.subheader("📊 Event Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Events", len(df))
        
        with col2:
            critical_count = len(df[df['severity'] == 'critical'])
            st.metric("Critical Events", critical_count)
        
        with col3:
            warning_count = len(df[df['severity'] == 'warning'])
            st.metric("Warning Events", warning_count)
        
        with col4:
            if filters['cow_id']:
                st.metric("Cow ID", filters['cow_id'])
            else:
                unique_cows = df['cow_id'].nunique()
                st.metric("Cows", unique_cows)
        
        st.markdown("---")
        
        # Create and display timeline
        st.subheader("📈 Interactive Timeline")
        
        try:
            fig = create_timeline_visualization(
                df,
                title=f"Event Timeline ({filters['start_time'].strftime('%Y-%m-%d')} to {filters['end_time'].strftime('%Y-%m-%d')})",
                height=600,
                show_legend=True,
            )
            
            # Display chart
            selected_points = st.plotly_chart(
                fig,
                use_container_width=True,
                key="timeline_chart",
            )
            
        except Exception as e:
            st.error(f"❌ Error creating timeline visualization: {str(e)}")
            logger.error(f"Error creating visualization: {e}", exc_info=True)
        
        st.markdown("---")
        
        # Additional visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Category Distribution")
            try:
                builder = st.session_state.viz_builder
                cat_fig = builder.create_category_distribution(df, height=400)
                st.plotly_chart(cat_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating category chart: {e}")
        
        with col2:
            st.subheader("🎯 Severity Distribution")
            try:
                builder = st.session_state.viz_builder
                sev_fig = builder.create_severity_distribution(df, height=400)
                st.plotly_chart(sev_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating severity chart: {e}")
        
        st.markdown("---")
        
        # Event list view
        st.subheader("📋 Event List")
        
        # Add search/filter
        search_term = st.text_input("🔍 Search events", placeholder="Search by title or description...")
        
        # Filter by search term
        if search_term:
            mask = (
                df['title'].str.contains(search_term, case=False, na=False) |
                df['description'].str.contains(search_term, case=False, na=False)
            )
            display_df = df[mask]
        else:
            display_df = df
        
        # Display events table
        if not display_df.empty:
            # Format for display
            display_cols = ['timestamp', 'cow_id', 'category_label', 'title', 'severity', 'status']
            display_table = display_df[display_cols].copy()
            display_table.columns = ['Time', 'Cow ID', 'Category', 'Title', 'Severity', 'Status']
            
            # Show table
            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True,
                height=400,
            )
            
            # Export button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Event Data (CSV)",
                data=csv,
                file_name=f"event_timeline_{filters['start_time'].strftime('%Y%m%d')}_{filters['end_time'].strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No events match the search criteria.")
        
        # Event detail view in expander
        if not display_df.empty:
            st.markdown("---")
            with st.expander("🔍 View Event Details (Click to expand)"):
                selected_idx = st.selectbox(
                    "Select an event to view details",
                    options=range(len(display_df)),
                    format_func=lambda i: f"{display_df.iloc[i]['timestamp'].strftime('%Y-%m-%d %H:%M')} - {display_df.iloc[i]['title']}",
                )
                
                if selected_idx is not None:
                    display_event_details(display_df.iloc[selected_idx])
    
    elif 'events_df' in st.session_state and st.session_state.events_df.empty:
        st.info("ℹ️ No events found for the selected filters and time range. Try adjusting your filters.")
    
    else:
        st.info("👆 Click 'Load/Refresh Events' to load data from the database.")
        
        # Show helpful tips
        with st.expander("💡 Tips for Using the Event Timeline"):
            st.markdown("""
            **Getting Started:**
            1. Select your desired date range and event categories from the sidebar
            2. Click 'Load/Refresh Events' to query the database
            3. Interact with the timeline by zooming, panning, and hovering over events
            
            **Features:**
            - **Color Coding**: Events are color-coded by category (red=critical alerts, orange=warnings, blue=behavioral, purple=health, gray=sensor)
            - **Hover Tooltips**: Hover over any event marker to see detailed information
            - **Time Range Selector**: Use the range slider below the timeline to zoom into specific periods
            - **Quick Filters**: Use the preset time ranges (24h, 7d, 30d, 90d) for quick filtering
            - **Search**: Use the search box to find specific events by title or description
            - **Export**: Download event data as CSV for further analysis
            
            **Event Categories:**
            - 🚨 **Alerts**: Critical health alerts and warnings from the alert system
            - 🔄 **Behavioral Changes**: State transitions (e.g., lying to standing, feeding to walking)
            - 💊 **Health Events**: Temperature anomalies and physiological concerns
            - 📡 **Sensor Issues**: Sensor malfunctions and data quality problems
            """)


def _generate_mock_events(filters):
    """
    Generate mock events for demonstration when database is unavailable.
    
    Args:
        filters: Filter settings dictionary
        
    Returns:
        DataFrame with mock events
    """
    import numpy as np
    
    events = []
    start = filters['start_time']
    end = filters['end_time']
    
    # Generate random events
    num_events = 50
    timestamps = pd.date_range(start=start, end=end, periods=num_events)
    
    categories = ['alerts_critical', 'alerts_warning', 'behavioral', 'health', 'sensor']
    severities = ['critical', 'warning', 'info']
    
    aggregator = EventAggregator()
    
    for ts in timestamps:
        category = np.random.choice(categories)
        
        if category.startswith('alerts'):
            severity = 'critical' if 'critical' in category else 'warning'
            title = f"Alert: Temperature spike detected"
            event_type = 'fever'
        elif category == 'behavioral':
            severity = 'info'
            title = "Behavioral transition: Standing → Lying"
            event_type = 'state_transition'
        elif category == 'health':
            severity = np.random.choice(['warning', 'info'])
            title = "Temperature anomaly detected"
            event_type = 'temperature_anomaly'
        else:
            severity = 'warning'
            title = "Sensor data quality degraded"
            event_type = 'sensor_malfunction'
        
        event = {
            'timestamp': ts,
            'cow_id': filters['cow_id'] or np.random.randint(1001, 1006),
            'category': category,
            'event_type': event_type,
            'severity': severity,
            'title': title,
            'description': 'Mock event for demonstration',
            'sensor_values': 'temp: 38.5°C',
            'status': 'active',
            'source': 'mock',
            'event_id': f"mock_{ts.timestamp()}",
            'category_label': aggregator.CATEGORY_LABELS[category],
            'color': aggregator.CATEGORY_COLORS[category],
            'marker': aggregator.CATEGORY_MARKERS[category],
            'y_position': aggregator.EVENT_CATEGORIES[category],
        }
        
        events.append(event)
    
    df = pd.DataFrame(events)
    
    # Apply severity filter
    if filters['severities']:
        df = df[df['severity'].isin(filters['severities'])]
    
    # Apply category filter
    if filters['categories']:
        df = df[df['category'].isin(filters['categories'])]
    
    return df


# Footer
def render_footer():
    """Render page footer with information."""
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 10px;'>
        <p><small>Event Timeline Visualization v1.0 - Artemis Health Monitoring System</small></p>
        <p><small>💡 Use filters to customize your view • 📊 Export data for analysis • 🔍 Click events for details</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    render_footer()
