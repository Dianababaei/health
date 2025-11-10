"""
Artemis Health Dashboard - Main Application

Multi-page Streamlit application for livestock health monitoring.
Provides real-time metrics, behavioral analysis, temperature monitoring,
alerts dashboard, and health trends visualization.
"""

import streamlit as st
from pathlib import Path
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.utils.data_loader import load_config, DataLoader


def load_custom_css():
    """Load custom CSS styling."""
    css_file = Path(__file__).parent / "styles" / "custom.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state.auto_refresh_enabled = True
    
    if 'selected_time_range' not in st.session_state:
        st.session_state.selected_time_range = 24  # hours
    
    if 'selected_cow_id' not in st.session_state:
        st.session_state.selected_cow_id = None
    
    if 'config' not in st.session_state:
        st.session_state.config = load_config()
    
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = DataLoader(st.session_state.config)


def setup_page_config(config):
    """Configure Streamlit page settings."""
    dashboard_config = config.get('dashboard', {})
    
    st.set_page_config(
        page_title=dashboard_config.get('title', 'Artemis Health'),
        page_icon=dashboard_config.get('page_icon', '🐄'),
        layout=dashboard_config.get('layout', 'wide'),
        initial_sidebar_state=dashboard_config.get('initial_sidebar_state', 'expanded'),
    )


def render_sidebar():
    """Render sidebar with navigation and controls."""
    with st.sidebar:
        # Branding
        st.markdown("# 🐄 Artemis Health")
        st.markdown("*Livestock Health Monitoring*")
        st.markdown("---")
        
        # Data refresh controls
        st.subheader("Data Controls")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox(
                "Auto-refresh",
                value=st.session_state.auto_refresh_enabled,
                help="Automatically refresh data every 60 seconds"
            )
            st.session_state.auto_refresh_enabled = auto_refresh
        
        # Last refresh time
        st.caption(f"Last refresh: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Time range selector
        st.subheader("Time Range")
        time_range_options = {
            "1 hour": 1,
            "6 hours": 6,
            "24 hours": 24,
            "3 days": 72,
            "7 days": 168,
        }
        
        selected_label = st.selectbox(
            "Select time range",
            options=list(time_range_options.keys()),
            index=2,  # Default to 24 hours
        )
        st.session_state.selected_time_range = time_range_options[selected_label]
        
        st.markdown("---")
        
        # System status
        st.subheader("System Status")
        
        # Get alert summary
        try:
            alert_summary = st.session_state.data_loader.get_alert_summary()
            active_alerts = alert_summary.get('active_alerts', 0)
            
            if active_alerts > 0:
                st.error(f"⚠️ {active_alerts} active alert(s)")
            else:
                st.success("✅ No active alerts")
        except:
            st.info("ℹ️ Alert status unavailable")
        
        # Data status
        try:
            metrics = st.session_state.data_loader.get_latest_metrics()
            data_points = metrics.get('data_points', 0)
            
            if data_points > 0:
                st.success(f"✅ {data_points} data points loaded")
            else:
                st.warning("⚠️ No data available")
        except:
            st.warning("⚠️ No data available")
        
        st.markdown("---")
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Artemis Health Dashboard**
            
            A comprehensive livestock health monitoring system that provides:
            - Real-time sensor data analysis
            - Behavioral state tracking
            - Temperature monitoring
            - Alert management
            - Health trend analysis
            
            Version: 1.0.0
            """)


def handle_auto_refresh():
    """Handle automatic data refresh."""
    if st.session_state.auto_refresh_enabled:
        config = st.session_state.config
        refresh_interval = config.get('dashboard', {}).get('auto_refresh_interval_seconds', 60)
        
        # Check if refresh interval has passed
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
        
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Display countdown to next refresh
        time_remaining = int(refresh_interval - time_since_refresh)
        if time_remaining > 0:
            st.sidebar.caption(f"Next refresh in: {time_remaining}s")


def main():
    """Main application entry point."""
    # Load configuration
    config = load_config()
    
    # Setup page
    setup_page_config(config)
    
    # Initialize session state
    initialize_session_state()
    
    # Load custom CSS
    load_custom_css()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area - Home/Overview page
    st.title("📊 Overview Dashboard")
    st.markdown("*Real-time livestock health monitoring and system status*")
    st.markdown("---")
    
    # Show loading state
    with st.spinner("Loading dashboard data..."):
        try:
            # Get latest metrics
            data_loader = st.session_state.data_loader
            metrics = data_loader.get_latest_metrics()
            alert_summary = data_loader.get_alert_summary()
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                temp = metrics.get('temperature')
                if temp:
                    st.metric(
                        label="🌡️ Temperature",
                        value=f"{temp:.1f}°C",
                        delta=None,
                    )
                else:
                    st.metric(label="🌡️ Temperature", value="N/A")
            
            with col2:
                activity = metrics.get('activity_level', 0)
                st.metric(
                    label="📊 Activity Level",
                    value=f"{activity:.2f}",
                    delta=None,
                )
            
            with col3:
                current_state = metrics.get('current_state', 'unknown')
                st.metric(
                    label="🐮 Current State",
                    value=current_state.capitalize(),
                    delta=None,
                )
            
            with col4:
                active_alerts = alert_summary.get('active_alerts', 0)
                st.metric(
                    label="🚨 Active Alerts",
                    value=str(active_alerts),
                    delta=None,
                )
            
            st.markdown("---")
            
            # Recent data section
            st.subheader("📈 Recent Sensor Data")
            
            sensor_data = data_loader.load_sensor_data(
                time_range_hours=st.session_state.selected_time_range,
                max_rows=100
            )
            
            if not sensor_data.empty:
                st.info(f"✅ Displaying {len(sensor_data)} data points from the last {st.session_state.selected_time_range} hour(s)")
                
                # Display data table
                with st.expander("View Raw Data", expanded=False):
                    st.dataframe(
                        sensor_data.tail(20),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.warning("⚠️ No sensor data available. Please check data sources.")
            
            # Recent alerts section
            st.markdown("---")
            st.subheader("🚨 Recent Alerts")
            
            recent_alerts = data_loader.load_alerts(max_alerts=5)
            
            if recent_alerts:
                for alert in recent_alerts:
                    severity = alert.get('severity', 'unknown')
                    alert_type = alert.get('malfunction_type', 'Unknown')
                    detection_time = alert.get('detection_time', 'Unknown')
                    
                    with st.container():
                        if severity == 'critical':
                            st.error(f"**{alert_type}** - {detection_time}")
                        elif severity == 'high':
                            st.warning(f"**{alert_type}** - {detection_time}")
                        else:
                            st.info(f"**{alert_type}** - {detection_time}")
                
                st.info("ℹ️ View all alerts on the Alerts Dashboard page")
            else:
                st.success("✅ No recent alerts. System operating normally.")
            
        except Exception as e:
            st.error(f"❌ Error loading dashboard data: {str(e)}")
            st.info("💡 Please check that data files are available in the configured locations.")
    
    # Navigation info
    st.markdown("---")
    st.info("""
    📌 **Navigation**: Use the sidebar to access different dashboard pages:
    - **Overview** (current page): Real-time metrics and system status
    - **Behavioral Analysis**: State timeline and activity patterns  
    - **Temperature Monitoring**: Circadian rhythm and temperature trends
    - **Alerts Dashboard**: Active alerts and alert history
    - **Health Trends**: Multi-day analysis and health scores
    """)
    
    # Handle auto-refresh
    handle_auto_refresh()


if __name__ == "__main__":
    main()
