"""
Artemis Health Dashboard - Main Application
===========================================
Multi-page Streamlit dashboard for cattle health monitoring system.
"""

import streamlit as st
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from config import PAGE_CONFIG


def main():
    """
    Main dashboard application entry point.
    """
    # Configure page
    st.set_page_config(
        page_title=PAGE_CONFIG['page_title'],
        page_icon=PAGE_CONFIG['page_icon'],
        layout=PAGE_CONFIG['layout'],
        initial_sidebar_state=PAGE_CONFIG['initial_sidebar_state'],
    )
    
    # Main page content
    st.title("🐄 Artemis Health - Livestock Monitoring Dashboard")
    
    st.markdown("""
    ## Welcome to Artemis Health
    
    A comprehensive livestock health monitoring and analysis system that uses neck-mounted sensors 
    to track animal behavior, physiology, and health status in real-time.
    
    ### Available Dashboards
    
    Navigate using the sidebar to access different visualization and monitoring tools:
    
    #### 📈 Behavioral Timeline
    Interactive timeline showing behavioral states (lying, standing, walking, ruminating, feeding) 
    over time with detailed statistics and export capabilities.
    
    - **Features:**
      - Time range selector (24h, 7d, 30d views)
      - Interactive zoom and pan controls
      - Hover tooltips with detailed state information
      - Duration statistics and transition counts
      - Export timeline data to CSV
    
    ### System Overview
    
    The Artemis Health system processes data through three intelligent layers:
    
    1. **Physical Behavior Layer** - Recognizes posture, activity patterns, and specific behaviors
    2. **Physiological Analysis Layer** - Monitors body temperature patterns and circadian rhythms
    3. **Health Intelligence Layer** - Provides automated health scoring and early warning alerts
    
    ### Sensor Data Collected
    
    - **Temperature (°C)** - Body temperature for fever, heat stress, and estrus detection
    - **3-Axis Accelerometer (Fxa, Mya, Rza)** - Movement and posture tracking
    - **3-Axis Gyroscope (Sxg, Lyg, Dzg)** - Head orientation and rotation patterns
    
    ### Getting Started
    
    1. Select a dashboard from the sidebar navigation
    2. Choose a cow ID and time range
    3. Explore the interactive visualizations
    4. Export data for further analysis
    
    ### Data Status
    """)
    
    # Display data connection status
    import os
    from utils.db_connection import get_database_connection
    
    try:
        conn = get_database_connection()
        if conn is not None:
            st.success("✅ **Database Connection:** Active")
            if hasattr(conn, 'close'):
                conn.close()
        else:
            use_mock = os.getenv('USE_MOCK_DATA', 'false').lower() == 'true'
            if use_mock:
                st.info("ℹ️ **Data Mode:** Using simulated mock data for demonstration")
            else:
                st.warning("⚠️ **Database Connection:** Not configured. Using mock data.")
    except Exception as e:
        st.error(f"❌ **Database Connection Error:** {e}")
    
    # System information
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Real-Time Monitoring")
        st.markdown("Track behavioral states and physiological metrics as they happen.")
    
    with col2:
        st.markdown("### 🔔 Intelligent Alerts")
        st.markdown("Automated health alerts for fever, heat stress, and abnormal behavior.")
    
    with col3:
        st.markdown("### 📈 Trend Analysis")
        st.markdown("Multi-day trends for comprehensive health assessment.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p><b>Artemis Health</b> - Livestock Health Monitoring System</p>
        <p><small>Powered by TimescaleDB, Streamlit, and Plotly</small></p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
