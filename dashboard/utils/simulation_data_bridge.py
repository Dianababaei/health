"""
Simulation Data Bridge - Connect Simulation to Dashboard Pages

This utility allows dashboard pages to use simulation data from the
Simulation Testing page (99_Simulation_Testing.py) when real data is not available.

Usage in any dashboard page:
    from dashboard.utils.simulation_data_bridge import get_data_source, is_using_simulation

    # Get data (automatically uses simulation if available, otherwise real data)
    df = get_data_source()

    # Check if using simulation
    if is_using_simulation():
        st.info("📊 Using simulation data")
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def is_using_simulation() -> bool:
    """
    Check if simulation data is available and should be used.

    Returns:
        True if simulation data exists in session state
    """
    return 'simulation_data' in st.session_state and st.session_state.simulation_data is not None


def get_simulation_sensor_data() -> Optional[pd.DataFrame]:
    """
    Get sensor data from simulation session state.

    Returns:
        DataFrame with sensor data, or None if not available
    """
    if not is_using_simulation():
        return None

    sim_data = st.session_state.simulation_data
    return sim_data.get('df', None)


def get_simulation_alerts() -> List[Dict[str, Any]]:
    """
    Get alerts from simulation session state.

    Returns:
        List of alert dictionaries, or empty list if not available
    """
    if not is_using_simulation():
        return []

    sim_data = st.session_state.simulation_data
    return sim_data.get('alerts', [])


def get_simulation_trend_report() -> Optional[Any]:
    """
    Get health trend report from simulation session state.

    Returns:
        HealthTrendReport object, or None if not available
    """
    if not is_using_simulation():
        return None

    sim_data = st.session_state.simulation_data
    return sim_data.get('trend_report', None)


def get_simulation_cow_id() -> Optional[str]:
    """
    Get cow ID from simulation session state.

    Returns:
        Cow ID string, or None if not available
    """
    if not is_using_simulation():
        return None

    sim_data = st.session_state.simulation_data
    return sim_data.get('cow_id', None)


def get_simulation_baseline_temp() -> Optional[float]:
    """
    Get baseline temperature from simulation session state.

    Returns:
        Baseline temperature float, or None if not available
    """
    if not is_using_simulation():
        return None

    sim_data = st.session_state.simulation_data
    return sim_data.get('baseline_temp', 38.5)


def get_data_source(data_loader=None, prefer_simulation: bool = True) -> Optional[pd.DataFrame]:
    """
    Get sensor data from simulation or real source.

    This is the main function to use in dashboard pages. It automatically
    chooses between simulation and real data based on availability.

    Args:
        data_loader: DataLoader instance for loading real data
        prefer_simulation: If True and simulation data exists, use it.
                          If False, always use real data.

    Returns:
        DataFrame with sensor data, or None if no data available
    """
    # Try simulation first if preferred
    if prefer_simulation and is_using_simulation():
        logger.info("Using simulation data")
        return get_simulation_sensor_data()

    # Otherwise use real data
    if data_loader is not None:
        try:
            logger.info("Using real data from data_loader")
            return data_loader.load_sensor_data()
        except Exception as e:
            logger.error(f"Error loading real data: {e}")

    # Fallback to simulation if real data failed
    if is_using_simulation():
        logger.warning("Real data unavailable, falling back to simulation")
        return get_simulation_sensor_data()

    return None


def get_alerts_source(data_loader=None, prefer_simulation: bool = True) -> List[Dict[str, Any]]:
    """
    Get alerts from simulation or real source.

    Args:
        data_loader: DataLoader instance for loading real alerts
        prefer_simulation: If True and simulation alerts exist, use them

    Returns:
        List of alert dictionaries
    """
    # Try simulation first if preferred
    if prefer_simulation and is_using_simulation():
        logger.info("Using simulation alerts")
        return get_simulation_alerts()

    # Otherwise use real alerts
    if data_loader is not None:
        try:
            logger.info("Using real alerts from data_loader")
            return data_loader.load_alerts()
        except Exception as e:
            logger.error(f"Error loading real alerts: {e}")

    # Fallback to simulation if real data failed
    if is_using_simulation():
        logger.warning("Real alerts unavailable, falling back to simulation")
        return get_simulation_alerts()

    return []


def render_data_source_indicator():
    """
    Render a small indicator showing whether using simulation or real data.

    Use this at the top of dashboard pages to inform users.
    """
    if is_using_simulation():
        sim_data = st.session_state.simulation_data
        cow_id = sim_data.get('cow_id', 'Unknown')
        duration_days = len(sim_data.get('df', [])) / 1440  # 1440 minutes per day

        st.info(
            f"📊 **Using Simulation Data**: Cow `{cow_id}` ({duration_days:.1f} days)\n\n"
            f"*Go to Simulation Testing page to generate new data or use real data*",
            icon="🧪"
        )
    else:
        st.success("📡 **Using Real Sensor Data**", icon="✅")


def clear_simulation_data():
    """
    Clear simulation data from session state.

    Useful for forcing pages to use real data instead.
    """
    if 'simulation_data' in st.session_state:
        del st.session_state.simulation_data
        logger.info("Simulation data cleared from session state")


def get_latest_reading_from_simulation() -> Optional[Dict[str, Any]]:
    """
    Get the latest sensor reading from simulation data.

    This mimics the behavior of get_latest_sensor_readings() for real data.

    Returns:
        Dictionary with latest sensor reading fields, or None
    """
    df = get_simulation_sensor_data()
    if df is None or len(df) == 0:
        return None

    # Get last row
    latest = df.iloc[-1]

    # Convert to dictionary matching expected format
    reading = {
        'timestamp': latest['timestamp'],
        'temperature': latest['temperature'],
        'fxa': latest['fxa'],
        'mya': latest['mya'],
        'rza': latest['rza'],
        'sxg': latest['sxg'],
        'lyg': latest['lyg'],
        'dzg': latest['dzg'],
        'behavioral_state': latest['state'],
        'freshness_seconds': 0,  # Simulation data is "fresh"
        'is_stale': False,
    }

    return reading


def get_time_range_from_simulation(hours: int = 24) -> Optional[pd.DataFrame]:
    """
    Get sensor data for a specific time range from simulation.

    Args:
        hours: Number of hours to retrieve (from end of simulation)

    Returns:
        DataFrame filtered to time range, or None
    """
    df = get_simulation_sensor_data()
    if df is None or len(df) == 0:
        return None

    # Calculate how many minutes we need
    minutes_needed = hours * 60

    # Get last N rows (1 row per minute)
    if len(df) > minutes_needed:
        return df.tail(minutes_needed).copy()
    else:
        return df.copy()


def export_simulation_to_csv(output_path: str) -> bool:
    """
    Export current simulation data to CSV file.

    This allows you to save simulation data and load it as "real" data later.

    Args:
        output_path: Path where CSV should be saved

    Returns:
        True if successful, False otherwise
    """
    try:
        df = get_simulation_sensor_data()
        if df is None:
            logger.error("No simulation data to export")
            return False

        df.to_csv(output_path, index=False)
        logger.info(f"Exported simulation data to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting simulation data: {e}")
        return False


def get_simulation_metadata() -> Optional[Dict[str, Any]]:
    """
    Get metadata about the current simulation.

    Returns:
        Dictionary with simulation metadata, or None
    """
    if not is_using_simulation():
        return None

    sim_data = st.session_state.simulation_data
    df = sim_data.get('df', pd.DataFrame())

    metadata = {
        'cow_id': sim_data.get('cow_id', 'Unknown'),
        'baseline_temp': sim_data.get('baseline_temp', 38.5),
        'total_samples': len(df),
        'duration_days': len(df) / 1440 if len(df) > 0 else 0,
        'start_time': df['timestamp'].min() if 'timestamp' in df.columns and len(df) > 0 else None,
        'end_time': df['timestamp'].max() if 'timestamp' in df.columns and len(df) > 0 else None,
        'num_alerts': len(sim_data.get('alerts', [])),
        'conditions_injected': sim_data.get('conditions', []),
    }

    return metadata
