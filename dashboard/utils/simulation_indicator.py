"""
Simulation Indicator Utility

Shows a visual indicator when pages are using simulation data.
This persists across page navigation since it checks the actual files.
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime


def check_simulation_active():
    """
    Check if simulation data exists in the data/simulation directory.

    Returns:
        tuple: (is_active: bool, metadata: dict or None)
    """
    sim_dir = Path(__file__).parent.parent.parent / 'data' / 'simulation'

    if not sim_dir.exists():
        return False, None

    # Look for sensor data files
    sensor_files = list(sim_dir.glob('*_sensor_data.csv'))

    if not sensor_files:
        return False, None

    # Get most recent file
    latest_file = sorted(sensor_files, key=lambda x: x.stat().st_mtime)[-1]

    # Extract cow ID from filename
    cow_id = latest_file.stem.replace('_sensor_data', '')

    # Try to load metadata
    metadata_file = sim_dir / f'{cow_id}_metadata.json'
    metadata = None

    if metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except:
            pass

    if metadata is None:
        # Basic metadata from file
        metadata = {
            'cow_id': cow_id,
            'file': str(latest_file),
            'generated_at': datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
        }

    return True, metadata


def render_simulation_indicator(location="top"):
    """
    Render a visual indicator showing simulation mode is active.

    Args:
        location: Where to show indicator ("top", "sidebar", "both")
    """
    is_active, metadata = check_simulation_active()

    if not is_active:
        # Show that real data mode is active
        if location in ["top", "both"]:
            st.info("📡 **Real Data Mode** - Waiting for sensor data", icon="ℹ️")
        return False

    # Simulation is active
    cow_id = metadata.get('cow_id', 'Unknown')
    duration = metadata.get('duration_days', '?')
    num_samples = metadata.get('total_samples', '?')
    generated_at = metadata.get('generated_at', 'Unknown')

    # Parse datetime for display
    try:
        gen_time = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
        time_ago = datetime.now() - gen_time
        if time_ago.days > 0:
            time_str = f"{time_ago.days} days ago"
        elif time_ago.seconds > 3600:
            time_str = f"{time_ago.seconds // 3600} hours ago"
        else:
            time_str = f"{time_ago.seconds // 60} minutes ago"
    except:
        time_str = "recently"

    # Show in main area
    if location in ["top", "both"]:
        st.success(
            f"🧪 **Simulation Mode Active** - Using simulated cow data\n\n"
            f"**Cow**: `{cow_id}` | **Duration**: {duration} days | **Samples**: {num_samples:,} | **Generated**: {time_str}\n\n"
            f"*Go to 🧪 Simulation Testing page to generate new data or disable simulation mode*",
            icon="✅"
        )

    # Show in sidebar
    if location in ["sidebar", "both"]:
        with st.sidebar:
            st.success(f"🧪 **Simulation Active**\n\nCow: `{cow_id}`")

    return True


def get_simulation_metadata():
    """
    Get metadata about active simulation.

    Returns:
        dict or None: Metadata dictionary if simulation is active
    """
    is_active, metadata = check_simulation_active()
    return metadata if is_active else None


def clear_simulation_files():
    """
    Clear all simulation data files.

    This disables simulation mode and returns to real data mode.
    """
    sim_dir = Path(__file__).parent.parent.parent / 'data' / 'simulation'

    if not sim_dir.exists():
        return 0

    files_removed = 0
    for file in sim_dir.glob('*'):
        try:
            file.unlink()
            files_removed += 1
        except:
            pass

    return files_removed


def render_clear_button():
    """
    Render a button to clear simulation data and return to real data mode.
    """
    is_active, metadata = check_simulation_active()

    if is_active:
        if st.button("🗑️ Clear Simulation Data (Return to Real Data Mode)", type="secondary"):
            count = clear_simulation_files()

            # Also clear session state
            if 'simulation_data' in st.session_state:
                del st.session_state.simulation_data
            if 'data_loader' in st.session_state:
                del st.session_state.data_loader  # Force reload

            st.success(f"✅ Cleared {count} simulation files. Reloading...")
            st.rerun()
