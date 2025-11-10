"""
Dashboard Configuration
======================
Configuration constants for the Artemis Health dashboard including
color schemes, time ranges, and chart settings.
"""

import os
from datetime import timedelta
from typing import Dict, List, Tuple

# ============================================================================
# Behavioral State Color Scheme
# ============================================================================
# Consistent color mapping for all visualizations across the dashboard
# Based on technical specifications for behavioral timeline visualization

BEHAVIOR_COLORS: Dict[str, str] = {
    'lying': '#4A90E2',      # Blue
    'standing': '#7ED321',   # Green
    'walking': '#F5A623',    # Orange
    'ruminating': '#BD10E0', # Purple
    'feeding': '#F8E71C',    # Yellow
    'unknown': '#95A5A6',    # Gray (for uncertain classifications)
}

# Display names for behavioral states
BEHAVIOR_LABELS: Dict[str, str] = {
    'lying': 'Lying',
    'standing': 'Standing',
    'walking': 'Walking',
    'ruminating': 'Ruminating',
    'feeding': 'Feeding',
    'unknown': 'Unknown',
}

# Order for displaying states in legends and statistics
BEHAVIOR_DISPLAY_ORDER: List[str] = [
    'lying',
    'standing',
    'walking',
    'ruminating',
    'feeding',
    'unknown',
]

# ============================================================================
# Time Range Configuration
# ============================================================================

# Available time range options for timeline selector
TIME_RANGES: Dict[str, Dict] = {
    'Last 24 Hours': {
        'hours': 24,
        'label': 'Last 24 Hours',
        'description': 'Hourly view of the last 24 hours',
        'granularity': 'minute',  # Data granularity: minute-level
    },
    'Last 7 Days': {
        'hours': 24 * 7,
        'label': 'Last 7 Days',
        'description': 'Daily view of the last week',
        'granularity': 'minute',  # Data granularity: minute-level
    },
    'Last 30 Days': {
        'hours': 24 * 30,
        'label': 'Last 30 Days',
        'description': 'Weekly view of the last month',
        'granularity': 'hour',  # Aggregate to hourly for performance
    },
}

# Default time range selection
DEFAULT_TIME_RANGE = 'Last 24 Hours'

# ============================================================================
# Chart Configuration
# ============================================================================

# Plotly timeline chart settings
TIMELINE_CHART_CONFIG: Dict = {
    'height': 400,  # Chart height in pixels
    'margin': {'l': 100, 'r': 20, 't': 50, 'b': 100},
    'showlegend': True,
    'legend': {
        'orientation': 'h',
        'yanchor': 'bottom',
        'y': -0.3,
        'xanchor': 'center',
        'x': 0.5,
    },
    'xaxis': {
        'title': 'Time',
        'showgrid': True,
        'gridcolor': '#E5E5E5',
    },
    'yaxis': {
        'title': 'Behavioral State',
        'showgrid': False,
    },
    'hovermode': 'closest',
    'plot_bgcolor': '#FFFFFF',
    'paper_bgcolor': '#FFFFFF',
}

# Interactive features configuration
CHART_INTERACTION_CONFIG: Dict = {
    'displayModeBar': True,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'displaylogo': False,
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'behavior_timeline',
        'height': 800,
        'width': 1400,
        'scale': 2,
    },
}

# Performance thresholds
PERFORMANCE_THRESHOLDS: Dict = {
    'max_data_points': 10000,  # Maximum data points before aggregation
    'aggregate_threshold_days': 30,  # Aggregate to hourly after this many days
    'query_timeout_seconds': 30,  # Database query timeout
    'render_timeout_seconds': 5,  # Chart rendering timeout
}

# ============================================================================
# Database Configuration
# ============================================================================

# Database connection settings (from environment variables)
DATABASE_CONFIG: Dict = {
    'url': os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/artemis_health'),
    'pool_size': int(os.getenv('DB_POOL_SIZE', '5')),
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '10')),
    'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
    'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '3600')),
}

# ============================================================================
# Statistics Configuration
# ============================================================================

# Minimum duration to consider a state as significant (in minutes)
MIN_STATE_DURATION = 1

# Minimum duration to highlight as "longest continuous period" (in minutes)
MIN_HIGHLIGHT_DURATION = 30

# Format for duration display
DURATION_FORMAT = '{hours}h {minutes}m'

# ============================================================================
# Export Configuration
# ============================================================================

# CSV export settings
EXPORT_CONFIG: Dict = {
    'filename_template': 'behavior_timeline_{cow_id}_{start_date}_{end_date}.csv',
    'date_format': '%Y%m%d',
    'timestamp_format': '%Y-%m-%d %H:%M:%S',
    'include_confidence': True,
    'include_duration': True,
}

# ============================================================================
# UI Configuration
# ============================================================================

# Streamlit page configuration
PAGE_CONFIG: Dict = {
    'page_title': 'Artemis Health - Behavior Timeline',
    'page_icon': '🐄',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

# Dashboard refresh interval (in seconds)
DASHBOARD_REFRESH_INTERVAL = int(os.getenv('DASHBOARD_REFRESH_INTERVAL', '60'))

# Maximum number of animals to display in dropdown
MAX_ANIMALS_IN_DROPDOWN = int(os.getenv('MAX_DISPLAY_ANIMALS', '50'))

# ============================================================================
# Helper Functions
# ============================================================================

def get_time_range_timedelta(time_range_key: str) -> timedelta:
    """
    Convert time range key to timedelta object.
    
    Args:
        time_range_key: Key from TIME_RANGES dict
        
    Returns:
        timedelta object representing the time range
    """
    if time_range_key not in TIME_RANGES:
        time_range_key = DEFAULT_TIME_RANGE
    
    hours = TIME_RANGES[time_range_key]['hours']
    return timedelta(hours=hours)


def get_state_color(state: str) -> str:
    """
    Get color for a behavioral state.
    
    Args:
        state: Behavioral state name
        
    Returns:
        Hex color code
    """
    return BEHAVIOR_COLORS.get(state.lower(), BEHAVIOR_COLORS['unknown'])


def get_state_label(state: str) -> str:
    """
    Get display label for a behavioral state.
    
    Args:
        state: Behavioral state name
        
    Returns:
        Display label
    """
    return BEHAVIOR_LABELS.get(state.lower(), state.capitalize())


def format_duration(minutes: float) -> str:
    """
    Format duration in minutes to human-readable string.
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        Formatted duration string (e.g., "2h 30m")
    """
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if hours > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{mins}m"
