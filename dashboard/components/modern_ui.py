"""
Modern UI Component Library

Reusable, consistent components following 2024 UX best practices.
Clean, minimal, action-oriented design.
"""

import streamlit as st
from typing import Optional, Callable, List, Dict, Any


# ============================================================================
# COLOR SYSTEM (2024 Best Practices)
# ============================================================================

COLORS = {
    # Status colors
    'excellent': '#2ecc71',    # Green
    'good': '#27ae60',         # Dark green
    'fair': '#f39c12',         # Orange
    'warning': '#f39c12',      # Orange
    'poor': '#e67e22',         # Dark orange
    'critical': '#e74c3c',     # Red
    'neutral': '#95a5a6',      # Gray

    # Background
    'bg_primary': '#ffffff',
    'bg_secondary': '#f8f9fa',
    'bg_card': '#ffffff',

    # Text
    'text_primary': '#2c3e50',
    'text_secondary': '#7f8c8d',
    'text_white': '#ffffff',
}


# ============================================================================
# STATUS CARD - Primary component for metrics
# ============================================================================

def render_status_card(
    title: str,
    value: str,
    status: str = "neutral",
    icon: str = "📊",
    subtitle: Optional[str] = None,
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
):
    """
    Modern status card with optional action button.

    Args:
        title: Card title
        value: Main value (large text)
        status: Status color (excellent/good/fair/warning/critical/neutral)
        icon: Emoji icon
        subtitle: Optional subtitle text
        action_label: Optional button label
        action_callback: Optional button callback

    Example:
        render_status_card(
            title="Herd Health",
            value="87/100",
            status="good",
            icon="🐮",
            subtitle="Good",
            action_label="View Details",
            action_callback=lambda: st.write("Details")
        )
    """
    color = COLORS.get(status, COLORS['neutral'])

    # Card container
    st.markdown(f"""
    <div style="
        background: {COLORS['bg_card']};
        padding: 24px;
        border-radius: 12px;
        border-left: 4px solid {color};
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    ">
        <div style="
            color: {COLORS['text_secondary']};
            font-size: 13px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        ">{icon} {title}</div>

        <div style="
            font-size: 36px;
            font-weight: 700;
            color: {COLORS['text_primary']};
            margin: 12px 0;
            line-height: 1;
        ">{value}</div>

        {f'<div style="color: {color}; font-size: 14px; font-weight: 600; text-transform: uppercase;">{subtitle}</div>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)

    # Action button if provided
    if action_label and action_callback:
        if st.button(action_label, key=f"action_{title}_{value}", use_container_width=True):
            action_callback()


# ============================================================================
# MINI METRIC - Compact metric for grids
# ============================================================================

def render_mini_metric(
    label: str,
    value: str,
    status: str = "neutral",
    delta: Optional[str] = None
):
    """
    Compact metric card for grid layouts.

    Args:
        label: Metric label
        value: Metric value
        status: Status indicator (good/warning/critical/neutral)
        delta: Optional change indicator
    """
    icons = {
        'excellent': '✅',
        'good': '✅',
        'fair': '⚠️',
        'warning': '⚠️',
        'poor': '🔴',
        'critical': '🔴',
        'neutral': 'ℹ️'
    }

    icon = icons.get(status, icons['neutral'])
    color = COLORS.get(status, COLORS['neutral'])

    st.markdown(f"""
    <div style="
        background: {COLORS['bg_card']};
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    ">
        <div style="font-size: 28px; margin-bottom: 8px;">{icon}</div>
        <div style="
            font-size: 24px;
            font-weight: 700;
            color: {color};
            margin-bottom: 4px;
        ">{value}</div>
        <div style="
            font-size: 12px;
            color: {COLORS['text_secondary']};
            font-weight: 500;
        ">{label}</div>
        {f'<div style="font-size: 11px; color: {COLORS["text_secondary"]}; margin-top: 4px;">{delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# ALERT BANNER - Prominent alerts with actions
# ============================================================================

def render_alert_banner(
    message: str,
    severity: str = "warning",
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None,
    dismissible: bool = False
):
    """
    Prominent alert banner with action button.

    Args:
        message: Alert message
        severity: Severity level (critical/warning/info)
        action_label: Optional action button text
        action_callback: Optional action callback
        dismissible: Whether alert can be dismissed
    """
    colors = {
        'critical': ('#e74c3c', '#ffffff'),
        'warning': ('#f39c12', '#ffffff'),
        'info': ('#3498db', '#ffffff'),
        'success': ('#2ecc71', '#ffffff')
    }

    bg_color, text_color = colors.get(severity, colors['info'])

    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown(f"""
        <div style="
            background: {bg_color};
            color: {text_color};
            padding: 16px 20px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        ">{message}</div>
        """, unsafe_allow_html=True)

    with col2:
        if action_label and action_callback:
            if st.button(action_label, type="primary", use_container_width=True):
                action_callback()


# ============================================================================
# ACTION LIST - List of items requiring action
# ============================================================================

def render_action_item(
    title: str,
    description: str,
    severity: str = "warning",
    action_label: str = "View",
    action_callback: Optional[Callable] = None,
    metadata: Optional[str] = None
):
    """
    Single action item in a list.

    Args:
        title: Item title
        description: Item description
        severity: Severity (critical/warning/info)
        action_label: Action button label
        action_callback: Action callback
        metadata: Optional metadata (e.g., timestamp)
    """
    severity_icons = {
        'critical': '🔴',
        'warning': '🟡',
        'info': 'ℹ️',
        'success': '✅'
    }

    icon = severity_icons.get(severity, severity_icons['info'])

    col1, col2 = st.columns([5, 1])

    with col1:
        st.markdown(f"""
        <div style="
            padding: 16px;
            background: {COLORS['bg_card']};
            border-radius: 8px;
            border-left: 3px solid {COLORS.get(severity, COLORS['neutral'])};
            margin-bottom: 12px;
        ">
            <div style="
                font-weight: 600;
                color: {COLORS['text_primary']};
                margin-bottom: 4px;
            ">{icon} {title}</div>
            <div style="
                font-size: 13px;
                color: {COLORS['text_secondary']};
            ">{description}</div>
            {f'<div style="font-size: 11px; color: {COLORS["text_secondary"]}; margin-top: 8px;">{metadata}</div>' if metadata else ''}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if action_callback:
            if st.button(action_label, key=f"action_{title}", use_container_width=True):
                action_callback()


# ============================================================================
# SECTION HEADER - Consistent section headers
# ============================================================================

def render_section_header(
    title: str,
    subtitle: Optional[str] = None,
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
):
    """
    Consistent section header with optional action.

    Args:
        title: Section title
        subtitle: Optional subtitle
        action_label: Optional action button
        action_callback: Optional action callback
    """
    col1, col2 = st.columns([4, 1])

    with col1:
        st.markdown(f"""
        <div style="margin: 24px 0 16px 0;">
            <h2 style="
                font-size: 24px;
                font-weight: 700;
                color: {COLORS['text_primary']};
                margin: 0;
            ">{title}</h2>
            {f'<p style="color: {COLORS["text_secondary"]}; font-size: 14px; margin: 4px 0 0 0;">{subtitle}</p>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if action_label and action_callback:
            st.write("")  # Spacing
            if st.button(action_label, key=f"header_{title}"):
                action_callback()


# ============================================================================
# EMPTY STATE - When no data available
# ============================================================================

def render_empty_state(
    icon: str = "📭",
    title: str = "No Data Available",
    message: str = "There's no data to display yet.",
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
):
    """
    Empty state placeholder.

    Args:
        icon: Emoji icon
        title: Empty state title
        message: Description message
        action_label: Optional action button
        action_callback: Optional action callback
    """
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 60px 20px;
        background: {COLORS['bg_secondary']};
        border-radius: 12px;
        margin: 20px 0;
    ">
        <div style="font-size: 64px; margin-bottom: 16px;">{icon}</div>
        <h3 style="
            color: {COLORS['text_primary']};
            font-size: 20px;
            margin-bottom: 8px;
        ">{title}</h3>
        <p style="
            color: {COLORS['text_secondary']};
            font-size: 14px;
            max-width: 400px;
            margin: 0 auto;
        ">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label, type="primary", use_container_width=True):
                action_callback()


# ============================================================================
# STAT GRID - Grid of statistics
# ============================================================================

def render_stat_grid(stats: List[Dict[str, Any]], columns: int = 4):
    """
    Grid of statistics.

    Args:
        stats: List of stat dictionaries with keys: label, value, status, delta
        columns: Number of columns

    Example:
        render_stat_grid([
            {'label': 'Healthy', 'value': '45', 'status': 'good'},
            {'label': 'At Risk', 'value': '3', 'status': 'warning'},
        ], columns=2)
    """
    cols = st.columns(columns)

    for i, stat in enumerate(stats):
        with cols[i % columns]:
            render_mini_metric(
                label=stat.get('label', ''),
                value=stat.get('value', ''),
                status=stat.get('status', 'neutral'),
                delta=stat.get('delta', None)
            )


# ============================================================================
# LOADING STATE - Consistent loading indicator
# ============================================================================

def render_loading_state(message: str = "Loading..."):
    """
    Consistent loading state.

    Args:
        message: Loading message
    """
    st.markdown(f"""
    <div style="
        text-align: center;
        padding: 40px;
        color: {COLORS['text_secondary']};
    ">
        <div style="font-size: 14px;">{message}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# HEALTH SCORE GAUGE - Visual health score display
# ============================================================================

def render_health_score_gauge(score: float, size: str = "large"):
    """
    Visual health score gauge.

    Args:
        score: Score 0-100
        size: Size (large/medium/small)
    """
    # Determine color based on score
    if score >= 90:
        color = COLORS['excellent']
        status = "EXCELLENT"
    elif score >= 75:
        color = COLORS['good']
        status = "GOOD"
    elif score >= 60:
        color = COLORS['fair']
        status = "FAIR"
    elif score >= 40:
        color = COLORS['poor']
        status = "POOR"
    else:
        color = COLORS['critical']
        status = "CRITICAL"

    sizes = {
        'large': ('120px', '48px', '18px'),
        'medium': ('80px', '32px', '14px'),
        'small': ('60px', '24px', '12px')
    }

    gauge_size, score_size, status_size = sizes.get(size, sizes['large'])

    st.markdown(f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
    ">
        <div style="
            width: {gauge_size};
            height: {gauge_size};
            border-radius: 50%;
            background: conic-gradient(
                {color} 0deg {score * 3.6}deg,
                {COLORS['bg_secondary']} {score * 3.6}deg 360deg
            );
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        ">
            <div style="
                width: calc(100% - 12px);
                height: calc(100% - 12px);
                border-radius: 50%;
                background: {COLORS['bg_card']};
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
            ">
                <div style="
                    font-size: {score_size};
                    font-weight: 700;
                    color: {color};
                ">{score:.0f}</div>
                <div style="
                    font-size: 12px;
                    color: {COLORS['text_secondary']};
                ">/100</div>
            </div>
        </div>
        <div style="
            margin-top: 12px;
            font-size: {status_size};
            font-weight: 700;
            color: {color};
            text-transform: uppercase;
            letter-spacing: 1px;
        ">{status}</div>
    </div>
    """, unsafe_allow_html=True)
