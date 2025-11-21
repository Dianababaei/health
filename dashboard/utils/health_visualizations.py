"""
Health Visualizations
====================
Visualization utilities for health score gauge and trend charts.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime


def create_health_gauge(
    current_score: float,
    baseline_score: Optional[float] = None,
    show_delta: bool = True
) -> go.Figure:
    """
    Create a gauge chart displaying health score (0-100) with color zones.
    
    Color zones:
    - Red (0-40): Critical/Poor health
    - Yellow (40-70): Fair/Monitor closely
    - Green (70-100): Good/Excellent health
    
    Args:
        current_score: Current health score (0-100)
        baseline_score: Baseline health score for comparison (optional)
        show_delta: Whether to show delta from baseline
        
    Returns:
        Plotly Figure object
    """
    # Define color zones
    color_zones = [
        {'range': [0, 40], 'color': '#FF4444'},      # Red
        {'range': [40, 70], 'color': '#FFB347'},     # Yellow/Orange
        {'range': [70, 100], 'color': '#4CAF50'},    # Green
    ]
    
    # Determine gauge color based on current score
    if current_score < 40:
        gauge_color = '#FF4444'
        status_text = 'Poor'
    elif current_score < 70:
        gauge_color = '#FFB347'
        status_text = 'Fair'
    else:
        gauge_color = '#4CAF50'
        status_text = 'Good'
    
    # Calculate delta if baseline provided
    delta_text = ""
    if baseline_score is not None and show_delta:
        delta = current_score - baseline_score
        if delta > 0:
            delta_text = f"<br><span style='color: green; font-size: 14px;'>▲ +{delta:.1f} from baseline</span>"
        elif delta < 0:
            delta_text = f"<br><span style='color: red; font-size: 14px;'>▼ {delta:.1f} from baseline</span>"
        else:
            delta_text = f"<br><span style='color: gray; font-size: 14px;'>● No change from baseline</span>"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>Health Score</b><br><span style='font-size: 16px;'>{status_text}</span>{delta_text}",
            'font': {'size': 20}
        },
        number={'suffix': "/100", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#FFE5E5'},      # Light red
                {'range': [40, 70], 'color': '#FFF4E5'},     # Light yellow
                {'range': [70, 100], 'color': '#E8F5E9'},    # Light green
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 40 if current_score < 40 else (70 if current_score < 70 else 100)
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig


def create_health_history_chart(
    health_data: pd.DataFrame,
    time_range_label: str = "Historical View",
    show_baseline: bool = True,
    baseline_score: Optional[float] = None
) -> go.Figure:
    """
    Create an interactive line chart showing health score history.
    
    Args:
        health_data: DataFrame with 'timestamp' and 'health_score' columns
        time_range_label: Label for the time range
        show_baseline: Whether to show baseline reference line
        baseline_score: Baseline health score value
        
    Returns:
        Plotly Figure object
    """
    if health_data.empty:
        # Create empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No health score data available for the selected time range",
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
    
    # Create figure
    fig = go.Figure()
    
    # Add health score line with color based on zones
    # We'll color the line by segments
    df = health_data.copy().sort_values('timestamp')
    
    # Add the main line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['health_score'],
        mode='lines+markers',
        name='Health Score',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=6, color='#2196F3'),
        hovertemplate='<b>%{x|%Y-%m-%d %H:%M}</b><br>' +
                      'Health Score: %{y:.1f}/100<br>' +
                      '<extra></extra>',
    ))
    
    # Add baseline reference line if provided
    if show_baseline and baseline_score is not None:
        fig.add_trace(go.Scatter(
            x=[df['timestamp'].min(), df['timestamp'].max()],
            y=[baseline_score, baseline_score],
            mode='lines',
            name=f'Baseline ({baseline_score:.1f})',
            line=dict(color='gray', width=2, dash='dash'),
            hovertemplate='Baseline: %{y:.1f}<extra></extra>',
        ))
    
    # Add colored zones as background shapes
    fig.add_hrect(
        y0=0, y1=40,
        fillcolor='rgba(255, 68, 68, 0.1)',
        layer='below',
        line_width=0,
        annotation_text="Poor",
        annotation_position="right",
    )
    fig.add_hrect(
        y0=40, y1=70,
        fillcolor='rgba(255, 179, 71, 0.1)',
        layer='below',
        line_width=0,
        annotation_text="Fair",
        annotation_position="right",
    )
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor='rgba(76, 175, 80, 0.1)',
        layer='below',
        line_width=0,
        annotation_text="Good",
        annotation_position="right",
    )
    
    # Update layout
    fig.update_layout(
        title=f"Health Score History - {time_range_label}",
        xaxis_title="Time",
        yaxis_title="Health Score",
        height=450,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridcolor='#E5E5E5',
            gridwidth=1,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E5E5E5',
            gridwidth=1,
            range=[0, 105],  # Slightly above 100 for visibility
        ),
    )
    
    return fig


def create_contributing_factors_display(
    factors: Dict[str, float],
    show_as_chart: bool = True
) -> Optional[go.Figure]:
    """
    Create a display for contributing factors breakdown.
    
    Args:
        factors: Dictionary of factor names and their percentage contributions
        show_as_chart: If True, returns a Plotly figure; if False, returns None (use Streamlit components)
        
    Returns:
        Plotly Figure object if show_as_chart is True, otherwise None
    """
    if not factors:
        return None
    
    if show_as_chart:
        # Create horizontal bar chart
        factor_names = list(factors.keys())
        factor_values = list(factors.values())
        
        # Define colors for each factor
        color_map = {
            'temperature_stability': '#FF6B6B',
            'activity_level': '#4ECDC4',
            'behavioral_consistency': '#45B7D1',
            'rumination_quality': '#96CEB4',
            'alert_impact': '#FFEAA7',
        }
        
        colors = [color_map.get(name, '#95A5A6') for name in factor_names]
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            y=factor_names,
            x=factor_values,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{v:.1f}%" for v in factor_values],
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Contribution: %{x:.1f}%<extra></extra>',
        ))
        
        fig.update_layout(
            title="Contributing Factors Breakdown",
            xaxis_title="Contribution (%)",
            yaxis_title="",
            height=300,
            margin=dict(l=20, r=20, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridcolor='#E5E5E5',
                range=[0, max(factor_values) * 1.1],
            ),
            yaxis=dict(
                showgrid=False,
            ),
        )
        
        return fig
    
    return None


def display_contributing_factors_streamlit(factors: Dict[str, float]):
    """
    Display contributing factors using Streamlit components.
    
    Args:
        factors: Dictionary of factor names and their percentage contributions
    """
    if not factors:
        st.info("No contributing factors data available")
        return
    
    st.subheader("Contributing Factors Breakdown")
    
    # Define colors and icons for each factor
    factor_config = {
        'temperature_stability': {'icon': '🌡️', 'color': '#FF6B6B', 'label': 'Temperature Stability', 'desc': 'Higher is better'},
        'activity_level': {'icon': '🏃', 'color': '#4ECDC4', 'label': 'Activity Level', 'desc': 'Higher is better'},
        'behavioral_consistency': {'icon': '🎯', 'color': '#45B7D1', 'label': 'Behavioral Consistency', 'desc': 'Higher is better'},
        'rumination_quality': {'icon': '🐄', 'color': '#96CEB4', 'label': 'Rumination Quality', 'desc': 'Higher is better'},
        'alert_impact': {'icon': '⚠️', 'color': '#FFEAA7', 'label': 'Alert Status', 'desc': '0%=many alerts, 100%=no alerts'},
    }

    # Display each factor with progress bar
    for factor_name, percentage in factors.items():
        config = factor_config.get(factor_name, {'icon': '📊', 'color': '#95A5A6', 'label': factor_name, 'desc': ''})

        # Format label with description for alert_impact
        if factor_name == 'alert_impact':
            label = f"{config['icon']} {config['label']}"
            st.markdown(f"**{label}** <small style='color: #888;'>({config['desc']})</small>", unsafe_allow_html=True)
        else:
            st.markdown(f"**{config['icon']} {config['label']}**")

        col1, col2 = st.columns([4, 1])

        with col1:
            # Progress bar with custom color
            st.progress(percentage / 100.0)

        with col2:
            st.markdown(f"**{percentage:.1f}%**")

        st.markdown("<br>", unsafe_allow_html=True)


def create_trend_indicator(
    trend_direction: str,
    trend_rate: float
) -> str:
    """
    Create a visual trend indicator.
    
    Args:
        trend_direction: 'improving', 'stable', or 'deteriorating'
        trend_rate: Rate of change in points per day
        
    Returns:
        HTML string with trend indicator
    """
    if trend_direction == 'improving':
        icon = "📈"
        color = "green"
        text = f"Improving (+{abs(trend_rate):.1f} pts/day)"
    elif trend_direction == 'deteriorating':
        icon = "📉"
        color = "red"
        text = f"Deteriorating ({trend_rate:.1f} pts/day)"
    else:
        icon = "➡️"
        color = "gray"
        text = "Stable"
    
    html = f"""
    <div style='display: inline-flex; align-items: center; padding: 8px 16px; 
                background-color: {color}15; border-left: 4px solid {color}; 
                border-radius: 4px; margin: 8px 0;'>
        <span style='font-size: 24px; margin-right: 8px;'>{icon}</span>
        <span style='color: {color}; font-weight: bold; font-size: 16px;'>{text}</span>
    </div>
    """
    
    return html


def get_health_status_message(score: float) -> tuple[str, str, str]:
    """
    Get health status message, icon, and color based on score.
    
    Args:
        score: Health score (0-100)
        
    Returns:
        Tuple of (status_text, icon, color)
    """
    if score >= 80:
        return "Excellent - Animal is healthy", "✅", "green"
    elif score >= 70:
        return "Good - Minor concerns", "⚠️", "blue"
    elif score >= 60:
        return "Fair - Monitor closely", "⚠️", "orange"
    elif score >= 40:
        return "Poor - Requires attention", "🚨", "red"
    else:
        return "Critical - Immediate attention needed", "🚨", "darkred"
