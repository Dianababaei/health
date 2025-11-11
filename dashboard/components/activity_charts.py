"""
Activity and Behavior Monitoring Charts
========================================
Plotly-based interactive visualizations for cattle activity and behavior monitoring.

Features:
- Movement intensity time-series charts
- Activity vs rest duration bar charts
- Daily activity pattern heatmaps
- Historical baseline comparisons
- Stress behavior markers
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# Activity classification
REST_STATES = ['lying']
ACTIVE_STATES = ['standing', 'walking', 'feeding', 'ruminating']


def calculate_movement_intensity(
    df: pd.DataFrame,
    fxa_col: str = 'fxa',
    mya_col: str = 'mya',
    rza_col: str = 'rza'
) -> pd.DataFrame:
    """
    Calculate movement intensity from accelerometer data.
    
    Args:
        df: DataFrame with accelerometer columns
        fxa_col: Forward acceleration column name
        mya_col: Lateral acceleration column name
        rza_col: Vertical acceleration column name
    
    Returns:
        DataFrame with added movement_intensity column
    """
    result = df.copy()
    
    if all(col in df.columns for col in [fxa_col, mya_col, rza_col]):
        result['movement_intensity'] = np.sqrt(
            df[fxa_col]**2 + df[mya_col]**2 + df[rza_col]**2
        )
    else:
        result['movement_intensity'] = 0.0
        logger.warning("Missing accelerometer columns for movement intensity calculation")
    
    return result


def classify_activity_state(df: pd.DataFrame, state_col: str = 'behavioral_state') -> pd.DataFrame:
    """
    Classify behavioral states into activity (active/rest) categories.
    
    Args:
        df: DataFrame with behavioral state column
        state_col: Column name containing behavioral states
    
    Returns:
        DataFrame with added activity_state column
    """
    result = df.copy()
    
    if state_col in df.columns:
        result['activity_state'] = df[state_col].apply(
            lambda x: 'rest' if x in REST_STATES else 'active'
        )
    else:
        result['activity_state'] = 'unknown'
        logger.warning(f"Column {state_col} not found in dataframe")
    
    return result


def aggregate_hourly_activity(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    state_col: str = 'behavioral_state',
    intensity_col: str = 'movement_intensity'
) -> pd.DataFrame:
    """
    Aggregate activity data by hour.
    
    Args:
        df: DataFrame with activity data
        timestamp_col: Timestamp column name
        state_col: Behavioral state column name
        intensity_col: Movement intensity column name
    
    Returns:
        DataFrame with hourly aggregations
    """
    if df.empty or timestamp_col not in df.columns:
        return pd.DataFrame()
    
    result = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    
    # Create hour column
    result['hour'] = result[timestamp_col].dt.floor('H')
    
    # Aggregate by hour
    aggregations = {}
    
    # Movement intensity statistics
    if intensity_col in result.columns:
        aggregations[intensity_col] = ['mean', 'std', 'min', 'max']
    
    # Count records per hour
    aggregations[timestamp_col] = 'count'
    
    hourly_data = result.groupby('hour').agg(aggregations).reset_index()
    
    # Flatten column names
    hourly_data.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                           for col in hourly_data.columns.values]
    
    # Calculate activity/rest durations per hour
    if state_col in result.columns:
        # Classify as active/rest
        result = classify_activity_state(result, state_col)
        
        # Count active and rest minutes per hour
        activity_counts = result.groupby(['hour', 'activity_state']).size().unstack(fill_value=0)
        
        if 'active' in activity_counts.columns:
            hourly_data['active_count'] = hourly_data['hour'].map(activity_counts['active'])
        else:
            hourly_data['active_count'] = 0
            
        if 'rest' in activity_counts.columns:
            hourly_data['rest_count'] = hourly_data['hour'].map(activity_counts['rest'])
        else:
            hourly_data['rest_count'] = 0
        
        # Calculate percentages
        total_counts = hourly_data['active_count'] + hourly_data['rest_count']
        hourly_data['active_percentage'] = (hourly_data['active_count'] / total_counts * 100).fillna(0)
        hourly_data['rest_percentage'] = (hourly_data['rest_count'] / total_counts * 100).fillna(0)
    
    return hourly_data


def aggregate_daily_activity(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    state_col: str = 'behavioral_state',
    intensity_col: str = 'movement_intensity'
) -> pd.DataFrame:
    """
    Aggregate activity data by day.
    
    Args:
        df: DataFrame with activity data
        timestamp_col: Timestamp column name
        state_col: Behavioral state column name
        intensity_col: Movement intensity column name
    
    Returns:
        DataFrame with daily aggregations
    """
    if df.empty or timestamp_col not in df.columns:
        return pd.DataFrame()
    
    result = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    
    # Create day column
    result['day'] = result[timestamp_col].dt.date
    
    # Aggregate by day
    aggregations = {}
    
    # Movement intensity statistics
    if intensity_col in result.columns:
        aggregations[intensity_col] = ['mean', 'std', 'min', 'max']
    
    # Count records per day
    aggregations[timestamp_col] = 'count'
    
    daily_data = result.groupby('day').agg(aggregations).reset_index()
    
    # Flatten column names
    daily_data.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                         for col in daily_data.columns.values]
    
    # Calculate activity/rest durations per day
    if state_col in result.columns:
        # Classify as active/rest
        result = classify_activity_state(result, state_col)
        
        # Count active and rest minutes per day
        activity_counts = result.groupby(['day', 'activity_state']).size().unstack(fill_value=0)
        
        if 'active' in activity_counts.columns:
            daily_data['active_count'] = daily_data['day'].map(activity_counts['active'])
        else:
            daily_data['active_count'] = 0
            
        if 'rest' in activity_counts.columns:
            daily_data['rest_count'] = daily_data['day'].map(activity_counts['rest'])
        else:
            daily_data['rest_count'] = 0
        
        # Calculate percentages
        total_counts = daily_data['active_count'] + daily_data['rest_count']
        daily_data['active_percentage'] = (daily_data['active_count'] / total_counts * 100).fillna(0)
        daily_data['rest_percentage'] = (daily_data['rest_count'] / total_counts * 100).fillna(0)
    
    return daily_data


def calculate_historical_baseline(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    intensity_col: str = 'movement_intensity',
    baseline_days: int = 7
) -> Dict[str, float]:
    """
    Calculate historical baseline metrics for comparison.
    
    Args:
        df: DataFrame with activity data
        timestamp_col: Timestamp column name
        intensity_col: Movement intensity column name
        baseline_days: Number of days to use for baseline
    
    Returns:
        Dictionary with baseline metrics
    """
    if df.empty or timestamp_col not in df.columns:
        return {}
    
    result = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    
    # Calculate cutoff for baseline period
    max_date = result[timestamp_col].max()
    baseline_cutoff = max_date - timedelta(days=baseline_days)
    
    # Filter to baseline period
    baseline_data = result[result[timestamp_col] >= baseline_cutoff]
    
    baseline = {}
    
    if intensity_col in baseline_data.columns:
        baseline['avg_intensity'] = baseline_data[intensity_col].mean()
        baseline['std_intensity'] = baseline_data[intensity_col].std()
        baseline['min_intensity'] = baseline_data[intensity_col].min()
        baseline['max_intensity'] = baseline_data[intensity_col].max()
    
    baseline['period_days'] = baseline_days
    baseline['data_points'] = len(baseline_data)
    
    return baseline


def detect_stress_periods(
    df: pd.DataFrame,
    intensity_col: str = 'movement_intensity',
    threshold_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Detect periods of potentially stress-related erratic movement.
    
    Args:
        df: DataFrame with movement intensity data
        intensity_col: Movement intensity column name
        threshold_multiplier: Multiplier for std dev to flag stress
    
    Returns:
        DataFrame with added stress_indicator column
    """
    result = df.copy()
    
    if intensity_col not in df.columns or df.empty:
        result['stress_indicator'] = False
        return result
    
    # Calculate rolling statistics
    window_size = min(30, len(df) // 10)  # 30 minutes or 10% of data
    if window_size < 3:
        window_size = 3
    
    rolling_mean = df[intensity_col].rolling(window=window_size, center=True).mean()
    rolling_std = df[intensity_col].rolling(window=window_size, center=True).std()
    
    # Flag values above threshold
    threshold = rolling_mean + (threshold_multiplier * rolling_std)
    result['stress_indicator'] = df[intensity_col] > threshold
    
    # Fill NaN values (from rolling window edges)
    result['stress_indicator'] = result['stress_indicator'].fillna(False)
    
    return result


def create_movement_intensity_chart(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    intensity_col: str = 'movement_intensity',
    title: str = "Movement Intensity Over Time",
    show_stress_markers: bool = True,
    baseline: Optional[Dict[str, float]] = None
) -> go.Figure:
    """
    Create interactive time-series chart showing movement intensity.
    
    Args:
        df: DataFrame with movement intensity data
        timestamp_col: Timestamp column name
        intensity_col: Movement intensity column name
        title: Chart title
        show_stress_markers: Whether to show stress period markers
        baseline: Optional baseline metrics for comparison
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    if df.empty or timestamp_col not in df.columns or intensity_col not in df.columns:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Main intensity line
    fig.add_trace(go.Scatter(
        x=df[timestamp_col],
        y=df[intensity_col],
        mode='lines',
        name='Movement Intensity',
        line=dict(color='#2E86AB', width=2),
        hovertemplate='<b>Time:</b> %{x}<br><b>Intensity:</b> %{y:.3f}<extra></extra>'
    ))
    
    # Add baseline reference line if provided
    if baseline and 'avg_intensity' in baseline:
        fig.add_hline(
            y=baseline['avg_intensity'],
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Baseline Avg: {baseline['avg_intensity']:.3f}",
            annotation_position="right"
        )
    
    # Add stress period markers
    if show_stress_markers:
        df_with_stress = detect_stress_periods(df, intensity_col)
        stress_periods = df_with_stress[df_with_stress['stress_indicator']]
        
        if not stress_periods.empty:
            fig.add_trace(go.Scatter(
                x=stress_periods[timestamp_col],
                y=stress_periods[intensity_col],
                mode='markers',
                name='Elevated Activity',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond',
                    line=dict(width=1, color='darkred')
                ),
                hovertemplate='<b>Elevated Activity</b><br>Time: %{x}<br>Intensity: %{y:.3f}<extra></extra>'
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Movement Intensity",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_activity_rest_bar_chart(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    state_col: str = 'behavioral_state',
    aggregation: str = 'daily',
    title: str = "Activity vs Rest Duration"
) -> go.Figure:
    """
    Create stacked bar chart showing active vs rest time.
    
    Args:
        df: DataFrame with behavioral state data
        timestamp_col: Timestamp column name
        state_col: Behavioral state column name
        aggregation: 'hourly' or 'daily' aggregation
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    if df.empty or timestamp_col not in df.columns or state_col not in df.columns:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Aggregate data
    if aggregation == 'hourly':
        agg_data = aggregate_hourly_activity(df, timestamp_col, state_col)
        x_label = "Hour"
        x_data = agg_data['hour'] if 'hour' in agg_data.columns else []
    else:  # daily
        agg_data = aggregate_daily_activity(df, timestamp_col, state_col)
        x_label = "Day"
        x_data = agg_data['day'] if 'day' in agg_data.columns else []
    
    if agg_data.empty:
        fig.add_annotation(
            text="No aggregated data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Create stacked bar chart
    fig.add_trace(go.Bar(
        x=x_data,
        y=agg_data['active_count'] if 'active_count' in agg_data.columns else [],
        name='Active',
        marker_color='#7ED321',
        hovertemplate='<b>Active</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.add_trace(go.Bar(
        x=x_data,
        y=agg_data['rest_count'] if 'rest_count' in agg_data.columns else [],
        name='Rest',
        marker_color='#4A90E2',
        hovertemplate='<b>Rest</b><br>Count: %{y}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title="Duration (minutes)",
        barmode='stack',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_daily_activity_heatmap(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    intensity_col: str = 'movement_intensity',
    title: str = "24-Hour Activity Pattern"
) -> go.Figure:
    """
    Create heatmap showing activity distribution across 24 hours.
    
    Args:
        df: DataFrame with activity data
        timestamp_col: Timestamp column name
        intensity_col: Movement intensity column name
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    if df.empty or timestamp_col not in df.columns or intensity_col not in df.columns:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    result = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    
    # Extract hour and day
    result['hour'] = result[timestamp_col].dt.hour
    result['day'] = result[timestamp_col].dt.date
    
    # Create pivot table: rows = days, columns = hours
    pivot_data = result.pivot_table(
        values=intensity_col,
        index='day',
        columns='hour',
        aggfunc='mean'
    )
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=[str(d) for d in pivot_data.index],
        colorscale='YlOrRd',
        hovertemplate='<b>Day:</b> %{y}<br><b>Hour:</b> %{x}:00<br><b>Avg Intensity:</b> %{z:.3f}<extra></extra>',
        colorbar=dict(title="Intensity")
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        template='plotly_white',
        height=400,
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2
        )
    )
    
    return fig


def create_historical_comparison_chart(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    intensity_col: str = 'movement_intensity',
    baseline_days: int = 7,
    title: str = "Activity Comparison: Current vs Historical"
) -> go.Figure:
    """
    Create chart comparing current activity with historical baseline.
    
    Args:
        df: DataFrame with activity data
        timestamp_col: Timestamp column name
        intensity_col: Movement intensity column name
        baseline_days: Number of days for baseline calculation
        title: Chart title
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    if df.empty or timestamp_col not in df.columns or intensity_col not in df.columns:
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    result = df.copy()
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[timestamp_col]):
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])
    
    # Sort by timestamp
    result = result.sort_values(timestamp_col)
    
    # Calculate baseline
    baseline = calculate_historical_baseline(result, timestamp_col, intensity_col, baseline_days)
    
    # Split data into baseline period and recent period
    max_date = result[timestamp_col].max()
    baseline_cutoff = max_date - timedelta(days=baseline_days)
    recent_cutoff = max_date - timedelta(days=1)  # Last 24 hours
    
    baseline_data = result[result[timestamp_col] < baseline_cutoff]
    recent_data = result[result[timestamp_col] >= recent_cutoff]
    
    # Aggregate by hour of day for comparison
    if not baseline_data.empty:
        baseline_data = baseline_data.copy()
        baseline_data['hour'] = baseline_data[timestamp_col].dt.hour
        baseline_hourly = baseline_data.groupby('hour')[intensity_col].mean()
        
        fig.add_trace(go.Scatter(
            x=baseline_hourly.index,
            y=baseline_hourly.values,
            mode='lines+markers',
            name=f'Historical Avg ({baseline_days}d)',
            line=dict(color='gray', width=2, dash='dash'),
            marker=dict(size=6),
            hovertemplate='<b>Historical</b><br>Hour: %{x}:00<br>Avg Intensity: %{y:.3f}<extra></extra>'
        ))
    
    if not recent_data.empty:
        recent_data = recent_data.copy()
        recent_data['hour'] = recent_data[timestamp_col].dt.hour
        recent_hourly = recent_data.groupby('hour')[intensity_col].mean()
        
        fig.add_trace(go.Scatter(
            x=recent_hourly.index,
            y=recent_hourly.values,
            mode='lines+markers',
            name='Current (24h)',
            line=dict(color='#2E86AB', width=2),
            marker=dict(size=8),
            hovertemplate='<b>Current</b><br>Hour: %{x}:00<br>Avg Intensity: %{y:.3f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Average Movement Intensity",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=2,
            range=[-0.5, 23.5]
        )
    )
    
    return fig


def get_activity_summary_stats(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    state_col: str = 'behavioral_state',
    intensity_col: str = 'movement_intensity'
) -> Dict[str, Any]:
    """
    Calculate summary statistics for activity data.
    
    Args:
        df: DataFrame with activity data
        timestamp_col: Timestamp column name
        state_col: Behavioral state column name
        intensity_col: Movement intensity column name
    
    Returns:
        Dictionary with summary statistics
    """
    stats = {}
    
    if df.empty:
        return stats
    
    # Time span
    if timestamp_col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        time_span = (df[timestamp_col].max() - df[timestamp_col].min()).total_seconds() / 3600
        stats['time_span_hours'] = time_span
    
    # Movement intensity statistics
    if intensity_col in df.columns:
        stats['avg_intensity'] = df[intensity_col].mean()
        stats['std_intensity'] = df[intensity_col].std()
        stats['min_intensity'] = df[intensity_col].min()
        stats['max_intensity'] = df[intensity_col].max()
    
    # Activity/rest breakdown
    if state_col in df.columns:
        df_classified = classify_activity_state(df, state_col)
        
        activity_counts = df_classified['activity_state'].value_counts()
        total_count = len(df_classified)
        
        stats['active_count'] = activity_counts.get('active', 0)
        stats['rest_count'] = activity_counts.get('rest', 0)
        stats['active_percentage'] = (stats['active_count'] / total_count * 100) if total_count > 0 else 0
        stats['rest_percentage'] = (stats['rest_count'] / total_count * 100) if total_count > 0 else 0
        
        # State transitions
        state_changes = (df_classified[state_col] != df_classified[state_col].shift(1)).sum()
        stats['state_transitions'] = state_changes
    
    # Stress indicators
    if intensity_col in df.columns:
        df_with_stress = detect_stress_periods(df, intensity_col)
        stress_count = df_with_stress['stress_indicator'].sum()
        stats['stress_periods'] = int(stress_count)
        stats['stress_percentage'] = (stress_count / len(df_with_stress) * 100) if len(df_with_stress) > 0 else 0
    
    stats['total_records'] = len(df)
    
    return stats
