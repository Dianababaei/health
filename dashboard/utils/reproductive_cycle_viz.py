"""
Reproductive Cycle Visualization Utility
========================================
Utilities for detecting and visualizing reproductive cycles including
estrus cycles (21-day) and pregnancy tracking (60+ days).
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any


# Cycle parameters
ESTRUS_CYCLE_DAYS = 21
PREGNANCY_DURATION_DAYS = 283  # Average cattle gestation
ESTRUS_DETECTION_TEMP_THRESHOLD = 0.3  # Temperature increase during estrus (°C)
ESTRUS_DETECTION_ACTIVITY_THRESHOLD = 1.5  # Activity multiplier during estrus


def detect_estrus_events(
    data: pd.DataFrame,
    temp_column: str = 'temperature',
    activity_column: str = 'activity_level',
    time_column: str = 'timestamp',
    min_duration_hours: int = 12,
    max_duration_hours: int = 36
) -> List[Dict[str, Any]]:
    """
    Detect estrus events from temperature and activity patterns.
    
    Estrus is characterized by:
    - Increased activity (mounting behavior, restlessness)
    - Slight temperature increase
    - Duration: 12-36 hours typically
    
    Args:
        data: DataFrame with sensor data
        temp_column: Column name for temperature
        activity_column: Column name for activity level
        time_column: Column name for timestamps
        min_duration_hours: Minimum duration to consider as estrus
        max_duration_hours: Maximum duration to consider as estrus
        
    Returns:
        List of detected estrus events with start/end times and confidence
    """
    if data.empty or time_column not in data.columns:
        return []
    
    df = data.sort_values(time_column).copy()
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Calculate baseline values (rolling 3-day average)
    if temp_column in df.columns:
        baseline_temp = df[temp_column].rolling(window=72, min_periods=24, center=True).mean()
        temp_deviation = df[temp_column] - baseline_temp
    else:
        temp_deviation = pd.Series(0, index=df.index)
    
    if activity_column in df.columns:
        baseline_activity = df[activity_column].rolling(window=72, min_periods=24, center=True).mean()
        activity_ratio = df[activity_column] / baseline_activity
    else:
        activity_ratio = pd.Series(1, index=df.index)
    
    # Detect potential estrus periods
    estrus_candidates = (
        (temp_deviation > ESTRUS_DETECTION_TEMP_THRESHOLD) |
        (activity_ratio > ESTRUS_DETECTION_ACTIVITY_THRESHOLD)
    )
    
    # Find continuous periods
    estrus_events = []
    in_estrus = False
    start_idx = None
    
    for idx in range(len(df)):
        if estrus_candidates.iloc[idx] and not in_estrus:
            # Start of potential estrus
            in_estrus = True
            start_idx = idx
        elif not estrus_candidates.iloc[idx] and in_estrus:
            # End of potential estrus
            in_estrus = False
            
            # Check duration
            start_time = df.iloc[start_idx][time_column]
            end_time = df.iloc[idx - 1][time_column]
            duration_hours = (end_time - start_time).total_seconds() / 3600
            
            if min_duration_hours <= duration_hours <= max_duration_hours:
                # Calculate confidence based on temperature and activity patterns
                event_data = df.iloc[start_idx:idx]
                temp_conf = float(temp_deviation.iloc[start_idx:idx].mean()) / ESTRUS_DETECTION_TEMP_THRESHOLD
                activity_conf = float(activity_ratio.iloc[start_idx:idx].mean()) / ESTRUS_DETECTION_ACTIVITY_THRESHOLD
                confidence = min(1.0, (temp_conf + activity_conf) / 2)
                
                estrus_events.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_hours': float(duration_hours),
                    'confidence': float(max(0.0, min(1.0, confidence))),
                    'avg_temp_deviation': float(temp_deviation.iloc[start_idx:idx].mean()),
                    'avg_activity_ratio': float(activity_ratio.iloc[start_idx:idx].mean()),
                })
    
    return estrus_events


def predict_next_estrus(
    estrus_events: List[Dict[str, Any]],
    current_date: Optional[datetime] = None
) -> Optional[Dict[str, Any]]:
    """
    Predict the next estrus event based on historical cycle pattern.
    
    Args:
        estrus_events: List of detected estrus events
        current_date: Current date (defaults to now)
        
    Returns:
        Dictionary with predicted next estrus date and confidence
    """
    if not estrus_events:
        return None
    
    if current_date is None:
        current_date = datetime.now()
    
    # Sort events by start time
    sorted_events = sorted(estrus_events, key=lambda x: x['start_time'])
    
    if len(sorted_events) < 2:
        # Only one event, use standard cycle length
        last_event = sorted_events[-1]
        next_date = last_event['start_time'] + timedelta(days=ESTRUS_CYCLE_DAYS)
        return {
            'predicted_date': next_date,
            'confidence': 0.5,  # Low confidence with only one observation
            'cycle_length_days': ESTRUS_CYCLE_DAYS,
            'days_until': (next_date - current_date).days,
        }
    
    # Calculate average cycle length from recent events
    cycle_lengths = []
    for i in range(1, len(sorted_events)):
        prev_event = sorted_events[i - 1]
        curr_event = sorted_events[i]
        cycle_length = (curr_event['start_time'] - prev_event['start_time']).days
        
        # Only include reasonable cycle lengths (18-24 days)
        if 18 <= cycle_length <= 24:
            cycle_lengths.append(cycle_length)
    
    if not cycle_lengths:
        # No valid cycles found
        return None
    
    avg_cycle_length = np.mean(cycle_lengths)
    cycle_std = np.std(cycle_lengths) if len(cycle_lengths) > 1 else 2.0
    
    # Predict next estrus
    last_event = sorted_events[-1]
    next_date = last_event['start_time'] + timedelta(days=int(avg_cycle_length))
    
    # Confidence based on cycle regularity
    confidence = max(0.3, min(1.0, 1.0 - (cycle_std / 3.0)))
    
    return {
        'predicted_date': next_date,
        'confidence': float(confidence),
        'cycle_length_days': float(avg_cycle_length),
        'cycle_std': float(cycle_std),
        'days_until': (next_date - current_date).days,
        'prediction_range_days': int(cycle_std * 2),  # 95% confidence interval
    }


def detect_pregnancy(
    data: pd.DataFrame,
    estrus_events: List[Dict[str, Any]],
    temp_column: str = 'temperature',
    activity_column: str = 'activity_level',
    time_column: str = 'timestamp'
) -> Optional[Dict[str, Any]]:
    """
    Detect potential pregnancy based on missing estrus cycles and physiological changes.
    
    Pregnancy indicators:
    - Missing expected estrus cycles (>25 days since last estrus)
    - Reduced activity levels
    - Slightly elevated baseline temperature
    
    Args:
        data: DataFrame with sensor data
        estrus_events: List of detected estrus events
        temp_column: Column name for temperature
        activity_column: Column name for activity level
        time_column: Column name for timestamps
        
    Returns:
        Dictionary with pregnancy detection results or None
    """
    if data.empty or not estrus_events:
        return None
    
    df = data.sort_values(time_column).copy()
    df[time_column] = pd.to_datetime(df[time_column])
    current_date = df[time_column].max()
    
    # Get last estrus event
    last_estrus = max(estrus_events, key=lambda x: x['start_time'])
    days_since_estrus = (current_date - last_estrus['start_time']).days
    
    # Check if estrus cycle is significantly overdue
    if days_since_estrus < 25:  # Normal cycle is 21 days, allow some variance
        return None
    
    # Analyze recent physiological changes
    recent_data = df[df[time_column] >= (current_date - timedelta(days=14))]
    baseline_data = df[
        (df[time_column] >= (last_estrus['start_time'] - timedelta(days=14))) &
        (df[time_column] < last_estrus['start_time'])
    ]
    
    if recent_data.empty or baseline_data.empty:
        return None
    
    indicators = []
    
    # Temperature indicator (slightly elevated)
    if temp_column in df.columns:
        recent_temp = recent_data[temp_column].mean()
        baseline_temp = baseline_data[temp_column].mean()
        temp_increase = recent_temp - baseline_temp
        
        if 0.1 <= temp_increase <= 0.4:  # Slight increase is normal in pregnancy
            indicators.append(('temperature', 0.7))
    
    # Activity indicator (reduced activity)
    if activity_column in df.columns:
        recent_activity = recent_data[activity_column].mean()
        baseline_activity = baseline_data[activity_column].mean()
        activity_ratio = recent_activity / baseline_activity if baseline_activity > 0 else 1.0
        
        if activity_ratio < 0.8:  # Reduced activity
            indicators.append(('activity', 0.6))
    
    # Missed cycle indicator (strongest indicator)
    expected_cycles = days_since_estrus // ESTRUS_CYCLE_DAYS
    if expected_cycles >= 2:
        indicators.append(('missed_cycles', 0.9))
    elif expected_cycles == 1:
        indicators.append(('missed_cycles', 0.7))
    
    if not indicators:
        return None
    
    # Calculate overall confidence
    confidence = np.mean([conf for _, conf in indicators])
    
    # Estimate conception date (around last estrus)
    conception_date = last_estrus['start_time'] + timedelta(days=1)
    days_pregnant = (current_date - conception_date).days
    expected_calving = conception_date + timedelta(days=PREGNANCY_DURATION_DAYS)
    
    return {
        'detected': True,
        'confidence': float(confidence),
        'conception_date': conception_date,
        'days_pregnant': days_pregnant,
        'expected_calving_date': expected_calving,
        'days_until_calving': (expected_calving - current_date).days,
        'indicators': dict(indicators),
        'days_since_last_estrus': days_since_estrus,
    }


def create_cycle_timeline(
    estrus_events: List[Dict[str, Any]],
    pregnancy_status: Optional[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    height: int = 300
) -> go.Figure:
    """
    Create an interactive timeline visualization of reproductive cycles.
    
    Args:
        estrus_events: List of detected estrus events
        pregnancy_status: Pregnancy detection results
        start_date: Start date for timeline
        end_date: End date for timeline
        height: Figure height in pixels
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add estrus events as colored bars
    for event in estrus_events:
        if start_date <= event['start_time'] <= end_date:
            fig.add_trace(go.Scatter(
                x=[event['start_time'], event['end_time']],
                y=[1, 1],
                mode='lines',
                line=dict(color='orange', width=20),
                name='Estrus Event',
                showlegend=False,
                hovertemplate=(
                    f"<b>Estrus Event</b><br>"
                    f"Start: {event['start_time'].strftime('%Y-%m-%d %H:%M')}<br>"
                    f"Duration: {event['duration_hours']:.1f} hours<br>"
                    f"Confidence: {event['confidence']:.1%}<br>"
                    "<extra></extra>"
                ),
            ))
    
    # Add pregnancy period if detected
    if pregnancy_status and pregnancy_status.get('detected'):
        conception_date = pregnancy_status['conception_date']
        calving_date = pregnancy_status['expected_calving_date']
        
        if conception_date <= end_date:
            fig.add_trace(go.Scatter(
                x=[conception_date, min(calving_date, end_date)],
                y=[2, 2],
                mode='lines',
                line=dict(color='green', width=20),
                name='Pregnancy Period',
                showlegend=False,
                hovertemplate=(
                    f"<b>Pregnancy Detected</b><br>"
                    f"Conception: {conception_date.strftime('%Y-%m-%d')}<br>"
                    f"Days Pregnant: {pregnancy_status['days_pregnant']}<br>"
                    f"Expected Calving: {calving_date.strftime('%Y-%m-%d')}<br>"
                    f"Confidence: {pregnancy_status['confidence']:.1%}<br>"
                    "<extra></extra>"
                ),
            ))
    
    # Update layout
    fig.update_layout(
        title="Reproductive Cycle Timeline",
        xaxis_title="Date",
        yaxis=dict(
            tickmode='array',
            tickvals=[1, 2],
            ticktext=['Estrus', 'Pregnancy'],
            range=[0.5, 2.5],
        ),
        height=height,
        showlegend=False,
        hovermode='closest',
    )
    
    return fig


def create_cycle_calendar(
    estrus_events: List[Dict[str, Any]],
    pregnancy_status: Optional[Dict[str, Any]],
    months: int = 6
) -> pd.DataFrame:
    """
    Create a calendar view of reproductive cycles.
    
    Args:
        estrus_events: List of detected estrus events
        pregnancy_status: Pregnancy detection results
        months: Number of months to display
        
    Returns:
        DataFrame with calendar data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months * 30)
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    calendar_data = []
    for date in dates:
        day_data = {
            'date': date,
            'day': date.day,
            'month': date.strftime('%B'),
            'year': date.year,
            'event': 'normal',
            'description': '',
        }
        
        # Check for estrus events
        for event in estrus_events:
            if event['start_time'].date() <= date.date() <= event['end_time'].date():
                day_data['event'] = 'estrus'
                day_data['description'] = f"Estrus (confidence: {event['confidence']:.1%})"
                break
        
        # Check for pregnancy
        if pregnancy_status and pregnancy_status.get('detected'):
            conception_date = pregnancy_status['conception_date'].date()
            if date.date() >= conception_date:
                day_data['event'] = 'pregnancy'
                days_pregnant = (date.date() - conception_date).days
                day_data['description'] = f"Pregnancy day {days_pregnant}"
        
        calendar_data.append(day_data)
    
    return pd.DataFrame(calendar_data)


def calculate_cycle_statistics(
    estrus_events: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate statistics about reproductive cycles.
    
    Args:
        estrus_events: List of detected estrus events
        
    Returns:
        Dictionary with cycle statistics
    """
    if not estrus_events:
        return {
            'total_events': 0,
            'avg_cycle_length': None,
            'cycle_regularity': None,
            'avg_duration_hours': None,
        }
    
    sorted_events = sorted(estrus_events, key=lambda x: x['start_time'])
    
    # Calculate cycle lengths
    cycle_lengths = []
    for i in range(1, len(sorted_events)):
        prev_event = sorted_events[i - 1]
        curr_event = sorted_events[i]
        cycle_length = (curr_event['start_time'] - prev_event['start_time']).days
        
        if 18 <= cycle_length <= 24:  # Valid cycle length
            cycle_lengths.append(cycle_length)
    
    # Calculate durations
    durations = [event['duration_hours'] for event in estrus_events]
    
    stats = {
        'total_events': len(estrus_events),
        'valid_cycles': len(cycle_lengths),
        'first_event': sorted_events[0]['start_time'],
        'last_event': sorted_events[-1]['start_time'],
        'avg_duration_hours': float(np.mean(durations)) if durations else None,
        'std_duration_hours': float(np.std(durations)) if len(durations) > 1 else None,
    }
    
    if cycle_lengths:
        stats['avg_cycle_length'] = float(np.mean(cycle_lengths))
        stats['std_cycle_length'] = float(np.std(cycle_lengths)) if len(cycle_lengths) > 1 else None
        stats['cycle_regularity'] = float(1.0 - min(1.0, np.std(cycle_lengths) / ESTRUS_CYCLE_DAYS)) if len(cycle_lengths) > 1 else None
    else:
        stats['avg_cycle_length'] = None
        stats['std_cycle_length'] = None
        stats['cycle_regularity'] = None
    
    return stats
