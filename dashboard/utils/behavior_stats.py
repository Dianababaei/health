"""
Behavior Statistics Utilities
=============================
Functions for calculating behavioral state statistics including
durations, transitions, and longest continuous periods.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def calculate_state_durations(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate total time spent in each behavioral state.
    
    Args:
        df: DataFrame with columns: timestamp, state
        
    Returns:
        Dictionary mapping state names to total minutes
    """
    if df.empty:
        return {}
    
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Calculate duration for each record
    # Duration is the time until the next record (or end of period)
    durations = {}
    
    for state in df['state'].unique():
        state_df = df[df['state'] == state].copy()
        
        if len(state_df) == 0:
            durations[state] = 0.0
            continue
        
        # Calculate time differences
        total_minutes = 0.0
        
        for i in range(len(state_df)):
            # Find the next record in the full dataframe
            current_idx = state_df.index[i]
            
            if current_idx < len(df) - 1:
                # Use the next timestamp
                next_timestamp = df.loc[current_idx + 1, 'timestamp']
                current_timestamp = df.loc[current_idx, 'timestamp']
                duration = (next_timestamp - current_timestamp).total_seconds() / 60
            else:
                # Last record - assume 1 minute duration
                duration = 1.0
            
            total_minutes += duration
        
        durations[state] = total_minutes
    
    return durations


def calculate_state_transitions(df: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate the number of transitions between behavioral states.
    
    Args:
        df: DataFrame with columns: timestamp, state
        
    Returns:
        Dictionary with transition counts:
        - 'total': Total number of state changes
        - '<state1>_to_<state2>': Count of specific transitions
    """
    if df.empty or len(df) < 2:
        return {'total': 0}
    
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    transitions = defaultdict(int)
    total_transitions = 0
    
    # Count transitions
    for i in range(len(df) - 1):
        current_state = df.loc[i, 'state']
        next_state = df.loc[i + 1, 'state']
        
        if current_state != next_state:
            transition_key = f"{current_state}_to_{next_state}"
            transitions[transition_key] += 1
            total_transitions += 1
    
    transitions['total'] = total_transitions
    return dict(transitions)


def calculate_longest_periods(df: pd.DataFrame) -> Dict[str, Tuple[float, pd.Timestamp, pd.Timestamp]]:
    """
    Calculate the longest continuous period for each behavioral state.
    
    Args:
        df: DataFrame with columns: timestamp, state
        
    Returns:
        Dictionary mapping state names to tuples of:
        (duration_minutes, start_timestamp, end_timestamp)
    """
    if df.empty:
        return {}
    
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    longest_periods = {}
    
    for state in df['state'].unique():
        max_duration = 0.0
        max_start = None
        max_end = None
        
        # Find continuous periods of this state
        current_duration = 0.0
        current_start = None
        current_end = None
        
        for i in range(len(df)):
            if df.loc[i, 'state'] == state:
                if current_start is None:
                    # Start of a new period
                    current_start = df.loc[i, 'timestamp']
                    current_end = df.loc[i, 'timestamp']
                    current_duration = 1.0  # Minimum 1 minute
                else:
                    # Continue existing period
                    current_end = df.loc[i, 'timestamp']
                    # Add duration to next timestamp or 1 minute if last
                    if i < len(df) - 1:
                        next_timestamp = df.loc[i + 1, 'timestamp']
                        duration = (next_timestamp - df.loc[i, 'timestamp']).total_seconds() / 60
                        current_duration += duration
                    else:
                        current_duration += 1.0
            else:
                # End of period
                if current_start is not None:
                    if current_duration > max_duration:
                        max_duration = current_duration
                        max_start = current_start
                        max_end = current_end
                    
                    # Reset for next period
                    current_start = None
                    current_end = None
                    current_duration = 0.0
        
        # Check last period
        if current_start is not None and current_duration > max_duration:
            max_duration = current_duration
            max_start = current_start
            max_end = current_end
        
        if max_start is not None:
            longest_periods[state] = (max_duration, max_start, max_end)
    
    return longest_periods


def calculate_state_percentages(durations: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate percentage of time spent in each state.
    
    Args:
        durations: Dictionary mapping state names to total minutes
        
    Returns:
        Dictionary mapping state names to percentages (0-100)
    """
    if not durations:
        return {}
    
    total_time = sum(durations.values())
    
    if total_time == 0:
        return {state: 0.0 for state in durations}
    
    percentages = {
        state: (duration / total_time) * 100
        for state, duration in durations.items()
    }
    
    return percentages


def generate_statistics_summary(df: pd.DataFrame) -> Dict:
    """
    Generate comprehensive statistics summary for behavioral states.
    
    Args:
        df: DataFrame with columns: timestamp, state
        
    Returns:
        Dictionary containing:
        - durations: Total time per state (minutes)
        - percentages: Percentage time per state
        - transitions: State transition counts
        - longest_periods: Longest continuous period per state
        - total_time: Total time range covered (minutes)
        - data_points: Number of data points
    """
    if df.empty:
        return {
            'durations': {},
            'percentages': {},
            'transitions': {'total': 0},
            'longest_periods': {},
            'total_time': 0.0,
            'data_points': 0,
        }
    
    # Calculate all statistics
    durations = calculate_state_durations(df)
    percentages = calculate_state_percentages(durations)
    transitions = calculate_state_transitions(df)
    longest_periods = calculate_longest_periods(df)
    
    # Calculate total time range
    df = df.sort_values('timestamp')
    total_time = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
    
    return {
        'durations': durations,
        'percentages': percentages,
        'transitions': transitions,
        'longest_periods': longest_periods,
        'total_time': total_time,
        'data_points': len(df),
    }


def format_duration_text(minutes: float) -> str:
    """
    Format duration in minutes to human-readable text.
    
    Args:
        minutes: Duration in minutes
        
    Returns:
        Formatted string (e.g., "2h 30m", "45m", "3d 5h")
    """
    if minutes < 1:
        return "< 1m"
    
    days = int(minutes // (24 * 60))
    remaining_minutes = minutes % (24 * 60)
    hours = int(remaining_minutes // 60)
    mins = int(remaining_minutes % 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0 or (days == 0 and hours == 0):
        parts.append(f"{mins}m")
    
    return " ".join(parts)


def prepare_timeline_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare behavioral state data for timeline visualization.
    
    Consolidates consecutive identical states into single segments
    with start and end times.
    
    Args:
        df: DataFrame with columns: timestamp, state, confidence
        
    Returns:
        DataFrame with columns: state, start, finish, confidence, duration_minutes
    """
    if df.empty:
        return pd.DataFrame(columns=['state', 'start', 'finish', 'confidence', 'duration_minutes'])
    
    # Ensure data is sorted by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    segments = []
    current_state = None
    current_start = None
    current_confidence_sum = 0.0
    current_confidence_count = 0
    
    for i in range(len(df)):
        state = df.loc[i, 'state']
        timestamp = df.loc[i, 'timestamp']
        confidence = df.loc[i, 'confidence'] if 'confidence' in df.columns else 0.95
        
        if state != current_state:
            # Save previous segment
            if current_state is not None:
                duration = (timestamp - current_start).total_seconds() / 60
                avg_confidence = current_confidence_sum / current_confidence_count if current_confidence_count > 0 else 0.95
                
                segments.append({
                    'state': current_state,
                    'start': current_start,
                    'finish': timestamp,
                    'confidence': avg_confidence,
                    'duration_minutes': duration,
                })
            
            # Start new segment
            current_state = state
            current_start = timestamp
            current_confidence_sum = confidence
            current_confidence_count = 1
        else:
            # Continue current segment
            current_confidence_sum += confidence
            current_confidence_count += 1
    
    # Save last segment
    if current_state is not None:
        # For the last segment, estimate finish time as 1 minute after last timestamp
        finish_time = df.loc[len(df) - 1, 'timestamp'] + timedelta(minutes=1)
        duration = (finish_time - current_start).total_seconds() / 60
        avg_confidence = current_confidence_sum / current_confidence_count if current_confidence_count > 0 else 0.95
        
        segments.append({
            'state': current_state,
            'start': current_start,
            'finish': finish_time,
            'confidence': avg_confidence,
            'duration_minutes': duration,
        })
    
    return pd.DataFrame(segments)


def aggregate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute-level behavioral states to hourly mode.
    
    For each hour, determines the most frequent state and averages
    confidence values.
    
    Args:
        df: DataFrame with columns: timestamp, state, confidence
        
    Returns:
        DataFrame with hourly aggregated states
    """
    if df.empty:
        return df
    
    # Ensure timestamp is datetime
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add hour column
    df['hour'] = df['timestamp'].dt.floor('h')
    
    # Group by hour and find mode state
    hourly_data = []
    
    for hour, group in df.groupby('hour'):
        # Find most frequent state
        mode_state = group['state'].mode()
        state = mode_state.iloc[0] if len(mode_state) > 0 else group['state'].iloc[0]
        
        # Average confidence
        confidence = group['confidence'].mean() if 'confidence' in group.columns else 0.95
        
        # Average motion intensity if available
        motion_intensity = group['motion_intensity'].mean() if 'motion_intensity' in group.columns else None
        
        hourly_data.append({
            'timestamp': hour,
            'state': state,
            'confidence': confidence,
            'motion_intensity': motion_intensity,
        })
    
    return pd.DataFrame(hourly_data)
