"""
Health Score Data Loader
=========================
Utilities for loading and processing health score data from the database.
Includes baseline calculations, contributing factors retrieval, and mock data generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Try to import database libraries
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


def query_health_scores(
    cow_id: int,
    start_time: datetime,
    end_time: datetime,
    connection=None
) -> pd.DataFrame:
    """
    Query health scores for a specific cow and time range.
    
    Args:
        cow_id: Cow identifier
        start_time: Start of time range (inclusive)
        end_time: End of time range (inclusive)
        connection: Database connection (if None, will use mock data)
        
    Returns:
        DataFrame with columns: timestamp, health_score, temperature_component,
        activity_component, behavior_component, rumination_component, alert_penalty,
        trend_direction, trend_rate, contributing_factors
    """
    # If no connection, return mock data
    if connection is None:
        return _generate_mock_health_scores(cow_id, start_time, end_time)
    
    # Build query
    query = """
        SELECT 
            timestamp,
            cow_id,
            health_score,
            temperature_component,
            activity_component,
            behavior_component,
            rumination_component,
            alert_penalty,
            trend_direction,
            trend_rate,
            days_since_baseline,
            contributing_factors
        FROM health_scores
        WHERE cow_id = %(cow_id)s
            AND timestamp >= %(start_time)s
            AND timestamp <= %(end_time)s
        ORDER BY timestamp ASC
    """
    
    try:
        # Execute query
        if HAS_PSYCOPG2 and hasattr(connection, 'cursor'):
            # psycopg2 connection
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, {'cow_id': cow_id, 'start_time': start_time, 'end_time': end_time})
            rows = cursor.fetchall()
            cursor.close()
            df = pd.DataFrame(rows)
        elif HAS_SQLALCHEMY:
            # SQLAlchemy connection
            df = pd.read_sql(
                text(query),
                connection,
                params={'cow_id': cow_id, 'start_time': start_time, 'end_time': end_time}
            )
        else:
            return _generate_mock_health_scores(cow_id, start_time, end_time)
        
        # Ensure timestamp is datetime
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error querying health scores: {e}")
        return _generate_mock_health_scores(cow_id, start_time, end_time)


def calculate_baseline_health_score(
    cow_id: int,
    baseline_days: int = 30,
    end_time: Optional[datetime] = None,
    connection=None
) -> Tuple[float, datetime, datetime]:
    """
    Calculate baseline health score (rolling average over baseline period).
    
    Args:
        cow_id: Cow identifier
        baseline_days: Number of days to use for baseline calculation
        end_time: End time for baseline calculation (default: now)
        connection: Database connection (if None, will use mock data)
        
    Returns:
        Tuple of (baseline_score, baseline_start_time, baseline_end_time)
    """
    if end_time is None:
        end_time = datetime.now()
    
    start_time = end_time - timedelta(days=baseline_days)
    
    # Query health scores for baseline period
    df = query_health_scores(cow_id, start_time, end_time, connection)
    
    if df.empty or 'health_score' not in df.columns:
        logger.warning(f"No health scores found for cow {cow_id} in baseline period")
        return 75.0, start_time, end_time  # Default baseline
    
    # Calculate baseline as mean of health scores
    baseline_score = float(df['health_score'].mean())
    
    return baseline_score, start_time, end_time


def get_contributing_factors(
    cow_id: int,
    timestamp: datetime,
    connection=None
) -> Dict[str, float]:
    """
    Get contributing factors breakdown for a specific health score.
    
    Args:
        cow_id: Cow identifier
        timestamp: Timestamp of health score
        connection: Database connection (if None, will use mock data)
        
    Returns:
        Dictionary with contributing factors as percentages (summing to 100)
    """
    # Query the specific health score
    end_time = timestamp + timedelta(minutes=1)
    df = query_health_scores(cow_id, timestamp, end_time, connection)
    
    if df.empty:
        # Return default breakdown
        return {
            'temperature_stability': 25.0,
            'activity_level': 25.0,
            'behavioral_consistency': 25.0,
            'rumination_quality': 20.0,
            'alert_impact': 5.0,
        }
    
    # Get the first (and should be only) row
    row = df.iloc[0]
    
    # Extract components and convert to percentages
    # Each component is 0-1, we need to normalize to percentage contributions
    temp_comp = row.get('temperature_component', 0.25)
    activity_comp = row.get('activity_component', 0.25)
    behavior_comp = row.get('behavior_component', 0.25)
    rumination_comp = row.get('rumination_component', 0.20)
    alert_penalty = row.get('alert_penalty', 0.05)
    
    # Calculate weights (these should sum to ~1.0)
    total = temp_comp + activity_comp + behavior_comp + rumination_comp
    
    if total > 0:
        # Normalize to percentages
        factors = {
            'temperature_stability': (temp_comp / total) * 95,  # Reserve 5% for alerts
            'activity_level': (activity_comp / total) * 95,
            'behavioral_consistency': (behavior_comp / total) * 95,
            'rumination_quality': (rumination_comp / total) * 95,
            'alert_impact': 5.0,  # Fixed 5% for alerts
        }
    else:
        # Default equal distribution
        factors = {
            'temperature_stability': 25.0,
            'activity_level': 25.0,
            'behavioral_consistency': 25.0,
            'rumination_quality': 20.0,
            'alert_impact': 5.0,
        }
    
    return factors


def get_latest_health_score(
    cow_id: int,
    connection=None
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent health score for a cow.
    
    Args:
        cow_id: Cow identifier
        connection: Database connection (if None, will use mock data)
        
    Returns:
        Dictionary with latest health score data or None if not found
    """
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)  # Look back 1 day
    
    df = query_health_scores(cow_id, start_time, end_time, connection)
    
    if df.empty:
        return None
    
    # Get the most recent row
    latest = df.iloc[-1]
    
    return {
        'timestamp': latest['timestamp'],
        'health_score': int(latest['health_score']),
        'temperature_component': float(latest.get('temperature_component', 0)),
        'activity_component': float(latest.get('activity_component', 0)),
        'behavior_component': float(latest.get('behavior_component', 0)),
        'rumination_component': float(latest.get('rumination_component', 0)),
        'alert_penalty': float(latest.get('alert_penalty', 0)),
        'trend_direction': latest.get('trend_direction', 'stable'),
        'trend_rate': float(latest.get('trend_rate', 0)),
    }


def _generate_mock_health_scores(
    cow_id: int,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame:
    """
    Generate mock health score data for demonstration purposes.
    
    Args:
        cow_id: Cow identifier
        start_time: Start of time range
        end_time: End of time range
        
    Returns:
        DataFrame with mock health scores
    """
    # Generate timestamps at 1-hour intervals
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1H')
    
    if len(timestamps) == 0:
        # Return empty dataframe with proper columns
        return pd.DataFrame(columns=[
            'timestamp', 'cow_id', 'health_score', 'temperature_component',
            'activity_component', 'behavior_component', 'rumination_component',
            'alert_penalty', 'trend_direction', 'trend_rate', 'days_since_baseline',
            'contributing_factors'
        ])
    
    # Generate realistic health score patterns
    # Start with a base score and add some variation
    base_score = 75
    
    health_scores = []
    temp_components = []
    activity_components = []
    behavior_components = []
    rumination_components = []
    alert_penalties = []
    trend_directions = []
    trend_rates = []
    
    for i, ts in enumerate(timestamps):
        # Add time-based variation (slight decline during hot parts of day)
        hour = ts.hour
        time_variation = -5 if 12 <= hour <= 16 else 0
        
        # Add random walk variation
        if i == 0:
            score = base_score + time_variation
        else:
            # Random walk with mean reversion
            prev_score = health_scores[-1]
            change = np.random.normal(0, 2) + (base_score - prev_score) * 0.1
            score = prev_score + change + time_variation
        
        # Clamp to 0-100
        score = max(0, min(100, score))
        health_scores.append(score)
        
        # Generate component scores (0-1 scale)
        # These should roughly correlate with overall health score
        score_factor = score / 100.0
        temp_components.append(min(1.0, max(0.0, score_factor + np.random.normal(0, 0.1))))
        activity_components.append(min(1.0, max(0.0, score_factor + np.random.normal(0, 0.1))))
        behavior_components.append(min(1.0, max(0.0, score_factor + np.random.normal(0, 0.1))))
        rumination_components.append(min(1.0, max(0.0, score_factor + np.random.normal(0, 0.1))))
        
        # Alert penalty (usually small)
        alert_penalty = 0.05 if np.random.random() < 0.1 else 0.0
        alert_penalties.append(alert_penalty)
        
        # Determine trend
        if i < 5:
            trend_direction = 'stable'
            trend_rate = 0.0
        else:
            recent_scores = health_scores[-5:]
            if recent_scores[-1] > recent_scores[0] + 2:
                trend_direction = 'improving'
                trend_rate = (recent_scores[-1] - recent_scores[0]) / 5
            elif recent_scores[-1] < recent_scores[0] - 2:
                trend_direction = 'deteriorating'
                trend_rate = (recent_scores[-1] - recent_scores[0]) / 5
            else:
                trend_direction = 'stable'
                trend_rate = 0.0
        
        trend_directions.append(trend_direction)
        trend_rates.append(trend_rate)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cow_id': cow_id,
        'health_score': health_scores,
        'temperature_component': temp_components,
        'activity_component': activity_components,
        'behavior_component': behavior_components,
        'rumination_component': rumination_components,
        'alert_penalty': alert_penalties,
        'trend_direction': trend_directions,
        'trend_rate': trend_rates,
        'days_since_baseline': 0,
        'contributing_factors': None,
    })
    
    return df
