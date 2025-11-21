"""
Health Score Data Loader
=========================
Utilities for loading and processing health score data from SQLite database.
Includes baseline calculations, contributing factors retrieval.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def query_health_scores(
    cow_id: str,
    start_time: datetime,
    end_time: datetime,
    connection=None
) -> pd.DataFrame:
    """
    Query health scores for a specific cow and time range from SQLite database.

    Args:
        cow_id: Cow identifier
        start_time: Start of time range (inclusive)
        end_time: End of time range (inclusive)
        connection: Database connection (DEPRECATED - always uses SQLite)

    Returns:
        DataFrame with columns: timestamp, total_score, temperature_score,
        activity_score, behavioral_score, alert_score, health_category
    """
    # Use SQLite health score manager
    from health_intelligence.logging.health_score_manager import HealthScoreManager

    try:
        manager = HealthScoreManager(db_path="data/alert_state.db")

        # Query health scores from SQLite
        df = manager.query_health_scores(
            cow_id=cow_id,
            start_time=start_time.isoformat() if isinstance(start_time, datetime) else start_time,
            end_time=end_time.isoformat() if isinstance(end_time, datetime) else end_time,
            sort_order="ASC"
        )

        # Rename columns to match expected format
        if not df.empty:
            # Map SQLite columns to expected column names
            df = df.rename(columns={
                'total_score': 'health_score',
                'temperature_score': 'temperature_component',
                'activity_score': 'activity_component',
                'behavioral_score': 'behavior_component',
                'alert_score': 'alert_penalty'
            })

            # Add missing columns with default values
            if 'rumination_component' not in df.columns:
                df['rumination_component'] = 0.0

            if 'trend_direction' not in df.columns:
                df['trend_direction'] = 'stable'

            if 'trend_rate' not in df.columns:
                df['trend_rate'] = 0.0

            if 'days_since_baseline' not in df.columns:
                df['days_since_baseline'] = 0

            if 'contributing_factors' not in df.columns:
                df['contributing_factors'] = None

        return df

    except Exception as e:
        logger.error(f"Error querying health scores from SQLite: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()


def calculate_baseline_health_score(
    cow_id: str,
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
        connection: Database connection (DEPRECATED - always uses SQLite)

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
    cow_id: str,
    timestamp: datetime,
    connection=None
) -> Dict[str, float]:
    """
    Get contributing factors breakdown for a specific health score.

    Args:
        cow_id: Cow identifier
        timestamp: Timestamp of health score
        connection: Database connection (DEPRECATED - always uses SQLite)

    Returns:
        Dictionary with contributing factors as percentages (summing to 100)
    """
    # Query the specific health score
    end_time = timestamp + timedelta(minutes=1)
    df = query_health_scores(cow_id, timestamp, end_time, connection)

    if df.empty:
        # Return default breakdown
        return {
            'temperature_stability': 30.0,
            'activity_level': 25.0,
            'behavioral_consistency': 25.0,
            'rumination_quality': 0.0,
            'alert_impact': 20.0,
        }

    # Get the first (and should be only) row
    row = df.iloc[0]

    # Extract components (already 0-1 normalized)
    temp_comp = row.get('temperature_component', 0.30)
    activity_comp = row.get('activity_component', 0.25)
    behavior_comp = row.get('behavior_component', 0.25)
    alert_comp = row.get('alert_penalty', 0.20)

    # Calculate rumination from current sensor data (not stored in health_scores table)
    rumination_comp = 0.0
    try:
        from pathlib import Path
        import pandas as pd

        sensor_file = Path('data/dashboard') / f'{cow_id}_sensor_data.csv'
        if sensor_file.exists():
            sensor_df = pd.read_csv(sensor_file)
            if 'state' in sensor_df.columns:
                # Count all rumination states
                total_samples = len(sensor_df)
                ruminating_states = ['ruminating', 'ruminating_lying', 'ruminating_standing']
                ruminating_count = sensor_df['state'].isin(ruminating_states).sum()
                rumination_percentage = (ruminating_count / total_samples * 100) if total_samples > 0 else 0
                rumination_comp = rumination_percentage  # Already in 0-100 scale
    except Exception as e:
        # If we can't load sensor data, default to 0
        pass

    # Convert to percentages (components are 0-1 normalized, rumination is already 0-100)
    factors = {
        'temperature_stability': temp_comp * 100,
        'activity_level': activity_comp * 100,
        'behavioral_consistency': behavior_comp * 100,
        'rumination_quality': rumination_comp,  # Already 0-100 scale
        'alert_impact': alert_comp * 100,
    }

    return factors


def get_latest_health_score(
    cow_id: str,
    connection=None
) -> Optional[Dict[str, Any]]:
    """
    Get the most recent health score for a cow.

    Args:
        cow_id: Cow identifier
        connection: Database connection (DEPRECATED - always uses SQLite)

    Returns:
        Dictionary with latest health score data or None if not found
    """
    from health_intelligence.logging.health_score_manager import HealthScoreManager

    try:
        manager = HealthScoreManager(db_path="data/alert_state.db")
        latest = manager.get_latest_score(cow_id)

        if latest is None:
            return None

        # Convert to expected format
        return {
            'timestamp': latest['timestamp'],
            'health_score': int(latest['total_score']),
            'temperature_component': float(latest.get('temperature_score', 0)),
            'activity_component': float(latest.get('activity_score', 0)),
            'behavior_component': float(latest.get('behavioral_score', 0)),
            'rumination_component': 0.0,
            'alert_penalty': float(latest.get('alert_score', 0)),
            'trend_direction': 'stable',
            'trend_rate': 0.0,
        }

    except Exception as e:
        logger.error(f"Error getting latest health score: {e}")
        return None


def get_health_score_statistics(
    cow_id: str,
    connection=None
) -> Dict[str, Any]:
    """
    Get statistics about health scores for a cow.

    Args:
        cow_id: Cow identifier
        connection: Database connection (DEPRECATED - always uses SQLite)

    Returns:
        Dictionary with statistics
    """
    from health_intelligence.logging.health_score_manager import HealthScoreManager

    try:
        manager = HealthScoreManager(db_path="data/alert_state.db")
        stats = manager.get_statistics(cow_id=cow_id)

        return stats

    except Exception as e:
        logger.error(f"Error getting health score statistics: {e}")
        return {}
