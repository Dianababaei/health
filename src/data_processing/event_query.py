"""
Event Query Module
==================
Query events from multiple sources (alerts, behavioral states, physiological metrics, 
sensor malfunctions) by type and date range for timeline visualization.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def query_alerts(
    connection,
    cow_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    alert_types: Optional[List[str]] = None,
    severities: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Query alerts from the alerts table.
    
    Args:
        connection: Database connection
        cow_id: Filter by cow ID (optional)
        start_time: Start of time range (optional)
        end_time: End of time range (optional)
        alert_types: Filter by alert types (optional)
        severities: Filter by severities (optional)
        
    Returns:
        DataFrame with columns: timestamp, cow_id, alert_type, severity, 
        title, details, status, sensor_values
    """
    query = """
        SELECT 
            timestamp,
            cow_id,
            alert_type,
            severity,
            title,
            details,
            status,
            sensor_values,
            alert_id
        FROM alerts
        WHERE 1=1
    """
    
    params = []
    
    if cow_id is not None:
        query += " AND cow_id = %s"
        params.append(cow_id)
    
    if start_time is not None:
        query += " AND timestamp >= %s"
        params.append(start_time)
    
    if end_time is not None:
        query += " AND timestamp <= %s"
        params.append(end_time)
    
    if alert_types:
        placeholders = ','.join(['%s'] * len(alert_types))
        query += f" AND alert_type IN ({placeholders})"
        params.extend(alert_types)
    
    if severities:
        placeholders = ','.join(['%s'] * len(severities))
        query += f" AND severity IN ({placeholders})"
        params.extend(severities)
    
    query += " ORDER BY timestamp ASC"
    
    try:
        df = pd.read_sql(query, connection, params=params)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Error querying alerts: {e}")
        return pd.DataFrame()


def query_behavioral_transitions(
    connection,
    cow_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Query behavioral state transitions (changes in state).
    
    Args:
        connection: Database connection
        cow_id: Filter by cow ID (optional)
        start_time: Start of time range (optional)
        end_time: End of time range (optional)
        
    Returns:
        DataFrame with behavioral state transition events
    """
    # Get behavioral states with lag to detect transitions
    query = """
        WITH state_changes AS (
            SELECT 
                timestamp,
                cow_id,
                state,
                confidence,
                LAG(state) OVER (PARTITION BY cow_id ORDER BY timestamp) AS prev_state
            FROM behavioral_states
            WHERE 1=1
    """
    
    params = []
    
    if cow_id is not None:
        query += " AND cow_id = %s"
        params.append(cow_id)
    
    if start_time is not None:
        query += " AND timestamp >= %s"
        params.append(start_time)
    
    if end_time is not None:
        query += " AND timestamp <= %s"
        params.append(end_time)
    
    query += """
        )
        SELECT 
            timestamp,
            cow_id,
            state,
            prev_state,
            confidence
        FROM state_changes
        WHERE prev_state IS NOT NULL AND state != prev_state
        ORDER BY timestamp ASC
    """
    
    try:
        df = pd.read_sql(query, connection, params=params)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Error querying behavioral transitions: {e}")
        return pd.DataFrame()


def query_temperature_anomalies(
    connection,
    cow_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    anomaly_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Query temperature anomaly events from physiological_metrics.
    
    Args:
        connection: Database connection
        cow_id: Filter by cow ID (optional)
        start_time: Start of time range (optional)
        end_time: End of time range (optional)
        anomaly_threshold: Minimum anomaly score to include (default: 0.7)
        
    Returns:
        DataFrame with temperature anomaly events
    """
    query = """
        SELECT 
            timestamp,
            cow_id,
            current_temp,
            baseline_temp,
            temp_deviation,
            temp_anomaly_score,
            circadian_rhythm_stability
        FROM physiological_metrics
        WHERE temp_anomaly_score >= %s
    """
    
    params = [anomaly_threshold]
    
    if cow_id is not None:
        query += " AND cow_id = %s"
        params.append(cow_id)
    
    if start_time is not None:
        query += " AND timestamp >= %s"
        params.append(start_time)
    
    if end_time is not None:
        query += " AND timestamp <= %s"
        params.append(end_time)
    
    query += " ORDER BY timestamp ASC"
    
    try:
        df = pd.read_sql(query, connection, params=params)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Error querying temperature anomalies: {e}")
        return pd.DataFrame()


def query_sensor_malfunctions(
    connection,
    cow_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Query sensor malfunction events from raw_sensor_readings.
    
    Detects malfunctions by:
    - Poor data quality indicators
    - Gaps in sensor data (>5 minutes)
    
    Args:
        connection: Database connection
        cow_id: Filter by cow ID (optional)
        start_time: Start of time range (optional)
        end_time: End of time range (optional)
        
    Returns:
        DataFrame with sensor malfunction events
    """
    # Query for poor data quality
    quality_query = """
        SELECT 
            timestamp,
            cow_id,
            sensor_id,
            data_quality,
            temperature
        FROM raw_sensor_readings
        WHERE data_quality != 'good'
    """
    
    params = []
    
    if cow_id is not None:
        quality_query += " AND cow_id = %s"
        params.append(cow_id)
    
    if start_time is not None:
        quality_query += " AND timestamp >= %s"
        params.append(start_time)
    
    if end_time is not None:
        quality_query += " AND timestamp <= %s"
        params.append(end_time)
    
    quality_query += " ORDER BY timestamp ASC"
    
    try:
        df = pd.read_sql(quality_query, connection, params=params)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        logger.error(f"Error querying sensor malfunctions: {e}")
        return pd.DataFrame()


def query_all_events(
    connection,
    cow_id: Optional[int] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    event_types: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Query all event types within a time range.
    
    Args:
        connection: Database connection
        cow_id: Filter by cow ID (optional)
        start_time: Start of time range (optional, default: 7 days ago)
        end_time: End of time range (optional, default: now)
        event_types: Filter by event types (optional)
            Options: 'alerts', 'behavioral', 'temperature', 'sensor'
        
    Returns:
        Dictionary with event type keys and DataFrame values
    """
    # Set defaults for time range
    if end_time is None:
        end_time = datetime.now()
    if start_time is None:
        start_time = end_time - timedelta(days=7)
    
    # Determine which event types to query
    if event_types is None:
        event_types = ['alerts', 'behavioral', 'temperature', 'sensor']
    
    results = {}
    
    # Query each event type
    if 'alerts' in event_types:
        logger.info(f"Querying alerts from {start_time} to {end_time}")
        results['alerts'] = query_alerts(
            connection, cow_id, start_time, end_time
        )
    
    if 'behavioral' in event_types:
        logger.info(f"Querying behavioral transitions from {start_time} to {end_time}")
        results['behavioral'] = query_behavioral_transitions(
            connection, cow_id, start_time, end_time
        )
    
    if 'temperature' in event_types:
        logger.info(f"Querying temperature anomalies from {start_time} to {end_time}")
        results['temperature'] = query_temperature_anomalies(
            connection, cow_id, start_time, end_time
        )
    
    if 'sensor' in event_types:
        logger.info(f"Querying sensor malfunctions from {start_time} to {end_time}")
        results['sensor'] = query_sensor_malfunctions(
            connection, cow_id, start_time, end_time
        )
    
    return results


def get_event_date_range(connection, cow_id: Optional[int] = None) -> Tuple[datetime, datetime]:
    """
    Get the date range of available event data.
    
    Args:
        connection: Database connection
        cow_id: Filter by cow ID (optional)
        
    Returns:
        Tuple of (earliest_timestamp, latest_timestamp)
    """
    query = """
        SELECT 
            MIN(timestamp) as min_time,
            MAX(timestamp) as max_time
        FROM (
            SELECT timestamp FROM alerts WHERE 1=1 {cow_filter}
            UNION ALL
            SELECT timestamp FROM behavioral_states WHERE 1=1 {cow_filter}
            UNION ALL
            SELECT timestamp FROM physiological_metrics WHERE 1=1 {cow_filter}
        ) AS all_events
    """
    
    cow_filter = f"AND cow_id = {cow_id}" if cow_id is not None else ""
    query = query.format(cow_filter=cow_filter)
    
    try:
        result = pd.read_sql(query, connection)
        if not result.empty and result['min_time'].iloc[0] is not None:
            return (
                pd.to_datetime(result['min_time'].iloc[0]),
                pd.to_datetime(result['max_time'].iloc[0])
            )
    except Exception as e:
        logger.error(f"Error getting event date range: {e}")
    
    # Return default range if query fails
    now = datetime.now()
    return (now - timedelta(days=30), now)
