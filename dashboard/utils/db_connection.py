"""
Database Connection Utilities
=============================
Utilities for connecting to the TimescaleDB database and querying
behavioral state data for the dashboard.
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import streamlit as st

# Try to import database libraries
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False


def get_database_connection():
    """
    Get a database connection using connection pooling.
    
    Uses Streamlit's caching to maintain a connection pool across
    the session. Falls back to mock data if database is unavailable.
    
    Returns:
        Database connection object or None if unavailable
    """
    database_url = os.getenv('DATABASE_URL', 'postgresql://username:password@localhost:5432/artemis_health')
    
    # Check if we should use mock data
    use_mock = os.getenv('USE_MOCK_DATA', 'false').lower() == 'true'
    if use_mock:
        return None
    
    # Try psycopg2 first (more efficient for simple queries)
    if HAS_PSYCOPG2 and database_url.startswith('postgresql'):
        try:
            conn = psycopg2.connect(database_url)
            return conn
        except Exception as e:
            st.warning(f"Database connection failed: {e}. Using mock data.")
            return None
    
    # Try SQLAlchemy (more flexible)
    if HAS_SQLALCHEMY:
        try:
            engine = create_engine(database_url, poolclass=NullPool)
            conn = engine.connect()
            return conn
        except Exception as e:
            st.warning(f"Database connection failed: {e}. Using mock data.")
            return None
    
    # No database libraries available
    st.warning("No database libraries available (psycopg2 or sqlalchemy). Using mock data.")
    return None


def query_behavioral_states(
    cow_id: int,
    start_time: datetime,
    end_time: datetime,
    connection=None
) -> pd.DataFrame:
    """
    Query behavioral states for a specific cow and time range.
    
    Args:
        cow_id: Cow identifier
        start_time: Start of time range (inclusive)
        end_time: End of time range (inclusive)
        connection: Database connection (if None, will create one)
        
    Returns:
        DataFrame with columns: timestamp, state, confidence, duration_minutes,
        motion_intensity, posture_context
    """
    # If no connection provided, try to get one
    if connection is None:
        connection = get_database_connection()
    
    # If still no connection, return mock data
    if connection is None:
        return _generate_mock_behavioral_data(cow_id, start_time, end_time)
    
    # Build query
    query = """
        SELECT 
            timestamp,
            cow_id,
            state,
            confidence,
            duration_minutes,
            motion_intensity,
            posture_context
        FROM behavioral_states
        WHERE cow_id = %s
            AND timestamp >= %s
            AND timestamp <= %s
        ORDER BY timestamp ASC
    """
    
    try:
        # Execute query
        if HAS_PSYCOPG2 and hasattr(connection, 'cursor'):
            # psycopg2 connection
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(query, (cow_id, start_time, end_time))
            rows = cursor.fetchall()
            cursor.close()
            df = pd.DataFrame(rows)
        elif HAS_SQLALCHEMY:
            # SQLAlchemy connection
            df = pd.read_sql(query, connection, params=(cow_id, start_time, end_time))
        else:
            return _generate_mock_behavioral_data(cow_id, start_time, end_time)
        
        # Ensure timestamp is datetime
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        st.error(f"Error querying behavioral states: {e}")
        return _generate_mock_behavioral_data(cow_id, start_time, end_time)


def get_available_cows(connection=None) -> List[int]:
    """
    Get list of cow IDs that have behavioral state data.
    
    Args:
        connection: Database connection (if None, will create one)
        
    Returns:
        List of cow IDs
    """
    # If no connection provided, try to get one
    if connection is None:
        connection = get_database_connection()
    
    # If still no connection, return mock cow IDs
    if connection is None:
        return [1001, 1002, 1003, 1004, 1005]
    
    # Build query
    query = """
        SELECT DISTINCT cow_id
        FROM behavioral_states
        ORDER BY cow_id
        LIMIT 100
    """
    
    try:
        # Execute query
        if HAS_PSYCOPG2 and hasattr(connection, 'cursor'):
            # psycopg2 connection
            cursor = connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            cow_ids = [row[0] for row in rows]
        elif HAS_SQLALCHEMY:
            # SQLAlchemy connection
            result = connection.execute(query)
            cow_ids = [row[0] for row in result]
        else:
            return [1001, 1002, 1003, 1004, 1005]
        
        return cow_ids if cow_ids else [1001, 1002, 1003, 1004, 1005]
        
    except Exception as e:
        st.warning(f"Error getting cow list: {e}. Using mock data.")
        return [1001, 1002, 1003, 1004, 1005]


def _generate_mock_behavioral_data(
    cow_id: int,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame:
    """
    Generate mock behavioral state data for demonstration purposes.
    
    Args:
        cow_id: Cow identifier
        start_time: Start of time range
        end_time: End of time range
        
    Returns:
        DataFrame with mock behavioral states
    """
    import numpy as np
    
    # Generate timestamps at 1-minute intervals
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    # Behavioral state patterns (simulate realistic daily patterns)
    # Morning: lying -> standing -> feeding -> walking -> ruminating
    # Afternoon: standing -> walking -> feeding -> ruminating -> lying
    # Evening: lying -> ruminating -> standing -> lying
    
    states = []
    confidences = []
    durations = []
    motion_intensities = []
    posture_contexts = []
    
    for i, ts in enumerate(timestamps):
        hour = ts.hour
        minute = ts.minute
        
        # Simulate circadian patterns
        if 0 <= hour < 6:  # Night: mostly lying with some ruminating
            state = 'lying' if np.random.random() < 0.8 else 'ruminating'
            confidence = np.random.uniform(0.85, 0.98)
            motion_intensity = np.random.uniform(0.0, 0.15)
            posture_context = 'lying'
        elif 6 <= hour < 9:  # Early morning: standing, feeding
            state = np.random.choice(['standing', 'feeding', 'walking'], p=[0.4, 0.4, 0.2])
            confidence = np.random.uniform(0.80, 0.95)
            motion_intensity = np.random.uniform(0.2, 0.6)
            posture_context = 'standing' if state != 'lying' else None
        elif 9 <= hour < 12:  # Mid-morning: ruminating, walking
            state = np.random.choice(['ruminating', 'standing', 'walking'], p=[0.5, 0.3, 0.2])
            confidence = np.random.uniform(0.85, 0.98)
            motion_intensity = np.random.uniform(0.1, 0.4)
            posture_context = 'standing' if state == 'ruminating' else ('standing' if state != 'lying' else None)
        elif 12 <= hour < 15:  # Midday: lying, standing
            state = np.random.choice(['lying', 'standing', 'ruminating'], p=[0.5, 0.3, 0.2])
            confidence = np.random.uniform(0.82, 0.96)
            motion_intensity = np.random.uniform(0.0, 0.3)
            posture_context = 'lying' if state in ['lying', 'ruminating'] else 'standing'
        elif 15 <= hour < 18:  # Afternoon: feeding, walking
            state = np.random.choice(['feeding', 'walking', 'standing'], p=[0.4, 0.3, 0.3])
            confidence = np.random.uniform(0.80, 0.94)
            motion_intensity = np.random.uniform(0.3, 0.7)
            posture_context = 'standing'
        else:  # Evening: lying, ruminating
            state = np.random.choice(['lying', 'ruminating', 'standing'], p=[0.5, 0.3, 0.2])
            confidence = np.random.uniform(0.83, 0.97)
            motion_intensity = np.random.uniform(0.0, 0.25)
            posture_context = 'lying' if state in ['lying', 'ruminating'] else 'standing'
        
        states.append(state)
        confidences.append(confidence)
        motion_intensities.append(motion_intensity)
        posture_contexts.append(posture_context)
        
        # Duration is null for ongoing states (will be calculated later)
        durations.append(None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'cow_id': cow_id,
        'state': states,
        'confidence': confidences,
        'duration_minutes': durations,
        'motion_intensity': motion_intensities,
        'posture_context': posture_contexts,
    })
    
    return df


@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_query_behavioral_states(
    cow_id: int,
    start_time: datetime,
    end_time: datetime
) -> pd.DataFrame:
    """
    Cached version of query_behavioral_states for better performance.
    
    Args:
        cow_id: Cow identifier
        start_time: Start of time range
        end_time: End of time range
        
    Returns:
        DataFrame with behavioral states
    """
    return query_behavioral_states(cow_id, start_time, end_time)


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_get_available_cows() -> List[int]:
    """
    Cached version of get_available_cows for better performance.
    
    Returns:
        List of cow IDs
    """
    return get_available_cows()
