"""
Database Connection Utilities
=============================
Utilities for connecting to the SQLite database and querying
behavioral state data for the dashboard.

NOTE: This project uses SQLite only. PostgreSQL support has been removed.
"""

import os
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import streamlit as st


def get_database_connection():
    """
    Get a SQLite database connection.

    This function is deprecated and kept for backward compatibility.
    Direct use of SQLite connections is recommended instead.

    Returns:
        None (SQLite connections are created per-query)
    """
    # Return None - behavioral states are not stored in database yet
    # Health scores use HealthScoreManager directly
    return None


def query_behavioral_states(
    cow_id: str,
    start_time: datetime,
    end_time: datetime,
    connection=None
) -> pd.DataFrame:
    """
    Query behavioral states for a specific cow and time range.

    NOTE: Behavioral states are not currently stored in database.
    This function returns empty DataFrame. Behavioral states are
    loaded from CSV files via DataLoader instead.

    Args:
        cow_id: Cow identifier
        start_time: Start of time range (inclusive)
        end_time: End of time range (inclusive)
        connection: Database connection (DEPRECATED - not used)

    Returns:
        Empty DataFrame (behavioral states not in database)
    """
    # Behavioral states are loaded from CSV files, not database
    return pd.DataFrame()


def get_available_cows(connection=None) -> List[str]:
    """
    Get list of cow IDs that have data.

    NOTE: Currently returns empty list as behavioral states
    are not stored in database. Use DataLoader to get available
    cow IDs from CSV files.

    Args:
        connection: Database connection (DEPRECATED - not used)

    Returns:
        Empty list (cow IDs loaded from CSV files instead)
    """
    # Cow IDs are determined from CSV files in data/dashboard folder
    return []


@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_query_behavioral_states(
    cow_id: str,
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
        Empty DataFrame (behavioral states not in database)
    """
    return query_behavioral_states(cow_id, start_time, end_time)


@st.cache_data(ttl=600)  # Cache for 10 minutes
def cached_get_available_cows() -> List[str]:
    """
    Cached version of get_available_cows for better performance.

    Returns:
        Empty list (cow IDs loaded from CSV files instead)
    """
    return get_available_cows()
