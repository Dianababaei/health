"""
Sensor Data Management Module

Provides persistent sensor data storage with SQLite database.
Implements best practices for livestock health monitoring:
- Append new data to existing records (no overwrites)
- Rolling window calculations for health scores
- Long-term baseline tracking

References:
- IoT-Based Cow Health Monitoring System (PMC7302546)
- CowManager multi-level data aggregation approach
"""

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SensorDataManager:
    """
    Manages sensor data with persistent SQLite storage.

    Features:
    - Append new sensor data (no overwrites)
    - Query by cow ID and time range
    - Rolling window data retrieval for health score calculation
    - Automatic deduplication by timestamp
    """

    def __init__(self, db_path: str = "data/alert_state.db"):
        """
        Initialize sensor data manager.

        Args:
            db_path: Path to SQLite database file (shared with alerts)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(f"SensorDataManager initialized: db={self.db_path}")

    def _init_database(self):
        """Initialize database schema for sensor data."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create sensor_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cow_id TEXT NOT NULL,
                    temperature REAL,
                    fxa REAL,
                    mya REAL,
                    rza REAL,
                    sxg REAL,
                    lyg REAL,
                    dzg REAL,
                    state TEXT,
                    motion_intensity REAL,
                    created_at TEXT NOT NULL,
                    UNIQUE(cow_id, timestamp)
                )
            """)

            # Create indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_data_cow_id
                ON sensor_data (cow_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp
                ON sensor_data (timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_data_cow_timestamp
                ON sensor_data (cow_id, timestamp)
            """)

            conn.commit()
            conn.close()
            logger.info("Sensor data database schema initialized")

        except Exception as e:
            logger.error(f"Error initializing sensor data database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def append_sensor_data(self, df: pd.DataFrame, cow_id: str) -> Tuple[int, int]:
        """
        Append sensor data to database (no overwrites, skip duplicates).

        Args:
            df: DataFrame with sensor columns (timestamp, temperature, fxa, mya, rza, etc.)
            cow_id: Cow identifier

        Returns:
            Tuple of (inserted_count, skipped_count)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            now = datetime.now().isoformat()

            inserted = 0
            skipped = 0

            for _, row in df.iterrows():
                try:
                    timestamp = row['timestamp']
                    if isinstance(timestamp, pd.Timestamp):
                        timestamp = timestamp.isoformat()

                    cursor.execute("""
                        INSERT OR IGNORE INTO sensor_data (
                            timestamp, cow_id, temperature, fxa, mya, rza,
                            sxg, lyg, dzg, state, motion_intensity, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        cow_id,
                        row.get('temperature'),
                        row.get('fxa'),
                        row.get('mya'),
                        row.get('rza'),
                        row.get('sxg'),
                        row.get('lyg'),
                        row.get('dzg'),
                        row.get('state'),
                        row.get('motion_intensity'),
                        now
                    ))

                    if cursor.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1

                except Exception as e:
                    logger.warning(f"Error inserting row: {e}")
                    skipped += 1

            conn.commit()
            conn.close()

            logger.info(f"Sensor data append: inserted={inserted}, skipped={skipped} (duplicates)")
            return inserted, skipped

        except Exception as e:
            logger.error(f"Error appending sensor data: {e}")
            return 0, len(df)

    def get_sensor_data(
        self,
        cow_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query sensor data for a cow.

        Args:
            cow_id: Cow identifier
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            limit: Maximum rows to return (optional)

        Returns:
            DataFrame with sensor data
        """
        try:
            conn = self._get_connection()

            query = """
                SELECT timestamp, cow_id, temperature, fxa, mya, rza,
                       sxg, lyg, dzg, state, motion_intensity
                FROM sensor_data
                WHERE cow_id = ?
            """
            params = [cow_id]

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            query += " ORDER BY timestamp ASC"

            if limit:
                query += f" LIMIT {limit}"

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            return df

        except Exception as e:
            logger.error(f"Error querying sensor data: {e}")
            return pd.DataFrame()

    def get_rolling_window_data(
        self,
        cow_id: str,
        window_hours: int = 24
    ) -> pd.DataFrame:
        """
        Get sensor data for rolling window health score calculation.

        Best Practice: Use recent 24-hour window for current health score.

        Args:
            cow_id: Cow identifier
            window_hours: Hours of data to retrieve (default: 24)

        Returns:
            DataFrame with recent sensor data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=window_hours)
        return self.get_sensor_data(cow_id, start_time, end_time)

    def get_baseline_data(
        self,
        cow_id: str,
        baseline_days: int = 7
    ) -> pd.DataFrame:
        """
        Get sensor data for baseline calculation.

        Best Practice: Use 7-day rolling baseline for comparison.

        Args:
            cow_id: Cow identifier
            baseline_days: Days of data for baseline (default: 7)

        Returns:
            DataFrame with baseline period data
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=baseline_days)
        return self.get_sensor_data(cow_id, start_time, end_time)

    def get_all_data(self, cow_id: str) -> pd.DataFrame:
        """
        Get all sensor data for a cow (for historical analysis).

        Args:
            cow_id: Cow identifier

        Returns:
            DataFrame with all sensor data
        """
        return self.get_sensor_data(cow_id)

    def get_data_statistics(self, cow_id: str) -> Dict[str, Any]:
        """
        Get statistics about stored sensor data.

        Args:
            cow_id: Cow identifier

        Returns:
            Dictionary with data statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Total records
            cursor.execute(
                "SELECT COUNT(*) as count FROM sensor_data WHERE cow_id = ?",
                (cow_id,)
            )
            total = cursor.fetchone()['count']

            # Date range
            cursor.execute("""
                SELECT MIN(timestamp) as first, MAX(timestamp) as last
                FROM sensor_data WHERE cow_id = ?
            """, (cow_id,))
            row = cursor.fetchone()

            # Days of data
            if row['first'] and row['last']:
                first_dt = datetime.fromisoformat(row['first'])
                last_dt = datetime.fromisoformat(row['last'])
                days = (last_dt - first_dt).days + 1
            else:
                days = 0

            conn.close()

            return {
                'total_records': total,
                'first_timestamp': row['first'],
                'last_timestamp': row['last'],
                'days_of_data': days,
                'cow_id': cow_id
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def calculate_baseline_temperature(
        self,
        cow_id: str,
        baseline_days: int = 7
    ) -> Optional[float]:
        """
        Calculate baseline temperature from recent data.

        Best Practice: Use median of 7-day window to establish individual baseline.

        Args:
            cow_id: Cow identifier
            baseline_days: Days to use for baseline calculation

        Returns:
            Baseline temperature or None if insufficient data
        """
        try:
            df = self.get_baseline_data(cow_id, baseline_days)

            if df.empty or 'temperature' not in df.columns:
                return None

            temps = df['temperature'].dropna()
            if len(temps) < 100:  # Need at least ~100 samples
                return None

            # Use median (robust to outliers like fever spikes)
            return float(temps.median())

        except Exception as e:
            logger.error(f"Error calculating baseline temperature: {e}")
            return None

    def delete_old_data(self, days: int = 90) -> int:
        """
        Delete sensor data older than specified days.

        Best Practice: Keep 90 days for long-term baseline analysis.

        Args:
            days: Delete data older than this many days

        Returns:
            Number of deleted records
        """
        try:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM sensor_data WHERE timestamp < ?",
                (cutoff,)
            )

            deleted = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Deleted {deleted} old sensor data records (>{days} days)")
            return deleted

        except Exception as e:
            logger.error(f"Error deleting old data: {e}")
            return 0
