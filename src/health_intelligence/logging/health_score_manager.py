"""
Health Score State Management Module

Provides persistent health score tracking with SQLite database.
Stores calculated health scores from the HealthScorer for dashboard display.
"""

import logging
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


logger = logging.getLogger(__name__)


class HealthScoreManager:
    """
    Manages health scores with persistent SQLite storage.

    Features:
    - Store calculated health scores from HealthScorer
    - Query health scores by cow ID and time range
    - Component score breakdown (temperature, activity, behavioral, alert)
    - Time-series health score tracking
    """

    def __init__(self, db_path: str = "data/alert_state.db"):
        """
        Initialize health score manager.

        Args:
            db_path: Path to SQLite database file (shared with alerts)
        """
        self.db_path = Path(db_path)

        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"HealthScoreManager initialized: db={self.db_path}")

    def _init_database(self):
        """Initialize database schema for health scores."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create health_scores table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS health_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cow_id TEXT NOT NULL,
                    total_score REAL NOT NULL,
                    temperature_score REAL NOT NULL,
                    activity_score REAL NOT NULL,
                    behavioral_score REAL NOT NULL,
                    alert_score REAL NOT NULL,
                    health_category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_health_scores_cow_id
                ON health_scores (cow_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_health_scores_timestamp
                ON health_scores (timestamp)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_health_scores_cow_timestamp
                ON health_scores (cow_id, timestamp)
            """)

            conn.commit()
            conn.close()

            logger.info("Health scores database schema initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing health scores database: {e}")
            raise

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable row access by column name
        return conn

    def save_health_score(self, health_score_data: Dict[str, Any]) -> bool:
        """
        Save a health score to the database.

        Args:
            health_score_data: Health score data dictionary from HealthScore.to_dict()

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Attempting to save health score: {health_score_data.get('cow_id', 'unknown')}")

            conn = self._get_connection()
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            # Extract data from health score dict
            timestamp = health_score_data.get('timestamp')
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()

            logger.debug(f"Health score timestamp: {timestamp}")

            cow_id = health_score_data.get('cow_id', 'unknown')
            total_score = health_score_data.get('total_score', 0.0)
            health_category = health_score_data.get('health_category', 'unknown')
            confidence = health_score_data.get('confidence', 0.0)

            # Component scores (already normalized 0-1)
            temperature_score = health_score_data.get('temperature_component', 0.0)
            activity_score = health_score_data.get('activity_component', 0.0)
            behavioral_score = health_score_data.get('behavioral_component', 0.0)
            alert_score = health_score_data.get('alert_component', 0.0)

            logger.debug(f"Component scores - temp:{temperature_score}, activity:{activity_score}, behavioral:{behavioral_score}, alert:{alert_score}")

            # Store full metadata as JSON
            metadata = json.dumps(health_score_data)

            cursor.execute("""
                INSERT INTO health_scores (
                    timestamp, cow_id, total_score,
                    temperature_score, activity_score, behavioral_score, alert_score,
                    health_category, confidence, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, cow_id, total_score,
                temperature_score, activity_score, behavioral_score, alert_score,
                health_category, confidence, metadata, now
            ))

            conn.commit()
            conn.close()

            logger.info(f"✅ Health score saved successfully: cow_id={cow_id}, score={total_score:.2f}, category={health_category}")
            return True

        except Exception as e:
            logger.error(f"❌ Error saving health score: {e}", exc_info=True)
            return False

    def query_health_scores(
        self,
        cow_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: Optional[int] = None,
        sort_order: str = "DESC"
    ) -> pd.DataFrame:
        """
        Query health scores with filters.

        Args:
            cow_id: Filter by cow ID
            start_time: Filter by start time (ISO8601)
            end_time: Filter by end time (ISO8601)
            limit: Maximum number of results
            sort_order: Sort order (ASC or DESC)

        Returns:
            DataFrame with health scores
        """
        try:
            conn = self._get_connection()

            # Build query
            query = """
                SELECT
                    timestamp, cow_id, total_score,
                    temperature_score, activity_score, behavioral_score, alert_score,
                    health_category, confidence, metadata
                FROM health_scores
                WHERE 1=1
            """
            params = []

            if cow_id:
                query += " AND cow_id = ?"
                params.append(cow_id)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)

            # Add sorting
            query += f" ORDER BY timestamp {sort_order}"

            # Add limit
            if limit:
                query += " LIMIT ?"
                params.append(limit)

            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            # Parse timestamp column
            if not df.empty and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Parse metadata JSON
            if not df.empty and 'metadata' in df.columns:
                df['metadata'] = df['metadata'].apply(
                    lambda x: json.loads(x) if x else {}
                )

            logger.info(f"Queried {len(df)} health score records")
            return df

        except Exception as e:
            logger.error(f"Error querying health scores: {e}")
            return pd.DataFrame()

    def get_latest_score(self, cow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent health score for a cow.

        Args:
            cow_id: Cow identifier

        Returns:
            Health score dictionary or None if not found
        """
        try:
            df = self.query_health_scores(cow_id=cow_id, limit=1, sort_order="DESC")

            if df.empty:
                return None

            # Convert row to dict
            row = df.iloc[0]
            return {
                'timestamp': row['timestamp'],
                'cow_id': row['cow_id'],
                'total_score': row['total_score'],
                'temperature_score': row['temperature_score'],
                'activity_score': row['activity_score'],
                'behavioral_score': row['behavioral_score'],
                'alert_score': row['alert_score'],
                'health_category': row['health_category'],
                'confidence': row['confidence'],
                'metadata': row['metadata']
            }

        except Exception as e:
            logger.error(f"Error getting latest score: {e}")
            return None

    def get_score_history(
        self,
        cow_id: str,
        days: int = 7
    ) -> pd.DataFrame:
        """
        Get health score history for a cow.

        Args:
            cow_id: Cow identifier
            days: Number of days of history

        Returns:
            DataFrame with health score history
        """
        try:
            from datetime import timedelta

            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            df = self.query_health_scores(
                cow_id=cow_id,
                start_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
                sort_order="ASC"
            )

            return df

        except Exception as e:
            logger.error(f"Error getting score history: {e}")
            return pd.DataFrame()

    def get_statistics(self, cow_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get health score statistics.

        Args:
            cow_id: Optional cow ID filter

        Returns:
            Dictionary with statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            stats = {}

            where_clause = "WHERE cow_id = ?" if cow_id else ""
            params = [cow_id] if cow_id else []

            # Total scores
            cursor.execute(
                f"SELECT COUNT(*) as count FROM health_scores {where_clause}",
                params
            )
            stats['total_scores'] = cursor.fetchone()['count']

            # Average score
            cursor.execute(
                f"SELECT AVG(total_score) as avg_score FROM health_scores {where_clause}",
                params
            )
            row = cursor.fetchone()
            stats['average_score'] = row['avg_score'] if row['avg_score'] else 0.0

            # Scores by category
            cursor.execute(f"""
                SELECT health_category, COUNT(*) as count
                FROM health_scores {where_clause}
                GROUP BY health_category
            """, params)
            stats['by_category'] = {row['health_category']: row['count'] for row in cursor.fetchall()}

            # Latest score
            cursor.execute(f"""
                SELECT total_score, health_category, timestamp
                FROM health_scores {where_clause}
                ORDER BY timestamp DESC
                LIMIT 1
            """, params)
            row = cursor.fetchone()
            if row:
                stats['latest_score'] = {
                    'score': row['total_score'],
                    'category': row['health_category'],
                    'timestamp': row['timestamp']
                }

            conn.close()
            return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def delete_old_scores(self, days: int = 30) -> int:
        """
        Delete health scores older than specified days.

        Args:
            days: Delete scores older than this many days

        Returns:
            Number of deleted records
        """
        try:
            from datetime import timedelta

            cutoff_time = (datetime.now() - timedelta(days=days)).isoformat()

            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM health_scores WHERE timestamp < ?",
                (cutoff_time,)
            )

            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            logger.info(f"Deleted {deleted_count} old health score records")
            return deleted_count

        except Exception as e:
            logger.error(f"Error deleting old scores: {e}")
            return 0

    def __repr__(self) -> str:
        """String representation."""
        return f"HealthScoreManager(db={self.db_path})"
