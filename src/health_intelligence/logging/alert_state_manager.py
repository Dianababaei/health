"""
Alert State Management Module

Provides persistent alert state tracking with SQLite database,
state transition management, and query capabilities.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


logger = logging.getLogger(__name__)


class AlertStatus(Enum):
    """Alert lifecycle states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class AlertStateManager:
    """
    Manages alert state with persistent SQLite storage.
    
    Features:
    - Alert lifecycle state tracking
    - State transition validation
    - State history with timestamps
    - Query and filtering capabilities
    - Analytics support
    """
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        AlertStatus.ACTIVE: [
            AlertStatus.ACKNOWLEDGED,
            AlertStatus.RESOLVED,
            AlertStatus.FALSE_POSITIVE
        ],
        AlertStatus.ACKNOWLEDGED: [
            AlertStatus.RESOLVED,
            AlertStatus.FALSE_POSITIVE
        ],
        AlertStatus.RESOLVED: [],  # Terminal state
        AlertStatus.FALSE_POSITIVE: []  # Terminal state
    }
    
    def __init__(self, db_path: str = "data/alert_state.db"):
        """
        Initialize alert state manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        
        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"AlertStateManager initialized: db={self.db_path}")
    
    def _init_database(self):
        """Initialize database schema."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    cow_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolution_notes TEXT,
                    sensor_values TEXT,
                    detection_details TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            # Create state_history table for tracking all state changes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    old_status TEXT,
                    new_status TEXT NOT NULL,
                    changed_at TEXT NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts (alert_id)
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_cow_id 
                ON alerts (cow_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_status 
                ON alerts (status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_created_at 
                ON alerts (created_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_state_history_alert_id 
                ON state_history (alert_id)
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable row access by column name
        return conn
    
    def create_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Create a new alert in the database.
        
        Args:
            alert_data: Alert data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Prepare alert data
            alert_id = alert_data.get('alert_id')
            if not alert_id:
                logger.error("Alert ID is required")
                return False
            
            now = datetime.now().isoformat()
            
            # Convert complex fields to JSON strings
            import json
            sensor_values = json.dumps(alert_data.get('sensor_values', {}))
            detection_details = json.dumps(alert_data.get('detection_details', {}))
            
            cursor.execute("""
                INSERT INTO alerts (
                    alert_id, cow_id, alert_type, severity, confidence,
                    status, created_at, updated_at, resolution_notes,
                    sensor_values, detection_details, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id,
                alert_data.get('cow_id', 'unknown'),
                alert_data.get('alert_type', 'unknown'),
                alert_data.get('severity', 'info'),
                alert_data.get('confidence', 0.0),
                alert_data.get('status', AlertStatus.ACTIVE.value),
                now,
                now,
                alert_data.get('resolution_notes', ''),
                sensor_values,
                detection_details,
                alert_data.get('timestamp', now)
            ))
            
            # Record initial state in history
            cursor.execute("""
                INSERT INTO state_history (
                    alert_id, old_status, new_status, changed_at, notes
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                alert_id,
                None,
                alert_data.get('status', AlertStatus.ACTIVE.value),
                now,
                'Alert created'
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Alert created: {alert_id}")
            return True
            
        except sqlite3.IntegrityError:
            logger.warning(f"Alert already exists: {alert_data.get('alert_id')}")
            return False
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            return False
    
    def update_status(
        self,
        alert_id: str,
        new_status: str,
        notes: Optional[str] = None
    ) -> bool:
        """
        Update alert status with validation.
        
        Args:
            alert_id: Alert ID
            new_status: New status value
            notes: Optional notes about the status change
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate status value
            try:
                new_status_enum = AlertStatus(new_status)
            except ValueError:
                logger.error(f"Invalid status value: {new_status}")
                return False
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get current status
            cursor.execute(
                "SELECT status FROM alerts WHERE alert_id = ?",
                (alert_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                logger.error(f"Alert not found: {alert_id}")
                conn.close()
                return False
            
            old_status = row['status']
            
            # Validate state transition
            if not self._is_valid_transition(old_status, new_status):
                logger.error(
                    f"Invalid state transition: {old_status} -> {new_status}"
                )
                conn.close()
                return False
            
            # Update alert status
            now = datetime.now().isoformat()
            cursor.execute("""
                UPDATE alerts 
                SET status = ?, updated_at = ?, resolution_notes = ?
                WHERE alert_id = ?
            """, (new_status, now, notes or '', alert_id))
            
            # Record state change in history
            cursor.execute("""
                INSERT INTO state_history (
                    alert_id, old_status, new_status, changed_at, notes
                ) VALUES (?, ?, ?, ?, ?)
            """, (alert_id, old_status, new_status, now, notes or ''))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Alert status updated: {alert_id} ({old_status} -> {new_status})")
            return True
            
        except Exception as e:
            logger.error(f"Error updating alert status: {e}")
            return False
    
    def _is_valid_transition(self, old_status: str, new_status: str) -> bool:
        """
        Check if state transition is valid.
        
        Args:
            old_status: Current status
            new_status: Desired new status
            
        Returns:
            True if transition is valid, False otherwise
        """
        try:
            old_enum = AlertStatus(old_status)
            new_enum = AlertStatus(new_status)
            
            # Allow staying in same state
            if old_enum == new_enum:
                return True
            
            return new_enum in self.VALID_TRANSITIONS.get(old_enum, [])
            
        except ValueError:
            return False
    
    def acknowledge_alert(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert ID
            notes: Optional acknowledgment notes
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(
            alert_id,
            AlertStatus.ACKNOWLEDGED.value,
            notes or "Alert acknowledged"
        )
    
    def resolve_alert(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert ID
            notes: Optional resolution notes
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(
            alert_id,
            AlertStatus.RESOLVED.value,
            notes or "Alert resolved"
        )
    
    def mark_false_positive(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """
        Mark an alert as false positive.
        
        Args:
            alert_id: Alert ID
            notes: Optional notes
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_status(
            alert_id,
            AlertStatus.FALSE_POSITIVE.value,
            notes or "Marked as false positive"
        )
    
    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        """
        Get alert by ID.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            Alert dictionary or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (alert_id,)
            )
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_dict(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting alert: {e}")
            return None
    
    def query_alerts(
        self,
        cow_id: Optional[str] = None,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        sort_by: str = "created_at",
        sort_order: str = "DESC"
    ) -> List[Dict[str, Any]]:
        """
        Query alerts with filters.
        
        Args:
            cow_id: Filter by cow ID
            alert_type: Filter by alert type
            severity: Filter by severity
            status: Filter by status
            start_date: Filter by start date (ISO8601)
            end_date: Filter by end date (ISO8601)
            limit: Maximum number of results
            sort_by: Column to sort by
            sort_order: Sort order (ASC or DESC)
            
        Returns:
            List of alert dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if cow_id:
                query += " AND cow_id = ?"
                params.append(cow_id)
            
            if alert_type:
                query += " AND alert_type = ?"
                params.append(alert_type)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND created_at >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND created_at <= ?"
                params.append(end_date)
            
            # Add sorting
            query += f" ORDER BY {sort_by} {sort_order}"
            
            # Add limit
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error querying alerts: {e}")
            return []
    
    def get_state_history(self, alert_id: str) -> List[Dict[str, Any]]:
        """
        Get state history for an alert.
        
        Args:
            alert_id: Alert ID
            
        Returns:
            List of state change dictionaries
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM state_history 
                WHERE alert_id = ?
                ORDER BY changed_at ASC
            """, (alert_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting state history: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            # Total alerts
            cursor.execute("SELECT COUNT(*) as count FROM alerts")
            stats['total_alerts'] = cursor.fetchone()['count']
            
            # Alerts by status
            cursor.execute("""
                SELECT status, COUNT(*) as count 
                FROM alerts 
                GROUP BY status
            """)
            stats['by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
            
            # Alerts by severity
            cursor.execute("""
                SELECT severity, COUNT(*) as count 
                FROM alerts 
                GROUP BY severity
            """)
            stats['by_severity'] = {row['severity']: row['count'] for row in cursor.fetchall()}
            
            # Alerts by type
            cursor.execute("""
                SELECT alert_type, COUNT(*) as count 
                FROM alerts 
                GROUP BY alert_type
                ORDER BY count DESC
                LIMIT 10
            """)
            stats['by_type'] = {row['alert_type']: row['count'] for row in cursor.fetchall()}
            
            # Average resolution time (for resolved alerts)
            cursor.execute("""
                SELECT AVG(
                    (julianday(updated_at) - julianday(created_at)) * 24 * 60
                ) as avg_minutes
                FROM alerts
                WHERE status = 'resolved'
            """)
            row = cursor.fetchone()
            stats['avg_resolution_time_minutes'] = row['avg_minutes'] if row['avg_minutes'] else 0
            
            # False positive rate
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN status = 'false_positive' THEN 1 END) * 100.0 / COUNT(*) as rate
                FROM alerts
                WHERE status IN ('resolved', 'false_positive')
            """)
            row = cursor.fetchone()
            stats['false_positive_rate'] = row['rate'] if row['rate'] else 0
            
            conn.close()
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """
        Convert database row to dictionary.
        
        Args:
            row: SQLite row object
            
        Returns:
            Dictionary with row data
        """
        import json
        
        data = dict(row)
        
        # Parse JSON fields
        if 'sensor_values' in data and data['sensor_values']:
            try:
                data['sensor_values'] = json.loads(data['sensor_values'])
            except:
                pass
        
        if 'detection_details' in data and data['detection_details']:
            try:
                data['detection_details'] = json.loads(data['detection_details'])
            except:
                pass
        
        return data
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AlertStateManager(db={self.db_path})"
