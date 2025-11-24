"""
Shared Test Utilities and Fixtures for Real-time Service Tests

Provides:
- Sample MQTT message fixtures (valid and invalid)
- Mock configuration objects
- Temporary database setup/teardown helpers
- Common assertion utilities
"""

import sqlite3
import tempfile
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


# =============================================================================
# Sample MQTT Message Fixtures
# =============================================================================

VALID_MQTT_MESSAGE = {
    "timestamp": "2024-01-15T12:30:45Z",
    "cow_id": "COW_001",
    "temperature": 38.5,
    "fxa": 0.25,
    "mya": 0.15,
    "rza": 0.80,
    "sxg": 0.10,
    "lyg": 0.05,
    "dzg": 0.02,
    "state": "standing"
}

VALID_MQTT_MESSAGE_WITH_MOTION = {
    "timestamp": "2024-01-15T12:31:00Z",
    "cow_id": "COW_001",
    "temperature": 38.6,
    "fxa": 0.30,
    "mya": 0.20,
    "rza": 0.85,
    "motion_intensity": 0.42
}

INVALID_MQTT_MESSAGES = {
    "missing_timestamp": {
        "cow_id": "COW_001",
        "temperature": 38.5,
        "fxa": 0.25
    },
    "missing_cow_id": {
        "timestamp": "2024-01-15T12:30:45Z",
        "temperature": 38.5,
        "fxa": 0.25
    },
    "invalid_temperature_type": {
        "timestamp": "2024-01-15T12:30:45Z",
        "cow_id": "COW_001",
        "temperature": "not_a_number",
        "fxa": 0.25
    },
    "invalid_timestamp_format": {
        "timestamp": "not-a-valid-timestamp",
        "cow_id": "COW_001",
        "temperature": 38.5,
        "fxa": 0.25
    },
    "negative_temperature": {
        "timestamp": "2024-01-15T12:30:45Z",
        "cow_id": "COW_001",
        "temperature": -5.0,
        "fxa": 0.25
    },
    "extreme_temperature": {
        "timestamp": "2024-01-15T12:30:45Z",
        "cow_id": "COW_001",
        "temperature": 50.0,
        "fxa": 0.25
    },
    "missing_required_fields": {
        "timestamp": "2024-01-15T12:30:45Z",
        "cow_id": "COW_001"
    }
}

MALFORMED_JSON_MESSAGES = [
    '{"timestamp": "2024-01-15T12:30:45Z", "cow_id": "COW_001", "temperature": 38.5',  # Missing closing brace
    '{"timestamp": "2024-01-15T12:30:45Z", cow_id: "COW_001"}',  # Missing quotes
    'not json at all',
    '',
    None
]


# =============================================================================
# Sample Sensor Data
# =============================================================================

def create_sample_sensor_data(
    cow_id: str = "COW_001",
    num_records: int = 10,
    start_time: Optional[datetime] = None,
    interval_seconds: int = 60,
    temperature: float = 38.5,
    temp_variation: float = 0.2
) -> pd.DataFrame:
    """
    Create sample sensor data DataFrame for testing.
    
    Args:
        cow_id: Cow identifier
        num_records: Number of records to generate
        start_time: Start timestamp (defaults to now)
        interval_seconds: Seconds between records
        temperature: Base temperature
        temp_variation: Random variation in temperature
    
    Returns:
        DataFrame with sensor data
    """
    if start_time is None:
        start_time = datetime.now()
    
    timestamps = [start_time + timedelta(seconds=i * interval_seconds) for i in range(num_records)]
    
    data = {
        'timestamp': timestamps,
        'cow_id': [cow_id] * num_records,
        'temperature': [temperature + (i % 3 - 1) * temp_variation for i in range(num_records)],
        'fxa': [0.25 + (i % 5) * 0.05 for i in range(num_records)],
        'mya': [0.15 + (i % 4) * 0.03 for i in range(num_records)],
        'rza': [0.80 + (i % 3) * 0.05 for i in range(num_records)],
        'state': ['standing' if i % 2 == 0 else 'lying' for i in range(num_records)]
    }
    
    return pd.DataFrame(data)


def create_fever_sensor_data(
    cow_id: str = "COW_001",
    num_records: int = 5
) -> pd.DataFrame:
    """Create sensor data that should trigger fever alert."""
    start_time = datetime.now()
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_records)]
    
    data = {
        'timestamp': timestamps,
        'cow_id': [cow_id] * num_records,
        'temperature': [39.7, 39.8, 39.7, 39.9, 39.8],  # High temp
        'fxa': [0.02, 0.01, 0.02, 0.01, 0.02],  # Low motion
        'mya': [0.01, 0.01, 0.01, 0.01, 0.01],
        'rza': [0.01, 0.01, 0.01, 0.01, 0.01],
        'state': ['standing'] * num_records
    }
    
    return pd.DataFrame(data)


def create_estrus_sensor_data(
    cow_id: str = "COW_001",
    num_records: int = 10
) -> pd.DataFrame:
    """Create sensor data that should trigger estrus detection."""
    start_time = datetime.now() - timedelta(hours=12)
    timestamps = [start_time + timedelta(hours=i) for i in range(num_records)]
    
    data = {
        'timestamp': timestamps,
        'cow_id': [cow_id] * num_records,
        'temperature': [38.9, 39.0, 39.1, 39.0, 38.9, 38.8, 38.7, 38.6, 38.5, 38.5],  # Elevated temp
        'fxa': [0.5, 0.6, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.25, 0.25],  # High activity
        'mya': [0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.15, 0.15],
        'rza': [0.8, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5, 0.5],
        'state': ['walking', 'walking', 'walking', 'standing', 'standing', 'standing', 'lying', 'lying', 'lying', 'lying']
    }
    
    return pd.DataFrame(data)


# =============================================================================
# Mock Configuration Objects
# =============================================================================

DEFAULT_MQTT_CONFIG = {
    "broker_host": "localhost",
    "broker_port": 1883,
    "topic": "cattle/sensors/+",
    "client_id": "test_subscriber",
    "qos": 1,
    "keepalive": 60,
    "reconnect_delay_min": 1,
    "reconnect_delay_max": 60
}

DEFAULT_PIPELINE_CONFIG = {
    "detection_interval_minutes": 5,
    "rolling_window_hours": 24,
    "min_data_points": 10,
    "enable_immediate_alerts": True,
    "enable_estrus_detection": True,
    "enable_health_scoring": True
}

DEFAULT_SCHEDULER_CONFIG = {
    "immediate_alert_interval_seconds": 120,  # 2 minutes
    "estrus_detection_interval_minutes": 60,  # 1 hour
    "health_score_interval_minutes": 15,  # 15 minutes
    "max_workers": 3,
    "coalesce": True,
    "misfire_grace_time": 30
}


# =============================================================================
# Temporary Database Helpers
# =============================================================================

class TempDatabaseHelper:
    """Helper for creating and managing temporary test databases."""
    
    def __init__(self):
        self.temp_dir = None
        self.db_path = None
        self.conn = None
    
    def setup(self) -> str:
        """
        Create temporary database with schema.
        
        Returns:
            Path to temporary database
        """
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_alert_state.db")
        
        # Create database with schema
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create tables (matching production schema)
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
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                cow_id TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL,
                sensor_values TEXT,
                details TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS health_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cow_id TEXT NOT NULL,
                score REAL NOT NULL,
                temperature_component REAL,
                activity_component REAL,
                behavioral_component REAL,
                alert_component REAL,
                confidence REAL,
                created_at TEXT NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cows (
                cow_id TEXT PRIMARY KEY,
                name TEXT,
                breed TEXT,
                birth_date TEXT,
                status TEXT,
                last_updated TEXT
            )
        """)
        
        self.conn.commit()
        
        return self.db_path
    
    def insert_test_cows(self, cow_ids: List[str]):
        """Insert test cow records."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        for cow_id in cow_ids:
            cursor.execute("""
                INSERT OR IGNORE INTO cows (cow_id, name, breed, status, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (cow_id, f"Test {cow_id}", "Holstein", "active", now))
        
        self.conn.commit()
    
    def insert_sensor_data(self, df: pd.DataFrame):
        """Insert sensor data from DataFrame."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT OR IGNORE INTO sensor_data (
                    timestamp, cow_id, temperature, fxa, mya, rza, state, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                row['cow_id'],
                row.get('temperature'),
                row.get('fxa'),
                row.get('mya'),
                row.get('rza'),
                row.get('state'),
                now
            ))
        
        self.conn.commit()
    
    def get_alert_count(self, cow_id: Optional[str] = None) -> int:
        """Get count of alerts in database."""
        cursor = self.conn.cursor()
        if cow_id:
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE cow_id = ?", (cow_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM alerts")
        return cursor.fetchone()[0]
    
    def get_health_score_count(self, cow_id: Optional[str] = None) -> int:
        """Get count of health scores in database."""
        cursor = self.conn.cursor()
        if cow_id:
            cursor.execute("SELECT COUNT(*) FROM health_scores WHERE cow_id = ?", (cow_id,))
        else:
            cursor.execute("SELECT COUNT(*) FROM health_scores")
        return cursor.fetchone()[0]
    
    def teardown(self):
        """Clean up temporary database."""
        if self.conn:
            self.conn.close()
        
        if self.db_path and os.path.exists(self.db_path):
            os.unlink(self.db_path)
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)


# =============================================================================
# Common Assertion Utilities
# =============================================================================

def assert_valid_alert(alert: Dict[str, Any]):
    """Assert that alert dictionary has required fields."""
    required_fields = ['alert_id', 'timestamp', 'cow_id', 'alert_type', 'severity', 'status']
    for field in required_fields:
        assert field in alert, f"Alert missing required field: {field}"
    
    assert alert['severity'] in ['warning', 'critical'], f"Invalid severity: {alert['severity']}"
    assert alert['status'] in ['active', 'resolved'], f"Invalid status: {alert['status']}"


def assert_valid_health_score(score: Dict[str, Any]):
    """Assert that health score dictionary has required fields."""
    required_fields = ['timestamp', 'cow_id', 'total_score']
    for field in required_fields:
        assert field in score, f"Health score missing required field: {field}"
    
    assert 0 <= score['total_score'] <= 100, f"Invalid score: {score['total_score']}"


def assert_mqtt_message_format(message: Dict[str, Any]):
    """Assert that MQTT message has expected format."""
    required_fields = ['timestamp', 'cow_id']
    for field in required_fields:
        assert field in message, f"Message missing required field: {field}"
    
    # At least one sensor field should be present
    sensor_fields = ['temperature', 'fxa', 'mya', 'rza']
    has_sensor_data = any(field in message for field in sensor_fields)
    assert has_sensor_data, "Message has no sensor data fields"
