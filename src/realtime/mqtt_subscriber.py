"""
MQTT Subscriber for Real-Time Sensor Data Ingestion

Connects to MQTT broker, receives sensor messages from cattle neck-mounted devices,
validates data, and stores readings to the database using SensorDataManager.

Features:
- JSON message parsing and schema validation
- Exponential backoff reconnection logic
- Integration with SensorDataManager for database writes
- Comprehensive error handling and logging
- Message statistics tracking

References:
- paho-mqtt documentation: https://eclipse.dev/paho/files/paho.mqtt.python/html/
- MQTT QoS levels for reliable message delivery
"""

import json
import logging
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import pandas as pd

try:
    import paho.mqtt.client as mqtt
except ImportError:
    raise ImportError(
        "paho-mqtt library is required. Install with: pip install paho-mqtt"
    )

from src.health_intelligence.logging.sensor_data_manager import SensorDataManager

logger = logging.getLogger(__name__)


class MQTTSubscriber:
    """
    MQTT subscriber for real-time sensor data ingestion.
    
    Connects to MQTT broker, receives sensor messages, validates data,
    and stores readings to the database.
    
    Attributes:
        config: Configuration dictionary with MQTT settings
        sensor_manager: SensorDataManager for database operations
        client: MQTT client instance
        connected: Connection status flag
        running: Service running status flag
        stats: Message statistics tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MQTT subscriber.
        
        Args:
            config: Configuration dictionary with structure:
                {
                    'mqtt': {
                        'broker_host': 'localhost',
                        'broker_port': 1883,
                        'topic_pattern': 'artemis/sensors/+',
                        'client_id': 'artemis_subscriber',
                        'keepalive': 60,
                        'qos': 1,
                        'username': None,  # optional
                        'password': None,  # optional
                        'reconnect': {
                            'initial_delay': 5,
                            'max_delay': 60,
                            'backoff_multiplier': 2
                        }
                    },
                    'database': {
                        'path': 'data/alert_state.db'
                    }
                }
        """
        self.config = config
        mqtt_config = config.get('mqtt', {})
        db_config = config.get('database', {})
        
        # MQTT settings
        self.broker_host = mqtt_config.get('broker_host', 'localhost')
        self.broker_port = mqtt_config.get('broker_port', 1883)
        self.topic_pattern = mqtt_config.get('topic_pattern', 'artemis/sensors/+')
        self.client_id = mqtt_config.get('client_id', 'artemis_subscriber')
        self.keepalive = mqtt_config.get('keepalive', 60)
        self.qos = mqtt_config.get('qos', 1)
        self.username = mqtt_config.get('username')
        self.password = mqtt_config.get('password')
        
        # Reconnection settings
        reconnect_config = mqtt_config.get('reconnect', {})
        self.initial_delay = reconnect_config.get('initial_delay', 5)
        self.max_delay = reconnect_config.get('max_delay', 60)
        self.backoff_multiplier = reconnect_config.get('backoff_multiplier', 2)
        self.current_delay = self.initial_delay
        
        # Initialize SensorDataManager
        db_path = db_config.get('path', 'data/alert_state.db')
        self.sensor_manager = SensorDataManager(db_path=db_path)
        
        # Initialize MQTT client
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        
        # Set credentials if provided
        if self.username and self.password:
            self.client.username_pw_set(self.username, self.password)
        
        # Status flags
        self.connected = False
        self.running = False
        
        # Message statistics
        self.stats = {
            'total_received': 0,
            'valid_messages': 0,
            'invalid_messages': 0,
            'stored_messages': 0,
            'duplicate_messages': 0,
            'storage_errors': 0,
            'last_stats_log': time.time(),
            'last_message_time': None
        }
        
        # Statistics lock for thread safety
        self.stats_lock = threading.Lock()
        
        logger.info(
            f"MQTTSubscriber initialized: broker={self.broker_host}:{self.broker_port}, "
            f"topic={self.topic_pattern}, db={db_path}"
        )
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker with error handling.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"Connecting to MQTT broker at {self.broker_host}:{self.broker_port}...")
            self.client.connect(self.broker_host, self.broker_port, self.keepalive)
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """
        Callback when connection is established.
        
        Args:
            client: MQTT client instance
            userdata: User data (unused)
            flags: Connection flags
            rc: Result code (0 = success)
        """
        if rc == 0:
            self.connected = True
            self.current_delay = self.initial_delay  # Reset backoff on successful connection
            logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
            
            # Subscribe to topic pattern
            try:
                client.subscribe(self.topic_pattern, qos=self.qos)
                logger.info(f"Subscribed to topic pattern: {self.topic_pattern} (QoS {self.qos})")
            except Exception as e:
                logger.error(f"Failed to subscribe to topic {self.topic_pattern}: {e}")
        else:
            self.connected = False
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier",
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorized"
            }
            error_msg = error_messages.get(rc, f"Unknown error code: {rc}")
            logger.error(f"Connection failed: {error_msg}")
    
    def _on_disconnect(self, client, userdata, rc):
        """
        Callback when disconnection occurs.
        
        Args:
            client: MQTT client instance
            userdata: User data (unused)
            rc: Result code (0 = clean disconnect, >0 = unexpected)
        """
        self.connected = False
        
        if rc == 0:
            logger.info("Disconnected from MQTT broker (clean disconnect)")
        else:
            logger.warning(f"Disconnected from MQTT broker (reason code: {rc})")
            
            # Implement exponential backoff reconnection
            if self.running:
                logger.info(f"Attempting reconnection in {self.current_delay} seconds...")
                time.sleep(self.current_delay)
                
                # Increase delay for next reconnection attempt
                self.current_delay = min(
                    self.current_delay * self.backoff_multiplier,
                    self.max_delay
                )
                
                # Attempt reconnection
                if not self.connect():
                    logger.error("Reconnection attempt failed")
    
    def _on_message(self, client, userdata, msg):
        """
        Callback when message is received.
        
        Args:
            client: MQTT client instance
            userdata: User data (unused)
            msg: MQTT message with topic and payload
        """
        with self.stats_lock:
            self.stats['total_received'] += 1
            self.stats['last_message_time'] = time.time()
        
        # Parse and validate message
        data = self.parse_message(msg.payload)
        
        if data is None:
            with self.stats_lock:
                self.stats['invalid_messages'] += 1
            return
        
        with self.stats_lock:
            self.stats['valid_messages'] += 1
        
        # Store sensor data
        success, is_duplicate = self.store_sensor_data(data)
        
        with self.stats_lock:
            if success:
                if is_duplicate:
                    self.stats['duplicate_messages'] += 1
                else:
                    self.stats['stored_messages'] += 1
            else:
                self.stats['storage_errors'] += 1
        
        # Log statistics periodically (every 60 seconds or 100 messages)
        self._log_stats_if_needed()
    
    def parse_message(self, payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse JSON message and validate schema.
        
        Args:
            payload: Raw message payload (JSON bytes)
        
        Returns:
            Dictionary with validated data, or None if invalid
        """
        # Parse JSON
        try:
            data = json.loads(payload.decode('utf-8'))
        except json.JSONDecodeError as e:
            payload_preview = payload[:100].decode('utf-8', errors='replace')
            logger.error(f"Malformed JSON: {e}. Payload preview: {payload_preview}")
            return None
        except Exception as e:
            payload_preview = payload[:100].decode('utf-8', errors='replace')
            logger.error(f"Error decoding payload: {e}. Payload preview: {payload_preview}")
            return None
        
        # Validate required fields
        required_fields = ['cow_id', 'timestamp', 'temperature', 'fxa', 'mya', 'rza']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            cow_id = data.get('cow_id', 'unknown')
            logger.error(
                f"Missing required fields for cow_id={cow_id}: {', '.join(missing_fields)}"
            )
            return None
        
        # Validate cow_id (must be string)
        if not isinstance(data['cow_id'], str):
            logger.error(f"Invalid cow_id type: expected string, got {type(data['cow_id'])}")
            return None
        
        # Validate timestamp (must be parseable datetime)
        try:
            # Try to parse timestamp to validate format
            timestamp_str = data['timestamp']
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            logger.error(
                f"Invalid timestamp for cow_id={data['cow_id']}: {data.get('timestamp')}. Error: {e}"
            )
            return None
        
        # Validate temperature range (35.0-42.0°C for cattle)
        try:
            temp = float(data['temperature'])
            if not (35.0 <= temp <= 42.0):
                logger.error(
                    f"Temperature out of range for cow_id={data['cow_id']}: {temp}°C "
                    f"(valid range: 35.0-42.0°C)"
                )
                return None
            data['temperature'] = temp
        except (ValueError, TypeError) as e:
            logger.error(
                f"Invalid temperature value for cow_id={data['cow_id']}: {data.get('temperature')}. "
                f"Error: {e}"
            )
            return None
        
        # Validate accelerometer ranges (-2.0 to +2.0 g for all axes)
        accel_fields = ['fxa', 'mya', 'rza']
        for field in accel_fields:
            try:
                value = float(data[field])
                if not (-2.0 <= value <= 2.0):
                    logger.error(
                        f"Accelerometer {field} out of range for cow_id={data['cow_id']}: {value}g "
                        f"(valid range: -2.0 to +2.0g)"
                    )
                    return None
                data[field] = value
            except (ValueError, TypeError) as e:
                logger.error(
                    f"Invalid {field} value for cow_id={data['cow_id']}: {data.get(field)}. "
                    f"Error: {e}"
                )
                return None
        
        # Optional fields: gyroscope data (sxg, lyg, dzg)
        optional_fields = ['sxg', 'lyg', 'dzg', 'state', 'motion_intensity']
        for field in optional_fields:
            if field in data:
                try:
                    # Convert numeric fields to float
                    if field in ['sxg', 'lyg', 'dzg', 'motion_intensity']:
                        data[field] = float(data[field])
                except (ValueError, TypeError):
                    # Remove invalid optional fields
                    logger.warning(
                        f"Invalid optional field {field} for cow_id={data['cow_id']}, removing"
                    )
                    del data[field]
        
        return data
    
    def store_sensor_data(self, data: Dict[str, Any]) -> Tuple[bool, bool]:
        """
        Convert data to DataFrame and store using SensorDataManager.
        
        Args:
            data: Validated sensor data dictionary
        
        Returns:
            Tuple of (success: bool, is_duplicate: bool)
        """
        try:
            # Convert single record to DataFrame
            df = pd.DataFrame([data])
            
            # Extract cow_id (required for append_sensor_data)
            cow_id = data['cow_id']
            
            # Call SensorDataManager to append data
            inserted, skipped = self.sensor_manager.append_sensor_data(df, cow_id)
            
            if inserted > 0:
                return (True, False)  # Successfully stored, not a duplicate
            elif skipped > 0:
                return (True, True)   # Skipped (duplicate timestamp)
            else:
                logger.warning(f"No rows inserted or skipped for cow_id={cow_id}")
                return (False, False)
        
        except Exception as e:
            cow_id = data.get('cow_id', 'unknown')
            logger.error(f"Database insertion failure for cow_id={cow_id}: {e}")
            return (False, False)
    
    def _log_stats_if_needed(self):
        """Log message statistics periodically (every 60s or 100 messages)."""
        with self.stats_lock:
            current_time = time.time()
            time_since_last_log = current_time - self.stats['last_stats_log']
            messages_since_start = self.stats['total_received']
            
            # Log every 60 seconds or every 100 messages (whichever comes first)
            should_log = (
                time_since_last_log >= 60 or
                (messages_since_start > 0 and messages_since_start % 100 == 0)
            )
            
            if should_log:
                total = self.stats['total_received']
                valid = self.stats['valid_messages']
                invalid = self.stats['invalid_messages']
                stored = self.stats['stored_messages']
                duplicates = self.stats['duplicate_messages']
                errors = self.stats['storage_errors']
                
                logger.info(
                    f"Message stats: total={total}, valid={valid}, invalid={invalid}, "
                    f"stored={stored}, duplicates={duplicates}, storage_errors={errors}"
                )
                logger.info(
                    f"Database stats: Stored {stored}/{total} messages "
                    f"({duplicates} duplicates)"
                )
                
                self.stats['last_stats_log'] = current_time
    
    def start(self):
        """Start MQTT network loop (threaded)."""
        if self.running:
            logger.warning("MQTT subscriber already running")
            return
        
        self.running = True
        
        # Connect to broker
        if not self.connect():
            logger.error("Failed to start MQTT subscriber: connection failed")
            self.running = False
            return
        
        # Start network loop in separate thread
        self.client.loop_start()
        logger.info("MQTT subscriber started (threaded loop)")
    
    def stop(self):
        """Graceful shutdown and disconnect from broker."""
        if not self.running:
            logger.warning("MQTT subscriber not running")
            return
        
        logger.info("Stopping MQTT subscriber...")
        self.running = False
        
        # Log final statistics
        with self.stats_lock:
            total = self.stats['total_received']
            valid = self.stats['valid_messages']
            invalid = self.stats['invalid_messages']
            stored = self.stats['stored_messages']
            duplicates = self.stats['duplicate_messages']
            errors = self.stats['storage_errors']
            
            logger.info(
                f"Final message stats: total={total}, valid={valid}, invalid={invalid}, "
                f"stored={stored}, duplicates={duplicates}, storage_errors={errors}"
            )
        
        # Stop network loop
        self.client.loop_stop()
        
        # Disconnect from broker
        self.client.disconnect()
        
        logger.info("MQTT subscriber stopped")
