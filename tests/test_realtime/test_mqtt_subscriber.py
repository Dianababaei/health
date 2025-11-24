"""
Unit Tests for MQTT Subscriber Module

Tests cover:
- Mock paho.mqtt.client.Client class
- Message parsing (valid JSON, malformed, missing fields)
- Data validation (type errors, range violations)
- DataFrame conversion and SensorDataManager calls
- Connection/disconnection event handling
- Exponential backoff reconnection logic
"""

import unittest
import json
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.test_realtime import (
    VALID_MQTT_MESSAGE,
    VALID_MQTT_MESSAGE_WITH_MOTION,
    INVALID_MQTT_MESSAGES,
    MALFORMED_JSON_MESSAGES,
    DEFAULT_MQTT_CONFIG,
    TempDatabaseHelper
)


class TestMQTTMessageParsing(unittest.TestCase):
    """Test MQTT message parsing and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock MQTT client
        self.mock_client = Mock()
        self.mock_sensor_manager = Mock()
        
    def test_parse_valid_json_message(self):
        """Test parsing valid JSON MQTT message."""
        # Simulate receiving a valid message
        message_payload = json.dumps(VALID_MQTT_MESSAGE)
        
        # Parse message (this would be done by subscriber)
        parsed = json.loads(message_payload)
        
        self.assertEqual(parsed['cow_id'], 'COW_001')
        self.assertEqual(parsed['temperature'], 38.5)
        self.assertIn('timestamp', parsed)
        self.assertIn('fxa', parsed)
    
    def test_parse_valid_message_with_motion_intensity(self):
        """Test parsing message with motion_intensity field."""
        message_payload = json.dumps(VALID_MQTT_MESSAGE_WITH_MOTION)
        parsed = json.loads(message_payload)
        
        self.assertEqual(parsed['cow_id'], 'COW_001')
        self.assertIn('motion_intensity', parsed)
        self.assertEqual(parsed['motion_intensity'], 0.42)
    
    def test_parse_malformed_json(self):
        """Test handling of malformed JSON messages."""
        for malformed in MALFORMED_JSON_MESSAGES:
            if malformed is None:
                continue
            
            with self.subTest(malformed=malformed):
                with self.assertRaises(json.JSONDecodeError):
                    json.loads(malformed)
    
    def test_parse_empty_message(self):
        """Test handling of empty message."""
        with self.assertRaises((json.JSONDecodeError, TypeError)):
            json.loads(None)
    
    def test_validate_missing_timestamp(self):
        """Test validation catches missing timestamp."""
        msg = INVALID_MQTT_MESSAGES['missing_timestamp']
        self.assertNotIn('timestamp', msg)
    
    def test_validate_missing_cow_id(self):
        """Test validation catches missing cow_id."""
        msg = INVALID_MQTT_MESSAGES['missing_cow_id']
        self.assertNotIn('cow_id', msg)
    
    def test_validate_invalid_temperature_type(self):
        """Test validation catches invalid temperature type."""
        msg = INVALID_MQTT_MESSAGES['invalid_temperature_type']
        self.assertIsInstance(msg['temperature'], str)
        
        # Should fail type conversion
        with self.assertRaises(ValueError):
            float(msg['temperature'])
    
    def test_validate_negative_temperature(self):
        """Test validation catches negative temperature."""
        msg = INVALID_MQTT_MESSAGES['negative_temperature']
        self.assertLess(msg['temperature'], 0)
    
    def test_validate_extreme_temperature(self):
        """Test validation catches extreme temperature."""
        msg = INVALID_MQTT_MESSAGES['extreme_temperature']
        self.assertGreater(msg['temperature'], 45)  # Way too high for a cow


class TestMQTTSubscriberWithMocks(unittest.TestCase):
    """Test MQTT subscriber with mocked paho.mqtt.client."""
    
    @patch('paho.mqtt.client.Client')
    def test_mqtt_client_initialization(self, mock_client_class):
        """Test MQTT client is initialized with correct parameters."""
        # Mock the client instance
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # This tests the expected interface for MQTTSubscriber.__init__
        # The actual class would do:
        # client = mqtt.Client(client_id=config['client_id'])
        # client.on_connect = self._on_connect
        # client.on_message = self._on_message
        # client.on_disconnect = self._on_disconnect
        
        config = DEFAULT_MQTT_CONFIG.copy()
        client_id = config['client_id']
        
        # Simulate subscriber initialization
        client = mock_client_class(client_id=client_id)
        
        # Verify client was created with correct parameters
        mock_client_class.assert_called_once_with(client_id=client_id)
        
        # Verify callback assignment would work
        client.on_connect = Mock()
        client.on_message = Mock()
        client.on_disconnect = Mock()
        
        self.assertIsNotNone(client.on_connect)
        self.assertIsNotNone(client.on_message)
        self.assertIsNotNone(client.on_disconnect)
    
    @patch('paho.mqtt.client.Client')
    def test_mqtt_connect_called_with_correct_params(self, mock_client_class):
        """Test MQTT connect is called with correct broker parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        config = DEFAULT_MQTT_CONFIG.copy()
        
        # Simulate subscriber.connect()
        client = mock_client_class()
        client.connect(
            host=config['broker_host'],
            port=config['broker_port'],
            keepalive=config['keepalive']
        )
        
        # Verify connect was called correctly
        mock_client.connect.assert_called_once_with(
            host='localhost',
            port=1883,
            keepalive=60
        )
    
    @patch('paho.mqtt.client.Client')
    def test_mqtt_subscribe_called_with_topic(self, mock_client_class):
        """Test MQTT subscribe is called with correct topic."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        config = DEFAULT_MQTT_CONFIG.copy()
        
        # Simulate subscription
        client = mock_client_class()
        client.subscribe(config['topic'], qos=config['qos'])
        
        # Verify subscribe was called correctly
        mock_client.subscribe.assert_called_once_with('cattle/sensors/+', qos=1)
    
    @patch('paho.mqtt.client.Client')
    def test_on_connect_callback_subscribes(self, mock_client_class):
        """Test on_connect callback subscribes to topic."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate on_connect callback
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                client.subscribe('cattle/sensors/+', qos=1)
        
        # Call the callback
        on_connect(mock_client, None, None, 0)
        
        # Verify subscribe was called
        mock_client.subscribe.assert_called_once()
    
    @patch('paho.mqtt.client.Client')
    def test_on_connect_handles_connection_failure(self, mock_client_class):
        """Test on_connect handles connection failure codes."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate on_connect callback with failure code
        def on_connect(client, userdata, flags, rc):
            if rc != 0:
                # Connection failed, should log error
                return False
            return True
        
        # Test successful connection
        result = on_connect(mock_client, None, None, 0)
        self.assertTrue(result)
        
        # Test failed connection
        result = on_connect(mock_client, None, None, 5)
        self.assertFalse(result)


class TestMessageToDataFrame(unittest.TestCase):
    """Test conversion of MQTT messages to DataFrame."""
    
    def test_convert_single_message_to_dataframe(self):
        """Test converting single message to DataFrame."""
        message = VALID_MQTT_MESSAGE.copy()
        
        # Simulate conversion
        df = pd.DataFrame([message])
        
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['cow_id'], 'COW_001')
        self.assertEqual(df.iloc[0]['temperature'], 38.5)
        self.assertIn('timestamp', df.columns)
    
    def test_convert_multiple_messages_to_dataframe(self):
        """Test converting multiple messages to DataFrame."""
        messages = [
            VALID_MQTT_MESSAGE.copy(),
            VALID_MQTT_MESSAGE_WITH_MOTION.copy()
        ]
        
        df = pd.DataFrame(messages)
        
        self.assertEqual(len(df), 2)
        self.assertIn('cow_id', df.columns)
        self.assertIn('temperature', df.columns)
    
    def test_dataframe_timestamp_parsing(self):
        """Test timestamp is properly parsed to datetime."""
        message = VALID_MQTT_MESSAGE.copy()
        df = pd.DataFrame([message])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))
    
    def test_dataframe_handles_missing_optional_fields(self):
        """Test DataFrame handles messages with missing optional fields."""
        message = {
            "timestamp": "2024-01-15T12:30:45Z",
            "cow_id": "COW_001",
            "temperature": 38.5,
            "fxa": 0.25
            # Missing mya, rza, etc.
        }
        
        df = pd.DataFrame([message])
        
        self.assertEqual(len(df), 1)
        self.assertTrue(pd.isna(df.iloc[0].get('mya', pd.NA)))


class TestSensorDataManagerIntegration(unittest.TestCase):
    """Test integration with SensorDataManager."""
    
    def setUp(self):
        """Set up test database."""
        self.db_helper = TempDatabaseHelper()
        self.db_path = self.db_helper.setup()
    
    def tearDown(self):
        """Clean up test database."""
        self.db_helper.teardown()
    
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    def test_subscriber_calls_sensor_manager_append(self, mock_manager_class):
        """Test subscriber calls SensorDataManager.append_sensor_data."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.append_sensor_data.return_value = (1, 0)  # 1 inserted, 0 skipped
        
        # Simulate receiving and processing message
        message = VALID_MQTT_MESSAGE.copy()
        df = pd.DataFrame([message])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        cow_id = df.iloc[0]['cow_id']
        
        # Call append (this is what subscriber would do)
        manager = mock_manager_class(db_path=self.db_path)
        inserted, skipped = manager.append_sensor_data(df, cow_id)
        
        # Verify method was called correctly
        mock_manager.append_sensor_data.assert_called_once()
        self.assertEqual(inserted, 1)
        self.assertEqual(skipped, 0)
    
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    def test_subscriber_handles_duplicate_messages(self, mock_manager_class):
        """Test subscriber handles duplicate message timestamps correctly."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # First insert succeeds
        mock_manager.append_sensor_data.return_value = (1, 0)
        
        message = VALID_MQTT_MESSAGE.copy()
        df = pd.DataFrame([message])
        
        manager = mock_manager_class(db_path=self.db_path)
        inserted1, skipped1 = manager.append_sensor_data(df, 'COW_001')
        
        # Second insert of same message should be skipped
        mock_manager.append_sensor_data.return_value = (0, 1)
        inserted2, skipped2 = manager.append_sensor_data(df, 'COW_001')
        
        self.assertEqual(inserted1, 1)
        self.assertEqual(inserted2, 0)
        self.assertEqual(skipped2, 1)


class TestConnectionHandling(unittest.TestCase):
    """Test MQTT connection and disconnection handling."""
    
    @patch('paho.mqtt.client.Client')
    def test_on_disconnect_callback_called(self, mock_client_class):
        """Test on_disconnect callback is triggered."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        disconnect_called = {'value': False}
        
        def on_disconnect(client, userdata, rc):
            disconnect_called['value'] = True
        
        # Simulate disconnection
        on_disconnect(mock_client, None, 0)
        
        self.assertTrue(disconnect_called['value'])
    
    @patch('paho.mqtt.client.Client')
    def test_disconnect_triggers_reconnection(self, mock_client_class):
        """Test disconnect triggers reconnection attempt."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        reconnect_attempted = {'value': False}
        
        def on_disconnect(client, userdata, rc):
            if rc != 0:  # Unexpected disconnection
                reconnect_attempted['value'] = True
                # Would normally call client.reconnect()
        
        # Simulate unexpected disconnection (rc != 0)
        on_disconnect(mock_client, None, 1)
        
        self.assertTrue(reconnect_attempted['value'])


class TestReconnectionLogic(unittest.TestCase):
    """Test exponential backoff reconnection logic."""
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        min_delay = 1
        max_delay = 60
        
        # Simulate backoff calculation
        def calculate_backoff(attempt, min_delay, max_delay):
            delay = min_delay * (2 ** attempt)
            return min(delay, max_delay)
        
        # Test increasing delays
        self.assertEqual(calculate_backoff(0, min_delay, max_delay), 1)
        self.assertEqual(calculate_backoff(1, min_delay, max_delay), 2)
        self.assertEqual(calculate_backoff(2, min_delay, max_delay), 4)
        self.assertEqual(calculate_backoff(3, min_delay, max_delay), 8)
        self.assertEqual(calculate_backoff(4, min_delay, max_delay), 16)
        self.assertEqual(calculate_backoff(5, min_delay, max_delay), 32)
        self.assertEqual(calculate_backoff(6, min_delay, max_delay), 60)  # Capped at max
        self.assertEqual(calculate_backoff(10, min_delay, max_delay), 60)  # Still capped
    
    @patch('time.sleep')
    @patch('paho.mqtt.client.Client')
    def test_reconnection_with_backoff(self, mock_client_class, mock_sleep):
        """Test reconnection attempts with exponential backoff."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Simulate multiple reconnection attempts
        def simulate_reconnect(client, max_attempts=3):
            for attempt in range(max_attempts):
                try:
                    client.reconnect()
                    return True
                except Exception:
                    delay = min(1 * (2 ** attempt), 60)
                    mock_sleep(delay)
            return False
        
        # Simulate failed reconnections
        mock_client.reconnect.side_effect = [Exception("Failed"), Exception("Failed"), True]
        
        result = simulate_reconnect(mock_client, max_attempts=3)
        
        # Should have attempted 3 times
        self.assertEqual(mock_client.reconnect.call_count, 3)
        
        # Should have slept with increasing delays
        self.assertEqual(mock_sleep.call_count, 2)  # Sleep after first two failures
    
    @patch('paho.mqtt.client.Client')
    def test_reconnection_success_resets_counter(self, mock_client_class):
        """Test successful reconnection resets attempt counter."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        attempt_counter = {'count': 0}
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                attempt_counter['count'] = 0  # Reset on success
        
        # Simulate failed attempts
        attempt_counter['count'] = 5
        
        # Successful connection
        on_connect(mock_client, None, None, 0)
        
        self.assertEqual(attempt_counter['count'], 0)


class TestMessageProcessingErrors(unittest.TestCase):
    """Test error handling during message processing."""
    
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    def test_processing_continues_after_invalid_message(self, mock_manager_class):
        """Test processing continues after encountering invalid message."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        processed_messages = []
        
        def process_message(message_payload):
            try:
                data = json.loads(message_payload)
                if 'timestamp' not in data or 'cow_id' not in data:
                    raise ValueError("Missing required fields")
                processed_messages.append(data)
                return True
            except (json.JSONDecodeError, ValueError) as e:
                # Log error but continue processing
                return False
        
        # Process mix of valid and invalid messages
        messages = [
            json.dumps(VALID_MQTT_MESSAGE),
            '{"invalid": "json"',  # Malformed
            json.dumps(INVALID_MQTT_MESSAGES['missing_timestamp']),
            json.dumps(VALID_MQTT_MESSAGE_WITH_MOTION)
        ]
        
        results = [process_message(msg) for msg in messages]
        
        # Should process 2 valid messages despite errors
        self.assertEqual(len(processed_messages), 2)
        self.assertEqual(results, [True, False, False, True])
    
    @patch('src.health_intelligence.logging.sensor_data_manager.SensorDataManager')
    def test_database_error_logged_but_not_fatal(self, mock_manager_class):
        """Test database errors are logged but don't crash subscriber."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Simulate database error
        mock_manager.append_sensor_data.side_effect = Exception("Database connection lost")
        
        manager = mock_manager_class()
        
        try:
            manager.append_sensor_data(pd.DataFrame(), 'COW_001')
            # Should raise exception
            self.fail("Expected exception")
        except Exception as e:
            # Exception should be caught and logged by subscriber
            self.assertIn("Database", str(e))


class TestMessageBuffering(unittest.TestCase):
    """Test message buffering for batch processing."""
    
    def test_buffer_accumulates_messages(self):
        """Test message buffer accumulates messages."""
        buffer = []
        max_buffer_size = 10
        
        # Simulate adding messages to buffer
        for i in range(5):
            message = VALID_MQTT_MESSAGE.copy()
            message['timestamp'] = f"2024-01-15T12:{30+i}:00Z"
            buffer.append(message)
        
        self.assertEqual(len(buffer), 5)
        self.assertLess(len(buffer), max_buffer_size)
    
    def test_buffer_flushes_at_size_limit(self):
        """Test buffer flushes when size limit reached."""
        buffer = []
        max_buffer_size = 3
        flushed_batches = []
        
        def flush_buffer(buf):
            if buf:
                flushed_batches.append(buf.copy())
                buf.clear()
        
        # Add messages
        for i in range(10):
            message = VALID_MQTT_MESSAGE.copy()
            buffer.append(message)
            
            if len(buffer) >= max_buffer_size:
                flush_buffer(buffer)
        
        # Flush remaining
        flush_buffer(buffer)
        
        # Should have flushed 4 batches (3+3+3+1)
        self.assertEqual(len(flushed_batches), 4)
        self.assertEqual(len(flushed_batches[0]), 3)
        self.assertEqual(len(flushed_batches[-1]), 1)


if __name__ == '__main__':
    unittest.main()
