#!/usr/bin/env python3
"""
Test MQTT Publisher for Sensor Data

Publishes test sensor messages to MQTT broker to test the subscriber.
Simulates realistic cattle sensor data with proper ranges.

Prerequisites:
1. Install paho-mqtt: pip install paho-mqtt
2. Run Mosquitto broker locally: mosquitto -v -p 1883

Usage:
    python examples/mqtt_test_publisher.py
"""

import json
import time
import random
import logging
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("Error: paho-mqtt library is required. Install with: pip install paho-mqtt")
    exit(1)


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def generate_sensor_data(cow_id: str) -> dict:
    """
    Generate realistic sensor data for a cow.
    
    Args:
        cow_id: Cow identifier
    
    Returns:
        Dictionary with sensor data
    """
    # Normal cattle temperature: 38.0-39.0°C
    base_temp = 38.5
    temp_variation = random.uniform(-0.5, 0.5)
    
    # Accelerometer data (g-force)
    # Normal range: -1.0 to 1.0 g for most movements
    fxa = random.uniform(-0.5, 0.5)  # Forward/backward
    mya = random.uniform(-0.3, 0.3)  # Lateral
    rza = random.uniform(-1.0, -0.5)  # Vertical (mostly gravity)
    
    # Optional: Gyroscope data (degrees/second)
    sxg = random.uniform(-10, 10)    # Roll
    lyg = random.uniform(-5, 5)      # Pitch
    dzg = random.uniform(-5, 5)      # Yaw
    
    return {
        "cow_id": cow_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "temperature": round(base_temp + temp_variation, 2),
        "fxa": round(fxa, 3),
        "mya": round(mya, 3),
        "rza": round(rza, 3),
        "sxg": round(sxg, 2),
        "lyg": round(lyg, 2),
        "dzg": round(dzg, 2)
    }


def generate_invalid_data(cow_id: str, error_type: str) -> dict:
    """
    Generate invalid sensor data for testing error handling.
    
    Args:
        cow_id: Cow identifier
        error_type: Type of error to generate
    
    Returns:
        Dictionary with invalid sensor data
    """
    data = generate_sensor_data(cow_id)
    
    if error_type == "missing_field":
        # Remove a required field
        del data['temperature']
    elif error_type == "invalid_temp":
        # Temperature out of range
        data['temperature'] = 50.0
    elif error_type == "invalid_accel":
        # Accelerometer out of range
        data['fxa'] = 5.0
    elif error_type == "invalid_timestamp":
        # Invalid timestamp format
        data['timestamp'] = "not-a-timestamp"
    elif error_type == "malformed_json":
        # Return string instead of dict (will be malformed when sent)
        return "not a json object"
    
    return data


def publish_messages(
    broker_host: str = "localhost",
    broker_port: int = 1883,
    num_messages: int = 10,
    interval: float = 1.0,
    cow_ids: list = None
):
    """
    Publish test sensor messages to MQTT broker.
    
    Args:
        broker_host: MQTT broker hostname
        broker_port: MQTT broker port
        num_messages: Number of messages to publish
        interval: Interval between messages (seconds)
        cow_ids: List of cow IDs to use (default: ["COW001", "COW002"])
    """
    logger = logging.getLogger(__name__)
    
    if cow_ids is None:
        cow_ids = ["COW001", "COW002"]
    
    # Create MQTT client
    client = mqtt.Client(client_id="test_publisher")
    
    try:
        # Connect to broker
        logger.info(f"Connecting to MQTT broker at {broker_host}:{broker_port}...")
        client.connect(broker_host, broker_port, 60)
        client.loop_start()
        
        logger.info(f"Publishing {num_messages} messages (interval: {interval}s)...")
        
        for i in range(num_messages):
            # Select random cow
            cow_id = random.choice(cow_ids)
            
            # Generate sensor data
            # 90% valid messages, 10% invalid for testing
            if random.random() < 0.9:
                data = generate_sensor_data(cow_id)
                message_type = "valid"
            else:
                error_types = ["missing_field", "invalid_temp", "invalid_accel", "invalid_timestamp"]
                error_type = random.choice(error_types)
                data = generate_invalid_data(cow_id, error_type)
                message_type = f"invalid ({error_type})"
            
            # Publish to topic
            topic = f"artemis/sensors/{cow_id}"
            payload = json.dumps(data)
            
            result = client.publish(topic, payload, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"[{i+1}/{num_messages}] Published to {topic} ({message_type})")
            else:
                logger.error(f"Failed to publish message {i+1}")
            
            # Wait before next message
            if i < num_messages - 1:
                time.sleep(interval)
        
        logger.info("Finished publishing messages")
        
        # Wait a bit for messages to be sent
        time.sleep(1)
        
        # Disconnect
        client.loop_stop()
        client.disconnect()
        logger.info("Disconnected from broker")
    
    except Exception as e:
        logger.error(f"Error publishing messages: {e}", exc_info=True)
        client.loop_stop()
        client.disconnect()


def main():
    """Main function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== MQTT Test Publisher ===")
    logger.info("This script publishes test sensor messages to the MQTT broker")
    logger.info("")
    
    # Configuration
    broker_host = "localhost"
    broker_port = 1883
    num_messages = 20
    interval = 0.5  # seconds between messages
    cow_ids = ["COW001", "COW002", "COW003"]
    
    logger.info(f"Configuration:")
    logger.info(f"  Broker: {broker_host}:{broker_port}")
    logger.info(f"  Messages: {num_messages}")
    logger.info(f"  Interval: {interval}s")
    logger.info(f"  Cow IDs: {', '.join(cow_ids)}")
    logger.info("")
    
    try:
        publish_messages(
            broker_host=broker_host,
            broker_port=broker_port,
            num_messages=num_messages,
            interval=interval,
            cow_ids=cow_ids
        )
        logger.info("Test publishing completed successfully")
    except KeyboardInterrupt:
        logger.info("\nPublishing interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
