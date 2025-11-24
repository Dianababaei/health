#!/usr/bin/env python3
"""
Example: Using MQTT Subscriber for Real-Time Sensor Data Ingestion

This example demonstrates how to use the MQTTSubscriber class to connect
to an MQTT broker, receive sensor messages, validate data, and store readings
to the database.

Prerequisites:
1. Install paho-mqtt: pip install paho-mqtt
2. Run Mosquitto broker locally: mosquitto -v -p 1883
3. Ensure config/realtime_config.yaml is properly configured

Usage:
    python examples/mqtt_subscriber_example.py
"""

import sys
import time
import logging
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.realtime.mqtt_subscriber import MQTTSubscriber


def load_config(config_path: str = "config/realtime_config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging():
    """Configure logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function to run the MQTT subscriber example."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting MQTT Subscriber Example")
    
    try:
        # Load configuration
        logger.info("Loading configuration from config/realtime_config.yaml")
        config = load_config()
        
        # Create MQTTSubscriber instance
        logger.info("Initializing MQTT subscriber...")
        subscriber = MQTTSubscriber(config)
        
        # Start the subscriber
        logger.info("Starting MQTT subscriber...")
        subscriber.start()
        
        # Run for a while (or until interrupted)
        logger.info("Subscriber is running. Press Ctrl+C to stop.")
        logger.info(f"Listening on topic: {subscriber.topic_pattern}")
        logger.info(f"Broker: {subscriber.broker_host}:{subscriber.broker_port}")
        
        try:
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nReceived interrupt signal, shutting down...")
        
        # Stop the subscriber gracefully
        subscriber.stop()
        logger.info("MQTT subscriber stopped successfully")
    
    except FileNotFoundError:
        logger.error("Configuration file not found: config/realtime_config.yaml")
        logger.error("Please create the configuration file before running.")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error running MQTT subscriber: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
