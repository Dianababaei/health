"""
MQTT Simulator for Real-Time Testing
Simulates realistic cow sensor data and publishes to MQTT broker
"""

import json
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt


def generate_lying_data():
    """Generate sensor data for lying behavior"""
    return {
        "fxa": random.uniform(-0.05, 0.05),
        "mya": random.uniform(-0.05, 0.05),
        "rza": random.uniform(-0.95, -0.80),  # Gravity pointing down
    }


def generate_standing_data():
    """Generate sensor data for standing behavior"""
    return {
        "fxa": random.uniform(-0.1, 0.1),
        "mya": random.uniform(-0.1, 0.1),
        "rza": random.uniform(-0.2, 0.2),  # Near zero
    }


def generate_walking_data():
    """Generate sensor data for walking behavior"""
    return {
        "fxa": random.uniform(-0.3, 0.3),
        "mya": random.uniform(-0.3, 0.3),
        "rza": random.uniform(-0.3, 0.3),  # Active movement
    }


def simulate_cow_data(cow_id, scenario="normal"):
    """
    Simulate cow sensor data based on scenario

    Args:
        cow_id: Cow identifier
        scenario: "normal", "fever", "heat_stress", or "inactive"
    """

    # Base temperature
    if scenario == "fever":
        temp = random.uniform(40.0, 40.8)  # High fever
        motion = generate_lying_data()  # Sick cow lying down
    elif scenario == "heat_stress":
        temp = random.uniform(39.2, 39.8)  # Elevated temp
        motion = generate_walking_data()  # Restless, moving around
    elif scenario == "inactive":
        temp = random.uniform(38.0, 38.8)  # Normal temp
        motion = {
            "fxa": random.uniform(-0.02, 0.02),
            "mya": random.uniform(-0.02, 0.02),
            "rza": random.uniform(-0.02, 0.02),  # Very still
        }
    else:  # normal
        temp = random.uniform(38.0, 38.8)  # Normal temp
        # Random behavior
        behavior = random.choice([generate_lying_data, generate_standing_data, generate_walking_data])
        motion = behavior()

    return {
        "cow_id": cow_id,
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "temperature": round(temp, 2),
        "fxa": round(motion["fxa"], 4),
        "mya": round(motion["mya"], 4),
        "rza": round(motion["rza"], 4),
    }


def main():
    """Main simulation loop"""
    print("="*60)
    print("MQTT SIMULATOR - Real-Time Cow Data")
    print("="*60)
    print()

    # Connect to MQTT broker
    client = mqtt.Client()

    # Use port 1884 (custom config) or 1883 (standard)
    port = 1884

    try:
        client.connect("localhost", port, 60)
        print(f"✓ Connected to MQTT broker at localhost:{port}")
    except Exception as e:
        print(f"✗ Failed to connect to MQTT broker: {e}")
        print("  Make sure Mosquitto is running: mosquitto -v")
        return

    print()
    print("Select simulation scenario:")
    print("1. Normal cow (random behaviors)")
    print("2. Fever (high temp + lying)")
    print("3. Heat stress (elevated temp + active)")
    print("4. Inactive (normal temp + no movement)")
    print()

    choice = input("Enter choice (1-4) [default: 1]: ").strip() or "1"

    scenarios = {
        "1": "normal",
        "2": "fever",
        "3": "heat_stress",
        "4": "inactive"
    }

    scenario = scenarios.get(choice, "normal")
    cow_id = input("Enter cow ID [default: COW_001]: ").strip() or "COW_001"
    interval = input("Enter interval in seconds [default: 5]: ").strip() or "5"

    try:
        interval = int(interval)
    except ValueError:
        interval = 5

    print()
    print(f"Starting simulation: {scenario.upper()} scenario for {cow_id}")
    print(f"Sending data every {interval} seconds")
    print("Press Ctrl+C to stop")
    print()

    count = 0
    try:
        while True:
            # Generate data
            data = simulate_cow_data(cow_id, scenario)

            # Publish to MQTT
            topic = f"artemis/sensors/{cow_id}"
            payload = json.dumps(data)

            client.publish(topic, payload)

            count += 1
            print(f"[{count:3d}] Published: {data['timestamp']} | Temp: {data['temperature']:.1f}°C | Motion: ({data['fxa']:.2f}, {data['mya']:.2f}, {data['rza']:.2f})")

            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        print(f"\n✓ Simulation stopped. Sent {count} messages.")
        client.disconnect()


if __name__ == "__main__":
    main()
