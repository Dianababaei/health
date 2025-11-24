# Real-Time Service Module

Real-time data ingestion service for Artemis Health livestock monitoring system.

## Overview

This module provides MQTT-based real-time sensor data ingestion from cattle neck-mounted devices. It receives sensor messages via MQTT, validates data, and stores readings to the database for health monitoring and analysis.

## Architecture

```
Cattle Sensors → MQTT Broker → MQTTSubscriber → SensorDataManager → SQLite Database
                                      ↓
                                 Validation
                                 Error Handling
                                 Statistics
```

## Components

### MQTTSubscriber (`mqtt_subscriber.py`)

Main subscriber class that:
- Connects to MQTT broker with automatic reconnection
- Subscribes to sensor data topics
- Parses and validates JSON messages
- Stores validated data to database
- Tracks message statistics
- Handles errors gracefully

**Key Features:**
- ✅ Exponential backoff reconnection (5s → 10s → 20s → 40s → 60s max)
- ✅ JSON schema validation with range checks
- ✅ Thread-safe statistics tracking
- ✅ QoS 1 for reliable message delivery
- ✅ Comprehensive logging
- ✅ Graceful shutdown

## Message Format

### Valid Message Example

```json
{
  "cow_id": "COW001",
  "timestamp": "2025-11-19T10:30:00Z",
  "temperature": 38.5,
  "fxa": -0.04,
  "mya": 0.01,
  "rza": -0.88,
  "sxg": -2.88,
  "lyg": 0.14,
  "dzg": 1.87
}
```

### Required Fields

| Field | Type | Description | Valid Range |
|-------|------|-------------|-------------|
| `cow_id` | string | Cattle identifier | Any non-empty string |
| `timestamp` | string | ISO 8601 datetime | Valid ISO format |
| `temperature` | float | Body temperature (°C) | 35.0 - 42.0 |
| `fxa` | float | Forward/backward acceleration (g) | -2.0 to +2.0 |
| `mya` | float | Lateral acceleration (g) | -2.0 to +2.0 |
| `rza` | float | Vertical acceleration (g) | -2.0 to +2.0 |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `sxg` | float | Roll angular velocity (deg/s) |
| `lyg` | float | Pitch angular velocity (deg/s) |
| `dzg` | float | Yaw angular velocity (deg/s) |
| `state` | string | Behavioral state |
| `motion_intensity` | float | Overall motion level |

## Configuration

Configuration is loaded from `config/realtime_config.yaml`:

```yaml
mqtt:
  broker_host: localhost
  broker_port: 1883
  topic_pattern: "artemis/sensors/+"
  client_id: "artemis_subscriber"
  keepalive: 60
  qos: 1
  
database:
  path: "data/alert_state.db"
```

See `config/realtime_config.yaml` for full configuration options.

## Usage

### Basic Usage

```python
import yaml
from src.realtime.mqtt_subscriber import MQTTSubscriber

# Load configuration
with open('config/realtime_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create and start subscriber
subscriber = MQTTSubscriber(config)
subscriber.start()

# Run until interrupted
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    subscriber.stop()
```

### Running the Example

```bash
# Start the subscriber
python examples/mqtt_subscriber_example.py

# In another terminal, publish test messages
python examples/mqtt_test_publisher.py
```

## Error Handling

### Validation Errors

**Malformed JSON:**
```
ERROR - Malformed JSON: Expecting value: line 1 column 1 (char 0). Payload preview: not a json...
```

**Missing Required Fields:**
```
ERROR - Missing required fields for cow_id=COW001: temperature, timestamp
```

**Out-of-Range Values:**
```
ERROR - Temperature out of range for cow_id=COW001: 50.0°C (valid range: 35.0-42.0°C)
ERROR - Accelerometer fxa out of range for cow_id=COW002: 5.0g (valid range: -2.0 to +2.0g)
```

**Invalid Timestamp:**
```
ERROR - Invalid timestamp for cow_id=COW001: not-a-timestamp. Error: Invalid isoformat string
```

### Connection Errors

**Connection Failure:**
```
ERROR - Failed to connect to MQTT broker: [Errno 111] Connection refused
INFO - Attempting reconnection in 5 seconds...
INFO - Attempting reconnection in 10 seconds...
```

**Disconnection:**
```
WARNING - Disconnected from MQTT broker (reason code: 1)
INFO - Attempting reconnection in 5 seconds...
```

### Database Errors

**Insertion Failure:**
```
ERROR - Database insertion failure for cow_id=COW001: database is locked
```

**Duplicate Timestamp:**
```
INFO - Database stats: Stored 95/100 messages (5 duplicates)
```

## Statistics Logging

Statistics are logged every 60 seconds or every 100 messages:

```
INFO - Message stats: total=100, valid=92, invalid=8, stored=87, duplicates=5, storage_errors=0
INFO - Database stats: Stored 87/100 messages (5 duplicates)
```

### Statistics Tracked

- **total_received**: Total messages received
- **valid_messages**: Messages passing validation
- **invalid_messages**: Messages failing validation
- **stored_messages**: Messages successfully stored (non-duplicates)
- **duplicate_messages**: Messages skipped (duplicate timestamp)
- **storage_errors**: Database insertion failures

## Performance

### Expected Throughput

- **Message Rate:** 100+ messages/second (threaded loop)
- **Typical Farm:** ~50 cows × 1 msg/minute = 50 msg/minute
- **Large Farm:** ~500 cows × 1 msg/minute = 500 msg/minute

### Resource Usage

- **CPU:** Minimal (async I/O)
- **Memory:** ~10-20 MB baseline + message queue
- **Database:** Shared SQLite file with alert system

### Scalability

For large-scale deployments:
1. Use PostgreSQL instead of SQLite
2. Add message queue (RabbitMQ, Kafka)
3. Scale horizontally with multiple subscribers
4. Implement load balancing

## Testing

### Prerequisites

1. Install dependencies:
   ```bash
   pip install paho-mqtt pyyaml pandas
   ```

2. Start Mosquitto broker:
   ```bash
   mosquitto -v -p 1883
   ```

### Test Workflow

1. **Start Subscriber:**
   ```bash
   python examples/mqtt_subscriber_example.py
   ```

2. **Publish Test Messages:**
   ```bash
   python examples/mqtt_test_publisher.py
   ```

3. **Verify Database:**
   ```bash
   sqlite3 data/alert_state.db
   SELECT COUNT(*) FROM sensor_data;
   SELECT * FROM sensor_data LIMIT 5;
   ```

### Expected Output

**Subscriber:**
```
INFO - MQTTSubscriber initialized: broker=localhost:1883, topic=artemis/sensors/+
INFO - Connecting to MQTT broker at localhost:1883...
INFO - Connected to MQTT broker at localhost:1883
INFO - Subscribed to topic pattern: artemis/sensors/+ (QoS 1)
INFO - Message stats: total=20, valid=18, invalid=2, stored=18, duplicates=0, storage_errors=0
```

**Publisher:**
```
INFO - [1/20] Published to artemis/sensors/COW001 (valid)
INFO - [2/20] Published to artemis/sensors/COW002 (valid)
INFO - [3/20] Published to artemis/sensors/COW001 (invalid (missing_field))
```

## Deployment

### Production Setup

1. **Configure TLS:**
   ```yaml
   mqtt:
     broker_port: 8883
     use_tls: true
     ca_certs: "/path/to/ca.crt"
   ```

2. **Enable Authentication:**
   ```yaml
   mqtt:
     username: "artemis_subscriber"
     password: "secure_password"
   ```

3. **Setup Logging:**
   ```yaml
   logging:
     file:
       enabled: true
       path: "/var/log/artemis/realtime_service.log"
   ```

4. **Run as Service:**
   ```bash
   # Create systemd service
   sudo systemctl enable artemis-realtime
   sudo systemctl start artemis-realtime
   ```

### Monitoring

Monitor these metrics in production:

- **Connection Status:** Should be continuously connected
- **Message Rate:** Should match expected sensor frequency
- **Invalid Messages:** Should be <1% in normal operation
- **Storage Errors:** Should be 0 (investigate immediately if >0)
- **Duplicate Rate:** Should be <5% (higher indicates timestamp issues)

### Troubleshooting

**No Messages Received:**
1. Check MQTT broker is running: `mosquitto -v -p 1883`
2. Verify topic pattern matches published topics
3. Check network connectivity: `telnet localhost 1883`
4. Review broker logs for connection issues

**High Invalid Message Rate:**
1. Review validation error logs for patterns
2. Check sensor firmware versions
3. Verify message format from sensors
4. Consider adjusting validation ranges

**Database Errors:**
1. Check database file permissions
2. Verify disk space available
3. Check for database locks (SQLite limitations)
4. Consider migrating to PostgreSQL

**Memory Growth:**
1. Monitor message queue size
2. Check for message processing bottlenecks
3. Increase worker threads if needed
4. Review database write performance

## Integration

### With Alert System

```python
from src.realtime.mqtt_subscriber import MQTTSubscriber
from src.health_intelligence.alerts.alert_generator import AlertGenerator

# Start subscriber
subscriber = MQTTSubscriber(config)
subscriber.start()

# Alert generator can query same database
alert_gen = AlertGenerator(db_path='data/alert_state.db')
```

### With Dashboard

Dashboard automatically reads from shared database:
```python
# Dashboard queries sensor_data table
df = pd.read_sql("SELECT * FROM sensor_data WHERE cow_id=?", conn, params=[cow_id])
```

## References

- **MQTT Protocol:** http://mqtt.org/
- **Paho MQTT Python:** https://eclipse.dev/paho/files/paho.mqtt.python/html/
- **QoS Levels:** https://www.hivemq.com/blog/mqtt-essentials-part-6-mqtt-quality-of-service-levels/
- **Best Practices:** https://www.hivemq.com/blog/mqtt-client-library-encyclopedia-paho-python/

## License

Part of Artemis Health livestock monitoring system.
