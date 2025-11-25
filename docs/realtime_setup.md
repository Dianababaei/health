# Real-Time MQTT Service Setup Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Mosquitto MQTT Broker Installation](#mosquitto-mqtt-broker-installation)
5. [Service Configuration](#service-configuration)
6. [Running the Service](#running-the-service)
7. [Testing](#testing)
8. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)
9. [Production Deployment](#production-deployment)

---

## Overview

The Artemis Real-Time Service enables continuous livestock health monitoring by:
- **Receiving sensor data** via MQTT protocol (one message per minute per animal)
- **Validating and storing** data to the database
- **Running scheduled detectors** at appropriate intervals (fever every 2 minutes, estrus every 6 hours, etc.)
- **Generating alerts** automatically when health issues are detected

This complements the CSV upload workflow, allowing both batch processing and real-time streaming.

---

## Architecture

```
Neck-Mounted Sensors → MQTT Broker → Real-Time Service → Database → Dashboard
                       (Mosquitto)   (Python Service)    (SQLite)   (Streamlit)
```

### Data Flow
1. **Sensors** publish JSON messages to MQTT broker every minute
2. **MQTT Subscriber** receives messages, validates format, and stores to database
3. **Detector Scheduler** runs health checks periodically:
   - Immediate detection (fever, heat stress): Every 2 minutes
   - Inactivity detection: Every 30 minutes
   - Health score calculation: Every 15 minutes
   - Estrus detection: Every 6 hours
4. **Dashboard** displays real-time health metrics and alerts

### Key Components
- **`src/realtime/mqtt_subscriber.py`** - MQTT client with JSON validation
- **`src/realtime/scheduler.py`** - APScheduler integration for periodic tasks
- **`src/realtime/pipeline.py`** - Detector orchestration
- **`run_realtime_service.py`** - Main entry point with lifecycle management
- **`config/realtime_config.yaml`** - Configuration file

---

## Prerequisites

### Software Requirements
- **Python 3.11+** (confirmed working version)
- **MQTT Broker** (Mosquitto recommended)
- **Operating System**: Windows, Linux, or macOS

### Python Dependencies
Install required packages:
```bash
python -m pip install paho-mqtt apscheduler pyyaml pandas numpy sqlite3
```

Or use the project's requirements file if available:
```bash
python -m pip install -r requirements.txt
```

---

## Mosquitto MQTT Broker Installation

### Windows Installation

1. **Download Mosquitto**
   - Visit: https://mosquitto.org/download/
   - Download the Windows installer (64-bit recommended)
   - Example: `mosquitto-2.0.18-install-windows-x64.exe`

2. **Install Mosquitto**
   - Run the installer as Administrator
   - Default installation path: `C:\Program Files\mosquitto`
   - Keep all default options selected

3. **Install Mosquitto as a Service**
   - Open Command Prompt as Administrator
   - Navigate to Mosquitto directory:
     ```cmd
     cd "C:\Program Files\mosquitto"
     ```
   - Install the service:
     ```cmd
     mosquitto install
     ```

4. **Create Configuration File**
   - Edit `C:\Program Files\mosquitto\mosquitto.conf`
   - Add the following lines:
     ```
     listener 1883
     allow_anonymous true
     ```
   - **Note**: `allow_anonymous true` is for development only. See [Production Deployment](#production-deployment) for secure configuration.

5. **Start the Service**
   - Open Services (Win + R → `services.msc`)
   - Find "Mosquitto Broker"
   - Right-click → Start
   - Set Startup Type to "Automatic" for auto-start on boot

6. **Verify Installation**
   ```cmd
   netstat -an | findstr "1883"
   ```
   - You should see `0.0.0.0:1883` or `:::1883` (broker is listening)

### Linux Installation (Ubuntu/Debian)

```bash
# Update package list
sudo apt update

# Install Mosquitto broker and clients
sudo apt install mosquitto mosquitto-clients

# Enable and start service
sudo systemctl enable mosquitto
sudo systemctl start mosquitto

# Check status
sudo systemctl status mosquitto

# Verify broker is listening
netstat -tuln | grep 1883
```

**Configuration** (optional):
```bash
# Edit config file
sudo nano /etc/mosquitto/mosquitto.conf

# Add these lines for development:
listener 1883
allow_anonymous true

# Restart service
sudo systemctl restart mosquitto
```

### macOS Installation

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Mosquitto
brew install mosquitto

# Start Mosquitto service
brew services start mosquitto

# Verify installation
mosquitto -h
```

**Configuration** (optional):
```bash
# Edit config file
nano /opt/homebrew/etc/mosquitto/mosquitto.conf

# Add these lines:
listener 1883
allow_anonymous true

# Restart service
brew services restart mosquitto
```

---

## Service Configuration

### 1. Configuration File Location
The service uses `config/realtime_config.yaml` for all settings.

### 2. Key Configuration Sections

#### MQTT Broker Settings
```yaml
mqtt:
  broker_host: localhost         # Change to broker IP if remote
  broker_port: 1883              # Standard MQTT port
  topic_pattern: "artemis/sensors/+"  # Topic pattern (+ = cow_id wildcard)
  client_id: "artemis_subscriber"
  keepalive: 60
  qos: 1                         # Quality of Service (0, 1, or 2)

  # Optional authentication
  username: null                 # Set if broker requires auth
  password: null

  # Reconnection settings
  reconnect:
    initial_delay: 5             # Seconds to wait before first retry
    max_delay: 60                # Maximum delay between retries
    backoff_multiplier: 2        # Exponential backoff factor
```

#### Database Settings
```yaml
database:
  path: "data/alert_state.db"   # SQLite database path (shared with dashboard)
```

#### Detector Schedules
```yaml
detector_schedules:
  immediate_detection:
    interval_minutes: 2          # Fever, heat stress, sensor malfunction
    enabled: true

  inactivity_detection:
    interval_minutes: 30
    enabled: true

  health_scoring:
    interval_minutes: 15
    enabled: true

  estrus_detection:
    interval_hours: 6
    enabled: true
```

#### Logging Settings
```yaml
logging:
  level: INFO                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: "logs/realtime_service.log"
  max_bytes: 10485760            # 10 MB
  backup_count: 5
  log_to_console: true
```

### 3. Message Format

Sensors must publish JSON messages to topic `artemis/sensors/{cow_id}`:

```json
{
  "cow_id": "COW_001",
  "timestamp": "2025-11-19T14:30:00Z",
  "temperature": 38.5,
  "fxa": -0.04,
  "mya": 0.01,
  "rza": -0.88,
  "sxg": -2.88,
  "lyg": 0.14,
  "dzg": 1.87
}
```

**Required Fields**:
- `cow_id` (string): Animal identifier
- `timestamp` (ISO 8601 string): Message timestamp
- `temperature` (float): Body temperature in °C (valid range: 35.0-42.0)
- `fxa`, `mya`, `rza` (float): Accelerometer readings in g-force (valid range: -2.0 to +2.0)

**Optional Fields**:
- `sxg`, `lyg`, `dzg` (float): Gyroscope readings
- `state` (string): Behavioral state (lying, standing, walking, feeding)
- `motion_intensity` (float): Pre-calculated motion intensity

**Validation Rules**:
- Invalid messages are logged but not stored
- Duplicate timestamps are skipped (idempotency)
- Out-of-range values are rejected

---

## Running the Service

### Start the Service

```bash
# Navigate to project directory
cd i:\livestock\health

# Run the service
python run_realtime_service.py
```

**Expected Output**:
```
======================================================================
  Realtime Livestock Health Monitoring Service v1.0.0
======================================================================

Loading configuration from: config/realtime_config.yaml
Configuration loaded successfully

2025-11-25 10:30:00 - INFO - Testing database connection: data/alert_state.db
2025-11-25 10:30:00 - INFO - Database connection successful
2025-11-25 10:30:00 - INFO - Initializing MQTTSubscriber...
2025-11-25 10:30:00 - INFO - Connected to MQTT broker at localhost:1883
2025-11-25 10:30:00 - INFO - Subscribed to topic pattern: artemis/sensors/+ (QoS 1)
2025-11-25 10:30:00 - INFO - DetectorScheduler started successfully
======================================================================
Service is now running - Press Ctrl+C to stop
======================================================================

✓ Service running successfully
  Press Ctrl+C to stop...
```

### Stop the Service

Press **Ctrl+C** to initiate graceful shutdown:
- MQTT subscriber stops receiving messages
- Detector scheduler completes current tasks (up to 30 seconds)
- Database connections are closed
- Service logs final statistics

### Run as Background Service (Linux/macOS)

#### Using nohup
```bash
nohup python run_realtime_service.py > logs/service_stdout.log 2>&1 &
echo $! > service.pid
```

**To stop**:
```bash
kill $(cat service.pid)
```

#### Using screen
```bash
screen -S artemis_service
python run_realtime_service.py

# Detach: Ctrl+A, then D
# Reattach: screen -r artemis_service
```

#### Using systemd (Linux)
Create `/etc/systemd/system/artemis-realtime.service`:
```ini
[Unit]
Description=Artemis Real-Time Health Monitoring Service
After=network.target mosquitto.service

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/livestock/health
ExecStart=/usr/bin/python3 run_realtime_service.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable artemis-realtime
sudo systemctl start artemis-realtime
sudo systemctl status artemis-realtime
```

---

## Testing

### 1. Test Broker Connectivity

#### Windows
```cmd
cd "C:\Program Files\mosquitto"

# Subscribe to test topic (Terminal 1)
mosquitto_sub -h localhost -t "test/topic" -v

# Publish test message (Terminal 2)
mosquitto_pub -h localhost -t "test/topic" -m "Hello MQTT"
```

#### Linux/macOS
```bash
# Subscribe to test topic (Terminal 1)
mosquitto_sub -h localhost -t "test/topic" -v

# Publish test message (Terminal 2)
mosquitto_pub -h localhost -t "test/topic" -m "Hello MQTT"
```

**Expected**: You should see "test/topic Hello MQTT" in Terminal 1.

### 2. Test Sensor Message Publishing

With the real-time service running, publish a test sensor message:

```bash
mosquitto_pub -h localhost -t "artemis/sensors/TEST_COW_001" -m '{
  "cow_id": "TEST_COW_001",
  "timestamp": "2025-11-25T14:30:00Z",
  "temperature": 38.5,
  "fxa": -0.04,
  "mya": 0.01,
  "rza": -0.88,
  "sxg": -2.88,
  "lyg": 0.14,
  "dzg": 1.87
}'
```

**Expected Service Log Output**:
```
2025-11-25 14:30:05 - INFO - Message stats: total=1, valid=1, invalid=0, stored=1, duplicates=0, storage_errors=0
```

### 3. Test Invalid Message Handling

```bash
# Missing required field (temperature)
mosquitto_pub -h localhost -t "artemis/sensors/TEST_COW_002" -m '{
  "cow_id": "TEST_COW_002",
  "timestamp": "2025-11-25T14:31:00Z",
  "fxa": -0.04,
  "mya": 0.01,
  "rza": -0.88
}'

# Invalid temperature range
mosquitto_pub -h localhost -t "artemis/sensors/TEST_COW_003" -m '{
  "cow_id": "TEST_COW_003",
  "timestamp": "2025-11-25T14:32:00Z",
  "temperature": 50.0,
  "fxa": -0.04,
  "mya": 0.01,
  "rza": -0.88
}'
```

**Expected Service Log Output**:
```
2025-11-25 14:31:05 - ERROR - Missing required fields for cow_id=TEST_COW_002: temperature
2025-11-25 14:32:05 - ERROR - Temperature out of range for cow_id=TEST_COW_003: 50.0°C (valid range: 35.0-42.0°C)
```

### 4. Subscribe to Service Topics

Monitor all sensor messages:
```bash
mosquitto_sub -h localhost -t "artemis/sensors/#" -v
```

### 5. Test Detector Triggering

Publish data that should trigger fever alert:
```bash
mosquitto_pub -h localhost -t "artemis/sensors/FEVER_TEST" -m '{
  "cow_id": "FEVER_TEST",
  "timestamp": "2025-11-25T14:35:00Z",
  "temperature": 40.5,
  "fxa": 0.01,
  "mya": 0.01,
  "rza": 0.01
}'
```

Wait 2-3 minutes for immediate detection to run, then check dashboard Alerts page.

### 6. Verify Database Storage

```bash
# Check raw sensor data
sqlite3 data/alert_state.db "SELECT * FROM sensor_data WHERE cow_id='TEST_COW_001' ORDER BY timestamp DESC LIMIT 5;"

# Check alerts
sqlite3 data/alert_state.db "SELECT * FROM alerts WHERE cow_id='FEVER_TEST' ORDER BY timestamp DESC LIMIT 5;"
```

### 7. Automated Testing with Python

Create a test script `test_mqtt_integration.py`:
```python
import time
import json
import paho.mqtt.client as mqtt
from datetime import datetime

# MQTT settings
broker = "localhost"
port = 1883
topic_base = "artemis/sensors"

# Create test message
def create_test_message(cow_id, temp=38.5):
    return {
        "cow_id": cow_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "temperature": temp,
        "fxa": -0.04,
        "mya": 0.01,
        "rza": -0.88
    }

# Publish test messages
client = mqtt.Client()
client.connect(broker, port)

for i in range(10):
    cow_id = f"TEST_COW_{i:03d}"
    message = create_test_message(cow_id, temp=38.0 + i * 0.1)
    topic = f"{topic_base}/{cow_id}"

    client.publish(topic, json.dumps(message))
    print(f"Published message for {cow_id}")
    time.sleep(1)

client.disconnect()
print("Test complete")
```

Run the test:
```bash
python test_mqtt_integration.py
```

---

## Monitoring and Troubleshooting

### Log Files

**Service Log**: `logs/realtime_service.log`
- Connection status
- Message validation errors
- Detector execution results
- Performance statistics

**Check Recent Logs**:
```bash
# Windows
type logs\realtime_service.log | more

# Linux/macOS
tail -f logs/realtime_service.log
```

### Common Issues

#### 1. Service Won't Start - "Connection refused"

**Symptoms**:
```
ERROR - Failed to connect to MQTT broker: [Errno 111] Connection refused
```

**Solutions**:
- Verify Mosquitto is running:
  - Windows: `services.msc` → Check "Mosquitto Broker" status
  - Linux: `sudo systemctl status mosquitto`
  - macOS: `brew services list | grep mosquitto`
- Check firewall is not blocking port 1883
- Verify `broker_host` and `broker_port` in config

#### 2. Messages Not Being Stored

**Symptoms**:
```
INFO - Message stats: total=10, valid=0, invalid=10, stored=0
```

**Solutions**:
- Check message format matches expected schema
- Review service log for validation errors
- Verify timestamp format is ISO 8601
- Check temperature and accelerometer ranges

#### 3. Detector Not Running

**Symptoms**:
- No alerts generated despite abnormal data
- Service log shows no detector execution messages

**Solutions**:
- Check `detector_schedules` in config (ensure `enabled: true`)
- Verify sufficient data exists in database (need historical data for trends)
- Check detector-specific requirements (e.g., inactivity needs 4+ hours of data)

#### 4. Database Lock Errors

**Symptoms**:
```
ERROR - Database insertion failure: database is locked
```

**Solutions**:
- Ensure dashboard is not running intensive queries concurrently
- Check `database.timeout` in config (increase if needed)
- Consider enabling WAL mode: `database.enable_wal: true`

#### 5. High Memory Usage

**Symptoms**:
- Service consumes excessive RAM over time

**Solutions**:
- Reduce `performance.max_queue_size` in config
- Lower `performance.cache_max_size`
- Restart service periodically (e.g., daily cron job)

### Performance Monitoring

**Check Message Processing Rate**:
```bash
grep "Message stats" logs/realtime_service.log | tail -1
```

**Expected Output**:
```
2025-11-25 15:00:00 - INFO - Message stats: total=360, valid=360, invalid=0, stored=360, duplicates=0, storage_errors=0
```

**Monitor Database Size**:
```bash
# Windows
dir data\alert_state.db

# Linux/macOS
ls -lh data/alert_state.db
```

---

## Production Deployment

### Security Hardening

#### 1. Enable MQTT Authentication

Edit Mosquitto config (`mosquitto.conf`):
```
listener 1883
allow_anonymous false
password_file /path/to/mosquitto_passwd
```

Create password file:
```bash
mosquitto_passwd -c /path/to/mosquitto_passwd artemis_user
```

Update `config/realtime_config.yaml`:
```yaml
mqtt:
  username: artemis_user
  password: your_secure_password  # Use environment variable in production
```

#### 2. Enable TLS/SSL Encryption

Edit Mosquitto config:
```
listener 8883
cafile /path/to/ca.crt
certfile /path/to/server.crt
keyfile /path/to/server.key
require_certificate false
```

Update service config:
```yaml
mqtt:
  broker_port: 8883
  use_tls: true
  ca_certs: "/path/to/ca.crt"
```

#### 3. Network Segmentation

- Run MQTT broker on isolated network segment
- Use firewall rules to restrict access to port 1883/8883
- Consider VPN for remote sensor connectivity

### High Availability

#### 1. Database Backup

Create automated backup script `backup_database.sh`:
```bash
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups"
DB_PATH="data/alert_state.db"

mkdir -p $BACKUP_DIR
sqlite3 $DB_PATH ".backup $BACKUP_DIR/alert_state_$DATE.db"
echo "Backup created: $BACKUP_DIR/alert_state_$DATE.db"

# Keep only last 7 days of backups
find $BACKUP_DIR -name "alert_state_*.db" -mtime +7 -delete
```

Schedule with cron:
```bash
0 2 * * * /path/to/backup_database.sh
```

#### 2. Service Monitoring

Use systemd with automatic restart:
```ini
[Service]
Restart=on-failure
RestartSec=10
```

Or use a monitoring tool like Supervisor:
```ini
[program:artemis-realtime]
command=/usr/bin/python3 run_realtime_service.py
directory=/path/to/livestock/health
autostart=true
autorestart=true
stderr_logfile=/var/log/artemis/err.log
stdout_logfile=/var/log/artemis/out.log
```

#### 3. Health Check Endpoint

The service includes health check capability (see `config/realtime_config.yaml`):
```yaml
service:
  enable_health_check: true
  health_check_port: 8080
```

Check service health:
```bash
curl http://localhost:8080/health
```

### Scaling Considerations

For farms with **100+ animals**:

1. **Database**: Migrate from SQLite to PostgreSQL
   - Better concurrent write performance
   - Improved query optimization
   - Support for horizontal scaling

2. **Message Broker**: Use MQTT clustering
   - Mosquitto bridge mode for multi-broker setup
   - HiveMQ or VerneMQ for enterprise-grade clustering

3. **Service Architecture**: Separate ingestion from detection
   - Ingestion service: MQTT subscriber only
   - Detection service: Scheduled detectors only
   - Communicate via message queue (RabbitMQ, Kafka)

---

## Quick Reference

### Service Commands

```bash
# Start service
python run_realtime_service.py

# Stop service
Ctrl+C

# Check logs
tail -f logs/realtime_service.log

# Test MQTT broker
mosquitto_pub -h localhost -t "test/topic" -m "test"
mosquitto_sub -h localhost -t "test/topic" -v
```

### Configuration File Locations

- **Service Config**: `config/realtime_config.yaml`
- **Mosquitto Config**:
  - Windows: `C:\Program Files\mosquitto\mosquitto.conf`
  - Linux: `/etc/mosquitto/mosquitto.conf`
  - macOS: `/opt/homebrew/etc/mosquitto/mosquitto.conf`
- **Service Log**: `logs/realtime_service.log`
- **Database**: `data/alert_state.db`

### MQTT Topic Structure

```
artemis/sensors/{cow_id}
```

Example topics:
- `artemis/sensors/COW_001`
- `artemis/sensors/COW_042`
- `artemis/sensors/HERD_A_123`

Subscribe to all sensors: `artemis/sensors/#`

---

## Support and Resources

- **Mosquitto Documentation**: https://mosquitto.org/documentation/
- **Paho MQTT Python Client**: https://eclipse.dev/paho/files/paho.mqtt.python/html/
- **APScheduler Documentation**: https://apscheduler.readthedocs.io/

For issues with the Artemis system, check `docs/troubleshooting.md` or review the main `README.md`.
