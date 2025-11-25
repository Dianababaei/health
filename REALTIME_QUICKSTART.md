# Real-Time Service - Quick Start Guide

## 5-Minute Setup

### Step 1: Install Mosquitto MQTT Broker (Windows)

1. **Download Mosquitto**
   - Go to: https://mosquitto.org/download/
   - Download: `mosquitto-2.0.18-install-windows-x64.exe`

2. **Install**
   - Run the installer as Administrator
   - Use default installation path: `C:\Program Files\mosquitto`
   - Keep all default options

3. **Configure**
   - Open Notepad as Administrator
   - Open file: `C:\Program Files\mosquitto\mosquitto.conf`
   - Add these two lines at the end:
     ```
     listener 1883
     allow_anonymous true
     ```
   - Save and close

4. **Start Mosquitto Service**
   - Press `Win + R`
   - Type: `services.msc` and press Enter
   - Find "Mosquitto Broker" in the list
   - Right-click → Start
   - Right-click → Properties → Set "Startup type" to "Automatic"

5. **Verify Installation**
   - Open Command Prompt
   - Run:
     ```cmd
     netstat -an | findstr "1883"
     ```
   - You should see `0.0.0.0:1883` (broker is listening)

### Step 2: Install Python Dependencies

Open Command Prompt in your project directory:
```bash
cd i:\livestock\health
python -m pip install paho-mqtt apscheduler pyyaml
```

### Step 3: Start the Real-Time Service

```bash
python run_realtime_service.py
```

**Expected Output:**
```
======================================================================
  Realtime Livestock Health Monitoring Service v1.0.0
======================================================================

Loading configuration from: config/realtime_config.yaml
Configuration loaded successfully

2025-11-25 10:30:00 - INFO - Connected to MQTT broker at localhost:1883
2025-11-25 10:30:00 - INFO - Subscribed to topic pattern: artemis/sensors/+
======================================================================
Service is now running - Press Ctrl+C to stop
======================================================================

✓ Service running successfully
  Press Ctrl+C to stop...
```

Leave this running and open a new Command Prompt for testing.

---

## Testing the Service

### Test 1: Basic Connectivity

Open a new Command Prompt:
```cmd
cd "C:\Program Files\mosquitto"

mosquitto_pub -h localhost -t "test/hello" -m "Hello MQTT"
```

No error = broker is working!

### Test 2: Send a Real Sensor Message

```cmd
cd "C:\Program Files\mosquitto"

mosquitto_pub -h localhost -t "artemis/sensors/TEST_COW_001" -m "{\"cow_id\": \"TEST_COW_001\", \"timestamp\": \"2025-11-25T14:30:00Z\", \"temperature\": 38.5, \"fxa\": -0.04, \"mya\": 0.01, \"rza\": -0.88}"
```

**Check the service window** - you should see:
```
2025-11-25 14:30:05 - INFO - Message stats: total=1, valid=1, stored=1, duplicates=0
```

### Test 3: Send Multiple Messages

Create a test file `test_mqtt.py`:
```python
import json
import time
from datetime import datetime, timedelta
import paho.mqtt.client as mqtt

# Connect to broker
client = mqtt.Client()
client.connect("localhost", 1883)

# Send 10 test messages
start_time = datetime.utcnow()
for i in range(10):
    message = {
        "cow_id": "TEST_COW_001",
        "timestamp": (start_time + timedelta(minutes=i)).isoformat() + "Z",
        "temperature": 38.5 + (i * 0.05),
        "fxa": -0.04,
        "mya": 0.01,
        "rza": -0.88,
        "sxg": -2.88,
        "lyg": 0.14,
        "dzg": 1.87
    }

    topic = f"artemis/sensors/TEST_COW_001"
    client.publish(topic, json.dumps(message))
    print(f"Sent message {i+1}/10 - Temp: {message['temperature']:.2f}°C")
    time.sleep(0.5)

client.disconnect()
print("\n✓ Test complete! Check the service log for processing results.")
```

Run it:
```bash
python test_mqtt.py
```

### Test 4: Trigger a Fever Alert

Send high temperature with low motion:
```cmd
cd "C:\Program Files\mosquitto"

mosquitto_pub -h localhost -t "artemis/sensors/FEVER_TEST" -m "{\"cow_id\": \"FEVER_TEST\", \"timestamp\": \"2025-11-25T15:00:00Z\", \"temperature\": 40.5, \"fxa\": 0.01, \"mya\": 0.01, \"rza\": 0.01}"
```

Wait 2-3 minutes (detector runs every 2 minutes), then check the dashboard Alerts page.

### Test 5: Verify Database Storage

```bash
sqlite3 data/alert_state.db "SELECT cow_id, timestamp, temperature FROM sensor_data ORDER BY timestamp DESC LIMIT 5;"
```

You should see your test messages stored in the database.

---

## Quick Reference

### Start/Stop Service
```bash
# Start
python run_realtime_service.py

# Stop
Ctrl+C
```

### Monitor Logs
```bash
# View recent logs
type logs\realtime_service.log

# Watch logs in real-time (PowerShell)
Get-Content logs\realtime_service.log -Wait -Tail 20
```

### Publish Test Message
```cmd
cd "C:\Program Files\mosquitto"

mosquitto_pub -h localhost -t "artemis/sensors/COW_001" -m "{\"cow_id\": \"COW_001\", \"timestamp\": \"2025-11-25T15:30:00Z\", \"temperature\": 38.5, \"fxa\": -0.04, \"mya\": 0.01, \"rza\": -0.88}"
```

### Message Format
```json
{
  "cow_id": "COW_001",           // Required: Animal ID (string)
  "timestamp": "2025-11-25T15:30:00Z",  // Required: ISO 8601 format
  "temperature": 38.5,           // Required: 35.0-42.0°C
  "fxa": -0.04,                  // Required: -2.0 to +2.0 g
  "mya": 0.01,                   // Required: -2.0 to +2.0 g
  "rza": -0.88,                  // Required: -2.0 to +2.0 g
  "sxg": -2.88,                  // Optional: gyroscope
  "lyg": 0.14,                   // Optional: gyroscope
  "dzg": 1.87                    // Optional: gyroscope
}
```

### Topic Pattern
```
artemis/sensors/{cow_id}
```

Examples:
- `artemis/sensors/COW_001`
- `artemis/sensors/COW_042`
- `artemis/sensors/TEST_COW_001`

---

## Troubleshooting

### "Connection refused" Error

**Problem**: Service can't connect to MQTT broker

**Solution**:
1. Check Mosquitto service is running:
   - Win + R → `services.msc`
   - Find "Mosquitto Broker"
   - Status should be "Running"
2. Check port 1883 is open:
   ```cmd
   netstat -an | findstr "1883"
   ```
3. Check firewall isn't blocking port 1883

### "Invalid messages" in Log

**Problem**: Messages not being stored

**Solution**:
- Check JSON format (use online JSON validator)
- Verify timestamp format: `2025-11-25T15:30:00Z`
- Check temperature range: 35.0-42.0°C
- Check accelerometer range: -2.0 to +2.0 g

### Service Won't Start

**Problem**: Python import errors

**Solution**:
```bash
# Install missing dependencies
python -m pip install paho-mqtt apscheduler pyyaml pandas numpy
```

### No Alerts Generated

**Problem**: Detector not triggering

**Solution**:
- Wait 2-3 minutes after sending data (detector runs every 2 minutes)
- Check data meets alert thresholds:
  - Fever: temp > 39.5°C AND motion_intensity < 0.15
  - Heat stress: temp > 39.0°C AND activity > 0.6
- Verify detector is enabled in config:
  ```yaml
  detector_schedules:
    immediate_detection:
      enabled: true
  ```

---

## What Happens After Messages Are Sent?

1. **Immediate** (< 1 second):
   - Message received by MQTT subscriber
   - JSON validated
   - Stored to `sensor_data` table in database

2. **Within 2 minutes**:
   - Immediate detector runs (fever, heat stress, sensor malfunction)
   - Alerts stored to `alerts` table if thresholds exceeded

3. **Within 15 minutes**:
   - Health score calculator runs
   - Score stored to `health_scores` table

4. **Within 30 minutes**:
   - Inactivity detector runs (if enough data)

5. **Within 6 hours**:
   - Estrus detector runs (if 24+ hours of data)

6. **Dashboard**:
   - Refresh dashboard to see new data
   - Home page shows latest sensor readings
   - Alerts page shows new alerts
   - Health Analysis page shows updated scores

---

## Next Steps

1. **Install Mosquitto** following Step 1 above
2. **Start the service** with `python run_realtime_service.py`
3. **Run Test 2** to send a test message
4. **Check the dashboard** to see the data appear

For detailed documentation, see [docs/realtime_setup.md](docs/realtime_setup.md).

For production deployment, refer to the "Production Deployment" section in the full documentation.
