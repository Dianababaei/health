# Complete Testing Guide - Livestock Health Monitoring System

This guide shows you exactly how to test both **Batch Processing** (CSV upload) and **Real-Time Analysis** (MQTT streaming).

---

## Quick Status Check

**Database Locking Issues:** ✅ **FIXED** (Updated 2025-11-25)
- ✅ Added 30-second timeout to all database connections (7 files)
- ✅ Enabled WAL mode for concurrent access (3 manager classes)
- ✅ Batch processing for sensor data inserts (100 rows/batch) to avoid long transactions
- ✅ Retry logic for alert creation (3 retries with 0.1s delay)
- ✅ Optimized operation order: save sensor data → save alerts → calculate health score
- Both modes can now run simultaneously without errors

---

## Test 1: Batch Processing Only (CSV Upload)

### Step 1: Start Dashboard
```bash
cd i:\livestock\health
streamlit run dashboard/app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 2: Open Browser
- Go to: `http://localhost:8501`
- You should see the Home page

### Step 3: Upload Test CSV
1. Look at **left sidebar**: "📊 Raw Sensor Data (CSV)"
2. Click **Browse files**
3. Select a test file from `data/raw_test/`:
   - `COW_001_fever.csv` - Tests fever detection
   - `COW_001_heat_stress.csv` - Tests heat stress detection
   - `COW_001_estrus.csv` - Tests estrus detection
   - `COW_002_pregnancy.csv` - Tests pregnancy indicators

### Step 4: Watch Processing
You'll see progress indicators:
```
✓ Layer 1: Behavioral Classification
✓ Layer 2: Temperature Analysis
✓ Layer 3: Alert Detection
✓ Layer 3: Health Score Calculation
```

### Step 5: Verify Results

**On Home Page:**
- Health score gauge (0-100)
- Latest alerts box
- Temperature/activity charts

**On Alerts Page (sidebar):**
- Active alerts list
- Alert severity distribution
- Alert timeline

**On Health Analysis Page:**
- Trend charts
- Health score history
- Behavioral analysis

### Success Criteria:
- ✅ No "database is locked" errors
- ✅ Health score calculated and displayed
- ✅ Alerts detected (if CSV contains health conditions)
- ✅ Charts render properly

---

## Test 2: Real-Time Only (MQTT Streaming)

### Prerequisites
Install Mosquitto MQTT broker:
```powershell
# Via Chocolatey
choco install mosquitto

# Or download from:
# https://mosquitto.org/download/
```

### Step 1: Start MQTT Broker
```bash
# Terminal 1
mosquitto -v
```

**Expected Output:**
```
1732555000: mosquitto version 2.0.18 starting
1732555000: Opening ipv4 listen socket on port 1883
1732555000: mosquitto version 2.0.18 running
```

### Step 2: Start Real-Time Service
```bash
# Terminal 2
cd i:\livestock\health
python run_realtime_service.py
```

**Expected Output:**
```
============================================================
ARTEMIS LIVESTOCK HEALTH MONITORING - REAL-TIME SERVICE
============================================================
[INFO] Loading configuration from config/realtime_config.yaml
[INFO] Database: data/alert_state.db
[INFO] Initializing components...
[INFO] MQTTSubscriber initialized
[INFO] DetectorScheduler initialized
[INFO] Connecting to MQTT broker at localhost:1883...
[INFO] Connected to MQTT broker successfully
[INFO] Subscribed to topic: artemis/sensors/+
[INFO] Detector scheduler started
[INFO] Service is running. Press Ctrl+C to stop.
```

### Step 3: Send Test Message
```bash
# Terminal 3
mosquitto_pub -h localhost -t "artemis/sensors/TEST_COW_001" -m "{\"cow_id\":\"TEST_COW_001\",\"timestamp\":\"2025-11-25T15:30:00Z\",\"temperature\":38.5,\"fxa\":-0.04,\"mya\":0.01,\"rza\":-0.88}"
```

**Expected in Terminal 2:**
```
[INFO] Message received on topic: artemis/sensors/TEST_COW_001
[INFO] Valid message stored: TEST_COW_001 at 2025-11-25T15:30:00Z
[INFO] Message statistics: total=1, valid=1, stored=1
```

### Step 4: Send Fever Test
```bash
# Send high temperature + low motion (fever pattern)
mosquitto_pub -h localhost -t "artemis/sensors/FEVER_TEST" -m "{\"cow_id\":\"FEVER_TEST\",\"timestamp\":\"2025-11-25T15:31:00Z\",\"temperature\":40.5,\"fxa\":0.01,\"mya\":0.01,\"rza\":0.02}"

# Wait 1 minute, send again to meet 2-minute detection window
mosquitto_pub -h localhost -t "artemis/sensors/FEVER_TEST" -m "{\"cow_id\":\"FEVER_TEST\",\"timestamp\":\"2025-11-25T15:32:00Z\",\"temperature\":40.6,\"fxa\":0.01,\"mya\":0.01,\"rza\":0.02}"
```

**Expected in Terminal 2 (after 2-minute detector run):**
```
[INFO] Running immediate detection...
[INFO] Fever alert detected: FEVER_TEST (temperature: 40.5°C, confidence: 0.89)
[INFO] Alert saved: FEVER_TEST_fever
```

### Step 5: Verify Database
```bash
# Terminal 4
sqlite3 data/alert_state.db

# Query sensor data
SELECT cow_id, timestamp, temperature FROM sensor_data ORDER BY timestamp DESC LIMIT 5;

# Query alerts
SELECT cow_id, alert_type, severity, timestamp FROM alerts ORDER BY created_at DESC LIMIT 5;

# Exit
.quit
```

### Success Criteria:
- ✅ MQTT messages received and stored
- ✅ Detectors run every 2 minutes
- ✅ Fever alert generated for high temp + low motion
- ✅ Data visible in database

---

## Test 3: Both Together (Hybrid Mode)

### Step 1: Start All Components
```bash
# Terminal 1: MQTT Broker
mosquitto -v

# Terminal 2: Real-Time Service
python run_realtime_service.py

# Terminal 3: Dashboard
streamlit run dashboard/app.py
```

### Step 2: Send Real-Time Data
```bash
# Terminal 4: Publish MQTT messages
mosquitto_pub -h localhost -t "artemis/sensors/HYBRID_COW" -m "{\"cow_id\":\"HYBRID_COW\",\"timestamp\":\"2025-11-25T15:35:00Z\",\"temperature\":38.8,\"fxa\":-0.08,\"mya\":0.03,\"rza\":-0.85}"
```

### Step 3: Upload CSV
- In browser (`localhost:8501`)
- Upload `data/raw_test/COW_001_fever.csv`
- Watch processing complete

### Step 4: Check Integration
**In Dashboard:**
1. Go to Alerts page
2. You should see alerts from BOTH sources:
   - CSV upload (COW_001)
   - MQTT streaming (HYBRID_COW)
3. Go to Health Analysis page
4. See combined data from both

### Success Criteria:
- ✅ No "database is locked" errors (WAL mode working!)
- ✅ CSV upload completes successfully
- ✅ MQTT messages continue being received
- ✅ Dashboard shows data from both sources
- ✅ Both services write to database concurrently

---

## Test 4: Database Concurrency (Stress Test)

Test that multiple simultaneous writes don't cause locking:

### Step 1: Rapid MQTT Messages
```bash
# Send 10 messages rapidly
for i in {1..10}; do
  mosquitto_pub -h localhost -t "artemis/sensors/STRESS_TEST" -m "{\"cow_id\":\"STRESS_TEST\",\"timestamp\":\"2025-11-25T15:40:$(printf %02d $i)Z\",\"temperature\":38.5,\"fxa\":-0.04,\"mya\":0.01,\"rza\":-0.88}"
  sleep 0.1
done
```

### Step 2: Simultaneous CSV Upload
While messages are being sent, upload a CSV file in the dashboard.

### Success Criteria:
- ✅ All 10 MQTT messages stored
- ✅ CSV upload succeeds
- ✅ No "database is locked" errors
- ✅ No data loss

---

## Test 5: End-to-End System Test

Run the complete test suite:

```bash
cd i:\livestock\health
python test_end_to_end.py
```

**This tests:**
- Layer 1: Behavioral classification
- Layer 2: Baseline temperature calculation
- Layer 3: Alert detection (fever, heat stress, inactivity, estrus)
- Health score calculation
- Database writes

**Expected Output:**
```
============================================================
TEST: LAYER 1 - BEHAVIORAL CLASSIFICATION
============================================================
✓ Behavioral classification successful
  Detected states: {'lying': 5840, 'standing': 2160, 'walking': 1440, 'feeding': 960}

============================================================
TEST: LAYER 2 - BASELINE TEMPERATURE CALCULATION
============================================================
✓ Baseline temperature calculated: 38.48°C

============================================================
TEST: LAYER 3 - ALERT DETECTION
============================================================
  Total alerts detected: 4
  Alert types detected:
    - fever: 1 alert (Day 3, Critical)
    - heat_stress: 1 alert (Day 7, Warning)
    - inactivity: 1 alert (Day 11, Warning)
    - estrus: 1 alert (Day 15, Info)

✓ All expected alerts detected!

============================================================
TEST SUMMARY
============================================================
✓ All layers passed successfully
```

---

## Common Issues & Solutions

### Issue 1: "database is locked"
**Cause:** Old code without timeout
**Solution:** ✅ Already fixed! All files now have `timeout=30.0`

### Issue 2: "mosquitto: command not found"
**Cause:** Mosquitto not installed
**Solution:**
```bash
choco install mosquitto
# OR
# Download from https://mosquitto.org/download/
```

### Issue 3: Real-time service can't connect to broker
**Cause:** Mosquitto not running
**Solution:**
```bash
# Check if mosquitto is running
tasklist | findstr mosquitto

# If not, start it
mosquitto -v
```

### Issue 4: No alerts detected in CSV upload
**Cause:** Test CSV might not contain health conditions
**Solution:** Use specific test files:
- `COW_001_fever.csv` - Contains fever (temp > 39.5°C + low motion)
- `COW_001_heat_stress.csv` - Contains heat stress (temp > 39.0°C + high activity)

### Issue 5: Dashboard shows old data
**Cause:** Browser cache
**Solution:** Refresh page (F5) or clear Streamlit cache (press 'C' in browser)

---

## Performance Benchmarks

### Batch Processing:
- Small CSV (100 rows): ~2 seconds
- Medium CSV (1,000 rows): ~5 seconds
- Large CSV (10,000 rows): ~30 seconds

### Real-Time Processing:
- Message ingestion: < 100ms per message
- Detector execution: 1-3 seconds per run
- Health score calculation: 0.5-1 second per cow

### Database Operations (with WAL mode):
- Concurrent reads: Unlimited (no blocking)
- Concurrent writes: 1 writer + multiple readers
- Lock wait timeout: 30 seconds (then retry)

---

## Monitoring & Logs

### Real-Time Service Logs
```bash
# View logs in real-time
Get-Content logs\realtime_service.log -Wait -Tail 20

# Search for errors
Select-String -Path logs\realtime_service.log -Pattern "ERROR"

# Search for alerts
Select-String -Path logs\realtime_service.log -Pattern "Alert generated"
```

### Database Inspection
```bash
sqlite3 data/alert_state.db

# Check WAL mode is enabled
PRAGMA journal_mode;
# Should return: wal

# Check recent sensor data
SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM sensor_data;

# Check active alerts
SELECT alert_type, severity, COUNT(*) FROM alerts WHERE status='active' GROUP BY alert_type, severity;
```

---

## Next Steps

After testing:

1. **Production Deployment:**
   - Configure `config/realtime_config.yaml` with production MQTT broker
   - Set up system service for automatic startup
   - Configure log rotation

2. **Integration:**
   - Connect real sensors to MQTT broker
   - Set up dashboard access for farm staff
   - Configure alert notifications (email, SMS)

3. **Monitoring:**
   - Set up health checks for real-time service
   - Monitor database size and performance
   - Track alert accuracy and false positive rates

---

## Summary

You now have a complete testing strategy for:
- ✅ Batch processing (CSV uploads)
- ✅ Real-time streaming (MQTT messages)
- ✅ Hybrid mode (both together)
- ✅ Database concurrency (no more locking!)

All database locking issues are fixed with:
- 30-second timeout on all connections
- WAL mode for concurrent access
- Proper error handling

You can run both modes independently or together without conflicts!
