# Artemis Livestock Health Monitoring - Quick Start Guide

Two simple ways to monitor cow health: **CSV Upload** or **Real-Time MQTT**

---

## Choose Your Mode

### 🗂️ Mode 1: CSV Upload (Batch Processing)
**When to use:** Analyze historical data or test the system

**Best for:**
- ✅ Reviewing past records
- ✅ Testing with sample data
- ✅ One-time analysis

**How it works:** Upload CSV file → Instant analysis → View results

---

### 📡 Mode 2: Real-Time MQTT (Live Monitoring)
**When to use:** Monitor live farm operations

**Best for:**
- ✅ Continuous health monitoring
- ✅ Immediate alerts
- ✅ Production farms

**How it works:** Sensors send data every minute → Automatic health checks → Live alerts

---

## 🗂️ Mode 1: CSV Upload - 3 Easy Steps

### Step 1: Start Dashboard
```bash
streamlit run dashboard/app.py
```
Open browser: `http://localhost:8501`

### Step 2: Upload CSV File
- Sidebar: Click **"📊 Raw Sensor Data (CSV)"**
- Click **Browse files**
- Select your CSV file

### Step 3: View Results
Dashboard shows:
- Health score (0-100)
- Alerts (Fever, Heat Stress, etc.)
- Temperature trends

### CSV Format
```csv
timestamp,cow_id,temperature,fxa,mya,rza
2025-11-26T10:00:00,COW_001,38.5,-0.04,0.01,-0.88
```

**That's it!** The system automatically analyzes and shows results.

---

## 📡 Mode 2: Real-Time MQTT - 4 Easy Steps

### Step 1: Install Mosquitto
```bash
choco install mosquitto
```

### Step 2: Start MQTT Broker
```bash
mosquitto -c mosquitto_test.conf -v
```

### Step 3: Start Real-Time Service
```bash
python run_realtime_service.py
```

### Step 4: Send Sensor Data
Your sensors send JSON to `artemis/sensors/{cow_id}`:
```json
{
  "cow_id": "COW_001",
  "timestamp": "2025-11-26T10:30:00Z",
  "temperature": 38.5,
  "fxa": -0.04,
  "mya": 0.01,
  "rza": -0.88
}
```

**Testing without sensors?** Run the simulator:
```bash
python simulate_mqtt_data.py
```

**View live data:** Start dashboard: `streamlit run dashboard/app.py`

---

## Quick Comparison

| Feature | CSV Upload | Real-Time MQTT |
|---------|-----------|----------------|
| **Setup Time** | 1 minute | 5 minutes |
| **Best For** | Past data | Live monitoring |
| **Processing** | Instant | Continuous |
| **Alerts** | After upload | Every 2 minutes |
| **When to Use** | Testing, history | Production farms |

---

## Common Questions

**Q: Can I use both modes together?**
A: Yes! Run real-time monitoring + upload historical CSV files anytime.

**Q: Which mode for testing?**
A: CSV Upload - just drag and drop a test file.

**Q: Which mode for my farm?**
A: Real-Time MQTT - continuous monitoring with instant alerts.

**Q: How often do sensors send data?**
A: Every 1 minute (one reading per cow per minute).

**Q: Where is data stored?**
A: SQLite database at `data/alert_state.db`

---

## Need Help?

**Quick Fixes:**
- Dashboard not loading? Press `Ctrl+C` and restart
- CSV upload fails? Check file has: timestamp, cow_id, temperature, fxa, mya, rza
- Real-time not working? Restart service: `Ctrl+C` → `python run_realtime_service.py`

**More Help:**
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Detailed testing steps
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues

---

**Last Updated**: 2025-11-26 | **Version**: 1.0.0
