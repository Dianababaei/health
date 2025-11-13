# Demo Data - Quick Start Guide

## ✅ Demo Data Already Generated!

I've generated realistic demo data for you to explore the dashboard features.

### Demo Data Details:

**Cow:** DEMO_COW_001
**Duration:** 14 days
**Condition:** Fever on Day 3 (2 days duration)
**Data:** 20,160 sensor readings (1 per minute)

**Fever Details:**
- Temperature: ~40.0°C (baseline: 38.5°C)
- Motion: 0.02-0.08 (very low, indicating sick cow)
- Duration: 2 days (48 hours)
- Starts: Day 3 at midnight

---

## Quick Start - View Demo Data

### Option 1: Files Already in Simulation Directory

The demo files are already copied to `data/simulation/`:
- `DEMO_COW_001_sensor_data.csv` (3.4 MB, 20,160 readings)
- `DEMO_COW_001_alerts.json` (fever + inactivity alerts)
- `DEMO_COW_001_metadata.json` (simulation settings)

**Just run the app:**
```bash
streamlit run dashboard/app.py
```

The data should load automatically!

---

### Option 2: Upload Files Manually

1. **Run the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Go to Home page**

3. **Upload files from `data/demo/` folder:**
   - Upload `DEMO_COW_001_sensor_data.csv`
   - Upload `DEMO_COW_001_alerts.json`
   - Upload `DEMO_COW_001_metadata.json`

4. **Click "Refresh to Load Data"**

5. **Explore!**

---

## What You'll See

### Home Page:
- Health score
- Temperature reading
- Alert notifications (fever + inactivity)
- Activity metrics
- Quick overview

### Alerts Page:
- Fever alert details
- Inactivity alerts
- Alert timeline
- Severity indicators

### Health Analysis Page:
- Temperature trends over 14 days
- Behavioral patterns
- Health score trends
- Activity correlation

---

## Understanding the Demo

### Timeline:

**Days 1-2 (Normal):**
- Temperature: ~38.5°C
- Normal activity patterns
- No alerts

**Days 3-4 (Fever):**
- Temperature: ~40°C ⚠️
- Very low motion (cow is sick)
- Fever alerts triggered
- Inactivity alerts triggered

**Days 5-14 (Recovery):**
- Temperature: returns to ~38.5°C
- Activity increases
- Health improving

### Key Features to Explore:

1. **Temperature Chart**
   - See the spike on day 3
   - Watch recovery over time

2. **Alert System**
   - Fever detected automatically
   - Inactivity flagged (sick cow lying down)

3. **Behavioral States**
   - Normal: lying, standing, feeding, ruminating
   - Fever period: mostly lying (low energy)

4. **Health Score**
   - Drops during fever
   - Recovers gradually

---

## Generate New Demo Data

Want to try different scenarios?

```bash
python generate_demo_data.py
```

This will regenerate data with:
- Fresh random behavioral patterns
- Same fever scenario (day 3, 2 days)
- New alert detections

---

## Explore Different Scenarios

### Coming Soon:
- Estrus detection demo
- Pregnancy tracking demo
- Heat stress scenario
- Multiple cows comparison

For now, the fever scenario demonstrates:
✅ Alert detection algorithms
✅ Temperature monitoring
✅ Activity correlation
✅ Health scoring
✅ Multi-day trends

---

## Files Location

```
data/
├── demo/              # Original demo files
│   ├── DEMO_COW_001_sensor_data.csv
│   ├── DEMO_COW_001_alerts.json
│   └── DEMO_COW_001_metadata.json
│
└── simulation/        # Copy for auto-loading
    ├── DEMO_COW_001_sensor_data.csv  (auto-loads in dashboard)
    ├── DEMO_COW_001_alerts.json
    └── DEMO_COW_001_metadata.json
```

---

## Troubleshooting

**Problem:** No data showing in dashboard

**Solutions:**
1. Check files are in `data/simulation/` directory
2. Refresh browser (F5)
3. Try manual upload (Option 2 above)
4. Check Home page sidebar for upload section

---

**Problem:** Want to reset and try again

**Solution:**
```bash
# Regenerate demo data
python generate_demo_data.py

# Copy to simulation directory
cp data/demo/DEMO_COW_001_*.* data/simulation/

# Restart dashboard
streamlit run dashboard/app.py
```

---

## Next Steps

After exploring the demo:

1. **Try the simulation app** to generate custom scenarios:
   ```bash
   streamlit run simulation_app.py
   ```

2. **Connect real sensors** when available

3. **Customize alert thresholds** in the Health Analysis page

4. **Export data** for further analysis

---

**Enjoy exploring the Artemis Livestock Health Monitoring System!**
