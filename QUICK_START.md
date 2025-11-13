# Quick Start Guide - Artemis Livestock Health Monitoring

## 🚀 Fastest Way to See the Demo

### Option 1: Use Pre-Generated Demo Data (Recommended)

**Demo data is already ready!**

```bash
# Just run the main dashboard
streamlit run dashboard/app.py
```

OR double-click: `run_demo.bat`

**The demo data is already loaded** in `data/simulation/` folder:
- 14 days of sensor data (20,160 readings)
- 2 health alerts (1 fever + 1 inactivity)
- Fever scenario on Day 3

**Just open the dashboard and explore!**

---

### Option 2: Generate Fresh Demo Data

Want new random data?

```bash
# Generate new demo data
python generate_demo_data.py

# Run dashboard
streamlit run dashboard/app.py
```

---

## 📊 What You'll See

### Home Page
- Health score overview
- Temperature reading
- **2 alerts** (fever + inactivity)
- Activity metrics

### Alerts Page
- **Fever alert** (critical) - Day 3
- **Inactivity alert** (warning) - sick cow lying down
- Alert details and timeline

### Health Analysis Page
- 14-day temperature chart (see fever spike!)
- Behavioral patterns
- Health trends

---

## 🧪 Generate Custom Scenarios

Want to create your own test scenarios?

```bash
# Run standalone simulation app
streamlit run simulation_app.py
```

1. Configure settings (duration, fever, estrus, etc.)
2. Click "Generate Data"
3. Download 3 files
4. Upload in main dashboard

---

## 📁 Files Overview

```
data/
├── simulation/           ← Main dashboard reads from here
│   ├── DEMO_COW_001_sensor_data.csv  (already there!)
│   ├── DEMO_COW_001_alerts.json
│   └── DEMO_COW_001_metadata.json
│
└── alert_state.db       ← Alerts database (already populated!)
```

---

## ⚡ Quick Commands

```bash
# View demo (fastest)
run_demo.bat              # Windows
streamlit run dashboard/app.py  # Any OS

# Generate new demo
python generate_demo_data.py

# Custom simulation
streamlit run simulation_app.py

# Main dashboard
streamlit run dashboard/app.py
```

---

## 🔧 Troubleshooting

**Problem:** ModuleNotFoundError: yaml

**Solution:** This is optional, demo works without it. But if you want to install:
```bash
pip install pyyaml
```

---

**Problem:** No data showing

**Solution:** Data is already in `data/simulation/`, just refresh browser (F5)

---

**Problem:** Want to reset everything

**Solution:**
```bash
python generate_demo_data.py
cp data/demo/DEMO_COW_001_*.* data/simulation/
```

---

## 📚 Documentation

- [APP_STRUCTURE.md](APP_STRUCTURE.md) - 3-page dashboard structure
- [SIMULATION_WORKFLOW.md](SIMULATION_WORKFLOW.md) - How to use simulation app
- [DATA_STORAGE_EXPLAINED.md](DATA_STORAGE_EXPLAINED.md) - Where data is stored
- [SYSTEM_REVIEW.md](SYSTEM_REVIEW.md) - Technical details
- [DEMO_DATA_README.md](DEMO_DATA_README.md) - Demo data details

---

## 🎯 30-Second Demo

```bash
# 1. Run dashboard
streamlit run dashboard/app.py

# 2. Open browser (auto-opens to http://localhost:8501)

# 3. Explore:
#    - Home: See health score + 2 alerts
#    - Alerts: View fever + inactivity details
#    - Health Analysis: See 14-day trends

# That's it! Demo data is pre-loaded.
```

---

**Enjoy exploring the Artemis Livestock Health Monitoring System!** 🐄
