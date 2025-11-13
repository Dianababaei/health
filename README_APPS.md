# Livestock Health Monitoring - Application Setup

## Two Separate Applications

This project has **two independent Streamlit apps**:

### 1. Simulation Data Generator (`simulation_app.py`)

**Purpose:** Generate realistic test data for development and testing

**Run:**
```bash
streamlit run simulation_app.py
# OR double-click: run_simulation.bat
```

**Features:**
- Configure simulation parameters (duration, temperature, conditions)
- Generate synthetic sensor data with health conditions
- Detect alerts automatically
- Download 3 files: CSV (sensor data), JSON (alerts), JSON (metadata)

**Use when:** You need test data for development

---

### 2. Main Dashboard (`dashboard/app.py`)

**Purpose:** Production livestock health monitoring dashboard

**Run:**
```bash
streamlit run dashboard/app.py
# OR double-click: run_main_app.bat
```

**Features:**
- Home page with health overview
- Alerts page with detailed alert management
- Health trends analysis
- Temperature monitoring
- Upload simulation data for testing

**Use when:** Monitoring actual livestock OR testing with simulated data

---

## Typical Workflow

### For Development/Testing:

1. **Generate test data:**
   - Run `simulation_app.py`
   - Configure settings (enable Fever, set duration, etc.)
   - Click "Generate Data"
   - Download 3 files
   - Close simulation app

2. **Test main app:**
   - Run `dashboard/app.py`
   - Go to Home page
   - Upload 3 files in sidebar
   - Click "Refresh to Load Data"
   - Test features with simulated data

### For Production:

- Run `dashboard/app.py`
- Connect to real data sources
- Monitor live cow health

---

## File Structure

```
livestock/health/
├── simulation_app.py          # Standalone simulation generator
├── dashboard/
│   ├── app.py                 # Main dashboard app
│   └── pages/
│       ├── 0_Home.py          # Home page (with upload)
│       ├── 2_Alerts.py        # Alerts page
│       ├── 3_Health_Trends.py # Trends analysis
│       └── 4_Temperature.py   # Temperature monitoring
├── run_simulation.bat         # Windows: Run simulation
├── run_main_app.bat           # Windows: Run main app
└── SIMULATION_WORKFLOW.md     # Detailed workflow guide
```

---

## Why Separate Apps?

✅ **No interference:** Simulation can't accidentally affect production data
✅ **No caching issues:** Each app runs independently
✅ **Clear separation:** Development vs Production
✅ **Flexibility:** Run one or both as needed

---

## Quick Commands

```bash
# Generate test data
streamlit run simulation_app.py

# Run main dashboard
streamlit run dashboard/app.py

# Run both (separate terminals)
# Terminal 1:
streamlit run simulation_app.py

# Terminal 2:
streamlit run dashboard/app.py
```

---

## Need Help?

See [SIMULATION_WORKFLOW.md](SIMULATION_WORKFLOW.md) for detailed step-by-step instructions.
