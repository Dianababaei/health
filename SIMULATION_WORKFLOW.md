# Simulation Workflow - Completely Separate Apps

## Two Separate Apps

1. **Simulation App** (`simulation_app.py`) - Generate data only
2. **Main Dashboard** (`dashboard/app.py`) - Your livestock monitoring app

## Workflow

### Step 1: Generate Data

```bash
streamlit run simulation_app.py
```

- Configure settings (duration, temperature, enable Fever)
- Click "Generate Data"
- Download 3 files (CSV, JSON, JSON)
- Close this app

### Step 2: Use Main App

```bash
streamlit run dashboard/app.py
```

- Go to Home page
- Upload 3 files in sidebar
- Click "Refresh to Load Data"
- View alerts on Home and Alerts pages

## Why Separate Apps?

✅ Zero interference between simulation and main app
✅ No caching issues
✅ No accidental data mixing
✅ Clean, simple workflow

## Quick Start

**Windows:**
```bash
# Generate data
python -m streamlit run simulation_app.py

# Use main app
python -m streamlit run dashboard/app.py
```

**Linux/Mac:**
```bash
# Generate data
streamlit run simulation_app.py

# Use main app  
streamlit run dashboard/app.py
```
