# Data Directory Structure

## Overview

This directory contains all data used by the Artemis Livestock Health Monitoring System.

---

## Directory Layout

```
data/
├── simulation/        ← Dashboard data (demo + uploads)
├── raw/              ← Raw sensor data for ML training
├── processed/        ← Processed features for ML training
├── labels/           ← Ground truth labels for ML training
├── outputs/          ← ML model outputs
├── alert_state.db    ← SQLite database for alert management
└── README.md         ← This file
```

---

## Folder Descriptions

### `simulation/` - Dashboard Data

**Purpose:** Single folder for all dashboard data (demo + uploads)

**Contents:**
- **Demo data:** `DEMO_COW_001_*` files (14 days, fever scenario)
- **User uploads:** `{COW_ID}_*` files from dashboard upload
- **Simulation app:** `SIM_COW_001_*` files from simulation_app.py

**File formats:**
- `{COW_ID}_sensor_data.csv` - Sensor data with behavioral states
- `{COW_ID}_alerts.json` - Detected alerts
- `{COW_ID}_metadata.json` - Processing metadata
- `{COW_ID}_trend_report.json` - Health trend analysis

**Usage:**
- Dashboard reads from here automatically
- Upload workflow saves processed data here
- Demo data pre-loaded on first run

**Updated by:**
- `python generate_demo_data.py` - Generates demo data
- Dashboard upload workflow - Processes raw sensor CSV
- `streamlit run simulation_app.py` - Generates custom scenarios

---

### `raw/` - Raw Sensor Data for ML Training

**Purpose:** Original sensor data for training ML models

**Contents:**
- Raw accelerometer and gyroscope readings
- Ground truth labels (if available)
- Data from real sensors or simulations

**Usage:**
- Feature engineering pipeline input
- ML model training
- Not used by dashboard

**Tools:** `src/feature_engineering/generate_dataset.py`

---

### `processed/` - Processed Features for ML

**Purpose:** Engineered features ready for ML training

**Contents:**
- Statistical features (mean, std, min, max)
- Frequency domain features (FFT)
- Time-domain features (rolling windows)

**Usage:**
- ML model training input
- Not used by dashboard

**Tools:** `src/feature_engineering/`

---

### `labels/` - Ground Truth Labels

**Purpose:** Manually labeled behavioral states for training

**Contents:**
- CSV files with timestamp + labeled state
- Used to train behavior classification models

**Usage:**
- Supervised learning
- Model evaluation

**Tools:** Manual labeling or annotation tools

---

### `outputs/` - ML Model Outputs

**Purpose:** Results from model training and evaluation

**Contents:**
- Trained model files (.pkl)
- Evaluation metrics
- Confusion matrices
- Performance reports

**Usage:**
- Model validation
- Performance tracking

**Tools:** `src/models/train_behavior_classifiers.py`

---

### `alert_state.db` - Alert Management Database

**Purpose:** SQLite database for alert state tracking

**Schema:**
```sql
CREATE TABLE alerts (
    alert_id TEXT PRIMARY KEY,
    cow_id TEXT,
    alert_type TEXT,
    severity TEXT,
    status TEXT,  -- 'active', 'acknowledged', 'resolved'
    timestamp TEXT,
    confidence REAL,
    sensor_values TEXT,  -- JSON
    detection_details TEXT,  -- JSON
    created_at TEXT,
    updated_at TEXT
);
```

**Usage:**
- Alerts page reads from here
- Supports acknowledge/resolve operations
- Alert history tracking

**Managed by:** `src/health_intelligence/logging/alert_state_manager.py`

---

## Data Flow

### Dashboard Upload Workflow

```
1. User uploads raw sensor CSV
   ↓
2. Dashboard processes through 3 layers
   ↓
3. Saves to simulation/ folder:
   - {COW_ID}_sensor_data.csv (with states)
   - {COW_ID}_alerts.json
   - {COW_ID}_metadata.json
   ↓
4. Also saves alerts to alert_state.db
   ↓
5. Dashboard loads from simulation/ folder
```

### Demo Data Workflow

```
1. Run: python generate_demo_data.py
   ↓
2. Generates files directly in simulation/ folder
   ↓
3. Dashboard pre-loaded and ready
```

### ML Training Workflow

```
1. Collect sensor data → raw/
   ↓
2. Label behavioral states → labels/
   ↓
3. Feature engineering → processed/
   ↓
4. Train models → outputs/
   ↓
5. Deploy models to dashboard
```

---

## Cleanup Recommendations

### Safe to Delete

- **Old files in `simulation/`** - If you've uploaded many test files
  ```bash
  # Keep only demo data
  cd data/simulation
  rm -f SIM_COW_* COW_*
  # Keeps DEMO_COW_001_* files
  ```

### Keep

- **`simulation/DEMO_COW_001_*`** - Pre-loaded demo data
- **`alert_state.db`** - Alert history (can reset if needed)
- **`raw/`, `processed/`, `labels/`, `outputs/`** - ML training data

### Reset Everything

```bash
# Reset dashboard data (start fresh)
rm -rf data/simulation/*
rm data/alert_state.db

# Regenerate demo
python generate_demo_data.py
```

---

## Storage Requirements

**Demo data (14 days):**
- Sensor CSV: ~2 MB (20,160 rows)
- Alerts JSON: <1 KB (2 alerts)
- Metadata JSON: <1 KB
- Total: ~2 MB

**Alert database:**
- Empty: 100 KB
- With 1000 alerts: ~500 KB
- With 10,000 alerts: ~5 MB

**ML training data:**
- Depends on dataset size
- Typically 10-100 MB per cow for weeks of data

---

## Redundancy Removed

**Deleted folders:**
- `data/simulated/` - Empty, unused ❌
- `data/demo/` - Duplicate of simulation/ ❌
- `data/test_raw_sensor.csv` - Test file ❌

**Simplified structure:**
- Single folder `simulation/` for all dashboard data
- Demo data pre-loaded in `simulation/`
- No more copying between folders

---

## Summary

| Folder | Purpose | Used By | Generated By |
|--------|---------|---------|--------------|
| `simulation/` | Dashboard data (demo + uploads) | Dashboard | All workflows |
| `raw/` | ML training input | Feature engineering | Sensor collection |
| `processed/` | ML features | Model training | Feature engineering |
| `labels/` | Ground truth | Model training | Manual labeling |
| `outputs/` | ML results | Model evaluation | Model training |
| `alert_state.db` | Alert management | Alerts page | Upload workflow |

✅ **Simplified: 1 folder instead of 3!**
