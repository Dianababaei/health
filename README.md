# Artemis Livestock Health Monitoring System

Production-ready cattle health monitoring using neck-mounted sensors. Transforms continuous motion and temperature data into actionable health insights for early disease detection and improved animal welfare.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the dashboard
streamlit run dashboard/app.py
```

Open browser at http://localhost:8501

---

## System Overview

### Three-Layer Intelligence Architecture

| Layer | Function | Output |
|-------|----------|--------|
| **Layer 1** | Behavioral Classification | lying, standing, walking, feeding states |
| **Layer 2** | Physiological Analysis | Temperature baselines, circadian adjustment |
| **Layer 3** | Health Intelligence | Alerts, health scores, reproductive events |

### Alert Types

| Alert | Severity | Trigger |
|-------|----------|---------|
| Fever | Critical | Temperature >39.5C + low activity |
| Heat Stress | Warning | Temperature >39.0C sustained |
| Prolonged Inactivity | Warning | Lying >6 hours continuously |
| Sensor Malfunction | Critical | Temperature out of range or stuck values |
| Estrus | Info | Temperature + activity patterns (informational only) |

### Health Score (0-100)

Calculated from four components:
- **Temperature Stability (30%)**: Deviation from baseline
- **Activity Level (25%)**: Movement patterns
- **Behavioral Consistency (25%)**: Normal behavior ratios
- **Alert Penalty (20%)**: Active alerts reduce score

| Score | Category | Action |
|-------|----------|--------|
| 80-100 | Excellent | Normal monitoring |
| 60-79 | Good | Routine monitoring |
| 40-59 | Moderate | Increased monitoring |
| 0-39 | Poor | Immediate attention |

---

## Dashboard Pages

### 1. Home
- Upload raw sensor CSV data
- View current health score
- See active alerts summary
- Monitor live sensor readings

### 2. Alerts
- View all detected alerts
- Acknowledge/resolve alerts
- Alert history and timeline

### 3. Health Analysis
- Health score trends over time
- Component breakdown (temperature, activity, behavioral)
- Baseline comparison
- Recommendations

---

## Data Upload

### Required CSV Format

```csv
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg
2025-11-13 16:10:00,38.5,-0.04,0.01,-0.88,-2.88,0.14,1.87
```

| Column | Description | Required |
|--------|-------------|----------|
| timestamp | ISO datetime | Yes |
| temperature | Body temp (C) | Yes |
| fxa | Forward acceleration (g) | Yes |
| mya | Lateral acceleration (g) | Yes |
| rza | Vertical acceleration (g) | Yes |
| sxg, lyg, dzg | Gyroscope (deg/s) | Optional |

### Upload Process

1. Open dashboard Home page
2. Set Cow ID and baseline temperature in sidebar
3. Upload CSV file
4. System automatically processes through all 3 layers
5. View results on all pages

### Data Storage

- **Sensor data**: Appended to SQLite (no overwrites, deduplication by timestamp)
- **Health scores**: New record per upload, stored in `health_scores` table
- **Alerts**: Stored in `alerts` table with status tracking
- **Database file**: `data/alert_state.db`

---

## Project Structure

```
livestock/health/
├── dashboard/              # Streamlit web interface
│   ├── pages/             # Home, Alerts, Health Analysis
│   ├── components/        # Notification panel, UI components
│   └── utils/             # Data loading, visualizations
├── src/
│   ├── data_processing/   # CSV parsing, validation, windowing
│   ├── layer1/            # Rule-based behavioral classifier
│   ├── layer1_behavior/   # Activity metrics calculation
│   ├── layer2_physiological/ # Baseline tracking, trends
│   └── health_intelligence/
│       ├── alerts/        # Immediate alert detection
│       ├── scoring/       # Health score calculation
│       ├── reproductive/  # Estrus/pregnancy detection
│       └── logging/       # Database managers
├── data/
│   ├── dashboard/         # Uploaded sensor data
│   └── alert_state.db     # SQLite database
└── tools/                 # Test data generators
```

---

## Scientific Basis

All thresholds are based on peer-reviewed research:

- **Behavioral Classification**: 20+ published cattle accelerometry studies
- **Temperature Thresholds**: Clinical veterinary standards
- **Activity Patterns**: Cattle-specific behavioral research
- **Estrus Detection**: Reproductive biology studies

### Important Limitations

**Rumination Detection: DISABLED**
- Requires >=10 Hz sampling to detect 1 Hz jaw movement
- Current system uses 1 sample/minute
- Not scientifically valid at this sampling rate
- References: Schirmann et al. 2009, Burfeind et al. 2011

**Reproductive Alerts: INFORMATIONAL ONLY**
- Estrus and pregnancy alerts require veterinary confirmation
- Not for diagnostic use

---

## Troubleshooting

### Health Scores Not Showing

1. Check database exists: `data/alert_state.db`
2. Verify upload shows "Saved to database" message
3. Check cow ID matches between uploads

### No Alerts Detected

1. Verify data contains anomalies (fever, inactivity)
2. Check temperature column has valid values
3. Review alert thresholds in `src/health_intelligence/alerts/`

### Dashboard Not Loading

```bash
# Check dependencies
pip install streamlit pandas numpy scipy plotly

# Restart dashboard
streamlit run dashboard/app.py --server.port 8501
```

### Reset Database

```bash
# Windows
del data\alert_state.db

# Linux/Mac
rm data/alert_state.db
```

Next upload will create fresh database.

---

## Production Deployment

### Single Farm (Default)
- Use SQLite database (included)
- No additional setup required

### Multi-Farm / Enterprise
- Migrate to PostgreSQL for scalability
- Add authentication layer
- Implement backup procedures
- Consider TimescaleDB for time-series optimization

### Monitoring

Watch for these metrics in dashboard:
- **Health Scores**: Should increase after each upload
- **Active Alerts**: Alerts requiring attention
- **Data Range**: Time span covered by uploads

---

## Technology Stack

- **Python**: 3.8+
- **Web Framework**: Streamlit
- **Database**: SQLite (PostgreSQL for production)
- **Data Processing**: pandas, NumPy, SciPy
- **Visualization**: Plotly

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages:
- streamlit
- pandas
- numpy
- scipy
- plotly
- scikit-learn

---

## Sensor Specifications

- **Sampling Rate**: 1 sample per minute (minimum)
- **Required Sensors**: 3-axis accelerometer, temperature
- **Optional**: 3-axis gyroscope (enhances accuracy)
- **Mounting**: Neck collar

---

**Artemis Livestock Health Monitoring System**
