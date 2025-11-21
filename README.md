# Artemis Livestock Health Monitoring System

Production-ready cattle health monitoring using neck-mounted sensors. Transforms continuous motion and temperature data into actionable health insights for early disease detection and improved animal welfare.

---

## 🚀 Quick Start

```bash
# Start the dashboard
streamlit run dashboard/app.py
```

Demo data is pre-loaded for immediate evaluation.

---

## 📊 System Overview

### Three-Layer Intelligence Architecture

**Layer 1 - Behavioral Classification**
- Real-time activity detection: lying, standing, walking, ruminating, feeding
- Activity metrics: motion intensity, rest duration, behavioral patterns
- Based on 20+ peer-reviewed studies (cattle accelerometry research)

**Layer 2 - Physiological Analysis**
- Temperature baseline tracking with circadian rhythm adjustment
- Multi-day trend analysis for early deviation detection
- Temperature-activity correlation analysis

**Layer 3 - Health Intelligence**
- **Critical Alerts**: Fever (>39.5°C + low motion), heat stress, prolonged inactivity
- **Health Scoring**: 0-100 composite score (temperature, activity, behavioral, alerts)
- **Reproductive Monitoring**: Estrus detection (informational alerts only)
- **Sensor Quality**: Malfunction detection and data validation

---

## 🎯 Dashboard Features

### Three Main Pages:

1. **Home** - Real-time overview, health metrics, data upload
2. **Alerts** - Alert management, history, and dismissal tracking
3. **Health Analysis** - Multi-day trends, behavioral patterns, scoring history

---

## 📁 Project Structure

```
livestock/health/
├── dashboard/              # Streamlit web interface
│   ├── pages/             # Dashboard pages (Home, Alerts, Analysis)
│   ├── components/        # Reusable UI components
│   └── utils/             # Data loading, visualization utilities
├── src/
│   ├── data_processing/   # Data ingestion, validation, windowing
│   ├── layer1/            # Behavioral classification
│   ├── layer1_behavior/   # Activity metrics calculation
│   ├── layer2_physiological/ # Baseline tracking, trend analysis
│   └── health_intelligence/
│       ├── alerts/        # Immediate alert detection
│       ├── scoring/       # Health score calculation
│       ├── reproductive/  # Estrus/pregnancy detection
│       └── logging/       # Alert & score persistence
├── data/
│   ├── dashboard/         # User-uploaded sensor data
│   └── alert_state.db     # SQLite database (alerts & health scores)
└── tools/                 # Test data generators and utilities
```

---

## 📚 Documentation

**Getting Started:**
- [QUICK_START.md](QUICK_START.md) - 30-second setup guide
- [UPLOAD_WORKFLOW.md](UPLOAD_WORKFLOW.md) - How to upload sensor data

**Production & Operations:**
- [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) - Deployment procedures and best practices
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
- [DATA_STORAGE_EXPLAINED.md](DATA_STORAGE_EXPLAINED.md) - Data flow and storage

---

## 🔬 Scientific Validation

All algorithms are based on peer-reviewed research:

- **Behavioral Classification**: Validated against 20+ published studies on cattle accelerometry
- **Temperature Thresholds**: Clinical veterinary standards (fever >39.5°C, heat stress >39.0°C)
- **Activity Patterns**: Cattle-specific thresholds from animal behavior research
- **Estrus Detection**: Physiological parameters from reproductive biology studies

**Important**: Estrus and pregnancy alerts are INFORMATIONAL ONLY and require veterinary confirmation.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.8+
- **Database**: SQLite (production: PostgreSQL recommended for multi-server)
- **Web Framework**: Streamlit
- **Data Processing**: pandas, NumPy, SciPy
- **Sensor Specifications**: 3-axis accelerometer + gyroscope + temperature (1 sample/minute)

---

## 📋 Requirements

```bash
# Install dependencies
pip install -r requirements.txt

# Key packages:
# - streamlit
# - pandas
# - numpy
# - scipy
# - scikit-learn
```

---

## ⚠️ Important Notes

### Informational Alerts
- **Estrus Detection**: Informational only - requires manual observation and veterinary consultation
- **Pregnancy Detection**: Experimental feature - not for diagnostic use

### Data Privacy
- All sensor data stored locally
- No external data transmission
- Complies with farm data management standards

### Sensor Requirements
- Sampling rate: 1 sample per minute minimum
- Required sensors: 3-axis accelerometer (Fxa, Mya, Rza), temperature
- Optional: Gyroscope (Lyg, Rzg) for enhanced accuracy

---

## ✅ Production Status

**System is production-ready** with the following validations:
- ✅ All core algorithms implemented and tested
- ✅ Scientific accuracy verified against literature
- ✅ Real-time alert detection (<2 minute latency)
- ✅ Database persistence for alerts and health scores
- ✅ Comprehensive error handling and logging
- ✅ Data validation and sensor malfunction detection
- ✅ Clean codebase with proper documentation

**Recommended for Production Deployment:**
- Single-farm operations: Use as-is with SQLite
- Multi-farm/enterprise: Migrate to PostgreSQL for scalability
- Add authentication layer if exposed over network
- Implement backup procedures for alert database

---

## 📞 Support

For technical support or questions about deployment, refer to:
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md) for deployment procedures

---

**Built for livestock health and welfare 🐄**
