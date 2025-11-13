# Artemis Livestock Health Monitoring System

Real-time cattle health monitoring using neck-mounted sensors. Transforms continuous motion and temperature data into actionable insights for early disease detection and improved welfare.

---

## 🚀 Quick Start

**Fastest way to explore:**

```bash
streamlit run dashboard/app.py
```

**Demo data is pre-loaded!** No setup needed.

See [QUICK_START.md](QUICK_START.md) for details.

---

## 📊 System Overview

### Three-Layer Intelligence

**Layer 1 - Physical Behavior**
- Detects: lying, standing, walking, ruminating, feeding
- Tracks: activity levels, rest duration, stress patterns

**Layer 2 - Physiological Analysis**
- Temperature patterns & circadian rhythm
- Temperature-activity correlation
- Multi-day trend tracking

**Layer 3 - Health Intelligence**
- Fever, heat stress, inactivity alerts
- Estrus & pregnancy detection (initial)
- Health scoring (0-100)
- Sensor malfunction detection

---

## 🎯 Dashboard (3 Pages)

1. **🏠 Home** - Overview + metrics
2. **🚨 Alerts** - Alert management
3. **📊 Health Analysis** - Trends

---

## 📁 Applications

| App | Purpose | Command |
|-----|---------|---------|
| **Main Dashboard** | Production monitoring | `streamlit run dashboard/app.py` |
| **Simulator** | Generate test data | `streamlit run simulation_app.py` |
| **Demo Generator** | Quick demo | `python generate_demo_data.py` |

---

## 📦 Pre-Loaded Demo

**14 days of data included:**
- Days 1-2: Normal
- Days 3-4: Fever (40°C)
- Days 5-14: Recovery

**2 alerts ready:** Fever + Inactivity

---

## 📚 Documentation

- [QUICK_START.md](QUICK_START.md) - Get started in 30 seconds
- [APP_STRUCTURE.md](APP_STRUCTURE.md) - Page organization
- [SYSTEM_REVIEW.md](SYSTEM_REVIEW.md) - Algorithm verification
- [DATA_STORAGE_EXPLAINED.md](DATA_STORAGE_EXPLAINED.md) - Data flow

---

## 🛠️ Tech Stack

- Database: TimescaleDB
- Language: Python
- UI: Streamlit
- Analytics: pandas, scikit-learn
- Sensors: 3-axis accel + gyro + temp (1/min sampling)

---

## ✅ Status

**Production Ready:**
- All 15 objectives implemented
- Algorithms verified
- Demo data included
- Complete documentation

---

**Built for livestock health and welfare** 🐄
