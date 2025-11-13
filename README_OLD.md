# Artemis Health - Livestock Health Monitoring System

A comprehensive livestock health monitoring and analysis system that uses neck-mounted sensors to track animal behavior, physiology, and health status in real-time.

## Overview

The Artemis Health system collects sensor data every minute from neck-mounted devices and processes it through three intelligent analysis layers:

1. **Physical Behavior Layer** - Recognizes posture, activity patterns, and specific behaviors (ruminating, feeding, resting)
2. **Physiological Analysis Layer** - Monitors body temperature patterns, circadian rhythms, and health trends
3. **Health Intelligence Layer** - Provides automated health scoring, early warning alerts, and predictive analytics

### Sensor Data Collected

- **Temperature (°C)** - Body temperature for fever, heat stress, and estrus detection
- **3-Axis Accelerometer (Fxa, Mya, Rza)** - Movement and posture tracking
- **3-Axis Gyroscope (Sxg, Lyg, Dzg)** - Head orientation and rotation patterns

## Project Structure

```
livestock-health-monitoring/
├── data/                   # Data storage
│   ├── raw/               # Raw sensor CSV files
│   ├── processed/         # Cleaned/normalized data
│   ├── simulated/         # Generated test data
│   └── labels/            # Ground truth labels
├── models/                # Machine learning models
│   ├── trained/           # Saved ML models
│   ├── configs/           # Model hyperparameters
│   └── evaluations/       # Performance metrics
├── logs/                  # Application logging
│   ├── alerts/            # Alert history (JSON/CSV)
│   ├── health_scores/     # Health score tracking
│   └── system/            # Application logs
├── dashboard/             # Streamlit dashboard
│   ├── pages/             # Multi-page modules
│   ├── components/        # Reusable UI components
│   └── assets/            # Static resources
├── src/                   # Source code
│   ├── layer1/            # Behavior classification
│   ├── layer2/            # Physiological analysis
│   ├── layer3/            # Health intelligence
│   └── utils/             # Shared utilities
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git exclusions
├── .env.example          # Environment configuration template
└── README.md             # This file
```

## Prerequisites

- **Python 3.9+** (recommended: Python 3.9, 3.10, or 3.11)
- **pip** (Python package manager)
- **Git** (for version control)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd livestock-health-monitoring
```

### 2. Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

Run the following command to verify all libraries are installed correctly:

```bash
python -c "import pandas, numpy, sklearn, streamlit, scipy, statsmodels; print('✓ All imports successful')"
```

### 5. Configure Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your specific configuration (database URLs, API keys, etc.)

## Development Workflow

### Activating the Environment

Always activate the virtual environment before working on the project:

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### Running the Dashboard

```bash
streamlit run dashboard/app.py
```

### Running Tests

```bash
pytest tests/
```

### Deactivating the Environment

When you're done working:

```bash
deactivate
```

## Core Dependencies

- **pandas (>=1.5.0)** - Data manipulation and analysis
- **numpy (>=1.23.0)** - Numerical computing
- **scikit-learn (>=1.1.0)** - Machine learning algorithms
- **matplotlib (>=3.6.0)** - Static data visualization
- **plotly (>=5.11.0)** - Interactive visualizations
- **streamlit (>=1.25.0)** - Dashboard framework
- **scipy (>=1.9.0)** - Scientific computing
- **statsmodels (>=0.14.0)** - Statistical modeling

## Key Features

### Layer 1: Behavioral Analysis
- Posture detection (lying, standing, walking)
- Activity level quantification
- Rest duration tracking
- Rumination and feeding pattern recognition

### Layer 2: Physiological Monitoring
- Temperature trend analysis
- Circadian rhythm tracking
- Activity-temperature correlation
- Long-term health trend assessment

### Layer 3: Health Intelligence
- **Instant Alerts:**
  - Fever detection (temp > 39.5°C with low activity)
  - Heat stress warnings
  - Prolonged inactivity alerts
- **Predictive Analytics:**
  - Estrus detection
  - Pregnancy monitoring
  - Recovery/deterioration trends
- **Health Scoring (0-100):** Comprehensive wellness metric
- **Automated Alert System:** Real-time notifications

## Data Flow

1. **Data Collection** → Raw sensor readings (CSV format)
2. **Data Processing** → Cleaning, normalization, feature extraction
3. **Layer 1 Analysis** → Behavior classification
4. **Layer 2 Analysis** → Physiological pattern detection
5. **Layer 3 Analysis** → Health scoring and alert generation
6. **Dashboard Visualization** → Real-time monitoring interface

## Contributing

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Ensure all tests pass
4. Submit a pull request with description

## Troubleshooting

### Import Errors
If you encounter import errors, ensure:
- Virtual environment is activated
- All dependencies are installed: `pip install -r requirements.txt`
- Python version is 3.9 or higher: `python --version`

### Permission Issues
On macOS/Linux, you may need to make scripts executable:
```bash
chmod +x scripts/*.sh
```

### Version Conflicts
If you experience dependency conflicts:
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

## License

[Add your license information here]

## Contact

[Add contact information or team details here]

---

**Note:** This system is designed for research and monitoring purposes. Always consult with veterinary professionals for clinical decisions.
