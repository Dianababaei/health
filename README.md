# Artemis Health - Animal Health Monitoring System

An intelligent animal health monitoring system that uses neck-mounted sensors to track real-time health metrics and behavior patterns. The system analyzes temperature, acceleration, and gyroscopic data to provide automated health assessments, early warnings, and behavioral insights.

## Overview

The Artemis Health system processes sensor data collected **every minute** from neck-mounted devices to monitor:

- **Physical Behavior**: Posture detection, activity levels, movement patterns
- **Physiological Analysis**: Temperature monitoring, circadian rhythm tracking
- **Health Intelligence**: Automated alerts for fever, heat stress, estrus detection, and health scoring

For detailed information about the sensor data and analysis layers, see [description.md](description.md).

---

## Environment Setup

### Prerequisites

**Python Version**: This project requires **Python 3.8 or higher** for compatibility with modern machine learning libraries.

To check your Python version:
```bash
python --version
# or
python3 --version
```

If you need to install or upgrade Python, visit [python.org](https://www.python.org/downloads/).

---

### Option 1: Setup with Python venv (Recommended)

Python's built-in `venv` module provides a lightweight virtual environment solution.

#### 1. Create Virtual Environment

```bash
# Create a virtual environment in the .venv directory
python -m venv .venv
```

#### 2. Activate Virtual Environment

**On Linux/macOS:**
```bash
source .venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
.venv\Scripts\activate.bat
```

**On Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

> **Note**: If you encounter a script execution error on Windows PowerShell, you may need to enable script execution:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 3. Install Dependencies

```bash
# Upgrade pip to the latest version
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

#### 4. Verify Installation

```bash
# List all installed packages
pip list

# Verify package versions
pip show pandas numpy scikit-learn streamlit
```

#### 5. Deactivate (when done)

```bash
deactivate
```

---

### Option 2: Setup with Conda

Conda provides robust package management and environment isolation, especially useful for complex scientific computing dependencies.

#### 1. Create Conda Environment

```bash
# Create a new conda environment named 'artemis-health' with Python 3.8
conda create -n artemis-health python=3.8
```

#### 2. Activate Conda Environment

```bash
conda activate artemis-health
```

#### 3. Install Dependencies

```bash
# Upgrade pip within the conda environment
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

Alternatively, you can install some packages via conda for better optimization:

```bash
# Install core scientific packages via conda
conda install pandas numpy scikit-learn scipy matplotlib seaborn

# Install remaining packages via pip
pip install streamlit plotly xgboost pytest pytest-cov joblib
```

#### 4. Verify Installation

```bash
# List all packages in the environment
conda list

# Or use pip list
pip list
```

#### 5. Deactivate (when done)

```bash
conda deactivate
```

---

## Testing the Installation

After setting up your environment and installing dependencies, verify that all critical libraries can be imported successfully.

### Quick Import Test

```bash
python -c "import pandas, numpy, sklearn, streamlit, scipy, plotly, joblib, xgboost; print('All imports successful!')"
```

### Detailed Import Test

Create a test script or run in Python shell:

```python
# Test core data processing libraries
import pandas as pd
import numpy as np
print(f"✓ pandas {pd.__version__}")
print(f"✓ numpy {np.__version__}")

# Test ML libraries
import sklearn
import scipy
import xgboost as xgb
print(f"✓ scikit-learn {sklearn.__version__}")
print(f"✓ scipy {scipy.__version__}")
print(f"✓ xgboost {xgb.__version__}")

# Test visualization libraries
import matplotlib
import seaborn as sns
import plotly
print(f"✓ matplotlib {matplotlib.__version__}")
print(f"✓ seaborn {sns.__version__}")
print(f"✓ plotly {plotly.__version__}")

# Test dashboard library
import streamlit as st
print(f"✓ streamlit {st.__version__}")

# Test model persistence
import joblib
print(f"✓ joblib {joblib.__version__}")

# Test testing framework
import pytest
print(f"✓ pytest {pytest.__version__}")

print("\n✅ All critical imports successful!")
```

### Expected Output

You should see version numbers for all libraries without any import errors. If you encounter errors, ensure:

1. The virtual environment is activated
2. All packages were installed without errors
3. You're using Python 3.8 or higher

---

## Dependency Management

### Core Dependencies

The project uses the following key libraries:

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, scipy
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard**: streamlit
- **Testing**: pytest, pytest-cov
- **Model Persistence**: joblib

### Version Pinning

All dependencies in `requirements.txt` are pinned to specific major.minor versions (e.g., `>=1.3.0,<2.0.0`) to ensure:

- **Reproducibility**: Team members get consistent environments
- **Stability**: Avoid breaking changes from major version updates
- **Compatibility**: Ensure all libraries work together

### Updating Dependencies

To update a specific package:

```bash
pip install --upgrade package-name
```

To update all packages (use with caution):

```bash
pip install --upgrade -r requirements.txt
```

After updating, regenerate requirements with exact versions:

```bash
pip freeze > requirements-lock.txt
```

---

## Troubleshooting

### Common Issues

**Issue**: `pip: command not found`
- **Solution**: Ensure Python is properly installed and added to PATH

**Issue**: Permission denied when installing packages
- **Solution**: Use a virtual environment (don't use `sudo pip`)

**Issue**: Package installation fails with compiler errors
- **Solution**: Install build tools:
  - **Linux**: `sudo apt-get install python3-dev build-essential`
  - **macOS**: `xcode-select --install`
  - **Windows**: Install Visual C++ Build Tools

**Issue**: ImportError even after installation
- **Solution**: Verify you're in the correct virtual environment with `which python` (Linux/macOS) or `where python` (Windows)

### Getting Help

If you encounter issues:

1. Check that your Python version is >=3.8
2. Ensure the virtual environment is activated
3. Try creating a fresh virtual environment
4. Check for conflicting system-wide Python packages

---

## Generating Synthetic Datasets

The project includes a synthetic data generator for training and testing machine learning models.

### Quick Start

**Linux/macOS:**
```bash
./generate_datasets.sh
```

**Windows:**
```cmd
generate_datasets.bat
```

**Manual:**
```bash
cd src/data
python dataset_generator.py
```

### Generated Files

The generator creates:
- `data/synthetic/train.csv` - Training set (70%)
- `data/synthetic/val.csv` - Validation set (15%)
- `data/synthetic/test.csv` - Test set (15%)
- `data/synthetic/multiday_1.csv` through `multiday_5.csv` - Multi-day datasets
- `data/synthetic/dataset_metadata.json` - Dataset documentation

### Dataset Features

- **6,600+ minutes** of balanced behavior data (1,100+ per behavior)
- **6 behavior classes**: lying, standing, walking, ruminating, feeding, stress
- **Realistic transitions** between behaviors
- **Circadian temperature patterns** (lower at night, higher during day)
- **Multi-day sequences** with daily activity schedules (7-14 days each)
- **Stratified splits** maintaining behavior distribution

For detailed information, see [data/synthetic/README.md](data/synthetic/README.md).

---

## Next Steps

Once your environment is set up:

1. Explore the sensor data description in [description.md](description.md)
2. Generate synthetic datasets for model development
3. Review the configuration management system
4. Begin implementing the data processing pipeline
5. Develop the analysis layers (behavior, physiology, health intelligence)

---

## Project Structure

```
artemis-health/
├── .venv/                  # Virtual environment (not in version control)
├── config/                 # Configuration files
├── data/
│   └── synthetic/         # Generated synthetic datasets
├── src/
│   ├── data/              # Data generation and processing
│   │   ├── synthetic_generator.py    # Core data generator
│   │   └── dataset_generator.py      # Dataset creation script
│   ├── layer1/            # Physical behavior layer
│   ├── layer2/            # Physiological analysis layer
│   ├── layer3/            # Health intelligence layer
│   └── utils/             # Utility modules
├── tests/                 # Test files
├── notebooks/             # Jupyter notebooks for analysis
├── description.md         # Project and sensor data overview
├── requirements.txt       # Python dependencies
├── generate_datasets.sh   # Dataset generation script (Unix)
├── generate_datasets.bat  # Dataset generation script (Windows)
└── README.md             # This file
```

---

## License

[Add license information here]

## Contributors

[Add contributor information here]
