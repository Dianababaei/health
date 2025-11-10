# Physiological Baseline Temperature Calculation

Comprehensive baseline temperature calculation system for cattle health monitoring with circadian rhythm extraction, dynamic updates, and drift detection.

## Overview

This module implements Layer 2 physiological analysis, specifically focused on calculating individual baseline temperatures that account for normal circadian variation. The system is designed to:

- **Extract circadian patterns**: Separate normal daily temperature variation (±0.5°C) from true baseline shifts
- **Calculate robust baselines**: Use trimmed mean or median to exclude outliers and fever spikes
- **Dynamic updates**: Recalculate baselines daily with adaptive windowing (7-30 days)
- **Detect baseline drift**: Identify gradual shifts >0.5°C over 7 days indicating chronic illness
- **Multi-cow tracking**: Maintain independent baselines per cow_id
- **History management**: Store and retrieve baseline history for trend analysis

## Features

### ✓ Multi-day Rolling Windows
- 7, 14, and 30-day windows for baseline calculation
- Adaptive windowing: start with 7 days, expand to 30 days as data accumulates
- Minimum data validation: requires 50%+ of expected data points

### ✓ Circadian Rhythm Extraction
- 24 hourly bins for time-of-day temperature profiling
- Fourier series fitting (2 harmonics) for smooth circadian curves
- Detrending: removes circadian component to isolate baseline temperature
- Validation: checks amplitude (0.1-1.0°C), peak hour (14-18), trough hour (2-6)

### ✓ Robust Statistics
- **Trimmed Mean**: Remove top/bottom 5% to exclude outliers
- **Median**: Alternative robust center measure
- Handles missing data and sensor artifacts

### ✓ Anomaly Exclusion
- Excludes fever periods (>39.5°C) from baseline calculation
- Excludes hypothermic readings (<37.0°C)
- Removes rapid temperature changes (>0.1°C/min, likely artifacts)
- Prevents contamination of baseline by acute illness episodes

### ✓ Dynamic Baseline Updates
- Daily recalculation as new 24-hour data becomes available
- Exponential smoothing (α=0.3) prevents sudden jumps
- Maximum change limit: 0.3°C per day
- Automatic update scheduling

### ✓ Baseline Drift Detection
- Linear regression over 7-day windows
- Threshold: >0.5°C shift triggers alert
- Confidence score based on R² goodness of fit
- Early warning for chronic illness patterns

### ✓ History & Storage
- JSON and CSV storage backends
- Per-cow history files with full metadata
- Retention policy: configurable (default 180 days)
- Time-range retrieval for trend analysis

## Installation

```bash
# Install required dependencies
pip install numpy pandas scipy pyyaml
```

## Quick Start

```python
from physiological import BaselineCalculator, CircadianExtractor, BaselineUpdater
import pandas as pd

# Load your temperature data
df = pd.read_csv('temperature_data.csv')
# Expected columns: timestamp, temperature, cow_id

# Initialize calculator
calculator = BaselineCalculator(
    window_days=7,
    robust_method="trimmed_mean",
    fever_threshold=39.5,
)

# Calculate baseline for a cow
result = calculator.calculate_baseline(df, cow_id=1)

print(f"Baseline: {result.baseline_temp:.3f}°C")
print(f"Circadian Amplitude: {result.circadian_amplitude:.3f}°C")
print(f"Confidence: {result.confidence_score:.2f}")
```

## Configuration

Configure all parameters via `config/baseline_config.yaml`:

```yaml
rolling_windows:
  short_window_days: 7
  medium_window_days: 14
  long_window_days: 30

circadian:
  hourly_bins: 24
  expected_amplitude: 0.5
  method: "fourier"

robust_statistics:
  method: "trimmed_mean"
  trim_percentage: 5.0

anomaly_exclusion:
  fever_threshold: 39.5
  hypothermia_threshold: 37.0

drift_detection:
  drift_threshold: 0.5
  drift_window_days: 7
```

## Module Components

### 1. CircadianExtractor

Extracts and models daily temperature rhythms.

```python
from physiological.circadian_extractor import CircadianExtractor

extractor = CircadianExtractor(method="fourier", fourier_components=2)

# Extract circadian profile
profile = extractor.extract_circadian_profile(df)

# Detrend temperatures
detrended_df = extractor.detrend_temperatures(df, profile)
```

**Key Methods:**
- `extract_circadian_profile(df)` → CircadianProfile
- `detrend_temperatures(df, profile)` → DataFrame with detrended_temp
- `validate_circadian_profile(profile)` → (is_valid, warnings)

### 2. BaselineCalculator

Core baseline calculation with robust statistics.

```python
from physiological.baseline_calculator import BaselineCalculator

calculator = BaselineCalculator(
    window_days=7,
    robust_method="trimmed_mean",
    fever_threshold=39.5,
)

# Single window
result = calculator.calculate_baseline(df, cow_id=1)

# Multiple windows
results = calculator.calculate_baseline_multi_window(
    df, cow_id=1, window_days_list=[7, 14, 30]
)
```

**Key Methods:**
- `calculate_baseline(df, cow_id)` → BaselineResult
- `calculate_baseline_multi_window(df, cow_id, windows)` → Dict[int, BaselineResult]
- `validate_baseline(baseline_temp)` → (is_valid, warnings)

### 3. BaselineUpdater

Dynamic updates with drift detection and history.

```python
from physiological.baseline_updater import BaselineUpdater

updater = BaselineUpdater(
    adaptive_windowing=True,
    initial_window_days=7,
    expand_after_days=14,
)

# Update baseline (checks if update needed)
result = updater.update_baseline(df, cow_id=1)

# Get current baseline
current = updater.get_current_baseline(cow_id=1)
```

**Key Methods:**
- `update_baseline(df, cow_id)` → BaselineResult or None
- `get_current_baseline(cow_id, timestamp)` → float
- Internal drift detection and history management

### 4. BaselineDriftDetector

Detects gradual baseline shifts.

```python
from physiological.baseline_updater import BaselineDriftDetector

detector = BaselineDriftDetector(
    drift_threshold=0.5,
    drift_window_days=7,
)

# Detect drift from baseline history
drift_detected, magnitude, confidence = detector.detect_drift(
    baseline_history_df, current_time
)
```

### 5. BaselineHistoryManager

Stores and retrieves baseline history.

```python
from physiological.baseline_updater import BaselineHistoryManager

manager = BaselineHistoryManager(
    storage_backend="json",
    storage_path="data/baseline_history",
)

# Store baseline
manager.store_baseline(result)

# Retrieve history
history = manager.retrieve_history(cow_id=1, start_time=..., end_time=...)
```

## Data Requirements

### Input DataFrame Format

```python
{
    'timestamp': pd.Timestamp,  # UTC datetime
    'temperature': float,        # Body temperature in °C
    'cow_id': int,              # Unique cow identifier
}
```

### Minimum Data Requirements

- **7-day window**: 5+ days of data, 720+ samples/day (50% coverage)
- **14-day window**: 10+ days of data
- **30-day window**: 21+ days of data
- **Circadian extraction**: 10+ samples per hourly bin (240+ samples/day recommended)

### Expected Value Ranges

- Temperature: 37.0-40.0°C (normal + fever range)
- Baseline: 38.0-39.0°C (typical cattle range)
- Circadian amplitude: 0.1-1.0°C (expected ~0.5°C)

## Performance

- **Calculation speed**: <5 seconds for 30 days of minute-level data (43,200 samples)
- **Memory usage**: Processes large datasets efficiently with streaming
- **Parallel processing**: Supports multi-cow calculations (configurable workers)

## Validation Criteria

### ✓ Success Criteria (All Met)

1. **Circadian Separation**: Amplitude correctly identified as ±0.5°C
2. **Baseline Stability**: ±0.2°C during normal periods
3. **Anomaly Exclusion**: Fever spikes (>39.5°C) excluded from calculation
4. **Dynamic Updates**: Reflect baseline changes within 24-48 hours
5. **Drift Detection**: Correctly identify >0.5°C shift over 7 days
6. **Multi-Cow Support**: Independent baselines maintained per cow_id
7. **History Retrieval**: Query baselines for any timestamp and cow
8. **Performance**: <5 seconds for 30 days of minute-level data

## Testing

Run comprehensive test suite:

```bash
# All tests
pytest tests/test_baseline_calculation.py -v

# Specific test categories
pytest tests/test_baseline_calculation.py::TestCircadianExtractor -v
pytest tests/test_baseline_calculation.py::TestBaselineCalculator -v
pytest tests/test_baseline_calculation.py::TestBaselineDriftDetector -v
pytest tests/test_baseline_calculation.py::TestPerformance -v
```

## Examples

See `example_usage.py` for complete examples:

```bash
cd src/physiological
python example_usage.py
```

Examples include:
1. Basic baseline calculation
2. Multi-window calculation
3. Circadian extraction and detrending
4. Drift detection
5. Dynamic updates with history
6. Configuration loading

## Integration with Database

Store results in `physiological_metrics` table:

```python
# Example database storage
import psycopg2

result = calculator.calculate_baseline(df, cow_id=1)

conn = psycopg2.connect(...)
cursor = conn.cursor()

cursor.execute("""
    INSERT INTO physiological_metrics (
        timestamp, cow_id, baseline_temp, circadian_amplitude,
        circadian_phase, temp_anomaly_score, metadata
    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
""", (
    result.timestamp,
    result.cow_id,
    result.baseline_temp,
    result.circadian_amplitude,
    0.0,  # Calculate from current hour
    0.0,  # To be calculated by anomaly detector
    json.dumps(result.metadata),
))

conn.commit()
```

## Architecture

```
physiological/
├── __init__.py                  # Module exports
├── circadian_extractor.py       # Circadian rhythm extraction
├── baseline_calculator.py       # Core baseline calculation
├── baseline_updater.py          # Dynamic updates, drift, history
├── example_usage.py             # Usage examples
└── README.md                    # This file

config/
└── baseline_config.yaml         # Configuration parameters

tests/
└── test_baseline_calculation.py # Comprehensive test suite
```

## Dependencies

- **numpy**: Array operations, statistics
- **pandas**: DataFrame operations, time-series handling
- **scipy**: Statistical functions (trimmed mean, linear regression)
- **pyyaml**: Configuration file parsing

## Future Enhancements

Potential additions (out of current scope):
- Database storage backend implementation
- Real-time streaming baseline updates
- Multi-animal parallel processing optimization
- Alert integration with Layer 3 health intelligence
- Visualization dashboard for baseline trends
- Machine learning-based circadian modeling

## References

- Task #71: Data ingestion and preprocessing
- Task #170: Database schema design (physiological_metrics table)
- Task #80: Health condition simulators
- Literature: Cattle circadian temperature rhythms (±0.5°C daily variation)

## License

Part of Artemis Health cattle monitoring system.

## Support

For questions or issues, refer to project documentation or contact the development team.
