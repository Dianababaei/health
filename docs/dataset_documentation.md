# Simulated Dataset Documentation

## Overview

This document describes the simulated cattle sensor datasets generated for the Artemis Health system. These datasets provide comprehensive ground truth labels for algorithm development, testing, and validation.

**Generation Date:** January 2024  
**Module:** `src/simulation/dataset_generator.py`  
**Script:** `scripts/generate_datasets.py`

---

## Dataset Time Scales

Three time scales are provided, each optimized for specific validation tasks:

### 1. Short-term Dataset (7 days)

**File:** `data/simulated/short_term_7d.csv`

- **Duration:** 10,080 data points (7 days × 24 hours × 60 minutes)
- **Purpose:** Rapid algorithm testing and development iteration
- **Scenarios:**
  - Basic behavioral patterns (lying, standing, walking, ruminating, feeding)
  - 1-2 simple health events (fever, heat stress)
  - Normal circadian patterns
- **Use Cases:**
  - Quick validation of Layer 1 classification algorithms
  - Basic alert system testing
  - Development iteration without long wait times

### 2. Medium-term Dataset (30 days)

**File:** `data/simulated/medium_term_30d.csv`

- **Duration:** 43,200 data points (30 days × 24 hours × 60 minutes)
- **Purpose:** Circadian rhythm validation and temperature pattern analysis
- **Scenarios:**
  - Complete circadian cycles
  - 1-2 estrus cycles (21-day intervals)
  - Activity trends and patterns
  - Multiple health events
- **Use Cases:**
  - Layer 2 temperature analysis validation
  - Circadian rhythm extraction testing
  - Estrus detection algorithm development

### 3. Long-term Dataset (90-180 days)

**Files:** `data/simulated/long_term_90d.csv`, `data/simulated/long_term_180d.csv`

- **Duration:** 129,600 - 259,200 data points
- **Purpose:** Reproductive cycle tracking and long-term health trends
- **Scenarios:**
  - Multiple estrus cycles (~21-day intervals)
  - Pregnancy progression (may occur)
  - Seasonal variations
  - Recovery trends
  - Multiple illness episodes
- **Use Cases:**
  - Layer 3 health intelligence validation
  - Reproductive monitoring algorithms
  - Long-term trend analysis
  - Pregnancy detection validation

---

## Data Format

### CSV Structure

All datasets follow the same CSV format with the following columns:

```csv
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg,behavioral_state,temperature_status,health_events,sensor_quality
2024-01-01T00:00:00,38.5,-0.02,0.01,-0.85,2.1,-1.5,0.8,lying,normal,none,normal
```

#### Sensor Columns (7 parameters)

| Column | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| `temperature` | Body temperature | °C | 38.0 - 39.5 |
| `fxa` | Forward-backward acceleration | g | -2.0 to 2.0 |
| `mya` | Lateral acceleration | g | -2.0 to 2.0 |
| `rza` | Vertical acceleration | g | -1.0 to 1.0 |
| `sxg` | Roll angular velocity | °/s | -50 to 50 |
| `lyg` | Pitch angular velocity | °/s | -50 to 50 |
| `dzg` | Yaw angular velocity | °/s | -50 to 50 |

#### Ground Truth Label Columns

| Column | Description | Values |
|--------|-------------|--------|
| `behavioral_state` | Current behavioral state | `lying`, `standing`, `walking`, `ruminating`, `feeding` |
| `temperature_status` | Temperature classification | `normal`, `elevated`, `fever`, `heat_stress`, `dropping` |
| `health_events` | Active health events | `none`, `estrus`, `pregnancy_indication`, `illness`, `heat_stress` |
| `sensor_quality` | Sensor data quality | `normal`, `noisy`, `malfunction` |

### Daily Aggregate Files

Each dataset includes a daily aggregate file (e.g., `short_term_7d_daily.csv`):

```csv
date,estrus_day,pregnancy_day,health_score,mean_temperature,activity_level,lying_percent,standing_percent,walking_percent,ruminating_percent,feeding_percent
2024-01-01,false,false,95.2,38.52,0.45,45.2,18.3,12.1,15.8,8.6
```

**Daily Labels:**
- `estrus_day`: Boolean indicating estrus event occurred
- `pregnancy_day`: Boolean indicating confirmed pregnancy
- `health_score`: 0-100 health score (ground truth)
- `mean_temperature`: Daily average temperature
- `activity_level`: Relative activity score
- `*_percent`: Percentage of day in each behavioral state

---

## Metadata Files

Each dataset includes a JSON metadata file (e.g., `metadata_short_term_7d.json`):

```json
{
  "dataset_name": "short_term_7d",
  "generation_timestamp": "2024-01-15T10:30:00",
  "statistics": {
    "total_data_points": 10080,
    "duration_hours": 168,
    "duration_days": 7,
    "behavioral_state_distribution": {
      "lying": 45.2,
      "standing": 18.3,
      "walking": 12.1,
      "ruminating": 15.8,
      "feeding": 8.6
    },
    "health_event_counts": {
      "none": 9800,
      "illness": 200,
      "heat_stress": 80
    }
  },
  "generation_config": {
    "seed": 42,
    "animal_id": "cow_short_001",
    "start_time": "2024-01-01T00:00:00"
  },
  "validation": {
    "is_valid": true,
    "errors": [],
    "warnings": []
  }
}
```

---

## Health Event Scenarios

### Estrus Cycles

- **Frequency:** Every ~21 days (18-24 day variation)
- **Duration:** 12-30 hours (typically 18 hours)
- **Characteristics:**
  - Temperature increase: 0.3-0.6°C
  - Activity increase: 1.2-1.5× normal
  - Labeled as `estrus` in `health_events` column
  - Gaussian-like temperature curve centered at peak

### Pregnancy Progression

- **Conception:** May occur during estrus (30% probability)
- **Confirmation:** After 30 days gestation
- **Characteristics:**
  - Slight temperature elevation: +0.15°C
  - Gradual activity reduction over gestation
  - Labeled as `pregnancy_indication` after day 30
  - No further estrus cycles after conception

### Illness/Fever Events

- **Frequency:** 0-2 events per dataset (random)
- **Duration:** 24-72 hours
- **Characteristics:**
  - Temperature increase: 1.0-2.0°C
  - Activity reduction: 40-70% of normal
  - Labeled as `illness` in `health_events`
  - Trapezoidal fever curve (ramp up, plateau, ramp down)

### Heat Stress Events

- **Frequency:** 0-3 events per long dataset
- **Duration:** 2-8 hours
- **Timing:** Typically afternoon hours (12:00-16:00)
- **Characteristics:**
  - Temperature increase: 0.5-1.2°C
  - Slight activity increase (restlessness): 1.15×
  - Labeled as `heat_stress` in both columns

### Sensor Quality Degradation

- **Frequency:** 5-10% of total time
- **Duration:** Variable (0.5 - several hours)
- **Characteristics:**
  - Noise multiplier: 2.0-4.0× normal
  - Labeled as `noisy` or `malfunction`
  - Affects all sensor readings

---

## Behavioral State Characteristics

### Lying
- **Percentage:** 40-50% of day
- **Key Signature:** Rza < -0.5g
- **Timing:** More frequent at night (22:00-06:00)

### Standing
- **Percentage:** 15-20% of day
- **Key Signature:** Rza > 0.7g, low motion

### Walking
- **Percentage:** 10-15% of day
- **Key Signature:** Rhythmic patterns (~1 Hz), forward acceleration

### Ruminating
- **Percentage:** 15-20% of day
- **Key Signature:** Mya oscillations (50 cycles/min), rhythmic

### Feeding
- **Percentage:** 8-12% of day
- **Key Signature:** Negative pitch (head down), forward movement
- **Timing:** Peaks at 06:00-10:00 and 16:00-20:00

---

## Train/Validation/Test Splits

Datasets are automatically split into three sets:

```
data/simulated/splits/
├── short_term_7d_train.csv       (70% of data)
├── short_term_7d_validation.csv  (15% of data)
└── short_term_7d_test.csv        (15% of data)
```

**Split Strategy:**
- **Temporal ordering:** Train data comes before validation, which comes before test
- **Balanced distribution:** All splits have representative behavioral states and health events
- **No data leakage:** Strict temporal separation

---

## Usage Examples

### Command Line

```bash
# Generate all datasets
python scripts/generate_datasets.py

# Generate specific dataset with seed
python scripts/generate_datasets.py --dataset short --seed 42

# Generate long-term (180 days) dataset
python scripts/generate_datasets.py --dataset long --duration 180 --output custom_dir
```

### Python API

```python
from src.simulation.dataset_generator import (
    generate_short_term_dataset,
    generate_medium_term_dataset,
    generate_long_term_dataset,
    generate_all_datasets
)

# Generate single dataset
result = generate_short_term_dataset(
    animal_id="cow_001",
    seed=42,
    output_dir="data/simulated"
)

# Generate all datasets
results = generate_all_datasets(output_dir="data/simulated", seed=42)
```

### Loading Data

```python
import pandas as pd

# Load main dataset
data = pd.read_csv('data/simulated/short_term_7d.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Load daily aggregates
daily = pd.read_csv('data/simulated/short_term_7d_daily.csv')
daily['date'] = pd.to_datetime(daily['date'])

# Load training split
train_data = pd.read_csv('data/simulated/splits/short_term_7d_train.csv')
```

---

## Dataset Validation

Generated datasets are automatically validated for:

1. **Continuity:** No gaps in timestamp sequence
2. **Value Ranges:** All sensor values within realistic ranges
3. **Label Consistency:** Health events follow logical progression
4. **State Distribution:** Behavioral states within expected percentages
5. **Temporal Patterns:** Circadian rhythms present in 30+ day datasets

Validation results are included in metadata JSON files.

---

## Data Quality Metrics

### Expected Distributions

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Lying percentage | 40-50% | Higher at night |
| Ruminating percentage | 15-20% | Critical for health |
| Temperature (healthy) | 38.0-39.0°C | Circadian ±0.4°C |
| Temperature (fever) | 39.5-41.0°C | During illness |
| Sensor quality (normal) | 90-95% | 5-10% degraded |

### Circadian Patterns

**Temperature:**
- **Peak:** 16:00 (4 PM)
- **Nadir:** 04:00 (4 AM)
- **Variation:** ~0.4°C between peak and nadir

**Activity:**
- **High:** 06:00-20:00 (daylight hours)
- **Low:** 22:00-06:00 (nighttime)
- **Feeding peaks:** 06:00-10:00, 16:00-20:00

---

## Known Limitations

1. **Single Animal:** Each dataset represents one animal (no herd interactions)
2. **Idealized Scenarios:** Health events are somewhat regularized
3. **No Environmental Factors:** Weather, season effects are simplified
4. **Perfect Sensors (Baseline):** Sensor drift and bias not included (except during degradation events)
5. **Fixed Time Step:** Always 1-minute intervals

---

## Comparison with Real Data

Generated datasets are designed to match research literature:

- **Behavioral signatures:** Based on accelerometer studies
- **State durations:** Match observed cattle time budgets
- **Temperature patterns:** Based on circadian rhythm research
- **Health events:** Aligned with veterinary literature
- **Noise levels:** Match commercial sensor specifications

However, simulated data is **cleaner and more regular** than real-world data. Use for:
- ✓ Algorithm development and testing
- ✓ Ground truth validation
- ✓ Edge case testing
- ✗ Final production validation (use real data)

---

## Citation and References

When using these datasets in research or publications, please cite:

```
Artemis Health Simulated Cattle Sensor Dataset
Generated using research-based behavioral state simulation
Module: src/simulation/dataset_generator.py
```

**Key References:**
1. Cattle behavioral state classification using accelerometers
2. Circadian rhythm patterns in dairy cattle
3. Estrus detection using body temperature and activity
4. Rumination patterns and health monitoring

---

## Support and Issues

For questions, issues, or feature requests:
- See module documentation: `src/simulation/README.md`
- Check metadata files for generation parameters
- Validate datasets using built-in validation functions

---

## Version History

- **v1.0** (January 2024): Initial release with three time scales
  - Short-term (7 days)
  - Medium-term (30 days)
  - Long-term (90-180 days)
  - Complete ground truth labels
  - Automatic train/val/test splits
