# Feature Extraction Configuration and Documentation

**Document Version:** 1.0  
**Last Updated:** 2025-01-08  
**Purpose:** Document feature engineering pipeline for cattle behavior classification (ruminating and feeding detection)

---

## Table of Contents

1. [Overview](#overview)
2. [Feature Extraction Pipeline](#feature-extraction-pipeline)
3. [Ruminating Features](#ruminating-features)
4. [Feeding Features](#feeding-features)
5. [Rolling Window Statistics](#rolling-window-statistics)
6. [Motion Intensity Metrics](#motion-intensity-metrics)
7. [Temporal Features](#temporal-features)
8. [Dataset Preparation](#dataset-preparation)
9. [Data Quality Validation](#data-quality-validation)
10. [Usage Examples](#usage-examples)
11. [Configuration Parameters](#configuration-parameters)

---

## Overview

This document describes the feature engineering pipeline for detecting ruminating and feeding behaviors in cattle using neck-mounted accelerometer and gyroscope sensors. The pipeline extracts time-domain and frequency-domain features based on literature-backed behavioral signatures.

### Key Behavioral Signatures

**Ruminating (40-60 cycles/min)**
- Mya (lateral acceleration): Rhythmic chewing oscillations at 0.67-1.0 Hz
- Lyg (pitch gyroscope): Synchronized head bobbing at chewing frequency
- Can occur during lying (Rza < -0.5g) or standing (Rza > 0.7g)

**Feeding (Head-down grazing/eating)**
- Lyg (pitch gyroscope): Sustained negative values indicating head-down position
- Mya (lateral acceleration): Side-to-side browsing movements
- Rza (vertical acceleration): Standing posture (> 0.7g)
- Bite frequency: 30-90 bites/min (0.5-1.5 Hz), more variable than ruminating

---

## Feature Extraction Pipeline

The feature extraction process operates on rolling windows of sensor data:

1. **Window Configuration**: 10-minute sliding windows (configurable 5-10 minutes)
2. **Sampling Rate**: 1 sample per minute (matches operational deployment)
3. **Feature Categories**:
   - Current sensor values (7 features)
   - Ruminating-specific features (15+ features)
   - Feeding-specific features (18+ features)
   - Rolling window statistics (56 features)
   - Motion intensity metrics (9 features)
   - Temporal features (5 features)

Total: **110+ features per window**

---

## Ruminating Features

### Frequency Analysis

**Mya Chewing Pattern Detection**
- `mya_dominant_freq`: Dominant frequency in Mya signal (target: 0.67-1.0 Hz)
- `mya_dominant_power`: Power at dominant frequency
- `mya_spectral_energy`: Total spectral energy in Mya signal
- `mya_freq_in_target_range`: Binary indicator if frequency in 40-60 cycles/min range
- `mya_target_band_energy`: Energy specifically in 0.67-1.0 Hz band

**Lyg Head Bobbing Pattern**
- `lyg_dominant_freq`: Dominant frequency in Lyg signal
- `lyg_dominant_power`: Power at dominant frequency
- `lyg_spectral_energy`: Total spectral energy
- `lyg_freq_in_target_range`: Binary indicator for target range
- `lyg_target_band_energy`: Energy in chewing frequency band

### Variance and Rhythmicity

- `ruminating_lyg_variance`: Variance of pitch angular velocity
- `ruminating_lyg_std`: Standard deviation of Lyg
- `ruminating_lyg_range`: Range (max - min) of Lyg values
- `ruminating_mya_rhythmicity`: Rhythmicity score from autocorrelation (0-1)
- `ruminating_lyg_rhythmicity`: Rhythmicity score for Lyg signal

### Cross-Correlation

- `ruminating_mya_lyg_xcorr`: Cross-correlation between Mya and Lyg at zero lag
- `ruminating_mya_lyg_corr`: Pearson correlation coefficient between Mya and Lyg

### Spectral Power

- `ruminating_mya_spectral_power`: Power in 0.67-1.0 Hz band for Mya
- `ruminating_lyg_spectral_power`: Power in 0.67-1.0 Hz band for Lyg
- `ruminating_combined_spectral_power`: Sum of Mya and Lyg spectral power

**Implementation Note**: FFT-based frequency analysis with proper DC component removal and normalization.

---

## Feeding Features

### Head-Down Position Indicators

**Lyg Pitch Angle Analysis**
- `feeding_lyg_mean`: Mean pitch angular velocity (negative = head down)
- `feeding_lyg_median`: Median Lyg value
- `feeding_lyg_negative_ratio`: Proportion of samples with Lyg < 0
- `feeding_lyg_mean_negative`: Mean of negative Lyg values only

### Head Movement Variability

- `feeding_lyg_variance`: Variance of pitch movements during feeding
- `feeding_lyg_std`: Standard deviation of Lyg
- `feeding_lyg_range`: Range of pitch movements

### Lateral Browsing Motion

**Mya Side-to-Side Analysis**
- `feeding_mya_variance`: Variance of lateral acceleration
- `feeding_mya_std`: Standard deviation of Mya
- `feeding_mya_range`: Range of lateral movements
- `feeding_mya_mean_abs`: Mean absolute lateral acceleration

### Sustained Head-Down Duration

- `feeding_head_down_duration`: Maximum contiguous duration with Lyg < -10°/s
- `feeding_head_down_ratio`: Proportion of window with head down
- `feeding_head_down_count`: Number of distinct head-down periods

### Posture Context

- `feeding_rza_mean`: Mean vertical acceleration (posture indicator)
- `feeding_standing_ratio`: Proportion of window in standing posture (Rza > 0.7g)

### Bite Frequency Estimation

**Mya Bite Pattern (0.5-1.5 Hz range)**
- `feeding_mya_bite_dominant_freq`: Dominant bite frequency
- `feeding_mya_bite_dominant_power`: Power at bite frequency
- `feeding_mya_bite_spectral_energy`: Total spectral energy
- `feeding_mya_bite_freq_in_target_range`: Binary indicator for bite frequency range
- `feeding_mya_bite_target_band_energy`: Energy in 0.5-1.5 Hz band

---

## Rolling Window Statistics

Computed for each sensor axis over the feature window:

**Sensors**: fxa, mya, rza, sxg, lyg, dzg, temperature

**Statistics per sensor**:
- `rolling_{sensor}_mean`: Mean value
- `rolling_{sensor}_std`: Standard deviation
- `rolling_{sensor}_min`: Minimum value
- `rolling_{sensor}_max`: Maximum value
- `rolling_{sensor}_range`: Range (max - min)
- `rolling_{sensor}_median`: Median value
- `rolling_{sensor}_q25`: 25th percentile
- `rolling_{sensor}_q75`: 75th percentile

**Total**: 8 statistics × 7 sensors = **56 features**

---

## Motion Intensity Metrics

### Overall Acceleration Magnitude

- `motion_accel_magnitude_mean`: Mean of √(fxa² + mya² + rza²)
- `motion_accel_magnitude_std`: Standard deviation of acceleration magnitude
- `motion_accel_magnitude_max`: Maximum acceleration magnitude

### Overall Gyroscope Magnitude

- `motion_gyro_magnitude_mean`: Mean of √(sxg² + lyg² + dzg²)
- `motion_gyro_magnitude_std`: Standard deviation of gyroscope magnitude
- `motion_gyro_magnitude_max`: Maximum gyroscope magnitude

### Combined Metrics

- `motion_intensity_score`: Combined score from accel std + scaled gyro std
- `motion_svm`: Signal Vector Magnitude (sum of absolute accelerations)

---

## Temporal Features

### Time of Day

- `temporal_hour`: Hour of day (0-23)
- `temporal_minute`: Minute of hour (0-59)
- `temporal_hour_sin`: sin(2π × hour/24) for cyclic encoding
- `temporal_hour_cos`: cos(2π × hour/24) for cyclic encoding
- `temporal_is_daytime`: Binary indicator (1 if 6 AM - 6 PM, 0 otherwise)

### Window Metadata

- `temporal_window_size`: Number of samples in current window

**Rationale**: Cattle behaviors show strong circadian patterns (e.g., increased rumination at night, feeding at dawn/dusk).

---

## Dataset Preparation

### Data Generation

**Source**: Simulated sensor data from Task #70/#80 (simulation engine with health condition variations)

**Target Classes**:
- `ruminating_lying`: Rumination while in lying posture
- `ruminating_standing`: Rumination while standing
- `feeding`: Active feeding/grazing behavior

**Sample Requirements**:
- Minimum 1000+ samples per class (verified)
- 10-minute windows per sample
- Realistic sensor signatures matching literature

### Class Balancing

**Strategy**: SMOTE (Synthetic Minority Over-sampling Technique)

**Threshold**: Applied if class imbalance ratio > 2:1

**Implementation**:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

**Alternative methods**: 
- `undersample`: Random undersampling of majority class
- `hybrid`: SMOTE + undersampling combination

### Train/Validation/Test Split

**Ratios**: 70% / 15% / 15%

**Stratification**: Enabled to preserve class distributions across splits

**Implementation**:
```python
# First split: train vs (val + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
# Second split: val vs test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

**Validation**: Class distributions must be within ±5% across splits

### Feature Normalization

**Method**: StandardScaler (z-score normalization)

**Data Leakage Prevention**: Scaler fitted ONLY on training data

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)  # Fit on training data only
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)
```

**Excluded from normalization**:
- Binary features (e.g., `temporal_is_daytime`, `*_in_target_range`)
- Already-normalized features (e.g., cyclic sin/cos encodings)

---

## Data Quality Validation

### Validation Checks

1. **No Data Overlap**: Train/val/test sets have no common samples
2. **Similar Distributions**: Class distributions within ±5% tolerance across splits
3. **No NaN/Inf Values**: All features have valid numerical values
4. **Reasonable Ranges**: No features with zero variance

### Temporal Leakage Prevention

- Each behavioral sample is independently generated
- No time-series continuity between samples
- Samples from different time periods to avoid temporal autocorrelation
- Window-based feature extraction ensures independence

### Feature Distribution Analysis

Generated statistics include:
- Per-feature mean, std, min, max across splits
- Class distribution percentages per split
- Imbalance ratios before/after balancing
- Sample counts for train/val/test sets

---

## Usage Examples

### Generate Complete Dataset

```python
from src.feature_engineering.dataset_builder import DatasetBuilder

# Initialize builder
builder = DatasetBuilder(
    sampling_rate=1.0,
    window_minutes=10,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    balance_threshold=2.0,
    random_seed=42
)

# Build dataset (generates simulated data)
results = builder.build_dataset(
    data_source='generate',
    output_dir='data/processed',
    target_behaviors=['ruminating_lying', 'ruminating_standing', 'feeding'],
    apply_balancing=True,
    balance_method='smote'
)
```

### Load Prepared Dataset

```python
from src.feature_engineering.dataset_builder import load_prepared_dataset

# Load training set
train_data = load_prepared_dataset('data/processed/training_features.pkl')
X_train = train_data['X']
y_train = train_data['y']
feature_names = train_data['feature_names']
scaler = train_data['scaler']

# Load validation/test sets
val_data = load_prepared_dataset('data/processed/validation_features.pkl')
test_data = load_prepared_dataset('data/processed/test_features.pkl')
```

### Extract Features from Custom Data

```python
from src.feature_engineering.behavior_features import extract_features_from_dataframe
import pandas as pd

# Load raw sensor data
df = pd.read_csv('data/simulated/sensor_data.csv')

# Extract features
features_df = extract_features_from_dataframe(
    df,
    sampling_rate=1.0,
    window_minutes=10
)
```

---

## Configuration Parameters

### Feature Extraction

| Parameter | Default | Description | Valid Range |
|-----------|---------|-------------|-------------|
| `sampling_rate` | 1.0 | Samples per minute | 0.1 - 60.0 |
| `window_minutes` | 10 | Feature window size (minutes) | 5 - 30 |

### Dataset Preparation

| Parameter | Default | Description | Valid Range |
|-----------|---------|-------------|-------------|
| `train_ratio` | 0.7 | Training set proportion | 0.5 - 0.8 |
| `val_ratio` | 0.15 | Validation set proportion | 0.1 - 0.2 |
| `test_ratio` | 0.15 | Test set proportion | 0.1 - 0.2 |
| `balance_threshold` | 2.0 | Max imbalance ratio before balancing | 1.5 - 5.0 |
| `random_seed` | 42 | Random seed for reproducibility | Any integer |

### Frequency Targets

| Behavior | Target Frequency | Hz Range | Cycles/Min |
|----------|------------------|----------|------------|
| Ruminating | 0.67 - 1.0 Hz | Chewing | 40 - 60 |
| Feeding | 0.5 - 1.5 Hz | Biting | 30 - 90 |

---

## Output Files

### Dataset Files (Pickle Format)

**`data/processed/training_features.pkl`**
- X: Training feature matrix (pandas DataFrame)
- y: Training labels (pandas Series)
- feature_names: List of feature column names
- class_names: List of unique class labels
- scaler: Fitted StandardScaler object
- metadata: Dict with creation date, n_samples, n_features, etc.

**`data/processed/validation_features.pkl`**
- Same structure as training, scaler=None (use training scaler)

**`data/processed/test_features.pkl`**
- Same structure as training, scaler=None (use training scaler)

**`data/processed/dataset_statistics.pkl`**
- original_balance: Class distribution before balancing
- balanced: Class distribution after balancing
- validation: Data quality validation results

---

## Literature References

1. **Ruminating Detection**:
   - Schirmann et al. (2009): Rumination chewing frequency 40-60 cycles/min
   - Borchers et al. (2016): FFT analysis achieves 85-92% accuracy
   - Burfeind et al. (2011): Rumination patterns and health monitoring

2. **Feeding Detection**:
   - Umemura et al. (2009): Lyg pitch angle as primary feeding indicator
   - Gregorini et al. (2009): Grazing bite frequency 30-90 bites/min
   - Arcidiacono et al. (2017): Multi-sensor feeding classification

3. **Feature Engineering**:
   - González et al. (2008): Standard deviation for activity detection
   - Riaboff et al. (2022): Multi-axis variance for anomaly detection
   - Barker et al. (2018): Multi-sensor fusion improvements

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-08 | Initial documentation for feature extraction pipeline |

---

## Contact and Support

For questions or issues with the feature engineering pipeline:
- Review code in `src/feature_engineering/`
- Check simulation documentation in `docs/simulation_implementation.md`
- Refer to behavioral signatures in `docs/behavioral_sensor_signatures.md`

---

**END OF DOCUMENT**
