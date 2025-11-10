# Feature Engineering Module

This module implements feature extraction and dataset preparation for cattle behavior classification, specifically focused on detecting ruminating and feeding behaviors from neck-mounted sensor data.

## Overview

The module consists of three main components:

1. **`behavior_features.py`**: Feature extraction functions
   - Ruminating detection features (40-60 cycles/min chewing patterns)
   - Feeding detection features (head-down position, lateral browsing)
   - Rolling window statistics
   - Motion intensity metrics
   - Temporal features

2. **`dataset_builder.py`**: Dataset preparation pipeline
   - Data loading and generation
   - Class balancing (SMOTE)
   - Train/validation/test splitting (70/15/15)
   - Feature normalization (StandardScaler)
   - Data quality validation

3. **`generate_dataset.py`**: Script to generate complete datasets

## Quick Start

### Generate Training Datasets

```bash
cd /path/to/project
python -m src.feature_engineering.generate_dataset
```

This will:
1. Generate simulated sensor data with 1000+ samples per class
2. Extract 110+ features per sample
3. Balance classes using SMOTE if needed
4. Split into train/val/test sets (70/15/15)
5. Normalize features
6. Export to `data/processed/`:
   - `training_features.pkl`
   - `validation_features.pkl`
   - `test_features.pkl`
   - `dataset_statistics.pkl`

### Load Prepared Datasets

```python
from src.feature_engineering import load_prepared_dataset

# Load training data
train_data = load_prepared_dataset('data/processed/training_features.pkl')
X_train = train_data['X']  # Feature matrix (pandas DataFrame)
y_train = train_data['y']  # Labels (pandas Series)
feature_names = train_data['feature_names']  # List of feature names
scaler = train_data['scaler']  # StandardScaler (for training set only)

# Load validation and test sets
val_data = load_prepared_dataset('data/processed/validation_features.pkl')
test_data = load_prepared_dataset('data/processed/test_features.pkl')
```

### Extract Features from Raw Data

```python
from src.feature_engineering import extract_features_from_dataframe
import pandas as pd

# Load raw sensor data
df = pd.read_csv('data/simulated/sensor_data.csv')

# Extract features
features_df = extract_features_from_dataframe(
    df,
    sampling_rate=1.0,      # 1 sample per minute
    window_minutes=10        # 10-minute windows
)
```

### Custom Dataset Building

```python
from src.feature_engineering import DatasetBuilder

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

# Build dataset from existing CSV
results = builder.build_dataset(
    data_source='data/simulated/my_data.csv',
    output_dir='data/processed',
    target_behaviors=['ruminating_lying', 'ruminating_standing', 'feeding'],
    apply_balancing=True,
    balance_method='smote'
)

# Or generate simulated data
results = builder.build_dataset(
    data_source='generate',
    output_dir='data/processed',
    target_behaviors=['ruminating_lying', 'ruminating_standing', 'feeding'],
    apply_balancing=True,
    balance_method='smote'
)
```

## Feature Categories

### Ruminating Features (15+)
- **Frequency Analysis**: Mya and Lyg dominant frequencies, spectral power in 0.67-1.0 Hz band
- **Rhythmicity**: Autocorrelation-based rhythmicity scores
- **Cross-Correlation**: Mya-Lyg signal correlation
- **Variance**: Lyg pitch variance and range

### Feeding Features (18+)
- **Head Position**: Lyg mean/median, negative ratio (head-down indicator)
- **Head Movement**: Lyg variance, range
- **Lateral Motion**: Mya variance, range, mean absolute
- **Head-Down Duration**: Maximum contiguous duration, ratio, count
- **Posture**: Rza mean, standing ratio
- **Bite Frequency**: Mya dominant frequency in 0.5-1.5 Hz band

### Rolling Window Statistics (56)
For each sensor (fxa, mya, rza, sxg, lyg, dzg, temperature):
- Mean, std, min, max, range, median, q25, q75

### Motion Intensity (9)
- Acceleration magnitude (mean, std, max)
- Gyroscope magnitude (mean, std, max)
- Combined intensity score
- Signal Vector Magnitude (SVM)

### Temporal Features (5)
- Hour, minute
- Cyclic encoding (sin/cos)
- Day/night indicator

**Total: 110+ features**

## Dataset Specifications

### Target Classes
- `ruminating_lying`: Rumination during lying posture
- `ruminating_standing`: Rumination while standing
- `feeding`: Active feeding/grazing

### Sample Requirements
- Minimum: 1000+ samples per class
- Window size: 10 minutes per sample
- Sampling rate: 1 sample/minute

### Class Balancing
- **Threshold**: Applied if imbalance ratio > 2:1
- **Method**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Alternatives**: Random undersampling, hybrid approach

### Data Splits
- **Training**: 70%
- **Validation**: 15%
- **Test**: 15%
- **Stratification**: Enabled to preserve class distributions

### Feature Normalization
- **Method**: StandardScaler (z-score normalization)
- **Fitting**: Only on training data (prevents data leakage)
- **Application**: Applied to all numeric features except binary/cyclic

## Data Quality Validation

The pipeline performs automatic validation:

1. **No Data Overlap**: Ensures train/val/test sets are disjoint
2. **Similar Distributions**: Validates class distributions are within ±5% across splits
3. **No NaN/Inf Values**: Checks all features have valid values
4. **Reasonable Ranges**: Ensures no zero-variance features

## Dependencies

```
numpy>=1.23.0
pandas>=1.5.0
scipy>=1.9.0
scikit-learn>=1.1.0
imbalanced-learn>=0.10.0
```

Install with:
```bash
pip install -r requirements.txt
```

## Documentation

For detailed documentation, see:
- **Feature Extraction Config**: `docs/feature_extraction_config.md`
- **Behavioral Signatures**: `docs/behavioral_sensor_signatures.md`
- **Simulation Engine**: `src/simulation/README.md`

## File Structure

```
src/feature_engineering/
├── __init__.py                 # Module exports
├── behavior_features.py        # Feature extraction functions
├── dataset_builder.py          # Dataset preparation pipeline
├── generate_dataset.py         # Dataset generation script
└── README.md                   # This file

data/processed/                 # Generated datasets (created by script)
├── training_features.pkl
├── validation_features.pkl
├── test_features.pkl
└── dataset_statistics.pkl
```

## Success Criteria

✓ Training dataset contains balanced classes with 1000+ samples per behavior  
✓ Feature extraction captures known behavioral signatures (40-60 cycles/min ruminating, head-down feeding)  
✓ 70/15/15 split maintains class distributions within ±5% across sets  
✓ No temporal overlap between train/validation/test sets  
✓ Feature normalization prevents data leakage (fitted only on training data)  
✓ Extracted features have reasonable distributions (no NaN/Inf, sensible ranges)  
✓ Datasets compatible with scikit-learn models  

## Next Steps

After generating datasets:

1. **Train ML Models** (Task #90):
   ```python
   from sklearn.ensemble import RandomForestClassifier
   from src.feature_engineering import load_prepared_dataset
   
   # Load data
   train_data = load_prepared_dataset('data/processed/training_features.pkl')
   X_train, y_train = train_data['X'], train_data['y']
   
   # Train model
   model = RandomForestClassifier(random_state=42)
   model.fit(X_train, y_train)
   ```

2. **Evaluate Models**:
   - Validate on validation set during training
   - Final evaluation on held-out test set

3. **Feature Importance Analysis**:
   - Identify most discriminative features
   - Verify alignment with literature-backed signatures

## Troubleshooting

**Import errors**: Make sure to run from project root or set PYTHONPATH
```bash
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

**Memory issues**: Reduce `samples_per_state` or `window_minutes` in configuration

**SMOTE errors**: Ensure imbalanced-learn is installed: `pip install imbalanced-learn`

**Feature extraction warnings**: Check input data has all required sensor columns (fxa, mya, rza, sxg, lyg, dzg, temperature)
