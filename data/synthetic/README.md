# Synthetic Dataset Generation

This directory contains synthetic sensor data for training and evaluating animal behavior models.

## Generating Datasets

To generate all datasets, run:

```bash
cd src/data
python dataset_generator.py
```

This will generate:
- `train.csv` - Training dataset (70% of balanced + transition data)
- `val.csv` - Validation dataset (15% of balanced + transition data)
- `test.csv` - Test dataset (15% of balanced + transition data)
- `multiday_1.csv` through `multiday_5.csv` - Multi-day continuous datasets
- `dataset_metadata.json` - Metadata and statistics for all datasets

## Dataset Contents

### Balanced Dataset
- 6,600 total minutes (1,100 minutes per behavior)
- Equal distribution across 6 behaviors
- Random samples with circadian patterns

### Transition Dataset
- Focus on realistic behavior sequences
- 150+ transition examples (15 per common pair)
- Smooth transitions between behaviors

### Multi-Day Datasets
- 5 datasets spanning 7-14 days each
- Realistic daily activity schedules
- Circadian temperature and activity patterns
- Day-to-day variation

## CSV Format

All CSV files follow this schema:

```csv
timestamp,temp,Fxa,Mya,Rza,Sxg,Lyg,Dzg,behavior_label
2024-01-01 00:00:00,38.3,-0.2,0.1,-0.5,15.2,-8.3,5.1,lying
```

**Columns:**
- `timestamp` - ISO 8601 datetime
- `temp` - Body temperature (°C)
- `Fxa` - Forward-backward acceleration (g)
- `Mya` - Side-to-side acceleration (g)
- `Rza` - Up-down acceleration (g)
- `Sxg` - Roll angular velocity (°/s)
- `Lyg` - Pitch angular velocity (°/s)
- `Dzg` - Yaw angular velocity (°/s)
- `behavior_label` - Behavior class (lying, standing, walking, ruminating, feeding, stress)

## Behavior Classes

1. **lying** - Resting position, low activity
2. **standing** - Upright stationary position
3. **walking** - Forward locomotion
4. **ruminating** - Chewing cud, characteristic jaw movements
5. **feeding** - Eating, head-down position
6. **stress** - Elevated temperature and erratic movements

## Validation

The generator performs automatic validation:
- No NaN or missing values
- Train/val/test splits are 70/15/15 (±2%)
- All behaviors represented in each split
- Sensor values within expected ranges
- No duplicate timestamps

## Metadata

See `dataset_metadata.json` for detailed statistics including:
- Sample counts per behavior
- Date ranges
- Behavior distributions
- Split statistics
