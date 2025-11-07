# Quick Start Guide - Synthetic Dataset Generation

## TL;DR

```bash
# Validate implementation
python validate_implementation.py

# Generate all datasets
./generate_datasets.sh  # Unix/Linux/macOS
# OR
generate_datasets.bat   # Windows

# Load and use
python
>>> import pandas as pd
>>> train = pd.read_csv('data/synthetic/train.csv')
>>> print(len(train))  # ~11,000 samples
```

## What Gets Generated

| File | Size | Description |
|------|------|-------------|
| `train.csv` | ~11K rows | Training set (70%) |
| `val.csv` | ~2.4K rows | Validation set (15%) |
| `test.csv` | ~2.4K rows | Test set (15%) |
| `multiday_1.csv` | ~10K rows | 7-day continuous data |
| `multiday_2.csv` | ~14K rows | 10-day continuous data |
| `multiday_3.csv` | ~20K rows | 14-day continuous data |
| `multiday_4.csv` | ~10K rows | 7-day continuous data |
| `multiday_5.csv` | ~14K rows | 10-day continuous data |
| `dataset_metadata.json` | - | Statistics and metadata |

## Data Format

```csv
timestamp,temp,Fxa,Mya,Rza,Sxg,Lyg,Dzg,behavior_label
2024-01-01 00:00:00,38.3,-0.2,0.1,-0.5,15.2,-8.3,5.1,lying
```

## Behaviors

1. **lying** - Resting, low activity
2. **standing** - Stationary upright
3. **walking** - Active locomotion
4. **ruminating** - Chewing cud
5. **feeding** - Eating
6. **stress** - Elevated activity/temperature

## Key Features

✅ **6,600+ minutes** of balanced data  
✅ **Circadian rhythms** (temperature varies by time of day)  
✅ **Realistic transitions** (smooth sensor changes)  
✅ **Daily patterns** (activity schedules vary by hour)  
✅ **Multi-day sequences** (7-14 day continuous datasets)  
✅ **Stratified splits** (equal behavior distribution)  

## Quick Load Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training data
df = pd.read_csv('data/synthetic/train.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Check behavior distribution
print(df['behavior_label'].value_counts())

# Plot temperature by behavior
df.boxplot(column='temp', by='behavior_label', figsize=(10, 6))
plt.ylabel('Temperature (°C)')
plt.xlabel('Behavior')
plt.title('Temperature Distribution by Behavior')
plt.suptitle('')  # Remove default title
plt.show()

# Plot circadian pattern (multiday dataset)
multiday = pd.read_csv('data/synthetic/multiday_1.csv')
multiday['timestamp'] = pd.to_datetime(multiday['timestamp'])
multiday['hour'] = multiday['timestamp'].dt.hour

# Average temperature by hour
hourly_temp = multiday.groupby('hour')['temp'].mean()
plt.plot(hourly_temp.index, hourly_temp.values)
plt.xlabel('Hour of Day')
plt.ylabel('Average Temperature (°C)')
plt.title('Circadian Temperature Pattern')
plt.grid(True)
plt.show()
```

## Common Use Cases

### 1. Train Behavior Classifier
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Load data
train = pd.read_csv('data/synthetic/train.csv')

# Features and labels
X = train[['temp', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg']]
y = train['behavior_label']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

### 2. Analyze Transitions
```python
# Load multi-day data
df = pd.read_csv('data/synthetic/multiday_1.csv')

# Find behavior changes
df['prev_behavior'] = df['behavior_label'].shift(1)
transitions = df[df['behavior_label'] != df['prev_behavior']]

# Count transition types
transition_counts = transitions.groupby(['prev_behavior', 'behavior_label']).size()
print(transition_counts.sort_values(ascending=False))
```

### 3. Study Circadian Patterns
```python
import seaborn as sns

# Load multi-day data
df = pd.read_csv('data/synthetic/multiday_1.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour

# Heatmap of behavior by hour
behavior_by_hour = pd.crosstab(df['hour'], df['behavior_label'], normalize='index')
sns.heatmap(behavior_by_hour, cmap='YlOrRd', annot=True, fmt='.2f')
plt.title('Behavior Distribution by Hour of Day')
plt.ylabel('Hour')
plt.xlabel('Behavior')
plt.show()
```

## Validation Checks

The generator automatically validates:

✅ No missing values  
✅ Correct column names  
✅ 70/15/15 split ratios (±2%)  
✅ All behaviors in each split  
✅ Temperature: 37-41°C  
✅ Acceleration: ±3g  
✅ Gyroscope: ±100°/s  

## Generation Time

- **Total**: ~2-3 minutes
- **Balanced dataset**: ~10 seconds
- **Transition dataset**: ~10 seconds
- **Multi-day datasets**: ~15-30 seconds each

## Need Help?

- Full documentation: `DATASET_GENERATION.md`
- Dataset details: `README.md`
- Test implementation: `python validate_implementation.py`
- Run tests: `python tests/test_data_generator.py`

## Customization

Edit `src/data/dataset_generator.py`:

```python
# Change balanced dataset size
balanced_df = generate_balanced_dataset(
    generator,
    minutes_per_behavior=2000,  # Default: 1100
    start_date=datetime(2024, 1, 1),
)

# Change number of transitions
transition_df = generate_transition_dataset(
    generator,
    transitions_per_pair=30,  # Default: 15
    start_date=datetime(2024, 2, 1),
)

# Add more multi-day datasets
multiday_configs = [
    (1, 7, datetime(2024, 3, 1)),   # 7 days
    (2, 14, datetime(2024, 4, 1)),  # 14 days
    # Add more...
]
```
