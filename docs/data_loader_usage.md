# Data Loader Module Usage Guide

## Overview

The `data_loader` module provides functions to load sensor data from CSV files for the Artemis Health Monitoring System. It handles neck-mounted sensor measurements including temperature and 6 motion sensors (acceleration and gyroscope data).

## CSV File Format

### Required Columns

CSV files must contain exactly 8 columns in any order:

| Column Name  | Data Type | Description                                      |
|--------------|-----------|--------------------------------------------------|
| `timestamp`  | DateTime  | Measurement timestamp (multiple formats supported)|
| `temperature`| Float     | Body temperature in °C                           |
| `Fxa`        | Float     | X-axis acceleration (forward-backward)           |
| `Mya`        | Float     | Y-axis acceleration (side-to-side)               |
| `Rza`        | Float     | Z-axis acceleration (up-down)                    |
| `Sxg`        | Float     | X-axis angular velocity (roll)                   |
| `Lyg`        | Float     | Y-axis angular velocity (pitch)                  |
| `Dzg`        | Float     | Z-axis angular velocity (yaw)                    |

### Example CSV

```csv
timestamp,temperature,Fxa,Mya,Rza,Sxg,Lyg,Dzg
2024-01-15 10:00:00,38.5,1.2,0.5,2.1,0.1,0.4,1.0
2024-01-15 10:01:00,38.6,1.3,0.6,2.2,0.2,0.5,1.1
2024-01-15 10:02:00,38.7,1.4,0.7,2.3,0.3,0.6,1.2
```

## Supported Timestamp Formats

The module automatically detects and parses multiple timestamp formats:

1. **Standard format**: `YYYY-MM-DD HH:MM:SS`
   - Example: `2024-01-15 10:00:00`

2. **ISO 8601**: `YYYY-MM-DDTHH:MM:SS`
   - Example: `2024-01-15T10:00:00`
   - With timezone: `2024-01-15T10:00:00Z`
   - With microseconds: `2024-01-15T10:00:00.123456`

3. **Unix epoch (seconds)**: Numeric timestamp
   - Example: `1705315200`

4. **Unix epoch (milliseconds)**: Numeric timestamp
   - Example: `1705315200000`

5. **Alternative formats**:
   - `DD/MM/YYYY HH:MM:SS`
   - `MM/DD/YYYY HH:MM:SS`
   - `YYYY/MM/DD HH:MM:SS`

## Quick Start

### Basic Usage

```python
from src.data_loader import read_sensor_csv, read_sensor_directory

# Load a single CSV file
df = read_sensor_csv('data/sensor_data.csv')

# Load all CSV files from a directory
df = read_sensor_directory('data/sensor_readings/')
```

## Functions

### read_sensor_csv()

Load a single CSV file with sensor data.

**Signature:**
```python
read_sensor_csv(filepath: Union[str, Path], chunksize: int = None) -> pd.DataFrame
```

**Parameters:**
- `filepath`: Path to the CSV file (string or Path object)
- `chunksize`: Optional parameter for reading large files in chunks (not yet implemented)

**Returns:**
- `pd.DataFrame` with:
  - `DatetimeIndex` set from timestamp column
  - 7 numeric columns for sensor measurements
  - Sorted chronologically

**Raises:**
- `FileNotFoundError`: If the file does not exist
- `ValueError`: If required columns are missing or timestamps cannot be parsed
- `pd.errors.EmptyDataError`: If the file is empty

**Example:**
```python
from src.data_loader import read_sensor_csv

# Load a CSV file
df = read_sensor_csv('data/animal_A12345_20240115.csv')

# Display basic information
print(f"Records loaded: {len(df)}")
print(f"Time range: {df.index[0]} to {df.index[-1]}")
print(f"Columns: {list(df.columns)}")

# Access sensor data
avg_temp = df['temperature'].mean()
print(f"Average temperature: {avg_temp:.2f}°C")
```

### read_sensor_directory()

Load and concatenate all CSV files from a directory.

**Signature:**
```python
read_sensor_directory(dirpath: Union[str, Path], pattern: str = '*.csv') -> pd.DataFrame
```

**Parameters:**
- `dirpath`: Path to the directory containing CSV files
- `pattern`: Glob pattern for finding CSV files (default: `'*.csv'`)

**Returns:**
- `pd.DataFrame` with:
  - Combined data from all CSV files
  - `DatetimeIndex` sorted chronologically
  - Duplicate timestamps removed (keeping first occurrence)

**Raises:**
- `FileNotFoundError`: If the directory does not exist
- `ValueError`: If no CSV files are found

**Example:**
```python
from src.data_loader import read_sensor_directory

# Load all CSV files from a directory
df = read_sensor_directory('data/january_2024/')

print(f"Total records: {len(df)}")
print(f"Time span: {df.index[-1] - df.index[0]}")

# Load with custom pattern (only specific animal)
df_animal = read_sensor_directory('data/', pattern='animal_A*.csv')
```

## Error Handling

### Missing Columns

If a CSV file is missing required columns, a descriptive error is raised:

```python
try:
    df = read_sensor_csv('incomplete.csv')
except ValueError as e:
    print(e)
    # Output: CSV file 'incomplete.csv' is missing required columns: ['Dzg', 'Lyg', 'Sxg'].
    #         Expected columns: ['timestamp', 'temperature', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg']
```

### Unparseable Timestamps

If timestamps cannot be parsed, an error with sample values is provided:

```python
try:
    df = read_sensor_csv('bad_timestamps.csv')
except ValueError as e:
    print(e)
    # Output: Unable to parse timestamp column. Tried multiple formats but failed.
    #         Sample values: ['invalid', 'not_a_date', 'xyz'].
    #         Expected formats: Unix epoch, ISO 8601, or common date strings like 'YYYY-MM-DD HH:MM:SS'
```

### Non-Numeric Sensor Values

Non-numeric values in sensor columns are automatically coerced to NaN with warnings logged:

```python
# If CSV contains non-numeric values like 'N/A' or 'error'
df = read_sensor_csv('data_with_errors.csv')

# Check for missing values
missing = df.isna().sum()
print(missing)
# Output shows which columns have NaN values
```

### File Not Found

```python
try:
    df = read_sensor_csv('nonexistent.csv')
except FileNotFoundError as e:
    print(e)
    # Output: The file 'nonexistent.csv' does not exist. Please check the file path and try again.
```

## Data Analysis Examples

### Basic Statistics

```python
from src.data_loader import read_sensor_csv

df = read_sensor_csv('data/sensor_data.csv')

# Temperature statistics
print("Temperature Statistics:")
print(f"  Mean: {df['temperature'].mean():.2f}°C")
print(f"  Min: {df['temperature'].min():.2f}°C")
print(f"  Max: {df['temperature'].max():.2f}°C")
print(f"  Std: {df['temperature'].std():.2f}°C")

# All sensor statistics
print("\nAll Sensors:")
print(df.describe())
```

### Time-Based Analysis

```python
import pandas as pd
from src.data_loader import read_sensor_csv

df = read_sensor_csv('data/24hour_data.csv')

# Resample to hourly averages
hourly = df.resample('H').mean()
print(hourly['temperature'].head())

# Resample to 10-minute windows
ten_min = df.resample('10T').mean()

# Calculate daily statistics
daily_stats = df.resample('D').agg({
    'temperature': ['mean', 'min', 'max'],
    'Fxa': 'mean',
    'Mya': 'mean',
    'Rza': 'mean'
})
```

### Activity Analysis

```python
import numpy as np
from src.data_loader import read_sensor_csv

df = read_sensor_csv('data/sensor_data.csv')

# Calculate total acceleration magnitude
df['total_accel'] = np.sqrt(df['Fxa']**2 + df['Mya']**2 + df['Rza']**2)

# Calculate total angular velocity
df['total_gyro'] = np.sqrt(df['Sxg']**2 + df['Lyg']**2 + df['Dzg']**2)

# Identify high activity periods
threshold = df['total_accel'].quantile(0.75)
high_activity = df[df['total_accel'] > threshold]

print(f"High activity periods: {len(high_activity)} records")
print(f"Percentage: {len(high_activity)/len(df)*100:.1f}%")
```

### Temperature Anomaly Detection

```python
from src.data_loader import read_sensor_csv

df = read_sensor_csv('data/sensor_data.csv')

# Define temperature thresholds
FEVER_THRESHOLD = 39.5
NORMAL_RANGE = (38.0, 39.0)

# Detect potential fever
fever_periods = df[df['temperature'] > FEVER_THRESHOLD]
if len(fever_periods) > 0:
    print(f"⚠️  Fever detected: {len(fever_periods)} records above {FEVER_THRESHOLD}°C")
    print(f"Duration: {fever_periods.index[-1] - fever_periods.index[0]}")

# Detect abnormal temperatures
abnormal = df[(df['temperature'] < NORMAL_RANGE[0]) | 
              (df['temperature'] > NORMAL_RANGE[1])]
print(f"Abnormal temperature readings: {len(abnormal)}/{len(df)}")
```

## Logging

The module uses the Artemis logging framework. Logs are written to `logs/system/system.log`.

### Log Levels

- **DEBUG**: Detailed diagnostic information (timestamp format detection, data dimensions)
- **INFO**: General operational messages (file loading progress, record counts)
- **WARNING**: Data quality issues (non-numeric values, duplicate timestamps)
- **ERROR**: Failures (missing files, parsing errors)

### Example Log Output

```
2024-01-15 10:00:00 - artemis.data.loader - INFO - Loading sensor data from: data/sensor_data.csv
2024-01-15 10:00:00 - artemis.data.loader - DEBUG - Loaded 1440 rows from data/sensor_data.csv
2024-01-15 10:00:00 - artemis.data.loader - DEBUG - Column validation passed for data/sensor_data.csv
2024-01-15 10:00:00 - artemis.data.loader - DEBUG - Successfully parsed timestamps using format: %Y-%m-%d %H:%M:%S
2024-01-15 10:00:00 - artemis.data.loader - WARNING - Non-numeric values in column 'Fxa' at rows [42, 103]. Sample values: ['N/A', 'error']. These values have been set to NaN.
2024-01-15 10:00:00 - artemis.data.loader - INFO - Successfully loaded 1440 records from data/sensor_data.csv (time range: 2024-01-15 00:00:00 to 2024-01-15 23:59:00)
```

## Best Practices

### 1. Data Validation

Always check for missing values after loading:

```python
df = read_sensor_csv('data.csv')

# Check for NaN values
if df.isna().any().any():
    print("Warning: Dataset contains missing values")
    print(df.isna().sum())
```

### 2. Time Range Verification

Verify the expected time range:

```python
df = read_sensor_csv('data.csv')

expected_duration = pd.Timedelta(days=1)
actual_duration = df.index[-1] - df.index[0]

if actual_duration < expected_duration:
    print(f"Warning: Expected {expected_duration} but got {actual_duration}")
```

### 3. Memory Efficiency

For very large datasets, consider loading and processing in batches:

```python
# Future feature: chunksize parameter
# df_chunks = read_sensor_csv('large_file.csv', chunksize=10000)
# for chunk in df_chunks:
#     process_chunk(chunk)
```

### 4. Directory Organization

Organize CSV files by date or animal ID for efficient batch loading:

```
data/
├── animal_A12345/
│   ├── 2024-01-15.csv
│   ├── 2024-01-16.csv
│   └── 2024-01-17.csv
└── animal_A67890/
    ├── 2024-01-15.csv
    └── 2024-01-16.csv
```

```python
# Load all data for a specific animal
df = read_sensor_directory('data/animal_A12345/')
```

## Troubleshooting

### Issue: "No CSV files found in directory"

**Cause:** Directory is empty or pattern doesn't match any files.

**Solution:**
- Check directory path is correct
- Verify CSV files exist in the directory
- Adjust the `pattern` parameter if using custom file naming

### Issue: "Missing required columns"

**Cause:** CSV file doesn't have all required columns.

**Solution:**
- Check CSV has all 8 columns: `timestamp, temperature, Fxa, Mya, Rza, Sxg, Lyg, Dzg`
- Verify column names match exactly (case-sensitive)
- Ensure no typos in column headers

### Issue: "Unable to parse timestamp column"

**Cause:** Timestamp format not recognized.

**Solution:**
- Check sample timestamp values in the error message
- Convert timestamps to one of the supported formats
- Use standard format: `YYYY-MM-DD HH:MM:SS`

### Issue: High memory usage with large files

**Cause:** Loading entire file into memory at once.

**Solution:**
- Split large files into smaller chunks
- Process data by time periods
- Use directory loading with smaller files

## Running Examples

The module includes comprehensive examples:

```bash
# Run the example script
python examples/data_loader_example.py
```

This will demonstrate:
1. Loading a single CSV file
2. Handling different timestamp formats
3. Loading multiple files from a directory
4. Basic data analysis

## Testing

Run the test suite to verify functionality:

```bash
# Run all data loader tests
pytest tests/test_data_loader.py -v

# Run specific test class
pytest tests/test_data_loader.py::TestReadSensorCSV -v

# Run with coverage report
pytest tests/test_data_loader.py --cov=src.data_loader
```

## See Also

- [Logging Usage Guide](logging_usage.md) - Information about the logging framework
- [Behavior Specifications](behavior_specifications.md) - Details on sensor data analysis
- Main [README](../README.md) - Project overview and setup instructions
