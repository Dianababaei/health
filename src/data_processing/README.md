# Data Processing Module

This module provides comprehensive data ingestion capabilities for cattle sensor data with support for batch and incremental loading modes, extensive validation, and error handling.

## Features

- **Batch Loading Mode**: Load entire CSV files for historical analysis
- **Incremental Loading Mode**: Monitor files for new data (simulates real-time ingestion)
- **Flexible Timestamp Parsing**: Supports ISO 8601, Unix timestamps (seconds/milliseconds), and custom formats
- **CSV Auto-Detection**: Automatically detects delimiters (comma, semicolon, tab) and file encoding
- **Comprehensive Validation**: 
  - Required column checks
  - Data type validation
  - Sensor value range validation
  - Timestamp chronology validation
- **Error Handling**: Graceful handling with detailed error reporting and logging
- **File Position Tracking**: Resume incremental reads from last position

## Module Structure

```
data_processing/
├── __init__.py
├── ingestion.py     # Main ingestion module
├── parsers.py       # Timestamp and CSV parsing utilities
├── validators.py    # Data validation functions
└── README.md        # This file
```

## Quick Start

### Batch Loading

```python
from data_processing.ingestion import DataIngestionModule

# Initialize module
ingestion = DataIngestionModule(log_dir='logs')

# Load entire CSV file
df, summary = ingestion.load_batch('data/simulated/sensor_data.csv')

print(f"Loaded {len(df)} rows")
print(f"Valid rows: {summary.valid_rows}")
print(f"Errors: {len(summary.errors)}")
```

### Incremental Loading

```python
# First read
df1, summary1 = ingestion.load_incremental('data/simulated/sensor_data.csv')
print(f"First read: {len(df1)} rows")

# Later, after new data is appended
df2, summary2 = ingestion.load_incremental('data/simulated/sensor_data.csv')
print(f"New data: {len(df2)} rows")
```

### File Monitoring

```python
def process_new_data(df, summary):
    """Callback function for new data."""
    print(f"Received {len(df)} new rows")
    # Process data here...

# Monitor file every 60 seconds
ingestion.monitor_file(
    'data/simulated/sensor_data.csv',
    interval_seconds=60,
    callback=process_new_data
)
```

## CSV Format

### Required Columns

- `timestamp`: ISO 8601 format or Unix timestamp
- `temperature`: Float (°C)
- `fxa`, `mya`, `rza`: Float (m/s² or g-units) - accelerometer data
- `sxg`, `lyg`, `dzg`: Float (°/s or rad/s) - gyroscope data

### Optional Columns

- `cow_id`: String/Integer (for multi-animal systems)
- `sensor_id`: String (for sensor identification)
- `state`: String (ground truth labels for training data)

### Example CSV

```csv
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg
2024-01-01T00:00:00,38.5,-0.02,0.01,-0.85,2.1,-1.5,0.8
2024-01-01T00:01:00,38.5,-0.03,0.02,-0.82,2.3,-1.4,0.9
```

## Validation

The module performs comprehensive validation:

### Column Validation
- Checks all 7 required sensor columns are present
- Validates timestamp column exists

### Data Type Validation
- Converts sensor values to numeric types
- Identifies and reports non-numeric values
- Marks invalid rows without crashing

### Range Validation
- Temperature: 35-42°C
- Accelerometer (fxa, mya): -5 to 5 g
- Accelerometer (rza): -2 to 2 g
- Gyroscope (sxg, lyg, dzg): -200 to 200 °/s

### Timestamp Validation
- Verifies chronological order
- Detects duplicate timestamps
- Checks for irregular time intervals (expected: ~60 seconds)
- Validates timestamp ranges (not in distant past/future)

## Error Handling

Errors are categorized and logged:

- **File-Level Errors**: File not found, empty file, encoding errors
- **Column Errors**: Missing required columns
- **Type Errors**: Non-numeric values in sensor columns
- **Range Errors**: Values outside acceptable ranges
- **Timestamp Errors**: Unparseable timestamps, non-chronological order

Example error report:

```
============================================================
Data Ingestion Summary
============================================================
File: data/sample.csv
Duration: 0.23 seconds
Total rows read: 100
Valid rows: 92
Skipped rows: 8
Errors: 5
Warnings: 3

Top Errors (first 5):
  - [invalid_type] Non-numeric value in column 'fxa' (row 12)
  - [value_out_of_range] Value 999.90 for 'temperature' outside valid range [35.0, 42.0] (row 15)
  ...
============================================================
```

## Advanced Usage

### Custom Validation

```python
# Disable automatic validation
ingestion = DataIngestionModule(validate_data=False)
df, summary = ingestion.load_batch('data/sample.csv')

# Perform custom validation
from data_processing.validators import DataValidator
validator = DataValidator()
df_clean, validation_summary = validator.validate_dataframe(df)
```

### Chunked Loading (Large Files)

```python
# Process large files in chunks to save memory
df, summary = ingestion.load_batch(
    'data/large_file.csv',
    chunk_size=10000  # Process 10,000 rows at a time
)
```

### Export Summary Report

```python
df, summary = ingestion.load_batch('data/sample.csv')
ingestion.export_summary(summary, 'reports/ingestion_summary.txt')
```

## Testing

Run the test suite:

```bash
cd tests
python test_ingestion.py
```

Test fixtures are provided in `tests/fixtures/`:
- `sample_valid.csv`: Valid sensor data
- `sample_malformed.csv`: Data with various errors
- `sample_edge_cases.csv`: Unix timestamps and optional columns

## Integration with Simulation

The module is designed to work seamlessly with the simulation engine:

```python
from simulation.engine import SimulationEngine
from data_processing.ingestion import DataIngestionModule

# Generate simulated data
engine = SimulationEngine(random_seed=42)
df = engine.generate_continuous_data(duration_hours=24)
engine.export_to_csv(df, 'data/simulated/test_data.csv')

# Ingest simulated data
ingestion = DataIngestionModule()
df_loaded, summary = ingestion.load_batch('data/simulated/test_data.csv')

print(f"Generated {len(df)} rows, loaded {len(df_loaded)} rows")
```

## Logging

Logs are written to `logs/ingestion_errors.log` by default.

Configure logging:

```python
ingestion = DataIngestionModule(
    log_dir='custom_logs',
    log_file='custom_errors.log'
)
```

## Performance

- **Batch Loading**: Handles files with 90+ days of data (130,000+ rows)
- **Incremental Loading**: Efficiently tracks file position for minimal overhead
- **Memory Management**: Optional chunked processing for large files
- **Validation Speed**: ~10,000-50,000 rows/second depending on validation complexity

## Next Steps

After ingestion, data is ready for:
- Layer 1 processing (behavior detection)
- Feature extraction
- Model training
- Real-time alert generation
