# Data Ingestion Module - Quick Start Guide

## Installation

1. Ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

## Basic Usage (3 Lines of Code)

```python
from data_processing.ingestion import DataIngestionModule

ingestion = DataIngestionModule()
df, summary = ingestion.load_batch('path/to/your/data.csv')
```

That's it! Your CSV data is now loaded, validated, and ready to use.

## What You Get

The module automatically:
- ✅ Detects CSV delimiter (comma, semicolon, tab)
- ✅ Detects file encoding (UTF-8, ASCII, etc.)
- ✅ Parses timestamps (ISO 8601, Unix, custom formats)
- ✅ Validates all 7 required sensor columns
- ✅ Checks data types and value ranges
- ✅ Verifies timestamp chronology
- ✅ Reports errors with line numbers
- ✅ Logs everything for debugging

## Common Use Cases

### 1. Load Historical Data
```python
from data_processing.ingestion import DataIngestionModule

ingestion = DataIngestionModule()
df, summary = ingestion.load_batch('data/historical_sensor_data.csv')

print(f"Loaded {summary.valid_rows} valid rows")
if summary.errors:
    print(f"Found {len(summary.errors)} errors")
```

### 2. Simulate Real-Time Ingestion
```python
# First load
df1, _ = ingestion.load_incremental('data/live_data.csv')

# Later, after new data arrives...
df2, _ = ingestion.load_incremental('data/live_data.csv')
# df2 contains only the new rows!
```

### 3. Monitor File for New Data
```python
def process_data(df, summary):
    print(f"New data arrived: {len(df)} rows")
    # Your processing logic here...

ingestion.monitor_file(
    'data/live_data.csv',
    interval_seconds=60,  # Check every minute
    callback=process_data
)
```

### 4. Handle Errors Gracefully
```python
df, summary = ingestion.load_batch('data/possibly_malformed.csv')

if not summary.validation_summary.is_valid():
    print("Data has issues:")
    for error in summary.errors[:5]:
        print(f"  - {error}")
    
    # But you can still use valid rows!
    valid_df = df[df['temperature'].notna()]
```

### 5. Work with Large Files
```python
# Process in chunks to save memory
df, summary = ingestion.load_batch(
    'data/huge_file.csv',
    chunk_size=10000
)
```

## Expected CSV Format

Your CSV should have these columns:
```
timestamp,temperature,fxa,mya,rza,sxg,lyg,dzg
2024-01-01T00:00:00,38.5,-0.02,0.01,-0.85,2.1,-1.5,0.8
```

Optional columns (automatically detected):
- `cow_id` - Animal identifier
- `sensor_id` - Sensor identifier
- `state` - Ground truth label

## Checking Results

```python
df, summary = ingestion.load_batch('data/file.csv')

# Check if everything was OK
if summary.validation_summary.is_valid():
    print("✓ All data valid!")
else:
    print(f"⚠ {len(summary.errors)} errors found")

# View summary report
print(summary)

# Export detailed report
ingestion.export_summary(summary, 'reports/summary.txt')
```

## Integration with Simulation

```python
from simulation.engine import SimulationEngine
from data_processing.ingestion import DataIngestionModule

# Generate test data
engine = SimulationEngine()
df = engine.generate_continuous_data(duration_hours=24)
engine.export_to_csv(df, 'data/simulated/test.csv')

# Ingest it
ingestion = DataIngestionModule()
df_loaded, summary = ingestion.load_batch('data/simulated/test.csv')
```

## Running Examples

```bash
cd src/data_processing
python example_usage.py
```

## Running Tests

```bash
cd tests
python test_ingestion.py
```

## Troubleshooting

### File Not Found
```python
# Check file path
import os
print(os.path.exists('data/file.csv'))  # Should be True
```

### Encoding Issues
```python
# Module auto-detects encoding, but you can force it:
df = ingestion.csv_parser.read_csv('file.csv', encoding='latin-1')
```

### Timestamp Format Not Detected
```python
# Use custom timestamp parser
from data_processing.parsers import TimestampParser

parser = TimestampParser()
parser.detected_format = '%d/%m/%Y %H:%M:%S'  # Custom format
```

### Too Many Validation Errors
```python
# Load without validation, then manually validate
ingestion = DataIngestionModule(validate_data=False)
df, _ = ingestion.load_batch('file.csv')

# Inspect data manually
print(df.info())
print(df.describe())
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for more examples
- Run tests to see all features in action

## Need Help?

- Check logs in `logs/ingestion_errors.log`
- Review validation summary for specific errors
- Run examples to see expected behavior
