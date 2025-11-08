# Data Processing Module

This module provides comprehensive data validation and processing utilities for the Artemis Health livestock monitoring system.

## Features

### Data Validation

The validation module provides robust validation for sensor data with:

- **Completeness Check**: Verifies all 7 required parameters are present (Temperature, Fxa, Mya, Rza, Sxg, Lyg, Dzg)
- **Data Type Validation**: Ensures proper data types with automatic conversion and error handling
- **Range Validation**: Validates sensor values against expected ranges based on cattle physiology and sensor specifications
- **Timestamp Continuity**: Detects gaps >5 minutes between consecutive readings
- **Out-of-Range Detection**: Identifies critical conditions (hypothermia, severe fever, extreme accelerations)
- **Severity Classification**: Issues are categorized as ERROR, WARNING, or INFO
- **Detailed Reporting**: Comprehensive validation reports with statistics and issue tracking

## Usage

### Basic Validation

```python
import pandas as pd
from src.data_processing import validate_sensor_data

# Load your sensor data
data = pd.read_csv('sensor_data.csv')

# Validate the data
result = validate_sensor_data(data)

# Access results
clean_data = result['clean_data']
flagged_data = result['flagged_data']
report = result['report']
summary = result['summary']

print(f"Clean records: {summary['clean_records']}")
print(f"Flagged records: {summary['flagged_records']}")
print(f"Error count: {summary['error_count']}")
print(f"Warning count: {summary['warning_count']}")
```

### Advanced Usage with Custom Settings

```python
from src.data_processing import DataValidator
import logging

# Create validator with custom settings
validator = DataValidator(
    gap_threshold_minutes=10,  # Allow larger gaps
    log_level=logging.DEBUG    # More verbose logging
)

# Perform validation
clean_data, flagged_data, report = validator.validate(data)

# Access detailed report
print(report.get_summary())

# Convert issues to DataFrame for analysis
issues_df = report.to_dataframe()
issues_df.to_csv('validation_issues.csv', index=False)

# Get report as dictionary
report_dict = report.to_dict()
```

### Accessing Validation Issues

```python
# Iterate through issues
for issue in report.issues:
    print(f"{issue.severity.value}: {issue.message}")
    if issue.row_index is not None:
        print(f"  Row: {issue.row_index}")
    if issue.column:
        print(f"  Column: {issue.column}")
    if issue.value is not None:
        print(f"  Value: {issue.value}")

# Filter by severity
errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR]
warnings = [i for i in report.issues if i.severity == ValidationSeverity.WARNING]

# Filter by category
range_issues = [i for i in report.issues if i.category == 'range']
continuity_issues = [i for i in report.issues if i.category == 'timestamp_continuity']
```

## Validation Rules

### Data Completeness

- **Required Columns**: timestamp, temperature, fxa, mya, rza, sxg, lyg, dzg
- **Missing Columns**: ERROR severity
- **Missing Values**: ERROR severity

### Data Types

- **Timestamp**: Must be valid datetime format, not in future (ERROR if invalid)
- **Numeric Fields**: Must be convertible to float (ERROR if invalid)

### Range Validation

#### Temperature
- **Normal Range**: 35.0-42.0°C (WARNING if outside)
- **Critical Low**: <35.0°C - Hypothermia risk (ERROR)
- **Critical High**: >42.0°C - Severe fever (ERROR)

#### Accelerations (Fxa, Mya, Rza)
- **Typical Range**: -2.0 to +2.0g (WARNING if outside)
- **Extreme Threshold**: >5g - Physically impossible (ERROR)

#### Angular Velocities (Sxg, Lyg, Dzg)
- **Sensor Range**: -250 to +250 deg/s (WARNING if outside)

### Timestamp Continuity

- **Expected Interval**: 1 minute
- **Gap Threshold**: 5 minutes (configurable)
- **Gaps Detected**: WARNING severity
- **Duplicate Timestamps**: ERROR severity

## Validation Report Structure

### Summary Statistics

```json
{
  "total_records": 1000,
  "clean_records": 950,
  "flagged_records": 50,
  "total_issues": 75,
  "error_count": 5,
  "warning_count": 68,
  "info_count": 2,
  "duration_seconds": 0.234,
  "issues_by_category": {
    "completeness": 2,
    "range": 65,
    "timestamp_continuity": 5,
    "critical_out_of_range": 3
  }
}
```

### Individual Issue Structure

```json
{
  "severity": "WARNING",
  "category": "range",
  "message": "Temperature outside normal range (35.0-42.0°C)",
  "row_index": 123,
  "column": "temperature",
  "value": 42.3,
  "timestamp": "2024-01-15T10:30:00"
}
```

## Integration with Pipeline

```python
import pandas as pd
from src.data_processing import validate_sensor_data

def process_sensor_data(file_path):
    """Complete data processing pipeline with validation."""
    
    # Load data
    raw_data = pd.read_csv(file_path)
    print(f"Loaded {len(raw_data)} records")
    
    # Validate
    result = validate_sensor_data(raw_data)
    
    # Log validation results
    summary = result['summary']
    print(f"\nValidation Results:")
    print(f"  Clean: {summary['clean_records']}")
    print(f"  Flagged: {summary['flagged_records']}")
    print(f"  Errors: {summary['error_count']}")
    print(f"  Warnings: {summary['warning_count']}")
    
    # Export flagged data for review
    if len(result['flagged_data']) > 0:
        result['flagged_data'].to_csv('flagged_records.csv', index=False)
        result['report'].to_dataframe().to_csv('validation_issues.csv', index=False)
        print(f"\nFlagged records exported to 'flagged_records.csv'")
        print(f"Validation issues exported to 'validation_issues.csv'")
    
    # Continue with clean data
    clean_data = result['clean_data']
    
    # Further processing...
    return clean_data

# Use in pipeline
processed_data = process_sensor_data('data/raw/sensor_data.csv')
```

## Testing

Run the comprehensive test suite:

```bash
# Run all validation tests
python -m pytest tests/test_validation.py -v

# Run specific test class
python -m pytest tests/test_validation.py::TestDataValidatorCompleteness -v

# Run with coverage
python -m pytest tests/test_validation.py --cov=src.data_processing.validation
```

## Logging

The validation module uses Python's built-in logging. Configure as needed:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation.log'),
        logging.StreamHandler()
    ]
)

# Now run validation
result = validate_sensor_data(data)
```

## Performance

- **Validation Speed**: ~10,000-50,000 records/second (depends on data quality)
- **Memory Usage**: Minimal overhead (creates single copy of input data)
- **Batch Processing**: Supports datasets of any size

## Best Practices

1. **Always validate data** before feeding into analysis or ML pipelines
2. **Review flagged data** before discarding - some warnings may be valid edge cases
3. **Save validation reports** for audit trails and quality monitoring
4. **Adjust thresholds** based on your specific use case
5. **Monitor validation metrics** over time to detect data quality trends

## Troubleshooting

### High Error Rate

- Check data source formatting
- Verify sensor calibration
- Review timestamp generation logic

### Many Warnings

- Consider if ranges are too strict for your scenario
- Check if sensor is properly mounted
- Verify environmental conditions

### Performance Issues

- Process data in smaller batches
- Consider parallel processing for very large datasets
- Profile specific validation checks if needed
