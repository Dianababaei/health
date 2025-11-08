# Activity Metrics Module

## Overview

The Activity Metrics module provides comprehensive behavioral activity tracking and analysis for livestock monitoring systems. It processes classified behavioral state data to generate duration metrics, movement intensity calculations, activity ratios, and state transition analytics.

## Features

### Core Metrics
- **Duration Tracking**: Calculate time spent in each behavioral state (lying, standing, walking, feeding, ruminating)
- **Movement Intensity**: Compute accelerometer magnitude from Fxa, Mya, Rza values
- **Activity/Rest Ratios**: Classify and analyze rest vs. active behavior patterns
- **State Transitions**: Count and analyze behavioral state changes

### Aggregations
- **Hourly Aggregation**: 60-minute windows with state breakdowns
- **Daily Aggregation**: 24-hour summaries (1440 minutes)

### Export Capabilities
- **Behavioral State Logs**: Structured output with timestamp, state, confidence, duration, and intensity
- **CSV Format**: Standard comma-separated values
- **JSON Format**: Structured JSON with configurable orientation

### Edge Case Handling
- Missing data windows
- Single-state days (e.g., all lying if sick)
- Irregular sampling intervals
- Data validation and quality checks

## Installation

```python
from layer1_behavior import ActivityTracker

# Initialize tracker
tracker = ActivityTracker(output_dir="data/outputs/behavioral_logs")
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from layer1_behavior import ActivityTracker

# Load classified behavioral data
data = pd.read_csv('behavioral_states.csv')

# Initialize tracker
tracker = ActivityTracker()

# Generate behavioral log
log = tracker.generate_behavioral_log(data)

# Export to CSV
tracker.export_to_csv(log, 'behavioral_log.csv')
```

### Full Pipeline

```python
# Run complete analysis pipeline
results = tracker.process_full_pipeline(
    data,
    export_logs=True,
    export_hourly=True,
    export_daily=True,
    csv_format=True,
    json_format=True
)

# Access results
behavioral_log = results['behavioral_log']
hourly_agg = results['hourly_aggregation']
daily_agg = results['daily_aggregation']
activity_ratio = results['activity_rest_ratio']
transitions = results['state_transitions']
```

## Input Data Format

The module expects a pandas DataFrame with the following columns:

### Required Columns
- `timestamp`: DateTime column with minute-level sampling
- `behavioral_state`: String column with values: 'lying', 'standing', 'walking', 'feeding', 'ruminating'

### Optional Columns (for movement intensity)
- `fxa`: X-axis acceleration (forward-backward)
- `mya`: Y-axis acceleration (side-to-side)
- `rza`: Z-axis acceleration (up-down)

### Optional Columns (for confidence tracking)
- `confidence_score`: Float between 0-1 indicating classification confidence

### Example Input

```csv
timestamp,behavioral_state,fxa,mya,rza,confidence_score
2024-01-01T00:00:00,lying,0.01,0.01,-0.85,0.95
2024-01-01T00:01:00,lying,0.02,0.01,-0.84,0.93
2024-01-01T00:02:00,standing,0.10,0.05,-0.80,0.91
```

## Output Format

### Behavioral State Log

```csv
timestamp,behavioral_state,confidence_score,duration_minutes,movement_intensity
2024-01-01T00:00:00,lying,0.95,120,0.85
2024-01-01T02:00:00,standing,0.91,45,0.94
2024-01-01T02:45:00,walking,0.88,30,1.52
```

### Hourly Aggregation

```csv
hour,lying_minutes,standing_minutes,walking_minutes,feeding_minutes,ruminating_minutes,total_minutes,avg_movement_intensity,state_transitions,rest_percentage,active_percentage
2024-01-01T00:00:00,45,10,5,0,0,60,0.89,3,75.0,25.0
```

### Daily Aggregation

```csv
day,lying_minutes,standing_minutes,walking_minutes,feeding_minutes,ruminating_minutes,total_minutes,avg_movement_intensity,state_transitions,rest_percentage,active_percentage
2024-01-01,720,320,180,120,100,1440,1.05,48,50.0,50.0
```

## API Reference

### ActivityTracker Class

#### `__init__(output_dir='data/outputs/behavioral_logs')`
Initialize activity tracker with output directory.

#### `calculate_durations(data, timestamp_col='timestamp', state_col='behavioral_state')`
Calculate duration for consecutive same-state periods.

**Returns**: DataFrame with `duration_minutes` column

#### `calculate_movement_intensity(data, fxa_col='fxa', mya_col='mya', rza_col='rza')`
Calculate movement intensity magnitude: sqrt(Fxa² + Mya² + Rza²)

**Returns**: DataFrame with `movement_intensity` column

#### `aggregate_hourly(data, timestamp_col='timestamp', state_col='behavioral_state', intensity_col='movement_intensity')`
Aggregate behavioral data by hour.

**Returns**: DataFrame with hourly statistics

#### `aggregate_daily(data, timestamp_col='timestamp', state_col='behavioral_state', intensity_col='movement_intensity')`
Aggregate behavioral data by day.

**Returns**: DataFrame with daily statistics

#### `calculate_activity_rest_ratio(data, state_col='behavioral_state')`
Calculate overall activity/rest ratio.

**Returns**: Dictionary with percentages and counts

#### `count_state_transitions(data, state_col='behavioral_state', timestamp_col='timestamp')`
Count state transitions and analyze patterns.

**Returns**: Dictionary with transition statistics

#### `generate_behavioral_log(data, timestamp_col='timestamp', state_col='behavioral_state', confidence_col=None, include_movement_intensity=True)`
Generate behavioral state log with all required fields.

**Returns**: DataFrame with behavioral log format

#### `export_to_csv(data, filename, include_timestamp=True)`
Export data to CSV format.

**Returns**: Path to exported file

#### `export_to_json(data, filename, include_timestamp=True, orient='records')`
Export data to JSON format.

**Returns**: Path to exported file

#### `process_full_pipeline(data, export_logs=True, export_hourly=True, export_daily=True, csv_format=True, json_format=True)`
Run complete activity metrics pipeline.

**Returns**: Dictionary with all computed metrics and export paths

#### `handle_missing_data(data, timestamp_col='timestamp', state_col='behavioral_state', fill_method='forward')`
Handle missing data windows.

**Returns**: DataFrame with handled missing data

#### `validate_daily_totals(daily_agg, expected_minutes=1440, tolerance=5.0)`
Validate that daily aggregations sum to expected total.

**Returns**: Dictionary with validation results

## Performance

- **Processing Speed**: Handles 43,200 records (30 days) in < 10 seconds
- **Memory Usage**: Efficient pandas operations with minimal memory overhead
- **Scalability**: Tested with multi-day datasets up to 90 days

## Validation Metrics

### Duration Tracking
- Correctly sums consecutive same-state periods
- Verified on synthetic datasets with known durations

### Hourly Aggregations
- Partitions 60-minute windows correctly
- All minutes accounted for (±2 minute tolerance)

### Daily Aggregations
- Sums to 1440 minutes (24 hours)
- ±5 minute tolerance for data quality variations

### Movement Intensity
- Correlates with expected activity levels
- Walking > Standing > Lying (validated on test data)

### Activity/Rest Ratios
- Matches ground truth on labeled validation data
- ±5% error tolerance

## Examples

### Calculate Duration for Specific States

```python
# Calculate durations
df_with_durations = tracker.calculate_durations(data)

# Filter by state
lying_durations = df_with_durations[df_with_durations['behavioral_state'] == 'lying']
print(f"Average lying duration: {lying_durations['duration_minutes'].mean():.2f} minutes")
```

### Analyze Activity Patterns by Hour

```python
# Get hourly aggregation
hourly = tracker.aggregate_hourly(data)

# Find peak activity hours
hourly['activity_score'] = hourly['walking_minutes'] + hourly['feeding_minutes']
peak_hour = hourly.loc[hourly['activity_score'].idxmax(), 'hour']
print(f"Peak activity hour: {peak_hour}")
```

### Detect Abnormal Days

```python
# Get daily aggregation
daily = tracker.aggregate_daily(data)

# Find days with excessive rest
abnormal_days = daily[daily['rest_percentage'] > 80]
print(f"Found {len(abnormal_days)} days with >80% rest time")
```

### Track State Transitions

```python
# Count transitions
transitions = tracker.count_state_transitions(data)

print(f"Total transitions: {transitions['total_transitions']}")
print("\nTransition patterns:")
for pattern, count in transitions['transition_matrix'].items():
    print(f"  {pattern}: {count}")
```

## Testing

Run the test suite:

```bash
pytest tests/test_activity_metrics.py -v
```

Test coverage includes:
- Duration calculation (basic, known values, single-state)
- Movement intensity (magnitude, correlation with activity)
- Hourly aggregation (60-minute windows, all minutes accounted)
- Daily aggregation (1440-minute days, state breakdowns)
- Activity/rest ratios (known values, edge cases)
- State transitions (basic, known patterns, no change)
- Behavioral log generation (required fields, confidence scores)
- Export functions (CSV, JSON, with timestamps)
- Full pipeline (complete execution, large datasets)
- Edge cases (missing data, single-state days, empty data)

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.23.0
- python >= 3.8

## Integration

This module integrates with:
- **Data Ingestion Module**: Accepts output from classification pipeline
- **Layer 2 (Physiology)**: Provides behavioral context for temperature analysis
- **Layer 3 (Health Intelligence)**: Supplies activity metrics for health scoring
- **Dashboard**: Exports data for visualization and monitoring

## Troubleshooting

### Issue: Daily totals don't sum to 1440 minutes

**Solution**: Check for missing data windows or irregular sampling intervals. Use `handle_missing_data()` to fill gaps.

### Issue: Movement intensity values seem incorrect

**Solution**: Verify that accelerometer columns (fxa, mya, rza) are present and contain valid numeric values. Check units and scale.

### Issue: No state transitions detected

**Solution**: Ensure data is sorted by timestamp. Single-state days (e.g., all lying) will naturally have zero transitions.

### Issue: Export fails with "directory not found"

**Solution**: The module automatically creates the output directory. Check write permissions and path validity.

## License

Part of the Artemis Health livestock monitoring system.

## Contact

For issues or questions, refer to the main project documentation.
