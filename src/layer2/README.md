# Layer 2: Physiological Analysis

## Overview

Layer 2 provides physiological analysis capabilities for livestock monitoring, focusing on temperature pattern analysis and correlation with behavioral data to detect health conditions.

## Modules

### 1. Baseline Calculator (`baseline.py`)

Calculates individual animal temperature baselines for detecting anomalies.

**Features:**
- Rolling window baseline calculation
- Percentile-based robust estimation  
- Time-of-day specific baselines
- Circadian rhythm consideration

**Example Usage:**
```python
from layer2 import BaselineCalculator

calculator = BaselineCalculator(window_hours=24)
df_with_baseline = calculator.calculate_baseline(temperature_data)

# Get individual baseline
baseline_stats = calculator.calculate_individual_baseline(
    temperature_data, 
    lookback_days=7
)
print(f"Baseline: {baseline_stats['baseline']:.2f}°C")
```

### 2. Temperature Anomaly Detector (`temperature_anomaly.py`)

Detects temperature anomalies based on baseline deviations and absolute thresholds.

**Features:**
- Absolute threshold detection (fever >39.0°C, heat stress >39.5°C)
- Baseline deviation detection
- Severity scoring (0.0-1.0)
- Anomaly event extraction

**Example Usage:**
```python
from layer2 import TemperatureAnomalyDetector, AnomalyType

detector = TemperatureAnomalyDetector(
    fever_threshold=39.0,
    heat_stress_threshold=39.5
)

# Detect anomalies
df_with_anomalies = detector.detect_anomalies(temperature_data)

# Extract anomaly events
events = detector.extract_anomaly_events(df_with_anomalies)
for event in events:
    print(f"{event.timestamp}: {event.anomaly_type.value} "
          f"(severity: {event.severity:.2f})")
```

### 3. Temperature-Activity Correlator (`temp_activity_correlation.py`)

**Primary Module**: Combines temperature data with behavioral states to identify health patterns.

**Features:**
- Time-aligned merging of temperature and behavioral data
- Fever pattern detection (high temp + low activity)
- Heat stress pattern detection (high temp + high activity)
- Pearson correlation analysis with multiple time windows
- Lag analysis
- Pattern confidence scoring (0-100)
- Structured correlation event generation

**Health Patterns Detected:**

#### Fever Pattern
- **Temperature**: >39.0°C
- **Activity**: Reduced motion
  - Lying state with duration >60 minutes OR
  - Movement intensity <20% of individual baseline OR
  - Activity level in bottom 25th percentile
- **Distinguishes from**: Normal nighttime rest using time-of-day context

#### Heat Stress Pattern
- **Temperature**: >39.5°C  
- **Activity**: Elevated activity
  - Walking state with movement intensity >baseline OR
  - Activity level in top 50th percentile OR
  - Frequent state transitions indicating restlessness
- **Context**: More likely during daytime hours (08:00-20:00)

**Example Usage:**
```python
from layer2 import TemperatureActivityCorrelator, HealthPattern

# Initialize correlator
correlator = TemperatureActivityCorrelator(
    fever_temp_threshold=39.0,
    heat_stress_temp_threshold=39.5
)

# Run full correlation analysis
merged_data, events, metrics = correlator.process_full_correlation(
    temperature_data,
    behavioral_data
)

# Process detected events
print(f"Total events detected: {len(events)}")
for event in events:
    print(f"\n{event.pattern_type.value.upper()} Pattern:")
    print(f"  Timestamp: {event.timestamp}")
    print(f"  Confidence: {event.confidence:.1f}/100")
    print(f"  Temperature: {event.temperature:.2f}°C")
    print(f"  Activity Level: {event.activity_level:.2f}")
    print(f"  Duration: {event.duration_minutes:.0f} minutes")
    print(f"  Behavioral State: {event.behavioral_state}")
```

## Data Flow

```
Temperature Data (Layer 2) ──┐
                             ├─→ Time-Aligned Merge ─→ Correlation Analysis ─→ Health Events
Behavioral Data (Layer 1) ───┘
```

1. **Input**: Temperature readings + Behavioral state classifications
2. **Processing**: Time-aligned merging with 1-2 minute lag tolerance
3. **Analysis**: Pattern detection + Correlation metrics + Confidence scoring
4. **Output**: Structured correlation events for Layer 3 (Health Intelligence)

## Input Data Format

### Temperature Data
```csv
timestamp,temperature,baseline_temp
2024-01-01T00:00:00,38.5,38.4
2024-01-01T00:01:00,38.6,38.4
```

### Behavioral Data (from Layer 1)
```csv
timestamp,behavioral_state,movement_intensity,duration_minutes
2024-01-01T00:00:00,lying,0.15,120
2024-01-01T00:01:00,lying,0.14,119
```

## Output Format

### Correlation Events
```python
CorrelationEvent(
    timestamp=datetime(2024, 1, 1, 10, 0),
    pattern_type=HealthPattern.FEVER,
    confidence=85.5,
    temperature=40.2,
    activity_level=0.12,
    behavioral_state='lying',
    duration_minutes=180.0,
    correlation_coefficient=-0.72,
    contributing_factors={
        'avg_temperature': 40.1,
        'max_temperature': 40.5,
        'avg_activity': 0.11,
        'min_activity': 0.08,
        'sample_count': 180
    }
)
```

## Success Criteria

✅ **Fever Pattern Detection**: >90% accuracy on simulated data
- Correctly identifies temp >39.0°C + reduced motion
- Distinguishes from normal nighttime rest (false positive rate <10%)

✅ **Heat Stress Pattern Detection**: >85% accuracy on simulated data
- Correctly identifies temp >39.5°C + elevated activity
- Uses time-of-day context for improved accuracy

✅ **Correlation Metrics**: Meaningful coefficients in -1 to +1 range
- Multiple time windows: 1-hour, 4-hour, 24-hour
- Lag analysis to determine temporal relationships

✅ **Confidence Scoring**: 0-100 scale aligned with clinical significance
- Combines temperature excess, activity patterns, duration, and correlation strength

✅ **Performance**: Processes minute-by-minute data in <50ms per data point
- Tested on 7-day datasets (10,080 records)

✅ **Layer 3 Compatibility**: Generates structured events for alert system
- Serializable to JSON/dict format
- Contains all necessary metadata for downstream processing

## Integration with Other Layers

### Layer 1 Integration
- Reads behavioral state classifications: lying, standing, walking, ruminating, feeding
- Accesses activity metrics: movement intensity, rest duration, state transitions
- Processes minute-by-minute behavioral data aligned with temperature readings
- Handles asynchronous data availability (behavioral data may lag by 1-2 minutes)

### Layer 3 Integration  
- Provides correlation events for health intelligence layer
- Supplies pattern confidence scores for alert prioritization
- Enables multi-pattern health assessment (fever + heat stress + normal)

## Configuration

### Thresholds (Customizable)

```python
correlator = TemperatureActivityCorrelator(
    fever_temp_threshold=39.0,          # Fever temperature (°C)
    heat_stress_temp_threshold=39.5,    # Heat stress temperature (°C)
    reduced_motion_threshold=0.2,        # Activity level for "reduced motion"
    elevated_activity_threshold=0.5,     # Activity level for "elevated"
    time_alignment_tolerance_minutes=2,  # Merge tolerance
    min_pattern_duration_minutes=30      # Minimum pattern duration
)
```

### Baseline Calculation

```python
calculator = BaselineCalculator(
    window_hours=24,              # Rolling window size
    percentile_lower=25.0,        # Lower bound percentile
    percentile_upper=75.0,        # Upper bound percentile
    min_samples=60                # Minimum samples for valid baseline
)
```

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_correlation.py -v
```

**Test Coverage:**
- Baseline calculation (rolling, individual, hourly)
- Anomaly detection (fever, heat stress, normal)
- Fever pattern detection with edge cases
- Heat stress pattern detection with time context
- Correlation metrics (Pearson, lag analysis)
- Pattern confidence scoring
- Event generation and serialization
- Performance on large datasets (7 days)
- False positive rate validation

## Performance Characteristics

- **Processing Speed**: <50ms per data point (tested on 10,080 records)
- **Memory Usage**: Efficient pandas operations with rolling windows
- **Scalability**: Handles multi-day datasets (7+ days) without issues
- **Accuracy**: >90% fever detection, >85% heat stress detection

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scipy >= 1.7.0 (for correlation analysis)
- Python >= 3.8

## License

Part of Artemis Health livestock monitoring system.
