# Multi-Day Health Trend Tracker Guide

## Overview

The Multi-Day Health Trend Tracker is a comprehensive system for analyzing cattle health trends over multiple time periods (7, 14, 30, and 90 days). It combines temperature data, activity patterns, alert frequency, and behavioral changes to generate unified health assessments.

## Key Features

- **Multi-Period Analysis**: Tracks trends over 7, 14, 30, and 90-day windows
- **Multi-Factor Integration**: Combines temperature, activity, alerts, and behavioral data
- **Trend Classification**: Categorizes health as improving, stable, or deteriorating
- **Confidence Scoring**: Provides reliability metrics based on data completeness
- **Dashboard-Ready Output**: JSON-serializable reports for visualization
- **Comparative Analysis**: Period-over-period comparisons and delta calculations

## Architecture

### Data Sources

The trend tracker integrates data from multiple layers:

1. **Layer 2 Physiological**: Temperature baselines and anomalies
2. **Layer 1 Behavioral**: Activity classifications and state distributions
3. **Health Intelligence**: Alert logs with timestamps and severity

### Core Components

```python
from health_intelligence import (
    MultiDayHealthTrendTracker,
    TrendIndicator,
    TimeWindowMetrics,
    HealthTrendReport
)
```

#### MultiDayHealthTrendTracker

Main analysis engine that processes multi-day health data.

**Parameters:**
- `temperature_baseline` (float): Normal baseline temperature (default: 38.5°C)

**Methods:**
- `analyze_trends()`: Generate comprehensive health trend report

#### TrendIndicator (Enum)

Health trend classification:
- `IMPROVING`: Health metrics improving
- `STABLE`: Consistent healthy patterns
- `DETERIORATING`: Declining health indicators
- `INSUFFICIENT_DATA`: Not enough data for analysis

#### TimeWindowMetrics

Metrics for a specific time period containing:
- Temperature metrics (mean, std, drift, anomalies)
- Activity metrics (total time, intensity, diversity)
- Alert metrics (count, severity, type distribution)
- Behavioral metrics (state distribution, transitions)
- Trend indicator and confidence score

#### HealthTrendReport

Complete trend report with:
- Trends for each period (7/14/30/90 days)
- Overall trend assessment
- Period comparisons
- Significant changes
- Actionable recommendations

## Usage

### Basic Analysis

```python
from health_intelligence import MultiDayHealthTrendTracker
import pandas as pd

# Initialize tracker
tracker = MultiDayHealthTrendTracker(temperature_baseline=38.5)

# Prepare your data
temperature_data = pd.DataFrame({
    'timestamp': [...],  # datetime values
    'temperature': [...]  # float values in °C
})

activity_data = pd.DataFrame({
    'timestamp': [...],  # datetime values
    'behavioral_state': [...],  # 'lying', 'standing', 'walking', etc.
    'movement_intensity': [...]  # float 0-1
})

alert_history = [
    {
        'timestamp': datetime(...),
        'alert_type': 'fever',
        'severity': 'critical',
        'cow_id': 'COW_001'
    },
    # ... more alerts
]

behavioral_states = activity_data.copy()  # Same structure

# Analyze trends
report = tracker.analyze_trends(
    cow_id='COW_001',
    temperature_data=temperature_data,
    activity_data=activity_data,
    alert_history=alert_history,
    behavioral_states=behavioral_states
)

# Access results
print(f"Overall Trend: {report.overall_trend.value}")
print(f"Confidence: {report.overall_confidence:.1%}")

# Check 7-day trend
if report.trend_7day:
    print(f"7-day trend: {report.trend_7day.trend_indicator.value}")
    print(f"Temperature drift: {report.trend_7day.temperature_baseline_drift:+.2f}°C")
    print(f"Alert count: {report.trend_7day.alert_count}")
```

### Dashboard Integration

Export to JSON for dashboard visualization:

```python
# Convert to JSON-serializable dict
dashboard_data = report.to_dict()

# Access structured data
print(dashboard_data['overall_trend'])  # 'improving', 'stable', or 'deteriorating'
print(dashboard_data['overall_confidence'])  # 0.0-1.0

# Period-specific metrics
if dashboard_data['trend_7day']:
    temp_metrics = dashboard_data['trend_7day']['temperature']
    activity_metrics = dashboard_data['trend_7day']['activity']
    alert_metrics = dashboard_data['trend_7day']['alerts']

# Period comparisons
comparisons = dashboard_data['period_comparisons']
if '7d_vs_14d' in comparisons:
    temp_change = comparisons['7d_vs_14d']['temperature_delta']
    activity_change = comparisons['7d_vs_14d']['activity_delta']
```

## Metrics Explained

### Temperature Metrics

- **Mean Temperature**: Average temperature over period
- **Standard Deviation**: Temperature variability (lower is more stable)
- **Baseline Drift**: Deviation from normal baseline (38.5°C default)
- **Anomaly Count**: Number of readings outside normal range (>39.5°C or <37.5°C)

### Activity Metrics

- **Total Active Minutes**: Time spent in active states (walking, feeding, etc.)
- **Mean Activity Level**: Average movement intensity (0-1 scale)
- **Rest Minutes**: Time spent lying or ruminating
- **Behavioral Diversity**: Shannon entropy of state distribution (0-1, higher = more varied)

### Alert Metrics

- **Total Count**: Number of alerts in period
- **Severity Distribution**: Breakdown by 'critical' vs 'warning'
- **Type Distribution**: Breakdown by alert type (fever, heat stress, etc.)

### Behavioral Metrics

- **State Distribution**: Percentage of time in each state
- **Changes Per Day**: Average number of state transitions per day

## Trend Classification Logic

The system classifies trends based on a weighted health score:

```
Health Score = (Temperature Score × 0.35) +
               (Activity Score × 0.30) +
               (Alert Score × 0.20) +
               (Behavioral Score × 0.15)
```

**Score Components:**

- **Temperature Score**: Lower drift and variance = higher score
- **Activity Score**: Moderate activity and high diversity = higher score
- **Alert Score**: Fewer alerts (especially critical) = higher score
- **Behavioral Score**: Normal transition frequency = higher score

**Classification:**

- Health Score > 0.7 → **IMPROVING**
- Health Score 0.5-0.7 → **STABLE**
- Health Score < 0.5 → **DETERIORATING**

## Confidence Scoring

Confidence reflects data quality and consistency:

```
Confidence = Data Completeness × Health Score
```

- **Data Completeness**: Percentage of expected samples present (requires ≥60%)
- **Health Score**: Combined health metric

Lower confidence suggests:
- Sparse data (gaps in sensor readings)
- Inconsistent patterns
- Recent sensor changes

## Period Comparisons

The system generates comparative metrics between adjacent periods:

- **7d vs 14d**: Recent changes (acute conditions)
- **14d vs 30d**: Medium-term trends (recovery progress)
- **30d vs 90d**: Long-term patterns (seasonal effects, reproductive cycles)

**Comparison Metrics:**

- Temperature delta (°C)
- Activity level delta (0-1 scale)
- Alert count delta (integer)
- Trend direction change (text description)

## Significant Changes

The system automatically identifies noteworthy changes:

- High alert frequency (>3 alerts in 7 days)
- Significant temperature drift (>0.5°C from baseline)
- Low activity level (<0.3 on 0-1 scale)
- Period-over-period changes (>0.3°C temperature, >0.2 activity)

## Recommendations

Generated based on overall trend and specific metrics:

### Deteriorating Trend
- "PRIORITY: Schedule veterinary examination"
- "Increase monitoring frequency"
- "Investigate root cause of frequent alerts"

### Stable Trend
- "Continue routine monitoring"

### Improving Trend
- "Positive health trend - maintain current care"

### Specific Conditions
- High alerts: "Investigate root cause of frequent alerts"
- Low activity: "Check for lameness or illness causing low activity"
- Low diversity: "Low behavioral diversity - monitor for illness"

## Data Requirements

### Minimum Data

- **60% completeness** required for any period analysis
- At least 7 days of data to generate any trends
- Minute-level sampling preferred (1440 samples/day)

### Data Formats

**Temperature Data:**
```python
pd.DataFrame({
    'timestamp': pd.DatetimeIndex,  # Required
    'temperature': float            # Required (°C)
})
```

**Activity Data:**
```python
pd.DataFrame({
    'timestamp': pd.DatetimeIndex,     # Required
    'behavioral_state': str,           # Required ('lying', 'standing', etc.)
    'movement_intensity': float        # Optional (0-1 scale)
})
```

**Alert History:**
```python
[{
    'timestamp': datetime,    # Required
    'alert_type': str,       # Required
    'severity': str,         # Required ('critical' or 'warning')
    'cow_id': str            # Optional
}, ...]
```

## Performance Considerations

- **Processing Time**: Typically <2 seconds for all 4 periods
- **Memory Usage**: Efficient with streaming data processing
- **Data Volume**: Handles 90+ days of minute-level data (130k+ samples)

## Edge Cases

### Insufficient Data

When data completeness < 60%:
- Period returns `None`
- Overall trend shows `INSUFFICIENT_DATA`
- Confidence score is 0.0

### New Animals

Animals with <7 days of history:
- No trends can be generated
- System returns insufficient data status
- Wait for data accumulation

### Data Gaps

Gaps in sensor readings:
- Reduces data completeness percentage
- Lowers confidence scores
- May affect trend classification accuracy

### Sensor Changes

If sensors are replaced mid-period:
- May cause baseline shifts
- Increases variability metrics
- Reduces confidence scores

## Integration with Dashboard

### Visualization Recommendations

**7-Day Trend**
- Real-time health status indicator
- Recent alert timeline
- Current vs baseline temperature gauge

**14-Day Trend**
- Post-treatment recovery progress
- Activity level trends (line chart)
- Alert frequency bar chart

**30-Day Trend**
- Monthly health summary cards
- Behavioral state distribution (pie chart)
- Temperature stability chart

**90-Day Trend**
- Reproductive cycle patterns
- Seasonal health variations
- Long-term improvement/decline indicators

### Color Coding

```python
# Trend indicator colors
IMPROVING: green (#4CAF50)
STABLE: blue (#2196F3)
DETERIORATING: red (#F44336)
INSUFFICIENT_DATA: gray (#9E9E9E)

# Confidence indicator
>80%: solid color
60-80%: lighter shade
<60%: very light/transparent
```

## Troubleshooting

### "Insufficient Data" for all periods

**Causes:**
- Less than 7 days of data available
- Data completeness < 60%
- Empty DataFrames passed

**Solutions:**
- Wait for more data accumulation
- Check data loading logic
- Verify timestamp ranges

### Unexpected Trend Classification

**Causes:**
- Alert frequency heavily weights the score
- Temperature anomalies affect classification
- Low behavioral diversity indicates issues

**Solutions:**
- Review alert thresholds
- Check for sensor calibration issues
- Investigate environmental factors

### Low Confidence Scores

**Causes:**
- Sparse sensor data
- Frequent data gaps
- High metric variability

**Solutions:**
- Improve sensor reliability
- Increase sampling frequency
- Validate data quality

## Examples

See `examples/trend_tracker_example.py` for complete working examples including:
- Synthetic data generation
- Recovery pattern simulation
- Report formatting
- Dashboard JSON export

## API Reference

### MultiDayHealthTrendTracker

```python
class MultiDayHealthTrendTracker:
    def __init__(self, temperature_baseline: float = 38.5)

    def analyze_trends(
        self,
        cow_id: str,
        temperature_data: pd.DataFrame,
        activity_data: pd.DataFrame,
        alert_history: List[Dict[str, Any]],
        behavioral_states: pd.DataFrame,
        analysis_date: Optional[datetime] = None
    ) -> HealthTrendReport
```

### HealthTrendReport

```python
@dataclass
class HealthTrendReport:
    cow_id: str
    analysis_timestamp: datetime
    trend_7day: Optional[TimeWindowMetrics]
    trend_14day: Optional[TimeWindowMetrics]
    trend_30day: Optional[TimeWindowMetrics]
    trend_90day: Optional[TimeWindowMetrics]
    overall_trend: TrendIndicator
    overall_confidence: float
    period_comparisons: Dict[str, Dict[str, Any]]
    significant_changes: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]
```

### TimeWindowMetrics

```python
@dataclass
class TimeWindowMetrics:
    period_days: int
    start_date: datetime
    end_date: datetime
    data_completeness: float
    temperature_mean: float
    temperature_std: float
    temperature_baseline_drift: float
    temperature_anomaly_count: int
    total_activity_minutes: float
    activity_level_mean: float
    rest_minutes: float
    activity_diversity: float
    alert_count: int
    alert_severity_distribution: Dict[str, int]
    alert_type_distribution: Dict[str, int]
    state_distribution: Dict[str, float]
    state_changes_per_day: float
    trend_indicator: TrendIndicator
    confidence_score: float
```

## Future Enhancements

Potential improvements for future versions:

1. **Machine Learning Integration**: Use Layer 2 trend analyzer predictions
2. **Comparative Benchmarking**: Compare against herd averages
3. **Seasonal Adjustments**: Account for seasonal baseline variations
4. **Reproductive Cycle Integration**: Sync with estrus/pregnancy tracking
5. **Custom Alert Weights**: Configure alert severity impact on scoring
6. **Historical Trending**: Track trend changes over extended periods
7. **Multi-Animal Analysis**: Herd-level trend aggregation

## References

- Layer 2 Physiological Analysis: `src/layer2_physiological/`
- Layer 1 Behavioral Classification: `src/layer1_behavior/`
- Immediate Alert Detection: `src/health_intelligence/alerts/`
- Test Suite: `tests/test_trend_tracker.py`
