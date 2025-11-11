# Multi-Day Health Trend Tracker

## Quick Start

```python
from health_intelligence import MultiDayHealthTrendTracker
import pandas as pd
from datetime import datetime, timedelta

# Initialize tracker
tracker = MultiDayHealthTrendTracker(temperature_baseline=38.5)

# Prepare your data (pandas DataFrames with timestamp column)
temperature_data = pd.DataFrame({...})  # 'timestamp', 'temperature'
activity_data = pd.DataFrame({...})      # 'timestamp', 'behavioral_state', 'movement_intensity'
alert_history = [...]                     # List of alert dicts
behavioral_states = activity_data.copy()

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

# Export for dashboard
dashboard_data = report.to_dict()
```

## Features

✓ **Multi-Period Analysis** - 7, 14, 30, 90-day windows
✓ **Multi-Factor Integration** - Temperature, activity, alerts, behavior
✓ **Trend Classification** - Improving, stable, or deteriorating
✓ **Confidence Scoring** - Data quality metrics
✓ **Dashboard-Ready Output** - JSON-serializable reports
✓ **Comparative Analysis** - Period-over-period deltas

## Output Structure

```json
{
  "cow_id": "COW_001",
  "overall_trend": "improving",
  "overall_confidence": 0.85,
  "trend_7day": {
    "temperature": {
      "mean": 38.6,
      "std": 0.15,
      "baseline_drift": 0.1,
      "anomaly_count": 2
    },
    "activity": {
      "total_minutes": 5000,
      "mean_level": 0.55,
      "rest_minutes": 5080,
      "diversity": 0.85
    },
    "alerts": {
      "count": 3,
      "severity_distribution": {"warning": 2, "critical": 1}
    },
    "trend_indicator": "improving",
    "confidence": 0.87
  },
  "period_comparisons": {
    "7d_vs_14d": {
      "temperature_delta": -0.2,
      "activity_delta": 0.15,
      "alert_delta": -5
    }
  },
  "significant_changes": [
    "Temperature drift: +0.30°C from baseline"
  ],
  "recommendations": [
    "Positive health trend - maintain current care"
  ]
}
```

## Trend Indicators

| Indicator | Description | Action |
|-----------|-------------|--------|
| `improving` | Health metrics improving | Continue care |
| `stable` | Consistent healthy patterns | Routine monitoring |
| `deteriorating` | Declining health indicators | Veterinary review |
| `insufficient_data` | Not enough data | Wait for accumulation |

## Data Requirements

**Minimum:**
- 60% data completeness
- 7 days for shortest period
- Minute-level sampling preferred

**Temperature Data:**
```python
pd.DataFrame({
    'timestamp': DatetimeIndex,
    'temperature': float  # °C
})
```

**Activity Data:**
```python
pd.DataFrame({
    'timestamp': DatetimeIndex,
    'behavioral_state': str,        # 'lying', 'standing', etc.
    'movement_intensity': float     # 0-1 scale (optional)
})
```

**Alert History:**
```python
[{
    'timestamp': datetime,
    'alert_type': str,      # 'fever', 'heat_stress', etc.
    'severity': str,        # 'critical' or 'warning'
    'cow_id': str
}, ...]
```

## Examples

See:
- `examples/trend_tracker_example.py` - Full working example
- `docs/trend_tracker_guide.md` - Comprehensive guide
- `tests/test_trend_tracker.py` - Unit tests

## Testing

```bash
# Run unit tests
python tests/test_trend_tracker.py

# Run example
python examples/trend_tracker_example.py
```

## Performance

- Processing time: <2 seconds for all 4 periods
- Memory efficient: Handles 90+ days of minute-level data
- Data volume: 130,000+ samples processed efficiently

## Architecture

```
┌─────────────────────────────────────────┐
│  MultiDayHealthTrendTracker             │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐  ┌──────────┐           │
│  │ 7-Day    │  │ 14-Day   │           │
│  │ Analysis │  │ Analysis │  ...      │
│  └──────────┘  └──────────┘           │
│       │              │                 │
│       ▼              ▼                 │
│  ┌────────────────────────┐           │
│  │ Temperature Metrics     │           │
│  │ Activity Metrics        │           │
│  │ Alert Metrics           │           │
│  │ Behavioral Metrics      │           │
│  └────────────────────────┘           │
│       │                                │
│       ▼                                │
│  ┌────────────────────────┐           │
│  │ Trend Classification    │           │
│  │ Confidence Scoring      │           │
│  └────────────────────────┘           │
│       │                                │
│       ▼                                │
│  HealthTrendReport                     │
│  (Dashboard-ready JSON)                │
└─────────────────────────────────────────┘
```

## Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **Layer 2 Physiological**: Temperature trend analysis (optional)
- **Layer 1 Behavioral**: Activity classifications
- **Health Intelligence Alerts**: Alert logging

## Configuration

```python
# Custom baseline temperature
tracker = MultiDayHealthTrendTracker(temperature_baseline=38.3)

# Custom analysis date (default: now)
report = tracker.analyze_trends(
    ...,
    analysis_date=datetime(2025, 11, 1)
)
```

## Troubleshooting

**Q: "Insufficient Data" for all periods**
A: Ensure you have ≥7 days of data with ≥60% completeness

**Q: Unexpected trend classification**
A: Check alert frequency - it heavily weights the health score

**Q: Low confidence scores**
A: Improve sensor reliability and reduce data gaps

## API Reference

### MultiDayHealthTrendTracker

**`__init__(temperature_baseline: float = 38.5)`**

**`analyze_trends(...) -> HealthTrendReport`**
- `cow_id`: Animal identifier (str)
- `temperature_data`: Temperature DataFrame
- `activity_data`: Activity DataFrame
- `alert_history`: List of alert dicts
- `behavioral_states`: Behavioral state DataFrame
- `analysis_date`: Optional analysis date (datetime)

### HealthTrendReport

**Attributes:**
- `cow_id`: str
- `trend_7day`, `trend_14day`, `trend_30day`, `trend_90day`: Optional[TimeWindowMetrics]
- `overall_trend`: TrendIndicator
- `overall_confidence`: float (0-1)
- `period_comparisons`: Dict
- `significant_changes`: List[str]
- `recommendations`: List[str]

**Methods:**
- `to_dict()`: Convert to JSON-serializable dict

## License

Part of the Artemis Health livestock monitoring system.

## Support

For issues or questions:
- Review `docs/trend_tracker_guide.md`
- Check test cases in `tests/test_trend_tracker.py`
- See example usage in `examples/trend_tracker_example.py`
