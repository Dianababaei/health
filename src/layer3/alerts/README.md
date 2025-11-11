# Pattern Alert Detection System

## Overview

The Pattern Alert Detection System identifies reproductive health events (estrus and pregnancy) using sliding window analysis of temperature and activity data. It implements multi-minute confirmation periods and confidence scoring to distinguish true reproductive patterns from illness or sensor anomalies.

## Components

### PatternAlertDetector

Main detection engine that analyzes sensor data to identify estrus and pregnancy patterns.

**Key Features:**
- Sliding window analysis with configurable sizes
- Multi-source data integration (temperature, activity, baselines)
- Confidence scoring that increases as patterns persist
- Temperature rise validation to distinguish reproductive patterns from fever
- Alert state management (pending → confirmed)

### Alert Types

#### 1. Estrus Detection

**Pattern Criteria:**
- Temperature rise: 0.3-0.6°C above baseline
- Activity increase: >15% above baseline
- Gradual temperature rise rate: <0.15°C/minute
- Positive temperature-activity correlation

**Detection Window:**
- Default: 10 minutes (configurable 5-10 minutes)
- Update frequency: Every minute
- Confirmation threshold: 5 minutes of pattern persistence

**Confidence Scoring:**
- Base confidence: 0.7
- Bonus for optimal temperature rise (0.4-0.5°C): +0.1
- Bonus for strong temp-activity correlation (>0.5): +0.1
- Bonus for pattern duration ≥5 minutes: +0.1

#### 2. Pregnancy Indication

**Pattern Criteria:**
- Temperature stability: CV < 0.05
- Activity reduction: >10% below baseline
- Prior estrus detection: 7-30 days before
- Pattern persistence: 7-14 day window

**Detection Window:**
- Default: 14 days (configurable 7-30 days)
- Update frequency: Hourly
- Confirmation threshold: 10 days of observation

**Confidence Scoring:**
- Base confidence: 0.5
- Bonus for 7+ days observation: +0.1
- Bonus for 10+ days observation: +0.1
- Bonus for 14+ days observation: +0.1
- Bonus for very stable temperature (CV < 0.03): +0.1
- Bonus for strong activity reduction (>15%): +0.1

### Temperature Rise Validation

The system distinguishes estrus from fever using multiple criteria:

| Metric | Estrus | Fever |
|--------|--------|-------|
| Temperature rise | 0.3-0.6°C | >0.8°C |
| Rise rate | Gradual (<0.15°C/min) | Rapid (>0.15°C/min) |
| Activity | Increased (+15-30%) | Decreased (-30-50%) |
| Temp-Activity correlation | Positive (>0) | Negative or zero |
| Pattern duration | 5-10 minutes consistent | Variable/inconsistent |

## Usage

### Basic Estrus Detection

```python
from layer3.alerts.pattern_detector import PatternAlertDetector
import pandas as pd

# Initialize detector
detector = PatternAlertDetector(
    estrus_window_minutes=10,
    pregnancy_window_days=14
)

# Prepare data
temperature_data = pd.DataFrame({
    'timestamp': timestamps,
    'temperature': temp_values
})

activity_data = pd.DataFrame({
    'timestamp': timestamps,
    'movement_intensity': activity_values,
    'behavioral_state': states
})

# Detect patterns
alerts = detector.detect_patterns(
    cow_id='cow_001',
    temperature_data=temperature_data,
    activity_data=activity_data,
    baseline_temp=38.5,
    activity_baseline={'mean': 0.5, 'std': 0.1}
)

# Process alerts
for alert in alerts:
    print(f"Alert: {alert.alert_type}")
    print(f"Confidence: {alert.confidence:.2f}")
    print(f"Status: {alert.status}")
```

### Pregnancy Detection with Estrus History

```python
# Record estrus history
estrus_history = [
    {
        'timestamp': estrus_datetime,
        'alert_id': 'estrus_001',
        'cow_id': 'cow_001'
    }
]

# Detect patterns (now includes pregnancy detection)
alerts = detector.detect_patterns(
    cow_id='cow_001',
    temperature_data=temperature_data,
    activity_data=activity_data,
    baseline_temp=38.5,
    activity_baseline={'mean': 0.5, 'std': 0.1},
    estrus_history=estrus_history
)

# Filter pregnancy alerts
pregnancy_alerts = [
    a for a in alerts 
    if a.alert_type == 'pregnancy_indication'
]
```

### Alert Management

```python
# Get active alerts for a cow
cow_alerts = detector.get_active_alerts('cow_001')

# Get all active alerts
all_alerts = detector.get_active_alerts()

# Update alert status
detector.update_alert_status('alert_id', AlertStatus.CONFIRMED)

# Clear resolved alerts
detector.clear_resolved_alerts('cow_001')
```

## Alert Data Structure

Each `PatternAlert` contains:

```python
{
    'alert_id': str,              # Unique identifier
    'timestamp': datetime,         # Detection time
    'cow_id': str,                # Animal ID
    'alert_type': str,            # 'estrus' or 'pregnancy_indication'
    'confidence': float,          # 0.0-1.0
    'detection_window': {
        'start': datetime,
        'end': datetime,
        'duration': str
    },
    'pattern_metrics': {
        'temp_rise': float,           # For estrus
        'activity_increase': float,   # For estrus
        'temp_cv': float,             # For pregnancy
        'activity_reduction': float,  # For pregnancy
        # ... other metrics
    },
    'supporting_data': {
        'temperature_trend': list,
        'activity_trend': list,
        'baseline_temp': float,
        'data_completeness': float
    },
    'status': str,                # 'pending', 'confirmed', 'resolved'
    'related_events': list        # Linked event IDs
}
```

## Configuration

### Window Sizes

Adjust detection windows based on data frequency and requirements:

```python
# Short estrus window (5 minutes)
detector = PatternAlertDetector(estrus_window_minutes=5)

# Extended pregnancy window (30 days)
detector = PatternAlertDetector(pregnancy_window_days=30)
```

### Detection Thresholds

Thresholds are class constants that can be modified:

```python
# Adjust estrus thresholds
PatternAlertDetector.ESTRUS_TEMP_RISE_MIN = 0.25  # Lower threshold
PatternAlertDetector.ESTRUS_ACTIVITY_INCREASE_MIN = 0.12  # Lower threshold

# Adjust pregnancy thresholds
PatternAlertDetector.PREGNANCY_TEMP_STABILITY_CV_MAX = 0.06  # More lenient
PatternAlertDetector.PREGNANCY_ACTIVITY_REDUCTION_MIN = 0.08  # Lower threshold
```

## Performance Considerations

### Data Requirements

**Estrus Detection:**
- Minimum: 80% data completeness over detection window
- Recommended: 1 sample per minute
- Window size: 5-10 minutes

**Pregnancy Detection:**
- Minimum: 75% data completeness over detection window
- Recommended: 1 sample per minute (1440 samples/day)
- Window size: 7-14 days minimum

### Memory Usage

The detector maintains active alerts in memory. For large herds:
- Clear resolved alerts periodically
- Implement external storage for historical alerts
- Use batch processing for offline analysis

### Processing Time

- Estrus detection: <100ms per cow
- Pregnancy detection: <500ms per cow
- Scales linearly with data volume

## Integration with Reproductive Cycle Tracker

Pattern alerts should be recorded in the `ReproductiveCycleTracker` for:
- Estrus cycle prediction
- Estrus-pregnancy linkage
- Reproductive history tracking

```python
from layer3.reproductive_cycle_tracker import ReproductiveCycleTracker

tracker = ReproductiveCycleTracker()

# Record detected estrus
for alert in alerts:
    if alert.alert_type == 'estrus':
        tracker.record_estrus(
            cow_id=alert.cow_id,
            event_id=alert.alert_id,
            timestamp=alert.timestamp,
            confidence=alert.confidence
        )
    elif alert.alert_type == 'pregnancy_indication':
        tracker.record_pregnancy(
            cow_id=alert.cow_id,
            event_id=alert.alert_id,
            timestamp=alert.timestamp,
            confidence=alert.confidence,
            linked_estrus_id=alert.related_events[0] if alert.related_events else None
        )
```

## Testing

Comprehensive unit tests are provided in `tests/test_pattern_alerts.py`:

```bash
python -m pytest tests/test_pattern_alerts.py -v
```

Test coverage includes:
- Valid estrus pattern detection
- Temperature rise vs fever distinction
- Pregnancy detection with estrus linkage
- Confidence scoring validation
- Alert status transitions
- Edge cases and error handling

## Future Enhancements

Potential improvements:
- Machine learning-based pattern recognition
- Individual cow baseline adaptation
- Multi-day estrus pattern tracking
- Pregnancy progression monitoring
- Integration with veterinary systems
- Real-time dashboard visualization
