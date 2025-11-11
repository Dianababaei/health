# Health Scoring System

Modular health scoring system that calculates a 0-100 health score for cattle from independent component scores (temperature, activity, behavioral patterns, and alert frequency).

## Overview

The health scoring system provides a comprehensive, single-number health metric that combines multiple data sources into an actionable score. The system is designed for:

- **Modularity**: Each component is independently testable and replaceable
- **Configurability**: Weights and formulas can be adjusted via YAML without code changes
- **Extensibility**: Easy integration of custom scoring logic or ML models
- **Transparency**: Detailed breakdowns show contribution of each component

## Score Ranges

| Score | Category | Color | Description |
|-------|----------|-------|-------------|
| 80-100 | Excellent | 🟢 Green | Optimal health, no concerns |
| 60-79 | Good | 🟡 Yellow | Minor concerns, monitor closely |
| 40-59 | Moderate | 🟠 Orange | Health issues requiring attention |
| 0-39 | Poor | 🔴 Red | Critical intervention needed |

## Architecture

### Component-Based Design

The system calculates scores from four independent components, each contributing 0-25 points:

1. **Temperature Stability (30% weight by default)**
   - Deviation from individual baseline
   - Circadian rhythm regularity
   - Fever incident penalties

2. **Activity Level (25% weight by default)**
   - Movement intensity vs baseline
   - Activity duration adequacy
   - Prolonged inactivity penalties

3. **Behavioral Patterns (25% weight by default)**
   - Rumination time adequacy
   - Feeding regularity
   - Stress behavior detection

4. **Alert Frequency (20% weight by default)**
   - Active alert count and severity
   - Alert resolution trends
   - Recent alert history

### Placeholder Formulas

**IMPORTANT**: The current implementation uses placeholder formulas designed for easy replacement. These are simplified heuristics meant as a starting point for custom implementations.

#### Temperature Component
```python
score = 25 - (temp_deviation * 10) - (fever_count * 5) + circadian_bonus
```

#### Activity Component
```python
score = 25 - (activity_deviation * 8) - (inactivity_count * 7) + duration_bonus
```

#### Behavioral Component
```python
score = 25 - (rumination_deficit * 5) - (stress_behavior_count * 10) + diversity_bonus
```

#### Alert Component
```python
score = 25 - (critical_alerts * 10) - (warning_alerts * 5) + resolution_bonus
```

## Installation

The health scoring system is part of the `src.health_intelligence.scoring` module:

```python
from src.health_intelligence.scoring import HealthScorer, HealthScore
```

## Basic Usage

### Simple Score Calculation

```python
from src.health_intelligence.scoring import HealthScorer
import pandas as pd

# Initialize scorer (loads config from config/health_score_weights.yaml)
scorer = HealthScorer()

# Prepare data
temperature_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
    'temperature': [38.5] * 1440
})

activity_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
    'movement_intensity': [0.3] * 1440
})

behavioral_data = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min'),
    'behavioral_state': ['standing'] * 1440
})

# Calculate score
score = scorer.calculate_score(
    cow_id="COW_001",
    temperature_data=temperature_data,
    activity_data=activity_data,
    behavioral_data=behavioral_data,
    baseline_temp=38.5,
    baseline_activity=0.3
)

print(f"Health Score: {score.total_score:.1f}/100 ({score.health_category})")
print(f"Confidence: {score.confidence:.2f}")
```

### With Alert Data

```python
active_alerts = [
    {'severity': 'warning', 'type': 'heat_stress', 'timestamp': datetime.now()},
    {'severity': 'critical', 'type': 'fever', 'timestamp': datetime.now()}
]

score = scorer.calculate_score(
    cow_id="COW_001",
    temperature_data=temperature_data,
    activity_data=activity_data,
    behavioral_data=behavioral_data,
    active_alerts=active_alerts,
    baseline_temp=38.5
)
```

### Score Breakdown

```python
# Get detailed breakdown
breakdown = scorer.get_score_breakdown(score)

for component_name, details in breakdown['components'].items():
    print(f"{component_name}:")
    print(f"  Raw Score: {details['raw_score']}/25")
    print(f"  Weight: {details['weight']}")
    print(f"  Contribution: {details['contribution_to_total']}")
    print(f"  Confidence: {details['confidence']}")
```

## Configuration

### Weight Configuration

Edit `config/health_score_weights.yaml` to adjust component weights:

```yaml
component_weights:
  temperature_stability: 0.30  # 30% of total score
  activity_level: 0.25         # 25% of total score
  behavioral_patterns: 0.25    # 25% of total score
  alert_frequency: 0.20        # 20% of total score
```

**Note**: Weights should sum to 1.0. The system will auto-normalize if they don't.

### Component-Specific Settings

Each component has its own configuration section:

```yaml
temperature_stability:
  optimal_deviation: 0.5       # °C deviation for full points
  max_deviation: 2.0           # °C deviation for zero points
  fever_penalty_per_incident: 5.0
  circadian_bonus_enabled: true

activity_level:
  optimal_deviation: 0.15      # 15% deviation for full points
  max_deviation: 0.50          # 50% deviation for zero points
  inactivity_penalty_per_incident: 7.0

behavioral_patterns:
  optimal_rumination_min: 400  # minutes per day
  optimal_rumination_max: 600
  stress_behavior_penalty: 10.0

alert_frequency:
  critical_alert_penalty: 10.0
  warning_alert_penalty: 5.0
  resolution_bonus_enabled: true
```

### Calculation Settings

```yaml
calculation:
  update_interval_minutes: 60    # Calculate score every hour
  rolling_window_hours: 24       # Use last 24 hours of data
  min_data_completeness: 0.70    # Require 70% data availability
  smoothing_enabled: true
  smoothing_factor: 0.3          # EMA smoothing factor
```

## Advanced Usage

### Custom Scoring Components

Replace default components with custom implementations:

```python
from src.health_intelligence.scoring.components import BaseScoreComponent, ComponentScore

class CustomTemperatureScorer(BaseScoreComponent):
    """Custom temperature scoring with ML model."""
    
    def __init__(self, config=None):
        super().__init__(config)
        # Load your ML model
        self.model = load_my_model()
    
    def get_required_columns(self):
        return ['timestamp', 'temperature']
    
    def calculate_score(self, cow_id, data, **kwargs):
        # Your custom scoring logic
        prediction = self.model.predict(data)
        
        return ComponentScore(
            score=prediction * 25,  # Scale to 0-25
            normalized_score=prediction,
            confidence=0.95,
            details={'model_version': '1.0', 'features_used': [...]}
        )

# Use custom component
custom_component = CustomTemperatureScorer()
scorer = HealthScorer(
    custom_components={'temperature_stability': custom_component}
)
```

### Dynamic Weight Adjustment

```python
# Update weights at runtime
scorer.update_weights({
    'temperature_stability': 0.4,
    'activity_level': 0.3,
    'behavioral_patterns': 0.2,
    'alert_frequency': 0.1
})

# Weights are automatically normalized
print(scorer.weights)
```

### Score Smoothing

Enable smoothing to reduce noise from minute-to-minute variations:

```python
# Calculate scores sequentially with smoothing
previous_score = None

for timestamp in time_range:
    score = scorer.calculate_score(
        cow_id="COW_001",
        temperature_data=get_data_for_timestamp(timestamp),
        previous_score=previous_score
    )
    
    previous_score = score.total_score
    
    print(f"{timestamp}: {score.total_score:.1f} "
          f"(smoothed: {score.metadata.get('smoothing_applied', False)})")
```

## Database Integration

### Storing Scores

The schema includes a `health_scores` table (see `schema.sql`):

```sql
CREATE TABLE health_scores (
    timestamp TIMESTAMPTZ NOT NULL,
    cow_id INTEGER NOT NULL,
    total_score FLOAT,
    temperature_score FLOAT,
    activity_score FLOAT,
    behavioral_score FLOAT,
    alert_score FLOAT,
    metadata JSONB,
    PRIMARY KEY (timestamp, cow_id)
);
```

Convert HealthScore to database record:

```python
# Calculate score
score = scorer.calculate_score(...)

# Get database record
db_record = score.to_database_record()

# Insert into database (using your DB library)
cursor.execute(
    """
    INSERT INTO health_scores 
    (timestamp, cow_id, total_score, temperature_score, 
     activity_score, behavioral_score, alert_score, metadata)
    VALUES (%(timestamp)s, %(cow_id)s, %(total_score)s, 
            %(temperature_score)s, %(activity_score)s, 
            %(behavioral_score)s, %(alert_score)s, %(metadata)s)
    """,
    db_record
)
```

## Testing

### Running Tests

```bash
# Run all health scorer tests
python -m pytest tests/test_health_scorer.py -v

# Run component tests
python -m pytest tests/test_score_components.py -v

# Run with coverage
python -m pytest tests/test_health_scorer.py tests/test_score_components.py --cov=src.health_intelligence.scoring
```

### Test Coverage

The test suite covers:
- ✅ All scoring components individually
- ✅ Score aggregation and weighting
- ✅ Configuration loading and validation
- ✅ Health category classification
- ✅ Score smoothing
- ✅ Custom component integration
- ✅ Edge cases (missing data, extreme values, empty inputs)
- ✅ Score range validation

## Integration Points

### With Alert System

```python
from src.health_intelligence.alerts.immediate_detector import ImmediateAlertDetector

# Detect alerts
detector = ImmediateAlertDetector()
alerts = detector.detect_alerts(sensor_data, cow_id="COW_001")

# Calculate score including alerts
active_alerts = [a.to_dict() for a in alerts if a.status == 'active']
score = scorer.calculate_score(
    cow_id="COW_001",
    temperature_data=temp_data,
    active_alerts=active_alerts
)
```

### With Trend Analysis

```python
from src.layer2_physiological.trend_analysis import MultiDayTrendAnalyzer

# Analyze trends
trend_analyzer = MultiDayTrendAnalyzer()
trends = trend_analyzer.analyze_trends(
    cow_id=1042,
    temperature_data=temp_data,
    activity_data=activity_data
)

# Use trend data in scoring
score = scorer.calculate_score(
    cow_id="COW_001",
    temperature_data=temp_data,
    activity_data=activity_data,
    circadian_score=trends.trends[7].temperature_metrics.circadian_health_score
)
```

### With Behavioral Classification

```python
from src.layer1_behavior.activity_metrics import ActivityTracker

# Calculate activity metrics
tracker = ActivityTracker()
activity_df = tracker.calculate_movement_intensity(raw_sensor_data)

# Use in scoring
score = scorer.calculate_score(
    cow_id="COW_001",
    activity_data=activity_df,
    baseline_activity=0.3
)
```

## Scheduled Scoring

For production deployment, calculate scores on a schedule:

```python
import schedule
import time

def calculate_hourly_scores():
    """Calculate health scores for all animals."""
    scorer = HealthScorer()
    
    for cow_id in get_active_animals():
        # Fetch data for last 24 hours
        temp_data = fetch_temperature_data(cow_id, hours=24)
        activity_data = fetch_activity_data(cow_id, hours=24)
        behavioral_data = fetch_behavioral_data(cow_id, hours=24)
        active_alerts = fetch_active_alerts(cow_id)
        
        # Calculate score
        score = scorer.calculate_score(
            cow_id=cow_id,
            temperature_data=temp_data,
            activity_data=activity_data,
            behavioral_data=behavioral_data,
            active_alerts=active_alerts
        )
        
        # Store in database
        save_health_score(score)
        
        # Trigger alerts for poor health
        if score.health_category == 'poor':
            send_notification(cow_id, score)

# Schedule hourly execution
schedule.every().hour.at(":00").do(calculate_hourly_scores)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Future Enhancements

The system is designed for easy integration of advanced features:

### 1. Machine Learning Models
Replace placeholder formulas with trained ML models:
- Gradient boosting for component scores
- Neural networks for pattern recognition
- Time-series forecasting for trend prediction

### 2. Breed-Specific Scoring
Adjust baselines and thresholds per breed:
```yaml
breed_profiles:
  holstein:
    temperature_baseline: 38.5
    activity_baseline: 0.35
  jersey:
    temperature_baseline: 38.7
    activity_baseline: 0.30
```

### 3. Time-of-Day Weighting
Adjust component weights based on time of day or season

### 4. Multi-Animal Context
Consider herd-level patterns and social interactions

### 5. Predictive Scoring
Forecast health scores 24-48 hours ahead

## Troubleshooting

### Low Scores Despite Good Data

Check component breakdown to identify which component is lowering the score:
```python
breakdown = scorer.get_score_breakdown(score)
for name, details in breakdown['components'].items():
    if details['raw_score'] < 15:  # Low score
        print(f"Issue in {name}: {details['details']}")
```

### Confidence Too Low

Low confidence usually indicates:
- Missing data (check data completeness)
- Data quality issues (check sensor malfunction alerts)
- Insufficient historical baseline

### Weights Not Summing to 1.0

Enable weight validation in config:
```yaml
validation:
  enforce_weight_sum: true
  weight_sum_tolerance: 0.01
```

### Score Not Updating

Check calculation interval and data freshness:
```yaml
calculation:
  update_interval_minutes: 60
  rolling_window_hours: 24
```

## Documentation

- **API Reference**: See docstrings in `health_scorer.py` and component files
- **Configuration Reference**: See comments in `config/health_score_weights.yaml`
- **Test Examples**: See `tests/test_health_scorer.py`

## Support

For issues or questions:
1. Check test files for usage examples
2. Review configuration file comments
3. Examine component details in score breakdown
4. Verify data quality and completeness

## License

Part of the Artemis Health Monitoring System.
