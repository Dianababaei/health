# Circadian Rhythm & Daily Activity Patterns Implementation

## Overview

This document describes the implementation of circadian rhythm modeling and daily activity patterns for the synthetic data generator. The implementation enables realistic simulation of livestock behavior and physiological patterns over 24-hour cycles.

## Features

### 1. Circadian Temperature Modeling

The system implements sinusoidal temperature variation over a 24-hour period using the formula:

```
temperature = base_temp + amplitude * sin(2π * (hour - acrophase) / 24)
```

**Parameters:**
- `base_temp`: Base body temperature (~38.5°C for livestock)
- `amplitude`: Temperature variation amplitude (0.5-1.0°C)
- `acrophase`: Hour of peak temperature (typically 14-18h)

**Characteristics:**
- Lower temperatures during night hours (00:00-06:00)
- Peak temperatures in afternoon (14:00-18:00)
- Smooth sinusoidal variation throughout the day
- Small random noise added for realism

### 2. Time-of-Day Activity Schedules

Five distinct time periods with characteristic behavior distributions:

#### Night (00:00-06:00)
- **Primary behavior:** Lying (80%)
- **Secondary:** Ruminating (15%)
- **Minimal activity:** Standing (5%)
- **Characteristics:** Rest period, minimal movement

#### Morning (06:00-10:00)
- **Active period:** Feeding (30%), Ruminating (25%)
- **Moderate activity:** Standing (20%), Walking (15%)
- **Less rest:** Lying (10%)
- **Characteristics:** Transition to active state, morning feeding

#### Midday (10:00-14:00)
- **Rest period:** Lying (50%), Ruminating (30%)
- **Moderate:** Standing (20%)
- **No active behaviors:** Walking/Feeding (0%)
- **Characteristics:** Rest and digestion period

#### Afternoon (14:00-18:00)
- **Balanced activity:** Feeding (25%), Walking (20%), Standing (20%), Ruminating (20%)
- **Some rest:** Lying (15%)
- **Characteristics:** Peak activity period

#### Evening (18:00-24:00)
- **Wind down:** Lying (40%), Ruminating (30%)
- **Moderate:** Standing (20%)
- **Minimal:** Feeding (10%)
- **Characteristics:** Transition to rest

### 3. Daily Sequence Templates

Three predefined templates for different activity levels:

#### Typical Day
- Balanced activity throughout the day
- 6 hours of night lying
- Multiple feeding periods (morning, midday, evening)
- Regular rumination intervals
- Moderate walking activity

#### High Activity Day
- Shorter night rest (5 hours)
- Extended feeding periods
- More frequent walking
- Higher overall activity levels
- Shorter rest periods

#### Low Activity Day
- Full night rest (6 hours)
- Reduced feeding times
- Minimal walking
- Extended rest periods
- More lying behavior throughout day

### 4. Probabilistic Sequence Generation

In addition to template-based generation, the system supports probabilistic behavior sampling:

**Method:** `generate_probabilistic_sequence()`

**Features:**
- Samples behaviors according to time-of-day schedules
- Uses behavior transition matrix for realistic transitions
- Respects minimum and maximum behavior durations
- Combines transition probabilities (60%) with time-of-day schedules (40%)

**Behavior Transition Matrix:**
- Ensures realistic behavior changes
- Example: Feeding → Ruminating (40% probability)
- Example: Lying → Standing (40% probability)

### 5. Smooth Behavior Transitions

**Transition Logic:**
1. Current behavior influences next behavior choice
2. Time-of-day schedule modulates transition probabilities
3. Duration constraints prevent unrealistic short/long behaviors
4. Weighted combination ensures both realism and circadian adherence

**Duration Constraints:**

| Behavior    | Min Duration | Max Duration |
|-------------|--------------|--------------|
| Lying       | 10 min       | 180 min      |
| Standing    | 5 min        | 60 min       |
| Walking     | 5 min        | 60 min       |
| Feeding     | 10 min       | 90 min       |
| Ruminating  | 10 min       | 120 min      |

## Implementation Details

### Key Classes and Methods

#### `SyntheticDataGenerator`

**Initialization:**
```python
generator = SyntheticDataGenerator(random_seed=42)
```

**Key Methods:**

1. **`calculate_circadian_temperature(hour, base_temp, amplitude, acrophase)`**
   - Calculates temperature for a given hour
   - Applies circadian rhythm formula
   - Adds realistic noise

2. **`add_circadian_rhythm(df, timestamp_col, temp_col)`**
   - Applies circadian rhythm to existing dataframe
   - Converts timestamps to hours
   - Generates temperature column

3. **`generate_daily_sequence(template, randomize, randomization_factor)`**
   - Generates behavior sequence from template
   - Optional randomization for variation
   - Returns list of (start, end, behavior) tuples

4. **`generate_probabilistic_sequence(duration_minutes, start_behavior)`**
   - Generates sequence using probabilistic sampling
   - Follows time-of-day schedules
   - Uses transition matrix for behavior changes

5. **`generate_dataset(num_days, start_date, sampling_interval_minutes, ...)`**
   - Complete dataset generation
   - Combines all features
   - Returns pandas DataFrame

6. **`generate_sensor_data(behavior, num_samples)`**
   - Generates accelerometer and gyroscope data
   - Behavior-specific sensor characteristics
   - Returns realistic sensor readings

### Configuration File: `config/behavior_patterns.py`

**Contents:**
- `BEHAVIORS`: List of all behavior types
- `HOURLY_SCHEDULE`: Hour-to-schedule mapping
- `SEQUENCE_TEMPLATES`: Predefined daily sequences
- `TRANSITION_MATRIX`: Behavior transition probabilities
- `MIN_BEHAVIOR_DURATION`: Minimum durations per behavior
- `MAX_BEHAVIOR_DURATION`: Maximum durations per behavior

## Usage Examples

### Basic Dataset Generation

```python
from src.data.synthetic_generator import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator(random_seed=42)

# Generate 3 days of data
df = generator.generate_dataset(
    num_days=3,
    animal_id='COW_001',
    sequence_type='probabilistic'
)

# Save to CSV
df.to_csv('synthetic_data.csv', index=False)
```

### Template-Based Generation

```python
# Generate high-activity day
df = generator.generate_dataset(
    num_days=1,
    sequence_type='template',
    template='high_activity'
)
```

### Custom Circadian Parameters

```python
# Set custom parameters
generator.set_circadian_parameters(
    base_temp=38.7,
    amplitude=0.9,
    acrophase=17
)

df = generator.generate_dataset(num_days=1)
```

### Apply Circadian Rhythm to Existing Data

```python
import pandas as pd

# Create dataframe with timestamps
df = pd.DataFrame({
    'timestamp': pd.date_range('2024-01-01', periods=1440, freq='1min')
})

# Apply circadian temperature
df = generator.add_circadian_rhythm(df)
```

## Validation & Success Criteria

### ✅ Temperature Patterns
- Clear 24-hour sinusoidal pattern
- Variation of 0.5-1.0°C
- Peak temperatures at 14:00-18:00
- Lower temperatures at night (00:00-06:00)

### ✅ Behavior Patterns
- Nighttime data >70% lying behavior
- Feeding peaks during morning/afternoon
- Activity levels correlate with time of day
- Smooth transitions between behaviors

### ✅ Multi-Day Consistency
- Consistent circadian patterns day-over-day
- Temperature peaks align across days
- Similar behavior distributions per time period

### ✅ Randomization
- Not rigid schedules
- Probabilistic variation
- Realistic duration variations
- Natural behavior sequences

## Testing

Run the demo script to validate implementation:

```bash
python examples/demo_circadian_generation.py
```

**Demo outputs:**
1. Circadian temperature visualization
2. Daily sequence template comparisons
3. Time-of-day behavior analysis
4. Multi-day consistency validation
5. Complete dataset generation

## Future Enhancements

Potential improvements for future versions:

1. **Seasonal Variations**: Adjust circadian parameters by season
2. **Individual Differences**: Animal-specific circadian profiles
3. **Health-Related Patterns**: Fever, heat stress simulation
4. **Weather Effects**: Temperature correlation with ambient conditions
5. **Estrus Detection**: Specific patterns for reproductive cycles
6. **Multi-Animal Coordination**: Herd behavior patterns

## References

- Circadian rhythm formula based on livestock physiological studies
- Activity schedules derived from typical livestock behavior patterns
- Behavior transition probabilities based on natural behavior sequences

## Dependencies

- `numpy`: Numerical computations and random sampling
- `pandas`: Data structure and manipulation
- Python standard library: `datetime`, `typing`

## File Structure

```
├── src/
│   └── data/
│       ├── __init__.py
│       └── synthetic_generator.py    # Main implementation
├── config/
│   └── behavior_patterns.py          # Configuration
├── examples/
│   └── demo_circadian_generation.py  # Demo script
└── docs/
    └── circadian_rhythm_implementation.md  # This file
```
