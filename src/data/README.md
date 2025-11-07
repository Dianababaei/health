# Data Module

This module contains tools for synthetic data generation with realistic circadian rhythms and daily activity patterns.

## Contents

- **`synthetic_generator.py`**: Main synthetic data generator with circadian rhythm modeling

## Quick Start

```python
from src.data.synthetic_generator import SyntheticDataGenerator

# Create generator
generator = SyntheticDataGenerator(random_seed=42)

# Generate 3 days of synthetic data
df = generator.generate_dataset(
    num_days=3,
    animal_id='COW_001',
    sequence_type='probabilistic'
)

print(df.head())
```

## Features

- **Circadian Temperature Modeling**: Sinusoidal 24-hour temperature variation
- **Time-of-Day Activity Patterns**: Behavior distributions that change throughout the day
- **Multiple Sequence Templates**: Typical, high-activity, and low-activity day templates
- **Probabilistic Generation**: Dynamic behavior sampling based on time and transitions
- **Realistic Sensor Data**: Accelerometer and gyroscope readings for each behavior
- **Multi-Day Generation**: Consistent patterns across multiple days

## Output Format

Generated datasets include the following columns:

| Column      | Description                                    |
|-------------|------------------------------------------------|
| timestamp   | Date and time of measurement                   |
| temperature | Body temperature (°C) with circadian rhythm    |
| Fxa         | Acceleration X-axis (forward-backward)         |
| Mya         | Acceleration Y-axis (lateral)                  |
| Rza         | Acceleration Z-axis (vertical)                 |
| Sxg         | Gyroscope X-axis (roll)                        |
| Lyg         | Gyroscope Y-axis (pitch)                       |
| Dzg         | Gyroscope Z-axis (yaw)                         |
| behavior    | Current behavior label                         |
| animal_id   | Animal identifier (if specified)               |

## Configuration

Behavior patterns and schedules are defined in `config/behavior_patterns.py`:

- Time-of-day activity schedules (5 time periods)
- Daily sequence templates (3 templates)
- Behavior transition probabilities
- Duration constraints

## Documentation

See `docs/circadian_rhythm_implementation.md` for detailed documentation.

## Examples

Run the demo:
```bash
python examples/demo_circadian_generation.py
```
