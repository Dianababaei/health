# Synthetic Data Generator

Generate realistic synthetic sensor data for animal behavior monitoring.

## Quick Start

```python
from src.data import SyntheticDataGenerator

# Initialize generator
generator = SyntheticDataGenerator()

# Generate single behavior
df = generator.generate(
    behavior='walking',
    duration_minutes=30,
    start_time='2024-01-01 08:00',
    noise_level=0.1,
    seed=42
)

# Generate behavior sequence
df_sequence = generator.generate_sequence([
    {'behavior': 'lying', 'duration': 120},
    {'behavior': 'standing', 'duration': 10},
    {'behavior': 'walking', 'duration': 30}
], start_time='2024-01-01 06:00', seed=42)

# Export to CSV
generator.export_to_csv(df, 'output/walking_data.csv')
```

## Available Behaviors

- `lying` - Animal lying down, minimal movement
- `standing` - Animal standing still
- `walking` - Animal walking with rhythmic gait
- `ruminating` - Animal chewing cud with rhythmic jaw movement
- `feeding` - Animal eating with head down
- `resting` - Animal resting quietly

## Sensor Channels

- `temp` - Body temperature (°C)
- `Fxa` - Forward-backward acceleration (m/s²)
- `Mya` - Lateral acceleration (m/s²)
- `Rza` - Vertical acceleration (m/s²)
- `Sxg` - Roll angular velocity (rad/s)
- `Lyg` - Pitch angular velocity (rad/s)
- `Dzg` - Yaw angular velocity (rad/s)

## Features

- **Realistic Parameters**: Based on physical constraints and animal physiology
- **Gaussian Noise**: Configurable signal-to-noise ratio
- **Frequency Components**: Rhythmic behaviors include sine wave patterns
- **Smooth Transitions**: Sigmoid interpolation between behaviors
- **Reproducible**: Seed-based random generation
- **CSV Export**: ISO timestamp format with metadata
