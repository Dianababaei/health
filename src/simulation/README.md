# Cattle Behavior Simulation Module

A realistic data simulation engine for generating cattle behavioral sensor data with accurate physiological and movement patterns.

## Overview

This module generates synthetic sensor data that closely mimics real cattle behavior patterns from neck-mounted sensors. It includes:

- **5 Core Behavioral States**: Lying, Standing, Walking, Ruminating, Feeding
- **Stress Behavior Overlays**: Erratic movement patterns
- **Smooth State Transitions**: Gradual sensor value interpolation
- **Realistic Durations**: State-specific duration distributions
- **Sensor Signature Accuracy**: Based on documented cattle behavior literature

## Module Structure

```
simulation/
├── __init__.py           # Package exports
├── states.py             # Behavioral state generators
├── transitions.py        # State transition logic
├── engine.py             # Main simulation engine
├── example_usage.py      # Usage examples
└── README.md            # This file
```

## Behavioral States

### 1. Lying State
- **Rza**: -0.5g to -1.0g (horizontal/tilted body)
- **Motion**: Minimal (±0.05g accelerations, ±5°/s gyroscopes)
- **Temperature**: Baseline ± 0.2°C
- **Duration**: 30-120 minutes per bout

### 2. Standing State
- **Rza**: 0.7g to 0.95g (upright posture)
- **Motion**: Low variance with occasional weight shifting
- **Temperature**: Baseline ± 0.1°C
- **Duration**: 5-30 minutes per bout

### 3. Walking State
- **Fxa**: Rhythmic 0.3-0.8 m/s² at 0.5-1.5 Hz (gait frequency)
- **Rza**: 0.7-0.9g (slight forward tilt)
- **Mya**: Side-to-side oscillation 0.2-0.5 m/s²
- **Gyroscopes**: Moderate angular velocities (10-30°/s)
- **Temperature**: +0.1-0.3°C increase during extended walking
- **Duration**: 2-15 minutes per bout

### 4. Ruminating State
- **Mya**: Chewing oscillations at 40-60 cycles/minute (0.67-1.0 Hz)
- **Lyg**: Head bobbing synchronized with chewing (±10-15°/s)
- **Base Posture**: Can occur during lying OR standing
- **Other Axes**: Maintain base state values
- **Duration**: 20-60 minutes per session

### 5. Feeding State
- **Lyg**: Negative pitch -20° to -45° (head down)
- **Fxa**: Moderate forward movement 0.1-0.3 m/s²
- **Mya**: Chewing pattern with variability
- **Rza**: Standing orientation 0.7-0.9g
- **Duration**: 15-45 minutes per session

### Stress Behavior Overlay
- **Erratic Movement**: High variance (Fxa, Mya > 1.0 m/s², gyroscopes > 40°/s)
- **Irregular Patterns**: Loss of rhythmic characteristics
- **Duration**: 5-20 minutes of elevated indicators

## Quick Start

### Basic Usage

```python
from simulation import SimulationEngine
from datetime import datetime

# Initialize engine
engine = SimulationEngine(
    baseline_temperature=38.5,
    sampling_rate=1.0,  # 1 sample per minute
    random_seed=42
)

# Generate 24 hours of continuous data
df = engine.generate_continuous_data(
    duration_hours=24,
    start_datetime=datetime.now(),
    include_stress=True,
    stress_probability=0.05
)

# Export to CSV
engine.export_to_csv(df, 'output/simulation.csv')
```

### Generate Labeled Training Dataset

```python
# Generate balanced dataset for ML training
df = engine.generate_labeled_dataset(
    samples_per_state=100,
    duration_per_sample_minutes=10,
    include_stress=True
)

print(df['state'].value_counts())
```

### Generate Single State Data

```python
from simulation import BehaviorState

# Generate 30 minutes of walking data
df = engine.generate_single_state_data(
    state=BehaviorState.WALKING,
    duration_minutes=30
)
```

### Multi-Animal Simulation

```python
from simulation import BatchSimulator

batch = BatchSimulator(engine)

# Generate data for 10 animals
df = batch.generate_multi_animal_dataset(
    num_animals=10,
    hours_per_animal=24,
    output_dir='data/simulated/animals',
    individual_files=True
)
```

## State Transitions

The simulation includes realistic state transition logic:

### Transition Times
- **Lying → Standing**: 5-15 seconds (gradual Rza increase)
- **Standing → Walking**: 2-5 seconds (rhythm onset)
- **Walking → Standing**: 2-5 seconds (deceleration)
- **Ruminating transitions**: 1-3 seconds (overlay state)

### Transition Probabilities
States follow realistic transition patterns:
- Lying tends to persist (70% stay lying)
- Walking sessions are shorter (50% stay walking)
- Ruminating sessions are long (70-80% stay ruminating)

### Interpolation Methods
- **Ease-in-out**: Smooth acceleration/deceleration (lying↔standing)
- **Ease-in**: Gradual acceleration (standing→walking)
- **Ease-out**: Gradual deceleration (walking→standing)

## Sensor Data Format

Generated data includes 8 sensor readings per timestamp:

| Column | Description | Units | Typical Range |
|--------|-------------|-------|---------------|
| `timestamp` | Date and time | datetime | - |
| `temperature` | Body temperature | °C | 36.0 - 42.0 |
| `fxa` | Forward-backward acceleration | m/s² | -3.0 to 3.0 |
| `mya` | Lateral acceleration | m/s² | -3.0 to 3.0 |
| `rza` | Vertical acceleration / orientation | g | -1.5 to 1.5 |
| `sxg` | Roll angular velocity | °/s | -100 to 100 |
| `lyg` | Pitch angular velocity | °/s | -100 to 100 |
| `dzg` | Yaw angular velocity | °/s | -100 to 100 |
| `state` | Behavioral state label | string | (optional) |

## Data Validation

The engine includes built-in validation:

```python
# Validate data quality
is_valid, warnings = engine.validate_generated_data(df)

if not is_valid:
    for warning in warnings:
        print(f"Warning: {warning}")

# Get state statistics
stats = engine.get_state_statistics(df)
for state, metrics in stats.items():
    print(f"{state}: {metrics}")
```

## Advanced Features

### Custom Baseline Temperature

```python
# Simulate animal with higher baseline temperature
engine = SimulationEngine(baseline_temperature=39.0)
```

### Higher Sampling Rate

```python
# 6 samples per minute (every 10 seconds)
engine = SimulationEngine(sampling_rate=6.0)
```

### Stress Intensity Control

```python
from simulation import StressBehaviorOverlay

# Apply variable stress intensity
stressed = StressBehaviorOverlay.apply_stress(
    readings,
    stress_intensity=1.5  # 150% normal stress
)
```

### Custom State Sequence

```python
from simulation import StateTransitionManager

manager = StateTransitionManager()

# Generate specific state sequence
sequence = manager.get_state_sequence_probabilities(
    sequence_length=20,
    start_state=BehaviorState.LYING
)
```

## Use Cases

### 1. ML Model Training
Generate labeled datasets with balanced state representation for training behavioral classification models.

### 2. Algorithm Testing
Create controlled test scenarios with specific behavioral patterns to validate analysis algorithms.

### 3. Edge Case Simulation
Generate rare events (stress, rapid transitions) that are difficult to capture in real data.

### 4. Data Augmentation
Expand limited real datasets with synthetic data to improve model generalization.

### 5. System Validation
Verify that downstream analysis pipelines correctly identify known behavioral patterns.

## Implementation Details

### Sensor Signature Generation
Each state generator produces realistic sensor patterns using:
- **Sinusoidal patterns**: For rhythmic behaviors (walking, chewing)
- **Random walks**: For natural variability
- **Clipping**: To enforce physiological constraints
- **Noise injection**: To prevent unrealistic repetition

### State Duration Sampling
Durations are sampled from triangular distributions with:
- Minimum and maximum bounds per state
- Mode positioned at 60% between min and max
- Natural skew toward typical durations

### Within-State Variability
To prevent repetitive patterns:
- Amplitude varies per bout (e.g., different gait speeds)
- Phase offsets for multi-axis rhythms
- Random perturbations added to base patterns
- Occasional "micro-events" (weight shifts, head movements)

## Validation Against Literature

All sensor ranges are based on documented cattle behavior research:

- ✅ Lying Rza matches accelerometer studies (-0.5 to -1.0g)
- ✅ Walking gait frequencies align with biomechanics literature (0.5-1.5 Hz)
- ✅ Rumination rates match ethological observations (40-60 cycles/min)
- ✅ Feeding head angles consistent with grazing posture studies (-20° to -45°)
- ✅ Temperature ranges reflect normal bovine physiology (36-42°C)

## Examples

See `example_usage.py` for comprehensive examples including:
1. Continuous 24-hour simulation
2. Single state generation
3. Labeled dataset creation
4. Multi-animal simulation
5. State signature comparison
6. Data validation

Run examples:
```bash
python -m simulation.example_usage
```

## API Reference

### SimulationEngine

**Constructor:**
```python
SimulationEngine(
    baseline_temperature: float = 38.5,
    sampling_rate: float = 1.0,
    random_seed: Optional[int] = None
)
```

**Methods:**
- `generate_continuous_data(duration_hours, start_datetime, include_stress, stress_probability)`
- `generate_single_state_data(state, duration_minutes, start_datetime)`
- `generate_labeled_dataset(samples_per_state, duration_per_sample_minutes, include_stress)`
- `validate_generated_data(df)`
- `get_state_statistics(df)`
- `export_to_csv(df, filepath, include_state_labels)`

### State Generators

All state generators inherit from `BehavioralStateGenerator`:

**Methods:**
- `generate(duration_minutes, start_time)`: Generate sensor readings
- `sample_duration()`: Sample realistic duration for this state
- `get_typical_duration_range()`: Get (min, max) duration bounds

**Available Generators:**
- `LyingStateGenerator`
- `StandingStateGenerator`
- `WalkingStateGenerator`
- `RuminatingStateGenerator`
- `FeedingStateGenerator`

### StateTransitionManager

**Methods:**
- `get_next_state(current_state)`: Sample next state from probabilities
- `create_transition(from_state, to_state, start_reading, end_reading)`: Interpolate transition
- `is_valid_transition(from_state, to_state)`: Check transition validity
- `get_state_sequence_probabilities(sequence_length, start_state)`: Generate state sequence

## Contributing

When adding new behavioral states or modifying sensor signatures:

1. Reference published literature for sensor ranges
2. Implement within-state variability to prevent repetition
3. Define realistic duration ranges
4. Add appropriate state transitions
5. Include validation checks
6. Update this documentation

## License

[Your license information]

## References

Based on sensor signatures documented in:
- Task #169: Foundational Documentation - Behavioral Sensor Signatures
- Task #78: Design Realistic Data Simulation Engine

## Support

For questions or issues, please [add contact information].
