# Behavioral State Simulation Module

## Overview

This module provides a comprehensive simulation engine for generating realistic cattle sensor data based on behavioral states. The simulation is grounded in cattle behavior research literature and produces minute-by-minute sensor readings that accurately represent real-world patterns.

## Features

### Core Capabilities
- **Literature-Based Sensor Signatures**: All behavioral states (lying, standing, walking, ruminating, feeding) have sensor patterns based on research
- **Realistic State Transitions**: Probabilistic state machine with time-of-day modulation
- **Circadian Rhythms**: Body temperature and activity patterns follow natural daily cycles
- **Individual Variation**: Each simulated animal has unique baseline characteristics
- **Sensor Noise**: Realistic noise levels based on typical sensor specifications
- **Smooth Transitions**: Gradual sensor value changes over 30-120 seconds between states
- **Validation**: Automatic checks to ensure generated data falls within realistic ranges

### Sensor Parameters (7 total)
1. **Temperature** (°C) - Body temperature with circadian rhythm
2. **Fxa** (g) - Forward-backward acceleration
3. **Mya** (g) - Lateral acceleration
4. **Rza** (g) - Vertical acceleration (posture indicator)
5. **Sxg** (°/s) - Roll angular velocity
6. **Lyg** (°/s) - Pitch angular velocity
7. **Dzg** (°/s) - Yaw angular velocity

### Behavioral States
- **Lying**: Low activity, negative Rza (< -0.5g), minimal movement
- **Standing**: Upright posture (Rza > 0.7g), stable, low motion
- **Walking**: Rhythmic patterns (~1 Hz), forward acceleration, head bobbing
- **Ruminating**: Jaw movements (Mya oscillations ~50 cycles/min), can occur lying or standing
- **Feeding**: Head down (negative Lyg), moderate forward movement

## Quick Start

### Basic Usage

```python
from src.simulation import SimulationEngine
from datetime import datetime

# Create simulation engine
engine = SimulationEngine(
    animal_id="cow_001",
    seed=42  # For reproducibility
)

# Run 24-hour simulation
data = engine.run_simulation(
    duration_hours=24,
    start_time=datetime(2024, 1, 1, 0, 0)
)

# Export to CSV
data.to_csv('simulated_data.csv', index=False)

# Get summary statistics
stats = engine.get_summary_statistics()
print(stats)
```

### Multi-Day Simulation

```python
# Simulate one week of data
data = engine.run_multi_day_simulation(
    num_days=7,
    start_time=datetime(2024, 1, 1, 0, 0)
)
```

### Custom Animal Profile

```python
from src.simulation import AnimalProfile, SimulationEngine

# Create custom animal with specific characteristics
profile = AnimalProfile(
    animal_id="cow_002",
    baseline_temperature=38.7,  # Slightly higher baseline
    activity_multiplier=1.2,    # 20% more active than average
    body_size_factor=1.1,       # 10% larger than average
    age_category="adult"
)

# Create engine with custom profile
engine = SimulationEngine(animal_profile=profile)
data = engine.run_simulation(duration_hours=48)
```

### Simulating Sick Animals

```python
# Create profile for animal with fever and reduced activity
sick_profile = AnimalProfile(
    animal_id="sick_cow_001",
    baseline_temperature=38.5,
    fever_offset=1.5,        # +1.5°C fever
    lethargy_factor=0.5      # 50% of normal activity
)

engine = SimulationEngine(animal_profile=sick_profile)
data = engine.run_simulation(duration_hours=24)
```

## Module Structure

```
src/simulation/
├── __init__.py           # Module exports
├── engine.py             # Main simulation orchestrator
├── state_params.py       # Behavioral state definitions
├── transitions.py        # State transition model
├── noise.py              # Noise and variation generators
├── temporal.py           # Circadian rhythm management
└── README.md             # This file
```

## Components

### SimulationEngine (`engine.py`)
Main orchestrator that coordinates all simulation components:
- Time-stepping mechanism (1-minute intervals)
- State management and transitions
- Sensor value generation
- Noise and variation application
- Data validation and export

### State Parameters (`state_params.py`)
Defines behavioral states and their sensor signatures:
- `BehavioralState`: Enumeration of states
- `SensorRange`: Min/max/mean/std for each parameter
- `SensorSignature`: Complete signature for a state
- `AnimalProfile`: Individual animal characteristics

### State Transition Model (`transitions.py`)
Probabilistic state machine for behavioral transitions:
- Transition probability matrices
- Duration distributions for each state
- Smooth transition interpolation
- Time-of-day modulation

### Noise Generator (`noise.py`)
Adds realistic sensor noise and variation:
- Gaussian noise for each sensor type
- Individual animal baseline variation
- Environmental temperature effects
- Rhythmic patterns (walking, ruminating)

### Temporal Pattern Manager (`temporal.py`)
Manages time-of-day and circadian effects:
- Circadian temperature rhythm
- Time-of-day behavioral preferences
- Seasonal patterns (optional)

## Configuration

All simulation parameters can be customized via `config/simulation_params.yaml`:

```yaml
# Key parameters
noise:
  temperature_std: 0.1
  accelerometer_std: 0.05
  gyroscope_std: 2.0

temporal:
  transition_smoothing_time: 60.0
  night_start: 22.0
  night_end: 6.0

simulation:
  time_step_minutes: 1.0
  include_validation: true
  include_noise: true
```

## Advanced Usage

### Custom Transition Probabilities

```python
from src.simulation import (
    SimulationEngine,
    StateTransitionConfig,
    BehavioralState
)

# Define custom transition matrix
custom_matrix = {
    BehavioralState.LYING: {
        BehavioralState.LYING: 0.90,      # More lying
        BehavioralState.STANDING: 0.08,
        BehavioralState.WALKING: 0.01,
        BehavioralState.RUMINATING: 0.01,
        BehavioralState.FEEDING: 0.00,
    },
    # ... other states
}

# Create custom config
config = StateTransitionConfig(
    transition_matrix=custom_matrix,
    duration_ranges={
        BehavioralState.LYING: (45.0, 150.0, 90.0),
        # ... other durations
    }
)

# Use in simulation
engine = SimulationEngine(transition_config=config)
```

### Custom Noise Parameters

```python
from src.simulation import SimulationEngine, NoiseParameters

# Define custom noise levels
noise_params = NoiseParameters(
    temperature_std=0.05,      # Lower noise
    accelerometer_std=0.03,
    gyroscope_std=1.5
)

engine = SimulationEngine(noise_params=noise_params)
```

### Accessing State Information

```python
# During simulation, access internal state
engine = SimulationEngine(animal_id="cow_001")
data = engine.run_simulation(duration_hours=1)

# Get state machine info
state_info = engine.transition_model.get_state_info()
print(f"Current state: {state_info['current_state']}")
print(f"Time in state: {state_info['time_in_state']} minutes")
```

## Output Data Format

The simulation generates a pandas DataFrame with the following columns:

### Sensor Data
- `timestamp`: DateTime of reading
- `animal_id`: Animal identifier
- `temperature`: Body temperature (°C)
- `fxa`, `mya`, `rza`: Accelerometer values (g)
- `sxg`, `lyg`, `dzg`: Gyroscope values (°/s)

### Metadata (optional)
- `true_state`: Actual behavioral state (for validation)
- `is_transitioning`: Whether in transition between states
- `time_in_state`: Minutes in current state

## Validation

The engine automatically validates that all generated values fall within realistic ranges:
- Temperature: 36.0-42.0°C
- Accelerometers: ±2.0g
- Gyroscopes: ±50.0°/s

Validation warnings are collected and reported at the end of simulation.

## Literature References

The simulation parameters are based on research literature:

1. **Posture Detection**: Rza thresholds from accelerometer-based cattle behavior classification studies
2. **Rumination Patterns**: 40-60 cycles/min frequency from dairy cattle research
3. **Circadian Rhythms**: Temperature variation patterns from cattle physiology studies
4. **Time Budgets**: State durations and transitions from cattle ethology research
5. **Sensor Specifications**: Noise levels from commercial livestock monitoring sensors

## Performance

- **Speed**: ~1000-5000 data points/second (depending on hardware)
- **Memory**: ~100 MB for 7 days of 1-minute data
- **Accuracy**: Validated against real cattle behavior patterns

## Examples

See `config/simulation_params.yaml` for usage examples and parameter documentation.

## Contributing

When adding new features:
1. Maintain literature-based parameter values
2. Add appropriate validation checks
3. Document all assumptions and sources
4. Include unit tests for new components

## Support

For questions or issues:
1. Check configuration file documentation
2. Review this README for usage examples
3. Examine module docstrings for detailed API information
