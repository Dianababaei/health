"""
Behavioral State Simulation

Modules for generating realistic cattle sensor data based on behavioral states.
This simulation engine creates synthetic data that accurately represents the sensor
signatures of different cattle behaviors (lying, standing, walking, ruminating, feeding).

The simulation includes:
- Realistic state transitions with probabilistic models
- Circadian rhythm patterns
- Sensor noise and individual animal variation
- Smooth transitions between behavioral states
- Literature-based sensor signature definitions
- Health event simulation (estrus, pregnancy, illness)
- Ground truth label generation
- Complete dataset generation at multiple time scales

Components:
- engine: Main simulation orchestrator with time-stepping mechanism
- state_params: Behavioral state parameter definitions and sensor signatures
- transitions: State transition probability matrices and duration models
- noise: Noise generation and individual variation parameters
- temporal: Circadian rhythm and time-of-day pattern management
- health_events: Health condition simulators (estrus, pregnancy, illness, heat stress)
- label_generator: Ground truth label generation
- dataset_generator: Complete dataset generation orchestrator
- export: Dataset export and metadata utilities
"""

# Conditional imports - only import if modules exist
try:
    from .engine import SimulationEngine
except ImportError:
    pass

try:
    from .state_params import BehavioralState, SensorSignature, AnimalProfile
except ImportError:
    pass

try:
    from .transitions import StateTransitionManager as StateTransitionModel
except ImportError:
    pass

try:
    from .noise import NoiseGenerator
except ImportError:
    pass

try:
    from .temporal import TemporalPatternManager
except ImportError:
    pass

try:
    from .health_events import HealthEventSimulator, HealthEventType
except ImportError:
    pass

try:
    from .label_generator import LabelGenerator
except ImportError:
    pass

try:
    from .dataset_generator import (
        DatasetGenerator,
        generate_short_term_dataset,
        generate_medium_term_dataset,
        generate_long_term_dataset,
        generate_all_datasets
    )
except ImportError:
    pass

try:
    from .export import DatasetExporter, DatasetSplitter
except ImportError:
    pass

# Always available
from .circadian_rhythm import CircadianRhythmGenerator
from .health_conditions import (
    FeverSimulator,
    HeatStressSimulator,
    EstrusSimulator,
    PregnancySimulator
)

__all__ = [
    'SimulationEngine',
    'BehavioralState',
    'SensorSignature',
    'AnimalProfile',
    'StateTransitionModel',
    'NoiseGenerator',
    'TemporalPatternManager',
    'HealthEventSimulator',
    'HealthEventType',
    'LabelGenerator',
    'DatasetGenerator',
    'generate_short_term_dataset',
    'generate_medium_term_dataset',
    'generate_long_term_dataset',
    'generate_all_datasets',
    'DatasetExporter',
    'DatasetSplitter',
]
