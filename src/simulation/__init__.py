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

from .engine import SimulationEngine
from .state_params import BehavioralState, SensorSignature, AnimalProfile
from .transitions import StateTransitionModel
from .noise import NoiseGenerator
from .temporal import TemporalPatternManager
from .health_events import HealthEventSimulator, HealthEventType
from .label_generator import LabelGenerator
from .dataset_generator import (
    DatasetGenerator,
    generate_short_term_dataset,
    generate_medium_term_dataset,
    generate_long_term_dataset,
    generate_all_datasets
)
from .export import DatasetExporter, DatasetSplitter

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
