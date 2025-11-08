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

Components:
- engine: Main simulation orchestrator with time-stepping mechanism
- state_params: Behavioral state parameter definitions and sensor signatures
- transitions: State transition probability matrices and duration models
- noise: Noise generation and individual variation parameters
- temporal: Circadian rhythm and time-of-day pattern management
"""

from .engine import SimulationEngine
from .state_params import BehavioralState, SensorSignature, AnimalProfile
from .transitions import StateTransitionModel
from .noise import NoiseGenerator
from .temporal import TemporalPatternManager

__all__ = [
    'SimulationEngine',
    'BehavioralState',
    'SensorSignature',
    'AnimalProfile',
    'StateTransitionModel',
    'NoiseGenerator',
    'TemporalPatternManager',
]
