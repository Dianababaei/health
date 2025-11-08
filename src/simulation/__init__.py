"""
Simulation module for generating realistic cattle behavioral sensor data.

This module provides state generators for different behavioral patterns
(lying, standing, walking, ruminating, feeding) with realistic sensor
signatures based on documented cattle behavior patterns.
"""

from .states import (
    LyingStateGenerator,
    StandingStateGenerator,
    WalkingStateGenerator,
    RuminatingStateGenerator,
    FeedingStateGenerator,
    StressBehaviorOverlay,
)
from .transitions import StateTransitionManager
from .engine import SimulationEngine

__all__ = [
    "LyingStateGenerator",
    "StandingStateGenerator",
    "WalkingStateGenerator",
    "RuminatingStateGenerator",
    "FeedingStateGenerator",
    "StressBehaviorOverlay",
    "StateTransitionManager",
    "SimulationEngine",
]
