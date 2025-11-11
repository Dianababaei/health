"""
Layer 3: Health Intelligence and Early Warning

Modules for combining behavioral and physiological data to produce
automated health assessments and alerts.

Sub-analyses:
- Instant health detection (fever, heat stress, inactivity alerts)
- Health trend monitoring
- Estrus and pregnancy detection
- Health scoring (0-100 scale)
- Automated alert system
"""

from .reproductive_cycle_tracker import (
    ReproductiveCycleTracker,
    EstrusRecord,
    PregnancyRecord,
    ReproductiveCycleState
)

__all__ = [
    'ReproductiveCycleTracker',
    'EstrusRecord',
    'PregnancyRecord',
    'ReproductiveCycleState'
]
