"""
Artemis Health Monitoring System

A comprehensive animal health monitoring system using neck-mounted sensors
to track physical behavior, physiological conditions, and health intelligence.

The system analyzes:
- Layer 1: Physical Behavior (posture, activity, movement patterns)
- Layer 2: Physiological Analysis (temperature, circadian rhythm, correlations)
- Layer 3: Health Intelligence (alerts, predictions, health scoring)

Usage:
    Import and use the package with automatic logging initialization:
    
    >>> import src
    >>> # Logging is automatically initialized
    
    Or initialize logging manually:
    
    >>> from src.utils.logger import setup_logging
    >>> setup_logging()
"""

__version__ = '0.1.0'
__author__ = 'Artemis Health Team'

# Optional: Automatically initialize logging when package is imported
# Uncomment the following lines to enable automatic initialization
# from src.utils.logger import ensure_logging_initialized
# ensure_logging_initialized()

# Note: If you prefer manual initialization, keep the above lines commented
# and call setup_logging() explicitly in your main application entry point
