"""
Behavioral Classification Module

Integrated classification pipeline combining rule-based, ML, and statistical methods
for cattle behavioral state detection.

Components:
- HybridClassificationPipeline: Main integrated pipeline
- RuleBasedClassifier: Threshold-based detection (from layer1)
- MLClassifierWrapper: ML model wrapper for complex behaviors
- StressDetector: Multi-axis variance stress detection
- StateTransitionSmoother: Temporal consistency filtering

Usage:
    from classification import HybridClassificationPipeline
    
    pipeline = HybridClassificationPipeline(config_path='pipeline_config.yaml')
    results = pipeline.classify_batch(sensor_data)
"""

from .hybrid_pipeline import HybridClassificationPipeline, HybridClassificationResult
from .stress_detector import StressDetector, StressDetectionResult
from .state_transition_smoother import StateTransitionSmoother, SmoothedClassification
from .ml_classifier_wrapper import MLClassifierWrapper, MLClassificationResult

__all__ = [
    'HybridClassificationPipeline',
    'HybridClassificationResult',
    'StressDetector',
    'StressDetectionResult',
    'StateTransitionSmoother',
    'SmoothedClassification',
    'MLClassifierWrapper',
    'MLClassificationResult'
]

__version__ = '1.0.0'
