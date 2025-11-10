"""
Hybrid Behavioral Classification Pipeline

This module implements an integrated classification pipeline that combines:
1. Rule-based classifier (lying, standing, walking)
2. ML models (ruminating, feeding)
3. Stress detection (multi-axis variance)
4. State transition smoothing (temporal consistency)

Pipeline Architecture:
- Sequential routing: rule-based first (fast), ML second (complex)
- Stress detection as supplementary flag
- Confidence-based conflict resolution
- Error handling for missing data and sensor malfunctions

Success Criteria:
- >80% accuracy across all 5 behavioral states
- <1 second processing time per minute of data
- Graceful handling of edge cases
"""

import os
import yaml
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Import from other modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from layer1.rule_based_classifier import RuleBasedClassifier, BehavioralState, ClassificationResult
from classification.ml_classifier_wrapper import MLClassifierWrapper, MLClassificationResult
from classification.stress_detector import StressDetector, StressDetectionResult
from classification.state_transition_smoother import StateTransitionSmoother, SmoothedClassification
from data_processing.feature_engineering import engineer_features


@dataclass
class HybridClassificationResult:
    """Complete result from hybrid pipeline."""
    timestamp: pd.Timestamp
    state: BehavioralState
    confidence: float
    secondary_state: Optional[BehavioralState]
    is_stressed: bool
    stress_score: float
    classification_source: str  # 'rule_based', 'ml_model', 'smoothed'
    original_state: Optional[BehavioralState]
    smoothing_applied: bool
    sensor_quality_flag: bool  # True if sensor malfunction detected
    details: Dict


class HybridClassificationPipeline:
    """
    Integrated behavioral classification pipeline.
    
    Orchestrates multiple classification components:
    - Rule-based classifier for posture-based behaviors
    - ML models for complex behaviors (ruminating, feeding)
    - Stress detector for supplementary stress flags
    - State transition smoother for temporal consistency
    
    Attributes:
        rule_classifier: Rule-based classifier instance
        ml_classifier: ML classifier wrapper instance
        stress_detector: Stress detector instance
        smoother: State transition smoother instance
        config: Pipeline configuration dictionary
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        ruminating_model_path: Optional[str] = None,
        feeding_model_path: Optional[str] = None
    ):
        """
        Initialize hybrid classification pipeline.
        
        Args:
            config_path: Path to YAML configuration file (optional)
            ruminating_model_path: Path to ruminating ML model (optional)
            feeding_model_path: Path to feeding ML model (optional)
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._initialize_components(ruminating_model_path, feeding_model_path)
        
        # Performance tracking
        self.processing_times: List[float] = []
        self.classification_count = 0
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load pipeline configuration from YAML or use defaults.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            'rule_classifier': {
                'min_duration_samples': 2,
                'enable_smoothing': True,
                'enable_rumination': False,
                'enable_feeding': True,
                'sampling_rate': 1.0
            },
            'ml_classifier': {
                'use_fallback': True,
                'confidence_threshold': 0.6
            },
            'stress_detector': {
                'window_size': 5,
                'variance_threshold_sigma': 2.0,
                'min_axes_threshold': 3
            },
            'smoother': {
                'min_duration': 2,
                'window_size': 5,
                'confidence_threshold': 0.6,
                'use_transition_matrix': True,
                'use_sliding_window': True
            },
            'pipeline': {
                'enable_stress_detection': True,
                'enable_smoothing': True,
                'rule_priority': True,  # Rule-based takes priority in conflicts
                'feature_window_size': 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
            except Exception as e:
                warnings.warn(f"Failed to load config from {config_path}: {e}. Using defaults.")
        
        return default_config
    
    def _initialize_components(
        self,
        ruminating_model_path: Optional[str],
        feeding_model_path: Optional[str]
    ):
        """Initialize all pipeline components."""
        # Rule-based classifier
        rule_config = self.config['rule_classifier']
        self.rule_classifier = RuleBasedClassifier(
            min_duration_samples=rule_config['min_duration_samples'],
            enable_smoothing=rule_config['enable_smoothing'],
            enable_rumination=rule_config['enable_rumination'],
            enable_feeding=rule_config['enable_feeding'],
            sampling_rate=rule_config['sampling_rate']
        )
        
        # ML classifier wrapper
        ml_config = self.config['ml_classifier']
        self.ml_classifier = MLClassifierWrapper(
            ruminating_model_path=ruminating_model_path,
            feeding_model_path=feeding_model_path,
            use_fallback=ml_config['use_fallback']
        )
        
        # Stress detector
        stress_config = self.config['stress_detector']
        self.stress_detector = StressDetector(
            window_size=stress_config['window_size'],
            variance_threshold_sigma=stress_config['variance_threshold_sigma'],
            min_axes_threshold=stress_config['min_axes_threshold']
        )
        
        # State transition smoother
        smoother_config = self.config['smoother']
        self.smoother = StateTransitionSmoother(
            min_duration=smoother_config['min_duration'],
            window_size=smoother_config['window_size'],
            confidence_threshold=smoother_config['confidence_threshold'],
            use_transition_matrix=smoother_config['use_transition_matrix'],
            use_sliding_window=smoother_config['use_sliding_window']
        )
    
    def classify_batch(
        self,
        sensor_data: pd.DataFrame,
        sensor_quality_flags: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Classify behavioral states for a batch of sensor data.
        
        Args:
            sensor_data: DataFrame with sensor readings (timestamp, fxa, mya, rza, sxg, lyg, dzg, temperature)
            sensor_quality_flags: Optional DataFrame with malfunction flags
            
        Returns:
            DataFrame with classification results including:
            - state: Primary behavioral state
            - confidence: Confidence score (0-1)
            - is_stressed: Stress flag
            - stress_score: Stress level (0-1)
            - classification_source: Source of classification
            - smoothing_applied: Whether smoothing was applied
        """
        start_time = datetime.now()
        
        # Validate input
        required_cols = ['fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg', 'temperature']
        if not all(col in sensor_data.columns for col in required_cols):
            raise ValueError(f"Missing required sensor columns. Need: {required_cols}")
        
        # Add timestamps if not present
        if 'timestamp' not in sensor_data.columns:
            sensor_data = sensor_data.copy()
            sensor_data['timestamp'] = pd.date_range(
                start='2024-01-01',
                periods=len(sensor_data),
                freq='1min'
            )
        
        # Step 1: Rule-based classification (lying, standing, walking)
        rule_results = self._apply_rule_classifier(sensor_data)
        
        # Step 2: Extract features for ML inference
        features = self._extract_ml_features(sensor_data)
        
        # Step 3: ML classification (ruminating, feeding)
        ml_results = self._apply_ml_classifier(sensor_data, features, rule_results)
        
        # Step 4: Merge classifications (resolve conflicts)
        merged_results = self._merge_classifications(rule_results, ml_results)
        
        # Step 5: Stress detection
        if self.config['pipeline']['enable_stress_detection']:
            stress_results = self._detect_stress(sensor_data)
            merged_results = self._add_stress_flags(merged_results, stress_results)
        else:
            merged_results['is_stressed'] = False
            merged_results['stress_score'] = 0.0
        
        # Step 6: State transition smoothing
        if self.config['pipeline']['enable_smoothing']:
            merged_results = self._apply_smoothing(merged_results)
        else:
            merged_results['smoothing_applied'] = False
        
        # Step 7: Add sensor quality flags
        if sensor_quality_flags is not None:
            merged_results = self._add_quality_flags(merged_results, sensor_quality_flags)
        else:
            merged_results['sensor_quality_flag'] = False
        
        # Update statistics
        elapsed_time = (datetime.now() - start_time).total_seconds()
        self.processing_times.append(elapsed_time)
        self.classification_count += len(sensor_data)
        
        return merged_results
    
    def _apply_rule_classifier(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rule-based classifier to sensor data.
        
        Args:
            sensor_data: Raw sensor data
            
        Returns:
            DataFrame with rule-based classifications
        """
        # Use variance-based classification for better accuracy
        window_size = self.config['pipeline']['feature_window_size']
        
        results = self.rule_classifier.classify_batch_with_variance(
            sensor_data,
            window_size=window_size
        )
        
        # Rename columns for clarity
        results = results.rename(columns={
            'state': 'rule_state',
            'confidence': 'rule_confidence',
            'rule_fired': 'rule_name'
        })
        
        return results
    
    def _extract_ml_features(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for ML inference.
        
        Args:
            sensor_data: Raw sensor data
            
        Returns:
            DataFrame with engineered features
        """
        window_size = self.config['pipeline']['feature_window_size']
        
        # Use feature engineering module
        features = engineer_features(
            sensor_data,
            window_size=window_size,
            sampling_rate=1.0,
            include_rhythmic=True
        )
        
        return features
    
    def _apply_ml_classifier(
        self,
        sensor_data: pd.DataFrame,
        features: pd.DataFrame,
        rule_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply ML classifiers for ruminating and feeding.
        
        Args:
            sensor_data: Raw sensor data
            features: Engineered features
            rule_results: Rule-based classification results
            
        Returns:
            DataFrame with ML classifications
        """
        ml_results = []
        
        for idx in range(len(sensor_data)):
            # Get features for this sample
            if idx < len(features):
                sample_features = features.iloc[idx:idx+1]
            else:
                sample_features = features.iloc[-1:]
            
            # Get rule-based state (to avoid conflicts)
            rule_state = rule_results['rule_state'].iloc[idx]
            
            # Only apply ML for ambiguous or specific cases
            ml_state = None
            ml_confidence = 0.0
            ml_source = None
            
            # Check for ruminating (not while walking)
            if rule_state not in ['walking', 'lying']:
                ruminating_result = self.ml_classifier.classify_ruminating(sample_features)
                if ruminating_result.state == BehavioralState.RUMINATING:
                    if ruminating_result.confidence > self.config['ml_classifier']['confidence_threshold']:
                        ml_state = 'ruminating'
                        ml_confidence = ruminating_result.confidence
                        ml_source = ruminating_result.model_name
            
            # Check for feeding (not while lying)
            if ml_state is None and rule_state != 'lying':
                feeding_result = self.ml_classifier.classify_feeding(sample_features)
                if feeding_result.state == BehavioralState.FEEDING:
                    if feeding_result.confidence > self.config['ml_classifier']['confidence_threshold']:
                        ml_state = 'feeding'
                        ml_confidence = feeding_result.confidence
                        ml_source = feeding_result.model_name
            
            ml_results.append({
                'ml_state': ml_state,
                'ml_confidence': ml_confidence,
                'ml_source': ml_source
            })
        
        ml_df = pd.DataFrame(ml_results)
        return ml_df
    
    def _merge_classifications(
        self,
        rule_results: pd.DataFrame,
        ml_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge rule-based and ML classifications with conflict resolution.
        
        Args:
            rule_results: Rule-based results
            ml_results: ML results
            
        Returns:
            Merged DataFrame with final classifications
        """
        merged = rule_results.copy()
        
        # Add ML columns
        merged['ml_state'] = ml_results['ml_state']
        merged['ml_confidence'] = ml_results['ml_confidence']
        
        # Resolve conflicts (rule vs ML)
        final_states = []
        final_confidences = []
        sources = []
        
        for idx in range(len(merged)):
            rule_state = merged['rule_state'].iloc[idx]
            rule_conf = merged['rule_confidence'].iloc[idx]
            ml_state = merged['ml_state'].iloc[idx]
            ml_conf = merged['ml_confidence'].iloc[idx]
            
            # Priority logic
            if ml_state is not None and ml_conf > rule_conf:
                # ML has high confidence - use it
                final_state = ml_state
                final_conf = ml_conf
                source = 'ml_model'
            elif self.config['pipeline']['rule_priority'] and rule_state in ['lying', 'standing', 'walking']:
                # Rule-based has priority for posture-based behaviors
                final_state = rule_state
                final_conf = rule_conf
                source = 'rule_based'
            elif ml_state is not None:
                # Use ML if available
                final_state = ml_state
                final_conf = ml_conf
                source = 'ml_model'
            else:
                # Fall back to rule-based
                final_state = rule_state
                final_conf = rule_conf
                source = 'rule_based'
            
            final_states.append(final_state)
            final_confidences.append(final_conf)
            sources.append(source)
        
        merged['state'] = final_states
        merged['confidence'] = final_confidences
        merged['classification_source'] = sources
        
        return merged
    
    def _detect_stress(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect stress behaviors.
        
        Args:
            sensor_data: Raw sensor data
            
        Returns:
            DataFrame with stress detection results
        """
        return self.stress_detector.detect_stress_batch(sensor_data)
    
    def _add_stress_flags(
        self,
        classifications: pd.DataFrame,
        stress_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add stress flags to classifications.
        
        Args:
            classifications: Classification results
            stress_results: Stress detection results
            
        Returns:
            DataFrame with stress flags added
        """
        classifications['is_stressed'] = stress_results['is_stressed']
        classifications['stress_score'] = stress_results['stress_score']
        classifications['stress_details'] = stress_results['stress_details']
        
        return classifications
    
    def _apply_smoothing(self, classifications: pd.DataFrame) -> pd.DataFrame:
        """
        Apply state transition smoothing.
        
        Args:
            classifications: Classification results
            
        Returns:
            DataFrame with smoothed states
        """
        smoothed = self.smoother.smooth_batch(
            classifications,
            state_column='state',
            confidence_column='confidence'
        )
        
        # Update with smoothed states
        classifications['original_state'] = smoothed['original_state']
        classifications['state'] = smoothed['smoothed_state']
        classifications['smoothing_applied'] = smoothed['smoothing_applied']
        
        return classifications
    
    def _add_quality_flags(
        self,
        classifications: pd.DataFrame,
        quality_flags: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add sensor quality flags.
        
        Args:
            classifications: Classification results
            quality_flags: Sensor quality/malfunction flags
            
        Returns:
            DataFrame with quality flags added
        """
        # Assume quality_flags has 'sensor_malfunction' column
        if 'sensor_malfunction' in quality_flags.columns:
            classifications['sensor_quality_flag'] = quality_flags['sensor_malfunction']
        else:
            classifications['sensor_quality_flag'] = False
        
        return classifications
    
    def export_results(
        self,
        results: pd.DataFrame,
        output_path: str,
        format: str = 'csv'
    ):
        """
        Export classification results to file.
        
        Args:
            results: Classification results DataFrame
            output_path: Output file path
            format: Output format ('csv' or 'json')
        """
        # Select key columns for export
        export_cols = [
            'timestamp', 'state', 'confidence', 'is_stressed',
            'stress_score', 'classification_source', 'smoothing_applied'
        ]
        
        export_data = results[[col for col in export_cols if col in results.columns]]
        
        if format == 'csv':
            export_data.to_csv(output_path, index=False)
        elif format == 'json':
            export_data.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics."""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        return {
            'total_classifications': self.classification_count,
            'batches_processed': len(self.processing_times),
            'avg_processing_time_seconds': avg_time,
            'avg_time_per_sample_ms': (avg_time / max(1, self.classification_count)) * 1000,
            'rule_classifier_stats': self.rule_classifier.get_statistics(),
            'ml_classifier_info': self.ml_classifier.get_model_info(),
            'smoother_stats': self.smoother.get_statistics()
        }
    
    def reset(self):
        """Reset pipeline state."""
        self.rule_classifier.reset()
        self.smoother.reset()
        self.processing_times = []
        self.classification_count = 0
