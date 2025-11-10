"""
ML Classifier Wrapper Module

This module provides a wrapper for trained ML models used to classify
ruminating and feeding behaviors. Handles model loading, feature extraction,
and prediction with confidence scores.

NOTE: This is designed to work with models from Task #90. If models are not
available, it provides a rule-based fallback using heuristics.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import warnings


class BehavioralState(Enum):
    """Enumeration of cattle behavioral states."""
    LYING = "lying"
    STANDING = "standing"
    WALKING = "walking"
    RUMINATING = "ruminating"
    FEEDING = "feeding"
    TRANSITION = "transition"
    UNCERTAIN = "uncertain"


@dataclass
class MLClassificationResult:
    """Result of ML classification."""
    state: BehavioralState
    confidence: float
    probabilities: Dict[str, float]
    features_used: List[str]
    model_name: str


class MLClassifierWrapper:
    """
    Wrapper for trained ML models for ruminating and feeding classification.
    
    Loads pre-trained models and handles feature extraction and prediction.
    Falls back to rule-based heuristics if models are not available.
    
    Attributes:
        ruminating_model: Trained ruminating classifier
        feeding_model: Trained feeding classifier
        models_available: Whether models were successfully loaded
    """
    
    def __init__(
        self,
        ruminating_model_path: Optional[str] = None,
        feeding_model_path: Optional[str] = None,
        use_fallback: bool = True
    ):
        """
        Initialize ML classifier wrapper.
        
        Args:
            ruminating_model_path: Path to ruminating classifier pickle file
            feeding_model_path: Path to feeding classifier pickle file
            use_fallback: Use rule-based fallback if models not available (default: True)
        """
        self.ruminating_model = None
        self.feeding_model = None
        self.models_available = False
        self.use_fallback = use_fallback
        
        # Try to load models
        if ruminating_model_path:
            self.ruminating_model = self._load_model(ruminating_model_path)
        
        if feeding_model_path:
            self.feeding_model = self._load_model(feeding_model_path)
        
        self.models_available = (
            self.ruminating_model is not None or
            self.feeding_model is not None
        )
        
        if not self.models_available and use_fallback:
            warnings.warn(
                "ML models not available. Using rule-based fallback for "
                "ruminating and feeding detection."
            )
    
    def _load_model(self, model_path: str):
        """
        Load a pickled model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model or None if loading fails
        """
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
            else:
                warnings.warn(f"Model file not found: {model_path}")
                return None
        except Exception as e:
            warnings.warn(f"Failed to load model from {model_path}: {e}")
            return None
    
    def classify_ruminating(
        self,
        features: pd.DataFrame,
        use_model: bool = True
    ) -> MLClassificationResult:
        """
        Classify ruminating behavior using ML model or fallback.
        
        Args:
            features: DataFrame with extracted features
            use_model: Whether to use ML model (if available)
            
        Returns:
            MLClassificationResult with prediction
        """
        if use_model and self.ruminating_model is not None:
            return self._classify_with_model(
                features,
                self.ruminating_model,
                'ruminating'
            )
        else:
            return self._classify_ruminating_fallback(features)
    
    def classify_feeding(
        self,
        features: pd.DataFrame,
        use_model: bool = True
    ) -> MLClassificationResult:
        """
        Classify feeding behavior using ML model or fallback.
        
        Args:
            features: DataFrame with extracted features
            use_model: Whether to use ML model (if available)
            
        Returns:
            MLClassificationResult with prediction
        """
        if use_model and self.feeding_model is not None:
            return self._classify_with_model(
                features,
                self.feeding_model,
                'feeding'
            )
        else:
            return self._classify_feeding_fallback(features)
    
    def _classify_with_model(
        self,
        features: pd.DataFrame,
        model,
        behavior: str
    ) -> MLClassificationResult:
        """
        Classify using trained ML model.
        
        Args:
            features: Feature dataframe
            model: Trained sklearn model
            behavior: Behavior name ('ruminating' or 'feeding')
            
        Returns:
            MLClassificationResult
        """
        try:
            # Get predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)
            
            # Get class names
            if hasattr(model, 'classes_'):
                classes = model.classes_
            else:
                classes = ['negative', 'positive']
            
            # Convert to probabilities dict
            prob_dict = {
                str(cls): float(prob)
                for cls, prob in zip(classes, probabilities[0])
            }
            
            # Determine if behavior is detected
            if predictions[0] == 1 or predictions[0] == 'positive':
                state = BehavioralState(behavior)
                confidence = probabilities[0][1]  # Probability of positive class
            else:
                state = BehavioralState.UNCERTAIN
                confidence = probabilities[0][0]  # Probability of negative class
            
            return MLClassificationResult(
                state=state,
                confidence=float(confidence),
                probabilities=prob_dict,
                features_used=list(features.columns),
                model_name=f"ML_{behavior}_classifier"
            )
        
        except Exception as e:
            warnings.warn(f"ML model prediction failed: {e}. Using fallback.")
            if behavior == 'ruminating':
                return self._classify_ruminating_fallback(features)
            else:
                return self._classify_feeding_fallback(features)
    
    def _classify_ruminating_fallback(
        self,
        features: pd.DataFrame
    ) -> MLClassificationResult:
        """
        Rule-based fallback for ruminating detection.
        
        Uses frequency domain features and rhythmic patterns.
        
        Args:
            features: Feature dataframe (should contain rhythmic features)
            
        Returns:
            MLClassificationResult
        """
        # Look for rhythmic features
        has_mya_freq = 'mya_dominant_frequency' in features.columns
        has_lyg_freq = 'lyg_dominant_frequency' in features.columns
        has_regularity = 'mya_regularity_score' in features.columns or 'lyg_regularity_score' in features.columns
        
        if not (has_mya_freq or has_lyg_freq):
            # No frequency features - can't detect ruminating
            return MLClassificationResult(
                state=BehavioralState.UNCERTAIN,
                confidence=0.0,
                probabilities={'ruminating': 0.0, 'not_ruminating': 1.0},
                features_used=list(features.columns),
                model_name="fallback_ruminating"
            )
        
        # Check frequency ranges (ruminating: 0.67-1.0 Hz)
        ruminating_detected = False
        confidence = 0.0
        
        if has_mya_freq:
            mya_freq = features['mya_dominant_frequency'].iloc[0]
            if 0.67 <= mya_freq <= 1.0:
                ruminating_detected = True
                confidence = 0.7
        
        if has_lyg_freq:
            lyg_freq = features['lyg_dominant_frequency'].iloc[0]
            if 0.67 <= lyg_freq <= 1.0:
                ruminating_detected = True
                confidence = max(confidence, 0.7)
        
        # Boost confidence if regularity is high
        if has_regularity and ruminating_detected:
            if 'mya_regularity_score' in features.columns:
                regularity = features['mya_regularity_score'].iloc[0]
                confidence = min(0.85, confidence + regularity * 0.15)
        
        if ruminating_detected:
            state = BehavioralState.RUMINATING
        else:
            state = BehavioralState.UNCERTAIN
            confidence = 0.3
        
        return MLClassificationResult(
            state=state,
            confidence=confidence,
            probabilities={
                'ruminating': confidence if ruminating_detected else 1.0 - confidence,
                'not_ruminating': 1.0 - confidence if ruminating_detected else confidence
            },
            features_used=list(features.columns),
            model_name="fallback_ruminating"
        )
    
    def _classify_feeding_fallback(
        self,
        features: pd.DataFrame
    ) -> MLClassificationResult:
        """
        Rule-based fallback for feeding detection.
        
        Uses head-down position (pitch angle) and lateral movement.
        
        Args:
            features: Feature dataframe
            
        Returns:
            MLClassificationResult
        """
        feeding_detected = False
        confidence = 0.0
        
        # Check for head-down position (negative pitch)
        has_pitch = 'pitch_angle' in features.columns
        has_head_movement = 'head_movement_intensity' in features.columns
        
        if has_pitch:
            pitch = features['pitch_angle'].iloc[0]
            # Pitch < -0.5 radians (~-28 degrees) indicates head-down
            if pitch < -0.5:
                feeding_detected = True
                confidence = 0.65
                
                # Boost confidence with head movement
                if has_head_movement:
                    head_movement = features['head_movement_intensity'].iloc[0]
                    if head_movement > 10.0:  # degrees/sec
                        confidence = min(0.80, confidence + 0.15)
        
        if feeding_detected:
            state = BehavioralState.FEEDING
        else:
            state = BehavioralState.UNCERTAIN
            confidence = 0.3
        
        return MLClassificationResult(
            state=state,
            confidence=confidence,
            probabilities={
                'feeding': confidence if feeding_detected else 1.0 - confidence,
                'not_feeding': 1.0 - confidence if feeding_detected else confidence
            },
            features_used=list(features.columns),
            model_name="fallback_feeding"
        )
    
    def extract_features_for_ml(
        self,
        sensor_data: pd.DataFrame,
        window_size: int = 5
    ) -> pd.DataFrame:
        """
        Extract features from sensor data for ML inference.
        
        This is a simplified feature extraction. For full feature engineering,
        use src/data_processing/feature_engineering.py
        
        Args:
            sensor_data: DataFrame with raw sensor readings
            window_size: Window size for rolling statistics
            
        Returns:
            DataFrame with extracted features
        """
        features = sensor_data.copy()
        
        # Calculate basic features
        if all(col in features.columns for col in ['fxa', 'mya', 'rza']):
            # Motion intensity
            features['motion_intensity'] = np.sqrt(
                features['fxa']**2 + features['mya']**2 + features['rza']**2
            )
            
            # Pitch angle
            features['pitch_angle'] = np.arcsin(features['rza'].clip(-1, 1))
        
        if all(col in features.columns for col in ['lyg', 'dzg']):
            # Head movement intensity
            features['head_movement_intensity'] = np.sqrt(
                features['lyg']**2 + features['dzg']**2
            )
        
        # Rolling statistics
        for col in ['fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']:
            if col in features.columns:
                features[f'{col}_mean'] = features[col].rolling(
                    window=window_size, min_periods=1
                ).mean()
                features[f'{col}_std'] = features[col].rolling(
                    window=window_size, min_periods=1
                ).std()
        
        return features
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        return {
            'ruminating_model_available': self.ruminating_model is not None,
            'feeding_model_available': self.feeding_model is not None,
            'models_available': self.models_available,
            'use_fallback': self.use_fallback
        }
