"""
Model Inference Module

Provides a simple interface for loading trained models and making predictions
on new sensor data. Designed for production deployment.

Usage:
    from src.models.inference import BehaviorPredictor
    
    predictor = BehaviorPredictor()
    predictions = predictor.predict(features_df)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class BehaviorPredictor:
    """
    Production-ready predictor for cattle behavior classification.
    
    Loads trained models and provides simple predict() interface.
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Initialize predictor with trained models.
        
        Args:
            models_dir: Directory containing trained models
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.loaded = False
        
        logger.info(f"Initialized BehaviorPredictor with models_dir: {models_dir}")
    
    def load_models(self):
        """Load trained models and scalers from disk."""
        logger.info("Loading trained models...")
        
        # Load ruminating model
        ruminating_path = self.models_dir / 'ruminating_classifier.pkl'
        if ruminating_path.exists():
            self.models['ruminating'] = joblib.load(ruminating_path)
            logger.info(f"Loaded ruminating model from {ruminating_path}")
            
            # Load scaler if exists
            ruminating_scaler_path = self.models_dir / 'ruminating_scaler.pkl'
            if ruminating_scaler_path.exists():
                self.scalers['ruminating'] = joblib.load(ruminating_scaler_path)
                logger.info("Loaded ruminating scaler")
        else:
            logger.warning(f"Ruminating model not found at {ruminating_path}")
        
        # Load feeding model
        feeding_path = self.models_dir / 'feeding_classifier.pkl'
        if feeding_path.exists():
            self.models['feeding'] = joblib.load(feeding_path)
            logger.info(f"Loaded feeding model from {feeding_path}")
            
            # Load scaler if exists
            feeding_scaler_path = self.models_dir / 'feeding_scaler.pkl'
            if feeding_scaler_path.exists():
                self.scalers['feeding'] = joblib.load(feeding_scaler_path)
                logger.info("Loaded feeding scaler")
        else:
            logger.warning(f"Feeding model not found at {feeding_path}")
        
        if not self.models:
            raise FileNotFoundError(
                f"No trained models found in {self.models_dir}. "
                f"Please train models first or check the path."
            )
        
        self.loaded = True
        logger.info(f"Loaded {len(self.models)} models successfully")
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from DataFrame.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Feature matrix (numpy array)
        """
        # Exclude non-feature columns
        exclude_cols = ['is_ruminating', 'is_feeding', 'timestamp', 'cow_id',
                       'sensor_id', 'behavioral_state', 'sample_id', 'animal_id']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            raise ValueError("No valid feature columns found in DataFrame")
        
        X = df[feature_cols].values
        
        return X
    
    def predict_behavior(self,
                        X: np.ndarray,
                        behavior: str,
                        return_proba: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict a specific behavior.
        
        Args:
            X: Feature matrix
            behavior: 'ruminating' or 'feeding'
            return_proba: If True, return both predictions and probabilities
            
        Returns:
            Predictions array, or (predictions, probabilities) if return_proba=True
        """
        if not self.loaded:
            self.load_models()
        
        if behavior not in self.models:
            raise ValueError(f"Model for '{behavior}' not loaded")
        
        model = self.models[behavior]
        
        # Scale features if scaler exists
        X_scaled = X
        if behavior in self.scalers:
            X_scaled = self.scalers[behavior].transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        if return_proba:
            probabilities = model.predict_proba(X_scaled)[:, 1]
            return predictions, probabilities
        
        return predictions
    
    def predict(self,
               df: pd.DataFrame,
               behaviors: Optional[list] = None) -> pd.DataFrame:
        """
        Predict all loaded behaviors on a DataFrame.
        
        Args:
            df: DataFrame with features
            behaviors: List of behaviors to predict (default: all loaded)
            
        Returns:
            DataFrame with original data plus prediction columns
        """
        if not self.loaded:
            self.load_models()
        
        if behaviors is None:
            behaviors = list(self.models.keys())
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make predictions for each behavior
        result_df = df.copy()
        
        for behavior in behaviors:
            if behavior not in self.models:
                logger.warning(f"Model for '{behavior}' not loaded, skipping")
                continue
            
            predictions, probabilities = self.predict_behavior(
                X, behavior, return_proba=True
            )
            
            result_df[f'{behavior}_prediction'] = predictions
            result_df[f'{behavior}_probability'] = probabilities
        
        return result_df
    
    def predict_single(self, features: Dict[str, float]) -> Dict[str, Dict]:
        """
        Predict behaviors for a single sample.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Dictionary with predictions and probabilities for each behavior
        """
        if not self.loaded:
            self.load_models()
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Make predictions
        results = {}
        
        for behavior in self.models.keys():
            predictions, probabilities = self.predict_behavior(
                X, behavior, return_proba=True
            )
            
            results[behavior] = {
                'prediction': bool(predictions[0]),
                'probability': float(probabilities[0])
            }
        
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        if not self.loaded:
            self.load_models()
        
        info = {}
        
        for behavior, model in self.models.items():
            model_info = {
                'type': type(model).__name__,
                'has_scaler': behavior in self.scalers
            }
            
            # Get feature importances if available (Random Forest)
            if hasattr(model, 'feature_importances_'):
                model_info['has_feature_importances'] = True
                model_info['n_features'] = len(model.feature_importances_)
            
            # Get other model-specific info
            if hasattr(model, 'n_estimators'):  # Random Forest
                model_info['n_estimators'] = model.n_estimators
            
            if hasattr(model, 'C'):  # SVM
                model_info['C'] = model.C
                model_info['kernel'] = model.kernel
            
            info[behavior] = model_info
        
        return info


def quick_predict(features_df: pd.DataFrame, 
                 models_dir: str = 'models') -> pd.DataFrame:
    """
    Quick prediction function for convenience.
    
    Args:
        features_df: DataFrame with features
        models_dir: Directory with trained models
        
    Returns:
        DataFrame with predictions
    """
    predictor = BehaviorPredictor(models_dir)
    return predictor.predict(features_df)


# Example usage
if __name__ == '__main__':
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("BEHAVIOR PREDICTOR - INFERENCE EXAMPLE")
    print("="*70)
    
    try:
        # Initialize predictor
        predictor = BehaviorPredictor()
        predictor.load_models()
        
        # Show model info
        print("\nLoaded Models:")
        info = predictor.get_model_info()
        for behavior, model_info in info.items():
            print(f"\n  {behavior.capitalize()}:")
            for key, value in model_info.items():
                print(f"    {key}: {value}")
        
        # Try loading validation data for demo
        print("\n" + "-"*70)
        print("Testing with validation data...")
        
        data_path = Path('data/processed/validation_features.pkl')
        if data_path.exists():
            df = pd.read_pickle(data_path)
            
            # Predict on first 10 samples
            sample_df = df.head(10)
            results = predictor.predict(sample_df)
            
            print("\nPredictions for first 10 samples:")
            print("-"*70)
            
            for behavior in predictor.models.keys():
                pred_col = f'{behavior}_prediction'
                prob_col = f'{behavior}_probability'
                true_col = f'is_{behavior}'
                
                if true_col in results.columns:
                    correct = (results[pred_col] == results[true_col]).sum()
                    print(f"\n{behavior.capitalize()}:")
                    print(f"  Correct predictions: {correct}/10")
                    print(f"  Sample probabilities: {results[prob_col].values[:5]}")
            
            print("\n" + "="*70)
            print("SUCCESS! Inference working correctly.")
            print("="*70)
        else:
            print(f"\nValidation data not found at {data_path}")
            print("Skipping prediction test.")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure models are trained first:")
        print("  python train_behavior_classifiers.py")
        print("  python evaluate_models.py --save-best")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
