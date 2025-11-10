"""
Behavior Classifier Training Module

This module trains Random Forest and SVM classifiers for binary classification
of cattle behaviors (ruminating and feeding). It includes:
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- 5-fold cross-validation
- Training time tracking
- Model serialization
- Comprehensive logging

Expected data format:
    - Features: All sensor-derived features (motion_intensity, pitch_angle, etc.)
    - Labels: Binary columns 'is_ruminating' and 'is_feeding'
"""

import os
import sys
import time
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, make_scorer
)
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)


class BehaviorClassifierTrainer:
    """
    Trainer class for behavior classification models.
    
    Handles training of Random Forest and SVM classifiers for binary
    classification of cattle behaviors (ruminating and feeding).
    """
    
    def __init__(self, 
                 data_dir: str = 'data/processed',
                 models_dir: str = 'models',
                 search_type: str = 'randomized',
                 n_iter: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing training/validation/test data
            models_dir: Directory to save trained models
            search_type: 'grid' or 'randomized' for hyperparameter search
            n_iter: Number of iterations for RandomizedSearchCV
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.search_type = search_type
        self.n_iter = n_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / 'trained').mkdir(parents=True, exist_ok=True)
        
        # Storage for trained models and results
        self.models = {}
        self.scalers = {}
        self.training_results = {}
        
        logger.info(f"Initialized trainer with search type: {search_type}")
        logger.info(f"Using {cv_folds}-fold cross-validation")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load training, validation, and test datasets.
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Loading datasets...")
        
        train_path = self.data_dir / 'training_features.pkl'
        val_path = self.data_dir / 'validation_features.pkl'
        test_path = self.data_dir / 'test_features.pkl'
        
        # Check if files exist
        for path, name in [(train_path, 'Training'), (val_path, 'Validation'), (test_path, 'Test')]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{name} data not found at {path}. "
                    f"Please run Task #89 (Prepare Training Dataset) first."
                )
        
        train_df = pd.read_pickle(train_path)
        val_df = pd.read_pickle(val_path)
        test_df = pd.read_pickle(test_path)
        
        logger.info(f"Loaded training set: {len(train_df)} samples")
        logger.info(f"Loaded validation set: {len(val_df)} samples")
        logger.info(f"Loaded test set: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for modeling.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column ('is_ruminating' or 'is_feeding')
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Identify feature columns (exclude target columns and metadata)
        exclude_cols = ['is_ruminating', 'is_feeding', 'timestamp', 'cow_id', 
                       'sensor_id', 'behavioral_state', 'sample_id', 'animal_id']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        logger.info(f"Prepared {X.shape[1]} features for {target_col}")
        logger.info(f"Class distribution: {np.bincount(y.astype(int))}")
        
        return X, y
    
    def get_rf_param_grid(self) -> Dict:
        """Get Random Forest hyperparameter search space."""
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None]
        }
    
    def get_svm_param_grid(self) -> Dict:
        """Get SVM hyperparameter search space."""
        return {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced', None]
        }
    
    def train_model(self,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   model_type: str,
                   behavior: str,
                   scale_features: bool = True) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_type: 'rf' or 'svm'
            behavior: 'ruminating' or 'feeding'
            scale_features: Whether to scale features (important for SVM)
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()} for {behavior}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Scale features if requested (essential for SVM)
        if scale_features:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            scaler_key = f"{model_type}_{behavior}"
            self.scalers[scaler_key] = scaler
            logger.info("Features scaled using StandardScaler")
        else:
            X_train_scaled = X_train
            scaler = None
        
        # Initialize base model
        if model_type == 'rf':
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
            param_grid = self.get_rf_param_grid()
        elif model_type == 'svm':
            base_model = SVC(
                random_state=self.random_state,
                probability=True  # Enable probability estimates
            )
            param_grid = self.get_svm_param_grid()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Setup hyperparameter search
        f1_scorer = make_scorer(f1_score, average='binary')
        
        if self.search_type == 'randomized':
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_grid,
                n_iter=self.n_iter,
                cv=self.cv_folds,
                scoring=f1_scorer,
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            logger.info(f"Using RandomizedSearchCV with {self.n_iter} iterations")
        else:  # grid search
            search = GridSearchCV(
                base_model,
                param_grid=param_grid,
                cv=self.cv_folds,
                scoring=f1_scorer,
                n_jobs=-1,
                verbose=1
            )
            logger.info("Using GridSearchCV")
        
        # Fit the model
        logger.info("Starting hyperparameter search...")
        search.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        
        # Get best model
        best_model = search.best_estimator_
        
        # Store results
        results = {
            'model': best_model,
            'scaler': scaler,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'cv_results': search.cv_results_,
            'training_time': training_time,
            'model_type': model_type,
            'behavior': behavior
        }
        
        # Log results
        logger.info(f"\nTraining completed in {training_time:.2f} seconds")
        logger.info(f"Best CV F1-score: {search.best_score_:.4f}")
        logger.info(f"Best parameters:")
        for param, value in search.best_params_.items():
            logger.info(f"  {param}: {value}")
        
        # Store in class
        model_key = f"{model_type}_{behavior}"
        self.models[model_key] = best_model
        self.training_results[model_key] = results
        
        return results
    
    def train_all_models(self, train_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Train all 4 models (RF and SVM for both behaviors).
        
        Args:
            train_df: Training DataFrame
            
        Returns:
            Dictionary of all training results
        """
        behaviors = ['ruminating', 'feeding']
        model_types = ['rf', 'svm']
        
        all_results = {}
        
        for behavior in behaviors:
            target_col = f'is_{behavior}'
            
            # Prepare data
            X_train, y_train = self.prepare_features(train_df, target_col)
            
            for model_type in model_types:
                # Train with scaling for SVM, without for RF
                scale_features = (model_type == 'svm')
                
                results = self.train_model(
                    X_train, y_train,
                    model_type=model_type,
                    behavior=behavior,
                    scale_features=scale_features
                )
                
                model_key = f"{model_type}_{behavior}"
                all_results[model_key] = results
        
        logger.info(f"\n{'='*60}")
        logger.info("All models trained successfully!")
        logger.info(f"{'='*60}")
        
        return all_results
    
    def save_models(self):
        """Save all trained models and scalers to disk."""
        logger.info("\nSaving models to disk...")
        
        for model_key, model in self.models.items():
            model_path = self.models_dir / 'trained' / f'{model_key}_model.pkl'
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_key} to {model_path}")
            
            # Save scaler if exists
            if model_key in self.scalers:
                scaler_path = self.models_dir / 'trained' / f'{model_key}_scaler.pkl'
                joblib.dump(self.scalers[model_key], scaler_path)
                logger.info(f"Saved {model_key} scaler to {scaler_path}")
        
        # Save training results metadata
        results_path = self.models_dir / 'trained' / 'training_results.pkl'
        with open(results_path, 'wb') as f:
            pickle.dump(self.training_results, f)
        logger.info(f"Saved training results to {results_path}")
    
    def save_training_summary(self):
        """Save a human-readable training summary."""
        summary_path = self.models_dir / 'trained' / 'training_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("BEHAVIOR CLASSIFIER TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            for model_key, results in self.training_results.items():
                f.write(f"\n{model_key.upper()}\n")
                f.write("-"*70 + "\n")
                f.write(f"Model Type: {results['model_type'].upper()}\n")
                f.write(f"Behavior: {results['behavior']}\n")
                f.write(f"Training Time: {results['training_time']:.2f} seconds\n")
                f.write(f"Best CV F1-Score: {results['best_cv_score']:.4f}\n")
                f.write(f"\nBest Hyperparameters:\n")
                for param, value in results['best_params'].items():
                    f.write(f"  {param}: {value}\n")
                f.write("\n")
        
        logger.info(f"Saved training summary to {summary_path}")


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train behavior classification models'
    )
    parser.add_argument(
        '--data-dir',
        default='data/processed',
        help='Directory containing training data'
    )
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--search-type',
        choices=['grid', 'randomized'],
        default='randomized',
        help='Hyperparameter search strategy'
    )
    parser.add_argument(
        '--n-iter',
        type=int,
        default=50,
        help='Number of iterations for randomized search'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BehaviorClassifierTrainer(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        search_type=args.search_type,
        n_iter=args.n_iter,
        cv_folds=args.cv_folds
    )
    
    try:
        # Load data
        train_df, val_df, test_df = trainer.load_data()
        
        # Train all models
        results = trainer.train_all_models(train_df)
        
        # Save models and results
        trainer.save_models()
        trainer.save_training_summary()
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info("\nNext steps:")
        logger.info("1. Run evaluate_models.py to evaluate on validation set")
        logger.info("2. Compare model performance and select best models")
        logger.info("3. Perform final evaluation on test set")
        
    except FileNotFoundError as e:
        logger.error(f"\nError: {e}")
        logger.error("\nPlease ensure Task #89 (Prepare Training Dataset) has been completed.")
        logger.error("The following files are required:")
        logger.error("  - data/processed/training_features.pkl")
        logger.error("  - data/processed/validation_features.pkl")
        logger.error("  - data/processed/test_features.pkl")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error during training: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
