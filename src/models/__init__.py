"""
Behavior Classification Models Module

This module provides machine learning models for binary classification of 
cattle behaviors (ruminating and feeding) using sensor data.

Main Components:
- BehaviorClassifierTrainer: Train RF and SVM models with hyperparameter tuning
- ModelEvaluator: Evaluate models with comprehensive metrics and visualizations

Usage:
    from src.models.train_behavior_classifiers import BehaviorClassifierTrainer
    from src.models.evaluate_models import ModelEvaluator
    
    # Train models
    trainer = BehaviorClassifierTrainer()
    train_df, val_df, test_df = trainer.load_data()
    results = trainer.train_all_models(train_df)
    trainer.save_models()
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluator.load_models()
    evaluator.evaluate_all_models(val_df, 'validation')
    evaluator.generate_all_plots('validation')
    evaluator.generate_report('validation')
"""

__version__ = '1.0.0'

from pathlib import Path

# Define module paths
MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

__all__ = [
    '__version__',
    'MODULE_DIR',
    'PROJECT_ROOT',
    'DATA_DIR',
    'MODELS_DIR',
    'RESULTS_DIR'
]
