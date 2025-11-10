#!/usr/bin/env python3
"""
Quick Start Script for Model Training and Evaluation

This script provides a simple entry point for training and evaluating
behavior classification models. It runs the complete workflow:
1. Train Random Forest and SVM models
2. Evaluate on validation set
3. Generate reports and visualizations
4. Save best models for production

Usage:
    python train_and_evaluate_models.py [OPTIONS]
    
    Options:
        --quick         Quick training (10 iterations, 3-fold CV)
        --thorough      Thorough training (100 iterations, 10-fold CV)
        --eval-only     Only evaluate existing models
        --test          Run final test set evaluation
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.train_behavior_classifiers import BehaviorClassifierTrainer
from src.models.evaluate_models import ModelEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models(quick=False, thorough=False):
    """
    Train all behavior classification models.
    
    Args:
        quick: Use fewer iterations for quick training
        thorough: Use more iterations for thorough search
    """
    logger.info("="*70)
    logger.info("STEP 1: TRAINING MODELS")
    logger.info("="*70)
    
    # Determine training parameters
    if quick:
        n_iter = 10
        cv_folds = 3
        logger.info("Using QUICK training mode (10 iterations, 3-fold CV)")
    elif thorough:
        n_iter = 100
        cv_folds = 10
        logger.info("Using THOROUGH training mode (100 iterations, 10-fold CV)")
    else:
        n_iter = 50
        cv_folds = 5
        logger.info("Using DEFAULT training mode (50 iterations, 5-fold CV)")
    
    # Initialize trainer
    trainer = BehaviorClassifierTrainer(
        data_dir='data/processed',
        models_dir='models',
        search_type='randomized',
        n_iter=n_iter,
        cv_folds=cv_folds
    )
    
    try:
        # Load data
        logger.info("\nLoading datasets...")
        train_df, val_df, test_df = trainer.load_data()
        
        # Train all models
        logger.info("\nTraining models (this may take 10-60 minutes)...")
        results = trainer.train_all_models(train_df)
        
        # Save models
        logger.info("\nSaving models to disk...")
        trainer.save_models()
        trainer.save_training_summary()
        
        logger.info("\n" + "="*70)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*70)
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: {e}")
        logger.error("\nPlease ensure Task #89 (Prepare Training Dataset) is completed.")
        logger.error("Required files:")
        logger.error("  - data/processed/training_features.pkl")
        logger.error("  - data/processed/validation_features.pkl")
        logger.error("  - data/processed/test_features.pkl")
        return False
    except Exception as e:
        logger.error(f"\n❌ Unexpected error during training: {e}", exc_info=True)
        return False


def evaluate_models(split='validation', save_best=True):
    """
    Evaluate trained models.
    
    Args:
        split: 'validation' or 'test'
        save_best: Whether to save best models to production filenames
    """
    logger.info("="*70)
    logger.info(f"STEP 2: EVALUATING MODELS ON {split.upper()} SET")
    logger.info("="*70)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        data_dir='data/processed',
        models_dir='models',
        results_dir='results'
    )
    
    try:
        # Load models
        logger.info("\nLoading trained models...")
        evaluator.load_models()
        
        # Load data
        logger.info(f"\nLoading {split} dataset...")
        df = evaluator.load_data(split)
        
        # Evaluate all models
        logger.info("\nEvaluating all models...")
        results = evaluator.evaluate_all_models(df, split)
        
        # Generate visualizations
        logger.info("\nGenerating visualizations...")
        evaluator.generate_all_plots(split)
        
        # Generate report
        logger.info("\nGenerating evaluation report...")
        evaluator.generate_report(split)
        
        # Save best models if requested
        if save_best and split == 'validation':
            logger.info("\nSaving best models for production...")
            evaluator.save_best_models(split)
        
        logger.info("\n" + "="*70)
        logger.info("✅ EVALUATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to:")
        logger.info(f"  - results/model_evaluation_report_{split}.md")
        logger.info(f"  - results/confusion_matrices/")
        logger.info(f"  - results/roc_curves/")
        
        if save_best and split == 'validation':
            logger.info(f"\nBest models saved to:")
            logger.info(f"  - models/ruminating_classifier.pkl")
            logger.info(f"  - models/feeding_classifier.pkl")
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"\n❌ Error: {e}")
        if "model" in str(e).lower():
            logger.error("\nPlease train models first.")
        return False
    except Exception as e:
        logger.error(f"\n❌ Unexpected error during evaluation: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train and evaluate behavior classification models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default training and evaluation
  python train_and_evaluate_models.py
  
  # Quick training (for testing)
  python train_and_evaluate_models.py --quick
  
  # Thorough training (for production)
  python train_and_evaluate_models.py --thorough
  
  # Only evaluate existing models
  python train_and_evaluate_models.py --eval-only
  
  # Final test set evaluation
  python train_and_evaluate_models.py --test
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick training mode (10 iterations, 3-fold CV)'
    )
    parser.add_argument(
        '--thorough',
        action='store_true',
        help='Thorough training mode (100 iterations, 10-fold CV)'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate existing models (skip training)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Evaluate on test set (use only once after validation)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("BEHAVIOR CLASSIFICATION MODELS - TRAINING & EVALUATION")
    print("="*70)
    print()
    
    success = True
    
    # Training phase
    if not args.eval_only:
        success = train_models(quick=args.quick, thorough=args.thorough)
        if not success:
            logger.error("\nTraining failed. Exiting.")
            sys.exit(1)
    
    # Evaluation phase
    if success:
        split = 'test' if args.test else 'validation'
        
        if args.test:
            logger.warning("\n⚠️  WARNING: You are about to evaluate on the TEST SET.")
            logger.warning("The test set should only be used ONCE for final evaluation.")
            response = input("\nContinue? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Test evaluation cancelled.")
                sys.exit(0)
        
        success = evaluate_models(split=split, save_best=(not args.test))
        if not success:
            logger.error("\nEvaluation failed. Exiting.")
            sys.exit(1)
    
    # Summary
    print("\n" + "="*70)
    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext Steps:")
    
    if args.test:
        print("1. Review test results in results/model_evaluation_report_test.md")
        print("2. Deploy models to production (Task #91)")
    else:
        print("1. Review results in results/model_evaluation_report_validation.md")
        print("2. Check visualizations in results/confusion_matrices/ and results/roc_curves/")
        print("3. If results are satisfactory, run final test evaluation:")
        print("   python train_and_evaluate_models.py --eval-only --test")
        print("4. Deploy to production (Task #91)")
    
    print()


if __name__ == '__main__':
    main()
