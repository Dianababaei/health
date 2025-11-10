"""
Example Usage Scripts for Behavior Classification Models

This module provides practical examples of training, evaluating, and using
behavior classification models.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def example_1_train_models():
    """Example 1: Train all models with default settings."""
    print("="*70)
    print("EXAMPLE 1: Train All Models with Default Settings")
    print("="*70)
    
    from src.models.train_behavior_classifiers import BehaviorClassifierTrainer
    
    # Initialize trainer with default settings
    trainer = BehaviorClassifierTrainer(
        data_dir='data/processed',
        models_dir='models',
        search_type='randomized',
        n_iter=50,
        cv_folds=5
    )
    
    try:
        # Load data
        print("\n1. Loading datasets...")
        train_df, val_df, test_df = trainer.load_data()
        
        # Train all models
        print("\n2. Training all models (this may take 10-30 minutes)...")
        results = trainer.train_all_models(train_df)
        
        # Save models
        print("\n3. Saving models to disk...")
        trainer.save_models()
        trainer.save_training_summary()
        
        print("\n" + "="*70)
        print("SUCCESS! Models trained and saved.")
        print("="*70)
        print("\nNext step: Run example_2_evaluate_models()")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure Task #89 (Prepare Training Dataset) is completed first.")
        return False
    
    return True


def example_2_evaluate_models():
    """Example 2: Evaluate trained models on validation set."""
    print("="*70)
    print("EXAMPLE 2: Evaluate Models on Validation Set")
    print("="*70)
    
    from src.models.evaluate_models import ModelEvaluator
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        data_dir='data/processed',
        models_dir='models',
        results_dir='results'
    )
    
    try:
        # Load models
        print("\n1. Loading trained models...")
        evaluator.load_models()
        
        # Load validation data
        print("\n2. Loading validation dataset...")
        val_df = evaluator.load_data('validation')
        
        # Evaluate all models
        print("\n3. Evaluating all models...")
        results = evaluator.evaluate_all_models(val_df, 'validation')
        
        # Generate visualizations
        print("\n4. Generating visualizations...")
        evaluator.generate_all_plots('validation')
        
        # Generate report
        print("\n5. Generating evaluation report...")
        evaluator.generate_report('validation')
        
        # Save best models
        print("\n6. Saving best models for production...")
        evaluator.save_best_models('validation')
        
        print("\n" + "="*70)
        print("SUCCESS! Evaluation complete.")
        print("="*70)
        print("\nCheck the following:")
        print("  - results/model_evaluation_report_validation.md")
        print("  - results/confusion_matrices/")
        print("  - results/roc_curves/")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run example_1_train_models() first.")
        return False
    
    return True


def example_3_quick_training():
    """Example 3: Quick training with fewer iterations (for testing)."""
    print("="*70)
    print("EXAMPLE 3: Quick Training (Fewer Iterations)")
    print("="*70)
    
    from src.models.train_behavior_classifiers import BehaviorClassifierTrainer
    
    # Initialize with fewer iterations for faster training
    trainer = BehaviorClassifierTrainer(
        data_dir='data/processed',
        models_dir='models',
        search_type='randomized',
        n_iter=10,  # Much faster, but less thorough
        cv_folds=3   # Fewer folds for speed
    )
    
    try:
        train_df, val_df, test_df = trainer.load_data()
        results = trainer.train_all_models(train_df)
        trainer.save_models()
        trainer.save_training_summary()
        
        print("\n" + "="*70)
        print("SUCCESS! Quick training complete.")
        print("="*70)
        print("\nNote: For production, use more iterations (50+) and folds (5+)")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return False
    
    return True


def example_4_load_and_predict():
    """Example 4: Load trained model and make predictions."""
    print("="*70)
    print("EXAMPLE 4: Load Model and Make Predictions")
    print("="*70)
    
    import joblib
    import pandas as pd
    import numpy as np
    from pathlib import Path
    
    try:
        # Load best ruminating model
        print("\n1. Loading trained ruminating model...")
        model_path = Path('models/ruminating_classifier.pkl')
        
        if not model_path.exists():
            print(f"Model not found at {model_path}")
            print("Please run example_2_evaluate_models() with --save-best first.")
            return False
        
        model = joblib.load(model_path)
        
        # Check if scaler exists (for SVM models)
        scaler_path = Path('models/ruminating_scaler.pkl')
        scaler = None
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print("   (Loaded feature scaler)")
        
        # Load validation data as example
        print("\n2. Loading sample data...")
        data_path = Path('data/processed/validation_features.pkl')
        df = pd.read_pickle(data_path)
        
        # Prepare features
        print("\n3. Preparing features...")
        exclude_cols = ['is_ruminating', 'is_feeding', 'timestamp', 'cow_id',
                       'sensor_id', 'behavioral_state', 'sample_id', 'animal_id']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        
        # Scale if needed
        if scaler is not None:
            print("   Scaling features...")
            X = scaler.transform(X)
        
        # Make predictions
        print("\n4. Making predictions...")
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Show results
        print("\n5. Results (first 10 samples):")
        print("-" * 70)
        for i in range(min(10, len(predictions))):
            true_label = df['is_ruminating'].iloc[i]
            pred_label = predictions[i]
            prob = probabilities[i]
            
            status = "✓" if pred_label == true_label else "✗"
            print(f"  Sample {i}: True={true_label}, Pred={pred_label}, "
                  f"Prob={prob:.3f} {status}")
        
        # Calculate accuracy
        accuracy = (predictions == df['is_ruminating'].values).mean()
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        print("\n" + "="*70)
        print("SUCCESS! Predictions complete.")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return False
    
    return True


def example_5_compare_models():
    """Example 5: Compare RF and SVM performance."""
    print("="*70)
    print("EXAMPLE 5: Compare Random Forest vs SVM")
    print("="*70)
    
    from src.models.evaluate_models import ModelEvaluator
    
    evaluator = ModelEvaluator()
    
    try:
        evaluator.load_models()
        val_df = evaluator.load_data('validation')
        evaluator.evaluate_all_models(val_df, 'validation')
        
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        for behavior in ['ruminating', 'feeding']:
            print(f"\n{behavior.upper()} DETECTION:")
            print("-" * 70)
            
            comparison = evaluator.compare_models(behavior, 'validation')
            
            print(f"  Random Forest:")
            print(f"    F1-Score:  {comparison['rf_f1']:.4f}")
            print(f"    Precision: {comparison['rf_precision']:.4f}")
            print(f"    Recall:    {comparison['rf_recall']:.4f}")
            print(f"    ROC-AUC:   {comparison['rf_roc_auc']:.4f}")
            
            print(f"\n  SVM:")
            print(f"    F1-Score:  {comparison['svm_f1']:.4f}")
            print(f"    Precision: {comparison['svm_precision']:.4f}")
            print(f"    Recall:    {comparison['svm_recall']:.4f}")
            print(f"    ROC-AUC:   {comparison['svm_roc_auc']:.4f}")
            
            print(f"\n  ⭐ WINNER: {comparison['best_model'].upper()} "
                  f"(F1={comparison['best_f1']:.4f})")
        
        print("\n" + "="*70)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return False
    
    return True


def example_6_test_set_evaluation():
    """Example 6: Final evaluation on test set."""
    print("="*70)
    print("EXAMPLE 6: Final Test Set Evaluation")
    print("="*70)
    print("\nWARNING: Only run this ONCE after validation is complete!")
    print("Test set should remain unseen until final evaluation.\n")
    
    response = input("Continue with test set evaluation? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return False
    
    from src.models.evaluate_models import ModelEvaluator
    
    evaluator = ModelEvaluator()
    
    try:
        evaluator.load_models()
        test_df = evaluator.load_data('test')
        
        print("\nEvaluating on test set...")
        evaluator.evaluate_all_models(test_df, 'test')
        evaluator.generate_all_plots('test')
        evaluator.generate_report('test')
        
        print("\n" + "="*70)
        print("SUCCESS! Test evaluation complete.")
        print("="*70)
        print("\nCheck: results/model_evaluation_report_test.md")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return False
    
    return True


def main():
    """Main menu for example selection."""
    print("\n" + "="*70)
    print("BEHAVIOR CLASSIFICATION MODELS - EXAMPLE USAGE")
    print("="*70)
    print("\nAvailable Examples:")
    print("  1. Train all models with default settings")
    print("  2. Evaluate models on validation set")
    print("  3. Quick training (fewer iterations, for testing)")
    print("  4. Load model and make predictions")
    print("  5. Compare Random Forest vs SVM performance")
    print("  6. Final test set evaluation (use only once!)")
    print("  7. Run full workflow (train + evaluate)")
    print("  0. Exit")
    
    while True:
        print("\n" + "-"*70)
        choice = input("Select example (0-7): ").strip()
        
        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            example_1_train_models()
        elif choice == '2':
            example_2_evaluate_models()
        elif choice == '3':
            example_3_quick_training()
        elif choice == '4':
            example_4_load_and_predict()
        elif choice == '5':
            example_5_compare_models()
        elif choice == '6':
            example_6_test_set_evaluation()
        elif choice == '7':
            print("\nRunning full workflow...")
            if example_1_train_models():
                example_2_evaluate_models()
        else:
            print("Invalid choice. Please select 0-7.")


if __name__ == '__main__':
    main()
