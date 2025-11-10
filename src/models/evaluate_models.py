"""
Model Evaluation Module

This module provides comprehensive evaluation tools for trained behavior
classification models, including:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices with visualization
- ROC curves and AUC scores
- Precision-Recall curves
- Model comparison and selection
- Performance reports
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


class ModelEvaluator:
    """
    Evaluator class for behavior classification models.
    
    Evaluates trained models on validation/test sets and generates
    comprehensive performance reports and visualizations.
    """
    
    def __init__(self,
                 data_dir: str = 'data/processed',
                 models_dir: str = 'models',
                 results_dir: str = 'results'):
        """
        Initialize the evaluator.
        
        Args:
            data_dir: Directory containing datasets
            models_dir: Directory with trained models
            results_dir: Directory to save evaluation results
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create output directories
        (self.results_dir / 'confusion_matrices').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'roc_curves').mkdir(parents=True, exist_ok=True)
        
        # Storage for evaluation results
        self.evaluation_results = {}
        self.models = {}
        self.scalers = {}
        
        logger.info("Initialized ModelEvaluator")
    
    def load_models(self):
        """Load all trained models and scalers."""
        logger.info("Loading trained models...")
        
        trained_dir = self.models_dir / 'trained'
        
        # Load all model files
        for model_file in trained_dir.glob('*_model.pkl'):
            model_key = model_file.stem.replace('_model', '')
            self.models[model_key] = joblib.load(model_file)
            logger.info(f"Loaded {model_key}")
            
            # Load corresponding scaler if exists
            scaler_file = trained_dir / f'{model_key}_scaler.pkl'
            if scaler_file.exists():
                self.scalers[model_key] = joblib.load(scaler_file)
                logger.info(f"Loaded scaler for {model_key}")
        
        if not self.models:
            raise FileNotFoundError(
                f"No trained models found in {trained_dir}. "
                f"Please run train_behavior_classifiers.py first."
            )
        
        logger.info(f"Loaded {len(self.models)} models")
    
    def load_data(self, split: str = 'validation') -> pd.DataFrame:
        """
        Load dataset for evaluation.
        
        Args:
            split: 'validation' or 'test'
            
        Returns:
            DataFrame with features and labels
        """
        filename = f'{split}_features.pkl'
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_pickle(filepath)
        logger.info(f"Loaded {split} set: {len(df)} samples")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for evaluation.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            
        Returns:
            Tuple of (X, y) arrays
        """
        # Identify feature columns
        exclude_cols = ['is_ruminating', 'is_feeding', 'timestamp', 'cow_id',
                       'sensor_id', 'behavioral_state', 'sample_id', 'animal_id']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y
    
    def evaluate_model(self,
                      model_key: str,
                      X: np.ndarray,
                      y: np.ndarray,
                      split: str = 'validation') -> Dict[str, Any]:
        """
        Evaluate a single model.
        
        Args:
            model_key: Model identifier (e.g., 'rf_ruminating')
            X: Feature matrix
            y: True labels
            split: Dataset split name
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\nEvaluating {model_key} on {split} set...")
        
        model = self.models[model_key]
        
        # Scale features if scaler exists
        if model_key in self.scalers:
            X = self.scalers[model_key].transform(X)
        
        # Get predictions and probabilities
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred, 
                                                          target_names=['Negative', 'Positive']),
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, y_proba)
        metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        metrics['roc_auc'] = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y, y_proba)
        metrics['pr_curve'] = {'precision': precision, 'recall': recall, 
                              'thresholds': pr_thresholds}
        metrics['avg_precision'] = average_precision_score(y, y_proba)
        
        # Log results
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, df: pd.DataFrame, 
                           split: str = 'validation') -> Dict[str, Dict]:
        """
        Evaluate all models on a dataset.
        
        Args:
            df: DataFrame with features and labels
            split: Dataset split name
            
        Returns:
            Dictionary of evaluation results for all models
        """
        behaviors = ['ruminating', 'feeding']
        results = {}
        
        for behavior in behaviors:
            target_col = f'is_{behavior}'
            X, y = self.prepare_features(df, target_col)
            
            for model_key in self.models.keys():
                if behavior in model_key:
                    metrics = self.evaluate_model(model_key, X, y, split)
                    results[model_key] = metrics
        
        # Store results
        if split not in self.evaluation_results:
            self.evaluation_results[split] = {}
        self.evaluation_results[split] = results
        
        return results
    
    def plot_confusion_matrix(self, model_key: str, cm: np.ndarray,
                             split: str = 'validation'):
        """
        Plot and save confusion matrix.
        
        Args:
            model_key: Model identifier
            cm: Confusion matrix
            split: Dataset split name
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix: {model_key.upper()}\n({split.capitalize()} Set)',
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save figure
        filename = f'{model_key}_{split}_confusion_matrix.png'
        filepath = self.results_dir / 'confusion_matrices' / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved confusion matrix to {filepath}")
    
    def plot_roc_curve(self, model_key: str, roc_data: Dict,
                      roc_auc: float, split: str = 'validation'):
        """
        Plot and save ROC curve.
        
        Args:
            model_key: Model identifier
            roc_data: Dictionary with 'fpr' and 'tpr'
            roc_auc: AUC score
            split: Dataset split name
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(roc_data['fpr'], roc_data['tpr'], 
                color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve: {model_key.upper()}\n({split.capitalize()} Set)',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f'{model_key}_{split}_roc_curve.png'
        filepath = self.results_dir / 'roc_curves' / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curve to {filepath}")
    
    def plot_precision_recall_curve(self, model_key: str, pr_data: Dict,
                                   avg_precision: float, split: str = 'validation'):
        """
        Plot and save Precision-Recall curve.
        
        Args:
            model_key: Model identifier
            pr_data: Dictionary with 'precision' and 'recall'
            avg_precision: Average precision score
            split: Dataset split name
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(pr_data['recall'], pr_data['precision'],
                color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve: {model_key.upper()}\n({split.capitalize()} Set)',
                 fontsize=14, fontweight='bold')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        filename = f'{model_key}_{split}_pr_curve.png'
        filepath = self.results_dir / 'roc_curves' / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved PR curve to {filepath}")
    
    def generate_all_plots(self, split: str = 'validation'):
        """
        Generate all visualization plots.
        
        Args:
            split: Dataset split name
        """
        logger.info(f"\nGenerating visualizations for {split} set...")
        
        results = self.evaluation_results.get(split, {})
        
        for model_key, metrics in results.items():
            # Confusion matrix
            self.plot_confusion_matrix(model_key, metrics['confusion_matrix'], split)
            
            # ROC curve
            self.plot_roc_curve(model_key, metrics['roc_curve'], 
                              metrics['roc_auc'], split)
            
            # Precision-Recall curve
            self.plot_precision_recall_curve(model_key, metrics['pr_curve'],
                                           metrics['avg_precision'], split)
    
    def compare_models(self, behavior: str, split: str = 'validation') -> Dict:
        """
        Compare RF and SVM models for a specific behavior.
        
        Args:
            behavior: 'ruminating' or 'feeding'
            split: Dataset split name
            
        Returns:
            Comparison results with best model selection
        """
        rf_key = f'rf_{behavior}'
        svm_key = f'svm_{behavior}'
        
        results = self.evaluation_results.get(split, {})
        
        rf_metrics = results.get(rf_key, {})
        svm_metrics = results.get(svm_key, {})
        
        comparison = {
            'behavior': behavior,
            'rf_f1': rf_metrics.get('f1', 0),
            'svm_f1': svm_metrics.get('f1', 0),
            'rf_precision': rf_metrics.get('precision', 0),
            'svm_precision': svm_metrics.get('precision', 0),
            'rf_recall': rf_metrics.get('recall', 0),
            'svm_recall': svm_metrics.get('recall', 0),
            'rf_accuracy': rf_metrics.get('accuracy', 0),
            'svm_accuracy': svm_metrics.get('accuracy', 0),
            'rf_roc_auc': rf_metrics.get('roc_auc', 0),
            'svm_roc_auc': svm_metrics.get('roc_auc', 0)
        }
        
        # Select best model based on F1-score
        if comparison['rf_f1'] > comparison['svm_f1']:
            comparison['best_model'] = 'rf'
            comparison['best_f1'] = comparison['rf_f1']
        else:
            comparison['best_model'] = 'svm'
            comparison['best_f1'] = comparison['svm_f1']
        
        return comparison
    
    def generate_report(self, split: str = 'validation'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            split: Dataset split name
        """
        logger.info(f"\nGenerating evaluation report for {split} set...")
        
        report_path = self.results_dir / f'model_evaluation_report_{split}.md'
        
        with open(report_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Dataset Split:** {split.capitalize()}\n\n")
            f.write("---\n\n")
            
            # Performance summary
            f.write("## Performance Summary\n\n")
            
            results = self.evaluation_results.get(split, {})
            
            # Create comparison table
            f.write("### Model Comparison Table\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
            f.write("|-------|----------|-----------|--------|----------|----------|\n")
            
            for model_key in sorted(results.keys()):
                metrics = results[model_key]
                f.write(f"| **{model_key.upper()}** | "
                       f"{metrics['accuracy']:.4f} | "
                       f"{metrics['precision']:.4f} | "
                       f"{metrics['recall']:.4f} | "
                       f"{metrics['f1']:.4f} | "
                       f"{metrics['roc_auc']:.4f} |\n")
            
            f.write("\n---\n\n")
            
            # Best model selection for each behavior
            f.write("## Best Model Selection\n\n")
            
            for behavior in ['ruminating', 'feeding']:
                comparison = self.compare_models(behavior, split)
                
                f.write(f"### {behavior.capitalize()} Detection\n\n")
                f.write(f"**Winner:** {comparison['best_model'].upper()} "
                       f"(F1-Score: {comparison['best_f1']:.4f})\n\n")
                
                f.write("**Comparison:**\n")
                f.write(f"- Random Forest F1: {comparison['rf_f1']:.4f}\n")
                f.write(f"- SVM F1: {comparison['svm_f1']:.4f}\n")
                f.write(f"- Difference: {abs(comparison['rf_f1'] - comparison['svm_f1']):.4f}\n\n")
                
                # Check balance
                best_key = f"{comparison['best_model']}_{behavior}"
                best_metrics = results[best_key]
                prec_recall_diff = abs(best_metrics['precision'] - best_metrics['recall'])
                
                f.write(f"**Precision-Recall Balance:** ")
                if prec_recall_diff <= 0.1:
                    f.write(f"✓ Balanced ({prec_recall_diff:.4f} difference)\n")
                else:
                    f.write(f"⚠ Imbalanced ({prec_recall_diff:.4f} difference)\n")
                
                f.write("\n")
            
            f.write("---\n\n")
            
            # Detailed metrics for each model
            f.write("## Detailed Performance Metrics\n\n")
            
            for model_key in sorted(results.keys()):
                metrics = results[model_key]
                
                f.write(f"### {model_key.upper()}\n\n")
                f.write("**Core Metrics:**\n")
                f.write(f"- Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"- Precision: {metrics['precision']:.4f}\n")
                f.write(f"- Recall: {metrics['recall']:.4f}\n")
                f.write(f"- F1-Score: {metrics['f1']:.4f}\n")
                f.write(f"- ROC-AUC: {metrics['roc_auc']:.4f}\n")
                f.write(f"- Average Precision: {metrics['avg_precision']:.4f}\n\n")
                
                f.write("**Confusion Matrix:**\n")
                cm = metrics['confusion_matrix']
                f.write(f"- True Negatives: {cm[0, 0]}\n")
                f.write(f"- False Positives: {cm[0, 1]}\n")
                f.write(f"- False Negatives: {cm[1, 0]}\n")
                f.write(f"- True Positives: {cm[1, 1]}\n\n")
                
                f.write("**Classification Report:**\n")
                f.write("```\n")
                f.write(metrics['classification_report'])
                f.write("```\n\n")
                
                f.write("---\n\n")
            
            # Success criteria evaluation
            f.write("## Success Criteria Evaluation\n\n")
            
            for behavior in ['ruminating', 'feeding']:
                comparison = self.compare_models(behavior, split)
                best_key = f"{comparison['best_model']}_{behavior}"
                best_metrics = results[best_key]
                
                f.write(f"### {behavior.capitalize()}\n\n")
                
                # F1 > 0.75
                f1_check = "✓" if best_metrics['f1'] > 0.75 else "✗"
                f.write(f"- {f1_check} F1-Score > 0.75: {best_metrics['f1']:.4f}\n")
                
                # Balanced precision-recall
                prec_recall_diff = abs(best_metrics['precision'] - best_metrics['recall'])
                balance_check = "✓" if prec_recall_diff <= 0.1 else "✗"
                f.write(f"- {balance_check} Precision-Recall balanced (within 10%): "
                       f"{prec_recall_diff:.4f}\n")
                
                # Confusion matrix diagonal dominance
                cm = best_metrics['confusion_matrix']
                true_pos = cm[0, 0] + cm[1, 1]
                false_pos = cm[0, 1] + cm[1, 0]
                diagonal_check = "✓" if true_pos > false_pos else "✗"
                f.write(f"- {diagonal_check} Confusion matrix diagonal dominance: "
                       f"{true_pos} correct vs {false_pos} incorrect\n\n")
            
            f.write("---\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Selected Models for Production\n\n")
            
            for behavior in ['ruminating', 'feeding']:
                comparison = self.compare_models(behavior, split)
                f.write(f"- **{behavior.capitalize()}:** {comparison['best_model'].upper()} "
                       f"(F1={comparison['best_f1']:.4f})\n")
            
            f.write("\n### Next Steps\n\n")
            f.write("1. Review confusion matrices and ROC curves in `results/` directories\n")
            f.write("2. If validation performance is satisfactory, proceed to test set evaluation\n")
            f.write("3. Consider threshold tuning if precision-recall balance needs adjustment\n")
            f.write("4. Deploy selected models to production pipeline\n\n")
        
        logger.info(f"Saved evaluation report to {report_path}")
    
    def save_best_models(self, split: str = 'validation'):
        """
        Save best models for each behavior to designated filenames.
        
        Args:
            split: Dataset split used for selection
        """
        logger.info("\nSaving best models for production...")
        
        for behavior in ['ruminating', 'feeding']:
            comparison = self.compare_models(behavior, split)
            best_model_key = f"{comparison['best_model']}_{behavior}"
            
            # Copy best model to standard filename
            src_model = self.models_dir / 'trained' / f'{best_model_key}_model.pkl'
            dst_model = self.models_dir / f'{behavior}_classifier.pkl'
            
            model = joblib.load(src_model)
            joblib.dump(model, dst_model)
            logger.info(f"Saved best {behavior} model to {dst_model}")
            
            # Copy scaler if exists
            if best_model_key in self.scalers:
                src_scaler = self.models_dir / 'trained' / f'{best_model_key}_scaler.pkl'
                dst_scaler = self.models_dir / f'{behavior}_scaler.pkl'
                
                scaler = joblib.load(src_scaler)
                joblib.dump(scaler, dst_scaler)
                logger.info(f"Saved scaler to {dst_scaler}")


def main():
    """Main evaluation script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate trained behavior classification models'
    )
    parser.add_argument(
        '--data-dir',
        default='data/processed',
        help='Directory containing datasets'
    )
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory with trained models'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--split',
        choices=['validation', 'test'],
        default='validation',
        help='Dataset split to evaluate'
    )
    parser.add_argument(
        '--save-best',
        action='store_true',
        help='Save best models to production filenames'
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    try:
        # Load models
        evaluator.load_models()
        
        # Load data
        df = evaluator.load_data(args.split)
        
        # Evaluate all models
        results = evaluator.evaluate_all_models(df, args.split)
        
        # Generate visualizations
        evaluator.generate_all_plots(args.split)
        
        # Generate report
        evaluator.generate_report(args.split)
        
        # Save best models if requested
        if args.save_best:
            evaluator.save_best_models(args.split)
        
        logger.info("\n" + "="*70)
        logger.info("EVALUATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nResults saved to: {args.results_dir}")
        logger.info("  - Confusion matrices: results/confusion_matrices/")
        logger.info("  - ROC/PR curves: results/roc_curves/")
        logger.info(f"  - Report: results/model_evaluation_report_{args.split}.md")
        
    except FileNotFoundError as e:
        logger.error(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nUnexpected error during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
