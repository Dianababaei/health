# Behavior Classification Models

This module implements machine learning models for binary classification of cattle behaviors (ruminating and feeding) using sensor data from neck-mounted devices.

## Overview

The module provides:
- **Training Pipeline**: Automated training of Random Forest and SVM classifiers
- **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV with cross-validation
- **Model Evaluation**: Comprehensive metrics, confusion matrices, ROC/PR curves
- **Model Selection**: Automatic selection of best model per behavior based on F1-score
- **Production Deployment**: Serialized models ready for inference

## Table of Contents

1. [Quick Start](#quick-start)
2. [Model Architecture](#model-architecture)
3. [Training Pipeline](#training-pipeline)
4. [Evaluation Pipeline](#evaluation-pipeline)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

Ensure you have completed Task #89 (Prepare Training Dataset) which generates:
- `data/processed/training_features.pkl`
- `data/processed/validation_features.pkl`
- `data/processed/test_features.pkl`

### Basic Workflow

```bash
# 1. Train all models (takes 10-30 minutes depending on data size)
cd src/models
python train_behavior_classifiers.py

# 2. Evaluate on validation set
python evaluate_models.py --split validation --save-best

# 3. Review results
# - Check results/model_evaluation_report_validation.md
# - View confusion matrices in results/confusion_matrices/
# - View ROC/PR curves in results/roc_curves/

# 4. (Optional) Final test set evaluation
python evaluate_models.py --split test
```

---

## Model Architecture

### Algorithms

#### Random Forest Classifier
- **Type**: Ensemble learning (decision trees)
- **Advantages**:
  - Handles non-linear relationships
  - Feature importance insights
  - Robust to outliers
  - No feature scaling required
- **Use Cases**: General-purpose classification, interpretable models

#### Support Vector Machine (SVM)
- **Type**: Kernel-based classifier (RBF kernel)
- **Advantages**:
  - Effective in high-dimensional spaces
  - Memory efficient
  - Works well with clear decision boundaries
- **Use Cases**: Maximum-margin classification, non-linear patterns

### Target Behaviors

#### Ruminating (Binary Classification)
- **Positive Class**: Animal is ruminating
- **Negative Class**: Animal is not ruminating
- **Key Features**:
  - Rhythmic head movements (Lyg, Dzg)
  - Lateral jaw motion (Mya)
  - Frequency patterns (0.67-1.0 Hz = 40-60 cycles/min)
  - Low overall motion intensity

#### Feeding (Binary Classification)
- **Positive Class**: Animal is feeding
- **Negative Class**: Animal is not feeding
- **Key Features**:
  - Head-down posture (negative pitch angle)
  - Forward-backward head motion (Fxa)
  - Moderate angular velocities (Lyg)
  - Standing posture (Rza > 0.7g)

---

## Training Pipeline

### Script: `train_behavior_classifiers.py`

The training pipeline handles:
1. Data loading and preprocessing
2. Feature scaling (for SVM)
3. Hyperparameter search with cross-validation
4. Model training and serialization
5. Training summary generation

### Hyperparameter Search Spaces

#### Random Forest
```python
{
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', None]
}
```

#### SVM
```python
{
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly'],
    'class_weight': ['balanced', None]
}
```

### Cross-Validation

- **Strategy**: 5-fold stratified cross-validation
- **Scoring Metric**: F1-score (balanced precision-recall)
- **Optimization**: Maximize F1-score on validation folds

### Command-Line Options

```bash
python train_behavior_classifiers.py [OPTIONS]

Options:
  --data-dir PATH         Data directory (default: data/processed)
  --models-dir PATH       Models output directory (default: models)
  --search-type TYPE      'grid' or 'randomized' (default: randomized)
  --n-iter N             Iterations for randomized search (default: 50)
  --cv-folds N           Cross-validation folds (default: 5)
```

### Output Files

```
models/
└── trained/
    ├── rf_ruminating_model.pkl          # Trained RF model
    ├── rf_feeding_model.pkl
    ├── svm_ruminating_model.pkl         # Trained SVM model
    ├── svm_ruminating_scaler.pkl        # Feature scaler for SVM
    ├── svm_feeding_model.pkl
    ├── svm_feeding_scaler.pkl
    ├── training_results.pkl             # Detailed results (CV scores, etc.)
    └── training_summary.txt             # Human-readable summary
```

---

## Evaluation Pipeline

### Script: `evaluate_models.py`

The evaluation pipeline provides:
1. Model loading and inference
2. Comprehensive metric calculation
3. Visualization generation (confusion matrices, ROC, PR curves)
4. Model comparison and selection
5. Report generation

### Metrics Calculated

#### Core Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value (TP / (TP + FP))
- **Recall**: Sensitivity/True positive rate (TP / (TP + FN))
- **F1-Score**: Harmonic mean of precision and recall

#### Advanced Metrics
- **ROC-AUC**: Area under ROC curve (discrimination ability)
- **Average Precision**: Area under PR curve
- **Confusion Matrix**: True/false positives/negatives

### Visualizations

#### Confusion Matrix
Shows classification results:
```
              Predicted
              Neg    Pos
Actual  Neg   TN     FP
        Pos   FN     TP
```

#### ROC Curve
- X-axis: False Positive Rate
- Y-axis: True Positive Rate
- AUC close to 1.0 indicates good discrimination

#### Precision-Recall Curve
- X-axis: Recall
- Y-axis: Precision
- Useful for imbalanced datasets

### Command-Line Options

```bash
python evaluate_models.py [OPTIONS]

Options:
  --data-dir PATH         Data directory (default: data/processed)
  --models-dir PATH       Models directory (default: models)
  --results-dir PATH      Results output directory (default: results)
  --split SPLIT          'validation' or 'test' (default: validation)
  --save-best            Save best models to production filenames
```

### Output Files

```
results/
├── model_evaluation_report_validation.md    # Comprehensive report
├── confusion_matrices/
│   ├── rf_ruminating_validation_confusion_matrix.png
│   ├── rf_feeding_validation_confusion_matrix.png
│   ├── svm_ruminating_validation_confusion_matrix.png
│   └── svm_feeding_validation_confusion_matrix.png
└── roc_curves/
    ├── rf_ruminating_validation_roc_curve.png
    ├── rf_ruminating_validation_pr_curve.png
    ├── rf_feeding_validation_roc_curve.png
    ├── rf_feeding_validation_pr_curve.png
    ├── svm_ruminating_validation_roc_curve.png
    ├── svm_ruminating_validation_pr_curve.png
    ├── svm_feeding_validation_roc_curve.png
    └── svm_feeding_validation_pr_curve.png
```

---

## Usage Examples

### Example 1: Train with Default Settings

```python
from train_behavior_classifiers import BehaviorClassifierTrainer

# Initialize trainer
trainer = BehaviorClassifierTrainer()

# Load data
train_df, val_df, test_df = trainer.load_data()

# Train all models
results = trainer.train_all_models(train_df)

# Save models
trainer.save_models()
trainer.save_training_summary()
```

### Example 2: Train with Custom Settings

```python
trainer = BehaviorClassifierTrainer(
    data_dir='data/processed',
    models_dir='models',
    search_type='grid',      # Use GridSearchCV
    n_iter=100,              # More iterations
    cv_folds=10,             # 10-fold CV
    random_state=42
)

train_df, val_df, test_df = trainer.load_data()
results = trainer.train_all_models(train_df)
trainer.save_models()
```

### Example 3: Evaluate Models

```python
from evaluate_models import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Load models and data
evaluator.load_models()
val_df = evaluator.load_data('validation')

# Evaluate all models
results = evaluator.evaluate_all_models(val_df, 'validation')

# Generate visualizations
evaluator.generate_all_plots('validation')

# Generate report
evaluator.generate_report('validation')

# Save best models
evaluator.save_best_models('validation')
```

### Example 4: Load and Use Trained Model

```python
import joblib
import pandas as pd

# Load best model
model = joblib.load('models/ruminating_classifier.pkl')
scaler = joblib.load('models/ruminating_scaler.pkl')  # If SVM

# Load new data
new_data = pd.read_pickle('data/processed/new_features.pkl')

# Prepare features
exclude_cols = ['is_ruminating', 'is_feeding', 'timestamp']
feature_cols = [col for col in new_data.columns if col not in exclude_cols]
X = new_data[feature_cols].values

# Scale if needed (for SVM)
if scaler is not None:
    X = scaler.transform(X)

# Predict
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# Add to dataframe
new_data['ruminating_prediction'] = predictions
new_data['ruminating_probability'] = probabilities
```

### Example 5: Custom Model Comparison

```python
from evaluate_models import ModelEvaluator

evaluator = ModelEvaluator()
evaluator.load_models()
val_df = evaluator.load_data('validation')
evaluator.evaluate_all_models(val_df, 'validation')

# Compare models for ruminating
comparison = evaluator.compare_models('ruminating', 'validation')

print(f"Best model: {comparison['best_model'].upper()}")
print(f"F1-Score: {comparison['best_f1']:.4f}")
print(f"RF F1: {comparison['rf_f1']:.4f}")
print(f"SVM F1: {comparison['svm_f1']:.4f}")
```

---

## API Reference

### BehaviorClassifierTrainer

```python
class BehaviorClassifierTrainer:
    """Trainer for behavior classification models."""
    
    def __init__(self, 
                 data_dir: str = 'data/processed',
                 models_dir: str = 'models',
                 search_type: str = 'randomized',
                 n_iter: int = 50,
                 cv_folds: int = 5,
                 random_state: int = 42)
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    
    def prepare_features(self, df: pd.DataFrame, target_col: str) 
        -> Tuple[np.ndarray, np.ndarray]
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   model_type: str, behavior: str, scale_features: bool = True) 
        -> Dict[str, Any]
    
    def train_all_models(self, train_df: pd.DataFrame) -> Dict[str, Dict]
    
    def save_models(self)
    
    def save_training_summary(self)
```

### ModelEvaluator

```python
class ModelEvaluator:
    """Evaluator for trained models."""
    
    def __init__(self, data_dir: str = 'data/processed',
                 models_dir: str = 'models',
                 results_dir: str = 'results')
    
    def load_models(self)
    
    def load_data(self, split: str = 'validation') -> pd.DataFrame
    
    def prepare_features(self, df: pd.DataFrame, target_col: str) 
        -> Tuple[np.ndarray, np.ndarray]
    
    def evaluate_model(self, model_key: str, X: np.ndarray, y: np.ndarray,
                      split: str = 'validation') -> Dict[str, Any]
    
    def evaluate_all_models(self, df: pd.DataFrame, split: str = 'validation') 
        -> Dict[str, Dict]
    
    def plot_confusion_matrix(self, model_key: str, cm: np.ndarray,
                             split: str = 'validation')
    
    def plot_roc_curve(self, model_key: str, roc_data: Dict, roc_auc: float,
                      split: str = 'validation')
    
    def plot_precision_recall_curve(self, model_key: str, pr_data: Dict,
                                   avg_precision: float, split: str = 'validation')
    
    def generate_all_plots(self, split: str = 'validation')
    
    def compare_models(self, behavior: str, split: str = 'validation') -> Dict
    
    def generate_report(self, split: str = 'validation')
    
    def save_best_models(self, split: str = 'validation')
```

---

## Performance Benchmarks

### Expected Performance (Literature-Based)

Based on cattle behavior classification research:

| Behavior | Algorithm | Expected F1-Score | Expected Accuracy |
|----------|-----------|-------------------|-------------------|
| Ruminating | RF | 0.85-0.92 | 0.88-0.95 |
| Ruminating | SVM | 0.83-0.90 | 0.86-0.93 |
| Feeding | RF | 0.88-0.94 | 0.90-0.96 |
| Feeding | SVM | 0.85-0.92 | 0.88-0.94 |

### Success Criteria

- ✓ Validation F1-score > 0.75 for both behaviors
- ✓ Precision and recall within 10% of each other
- ✓ Test performance within 5% of validation (no overfitting)
- ✓ Confusion matrix diagonal dominance
- ✓ Inference time < 100ms per sample

### Training Time

| Model | Dataset Size | Search Type | Approximate Time |
|-------|-------------|-------------|------------------|
| RF | 10,000 samples | Randomized (50 iter) | 5-10 minutes |
| RF | 50,000 samples | Randomized (50 iter) | 15-30 minutes |
| SVM | 10,000 samples | Randomized (50 iter) | 10-20 minutes |
| SVM | 50,000 samples | Randomized (50 iter) | 30-60 minutes |

### Inference Time

- **Random Forest**: 0.1-1 ms per sample
- **SVM**: 0.5-2 ms per sample
- **Both**: Well below 100ms real-time requirement

---

## Troubleshooting

### Issue: FileNotFoundError for training data

**Error:**
```
FileNotFoundError: Training data not found at data/processed/training_features.pkl
```

**Solution:**
Run Task #89 (Prepare Training Dataset) first:
```bash
# Generate training data from simulated or real sensor data
python scripts/prepare_training_data.py
```

### Issue: Low F1-scores (< 0.75)

**Possible Causes:**
1. Insufficient or poor-quality training data
2. Class imbalance not addressed
3. Features not properly engineered
4. Hyperparameter search space too narrow

**Solutions:**
1. Check data quality and class distribution
2. Use `class_weight='balanced'` (already included in search space)
3. Review feature engineering (Task #89)
4. Expand hyperparameter search space
5. Increase `n_iter` for randomized search

### Issue: Imbalanced precision and recall

**Symptoms:**
- High precision, low recall: Model too conservative
- Low precision, high recall: Model too aggressive

**Solutions:**
1. Adjust classification threshold using PR curve
2. Re-weight classes based on false positive/negative costs
3. Use different scoring metric during training

### Issue: Overfitting (validation >> test performance)

**Symptoms:**
- High validation F1-score
- Much lower test F1-score (> 5% difference)

**Solutions:**
1. Increase regularization (lower `max_depth`, higher `min_samples_split` for RF)
2. Use smaller `C` values for SVM
3. Add more training data
4. Reduce feature set (remove correlated features)

### Issue: Training takes too long

**Solutions:**
1. Use `search_type='randomized'` instead of `'grid'`
2. Reduce `n_iter` (try 20-30 instead of 50)
3. Reduce cross-validation folds (try 3 instead of 5)
4. Use smaller hyperparameter search space
5. Sample training data if > 100,000 samples

### Issue: Model serialization fails

**Error:**
```
PicklingError: Can't pickle <object>: attribute lookup failed
```

**Solutions:**
1. Ensure all custom classes are importable
2. Use `joblib.dump()` instead of `pickle.dump()` (already implemented)
3. Check Python version compatibility

---

## Integration with Pipeline

### Upstream Dependencies

**Task #89: Prepare Training Dataset**
- Generates `training_features.pkl`, `validation_features.pkl`, `test_features.pkl`
- Includes feature engineering (motion intensity, pitch angle, etc.)
- Provides binary labels for ruminating and feeding

### Downstream Usage

**Task #91: Implement Hybrid Classification Pipeline**
- Loads best models: `models/ruminating_classifier.pkl`, `models/feeding_classifier.pkl`
- Combines ML predictions with rule-based classifiers
- Deploys to production monitoring system

---

## References

### Literature
- Nielsen et al. (2010): Cattle behavior classification using accelerometers
- Borchers et al. (2016): Commercial accelerometer validation
- Umemura et al. (2009): Decision-tree posture classification
- Smith et al. (2016): Multi-class behavior recognition

### Internal Documentation
- `docs/behavioral_thresholds_literature_review.md` - Expected behavioral patterns
- `docs/normalization_feature_engineering.md` - Feature engineering details
- `docs/dataset_generation_implementation.md` - Data preparation

---

## License

Part of Artemis Health Livestock Monitoring System.

---

## Contact

For questions or issues, please refer to the main project documentation or contact the development team.
