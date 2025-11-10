# ML Model Training Guide

## Overview

This guide provides complete instructions for training and evaluating machine learning models for cattle behavior classification (ruminating and feeding detection). The guide covers the entire workflow from data preparation to model deployment.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Detailed Workflow](#detailed-workflow)
4. [Model Architecture](#model-architecture)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Model Selection](#model-selection)
8. [Production Deployment](#production-deployment)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Completed Dependencies

Ensure Task #89 (Prepare Training Dataset) is complete. This creates:
- `data/processed/training_features.pkl`
- `data/processed/validation_features.pkl`
- `data/processed/test_features.pkl`

### 2. Required Data Format

Each dataset file should contain:
- **Feature columns**: Sensor-derived features (motion_intensity, pitch_angle, head_movement_intensity, etc.)
- **Target columns**: Binary labels (`is_ruminating`, `is_feeding`)
- **Optional columns**: `timestamp`, `cow_id`, `sensor_id`, `behavioral_state`

### 3. Python Environment

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

Key packages:
- `scikit-learn>=1.1.0` - ML algorithms
- `pandas>=1.5.0` - Data handling
- `numpy>=1.23.0` - Numerical operations
- `matplotlib>=3.6.0` - Plotting
- `seaborn>=0.12.0` - Statistical plots

---

## Quick Start

### Option 1: Using the Quick-Start Script (Recommended)

```bash
# Train and evaluate with default settings
python train_and_evaluate_models.py

# Quick training for testing (faster, less thorough)
python train_and_evaluate_models.py --quick

# Thorough training for production (slower, more thorough)
python train_and_evaluate_models.py --thorough

# Only evaluate existing models
python train_and_evaluate_models.py --eval-only

# Final test set evaluation (use once!)
python train_and_evaluate_models.py --test
```

### Option 2: Manual Step-by-Step

```bash
# Step 1: Train models
cd src/models
python train_behavior_classifiers.py

# Step 2: Evaluate on validation set
python evaluate_models.py --split validation --save-best

# Step 3: Review results
cat ../../results/model_evaluation_report_validation.md

# Step 4: (Optional) Final test evaluation
python evaluate_models.py --split test
```

### Option 3: Interactive Examples

```bash
cd src/models
python example_usage.py
# Follow the interactive menu
```

---

## Detailed Workflow

### Step 1: Training Models

#### 1.1 Initialize Trainer

```python
from src.models.train_behavior_classifiers import BehaviorClassifierTrainer

trainer = BehaviorClassifierTrainer(
    data_dir='data/processed',
    models_dir='models',
    search_type='randomized',  # 'grid' or 'randomized'
    n_iter=50,                 # Number of iterations for randomized search
    cv_folds=5,                # Cross-validation folds
    random_state=42            # For reproducibility
)
```

#### 1.2 Load Data

```python
train_df, val_df, test_df = trainer.load_data()
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")
```

#### 1.3 Train All Models

```python
# Trains 4 models: RF-ruminating, RF-feeding, SVM-ruminating, SVM-feeding
results = trainer.train_all_models(train_df)
```

This will:
- Train Random Forest models (no feature scaling)
- Train SVM models (with StandardScaler)
- Perform hyperparameter search with cross-validation
- Log training time and best parameters
- Store models in memory

#### 1.4 Save Models

```python
trainer.save_models()
trainer.save_training_summary()
```

Output files:
- `models/trained/rf_ruminating_model.pkl`
- `models/trained/rf_feeding_model.pkl`
- `models/trained/svm_ruminating_model.pkl`
- `models/trained/svm_ruminating_scaler.pkl`
- `models/trained/svm_feeding_model.pkl`
- `models/trained/svm_feeding_scaler.pkl`
- `models/trained/training_results.pkl`
- `models/trained/training_summary.txt`

### Step 2: Evaluating Models

#### 2.1 Initialize Evaluator

```python
from src.models.evaluate_models import ModelEvaluator

evaluator = ModelEvaluator(
    data_dir='data/processed',
    models_dir='models',
    results_dir='results'
)
```

#### 2.2 Load Models

```python
evaluator.load_models()
```

#### 2.3 Load Validation Data

```python
val_df = evaluator.load_data('validation')
```

#### 2.4 Evaluate All Models

```python
results = evaluator.evaluate_all_models(val_df, 'validation')
```

This calculates:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC and Average Precision
- Confusion matrices
- ROC curves
- Precision-Recall curves

#### 2.5 Generate Visualizations

```python
evaluator.generate_all_plots('validation')
```

Creates:
- 4 confusion matrix plots
- 4 ROC curve plots
- 4 Precision-Recall curve plots

#### 2.6 Generate Report

```python
evaluator.generate_report('validation')
```

Creates: `results/model_evaluation_report_validation.md`

#### 2.7 Save Best Models

```python
evaluator.save_best_models('validation')
```

Saves best models to:
- `models/ruminating_classifier.pkl`
- `models/feeding_classifier.pkl`

### Step 3: Model Deployment

#### 3.1 Load Models for Inference

```python
from src.models.inference import BehaviorPredictor

predictor = BehaviorPredictor()
predictor.load_models()
```

#### 3.2 Make Predictions

```python
# Load new data
new_data = pd.read_pickle('data/processed/new_features.pkl')

# Predict
predictions = predictor.predict(new_data)

# Access predictions
print(predictions[['ruminating_prediction', 'ruminating_probability']])
print(predictions[['feeding_prediction', 'feeding_probability']])
```

#### 3.3 Single Sample Prediction

```python
features = {
    'motion_intensity': 0.85,
    'pitch_angle': -0.3,
    'head_movement_intensity': 12.5,
    # ... other features
}

results = predictor.predict_single(features)
print(results)
# {
#   'ruminating': {'prediction': True, 'probability': 0.87},
#   'feeding': {'prediction': False, 'probability': 0.23}
# }
```

---

## Model Architecture

### Algorithms Used

#### 1. Random Forest Classifier

**Type**: Ensemble learning (decision trees)

**Advantages**:
- Handles non-linear relationships naturally
- Provides feature importance rankings
- Robust to outliers and missing values
- No feature scaling required
- Fast training and inference

**Configuration**:
- n_estimators: 50-300 trees
- max_depth: 10-30 or unlimited
- Feature sampling: sqrt or log2
- Class weighting: Balanced or None

**Typical Use**: General-purpose classification, interpretable results

#### 2. Support Vector Machine (SVM)

**Type**: Kernel-based maximum-margin classifier

**Advantages**:
- Effective in high-dimensional spaces
- Memory efficient (uses support vectors only)
- Works well with clear decision boundaries
- Flexible kernel functions (RBF, polynomial)

**Configuration**:
- Kernel: RBF (radial basis function) or polynomial
- C: Regularization (0.1-100)
- Gamma: Kernel coefficient (scale, auto, 0.001-0.1)
- Class weighting: Balanced or None

**Preprocessing**: Requires StandardScaler for feature normalization

**Typical Use**: Maximum-margin classification, non-linear patterns

### Target Behaviors

#### Ruminating Detection

**Characteristics**:
- Rhythmic jaw movements (40-60 cycles/min)
- Lateral head motion (Mya oscillations)
- Low overall motion intensity
- Standing or lying posture
- Frequency content in 0.67-1.0 Hz range

**Key Features**:
- `head_movement_intensity` (Lyg, Dzg)
- `rhythmic_pattern_features` (FFT peaks)
- `zero_crossing_rate` (Mya)
- `activity_score` (low)

#### Feeding Detection

**Characteristics**:
- Head-down posture (negative pitch)
- Forward-backward head motion
- Moderate angular velocities
- Standing posture (Rza > 0.7g)
- Intermittent movement patterns

**Key Features**:
- `pitch_angle` (negative)
- `head_movement_intensity` (moderate)
- `motion_intensity` (moderate)
- `rza` (standing indicator)

---

## Hyperparameter Tuning

### Search Strategy

#### RandomizedSearchCV (Default)

**Advantages**:
- Faster than GridSearch
- Explores parameter space efficiently
- Good results with fewer iterations

**Configuration**:
- n_iter: 50 (default), 10 (quick), 100 (thorough)
- cv: 5-fold stratified cross-validation
- scoring: F1-score

**When to Use**: Default choice for most cases

#### GridSearchCV

**Advantages**:
- Exhaustive search of parameter space
- Guarantees finding best combination in grid
- Deterministic results

**Configuration**:
- Tests all parameter combinations
- cv: 5-fold stratified cross-validation
- scoring: F1-score

**When to Use**: Small parameter space or need exhaustive search

### Parameter Spaces

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

**Total combinations**: 4 × 4 × 3 × 3 × 2 × 2 = 576
- GridSearchCV: ~5-10 hours
- RandomizedSearchCV (50 iter): ~30-60 minutes

#### SVM

```python
{
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly'],
    'class_weight': ['balanced', None]
}
```

**Total combinations**: 4 × 5 × 2 × 2 = 80
- GridSearchCV: ~2-3 hours
- RandomizedSearchCV (50 iter): ~20-40 minutes

### Cross-Validation

**Strategy**: Stratified K-Fold

**Settings**:
- K=5 (default): Balanced speed vs accuracy
- K=3 (quick): Faster, less reliable
- K=10 (thorough): Slower, more reliable

**Stratification**: Maintains class balance in each fold

**Scoring**: F1-score (harmonic mean of precision and recall)

---

## Evaluation Metrics

### Primary Metrics

#### Accuracy
- **Definition**: (TP + TN) / Total
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Overall correctness
- **Limitation**: Misleading for imbalanced datasets

#### Precision
- **Definition**: TP / (TP + FP)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Positive predictive value
- **Use**: Minimize false positives

#### Recall (Sensitivity)
- **Definition**: TP / (TP + FN)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: True positive rate
- **Use**: Minimize false negatives

#### F1-Score
- **Definition**: 2 × (Precision × Recall) / (Precision + Recall)
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Balanced precision-recall metric
- **Use**: Primary metric for model selection

### Advanced Metrics

#### ROC-AUC
- **Definition**: Area under ROC curve
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Discrimination ability across all thresholds
- **Use**: Compare models, threshold selection

#### Average Precision
- **Definition**: Area under Precision-Recall curve
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Summary of PR curve
- **Use**: Better for imbalanced datasets than ROC-AUC

### Success Criteria

✅ **Required Performance**:
- Validation F1-score > 0.75 for both behaviors
- Precision and recall within 10% of each other
- Test performance within 5% of validation (no overfitting)
- Confusion matrix diagonal dominance

📊 **Expected Performance** (Literature-Based):
- Ruminating: F1 = 0.85-0.92
- Feeding: F1 = 0.88-0.94

---

## Model Selection

### Selection Process

#### Step 1: Train Both Algorithms

For each behavior:
- Train Random Forest
- Train SVM

Total: 4 models

#### Step 2: Evaluate on Validation Set

Calculate metrics for all 4 models:
- Accuracy, Precision, Recall, F1-score
- ROC-AUC, Average Precision
- Confusion matrix

#### Step 3: Compare Performance

For each behavior:
```python
comparison = {
    'rf_f1': 0.89,
    'svm_f1': 0.87,
    'rf_precision': 0.88,
    'svm_precision': 0.86,
    'rf_recall': 0.90,
    'svm_recall': 0.88
}
```

#### Step 4: Select Best Model

Selection criteria (in order):
1. **F1-score**: Higher is better
2. **Precision-Recall balance**: Difference < 10%
3. **ROC-AUC**: Higher is better
4. **Inference speed**: Faster is better (both are fast enough)

#### Step 5: Save Best Models

```python
evaluator.save_best_models('validation')
```

Saves to production filenames:
- `models/ruminating_classifier.pkl`
- `models/feeding_classifier.pkl`

### Example Selection Rationale

**Ruminating Detection**:
- RF F1: 0.89, Precision: 0.88, Recall: 0.90
- SVM F1: 0.87, Precision: 0.89, Recall: 0.85
- **Winner**: Random Forest (higher F1, better recall)

**Feeding Detection**:
- RF F1: 0.91, Precision: 0.90, Recall: 0.92
- SVM F1: 0.92, Precision: 0.93, Recall: 0.91
- **Winner**: SVM (higher F1, better precision)

---

## Production Deployment

### Loading Models

```python
from src.models.inference import BehaviorPredictor

predictor = BehaviorPredictor('models')
predictor.load_models()
```

### Batch Prediction

```python
# Load data
new_data = pd.read_csv('data/new_sensor_readings.csv')

# Engineer features (use Task #89 pipeline)
from src.data_processing import feature_engineering
features = feature_engineering.extract_all_features(new_data)

# Predict
predictions = predictor.predict(features)

# Export results
predictions.to_csv('data/predictions.csv', index=False)
```

### Real-Time Prediction

```python
# For streaming data
for sample in data_stream:
    # Extract features
    features = extract_features(sample)
    
    # Predict
    result = predictor.predict_single(features)
    
    # Act on prediction
    if result['ruminating']['prediction']:
        log_ruminating_event(sample['cow_id'], sample['timestamp'])
    
    if result['feeding']['prediction']:
        log_feeding_event(sample['cow_id'], sample['timestamp'])
```

### Performance Monitoring

```python
# Track prediction distribution
from collections import Counter

predictions = predictor.predict(daily_data)
ruminating_count = predictions['ruminating_prediction'].sum()
feeding_count = predictions['feeding_prediction'].sum()

# Alert if distribution is unusual
if ruminating_count < expected_min or ruminating_count > expected_max:
    alert_model_drift('ruminating', ruminating_count)
```

---

## Troubleshooting

### Common Issues

#### 1. Training Data Not Found

**Error**: `FileNotFoundError: Training data not found at data/processed/training_features.pkl`

**Cause**: Task #89 (Prepare Training Dataset) not completed

**Solution**:
```bash
# Generate training data first
python scripts/prepare_training_data.py
```

#### 2. Low F1-Scores (< 0.75)

**Symptoms**: Models perform poorly on validation set

**Possible Causes**:
- Poor feature engineering
- Insufficient training data
- Class imbalance not addressed
- Hyperparameters not optimized

**Solutions**:
1. Review feature engineering (Task #89)
2. Check data quality and quantity
3. Increase hyperparameter search iterations:
   ```bash
   python train_behavior_classifiers.py --n-iter 100 --cv-folds 10
   ```
4. Balance classes using `class_weight='balanced'` (already in search space)

#### 3. Overfitting

**Symptoms**: High validation score, much lower test score

**Causes**:
- Model too complex
- Insufficient regularization
- Data leakage

**Solutions**:
1. Increase regularization:
   - RF: Lower `max_depth`, higher `min_samples_split`
   - SVM: Lower `C` value
2. Add more training data
3. Verify no data leakage between splits

#### 4. Imbalanced Precision-Recall

**Symptoms**: Precision = 0.95, Recall = 0.65 (or vice versa)

**Causes**:
- Class imbalance
- Threshold not optimal
- Model bias

**Solutions**:
1. Adjust classification threshold using ROC/PR curves
2. Use `class_weight='balanced'`
3. Collect more samples from minority class

#### 5. Slow Training

**Symptoms**: Training takes > 2 hours

**Solutions**:
1. Use RandomizedSearchCV instead of GridSearchCV
2. Reduce `n_iter` (try 20-30)
3. Reduce `cv_folds` (try 3)
4. Sample training data if > 100,000 samples

---

## Best Practices

### 1. Data Preparation

✅ **Do**:
- Verify data quality before training
- Check class balance
- Engineer meaningful features
- Split data properly (temporal or random)

❌ **Don't**:
- Mix training and test data
- Ignore missing values
- Use raw sensor values without features

### 2. Training

✅ **Do**:
- Start with RandomizedSearchCV
- Use 5-fold cross-validation
- Optimize for F1-score
- Log all hyperparameters

❌ **Don't**:
- Use test set during training
- Ignore class imbalance
- Skip cross-validation
- Forget to save models

### 3. Evaluation

✅ **Do**:
- Evaluate on validation set first
- Generate visualizations
- Check confusion matrices
- Verify success criteria

❌ **Don't**:
- Touch test set until final evaluation
- Rely solely on accuracy
- Ignore precision-recall balance
- Skip visual inspection

### 4. Deployment

✅ **Do**:
- Use best models from validation
- Monitor prediction distributions
- Track inference time
- Log predictions for audit

❌ **Don't**:
- Deploy without validation
- Ignore model drift
- Skip performance monitoring
- Forget model versioning

---

## References

### Internal Documentation

- `src/models/README.md` - Module documentation
- `src/models/IMPLEMENTATION_SUMMARY.md` - Implementation details
- `docs/behavioral_thresholds_literature_review.md` - Expected patterns
- `docs/normalization_feature_engineering.md` - Feature engineering

### Literature

- Nielsen et al. (2010): Quantification of activity of dairy cows using accelerometers
- Borchers et al. (2016): A validation of technologies monitoring dairy cow feeding, ruminating, and lying behaviors
- Smith et al. (2016): Behavior classification of cows fitted with motion collars
- Umemura et al. (2009): Technical note: Estimation of DM intake by dairy cows using accelerometers

---

## Support

For questions or issues:
1. Check troubleshooting section above
2. Review example scripts in `src/models/example_usage.py`
3. Consult module documentation in `src/models/README.md`
4. Check logs in `logs/` directory

---

**Last Updated**: 2025-01-08
**Version**: 1.0.0
