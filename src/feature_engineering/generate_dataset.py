"""
Generate and prepare training/validation/test datasets for ML models.

This script:
1. Generates simulated sensor data with labeled behavioral states
2. Extracts features for ruminating and feeding detection
3. Balances classes if needed
4. Splits into train/validation/test sets
5. Normalizes features
6. Exports to pickle files for ML training

Run this script to create the datasets needed for Task #90 (ML model training).
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Change to project root directory for proper imports
os.chdir(project_root)

from src.feature_engineering.dataset_builder import DatasetBuilder


def main():
    """Generate complete dataset for ML training."""
    
    print("\n" + "=" * 70)
    print(" " * 15 + "DATASET GENERATION FOR ML TRAINING")
    print("=" * 70 + "\n")
    
    # Configuration
    config = {
        'sampling_rate': 1.0,  # 1 sample per minute
        'window_minutes': 10,   # 10-minute feature windows
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        'balance_threshold': 2.0,
        'random_seed': 42
    }
    
    # Target behaviors: ruminating (lying and standing) and feeding
    target_behaviors = [
        'ruminating_lying',
        'ruminating_standing', 
        'feeding'
    ]
    
    print("Configuration:")
    print(f"  Sampling rate: {config['sampling_rate']} samples/minute")
    print(f"  Feature window: {config['window_minutes']} minutes")
    print(f"  Split ratios: {config['train_ratio']:.0%} train / {config['val_ratio']:.0%} val / {config['test_ratio']:.0%} test")
    print(f"  Target behaviors: {target_behaviors}")
    print(f"  Balance threshold: {config['balance_threshold']}:1")
    print(f"  Random seed: {config['random_seed']}")
    print()
    
    # Initialize dataset builder
    builder = DatasetBuilder(
        sampling_rate=config['sampling_rate'],
        window_minutes=config['window_minutes'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        balance_threshold=config['balance_threshold'],
        random_seed=config['random_seed']
    )
    
    # Build dataset
    # Use 'generate' to create simulated data
    # This will generate 1000+ samples per class as required
    results = builder.build_dataset(
        data_source='generate',
        output_dir='data/processed',
        target_behaviors=target_behaviors,
        apply_balancing=True,
        balance_method='smote'
    )
    
    # Print summary
    print("\n" + "=" * 70)
    print("DATASET GENERATION SUMMARY")
    print("=" * 70)
    
    print("\nGenerated files:")
    for name, path in results['files'].items():
        print(f"  {name}: {path}")
    
    print("\nValidation results:")
    for check, passed in results['validation'].items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check}: {status}")
    
    print("\nDataset statistics:")
    stats = results['statistics']
    
    if 'original_balance' in stats:
        print("\n  Original class distribution:")
        for cls, count in stats['original_balance']['class_counts'].items():
            print(f"    {cls}: {count} samples")
        print(f"  Imbalance ratio: {stats['original_balance']['imbalance_ratio']:.2f}")
    
    if 'balanced' in stats:
        print("\n  Balanced class distribution:")
        for cls, count in stats['balanced']['class_counts'].items():
            print(f"    {cls}: {count} samples")
    
    print("\n" + "=" * 70)
    print("SUCCESS: Datasets ready for ML training!")
    print("=" * 70)
    
    print("\nNext steps:")
    print("  1. Train ML models using the generated datasets")
    print("  2. Evaluate model performance on validation set")
    print("  3. Final testing on held-out test set")
    print("\nDatasets can be loaded using:")
    print("  from src.feature_engineering.dataset_builder import load_prepared_dataset")
    print("  dataset = load_prepared_dataset('data/processed/training_features.pkl')")
    print()


if __name__ == '__main__':
    main()
