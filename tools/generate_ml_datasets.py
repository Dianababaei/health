#!/usr/bin/env python3
"""
Generate ML Training Datasets

Standalone script to generate training/validation/test datasets for
cattle behavior classification (ruminating and feeding detection).

Run from project root:
    python generate_ml_datasets.py
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Ensure project root is in path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
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
        print("Starting dataset generation...")
        print("This will generate 1000+ samples per class using the simulation engine.\n")
        
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
        
        return results

    if __name__ == '__main__':
        results = main()

except ImportError as e:
    print("\n" + "=" * 70)
    print("ERROR: Missing Dependencies")
    print("=" * 70)
    print(f"\nImport error: {e}")
    print("\nPlease install required packages:")
    print("  pip install -r requirements.txt")
    print("\nRequired packages:")
    print("  - numpy")
    print("  - pandas")
    print("  - scipy")
    print("  - scikit-learn")
    print("  - imbalanced-learn")
    print()
    sys.exit(1)
except Exception as e:
    print("\n" + "=" * 70)
    print("ERROR: Dataset Generation Failed")
    print("=" * 70)
    print(f"\nError: {e}")
    print("\nPlease check:")
    print("  1. All dependencies are installed (pip install -r requirements.txt)")
    print("  2. You are running from the project root directory")
    print("  3. The src/ directory structure is intact")
    print()
    import traceback
    traceback.print_exc()
    sys.exit(1)
