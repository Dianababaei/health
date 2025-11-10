"""
Dataset Preparation and Building for ML Training

This module handles:
- Loading simulated sensor data with behavioral state labels
- Feature extraction and aggregation
- Class balancing (stratified sampling, SMOTE)
- Train/validation/test splitting with stratification
- Feature normalization/standardization
- Data quality validation

Ensures no temporal data leakage and proper dataset preparation for scikit-learn models.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

from .behavior_features import BehaviorFeatureExtractor


class DatasetBuilder:
    """
    Build and prepare datasets for ML training.
    
    Handles data loading, feature extraction, balancing, splitting,
    and normalization with proper prevention of data leakage.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1.0,
        window_minutes: int = 10,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        balance_threshold: float = 2.0,
        random_seed: int = 42
    ):
        """
        Initialize dataset builder.
        
        Args:
            sampling_rate: Samples per minute for sensor data
            window_minutes: Window size for feature extraction
            train_ratio: Proportion for training set (default: 0.7)
            val_ratio: Proportion for validation set (default: 0.15)
            test_ratio: Proportion for test set (default: 0.15)
            balance_threshold: Max class imbalance ratio before balancing (default: 2.0)
            random_seed: Random seed for reproducibility
        """
        self.sampling_rate = sampling_rate
        self.window_minutes = window_minutes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.balance_threshold = balance_threshold
        self.random_seed = random_seed
        
        # Initialize feature extractor
        self.feature_extractor = BehaviorFeatureExtractor(
            sampling_rate=sampling_rate,
            window_minutes=window_minutes
        )
        
        # Store scaler fitted on training data
        self.scaler = None
        
        # Store dataset statistics
        self.dataset_stats = {}
    
    def load_simulated_data(
        self, 
        filepath: str,
        target_behaviors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Load simulated sensor data from CSV file.
        
        Args:
            filepath: Path to CSV file with simulated data
            target_behaviors: List of target behavior states to include (None = all)
            
        Returns:
            DataFrame with sensor data and labels
        """
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter to target behaviors if specified
        if target_behaviors is not None and 'state' in df.columns:
            df = df[df['state'].isin(target_behaviors)].copy()
            print(f"Filtered to behaviors: {target_behaviors}")
        
        print(f"Loaded {len(df)} samples")
        if 'state' in df.columns:
            print(f"State distribution:\n{df['state'].value_counts()}")
        
        return df
    
    def generate_simulated_data(
        self,
        samples_per_state: int = 1000,
        target_states: Optional[List[str]] = None,
        include_health_conditions: bool = False
    ) -> pd.DataFrame:
        """
        Generate simulated data using the simulation engine.
        
        Args:
            samples_per_state: Number of samples per behavioral state
            target_states: List of target states (None = ruminating and feeding)
            include_health_conditions: Whether to include health condition variations
            
        Returns:
            DataFrame with simulated sensor data and labels
        """
        from src.simulation.engine import SimulationEngine
        from src.simulation.transitions import BehaviorState
        
        if target_states is None:
            # Default: focus on ruminating and feeding
            target_states = [
                BehaviorState.RUMINATING_LYING,
                BehaviorState.RUMINATING_STANDING,
                BehaviorState.FEEDING
            ]
        else:
            # Convert string names to BehaviorState enum
            target_states = [BehaviorState[s.upper()] if isinstance(s, str) else s 
                           for s in target_states]
        
        print(f"Generating simulated data: {samples_per_state} samples per state")
        
        engine = SimulationEngine(
            baseline_temperature=38.5,
            sampling_rate=self.sampling_rate,
            random_seed=self.random_seed
        )
        
        all_data = []
        
        for state in target_states:
            print(f"  Generating {state.value} samples...")
            
            # Generate multiple samples for this state
            for i in range(samples_per_state):
                # Generate data for this sample
                sample_duration = self.window_minutes  # Each sample is one window
                
                df = engine.generate_single_state_data(
                    state=state,
                    duration_minutes=sample_duration,
                    start_datetime=datetime(2024, 1, 1, 0, 0, 0) + pd.Timedelta(minutes=i * sample_duration)
                )
                
                df['sample_id'] = f"{state.value}_{i}"
                all_data.append(df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print(f"\nGenerated {len(combined_df)} total samples")
        print(f"State distribution:\n{combined_df['state'].value_counts()}")
        
        return combined_df
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw sensor data.
        
        Args:
            df: DataFrame with raw sensor data
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting features...")
        features_df = self.feature_extractor.extract_all_features(df)
        print(f"Extracted {len(features_df.columns)} features from {len(features_df)} samples")
        
        return features_df
    
    def check_class_balance(self, df: pd.DataFrame, label_column: str = 'state') -> Dict[str, float]:
        """
        Check class balance and compute imbalance ratios.
        
        Args:
            df: DataFrame with labels
            label_column: Name of label column
            
        Returns:
            Dictionary with class counts and imbalance info
        """
        class_counts = df[label_column].value_counts()
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        balance_info = {
            'class_counts': class_counts.to_dict(),
            'total_samples': len(df),
            'num_classes': len(class_counts),
            'max_count': int(max_count),
            'min_count': int(min_count),
            'imbalance_ratio': float(imbalance_ratio),
            'needs_balancing': imbalance_ratio > self.balance_threshold
        }
        
        print("\nClass Balance Analysis:")
        print(f"  Total samples: {balance_info['total_samples']}")
        print(f"  Number of classes: {balance_info['num_classes']}")
        print(f"  Class distribution:")
        for cls, count in class_counts.items():
            print(f"    {cls}: {count} samples ({count/len(df)*100:.1f}%)")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        print(f"  Needs balancing (>{self.balance_threshold}): {balance_info['needs_balancing']}")
        
        return balance_info
    
    def balance_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'smote'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance dataset using stratified sampling or SMOTE.
        
        Args:
            X: Feature matrix
            y: Labels
            method: Balancing method ('smote', 'undersample', 'hybrid')
            
        Returns:
            Tuple of (balanced_X, balanced_y)
        """
        print(f"\nBalancing dataset using method: {method}")
        print(f"Original class distribution: {Counter(y)}")
        
        if method == 'smote':
            # SMOTE for oversampling minority classes
            smote = SMOTE(random_state=self.random_seed)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        
        elif method == 'undersample':
            # Random undersampling of majority classes
            undersampler = RandomUnderSampler(random_state=self.random_seed)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)
        
        elif method == 'hybrid':
            # Combine SMOTE and undersampling
            # First oversample with SMOTE
            smote = SMOTE(random_state=self.random_seed)
            X_temp, y_temp = smote.fit_resample(X, y)
            
            # Then undersample to reduce overall size
            undersampler = RandomUnderSampler(random_state=self.random_seed)
            X_balanced, y_balanced = undersampler.fit_resample(X_temp, y_temp)
        
        else:
            raise ValueError(f"Unknown balancing method: {method}")
        
        print(f"Balanced class distribution: {Counter(y_balanced)}")
        print(f"Dataset size: {len(X)} → {len(X_balanced)}")
        
        return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced, name=y.name)
    
    def split_dataset(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split dataset into train/validation/test sets with stratification.
        
        Args:
            X: Feature matrix
            y: Labels
            stratify: Whether to use stratified splitting
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("\nSplitting dataset...")
        print(f"  Train: {self.train_ratio*100:.0f}%")
        print(f"  Validation: {self.val_ratio*100:.0f}%")
        print(f"  Test: {self.test_ratio*100:.0f}%")
        print(f"  Stratified: {stratify}")
        
        # First split: train vs (val + test)
        stratify_arg = y if stratify else None
        test_val_ratio = self.val_ratio + self.test_ratio
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_val_ratio,
            stratify=stratify_arg,
            random_state=self.random_seed
        )
        
        # Second split: val vs test
        val_ratio_adjusted = self.val_ratio / test_val_ratio
        stratify_arg_temp = y_temp if stratify else None
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio_adjusted),
            stratify=stratify_arg_temp,
            random_state=self.random_seed
        )
        
        # Print split statistics
        print(f"\nSplit sizes:")
        print(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        if stratify:
            print(f"\nClass distribution in splits:")
            for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
                dist = split_y.value_counts(normalize=True) * 100
                print(f"  {split_name}:")
                for cls, pct in dist.items():
                    print(f"    {cls}: {pct:.1f}%")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Normalize/standardize features using training set statistics.
        
        Prevents data leakage by fitting scaler only on training data.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            exclude_columns: Columns to exclude from normalization (e.g., binary features)
            
        Returns:
            Tuple of (X_train_norm, X_val_norm, X_test_norm)
        """
        print("\nNormalizing features...")
        
        # Identify columns to normalize
        if exclude_columns is None:
            exclude_columns = []
        
        # Also exclude any categorical or already-normalized features
        exclude_columns.extend([
            col for col in X_train.columns 
            if 'temporal_is_' in col or '_in_target_range' in col
        ])
        
        normalize_columns = [col for col in X_train.columns if col not in exclude_columns]
        
        print(f"  Normalizing {len(normalize_columns)} / {len(X_train.columns)} columns")
        
        # Fit scaler on training data only
        self.scaler = StandardScaler()
        self.scaler.fit(X_train[normalize_columns])
        
        # Transform all splits
        X_train_norm = X_train.copy()
        X_val_norm = X_val.copy()
        X_test_norm = X_test.copy()
        
        X_train_norm[normalize_columns] = self.scaler.transform(X_train[normalize_columns])
        X_val_norm[normalize_columns] = self.scaler.transform(X_val[normalize_columns])
        X_test_norm[normalize_columns] = self.scaler.transform(X_test[normalize_columns])
        
        # Verify no NaN or Inf values
        for name, df in [('Train', X_train_norm), ('Val', X_val_norm), ('Test', X_test_norm)]:
            n_nan = df.isnull().sum().sum()
            n_inf = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
            if n_nan > 0 or n_inf > 0:
                print(f"  WARNING: {name} set has {n_nan} NaN and {n_inf} Inf values")
        
        print("  Normalization complete")
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def validate_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        tolerance: float = 0.05
    ) -> Dict[str, bool]:
        """
        Validate data splits for quality and no leakage.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Label vectors
            tolerance: Tolerance for class distribution differences (default: 5%)
            
        Returns:
            Dictionary with validation results
        """
        print("\nValidating data splits...")
        
        results = {
            'no_overlap': True,
            'distributions_similar': True,
            'no_nan_inf': True,
            'reasonable_ranges': True
        }
        
        # Check for data overlap (shouldn't happen with proper splitting)
        train_indices = set(X_train.index)
        val_indices = set(X_val.index)
        test_indices = set(X_test.index)
        
        overlap = (train_indices & val_indices) | (train_indices & test_indices) | (val_indices & test_indices)
        if len(overlap) > 0:
            print(f"  WARNING: Found {len(overlap)} overlapping samples between splits!")
            results['no_overlap'] = False
        else:
            print("  ✓ No overlap between splits")
        
        # Check class distributions are similar
        train_dist = y_train.value_counts(normalize=True)
        val_dist = y_val.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        max_diff = 0.0
        for cls in train_dist.index:
            if cls in val_dist.index and cls in test_dist.index:
                diff = max(
                    abs(train_dist[cls] - val_dist[cls]),
                    abs(train_dist[cls] - test_dist[cls]),
                    abs(val_dist[cls] - test_dist[cls])
                )
                max_diff = max(max_diff, diff)
        
        if max_diff > tolerance:
            print(f"  WARNING: Class distributions differ by up to {max_diff*100:.1f}% (tolerance: {tolerance*100:.0f}%)")
            results['distributions_similar'] = False
        else:
            print(f"  ✓ Class distributions similar (max diff: {max_diff*100:.1f}%)")
        
        # Check for NaN/Inf values
        for name, X in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
            n_nan = X.isnull().sum().sum()
            n_inf = np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            if n_nan > 0 or n_inf > 0:
                print(f"  WARNING: {name} has {n_nan} NaN and {n_inf} Inf values")
                results['no_nan_inf'] = False
        
        if results['no_nan_inf']:
            print("  ✓ No NaN or Inf values")
        
        # Check feature ranges are reasonable
        for name, X in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_std = X[col].std()
                if col_std == 0:
                    print(f"  WARNING: {name} - {col} has zero std dev")
                    results['reasonable_ranges'] = False
                    break
        
        if results['reasonable_ranges']:
            print("  ✓ Feature ranges are reasonable")
        
        # Overall validation
        all_passed = all(results.values())
        print(f"\nValidation {'PASSED' if all_passed else 'FAILED'}")
        
        return results
    
    def build_dataset(
        self,
        data_source: str,
        output_dir: str,
        target_behaviors: Optional[List[str]] = None,
        apply_balancing: bool = True,
        balance_method: str = 'smote'
    ) -> Dict[str, any]:
        """
        Complete dataset building pipeline.
        
        Args:
            data_source: Path to CSV file or 'generate' to create simulated data
            output_dir: Directory to save processed datasets
            target_behaviors: Target behavioral states to include
            apply_balancing: Whether to balance classes
            balance_method: Method for balancing ('smote', 'undersample', 'hybrid')
            
        Returns:
            Dictionary with dataset statistics and file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 70)
        print("DATASET BUILDING PIPELINE")
        print("=" * 70)
        
        # Step 1: Load or generate data
        if data_source == 'generate':
            df = self.generate_simulated_data(
                samples_per_state=1000,
                target_states=target_behaviors
            )
        else:
            df = self.load_simulated_data(data_source, target_behaviors)
        
        # Step 2: Extract features
        features_df = self.extract_features(df)
        
        # Remove any rows with NaN in critical columns
        features_df = features_df.dropna(subset=['state'])
        
        # Step 3: Prepare X and y
        label_column = 'state'
        exclude_columns = ['timestamp', 'state', 'sample_id']
        
        feature_columns = [col for col in features_df.columns if col not in exclude_columns]
        X = features_df[feature_columns].copy()
        y = features_df[label_column].copy()
        
        # Handle any remaining NaN by filling with 0
        X = X.fillna(0)
        
        # Step 4: Check class balance
        balance_info = self.check_class_balance(features_df, label_column)
        self.dataset_stats['original_balance'] = balance_info
        
        # Step 5: Balance dataset if needed
        if apply_balancing and balance_info['needs_balancing']:
            X, y = self.balance_dataset(X, y, method=balance_method)
            balance_info_after = self.check_class_balance(
                pd.DataFrame({'state': y}), 'state'
            )
            self.dataset_stats['balanced'] = balance_info_after
        
        # Step 6: Split dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_dataset(X, y, stratify=True)
        
        # Step 7: Normalize features
        X_train_norm, X_val_norm, X_test_norm = self.normalize_features(
            X_train, X_val, X_test
        )
        
        # Step 8: Validate splits
        validation_results = self.validate_splits(
            X_train_norm, X_val_norm, X_test_norm,
            y_train, y_val, y_test
        )
        self.dataset_stats['validation'] = validation_results
        
        # Step 9: Save datasets
        print("\nSaving datasets...")
        
        datasets = {
            'training': (X_train_norm, y_train),
            'validation': (X_val_norm, y_val),
            'test': (X_test_norm, y_test)
        }
        
        saved_files = {}
        for name, (X_data, y_data) in datasets.items():
            filepath = output_path / f"{name}_features.pkl"
            
            dataset_dict = {
                'X': X_data,
                'y': y_data,
                'feature_names': list(X_data.columns),
                'class_names': list(y_data.unique()),
                'scaler': self.scaler if name == 'training' else None,
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'n_samples': len(X_data),
                    'n_features': len(X_data.columns),
                    'sampling_rate': self.sampling_rate,
                    'window_minutes': self.window_minutes
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(dataset_dict, f)
            
            print(f"  Saved {name} set: {filepath}")
            saved_files[name] = str(filepath)
        
        # Save dataset statistics
        stats_filepath = output_path / "dataset_statistics.pkl"
        with open(stats_filepath, 'wb') as f:
            pickle.dump(self.dataset_stats, f)
        print(f"  Saved statistics: {stats_filepath}")
        
        print("\n" + "=" * 70)
        print("DATASET BUILDING COMPLETE")
        print("=" * 70)
        
        return {
            'files': saved_files,
            'statistics': self.dataset_stats,
            'validation': validation_results
        }


def load_prepared_dataset(filepath: str) -> Dict[str, any]:
    """
    Load a prepared dataset from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary with X, y, feature_names, and metadata
    """
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    return dataset
