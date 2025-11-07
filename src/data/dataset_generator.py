"""
Dataset generator script for creating synthetic animal behavior datasets.

Generates:
1. Balanced dataset with 1000+ minutes per behavior (6000+ total)
2. Transition-focused dataset with realistic behavior sequences
3. 3-5 multi-day datasets (7-14 days each) with circadian patterns
4. Train/val/test splits (70/15/15)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from synthetic_generator import SyntheticDataGenerator, BEHAVIOR_PATTERNS


def create_output_directory(base_path: str = "data/synthetic") -> Path:
    """Create output directory if it doesn't exist."""
    # Get project root (2 levels up from this script)
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / base_path
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_balanced_dataset(
    generator: SyntheticDataGenerator,
    minutes_per_behavior: int = 1100,
    start_date: datetime = datetime(2024, 1, 1),
) -> pd.DataFrame:
    """
    Generate balanced dataset with equal distribution across behaviors.
    
    Args:
        generator: SyntheticDataGenerator instance
        minutes_per_behavior: Minutes to generate per behavior (default 1100)
        start_date: Starting date for timestamps
    
    Returns:
        DataFrame with balanced behavior samples
    """
    print(f"\n🔄 Generating balanced dataset ({minutes_per_behavior} min/behavior)...")
    
    all_samples = []
    behaviors = list(BEHAVIOR_PATTERNS.keys())
    
    for behavior in behaviors:
        print(f"  - Generating {behavior}: {minutes_per_behavior} minutes")
        
        # Generate in chunks of 5-30 minutes for variety
        remaining = minutes_per_behavior
        current_time = start_date
        
        while remaining > 0:
            chunk_size = min(remaining, np.random.randint(5, 31))
            
            sample = generator.generate_behavior_sample(
                behavior=behavior,
                duration_minutes=chunk_size,
                start_time=current_time,
                apply_circadian=True,
            )
            all_samples.append(sample)
            
            remaining -= chunk_size
            current_time += timedelta(minutes=chunk_size + np.random.randint(1, 10))
    
    # Combine and shuffle
    df = pd.concat(all_samples, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Reset timestamps to be sequential
    df['timestamp'] = [start_date + timedelta(minutes=i) for i in range(len(df))]
    
    print(f"✓ Generated {len(df)} total samples")
    return df


def generate_transition_dataset(
    generator: SyntheticDataGenerator,
    transitions_per_pair: int = 15,
    start_date: datetime = datetime(2024, 2, 1),
) -> pd.DataFrame:
    """
    Generate dataset focused on behavior transitions.
    
    Args:
        generator: SyntheticDataGenerator instance
        transitions_per_pair: Number of transition examples per behavior pair
        start_date: Starting date for timestamps
    
    Returns:
        DataFrame with behavior sequences and transitions
    """
    print(f"\n🔄 Generating transition dataset...")
    
    # Define common transition pairs
    common_transitions = [
        ('lying', 'standing'),
        ('standing', 'lying'),
        ('standing', 'walking'),
        ('walking', 'standing'),
        ('standing', 'feeding'),
        ('feeding', 'ruminating'),
        ('ruminating', 'lying'),
        ('lying', 'ruminating'),
        ('walking', 'feeding'),
        ('feeding', 'standing'),
    ]
    
    all_sequences = []
    current_time = start_date
    
    for from_behavior, to_behavior in common_transitions:
        print(f"  - Generating {from_behavior} → {to_behavior}: {transitions_per_pair} examples")
        
        for _ in range(transitions_per_pair):
            # Create sequence: from_behavior → transition → to_behavior
            from_duration = np.random.randint(10, 25)
            to_duration = np.random.randint(10, 25)
            
            sequence = generator.generate_behavior_sequence(
                behaviors=[
                    (from_behavior, from_duration),
                    (to_behavior, to_duration),
                ],
                start_time=current_time,
                apply_circadian=True,
                smooth_transitions=True,
            )
            
            all_sequences.append(sequence)
            current_time += timedelta(minutes=from_duration + to_duration + 10)
    
    df = pd.concat(all_sequences, ignore_index=True)
    print(f"✓ Generated {len(df)} samples with {len(common_transitions)} transition types")
    return df


def generate_multiday_dataset(
    generator: SyntheticDataGenerator,
    num_days: int,
    dataset_id: int,
    start_date: datetime,
) -> pd.DataFrame:
    """
    Generate multi-day continuous dataset with circadian patterns.
    
    Args:
        generator: SyntheticDataGenerator instance
        num_days: Number of days to generate
        dataset_id: Dataset identifier
        start_date: Starting date
    
    Returns:
        DataFrame with multi-day continuous data
    """
    print(f"\n🔄 Generating multi-day dataset #{dataset_id} ({num_days} days)...")
    
    all_days = []
    current_date = start_date
    
    for day in range(num_days):
        print(f"  - Day {day + 1}/{num_days}: {current_date.date()}")
        
        # Generate daily schedule with variation
        daily_schedule = generator.generate_daily_schedule(current_date)
        
        # Generate behavior sequence for the day
        day_data = generator.generate_behavior_sequence(
            behaviors=daily_schedule,
            start_time=current_date,
            apply_circadian=True,
            smooth_transitions=True,
        )
        
        all_days.append(day_data)
        current_date += timedelta(days=1)
    
    df = pd.concat(all_days, ignore_index=True)
    print(f"✓ Generated {len(df)} samples ({len(df) / 60:.1f} hours)")
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratified: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Training set ratio (default 0.70)
        val_ratio: Validation set ratio (default 0.15)
        test_ratio: Test set ratio (default 0.15)
        stratified: Whether to maintain behavior distribution (default True)
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if stratified:
        # Stratified split to maintain behavior distribution
        train_samples = []
        val_samples = []
        test_samples = []
        
        for behavior in df['behavior_label'].unique():
            behavior_df = df[df['behavior_label'] == behavior]
            n = len(behavior_df)
            
            # Calculate split indices
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            # Shuffle behavior samples
            behavior_df = behavior_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            train_samples.append(behavior_df.iloc[:train_end])
            val_samples.append(behavior_df.iloc[train_end:val_end])
            test_samples.append(behavior_df.iloc[val_end:])
        
        train_df = pd.concat(train_samples, ignore_index=True)
        val_df = pd.concat(val_samples, ignore_index=True)
        test_df = pd.concat(test_samples, ignore_index=True)
        
        # Shuffle combined datasets
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=43).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=44).reset_index(drop=True)
    else:
        # Simple time-based split (for multi-day datasets)
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)
    
    return train_df, val_df, test_df


def save_dataset(df: pd.DataFrame, output_path: Path, name: str):
    """Save dataset to CSV file."""
    filepath = output_path / f"{name}.csv"
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved: {filepath} ({len(df)} samples)")


def generate_dataset_metadata(
    balanced_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    multiday_dfs: List[Tuple[int, int, pd.DataFrame]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path,
):
    """Generate metadata file documenting all datasets."""
    
    def get_behavior_stats(df: pd.DataFrame) -> Dict:
        """Get behavior distribution statistics."""
        stats = {}
        for behavior in df['behavior_label'].unique():
            count = len(df[df['behavior_label'] == behavior])
            stats[behavior] = {
                'count': int(count),
                'percentage': round(count / len(df) * 100, 2),
            }
        return stats
    
    def get_date_range(df: pd.DataFrame) -> Dict:
        """Get date range of dataset."""
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat(),
            'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
        }
    
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'datasets': {
            'balanced': {
                'description': 'Balanced dataset with equal behavior distribution',
                'total_samples': len(balanced_df),
                'total_minutes': len(balanced_df),
                'behaviors': get_behavior_stats(balanced_df),
                'date_range': get_date_range(balanced_df),
            },
            'transition': {
                'description': 'Dataset focused on behavior transitions',
                'total_samples': len(transition_df),
                'total_minutes': len(transition_df),
                'behaviors': get_behavior_stats(transition_df),
                'date_range': get_date_range(transition_df),
            },
            'multiday': {},
        },
        'splits': {
            'train': {
                'samples': len(train_df),
                'percentage': round(len(train_df) / (len(train_df) + len(val_df) + len(test_df)) * 100, 2),
                'behaviors': get_behavior_stats(train_df),
            },
            'validation': {
                'samples': len(val_df),
                'percentage': round(len(val_df) / (len(train_df) + len(val_df) + len(test_df)) * 100, 2),
                'behaviors': get_behavior_stats(val_df),
            },
            'test': {
                'samples': len(test_df),
                'percentage': round(len(test_df) / (len(train_df) + len(val_df) + len(test_df)) * 100, 2),
                'behaviors': get_behavior_stats(test_df),
            },
        },
    }
    
    # Add multiday dataset metadata
    for dataset_id, num_days, df in multiday_dfs:
        metadata['datasets']['multiday'][f'dataset_{dataset_id}'] = {
            'description': f'Multi-day continuous dataset spanning {num_days} days',
            'total_samples': len(df),
            'total_minutes': len(df),
            'duration_days': num_days,
            'behaviors': get_behavior_stats(df),
            'date_range': get_date_range(df),
        }
    
    # Save metadata
    metadata_path = output_path / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata: {metadata_path}")


def print_summary_statistics(
    balanced_df: pd.DataFrame,
    transition_df: pd.DataFrame,
    multiday_dfs: List[Tuple[int, int, pd.DataFrame]],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Print summary statistics for all generated datasets."""
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION SUMMARY")
    print("=" * 70)
    
    # Balanced dataset
    print("\n📊 BALANCED DATASET:")
    print(f"  Total samples: {len(balanced_df):,}")
    print(f"  Duration: {len(balanced_df) / 60:.1f} hours ({len(balanced_df) / 1440:.1f} days)")
    print("  Behavior distribution:")
    for behavior in sorted(balanced_df['behavior_label'].unique()):
        count = len(balanced_df[balanced_df['behavior_label'] == behavior])
        pct = count / len(balanced_df) * 100
        print(f"    - {behavior:12s}: {count:5,} samples ({pct:5.1f}%)")
    
    # Transition dataset
    print("\n🔄 TRANSITION DATASET:")
    print(f"  Total samples: {len(transition_df):,}")
    print(f"  Duration: {len(transition_df) / 60:.1f} hours ({len(transition_df) / 1440:.1f} days)")
    print("  Behavior distribution:")
    for behavior in sorted(transition_df['behavior_label'].unique()):
        count = len(transition_df[transition_df['behavior_label'] == behavior])
        pct = count / len(transition_df) * 100
        print(f"    - {behavior:12s}: {count:5,} samples ({pct:5.1f}%)")
    
    # Multi-day datasets
    print(f"\n📅 MULTI-DAY DATASETS ({len(multiday_dfs)} datasets):")
    for dataset_id, num_days, df in multiday_dfs:
        print(f"\n  Dataset #{dataset_id} ({num_days} days):")
        print(f"    Total samples: {len(df):,}")
        print(f"    Duration: {len(df) / 60:.1f} hours")
        print("    Behavior distribution:")
        for behavior in sorted(df['behavior_label'].unique()):
            count = len(df[df['behavior_label'] == behavior])
            pct = count / len(df) * 100
            print(f"      - {behavior:12s}: {count:5,} samples ({pct:5.1f}%)")
    
    # Train/Val/Test splits
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print("\n📂 TRAIN/VALIDATION/TEST SPLITS:")
    print(f"  Total samples: {total_samples:,}")
    print(f"\n  Training set:")
    print(f"    Samples: {len(train_df):,} ({len(train_df)/total_samples*100:.1f}%)")
    for behavior in sorted(train_df['behavior_label'].unique()):
        count = len(train_df[train_df['behavior_label'] == behavior])
        print(f"    - {behavior:12s}: {count:5,} samples")
    
    print(f"\n  Validation set:")
    print(f"    Samples: {len(val_df):,} ({len(val_df)/total_samples*100:.1f}%)")
    for behavior in sorted(val_df['behavior_label'].unique()):
        count = len(val_df[val_df['behavior_label'] == behavior])
        print(f"    - {behavior:12s}: {count:5,} samples")
    
    print(f"\n  Test set:")
    print(f"    Samples: {len(test_df):,} ({len(test_df)/total_samples*100:.1f}%)")
    for behavior in sorted(test_df['behavior_label'].unique()):
        count = len(test_df[test_df['behavior_label'] == behavior])
        print(f"    - {behavior:12s}: {count:5,} samples")
    
    print("\n" + "=" * 70)


def validate_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Validate generated datasets meet requirements."""
    
    print("\n🔍 VALIDATION CHECKS:")
    
    # Check for NaN values
    all_df = pd.concat([train_df, val_df, test_df])
    has_nan = all_df.isnull().any().any()
    print(f"  ✓ No NaN values: {not has_nan}")
    
    # Check split ratios
    total = len(train_df) + len(val_df) + len(test_df)
    train_ratio = len(train_df) / total
    val_ratio = len(val_df) / total
    test_ratio = len(test_df) / total
    
    train_ok = abs(train_ratio - 0.70) <= 0.02
    val_ok = abs(val_ratio - 0.15) <= 0.02
    test_ok = abs(test_ratio - 0.15) <= 0.02
    
    print(f"  ✓ Train ratio: {train_ratio:.3f} (target: 0.70 ±0.02) {'✓' if train_ok else '✗'}")
    print(f"  ✓ Val ratio: {val_ratio:.3f} (target: 0.15 ±0.02) {'✓' if val_ok else '✗'}")
    print(f"  ✓ Test ratio: {test_ratio:.3f} (target: 0.15 ±0.02) {'✓' if test_ok else '✗'}")
    
    # Check for duplicate timestamps
    train_dupes = train_df['timestamp'].duplicated().sum()
    val_dupes = val_df['timestamp'].duplicated().sum()
    test_dupes = test_df['timestamp'].duplicated().sum()
    
    print(f"  ✓ No duplicate timestamps in train: {train_dupes == 0}")
    print(f"  ✓ No duplicate timestamps in val: {val_dupes == 0}")
    print(f"  ✓ No duplicate timestamps in test: {test_dupes == 0}")
    
    # Check value ranges
    temp_ok = (all_df['temp'] >= 37.0).all() and (all_df['temp'] <= 41.0).all()
    acc_ok = (all_df[['Fxa', 'Mya', 'Rza']].abs() <= 3.0).all().all()
    gyro_ok = (all_df[['Sxg', 'Lyg', 'Dzg']].abs() <= 100).all().all()
    
    print(f"  ✓ Temperature in valid range (37-41°C): {temp_ok}")
    print(f"  ✓ Acceleration in valid range (±3g): {acc_ok}")
    print(f"  ✓ Gyroscope in valid range (±100°/s): {gyro_ok}")
    
    # Check all behaviors present
    all_behaviors = set(BEHAVIOR_PATTERNS.keys())
    train_behaviors = set(train_df['behavior_label'].unique())
    val_behaviors = set(val_df['behavior_label'].unique())
    test_behaviors = set(test_df['behavior_label'].unique())
    
    print(f"  ✓ All behaviors in train: {train_behaviors == all_behaviors}")
    print(f"  ✓ All behaviors in val: {val_behaviors == all_behaviors}")
    print(f"  ✓ All behaviors in test: {test_behaviors == all_behaviors}")
    
    return all([train_ok, val_ok, test_ok, not has_nan, temp_ok, acc_ok, gyro_ok])


def main():
    """Main dataset generation pipeline."""
    
    print("=" * 70)
    print("SYNTHETIC DATASET GENERATOR")
    print("=" * 70)
    
    # Initialize generator with seed for reproducibility
    generator = SyntheticDataGenerator(seed=42)
    
    # Create output directory
    output_path = create_output_directory("data/synthetic")
    
    # Generate balanced dataset
    balanced_df = generate_balanced_dataset(
        generator,
        minutes_per_behavior=1100,
        start_date=datetime(2024, 1, 1),
    )
    
    # Generate transition dataset
    transition_df = generate_transition_dataset(
        generator,
        transitions_per_pair=15,
        start_date=datetime(2024, 2, 1),
    )
    
    # Generate multi-day datasets
    multiday_configs = [
        (1, 7, datetime(2024, 3, 1)),    # 7 days
        (2, 10, datetime(2024, 4, 1)),   # 10 days
        (3, 14, datetime(2024, 5, 1)),   # 14 days
        (4, 7, datetime(2024, 6, 1)),    # 7 days
        (5, 10, datetime(2024, 7, 1)),   # 10 days
    ]
    
    multiday_dfs = []
    for dataset_id, num_days, start_date in multiday_configs:
        df = generate_multiday_dataset(
            generator,
            num_days=num_days,
            dataset_id=dataset_id,
            start_date=start_date,
        )
        multiday_dfs.append((dataset_id, num_days, df))
        
        # Save individual multi-day dataset
        save_dataset(df, output_path, f"multiday_{dataset_id}")
    
    # Combine balanced and transition datasets for train/val/test split
    print("\n🔄 Creating train/validation/test splits...")
    combined_df = pd.concat([balanced_df, transition_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_df, val_df, test_df = split_dataset(
        combined_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        stratified=True,
    )
    
    # Save train/val/test splits
    print("\n💾 Saving datasets...")
    save_dataset(train_df, output_path, "train")
    save_dataset(val_df, output_path, "val")
    save_dataset(test_df, output_path, "test")
    
    # Generate metadata
    generate_dataset_metadata(
        balanced_df,
        transition_df,
        multiday_dfs,
        train_df,
        val_df,
        test_df,
        output_path,
    )
    
    # Print summary
    print_summary_statistics(
        balanced_df,
        transition_df,
        multiday_dfs,
        train_df,
        val_df,
        test_df,
    )
    
    # Validate datasets
    validation_passed = validate_datasets(train_df, val_df, test_df)
    
    if validation_passed:
        print("\n✅ All validation checks passed!")
    else:
        print("\n⚠️  Some validation checks failed. Please review.")
    
    print("\n✅ Dataset generation complete!")


if __name__ == "__main__":
    main()
