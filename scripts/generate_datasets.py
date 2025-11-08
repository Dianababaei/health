#!/usr/bin/env python3
"""
Dataset Generation Script

Command-line script to generate simulated cattle sensor datasets at different time scales.

Usage:
    # Generate all datasets
    python scripts/generate_datasets.py

    # Generate specific dataset
    python scripts/generate_datasets.py --dataset short
    python scripts/generate_datasets.py --dataset medium
    python scripts/generate_datasets.py --dataset long --duration 180

    # Specify output directory and seed
    python scripts/generate_datasets.py --output data/simulated --seed 42

    # Generate without splits
    python scripts/generate_datasets.py --no-splits
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation.dataset_generator import (
    generate_short_term_dataset,
    generate_medium_term_dataset,
    generate_long_term_dataset,
    generate_all_datasets
)


def print_header():
    """Print script header."""
    print("=" * 70)
    print("  Artemis Health - Dataset Generation Script")
    print("  Simulated Cattle Sensor Data with Ground Truth Labels")
    print("=" * 70)
    print()


def print_dataset_info(result: dict):
    """Print information about generated dataset."""
    print(f"\n  Dataset: {result['dataset_name']}")
    print(f"  CSV Path: {result['paths']['csv_path']}")
    print(f"  Metadata: {result['paths']['metadata_path']}")
    
    if 'split_paths' in result:
        print(f"  Train/Val/Test Splits:")
        for split_name, split_path in result['split_paths'].items():
            stats = result['split_statistics'][split_name]
            print(f"    - {split_name}: {stats['num_samples']} samples ({stats['percentage']:.1f}%)")
    
    print()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate simulated cattle sensor datasets with ground truth labels",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset',
        choices=['short', 'medium', 'long', 'all'],
        default='all',
        help='Which dataset(s) to generate (default: all)'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        choices=[90, 180],
        default=90,
        help='Duration in days for long-term dataset (default: 90)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/simulated',
        help='Output directory (default: data/simulated)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )
    
    parser.add_argument(
        '--no-splits',
        action='store_true',
        help='Skip generation of train/val/test splits'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    if not args.quiet:
        print_header()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not args.quiet:
        print(f"Output directory: {output_dir}")
        print(f"Random seed: {args.seed if args.seed else 'None (random)'}")
        print(f"Generate splits: {not args.no_splits}")
        print()
    
    start_time = datetime.now()
    
    try:
        if args.dataset == 'all':
            if not args.quiet:
                print("Generating all datasets...")
                print()
            
            results = generate_all_datasets(
                output_dir=str(output_dir),
                seed=args.seed
            )
            
            if not args.quiet:
                print("\nSummary:")
                for dataset_type, result in results.items():
                    print_dataset_info(result)
        
        elif args.dataset == 'short':
            if not args.quiet:
                print("Generating short-term dataset (7 days)...")
            
            result = generate_short_term_dataset(
                animal_id="cow_short_001",
                seed=args.seed,
                output_dir=str(output_dir)
            )
            
            if not args.quiet:
                print_dataset_info(result)
        
        elif args.dataset == 'medium':
            if not args.quiet:
                print("Generating medium-term dataset (30 days)...")
            
            result = generate_medium_term_dataset(
                animal_id="cow_medium_001",
                seed=args.seed,
                output_dir=str(output_dir)
            )
            
            if not args.quiet:
                print_dataset_info(result)
        
        elif args.dataset == 'long':
            if not args.quiet:
                print(f"Generating long-term dataset ({args.duration} days)...")
            
            result = generate_long_term_dataset(
                duration_days=args.duration,
                animal_id="cow_long_001",
                seed=args.seed,
                output_dir=str(output_dir)
            )
            
            if not args.quiet:
                print_dataset_info(result)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if not args.quiet:
            print("=" * 70)
            print(f"✓ Dataset generation completed successfully!")
            print(f"  Total time: {duration:.1f} seconds")
            print("=" * 70)
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error during dataset generation: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
