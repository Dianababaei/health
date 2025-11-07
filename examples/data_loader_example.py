#!/usr/bin/env python3
"""
Example usage of the data_loader module.

This script demonstrates how to:
1. Load a single CSV file with sensor data
2. Load multiple CSV files from a directory
3. Handle different timestamp formats
4. Access and analyze the loaded data
"""

import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import read_sensor_csv, read_sensor_directory
from src.utils.logger import setup_logging, get_logger
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile


def create_sample_csv(filepath, num_rows=100, timestamp_format='standard'):
    """
    Create a sample CSV file with sensor data for demonstration.
    
    Args:
        filepath: Path where to save the CSV file
        num_rows: Number of data rows to generate
        timestamp_format: Format for timestamps ('standard', 'iso8601', or 'epoch')
    """
    print(f"Creating sample CSV file: {filepath}")
    
    data = {
        'timestamp': [],
        'temperature': [],
        'Fxa': [],
        'Mya': [],
        'Rza': [],
        'Sxg': [],
        'Lyg': [],
        'Dzg': []
    }
    
    base_time = datetime(2024, 1, 15, 10, 0, 0)
    
    for i in range(num_rows):
        current_time = base_time + timedelta(minutes=i)
        
        # Format timestamp based on specified format
        if timestamp_format == 'standard':
            timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        elif timestamp_format == 'iso8601':
            timestamp_str = current_time.strftime('%Y-%m-%dT%H:%M:%S')
        elif timestamp_format == 'epoch':
            timestamp_str = str(int(current_time.timestamp()))
        else:
            timestamp_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
        
        data['timestamp'].append(timestamp_str)
        # Generate realistic sensor values with some variation
        data['temperature'].append(round(38.5 + np.random.normal(0, 0.2), 2))
        data['Fxa'].append(round(1.0 + np.random.normal(0, 0.3), 3))
        data['Mya'].append(round(0.5 + np.random.normal(0, 0.2), 3))
        data['Rza'].append(round(2.0 + np.random.normal(0, 0.4), 3))
        data['Sxg'].append(round(0.1 + np.random.normal(0, 0.05), 3))
        data['Lyg'].append(round(0.4 + np.random.normal(0, 0.1), 3))
        data['Dzg'].append(round(1.0 + np.random.normal(0, 0.2), 3))
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"  Created file with {num_rows} rows")


def example_1_single_file():
    """Example 1: Load a single CSV file."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Loading a Single CSV File")
    print("=" * 70)
    
    # Create a temporary directory and sample file
    temp_dir = Path(tempfile.mkdtemp())
    csv_file = temp_dir / "sensor_data.csv"
    
    try:
        # Create sample data
        create_sample_csv(csv_file, num_rows=100)
        
        # Load the CSV file
        print(f"\nLoading CSV file: {csv_file}")
        df = read_sensor_csv(csv_file)
        
        # Display information about the loaded data
        print("\n--- Data Summary ---")
        print(f"Number of records: {len(df)}")
        print(f"Time range: {df.index[0]} to {df.index[-1]}")
        print(f"Columns: {list(df.columns)}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        
        print(f"\n--- Basic Statistics ---")
        print(df.describe())
        
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)


def example_2_different_timestamp_formats():
    """Example 2: Load CSV files with different timestamp formats."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Loading CSV Files with Different Timestamp Formats")
    print("=" * 70)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create files with different timestamp formats
        formats = {
            'standard': 'standard.csv',
            'iso8601': 'iso8601.csv',
            'epoch': 'epoch.csv'
        }
        
        for fmt, filename in formats.items():
            filepath = temp_dir / filename
            create_sample_csv(filepath, num_rows=50, timestamp_format=fmt)
            
            # Load and display
            print(f"\nLoading {filename} (format: {fmt}):")
            df = read_sensor_csv(filepath)
            print(f"  Successfully loaded {len(df)} records")
            print(f"  Time range: {df.index[0]} to {df.index[-1]}")
            print(f"  Average temperature: {df['temperature'].mean():.2f}°C")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def example_3_directory_loading():
    """Example 3: Load multiple CSV files from a directory."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Loading Multiple CSV Files from a Directory")
    print("=" * 70)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create multiple CSV files representing different days
        print("\nCreating multiple CSV files...")
        for day in range(1, 4):
            filename = f"animal_A12345_day{day}.csv"
            filepath = temp_dir / filename
            create_sample_csv(filepath, num_rows=60)
        
        # Also create a file with different pattern
        other_file = temp_dir / "other_data.csv"
        create_sample_csv(other_file, num_rows=30)
        
        # Load all CSV files from directory
        print(f"\n--- Loading all CSV files from {temp_dir} ---")
        df_all = read_sensor_directory(temp_dir)
        print(f"Total records loaded: {len(df_all)}")
        print(f"Time range: {df_all.index[0]} to {df_all.index[-1]}")
        
        # Load with pattern filtering
        print(f"\n--- Loading only 'animal_*.csv' files ---")
        df_filtered = read_sensor_directory(temp_dir, pattern='animal_*.csv')
        print(f"Total records loaded: {len(df_filtered)}")
        print(f"Time range: {df_filtered.index[0]} to {df_filtered.index[-1]}")
        
        # Display statistics
        print(f"\n--- Temperature Statistics ---")
        print(f"Mean: {df_all['temperature'].mean():.2f}°C")
        print(f"Min: {df_all['temperature'].min():.2f}°C")
        print(f"Max: {df_all['temperature'].max():.2f}°C")
        print(f"Std Dev: {df_all['temperature'].std():.2f}°C")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def example_4_data_analysis():
    """Example 4: Perform basic analysis on loaded data."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Basic Data Analysis")
    print("=" * 70)
    
    temp_dir = Path(tempfile.mkdtemp())
    csv_file = temp_dir / "sensor_data.csv"
    
    try:
        # Create sample data with more rows
        create_sample_csv(csv_file, num_rows=1440)  # 24 hours of minute data
        
        # Load the data
        df = read_sensor_csv(csv_file)
        
        print(f"\nLoaded {len(df)} records (24 hours of minute-by-minute data)")
        
        # Resample to hourly averages
        print("\n--- Hourly Averages ---")
        hourly = df.resample('H').mean()
        print(f"Temperature by hour:")
        for hour, row in hourly.head(5).iterrows():
            print(f"  {hour.strftime('%H:%M')}: {row['temperature']:.2f}°C")
        
        # Find periods with high activity (based on acceleration)
        print("\n--- Activity Analysis ---")
        df['total_acceleration'] = np.sqrt(df['Fxa']**2 + df['Mya']**2 + df['Rza']**2)
        high_activity = df[df['total_acceleration'] > df['total_acceleration'].quantile(0.75)]
        print(f"High activity periods: {len(high_activity)} records ({len(high_activity)/len(df)*100:.1f}%)")
        
        # Temperature trend
        print("\n--- Temperature Trend ---")
        temp_change = df['temperature'].iloc[-1] - df['temperature'].iloc[0]
        print(f"Temperature change over 24 hours: {temp_change:+.2f}°C")
        if abs(temp_change) > 0.5:
            print("  Warning: Significant temperature change detected!")
        else:
            print("  Temperature is stable.")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def main():
    """Run all examples."""
    # Initialize logging
    setup_logging(development_mode=True)
    logger = get_logger('artemis.examples.data_loader')
    
    print("=" * 70)
    print("DATA LOADER MODULE - USAGE EXAMPLES")
    print("=" * 70)
    print("\nThis script demonstrates the data_loader module functionality.")
    print("Sample CSV files will be created temporarily for demonstration.")
    
    try:
        # Run examples
        example_1_single_file()
        example_2_different_timestamp_formats()
        example_3_directory_loading()
        example_4_data_analysis()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
