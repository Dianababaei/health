"""
Example usage of the data ingestion module.

Demonstrates batch loading, incremental loading, file monitoring,
and integration with the simulation engine.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.ingestion import DataIngestionModule
from simulation.engine import SimulationEngine


def example_batch_loading():
    """Example: Load CSV file in batch mode."""
    print("=" * 60)
    print("Example 1: Batch Loading")
    print("=" * 60)
    
    # Initialize ingestion module
    ingestion = DataIngestionModule(log_dir='logs')
    
    # Load CSV file
    csv_file = 'tests/fixtures/sample_valid.csv'
    df, summary = ingestion.load_batch(csv_file)
    
    print(f"\nLoaded {len(df)} rows from {csv_file}")
    print(f"Valid rows: {summary.valid_rows}")
    print(f"Skipped rows: {summary.skipped_rows}")
    print(f"Errors: {len(summary.errors)}")
    print(f"Warnings: {len(summary.warnings)}")
    
    if len(df) > 0:
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nSensor statistics:")
        sensor_cols = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
        print(df[sensor_cols].describe())
    
    return df, summary


def example_incremental_loading():
    """Example: Load CSV file incrementally."""
    print("\n" + "=" * 60)
    print("Example 2: Incremental Loading")
    print("=" * 60)
    
    # Initialize ingestion module
    ingestion = DataIngestionModule(log_dir='logs')
    
    csv_file = 'tests/fixtures/sample_valid.csv'
    
    # First read (loads all data)
    print("\nFirst read:")
    df1, summary1 = ingestion.load_incremental(csv_file)
    print(f"  Loaded {len(df1)} rows")
    print(f"  File position: {ingestion.get_file_position(csv_file)}")
    
    # Second read (no new data)
    print("\nSecond read (no new data):")
    df2, summary2 = ingestion.load_incremental(csv_file)
    print(f"  Loaded {len(df2)} rows")
    
    # Reset and read again
    print("\nReset position and read again:")
    ingestion.reset_file_position(csv_file)
    df3, summary3 = ingestion.load_incremental(csv_file)
    print(f"  Loaded {len(df3)} rows")
    
    return df1, summary1


def example_malformed_data():
    """Example: Handle malformed data gracefully."""
    print("\n" + "=" * 60)
    print("Example 3: Handling Malformed Data")
    print("=" * 60)
    
    # Initialize ingestion module
    ingestion = DataIngestionModule(log_dir='logs')
    
    # Load malformed CSV
    csv_file = 'tests/fixtures/sample_malformed.csv'
    df, summary = ingestion.load_batch(csv_file)
    
    print(f"\nLoaded {len(df)} rows from malformed file")
    print(f"Valid rows: {summary.valid_rows}")
    print(f"Skipped rows: {summary.skipped_rows}")
    
    print(f"\nErrors found: {len(summary.errors)}")
    if summary.errors:
        print("Sample errors:")
        for error in summary.errors[:5]:
            print(f"  - {error}")
    
    print(f"\nWarnings found: {len(summary.warnings)}")
    if summary.warnings:
        print("Sample warnings:")
        for warning in summary.warnings[:5]:
            print(f"  - {warning}")
    
    # Export summary report
    report_file = 'logs/malformed_data_summary.txt'
    ingestion.export_summary(summary, report_file)
    print(f"\nDetailed summary exported to: {report_file}")
    
    return df, summary


def example_edge_cases():
    """Example: Handle edge cases (Unix timestamps, optional columns)."""
    print("\n" + "=" * 60)
    print("Example 4: Edge Cases (Unix Timestamps)")
    print("=" * 60)
    
    # Initialize ingestion module
    ingestion = DataIngestionModule(log_dir='logs')
    
    # Load CSV with Unix timestamps
    csv_file = 'tests/fixtures/sample_edge_cases.csv'
    df, summary = ingestion.load_batch(csv_file)
    
    print(f"\nLoaded {len(df)} rows with Unix timestamps")
    print(f"Valid rows: {summary.valid_rows}")
    
    if len(df) > 0:
        print("\nTimestamp format detected:")
        print(f"  First timestamp: {df['timestamp'].iloc[0]}")
        print(f"  Last timestamp: {df['timestamp'].iloc[-1]}")
        
        print("\nOptional columns found:")
        optional_cols = ['cow_id', 'sensor_id']
        for col in optional_cols:
            if col in df.columns:
                print(f"  - {col}: {df[col].iloc[0]}")
    
    return df, summary


def example_integration_with_simulation():
    """Example: Generate simulated data and ingest it."""
    print("\n" + "=" * 60)
    print("Example 5: Integration with Simulation Engine")
    print("=" * 60)
    
    # Generate simulated data
    print("\nGenerating 6 hours of simulated sensor data...")
    engine = SimulationEngine(
        baseline_temperature=38.5,
        sampling_rate=1.0,
        random_seed=42
    )
    
    df_simulated = engine.generate_continuous_data(
        duration_hours=6,
        start_datetime=datetime(2024, 1, 1, 0, 0, 0),
        include_stress=True,
        stress_probability=0.05
    )
    
    print(f"Generated {len(df_simulated)} samples")
    
    # Export to CSV
    output_file = 'data/simulated/example_6h.csv'
    os.makedirs('data/simulated', exist_ok=True)
    engine.export_to_csv(df_simulated, output_file)
    
    # Ingest the simulated data
    print(f"\nIngesting simulated data from {output_file}...")
    ingestion = DataIngestionModule(log_dir='logs')
    df_loaded, summary = ingestion.load_batch(output_file)
    
    print(f"\nIngestion results:")
    print(f"  Generated: {len(df_simulated)} rows")
    print(f"  Loaded: {len(df_loaded)} rows")
    print(f"  Valid: {summary.valid_rows} rows")
    print(f"  Errors: {len(summary.errors)}")
    
    # Verify data integrity
    if len(df_simulated) == len(df_loaded):
        print("\n✓ All generated data successfully ingested!")
    
    return df_loaded, summary


def example_file_monitoring():
    """Example: Monitor file for new data."""
    print("\n" + "=" * 60)
    print("Example 6: File Monitoring (Simulated Real-Time)")
    print("=" * 60)
    
    print("\nThis example would monitor a file for new data.")
    print("For demonstration, we'll do a limited number of checks.")
    
    # Initialize ingestion module
    ingestion = DataIngestionModule(log_dir='logs')
    
    # Track received data
    received_batches = []
    
    def data_callback(df, summary):
        """Callback for new data."""
        received_batches.append(len(df))
        print(f"  Received batch with {len(df)} rows")
    
    # Monitor file for 3 iterations (3 seconds)
    csv_file = 'tests/fixtures/sample_valid.csv'
    
    # Reset position to ensure we get data on first check
    ingestion.reset_file_position(csv_file)
    
    print(f"\nMonitoring {csv_file} for new data...")
    print("(Will check 3 times with 1 second interval)")
    
    ingestion.monitor_file(
        csv_file,
        interval_seconds=1,
        callback=data_callback,
        max_iterations=3
    )
    
    print(f"\nMonitoring complete!")
    print(f"Total batches received: {len(received_batches)}")
    print(f"Total rows received: {sum(received_batches)}")


def example_large_file_chunked():
    """Example: Load large file with chunked processing."""
    print("\n" + "=" * 60)
    print("Example 7: Large File with Chunked Processing")
    print("=" * 60)
    
    # Generate a larger dataset
    print("\nGenerating 24 hours of simulated data...")
    engine = SimulationEngine(random_seed=42)
    df_large = engine.generate_continuous_data(
        duration_hours=24,
        start_datetime=datetime(2024, 1, 1, 0, 0, 0),
        include_stress=True
    )
    
    print(f"Generated {len(df_large)} samples")
    
    # Export
    output_file = 'data/simulated/example_24h.csv'
    os.makedirs('data/simulated', exist_ok=True)
    engine.export_to_csv(df_large, output_file)
    
    # Load with chunking
    print(f"\nLoading with chunk size of 500...")
    ingestion = DataIngestionModule(log_dir='logs')
    df_loaded, summary = ingestion.load_batch(
        output_file,
        chunk_size=500
    )
    
    print(f"\nChunked loading results:")
    print(f"  Total rows loaded: {len(df_loaded)}")
    print(f"  Valid rows: {summary.valid_rows}")
    print(f"  Processing time: {(summary.end_time - summary.start_time).total_seconds():.2f}s")
    
    return df_loaded, summary


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "DATA INGESTION MODULE EXAMPLES" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # Run examples
        example_batch_loading()
        example_incremental_loading()
        example_malformed_data()
        example_edge_cases()
        example_integration_with_simulation()
        example_file_monitoring()
        example_large_file_chunked()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
