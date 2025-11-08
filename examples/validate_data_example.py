"""
Example: Using the Data Validation Module

This script demonstrates how to use the data validation module
to validate livestock sensor data.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import validate_sensor_data, DataValidator, ValidationSeverity


def create_sample_data():
    """Create sample sensor data with various quality issues."""
    print("Creating sample sensor data...")
    
    # Create a mix of valid and invalid data
    timestamps = pd.date_range('2024-01-01 00:00', periods=100, freq='1min')
    
    data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.random.uniform(37.5, 39.5, 100),
        'fxa': np.random.uniform(-1.0, 1.0, 100),
        'mya': np.random.uniform(-1.0, 1.0, 100),
        'rza': np.random.uniform(-1.0, 0.0, 100),
        'sxg': np.random.uniform(-50.0, 50.0, 100),
        'lyg': np.random.uniform(-50.0, 50.0, 100),
        'dzg': np.random.uniform(-50.0, 50.0, 100)
    })
    
    # Introduce some issues for demonstration
    
    # 1. Missing value
    data.loc[10, 'temperature'] = np.nan
    
    # 2. Severe fever (critical)
    data.loc[20, 'temperature'] = 43.0
    
    # 3. Hypothermia (critical)
    data.loc[30, 'temperature'] = 34.0
    
    # 4. Out-of-range temperature (warning)
    data.loc[40, 'temperature'] = 34.8
    
    # 5. Extreme acceleration (critical)
    data.loc[50, 'fxa'] = 6.5
    
    # 6. Out-of-range acceleration (warning)
    data.loc[60, 'fxa'] = 2.5
    
    return data


def example_basic_validation():
    """Example 1: Basic validation using convenience function."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Validation")
    print("="*70)
    
    # Create sample data
    data = create_sample_data()
    print(f"\nOriginal data: {len(data)} records")
    
    # Perform validation
    result = validate_sensor_data(data)
    
    # Display results
    summary = result['summary']
    print(f"\n--- Validation Results ---")
    print(f"Total records:    {summary['total_records']}")
    print(f"Clean records:    {summary['clean_records']} ({summary['clean_records']/summary['total_records']*100:.1f}%)")
    print(f"Flagged records:  {summary['flagged_records']} ({summary['flagged_records']/summary['total_records']*100:.1f}%)")
    print(f"\nIssues by severity:")
    print(f"  ERRORS:   {summary['error_count']}")
    print(f"  WARNINGS: {summary['warning_count']}")
    print(f"  INFO:     {summary['info_count']}")
    print(f"\nIssues by category:")
    for category, count in summary['issues_by_category'].items():
        print(f"  {category}: {count}")
    
    return result


def example_detailed_analysis(result):
    """Example 2: Detailed analysis of validation issues."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Detailed Issue Analysis")
    print("="*70)
    
    report = result['report']
    
    # Group issues by severity
    print("\n--- ERROR Level Issues ---")
    errors = [i for i in report.issues if i.severity == ValidationSeverity.ERROR]
    for issue in errors[:5]:  # Show first 5
        print(f"\nRow {issue.row_index}: {issue.message}")
        if issue.column:
            print(f"  Column: {issue.column}")
        if issue.value is not None:
            print(f"  Value: {issue.value}")
    
    if len(errors) > 5:
        print(f"\n... and {len(errors) - 5} more errors")
    
    print("\n--- WARNING Level Issues ---")
    warnings = [i for i in report.issues if i.severity == ValidationSeverity.WARNING]
    for issue in warnings[:5]:  # Show first 5
        print(f"\nRow {issue.row_index}: {issue.message}")
        if issue.column:
            print(f"  Column: {issue.column}")
        if issue.value is not None:
            print(f"  Value: {issue.value}")
    
    if len(warnings) > 5:
        print(f"\n... and {len(warnings) - 5} more warnings")


def example_custom_settings():
    """Example 3: Validation with custom settings."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Validation Settings")
    print("="*70)
    
    # Create sample data
    data = create_sample_data()
    
    # Create validator with custom settings
    validator = DataValidator(
        gap_threshold_minutes=10,  # Allow larger gaps
    )
    
    print("\nUsing custom settings:")
    print("  - Gap threshold: 10 minutes (instead of default 5)")
    
    # Perform validation
    clean_data, flagged_data, report = validator.validate(data)
    
    summary = report.get_summary()
    print(f"\nResults with custom settings:")
    print(f"  Clean records: {summary['clean_records']}")
    print(f"  Flagged records: {summary['flagged_records']}")


def example_export_results(result):
    """Example 4: Exporting validation results."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Exporting Results")
    print("="*70)
    
    # Create output directory
    output_dir = 'validation_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Export clean data
    clean_path = os.path.join(output_dir, 'clean_data.csv')
    result['clean_data'].to_csv(clean_path, index=False)
    print(f"\nClean data exported to: {clean_path}")
    
    # Export flagged data
    flagged_path = os.path.join(output_dir, 'flagged_data.csv')
    result['flagged_data'].to_csv(flagged_path, index=False)
    print(f"Flagged data exported to: {flagged_path}")
    
    # Export issues as DataFrame
    issues_df = result['report'].to_dataframe()
    issues_path = os.path.join(output_dir, 'validation_issues.csv')
    issues_df.to_csv(issues_path, index=False)
    print(f"Validation issues exported to: {issues_path}")
    
    # Export summary as JSON
    import json
    summary_path = os.path.join(output_dir, 'validation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(result['summary'], f, indent=2)
    print(f"Validation summary exported to: {summary_path}")
    
    print(f"\nAll validation results exported to '{output_dir}/' directory")


def example_integration_pipeline():
    """Example 5: Integration with data processing pipeline."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Integration Pipeline")
    print("="*70)
    
    # Simulate a complete data processing pipeline
    print("\n1. Loading data...")
    data = create_sample_data()
    print(f"   Loaded {len(data)} records")
    
    print("\n2. Validating data...")
    result = validate_sensor_data(data)
    summary = result['summary']
    print(f"   Clean: {summary['clean_records']}, Flagged: {summary['flagged_records']}")
    
    print("\n3. Processing decision...")
    if summary['error_count'] > 0:
        print(f"   WARNING: {summary['error_count']} critical errors found!")
        print("   Recommend reviewing data source.")
    
    if summary['flagged_records'] / summary['total_records'] > 0.1:
        print(f"   WARNING: {summary['flagged_records']/summary['total_records']*100:.1f}% of data flagged!")
        print("   Consider investigating data quality issues.")
    else:
        print("   Data quality acceptable for processing.")
    
    print("\n4. Continuing with clean data...")
    clean_data = result['clean_data']
    print(f"   Processing {len(clean_data)} clean records...")
    
    # Here you would continue with your analysis
    print("   [Further analysis would go here]")
    
    return clean_data


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("DATA VALIDATION MODULE - EXAMPLES")
    print("="*70)
    
    # Example 1: Basic validation
    result = example_basic_validation()
    
    # Example 2: Detailed analysis
    example_detailed_analysis(result)
    
    # Example 3: Custom settings
    example_custom_settings()
    
    # Example 4: Export results
    example_export_results(result)
    
    # Example 5: Integration pipeline
    example_integration_pipeline()
    
    print("\n" + "="*70)
    print("All examples completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
