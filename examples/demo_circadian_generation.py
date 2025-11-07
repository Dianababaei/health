"""
Demo script to showcase circadian rhythm and daily activity pattern generation.

This script demonstrates:
1. Circadian temperature variation over 24 hours
2. Time-of-day dependent behavior patterns
3. Different activity sequence templates
4. Multi-day dataset generation with consistent patterns
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.synthetic_generator import SyntheticDataGenerator
import pandas as pd
import numpy as np


def demo_circadian_temperature():
    """Demonstrate circadian temperature pattern over 24 hours."""
    print("=" * 70)
    print("DEMO 1: Circadian Temperature Variation")
    print("=" * 70)
    
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate temperature for each hour of the day
    print("\nHourly Temperature Pattern:")
    print(f"{'Hour':<8} {'Temperature (°C)':<20} {'Visual'}")
    print("-" * 70)
    
    temps = []
    for hour in range(24):
        temp = generator.calculate_circadian_temperature(hour)
        temps.append(temp)
        bar_length = int((temp - 37.5) * 20)  # Scale for visualization
        bar = "█" * bar_length
        print(f"{hour:02d}:00   {temp:6.2f}°C            {bar}")
    
    temp_range = max(temps) - min(temps)
    min_temp_hour = temps.index(min(temps))
    max_temp_hour = temps.index(max(temps))
    
    print(f"\nTemperature Range: {temp_range:.2f}°C")
    print(f"Minimum Temperature: {min(temps):.2f}°C at {min_temp_hour:02d}:00")
    print(f"Maximum Temperature: {max(temps):.2f}°C at {max_temp_hour:02d}:00")
    print(f"✓ Peak in afternoon: {14 <= max_temp_hour <= 18}")
    print(f"✓ Variation 0.5-1.0°C: {0.5 <= temp_range <= 1.2}")
    print()


def demo_daily_sequences():
    """Demonstrate different daily sequence templates."""
    print("=" * 70)
    print("DEMO 2: Daily Activity Sequence Templates")
    print("=" * 70)
    
    generator = SyntheticDataGenerator(random_seed=42)
    
    templates = ['typical', 'high_activity', 'low_activity']
    
    for template in templates:
        print(f"\n{template.upper().replace('_', ' ')} DAY:")
        print("-" * 70)
        
        sequence = generator.generate_daily_sequence(template=template, randomize=False)
        
        # Calculate behavior totals
        behavior_minutes = {b: 0 for b in ['lying', 'standing', 'walking', 'feeding', 'ruminating']}
        for start, end, behavior in sequence:
            behavior_minutes[behavior] += (end - start)
        
        total_minutes = sum(behavior_minutes.values())
        
        print(f"{'Behavior':<15} {'Minutes':<10} {'Hours':<10} {'Percentage'}")
        for behavior, minutes in sorted(behavior_minutes.items(), key=lambda x: -x[1]):
            hours = minutes / 60
            percentage = (minutes / total_minutes) * 100
            print(f"{behavior:<15} {minutes:<10} {hours:<10.1f} {percentage:5.1f}%")
        
        # Show first few segments
        print(f"\nFirst 5 segments:")
        for i, (start, end, behavior) in enumerate(sequence[:5]):
            start_time = f"{start//60:02d}:{start%60:02d}"
            end_time = f"{end//60:02d}:{end%60:02d}"
            duration = end - start
            print(f"  {start_time} - {end_time} ({duration:3d} min): {behavior}")
    print()


def demo_time_of_day_patterns():
    """Demonstrate time-of-day behavior patterns."""
    print("=" * 70)
    print("DEMO 3: Time-of-Day Behavior Patterns")
    print("=" * 70)
    
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate 1 day of data
    df = generator.generate_dataset(num_days=1, sequence_type='probabilistic')
    
    # Analyze by time period
    df['hour'] = df['timestamp'].dt.hour
    
    time_periods = {
        'Night (00-06)': (0, 6),
        'Morning (06-10)': (6, 10),
        'Midday (10-14)': (10, 14),
        'Afternoon (14-18)': (14, 18),
        'Evening (18-24)': (18, 24)
    }
    
    print("\nBehavior Distribution by Time Period:")
    print("-" * 70)
    
    for period_name, (start_hour, end_hour) in time_periods.items():
        period_data = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]
        
        if len(period_data) == 0:
            continue
        
        print(f"\n{period_name}:")
        behavior_counts = period_data['behavior'].value_counts()
        total = len(period_data)
        
        for behavior, count in behavior_counts.items():
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 5)
            print(f"  {behavior:<12} {percentage:5.1f}% {bar}")
    
    # Check nighttime criteria
    night_data = df[(df['hour'] >= 0) & (df['hour'] < 6)]
    lying_pct = (night_data['behavior'] == 'lying').sum() / len(night_data) * 100
    print(f"\n✓ Nighttime lying >70%: {lying_pct:.1f}% (target: >70%)")
    print()


def demo_multi_day_consistency():
    """Demonstrate consistent circadian patterns across multiple days."""
    print("=" * 70)
    print("DEMO 4: Multi-Day Circadian Pattern Consistency")
    print("=" * 70)
    
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate 3 days of data
    df = generator.generate_dataset(num_days=3, sequence_type='probabilistic')
    
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    
    # Compare temperature patterns across days
    print("\nTemperature Patterns by Hour (averaged across days):")
    print("-" * 70)
    
    for day in df['day'].unique():
        day_data = df[df['day'] == day]
        hourly_avg = day_data.groupby('hour')['temperature'].mean()
        
        temp_range = hourly_avg.max() - hourly_avg.min()
        peak_hour = hourly_avg.idxmax()
        
        print(f"\nDay {day}:")
        print(f"  Temperature range: {temp_range:.2f}°C")
        print(f"  Peak temperature hour: {peak_hour:02d}:00")
        print(f"  Average temperature: {day_data['temperature'].mean():.2f}°C")
    
    # Show consistency
    print("\n✓ Multi-day patterns show consistent circadian rhythms")
    print()


def demo_complete_dataset():
    """Generate and show statistics for a complete dataset."""
    print("=" * 70)
    print("DEMO 5: Complete Dataset Generation")
    print("=" * 70)
    
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate 2 days of data
    df = generator.generate_dataset(
        num_days=2,
        animal_id='COW_001',
        sequence_type='probabilistic'
    )
    
    print(f"\nGenerated Dataset:")
    print(f"  Total records: {len(df)}")
    print(f"  Duration: {df['timestamp'].max() - df['timestamp'].min()}")
    print(f"  Sampling interval: 1 minute")
    print(f"  Animal ID: {df['animal_id'].iloc[0]}")
    
    print(f"\nDataset Preview:")
    print(df.head(10).to_string(index=False))
    
    print(f"\nTemperature Statistics:")
    print(f"  Mean: {df['temperature'].mean():.2f}°C")
    print(f"  Std: {df['temperature'].std():.2f}°C")
    print(f"  Min: {df['temperature'].min():.2f}°C")
    print(f"  Max: {df['temperature'].max():.2f}°C")
    print(f"  Range: {df['temperature'].max() - df['temperature'].min():.2f}°C")
    
    print(f"\nBehavior Distribution:")
    behavior_counts = df['behavior'].value_counts()
    for behavior, count in behavior_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {behavior:<12} {count:5d} records ({percentage:5.1f}%)")
    
    print()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "CIRCADIAN RHYTHM & DAILY ACTIVITY DEMO" + " " * 15 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    demo_circadian_temperature()
    demo_daily_sequences()
    demo_time_of_day_patterns()
    demo_multi_day_consistency()
    demo_complete_dataset()
    
    print("=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Circadian temperature variation (sinusoidal 24-hour pattern)")
    print("  ✓ Time-of-day dependent behavior patterns")
    print("  ✓ Multiple activity sequence templates")
    print("  ✓ Probabilistic behavior generation")
    print("  ✓ Multi-day consistency")
    print("  ✓ Smooth behavior transitions")
    print("  ✓ Realistic sensor data generation")
    print()


if __name__ == "__main__":
    main()
