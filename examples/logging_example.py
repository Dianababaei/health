#!/usr/bin/env python3
"""
Example usage of the Artemis Health Monitoring System logging framework.

This script demonstrates:
1. Initializing the logging system
2. Using named loggers for different components
3. Logging health alerts
4. Logging training metrics
5. Different log levels and their appropriate uses
"""

from src.utils.logger import setup_logging, get_logger, log_alert, log_training_metrics
import time


def main():
    """Main example function demonstrating logging capabilities."""
    
    # Initialize logging system
    print("Initializing logging system...")
    setup_logging(development_mode=True)  # Use development mode for this example
    
    # Create loggers for different components
    main_logger = get_logger('artemis.main')
    layer1_logger = get_logger('artemis.layer1.behavior')
    layer2_logger = get_logger('artemis.layer2.temperature')
    layer3_logger = get_logger('artemis.layer3.alerts')
    training_logger = get_logger('artemis.training.model')
    
    # Example 1: Application startup
    main_logger.info("=" * 60)
    main_logger.info("Artemis Health Monitoring System - Logging Example")
    main_logger.info("=" * 60)
    main_logger.info("System initialized successfully")
    
    # Example 2: Data processing logs (Layer 1 - Behavior)
    main_logger.info("\n--- Example 1: Data Processing (Layer 1) ---")
    layer1_logger.info("Starting behavior analysis for animal ID: A12345")
    layer1_logger.debug("Loading sensor data from file: sensor_data_20240115.csv")
    layer1_logger.debug("Data shape: (1440, 7) - 24 hours of minute-by-minute data")
    layer1_logger.info("Analyzing posture patterns...")
    layer1_logger.debug("Detected postures: Standing (60%), Lying (35%), Walking (5%)")
    layer1_logger.info("Behavior analysis completed successfully")
    
    # Example 3: Temperature analysis with warnings (Layer 2)
    main_logger.info("\n--- Example 2: Temperature Analysis (Layer 2) ---")
    layer2_logger.info("Starting temperature trend analysis")
    layer2_logger.debug("Average temperature: 38.7°C, Min: 38.2°C, Max: 39.1°C")
    layer2_logger.warning("Temperature approaching upper threshold: 39.1°C (threshold: 39.5°C)")
    layer2_logger.info("Circadian rhythm pattern: Normal")
    
    # Example 4: Health alerts (Layer 3)
    main_logger.info("\n--- Example 3: Health Alerts (Layer 3) ---")
    
    # Fever alert
    layer3_logger.warning("Fever condition detected for animal A12345")
    log_alert(
        severity='WARNING',
        alert_type='fever',
        message='Elevated temperature with reduced activity detected',
        triggering_values={
            'animal_id': 'A12345',
            'temperature': 39.8,
            'threshold': 39.5,
            'activity_level': 0.32,
            'duration_minutes': 45
        }
    )
    
    time.sleep(0.1)  # Small delay for visual separation
    
    # Heat stress alert
    layer3_logger.error("Critical heat stress condition detected")
    log_alert(
        severity='ERROR',
        alert_type='heat_stress',
        message='High temperature combined with high activity - immediate attention required',
        triggering_values={
            'animal_id': 'A67890',
            'temperature': 40.2,
            'activity_level': 0.85,
            'ambient_temperature': 35.0,
            'humidity': 78
        }
    )
    
    time.sleep(0.1)
    
    # Estrus detection
    layer3_logger.info("Estrus cycle detected")
    log_alert(
        severity='INFO',
        alert_type='estrus',
        message='Fertility window detected - optimal breeding time',
        triggering_values={
            'animal_id': 'A11111',
            'temperature_rise': 0.4,
            'activity_increase': 1.25,
            'confidence_score': 0.89,
            'estimated_window_hours': 18
        }
    )
    
    # Example 5: Model training logs
    main_logger.info("\n--- Example 4: Model Training ---")
    training_logger.info("Starting posture classification model training")
    training_logger.info("Training dataset: 50,000 samples, Validation: 10,000 samples")
    
    # Simulate training epochs
    for epoch in range(1, 6):
        # Simulate improving metrics
        train_loss = 0.5 - (epoch * 0.06)
        train_acc = 0.70 + (epoch * 0.04)
        val_loss = 0.55 - (epoch * 0.05)
        val_acc = 0.68 + (epoch * 0.04)
        
        log_training_metrics(
            epoch=epoch,
            metrics={
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': 0.001
            },
            hyperparameters={
                'batch_size': 32,
                'optimizer': 'Adam',
                'learning_rate': 0.001
            } if epoch == 1 else None  # Only log hyperparams on first epoch
        )
        
        time.sleep(0.05)  # Small delay between epochs
    
    training_logger.info("Training completed successfully")
    training_logger.info("Model saved to: models/posture_classifier_v1.pkl")
    
    # Example 6: Error handling
    main_logger.info("\n--- Example 5: Error Handling ---")
    try:
        # Simulate an error
        raise ValueError("Sensor data validation failed: Temperature out of range (-10°C)")
    except Exception as e:
        layer2_logger.error(f"Error processing temperature data: {e}", exc_info=True)
    
    # Example 7: Different log levels demonstration
    main_logger.info("\n--- Example 6: Log Levels Demonstration ---")
    demo_logger = get_logger('artemis.demo')
    
    demo_logger.debug("DEBUG: Detailed diagnostic information (very verbose)")
    demo_logger.info("INFO: General informational messages about normal operations")
    demo_logger.warning("WARNING: Something unexpected happened, but we can continue")
    demo_logger.error("ERROR: A serious problem occurred that prevented an operation")
    
    # Final message
    main_logger.info("\n" + "=" * 60)
    main_logger.info("Logging example completed successfully")
    main_logger.info("Check the log files in:")
    main_logger.info("  - logs/system/system.log")
    main_logger.info("  - logs/alerts/alerts.log")
    main_logger.info("  - logs/training/training.log")
    main_logger.info("=" * 60)


if __name__ == '__main__':
    main()
