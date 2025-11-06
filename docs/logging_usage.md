# Logging Framework Usage Guide

## Overview

The Artemis Health Monitoring System uses a comprehensive logging framework with three separate log files:

- **System Log** (`logs/system/system.log`): General application flow, data loading, and processing steps
- **Alerts Log** (`logs/alerts/alerts.log`): Layer 3 health alerts (fever, heat stress, estrus, etc.)
- **Training Log** (`logs/training/training.log`): Model training/evaluation metrics and hyperparameters

## Quick Start

### Initialize Logging

At your application's entry point, initialize the logging system:

```python
from src.utils.logger import setup_logging

# Initialize with default settings
setup_logging()

# Or specify a custom config path
setup_logging(config_path='config/config.yaml')

# Or enable development mode (DEBUG level + console output)
setup_logging(development_mode=True)
```

### Basic Usage

#### 1. Using Named Loggers

Create named loggers for different modules following the naming convention: `artemis.<layer>.<component>`

```python
from src.utils.logger import get_logger

# Create a logger for Layer 1 (Physical Behavior)
logger = get_logger('artemis.layer1.posture')

# Log messages at different levels
logger.debug('Processing accelerometer data...')
logger.info('Posture detected: Standing')
logger.warning('Unusual posture pattern detected')
logger.error('Failed to process sensor data')
```

#### 2. Logging Health Alerts (Layer 3)

Use the convenience function for structured health alerts:

```python
from src.utils.logger import log_alert

# Log a fever alert
log_alert(
    severity='WARNING',
    alert_type='fever',
    message='High temperature detected for animal ID 12345',
    triggering_values={
        'temperature': 39.8,
        'threshold': 39.5,
        'activity_level': 'low',
        'duration_minutes': 45
    }
)

# Log a heat stress alert
log_alert(
    severity='ERROR',
    alert_type='heat_stress',
    message='Critical heat stress condition',
    triggering_values={
        'temperature': 40.2,
        'activity_level': 'high',
        'ambient_temp': 35.0
    }
)

# Log an estrus detection
log_alert(
    severity='INFO',
    alert_type='estrus',
    message='Estrus cycle detected',
    triggering_values={
        'temp_rise': 0.4,
        'activity_increase': 1.25,
        'confidence': 0.89
    }
)
```

#### 3. Logging Training Metrics

Use the convenience function for model training logs:

```python
from src.utils.logger import log_training_metrics

# Log training progress
log_training_metrics(
    epoch=10,
    metrics={
        'loss': 0.234,
        'accuracy': 0.89,
        'val_loss': 0.267,
        'val_accuracy': 0.86
    },
    hyperparameters={
        'learning_rate': 0.001,
        'batch_size': 32,
        'optimizer': 'Adam'
    }
)
```

## Configuration

The logging system is configured via `config/config.yaml`:

```yaml
logging:
  # Root logging level
  level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  
  # Log message format
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  
  # Log rotation settings
  rotation:
    type: size  # 'size' or 'time'
    max_bytes: 10485760  # 10MB
    backup_count: 5
    when: midnight  # For time-based rotation
  
  # Handler configurations
  handlers:
    system:
      enabled: true
      level: INFO
      file: logs/system/system.log
    
    alerts:
      enabled: true
      level: INFO
      file: logs/alerts/alerts.log
    
    training:
      enabled: true
      level: DEBUG
      file: logs/training/training.log
    
    console:
      enabled: true
      level: INFO
```

## Log Rotation

Logs are automatically rotated to prevent files from growing unbounded:

### Size-Based Rotation (Default)
- Logs rotate when they reach 10MB
- Keeps 5 backup files (system.log.1, system.log.2, etc.)
- Oldest backup is deleted when limit is reached

### Time-Based Rotation
- Can be configured to rotate at midnight, daily, hourly, etc.
- Configure in `config.yaml` by setting `rotation.type: time`

## Logger Naming Convention

Use a hierarchical naming structure for loggers:

```
artemis.<layer>.<component>.<subcomponent>
```

Examples:
- `artemis.layer1.behavior` - Physical behavior analysis
- `artemis.layer1.posture` - Posture detection
- `artemis.layer2.temperature` - Temperature analysis
- `artemis.layer2.circadian` - Circadian rhythm tracking
- `artemis.layer3.fever_detection` - Fever alert system
- `artemis.layer3.estrus_detection` - Estrus detection
- `artemis.data.loader` - Data loading utilities
- `artemis.training.model` - Model training
- `artemis.alerts` - Health alerts (special logger)
- `artemis.training` - Training logs (special logger)

## Log Levels

Choose appropriate log levels for different scenarios:

- **DEBUG**: Detailed diagnostic information (verbose, for development)
  - Data transformations, intermediate calculations
  - Sensor readings, feature values
  
- **INFO**: General operational messages (normal operation)
  - Application startup/shutdown
  - Processing milestones
  - Successful operations
  
- **WARNING**: Recoverable issues that should be noted
  - Missing optional data
  - Degraded performance
  - Unusual but valid conditions
  
- **ERROR**: Failures that prevent specific operations
  - Failed to load data
  - Processing errors
  - Health alerts that require attention

## Best Practices

### 1. Choose Descriptive Logger Names
```python
# Good
logger = get_logger('artemis.layer1.activity_classifier')

# Less good
logger = get_logger('artemis.classifier')
```

### 2. Include Context in Log Messages
```python
# Good
logger.info(f'Processing sensor data for animal {animal_id}, timestamp {timestamp}')

# Less good
logger.info('Processing data')
```

### 3. Use Appropriate Log Levels
```python
# Good - Different levels for different situations
logger.debug(f'Raw sensor values: temp={temp}, accel={accel}')
logger.info('Data processing completed successfully')
logger.warning('Temperature sensor reading is at upper normal limit')
logger.error('Failed to calculate activity score due to missing data')

# Less good - Everything at INFO level
logger.info('Raw sensor values')
logger.info('Failed to calculate score')
```

### 4. Use Structured Data for Alerts
```python
# Good - Structured triggering values
log_alert(
    severity='WARNING',
    alert_type='fever',
    message='Fever detected',
    triggering_values={'temp': 39.8, 'duration': 45}
)

# Less good - Unstructured message
logger.warning('Fever detected temp 39.8 for 45 mins')
```

### 5. Log Exceptions with Traceback
```python
try:
    process_sensor_data(data)
except Exception as e:
    logger.error(f'Failed to process sensor data: {e}', exc_info=True)
```

## Examples by Use Case

### Application Startup
```python
from src.utils.logger import setup_logging, get_logger

# Initialize logging
setup_logging()

logger = get_logger('artemis.main')
logger.info('Artemis Health Monitoring System starting...')
logger.info('Loading configuration from config/config.yaml')
logger.info('System initialized successfully')
```

### Data Processing
```python
logger = get_logger('artemis.data.processor')

logger.info(f'Loading sensor data from {file_path}')
logger.debug(f'Data shape: {data.shape}, columns: {data.columns}')
logger.info(f'Loaded {len(data)} records')

if missing_values > 0:
    logger.warning(f'{missing_values} missing values detected, applying interpolation')
```

### Health Alert Detection
```python
logger = get_logger('artemis.layer3.fever_detection')

if temperature > fever_threshold and activity < low_activity_threshold:
    logger.warning(f'Potential fever detected for animal {animal_id}')
    
    log_alert(
        severity='WARNING',
        alert_type='fever',
        message=f'Fever detected for animal {animal_id}',
        triggering_values={
            'temperature': temperature,
            'threshold': fever_threshold,
            'activity': activity,
            'timestamp': timestamp
        }
    )
```

### Model Training
```python
logger = get_logger('artemis.training.behavior_model')

logger.info('Starting behavior classification model training')
logger.info(f'Training set size: {len(train_data)}, validation set: {len(val_data)}')

for epoch in range(num_epochs):
    # Training code...
    
    log_training_metrics(
        epoch=epoch,
        metrics={
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        },
        hyperparameters={
            'learning_rate': lr,
            'batch_size': batch_size
        } if epoch == 0 else None  # Only log hyperparams on first epoch
    )

logger.info('Training completed successfully')
```

## Troubleshooting

### Logs Not Being Created
- Ensure `setup_logging()` is called before any logging operations
- Check that the log directories exist or can be created
- Verify file permissions for the log directories

### Duplicate Log Messages
- Check if `setup_logging()` is being called multiple times
- Use `ensure_logging_initialized()` to prevent multiple initializations
- Verify logger propagation settings

### Log Files Growing Too Large
- Adjust `rotation.max_bytes` in config.yaml
- Reduce `rotation.backup_count` to keep fewer old logs
- Consider time-based rotation for predictable log sizes

### Missing Log Messages
- Check the log level configuration (messages below the configured level won't appear)
- Verify the handler is enabled in config.yaml
- Check logger propagation settings for named loggers
