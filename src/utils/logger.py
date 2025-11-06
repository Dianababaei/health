"""
Logging Framework for Artemis Health Monitoring System

This module provides comprehensive logging setup with separate handlers for:
- System logs: General application flow and processing
- Alerts logs: Layer 3 health alerts (fever, heat stress, estrus, etc.)
- Training logs: Model training/evaluation metrics and hyperparameters

Features:
- Log rotation to prevent unbounded file growth
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Named loggers for different modules
- Console output for development
- Integration with config.yaml for settings
"""

import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


# Default logging configuration
DEFAULT_CONFIG = {
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'date_format': '%Y-%m-%d %H:%M:%S',
        'rotation': {
            'type': 'size',  # 'size' or 'time'
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5,
            'when': 'midnight',  # For time-based rotation
        },
        'handlers': {
            'system': {
                'enabled': True,
                'level': 'INFO',
                'file': 'logs/system/system.log',
            },
            'alerts': {
                'enabled': True,
                'level': 'INFO',
                'file': 'logs/alerts/alerts.log',
            },
            'training': {
                'enabled': True,
                'level': 'DEBUG',
                'file': 'logs/training/training.log',
            },
            'console': {
                'enabled': True,
                'level': 'INFO',
            },
        },
    }
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load logging configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default path.
        
    Returns:
        Dictionary containing logging configuration
    """
    if config_path is None:
        config_path = 'config/config.yaml'
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and 'logging' in config:
                    # Merge with defaults to ensure all keys exist
                    merged_config = DEFAULT_CONFIG.copy()
                    merged_config['logging'].update(config['logging'])
                    return merged_config
    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        print("Using default logging configuration")
    
    return DEFAULT_CONFIG


def create_log_directory(log_file_path: str) -> None:
    """
    Create directory for log file if it doesn't exist.
    
    Args:
        log_file_path: Path to log file
    """
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)


def create_rotating_handler(
    log_file: str,
    level: str,
    log_format: str,
    rotation_config: Dict[str, Any]
) -> logging.Handler:
    """
    Create a rotating file handler based on rotation configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Format string for log messages
        rotation_config: Dictionary with rotation settings
        
    Returns:
        Configured rotating file handler
    """
    create_log_directory(log_file)
    
    rotation_type = rotation_config.get('type', 'size')
    
    if rotation_type == 'time':
        # Time-based rotation (daily by default)
        handler = TimedRotatingFileHandler(
            log_file,
            when=rotation_config.get('when', 'midnight'),
            backupCount=rotation_config.get('backup_count', 5)
        )
    else:
        # Size-based rotation (10MB by default)
        handler = RotatingFileHandler(
            log_file,
            maxBytes=rotation_config.get('max_bytes', 10485760),
            backupCount=rotation_config.get('backup_count', 5)
        )
    
    handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    
    return handler


def create_console_handler(level: str, log_format: str) -> logging.StreamHandler:
    """
    Create a console handler for stdout output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: Format string for log messages
        
    Returns:
        Configured console handler
    """
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    
    return handler


def setup_logging(config_path: Optional[str] = None, development_mode: bool = False) -> None:
    """
    Initialize logging system with separate handlers for system, alerts, and training logs.
    
    This function should be called once at application startup to configure all loggers.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default path.
        development_mode: If True, sets DEBUG level and enables console output
        
    Example:
        >>> from src.utils.logger import setup_logging
        >>> setup_logging()
        >>> 
        >>> # Use named loggers in your modules
        >>> logger = logging.getLogger('artemis.layer1')
        >>> logger.info('Processing sensor data...')
    """
    # Load configuration
    config = load_config(config_path)
    logging_config = config['logging']
    
    # Override level if in development mode
    if development_mode:
        logging_config['level'] = 'DEBUG'
        logging_config['handlers']['console']['enabled'] = True
        logging_config['handlers']['console']['level'] = 'DEBUG'
    
    # Get log format
    log_format = logging_config.get('format', DEFAULT_CONFIG['logging']['format'])
    rotation_config = logging_config.get('rotation', DEFAULT_CONFIG['logging']['rotation'])
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_config['level'].upper()))
    
    # Remove any existing handlers
    root_logger.handlers.clear()
    
    # Create handlers for each log type
    handlers_config = logging_config.get('handlers', {})
    
    # System log handler
    if handlers_config.get('system', {}).get('enabled', True):
        system_config = handlers_config['system']
        system_handler = create_rotating_handler(
            system_config.get('file', 'logs/system/system.log'),
            system_config.get('level', 'INFO'),
            log_format,
            rotation_config
        )
        root_logger.addHandler(system_handler)
    
    # Alerts log handler (for Layer 3 health alerts)
    if handlers_config.get('alerts', {}).get('enabled', True):
        alerts_config = handlers_config['alerts']
        alerts_handler = create_rotating_handler(
            alerts_config.get('file', 'logs/alerts/alerts.log'),
            alerts_config.get('level', 'INFO'),
            log_format,
            rotation_config
        )
        # Create dedicated alerts logger
        alerts_logger = logging.getLogger('artemis.alerts')
        alerts_logger.addHandler(alerts_handler)
        alerts_logger.setLevel(getattr(logging, alerts_config.get('level', 'INFO').upper()))
        alerts_logger.propagate = False  # Don't propagate to root logger
    
    # Training log handler (for model training/evaluation)
    if handlers_config.get('training', {}).get('enabled', True):
        training_config = handlers_config['training']
        training_handler = create_rotating_handler(
            training_config.get('file', 'logs/training/training.log'),
            training_config.get('level', 'DEBUG'),
            log_format,
            rotation_config
        )
        # Create dedicated training logger
        training_logger = logging.getLogger('artemis.training')
        training_logger.addHandler(training_handler)
        training_logger.setLevel(getattr(logging, training_config.get('level', 'DEBUG').upper()))
        training_logger.propagate = False  # Don't propagate to root logger
    
    # Console handler (for development)
    if handlers_config.get('console', {}).get('enabled', True):
        console_config = handlers_config['console']
        console_handler = create_console_handler(
            console_config.get('level', 'INFO'),
            log_format
        )
        root_logger.addHandler(console_handler)
    
    # Log initialization message
    root_logger.info("Logging system initialized successfully")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger for a specific module.
    
    Use this function to create loggers for different components of the system.
    Naming convention: 'artemis.<layer>.<component>'
    
    Args:
        name: Logger name (e.g., 'artemis.layer1.behavior', 'artemis.layer2.physiology')
        
    Returns:
        Configured logger instance
        
    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger('artemis.layer1.posture')
        >>> logger.info('Detecting posture...')
        >>> logger.warning('Unusual posture detected')
    """
    return logging.getLogger(name)


def log_alert(
    severity: str,
    alert_type: str,
    message: str,
    triggering_values: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a health alert to the alerts log file.
    
    This is a convenience function for Layer 3 health alerts with structured information.
    
    Args:
        severity: Alert severity (INFO, WARNING, ERROR)
        alert_type: Type of alert (fever, heat_stress, estrus, etc.)
        message: Alert message
        triggering_values: Dictionary of values that triggered the alert
        
    Example:
        >>> from src.utils.logger import log_alert
        >>> log_alert(
        ...     severity='WARNING',
        ...     alert_type='fever',
        ...     message='High temperature detected',
        ...     triggering_values={'temperature': 39.8, 'threshold': 39.5, 'activity': 'low'}
        ... )
    """
    alerts_logger = logging.getLogger('artemis.alerts')
    
    # Format triggering values if provided
    values_str = ""
    if triggering_values:
        values_str = f" | Triggering values: {triggering_values}"
    
    log_message = f"[{alert_type.upper()}] {message}{values_str}"
    
    # Log at appropriate level
    level = getattr(logging, severity.upper(), logging.INFO)
    alerts_logger.log(level, log_message)


def log_training_metrics(
    epoch: int,
    metrics: Dict[str, float],
    hyperparameters: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log training metrics and hyperparameters to the training log file.
    
    This is a convenience function for logging model training information.
    
    Args:
        epoch: Training epoch number
        metrics: Dictionary of metric names and values
        hyperparameters: Optional dictionary of hyperparameters
        
    Example:
        >>> from src.utils.logger import log_training_metrics
        >>> log_training_metrics(
        ...     epoch=10,
        ...     metrics={'loss': 0.234, 'accuracy': 0.89, 'val_loss': 0.267},
        ...     hyperparameters={'learning_rate': 0.001, 'batch_size': 32}
        ... )
    """
    training_logger = logging.getLogger('artemis.training')
    
    # Format metrics
    metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" 
                              for k, v in metrics.items()])
    
    log_message = f"Epoch {epoch} | {metrics_str}"
    
    # Add hyperparameters if provided
    if hyperparameters:
        hp_str = " | ".join([f"{k}: {v}" for k, v in hyperparameters.items()])
        log_message += f" | Hyperparameters: {hp_str}"
    
    training_logger.info(log_message)


# Module-level initialization guard
_logging_initialized = False


def ensure_logging_initialized(config_path: Optional[str] = None) -> None:
    """
    Ensure logging is initialized only once.
    
    This function can be called multiple times safely; it will only initialize logging once.
    
    Args:
        config_path: Path to config.yaml file
    """
    global _logging_initialized
    if not _logging_initialized:
        setup_logging(config_path)
        _logging_initialized = True
