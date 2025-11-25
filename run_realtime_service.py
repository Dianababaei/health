#!/usr/bin/env python3
"""
Realtime Livestock Health Monitoring Service
==============================================

Main service entry point that orchestrates MQTT subscriber and detector scheduler
with proper lifecycle management and graceful shutdown handling.

Usage:
    python run_realtime_service.py

Exit Codes:
    0   - Clean shutdown
    1   - Configuration error
    2   - Database connection error
    3   - MQTT connection error (after retries)
    99  - Unexpected exception
"""

import sys
import signal
import time
import logging
import sqlite3
import traceback
import yaml
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Version info
__version__ = "1.0.0"

# Exit codes
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_DATABASE_ERROR = 2
EXIT_MQTT_ERROR = 3
EXIT_UNEXPECTED_ERROR = 99

# Service configuration
DATABASE_PATH = "data/alert_state.db"
CONFIG_PATH = "config/realtime_config.yaml"
LOG_PATH = "logs/realtime_service.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5
SHUTDOWN_TIMEOUT = 30  # seconds
MQTT_MAX_RETRIES = 5
MQTT_RETRY_DELAY = 5  # seconds

# Global state
service_running = False
start_time = None


class ServiceOrchestrator:
    """Orchestrates all service components with lifecycle management."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        """Initialize the service orchestrator.
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.pipeline = None
        self.scheduler = None
        self.subscriber = None
        self.db_connection = None
        
    def initialize_components(self) -> bool:
        """Initialize all service components.
        
        Returns:
            True if all components initialized successfully, False otherwise
        """
        try:
            # Test database connection first (fail fast)
            self.logger.info(f"Testing database connection: {DATABASE_PATH}")
            if not self._test_database_connection():
                self.logger.error(f"Database not accessible: {DATABASE_PATH}")
                return False
            self.logger.info("Database connection successful")
            
            # Initialize DetectorPipeline
            self.logger.info("Initializing DetectorPipeline...")
            try:
                from realtime.pipeline import DetectorPipeline
                self.pipeline = DetectorPipeline(
                    config=self.config.get('detector', {}),
                    db_path=DATABASE_PATH
                )
                self.logger.info("DetectorPipeline initialized")
            except ImportError as e:
                self.logger.warning(f"DetectorPipeline not available: {e}")
                self.logger.warning("Service will continue without pipeline component")
                self.pipeline = None
            except Exception as e:
                self.logger.error(f"Failed to initialize DetectorPipeline: {e}")
                return False
            
            # Initialize DetectorScheduler
            self.logger.info("Initializing DetectorScheduler...")
            try:
                from realtime.scheduler import DetectorScheduler
                self.scheduler = DetectorScheduler(
                    pipeline=self.pipeline,
                    config=self.config
                )
                self.logger.info("DetectorScheduler initialized")
            except ImportError as e:
                self.logger.warning(f"DetectorScheduler not available: {e}")
                self.logger.warning("Service will continue without scheduler component")
                self.scheduler = None
            except Exception as e:
                self.logger.error(f"Failed to initialize DetectorScheduler: {e}")
                return False
            
            # Initialize MQTTSubscriber
            self.logger.info("Initializing MQTTSubscriber...")
            try:
                from realtime.mqtt_subscriber import MQTTSubscriber
                # MQTTSubscriber expects config with 'mqtt' and 'database' sections
                subscriber_config = {
                    'mqtt': self.config.get('mqtt', {}),
                    'database': {'path': DATABASE_PATH}
                }
                self.subscriber = MQTTSubscriber(config=subscriber_config)
                self.logger.info("MQTTSubscriber initialized")
            except ImportError as e:
                self.logger.warning(f"MQTTSubscriber not available: {e}")
                self.logger.warning("Service will continue without MQTT component")
                self.subscriber = None
            except Exception as e:
                self.logger.error(f"Failed to initialize MQTTSubscriber: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def _test_database_connection(self) -> bool:
        """Test database accessibility.
        
        Returns:
            True if database is accessible, False otherwise
        """
        try:
            db_path = Path(DATABASE_PATH)
            
            # Check if parent directory exists
            if not db_path.parent.exists():
                self.logger.error(f"Database directory does not exist: {db_path.parent}")
                return False
            
            # Try to connect
            conn = sqlite3.connect(str(db_path), timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Database connection test failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error testing database: {e}")
            return False
    
    def start_services(self) -> bool:
        """Start all service components in proper order.
        
        Returns:
            True if all services started successfully, False otherwise
        """
        try:
            # Start DetectorScheduler first
            if self.scheduler:
                self.logger.info("Starting DetectorScheduler...")
                try:
                    self.scheduler.start()
                    self.logger.info("DetectorScheduler started successfully")
                except Exception as e:
                    self.logger.error(f"Failed to start DetectorScheduler: {e}")
                    return False
            
            # Connect and start MQTTSubscriber
            if self.subscriber:
                self.logger.info("Connecting to MQTT broker...")
                retry_count = 0
                while retry_count < MQTT_MAX_RETRIES:
                    try:
                        self.subscriber.connect()
                        self.logger.info("Connected to MQTT broker")
                        
                        self.logger.info("Starting MQTTSubscriber...")
                        self.subscriber.start()
                        self.logger.info("MQTTSubscriber started successfully")
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        self.logger.warning(
                            f"MQTT connection attempt {retry_count}/{MQTT_MAX_RETRIES} failed: {e}"
                        )
                        
                        if retry_count >= MQTT_MAX_RETRIES:
                            self.logger.error("Max MQTT connection retries reached")
                            return False
                        
                        self.logger.info(f"Retrying in {MQTT_RETRY_DELAY} seconds...")
                        time.sleep(MQTT_RETRY_DELAY)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Service startup failed: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def shutdown_services(self) -> None:
        """Shutdown all services gracefully with timeout handling."""
        self.logger.info("Shutdown initiated...")
        shutdown_start = time.time()
        
        # Stop MQTTSubscriber first
        if self.subscriber:
            try:
                self.logger.info("Stopping MQTTSubscriber...")
                self.subscriber.stop()
                self.logger.info("MQTTSubscriber stopped")
            except Exception as e:
                self.logger.error(f"Error stopping MQTTSubscriber: {e}")
        
        # Stop DetectorScheduler with timeout
        if self.scheduler:
            try:
                self.logger.info(f"Stopping DetectorScheduler (timeout: {SHUTDOWN_TIMEOUT}s)...")
                remaining_time = SHUTDOWN_TIMEOUT - (time.time() - shutdown_start)
                
                if remaining_time > 0:
                    self.scheduler.stop(timeout=int(remaining_time))
                    self.logger.info("DetectorScheduler stopped")
                else:
                    self.logger.warning("Shutdown timeout reached, forcing scheduler stop")
                    self.scheduler.stop(timeout=0)
                    
            except Exception as e:
                self.logger.error(f"Error stopping DetectorScheduler: {e}")
        
        # Close database connections explicitly
        if self.db_connection:
            try:
                self.logger.info("Closing database connections...")
                self.db_connection.close()
                self.logger.info("Database connections closed")
            except Exception as e:
                self.logger.error(f"Error closing database: {e}")
        
        # Calculate uptime
        if start_time:
            uptime = time.time() - start_time
            uptime_str = format_duration(uptime)
            self.logger.info(f"Shutdown complete. Service uptime: {uptime_str}")
        else:
            self.logger.info("Shutdown complete")


def setup_logging(config: dict) -> logging.Logger:
    """Set up logging with rotation and console output.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured logger instance
    """
    # Get log level from config (default: INFO)
    log_level_str = config.get('logging', {}).get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    # Create logger
    logger = logging.getLogger('realtime_service')
    logger.setLevel(log_level)
    logger.handlers.clear()  # Clear any existing handlers
    
    # Create log directory if it doesn't exist
    log_dir = Path(LOG_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Log format
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Rotating file handler (10MB max, 5 backups)
    file_handler = RotatingFileHandler(
        LOG_PATH,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    
    return logger


def load_configuration() -> Optional[dict]:
    """Load and validate configuration file.
    
    Returns:
        Configuration dictionary if successful, None otherwise
    """
    try:
        config_path = Path(CONFIG_PATH)
        
        if not config_path.exists():
            print(f"ERROR: Configuration file not found: {CONFIG_PATH}", file=sys.stderr)
            print("Creating default configuration structure...", file=sys.stderr)
            
            # Create default config
            default_config = {
                'service': {
                    'name': 'realtime_health_monitoring',
                    'version': __version__
                },
                'logging': {
                    'level': 'INFO'
                },
                'mqtt': {
                    'broker': 'localhost',
                    'port': 1883,
                    'topics': ['livestock/sensors/#'],
                    'client_id': 'artemis_health_service'
                },
                'detector': {
                    'check_interval': 60,
                    'alert_thresholds': {}
                },
                'database': {
                    'path': DATABASE_PATH
                }
            }
            
            # Create config directory if needed
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save default config
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False, sort_keys=False)
            
            print(f"Default configuration created at: {CONFIG_PATH}", file=sys.stderr)
            print("Please review and update the configuration as needed.", file=sys.stderr)
            
            return default_config
        
        # Load existing config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            print(f"ERROR: Configuration file is empty: {CONFIG_PATH}", file=sys.stderr)
            return None
        
        # Validate required sections
        required_sections = ['mqtt', 'detector']
        missing_sections = [s for s in required_sections if s not in config]
        
        if missing_sections:
            print(
                f"WARNING: Missing configuration sections: {', '.join(missing_sections)}",
                file=sys.stderr
            )
            print("Using default values for missing sections", file=sys.stderr)
            
            # Add default sections
            if 'mqtt' not in config:
                config['mqtt'] = {
                    'broker': 'localhost',
                    'port': 1883,
                    'topics': ['livestock/sensors/#']
                }
            if 'detector' not in config:
                config['detector'] = {
                    'check_interval': 60
                }
        
        return config
        
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML in configuration file: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {e}", file=sys.stderr)
        return None


def signal_handler(signum, frame):
    """Handle shutdown signals (SIGINT, SIGTERM).
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global service_running
    
    signal_name = signal.Signals(signum).name
    print(f"\n\nReceived {signal_name} signal", file=sys.stderr)
    service_running = False


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def main():
    """Main service entry point."""
    global service_running, start_time
    
    # Print startup banner
    print("=" * 70)
    print(f"  Realtime Livestock Health Monitoring Service v{__version__}")
    print("=" * 70)
    print()
    
    # Load configuration
    print(f"Loading configuration from: {CONFIG_PATH}")
    config = load_configuration()
    
    if not config:
        print("ERROR: Failed to load configuration", file=sys.stderr)
        return EXIT_CONFIG_ERROR
    
    print("Configuration loaded successfully")
    print()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("=" * 70)
    logger.info(f"Realtime Health Monitoring Service v{__version__} starting...")
    logger.info("=" * 70)
    logger.info(f"Configuration: {CONFIG_PATH}")
    logger.info(f"Database: {DATABASE_PATH}")
    logger.info(f"Log file: {LOG_PATH}")
    
    # Log configuration summary
    logger.info("Configuration summary:")
    if 'mqtt' in config:
        mqtt_cfg = config['mqtt']
        logger.info(f"  MQTT Broker: {mqtt_cfg.get('broker')}:{mqtt_cfg.get('port')}")
        logger.info(f"  MQTT Topics: {mqtt_cfg.get('topics')}")
    if 'detector' in config:
        detector_cfg = config['detector']
        logger.info(f"  Detector Check Interval: {detector_cfg.get('check_interval')}s")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers registered (SIGINT, SIGTERM)")
    
    # Initialize orchestrator
    orchestrator = ServiceOrchestrator(config, logger)
    
    try:
        # Initialize components
        logger.info("Initializing service components...")
        if not orchestrator.initialize_components():
            logger.error("Component initialization failed")
            return EXIT_DATABASE_ERROR
        
        # Start services
        logger.info("Starting services...")
        if not orchestrator.start_services():
            logger.error("Service startup failed")
            orchestrator.shutdown_services()
            return EXIT_MQTT_ERROR
        
        # Mark service as running
        service_running = True
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("Service is now running - Press Ctrl+C to stop")
        logger.info("=" * 70)
        print()
        print("✓ Service running successfully")
        print("  Press Ctrl+C to stop...")
        print()
        
        # Main service loop
        while service_running:
            time.sleep(1)
        
        # Graceful shutdown
        orchestrator.shutdown_services()
        return EXIT_SUCCESS
        
    except KeyboardInterrupt:
        # This shouldn't happen due to signal handler, but just in case
        logger.info("Keyboard interrupt received")
        orchestrator.shutdown_services()
        return EXIT_SUCCESS
        
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        logger.critical(traceback.format_exc())
        
        try:
            orchestrator.shutdown_services()
        except:
            pass
        
        return EXIT_UNEXPECTED_ERROR


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
