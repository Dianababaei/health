"""
Configuration loader for Artemis Health monitoring system.

This module provides typed access to configuration parameters with validation.
Supports environment-specific overrides and ensures all critical parameters
are present and valid.

Usage:
    from src.utils.config_loader import load_config
    
    config = load_config()
    fever_threshold = config.alerts.temperature.fever_threshold
    sensor_channels = config.sensor.channels
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Sensor Configuration Models
# =============================================================================

class SensorChannel(BaseModel):
    """Configuration for a single sensor channel."""
    name: str
    unit: str
    description: str
    min_valid: float
    max_valid: float
    
    @field_validator('max_valid')
    @classmethod
    def max_greater_than_min(cls, v, info):
        """Ensure max_valid is greater than min_valid."""
        if 'min_valid' in info.data and v <= info.data['min_valid']:
            raise ValueError(f"max_valid must be greater than min_valid")
        return v


class SensorConfig(BaseModel):
    """Sensor specifications and data collection parameters."""
    channels: List[SensorChannel]
    sampling_rate: int = Field(gt=0, description="Sampling rate in minutes")
    malfunction_threshold: float = Field(ge=0, le=1, description="Fraction threshold for malfunction")
    
    @field_validator('channels')
    @classmethod
    def validate_channel_count(cls, v):
        """Ensure we have exactly 7 sensor channels."""
        if len(v) != 7:
            raise ValueError(f"Expected 7 sensor channels, got {len(v)}")
        
        expected_channels = {'Temperature', 'Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg'}
        actual_channels = {channel.name for channel in v}
        
        if actual_channels != expected_channels:
            missing = expected_channels - actual_channels
            extra = actual_channels - expected_channels
            msg = []
            if missing:
                msg.append(f"Missing channels: {missing}")
            if extra:
                msg.append(f"Extra channels: {extra}")
            raise ValueError("; ".join(msg))
        
        return v


# =============================================================================
# Feature Extraction Configuration Models
# =============================================================================

class WindowSizes(BaseModel):
    """Time window sizes for feature extraction."""
    short: int = Field(gt=0, description="Short window in minutes")
    medium: int = Field(gt=0, description="Medium window in minutes")
    long: int = Field(gt=0, description="Long window in minutes")
    
    @model_validator(mode='after')
    def validate_window_ordering(self):
        """Ensure windows are ordered: short <= medium <= long."""
        if not (self.short <= self.medium <= self.long):
            raise ValueError(
                f"Window sizes must be ordered: short <= medium <= long, "
                f"got short={self.short}, medium={self.medium}, long={self.long}"
            )
        return self


class FeatureExtractionConfig(BaseModel):
    """Feature extraction parameters."""
    window_sizes: WindowSizes
    statistical_features: List[str]
    derived_features: List[str]


# =============================================================================
# Model Configuration Models
# =============================================================================

class ModelParams(BaseModel):
    """Parameters for a single ML model."""
    type: str
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    min_samples_split: Optional[int] = None
    min_samples_leaf: Optional[int] = None
    learning_rate: Optional[float] = None
    contamination: Optional[float] = None
    random_state: Optional[int] = None


class ModelConfigs(BaseModel):
    """Configuration for all ML models."""
    behavior_classifier: ModelParams
    health_classifier: ModelParams
    anomaly_detector: ModelParams


class TrainingConfig(BaseModel):
    """Training parameters."""
    test_size: float = Field(gt=0, lt=1)
    validation_size: float = Field(gt=0, lt=1)
    cross_validation_folds: int = Field(gt=1)
    random_state: int


class ModelConfig(BaseModel):
    """Machine learning model configuration."""
    models: ModelConfigs
    training: TrainingConfig


# =============================================================================
# Alert Configuration Models
# =============================================================================

class TemperatureAlerts(BaseModel):
    """Temperature-based alert thresholds."""
    fever_threshold: float = Field(gt=0, description="Temperature threshold for fever in °C")
    heat_stress_high: float = Field(gt=0, description="Critical heat stress temperature in °C")
    hypothermia_threshold: float = Field(gt=0, description="Low temperature threshold in °C")
    
    @model_validator(mode='after')
    def validate_temperature_thresholds(self):
        """Ensure temperature thresholds are logically ordered."""
        if not (self.hypothermia_threshold < self.fever_threshold < self.heat_stress_high):
            raise ValueError(
                "Temperature thresholds must be ordered: "
                "hypothermia < fever < heat_stress_high"
            )
        return self


class ActivityAlerts(BaseModel):
    """Activity-based alert thresholds."""
    inactivity_duration: int = Field(gt=0, description="Inactivity duration in minutes")
    high_activity_threshold: float = Field(gt=0, description="High activity threshold in m/s²")
    low_activity_threshold: float = Field(ge=0, description="Low activity threshold in m/s²")


class FeverAlert(BaseModel):
    """Combined fever detection parameters."""
    temp_threshold: float = Field(gt=0)
    activity_threshold: float = Field(ge=0)
    duration_minutes: int = Field(gt=0)


class HeatStressAlert(BaseModel):
    """Heat stress detection parameters."""
    temp_threshold: float = Field(gt=0)
    activity_threshold: float = Field(gt=0)
    duration_minutes: int = Field(gt=0)


class EstrusAlert(BaseModel):
    """Estrus detection parameters."""
    temp_increase: float = Field(gt=0)
    temp_increase_max: float = Field(gt=0)
    activity_multiplier: float = Field(gt=1)
    duration_hours: int = Field(gt=0)
    
    @model_validator(mode='after')
    def validate_temp_increase(self):
        """Ensure max temp increase is greater than min."""
        if self.temp_increase_max <= self.temp_increase:
            raise ValueError(
                "temp_increase_max must be greater than temp_increase"
            )
        return self


class CombinedAlerts(BaseModel):
    """Combined condition alert parameters."""
    fever: FeverAlert
    heat_stress: HeatStressAlert
    estrus: EstrusAlert


class ConfidenceThresholds(BaseModel):
    """ML prediction confidence thresholds."""
    min_prediction_confidence: float = Field(ge=0, le=1)
    alert_confidence: float = Field(ge=0, le=1)
    critical_alert_confidence: float = Field(ge=0, le=1)
    
    @model_validator(mode='after')
    def validate_confidence_ordering(self):
        """Ensure confidence thresholds are logically ordered."""
        if not (self.min_prediction_confidence <= self.alert_confidence <= self.critical_alert_confidence):
            raise ValueError(
                "Confidence thresholds must be ordered: "
                "min_prediction <= alert <= critical_alert"
            )
        return self


class AlertsConfig(BaseModel):
    """Alert threshold configuration."""
    temperature: TemperatureAlerts
    activity: ActivityAlerts
    combined: CombinedAlerts
    confidence: ConfidenceThresholds


# =============================================================================
# Paths Configuration Models
# =============================================================================

class DataPaths(BaseModel):
    """Data directory paths."""
    raw: str
    processed: str
    features: str
    interim: str


class ModelPaths(BaseModel):
    """Model directory paths."""
    trained: str
    checkpoints: str
    exports: str


class OutputPaths(BaseModel):
    """Output directory paths."""
    reports: str
    alerts: str
    visualizations: str
    predictions: str


class LogPaths(BaseModel):
    """Log file paths."""
    application: str
    errors: str
    alerts: str
    data_quality: str


class PathsConfig(BaseModel):
    """File and directory paths configuration."""
    data: DataPaths
    models: ModelPaths
    outputs: OutputPaths
    logs: LogPaths


# =============================================================================
# Logging Configuration Models
# =============================================================================

class RotationConfig(BaseModel):
    """Log file rotation settings."""
    max_bytes: int = Field(gt=0)
    backup_count: int = Field(ge=0)


class LoggerConfig(BaseModel):
    """Individual logger configuration."""
    level: str
    handlers: List[str]


class LoggersConfig(BaseModel):
    """Configuration for all loggers."""
    root: LoggerConfig
    data_processing: LoggerConfig
    model_training: LoggerConfig
    alerts: LoggerConfig
    errors: LoggerConfig


class ConsoleConfig(BaseModel):
    """Console logging configuration."""
    enabled: bool
    level: str
    colorize: bool


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str
    format: str
    date_format: str
    rotation: RotationConfig
    loggers: LoggersConfig
    console: ConsoleConfig


# =============================================================================
# Environment Configuration Models
# =============================================================================

class DevelopmentConfig(BaseModel):
    """Development environment settings."""
    debug: bool
    verbose_logging: bool
    save_intermediate_results: bool


class ProductionConfig(BaseModel):
    """Production environment settings."""
    debug: bool
    verbose_logging: bool
    save_intermediate_results: bool
    performance_monitoring: bool


class TestingConfig(BaseModel):
    """Testing environment settings."""
    debug: bool
    use_sample_data: bool
    mock_alerts: bool


class EnvironmentConfig(BaseModel):
    """Environment-specific configuration."""
    name: Literal["development", "testing", "production"]
    development: DevelopmentConfig
    production: ProductionConfig
    testing: TestingConfig


# =============================================================================
# Health Score Configuration Models
# =============================================================================

class HealthScoreWeights(BaseModel):
    """Weight factors for health score components."""
    temperature_stability: float = Field(ge=0, le=1)
    activity_level: float = Field(ge=0, le=1)
    behavioral_patterns: float = Field(ge=0, le=1)
    trend_analysis: float = Field(ge=0, le=1)
    
    @model_validator(mode='after')
    def validate_weights_sum(self):
        """Ensure weights sum to 1.0."""
        total = (
            self.temperature_stability +
            self.activity_level +
            self.behavioral_patterns +
            self.trend_analysis
        )
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Health score weights must sum to 1.0, got {total}")
        return self


class HealthScoreThresholds(BaseModel):
    """Score thresholds for health status levels."""
    excellent: int = Field(ge=0, le=100)
    good: int = Field(ge=0, le=100)
    fair: int = Field(ge=0, le=100)
    poor: int = Field(ge=0, le=100)
    critical: int = Field(ge=0, le=100)
    
    @model_validator(mode='after')
    def validate_threshold_ordering(self):
        """Ensure thresholds are logically ordered."""
        if not (self.critical <= self.poor <= self.fair <= self.good <= self.excellent):
            raise ValueError(
                "Health score thresholds must be ordered: "
                "critical <= poor <= fair <= good <= excellent"
            )
        return self


class BaselineConfig(BaseModel):
    """Baseline calculation parameters."""
    lookback_days: int = Field(gt=0)
    min_data_points: int = Field(gt=0)
    update_frequency_hours: int = Field(gt=0)


class HealthScoreConfig(BaseModel):
    """Health scoring system configuration."""
    weights: HealthScoreWeights
    thresholds: HealthScoreThresholds
    baseline: BaselineConfig


# =============================================================================
# Data Quality Configuration Models
# =============================================================================

class MissingDataConfig(BaseModel):
    """Missing data handling parameters."""
    max_consecutive_missing: int = Field(gt=0)
    max_missing_percentage: float = Field(ge=0, le=1)


class OutlierConfig(BaseModel):
    """Outlier detection parameters."""
    method: Literal["iqr", "zscore", "isolation_forest"]
    iqr_multiplier: float = Field(gt=0)
    zscore_threshold: float = Field(gt=0)


class MalfunctionConfig(BaseModel):
    """Sensor malfunction detection parameters."""
    flat_line_duration: int = Field(gt=0)
    spike_threshold: float = Field(gt=0)
    check_frequency_minutes: int = Field(gt=0)


class DataQualityConfig(BaseModel):
    """Data quality checking configuration."""
    missing_data: MissingDataConfig
    outliers: OutlierConfig
    malfunction: MalfunctionConfig


# =============================================================================
# Main Configuration Model
# =============================================================================

class Config(BaseModel):
    """
    Main configuration model for the Artemis Health monitoring system.
    
    This model provides typed access to all configuration parameters with
    built-in validation to ensure consistency and correctness.
    """
    sensor: SensorConfig
    feature_extraction: FeatureExtractionConfig
    model: ModelConfig
    alerts: AlertsConfig
    paths: PathsConfig
    logging: LoggingConfig
    environment: EnvironmentConfig
    health_score: HealthScoreConfig
    data_quality: DataQualityConfig
    
    def get_channel_by_name(self, name: str) -> Optional[SensorChannel]:
        """
        Get sensor channel configuration by name.
        
        Args:
            name: Channel name (e.g., 'Temperature', 'Fxa')
            
        Returns:
            SensorChannel if found, None otherwise
        """
        for channel in self.sensor.channels:
            if channel.name == name:
                return channel
        return None
    
    def get_channel_range(self, name: str) -> Optional[tuple[float, float]]:
        """
        Get valid range for a sensor channel.
        
        Args:
            name: Channel name
            
        Returns:
            Tuple of (min_valid, max_valid) if found, None otherwise
        """
        channel = self.get_channel_by_name(name)
        if channel:
            return (channel.min_valid, channel.max_valid)
        return None
    
    def is_fever_detected(self, temperature: float, activity: float) -> bool:
        """
        Check if readings indicate fever based on configured thresholds.
        
        Args:
            temperature: Current temperature in °C
            activity: Current activity level in m/s²
            
        Returns:
            True if fever conditions are met
        """
        fever_params = self.alerts.combined.fever
        return (
            temperature >= fever_params.temp_threshold and
            activity <= fever_params.activity_threshold
        )
    
    def is_heat_stress(self, temperature: float, activity: float) -> bool:
        """
        Check if readings indicate heat stress based on configured thresholds.
        
        Args:
            temperature: Current temperature in °C
            activity: Current activity level in m/s²
            
        Returns:
            True if heat stress conditions are met
        """
        stress_params = self.alerts.combined.heat_stress
        return (
            temperature >= stress_params.temp_threshold and
            activity >= stress_params.activity_threshold
        )


# =============================================================================
# Configuration Loading Functions
# =============================================================================

# Global configuration instance
_config_instance: Optional[Config] = None


def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> Config:
    """
    Load configuration from YAML file with validation.
    
    Args:
        config_path: Path to config.yaml file. If None, uses default path
                    (config/config.yaml from project root)
        force_reload: If True, reload config even if already loaded
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
        yaml.YAMLError: If YAML parsing fails
    """
    global _config_instance
    
    # Return cached instance if available and not forcing reload
    if _config_instance is not None and not force_reload:
        return _config_instance
    
    # Determine config file path
    if config_path is None:
        # Default to config/config.yaml relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    # Check if config file exists
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please ensure config/config.yaml exists in the project root."
        )
    
    # Load YAML file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")
    
    # Validate and create Config object
    try:
        _config_instance = Config(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    return _config_instance


def get_config() -> Config:
    """
    Get the current configuration instance.
    
    Returns:
        Config object
        
    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if _config_instance is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() first."
        )
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """
    Force reload configuration from file.
    
    Args:
        config_path: Path to config file (optional)
        
    Returns:
        Reloaded Config object
    """
    return load_config(config_path=config_path, force_reload=True)


# =============================================================================
# Environment Variable Support
# =============================================================================

def load_config_with_env_override(
    config_path: Optional[str] = None,
    env_prefix: str = "ARTEMIS_"
) -> Config:
    """
    Load configuration with environment variable overrides.
    
    Environment variables should be prefixed (default: ARTEMIS_) and use
    double underscores to indicate nesting. For example:
        ARTEMIS_ALERTS__TEMPERATURE__FEVER_THRESHOLD=40.0
    
    Args:
        config_path: Path to config file (optional)
        env_prefix: Prefix for environment variables
        
    Returns:
        Config object with environment overrides applied
    """
    # Load base config
    config = load_config(config_path)
    
    # Apply environment variable overrides
    config_dict = config.model_dump()
    
    for env_key, env_value in os.environ.items():
        if env_key.startswith(env_prefix):
            # Remove prefix and split by double underscore
            key_path = env_key[len(env_prefix):].lower().split('__')
            
            # Navigate to the nested dict location
            current = config_dict
            for key in key_path[:-1]:
                if key in current:
                    current = current[key]
                else:
                    break
            else:
                # Set the value (try to convert to appropriate type)
                final_key = key_path[-1]
                if final_key in current:
                    # Try to maintain the type of the original value
                    original_type = type(current[final_key])
                    try:
                        if original_type == bool:
                            current[final_key] = env_value.lower() in ('true', '1', 'yes')
                        elif original_type == int:
                            current[final_key] = int(env_value)
                        elif original_type == float:
                            current[final_key] = float(env_value)
                        else:
                            current[final_key] = env_value
                    except (ValueError, TypeError):
                        current[final_key] = env_value
    
    # Re-validate with overrides
    return Config(**config_dict)


if __name__ == "__main__":
    # Test configuration loading
    try:
        config = load_config()
        print("✓ Configuration loaded successfully!")
        print(f"  - Environment: {config.environment.name}")
        print(f"  - Fever threshold: {config.alerts.temperature.fever_threshold}°C")
        print(f"  - Number of sensor channels: {len(config.sensor.channels)}")
        print(f"  - Feature window sizes: {config.feature_extraction.window_sizes.short}, "
              f"{config.feature_extraction.window_sizes.medium}, "
              f"{config.feature_extraction.window_sizes.long} minutes")
    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
