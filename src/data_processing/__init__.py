"""
Data Processing Module

This module provides data ingestion, validation, preprocessing, normalization,
and feature engineering utilities for the Artemis Health livestock monitoring system.
"""

from .validation import (
    DataValidator,
    ValidationReport,
    ValidationIssue,
    ValidationSeverity,
    validate_sensor_data
)

from .normalization import (
    normalize_temperature,
    standardize_acceleration,
    standardize_angular_velocity,
    normalize_sensor_data,
    inverse_normalize_temperature,
    inverse_standardize_acceleration,
    inverse_standardize_angular_velocity,
    TEMPERATURE_MIN,
    TEMPERATURE_MAX,
    ACCELERATION_EXPECTED_MIN,
    ACCELERATION_EXPECTED_MAX,
    ACCELERATION_EXPECTED_MEAN,
    ACCELERATION_EXPECTED_STD,
    GYROSCOPE_EXPECTED_MIN,
    GYROSCOPE_EXPECTED_MAX,
    GYROSCOPE_EXPECTED_MEAN,
    GYROSCOPE_EXPECTED_STD
)

from .feature_engineering import (
    calculate_motion_intensity,
    calculate_pitch_angle,
    calculate_roll_angle,
    calculate_activity_score,
    calculate_postural_stability,
    calculate_head_movement_intensity,
    extract_rhythmic_features,
    engineer_features,
    create_feature_vector
)

__all__ = [
    # Validation
    'DataValidator',
    'ValidationReport',
    'ValidationIssue',
    'ValidationSeverity',
    'validate_sensor_data',
    
    # Normalization
    'normalize_temperature',
    'standardize_acceleration',
    'standardize_angular_velocity',
    'normalize_sensor_data',
    'inverse_normalize_temperature',
    'inverse_standardize_acceleration',
    'inverse_standardize_angular_velocity',
    'TEMPERATURE_MIN',
    'TEMPERATURE_MAX',
    'ACCELERATION_EXPECTED_MIN',
    'ACCELERATION_EXPECTED_MAX',
    'ACCELERATION_EXPECTED_MEAN',
    'ACCELERATION_EXPECTED_STD',
    'GYROSCOPE_EXPECTED_MIN',
    'GYROSCOPE_EXPECTED_MAX',
    'GYROSCOPE_EXPECTED_MEAN',
    'GYROSCOPE_EXPECTED_STD',
    
    # Feature Engineering
    'calculate_motion_intensity',
    'calculate_pitch_angle',
    'calculate_roll_angle',
    'calculate_activity_score',
    'calculate_postural_stability',
    'calculate_head_movement_intensity',
    'extract_rhythmic_features',
    'engineer_features',
    'create_feature_vector'
]
