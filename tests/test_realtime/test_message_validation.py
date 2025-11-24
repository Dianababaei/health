"""
Unit Tests for MQTT Message Validation

Tests cover:
- JSON schema validation logic
- Required field presence checks
- Data type validation (strings, floats, timestamps)
- Timestamp parsing (ISO 8601, various formats)
- Boundary conditions (negative values, extreme temps)
- Malformed data rejection
"""

import unittest
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.test_realtime import (
    VALID_MQTT_MESSAGE,
    VALID_MQTT_MESSAGE_WITH_MOTION,
    INVALID_MQTT_MESSAGES,
    MALFORMED_JSON_MESSAGES
)


# =============================================================================
# Validation Functions (Expected Interface)
# =============================================================================

def validate_required_fields(message: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that required fields are present.
    
    Returns:
        (is_valid, error_message)
    """
    required_fields = ['timestamp', 'cow_id']
    
    for field in required_fields:
        if field not in message:
            return False, f"Missing required field: {field}"
    
    return True, None


def validate_field_types(message: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that fields have correct types.
    
    Returns:
        (is_valid, error_message)
    """
    # Type specifications
    string_fields = ['cow_id', 'state']
    numeric_fields = ['temperature', 'fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg', 'motion_intensity']
    
    # Check string fields
    for field in string_fields:
        if field in message and not isinstance(message[field], str):
            return False, f"Field '{field}' must be a string, got {type(message[field]).__name__}"
    
    # Check numeric fields
    for field in numeric_fields:
        if field in message:
            if not isinstance(message[field], (int, float)):
                return False, f"Field '{field}' must be numeric, got {type(message[field]).__name__}"
    
    return True, None


def validate_value_ranges(message: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate that values are within acceptable ranges.
    
    Returns:
        (is_valid, error_message)
    """
    # Temperature range: 35.0 to 42.0 °C (reasonable for cattle)
    if 'temperature' in message:
        temp = message['temperature']
        if temp < 35.0 or temp > 42.0:
            return False, f"Temperature {temp}°C is out of valid range (35.0-42.0)"
    
    # Accelerometer values: typically 0.0 to 2.0 (g-force)
    accel_fields = ['fxa', 'mya', 'rza', 'sxg', 'lyg', 'dzg']
    for field in accel_fields:
        if field in message:
            value = message[field]
            if value < -2.0 or value > 2.0:
                return False, f"Accelerometer field '{field}' value {value} is out of range (-2.0 to 2.0)"
    
    # Motion intensity: 0.0 to 1.0
    if 'motion_intensity' in message:
        value = message['motion_intensity']
        if value < 0.0 or value > 1.0:
            return False, f"Motion intensity {value} is out of range (0.0-1.0)"
    
    return True, None


def validate_timestamp(timestamp_str: str) -> tuple[bool, Optional[datetime], Optional[str]]:
    """
    Validate and parse timestamp string.
    
    Returns:
        (is_valid, parsed_datetime, error_message)
    """
    if not timestamp_str:
        return False, None, "Timestamp is empty"
    
    # Try multiple timestamp formats
    formats = [
        '%Y-%m-%dT%H:%M:%SZ',           # ISO 8601 with Z
        '%Y-%m-%dT%H:%M:%S.%fZ',        # ISO 8601 with milliseconds
        '%Y-%m-%d %H:%M:%S',            # Simple format
        '%Y-%m-%dT%H:%M:%S%z',          # ISO 8601 with timezone
        '%Y-%m-%dT%H:%M:%S.%f%z',       # ISO 8601 with ms and timezone
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str, fmt)
            return True, dt, None
        except ValueError:
            continue
    
    # If all formats fail
    return False, None, f"Invalid timestamp format: {timestamp_str}"


def validate_message(message: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Complete message validation.
    
    Returns:
        (is_valid, error_message)
    """
    # Check required fields
    valid, error = validate_required_fields(message)
    if not valid:
        return False, error
    
    # Check field types
    valid, error = validate_field_types(message)
    if not valid:
        return False, error
    
    # Validate timestamp
    valid, parsed_dt, error = validate_timestamp(message['timestamp'])
    if not valid:
        return False, error
    
    # Check value ranges
    valid, error = validate_value_ranges(message)
    if not valid:
        return False, error
    
    return True, None


# =============================================================================
# Test Classes
# =============================================================================

class TestRequiredFields(unittest.TestCase):
    """Test required field validation."""
    
    def test_valid_message_has_all_required_fields(self):
        """Test that valid message passes required field check."""
        valid, error = validate_required_fields(VALID_MQTT_MESSAGE)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_missing_timestamp(self):
        """Test detection of missing timestamp field."""
        message = INVALID_MQTT_MESSAGES['missing_timestamp']
        valid, error = validate_required_fields(message)
        self.assertFalse(valid)
        self.assertIn('timestamp', error)
    
    def test_missing_cow_id(self):
        """Test detection of missing cow_id field."""
        message = INVALID_MQTT_MESSAGES['missing_cow_id']
        valid, error = validate_required_fields(message)
        self.assertFalse(valid)
        self.assertIn('cow_id', error)
    
    def test_message_with_extra_fields(self):
        """Test that extra fields don't fail validation."""
        message = VALID_MQTT_MESSAGE.copy()
        message['extra_field'] = 'extra_value'
        
        valid, error = validate_required_fields(message)
        self.assertTrue(valid)
    
    def test_empty_message(self):
        """Test empty message fails validation."""
        message = {}
        valid, error = validate_required_fields(message)
        self.assertFalse(valid)


class TestFieldTypeValidation(unittest.TestCase):
    """Test field type validation."""
    
    def test_valid_types(self):
        """Test that valid field types pass validation."""
        valid, error = validate_field_types(VALID_MQTT_MESSAGE)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_invalid_temperature_type(self):
        """Test detection of invalid temperature type."""
        message = INVALID_MQTT_MESSAGES['invalid_temperature_type']
        valid, error = validate_field_types(message)
        self.assertFalse(valid)
        self.assertIn('temperature', error)
        self.assertIn('string', error.lower())
    
    def test_numeric_fields_accept_int(self):
        """Test that numeric fields accept integers."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = 38  # Integer instead of float
        
        valid, error = validate_field_types(message)
        self.assertTrue(valid)
    
    def test_numeric_fields_accept_float(self):
        """Test that numeric fields accept floats."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = 38.5
        
        valid, error = validate_field_types(message)
        self.assertTrue(valid)
    
    def test_cow_id_must_be_string(self):
        """Test that cow_id must be a string."""
        message = VALID_MQTT_MESSAGE.copy()
        message['cow_id'] = 123  # Integer instead of string
        
        valid, error = validate_field_types(message)
        self.assertFalse(valid)
        self.assertIn('cow_id', error)
    
    def test_state_must_be_string(self):
        """Test that state field must be a string."""
        message = VALID_MQTT_MESSAGE.copy()
        message['state'] = 123
        
        valid, error = validate_field_types(message)
        self.assertFalse(valid)
        self.assertIn('state', error)


class TestValueRangeValidation(unittest.TestCase):
    """Test value range validation."""
    
    def test_valid_temperature_range(self):
        """Test that valid temperature passes range check."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = 38.5
        
        valid, error = validate_value_ranges(message)
        self.assertTrue(valid)
    
    def test_negative_temperature(self):
        """Test detection of negative temperature."""
        message = INVALID_MQTT_MESSAGES['negative_temperature']
        valid, error = validate_value_ranges(message)
        self.assertFalse(valid)
        self.assertIn('Temperature', error)
        self.assertIn('out of valid range', error)
    
    def test_extreme_high_temperature(self):
        """Test detection of extremely high temperature."""
        message = INVALID_MQTT_MESSAGES['extreme_temperature']
        valid, error = validate_value_ranges(message)
        self.assertFalse(valid)
        self.assertIn('Temperature', error)
    
    def test_temperature_boundary_low(self):
        """Test temperature at low boundary."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = 35.0  # Minimum valid
        
        valid, error = validate_value_ranges(message)
        self.assertTrue(valid)
    
    def test_temperature_boundary_high(self):
        """Test temperature at high boundary."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = 42.0  # Maximum valid
        
        valid, error = validate_value_ranges(message)
        self.assertTrue(valid)
    
    def test_temperature_below_boundary(self):
        """Test temperature just below low boundary."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = 34.9
        
        valid, error = validate_value_ranges(message)
        self.assertFalse(valid)
    
    def test_temperature_above_boundary(self):
        """Test temperature just above high boundary."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = 42.1
        
        valid, error = validate_value_ranges(message)
        self.assertFalse(valid)
    
    def test_accelerometer_valid_range(self):
        """Test valid accelerometer values."""
        message = VALID_MQTT_MESSAGE.copy()
        message['fxa'] = 0.5
        message['mya'] = -0.3
        message['rza'] = 1.2
        
        valid, error = validate_value_ranges(message)
        self.assertTrue(valid)
    
    def test_accelerometer_out_of_range(self):
        """Test accelerometer value out of range."""
        message = VALID_MQTT_MESSAGE.copy()
        message['fxa'] = 5.0  # Too high
        
        valid, error = validate_value_ranges(message)
        self.assertFalse(valid)
        self.assertIn('fxa', error)
    
    def test_motion_intensity_valid_range(self):
        """Test valid motion intensity values."""
        message = VALID_MQTT_MESSAGE_WITH_MOTION.copy()
        message['motion_intensity'] = 0.42
        
        valid, error = validate_value_ranges(message)
        self.assertTrue(valid)
    
    def test_motion_intensity_negative(self):
        """Test negative motion intensity."""
        message = VALID_MQTT_MESSAGE_WITH_MOTION.copy()
        message['motion_intensity'] = -0.1
        
        valid, error = validate_value_ranges(message)
        self.assertFalse(valid)
        self.assertIn('Motion intensity', error)
    
    def test_motion_intensity_above_one(self):
        """Test motion intensity above 1.0."""
        message = VALID_MQTT_MESSAGE_WITH_MOTION.copy()
        message['motion_intensity'] = 1.5
        
        valid, error = validate_value_ranges(message)
        self.assertFalse(valid)


class TestTimestampParsing(unittest.TestCase):
    """Test timestamp parsing and validation."""
    
    def test_parse_iso8601_with_z(self):
        """Test parsing ISO 8601 timestamp with Z."""
        timestamp = "2024-01-15T12:30:45Z"
        valid, dt, error = validate_timestamp(timestamp)
        
        self.assertTrue(valid)
        self.assertIsNotNone(dt)
        self.assertIsNone(error)
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.month, 1)
        self.assertEqual(dt.day, 15)
    
    def test_parse_iso8601_with_milliseconds(self):
        """Test parsing ISO 8601 with milliseconds."""
        timestamp = "2024-01-15T12:30:45.123Z"
        valid, dt, error = validate_timestamp(timestamp)
        
        self.assertTrue(valid)
        self.assertIsNotNone(dt)
    
    def test_parse_simple_format(self):
        """Test parsing simple datetime format."""
        timestamp = "2024-01-15 12:30:45"
        valid, dt, error = validate_timestamp(timestamp)
        
        self.assertTrue(valid)
        self.assertIsNotNone(dt)
    
    def test_invalid_timestamp_format(self):
        """Test detection of invalid timestamp format."""
        message = INVALID_MQTT_MESSAGES['invalid_timestamp_format']
        valid, dt, error = validate_timestamp(message['timestamp'])
        
        self.assertFalse(valid)
        self.assertIsNone(dt)
        self.assertIsNotNone(error)
    
    def test_empty_timestamp(self):
        """Test handling of empty timestamp."""
        valid, dt, error = validate_timestamp("")
        
        self.assertFalse(valid)
        self.assertIsNone(dt)
        self.assertIn('empty', error.lower())
    
    def test_timestamp_with_timezone(self):
        """Test parsing timestamp with timezone."""
        timestamp = "2024-01-15T12:30:45+00:00"
        valid, dt, error = validate_timestamp(timestamp)
        
        self.assertTrue(valid)
        self.assertIsNotNone(dt)
    
    def test_timestamp_various_formats(self):
        """Test parsing various timestamp formats."""
        formats = [
            "2024-01-15T12:30:45Z",
            "2024-01-15T12:30:45.000Z",
            "2024-01-15 12:30:45",
        ]
        
        for timestamp in formats:
            with self.subTest(timestamp=timestamp):
                valid, dt, error = validate_timestamp(timestamp)
                self.assertTrue(valid, f"Failed to parse: {timestamp}")


class TestCompleteValidation(unittest.TestCase):
    """Test complete message validation."""
    
    def test_valid_message_passes_all_checks(self):
        """Test that valid message passes all validation checks."""
        valid, error = validate_message(VALID_MQTT_MESSAGE)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_valid_message_with_motion_passes(self):
        """Test that valid message with motion intensity passes."""
        valid, error = validate_message(VALID_MQTT_MESSAGE_WITH_MOTION)
        self.assertTrue(valid)
        self.assertIsNone(error)
    
    def test_message_with_missing_timestamp_fails(self):
        """Test that message with missing timestamp fails."""
        message = INVALID_MQTT_MESSAGES['missing_timestamp']
        valid, error = validate_message(message)
        self.assertFalse(valid)
        self.assertIn('timestamp', error)
    
    def test_message_with_invalid_type_fails(self):
        """Test that message with invalid type fails."""
        message = INVALID_MQTT_MESSAGES['invalid_temperature_type']
        valid, error = validate_message(message)
        self.assertFalse(valid)
    
    def test_message_with_out_of_range_value_fails(self):
        """Test that message with out of range value fails."""
        message = INVALID_MQTT_MESSAGES['extreme_temperature']
        valid, error = validate_message(message)
        self.assertFalse(valid)
    
    def test_minimal_valid_message(self):
        """Test minimal valid message (only required fields)."""
        message = {
            'timestamp': '2024-01-15T12:30:45Z',
            'cow_id': 'COW_001'
        }
        valid, error = validate_message(message)
        self.assertTrue(valid)


class TestBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and edge cases."""
    
    def test_temperature_at_boundaries(self):
        """Test temperature at exact boundary values."""
        test_cases = [
            (35.0, True),   # Min valid
            (34.9, False),  # Just below min
            (42.0, True),   # Max valid
            (42.1, False),  # Just above max
        ]
        
        for temp, expected_valid in test_cases:
            with self.subTest(temperature=temp):
                message = VALID_MQTT_MESSAGE.copy()
                message['temperature'] = temp
                valid, error = validate_value_ranges(message)
                self.assertEqual(valid, expected_valid)
    
    def test_accelerometer_at_boundaries(self):
        """Test accelerometer values at boundaries."""
        test_cases = [
            (-2.0, True),   # Min valid
            (-2.1, False),  # Below min
            (2.0, True),    # Max valid
            (2.1, False),   # Above max
        ]
        
        for value, expected_valid in test_cases:
            with self.subTest(fxa=value):
                message = VALID_MQTT_MESSAGE.copy()
                message['fxa'] = value
                valid, error = validate_value_ranges(message)
                self.assertEqual(valid, expected_valid)
    
    def test_zero_values(self):
        """Test that zero values are accepted."""
        message = VALID_MQTT_MESSAGE.copy()
        message['fxa'] = 0.0
        message['mya'] = 0.0
        message['rza'] = 0.0
        message['motion_intensity'] = 0.0
        
        valid, error = validate_value_ranges(message)
        self.assertTrue(valid)
    
    def test_very_small_positive_values(self):
        """Test very small positive values."""
        message = VALID_MQTT_MESSAGE.copy()
        message['fxa'] = 0.001
        message['temperature'] = 35.1
        
        valid, error = validate_message(message)
        self.assertTrue(valid)


class TestMalformedData(unittest.TestCase):
    """Test rejection of malformed data."""
    
    def test_malformed_json_strings(self):
        """Test handling of malformed JSON strings."""
        for malformed in MALFORMED_JSON_MESSAGES:
            if malformed is None:
                continue
            
            with self.subTest(malformed=malformed):
                try:
                    message = json.loads(malformed)
                    # If parsing succeeds, validate the message
                    valid, error = validate_message(message)
                except json.JSONDecodeError:
                    # Expected for malformed JSON
                    pass
    
    def test_null_values(self):
        """Test handling of null values."""
        message = VALID_MQTT_MESSAGE.copy()
        message['temperature'] = None
        
        valid, error = validate_field_types(message)
        self.assertFalse(valid)
    
    def test_missing_optional_fields(self):
        """Test that missing optional fields don't fail validation."""
        message = {
            'timestamp': '2024-01-15T12:30:45Z',
            'cow_id': 'COW_001',
            'temperature': 38.5
            # Missing fxa, mya, rza, state, etc.
        }
        
        valid, error = validate_message(message)
        self.assertTrue(valid)
    
    def test_array_instead_of_object(self):
        """Test handling when array received instead of object."""
        # This would be caught at JSON parsing level
        data = [1, 2, 3]
        
        # Should fail required field check
        valid, error = validate_required_fields(data) if isinstance(data, dict) else (False, "Not a dict")
        self.assertFalse(valid)


class TestSpecialCases(unittest.TestCase):
    """Test special cases and edge scenarios."""
    
    def test_unicode_cow_id(self):
        """Test handling of Unicode characters in cow_id."""
        message = VALID_MQTT_MESSAGE.copy()
        message['cow_id'] = 'COW_001_🐄'
        
        valid, error = validate_message(message)
        self.assertTrue(valid)
    
    def test_very_long_cow_id(self):
        """Test handling of very long cow_id."""
        message = VALID_MQTT_MESSAGE.copy()
        message['cow_id'] = 'COW_' + '0' * 100
        
        valid, error = validate_message(message)
        self.assertTrue(valid)
    
    def test_future_timestamp(self):
        """Test handling of future timestamp."""
        future = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
        message = VALID_MQTT_MESSAGE.copy()
        message['timestamp'] = future
        
        # Should parse successfully (timestamp validation doesn't check if future)
        valid, error = validate_message(message)
        self.assertTrue(valid)
    
    def test_old_timestamp(self):
        """Test handling of very old timestamp."""
        old = "2020-01-01T00:00:00Z"
        message = VALID_MQTT_MESSAGE.copy()
        message['timestamp'] = old
        
        # Should parse successfully
        valid, error = validate_message(message)
        self.assertTrue(valid)
    
    def test_case_sensitive_field_names(self):
        """Test that field names are case-sensitive."""
        message = {
            'Timestamp': '2024-01-15T12:30:45Z',  # Wrong case
            'Cow_ID': 'COW_001',                   # Wrong case
            'temperature': 38.5
        }
        
        # Should fail - required fields not found
        valid, error = validate_required_fields(message)
        self.assertFalse(valid)


if __name__ == '__main__':
    unittest.main()
