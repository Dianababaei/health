"""
Behavior Pattern Definitions for Livestock Sensor Data

This module defines realistic parameter ranges for 6 distinct behaviors based on
livestock physiology research and neck-mounted sensor characteristics.

Sensor Configuration:
- Location: Neck-mounted collar
- Sampling Rate: 1 sample per minute
- Sensors: 3-axis accelerometer, 3-axis gyroscope, temperature sensor

Channel Descriptions:
- Fxa: Forward-backward acceleration (X-axis) [m/s²]
- Mya: Lateral acceleration (Y-axis) [m/s²]
- Rza: Vertical acceleration (Z-axis) [m/s²] - KEY for posture detection
- Sxg: Roll angular velocity (X-axis) [°/s]
- Lyg: Pitch angular velocity (Y-axis) [°/s] - KEY for head up/down movements
- Dzg: Yaw angular velocity (Z-axis) [°/s] - KEY for head rotation
- temp: Body temperature [°C]

Reference:
- Normal cattle body temperature: 38.3-39.1°C (101-102.5°F)
- Fever threshold: >39.5°C (103°F)
- Gravity constant: 9.8 m/s²
"""

import numpy as np

# Physical constants
GRAVITY = 9.8  # m/s²
NORMAL_TEMP_MEAN = 38.6  # °C - Normal cattle body temperature
FEVER_THRESHOLD = 39.5  # °C

# Frequency definitions for rhythmic behaviors
WALKING_FREQUENCY_HZ = 1.5  # 1-2 Hz typical walking gait
RUMINATING_FREQUENCY_HZ = 1.25  # 60-90 cycles/min = 1-1.5 Hz chewing pattern

BEHAVIOR_PATTERNS = {
    'lying': {
        'description': 'Animal lying down on side, minimal movement, horizontal body orientation',
        'Fxa': {
            'mean': 0.0,
            'std': 0.8,
            'min': -2.0,
            'max': 2.0,
            'unit': 'm/s²',
            'notes': 'Minimal forward-backward movement while lying'
        },
        'Mya': {
            'mean': 0.0,
            'std': 0.7,
            'min': -2.0,
            'max': 2.0,
            'unit': 'm/s²',
            'notes': 'Minimal lateral movement while lying'
        },
        'Rza': {
            'mean': 0.5,
            'std': 1.2,
            'min': -2.0,
            'max': 3.0,
            'unit': 'm/s²',
            'notes': 'Near-horizontal orientation, neck resting position (not vertical)'
        },
        'Sxg': {
            'mean': 0.0,
            'std': 25.0,
            'min': -80.0,
            'max': 80.0,
            'unit': '°/s',
            'notes': 'Minimal rolling motion while lying'
        },
        'Lyg': {
            'mean': 0.0,
            'std': 30.0,
            'min': -100.0,
            'max': 100.0,
            'unit': '°/s',
            'notes': 'Occasional head adjustments while lying'
        },
        'Dzg': {
            'mean': 0.0,
            'std': 35.0,
            'min': -120.0,
            'max': 120.0,
            'unit': '°/s',
            'notes': 'Small head rotations for awareness while lying'
        },
        'temp': {
            'mean': 38.5,
            'std': 0.15,
            'min': 38.2,
            'max': 38.9,
            'unit': '°C',
            'notes': 'Baseline temperature during rest, slightly lower than standing'
        },
        'frequency_components': {
            'dominant_frequency_hz': None,
            'rhythmic': False,
            'notes': 'No rhythmic patterns, random small movements'
        }
    },
    
    'standing': {
        'description': 'Animal standing still or with minimal movement, vertical neck orientation',
        'Fxa': {
            'mean': 0.0,
            'std': 1.2,
            'min': -3.5,
            'max': 3.5,
            'unit': 'm/s²',
            'notes': 'Small weight shifts and postural adjustments'
        },
        'Mya': {
            'mean': 0.0,
            'std': 1.1,
            'min': -3.5,
            'max': 3.5,
            'unit': 'm/s²',
            'notes': 'Small lateral movements while standing'
        },
        'Rza': {
            'mean': 9.0,
            'std': 1.5,
            'min': 6.0,
            'max': 11.5,
            'unit': 'm/s²',
            'notes': 'Near-vertical orientation (neck upright), close to gravity (9.8 m/s²)'
        },
        'Sxg': {
            'mean': 0.0,
            'std': 40.0,
            'min': -120.0,
            'max': 120.0,
            'unit': '°/s',
            'notes': 'Moderate head movements for scanning environment'
        },
        'Lyg': {
            'mean': 0.0,
            'std': 50.0,
            'min': -150.0,
            'max': 150.0,
            'unit': '°/s',
            'notes': 'Head up/down movements while alert standing'
        },
        'Dzg': {
            'mean': 0.0,
            'std': 55.0,
            'min': -180.0,
            'max': 180.0,
            'unit': '°/s',
            'notes': 'Head rotations for environmental awareness'
        },
        'temp': {
            'mean': 38.6,
            'std': 0.18,
            'min': 38.3,
            'max': 39.1,
            'unit': '°C',
            'notes': 'Normal baseline temperature'
        },
        'frequency_components': {
            'dominant_frequency_hz': None,
            'rhythmic': False,
            'notes': 'Random movements without rhythmic pattern'
        }
    },
    
    'walking': {
        'description': 'Animal walking with rhythmic gait pattern, moderate activity level',
        'Fxa': {
            'mean': 0.5,
            'std': 3.5,
            'min': -8.0,
            'max': 9.0,
            'unit': 'm/s²',
            'notes': 'Forward motion with rhythmic oscillations from gait'
        },
        'Mya': {
            'mean': 0.0,
            'std': 3.2,
            'min': -8.0,
            'max': 8.0,
            'unit': 'm/s²',
            'notes': 'Side-to-side sway from walking gait'
        },
        'Rza': {
            'mean': 8.5,
            'std': 2.8,
            'min': 3.0,
            'max': 13.0,
            'unit': 'm/s²',
            'notes': 'Vertical oscillations from gait, generally upright posture'
        },
        'Sxg': {
            'mean': 0.0,
            'std': 120.0,
            'min': -350.0,
            'max': 350.0,
            'unit': '°/s',
            'notes': 'Rhythmic rolling motion from walking gait'
        },
        'Lyg': {
            'mean': 0.0,
            'std': 140.0,
            'min': -400.0,
            'max': 400.0,
            'unit': '°/s',
            'notes': 'Rhythmic pitch changes from head movement during walking'
        },
        'Dzg': {
            'mean': 0.0,
            'std': 110.0,
            'min': -350.0,
            'max': 350.0,
            'unit': '°/s',
            'notes': 'Moderate yaw from forward directional movement'
        },
        'temp': {
            'mean': 38.7,
            'std': 0.20,
            'min': 38.4,
            'max': 39.2,
            'unit': '°C',
            'notes': 'Slight elevation from physical activity (+0.1-0.3°C)'
        },
        'frequency_components': {
            'dominant_frequency_hz': WALKING_FREQUENCY_HZ,
            'rhythmic': True,
            'frequency_range_hz': (1.0, 2.0),
            'notes': 'Rhythmic pattern at 1-2 Hz corresponding to walking gait cycle'
        }
    },
    
    'ruminating': {
        'description': 'Animal chewing cud with characteristic rhythmic jaw movements, can be lying or standing',
        'Fxa': {
            'mean': 0.0,
            'std': 0.9,
            'min': -2.5,
            'max': 2.5,
            'unit': 'm/s²',
            'notes': 'Minimal body movement, can be lying or standing position'
        },
        'Mya': {
            'mean': 0.0,
            'std': 1.2,
            'min': -3.5,
            'max': 3.5,
            'unit': 'm/s²',
            'notes': 'Slight lateral movements from chewing action'
        },
        'Rza': {
            'mean': 5.0,
            'std': 3.5,
            'min': -2.0,
            'max': 11.0,
            'unit': 'm/s²',
            'notes': 'Variable - can be lying (~0-3) or standing (~7-10), mid-range typical'
        },
        'Sxg': {
            'mean': 0.0,
            'std': 35.0,
            'min': -100.0,
            'max': 100.0,
            'unit': '°/s',
            'notes': 'Minimal rolling, body stationary during rumination'
        },
        'Lyg': {
            'mean': 0.0,
            'std': 180.0,
            'min': -450.0,
            'max': 450.0,
            'unit': '°/s',
            'notes': 'KEY FEATURE: Rhythmic up-down jaw movements (150-300°/s range)'
        },
        'Dzg': {
            'mean': 0.0,
            'std': 130.0,
            'min': -350.0,
            'max': 350.0,
            'unit': '°/s',
            'notes': 'KEY FEATURE: Side-to-side chewing motion (100-200°/s range)'
        },
        'temp': {
            'mean': 38.65,
            'std': 0.18,
            'min': 38.3,
            'max': 39.0,
            'unit': '°C',
            'notes': 'Baseline to slightly elevated from digestive processes'
        },
        'frequency_components': {
            'dominant_frequency_hz': RUMINATING_FREQUENCY_HZ,
            'rhythmic': True,
            'frequency_range_hz': (1.0, 1.5),
            'cycles_per_minute': (60, 90),
            'notes': 'Highly rhythmic 60-90 cycles/min in Lyg (pitch) and Dzg (yaw) - DIAGNOSTIC SIGNATURE'
        }
    },
    
    'feeding': {
        'description': 'Animal eating/grazing with head down, repetitive reaching and pulling motions',
        'Fxa': {
            'mean': 0.2,
            'std': 2.5,
            'min': -6.0,
            'max': 7.0,
            'unit': 'm/s²',
            'notes': 'Forward reaching and pulling feed motions'
        },
        'Mya': {
            'mean': 0.0,
            'std': 2.3,
            'min': -6.0,
            'max': 6.0,
            'unit': 'm/s²',
            'notes': 'Lateral movements when reaching for feed'
        },
        'Rza': {
            'mean': 3.5,
            'std': 2.2,
            'min': -1.0,
            'max': 8.0,
            'unit': 'm/s²',
            'notes': 'Head-down position (lower than standing, higher than lying)'
        },
        'Sxg': {
            'mean': 0.0,
            'std': 65.0,
            'min': -200.0,
            'max': 200.0,
            'unit': '°/s',
            'notes': 'Moderate rolling from head movements while feeding'
        },
        'Lyg': {
            'mean': -15.0,
            'std': 100.0,
            'min': -320.0,
            'max': 250.0,
            'unit': '°/s',
            'notes': 'Downward bias (negative mean) from head-down feeding posture, less rhythmic than ruminating'
        },
        'Dzg': {
            'mean': 0.0,
            'std': 80.0,
            'min': -250.0,
            'max': 250.0,
            'unit': '°/s',
            'notes': 'Head rotations when selecting and reaching for feed'
        },
        'temp': {
            'mean': 38.65,
            'std': 0.20,
            'min': 38.3,
            'max': 39.1,
            'unit': '°C',
            'notes': 'Baseline to slightly elevated during active feeding'
        },
        'frequency_components': {
            'dominant_frequency_hz': 0.5,
            'rhythmic': False,
            'notes': 'Semi-rhythmic but irregular - bite-chew-swallow cycles, not as periodic as ruminating'
        }
    },
    
    'stress': {
        'description': 'Agitated behavior with erratic movements, high variance, no consistent patterns',
        'Fxa': {
            'mean': 0.0,
            'std': 5.5,
            'min': -12.0,
            'max': 12.0,
            'unit': 'm/s²',
            'notes': 'High variance, erratic movements - pacing, restlessness'
        },
        'Mya': {
            'mean': 0.0,
            'std': 5.2,
            'min': -12.0,
            'max': 12.0,
            'unit': 'm/s²',
            'notes': 'High variance, unpredictable lateral movements'
        },
        'Rza': {
            'mean': 6.5,
            'std': 4.5,
            'min': -4.0,
            'max': 14.0,
            'unit': 'm/s²',
            'notes': 'Highly variable - rapid posture changes, erratic orientation'
        },
        'Sxg': {
            'mean': 0.0,
            'std': 280.0,
            'min': -650.0,
            'max': 650.0,
            'unit': '°/s',
            'notes': 'Very high variance, sudden rolling movements from agitation'
        },
        'Lyg': {
            'mean': 0.0,
            'std': 300.0,
            'min': -700.0,
            'max': 700.0,
            'unit': '°/s',
            'notes': 'Very high variance, rapid head movements up/down'
        },
        'Dzg': {
            'mean': 0.0,
            'std': 290.0,
            'min': -700.0,
            'max': 700.0,
            'unit': '°/s',
            'notes': 'Very high variance, rapid head rotations from alertness/agitation'
        },
        'temp': {
            'mean': 39.0,
            'std': 0.35,
            'min': 38.5,
            'max': 39.8,
            'unit': '°C',
            'notes': 'Elevated temperature from sustained stress response (38.8-39.5°C typical)'
        },
        'frequency_components': {
            'dominant_frequency_hz': None,
            'rhythmic': False,
            'notes': 'No rhythmic pattern - characterized by chaos, high variance, unpredictability'
        }
    }
}

# Validation rules for physical plausibility
VALIDATION_RULES = {
    'acceleration_absolute_max': 15.0,  # m/s² - beyond this is likely sensor error
    'angular_velocity_absolute_max': 800.0,  # °/s - beyond this is likely sensor error
    'temperature_min': 36.0,  # °C - below this indicates hypothermia or sensor error
    'temperature_max': 42.0,  # °C - above this is life-threatening or sensor error
    'rza_lying_threshold': 4.0,  # m/s² - below this suggests lying position
    'rza_standing_threshold': 7.0,  # m/s² - above this suggests standing position
    'angular_velocity_stress_threshold': 250.0,  # °/s std - above suggests stress/agitation
    'ruminating_frequency_range': (0.9, 1.6),  # Hz - characteristic chewing frequency
    'walking_frequency_range': (0.8, 2.5)  # Hz - characteristic gait frequency
}

# Transition patterns between behaviors
BEHAVIOR_TRANSITIONS = {
    'lying_to_standing': {
        'duration_seconds': (10, 30),
        'rza_change': 'gradual increase from ~0-2 to ~8-10',
        'notes': 'Animal rises from lying position to standing'
    },
    'standing_to_lying': {
        'duration_seconds': (5, 20),
        'rza_change': 'gradual decrease from ~8-10 to ~0-2',
        'notes': 'Animal lowers itself to lying position'
    },
    'standing_to_walking': {
        'duration_seconds': (2, 5),
        'pattern_change': 'acceleration std increases, rhythmic components appear',
        'notes': 'Initiation of walking gait'
    },
    'walking_to_standing': {
        'duration_seconds': (2, 5),
        'pattern_change': 'acceleration std decreases, rhythmic components fade',
        'notes': 'Deceleration and stopping'
    },
    'any_to_feeding': {
        'duration_seconds': (3, 10),
        'rza_change': 'decrease to head-down position (~3-4)',
        'notes': 'Head lowered to feeding position'
    },
    'any_to_ruminating': {
        'duration_seconds': (2, 5),
        'pattern_change': 'rhythmic Lyg/Dzg patterns emerge at 1-1.5 Hz',
        'notes': 'Onset of rhythmic chewing behavior'
    },
    'any_to_stress': {
        'duration_seconds': (1, 3),
        'pattern_change': 'sudden increase in variance across all channels',
        'notes': 'Rapid onset of agitated behavior'
    }
}

def get_behavior_signature(behavior_name):
    """
    Get the defining signature characteristics for a specific behavior.
    
    Args:
        behavior_name (str): Name of the behavior
        
    Returns:
        dict: Key distinguishing features of the behavior
    """
    signatures = {
        'lying': {
            'primary_indicators': ['Rza < 4 m/s²', 'Low angular velocity std (< 50°/s)'],
            'distinguishing_features': 'Horizontal body orientation with minimal movement'
        },
        'standing': {
            'primary_indicators': ['Rza > 7 m/s²', 'Low to moderate movement'],
            'distinguishing_features': 'Vertical neck orientation near gravity'
        },
        'walking': {
            'primary_indicators': ['Rhythmic patterns 1-2 Hz', 'High acceleration std (> 3 m/s²)'],
            'distinguishing_features': 'Periodic oscillations in all axes'
        },
        'ruminating': {
            'primary_indicators': ['Lyg rhythmic 1-1.5 Hz (60-90 cycles/min)', 'Dzg rhythmic 1-1.5 Hz'],
            'distinguishing_features': 'DIAGNOSTIC: Characteristic jaw movement frequency'
        },
        'feeding': {
            'primary_indicators': ['Rza ~3-4 m/s² (head down)', 'Lyg negative bias'],
            'distinguishing_features': 'Head-down position with repetitive reaching motions'
        },
        'stress': {
            'primary_indicators': ['Very high std all channels', 'Elevated temperature', 'No rhythm'],
            'distinguishing_features': 'Chaotic, erratic patterns with high variance'
        }
    }
    return signatures.get(behavior_name, {})

def validate_parameters(params):
    """
    Validate that parameters are physically plausible.
    
    Args:
        params (dict): Parameter dictionary for a behavior
        
    Returns:
        tuple: (is_valid, list of validation errors)
    """
    errors = []
    
    # Check acceleration channels
    for channel in ['Fxa', 'Mya', 'Rza']:
        if channel in params:
            if abs(params[channel]['max']) > VALIDATION_RULES['acceleration_absolute_max']:
                errors.append(f"{channel} max exceeds physical limits")
            if abs(params[channel]['min']) > VALIDATION_RULES['acceleration_absolute_max']:
                errors.append(f"{channel} min exceeds physical limits")
    
    # Check angular velocity channels
    for channel in ['Sxg', 'Lyg', 'Dzg']:
        if channel in params:
            if abs(params[channel]['max']) > VALIDATION_RULES['angular_velocity_absolute_max']:
                errors.append(f"{channel} max exceeds physical limits")
            if abs(params[channel]['min']) > VALIDATION_RULES['angular_velocity_absolute_max']:
                errors.append(f"{channel} min exceeds physical limits")
    
    # Check temperature
    if 'temp' in params:
        if params['temp']['min'] < VALIDATION_RULES['temperature_min']:
            errors.append(f"Temperature min too low: {params['temp']['min']}")
        if params['temp']['max'] > VALIDATION_RULES['temperature_max']:
            errors.append(f"Temperature max too high: {params['temp']['max']}")
    
    return len(errors) == 0, errors

# Export all behavior patterns with validation
def get_all_behaviors():
    """Get all behavior patterns with validation."""
    validated = {}
    for behavior, params in BEHAVIOR_PATTERNS.items():
        is_valid, errors = validate_parameters(params)
        if not is_valid:
            print(f"Warning: {behavior} has validation errors: {errors}")
        validated[behavior] = {
            'parameters': params,
            'signature': get_behavior_signature(behavior),
            'valid': is_valid
        }
    return validated

if __name__ == "__main__":
    # Display summary of all behavior patterns
    print("=" * 80)
    print("LIVESTOCK BEHAVIOR PATTERN DEFINITIONS")
    print("=" * 80)
    
    for behavior, params in BEHAVIOR_PATTERNS.items():
        print(f"\n{behavior.upper()}")
        print("-" * 80)
        print(f"Description: {params['description']}")
        print(f"\nParameters:")
        for channel in ['Fxa', 'Mya', 'Rza', 'Sxg', 'Lyg', 'Dzg', 'temp']:
            if channel in params:
                p = params[channel]
                print(f"  {channel}: mean={p['mean']:6.1f}, std={p['std']:6.1f}, "
                      f"range=[{p['min']:6.1f}, {p['max']:6.1f}] {p['unit']}")
        
        if params['frequency_components']['rhythmic']:
            fc = params['frequency_components']
            print(f"\nFrequency: {fc['dominant_frequency_hz']} Hz (rhythmic)")
        
        is_valid, errors = validate_parameters(params)
        print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
        if errors:
            for error in errors:
                print(f"  - {error}")
    
    print("\n" + "=" * 80)
    print(f"Total behaviors defined: {len(BEHAVIOR_PATTERNS)}")
    print(f"Total parameters: {len(BEHAVIOR_PATTERNS) * 7} (6 behaviors × 7 channels)")
    print("=" * 80)
