"""
Behavior Pattern Definitions for Synthetic Data Generation

This module defines the sensor parameter specifications for different animal behaviors.
Each behavior specifies mean and standard deviation values for all 7 sensor channels,
plus optional frequency components for rhythmic behaviors.

Sensor Channels:
- temp: Body temperature (°C)
- Fxa: Forward-backward acceleration (m/s²)
- Mya: Lateral acceleration (m/s²)
- Rza: Vertical acceleration (m/s²)
- Sxg: Roll angular velocity (rad/s)
- Lyg: Pitch angular velocity (rad/s)
- Dzg: Yaw angular velocity (rad/s)
"""

BEHAVIOR_PATTERNS = {
    'lying': {
        'description': 'Animal lying down, minimal movement',
        'parameters': {
            'temp': {'mean': 38.5, 'std': 0.3},
            'Fxa': {'mean': 0.0, 'std': 0.5},
            'Mya': {'mean': 0.0, 'std': 0.3},
            'Rza': {'mean': 0.0, 'std': 0.5},  # Horizontal orientation
            'Sxg': {'mean': 0.0, 'std': 0.05},
            'Lyg': {'mean': 0.0, 'std': 0.05},
            'Dzg': {'mean': 0.0, 'std': 0.05}
        },
        'frequencies': {}  # No rhythmic components
    },
    
    'standing': {
        'description': 'Animal standing still',
        'parameters': {
            'temp': {'mean': 38.6, 'std': 0.3},
            'Fxa': {'mean': 0.0, 'std': 0.8},
            'Mya': {'mean': 0.0, 'std': 0.6},
            'Rza': {'mean': 9.8, 'std': 1.0},  # Vertical orientation (gravity)
            'Sxg': {'mean': 0.0, 'std': 0.08},
            'Lyg': {'mean': 0.0, 'std': 0.08},
            'Dzg': {'mean': 0.0, 'std': 0.08}
        },
        'frequencies': {}
    },
    
    'walking': {
        'description': 'Animal walking with rhythmic gait',
        'parameters': {
            'temp': {'mean': 38.7, 'std': 0.4},
            'Fxa': {'mean': 2.0, 'std': 1.5},  # Forward motion
            'Mya': {'mean': 0.0, 'std': 1.2},  # Lateral sway
            'Rza': {'mean': 9.8, 'std': 1.5},  # Upright with bounce
            'Sxg': {'mean': 0.0, 'std': 0.15},
            'Lyg': {'mean': 0.0, 'std': 0.12},
            'Dzg': {'mean': 0.0, 'std': 0.10}
        },
        'frequencies': {
            'Fxa': [{'freq': 1.5, 'amplitude': 1.0}],  # ~90 steps/min
            'Mya': [{'freq': 1.5, 'amplitude': 0.8}],
            'Rza': [{'freq': 1.5, 'amplitude': 0.6}]
        }
    },
    
    'ruminating': {
        'description': 'Animal chewing cud with rhythmic jaw movement',
        'parameters': {
            'temp': {'mean': 38.5, 'std': 0.3},
            'Fxa': {'mean': 0.0, 'std': 0.5},
            'Mya': {'mean': 0.0, 'std': 1.0},  # Lateral jaw movement
            'Rza': {'mean': 9.8, 'std': 0.8},  # Usually standing
            'Sxg': {'mean': 0.0, 'std': 0.06},
            'Lyg': {'mean': 0.0, 'std': 0.15},  # Vertical jaw movement
            'Dzg': {'mean': 0.0, 'std': 0.06}
        },
        'frequencies': {
            'Mya': [{'freq': 1.0, 'amplitude': 0.7}],  # ~60 chews/min
            'Lyg': [{'freq': 1.0, 'amplitude': 0.10}]
        }
    },
    
    'feeding': {
        'description': 'Animal eating with head down',
        'parameters': {
            'temp': {'mean': 38.6, 'std': 0.3},
            'Fxa': {'mean': 0.5, 'std': 1.0},  # Some forward movement
            'Mya': {'mean': 0.0, 'std': 0.8},
            'Rza': {'mean': 7.0, 'std': 1.5},  # Head down, less vertical
            'Sxg': {'mean': 0.0, 'std': 0.10},
            'Lyg': {'mean': -0.3, 'std': 0.20},  # Head tilted down
            'Dzg': {'mean': 0.0, 'std': 0.12}
        },
        'frequencies': {
            'Lyg': [{'freq': 0.5, 'amplitude': 0.08}]  # Slow head movements
        }
    },
    
    'resting': {
        'description': 'Animal resting quietly (standing or lying)',
        'parameters': {
            'temp': {'mean': 38.4, 'std': 0.3},
            'Fxa': {'mean': 0.0, 'std': 0.3},
            'Mya': {'mean': 0.0, 'std': 0.2},
            'Rza': {'mean': 5.0, 'std': 3.0},  # Variable position
            'Sxg': {'mean': 0.0, 'std': 0.03},
            'Lyg': {'mean': 0.0, 'std': 0.03},
            'Dzg': {'mean': 0.0, 'std': 0.03}
        },
        'frequencies': {}
    }
}

# Validation function
def validate_behavior(behavior_name):
    """Check if behavior name exists in patterns"""
    return behavior_name in BEHAVIOR_PATTERNS

def get_behavior_params(behavior_name):
    """Get parameters for a specific behavior"""
    if not validate_behavior(behavior_name):
        raise ValueError(f"Unknown behavior: {behavior_name}. "
                        f"Valid behaviors: {list(BEHAVIOR_PATTERNS.keys())}")
    return BEHAVIOR_PATTERNS[behavior_name]

def get_all_behaviors():
    """Get list of all available behaviors"""
    return list(BEHAVIOR_PATTERNS.keys())
