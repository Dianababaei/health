"""
Behavior patterns and daily activity schedules for livestock monitoring.

This module defines:
- Time-of-day activity schedules with behavior probabilities
- Daily sequence templates for different activity levels
- Behavior transition rules
"""

# Behavior types
BEHAVIORS = ['lying', 'standing', 'walking', 'feeding', 'ruminating']

# Time-of-day activity schedules
# Format: {behavior: probability} for each time period
# Probabilities should sum to ~100 for each period

NIGHT_SCHEDULE = {
    # 00:00-06:00: Mostly lying (rest period)
    'lying': 0.80,
    'ruminating': 0.15,
    'standing': 0.05,
    'walking': 0.0,
    'feeding': 0.0
}

MORNING_SCHEDULE = {
    # 06:00-10:00: Transition to active - morning feeding
    'feeding': 0.30,
    'ruminating': 0.25,
    'standing': 0.20,
    'walking': 0.15,
    'lying': 0.10
}

MIDDAY_SCHEDULE = {
    # 10:00-14:00: Rest period - midday rest
    'lying': 0.50,
    'ruminating': 0.30,
    'standing': 0.20,
    'walking': 0.0,
    'feeding': 0.0
}

AFTERNOON_SCHEDULE = {
    # 14:00-18:00: Active - afternoon feeding and activity
    'feeding': 0.25,
    'walking': 0.20,
    'standing': 0.20,
    'ruminating': 0.20,
    'lying': 0.15
}

EVENING_SCHEDULE = {
    # 18:00-24:00: Wind down - preparing for rest
    'lying': 0.40,
    'ruminating': 0.30,
    'standing': 0.20,
    'feeding': 0.10,
    'walking': 0.0
}

# Hourly schedule map (hour -> schedule)
HOURLY_SCHEDULE = {}
for hour in range(24):
    if 0 <= hour < 6:
        HOURLY_SCHEDULE[hour] = NIGHT_SCHEDULE
    elif 6 <= hour < 10:
        HOURLY_SCHEDULE[hour] = MORNING_SCHEDULE
    elif 10 <= hour < 14:
        HOURLY_SCHEDULE[hour] = MIDDAY_SCHEDULE
    elif 14 <= hour < 18:
        HOURLY_SCHEDULE[hour] = AFTERNOON_SCHEDULE
    else:  # 18 <= hour < 24
        HOURLY_SCHEDULE[hour] = EVENING_SCHEDULE


# Daily sequence templates
# Format: List of (start_minute, end_minute, behavior)
# These provide realistic daily behavior sequences with typical durations

TYPICAL_DAY_SEQUENCE = [
    # Night (00:00-06:00)
    (0, 360, 'lying'),           # 6 hours lying (night rest)
    
    # Morning (06:00-10:00)
    (360, 370, 'standing'),      # 10 min standing (wake up)
    (370, 410, 'feeding'),       # 40 min morning feeding
    (410, 440, 'walking'),       # 30 min walking to water/pasture
    (440, 500, 'ruminating'),    # 60 min ruminating
    (500, 540, 'standing'),      # 40 min standing
    (540, 600, 'feeding'),       # 60 min feeding
    
    # Midday (10:00-14:00)
    (600, 660, 'ruminating'),    # 60 min ruminating
    (660, 780, 'lying'),         # 120 min midday rest
    (780, 840, 'ruminating'),    # 60 min ruminating
    
    # Afternoon (14:00-18:00)
    (840, 860, 'standing'),      # 20 min standing
    (860, 910, 'feeding'),       # 50 min afternoon feeding
    (910, 950, 'walking'),       # 40 min walking
    (950, 1020, 'ruminating'),   # 70 min ruminating
    (1020, 1080, 'standing'),    # 60 min standing
    
    # Evening (18:00-24:00)
    (1080, 1110, 'feeding'),     # 30 min evening feeding
    (1110, 1150, 'ruminating'),  # 40 min ruminating
    (1150, 1260, 'lying'),       # 110 min lying
    (1260, 1320, 'ruminating'),  # 60 min ruminating
    (1320, 1440, 'lying'),       # 120 min lying (transition to night)
]

HIGH_ACTIVITY_SEQUENCE = [
    # Night (00:00-06:00)
    (0, 300, 'lying'),           # 5 hours lying (less rest)
    (300, 360, 'ruminating'),    # 60 min ruminating
    
    # Morning (06:00-10:00)
    (360, 370, 'standing'),      # 10 min standing
    (370, 420, 'feeding'),       # 50 min morning feeding (more)
    (420, 460, 'walking'),       # 40 min walking (more)
    (460, 510, 'ruminating'),    # 50 min ruminating
    (510, 540, 'standing'),      # 30 min standing
    (540, 600, 'feeding'),       # 60 min feeding
    
    # Midday (10:00-14:00)
    (600, 640, 'ruminating'),    # 40 min ruminating
    (640, 700, 'lying'),         # 60 min rest (shorter)
    (700, 760, 'walking'),       # 60 min walking (more active)
    (760, 840, 'ruminating'),    # 80 min ruminating
    
    # Afternoon (14:00-18:00)
    (840, 870, 'standing'),      # 30 min standing
    (870, 930, 'feeding'),       # 60 min feeding (more)
    (930, 980, 'walking'),       # 50 min walking (more)
    (980, 1030, 'ruminating'),   # 50 min ruminating
    (1030, 1080, 'standing'),    # 50 min standing
    
    # Evening (18:00-24:00)
    (1080, 1120, 'feeding'),     # 40 min feeding
    (1120, 1180, 'walking'),     # 60 min walking (more active)
    (1180, 1240, 'ruminating'),  # 60 min ruminating
    (1240, 1320, 'lying'),       # 80 min lying
    (1320, 1380, 'ruminating'),  # 60 min ruminating
    (1380, 1440, 'lying'),       # 60 min lying
]

LOW_ACTIVITY_SEQUENCE = [
    # Night (00:00-06:00)
    (0, 360, 'lying'),           # 6 hours lying (full rest)
    
    # Morning (06:00-10:00)
    (360, 380, 'standing'),      # 20 min standing
    (380, 410, 'feeding'),       # 30 min morning feeding (less)
    (410, 430, 'walking'),       # 20 min walking (less)
    (430, 500, 'ruminating'),    # 70 min ruminating
    (500, 600, 'lying'),         # 100 min lying (more rest)
    
    # Midday (10:00-14:00)
    (600, 660, 'ruminating'),    # 60 min ruminating
    (660, 840, 'lying'),         # 180 min lying (long rest)
    
    # Afternoon (14:00-18:00)
    (840, 860, 'standing'),      # 20 min standing
    (860, 900, 'feeding'),       # 40 min feeding (less)
    (900, 920, 'walking'),       # 20 min walking (less)
    (920, 1000, 'ruminating'),   # 80 min ruminating
    (1000, 1080, 'lying'),       # 80 min lying (more rest)
    
    # Evening (18:00-24:00)
    (1080, 1110, 'feeding'),     # 30 min feeding
    (1110, 1200, 'ruminating'),  # 90 min ruminating
    (1200, 1440, 'lying'),       # 240 min lying (long rest)
]

# Dictionary of all sequence templates
SEQUENCE_TEMPLATES = {
    'typical': TYPICAL_DAY_SEQUENCE,
    'high_activity': HIGH_ACTIVITY_SEQUENCE,
    'low_activity': LOW_ACTIVITY_SEQUENCE
}

# Behavior transition probabilities (from -> to)
# Higher values = more likely transition
TRANSITION_MATRIX = {
    'lying': {'standing': 0.4, 'ruminating': 0.3, 'lying': 0.2, 'walking': 0.1, 'feeding': 0.0},
    'standing': {'walking': 0.3, 'feeding': 0.25, 'ruminating': 0.2, 'lying': 0.15, 'standing': 0.1},
    'walking': {'standing': 0.4, 'feeding': 0.3, 'ruminating': 0.15, 'lying': 0.1, 'walking': 0.05},
    'feeding': {'ruminating': 0.4, 'standing': 0.3, 'walking': 0.15, 'lying': 0.1, 'feeding': 0.05},
    'ruminating': {'lying': 0.35, 'standing': 0.3, 'ruminating': 0.2, 'walking': 0.1, 'feeding': 0.05}
}

# Minimum duration for each behavior (in minutes)
MIN_BEHAVIOR_DURATION = {
    'lying': 10,
    'standing': 5,
    'walking': 5,
    'feeding': 10,
    'ruminating': 10
}

# Maximum duration for each behavior (in minutes)
MAX_BEHAVIOR_DURATION = {
    'lying': 180,      # 3 hours
    'standing': 60,    # 1 hour
    'walking': 60,     # 1 hour
    'feeding': 90,     # 1.5 hours
    'ruminating': 120  # 2 hours
}
