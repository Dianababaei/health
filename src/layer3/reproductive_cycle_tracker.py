"""
Reproductive Cycle Tracking System

Tracks estrus cycles and pregnancy events for individual cows, predicting next
expected estrus based on 21-day cycle patterns and managing reproductive history
for pregnancy detection.

Features:
- 21-day estrus cycle tracking per cow
- Next estrus prediction based on historical patterns
- Estrus-pregnancy event linkage
- Reproductive history retention (90-180 days)
- Cycle irregularity detection
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EstrusRecord:
    """Record of a single estrus event."""
    event_id: str
    cow_id: str
    timestamp: datetime
    confidence: float
    cycle_day: Optional[int] = None  # Day in cycle (0-21)
    cycle_number: Optional[int] = None  # Which cycle this is for the cow
    is_predicted: bool = False  # Whether this was predicted or detected
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'cow_id': self.cow_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'cycle_day': self.cycle_day,
            'cycle_number': self.cycle_number,
            'is_predicted': self.is_predicted
        }


@dataclass
class PregnancyRecord:
    """Record of a pregnancy indication event."""
    event_id: str
    cow_id: str
    timestamp: datetime
    confidence: float
    conception_date: Optional[datetime] = None
    linked_estrus_id: Optional[str] = None
    days_pregnant: int = 0
    is_confirmed: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'cow_id': self.cow_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence': self.confidence,
            'conception_date': self.conception_date.isoformat() if self.conception_date else None,
            'linked_estrus_id': self.linked_estrus_id,
            'days_pregnant': self.days_pregnant,
            'is_confirmed': self.is_confirmed
        }


@dataclass
class ReproductiveCycleState:
    """Current reproductive state for a cow."""
    cow_id: str
    current_cycle_day: int = 0
    cycle_number: int = 0
    last_estrus: Optional[EstrusRecord] = None
    next_predicted_estrus: Optional[datetime] = None
    cycle_length_mean: float = 21.0
    cycle_length_std: float = 2.0
    is_pregnant: bool = False
    pregnancy_record: Optional[PregnancyRecord] = None
    estrus_history: List[EstrusRecord] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'cow_id': self.cow_id,
            'current_cycle_day': self.current_cycle_day,
            'cycle_number': self.cycle_number,
            'last_estrus': self.last_estrus.to_dict() if self.last_estrus else None,
            'next_predicted_estrus': self.next_predicted_estrus.isoformat() if self.next_predicted_estrus else None,
            'cycle_length_mean': self.cycle_length_mean,
            'cycle_length_std': self.cycle_length_std,
            'is_pregnant': self.is_pregnant,
            'pregnancy_record': self.pregnancy_record.to_dict() if self.pregnancy_record else None,
            'estrus_history_count': len(self.estrus_history)
        }


class ReproductiveCycleTracker:
    """
    Tracks reproductive cycles and events for individual cows.
    
    Manages:
    - Estrus cycle tracking (21-day average cycle)
    - Next estrus prediction based on historical patterns
    - Estrus-pregnancy event linkage
    - Reproductive history with configurable retention
    - Cycle irregularity detection
    """
    
    # Default cycle parameters (based on bovine reproductive physiology)
    DEFAULT_CYCLE_LENGTH = 21.0  # days
    DEFAULT_CYCLE_STD = 2.0  # days
    CYCLE_RANGE_MIN = 18  # days
    CYCLE_RANGE_MAX = 24  # days
    
    # Pregnancy parameters
    GESTATION_PERIOD = 283  # days (approximately 9 months)
    PREGNANCY_CONFIRMATION_DAYS = 30
    
    # History retention
    DEFAULT_RETENTION_DAYS = 180  # 6 months
    
    def __init__(
        self,
        cycle_length: float = DEFAULT_CYCLE_LENGTH,
        cycle_std: float = DEFAULT_CYCLE_STD,
        retention_days: int = DEFAULT_RETENTION_DAYS
    ):
        """
        Initialize reproductive cycle tracker.
        
        Args:
            cycle_length: Expected cycle length in days (default: 21)
            cycle_std: Standard deviation for cycle length (default: 2)
            retention_days: Days to retain reproductive history (default: 180)
        """
        self.cycle_length = cycle_length
        self.cycle_std = cycle_std
        self.retention_days = retention_days
        
        # Track reproductive state per cow
        self.cow_states: Dict[str, ReproductiveCycleState] = {}
        
        # Historical event storage
        self.estrus_events: Dict[str, List[EstrusRecord]] = defaultdict(list)
        self.pregnancy_events: Dict[str, List[PregnancyRecord]] = defaultdict(list)
        
        logger.info(
            f"ReproductiveCycleTracker initialized: "
            f"cycle_length={cycle_length} days, "
            f"retention={retention_days} days"
        )
    
    def record_estrus(
        self,
        cow_id: str,
        event_id: str,
        timestamp: datetime,
        confidence: float
    ) -> EstrusRecord:
        """
        Record a detected estrus event and update cycle tracking.
        
        Args:
            cow_id: Animal identifier
            event_id: Unique event identifier
            timestamp: Detection timestamp
            confidence: Detection confidence (0.0-1.0)
            
        Returns:
            EstrusRecord object
        """
        # Get or create cycle state
        if cow_id not in self.cow_states:
            self.cow_states[cow_id] = ReproductiveCycleState(cow_id=cow_id)
        
        state = self.cow_states[cow_id]
        
        # Calculate cycle information
        cycle_day = 0  # Estrus is day 0 of new cycle
        cycle_number = state.cycle_number + 1
        
        # Update cycle length statistics if we have previous estrus
        if state.last_estrus is not None:
            days_since_last = (timestamp - state.last_estrus.timestamp).days
            
            # Update running average of cycle length
            if self.CYCLE_RANGE_MIN <= days_since_last <= self.CYCLE_RANGE_MAX:
                # Valid cycle - update statistics
                prev_mean = state.cycle_length_mean
                n = len(state.estrus_history)
                
                # Running average
                state.cycle_length_mean = (prev_mean * n + days_since_last) / (n + 1)
                
                # Update standard deviation estimate
                if n > 0:
                    variances = [(e.timestamp - state.estrus_history[i-1].timestamp).days - state.cycle_length_mean 
                                for i, e in enumerate(state.estrus_history[1:], 1)]
                    variances.append(days_since_last - state.cycle_length_mean)
                    state.cycle_length_std = np.std(variances) if len(variances) > 1 else self.cycle_std
            else:
                logger.warning(
                    f"Irregular cycle detected for cow {cow_id}: "
                    f"{days_since_last} days (expected {self.CYCLE_RANGE_MIN}-{self.CYCLE_RANGE_MAX})"
                )
        
        # Create estrus record
        record = EstrusRecord(
            event_id=event_id,
            cow_id=cow_id,
            timestamp=timestamp,
            confidence=confidence,
            cycle_day=cycle_day,
            cycle_number=cycle_number,
            is_predicted=False
        )
        
        # Update state
        state.last_estrus = record
        state.current_cycle_day = 0
        state.cycle_number = cycle_number
        state.estrus_history.append(record)
        
        # Predict next estrus
        state.next_predicted_estrus = self._predict_next_estrus(state)
        
        # Store in history
        self.estrus_events[cow_id].append(record)
        
        logger.info(
            f"Estrus recorded for cow {cow_id}: cycle #{cycle_number}, "
            f"avg_cycle_length={state.cycle_length_mean:.1f} days"
        )
        
        return record
    
    def record_pregnancy(
        self,
        cow_id: str,
        event_id: str,
        timestamp: datetime,
        confidence: float,
        linked_estrus_id: Optional[str] = None
    ) -> PregnancyRecord:
        """
        Record a detected pregnancy indication and link to estrus event.
        
        Args:
            cow_id: Animal identifier
            event_id: Unique event identifier
            timestamp: Detection timestamp
            confidence: Detection confidence (0.0-1.0)
            linked_estrus_id: ID of related estrus event
            
        Returns:
            PregnancyRecord object
        """
        # Get cycle state
        state = self.cow_states.get(cow_id)
        
        if state is None:
            logger.warning(f"No cycle state for cow {cow_id}, creating new state")
            state = ReproductiveCycleState(cow_id=cow_id)
            self.cow_states[cow_id] = state
        
        # Find linked estrus event
        linked_estrus = None
        conception_date = None
        
        if linked_estrus_id:
            for estrus in self.estrus_events[cow_id]:
                if estrus.event_id == linked_estrus_id:
                    linked_estrus = estrus
                    conception_date = estrus.timestamp
                    break
        
        # If no linked estrus provided, use most recent
        if conception_date is None and state.last_estrus:
            conception_date = state.last_estrus.timestamp
            linked_estrus_id = state.last_estrus.event_id
        
        # Calculate days pregnant
        days_pregnant = 0
        if conception_date:
            days_pregnant = (timestamp - conception_date).days
        
        # Determine if confirmed (30+ days)
        is_confirmed = days_pregnant >= self.PREGNANCY_CONFIRMATION_DAYS
        
        # Create pregnancy record
        record = PregnancyRecord(
            event_id=event_id,
            cow_id=cow_id,
            timestamp=timestamp,
            confidence=confidence,
            conception_date=conception_date,
            linked_estrus_id=linked_estrus_id,
            days_pregnant=days_pregnant,
            is_confirmed=is_confirmed
        )
        
        # Update state
        state.is_pregnant = True
        state.pregnancy_record = record
        
        # Store in history
        self.pregnancy_events[cow_id].append(record)
        
        logger.info(
            f"Pregnancy recorded for cow {cow_id}: "
            f"days_pregnant={days_pregnant}, "
            f"confirmed={is_confirmed}, "
            f"linked_estrus={linked_estrus_id}"
        )
        
        return record
    
    def get_reproductive_state(self, cow_id: str) -> Optional[ReproductiveCycleState]:
        """
        Get current reproductive state for a cow.
        
        Args:
            cow_id: Animal identifier
            
        Returns:
            ReproductiveCycleState or None if not tracked
        """
        return self.cow_states.get(cow_id)
    
    def predict_next_estrus(self, cow_id: str) -> Optional[datetime]:
        """
        Predict next estrus date for a cow.
        
        Args:
            cow_id: Animal identifier
            
        Returns:
            Predicted datetime or None if insufficient data
        """
        state = self.cow_states.get(cow_id)
        
        if state is None or state.last_estrus is None:
            return None
        
        # If pregnant, no estrus expected
        if state.is_pregnant:
            return None
        
        return state.next_predicted_estrus
    
    def _predict_next_estrus(self, state: ReproductiveCycleState) -> Optional[datetime]:
        """
        Internal method to predict next estrus based on cycle history.
        
        Uses individual cow's cycle length mean and std for prediction.
        """
        if state.last_estrus is None:
            return None
        
        # If pregnant, no next estrus
        if state.is_pregnant:
            return None
        
        # Predict based on individual cycle length
        predicted_days = state.cycle_length_mean
        next_estrus = state.last_estrus.timestamp + timedelta(days=predicted_days)
        
        return next_estrus
    
    def get_estrus_history(
        self,
        cow_id: str,
        days: Optional[int] = None
    ) -> List[EstrusRecord]:
        """
        Get estrus history for a cow.
        
        Args:
            cow_id: Animal identifier
            days: Number of days to look back (None for all)
            
        Returns:
            List of EstrusRecord objects
        """
        if cow_id not in self.estrus_events:
            return []
        
        events = self.estrus_events[cow_id]
        
        if days is None:
            return events
        
        # Filter to specified timeframe
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in events if e.timestamp >= cutoff]
    
    def get_pregnancy_history(
        self,
        cow_id: str,
        days: Optional[int] = None
    ) -> List[PregnancyRecord]:
        """
        Get pregnancy history for a cow.
        
        Args:
            cow_id: Animal identifier
            days: Number of days to look back (None for all)
            
        Returns:
            List of PregnancyRecord objects
        """
        if cow_id not in self.pregnancy_events:
            return []
        
        events = self.pregnancy_events[cow_id]
        
        if days is None:
            return events
        
        # Filter to specified timeframe
        cutoff = datetime.now() - timedelta(days=days)
        return [e for e in events if e.timestamp >= cutoff]
    
    def get_linked_events(
        self,
        cow_id: str,
        estrus_id: str
    ) -> Tuple[Optional[EstrusRecord], List[PregnancyRecord]]:
        """
        Get all events linked to a specific estrus event.
        
        Args:
            cow_id: Animal identifier
            estrus_id: Estrus event identifier
            
        Returns:
            Tuple of (estrus_record, linked_pregnancy_records)
        """
        # Find estrus event
        estrus_record = None
        for estrus in self.estrus_events.get(cow_id, []):
            if estrus.event_id == estrus_id:
                estrus_record = estrus
                break
        
        # Find linked pregnancies
        linked_pregnancies = [
            p for p in self.pregnancy_events.get(cow_id, [])
            if p.linked_estrus_id == estrus_id
        ]
        
        return estrus_record, linked_pregnancies
    
    def update_cycle_day(self, cow_id: str, current_time: datetime):
        """
        Update current cycle day based on time since last estrus.
        
        Args:
            cow_id: Animal identifier
            current_time: Current timestamp
        """
        state = self.cow_states.get(cow_id)
        
        if state is None or state.last_estrus is None:
            return
        
        # Calculate days since last estrus
        days_since = (current_time - state.last_estrus.timestamp).days
        state.current_cycle_day = days_since
        
        # Update pregnancy days if pregnant
        if state.is_pregnant and state.pregnancy_record:
            if state.pregnancy_record.conception_date:
                state.pregnancy_record.days_pregnant = (
                    current_time - state.pregnancy_record.conception_date
                ).days
    
    def cleanup_old_records(self, cutoff_date: Optional[datetime] = None):
        """
        Remove records older than retention period.
        
        Args:
            cutoff_date: Date before which to remove records (default: retention_days ago)
        """
        if cutoff_date is None:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        removed_count = 0
        
        # Clean estrus events
        for cow_id in list(self.estrus_events.keys()):
            before = len(self.estrus_events[cow_id])
            self.estrus_events[cow_id] = [
                e for e in self.estrus_events[cow_id]
                if e.timestamp >= cutoff_date
            ]
            after = len(self.estrus_events[cow_id])
            removed_count += (before - after)
            
            # Update state history
            if cow_id in self.cow_states:
                self.cow_states[cow_id].estrus_history = [
                    e for e in self.cow_states[cow_id].estrus_history
                    if e.timestamp >= cutoff_date
                ]
        
        # Clean pregnancy events
        for cow_id in list(self.pregnancy_events.keys()):
            before = len(self.pregnancy_events[cow_id])
            self.pregnancy_events[cow_id] = [
                e for e in self.pregnancy_events[cow_id]
                if e.timestamp >= cutoff_date
            ]
            removed_count += (before - after)
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old reproductive records")
    
    def get_cycle_statistics(self, cow_id: str) -> Dict:
        """
        Get cycle statistics for a cow.
        
        Args:
            cow_id: Animal identifier
            
        Returns:
            Dictionary with cycle statistics
        """
        state = self.cow_states.get(cow_id)
        
        if state is None:
            return {
                'tracked': False,
                'cycle_count': 0
            }
        
        estrus_history = self.estrus_events.get(cow_id, [])
        
        # Calculate inter-cycle intervals
        intervals = []
        for i in range(1, len(estrus_history)):
            interval = (estrus_history[i].timestamp - estrus_history[i-1].timestamp).days
            intervals.append(interval)
        
        return {
            'tracked': True,
            'cycle_count': len(estrus_history),
            'current_cycle_day': state.current_cycle_day,
            'cycle_length_mean': state.cycle_length_mean,
            'cycle_length_std': state.cycle_length_std,
            'last_estrus': state.last_estrus.timestamp if state.last_estrus else None,
            'next_predicted_estrus': state.next_predicted_estrus,
            'is_pregnant': state.is_pregnant,
            'intervals': intervals,
            'interval_min': min(intervals) if intervals else None,
            'interval_max': max(intervals) if intervals else None
        }
