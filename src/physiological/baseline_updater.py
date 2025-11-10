"""
Baseline Updater Module

Manages dynamic baseline updates, adaptive windowing, drift detection,
and baseline history storage/retrieval.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import asdict

from .baseline_calculator import BaselineCalculator, BaselineResult

logger = logging.getLogger(__name__)


class BaselineDriftDetector:
    """
    Detect gradual baseline shifts over time.
    
    Identifies chronic illness patterns by tracking baseline temperature
    changes exceeding specified thresholds.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.5,
        drift_window_days: int = 7,
        drift_method: str = "linear_regression",
        min_confidence: float = 0.7,
    ):
        """
        Initialize drift detector.
        
        Args:
            drift_threshold: Temperature shift threshold (°C)
            drift_window_days: Time window for drift calculation (days)
            drift_method: "linear_regression" or "delta"
            min_confidence: Minimum R² for linear regression
        """
        self.drift_threshold = drift_threshold
        self.drift_window_days = drift_window_days
        self.drift_method = drift_method
        self.min_confidence = min_confidence
        
        logger.info(
            f"BaselineDriftDetector initialized: threshold={drift_threshold}°C, "
            f"window={drift_window_days}d"
        )
    
    def detect_drift(
        self,
        baseline_history: pd.DataFrame,
        current_time: datetime,
    ) -> Tuple[bool, float, float]:
        """
        Detect baseline drift in recent history.
        
        Args:
            baseline_history: DataFrame with columns [timestamp, baseline_temp]
            current_time: Current timestamp
            
        Returns:
            Tuple of (drift_detected, drift_magnitude, confidence)
        """
        if baseline_history.empty:
            return False, 0.0, 0.0
        
        # Filter to drift window
        window_start = current_time - timedelta(days=self.drift_window_days)
        window_df = baseline_history[
            (baseline_history['timestamp'] >= window_start) &
            (baseline_history['timestamp'] <= current_time)
        ].copy()
        
        if len(window_df) < 3:
            # Need at least 3 points for meaningful drift detection
            return False, 0.0, 0.0
        
        # Sort by timestamp
        window_df = window_df.sort_values('timestamp')
        
        if self.drift_method == "linear_regression":
            return self._detect_drift_regression(window_df)
        else:
            return self._detect_drift_delta(window_df)
    
    def _detect_drift_regression(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, float, float]:
        """
        Detect drift using linear regression.
        
        Args:
            df: DataFrame with timestamp and baseline_temp columns
            
        Returns:
            Tuple of (drift_detected, drift_magnitude, confidence)
        """
        # Convert timestamps to days since first reading
        first_time = df['timestamp'].iloc[0]
        df['days'] = (df['timestamp'] - first_time).dt.total_seconds() / 86400.0
        
        # Fit linear regression
        x = df['days'].values
        y = df['baseline_temp'].values
        
        # Calculate slope and R²
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]  # °C per day
        
        # Calculate R² (goodness of fit)
        y_pred = np.polyval(coeffs, x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate drift magnitude over the window
        drift_magnitude = slope * self.drift_window_days
        
        # Detect drift
        drift_detected = (
            abs(drift_magnitude) >= self.drift_threshold and
            r_squared >= self.min_confidence
        )
        
        logger.info(
            f"Drift regression: slope={slope:.4f}°C/day, "
            f"drift={drift_magnitude:.3f}°C, R²={r_squared:.3f}"
        )
        
        return drift_detected, float(drift_magnitude), float(r_squared)
    
    def _detect_drift_delta(
        self,
        df: pd.DataFrame,
    ) -> Tuple[bool, float, float]:
        """
        Detect drift using simple delta between first and last readings.
        
        Args:
            df: DataFrame with baseline_temp column
            
        Returns:
            Tuple of (drift_detected, drift_magnitude, confidence)
        """
        first_temp = df['baseline_temp'].iloc[0]
        last_temp = df['baseline_temp'].iloc[-1]
        
        drift_magnitude = last_temp - first_temp
        
        # Confidence based on consistency of trend
        temps = df['baseline_temp'].values
        trend_direction = np.sign(drift_magnitude)
        consistent_changes = 0
        
        for i in range(1, len(temps)):
            if np.sign(temps[i] - temps[i-1]) == trend_direction:
                consistent_changes += 1
        
        confidence = consistent_changes / (len(temps) - 1) if len(temps) > 1 else 0
        
        drift_detected = (
            abs(drift_magnitude) >= self.drift_threshold and
            confidence >= self.min_confidence
        )
        
        logger.info(
            f"Drift delta: magnitude={drift_magnitude:.3f}°C, "
            f"confidence={confidence:.3f}"
        )
        
        return drift_detected, float(drift_magnitude), float(confidence)


class BaselineHistoryManager:
    """
    Manage baseline history storage and retrieval.
    
    Supports database and file-based storage backends.
    """
    
    def __init__(
        self,
        storage_backend: str = "json",
        storage_path: Optional[str] = None,
        retain_days: int = 180,
    ):
        """
        Initialize history manager.
        
        Args:
            storage_backend: "json", "csv", or "database"
            storage_path: Path for file storage (default: data/baseline_history)
            retain_days: Days of history to retain
        """
        self.storage_backend = storage_backend
        self.retain_days = retain_days
        
        if storage_path is None:
            storage_path = "data/baseline_history"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"BaselineHistoryManager initialized: backend={storage_backend}, "
            f"path={self.storage_path}"
        )
    
    def store_baseline(self, result: BaselineResult) -> None:
        """
        Store baseline result in history.
        
        Args:
            result: BaselineResult to store
        """
        if self.storage_backend == "json":
            self._store_json(result)
        elif self.storage_backend == "csv":
            self._store_csv(result)
        else:
            logger.warning(f"Database storage not yet implemented")
    
    def _store_json(self, result: BaselineResult) -> None:
        """Store baseline as JSON file."""
        # Create cow-specific directory
        cow_dir = self.storage_path / f"cow_{result.cow_id}"
        cow_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        filename = f"baseline_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = cow_dir / filename
        
        # Convert to dict and save
        data = result.to_dict()
        # Convert datetime to string for JSON serialization
        data['timestamp'] = data['timestamp'].isoformat()
        if 'window_start' in data:
            data['window_start'] = data['window_start'].isoformat()
        if 'window_end' in data:
            data['window_end'] = data['window_end'].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Stored baseline to {filepath}")
    
    def _store_csv(self, result: BaselineResult) -> None:
        """Store baseline in CSV file (append mode)."""
        cow_file = self.storage_path / f"cow_{result.cow_id}_baselines.csv"
        
        # Convert to DataFrame row
        data = result.to_dict()
        df = pd.DataFrame([data])
        
        # Append to CSV
        if cow_file.exists():
            df.to_csv(cow_file, mode='a', header=False, index=False)
        else:
            df.to_csv(cow_file, index=False)
        
        logger.debug(f"Appended baseline to {cow_file}")
    
    def retrieve_history(
        self,
        cow_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Retrieve baseline history for a cow.
        
        Args:
            cow_id: Cow identifier
            start_time: Start of time range (optional)
            end_time: End of time range (optional)
            
        Returns:
            DataFrame with baseline history
        """
        if self.storage_backend == "json":
            return self._retrieve_json(cow_id, start_time, end_time)
        elif self.storage_backend == "csv":
            return self._retrieve_csv(cow_id, start_time, end_time)
        else:
            logger.warning("Database retrieval not yet implemented")
            return pd.DataFrame()
    
    def _retrieve_json(
        self,
        cow_id: int,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> pd.DataFrame:
        """Retrieve baseline history from JSON files."""
        cow_dir = self.storage_path / f"cow_{cow_id}"
        
        if not cow_dir.exists():
            logger.warning(f"No history found for cow {cow_id}")
            return pd.DataFrame()
        
        # Load all JSON files
        records = []
        for json_file in cow_dir.glob("baseline_*.json"):
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Convert timestamp strings back to datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                if 'window_start' in data:
                    data['window_start'] = pd.to_datetime(data['window_start'])
                if 'window_end' in data:
                    data['window_end'] = pd.to_datetime(data['window_end'])
                records.append(data)
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df = df.sort_values('timestamp')
        
        # Filter by time range
        if start_time is not None:
            df = df[df['timestamp'] >= start_time]
        if end_time is not None:
            df = df[df['timestamp'] <= end_time]
        
        return df
    
    def _retrieve_csv(
        self,
        cow_id: int,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
    ) -> pd.DataFrame:
        """Retrieve baseline history from CSV file."""
        cow_file = self.storage_path / f"cow_{cow_id}_baselines.csv"
        
        if not cow_file.exists():
            logger.warning(f"No history found for cow {cow_id}")
            return pd.DataFrame()
        
        df = pd.read_csv(cow_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by time range
        if start_time is not None:
            df = df[df['timestamp'] >= start_time]
        if end_time is not None:
            df = df[df['timestamp'] <= end_time]
        
        return df.sort_values('timestamp')
    
    def cleanup_old_history(self, current_time: datetime) -> None:
        """
        Remove baseline history older than retention period.
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - timedelta(days=self.retain_days)
        
        if self.storage_backend == "json":
            self._cleanup_json(cutoff_time)
        elif self.storage_backend == "csv":
            self._cleanup_csv(cutoff_time)
    
    def _cleanup_json(self, cutoff_time: datetime) -> None:
        """Cleanup old JSON files."""
        deleted_count = 0
        
        for cow_dir in self.storage_path.glob("cow_*"):
            if not cow_dir.is_dir():
                continue
            
            for json_file in cow_dir.glob("baseline_*.json"):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    timestamp = pd.to_datetime(data['timestamp'])
                    
                    if timestamp < cutoff_time:
                        json_file.unlink()
                        deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old baseline records")
    
    def _cleanup_csv(self, cutoff_time: datetime) -> None:
        """Cleanup old rows in CSV files."""
        for cow_file in self.storage_path.glob("cow_*_baselines.csv"):
            df = pd.read_csv(cow_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            initial_count = len(df)
            df = df[df['timestamp'] >= cutoff_time]
            
            if len(df) < initial_count:
                df.to_csv(cow_file, index=False)
                logger.info(
                    f"Cleaned up {initial_count - len(df)} old records from {cow_file.name}"
                )


class BaselineUpdater:
    """
    Dynamic baseline updater with adaptive windowing and drift detection.
    
    Orchestrates daily baseline recalculation, adaptive window sizing,
    drift detection, and history management.
    """
    
    def __init__(
        self,
        calculator: Optional[BaselineCalculator] = None,
        drift_detector: Optional[BaselineDriftDetector] = None,
        history_manager: Optional[BaselineHistoryManager] = None,
        update_frequency_hours: int = 24,
        adaptive_windowing: bool = True,
        initial_window_days: int = 7,
        expand_after_days: int = 14,
        smoothing_alpha: float = 0.3,
        max_change_per_day: float = 0.3,
    ):
        """
        Initialize baseline updater.
        
        Args:
            calculator: BaselineCalculator instance
            drift_detector: BaselineDriftDetector instance
            history_manager: BaselineHistoryManager instance
            update_frequency_hours: Hours between updates
            adaptive_windowing: Enable adaptive window sizing
            initial_window_days: Initial window size (days)
            expand_after_days: Days before expanding window
            smoothing_alpha: Exponential smoothing factor (0-1)
            max_change_per_day: Maximum baseline change per day (°C)
        """
        self.calculator = calculator or BaselineCalculator()
        self.drift_detector = drift_detector or BaselineDriftDetector()
        self.history_manager = history_manager or BaselineHistoryManager()
        
        self.update_frequency_hours = update_frequency_hours
        self.adaptive_windowing = adaptive_windowing
        self.initial_window_days = initial_window_days
        self.expand_after_days = expand_after_days
        self.smoothing_alpha = smoothing_alpha
        self.max_change_per_day = max_change_per_day
        
        # Track last update times per cow
        self.last_update_times: Dict[int, datetime] = {}
        
        logger.info(
            f"BaselineUpdater initialized: adaptive={adaptive_windowing}, "
            f"initial_window={initial_window_days}d"
        )
    
    def update_baseline(
        self,
        df: pd.DataFrame,
        cow_id: int,
        current_time: Optional[datetime] = None,
        force_update: bool = False,
    ) -> Optional[BaselineResult]:
        """
        Update baseline for a cow if needed.
        
        Args:
            df: DataFrame with temperature data
            cow_id: Cow identifier
            current_time: Current timestamp
            force_update: Force update regardless of schedule
            
        Returns:
            BaselineResult if updated, None if not due for update
        """
        if current_time is None:
            current_time = df['timestamp'].max()
        
        # Check if update is needed
        if not force_update and not self._is_update_needed(cow_id, current_time):
            logger.debug(f"Cow {cow_id}: Update not needed yet")
            return None
        
        # Determine adaptive window size
        window_days = self._get_adaptive_window_size(cow_id, df, current_time)
        
        # Update calculator window size
        original_window = self.calculator.window_days
        self.calculator.window_days = window_days
        
        try:
            # Calculate baseline
            result = self.calculator.calculate_baseline(
                df, cow_id, current_time=current_time
            )
            
            # Apply smoothing if previous baseline exists
            result = self._apply_smoothing(cow_id, result)
            
            # Validate change magnitude
            if not self._validate_change_magnitude(cow_id, result):
                logger.warning(
                    f"Cow {cow_id}: Baseline change exceeds threshold, "
                    f"using smoothed value"
                )
            
            # Store in history
            self.history_manager.store_baseline(result)
            
            # Update last update time
            self.last_update_times[cow_id] = current_time
            
            # Check for drift
            self._check_drift(cow_id, current_time)
            
            return result
        
        finally:
            # Restore original window size
            self.calculator.window_days = original_window
    
    def _is_update_needed(self, cow_id: int, current_time: datetime) -> bool:
        """Check if baseline update is needed."""
        if cow_id not in self.last_update_times:
            return True
        
        last_update = self.last_update_times[cow_id]
        hours_since_update = (current_time - last_update).total_seconds() / 3600
        
        return hours_since_update >= self.update_frequency_hours
    
    def _get_adaptive_window_size(
        self,
        cow_id: int,
        df: pd.DataFrame,
        current_time: datetime,
    ) -> int:
        """Determine adaptive window size based on data availability."""
        if not self.adaptive_windowing:
            return self.initial_window_days
        
        # Check how much historical data is available
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        n_unique_days = df['date'].nunique()
        
        if n_unique_days < self.expand_after_days:
            # Use shorter window initially
            window_days = min(self.initial_window_days, n_unique_days)
        else:
            # Expand to longer window
            window_days = min(30, n_unique_days)
        
        logger.info(
            f"Cow {cow_id}: Using {window_days}-day window "
            f"({n_unique_days} days available)"
        )
        
        return window_days
    
    def _apply_smoothing(
        self,
        cow_id: int,
        result: BaselineResult,
    ) -> BaselineResult:
        """Apply exponential smoothing to baseline updates."""
        # Retrieve previous baseline
        history = self.history_manager.retrieve_history(cow_id)
        
        if history.empty or len(history) < 1:
            # No previous baseline, use current as-is
            return result
        
        # Get most recent previous baseline
        prev_baseline = history.iloc[-1]['baseline_temp']
        
        # Apply exponential smoothing
        smoothed_baseline = (
            self.smoothing_alpha * result.baseline_temp +
            (1 - self.smoothing_alpha) * prev_baseline
        )
        
        logger.debug(
            f"Cow {cow_id}: Smoothed baseline {result.baseline_temp:.3f} -> "
            f"{smoothed_baseline:.3f}°C"
        )
        
        result.baseline_temp = smoothed_baseline
        return result
    
    def _validate_change_magnitude(
        self,
        cow_id: int,
        result: BaselineResult,
    ) -> bool:
        """Validate that baseline change is within acceptable limits."""
        history = self.history_manager.retrieve_history(cow_id)
        
        if history.empty or len(history) < 1:
            return True
        
        # Get most recent previous baseline
        prev_result = history.iloc[-1]
        prev_baseline = prev_result['baseline_temp']
        prev_time = prev_result['timestamp']
        
        # Calculate change rate
        time_diff_days = (result.timestamp - prev_time).total_seconds() / 86400
        baseline_change = abs(result.baseline_temp - prev_baseline)
        
        if time_diff_days > 0:
            change_per_day = baseline_change / time_diff_days
            
            if change_per_day > self.max_change_per_day:
                # Apply additional smoothing
                max_change = self.max_change_per_day * time_diff_days
                direction = np.sign(result.baseline_temp - prev_baseline)
                result.baseline_temp = prev_baseline + direction * max_change
                return False
        
        return True
    
    def _check_drift(self, cow_id: int, current_time: datetime) -> None:
        """Check for baseline drift and log warning if detected."""
        history = self.history_manager.retrieve_history(cow_id)
        
        if len(history) < 3:
            return
        
        drift_detected, magnitude, confidence = self.drift_detector.detect_drift(
            history, current_time
        )
        
        if drift_detected:
            logger.warning(
                f"BASELINE DRIFT DETECTED for Cow {cow_id}: "
                f"magnitude={magnitude:+.3f}°C over {self.drift_detector.drift_window_days} days "
                f"(confidence={confidence:.2f})"
            )
            # In a full implementation, this would create an alert
    
    def get_current_baseline(
        self,
        cow_id: int,
        timestamp: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Get current baseline for a cow at specified timestamp.
        
        Args:
            cow_id: Cow identifier
            timestamp: Timestamp to retrieve baseline for (default: most recent)
            
        Returns:
            Baseline temperature or None if not available
        """
        history = self.history_manager.retrieve_history(cow_id)
        
        if history.empty:
            return None
        
        if timestamp is None:
            # Return most recent
            return history.iloc[-1]['baseline_temp']
        
        # Find closest baseline before timestamp
        past_baselines = history[history['timestamp'] <= timestamp]
        
        if past_baselines.empty:
            return None
        
        return past_baselines.iloc[-1]['baseline_temp']
