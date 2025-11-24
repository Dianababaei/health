"""
Detector Pipeline Module

Orchestrates data queries and execution of all detection systems:
- Immediate alerts (fever, heat stress, inactivity, sensor malfunction)
- Estrus detection (reproductive health monitoring)
- Health scoring (overall health assessment)

Features:
- Active cow identification based on recent data presence
- Time-windowed data queries optimized for each detector type
- Integration with all existing detector components
- Graceful error handling and comprehensive logging
- Performance optimized for processing multiple cows
"""

import logging
import time
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.health_intelligence.logging.sensor_data_manager import SensorDataManager
from src.health_intelligence.logging.alert_state_manager import AlertStateManager
from src.health_intelligence.logging.health_score_manager import HealthScoreManager
from src.health_intelligence.alerts.immediate_detector import ImmediateAlertDetector
from src.health_intelligence.reproductive.estrus_detector import EstrusDetector
from src.health_intelligence.scoring.simple_health_scorer import SimpleHealthScorer


logger = logging.getLogger(__name__)


class DetectorPipeline:
    """
    Detector pipeline orchestrating data queries and detector execution.
    
    Manages the complete detection workflow:
    1. Identify active cows (those with recent sensor data)
    2. Query appropriate time windows for each detector type
    3. Execute all detection systems with proper data formatting
    4. Store results via appropriate managers
    5. Handle errors gracefully and provide comprehensive logging
    
    Features:
    - Configurable active cow threshold (default 24h)
    - Optimized time-windowed queries (2min, 48h, 24h)
    - Per-cow error handling (failures don't stop other cows)
    - Performance metrics and execution timing
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        db_path: str = "data/alert_state.db"
    ):
        """
        Initialize detector pipeline.
        
        Args:
            config: Configuration dictionary (optional)
            db_path: Path to shared database
        """
        self.config = config or {}
        self.db_path = Path(db_path)
        
        # Active cow threshold (hours)
        self.active_cow_threshold_hours = self.config.get(
            'active_cow_threshold_hours', 24
        )
        
        # Initialize managers
        logger.info("Initializing detector pipeline managers...")
        self.sensor_manager = SensorDataManager(db_path=str(self.db_path))
        self.alert_manager = AlertStateManager(db_path=str(self.db_path))
        self.health_score_manager = HealthScoreManager(db_path=str(self.db_path))
        
        # Initialize detectors
        logger.info("Initializing detection systems...")
        config_path = self.config.get(
            'alert_thresholds_path',
            'config/alert_thresholds.yaml'
        )
        self.immediate_detector = ImmediateAlertDetector(config_path=config_path)
        
        baseline_temp = self.config.get('baseline_temperature', 38.5)
        self.estrus_detector = EstrusDetector(baseline_temp=baseline_temp)
        
        self.health_scorer = SimpleHealthScorer()
        
        logger.info(f"DetectorPipeline initialized: db={self.db_path}, "
                   f"active_threshold={self.active_cow_threshold_hours}h")
    
    def get_active_cows(self) -> List[str]:
        """
        Query distinct cow_ids with data within threshold.
        
        Returns list of cow_ids that have sensor data within the
        active_cow_threshold_hours window (e.g., last 24h).
        
        Returns:
            List of cow IDs that are considered active
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Calculate cutoff time
            cutoff_time = datetime.now() - timedelta(
                hours=self.active_cow_threshold_hours
            )
            
            # Query distinct cow_ids with recent data
            query = """
                SELECT DISTINCT cow_id
                FROM sensor_data
                WHERE timestamp >= ?
                ORDER BY cow_id
            """
            
            cursor.execute(query, (cutoff_time.isoformat(),))
            rows = cursor.fetchall()
            conn.close()
            
            cow_ids = [row[0] for row in rows]
            
            logger.info(f"Found {len(cow_ids)} active cows with data in last "
                       f"{self.active_cow_threshold_hours}h")
            
            return cow_ids
            
        except Exception as e:
            logger.error(f"Error getting active cows: {e}", exc_info=True)
            return []
    
    def query_sensor_window(
        self,
        cow_id: str,
        hours: float
    ) -> pd.DataFrame:
        """
        Query sensor_data table for cow within time window.
        
        Args:
            cow_id: Cow identifier
            hours: Number of hours to look back
        
        Returns:
            DataFrame with raw sensor data (all columns)
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            df = self.sensor_manager.get_sensor_data(
                cow_id=cow_id,
                start_time=start_time,
                end_time=end_time
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error querying sensor window for {cow_id}: {e}")
            return pd.DataFrame()
    
    def prepare_detector_dataframe(
        self,
        raw_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert raw data to DataFrame for immediate detector.
        
        Expected columns: [timestamp, temperature, fxa, mya, rza]
        
        Args:
            raw_data: Raw sensor data from database
        
        Returns:
            DataFrame formatted for ImmediateAlertDetector
        """
        if raw_data.empty:
            return pd.DataFrame()
        
        try:
            # Select required columns
            required_cols = ['timestamp', 'temperature', 'fxa', 'mya', 'rza']
            
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in raw_data.columns]
            if missing_cols:
                logger.warning(f"Missing columns for detector: {missing_cols}")
                return pd.DataFrame()
            
            # Create detector dataframe with only required columns
            detector_df = raw_data[required_cols].copy()
            
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(detector_df['timestamp']):
                detector_df['timestamp'] = pd.to_datetime(detector_df['timestamp'])
            
            # Sort by timestamp
            detector_df = detector_df.sort_values('timestamp').reset_index(drop=True)
            
            return detector_df
            
        except Exception as e:
            logger.error(f"Error preparing detector dataframe: {e}")
            return pd.DataFrame()
    
    def prepare_estrus_dataframes(
        self,
        raw_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Separate into temperature_data and activity_data DataFrames.
        
        Args:
            raw_data: Raw sensor data from database
        
        Returns:
            Tuple of (temperature_data, activity_data) DataFrames
            - temperature_data: columns ['timestamp', 'temperature']
            - activity_data: columns ['timestamp', 'movement_intensity' or 'fxa']
        """
        if raw_data.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        try:
            # Temperature data
            temp_data = pd.DataFrame()
            if 'timestamp' in raw_data.columns and 'temperature' in raw_data.columns:
                temp_data = raw_data[['timestamp', 'temperature']].copy()
                temp_data = temp_data.dropna(subset=['temperature'])
            
            # Activity data (prefer movement_intensity, fallback to fxa)
            activity_data = pd.DataFrame()
            if 'timestamp' in raw_data.columns:
                if 'movement_intensity' in raw_data.columns:
                    activity_data = raw_data[['timestamp', 'movement_intensity']].copy()
                    activity_data = activity_data.dropna(subset=['movement_intensity'])
                elif 'fxa' in raw_data.columns:
                    activity_data = raw_data[['timestamp', 'fxa']].copy()
                    activity_data = activity_data.dropna(subset=['fxa'])
            
            # Ensure timestamps are datetime
            if not temp_data.empty:
                if not pd.api.types.is_datetime64_any_dtype(temp_data['timestamp']):
                    temp_data['timestamp'] = pd.to_datetime(temp_data['timestamp'])
                temp_data = temp_data.sort_values('timestamp').reset_index(drop=True)
            
            if not activity_data.empty:
                if not pd.api.types.is_datetime64_any_dtype(activity_data['timestamp']):
                    activity_data['timestamp'] = pd.to_datetime(activity_data['timestamp'])
                activity_data = activity_data.sort_values('timestamp').reset_index(drop=True)
            
            return temp_data, activity_data
            
        except Exception as e:
            logger.error(f"Error preparing estrus dataframes: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def run_immediate_detection(self) -> Dict[str, Any]:
        """
        Run immediate alert detection for all active cows.
        
        For each cow:
        1. Query 2-minute window of sensor data
        2. Prepare DataFrame with required columns
        3. Run ImmediateAlertDetector.detect_alerts()
        4. Convert Alert objects to dicts
        5. Store via AlertStateManager.create_alert()
        
        Returns:
            Dictionary with execution metrics
        """
        start_time = time.time()
        
        try:
            # Get active cows
            active_cows = self.get_active_cows()
            
            if not active_cows:
                logger.warning("No active cows found for immediate detection")
                return {
                    'success': True,
                    'cows_processed': 0,
                    'alerts_generated': 0,
                    'execution_time_seconds': 0.0
                }
            
            logger.info(f"Running immediate detection for {len(active_cows)} cows")
            
            total_alerts = 0
            cows_processed = 0
            cows_failed = 0
            
            for cow_id in active_cows:
                try:
                    # Query 2-minute window (immediate detection needs recent data)
                    raw_data = self.query_sensor_window(cow_id, hours=2/60.0)  # 2 minutes
                    
                    if raw_data.empty:
                        logger.debug(f"{cow_id}: No data in 2-minute window")
                        continue
                    
                    # Check for sufficient data
                    if len(raw_data) < 2:
                        logger.debug(f"{cow_id}: Insufficient data ({len(raw_data)} readings)")
                        continue
                    
                    # Prepare detector dataframe
                    detector_df = self.prepare_detector_dataframe(raw_data)
                    
                    if detector_df.empty:
                        logger.debug(f"{cow_id}: Failed to prepare detector dataframe")
                        continue
                    
                    # Get behavioral state if available
                    behavioral_state = None
                    if 'state' in raw_data.columns and not raw_data['state'].isna().all():
                        behavioral_state = raw_data['state'].iloc[-1]  # Most recent state
                    
                    # Get baseline temperature if available
                    baseline_temp = self.sensor_manager.calculate_baseline_temperature(cow_id)
                    
                    # Run immediate detector
                    alerts = self.immediate_detector.detect_alerts(
                        sensor_data=detector_df,
                        cow_id=cow_id,
                        behavioral_state=behavioral_state,
                        baseline_temp=baseline_temp
                    )
                    
                    # Store alerts
                    if alerts:
                        alert_types = []
                        for alert in alerts:
                            alert_dict = alert.to_dict()
                            success = self.alert_manager.create_alert(alert_dict)
                            if success:
                                total_alerts += 1
                                alert_types.append(alert.alert_type)
                        
                        if alert_types:
                            logger.info(f"{cow_id}: {len(alerts)} alerts detected "
                                      f"({', '.join(set(alert_types))})")
                    
                    cows_processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to run immediate detection for {cow_id}: {e}")
                    cows_failed += 1
                    continue
            
            execution_time = time.time() - start_time
            
            logger.info(f"Generated {total_alerts} alerts from immediate detection "
                       f"in {execution_time:.2f}s")
            
            return {
                'success': True,
                'cows_processed': cows_processed,
                'cows_failed': cows_failed,
                'alerts_generated': total_alerts,
                'execution_time_seconds': round(execution_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in run_immediate_detection: {e}", exc_info=True)
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': round(execution_time, 2)
            }
    
    def run_estrus_detection(self) -> Dict[str, Any]:
        """
        Run estrus detection for all active cows.
        
        For each cow:
        1. Query 48-hour window of sensor data
        2. Prepare separate temperature and activity DataFrames
        3. Run EstrusDetector.detect_estrus()
        4. Convert EstrusEvent objects to informational alerts
        5. Store via AlertStateManager.create_alert()
        
        Returns:
            Dictionary with execution metrics
        """
        start_time = time.time()
        
        try:
            # Get active cows
            active_cows = self.get_active_cows()
            
            if not active_cows:
                logger.warning("No active cows found for estrus detection")
                return {
                    'success': True,
                    'cows_processed': 0,
                    'events_detected': 0,
                    'execution_time_seconds': 0.0
                }
            
            logger.info(f"Running estrus detection for {len(active_cows)} cows")
            
            total_events = 0
            cows_processed = 0
            cows_failed = 0
            
            for cow_id in active_cows:
                try:
                    # Query 48-hour window (estrus detection needs historical context)
                    raw_data = self.query_sensor_window(cow_id, hours=48)
                    
                    if raw_data.empty:
                        logger.debug(f"{cow_id}: No data in 48-hour window")
                        continue
                    
                    # Check for sufficient data
                    if len(raw_data) < 100:  # Need reasonable amount of data
                        logger.debug(f"{cow_id}: Insufficient data ({len(raw_data)} readings)")
                        continue
                    
                    # Prepare estrus dataframes
                    temp_data, activity_data = self.prepare_estrus_dataframes(raw_data)
                    
                    if temp_data.empty or activity_data.empty:
                        logger.debug(f"{cow_id}: Missing temperature or activity data")
                        continue
                    
                    # Run estrus detector
                    events = self.estrus_detector.detect_estrus(
                        cow_id=cow_id,
                        temperature_data=temp_data,
                        activity_data=activity_data,
                        lookback_hours=48
                    )
                    
                    # Convert events to alerts and store
                    if events:
                        for event in events:
                            # Convert EstrusEvent to alert format
                            alert_dict = {
                                'alert_id': f"estrus_{cow_id}_{event.timestamp.strftime('%Y%m%d_%H%M%S')}",
                                'cow_id': cow_id,
                                'alert_type': 'estrus_event',
                                'severity': 'info',  # Informational only
                                'confidence': self._estrus_confidence_to_float(event.confidence),
                                'status': 'active',
                                'timestamp': event.timestamp.isoformat(),
                                'sensor_values': {
                                    'temperature_rise': event.temperature_rise,
                                    'activity_increase': event.activity_increase
                                },
                                'detection_details': {
                                    'duration_hours': event.duration_hours,
                                    'indicators': event.indicators,
                                    'message': event.message,
                                    'detection_type': 'estrus'
                                }
                            }
                            
                            success = self.alert_manager.create_alert(alert_dict)
                            if success:
                                total_events += 1
                        
                        logger.info(f"{cow_id}: {len(events)} estrus events detected")
                    
                    cows_processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to run estrus detection for {cow_id}: {e}")
                    cows_failed += 1
                    continue
            
            execution_time = time.time() - start_time
            
            logger.info(f"Detected {total_events} estrus events in {execution_time:.2f}s")
            
            return {
                'success': True,
                'cows_processed': cows_processed,
                'cows_failed': cows_failed,
                'events_detected': total_events,
                'execution_time_seconds': round(execution_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in run_estrus_detection: {e}", exc_info=True)
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': round(execution_time, 2)
            }
    
    def run_health_scoring(self) -> Dict[str, Any]:
        """
        Run health scoring for all active cows.
        
        For each cow:
        1. Query 24-hour window of sensor data
        2. Get active alerts from AlertStateManager
        3. Run SimpleHealthScorer.calculate_score()
        4. Store result via HealthScoreManager.save_health_score()
        
        Returns:
            Dictionary with execution metrics
        """
        start_time = time.time()
        
        try:
            # Get active cows
            active_cows = self.get_active_cows()
            
            if not active_cows:
                logger.warning("No active cows found for health scoring")
                return {
                    'success': True,
                    'cows_processed': 0,
                    'scores_calculated': 0,
                    'execution_time_seconds': 0.0
                }
            
            logger.info(f"Running health scoring for {len(active_cows)} cows")
            
            total_scores = 0
            cows_processed = 0
            cows_failed = 0
            
            for cow_id in active_cows:
                try:
                    # Query 24-hour window (health scoring standard window)
                    raw_data = self.query_sensor_window(cow_id, hours=24)
                    
                    if raw_data.empty:
                        logger.debug(f"{cow_id}: No data in 24-hour window")
                        continue
                    
                    # Check for sufficient data
                    if len(raw_data) < 5:
                        logger.debug(f"{cow_id}: Insufficient data ({len(raw_data)} readings)")
                        continue
                    
                    # Get active alerts for this cow
                    active_alerts = self.alert_manager.query_alerts(
                        cow_id=cow_id,
                        status='active'
                    )
                    
                    # Get baseline temperature
                    baseline_temp = self.sensor_manager.calculate_baseline_temperature(cow_id)
                    if baseline_temp is None:
                        baseline_temp = 38.5  # Use default if not available
                    
                    # Calculate health score
                    score_result = self.health_scorer.calculate_score(
                        cow_id=cow_id,
                        sensor_data=raw_data,
                        baseline_temp=baseline_temp,
                        active_alerts=active_alerts,
                        timestamp=datetime.now()
                    )
                    
                    # Store health score
                    success = self.health_score_manager.save_health_score(score_result)
                    
                    if success:
                        total_scores += 1
                        logger.info(f"{cow_id}: Health score={score_result['total_score']:.1f}, "
                                  f"category={score_result['health_category']}")
                    
                    cows_processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to run health scoring for {cow_id}: {e}")
                    cows_failed += 1
                    continue
            
            execution_time = time.time() - start_time
            
            logger.info(f"Calculated {total_scores} health scores in {execution_time:.2f}s")
            
            return {
                'success': True,
                'cows_processed': cows_processed,
                'cows_failed': cows_failed,
                'scores_calculated': total_scores,
                'execution_time_seconds': round(execution_time, 2)
            }
            
        except Exception as e:
            logger.error(f"Error in run_health_scoring: {e}", exc_info=True)
            execution_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': round(execution_time, 2)
            }
    
    def _estrus_confidence_to_float(self, confidence) -> float:
        """
        Convert EstrusConfidence enum to float value.
        
        Args:
            confidence: EstrusConfidence enum value
        
        Returns:
            Float confidence value (0.0-1.0)
        """
        confidence_map = {
            'low': 0.5,
            'medium': 0.75,
            'high': 0.9
        }
        
        # Handle both enum and string
        if hasattr(confidence, 'value'):
            confidence_str = confidence.value
        else:
            confidence_str = str(confidence).lower()
        
        return confidence_map.get(confidence_str, 0.5)
    
    def run_all_detectors(self) -> Dict[str, Any]:
        """
        Run all detection systems in sequence.
        
        Executes:
        1. Immediate alert detection (2-minute window)
        2. Estrus detection (48-hour window)
        3. Health scoring (24-hour window)
        
        Returns:
            Dictionary with combined execution metrics
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("Starting detector pipeline execution")
        logger.info("=" * 60)
        
        results = {}
        
        # Run immediate detection
        logger.info("\n--- Running Immediate Detection ---")
        results['immediate_detection'] = self.run_immediate_detection()
        
        # Run estrus detection
        logger.info("\n--- Running Estrus Detection ---")
        results['estrus_detection'] = self.run_estrus_detection()
        
        # Run health scoring
        logger.info("\n--- Running Health Scoring ---")
        results['health_scoring'] = self.run_health_scoring()
        
        total_execution_time = time.time() - start_time
        
        # Summary statistics
        total_alerts = results['immediate_detection'].get('alerts_generated', 0)
        total_events = results['estrus_detection'].get('events_detected', 0)
        total_scores = results['health_scoring'].get('scores_calculated', 0)
        
        logger.info("=" * 60)
        logger.info("Detector Pipeline Execution Complete")
        logger.info(f"Total execution time: {total_execution_time:.2f}s")
        logger.info(f"Immediate alerts: {total_alerts}")
        logger.info(f"Estrus events: {total_events}")
        logger.info(f"Health scores: {total_scores}")
        logger.info("=" * 60)
        
        results['summary'] = {
            'total_execution_time_seconds': round(total_execution_time, 2),
            'total_alerts_generated': total_alerts,
            'total_estrus_events': total_events,
            'total_health_scores': total_scores,
            'timestamp': datetime.now().isoformat()
        }
        
        return results
