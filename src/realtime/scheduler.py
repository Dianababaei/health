"""
APScheduler Integration Module for Detector Execution

This module orchestrates the execution of health detection algorithms at different
frequencies using APScheduler for the real-time MQTT service.

Scheduled Jobs:
- Immediate Detection (2-min): Fever and heat stress alerts
- Inactivity Detection (30-min): Prolonged stillness detection (requires 4+ hours data)
- Health Scoring (15-min): Comprehensive health score calculation
- Estrus Detection (6-hour): Reproductive cycle pattern analysis

Features:
- BackgroundScheduler for non-blocking operation
- Interval-based triggers with configurable frequencies
- Coalesced missed runs to prevent backlog buildup
- Max 1 instance per job to prevent overlapping executions
- Graceful shutdown with wait for running jobs
- Comprehensive error handling and execution tracking
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class DetectorScheduler:
    """
    Manages scheduled execution of detector pipeline methods.
    
    Orchestrates four detector jobs at different intervals:
    - Immediate alerts: Every 2 minutes
    - Inactivity detection: Every 30 minutes
    - Health scoring: Every 15 minutes
    - Estrus detection: Every 6 hours
    
    Features:
    - Non-blocking background execution
    - Configurable intervals from YAML config
    - Job execution statistics tracking
    - Graceful shutdown support
    - Error handling with continued operation
    
    Attributes:
        pipeline: DetectorPipeline instance with detector methods
        config: Configuration dictionary from realtime_config.yaml
        scheduler: APScheduler BackgroundScheduler instance
        job_stats: Dictionary tracking execution counts and failures
    """
    
    def __init__(self, pipeline: Any, config: Dict[str, Any]):
        """
        Initialize detector scheduler.
        
        Args:
            pipeline: DetectorPipeline instance with detector methods:
                - run_immediate_detection()
                - run_inactivity_detection()
                - run_health_scoring()
                - run_estrus_detection()
            config: Configuration dictionary with 'detector_schedules' section
                containing interval settings in minutes for each job type
        
        Raises:
            ValueError: If required methods are missing from pipeline
            KeyError: If required config sections are missing
        """
        self.pipeline = pipeline
        self.config = config
        self.scheduler: Optional[BackgroundScheduler] = None
        
        # Validate pipeline has required methods
        self._validate_pipeline()
        
        # Extract schedule configuration
        self.schedule_config = self._extract_schedule_config()
        
        # Job execution statistics
        self.job_stats = {
            'immediate_detection': {'success': 0, 'failure': 0, 'last_run': None, 'last_duration': None},
            'inactivity_detection': {'success': 0, 'failure': 0, 'last_run': None, 'last_duration': None},
            'health_scoring': {'success': 0, 'failure': 0, 'last_run': None, 'last_duration': None},
            'estrus_detection': {'success': 0, 'failure': 0, 'last_run': None, 'last_duration': None},
        }
        
        logger.info("DetectorScheduler initialized with schedule config: %s", self.schedule_config)
    
    def _validate_pipeline(self) -> None:
        """
        Validate that pipeline has all required detector methods.
        
        Raises:
            ValueError: If required methods are missing
        """
        required_methods = [
            'run_immediate_detection',
            'run_inactivity_detection',
            'run_health_scoring',
            'run_estrus_detection'
        ]
        
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(self.pipeline, method_name) or not callable(getattr(self.pipeline, method_name)):
                missing_methods.append(method_name)
        
        if missing_methods:
            raise ValueError(
                f"DetectorPipeline is missing required methods: {', '.join(missing_methods)}"
            )
        
        logger.debug("Pipeline validation successful - all required methods present")
    
    def _extract_schedule_config(self) -> Dict[str, int]:
        """
        Extract schedule intervals from configuration.
        
        Returns:
            Dictionary with interval minutes for each job type
        
        Raises:
            KeyError: If detector_schedules section is missing
        """
        if 'detector_schedules' not in self.config:
            logger.warning(
                "No 'detector_schedules' in config, using default intervals"
            )
            # Return default intervals as fallback
            return {
                'immediate_interval_minutes': 2,
                'inactivity_interval_minutes': 30,
                'health_scoring_interval_minutes': 15,
                'estrus_interval_minutes': 360,  # 6 hours
            }
        
        schedule_section = self.config['detector_schedules']
        
        # Extract intervals with defaults
        return {
            'immediate_interval_minutes': schedule_section.get('immediate_interval_minutes', 2),
            'inactivity_interval_minutes': schedule_section.get('inactivity_interval_minutes', 30),
            'health_scoring_interval_minutes': schedule_section.get('health_scoring_interval_minutes', 15),
            'estrus_interval_minutes': schedule_section.get('estrus_interval_minutes', 360),
        }
    
    def job_wrapper(self, detector_method: Callable, job_name: str) -> None:
        """
        Wrapper for job execution with logging, error handling, and timing.
        
        Wraps detector method calls with:
        - Start/completion logging
        - Execution time tracking
        - Exception handling (logs but doesn't propagate)
        - Statistics updates
        
        Args:
            detector_method: Callable detector method to execute
            job_name: Name of the job for logging and stats tracking
        """
        start_time = time.time()
        logger.info(f"Starting job: {job_name}")
        
        try:
            # Execute the detector method
            detector_method()
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update statistics
            self.job_stats[job_name]['success'] += 1
            self.job_stats[job_name]['last_run'] = datetime.now()
            self.job_stats[job_name]['last_duration'] = execution_time
            
            logger.info(
                f"Completed job: {job_name} in {execution_time:.2f}s "
                f"(total success: {self.job_stats[job_name]['success']})"
            )
            
        except Exception as e:
            # Calculate execution time even for failures
            execution_time = time.time() - start_time
            
            # Update failure statistics
            self.job_stats[job_name]['failure'] += 1
            self.job_stats[job_name]['last_run'] = datetime.now()
            self.job_stats[job_name]['last_duration'] = execution_time
            
            # Log error with full context but continue scheduler operation
            logger.error(
                f"Job {job_name} failed after {execution_time:.2f}s "
                f"(total failures: {self.job_stats[job_name]['failure']}): {e}",
                exc_info=True
            )
    
    def setup_jobs(self) -> None:
        """
        Configure all four detection jobs with interval triggers.
        
        Sets up scheduled jobs with APScheduler:
        - Immediate detection: 2-minute intervals (configurable)
        - Inactivity detection: 30-minute intervals (configurable)
        - Health scoring: 15-minute intervals (configurable)
        - Estrus detection: 6-hour intervals (configurable)
        
        All jobs configured with:
        - coalesce=True: Skip missed runs to prevent backlog
        - max_instances=1: Prevent overlapping executions
        - Job-specific IDs for management
        """
        if self.scheduler is None:
            raise RuntimeError("Scheduler not initialized. Call start() first.")
        
        logger.info("Setting up detector jobs...")
        
        # Job 1: Immediate Detection (fever, heat stress)
        immediate_interval = self.schedule_config['immediate_interval_minutes']
        self.scheduler.add_job(
            func=lambda: self.job_wrapper(
                self.pipeline.run_immediate_detection,
                'immediate_detection'
            ),
            trigger=IntervalTrigger(minutes=immediate_interval),
            id='immediate_detection',
            name='Immediate Alert Detection (Fever, Heat Stress)',
            coalesce=True,
            max_instances=1,
            replace_existing=True
        )
        logger.info(f"Scheduled immediate_detection job: every {immediate_interval} minutes")
        
        # Job 2: Inactivity Detection (requires 4+ hour window)
        inactivity_interval = self.schedule_config['inactivity_interval_minutes']
        self.scheduler.add_job(
            func=lambda: self.job_wrapper(
                self.pipeline.run_inactivity_detection,
                'inactivity_detection'
            ),
            trigger=IntervalTrigger(minutes=inactivity_interval),
            id='inactivity_detection',
            name='Inactivity Detection (Prolonged Stillness)',
            coalesce=True,
            max_instances=1,
            replace_existing=True
        )
        logger.info(f"Scheduled inactivity_detection job: every {inactivity_interval} minutes")
        
        # Job 3: Health Scoring
        health_scoring_interval = self.schedule_config['health_scoring_interval_minutes']
        self.scheduler.add_job(
            func=lambda: self.job_wrapper(
                self.pipeline.run_health_scoring,
                'health_scoring'
            ),
            trigger=IntervalTrigger(minutes=health_scoring_interval),
            id='health_scoring',
            name='Health Score Calculation',
            coalesce=True,
            max_instances=1,
            replace_existing=True
        )
        logger.info(f"Scheduled health_scoring job: every {health_scoring_interval} minutes")
        
        # Job 4: Estrus Detection (pattern analysis)
        estrus_interval = self.schedule_config['estrus_interval_minutes']
        self.scheduler.add_job(
            func=lambda: self.job_wrapper(
                self.pipeline.run_estrus_detection,
                'estrus_detection'
            ),
            trigger=IntervalTrigger(minutes=estrus_interval),
            id='estrus_detection',
            name='Estrus Detection (Reproductive Cycle)',
            coalesce=True,
            max_instances=1,
            replace_existing=True
        )
        logger.info(f"Scheduled estrus_detection job: every {estrus_interval} minutes")
        
        logger.info("All detector jobs configured successfully")
    
    def start(self) -> None:
        """
        Initialize and start the BackgroundScheduler.
        
        Creates BackgroundScheduler instance, sets up all jobs, and starts
        the scheduler for non-blocking background execution.
        
        Raises:
            RuntimeError: If scheduler is already running
        """
        if self.scheduler is not None and self.scheduler.running:
            raise RuntimeError("Scheduler is already running")
        
        logger.info("Initializing BackgroundScheduler...")
        
        # Create BackgroundScheduler instance
        self.scheduler = BackgroundScheduler(
            timezone='UTC',
            job_defaults={
                'coalesce': True,
                'max_instances': 1
            }
        )
        
        # Set up all detector jobs
        self.setup_jobs()
        
        # Start the scheduler
        self.scheduler.start()
        
        logger.info("DetectorScheduler started successfully - all jobs running in background")
        
        # Log initial job status
        self._log_job_status()
    
    def stop(self, wait: bool = True) -> None:
        """
        Gracefully shutdown the scheduler.
        
        Stops the scheduler and optionally waits for running jobs to complete.
        
        Args:
            wait: If True, waits for currently executing jobs to finish.
                 If False, jobs are cancelled immediately.
                 Default is True for graceful shutdown.
        """
        if self.scheduler is None:
            logger.warning("Scheduler not initialized, nothing to stop")
            return
        
        if not self.scheduler.running:
            logger.warning("Scheduler not running, nothing to stop")
            return
        
        logger.info(f"Stopping DetectorScheduler (wait={wait})...")
        
        # Log final statistics before shutdown
        self._log_final_statistics()
        
        # Shutdown scheduler
        self.scheduler.shutdown(wait=wait)
        
        logger.info("DetectorScheduler stopped successfully")
    
    def _log_job_status(self) -> None:
        """Log current status of all scheduled jobs."""
        if self.scheduler is None or not self.scheduler.running:
            return
        
        jobs = self.scheduler.get_jobs()
        logger.info(f"Active jobs: {len(jobs)}")
        for job in jobs:
            logger.info(f"  - {job.name} (ID: {job.id}, Next run: {job.next_run_time})")
    
    def _log_final_statistics(self) -> None:
        """Log final execution statistics for all jobs."""
        logger.info("=== Final Job Execution Statistics ===")
        for job_name, stats in self.job_stats.items():
            total_runs = stats['success'] + stats['failure']
            success_rate = (stats['success'] / total_runs * 100) if total_runs > 0 else 0
            
            logger.info(
                f"{job_name}: "
                f"Total: {total_runs}, "
                f"Success: {stats['success']}, "
                f"Failure: {stats['failure']}, "
                f"Success Rate: {success_rate:.1f}%, "
                f"Last Run: {stats['last_run']}, "
                f"Last Duration: {stats['last_duration']:.2f}s" if stats['last_duration'] else "N/A"
            )
    
    def get_job_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get current job execution statistics.
        
        Returns:
            Dictionary with statistics for each job including:
            - success: Number of successful executions
            - failure: Number of failed executions
            - last_run: Timestamp of last execution
            - last_duration: Duration of last execution in seconds
        """
        return self.job_stats.copy()
    
    def get_next_run_times(self) -> Dict[str, Optional[datetime]]:
        """
        Get next scheduled run times for all jobs.
        
        Returns:
            Dictionary mapping job names to their next scheduled run time
        """
        if self.scheduler is None or not self.scheduler.running:
            return {}
        
        next_runs = {}
        for job in self.scheduler.get_jobs():
            next_runs[job.id] = job.next_run_time
        
        return next_runs
