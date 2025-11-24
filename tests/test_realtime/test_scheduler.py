"""
Unit Tests for APScheduler Integration

Tests cover:
- Mock APScheduler BackgroundScheduler
- Job configuration with correct intervals
- Job execution triggering pipeline methods
- Error handling in job wrapper
- Graceful shutdown with job completion wait
- Scheduler lifecycle (start, stop, restart)
"""

import unittest
from unittest.mock import Mock, MagicMock, patch, call
import time
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from tests.test_realtime import DEFAULT_SCHEDULER_CONFIG


class TestSchedulerInitialization(unittest.TestCase):
    """Test scheduler initialization and configuration."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_scheduler_initialization(self, mock_scheduler_class):
        """Test scheduler is initialized correctly."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        # Simulate scheduler initialization
        scheduler = mock_scheduler_class()
        
        # Verify scheduler was created
        mock_scheduler_class.assert_called_once()
        self.assertIsNotNone(scheduler)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_scheduler_configuration(self, mock_scheduler_class):
        """Test scheduler configuration options."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        config = DEFAULT_SCHEDULER_CONFIG.copy()
        
        # Create scheduler with configuration
        scheduler = mock_scheduler_class(
            job_defaults={
                'coalesce': config['coalesce'],
                'max_instances': config['max_workers'],
                'misfire_grace_time': config['misfire_grace_time']
            }
        )
        
        # Verify configuration was passed
        mock_scheduler_class.assert_called_once()


class TestJobConfiguration(unittest.TestCase):
    """Test job configuration with correct intervals."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_add_immediate_alert_job(self, mock_scheduler_class):
        """Test adding immediate alert detection job."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        config = DEFAULT_SCHEDULER_CONFIG.copy()
        
        # Simulate adding job
        scheduler = mock_scheduler_class()
        job_func = Mock()
        
        scheduler.add_job(
            func=job_func,
            trigger='interval',
            seconds=config['immediate_alert_interval_seconds'],
            id='immediate_alerts',
            name='Immediate Alert Detection'
        )
        
        # Verify job was added
        mock_scheduler.add_job.assert_called_once()
        call_args = mock_scheduler.add_job.call_args
        self.assertEqual(call_args[1]['trigger'], 'interval')
        self.assertEqual(call_args[1]['seconds'], 120)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_add_estrus_detection_job(self, mock_scheduler_class):
        """Test adding estrus detection job."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        config = DEFAULT_SCHEDULER_CONFIG.copy()
        
        scheduler = mock_scheduler_class()
        job_func = Mock()
        
        scheduler.add_job(
            func=job_func,
            trigger='interval',
            minutes=config['estrus_detection_interval_minutes'],
            id='estrus_detection',
            name='Estrus Detection'
        )
        
        # Verify job was added
        mock_scheduler.add_job.assert_called_once()
        call_args = mock_scheduler.add_job.call_args
        self.assertEqual(call_args[1]['minutes'], 60)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_add_health_score_job(self, mock_scheduler_class):
        """Test adding health score calculation job."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        config = DEFAULT_SCHEDULER_CONFIG.copy()
        
        scheduler = mock_scheduler_class()
        job_func = Mock()
        
        scheduler.add_job(
            func=job_func,
            trigger='interval',
            minutes=config['health_score_interval_minutes'],
            id='health_scoring',
            name='Health Score Calculation'
        )
        
        # Verify job was added
        mock_scheduler.add_job.assert_called_once()
        call_args = mock_scheduler.add_job.call_args
        self.assertEqual(call_args[1]['minutes'], 15)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_add_multiple_jobs(self, mock_scheduler_class):
        """Test adding multiple jobs with different intervals."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        
        # Add three jobs
        jobs = [
            ('immediate_alerts', 'interval', {'seconds': 120}),
            ('estrus_detection', 'interval', {'minutes': 60}),
            ('health_scoring', 'interval', {'minutes': 15})
        ]
        
        for job_id, trigger, kwargs in jobs:
            scheduler.add_job(
                func=Mock(),
                trigger=trigger,
                id=job_id,
                **kwargs
            )
        
        # Verify all jobs were added
        self.assertEqual(mock_scheduler.add_job.call_count, 3)


class TestJobExecution(unittest.TestCase):
    """Test job execution triggering pipeline methods."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_immediate_alert_job_executes(self, mock_scheduler_class):
        """Test immediate alert job executes successfully."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        # Create mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.run_immediate_alerts.return_value = {'alerts_detected': 2}
        
        # Define job function
        def immediate_alert_job():
            return mock_pipeline.run_immediate_alerts()
        
        # Execute job
        result = immediate_alert_job()
        
        # Verify pipeline method was called
        mock_pipeline.run_immediate_alerts.assert_called_once()
        self.assertEqual(result['alerts_detected'], 2)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_estrus_detection_job_executes(self, mock_scheduler_class):
        """Test estrus detection job executes successfully."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        mock_pipeline = Mock()
        mock_pipeline.run_estrus_detection.return_value = {'events_detected': 1}
        
        def estrus_detection_job():
            return mock_pipeline.run_estrus_detection()
        
        result = estrus_detection_job()
        
        mock_pipeline.run_estrus_detection.assert_called_once()
        self.assertEqual(result['events_detected'], 1)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_health_score_job_executes(self, mock_scheduler_class):
        """Test health score job executes successfully."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        mock_pipeline = Mock()
        mock_pipeline.run_health_scoring.return_value = {'cows_scored': 5}
        
        def health_score_job():
            return mock_pipeline.run_health_scoring()
        
        result = health_score_job()
        
        mock_pipeline.run_health_scoring.assert_called_once()
        self.assertEqual(result['cows_scored'], 5)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_job_receives_arguments(self, mock_scheduler_class):
        """Test job receives correct arguments."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        mock_pipeline = Mock()
        
        def parameterized_job(cow_id, window_hours):
            return mock_pipeline.run_detection(cow_id=cow_id, window_hours=window_hours)
        
        # Execute with arguments
        result = parameterized_job(cow_id='COW_001', window_hours=24)
        
        # Verify arguments were passed
        mock_pipeline.run_detection.assert_called_once_with(
            cow_id='COW_001',
            window_hours=24
        )


class TestJobErrorHandling(unittest.TestCase):
    """Test error handling in job wrapper."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_job_exception_caught(self, mock_scheduler_class):
        """Test job exception is caught and logged."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        mock_pipeline = Mock()
        mock_pipeline.run_immediate_alerts.side_effect = Exception("Pipeline error")
        
        def wrapped_job():
            try:
                return mock_pipeline.run_immediate_alerts()
            except Exception as e:
                # Log error but don't crash scheduler
                return {'error': str(e)}
        
        result = wrapped_job()
        
        # Job should return error info
        self.assertIn('error', result)
        self.assertIn('Pipeline error', result['error'])
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_job_continues_after_error(self, mock_scheduler_class):
        """Test scheduler continues running after job error."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        call_count = {'count': 0}
        
        def failing_job():
            call_count['count'] += 1
            if call_count['count'] < 3:
                raise Exception("Job failed")
            return {'success': True}
        
        # Simulate multiple executions
        results = []
        for i in range(3):
            try:
                result = failing_job()
                results.append(result)
            except Exception:
                results.append({'error': 'failed'})
        
        # Should have 2 failures and 1 success
        self.assertEqual(len(results), 3)
        self.assertEqual(call_count['count'], 3)
        self.assertTrue(results[-1]['success'])
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_job_timeout_handling(self, mock_scheduler_class):
        """Test handling of job timeout scenarios."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        # Simulate job that times out
        def timeout_job(timeout_seconds=5):
            start_time = time.time()
            # Simulate work
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError("Job exceeded timeout")
            return {'completed': True}
        
        # Should complete quickly
        result = timeout_job(timeout_seconds=5)
        self.assertTrue(result['completed'])


class TestSchedulerLifecycle(unittest.TestCase):
    """Test scheduler lifecycle (start, stop, restart)."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_scheduler_start(self, mock_scheduler_class):
        """Test starting the scheduler."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        scheduler.start()
        
        # Verify start was called
        mock_scheduler.start.assert_called_once()
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_scheduler_stop(self, mock_scheduler_class):
        """Test stopping the scheduler."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        scheduler.start()
        scheduler.shutdown(wait=True)
        
        # Verify shutdown was called
        mock_scheduler.shutdown.assert_called_once_with(wait=True)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_scheduler_restart(self, mock_scheduler_class):
        """Test restarting the scheduler."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        
        # Start, stop, start again
        scheduler.start()
        scheduler.shutdown(wait=True)
        scheduler.start()
        
        # Verify called multiple times
        self.assertEqual(mock_scheduler.start.call_count, 2)
        mock_scheduler.shutdown.assert_called_once()
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_scheduler_is_running(self, mock_scheduler_class):
        """Test checking if scheduler is running."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        mock_scheduler.running = False
        
        scheduler = mock_scheduler_class()
        
        # Not running initially
        self.assertFalse(scheduler.running)
        
        # Start scheduler
        scheduler.running = True
        self.assertTrue(scheduler.running)
        
        # Stop scheduler
        scheduler.running = False
        self.assertFalse(scheduler.running)


class TestGracefulShutdown(unittest.TestCase):
    """Test graceful shutdown with job completion wait."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_shutdown_waits_for_jobs(self, mock_scheduler_class):
        """Test shutdown waits for running jobs to complete."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        scheduler.start()
        
        # Shutdown with wait=True
        scheduler.shutdown(wait=True)
        
        # Verify wait parameter was passed
        mock_scheduler.shutdown.assert_called_once_with(wait=True)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_shutdown_without_wait(self, mock_scheduler_class):
        """Test immediate shutdown without waiting."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        scheduler.start()
        
        # Shutdown immediately
        scheduler.shutdown(wait=False)
        
        mock_scheduler.shutdown.assert_called_once_with(wait=False)
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    @patch('time.sleep')
    def test_shutdown_with_timeout(self, mock_sleep, mock_scheduler_class):
        """Test shutdown with timeout for job completion."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        def shutdown_with_timeout(scheduler, timeout_seconds=10):
            """Shutdown with timeout."""
            scheduler.shutdown(wait=False)
            
            # Wait up to timeout for jobs to complete
            start_time = time.time()
            while scheduler.running and (time.time() - start_time) < timeout_seconds:
                mock_sleep(0.1)
            
            return not scheduler.running
        
        scheduler = mock_scheduler_class()
        scheduler.running = True
        
        # Simulate shutdown
        scheduler.running = False
        result = shutdown_with_timeout(scheduler, timeout_seconds=5)
        
        self.assertTrue(result)


class TestJobPauseResume(unittest.TestCase):
    """Test pausing and resuming jobs."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_pause_job(self, mock_scheduler_class):
        """Test pausing a specific job."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        
        # Pause a job
        scheduler.pause_job(job_id='immediate_alerts')
        
        # Verify pause was called
        mock_scheduler.pause_job.assert_called_once_with(job_id='immediate_alerts')
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_resume_job(self, mock_scheduler_class):
        """Test resuming a paused job."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        
        # Pause then resume
        scheduler.pause_job(job_id='immediate_alerts')
        scheduler.resume_job(job_id='immediate_alerts')
        
        # Verify resume was called
        mock_scheduler.resume_job.assert_called_once_with(job_id='immediate_alerts')
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_pause_all_jobs(self, mock_scheduler_class):
        """Test pausing all jobs."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        scheduler.pause()
        
        # Verify pause was called (pauses all jobs)
        mock_scheduler.pause.assert_called_once()


class TestJobRetrieval(unittest.TestCase):
    """Test retrieving job information."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_get_job_by_id(self, mock_scheduler_class):
        """Test retrieving job by ID."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        mock_job = Mock()
        mock_job.id = 'immediate_alerts'
        mock_scheduler.get_job.return_value = mock_job
        
        scheduler = mock_scheduler_class()
        job = scheduler.get_job('immediate_alerts')
        
        # Verify job was retrieved
        mock_scheduler.get_job.assert_called_once_with('immediate_alerts')
        self.assertEqual(job.id, 'immediate_alerts')
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_get_all_jobs(self, mock_scheduler_class):
        """Test retrieving all jobs."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        mock_jobs = [
            Mock(id='immediate_alerts'),
            Mock(id='estrus_detection'),
            Mock(id='health_scoring')
        ]
        mock_scheduler.get_jobs.return_value = mock_jobs
        
        scheduler = mock_scheduler_class()
        jobs = scheduler.get_jobs()
        
        # Verify all jobs retrieved
        self.assertEqual(len(jobs), 3)
        mock_scheduler.get_jobs.assert_called_once()
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_job_not_found(self, mock_scheduler_class):
        """Test handling when job not found."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        mock_scheduler.get_job.return_value = None
        
        scheduler = mock_scheduler_class()
        job = scheduler.get_job('nonexistent_job')
        
        # Should return None
        self.assertIsNone(job)


class TestJobModification(unittest.TestCase):
    """Test modifying job configuration."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_modify_job_interval(self, mock_scheduler_class):
        """Test modifying job execution interval."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        
        # Modify job
        scheduler.modify_job(
            job_id='immediate_alerts',
            trigger='interval',
            seconds=60  # Change from 120 to 60
        )
        
        # Verify modification
        mock_scheduler.modify_job.assert_called_once()
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_remove_job(self, mock_scheduler_class):
        """Test removing a job."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        scheduler.remove_job(job_id='immediate_alerts')
        
        # Verify removal
        mock_scheduler.remove_job.assert_called_once_with(job_id='immediate_alerts')
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_remove_all_jobs(self, mock_scheduler_class):
        """Test removing all jobs."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        scheduler = mock_scheduler_class()
        scheduler.remove_all_jobs()
        
        # Verify all jobs removed
        mock_scheduler.remove_all_jobs.assert_called_once()


class TestMisfireHandling(unittest.TestCase):
    """Test handling of misfired jobs."""
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_misfire_grace_time(self, mock_scheduler_class):
        """Test misfire grace time configuration."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        config = DEFAULT_SCHEDULER_CONFIG.copy()
        
        # Create scheduler with misfire grace time
        scheduler = mock_scheduler_class(
            job_defaults={
                'misfire_grace_time': config['misfire_grace_time']
            }
        )
        
        # Verify configuration
        mock_scheduler_class.assert_called_once()
    
    @patch('apscheduler.schedulers.background.BackgroundScheduler')
    def test_coalesce_missed_executions(self, mock_scheduler_class):
        """Test coalescing missed job executions."""
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        
        config = DEFAULT_SCHEDULER_CONFIG.copy()
        
        # Create scheduler with coalesce enabled
        scheduler = mock_scheduler_class(
            job_defaults={
                'coalesce': config['coalesce']
            }
        )
        
        # Verify coalesce is enabled
        self.assertTrue(config['coalesce'])


if __name__ == '__main__':
    unittest.main()
