"""
Event Aggregator Module
========================
Merge and normalize events from multiple sources (alerts, behavioral states, 
physiological metrics, sensor malfunctions) into a unified timeline format.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EventAggregator:
    """
    Aggregates events from multiple sources into a unified format for timeline visualization.
    
    Event categories:
    - alerts: Critical health alerts, warnings, and notifications
    - behavioral: Behavioral state transitions
    - health: Temperature anomalies and physiological events
    - sensor: Sensor malfunctions and data quality issues
    """
    
    # Event category to Y-axis position mapping
    EVENT_CATEGORIES = {
        'alerts_critical': 4,
        'alerts_warning': 3,
        'behavioral': 2,
        'health': 1,
        'sensor': 0,
    }
    
    # Event category labels
    CATEGORY_LABELS = {
        'alerts_critical': 'Critical Alerts',
        'alerts_warning': 'Warnings',
        'behavioral': 'Behavioral Changes',
        'health': 'Health Events',
        'sensor': 'Sensor Issues',
    }
    
    # Color mapping by category
    CATEGORY_COLORS = {
        'alerts_critical': '#E74C3C',  # Red
        'alerts_warning': '#F39C12',   # Orange/Yellow
        'behavioral': '#3498DB',       # Blue
        'health': '#9B59B6',          # Purple
        'sensor': '#95A5A6',          # Gray
    }
    
    # Marker symbols by category (Plotly marker symbols)
    CATEGORY_MARKERS = {
        'alerts_critical': 'x',           # X for critical
        'alerts_warning': 'triangle-up',  # Triangle for warnings
        'behavioral': 'circle',           # Circle for behavioral
        'health': 'diamond',              # Diamond for health
        'sensor': 'square',               # Square for sensor
    }
    
    def __init__(self):
        """Initialize event aggregator."""
        self.events = []
    
    def process_alerts(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process alerts from alerts table into unified event format.
        
        Args:
            df: DataFrame from query_alerts
            
        Returns:
            List of normalized event dictionaries
        """
        if df.empty:
            return []
        
        events = []
        
        for _, row in df.iterrows():
            # Determine category based on severity
            if row['severity'] == 'critical':
                category = 'alerts_critical'
            elif row['severity'] == 'warning':
                category = 'alerts_warning'
            else:
                category = 'alerts_warning'  # Info alerts as warnings
            
            # Build event description
            description = row['title']
            
            # Extract key details
            details = row.get('details', {})
            if isinstance(details, dict):
                detail_items = []
                for key, value in details.items():
                    if key not in ['timestamp', 'cow_id']:
                        detail_items.append(f"{key}: {value}")
                details_str = ", ".join(detail_items[:3])  # Limit to 3 items
            else:
                details_str = str(details)
            
            # Get sensor values
            sensor_values = row.get('sensor_values', {})
            if isinstance(sensor_values, dict):
                sensor_str = ", ".join([f"{k}: {v}" for k, v in list(sensor_values.items())[:3]])
            else:
                sensor_str = ""
            
            event = {
                'timestamp': row['timestamp'],
                'cow_id': row['cow_id'],
                'category': category,
                'event_type': row['alert_type'],
                'severity': row['severity'],
                'title': description,
                'description': details_str,
                'sensor_values': sensor_str,
                'status': row.get('status', 'active'),
                'source': 'alerts',
                'event_id': row.get('alert_id', f"alert_{row['timestamp']}"),
            }
            
            events.append(event)
        
        logger.info(f"Processed {len(events)} alert events")
        return events
    
    def process_behavioral_transitions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process behavioral state transitions into unified event format.
        
        Args:
            df: DataFrame from query_behavioral_transitions
            
        Returns:
            List of normalized event dictionaries
        """
        if df.empty:
            return []
        
        events = []
        
        for _, row in df.iterrows():
            # Build description
            prev_state = row.get('prev_state', 'unknown')
            current_state = row.get('state', 'unknown')
            confidence = row.get('confidence', 0)
            
            title = f"{prev_state.title()} → {current_state.title()}"
            description = f"State transition from {prev_state} to {current_state}"
            
            event = {
                'timestamp': row['timestamp'],
                'cow_id': row['cow_id'],
                'category': 'behavioral',
                'event_type': 'state_transition',
                'severity': 'info',
                'title': title,
                'description': description,
                'sensor_values': f"confidence: {confidence:.2%}",
                'status': 'active',
                'source': 'behavioral_states',
                'event_id': f"behavior_{row['cow_id']}_{row['timestamp']}",
            }
            
            events.append(event)
        
        logger.info(f"Processed {len(events)} behavioral transition events")
        return events
    
    def process_temperature_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process temperature anomalies into unified event format.
        
        Args:
            df: DataFrame from query_temperature_anomalies
            
        Returns:
            List of normalized event dictionaries
        """
        if df.empty:
            return []
        
        events = []
        
        for _, row in df.iterrows():
            temp = row.get('current_temp', 0)
            baseline = row.get('baseline_temp', 0)
            deviation = row.get('temp_deviation', 0)
            anomaly_score = row.get('temp_anomaly_score', 0)
            
            # Determine severity based on anomaly score
            if anomaly_score >= 0.9:
                severity = 'critical'
            elif anomaly_score >= 0.7:
                severity = 'warning'
            else:
                severity = 'info'
            
            title = f"Temperature Anomaly Detected"
            description = f"Temperature: {temp:.1f}°C (baseline: {baseline:.1f}°C, deviation: {deviation:+.1f}°C)"
            
            event = {
                'timestamp': row['timestamp'],
                'cow_id': row['cow_id'],
                'category': 'health',
                'event_type': 'temperature_anomaly',
                'severity': severity,
                'title': title,
                'description': description,
                'sensor_values': f"anomaly_score: {anomaly_score:.2%}",
                'status': 'active',
                'source': 'physiological_metrics',
                'event_id': f"temp_{row['cow_id']}_{row['timestamp']}",
            }
            
            events.append(event)
        
        logger.info(f"Processed {len(events)} temperature anomaly events")
        return events
    
    def process_sensor_malfunctions(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Process sensor malfunctions into unified event format.
        
        Args:
            df: DataFrame from query_sensor_malfunctions
            
        Returns:
            List of normalized event dictionaries
        """
        if df.empty:
            return []
        
        events = []
        
        for _, row in df.iterrows():
            data_quality = row.get('data_quality', 'unknown')
            sensor_id = row.get('sensor_id', 'unknown')
            
            # Determine severity based on data quality
            if data_quality in ['poor', 'sensor_error']:
                severity = 'critical'
            else:
                severity = 'warning'
            
            title = f"Sensor Issue Detected"
            description = f"Sensor {sensor_id}: {data_quality} data quality"
            
            event = {
                'timestamp': row['timestamp'],
                'cow_id': row['cow_id'],
                'category': 'sensor',
                'event_type': 'sensor_malfunction',
                'severity': severity,
                'title': title,
                'description': description,
                'sensor_values': f"sensor_id: {sensor_id}, quality: {data_quality}",
                'status': 'active',
                'source': 'raw_sensor_readings',
                'event_id': f"sensor_{row['cow_id']}_{row['timestamp']}",
            }
            
            events.append(event)
        
        logger.info(f"Processed {len(events)} sensor malfunction events")
        return events
    
    def aggregate_events(self, event_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Aggregate all events from multiple sources into a unified DataFrame.
        
        Args:
            event_data: Dictionary with keys 'alerts', 'behavioral', 'temperature', 'sensor'
                        and DataFrame values from event queries
            
        Returns:
            Unified DataFrame with all events sorted chronologically
        """
        all_events = []
        
        # Process each event type
        if 'alerts' in event_data and not event_data['alerts'].empty:
            all_events.extend(self.process_alerts(event_data['alerts']))
        
        if 'behavioral' in event_data and not event_data['behavioral'].empty:
            all_events.extend(self.process_behavioral_transitions(event_data['behavioral']))
        
        if 'temperature' in event_data and not event_data['temperature'].empty:
            all_events.extend(self.process_temperature_anomalies(event_data['temperature']))
        
        if 'sensor' in event_data and not event_data['sensor'].empty:
            all_events.extend(self.process_sensor_malfunctions(event_data['sensor']))
        
        # Convert to DataFrame
        if not all_events:
            logger.warning("No events found to aggregate")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_events)
        
        # Sort by timestamp
        df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
        
        # Add display properties
        df['category_label'] = df['category'].map(self.CATEGORY_LABELS)
        df['color'] = df['category'].map(self.CATEGORY_COLORS)
        df['marker'] = df['category'].map(self.CATEGORY_MARKERS)
        df['y_position'] = df['category'].map(self.EVENT_CATEGORIES)
        
        logger.info(f"Aggregated {len(df)} total events from {len(event_data)} sources")
        
        return df
    
    def filter_events(
        self,
        df: pd.DataFrame,
        categories: Optional[List[str]] = None,
        severities: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Filter aggregated events by criteria.
        
        Args:
            df: Aggregated events DataFrame
            categories: Filter by categories (e.g., ['alerts_critical', 'behavioral'])
            severities: Filter by severities (e.g., ['critical', 'warning'])
            start_time: Filter events after this time
            end_time: Filter events before this time
            
        Returns:
            Filtered DataFrame
        """
        if df.empty:
            return df
        
        filtered_df = df.copy()
        
        # Filter by categories
        if categories:
            filtered_df = filtered_df[filtered_df['category'].isin(categories)]
        
        # Filter by severities
        if severities:
            filtered_df = filtered_df[filtered_df['severity'].isin(severities)]
        
        # Filter by time range
        if start_time:
            filtered_df = filtered_df[filtered_df['timestamp'] >= start_time]
        
        if end_time:
            filtered_df = filtered_df[filtered_df['timestamp'] <= end_time]
        
        logger.info(f"Filtered events: {len(df)} → {len(filtered_df)}")
        
        return filtered_df
    
    def get_event_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics for aggregated events.
        
        Args:
            df: Aggregated events DataFrame
            
        Returns:
            Dictionary with event statistics
        """
        if df.empty:
            return {
                'total_events': 0,
                'by_category': {},
                'by_severity': {},
                'by_type': {},
                'time_range': None,
            }
        
        stats = {
            'total_events': len(df),
            'by_category': df['category'].value_counts().to_dict(),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_type': df['event_type'].value_counts().to_dict(),
            'time_range': (df['timestamp'].min(), df['timestamp'].max()),
        }
        
        return stats
