"""
Alert Frequency Score Component

Calculates health score based on alert metrics:
- Number of active alerts
- Alert severity weighting
- Recent alert resolution rate (improvement trend)

Score Range: 0-25 points
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base_component import BaseScoreComponent, ComponentScore


class AlertScoreComponent(BaseScoreComponent):
    """
    Alert frequency scoring component.
    
    Placeholder Formula:
        score = 25 - (critical_alerts * 10) - (warning_alerts * 5)
    
    Where:
        - critical_alerts: Number of active critical alerts
        - warning_alerts: Number of active warning alerts
    
    Future Integration Points:
        - Alert trend analysis (worsening vs improving)
        - Alert type weighting (some types more serious)
        - Time-weighted alert scoring (recent alerts more important)
        - False positive rate adjustment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize alert component.
        
        Args:
            config: Configuration dict with keys:
                - critical_alert_penalty: Points per critical alert (default 10.0)
                - warning_alert_penalty: Points per warning alert (default 5.0)
                - resolution_bonus_enabled: Add bonus for resolved alerts (default True)
                - resolution_bonus_per_alert: Bonus per resolved alert (default 2.0)
        """
        super().__init__(config)
        
        # Load configuration with defaults
        self.critical_penalty = self.config.get('critical_alert_penalty', 10.0)
        self.warning_penalty = self.config.get('warning_alert_penalty', 5.0)
        self.info_penalty = self.config.get('info_alert_penalty', 1.0)
        self.max_critical_penalty = self.config.get('max_critical_penalty', 20.0)
        self.max_warning_penalty = self.config.get('max_warning_penalty', 15.0)
        self.resolution_bonus_enabled = self.config.get('resolution_bonus_enabled', True)
        self.resolution_bonus = self.config.get('resolution_bonus_per_alert', 2.0)
    
    def get_required_columns(self) -> list:
        """Get required DataFrame columns."""
        # Alert data is typically passed as a list, not DataFrame
        return []
    
    def calculate_score(
        self,
        cow_id: str,
        data: pd.DataFrame,
        active_alerts: Optional[List[Dict]] = None,
        resolved_alerts: Optional[List[Dict]] = None,
        alert_history: Optional[List[Dict]] = None,
        **kwargs
    ) -> ComponentScore:
        """
        Calculate alert frequency score.
        
        Args:
            cow_id: Animal identifier
            data: DataFrame (not used for alert component, kept for consistency)
            active_alerts: List of active alert dicts with 'severity' key
            resolved_alerts: List of recently resolved alert dicts
            alert_history: Full alert history for trend analysis
            **kwargs: Additional parameters
        
        Returns:
            ComponentScore with alert frequency score (0-25 points)
        """
        details = {}
        score = 25.0  # Start with perfect score
        confidence = 0.9  # High confidence - alerts are explicit data
        warnings = []
        
        # Count alerts by severity
        critical_count = 0
        warning_count = 0
        info_count = 0
        
        if active_alerts:
            for alert in active_alerts:
                severity = alert.get('severity', 'warning').lower()
                if severity == 'critical':
                    critical_count += 1
                elif severity == 'warning':
                    warning_count += 1
                elif severity == 'info':
                    info_count += 1
        
        details['critical_alerts'] = critical_count
        details['warning_alerts'] = warning_count
        details['info_alerts'] = info_count
        details['total_active_alerts'] = critical_count + warning_count + info_count
        
        # Apply critical alert penalty (placeholder formula: critical_alerts * 10)
        critical_penalty = min(
            critical_count * self.critical_penalty,
            self.max_critical_penalty
        )
        score -= critical_penalty
        details['critical_penalty'] = round(critical_penalty, 2)
        
        # Apply warning alert penalty (placeholder formula: warning_alerts * 5)
        warning_penalty = min(
            warning_count * self.warning_penalty,
            self.max_warning_penalty
        )
        score -= warning_penalty
        details['warning_penalty'] = round(warning_penalty, 2)
        
        # Apply info alert penalty (minor)
        info_penalty = info_count * self.info_penalty
        score -= info_penalty
        details['info_penalty'] = round(info_penalty, 2)
        
        # Calculate resolution bonus if enabled
        resolution_count = 0
        if self.resolution_bonus_enabled and resolved_alerts:
            resolution_count = len(resolved_alerts)
            details['resolved_alerts'] = resolution_count
            
            # Bonus for resolving alerts (shows improvement)
            resolution_bonus_total = min(
                resolution_count * self.resolution_bonus,
                5.0  # Cap bonus at 5 points
            )
            score += resolution_bonus_total
            details['resolution_bonus'] = round(resolution_bonus_total, 2)
        
        # Analyze alert trend if history available
        if alert_history:
            trend = self._analyze_alert_trend(alert_history)
            details['alert_trend'] = trend
            
            if trend == 'improving':
                # Bonus for improving trend
                trend_bonus = 2.0
                score += trend_bonus
                details['trend_bonus'] = trend_bonus
            elif trend == 'deteriorating':
                # Additional penalty for worsening trend
                trend_penalty = 3.0
                score -= trend_penalty
                details['trend_penalty'] = trend_penalty
        
        # Adjust confidence based on data availability
        if not active_alerts and not resolved_alerts and not alert_history:
            # No alert data at all - lower confidence
            confidence = 0.7
            warnings.append("No alert data provided")
        
        # Clamp score to valid range [0, 25]
        score = max(0.0, min(25.0, score))
        normalized_score = self.normalize_score(score)
        
        details['raw_score'] = round(score, 2)
        details['formula'] = 'placeholder: 25 - (critical_alerts * 10) - (warning_alerts * 5)'
        
        return ComponentScore(
            score=score,
            normalized_score=normalized_score,
            confidence=confidence,
            details=details,
            warnings=warnings
        )
    
    def _analyze_alert_trend(self, alert_history: List[Dict]) -> str:
        """
        Analyze alert trend over time.
        
        Args:
            alert_history: List of alert dicts with 'timestamp' key
        
        Returns:
            Trend classification: 'improving', 'stable', or 'deteriorating'
        """
        if not alert_history or len(alert_history) < 2:
            return 'stable'
        
        # Sort by timestamp
        sorted_alerts = sorted(
            alert_history,
            key=lambda x: x.get('timestamp', datetime.min)
        )
        
        # Compare first half to second half
        midpoint = len(sorted_alerts) // 2
        first_half_count = midpoint
        second_half_count = len(sorted_alerts) - midpoint
        
        # Calculate alert rate per unit time
        if first_half_count > second_half_count * 1.2:
            return 'improving'  # Fewer recent alerts
        elif second_half_count > first_half_count * 1.2:
            return 'deteriorating'  # More recent alerts
        else:
            return 'stable'
