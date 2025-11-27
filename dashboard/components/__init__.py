"""
Dashboard Components Package

Contains reusable UI components for the Artemis Health Dashboard.
"""

# Note: alerts_panel is deprecated, using notification_panel instead
from .notification_panel import (
    render_notification_panel,
    render_alert_card,
    render_acknowledged_alerts_panel,
    render_alert_summary_metrics,
    render_severity_distribution
)
from .alert_history import (
    render_alert_history,
    render_alerts_table,
    render_alerts_detailed,
    render_alerts_analytics,
    render_search_alerts
)

__all__ = [
    # Notification panel
    'render_notification_panel',
    'render_alert_card',
    'render_acknowledged_alerts_panel',
    'render_alert_summary_metrics',
    'render_severity_distribution',
    # Alert history
    'render_alert_history',
    'render_alerts_table',
    'render_alerts_detailed',
    'render_alerts_analytics',
    'render_search_alerts',
]
