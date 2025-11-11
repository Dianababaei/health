"""
Timeline Visualization Module
==============================
Create interactive Plotly timeline charts for event visualization with 
color-coded markers, hover tooltips, and click handlers.
"""

import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)


class TimelineVizBuilder:
    """
    Builder for creating interactive event timeline visualizations using Plotly.
    
    Features:
    - Scatter plot with time on X-axis and event categories on Y-axis
    - Color-coded markers by event category and severity
    - Custom marker symbols for each event type
    - Interactive hover tooltips with event details
    - Click handlers for expanded event information
    - Responsive layout with zoom/pan capabilities
    """
    
    def __init__(self):
        """Initialize timeline visualization builder."""
        self.fig = None
    
    def create_timeline_chart(
        self,
        df: pd.DataFrame,
        title: str = "Event Timeline",
        height: int = 600,
        show_legend: bool = True,
    ) -> go.Figure:
        """
        Create interactive timeline chart from aggregated events DataFrame.
        
        Args:
            df: Aggregated events DataFrame with required columns:
                - timestamp: Event timestamp
                - category: Event category
                - category_label: Display label for category
                - color: Marker color
                - marker: Marker symbol
                - y_position: Y-axis position
                - title: Event title
                - description: Event description
                - severity: Event severity
                - sensor_values: Sensor values string
            title: Chart title
            height: Chart height in pixels
            show_legend: Whether to show legend
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return self._create_empty_chart(title, height)
        
        # Create figure
        fig = go.Figure()
        
        # Group by category for separate traces (for legend)
        for category in df['category'].unique():
            category_df = df[df['category'] == category]
            
            if category_df.empty:
                continue
            
            # Get display properties for this category
            category_label = category_df['category_label'].iloc[0]
            color = category_df['color'].iloc[0]
            marker_symbol = category_df['marker'].iloc[0]
            
            # Create hover text
            hover_texts = []
            for _, row in category_df.iterrows():
                hover_text = self._create_hover_text(row)
                hover_texts.append(hover_text)
            
            # Add scatter trace for this category
            fig.add_trace(go.Scatter(
                x=category_df['timestamp'],
                y=category_df['y_position'],
                mode='markers',
                name=category_label,
                marker=dict(
                    size=12,
                    color=color,
                    symbol=marker_symbol,
                    line=dict(width=1, color='white'),
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
                customdata=category_df[['event_id', 'event_type', 'severity', 'cow_id']].values,
            ))
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=20, color='#2C3E50'),
                x=0.5,
                xanchor='center',
            ),
            xaxis=dict(
                title='Time',
                showgrid=True,
                gridcolor='#E5E5E5',
                tickformat='%Y-%m-%d %H:%M',
                rangeslider=dict(visible=True),
                type='date',
            ),
            yaxis=dict(
                title='Event Category',
                showgrid=False,
                tickmode='array',
                tickvals=list(range(5)),
                ticktext=[
                    'Sensor Issues',
                    'Health Events',
                    'Behavioral Changes',
                    'Warnings',
                    'Critical Alerts',
                ],
            ),
            height=height,
            hovermode='closest',
            showlegend=show_legend,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='right',
                x=1.15,
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='#E5E5E5',
                borderwidth=1,
            ),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
            margin=dict(l=80, r=150, t=80, b=80),
        )
        
        # Add range slider for zooming
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ])
            )
        )
        
        self.fig = fig
        logger.info(f"Created timeline chart with {len(df)} events")
        
        return fig
    
    def _create_hover_text(self, row: pd.Series) -> str:
        """
        Create hover text for an event.
        
        Args:
            row: Event data row
            
        Returns:
            HTML-formatted hover text
        """
        timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        hover_text = f"""
        <b>{row['title']}</b><br>
        <b>Time:</b> {timestamp}<br>
        <b>Cow ID:</b> {row['cow_id']}<br>
        <b>Type:</b> {row['event_type'].replace('_', ' ').title()}<br>
        <b>Severity:</b> {row['severity'].upper()}<br>
        """
        
        if row.get('description'):
            hover_text += f"<b>Details:</b> {row['description']}<br>"
        
        if row.get('sensor_values'):
            hover_text += f"<b>Sensor Data:</b> {row['sensor_values']}"
        
        return hover_text
    
    def _create_empty_chart(self, title: str, height: int) -> go.Figure:
        """
        Create an empty chart with informational message.
        
        Args:
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        fig.add_annotation(
            text="No events found for the selected filters and time range",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color='gray'),
        )
        
        fig.update_layout(
            title=title,
            height=height,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='#F8F9FA',
            paper_bgcolor='white',
        )
        
        return fig
    
    def add_event_markers(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        marker_size: int = 12,
    ) -> go.Figure:
        """
        Add or update event markers on existing figure.
        
        Args:
            fig: Existing Plotly figure
            df: Events DataFrame
            marker_size: Size of markers
            
        Returns:
            Updated Plotly Figure
        """
        # This is handled in create_timeline_chart, but kept for API compatibility
        return fig
    
    def create_density_heatmap(
        self,
        df: pd.DataFrame,
        time_bucket: str = '1H',
        title: str = "Event Density Heatmap",
        height: int = 400,
    ) -> go.Figure:
        """
        Create a heatmap showing event density over time by category.
        
        Args:
            df: Aggregated events DataFrame
            time_bucket: Time bucket size (e.g., '1H', '30min', '1D')
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return self._create_empty_chart(title, height)
        
        # Create time buckets
        df = df.copy()
        df['time_bucket'] = pd.to_datetime(df['timestamp']).dt.floor(time_bucket)
        
        # Count events by time bucket and category
        density = df.groupby(['time_bucket', 'category_label']).size().reset_index(name='count')
        
        # Pivot for heatmap
        heatmap_data = density.pivot(
            index='category_label',
            columns='time_bucket',
            values='count'
        ).fillna(0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='YlOrRd',
            hovertemplate='Time: %{x}<br>Category: %{y}<br>Events: %{z}<extra></extra>',
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Event Category',
            height=height,
            plot_bgcolor='white',
            paper_bgcolor='white',
        )
        
        return fig
    
    def create_category_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Event Distribution by Category",
        height: int = 400,
    ) -> go.Figure:
        """
        Create a bar chart showing event distribution by category.
        
        Args:
            df: Aggregated events DataFrame
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return self._create_empty_chart(title, height)
        
        # Count events by category
        category_counts = df.groupby(['category_label', 'color']).size().reset_index(name='count')
        category_counts = category_counts.sort_values('count', ascending=False)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=category_counts['category_label'],
                y=category_counts['count'],
                marker_color=category_counts['color'],
                text=category_counts['count'],
                textposition='auto',
                hovertemplate='%{x}<br>Events: %{y}<extra></extra>',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Event Category',
            yaxis_title='Event Count',
            height=height,
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
        )
        
        return fig
    
    def create_severity_distribution(
        self,
        df: pd.DataFrame,
        title: str = "Event Distribution by Severity",
        height: int = 400,
    ) -> go.Figure:
        """
        Create a pie chart showing event distribution by severity.
        
        Args:
            df: Aggregated events DataFrame
            title: Chart title
            height: Chart height
            
        Returns:
            Plotly Figure object
        """
        if df.empty:
            return self._create_empty_chart(title, height)
        
        # Count events by severity
        severity_counts = df['severity'].value_counts()
        
        # Color mapping for severity
        severity_colors = {
            'critical': '#E74C3C',
            'warning': '#F39C12',
            'info': '#3498DB',
        }
        
        colors = [severity_colors.get(sev, '#95A5A6') for sev in severity_counts.index]
        
        # Create pie chart
        fig = go.Figure(data=[
            go.Pie(
                labels=[s.upper() for s in severity_counts.index],
                values=severity_counts.values,
                marker_colors=colors,
                hovertemplate='%{label}<br>Events: %{value}<br>%{percent}<extra></extra>',
            )
        ])
        
        fig.update_layout(
            title=title,
            height=height,
            paper_bgcolor='white',
        )
        
        return fig


def create_timeline_visualization(
    df: pd.DataFrame,
    title: str = "Event Timeline",
    height: int = 600,
    show_legend: bool = True,
) -> go.Figure:
    """
    Convenience function to create timeline visualization.
    
    Args:
        df: Aggregated events DataFrame
        title: Chart title
        height: Chart height
        show_legend: Whether to show legend
        
    Returns:
        Plotly Figure object
    """
    builder = TimelineVizBuilder()
    return builder.create_timeline_chart(df, title, height, show_legend)
