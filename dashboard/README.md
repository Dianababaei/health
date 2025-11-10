# Artemis Health Dashboard

Interactive web-based dashboard for visualizing cattle health monitoring data from the Artemis Health system.

## Overview

The dashboard provides real-time visualization and analysis of behavioral states tracked by neck-mounted sensors. Built with Streamlit and Plotly, it offers interactive charts, detailed statistics, and data export capabilities.

## Features

### Behavioral Timeline Visualization

- **Interactive Timeline Chart**: Horizontal timeline showing behavioral states over time
- **Time Range Selector**: Toggle between hourly (24h), daily (7d), and monthly (30d) views
- **Color-Coded States**: Consistent color scheme for 5 behavioral states:
  - 🔵 Lying: Blue (#4A90E2)
  - 🟢 Standing: Green (#7ED321)
  - 🟠 Walking: Orange (#F5A623)
  - 🟣 Ruminating: Purple (#BD10E0)
  - 🟡 Feeding: Yellow (#F8E71C)

### Interactive Features

- **Zoom**: Scroll or pinch to focus on specific time periods
- **Pan**: Click and drag to move along the timeline
- **Hover Tooltips**: Detailed information on state name, timestamps, duration, and confidence
- **Export**: Download timeline data as CSV

### Statistics Panel

- **Duration Summaries**: Total time and percentage spent in each state
- **State Transitions**: Count of state changes throughout the period
- **Longest Periods**: Longest continuous duration for each behavioral state

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

### Required Dependencies

```bash
pip install streamlit>=1.25.0
pip install plotly>=5.11.0
pip install pandas>=1.5.0
pip install numpy>=1.23.0
```

### Optional Database Dependencies

For connecting to TimescaleDB/PostgreSQL:

```bash
pip install psycopg2-binary>=2.9.0
pip install sqlalchemy>=2.0.0
```

**Note**: The dashboard works with mock data if database connection is unavailable.

## Running the Dashboard

### Main Dashboard (Home Page)

```bash
streamlit run dashboard/app.py
```

### Behavioral Timeline Page (Direct)

```bash
streamlit run dashboard/pages/behavior_timeline.py
```

### Using Mock Data

To use simulated data for demonstration:

```bash
export USE_MOCK_DATA=true
streamlit run dashboard/app.py
```

On Windows:
```cmd
set USE_MOCK_DATA=true
streamlit run dashboard/app.py
```

## Configuration

### Environment Variables

The dashboard uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```bash
# Database connection (optional)
DATABASE_URL=postgresql://username:password@localhost:5432/artemis_health

# Dashboard settings
DASHBOARD_REFRESH_INTERVAL=60
MAX_DISPLAY_ANIMALS=50

# Use mock data instead of database
USE_MOCK_DATA=false
```

### Configuration File

The `dashboard/config.py` file contains:

- Behavioral state color scheme
- Time range options
- Chart settings
- Performance thresholds
- Export settings

You can modify these settings to customize the dashboard appearance and behavior.

## Project Structure

```
dashboard/
├── app.py                      # Main dashboard entry point
├── config.py                   # Configuration constants
├── README.md                   # This file
├── pages/                      # Dashboard pages
│   └── behavior_timeline.py   # Behavioral timeline visualization
├── utils/                      # Utility modules
│   ├── __init__.py            # Package initialization
│   ├── db_connection.py       # Database connection utilities
│   └── behavior_stats.py      # Statistics calculation functions
├── components/                 # Reusable UI components (future)
└── assets/                     # Static assets (future)
```

## Usage

### Selecting a Cow

1. Launch the dashboard
2. Navigate to "Behavioral Timeline" in the sidebar
3. Select a cow ID from the dropdown menu

### Choosing Time Range

Use the time range selector to view:

- **Last 24 Hours**: Minute-level data with hourly patterns
- **Last 7 Days**: Minute-level data with daily patterns
- **Last 30 Days**: Hourly aggregated data for better performance

### Interacting with the Timeline

- **Zoom In**: Scroll on the timeline or use the zoom buttons
- **Zoom Out**: Scroll in the opposite direction or use zoom buttons
- **Pan**: Click and drag left/right to move through time
- **Hover**: Hover over segments to see detailed information
- **Export Image**: Click the camera icon to download as PNG

### Exporting Data

Click the "📥 Export Timeline Data (CSV)" button to download behavioral state data including:

- Cow ID
- State name
- Start timestamp
- End timestamp
- Duration (minutes)
- Confidence score

## Data Processing

### Timeline Preparation

The dashboard consolidates consecutive identical states into single segments:

- Groups continuous periods of the same state
- Calculates accurate durations
- Averages confidence scores for each segment
- Handles transitions between states

### Performance Optimization

For large datasets (30+ days):

- Automatically aggregates minute-level data to hourly mode
- Limits data points to maintain responsive rendering
- Caches query results for 5-10 minutes
- Uses Plotly's efficient rendering engine

## Statistics Calculations

### State Durations

Total time spent in each behavioral state, calculated by:

1. Sorting data chronologically
2. Computing time differences between consecutive records
3. Summing durations by state
4. Converting to percentages

### State Transitions

Counts the number of times the animal changed states:

- Compares each consecutive pair of records
- Increments transition counter when states differ
- Tracks specific transition types (e.g., lying → standing)

### Longest Continuous Periods

Finds the longest uninterrupted duration for each state:

1. Groups consecutive records of same state
2. Calculates duration of each period
3. Identifies maximum duration per state
4. Records start and end timestamps

## Mock Data

When database connection is unavailable, the dashboard generates realistic mock data:

- **Circadian Patterns**: Simulates natural daily behavioral rhythms
- **Night (0-6h)**: Mostly lying with some ruminating
- **Morning (6-9h)**: Standing and feeding activities
- **Midday (9-15h)**: Ruminating and resting
- **Afternoon (15-18h)**: Feeding and walking
- **Evening (18-24h)**: Lying and ruminating

Mock data includes realistic confidence scores, motion intensities, and state transitions.

## Troubleshooting

### Database Connection Issues

**Problem**: "Database connection failed" warning

**Solutions**:
1. Check `DATABASE_URL` in `.env` file
2. Verify database is running and accessible
3. Install database drivers: `pip install psycopg2-binary`
4. Use mock data mode: `USE_MOCK_DATA=true`

### No Data for Selected Cow

**Problem**: "No behavioral data found" message

**Solutions**:
1. Check if cow ID exists in database
2. Verify time range has data
3. Check database connectivity
4. Use mock data to verify dashboard functionality

### Slow Performance

**Problem**: Chart takes long to render

**Solutions**:
1. Select shorter time range (24h instead of 30d)
2. Dashboard automatically aggregates large datasets
3. Clear cache with "🔄 Refresh Data" button
4. Reduce `MAX_DISPLAY_ANIMALS` in config

### Import Errors

**Problem**: Module import failures

**Solutions**:
1. Ensure virtual environment is activated
2. Install all dependencies: `pip install -r requirements.txt`
3. Run from project root directory
4. Check Python version (3.9+ required)

## Development

### Adding New Pages

1. Create new page in `dashboard/pages/`
2. Follow naming convention: `page_name.py`
3. Import utilities from `dashboard/utils/`
4. Use consistent styling from `config.py`

### Adding New Statistics

1. Add calculation function to `utils/behavior_stats.py`
2. Update `generate_statistics_summary()` to include new stat
3. Display in timeline page's statistics panel

### Customizing Colors

Modify `BEHAVIOR_COLORS` in `config.py`:

```python
BEHAVIOR_COLORS = {
    'lying': '#YOUR_COLOR',
    'standing': '#YOUR_COLOR',
    # ...
}
```

## Performance Considerations

### Recommended Limits

- **24-hour view**: Up to 1,440 data points (minute-level)
- **7-day view**: Up to 10,080 data points (minute-level)
- **30-day view**: Up to 720 data points (hourly aggregation)

### Caching

The dashboard uses Streamlit's caching:

- Query results: 5-minute TTL
- Cow list: 10-minute TTL
- Clear cache with refresh button

### Database Optimization

For best performance with TimescaleDB:

- Use indexed queries on `cow_id` and `timestamp`
- Enable hypertable compression for old data
- Use continuous aggregates for long-term statistics

## Future Enhancements

Potential features for future development:

- Multi-cow comparison view
- Real-time data streaming
- Alert notifications panel
- Health score integration
- Temperature overlay on timeline
- Prediction confidence visualization
- Custom date range picker
- PDF report generation

## Support

For issues, questions, or contributions:

1. Check this README for common solutions
2. Review the main project documentation
3. Examine example code in `pages/behavior_timeline.py`
4. Verify environment configuration in `.env`

## License

Part of the Artemis Health Livestock Monitoring System.

---

**Note**: This dashboard is designed for research and monitoring purposes. Always consult with veterinary professionals for clinical decisions.
