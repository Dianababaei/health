# Health Score Gauge & Trends Dashboard Implementation

## Overview

This implementation adds a comprehensive health score monitoring dashboard to the Artemis Health system, featuring:

- **Health Score Gauge** (0-100 scale) with color-coded zones
- **Historical Trend Analysis** with interactive line charts
- **Contributing Factors Breakdown** showing component percentages
- **Baseline Comparison** with delta indicators
- **Time Range Selection** (7/14/30/90 days)
- **Loading States & Error Handling** for robust operation

## Files Created/Modified

### 1. `src/data_processing/health_score_loader.py` (NEW)
Backend data loader for health scores from the database.

**Key Functions:**
- `query_health_scores()` - Query health scores for a cow within a time range
- `calculate_baseline_health_score()` - Calculate rolling average baseline (default: 30 days)
- `get_contributing_factors()` - Retrieve factor breakdown for a specific score
- `get_latest_health_score()` - Get most recent health score for a cow
- `_generate_mock_health_scores()` - Generate realistic mock data for testing

**Features:**
- Database connection support (PostgreSQL/TimescaleDB)
- Mock data generation when database unavailable
- Proper error handling and logging
- Component scores (temperature, activity, behavior, rumination, alerts)
- Trend direction calculation (improving/stable/deteriorating)

### 2. `dashboard/utils/health_visualizations.py` (NEW)
Visualization utilities using Plotly for health score displays.

**Key Functions:**
- `create_health_gauge()` - Gauge chart with 0-100 scale and color zones
  - Red (0-40): Poor/Critical
  - Yellow (40-70): Fair/Monitor
  - Green (70-100): Good/Excellent
  - Shows delta from baseline
  
- `create_health_history_chart()` - Interactive line chart
  - Smooth line interpolation
  - Color-coded background zones
  - Baseline reference line
  - Zoom/pan capabilities
  - Hover tooltips with timestamp details
  
- `display_contributing_factors_streamlit()` - Factor breakdown display
  - Temperature Stability (🌡️)
  - Activity Level (🏃)
  - Behavioral Consistency (🎯)
  - Rumination Quality (🐄)
  - Alert Impact (⚠️)
  - Progress bars with percentages
  
- `create_trend_indicator()` - Visual trend indicator
- `get_health_status_message()` - Status messages based on score

### 3. `dashboard/pages/5_Health_Trends.py` (UPDATED)
Main health trends page with integrated components.

**New Features:**
- Time range selector (7/14/30/90 days)
- Cow ID selector for multi-cow support
- Baseline period selector (7/14/30/60 days)
- Real-time data loading with spinner
- Health score gauge visualization
- Historical score line chart
- Contributing factors breakdown
- Score distribution analysis
- Automated recommendations
- Comprehensive error handling

**Layout:**
```
┌─────────────────────────────────────────────────┐
│ Control Panel (Time Range, Cow ID, Baseline)   │
├─────────────────────────────────────────────────┤
│ Current Health Status                           │
│  ┌──────────────┬──────────────┐               │
│  │ Health Gauge │ Status & Metrics│             │
│  └──────────────┴──────────────┘               │
├─────────────────────────────────────────────────┤
│ Health Score History (Line Chart)               │
├─────────────────────────────────────────────────┤
│ Contributing Factors Breakdown                  │
│  🌡️ Temperature Stability    ████████ 25%      │
│  🏃 Activity Level          ████████ 25%      │
│  🎯 Behavioral Consistency  ████████ 25%      │
│  🐄 Rumination Quality      ███████  20%      │
│  ⚠️ Alert Impact            ██       5%       │
├─────────────────────────────────────────────────┤
│ Trend Analysis & Recommendations                │
└─────────────────────────────────────────────────┘
```

## Technical Implementation

### Database Schema Support
Works with the existing `health_scores` table:
```sql
health_scores (
    timestamp TIMESTAMPTZ,
    cow_id INTEGER,
    health_score INTEGER (0-100),
    temperature_component DOUBLE PRECISION (0.0-1.0),
    activity_component DOUBLE PRECISION (0.0-1.0),
    behavior_component DOUBLE PRECISION (0.0-1.0),
    rumination_component DOUBLE PRECISION (0.0-1.0),
    alert_penalty DOUBLE PRECISION (0.0-1.0),
    trend_direction TEXT ('improving', 'stable', 'deteriorating'),
    trend_rate DOUBLE PRECISION,
    days_since_baseline INTEGER,
    contributing_factors JSONB
)
```

### Health Score Calculation
The health score is a composite metric (0-100) based on:
1. **Temperature Component (25%)**: Stability and normal range adherence
2. **Activity Component (25%)**: Movement patterns and activity levels
3. **Behavior Component (25%)**: Behavioral state consistency
4. **Rumination Component (20%)**: Rumination frequency and quality
5. **Alert Penalty (5%)**: Recent health alerts impact

### Baseline Calculation
- Rolling average over selected period (default: 30 days)
- Used for comparison and trend detection
- Shows delta: "+5.0 from baseline" or "-3.2 from baseline"

### Mock Data Generation
When database is unavailable, generates realistic mock data:
- Random walk with mean reversion
- Time-of-day variations (lower scores during hot hours)
- Correlated component scores
- Trend direction based on recent history
- 1-hour data points for smooth visualization

## Color Coding System

### Health Score Zones
- **🟢 Green (70-100)**: Healthy - Normal monitoring
- **🟡 Yellow (40-70)**: Fair - Increased monitoring
- **🔴 Red (0-40)**: Poor - Immediate attention required

### Visual Indicators
- **📈 Improving**: Green with upward arrow
- **➡️ Stable**: Gray with horizontal arrow
- **📉 Deteriorating**: Red with downward arrow

## Features Implemented

### ✅ Core Requirements
- [x] Health score gauge (0-100) with color zones
- [x] Historical line chart (7/14/30/90 day views)
- [x] Contributing factors breakdown (percentages sum to 100%)
- [x] Baseline comparison with delta indicator
- [x] Time range selector
- [x] Hover tooltips with timestamp details
- [x] Loading states ("Loading health score data...")
- [x] Missing data handling ("No data available" messages)

### ✅ Enhanced Features
- [x] Cow ID selector for multi-cow monitoring
- [x] Baseline period configuration
- [x] Score distribution analysis
- [x] Automated health recommendations
- [x] Trend direction indicators
- [x] Component score display
- [x] Summary statistics (avg/min/max)
- [x] Error handling with detailed messages

## Usage

### Starting the Dashboard
```bash
# From project root
streamlit run dashboard/app.py
```

Navigate to "📈 Health Trends Analysis" in the sidebar.

### Selecting Options
1. **Time Range**: Choose 7, 14, 30, or 90 days
2. **Cow ID**: Enter the cow identifier (default: 1042)
3. **Baseline Period**: Select baseline calculation window
4. **Refresh**: Click to reload data

### Interpreting the Gauge
- **Score Value**: Current health score (0-100)
- **Color**: Visual indicator of health status
- **Delta**: Change from baseline (green ▲ = improving, red ▼ = declining)
- **Status**: Text description (Excellent/Good/Fair/Poor/Critical)

### Reading the History Chart
- **Blue Line**: Health score over time
- **Gray Dashed Line**: Baseline reference
- **Background Zones**: Color-coded health zones
- **Hover**: Shows exact timestamp and score
- **Zoom**: Click and drag to zoom into a period
- **Pan**: Shift+drag to move view

### Understanding Contributing Factors
- Percentages show relative contribution to overall score
- All factors sum to 100%
- Progress bars indicate component strength
- Lower percentages indicate areas of concern

## Testing

### With Database
Ensure TimescaleDB is running with health_scores data:
```bash
# Check database connection
psql -d artemis_health -c "SELECT COUNT(*) FROM health_scores;"
```

### Without Database (Mock Mode)
Set environment variable:
```bash
export USE_MOCK_DATA=true
streamlit run dashboard/app.py
```

Mock data will show realistic health score patterns with time-based variations.

## Performance Considerations

- **Query Optimization**: Indexed queries on cow_id and timestamp
- **Data Aggregation**: Hourly data points for smooth charts
- **Caching**: Streamlit session state for configuration
- **Responsive Design**: Works well with 90-180 days of data
- **Error Handling**: Graceful degradation when data unavailable

## Future Enhancements

The following features can be added in future iterations:
- Multi-cow comparison view
- Predictive health modeling
- Alert threshold configuration
- PDF report export
- Herd-level health statistics
- Anomaly detection highlighting
- Custom time range picker
- Mobile-responsive layout
- Real-time updates (WebSocket)
- Historical baseline overlays

## Dependencies

- `streamlit` - Dashboard framework
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `psycopg2` (optional) - PostgreSQL connection
- `sqlalchemy` (optional) - Database abstraction

## Troubleshooting

### No Data Displayed
- Check cow_id exists in database
- Verify time range has data
- Try extending time range
- Enable mock data mode for testing

### Database Connection Errors
- Verify DATABASE_URL environment variable
- Check PostgreSQL is running
- Confirm credentials are correct
- Use mock data mode as fallback

### Slow Performance
- Reduce time range (use 7 or 14 days)
- Check database indexing
- Monitor query execution time
- Consider data downsampling

## Summary

This implementation provides a comprehensive health monitoring dashboard that:
- Visualizes health scores with intuitive gauge and charts
- Tracks historical trends with baseline comparisons
- Breaks down contributing factors for detailed analysis
- Handles missing data and errors gracefully
- Supports both database and mock data modes
- Scales to 90+ days of historical data

The system meets all technical specifications and success criteria outlined in the task requirements.
