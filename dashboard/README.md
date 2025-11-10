# Artemis Health Dashboard

A comprehensive web-based dashboard for livestock health monitoring built with Streamlit.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt` in project root)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure YAML support is available:
```bash
pip install pyyaml
```

### Running the Dashboard

From the project root directory:

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your web browser at `http://localhost:8501`.

## 📊 Dashboard Pages

The application consists of 5 main pages accessible via sidebar navigation:

### 1. Overview (Home Page)
- **Purpose**: Real-time system status and key metrics
- **Features**:
  - Current temperature, activity level, and behavioral state
  - Active alert count
  - Recent sensor data preview
  - System health indicators

### 2. Behavioral Analysis
- **Purpose**: Analyze cattle behavioral states and activity patterns
- **Features**:
  - Behavioral state distribution (lying, standing, walking, ruminating, feeding)
  - Activity patterns (accelerometer and gyroscope data)
  - State transition timeline
  - Movement intensity metrics

### 3. Temperature Monitoring
- **Purpose**: Track body temperature trends and circadian rhythms
- **Features**:
  - Current temperature status with thresholds
  - Temperature distribution analysis
  - Hourly temperature patterns
  - Circadian rhythm analysis (24-hour cycle)
  - Temperature-related alerts

### 4. Alerts Dashboard
- **Purpose**: Monitor and manage system alerts
- **Features**:
  - Active alerts summary
  - Alert breakdown by severity and type
  - Alert history with filtering
  - Alert statistics and trends
  - Export functionality for alert history

### 5. Health Trends
- **Purpose**: Long-term health analysis and trend tracking
- **Features**:
  - Overall health score calculation
  - Multi-component health assessment
  - Multi-day trend analysis
  - Health events timeline
  - Automated health recommendations

## ⚙️ Configuration

Dashboard behavior is controlled by `dashboard/config.yaml`:

### Key Configuration Options

- **Auto-refresh**: Enable/disable automatic data refresh (default: 60 seconds)
- **Data sources**: Configure paths to sensor data and alert logs
- **Display settings**: Customize chart appearance and row limits
- **Thresholds**: Set temperature and activity thresholds
- **Color scheme**: Customize colors for states and alerts

### Example Configuration

```yaml
dashboard:
  title: "Artemis Health - Livestock Monitoring"
  auto_refresh_interval_seconds: 60
  layout: "wide"

data_sources:
  simulated_data_dir: "data/simulated"
  alert_log_file: "logs/malfunction_alerts.json"

metrics:
  temperature:
    normal_min: 38.0
    normal_max: 39.5
    fever_threshold: 39.5
```

## 📁 Project Structure

```
dashboard/
├── app.py                      # Main application entry point
├── config.yaml                 # Dashboard configuration
├── README.md                   # This file
├── pages/                      # Multi-page application pages
│   ├── 1_Overview.py          # Overview dashboard
│   ├── 2_Behavioral_Analysis.py
│   ├── 3_Temperature_Monitoring.py
│   ├── 4_Alerts_Dashboard.py
│   └── 5_Health_Trends.py
├── utils/                      # Utility modules
│   ├── __init__.py
│   └── data_loader.py         # Data loading utilities
└── styles/                     # Custom styling
    └── custom.css             # CSS stylesheet
```

## 🔄 Data Loading

The dashboard loads data from:

1. **Sensor Data**: CSV files in `data/simulated/` or `data/processed/`
   - Expected columns: `timestamp`, `temperature`, `fxa`, `mya`, `rza`, `sxg`, `lyg`, `dzg`
   - Optional columns: `behavioral_state`, `cow_id`, `sensor_id`

2. **Alert Logs**: JSON files in `logs/malfunction_alerts.json`
   - Format: One JSON object per line
   - Required fields: `detection_time`, `severity`, `malfunction_type`

### Data Requirements

- **Timestamps**: ISO 8601 format or Unix timestamps
- **Temperature**: In Celsius (°C)
- **Accelerometer**: In g-forces (g)
- **Gyroscope**: In degrees per second (°/s)

## 🎨 Customization

### Custom Styling

Modify `dashboard/styles/custom.css` to customize:
- Color scheme
- Font sizes and families
- Card layouts
- Responsive breakpoints

The CSS is automatically loaded by the application.

### Session State

The dashboard uses Streamlit session state to persist:
- Selected time ranges
- Auto-refresh settings
- User preferences
- Cached data

Session state is preserved across page navigation.

## 🔧 Features

### Auto-Refresh
- Configurable refresh interval (default: 60 seconds)
- Manual refresh button
- Visual countdown to next refresh
- Can be toggled on/off in sidebar

### Responsive Layout
- Wide layout optimized for desktop
- Responsive columns adapt to screen size
- Mobile-friendly navigation
- Consistent spacing and alignment

### Error Handling
- Graceful error messages for missing data
- Loading states during data fetch
- Informative warnings for configuration issues
- Fallback to default values when needed

### Data Caching
- Efficient data loading with TTL cache
- Configurable cache duration (default: 5 minutes)
- Automatic cache invalidation

## 📈 Development Status

This is the initial infrastructure release. Current implementation includes:

✅ **Completed:**
- Multi-page Streamlit application structure
- Data loading utilities for CSV and JSON
- Session state management
- Auto-refresh mechanism
- Responsive layout
- Custom styling
- All 5 dashboard pages with placeholder content
- Configuration system

🚧 **Planned (Subsequent Tasks):**
- Interactive charts (Plotly/Matplotlib)
- Real-time data streaming
- Alert acknowledgment system
- Export functionality (PDF reports)
- Advanced filtering and search
- User authentication
- Database integration
- Historical data comparison

## 🐛 Troubleshooting

### Dashboard won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're running from the project root directory
- Verify Python version is 3.8 or higher

### No data displayed
- Check that data files exist in configured directories
- Verify file formats match expected schema
- Review `data/simulated/` or `data/processed/` for CSV files
- Check `logs/malfunction_alerts.json` for alert data

### Auto-refresh not working
- Check `config.yaml` for `auto_refresh_enabled: true`
- Ensure `auto_refresh_interval_seconds` is set
- Try toggling auto-refresh off and on in sidebar

### Custom CSS not loading
- Verify `dashboard/styles/custom.css` exists
- Check browser console for CSS errors
- Clear Streamlit cache: `streamlit cache clear`

## 📝 Notes

- This dashboard is designed to work with the Artemis Health data processing pipeline
- Data is loaded from local files; database integration can be added later
- All pages include placeholder visualizations that will be populated in subsequent development
- The dashboard supports both simulated and real sensor data

## 🔗 Related Documentation

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Data Ingestion Module](../src/data_processing/README.md)
- [Alert Generator](../src/alerts/alert_generator.py)
- [Simulation Engine](../src/simulation/)

## 📄 License

Part of the Artemis Health livestock monitoring system.
