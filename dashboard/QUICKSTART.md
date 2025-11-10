# Quick Start Guide - Behavioral Timeline Dashboard

Get the behavioral timeline visualization running in 5 minutes!

## 1. Install Dependencies

From the project root directory:

```bash
pip install streamlit plotly pandas numpy
```

## 2. Run with Mock Data

No database setup required! The dashboard will use simulated data:

```bash
# Set environment variable for mock data
export USE_MOCK_DATA=true

# Run the behavioral timeline page
streamlit run dashboard/pages/behavior_timeline.py
```

**On Windows:**
```cmd
set USE_MOCK_DATA=true
streamlit run dashboard/pages/behavior_timeline.py
```

## 3. Explore the Dashboard

The browser will automatically open to `http://localhost:8501`

- **Select a Cow**: Choose from dropdown (mock IDs: 1001-1005)
- **Choose Time Range**: Last 24 Hours, Last 7 Days, or Last 30 Days
- **Interact**: Zoom, pan, and hover over the timeline
- **View Statistics**: See duration summaries and transition counts
- **Export Data**: Download timeline as CSV

## What You'll See

### Timeline Chart
- Horizontal segments showing behavioral states over time
- Color-coded: Blue (lying), Green (standing), Orange (walking), Purple (ruminating), Yellow (feeding)
- Interactive zoom and pan controls
- Hover tooltips with detailed information

### Statistics Panel
- Total time and percentage in each state
- Number of state transitions
- Longest continuous periods for each behavior

### Mock Data Features
- Realistic circadian patterns (night resting, day activity)
- Natural state transitions
- Confidence scores and motion intensity values

## Next Steps

### Run the Full Dashboard

```bash
streamlit run dashboard/app.py
```

This gives you the home page with navigation to all dashboard sections.

### Connect to Real Database

1. Install database drivers:
   ```bash
   pip install psycopg2-binary sqlalchemy
   ```

2. Create `.env` file in project root:
   ```bash
   DATABASE_URL=postgresql://username:password@localhost:5432/artemis_health
   USE_MOCK_DATA=false
   ```

3. Run the dashboard:
   ```bash
   streamlit run dashboard/pages/behavior_timeline.py
   ```

## Troubleshooting

### "Module not found" Error

**Solution**: Make sure you're in the project root directory and virtual environment is activated:

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dashboard Doesn't Show Data

**Solution**: Ensure `USE_MOCK_DATA=true` is set:

```bash
# Check if environment variable is set
echo $USE_MOCK_DATA  # Should print 'true'

# If not set, export it
export USE_MOCK_DATA=true
```

### Port Already in Use

**Solution**: Streamlit default port (8501) is occupied:

```bash
streamlit run dashboard/pages/behavior_timeline.py --server.port 8502
```

## Features Checklist

Test these features with mock data:

- [ ] Select different cow IDs (1001-1005)
- [ ] Switch between time ranges (24h, 7d, 30d)
- [ ] Zoom into specific hours by scrolling
- [ ] Pan left/right by dragging
- [ ] Hover over segments to see details
- [ ] View statistics panel below chart
- [ ] Export timeline data as CSV
- [ ] Refresh data with the refresh button
- [ ] Download chart as PNG (camera icon)

## Example Data Pattern

Mock data simulates realistic daily cattle behavior:

**Night (0-6am)**: Mostly lying (80%) with occasional ruminating
**Morning (6-9am)**: Standing and feeding activities
**Midday (9-3pm)**: Mix of ruminating, standing, and lying
**Afternoon (3-6pm)**: Active feeding and walking
**Evening (6-12am)**: Transition to lying and ruminating

## Configuration

Customize the dashboard by editing `dashboard/config.py`:

- Behavior colors
- Time range options
- Chart dimensions
- Performance thresholds

## Additional Resources

- **Full Documentation**: See `dashboard/README.md`
- **Configuration Details**: See `dashboard/config.py`
- **Statistics Functions**: See `dashboard/utils/behavior_stats.py`
- **Database Setup**: See main project `README.md`

## Success Indicators

You've successfully set up the dashboard when you can:

✅ See the timeline chart with colored behavioral states
✅ Interact with zoom and pan controls
✅ View hover tooltips with timestamps and durations
✅ See statistics panel with state durations
✅ Export timeline data as CSV
✅ Switch between different time ranges

---

**Ready to explore!** 🎉

For detailed documentation, see `dashboard/README.md`
