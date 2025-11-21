# Production Monitoring Guide

This guide explains how to monitor your system in production without needing to run Python scripts.

## Dashboard Overview

All critical information is now visible directly in the dashboard UI.

### Home Page - Database Status

At the top of the Home page, you'll see 4 key metrics:

```
📊 Health Scores    🔴 Active Alerts    📋 Total Alerts    💯 Current Score
     15                    3                  25              72 (GOOD)
```

**What each metric means:**

- **📊 Health Scores**: Total number of health score records saved to database
  - If this is 0, health scores aren't being saved
  - Should increase by 1 each time you upload CSV data

- **🔴 Active Alerts**: Number of alerts currently requiring attention
  - These are alerts with status = 'active'
  - Need to be acknowledged or resolved

- **📋 Total Alerts**: All alerts in database (active + acknowledged + resolved)
  - Shows historical alert count
  - Helps track health trends over time

- **💯 Current Score**: Latest health score from database
  - Shows score value (0-100) and category (EXCELLENT, GOOD, MODERATE, POOR)
  - If shows "--", no health scores in database yet

### Alerts Page - Alert Status

The Alerts page shows a status bar with:

```
🔴 Active    👁️ Acknowledged    ✅ Resolved    📊 Total
    3              5                17           25
```

**Alert Lifecycle:**

1. **Active** (🔴): Newly detected alerts requiring attention
2. **Acknowledged** (👁️): Alerts you've seen and are monitoring
3. **Resolved** (✅): Alerts that have been addressed or cleared

### Health Analysis Page - Data Range

Shows database statistics:

```
📊 Total Records    📈 Average Score    📅 Data Range
      45                  67.5              14 days
```

**What to monitor:**

- **Total Records**: Should match number of uploads
- **Average Score**: Overall health trend (higher is better)
- **Data Range**: Time span covered by your data

## Upload Process - What to Watch For

When you upload CSV data, watch for these messages in order:

### 1. Data Loading
```
✅ Loaded 20160 sensor readings
```
- Confirms CSV was read successfully
- Shows total number of rows

### 2. Layer 1 - Behavior Classification
```
✅ Layer 1: Behavior classified
```
- Sensor data classified into states (lying, standing, walking, etc.)

### 3. Layer 3 - Alert Detection
```
✅ Layer 3: Detected 2 alerts
```
- Shows how many alerts were found in the data
- If 0 alerts, data is healthy (no issues detected)

### 4. Alert Saving (NEW)
```
💾 Saved 2/2 alerts to database
```
- Confirms alerts were saved to database
- Should match "Detected N alerts" count
- If missing or shows 0/2, alerts aren't being saved

### 5. Alert Details (NEW)
```
📊 Using 2 alerts for health score calculation:
  - fever (critical)
  - inactivity (warning)
```
- Shows which alerts are affecting the health score
- Lists alert types and severity levels

### 6. Health Score Calculation
```
✅ Health score: 24.5/100 (poor) - Saved to database
```
- Shows calculated health score
- **MUST say "Saved to database"**
- If says "Failed to save", check logs

### 7. Component Breakdown (NEW)
```
Temperature  Activity  Behavioral  Alert Impact
   0.08        0.00       0.08         0.40
```
- Shows 4 component scores (0-1 scale)
- Lower values indicate problems
- **Alert Impact**: 1.0 = no alerts, <1.0 = alerts present

### 8. Processing Complete
```
✅ Processing complete!

Refresh to View Data
```
- Click the button to refresh and see updated data

## What Good Upload Looks Like

**Healthy Data (No Alerts):**
```
✅ Loaded 20160 sensor readings
✅ Layer 1: Behavior classified
✅ Layer 3: Detected 0 alerts
✅ Health score: 100.0/100 (excellent) - Saved to database

Temperature  Activity  Behavioral  Alert Impact
   1.00        1.00       1.00         1.00
```

**Unhealthy Data (With Alerts):**
```
✅ Loaded 20160 sensor readings
✅ Layer 1: Behavior classified
✅ Layer 3: Detected 2 alerts
💾 Saved 2/2 alerts to database
📊 Using 2 alerts for health score calculation:
  - fever (critical)
  - inactivity (warning)
✅ Health score: 24.5/100 (poor) - Saved to database

Temperature  Activity  Behavioral  Alert Impact
   0.08        0.00       0.08         0.40
```

## Warning Signs

### ⚠️ Health Scores Not Saving

**What you'll see:**
```
⚠️ Health score calculated: 85.0/100 (excellent) - Failed to save to database
Check logs for database errors
```

**What to do:**
1. Check that `data` folder exists and is writable
2. Check terminal/console for error messages
3. Restart Streamlit application

### ⚠️ Alerts Not Saving

**What you'll see:**
```
✅ Layer 3: Detected 2 alerts
(missing: "💾 Saved 2/2 alerts to database")
```

**What to do:**
1. Check terminal for error messages
2. Verify database file exists: `data/alert_state.db`
3. Check file permissions

### ⚠️ Component Scores All Zero

**What you'll see:**
```
Temperature  Activity  Behavioral  Alert Impact
   0.00        0.00       0.00         0.00
```

**This means:**
- Data quality issues (missing columns, invalid values)
- CSV doesn't match expected format

**What to do:**
1. Check CSV has required columns: `timestamp, temperature, fxa, mya, rza`
2. Check for NaN/null values in data
3. Verify data types (numbers not text)

## Quick Health Check

**Daily Monitoring Checklist:**

1. **Open Home Page**
   - [ ] "📊 Health Scores" count increased after upload
   - [ ] "🔴 Active Alerts" shows current issues
   - [ ] "💯 Current Score" displays latest score

2. **Check Alert Status**
   - [ ] Navigate to Alerts page
   - [ ] Review active alerts (red badges)
   - [ ] Acknowledge or resolve alerts as needed

3. **Review Trends**
   - [ ] Navigate to Health Analysis page
   - [ ] Check health score trend over time
   - [ ] Look for declining patterns

## Database Maintenance

**When to Clear Database:**

The database grows over time. Consider clearing when:
- Testing new features
- Starting a new monitoring period
- Database becomes too large (>100 MB)

**How to Clear (Production Method):**

1. Stop the Streamlit application (Ctrl+C)
2. Delete database file:
   ```bash
   # Windows
   del data\alert_state.db

   # Linux/Mac
   rm data/alert_state.db
   ```
3. Restart Streamlit
4. Upload fresh data

**Automatic Cleanup (Future Feature):**
- Could add automatic deletion of records older than X days
- Would need to modify `HealthScoreManager.delete_old_scores()`

## Troubleshooting Quick Reference

| Problem | Dashboard Indicator | Solution |
|---------|-------------------|----------|
| No health scores | "📊 Health Scores: 0" | Check upload shows "Saved to database" |
| Alerts not appearing | "🔴 Active Alerts: 0" but data has issues | Check upload shows "💾 Saved N/N alerts" |
| Health score always 100 | Even with fever data | Check "Alert Impact" component (should be <1.0) |
| "No data yet" on Home | Empty health score gauge | Upload CSV data through sidebar |
| Health Analysis empty | "No health score data available" | Upload data, check date range filter |

## Component Score Reference

Each component contributes to the total health score:

| Component | Weight | Perfect Score | What Lowers It |
|-----------|--------|---------------|----------------|
| Temperature | 30% | 1.0 | Deviation from baseline, fever (>39.5°C) |
| Activity | 25% | 1.0 | Low movement, inactivity |
| Behavioral | 25% | 1.0 | Excessive lying, no rumination |
| Alert Impact | 20% | 1.0 | Critical/warning alerts present |

**Reading Component Scores:**

```
Temperature  Activity  Behavioral  Alert Impact
   0.80        0.95       0.90         1.00
```

- **0.80**: Slight temperature deviation (good)
- **0.95**: Normal activity (excellent)
- **0.90**: Good behavioral patterns
- **1.00**: No alerts (excellent)

**Total Score Calculation:**
```
Total = (0.80 × 30) + (0.95 × 25) + (0.90 × 25) + (1.00 × 20)
      = 24.0 + 23.75 + 22.5 + 20.0
      = 90.25/100 (EXCELLENT)
```

## Auto-Refresh (Future Enhancement)

Currently, you need to:
1. Upload data
2. Click "🔄 Refresh to View Data"

**Future improvement ideas:**
- Auto-refresh every 5 minutes
- WebSocket for real-time updates
- Push notifications for critical alerts

## Exporting Data

**Export Health Scores (Manual):**
```bash
python inspect_database.py
# Choose option 5: Export table to CSV
# Enter table name: health_scores
```

**Export Alerts (Manual):**
```bash
python inspect_database.py
# Choose option 5: Export table to CSV
# Enter table name: alerts
```

## Performance Tips

**For Large Datasets:**

1. **Limit time range on Health Analysis page**
   - Use "Last 7 days" instead of "Last 90 days"
   - Reduces query time

2. **Acknowledge old alerts**
   - Move resolved alerts to "acknowledged" or "resolved" status
   - Reduces "active alerts" query load

3. **Regular database cleanup**
   - Delete health scores older than 90 days
   - Keep database size manageable

## Contact & Support

If you see persistent errors:

1. Check terminal/console output for detailed error messages
2. Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
3. Check database status with test scripts (development only)

**Production Monitoring:**
- All status visible in dashboard UI
- No need to run Python scripts
- Check metrics at top of each page
