# Troubleshooting Guide

## Issue: Health Scores Not Showing

### Quick Diagnosis

Run this command to check database status:
```bash
python check_db_simple.py
```

Expected output if working correctly:
- `health_scores` table should have at least 1 row after uploading data
- Alerts may or may not exist (depends on data quality)

### Problem: Health Scores Not Being Saved During Upload

**Symptoms:**
- Dashboard shows "No health score data available"
- Health Analysis page shows no data
- Home page shows "No data yet" instead of health score gauge
- `check_db_simple.py` shows 0 health scores but alerts exist

**Root Cause:**
The health score calculation or saving is failing silently during the upload workflow.

**Solution:**

1. **Upload CSV data through dashboard** and watch for messages:
   - Should see: "✅ Health score: XX.X/100 (category) - Saved to database"
   - If you see: "⚠️ Health score calculated... - Failed to save to database"
     - This means the save operation failed
     - Check terminal/console for error messages

2. **Check terminal output** when running Streamlit:
   - Look for `DEBUG: Health score saved to database - cow_id=COW_001, score=XX.XX`
   - Look for `DEBUG ERROR: Failed to save health score: ...`

3. **Verify database file exists and is writable:**
   ```bash
   dir data\alert_state.db
   ```
   - Should show file size > 0
   - Make sure you have write permissions

4. **Test health score save independently:**
   ```bash
   python test_health_score_save.py
   ```
   - This creates sample data and tests the health score calculation and saving
   - Should show "TEST PASSED" at the end
   - If this fails, there's a problem with the HealthScoreManager

### Problem: Health Scores Exist But Not Displaying

**Symptoms:**
- `check_db_simple.py` shows health scores exist
- Dashboard still shows "No data yet" or "No health score data available"

**Solution:**

1. **Check cow ID mismatch:**
   - Health scores are saved with `cow_id` field
   - Dashboard loads scores for the specific cow ID you select
   - Make sure the cow ID in sidebar matches the cow ID in database

2. **Check timestamp/date range:**
   - Health Analysis page filters by date range (default: Last 7 days)
   - If health scores are older than 7 days, they won't show
   - Try selecting "Last 30 days" or "Last 90 days"

3. **Restart Streamlit:**
   - Sometimes Streamlit caches old data
   - Stop the dashboard (Ctrl+C)
   - Start again: `streamlit run dashboard/pages/0_Home.py`

### Problem: Health Score is 100 Even With Alerts in Data

**Symptoms:**
- Generated data includes fever condition
- Dashboard shows "Detected N alerts" during upload
- But health score is still 100/100 (excellent)
- Database shows 0 alerts after upload

**Root Cause:**
Alerts are being detected but either:
1. Not being saved to the database
2. Not being passed to health score calculator
3. Alert detection thresholds not being met

**Solution:**

After the fix, you should see during upload:
1. "✅ Layer 3: Detected N alerts"
2. "💾 Saved N/N alerts to database" (NEW - confirms alerts saved)
3. "📊 Using N alerts for health score calculation:" (NEW - confirms alerts used)
4. Component breakdown showing Alert Impact < 1.0

**Verify alerts are saved:**
```bash
python check_db_simple.py
```
Should show alerts in database with their severities.

**Check component scores:**
The upload summary now shows 4 component metrics:
- Temperature: 0-1 (lower = worse)
- Activity: 0-1 (lower = worse)
- Behavioral: 0-1 (lower = worse)
- Alert Impact: 0-1 (lower = more alerts)

If Alert Impact is 1.0, no alerts were used in calculation.

### Problem: New Upload Doesn't Update Health Scores

**Symptoms:**
- You upload new CSV data
- Health scores don't update to reflect new data
- Old health score still shows

**Why This Happens:**
Each upload creates a NEW health score record with a new timestamp. The system shows the LATEST health score by timestamp.

**Solution:**

1. **Verify upload completed successfully:**
   - Watch for "✅ Processing complete!" message
   - Check for "✅ Health score: XX.X/100 - Saved to database"

2. **Check database to see if new record was added:**
   ```bash
   python check_db_simple.py
   ```
   - Should show increased row count for `health_scores`
   - Latest timestamp should match your upload time

3. **If upload fails silently:**
   - Check for error messages in dashboard
   - Look at terminal output for Python errors
   - Common issues:
     - Missing required columns in CSV
     - Invalid data formats
     - File permissions

### How to Clear Old Data and Start Fresh

If you want to start over with clean data:

```bash
# Option 1: Use interactive tool
python inspect_database.py
# Choose option 8: Clear entire database

# Option 2: Delete database file
# Windows:
del data\alert_state.db
# Linux/Mac:
rm data/alert_state.db

# Next upload will create a fresh database
```

### How to Manually Check Database Contents

**Using Python script:**
```bash
python inspect_database.py
```
- Choose option 4: Count records in all tables
- Choose option 3: View table data (enter "health_scores")

**Using SQL queries:**
```bash
python check_db_simple.py
```

### Expected Workflow

When everything works correctly:

1. **Upload CSV through dashboard:**
   - "✅ Loaded 300 sensor readings"
   - "✅ Layer 1: Behavior classified"
   - "⚙️ Layer 2: Temperature analysis complete"
   - "✅ Layer 3: Detected N alerts"
   - "✅ Health score: XX.X/100 (category) - Saved to database"
   - "✅ Processing complete!"

2. **Home page shows:**
   - Health score gauge (large circular gauge)
   - Recent alerts list
   - Live sensor feed

3. **Health Analysis page shows:**
   - Health score trend over time
   - Component breakdowns
   - Baseline comparisons

4. **Database contains:**
   - At least 1 row in `health_scores` table
   - 0 or more rows in `alerts` table (depends on data)
   - State history for any status changes

### Common Upload Errors

**Missing Columns Error:**
```
❌ Missing required columns: state
```
**Solution:** Your CSV must have: `timestamp, temperature, fxa, mya, rza`
The `state` column is auto-generated during upload.

**Temperature Data Error:**
```
No temperature data available
```
**Solution:** Check that `temperature` column has valid numeric values (not NaN).

**Activity Data Error:**
```
No activity data available
```
**Solution:** Check that `fxa`, `mya`, `rza` columns have valid numeric values.

### Debug Mode

To get more detailed logging, add this to the top of `0_Home.py`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Then restart Streamlit and watch the terminal for detailed DEBUG messages.

### Still Having Issues?

1. Check file: [DATABASE_GUIDE.md](DATABASE_GUIDE.md) for database documentation
2. Run: `python test_health_score_save.py` to verify basic functionality
3. Run: `python check_db_simple.py` to check database status
4. Look for error messages in:
   - Streamlit dashboard (red error boxes)
   - Terminal/console output (Python tracebacks)
   - Browser console (F12, for JavaScript errors)

### Known Limitations

1. **Single Cow Mode:** Currently fixed to "COW_001"
2. **No Real-time Updates:** Dashboard doesn't auto-refresh (refresh browser manually)
3. **Date Range Filtering:** Health Analysis page only shows last 7 days by default
4. **No Data Validation:** Invalid CSV data may cause silent failures

### Quick Test Checklist

- [ ] Database file exists: `data/alert_state.db`
- [ ] Database has tables: `alerts`, `health_scores`, `state_history`
- [ ] Test script passes: `python test_health_score_save.py` shows "TEST PASSED"
- [ ] Upload shows success message: "Saved to database"
- [ ] Check database shows new record: `python check_db_simple.py`
- [ ] Home page shows health score gauge (not "No data yet")
- [ ] Health Analysis page shows data (not "No health score data available")
