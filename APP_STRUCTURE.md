# Artemis Livestock Health Monitoring - App Structure

## Simplified 3-Page Dashboard

### 1. 🏠 Home
**Purpose:** Overview and immediate status

**Content:**
- Health score (0-100)
- Current temperature
- Active alerts summary
- Recent sensor readings
- Upload simulation data (sidebar)
- Quick navigation buttons

**Use when:** First check of the day, quick status overview

---

### 2. 🚨 Alerts
**Purpose:** Comprehensive alert management

**Content:**
- Active alerts list
- Alert history
- Severity distribution
- Search and filter alerts
- Acknowledge/resolve alerts
- Alert timeline

**Use when:** Investigating health issues, managing alerts

---

### 3. 📊 Health Analysis
**Purpose:** Deep dive into health trends and patterns

**Content:**
- Health score trends over time
- Temperature patterns and circadian rhythm
- Behavioral state distribution
- Activity level trends
- Multi-day health evaluation
- Baseline comparisons
- Reproductive cycle indicators

**Use when:** Long-term monitoring, trend analysis, reproductive tracking

---

## Why Only 3 Pages?

✅ **Less confusing** - Clear, focused purpose for each page
✅ **Faster navigation** - 2 clicks max to any information
✅ **Better UX** - No duplicate information across pages
✅ **Easier maintenance** - Less code to maintain

---

## Navigation Flow

```
Home (Overview)
 ├─→ Alerts (Click "View All Alerts")
 └─→ Health Analysis (Click "Detailed Analysis")
```

**Typical workflows:**

1. **Morning Check:**
   - Open Home → Check health score + alerts
   - If alerts exist → Go to Alerts page
   - If trends concerning → Go to Health Analysis

2. **Alert Investigation:**
   - Home → Click "View All Alerts"
   - Review severity and details
   - Acknowledge/resolve as needed

3. **Health Monitoring:**
   - Home → Click "Detailed Analysis"
   - Review multi-day trends
   - Check reproductive cycles
   - Compare to baseline

---

## Page Details

### Home Page Features:
- 📊 Health Score Card
- 🌡️ Current Temperature
- 🚨 Alert Count
- 🏃 Activity Level
- 📈 Mini trend sparklines
- 📤 Data upload (sidebar)

### Alerts Page Features:
- 🔔 Active alerts panel
- 📜 Alert history
- 🎯 Severity filter
- 🔍 Search function
- ✅ Acknowledge button
- ✓ Resolve button

### Health Analysis Page Features:
- 📊 Health score gauge
- 📈 Multi-day trend charts
- 🌡️ Temperature analysis
- 🏃 Activity patterns
- 🔄 Circadian rhythm
- 🐄 Reproductive tracking
- 📉 Baseline comparison

---

## Removed Pages

| Old Page | Why Removed | New Location |
|----------|------------|--------------|
| Temperature | Redundant | Integrated into Home + Health Analysis |
| Simulation | Separate app | `simulation_app.py` (standalone) |

---

## Benefits of This Structure

**For Users:**
- Clearer mental model (Home → Alerts → Analysis)
- Less clicking around
- No confusion about which page has what
- Faster to find information

**For Developers:**
- Less code duplication
- Easier to maintain
- Clear component boundaries
- Better performance (fewer pages to load)

---

## Future Considerations

If you need more specialized pages later:
- **Reproductive Health** - Dedicated estrus/pregnancy tracking
- **Herd Overview** - Multi-cow comparison
- **Reports** - PDF/CSV export functionality

But for now, 3 pages is optimal for clarity and usability.
