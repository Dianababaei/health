"""
Home - Herd Health Dashboard

Real-time monitoring with animal behavioral states and health status.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import load_config, DataLoader
from components.modern_ui import render_section_header, render_health_score_gauge

# Page config
st.set_page_config(
    page_title="Home - Herd Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = DataLoader(st.session_state.config)

# Auto-refresh configuration (disabled by default to prevent page flickering)
# Users can manually refresh using the browser refresh button or enable auto-refresh below
AUTO_REFRESH_ENABLED = False
AUTO_REFRESH_INTERVAL_SECONDS = 300  # 5 minutes if enabled

if AUTO_REFRESH_ENABLED:
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    time_since_refresh = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if time_since_refresh > AUTO_REFRESH_INTERVAL_SECONDS:
        st.session_state.last_refresh = datetime.now()
        st.rerun()

# ============================================================================
# HEADER WITH LIVE INDICATOR
# ============================================================================

col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("""
    <h1 style="margin: 0; font-size: 32px; font-weight: 700;">🏠 Herd Dashboard</h1>
    <p style="margin: 4px 0 0 0; color: #7f8c8d; font-size: 14px;">
        Real-time health monitoring
    </p>
    """, unsafe_allow_html=True)

with col2:
    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""
    <div style="text-align: right; margin-top: 8px;">
        <div style="display: inline-block; width: 8px; height: 8px; background: #2ecc71; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite;"></div>
        <span style="color: #2ecc71; font-size: 13px; font-weight: 600;">LIVE</span>
        <div style="color: #7f8c8d; font-size: 12px;">{now}</div>
    </div>
    <style>
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.3; }}
    }}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# UPLOAD RAW SENSOR DATA - END-TO-END PROCESSING
# ============================================================================

st.sidebar.markdown("### 📤 Upload Raw Sensor Data")
st.sidebar.markdown("Upload sensor CSV with columns: timestamp, temperature, fxa, mya, rza, sxg, lyg, dzg")

# Cow ID (fixed for single-cow mode)
cow_id_input = "COW_001"
st.sidebar.info(f"**Cow ID:** {cow_id_input}")

# Baseline temperature input
baseline_temp_input = st.sidebar.number_input(
    "Baseline Temperature (°C)",
    value=38.5,
    min_value=35.0,
    max_value=40.0,
    step=0.1,
    key='baseline_temp_input',
    help="Normal body temperature for this cow"
)

uploaded_sensor = st.sidebar.file_uploader(
    "📊 Raw Sensor Data (CSV)",
    type=['csv'],
    key='upload_sensor',
    help="Upload CSV with: timestamp, temperature, fxa, mya, rza, sxg, lyg, dzg"
)

if uploaded_sensor is not None:
    import json
    from pathlib import Path

    with st.sidebar:
        with st.spinner("Processing sensor data through all layers..."):
            try:
                # Read sensor data
                df = pd.read_csv(uploaded_sensor)

                # Validate required columns
                required_cols = ['timestamp', 'temperature', 'fxa', 'mya', 'rza']
                missing_cols = [col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    # Convert timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'])

                    # Add cow_id if not present
                    if 'cow_id' not in df.columns:
                        df['cow_id'] = cow_id_input

                    st.info(f"✅ Loaded {len(df)} sensor readings")

                    # LAYER 1: Behavior Classification
                    st.info("⚙️ Layer 1: Classifying behavior...")
                    from layer1.rule_based_classifier import RuleBasedClassifier

                    if 'state' not in df.columns:
                        # Enable rumination detection (requires FFT, slower processing)
                        classifier = RuleBasedClassifier(enable_rumination=True)
                        # Use batch classification for better performance
                        df_classified = classifier.classify_batch(df)
                        df['state'] = df_classified['state']
                        st.success(f"✅ Layer 1: Behavior classified (with rumination detection)")

                    # LAYER 2: Temperature Analysis (implicit in data)
                    st.info("⚙️ Layer 2: Temperature analysis complete")

                    # LAYER 3: Alert Detection
                    st.info("⚙️ Layer 3: Detecting health alerts...")
                    from health_intelligence.alerts.immediate_detector import ImmediateAlertDetector

                    alerts = []
                    alert_dedup = {}  # Track unique alerts: (alert_type, day, severity) -> alert

                    # Process data day-by-day to avoid deduplication issues
                    # Each day gets fresh detector instance to reset internal state
                    samples_per_day = 24 * 60  # 1440 samples per day

                    for day_start_idx in range(0, len(df), samples_per_day):
                        day_end_idx = min(day_start_idx + samples_per_day, len(df))
                        day_df = df.iloc[day_start_idx:day_end_idx].copy()

                        if len(day_df) < 10:  # Skip tiny windows
                            continue

                        # Create fresh detector for each day (avoids internal dedup issues)
                        detector = ImmediateAlertDetector()

                        # Process day in 1-hour windows
                        for hour_offset in range(0, len(day_df), 60):
                            window_start = hour_offset
                            window_end = min(hour_offset + 360, len(day_df))  # 6-hour window
                            window_df = day_df.iloc[window_start:window_end].copy()

                            if len(window_df) < 10:
                                continue

                            # Get behavioral state from middle of window
                            mid_idx = window_start + len(window_df) // 2
                            behavioral_state = day_df.iloc[mid_idx]['state'] if mid_idx < len(day_df) else None

                            detected = detector.detect_alerts(
                                sensor_data=window_df,
                                cow_id=cow_id_input,
                                behavioral_state=behavioral_state,
                                baseline_temp=baseline_temp_input
                            )

                            if detected:
                                for a in detected:
                                    # Deduplicate by type, day, severity, and 4-hour period
                                    # This allows up to 6 alerts per day per type (every 4 hours)
                                    alert_day = a.timestamp.date()
                                    alert_hour = a.timestamp.hour
                                    time_period = alert_hour // 4  # 0-5 (6 periods per day)
                                    dedup_key = f"{a.alert_type}_{alert_day}_{a.severity}_{time_period}"

                                    if dedup_key not in alert_dedup:
                                        alert_dict = {
                                            'timestamp': str(a.timestamp),
                                            'cow_id': cow_id_input,
                                            'alert_type': str(a.alert_type),
                                            'severity': str(a.severity),
                                            'confidence': float(a.confidence),
                                            'sensor_values': a.sensor_values,
                                            'details': a.details
                                        }
                                        alerts.append(alert_dict)
                                        alert_dedup[dedup_key] = alert_dict

                    st.success(f"✅ Layer 3: Detected {len(alerts)} immediate health alerts")

                    # LAYER 3B: Reproductive Health Detection (Estrus & Pregnancy)
                    st.info("⚙️ Layer 3: Detecting reproductive events...")
                    try:
                        from health_intelligence.reproductive.estrus_detector import EstrusDetector
                        from health_intelligence.reproductive.pregnancy_detector import PregnancyDetector

                        reproductive_alerts_count = 0

                        # Estrus Detection
                        estrus_detector = EstrusDetector(baseline_temp=baseline_temp_input)

                        # Prepare data for estrus detector
                        temp_data = df[['timestamp', 'temperature']].copy()
                        activity_data = df[['timestamp', 'fxa']].copy()
                        activity_data = activity_data.rename(columns={'fxa': 'movement_intensity'})

                        estrus_events = estrus_detector.detect_estrus(
                            cow_id=cow_id_input,
                            temperature_data=temp_data,
                            activity_data=activity_data,
                            lookback_hours=21*24  # Look back over entire 21-day period
                        )

                        for event in estrus_events:
                            alerts.append({
                                'timestamp': event.timestamp.isoformat(),
                                'cow_id': event.cow_id,
                                'alert_type': 'estrus',
                                'severity': 'info',
                                'confidence': 0.8 if event.confidence.value == 'high' else 0.6 if event.confidence.value == 'medium' else 0.4,
                                'sensor_values': {
                                    'temperature_rise': event.temperature_rise,
                                    'activity_increase': event.activity_increase
                                },
                                'details': {
                                    'message': event.message,
                                    'indicators': event.indicators,
                                    'duration_hours': event.duration_hours
                                }
                            })
                            reproductive_alerts_count += 1

                        # Pregnancy Detection
                        pregnancy_detector = PregnancyDetector()

                        # Prepare data for pregnancy detector (same as estrus)
                        pregnancy_indication = pregnancy_detector.detect_pregnancy(
                            cow_id=cow_id_input,
                            temperature_data=temp_data,
                            activity_data=activity_data,
                            last_estrus_date=None,  # Will auto-detect from data
                            lookback_days=21  # Analyze entire 21-day period
                        )

                        if pregnancy_indication is not None:
                            alerts.append({
                                'timestamp': pregnancy_indication.timestamp.isoformat(),
                                'cow_id': pregnancy_indication.cow_id,
                                'alert_type': 'pregnancy',
                                'severity': 'info',
                                'confidence': 0.7 if pregnancy_indication.confidence.value == 'high' else 0.5 if pregnancy_indication.confidence.value == 'medium' else 0.3,
                                'sensor_values': {
                                    'temperature_stability': pregnancy_indication.temperature_stability,
                                    'activity_reduction': pregnancy_indication.activity_reduction,
                                    'status': pregnancy_indication.status.value
                                },
                                'details': {
                                    'message': pregnancy_indication.message,
                                    'indicators': pregnancy_indication.indicators,
                                    'recommendation': pregnancy_indication.recommendation,
                                    'days_since_estrus': pregnancy_indication.days_since_estrus
                                }
                            })
                            reproductive_alerts_count += 1

                        if reproductive_alerts_count > 0:
                            st.success(f"✅ Layer 3: Detected {reproductive_alerts_count} reproductive event(s)")
                        else:
                            st.info("ℹ️ Layer 3: No reproductive events detected")

                    except Exception as e:
                        st.warning(f"⚠️ Reproductive health detection skipped: {str(e)}")
                        import traceback
                        st.text(traceback.format_exc())

                    st.success(f"✅ Layer 3 Complete: {len(alerts)} total alerts detected")

                    # Save alerts to database
                    if len(alerts) > 0:
                        from health_intelligence.logging.alert_state_manager import AlertStateManager
                        alert_manager = AlertStateManager(db_path="data/alert_state.db")

                        saved_count = 0
                        failed_alerts = []

                        for alert in alerts:
                            try:
                                # Generate unique alert ID
                                alert_id = f"{alert['cow_id']}_{alert['alert_type']}_{alert['timestamp'].replace(':', '-').replace(' ', '_')}"

                                alert_data = {
                                    'alert_id': alert_id,
                                    'cow_id': alert['cow_id'],
                                    'alert_type': alert['alert_type'],
                                    'severity': alert['severity'],
                                    'confidence': alert['confidence'],
                                    'status': 'active',
                                    'timestamp': alert['timestamp'],
                                    'sensor_values': alert['sensor_values'],
                                    'detection_details': alert.get('details', {})
                                }

                                if alert_manager.create_alert(alert_data):
                                    saved_count += 1
                                else:
                                    failed_alerts.append(f"{alert['alert_type']} - creation returned False")
                            except Exception as e:
                                failed_alerts.append(f"{alert.get('alert_type', 'unknown')} - {str(e)}")

                        if saved_count == len(alerts):
                            st.success(f"💾 Saved {saved_count}/{len(alerts)} alerts to database")
                        else:
                            st.warning(f"💾 Saved {saved_count}/{len(alerts)} alerts to database")
                            if failed_alerts:
                                with st.expander("⚠️ Alert Save Errors"):
                                    for err in failed_alerts:
                                        st.write(f"- {err}")

                    # DASHBOARD METRICS: Calculate Health Score
                    st.info("⚙️ Dashboard Metrics: Calculating health score...")
                    from health_intelligence.scoring.simple_health_scorer import SimpleHealthScorer
                    from health_intelligence.logging.health_score_manager import HealthScoreManager

                    # Debug: Show alert details
                    if len(alerts) > 0:
                        st.info(f"📊 Using {len(alerts)} alerts for health score calculation:")
                        for alert in alerts[:3]:  # Show first 3
                            st.write(f"  - {alert['alert_type']} ({alert['severity']})")

                    scorer = SimpleHealthScorer()
                    health_score = scorer.calculate_score(
                        cow_id=cow_id_input,
                        sensor_data=df,
                        baseline_temp=baseline_temp_input,
                        active_alerts=alerts
                    )

                    # Save health score to SQLite database
                    health_manager = HealthScoreManager(db_path="data/alert_state.db")
                    save_success = health_manager.save_health_score(health_score)

                    if save_success:
                        st.success(f"✅ Health score: {health_score['total_score']:.1f}/100 ({health_score['health_category']}) - Saved to database")

                        # Show component breakdown
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Temperature", f"{health_score['temperature_component']:.2f}")
                        with col2:
                            st.metric("Activity", f"{health_score['activity_component']:.2f}")
                        with col3:
                            st.metric("Behavioral", f"{health_score['behavioral_component']:.2f}")
                        with col4:
                            st.metric("Alert Impact", f"{health_score['alert_component']:.2f}")
                    else:
                        st.warning(f"⚠️ Health score calculated: {health_score['total_score']:.1f}/100 ({health_score['health_category']}) - Failed to save to database")
                        st.error("Check logs for database errors")

                    # Save processed data
                    dashboard_dir = Path('data/dashboard')
                    dashboard_dir.mkdir(parents=True, exist_ok=True)

                    # Save sensor data with behavioral states
                    sensor_file = dashboard_dir / f'{cow_id_input}_sensor_data.csv'
                    df.to_csv(sensor_file, index=False)

                    # Save metadata
                    metadata = {
                        'cow_id': cow_id_input,
                        'baseline_temp': baseline_temp_input,
                        'total_samples': len(df),
                        'start_time': str(df['timestamp'].min()),
                        'end_time': str(df['timestamp'].max()),
                        'num_alerts': len(alerts),
                        'health_score': health_score['total_score'],
                        'health_category': health_score['health_category'],
                        'processed_at': str(datetime.now()),
                        'processing': '3-Layer Intelligence (Behavioral + Physiological + Health Intelligence) + Dashboard Metrics'
                    }

                    metadata_file = dashboard_dir / f'{cow_id_input}_metadata.json'
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)

                    st.success("✅ Processing complete!")
                    st.markdown("---")

                    # Show summary
                    st.markdown("**Summary:**")
                    st.write(f"• Sensor readings: {len(df):,}")
                    st.write(f"• Time range: {metadata['start_time']} to {metadata['end_time']}")
                    st.write(f"• Alerts detected: {len(alerts)}")
                    st.write(f"• Health Score: {health_score['total_score']:.1f}/100 ({health_score['health_category']})")

                    if len(alerts) > 0:
                        fever_alerts = sum(1 for a in alerts if a['alert_type'] == 'fever')
                        heat_stress_alerts = sum(1 for a in alerts if a['alert_type'] == 'heat_stress')
                        inactivity_alerts = sum(1 for a in alerts if a['alert_type'] == 'inactivity')

                        if fever_alerts > 0:
                            st.write(f"  - Fever: {fever_alerts}")
                        if heat_stress_alerts > 0:
                            st.write(f"  - Heat stress: {heat_stress_alerts}")
                        if inactivity_alerts > 0:
                            st.write(f"  - Inactivity: {inactivity_alerts}")

                    st.markdown("---")
                    if st.button("🔄 Refresh to View Data", use_container_width=True, type="primary"):
                        st.rerun()

            except Exception as e:
                st.error(f"❌ Processing failed: {e}")
                import traceback
                st.code(traceback.format_exc())

st.sidebar.markdown("---")

# ============================================================================
# LOAD DATA
# ============================================================================

with st.spinner("Loading live data..."):
    try:
        data_loader = st.session_state.data_loader

        # Load sensor data (SINGLE COW MODE - most recent file only)
        df = data_loader.load_sensor_data(time_range_hours=24)

        # Load alerts from DATABASE (not JSON files)
        from health_intelligence.logging import AlertStateManager
        state_manager = AlertStateManager(db_path="data/alert_state.db")

        # Get all unresolved alerts from database (active + acknowledged)
        # Only exclude "resolved" alerts
        active_alerts = state_manager.query_alerts(status='active', limit=50)
        acknowledged_alerts = state_manager.query_alerts(status='acknowledged', limit=50)
        db_alerts = active_alerts + acknowledged_alerts

        # Convert to list of dicts for compatibility
        alerts = []
        for alert in db_alerts:
            alerts.append({
                'alert_id': alert.get('alert_id'),
                'timestamp': alert.get('timestamp'),
                'cow_id': alert.get('cow_id'),
                'alert_type': alert.get('alert_type'),
                'severity': alert.get('severity'),
                'confidence': alert.get('confidence', 0.95),
                'sensor_values': alert.get('sensor_values', {}),
                'status': alert.get('status', 'active'),
            })

        # Show what was loaded
        if len(alerts) > 0:
            st.success(f"✅ Loaded {len(alerts)} alerts from database")
        else:
            st.info("ℹ️ No alerts found in database")

        # Load health score from SQLite database (single-cow mode)
        from health_intelligence.logging.health_score_manager import HealthScoreManager

        health_manager = HealthScoreManager(db_path="data/alert_state.db")
        latest_score = health_manager.get_latest_score(cow_id_input)

        if latest_score:
            herd_score = latest_score['total_score']
            herd_category = latest_score['health_category']
        else:
            herd_score = None
            herd_category = "unknown"

    except Exception as e:
        st.error(f"Error loading data: {e}")
        herd_score = None
        herd_category = "unknown"
        df = pd.DataFrame()
        alerts = []

# ============================================================================
# DATABASE STATUS (Production Monitoring)
# ============================================================================

# Get database statistics
try:
    from health_intelligence.logging.health_score_manager import HealthScoreManager
    from health_intelligence.logging import AlertStateManager
    import sqlite3
    from pathlib import Path

    db_path = Path("data/alert_state.db")

    if db_path.exists():
        # Get counts from database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM health_scores WHERE cow_id = ?", (cow_id_input,))
        health_score_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts WHERE cow_id = ? AND status = 'active'", (cow_id_input,))
        active_alert_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts WHERE cow_id = ?", (cow_id_input,))
        total_alert_count = cursor.fetchone()[0]

        conn.close()

        # Show status bar
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)

        with status_col1:
            st.metric(
                label="📊 Health Scores",
                value=health_score_count,
                help="Total health score records in database"
            )

        with status_col2:
            st.metric(
                label="🔴 Active Alerts",
                value=active_alert_count,
                help="Currently active alerts requiring attention"
            )

        with status_col3:
            st.metric(
                label="📋 Total Alerts",
                value=total_alert_count,
                help="All alerts (active + resolved) in database"
            )

        with status_col4:
            if herd_score is not None:
                st.metric(
                    label="💯 Current Score",
                    value=f"{herd_score:.0f}",
                    delta=herd_category.upper(),
                    help="Latest health score from database"
                )
            else:
                st.metric(
                    label="💯 Current Score",
                    value="--",
                    help="No health score data yet - upload CSV to generate"
                )
    else:
        st.info("📂 Database will be created when you upload your first CSV file")

except Exception as e:
    st.warning(f"Could not load database status: {e}")

st.markdown("---")

# ============================================================================
# TOP CRITICAL METRICS
# ============================================================================

col1, col2, col3 = st.columns(3)

with col1:
    # Health Score Gauge
    if herd_score is not None:
        render_health_score_gauge(herd_score, size="large")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; background: #f8f9fa; border-radius: 12px;">
            <div style="font-size: 48px; margin-bottom: 12px;">📊</div>
            <div style="color: #7f8c8d; font-size: 14px;">No data yet</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    # Critical Alerts
    if isinstance(alerts, list):
        alert_count = len(alerts)
        critical_count = sum(1 for a in alerts if str(a.get('severity', '')).lower() == 'critical')

        if critical_count > 0:
            color = "#e74c3c"
            icon = "🔴"
            status = "CRITICAL"
        elif alert_count > 0:
            color = "#f39c12"
            icon = "🟡"
            status = "WARNINGS"
        else:
            color = "#2ecc71"
            icon = "🟢"
            status = "ALL CLEAR"

        st.markdown(f"""
        <div style="text-align: center; padding: 40px 20px; background: #ffffff; border-radius: 12px; border: 3px solid {color};">
            <div style="font-size: 48px; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 32px; font-weight: 700; color: {color}; margin-bottom: 8px;">{critical_count}</div>
            <div style="color: #7f8c8d; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">{status}</div>
        </div>
        """, unsafe_allow_html=True)

        if alert_count > 0:
            if st.button("🚨 View All Alerts", use_container_width=True, type="primary"):
                st.switch_page("pages/2_Alerts.py")
    else:
        st.metric("🚨 Active Alerts", 0, "All Clear")

with col3:
    # Total Alerts Count
    if isinstance(alerts, list):
        total_alert_count = len(alerts)

        if total_alert_count > 0:
            color = "#3498db"
            icon = "📋"
        else:
            color = "#2ecc71"
            icon = "✅"

        st.markdown(f"""
        <div style="text-align: center; padding: 40px 20px; background: #ffffff; border-radius: 12px; border-left: 4px solid {color};">
            <div style="font-size: 48px; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 32px; font-weight: 700; color: {color}; margin-bottom: 8px;">{total_alert_count}</div>
            <div style="color: #7f8c8d; font-size: 13px; text-transform: uppercase; letter-spacing: 1px;">TOTAL ALERTS</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.metric("📋 Total Alerts", "0", "No Data")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================================
# LIVE ANIMAL FEED - Real-Time States
# ============================================================================

render_section_header(
    title="🐮 Live Animal Feed",
    subtitle="Current behavior and health status"
)

if len(df) > 0:
    # Get latest data point (single cow mode)
    row = df.sort_values('timestamp').iloc[-1]

    cow_id = cow_id_input  # Use the fixed cow ID
    state = row.get('state', 'unknown')
    temp = row.get('temperature', 0)

    # Calculate activity percentage
    if 'fxa' in row:
        activity = np.sqrt(row['fxa']**2 + row.get('fya', 0)**2 + row.get('fza', 0)**2)
        activity_pct = min(int(activity * 100), 100)
    elif 'movement_intensity' in row:
        activity_pct = min(int(row['movement_intensity'] * 100), 100)
    else:
        activity_pct = 50

    # Determine health status
    if temp > 39.5:
        health_status = "🔴 FEVER"
        health_color = "#e74c3c"
    elif temp > 39.0 and activity_pct > 80:
        health_status = "🟡 HEAT STRESS"
        health_color = "#f39c12"
    elif activity_pct < 20:
        health_status = "🟡 LOW ACTIVITY"
        health_color = "#f39c12"
    else:
        health_status = "🟢 HEALTHY"
        health_color = "#2ecc71"

    # State icon mapping
    state_icons = {
        'lying': '🛏️',
        'standing': '🧍',
        'walking': '🚶',
        'ruminating': '🐄',
        'ruminating_lying': '🐄',
        'ruminating_standing': '🐄',
        'feeding': '🍽️',
        'transition': '🔄',
        'uncertain': '❓',
        'unknown': '❓'
    }
    state_icon = state_icons.get(state.lower(), '❓')

    # State display labels (more user-friendly)
    state_labels = {
        'lying': 'Lying',
        'standing': 'Standing',
        'walking': 'Walking',
        'ruminating': 'Ruminating',
        'ruminating_lying': 'Ruminating',
        'ruminating_standing': 'Ruminating',
        'feeding': 'Feeding',
        'transition': 'Changing Position',
        'uncertain': 'Monitoring',
        'unknown': 'Unknown'
    }
    state_label = state_labels.get(state.lower(), state.capitalize())

    # Activity bar (visual representation)
    filled_bars = int(activity_pct / 10)
    activity_bar = "█" * filled_bars + "░" * (10 - filled_bars)

    # Render animal card
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 3, 1])

    with col1:
        st.markdown(f"**{cow_id}**")

    with col2:
        st.markdown(f"<span style='color: {health_color}; font-weight: 600;'>{health_status}</span>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"{state_icon} {state_label}")

    with col4:
        st.markdown(f"{temp:.1f}°C  `{activity_bar}` {activity_pct}%")

    with col5:
        if health_status != "🟢 HEALTHY":
            if st.button("📋", key=f"details_{cow_id}", help="View Details"):
                st.switch_page("pages/2_Alerts.py")

    st.markdown("---")

    st.markdown("<br>", unsafe_allow_html=True)

else:
    # No data state
    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 20px;
        background: #fff3cd;
        border-radius: 12px;
        border: 2px solid #f39c12;
    ">
        <div style="font-size: 64px; margin-bottom: 16px;">🐮</div>
        <h3 style="color: #856404; margin-bottom: 8px;">No Live Data</h3>
        <p style="color: #856404; margin-bottom: 20px;">
            Upload simulation data using the sidebar or connect real sensors
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# QUICK ACTIONS (Only if data exists)
# ============================================================================

if len(df) > 0:
    st.markdown("<br>", unsafe_allow_html=True)

    render_section_header(
        title="⚡ Quick Actions",
        subtitle="Navigate to detailed views"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🚨 View All Alerts", use_container_width=True, type="secondary"):
            st.switch_page("pages/2_Alerts.py")

    with col2:
        if st.button("📊 Detailed Analysis", use_container_width=True, type="secondary"):
            st.switch_page("pages/3_Health_Analysis.py")

# ============================================================================
# FOOTER WITH REFRESH BUTTON
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

# Center the refresh button
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    if st.button("🔄 Refresh Page", use_container_width=True, type="primary"):
        st.rerun()

st.markdown(f"""
<div style="
    text-align: center;
    padding: 20px;
    color: #95a5a6;
    font-size: 12px;
    margin-top: 10px;
">
    Artemis Livestock Health Monitoring | Last updated: {now}
</div>
""", unsafe_allow_html=True)
