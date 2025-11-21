"""
Automated Test Runner for Artemis Health Monitoring System

Runs all test scenarios from COMPREHENSIVE_TESTING_GUIDE.md
Generates a test report with results.

Usage:
    python run_all_tests.py
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd
import sqlite3

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class TestRunner:
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def print_header(self, text):
        """Print section header."""
        print(f"\n{BLUE}{'='*80}")
        print(f"{text}")
        print(f"{'='*80}{RESET}\n")

    def print_test(self, name, status, details=""):
        """Print test result."""
        if status == "PASS":
            icon = f"{GREEN}[PASS]"
            self.passed += 1
        elif status == "FAIL":
            icon = f"{RED}[FAIL]"
            self.failed += 1
        else:  # WARN
            icon = f"{YELLOW}[WARN]"
            self.warnings += 1

        print(f"{icon} {name}{RESET}")
        if details:
            print(f"   {details}")

        self.results.append({
            'test': name,
            'status': status,
            'details': details
        })

    def check_environment(self):
        """Test: Environment setup."""
        self.print_header("TEST 1: Environment Setup")

        # Check Python version
        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            self.print_test("Python version", "PASS", f"Python {version.major}.{version.minor}.{version.micro}")
        else:
            self.print_test("Python version", "FAIL", f"Python {version.major}.{version.minor} < 3.8")

        # Check Streamlit
        try:
            import streamlit
            self.print_test("Streamlit installed", "PASS", f"v{streamlit.__version__}")
        except ImportError:
            self.print_test("Streamlit installed", "FAIL", "Not installed")

        # Check pandas
        try:
            import pandas
            self.print_test("Pandas installed", "PASS", f"v{pandas.__version__}")
        except ImportError:
            self.print_test("Pandas installed", "FAIL", "Not installed")

        # Check required directories
        dirs = ['data', 'data/dashboard', 'src', 'dashboard']
        for dir_name in dirs:
            if Path(dir_name).exists():
                self.print_test(f"Directory '{dir_name}'", "PASS", "Exists")
            else:
                self.print_test(f"Directory '{dir_name}'", "FAIL", "Missing")

    def check_database(self):
        """Test: Database structure."""
        self.print_header("TEST 2: Database Structure")

        db_path = Path("data/alert_state.db")

        if not db_path.exists():
            self.print_test("Database exists", "WARN", "No database yet (will be created on first upload)")
            return

        self.print_test("Database exists", "PASS", str(db_path))

        # Check tables
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            required_tables = ['alerts', 'health_scores']
            for table in required_tables:
                if table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    self.print_test(f"Table '{table}'", "PASS", f"{count} records")
                else:
                    self.print_test(f"Table '{table}'", "FAIL", "Missing")

            conn.close()
        except Exception as e:
            self.print_test("Database check", "FAIL", str(e))

    def check_test_data_generators(self):
        """Test: Test data generators exist."""
        self.print_header("TEST 3: Test Data Generators")

        generators = [
            ('generate_raw_sensor_data.py', 'Basic health monitoring'),
            ('generate_raw_sensor_data_2.py', 'General health'),
            ('generate_raw_sensor_data_3.py', 'Reproductive focus'),
            ('generate_raw_sensor_data_4.py', 'Rumination focus'),
        ]

        for filename, description in generators:
            path = Path(filename)
            if path.exists():
                self.print_test(f"Generator: {description}", "PASS", filename)
            else:
                self.print_test(f"Generator: {description}", "FAIL", f"{filename} missing")

    def check_sensor_data(self):
        """Test: Sensor data validation."""
        self.print_header("TEST 4: Current Sensor Data")

        sensor_file = Path("data/dashboard/COW_001_sensor_data.csv")

        if not sensor_file.exists():
            self.print_test("Sensor data exists", "WARN", "No data uploaded yet")
            return

        self.print_test("Sensor data exists", "PASS", str(sensor_file))

        try:
            df = pd.read_csv(sensor_file)

            # Check required columns
            required_cols = ['timestamp', 'temperature', 'fxa', 'mya', 'rza', 'state']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                self.print_test("Required columns", "FAIL", f"Missing: {', '.join(missing_cols)}")
            else:
                self.print_test("Required columns", "PASS", "All present")

            # Check data size
            self.print_test("Data size", "PASS", f"{len(df):,} samples")

            # Check state distribution
            if 'state' in df.columns:
                state_counts = df['state'].value_counts()
                self.print_test("State classification", "PASS", f"{len(state_counts)} different states")

                # Check rumination
                ruminating = df['state'].str.contains('ruminat', case=False, na=False).sum()
                rumination_pct = (ruminating / len(df) * 100)

                if rumination_pct > 0:
                    self.print_test("Rumination detected", "PASS", f"{rumination_pct:.1f}% ({ruminating} samples)")
                else:
                    self.print_test("Rumination detected", "WARN", "0% (may need dataset 4)")

        except Exception as e:
            self.print_test("Sensor data validation", "FAIL", str(e))

    def check_alerts(self):
        """Test: Alert detection and management."""
        self.print_header("TEST 5: Alert System")

        db_path = Path("data/alert_state.db")

        if not db_path.exists():
            self.print_test("Alert database", "WARN", "No alerts yet")
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check total alerts
            cursor.execute("SELECT COUNT(*) FROM alerts")
            total = cursor.fetchone()[0]

            if total > 0:
                self.print_test("Total alerts", "PASS", f"{total} alerts in database")
            else:
                self.print_test("Total alerts", "WARN", "No alerts detected")
                conn.close()
                return

            # Check alert types
            cursor.execute("SELECT alert_type, COUNT(*) FROM alerts GROUP BY alert_type")
            alert_types = cursor.fetchall()

            type_str = ", ".join([f"{t[0]}: {t[1]}" for t in alert_types])
            self.print_test("Alert types", "PASS", type_str)

            # Check alert statuses
            cursor.execute("SELECT status, COUNT(*) FROM alerts GROUP BY status")
            statuses = cursor.fetchall()

            status_str = ", ".join([f"{s[0]}: {s[1]}" for s in statuses])
            self.print_test("Alert statuses", "PASS", status_str)

            # Check for reproductive alerts
            cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_type IN ('estrus', 'pregnancy')")
            repro_count = cursor.fetchone()[0]

            if repro_count > 0:
                self.print_test("Reproductive alerts", "PASS", f"{repro_count} detected")
            else:
                self.print_test("Reproductive alerts", "WARN", "None detected (may need dataset 3)")

            conn.close()

        except Exception as e:
            self.print_test("Alert system check", "FAIL", str(e))

    def check_health_scores(self):
        """Test: Health score calculation."""
        self.print_header("TEST 6: Health Score System")

        db_path = Path("data/alert_state.db")

        if not db_path.exists():
            self.print_test("Health score database", "WARN", "No health scores yet")
            return

        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # Check total health scores
            cursor.execute("SELECT COUNT(*) FROM health_scores")
            total = cursor.fetchone()[0]

            if total > 0:
                self.print_test("Health score records", "PASS", f"{total} scores in database")
            else:
                self.print_test("Health score records", "WARN", "No scores yet")
                conn.close()
                return

            # Get latest health score
            cursor.execute("""
                SELECT total_score, temperature_score, activity_score,
                       behavioral_score, alert_score
                FROM health_scores
                ORDER BY created_at DESC
                LIMIT 1
            """)
            latest = cursor.fetchone()

            if latest:
                total_score, temp_score, activity_score, behavioral_score, alert_score = latest

                self.print_test("Latest health score", "PASS", f"{total_score:.1f}/100")

                # Check component scores
                components_ok = True
                if temp_score < 0 or temp_score > 1:
                    components_ok = False
                if activity_score < 0 or activity_score > 1:
                    components_ok = False
                if behavioral_score < 0 or behavioral_score > 1:
                    components_ok = False
                if alert_score < 0 or alert_score > 1:
                    components_ok = False

                if components_ok:
                    self.print_test("Component scores", "PASS", "All in valid range (0-1)")
                else:
                    self.print_test("Component scores", "FAIL", "Some scores out of range")

                # Check if score reflects alerts
                cursor.execute("SELECT COUNT(*) FROM alerts WHERE status = 'active'")
                active_alerts = cursor.fetchone()[0]

                if active_alerts > 10 and total_score > 80:
                    self.print_test("Score accuracy", "WARN", f"Score {total_score:.1f} seems high for {active_alerts} active alerts")
                elif active_alerts == 0 and total_score < 70:
                    self.print_test("Score accuracy", "WARN", f"Score {total_score:.1f} seems low for {active_alerts} alerts")
                else:
                    self.print_test("Score accuracy", "PASS", f"Score {total_score:.1f} matches {active_alerts} alerts")

            conn.close()

        except Exception as e:
            self.print_test("Health score check", "FAIL", str(e))

    def check_dashboard_files(self):
        """Test: Dashboard components exist."""
        self.print_header("TEST 7: Dashboard Components")

        dashboard_files = [
            ('dashboard/pages/0_Home.py', 'Home page'),
            ('dashboard/pages/2_Alerts.py', 'Alerts page'),
            ('dashboard/pages/3_Health_Analysis.py', 'Health Analysis page'),
            ('dashboard/components/notification_panel.py', 'Notification panel'),
            ('dashboard/utils/data_loader.py', 'Data loader'),
            ('dashboard/utils/health_visualizations.py', 'Health visualizations'),
        ]

        for filepath, description in dashboard_files:
            path = Path(filepath)
            if path.exists():
                self.print_test(f"Component: {description}", "PASS", filepath)
            else:
                self.print_test(f"Component: {description}", "FAIL", f"{filepath} missing")

    def check_recent_fixes(self):
        """Test: Verify recent fixes are applied."""
        self.print_header("TEST 8: Recent Fixes Verification")

        # Check notification_panel.py for timestamp fix
        notification_panel = Path("dashboard/components/notification_panel.py")
        if notification_panel.exists():
            content = notification_panel.read_text()
            if "alert.get('timestamp'" in content:
                self.print_test("Timestamp fix (use 'timestamp' not 'created_at')", "PASS", "Applied")
            else:
                self.print_test("Timestamp fix", "WARN", "May not be applied")

        # Check simple_health_scorer.py for rumination fix
        scorer = Path("src/health_intelligence/scoring/simple_health_scorer.py")
        if scorer.exists():
            content = scorer.read_text()
            if "ruminating_lying" in content and "ruminating_standing" in content:
                self.print_test("Rumination detection fix", "PASS", "Applied")
            else:
                self.print_test("Rumination detection fix", "WARN", "May not be applied")

        # Check health_score_loader.py for rumination display fix
        loader = Path("src/data_processing/health_score_loader.py")
        if loader.exists():
            content = loader.read_text()
            if "ruminating_states" in content or "isin(ruminating" in content:
                self.print_test("Rumination display fix", "PASS", "Applied")
            else:
                self.print_test("Rumination display fix", "WARN", "May not be applied")

        # Check 2_Alerts.py for pagination fix
        alerts_page = Path("dashboard/pages/2_Alerts.py")
        if alerts_page.exists():
            content = alerts_page.read_text()
            if "max_alerts=100" in content:
                self.print_test("Alert pagination fix (100 alerts)", "PASS", "Applied")
            elif "max_alerts=10" in content:
                self.print_test("Alert pagination fix", "FAIL", "Still limited to 10")
            else:
                self.print_test("Alert pagination fix", "WARN", "Cannot verify")

    def generate_report(self):
        """Generate test report."""
        self.print_header("TEST SUMMARY")

        total = self.passed + self.failed + self.warnings

        print(f"Total Tests: {total}")
        print(f"{GREEN}Passed: {self.passed}{RESET}")
        print(f"{RED}Failed: {self.failed}{RESET}")
        print(f"{YELLOW}Warnings: {self.warnings}{RESET}")

        pass_rate = (self.passed / total * 100) if total > 0 else 0
        print(f"\nPass Rate: {pass_rate:.1f}%")

        if self.failed == 0:
            if self.warnings == 0:
                print(f"\n{GREEN}SUCCESS: All tests passed!{RESET}")
                recommendation = "READY FOR PRODUCTION"
            else:
                print(f"\n{YELLOW}All tests passed with warnings{RESET}")
                recommendation = "READY (check warnings)"
        else:
            print(f"\n{RED}FAILED: Some tests failed{RESET}")
            recommendation = "NEEDS FIXES"

        print(f"\nRecommendation: {recommendation}")

        # Save report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(f"Artemis Health Monitoring - Test Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Summary:\n")
            f.write(f"  Total: {total}\n")
            f.write(f"  Passed: {self.passed}\n")
            f.write(f"  Failed: {self.failed}\n")
            f.write(f"  Warnings: {self.warnings}\n")
            f.write(f"  Pass Rate: {pass_rate:.1f}%\n\n")

            f.write(f"Recommendation: {recommendation}\n\n")

            f.write(f"Detailed Results:\n")
            f.write(f"{'='*80}\n\n")

            for result in self.results:
                f.write(f"{result['status']}: {result['test']}\n")
                if result['details']:
                    f.write(f"  {result['details']}\n")
                f.write("\n")

        print(f"\n{BLUE}Report saved to: {report_file}{RESET}")

    def run_all_tests(self):
        """Run all test suites."""
        print(f"{BLUE}")
        print("="*80)
        print("   Artemis Health Monitoring System - Automated Test Runner")
        print("="*80)
        print(f"{RESET}")

        self.check_environment()
        self.check_database()
        self.check_test_data_generators()
        self.check_sensor_data()
        self.check_alerts()
        self.check_health_scores()
        self.check_dashboard_files()
        self.check_recent_fixes()

        self.generate_report()


def main():
    """Main entry point."""
    runner = TestRunner()
    runner.run_all_tests()


if __name__ == "__main__":
    main()
