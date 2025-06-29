import unittest
import subprocess
import os
import shutil
from scripts.session_manager import SessionManager

class TestIntelligenceReportGenerator(unittest.TestCase):

    def setUp(self):
        """Set up for the test."""
        self.new_report_path = None
        self.manager = SessionManager()

    def tearDown(self):
        """Clean up any created report files."""
        if self.new_report_path and os.path.exists(self.new_report_path):
            os.remove(self.new_report_path)

    def test_generator_runs_successfully(self):
        """
        Tests if the intelligence_report_generator.py script runs
        on a known data file and produces a report.
        """
        # Use a stable, known session to ensure test independence
        session_dir = "data/session_001"
        analysis_file = "trending_analysis_2025-06-29_22-03-34.csv"
        test_file = os.path.join(session_dir, "analysis", analysis_file)

        self.assertTrue(os.path.exists(test_file), f"Test data not found at {test_file}")
        
        reports_dir = os.path.join(session_dir, "reports")
        initial_reports = set(os.listdir(reports_dir))

        result = subprocess.run(
            ["python", "scripts/intelligence_report_generator.py", test_file],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        self.assertNotIn("ERROR", result.stdout)

        final_reports = set(os.listdir(reports_dir))
        new_reports = final_reports - initial_reports
        
        self.assertEqual(len(new_reports), 1, "Expected one new report to be created.")
        
        self.new_report_path = os.path.join(reports_dir, new_reports.pop())

if __name__ == '__main__':
    unittest.main() 