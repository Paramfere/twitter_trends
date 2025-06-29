import unittest
import subprocess
import os
import shutil
from scripts.session_manager import SessionManager

class TestFetchTopics(unittest.TestCase):

    def setUp(self):
        """Record the latest session before the test."""
        self.manager = SessionManager()
        self.initial_session_info = self.manager.get_latest_session()

    def tearDown(self):
        """Clean up the session created during the test."""
        latest_session_info = self.manager.get_latest_session()
        initial_session_name = self.initial_session_info[0] if self.initial_session_info else None

        # If there's a new session that wasn't there before, delete it
        if latest_session_info and latest_session_info[0] != initial_session_name:
            session_dir_to_delete = latest_session_info[1]
            if os.path.exists(session_dir_to_delete):
                shutil.rmtree(session_dir_to_delete)

    def test_fetch_topics_script_runs_successfully(self):
        """
        Tests if the fetch_topics.py script completes without errors
        and creates a new session directory with expected files.
        """
        initial_session_dir = self.initial_session_info[1] if self.initial_session_info else None
        
        # Run the script
        result = subprocess.run(
            ["python", "scripts/fetch_topics.py"],
            capture_output=True,
            text=True
        )
        
        # Check for errors
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        self.assertNotIn("ERROR", result.stdout)
        
        # Check if a new session was created
        latest_session_info = self.manager.get_latest_session()
        self.assertIsNotNone(latest_session_info, "manager.get_latest_session() returned None")
        
        latest_session_dir = latest_session_info[1]

        self.assertNotEqual(str(initial_session_dir), str(latest_session_dir), "No new session directory was created.")
        
        # Check for expected files in the new session
        raw_data_path = os.path.join(latest_session_dir, "raw_data")
        analysis_path = os.path.join(latest_session_dir, "analysis")
        
        self.assertTrue(os.path.isdir(raw_data_path), f"raw_data directory not found in {latest_session_dir}")
        self.assertTrue(os.path.isdir(analysis_path), f"analysis directory not found in {latest_session_dir}")
        
        self.assertGreater(len(os.listdir(raw_data_path)), 0, "raw_data directory is empty.")
        self.assertGreater(len(os.listdir(analysis_path)), 0, "analysis directory is empty.")

if __name__ == '__main__':
    unittest.main() 