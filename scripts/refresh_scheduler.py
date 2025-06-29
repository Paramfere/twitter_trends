#!/usr/bin/env python3

"""
30-Minute Refresh Scheduler - Continuously monitor trends with velocity tracking
and cross-session momentum analysis.
"""

import logging
import subprocess
import sys
import time
import signal
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RefreshScheduler:
    """Manages 30-minute refresh cycles with historical tracking."""
    
    def __init__(self, interval_minutes: int = 30):
        """Initialize the refresh scheduler."""
        self.interval_minutes = interval_minutes
        self.interval_seconds = interval_minutes * 60
        self.running = False
        self.cycle_count = 0
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run_complete_analysis(self) -> tuple[bool, str]:
        """Run the complete analysis pipeline and return success status and session dir."""
        try:
            logger.info("🚀 Starting complete analysis pipeline...")
            
            # Run the complete analysis
            result = subprocess.run([
                sys.executable, "scripts/run_complete_analysis.py"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            if result.returncode != 0:
                logger.error(f"Analysis pipeline failed: {result.stderr}")
                return False, ""
            
            # Extract session directory from output
            lines = result.stdout.split('\n')
            session_dir = None
            
            for line in lines:
                if "Session:" in line and "data/fetch_session_" in line:
                    session_dir = line.split("Session: ")[1].strip()
                    break
            
            if not session_dir:
                logger.error("Could not extract session directory from output")
                return False, ""
            
            logger.info(f"✅ Analysis completed: {session_dir}")
            return True, session_dir
            
        except Exception as e:
            logger.error(f"Error running analysis: {e}")
            return False, ""
    
    def run_velocity_analysis(self, session_dir: str) -> bool:
        """Run velocity analysis on the session."""
        try:
            logger.info("🚀 Running velocity analysis...")
            
            # Import and run velocity tracker
            sys.path.insert(0, str(Path.cwd()))
            from scripts.velocity_tracker import VelocityTracker
            
            tracker = VelocityTracker()
            output_path = tracker.generate_velocity_report(Path(session_dir))
            
            logger.info(f"✅ Velocity analysis completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error in velocity analysis: {e}")
            return False
    
    def display_cycle_summary(self, cycle_num: int, session_dir: str):
        """Display summary of the current cycle."""
        try:
            # Read some key metrics from the session
            session_path = Path(session_dir)
            analysis_dir = session_path / "analysis"
            csv_files = list(analysis_dir.glob("trending_analysis_*.csv"))
            
            if csv_files:
                df = pd.read_csv(csv_files[0])
                tech_topics = len(df[df['category'] == 'Technology'])
                high_sig_topics = len(df[df['significance_score'] >= 7])
                total_volume = df['tweet_volume'].apply(self._parse_volume).sum()
                
                print(f"\n📊 CYCLE {cycle_num} SUMMARY")
                print("=" * 50)
                print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"📁 Session: {session_path.name}")
                print(f"📈 Total Topics: {len(df):,}")
                print(f"💻 Tech Topics: {tech_topics:,}")
                print(f"⭐ High Significance: {high_sig_topics:,}")
                print(f"💬 Total Volume: {total_volume:,} tweets")
                print("=" * 50)
            
        except Exception as e:
            logger.warning(f"Could not display cycle summary: {e}")
    
    def _parse_volume(self, volume_str):
        """Parse tweet volume for summary."""
        try:
            if pd.isna(volume_str):
                return 0
            volume_str = str(volume_str).lower().replace(',', '').replace(' tweets', '').replace('tweets', '')
            if 'k' in volume_str:
                return int(float(volume_str.replace('k', '')) * 1000)
            elif 'm' in volume_str:
                return int(float(volume_str.replace('m', '')) * 1000000)
            else:
                import re
                numbers = re.findall(r'\d+', volume_str)
                return int(numbers[0]) if numbers else 0
        except:
            return 0
    
    def cleanup_old_sessions(self, keep_hours: int = 24):
        """Clean up sessions older than specified hours."""
        try:
            data_dir = Path("data")
            session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("fetch_session_")]
            
            cutoff_time = datetime.now() - timedelta(hours=keep_hours)
            cleaned_count = 0
            
            for session_dir in session_dirs:
                # Extract timestamp from session name
                timestamp_str = session_dir.name.replace("fetch_session_", "")
                try:
                    session_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                    if session_time < cutoff_time:
                        import shutil
                        shutil.rmtree(session_dir)
                        cleaned_count += 1
                        logger.info(f"🗑️ Cleaned up old session: {session_dir.name}")
                except ValueError:
                    continue
            
            if cleaned_count > 0:
                logger.info(f"🧹 Cleaned up {cleaned_count} old sessions")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def run_continuous_monitoring(self):
        """Run continuous 30-minute monitoring cycles."""
        self.running = True
        self.cycle_count = 0
        
        print("🚀 STARTING 30-MINUTE REFRESH MONITORING")
        print("=" * 60)
        print(f"⏰ Interval: {self.interval_minutes} minutes")
        print(f"📊 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🛑 Press Ctrl+C to stop gracefully")
        print("=" * 60)
        
        try:
            while self.running:
                self.cycle_count += 1
                cycle_start = time.time()
                
                logger.info(f"🔄 Starting cycle {self.cycle_count}")
                
                # Run complete analysis
                success, session_dir = self.run_complete_analysis()
                
                if success and session_dir:
                    # Run velocity analysis
                    self.run_velocity_analysis(session_dir)
                    
                    # Display summary
                    self.display_cycle_summary(self.cycle_count, session_dir)
                    
                    # Cleanup old sessions every 4 cycles (2 hours)
                    if self.cycle_count % 4 == 0:
                        self.cleanup_old_sessions()
                
                else:
                    logger.error(f"❌ Cycle {self.cycle_count} failed")
                
                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, self.interval_seconds - cycle_duration)
                
                if self.running and sleep_time > 0:
                    next_run = datetime.now() + timedelta(seconds=sleep_time)
                    logger.info(f"😴 Cycle {self.cycle_count} completed in {cycle_duration:.1f}s. Next run at {next_run.strftime('%H:%M:%S')}")
                    
                    # Sleep in chunks to allow for graceful shutdown
                    sleep_chunks = int(sleep_time / 10) + 1
                    chunk_size = sleep_time / sleep_chunks
                    
                    for _ in range(sleep_chunks):
                        if not self.running:
                            break
                        time.sleep(chunk_size)
        
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        
        finally:
            self.running = False
            print(f"\n🏁 MONITORING STOPPED")
            print(f"📊 Total cycles completed: {self.cycle_count}")
            print(f"⏰ Total runtime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("✅ Graceful shutdown complete")

def main():
    """Main function with CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="30-minute refresh scheduler with velocity tracking")
    parser.add_argument("--interval", "-i", type=int, default=30, help="Refresh interval in minutes (default: 30)")
    parser.add_argument("--once", action="store_true", help="Run once instead of continuous monitoring")
    
    args = parser.parse_args()
    
    scheduler = RefreshScheduler(interval_minutes=args.interval)
    
    if args.once:
        # Run single cycle
        print("🚀 Running single analysis cycle...")
        success, session_dir = scheduler.run_complete_analysis()
        
        if success and session_dir:
            scheduler.run_velocity_analysis(session_dir)
            scheduler.display_cycle_summary(1, session_dir)
            print("✅ Single cycle completed")
        else:
            print("❌ Single cycle failed")
    else:
        # Run continuous monitoring
        scheduler.run_continuous_monitoring()

if __name__ == "__main__":
    main() 