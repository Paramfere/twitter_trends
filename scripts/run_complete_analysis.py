#!/usr/bin/env python3

"""
Complete analysis pipeline that runs trending topics fetch, rule-based analysis,
and marketing analysis with organized folder structure.
"""

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_fetch_analysis():
    """Run the fetch topics script and return the session directory."""
    print("🚀 Starting complete trending topics analysis pipeline...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the fetch script
        print("📊 Step 1: Fetching and analyzing trending topics...")
        result = subprocess.run([
            sys.executable, "scripts/fetch_topics.py"
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"❌ Error in fetch script: {result.stderr}")
            return None
        
        print(result.stdout)
        
        # Find the most recent session directory (new or old format)
        data_dir = Path("data")
        session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and (d.name.startswith("session_") or d.name.startswith("fetch_session_"))]
        
        if not session_dirs:
            print("❌ No session directory found")
            return None
        
        # Get the most recent session
        latest_session = max(session_dirs, key=lambda x: x.stat().st_mtime)
        
        print(f"✅ Step 1 completed! Session folder: {latest_session}")
        print(f"⏱️  Processing time: {time.time() - start_time:.1f}s")
        
        return latest_session
        
    except Exception as e:
        print(f"❌ Error in fetch analysis: {e}")
        return None

def run_marketing_analysis(session_dir: Path):
    """Run marketing analysis for the session."""
    print("\n📈 Step 2: Generating marketing analysis...")
    
    try:
        # Find the analysis CSV in the session
        analysis_dir = session_dir / "analysis"
        csv_files = list(analysis_dir.glob("trending_analysis_*.csv"))
        
        if not csv_files:
            print("❌ No analysis CSV found in session")
            return False
        
        # Use the most recent CSV
        csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Import and run marketing analysis
        sys.path.insert(0, str(Path.cwd()))
        from scripts.marketing_analysis import MarketingInsightAnalyzer
        from scripts.playbook_analyzer import PlaybookAnalyzer
        
        # Run marketing analysis
        marketing_analyzer = MarketingInsightAnalyzer()
        marketing_output = marketing_analyzer.analyze_csv_with_session(str(csv_file))
        
        # Run playbook analysis
        playbook_analyzer = PlaybookAnalyzer()
        playbook_output = playbook_analyzer.generate_playbook_report(
            pd.read_csv(str(csv_file)), str(session_dir)
        )
        
        print(f"✅ Step 2 completed!")
        print(f"   📈 Marketing analysis: {marketing_output}")
        print(f"   🎯 Posting playbook: {playbook_output}")
        return True
        
    except Exception as e:
        print(f"❌ Error in marketing analysis: {e}")
        return False

def display_session_summary(session_dir: Path):
    """Display a summary of all files created in the session."""
    print(f"\n📁 SESSION SUMMARY: {session_dir.name}")
    print("=" * 60)
    
    # Count files in each subdirectory
    subdirs = ["raw_data", "analysis", "reports"]
    
    for subdir_name in subdirs:
        subdir = session_dir / subdir_name
        if subdir.exists():
            files = list(subdir.glob("*"))
            print(f"📂 {subdir_name}/")
            for file in files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   📄 {file.name} ({size_mb:.2f} MB)")
    
    print("\n🎯 QUICK ACCESS PATHS:")
    print(f"   Raw Data: {session_dir / 'raw_data'}")
    print(f"   Analysis: {session_dir / 'analysis'}")
    print(f"   Reports:  {session_dir / 'reports'}")

def run_ai_enhancement(session_dir: Path):
    """Run AI enhancement for the session."""
    print("\n🤖 Step 3: Generating AI-enhanced summary...")
    
    try:
        # Run AI session enhancer
        result = subprocess.run([
            sys.executable, "scripts/ai_session_enhancer.py", str(session_dir)
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print(f"❌ Error in AI enhancement: {result.stderr}")
            return False
        
        print(result.stdout)
        print(f"✅ Step 3 completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in AI enhancement: {e}")
        return False

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete analysis pipeline")
    parser.add_argument("--with-ai-enhancement", action="store_true", 
                       help="Include AI enhancement step (requires OpenAI API key)")
    args = parser.parse_args()
    
    overall_start = time.time()
    
    try:
        # Step 1: Fetch and analyze topics
        session_dir = run_fetch_analysis()
        if not session_dir:
            print("❌ Pipeline failed at Step 1")
            return
        
        # Step 2: Generate marketing analysis
        success = run_marketing_analysis(session_dir)
        if not success:
            print("⚠️  Pipeline completed Step 1 but failed at Step 2")
        
        # Step 3: AI Enhancement (optional)
        if args.with_ai_enhancement:
            ai_success = run_ai_enhancement(session_dir)
            if not ai_success:
                print("⚠️  AI enhancement failed, but other steps completed")
        
        # Display summary
        display_session_summary(session_dir)
        
        # Final summary
        total_time = time.time() - overall_start
        steps_completed = 2 + (1 if args.with_ai_enhancement else 0)
        print(f"\n🎉 PIPELINE COMPLETE! ({steps_completed} steps)")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📁 Session: {session_dir}")
        
        # Show next steps
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Review analysis: {session_dir / 'analysis'}")
        print(f"   • Check marketing insights: {session_dir / 'reports'}")
        if args.with_ai_enhancement:
            print(f"   • Review AI summary: {session_dir / 'reports'}")
        print(f"   • Run again in 30 minutes for trend comparison")
        
        # Show AI enhancement option if not used
        if not args.with_ai_enhancement:
            print(f"\n💡 PRO TIP:")
            print(f"   Add --with-ai-enhancement for comprehensive AI analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.exception("Pipeline error")

if __name__ == "__main__":
    main() 