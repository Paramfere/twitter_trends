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
        print(f"📝 Note: Fetching 10 tweets per topic, ranking by engagement and quality instead of filtering")
        
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

def run_velocity_analysis(session_dir: Path):
    """Run velocity analysis for the session."""
    print("\n📊 Generating velocity report...")
    
    try:
        # Import velocity report generator
        from scripts.velocity_report_generator import VelocityReportGenerator
        
        # Find the analysis CSV
        analysis_dir = session_dir / "analysis"
        csv_files = list(analysis_dir.glob("trending_analysis_*.csv"))
        if not csv_files:
            print("❌ No analysis CSV found for velocity report")
            return False
            
        csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Generate velocity report
        generator = VelocityReportGenerator()
        report = generator.generate_velocity_report(str(csv_file), session_dir.name)
        report_path = generator.save_velocity_report(report, session_dir)
        
        print(f"✅ Velocity report generated: {report_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error in velocity analysis: {e}")
        return False

def run_web3_analysis(session_dir: Path):
    """Run Web3 playbook analysis for the session."""
    print("\n🌐 Generating Web3 playbook...")
    
    try:
        # Import Web3 playbook generator
        from scripts.web3_playbook_generator import Web3PlaybookGenerator
        
        # Find the analysis CSV
        analysis_dir = session_dir / "analysis"
        csv_files = list(analysis_dir.glob("trending_analysis_*.csv"))
        if not csv_files:
            print("❌ No analysis CSV found for Web3 playbook")
            return False
            
        csv_file = max(csv_files, key=lambda x: x.stat().st_mtime)
        
        # Find baseline CSV if available (previous session)
        baseline_csv = None
        data_dir = Path("data")
        session_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("session_")], 
                            key=lambda x: x.stat().st_mtime)
        if len(session_dirs) > 1:
            prev_session = session_dirs[-2]
            prev_csv_files = list((prev_session / "analysis").glob("trending_analysis_*.csv"))
            if prev_csv_files:
                baseline_csv = str(max(prev_csv_files, key=lambda x: x.stat().st_mtime))
        
        # Generate Web3 playbook
        generator = Web3PlaybookGenerator()
        playbook = generator.generate_playbook(str(csv_file), str(baseline_csv) if baseline_csv else None)
        report_path = generator.save_playbook(playbook, session_dir)
        
        print(f"✅ Web3 playbook generated: {report_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error in Web3 analysis: {e}")
        return False

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete analysis pipeline")
    parser.add_argument("--with-ai-enhancement", action="store_true", 
                       help="Include AI enhancement step (requires OpenAI API key)")
    parser.add_argument("--skip-velocity", action="store_true",
                       help="Skip velocity report generation")
    parser.add_argument("--skip-web3", action="store_true",
                       help="Skip Web3 playbook generation")
    parser.add_argument("--regenerate-reports", action="store_true",
                       help="Force regeneration of velocity and web3 reports (normally skipped to avoid duplication)")
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
        
        # Skip velocity and web3 reports unless explicitly requested (to avoid duplication)
        # They are already generated by fetch_topics.py
        
        # Step 3: Generate velocity report (if requested)
        if args.regenerate_reports and not args.skip_velocity:
            print("Note: Regenerating velocity report (already generated by fetch_topics.py)")
            velocity_success = run_velocity_analysis(session_dir)
            if not velocity_success:
                print("⚠️  Velocity report generation failed")
        
        # Step 4: Generate Web3 playbook (if requested)
        if args.regenerate_reports and not args.skip_web3:
            print("Note: Regenerating Web3 playbook (already generated by fetch_topics.py)")
            web3_success = run_web3_analysis(session_dir)
            if not web3_success:
                print("⚠️  Web3 playbook generation failed")
        
        # Step 5: AI Enhancement (optional)
        if args.with_ai_enhancement:
            ai_success = run_ai_enhancement(session_dir)
            if not ai_success:
                print("⚠️  AI enhancement failed, but other steps completed")
        
        # Display summary
        display_session_summary(session_dir)
        
        # Final summary
        total_time = time.time() - overall_start
        steps = ["Fetch & Analysis", "Marketing"]
        if args.regenerate_reports:
            if not args.skip_velocity:
                steps.append("Velocity")
            if not args.skip_web3:
                steps.append("Web3")
        if args.with_ai_enhancement:
            steps.append("AI")
            
        print(f"\n🎉 PIPELINE COMPLETE! ({len(steps)} steps)")
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📁 Session: {session_dir}")
        
        # Show next steps
        print(f"\n💡 NEXT STEPS:")
        print(f"   • Review analysis: {session_dir / 'analysis'}")
        print(f"   • Check reports: {session_dir / 'reports'}")
        if not args.skip_velocity:
            print(f"   • Review velocity trends: {session_dir / 'reports/velocity_report.md'}")
        if not args.skip_web3:
            print(f"   • Check Web3 playbook: {session_dir / 'reports/web3_playbook.md'}")
        if args.with_ai_enhancement:
            print(f"   • Review AI summary: {session_dir / 'reports/ai_summary.md'}")
        print(f"   • Run again in 30 minutes for trend comparison")
        
        # Show AI enhancement option if not used
        if not args.with_ai_enhancement:
            print(f"\n💡 PRO TIP:")
            print(f"   Add --with-ai-enhancement for comprehensive AI analysis")
        
    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        raise

if __name__ == "__main__":
    main() 